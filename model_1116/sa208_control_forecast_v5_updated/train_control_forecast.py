#!/usr/bin/env python3
"""v4 控制导向预测训练流水线（方案 A 物理先验先行剥离 + 方案 B 近端煤量特征解耦 + 趋势锁死 Loss）。

=============================================================================
训练流水线重构与优化说明（方案 A + 方案 B 落地版）
=============================================================================
1. 【方案 A：物理先验先行剥离 + 净扰动残差学习（Physics-First Residual Learning）】：
   - 废除原先在 Stage 1 设 `closed_loop_coupling=0.0` 并传入 `hold_u` 直接拟合闭环总温变的做法
     （原做法导致黑盒神经网络在 Stage 1 把历史在途余热和可预测闭环调煤全部吞进 `disturb_delta_c`，
     致使 Stage 2 留给物理通道的残差只剩负相关噪声，进而将粉线和棕线压成 0℃）；
   - 新 Stage 1（物理先验锁定残差预训练）：
     冻结 `coal_response`（FOPDT 锁定在工程先验中心 K=15, τ=200s）与 `hist_coal_scale`（γ=0.85），
     同时开启物理受迫通道 `closed_loop_coupling = args.stage1_coupling` (默认 0.80)，直接传入真实未来煤量 `fut_u`。
     此时物理模块先行扣除确定性的在途余热与未来调煤热贡献，强迫 `SmoothTrendBasisDecoder` 纯粹拟合
     剥离尾煤热惯性后的净外部扰动残差 `ΔT_true - (γ*ΔT_hist_coal + c_cl*ΔT_fut_coal)`；
   - 新 Stage 2（受限 LPV 状态调制慢速微调）：
     在同向主动激励样本上微调标定后，以 `0.02 * lr` 超低学习率放开 LPV 状态调制器 `coal_response.scheduler`，
     在强 `prior_loss` 与保底耦合下界（`min_coupling=0.65`、`hist_coal_scale >= 0.60`）约束下微调动态工况参数。

2. 【方案 B：近端煤量特征解耦（防止黑盒利用历史煤量抵消在途余热）】：
   - 编码器侧自动屏蔽近端在途窗口内的瞬时煤量与单步脉冲通道（仅保留 30min 慢变负荷均值与波动包络），
     精确的近端尾煤历史专供物理 FOPDT 状态机使用，确保评估图中的在途余热（棕线）与未来受迫响应（粉线）完整呈现真实物理量纲。

3. 【趋势与相位锁死 Loss 及选模准则】：
   - 联合优化远期加权 Huber、去均值波形皮尔逊相关损失 (`corr_loss`)、四级跨步差分金字塔 (`ms_trend`)、
     3/6/10/15min 净方向同号损失 (`sign_loss`) 与相位门控振幅损失。

安全边界：离线只读 CSV，不访问 IOT、不下发动作；
未来真实煤量仅作为事后 Oracle 条件输入，保持当前煤量仅作历史标签诊断。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from control_forecast import (
    ControlOrientedForecaster,
    ControlTrajectoryLoss,
    ForecastConfig,
    calibrate_closed_loop_coupling,
    calibrate_gain_scale,
    kpi_metrics,
    max_in_flight_steps,
    virtual_step_check,
)

# 训练集与验证集时间切分边界（中间预留 60min 隔离带防止历史窗口重叠）
CUT = pd.Timestamp("2026-10-01 00:00:00")
VAL_END = pd.Timestamp("2026-10-15 00:00:00")
SEED = 20261116
C_TEMP, C_COAL = 3, 5
COVAR_COLS = [
    "c5b_valve",
    "sat_temp",
    "decompose_press",
    "alt_fuel",
    "kiln_current",
    "ladder_pressure",
]
COVAR_DEFAULT_SCALES = np.array([15.0, 40.0, 150.0, 5.0, 100.0, 150.0], dtype=np.float32)


# ===========================================================================
# 1. 预计算加速滑动窗口数据集
# ===========================================================================

class WindowDataset(Dataset):
    """支持全局因果特征预计算的滑动窗口数据集。"""

    def __init__(
        self,
        temp: np.ndarray,
        coal: np.ndarray,
        times: pd.DatetimeIndex,
        origins: np.ndarray,
        config: ForecastConfig,
        covars: np.ndarray | None = None,
        subsample: int | None = None,
    ):
        self.temp = np.asarray(temp, dtype=np.float32)
        self.coal = np.asarray(coal, dtype=np.float32)
        self.times = times
        self.covars = np.asarray(covars, dtype=np.float32) if covars is not None else None
        self.config = config
        self.origins = origins[::subsample] if subsample else origins

        inv_t_scale = 1.0 / config.temperature_scale_c
        self._t_scaled = self.temp * inv_t_scale
        self._dt1 = np.zeros_like(self.temp)
        self._dt1[1:] = np.diff(self._t_scaled)
        self._dt2 = np.zeros_like(self.temp)
        self._dt2[1:] = np.diff(self._dt1)

        inv_u_scale = 1.0 / config.coal_scale_tph
        self._uf = (self.coal - config.coal_mean_tph) * inv_u_scale
        self._pulse = np.zeros_like(self.coal)
        self._pulse[1:] = np.clip(np.diff(self.coal) * inv_u_scale, -5.0, 5.0)
        # Invalid rows are excluded from every sampled window, but cumulative sums
        # must remain finite across those gaps or one NaN poisons all later windows.
        coal_for_prefix = np.nan_to_num(self.coal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
        self._coal_cs = np.r_[0.0, np.cumsum(coal_for_prefix, dtype=np.float64)]
        self._coal_cs2 = np.r_[0.0, np.cumsum(coal_for_prefix ** 2)]
        self._below = self.coal < config.sat_low_tph
        self._win_pos = np.arange(config.history_steps, dtype=np.int32)
        self._win_end = np.arange(1, config.history_steps + 1, dtype=np.int64)
        self._mean_win = min(config.coal_mean_window, config.history_steps)
        self._std_win = min(config.coal_std_window, config.history_steps)
        self._mean_start = np.maximum(0, self._win_end - self._mean_win)
        self._mean_count = (self._win_end - self._mean_start).astype(np.float64)
        self._std_start = np.maximum(0, self._win_end - self._std_win)
        self._std_count = (self._win_end - self._std_start).astype(np.float64)

        if self.covars is not None:
            c = self.covars.shape[1]
            scales = (
                COVAR_DEFAULT_SCALES[:c]
                if c <= len(COVAR_DEFAULT_SCALES)
                else np.ones(c, dtype=np.float32)
            )
            self._cov_scaled = self.covars / scales[None, :]
            self._cov_dx = np.zeros_like(self._cov_scaled)
            self._cov_dx[1:] = self._cov_scaled[1:] - self._cov_scaled[:-1]

    def __len__(self) -> int:
        return len(self.origins)

    def _fast_temp_features(self, s_idx: int, e_idx: int) -> np.ndarray:
        t_win = self._t_scaled[s_idx:e_idx]
        rel = t_win - t_win[-1]
        dt1 = self._dt1[s_idx:e_idx].copy()
        dt1[0] = 0.0
        dt2 = self._dt2[s_idx:e_idx].copy()
        dt2[0] = 0.0
        if len(dt2) > 1:
            dt2[1] = dt1[1]
        return np.stack([rel, dt1, dt2], axis=-1)

    def _fast_coal_features(self, s_idx: int, e_idx: int) -> np.ndarray:
        cfg = self.config
        inv_u_scale = 1.0 / cfg.coal_scale_tph
        uf = self._uf[s_idx:e_idx]
        pulse = self._pulse[s_idx:e_idx].copy()
        pulse[0] = 0.0

        cs_win = self._coal_cs[s_idx : e_idx + 1] - self._coal_cs[s_idx]
        cs2_win = self._coal_cs2[s_idx : e_idx + 1] - self._coal_cs2[s_idx]

        mean_x = (cs_win[self._win_end] - cs_win[self._mean_start]) / self._mean_count
        mean_feat = ((mean_x.astype(np.float32) - cfg.coal_mean_tph) * inv_u_scale).astype(
            np.float32
        )

        std_mean = (cs_win[self._win_end] - cs_win[self._std_start]) / self._std_count
        std_var = np.maximum(
            (cs2_win[self._win_end] - cs2_win[self._std_start]) / self._std_count - std_mean ** 2,
            0.0,
        )
        std_feat = (np.maximum(np.sqrt(std_var).astype(np.float32), 1e-4) * inv_u_scale).astype(
            np.float32
        )

        below_win = self._below[s_idx:e_idx]
        last_reset = np.maximum.accumulate(np.where(~below_win, self._win_pos, -1))
        run = np.where(below_win, self._win_pos - last_reset, 0)
        sat = np.minimum(run, cfg.sat_low_max_steps).astype(np.float32) / cfg.sat_low_max_steps
        return np.stack([uf, pulse, mean_feat, std_feat, sat], axis=-1)

    def _fast_covar_features(self, s_idx: int, e_idx: int) -> np.ndarray:
        c_win = self._cov_scaled[s_idx:e_idx]
        rel = c_win - c_win[-1:]
        dx = self._cov_dx[s_idx:e_idx].copy()
        dx[0] = 0.0
        return np.clip(rel + 0.5 * dx, -5.0, 5.0).astype(np.float32)

    def __getitem__(self, j: int):
        i = int(self.origins[j])
        cfg = self.config
        s_idx = i - cfg.history_steps + 1
        e_idx = i + 1
        fut_slice = slice(i + 1, i + cfg.horizon_steps + 1)

        th = self._fast_temp_features(s_idx, e_idx)
        uh = self._fast_coal_features(s_idx, e_idx)
        if self.covars is not None:
            ch = self._fast_covar_features(s_idx, e_idx)
        else:
            ch = np.zeros((1, 1), dtype=np.float32)

        u_hist = self.coal[s_idx:e_idx]
        inv_u_scale = 1.0 / cfg.coal_scale_tph
        m = float((u_hist[-12:].mean() - cfg.coal_mean_tph) * inv_u_scale)
        s = float((u_hist[-cfg.coal_std_window :] - cfg.coal_mean_tph).std() * inv_u_scale)

        return (
            torch.from_numpy(th),
            torch.from_numpy(uh),
            torch.from_numpy(ch),
            torch.tensor(self.temp[i], dtype=torch.float32),
            torch.tensor(self.coal[i], dtype=torch.float32),
            torch.from_numpy(self.temp[fut_slice]),
            torch.from_numpy(self.coal[fut_slice]),
            torch.tensor(m, dtype=torch.float32),
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(self.times[i].value, dtype=torch.int64),
        )


# ===========================================================================
# 2. 数据加载、因果清洗与严格防穿越切分
# ===========================================================================

def load_data(
    csv_path: str | Path,
    stride: int,
    use_covariates: bool = False,
    in_flight_settling_ratio: float = 0.99,
    decouple_recent_coal: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    ForecastConfig,
    dict,
]:
    """读取 5s 源 CSV 数据，重采样至 15s，执行开窑工况连续窗口过滤与严格时间轴切分。"""
    base_cols = ["time", "decompose_temp", "feed_actual", "sat_temp"]
    cols = list(dict.fromkeys(base_cols + (COVAR_COLS if use_covariates else [])))
    frame = pd.read_csv(csv_path, usecols=cols, low_memory=False)
    source_rows = len(frame)

    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    if frame["time"].duplicated().any() or not frame["time"].is_monotonic_increasing:
        raise ValueError("时间戳重复或乱序")
    dt = frame["time"].diff().dropna()
    if not (dt == pd.Timedelta(seconds=5)).all():
        raise ValueError("源数据不是严格 5s 间隔")

    frame = frame.iloc[::3].reset_index(drop=True)
    times = pd.DatetimeIndex(frame["time"])
    temp = frame["decompose_temp"].to_numpy(np.float32)
    coal = frame["feed_actual"].to_numpy(np.float32)
    sat = frame["sat_temp"].to_numpy(np.float32)

    valid = (
        np.isfinite(temp)
        & np.isfinite(coal)
        & np.isfinite(sat)
        & (temp > 800.0)
        & (sat > 800.0)
    )
    covars = None
    if use_covariates:
        cov_list = []
        for c in COVAR_COLS:
            s_cov = pd.Series(frame[c]).ffill()
            arr = s_cov.to_numpy(np.float32)
            valid = valid & np.isfinite(arr)
            cov_list.append(np.nan_to_num(arr, nan=0.0))
        covars = np.stack(cov_list, axis=-1)

    train_time_mask = np.asarray(times < CUT, dtype=bool)
    stat_mask = valid & train_time_mask
    if not stat_mask.any():
        stat_mask = valid
    cfg = ForecastConfig(
        coal_mean_tph=float(np.nanmean(coal[stat_mask])),
        coal_scale_tph=max(float(np.nanstd(coal[stat_mask])), 1e-3),
        in_flight_settling_ratio=in_flight_settling_ratio,
        decouple_recent_coal=decouple_recent_coal,
    )

    bad_prefix = np.r_[0, np.cumsum(~valid)]
    first = cfg.history_steps - 1
    last = len(frame) - cfg.horizon_steps - 1
    origins = np.arange(first, last + 1, stride, dtype=np.int64)
    keep = (
        bad_prefix[origins + cfg.horizon_steps + 1]
        - bad_prefix[origins - cfg.history_steps + 1]
        == 0
    )
    origins = origins[keep]

    t0 = times.take(origins)
    t1 = times.take(origins + cfg.horizon_steps)
    train = origins[t1 < CUT]
    val = origins[(t0 >= CUT + pd.Timedelta(minutes=60)) & (t1 < VAL_END)]
    if len(train) == 0 or len(val) == 0:
        raise ValueError(f"切分为空 train={len(train)} val={len(val)}")

    return (
        temp,
        coal,
        covars,
        times,
        train,
        val,
        cfg,
        {
            "source_rows": source_rows,
            "rows_15s": len(frame),
            "valid_fraction": float(valid.mean()),
            "stride_samples": stride,
            "train_origins": len(train),
            "validation_origins": len(val),
            "train_last_origin": str(times[train[-1]]),
            "validation_first_origin": str(times[val[0]]),
            "validation_last_target": str(times[val[-1] + cfg.horizon_steps]),
            "use_covariates": use_covariates,
            "covariate_columns": COVAR_COLS if use_covariates else [],
        },
    )


# ===========================================================================
# 3. 评价指标、训练集子集标定与方案 A/B 两阶段训练循环
# ===========================================================================

def base_metrics(
    target: np.ndarray, pred: np.ndarray, cur: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    """计算基础点预测误差与宏观方向一致性指标。"""
    del cur
    err = pred - target
    w = weights[None, :]
    horizon_bias = err.mean(0)
    sig_end = np.abs(target[:, -1]) > 3.0
    end_dir_acc = (
        float((np.sign(pred[sig_end, -1]) == np.sign(target[sig_end, -1])).mean())
        if sig_end.any()
        else 1.0
    )
    return {
        "weighted_15min_rmse_c": float(np.sqrt(np.mean(np.sum(w * err ** 2, 1) / w.sum()))),
        "endpoint_rmse_c": float(np.sqrt((err[:, -1] ** 2).mean())),
        "endpoint_direction_accuracy": end_dir_acc,
        "all_point_p90_abs_error_c": float(np.quantile(np.abs(err), 0.9)),
        "max_abs_horizon_mean_bias_c": float(np.abs(horizon_bias).max()),
        "step0_mean_bias_c": float(horizon_bias[0]),
        "step0_rmse_c": float(np.sqrt((err[:, 0] ** 2).mean())),
        "per_window_mean_mae_c": float(np.abs(err.mean(1)).mean()),
        "mean_abs_error_c": float(np.abs(err).mean()),
    }


@torch.no_grad()
def calibrate_on_train_subset(
    model: ControlOrientedForecaster,
    train_loader: DataLoader,
    device: torch.device,
    max_batches: int = 24,
    apply_gain_scale: bool = True,
    min_coupling: float = 0.65,
) -> dict[str, float]:
    """仅在训练集主动激励子集上微调闭环增益比例与耦合系数（严禁使用验证集数据，且保底 min_coupling>=0.65）。"""
    model.eval()
    frees, forceds, truths = [], [], []
    for b_idx, (th, uh, ch, cur_t, cur_u, fut_t, fut_u, m, s, _) in enumerate(train_loader):
        if b_idx >= max_batches:
            break
        th, uh, ch, cur_t, cur_u, fut_t, fut_u, m, s = [
            x.to(device) for x in (th, uh, ch, cur_t, cur_u, fut_t, fut_u, m, s)
        ]
        cov_in = ch if model.c_covar > 0 else None
        out = model(th, uh, cur_u, fut_u, m, s, covar_hist=cov_in)
        frees.append(out["free_delta_c"].float().cpu())
        forceds.append(out["forced_delta_c"].float().cpu())
        truths.append((fut_t - cur_t[:, None]).float().cpu())

    free_t = torch.cat(frees, dim=0)
    forced_t = torch.cat(forceds, dim=0)
    truth_t = torch.cat(truths, dim=0)

    gain_scale = calibrate_gain_scale(free_t, forced_t, truth_t)
    if apply_gain_scale:
        cur_cal = float(model.coal_response.gain_calibration.item())
        new_cal = np.clip(
            cur_cal * gain_scale,
            model.cfg.calibration_low,
            model.cfg.calibration_high,
        )
        effective_ratio = new_cal / max(cur_cal, 1e-6)
        model.coal_response.calibration.copy_(
            torch.tensor(
                np.log(new_cal),
                device=model.coal_response.calibration.device,
                dtype=model.coal_response.calibration.dtype,
            )
        )
        forced_t = forced_t * effective_ratio

    coupling = calibrate_closed_loop_coupling(
        free_t, forced_t, truth_t, min_coupling=min_coupling
    )
    model.set_closed_loop_coupling(coupling)
    return {
        "calibrated_gain_scale": gain_scale,
        "closed_loop_coupling": coupling,
        "gain_calibration_param": float(model.coal_response.gain_calibration.item()),
        "hist_coal_scale": float(
            model.hist_coal_scale.clamp(model.cfg.hist_coal_scale_low, 1.0).item()
        ),
    }


@torch.no_grad()
def evaluate(
    model: ControlOrientedForecaster,
    loader: DataLoader,
    cfg: ForecastConfig,
    device: torch.device,
    loss_fn: ControlTrajectoryLoss,
    with_kpi: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    """在验证集上执行离线 Oracle 评估与保持当前煤量（Hold-Current）诊断。"""
    del loss_fn
    model.eval()
    acc = {
        k: []
        for k in (
            "free",
            "forced",
            "hist_coal",
            "disturb",
            "truth",
            "current",
            "future_coal",
            "time",
            "hold",
            "eff_steps",
        )
    }
    kpis = {}
    for th, uh, ch, cur_t, cur_u, fut_t, fut_u, m, s, ts in loader:
        th, uh, ch, cur_t, cur_u, fut_t, fut_u, m, s = [
            x.to(device) for x in (th, uh, ch, cur_t, cur_u, fut_t, fut_u, m, s)
        ]
        cov_in = ch if model.c_covar > 0 else None
        out = model(th, uh, cur_u, fut_u, m, s, covar_hist=cov_in)
        hold_u = cur_u[:, None].expand(-1, cfg.horizon_steps)
        hold_out = model(th, uh, cur_u, hold_u, m, s, covar_hist=cov_in)

        acc["hold"].append(hold_out["temperature_delta_c"].cpu().numpy())
        acc["free"].append(out["temperature_delta_c"].cpu().numpy())
        acc["forced"].append(out["forced_delta_c"].cpu().numpy())
        acc["hist_coal"].append(out["hist_coal_delta_c"].cpu().numpy())
        acc["disturb"].append(out["disturb_delta_c"].cpu().numpy())
        acc["truth"].append((fut_t - cur_t[:, None]).cpu().numpy())
        acc["current"].append(cur_t.cpu().numpy())
        acc["future_coal"].append(fut_u.cpu().numpy())
        acc["time"].append(ts.cpu().numpy())
        acc["eff_steps"].append(out["in_flight_eff_steps"].cpu().numpy())

        if with_kpi and len(kpis) == 0:
            kpis["virtual_step"] = virtual_step_check(
                model,
                th[:8],
                uh[:8],
                cur_u[:8],
                m[:8],
                s[:8],
                covar_hist=cov_in[:8] if cov_in is not None else None,
            )

    arr = {k: np.concatenate(v) for k, v in acc.items()}
    x = np.linspace(0, 1, cfg.horizon_steps)
    weights = 0.5 + 1.5 * x ** 2
    weights /= weights.mean()

    truth = arr["truth"]
    report = base_metrics(truth, arr["free"], arr["current"], weights)
    report["evaluation_mode"] = "oracle_future_actual_coal"
    report["hold_current_diagnostic"] = base_metrics(truth, arr["hold"], arr["current"], weights)
    eff_hist_scale = float(
        model.hist_coal_scale.clamp(cfg.hist_coal_scale_low, 1.0).item()
        if model.use_in_flight
        else 0.0
    )
    eff_coupling = float(model.closed_loop_coupling.item())
    report.update(
        {
            "samples": len(truth),
            "evaluation_temperature_mean_c": float(arr["current"].mean()),
            "closed_loop_coupling": eff_coupling,
            "hist_coal_scale": eff_hist_scale,
            "hist_coal_delta_std_c": float(arr["hist_coal"].std()),
            "scaled_hist_coal_std_c": float((eff_hist_scale * arr["hist_coal"]).std()),
            "coupled_forced_std_c": float((eff_coupling * arr["forced"]).std()),
            "in_flight_eff_steps_mean": float(arr["eff_steps"].mean()),
            "in_flight_eff_steps_std": float(arr["eff_steps"].std()),
        }
    )
    if with_kpi:
        abs_pred = arr["current"][:, None] + arr["free"]
        abs_true = arr["current"][:, None] + truth
        report["kpi"] = kpi_metrics(abs_pred, abs_true)
        report["virtual_step_check"] = kpis.get("virtual_step", {})
    return report, arr


def train_one_fold(args: argparse.Namespace, cfg: ForecastConfig, device: torch.device) -> dict:
    """执行方案 A + 方案 B 的两阶段训练：
    - Stage 1（物理先验先行剥离 + 净扰动残差学习）：
      冻结 FOPDT 于工程先验中心 (K=15, τ=200s) 且固定 `hist_coal_scale=0.85`、`closed_loop_coupling=stage1_coupling` (0.80)，
      直接传入真实未来煤量 `fut_u`，由物理通道先行扣除在途与未来尾煤温变，强迫神经网络纯粹拟合净外部扰动残差；
    - Stage 2（受限 LPV 状态调制慢速微调）：
      以 `0.02 * lr` 小学习率放开 LPV 状态调制网络 `coal_response.scheduler`，在强物理先验锚定下联合微调。
    """
    del cfg
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    temp, coal, covars, times, train_ids, val_ids, cfg, split_info = load_data(
        args.csv,
        args.window_stride,
        use_covariates=args.use_covariates,
        in_flight_settling_ratio=args.in_flight_settling_ratio,
        decouple_recent_coal=not args.no_coal_decouple,
    )
    c_covar = len(COVAR_COLS) if args.use_covariates else 0
    model = ControlOrientedForecaster(
        cfg, C_TEMP, C_COAL, c_covar=c_covar, use_in_flight=not args.no_in_flight
    ).to(device)

    train_ds = WindowDataset(temp, coal, times, train_ids, cfg, covars=covars)
    val_ds = WindowDataset(temp, coal, times, val_ids, cfg, covars=covars)
    gen = torch.Generator().manual_seed(SEED)
    use_pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=use_pin,
        generator=gen,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_pin,
    )

    loss_fn = ControlTrajectoryLoss(
        horizon=cfg.horizon_steps,
        high_threshold_c=950,
        low_threshold_c=880,
        event_weight=args.event_weight,
        amp_weight=args.amp_weight,
        calm_p2p_c=args.calm_p2p,
        trend_weight=args.trend_weight,
        corr_weight=args.corr_weight,
        sign_weight=args.sign_weight,
        smooth_weight=args.smooth_weight,
        mean_bias_weight=args.mean_bias_weight,
        max_step_c=cfg.max_step_delta_c,
    ).to(device)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    derived_max_in_flight = max_in_flight_steps(cfg)
    protocol = {
        "version": "v4_covar" if args.use_covariates else "v4_in_flight",
        "seed": SEED,
        "training_paradigm": (
            "Scheme A (Physics-First Prior Subtraction & Net Disturbance Residual Learning) "
            "+ Scheme B (Near-Term Coal Feature Information Bottleneck Decoupling)"
        ),
        "use_in_flight": not args.no_in_flight,
        "decouple_recent_coal": not args.no_coal_decouple,
        "use_covariates": bool(args.use_covariates),
        "decoder_architecture": "SmoothTrendBasisDecoder (6 analytical basis + low-pass attention residual)",
        "input": (
            f"past60min 温度相空间(3ch+因果正交相位) + 慢变包络解耦尾煤(5ch) + FOPDT动态窗口在途余热状态机(max={derived_max_in_flight}步)"
            + (f" + 6ch热工协变量({','.join(COVAR_COLS)})" if args.use_covariates else "")
        ),
        "output": "future 15min 温度轨迹（ΔT_dist + γ*ΔT_hist_coal + c_cl*ΔT_fut_coal）",
        "validation_action_contract": (
            "future actual coal is Oracle-only; hold-current is counterfactual-ready physics-decoupled diagnostic"
        ),
        "fopdt_prior": {
            "K_base": cfg.gain_c_per_tph,
            "tau_base_s": cfg.tau_seconds,
            "dead_steps": cfg.dead_steps,
            "K_range": [cfg.gain_low, cfg.gain_high],
            "tau_range_s": [cfg.tau_low, cfg.tau_high],
            "in_flight_steps_floor": cfg.in_flight_steps,
            "in_flight_settling_ratio": cfg.in_flight_settling_ratio,
            "in_flight_max_steps_derived": derived_max_in_flight,
            "stage1_coupling": args.stage1_coupling,
            "min_closed_loop_coupling": args.min_coupling,
            "hist_coal_scale_low": cfg.hist_coal_scale_low,
        },
        "loss": {
            "event_weight_sq": args.event_weight,
            "corr_weight_pearson": args.corr_weight,
            "trend_weight_multiscale": args.trend_weight,
            "sign_weight_anchors": args.sign_weight,
            "amp_weight_phase_gated": args.amp_weight,
            "calm_p2p_gate_c": args.calm_p2p,
            "smooth_weight_origin_anchored": args.smooth_weight,
            "mean_bias_weight_dual_dim": args.mean_bias_weight,
        },
        "fold": "train target end < 2026-10-01; validation [2026-10-01 01:00, 2026-10-15)",
        "open_kiln_filter": "temp>800 & sat>800，完整连续窗口（统计量严格仅由训练集估计）",
        "config": cfg.__dict__,
        "channels": {"temp": C_TEMP, "coal": C_COAL, "covar": c_covar},
        "split_info": split_info,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "stage1_epochs": args.stage1_epochs,
        "recalibrate_each_epoch": bool(args.recalibrate_each_epoch),
        "test_inference": False,
        "IOT_access": False,
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------- Stage 1（方案 A）：锁定物理先验中心，先行剥离在途与未来煤量热效应，纯训练净扰动残差网络 ----------
    model.set_closed_loop_coupling(args.stage1_coupling)
    for p in model.parameters():
        p.requires_grad_(True)
    for p in model.coal_response.parameters():
        p.requires_grad_(False)
    # Stage 1 同步锁定 hist_coal_scale 于物理初值 (0.85)，防止黑盒与在途余热争抢梯度
    model.hist_coal_scale.requires_grad_(False)

    backbone_params = [p for p in model.parameters() if p.requires_grad]
    opt1 = torch.optim.AdamW(backbone_params, lr=args.lr)

    best = float("inf")
    best_report = None
    stale = 0
    started = time.time()
    history = []

    for epoch in range(1, args.stage1_epochs + 1):
        model.train()
        tl, n = 0.0, 0
        for th, uh, ch, cur_t, cur_u, fut_t, fut_u, m, s, _ in train_loader:
            th, uh, ch, cur_t, cur_u, fut_t, fut_u, m, s = [
                x.to(device, non_blocking=True)
                for x in (th, uh, ch, cur_t, cur_u, fut_t, fut_u, m, s)
            ]
            opt1.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                cov_in = ch if model.c_covar > 0 else None
                # 方案 A 关键点：Stage 1 直接传入 fut_u 并以合成轨迹 temperature_delta_c 计算损失，
                # 等价于先从 truth 中扣除固定的物理在途余热与未来煤量热响应，让 disturb_delta_c 纯粹拟合净外部扰动
                outv = model(th, uh, cur_u, fut_u, m, s, covar_hist=cov_in)
                truth = fut_t - cur_t[:, None]
                lp = loss_fn(outv["temperature_delta_c"], truth, cur_t)
                loss = lp["total"] + outv["prior_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone_params, 1.0)
            opt1.step()
            tl += float(loss.detach()) * len(th)
            n += len(th)

        rep, _ = evaluate(model, val_loader, cfg, device, loss_fn, with_kpi=True)
        kpi_s1 = rep.get("kpi") or {}
        s1_score = (
            rep["weighted_15min_rmse_c"]
            + 10.0 * max(0.0, 0.65 - float(kpi_s1.get("mean_pearson_r") or 0.0))
            + 10.0 * max(0.0, 0.75 - float(kpi_s1.get("direction_accuracy_multi_horizon") or 0.0))
        )
        row = {
            "epoch": f"s1-{epoch}",
            "train_loss": tl / max(n, 1),
            **{k: v for k, v in rep.items() if k not in ("kpi", "virtual_step_check")},
            "kpi": rep.get("kpi"),
            "selection_score": s1_score,
            "calibration": float(model.coal_response.gain_calibration.item()),
            "elapsed_s": time.time() - started,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print("stage1", json.dumps(row, ensure_ascii=False), flush=True)

        if s1_score < best:
            best = s1_score
            best_report = rep
            stale = 0
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "stage": 1},
                out / "best_stage1.pt",
            )
        else:
            stale += 1
        if stale >= 3:
            break

    if (out / "best_stage1.pt").exists():
        ckpt1 = torch.load(out / "best_stage1.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt1["state_dict"])

    # Stage 1 结束后，在训练集同向主动激励窗口上微调增益比例与闭环耦合系数（保底 min_coupling>=0.65）
    cal_info = calibrate_on_train_subset(
        model,
        train_loader,
        device,
        max_batches=24,
        apply_gain_scale=True,
        min_coupling=args.min_coupling,
    )
    if best_report is not None:
        best_report["stage1_calibration"] = cal_info
    (out / "stage1_report.json").write_text(
        json.dumps(best_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------- Stage 2：解冻 LPV 状态调制网络，在强物理先验锚定下慢速联合微调 ----------
    for p in model.parameters():
        p.requires_grad_(True)
    lpv_param_ids = {id(p) for p in model.coal_response.parameters()}
    base_group = [p for p in model.parameters() if id(p) not in lpv_param_ids]
    lpv_group = list(model.coal_response.parameters())
    opt2 = torch.optim.AdamW(
        [
            {"params": base_group, "lr": args.lr * 0.5},
            {"params": lpv_group, "lr": args.lr * 0.02},
        ]
    )

    best = float("inf")
    best_report = None
    stale = 0
    s2_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tl, n = 0.0, 0
        for th, uh, ch, cur_t, cur_u, fut_t, fut_u, m, s, _ in train_loader:
            th, uh, ch, cur_t, cur_u, fut_t, fut_u, m, s = [
                x.to(device, non_blocking=True)
                for x in (th, uh, ch, cur_t, cur_u, fut_t, fut_u, m, s)
            ]
            opt2.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                cov_in = ch if model.c_covar > 0 else None
                outv = model(th, uh, cur_u, fut_u, m, s, covar_hist=cov_in)
                truth = fut_t - cur_t[:, None]
                lp = loss_fn(outv["temperature_delta_c"], truth, cur_t)
                loss = lp["total"] + 50.0 * outv["prior_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            tl += float(loss.detach()) * len(th)
            n += len(th)

        if args.recalibrate_each_epoch:
            calibrate_on_train_subset(
                model,
                train_loader,
                device,
                max_batches=8,
                apply_gain_scale=False,
                min_coupling=args.min_coupling,
            )

        rep, arr = evaluate(model, val_loader, cfg, device, loss_fn, with_kpi=True)
        kpi_dict = rep.get("kpi") or {}
        body_r = float(kpi_dict.get("body_p2p_ratio") or 0.0)
        mean_r = float(kpi_dict.get("mean_pearson_r") or 0.0)
        dir_acc = float(kpi_dict.get("direction_accuracy_multi_horizon") or 0.0)

        sel_score = (
            rep["weighted_15min_rmse_c"]
            + 12.0 * max(0.0, 0.80 - body_r)
            + 8.0 * max(0.0, body_r - 1.25)
            + 10.0 * max(0.0, 0.65 - mean_r)
            + 10.0 * max(0.0, 0.75 - dir_acc)
        )
        row = {
            "epoch": f"s2-{epoch}",
            "train_loss": tl / max(n, 1),
            **{k: v for k, v in rep.items() if k not in ("kpi", "virtual_step_check")},
            "kpi": rep.get("kpi"),
            "virtual_step_check": rep.get("virtual_step_check"),
            "selection_score": sel_score,
            "elapsed_s": time.time() - s2_start,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print("stage2", json.dumps(row, ensure_ascii=False), flush=True)

        if sel_score < best:
            best = sel_score
            best_report = rep
            stale = 0
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "stage": 2},
                out / "best.pt",
            )
            np.savez_compressed(
                out / "validation_predictions.npz",
                free=arr["free"],
                forced=arr["forced"],
                hist_coal=arr["hist_coal"],
                disturb=arr["disturb"],
                hold=arr["hold"],
                truth=arr["truth"],
                current=arr["current"],
                future_coal=arr["future_coal"],
                time=arr["time"],
                eff_steps=arr["eff_steps"],
            )
        else:
            stale += 1
        if epoch >= args.min_epochs and stale >= args.patience:
            break

    (out / "validation_metrics.json").write_text(
        json.dumps(best_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "status.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "version": "v4_covar" if args.use_covariates else "v4_in_flight",
                "best_validation_weighted_rmse_c": (
                    best_report["weighted_15min_rmse_c"] if best_report else best
                ),
                "best_selection_score": best,
                "elapsed_seconds": time.time() - started,
                "warning": (
                    "开发验证集，非盲测；主要指标为 future_actual_coal Oracle；"
                    "物理先验先行剥离已启用；没有在线性能或因果辨识结论"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return best_report


def main() -> None:
    p = argparse.ArgumentParser(description="v4 控制导向分解炉温度轨迹预测训练脚本（方案 A + 方案 B）")
    p.add_argument("--csv", required=True, help="5s 原始工业运行数据 CSV 路径")
    p.add_argument("--output", required=True, help="模型权重与评估报告输出目录")
    p.add_argument("--device", default="cuda:1", help="训练设备（如 cuda:0 / cpu）")
    p.add_argument("--window-stride", type=int, default=16, help="训练滑动窗口步长（15s 单位）")
    p.add_argument("--batch-size", type=int, default=384, help="训练与验证批次大小")
    p.add_argument("--stage1-epochs", type=int, default=5, help="Stage 1 物理先验锁定残差预训练轮数")
    p.add_argument("--epochs", type=int, default=12, help="Stage 2 LPV 慢速微调最大轮数")
    p.add_argument("--min-epochs", type=int, default=5, help="Stage 2 早停最小轮数")
    p.add_argument("--patience", type=int, default=4, help="Stage 2 早停容忍轮数")
    p.add_argument("--workers", type=int, default=4, help="DataLoader 工作进程数")
    p.add_argument("--lr", type=float, default=3e-4, help="主干网络基准学习率")
    p.add_argument("--event-weight", type=float, default=1.0, help="工艺红线越界惩罚权重")
    p.add_argument("--amp-weight", type=float, default=0.5, help="皮尔逊相位门控振幅匹配权重")
    p.add_argument("--calm-p2p", type=float, default=12.0, help="激活振幅损失的真实峰谷差阈值(℃)")
    p.add_argument("--trend-weight", type=float, default=0.45, help="四级多尺度跨步差分趋势权重")
    p.add_argument("--corr-weight", type=float, default=0.85, help="去均值波形皮尔逊相关损失权重")
    p.add_argument("--sign-weight", type=float, default=0.60, help="3/6/10/15min关键锚点净方向同号损失权重")
    p.add_argument("--smooth-weight", type=float, default=0.08, help="二阶差分曲率平滑权重")
    p.add_argument("--mean-bias-weight", type=float, default=0.5, help="双维均值零偏锚定权重")
    p.add_argument("--stage1-coupling", type=float, default=0.80, help="Stage 1 物理受迫通道先行剥离耦合系数")
    p.add_argument("--min-coupling", type=float, default=0.65, help="闭环受迫响应耦合系数物理保底下限")
    p.add_argument(
        "--in-flight-settling-ratio",
        type=float,
        default=0.99,
        help="FOPDT 在途余热有效窗口收敛比例阈值（默认 0.99，对应 4.605 倍动态时间常数）",
    )
    p.add_argument("--use-covariates", action="store_true", help="方向一：接入其余6个热工协变量")
    p.add_argument("--no-in-flight", action="store_true", help="禁用方向二在途余热分解")
    p.add_argument("--no-coal-decouple", action="store_true", help="禁用方案 B 编码器近端高频煤量解耦屏蔽")
    p.add_argument(
        "--recalibrate-each-epoch",
        action="store_true",
        help="是否在 Stage 2 每轮结束时重新网格搜索耦合系数（默认关闭以保持梯度稳定）",
    )
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 但不可用")
    device = torch.device(args.device)
    report = train_one_fold(args, ForecastConfig(), device)
    print("FINAL", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()