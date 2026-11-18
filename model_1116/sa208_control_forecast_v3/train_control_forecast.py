#!/usr/bin/env python3
"""v3 控制导向预测训练（sa-208 离线，只读 CSV；不访问 IOT、不下发动作）。

相对 v2 / v2_meanbias 的训练流程整改：
1. 修正两阶段训练顺序倒置问题：
   - 原 Stage 1 将自由响应主干冻结在“随机初始化”状态并强行用闭环负相关未来煤量（corr≈-0.67）
     拟合 ΔT，导致 K_t 在第 1 轮即塌缩至下限 7.5、τ_t 塌缩至上限 280s（Stage 1 RMSE 高达 63.9℃）。
   - 现 Stage 1 先冻结物理 FOPDT 于先验中心（K=15, τ=200s），训练双流自由响应主干学习 15~30min
     自治周期振荡轨迹；随后在训练集子集上调用 `calibrate_gain_scale` 与 `calibrate_closed_loop_coupling`
     完成闭环耦合标定（严禁使用验证集标定）。
2. Stage 2 联合微调：
   - 采用分组学习率（主干 lr，LPV 调制器 0.25*lr）与完整先验锚定（同时约束 K_t、τ_t、calibration 及边界势垒），
     结合起点连续性锚定 Loss、相位对齐门控振幅 Loss 与双维均值偏置锚定，消除 step-0 +26℃ 伪尖峰。
3. 验证协议保持严格边界：
   - 未来真实煤量仅作为事后 Oracle 条件输入，另报“当前煤量保持”历史标签诊断；二者均不代表在线闭环性能。
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
    ControlOrientedForecaster, ControlTrajectoryLoss, ForecastConfig,
    build_temp_features, build_coal_features, calibrate_closed_loop_coupling,
    calibrate_gain_scale, coal_recent_stats, kpi_metrics, virtual_step_check,
)

CUT = pd.Timestamp("2026-10-01 00:00:00")
VAL_END = pd.Timestamp("2026-10-15 00:00:00")
SEED = 20261116
C_TEMP, C_COAL = 3, 5


class WindowDataset(Dataset):
    def __init__(self, temp, coal, times, origins, config, subsample: int | None = None):
        self.temp, self.coal, self.times = temp, coal, times
        self.config = config
        self.origins = origins[::subsample] if subsample else origins

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, j):
        i = int(self.origins[j]); cfg = self.config
        t_hist = self.temp[i - cfg.history_steps + 1:i + 1]
        u_hist = self.coal[i - cfg.history_steps + 1:i + 1]
        idx = np.arange(i + 1, i + cfg.horizon_steps + 1)
        th = build_temp_features(t_hist, cfg)
        uh = build_coal_features(u_hist, cfg)
        m, s = coal_recent_stats(u_hist, cfg)
        return (torch.from_numpy(th.astype(np.float32)),
                torch.from_numpy(uh.astype(np.float32)),
                torch.tensor(self.temp[i], dtype=torch.float32),
                torch.tensor(self.coal[i], dtype=torch.float32),
                torch.from_numpy(self.temp[idx].astype(np.float32)),
                torch.from_numpy(self.coal[idx].astype(np.float32)),
                torch.tensor(m, dtype=torch.float32),
                torch.tensor(s, dtype=torch.float32),
                torch.tensor(self.times[i].value, dtype=torch.int64))


def load_data(csv_path, stride):
    cols = ["time", "decompose_temp", "feed_actual", "sat_temp"]
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
    valid = (np.isfinite(temp) & np.isfinite(coal) & np.isfinite(sat)
             & (temp > 800.0) & (sat > 800.0))
    bad_prefix = np.r_[0, np.cumsum(~valid)]
    cfg = ForecastConfig(coal_mean_tph=float(np.nanmean(coal[valid])),
                         coal_scale_tph=max(float(np.nanstd(coal[valid])), 1e-3))
    first = cfg.history_steps - 1
    last = len(frame) - cfg.horizon_steps - 1
    origins = np.arange(first, last + 1, stride, dtype=np.int64)
    keep = (bad_prefix[origins + cfg.horizon_steps + 1]
            - bad_prefix[origins - cfg.history_steps + 1] == 0)
    origins = origins[keep]
    t0 = times.take(origins); t1 = times.take(origins + cfg.horizon_steps)
    train = origins[t1 < CUT]
    val = origins[(t0 >= CUT + pd.Timedelta(minutes=60)) & (t1 < VAL_END)]
    if len(train) == 0 or len(val) == 0:
        raise ValueError(f"切分为空 train={len(train)} val={len(val)}")
    return temp, coal, times, train, val, cfg, {
        "source_rows": source_rows, "rows_15s": len(frame),
        "valid_fraction": float(valid.mean()), "stride_samples": stride,
        "train_origins": len(train), "validation_origins": len(val),
        "train_last_origin": str(times[train[-1]]),
        "validation_first_origin": str(times[val[0]]),
        "validation_last_target": str(times[val[-1] + cfg.horizon_steps]),
    }


def base_metrics(target, pred, cur, weights):
    err = pred - target
    w = weights[None, :]
    horizon_bias = err.mean(0)
    return {
        "weighted_15min_rmse_c": float(np.sqrt(np.mean(np.sum(w * err**2, 1) / w.sum()))),
        "endpoint_rmse_c": float(np.sqrt((err[:, -1]**2).mean())),
        "all_point_p90_abs_error_c": float(np.quantile(np.abs(err), 0.9)),
        "max_abs_horizon_mean_bias_c": float(np.abs(horizon_bias).max()),
        "step0_mean_bias_c": float(horizon_bias[0]),
        "step0_rmse_c": float(np.sqrt((err[:, 0]**2).mean())),
        "per_window_mean_mae_c": float(np.abs(err.mean(1)).mean()),
        "mean_abs_error_c": float(np.abs(err).mean()),
    }


@torch.no_grad()
def calibrate_on_train_subset(model, train_loader, device, max_batches: int = 16) -> dict[str, float]:
    """仅在训练集子集上标定闭环增益比例与耦合系数（严禁使用验证集数据）。"""
    model.eval()
    frees, forceds, truths = [], [], []
    for b_idx, (th, uh, cur_t, cur_u, fut_t, fut_u, m, s, _) in enumerate(train_loader):
        if b_idx >= max_batches:
            break
        th, uh, cur_t, cur_u, fut_t, fut_u, m, s = [
            x.to(device) for x in (th, uh, cur_t, cur_u, fut_t, fut_u, m, s)
        ]
        out = model(th, uh, cur_u, fut_u, m, s)
        frees.append(out["free_delta_c"].float().cpu())
        forceds.append(out["forced_delta_c"].float().cpu())
        truths.append((fut_t - cur_t[:, None]).float().cpu())
    free_t = torch.cat(frees, dim=0)
    forced_t = torch.cat(forceds, dim=0)
    truth_t = torch.cat(truths, dim=0)
    gain_scale = calibrate_gain_scale(free_t, forced_t, truth_t)
    coupling = calibrate_closed_loop_coupling(free_t, forced_t, truth_t, min_coupling=0.10)
    model.set_closed_loop_coupling(coupling)
    return {"calibrated_gain_scale": gain_scale, "closed_loop_coupling": coupling}


@torch.no_grad()
def evaluate(model, loader, cfg, device, loss_fn, with_kpi: bool):
    model.eval()
    acc = {k: [] for k in ("free", "forced", "truth", "current", "future_coal", "time", "hold")}
    kpis = {}
    for th, uh, cur_t, cur_u, fut_t, fut_u, m, s, ts in loader:
        th, uh, cur_t, cur_u, fut_t, fut_u, m, s = [x.to(device) for x in
                                                    (th, uh, cur_t, cur_u, fut_t, fut_u, m, s)]
        out = model(th, uh, cur_u, fut_u, m, s)
        # 当前煤量保持只是历史标签上的诊断，不是有效反事实；单独报告，绝不混入 Oracle 指标。
        hold_u = cur_u[:, None].expand(-1, cfg.horizon_steps)
        hold_out = model(th, uh, cur_u, hold_u, m, s)
        acc["hold"].append(hold_out["temperature_delta_c"].cpu().numpy())
        acc["free"].append(out["temperature_delta_c"].cpu().numpy())
        acc["forced"].append(out["forced_delta_c"].cpu().numpy())
        acc["truth"].append((fut_t - cur_t[:, None]).cpu().numpy())
        acc["current"].append(cur_t.cpu().numpy())
        acc["future_coal"].append(fut_u.cpu().numpy())
        acc["time"].append(ts.cpu().numpy())
        if with_kpi and len(kpis) == 0:
            # 取第一个 batch 做 KPI-4 虚拟阶跃（仅诊断，限制成本）
            kpis["virtual_step"] = virtual_step_check(
                model, th[:8], uh[:8], cur_u[:8], m[:8], s[:8])
    arr = {k: np.concatenate(v) for k, v in acc.items()}
    x = np.linspace(0, 1, cfg.horizon_steps)
    weights = (0.5 + 1.5 * x**2); weights /= weights.mean()
    truth = arr["truth"]
    report = base_metrics(truth, arr["free"], arr["current"], weights)
    report["evaluation_mode"] = "oracle_future_actual_coal"
    report["hold_current_diagnostic"] = base_metrics(truth, arr["hold"], arr["current"], weights)
    report.update({
        "samples": len(truth),
        "evaluation_temperature_mean_c": float(arr["current"].mean()),
        "closed_loop_coupling": float(model.closed_loop_coupling.item()),
    })
    if with_kpi:
        abs_pred = arr["current"][:, None] + arr["free"]
        abs_true = arr["current"][:, None] + truth
        report["kpi"] = kpi_metrics(abs_pred, abs_true)
        report["virtual_step_check"] = kpis.get("virtual_step", {})
    return report, arr


def train_one_fold(args, cfg, device):
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    temp, coal, times, train_ids, val_ids, cfg, split_info = load_data(args.csv, args.window_stride)
    model = ControlOrientedForecaster(cfg, C_TEMP, C_COAL).to(device)
    train_ds = WindowDataset(temp, coal, times, train_ids, cfg)
    val_ds = WindowDataset(temp, coal, times, val_ids, cfg)
    gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, generator=gen,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    loss_fn = ControlTrajectoryLoss(horizon=cfg.horizon_steps, high_threshold_c=950,
                                    low_threshold_c=880, event_weight=args.event_weight,
                                    amp_weight=args.amp_weight, calm_p2p_c=args.calm_p2p,
                                    trend_weight=args.trend_weight,
                                    smooth_weight=args.smooth_weight,
                                    mean_bias_weight=args.mean_bias_weight,
                                    max_step_c=cfg.max_step_delta_c).to(device)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "version": "v3", "seed": SEED,
        "input": "past60min 温度相空间(3ch+因果正交相位) + 尾煤/包络(5ch)；推理时另有候选未来煤量",
        "output": "future 15min 温度轨迹（起点热惯性门控 + 增量锚定 T_t + ΔT）",
        "architecture": (
            "dual-stream PatchTST(shared encoder + causal phase) + context-conditioned query "
            "+ Cross-Attention + horizon Conv1d mixer + origin inertia gate + LPV-FOPDT"
        ),
        "validation_action_contract": (
            "future actual coal is Oracle-only; hold-current is historical-label diagnostic, not counterfactual"
        ),
        "stage1_interpretation": (
            "Stage 1 pretrains free-response backbone under physical prior center, followed by "
            "train-set closed-loop coupling calibration; not causal identification"
        ),
        "fopdt_prior": {"K_base": cfg.gain_c_per_tph, "tau_base_s": cfg.tau_seconds,
                        "dead_steps": cfg.dead_steps, "K_range": [cfg.gain_low, cfg.gain_high],
                        "tau_range_s": [cfg.tau_low, cfg.tau_high],
                        "calibration_range": [cfg.calibration_low, cfg.calibration_high],
                        "note": "工程先验+完整边界软势垒锚定，非因果辨识"},
        "loss": {"far_weight": "0.5->2.0 quadratic normalized", "peak_thresholds_c": [880, 950],
                 "event_weight_sq": args.event_weight, "amp_weight_phase_aligned": args.amp_weight,
                 "calm_p2p_gate_c": args.calm_p2p, "trend_weight_origin_anchored": args.trend_weight,
                 "smooth_weight_origin_anchored": args.smooth_weight,
                 "mean_bias_weight_dual_dim": args.mean_bias_weight},
        "fold": "train target end < 2026-10-01; validation [2026-10-01 01:00, 2026-10-15)",
        "open_kiln_filter": "temp>800 & sat>800，完整连续窗口",
        "config": cfg.__dict__, "channels": {"temp": C_TEMP, "coal": C_COAL},
        "split_info": split_info, "batch_size": args.batch_size,
        "max_epochs": args.epochs, "stage1_epochs": args.stage1_epochs,
        "test_inference": False, "IOT_access": False,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2),
                                       encoding="utf-8")

    # ---------- Stage 1：自治自由响应主干预训练（冻结 FOPDT 于先验中心，避免随机主干污染物理参数）----------
    model.set_closed_loop_coupling(0.0)
    for p in model.parameters():
        p.requires_grad_(True)
    for p in model.coal_response.parameters():
        p.requires_grad_(False)
    backbone_params = [p for p in model.parameters() if p.requires_grad]
    opt1 = torch.optim.AdamW(backbone_params, lr=args.lr)
    best = float("inf"); best_report = None; stale = 0; started = time.time()
    history = []
    for epoch in range(1, args.stage1_epochs + 1):
        model.train()
        tl, n = 0.0, 0
        for th, uh, cur_t, cur_u, fut_t, fut_u, m, s, _ in train_loader:
            th, uh, cur_t, cur_u, fut_t, fut_u, m, s = [x.to(device, non_blocking=True)
                                                        for x in (th, uh, cur_t, cur_u,
                                                                  fut_t, fut_u, m, s)]
            opt1.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                hold_u = cur_u[:, None].expand(-1, cfg.horizon_steps)
                outv = model(th, uh, cur_u, hold_u, m, s)
                truth = fut_t - cur_t[:, None]
                lp = loss_fn(outv["free_delta_c"], truth, cur_t)
                loss = lp["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone_params, 1.0)
            opt1.step()
            tl += float(loss.detach()) * len(th); n += len(th)
        rep, _ = evaluate(model, val_loader, cfg, device, loss_fn, with_kpi=False)
        row = {"epoch": f"s1-{epoch}", "train_loss": tl / max(n, 1), **rep,
               "calibration": float(model.coal_response.gain_calibration.item()),
               "elapsed_s": time.time() - started}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print("stage1", json.dumps(row, ensure_ascii=False), flush=True)
        if rep["weighted_15min_rmse_c"] < best:
            best = rep["weighted_15min_rmse_c"]; best_report = rep; stale = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "stage": 1}, out / "best_stage1.pt")
        else:
            stale += 1
        if stale >= 3:
            break

    # 载入 Stage 1 最佳自由响应权重，并在训练集子集上执行闭环耦合标定
    if (out / "best_stage1.pt").exists():
        ckpt1 = torch.load(out / "best_stage1.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt1["state_dict"])
    cal_info = calibrate_on_train_subset(model, train_loader, device)
    if best_report is not None:
        best_report["stage1_calibration"] = cal_info
    (out / "stage1_report.json").write_text(
        json.dumps(best_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- Stage 2：全模型联合微调（相位对齐振幅 Loss + 完整 LPV 先验锚定）----------
    for p in model.parameters():
        p.requires_grad_(True)
    lpv_param_ids = {id(p) for p in model.coal_response.parameters()}
    base_group = [p for p in model.parameters() if id(p) not in lpv_param_ids]
    lpv_group = list(model.coal_response.parameters())
    opt2 = torch.optim.AdamW([
        {"params": base_group, "lr": args.lr},
        {"params": lpv_group, "lr": args.lr * 0.04},
    ])
    best = float("inf"); best_report = None; stale = 0
    s2_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl, n = 0.0, 0
        for th, uh, cur_t, cur_u, fut_t, fut_u, m, s, _ in train_loader:
            th, uh, cur_t, cur_u, fut_t, fut_u, m, s = [x.to(device, non_blocking=True)
                                                        for x in (th, uh, cur_t, cur_u,
                                                                  fut_t, fut_u, m, s)]
            opt2.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                outv = model(th, uh, cur_u, fut_u, m, s)
                truth = fut_t - cur_t[:, None]
                lp = loss_fn(outv["temperature_delta_c"], truth, cur_t)
                # 强先验锚定保护 LPV 物理参数不因闭环 PID 负相关而塌缩至边界
                loss = lp["total"] + 35.0 * outv["prior_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            tl += float(loss.detach()) * len(th); n += len(th)
        # 每轮结束在训练集子集更新闭环耦合标定（严禁触碰验证集）
        calibrate_on_train_subset(model, train_loader, device, max_batches=8)
        rep, arr = evaluate(model, val_loader, cfg, device, loss_fn, with_kpi=True)
        row = {"epoch": f"s2-{epoch}", "train_loss": tl / max(n, 1), **{k: v for k, v in rep.items()
                                                                         if k not in ("kpi", "virtual_step_check")},
               "kpi": rep.get("kpi"), "virtual_step_check": rep.get("virtual_step_check"),
               "elapsed_s": time.time() - s2_start}
        # 控制导向选模准则：兼顾加权轨迹 RMSE 与剔除前 3 步的真实振幅还原率（目标 >=0.85）
        body_r = float((rep.get("kpi") or {}).get("body_p2p_ratio") or 0.0)
        sel_score = rep["weighted_15min_rmse_c"] + 15.0 * max(0.0, 0.85 - body_r)
        row["selection_score"] = sel_score
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print("stage2", json.dumps(row, ensure_ascii=False), flush=True)
        if sel_score < best:
            best = sel_score; best_report = rep; stale = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "stage": 2}, out / "best.pt")
            np.savez_compressed(out / "validation_predictions.npz",
                                free=arr["free"], forced=arr["forced"],
                                hold=arr["hold"], truth=arr["truth"],
                                current=arr["current"], future_coal=arr["future_coal"],
                                time=arr["time"])
        else:
            stale += 1
        if epoch >= args.min_epochs and stale >= args.patience:
            break

    (out / "validation_metrics.json").write_text(
        json.dumps(best_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "status.json").write_text(json.dumps({
        "state": "completed", "version": "v3",
        "best_validation_weighted_rmse_c": best_report["weighted_15min_rmse_c"] if best_report else best,
        "best_selection_score": best,
        "elapsed_seconds": time.time() - started,
        "warning": "开发验证集，非盲测；主要指标为 future_actual_coal Oracle；hold-current 仅为历史标签诊断，非反事实；没有在线性能或因果辨识结论"
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return best_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--window-stride", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=384)
    p.add_argument("--stage1-epochs", type=int, default=4)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--min-epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--event-weight", type=float, default=1.0)
    p.add_argument("--amp-weight", type=float, default=0.9)
    p.add_argument("--calm-p2p", type=float, default=12.0)
    p.add_argument("--trend-weight", type=float, default=0.15)
    p.add_argument("--smooth-weight", type=float, default=0.05)
    p.add_argument("--mean-bias-weight", type=float, default=0.6)
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 但不可用")
    device = torch.device(args.device)
    report = train_one_fold(args, ForecastConfig(), device)
    print("FINAL", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
