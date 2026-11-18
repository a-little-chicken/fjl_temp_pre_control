#!/usr/bin/env python3
"""v4 控制导向预测训练（支持方向二 120步在途余热三分量分解 与 方向一 多源热工协变量对照实验）。

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
    ControlOrientedForecaster, ControlTrajectoryLoss, ForecastConfig,
    build_coal_features, build_covar_features, build_temp_features,
    calibrate_closed_loop_coupling, calibrate_gain_scale, coal_recent_stats,
    kpi_metrics, virtual_step_check,
)

CUT = pd.Timestamp("2026-10-01 00:00:00")
VAL_END = pd.Timestamp("2026-10-15 00:00:00")
SEED = 20261116
C_TEMP, C_COAL = 3, 5
COVAR_COLS = ["c5b_valve", "sat_temp", "decompose_press", "alt_fuel", "kiln_current", "ladder_pressure"]


class WindowDataset(Dataset):
    def __init__(self, temp, coal, times, origins, config,
                 covars: np.ndarray | None = None, subsample: int | None = None):
        self.temp, self.coal, self.times = temp, coal, times
        self.covars = covars
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
        if self.covars is not None:
            ch = build_covar_features(self.covars[i - cfg.history_steps + 1:i + 1], cfg)
        else:
            ch = np.zeros((1, 1), dtype=np.float32)
        m, s = coal_recent_stats(u_hist, cfg)
        return (torch.from_numpy(th.astype(np.float32)),
                torch.from_numpy(uh.astype(np.float32)),
                torch.from_numpy(ch.astype(np.float32)),
                torch.tensor(self.temp[i], dtype=torch.float32),
                torch.tensor(self.coal[i], dtype=torch.float32),
                torch.from_numpy(self.temp[idx].astype(np.float32)),
                torch.from_numpy(self.coal[idx].astype(np.float32)),
                torch.tensor(m, dtype=torch.float32),
                torch.tensor(s, dtype=torch.float32),
                torch.tensor(self.times[i].value, dtype=torch.int64))


def load_data(csv_path, stride, use_covariates: bool = False):
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
    valid = (np.isfinite(temp) & np.isfinite(coal) & np.isfinite(sat)
             & (temp > 800.0) & (sat > 800.0))
    covars = None
    if use_covariates:
        cov_list = []
        for c in COVAR_COLS:
            arr = pd.Series(frame[c]).ffill().bfill().fillna(0.0).to_numpy(np.float32)
            cov_list.append(arr)
        covars = np.stack(cov_list, axis=-1)
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
    return temp, coal, covars, times, train, val, cfg, {
        "source_rows": source_rows, "rows_15s": len(frame),
        "valid_fraction": float(valid.mean()), "stride_samples": stride,
        "train_origins": len(train), "validation_origins": len(val),
        "train_last_origin": str(times[train[-1]]),
        "validation_first_origin": str(times[val[0]]),
        "validation_last_target": str(times[val[-1] + cfg.horizon_steps]),
        "use_covariates": use_covariates,
        "covariate_columns": COVAR_COLS if use_covariates else [],
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
    coupling = calibrate_closed_loop_coupling(free_t, forced_t, truth_t, min_coupling=0.10)
    model.set_closed_loop_coupling(coupling)
    return {
        "calibrated_gain_scale": gain_scale,
        "closed_loop_coupling": coupling,
        "hist_coal_scale": float(model.hist_coal_scale.clamp(0.0, 1.0).item()),
    }


@torch.no_grad()
def evaluate(model, loader, cfg, device, loss_fn, with_kpi: bool):
    model.eval()
    acc = {k: [] for k in ("free", "forced", "hist_coal", "disturb", "truth", "current", "future_coal", "time", "hold")}
    kpis = {}
    for th, uh, ch, cur_t, cur_u, fut_t, fut_u, m, s, ts in loader:
        th, uh, ch, cur_t, cur_u, fut_t, fut_u, m, s = [x.to(device) for x in
                                                        (th, uh, ch, cur_t, cur_u, fut_t, fut_u, m, s)]
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
        if with_kpi and len(kpis) == 0:
            kpis["virtual_step"] = virtual_step_check(
                model, th[:8], uh[:8], cur_u[:8], m[:8], s[:8],
                covar_hist=cov_in[:8] if cov_in is not None else None)
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
        "hist_coal_scale": float(model.hist_coal_scale.clamp(0.0, 1.0).item()),
        "hist_coal_delta_std_c": float(arr["hist_coal"].std()),
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
    temp, coal, covars, times, train_ids, val_ids, cfg, split_info = load_data(
        args.csv, args.window_stride, use_covariates=args.use_covariates)
    c_covar = len(COVAR_COLS) if args.use_covariates else 0
    model = ControlOrientedForecaster(
        cfg, C_TEMP, C_COAL, c_covar=c_covar, use_in_flight=not args.no_in_flight
    ).to(device)
    train_ds = WindowDataset(temp, coal, times, train_ids, cfg, covars=covars)
    val_ds = WindowDataset(temp, coal, times, val_ids, cfg, covars=covars)
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
        "version": "v4_covar" if args.use_covariates else "v4_in_flight",
        "seed": SEED,
        "use_in_flight": not args.no_in_flight,
        "use_covariates": bool(args.use_covariates),
        "input": (
            f"past60min 温度相空间(3ch+因果正交相位) + 尾煤/包络(5ch) + 15min在途余热状态机"
            + (f" + 6ch热工协变量({','.join(COVAR_COLS)})" if args.use_covariates else "")
        ),
        "output": "future 15min 温度轨迹（ΔT_dist + γ*ΔT_hist_coal + c_cl*ΔT_fut_coal）",
        "validation_action_contract": (
            "future actual coal is Oracle-only; hold-current is historical-label diagnostic, not counterfactual"
        ),
        "fopdt_prior": {"K_base": cfg.gain_c_per_tph, "tau_base_s": cfg.tau_seconds,
                        "dead_steps": cfg.dead_steps, "K_range": [cfg.gain_low, cfg.gain_high],
                        "tau_range_s": [cfg.tau_low, cfg.tau_high],
                        "in_flight_steps": cfg.in_flight_steps},
        "loss": {"event_weight_sq": args.event_weight, "amp_weight_phase_aligned": args.amp_weight,
                 "calm_p2p_gate_c": args.calm_p2p, "trend_weight_origin_anchored": args.trend_weight,
                 "smooth_weight_origin_anchored": args.smooth_weight,
                 "mean_bias_weight_dual_dim": args.mean_bias_weight},
        "fold": "train target end < 2026-10-01; validation [2026-10-01 01:00, 2026-10-15)",
        "open_kiln_filter": "temp>800 & sat>800，完整连续窗口",
        "config": cfg.__dict__, "channels": {"temp": C_TEMP, "coal": C_COAL, "covar": c_covar},
        "split_info": split_info, "batch_size": args.batch_size,
        "max_epochs": args.epochs, "stage1_epochs": args.stage1_epochs,
        "test_inference": False, "IOT_access": False,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2),
                                       encoding="utf-8")

    # ---------- Stage 1：自治自由响应主干 + 历史在途余热预训练（冻结 FOPDT 于先验中心）----------
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
        for th, uh, ch, cur_t, cur_u, fut_t, fut_u, m, s, _ in train_loader:
            th, uh, ch, cur_t, cur_u, fut_t, fut_u, m, s = [x.to(device, non_blocking=True)
                                                            for x in (th, uh, ch, cur_t, cur_u,
                                                                      fut_t, fut_u, m, s)]
            opt1.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                hold_u = cur_u[:, None].expand(-1, cfg.horizon_steps)
                cov_in = ch if model.c_covar > 0 else None
                outv = model(th, uh, cur_u, hold_u, m, s, covar_hist=cov_in)
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

    if (out / "best_stage1.pt").exists():
        ckpt1 = torch.load(out / "best_stage1.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt1["state_dict"])
    cal_info = calibrate_on_train_subset(model, train_loader, device)
    if best_report is not None:
        best_report["stage1_calibration"] = cal_info
    (out / "stage1_report.json").write_text(
        json.dumps(best_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- Stage 2：全模型联合微调（含在途余热三分量 + LPV 先验锚定）----------
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
        for th, uh, ch, cur_t, cur_u, fut_t, fut_u, m, s, _ in train_loader:
            th, uh, ch, cur_t, cur_u, fut_t, fut_u, m, s = [x.to(device, non_blocking=True)
                                                            for x in (th, uh, ch, cur_t, cur_u,
                                                                      fut_t, fut_u, m, s)]
            opt2.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                cov_in = ch if model.c_covar > 0 else None
                outv = model(th, uh, cur_u, fut_u, m, s, covar_hist=cov_in)
                truth = fut_t - cur_t[:, None]
                lp = loss_fn(outv["temperature_delta_c"], truth, cur_t)
                loss = lp["total"] + 35.0 * outv["prior_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            tl += float(loss.detach()) * len(th); n += len(th)
        calibrate_on_train_subset(model, train_loader, device, max_batches=8)
        rep, arr = evaluate(model, val_loader, cfg, device, loss_fn, with_kpi=True)
        body_r = float((rep.get("kpi") or {}).get("body_p2p_ratio") or 0.0)
        sel_score = rep["weighted_15min_rmse_c"] + 15.0 * max(0.0, 0.85 - body_r)
        row = {"epoch": f"s2-{epoch}", "train_loss": tl / max(n, 1), **{k: v for k, v in rep.items()
                                                                         if k not in ("kpi", "virtual_step_check")},
               "kpi": rep.get("kpi"), "virtual_step_check": rep.get("virtual_step_check"),
               "selection_score": sel_score, "elapsed_s": time.time() - s2_start}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print("stage2", json.dumps(row, ensure_ascii=False), flush=True)
        if sel_score < best:
            best = sel_score; best_report = rep; stale = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "stage": 2}, out / "best.pt")
            np.savez_compressed(out / "validation_predictions.npz",
                                free=arr["free"], forced=arr["forced"],
                                hist_coal=arr["hist_coal"], disturb=arr["disturb"],
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
        "state": "completed",
        "version": "v4_covar" if args.use_covariates else "v4_in_flight",
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
    p.add_argument("--use-covariates", action="store_true", help="方向一：接入其余6个热工协变量")
    p.add_argument("--no-in-flight", action="store_true", help="禁用方向二120步在途余热分解")
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 但不可用")
    device = torch.device(args.device)
    report = train_one_fold(args, ForecastConfig(), device)
    print("FINAL", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
