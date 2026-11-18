#!/usr/bin/env python3
"""v2 控制导向预测训练（sa-208 离线，只读 CSV；不访问 IOT、不下发动作）。

阶段一：冻结自由响应分支，仅在历史闭环样本上优化 LPV 参数调制与全局增益；
  这是有监督的闭环拟合，不是独立动作辨识，也不构成因果标定。
阶段二：联合微调自由响应分支与 LPV 参数调制，Loss 使用门控振幅匹配。
所有验证指标以未来真实煤量作为 Oracle 条件输入，并另报“当前煤量保持”历史标签诊断；
二者均不能代表部署性能或有效反事实。

输出评估：加权轨迹 RMSE、端点 RMSE、P90、时距偏差、峰谷/相位/振幅还原 KPI，
以及 KPI-4 虚拟 ±2t/h 阶跃响应检查（仅诊断）。残差臂 Oracle 评价保持
“未来真实煤量仅事后使用”的边界。
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
    build_temp_features, build_coal_features,
    coal_recent_stats, kpi_metrics, virtual_step_check,
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
    return {
        "weighted_15min_rmse_c": float(np.sqrt(np.mean(np.sum(w * err**2, 1) / w.sum()))),
        "endpoint_rmse_c": float(np.sqrt((err[:, -1]**2).mean())),
        "all_point_p90_abs_error_c": float(np.quantile(np.abs(err), 0.9)),
        "max_abs_horizon_mean_bias_c": float(np.abs(err.mean(0)).max()),
        "mean_abs_error_c": float(np.abs(err).mean()),
    }


@torch.no_grad()
def evaluate(model, loader, cfg, device, loss_fn, with_kpi: bool):
    model.eval()
    acc = {k: [] for k in ("free", "truth", "current", "future_coal", "time", "hold")}
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
                                    smooth_weight=args.smooth_weight).to(device)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "version": "v2", "seed": SEED,
        "input": "past60min 温度相空间(3ch) + 尾煤/包络(5ch)；推理时另有候选未来煤量",
        "output": "future 15min 温度轨迹（增量锚定 T_t + ΔT）",
        "architecture": "dual-stream PatchTST(shared encoder)+learned recency bias + Cross-Attention + LPV-FOPDT", 
        "validation_action_contract": "future actual coal is Oracle-only; hold-current is historical-label diagnostic, not counterfactual", 
        "stage1_interpretation": "supervised closed-loop fitting, not independent action identification or causal calibration",
        "fopdt_prior": {"K_base": cfg.gain_c_per_tph, "tau_base_s": cfg.tau_seconds,
                        "dead_steps": cfg.dead_steps, "K_range": [cfg.gain_low, cfg.gain_high],
                        "tau_range_s": [cfg.tau_low, cfg.tau_high],
                        "calibration_range": [cfg.calibration_low, cfg.calibration_high],
                        "note": "工程先验+闭环标定，非因果辨识"},
        "loss": {"far_weight": "0.5->2.0 quadratic normalized", "peak_thresholds_c": [880, 950],
                 "event_weight": args.event_weight, "amp_weight_gated": args.amp_weight,
                 "calm_p2p_gate_c": args.calm_p2p, "trend_weight": args.trend_weight,
                 "smooth_weight": args.smooth_weight},
        "fold": "train target end < 2026-10-01; validation [2026-10-01 01:00, 2026-10-15)",
        "open_kiln_filter": "temp>800 & sat>800，完整连续窗口",
        "config": cfg.__dict__, "channels": {"temp": C_TEMP, "coal": C_COAL},
        "split_info": split_info, "batch_size": args.batch_size,
        "max_epochs": args.epochs, "stage1_epochs": args.stage1_epochs,
        "test_inference": False, "IOT_access": False,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2),
                                       encoding="utf-8")



    # ---------- Stage 1：物理锚定（冻结自由响应，只调 LPV 标定+调制）----------
    stage1_params = list(model.coal_response.parameters())
    opt1 = torch.optim.AdamW(stage1_params, lr=args.lr * 2.0)
    best = float("inf"); best_report = None; stale = 0; started = time.time()
    history = []
    for epoch in range(1, args.stage1_epochs + 1):
        model.train()
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        model.query.requires_grad_(False)
        for p in list(model.cross_temp.parameters()) + list(model.cross_coal.parameters()) \
                + list(model.norm.parameters()) + list(model.head.parameters()):
            p.requires_grad_(False)
        tl, n = 0.0, 0
        for th, uh, cur_t, cur_u, fut_t, fut_u, m, s, _ in train_loader:
            th, uh, cur_t, cur_u, fut_t, fut_u, m, s = [x.to(device, non_blocking=True)
                                                        for x in (th, uh, cur_t, cur_u,
                                                                  fut_t, fut_u, m, s)]
            opt1.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                outv = model(th, uh, cur_u, fut_u, m, s)
                truth = fut_t - cur_t[:, None]
                # 阶段一目标：让 free+forced 贴合真实 ΔT，同时强先验锚定 K
                lp = loss_fn(outv["temperature_delta_c"], truth, cur_t)
                loss = lp["far_huber"] + 2.0 * lp["amplitude"] \
                    + 5.0 * outv["prior_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(stage1_params, 1.0)
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
    (out / "stage1_report.json").write_text(
        json.dumps(best_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- Stage 2：联合微调（全参数，门控振幅 Loss + 弱先验锚定）----------
    for p in model.parameters():
        p.requires_grad_(True)
    opt2 = torch.optim.AdamW(model.parameters(), lr=args.lr)
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
                loss = lp["total"] + 1.0 * outv["prior_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            tl += float(loss.detach()) * len(th); n += len(th)
        rep, arr = evaluate(model, val_loader, cfg, device, loss_fn, with_kpi=True)
        row = {"epoch": f"s2-{epoch}", "train_loss": tl / max(n, 1), **{k: v for k, v in rep.items()
                                                                         if k not in ("kpi", "virtual_step_check")},
               "kpi": rep.get("kpi"), "elapsed_s": time.time() - s2_start}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print("stage2", json.dumps(row, ensure_ascii=False), flush=True)
        if rep["weighted_15min_rmse_c"] < best:
            best = rep["weighted_15min_rmse_c"]; best_report = rep; stale = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "stage": 2}, out / "best.pt")
            np.savez_compressed(out / "validation_predictions.npz",
                                free=arr["free"], truth=arr["truth"],
                                current=arr["current"], future_coal=arr["future_coal"],
                                time=arr["time"])
        else:
            stale += 1
        if epoch >= args.min_epochs and stale >= args.patience:
            break

    (out / "validation_metrics.json").write_text(
        json.dumps(best_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "status.json").write_text(json.dumps({
        "state": "completed", "version": "v2",
        "best_validation_weighted_rmse_c": best,
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
    p.add_argument("--amp-weight", type=float, default=0.5)
    p.add_argument("--calm-p2p", type=float, default=12.0)
    p.add_argument("--trend-weight", type=float, default=0.1)
    p.add_argument("--smooth-weight", type=float, default=0.02)
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 但不可用")
    device = torch.device(args.device)
    report = train_one_fold(args, ForecastConfig(), device)
    print("FINAL", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
