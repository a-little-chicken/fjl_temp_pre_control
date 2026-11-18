#!/usr/bin/env python3
"""从 V3 已保存的开发验证预测中抽取 100 个不重叠窗口，生成肉眼复核图。

注意：V3 protocol 标注 test_inference=false；这些是开发验证窗口，不是独立测试集。
预测使用未来真实煤量 Oracle 条件，仅用于事后回放。
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "100样本肉眼评估"
SEED = 20261117
N = 100


def select_nonoverlap(times_ns: np.ndarray, n: int, seed: int) -> np.ndarray:
    """以固定随机种子抽样，并确保 15min 目标窗互不重叠（起点间隔至少 4 个窗口）。"""
    rng = np.random.default_rng(seed)
    # 时间顺序窗口起点相隔约 4min；隔 4 个索引，预测窗相隔约 16min。
    candidates = rng.permutation(len(times_ns))
    chosen: list[int] = []
    for idx in candidates:
        if all(abs(int(idx) - prev) >= 4 for prev in chosen):
            chosen.append(int(idx))
            if len(chosen) == n:
                break
    if len(chosen) != n:
        raise RuntimeError(f"只能抽到 {len(chosen)} 个互不重叠窗口")
    return np.array(sorted(chosen), dtype=np.int64)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "rmse_c": float(np.sqrt(np.mean(err ** 2))),
        "mae_c": float(np.mean(np.abs(err))),
        "endpoint_abs_error_c": float(abs(err[-1])),
        "p2p_true_c": float(np.ptp(y_true)),
        "p2p_pred_c": float(np.ptp(y_pred)),
        "p2p_ratio": float(np.ptp(y_pred) / max(np.ptp(y_true), 1e-6)),
        "bias_c": float(err.mean()),
    }


def draw_sample(ax_temp, ax_coal, sample_id, idx, arr, times):
    cur = float(arr["current"][idx])
    y_true = cur + arr["truth"][idx]
    y_pred = cur + arr["free"][idx]
    minutes = (np.arange(60) + 1) * 0.25
    mm = metrics(y_true, y_pred)
    ax_temp.plot(minutes, y_true, color="#222222", lw=1.5, label="Observed future")
    ax_temp.plot(minutes, y_pred, color="#1677c8", lw=1.5, label="V3 forecast (Oracle)")
    ax_temp.axhline(950, color="#d62728", ls="--", lw=0.8, alpha=0.75)
    ax_temp.axhline(880, color="#ff7f0e", ls="--", lw=0.8, alpha=0.75)
    ax_temp.axhline(cur, color="#777777", ls=":", lw=0.8)
    ax_temp.set_xlim(0, 15)
    ax_temp.grid(alpha=0.2)
    ts = pd.to_datetime(int(times[idx]))
    ax_temp.set_title(
        f"#{sample_id:03d} | {ts:%Y-%m-%d %H:%M} | RMSE {mm['rmse_c']:.1f} C | "
        f"P2P {mm['p2p_ratio']:.2f}", fontsize=9)
    ax_temp.set_ylabel("Temp (C)")
    ax_temp.tick_params(labelsize=8)
    ax_coal.plot(minutes, arr["future_coal"][idx], color="#8c564b", lw=1.1)
    ax_coal.set_xlim(0, 15)
    ax_coal.grid(alpha=0.2)
    ax_coal.set_ylabel("Coal (t/h)")
    ax_coal.set_xlabel("Forecast horizon (min)")
    ax_coal.tick_params(labelsize=8)
    return mm, ts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "validation_predictions.npz"
    arr = dict(np.load(path))
    n_all = len(arr["time"])
    if n_all != 3302:
        print(f"Note: expected 3302 validation windows; found {n_all}")
    ids = select_nonoverlap(arr["time"], N, SEED)
    records = []
    for rank, idx in enumerate(ids, 1):
        fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})
        mm, ts = draw_sample(a, b, rank, int(idx), arr, arr["time"])
        if rank == 1:
            a.legend(loc="upper right", fontsize=8)
        fig.suptitle("V3 development-validation visual audit | future actual coal = Oracle only",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"sample_{rank:03d}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        records.append({"sample_id": rank, "validation_index": int(idx),
                        "origin_time": ts.isoformat(), **mm})
    with (OUT / "samples.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

    # 10x10 overview for rapid scanning; detailed reading should use the individual PNG/PDF.
    fig, axes = plt.subplots(10, 10, figsize=(28, 24), sharex=True, sharey=True)
    for rank, (ax, idx) in enumerate(zip(axes.flat, ids), 1):
        cur = float(arr["current"][idx])
        yt = cur + arr["truth"][idx]
        yp = cur + arr["free"][idx]
        mins = (np.arange(60) + 1) * 0.25
        ax.plot(mins, yt, color="#222222", lw=0.65)
        ax.plot(mins, yp, color="#1677c8", lw=0.65)
        ax.axhline(950, color="#d62728", ls="--", lw=0.35)
        ax.axhline(880, color="#ff7f0e", ls="--", lw=0.35)
        ax.set_title(f"{rank:03d} | {np.sqrt(np.mean((yp-yt)**2)):.0f}C", fontsize=6)
        ax.set_xlim(0, 15); ax.grid(alpha=0.12)
        if rank % 10 == 1: ax.set_ylabel("C", fontsize=6)
        if rank > 90: ax.set_xlabel("min", fontsize=6)
        ax.tick_params(labelsize=5)
    fig.suptitle("V3: 100 non-overlapping development-validation windows\n"
                 "black=observed future; blue=forecast; dashed=950/880 C; Oracle coal input",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "overview_100.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # 10 pages, 10 samples/page, easier to inspect at readable scale.
    with PdfPages(OUT / "100_samples.pdf") as pdf:
        for page in range(10):
            subset = ids[page * 10:(page + 1) * 10]
            fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=True)
            for j, idx in enumerate(subset):
                ax_t = axes.flat[j]
                ax_c = ax_t.twinx()
                rank = page * 10 + j + 1
                cur = float(arr["current"][idx])
                yt = cur + arr["truth"][idx]
                yp = cur + arr["free"][idx]
                mins = (np.arange(60) + 1) * 0.25
                ax_t.plot(mins, yt, color="#222222", lw=1.1, label="Observed")
                ax_t.plot(mins, yp, color="#1677c8", lw=1.1, label="V3 forecast")
                ax_t.axhline(950, color="#d62728", ls="--", lw=0.65)
                ax_t.axhline(880, color="#ff7f0e", ls="--", lw=0.65)
                ax_t.set_xlim(0, 15); ax_t.grid(alpha=0.18)
                ax_t.set_ylabel("Temp C", fontsize=8)
                ax_c.plot(mins, arr["future_coal"][idx], color="#8c564b", lw=0.7, alpha=0.65)
                ax_c.set_ylabel("Coal t/h", fontsize=8)
                mm = metrics(yt, yp)
                ts = pd.to_datetime(int(arr["time"][idx]))
                ax_t.set_title(f"#{rank:03d} {ts:%m-%d %H:%M} | RMSE={mm['rmse_c']:.1f}C "
                               f"P2P ratio={mm['p2p_ratio']:.2f}", fontsize=9)
                if j == 0:
                    ax_t.legend(loc="upper left", fontsize=7)
                if j >= 8: ax_t.set_xlabel("Horizon (min)", fontsize=8)
            fig.suptitle(f"V3 validation visual review | page {page+1}/10 | Oracle-only future coal",
                         fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)

    rmse = np.array([r["rmse_c"] for r in records])
    p2p = np.array([r["p2p_ratio"] for r in records])
    bias = np.array([r["bias_c"] for r in records])
    summary = {
        "population": "3302 V3 development-validation windows, not an independent test set",
        "selection": f"fixed-seed random ({SEED}), 100 origins; at least 16 min between selected windows",
        "prediction_contract": "future actual coal is Oracle-only; not online-deployable performance",
        "n": N,
        "sample_rmse_median_c": float(np.median(rmse)),
        "sample_rmse_p10_p90_c": [float(np.quantile(rmse, .1)), float(np.quantile(rmse, .9))],
        "sample_rmse_min_max_c": [float(rmse.min()), float(rmse.max())],
        "p2p_ratio_median": float(np.median(p2p)),
        "p2p_ratio_in_0p85_1p15_fraction": float(np.mean((p2p >= .85) & (p2p <= 1.15))),
        "window_bias_median_c": float(np.median(bias)),
        "window_bias_abs_p90_c": float(np.quantile(np.abs(bias), .9)),
        "artifacts": ["overview_100.png", "100_samples.pdf", "samples.csv", "sample_001.png ... sample_100.png"],
    }
    import json
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
