#!/usr/bin/env python3
"""V4 in-flight 100-window visual review, paired to the exact V3 sample origins.

V4 artifact labels the population as development validation (not an independent test set).
Forecast includes future actual coal as Oracle input, so this is retrospective conditional playback.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
V3 = ROOT.parent / "sa208_control_forecast_v3"
OUT = ROOT / "100样本肉眼评估"


def metric(y, p):
    e = p - y
    return {"rmse_c": float(np.sqrt(np.mean(e * e))),
            "mae_c": float(np.abs(e).mean()),
            "endpoint_abs_error_c": float(abs(e[-1])),
            "p2p_true_c": float(np.ptp(y)),
            "p2p_pred_c": float(np.ptp(p)),
            "p2p_ratio": float(np.ptp(p) / max(np.ptp(y), 1e-6)),
            "bias_c": float(e.mean())}


def plot_one(axs, sample_id, idx, a4, a3, ts, hist_scale, coupling):
    ax_t, ax_c, ax_u = axs
    cur = float(a4["current"][idx])
    y = cur + a4["truth"][idx]
    p4 = cur + a4["free"][idx]
    p3 = cur + a3["free"][idx]
    minx = (np.arange(60) + 1) * 0.25
    m4 = metric(y, p4); m3 = metric(y, p3)
    ax_t.plot(minx, y, color="#222222", lw=1.45, label="Observed future")
    ax_t.plot(minx, p4, color="#1677c8", lw=1.4, label="V4 in-flight")
    ax_t.plot(minx, p3, color="#2ca02c", lw=1.0, ls="--", alpha=.8, label="V3 paired ref")
    ax_t.axhline(950, color="#d62728", ls=":", lw=.8)
    ax_t.axhline(880, color="#ff7f0e", ls=":", lw=.8)
    ax_t.axhline(cur, color="#777777", ls="-.", lw=.65)
    ax_t.set_xlim(0, 15); ax_t.grid(alpha=.2)
    dt = pd.to_datetime(int(ts[idx]))
    ax_t.set_title(f"#{sample_id:03d} {dt:%Y-%m-%d %H:%M} | V4 RMSE {m4['rmse_c']:.1f}C "
                   f"(V3 {m3['rmse_c']:.1f}C) | V4 P2P {m4['p2p_ratio']:.2f}", fontsize=9)
    ax_t.set_ylabel("Temp (C)", fontsize=8)
    # Components in delta-temperature domain. Components should sum approximately to V4 ΔT.
    ax_c.plot(minx, a4["disturb"][idx], color="#9467bd", lw=1.0, label="Disturbance ΔT")
    ax_c.plot(minx, hist_scale * a4["hist_coal"][idx], color="#8c564b", lw=1.0,
              label="Scaled in-flight coal ΔT")
    ax_c.plot(minx, coupling * a4["forced"][idx], color="#e377c2", lw=1.0,
              label="Coupled future-coal ΔT (Oracle)")
    ax_c.plot(minx, a4["free"][idx], color="#1677c8", lw=.8, ls="--", label="V4 total ΔT")
    ax_c.axhline(0, color="#777777", lw=.6)
    ax_c.set_xlim(0, 15); ax_c.grid(alpha=.18)
    ax_c.set_ylabel("ΔT (C)", fontsize=8)
    ax_u.plot(minx, a4["future_coal"][idx], color="#8c564b", lw=1.05)
    ax_u.set_xlim(0, 15); ax_u.grid(alpha=.18)
    ax_u.set_ylabel("Coal (t/h)", fontsize=8)
    ax_u.set_xlabel("Forecast horizon (min)", fontsize=8)
    for ax in axs:
        ax.tick_params(labelsize=7)
    return m4, m3, dt


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a4 = dict(np.load(ROOT / "validation_predictions.npz"))
    a3 = dict(np.load(V3 / "validation_predictions.npz"))
    if not np.array_equal(a4["time"], a3["time"]):
        raise ValueError("V3/V4 验证起点不一致，禁止直接配对")
    # Reuse the V3 draw, making the visual comparison paired rather than cherry-picking two samples.
    chosen = pd.read_csv(V3 / "100样本肉眼评估" / "samples.csv")
    ids = chosen["validation_index"].to_numpy(dtype=np.int64)
    if len(ids) != 100:
        raise ValueError(f"V3 sample list has {len(ids)} rows, expected 100")
    if not np.array_equal(chosen["sample_id"].to_numpy(), np.arange(1, 101)):
        raise ValueError("sample IDs are not canonical 1..100")
    met = json.loads((ROOT / "validation_metrics.json").read_text(encoding="utf-8"))
    hist_scale = float(met["hist_coal_scale"])
    coupling = float(met["closed_loop_coupling"])
    records = []
    for sample_id, idx in enumerate(ids, 1):
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                                gridspec_kw={"height_ratios": [2.2, 1.4, 1]})
        m4, m3, dt = plot_one(axs, sample_id, int(idx), a4, a3, a4["time"], hist_scale, coupling)
        if sample_id == 1:
            axs[0].legend(loc="upper right", fontsize=7, ncol=2)
            axs[1].legend(loc="upper right", fontsize=6, ncol=2)
        fig.suptitle("V4 in-flight visual audit | same origins as V3 | future actual coal = Oracle only",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"sample_{sample_id:03d}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        records.append({"sample_id": sample_id, "validation_index": int(idx),
                        "origin_time": dt.isoformat(),
                        **{f"v4_{k}": v for k, v in m4.items()},
                        **{f"v3_{k}": v for k, v in m3.items()},
                        "rmse_improved_vs_v3": m4["rmse_c"] < m3["rmse_c"]})

    with (OUT / "samples_paired.csv").open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(records[0]))
        wr.writeheader(); wr.writerows(records)

    # Overview: same 100 origins, truth / paired V3 / V4.
    fig, axes = plt.subplots(10, 10, figsize=(28, 24), sharex=True, sharey=True)
    for sid, (ax, idx) in enumerate(zip(axes.flat, ids), 1):
        cur = float(a4["current"][idx]); mn = (np.arange(60) + 1) * .25
        y = cur + a4["truth"][idx]
        p4 = cur + a4["free"][idx]; p3 = cur + a3["free"][idx]
        ax.plot(mn, y, c="#222222", lw=.65)
        ax.plot(mn, p4, c="#1677c8", lw=.65)
        ax.plot(mn, p3, c="#2ca02c", lw=.5, ls="--", alpha=.8)
        ax.axhline(950, c="#d62728", ls=":", lw=.3)
        ax.axhline(880, c="#ff7f0e", ls=":", lw=.3)
        ax.set_title(f"{sid:03d} V4 {metric(y,p4)['rmse_c']:.0f}C / V3 {metric(y,p3)['rmse_c']:.0f}C",
                     fontsize=5.7)
        ax.set_xlim(0, 15); ax.grid(alpha=.1); ax.tick_params(labelsize=5)
    fig.suptitle("100 paired validation windows | black=truth, blue=V4, green dashed=V3",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(OUT / "overview_100_paired.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # 10-page PDF, 10 samples per page.
    with PdfPages(OUT / "100_samples_paired.pdf") as pdf:
        for page in range(10):
            fig, axes = plt.subplots(5, 2, figsize=(14, 21))
            for j, idx in enumerate(ids[page * 10:(page + 1) * 10]):
                sid = page * 10 + j + 1
                # Each cell uses 3 small stacked axes: temp, decomposition, coal.
                cell = axes.flat[j]
                cell.remove()
                spec = cell.get_subplotspec().subgridspec(3, 1, height_ratios=[2.0, 1.1, .8], hspace=.08)
                a, b, c = [fig.add_subplot(spec[k]) for k in range(3)]
                m4, m3, dt = plot_one([a, b, c], sid, int(idx), a4, a3, a4["time"], hist_scale, coupling)
                if j == 0:
                    a.legend(loc="upper right", fontsize=6, ncol=2)
                    b.legend(loc="upper right", fontsize=5, ncol=2)
            fig.suptitle(f"V4 in-flight paired visual review | page {page+1}/10 | Oracle future coal",
                         fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, .97))
            pdf.savefig(fig); plt.close(fig)

    v4rm = np.array([r["v4_rmse_c"] for r in records])
    v3rm = np.array([r["v3_rmse_c"] for r in records])
    ratios = np.array([r["v4_p2p_ratio"] for r in records])
    summary = {
        "population": "3302 V4 development-validation windows; not independent test set",
        "selection": "exact same 100 non-overlapping validation origins as V3 visual review",
        "prediction_contract": "future actual coal is Oracle-only; no online claim",
        "n": len(ids),
        "v4_sample_rmse_median_c": float(np.median(v4rm)),
        "v4_sample_rmse_p10_p90_c": [float(np.quantile(v4rm,.1)), float(np.quantile(v4rm,.9))],
        "v4_sample_rmse_min_max_c": [float(v4rm.min()), float(v4rm.max())],
        "v3_paired_sample_rmse_median_c": float(np.median(v3rm)),
        "v4_rmse_better_count": int(np.sum(v4rm < v3rm)),
        "v4_rmse_worse_count": int(np.sum(v4rm > v3rm)),
        "paired_median_rmse_change_v4_minus_v3_c": float(np.median(v4rm-v3rm)),
        "v4_p2p_ratio_median": float(np.median(ratios)),
        "v4_p2p_ratio_in_0p85_1p15_fraction": float(np.mean((ratios>=.85)&(ratios<=1.15))),
        "artifacts": ["overview_100_paired.png", "100_samples_paired.pdf", "samples_paired.csv",
                      "sample_001.png ... sample_100.png"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
