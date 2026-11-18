#!/usr/bin/env python3
"""控制导向轨迹预测：复合损失函数与控制准入 KPI 评估模块。

代码结构说明：
- Section 1: `ControlTrajectoryLoss`（五项正交轨迹损失 + 低频 FFT 能量谱一致性约束）
- Section 2: `base_metrics`（基础点误差、超越静态均值回归基线的净增益 Skill Score、分时距 RMSE）
- Section 3: `kpi_metrics` 与 `virtual_step_check`（波峰波谷误差、一对一拐点相位匹配、实时阶跃响应自检）
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ===========================================================================
# Section 1: 瘦身正交化控制导向复合损失函数 (ControlTrajectoryLoss)
# ===========================================================================

class ControlTrajectoryLoss(nn.Module):
    """控制导向正交复合轨迹损失函数：
    1. `far_huber`  : 远期二次递增权重 Smooth L1 损失；
    2. `event_huber`: 触及高温(>=950℃)/低温(<=880℃)工艺红线时的二次方重罚；
    3. `corr`       : 去均值轨迹皮尔逊相关损失（1 - r），对齐波形凹凸走势与峰谷相位；
    4. `trend`      : 四级多尺度一阶差分斜率匹配（15s, 1min, 3min, 6min）+ 单步最大温变速率势垒；
    5. `amplitude`  : 相位门控软极值跨度 + 低频 FFT 幅值谱匹配 + 平静期假峰抑制；
    6. `mean_bias`  : 窗口内时间均值与批次跨样本均值双重零偏约束。
    """

    def __init__(
        self,
        horizon: int = 60,
        high_threshold_c: float = 950.0,
        low_threshold_c: float = 880.0,
        event_weight: float = 1.0,
        amp_weight: float = 0.5,
        calm_p2p_c: float = 12.0,
        trend_weight: float = 0.45,
        corr_weight: float = 0.85,
        mean_bias_weight: float = 0.5,
        huber_delta_c: float = 5.0,
        max_step_c: float = 6.0,
    ):
        super().__init__()
        self.high = high_threshold_c
        self.low = low_threshold_c
        self.event_weight = event_weight
        self.amp_weight = amp_weight
        self.calm_p2p_c = calm_p2p_c
        self.trend_weight = trend_weight
        self.corr_weight = corr_weight
        self.mean_bias_weight = mean_bias_weight
        self.huber_delta = huber_delta_c
        self.max_step_c = max_step_c

        x = torch.linspace(0.0, 1.0, horizon)
        weights = 0.5 + 1.5 * x.square()
        self.register_buffer("horizon_weights", weights / weights.mean())

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        current_temperature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if prediction.shape != target.shape or prediction.ndim != 2:
            raise ValueError("prediction/target 必须同形 [batch, horizon]")
        if prediction.shape[1] != self.horizon_weights.numel():
            raise ValueError("horizon 与 Loss 配置不一致")
        if current_temperature.ndim == 1:
            current_temperature = current_temperature[:, None]

        abs_pred = current_temperature + prediction
        abs_true = current_temperature + target
        h = prediction.shape[1]

        # (1) 远期加权 Huber 损失
        w = self.horizon_weights.to(prediction.dtype)[None, :]
        point = F.smooth_l1_loss(prediction, target, reduction="none", beta=self.huber_delta)
        base = (point * w).mean()

        # (2) 工艺红线越界二次方重罚
        extreme = ((abs_true >= self.high) | (abs_true <= self.low)).to(point.dtype)
        sq_event = (prediction - target).square() / max(self.huber_delta, 1.0)
        event = (sq_event * extreme * w).sum() / (extreme.sum() + 1e-6)

        # (3) 去均值波形皮尔逊相关损失
        p2p_t = abs_true.max(dim=1).values - abs_true.min(dim=1).values
        pred_c = prediction - prediction.mean(dim=1, keepdim=True)
        targ_c = target - target.mean(dim=1, keepdim=True)
        cov = (pred_c * targ_c).sum(dim=1)
        std_p = torch.sqrt(pred_c.square().sum(dim=1) + 1e-3)
        std_t = torch.sqrt(targ_c.square().sum(dim=1) + 1e-3)
        pearson_r = (cov / (std_p * std_t)).clamp(-1.0, 1.0)
        wave_gate = (p2p_t > 6.0).to(prediction.dtype)
        corr_loss = ((1.0 - pearson_r) * wave_gate).sum() / (wave_gate.sum() + 1e-6)

        # (4) 四级多尺度一阶差分斜率匹配 (15s, 1min, 3min, 6min) + 起点单步速率势垒
        pred_pad = F.pad(prediction, (1, 0), value=0.0)
        targ_pad = F.pad(target, (1, 0), value=0.0)
        dp1 = pred_pad[:, 1:] - pred_pad[:, :-1]
        roc_barrier = F.relu(dp1.abs() - self.max_step_c).square().mean()

        ms_trend = torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
        for stride, w_s in ((1, 0.20), (4, 0.30), (12, 0.25), (24, 0.25)):
            if h >= stride:
                dp_s = pred_pad[:, stride:] - pred_pad[:, :-stride]
                dy_s = targ_pad[:, stride:] - targ_pad[:, :-stride]
                ms_trend = ms_trend + w_s * F.smooth_l1_loss(
                    dp_s, dy_s, reduction="mean", beta=max(1.0, 0.5 * stride)
                )
        trend = ms_trend + 0.5 * roc_barrier

        # (5) 相位对齐振幅匹配 + 主频段 FFT 能量谱一致性 + 平静期假峰抑制
        gate = (p2p_t > self.calm_p2p_c).to(prediction.dtype)
        active = gate.sum()
        if h > 4:
            body_p2p_p = abs_pred[:, 3:].max(dim=1).values - abs_pred[:, 3:].min(dim=1).values
            body_p2p_t = abs_true[:, 3:].max(dim=1).values - abs_true[:, 3:].min(dim=1).values
        else:
            body_p2p_p = abs_pred.max(dim=1).values - abs_pred.min(dim=1).values
            body_p2p_t = p2p_t

        k_ext = max(2, min(6, h // 6))
        span_t = (
            abs_true.topk(k_ext, dim=1).values.mean(dim=1)
            + (-abs_true).topk(k_ext, dim=1).values.mean(dim=1)
        )
        span_p = (
            abs_pred.topk(k_ext, dim=1).values.mean(dim=1)
            + (-abs_pred).topk(k_ext, dim=1).values.mean(dim=1)
        )
        w_pk = F.softmax(target / 5.0, dim=1).detach()
        w_vv = F.softmax(-target / 5.0, dim=1).detach()
        aligned_t = (w_pk * target).sum(dim=1) - (w_vv * target).sum(dim=1)
        aligned_p = (w_pk * prediction).sum(dim=1) - (w_vv * prediction).sum(dim=1)

        # 低频主频段 (k = 1..min(6, H//2)) FFT 幅值谱匹配，锁死 5~15min 远期交流振荡能量
        max_k = min(7, h // 2 + 1)
        if max_k > 1:
            fft_p = torch.fft.rfft(pred_c.float(), dim=1)[:, 1:max_k].abs()
            fft_t = torch.fft.rfft(targ_c.float(), dim=1)[:, 1:max_k].abs()
            fft_amp_err = ((fft_p - fft_t).abs().mean(dim=1) * (2.0 / max(h, 1))).to(
                prediction.dtype
            )
        else:
            fft_amp_err = torch.zeros_like(aligned_t)

        phase_gate = pearson_r.detach().clamp(min=0.0, max=1.0)
        amp_per_win = (
            0.65 * (aligned_p - aligned_t).abs()
            + 0.40 * phase_gate * (body_p2p_p - body_p2p_t).abs()
            + 0.25 * phase_gate * (span_p - span_t).abs()
            + 0.35 * phase_gate * (abs_pred.std(dim=1) - abs_true.std(dim=1)).abs()
            + 0.35 * phase_gate * fft_amp_err
        )
        calm_gate = 1.0 - gate
        calm_overshoot = F.relu(body_p2p_p - p2p_t - 2.5).square()
        calm_suppress = (calm_overshoot * calm_gate).sum() / (calm_gate.sum() + 1e-6)
        amp = (amp_per_win * gate).sum() / (active + 1e-6) + 0.25 * calm_suppress

        # (6) 双维均值零偏锚定
        win_bias = (prediction.mean(dim=1) - target.mean(dim=1)).square().mean()
        step_bias = (prediction.mean(dim=0) - target.mean(dim=0)).square().mean()
        mean_bias = 0.5 * win_bias + 0.5 * step_bias

        total = (
            base
            + self.event_weight * event
            + self.corr_weight * corr_loss
            + self.trend_weight * trend
            + self.amp_weight * amp
            + self.mean_bias_weight * mean_bias
        )
        return {
            "total": total,
            "far_huber": base,
            "event_huber": event,
            "corr": corr_loss,
            "trend": trend,
            "amplitude": amp,
            "mean_bias": mean_bias,
            "mean_pearson_r": pearson_r.mean(),
            "amp_active_fraction": active / max(prediction.shape[0], 1),
        }


# ===========================================================================
# Section 2: 基础预测误差与超越静态均值回归的净增益指标 (base_metrics)
# ===========================================================================

def base_metrics(
    target: np.ndarray,
    pred: np.ndarray,
    cur: np.ndarray,
    weights: np.ndarray,
    ref_mean_temp_c: float = 900.5657,
) -> dict[str, float]:
    """计算基础点预测误差、分时距 RMSE 以及相对于一阶静态均值回归基线的 Skill Score。"""
    err = pred - target
    w = weights[None, :]
    horizon_bias = err.mean(0)
    step_rmse = np.sqrt((err ** 2).mean(axis=0))
    weighted_rmse = float(np.sqrt(np.mean(np.sum(w * err ** 2, 1) / w.sum())))

    # 静态均值回归参考基线: ΔT_mr(h) = (T_mean - T_t) * (1 - exp(-(h+1)/15))
    h = target.shape[1]
    decay = 1.0 - np.exp(-np.arange(1, h + 1, dtype=np.float64) / 15.0)[None, :]
    mr_pred = (ref_mean_temp_c - cur[:, None]) * decay
    mr_err = mr_pred - target
    mr_weighted_rmse = float(np.sqrt(np.mean(np.sum(w * mr_err ** 2, 1) / w.sum())))
    skill_vs_mr = float(1.0 - (weighted_rmse / max(mr_weighted_rmse, 1e-6)) ** 2)

    sig_end = np.abs(target[:, -1]) > 3.0
    end_dir_acc = (
        float((np.sign(pred[sig_end, -1]) == np.sign(target[sig_end, -1])).mean())
        if sig_end.any()
        else 1.0
    )

    out = {
        "weighted_15min_rmse_c": weighted_rmse,
        "mr_baseline_weighted_rmse_c": mr_weighted_rmse,
        "skill_score_vs_mr": skill_vs_mr,
        "endpoint_rmse_c": float(step_rmse[-1]),
        "endpoint_direction_accuracy": end_dir_acc,
        "all_point_p90_abs_error_c": float(np.quantile(np.abs(err), 0.9)),
        "max_abs_horizon_mean_bias_c": float(np.abs(horizon_bias).max()),
        "step0_mean_bias_c": float(horizon_bias[0]),
        "step0_rmse_c": float(step_rmse[0]),
        "per_window_mean_mae_c": float(np.abs(err.mean(1)).mean()),
        "mean_abs_error_c": float(np.abs(err).mean()),
    }
    for label, idx in (("3m", 11), ("5m", 19), ("10m", 39), ("15m", h - 1)):
        if idx < h:
            out[f"rmse_{label}_c"] = float(step_rmse[idx])
    return out


# ===========================================================================
# Section 3: 控制准入 KPI 评价体系与实时阶跃响应自检
# ===========================================================================

def find_turns(signs: np.ndarray, min_gap: int = 4) -> np.ndarray:
    """从一阶差分符号序列中提取满足最小间隔 min_gap 的局部拐点索引。"""
    idx = []
    last = -(10 ** 9)
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1] and signs[i] != 0 and signs[i - 1] != 0:
            if i - last >= min_gap:
                idx.append(i)
                last = i
    return np.array(idx, dtype=int)


def match_turns_one_to_one(
    pred_signs: np.ndarray,
    true_signs: np.ndarray,
    min_gap: int = 4,
    max_tol_steps: int = 8,
) -> tuple[list[int], int, int]:
    """同极性一对一不放回拐点匹配。"""
    ta = find_turns(pred_signs, min_gap=min_gap)
    tb = find_turns(true_signs, min_gap=min_gap)
    if tb.size == 0 or ta.size == 0:
        return [], int(tb.size), int(ta.size)
    pol_a = pred_signs[ta - 1]
    pol_b = true_signs[tb - 1]
    candidates = []
    for bi, (t_true, pb) in enumerate(zip(tb, pol_b)):
        for ai, (t_pred, pa) in enumerate(zip(ta, pol_a)):
            if pa == pb:
                dist = abs(int(t_pred) - int(t_true))
                if dist <= max_tol_steps:
                    candidates.append((dist, bi, ai))
    candidates.sort(key=lambda x: x[0])
    used_b, used_a = set(), set()
    errs = []
    for dist, bi, ai in candidates:
        if bi not in used_b and ai not in used_a:
            used_b.add(bi)
            used_a.add(ai)
            errs.append(dist)
    return errs, int(tb.size), int(ta.size)


def kpi_metrics(
    pred_abs: np.ndarray,
    true_abs: np.ndarray,
    high_c: float = 950.0,
    low_c: float = 880.0,
    max_turn_tol_steps: int = 8,
) -> dict:
    """计算工业现场级控制准入 KPI（极值 MAE、红线 MAE、一对一拐点相位、振幅还原比、去均值相关系数）。"""
    if isinstance(pred_abs, torch.Tensor):
        pred_abs = pred_abs.detach().cpu().numpy()
    if isinstance(true_abs, torch.Tensor):
        true_abs = true_abs.detach().cpu().numpy()
    pred_abs = np.asarray(pred_abs, dtype=np.float64)
    true_abs = np.asarray(true_abs, dtype=np.float64)
    out = {}

    pk_p, pk_t = pred_abs.max(axis=1), true_abs.max(axis=1)
    vv_p, vv_t = pred_abs.min(axis=1), true_abs.min(axis=1)
    out["peak_valley_mae_c"] = float(
        np.concatenate([np.abs(pk_p - pk_t), np.abs(vv_p - vv_t)]).mean()
    )
    hi = np.where(true_abs >= high_c)
    lo = np.where(true_abs <= low_c)
    both_t = np.concatenate([true_abs[hi], true_abs[lo]])
    both_p = np.concatenate([pred_abs[hi], pred_abs[lo]])
    out["redline_mae_c"] = float(np.abs(both_p - both_t).mean()) if both_t.size else None
    out["redline_points"] = int(both_t.size)

    try:
        from scipy.signal import savgol_filter

        w_len = min(5, pred_abs.shape[1] - 1)
        if w_len % 2 == 0:
            w_len -= 1
        if w_len >= 3:
            sp = savgol_filter(np.diff(pred_abs, axis=1), w_len, min(2, w_len - 1), axis=1)
            st = savgol_filter(np.diff(true_abs, axis=1), w_len, min(2, w_len - 1), axis=1)
        else:
            sp, st = np.diff(pred_abs, axis=1), np.diff(true_abs, axis=1)
    except Exception:
        sp, st = np.diff(pred_abs, axis=1), np.diff(true_abs, axis=1)

    sp = np.where(np.abs(sp) > 1e-2, np.sign(sp), 0)
    st = np.where(np.abs(st) > 1e-2, np.sign(st), 0)
    tp_tt, tp_pp = 0, 0
    errs = []
    for a, b in zip(sp, st):
        e_win, nb, na = match_turns_one_to_one(a, b, min_gap=4, max_tol_steps=max_turn_tol_steps)
        tp_tt += nb
        tp_pp += na
        errs.extend(e_win)
    out["turn_time_mae_steps"] = float(np.mean(errs)) if errs else None
    out["turn_within_2steps_fraction"] = float(np.mean(np.asarray(errs) <= 2)) if errs else None
    out["true_window_turns"] = int(tp_tt)
    out["pred_window_turns"] = int(tp_pp)
    out["turn_miss_fraction"] = 1.0 - min(1.0, (len(errs) / max(tp_tt, 1))) if tp_tt else None

    p2p_t = np.ptp(true_abs, axis=1)
    p2p_p = np.ptp(pred_abs, axis=1)
    out["p2p_ratio"] = float(p2p_p.mean() / p2p_t.mean()) if p2p_t.mean() > 1e-6 else None
    out["mean_truth_p2p_c"] = float(p2p_t.mean())
    out["mean_pred_p2p_c"] = float(p2p_p.mean())
    if pred_abs.shape[1] > 4:
        p2p_t_body = np.ptp(true_abs[:, 3:], axis=1)
        p2p_p_body = np.ptp(pred_abs[:, 3:], axis=1)
        out["body_p2p_ratio"] = float(p2p_p_body.mean() / max(p2p_t_body.mean(), 1e-6))

    pred_c = pred_abs - pred_abs.mean(axis=1, keepdims=True)
    true_c = true_abs - true_abs.mean(axis=1, keepdims=True)
    denom = np.sqrt((pred_c ** 2).sum(axis=1) * (true_c ** 2).sum(axis=1)) + 1e-6
    r_all = (pred_c * true_c).sum(axis=1) / denom
    active_mask = p2p_t > 6.0
    out["mean_pearson_r"] = (
        float(r_all[active_mask].mean()) if active_mask.any() else float(r_all.mean())
    )
    out["positive_corr_fraction"] = (
        float((r_all[active_mask] > 0.0).mean())
        if active_mask.any()
        else float((r_all > 0.0).mean())
    )

    h = pred_abs.shape[1]
    anc = [idx for idx in (11, 23, 39, h - 1) if 0 < idx < h]
    dy_anc = true_abs[:, anc] - true_abs[:, 0:1]
    dp_anc = pred_abs[:, anc] - pred_abs[:, 0:1]
    sig = np.abs(dy_anc) > 3.0
    out["direction_accuracy_multi_horizon"] = (
        float((np.sign(dy_anc[sig]) == np.sign(dp_anc[sig])).mean()) if sig.any() else 1.0
    )
    return out


def virtual_step_check(
    model: nn.Module,
    temp_hist: torch.Tensor,
    coal_hist: torch.Tensor,
    current_coal: torch.Tensor,
    coal_recent_mean: torch.Tensor,
    coal_recent_std: torch.Tensor,
    step_tph: float = 2.0,
) -> dict:
    """KPI-4 虚拟阶跃响应与实时 Step Response 序列自检。"""
    model.eval()
    with torch.no_grad():
        coal_hist_enc = model._decouple_coal_hist_for_encoder(coal_hist)
        h_temp, h_coal = model.encoder(temp_hist, coal_hist_enc)
        ctx = model.context_head(torch.cat([h_temp[:, -1], h_coal[:, -1]], dim=-1))
        cur = current_coal.reshape(-1, 1)
        up = cur.expand(-1, model.cfg.horizon_steps) + step_tph
        dn = cur.expand(-1, model.cfg.horizon_steps) - step_tph
        u_hist_m = (
            model._extract_history_coal_tph(coal_hist)
            if getattr(model, "use_in_flight", False)
            else None
        )
        base_out = model.coal_response(
            ctx,
            cur,
            cur.expand(-1, model.cfg.horizon_steps),
            coal_recent_mean,
            coal_recent_std,
            history_coal=u_hist_m,
        )
        f_up = model.coal_response(
            ctx, cur, up, coal_recent_mean, coal_recent_std, history_coal=u_hist_m
        )["forced_delta_c"]
        f_dn = model.coal_response(
            ctx, cur, dn, coal_recent_mean, coal_recent_std, history_coal=u_hist_m
        )["forced_delta_c"]
        k_t = base_out["k_t"]
        tau_t = base_out["tau_t"]
        m_eff = base_out["in_flight_eff_steps"]
        step_seq_full = base_out["step_response_full"]
        step_seq_forced = base_out["step_response_forced"]

    base_force = base_out["forced_delta_c"]
    du, dd = f_up - base_force, f_dn - base_force
    mid_idx = min(20, step_seq_full.shape[1] - 1)
    return {
        "step_up_endpoint_mean_c": float(du[:, -1].mean()),
        "step_down_endpoint_mean_c": float(dd[:, -1].mean()),
        "sign_correct": bool(du[:, -1].mean() > 0 and dd[:, -1].mean() < 0),
        "monotone_fraction_up": float((du[:, 1:] >= -0.05).float().mean()),
        "monotone_fraction_down": float((dd[:, 1:] <= 0.05).float().mean()),
        "k_t_mean": float(k_t.mean()),
        "k_t_std": float(k_t.std()) if k_t.numel() > 1 else 0.0,
        "tau_t_mean": float(tau_t.mean()),
        "tau_t_std": float(tau_t.std()) if tau_t.numel() > 1 else 0.0,
        "step_response_mid_mean_c": float(step_seq_full[:, mid_idx].mean()),
        "step_response_mid_std_c": (
            float(step_seq_full[:, mid_idx].std()) if step_seq_full.shape[0] > 1 else 0.0
        ),
        "step_response_forced_mid_mean_c": float(step_seq_forced[:, mid_idx].mean()),
        "in_flight_eff_steps_mean": float(m_eff.mean()),
        "in_flight_eff_steps_std": float(m_eff.std()) if m_eff.numel() > 1 else 0.0,
        "hist_coal_endpoint_std_c": (
            float(base_out["hist_coal_delta_c"][:, -1].std())
            if base_out["hist_coal_delta_c"].shape[0] > 1
            else 0.0
        ),
    }


def residual_target(
    target_temperature: torch.Tensor,
    current_temperature: torch.Tensor,
    current_coal: torch.Tensor,
    observed_future_coal: torch.Tensor,
    fopdt_out: dict[str, torch.Tensor],
) -> torch.Tensor:
    """计算扣除受迫响应后的残差目标（向后兼容辅助函数）。"""
    del current_coal, observed_future_coal
    if current_temperature.ndim == 1:
        current_temperature = current_temperature[:, None]
    return (target_temperature - current_temperature) - fopdt_out["forced_delta_c"]
