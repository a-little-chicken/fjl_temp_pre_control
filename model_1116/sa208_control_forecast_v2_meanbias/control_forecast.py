#!/usr/bin/env python3
"""v2 控制导向 15 分钟分解炉温度轨迹预测模型（按优化整改方案升级）。

相对 v1 的整改：
1. LPV-FOPDT：废除固定常数 FOPDT，K/tau 由工况上下文（隐含风煤比/负荷/饱和状态）
   在物理区间内调制；增益校准系数在训练前用闭环数据标定（stage-1 锚定）。
2. 双流解耦：温度相空间特征与尾煤/包络特征各自独立 Patch 投影 + 共享编码器，
   未来 Query 通过 Cross-Attention 同时检索双流，并带近端 ALiBi 偏置。
3. Loss：新增门控振幅匹配项（仅在真实波动窗口生效，防低波动误振幅）与
   K 先验锚定项；保留远期加权 Huber、880/950℃ 极值重罚、趋势与平滑项。

安全边界：推理输入只允许历史温度/实际尾煤与起点已知候选未来煤量；
未来真实煤量仅用于离线 Oracle 评价与残差标签；不访问 IOT、不下发动作。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ForecastConfig:
    history_steps: int = 240          # 60 min @ 15s
    horizon_steps: int = 60           # 15 min @ 15s
    patch_len: int = 8                # 2 min
    patch_stride: int = 4             # 1 min
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    dt_seconds: float = 15.0
    # --- 工程先验 FOPDT（待辨识；仅作为 LPV 调制的基准中心）---
    gain_c_per_tph: float = 15.0      # K_base
    tau_seconds: float = 200.0        # tau_base
    dead_steps: int = 1               # 15s（先验区间 15~30s -> 1~2 步）
    gain_low: float = 7.5             # K 物理下限
    gain_high: float = 22.5           # K 物理上限
    tau_low: float = 120.0
    tau_high: float = 280.0
    calibration_low: float = 0.3      # 闭环标定增益系数下限
    calibration_high: float = 3.0     # 闭环标定增益系数上限
    temperature_scale_c: float = 50.0
    coal_mean_tph: float = 10.0
    coal_scale_tph: float = 5.0
    # --- 尾煤侧特征窗口（因果，全部只用到当前步）---
    coal_mean_window: int = 120       # 30min 滚动均值
    coal_std_window: int = 20         # 5min 滚动标准差
    sat_low_tph: float = 5.0          # 触底指示
    sat_low_max_steps: int = 12       # 饱和计时上限（3min）


def num_patches(cfg: ForecastConfig) -> int:
    return (cfg.history_steps - cfg.patch_len) // cfg.patch_stride + 1


# ---------------------------------------------------------------------------
# 阶段一：状态自适应 FOPDT（LPV）
# ---------------------------------------------------------------------------

def fopdt_response(delta_u: torch.Tensor, k: torch.Tensor, alpha: torch.Tensor,
                   dead_steps: int) -> torch.Tensor:
    """一阶惯性+纯滞后递推。delta_u: [B,H] 相对保持基准的煤量增量；k, alpha: [B]。

    返回 [B,H] 温度增量（相对当前热状态基准）。
    """
    b, h = delta_u.shape
    state = torch.zeros(b, device=delta_u.device, dtype=delta_u.dtype)
    out = torch.empty_like(delta_u)
    expo = torch.arange(1, dead_steps + 1, device=delta_u.device, dtype=delta_u.dtype)
    for t in range(h):
        d = t - dead_steps
        if d >= 0:
            state = alpha * state + (1.0 - alpha) * k * delta_u[:, d]
        else:
            state = alpha.clamp(max=1.0 - 1e-6).pow(expo[d]) * state
        out[:, t] = state
    return out


class StateAdaptiveCoalFOPDT(nn.Module):
    """LPV-FOPDT：K_t/tau_t 由历史工况上下文在物理区间内调制，禁止学成负增益。

    `calibration` 为 stage-1 闭环标定的全局增益系数（初始由数据搜索给出，
    训练期仅允许在 [calibration_low, calibration_high] 内微调），用于吸收
    工程先验 K 与真实闭环边际增益之间的系统性偏差。
    """

    def __init__(self, cfg: ForecastConfig):
        super().__init__()
        if cfg.dead_steps < 1 or cfg.dead_steps > 2:
            raise ValueError("dead_steps 先验区间为 1~2 步 (15~30s)")
        self.cfg = cfg
        self.scheduler = nn.Sequential(
            nn.Linear(cfg.d_model + 2, 64), nn.GELU(), nn.Linear(64, 2), nn.Tanh(),
        )
        nn.init.zeros_(self.scheduler[2].weight)
        nn.init.zeros_(self.scheduler[2].bias)
        # 初始 1.0：先验 K 不被缩放，训练/标定阶段再调整
        self.calibration = nn.Parameter(torch.tensor(0.0))

    @property
    def gain_calibration(self) -> torch.Tensor:
        return torch.exp(self.calibration).clamp(self.cfg.calibration_low,
                                                 self.cfg.calibration_high)

    def params(self, context: torch.Tensor, coal_recent_mean: torch.Tensor,
               coal_recent_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (k_t [B], tau_t [B], alpha_t [B])。"""
        if coal_recent_mean.ndim == 1:
            coal_recent_mean = coal_recent_mean[:, None]
        if coal_recent_std.ndim == 1:
            coal_recent_std = coal_recent_std[:, None]
        sched_in = torch.cat([context, coal_recent_mean, coal_recent_std], dim=-1)
        mod = self.scheduler(sched_in)
        cfg = self.cfg
        k_t = cfg.gain_c_per_tph * (1.0 + 0.5 * mod[:, 0]) * self.gain_calibration
        k_t = k_t.clamp(cfg.gain_low, cfg.gain_high)
        tau_t = cfg.tau_seconds * (1.0 + 0.4 * mod[:, 1])
        tau_t = tau_t.clamp(cfg.tau_low, cfg.tau_high)
        alpha_t = torch.exp(-cfg.dt_seconds / tau_t)
        return k_t, tau_t, alpha_t

    def prior_loss(self, k_t: torch.Tensor) -> torch.Tensor:
        """先验锚定：无强证据时 K_t 保持在标定后的先验附近。"""
        return ((k_t - self.cfg.gain_c_per_tph * self.gain_calibration)
                / self.cfg.gain_c_per_tph).square().mean()

    def forward(self, context: torch.Tensor, current_coal: torch.Tensor,
                candidate_future_coal: torch.Tensor,
                coal_recent_mean: torch.Tensor, coal_recent_std: torch.Tensor) -> dict[str, torch.Tensor]:
        if candidate_future_coal.ndim != 2:
            raise ValueError("candidate_future_coal 形状应为 [batch, horizon]")
        if current_coal.ndim == 1:
            current_coal = current_coal[:, None]
        if not (torch.isfinite(candidate_future_coal).all() and torch.isfinite(current_coal).all()):
            raise ValueError("煤量输入必须有限")
        k_t, tau_t, alpha_t = self.params(context, coal_recent_mean, coal_recent_std)
        delta_u = candidate_future_coal - current_coal
        forced = fopdt_response(delta_u, k_t, alpha_t, self.cfg.dead_steps)
        return {"forced_delta_c": forced, "k_t": k_t, "tau_t": tau_t,
                "prior_loss": self.prior_loss(k_t)}


def calibrate_gain_scale(free: torch.Tensor, forced_at_scale: torch.Tensor,
                         truth_delta: torch.Tensor) -> float:
    """stage-1 闭环标定：搜索增益系数 c 使 free + c*forced 最接近真实 ΔT。

    forced_at_scale 为 c=1 时的 FOPDT 响应；返回最小 RMSE 对应的 c。
    这只能给出闭环意义上的最优折中，不是因果辨识。
    """
    err0 = float(torch.sqrt(((free - truth_delta) ** 2).mean()))
    best_c, best = 1.0, err0
    for c in np.linspace(0.3, 3.0, 28):
        e = float(torch.sqrt(((free + c * forced_at_scale - truth_delta) ** 2).mean()))
        if e < best:
            best, best_c = e, float(c)
    return best_c


# ---------------------------------------------------------------------------
# 阶段二：双流 Patch 编码 + Cross-Attention 解码
# ---------------------------------------------------------------------------

class DualStreamEncoder(nn.Module):
    """温度/尾煤各自独立 Patch 投影，共享 Transformer 权重；带近端 ALiBi 偏置。"""

    def __init__(self, cfg: ForecastConfig, c_temp: int, c_coal: int):
        super().__init__()
        self.cfg = cfg
        self.temp_proj = nn.Linear(cfg.patch_len * c_temp, cfg.d_model)
        self.coal_proj = nn.Linear(cfg.patch_len * c_coal, cfg.d_model)
        self.position = nn.Parameter(torch.zeros(1, num_patches(cfg), cfg.d_model))
        self.recency_bias = nn.Parameter(torch.zeros(num_patches(cfg)))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=2 * cfg.d_model,
            dropout=cfg.dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers,
                                             enable_nested_tensor=False)

    def _patches(self, x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        p = x.unfold(1, self.cfg.patch_len, self.cfg.patch_stride)
        p = p.permute(0, 1, 3, 2).contiguous().flatten(2)
        return proj(p) + self.position

    def forward(self, temp_hist: torch.Tensor, coal_hist: torch.Tensor):
        if temp_hist.ndim != 3 or coal_hist.ndim != 3:
            raise ValueError("历史输入必须是 [batch, history, C]")
        if temp_hist.shape[1] != self.cfg.history_steps or coal_hist.shape[1] != self.cfg.history_steps:
            raise ValueError("历史长度不匹配")
        if not (torch.isfinite(temp_hist).all() and torch.isfinite(coal_hist).all()):
            raise ValueError("历史输入必须有限")
        h_temp = self.encoder(self._patches(temp_hist, self.temp_proj))
        h_coal = self.encoder(self._patches(coal_hist, self.coal_proj))
        # 近端增强：越靠近当前时刻的 patch 偏置越大（可学习总强度 × 线性斜坡）
        ramp = torch.linspace(0.0, 1.0, num_patches(self.cfg), device=h_temp.device)
        boost = (self.recency_bias * ramp).view(1, -1, 1)
        return h_temp + boost, h_coal + boost


class ControlOrientedForecaster(nn.Module):
    """v2 双分支模型：自由响应（双流 PatchTST + Cross-Attention）+ LPV-FOPDT 受迫响应。"""

    def __init__(self, cfg: ForecastConfig, c_temp: int, c_coal: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = DualStreamEncoder(cfg, c_temp, c_coal)
        self.coal_response = StateAdaptiveCoalFOPDT(cfg)
        self.query = nn.Parameter(torch.empty(1, cfg.horizon_steps, cfg.d_model))
        nn.init.normal_(self.query, std=0.02)
        self.cross_temp = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                                dropout=cfg.dropout, batch_first=True)
        self.cross_coal = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                                dropout=cfg.dropout, batch_first=True)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 1),
        )
        self.context_head = nn.Linear(cfg.d_model * 2, cfg.d_model)

    def forward(self, temp_hist: torch.Tensor, coal_hist: torch.Tensor,
                current_coal: torch.Tensor, candidate_future_coal: torch.Tensor,
                coal_recent_mean: torch.Tensor, coal_recent_std: torch.Tensor) -> dict[str, torch.Tensor]:
        h_temp, h_coal = self.encoder(temp_hist, coal_hist)
        q = self.query.expand(temp_hist.shape[0], -1, -1)
        a_t, _ = self.cross_temp(q, h_temp, h_temp, need_weights=False)
        a_c, _ = self.cross_coal(q, h_coal, h_coal, need_weights=False)
        decoded = q + a_t + a_c
        free = self.head(self.norm(decoded)).squeeze(-1) * self.cfg.temperature_scale_c
        context = self.context_head(torch.cat([h_temp[:, -1], h_coal[:, -1]], dim=-1))
        forced = self.coal_response(context, current_coal, candidate_future_coal,
                                    coal_recent_mean, coal_recent_std)
        return {
            "free_delta_c": free,
            "forced_delta_c": forced["forced_delta_c"],
            "temperature_delta_c": free + forced["forced_delta_c"],
            "k_t": forced["k_t"],
            "tau_t": forced["tau_t"],
            "prior_loss": forced["prior_loss"],
        }


# ---------------------------------------------------------------------------
# 阶段三：控制导向 Loss
# ---------------------------------------------------------------------------

class ControlTrajectoryLoss(nn.Module):
    """远期加权 Huber + 极值重罚 + 门控振幅匹配 + 趋势 + 平滑。

    振幅项按真实窗口峰峰值门控（p2p <= calm_c 的窗口不惩罚），
    吸取 v1 事件 Loss “低波动误振幅 100%” 的教训。
    """

    def __init__(self, horizon: int = 60, high_threshold_c: float = 950.0,
                 low_threshold_c: float = 880.0, event_weight: float = 1.0,
                 amp_weight: float = 0.5, calm_p2p_c: float = 12.0,
                 trend_weight: float = 0.1, smooth_weight: float = 0.02,
                 mean_bias_weight: float = 0.2,
                 huber_delta_c: float = 5.0):
        super().__init__()
        self.high, self.low = high_threshold_c, low_threshold_c
        self.event_weight, self.amp_weight = event_weight, amp_weight
        self.calm_p2p_c, self.trend_weight, self.smooth_weight = calm_p2p_c, trend_weight, smooth_weight
        self.mean_bias_weight = mean_bias_weight
        self.huber_delta = huber_delta_c
        x = torch.linspace(0.0, 1.0, horizon)
        weights = 0.5 + 1.5 * x.square()
        self.register_buffer("horizon_weights", weights / weights.mean())

    def forward(self, prediction: torch.Tensor, target: torch.Tensor,
                current_temperature: torch.Tensor) -> dict[str, torch.Tensor]:
        if prediction.shape != target.shape or prediction.ndim != 2:
            raise ValueError("prediction/target 必须同形 [batch, horizon]")
        if prediction.shape[1] != self.horizon_weights.numel():
            raise ValueError("horizon 与 Loss 配置不一致")
        if current_temperature.ndim == 1:
            current_temperature = current_temperature[:, None]
        abs_pred = current_temperature + prediction
        abs_true = current_temperature + target

        w = self.horizon_weights.to(prediction.dtype)[None, :]
        point = F.smooth_l1_loss(prediction, target, reduction="none", beta=self.huber_delta)
        base = (point * w).mean()
        extreme = ((abs_true >= self.high) | (abs_true <= self.low)).to(point.dtype)
        event = (point * extreme * w).sum() / (extreme.sum() + 1e-6)

        p2p_t = (abs_true.max(dim=1).values - abs_true.min(dim=1).values)
        p2p_p = (abs_pred.max(dim=1).values - abs_pred.min(dim=1).values)
        gate = (p2p_t > self.calm_p2p_c).to(prediction.dtype)
        active = gate.sum()
        amp = (((p2p_p - p2p_t).abs() + 0.5 * (abs_pred.std(dim=1) - abs_true.std(dim=1)).abs())
               * gate).sum() / (active + 1e-6)

        dp = prediction[:, 1:] - prediction[:, :-1]
        dy = target[:, 1:] - target[:, :-1]
        trend = F.smooth_l1_loss(dp, dy, reduction="mean", beta=1.0)
        d2 = prediction[:, 2:] - 2 * prediction[:, 1:-1] + prediction[:, :-2]
        smooth = d2.square().mean()
        mean_bias = (prediction.mean(dim=1) - target.mean(dim=1)).square().mean()
        total = (base + self.event_weight * event + self.amp_weight * amp
                 + self.trend_weight * trend + self.smooth_weight * smooth
                 + self.mean_bias_weight * mean_bias)
        return {"total": total, "far_huber": base, "event_huber": event,
                "amplitude": amp, "trend": trend, "smooth": smooth,
                "mean_bias": mean_bias,
                "amp_active_fraction": active / max(prediction.shape[0], 1)}


# ---------------------------------------------------------------------------
# 因果特征构造（推理/训练共用，只用当前步及以前数据）
# ---------------------------------------------------------------------------

def build_temp_features(t_hist: np.ndarray, cfg: ForecastConfig) -> np.ndarray:
    """温度相空间：相对当前值、一阶/二阶差分（因果）。"""
    t = np.asarray(t_hist, dtype=np.float32)
    if t.shape != (cfg.history_steps,):
        raise ValueError("温度历史长度不匹配")
    rel = (t - t[-1]) / cfg.temperature_scale_c
    dt1 = np.empty_like(t); dt1[1:] = np.diff(t); dt1[0] = 0.0
    dt1 = dt1 / cfg.temperature_scale_c
    dt2 = np.empty_like(t); dt2[1:] = np.diff(dt1); dt2[0] = 0.0
    return np.stack([rel, dt1, dt2], axis=-1)


def build_coal_features(u_hist: np.ndarray, cfg: ForecastConfig) -> np.ndarray:
    """尾煤/包络：标准化煤量、阶跃脉冲、滚动均值、滚动标准差、触底计时（均因果）。"""
    u = np.asarray(u_hist, dtype=np.float32)
    if u.shape != (cfg.history_steps,):
        raise ValueError("煤量历史长度不匹配")
    s = 1.0 / cfg.coal_scale_tph
    uf = (u - cfg.coal_mean_tph) * s
    pulse = np.zeros_like(u); pulse[1:] = np.diff(u); pulse = np.clip(pulse * s, -5, 5)
    def rolling_mean_std(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
        cs = np.r_[0.0, np.cumsum(x, dtype=np.float64)]
        cs2 = np.r_[0.0, np.cumsum(x.astype(np.float64) ** 2)]
        end = np.arange(1, len(x) + 1)
        start = np.maximum(0, end - window)
        count = end - start
        mean_x = (cs[end] - cs[start]) / count
        var = np.maximum((cs2[end] - cs2[start]) / count - mean_x**2, 0.0)
        return mean_x.astype(np.float32), np.sqrt(var).astype(np.float32)

    mean, _ = rolling_mean_std(u, min(cfg.coal_mean_window, cfg.history_steps))
    mean = (mean - cfg.coal_mean_tph) * s
    _, std = rolling_mean_std(u, min(cfg.coal_std_window, cfg.history_steps))
    std = np.maximum(std, 1e-4) * s
    below = (u < cfg.sat_low_tph)
    pos = np.arange(len(u))
    last_reset = np.maximum.accumulate(np.where(~below, pos, -1))
    run = np.where(below, pos - last_reset, 0)
    sat = np.minimum(run, cfg.sat_low_max_steps).astype(np.float32) / cfg.sat_low_max_steps
    return np.stack([uf, pulse, mean, std, sat], axis=-1)


def coal_recent_stats(u_hist: np.ndarray, cfg: ForecastConfig) -> tuple[float, float]:
    recent = u_hist[-12:]
    return (float((recent.mean() - cfg.coal_mean_tph) / cfg.coal_scale_tph),
            float((u_hist[-cfg.coal_std_window:] - cfg.coal_mean_tph).std() / cfg.coal_scale_tph))


# ---------------------------------------------------------------------------
# 阶段三：准入 KPI（KPI-1 极值误差 / KPI-2 拐点相位 / KPI-3 振幅还原）
# ---------------------------------------------------------------------------

def find_turns(signs: np.ndarray, min_gap: int = 4) -> np.ndarray:
    idx = []
    last = -10**9
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1] and signs[i] != 0 and signs[i - 1] != 0:
            if i - last >= min_gap:
                idx.append(i); last = i
    return np.array(idx, dtype=int)


def kpi_metrics(pred_abs: np.ndarray, true_abs: np.ndarray,
                high_c: float = 950.0, low_c: float = 880.0) -> dict:
    if isinstance(pred_abs, torch.Tensor):
        pred_abs = pred_abs.detach().cpu().numpy()
    if isinstance(true_abs, torch.Tensor):
        true_abs = true_abs.detach().cpu().numpy()
    pred_abs = np.asarray(pred_abs); true_abs = np.asarray(true_abs)
    out = {}
    # KPI-1: 远端波峰/波谷误差（全程峰谷 + 越红线样本）
    pk_p = pred_abs.max(axis=1); pk_t = true_abs.max(axis=1)
    vv_p = pred_abs.min(axis=1); vv_t = true_abs.min(axis=1)
    out["peak_valley_mae_c"] = float(np.concatenate([np.abs(pk_p - pk_t), np.abs(vv_p - vv_t)]).mean())
    hi = np.where(true_abs >= high_c)
    lo = np.where(true_abs <= low_c)
    both_t = np.concatenate([true_abs[hi], true_abs[lo]])
    both_p = np.concatenate([pred_abs[hi], pred_abs[lo]])
    out["redline_mae_c"] = float(np.abs(both_p - both_t).mean()) if both_t.size else None
    out["redline_points"] = int(both_t.size)
    # KPI-2: 拐点时间对齐（平滑一阶差分过零）
    try:
        from scipy.signal import savgol_filter
        sp = savgol_filter(np.diff(pred_abs, axis=1), 5, 2, axis=1)
        st = savgol_filter(np.diff(true_abs, axis=1), 5, 2, axis=1)
    except Exception:
        sp, st = np.diff(pred_abs, axis=1), np.diff(true_abs, axis=1)
    sp = np.where(np.abs(sp) > 1e-3, np.sign(sp), 0)
    st = np.where(np.abs(st) > 1e-3, np.sign(st), 0)
    tp_tt = sum(find_turns(r).size for r in st)
    tp_pp = sum(find_turns(r).size for r in sp)
    errs = []
    for a, b in zip(sp, st):
        ta, tb = find_turns(a), find_turns(b)
        if tb.size == 0:
            continue
        for j in tb:
            if ta.size:
                errs.append(int(np.abs(ta - j).min()))
    out["turn_time_mae_steps"] = float(np.mean(errs)) if errs else None
    out["true_window_turns"] = int(tp_tt)
    out["pred_window_turns"] = int(tp_pp)
    out["turn_miss_fraction"] = 1.0 - min(1.0, (len(errs) / max(tp_tt, 1))) if tp_tt else None
    # KPI-3: 振幅还原率
    p2p_t = np.ptp(true_abs, axis=1); p2p_p = np.ptp(pred_abs, axis=1)
    out["p2p_ratio"] = float((p2p_p.mean() / p2p_t.mean())) if p2p_t.mean() > 1e-6 else None
    out["mean_truth_p2p_c"] = float(p2p_t.mean())
    out["mean_pred_p2p_c"] = float(p2p_p.mean())
    return out


def virtual_step_check(model, temp_hist, coal_hist, current_coal, coal_recent_mean,
                       coal_recent_std, step_tph: float = 2.0) -> dict:
    """KPI-4：固定历史，候选煤量 +step / -step 的受迫响应检查（仅离线诊断）。"""
    model.eval()
    with torch.no_grad():
        h_temp, h_coal = model.encoder(temp_hist, coal_hist)
        ctx = model.context_head(torch.cat([h_temp[:, -1], h_coal[:, -1]], dim=-1))
        cur = current_coal.reshape(-1, 1)
        up = cur.expand(-1, model.cfg.horizon_steps) + step_tph
        dn = cur.expand(-1, model.cfg.horizon_steps) - step_tph
        base_force = model.coal_response(ctx, cur, cur.expand(-1, model.cfg.horizon_steps),
                                         coal_recent_mean, coal_recent_std)["forced_delta_c"]
        f_up = model.coal_response(ctx, cur, up, coal_recent_mean, coal_recent_std)["forced_delta_c"]
        f_dn = model.coal_response(ctx, cur, dn, coal_recent_mean, coal_recent_std)["forced_delta_c"]
    du, dd = f_up - base_force, f_dn - base_force
    out = {
        "step_up_endpoint_mean_c": float(du[:, -1].mean()),
        "step_down_endpoint_mean_c": float(dd[:, -1].mean()),
        "sign_correct": bool(du[:, -1].mean() > 0 and dd[:, -1].mean() < 0),
        "monotone_fraction_up": float((du[:, 1:] >= -0.05).float().mean()),
        "monotone_fraction_down": float((dd[:, 1:] <= 0.05).float().mean()),
        "k_t_mean": float(model.coal_response.params(ctx, coal_recent_mean, coal_recent_std)[0].mean()),
        "tau_t_mean": float(model.coal_response.params(ctx, coal_recent_mean, coal_recent_std)[1].mean()),
    }
    return out


def residual_target(target_temperature, current_temperature, current_coal,
                    observed_future_coal, fopdt_out) -> torch.Tensor:
    """离线残差标签：真实 ΔT 减去 LPV-FOPDT 对已观测未来煤量的响应。

    仅用于训练/Oracle 评价；推理时禁止使用未来真实煤量。
    """
    if current_temperature.ndim == 1:
        current_temperature = current_temperature[:, None]
    return (target_temperature - current_temperature) - fopdt_out["forced_delta_c"]
