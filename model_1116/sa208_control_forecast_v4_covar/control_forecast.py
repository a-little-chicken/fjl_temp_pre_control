#!/usr/bin/env python3
"""v4 控制导向 15 分钟分解炉温度轨迹预测模型（支持 MPC 三分量在途余热解耦 + 可选多源热工协变量验证）。

相对 v3 的核心升级：
1. 方向二（核心升级）：120 步连续状态预热 LPV-FOPDT（显式分离“历史在途尾煤余热 ΔT_hist_coal”与“未来动作受迫响应 ΔT_fut_coal”）：
   - 废除 t=0 处热状态归零的静态平衡假设，利用过去 15 分钟（60 步）实际尾煤序列预热一阶热惯性状态 E(t)；
   - 显式拆解出供 MPC 直接使用的三大物理分量：
     ① `hist_coal_delta_c` (ΔT_hist_coal)：假设未来尾煤维持 u_t 不变时，历史已下发尾煤在未来 15min 继续释放的在途热惯性余波；
     ② `disturb_delta_c` (ΔT_dist)：剥离尾煤热惯性后的外部扰动（替代燃料/生料周期）自由演化分量；
     ③ `forced_delta_c` (ΔT_fut_coal)：未来 15min 尾煤偏离 u_t 激发的净受迫响应（关于未来控制增量严格单调可微）；
   - 将多时间尺度在途热惯性弛豫曲线注入解码器各预测步，并通过可学习耦合系数叠加至开环基准轨迹 `free_delta_c`。
2. 方向一（对比验证）：可选接入多源热工前馈协变量流（`c5b_valve` 分料阀、`sat_temp` 二次风温、
   `decompose_press` 分解炉负压、`alt_fuel` 替代燃料、`kiln_current` 窑电流、`ladder_pressure` 阶梯炉负压），
   通过独立 Patch 编码与 Cross-Attention 验证其余测点的边际信息增益。

安全边界：推理输入只允许历史测点与起点已知候选未来煤量；
未来真实煤量仅用于离线 Oracle 评价；不访问 IOT、不下发动作。
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
    in_flight_steps: int = 60         # 15 min (900s) 历史尾煤在途热惯性预热步数
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
    # --- 物理平滑与饱和先验 ---
    coal_ema_beta: float = 0.5        # 煤量增量因果平滑系数（常数阶跃下严格不变）
    coal_linear_step_tph: float = 2.5 # 线性阶跃区（±2.5 t/h 内严格线性）
    coal_max_extra_tph: float = 2.0   # 超出线性区后的软饱和裕度
    origin_ramp_steps: float = 1.5    # 起点热惯性过渡步数（抑制 t=+15s 非物理跳变）
    max_step_delta_c: float = 6.0     # 单步（15s）物理最大合理温变阈值


def num_patches(cfg: ForecastConfig) -> int:
    return (cfg.history_steps - cfg.patch_len) // cfg.patch_stride + 1


# ---------------------------------------------------------------------------
# 阶段一：状态自适应 FOPDT（LPV）与 120 步在途余热状态机
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


def fopdt_in_flight_decomposition(
    u_hist: torch.Tensor,
    u_fut: torch.Tensor,
    k: torch.Tensor,
    alpha: torch.Tensor,
    dead_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """120 步（历史 M 步 + 未来 H 步）连续状态递推，显式分离历史在途尾煤余热与未来动作净响应。

    输入：
      u_hist: [B, M] 过去 M 步（如 60 步 = 15min）有效尾煤序列（最后一个点 u_hist[:, -1] 为当前时刻 u_t）
      u_fut : [B, H] 未来 H 步候选/实际有效尾煤序列
      k, alpha: [B] 当前工况下的有效热增益与离散时间常数衰减因子
    返回：
      hist_coal_delta_c : [B, H] 假设未来维持 u_t 不变时，历史已下发尾煤在未来 H 步继续释放的在途余热 ΔT_hist_coal
      forced_delta_c    : [B, H] 未来煤量偏离 u_t 激发的净受迫响应 ΔT_fut_coal（当 u_fut == u_t 时严格为 0）
      total_coal_delta_c: [B, H] 尾煤总热贡献 ΔT_hist_coal + ΔT_fut_coal
    """
    b, m = u_hist.shape
    h = u_fut.shape[1]
    u_cur = u_hist[:, -1:]
    # 以当前煤量 u_t 为零参考基准计算相对热状态，数值更稳健且在常数历史下严格为 0
    rel_hist = u_hist - u_cur
    rel_fut = u_fut - u_cur
    rel_hold = torch.zeros_like(u_fut)

    rel_full_seq = torch.cat([rel_hist, rel_fut], dim=1)   # [B, M+H]
    rel_hold_seq = torch.cat([rel_hist, rel_hold], dim=1)  # [B, M+H]

    state_full = k * rel_full_seq[:, 0]
    state_hold = k * rel_hold_seq[:, 0]

    for s in range(m):
        eff_idx = max(0, s - dead_steps)
        u_eff = rel_full_seq[:, eff_idx]
        state_full = alpha * state_full + (1.0 - alpha) * k * u_eff
        state_hold = alpha * state_hold + (1.0 - alpha) * k * u_eff

    base_state_t = state_full  # [B]，时刻 t (s = m-1) 的在途热状态失衡度 E(t) - K*u_t
    hist_list, full_list = [], []
    for step in range(m, m + h):
        eff_idx = max(0, step - dead_steps)
        state_full = alpha * state_full + (1.0 - alpha) * k * rel_full_seq[:, eff_idx]
        state_hold = alpha * state_hold + (1.0 - alpha) * k * rel_hold_seq[:, eff_idx]
        hist_list.append(state_hold - base_state_t)
        full_list.append(state_full - base_state_t)

    hist_coal_delta = torch.stack(hist_list, dim=1)
    total_coal_delta = torch.stack(full_list, dim=1)
    forced_delta = total_coal_delta - hist_coal_delta
    return hist_coal_delta, forced_delta, total_coal_delta


class StateAdaptiveCoalFOPDT(nn.Module):
    """LPV-FOPDT：支持状态调制 K_t/tau_t、负荷饱和折减与 120 步历史在途余热显式分解。"""

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
        self.calibration = nn.Parameter(torch.tensor(0.0))

    @property
    def gain_calibration(self) -> torch.Tensor:
        return torch.exp(self.calibration).clamp(self.cfg.calibration_low,
                                                 self.cfg.calibration_high)

    def saturation_discount(self, coal_recent_mean: torch.Tensor,
                            coal_recent_std: torch.Tensor) -> torch.Tensor:
        """煤量侧饱和折减系数：常规工况 (mean=0) 为 1.0，极端高负荷下平滑折减至 [0.75, 1.0]。"""
        m = coal_recent_mean.squeeze(-1) if coal_recent_mean.ndim > 1 else coal_recent_mean
        s = coal_recent_std.squeeze(-1) if coal_recent_std.ndim > 1 else coal_recent_std
        overload = F.relu(m.abs() - 1.0) + F.relu(s - 1.0)
        return 1.0 - 0.25 * torch.tanh(overload.square())

    def smooth_coal_increment(self, delta_u: torch.Tensor) -> torch.Tensor:
        """因果指数平滑 + 分段 C1 软饱和：常数阶跃 (|Δu|<=2.5) 下输出与原输入完全恒等。"""
        cfg = self.cfg
        beta = cfg.coal_ema_beta
        if beta > 0.0 and delta_u.shape[1] > 1:
            ema = torch.empty_like(delta_u)
            ema[:, 0] = delta_u[:, 0]
            for t in range(1, delta_u.shape[1]):
                ema[:, t] = beta * ema[:, t - 1] + (1.0 - beta) * delta_u[:, t]
        else:
            ema = delta_u
        lin = cfg.coal_linear_step_tph
        extra = cfg.coal_max_extra_tph
        abs_u = ema.abs()
        excess = F.relu(abs_u - lin)
        compressed = torch.where(
            abs_u <= lin,
            abs_u,
            lin + extra * torch.tanh(excess / max(extra, 1e-3)),
        )
        return torch.sign(ema) * compressed

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

    def prior_loss(self, k_t: torch.Tensor,
                   tau_t: torch.Tensor | None = None) -> torch.Tensor:
        """完整先验锚定：同时将 K_t 锚定于 K_base、tau_t 锚定于 tau_base、calibration 锚定于 1.0，
        并施加边界软排斥势垒，防止闭环负相关梯度将参数推向 [gain_low, tau_high] 边界。
        """
        cfg = self.cfg
        k_anchor = ((k_t - cfg.gain_c_per_tph) / cfg.gain_c_per_tph).square().mean()
        cal_anchor = (self.gain_calibration - 1.0).square()
        k_span = max(cfg.gain_high - cfg.gain_low, 1e-3)
        k_margin = 0.15 * k_span
        k_bound = (F.relu(cfg.gain_low + k_margin - k_t).square()
                   + F.relu(k_t - (cfg.gain_high - k_margin)).square()).mean() / (k_margin ** 2)
        loss = k_anchor + 0.5 * cal_anchor + 0.25 * k_bound
        if tau_t is not None:
            tau_anchor = ((tau_t - cfg.tau_seconds) / cfg.tau_seconds).square().mean()
            t_span = max(cfg.tau_high - cfg.tau_low, 1e-3)
            t_margin = 0.15 * t_span
            t_bound = (F.relu(cfg.tau_low + t_margin - tau_t).square()
                       + F.relu(tau_t - (cfg.tau_high - t_margin)).square()).mean() / (t_margin ** 2)
            loss = loss + tau_anchor + 0.25 * t_bound
        return loss

    def forward(self, context: torch.Tensor, current_coal: torch.Tensor,
                candidate_future_coal: torch.Tensor,
                coal_recent_mean: torch.Tensor, coal_recent_std: torch.Tensor,
                history_coal: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if candidate_future_coal.ndim != 2:
            raise ValueError("candidate_future_coal 形状应为 [batch, horizon]")
        if current_coal.ndim == 1:
            current_coal = current_coal[:, None]
        if not (torch.isfinite(candidate_future_coal).all() and torch.isfinite(current_coal).all()):
            raise ValueError("煤量输入必须有限")
        k_t, tau_t, alpha_t = self.params(context, coal_recent_mean, coal_recent_std)
        disc = self.saturation_discount(coal_recent_mean, coal_recent_std)[:, None]
        raw_delta_u = (candidate_future_coal - current_coal) * disc
        delta_u = self.smooth_coal_increment(raw_delta_u)

        if history_coal is not None:
            if history_coal.ndim != 2:
                raise ValueError("history_coal 形状应为 [batch, in_flight_steps]")
            rel_hist_raw = (history_coal - current_coal) * disc
            lin = self.cfg.coal_linear_step_tph
            extra = self.cfg.coal_max_extra_tph
            abs_h = rel_hist_raw.abs()
            comp_h = torch.where(
                abs_h <= lin,
                abs_h,
                lin + extra * torch.tanh(F.relu(abs_h - lin) / max(extra, 1e-3)),
            )
            rel_hist = torch.sign(rel_hist_raw) * comp_h
            rel_hist[:, -1] = 0.0
            u_hist_eff = current_coal + rel_hist
            u_fut_eff = current_coal + delta_u
            hist_coal_delta, forced, total_coal_delta = fopdt_in_flight_decomposition(
                u_hist_eff, u_fut_eff, k_t, alpha_t, self.cfg.dead_steps
            )
        else:
            forced = fopdt_response(delta_u, k_t, alpha_t, self.cfg.dead_steps)
            hist_coal_delta = torch.zeros_like(forced)
            total_coal_delta = forced

        return {
            "forced_delta_c": forced,
            "hist_coal_delta_c": hist_coal_delta,
            "total_coal_delta_c": total_coal_delta,
            "k_t": k_t,
            "tau_t": tau_t,
            "saturation_discount": disc.squeeze(-1),
            "prior_loss": self.prior_loss(k_t, tau_t),
        }


def calibrate_gain_scale(free: torch.Tensor, forced_at_scale: torch.Tensor,
                         truth_delta: torch.Tensor) -> float:
    """stage-1 闭环标定：在 [0.3, 3.0] 网格搜索增益系数 c 使 free + c*forced 最接近真实 ΔT。"""
    grid = np.linspace(0.3, 3.0, 28)
    best_c = float(grid[0])
    best = float("inf")
    for c in grid:
        e = float(torch.sqrt(((free + float(c) * forced_at_scale - truth_delta) ** 2).mean()))
        if e < best:
            best, best_c = e, float(c)
    return best_c


def calibrate_closed_loop_coupling(free: torch.Tensor, forced_at_scale: torch.Tensor,
                                   truth_delta: torch.Tensor,
                                   min_coupling: float = 0.0) -> float:
    """闭环残差耦合校准：在 [min_coupling, 1.0] 搜索耦合权重，识别历史闭环 PID 负相关混杂。"""
    low = float(np.clip(min_coupling, 0.0, 1.0))
    grid = np.linspace(low, 1.0, 37)
    best_c = low
    best = float("inf")
    for c in grid:
        e = float(torch.sqrt(((free + float(c) * forced_at_scale - truth_delta) ** 2).mean()))
        if e < best:
            best, best_c = e, float(c)
    return best_c


# ---------------------------------------------------------------------------
# 阶段二：双流/多流 Patch 编码 + 历史在途余热融合解码
# ---------------------------------------------------------------------------

class DualStreamEncoder(nn.Module):
    """温度/尾煤（及可选热工协变量流）各自独立 Patch 投影，共享 Transformer 权重；带近端偏置与正交相位增强。"""

    def __init__(self, cfg: ForecastConfig, c_temp: int, c_coal: int, c_covar: int = 0):
        super().__init__()
        self.cfg = cfg
        self.c_covar = c_covar
        self.temp_proj = nn.Linear(cfg.patch_len * c_temp, cfg.d_model)
        self.coal_proj = nn.Linear(cfg.patch_len * c_coal, cfg.d_model)
        self.covar_proj = nn.Linear(cfg.patch_len * c_covar, cfg.d_model) if c_covar > 0 else None
        self.phase_proj = nn.Linear(cfg.patch_len * 2, cfg.d_model, bias=False)
        nn.init.normal_(self.phase_proj.weight, std=0.01)
        self.position = nn.Parameter(torch.zeros(1, num_patches(cfg), cfg.d_model))
        self.recency_bias = nn.Parameter(torch.zeros(num_patches(cfg)))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=2 * cfg.d_model,
            dropout=cfg.dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers,
                                             enable_nested_tensor=False)

    def _causal_phase_channels(self, temp_hist: torch.Tensor) -> torch.Tensor:
        """从温度相对序列因果提取 15~30min 周期的同相 (In-Phase) 与正交 (Quadrature) 分量。"""
        rel = temp_hist[:, :, 0:1].transpose(1, 2)  # [B, 1, T]
        k_fast = min(8, self.cfg.history_steps)     # 2 min
        k_slow = min(40, self.cfg.history_steps)    # 10 min
        fast = F.avg_pool1d(F.pad(rel, (k_fast - 1, 0), mode="replicate"), k_fast, stride=1)
        slow = F.avg_pool1d(F.pad(rel, (k_slow - 1, 0), mode="replicate"), k_slow, stride=1)
        in_phase = (fast - slow).transpose(1, 2)    # [B, T, 1]
        quad = torch.zeros_like(in_phase)
        step = min(4, self.cfg.history_steps - 1)
        if step > 0:
            quad[:, step:] = (in_phase[:, step:] - in_phase[:, :-step]) * (3.0 / step)
        return torch.cat([in_phase, quad], dim=-1)

    def _patches(self, x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        p = x.unfold(1, self.cfg.patch_len, self.cfg.patch_stride)
        p = p.permute(0, 1, 3, 2).contiguous().flatten(2)
        return proj(p) + self.position

    def encode_covariates(self, covar_hist: torch.Tensor) -> torch.Tensor:
        if self.covar_proj is None:
            raise ValueError("当前模型未启用协变量编码流 (c_covar=0)")
        if covar_hist.ndim != 3 or covar_hist.shape[1] != self.cfg.history_steps:
            raise ValueError("协变量历史形状不匹配")
        h_cov = self.encoder(self._patches(covar_hist, self.covar_proj))
        ramp = torch.linspace(0.0, 1.0, num_patches(self.cfg), device=h_cov.device)
        boost = (self.recency_bias * ramp).view(1, -1, 1)
        return h_cov + boost

    def forward(self, temp_hist: torch.Tensor, coal_hist: torch.Tensor):
        if temp_hist.ndim != 3 or coal_hist.ndim != 3:
            raise ValueError("历史输入必须是 [batch, history, C]")
        if temp_hist.shape[1] != self.cfg.history_steps or coal_hist.shape[1] != self.cfg.history_steps:
            raise ValueError("历史长度不匹配")
        if not (torch.isfinite(temp_hist).all() and torch.isfinite(coal_hist).all()):
            raise ValueError("历史输入必须有限")
        t_tokens = self._patches(temp_hist, self.temp_proj)
        phase_ch = self._causal_phase_channels(temp_hist)
        p_phase = phase_ch.unfold(1, self.cfg.patch_len, self.cfg.patch_stride)
        p_phase = p_phase.permute(0, 1, 3, 2).contiguous().flatten(2)
        t_tokens = t_tokens + self.phase_proj(p_phase)

        h_temp = self.encoder(t_tokens)
        h_coal = self.encoder(self._patches(coal_hist, self.coal_proj))
        ramp = torch.linspace(0.0, 1.0, num_patches(self.cfg), device=h_temp.device)
        boost = (self.recency_bias * ramp).view(1, -1, 1)
        return h_temp + boost, h_coal + boost


class ControlOrientedForecaster(nn.Module):
    """v4 控制导向预测模型：
    - 方向二：显式分离纯扰动 `disturb_delta_c`、历史在途尾煤余热 `hist_coal_delta_c` 与未来净受迫响应 `forced_delta_c`；
    - 方向一：可选接入第三条热工协变量流 (`c_covar > 0`)。
    """

    def __init__(self, cfg: ForecastConfig, c_temp: int, c_coal: int,
                 c_covar: int = 0, use_in_flight: bool = True):
        super().__init__()
        self.cfg = cfg
        self.c_covar = c_covar
        self.use_in_flight = use_in_flight
        self.encoder = DualStreamEncoder(cfg, c_temp, c_coal, c_covar=c_covar)
        self.coal_response = StateAdaptiveCoalFOPDT(cfg)
        self.query = nn.Parameter(torch.empty(1, cfg.horizon_steps, cfg.d_model))
        nn.init.normal_(self.query, std=0.02)
        ctx_in_dim = cfg.d_model * (3 if c_covar > 0 else 2)
        self.context_head = nn.Linear(ctx_in_dim, cfg.d_model)
        self.query_ctx = nn.Linear(cfg.d_model, cfg.d_model)
        self.cross_temp = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                                dropout=cfg.dropout, batch_first=True)
        self.cross_coal = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                                dropout=cfg.dropout, batch_first=True)
        self.cross_covar = (
            nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
            if c_covar > 0 else None
        )
        # 在途尾煤热惯性轨迹特征投影（输入: 归一化 ΔT_hist_coal 及其一阶差分 [B, H, 2] -> [B, H, d_model]）
        self.in_flight_proj = nn.Linear(2, cfg.d_model, bias=False)
        nn.init.normal_(self.in_flight_proj.weight, std=0.02)
        # 历史在途余热直接叠加系数（初始化为 0.20，允许网络自适应校准闭环在途余热权重）
        self.hist_coal_scale = nn.Parameter(torch.tensor(0.20 if use_in_flight else 0.0))

        self.horizon_mixer = nn.Sequential(
            nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=5, padding=2, groups=cfg.n_heads),
            nn.GELU(),
            nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=1),
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 1),
        )
        steps = torch.arange(1, cfg.horizon_steps + 1, dtype=torch.float32)
        origin_gate = 1.0 - torch.exp(-steps / max(cfg.origin_ramp_steps, 1e-3))
        self.register_buffer("origin_gate", origin_gate.view(1, -1))
        self.register_buffer("closed_loop_coupling", torch.tensor(1.0))

    def set_closed_loop_coupling(self, value: float) -> None:
        self.closed_loop_coupling.fill_(float(np.clip(value, 0.0, 1.0)))

    def _extract_history_coal_tph(self, coal_hist: torch.Tensor) -> torch.Tensor:
        """从标准化尾煤特征通道 0 还原过去 in_flight_steps 步的实际尾煤流量 (t/h)。"""
        m = min(self.cfg.in_flight_steps, coal_hist.shape[1])
        uf = coal_hist[:, -m:, 0]
        return uf * self.cfg.coal_scale_tph + self.cfg.coal_mean_tph

    def forward(self, temp_hist: torch.Tensor, coal_hist: torch.Tensor,
                current_coal: torch.Tensor, candidate_future_coal: torch.Tensor,
                coal_recent_mean: torch.Tensor, coal_recent_std: torch.Tensor,
                covar_hist: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h_temp, h_coal = self.encoder(temp_hist, coal_hist)
        ctx_parts = [h_temp[:, -1], h_coal[:, -1]]
        if self.c_covar > 0:
            if covar_hist is None:
                covar_hist = torch.zeros(
                    temp_hist.shape[0], self.cfg.history_steps, self.c_covar,
                    device=temp_hist.device, dtype=temp_hist.dtype,
                )
            h_cov = self.encoder.encode_covariates(covar_hist)
            ctx_parts.append(h_cov[:, -1])
        else:
            h_cov = None

        context = self.context_head(torch.cat(ctx_parts, dim=-1))
        u_hist_m = self._extract_history_coal_tph(coal_hist) if self.use_in_flight else None
        forced = self.coal_response(context, current_coal, candidate_future_coal,
                                    coal_recent_mean, coal_recent_std,
                                    history_coal=u_hist_m)
        hist_coal_delta = forced["hist_coal_delta_c"]

        q = self.query.expand(temp_hist.shape[0], -1, -1) + self.query_ctx(context).unsqueeze(1)
        a_t, _ = self.cross_temp(q, h_temp, h_temp, need_weights=False)
        a_c, _ = self.cross_coal(q, h_coal, h_coal, need_weights=False)
        decoded = q + a_t + a_c
        if self.cross_covar is not None and h_cov is not None:
            a_v, _ = self.cross_covar(q, h_cov, h_cov, need_weights=False)
            decoded = decoded + a_v

        if self.use_in_flight:
            # 将物理计算的在途尾煤余热曲线及其变化率显式注入各未来预测步
            hc_norm = hist_coal_delta / self.cfg.temperature_scale_c
            hc_diff = F.pad(hc_norm[:, 1:] - hc_norm[:, :-1], (1, 0), value=0.0)
            in_flight_feat = torch.stack([hc_norm, hc_diff], dim=-1)
            decoded = decoded + self.in_flight_proj(in_flight_feat)

        mixed = self.horizon_mixer(decoded.transpose(1, 2)).transpose(1, 2)
        decoded = decoded + mixed
        raw_disturb = self.head(self.norm(decoded)).squeeze(-1) * self.cfg.temperature_scale_c
        disturb = raw_disturb * self.origin_gate.to(raw_disturb.dtype)

        if self.use_in_flight:
            eff_hist_scale = self.hist_coal_scale.clamp(0.0, 1.0)
            gated_hist_coal = hist_coal_delta * self.origin_gate.to(hist_coal_delta.dtype)
            free = disturb + eff_hist_scale * gated_hist_coal
        else:
            free = disturb

        coupling = self.closed_loop_coupling.to(free.dtype)
        return {
            "disturb_delta_c": disturb,
            "hist_coal_delta_c": hist_coal_delta,
            "free_delta_c": free,
            "forced_delta_c": forced["forced_delta_c"],
            "total_coal_delta_c": forced["total_coal_delta_c"],
            "temperature_delta_c": free + coupling * forced["forced_delta_c"],
            "k_t": forced["k_t"],
            "tau_t": forced["tau_t"],
            "prior_loss": forced["prior_loss"],
        }


# ---------------------------------------------------------------------------
# 阶段三：面向控制的防作弊复合 Loss
# ---------------------------------------------------------------------------

class ControlTrajectoryLoss(nn.Module):
    """远期加权 Huber + 红线二次方重罚 + 相位对齐门控振幅匹配 + 起点锚定趋势/平滑 + 双维均值锚定。"""

    def __init__(self, horizon: int = 60, high_threshold_c: float = 950.0,
                 low_threshold_c: float = 880.0, event_weight: float = 1.0,
                 amp_weight: float = 0.5, calm_p2p_c: float = 12.0,
                 trend_weight: float = 0.1, smooth_weight: float = 0.02,
                 mean_bias_weight: float = 0.2,
                 huber_delta_c: float = 5.0,
                 max_step_c: float = 6.0):
        super().__init__()
        self.high, self.low = high_threshold_c, low_threshold_c
        self.event_weight, self.amp_weight = event_weight, amp_weight
        self.calm_p2p_c, self.trend_weight, self.smooth_weight = calm_p2p_c, trend_weight, smooth_weight
        self.mean_bias_weight = mean_bias_weight
        self.huber_delta = huber_delta_c
        self.max_step_c = max_step_c
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
        sq_event = (prediction - target).square() / max(self.huber_delta, 1.0)
        event = (sq_event * extreme * w).sum() / (extreme.sum() + 1e-6)

        p2p_t = (abs_true.max(dim=1).values - abs_true.min(dim=1).values)
        gate = (p2p_t > self.calm_p2p_c).to(prediction.dtype)
        active = gate.sum()
        if prediction.shape[1] > 4:
            body_p2p_p = abs_pred[:, 3:].max(dim=1).values - abs_pred[:, 3:].min(dim=1).values
            body_p2p_t = abs_true[:, 3:].max(dim=1).values - abs_true[:, 3:].min(dim=1).values
        else:
            body_p2p_p = abs_pred.max(dim=1).values - abs_pred.min(dim=1).values
            body_p2p_t = p2p_t
        k_ext = max(2, min(6, prediction.shape[1] // 6))
        span_t = abs_true.topk(k_ext, dim=1).values.mean(dim=1) + (-abs_true).topk(k_ext, dim=1).values.mean(dim=1)
        span_p = abs_pred.topk(k_ext, dim=1).values.mean(dim=1) + (-abs_pred).topk(k_ext, dim=1).values.mean(dim=1)
        w_pk = F.softmax(target / 5.0, dim=1).detach()
        w_vv = F.softmax(-target / 5.0, dim=1).detach()
        aligned_t = (w_pk * target).sum(dim=1) - (w_vv * target).sum(dim=1)
        aligned_p = (w_pk * prediction).sum(dim=1) - (w_vv * prediction).sum(1)
        amp_per_win = (0.65 * (body_p2p_p - body_p2p_t).abs()
                       + 0.35 * (span_p - span_t).abs()
                       + 0.6 * (aligned_p - aligned_t).abs()
                       + 0.5 * (abs_pred.std(dim=1) - abs_true.std(dim=1)).abs())
        amp = (amp_per_win * gate).sum() / (active + 1e-6)

        pred_pad = F.pad(prediction, (1, 0), value=0.0)
        targ_pad = F.pad(target, (1, 0), value=0.0)
        dp = pred_pad[:, 1:] - pred_pad[:, :-1]
        dy = targ_pad[:, 1:] - targ_pad[:, :-1]
        roc_barrier = F.relu(dp.abs() - self.max_step_c).square().mean()
        trend = F.smooth_l1_loss(dp, dy, reduction="mean", beta=1.0) + 0.5 * roc_barrier
        d2 = pred_pad[:, 2:] - 2 * pred_pad[:, 1:-1] + pred_pad[:, :-2]
        smooth = d2.square().mean()

        win_bias = (prediction.mean(dim=1) - target.mean(dim=1)).square().mean()
        step_bias = (prediction.mean(dim=0) - target.mean(dim=0)).square().mean()
        mean_bias = 0.5 * win_bias + 0.5 * step_bias

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

def causal_hilbert_phase(t_hist: np.ndarray, cfg: ForecastConfig) -> np.ndarray:
    """因果希尔伯特/正交相位特征：返回 [history_steps, 2] 的 (sin_phi, cos_phi) 因果估计。"""
    t = np.asarray(t_hist, dtype=np.float32)
    if t.shape != (cfg.history_steps,):
        raise ValueError("温度历史长度不匹配")
    alpha_fast = 2.0 / (min(8, cfg.history_steps) + 1.0)
    alpha_slow = 2.0 / (min(48, cfg.history_steps) + 1.0)
    fast = np.empty_like(t)
    slow = np.empty_like(t)
    fast[0] = t[0]
    slow[0] = t[0]
    for i in range(1, len(t)):
        fast[i] = (1.0 - alpha_fast) * fast[i - 1] + alpha_fast * t[i]
        slow[i] = (1.0 - alpha_slow) * slow[i - 1] + alpha_slow * t[i]
    in_phase = fast - slow
    quad = np.zeros_like(in_phase)
    lag = min(4, len(t) - 1)
    if lag > 0:
        quad[lag:] = (in_phase[lag:] - in_phase[:-lag]) * (3.0 / lag)
    amp = np.sqrt(in_phase ** 2 + quad ** 2 + 1.0)
    return np.stack([in_phase / amp, quad / amp], axis=-1).astype(np.float32)


def build_temp_features(t_hist: np.ndarray, cfg: ForecastConfig) -> np.ndarray:
    """温度相空间：相对当前值、因果一阶差分、因果二阶差分（严格因果）。"""
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


def build_covar_features(covar_hist: np.ndarray, cfg: ForecastConfig) -> np.ndarray:
    """方向一验证：构造其余热工协变量的局部去漂移因果特征。

    输入 covar_hist: [history_steps, M_raw]（如 c5b_valve, sat_temp, decompose_press, alt_fuel, kiln_current, ladder_pressure）
    由于工业现场传感器跨月存在显著基线漂移（如负压漂移 400Pa、窑电流漂移 300A），
    采用窗口内相对于当前时刻的因果偏移 + 因果一阶差分，输出 [history_steps, M_raw]。
    """
    x = np.asarray(covar_hist, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != cfg.history_steps:
        raise ValueError("协变量历史形状应为 [history_steps, C_covar]")
    # 固定物理量纲缩放（分料阀 15%, 二次风温 40℃, 负压 150Pa, 替代燃料 5t/h, 窑电流 100A, 阶梯炉负压 150Pa）
    default_scales = np.array([15.0, 40.0, 150.0, 5.0, 100.0, 150.0], dtype=np.float32)
    c = x.shape[1]
    scales = default_scales[:c] if c <= len(default_scales) else np.ones(c, dtype=np.float32)
    rel = np.clip((x - x[-1:]) / scales[None, :], -5.0, 5.0)
    return rel.astype(np.float32)


def coal_recent_stats(u_hist: np.ndarray, cfg: ForecastConfig) -> tuple[float, float]:
    recent = u_hist[-12:]
    return (float((recent.mean() - cfg.coal_mean_tph) / cfg.coal_scale_tph),
            float((u_hist[-cfg.coal_std_window:] - cfg.coal_mean_tph).std() / cfg.coal_scale_tph))


# ---------------------------------------------------------------------------
# 阶段三：准入 KPI（KPI-1 极值误差 / KPI-2 一对一拐点相位 / KPI-3 振幅还原）
# ---------------------------------------------------------------------------

def find_turns(signs: np.ndarray, min_gap: int = 4) -> np.ndarray:
    idx = []
    last = -10**9
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1] and signs[i] != 0 and signs[i - 1] != 0:
            if i - last >= min_gap:
                idx.append(i); last = i
    return np.array(idx, dtype=int)


def match_turns_one_to_one(pred_signs: np.ndarray, true_signs: np.ndarray,
                           min_gap: int = 4, max_tol_steps: int = 8) -> tuple[list[int], int, int]:
    """同极性一对一拐点匹配：在容差窗口 max_tol_steps 内按时间距离最近进行不放回匹配。"""
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


def kpi_metrics(pred_abs: np.ndarray, true_abs: np.ndarray,
                high_c: float = 950.0, low_c: float = 880.0,
                max_turn_tol_steps: int = 8) -> dict:
    if isinstance(pred_abs, torch.Tensor):
        pred_abs = pred_abs.detach().cpu().numpy()
    if isinstance(true_abs, torch.Tensor):
        true_abs = true_abs.detach().cpu().numpy()
    pred_abs = np.asarray(pred_abs); true_abs = np.asarray(true_abs)
    out = {}
    pk_p = pred_abs.max(axis=1); pk_t = true_abs.max(axis=1)
    vv_p = pred_abs.min(axis=1); vv_t = true_abs.min(axis=1)
    out["peak_valley_mae_c"] = float(np.concatenate([np.abs(pk_p - pk_t), np.abs(vv_p - vv_t)]).mean())
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
    p2p_t = np.ptp(true_abs, axis=1); p2p_p = np.ptp(pred_abs, axis=1)
    out["p2p_ratio"] = float((p2p_p.mean() / p2p_t.mean())) if p2p_t.mean() > 1e-6 else None
    out["mean_truth_p2p_c"] = float(p2p_t.mean())
    out["mean_pred_p2p_c"] = float(p2p_p.mean())
    if pred_abs.shape[1] > 4:
        p2p_t_body = np.ptp(true_abs[:, 3:], axis=1)
        p2p_p_body = np.ptp(pred_abs[:, 3:], axis=1)
        out["body_p2p_ratio"] = float(p2p_p_body.mean() / max(p2p_t_body.mean(), 1e-6))
    return out


def virtual_step_check(model, temp_hist, coal_hist, current_coal, coal_recent_mean,
                       coal_recent_std, step_tph: float = 2.0,
                       covar_hist: torch.Tensor | None = None) -> dict:
    """KPI-4：固定历史，候选煤量 +step / -step 的受迫响应检查（仅离线诊断）。"""
    model.eval()
    with torch.no_grad():
        h_temp, h_coal = model.encoder(temp_hist, coal_hist)
        ctx_parts = [h_temp[:, -1], h_coal[:, -1]]
        if getattr(model, "c_covar", 0) > 0:
            if covar_hist is None:
                covar_hist = torch.zeros(
                    temp_hist.shape[0], model.cfg.history_steps, model.c_covar,
                    device=temp_hist.device, dtype=temp_hist.dtype,
                )
            h_cov = model.encoder.encode_covariates(covar_hist)
            ctx_parts.append(h_cov[:, -1])
        ctx = model.context_head(torch.cat(ctx_parts, dim=-1))
        cur = current_coal.reshape(-1, 1)
        up = cur.expand(-1, model.cfg.horizon_steps) + step_tph
        dn = cur.expand(-1, model.cfg.horizon_steps) - step_tph
        u_hist_m = model._extract_history_coal_tph(coal_hist) if getattr(model, "use_in_flight", False) else None
        base_out = model.coal_response(ctx, cur, cur.expand(-1, model.cfg.horizon_steps),
                                       coal_recent_mean, coal_recent_std, history_coal=u_hist_m)
        f_up = model.coal_response(ctx, cur, up, coal_recent_mean, coal_recent_std,
                                   history_coal=u_hist_m)["forced_delta_c"]
        f_dn = model.coal_response(ctx, cur, dn, coal_recent_mean, coal_recent_std,
                                   history_coal=u_hist_m)["forced_delta_c"]
        k_t, tau_t, _ = model.coal_response.params(ctx, coal_recent_mean, coal_recent_std)
    base_force = base_out["forced_delta_c"]
    du, dd = f_up - base_force, f_dn - base_force
    out = {
        "step_up_endpoint_mean_c": float(du[:, -1].mean()),
        "step_down_endpoint_mean_c": float(dd[:, -1].mean()),
        "sign_correct": bool(du[:, -1].mean() > 0 and dd[:, -1].mean() < 0),
        "monotone_fraction_up": float((du[:, 1:] >= -0.05).float().mean()),
        "monotone_fraction_down": float((dd[:, 1:] <= 0.05).float().mean()),
        "k_t_mean": float(k_t.mean()),
        "k_t_std": float(k_t.std()) if k_t.numel() > 1 else 0.0,
        "tau_t_mean": float(tau_t.mean()),
        "tau_t_std": float(tau_t.std()) if tau_t.numel() > 1 else 0.0,
        "hist_coal_endpoint_std_c": float(base_out["hist_coal_delta_c"][:, -1].std())
        if base_out["hist_coal_delta_c"].shape[0] > 1 else 0.0,
    }
    return out


def residual_target(target_temperature, current_temperature, current_coal,
                    observed_future_coal, fopdt_out) -> torch.Tensor:
    """离线残差标签：真实 ΔT 减去 LPV-FOPDT 对已观测未来煤量的响应。"""
    if current_temperature.ndim == 1:
        current_temperature = current_temperature[:, None]
    return (target_temperature - current_temperature) - fopdt_out["forced_delta_c"]
