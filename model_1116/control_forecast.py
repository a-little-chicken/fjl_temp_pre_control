#!/usr/bin/env python3
"""控制导向 15 分钟分解炉温度轨迹预测核心模型（模块化重构版）。

=============================================================================
模块架构与文件章节导览
=============================================================================
- Section 1: 全局只读配置 `ForecastConfig` 与窗口辅助函数
- Section 2: 状态自适应阶跃响应序列生成与矩阵卷积模块 (`StateAdaptiveStepResponse`)
             * 支持多档位二阶 S 型阶跃响应基动态配比，实时输出当前状态下的 60 步 `step_response_full`；
             * 历史在途余热（棕线 `hist_coal_delta_c`）采用完整未截断阶跃序列按正常线性卷积计算；
             * 未来受迫响应（粉线 `forced_delta_c`）支持 4 种初期相位整形模式对比：
               1) `"none"`        : S 型阶跃序列不截断；
               2) `"hard_trunc"`  : 截断 < 20% 的初期过程响应并重标定；
               3) `"sigmoid_gate"`: 围绕 20% 响应阈值的平滑 Sigmoid 脉冲门控；
               4) `"power_soft"`  : 完成度幂次软整形 r_forced = r^γ (默认 γ=2.0，将 20% 压至 4%，处处光滑且保持终值增益 K_t)。
- Section 3: 双流 Patch-Transformer 编码器 (`DualStreamEncoder`) 与
             带 60min FFT 周期延续先验的多尺度空洞卷积解码器 (`FourierDilatedConvDecoder`)
- Section 4: 顶层控制导向预测主模型 (`ControlOrientedForecaster`)
- Section 5: 因果特征构造器与 `loss_and_metrics` 向后兼容重导出

安全边界：推理输入只允许历史测点与起点已知候选未来煤量；
未来真实煤量仅用于离线 Oracle 评价；不访问 IOT、不下发动作。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from loss_and_metrics import (
    ControlTrajectoryLoss,
    base_metrics,
    find_turns,
    kpi_metrics,
    match_turns_one_to_one,
    residual_target,
    virtual_step_check,
)


# ===========================================================================
# Section 1: 全局模型与物理先验超参数配置 (ForecastConfig)
# ===========================================================================

@dataclass(frozen=True)
class ForecastConfig:
    """控制导向预测模型的全局只读配置类（仅保留温度 + 尾煤双变量）。"""

    # --- 1. 时间窗与采样步长配置 ---
    history_steps: int = 240               # 历史观测步数：60 min @ 15s = 240 步
    horizon_steps: int = 60                # 未来预测步数：15 min @ 15s = 60 步
    in_flight_steps: int = 60              # 历史在途余热卷积步数：60 步 = 15 min (900s)
    in_flight_settling_ratio: float = 0.99 # 在途余热收敛比例阈值
    dt_seconds: float = 15.0               # 离散控制与预测步长（秒）

    # --- 2. 编码器与空洞卷积解码器超参数 ---
    patch_len: int = 8                     # 单个 Patch 覆盖步数：8 步 = 2 min
    patch_stride: int = 4                  # Patch 滑动步长：4 步 = 1 min（共 59 个 Patch）
    d_model: int = 64                      # Transformer 与空洞卷积隐层维度
    n_heads: int = 4                       # 多头注意力头数
    n_layers: int = 2                      # 共享 Transformer Encoder 层数
    decouple_recent_coal: bool = False     # 默认保留近端尾煤动态以捕捉早期控制器动作指纹
    coal_attn_dropout: float = 0.25        # 解码器对煤量编码流的注意力丢弃率
    dropout: float = 0.10                  # 基础 Dropout 比例

    # --- 3. 状态自适应阶跃响应序列 (State-Adaptive Step Response Sequence) 参数 ---
    gain_c_per_tph: float = 15.0           # 基准稳态总增益 K_base: 15.0 ℃/(t/h)
    tau_seconds: float = 200.0             # 二阶 S 型阶跃响应中心时间尺度基准 (s)
    dead_steps: int = 1                    # 基础纯滞后步数：1 步 = 15s（区间 1~2 步）
    gain_low: float = 12.0                 # 状态自适应稳态增益下限（around 15.0）
    gain_high: float = 18.0                # 状态自适应稳态增益上限（around 15.0）
    tau_low: float = 150.0                 # 快响应 S 型阶跃基时间尺度下限（秒）
    tau_high: float = 250.0                # 慢响应 S 型阶跃基时间尺度上限（秒）
    n_step_basis: int = 4                  # 快/中快/中慢/慢 4 档 S 型阶跃响应基数量
    forced_shaping_mode: str = "power_soft" # 受迫响应整形: "none" | "hard_trunc" | "sigmoid_gate" | "power_soft"
    forced_trunc_ratio: float = 0.20       # 硬截断或 Sigmoid 软门控的初期响应阈值 (20%)
    forced_power_gamma: float = 2.0        # power_soft 幂次整形指数 γ (r^2 将 20% 初期响应平滑压至 4%)

    # --- 4. 物理量纲归一化与尾煤因果特征窗口 ---
    temperature_scale_c: float = 50.0      # 温度特征缩放尺度（℃）
    coal_mean_tph: float = 10.0            # 尾煤流量训练集均值（t/h）
    coal_scale_tph: float = 5.0            # 尾煤流量训练集标准差（t/h）
    coal_mean_window: int = 120            # 尾煤 30min (120步) 因果滚动均值窗口
    coal_std_window: int = 20              # 尾煤 5min (20步) 因果滚动标准差窗口
    sat_low_tph: float = 5.0               # 尾煤低负荷触底阈值（t/h）
    sat_low_max_steps: int = 12            # 尾煤持续触底饱和计时上限（12 步 = 3min）

    # --- 5. 未来候选煤量增量平滑与起点门控 ---
    coal_ema_beta: float = 0.5             # 未来候选煤量增量因果指数平滑系数
    coal_linear_step_tph: float = 2.5      # 未来煤量增量线性工作区阈值（±2.5 t/h 内严格恒等）
    coal_max_extra_tph: float = 2.0        # 超出线性区后的 tanh 软饱和最大额外裕度（t/h）
    origin_ramp_steps: float = 1.5         # 预测起点热惯性门控过渡步数
    max_step_delta_c: float = 6.0          # 单步（15s）物理最大合理温变阈值（℃）


def num_patches(cfg: ForecastConfig) -> int:
    """计算历史窗口切分出的 Patch Token 数量。"""
    return (cfg.history_steps - cfg.patch_len) // cfg.patch_stride + 1


def max_in_flight_steps(cfg: ForecastConfig) -> int:
    """根据时间尺度上限、纯滞后与热释放收敛比例计算历史在途窗口上界步数。"""
    ratio = float(np.clip(cfg.in_flight_settling_ratio, 0.50, 0.999))
    n_tau = float(-np.log(1.0 - ratio))
    derived = cfg.dead_steps + int(np.ceil(n_tau * cfg.tau_high / cfg.dt_seconds))
    return min(cfg.history_steps, max(cfg.in_flight_steps, derived))


# ===========================================================================
# Section 2: 状态自适应阶跃响应序列生成与矩阵卷积模块 (StateAdaptiveStepResponse)
# ===========================================================================

def build_step_response_basis_bank(
    horizon_steps: int = 60,
    dt_seconds: float = 15.0,
    dead_steps: int = 1,
    tau_grid: tuple[float, ...] = (150.0, 183.3333, 216.6667, 250.0),
) -> torch.Tensor:
    """构造 [n_step_basis, H] 的多档位二阶 S 型归一化阶跃响应基矩阵 R_basis ∈ [0, 1]。
    每一行对应一种热惯性响应速度（从快响应到慢响应），全部严格单调递增且末端归一化为 1.0。
    """
    m = torch.arange(horizon_steps, dtype=torch.float32)
    t_sec = torch.clamp((m - float(dead_steps) + 1.0) * dt_seconds, min=0.0)
    basis_list = []
    for tau_s in tau_grid:
        # 二阶 Erlang/Gamma S 型曲线，满足 t=0 处 r(0)=0 且一阶导数 r'(0)=0
        tau2 = max(float(tau_s) * 0.85, 1.0)
        x = t_sec / tau2
        r = 1.0 - (1.0 + x) * torch.exp(-x)
        r = r / r[-1].clamp(min=1e-6)
        basis_list.append(r)
    return torch.stack(basis_list, dim=0)  # [n_step_basis, H]


def shape_forced_step_response(
    step_seq_full: torch.Tensor,
    mode: str = "power_soft",
    trunc_ratio: float = 0.20,
    power_gamma: float = 2.0,
) -> torch.Tensor:
    """对完整阶跃响应序列 step_seq_full [B, H] 进行初期相位整形，生成专供未来受迫粉线卷积的阶跃序列：
    - "none"        : 不截断，直接使用完整 S 型阶跃响应序列；
    - "hard_trunc"  : 直接硬截断 < trunc_ratio (20%) 的初期响应，并重标定保持终值增益 K_t；
    - "sigmoid_gate": 对脉冲响应序列施加围绕 trunc_ratio 的平滑 Sigmoid 门控并重归一化；
    - "power_soft"  : 完成度幂次软整形 r_forced = r_norm ^ gamma (默认 gamma=2.0，将 20% 平滑压至 4%，处处 C^∞ 光滑且终值严格保持 K_t)。
    """
    if mode == "none":
        return step_seq_full
    k_end = step_seq_full[:, -1:].clamp(min=1e-6)
    r_norm = (step_seq_full / k_end).clamp(0.0, 1.0)
    if mode == "hard_trunc":
        denom = max(1.0 - trunc_ratio, 1e-3)
        return (F.relu(r_norm - trunc_ratio) / denom) * k_end
    if mode == "sigmoid_gate":
        s_prev = F.pad(step_seq_full[:, :-1], (1, 0), value=0.0)
        h_full = (step_seq_full - s_prev).clamp(min=0.0)
        gate = torch.sigmoid((r_norm - trunc_ratio) / 0.04)
        h_gated = h_full * gate
        h_norm = h_gated / h_gated.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.cumsum(h_norm * k_end, dim=-1)
    if mode == "power_soft":
        return (r_norm.pow(power_gamma)) * k_end
    raise ValueError(f"未知的 forced_shaping_mode: {mode}")


def sequence_in_flight_and_forced_convolution(
    u_hist: torch.Tensor,
    u_fut: torch.Tensor,
    step_seq_full: torch.Tensor,
    shaping_mode: str = "power_soft",
    trunc_ratio: float = 0.20,
    power_gamma: float = 2.0,
) -> dict[str, torch.Tensor]:
    """基于每个样本实时生成的阶跃响应序列 step_seq_full [B, H] 执行双通道矩阵卷积：
    1. 历史在途余热 (hist_coal_delta_c, 棕线)：
       使用【完整未截断】的阶跃响应序列 step_seq_full 按正常线性卷积逻辑计算；
    2. 未来受迫响应 (forced_delta_c, 粉线)：
       对阶跃响应序列经 `shape_forced_step_response` 整形后再与未来煤量相对增量做下三角 Toeplitz 矩阵卷积。
    """
    b, m = u_hist.shape
    h = u_fut.shape[1]
    u_cur = u_hist[:, -1:]
    rel_hist = u_hist - u_cur  # [B, M]，末端 rel_hist[:, -1] == 0
    rel_fut = u_fut - u_cur    # [B, H]

    if step_seq_full.ndim == 1:
        step_seq_full = step_seq_full.unsqueeze(0)
    if step_seq_full.shape[0] == 1 and b > 1:
        step_seq_full = step_seq_full.expand(b, -1)

    # --- 1. 历史在途余热 (棕线)：使用完整未截断脉冲序列 h_full 正常卷积 ---
    s_prev = F.pad(step_seq_full[:, :-1], (1, 0), value=0.0)
    h_full = (step_seq_full - s_prev).clamp(min=0.0)  # [B, H]

    if m >= h:
        rh = rel_hist[:, -h:]
    else:
        pad_val = float(rel_hist[:, 0].mean().item()) if m > 0 else 0.0
        rh = F.pad(rel_hist, (h - m, 0), value=pad_val)

    t_idx = torch.arange(h, device=u_hist.device)[:, None]  # [H, 1]
    s_idx = torch.arange(h, device=u_hist.device)[None, :]  # [1, H]
    lag_t = (h + t_idx - s_idx).clamp(0, h - 1)             # [H, H]
    valid_t = (s_idx >= (t_idx + 1)).to(h_full.dtype)       # [H, H]
    lag_0 = (h - 1 - s_idx).clamp(0, h - 1).expand(h, h)    # [H, H]

    w_hist = h_full[:, lag_t] * valid_t.unsqueeze(0) - h_full[:, lag_0]  # [B, H, H]
    hist_coal_delta = torch.bmm(
        w_hist.to(rh.dtype), rh.unsqueeze(-1)
    ).squeeze(-1)  # [B, H]

    # --- 2. 未来受迫响应 (粉线)：使用相位整形后的阶跃序列 step_seq_forced 做下三角卷积 ---
    step_seq_forced = shape_forced_step_response(
        step_seq_full,
        mode=shaping_mode,
        trunc_ratio=trunc_ratio,
        power_gamma=power_gamma,
    )
    sf_prev = F.pad(step_seq_forced[:, :-1], (1, 0), value=0.0)
    h_forced = (step_seq_forced - sf_prev).clamp(min=0.0)  # [B, H]

    lag_fut = (t_idx - s_idx).clamp(0, h - 1)
    causal_mask = (t_idx >= s_idx).to(h_forced.dtype)
    w_fut = h_forced[:, lag_fut] * causal_mask.unsqueeze(0)              # [B, H, H]
    forced_delta = torch.bmm(
        w_fut.to(rel_fut.dtype), rel_fut.unsqueeze(-1)
    ).squeeze(-1)  # [B, H]

    # 同时计算硬截断 <20% 序列供实时观测对比
    step_seq_hard_trunc = shape_forced_step_response(
        step_seq_full, mode="hard_trunc", trunc_ratio=trunc_ratio
    )

    return {
        "hist_coal_delta_c": hist_coal_delta,
        "forced_delta_c": forced_delta,
        "total_coal_delta_c": hist_coal_delta + forced_delta,
        "step_response_full": step_seq_full,
        "step_response_forced": step_seq_forced,
        "step_response_trunc": step_seq_hard_trunc,
        "impulse_response_full": h_full,
        "impulse_response_forced": h_forced,
    }


def fopdt_response(
    delta_u: torch.Tensor,
    k: torch.Tensor,
    alpha: torch.Tensor,
    dead_steps: int,
) -> torch.Tensor:
    """标准零初始状态一阶惯性+纯滞后（FOPDT）离散递推（保留供基础对比与单测调用）。"""
    b, h = delta_u.shape
    state = torch.zeros(b, device=delta_u.device, dtype=delta_u.dtype)
    out = torch.empty_like(delta_u)
    for t in range(h):
        d = t - dead_steps
        if d >= 0:
            state = alpha * state + (1.0 - alpha) * k * delta_u[:, d]
        out[:, t] = state
    return out


def fopdt_in_flight_decomposition(
    u_hist: torch.Tensor,
    u_fut: torch.Tensor,
    k: torch.Tensor,
    alpha: torch.Tensor,
    dead_steps: int,
    settling_ratio: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """一阶参数化在途与受迫连续状态分解（保留供基准恒等式测试调用）。"""
    _, m = u_hist.shape
    h = u_fut.shape[1]
    u_cur = u_hist[:, -1:]

    neg_log_alpha = (-torch.log(alpha.clamp(min=1e-4, max=1.0 - 1e-6))).clamp(min=1e-4)
    ratio = float(np.clip(settling_ratio, 0.50, 0.999))
    n_tau = float(-np.log(1.0 - ratio))
    m_eff = (dead_steps + n_tau / neg_log_alpha).clamp(
        min=float(min(m, dead_steps + 2)), max=float(m)
    )

    lookback = torch.arange(m - 1, -1, -1, device=u_hist.device, dtype=u_hist.dtype).unsqueeze(0)
    in_window_gate = torch.sigmoid((m_eff[:, None] - lookback) / 1.5).clone()
    in_window_gate[:, -1] = 1.0

    rel_hist = (u_hist - u_cur) * in_window_gate
    rel_fut = u_fut - u_cur
    rel_hold = torch.zeros_like(u_fut)

    rel_full_seq = torch.cat([rel_hist, rel_fut], dim=1)
    rel_hold_seq = torch.cat([rel_hist, rel_hold], dim=1)

    anchor_center = (m_eff - 1.0).clamp(min=0.0, max=float(m - 1))[:, None]
    boundary_weight = torch.exp(-0.5 * ((lookback - anchor_center) / 1.5).square())
    boundary_weight = boundary_weight / boundary_weight.sum(dim=1, keepdim=True).clamp(min=1e-6)
    init_rel_u = (rel_hist * boundary_weight).sum(dim=1)

    state_full = k * init_rel_u
    state_hold = state_full.clone()

    for s in range(m):
        eff_idx = max(0, s - dead_steps)
        state_full = alpha * state_full + (1.0 - alpha) * k * rel_full_seq[:, eff_idx]
        state_hold = state_full

    base_state_t = state_full
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


def _batch_zero_mean_projection(x: torch.Tensor) -> torch.Tensor:
    """批次零均值正交投影算子 P_{1^⊥}(x) = x - mean(x, dim=0, keepdim=True)。"""
    if x.shape[0] > 1:
        return x - x.mean(dim=0, keepdim=True)
    return x


class StateAdaptiveCoalFOPDT(nn.Module):
    """状态自适应阶跃响应序列生成器与双通道卷积模块（别名 `StateAdaptiveStepResponse`）：
    - 根据当前工况状态特征 (context, coal_recent_mean, coal_recent_std) 实时生成 60 步
      单调 S 型阶跃响应序列 `step_response_full` [B, H] 与整形后受迫序列 `step_response_forced` [B, H]；
    - 采用批次零均值正交投影保证训练期批次均值严格锚定在 (K=15.0, τ=200.0s)，滤除闭环负相关全局偏置梯度；
    - 维护 `running_logit_mean` 支持在线单样本 (batch_size=1) 实时输出随状态动态变化的 Step Response。
    """

    def __init__(self, cfg: ForecastConfig):
        super().__init__()
        if cfg.dead_steps < 1 or cfg.dead_steps > 2:
            raise ValueError("dead_steps 先验区间为 1~2 步 (15~30s)")
        self.cfg = cfg

        tau_grid = tuple(
            np.linspace(cfg.tau_low, cfg.tau_high, cfg.n_step_basis).tolist()
        )
        basis_bank = build_step_response_basis_bank(
            horizon_steps=cfg.horizon_steps,
            dt_seconds=cfg.dt_seconds,
            dead_steps=cfg.dead_steps,
            tau_grid=tau_grid,
        )
        self.register_buffer("basis_bank", basis_bank)  # [n_step_basis, H]
        self.register_buffer("tau_grid", torch.tensor(tau_grid, dtype=torch.float32))

        self.scheduler = nn.Sequential(
            nn.Linear(cfg.d_model + 2, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        nn.init.zeros_(self.scheduler[2].weight)
        nn.init.zeros_(self.scheduler[2].bias)
        self.register_buffer("running_logit_mean", torch.zeros(1, 2))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def _soft_saturate(self, x: torch.Tensor) -> torch.Tensor:
        """分段 C1 连续软饱和压缩（仅用于未来候选煤量增量，±coal_linear_step_tph 内严格恒等）。"""
        lin = self.cfg.coal_linear_step_tph
        extra = self.cfg.coal_max_extra_tph
        abs_x = x.abs()
        excess = F.relu(abs_x - lin)
        compressed = torch.where(
            abs_x <= lin,
            abs_x,
            lin + extra * torch.tanh(excess / max(extra, 1e-3)),
        )
        return torch.sign(x) * compressed

    def smooth_coal_increment(self, delta_u: torch.Tensor) -> torch.Tensor:
        """未来煤量增量因果指数平滑（EMA）+ 分段 C1 软饱和。"""
        beta = self.cfg.coal_ema_beta
        if beta > 0.0 and delta_u.shape[1] > 1:
            ema = torch.empty_like(delta_u)
            ema[:, 0] = delta_u[:, 0]
            for t in range(1, delta_u.shape[1]):
                ema[:, t] = beta * ema[:, t - 1] + (1.0 - beta) * delta_u[:, t]
        else:
            ema = delta_u
        return self._soft_saturate(ema)

    def effective_window_steps(self, tau_t: torch.Tensor) -> torch.Tensor:
        """根据当前评估出的等效时间尺度 tau_t [B] 计算有效在途窗口步数 M_eff [B]。"""
        ratio = float(np.clip(self.cfg.in_flight_settling_ratio, 0.50, 0.999))
        n_tau = float(-np.log(1.0 - ratio))
        return self.cfg.dead_steps + (n_tau * tau_t / self.cfg.dt_seconds)

    def params(
        self,
        context: torch.Tensor,
        coal_recent_mean: torch.Tensor,
        coal_recent_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """通过批次零均值正交投影生成状态自适应总增益 K_t 与等效时间尺度 tau_t。"""
        if coal_recent_mean.ndim == 1:
            coal_recent_mean = coal_recent_mean[:, None]
        if coal_recent_std.ndim == 1:
            coal_recent_std = coal_recent_std[:, None]
        sched_in = torch.cat([context, coal_recent_mean, coal_recent_std], dim=-1)
        raw_logits = self.scheduler(sched_in)  # [B, 2]

        if raw_logits.shape[0] > 1:
            if self.training:
                with torch.no_grad():
                    self.num_batches_tracked.add_(1)
                    momentum = max(0.05, 1.0 / float(self.num_batches_tracked.item()))
                    self.running_logit_mean.lerp_(
                        raw_logits.detach()
                        .to(self.running_logit_mean.dtype)
                        .mean(dim=0, keepdim=True),
                        momentum,
                    )
            centered_logits = _batch_zero_mean_projection(raw_logits)
            u = torch.tanh(centered_logits)
            u_mean = u.mean(dim=0, keepdim=True)
            mod = (u - u_mean) / (1.0 + u_mean.abs()).detach()
        else:
            centered_logits = raw_logits - self.running_logit_mean.to(raw_logits.dtype)
            mod = torch.tanh(centered_logits)

        cfg = self.cfg
        k_half_span = 0.5 * (cfg.gain_high - cfg.gain_low)
        tau_half_span = 0.5 * (cfg.tau_high - cfg.tau_low)

        k_t = (cfg.gain_c_per_tph + k_half_span * mod[:, 0]).clamp(cfg.gain_low, cfg.gain_high)
        tau_t = (cfg.tau_seconds + tau_half_span * mod[:, 1]).clamp(cfg.tau_low, cfg.tau_high)
        alpha_t = torch.exp(-cfg.dt_seconds / tau_t)
        return k_t, tau_t, alpha_t

    def build_state_step_sequence(
        self,
        context: torch.Tensor,
        coal_recent_mean: torch.Tensor,
        coal_recent_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """根据当前工况状态实时合成 60 步单调 S 型阶跃响应序列 `step_seq_full` [B, H]：
        - 利用状态自适应 `tau_t` 在多档位 S 型基矩阵 `basis_bank` 上做平滑高斯核插值配比；
        - 乘以状态自适应总增益 `k_t`，确保不同工况下的阶跃响应序列在高度和响应快慢上均动态变化，
          且 100% 光滑单调递增。
        """
        k_t, tau_t, alpha_t = self.params(context, coal_recent_mean, coal_recent_std)
        grid = self.tau_grid.to(tau_t.dtype)[None, :]  # [1, n_step_basis]
        sigma_tau = max(0.35 * (self.cfg.tau_high - self.cfg.tau_low), 10.0)
        dist_sq = ((tau_t[:, None] - grid) / sigma_tau).square()
        basis_weights = F.softmax(-dist_sq, dim=-1)  # [B, n_step_basis]

        norm_seq = torch.matmul(basis_weights, self.basis_bank.to(tau_t.dtype))  # [B, H]
        step_seq_full = k_t[:, None] * norm_seq  # [B, H]
        return step_seq_full, k_t, tau_t, alpha_t

    def prior_loss(
        self, k_t: torch.Tensor, tau_t: torch.Tensor | None = None
    ) -> torch.Tensor:
        """物理先验中心温和锚定与上下界排斥势垒损失。"""
        cfg = self.cfg
        k_anchor = ((k_t - cfg.gain_c_per_tph) / cfg.gain_c_per_tph).square().mean()
        k_span = max(cfg.gain_high - cfg.gain_low, 1e-3)
        k_margin = 0.15 * k_span
        k_bound = (
            F.relu(cfg.gain_low + k_margin - k_t).square()
            + F.relu(k_t - (cfg.gain_high - k_margin)).square()
        ).mean() / (k_margin ** 2)
        loss = k_anchor + 0.25 * k_bound
        if tau_t is not None:
            tau_anchor = ((tau_t - cfg.tau_seconds) / cfg.tau_seconds).square().mean()
            t_span = max(cfg.tau_high - cfg.tau_low, 1e-3)
            t_margin = 0.15 * t_span
            t_bound = (
                F.relu(cfg.tau_low + t_margin - tau_t).square()
                + F.relu(tau_t - (cfg.tau_high - t_margin)).square()
            ).mean() / (t_margin ** 2)
            loss = loss + tau_anchor + 0.25 * t_bound
        return loss

    def forward(
        self,
        context: torch.Tensor,
        current_coal: torch.Tensor,
        candidate_future_coal: torch.Tensor,
        coal_recent_mean: torch.Tensor,
        coal_recent_std: torch.Tensor,
        history_coal: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if candidate_future_coal.ndim != 2:
            raise ValueError("candidate_future_coal 形状应为 [batch, horizon]")
        if current_coal.ndim == 1:
            current_coal = current_coal[:, None]
        if not (torch.isfinite(candidate_future_coal).all() and torch.isfinite(current_coal).all()):
            raise ValueError("煤量输入必须有限")

        step_seq_full, k_t, tau_t, _ = self.build_state_step_sequence(
            context, coal_recent_mean, coal_recent_std
        )
        m_eff = self.effective_window_steps(tau_t)
        raw_delta_u = candidate_future_coal - current_coal
        delta_u = self.smooth_coal_increment(raw_delta_u)

        if history_coal is not None:
            if history_coal.ndim != 2:
                raise ValueError("history_coal 形状应为 [batch, M_in_flight]")
            u_cur_ref = history_coal[:, -1:]
            u_fut_eff = u_cur_ref + delta_u
        else:
            u_hist_const = current_coal.expand(-1, self.cfg.horizon_steps)
            history_coal = u_hist_const
            u_fut_eff = current_coal + delta_u

        conv_out = sequence_in_flight_and_forced_convolution(
            history_coal,
            u_fut_eff,
            step_seq_full,
            shaping_mode=self.cfg.forced_shaping_mode,
            trunc_ratio=self.cfg.forced_trunc_ratio,
            power_gamma=self.cfg.forced_power_gamma,
        )

        return {
            "forced_delta_c": conv_out["forced_delta_c"],
            "hist_coal_delta_c": conv_out["hist_coal_delta_c"],
            "total_coal_delta_c": conv_out["total_coal_delta_c"],
            "step_response_full": conv_out["step_response_full"],
            "step_response_forced": conv_out["step_response_forced"],
            "step_response_trunc": conv_out["step_response_trunc"],
            "impulse_response_full": conv_out["impulse_response_full"],
            "impulse_response_forced": conv_out["impulse_response_forced"],
            "k_t": k_t,
            "tau_t": tau_t,
            "in_flight_eff_steps": m_eff,
            "prior_loss": self.prior_loss(k_t, tau_t),
        }


# 提供语义清晰的别名，同时保持旧类名 100% 兼容
StateAdaptiveStepResponse = StateAdaptiveCoalFOPDT


# ===========================================================================
# Section 3: 双流 Patch-Transformer 编码器与带周期延续先验的空洞卷积解码器
# ===========================================================================

class DualStreamEncoder(nn.Module):
    """双流 Patch 编码器（温度流 + 尾煤流）：
    - 温度流与尾煤流各自独立做 Patch 线性投影，共享 Transformer 编码权重；
    - 温度流内嵌因果正交相位特征（同相 In-Phase + 正交 Quadrature）；
    - 输出叠加随时间线性递增的近端偏置（`recency_bias`）。
    """

    def __init__(self, cfg: ForecastConfig, c_temp: int, c_coal: int):
        super().__init__()
        self.cfg = cfg
        self.temp_proj = nn.Linear(cfg.patch_len * c_temp, cfg.d_model)
        self.coal_proj = nn.Linear(cfg.patch_len * c_coal, cfg.d_model)

        self.phase_proj = nn.Linear(cfg.patch_len * 2, cfg.d_model, bias=False)
        nn.init.normal_(self.phase_proj.weight, std=0.01)
        self.position = nn.Parameter(torch.zeros(1, num_patches(cfg), cfg.d_model))
        self.recency_bias = nn.Parameter(torch.zeros(num_patches(cfg)))

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=2 * cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )

    def _causal_phase_channels(self, temp_hist: torch.Tensor) -> torch.Tensor:
        """从温度相对序列中因果提取 15~30min 准周期振荡的同相 (In-Phase) 与正交 (Quadrature) 分量。"""
        rel = temp_hist[:, :, 0:1].transpose(1, 2)  # [B, 1, T]
        k_fast = min(8, self.cfg.history_steps)     # 2 min 快移动平均
        k_slow = min(40, self.cfg.history_steps)    # 10 min 慢移动平均
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

    def forward(
        self, temp_hist: torch.Tensor, coal_hist: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if temp_hist.ndim != 3 or coal_hist.ndim != 3:
            raise ValueError("历史输入必须是 [batch, history, C]")
        if (
            temp_hist.shape[1] != self.cfg.history_steps
            or coal_hist.shape[1] != self.cfg.history_steps
        ):
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


class MultiScaleDilatedResBlock(nn.Module):
    """多尺度 1D 空洞时序卷积残差块（dilation = 3 / 6 / 12，覆盖 1.5min / 3min / 6min 感受野）。"""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.conv_1m5 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=3, dilation=3)
        self.conv_3m = nn.Conv1d(d_model, d_model, kernel_size=3, padding=6, dilation=6)
        self.conv_6m = nn.Conv1d(d_model, d_model, kernel_size=3, padding=12, dilation=12)
        self.fuse = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        ms = (self.conv_1m5(xt) + self.conv_3m(xt) + self.conv_6m(xt)) / 3.0
        out = self.fuse(ms).transpose(1, 2)
        return self.norm(x + out)


class FourierDilatedConvDecoder(nn.Module):
    """连续时间傅里叶 Query + 60min FFT 周期延续先验 + 多尺度空洞卷积残差解码器。"""

    def __init__(self, cfg: ForecastConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.horizon_steps
        tau = torch.linspace(1.0 / h, 1.0, h, dtype=torch.float32)

        fourier_list = [tau, tau.square()]
        for k in (1.0, 2.0, 3.0, 4.0):
            fourier_list.append(torch.sin(k * np.pi * tau))
            fourier_list.append(torch.cos(k * np.pi * tau))
        fourier_feats = torch.stack(fourier_list, dim=-1).unsqueeze(0)  # [1, H, 10]
        self.register_buffer("fourier_feats", fourier_feats)

        self.query_pos_proj = nn.Linear(fourier_feats.shape[-1], cfg.d_model)
        self.recent_kinematics_proj = nn.Linear(3, cfg.d_model)

        self.dilated_blocks = nn.ModuleList(
            [
                MultiScaleDilatedResBlock(cfg.d_model, dropout=cfg.dropout),
                MultiScaleDilatedResBlock(cfg.d_model, dropout=cfg.dropout),
            ]
        )
        self.out_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )
        # 60 分钟主周期谐波外推底稿融合门控（初值设为 0.25）
        self.wave_prior_scale = nn.Parameter(torch.tensor(0.25))

    def extract_harmonic_wave_prior(self, temp_hist: torch.Tensor) -> torch.Tensor:
        """从过去历史温度序列中利用实数 FFT 提取 10~30min 主周期谐波，并向未来解析外推 H 步。"""
        x = temp_hist[:, :, 0].float()
        n_hist = x.shape[1]
        h = self.cfg.horizon_steps
        if n_hist < 16:
            return torch.zeros(x.shape[0], h, device=x.device, dtype=temp_hist.dtype)

        x_centered = x - x.mean(dim=1, keepdim=True)
        spec = torch.fft.rfft(x_centered, n=n_hist, dim=1)
        max_bin = spec.shape[1] - 1
        valid_bins = [k for k in (2, 3, 4, 5, 6) if k <= max_bin]
        if not valid_bins:
            return torch.zeros(x.shape[0], h, device=x.device, dtype=temp_hist.dtype)

        k_bins = torch.tensor(valid_bins, device=x.device, dtype=torch.long)
        spec_band = torch.index_select(spec, dim=1, index=k_bins)  # [B, K_band]
        t_idx = torch.arange(n_hist - 1, n_hist + h, device=x.device, dtype=torch.float32)
        angles = 2.0 * np.pi * k_bins[:, None].float() * t_idx[None, :] / float(n_hist)
        cos_basis = torch.cos(angles)
        sin_basis = torch.sin(angles)

        wave_all = (2.0 / float(n_hist)) * (
            torch.matmul(spec_band.real, cos_basis) - torch.matmul(spec_band.imag, sin_basis)
        )
        wave_fut = (wave_all[:, 1:] - wave_all[:, 0:1]).to(temp_hist.dtype)
        return wave_fut

    def build_queries(
        self, context: torch.Tensor, temp_hist: torch.Tensor, query_ctx_proj: nn.Linear
    ) -> torch.Tensor:
        """结合连续时间傅里叶位置编码、全局上下文与近期温度变化率构造解码 Query [B, H, d_model]。"""
        pos_emb = self.query_pos_proj(self.fourier_feats.to(context.dtype))
        rel_t = temp_hist[:, :, 0]
        slope_1m = -rel_t[:, -min(4, rel_t.shape[1])]
        slope_3m = -rel_t[:, -min(12, rel_t.shape[1])]
        slope_6m = -rel_t[:, -min(24, rel_t.shape[1])]
        kin = self.recent_kinematics_proj(
            torch.stack([slope_1m, slope_3m, slope_6m], dim=-1)
        ).unsqueeze(1)
        return pos_emb + query_ctx_proj(context).unsqueeze(1) + kin

    def forward(
        self,
        decoded_seq: torch.Tensor,
        origin_gate: torch.Tensor,
        temp_hist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = decoded_seq
        for block in self.dilated_blocks:
            h = block(h)
        raw_disturb = self.out_head(h).squeeze(-1)
        if temp_hist is not None:
            wave_prior = self.extract_harmonic_wave_prior(temp_hist)
            raw_disturb = raw_disturb + self.wave_prior_scale.clamp(0.0, 0.8) * wave_prior
        gate = origin_gate.to(raw_disturb.dtype)
        return raw_disturb * self.cfg.temperature_scale_c * gate


# ===========================================================================
# Section 4: 顶层控制导向预测主模型 (ControlOrientedForecaster)
# ===========================================================================

class ControlOrientedForecaster(nn.Module):
    """控制导向预测主模型：
    - 三分量严格 1.0 等权相加：ΔT_pred = ΔT_dist + 1.0 * ΔT_hist_coal + 1.0 * ΔT_fut_coal；
    - 实时生成并返回当前状态下的 60 步阶跃响应序列 `step_response_full` 与 `step_response_forced`。
    """

    def __init__(
        self,
        cfg: ForecastConfig,
        c_temp: int = 3,
        c_coal: int = 5,
        use_in_flight: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.use_in_flight = use_in_flight
        self.encoder = DualStreamEncoder(cfg, c_temp, c_coal)
        self.coal_response = StateAdaptiveCoalFOPDT(cfg)

        self.query = nn.Parameter(torch.zeros(1, cfg.horizon_steps, cfg.d_model))
        self.context_head = nn.Linear(cfg.d_model * 2, cfg.d_model)
        self.query_ctx = nn.Linear(cfg.d_model, cfg.d_model)

        self.cross_temp = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.cross_coal = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.coal_attn_drop = nn.Dropout(cfg.coal_attn_dropout)
        self.decoder = FourierDilatedConvDecoder(cfg)

        steps = torch.arange(1, cfg.horizon_steps + 1, dtype=torch.float32)
        origin_gate = 1.0 - torch.exp(-steps / max(cfg.origin_ramp_steps, 1e-3))
        self.register_buffer("origin_gate", origin_gate.view(1, -1))

    def _extract_history_coal_tph(self, coal_hist: torch.Tensor) -> torch.Tensor:
        """从原始标准化尾煤特征通道 0 还原过去历史尾煤流量序列 (t/h)。"""
        m = min(max_in_flight_steps(self.cfg), coal_hist.shape[1])
        uf = coal_hist[:, -m:, 0]
        return uf * self.cfg.coal_scale_tph + self.cfg.coal_mean_tph

    def _decouple_coal_hist_for_encoder(self, coal_hist: torch.Tensor) -> torch.Tensor:
        """可选的编码器近端高频煤量脱敏（默认 decouple_recent_coal=False 不屏蔽）。"""
        if not (self.use_in_flight and self.cfg.decouple_recent_coal):
            return coal_hist
        m = min(max_in_flight_steps(self.cfg), coal_hist.shape[1])
        masked = coal_hist.clone()
        if coal_hist.shape[2] >= 3:
            masked[:, -m:, 0] = coal_hist[:, -m:, 2]
        if coal_hist.shape[2] >= 2:
            masked[:, -m:, 1] = 0.0
        return masked

    def forward(
        self,
        temp_hist: torch.Tensor,
        coal_hist: torch.Tensor,
        current_coal: torch.Tensor,
        candidate_future_coal: torch.Tensor,
        coal_recent_mean: torch.Tensor,
        coal_recent_std: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # 1. 提取近端历史尾煤序列供阶跃序列卷积模块计算历史在途余热
        u_hist_m = self._extract_history_coal_tph(coal_hist) if self.use_in_flight else None

        # 2. 双流 PatchTST 编码历史温度与尾煤上下文
        coal_hist_enc = self._decouple_coal_hist_for_encoder(coal_hist)
        h_temp, h_coal = self.encoder(temp_hist, coal_hist_enc)
        context = self.context_head(torch.cat([h_temp[:, -1], h_coal[:, -1]], dim=-1))

        # 3. 状态自适应阶跃序列生成与双通道矩阵卷积（正常在途棕线 + 相位整形受迫粉线）
        forced = self.coal_response(
            context,
            current_coal,
            candidate_future_coal,
            coal_recent_mean,
            coal_recent_std,
            history_coal=u_hist_m,
        )
        hist_coal_delta = forced["hist_coal_delta_c"]
        forced_delta = forced["forced_delta_c"]

        # 4. 连续时间傅里叶 Query + 跨通道注意力 + 60min 周期延续空洞卷积解码
        base_q = self.decoder.build_queries(context, temp_hist, self.query_ctx)
        q = base_q + 0.05 * self.query.expand(temp_hist.shape[0], -1, -1)
        a_t, _ = self.cross_temp(q, h_temp, h_temp, need_weights=False)
        a_c, _ = self.cross_coal(q, h_coal, h_coal, need_weights=False)
        decoded = q + a_t + 0.5 * self.coal_attn_drop(a_c)

        disturb = self.decoder(decoded, self.origin_gate, temp_hist=temp_hist)

        # 5. 三分量严格 1.0 等权相加，并实时输出当前状态下的阶跃响应序列
        free = disturb + hist_coal_delta if self.use_in_flight else disturb
        temperature_delta = free + forced_delta

        return {
            "disturb_delta_c": disturb,
            "hist_coal_delta_c": hist_coal_delta,
            "free_delta_c": free,
            "forced_delta_c": forced_delta,
            "total_coal_delta_c": forced["total_coal_delta_c"],
            "temperature_delta_c": temperature_delta,
            "step_response_full": forced["step_response_full"],
            "step_response_forced": forced["step_response_forced"],
            "step_response_trunc": forced["step_response_trunc"],
            "k_t": forced["k_t"],
            "tau_t": forced["tau_t"],
            "in_flight_eff_steps": forced["in_flight_eff_steps"],
            "prior_loss": forced["prior_loss"],
        }


# ===========================================================================
# Section 5: 严格因果特征工程函数（离线训练与在线推理共用）
# ===========================================================================

def causal_hilbert_phase(t_hist: np.ndarray, cfg: ForecastConfig) -> np.ndarray:
    """因果希尔伯特正交相位提取：返回 [history_steps, 2] 的归一化 (in_phase, quadrature)。"""
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
    """构造 3 通道因果温度相空间特征 [rel, dt1, dt2]。"""
    t = np.asarray(t_hist, dtype=np.float32)
    if t.shape != (cfg.history_steps,):
        raise ValueError("温度历史长度不匹配")
    inv_scale = 1.0 / cfg.temperature_scale_c
    rel = (t - t[-1]) * inv_scale
    dt1 = np.empty_like(t)
    dt1[1:] = np.diff(t) * inv_scale
    dt1[0] = 0.0
    dt2 = np.empty_like(t)
    dt2[1:] = np.diff(dt1)
    dt2[0] = 0.0
    return np.stack([rel, dt1, dt2], axis=-1)


def build_coal_features(u_hist: np.ndarray, cfg: ForecastConfig) -> np.ndarray:
    """构造 5 通道因果尾煤与负荷包络特征 [uf, pulse, mean_30m, std_5m, sat_low_counter]。"""
    u = np.asarray(u_hist, dtype=np.float32)
    if u.shape != (cfg.history_steps,):
        raise ValueError("煤量历史长度不匹配")
    s = 1.0 / cfg.coal_scale_tph
    uf = (u - cfg.coal_mean_tph) * s
    pulse = np.zeros_like(u)
    pulse[1:] = np.diff(u)
    pulse = np.clip(pulse * s, -5.0, 5.0)

    cs = np.r_[0.0, np.cumsum(u, dtype=np.float64)]
    cs2 = np.r_[0.0, np.cumsum(u.astype(np.float64) ** 2)]
    end = np.arange(1, len(u) + 1)

    def _rolling_from_cumsum(window: int) -> tuple[np.ndarray, np.ndarray]:
        start = np.maximum(0, end - window)
        count = end - start
        mean_x = (cs[end] - cs[start]) / count
        var = np.maximum((cs2[end] - cs2[start]) / count - mean_x ** 2, 0.0)
        return mean_x.astype(np.float32), np.sqrt(var).astype(np.float32)

    mean, _ = _rolling_from_cumsum(min(cfg.coal_mean_window, cfg.history_steps))
    mean = (mean - cfg.coal_mean_tph) * s
    _, std = _rolling_from_cumsum(min(cfg.coal_std_window, cfg.history_steps))
    std = np.maximum(std, 1e-4) * s

    below = u < cfg.sat_low_tph
    pos = np.arange(len(u))
    last_reset = np.maximum.accumulate(np.where(~below, pos, -1))
    run = np.where(below, pos - last_reset, 0)
    sat = np.minimum(run, cfg.sat_low_max_steps).astype(np.float32) / cfg.sat_low_max_steps
    return np.stack([uf, pulse, mean, std, sat], axis=-1)


def coal_recent_stats(u_hist: np.ndarray, cfg: ForecastConfig) -> tuple[float, float]:
    """提取最近 3 分钟尾煤均值与最近 5 分钟尾煤波动率标准差。"""
    recent = u_hist[-12:]
    return (
        float((recent.mean() - cfg.coal_mean_tph) / cfg.coal_scale_tph),
        float((u_hist[-cfg.coal_std_window :] - cfg.coal_mean_tph).std() / cfg.coal_scale_tph),
    )


__all__ = [
    "ControlOrientedForecaster",
    "ControlTrajectoryLoss",
    "DualStreamEncoder",
    "ForecastConfig",
    "FourierDilatedConvDecoder",
    "MultiScaleDilatedResBlock",
    "StateAdaptiveCoalFOPDT",
    "StateAdaptiveStepResponse",
    "base_metrics",
    "build_coal_features",
    "build_step_response_basis_bank",
    "build_temp_features",
    "causal_hilbert_phase",
    "coal_recent_stats",
    "find_turns",
    "fopdt_in_flight_decomposition",
    "fopdt_response",
    "kpi_metrics",
    "match_turns_one_to_one",
    "max_in_flight_steps",
    "num_patches",
    "residual_target",
    "sequence_in_flight_and_forced_convolution",
    "shape_forced_step_response",
    "virtual_step_check",
]
