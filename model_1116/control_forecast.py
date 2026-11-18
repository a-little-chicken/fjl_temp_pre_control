#!/usr/bin/env python3
"""控制导向 15 分钟分解炉温度轨迹预测模型（三分量严格等权相加 + LPV 批次零均值正交投影 + 连续时间傅里叶-多尺度空洞卷积解码器）。

=============================================================================
模块架构与核心设计说明
=============================================================================
1. 物理-深度学习灰箱三分量严格等权相加（Equal-Weight Physics-Guided Decomposition）：
   模型将未来 15 分钟（60 步 @ 15s）的温度相对变化轨迹 ΔT_{t+1:t+60} 显式解耦为三大物理分量，
   系数全固定为 1.0，无任何后验缩放系数或软饱和折减：
       ΔT_pred = ΔT_dist + 1.0 * ΔT_hist_coal + 1.0 * ΔT_fut_coal
   - `disturb_delta_c` (ΔT_dist)：剥离尾煤热惯性后的外部未测扰动（生料波动、替代燃料热值波动等）自由演化分量；
   - `hist_coal_delta_c` (ΔT_hist_coal)：假设未来尾煤维持 u_t 不变，历史已下发尾煤继续释放的在途余热（原始历史煤量直通递推）；
   - `forced_delta_c` (ΔT_fut_coal)：未来候选尾煤偏离 u_t 激发的净受迫响应（关于未来控制增量严格单调可微）。

2. LPV 物理模块“批次零均值正交投影”（Batch Zero-Mean Orthogonal Projection）：
   - 不断开梯度（不 detach），在 `StateAdaptiveCoalFOPDT.params` 中对工况调制向量施加批次维零均值正交投影：
         P_{1^⊥}(z) = z - mean(z, dim=0, keepdim=True)
   - 前向保证批次平均增益严格锚定在工程先验中心 K_base = 15.0 ℃/(t/h)、时间常数锚定在 τ_base = 200.0 s；
   - 反向传播通过自伴正交投影矩阵自动滤除闭环负相关导致的全局负偏置梯度，完整保留跨工况的相对动态自适应调制
     （K_t ∈ [12.0, 18.0] around 15.0，τ_t ∈ [150.0, 250.0]s around 200.0s）。

3. 双流编码器（近端高频煤量脱敏）+ 连续时间傅里叶 Query 与多尺度 1D 空洞时序卷积残差解码器：
   - 移除全部热工协变量，仅保留高信噪比的 `decompose_temp` + `feed_actual` 双变量；
   - 保留方案 B 编码器侧近端在途窗口高频煤量脱敏（`_decouple_coal_hist_for_encoder`），防止黑盒分支反向抵消物理在途余热；
   - 移除 `SmoothTrendBasisDecoder`，改为 `FourierDilatedConvDecoder`：
     采用多倍频连续时间傅里叶位置编码生成平滑 Query，配合多尺度 1D 空洞时序卷积残差网络
     （dilation = 3 / 6 / 12，感受野精确覆盖 1.5min / 3min / 6min）与起点热惯性门控 `origin_gate`，
     彻底解决双拐点周期波被拍扁的问题，同时允许 `disturb_delta_c` 自如输出高幅值扰动去平衡闭环 `forced_delta_c`。

4. Loss 瘦身正交化（`ControlTrajectoryLoss`）：
   - 移除压制波峰振幅的 `d2.square()` 二阶曲率重罚、`res_reg` 与冗余 `sign_loss`；
   - 保留正交五项：远期加权 Huber (`far_huber`) + 红线越界惩罚 (`event_huber`) + 多尺度一阶差分斜率匹配 (`trend`)
     + 去均值皮尔逊相关 (`corr`) 与相位对齐振幅匹配 (`amplitude`) + 双维均值零偏约束 (`mean_bias`)。

安全边界：推理输入只允许历史测点与起点已知候选未来煤量；
未来真实煤量仅用于离线 Oracle 评价；不访问 IOT、不下发动作。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ===========================================================================
# 1. 全局模型与物理先验超参数配置
# ===========================================================================

@dataclass(frozen=True)
class ForecastConfig:
    """控制导向预测模型的全局只读配置类（仅保留温度+尾煤双变量）。"""

    # --- 时间窗与采样步长配置 ---
    history_steps: int = 240               # 历史观测步数：60 min @ 15s = 240 步
    horizon_steps: int = 60                # 未来预测步数：15 min @ 15s = 60 步
    in_flight_steps: int = 60              # 历史在途余热最小保底步数（实际运行上界由 tau_high 动态派生）
    in_flight_settling_ratio: float = 0.99 # 在途余热收敛准则：默认 99% 热释放完成（对应约 4.605 倍时间常数）
    dt_seconds: float = 15.0               # 离散控制与预测步长（秒）

    # --- Patch-Transformer 主干与空洞卷积解码器参数 ---
    patch_len: int = 8                     # 单个 Patch 覆盖步数：8 步 = 2 min
    patch_stride: int = 4                  # Patch 滑动步长：4 步 = 1 min（共 59 个 Patch）
    d_model: int = 64                      # Transformer 与空洞卷积隐层维度
    n_heads: int = 4                       # 多头注意力头数
    n_layers: int = 2                      # 共享 Transformer Encoder 层数
    decouple_recent_coal: bool = True      # 方案 B：在编码器侧屏蔽近端高频煤量脉冲以防黑盒抢夺在途余热
    coal_attn_dropout: float = 0.25        # 解码器对煤量编码流的注意力丢弃率
    dropout: float = 0.1                   # 基础 Dropout 比例

    # --- 工程先验 FOPDT（LPV 批次零均值正交投影的基准中心与物理边界）---
    gain_c_per_tph: float = 15.0           # 基准稳态增益 K_base: 15.0 ℃/(t/h)
    tau_seconds: float = 200.0             # 基准一阶热惯性时间常数 tau_base: 200.0 s
    dead_steps: int = 1                    # 纯滞后步数：1 步 = 15s（先验合理区间 1~2 步，即 15~30s）
    gain_low: float = 12.0                 # 稳态增益 K_t 物理下限（around 15.0）
    gain_high: float = 18.0                # 稳态增益 K_t 物理上限（around 15.0）
    tau_low: float = 150.0                 # 时间常数 tau_t 物理下限（秒，around 200.0s）
    tau_high: float = 250.0                # 时间常数 tau_t 物理上限（秒，around 200.0s）

    # --- 物理量纲归一化基准 ---
    temperature_scale_c: float = 50.0      # 温度特征缩放尺度（℃）
    coal_mean_tph: float = 10.0            # 尾煤流量训练集均值（t/h，由训练脚本从训练集动态统计覆盖）
    coal_scale_tph: float = 5.0            # 尾煤流量训练集标准差（t/h，由训练脚本从训练集动态统计覆盖）

    # --- 尾煤侧因果统计特征窗口（全部仅使用 <= t 的历史数据）---
    coal_mean_window: int = 120            # 尾煤 30min (120步) 因果滚动均值窗口
    coal_std_window: int = 20              # 尾煤 5min (20步) 因果滚动标准差窗口
    sat_low_tph: float = 5.0               # 尾煤低负荷触底阈值（t/h）
    sat_low_max_steps: int = 12            # 尾煤持续触底饱和计时上限（12 步 = 3min）

    # --- 未来候选煤量增量平滑与线性区配置 ---
    coal_ema_beta: float = 0.5             # 未来候选煤量增量因果指数平滑系数（常数阶跃下保持严格不变）
    coal_linear_step_tph: float = 2.5      # 未来煤量增量线性工作区阈值（±2.5 t/h 内严格恒等线性响应）
    coal_max_extra_tph: float = 2.0        # 超出线性区后的 tanh 软饱和最大额外裕度（t/h）
    origin_ramp_steps: float = 1.5         # 预测起点热惯性门控过渡步数（抑制 t=+15s 非物理跳变）
    max_step_delta_c: float = 6.0          # 单步（15s）物理最大合理温变阈值（℃）


def num_patches(cfg: ForecastConfig) -> int:
    """计算历史窗口切分出的 Patch Token 数量。"""
    return (cfg.history_steps - cfg.patch_len) // cfg.patch_stride + 1


def max_in_flight_steps(cfg: ForecastConfig) -> int:
    """根据 FOPDT 时间常数上限 `tau_high`、纯滞后 `dead_steps` 与热释放收敛比例自动计算最大在途预热步数。"""
    ratio = float(np.clip(cfg.in_flight_settling_ratio, 0.50, 0.999))
    n_tau = float(-np.log(1.0 - ratio))
    derived = cfg.dead_steps + int(np.ceil(n_tau * cfg.tau_high / cfg.dt_seconds))
    return min(cfg.history_steps, max(cfg.in_flight_steps, derived))


# ===========================================================================
# 2. 状态自适应 FOPDT（LPV 批次零均值正交投影）与在途余热状态机
# ===========================================================================

def fopdt_response(
    delta_u: torch.Tensor,
    k: torch.Tensor,
    alpha: torch.Tensor,
    dead_steps: int,
) -> torch.Tensor:
    """标准零初始状态一阶惯性+纯滞后（FOPDT）离散递推（无历史在途分解时的回退函数）。"""
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
    """与 FOPDT 动态评估参数 (k, alpha/tau_t, dead_steps) 对齐的连续状态递推，
    显式分离“历史在途尾煤余热”与“未来动作净受迫响应”，两者严格满足线性叠加恒等式。
    """
    _, m = u_hist.shape
    h = u_fut.shape[1]
    u_cur = u_hist[:, -1:]

    # 1. 根据每个样本当前的 alpha_t 动态计算 FOPDT 有效记忆窗口步数 M_eff [B]
    neg_log_alpha = (-torch.log(alpha.clamp(min=1e-4, max=1.0 - 1e-6))).clamp(min=1e-4)
    ratio = float(np.clip(settling_ratio, 0.50, 0.999))
    n_tau = float(-np.log(1.0 - ratio))
    m_eff = (dead_steps + n_tau / neg_log_alpha).clamp(
        min=float(min(m, dead_steps + 2)), max=float(m)
    )

    # 2. 构造回溯步数坐标 lookback: [1, M]（index 0 对应回溯 m-1 步，index m-1 对应当前时刻回溯 0 步）
    lookback = torch.arange(m - 1, -1, -1, device=u_hist.device, dtype=u_hist.dtype).unsqueeze(0)
    in_window_gate = torch.sigmoid((m_eff[:, None] - lookback) / 1.5)
    in_window_gate = in_window_gate.clone()
    in_window_gate[:, -1] = 1.0

    raw_rel_hist = u_hist - u_cur
    rel_hist = raw_rel_hist * in_window_gate
    rel_fut = u_fut - u_cur
    rel_hold = torch.zeros_like(u_fut)

    rel_full_seq = torch.cat([rel_hist, rel_fut], dim=1)   # [B, M+H]
    rel_hold_seq = torch.cat([rel_hist, rel_hold], dim=1)  # [B, M+H]

    # 3. 围绕每个样本自身的有效记忆窗口起点 (lookback ≈ m_eff - 1) 提取局部稳态基准初值
    anchor_center = (m_eff - 1.0).clamp(min=0.0, max=float(m - 1))[:, None]
    boundary_weight = torch.exp(-0.5 * ((lookback - anchor_center) / 1.5).square())
    boundary_weight = boundary_weight / boundary_weight.sum(dim=1, keepdim=True).clamp(min=1e-6)
    init_rel_u = (rel_hist * boundary_weight).sum(dim=1)

    state_full = k * init_rel_u
    state_hold = state_full.clone()

    # 阶段 1：在历史 M 步 (s = 0 ... M-1) 上递推预热
    for s in range(m):
        eff_idx = max(0, s - dead_steps)
        u_eff = rel_full_seq[:, eff_idx]
        state_full = alpha * state_full + (1.0 - alpha) * k * u_eff
        state_hold = state_full

    base_state_t = state_full
    hist_list, full_list = [], []

    # 阶段 2：向未来 H 步 (step = M ... M+H-1) 平行递推
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
    """批次零均值正交投影算子 P_{1^⊥}(x) = x - mean(x, dim=0, keepdim=True)：
    - 保持计算图完整不断开（不 detach）；
    - 当 batch_size > 1 时，前向消除批次全局偏移，反向正交滤除闭环负相关导致的全局同向偏置梯度，
      仅保留跨样本/跨工况的相对动态差异梯度。
    """
    if x.shape[0] > 1:
        return x - x.mean(dim=0, keepdim=True)
    return x


class StateAdaptiveCoalFOPDT(nn.Module):
    """线性变参数一阶加纯滞后模块（LPV-FOPDT）：
    采用批次零均值正交投影实现跨工况相对动态调制（K_t ∈ [12, 18] around 15.0，τ_t ∈ [150, 250]s around 200s），
    无 gain_calibration、无 saturation_discount，历史煤量无 _soft_saturate 平移直通递推。
    """

    def __init__(self, cfg: ForecastConfig):
        super().__init__()
        if cfg.dead_steps < 1 or cfg.dead_steps > 2:
            raise ValueError("dead_steps 先验区间为 1~2 步 (15~30s)")
        self.cfg = cfg
        self.scheduler = nn.Sequential(
            nn.Linear(cfg.d_model + 2, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        nn.init.zeros_(self.scheduler[2].weight)
        nn.init.zeros_(self.scheduler[2].bias)

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
        """未来煤量增量因果指数平滑（EMA）+ 分段 C1 软饱和：在常数阶跃 (|Δu| <= 2.5 t/h) 下输出与原输入完全恒等。"""
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
        """根据当前评估出的时间常数 tau_t [B] 计算每个样本对应的有效在途窗口步数 M_eff [B]。"""
        ratio = float(np.clip(self.cfg.in_flight_settling_ratio, 0.50, 0.999))
        n_tau = float(-np.log(1.0 - ratio))
        return self.cfg.dead_steps + (n_tau * tau_t / self.cfg.dt_seconds)

    def params(
        self,
        context: torch.Tensor,
        coal_recent_mean: torch.Tensor,
        coal_recent_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """通过批次零均值正交投影生成工况自适应参数 (K_t, tau_t, alpha_t)，不断开梯度。"""
        if coal_recent_mean.ndim == 1:
            coal_recent_mean = coal_recent_mean[:, None]
        if coal_recent_std.ndim == 1:
            coal_recent_std = coal_recent_std[:, None]
        sched_in = torch.cat([context, coal_recent_mean, coal_recent_std], dim=-1)
        raw_logits = self.scheduler(sched_in)
        centered_logits = _batch_zero_mean_projection(raw_logits)
        u = torch.tanh(centered_logits)
        if u.shape[0] > 1:
            u_mean = u.mean(dim=0, keepdim=True)
            mod = (u - u_mean) / (1.0 + u_mean.abs()).detach()
        else:
            mod = u

        cfg = self.cfg
        k_half_span = 0.5 * (cfg.gain_high - cfg.gain_low)
        tau_half_span = 0.5 * (cfg.tau_high - cfg.tau_low)

        k_t = (cfg.gain_c_per_tph + k_half_span * mod[:, 0]).clamp(cfg.gain_low, cfg.gain_high)
        tau_t = (cfg.tau_seconds + tau_half_span * mod[:, 1]).clamp(cfg.tau_low, cfg.tau_high)
        alpha_t = torch.exp(-cfg.dt_seconds / tau_t)
        return k_t, tau_t, alpha_t

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

        k_t, tau_t, alpha_t = self.params(context, coal_recent_mean, coal_recent_std)
        m_eff = self.effective_window_steps(tau_t)
        raw_delta_u = candidate_future_coal - current_coal
        delta_u = self.smooth_coal_increment(raw_delta_u)

        if history_coal is not None:
            if history_coal.ndim != 2:
                raise ValueError("history_coal 形状应为 [batch, M_in_flight]")
            # 历史煤量直接原值进入 FOPDT 状态机，不做 saturation_discount 与 _soft_saturate 平移
            u_cur_ref = history_coal[:, -1:]
            u_fut_eff = u_cur_ref + delta_u
            hist_coal_delta, forced, total_coal_delta = fopdt_in_flight_decomposition(
                history_coal,
                u_fut_eff,
                k_t,
                alpha_t,
                self.cfg.dead_steps,
                settling_ratio=self.cfg.in_flight_settling_ratio,
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
            "in_flight_eff_steps": m_eff,
            "prior_loss": self.prior_loss(k_t, tau_t),
        }


# ===========================================================================
# 3. 双流 Patch-Transformer 编码器与连续时间傅里叶-多尺度空洞卷积解码器
# ===========================================================================

class DualStreamEncoder(nn.Module):
    """双流 Patch 编码器（仅保留温度流与尾煤流，移除全部热工协变量）：
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
    """多尺度 1D 空洞时序卷积残差块：
    在 15s 步长下，采用 kernel_size=3 配合三路空洞率：
    - dilation = 3  -> 感受野跨度 (3-1)*3 * 15s = 90s  (1.5 min)
    - dilation = 6  -> 感受野跨度 (3-1)*6 * 15s = 180s (3.0 min)
    - dilation = 12 -> 感受野跨度 (3-1)*12 * 15s = 360s (6.0 min)
    三路并行融合后经逐点投影与残差连接，完整捕捉单/双拐点周期振荡波形。
    """

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
        # x: [B, H, d_model]
        xt = x.transpose(1, 2)  # [B, d_model, H]
        ms = (self.conv_1m5(xt) + self.conv_3m(xt) + self.conv_6m(xt)) / 3.0
        out = self.fuse(ms).transpose(1, 2)
        return self.norm(x + out)


class FourierDilatedConvDecoder(nn.Module):
    """连续时间傅里叶 Query + 多尺度 1D 空洞时序卷积残差解码器 + 起点门控 origin_gate：
    - 移除低阶解析趋势基函数瓶颈与均值池化低通滤波；
    - 通过连续时间多倍频傅里叶基构造时间查询向量 Query；
    - 通过两层多尺度 1D 空洞卷积残差网络（感受野覆盖 1.5min / 3min / 6min）直接解码高自由度扰动波形；
    - 乘以起点门控 `origin_gate` 保证预测起点平滑衔接，允许 `disturb` 自如输出高幅值双拐点周期波以平衡闭环 `forced`。
    """

    def __init__(self, cfg: ForecastConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.horizon_steps
        tau = torch.linspace(1.0 / h, 1.0, h, dtype=torch.float32)  # [H] in (0, 1]

        # 连续时间傅里叶特征 (k = 1, 2, 3, 4 倍频正余弦 + 线性/二次连续时间坐标，共 10 维)
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

    def build_queries(
        self, context: torch.Tensor, temp_hist: torch.Tensor, query_ctx_proj: nn.Linear
    ) -> torch.Tensor:
        """结合连续时间傅里叶位置编码、全局上下文与近期温度变化率构造解码 Query [B, H, d_model]。"""
        pos_emb = self.query_pos_proj(self.fourier_feats.to(context.dtype))  # [1, H, d_model]
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
    ) -> torch.Tensor:
        h = decoded_seq
        for block in self.dilated_blocks:
            h = block(h)
        raw_disturb = self.out_head(h).squeeze(-1)  # [B, H]
        gate = origin_gate.to(raw_disturb.dtype)
        return raw_disturb * self.cfg.temperature_scale_c * gate


class ControlOrientedForecaster(nn.Module):
    """控制导向预测主模型（三分量严格 1.0 等权相加 + LPV 批次零均值正交投影 + 傅里叶多尺度空洞卷积解码器）。"""

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
        """从原始标准化尾煤特征通道 0 还原覆盖 FOPDT 最大有效时间常数所需的历史尾煤流量序列 (t/h)。"""
        m = min(max_in_flight_steps(self.cfg), coal_hist.shape[1])
        uf = coal_hist[:, -m:, 0]
        return uf * self.cfg.coal_scale_tph + self.cfg.coal_mean_tph

    def _decouple_coal_hist_for_encoder(self, coal_hist: torch.Tensor) -> torch.Tensor:
        """方案 B（编码器近端高频煤量脱敏）：
        当启用在途余热物理模块 (`use_in_flight=True` 且 `decouple_recent_coal=True`) 时，
        将送入神经网络编码器的 `coal_hist` 在近端在途窗口 `[-M:, :]` 内的：
        - 通道 0（瞬时煤量 `uf`）替换为通道 2（30min 慢变滚动均值 `mean`）；
        - 通道 1（单步阶跃脉冲 `pulse`）置为 0.0；
        保留慢变负荷均值、波动率标准差与触底饱和通道，防止黑盒扰动分支吞噬物理在途余热。
        """
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
        """前向传播（三分量严格 1.0 等权相加）：
        ΔT_pred = ΔT_dist + 1.0 * ΔT_hist_coal + 1.0 * ΔT_fut_coal
        """
        # 1. 提取精确近端历史尾煤序列直通物理 FOPDT 状态机
        u_hist_m = self._extract_history_coal_tph(coal_hist) if self.use_in_flight else None

        # 2. 编码器近端高频煤量脱敏与双流 Transformer 编码
        coal_hist_enc = self._decouple_coal_hist_for_encoder(coal_hist)
        h_temp, h_coal = self.encoder(temp_hist, coal_hist_enc)
        context = self.context_head(torch.cat([h_temp[:, -1], h_coal[:, -1]], dim=-1))

        # 3. 物理模块计算在途尾煤余热与未来受迫响应（内部对 K_t, tau_t 执行批次零均值正交投影，不断开梯度）
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

        # 4. 连续时间傅里叶 Query + 跨通道注意力 + 多尺度 1D 空洞时序卷积残差解码
        base_q = self.decoder.build_queries(context, temp_hist, self.query_ctx)
        q = base_q + 0.05 * self.query.expand(temp_hist.shape[0], -1, -1)
        a_t, _ = self.cross_temp(q, h_temp, h_temp, need_weights=False)
        a_c, _ = self.cross_coal(q, h_coal, h_coal, need_weights=False)
        decoded = q + a_t + 0.5 * self.coal_attn_drop(a_c)

        disturb = self.decoder(decoded, self.origin_gate)

        # 5. 三分量严格 1.0 等权相加：ΔT_pred = ΔT_dist + 1.0 * ΔT_hist_coal + 1.0 * ΔT_fut_coal
        free = disturb + hist_coal_delta if self.use_in_flight else disturb
        temperature_delta = free + forced_delta

        return {
            "disturb_delta_c": disturb,
            "hist_coal_delta_c": hist_coal_delta,
            "free_delta_c": free,
            "forced_delta_c": forced_delta,
            "total_coal_delta_c": forced["total_coal_delta_c"],
            "temperature_delta_c": temperature_delta,
            "k_t": forced["k_t"],
            "tau_t": forced["tau_t"],
            "in_flight_eff_steps": forced["in_flight_eff_steps"],
            "prior_loss": forced["prior_loss"],
        }


# ===========================================================================
# 4. 瘦身正交化控制导向复合损失函数
# ===========================================================================

class ControlTrajectoryLoss(nn.Module):
    """瘦身正交化控制导向复合轨迹损失（移除 d2.square() 曲率重罚、res_reg 与冗余 sign_loss）：
    1. `far_huber`  : 远期二次递增权重 Smooth L1 损失；
    2. `event_huber`: 触及高温(>=950℃)/低温(<=880℃)红线时的二次方重罚；
    3. `trend`      : 四级多尺度一阶差分斜率匹配（15s, 1min, 3min, 6min）+ 单步最大温变速率势垒；
    4. `corr`       : 去均值轨迹皮尔逊相关损失（1 - r），对齐波形凹凸走势与峰谷相位；
    5. `amplitude`  : 相位对齐软极值跨度与皮尔逊门控振幅匹配 + 平静期假峰抑制；
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
        self.high, self.low = high_threshold_c, low_threshold_c
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

        # (4) 多尺度一阶差分斜率匹配 (15s, 1min, 3min, 6min) + 起点单步速率势垒
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

        # (5) 相位对齐振幅匹配 + 平静期假峰抑制
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

        phase_gate = pearson_r.detach().clamp(min=0.0, max=1.0)
        amp_per_win = (
            0.65 * (aligned_p - aligned_t).abs()
            + 0.45 * phase_gate * (body_p2p_p - body_p2p_t).abs()
            + 0.30 * phase_gate * (span_p - span_t).abs()
            + 0.40 * phase_gate * (abs_pred.std(dim=1) - abs_true.std(dim=1)).abs()
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
# 5. 严格因果特征工程（仅保留温度+尾煤双变量，离线训练与在线推理共用）
# ===========================================================================

def causal_hilbert_phase(t_hist: np.ndarray, cfg: ForecastConfig) -> np.ndarray:
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
    recent = u_hist[-12:]
    return (
        float((recent.mean() - cfg.coal_mean_tph) / cfg.coal_scale_tph),
        float((u_hist[-cfg.coal_std_window :] - cfg.coal_mean_tph).std() / cfg.coal_scale_tph),
    )


# ===========================================================================
# 6. 控制准入 KPI 评价体系
# ===========================================================================

def find_turns(signs: np.ndarray, min_gap: int = 4) -> np.ndarray:
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
    out["mean_pearson_r"] = float(r_all[active_mask].mean()) if active_mask.any() else float(r_all.mean())
    out["positive_corr_fraction"] = (
        float((r_all[active_mask] > 0.0).mean()) if active_mask.any() else float((r_all > 0.0).mean())
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
    model: ControlOrientedForecaster,
    temp_hist: torch.Tensor,
    coal_hist: torch.Tensor,
    current_coal: torch.Tensor,
    coal_recent_mean: torch.Tensor,
    coal_recent_std: torch.Tensor,
    step_tph: float = 2.0,
) -> dict:
    """KPI-4 虚拟阶跃响应自检（离线物理诊断）。"""
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
        k_t, tau_t, _ = model.coal_response.params(ctx, coal_recent_mean, coal_recent_std)
        m_eff = base_out["in_flight_eff_steps"]

    base_force = base_out["forced_delta_c"]
    du, dd = f_up - base_force, f_dn - base_force
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
        "in_flight_eff_steps_mean": float(m_eff.mean()),
        "in_flight_eff_steps_std": float(m_eff.std()) if m_eff.numel() > 1 else 0.0,
        "hist_coal_endpoint_std_c": float(base_out["hist_coal_delta_c"][:, -1].std())
        if base_out["hist_coal_delta_c"].shape[0] > 1
        else 0.0,
    }


def residual_target(
    target_temperature: torch.Tensor,
    current_temperature: torch.Tensor,
    current_coal: torch.Tensor,
    observed_future_coal: torch.Tensor,
    fopdt_out: dict[str, torch.Tensor],
) -> torch.Tensor:
    del current_coal, observed_future_coal
    if current_temperature.ndim == 1:
        current_temperature = current_temperature[:, None]
    hist_coal = fopdt_out.get("hist_coal_delta_c", 0.0)
    return (target_temperature - current_temperature) - hist_coal - fopdt_out["forced_delta_c"]
