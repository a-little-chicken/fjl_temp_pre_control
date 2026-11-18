#!/usr/bin/env python3
"""控制导向分解炉温度轨迹预测：模块化单元测试套件。

测试结构：
1. `StepResponseSequenceTests`: 验证状态自适应阶跃响应序列生成、单调性、跨状态差异（含 batch_size=1 在线推理）、
   历史在途正常卷积与未来受迫 4 种相位整形模式（"none", "hard_trunc", "sigmoid_gate", "power_soft"）。
2. `FeatureAndDecoderTests`   : 验证严格因果特征构造、低煤量饱和计数器与 60min FFT 周期延续先验。
3. `LossTests`                : 验证正交复合损失（含 FFT 幅值谱匹配）、平静期门控与 Step-0 脉冲防作弊。
4. `ModelEndToEndTests`       : 验证三分量 1.0 等权相加、实时 Step Response 输出、双拐点周期波拟合与 KPI 评价。
"""
import unittest

import numpy as np
import torch
from torch import nn

from control_forecast import (
    ControlOrientedForecaster,
    ControlTrajectoryLoss,
    ForecastConfig,
    FourierDilatedConvDecoder,
    StateAdaptiveCoalFOPDT,
    StateAdaptiveStepResponse,
    build_coal_features,
    build_step_response_basis_bank,
    build_temp_features,
    causal_hilbert_phase,
    coal_recent_stats,
    fopdt_in_flight_decomposition,
    fopdt_response,
    kpi_metrics,
    match_turns_one_to_one,
    sequence_in_flight_and_forced_convolution,
    shape_forced_step_response,
    virtual_step_check,
)


def make_cfg(**kw) -> ForecastConfig:
    base = dict(
        history_steps=32,
        horizon_steps=8,
        in_flight_steps=16,
        patch_len=4,
        patch_stride=2,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )
    base.update(kw)
    return ForecastConfig(**base)


class StepResponseSequenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        self.layer = StateAdaptiveCoalFOPDT(self.cfg)

    def _ctx(self, b=4):
        return torch.randn(b, self.cfg.d_model)

    def test_removed_legacy_scalers_and_discounts_and_alias_exists(self):
        self.assertIs(StateAdaptiveStepResponse, StateAdaptiveCoalFOPDT)
        self.assertFalse(hasattr(self.layer, "calibration"))
        self.assertFalse(hasattr(self.layer, "gain_calibration"))
        self.assertFalse(hasattr(self.layer, "saturation_discount"))

    def test_hold_current_has_zero_forced(self):
        ctx = self._ctx(2)
        cur = torch.tensor([10.0, 12.0])
        fut = cur[:, None].expand(-1, 8)
        out = self.layer(ctx, cur, fut, torch.zeros(2, 1), torch.zeros(2, 1))
        self.assertTrue(torch.all(out["forced_delta_c"] == 0))
        self.assertEqual(out["step_response_full"].shape, (2, 8))
        self.assertEqual(out["step_response_forced"].shape, (2, 8))

    def test_in_flight_normal_convolution_and_four_forced_shaping_modes(self):
        h = 60
        basis = build_step_response_basis_bank(horizon_steps=h)
        self.assertEqual(basis.shape, (4, h))
        # Basis rows are strictly monotone in [0, 1] and end at 1.0
        self.assertTrue(torch.all(basis[:, 1:] >= basis[:, :-1]))
        torch.testing.assert_close(basis[:, -1], torch.ones(4), atol=1e-5, rtol=1e-5)

        step_seq_full = 15.0 * basis[1:2]  # [1, 60]
        u_hist_step = torch.cat([torch.full((1, 59), 4.0), torch.full((1, 1), 12.0)], dim=1)
        u_fut_step = torch.full((1, h), 14.0)  # +2 t/h future step from u_cur=12.0

        out_none = sequence_in_flight_and_forced_convolution(
            u_hist_step, u_fut_step, step_seq_full, shaping_mode="none"
        )
        out_hard = sequence_in_flight_and_forced_convolution(
            u_hist_step, u_fut_step, step_seq_full, shaping_mode="hard_trunc", trunc_ratio=0.20
        )
        out_sig = sequence_in_flight_and_forced_convolution(
            u_hist_step, u_fut_step, step_seq_full, shaping_mode="sigmoid_gate", trunc_ratio=0.20
        )
        out_pow = sequence_in_flight_and_forced_convolution(
            u_hist_step, u_fut_step, step_seq_full, shaping_mode="power_soft", power_gamma=2.0
        )

        # 1. 历史在途余热 (棕线) 在所有 4 种模式下完全相同（均采用完整未截断序列正常卷积）
        for other in (out_hard, out_sig, out_pow):
            torch.testing.assert_close(
                out_none["hist_coal_delta_c"], other["hist_coal_delta_c"], atol=1e-5, rtol=1e-5
            )
        self.assertTrue(torch.all(out_none["hist_coal_delta_c"] > 0))

        # 2. 4 种模式的受迫阶跃响应在 15min 终点 (step 59) 均 100% 保持相同总增益 K=15.0
        for d_out in (out_none, out_hard, out_sig, out_pow):
            seq_f = d_out["step_response_forced"]
            self.assertTrue(torch.all(seq_f[:, 1:] >= seq_f[:, :-1] - 1e-6))
            self.assertAlmostEqual(float(seq_f[0, -1].item()), 15.0, places=4)

        # 3. 初期 (<20% 阶段，如第 6 步 = 1.5min)：
        #    hard_trunc 严格为 0；power_soft (r^2) 远小于不截断 none
        self.assertEqual(float(out_hard["step_response_forced"][0, 6].item()), 0.0)
        self.assertLess(
            float(out_pow["step_response_forced"][0, 8].item()),
            0.35 * float(out_none["step_response_forced"][0, 8].item()),
        )

    def test_realtime_step_response_varies_across_states_in_batch_and_single_eval(self):
        torch.manual_seed(42)
        cfg = ForecastConfig(history_steps=32, horizon_steps=20, d_model=16)
        layer = StateAdaptiveCoalFOPDT(cfg)
        nn.init.normal_(layer.scheduler[0].weight, std=0.5)
        nn.init.normal_(layer.scheduler[2].weight, std=0.5)
        nn.init.constant_(layer.scheduler[2].bias, -3.0)

        # 训练期 Batch (B=8): 更新 running_logit_mean，验证不同状态生成不同 Step Response 且批次均值锚定
        layer.train()
        ctx = torch.randn(8, cfg.d_model, requires_grad=True)
        m = torch.linspace(-1.5, 1.5, 8).unsqueeze(1)
        s = torch.linspace(0.2, 1.8, 8).unsqueeze(1)
        step_seq, k_t, tau_t, _ = layer.build_state_step_sequence(ctx, m, s)

        self.assertEqual(step_seq.shape, (8, 20))
        self.assertAlmostEqual(float(k_t.mean().item()), 15.0, places=5)
        self.assertAlmostEqual(float(tau_t.mean().item()), 200.0, places=4)
        # 单调递增且跨状态显著变化
        self.assertTrue(torch.all(step_seq[:, 1:] >= step_seq[:, :-1] - 1e-6))
        self.assertGreater(float(step_seq[:, -1].std().item()), 0.2)
        self.assertGreater(float(step_seq[:, 8].std().item()), 0.1)

        # 在线单样本推理期 (eval mode, batch_size=1)：不同状态依然实时输出不同的 Step Response
        layer.eval()
        with torch.no_grad():
            seq_state_a, k_a, tau_a, _ = layer.build_state_step_sequence(
                ctx[0:1], m[0:1], s[0:1]
            )
            seq_state_b, k_b, tau_b, _ = layer.build_state_step_sequence(
                ctx[-1:], m[-1:], s[-1:]
            )
        self.assertEqual(seq_state_a.shape, (1, 20))
        self.assertNotAlmostEqual(float(k_a.item()), float(k_b.item()), places=2)
        self.assertNotAlmostEqual(float(tau_a.item()), float(tau_b.item()), places=2)
        self.assertGreater(float((seq_state_a - seq_state_b).abs().max().item()), 0.2)

    def test_batch_zero_mean_orthogonal_projection_filters_global_bias_and_keeps_gradients(self):
        torch.manual_seed(42)
        nn.init.normal_(self.layer.scheduler[2].weight, std=0.5)
        nn.init.constant_(self.layer.scheduler[2].bias, -5.0)
        ctx = torch.randn(8, self.cfg.d_model, requires_grad=True)
        m = torch.randn(8, 1)
        s = torch.rand(8, 1)
        k, tau, _ = self.layer.params(ctx, m, s)

        self.assertAlmostEqual(float(k.mean().item()), 15.0, places=5)
        self.assertAlmostEqual(float(tau.mean().item()), 200.0, places=4)
        self.assertGreater(float(k.std().item()), 0.1)
        self.assertGreater(float(tau.std().item()), 1.0)
        self.assertTrue(torch.all((k >= 12.0) & (k <= 18.0)))
        self.assertTrue(torch.all((tau >= 150.0) & (tau <= 250.0)))

        uniform_loss = k.sum() + tau.sum()
        uniform_loss.backward(retain_graph=True)
        self.assertAlmostEqual(
            float(self.layer.scheduler[2].weight.grad.abs().max().item()), 0.0, places=6
        )

        self.layer.zero_grad()
        ctx.grad = None
        weights = torch.linspace(-1.0, 1.0, 8)
        rel_loss = (k * weights).sum() + (tau * weights).sum()
        rel_loss.backward()
        self.assertGreater(float(self.layer.scheduler[2].weight.grad.abs().max().item()), 1e-3)
        self.assertIsNotNone(ctx.grad)
        self.assertGreater(float(ctx.grad.abs().max().item()), 1e-3)

    def test_positive_and_negative_step_response(self):
        ctx = self._ctx(1)
        cur = torch.tensor([10.0])
        fut_up = torch.full((1, 8), 12.0)
        forced_up = self.layer(ctx, cur, fut_up, torch.zeros(1, 1), torch.zeros(1, 1))[
            "forced_delta_c"
        ].detach()
        self.assertEqual(float(forced_up[0, 0]), 0.0)
        self.assertTrue(torch.all(forced_up[0, 2:] > 0))

        fut_dn = torch.full((1, 8), 8.0)
        forced_dn = self.layer(ctx, cur, fut_dn, torch.zeros(1, 1), torch.zeros(1, 1))[
            "forced_delta_c"
        ].detach()
        self.assertTrue(torch.all(forced_dn[0, 2:] < 0))

    def test_prior_loss_penalizes_k_and_tau_boundary_collapse(self):
        k_center = torch.tensor([self.cfg.gain_c_per_tph])
        t_center = torch.tensor([self.cfg.tau_seconds])
        l_center = float(self.layer.prior_loss(k_center, t_center).detach())
        self.assertAlmostEqual(l_center, 0.0, places=6)
        k_bound = torch.tensor([self.cfg.gain_low])
        t_bound = torch.tensor([self.cfg.tau_high])
        l_bound = float(self.layer.prior_loss(k_bound, t_bound).detach())
        self.assertGreater(l_bound, 0.5)


class FeatureAndDecoderTests(unittest.TestCase):
    def test_features_causal_and_shapes(self):
        cfg = make_cfg()
        t = torch.linspace(900, 940, cfg.history_steps).numpy()
        u = 10 + torch.sin(torch.linspace(0, 6, cfg.history_steps)).numpy()
        tf = build_temp_features(t, cfg)
        uf = build_coal_features(u, cfg)
        self.assertEqual(tf.shape, (cfg.history_steps, 3))
        self.assertEqual(uf.shape, (cfg.history_steps, 5))
        t2 = t.copy()
        t2[-1] += 10.0
        tf2 = build_temp_features(t2, cfg)
        self.assertTrue((tf[:-1, 1:] == tf2[:-1, 1:]).all())
        self.assertFalse((tf[-1] == tf2[-1]).all())
        hp = causal_hilbert_phase(t, cfg)
        hp2 = causal_hilbert_phase(t2, cfg)
        self.assertEqual(hp.shape, (cfg.history_steps, 2))
        np.testing.assert_allclose(hp[:-1], hp2[:-1], atol=1e-6)
        m, s = coal_recent_stats(u, cfg)
        self.assertIsInstance(m, float)
        self.assertIsInstance(s, float)

    def test_low_coal_saturation_counter_is_causal_and_bounded(self):
        cfg = make_cfg(sat_low_tph=5.0, sat_low_max_steps=4)
        u = np.array([4, 4, 4, 4, 4, 10, 10, 10] + [10] * 24, dtype=np.float32)
        feat = build_coal_features(u, cfg)
        np.testing.assert_allclose(feat[:5, 4], [0.25, 0.5, 0.75, 1.0, 1.0])
        self.assertEqual(float(feat[5, 4]), 0.0)
        self.assertTrue(np.all((feat[:, 4] >= 0) & (feat[:, 4] <= 1)))

    def test_fft_harmonic_wave_prior_extrapolates_periodic_sine(self):
        cfg = ForecastConfig(history_steps=240, horizon_steps=60, d_model=16)
        dec = FourierDilatedConvDecoder(cfg)
        # 构造周期为 15min (60步，共 4 个完整周期) 的余弦振荡信号
        t_all = torch.arange(300, dtype=torch.float32)
        wave = torch.cos(2.0 * np.pi * t_all / 60.0)  # 周期 60 步 (k=4)
        hist_rel = (wave[:240] - wave[239]).view(1, 240, 1)
        expected_fut = wave[240:] - wave[239]
        prior_fut = dec.extract_harmonic_wave_prior(hist_rel)[0]
        corr = float(torch.corrcoef(torch.stack([prior_fut, expected_fut]))[0, 1].item())
        self.assertGreater(corr, 0.98)


class LossTests(unittest.TestCase):
    def test_streamlined_loss_removes_d2_smooth_and_sign_and_matches_exact_sum(self):
        loss = ControlTrajectoryLoss(
            horizon=8,
            event_weight=1.0,
            corr_weight=0.85,
            trend_weight=0.45,
            amp_weight=0.5,
            mean_bias_weight=0.2,
        )
        target = torch.zeros(2, 8)
        pred = torch.full((2, 8), 5.0)
        out = loss(pred, target, torch.full((2,), 900.0))
        self.assertNotIn("smooth", out)
        self.assertNotIn("sign", out)
        self.assertEqual(float(out["mean_bias"]), 25.0)
        expected_total = (
            out["far_huber"]
            + 1.0 * out["event_huber"]
            + 0.85 * out["corr"]
            + 0.45 * out["trend"]
            + 0.5 * out["amplitude"]
            + 0.2 * out["mean_bias"]
        )
        self.assertAlmostEqual(float(out["total"]), float(expected_total), places=5)

    def test_gated_amplitude_ignores_calm_windows(self):
        loss = ControlTrajectoryLoss(horizon=8, calm_p2p_c=12.0)
        calm_true = torch.zeros(1, 8)
        calm_pred = torch.full((1, 8), 3.0)
        r = loss(calm_pred, calm_true, torch.zeros(1))
        self.assertEqual(float(r["amp_active_fraction"]), 0.0)
        wavy_true = torch.sin(torch.linspace(0, 4, 8)).unsqueeze(0) * 30
        wavy_pred = torch.zeros(1, 8)
        r2 = loss(wavy_pred, wavy_true, torch.zeros(1))
        self.assertGreater(float(r2["amplitude"]), 0.0)
        self.assertEqual(float(r2["amp_active_fraction"]), 1.0)

    def test_origin_anchored_trend_rejects_step0_spike_hacking(self):
        loss = ControlTrajectoryLoss(horizon=8, calm_p2p_c=12.0)
        cur_t = torch.full((1,), 900.0)
        target = torch.tensor([[0.0, 2.0, 5.0, 9.0, 12.0, 8.0, 2.0, -4.0]])
        spike_hack = torch.tensor([[28.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.0]])
        out_good = loss(target, target, cur_t)
        out_hack = loss(spike_hack, target, cur_t)
        self.assertGreater(float(out_hack["trend"]), float(out_good["trend"]) + 10.0)
        self.assertGreater(float(out_hack["amplitude"]), 5.0)

    def test_phase_aligned_amplitude_penalizes_anti_phase_wave(self):
        loss = ControlTrajectoryLoss(horizon=8, calm_p2p_c=10.0)
        cur_t = torch.full((1,), 900.0)
        target = torch.sin(torch.linspace(0.2, 3.0, 8)).unsqueeze(0) * 20.0 - 10.0
        in_phase = target.clone()
        anti_phase = -target.clone()
        self.assertAlmostEqual(float(loss(in_phase, target, cur_t)["amplitude"]), 0.0, places=5)
        self.assertGreater(float(loss(anti_phase, target, cur_t)["amplitude"]), 10.0)


class ModelEndToEndTests(unittest.TestCase):
    def test_equal_weight_three_component_addition_and_realtime_step_response_outputs(self):
        torch.manual_seed(7)
        cfg = make_cfg()
        model = ControlOrientedForecaster(cfg, c_temp=3, c_coal=5)
        self.assertIsInstance(model.decoder, FourierDilatedConvDecoder)

        b0 = model.decoder.dilated_blocks[0]
        self.assertEqual(b0.conv_1m5.dilation, (3,))
        self.assertEqual(b0.conv_3m.dilation, (6,))
        self.assertEqual(b0.conv_6m.dilation, (12,))

        B = 4
        th = torch.randn(B, cfg.history_steps, 3)
        uh = torch.randn(B, cfg.history_steps, 5)
        cur_u = torch.full((B,), 10.0)
        fut_u = torch.full((B, 8), 11.5)
        m = torch.zeros(B)
        s = torch.ones(B)
        out = model(th, uh, cur_u, fut_u, m, s)
        self.assertEqual(out["temperature_delta_c"].shape, (B, 8))
        self.assertEqual(out["step_response_full"].shape, (B, 8))
        self.assertEqual(out["step_response_forced"].shape, (B, 8))
        self.assertEqual(out["step_response_trunc"].shape, (B, 8))

        expected_free = out["disturb_delta_c"] + out["hist_coal_delta_c"]
        expected_pred = expected_free + out["forced_delta_c"]
        torch.testing.assert_close(out["free_delta_c"], expected_free, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(out["temperature_delta_c"], expected_pred, atol=1e-6, rtol=1e-6)

        loss_fn = ControlTrajectoryLoss(horizon=8)
        target = torch.randn(B, 8) * 5
        cur_t = torch.full((B,), 915.0)
        lp = loss_fn(out["temperature_delta_c"], target, cur_t)
        (lp["total"] + out["prior_loss"]).backward()
        self.assertTrue(
            all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        )
        self.assertIsNotNone(model.coal_response.scheduler[0].weight.grad)

        kpi = kpi_metrics(
            cur_t[:, None] + out["temperature_delta_c"],
            cur_t[:, None] + target,
        )
        self.assertIn("p2p_ratio", kpi)
        self.assertIn("body_p2p_ratio", kpi)
        self.assertIn("peak_valley_mae_c", kpi)
        vs = virtual_step_check(model, th, uh, cur_u, m, s, step_tph=2.0)
        self.assertTrue(vs["sign_correct"])
        self.assertGreater(vs["monotone_fraction_up"], 0.95)
        self.assertGreater(vs["monotone_fraction_down"], 0.95)
        self.assertAlmostEqual(vs["k_t_mean"], 15.0, places=4)

    def test_fourier_dilated_decoder_fits_dual_turning_point_wave(self):
        torch.manual_seed(11)
        cfg = ForecastConfig(
            history_steps=60,
            horizon_steps=60,
            in_flight_steps=30,
            patch_len=8,
            patch_stride=4,
            d_model=32,
            n_heads=4,
            n_layers=1,
            dropout=0.0,
        )
        model = ControlOrientedForecaster(cfg, c_temp=3, c_coal=5)
        B = 2
        th = torch.randn(B, cfg.history_steps, 3)
        uh = torch.zeros(B, cfg.history_steps, 5)
        cur_u = torch.full((B,), 10.0)
        fut_u = torch.full((B, cfg.horizon_steps), 10.0)
        m = torch.zeros(B)
        s = torch.ones(B)

        t_grid = torch.linspace(0.1, 2.0 * np.pi, cfg.horizon_steps)
        target = (25.0 * torch.sin(t_grid)).unsqueeze(0).expand(B, -1)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        for _ in range(80):
            opt.zero_grad()
            pred = model(th, uh, cur_u, fut_u, m, s)["temperature_delta_c"]
            loss = (pred - target).square().mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = model(th, uh, cur_u, fut_u, m, s)["temperature_delta_c"]
            rmse = float(torch.sqrt((pred - target).square().mean()).item())
            p2p = float((pred.max(dim=1).values - pred.min(dim=1).values).mean().item())
        self.assertLess(rmse, 2.0)
        self.assertGreater(p2p, 46.0)

    def test_one_to_one_turn_matching_reports_misses_honestly(self):
        true_signs = np.array([1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1])
        pred_signs = np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        errs, nb, na = match_turns_one_to_one(pred_signs, true_signs, min_gap=3, max_tol_steps=3)
        self.assertEqual(nb, 3)
        self.assertEqual(na, 1)
        self.assertEqual(len(errs), 1)
        self.assertEqual(errs[0], 1)


if __name__ == "__main__":
    unittest.main()
