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
    build_coal_features,
    build_temp_features,
    causal_hilbert_phase,
    coal_recent_stats,
    fopdt_in_flight_decomposition,
    fopdt_response,
    kpi_metrics,
    match_turns_one_to_one,
    virtual_step_check,
)


def make_cfg(**kw):
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


class LPVFOPDTTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        self.layer = StateAdaptiveCoalFOPDT(self.cfg)

    def _ctx(self, b=4):
        return torch.randn(b, self.cfg.d_model)

    def test_removed_legacy_scalers_and_discounts(self):
        self.assertFalse(hasattr(self.layer, "calibration"))
        self.assertFalse(hasattr(self.layer, "gain_calibration"))
        self.assertFalse(hasattr(self.layer, "saturation_discount"))

    def test_hold_current_has_zero_forced(self):
        ctx = self._ctx(2)
        cur = torch.tensor([10.0, 12.0])
        fut = cur[:, None].expand(-1, 8)
        out = self.layer(ctx, cur, fut, torch.zeros(2, 1), torch.zeros(2, 1))
        self.assertTrue(torch.all(out["forced_delta_c"] == 0))

    def test_in_flight_decomposition_identities_and_unmodified_history_coal(self):
        k = torch.tensor([15.0, 15.0])
        alpha = torch.tensor([0.9, 0.9])
        u_hist_const = torch.full((2, 16), 10.0)
        u_fut = torch.full((2, 8), 12.0)
        hc, fc, tc = fopdt_in_flight_decomposition(u_hist_const, u_fut, k, alpha, dead_steps=1)
        torch.testing.assert_close(hc, torch.zeros_like(hc), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(tc, fc, atol=1e-5, rtol=1e-5)

        # Large historical coal step (e.g. +8 t/h > 2.5 t/h) is NOT compressed by _soft_saturate
        u_hist_large_step = torch.cat([torch.full((2, 12), 4.0), torch.full((2, 4), 12.0)], dim=1)
        u_hold = torch.full((2, 8), 12.0)
        ctx = self._ctx(2)
        k_layer, _, alpha_layer = self.layer.params(ctx, torch.zeros(2, 1), torch.zeros(2, 1))
        hc_direct, fc_direct, tc_direct = fopdt_in_flight_decomposition(
            u_hist_large_step,
            u_hold,
            k_layer,
            alpha_layer,
            dead_steps=self.cfg.dead_steps,
            settling_ratio=self.cfg.in_flight_settling_ratio,
        )
        out = self.layer(
            ctx,
            torch.full((2,), 12.0),
            u_hold,
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            history_coal=u_hist_large_step,
        )
        torch.testing.assert_close(out["hist_coal_delta_c"], hc_direct, atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.all(fc_direct == 0))
        self.assertTrue(torch.all(hc_direct > 0))
        torch.testing.assert_close(tc_direct, hc_direct, atol=1e-5, rtol=1e-5)

    def test_scheduler_starts_at_prior_center_and_respects_12_18_and_150_250_bounds(self):
        self.assertEqual(self.cfg.gain_low, 12.0)
        self.assertEqual(self.cfg.gain_high, 18.0)
        self.assertEqual(self.cfg.tau_low, 150.0)
        self.assertEqual(self.cfg.tau_high, 250.0)
        ctx = self._ctx(4)
        m = torch.randn(4, 1)
        s = torch.rand(4, 1)
        k, tau, alpha = self.layer.params(ctx, m, s)
        torch.testing.assert_close(k, torch.full_like(k, 15.0))
        torch.testing.assert_close(tau, torch.full_like(tau, 200.0))
        self.assertTrue(torch.all((alpha > 0) & (alpha < 1)))

    def test_batch_zero_mean_orthogonal_projection_filters_global_bias_and_keeps_gradients(self):
        torch.manual_seed(42)
        nn.init.normal_(self.layer.scheduler[2].weight, std=0.5)
        nn.init.constant_(self.layer.scheduler[2].bias, -5.0)  # strong artificial negative bias
        ctx = torch.randn(8, self.cfg.d_model, requires_grad=True)
        m = torch.randn(8, 1)
        s = torch.rand(8, 1)
        k, tau, _ = self.layer.params(ctx, m, s)

        # Forward: batch mean is anchored exactly at (15.0, 200.0) and relative variation is preserved
        self.assertAlmostEqual(float(k.mean().item()), 15.0, places=5)
        self.assertAlmostEqual(float(tau.mean().item()), 200.0, places=4)
        self.assertGreater(float(k.std().item()), 0.1)
        self.assertGreater(float(tau.std().item()), 1.0)
        self.assertTrue(torch.all((k >= 12.0) & (k <= 18.0)))
        self.assertTrue(torch.all((tau >= 150.0) & (tau <= 250.0)))

        # Backward 1: pure uniform global shift loss sum(k) produces ZERO gradient on scheduler (filtered out!)
        uniform_loss = k.sum() + tau.sum()
        uniform_loss.backward(retain_graph=True)
        self.assertAlmostEqual(
            float(self.layer.scheduler[2].weight.grad.abs().max().item()), 0.0, places=6
        )

        # Backward 2: relative cross-condition loss flows gradients without detach into both scheduler and context
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
        self.assertTrue(torch.all(forced_up[0, 1:] > 0))

        fut_dn = torch.full((1, 8), 8.0)
        forced_dn = self.layer(ctx, cur, fut_dn, torch.zeros(1, 1), torch.zeros(1, 1))[
            "forced_delta_c"
        ].detach()
        self.assertTrue(torch.all(forced_dn[0, 1:] < 0))

    def test_prior_loss_penalizes_k_and_tau_boundary_collapse(self):
        k_center = torch.tensor([self.cfg.gain_c_per_tph])
        t_center = torch.tensor([self.cfg.tau_seconds])
        l_center = float(self.layer.prior_loss(k_center, t_center).detach())
        self.assertAlmostEqual(l_center, 0.0, places=6)
        k_bound = torch.tensor([self.cfg.gain_low])
        t_bound = torch.tensor([self.cfg.tau_high])
        l_bound = float(self.layer.prior_loss(k_bound, t_bound).detach())
        self.assertGreater(l_bound, 0.5)


class FeatureTests(unittest.TestCase):
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
    def test_equal_weight_three_component_addition_and_decoder_architecture(self):
        torch.manual_seed(7)
        cfg = make_cfg()
        model = ControlOrientedForecaster(cfg, c_temp=3, c_coal=5)
        self.assertIsInstance(model.decoder, FourierDilatedConvDecoder)
        self.assertFalse(hasattr(model, "hist_coal_scale"))
        self.assertFalse(hasattr(model, "closed_loop_coupling"))
        self.assertFalse(hasattr(model, "trend_decoder"))

        # Check multi-scale dilated convolutions (dilation = 3, 6, 12 for 1.5min / 3min / 6min)
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

        # Exact 1.0 equal-weight identity: ΔT_pred = ΔT_dist + 1.0*ΔT_hist_coal + 1.0*ΔT_fut_coal
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
        # Verify LPV scheduler receives finite gradients without detach
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

        # Dual turning point target wave (full period with both peak +25C and valley -25C inside 60 steps)
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
