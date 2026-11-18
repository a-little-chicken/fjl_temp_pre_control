import unittest

import numpy as np
import torch

from control_forecast import (
    ControlOrientedForecaster, ControlTrajectoryLoss, ForecastConfig,
    StateAdaptiveCoalFOPDT, build_coal_features, build_temp_features,
    calibrate_closed_loop_coupling, calibrate_gain_scale, causal_hilbert_phase,
    coal_recent_stats, fopdt_response, kpi_metrics, match_turns_one_to_one,
    virtual_step_check,
)


def make_cfg(**kw):
    base = dict(history_steps=32, horizon_steps=8, patch_len=4, patch_stride=2,
                d_model=16, n_heads=4, n_layers=1, dropout=0.0)
    base.update(kw)
    return ForecastConfig(**base)


class LPVFOPDTTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        self.layer = StateAdaptiveCoalFOPDT(self.cfg)

    def _ctx(self, b=2):
        return torch.randn(b, self.cfg.d_model)

    def test_hold_current_has_zero_forced(self):
        ctx = self._ctx()
        cur = torch.tensor([10.0, 12.0])
        fut = cur[:, None].expand(-1, 8)
        out = self.layer(ctx, cur, fut, torch.zeros(2, 1), torch.zeros(2, 1))
        self.assertTrue(torch.all(out["forced_delta_c"] == 0))

    def test_scheduler_starts_at_prior_center(self):
        ctx = self._ctx(3)
        m = torch.randn(3, 1); s = torch.rand(3, 1)
        k, tau, _ = self.layer.params(ctx, m, s)
        torch.testing.assert_close(k, torch.full_like(k, self.cfg.gain_c_per_tph))
        torch.testing.assert_close(tau, torch.full_like(tau, self.cfg.tau_seconds))

    def test_gain_stays_in_physical_range(self):
        ctx = self._ctx(4)
        m = torch.randn(4, 1); s = torch.abs(torch.randn(4, 1))
        k, tau, alpha = self.layer.params(ctx, m, s)
        self.assertTrue(torch.all(k >= self.cfg.gain_low - 1e-5))
        self.assertTrue(torch.all(k <= self.cfg.gain_high + 1e-5))
        self.assertTrue(torch.all(tau >= self.cfg.tau_low - 1e-4))
        self.assertTrue(torch.all(tau <= self.cfg.tau_high + 1e-4))
        self.assertTrue(torch.all((alpha > 0) & (alpha < 1)))

    def test_positive_step_respects_dead_time_and_sign(self):
        ctx = self._ctx(1)
        cur = torch.tensor([10.0]); fut = torch.full((1, 8), 12.0)
        forced = self.layer(ctx, cur, fut, torch.zeros(1, 1), torch.zeros(1, 1))["forced_delta_c"].detach()
        self.assertEqual(float(forced[0, 0]), 0.0)
        self.assertTrue(torch.all(forced[0, 1:] > 0))
        self.assertLess(float(forced[0, -1]), 30.0 * self.cfg.gain_high / self.cfg.gain_c_per_tph)

    def test_negative_step_negative_response(self):
        ctx = self._ctx(1)
        cur = torch.tensor([10.0]); fut = torch.full((1, 8), 8.0)
        forced = self.layer(ctx, cur, fut, torch.zeros(1, 1), torch.zeros(1, 1))["forced_delta_c"].detach()
        self.assertTrue(torch.all(forced[0, 1:] < 0))

    def test_dead_time_two_steps(self):
        cfg = make_cfg(dead_steps=2)
        layer = StateAdaptiveCoalFOPDT(cfg)
        ctx = torch.randn(1, cfg.d_model)
        forced = layer(ctx, torch.tensor([10.0]), torch.full((1, 4), 11.0),
                       torch.zeros(1, 1), torch.zeros(1, 1))["forced_delta_c"].detach()
        self.assertEqual(float(forced[0, 0]), 0.0)
        self.assertEqual(float(forced[0, 1]), 0.0)
        self.assertGreater(float(forced[0, 2]), 0.0)

    def test_fopdt_response_matches_closed_form_constant_input(self):
        k = torch.tensor([10.0]); alpha = torch.tensor([0.9])
        du = torch.full((1, 5), 2.0)
        r = fopdt_response(du, k, alpha, dead_steps=1)
        self.assertLess(float(r[0, -1]), float(10.0 * 2.0))
        self.assertTrue(torch.all(r[0, 1:] >= 0))

    def test_prior_loss_penalizes_k_and_tau_boundary_collapse(self):
        k_center = torch.tensor([self.cfg.gain_c_per_tph])
        t_center = torch.tensor([self.cfg.tau_seconds])
        l_center = float(self.layer.prior_loss(k_center, t_center).detach())
        self.assertAlmostEqual(l_center, 0.0, places=6)
        k_bound = torch.tensor([self.cfg.gain_low])
        t_bound = torch.tensor([self.cfg.tau_high])
        l_bound = float(self.layer.prior_loss(k_bound, t_bound).detach())
        self.assertGreater(l_bound, 0.8)

    def test_smooth_coal_increment_preserves_small_steps_and_compresses_extremes(self):
        step_small = torch.full((1, 8), 2.0)
        torch.testing.assert_close(self.layer.smooth_coal_increment(step_small), step_small)
        step_huge = torch.full((1, 8), 15.0)
        compressed = self.layer.smooth_coal_increment(step_huge)
        self.assertTrue(torch.all(compressed <= self.cfg.coal_linear_step_tph + self.cfg.coal_max_extra_tph + 1e-4))
        self.assertTrue(torch.all(compressed > self.cfg.coal_linear_step_tph))

    def test_calibration_handles_negative_closed_loop_correlation(self):
        free = torch.zeros(16, 8)
        truth = torch.randn(16, 8) * 10.0
        forced_neg = -0.8 * truth
        c_gain = calibrate_gain_scale(free, forced_neg, truth)
        self.assertAlmostEqual(c_gain, 0.3, places=4)
        c_cl = calibrate_closed_loop_coupling(free, forced_neg, truth)
        self.assertEqual(c_cl, 0.0)


class FeatureTests(unittest.TestCase):
    def test_features_causal_and_shapes(self):
        cfg = make_cfg()
        t = torch.linspace(900, 940, cfg.history_steps).numpy()
        u = 10 + torch.sin(torch.linspace(0, 6, cfg.history_steps)).numpy()
        tf = build_temp_features(t, cfg)
        uf = build_coal_features(u, cfg)
        self.assertEqual(tf.shape, (cfg.history_steps, 3))
        self.assertEqual(uf.shape, (cfg.history_steps, 5))
        t2 = t.copy(); t2[-1] += 10.0
        tf2 = build_temp_features(t2, cfg)
        self.assertTrue((tf[:-1, 1:] == tf2[:-1, 1:]).all())
        self.assertFalse((tf[-1] == tf2[-1]).all())
        hp = causal_hilbert_phase(t, cfg)
        hp2 = causal_hilbert_phase(t2, cfg)
        self.assertEqual(hp.shape, (cfg.history_steps, 2))
        np.testing.assert_allclose(hp[:-1], hp2[:-1], atol=1e-6)
        m, s = coal_recent_stats(u, cfg)
        self.assertIsInstance(m, float); self.assertIsInstance(s, float)

    def test_low_coal_saturation_counter_is_causal_and_bounded(self):
        cfg = make_cfg(sat_low_tph=5.0, sat_low_max_steps=4)
        u = np.array([4, 4, 4, 4, 4, 10, 10, 10] + [10] * 24, dtype=np.float32)
        feat = build_coal_features(u, cfg)
        np.testing.assert_allclose(feat[:5, 4], [0.25, 0.5, 0.75, 1.0, 1.0])
        self.assertEqual(float(feat[5, 4]), 0.0)
        self.assertTrue(np.all((feat[:, 4] >= 0) & (feat[:, 4] <= 1)))


class LossTests(unittest.TestCase):
    def test_mean_bias_penalty_detects_constant_offset(self):
        loss = ControlTrajectoryLoss(horizon=8, mean_bias_weight=0.2)
        target = torch.zeros(2, 8)
        pred = torch.full((2, 8), 5.0)
        out = loss(pred, target, torch.full((2,), 900.0))
        self.assertEqual(float(out["mean_bias"]), 25.0)
        self.assertAlmostEqual(float(out["total"] - out["far_huber"] - out["event_huber"]
                                     - 0.5 * out["amplitude"] - 0.1 * out["trend"]
                                     - 0.02 * out["smooth"]), 5.0, places=4)

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

    def test_origin_anchored_loss_rejects_step0_spike_hacking(self):
        loss = ControlTrajectoryLoss(horizon=8, calm_p2p_c=12.0)
        cur_t = torch.full((1,), 900.0)
        target = torch.tensor([[0.0, 2.0, 5.0, 9.0, 12.0, 8.0, 2.0, -4.0]])
        spike_hack = torch.tensor([[28.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.0]])
        out_good = loss(target, target, cur_t)
        out_hack = loss(spike_hack, target, cur_t)
        self.assertGreater(float(out_hack["trend"]), float(out_good["trend"]) + 10.0)
        self.assertGreater(float(out_hack["smooth"]), float(out_good["smooth"]) + 100.0)
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
    def test_forward_gradients_and_kpi(self):
        torch.manual_seed(7)
        cfg = make_cfg()
        model = ControlOrientedForecaster(cfg, c_temp=3, c_coal=5)
        B = 3
        th = torch.randn(B, cfg.history_steps, 3)
        uh = torch.randn(B, cfg.history_steps, 5)
        cur_u = torch.full((B,), 10.0)
        fut_u = torch.full((B, 8), 11.0)
        m = torch.zeros(B); s = torch.ones(B)
        out = model(th, uh, cur_u, fut_u, m, s)
        self.assertEqual(out["temperature_delta_c"].shape, (B, 8))
        loss_fn = ControlTrajectoryLoss(horizon=8)
        target = torch.randn(B, 8) * 5
        cur_t = torch.full((B,), 915.0)
        lp = loss_fn(out["temperature_delta_c"], target, cur_t)
        lp["total"].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        kpi = kpi_metrics(cur_t[:, None] + out["temperature_delta_c"],
                          cur_t[:, None] + target)
        self.assertIn("p2p_ratio", kpi)
        self.assertIn("body_p2p_ratio", kpi)
        self.assertIn("peak_valley_mae_c", kpi)
        vs = virtual_step_check(model, th, uh, cur_u, m, s, step_tph=2.0)
        self.assertTrue(vs["sign_correct"])
        self.assertGreater(vs["monotone_fraction_up"], 0.95)
        self.assertGreater(vs["monotone_fraction_down"], 0.95)
        self.assertIn("k_t_mean", vs)
        self.assertGreater(vs["k_t_mean"], 0.0)

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
