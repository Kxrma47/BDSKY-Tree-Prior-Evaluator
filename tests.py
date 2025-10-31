
import math
import unittest
from code import SkylineParams, TreeEvents, BdSkyEvaluator, epidemiology_to_rates

class TestBDSKY(unittest.TestCase):
    def setUp(self):
        self.t = [0.0, 1.0, 3.0]
        R     = [2.0, 0.8]
        delta = [0.5, 1.0]
        s     = [0.2, 0.3]
        lam, mu, psi = epidemiology_to_rates(R, delta, s)
        self.params = SkylineParams(t=self.t, lam=lam.tolist(), mu=mu.tolist(), psi=psi.tolist(), rho=[0.0, 0.0])

    def test_basic_log_density_finite(self):
        events = TreeEvents(x=[0.4], y=[1.5], N_boundary=[0,0], n_boundary=[0,0])
        ev = BdSkyEvaluator(self.params)
        ll = ev.log_density(events)
        self.assertTrue(math.isfinite(ll))

    def test_more_sampling_increases_prob(self):
        events = TreeEvents(x=[0.4], y=[1.5], N_boundary=[0,0], n_boundary=[0,0])
        ev1 = BdSkyEvaluator(self.params)
        ll1 = ev1.log_density(events)
        p2 = SkylineParams(t=self.t, lam=self.params.lam, mu=self.params.mu, psi=[x*2 for x in self.params.psi], rho=[0.0,0.0])
        ev2 = BdSkyEvaluator(p2)
        ll2 = ev2.log_density(events)
        self.assertGreater(ll2, ll1)

    def test_rho_boundary_effect(self):
        events = TreeEvents(x=[0.4], y=[], N_boundary=[1,0], n_boundary=[0,0])
        p_low  = SkylineParams(t=self.t, lam=self.params.lam, mu=self.params.mu, psi=self.params.psi, rho=[0.1,0.0])
        p_high = SkylineParams(t=self.t, lam=self.params.lam, mu=self.params.mu, psi=self.params.psi, rho=[0.9,0.0])
        ev_low  = BdSkyEvaluator(p_low)
        ev_high = BdSkyEvaluator(p_high)
        ll_low  = ev_low.log_density(events)
        ll_high = ev_high.log_density(events)
        self.assertGreater(ll_high, ll_low)

    def test_density_positive(self):
        events = TreeEvents(x=[0.4], y=[1.5], N_boundary=[0,0], n_boundary=[0,0])
        ev = BdSkyEvaluator(self.params)
        d = ev.density(events)
        self.assertGreater(d, 0.0)

if __name__ == '__main__':
    unittest.main()
