
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional, List
import numpy as np

ArrayLike = Sequence[float]


@dataclass
class SkylineParams:
    '''Piecewise-constant BDSKY parameters on intervals [t[i], t[i+1]).

    Attributes:
        t: grid of length m+1 with t[0]=0 and t[-1]=tm
        lam: lambda (transmission) per interval, length m
        mu:  mu (becoming noninfectious by non-sampling causes) per interval, length m
        psi: psi (sampling rate) per interval, length m
        rho: rho pulse sampling probability at boundaries t[1..m]; length m.
             (rho[i] applies at time t[i+1] in 0-based indexing; to match paper's rho_i.)
    '''
    t: List[float]
    lam: List[float]
    mu: List[float]
    psi: List[float]
    rho: Optional[List[float]] = None

    def __post_init__(self):
        m = len(self.t) - 1
        assert m >= 1, "Grid t must have at least two points."
        assert len(self.lam) == m and len(self.mu) == m and len(self.psi) == m, "lam/mu/psi must have length m = len(t)-1."
        if self.rho is None:
            self.rho = [0.0]*m
        assert len(self.rho) == m, "rho must have length m (probability at each boundary t[i+1])."
        # validate monotonic grid
        for i in range(1, len(self.t)):
            if not (self.t[i] > self.t[i-1]):
                raise ValueError("t must be strictly increasing.")
        # basic parameter checks
        if any(l <= 0 for l in self.lam):
            raise ValueError("All lambda must be > 0.")
        if any(mv < 0 for mv in self.mu):
            raise ValueError("All mu must be >= 0.")
        if any(pv < 0 for pv in self.psi):
            raise ValueError("All psi must be >= 0.")
        if any(not (0.0 <= r <= 1.0) for r in self.rho):
            raise ValueError("All rho must be in [0,1].")


@dataclass
class TreeEvents:
    '''Sufficient tree event statistics needed by Theorem 1.

    Attributes:
        x: transmission / coalescent times (internal node times), ascending, len = N+n-1
        y: sequential sampling times, ascending, not equal to grid boundaries
        N_boundary: number of tips sampled exactly at each boundary t[i+1]; length m
        n_boundary: number of degree-two vertices at each boundary t[i+1]; length m; n_boundary[m-1] must be 0
    '''
    x: List[float]
    y: List[float]
    N_boundary: List[int]
    n_boundary: List[int]

    def __post_init__(self):
        # sort copies for safety
        self.x = sorted(self.x)
        self.y = sorted(self.y)
        if any(n < 0 for n in self.N_boundary + self.n_boundary):
            raise ValueError("Boundary counts must be non-negative.")


def epidemiology_to_rates(R: ArrayLike, delta: ArrayLike, s: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Convert (R, delta, s) to (lambda, mu, psi) elementwise.
    Equations (paper, Methods): lambda = R*delta; mu = delta - s*delta; psi = s*delta.
    '''
    R = np.asarray(R, dtype=float)
    delta = np.asarray(delta, dtype=float)
    s = np.asarray(s, dtype=float)
    lam = R * delta
    psi = s * delta
    mu = delta - psi
    if np.any(delta < 0) or np.any(s < 0) or np.any(s > 1):
        raise ValueError("Require delta >= 0 and s in [0,1].")
    if np.any(lam <= 0):
        raise ValueError("lambda must be positive (R*delta > 0).")
    return lam, mu, psi


class BdSkyEvaluator:
    '''Evaluator for the BDSKY sampled-tree density of Theorem 1.

    This computes either log-density (default) or density via exponentiating the log.
    '''
    def __init__(self, params: SkylineParams):
        self.p = params
        self.m = len(params.t) - 1
        # Precompute Ai per interval
        self.A = np.sqrt((np.array(self.p.lam) - np.array(self.p.mu) - np.array(self.p.psi))**2 +
                         4.0 * np.array(self.p.lam) * np.array(self.p.psi))
        # will fill Bi and p_i(t_i-1) recursively backward
        self.B = np.zeros(self.m, dtype=float)
        # p_next_at_boundary holds p_{i+1}(t_i) as we go backwards; pm+1(tm) = 1
        p_next_at_boundary = 1.0
        # compute B_i from top interval down to 0 (paper uses 1..m; we use 0..m-1)
        for i in reversed(range(self.m)):
            lam_i, mu_i, psi_i, A_i = self.p.lam[i], self.p.mu[i], self.p.psi[i], self.A[i]
            rho_i = self.p.rho[i]  # applies at boundary t[i+1]
            self.B[i] = ((1.0 - 2.0*(1.0 - rho_i)*p_next_at_boundary) * lam_i + mu_i + psi_i) / A_i
            # update p_next_at_boundary for the next lower boundary using p_i at left boundary t[i]
            p_i_at_ti_minus = self._p_interval(i, self.p.t[i])  # t in [t_i, t_{i+1})
            p_next_at_boundary = p_i_at_ti_minus
        # cache p1(0) and q1(0) and q_{i+1}(t_i)
        self.p1_at_0 = self._p_interval(0, 0.0)
        self.q1_at_0 = self._q_interval(0, 0.0)
        # Precompute q_{i+1}(t_i) for all i
        self.q_ip1_at_ti = np.ones(self.m, dtype=float)
        for i in range(self.m - 1):
            self.q_ip1_at_ti[i] = self._q_interval(i+1, self.p.t[i+1])
        self.q_ip1_at_ti[self.m - 1] = 1.0  # n_boundary[m-1] must be 0

    def _interval_index(self, t: float) -> int:
        '''Return i such that t in [t[i], t[i+1]).'''
        if t >= self.p.t[-1]:
            return self.m - 1
        # binary search
        lo, hi = 0, self.m
        while lo < hi:
            mid = (lo + hi) // 2
            if self.p.t[mid+1] <= t:
                lo = mid + 1
            elif self.p.t[mid] <= t < self.p.t[mid+1]:
                return mid
            else:
                hi = mid
        return min(max(lo, 0), self.m - 1)

    def _p_interval(self, i: int, t: float) -> float:
        '''Compute p_i(t): prob lineage at time t has no sampled descendants by tm.'''
        lam_i, mu_i, psi_i, A_i, B_i = self.p.lam[i], self.p.mu[i], self.p.psi[i], self.A[i], self.B[i]
        ti1 = self.p.t[i+1]  # t_{i+1}
        expo = np.exp(A_i*(ti1 - t))
        num = (lam_i + mu_i + psi_i - A_i) * (expo*(1.0 + B_i) - (1.0 - B_i))
        den = 2.0 * lam_i * (expo*(1.0 + B_i) + (1.0 - B_i))
        return float(num / den)

    def _q_interval(self, i: int, t: float) -> float:
        '''Compute q_i(t): density lineage at time t gives rise to an edge to t_{i+1}.'''
        A_i, B_i = self.A[i], self.B[i]
        ti1 = self.p.t[i+1]
        expo = np.exp(-A_i*(t - ti1))
        denom = (expo*(1.0 + B_i) + (1.0 - B_i))**2
        return float(4.0 * expo / denom)

    def log_density(self, events: TreeEvents, condition_on_at_least_one_sample: bool = True) -> float:
        '''Compute log f[T | lambda, mu, psi, rho, t, (S)].'''
        # Front factor
        if condition_on_at_least_one_sample:
            if self.p1_at_0 >= 1.0:
                return -np.inf
            logf = np.log(self.q1_at_0) - np.log(1.0 - self.p1_at_0)
        else:
            logf = np.log(self.q1_at_0)

        # Π over internal nodes
        for x in events.x:
            i = self._interval_index(x)
            lam_i = self.p.lam[i]
            qlx = self._q_interval(i, x)
            logf += np.log(lam_i) + np.log(max(qlx, 1e-300))

        # Π over sequential samples
        for y in events.y:
            i = self._interval_index(y)
            psi_i = self.p.psi[i]
            qly = self._q_interval(i, y)
            logf += np.log(psi_i) - np.log(max(qly, 1e-300))

        # Boundary terms
        for i in range(self.m):
            rho_i = self.p.rho[i]
            Ni = events.N_boundary[i] if i < len(events.N_boundary) else 0
            ni = events.n_boundary[i] if i < len(events.n_boundary) else 0
            if Ni:
                if rho_i == 0.0 and Ni > 0:
                    return -np.inf
                logf += Ni * np.log(max(rho_i, 1e-300))
            if ni:
                q_ip1_ti = self.q_ip1_at_ti[i]
                term = (1.0 - rho_i) * q_ip1_ti
                logf += ni * np.log(max(term, 1e-300))

        return float(logf)

    def density(self, events: TreeEvents, condition_on_at_least_one_sample: bool = True) -> float:
        return float(np.exp(self.log_density(events, condition_on_at_least_one_sample)))


if __name__ == "__main__":
    t = [0.0, 1.0, 3.0]  # two intervals
    R = [2.0, 0.8]
    delta = [0.5, 1.0]
    s = [0.2, 0.3]
    lam, mu, psi = epidemiology_to_rates(R, delta, s)
    rho = [0.0, 0.0]
    params = SkylineParams(t=t, lam=lam.tolist(), mu=mu.tolist(), psi=psi.tolist(), rho=rho)
    events = TreeEvents(
        x=[0.4],
        y=[1.5],
        N_boundary=[0, 0],
        n_boundary=[0, 0],
    )
    bd = BdSkyEvaluator(params)
    ll = bd.log_density(events)
    print(f"[demo] log-density = {ll:.6f}")
