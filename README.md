# BDSKY Tree Prior Evaluator

A Python implementation of the **Birth-Death Skyline (BDSKY)** model tree prior from Stadler et al. (PNAS 2012). This module computes the probability density of sampled phylogenetic trees under a piecewise-constant birth-death-sampling process, enabling direct estimation of epidemiological parameters (transmission rates R(t), removal rates δ(t), and sampling rates) from sequence data.

## Features

- **Full Theorem 1 Implementation**: Exact likelihood computation with backward recursion for B_i coefficients
- **Piecewise-Constant Rates**: Support for time-varying birth (λ), death (μ), and sampling (ψ) rates
- **Dual Sampling Modes**: Continuous-time serial sampling and boundary pulse sampling (ρ)
- **Epidemiological Parameterization**: Convert (R, δ, s) to (λ, μ, ψ) using standard epidemiological relationships
- **Numerically Stable**: Log-space arithmetic with epsilon guards to prevent underflow/overflow
- **Efficient**: O(m + N) complexity suitable for MCMC integration
- **Well-Tested**: Comprehensive unit tests covering edge cases and monotonicity properties

## Installation

Requires Python 3.7+ and NumPy:

```bash
pip install numpy
```

## Quick Start

```python
from code import SkylineParams, TreeEvents, BdSkyEvaluator, epidemiology_to_rates

# Define time grid and epidemiological parameters
t = [0.0, 1.0, 3.0]  # Two intervals: [0,1) and [1,3)
R = [2.0, 0.8]        # Basic reproduction numbers
delta = [0.5, 1.0]     # Removal rates
s = [0.2, 0.3]         # Sampling proportions

# Convert to BDSKY rates
lam, mu, psi = epidemiology_to_rates(R, delta, s)

# Create parameters object
params = SkylineParams(
    t=t,
    lam=lam.tolist(),
    mu=mu.tolist(),
    psi=psi.tolist(),
    rho=[0.0, 0.0]  # No boundary sampling
)

# Define tree events
events = TreeEvents(
    x=[0.4],           # Internal node times (transmissions)
    y=[1.5],           # Serial sampling times (not at boundaries)
    N_boundary=[0, 0], # Tips sampled at boundaries
    n_boundary=[0, 0]  # Degree-2 vertices at boundaries
)

# Evaluate log-density
evaluator = BdSkyEvaluator(params)
log_density = evaluator.log_density(events)
print(f"Log-density: {log_density:.6f}")
```

## API Documentation

### `SkylineParams`

Piecewise-constant BDSKY parameters on time intervals.

**Parameters:**
- `t` (List[float]): Time grid of length m+1 with t[0]=0 and t[-1]=t_m
- `lam` (List[float]): Transmission rates λ per interval, length m
- `mu` (List[float]): Death/removal rates μ per interval, length m  
- `psi` (List[float]): Sampling rates ψ per interval, length m
- `rho` (List[float], optional): Pulse sampling probabilities at boundaries t[1..m], length m

**Validation:**
- Ensures t is strictly increasing
- λ > 0, μ ≥ 0, ψ ≥ 0, ρ ∈ [0,1]
- All arrays have consistent lengths

### `TreeEvents`

Sufficient statistics from a sampled phylogenetic tree.

**Parameters:**
- `x` (List[float]): Internal node (transmission/coalescent) times, ascending
- `y` (List[float]): Serial sampling times, ascending, not equal to boundaries
- `N_boundary` (List[int]): Number of tips sampled at each boundary t[i+1], length m
- `n_boundary` (List[int]): Number of degree-2 vertices at each boundary, length m (n_boundary[m-1] must be 0)

### `epidemiology_to_rates(R, delta, s)`

Converts epidemiological parameters to BDSKY rates.

**Parameters:**
- `R` (ArrayLike): Basic reproduction numbers
- `delta` (ArrayLike): Removal rates δ = μ + ψ
- `s` (ArrayLike): Sampling proportions s = ψ/(μ + ψ)

**Returns:**
- `(lam, mu, psi)`: Tuple of numpy arrays for birth, death, and sampling rates

**Formulas:**
- λ = R·δ
- ψ = s·δ  
- μ = δ - ψ

### `BdSkyEvaluator`

Evaluates the BDSKY tree prior density using Theorem 1.

**Methods:**
- `log_density(events, condition_on_at_least_one_sample=True)`: Computes log f[T | λ, μ, ψ, ρ, t, S]
- `density(events, condition_on_at_least_one_sample=True)`: Computes f[T | ·] via exponentiation

## Mathematical Foundation

This implementation is based on **Theorem 1** from Stadler et al. (2012), which provides a closed-form expression for the sampled tree probability density:

```
f[T | λ, μ, ψ, ρ, t; S] = (q₁(0)/(1 - p₁(0))) ×
    ∏ᵢ λₗ₍ₓᵢ₎ qₗ₍ₓᵢ₎(xᵢ) ×
    ∏ᵢ ψₗ₍ᵧᵢ₎ / qₗ₍ᵧᵢ₎(yᵢ) ×
    ∏ᵢ ρᵢᴺᵢ ((1-ρᵢ) qᵢ₊₁(tᵢ))ⁿᵢ
```

Helper functions pᵢ(t) and qᵢ(t) are computed using:
- **Aᵢ** = √((λᵢ - μᵢ - ψᵢ)² + 4λᵢψᵢ)
- **Bᵢ** via backward recursion from pₘ₊₁(tₘ) = 1

## Running Tests

```bash
python tests.py
```

Expected output:
```
....
----------------------------------------------------------------------
Ran 4 tests in 0.000s

OK
```

## Advantages over Coalescent Skyline

- **Epidemiological Interpretation**: Estimates R(t) directly, not just effective population size Nₑ(t)
- **High Sampling Fractions**: Valid when sampling is intensive (outbreaks, large cohorts)
- **Dual Sampling**: Handles both continuous-time serial sampling and contemporaneous boundary sampling
- **Incidence vs Prevalence**: Separates transmission from prevalence dynamics

## Limitations & Assumptions

- **Removal-on-Sampling**: Default assumes sampled individuals stop transmitting (treatment/behavior change)
- **Identifiability**: One parameter can be traded off; requires informative priors or constraints (e.g., constant sampling proportion s)
- **Fixed Grid**: Rate shifts occur at user-specified times tᵢ; does not learn grid automatically

## References

Stadler, T., Kühnert, D., Bonhoeffer, S., & Drummond, A. J. (2013). Birth–death skyline plot reveals temporal changes of epidemic spread in HIV and hepatitis C virus (HCV). *Proceedings of the National Academy of Sciences*, 110(1), 228-233. https://doi.org/10.1073/pnas.1207965110

## License

[Specify your license here]

## Contributing

This is a research implementation of the BDSKY model. For bug reports or feature requests, please open an issue.

---

**Note**: This implementation is designed for integration into Bayesian phylogenetic inference frameworks (e.g., BEAST2) as a tree prior component.
