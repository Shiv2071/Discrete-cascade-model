# Simulation — Discrete Stochastic Cascade Model

This is the computational exploration surface for the formal system defined in Part I and Part II.

The system is not tuned for success. It is designed to expose:
- **Stable regimes** — quiescent states, energy-limited dynamics
- **Unstable transitions** — explosive cascades, regime switching
- **Collapse conditions** — absorption, hierarchy degeneration under symmetry

## Files

| File | Regime explored | Key observables |
|------|----------------|-----------------|
| `cascade_model.py` | Core `CascadeModel` class | All state variables: X, Y, E, S, F, bonds |
| `run_simulation.py` | Standard dynamics | Energy monotonicity, regime transitions, absorption time |
| `run_bigbang.py` | High-energy initial conditions | Cascade intensity, pair creation rate, rapid depletion |
| `run_cosmology.py` | Spatially concentrated initial state | Wavefront propagation, local vs global energy depletion, spatial snapshots |
| `run_quantum_bang.py` | Fluctuation-triggered dynamics | Stochastic ignition threshold, sensitivity to initial noise |
| `run_radical.py` | Topologically constrained graph | Bottleneck effects, asymmetric spatial absorption |

## What to observe when running

1. **Energy** (`E_tot`) always decreases. It never recovers. This is the built-in thermodynamic arrow (Theorem 1, Part I).
2. **X depletes faster than Y.** The forbidden Y–Y channel means X is consumed through two channels (X–Y and X–X) while Y only through one (X–Y). This asymmetric depletion is the mechanism that drives bond formation (Theorem 2, Part II).
3. **Ripple** (`F_avg`) spikes during explosive regime, then decays. The ripple is the system's internal measure of dynamical volatility.
4. **Structural state** (`S`) accumulates during active dynamics, then freezes permanently at absorption. The frozen pattern is path-dependent.
5. **Absorption is guaranteed** but timing is stochastic. Different seeds produce different histories but the same qualitative arc.

## Usage

```bash
pip install -r requirements.txt

# Standard run with diagnostic plot
python run_simulation.py --steps 5000 --sites 50 --plot

# High-energy regime
python run_bigbang.py --steps 10000 --sites 100

# Spatially concentrated initial state
python run_cosmology.py --sites 100 --steps 8000 --primordial-frac 0.25

# Save figures for the papers
python run_simulation.py --plot -o ../figures/diagnostics.png
python run_bigbang.py -o ../figures/bigbang_analogue.png
python run_cosmology.py -o ../figures/cosmology
```

## Parameters

Defaults in `CascadeModel` match the paper conventions. Override via keyword arguments. The key dimensionless ratios that control qualitative behaviour are documented in Part I, Appendix B (Parameter Sensitivity).

## Relation to the papers

The theorems in Part I and Part II are proven independently of this code. The simulations serve a different purpose: they make the system's behaviour space *visible*. The formal guarantees tell you what must happen; the simulations show you *how* it happens across different initial conditions and graph topologies.
