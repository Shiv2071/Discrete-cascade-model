# Discrete cascade model — simulation

This folder contains a Python implementation of the **discrete stochastic cascade dynamics** on a finite graph (Paper I: *Discrete Stochastic Cascade Dynamics on Finite Graphs with Irreversible Energy Depletion*).

## Model summary

- **State per site:** \(X\), \(Y\) (excitation counts), \(\mathcal{E}\) (capacity energy), \(S\) (structural state).
- **Update order:** interaction → ripple → regime (leakage/explosion) → bonds → energy → structure → diffusion.
- **Asymmetry:** X–Y and X–X interactions allowed; Y–Y forbidden. Intrinsic frequencies \(\omega_X\), \(\omega_Y\) can differ.
- **Ripple:** \(F = |\Delta^2 S|\) (discrete second difference of \(S\) in time).
- **Regimes:** quiescent (\(F \le C\)), leakage (\(C < F < C+\Delta\)), explosive (\(F \ge C+\Delta\)).
- **Main results (Paper I):** Energy monotonicity, finite activity bound, almost-sure absorption.

## Files

| File | Purpose |
|------|--------|
| `cascade_model.py` | Core class `CascadeModel` and `run_simulation()` (1D periodic chain). |
| `run_simulation.py` | Standard run: fixed initial conditions, optional plot (E_tot, X/Y, F_avg). |
| `run_bigbang.py` | Big Bang analogue: hot dense initial conditions, plots E_tot, X/Y, S_total, F_avg. |
| `run_cosmology.py` | Early-universe to “current”: concentrated primordial region (left fraction of chain), then full run; outputs global time series + spatial snapshots (X, Y, E vs site) at several times. |
| `run_radical.py` | **Unproven regime:** two "universes" connected by one bridge; all mass in one. See `RADICAL_README.md`. |
| `run_quantum_bang.py` | **Unproven narrative:** X,Y as quantum particles; fluctuation triggers Big Bang, then observable universe. See `QUANTUM_BANG_README.md`. |
| `requirements.txt` | `numpy`, `matplotlib`. |

## Usage

From this directory:

```bash
# Install
pip install -r requirements.txt

# Standard run (no plot)
python run_simulation.py --steps 5000 --sites 50 --seed 42

# With diagnostics plot
python run_simulation.py --steps 5000 --sites 50 --plot

# Save figures for the thesis/paper (from project root: model_simulation/ is subfolder)
python run_simulation.py --plot -o ../figures/diagnostics.png
python run_bigbang.py -o ../figures/bigbang_analogue.png

# Big Bang analogue (hot dense start, more steps/sites)
python run_bigbang.py --steps 10000 --sites 100

# Early-universe to current: concentrated “primordial” region then evolution
python run_cosmology.py --sites 100 --steps 8000 --primordial-frac 0.25 -o ../figures/cosmology
```

Outputs: console summary (steps, final energy, final X/Y, absorbing); with `--plot`, saves to the path given by `--output` or default `diagnostics.png` / `bigbang_analogue.png`. Use `-o ../figures/...` to write into the project’s `figures/` folder for inclusion in the LaTeX papers.

## Parameters

Defaults in `CascadeModel` match the thesis conventions: \(\alpha_{XY}\), \(\alpha_{XX}\), \(\omega_X\), \(\omega_Y\), \(k_{XY}\), \(k_{XX}\), \(C\), \(\Delta\), \(\lambda\), \(\eta\), \(\kappa\), \(\gamma_1\), \(\gamma_{XX}\), \(\gamma_2\), \(D_X\), \(D_Y\), and Landau bond parameters \(a_0\), \(b\), \(T_c\). Override via keyword arguments to `CascadeModel` or `run_simulation`.

## Relation to the papers

- The **math** (theorems, derivations, key equations) is in the LaTeX papers and in `../mathematical_model.txt` (Sections I–XIII).
- This simulation **implements** the same dynamics and is used to **observe** and **validate** the behaviour (e.g. monotone decrease of total energy, eventual absorption, Big Bang analogue arc).
