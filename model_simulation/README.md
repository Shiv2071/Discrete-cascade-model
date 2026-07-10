# Simulation: Discrete Stochastic Cascade Model

This is the computational exploration surface for the formal system defined in Part I and Part II.

The system is not tuned for success. It is designed to expose:
- **Stable regimes**: quiescent states, energy-limited dynamics
- **Unstable transitions**: explosive cascades, regime switching
- **Collapse conditions**: absorption, hierarchy degeneration under symmetry

## Files

| File | Regime explored | Key observables |
|------|----------------|-----------------|
| `cascade_model.py` | Core `CascadeModel` class | All state variables: X, Y, E, S, F, bonds |
| `run_simulation.py` | Standard dynamics | Energy monotonicity, regime transitions, absorption time |
| `run_high_energy.py` | High-energy initial conditions | Cascade intensity, pair creation rate, rapid depletion |
| `run_spatial_concentration.py` | Spatially concentrated initial state | Wavefront propagation, local vs global energy depletion, spatial snapshots |
| `run_minimal_seed.py` | Fluctuation-triggered dynamics | Stochastic ignition threshold, sensitivity to initial noise |
| `run_coupled_domains.py` | Topologically constrained graph | Bottleneck effects, asymmetric spatial absorption |

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
python run_high_energy.py --steps 10000 --sites 100

# Spatially concentrated initial state
python run_spatial_concentration.py --sites 100 --steps 8000 --primordial-frac 0.25

# Save figures for the papers
python run_simulation.py --plot -o ../figures/diagnostics.png
python run_high_energy.py -o ../figures/high_energy_cascade.png
python run_spatial_concentration.py -o ../figures/spatial_concentration
```

## Parameters

Defaults in `CascadeModel` match the paper conventions. Override via keyword arguments. The key dimensionless ratios that control qualitative behaviour are documented in Part I, Appendix B (Parameter Sensitivity).

## Relation to the papers

The theorems in Part I and Part II are proven independently of this code. The simulations serve a different purpose: they make the system's behaviour space *visible*. The formal guarantees tell you what must happen; the simulations show you *how* it happens across different initial conditions and graph topologies.

## Coupled cosmological DSCD system

The corrected dark-energy implementation is separate from both
`cascade_model.py` and the historical `CAS_LOG`/`CAS_POLY`/constant-\(w\)
scripts. Standard flat FLRW/GR supplies the background gravity; the expanding
stochastic DSCD state generates its density, pressure, `H(z)`, and BAO
distances.

Run in strict order:

```powershell
& ".venv-desi\Scripts\python.exe" "model_simulation\validate_dscd_cosmology_small.py" --force
& ".venv-desi\Scripts\python.exe" "model_simulation\run_dscd_cosmology_pilot.py" --force
& ".venv-desi\Scripts\python.exe" "model_simulation\run_dscd_identifiability.py" --force
& ".venv-desi\Scripts\python.exe" "model_simulation\run_dscd_retrospective.py" --force
& ".venv-desi\Scripts\python.exe" "model_simulation\audit_dscd_cosmology.py" --force
& ".venv-desi\Scripts\python.exe" "model_simulation\write_dscd_forecast_disposition.py" --force
```

The V1--V5 artifact records no use of real DESI values. Later artifacts form a
hash-linked chain. Audited v1 status: **`NO_FORECAST`** for the single frozen
configuration; that verdict is preserved as history and superseded by the v2
forecasting layer below.

Primary files:

- `DSCD_COSMOLOGY_SYSTEM_SPEC.md`
- `dscd_cosmology_config.py`
- `dscd_cosmology_state.py`
- `dscd_cosmology_dynamics.py`
- `dscd_cosmology_ensemble.py`
- `dscd_cosmology_observables.py`
- `dscd_cosmology_inference.py`
- `test_dscd_cosmology.py`
- `dscd_cosmology_results/audit.md`

## v2 forecasting layer (history-compatible ensemble)

Version 2 keeps the engine above unchanged and replaces the inference layer
with a state-space forecast: the present DSCD state and dynamics are latent,
sampled from a declared Sobol prior (depletion scale spanning four orders of
magnitude), weighted by the joint DR1+DR2 marginal likelihood with the common
BAO scale marginalized in closed form, and propagated forward to the DESI DR3
tracer layout. Eligibility gates test predictive convergence and end-to-end
synthetic coverage calibration, not parameter identifiability.

Run in strict order:

```powershell
& ".venv-desi\Scripts\python.exe" "model_simulation\validate_dscd_cosmology_small.py" --stage v6a --force
& ".venv-desi\Scripts\python.exe" "model_simulation\run_dscd_v2_forecast.py" --mode production --force
& ".venv-desi\Scripts\python.exe" "model_simulation\validate_dscd_forecast_calibration.py" --force
& ".venv-desi\Scripts\python.exe" "model_simulation\audit_dscd_v2_forecast.py" --force
& ".venv-desi\Scripts\python.exe" "model_simulation\write_dscd_v2_seal.py" --force   # only if audit passes
& ".venv-desi\Scripts\python.exe" -m unittest test_dscd_v2_forecast
```

Outputs live in `model_simulation/dscd_v2_results/`. The seal (if eligible) is
`18_DSCD_V2_DR3_FORECAST_RECORD.md` at the repository root.

Audited v2 status (10 July 2026): **`FORECAST_ELIGIBLE`**, 15/15 gates passed.
Joint ESS 205 of 2048 samples; interval medians stable under sample halving
(max shift 0.026 of the 68% width), disjoint seed banks (0.027), and DR2-only
conditioning (0.064); synthetic coverage 94.9% at the 95% level and 73.7% at
the 68% level. The sealed forecast medians lie within 0.06 DR2 sigma of
best-fit LambdaCDM and every interval excludes the CPL phantom excursion.

v2 files:

- `dscd_forecast_prior.py` (declared prior, sampler, weighted quantiles)
- `dscd_history_ensemble.py` (history scoring, scale marginalization, weights)
- `run_dscd_v2_forecast.py` (forward DR3 intervals, bands, gate inputs)
- `validate_dscd_forecast_calibration.py` (V6 synthetic coverage gate)
- `audit_dscd_v2_forecast.py` (G1-G4 gates and verdict)
- `write_dscd_v2_seal.py` (versioned seal writer)
- `test_dscd_v2_forecast.py` (unit tests)

## Clean DESI BAO analysis

The clean analysis is separate from the historical cosmology scripts. It uses the
official DESI DR1 and DR2 Gaussian BAO vectors and full covariance matrices from
`CobayaSampler/bao_data`, pinned to commit
`bb0c1c9009dc76d1391300e169e8df38fd1096db`.

Important scientific scope:

- Every DR1/DR2 calculation is retrospective because both releases are public.
- DR1 and DR2 overlap. The bidirectional calculation is therefore a release
  consistency diagnostic, not an independent blind prediction.
- The within-DR1 low-redshift/high-redshift split is a retrospective holdout.
- BAO constrains `theta = H0*r_d/c`; the code does not claim separate BAO-only
  measurements of `H0` or `r_d`.
- The old `13.23` QSO `D_H/r_d` value in historical scripts is not an official
  DESI DR1 datum. The official DR1 QSO measurement at `z=1.491` is
  `D_V/r_d = 26.07217182`.
- Historical scripts and their hardcoded chi-squared values are preserved for
  provenance but are not clean-pipeline results.

### Environment

On Windows PowerShell:

```powershell
python -m venv .venv-desi
& ".venv-desi\Scripts\python.exe" -m pip install -r "model_simulation\requirements_desi.txt"
```

The exact tested environment is recorded in `requirements_desi.txt`, and every
result JSON also records the resolved Python, NumPy, SciPy, and Matplotlib
versions.

### Commands

```powershell
# Fast audit run
& ".venv-desi\Scripts\python.exe" "model_simulation\run_desi_full_fit.py" --mode quick
& ".venv-desi\Scripts\python.exe" "model_simulation\run_desi_release_consistency.py" --mode quick
& ".venv-desi\Scripts\python.exe" "model_simulation\run_desi_redshift_holdout.py" --mode quick

# Canonical high-precision run
& ".venv-desi\Scripts\python.exe" "model_simulation\run_desi_full_fit.py" --mode publication
& ".venv-desi\Scripts\python.exe" "model_simulation\run_desi_release_consistency.py" --mode publication
& ".venv-desi\Scripts\python.exe" "model_simulation\run_desi_redshift_holdout.py" --mode publication

# Plot only from validated JSON
& ".venv-desi\Scripts\python.exe" "model_simulation\plot_clean_desi_results.py" --mode publication

# Apply numerical/stability gates and write the audit report
& ".venv-desi\Scripts\python.exe" "model_simulation\audit_clean_desi_results.py" --mode publication

# Unit tests
& ".venv-desi\Scripts\python.exe" -m unittest discover -s "model_simulation" -p "test_desi_bao_likelihood.py" -v
```

Outputs are written atomically to `model_simulation/clean_desi_results/`. Existing
JSON files are not replaced unless `--force` is supplied. Result files contain
source and configuration hashes, parameter bounds, optimizer diagnostics,
covariance-aware residuals, profiles, sensitivity checks, bootstrap summaries,
and explicit scientific warnings. Standard AIC/AICc is suppressed for boundary
or singular fits where its regularity assumptions do not hold.
