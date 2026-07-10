# DSCD v2: Methodology, Process Record, and Outlook

Date: 10 July 2026
Scope: complete record of how the sealed DESI DR3 forecast in
`18_DSCD_V2_DR3_FORECAST_RECORD.md` was produced, why the method has the shape
it has, and an honest assessment of the possible outcomes.

This document is descriptive. It changes nothing in the sealed record and is
not part of any hash chain. The authoritative artifacts live in
`model_simulation/dscd_v2_results/`.

---

## 1. The question being answered

Version 1 of the cosmological DSCD system asked an identifiability question:
*can compressed BAO measure the internal parameters of one hand-picked DSCD
configuration?* The answer was no (whitened-Jacobian condition number
1.06e7 for the depletion scale; all structural ablations far below
observational materiality), and the v1 audit correctly returned `NO_FORECAST`
for that frozen configuration. That verdict is preserved unchanged in
`17_CORRECTED_DSCD_DR3_DISPOSITION.md`.

Version 2 asks the forecasting question the v1 audit never posed:

> **Given the DESI DR1+DR2 expansion history, do the DSCD+GR realizations
> compatible with that history agree on what DR3 must measure?**

A dynamical system is predictive when its history-conditioned state ensemble
funnels into a narrow set of futures, even if no individual internal parameter
is separately measurable. Identifiability was the wrong requirement;
predictive convergence is the right one. This is a state-space forecast, not a
parameter fit.

## 2. The simulation engine (unchanged from v1)

One coupled stochastic dynamical system, `dscd-cosmology-v1`:

- **DSCD sector.** Cells on a finite weighted comoving ring carry activity
  species X and Y, an intensive free-capacity density rho_C, structural
  memory S (two histories), beat phase phi, and a local regime r. Interaction
  and leakage events are drawn from proper-time Poisson hazards; every event
  is *funded* from the remaining local density before it is committed, so
  rho_C decreases pathwise with no clipping. The ripple (discrete curvature of
  structural memory) selects quiescent / leakage / explosive regimes.
  Transport is a conservative graph Laplacian.
- **GR closure.** The graph-weighted mean density enters the flat Friedmann
  constraint E^2(N) = Omega_m e^(-3N) + Omega_r e^(-4N) + c_C rho_bar_C(N).
  Separate covariant conservation converts the generated depletion rate into a
  background pressure: w_C = -1 + D_C / (3 E rho_bar_C). Funded depletion
  means D_C >= 0, so w_C >= -1 wherever rho_bar_C > 0: the non-phantom bound
  is structural, not a prior on w.
- **Observables.** The trajectory yields H(z) directly; BAO distances
  D_H/r_d, D_M/r_d, D_V/r_d follow by quadrature with a single scale
  parameter theta = H_0 r_d / c.

Before any data contact, the engine passed the V1-V5 ladder (hand-checked
funded accounting, hazard distributions, conservative transport, exact
LambdaCDM recovery at zero depletion, synthetic distance recovery) plus
positivity, continuity, Friedmann, early-density, and seed-reproducibility
checks. For v2, gate **V6a** additionally stress-tested the engine at the
prior corners (depletion scale up to 10^3), verifying funded accounting,
positivity, closure convergence, and the non-phantom diagnostic there.

## 3. The v2 method, step by step

### 3.1 Declared prior over latent realizations

The present DSCD state is treated as latent. Eight dimensions, declared once
in `dscd_forecast_prior.py` and never tuned against data:

| Dimension | Support | Role |
|---|---|---|
| log10 depletion scale | [-1, 3] | common multiplier on funded costs and leakage |
| alpha_XY | [0.05, 0.60] | cross-species interaction strength |
| alpha_XX | [0.00, 0.20] | same-species depletion strength |
| omega_Y - omega_X | [0.00, 0.80] | beat asymmetry |
| X_0 | [2, 10] | initial activity per cell |
| Y_0/X_0 | [1, 2] | initial species asymmetry |
| explosion rate | [0.00, 0.20] | explosive-regime event rate |
| Omega_m | [0.20, 0.45] | present matter fraction |

Sampling: scrambled Sobol sequence (low-discrepancy, so the first half of the
draw is itself a valid design — used later by gate G1). Production run:
**2048 samples x 4 stochastic replicates**, deterministic collision-free seed
blocks per sample.

### 3.2 History-compatibility weighting

Each realization is evolved forward and scored against the official DESI DR1
(12 components) and DR2 (13 components) compressed BAO vectors with full
covariance matrices (pinned CobayaSampler/bao_data commit).

- **Seed scatter.** Each release covariance is inflated by the seed-ensemble
  covariance of the predicted vector, so stochastic realization scatter is
  part of the likelihood, not ignored.
- **Scale marginalization (the decisive correction).** For fixed dynamics,
  every BAO distance scales exactly as s = 1/theta. The posterior of s under
  a Gaussian likelihood is therefore Gaussian in closed form. Version 2
  *marginalizes* s analytically (flat prior) instead of profiling it, and
  propagates the residual scale uncertainty into every forward prediction
  through 8 Gauss-Hermite quadrature nodes. Profiling had collapsed this
  common-mode uncertainty and produced overconfident intervals — caught by
  the coverage calibration gate (Section 3.4), which is exactly what that
  gate exists for.
- **Importance weights.** Each sample receives weight proportional to its
  joint DR1+DR2 marginal likelihood. Convergence of the weighted ensemble is
  measured by the effective sample size ESS = (sum w)^2 / sum w^2.

### 3.3 Forward pooling to DR3

For every history-weighted realization, the forward DR3-layout prediction pool
keeps three axes: trajectory seed x scale node x observable. Weighted
quantiles over that pool give the credible intervals. The DR3 layout is
declared as the DR2 tracer set remeasured; if DR3 publishes different
redshifts, the recorded trajectories re-evaluate on the published layout
without refitting.

### 3.4 End-to-end coverage calibration (gate V6/G4)

The pipeline must be shown to produce honest intervals before its real
forecast means anything. Twelve synthetic truths are drawn from the prior,
each converted into synthetic DR1/DR2/DR3 data with the official covariances,
and each processed by the *identical* pipeline (512 samples per truth). The
fraction of withheld synthetic DR3 components covered by the predicted
intervals must match the nominal levels. Result: **94.9% at the 95% level,
73.7% at the 68% level** — PASS. (The first calibration run, before scale
marginalization, failed with 78%/45%; the method was corrected and the full
chain rerun.)

### 3.5 Convergence gates and audit

`audit_dscd_v2_forecast.py` checks a hash-chained gate set; all artifacts
record the SHA-256 of every source file and upstream artifact they consumed.

Technical gates: source hash currency, artifact chain links, official DESI
arrays byte-equal to the pinned likelihood, sample accounting, finite ordered
quantiles. Scientific gates and observed values:

| Gate | Requirement | Observed |
|---|---|---|
| G1 sampling convergence | median shift < 0.10 of 68% width under Sobol halving | 0.026 |
| G2 seed robustness | median shift < 0.10 under disjoint seed banks | 0.027 |
| G3 effective sample size | joint ESS >= 25 | 205.3 |
| G3 DR2-only conditioning | median shift < 0.25 | 0.064 |
| G4 coverage calibration | 95% level >= 0.85; 68% level in [0.50, 0.86] | 0.949 / 0.737 |

Verdict: **FORECAST_ELIGIBLE, 15/15 gates passed.** Only then was the seal
written. During the process the audit once returned `TECHNICAL_FAILURE`
because a documentation file pinned by the engine-stress artifact had been
edited after hashing; the entire chain (V6a, production forecast,
calibration, audit) was rerun rather than waived, and the rerun reproduced
the forecast numbers to within 0.0004.

### 3.6 The sealed result

Thirteen credible intervals on the DR3 layout, sealed in
`18_DSCD_V2_DR3_FORECAST_RECORD.md` with artifact hashes:

- 68% widths between **0.16 and 1.10 DR2 sigma** — the history genuinely
  constrains the future; the prior alone would be far wider.
- Forecast medians within **0.058 DR2 sigma of best-fit LambdaCDM**.
- Every interval excludes the phantom excursion preferred by unconstrained
  CPL (example: CPL predicts LRG1 D_H/r_d = 21.88; sealed 95% interval
  [22.28, 22.58]).
- Weighted parameter posterior: theta = 0.03387 +0.00020/-0.00023,
  Omega_m = 0.296 +/- 0.008. The depletion fraction spans 2.7e-4 to 0.125
  (68%): many internally different DSCD states remain history-compatible,
  exactly as v1 unidentifiability implied — and they nevertheless agree on
  DR3. That agreement *is* the result.

Seal integrity: any post-DR3 modification of prior bounds, sampler seed, or
gate thresholds voids the seal; a corrected forecast must be re-sealed under
a new version.

## 4. Reproduction

From the repository root (Windows PowerShell, `.venv-desi`):

```powershell
& ".venv-desi\Scripts\python.exe" "model_simulation\validate_dscd_cosmology_small.py" --stage v6a --force
& ".venv-desi\Scripts\python.exe" "model_simulation\run_dscd_v2_forecast.py" --mode production --force
& ".venv-desi\Scripts\python.exe" "model_simulation\validate_dscd_forecast_calibration.py" --force
& ".venv-desi\Scripts\python.exe" "model_simulation\audit_dscd_v2_forecast.py" --force
& ".venv-desi\Scripts\python.exe" "model_simulation\write_dscd_v2_seal.py" --force   # only if audit passes
& ".venv-desi\Scripts\python.exe" -m unittest test_dscd_v2_forecast
```

Runtime note: the workload is CPU-parallel across independent realizations
(ProcessPoolExecutor, cores-2 workers). It is not GPU-suitable: each
trajectory is a small-state, branch-heavy, strictly sequential computation
(funded integer event counts, regime switching, 65 dependent steps), which is
the opposite of what GPUs accelerate. Production forecast ~6 minutes and
calibration ~30 minutes on a 14-worker desktop.

## 5. Outlook: what "winning" means and the honest odds

Subjective assessments as of 10 July 2026, not pipeline outputs.

**Level 1 — the sealed intervals survive DR3 (~80-90%).** The forecast is
LambdaCDM-like and the falsification bar is the 99% interval widened by one
DR3 sigma. DR2 moved only modestly from DR1, and the coverage gate shows the
intervals are honest. The residual risk is a genuine surprise in one tracer
(DR3 roughly doubles survey volume; Lyman-alpha has surprised before).

**Level 2 — the structural non-phantom bet pays off (~50-70%).** DSCD forbids
w < -1 everywhere and always. DESI's DR1/DR2 *combined* analyses (BAO + CMB +
supernovae) currently prefer evolving dark energy whose reconstructed w(z)
dips phantom at 2-4 sigma; BAO alone is LambdaCDM-consistent. If that
combined hint hardens at DR3 precision, the DSCD mechanism is falsified
regardless of the intervals. If it fades as errors shrink — the historical
fate of most 2-3 sigma cosmology hints — the non-phantom commitment wins and
CPL's phantom crossing is exposed as a flexible parameterization absorbing
noise.

**Level 3 — DR3 proves dark energy is DSCD (~0%).** No background distance
measurement can do this. Surviving DR3 leaves DSCD viable and distinguished
from CPL, not confirmed against LambdaCDM: the posterior depletion fraction
already spans nearly-zero to 0.125. Separating DSCD from a cosmological
constant requires the perturbation layer (growth rate, clustering of the
dark sector) — a future system version with its own state variables,
equations, and validation ladder.

Both branches are clean. A verified pre-registered forecast plus a surviving
structural claim is a strong position for a new framework; a hardened phantom
signal kills the model honestly rather than leaving it unfalsifiable.

## 6. Document map

| File | Role |
|---|---|
| `16_CASCADE_ORIGIN_AND_DR3_PREDICTION_RECORD.md` | historical June record (surrogate analysis), preserved byte-for-byte |
| `17_CORRECTED_DSCD_DR3_DISPOSITION.md` | v1 NO_FORECAST disposition, preserved (supersession note only) |
| `18_DSCD_V2_DR3_FORECAST_RECORD.md` | the sealed v2 DR3 forecast |
| `19_DSCD_V2_METHODOLOGY_AND_OUTLOOK.md` | this document |
| `model_simulation/DSCD_COSMOLOGY_SYSTEM_SPEC.md` | normative system specification (v1 engine + v2 layer) |
| `model_simulation/dscd_v2_results/` | audit, calibration, forecast, and seal artifacts with hashes |
| `dark_energy_cascade_preprint.tex` / `.pdf` | the paper containing construction, audit, and sealed forecast |
