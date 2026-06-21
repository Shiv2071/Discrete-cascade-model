# Empirical Validation: DESI BAO Blind Retrodiction Test

**Date:** June 21, 2026  
**Author:** Shiv Goswami  
**Priority date:** 21 June 2026. Preprint in preparation.

---

## What was tested

The cascade model's dark energy equation of state was compared against real observational data for the first time.

**Two datasets used across two separate DESI data releases.**  
The specific observable tested is the Hubble distance ratio D\_H/r\_d at six redshift values spanning z = 0.51 to z = 2.33.

---

## Protocol (blind retrodiction)

The test was structured so the model could not see the answer before predicting it, and the calibration and blind-check data come from different observational releases.

**Step 1: Calibration (DESI Year 1 BAO, April 2024)**  
Three low-redshift data points from DESI Year 1 were used to fit each model:

| Redshift | Observable |
|----------|-----------|
| z = 0.510 | D\_H/r\_d from DESI BGS/LRG1 |
| z = 0.706 | D\_H/r\_d from DESI LRG2 |
| z = 0.930 | D\_H/r\_d from DESI LRG3 |

**Step 2: Blind prediction (DESI DR2 BAO)**  
Each model then predicted three high-redshift points from DESI DR2 that it had not seen during calibration and that came from a different data release:

| Redshift | Observable | Source |
|----------|-----------|--------|
| z = 1.317 | D\_H/r\_d | DESI DR2 ELG |
| z = 1.491 | D\_H/r\_d | DESI DR2 QSO |
| z = 2.330 | D\_H/r\_d | DESI DR2 Lya QSO |

The z = 2.330 point corresponds to light emitted 11 billion years ago and is the deepest check.

**Step 3: Compare**  
Predictions were compared against actual DESI measurements after they were made.

---

## Models compared

Three models were calibrated and tested under the same protocol:

- **LCDM** (cosmological constant, w = -1 always): zero free dark energy parameters
- **Cascade** (this model, one free parameter w0): constrained by the no-phantom theorem so w(z) > -1 always
- **CPL** (standard parameterization w0 + wa, two free parameters): no theoretical constraint on w

---

## Results on unseen high-redshift data

Chi-squared on the three blind check points (lower is better):

| Model | Chi-squared (blind) | Free parameters |
|-------|-------------------|-----------------|
| Cascade | **4.44** | 1 |
| LCDM | 5.54 | 0 |
| CPL | 22.81 | 2 |

CPL has twice as many free parameters as cascade and scored five times worse on data it had not seen.

### Deepest check: z = 2.330 (Lya QSO)

| Model | Deviation from observed |
|-------|------------------------|
| Cascade | +1.83 sigma |
| LCDM | +2.07 sigma |
| CPL | +4.05 sigma |

---

## Why cascade is more accurate

The no-phantom theorem (proved from cascade axioms in Part I of the paper) states that the effective equation of state satisfies w(z) > -1 at every active step. This is not a fitted assumption; it follows in one line from the energy monotonicity supermartingale established as Theorem 3.3 in Part I.

This theorem prevented the cascade model from overfitting the DESI Year 1 calibration data into phantom dark energy territory (w < -1). CPL has no equivalent constraint. During calibration on Year 1 data, CPL's extra freedom allowed it to fit more tightly by dipping into the phantom regime. When it then predicted the unseen DESI DR2 data, this phantom overfitting compounded into large errors. The cascade model's theoretical constraint produced better blind prediction across both datasets.

In short: a proved theorem about the model's internal dynamics turned out to be the mechanism that made it more accurate on real data.

---

## Scripts

The three scripts that implement this test are in `model_simulation/`:

| Script | Purpose |
|--------|---------|
| `cascade_retrodiction.py` | Full blind retrodiction protocol. Calibrates all three models, predicts unseen data, reports chi-squared and sigma residuals. |
| `reverse_cascade.py` | Inverts DESI H(z) data to recover the cascade's implied beta history from Big Bang to today. |
| `run_bigbang_analogue.py` | Runs the paper's Big Bang analogue with prescribed parameters (E0 = 500, X0 = 8, Y0 = 8) and records the emergent three-regime arc. |

### Run the retrodiction test

```bash
cd model_simulation
python cascade_retrodiction.py   # blind retrodiction test
python reverse_cascade.py        # DESI beta history inversion
python run_bigbang_analogue.py   # Big Bang arc with paper parameters
```

Output: chi-squared table, sigma residuals per data point, and a figure saved to `../figures/cascade_retrodiction.png`.

---

## Figures

`figures/cascade_retrodiction.png` shows:
- D\_H/r\_d curves for all three models vs DESI data
- Calibration points (used to fit) vs blind check points (not seen during fit)
- Bar chart of blind prediction residuals per model
- w(z) evolution of each model

`figures/reverse_cascade.png` shows the cascade's inferred beta history from recombination to today, derived by inverting the DESI measurements.

`figures/bigbang_paper_arc.png` shows the three-regime arc (Explosive, Leakage, Quiescent, Absorbing) from the Big Bang analogue simulation.

---

## Note on scope

The cascade model was not built for cosmology. It is a general mathematical framework for any system with asymmetric interacting classes, forbidden self-interaction, frequency mismatch, finite non-renewable resources, and irreversible structural accumulation. Dark energy is the first domain where its predictions have been tested against real data. The same structural logic applies to thermodynamic, biological, and other complex systems.
