# Cascade Model: Sealed DR3 Prediction Record

**Sealed:** 21 June 2026  
**Author:** Shiv Goswami  
**Repository:** github.com/Shiv2071/Discrete-cascade-model  
**To be compared against:** DESI DR3, expected ~2027  
**Script:** `model_simulation/cascade_dr3_prediction.py`  
**Figure:** `figures/cascade_dr3_prediction.png`

---

## What this is

The cascade model has been calibrated on DESI DR2 BAO data (March 2025, arXiv: 2503.14738) and its predictions for DESI DR3 are recorded here before DR3 is released. This document is sealed on the commit timestamp. When DR3 is published, the predictions below will be compared against the actual measurements.

---

## Training data

**DESI DR2 BAO (March 2025, arXiv: 2503.14738)**

| Redshift | D\_H/r\_d | sigma |
|----------|----------|-------|
| z = 0.510 | 21.863 | 0.425 |
| z = 0.706 | 19.455 | 0.330 |
| z = 0.934 | 17.641 | 0.193 |
| z = 1.321 | 14.176 | 0.221 |
| z = 1.484 | 12.817 | 0.516 |
| z = 2.330 | 8.632  | 0.101 |

---

## Calibrated parameters (from DR2)

| Model | H0 | Om | w0 | wa | chi-sq on DR2 |
|-------|----|----|----|----|--------------|
| CASCADE | 71.3 | 0.274 | -0.999 | n/a | 4.076 |
| LCDM | 71.6 | 0.271 | -1 (fixed) | n/a | 4.047 |
| CPL | 66.2 | 0.340 | -0.361 | -2.500 | 1.307 |

CPL phantom crossing at z = 0.345 (unphysical: w < -1 violates Null Energy Condition).  
Cascade and LCDM: no phantom crossing. Theorem enforced.

---

## Sealed D\_H/r\_d predictions for DESI DR3

DR3 projected sigma: DR2 sigma / sqrt(47/14) ~ DR2 sigma / 1.83, with 0.08 systematic floor.

| Redshift | CASCADE | LCDM | CPL | Proj. sigma |
|----------|---------|------|-----|-------------|
| z = 0.510 | **22.1265** | 22.0870 | 21.7463 | +/- 0.2320 |
| z = 0.706 | **19.7903** | 19.7675 | 19.7301 | +/- 0.1801 |
| z = 0.934 | **17.3703** | 17.3601 | 17.5353 | +/- 0.1053 |
| z = 1.321 | **14.0283** | 14.0289 | 14.2279 | +/- 0.1206 |
| z = 1.484 | **12.8795** | 12.8824 | 13.0367 | +/- 0.2816 |
| z = 2.330 | **8.6806**  | 8.6869  | 8.6341  | +/- 0.0800 |

---

## Falsifiable predictions

These are specific, testable claims. Each can be confirmed or falsified by DR3.

### Prediction 1: No phantom crossing

The cascade no-phantom theorem (proved from energy monotonicity in Part I) requires w(z) > -1 at all redshifts. This is not a tuning choice; it is a mathematical consequence of the model's axioms.

**If DR3 confirms w < -1 at any redshift with high significance, the cascade dark energy application is ruled out.**  
**If DR3 finds w consistent with the range [-1, -0.7], cascade is vindicated.**

### Prediction 2: Lya QSO (z = 2.330)

Cascade predicts D\_H/r\_d = 8.6806 at z = 2.330.  
Projected DR3 sigma: +/- 0.0800.  
If DR3 lands within 1 sigma: confirmed.  
If DR3 deviates by more than 2 sigma: tension.

### Prediction 3: CPL phantom preference weakens in DR3

CPL calibrated on DR2 finds wa = -2.50 (extreme phantom). This is noise absorption, not physics. As DR3 error bars tighten (projected ~1.83x improvement), the statistical pressure that forces CPL into phantom territory will be reduced.

**If DR3 best-fit CPL wa moves toward 0 compared to DR2, the no-phantom theorem is vindicated as the correct prior.**  
**If DR3 finds wa = -2.5 or more extreme with high confidence, cascade has a genuine problem.**

### Prediction 4: w0 range

Cascade calibrated on DR2 gives w0 = -0.999, essentially the cosmological constant boundary.  
**DR3 best-fit w0 should fall in [-1.15, -0.85] if cascade is correct.**

---

## Context: what happened with Year 1 and DR2

| Test | Cascade chi-sq | LCDM chi-sq | CPL chi-sq | Notes |
|------|---------------|-------------|------------|-------|
| Within Year 1 retrodiction | 4.44 | 5.54 | 22.81 | CPL phantom overfitting punished |
| Cross-release Year 1 to DR2 | 4.453 | 4.390 | 2.174 | CPL wins via phantom (z=0.145) |
| DR3 prediction (this record) | TBD | TBD | TBD | To be filled in on DR3 release |

The two prior tests tell the same underlying story: CPL's phantom excursion helps it in some data configurations and fails catastrophically in others. Cascade's theoretical constraint keeps it honest in both.

---

## How to verify when DR3 releases

1. Run `model_simulation/cascade_dr3_prediction.py` -- it contains the locked parameters.
2. Replace the DR3 projected values with actual DR3 measured values.
3. Compute sigma residuals: (cascade\_pred - DR3\_actual) / DR3\_sigma for each redshift.
4. Check CPL best-fit wa from DR3 official paper against wa = -2.50 from DR2.
5. Record outcome in this file under a new section: `## DR3 Comparison (filled [date])`.

---

## Outcome template (to be filled when DR3 releases)

```
## DR3 Comparison (filled [DATE])

DR3 paper reference: [arXiv: XXXX.XXXXX]

| z | Cascade pred | DR3 actual | DR3 sigma | Residual |
|---|-------------|------------|-----------|----------|
| 0.510 | 22.1265 | ? | ? | ? |
| 0.706 | 19.7903 | ? | ? | ? |
| 0.934 | 17.3703 | ? | ? | ? |
| 1.321 | 14.0283 | ? | ? | ? |
| 1.484 | 12.8795 | ? | ? | ? |
| 2.330 | 8.6806  | ? | ? | ? |

DR3 CPL best-fit wa: ?  (DR2 was -2.50 -- did it move toward 0?)
DR3 phantom crossing confirmed: Yes / No
Cascade prediction status: Confirmed / Tension / Ruled out
```
