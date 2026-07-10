# Cascade Model vs. Standard Dark Energy Approaches

> **Superseded comparison (10 July 2026).** The theorem transfers and
> cosmological claims in the table below are not supported by the corrected
> coupled-system audit. DSCD now generates a late-time background trajectory
> inside standard flat FLRW/GR; it does not derive gravity, intrinsic pressure,
> cosmic heat death, or a unique no-phantom physical theory. Current compressed
> BAO cannot identify its depletion scale or distinguish its defining
> mechanisms from ablations. See
> `model_simulation/DSCD_COSMOLOGY_SYSTEM_SPEC.md` and
> `model_simulation/dscd_cosmology_results/audit.md`.

**Author: Shiv Goswami**
**Date: June 23, 2026**

---

## Summary

The table below compares the cascade model against the main competing approaches on the dimensions that matter most for dark energy research. Each entry is a factual statement, not a rhetorical claim.

| Dimension | ΛCDM | Quintessence | String landscape | **Cascade model** |
|---|---|---|---|---|
| Explains why w ≠ -1 | No (asserts w = -1) | Partially (scalar field rolling) | No (anthropic selection) | **Yes: D(n) > 0 is proved whenever active (Thm 3.1, Part I)** |
| No-phantom guarantee (w ≥ -1) | Agnostic (CPL extensions cross it) | Yes for most potentials, not proved generally | Conjectured (Swampland de Sitter conjecture, unproved, contested) | **Proved theorem: β is a non-negative supermartingale (Thm 3.1, Part I). Holds in every realization, with probability one.** |
| Three cosmic epochs from one mechanism | No (needs separate inflation + CDM + Λ) | No | No | **Yes: explosive / leakage / quiescent regimes emerge from one set of rules (Def 2.7, Part I)** |
| Heat death as a proved result | Assumed | Assumed | Assumed | **Proved: almost sure absorption in finite time (Thm 3.3, Part I)** |
| Connection to particle physics / GR | Yes | Yes (scalar field theory) | Yes (string vacua) | No, currently none |
| UV completion | No | No | Attempted | No |
| Fine-tuning problem addressed | No | No (potential still fine-tuned) | Anthropic argument | No (β₀ is a free initial condition) |
| Identification of dark energy | Λ = vacuum energy (fine-tuned) | Scalar field φ (choice of potential) | Vacuum energy from string landscape | β(p,n) = local capacity energy (single asserted identification) |
| Falsifiable prediction made | No | No | No | **Yes: sealed DR3 prediction (June 21, 2026)** |
| Blind retrodiction test | N/A | N/A | N/A | **Yes: DESI Year 1, chi-sq 4.44 vs CPL 22.81 (EMPIRICAL_VALIDATION.md)** |

---

## The strongest claim

The no-phantom constraint is the most precisely grounded structural prediction in the table. String theory's Swampland program conjectures the same result (w > -1, or equivalently no stable de Sitter vacua) but this remains an active controversy with no proof. The cascade model proves it from one equation:

$$\beta(p,n+1) - \beta(p,n) = -[k N_{XY} + k_{XX} N_{XX} + L + M + \kappa B] \leq 0 \quad \text{a.s.}$$

If β is dark energy density (the single identification), then dark energy cannot increase. Therefore w ≥ -1 at every step with probability one. This follows from Theorem 3.1 of Part I with no approximation and no free parameters.

---

## The weakest point

The identification β ↔ ρ_DE is asserted, not derived from first principles. Every result in the cascade dark energy application is conditional on this identification. A formal derivation connecting the cascade's capacity energy to dark energy density through a physical argument would change the model's standing significantly.

---

## What DESI DR3 will decide

The sealed predictions in `DR3_PREDICTION_RECORD.md` make the cascade model directly falsifiable by DESI DR3 (expected 2027). Specifically:

1. If DR3 confirms w(z) > -1 at all redshifts: the cascade no-phantom theorem is vindicated as a prior.
2. If DR3 best-fit CPL wa moves toward 0 compared to DR2's wa = -2.50: the cascade model's interpretation of phantom CPL as a parameterization artifact is confirmed.
3. If DR3 finds strong w < -1 evidence: the cascade dark energy application is constrained or ruled out.

No other model in the table above has made a sealed quantitative prediction for DR3.

---

*Preprint: [Zenodo 10.5281/zenodo.20787562](https://doi.org/10.5281/zenodo.20787562)*
*Papers: [Part I](https://doi.org/10.5281/zenodo.21210030) | [Part II](https://doi.org/10.5281/zenodo.21227989)*
*Repository: [github.com/Shiv2071/Discrete-cascade-model](https://github.com/Shiv2071/Discrete-cascade-model)*
