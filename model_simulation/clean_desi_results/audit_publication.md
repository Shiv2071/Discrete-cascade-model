# Clean DESI numerical audit

Status: **COMPLETE_WITH_SCIENTIFIC_LIMITATIONS**

Mode: `publication`
Configuration: `3b2876c63b7f93a2d7ad9d8405eff2242ba45335061036db6684f560b6db4df0`

## Gate summary

- Passed: 57 / 64
- Technical failures: 0
- Scientific stability failures: 7

## Failed stability gates

- `DESI_DR1.CAS_POLY.bound_stability`: |delta chi2|=0.054107; active=['gamma']
- `DESI_DR2.CAS_POLY.bound_stability`: |delta chi2|=0.0648663; active=['gamma']
- `holdout.CAS_LOG.bound_stability`: delta target chi2=272.377; active=['omega_m']
- `holdout.CAS_POLY.bound_stability`: delta target chi2=-37.6349; active=['w0']
- `holdout.CPL_NONPHANTOM.bound_stability`: delta target chi2=348.418; active=['omega_m']
- `holdout.WCDM.bound_stability`: delta target chi2=87.2648; active=['omega_m']
- `holdout.WCDM_NONPHANTOM.bound_stability`: delta target chi2=87.2745; active=['omega_m']

## Interpretation constraints

- In-sample chi2 values are descriptive and retrospective.
- DR1-to-DR2 and DR2-to-DR1 scores are not independent because releases overlap.
- Holdout bootstrap tails are finite-simulation diagnostics, not preregistered p-values.
- Any model failing optimizer, bound, or quadrature stability must not be ranked.
- No manuscript, website, historical seal, or Git history was modified by this analysis.

The complete numerical values, residuals, optimizer traces, profiles, and bootstrap samples remain in the accompanying strict JSON files.
