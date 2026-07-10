# Clean DESI numerical audit

Status: **COMPLETE_WITH_SCIENTIFIC_LIMITATIONS**

Mode: `quick`
Configuration: `00c95ed84445c1b7c1c0fd49d62bcd24e516b281c639948deae876e72cb2bfb5`

## Gate summary

- Passed: 50 / 64
- Technical failures: 0
- Scientific stability failures: 14

## Failed stability gates

- `DESI_DR1.CAS_POLY.bound_stability`: |delta chi2|=0.0539595; active=['gamma']
- `DESI_DR1.CPL.optimizer_agreement`: global seed chi2 spread=0.00121962
- `DESI_DR1.CPL_NONPHANTOM.optimizer_agreement`: global seed chi2 spread=0.00103804
- `DESI_DR1.CPL_NONPHANTOM.quadrature_stability`: |delta chi2|=0.00869499
- `DESI_DR1.CPL_NONPHANTOM.bound_stability`: |delta chi2|=0.278558; active=['omega_m']
- `DESI_DR2.CAS_POLY.bound_stability`: |delta chi2|=0.0648261; active=['gamma']
- `DESI_DR2.CPL.optimizer_agreement`: global seed chi2 spread=0.0146821
- `DESI_DR2.CPL_NONPHANTOM.optimizer_agreement`: global seed chi2 spread=0.0124092
- `DESI_DR2.CPL_NONPHANTOM.bound_stability`: |delta chi2|=0.105156; active=[]
- `holdout.CAS_LOG.bound_stability`: delta target chi2=273.407; active=['omega_m']
- `holdout.CAS_POLY.bound_stability`: delta target chi2=-25.6443; active=['w0']
- `holdout.CPL_NONPHANTOM.bound_stability`: delta target chi2=336.605; active=['omega_m']
- `holdout.WCDM.bound_stability`: delta target chi2=83.7632; active=['omega_m']
- `holdout.WCDM_NONPHANTOM.bound_stability`: delta target chi2=88.1762; active=['omega_m']

## Interpretation constraints

- In-sample chi2 values are descriptive and retrospective.
- DR1-to-DR2 and DR2-to-DR1 scores are not independent because releases overlap.
- Holdout bootstrap tails are finite-simulation diagnostics, not preregistered p-values.
- Any model failing optimizer, bound, or quadrature stability must not be ranked.
- No manuscript, website, historical seal, or Git history was modified by this analysis.

The complete numerical values, residuals, optimizer traces, profiles, and bootstrap samples remain in the accompanying strict JSON files.
