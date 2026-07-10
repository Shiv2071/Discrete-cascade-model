# Cosmological DSCD scientific audit

Status: **NO_FORECAST**

## Gate summary

- Passed: 25 / 33
- Technical failures: 0
- Scientific failures: 8

## Result summary

- DR1 full-fit DSCD chi-squared: 12.736197
- DR2 full-fit DSCD chi-squared: 10.279706
- Retrospective DR1 high-redshift conditional chi-squared: 3.465984
- DR2 DSCD-LambdaCDM squared separation: 0.000101425

## Failed gates

- `identifiability.DSCD_depletion_scale`: condition=1.06133e+07; rank=3
- `DSCD_vs_LCDM_discriminability`: joint squared separation d2=0.000101425; required >=4
- `ablation.no_memory.material`: squared observational separation d2=2.14664e-06; required >=1
- `ablation.no_regime_feedback.material`: squared observational separation d2=2.82987e-06; required >=1
- `ablation.no_transport.material`: squared observational separation d2=5.00247e-06; required >=1
- `ablation.symmetric_depletion.material`: squared observational separation d2=1.9927e-06; required >=1
- `ablation.zero_beat.material`: squared observational separation d2=6.39204e-06; required >=1
- `DESI_DR1.optimizer_prediction_agreement`: raw prediction L2 difference=0.107753

## Interpretation

- The coupled dynamical system and all small numerical validations run successfully.
- One DSCD depletion combination is not identifiable with the compressed BAO layout, so it is frozen.
- The fixed DSCD background is close to LambdaCDM and its structural ablations are not observationally distinguishable.
- No corrected DR3 forecast is scientifically eligible under the preregistered gates.
