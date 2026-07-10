# Corrected cosmological DSCD forecast disposition

> **Superseded (10 July 2026).** This disposition applies to the version-1
> inference layer (one frozen configuration). Consistent with the amendment
> rule below, no numbers were added here: the version-2 forecasting layer
> (`dscd-forecast-v2`) issued its own versioned record with new source and
> artifact hashes after a complete pre-release audit. See
> `18_DSCD_V2_DR3_FORECAST_RECORD.md`. Everything below this note is
> unchanged.

Status: **NO_FORECAST**

Created (UTC): `2026-07-09T22:52:58.893428+00:00`
Scientific audit SHA-256: `81aa182332114cda9b5a2e535e523b572537c2ddc5d5371dfb7d7f77a162dc8a`
Historical record SHA-256: `6538695d3391ec93289bd3bf044b893874f8b5e1eff19875976ad48a27cdd77e`

## Decision

The corrected coupled DSCD+GR system is not eligible to issue a new DR3 forecast. No corrected prediction numbers are sealed.

The historical file `16_CASCADE_ORIGIN_AND_DR3_PREDICTION_RECORD.md` was preserved byte-for-byte. Its constant-w surrogate is historical provenance and is not a prediction from the corrected coupled DSCD+GR system.

## Blocking scientific gates

- `identifiability.DSCD_depletion_scale`: condition=1.06133e+07; rank=3
- `DSCD_vs_LCDM_discriminability`: joint squared separation d2=0.000101425; required >=4
- `ablation.no_memory.material`: squared observational separation d2=2.14664e-06; required >=1
- `ablation.no_regime_feedback.material`: squared observational separation d2=2.82987e-06; required >=1
- `ablation.no_transport.material`: squared observational separation d2=5.00247e-06; required >=1
- `ablation.symmetric_depletion.material`: squared observational separation d2=1.9927e-06; required >=1
- `ablation.zero_beat.material`: squared observational separation d2=6.39204e-06; required >=1
- `DESI_DR1.optimizer_prediction_agreement`: raw prediction L2 difference=0.107753

## Amendment rule

No corrected DR3 numerical forecast may be added to this disposition. A materially revised dynamical system requires a new versioned record, new source/configuration hashes, and a complete pre-release audit.
