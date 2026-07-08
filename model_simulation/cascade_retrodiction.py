"""
cascade_retrodiction.py
=======================
Retrodiction test: calibrate cascade from low-z DESI data,
predict backward to high-z, check against actual DESI measurement.

Protocol:
  CALIBRATION  z = 0.510, 0.706, 0.930   (recent universe -- fit here)
  BLIND CHECK  z = 1.317, 1.491, 2.330   (early universe -- predict, then compare)

Three models compared:
  LCDM     : w(z) = -1         (cosmological constant)
  CASCADE  : f_DE(z) = (1+z)^{3(1+w0)},  w0 free, w(z) >= -1 always
  CPL      : f_DE(z) = (1+z)^{3(1+w0+wa)} exp(-3wa z/(1+z)), w0 and wa free

Question: which model, calibrated on low-z, best predicts the high-z data it
has NOT seen?

Author: Shiv Goswami
Date:   June 21, 2026
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Constants ──────────────────────────────────────────────────────────────
c_kms   = 299792.458   # km/s
rd_Mpc  = 147.09       # Mpc  (Planck 2018 sound horizon)
H0_fid  = 67.4         # km/s/Mpc (fiducial; we fit this)
Om_fid  = 0.315        # matter density (we fit this)

# ── DESI Year 1 BAO data (April 2024, arXiv: 2404.03002): z, DH/rd, sigma ─
desi = np.array([
    [0.510, 20.98, 0.61],   # LRG1
    [0.706, 20.08, 0.60],   # LRG2
    [0.930, 17.88, 0.35],   # LRG3+ELG1
    [1.317, 13.82, 0.42],   # ELG2
    [1.491, 13.23, 0.55],   # QSO
    [2.330,  8.52, 0.17],   # Lya QSO  <-- the deepest check point
])

z_all   = desi[:, 0]
DH_all  = desi[:, 1]
sDH_all = desi[:, 2]

# Split: calibrate on low-z, predict high-z
CALIB_IDX = [0, 1, 2]     # z = 0.51, 0.706, 0.93
CHECK_IDX  = [3, 4, 5]    # z = 1.317, 1.491, 2.33  (blind prediction)

z_calib   = z_all[CALIB_IDX]
DH_calib  = DH_all[CALIB_IDX]
sDH_calib = sDH_all[CALIB_IDX]

z_check   = z_all[CHECK_IDX]
DH_check  = DH_all[CHECK_IDX]
sDH_check = sDH_all[CHECK_IDX]

# ── f_DE functions (dark energy density factor) ────────────────────────────
def fDE_lcdm(z, *args):
    return np.ones_like(np.asarray(z, float))

def fDE_cascade(z, w0):
    return (1.0 + np.asarray(z, float)) ** (3.0 * (1.0 + w0))

def fDE_cpl(z, w0, wa):
    z = np.asarray(z, float)
    return ((1+z) ** (3*(1+w0+wa))) * np.exp(-3*wa*z/(1+z))

# ── H(z) from Friedmann ────────────────────────────────────────────────────
def H_model(z, H0, Om, fDE_func, params):
    ODE = 1.0 - Om
    fDE = fDE_func(z, *params)
    return H0 * np.sqrt(Om * (1+z)**3 + ODE * fDE)

def DH_rd_model(z, H0, Om, fDE_func, params):
    return c_kms / (H_model(z, H0, Om, fDE_func, params) * rd_Mpc)

# ── Chi-squared fit on calibration points ─────────────────────────────────
def chi2(H0, Om, fDE_func, params, z_data, DH_data, sDH_data):
    pred = DH_rd_model(z_data, H0, Om, fDE_func, params)
    return np.sum(((pred - DH_data) / sDH_data) ** 2)

# Grid search for (H0, Om, model_params)
H0_grid  = np.linspace(60.0, 75.0, 60)
Om_grid  = np.linspace(0.25, 0.38, 55)
# Cascade: no-phantom theorem enforces w0 > -1 (hard structural prior)
w0_grid_cascade = np.linspace(-0.999, 0.0, 60)
# CPL: unconstrained (may go phantom)
w0_grid_cpl = np.linspace(-1.2,  0.0, 60)
wa_grid  = np.linspace(-2.0,  1.5, 55)

def fit_model(fDE_func, param_grids, label):
    best_chi2 = 1e30
    best_pars = None

    if len(param_grids) == 0:
        # LCDM: only fit H0, Om
        for H0 in H0_grid:
            for Om in Om_grid:
                c2 = chi2(H0, Om, fDE_func, [], z_calib, DH_calib, sDH_calib)
                if c2 < best_chi2:
                    best_chi2 = c2
                    best_pars = (H0, Om)
    elif len(param_grids) == 1:
        # CASCADE: fit H0, Om, w0
        for H0 in H0_grid:
            for Om in Om_grid:
                for w0 in param_grids[0]:
                    c2 = chi2(H0, Om, fDE_func, [w0], z_calib, DH_calib, sDH_calib)
                    if c2 < best_chi2:
                        best_chi2 = c2
                        best_pars = (H0, Om, w0)
    elif len(param_grids) == 2:
        # CPL: fit H0, Om, w0, wa
        for H0 in H0_grid:
            for Om in Om_grid:
                for w0 in param_grids[0]:
                    for wa in param_grids[1]:
                        c2 = chi2(H0, Om, fDE_func, [w0, wa],
                                  z_calib, DH_calib, sDH_calib)
                        if c2 < best_chi2:
                            best_chi2 = c2
                            best_pars = (H0, Om, w0, wa)

    return best_chi2, best_pars

print("=" * 68)
print("CASCADE RETRODICTION TEST")
print("=" * 68)
print(f"\nCALIBRATION on:  z = {z_calib}  (3 lowest-z DESI points)")
print(f"BLIND PREDICTION: z = {z_check}  (3 highest-z DESI points)")
print(f"\nFitting models...\n")

# Fit all three models
chi2_lcdm_cal, pars_lcdm = fit_model(fDE_lcdm, [], "LCDM")
print(f"  LCDM    fit done: chi2_cal = {chi2_lcdm_cal:.3f}")

chi2_cas_cal,  pars_cas  = fit_model(fDE_cascade, [w0_grid_cascade], "CASCADE")
print(f"  CASCADE fit done: chi2_cal = {chi2_cas_cal:.3f}")

chi2_cpl_cal,  pars_cpl  = fit_model(fDE_cpl, [w0_grid_cpl, wa_grid], "CPL")
print(f"  CPL     fit done: chi2_cal = {chi2_cpl_cal:.3f}")

# Extract best-fit parameters
H0_l, Om_l            = pars_lcdm[0], pars_lcdm[1]
H0_c, Om_c, w0_c      = pars_cas[0],  pars_cas[1],  pars_cas[2]
H0_p, Om_p, w0_p, wa_p = pars_cpl[0], pars_cpl[1],  pars_cpl[2], pars_cpl[3]

# ── Compute w(z) for each model ───────────────────────────────────────────
# Cascade: w(z) = w0 (constant in quiescent)
# CPL: w(z) = w0 + wa*z/(1+z)
# Check for phantom crossing in CPL
z_fine = np.linspace(0, 2.5, 500)
w_cpl_fine = w0_p + wa_p * z_fine / (1 + z_fine)
phantom_cross_z = None
if np.any(w_cpl_fine < -1.0):
    idx_cross = np.where(w_cpl_fine < -1.0)[0]
    phantom_cross_z = z_fine[idx_cross[0]]

# ── Blind prediction at high-z check points ────────────────────────────────
DH_pred_lcdm = DH_rd_model(z_check, H0_l, Om_l, fDE_lcdm,    [])
DH_pred_cas  = DH_rd_model(z_check, H0_c, Om_c, fDE_cascade,  [w0_c])
DH_pred_cpl  = DH_rd_model(z_check, H0_p, Om_p, fDE_cpl,      [w0_p, wa_p])

# Residuals: (predicted - actual) / sigma
res_lcdm = (DH_pred_lcdm - DH_check) / sDH_check
res_cas  = (DH_pred_cas  - DH_check) / sDH_check
res_cpl  = (DH_pred_cpl  - DH_check) / sDH_check

chi2_lcdm_check = np.sum(res_lcdm**2)
chi2_cas_check  = np.sum(res_cas**2)
chi2_cpl_check  = np.sum(res_cpl**2)

# ── Print results ──────────────────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"BEST-FIT PARAMETERS (calibrated on low-z)")
print(f"{'='*68}")
print(f"\n  LCDM:    H0={H0_l:.1f}, Om={Om_l:.3f},  w=-1 (fixed)")
print(f"  CASCADE: H0={H0_c:.1f}, Om={Om_c:.3f},  w0={w0_c:.3f}  (w >= -1 always)")
print(f"  CPL:     H0={H0_p:.1f}, Om={Om_p:.3f},  w0={w0_p:.3f}, wa={wa_p:.3f}")
if phantom_cross_z:
    print(f"           CPL phantom crossing at z = {phantom_cross_z:.3f}")

print(f"\n{'='*68}")
print(f"CALIBRATION chi-squared (3 low-z points, lower = better fit)")
print(f"{'='*68}")
print(f"  LCDM:    chi2 = {chi2_lcdm_cal:.3f}")
print(f"  CASCADE: chi2 = {chi2_cas_cal:.3f}")
print(f"  CPL:     chi2 = {chi2_cpl_cal:.3f}")

print(f"\n{'='*68}")
print(f"BLIND PREDICTION: residuals at HIGH-Z check points")
print(f"(model was NOT shown these points during calibration)")
print(f"{'='*68}")
print(f"\n{'z':>6}  {'DH/rd (actual)':>14}  {'LCDM pred':>10}  {'Res(sigma)':>10}  "
      f"{'CAS pred':>10}  {'Res(sigma)':>10}  {'CPL pred':>10}  {'Res(sigma)':>10}")
print("-" * 100)
for i, (z, dh_act, sdh) in enumerate(zip(z_check, DH_check, sDH_check)):
    print(f"{z:>6.3f}  {dh_act:>14.3f}  "
          f"{DH_pred_lcdm[i]:>10.3f}  {res_lcdm[i]:>+10.2f}s  "
          f"{DH_pred_cas[i]:>10.3f}  {res_cas[i]:>+10.2f}s  "
          f"{DH_pred_cpl[i]:>10.3f}  {res_cpl[i]:>+10.2f}s")

print(f"\n  Prediction chi2 (3 high-z points):")
print(f"    LCDM:    {chi2_lcdm_check:.3f}")
print(f"    CASCADE: {chi2_cas_check:.3f}  ({'BETTER' if chi2_cas_check < chi2_lcdm_check else 'worse'} than LCDM)")
print(f"    CPL:     {chi2_cpl_check:.3f}  ({'BETTER' if chi2_cpl_check < chi2_lcdm_check else 'worse'} than LCDM)")

# ── Verdict ────────────────────────────────────────────────────────────────
best_pred = min(
    ("LCDM",    chi2_lcdm_check),
    ("CASCADE", chi2_cas_check),
    ("CPL",     chi2_cpl_check),
    key=lambda x: x[1]
)
print(f"\n  Best blind prediction: {best_pred[0]}  (chi2={best_pred[1]:.3f})")

# Specifically: Lya QSO point z=2.33 -- the deepest check
i_lya = 2   # index 5 in full data, index 2 in check subset
print(f"\n  Deep check (Lya QSO, z=2.33):")
print(f"    Actual  DH/rd = {DH_check[i_lya]:.3f} +/- {sDH_check[i_lya]:.3f}")
print(f"    LCDM    pred  = {DH_pred_lcdm[i_lya]:.3f}  ({res_lcdm[i_lya]:+.2f} sigma)")
print(f"    CASCADE pred  = {DH_pred_cas[i_lya]:.3f}  ({res_cas[i_lya]:+.2f} sigma)")
print(f"    CPL     pred  = {DH_pred_cpl[i_lya]:.3f}  ({res_cpl[i_lya]:+.2f} sigma)")

# ── Plot ───────────────────────────────────────────────────────────────────
z_plot = np.linspace(0.3, 2.6, 400)

DH_lcdm_curve = DH_rd_model(z_plot, H0_l, Om_l, fDE_lcdm,   [])
DH_cas_curve  = DH_rd_model(z_plot, H0_c, Om_c, fDE_cascade, [w0_c])
DH_cpl_curve  = DH_rd_model(z_plot, H0_p, Om_p, fDE_cpl,     [w0_p, wa_p])

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#0a0a0a')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

ax_kw = dict(color='white', fontsize=10)
tk_kw = dict(colors='#aaaaaa', labelsize=8)
gr_kw = dict(alpha=0.18, color='#444444')

# ── P1: DH/rd curves vs DESI ──────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('#111111')

# Model curves
ax1.plot(z_plot, DH_lcdm_curve, color='#888888', lw=2, ls='--',
         label=f'$\\Lambda$CDM  (H0={H0_l:.1f}, $\\Omega_m$={Om_l:.3f})')
ax1.plot(z_plot, DH_cas_curve,  color='#ff6b35', lw=2.5,
         label=f'CASCADE  $w_0={w0_c:.3f}$  (w>-1 always)')
ax1.plot(z_plot, DH_cpl_curve,  color='#aa88ff', lw=2, ls=':',
         label=f'CPL  $w_0={w0_p:.3f}$, $w_a={wa_p:.3f}$')

# Calibration points (solid circles)
ax1.errorbar(z_calib, DH_calib, yerr=sDH_calib,
             fmt='o', color='#00d4ff', ms=10, elinewidth=2, capsize=5,
             label='DESI calibration (low-z)', zorder=8)

# Check points (hollow diamonds)
ax1.errorbar(z_check, DH_check, yerr=sDH_check,
             fmt='D', color='#ffdd57', ms=10, elinewidth=2, capsize=5,
             mfc='none', mew=2, label='DESI blind check (high-z)', zorder=8)

# Shade blind region
ax1.axvspan(z_check[0] - 0.05, z_check[-1] + 0.1, alpha=0.07, color='#ffdd57')
ax1.text(1.8, ax1.get_ylim()[1] if ax1.get_ylim()[1] != 0 else 22,
         'BLIND\nPREDICTION\nZONE', color='#ffdd57', fontsize=9, ha='center')

# Dividing line
ax1.axvline(0.93 + 0.05, color='white', lw=1, ls=':', alpha=0.5)

ax1.set_xlabel('Redshift $z$', **ax_kw)
ax1.set_ylabel('$D_H / r_d$', **ax_kw)
ax1.set_title(
    'RETRODICTION TEST: Calibrate on Low-z, Predict Backward to High-z\n'
    'Solid circles = calibration data  |  Hollow diamonds = blind prediction targets',
    color='white', fontsize=12, pad=8)
ax1.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#444444',
           labelcolor='white', loc='upper right')
ax1.tick_params(axis='both', **tk_kw)
ax1.grid(True, **gr_kw)
ax1.spines[['bottom','left','top','right']].set_color('#333333')

# ── P2: Residuals at check points ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#111111')

x_pos = np.arange(len(z_check))
w_bar = 0.25
ax2.bar(x_pos - w_bar, res_lcdm,   width=w_bar, color='#888888', alpha=0.85,
        label='$\\Lambda$CDM')
ax2.bar(x_pos,          res_cas,   width=w_bar, color='#ff6b35', alpha=0.85,
        label='CASCADE')
ax2.bar(x_pos + w_bar,  res_cpl,   width=w_bar, color='#aa88ff', alpha=0.85,
        label='CPL')

ax2.axhline(0,    color='white', lw=1.2, alpha=0.6)
ax2.axhline( 1.0, color='#ffdd57', lw=1, ls='--', alpha=0.5, label='$\\pm$1$\\sigma$')
ax2.axhline(-1.0, color='#ffdd57', lw=1, ls='--', alpha=0.5)
ax2.axhline( 2.0, color='#ff4444', lw=0.8, ls=':', alpha=0.4, label='$\\pm$2$\\sigma$')
ax2.axhline(-2.0, color='#ff4444', lw=0.8, ls=':', alpha=0.4)

ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'z={z:.3f}' for z in z_check], color='#aaaaaa', fontsize=8)
ax2.set_ylabel('Residual (sigma)', **ax_kw)
ax2.set_title('Blind Prediction Residuals', color='white', fontsize=11)
ax2.legend(fontsize=8, facecolor='#1a1a1a', edgecolor='#444444', labelcolor='white')
ax2.tick_params(axis='both', **tk_kw)
ax2.grid(True, axis='y', **gr_kw)
ax2.spines[['bottom','left','top','right']].set_color('#333333')

# Annotation
for i, (rl, rc, rp) in enumerate(zip(res_lcdm, res_cas, res_cpl)):
    ax2.text(i - w_bar, rl + 0.1 * np.sign(rl), f'{rl:+.2f}',
             color='white', fontsize=7, ha='center')
    ax2.text(i,         rc + 0.1 * np.sign(rc), f'{rc:+.2f}',
             color='white', fontsize=7, ha='center')
    ax2.text(i + w_bar, rp + 0.1 * np.sign(rp), f'{rp:+.2f}',
             color='white', fontsize=7, ha='center')

# ── P3: w(z) for each model ───────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('#111111')

ax3.axhline(w0_c, color='#ff6b35', lw=2.5, label=f'CASCADE $w_0={w0_c:.3f}$ (constant)')
ax3.axhline(-1.0, color='#888888', lw=2, ls='--', label='$\\Lambda$CDM $w=-1$')
ax3.plot(z_fine, w_cpl_fine, color='#aa88ff', lw=2,
         label=f'CPL $w_0={w0_p:.3f}$, $w_a={wa_p:.3f}$')

if phantom_cross_z:
    ax3.axvline(phantom_cross_z, color='#ff4444', lw=1, ls=':', alpha=0.7)
    ax3.text(phantom_cross_z + 0.05, -1.05,
             f'CPL phantom\ncrossing z={phantom_cross_z:.2f}',
             color='#ff4444', fontsize=7, alpha=0.9)

ax3.axhspan(-2.0, -1.0, alpha=0.07, color='red')
ax3.text(1.2, -1.5, 'PHANTOM FORBIDDEN\n(Cascade Theorem 2)',
         color='#ff4444', fontsize=8, ha='center', alpha=0.9)

# Mark calibration and check z ranges
for z in z_calib:
    ax3.axvline(z, color='#00d4ff', lw=0.8, alpha=0.3)
for z in z_check:
    ax3.axvline(z, color='#ffdd57', lw=0.8, alpha=0.3)

ax3.set_xlabel('Redshift $z$', **ax_kw)
ax3.set_ylabel('$w(z)$', **ax_kw)
ax3.set_title('$w(z)$ — Three Models', color='white', fontsize=11)
ax3.legend(fontsize=8, facecolor='#1a1a1a', edgecolor='#444444', labelcolor='white')
ax3.tick_params(axis='both', **tk_kw)
ax3.grid(True, **gr_kw)
ax3.set_ylim(-2.0, 0.5)
ax3.spines[['bottom','left','top','right']].set_color('#333333')

# Summary table in figure
summary_txt = (
    f"CALIBRATION (low-z)      chi2\n"
    f"  LCDM                   {chi2_lcdm_cal:.2f}\n"
    f"  CASCADE                {chi2_cas_cal:.2f}\n"
    f"  CPL                    {chi2_cpl_cal:.2f}\n\n"
    f"BLIND PREDICTION (high-z) chi2\n"
    f"  LCDM                   {chi2_lcdm_check:.2f}\n"
    f"  CASCADE                {chi2_cas_check:.2f}\n"
    f"  CPL                    {chi2_cpl_check:.2f}\n\n"
    f"Best predictor: {best_pred[0]}"
)
ax3.text(0.02, 0.98, summary_txt, transform=ax3.transAxes,
         color='#dddddd', fontsize=7.5, va='top', ha='left',
         fontfamily='monospace',
         bbox=dict(facecolor='#1a1a1a', alpha=0.85, edgecolor='#444444'))

fig.suptitle(
    'CASCADE RETRODICTION: Train on 3 Low-z Points  ->  Predict 3 High-z Points\n'
    'Lya QSO (z=2.33) is the final blind check — the deepest DESI point',
    color='white', fontsize=13, y=0.998, fontweight='bold')

out = r"F:\A mathematical model\figures\cascade_retrodiction.png"
plt.savefig(out, dpi=155, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

print(f"\nFigure: {out}")
print(f"\n{'='*68}")
print("FINAL VERDICT")
print(f"{'='*68}")
print(f"  CASCADE w0 (calibrated):       {w0_c:.3f}")
print(f"  CASCADE phantom crossing:      NONE  (w = {w0_c:.3f} > -1 always)")
if phantom_cross_z:
    print(f"  CPL phantom crossing:          z = {phantom_cross_z:.3f}")
else:
    print(f"  CPL phantom crossing:          None")
print(f"\n  Lya QSO prediction (z=2.33):")
print(f"    Cascade residual:   {res_cas[i_lya]:+.2f} sigma")
print(f"    LCDM residual:      {res_lcdm[i_lya]:+.2f} sigma")
print(f"    CPL residual:       {res_cpl[i_lya]:+.2f} sigma")
print(f"\n  Blind prediction winner:  {best_pred[0]}  (chi2={best_pred[1]:.3f})")
print(f"{'='*68}")
