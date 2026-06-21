"""
cascade_dr3_prediction.py
=========================
Sealed prediction for DESI DR3 (expected 2027).

Protocol:
  TRAINING DATA  : DESI DR2 BAO (March 2025, arXiv: 2503.14738)
  PREDICTION FOR : DESI DR3 BAO (expected ~2027, full 5-year survey)

The cascade model is calibrated on DR2, then its best-fit parameters
are used to predict D_H/r_d at the same redshift bins for DR3.

DR3 error bars are projected from DR2 assuming the full 5-year DESI
dataset (~47 million objects) vs DR2 (~14 million objects), giving
a statistical improvement factor of sqrt(47/14) ~ 1.83 in precision.
Projected DR3 sigma ~ DR2 sigma / 1.83  (conservative; systematics
will likely set a floor).

Key falsifiable predictions recorded here:
  1. D_H/r_d values at 6 redshift bins (cascade best-fit from DR2)
  2. No phantom crossing: w(z) > -1 at all redshifts
  3. If DR3 confirms CPL-style phantom crossing, cascade is ruled out in
     this dark energy application.
  4. If DR3 finds w consistent with -1 to -0.8 (no phantom), cascade
     prediction is confirmed.

Prediction sealed: June 21, 2026
Author: Shiv Goswami
To be compared against DESI DR3 upon release (~2027).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

# ── Constants ──────────────────────────────────────────────────────────────
c_kms  = 299792.458
rd_Mpc = 147.09

PREDICTION_DATE = "21 June 2026"
DR3_EXPECTED    = "~2027"
DR3_REFERENCE   = "DESI full 5-year survey (~47 million objects)"

# ── TRAINING: DESI DR2 BAO (March 2025, arXiv: 2503.14738) ────────────────
dr2 = np.array([
    [0.510, 21.863, 0.425],
    [0.706, 19.455, 0.330],
    [0.934, 17.641, 0.193],
    [1.321, 14.176, 0.221],
    [1.484, 12.817, 0.516],
    [2.330,  8.632, 0.101],
])
z_dr2   = dr2[:, 0]
DH_dr2  = dr2[:, 1]
sDH_dr2 = dr2[:, 2]

# ── DR3 projected error bars ───────────────────────────────────────────────
# DR2: ~14 million objects. DR3: ~47 million.
# Statistical improvement: sqrt(47/14) ~ 1.83
# Systematic floor assumed ~0.08 Mpc (conservative)
stat_improvement = np.sqrt(47.0 / 14.0)
sys_floor        = 0.08   # conservative systematic floor
sDH_dr3_proj = np.maximum(sDH_dr2 / stat_improvement, sys_floor)

# ── Model functions ────────────────────────────────────────────────────────
def fDE_lcdm(z, *a):    return np.ones_like(np.asarray(z, float))
def fDE_cascade(z, w0): return (1+np.asarray(z,float))**(3*(1+w0))
def fDE_cpl(z, w0, wa):
    z = np.asarray(z, float)
    return (1+z)**(3*(1+w0+wa)) * np.exp(-3*wa*z/(1+z))

def H_mod(z, H0, Om, f, p):
    return H0 * np.sqrt(Om*(1+z)**3 + (1-Om)*f(z,*p))

def DH_rd(z, H0, Om, f, p):
    return c_kms / (H_mod(z, H0, Om, f, p) * rd_Mpc)

def chi2(H0, Om, f, p, zd, dh, sdh):
    return np.sum(((DH_rd(zd, H0, Om, f, p) - dh)/sdh)**2)

# ── Grid search on DR2 ─────────────────────────────────────────────────────
H0g     = np.linspace(60.0,  76.0, 55)
Omg     = np.linspace(0.24,  0.38, 50)
w0g_cas = np.linspace(-0.999, 0.0, 55)   # cascade: w > -1 enforced
w0g_cpl = np.linspace(-1.5,   0.0, 55)   # CPL: unconstrained
wag     = np.linspace(-2.5,   2.0, 50)

def fit(func, grids, label):
    best, bp = 1e30, None
    if len(grids) == 0:
        for H0 in H0g:
            for Om in Omg:
                c = chi2(H0, Om, func, [], z_dr2, DH_dr2, sDH_dr2)
                if c < best: best, bp = c, (H0, Om)
    elif len(grids) == 1:
        for H0 in H0g:
            for Om in Omg:
                for w0 in grids[0]:
                    c = chi2(H0, Om, func, [w0], z_dr2, DH_dr2, sDH_dr2)
                    if c < best: best, bp = c, (H0, Om, w0)
    else:
        for H0 in H0g:
            for Om in Omg:
                for w0 in grids[0]:
                    for wa in grids[1]:
                        c = chi2(H0, Om, func, [w0, wa], z_dr2, DH_dr2, sDH_dr2)
                        if c < best: best, bp = c, (H0, Om, w0, wa)
    print(f"  {label:8}: chi2_dr2 = {best:.3f}  params = {tuple(round(x,4) for x in bp)}")
    return best, bp

print("=" * 72)
print("CASCADE DR3 SEALED PREDICTION")
print(f"Prediction date : {PREDICTION_DATE}")
print(f"To be opened vs : DESI DR3  {DR3_EXPECTED}")
print("=" * 72)
print(f"\nCalibrating on DESI DR2 (March 2025)...\n")

chi2_l, p_l = fit(fDE_lcdm,    [],               "LCDM   ")
chi2_c, p_c = fit(fDE_cascade, [w0g_cas],        "CASCADE")
chi2_p, p_p = fit(fDE_cpl,     [w0g_cpl, wag],  "CPL    ")

H0_l, Om_l             = p_l
H0_c, Om_c, w0_c       = p_c
H0_p, Om_p, w0_p, wa_p = p_p

# ── DR3 predictions ────────────────────────────────────────────────────────
# Predict at DR2 redshifts (DR3 will use same bins + possibly new ones)
DH_pred_l = DH_rd(z_dr2, H0_l, Om_l, fDE_lcdm,   [])
DH_pred_c = DH_rd(z_dr2, H0_c, Om_c, fDE_cascade, [w0_c])
DH_pred_p = DH_rd(z_dr2, H0_p, Om_p, fDE_cpl,     [w0_p, wa_p])

# CPL phantom check
z_fine     = np.linspace(0, 2.5, 1000)
w_cpl_fine = w0_p + wa_p * z_fine / (1 + z_fine)
phantom_z  = z_fine[np.where(w_cpl_fine < -1.0)[0][0]] if np.any(w_cpl_fine < -1.0) else None

print(f"\n{'='*72}")
print("CASCADE DR3 PREDICTION  (sealed {PREDICTION_DATE})")
print(f"{'='*72}")
print(f"\nCalibrated parameters from DR2:")
print(f"  CASCADE: H0={H0_c:.1f}, Om={Om_c:.3f}, w0={w0_c:.4f}  (w > -1, theorem)")
print(f"  LCDM:    H0={H0_l:.1f}, Om={Om_l:.3f}")
print(f"  CPL:     H0={H0_p:.1f}, Om={Om_p:.3f}, w0={w0_p:.3f}, wa={wa_p:.3f}")
if phantom_z:
    print(f"           CPL phantom crossing at z = {phantom_z:.3f} (unphysical)")

print(f"\n{'z':>6}  {'Cascade pred':>13}  {'LCDM pred':>10}  {'CPL pred':>10}  "
      f"{'DR3 proj sigma':>15}")
print("-" * 68)
for i in range(len(z_dr2)):
    print(f"{z_dr2[i]:>6.3f}  {DH_pred_c[i]:>13.4f}  {DH_pred_l[i]:>10.4f}"
          f"  {DH_pred_p[i]:>10.4f}  +/-{sDH_dr3_proj[i]:.4f}")

print(f"\n{'='*72}")
print("FALSIFIABLE PREDICTIONS (to be checked against DESI DR3)")
print(f"{'='*72}")
print(f"""
  1. w(z) > -1 at ALL redshifts in DR3.
     If DR3 confirms phantom crossing, cascade dark energy application is
     ruled out. If DR3 finds w consistent with [-1, -0.7], cascade stands.

  2. D_H/r_d at z = 2.330 (Lya QSO):
     CASCADE predicts  {DH_pred_c[-1]:.4f}
     LCDM predicts     {DH_pred_l[-1]:.4f}
     CPL predicts      {DH_pred_p[-1]:.4f}
     DR3 projected sigma: +/- {sDH_dr3_proj[-1]:.4f}
     If DR3 lands within +/- 1 sigma of cascade prediction: confirmed.

  3. The CPL phantom preference (wa = {wa_p:.2f}) will weaken or disappear
     in DR3 as error bars tighten. Phantom is noise absorption.
     If DR3 wa moves toward 0, cascade + no-phantom theorem is vindicated.

  4. Cascade w0 = {w0_c:.4f} (calibrated from DR2).
     DR3 best-fit w0 should be consistent with [{w0_c-0.15:.2f}, {w0_c+0.15:.2f}]
     if the cascade dark energy description is correct.
""")
print(f"  Prediction sealed: {PREDICTION_DATE}")
print(f"  Author: Shiv Goswami")
print(f"  Repository: github.com/Shiv2071/Discrete-cascade-model")

# ── Plot ───────────────────────────────────────────────────────────────────
z_plot = np.linspace(0.25, 2.6, 600)
DH_l_c  = DH_rd(z_plot, H0_l, Om_l, fDE_lcdm,   [])
DH_c_c  = DH_rd(z_plot, H0_c, Om_c, fDE_cascade, [w0_c])
DH_p_c  = DH_rd(z_plot, H0_p, Om_p, fDE_cpl,     [w0_p, wa_p])

fig = plt.figure(figsize=(17, 13))
fig.patch.set_facecolor('#080808')
gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32)

def ax_s(ax):
    ax.set_facecolor('#111111')
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    ax.grid(True, alpha=0.15, color='#444444')
    for s in ax.spines.values(): s.set_color('#333333')
    return ax

# P1: main prediction plot
ax1 = ax_s(fig.add_subplot(gs[0, :]))
ax1.plot(z_plot, DH_l_c,  color='#888888', lw=2,   ls='--',
         label=f'LCDM prediction (H0={H0_l:.1f})')
ax1.plot(z_plot, DH_c_c,  color='#ff6b35', lw=2.5,
         label=f'CASCADE prediction  w0={w0_c:.4f}  (w > -1, sealed)')
ax1.plot(z_plot, DH_p_c,  color='#aa88ff', lw=2,   ls=':',
         label=f'CPL prediction  w0={w0_p:.3f}, wa={wa_p:.3f}')

# DR2 training data
ax1.errorbar(z_dr2, DH_dr2, yerr=sDH_dr2,
             fmt='o', color='#00d4ff', ms=9, elinewidth=2, capsize=5, zorder=8,
             label='DESI DR2 (training, March 2025)')

# DR3 projected uncertainty bands around cascade prediction
for i in range(len(z_dr2)):
    ax1.errorbar(z_dr2[i] + 0.018, DH_pred_c[i], yerr=sDH_dr3_proj[i],
                 fmt='D', color='#ff6b35', ms=8, elinewidth=2, capsize=5,
                 mfc='none', mew=2, zorder=9,
                 label='CASCADE DR3 prediction +/- projected sigma' if i == 0 else '')

ax1.set_xlabel('Redshift z', color='white', fontsize=10)
ax1.set_ylabel('D_H / r_d', color='white', fontsize=10)
ax1.set_title(
    f'SEALED PREDICTION for DESI DR3 (~2027)  --  Calibrated on DESI DR2 (March 2025)\n'
    f'Hollow diamonds = cascade DR3 prediction with projected DR3 error bars',
    color='white', fontsize=11, pad=8)
ax1.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#444444', labelcolor='white',
           loc='upper right')

# P2: w(z) comparison
ax2 = ax_s(fig.add_subplot(gs[1, 0]))
ax2.axhline(w0_c,  color='#ff6b35', lw=2.5,
            label=f'CASCADE  w0={w0_c:.4f} (constant, sealed)')
ax2.axhline(-1.0,  color='#888888', lw=2, ls='--', label='LCDM  w=-1')
ax2.plot(z_fine, w_cpl_fine, color='#aa88ff', lw=2,
         label=f'CPL  w0={w0_p:.3f}, wa={wa_p:.3f}')
if phantom_z:
    ax2.axvline(phantom_z, color='#ff4444', lw=1.2, ls=':', alpha=0.8)
    ax2.text(phantom_z + 0.04, -1.1,
             f'CPL phantom\nz={phantom_z:.3f}', color='#ff4444', fontsize=7.5)
ax2.axhspan(-2.5, -1.0, alpha=0.07, color='red')
ax2.text(1.2, -1.6, 'PHANTOM\nFORBIDDEN\n(Cascade Theorem)', color='#ff4444',
         fontsize=8, ha='center')
ax2.set_xlabel('Redshift z', color='white', fontsize=10)
ax2.set_ylabel('w(z)', color='white', fontsize=10)
ax2.set_title('w(z): Cascade vs LCDM vs CPL', color='white', fontsize=11)
ax2.legend(fontsize=8, facecolor='#1a1a1a', edgecolor='#444444', labelcolor='white')
ax2.set_ylim(-2.5, 0.4)

# P3: prediction table
ax3 = ax_s(fig.add_subplot(gs[1, 1]))
ax3.axis('off')
rows = [['z', 'CASCADE', 'LCDM', 'CPL', 'DR3 sigma']]
for i in range(len(z_dr2)):
    rows.append([
        f'{z_dr2[i]:.3f}',
        f'{DH_pred_c[i]:.3f}',
        f'{DH_pred_l[i]:.3f}',
        f'{DH_pred_p[i]:.3f}',
        f'+/-{sDH_dr3_proj[i]:.3f}',
    ])
tbl = ax3.table(
    cellText=rows[1:], colLabels=rows[0],
    loc='center', cellLoc='center',
    colWidths=[0.13, 0.22, 0.20, 0.20, 0.25])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor('#1a1a1a' if r > 0 else '#2a2a2a')
    cell.set_edgecolor('#444444')
    cell.set_text_props(color='white' if c != 1 else '#ff6b35')
ax3.set_title('DR3 Predictions by Redshift', color='white', fontsize=11, pad=12)

# Seal stamp
fig.text(0.5, 0.01,
         f'SEALED  {PREDICTION_DATE}  |  Shiv Goswami  |  '
         f'github.com/Shiv2071/Discrete-cascade-model  |  '
         f'To be compared vs DESI DR3 {DR3_EXPECTED}',
         color='#555555', fontsize=8, ha='center')

fig.suptitle(
    f'CASCADE MODEL: SEALED DR3 PREDICTION  ({PREDICTION_DATE})\n'
    f'Trained on DESI DR2 (March 2025)  --  Locked before DR3 release',
    color='white', fontsize=13, y=0.998, fontweight='bold')

out = r"F:\A mathematical model\figures\cascade_dr3_prediction.png"
plt.savefig(out, dpi=155, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"\nFigure saved: {out}")
