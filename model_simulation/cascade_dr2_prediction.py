"""
cascade_dr2_prediction.py
=========================
True cross-release blind prediction:
  TRAIN  on DESI Year 1 BAO (April 2024, arXiv: 2404.03002)  -- 6 points
  PREDICT DESI DR2     BAO (March  2025, arXiv: 2503.14738)  -- 6 points

The model is calibrated on a completely separate observational campaign
and then asked to predict the results of a later one it has never seen.

Three models compared:
  LCDM    : w(z) = -1 (cosmological constant)
  CASCADE : f_DE(z) = (1+z)^{3(1+w0)},  w0 free, w >= -1 always (no-phantom theorem)
  CPL     : f_DE(z) = (1+z)^{3(1+w0+wa)} exp(-3wa z/(1+z)), w0 and wa free

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
rd_Mpc  = 147.09       # Mpc (Planck 2018 sound horizon)

# ── TRAINING DATA: DESI Year 1 BAO (April 2024, arXiv: 2404.03002) ────────
# z, DH/rd, sigma(DH/rd)
y1 = np.array([
    [0.510, 20.98, 0.61],   # LRG1
    [0.706, 20.08, 0.60],   # LRG2
    [0.930, 17.88, 0.35],   # LRG3+ELG1
    [1.317, 13.82, 0.42],   # ELG2
    [1.491, 13.23, 0.55],   # QSO
    [2.330,  8.52, 0.17],   # Lya QSO
])

# ── BLIND TARGET: DESI DR2 BAO (March 2025, arXiv: 2503.14738) ────────────
# z, DH/rd, sigma(DH/rd)
dr2 = np.array([
    [0.510, 21.863, 0.425],  # LRG1
    [0.706, 19.455, 0.330],  # LRG2
    [0.934, 17.641, 0.193],  # LRG3+ELG1
    [1.321, 14.176, 0.221],  # ELG2
    [1.484, 12.817, 0.516],  # QSO
    [2.330,  8.632, 0.101],  # Lya QSO
])

z_train   = y1[:, 0]
DH_train  = y1[:, 1]
sDH_train = y1[:, 2]

z_pred    = dr2[:, 0]
DH_dr2    = dr2[:, 1]
sDH_dr2   = dr2[:, 2]

# ── f_DE functions ─────────────────────────────────────────────────────────
def fDE_lcdm(z, *args):
    return np.ones_like(np.asarray(z, float))

def fDE_cascade(z, w0):
    return (1.0 + np.asarray(z, float)) ** (3.0 * (1.0 + w0))

def fDE_cpl(z, w0, wa):
    z = np.asarray(z, float)
    return ((1+z) ** (3*(1+w0+wa))) * np.exp(-3*wa*z/(1+z))

def H_model(z, H0, Om, fDE_func, params):
    ODE = 1.0 - Om
    return H0 * np.sqrt(Om*(1+z)**3 + ODE*fDE_func(z, *params))

def DH_rd_model(z, H0, Om, fDE_func, params):
    return c_kms / (H_model(z, H0, Om, fDE_func, params) * rd_Mpc)

def chi2(H0, Om, fDE_func, params, z_data, DH_data, sDH_data):
    pred = DH_rd_model(z_data, H0, Om, fDE_func, params)
    return np.sum(((pred - DH_data) / sDH_data) ** 2)

# ── Grid search (calibrate on Year 1) ────────────────────────────────────
H0_grid    = np.linspace(60.0,  76.0, 65)
Om_grid    = np.linspace(0.24,  0.38, 60)
# Cascade: no-phantom theorem requires w > -1 strictly; use -0.999 as boundary
w0_cas_grid = np.linspace(-0.999, 0.0, 65)
# CPL: unconstrained (to see what CPL does when given full freedom)
w0_cpl_grid = np.linspace(-1.5,  0.0, 65)
wa_grid     = np.linspace(-2.0,  2.0, 55)

def fit_model(fDE_func, param_grids, label):
    best_chi2 = 1e30
    best_pars = None
    if len(param_grids) == 0:
        for H0 in H0_grid:
            for Om in Om_grid:
                c2 = chi2(H0, Om, fDE_func, [], z_train, DH_train, sDH_train)
                if c2 < best_chi2:
                    best_chi2, best_pars = c2, (H0, Om)
    elif len(param_grids) == 1:
        for H0 in H0_grid:
            for Om in Om_grid:
                for w0 in param_grids[0]:
                    c2 = chi2(H0, Om, fDE_func, [w0], z_train, DH_train, sDH_train)
                    if c2 < best_chi2:
                        best_chi2, best_pars = c2, (H0, Om, w0)
    elif len(param_grids) == 2:
        for H0 in H0_grid:
            for Om in Om_grid:
                for w0 in param_grids[0]:
                    for wa in param_grids[1]:
                        c2 = chi2(H0, Om, fDE_func, [w0, wa], z_train, DH_train, sDH_train)
                        if c2 < best_chi2:
                            best_chi2, best_pars = c2, (H0, Om, w0, wa)
    print(f"  {label:8}: chi2_train = {best_chi2:.3f}  |  params = {best_pars}")
    return best_chi2, best_pars

print("=" * 72)
print("CASCADE DR2 BLIND PREDICTION")
print("=" * 72)
print("\nTRAINING on:  DESI Year 1 BAO  (April 2024, arXiv: 2404.03002)")
print("PREDICTING:   DESI DR2 BAO     (March 2025, arXiv: 2503.14738)")
print(f"\nCalibrating all models on {len(z_train)} Year 1 points...\n")

chi2_l_tr, p_l = fit_model(fDE_lcdm,    [],                       "LCDM   ")
chi2_c_tr, p_c = fit_model(fDE_cascade, [w0_cas_grid],            "CASCADE")
chi2_p_tr, p_p = fit_model(fDE_cpl,     [w0_cpl_grid, wa_grid],   "CPL    ")

H0_l, Om_l               = p_l
H0_c, Om_c, w0_c         = p_c
H0_p, Om_p, w0_p, wa_p   = p_p

# ── Blind predictions at DR2 redshifts ────────────────────────────────────
DH_pred_l = DH_rd_model(z_pred, H0_l, Om_l, fDE_lcdm,   [])
DH_pred_c = DH_rd_model(z_pred, H0_c, Om_c, fDE_cascade, [w0_c])
DH_pred_p = DH_rd_model(z_pred, H0_p, Om_p, fDE_cpl,     [w0_p, wa_p])

res_l = (DH_pred_l - DH_dr2) / sDH_dr2
res_c = (DH_pred_c - DH_dr2) / sDH_dr2
res_p = (DH_pred_p - DH_dr2) / sDH_dr2

chi2_l_pred = np.sum(res_l**2)
chi2_c_pred = np.sum(res_c**2)
chi2_p_pred = np.sum(res_p**2)

# ── Report ─────────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("BEST-FIT PARAMETERS (trained on DESI Year 1)")
print(f"{'='*72}")
print(f"  LCDM:    H0={H0_l:.1f}, Om={Om_l:.3f}")
print(f"  CASCADE: H0={H0_c:.1f}, Om={Om_c:.3f},  w0={w0_c:.3f}  (w >= -1 always)")
print(f"  CPL:     H0={H0_p:.1f}, Om={Om_p:.3f},  w0={w0_p:.3f}, wa={wa_p:.3f}")

z_fine = np.linspace(0, 2.5, 500)
w_cpl_fine = w0_p + wa_p * z_fine / (1 + z_fine)
phantom_z  = None
if np.any(w_cpl_fine < -1.0):
    phantom_z = z_fine[np.where(w_cpl_fine < -1.0)[0][0]]
    print(f"  CPL phantom crossing at z = {phantom_z:.3f}")
else:
    print(f"  CPL: no phantom crossing")

print(f"\n{'='*72}")
print("BLIND PREDICTION vs DESI DR2 -- point by point")
print(f"{'='*72}")
print(f"\n{'z':>6}  {'DR2 actual':>12}  {'LCDM':>8} {'Res':>7}  "
      f"{'CASCADE':>8} {'Res':>7}  {'CPL':>8} {'Res':>7}")
print("-" * 80)
for i in range(len(z_pred)):
    print(f"{z_pred[i]:>6.3f}  {DH_dr2[i]:>8.3f}+/-{sDH_dr2[i]:.3f}  "
          f"{DH_pred_l[i]:>8.3f} {res_l[i]:>+6.2f}s  "
          f"{DH_pred_c[i]:>8.3f} {res_c[i]:>+6.2f}s  "
          f"{DH_pred_p[i]:>8.3f} {res_p[i]:>+6.2f}s")

print(f"\n{'='*72}")
print("TOTAL chi-squared on DESI DR2 (all 6 points, lower = better)")
print(f"{'='*72}")
print(f"  LCDM:    {chi2_l_pred:.3f}")
print(f"  CASCADE: {chi2_c_pred:.3f}  ({'BETTER' if chi2_c_pred < chi2_l_pred else 'worse'} than LCDM)")
print(f"  CPL:     {chi2_p_pred:.3f}  ({'BETTER' if chi2_p_pred < chi2_l_pred else 'worse'} than LCDM)")

best = min([("LCDM", chi2_l_pred), ("CASCADE", chi2_c_pred), ("CPL", chi2_p_pred)],
           key=lambda x: x[1])
print(f"\n  Best cross-release predictor: {best[0]}  (chi2 = {best[1]:.3f})")

# Lya QSO (deepest, z=2.33) -- same redshift in both releases
i_lya = 5
print(f"\n  Deepest point check (Lya QSO, z=2.330):")
print(f"    DR2 actual: {DH_dr2[i_lya]:.3f} +/- {sDH_dr2[i_lya]:.3f}")
print(f"    LCDM pred:  {DH_pred_l[i_lya]:.3f}  ({res_l[i_lya]:+.2f} sigma)")
print(f"    CASCADE:    {DH_pred_c[i_lya]:.3f}  ({res_c[i_lya]:+.2f} sigma)")
print(f"    CPL pred:   {DH_pred_p[i_lya]:.3f}  ({res_p[i_lya]:+.2f} sigma)")

# ── Plot ───────────────────────────────────────────────────────────────────
z_plot = np.linspace(0.25, 2.55, 500)
DH_l_curve = DH_rd_model(z_plot, H0_l, Om_l, fDE_lcdm,   [])
DH_c_curve = DH_rd_model(z_plot, H0_c, Om_c, fDE_cascade, [w0_c])
DH_p_curve = DH_rd_model(z_plot, H0_p, Om_p, fDE_cpl,     [w0_p, wa_p])

fig = plt.figure(figsize=(17, 13))
fig.patch.set_facecolor('#080808')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

def ax_style(ax):
    ax.set_facecolor('#111111')
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    ax.grid(True, alpha=0.15, color='#444444')
    for s in ax.spines.values():
        s.set_color('#333333')
    return ax

# P1: DH/rd main plot
ax1 = ax_style(fig.add_subplot(gs[0, :]))
ax1.plot(z_plot, DH_l_curve, color='#888888', lw=2,   ls='--',
         label=f'LCDM (H0={H0_l:.1f}, Om={Om_l:.3f})')
ax1.plot(z_plot, DH_c_curve, color='#ff6b35', lw=2.5,
         label=f'CASCADE w0={w0_c:.3f} (w > -1 always)')
ax1.plot(z_plot, DH_p_curve, color='#aa88ff', lw=2,   ls=':',
         label=f'CPL w0={w0_p:.3f}, wa={wa_p:.3f}')

ax1.errorbar(z_train, DH_train, yerr=sDH_train,
             fmt='o', color='#00d4ff', ms=9, elinewidth=2, capsize=5, zorder=8,
             label='DESI Year 1 (training)')
ax1.errorbar(z_pred, DH_dr2, yerr=sDH_dr2,
             fmt='D', color='#ffdd57', ms=9, elinewidth=2, capsize=5,
             mfc='none', mew=2, zorder=8, label='DESI DR2 (blind target)')

ax1.axvspan(0.25, 2.55, alpha=0.0)
ax1.set_xlabel('Redshift z', color='white', fontsize=10)
ax1.set_ylabel('D_H / r_d', color='white', fontsize=10)
ax1.set_title(
    'CROSS-RELEASE BLIND PREDICTION: Train on DESI Year 1 (Apr 2024)  ->  Predict DESI DR2 (Mar 2025)\n'
    'Solid circles = training data  |  Hollow diamonds = blind targets (different observational campaign)',
    color='white', fontsize=11, pad=8)
ax1.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#444444', labelcolor='white',
           loc='upper right')

# P2: Residuals at DR2 points
ax2 = ax_style(fig.add_subplot(gs[1, 0]))
x_pos = np.arange(len(z_pred))
w_bar = 0.24
ax2.bar(x_pos - w_bar, res_l, width=w_bar, color='#888888', alpha=0.85, label='LCDM')
ax2.bar(x_pos,          res_c, width=w_bar, color='#ff6b35', alpha=0.85, label='CASCADE')
ax2.bar(x_pos + w_bar,  res_p, width=w_bar, color='#aa88ff', alpha=0.85, label='CPL')
ax2.axhline(0,    color='white',   lw=1.2, alpha=0.6)
ax2.axhline( 1.0, color='#ffdd57', lw=1,   ls='--', alpha=0.5)
ax2.axhline(-1.0, color='#ffdd57', lw=1,   ls='--', alpha=0.5)
ax2.axhline( 2.0, color='#ff4444', lw=0.8, ls=':',  alpha=0.4)
ax2.axhline(-2.0, color='#ff4444', lw=0.8, ls=':',  alpha=0.4)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'z={z:.3f}' for z in z_pred], color='#aaaaaa', fontsize=7.5)
ax2.set_ylabel('Residual (sigma)', color='white', fontsize=10)
ax2.set_title('DR2 Prediction Residuals by Point', color='white', fontsize=11)
ax2.legend(fontsize=8, facecolor='#1a1a1a', edgecolor='#444444', labelcolor='white')
for i, (rl, rc, rp) in enumerate(zip(res_l, res_c, res_p)):
    ax2.text(i - w_bar, rl + 0.08*np.sign(rl) if abs(rl) > 0.05 else 0.1,
             f'{rl:+.2f}', color='white', fontsize=6.5, ha='center')
    ax2.text(i,          rc + 0.08*np.sign(rc) if abs(rc) > 0.05 else 0.1,
             f'{rc:+.2f}', color='white', fontsize=6.5, ha='center')
    ax2.text(i + w_bar,  rp + 0.08*np.sign(rp) if abs(rp) > 0.05 else 0.1,
             f'{rp:+.2f}', color='white', fontsize=6.5, ha='center')

# P3: w(z)
ax3 = ax_style(fig.add_subplot(gs[1, 1]))
ax3.axhline(w0_c,  color='#ff6b35', lw=2.5, label=f'CASCADE w0={w0_c:.3f} (constant)')
ax3.axhline(-1.0,  color='#888888', lw=2,   ls='--', label='LCDM w=-1')
ax3.plot(z_fine, w_cpl_fine, color='#aa88ff', lw=2,
         label=f'CPL w0={w0_p:.3f}, wa={wa_p:.3f}')
if phantom_z:
    ax3.axvline(phantom_z, color='#ff4444', lw=1, ls=':', alpha=0.7)
    ax3.text(phantom_z + 0.05, -1.08,
             f'CPL phantom\nz={phantom_z:.2f}', color='#ff4444', fontsize=7)
ax3.axhspan(-2.0, -1.0, alpha=0.07, color='red')
ax3.text(1.25, -1.5, 'PHANTOM\nFORBIDDEN', color='#ff4444', fontsize=8,
         ha='center', alpha=0.9)

# chi-squared summary box
summary = (
    f"Chi-sq on DESI Year 1 (training)\n"
    f"  LCDM    {chi2_l_tr:.2f}\n"
    f"  CASCADE {chi2_c_tr:.2f}\n"
    f"  CPL     {chi2_p_tr:.2f}\n\n"
    f"Chi-sq on DESI DR2 (blind)\n"
    f"  LCDM    {chi2_l_pred:.2f}\n"
    f"  CASCADE {chi2_c_pred:.2f}\n"
    f"  CPL     {chi2_p_pred:.2f}\n\n"
    f"Winner: {best[0]}"
)
ax3.text(0.02, 0.98, summary, transform=ax3.transAxes, color='#dddddd',
         fontsize=7.5, va='top', ha='left', fontfamily='monospace',
         bbox=dict(facecolor='#1a1a1a', alpha=0.88, edgecolor='#444444'))

ax3.set_xlabel('Redshift z', color='white', fontsize=10)
ax3.set_ylabel('w(z)', color='white', fontsize=10)
ax3.set_title('w(z) -- Three Models', color='white', fontsize=11)
ax3.legend(fontsize=8, facecolor='#1a1a1a', edgecolor='#444444', labelcolor='white')
ax3.set_ylim(-2.0, 0.5)

fig.suptitle(
    'CASCADE CROSS-RELEASE PREDICTION\n'
    'Trained on DESI Year 1 (April 2024)  --  Predicting DESI DR2 (March 2025)',
    color='white', fontsize=13, y=0.998, fontweight='bold')

out = r"F:\A mathematical model\figures\cascade_dr2_prediction.png"
plt.savefig(out, dpi=155, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"\nFigure: {out}")
print(f"\n{'='*72}")
print("VERDICT")
print(f"{'='*72}")
print(f"  Model trained on:          DESI Year 1 BAO  (April 2024)")
print(f"  Predicted:                 DESI DR2 BAO     (March 2025)")
print(f"  Cascade chi-sq on DR2:     {chi2_c_pred:.3f}")
print(f"  LCDM    chi-sq on DR2:     {chi2_l_pred:.3f}")
print(f"  CPL     chi-sq on DR2:     {chi2_p_pred:.3f}")
print(f"  No-phantom theorem holds:   w0 = {w0_c:.3f} > -1  (constrained by theorem)")
print(f"  Best cross-release model:   {best[0]}")
