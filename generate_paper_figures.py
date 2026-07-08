"""
generate_paper_figures.py
Generates the three figures for dark_energy_cascade_preprint.tex.

Themes:
  --theme light  (default) : white publication style, written to figures/
                             under the filenames the paper includes.
  --theme dark              : website style (shivgoswami.com), same filenames;
                             copy them elsewhere before regenerating light.
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

os.makedirs('figures', exist_ok=True)

_ap = argparse.ArgumentParser()
_ap.add_argument('--theme', choices=['light', 'dark'], default='light')
THEME = _ap.parse_args().theme

if THEME == 'dark':
    # ── Site colour palette (matches hsl values in globals.css) ──────────
    BG        = '#090b0f'   # --bg:           hsl(220 12% 4%)
    BG_AX     = '#0f1114'   # --bg-secondary: hsl(220 10% 7%) — axes area
    T_PRI     = '#f0f1f5'   # --text-primary
    T_SEC     = '#b0b3bc'   # --text-secondary
    T_MUT     = '#6f7278'   # --text-muted
    ACCENT    = '#8ba3bb'   # cascade (blue-steel)
    RED_MUT   = '#b87878'   # muted red for CPL
    GRAY_MUT  = '#5a6370'   # muted gray for LCDM
    EDGE      = '#1e2530'   # axes/bar/legend edges
    REFLINE   = '#2a3545'   # reference lines
    GRIDC     = '#1a2030'   # grid
    LEG_BG    = '#111520'
else:
    # ── Publication (white) palette ───────────────────────────────────────
    BG        = 'white'
    BG_AX     = 'white'
    T_PRI     = '#111111'
    T_SEC     = '#222222'
    T_MUT     = '#555555'
    ACCENT    = '#1f5fa8'   # cascade (blue)
    RED_MUT   = '#b03a3a'   # CPL (red)
    GRAY_MUT  = '#666666'   # LCDM (gray)
    EDGE      = '#999999'
    REFLINE   = '#888888'
    GRIDC     = '#dddddd'
    LEG_BG    = 'white'

def apply_dark_style():
    """Apply the selected theme (name kept for historical reasons)."""
    rcParams.update({
        'figure.facecolor':  BG,
        'axes.facecolor':    BG_AX,
        'axes.edgecolor':    EDGE,
        'axes.labelcolor':   T_SEC,
        'axes.titlecolor':   T_PRI,
        'axes.grid':         True,
        'grid.color':        GRIDC,
        'grid.linewidth':    0.6,
        'xtick.color':       T_MUT,
        'ytick.color':       T_MUT,
        'xtick.labelcolor':  T_MUT,
        'ytick.labelcolor':  T_MUT,
        'legend.facecolor':  LEG_BG,
        'legend.edgecolor':  EDGE,
        'legend.labelcolor': T_SEC,
        'text.color':        T_SEC,
        'font.family':       'sans-serif',
        'font.size':         11,
        'axes.labelsize':    12,
        'axes.titlesize':    11.5,
        'legend.fontsize':   10,
        'figure.dpi':        180,
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })

# ---------------------------------------------------------------------------
# Figure 1 — w(z) profile
# ---------------------------------------------------------------------------
W0_CPL = -0.359
WA_CPL = -2.50

apply_dark_style()
z = np.linspace(0, 2.5, 600)
w_lcdm    = np.full_like(z, -1.0)
w_cascade = np.full_like(z, -0.999)
w_cpl     = W0_CPL + WA_CPL * z / (1.0 + z)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
fig.patch.set_facecolor(BG)

ax.axhspan(-3.0, -1.0, color=RED_MUT, alpha=0.07, zorder=0)
ax.axhline(-1.0, color=REFLINE, lw=0.9, ls='--', zorder=1)
ax.text(2.35, -1.035, '$w = -1$', ha='right', va='top', fontsize=9,
        color=T_MUT)
ax.text(1.55, -2.25, 'Phantom region\n(NEC violated)', ha='center',
        fontsize=8.5, color=RED_MUT, alpha=0.7)

ax.plot(z, w_lcdm,    color=GRAY_MUT, lw=1.5, ls='--', zorder=3,
        label=r'$\Lambda$CDM ($w = -1$)')
ax.plot(z, w_cascade, color=ACCENT,   lw=2.4, zorder=4,
        label=r'Cascade ($w_0 = -0.999$, no-phantom theorem)')
ax.plot(z, w_cpl,     color=RED_MUT,  lw=1.8, ls='-.', zorder=3,
        label=r'CPL ($w_0 = -0.36,\; w_a = -2.50$)')

z_cross = 0.345
ax.axvline(z_cross, color=RED_MUT, lw=0.7, ls=':', alpha=0.55)
ax.annotate('Phantom crossing\n$z = 0.345$',
            xy=(z_cross, -1.0), xytext=(0.72, -0.62),
            fontsize=8.5, color=RED_MUT,
            arrowprops=dict(arrowstyle='->', color=RED_MUT, lw=0.8))

ax.set_xlabel('Redshift $z$', color=T_SEC)
ax.set_ylabel('Equation of state $w(z)$', color=T_SEC)
ax.set_xlim(0, 2.5)
ax.set_ylim(-2.8, -0.15)
ax.legend(loc='lower left', framealpha=0.9)
ax.set_title('Dark energy $w(z)$: cascade no-phantom theorem vs CPL', pad=8)

# Zoomed inset
axins = ax.inset_axes([0.52, 0.55, 0.44, 0.38])
axins.set_facecolor(BG_AX)
for sp in axins.spines.values():
    sp.set_edgecolor(EDGE)
z_ins = np.linspace(0, 2.5, 600)
axins.plot(z_ins, np.full_like(z_ins, -1.0),   color=GRAY_MUT, lw=1.4, ls='--')
axins.plot(z_ins, np.full_like(z_ins, -0.999), color=ACCENT,   lw=2.0)
axins.fill_between(z_ins, -1.0, -0.999, color=ACCENT, alpha=0.22)
axins.set_xlim(0, 2.5)
axins.set_ylim(-1.0025, -0.9965)
axins.set_yticks([-1.000, -0.999])
axins.set_yticklabels(['$-1.000$', '$-0.999$'], fontsize=7.5, color=T_MUT)
axins.set_xticks([])
axins.set_title('No-phantom gap\n(cascade $>$ $-1$)', fontsize=7.5, pad=3,
                color=T_SEC)
axins.tick_params(labelsize=7.5, colors=T_MUT)
axins.grid(color=GRIDC, linewidth=0.5)
ax.indicate_inset_zoom(axins, edgecolor=REFLINE, lw=0.7)

fig.tight_layout()
fig.savefig('figures/fig1_wz_profile.pdf', bbox_inches='tight', facecolor=BG)
fig.savefig('figures/fig1_wz_profile.png', bbox_inches='tight', facecolor=BG)
plt.close()
print("Figure 1 saved.")

# ---------------------------------------------------------------------------
# Figure 2 — chi^2 comparison
# ---------------------------------------------------------------------------
apply_dark_style()

tests  = ['DESI Y1\nBlind retrodiction\n(3 withheld points)',
          'DESI DR2\nCross-release\n(6 withheld points)']
models = [r'$\Lambda$CDM', 'Cascade', 'CPL']
colors = [GRAY_MUT, ACCENT, RED_MUT]
hatches = ['', '//', 'xx']

chi2 = np.array([
    [5.54,  5.50, 22.81],
    [4.39,  4.45,  2.17],
])

x = np.arange(len(tests))
width = 0.22
offset = [-1, 0, 1]

fig, ax = plt.subplots(figsize=(7, 4.5))
fig.patch.set_facecolor(BG)

for i, (model, color, hatch) in enumerate(zip(models, colors, hatches)):
    vals = chi2[:, i]
    bars = ax.bar(x + offset[i] * width, vals, width,
                  label=model, color=color, alpha=0.75,
                  hatch=hatch, edgecolor=EDGE, linewidth=0.6, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.35,
                f'{v:.2f}', ha='center', va='bottom',
                fontsize=8.5, color=color)

ax.axhline(5.0, color=REFLINE, lw=0.8, ls=':', zorder=2)
ax.text(1.44, 5.4, 'Reference $\\chi^2 = 5$', ha='right', fontsize=8,
        color=T_MUT)

ax.annotate('CPL phantom excursion\namplifies withheld-data error\n(4$\\times$ worse than cascade)',
            xy=(0 + 1 * width, 22.81), xytext=(0.52, 20.0),
            fontsize=8, color=RED_MUT,
            arrowprops=dict(arrowstyle='->', color=RED_MUT, lw=0.8))
ax.annotate('Lower $\\chi^2$ via NEC-violating\nphantom crossing ($z=0.345$)',
            xy=(1 + 1 * width, 2.17), xytext=(0.92, 9.0),
            fontsize=8, color=RED_MUT,
            arrowprops=dict(arrowstyle='->', color=RED_MUT, lw=0.8))

ax.set_xticks(x)
ax.set_xticklabels(tests, fontsize=10)
ax.set_ylabel(r'$\chi^2$ on withheld data (lower is better)')
ax.set_ylim(0, 27)
ax.legend(framealpha=0.9, loc='upper right')
ax.set_title('Model performance on withheld data', pad=8)

fig.tight_layout()
fig.savefig('figures/fig2_chi2_comparison.pdf', bbox_inches='tight', facecolor=BG)
fig.savefig('figures/fig2_chi2_comparison.png', bbox_inches='tight', facecolor=BG)
plt.close()
print("Figure 2 saved.")

# ---------------------------------------------------------------------------
# Figure 3 — Sealed DR3 predictions
# ---------------------------------------------------------------------------
apply_dark_style()

z_vals    = np.array([0.510, 0.706, 0.934, 1.321, 1.484, 2.330])
cas_pred  = np.array([22.127, 19.790, 17.370, 14.028, 12.880,  8.681])
lcdm_pred = np.array([22.087, 19.768, 17.360, 14.029, 12.882,  8.687])
cpl_pred  = np.array([21.746, 19.730, 17.535, 14.228, 13.037,  8.634])
dr3_sigma = np.array([ 0.232,  0.180,  0.105,  0.121,  0.282,  0.080])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                gridspec_kw={'width_ratios': [2, 1]})
fig.patch.set_facecolor(BG)

jitter = 0.018
for z_i, cas, lcdm, cpl, sig in zip(z_vals, cas_pred, lcdm_pred, cpl_pred, dr3_sigma):
    ax1.errorbar(z_i - jitter, cas,  yerr=sig, fmt='o', color=ACCENT,   ms=6,
                 capsize=3.5, lw=1.5, zorder=5)
    ax1.errorbar(z_i,          lcdm, yerr=sig, fmt='s', color=GRAY_MUT, ms=5,
                 capsize=3.5, lw=1.2, alpha=0.8, zorder=4)
    ax1.errorbar(z_i + jitter, cpl,  yerr=sig, fmt='^', color=RED_MUT,  ms=5,
                 capsize=3.5, lw=1.2, alpha=0.8, zorder=4)

ax1.plot(z_vals - jitter, cas_pred,  'o-',  color=ACCENT,   lw=1.5,
         label='Cascade (sealed)')
ax1.plot(z_vals,          lcdm_pred, 's--', color=GRAY_MUT, lw=1.2, alpha=0.8,
         label=r'$\Lambda$CDM')
ax1.plot(z_vals + jitter, cpl_pred,  '^-.', color=RED_MUT,  lw=1.2, alpha=0.8,
         label='CPL')

ax1.set_xlabel('Redshift $z$')
ax1.set_ylabel('$D_H / r_d$')
ax1.legend(framealpha=0.9)
ax1.set_title('Sealed DR3 predictions\n(projected $\\pm1\\sigma$ error bars)', pad=6)

# Residual panel
ax2.axhline(0, color=REFLINE, lw=0.8, ls='--')
ax2.bar(np.arange(len(z_vals)) - 0.2,
        cas_pred - lcdm_pred, 0.35,
        color=ACCENT,  alpha=0.75, label=r'Cascade $-$ $\Lambda$CDM', zorder=3)
ax2.bar(np.arange(len(z_vals)) + 0.2,
        cpl_pred - lcdm_pred, 0.35,
        color=RED_MUT, alpha=0.7, label=r'CPL $-$ $\Lambda$CDM', zorder=3)

ax2.set_xticks(np.arange(len(z_vals)))
ax2.set_xticklabels([f'$z={z:.3f}$' for z in z_vals],
                    rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Residual $D_H/r_d$')
ax2.set_title('Residuals from $\\Lambda$CDM', pad=6)
ax2.legend(fontsize=8.5, framealpha=0.9)

fig.suptitle('DESI DR3 Sealed Predictions — Sealed 21 June 2026',
             fontsize=11, y=1.01, color=T_PRI)
fig.tight_layout()
fig.savefig('figures/fig3_dr3_predictions.pdf', bbox_inches='tight', facecolor=BG)
fig.savefig('figures/fig3_dr3_predictions.png', bbox_inches='tight', facecolor=BG)
plt.close()
print("Figure 3 saved.")

print(f"\nAll {THEME}-themed figures generated in figures/")
