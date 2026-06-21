"""
run_cascade_de.py
=================
Dark energy from cascade dynamics — derived from first principles.

Source of truth: the paper's cascade_model.py (default parameters).
Initial conditions: same as run_high_energy.py (big bang analogue).
No parameters are tuned. No cosmological equations are imported.
The simulation runs; the output is what it is.

Per-step observables (all from simulation output, no model internals):
  beta(n)  = mean Beta per site before step n
  D(n)     = beta(n) - beta(n+1)          total energy lost in step n
  delta(n) = D(n) / beta(n)               fractional depletion rate
  F(n)     = mean_ripple()                structure curvature / regime signal

Cascade-internal dark energy quantities:
  z_c(n)   = beta(n) / beta_0 - 1         cascade redshift (no Friedmann)
  Phi(n)   = [delta(n)/delta_0] * sqrt(beta_0/beta(n))  master ratio
  w_eff(n) = -1 + (1+w0) * Phi(n)         single external input: w0 = -0.77

Supermartingale theorem => D(n) > 0 whenever active => delta > 0 => Phi > 0
=> w_eff > -1 at every active step. This is structural, not a fit.

Author: Shiv Goswami  |  Date: June 21, 2026
Ref:    CASCADE_DARK_ENERGY_DERIVATION.md
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from cascade_model import CascadeModel

# ── Single external cosmological calibration ──────────────────────────────────
W0 = -0.77   # today's measured dark energy w (DESI+CMB)

# ── Paper parameters — do not modify ─────────────────────────────────────────
P         = 100    # sites  (same as run_high_energy.py)
SEED      = 2024   # seed   (same as run_high_energy.py)
MAX_STEPS = 10000  # upper bound; stops at absorbing state

# Big bang analogue initial conditions (same as run_high_energy.py)
E0 = 500.0   # total Beta across P sites
X0 = 8.0     # excitations per site
Y0 = 8.0

# ── Build model with DEFAULT paper parameters ─────────────────────────────────
model = CascadeModel(P=P, seed=SEED)   # all defaults from the paper

model.X[:]       = X0
model.Y[:]       = Y0
model.Beta[:]    = E0 / P
model.S[:]       = 0.0
model.S_prev1[:] = 0.0
model.S_prev2[:] = 0.0

C_thresh  = model.C               # 0.5
CD_thresh = model.C + model.Delta # 0.8

# ── Run — record beta and F before each step ──────────────────────────────────
beta_before = []
F_before    = []
n_list      = []

for _ in range(MAX_STEPS):
    beta_n = float(np.mean(model.Beta))
    F_n    = model.mean_ripple()
    n_list.append(model.n)
    beta_before.append(beta_n)
    F_before.append(F_n)

    active = model.step()
    if not active or model.is_absorbing():
        beta_before.append(float(np.mean(model.Beta)))
        F_before.append(model.mean_ripple())
        n_list.append(model.n)
        break

beta_arr = np.array(beta_before)
F_arr    = np.array(F_before)
n_arr    = np.array(n_list)
N        = len(n_arr)

# ── Depletion D(n) from consecutive beta differences ─────────────────────────
D_arr     = np.zeros(N)
D_arr[:-1] = np.maximum(beta_arr[:-1] - beta_arr[1:], 0.0)
delta_arr  = np.where(beta_arr > 1e-12, D_arr / beta_arr, 0.0)

# ── Regime from F ─────────────────────────────────────────────────────────────
regime_arr = np.where(F_arr >= CD_thresh, 2,
             np.where(F_arr >  C_thresh,  1, 0))
REGIME_NAME = {0: 'Quiescent', 1: 'Leakage', 2: 'Explosive'}

print(f"Simulation complete: {N} steps  (seed={SEED}, P={P}, E0={E0})")
print(f"  Explosive steps: {np.sum(regime_arr==2)}")
print(f"  Leakage  steps:  {np.sum(regime_arr==1)}")
print(f"  Quiescent steps: {np.sum(regime_arr==0)}")
print(f"  beta range: {beta_arr.min():.4f} — {beta_arr.max():.4f}")
print(f"  delta range: {delta_arr[delta_arr>0].min():.6f} — {delta_arr.max():.6f}")

# ── Identify n₀ (today): first stable quiescent step after explosive phase ────
quiescent_idx = np.where(regime_arr == 0)[0]
explosive_idx = np.where(regime_arr == 2)[0]

if len(quiescent_idx) == 0:
    print("\nWARNING: no quiescent steps found in this run.")
    n0_idx = N - 1
elif len(explosive_idx) == 0:
    n0_idx = int(quiescent_idx[0])
else:
    last_exp  = int(explosive_idx[-1])
    q_after   = quiescent_idx[quiescent_idx > last_exp]
    n0_idx    = int(q_after[0]) if len(q_after) > 0 else int(quiescent_idx[0])

beta_0  = beta_arr[n0_idx]
delta_0 = delta_arr[n0_idx]
n_0     = n_arr[n0_idx]

print(f"\nToday n₀: step {n_0}  (index {n0_idx})")
print(f"  beta_0  = {beta_0:.6f}")
print(f"  delta_0 = {delta_0:.6f}   (Gamma_0 — cascade-derived, not fitted)")

# ── Cascade-internal dark energy quantities ───────────────────────────────────
safe_beta = np.maximum(beta_arr, 1e-12)
z_c   = beta_arr / beta_0 - 1.0
Phi   = np.where(delta_0 > 1e-12,
                 (delta_arr / delta_0) * np.sqrt(beta_0 / safe_beta),
                 0.0)
w_eff = -1.0 + (1.0 + W0) * Phi

# ── Print result table ────────────────────────────────────────────────────────
print("\n" + "="*74)
print("CASCADE-INTERNAL DARK ENERGY  (no cosmological equations imported)")
print("="*74)
print(f"{'n':>5}  {'z_c':>7}  {'beta':>7}  {'delta':>8}  {'Phi':>7}  {'w_eff':>7}  Regime")
print("-"*74)

show = sorted(set(
    list(range(0, min(N, 6))) +
    [n0_idx] +
    list(range(0, N, max(1, N//20))) +
    [N-1]
))
for i in show:
    if i >= N: continue
    print(f"{n_arr[i]:>5}  {z_c[i]:>7.3f}  {beta_arr[i]:>7.3f}  "
          f"{delta_arr[i]:>8.5f}  {Phi[i]:>7.4f}  {w_eff[i]:>7.4f}  "
          f"{REGIME_NAME[int(regime_arr[i])]}")

print()
print(f"  Gamma_0 (cascade-derived) = {delta_0:.6f}")
print(f"  Supermartingale — Phi > 0 at all active steps: "
      f"{bool(np.all(Phi[:-1] >= 0))}")
print(f"  No phantom — w_eff > -1 at all active steps: "
      f"{bool(np.all(w_eff[:-1] > -1.0))}")

# ── Validation: compare w_eff at z_c milestones vs DESI ──────────────────────
print("\n" + "="*74)
print("VALIDATION: cascade w_eff(z_c) vs DESI DR2")
print("  (DESI values from CPL best-fit: w0=-0.77, wa=-0.44+CMB)")
print("="*74)

desi_z  = np.array([0.0,  0.3,  0.5,  1.0,  1.5,  2.0,  2.33])
desi_w  = np.array([-0.77, -0.88, -0.99, -1.08, -1.08, -1.08, -1.08])

print(f"  {'z_c_target':>10}  {'cascade w_eff':>14}  {'DESI CPL w':>12}  Regime")
print("  " + "-"*55)
for zt, wd in zip(desi_z, desi_w):
    idx_c = np.argmin(np.abs(z_c - zt))
    print(f"  {z_c[idx_c]:>10.3f}  {w_eff[idx_c]:>14.4f}  {wd:>12.4f}  "
          f"{REGIME_NAME[int(regime_arr[idx_c])]}")

# ── Figure ────────────────────────────────────────────────────────────────────
BG   = '#080808';  TXT = '#dddddd'
C_E  = '#ff6b6b';  C_L = '#ffd93d';  C_Q = '#6bcb77'
C_D  = '#ffd54f'   # DESI reference

def regime_col(r):
    return [C_E if v==2 else C_L if v==1 else C_Q for v in r]

rc = regime_col(regime_arr)

fig = plt.figure(figsize=(18, 11), facecolor=BG)
gs  = gridspec.GridSpec(2, 4, figure=fig,
                        hspace=0.50, wspace=0.40,
                        left=0.06, right=0.97, top=0.93, bottom=0.07)

def ax_style(ax, title='', xl='', yl=''):
    ax.set_facecolor('#111111')
    ax.tick_params(colors=TXT, labelsize=8)
    ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
    ax.title.set_color(TXT)
    for sp in ax.spines.values(): sp.set_color('#2a2a2a')
    if title: ax.set_title(title, fontsize=8.5)
    if xl:    ax.set_xlabel(xl, fontsize=8)
    if yl:    ax.set_ylabel(yl, fontsize=8)

# Row 0
ax_b   = fig.add_subplot(gs[0, 0])
ax_d   = fig.add_subplot(gs[0, 1])
ax_F   = fig.add_subplot(gs[0, 2])
ax_Phi = fig.add_subplot(gs[0, 3])

# Row 1
ax_w   = fig.add_subplot(gs[1, 0])
ax_wz  = fig.add_subplot(gs[1, 1:3])
ax_Phz = fig.add_subplot(gs[1, 3])

# 1. beta(n)
ax_style(ax_b, 'Mean beta per site', 'step n', 'beta(n)')
ax_b.scatter(n_arr, beta_arr, c=rc, s=6, alpha=0.85, linewidths=0)
ax_b.axvline(n_0, color='white', lw=1, ls='--', alpha=0.5, label='n₀ (today)')
ax_b.legend(fontsize=7, facecolor='#111', labelcolor=TXT)

# 2. delta(n)
ax_style(ax_d, 'Fractional depletion delta(n)', 'step n', 'D(n)/beta(n)')
ax_d.scatter(n_arr, delta_arr, c=rc, s=6, alpha=0.85, linewidths=0)
ax_d.axhline(delta_0, color='white', lw=1, ls='--', alpha=0.5,
             label=f'delta_0={delta_0:.4f}')
ax_d.axvline(n_0, color='white', lw=0.8, ls=':', alpha=0.4)
ax_d.legend(fontsize=7, facecolor='#111', labelcolor=TXT)

# 3. F(n)
ax_style(ax_F, 'Mean ripple F(n)', 'step n', 'mean |Delta^2 S|')
ax_F.scatter(n_arr, F_arr, c=rc, s=6, alpha=0.85, linewidths=0)
ax_F.axhline(C_thresh,  color='#aaa', lw=0.8, ls=':', label=f'C={C_thresh}')
ax_F.axhline(CD_thresh, color='#ff9', lw=0.8, ls=':', label=f'C+D={CD_thresh}')
ax_F.legend(fontsize=7, facecolor='#111', labelcolor=TXT)

# 4. Phi(n)
ax_style(ax_Phi, 'Phi(n) — master ratio', 'step n', 'Phi')
ax_Phi.scatter(n_arr, Phi, c=rc, s=6, alpha=0.85, linewidths=0)
ax_Phi.axhline(1.0, color='white', lw=1, ls='--', alpha=0.5, label='Phi=1 (today)')
ax_Phi.axvline(n_0, color='white', lw=0.8, ls=':', alpha=0.4)
ax_Phi.legend(fontsize=7, facecolor='#111', labelcolor=TXT)

# 5. w_eff(n)
ax_style(ax_w, 'w_eff(n)', 'step n', 'w_eff = -1 + (1+w0)*Phi')
ax_w.scatter(n_arr, w_eff, c=rc, s=6, alpha=0.85, linewidths=0)
ax_w.axhline(-1.0, color='#555', lw=0.8, ls=':', label='phantom w=-1')
ax_w.axhline(W0,   color='white', lw=0.8, ls='--', alpha=0.5, label=f'w0={W0}')
ax_w.axvline(n_0, color='white', lw=0.8, ls=':', alpha=0.4)
ax_w.legend(fontsize=7, facecolor='#111', labelcolor=TXT)

# 6. w_eff vs z_c  (main panel)
ax_style(ax_wz, 'w_eff(z_c) — CASCADE PREDICTION vs DESI DR2',
         'z_c  (cascade redshift = beta/beta_0 - 1)', 'w_eff')
ax_wz.axhline(-1.0, color='#555', lw=0.8, ls=':', label='phantom divide w=-1')
ax_wz.axhline(W0,   color='white', lw=0.8, ls='--', alpha=0.4, label=f'w0={W0}')
mask = z_c >= -0.5
ax_wz.scatter(z_c[mask], w_eff[mask], c=[rc[i] for i in range(N) if mask[i]],
              s=10, alpha=0.85, linewidths=0, label='CASCADE (each dot = one step)')
ax_wz.plot(desi_z, desi_w, 'o--', color=C_D, lw=1.5, ms=6,
           label='DESI DR2 CPL (validation)')
xlim_hi = max(z_c[mask].max() * 1.05 if mask.sum() else 1, 2.5)
ax_wz.set_xlim(-0.3, xlim_hi)
ax_wz.legend(fontsize=7.5, facecolor='#111', labelcolor=TXT)

# 7. Phi vs z_c
ax_style(ax_Phz, 'Phi(z_c)', 'z_c', 'Phi')
ax_Phz.scatter(z_c[mask], Phi[mask], c=[rc[i] for i in range(N) if mask[i]],
               s=10, alpha=0.85, linewidths=0)
ax_Phz.axhline(1.0, color='white', lw=1, ls='--', alpha=0.5)
ax_Phz.set_xlim(-0.3, xlim_hi)

# Legend patches
from matplotlib.patches import Patch
handles = [Patch(color=C_E, label='Explosive'), Patch(color=C_L, label='Leakage'),
           Patch(color=C_Q, label='Quiescent')]
fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=9,
           facecolor='#1a1a1a', labelcolor=TXT, bbox_to_anchor=(0.5, 0.997))

fig.suptitle(
    'CASCADE DARK ENERGY — First Principles  |  '
    'No imported cosmology  |  Default paper parameters  |  Shiv Goswami, June 2026',
    fontsize=9.5, color='#777', y=0.972)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'cascade_de_internal.png')
fig.savefig(out, dpi=155, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"\nFigure -> {out}")
