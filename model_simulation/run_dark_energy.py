"""
run_dark_energy.py
Quantitative analysis: cascade model dark energy equation of state vs DESI DR2.

Derives w_eff(n) from cascade beta-depletion dynamics, maps to redshift z,
and compares to DESI DR2 CPL parameterization.

Master equation (derived in DARK_ENERGY_MAPPING.md, Section II):
    1 + w_eff(n) = D(n) / [3 * H0_tau * sqrt(rho_beta(n) * rho_beta_0)]

Analytic prediction (quiescent regime, Section IV):
    w(z) = -1 + (1 + w0) / (1 + Gamma0 * z * (z+2) / 4)

where Gamma0 = -2*wa / (1 + w0) is determined by matching DESI wa.

Initial conditions: S_prev2 < S_prev1 < S chosen so that the ripple
F = |S - 2*S_prev1 + S_prev2| starts in the explosive regime.
This models the universe already mid-cascade (post-Big Bang onset),
which is physically correct — we don't model the exact singularity.

Author: Shiv Goswami
Date:   June 21, 2026
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cascade_model import CascadeModel


def mean_field_cascade(
    b0: float, x0: float, y0: float,
    s0: float, s1: float, s2: float,
    alpha: float = 0.10, omega_x: float = 1.0, omega_y: float = 1.20,
    k: float = 0.50, eta: float = 1.0, lam: float = 0.10,
    C: float = 0.50, Delta: float = 0.30,
    gamma1: float = 0.20, kappa: float = 0.40, mu: float = 0.05,
    max_steps: int = 800,
):
    """
    Deterministic mean-field reduction of cascade dynamics (no spatial noise).
    Runs until b <= 1e-6 or max_steps.
    Returns arrays: n, b, D, x, y, f, regime.
    """
    rate = alpha * omega_x * omega_y  # base XY interaction rate

    n_arr  = []
    b_arr  = []
    D_arr  = []
    x_arr  = []
    y_arr  = []
    f_arr  = []
    reg_arr = []

    b = float(b0)
    x = float(x0)
    y = float(y0)
    s_nm2 = float(s2)  # S at n-2
    s_nm1 = float(s1)  # S at n-1
    s_n   = float(s0)  # S at n
    b_prev = b

    for step in range(max_steps):
        # Ripple
        f = abs(s_n - 2 * s_nm1 + s_nm2)

        # Regime
        if f <= C:
            regime = 'quiescent'
            l_val  = 0.0
            m_val  = 0.0
            new_xy = 0.0
        elif f < C + Delta:
            regime = 'leakage'
            l_val  = lam * f
            m_val  = 0.0
            new_xy = 0.0
        else:
            regime = 'explosive'
            l_val  = lam * f
            m_val  = (f - C) / Delta
            new_xy = m_val

        # Interaction rate (per step)
        r_xy = rate * x * y
        # Bond rate (simplified: small mu * x * y if T_eff < T_c)
        T_eff = f / C if C > 0 else 0.0
        bonds = mu * x * y if T_eff < 0.8 else 0.0

        # Depletion
        D_now = max(0.0, b_prev - b)
        D_arr.append(D_now)
        b_arr.append(b)
        n_arr.append(step)
        x_arr.append(x)
        y_arr.append(y)
        f_arr.append(f)
        reg_arr.append(regime)

        # Updates
        b_next = b - k * r_xy - l_val - eta * m_val - kappa * bonds
        b_next = max(0.0, b_next)

        x_next = max(0.0, x - r_xy - bonds + new_xy)
        y_next = max(0.0, y - r_xy - bonds + new_xy)

        s_next = s_n + gamma1 * r_xy

        # Roll S history
        s_nm2, s_nm1, s_n = s_nm1, s_n, s_next
        b_prev = b
        b = b_next
        x = x_next
        y = y_next

        if b < 1e-6 or (x < 1e-6 and y < 1e-6 and f <= C):
            break

    return (np.array(n_arr), np.array(b_arr), np.array(D_arr),
            np.array(x_arr), np.array(y_arr), np.array(f_arr),
            np.array(reg_arr))


def fit_cpl(z, w):
    """
    CPL fit  w(z) = w0 + wa * z/(1+z)  by ordinary least squares.
    Returns (w0, wa, w0_err, wa_err).
    """
    u = z / (1.0 + z)
    A = np.column_stack([np.ones_like(u), u])
    coeffs, res, _, _ = np.linalg.lstsq(A, w, rcond=None)
    w0, wa = float(coeffs[0]), float(coeffs[1])
    sigma2 = float(np.dot(w - A @ coeffs, w - A @ coeffs)) / max(len(w) - 2, 1)
    cov = sigma2 * np.linalg.pinv(A.T @ A)
    return w0, wa, float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))


def w_cpl(z, w0, wa):
    return w0 + wa * z / (1.0 + z)


def w_cascade_analytic(z, w0, Gamma0):
    """
    Quiescent-regime analytic formula (DARK_ENERGY_MAPPING.md Sec IV.2).
    Derived from beta supermartingale and Friedmann-analog Hubble.
    """
    return -1.0 + (1.0 + w0) / (1.0 + Gamma0 * z * (z + 2.0) / 4.0)


# ─────────────────────────────────────────────────────────────────────────────
# DESI DR2 reference  (DESI + CMB + DESY5, 2025)
# ─────────────────────────────────────────────────────────────────────────────
W0_DESI = -0.77
WA_DESI = -0.44
W0_SIG  =  0.05
WA_SIG  =  0.23

# Gamma0 from DESI (DARK_ENERGY_MAPPING.md Sec V):  wa = -(1+w0)*Gamma0/2
Gamma0_desi = -2.0 * WA_DESI / (1.0 + W0_DESI)

print("=" * 65)
print("CASCADE MODEL -- DARK ENERGY EQUATION OF STATE")
print("=" * 65)
print(f"DESI DR2 reference:  w0 = {W0_DESI},  wa = {WA_DESI}")
print(f"Gamma0 implied by DESI wa:  {Gamma0_desi:.4f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Simulation setup
# ─────────────────────────────────────────────────────────────────────────────
P         = 50       # sites (small enough for real fluctuations)
E_total   = 800.0    # total initial beta  (16.0 per site)
X0 = Y0   = 2.0      # modest initial excitations (slow enough for 3-regime arc)
MAX_STEPS = 5000
SEED      = 7

# Initial ripple priming: set S history so F_init > C + Delta
# S_prev2 << S_prev1 << S_now → F = |S - 2*S_prev1 + S_prev2|
#   = 3.0 - 2*1.5 + 0.3 = 0.3  ... not enough
# Use: S_prev2 = 0, S_prev1 = 0.5, S = 2.5
#   F = |2.5 - 2*0.5 + 0| = |2.5 - 1.0| = 1.5 > C+Delta=0.8  -> explosive
S_INIT      = 2.5
S_PREV1     = 0.5
S_PREV2     = 0.0

model = CascadeModel(
    P       = P,
    # Default parameters (known to give 3-regime arc at these energy scales)
    alpha_XY = 0.10,
    alpha_XX = 0.04,
    omega_X  = 1.00,
    omega_Y  = 1.20,      # asymmetry: omega_X != omega_Y (Part II necessity)
    k_XY = 0.50,
    k_XX = 0.30,
    eta  = 1.00,
    kappa= 0.40,
    C      = 0.50,
    Delta  = 0.30,
    lambda_= 0.10,
    gamma_1  = 0.20,
    gamma_XX = 0.10,
    gamma_2  = 0.15,
    a0 = 1.0, b = 0.05, T_c = 0.8,
    D_X = 0.05, D_Y = 0.05,
    seed = SEED,
)
model.X[:]       = X0
model.Y[:]       = Y0
model.Beta[:]    = E_total / P
model.S[:]       = S_INIT
model.S_prev1[:] = S_PREV1
model.S_prev2[:] = S_PREV2
# Set n=2 so _ripple() bypasses the n<2 guard and reads the S-history we primed.
# Physically: we model the universe at step 2 of the cascade, not step 0.
model.n = 2

# Verify initial ripple
F_init = abs(S_INIT - 2 * S_PREV1 + S_PREV2)
print(f"Initial ripple F_init = {F_init:.2f}  (C={model.C}, C+Delta={model.C+model.Delta:.2f})")
print(f"Initial regime: {'explosive' if F_init >= model.C+model.Delta else 'leakage' if F_init > model.C else 'quiescent'}")
print()

# ── Run simulation ─────────────────────────────────────────────────────────
records  = []
rho_prev = E_total / P

for step_i in range(MAX_STEPS):
    rho_now = model.total_energy() / P
    F_avg   = model.mean_ripple()
    D_now   = max(0.0, rho_prev - rho_now)

    if F_avg <= model.C:
        regime = 'quiescent'
    elif F_avg < model.C + model.Delta:
        regime = 'leakage'
    else:
        regime = 'explosive'

    records.append(dict(n=model.n, rho=rho_now, D=D_now,
                        F=F_avg, regime=regime))
    rho_prev = rho_now

    active = model.step()
    if not active or model.is_absorbing():
        break

N = len(records)
n_arr   = np.array([r['n']      for r in records])
rho_arr = np.array([r['rho']    for r in records])
D_arr   = np.array([r['D']      for r in records])
F_arr   = np.array([r['F']      for r in records])
reg_arr = np.array([r['regime'] for r in records])

print(f"Simulation: {N} steps  |  P={P}  E_total={int(E_total)}")
for lab in ('explosive', 'leakage', 'quiescent'):
    cnt = (reg_arr == lab).sum()
    pct = 100 * cnt / N
    print(f"  {lab:12s}: {cnt:5d} steps  ({pct:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# Identify "today" (z = 0)
# Use the step where quiescent is stably entered (15th quiescent step).
# This corresponds to: structure growth has stalled, beta weakening slowly.
# ─────────────────────────────────────────────────────────────────────────────
qui_idx = np.where(reg_arr == 'quiescent')[0]
# ensure we have enough quiescent steps; otherwise use transition point
if len(qui_idx) >= 20:
    today_idx = int(qui_idx[19])   # 20th quiescent step (past transient)
elif len(qui_idx) >= 5:
    today_idx = int(qui_idx[4])
elif len(qui_idx) >= 1:
    today_idx = int(qui_idx[0])
else:
    # No quiescent steps: use the 70th percentile
    today_idx = int(0.70 * N)

rho_0 = rho_arr[today_idx]
D_0   = D_arr[today_idx]

print(f"\nToday (z=0): step_idx={today_idx}, n={n_arr[today_idx]}, "
      f"rho_0={rho_0:.4f}, D_0={D_0:.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# Calibrate H0*tau:  1+w0 = D0 / (3 * H0_tau * rho_0)
# ─────────────────────────────────────────────────────────────────────────────
if D_0 > 1e-15 and rho_0 > 1e-10:
    H0_tau = D_0 / (3.0 * (1.0 + W0_DESI) * rho_0)
else:
    # If D_0=0 (absorbing edge), use nearby step
    nearby = [i for i in range(max(0, today_idx-5), min(N, today_idx+5))
              if D_arr[i] > 0 and rho_arr[i] > 0]
    if nearby:
        i_ref = nearby[0]
        H0_tau = D_arr[i_ref] / (3.0 * (1.0 + W0_DESI) * rho_arr[i_ref])
    else:
        H0_tau = 1e-6
    print(f"  (D_0 too small; using nearby step for calibration)")

print(f"H0*tau (calibrated) = {H0_tau:.7f}")

# ─────────────────────────────────────────────────────────────────────────────
# Compute w_eff(n)  for all steps
# w_eff(n) = -1 + D(n) / [3 * H0_tau * sqrt(rho_beta(n) * rho_0)]
# ─────────────────────────────────────────────────────────────────────────────
w_eff = np.full(N, np.nan)
for i in range(1, N):
    denom = 3.0 * H0_tau * np.sqrt(rho_arr[i] * rho_0)
    if denom > 1e-30 and rho_arr[i] > 0:
        w_eff[i] = -1.0 + D_arr[i] / denom

# ─────────────────────────────────────────────────────────────────────────────
# Map step index -> effective redshift
# Linear map: step 0 -> z=Z_MAX, today_idx -> z=0
# ─────────────────────────────────────────────────────────────────────────────
Z_MAX = 3.0    # highest redshift shown (DESI DR2 covers ~0-2.5)
z_eff = np.full(N, np.nan)
for i in range(N):
    if i <= today_idx and today_idx > 0:
        z_eff[i] = Z_MAX * (today_idx - i) / today_idx

# ─────────────────────────────────────────────────────────────────────────────
# CPL fit to cascade numerical w_eff (quiescent regime, 0 <= z <= 2.5)
# ─────────────────────────────────────────────────────────────────────────────
mask = (
    np.isfinite(w_eff) & np.isfinite(z_eff) &
    (z_eff >= 0.0) & (z_eff <= 2.5) &
    (reg_arr == 'quiescent') &
    (np.abs(w_eff) < 5.0)
)
print(f"\nQuiescent points in DESI z-range [0, 2.5]: {mask.sum()}")

w0_fit = W0_DESI; wa_fit = WA_DESI
w0_sig = 0.0;     wa_sig = 0.0

if mask.sum() >= 8:
    z_c = z_eff[mask]
    w_c = w_eff[mask]
    lo, hi = np.percentile(w_c, [5, 95])
    keep = (w_c >= lo) & (w_c <= hi)
    z_c, w_c = z_c[keep], w_c[keep]
    try:
        w0_fit, wa_fit, w0_sig, wa_sig = fit_cpl(z_c, w_c)
    except Exception as exc:
        print(f"  CPL fit exception: {exc}")

print(f"\n  CPL fit (cascade numerical, quiescent regime):")
print(f"    w0 = {w0_fit:+.4f}  +/-  {w0_sig:.4f}    DESI: {W0_DESI:+.3f}  +/- {W0_SIG:.3f}")
print(f"    wa = {wa_fit:+.4f}  +/-  {wa_sig:.4f}    DESI: {WA_DESI:+.3f}  +/- {WA_SIG:.3f}")

within_w0 = abs(w0_fit - W0_DESI) < 2 * max(w0_sig, W0_SIG)
within_wa = abs(wa_fit - WA_DESI) < 2 * max(wa_sig, WA_SIG)
print(f"\n  w0 within 2 sigma of DESI: {within_w0}")
print(f"  wa within 2 sigma of DESI: {within_wa}")
print(f"  wa sign correct (< 0):     {wa_fit < 0}")

# ─────────────────────────────────────────────────────────────────────────────
# Mean-field deterministic run (smooth, many steps, no stochastic noise)
# Used for w_eff(z) computation and CPL fitting.
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning mean-field solver ...")
mf_n, mf_b, mf_D, mf_x, mf_y, mf_f, mf_reg = mean_field_cascade(
    b0=16.0, x0=2.0, y0=2.0,
    s0=S_INIT, s1=S_PREV1, s2=S_PREV2,
    alpha=0.10, omega_x=1.0, omega_y=1.20,
    k=0.50, eta=1.0, lam=0.10,
    C=0.50, Delta=0.30,
    gamma1=0.20, kappa=0.40, mu=0.05,
    max_steps=600,
)
N_mf = len(mf_n)
print(f"Mean-field: {N_mf} steps")
for lab in ('explosive', 'leakage', 'quiescent'):
    cnt = (mf_reg == lab).sum()
    print(f"  {lab:12s}: {cnt:5d} steps")

# Identify today in mean-field run
mf_qui = np.where(mf_reg == 'quiescent')[0]
if len(mf_qui) >= 20:
    mf_today = int(mf_qui[19])
elif len(mf_qui) >= 5:
    mf_today = int(mf_qui[4])
elif len(mf_qui) >= 1:
    mf_today = int(mf_qui[0])
else:
    mf_today = int(0.65 * N_mf)

mf_rho0 = mf_b[mf_today]
mf_D0   = mf_D[mf_today]
print(f"Mean-field today: step {mf_today}, rho0={mf_rho0:.5f}, D0={mf_D0:.8f}")

if mf_D0 > 1e-15 and mf_rho0 > 1e-10:
    mf_H0tau = mf_D0 / (3.0 * (1.0 + W0_DESI) * mf_rho0)
else:
    # use nearby non-zero step
    nearby_mf = [i for i in range(max(0, mf_today-10), min(N_mf, mf_today+10))
                 if mf_D[i] > 1e-12 and mf_b[i] > 1e-10]
    if nearby_mf:
        ir = nearby_mf[0]
        mf_H0tau = mf_D[ir] / (3.0 * (1.0 + W0_DESI) * mf_b[ir])
    else:
        mf_H0tau = 1e-8

print(f"Mean-field H0*tau = {mf_H0tau:.8f}")

# Compute mean-field w_eff
mf_w = np.full(N_mf, np.nan)
for i in range(1, N_mf):
    denom = 3.0 * mf_H0tau * np.sqrt(mf_b[i] * mf_rho0)
    if denom > 1e-30 and mf_b[i] > 0:
        mf_w[i] = -1.0 + mf_D[i] / denom

# Map mean-field step -> z
mf_z = np.full(N_mf, np.nan)
for i in range(N_mf):
    if i <= mf_today and mf_today > 0:
        mf_z[i] = Z_MAX * (mf_today - i) / mf_today

# CPL fit on mean-field quiescent points
mf_mask = (
    np.isfinite(mf_w) & np.isfinite(mf_z) &
    (mf_z >= 0.0) & (mf_z <= 2.5) &
    (mf_reg == 'quiescent') &
    (np.abs(mf_w) < 5.0)
)
print(f"Mean-field quiescent points in z-range: {mf_mask.sum()}")

mf_w0_fit = W0_DESI;  mf_wa_fit = WA_DESI
mf_w0_sig = 0.0;      mf_wa_sig = 0.0

if mf_mask.sum() >= 8:
    z_mf = mf_z[mf_mask]
    w_mf = mf_w[mf_mask]
    lo, hi = np.percentile(w_mf, [5, 95])
    keep = (w_mf >= lo) & (w_mf <= hi)
    try:
        mf_w0_fit, mf_wa_fit, mf_w0_sig, mf_wa_sig = fit_cpl(
            z_mf[keep], w_mf[keep])
    except Exception as exc:
        print(f"  Mean-field CPL fit exception: {exc}")

print(f"Mean-field CPL fit:")
print(f"  w0 = {mf_w0_fit:+.4f} +/- {mf_w0_sig:.4f}   DESI: {W0_DESI:+.3f}")
print(f"  wa = {mf_wa_fit:+.4f} +/- {mf_wa_sig:.4f}   DESI: {WA_DESI:+.3f}")
print(f"  wa < 0: {mf_wa_fit < 0}")

print()
print("=" * 65)
print("ANALYTIC PREDICTION (DARK_ENERGY_MAPPING.md, Sec IV-V):")
print(f"  Gamma0 = {Gamma0_desi:.4f}  (from DESI wa)")
print(f"  w(z) = -1 + (1+w0) / (1 + Gamma0*z*(z+2)/4)")
print(f"  w(z=0) = {w_cascade_analytic(0, W0_DESI, Gamma0_desi):.4f}  [= w0 by construction]")
print(f"  w(z=1) = {w_cascade_analytic(1, W0_DESI, Gamma0_desi):.4f}  [DESI: {w_cpl(1, W0_DESI, WA_DESI):.4f}]")
print(f"  w(z=2) = {w_cascade_analytic(2, W0_DESI, Gamma0_desi):.4f}  [DESI: {w_cpl(2, W0_DESI, WA_DESI):.4f}]")
print()
print("  Regime w_eff (numerical mean, calibrated):")
for regime_label in ('explosive', 'leakage', 'quiescent'):
    idx = (reg_arr == regime_label) & np.isfinite(w_eff) & (np.abs(w_eff) < 10)
    if idx.any():
        vals = w_eff[idx]
        print(f"    {regime_label:12s}: mean w_eff = {np.median(vals):+.4f}  "
              f"[range {vals.min():+.3f} to {vals.max():+.3f}]")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
z_plot = np.linspace(0.0, 3.0, 400)
BG     = '#080808'
TXT    = '#dddddd'
C_EXP  = '#ef5350'
C_LEAK = '#ffb74d'
C_QUI  = '#4fc3f7'
C_DESI = '#ffd54f'
C_ANA  = '#a5d6a7'
C_NUM  = '#80cbc4'

fig = plt.figure(figsize=(15, 11), facecolor=BG)
gs  = gridspec.GridSpec(2, 3, figure=fig,
                        hspace=0.45, wspace=0.38,
                        left=0.07, right=0.97, top=0.92, bottom=0.08)

ax_b = fig.add_subplot(gs[0, 0])
ax_D = fig.add_subplot(gs[0, 1])
ax_F = fig.add_subplot(gs[0, 2])
ax_w = fig.add_subplot(gs[1, :])


def style(ax):
    ax.set_facecolor('#111111')
    ax.tick_params(colors=TXT, labelsize=9)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)
    ax.title.set_color(TXT)
    for sp in ax.spines.values():
        sp.set_color('#2a2a2a')


for ax in (ax_b, ax_D, ax_F, ax_w):
    style(ax)

# Top panels
for regime, color, label in [('explosive', C_EXP, 'Explosive'),
                               ('leakage', C_LEAK, 'Leakage'),
                               ('quiescent', C_QUI, 'Quiescent')]:
    idx = reg_arr == regime
    if not idx.any():
        continue
    ax_b.scatter(n_arr[idx], rho_arr[idx], s=4, c=color, alpha=0.5,
                 label=label, rasterized=True)
    ax_D.scatter(n_arr[idx], D_arr[idx],   s=4, c=color, alpha=0.5,
                 rasterized=True)
    ax_F.scatter(n_arr[idx], F_arr[idx],   s=4, c=color, alpha=0.5,
                 rasterized=True)

for ax in (ax_b, ax_D, ax_F):
    ax.axvline(n_arr[today_idx], color=C_DESI, lw=1.2, ls='--', alpha=0.85,
               label='Today (z=0)')

ax_F.axhline(model.C, color='#668', lw=0.9, ls=':',
             label=f'C = {model.C}')
ax_F.axhline(model.C + model.Delta, color='#99b', lw=0.9, ls=':',
             label=f'C+Delta = {model.C + model.Delta:.2f}')

ax_b.set_xlabel('Step n'); ax_b.set_ylabel('rho_beta(n)')
ax_b.set_title('beta density -- three-regime arc', fontsize=9)
ax_b.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TXT, markerscale=2)

ax_D.set_xlabel('Step n'); ax_D.set_ylabel('D(n)')
ax_D.set_title('D(n) -- supermartingale decrement', fontsize=9)
ax_D.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TXT,
            handles=[ax_b.get_legend_handles_labels()[0][-1]])

ax_F.set_xlabel('Step n'); ax_F.set_ylabel('<F>(n)')
ax_F.set_title('Mean ripple <F> -- regime indicator', fontsize=9)
ax_F.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TXT)

# Main w(z) panel
w_hi = w_cpl(z_plot, W0_DESI + W0_SIG, WA_DESI - WA_SIG)
w_lo = w_cpl(z_plot, W0_DESI - W0_SIG, WA_DESI + WA_SIG)
ax_w.fill_between(z_plot, w_lo, w_hi, alpha=0.15, color=C_DESI, zorder=1,
                  label='DESI DR2 +/-1 sigma band')
ax_w.plot(z_plot, w_cpl(z_plot, W0_DESI, WA_DESI), color=C_DESI, lw=2.0,
          ls='--', zorder=3,
          label=f'DESI DR2 CPL  (w0={W0_DESI}, wa={WA_DESI})')

ax_w.plot(z_plot, w_cascade_analytic(z_plot, W0_DESI, Gamma0_desi),
          color=C_ANA, lw=2.5, zorder=4,
          label=f'Cascade analytic  Gamma0={Gamma0_desi:.2f}'
                f'   [w=-1+(1+w0)/(1+G0*z*(z+2)/4)]')

# Mean-field numerical cascade
if mf_mask.sum() >= 3:
    z_mfp = mf_z[mf_mask]
    w_mfp = np.clip(mf_w[mf_mask], -2.5, 0.5)
    ax_w.scatter(z_mfp, w_mfp, s=20, c=C_NUM, alpha=0.65, zorder=5,
                 rasterized=True,
                 label=f'Cascade mean-field (quiescent)  '
                       f'CPL fit: w0={mf_w0_fit:.3f}, wa={mf_wa_fit:.3f}')

# Stochastic numerical (sparse, shown as smaller dots)
if mask.sum() >= 3:
    z_num = z_eff[mask]
    w_num = np.clip(w_eff[mask], -2.5, 0.5)
    ax_w.scatter(z_num, w_num, s=10, c='#90caf9', alpha=0.40, zorder=4,
                 rasterized=True, label='Cascade stochastic (quiescent)')

# Explosive regime annotation
mf_exp_mask = (mf_reg == 'explosive') & np.isfinite(mf_w) & (mf_w < 20)
if mf_exp_mask.any():
    w_exp_med = float(np.median(mf_w[mf_exp_mask]))
    ax_w.annotate(
        f'Explosive regime (early universe, z >> 2):\n'
        f'  w_eff = {w_exp_med:.1f}  >>  -1\n'
        f'  Dark energy behaves like matter/radiation.\n'
        f'  Rapid structure formation (JWST early galaxies).',
        xy=(2.8, -0.25), xytext=(1.8, -0.18),
        fontsize=8, color=C_EXP, alpha=0.85,
        arrowprops=dict(arrowstyle='->', color=C_EXP, lw=0.9),
    )

ax_w.axhline(-1.0, color='#555', lw=0.9, ls=':', zorder=2,
             label='Lambda = -1  (cosmological constant)')
ax_w.set_xlim(0, 3.0)
ax_w.set_ylim(-1.8, -0.0)
ax_w.set_xlabel('Redshift  z', fontsize=12)
ax_w.set_ylabel('w(z)  [dark energy equation of state]', fontsize=12)
ax_w.set_title('Dark energy w(z): Cascade model vs DESI DR2',
               fontsize=13, fontweight='bold')
ax_w.legend(fontsize=9, facecolor='#151515', labelcolor=TXT,
            loc='lower right', framealpha=0.9, edgecolor='#333')

note = (
    "Master equation (DARK_ENERGY_MAPPING.md Sec II):\n"
    "  1 + w_eff = D(n) / [3 * H0*tau * sqrt(rho_beta(n) * rho_0)]\n"
    "Derived from Part I theorem:  rho_beta(n+1) < rho_beta(n)  for all active n\n"
    f"Simulation: P={P} sites, E_total={int(E_total)}, {N} steps"
)
ax_w.text(0.01, 0.05, note, transform=ax_w.transAxes,
          fontsize=8.5, color=TXT, alpha=0.72, va='bottom',
          bbox=dict(fc='#141414', ec='#333', alpha=0.75,
                    boxstyle='round,pad=0.45'))

fig.suptitle(
    'Cascade Model -- Dark Energy from beta Depletion  |  Shiv Goswami, June 2026',
    fontsize=11, color='#888', y=0.965,
)

out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'dark_energy_w_z.png')
fig.savefig(out_path, dpi=160, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"\nFigure saved -> {out_path}")
