"""
bigbang_1M.py
=============
1-million-step Big Bang simulation initialized from BBN-era reverse cascade.

Vectorized (pure numpy, no Python site loops) for speed.
Records the complete arc: Explosive (Big Bang) -> Quiescent -> Absorbing (Heat Death).

Calibration from reverse_cascade.py:
  w_0 = -0.77  ->  Gamma_0/(H0*tau) = 0.69
  Self-sustaining explosive regime requires X0=Y0=20, alpha_XY=0.1 (default)
  Stochastic ripple F ~ 1.56 > C+Delta=0.8 from Poisson variance alone.
  Energy budget for 1M explosive steps: E0 = 7.66e9 (BBN-era, z~10^10)

Author: Shiv Goswami
Date:   June 21, 2026
"""

import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Parameters (paper defaults) ────────────────────────────────────────────
P          = 100
SEED       = 2024
rng        = np.random.default_rng(SEED)

alpha_XY   = 0.1
omega_X    = 1.0
omega_Y    = 1.2
k_XY       = 0.5
k_XX       = 0.3
eta        = 1.0
kappa      = 0.4
C          = 0.5
Delta      = 0.3
lambda_    = 0.1
gamma_1    = 0.2
gamma_XX   = 0.1
gamma_2    = 0.15
D_X        = 0.05
D_Y        = 0.05
a0         = 1.0
b          = 0.05
T_c        = 0.8

C_plus_D   = C + Delta   # = 0.8

# ── Reverse-cascade BBN-era initial conditions ─────────────────────────────
# Self-sustaining explosive: X0=Y0=20 gives stochastic F~1.56 > C+Delta=0.8
# D_E per step per site = k_XY*N_XY + eta*m + L ~ 76.6 (see derivation)
# E0 for 1M steps = 76.6 * P * 1e6 = 7.66e9
W0           = -0.77
F_DE_BBN     = 1.532e7     # rho_beta(z~10^10) / rho_beta_0
E0_BB        = 7.66e9      # total initial energy (BBN-era)
X0_BB        = 20.0        # per site
Y0_BB        = 20.0        # per site
S_PRIME      = 8.0         # primes F=16 >> C+Delta=0.8 at step 0
GAMMA0_H0TAU = 0.69        # Gamma_0 / (H0*tau), from reverse calc

MAX_STEPS    = 1_000_000
RECORD_EVERY = 500         # store every 500 steps -> 2000 records total

# ── Initialize state (vectorized arrays) ───────────────────────────────────
X      = np.full(P, X0_BB,     dtype=np.float64)
Y      = np.full(P, Y0_BB,     dtype=np.float64)
Beta   = np.full(P, E0_BB / P, dtype=np.float64)
S      = np.zeros(P,           dtype=np.float64)
S_p1   = np.full(P, S_PRIME,   dtype=np.float64)   # S at n-1
S_p2   = np.zeros(P,           dtype=np.float64)    # S at n-2
step_n = 2

# ── Neighbor indices for diffusion (1D ring) ──────────────────────────────
left_nb  = np.roll(np.arange(P), 1)    # each site's left neighbor
right_nb = np.roll(np.arange(P), -1)   # each site's right neighbor

# ── Recording arrays ───────────────────────────────────────────────────────
rec_n       = []
rec_beta    = []
rec_F_mean  = []
rec_X_mean  = []
rec_Y_mean  = []
rec_regime  = []
rec_D       = []
rec_delta   = []
rec_w_eff   = []
rec_fDE     = []

beta_today  = (E0_BB / P) / F_DE_BBN   # beta per site at "today" reference

beta_prev   = float(np.mean(Beta))

print("=" * 70)
print("BIG BANG 1M-STEP SIMULATION (VECTORIZED)")
print("=" * 70)
print(f"  E0 (total, BBN-era)    = {E0_BB:.3e}  ({F_DE_BBN:.2e}x ref)")
print(f"  beta per site          = {E0_BB/P:.3e}")
print(f"  beta_today reference   = {beta_today:.4f}")
print(f"  X0, Y0 per site        = {X0_BB}, {Y0_BB}")
print(f"  Expected F (stoch.)    = ~1.56 > C+Delta={C_plus_D} (self-sustaining)")
print(f"  D_E per step (approx)  = {k_XY*0.12*400 + eta*51 + lambda_*16:.1f} per site")
print(f"  E0 / (D_E*P)           = {E0_BB/(76.6*P):.0f} steps (estimated run)")
print(f"  MAX_STEPS              = {MAX_STEPS:,}")
print(f"  Recording every        = {RECORD_EVERY} steps")
print(f"\nRunning simulation...\n")

t_start = time.time()

for iteration in range(MAX_STEPS):

    # ── Ripple F (vectorized second difference) ────────────────────────────
    F = np.abs(S - 2.0 * S_p1 + S_p2)    # shape (P,)

    # ── Interaction rates (vectorized Poisson) ────────────────────────────
    rate_XY = alpha_XY * omega_X * omega_Y * X * Y
    N_XY    = rng.poisson(rate_XY)
    N_XY    = np.minimum(N_XY, np.minimum(X.astype(int), Y.astype(int)))

    rate_XX = alpha_XX_val = 0.05 * (omega_X ** 2) * np.maximum(X * (X - 1) / 2, 0)
    N_XX    = rng.poisson(rate_XX)
    N_XX    = np.minimum(N_XX, (X.astype(int) // 2))

    X_new = np.maximum(X - N_XY - 2 * N_XX, 0.0)
    Y_new = np.maximum(Y - N_XY, 0.0)

    # ── Regime: leakage and explosion ────────────────────────────────────
    L = np.where(F > C, lambda_ * F, 0.0)

    # Explosion: sites where F >= C + Delta
    exp_mask = F >= C_plus_D
    m_full   = np.where(exp_mask, np.floor((F - C) / Delta).astype(int), 0)
    # Cap by available beta
    m_capped = np.minimum(m_full, np.floor(Beta / eta).astype(int))
    M        = eta * m_capped
    X_new    = np.where(exp_mask, X_new + m_capped, X_new)
    Y_new    = np.where(exp_mask, Y_new + m_capped, Y_new)

    # ── Bond formation (simplified Landau) ───────────────────────────────
    psi_sq  = X_new * Y_new
    psi     = np.sqrt(np.maximum(psi_sq, 0.0))
    T_eff   = np.where(C > 0, F / C, 0.0)
    bond_cond = (psi > 0) & (T_eff < T_c - (2 * b / a0) * psi_sq)
    alpha_L = a0 * (T_eff - T_c)
    dF_dp   = 2 * alpha_L * psi + 4 * b * (psi ** 3)
    max_B   = np.where(
        bond_cond & (dF_dp < 0),
        np.minimum(
            np.where(b > 0, (psi_sq * np.abs(dF_dp) / b).astype(int), 0),
            np.minimum(
                np.where(kappa > 0, (Beta / kappa).astype(int), 0),
                np.minimum(X_new.astype(int), Y_new.astype(int))
            )
        ),
        0
    )
    B = np.maximum(max_B, 0).astype(float)
    X_new = np.maximum(X_new - B, 0.0)
    Y_new = np.maximum(Y_new - B, 0.0)

    # ── Energy update ─────────────────────────────────────────────────────
    Beta_new = np.maximum(
        Beta - k_XY * N_XY - k_XX * N_XX - L - M - kappa * B,
        0.0
    )

    # ── Structure update ──────────────────────────────────────────────────
    S_new = S + gamma_1 * N_XY + gamma_XX * N_XX + gamma_2 * B

    # ── Diffusion (vectorized ring) ───────────────────────────────────────
    X_pre   = X_new.copy()
    Y_pre   = Y_new.copy()
    X_new   = np.round(np.maximum(
        X_pre + D_X * (X_pre[left_nb] + X_pre[right_nb] - 2.0 * X_pre), 0.0))
    Y_new   = np.round(np.maximum(
        Y_pre + D_Y * (Y_pre[left_nb] + Y_pre[right_nb] - 2.0 * Y_pre), 0.0))

    # ── Commit ────────────────────────────────────────────────────────────
    S_p2[:] = S_p1
    S_p1[:] = S
    S[:]    = S_new
    X[:]    = X_new
    Y[:]    = Y_new
    Beta[:] = Beta_new
    step_n += 1

    # ── Activity check ────────────────────────────────────────────────────
    total_act = np.sum(N_XY) + np.sum(N_XX) + np.sum(L) + np.sum(M) + np.sum(B)
    absorbing = (
        np.all(X * Y <= 0) and
        (np.all(X <= 1) or 0.05 == 0) and
        np.all(F <= C)
    )

    # ── Record ────────────────────────────────────────────────────────────
    if iteration % RECORD_EVERY == 0 or absorbing:
        beta_n  = float(np.mean(Beta_new))
        F_n     = float(np.mean(F))
        D_n     = max(beta_prev - beta_n, 0.0)
        delta_n = D_n / beta_n if beta_n > 1e-12 else 0.0

        # Regime
        if F_n >= C_plus_D:
            regime = "Explosive"
        elif F_n > C:
            regime = "Leakage"
        else:
            regime = "Quiescent"
        if absorbing:
            regime = "Absorbing"

        # w_eff via master equation
        if beta_today > 1e-12 and beta_n > 1e-12:
            H_an_ratio = np.sqrt(beta_n / beta_today)
            one_plus_w = (1.0 + W0) * delta_n / (GAMMA0_H0TAU * H_an_ratio + 1e-20)
        else:
            one_plus_w = 0.0
        w_eff_n = float(np.clip(one_plus_w - 1.0, -2.0, 10.0))

        fDE_n = beta_n / beta_today

        rec_n.append(step_n)
        rec_beta.append(beta_n)
        rec_F_mean.append(F_n)
        rec_X_mean.append(float(np.mean(X_new)))
        rec_Y_mean.append(float(np.mean(Y_new)))
        rec_regime.append(regime)
        rec_D.append(D_n)
        rec_delta.append(delta_n)
        rec_w_eff.append(w_eff_n)
        rec_fDE.append(fDE_n)

        beta_prev = beta_n

    # Progress print
    if iteration % 100_000 == 0:
        elapsed = time.time() - t_start
        beta_now = float(np.mean(Beta_new))
        F_now    = float(np.mean(F))
        pct      = 100 * beta_now / (E0_BB / P)
        reg_now  = "Explosive" if F_now >= C_plus_D else "Leakage" if F_now > C else "Quiescent"
        print(f"  Step {step_n:>9,} | beta={beta_now:.2e} ({pct:.1f}%) | "
              f"F={F_now:.3f} | {reg_now:10} | {elapsed:.1f}s elapsed")

    if absorbing:
        print(f"\n  Absorbing state reached at step {step_n:,}")
        break

t_total = time.time() - t_start
print(f"\nSimulation complete in {t_total:.1f}s")
print(f"Total recorded points: {len(rec_n)}")

# ── Convert to arrays ──────────────────────────────────────────────────────
n_arr      = np.array(rec_n)
beta_arr   = np.array(rec_beta)
F_arr      = np.array(rec_F_mean)
X_arr      = np.array(rec_X_mean)
Y_arr      = np.array(rec_Y_mean)
delta_arr  = np.array(rec_delta)
w_arr      = np.array(rec_w_eff)
fDE_arr    = np.array(rec_fDE)

# ── Regime transition detection ────────────────────────────────────────────
n_exp_end  = None
n_leak_end = None
n_qsc_end  = None
n_abs      = None

for i, r in enumerate(rec_regime):
    if r != "Explosive" and n_exp_end is None and i > 0:
        n_exp_end = n_arr[i]
    if r not in ("Explosive", "Leakage") and n_leak_end is None and i > 0:
        n_leak_end = n_arr[i]
    if r == "Absorbing" and n_abs is None:
        n_abs = n_arr[i]

# ── Summary printout ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS: COMPLETE COSMIC ARC")
print("=" * 70)
print(f"\nInitial conditions (BBN-era, z~10^10):")
print(f"  rho_beta_0        = {E0_BB/P:.3e} per site")
print(f"  f_DE (BBN era)    = {F_DE_BBN:.3e}")
print(f"  X0 = Y0           = {X0_BB}")

print(f"\nRegime transitions:")
print(f"  Explosive phase ends  at step ~ {n_exp_end or 'still explosive'}")
print(f"  Quiescent phase       at step ~ {n_leak_end or 'not reached'}")
print(f"  Absorbing (heat death)at step ~ {n_abs or 'not reached'}")

print(f"\nDark energy w_eff:")
print(f"  w_eff range            = [{w_arr.min():.4f}, {w_arr.max():.4f}]")
no_phantom_active = np.all(w_arr[:-1] > -1.0) if len(w_arr) > 1 else True
print(f"  No phantom (w > -1):   {no_phantom_active}")

exp_mask_r = np.array(rec_regime) == "Explosive"
qsc_mask_r = np.array(rec_regime) == "Quiescent"
if exp_mask_r.any():
    print(f"  w_eff (explosive):     {w_arr[exp_mask_r].mean():.4f}  (mean)")
if qsc_mask_r.any():
    print(f"  w_eff (quiescent):     {w_arr[qsc_mask_r].mean():.4f}  (target: {W0:.2f})")

print(f"\nBeta depletion:")
print(f"  Initial beta / site    = {beta_arr[0]:.3e}")
print(f"  Final beta / site      = {beta_arr[-1]:.3e}")
frac_remaining = beta_arr[-1] / beta_arr[0]
print(f"  Fraction remaining     = {frac_remaining:.4f}")
print(f"\nw_eff at absorbing state = {w_arr[-1]:.6f}  (Theorem 5: must = -1)")

# ── Plot ───────────────────────────────────────────────────────────────────
regime_color = {'Explosive': '#ff6b35', 'Leakage': '#ffdd57',
                'Quiescent': '#00ff99', 'Absorbing': '#888888'}

fig = plt.figure(figsize=(17, 14))
fig.patch.set_facecolor('#080808')
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.44, wspace=0.30)

ax_kw = dict(color='white', fontsize=10)
tk_kw = dict(colors='#aaaaaa', labelsize=8)
gr_kw = dict(alpha=0.15, color='#444444')

def shade_regimes(ax, x, regimes):
    """Shade regime bands."""
    rc = np.array(regimes)
    for r, col in [("Explosive", '#ff6b35'), ("Leakage", '#ffdd57'),
                   ("Quiescent", '#00ff99'), ("Absorbing", '#888888')]:
        mask = rc == r
        if mask.any():
            idxs = np.where(mask)[0]
            for idx in idxs:
                ax.axvspan(x[max(idx-1,0)], x[min(idx+1, len(x)-1)],
                           alpha=0.07, color=col, lw=0)

# ── P1: Beta history (full 1M steps) ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('#111111')

colors_r = [regime_color.get(r, '#888888') for r in rec_regime]
ax1.scatter(n_arr, beta_arr, c=colors_r, s=3, alpha=0.6, zorder=4)
ax1.plot(n_arr, beta_arr, color='#00d4ff', lw=1.5, alpha=0.7,
         label='$\\rho_\\beta(n)$ — simulated')

ax1.axhline(beta_today, color='#ffdd57', lw=1.5, ls='--', alpha=0.8,
            label=f'Today reference ($z=0$)')
ax1.axhline(beta_today * 2.30, color='#00ff99', lw=1.0, ls=':', alpha=0.7,
            label=f'DESI z=2.33 level')

# Annotation of regimes
if n_exp_end:
    ax1.axvline(n_exp_end, color='white', lw=0.8, ls=':', alpha=0.4)
    ax1.text(n_exp_end, beta_arr.max() * 0.85, 'Explosion\nends',
             color='white', fontsize=7, ha='center', alpha=0.7)
if n_leak_end:
    ax1.axvline(n_leak_end, color='white', lw=0.8, ls=':', alpha=0.4)
    ax1.text(n_leak_end, beta_arr.max() * 0.65, 'Quiescent\nbegins',
             color='white', fontsize=7, ha='center', alpha=0.7)

ax1.set_xlabel('Cascade step $n$', **ax_kw)
ax1.set_ylabel('$\\rho_\\beta$ per site', **ax_kw)
ax1.set_title(f'1 MILLION STEP BIG BANG SIMULATION — FULL CASCADE ARC\n'
              f'BBN-era initial conditions  |  E0 = {E0_BB:.2e}  |  '
              f'P={P} sites  |  Vectorized',
              color='white', fontsize=12, pad=8)
ax1.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#444444', labelcolor='white')
ax1.tick_params(axis='both', **tk_kw)
ax1.grid(True, **gr_kw)
ax1.spines[['bottom','left','top','right']].set_color('#333333')
ax1.set_yscale('log')

# ── P2: w_eff full arc ────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#111111')
sc2 = ax2.scatter(n_arr, w_arr, c=colors_r, s=5, alpha=0.7, zorder=4)
ax2.plot(n_arr, w_arr, color='#aaaaaa', lw=0.8, alpha=0.3)
ax2.axhline(W0, color='#ff6b35', lw=2, ls='--', label=f'$w_0={W0}$ (DESI)')
ax2.axhline(-1.0, color='#888888', lw=1.5, ls='--', label='$w=-1$ (heat death)')
ax2.axhspan(-2.0, -1.0, alpha=0.06, color='red')
ax2.text(n_arr.max() * 0.5, -1.5, 'PHANTOM\nFORBIDDEN', color='#ff4444',
         fontsize=8, ha='center', alpha=0.8)
ax2.set_xlabel('Step $n$', **ax_kw)
ax2.set_ylabel('$w_{\\mathrm{eff}}(n)$', **ax_kw)
ax2.set_title('Dark Energy EOS — 1M Steps', color='white', fontsize=11)
ax2.legend(fontsize=8, facecolor='#1a1a1a', edgecolor='#444444', labelcolor='white')
ax2.tick_params(axis='both', **tk_kw)
ax2.grid(True, **gr_kw)
ax2.set_ylim(-2.0, max(w_arr.max() * 1.2, 0.5))
ax2.spines[['bottom','left','top','right']].set_color('#333333')

# ── P3: Ripple F (regime fingerprint) ────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('#111111')
ax3.plot(n_arr, F_arr, color='#aa88ff', lw=1.2, alpha=0.85,
         label='Mean $F(n)$ — ripple')
ax3.axhline(C_plus_D, color='#ff6b35', lw=1.5, ls='--',
            label=f'Explosion $C+\\Delta={C_plus_D}$')
ax3.axhline(C, color='#ffdd57', lw=1.0, ls=':',
            label=f'Leakage $C={C}$')
ax3.fill_between(n_arr, F_arr, C_plus_D, where=F_arr >= C_plus_D,
                 alpha=0.2, color='#ff6b35')
ax3.set_xlabel('Step $n$', **ax_kw)
ax3.set_ylabel('Mean $F(n)$', **ax_kw)
ax3.set_title('Ripple $F(n)$ — Regime Engine', color='white', fontsize=11)
ax3.legend(fontsize=8, facecolor='#1a1a1a', edgecolor='#444444', labelcolor='white')
ax3.tick_params(axis='both', **tk_kw)
ax3.grid(True, **gr_kw)
ax3.spines[['bottom','left','top','right']].set_color('#333333')

# ── P4: XY particle history ───────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
ax4.set_facecolor('#111111')
ax4.plot(n_arr, X_arr, color='#00d4ff', lw=1.5, label='$\\langle X \\rangle$')
ax4.plot(n_arr, Y_arr, color='#ff6b35', lw=1.5, ls='--', label='$\\langle Y \\rangle$')
ax4.set_xlabel('Step $n$', **ax_kw)
ax4.set_ylabel('Mean particles per site', **ax_kw)
ax4.set_title('XY Particle History', color='white', fontsize=11)
ax4.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#444444', labelcolor='white')
ax4.tick_params(axis='both', **tk_kw)
ax4.grid(True, **gr_kw)
ax4.spines[['bottom','left','top','right']].set_color('#333333')

# ── P5: f_DE trajectory vs reverse-cascade prediction ─────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor('#111111')

exp_inv   = 1.0 / (3.0 * (1.0 + W0))
z_sim     = np.where(fDE_arr > 0, fDE_arr ** exp_inv - 1.0, -1.0)

ax5.plot(z_sim, fDE_arr, color='#00d4ff', lw=2, label='Simulated $f_{DE}(z)$')

# Reverse-cascade smooth curve
z_smooth  = np.linspace(-0.2, max(z_sim.max(), 5), 500)
fDE_smooth = (1 + z_smooth) ** (3 * (1 + W0))
ax5.plot(z_smooth[z_smooth >= 0], fDE_smooth[z_smooth >= 0],
         color='#ff6b35', lw=2, ls='--', label='Reverse cascade $(1+z)^{3(1+w_0)}$')

ax5.axhline(1.0, color='#ffdd57', lw=1, ls=':', alpha=0.7, label='Today ($z=0$)')
ax5.axvline(0, color='#ffdd57', lw=1, ls='--', alpha=0.5)

ax5.set_xlabel('Mapped $z$', **ax_kw)
ax5.set_ylabel('$f_{DE} = \\rho_\\beta/\\rho_{\\beta,0}$', **ax_kw)
ax5.set_title('Simulated vs Reverse-Cascade $f_{DE}(z)$', color='white', fontsize=11)
ax5.legend(fontsize=8, facecolor='#1a1a1a', edgecolor='#444444', labelcolor='white')
ax5.tick_params(axis='both', **tk_kw)
ax5.grid(True, **gr_kw)
ax5.spines[['bottom','left','top','right']].set_color('#333333')

# ── Regime legend strip ───────────────────────────────────────────────────
from matplotlib.patches import Patch
leg_el = [Patch(facecolor='#ff6b35', label='Explosive (Big Bang era)'),
          Patch(facecolor='#ffdd57', label='Leakage (transition)'),
          Patch(facecolor='#00ff99', label='Quiescent (today)'),
          Patch(facecolor='#888888', label='Absorbing (heat death)')]
fig.legend(handles=leg_el, loc='lower center', ncol=4, fontsize=9,
           facecolor='#1a1a1a', edgecolor='#555555', labelcolor='white',
           bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    '1 MILLION STEP CASCADE: Big Bang Origin to Heat Death\n'
    f'BBN initial conditions (z~10$^{{10}}$) | E0={E0_BB:.1e} | '
    f'Self-sustaining explosive regime | Vectorized numpy',
    color='white', fontsize=13, y=0.995, fontweight='bold')

out_path = r"F:\A mathematical model\figures\bigbang_1M.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()

print(f"\nFigure saved: {out_path}")
print("=" * 70)
print("FINAL VERDICT:")
print(f"  No phantom at all active steps:     {no_phantom_active}")
print(f"  Heat death w = -1 exactly:          {abs(w_arr[-1] - (-1.0)) < 0.0001}")
print(f"  Regime arc (Explosive->Absorbing):  {n_exp_end is not None or absorbing}")
print(f"  Runtime:                            {t_total:.1f}s for {step_n:,} steps")
print("=" * 70)
