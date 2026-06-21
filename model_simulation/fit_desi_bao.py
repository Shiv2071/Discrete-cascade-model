"""
fit_desi_bao.py
Direct chi-squared fit of four dark energy models to DESI DR2 BAO data.

Models compared:
  LCDM:     w = -1 (constant)                           2 free params
  CAS_LOG:  w = -1 + (1+w0)/(1 + k*ln(1+z)), k fixed   2 free params
  CPL:      w = w0 + wa*z/(1+z),  wa free               3 free params
  CAS_POLY: w = -1 + (1+w0)/(1+G0*z(z+2)/4), G0 free   3 free params

Free parameters in all models: theta = H0*r_d/c, Omega_m.
w0 = -0.77 fixed (DESI+CMB constraint, consistent across models).

DESI DR2 BAO data: arXiv:2503.14738 (Table 2) + arXiv:2503.14739 (Lya).

Author: Shiv Goswami
Date: June 21, 2026
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── DESI DR2 BAO data ───────────────────────────────────────────────────────
# (z_eff, type, val1, sig1, val2, sig2, rho_12)
# type='DV'  → only D_V/r_d measured (BGS, z=0.295)
# type='DMH' → D_M/r_d and D_H/r_d with off-diagonal correlation rho

DESI = [
    #  z_eff  type   DM/rd    sDM    DH/rd   sDH    rho
    (0.295, 'DV',  7.942, 0.075,  None,   None,   None),
    (0.510, 'DMH', 13.588, 0.167, 21.863, 0.425, -0.459),
    (0.706, 'DMH', 17.351, 0.177, 19.455, 0.330, -0.404),
    (0.934, 'DMH', 21.576, 0.152, 17.641, 0.193, -0.416),
    (1.321, 'DMH', 27.601, 0.318, 14.176, 0.221, -0.434),
    (1.484, 'DMH', 30.512, 0.760, 12.817, 0.516, -0.500),
    (2.330, 'DMH', 38.988, 0.531,  8.632, 0.101, -0.431),
]

N_DATA = len(DESI)
z_eff  = np.array([d[0] for d in DESI])
W0     = -0.77       # today's dark energy w (DESI+CMB, fixed)
R_D    = 147.05      # Mpc, Planck 2018 sound horizon
C_KMS  = 299792.458  # km/s

# ─── z integration grid ──────────────────────────────────────────────────────
Nz     = 1500
z_grid = np.linspace(0.0, 4.0, Nz)
dz     = np.diff(z_grid)        # (Nz-1,)
z1     = 1.0 + z_grid           # 1+z, shape (Nz,)
z3     = z1 ** 3
z4     = z1 ** 4
OMEGA_R = 9.0e-5

# Precompute interpolation indices and weights at z_eff
idx    = np.searchsorted(z_grid, z_eff, side='right') - 1
idx    = np.clip(idx, 0, Nz - 2)
zlo    = z_grid[idx];   zhi = z_grid[idx + 1]
w_hi   = (z_eff - zlo) / (zhi - zlo);   w_lo = 1.0 - w_hi


def interp_at_zeff(mat):
    """Interpolate rows of mat (Nz, nP) at z_eff. Returns (N_DATA, nP)."""
    return (w_lo[:, None] * mat[idx, :] + w_hi[:, None] * mat[idx + 1, :])


# ─── f_DE implementations ─────────────────────────────────────────────────────

def _trapz_cumulative(integrand, dz_arr):
    """Cumulative trapezoid integral. integrand: (Nz, nP). Returns (Nz, nP)."""
    cum = np.zeros_like(integrand)
    cum[1:] = np.cumsum(
        0.5 * (integrand[:-1] + integrand[1:]) * dz_arr[:, None], axis=0)
    return cum


def f_DE_lcdm():
    """w = -1 everywhere. f_DE = 1."""
    return np.ones((Nz, 1))


def f_DE_cas_log():
    """
    CAS_LOG: w(z) = -1 + (1+w0) / (1 + k*ln(1+z)),  k = 3*(1+w0)/2
    Analytic result: f_DE = (1 + k*ln(1+z))^2
    Derived from the proved β supermartingale + Friedmann Hubble analog.
    No free shape parameter.
    """
    k = 3.0 * (1.0 + W0) / 2.0
    f = (1.0 + k * np.log(z1))[:, None] ** 2  # (Nz, 1)
    return f


def f_DE_cpl(wa_arr):
    """
    CPL: w = w0 + wa*z/(1+z).
    f_DE = (1+z)^{3(1+w0+wa)} * exp(-3*wa*z/(1+z))
    wa_arr: (nWA,)
    Returns: (Nz, nWA)
    """
    wa = wa_arr[None, :]    # (1, nWA)
    z2d = z_grid[:, None]  # (Nz, 1)
    z12d = z1[:, None]     # (Nz, 1)
    return z12d ** (3.0 * (1.0 + W0 + wa)) * np.exp(-3.0 * wa * z2d / z12d)


def f_DE_cas_poly(G0_arr):
    """
    CAS_POLY: w = -1 + (1+w0) / (1 + G0*z*(z+2)/4)
    Requires numerical integration of (1+w)/(1+z) over z.
    G0_arr: (nG0,)
    Returns: (Nz, nG0)
    """
    G0 = G0_arr[None, :]   # (1, nG0)
    z2d = z_grid[:, None]
    z12d = z1[:, None]
    integrand = (1.0 + W0) / ((1.0 + G0 * z2d * (z2d + 2.0) / 4.0) * z12d)
    cum = _trapz_cumulative(integrand, dz)
    return np.exp(3.0 * cum)


# ─── chi-squared ─────────────────────────────────────────────────────────────

def chi2_vec(DM_at_z, DH_at_z):
    """
    DM_at_z, DH_at_z: (N_DATA, nP)
    Returns chi2: (nP,)
    """
    nP = DM_at_z.shape[1]
    chi2 = np.zeros(nP)

    # BGS: D_V only
    z0 = z_eff[0]
    DV = (z0 * DM_at_z[0] ** 2 * DH_at_z[0]) ** (1.0 / 3.0)
    chi2 += ((DV - DESI[0][2]) / DESI[0][3]) ** 2

    for i in range(1, N_DATA):
        _, _, DM_d, sDM, DH_d, sDH, rho = DESI[i]
        dM = DM_at_z[i] - DM_d
        dH = DH_at_z[i] - DH_d
        f  = 1.0 / (1.0 - rho ** 2)
        chi2 += f * ((dM / sDM) ** 2
                     - 2 * rho * (dM / sDM) * (dH / sDH)
                     + (dH / sDH) ** 2)
    return chi2


# ─── grid search ─────────────────────────────────────────────────────────────

def search(f_de_mat, param_label, param_grid,
           theta_grid, om_grid):
    """
    f_de_mat: (Nz, nP)  — precomputed for all shape params
    Returns (best_chi2, best_theta, best_Om, best_param)
    """
    nP = f_de_mat.shape[1]
    best  = np.full(nP, np.inf)
    bt    = np.zeros(nP)
    bom   = np.zeros(nP)
    total = len(theta_grid) * len(om_grid)
    done  = 0

    for th in theta_grid:
        for Om in om_grid:
            Ode = 1.0 - Om - OMEGA_R
            E2  = (Om * z3[:, None]
                   + Ode * f_de_mat
                   + OMEGA_R * z4[:, None])
            E2  = np.maximum(E2, 1e-10)
            E   = np.sqrt(E2)

            # DM/r_d via cumulative integral of 1/E
            inv_E = 1.0 / E  # (Nz, nP)
            DM_mat = np.zeros_like(inv_E)
            DM_mat[1:] = np.cumsum(
                0.5 * (inv_E[:-1] + inv_E[1:]) * dz[:, None], axis=0)
            DM_mat /= th

            DH_mat = 1.0 / (th * E)  # (Nz, nP)

            DM_ze = interp_at_zeff(DM_mat)   # (N_DATA, nP)
            DH_ze = interp_at_zeff(DH_mat)

            c2 = chi2_vec(DM_ze, DH_ze)
            imp = c2 < best
            best = np.where(imp, c2, best)
            bt   = np.where(imp, th, bt)
            bom  = np.where(imp, Om, bom)

            done += 1
            if done % 300 == 0:
                pct = 100 * done / total
                print(f"  {param_label}: {pct:.0f}% ...", end='\r')

    ib = np.argmin(best)
    return float(best[ib]), float(bt[ib]), float(bom[ib]), float(param_grid[ib])


# ─── parameter grids ─────────────────────────────────────────────────────────
THETA  = np.linspace(0.025, 0.038, 55)   # H0*r_d/c
OM     = np.linspace(0.24,  0.42,  55)   # Omega_m
WA     = np.linspace(-2.5,  1.0,   70)   # wa (CPL)
G0     = np.linspace(0.3,  12.0,   65)   # Gamma0 (cascade poly)

print("=" * 66)
print("DESI DR2 BAO FIT: CASCADE vs CPL vs LCDM")
print("=" * 66)
print(f"Data:  {N_DATA} tracers, 13 measurements")
print(f"Fixed: w0 = {W0},  r_d = {R_D} Mpc (Planck 2018)")
print()

# ── LCDM (2 params) ──────────────────────────────────────────────
print("Fitting LCDM (w = -1, 2 free params) ...")
fdm_lcdm = f_DE_lcdm()
chi2_L, th_L, Om_L, _ = search(fdm_lcdm, "LCDM", np.array([0.0]), THETA, OM)
H0_L = th_L * C_KMS / R_D
print(f"  chi2 = {chi2_L:.4f}  (11 dof)")
print(f"  H0 = {H0_L:.2f} km/s/Mpc,  Omega_m = {Om_L:.4f}")

# ── CASCADE LOG (2 params) ────────────────────────────────────────
print("\nFitting CASCADE_LOG (logarithmic w(z), 2 free params) ...")
fdm_log = f_DE_cas_log()
chi2_CL, th_CL, Om_CL, _ = search(fdm_log, "CAS_LOG", np.array([0.0]), THETA, OM)
H0_CL = th_CL * C_KMS / R_D
print(f"  chi2 = {chi2_CL:.4f}  (11 dof)")
print(f"  H0 = {H0_CL:.2f} km/s/Mpc,  Omega_m = {Om_CL:.4f}")
print(f"  w(z) = -1 + (1+w0)/(1 + {3*(1+W0)/2:.3f}*ln(1+z))")

# ── CPL (3 params) ────────────────────────────────────────────────
print("\nFitting CPL (w0 + wa*z/(1+z), 3 free params) ...")
fdm_cpl = f_DE_cpl(WA)
chi2_CPL, th_CPL, Om_CPL, wa_CPL = search(fdm_cpl, "CPL", WA, THETA, OM)
H0_CPL = th_CPL * C_KMS / R_D
print(f"  chi2 = {chi2_CPL:.4f}  (10 dof)")
print(f"  H0 = {H0_CPL:.2f} km/s/Mpc,  Omega_m = {Om_CPL:.4f},  wa = {wa_CPL:.3f}")

# ── CASCADE POLY (3 params) ───────────────────────────────────────
print("\nFitting CASCADE_POLY (polynomial w(z), 3 free params) ...")
fdm_poly = f_DE_cas_poly(G0)
chi2_CP, th_CP, Om_CP, G0_CP = search(fdm_poly, "CAS_POLY", G0, THETA, OM)
H0_CP = th_CP * C_KMS / R_D
print(f"  chi2 = {chi2_CP:.4f}  (10 dof)")
print(f"  H0 = {H0_CP:.2f} km/s/Mpc,  Omega_m = {Om_CP:.4f},  Gamma0 = {G0_CP:.3f}")

# ─── Summary ─────────────────────────────────────────────────────
print()
print("=" * 66)
print("RESULTS SUMMARY")
print("=" * 66)
print(f"{'Model':<15} {'chi2':>8}  {'dof':>4}  {'chi2/dof':>9}  {'Delta_chi2 vs LCDM':>20}")
for name, c2, dof in [
    ("LCDM",      chi2_L,   11),
    ("CAS_LOG",   chi2_CL,  11),
    ("CPL",       chi2_CPL, 10),
    ("CAS_POLY",  chi2_CP,  10),
]:
    delta = chi2_L - c2
    print(f"  {name:<13} {c2:>8.4f}  {dof:>4}  {c2/dof:>9.4f}  {delta:>+.4f}")

print()
print("  CAS_LOG preferred over LCDM:", "YES" if chi2_CL < chi2_L else "NO",
      f"  (Delta_chi2 = {chi2_L - chi2_CL:+.4f})")
print("  CAS_POLY preferred over CPL:", "YES" if chi2_CP < chi2_CPL else "NO",
      f"  (Delta_chi2 = {chi2_CPL - chi2_CP:+.4f})")
print("  CPL preferred over LCDM:    ", "YES" if chi2_CPL < chi2_L else "NO",
      f"  (Delta_chi2 = {chi2_L - chi2_CPL:+.4f})")

print()
print("PHYSICAL INTERPRETATION:")
print(f"  CAS_LOG:  k = {3*(1+W0)/2:.4f},  fully determined by w0 = {W0}")
print(f"            w stays > -1 at all z (no phantom)")
print(f"            w(z=1) = {-1+(1+W0)/(1+(3*(1+W0)/2)*np.log(2)):.4f}")
print(f"            w(z=2) = {-1+(1+W0)/(1+(3*(1+W0)/2)*np.log(3)):.4f}")
print(f"  CPL:      wa = {wa_CPL:.3f}")
phantom_z = W0 / (-wa_CPL - W0 - 1) if (wa_CPL + W0 + 1) < 0 else None
if phantom_z is not None and phantom_z > 0:
    print(f"            phantom crossing at z = {phantom_z:.3f}  (w < -1 for z > {phantom_z:.2f})")
else:
    print(f"            no phantom crossing in z > 0 range")
print(f"  CAS_POLY: Gamma0 = {G0_CP:.3f}")
print("=" * 66)

# ─── Figure ──────────────────────────────────────────────────────
z_plt = np.linspace(0.0, 3.0, 400)
z1_p  = 1.0 + z_plt

# w(z) curves
k_log = 3.0 * (1.0 + W0) / 2.0
w_log  = -1 + (1 + W0) / (1 + k_log * np.log(z1_p))
w_cpl  = W0 + wa_CPL * z_plt / z1_p
w_poly = -1 + (1 + W0) / (1 + G0_CP * z_plt * (z_plt + 2) / 4)
w_lcdm = np.full_like(z_plt, -1.0)

# f_DE on fine grid
def f_log_fine(z):
    return (1 + k_log * np.log(1 + z)) ** 2

def f_cpl_fine(z, wa):
    return (1+z)**(3*(1+W0+wa)) * np.exp(-3*wa*z/(1+z))

def f_poly_fine(z, G0):
    dz_ = z[1]-z[0]
    itg = (1+W0)/((1+G0*z*(z+2)/4)*(1+z))
    cum = np.zeros_like(z)
    cum[1:] = np.cumsum(0.5*(itg[:-1]+itg[1:])*dz_)
    return np.exp(3*cum)

# E(z) for best-fit params
def E_model(z, Om, f_de_z):
    Ode = 1 - Om - OMEGA_R
    return np.sqrt(Om*(1+z)**3 + Ode*f_de_z + OMEGA_R*(1+z)**4)

z_f = np.linspace(0.001, 3.0, 500)
f_log_f  = f_log_fine(z_f)
f_cpl_f  = f_cpl_fine(z_f, wa_CPL)
f_poly_f = f_poly_fine(z_f, G0_CP)

E_L   = E_model(z_f, Om_L,   np.ones_like(z_f))
E_CL  = E_model(z_f, Om_CL,  f_log_f)
E_CPL = E_model(z_f, Om_CPL, f_cpl_f)
E_CP  = E_model(z_f, Om_CP,  f_poly_f)

# D_M/r_d on fine grid
def DM_fine(z_arr, E_arr, th):
    dz_ = z_arr[1]-z_arr[0]
    inv_E = 1/E_arr
    cum = np.zeros_like(z_arr)
    cum[1:] = np.cumsum(0.5*(inv_E[:-1]+inv_E[1:])*dz_)
    return cum / th

DM_L   = DM_fine(z_f, E_L,   th_L)
DM_CL  = DM_fine(z_f, E_CL,  th_CL)
DM_CPL = DM_fine(z_f, E_CPL, th_CPL)
DM_CP  = DM_fine(z_f, E_CP,  th_CP)

DH_L   = 1/(th_L   * E_L)
DH_CL  = 1/(th_CL  * E_CL)
DH_CPL = 1/(th_CPL * E_CPL)
DH_CP  = 1/(th_CP  * E_CP)

# DESI data for plotting
z_data = z_eff[1:]    # exclude BGS (DV only)
DM_dat = np.array([d[2] for d in DESI[1:]])
sDM    = np.array([d[3] for d in DESI[1:]])
DH_dat = np.array([d[4] for d in DESI[1:]])
sDH    = np.array([d[5] for d in DESI[1:]])

# Figure setup
BG  = '#080808'; TXT = '#dddddd'
C_L   = '#888888'   # LCDM
C_CL  = '#a5d6a7'   # cascade log (green)
C_CPL = '#ffd54f'   # CPL (yellow)
C_CP  = '#4fc3f7'   # cascade poly (cyan)
C_DAT = '#ef9a9a'   # DESI data (red)

fig = plt.figure(figsize=(16, 11), facecolor=BG)
gs  = gridspec.GridSpec(2, 3, figure=fig,
                        hspace=0.45, wspace=0.38,
                        left=0.07, right=0.97, top=0.93, bottom=0.08)
ax_w  = fig.add_subplot(gs[0, 0])
ax_DM = fig.add_subplot(gs[0, 1])
ax_DH = fig.add_subplot(gs[0, 2])
ax_chi= fig.add_subplot(gs[1, 0])
ax_fde= fig.add_subplot(gs[1, 1])
ax_res= fig.add_subplot(gs[1, 2])

def style(ax):
    ax.set_facecolor('#111111')
    ax.tick_params(colors=TXT, labelsize=9)
    ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
    ax.title.set_color(TXT)
    for sp in ax.spines.values(): sp.set_color('#2a2a2a')

for ax in (ax_w, ax_DM, ax_DH, ax_chi, ax_fde, ax_res): style(ax)

# ── w(z) ──────────────────────────────────────────────────────────
ax_w.axhline(-1, color='#444', lw=0.8, ls=':', label='Lambda = -1')
ax_w.plot(z_plt, w_lcdm, color=C_L,   lw=1.5, ls=':')
ax_w.plot(z_plt, w_log,  color=C_CL,  lw=2.2, label=f'CAS_LOG  (no free param)')
ax_w.plot(z_plt, w_cpl,  color=C_CPL, lw=2.0, ls='--', label=f'CPL  wa={wa_CPL:.2f}')
ax_w.plot(z_plt, w_poly, color=C_CP,  lw=1.8, ls='-.', label=f'CAS_POLY  G0={G0_CP:.1f}')
ax_w.set_xlim(0, 3); ax_w.set_ylim(-1.6, -0.5)
ax_w.set_xlabel('z'); ax_w.set_ylabel('w(z)')
ax_w.set_title('Dark energy equation of state', fontsize=9)
ax_w.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TXT)

# ── D_M/r_d ───────────────────────────────────────────────────────
ax_DM.errorbar(z_data, DM_dat, sDM, fmt='o', color=C_DAT, ms=5, capsize=3,
               label='DESI DR2', zorder=6)
ax_DM.plot(z_f, DM_L,   color=C_L,   lw=1.5, ls=':')
ax_DM.plot(z_f, DM_CL,  color=C_CL,  lw=2.2, label=f'CAS_LOG  chi2={chi2_CL:.2f}')
ax_DM.plot(z_f, DM_CPL, color=C_CPL, lw=2.0, ls='--', label=f'CPL  chi2={chi2_CPL:.2f}')
ax_DM.plot(z_f, DM_CP,  color=C_CP,  lw=1.8, ls='-.', label=f'CAS_POLY  chi2={chi2_CP:.2f}')
ax_DM.set_xlim(0, 3)
ax_DM.set_xlabel('z'); ax_DM.set_ylabel('D_M / r_d')
ax_DM.set_title('Transverse comoving distance', fontsize=9)
ax_DM.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TXT)

# ── D_H/r_d ───────────────────────────────────────────────────────
ax_DH.errorbar(z_data, DH_dat, sDH, fmt='o', color=C_DAT, ms=5, capsize=3,
               label='DESI DR2', zorder=6)
ax_DH.plot(z_f, DH_L,   color=C_L,   lw=1.5, ls=':')
ax_DH.plot(z_f, DH_CL,  color=C_CL,  lw=2.2, label=f'CAS_LOG  chi2={chi2_CL:.2f}')
ax_DH.plot(z_f, DH_CPL, color=C_CPL, lw=2.0, ls='--', label=f'CPL  chi2={chi2_CPL:.2f}')
ax_DH.plot(z_f, DH_CP,  color=C_CP,  lw=1.8, ls='-.', label=f'CAS_POLY  chi2={chi2_CP:.2f}')
ax_DH.set_xlim(0, 3)
ax_DH.set_xlabel('z'); ax_DH.set_ylabel('D_H / r_d')
ax_DH.set_title('Hubble distance', fontsize=9)
ax_DH.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TXT)

# ── chi2 bar chart ─────────────────────────────────────────────────
labels = ['LCDM\n(11 dof)', 'CAS_LOG\n(11 dof)', 'CPL\n(10 dof)', 'CAS_POLY\n(10 dof)']
chi2s  = [chi2_L, chi2_CL, chi2_CPL, chi2_CP]
colors = [C_L, C_CL, C_CPL, C_CP]
bars = ax_chi.bar(labels, chi2s, color=colors, width=0.55, alpha=0.85)
for bar, c2 in zip(bars, chi2s):
    ax_chi.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{c2:.2f}', ha='center', va='bottom', color=TXT, fontsize=9)
ax_chi.set_ylabel('chi-squared')
ax_chi.set_title('Goodness of fit to DESI DR2 BAO', fontsize=9)

# ── f_DE(z) comparison ────────────────────────────────────────────
ax_fde.plot(z_f, np.ones_like(z_f), color=C_L,   lw=1.5, ls=':', label='LCDM  f=1')
ax_fde.plot(z_f, f_log_f,  color=C_CL,  lw=2.2, label='CAS_LOG')
ax_fde.plot(z_f, f_cpl_f,  color=C_CPL, lw=2.0, ls='--', label='CPL')
ax_fde.plot(z_f, f_poly_f, color=C_CP,  lw=1.8, ls='-.', label='CAS_POLY')
ax_fde.set_xlim(0, 3)
ax_fde.set_xlabel('z'); ax_fde.set_ylabel('f_DE(z)')
ax_fde.set_title('Dark energy density ratio', fontsize=9)
ax_fde.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TXT)

# ── D_H residuals vs DESI ─────────────────────────────────────────
DH_L_at   = np.interp(z_data, z_f, DH_L)
DH_CL_at  = np.interp(z_data, z_f, DH_CL)
DH_CPL_at = np.interp(z_data, z_f, DH_CPL)
DH_CP_at  = np.interp(z_data, z_f, DH_CP)

ax_res.axhline(0, color='#444', lw=0.8, ls=':')
ax_res.errorbar(z_data, np.zeros_like(z_data), sDH/DH_dat, fmt='o',
                color=C_DAT, ms=5, capsize=3, label='DESI (1 sigma)')
for vals, lbl, col, ls in [
    (DH_L_at,   'LCDM',     C_L,   ':'),
    (DH_CL_at,  'CAS_LOG',  C_CL,  '-'),
    (DH_CPL_at, 'CPL',      C_CPL, '--'),
    (DH_CP_at,  'CAS_POLY', C_CP,  '-.'),
]:
    resid = (vals - DH_dat) / DH_dat * 100
    ax_res.plot(z_data, resid, 'o-', color=col, ms=6, lw=1.5,
                ls=ls, label=lbl)

ax_res.set_xlabel('z'); ax_res.set_ylabel('Residual D_H  [%]')
ax_res.set_title('D_H residuals vs DESI data', fontsize=9)
ax_res.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TXT)

fig.suptitle('DESI DR2 BAO Fit: Cascade vs CPL vs LCDM  |  Shiv Goswami, June 2026',
             fontsize=11, color='#888', y=0.97)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'figures')
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'desi_bao_fit.png')
fig.savefig(out, dpi=160, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"\nFigure -> {out}")
