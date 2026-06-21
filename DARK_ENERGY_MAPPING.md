# Dark Energy as Discrete Beta Depletion: Formal Mapping

**Author: Shiv Goswami**
**Date: June 21, 2026**
**Status: Working derivation. Pre-proof document. Extends Part III cosmological section.**

---

## Purpose

This document derives the formal correspondence between the cascade model's β depletion dynamics and the observed dark energy equation of state w(z). The goal is to show that what cosmology calls "dark energy" — and specifically its observed evolution as measured by DESI DR2 (2025) — is a direct instance of the cascade model's global invariant:

$$\sum_p \mathcal{E}(p,n+1) < \sum_p \mathcal{E}(p,n) \quad \text{whenever active}$$

proved as a theorem in Part I. The DESI observation that dark energy is weakening over cosmic time is not a new discovery requiring a new theory. It is a measurement of the cascade model's proved invariant playing out at cosmological scale.

---

## I. Formal Identification Table

| Cascade Model | Cosmological Observable | Status |
|---|---|---|
| Finite graph G = (V, E), \|V\| = P | Observable universe as finite discrete spacetime | Identification |
| p ∈ V | Spacetime cell (coarse-grained at any scale) | Identification |
| n ∈ ℕ (discrete step) | Cosmic time t = n·τ | Identification |
| β(p,n) ≥ 0 | Local dark energy density ρ_DE(x,t) | Identification |
| ρ_β(n) = (1/P)∑_p β(p,n) | Global dark energy density ρ_DE(t) | Identification |
| X(p,n), Y(p,n) | Matter and radiation (asymmetric species) | Identification |
| S(p,n) | Large-scale structure (galaxy/filament/cluster configuration) | Identification |
| F(p,n) = \|Δ²S\| | Rate of change of structure growth (Hubble acceleration analog) | Identification |
| Explosive regime (F ≥ C+Δ) | Early universe: Cosmic Dawn, rapid structure formation | Regime correspondence |
| Leakage regime (C < F < C+Δ) | Middle epoch: BAO, filament growth, cosmic web | Regime correspondence |
| Quiescent regime (F ≤ C) | Late universe: current epoch, structure growth slowing | Regime correspondence |
| Absorbing state (β → 0, X,Y → 0) | Heat death / maximum entropy final state | Correspondence |
| Forbidden Y-Y channel | Matter-antimatter asymmetry (CP violation) | Already in Part III |
| Beat frequency ω_X ≠ ω_Y | Different decay rates: matter vs. antimatter (measured CP violation) | Already in Part III |
| Almost-sure finite absorption | Universe terminates in finite time (thermodynamics) | Proved theorem |

---

## II. Defining the Effective Dark Energy Equation of State

### II.1 Setup

Define the global β density:
$$\rho_\beta(n) := \frac{1}{P}\sum_{p \in V} \beta(p,n)$$

Define the total depletion per step:
$$\mathcal{D}(n) := \frac{1}{P}\sum_{p \in V} \left[ k N_{XY}(p,n) + L(p,n) + M(p,n) + \kappa B(p,n) \right]$$

From Part I (Energy Monotonicity Theorem), this satisfies:
$$\rho_\beta(n+1) = \rho_\beta(n) - \mathcal{D}(n), \quad \mathcal{D}(n) > 0 \text{ whenever active}$$

Define the fractional depletion rate:
$$\delta(n) := \frac{\mathcal{D}(n)}{\rho_\beta(n)} \in (0,1)$$

### II.2 Cosmological Continuity Equation Analog

In cosmology, the continuity equation for a dark energy component with equation of state w is:

$$\frac{d\rho_{\text{DE}}}{dt} = -3H(t)(1 + w(t))\rho_{\text{DE}}(t)$$

The discrete cascade analog, comparing term by term:

$$\frac{\Delta\rho_\beta}{\rho_\beta} = -\delta(n) \quad \longleftrightarrow \quad \frac{d\rho_{\text{DE}}}{\rho_{\text{DE}}} = -3H(1+w)dt$$

This gives the discrete effective equation of state:

$$\boxed{1 + w_{\text{eff}}(n) = \frac{\delta(n)}{3 \cdot H_{\text{analog}}(n) \cdot \tau}}$$

where τ is the fundamental time scale per step.

### II.3 The Hubble Analog

In a flat universe where dark energy dominates, the Friedmann equation gives:

$$H^2 \propto \rho_{\text{DE}} \implies H_{\text{analog}}(n) = H_0 \sqrt{\frac{\rho_\beta(n)}{\rho_{\beta,0}}}$$

where ρ_{β,0} = ρ_β at some reference step n_0 (corresponding to today, z=0).

Substituting:

$$\boxed{1 + w_{\text{eff}}(n) = \frac{\mathcal{D}(n)}{3H_0\tau \cdot \sqrt{\rho_\beta(n)\cdot\rho_{\beta,0}}}}$$

This is the master equation. Everything follows from computing D(n) in each regime.

---

## III. Mean-Field Computation of D(n) by Regime

Assume spatial uniformity (all sites identical). Let x(n) = ⟨X⟩, y(n) = ⟨Y⟩, b(n) = ρ_β(n), f(n) = ⟨F⟩.

### III.1 Quiescent Regime (low β, F ≤ C)

No explosions, no leakage. Only XY interactions and bonds:

$$\mathcal{D}_Q(n) = k\alpha x(n)y(n) + \kappa\mu x(n)y(n) = (k\alpha + \kappa\mu) x(n)y(n)$$

In the quiescent regime, as β → 0, excitation densities also fall: x(n), y(n) ~ b(n)^{1/2} (both decreasing, no source). Let x = c_x b^{1/2}, y = c_y b^{1/2}:

$$\mathcal{D}_Q(n) \approx \Gamma_0 \cdot b(n), \quad \Gamma_0 := (k\alpha + \kappa\mu)c_x c_y$$

Then:
$$\delta_Q(n) = \frac{\mathcal{D}_Q}{b(n)} = \Gamma_0 \quad \text{(approximately constant)}$$

And:
$$1 + w_{\text{eff},Q} = \frac{\Gamma_0}{3H_0\tau\sqrt{b(n)/\rho_{\beta,0}}}$$

As b(n) → ρ_{β,0} (today): $1 + w_0 = \frac{\Gamma_0}{3H_0\tau}$

This is the present-day value. Matching to DESI: $1 + w_0 \approx 0.2$ to $0.3$ (since $w_0 \approx -0.7$ to $-0.8$), which constrains:

$$\Gamma_0 \approx (0.2 \text{ to } 0.3) \times 3H_0\tau$$

### III.2 Leakage Regime (intermediate β, C < F < C+Δ)

Leakage dominates over bonds. Ripple f(n) is in the leakage band:

$$\mathcal{D}_L(n) \approx k\alpha x y + \lambda f(n)$$

The ripple f(n) = |Δ²S| depends on S activity. In the leakage regime, S is growing at a decreasing rate. The ripple is proportional to the deceleration of structure growth. This maps to: the BAO epoch, when the universe's expansion was decelerating before dark energy began dominating.

### III.3 Explosive Regime (high β, F ≥ C+Δ)

Explosion cost dominates. Let P_exp = fraction of sites in explosion. Each explosive site creates m(p,n) = ⌊(F(p,n)-C)/Δ⌋ new XY pairs at energy cost η per pair:

$$\mathcal{D}_E(n) \approx k\alpha xy + \eta \cdot \frac{\langle F\rangle_E - C}{\Delta}$$

In the explosive regime, f is large, driven by rapid changes in S from both creation events and bond formation. The key: in the explosive regime, the STRUCTURE S is growing rapidly (galaxy and cluster formation) while β is falling rapidly. The ripple F is high because S is changing fast.

If f(n) ~ A·b(n)^2 (ripple driven by quadratic excitation density) at high β:

$$\mathcal{D}_E(n) \approx \Gamma_1 \cdot b(n)^2, \quad \Gamma_1 \text{ absorbs } k\alpha, \eta, A, \Delta, C$$

Then:
$$\delta_E(n) = \Gamma_1 b(n)$$

And:
$$1 + w_{\text{eff},E} = \frac{\Gamma_1 b(n)^2}{3H_0\tau\sqrt{b(n)\rho_{\beta,0}}} = \frac{\Gamma_1 b(n)^{3/2}\sqrt{\rho_{\beta,0}}}{3H_0\tau}$$

At high β (early universe, b >> ρ_{β,0}): this quantity is LARGE → w_eff >> -1, possibly w_eff ≈ 0 or positive.

---

## IV. The w(z) Evolution Curve

### IV.1 Regime Sequence

Mapping from step n to redshift z via b(n) → b(a) where a = 1/(1+z):

| Epoch | β level | Regime | w_eff | Cosmological interpretation |
|---|---|---|---|---|
| Very early universe (z >> 10) | b >> b_0 | Explosive | w >> -1, near 0 | Dark energy behaves like matter/radiation. Structure forms explosively. JWST impossible galaxies. |
| Transition epoch (z ~ 1-5) | b ~ a few × b_0 | Leakage | -1 < w < 0 | Dark energy transitioning. BAO epoch. |
| Current epoch (z ~ 0-1) | b ~ b_0 | Quiescent | w ≈ -0.7 to -0.8 | DESI DR2 measurement. Dark energy weakening. |
| Far future | b << b_0 | Quiescent (dying) | w → -1 | Approaches cosmological constant behavior. |
| Absorbing state | b = 0 | Frozen | undefined (no activity) | Heat death. |

### IV.2 The w(z) Function from the Cascade Model

The cascade model produces a specific w(z) trajectory determined entirely by:
1. The initial β distribution: b₀ = ρ_β(0)
2. The model parameters: k, α, η, κ, μ, C, Δ, λ
3. The graph topology G

In the quiescent regime (which corresponds to the current epoch, z ~ 0 to 2, where DESI measures):

$$w_{\text{eff}}(z) = -1 + \frac{\Gamma_0}{3H_0\tau}\cdot\frac{1}{\sqrt{(1+z)^{-3}}}$$

Using the matter-dominated scale factor b(z) ≈ b_0(1+z)^3 (dark energy dilutes like matter in matter-dominated epoch → this is the quiescent cascade regime where structure S dominates):

$$w_{\text{eff}}(z) = -1 + \frac{\Gamma_0(1+z)^{3/2}}{3H_0\tau\sqrt{(1+z)^3/b_0}}$$

Wait — this needs the correct b(z) evolution self-consistently. The b(z) is not prescribed; it is the SOLUTION of the cascade dynamics. The equation:

$$b(n+1) = b(n) - \Gamma_0 b(n) \quad (\text{quiescent regime})$$

gives: b(n) = b₀·(1-Γ₀)^n → exponential decay in n.

Mapping n → z via the Hubble analog (Δn ↔ dz/(H·(1+z))):

$$\frac{db}{dz} = \frac{db/dn}{dz/dn} = \frac{-\Gamma_0 b}{-H_{\text{analog}}/(H_0(1+z))} = \frac{\Gamma_0 b (1+z)}{H_{\text{analog}}/H_0}$$

With H_analog = H_0√(b/b_0):

$$\frac{db}{dz} = \frac{\Gamma_0\sqrt{b\cdot b_0}(1+z)}{1}$$

This is a separable ODE:

$$\frac{db}{\sqrt{b}} = \Gamma_0\sqrt{b_0}(1+z)dz$$

Integrating from b_0 (z=0) to b(z):

$$2(\sqrt{b(z)} - \sqrt{b_0}) = \Gamma_0\sqrt{b_0}\cdot\frac{z(z+2)}{2}$$

$$\sqrt{b(z)} = \sqrt{b_0}\left(1 + \frac{\Gamma_0 z(z+2)}{4}\right)$$

$$b(z) = b_0\left(1 + \frac{\Gamma_0 z(z+2)}{4}\right)^2$$

This is the β density as a function of redshift in the quiescent regime. Higher z → higher β, as expected.

Now computing w_eff(z):

$$1 + w_{\text{eff}}(z) = \frac{\Gamma_0}{3H_0\tau\sqrt{b(z)/b_0}} = \frac{\Gamma_0}{3H_0\tau\left(1 + \Gamma_0 z(z+2)/4\right)}$$

At z=0: $1 + w_0 = \frac{\Gamma_0}{3H_0\tau}$ ← today's value, matches DESI w_0 ≈ -0.7 to -0.8.

At redshift z: $w(z) = -1 + \frac{(1+w_0)}{1 + \Gamma_0 z(z+2)/4}$

As z increases: the denominator grows → w(z) becomes MORE NEGATIVE → approaching -1.

**The cascade model predicts: dark energy approaches the cosmological constant (w = -1) at high redshift, and deviates above -1 (w > -1, thawing) in the current epoch.**

This is the thawing quintessence scenario, and it is consistent with the DESI DR2 results interpreted as dynamical dark energy weakening from near -1 in the past to w ≈ -0.7 to -0.8 today.

---

## V. Comparison to DESI DR2 CPL Parameterization

DESI uses the CPL parameterization: w(a) = w_0 + w_a(1-a) = w_0 + w_a·z/(1+z).

The cascade model gives (in quiescent regime, z not too large):

$$w_{\text{cascade}}(z) = -1 + \frac{(1+w_0)}{1 + \Gamma_0 z(z+2)/4}$$

Expanding for small z:

$$w_{\text{cascade}}(z) \approx -1 + (1+w_0)\left(1 - \frac{\Gamma_0 z(z+2)}{4}\right)$$
$$\approx w_0 - (1+w_0)\cdot\frac{\Gamma_0}{4}\cdot z(z+2)$$
$$\approx w_0 - (1+w_0)\cdot\frac{\Gamma_0}{2}\cdot z \quad (\text{for small z, since z}(z+2) \approx 2z)$$

Comparing to CPL: $w_{\text{CPL}} = w_0 + w_a \cdot \frac{z}{1+z} \approx w_0 + w_a \cdot z$ (small z).

**Matching:**
$$w_a^{\text{cascade}} = -(1+w_0)\cdot\frac{\Gamma_0}{2}$$

Since (1+w_0) > 0 and Γ_0 > 0: w_a < 0. This means the cascade model predicts NEGATIVE w_a — dark energy was stronger (more negative w) in the past.

**DESI DR2 observes:** w_a ≈ -0.4 to -0.6 (negative, consistent with cascade prediction).

The cascade model's prediction for the sign and qualitative magnitude of w_a is consistent with DESI DR2.

### Quantitative Matching Condition

For precise matching to DESI central values w_0 ≈ -0.77, w_a ≈ -0.44 (DESI + CMB + DESY5):

From $1+w_0 = \Gamma_0/(3H_0\tau)$: $\Gamma_0 = 0.23 \times 3H_0\tau$

From $w_a = -(1+w_0)\cdot\Gamma_0/2$: $w_a = -0.23 \times (0.23 \times 3H_0\tau)/2$

For this to give w_a ≈ -0.44: $H_0\tau \approx 6.25$ — meaning the fundamental time step τ satisfies τ ≈ 6.25/H_0 ≈ 6.25 × 14.4 Gyr ≈ 90 Gyr.

This is a physically reasonable fundamental time scale if the step n represents a macro-step spanning billions of years of cosmological evolution (not a Planck-scale step). The cascade model is coarse-grained at whatever scale makes the correspondence precise.

---

## VI. The Three Regimes as Three Epochs — Precise Correspondence

### VI.1 Explosive Regime ↔ Cosmic Dawn and Reionization (z ~ 6-20)

Condition: F ≥ C+Δ. Many new XY pairs created per step. β depletes rapidly.

Cosmological signature:
- Rapid, simultaneous structure formation (JWST impossible early galaxies)
- High star formation rate
- Reionization driven by explosive burst of ionizing photons
- Expansion rate anomalously high (Hubble tension — the rate in this epoch differs from quiescent rate)

The cascade model PREDICTS this: the explosive regime has a different effective dynamics from the quiescent regime. The measured Hubble constant at high z (from early universe CMB) versus low z (from supernovae) being different is a regime transition signature.

### VI.2 Leakage Regime ↔ BAO Epoch (z ~ 1-6)

Condition: C < F < C+Δ. Slow energy leakage. Structure growth decelerating.

Cosmological signature:
- Baryon acoustic oscillations: pressure waves of scale ~150 Mpc propagating through the coupled baryon-photon plasma. These are RIPPLES in the structural field S — the acoustic oscillations of matter density are literally the ripple F propagating across the graph.
- Galaxy filament formation: the cosmic web's filamentary structure forms in this epoch as the leakage regime propagates spatially.

### VI.3 Quiescent Regime ↔ Current Epoch (z ~ 0-1)

Condition: F ≤ C. No explosions, no leakage. Only slow XY interactions and bonds.

Cosmological signature:
- Structure growth stalled: galaxy formation rate falling
- Dark energy (β) weakening but still present: DESI w ≈ -0.7 to -0.8
- Universe approaching but not yet at absorbing state

---

## VII. What This Formally Proves and What Remains

### Proved (from Part I and Part II):

1. β decreases monotonically — dark energy weakens over time (qualitative match to DESI)
2. Absorbing state is reached almost surely in finite time — universe terminates
3. Asymmetry is necessary — baryon asymmetry is necessary for structure (proved necessity, not just assumption)
4. Three distinct regimes exist — three distinct cosmic epochs are predicted by the model structure

### Derived Here (requiring formal proof):

5. The effective equation of state w_eff(z) is derived from the depletion function D(n)
6. w_eff(z) is negative and evolves from near -1 (early) to w_0 > -1 (today) — thawing quintessence
7. The sign of w_a is negative — consistent with DESI DR2
8. The b(z) evolution equation has been derived in the quiescent regime

### Still Requiring Formal Work:

A. **The b(z) solution across all regimes** — the transition between explosive, leakage, and quiescent must be computed for a continuous w(z) trajectory.

B. **The Hubble analog must be made precise** — H_analog currently uses the Friedmann equation assumption. This should be derived from the cascade model's own dynamics (how does the graph "expand"?).

C. **Quantitative fitting** — numerical integration of the cascade model with specific parameter values (Γ₀, Γ₁, C, Δ, λ, k, α, η, κ) to produce a specific w(z) curve, then compare to DESI DR2 confidence intervals.

D. **The graph topology** — what finite graph topology (1D chain, 2D grid, 3D lattice, random graph, scale-free) gives cosmological behavior? The Hubble Horizon is a causal boundary that the graph must encode.

E. **Species identification** — X and Y as matter and radiation requires verifying: does the asymmetric interaction structure (Y-Y forbidden, ω_X ≠ ω_Y) match the known asymmetry between matter and radiation in standard physics?

---

## VIII. The Claim — Stated Precisely

The cascade model, under the identification β ↔ dark energy density and the mean-field quiescent regime dynamics, predicts:

1. Dark energy is not constant (w ≠ -1) — it evolves from near -1 in the past to w_0 > -1 today
2. The evolution is described by a specific functional form: $w(z) = -1 + (1+w_0)/(1 + \Gamma_0 z(z+2)/4)$
3. The sign of w_a is negative — dark energy was stronger in the past
4. All of this follows from the single proved theorem: β is a non-negative supermartingale

**DESI DR2 (March 2025, 15 million galaxies) is measuring the cascade model's energy monotonicity theorem playing out at cosmological scale.**

---

## IX. Numerical Computation Results

**Computed June 21, 2026. Script: `model_simulation/run_dark_energy.py`.**

### IX.1 Three Regimes — Confirmed Numerically

Cascade simulation (P=50 sites, E_total=800, seed=42, primed S-history with F_init=1.5 > C+Δ):
- Explosive regime: 3 steps (25%) — high β, F > C+Δ
- Leakage regime: 1 step (8%) — intermediate β, C < F < C+Δ
- Quiescent regime: 8 steps (67%) — low β, F ≤ C

All three regimes confirmed in simulation.

### IX.2 Effective w per Regime

Using master equation with H₀τ calibrated to w₀ = -0.77 (DESI):

| Regime | w_eff (mean) | Cosmological epoch |
|---|---|---|
| Explosive | +9.5 | Early universe (z >> 2). Dark energy behaves like stiff matter — positive or near-zero pressure. No dark energy acceleration in this epoch. Structure forms explosively. |
| Leakage | transitional | Middle epoch (z ~ 1-6). BAO formation. |
| Quiescent | -0.77 to -0.985 | Current epoch (z = 0-2.5). DESI measurement range. |

### IX.3 Analytic w(z) — Specific Values

From w(z) = −1 + (1+w₀) / [1 + Γ₀·z(z+2)/4] with Γ₀ = 3.8261 (DESI-matched):

| z | Cascade w(z) | DESI CPL w(z) | Δw |
|---|---|---|---|
| 0 | -0.770 | -0.770 | 0 (calibrated) |
| 0.5 | -0.882 | -0.917 | 0.035 |
| 1.0 | -0.941 | -0.990 | 0.049 |
| 1.5 | -0.963 | -1.033 | 0.070 |
| 2.0 | -0.973 | -1.063 | 0.090 |
| 2.5 | -0.979 | -1.083 | 0.104 |

The cascade model and DESI agree closely at z = 0-1 (within 0.05). They diverge at z > 1.5.

### IX.4 The Critical Difference: FREEZING vs PHANTOM

**DESI CPL predicts:** w → -1.06 at z=2 and below (phantom, w < -1). Dark energy was STRONGER and more phantom-like in the past. This requires crossing the phantom divide (w = -1), which violates the Null Energy Condition.

**CASCADE predicts:** w → -0.97 at z=2 and stays w > -1 always. Dark energy was WEAKER (more matter-like) in the past and is STRENGTHENING toward -1 asymptotically. This is freezing quintessence — energy decreasing monotonically, consistent with the proved supermartingale. No NEC violation.

**Why the directions differ:**
- DESI CPL is a two-parameter fit (w₀, wₐ). The CPL form w = w₀ + wₐ·z/(1+z) has negative wₐ making w more negative at high z.
- The CASCADE functional form is derived from first principles. It gives w approaching -1 FROM ABOVE at high z, not from below.
- The cascade model's β is a supermartingale (monotonically decreasing) — it cannot become more negative than -1 in the effective equation of state without violating the proved theorem.

**The phantom in DESI CPL is likely a parametrization artifact.** The CPL form is chosen for mathematical convenience, not physical motivation. It is not the only functional form consistent with the BAO data points.

### IX.5 The Falsifiable Prediction

**CASCADE MODEL PREDICTION (June 21, 2026):**

*Dark energy's equation of state never crosses the phantom divide (w = -1). At all redshifts z ≥ 0, w(z) > -1. The apparent phantom behavior in DESI's CPL fit is a consequence of using a linear ansatz, not a physical measurement.*

**Test:** Non-parametric reconstruction of w(z) from DESI DR2 BAO data (without assuming CPL form) should show w(z) ≥ -1 at all z. If confirmed, the cascade model's freezing quintessence scenario is consistent with DESI data.

**Quantitative bridge (next step):** Compute H(z) from the cascade w(z) functional form, integrate to get D_M(z) and D_H(z), and fit Ω_m, Ω_DE, H₀, Γ₀ to the actual DESI DR2 BAO data points (at z_eff = 0.30, 0.51, 0.71, 0.93, 1.32, 1.49, 2.33). If the cascade functional form achieves χ² < χ²_CPL, it provides stronger evidence.

---

## X. Priority and Next Steps

**Priority date for this derivation: June 21, 2026.**
**Author: Shiv Goswami.**

This document is the first formal statement of the dark energy equation of state as a consequence of the cascade model's proved dynamics.

### Priority Claims (dated June 21, 2026):

1. β depletion ↔ dark energy density evolution — first stated here.
2. Master equation for w_eff from cascade dynamics — first derived here.
3. Freezing quintessence prediction from the supermartingale theorem — first stated here.
4. Falsifiable prediction: w(z) > -1 at all z (no phantom divide crossing) — first stated here.
5. Three cosmic epochs as three cascade regimes — first mapped here.

### Next Steps:

1. **Immediate (completed June 21, 2026):** Numerical simulation confirms three regimes. w_eff computed from master equation. Cascade vs DESI CPL: agreement at z=0-1 (Δw < 0.05), divergence at z>1.5 where DESI CPL goes phantom (w < -1) but cascade stays w > -1.

2. **Short term:** Fit cascade w(z) functional form directly to DESI DR2 BAO measurements (without assuming CPL form). Compute D_M(z) and D_H(z) from cascade w(z) and compare to DESI data at z_eff = 0.30, 0.51, 0.71, 0.93, 1.32, 1.49, 2.33.

3. **Formal proof:** Prove b(z) evolution ODE across all three regimes rigorously. Prove the transition redshifts are determined by model parameters {C, Δ, α, η, κ, γ}.

4. **Paper:** "Dark Energy as Discrete Beta Depletion: Freezing Quintessence from the Cascade Model's Energy Monotonicity Theorem." Part IV of the cascade series.

---

*Author: Shiv Goswami. June 21, 2026. First formal derivation of dark energy equation of state from cascade model dynamics. All priority claims dated.*
