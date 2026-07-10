# Dark Energy: Seven Questions, Seven Cascade Theorems

**Author: Shiv Goswami**
**Date: June 21, 2026**
**Status: Superseded historical interpretation (10 July 2026).**

> **Correction:** The direct `beta -> rho_DE` theorem transfer described below is not the corrected model. The active implementation is one coupled DSCD+GR stochastic dynamical system (`model_simulation/DSCD_COSMOLOGY_SYSTEM_SPEC.md`) whose trajectory generates density, pressure, expansion, and BAO distances. Its conditional background relation is not a proof of intrinsic pressure, NEC, or ghost freedom. The version-1 audit returns `NO_FORECAST` for a single frozen configuration because the DSCD depletion scale is unidentifiable from compressed BAO. The version-2 forecasting layer (`dscd-forecast-v2`) supersedes that verdict: it samples a declared prior over latent DSCD states, weights realizations by the DR1+DR2 history, passes all predictive-convergence and coverage-calibration gates (`FORECAST_ELIGIBLE`), and seals thirteen DR3 credible intervals in `18_DSCD_V2_DR3_FORECAST_RECORD.md`. The text below is retained only as development history.

---

## One Identification

The analysis in this document rests on a single interpretive step:

> **The cascade model's local capacity energy β(p,n) is identified with local dark energy density ρ_DE(x,t).**

Under this identification, the proved theorems of Parts I and II translate directly into statements about dark energy. Everything below is a consequence of that identification plus the proved equations. The identification itself is not proved here; it is the hypothesis that the dark energy preprint (Zenodo: 10.5281/zenodo.20787562) tests against DESI data.

---

## Premise

Dark energy is the name cosmology gives to the observed acceleration of cosmic expansion: something with equation of state w approximately -0.77 permeates space and appears to be weakening over time. Standard cosmology parametrizes these phenomena (via the CPL form w0 + wa(1-a)) without deriving them from more fundamental principles.

The cascade model provides structural answers to all seven standard questions below. These answers do not import cosmological equations as inputs. They follow from cascade-internal dynamics, translated to cosmological language via the single identification above.

---

## The Cascade Equations (Proved in Part I)

**State at each site p, step n:**

$$\beta(p,n), \quad X(p,n), \quad Y(p,n), \quad S(p,n)$$

**Interaction rate** (stochastic, Poisson):

$$R_{XY}(p,n) = \alpha \cdot \omega_X \cdot \omega_Y \cdot X(p,n) \cdot Y(p,n)$$

**Ripple** (discrete second difference of structural state, Definition 2.6 of Part I):

$$F(p,n) = \bigl|S(p,n) - 2S(p,n-1) + S(p,n-2)\bigr|$$

**Regime thresholds** (Definition 2.7):
- Explosive: $F(p,n) \geq C + \Delta$, creates $m = \lfloor(F-C)/\Delta\rfloor$ new XY pairs at cost $M = \eta \cdot m$
- Leakage: $C < F(p,n) < C+\Delta$, leaks $L = \lambda F$
- Quiescent: $F(p,n) \leq C$, only XY interactions and bonds

**Energy update** (Equation 2.5 of Part I, the core equation):

$$\boxed{\beta(p,n+1) = \beta(p,n) - k \cdot N_{XY}(p,n) - k_{XX} \cdot N_{XX}(p,n) - L(p,n) - M(p,n) - \kappa \cdot B(p,n)}$$

**Total depletion per step** (Definition 3.1):

$$D(n) = \frac{1}{P}\sum_{p} \bigl[ k \cdot N_{XY} + k_{XX} \cdot N_{XX} + L + M + \kappa B \bigr]$$

**Energy Monotonicity Theorem (Theorem 3.1, Part I):**

$$\sum_p \beta(p,n+1) = \sum_p \beta(p,n) - D(n) \cdot P \leq \sum_p \beta(p,n) \quad \text{a.s., with strict inequality when active}$$

The total capacity energy $\mathcal{E}_\text{tot}(n)$ is a **non-negative supermartingale**.

**Finite Activity Theorem (Theorem 3.2, Part I):**

$$\sum_{n=0}^{\infty} D(n) \leq \mathcal{E}_\text{tot}(0) \quad \text{a.s.}$$

**Almost Sure Absorption (Theorem 3.3, Part I):**

There exists an a.s. finite stopping time $\tau < \infty$ such that $\Sigma(n)$ is absorbing for all $n \geq \tau$.

---

## Question 1: What IS dark energy?

**Cascade answer:** Under the identification, dark energy is $\beta(p,n)$, the local capacity energy at vertex $p$ at step $n$.

$\beta$ is not introduced to explain observations. It is a state variable in the cascade model that exists as the finite energy budget enabling future activity: every particle interaction, every bond, every structural increment. When $\beta(p,n) = 0$ at every site, nothing more can happen at $p$.

Under the identification, the global dark energy density at cascade step $n$ is:

$$\rho_\beta(n) = \frac{1}{P}\sum_p \beta(p,n)$$

---

## Question 2: Why does dark energy exist?

**Cascade answer:** It is the initial condition $\beta(p,0) = \beta_0 > 0$.

The existence of dark energy requires no deeper explanation beyond: the universe began with a finite energy budget. In the cascade model this is the parameter $\mathcal{E}_0 = \sum_p \beta(p,0) > 0$. That budget is what $\beta$ tracks.

---

## Question 3: Why does dark energy weaken over time?

**Cascade answer:** Because $D(n) > 0$ is proved whenever the system is active (Theorem 3.1(c), Part I).

From the energy update equation, whenever any interaction, leakage, explosion, or bond occurs at any site:

$$D(n) > 0 \implies \rho_\beta(n+1) < \rho_\beta(n)$$

This is a proved theorem, not a measurement. Every XY annihilation costs $k$ units of $\beta$. Every bond costs $\kappa$. Every explosion costs $\eta \cdot m$. The depletion is the accumulated cost of all activity, which is always non-negative and positive whenever the universe is active.

---

## Question 4: Why is $w_0 \approx -0.77$ today, not $-1$ exactly?

**Cascade answer (under mean-field approximation):** Because the universe is in the quiescent regime with $X, Y > 0$ still active.

In the quiescent regime ($F \leq C$), depletion comes only from XY interactions and bonds:

$$D_Q(n) = \frac{k}{P}\sum_p N_{XY}(p,n) + \frac{\kappa}{P}\sum_p B(p,n)$$

Under the mean-field approximation $\langle X \rangle \approx c_x \sqrt{\beta}$ and $\langle Y \rangle \approx c_y \sqrt{\beta}$ (scaling expected in the quiescent regime as activity winds down), the fractional depletion rate becomes approximately constant:

$$\delta_Q(n) := \frac{D_Q(n)}{\rho_\beta(n)} \approx \Gamma_0 > 0, \quad \Gamma_0 = (k\alpha\omega_X\omega_Y + \kappa\mu) \cdot c_x c_y$$

Matching this to the cosmological continuity equation $\dot\rho_{DE}/\rho_{DE} = -3H(1+w)$ via the identification gives:

$$1 + w_{\text{eff}} \approx \frac{\Gamma_0}{3 H_0 \tau} > 0 \implies w_0 > -1$$

The measured value $w_0 \approx -0.77$ constrains the microscopic cascade parameters through one equation. $w = -1$ exactly requires $\Gamma_0 = 0$, meaning zero activity everywhere, the absorbing state. DESI measures $w_0 \approx -0.77$: the universe is still active, still depleting $\beta$.

*Note: the mean-field scaling is an approximation, not a proved result of the cascade model. It is consistent with simulation but is not a theorem of Parts I or II.*

---

## Question 5: Why can $w$ never cross below $-1$?

**Cascade answer (directly proved, no approximation needed):** Because $\beta$ is a non-negative supermartingale (Theorem 3.1, Part I). This is the strongest result in this document.

If $w < -1$, the cosmological continuity equation gives $d\rho_\text{DE}/dt > 0$, i.e. dark energy density increasing. Under the identification, this would require $\beta$ to increase.

But Theorem 3.1(b) proves:

$$\beta(p,n+1) - \beta(p,n) = -[k N_{XY} + k_{XX} N_{XX} + L + M + \kappa B] \leq 0 \quad \text{a.s.}$$

Every term on the right is non-negative. $\beta$ can only decrease or stay constant. It cannot increase in any realization. Therefore:

$$\rho_\text{DE} \text{ cannot increase} \implies w \geq -1 \text{ at every step, in every realization, with probability one}$$

This is not a parametric constraint or an assumption imposed on $w$. It is a direct proved consequence of the energy update rule. The phantom divide $w = -1$ is a one-way barrier that the cascade model cannot cross: **if β is dark energy, the no-phantom constraint is a theorem, not a choice.**

DESI DR2's CPL best-fit crosses $w = -1$ at $z \approx 1.1$. The cascade model predicts this is a parametrization artifact of the CPL form (which imposes a linear w(z) ansatz). Non-parametric DESI reconstructions are consistent with $w \geq -1$ at all z, matching the cascade constraint.

---

## Question 6: Why are there three cosmic epochs?

**Cascade answer:** Because there are three dynamical regimes, determined by the ripple $F$ relative to thresholds $C$ and $C + \Delta$ (Definition 2.7, Part I).

The ripple $F(p,n) = |S(p,n) - 2S(p,n-1) + S(p,n-2)|$ measures how sharply structure is changing. From the structural update, $S$ grows whenever interactions or bonds occur. When $\beta$ is high (early universe), interactions are rapid, $S$ grows quickly, $F$ is large. Three distinct dynamical regimes arise from the $F$-threshold physics:

**Explosive regime** ($F \geq C + \Delta$): Each explosive event injects new XY pairs AND depletes $\beta$. The depletion rate per unit $\beta$ is high. Mapped to cosmology via the identification: dark energy behaves like stiff matter. This is the early universe: rapid structure formation, high star formation rate.

**Leakage regime** ($C < F < C + \Delta$): Intermediate depletion, declining $F$, transitional $w$. The BAO epoch: structure growing at a decelerating rate.

**Quiescent regime** ($F \leq C$): Slow, approximately constant depletion. Today. DESI measures $w \approx -0.77$ because the universe is in this regime.

The transitions between regimes are not tuned to match cosmic history. They emerge from a single proved fact: as $\beta$ falls, activity falls, $S$ increments shrink, $F = |\Delta^2 S|$ decreases (Remark 3.5 of Part I). The three epochs are dynamical consequences of a universe spending a finite energy budget.

---

## Question 7: How does the universe end, and why?

**Cascade answer (proved directly):** By absorption in finite time, almost surely (Theorem 3.3, Part I).

The total budget is $\mathcal{E}_0 = \sum_p \beta(p,0)$. Each event consumes at least $k_\text{min} = \min(k, \eta, \kappa) > 0$ units of $\beta$ (Theorem 3.2, Part I). Therefore:

$$\text{Total events} \leq \frac{\mathcal{E}_0}{k_\text{min}} < \infty \quad \text{a.s.}$$

After at most this many events, $\beta = 0$ or $X = Y = 0$ everywhere and $F \leq C$ everywhere: the absorbing state.

The absorbing state is the heat death of the universe, proved as a theorem, not assumed. At the absorbing state: $D = 0$, $\delta = 0$, $w = -1$ exactly. The cosmological constant is reached at the moment the universe can do nothing further. Under the identification, Lambda is not the cause of acceleration; it is the end state.

---

## Summary

| Question | Cascade answer | Source | Requires identification? | Approximation? |
|---|---|---|---|---|
| What is dark energy? | β(p,n): remaining capacity for activity | Axiom | Yes | No |
| Why does it exist? | Initial condition β₀ > 0 | Axiom | Yes | No |
| Why does it weaken? | D(n) > 0 whenever active | **Theorem 3.1, Part I** | Yes | No |
| Why is w₀ not -1? | δ_Q = Γ₀ > 0 in quiescent regime | Theorem 3.1 + mean-field | Yes | **Yes** |
| Why can't w < -1? | β cannot increase: β(n+1) ≤ β(n) a.s. | **Theorem 3.1, Part I** | Yes | No |
| Why three cosmic epochs? | Three F-threshold regimes with distinct D(n) | **Theorem 3.3 + Def 2.7, Part I** | Yes | No |
| Why does it end? | Total events ≤ E₀/k_min a.s. | **Theorem 3.3, Part I** | Yes | No |

**The strongest result is Question 5: the no-phantom constraint w ≥ -1 follows directly from the proved supermartingale theorem with no approximation and no free parameters, given only the identification β ↔ ρ_DE. Every other result follows from the same identification with varying degrees of approximation in the mean-field regime mapping.**

---

**Priority date: June 21, 2026. Author: Shiv Goswami.**

*The identification β ↔ ρ_DE is the single interpretive step. The proved theorems are those of Parts I and II of the cascade paper series. The dark energy application is tested quantitatively in the companion preprint (Zenodo: 10.5281/zenodo.20787562).*
