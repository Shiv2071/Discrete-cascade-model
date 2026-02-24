"""
Discrete stochastic cascade dynamics on a finite graph.
Implements the model from: Discrete Stochastic Cascade Dynamics on Finite Graphs
with Irreversible Energy Depletion (Paper 1).

State at each vertex p: X(p), Y(p), Beta(p), S(p).
Update order: interaction -> ripple -> regime (leakage/explosion) -> bonds -> energy -> structure -> diffusion.
"""

import numpy as np
from typing import Tuple, Optional


class CascadeModel:
    """
    Discrete cascade model on a graph (grid). All arrays are 1d of length P (number of vertices).
    """

    def __init__(
        self,
        P: int,
        # Graph: linear chain or 1d grid; neighbors stored in self.neighbors
        # Interaction
        alpha_XY: float = 0.1,
        alpha_XX: float = 0.05,
        omega_X: float = 1.0,
        omega_Y: float = 1.2,
        # Energy costs
        k_XY: float = 0.5,
        k_XX: float = 0.3,
        eta: float = 1.0,
        kappa: float = 0.4,
        # Regime
        C: float = 0.5,
        Delta: float = 0.3,
        lambda_: float = 0.1,  # leakage rate (lambda is reserved in Python)
        # Structure
        gamma_1: float = 0.2,
        gamma_XX: float = 0.1,
        gamma_2: float = 0.15,
        # Bond (Landau)
        a0: float = 1.0,
        b: float = 0.05,
        T_c: float = 0.8,
        # Diffusion
        D_X: float = 0.05,
        D_Y: float = 0.05,
        seed: Optional[int] = None,
        neighbors: Optional[list] = None,
    ):
        self.P = P
        self.alpha_XY = alpha_XY
        self.alpha_XX = alpha_XX
        self.omega_X = omega_X
        self.omega_Y = omega_Y
        self.k_XY = k_XY
        self.k_XX = k_XX
        self.eta = eta
        self.kappa = kappa
        self.C = C
        self.Delta = Delta
        self.lambda_ = lambda_
        self.gamma_1 = gamma_1
        self.gamma_XX = gamma_XX
        self.gamma_2 = gamma_2
        self.a0 = a0
        self.b = b
        self.T_c = T_c
        self.D_X = D_X
        self.D_Y = D_Y
        self.rng = np.random.default_rng(seed)

        # Build neighbor list: 1d chain with periodic boundary, or use custom graph
        if neighbors is not None:
            self.neighbors = list(neighbors)
            self.deg = max(len(nb) for nb in self.neighbors) if self.neighbors else 2
        else:
            self.neighbors = []
            for p in range(P):
                left = (p - 1) % P
                right = (p + 1) % P
                self.neighbors.append([left, right])
            self.deg = 2

        # State (updated in place)
        self.X = np.zeros(P, dtype=np.float64)
        self.Y = np.zeros(P, dtype=np.float64)
        self.Beta = np.zeros(P, dtype=np.float64)
        self.S = np.zeros(P, dtype=np.float64)
        # For ripple we need S at n, n-1, n-2
        self.S_prev1 = np.zeros(P, dtype=np.float64)
        self.S_prev2 = np.zeros(P, dtype=np.float64)

        # Step count
        self.n = 0

    def _rate_XY(self, p: int) -> float:
        return self.alpha_XY * self.omega_X * self.omega_Y * self.X[p] * self.Y[p]

    def _rate_XX(self, p: int) -> float:
        x = self.X[p]
        if x < 2:
            return 0.0
        return self.alpha_XX * (self.omega_X ** 2) * (x * (x - 1) / 2)

    def _ripple(self, p: int) -> float:
        """F(p,n) = |S(n) - 2*S(n-1) + S(n-2)|."""
        if self.n < 2:
            return 0.0
        return float(np.abs(self.S[p] - 2 * self.S_prev1[p] + self.S_prev2[p]))

    def step(self) -> bool:
        """
        Perform one time step. Returns True if any activity occurred (non-absorbing).
        """
        P = self.P
        X_new = np.zeros(P)
        Y_new = np.zeros(P)
        L = np.zeros(P)
        M = np.zeros(P)
        m_explosion = np.zeros(P)
        N_XY = np.zeros(P)
        N_XX = np.zeros(P)
        B = np.zeros(P)

        # --- 1. Interaction ---
        for p in range(P):
            r_xy = self._rate_XY(p)
            n_xy = self.rng.poisson(r_xy) if r_xy > 0 else 0
            n_xy = min(n_xy, int(self.X[p]), int(self.Y[p]))
            N_XY[p] = n_xy

            r_xx = self._rate_XX(p)
            n_xx = self.rng.poisson(r_xx) if r_xx > 0 else 0
            n_xx = min(n_xx, int(self.X[p]) // 2)
            N_XX[p] = n_xx

            X_new[p] = self.X[p] - n_xy - 2 * n_xx
            Y_new[p] = self.Y[p] - n_xy
            X_new[p] = max(0.0, X_new[p])
            Y_new[p] = max(0.0, Y_new[p])

        # --- 2. Ripple (use current S, S_prev1, S_prev2) ---
        F = np.array([self._ripple(p) for p in range(P)])

        # --- 3. Regime: leakage and explosion ---
        for p in range(P):
            if F[p] <= self.C:
                L[p] = 0.0
                M[p] = 0.0
            elif F[p] < self.C + self.Delta:
                L[p] = self.lambda_ * F[p]
                M[p] = 0.0
            else:
                L[p] = self.lambda_ * F[p]
                m = int((F[p] - self.C) / self.Delta)
                cost = self.eta * m
                if cost > self.Beta[p]:
                    m = int(self.Beta[p] / self.eta)
                    cost = self.eta * m
                m_explosion[p] = m
                M[p] = cost
                X_new[p] += m
                Y_new[p] += m

        # --- 4. Bond formation (Landau) ---
        for p in range(P):
            psi_sq = X_new[p] * Y_new[p]
            if psi_sq <= 0:
                continue
            psi = np.sqrt(psi_sq)
            T_eff = F[p] / self.C if self.C > 0 else 0.0
            # Landau: bond when T_eff < T_c - (2b/a0)*psi^2
            if T_eff >= self.T_c - (2 * self.b / self.a0) * psi_sq:
                continue
            alpha_L = self.a0 * (T_eff - self.T_c)
            dF_dpsi = 2 * alpha_L * psi + 4 * self.b * (psi ** 3)
            if dF_dpsi >= 0:
                continue
            max_bonds = min(
                int(psi_sq * np.abs(dF_dpsi) / self.b) if self.b > 0 else 0,
                int(self.Beta[p] / self.kappa) if self.kappa > 0 else 0,
                int(X_new[p]),
                int(Y_new[p]),
            )
            B[p] = max(0, max_bonds)
            X_new[p] -= B[p]
            Y_new[p] -= B[p]
            X_new[p] = max(0.0, X_new[p])
            Y_new[p] = max(0.0, Y_new[p])

        # --- 5. Energy update ---
        Beta_new = (
            self.Beta
            - self.k_XY * N_XY
            - self.k_XX * N_XX
            - L
            - M
            - self.kappa * B
        )
        Beta_new = np.maximum(Beta_new, 0.0)

        # --- 6. Structural update ---
        S_new = (
            self.S
            + self.gamma_1 * N_XY
            + self.gamma_XX * N_XX
            + self.gamma_2 * B
        )

        # --- 7. Diffusion (parallel update: use pre-diffusion values for all neighbors) ---
        X_pre = X_new.copy()
        Y_pre = Y_new.copy()
        for p in range(P):
            deg_p = len(self.neighbors[p])
            if deg_p == 0:
                continue
            sum_X_neigh = sum(X_pre[q] for q in self.neighbors[p])
            sum_Y_neigh = sum(Y_pre[q] for q in self.neighbors[p])
            X_new[p] = X_pre[p] + self.D_X * (sum_X_neigh - deg_p * X_pre[p])
            Y_new[p] = Y_pre[p] + self.D_Y * (sum_Y_neigh - deg_p * Y_pre[p])
        X_new = np.maximum(np.round(X_new), 0.0)
        Y_new = np.maximum(np.round(Y_new), 0.0)

        # Commit
        self.S_prev2[:] = self.S_prev1
        self.S_prev1[:] = self.S
        self.S[:] = S_new
        self.X[:] = X_new
        self.Y[:] = Y_new
        self.Beta[:] = Beta_new
        self.n += 1

        # Activity?
        total_activity = (
            np.sum(N_XY) + np.sum(N_XX) + np.sum(L) + np.sum(M) + np.sum(B)
        )
        return total_activity > 0

    def total_energy(self) -> float:
        return float(np.sum(self.Beta))

    def total_XY(self) -> Tuple[float, float]:
        return float(np.sum(self.X)), float(np.sum(self.Y))

    def total_structure(self) -> float:
        return float(np.sum(self.S))

    def mean_ripple(self) -> float:
        if self.n < 2:
            return 0.0
        F = np.array([self._ripple(p) for p in range(self.P)])
        return float(np.mean(F))

    def is_absorbing(self) -> bool:
        """Check if no further activity can occur (X*Y=0 everywhere, F<=C)."""
        if np.any(self.X * self.Y > 0):
            return False
        if np.any(self.X > 1) and self.alpha_XX > 0:
            return False
        F = np.array([self._ripple(p) for p in range(self.P)])
        if np.any(F > self.C):
            return False
        return True


def run_simulation(
    P: int = 50,
    max_steps: int = 5000,
    X0: float = 3.0,
    Y0: float = 3.0,
    E0: float = 100.0,
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[CascadeModel, dict]:
    """
    Initialize and run until absorbing or max_steps. Returns (model, diagnostics).
    """
    m = CascadeModel(P=P, seed=seed, **kwargs)
    m.X[:] = X0
    m.Y[:] = Y0
    m.Beta[:] = E0 / P  # uniform energy
    m.S[:] = 0.0
    m.S_prev1[:] = 0.0
    m.S_prev2[:] = 0.0

    history = {
        "n": [],
        "E_total": [],
        "X_total": [],
        "Y_total": [],
        "S_total": [],
        "F_avg": [],
        "active": [],
    }

    for _ in range(max_steps):
        E = m.total_energy()
        Tx, Ty = m.total_XY()
        S_tot = m.total_structure()
        F_avg = m.mean_ripple()
        active = m.step()
        history["n"].append(m.n)
        history["E_total"].append(E)
        history["X_total"].append(Tx)
        history["Y_total"].append(Ty)
        history["S_total"].append(S_tot)
        history["F_avg"].append(F_avg)
        history["active"].append(active)
        if not active or m.is_absorbing():
            break

    return m, history
