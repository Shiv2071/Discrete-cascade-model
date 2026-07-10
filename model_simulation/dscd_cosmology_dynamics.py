"""Coupled expanding DSCD dynamics and flat-FLRW evolution."""

from __future__ import annotations

from dataclasses import replace
from math import ceil, floor
from typing import Sequence

import numpy as np

from dscd_cosmology_config import DSCDCosmologyConfig, ring_neighbors
from dscd_cosmology_state import (
    DSCDCosmologyState,
    DSCDCosmologyTrajectory,
    EventRequests,
    StepDiagnostics,
)


REGIME_QUIESCENT = 0
REGIME_LEAKAGE = 1
REGIME_EXPLOSIVE = 2


def pressure_from_depletion(
    rho_c: float, depletion_rate: float, expansion: float
) -> float:
    """Separately conserved FLRW pressure implied by DSCD depletion."""
    if rho_c <= 0.0 or depletion_rate < 0.0 or expansion <= 0.0:
        raise ValueError("rho_c and expansion must be positive; depletion non-negative")
    return -rho_c + depletion_rate / (3.0 * expansion)


def interval_equation_of_state(
    rho_before: float, rho_after: float, dN: float
) -> float:
    """Exact expansion-equivalent equation of state on one lattice interval."""
    if rho_before <= 0.0 or rho_after <= 0.0 or dN <= 0.0:
        raise ValueError("interval density and dN must be positive")
    return -1.0 - np.log(rho_after / rho_before) / (3.0 * dN)


def fund_count(
    requested: int,
    maximum_by_state: int,
    cost: float,
    budget: float,
) -> tuple[int, float]:
    """Fund an integer event count before committing it.

    The returned budget is exact up to floating-point roundoff; no post-debit
    clipping is used.
    """
    if requested < 0 or maximum_by_state < 0 or cost < 0.0 or budget < 0.0:
        raise ValueError("funding inputs must be non-negative")
    count = min(int(requested), int(maximum_by_state))
    if cost > 0.0:
        count = min(count, int(floor((budget + 1.0e-15) / cost)))
    debit = count * cost
    remaining = budget - debit
    if remaining < -1.0e-12:
        raise ArithmeticError("funded event debit exceeded its budget")
    return count, max(0.0, remaining)


class DSCDCosmologySystem:
    """One stochastic DSCD dark sector coupled to flat FLRW."""

    def __init__(
        self,
        config: DSCDCosmologyConfig,
        neighbors: Sequence[Sequence[int]] | None = None,
        weights: Sequence[float] | None = None,
    ) -> None:
        self.config = config.validate()
        self.neighbors = tuple(
            tuple(int(item) for item in row)
            for row in (neighbors if neighbors is not None else ring_neighbors(config.n_cells))
        )
        if len(self.neighbors) != config.n_cells:
            raise ValueError("neighbor list length must equal n_cells")
        for index, row in enumerate(self.neighbors):
            if any(item < 0 or item >= config.n_cells or item == index for item in row):
                raise ValueError("neighbors contain an invalid cell index")
            for item in row:
                if index not in self.neighbors[item]:
                    raise ValueError("transport graph must be undirected")
        raw_weights = (
            np.ones(config.n_cells, dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float)
        )
        if (
            raw_weights.shape != (config.n_cells,)
            or np.any(~np.isfinite(raw_weights))
            or np.any(raw_weights <= 0.0)
        ):
            raise ValueError("weights must be finite and positive")
        self.weights = raw_weights / np.sum(raw_weights)

    def weighted_mean(self, values: np.ndarray) -> float:
        return float(np.dot(self.weights, np.asarray(values, dtype=float)))

    def initial_state(self) -> DSCDCosmologyState:
        cfg = self.config
        size = cfg.n_cells
        return DSCDCosmologyState(
            step=0,
            N=cfg.n_start,
            rho_c=np.full(size, cfg.initial_rho_c, dtype=float),
            x=np.full(size, cfg.initial_x, dtype=float),
            y=np.full(size, cfg.initial_y, dtype=float),
            structure=np.full(size, cfg.initial_structure, dtype=float),
            structure_prev1=np.full(size, cfg.initial_structure, dtype=float),
            structure_prev2=np.full(size, cfg.initial_structure, dtype=float),
            phase=np.full(size, cfg.initial_phase, dtype=float),
            dtau_prev1=0.0,
            dtau_prev2=0.0,
        ).validate()

    def expansion(self, state: DSCDCosmologyState, dark_coefficient: float) -> float:
        cfg = self.config
        if dark_coefficient <= 0.0:
            raise ValueError("dark_coefficient must be positive")
        e2 = (
            cfg.omega_m * np.exp(-3.0 * state.N)
            + cfg.omega_r * np.exp(-4.0 * state.N)
            + dark_coefficient * self.weighted_mean(state.rho_c)
        )
        if not np.isfinite(e2) or e2 <= 0.0:
            raise ValueError("Friedmann constraint produced non-positive E^2")
        return float(np.sqrt(e2))

    def interference(self, phase: np.ndarray) -> np.ndarray:
        cfg = self.config
        return cfg.coherence_floor + (1.0 - cfg.coherence_floor) * (
            1.0 + np.cos(phase)
        ) / 2.0

    def interaction_rates(
        self, state: DSCDCosmologyState
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        xy = (
            cfg.alpha_xy
            * cfg.omega_x
            * cfg.omega_y
            * self.interference(state.phase)
            * state.x
            * state.y
        )
        xx_pairs = np.maximum(state.x * (state.x - 1.0) / 2.0, 0.0)
        xx = cfg.alpha_xx * cfg.omega_x**2 * xx_pairs
        return xy, xx

    def ripple(self, state: DSCDCosmologyState, dtau: float) -> np.ndarray:
        if state.step < 2 or state.dtau_prev1 <= 0.0 or dtau <= 0.0:
            return np.zeros(state.n_cells, dtype=float)
        previous_velocity = (
            state.structure_prev1 - state.structure_prev2
        ) / max(state.dtau_prev2, state.dtau_prev1)
        current_velocity = (
            state.structure - state.structure_prev1
        ) / state.dtau_prev1
        return (
            2.0
            * np.abs(current_velocity - previous_velocity)
            / (dtau + state.dtau_prev1)
        )

    def _transport(
        self, values: np.ndarray, diffusion: float, dtau: float
    ) -> tuple[np.ndarray, float]:
        before = np.asarray(values, dtype=float)
        if diffusion == 0.0 or dtau == 0.0 or not any(self.neighbors):
            return before.copy(), 0.0
        max_degree = max(len(row) for row in self.neighbors)
        substeps = max(1, int(ceil(diffusion * dtau * max_degree / 0.45)))
        h = dtau / substeps
        current = before.copy()
        for _ in range(substeps):
            delta = np.zeros_like(current)
            for index, row in enumerate(self.neighbors):
                delta[index] = sum(current[item] - current[index] for item in row)
            current = current + diffusion * h * delta
            if np.any(current < -1.0e-12):
                raise ArithmeticError("transport stability failure produced a negative state")
            current = np.maximum(current, 0.0)
        residual = float(np.sum(current) - np.sum(before))
        return current, residual

    def step(
        self,
        state: DSCDCosmologyState,
        dN: float,
        dark_coefficient: float,
        rng: np.random.Generator,
        requests: EventRequests | None = None,
    ) -> tuple[DSCDCosmologyState, StepDiagnostics]:
        """Advance one e-fold interval using proper-time event hazards."""
        cfg = self.config
        state = state.copy().validate()
        if dN <= 0.0:
            raise ValueError("dN must be positive")
        expansion = self.expansion(state, dark_coefficient)
        dtau = dN / expansion
        xy_rate, xx_rate = self.interaction_rates(state)
        if requests is None:
            requested_xy = rng.poisson(np.maximum(xy_rate * dtau, 0.0))
            requested_xx = rng.poisson(np.maximum(xx_rate * dtau, 0.0))
        else:
            requested_xy = np.asarray(requests.xy, dtype=int)
            requested_xx = np.asarray(requests.xx, dtype=int)
        if requested_xy.shape != (state.n_cells,) or requested_xx.shape != (
            state.n_cells,
        ):
            raise ValueError("event requests have the wrong shape")

        budget = state.rho_c.copy()
        x = state.x.copy()
        y = state.y.copy()
        funded_xy = np.zeros(state.n_cells, dtype=int)
        funded_xx = np.zeros(state.n_cells, dtype=int)
        for index in range(state.n_cells):
            count, budget[index] = fund_count(
                int(requested_xy[index]),
                min(int(floor(x[index])), int(floor(y[index]))),
                cfg.cost_xy,
                float(budget[index]),
            )
            funded_xy[index] = count
            x[index] -= count
            y[index] -= count

            count, budget[index] = fund_count(
                int(requested_xx[index]),
                int(floor(x[index])) // 2,
                cfg.cost_xx,
                float(budget[index]),
            )
            funded_xx[index] = count
            x[index] -= 2 * count

        ripple = self.ripple(state, dtau)
        regimes = np.full(state.n_cells, REGIME_QUIESCENT, dtype=int)
        regimes[ripple > cfg.ripple_threshold] = REGIME_LEAKAGE
        regimes[ripple >= cfg.ripple_threshold + cfg.ripple_margin] = (
            REGIME_EXPLOSIVE
        )

        leakage = np.zeros(state.n_cells, dtype=float)
        for index in range(state.n_cells):
            if regimes[index] != REGIME_QUIESCENT:
                requested_leak = cfg.leakage_rate * ripple[index] * dtau
                leakage[index] = min(requested_leak, budget[index])
                budget[index] -= leakage[index]

        overshoot = np.maximum(
            ripple - cfg.ripple_threshold - cfg.ripple_margin, 0.0
        )
        if requests is None:
            explosion_requests = rng.poisson(
                cfg.explosion_rate
                * overshoot
                / cfg.ripple_margin
                * dtau
            )
        else:
            explosion_requests = np.asarray(requests.explosion, dtype=int)
        funded_explosion = np.zeros(state.n_cells, dtype=int)
        for index in range(state.n_cells):
            count, budget[index] = fund_count(
                int(explosion_requests[index]),
                int(explosion_requests[index]),
                cfg.explosion_cost,
                float(budget[index]),
            )
            funded_explosion[index] = count
            x[index] += count
            y[index] += count

        scaled_temperature = ripple / max(cfg.ripple_threshold, 1.0e-15)
        bond_activation = np.maximum(cfg.bond_temperature - scaled_temperature, 0.0)
        bond_hazard = cfg.bond_rate * np.sqrt(np.maximum(x * y, 0.0)) * bond_activation
        if requests is None:
            bond_requests = rng.poisson(np.maximum(bond_hazard * dtau, 0.0))
        else:
            bond_requests = np.asarray(requests.bonds, dtype=int)
        funded_bonds = np.zeros(state.n_cells, dtype=int)
        for index in range(state.n_cells):
            count, budget[index] = fund_count(
                int(bond_requests[index]),
                min(int(floor(x[index])), int(floor(y[index]))),
                cfg.bond_cost,
                float(budget[index]),
            )
            funded_bonds[index] = count
            x[index] -= count
            y[index] -= count

        structure_new = (
            state.structure
            + cfg.gamma_xy * funded_xy
            + cfg.gamma_xx * funded_xx
            + cfg.gamma_bond * funded_bonds
        )
        x_before_transport = x.copy()
        y_before_transport = y.copy()
        x, transport_x_residual = self._transport(x, cfg.diffusion_x, dtau)
        y, transport_y_residual = self._transport(y, cfg.diffusion_y, dtau)
        if np.any(x < 0.0) or np.any(y < 0.0) or np.any(budget < 0.0):
            raise ArithmeticError("a funded update produced a negative state")

        debit = state.rho_c - budget
        if np.any(debit < -1.0e-13):
            raise ArithmeticError("resource increased during a DSCD step")
        rho_before = self.weighted_mean(state.rho_c)
        rho_after = self.weighted_mean(budget)
        depletion_rate = (rho_before - rho_after) / dtau
        pressure = pressure_from_depletion(rho_before, depletion_rate, expansion)
        w_interval = interval_equation_of_state(rho_before, rho_after, dN)
        continuity_residual = (
            (rho_after - rho_before) / dtau
            + 3.0 * expansion * (rho_before + pressure)
        )

        next_state = DSCDCosmologyState(
            step=state.step + 1,
            N=state.N + dN,
            rho_c=budget,
            x=x,
            y=y,
            structure=structure_new,
            structure_prev1=state.structure.copy(),
            structure_prev2=state.structure_prev1.copy(),
            phase=state.phase + cfg.delta_omega * dtau,
            dtau_prev1=dtau,
            dtau_prev2=state.dtau_prev1 if state.dtau_prev1 > 0.0 else dtau,
        ).validate()
        diagnostics = StepDiagnostics(
            dtau=dtau,
            expansion=expansion,
            requested_xy=requested_xy,
            requested_xx=requested_xx,
            funded_xy=funded_xy,
            funded_xx=funded_xx,
            funded_explosion=funded_explosion,
            funded_bonds=funded_bonds,
            leakage=leakage,
            debit=debit,
            ripple=ripple,
            regimes=regimes,
            pressure_c=pressure,
            w_interval=w_interval,
            continuity_residual=continuity_residual,
            transport_x_residual=float(
                np.sum(x) - np.sum(x_before_transport)
            )
            if abs(transport_x_residual) < 1.0e-10
            else transport_x_residual,
            transport_y_residual=float(
                np.sum(y) - np.sum(y_before_transport)
            )
            if abs(transport_y_residual) < 1.0e-10
            else transport_y_residual,
        )
        return next_state, diagnostics

    def is_absorbing(self, state: DSCDCosmologyState) -> bool:
        """Conservative closure check: False whenever future activity is possible."""
        cfg = self.config
        if np.any(state.rho_c <= 0.0):
            available = state.rho_c > 0.0
        else:
            available = np.ones(state.n_cells, dtype=bool)
        if cfg.alpha_xy > 0.0 and np.any((state.x >= 1.0) & (state.y >= 1.0) & available):
            return False
        if cfg.alpha_xx > 0.0 and np.any((state.x >= 2.0) & available):
            return False
        if state.step >= 2:
            probe_dtau = max(state.dtau_prev1, 1.0e-12)
            if np.any(self.ripple(state, probe_dtau) > cfg.ripple_threshold):
                return False
        if cfg.diffusion_x > 0.0 or cfg.diffusion_y > 0.0:
            if (
                cfg.alpha_xy > 0.0
                and np.sum(state.x) >= 1.0
                and np.sum(state.y) >= 1.0
                and np.any(available)
            ):
                return False
            if cfg.alpha_xx > 0.0 and np.sum(state.x) >= 2.0 and np.any(available):
                return False
        return True

    def _simulate_once(
        self, dark_coefficient: float, seed: int
    ) -> DSCDCosmologyTrajectory:
        cfg = self.config
        steps = int(ceil(-cfg.n_start / cfg.dN))
        n_grid = np.linspace(cfg.n_start, 0.0, steps + 1)
        rng = np.random.default_rng(seed)
        state = self.initial_state()

        expansion = [self.expansion(state, dark_coefficient)]
        rho = [self.weighted_mean(state.rho_c)]
        pressure = [float("nan")]
        ripple = [0.0]
        x_total = [float(np.sum(state.x))]
        y_total = [float(np.sum(state.y))]
        structure_total = [float(np.sum(state.structure))]
        depletion_rate: list[float] = []
        w_interval: list[float] = []
        regime_fractions: list[list[float]] = []
        continuity_residual: list[float] = []

        for index in range(steps):
            dN = float(n_grid[index + 1] - n_grid[index])
            state.N = float(n_grid[index])
            next_state, diagnostics = self.step(
                state, dN, dark_coefficient, rng
            )
            state = next_state
            state.N = float(n_grid[index + 1])
            expansion.append(self.expansion(state, dark_coefficient))
            rho.append(self.weighted_mean(state.rho_c))
            pressure.append(diagnostics.pressure_c)
            ripple.append(self.weighted_mean(diagnostics.ripple))
            x_total.append(float(np.sum(state.x)))
            y_total.append(float(np.sum(state.y)))
            structure_total.append(float(np.sum(state.structure)))
            depletion_rate.append(float(np.sum(self.weights * diagnostics.debit) / diagnostics.dtau))
            w_interval.append(diagnostics.w_interval)
            counts = np.bincount(diagnostics.regimes, minlength=3).astype(float)
            regime_fractions.append((counts / state.n_cells).tolist())
            continuity_residual.append(diagnostics.continuity_residual)

        expansion_array = np.asarray(expansion)
        rho_array = np.asarray(rho)
        pressure_array = np.asarray(pressure)
        if pressure_array.size > 1:
            pressure_array[0] = pressure_array[1]
        e2_expected = (
            cfg.omega_m * np.exp(-3.0 * n_grid)
            + cfg.omega_r * np.exp(-4.0 * n_grid)
            + dark_coefficient * rho_array
        )
        friedmann_residual = expansion_array**2 - e2_expected
        return DSCDCosmologyTrajectory(
            N=n_grid,
            expansion=expansion_array,
            rho_c=rho_array,
            pressure_c=pressure_array,
            depletion_rate=np.asarray(depletion_rate),
            w_interval=np.asarray(w_interval),
            ripple=np.asarray(ripple),
            x_total=np.asarray(x_total),
            y_total=np.asarray(y_total),
            structure_total=np.asarray(structure_total),
            regime_fractions=np.asarray(regime_fractions),
            continuity_residual=np.asarray(continuity_residual),
            friedmann_residual=friedmann_residual,
            dark_coefficient=float(dark_coefficient),
            closure_converged=False,
            seed=int(seed),
            metadata={
                "weights": self.weights.tolist(),
                "neighbors": [list(row) for row in self.neighbors],
                "config": cfg.to_dict(),
            },
        )

    def simulate(self, seed: int | None = None) -> DSCDCosmologyTrajectory:
        """Solve the stochastic path and present-flatness coefficient together."""
        cfg = self.config
        selected_seed = cfg.seed if seed is None else int(seed)
        coefficient = cfg.omega_c_today / cfg.initial_rho_c
        converged = False
        trajectory: DSCDCosmologyTrajectory | None = None
        for _ in range(cfg.closure_iterations):
            trajectory = self._simulate_once(coefficient, selected_seed)
            target = cfg.omega_c_today / trajectory.rho_c[-1]
            relative = abs(target - coefficient) / max(abs(target), 1.0e-15)
            if relative <= cfg.closure_tolerance:
                coefficient = target
                converged = True
                break
            coefficient = 0.5 * coefficient + 0.5 * target
        trajectory = self._simulate_once(coefficient, selected_seed)
        closure_residual = abs(
            trajectory.expansion[-1] ** 2
            - (
                cfg.omega_m
                + cfg.omega_r
                + coefficient * trajectory.rho_c[-1]
            )
        )
        converged = converged or closure_residual <= max(
            cfg.closure_tolerance, 1.0e-12
        )
        return replace(
            trajectory,
            closure_converged=converged,
            metadata={
                **trajectory.metadata,
                "present_flatness_residual": float(closure_residual),
            },
        )
