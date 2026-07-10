# Computational Complexity of Simulations

> **Summary:** Each simulation step scales linearly with system size ($\mathcal{O}(P)$), and total expected runtime is bounded by the initial total energy ($\mathcal{O}(\mathcal{E}_{\mathrm{tot}}(0) \cdot P)$).

The computational exploration of the discrete cascade model relies on the efficient simulation of the update rules defined in Part I. We analyse the space and time complexity of the core update loop with respect to the number of vertices $P$, the maximum graph degree $d$, and the initial capacity energy $\mathcal{E}_{\mathrm{tot}}(0)$.

## Space Complexity

The state space requires storing arrays for the primary variables $X$, $Y$, $\mathcal{E}$, $S$, and $B$ at each vertex. The ripple calculation $F = |\Delta^2 S|$ requires a structural history of depth 3. All intermediate variables (e.g., $N_{XY}$, $N_{XX}$, $L$, $M$) are evaluated and overwritten per step. Thus, the memory footprint scales strictly linearly with the graph size.

**Space Complexity:** $\mathcal{O}(P)$ per step. 

Storing the full history for post-run analysis requires $\mathcal{O}(N_{\text{absorb}} \cdot P)$, which scales linearly and remains tractable for large $P$ under typical conditions.

## Time Complexity (Per Step)

The update rule decomposes into local deterministic/stochastic operations and spatial transport.

1. **Local updates:** The interaction rates, Poisson sampling, ripple evaluation, regime switching, and structural/energy increments are pointwise operations. These are evaluated across all $P$ vertices simultaneously via vectorised array operations, yielding $\mathcal{O}(P)$ time complexity.
2. **Spatial transport:** Diffusion requires passing excitation counts along edges. For general graphs with bounded maximum degree $d$, this operation requires $\mathcal{O}(d \cdot P)$ time. On the 1D chain topology used in our standard simulations ($d=2$), diffusion is implemented via array shifts (e.g., `np.roll`), which maintain $\mathcal{O}(P)$ complexity. For sparse graphs with bounded degree, $d$ is constant; for dense graphs, $d$ may scale with $P$, increasing complexity accordingly.

**Time Complexity per Step:** $\mathcal{O}(d \cdot P)$.

## Total Time Complexity (Full Run)

The total execution time depends on the number of steps $N_{\text{absorb}}$ required for the system to reach the absorbing frozen state. 

As established in Part I, the expected absorption time satisfies $\mathbb{E}[\tau] \le \mathcal{E}_{\mathrm{tot}}(0)/\delta$, implying that the number of effective update steps scales as $\mathcal{O}(\mathcal{E}_{\mathrm{tot}}(0))$.

Under the finite energy constraint, the total number of effective update steps is bounded by $\mathcal{O}(\mathcal{E}_{\mathrm{tot}}(0))$, yielding an overall expected (in probability) time complexity of $\mathcal{O}(d \cdot \mathcal{E}_{\mathrm{tot}}(0) \cdot P)$.

## Conclusion

The simulation does not exhibit combinatorial or exponential scaling under the finite energy constraint. The runtime is intrinsically bounded by irreversible energy depletion, enabling efficient exploration of system behaviour even for large graphs and high-energy initial conditions.
