# Radical simulation: two universes, one bridge

**Unproven regime.** The theorems in the papers assume a single connected graph; they do not address what happens when the graph has two large components (two "universes") connected by a narrow bridge, with all initial mass in one component.

## Setup

- **Universe A:** sites 0..P1−1. **Universe B:** sites P1..P1+P2−1.
- **Bridge:** a single edge between the last site of A and the first site of B.
- **Initial conditions:** all X, Y, and energy (Beta) in A; B is empty.
- **Dynamics:** unchanged (interaction → ripple → regime → bonds → energy → structure → **diffusion**). Only X and Y diffuse; **energy does not diffuse**.

## What we observe

- In many runs, **Universe B stays empty** (or gets only negligible X, Y) before A freezes: the source region burns through its excitations and energy so fast that diffusion across the single link never delivers much. So **transfer is bottlenecked by the bridge and by the fact that energy is non-diffusive**.
- If you weaken interactions and strengthen diffusion, transfer of X and Y into B can occur; the fraction that “crosses” then depends on the ratio of diffusion speed to depletion speed.

## Physics translation

- **Question:** Can one “universe” pass energy or matter to another through a narrow channel?
- **In this model:** Energy (Beta) never leaves the site it started on; only excitations (X, Y) diffuse. So “energy” does not cross. Matter (X, Y) can cross only by diffusion through the bridge. Whether any significant amount crosses before the source freezes is **not proved**; the simulation shows it is **parameter-dependent** and often negligible when the bridge is a single link and interactions are strong.
- **Possible reading:** The math does not assume multiple universes or transfer; it only defines dynamics on a graph. Running this regime suggests that **connectivity and the non-diffusion of energy** can make cross-region transfer negligible—a candidate principle for “why we don’t see another universe”: not because it isn’t there, but because the channel is thin and the source runs out before much crosses.

## How to run

```bash
python run_radical.py --P1 50 --P2 50 --steps 10000 -o ../figures/radical_two_universes.png
```

Tune `D_X`, `D_Y` and `alpha_XY`, `alpha_XX` inside the script to explore diffusion-dominated vs interaction-dominated regimes.
