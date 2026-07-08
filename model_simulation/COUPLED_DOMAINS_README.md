# Coupled-domains simulation: two regions, one bridge

**Exploratory regime.** The theorems in the papers assume a single connected graph; they do not address what happens when the graph has two large components connected by a narrow bridge, with all initial mass in one component.

## Setup

- **Region A:** sites 0..P1-1. **Region B:** sites P1..P1+P2-1.
- **Bridge:** a single edge between the last site of A and the first site of B.
- **Initial conditions:** all X, Y, and energy (Beta) in A; B is empty.
- **Dynamics:** unchanged (interaction -> ripple -> regime -> bonds -> energy -> structure -> diffusion). Only X and Y diffuse; energy does not diffuse.

## What we observe

- In many runs, **Region B stays empty** (or receives only negligible X, Y) before A freezes: the source region burns through its excitations and energy so fast that diffusion across the single link never delivers much. Transfer is bottlenecked by the bridge width and by the non-diffusion of energy.
- If you weaken interactions and strengthen diffusion, transfer of X and Y into B can occur; the fraction that crosses depends on the ratio of diffusion speed to depletion speed.

## Significance

Running this regime shows that **connectivity and the non-diffusion of energy** can make cross-region transfer negligible. The graph topology creates an effective barrier that the cascade dynamics cannot overcome when the bridge is narrow and interactions are strong. This is a parameter-dependent observation, not a proven theorem.

## How to run

```bash
python run_coupled_domains.py --P1 50 --P2 50 --steps 10000 -o ../figures/coupled_domains.png
```

Tune `D_X`, `D_Y` and `alpha_XY`, `alpha_XX` inside the script to explore diffusion-dominated vs interaction-dominated regimes.
