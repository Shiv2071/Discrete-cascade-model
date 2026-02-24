# Discrete cascade model

Discrete stochastic cascade dynamics on finite graphs with irreversible energy depletion: model specification, derivations, and Python simulations.

**Papers** (arXiv/manuscript): *Discrete Stochastic Cascade Dynamics on Finite Graphs with Irreversible Energy Depletion* (Paper I); *Necessity of Species Asymmetry for Structural Hierarchy in Discrete Stochastic Cascade Models* (Paper II). Source for the papers is not included here.

## Contents

- **[mathematical_model.txt](mathematical_model.txt)** — Full model specification (Sections I–XIV): state, dynamics, ripple rule, regimes, key equations, and derivations.
- **[model_simulation/](model_simulation/)** — Python implementation and run scripts. See that folder’s [README](model_simulation/README.md) for usage.
- **[figures/](figures/)** — Instructions to regenerate figures; see `figures/README.txt`.

## Quick start

```bash
pip install -r model_simulation/requirements.txt
cd model_simulation
python run_simulation.py
```

Requires Python 3 with `numpy` and `matplotlib`.
