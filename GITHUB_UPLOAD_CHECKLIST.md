# What to upload to GitHub (excluding papers)

Use this list when creating a public repo. **Exclude:** `discrete_cascade_thesis.tex`, `asymmetry_necessity_paper2.tex` (keep them for submission/arXiv separately if you prefer).

---

## 1. Model documentation (root)

| Item | Upload? | Notes |
|------|--------|------|
| `mathematical_model.txt` | **Yes** | Full model spec, derivations, key equations, Sections I–XIV. Lets others understand the math without the LaTeX papers. |
| `extension.txt` | Optional | Conceptual/motivation notes. Safe to include if you want the Zeno/hierarchy context public. |

---

## 2. Simulation (entire folder)

| Item | Upload? | Notes |
|------|--------|------|
| `model_simulation/cascade_model.py` | **Yes** | Core dynamics; required for any run. |
| `model_simulation/run_simulation.py` | **Yes** | Standard run + diagnostics plot. |
| `model_simulation/run_bigbang.py` | **Yes** | Big Bang analogue. |
| `model_simulation/run_cosmology.py` | **Yes** | Early-universe to current run. |
| `model_simulation/run_radical.py` | **Yes** | Two universes, one bridge. |
| `model_simulation/run_quantum_bang.py` | **Yes** | Quantum fluctuation narrative run. |
| `model_simulation/requirements.txt` | **Yes** | `numpy`, `matplotlib`. |
| `model_simulation/README.md` | **Yes** | How to run, parameters, relation to papers. |
| `model_simulation/RADICAL_README.md` | **Yes** | Explains radical (two-universes) run. |
| `model_simulation/QUANTUM_BANG_README.md` | **Yes** | Explains quantum-bang run + paper principles. |

---

## 3. Figures

| Item | Upload? | Notes |
|------|--------|------|
| `figures/README.txt` | **Yes** | Instructions to regenerate figures from `model_simulation/`. |
| `figures/*.png` | Optional | Pre-generated plots (diagnostics, bigbang_analogue, cosmology_*, quantum_bang_*, radical_*). Include if you want the repo to show outputs without running; omit if you prefer “run scripts to get figures.” |

---

## 4. Repo root README (create)

Add a short `README.md` in the repo root that:

- States: discrete cascade model (two species, irreversible energy, ripple-driven regimes). Papers available elsewhere (arXiv/manuscript).
- Links to `mathematical_model.txt` for the full math.
- Links to `model_simulation/` for code and how to run.
- Mentions: Python 3, `pip install -r model_simulation/requirements.txt`, then run scripts from `model_simulation/`.

---

## 5. Do not upload (if excluding papers)

- `discrete_cascade_thesis.tex`
- `asymmetry_necessity_paper2.tex`
- Any `.bib` or LaTeX auxiliary files if you want to keep the submission private until publication.

---

## Summary

**Upload:**  
`mathematical_model.txt` + (optional) `extension.txt` + entire `model_simulation/` + `figures/README.txt` + (optional) `figures/*.png` + root `README.md`.

**Exclude:**  
The two `.tex` papers (and any LaTeX-only aux files you want to keep private).

This gives a citable, runnable repo that documents the model and reproduces the simulations without including the manuscript PDFs/source.
