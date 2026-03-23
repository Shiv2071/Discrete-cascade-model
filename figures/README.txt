Diagnostic figures for the discrete cascade model.

Each figure is a window into a different region of the system's
behaviour space. Regenerate from model_simulation/:

  cd model_simulation
  python run_simulation.py --plot -o ../figures/diagnostics.png
  python run_bigbang.py -o ../figures/bigbang_analogue.png
  python run_cosmology.py -o ../figures/cosmology

Output:
  diagnostics.png           Standard dynamics (energy, X/Y counts, ripple)
  bigbang_analogue.png      High-energy initial conditions
  cosmology_timeseries.png  Spatially concentrated initial state (time series)
  cosmology_snapshots.png   Spatially concentrated initial state (spatial)

Then compile the LaTeX; Part I and Part II reference these paths.
