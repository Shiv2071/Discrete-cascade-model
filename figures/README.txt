Diagnostic figures for the discrete cascade model.

Each figure is a window into a different region of the system's
behaviour space. Regenerate from model_simulation/:

  cd model_simulation
  python run_simulation.py --plot -o ../figures/diagnostics.png
  python run_high_energy.py -o ../figures/high_energy_cascade.png
  python run_spatial_concentration.py -o ../figures/spatial_concentration

Output:
  diagnostics.png                          Standard dynamics (energy, X/Y counts, ripple)
  high_energy_cascade.png                  High-energy initial conditions
  spatial_concentration_timeseries.png     Spatially concentrated initial state (time series)
  spatial_concentration_snapshots.png      Spatially concentrated initial state (spatial)

Then compile the LaTeX; Part I and Part II reference these paths.
