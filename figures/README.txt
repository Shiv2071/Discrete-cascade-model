Figures for the thesis and Paper 2.
Generate simulation plots from model_simulation/:

  cd model_simulation
  python run_simulation.py --plot -o ../figures/diagnostics.png
  python run_bigbang.py -o ../figures/bigbang_analogue.png
  python run_cosmology.py -o ../figures/cosmology

Cosmology run produces cosmology_timeseries.png and cosmology_snapshots.png
(early distribution / Big Bang to current universe analogue).

Then compile the LaTeX; the thesis and paper reference these paths.
