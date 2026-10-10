# Boston sample

- `rates.csv`: weekly emergency-department influenza-like-illness rate per
  100,000 residents for 14 Boston neighborhoods, from 2017-12-31 to 2026-05-03.
  Source: Boston Public Health Commission public dashboard ("Influenza
  Neighborhood"). Blank cells are weeks BPHC suppressed for small counts.
- `edges.csv`: borders between neighborhoods (hand-coded in the research and
  checked against boundary files), plus borders to 7 anchor nodes for
  surrounding towns, the Charles River and Boston Harbor.
- `centroids.csv`: approximate neighborhood centroids (lat, lon).
- `weather.csv`: weekly weather per neighborhood. Weather data by Open-Meteo.com
  (CC BY 4.0), https://open-meteo.com/.

Exported from the GNN-Influenza-Boston research repo with
`scripts/export_sample.py`; identical to the data the paper's Boston backtest used.
