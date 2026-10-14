# References

These are the papers whose methods, models or findings areaFluForecast
builds on. Each entry says where the package uses it. The accompanying paper
cites more; its full bibliography is in the
[research repository](https://github.com/CodeMateo15/GNN-Influenza-Boston).

## Models

- **Deng S, Wang S, Rangwala H, Wang L, Ning Y** (2020). Cola-GNN:
  Cross-location attention based graph neural networks for long-term ILI
  prediction. *Proceedings of the 29th ACM International Conference on
  Information & Knowledge Management (CIKM '20)*, 245–254.
  [doi:10.1145/3340531.3411975](https://doi.org/10.1145/3340531.3411975)<br>
  *Used for:* the `colagnn` model, reimplemented from the paper.
- **Kipf TN, Welling M** (2017). Semi-supervised classification with graph
  convolutional networks. *5th International Conference on Learning
  Representations (ICLR 2017)*.
  [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)<br>
  *Used for:* graph mixing over the area map in `gnn_st`.
- **Hochreiter S, Schmidhuber J** (1997). Long short-term memory. *Neural
  Computation* 9(8):1735–1780.
  [doi:10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)<br>
  *Used for:* the `lstm` model.
- **Kandula S, Shaman J** (2019). Near-term forecasts of influenza-like
  illness: an evaluation of autoregressive time series approaches. *Epidemics*
  27:41–51.
  [doi:10.1016/j.epidem.2019.01.002](https://doi.org/10.1016/j.epidem.2019.01.002)<br>
  *Used for:* the `arima` baseline, and why a statistical baseline belongs in
  the comparison.

## Scoring and statistical tests

- **Bracher J, Ray EL, Gneiting T, Reich NG** (2021). Evaluating epidemic
  forecasts in an interval format. *PLoS Computational Biology*
  17(2):e1008618.
  [doi:10.1371/journal.pcbi.1008618](https://doi.org/10.1371/journal.pcbi.1008618)<br>
  *Used for:* the weighted interval score (WIS) and interval score in
  `af.scoring`.
- **Gneiting T, Raftery AE** (2007). Strictly proper scoring rules,
  prediction, and estimation. *Journal of the American Statistical
  Association* 102(477):359–378.
  [doi:10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)<br>
  *Used for:* proper scoring rules, the basis of WIS and the quantile loss.
- **Mathis SM, Webber AE, León TM, et al.** (2024). Evaluation of FluSight
  influenza forecasting in the 2021–22 and 2022–23 seasons with a new target
  laboratory-confirmed influenza hospitalizations. *Nature Communications*
  15:6289.
  [doi:10.1038/s41467-024-50601-9](https://doi.org/10.1038/s41467-024-50601-9)<br>
  *Used for:* the 23 FluSight quantiles in every forecast, and relative WIS.
- **Diebold FX, Mariano RS** (1995). Comparing predictive accuracy. *Journal
  of Business & Economic Statistics* 13(3):253–263.
  [doi:10.1080/07350015.1995.10524599](https://doi.org/10.1080/07350015.1995.10524599)<br>
  *Used for:* `af.scoring.diebold_mariano`, the test of whether a model's lead
  over persistence is bigger than week-to-week noise.
- **Harvey D, Leybourne S, Newbold P** (1997). Testing the equality of
  prediction mean squared errors. *International Journal of Forecasting*
  13(2):281–291.
  [doi:10.1016/S0169-2070(96)00719-4](https://doi.org/10.1016/S0169-2070%2896%2900719-4)<br>
  *Used for:* the small-sample correction to the Diebold-Mariano test.
- **Newey WK, West KD** (1987). A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix.
  *Econometrica* 55(3):703–708.
  [doi:10.2307/1913610](https://doi.org/10.2307/1913610)<br>
  *Used for:* the Diebold-Mariano variance, since errors *h* weeks ahead
  overlap.
- **Holm S** (1979). A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics* 6(2):65–70.
  [JSTOR 4615733](https://www.jstor.org/stable/4615733)<br>
  *Used for:* `af.scoring.holm`, the correction for comparing many models and
  horizons at once.
- **Tashman LJ** (2000). Out-of-sample tests of forecasting accuracy: an
  analysis and review. *International Journal of Forecasting* 16(4):437–450.
  [doi:10.1016/S0169-2070(00)00065-0](https://doi.org/10.1016/S0169-2070%2800%2900065-0)<br>
  *Used for:* the rolling-origin design of `af.backtest`.

## Data, features and the area graph

- **Olson DR, Heffernan RT, Paladini M, Konty K, Weiss D, Mostashari F**
  (2007). Monitoring the impact of influenza by age: emergency department fever
  and respiratory complaint surveillance in New York City. *PLoS Medicine*
  4(8):e247.
  [doi:10.1371/journal.pmed.0040247](https://doi.org/10.1371/journal.pmed.0040247)<br>
  *Used for:* emergency-department influenza-like illness as the forecast
  target, as in the Boston sample.
- **Shaman J, Kohn M** (2009). Absolute humidity modulates influenza
  survival, transmission, and seasonality. *Proceedings of the National
  Academy of Sciences* 106(9):3243–3248.
  [doi:10.1073/pnas.0806852106](https://doi.org/10.1073/pnas.0806852106)<br>
  *Used for:* weather features (`City.fetch_weather`).
- **Lowen AC, Steel J** (2014). Roles of humidity and temperature in shaping
  influenza seasonality. *Journal of Virology* 88(14):7692–7695.
  [doi:10.1128/JVI.03544-13](https://doi.org/10.1128/JVI.03544-13)<br>
  *Used for:* weather features (`City.fetch_weather`).
- **Viboud C, Bjørnstad ON, Smith DL, Simonsen L, Miller MA, Grenfell BT**
  (2006). Synchrony, waves, and spatial hierarchies in the spread of
  influenza. *Science* 312(5772):447–451.
  [doi:10.1126/science.1125237](https://doi.org/10.1126/science.1125237)<br>
  *Used for:* correlation edges, which link areas whose curves move together
  even when they don't share a border.
- **Morris M, Wheeler-Martin K, Simpson D, Mooney SJ, Gelman A, DiMaggio C**
  (2019). Bayesian hierarchical spatial models: implementing the Besag York
  Mollié model in Stan. *Spatial and Spatio-temporal Epidemiology* 31:100301.
  [doi:10.1016/j.sste.2019.100301](https://doi.org/10.1016/j.sste.2019.100301)<br>
  *Used for:* building the neighbour graph from boundary files
  (`af.adjacency`).

## Transfer and prior work

- **Panagopoulos G, Nikolentzos G, Vazirgiannis M** (2021). Transfer graph
  neural networks for pandemic forecasting. *Proceedings of the AAAI
  Conference on Artificial Intelligence* 35(6):4838–4845.
  [doi:10.1609/aaai.v35i6.16616](https://doi.org/10.1609/aaai.v35i6.16616)<br>
  *Used for:* transferring a graph model between cities (`af.pretrained`,
  `init_from`).
- **Yang W, Olson DR, Shaman J** (2016). Forecasting influenza outbreaks in
  boroughs and neighborhoods of New York City. *PLoS Computational Biology*
  12(11):e1005201.
  [doi:10.1371/journal.pcbi.1005201](https://doi.org/10.1371/journal.pcbi.1005201)<br>
  The closest prior work on neighborhood-scale influenza forecasting.
