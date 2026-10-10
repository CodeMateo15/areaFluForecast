# Chicago sample

- `rates.csv`: weekly influenza-like-illness **activity level** (1 = minimal to 10 =
  very high, capped at 10) for 58 Chicago ZIP codes, plus ZIP 60827 as an anchor,
  from 2019-09-29 to 2026-09-20. The level compares a ZIP's share of
  emergency-department visits for flu-like illness with its own summer baseline;
  it is **not a rate per 100,000**, so its scale differs from the Boston sample.
  Source: Chicago Department of Public Health, "Influenza Risk Level by ZIP Code"
  (Illinois ESSENCE syndromic surveillance), City of Chicago Data Portal dataset
  `8vvr-jv2g`. Blank cells: May-September 2020, when nothing was published.
- `edges.csv`: ZIP codes that share a border, derived from the City of Chicago
  "Boundaries - ZIP Codes" layer (`unjd-c2ca`).
- `centroids.csv`: ZIP centroids (lat, lon).
- `weather.csv`: weekly weather per ZIP. Weather data by Open-Meteo.com
  (CC BY 4.0), https://open-meteo.com/.

Exported from the GNN-Influenza-Boston research repo with
`scripts/export_sample.py --city chicago` (variant "full").

Required by the City of Chicago Data Portal terms of use:

> This site provides applications using data that has been modified for use from
> its original source, www.cityofchicago.org, the official website of the City of
> Chicago. The City of Chicago makes no claims as to the content, accuracy,
> timeliness, or completeness of any of the data provided at this site. The data
> provided at this site is subject to change at any time. It is understood that the
> data provided at this site is being used at one's own risk.
