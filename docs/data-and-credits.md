# Data and credits

## Bundled data

- **Boston sample** (`af.load_sample("boston")`): weekly emergency-department
  influenza-like-illness rates for 14 neighborhoods, from the Boston Public
  Health Commission's public dashboard. It also includes the borders between
  neighborhoods and 7 anchor nodes for surrounding towns and water.
- **Chicago sample** (`af.load_sample("chicago")`): weekly flu activity
  levels (1-10, not rates per 100,000) for 58 ZIP codes plus one anchor ZIP,
  2019-2026, from the Chicago Department of Public Health's *Influenza Risk
  Level by ZIP Code* (City of Chicago Data Portal), with borders derived from
  the City's *Boundaries - ZIP Codes*. The data are provisional, per the City's
  terms; the City of Chicago makes no claims as to their content, accuracy,
  timeliness or completeness, and they are used at one's own risk.
- **Weather** is by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0).
- **Nothing else is bundled.** The package contains no data from any other
  health department.

## Data used in the examples

- **Riverton and Lakeside** (examples 2 and 3) are synthetic: no real place,
  no real data.
- **Chicago** (example 5) is downloaded fresh from the City of Chicago's open
  data portal:
    - *Influenza Risk Level by ZIP Code*, from the Chicago Department of Public
      Health and Illinois ESSENCE
    - *Boundaries - ZIP Codes*

  All Chicago data are provisional, per the City's terms.

## Data in the accompanying paper

The paper's data availability statement describes the Columbus and Buenos Aires
data:
- Columbus weekly area-level counts are public in the
  [research repository](https://github.com/CodeMateo15/GNN-Influenza-Boston).
- Columbus Public Health's visit-level extracts are available from them on
  request.

## Credits

This work was supported by the Boston Public Health Commission, Columbus Public
Health, the PEAK Trailblazer fellowship, and Prof. Ariel Hamlin (Northeastern
University).

## Citing

The repository's `CITATION.cff` has the citation details. A Zenodo DOI will be
added with the first release.
