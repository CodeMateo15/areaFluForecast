# Installation

```bash
pip install areaFluForecast                 # once published on PyPI; until then:
pip install git+https://github.com/CodeMateo15/areaFluForecast
```

Optional extras:

```bash
pip install "areaFluForecast[plot]"   # forecast plots (matplotlib)
pip install "areaFluForecast[geo]"    # shapely, for boundary files that don't line up exactly
```

**Requirements:**
- Python 3.10 or newer.
- No GPU needed: everything runs on a normal laptop CPU.
- The install pulls in PyTorch and PyTorch Geometric. On Linux servers without
  a GPU, install the CPU build of PyTorch first; it's smaller:

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install areaFluForecast
    ```

## Check it works

The package ships with two samples: Boston (14 neighborhoods, 8 years of weekly
data) and Chicago (58 ZIP codes, 7 years of weekly flu activity levels on a 1-10
scale rather than rates, from the City's public open data). This page uses
Boston; swap in `--sample chicago` to try the other.

```bash
areaflu check --sample boston
```

```text
Boston: 14 areas, 436 weeks (2017-12-31 to 2026-05-03), flu season Oct-Mar
  graph: 23 borders between areas, 7 anchors with 21 borders
  ...
```

Then run a small end-to-end test. It uses tiny models, so it finishes in a
couple of minutes, and the numbers mean nothing:

```bash
areaflu backtest --sample boston --seasons 1 --horizons 1 --quick
```

## The command line

Everything in the Python API is also available from the shell:

```bash
areaflu check    --rates rates.csv --areas areas.geojson --id-col NAME
areaflu backtest --rates rates.csv --areas areas.geojson --id-col NAME --weather
areaflu forecast --rates rates.csv --areas areas.geojson --id-col NAME --weather
```

Or put the [`City.from_files`](api/city.md) arguments in a JSON file and pass
`--city city.json`. Run `areaflu backtest --help` for every option.
