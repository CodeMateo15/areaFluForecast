"""Write the example notebooks in examples/ (without outputs).

    python scripts/build_example_notebooks.py           # all (outputs are cleared)
    python scripts/build_example_notebooks.py 05 04     # only these
    python scripts/run_example_notebooks.py 05 04       # execute them, saving outputs

Edit the cell text here rather than in the .ipynb files, so the notebooks can be
regenerated after an API change.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SPEED_NOTE = (
    "> **Speed.** To keep this notebook to about half an hour on a laptop, every learned "
    "model below uses a 2-member ensemble (`n_seeds=2`) instead of the default 10. "
    "Small ensembles are noisier, so treat the numbers here as a demonstration. For "
    "real use, drop the `configs=SMALL` argument.")
SMALL = 'SMALL = {"gnn_st": {"n_seeds": 2}, "lstm": {"n_seeds": 2}, "colagnn": {"n_seeds": 2}}'
SETUP = """import warnings
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

import areafluforecast as af

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 140)
plt.rcParams["figure.dpi"] = 110"""


def quickstart() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# 1. Quickstart: forecasting flu in Boston's neighborhoods\n\n"
           "This notebook runs the whole workflow on the Boston data that ships with the package:\n\n"
           "1. **Look at the data.**\n"
           "2. **Backtest.** Would these models have worked on a past season? Each one is compared "
           "with *persistence*, the forecast \"next week looks like this week\".\n"
           "3. **Forecast** the weeks after the last observation, with uncertainty bands and a "
           "trust check.\n"
           "4. **Save** the results, including in the CDC FluSight format.\n\n"
           "Install first with `pip install \"areaFluForecast[plot]\"`.\n\n" + SPEED_NOTE +
           "\n\n> **What to expect.** Small, untuned ensembles understate the models. In the "
           "paper's evaluation, with equal tuning and 10 members, the graph model's WIS was 18-28% "
           "below persistence in Boston. Every model, tuned or not, runs about as far behind the "
           "curve as persistence does: its advantage is the size of the curve, not earlier "
           "turns. The backtest report's timing section shows this."),
        code(SETUP),
        md("## The data\n\n"
           "These are weekly emergency-department influenza-like-illness rates per 100,000 "
           "residents for 14 Boston neighborhoods, from the Boston Public Health Commission's "
           "public dashboard, together with which neighborhoods border each other and weekly "
           "weather. `summary()` is the first thing to check with your own data, too."),
        code('city = af.load_sample("boston")\nprint(city.summary())'),
        code('city.rates.tail()'),
        md("Blank cells are weeks BPHC suppressed because the count was small. The models skip "
           "them when scoring and never treat them as zero."),
        code('ax = city.rates.loc["2022-06":, ["Roxbury", "Fenway", "Charlestown", "Hyde Park"]].plot(figsize=(9, 3.2))\n'
             'ax.set_ylabel("ILI rate per 100,000"); ax.set_title("Four of the 14 neighborhoods");'),
        md("## Step 1: would it have worked here? (backtest)\n\n"
           "`backtest` trains every model on the weeks *before* a past season, forecasts every week "
           "of that season, and scores the forecasts. `seasons=1` uses the most recent complete "
           "season, which for this data runs June 2024 to May 2025."),
        code(SMALL + "\n\n"
             'bt = af.backtest(city, seasons=1, horizons=(1, 2), configs=SMALL, out_dir="runs")'),
        md("### Reading the report\n\n"
           "- **WIS** (weighted interval score) asks how good the whole forecast was, the "
           "uncertainty band as well as the single number. Lower is better, and it's in rate "
           "units. CDC FluSight ranks forecasts by it.\n"
           "- **Relative WIS** is the model's WIS divided by persistence's, on the same weeks. "
           "Below 1 means the model beat \"next week looks like this week\".\n"
           "- **p (Holm)** asks whether the gap could just be week-to-week noise. Below 0.05 "
           "means it is unlikely to be."),
        code("display(Markdown(bt.report(show=False)))"),
        md("## Step 2: forecast the next weeks\n\n"
           "`forecast` trains on every week whose outcome is already known, then forecasts from "
           "the last observed week. It also reads the backtest above and marks each model and "
           "horizon with how it did there. We ask for 3 weeks ahead even though the backtest only "
           "covered 1 and 2, to show what the trust check says about a horizon nobody tested."),
        code('fc = af.forecast(city, horizons=(1, 2, 3), configs=SMALL, out_dir="runs")'),
        code('fc.summary().query("model == \'gnn_st\'").head(14)'),
        code('fc.trust'),
        md("Each forecast row also carries 23 quantiles (`q0.010` ... `q0.990`), the full "
           "predictive distribution that FluSight scores."),
        code('fig = fc.plot("gnn_st", areas=["Roxbury", "Fenway", "Charlestown", "Hyde Park",\n'
             '                            "East Boston", "Dorchester"])'),
        md("## Step 3: save and share"),
        code('fc.to_csv("runs/boston_forecast.csv")\n'
             'flusight = fc.to_flusight("gnn_st")\n'
             'flusight.head()'),
        code('print("Everything this run wrote:", fc.path)\n'
             'sorted(p.name for p in fc.path.iterdir())'),
        md("## Next\n\n"
           "- **`02_your_own_data.ipynb`**: the file formats, building the map from a boundary "
           "file, and the errors you might hit.\n"
           "- **`03_short_history_transfer.ipynb`**: what to do with only two years of data.\n"
           "- **`04_reading_the_results.ipynb`**: a closer look at the backtest, and how to choose "
           "which model's forecast to use."),
    ]
    return nb


def own_data() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# 2. Using your own data\n\n"
           "The example city here, **Riverton**, is made up: 12 districts, 7 years of synthetic "
           "weekly rates, and a boundary file. It exists to show the formats. Replace the file "
           "paths with your own. `data/make_demo_cities.py` shows how the files were made.\n\n"
           "You need two things:\n\n"
           "| | what | format |\n|---|---|---|\n"
           "| 1 | weekly rates per area | CSV, long (`week, area, rate`) or wide (dates down, areas across) |\n"
           "| 2 | the map | GeoJSON boundaries **or** a CSV list of bordering pairs |\n\n" + SPEED_NOTE),
        code(SETUP + "\nimport json"),
        md("## 1. The rates file\n\n"
           "Here it's in long format, with one row per district per week. Rates are per 100,000 "
           "residents. Leave a week blank when it's missing or suppressed. Never fill it with 0."),
        code('raw = pd.read_csv("data/riverton_rates.csv")\nraw.head(8)'),
        code('print(f"{raw.district.nunique()} districts, {raw.week.nunique()} weeks, "\n'
             '      f"{raw.rate.isna().mean():.1%} blank")'),
        md("If you have **counts** rather than rates, `af.rates_from_counts` converts them using "
           "each area's population:"),
        code('counts = pd.DataFrame({"Old Town": [12, 30], "Harbor": [4, 9]},\n'
             '                      index=pd.to_datetime(["2025-12-07", "2025-12-14"]))\n'
             'af.rates_from_counts(counts, {"Old Town": 48_000, "Harbor": 21_500})'),
        md("## 2. The map\n\n"
           "This is a GeoJSON file with one feature per district. Its `district` property matches "
           "the names in the rates file. Two features have no data: **North Township** next door "
           "and **Lake Erin**. They're *anchors*, which tell the model where the city's edge is "
           "and what lies beyond it. The package works out which areas border which by itself."),
        code('geo = json.load(open("data/riverton_districts.geojson"))\n'
             '[f["properties"] for f in geo["features"]][:3] + ["..."]'),
        md("## 3. Build the city and check it\n\n"
           "Build it with `City.from_files`, then read `summary()` before training anything."),
        code('city = af.City.from_files(\n'
             '    "data/riverton_rates.csv",\n'
             '    areas="data/riverton_districts.geojson", id_col="district",   # the GeoJSON property\n'
             '    area_col="district",                                         # the rates column\n'
             '    anchors=["North Township", "Lake Erin"],\n'
             '    flu_months=(10, 11, 12, 1, 2, 3),       # Southern Hemisphere: e.g. (4, 5, 6, 7, 8, 9)\n'
             '    name="riverton")\n'
             'print(city.summary())'),
        md("**Check the map.** A wrong border here means the model gets wrong information, so "
           "plot it. Grey lines are borders between districts and dashed lines are borders to "
           "anchors."),
        code('fig, ax = plt.subplots(figsize=(6, 4.2))\n'
             'for f in geo["features"]:\n'
             '    ring = f["geometry"]["coordinates"][0]\n'
             '    is_anchor = f["properties"]["kind"] == "anchor"\n'
             '    ax.fill(*zip(*ring), alpha=0.15 if is_anchor else 0.35,\n'
             '            color="#9ecae1" if is_anchor else "#fdd49e", ec="white")\n'
             'cent = city.centroids\n'
             'for a, b in city.edges:\n'
             '    ax.plot(*zip(cent[a][::-1], cent[b][::-1]), color="grey", lw=1.2)\n'
             'for a, b in city.anchor_edges:\n'
             '    ax.plot(*zip(cent[a][::-1], cent[b][::-1]), color="grey", lw=0.8, ls="--")\n'
             'for name, (lat, lon) in cent.items():\n'
             '    ax.annotate(name, (lon, lat), ha="center", fontsize=7)\n'
             'ax.set_axis_off(); ax.set_title("Riverton: districts, borders and anchors");'),
        md("### What a mistake looks like\n\n"
           "The package refuses data it can't use and says what to fix. Here are two common ones."),
        code('try:\n'
             '    af.City(name="typo", rates=city.rates, edges=city.edges + [("Old Town", "Old Twon")])\n'
             'except af.DataError as err:\n'
             '    print("DataError:", err)'),
        code('try:\n'
             '    af.City(name="short", rates=city.rates.iloc[-60:], edges=city.edges)\n'
             'except af.DataError as err:\n'
             '    print("DataError:", err)'),
        md("> **Boundary files that don't line up.** Borders are found where polygons share "
           "vertices, which is how Census TIGER and most city open-data files are drawn. If "
           "your districts were digitised separately and leave small gaps, use "
           "`af.adjacency.adjacency_from_polygons(features, \"district\", method=\"touches\")` "
           "(needs `pip install shapely`) and pass the result as `edges=`. Check the plot "
           "afterwards."),
        md("## 4. Weather (optional)\n\n"
           "`city.fetch_weather()` downloads weekly weather at each district's centre from "
           "Open-Meteo. It's free and needs no key, and the downloads are cached. In the "
           "research behind this package, weather helped the graph model a little. Riverton is "
           "fictional, so it's left off here."),
        code('FETCH_WEATHER = False\nif FETCH_WEATHER:\n    city.fetch_weather()'),
        md("## 5. Backtest and forecast\n\n"
           "This is the same two calls as the Boston notebook. Riverton has four complete "
           "seasons, and we backtest the last two."),
        code(SMALL + '\n\nbt = af.backtest(city, seasons=2, horizons=(1, 2), configs=SMALL, out_dir="runs")'),
        code('bt.comparison[["horizon", "model", "wis", "relative_wis", "p_holm", "verdict"]]'
             '.sort_values(["horizon", "relative_wis"]).round(3)'),
        code('fc = af.forecast(city, horizons=(1, 2), configs=SMALL, out_dir="runs")\n'
             'fig = fc.plot("gnn_st", areas=city.areas[:8])'),
        md("Riverton's epidemics spread from district to district along borders, so a model that "
           "uses the map has real information to work with here. Whether your city behaves like "
           "this is exactly what the backtest tells you."),
    ]
    return nb


def transfer() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# 3. Only two years of data? Start from another city's model\n\n"
           "Learned models need history. **Lakeside** (made up, 7 districts) has only 2.3 years. "
           "The graph model's weights don't depend on how many areas a city has, so a model "
           "trained in a city with more history can be applied here, with no training at all "
           "(*zero-shot*) or as a starting point for local training.\n\n"
           "In the paper behind this package, a Boston model applied to Columbus with no "
           "Columbus training matched a model trained on Columbus itself and beat persistence. "
           "That won't hold everywhere, so here, as always, **backtest it on your own data "
           "first**.\n\n"
           "Pretrained Boston weights will be published with the package's first release. They "
           "use weather, so call `city.fetch_weather()` before using them. This notebook trains "
           "its own small source model on Riverton instead, without weather, so it runs offline.\n\n"
           + SPEED_NOTE),
        code(SETUP),
        code('lakeside = af.City.from_files("data/lakeside_rates.csv", edges="data/lakeside_edges.csv",\n'
             '                              name="lakeside")\n'
             'print(lakeside.summary())'),
        md("\"Complete seasons available: none\" means Lakeside has less than two years of "
           "history before any full season. The models can still train on what there is, but "
           "it's thin."),
        md("## 1. Train a transferable model in a city with more history\n\n"
           "`gnn_st_noadapt` is the graph model without its learned city-specific links, which is "
           "what makes it portable. `save_models=True` writes a checkpoint per horizon."),
        code('riverton = af.City.from_files(\n'
             '    "data/riverton_rates.csv", areas="data/riverton_districts.geojson", id_col="district",\n'
             '    area_col="district", anchors=["North Township", "Lake Erin"], name="riverton")\n\n'
             'source = af.resolve("gnn_st_noadapt", {"n_seeds": 2, "use_weather": False})\n'
             'trained = af.forecast(riverton, models=[source], horizons=(1,), backtest=None,\n'
             '                      save_models=True, out_dir="runs", progress=False)\n'
             'checkpoint = trained.meta["saved_models"][0]\n'
             'checkpoint'),
        md("## 2. Would it have worked in Lakeside?\n\n"
           "Backtest Lakeside's 2025-26 season (scored June 2025 to May 2026) with three "
           "approaches:\n\n"
           "- **pretrained**: the Riverton model applied as is, with no Lakeside training\n"
           "- **gnn_st_noadapt**: the same model trained only on Lakeside's short history\n"
           "- **persistence**: \"next week looks like this week\""),
        code('local = af.resolve("gnn_st_noadapt", {"n_seeds": 2, "use_weather": False})\n'
             'bt = af.backtest(lakeside, models=[local], seasons=[2025], horizons=(1,),\n'
             '                 init_from=checkpoint, zero_shot=True, out_dir="runs")'),
        code('bt.comparison[["model", "wis", "relative_wis", "p_holm", "verdict"]].round(3)'),
        md("A relative WIS below 1 means the approach beat persistence on Lakeside's own weeks. "
           "If the pretrained model does as well as, or better than, the local one, it's a "
           "reasonable choice until Lakeside has a few more seasons of its own.\n\n"
           "`zero_shot=False` instead *fine-tunes*: it starts from the Riverton weights and "
           "keeps training on Lakeside's data."),
        md("## 3. Forecast Lakeside with the pretrained model"),
        code('fc = af.forecast(lakeside, models=[], horizons=(1,), init_from=checkpoint, zero_shot=True,\n'
             '                 backtest=bt, out_dir="runs", progress=False)\n'
             'fc.summary()'),
        code('fig = fc.plot(areas=lakeside.areas)'),
    ]
    return nb


def reading_results() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# 4. Reading the results and choosing a model\n\n"
           "This notebook opens the Boston backtest saved by notebook 1. If notebook 1 hasn't "
           "been run, it runs a small backtest first.\n\n"
           "It answers four questions:\n\n"
           "1. Which model was best, and was it clearly better than persistence?\n"
           "2. Are the uncertainty bands honest?\n"
           "3. Where in the season did the models do well or badly?\n"
           "4. Which forecast should a department actually use?"),
        code(SETUP + "\nimport numpy as np\n"
             "from areafluforecast.intervals import QUANTILE_COLUMNS\n"
             "from areafluforecast.scoring import add_cell_scores, pit_values"),
        code('city = af.load_sample("boston")\n'
             'bt = af.latest_backtest(city, out_dir="runs")\n'
             'if bt is None:\n'
             '    SMALL = {"gnn_st": {"n_seeds": 2}, "lstm": {"n_seeds": 2}, "colagnn": {"n_seeds": 2}}\n'
             '    bt = af.backtest(city, seasons=1, horizons=(1, 2), configs=SMALL, out_dir="runs")\n'
             'print("backtest:", bt.path, "| seasons:", bt.meta["season_labels"])'),
        md("## 1. Who beat persistence?\n\n"
           "This bar chart shows relative WIS: the model's error over persistence's, on the same "
           "weeks. A bar left of the line beat \"next week looks like this week\"."),
        code('comp = bt.comparison.copy()\n'
             'fig, axes = plt.subplots(1, comp.horizon.nunique(), figsize=(10, 2.8), sharex=True)\n'
             'for ax, (h, part) in zip(np.atleast_1d(axes), comp.groupby("horizon")):\n'
             '    part = part.sort_values("relative_wis")\n'
             '    colors = ["#4C78A8" if v == "better than persistence" else "#BBBBBB" for v in part.verdict]\n'
             '    ax.barh(part.model, part.relative_wis, color=colors)\n'
             '    ax.axvline(1, color="black", lw=1)\n'
             '    ax.set_title(f"{h} week(s) ahead"); ax.set_xlabel("relative WIS (lower is better)")\n'
             'fig.suptitle("Blue = clearly better than persistence"); fig.tight_layout()'),
        code('comp[["horizon", "model", "relative_wis", "p_holm", "verdict"]].sort_values(["horizon", "relative_wis"]).round(3)'),
        md("## 2. Are the bands honest?\n\n"
           "A 95% band should contain the truth about 95% of the time. Much lower means the "
           "model is overconfident, and much higher means its bands are wider than they need "
           "to be."),
        code('(bt.scores.groupby(["horizon", "model"])["coverage95"].mean().unstack(0) * 100).round(1)'),
        md("A finer check is the **PIT histogram**. For each observed week it asks where the truth "
           "fell within the forecast's distribution. If the forecast is calibrated, the bars are "
           "flat. A U shape means the bands are too narrow, and a hump means they're too wide."),
        code('pred = bt.predictions.dropna(subset=["actual"])\n'
             'models = [m for m in ["gnn_st", "lstm", "persistence"] if m in set(pred.model)]\n'
             'fig, axes = plt.subplots(1, len(models), figsize=(10, 2.6), sharey=True)\n'
             'for ax, m in zip(axes, models):\n'
             '    part = pred[(pred.model == m) & (pred.horizon == 1)]\n'
             '    ax.hist(pit_values(part.actual, part[list(QUANTILE_COLUMNS)].to_numpy()), bins=10,\n'
             '            range=(0, 1), color="#4C78A8", edgecolor="white")\n'
             '    ax.set_title(f"{m}, 1 week ahead")\n'
             'fig.tight_layout()'),
        md("## 3. Where in the season?\n\n"
           "Errors are rarely spread evenly. Most of the WIS usually comes from the weeks around "
           "the peak."),
        code('scored = add_cell_scores(pred)\n'
             'weekly = scored[scored.horizon == 1].groupby(["target_date", "model"]).wis.mean().unstack()\n'
             'ax = weekly[[m for m in ["gnn_st", "lstm", "colagnn", "persistence"] if m in weekly]]\\\n'
             '    .plot(figsize=(9, 3.2))\n'
             'ax.set_ylabel("WIS, averaged over neighborhoods"); ax.set_title("1 week ahead, week by week");'),
        code('area = "Roxbury"\n'
             'fig, ax = plt.subplots(figsize=(9, 3.2))\n'
             'for m, color in [("gnn_st", "#4C78A8"), ("persistence", "#E45756")]:\n'
             '    part = pred[(pred.model == m) & (pred.horizon == 1) & (pred.area == area)].sort_values("target_date")\n'
             '    ax.fill_between(part.target_date, part.lower, part.upper, color=color, alpha=0.15)\n'
             '    ax.plot(part.target_date, part.predicted, color=color, label=f"{m} (95% band shaded)")\n'
             'ax.plot(part.target_date, part.actual, color="black", lw=1.4, label="observed")\n'
             'ax.legend(fontsize=8); ax.set_title(f"{area}, 1-week-ahead forecasts during the backtest season");'),
        md("## 3b. Is the forecast late?\n\n"
           "The timing diagnostic slides each model's forecasts earlier, one week at a time, "
           "and reports the slide that fits best. Persistence is exactly *h* weeks behind by "
           "construction. A model that matches it got its advantage from the size of the "
           "curve, not from earlier turns."),
        code('from areafluforecast.timing import describe, timing_table\n'
             'timing = bt.timing if not bt.timing.empty else timing_table(bt.predictions)\n'
             'for _, row in timing.sort_values(["horizon", "model"]).iterrows():\n'
             '    print(describe(row))'),
        md("## 4. Which forecast should we use?\n\n"
           "A simple, defensible rule: for each horizon, use the model with the lowest WIS among "
           "those **clearly better than persistence**. If none is clearly better, report "
           "persistence, or report the forecasts with a caveat. The backtest is evidence about "
           "your city, and \"none of the models helped\" is a legitimate finding."),
        code('def choose(comparison):\n'
             '    rows = []\n'
             '    for h, part in comparison.groupby("horizon"):\n'
             '        good = part[part.verdict == "better than persistence"]\n'
             '        best = good.nsmallest(1, "wis")\n'
             '        rows.append({"horizon": h,\n'
             '                     "use": best.model.iloc[0] if len(best) else "persistence",\n'
             '                     "relative_wis": round(float(best.relative_wis.iloc[0]), 3) if len(best) else 1.0})\n'
             '    return pd.DataFrame(rows)\n\n'
             'choose(bt.comparison)'),
        md("**Caveats to state with any forecast:**\n\n"
           "- **One season is one season.** Backtest two or more before relying on a model.\n"
           "- **Reporting delays aren't simulated.** If your data arrives late, a \"1-week\" "
           "forecast is really further ahead.\n"
           "- **Tuning matters.** `tune=True` picks each model's settings from 12 candidates, "
           "using only weeks before the season. It's how the paper compared models fairly, and "
           "it costs about 12× the time."),
    ]
    return nb


def chicago() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# 5. A real city from public data: Chicago\n\n"
           "This is what an analyst would do with nothing but a city's open-data portal. It "
           "downloads weekly influenza-like-illness activity by ZIP code and the ZIP boundaries, "
           "builds the city, checks it, backtests, and forecasts.\n\n"
           "**The data:** the Chicago Department of Public Health's *Influenza Risk Level by ZIP "
           "Code* ([data portal 8vvr-jv2g](https://data.cityofchicago.org/d/8vvr-jv2g)), compiled "
           "from Illinois' ESSENCE emergency-department syndromic surveillance. The boundaries come "
           "from *Boundaries - ZIP Codes* ([unjd-c2ca](https://data.cityofchicago.org/d/unjd-c2ca)). "
           "Both are public. All data are provisional, per the City's terms.\n\n"
           "> **Read this first: the target is not a rate.** Each ZIP's weekly *ILI activity "
           "level* runs from 1 to 10. It compares that ZIP's percent of ED visits for ILI with the "
           "same ZIP's summer average: level 1 is below the summer mean, and each level after "
           "that is roughly one more standard deviation above it, capped at 10. So the models "
           "forecast an activity index. Comparing against persistence still makes sense, but "
           "the numbers are levels, not cases per 100,000.\n\n"
           "> **Speed.** Chicago has 59 ZIP areas, which makes one graph-model ensemble member "
           "take about 11 minutes. This notebook uses 1 graph-model member and 2 members for the "
           "other models, and takes about 75 minutes. For real use, drop `configs=SMALL`."),
        code(SETUP + "\nimport io, json\nfrom pathlib import Path\nimport requests"),
        md("## 1. Download (cached, so reruns work offline)"),
        code('CACHE = Path("data/chicago_cache"); CACHE.mkdir(parents=True, exist_ok=True)\n\n'
             'def socrata_csv(dataset, **params):\n'
             '    """A Chicago open-data table as a DataFrame, cached on disk."""\n'
             '    path = CACHE / f"{dataset}.csv"\n'
             '    if not path.exists():\n'
             '        r = requests.get(f"https://data.cityofchicago.org/resource/{dataset}.csv",\n'
             '                         params={"$limit": 500000, **params}, timeout=300)\n'
             '        r.raise_for_status()\n'
             '        path.write_text(r.text)\n'
             '    return pd.read_csv(path)\n\n'
             'def socrata_geojson(dataset):\n'
             '    path = CACHE / f"{dataset}.geojson"\n'
             '    if not path.exists():\n'
             '        r = requests.get(f"https://data.cityofchicago.org/resource/{dataset}.geojson",\n'
             '                         params={"$limit": 1000}, timeout=300)\n'
             '        r.raise_for_status()\n'
             '        path.write_text(r.text)\n'
             '    return json.loads(path.read_text())\n\n'
             'ili = socrata_csv("8vvr-jv2g", **{"$select": "week_start,zip_code,ili_activity_level"})\n'
             'ili["week_start"] = pd.to_datetime(ili["week_start"])\n'
             'zips = socrata_geojson("unjd-c2ca")\n'
             'print(f"{len(ili):,} ZIP-weeks, {ili.zip_code.nunique()} ZIPs, "\n'
             '      f"{ili.week_start.min().date()} to {ili.week_start.max().date()}; "\n'
             '      f"{len(zips[\'features\'])} ZIP polygons")'),
        code('ili.ili_activity_level.value_counts().sort_index().rename("ZIP-weeks").to_frame().T'),
        md("Level 10 is common because the scale is capped. During a strong season, many ZIPs "
           "sit at the ceiling for weeks."),
        md("## 2. Match the two sources"),
        code('rates = ili.pivot_table(index="week_start", columns="zip_code", values="ili_activity_level")\n'
             'rates.columns = rates.columns.astype(str)\n'
             'mapped = {f["properties"]["zip"] for f in zips["features"]}\n'
             'print("ZIPs with data but no polygon (dropped):", sorted(set(rates.columns) - mapped))\n'
             'print("ZIPs with a polygon but no data:", sorted(mapped - set(rates.columns)))\n'
             'rates = rates[sorted(set(rates.columns) & mapped)]'),
        md("## 3. Build the city, add weather, and check it"),
        code('city = af.City.from_frames("chicago", rates, polygons=zips["features"], id_col="zip",\n'
             '                          label="Chicago (ILI activity level)")\n'
             'city.fetch_weather(verbose=False)       # Open-Meteo, free, cached\n'
             'print(city.summary())'),
        code('fig, ax = plt.subplots(figsize=(5.5, 7))\n'
             'for f in zips["features"]:\n'
             '    for poly in f["geometry"]["coordinates"]:\n'
             '        ax.fill(*zip(*[(x, y) for x, y, *_ in poly[0]]), color="#fdd49e", ec="white", lw=0.6)\n'
             'cent = city.centroids\n'
             'for a, b in city.edges:\n'
             '    ax.plot(*zip(cent[a][::-1], cent[b][::-1]), color="grey", lw=0.7)\n'
             'ax.set_aspect(1.35); ax.set_axis_off()\n'
             'ax.set_title(f"{city.n_areas} ZIPs, {len(city.edges)} borders found from the boundary file");'),
        code('ax = city.rates[["60623", "60629", "60614", "60640"]].loc["2023-08":].plot(figsize=(9, 3.2), lw=1.2)\n'
             'ax.set_ylabel("ILI activity level (1-10)"); ax.set_title("Four ZIPs, last three seasons");'),
        md("## 4. Would it have worked here?\n\n"
           "Backtest the last two complete seasons, 2024-25 and 2025-26, at 1 and 2 weeks ahead. "
           "The report now includes a **timing** section: how many weeks behind the curve each "
           "forecast runs, compared with persistence."),
        code('SMALL = {"gnn_st": {"n_seeds": 1}, "lstm": {"n_seeds": 2}, "colagnn": {"n_seeds": 2}}\n'
             'bt = af.backtest(city, seasons=2, horizons=(1, 2), configs=SMALL, out_dir="runs")'),
        code("display(Markdown(bt.report(show=False)))"),
        code('bt.timing.round(2)'),
        md("**How to read the timing lines.** Persistence is exactly *h* weeks behind by "
           "construction. A model that is also *h* weeks behind gets its advantage from the "
           "size of the curve, not from seeing turns earlier. In the research behind this "
           "package, that was true of every model, including the tuned 10-member ones. "
           "Forecasting the turn earlier needs inputs that move before ED visits do, such as "
           "wastewater or a shorter reporting delay. More past rates won't do it.\n\n"
           "*Chicago caveat:* the level is capped at 10, so an observed peak is often a run of "
           "10s, and \"peaks N weeks late\" picks the first week of that run. Trust the weeks-behind "
           "figure here more than the peak offset."),
        md("## 5. Forecast"),
        code('fc = af.forecast(city, horizons=(1, 2), configs=SMALL, out_dir="runs")\n'
             'fc.trust'),
        code('busiest = city.rates.iloc[-8:].mean().nlargest(8).index.tolist()\n'
             'fig = fc.plot("gnn_st", areas=busiest)'),
        code('fc.to_flusight("gnn_st").head()'),
        md("## What this example shows, and what it doesn't\n\n"
           "- **Shows:** the whole pipeline runs on a real city from two public downloads. The "
           "map comes from the boundary file, with no hand-drawn borders, and the backtest says "
           "honestly whether any model beat persistence here.\n"
           "- **Doesn't show:** forecasts of case rates. The target is an index relative to each "
           "ZIP's summer baseline. A department with its own ESSENCE access would use ILI "
           "visit counts or percentages instead, and the same code applies."),
    ]
    return nb


def main() -> None:
    import sys
    wanted = sys.argv[1:]
    EXAMPLES.mkdir(exist_ok=True)
    books = {"01_quickstart_boston.ipynb": quickstart(),
             "02_your_own_data.ipynb": own_data(),
             "03_short_history_transfer.ipynb": transfer(),
             "04_reading_the_results.ipynb": reading_results(),
             "05_real_city_chicago.ipynb": chicago()}
    for name, nb in books.items():
        if wanted and not any(name.startswith(w) for w in wanted):
            continue
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, EXAMPLES / name)
        print("wrote", name)
    old = EXAMPLES / "quickstart.ipynb"
    if old.exists():
        old.unlink()
        print("removed quickstart.ipynb (superseded by 01_quickstart_boston.ipynb)")


if __name__ == "__main__":
    main()
