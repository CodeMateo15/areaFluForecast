# Limits to know

## The forecasts run late

Every model here, like persistence, rises after the epidemic curve rises and
peaks after it peaks. At *h* weeks ahead it runs about *h* weeks behind. The
models beat persistence by getting the **size** of the epidemic right, not by
seeing **turns** earlier.

**Why:**
- **The models only see past rates.** A turn in a weekly flu curve isn't
  visible in the data until after it happens.
- **The training loss makes being late cheap.** When the turn is uncertain, a
  smooth, slightly late forecast has lower expected error than a guess at the
  turn, so the models lean heavily on last week's level.

**What was tried in the research:**
- 25 variations of the graph model
- forecasts that follow the recent trend
- a loss that charges more for missing big moves
- a growth-rate (Rt) feature
- a median loss

None changed the lag.

**What would help** is inputs that move before emergency-department visits do,
or arrive sooner: wastewater, for example, or a shorter reporting delay.

**What you see:** the backtest report's [timing section](reading-results.md#3-is-the-forecast-late)
measures this on your own data.

**For alerts:** act on a forecast's probability of crossing a threshold, using
the quantiles, rather than on its median, which falls short of peaks.

## Reporting delays are not simulated

A 1-week forecast assumes last week's data is complete. If your data arrives *L*
weeks late, treat the *h*-week forecast as roughly an (*h* + *L*)-week one.

## The backtest saw final data

If your counts get revised after the fact, the backtest scored the models on
final numbers, and your real-time forecasts will see first-reported numbers.
Retrospective accuracy can look better than real-time accuracy.

## One season is one season

A model can beat persistence in one season and not the next. Backtest two or
more seasons before relying on it.

## What the models don't use

- No travel or mobility data.
- No vaccination data.
- No lab or strain information.

The models see your rates, the map, and optionally weather and static
characteristics.
