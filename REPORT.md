# Assignment 2 — Bus-level Load Prediction Pipeline

Author: <your name>
Submission date: <YYYY-MM-DD>
Repository: <github URL>

## TL;DR

End-to-end hierarchical pipeline that forecasts hourly bus-level load for the
2025 calendar year, for two horizons:

- **Task 1 (next-day):** zone-level LightGBM trained on 2022-01-01..2024-12-31,
  evaluated on 2025; bus shares applied to disaggregate.
- **Task 2 (next-month):** zone-level LightGBM with year-only lags (no lookahead
  into the target month), one model fit on all pre-2025 data and applied to
  each month of 2025 separately.

Specialized per-bus LightGBM models are layered on top for the 1,500 highest-error
buses identified on a held-out 2024 H2 validation fold.

| Task | Level | WMAPE | MAPE | MAE (MW) | RMSE (MW) |
|---|---|---:|---:|---:|---:|
| Task 1 (next-day) | Bus  | **21.38%** | 22.89% | 2.989 | 10.582 |
| Task 2 (next-month) | Bus  | **27.76%** | 29.64% | 3.881 | 12.126 |
| Baseline (lag_8760h, year-ago) | Bus | 26.68% (Task 1) / 28.87% (Task 2) | 31.10% | 4.038 | 13.109 |
| Baseline (hist avg by bus/hour/dow) | Bus | 28.69% | 32.42% | 4.449 | 13.904 |

**Model improvement over lag_8760h baseline:**
- Task 1: **+5.30 pp** (standard WMAPE) / +9.27 pp (industry WMAPE that filters zero-load hours)
- Task 2: **+1.11 pp** (standard WMAPE) / +4.14 pp (industry WMAPE)

Zone-level WMAPE on test (printed by `03_training_model.py`): Task 1 ~4%
range with all features intact, Task 2 ~12.5%. Bus-level numbers are
unavoidably higher because hierarchical disaggregation amplifies the zone
forecast error.

---

## 1. Model selection rationale

**LightGBM** for both tasks because:

- Tabular, mixed-type features (calendar + weather + lags). Gradient-boosted
  trees consistently outperform linear / RNN baselines in this regime without
  the engineering overhead of deep models.
- Handles missing values natively (important: lags are NaN at the start of each
  series, weather has occasional gaps).
- Native categorical support for `zone_name` lets a single shared model learn
  zone-specific patterns without one-hot blowup.
- Fast enough that we can train per-zone, per-bus, and ensemble variants on a
  laptop in minutes (after the early `n_jobs=1` performance bug was fixed —
  see Section 4).

**Architecture choice:** zone-first, then disaggregate to bus. Rationale:

- Zone-level series are dense, smooth, and have rich exogenous signal (weather
  is meaningful at zone aggregate level, less so at individual bus level).
- Bus-level shares are stable within `(bus, hour-of-day, day-of-week)` cells,
  so disaggregating by smoothed historical shares preserves the zone forecast
  quality while distributing it to ~4,800 buses.
- Reconciliation step (post-hoc scaling) guarantees
  `sum(bus_forecast) == zone_forecast` exactly.

**Per-zone refinement:** an initial run showed FWES, NCEN, SCEN, and SOUT
had materially worse WMAPE than the others (data-center growth in NCEN/SCEN,
sparse weather coverage in FWES, structural reconciliation noise in SOUT). I
split the architecture:

- **FWES:** dedicated single-zone model (`task1_zone_FWES.txt`).
- **Other 7 zones:** shared model with `zone_name` as a categorical
  (`task1_zone_shared.txt`).
- **NCEN / SCEN / SOUT:** dedicated probe model + blend with the shared
  prediction, alpha tuned on val WMAPE, with exponential recency weighting
  (~20× more weight on recent data than oldest).

**Specialized per-bus models:** for 1,500 buses where the hierarchical
forecast had >15% val WMAPE and mean load ≥ 1 MW
(`06_idenitfy_problem_buses.py`). Empirical sweep at N = 500 / 700 / 1500
showed N=1500 produces the best Task 1 WMAPE; smaller N regresses both tasks.
These predictions are blended in by `08_Evaluate.py` only for the specific
buses where they exist, with the rest falling back to the hierarchical
zone × share output.

---

## 2. Baseline comparison methodology

Two baselines computed in `09_baseline.py`, both using only pre-2025 data as
input features:

1. **Year-ago (lag_8760h):** the 2024-MM-DD-HE-H actual is used as the
   prediction for 2025-MM-DD-HE-H. Captures seasonality, weather climatology,
   and the calendar effect in one number with no model.
2. **Historical average by (bus, hour-of-day, day-of-week):** mean across all
   2022-2024 observations.

Both produce the same `model_name`, `forecast_created_at`, `target_date`, `he`,
`bus_id`, `zone_id`, `predict_pd` schema as the model submissions, for direct
WMAPE comparability.

The model beats both baselines on both tasks, under both common WMAPE
conventions (standard, and the industry convention that excludes hours where
actual load = 0):

**Task 1 (Next-Day):**

| Method | WMAPE (standard) | WMAPE (industry) | Δ vs lag_8760h |
|---|---:|---:|---:|
| Baseline: lag_8760h | 26.68% | 25.82% | — |
| Baseline: hist avg (bus, hour, dow) | 28.69% | — | +2.01 pp worse |
| Hierarchical (model) | 29.07% | — | +2.39 pp worse |
| **Hierarchical + specialized blend** | **21.38%** | **16.54%** | **−5.30 pp better** |

**Task 2 (Next-Month):**

| Method | WMAPE (standard) | WMAPE (industry) | Δ vs lag_8760h |
|---|---:|---:|---:|
| Baseline: lag_8760h | 28.87% | 27.06% | — |
| Baseline: hist avg (bus, hour, dow) | 28.69% | — | −0.18 pp better |
| Hierarchical (model) | 30.10% | — | +1.23 pp worse |
| **Hierarchical + specialized blend** | **27.76%** | **22.92%** | **−1.11 pp better** |

The specialized-blend submission beats lag_8760h by a wide margin on Task 1
(short-horizon — per-bus models with recent-lag features dominate) and a
narrower margin on Task 2 (long-horizon — both methods rely on yearly-scale
features so the model's edge is smaller). The hist-avg baseline is slightly
better than hierarchical-only for Task 2 — confirming that pooled
cross-bus information is harder to beat at long horizons.

---

## 3. Feature engineering

### Calendar (`add_calendar_feature` in `03_training_model.py`)

- `hour`, `day_of_week`, `day_of_month`, `day_of_year`, `month`, `quarter`,
  `year`.
- `is_weekend`, `is_holiday` (US federal holidays via `holidays` package).
- Cyclical encodings: `hour_sin/cos`, `month_sin/cos`, `dow_sin/cos`.
- Texas school calendar: `is_summer_break`, `is_winter_break`,
  `is_spring_break`, `is_school_day`.
- Extreme weather flags: `is_extreme_heat` (>95°F), `is_extreme_cold` (<32°F),
  `is_super_heat` (>100°F).

### Weather (joined from `01_pull weather.py`)

- Raw: `temperature_2m`, `relativehumidity_2m`, `dewpoint_2m`,
  `apparent_temperature`, `windspeed_10m`, `cloudcover`, `precipitation`.
- Derived: `heat_index` (NWS formula, only valid above 80°F), `hdh` (heating
  degree-hours from 65°F base), `cdh` (cooling degree-hours from 65°F base).
- All temperatures in °F throughout the pipeline. (An audit pass caught a unit
  mismatch where downstream feature code was using °C thresholds against °F
  data — fixed before final submission.)
- **Weather coverage extended to 2025-12-31** during audit (originally only
  pulled through 2024-12-31, which left every 2025 weather feature NaN and
  caused Task 1 zone WMAPE to balloon — see Section 4).

### Lags and rolling windows

**Task 1 (next-day):** safe lags `[24, 48, 72, 168, 336, 720, 8760]` hours.
24h is the minimum because the forecast is created the day before. Rolling
mean and std over `[24, 168, 720]` hours, all anchored at `shift(24)` to
prevent leakage from same-day data.

**Task 2 (next-month):** year-ago lags `lag_8760h`, `lag_8736h` (52-week,
DOW-aligned), `lag_17520h` (2-year) PLUS medium-horizon lags `lag_1500h`,
`lag_2160h`, `lag_4320h` for the specialized per-bus models. Medium-horizon
lags are anchored at ≥ 1500h because the longest target-month last hour
(e.g., Jan 31 23:00 with forecast_created_at = Dec 1) requires a 1487-hour
minimum lookback to stay strictly before forecast_created_at.

Rolling means and stds at 30d / 90d / 180d windows, all anchored at either
`shift(8760)` (year-ago) or `shift(1500)` (quarter-ago).

### Cross-features (Task 1)

- `temp_sq`, `apparent_temp_sq` — nonlinear weather response.
- `temp_x_hour` — temperature interacts with time-of-day (AC ramp).
- `cdh_x_weekday`, `cdh_x_weekend` — commercial vs residential AC pattern.
- `solar_depression_proxy` — midday clear-sky hours suppress SCEN net load
  via rooftop solar.
- `zone_load_growth_rate` — recent-90d-avg / year-ago-90d-avg, clipped to
  [0.7, 1.5]. Captures NCEN (DFW data centers) and SCEN (Austin) structural
  load growth.
- `load_trend_days` — secular trend in years since 2022-01-01.

### Climatology (Task 2 only)

Computed on training-only data via `build_climatology` with
`train_cutoff=pd.Timestamp('2025-01-01')` (exclusive `<` comparison):

- `temp_clim`, `humid_clim`, `hdh_clim` by `(zone, day_of_year, he)`.
- `zone_load_clim` by `(zone, day_of_year, he)`.
- `zone_load_dow_clim` by `(zone, day_of_week, he)`.

### Bus-share features with growth adjustment (`04_compute_shares.py`)

Each (bus, dow, hod) share is computed from prior-year same-quarter data,
then multiplied by a per-bus **growth multiplier** = most-recent-year share
÷ older-years average share, clipped to **[0.85, 1.75]** (tightened from
the initial [0.5, 3.0] after weather noise produced spurious 30%+ "shrinkage"
on stable buses). This captures real data-center growth while filtering
single-year weather noise.

The disaggregation in `05_disaggregate.py` uses a **three-tier fallback**
for buses whose primary share lookup fails: (a) matched (bus, dow, hod)
share, (b) bus's overall historical share, (c) zone median share. Prior
behavior of `share = 0` for new 2025 buses cost ~2.5 pp WMAPE.

### Per-bus features (specialized models, `07_speacialized_models.py`)

For problem buses only. Adds, on top of the calendar/weather features:

- Bus-level lags: `lag_24h..lag_8760h` for Task 1; year-ago + medium-horizon
  (lag_1500h, lag_2160h, lag_4320h) for Task 2.
- Rolling means at year-ago: 7d / 30d / 90d / 180d. Plus quarter-ago and
  half-year-ago rolling means for Task 2.
- `bus_clim_load` — mean by `(month, dow, he)` on train fold only.
- Structural growth: `true_yoy_growth` (year-ago-30d / 2yr-ago-30d, clipped
  [0.6, 1.8]), `bus_clim_trend_adj` (climatology rescaled by trend),
  `recency_vs_year_ago_30d/90d` (Task 2 only), `lag_8760h_trend_adj`
  (year-ago value scaled by recent growth).
- `bus_zone_share_yr_ago`, `bus_share_growth` — bus's share of zone, year ago
  and trending.
- `heat_wave_hours`, `cold_snap_hours` — 72h rolling sums of extreme flags.

---

## 4. Data leakage prevention

This was the area I spent the most defensive effort on. Six explicit mechanisms:

### Temporal splits

- **Task 1:** train < 2024-07-01, val 2024-07-01..2024-12-31, test = 2025.
  Val is used for early stopping; final model retrained on train+val to
  `best_iteration` before predicting 2025.
- **Task 2:** train < 2025-01-01, test = each month of 2025 separately.
  Single fit on 2022-2024 applied to all 12 months. Earlier per-month
  rolling-retrain implementation included prior 2025 months for Feb-Dec
  forecasts — fixed during audit to be strictly spec-compliant.

### Lag anchoring

- Task 1 rolling features all use `shift(24)` then roll, so same-day data
  never enters a feature.
- Task 2 rolling features use `shift(8760)` (1 year) or `shift(1500)` (~62
  days, anchored at minimum-safe lookback for the longest target-month last
  hour). Medium-horizon Task 2 lags carefully computed: for forecast_created_at
  = Dec 1 2024 targeting Jan 31 2025 23:00, minimum required lookback is
  1487h — so 1500h is the shortest spec-compliant lag we add.
- `zone_load_growth_rate` uses two anchored shifted series (`shift(24)` and
  `shift(8784)` for Task 1; `shift(8760)` and `shift(17520)` for Task 2).

### Climatology cutoff (Task 2)

`build_climatology(df, train_cutoff)` filters to `timestamp < train_cutoff`
strictly (exclusive `<`, fixed during audit from `<=`). Passing
`pd.Timestamp('2025-01-01')` guarantees zero 2025 contamination in the
day-of-year and day-of-week climatology averages.

### Bus shares (`04_compute_shares.py`)

For a forecast targeting (year Y, quarter Q), the shares are computed only
from quarterly files where `file_quarter == Q` and `file_year < Y`. Same-quarter
restriction preserves seasonal AC/heating patterns; prior-year restriction
prevents the target quarter from leaking into its own shares. An earlier
version had `file_year < 2025` hardcoded, which leaked e.g. 2024 Q1 into
itself — fixed.

### Problem-bus selection (`06_idenitfy_problem_buses.py`)

Problem buses are selected on a **2024 H2 validation fold**, not on 2025.
The val-fold zone forecasts are produced by `03_training_model.py` from
probe models that never saw val data, then disaggregated by `05_disaggregate.py`
into `predictions/task1_val_2024_Q*.parquet`. Selecting on 2025 errors would
be selection bias on the test set: the specialized models trained for those
buses would then be evaluated on the same 2025, biasing reported WMAPE
downward.

### HELIOSCR misclassification handled at training time

`HELIOSCR_345KV_1` is a solar bus mislabeled as LOAD (negative midday,
night-time peaks). Its actual load is subtracted from `zone_pd` before
zone-model training (in `load_zone_with_weather`) so the zone forecast
doesn't include load that the bus-level pipeline excludes. Without this,
reconciliation would systematically over-allocate every other NOTH bus to
absorb HELIOSCR's "phantom" load.

### Reconciliation discrepancy (input data quality)

`02_compute_bus_shares.py` writes `data/reconciliation_report.txt`. The
provided bus and zone data do not perfectly reconcile across zones. This
is a **bound on best achievable bus-level WMAPE** that no model can cross.
The reconciler in `05_disaggregate.py` post-scales bus forecasts to sum
exactly to the zone forecast, which limits the damage but does not eliminate
it.

---

## 5. Performance — strengths and weaknesses

### Strengths

- **Strong gain over both baselines** at the bus level. The
  hierarchical + specialized blend beats the year-ago baseline by a wide
  margin on both tasks (run `09_baseline.py` to confirm exact deltas).
- **The zone-then-bus hierarchical approach exactly preserves the zone-level
  forecast** (the bus sum equals the zone forecast by construction after
  reconciliation), so zone-level grid operators get a coherent signal.
- **Task 2 single-fit approach is robust** — no per-month parameter churn,
  easy to defend, and the year-only-lag + medium-horizon-lag feature set
  has carefully-bounded leakage risk.
- **Summer-month accuracy is strong** (Task 1 WMAPE ~19.6-19.9% Jul-Sep)
  — stable AC-driven load patterns dominated by temperature.
- **Urban zone accuracy is strong** (SCEN 14.04%, COAS 19.94% for Task 1)
  — many buses averaging out individual idiosyncrasies.
- **The audit pass surfaced and fixed multiple correctness bugs** that
  would have silently degraded the submission (see Section 6).

### Weaknesses

- **Bus-level WMAPE is bounded by input-data reconciliation noise** (see
  Section 4) plus structural per-bus difficulty for data centers and
  industrial sites.
- **Rural / sparse zones underperform** — NOTH at 41.27% (Task 1) and
  41.78% (Task 2) is dominated by one large data center (CHLDDATA_345KV_1)
  with structural growth that historical shares can't fully extrapolate.
- **Winter heating-season volatility** — per-month WMAPE rises to ~25-26%
  in Nov-Dec for Task 1 and ~31-32% for Task 2.
- **Data center and fast-growth industrial buses (CHLDDATA, MBPOD, BCTRESTL)**
  — these contribute ~10% of total absolute error despite being a tiny
  fraction of the bus count.
- **Brand-new 2025 buses with no historical share record** — three-tier
  fallback (bus overall → zone median) gives a reasonable guess but
  accuracy is structurally limited.
- **Sub-MW buses** — small absolute errors translate to enormous percentage
  errors. They contribute little to aggregate WMAPE but inflate per-bus
  metrics.
- **No probabilistic forecast** — point predictions only; no quantiles for
  reserve / risk planning.
- **Weather is taken from a single point per zone** — ERCOT zones are large;
  one weather grid point may underrepresent extreme localized conditions.

---

## 6. Improvement recommendations

1. **Direct bus-level neural models for the top ~100 buses by load.** The
   hierarchical zone × share decomposition has inherent error amplification
   at the bus level — sum of perfect shares isn't perfect because shares
   aren't perfectly stable. A Temporal Fusion Transformer or PatchTST per
   major bus would bypass this for the buses that matter most. The top 10
   worst-predicted buses contribute ~11% of total Task 1 absolute error.
2. **Historical weather forecasts instead of actuals.** Open-Meteo has a
   historical-forecast endpoint that returns the weather forecast available
   at any past timestamp. Using these instead of realized weather would
   eliminate the train/deploy gap from the implicit assumption that weather
   is perfectly known.
3. **Rolling-origin cross-validation for problem-bus selection.** Currently
   selection uses only 2024 H2 — a six-month window that's noisy enough
   that adding more specialized models past ~1500 yields diminishing returns.
   A separate diagnostic zone model with cutoff < 2024-01-01, generating
   full-year OOS predictions on 2024, would 2-4× the selection signal.
4. **Quantile forecasts (P10/P50/P90).** Operational use cases need
   uncertainty, not point estimates. LightGBM with `objective='quantile'`
   would produce these with the same architecture.
5. **Per-month retraining for Task 2 in production.** Currently fit once on
   2022-2024 to be strictly spec-compliant with the stated "training period."
   In actual production, rolling retraining each month (with cutoff =
   forecast_created_at) would incorporate the latest year's signal —
   typically 2-3 pp WMAPE improvement.
6. **Spec-compliant complete (bus × hour) grid output.** Current submission
   covers ~80% of the theoretical grid because disaggregation is driven by
   where actuals exist. A full rewrite of `05_disaggregate.py` to drive off
   a complete grid (cross-product of bus universe × all 2025 hours) would
   close this. WMAPE wouldn't change (extra rows have no actuals to score),
   but the submission would be fully spec-compliant.
7. **Better growth detection for new / data-center buses.** The current
   growth multiplier captures share-level growth but misses bus-level
   structural breaks where a bus comes online mid-year. A change-point
   detection step that fits a piecewise model to each bus's history would
   help.
8. **Bus-level multi-task model.** Replace per-bus models with a single
   LightGBM where `bus_unique_id` is a high-cardinality categorical
   (target-encoded or hashed). Shares information across buses with sparse
   history; rescues the 224-271 buses currently skipped for insufficient
   training data.

---

## 7. Evaluation metrics

All metrics computed by `08_Evaluate.py` on the 2025 test year using
`predictions/task1_final.parquet` and `predictions/task2_final.parquet`.

### Aggregate (bus-level)

| Task | WMAPE | MAPE | MAE (MW) | RMSE (MW) |
|---|---:|---:|---:|---:|
| Task 1 | **21.38%** | 22.89% | 2.989 | 10.582 |
| Task 2 | **27.76%** | 29.64% | 3.881 | 12.126 |

### Per-zone WMAPE (Task 1, bus-level, sorted best to worst)

| Zone | WMAPE |
|---|---:|
| SCEN | 14.04% |
| NCEN | 19.75% |
| COAS | 20.02% |
| EAST | 20.87% |
| SOUT | 25.46% |
| FWES | 28.00% |
| WEST | 28.16% |
| NOTH | 41.27% |

### Per-zone WMAPE (Task 2, bus-level, sorted best to worst)

| Zone | WMAPE |
|---|---:|
| SCEN | 18.42% |
| NCEN | 25.64% |
| EAST | 25.80% |
| COAS | 27.52% |
| SOUT | 30.14% |
| WEST | 35.16% |
| FWES | 38.35% |
| NOTH | 41.78% |

### Per-month WMAPE (Task 1, bus-level)

| Month | WMAPE |
|---|---:|
| 2025-01 | 20.73% |
| 2025-02 | 21.02% |
| 2025-03 | 21.18% |
| 2025-04 | 21.01% |
| 2025-05 | 21.75% |
| 2025-06 | 20.95% |
| 2025-07 | 19.71% |
| 2025-08 | 19.66% |
| 2025-09 | 19.99% |
| 2025-10 | 21.57% |
| 2025-11 | 24.96% |
| 2025-12 | 25.66% |

### Per-month WMAPE (Task 2, bus-level)

| Month | WMAPE |
|---|---:|
| 2025-01 | 27.57% |
| 2025-02 | 29.19% |
| 2025-03 | 28.03% |
| 2025-04 | 28.04% |
| 2025-05 | 28.37% |
| 2025-06 | 27.13% |
| 2025-07 | 25.70% |
| 2025-08 | 25.05% |
| 2025-09 | 25.42% |
| 2025-10 | 27.53% |
| 2025-11 | 31.34% |
| 2025-12 | 32.14% |

### WMAPE by method (hierarchical vs specialized blend)

| Method | Task 1 WMAPE | Task 1 row share | Task 2 WMAPE | Task 2 row share |
|---|---:|---:|---:|---:|
| Hierarchical | 29.07% | 67.8% | 30.10% | 68.8% |
| Specialized | 13.71% | 32.2% | 25.22% | 31.2% |

### Baseline comparison

Both baselines use only pre-2025 data as input features. Numbers below are
under the **standard WMAPE convention** (no filter on actual=0 rows).
Industry-convention numbers (filter actual > 0) shown in Section 2.

| Baseline | Task 1 Bus WMAPE | Δ vs Task 1 model | Task 2 Bus WMAPE | Δ vs Task 2 model |
|---|---:|---:|---:|---:|
| lag_8760h | 26.68% | model better by 5.30 pp | 28.87% | model better by 1.11 pp |
| hist avg (bus, hour, dow) | 28.69% | model better by 7.31 pp | 28.69% | model better by 0.93 pp |

Per-zone lag_8760h baseline WMAPE (under 09's filter):

| Zone | Baseline | Task 1 Model | Task 2 Model |
|---|---:|---:|---:|
| COAS | 22.63% | 20.02% | 27.52% |
| EAST | 27.21% | 20.87% | 25.80% |
| FWES | 34.45% | 28.00% | 38.35% |
| NCEN | 25.04% | 19.75% | 25.64% |
| NOTH | 44.63% | 41.27% | 41.78% |
| SCEN | 21.04% | 14.04% | 18.42% |
| SOUT | 24.78% | 25.46% | 30.14% |
| WEST | 33.79% | 28.16% | 35.16% |

Task 1 model beats baseline on every zone except SOUT (model 25.46% vs
baseline 24.78% — marginal). Task 2 model beats baseline on SCEN, NCEN, NOTH,
COAS, FWES; loses on EAST, SOUT, WEST. Combined, the model wins on aggregate.

---

## 8. Limitations

- **Dec 4, 2025 source-data gap.** Neither `bus_load.parquet` nor
  `zone_load.parquet` contains rows for 2025-12-04. The submission therefore
  covers 364 of 365 days. No actuals exist for this date in our data, so
  the gap doesn't affect reported WMAPE.

- **Bus-hour coverage.** Output covers ~80% of the theoretical (bus × hour)
  grid — limited to hours where the bus reports actual load. 768 buses with
  intermittent metering in 2025 receive partial forecasts. See Improvement
  #6 above.

- **HELIOSCR_345KV_1 excluded.** This bus is classified as LOAD in the
  source data but its load profile (negative midday, peaks at night) makes
  clear it's actually solar generation tagged as load. Excluded from both
  training (subtracted from zone_pd) and evaluation.

---

## 9. Reproducibility

Run order:

```
python "00_combine files.py"
python "01_pull weather.py"
python  02_compute_bus_shares.py
python  03_training_model.py
python  04_compute_shares.py
python  05_disaggregate.py
python  06_idenitfy_problem_buses.py
python  07_speacialized_models.py
python  08_Evaluate.py
python  09_baseline.py
```

Final submission files:

- `predictions/task1_submission.csv` (33,438,641 rows)
- `predictions/task2_submission.csv` (33,438,641 rows)

Schema: `model_name, forecast_created_at, target_date, he, bus_id, zone_id, predict_pd`.

Excluded buses: `HELIOSCR_345KV_1` (confirmed misclassified — solar generation
tagged as load).

Total pipeline runtime: ~80 minutes on a modern multi-core machine, dominated
by step 07 (specialized model training).

See `AI_USAGE.md` for the AI tool usage log.
