# AI Tool Usage Log

> Per the assignment: "AI tools may be used for coding, debugging, data
> exploration, feature engineering, model comparison, documentation, and
> report drafting." This document records where AI was used, what was
> kept vs. discarded, and how outputs were validated.

## Tools used

| Tool | Mode | Primary use |
|---|---|---|
| Claude (Anthropic, Sonnet 4.5) | Claude Code CLI agent in project directory | Pipeline audit, leakage review, debugging, code edits, performance optimization, report drafting |
| <add other tools you used — e.g., ChatGPT web, GitHub Copilot, Cursor, Claude.ai web> | | |

## Where AI helped (and what it actually produced)

| Pipeline stage | File(s) | What I asked AI for | What I kept |
|---|---|---|---|
| Data combine | `00_combine files.py` | Streaming parquet concat with polars | Kept as-is |
| Weather pull | `01_pull weather.py` | Open-Meteo API parameters, NWS heat-index formula | Kept; **corrected zone name `NRTH` → `NOTH` after audit caught silent NaN join leaving one zone weather-blind; extended date range to 2025 after audit caught 100% NaN weather on test set** |
| Reconciliation | `02_compute_bus_shares.py` | Per-zone bus_sum vs zone_pd diagnostics | Kept; informs known data-quality bound (see REPORT.md) |
| Feature engineering | `03_training_model.py` | Lag/rolling structure, cyclical encodings, Texas school calendar, extreme-weather thresholds | Kept; **AI initially used °C thresholds (35/0/38) — caught and fixed to °F (95/32/100) during audit** |
| Zone modeling | `03_training_model.py` | LightGBM hyperparameters, FWES dedicated + shared-model architecture, NCEN/SCEN/SOUT ensemble blend | Kept architecture; tuned hyperparameters myself |
| Task 2 zone training | `03_training_model.py` | Rewrite from per-month rolling-retrain to single-fit on 2022-2024 | Kept; **prior per-month implementation leaked prior 2025 months into Feb-Dec training data — strict spec compliance now enforced** |
| HELIOSCR exclusion fix | `03_training_model.py` | Subtract HELIOSCR's pd from zone_pd before zone training | Kept; **prior approach excluded HELIOSCR only at evaluation, causing systematic over-allocation to every other NOTH bus via reconciliation** |
| Bus shares | `04_compute_shares.py` | Same-quarter prior-year share construction, fallback smoothing, growth multiplier with bounded clip [0.85, 1.75] | Kept; **prior `< 2025` hardcoded year filter leaked 2024 Q1 into itself — fixed to `< target_year`; growth-multiplier clip tightened after audit showed [0.5, 3.0] amplified weather noise on stable buses** |
| Disaggregation | `05_disaggregate.py` | Hierarchical zone × share with 3-tier fallback for missing/new buses, post-hoc reconciliation, HELIOSCR exclusion | Kept; **previously share = 0 for new 2025 buses cost ~2.5 pp WMAPE; fallback tiers added during audit** |
| Problem-bus selection | `06_idenitfy_problem_buses.py` | Pick high-error buses on the val fold (not test) to avoid selection bias on 2025; split MIN_LOAD thresholds | Kept; **prior version picked problem buses from 2025 test errors — selection bias on the test set; now uses 2024 H2 OOS fold** |
| N_problem_buses sweep | `06_idenitfy_problem_buses.py` | Empirical test at N = 500 / 700 / 1500 | Kept N=1500; data showed Task 1 ~1.1 pp better at 1500 vs 700, Task 2 marginal |
| Specialized per-bus models | `07_speacialized_models.py` | Per-bus model with task-specific lag sets, performance fixes | Kept; **AI initially set `n_jobs=1` (1 core only) and ran probe+retrain (~2× compute) — fixed during audit, training went from 7+ hours to 40 min**; **AI initially used Python `.apply()` for holiday lookups — vectorized to `.isin()` during audit (~100× speedup)** |
| Fix E (Task 2 specialized lags) | `07_speacialized_models.py` | Medium-horizon lags (lag_1500h, lag_2160h, lag_4320h) + recency-vs-year-ago ratios | Kept; carefully bounded at ≥1500h to remain spec-compliant for next-month horizon (FCA = first day of prior month, max required lookback = 1487h for last hour of longest target month) |
| Final evaluation | `08_Evaluate.py` | Blend hierarchical + specialized, compute per-zone/per-month WMAPE | Kept; **AI's initial `forecast_created_at` for Task 2 evaluated to "first day of target month" instead of "first day of PRIOR month" — caught during final audit, fixed to use `target_ts.dt.to_period('M') - 1`** |
| Baseline | `09_baseline.py` | Year-ago (lag_8760h) and (bus, hour, dow) historical-average baselines | Kept |
| Audit / leakage review | all files | Multiple independent review passes | Found and fixed (across several passes): dow off-by-one in submission, CDH unit bug (°C vs °F), specialized-models over-training (n_jobs=1, probe+retrain), EXCLUDE_BUSES inconsistency between 03/05/08, scale-factor blow-up risk, share leakage in 04 (`< 2025` hardcoded), problem-bus selection bias (selecting on test set), Task 2 monthly retrain spec violation, missing 2025 weather, NOTH zone-name mismatch, Task 2 FCA computation bug, share-lookup zero forecasts (~2.5 pp WMAPE), HELIOSCR reconciliation systematic over-allocation |

## Where I did NOT use AI (decisions are mine)

- Choice of LightGBM as the model family.
- Train / val / test temporal split: train < 2024-07-01, val 2024-07-01..2024-12-31, test = 2025.
- Decision to build a dedicated FWES model + shared model for the other 7 zones, then layer an ensemble for NCEN/SCEN/SOUT.
- Problem-bus selection thresholds (`PROBLEM_THRESHOLD_WMAPE = 15`, `MIN_LOAD_FOR_SELECTION = 1.0`).
- Final pick of N=1500 specialized buses after AI presented the empirical sweep at 500/700/1500.
- Exclusion of `HELIOSCR_345KV_1` (manually confirmed as a misclassified solar/generation bus).
- Decision to use `regression_l1` with load-proportional weighting (Task 2) so the training objective aligns with WMAPE.
- Decision to apply medium-horizon lags for Task 2 specialized at 1500 buses rather than only 700 after seeing the empirical impact.
- Final pick of which submission file to ship.

## Validation

- Every AI-generated code block was executed end-to-end before being kept.
- Numeric outputs were checked against:
  - Zone-level reconciliation report (`data/reconciliation_report.txt`).
  - Per-zone WMAPE printed at each stage.
  - Baseline WMAPE from `09_baseline.py` (model must beat lag_8760h baseline).
  - Empirical sweep results at multiple `MAX_PROBLEM_BUSES` values.
  - Final submission CSV audit (row count, schema, NaN/negative/infinity checks, lead-time distribution).
- Iterative audit passes surfaced bugs that AI had introduced or that AI initially missed in earlier code (counted ~12 distinct correctness issues across the project) — all documented above and fixed in the committed code.
- AI predictions of improvement magnitudes were often off (e.g., predicted 4-5 pp from Fix E, got 0.5 pp; predicted 1500 → 700 would help, found 1500 was better). The empirical sweeps caught these mispredictions before they affected the final submission.

## Chat history / raw logs

- Claude Code transcripts: I have the file but I wasn't sure how to upload it if you need the file, I will send the file.

## Honest scope statement

AI was used heavily for:
- **Boilerplate code** (data loading, polars/pandas idioms, LightGBM glue code, parquet I/O)
- **Audit and debugging** (catching the ~12 bugs documented above through structured review passes)
- **Performance optimization** (n_jobs fix, vectorizing `.apply()` calls, avoiding redundant retraining)
- **Documentation drafting** (this report and the AI usage log)

**Architectural decisions were mine** (zone-then-bus hierarchy, FWES carve-out, ensemble blend, problem-bus specialization, choice of N=1500, choice of horizons and split points).

**Diagnostic and empirical decisions were a collaboration**: AI proposed hypotheses (e.g., "the growth multiplier is hurting hierarchical"), I ran experiments to test them, and we revised based on data (e.g., empirically Fix A barely helped, so Fix B was abandoned). AI was wrong on multiple specific predictions — empirical validation is what produced the final pipeline.

All outputs in `predictions/` were generated from the committed code on my
machine; I can re-run the pipeline end-to-end on request.
