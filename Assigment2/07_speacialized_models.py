import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from tqdm import tqdm

try:
    import holidays
    US_HOLIDAYS = set(holidays.country_holidays('US', years=range(2018, 2030)).keys())
    US_HOLIDAYS_TS = pd.to_datetime(sorted(US_HOLIDAYS))
except ImportError:
    US_HOLIDAYS = set()
    US_HOLIDAYS_TS = pd.to_datetime([])

DATA_DIR = Path('data')
MODELS_DIR = Path('models')
PREDICTIONS_DIR = Path('predictions')

# Buses confirmed as misclassified (solar/generation tagged as load).
# Excluded from both training and evaluation.
EXCLUDE_BUSES = {'HELIOSCR_345KV_1'}

def compute_metrics(y_true, y_pred, label=""):
    y_true = np.asarray(y_true,dtype=np.float64)
    y_pred = np.asarray(y_pred,dtype=np.float64)
    mask = y_true > 1.0
    wmape = np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1e-5) * 100

    if label:
        print(f"\n{label}: WMAPE: {wmape:.2f}%")
    return {"wmape": wmape}

def load_problem_buses():
    return pl.read_parquet(DATA_DIR / 'problem_buses.parquet')

def load_bus_history(bus_ids, bus_data_path):
    print(f"Loading history for {len(bus_ids)} buses...")
    df = (
        pl.scan_parquet(bus_data_path)
        .filter(pl.col('bus_unique_id').is_in(bus_ids))
        .filter(pl.col('bus_type') == 'LOAD')
        .filter(pl.col('pd').is_not_null())
        .with_columns([
            (pl.col('date').cast(pl.Date).cast(pl.Datetime) + pl.duration(hours=pl.col('he')-1)).alias('timestamp'),
        ]).select(['bus_unique_id','zone_name','date','he','timestamp','pd','base_kv']).collect()
    )

    print(f'Loaded {df.height:,} rows for {df['bus_unique_id'].n_unique()} buses')

    return df

def add_features_for_bus_model(df_pd, min_lag_hours):
    df_pd['hour'] = df_pd['he']
    df_pd['day_of_week'] = df_pd['timestamp'].dt.dayofweek
    df_pd['day_of_year'] = df_pd['timestamp'].dt.dayofyear
    df_pd['day_of_month'] = df_pd['timestamp'].dt.day
    df_pd['month'] = df_pd['timestamp'].dt.month
    df_pd['quarter'] = df_pd['timestamp'].dt.quarter
    ts_date = df_pd['timestamp'].dt.normalize()
    df_pd['is_holiday']         = ts_date.isin(US_HOLIDAYS_TS).astype(int)
    df_pd['day_before_holiday'] = (ts_date + pd.Timedelta(days=1)).isin(US_HOLIDAYS_TS).astype(int)
    df_pd['day_after_holiday']  = (ts_date - pd.Timedelta(days=1)).isin(US_HOLIDAYS_TS).astype(int)
    df_pd['is_christmas_week'] = (
        ((df_pd['month'] == 12) & (df_pd['day_of_month'] >= 24)) |
        ((df_pd['month'] == 1)  & (df_pd['day_of_month'] == 1))
    ).astype(int)
    df_pd['is_thanksgiving_week'] = (
        (df_pd['month'] == 11) & (df_pd['day_of_month'] >= 22) & (df_pd['day_of_month'] <= 29)
    ).astype(int)
    df_pd['is_monday']         = (df_pd['day_of_week'] == 0).astype(int)
    df_pd['is_ac_season']      = df_pd['month'].isin([5,6,7,8,9]).astype(int)
    df_pd['is_heating_season'] = df_pd['month'].isin([12,1,2]).astype(int)

    df_pd["hour_sin"]  = np.sin(2*np.pi*df_pd["hour"]/24)
    df_pd["hour_cos"]  = np.cos(2*np.pi*df_pd["hour"]/24)
    df_pd["month_sin"] = np.sin(2*np.pi*df_pd["month"]/12)
    df_pd["month_cos"] = np.cos(2*np.pi*df_pd["month"]/12)
    df_pd["dow_sin"]   = np.sin(2*np.pi*df_pd["day_of_week"]/7)
    df_pd["dow_cos"]   = np.cos(2*np.pi*df_pd["day_of_week"]/7)
    df_pd["is_weekend"] = df_pd["day_of_week"].isin([5, 6]).astype(int)

    df_pd = df_pd.sort_values('timestamp')

    if min_lag_hours >= 8760:
        df_pd['lag_8760h'] = df_pd['pd'].shift(8760)
        shifted = df_pd['pd'].shift(8760)
        df_pd['roll_mean_year_ago_7d']  = shifted.rolling(168, min_periods=48).mean()
        df_pd['roll_mean_year_ago_30d'] = shifted.rolling(720, min_periods=180).mean()
        df_pd['roll_mean_year_ago_90d'] = shifted.rolling(2160, min_periods=500).mean()

        # ─── Fix E: Medium-horizon lags (62.5 - 180 days back) ────────────────
        # Task 2 forecast_created_at = first day of the prior month. For the
        # last hour of the longest target month, the minimum required lookback
        # is ~1487 hours (e.g. Jan 31 23:00 minus Dec 1 00:00). So 1500h is
        # the shortest spec-compliant lag we can add — anything shorter would
        # leak data from inside the target window.
        #
        # Why this matters: lag_8760h gives year-ago info, which can't see
        # any growth/decline that started in the last few months. For a data
        # center that came online in early 2025, year-ago says "not there
        # yet" while quarter-ago says "ramping up fast." That's the gap
        # Task 2 specialized was sitting at (24% WMAPE vs Task 1 spec at 13%).
        df_pd['lag_1500h'] = df_pd['pd'].shift(1500)
        df_pd['lag_2160h'] = df_pd['pd'].shift(2160)
        df_pd['lag_4320h'] = df_pd['pd'].shift(4320)

        shifted_q = df_pd['pd'].shift(1500)
        df_pd['roll_mean_q_ago_30d'] = shifted_q.rolling(720,  min_periods=180).mean()
        df_pd['roll_mean_q_ago_90d'] = shifted_q.rolling(2160, min_periods=500).mean()

        shifted_h = df_pd['pd'].shift(4320)  # ~6 months back
        df_pd['roll_mean_half_yr_ago_30d'] = shifted_h.rolling(720, min_periods=180).mean()

        if len(df_pd) > 17520:
            shifted2 = df_pd['pd'].shift(17520)
            df_pd['lag_17520h'] = df_pd['pd'].shift(17520)
            df_pd['roll_mean_2yr_ago_30d'] = shifted2.rolling(720, min_periods=180).mean()

        _yoy_raw = (
            df_pd['roll_mean_year_ago_30d'] /
            df_pd['roll_mean_year_ago_90d'].replace(0, np.nan)
        ).fillna(1.0)
        df_pd['yoy_load_trend']     = _yoy_raw.clip(0.3, 3.0)
        df_pd['is_new_load_regime'] = (_yoy_raw > 1.5).astype(int)

        # Fix #5: True structural growth features
        # DOW-aligned year-ago (52 exact weeks = 364 days preserves day-of-week)
        df_pd['lag_8736h'] = df_pd['pd'].shift(8736)

        # 6-month rolling: stable annual-level baseline (less seasonal noise than 30d)
        df_pd['roll_mean_year_ago_180d'] = shifted.rolling(4320, min_periods=720).mean()

        # Month-vs-annual: how "peaky" is this month vs the bus's full-year pattern?
        # Heating bus in Jan → ~1.8; flat data center → ~1.0 all year
        df_pd['month_vs_annual_yr_ago'] = (
            df_pd['roll_mean_year_ago_30d'] /
            df_pd['roll_mean_year_ago_180d'].replace(0, np.nan)
        ).clip(0.3, 3.0).fillna(1.0)

        # True YoY structural growth: same 30-day window 1yr ago vs 2yr ago
        # A bus growing 15% (data center) gets true_yoy_growth=1.15 → model predicts +15%
        if 'roll_mean_2yr_ago_30d' in df_pd.columns:
            df_pd['true_yoy_growth'] = (
                df_pd['roll_mean_year_ago_30d'] /
                df_pd['roll_mean_2yr_ago_30d'].replace(0, np.nan)
            ).clip(0.6, 1.8).fillna(1.0)

            # Trend-adjusted climatology: static bus_clim_load is 15% too low for growing bus
            if 'bus_clim_load' in df_pd.columns:
                df_pd['bus_clim_trend_adj'] = df_pd['bus_clim_load'] * df_pd['true_yoy_growth']

        # ─── Fix E (continued): Recency growth signal ─────────────────────────
        # The single most important feature for catching data-center growth:
        # ratio of recent-quarter average to year-ago average for the same bus.
        # A bus growing 50% in the last few months gets recency_growth=1.5 →
        # model can project that forward instead of being anchored at year-ago.
        # Clipped to [0.5, 2.5] so noise on tiny buses can't blow up the signal.
        df_pd['recency_vs_year_ago_30d'] = (
            df_pd['roll_mean_q_ago_30d'] /
            df_pd['roll_mean_year_ago_30d'].replace(0, np.nan)
        ).clip(0.5, 2.5).fillna(1.0)

        df_pd['recency_vs_year_ago_90d'] = (
            df_pd['roll_mean_q_ago_90d'] /
            df_pd['roll_mean_year_ago_90d'].replace(0, np.nan)
        ).clip(0.5, 2.5).fillna(1.0)

        # Trend-adjusted year-ago: scale the year-ago value by recent growth.
        # This gives the model a "what would year-ago be if it had been
        # growing at the recent rate" reference, which is much closer to truth
        # for fast-growing buses than the raw lag_8760h.
        df_pd['lag_8760h_trend_adj'] = df_pd['lag_8760h'] * df_pd['recency_vs_year_ago_30d']

        # Bus share of zone (year-ago) and whether that share is growing
        if 'zone_pd' in df_pd.columns:
            zone_yr_ago = df_pd['zone_pd'].shift(8760)
            bus_yr_ago  = df_pd['pd'].shift(8760)
            df_pd['bus_zone_share_yr_ago'] = (
                bus_yr_ago / zone_yr_ago.replace(0, np.nan)
            ).clip(0.0, 0.5).fillna(0.0)

            if len(df_pd) > 17520:
                zone_2yr_ago  = df_pd['zone_pd'].shift(17520)
                bus_2yr_ago_s = df_pd['pd'].shift(17520)
                share_yr_ago  = bus_yr_ago  / zone_yr_ago.replace(0, np.nan)
                share_2yr_ago = bus_2yr_ago_s / zone_2yr_ago.replace(0, np.nan)
                df_pd['bus_share_growth'] = (
                    share_yr_ago / share_2yr_ago.replace(0, np.nan)
                ).clip(0.5, 2.0).fillna(1.0)

    else:
        for lag in [24, 48, 168, 336, 720, 8760]:
            df_pd[f'lag_{lag}h'] = df_pd['pd'].shift(lag)

        shifted = df_pd['pd'].shift(24)
        for window in [24, 168, 720]:
            df_pd[f"roll_mean_{window}h"] = shifted.rolling(window, min_periods=max(1, window//4)).mean()
            df_pd[f"roll_std_{window}h"]  = shifted.rolling(window, min_periods=max(1, window//4)).std()

        if 'zone_pd' in df_pd.columns:
            df_pd['zone_pd_lag_24h']  = df_pd['zone_pd'].shift(24)
            df_pd['zone_pd_lag_168h'] = df_pd['zone_pd'].shift(168)
            df_pd['bus_zone_ratio_24h'] = (
                df_pd['lag_24h'] / df_pd['zone_pd_lag_24h'].replace(0, np.nan)
            ).clip(0.01, 10.0).fillna(0)

        df_pd['lag_ratio_week'] = (
            df_pd['lag_24h'] / df_pd['lag_168h'].replace(0, np.nan)
        ).clip(0.5, 2.0).fillna(1.0)
        df_pd['lag_ratio_year'] = (
            df_pd['lag_24h'] / df_pd['lag_8760h'].replace(0, np.nan)
        ).clip(0.5, 2.0).fillna(1.0)

    if 'temperature_2m' in df_pd.columns:
        # temperature_2m is in °F throughout the pipeline (01_pull weather.py sets unit=fahrenheit).
        df_pd['temp_sq']        = df_pd['temperature_2m'] ** 2
        df_pd['cdh']            = np.maximum(0, df_pd['temperature_2m'] - 65.0)  # 65°F cooling threshold
        df_pd['temp_x_hour']    = df_pd['temperature_2m'] * df_pd['hour']
        df_pd['temp_x_weekend'] = df_pd['temperature_2m'] * df_pd['is_weekend']
        df_pd['temp_lag_24h']   = df_pd['temperature_2m'].shift(24)
        df_pd['temp_delta_24h'] = df_pd['temperature_2m'] - df_pd['temp_lag_24h']

        # Extreme weather thresholds — °F
        df_pd['is_extreme_heat'] = (df_pd['temperature_2m'] > 95.0).astype(int)
        df_pd['is_extreme_cold'] = (df_pd['temperature_2m'] < 32.0).astype(int)
        # Heat wave / cold snap accumulation: 3rd consecutive day hits harder than 1st
        df_pd['heat_wave_hours'] = (
            (df_pd['temperature_2m'] > 95.0).astype(float).rolling(72, min_periods=1).sum()
        )
        df_pd['cold_snap_hours'] = (
            (df_pd['temperature_2m'] < 32.0).astype(float).rolling(72, min_periods=1).sum()
        )

    # Texas school calendar (approximate — school starts ~Aug 15, ends late May)
    df_pd['is_summer_break'] = (
        (df_pd['month'].isin([6, 7])) |
        ((df_pd['month'] == 8) & (df_pd['day_of_month'] <= 14))
    ).astype(int)
    df_pd['is_winter_break'] = (
        ((df_pd['month'] == 12) & (df_pd['day_of_month'] >= 20)) |
        ((df_pd['month'] == 1)  & (df_pd['day_of_month'] <= 5))
    ).astype(int)
    df_pd['is_spring_break'] = (
        (df_pd['month'] == 3) &
        (df_pd['day_of_month'] >= 10) & (df_pd['day_of_month'] <= 20)
    ).astype(int)
    df_pd['is_school_day'] = (
        (~df_pd['is_summer_break'].astype(bool)) &
        (~df_pd['is_winter_break'].astype(bool)) &
        (~df_pd['is_spring_break'].astype(bool)) &
        (df_pd['day_of_week'] < 5) &
        (df_pd['is_holiday'] == 0)
    ).astype(int)

    return df_pd

def build_bus_climatology(df, train_cutoff):
    train = df[df['timestamp'] < train_cutoff].copy()
    train['month'] = train['timestamp'].dt.month
    train['dow'] = train['timestamp'].dt.dayofweek
    clim = (
        train.groupby(['month', 'dow', 'he'])['pd']
        .mean().reset_index()
        .rename(columns={'pd': 'bus_clim_load'})
    )
    df = df.copy()
    df['month'] = df['timestamp'].dt.month
    df['dow'] = df['timestamp'].dt.dayofweek
    df = df.merge(clim, on=['month', 'dow', 'he'], how='left')
    fallback = train['pd'].mean() if len(train) > 0 else 0.0
    df['bus_clim_load'] = df['bus_clim_load'].fillna(fallback)
    return df

def clean_training_data(df):
    """Remove bad rows from a training split (never call on test data)."""
    # 1. Negative load is impossible for a load bus
    df = df[df['pd'] >= 0].copy()

    # 2. Consecutive zeros ≥6 hours = meter failure (genuinely zero for 6+ hours is extremely rare)
    df = df.sort_values('timestamp').reset_index(drop=True)
    is_zero  = df['pd'] == 0
    group_id = (is_zero != is_zero.shift()).cumsum()
    run_len  = is_zero.groupby(group_id).transform('count')
    df = df[~is_zero | (run_len < 6)].copy()

    # 3. Extreme outliers: >6σ above mean are almost certainly metering errors
    mean_pd = df['pd'].mean()
    std_pd  = df['pd'].std()
    if std_pd > 0:
        df = df[df['pd'] <= mean_pd + 6 * std_pd].copy()

    return df

def train_one_bus_model(bus_id,bus_df,weather_df,task):
    min_lag = 24 if task == 'task1' else 8760

    df = bus_df.merge(weather_df,on=['zone_name','date','he'],how='left')
    df = add_features_for_bus_model(df,min_lag_hours=min_lag)
    df = build_bus_climatology(df, train_cutoff='2025-01-01')
    df = df.dropna(subset=[f"lag_{min_lag}h"]).copy()

    if len(df) < 1000:
        return None, None

    # 3-way split: train / val (Oct-Dec 2024) / test (2025)
    train = df[df['timestamp'] < '2024-10-01'].copy()
    val   = df[(df['timestamp'] >= '2024-10-01') & (df['timestamp'] < '2025-01-01')].copy()
    test  = df[df['timestamp'] >= '2025-01-01'].copy()

    # Clean training splits only — test stays untouched for full prediction coverage
    train = clean_training_data(train)
    val   = clean_training_data(val)

    if len(train) < 500 or len(val) < 100 or len(test) == 0:
        return None, None

    EXCLUDE = ['pd','bus_unique_id','zone_name','date','timestamp','he','zone_pd']
    features = [c for c in df.select_dtypes(include='number').columns if c not in EXCLUDE]

    params = {
        "objective":        "regression_l1",
        "metric":           "mae",
        "num_leaves":       31,
        "learning_rate":    0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "min_data_in_leaf": 20,
        "lambda_l2":        1.0,
        "verbosity":        -1,
        "n_jobs":           -1,
        "random_state":     42,
    }

    # Load-proportional weights for Task 2 — aligns MAE training objective with WMAPE evaluation.
    # Val set is intentionally unweighted so early stopping measures unweighted MAE;
    # weighted early stopping gets dominated by peak hours and stops too early.
    load_w_train = (train['pd'] / train['pd'].mean()).clip(0.1, 10.0) if task == 'task2' else None

    # Single pass: train with early stopping on val. model.predict() uses best_iteration
    # automatically. We give up ~3 months of training data (val) in exchange for ~2x speedup,
    # which is a good trade given train already spans multiple years.
    train_set = lgb.Dataset(train[features].astype('float64'), label=train['pd'], weight=load_w_train)
    val_set   = lgb.Dataset(val[features].astype('float64'),   label=val['pd'], reference=train_set)
    model = lgb.train(
        params, train_set,
        num_boost_round=1000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    test_pred = np.clip(model.predict(test[features].astype('float64')), 0, None)
    test['bus_forecast_specialized'] = test_pred

    return model, test[['bus_unique_id','zone_name','date','he','timestamp','pd','bus_forecast_specialized']]

def train_all_specialized(task):
    print(f"\n{'='*70}")
    print(f"SPECIALIZED MODELS for {task.upper()}")
    print("="*70)
 
    bus_data_path = DATA_DIR / "bus_load.parquet"

    # Per the design: specialized models are only for the high-error "problem buses"
    # identified by 06_idenitfy_problem_buses.py (val-fold WMAPE > threshold).
    # Training a per-bus model for every LOAD bus would (a) take hours, and
    # (b) override the hierarchical pipeline entirely in 08's blend.
    problem_path = DATA_DIR / 'problem_buses.parquet'
    if not problem_path.exists():
        print(f"  ERROR: {problem_path} not found. Run 06_idenitfy_problem_buses.py first.")
        return
    bus_ids = pl.read_parquet(problem_path)['bus_unique_id'].to_list()
    bus_ids = [b for b in bus_ids if b not in EXCLUDE_BUSES]
    print(f"\n{len(bus_ids):,} problem buses to model "
          f"({len(EXCLUDE_BUSES)} excluded as misclassified)")
    bus_df = load_bus_history(bus_ids, bus_data_path)
 
    weather_pl = pl.read_parquet(DATA_DIR / "weather.parquet")
    weather_pd = weather_pl.to_pandas()
    weather_pd["date"] = pd.to_datetime(weather_pd["date"])
 
    bus_df_pd = bus_df.to_pandas()
    bus_df_pd["date"] = pd.to_datetime(bus_df_pd["date"])
    bus_df_pd["timestamp"] = pd.to_datetime(bus_df_pd["timestamp"])

    zone_pl = pl.read_parquet(DATA_DIR / "zone_load.parquet")
    if 'pd' in zone_pl.columns and 'zone_pd' not in zone_pl.columns:
        zone_pl = zone_pl.rename({'pd': 'zone_pd'})
    zone_pd_df = zone_pl.select(['zone_name','date','he','zone_pd']).to_pandas()
    zone_pd_df['date'] = pd.to_datetime(zone_pd_df['date'])
    bus_df_pd = bus_df_pd.merge(zone_pd_df, on=['zone_name','date','he'], how='left')
 
    model_dir = MODELS_DIR / f"specialized_{task}"
    model_dir.mkdir(parents=True, exist_ok=True)
 
    all_preds = []
    skipped = 0
 
    for bus_id in tqdm(bus_ids, desc=f"Training {task} per-bus models"):
        bus_data = bus_df_pd[bus_df_pd["bus_unique_id"] == bus_id].copy()
        if len(bus_data) < 1000:
            skipped += 1
            continue
 
        model, preds = train_one_bus_model(bus_id, bus_data, weather_pd, task)
        if model is None or preds is None:
            skipped += 1
            continue
 
        model.save_model(str(model_dir / f"{bus_id}.txt"))
        all_preds.append(preds)
 
    print(f"\n  Trained: {len(all_preds):,}")
    print(f"  Skipped: {skipped:,}")
 
    if all_preds:
        combined = pd.concat(all_preds, ignore_index=True)
        combined.to_parquet(PREDICTIONS_DIR / f"{task}_specialized.parquet", index=False)
        print(f"  Saved → predictions/{task}_specialized.parquet")
        compute_metrics(
            combined["pd"].values,
            combined["bus_forecast_specialized"].values,
            label=f"Specialized {task} (problem buses)",
        )
 
 
def main():
    train_all_specialized("task1")  
    train_all_specialized("task2")

    print("\n" + "="*70)
    print("STAGE 6 COMPLETE — next step: python 07_evaluate.py")
    print("="*70)
 
 
if __name__ == "__main__":
    main()
 