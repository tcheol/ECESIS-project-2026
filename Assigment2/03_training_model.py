import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path

try:
    import holidays
    US_HOLIDAYS = set(holidays.country_holidays('US', years=range(2018,2030)).keys())
except ImportError:
    US_HOLIDAYS = set()

DATA_DIR = Path('data')
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok = True)

# Buses that are excluded from the bus-level disaggregation in 05_disaggregate.py.
# zone_pd here is the TRUE zone load (sum of all buses including these), so we
# must subtract them before training the zone model — otherwise the zone forecast
# includes their contribution, but reconciliation only distributes to the
# remaining buses, systematically over-predicting them.
# (Worst impact was NOTH, where HELIOSCR's load was inflating every other bus's
#  forecast by ~10-15%, blowing NOTH bus-hierarchical WMAPE up to 40%.)
# Must match EXCLUDE_BUSES in 05_disaggregate.py.
EXCLUDE_BUSES = ['HELIOSCR_345KV_1']


def load_zone_with_weather():
    print("Loading zone data")
    zone = pl.read_parquet(DATA_DIR / "zone_load.parquet")

    if 'pd' in zone.columns and 'zone_pd' not in zone.columns:
        zone = zone.rename({'pd': 'zone_pd'})

    zone = zone.with_columns([
        pl.col('date').str.to_datetime(format="%Y-%m-%d").alias('date'),
        (pl.col('date').str.to_datetime(format="%Y-%m-%d") +
         pl.duration(hours=pl.col('he')-1)).alias('timestamp')
    ])

    zone = zone.filter(pl.col('zone_name') != "ISOLATED")

    print(f"  Loaded {zone.height:,} zone-hours")
    print(f"  Zones: {sorted(zone['zone_name'].unique().to_list())}")

    # ───────────────────────────────────────────────────────────────────────
    # Subtract excluded buses' load from zone_pd so the zone model trains
    # against the *disaggregable* portion of zone load only. Without this,
    # the zone forecast includes load that won't be allocated to any bus
    # at disaggregation time, and reconciliation will pad every remaining
    # bus's forecast upward to absorb the difference.
    # ───────────────────────────────────────────────────────────────────────
    excluded_load = (
        pl.scan_parquet(DATA_DIR / 'bus_load.parquet')
        .filter(pl.col('bus_unique_id').is_in(EXCLUDE_BUSES))
        .filter(pl.col('bus_type') == 'LOAD')
        .filter(pl.col('pd').is_not_null())
        .with_columns([
            pl.col('date').cast(pl.Date).cast(pl.Datetime('us')).alias('date'),
            pl.col('zone_name').cast(pl.Utf8).alias('zone_name'),
        ])
        .group_by(['zone_name', 'date', 'he'])
        .agg(pl.col('pd').sum().alias('excluded_pd'))
        .collect()
    )

    zone = zone.with_columns(pl.col('zone_name').cast(pl.Utf8))
    zone = (
        zone.join(excluded_load, on=['zone_name', 'date', 'he'], how='left')
        .with_columns(
            (pl.col('zone_pd') - pl.col('excluded_pd').fill_null(0.0))
            .clip(lower_bound=0.0)
            .alias('zone_pd')
        )
        .drop('excluded_pd')
    )

    n_affected = excluded_load.height
    total_subtracted = excluded_load['excluded_pd'].sum()
    by_zone = (
        excluded_load.group_by('zone_name')
        .agg(pl.col('excluded_pd').sum().alias('mwh'))
        .sort('mwh', descending=True)
    )
    print(f"  Excluded {len(EXCLUDE_BUSES)} bus(es) from zone_pd: "
          f"{n_affected:,} (zone,hour) cells, {total_subtracted:,.0f} MWh removed")
    for row in by_zone.iter_rows(named=True):
        print(f"    {row['zone_name']}: {row['mwh']:,.0f} MWh subtracted")

    print("Loading weather...")
    weather = pl.read_parquet(DATA_DIR/ "weather.parquet").with_columns(
        pl.col('date').cast(pl.Datetime("us"))
    )

    weather_cols = [
        'zone_name','date','he',
        'temperature_2m','relativehumidity_2m','dewpoint_2m',
        'apparent_temperature','windspeed_10m','cloudcover',
        'precipitation','heat_index','hdh'
    ]

    zone = zone.join(weather.select(weather_cols),
                     on=['zone_name','date','he'], how='left')
    
    df = zone.to_pandas()
    df['zone_name'] = df['zone_name'].astype('category')
    df = df.sort_values(['zone_name','timestamp']).reset_index(drop=True)
    # temperature_2m is in °F (set by 01_pull weather.py); 65°F is the standard cooling threshold.
    df['cdh'] = np.maximum(0, df['temperature_2m'] - 65.0)

    return df

def add_calendar_feature(df):
    df['hour'] = df['he']
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month
    df['quarter'] = df['timestamp'].dt.quarter
    df['year'] = df['timestamp'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['is_holiday'] = df['timestamp'].dt.date.apply(
        lambda d: d in US_HOLIDAYS
    ).astype(int)

    # Extreme weather — nonlinear load thresholds for Texas (temperature_2m is in °F)
    if 'temperature_2m' in df.columns:
        df['is_extreme_heat'] = (df['temperature_2m'] > 95.0).astype(int)   # >95°F
        df['is_extreme_cold'] = (df['temperature_2m'] < 32.0).astype(int)   # <32°F
        df['is_super_heat']   = (df['temperature_2m'] > 100.0).astype(int)  # >100°F

    # Texas school calendar (approximate)
    df['is_summer_break'] = (
        (df['month'].isin([6, 7])) |
        ((df['month'] == 8) & (df['day_of_month'] <= 14))
    ).astype(int)
    df['is_winter_break'] = (
        ((df['month'] == 12) & (df['day_of_month'] >= 20)) |
        ((df['month'] == 1)  & (df['day_of_month'] <= 5))
    ).astype(int)
    df['is_spring_break'] = (
        (df['month'] == 3) & (df['day_of_month'] >= 10) & (df['day_of_month'] <= 20)
    ).astype(int)
    df['is_school_day'] = (
        (~df['is_summer_break'].astype(bool)) &
        (~df['is_winter_break'].astype(bool)) &
        (~df['is_spring_break'].astype(bool)) &
        (df['day_of_week'] < 5) &
        (df['is_holiday'] == 0)
    ).astype(int)

    df["hour_sin"]  = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]  = np.cos(2*np.pi*df["hour"]/24)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["dow_sin"]   = np.sin(2*np.pi*df["day_of_week"]/7)
    df["dow_cos"]   = np.cos(2*np.pi*df["day_of_week"]/7)

    return df

def add_task1_features(df):
    SAFE_LAGS = [24,48,72,168,336,720,8760]
    df = df.sort_values(['zone_name','timestamp'])

    for lag in SAFE_LAGS:
        df[f'lag_{lag}h'] = (
            df.groupby('zone_name',observed = True) ['zone_pd'].shift(lag)
        )

    shifted = df.groupby('zone_name',observed = True)['zone_pd'].shift(24)

    for window in [24,168,720]:
        grp = shifted.groupby(df['zone_name'],observed=True)
        df[f'roll_mean_{window}h'] = grp.transform(
            lambda x: x.rolling(window, min_periods=max(1,window // 4)).mean()
        )
        df[f'roll_std_{window}h'] = grp.transform(
            lambda x: x.rolling(window, min_periods=max(1, window//4)).std()
        )

    df['temp_sq'] = df['temperature_2m'] ** 2
    df['temp_x_hour'] = df['temperature_2m'] * df['hour']
    df['apparent_temp_sq'] = df['apparent_temperature'] ** 2

    # Zone load growth rate: captures structural growth from data centers and population
    # NCEN (DFW) and SCEN (Austin) have large data center buildouts — this signals that
    shifted_24h   = df.groupby('zone_name', observed=True)['zone_pd'].shift(24)
    shifted_8784h = df.groupby('zone_name', observed=True)['zone_pd'].shift(8784)
    recent_90d    = shifted_24h.groupby(df['zone_name'], observed=True).transform(
        lambda x: x.rolling(2160, min_periods=500).mean()
    )
    year_ago_90d  = shifted_8784h.groupby(df['zone_name'], observed=True).transform(
        lambda x: x.rolling(2160, min_periods=500).mean()
    )
    df['zone_load_growth_rate'] = (
        recent_90d / year_ago_90d.replace(0, np.nan)
    ).clip(0.7, 1.5).fillna(1.0)

    # Secular growth trend — captures NCEN/SCEN data centre + population growth
    df['load_trend_days'] = (df['timestamp'] - pd.Timestamp('2022-01-01')).dt.days / 365.0

    # CDH split by weekday/weekend — commercial AC vs residential AC
    df['cdh_x_weekday'] = df['cdh'] * (df['day_of_week'] < 5).astype(float)
    df['cdh_x_weekend'] = df['cdh'] * (df['day_of_week'] >= 5).astype(float)

    # Solar depression proxy — midday clear-sky hours suppress SCEN net load via rooftop solar
    df['solar_depression_proxy'] = (
        df['hour'].between(10, 15).astype(float) *
        (1 - df['cloudcover'].fillna(50) / 100).clip(0, 1)
    )

    return df

def add_task2_features(df):
    df = df.sort_values(['zone_name','timestamp'])

    df['lag_8760h'] = (
        df.groupby('zone_name', observed=True)['zone_pd'].shift(8760)
    )

    shifted = df.groupby('zone_name', observed=True)['zone_pd'].shift(8760)
    df['roll_mean_year_ago_30d'] = shifted.groupby(
        df['zone_name'], observed=True
    ).transform(lambda x: x.rolling(720, min_periods=180).mean())
    df['rolling_mean_year_ago_90d'] = shifted.groupby(
        df['zone_name'], observed=True
    ).transform(lambda x: x.rolling(2160, min_periods=500).mean())

    # Zone load growth rate: year-ago 90-day avg vs two-years-ago 90-day avg
    # Safe for task2 — both anchored fully behind the forecast creation date
    shifted_8760h  = df.groupby('zone_name', observed=True)['zone_pd'].shift(8760)
    shifted_17520h = df.groupby('zone_name', observed=True)['zone_pd'].shift(17520)
    recent_yr_90d  = shifted_8760h.groupby(df['zone_name'], observed=True).transform(
        lambda x: x.rolling(2160, min_periods=500).mean()
    )
    prior_yr_90d   = shifted_17520h.groupby(df['zone_name'], observed=True).transform(
        lambda x: x.rolling(2160, min_periods=500).mean()
    )
    df['zone_load_growth_rate'] = (
        recent_yr_90d / prior_yr_90d.replace(0, np.nan)
    ).clip(0.7, 1.5).fillna(1.0)

    return df

def build_climatology(df, train_cutoff):
    # train_cutoff is treated as an EXCLUSIVE upper bound — anything strictly before
    # it is fair game for the climatology, anything at-or-after is held out.
    # Switched from <= to < so callers can pass `2025-01-01` and be guaranteed
    # zero 2025 contamination in the averages.
    train_df = df[df['timestamp'] < train_cutoff].copy()
    train_df['day'] = train_df['timestamp'].dt.dayofyear
    df['day'] = df['timestamp'].dt.dayofyear

    weather_clim = (
        train_df.groupby(['zone_name','day','he'], observed=True)
        .agg(
            temp_clim = ('temperature_2m', 'mean'),
            humid_clim = ('relativehumidity_2m','mean'),
            hdh_clim = ('hdh','mean'),
        ).reset_index()
    )
    load_clim = (
        train_df.groupby(['zone_name','day','he'],observed=True)
        .agg(zone_load_clim = ('zone_pd','mean')).reset_index()
    )

    train_df['dow'] = train_df['timestamp'].dt.dayofweek
    df['dow'] = df['timestamp'].dt.dayofweek

    load_clim_dow = (
        train_df.groupby(['zone_name','dow','he'],observed=True)
        .agg(zone_load_dow_clim = ('zone_pd','mean')).reset_index()
    )

    df = df.merge(weather_clim, on=['zone_name','day','he'],how='left')
    df = df.merge(load_clim, on=['zone_name','day','he'],how='left')
    df = df.merge(load_clim_dow, on=['zone_name','dow','he'],how='left')

    return df

def compute_metrics(y_true,y_pred,label=""):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
 
    mask = y_true > 1.0
    mape = (np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            if mask.sum() > 0 else np.nan)
    mae   = np.mean(np.abs(y_true - y_pred))
    rmse  = np.sqrt(np.mean((y_true - y_pred) ** 2))
    wmape = np.sum(np.abs(y_true - y_pred)) / max(np.sum(y_true), 1e-6) * 100
 
    if label:
        print(f"\n{label}:")
        print(f"  WMAPE: {wmape:6.2f}%   ← primary")
        print(f"  MAPE:  {mape:6.2f}%")
        print(f"  MAE:   {mae:.1f} MW")
        print(f"  RMSE:  {rmse:.1f} MW")
    return {"mape": mape, "wmape": wmape, "mae": mae, "rmse": rmse}

EXCLUDED_COLS = [
    'zone_pd','date','timestamp','he','load_bus_count','gen_bus_count','pg','day','dow',
]

def get_features(df):
    return [c for c in df.columns if c not in EXCLUDED_COLS]

def train_task1():
    print('\n' + '='*70)
    print('Task 1 (zone level): NEXT-DAY Forecast')
    print('='*70)
    print('Min safe lag: 24h')

    df = load_zone_with_weather()
    df = add_calendar_feature(df)
    df = add_task1_features(df)
    df = df.dropna(subset=['lag_24h','lag_168h']).copy()

    train = df[df['timestamp'] < '2024-07-01'].copy()
    val   = df[(df['timestamp'] >= '2024-07-01') & (df['timestamp'] < '2025-01-01')].copy()
    test  = df[df['timestamp'] >= '2025-01-01'].copy()

    print(f"\n  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    fwes_features   = [f for f in get_features(df) if f != 'zone_name']
    shared_features = get_features(df)
    print(f"  Features (FWES): {len(fwes_features)}  |  Features (shared): {len(shared_features)}")

    params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 50,
        'lambda_l2': 1.0,
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': 42,
    }

    all_test_preds = []

    # --- FWES: dedicated model ---
    print('\n--- FWES (dedicated model) ---')
    fwes_train = train[train['zone_name'] == 'FWES'].copy()
    fwes_val   = val[val['zone_name'] == 'FWES'].copy()
    fwes_test  = test[test['zone_name'] == 'FWES'].copy()

    fwes_w_train = fwes_train['zone_pd'] / fwes_train['zone_pd'].mean()
    fwes_train_set = lgb.Dataset(fwes_train[fwes_features], label=fwes_train['zone_pd'], weight=fwes_w_train)
    fwes_val_set   = lgb.Dataset(fwes_val[fwes_features], label=fwes_val['zone_pd'], reference=fwes_train_set)

    fwes_model = lgb.train(
        params, fwes_train_set,
        num_boost_round=3000,
        valid_sets=[fwes_train_set, fwes_val_set],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)],
    )
    print(f'  Best iteration: {fwes_model.best_iteration}')

    fwes_full     = pd.concat([fwes_train, fwes_val])
    fwes_w_full   = fwes_full['zone_pd'] / fwes_full['zone_pd'].mean()
    fwes_full_set = lgb.Dataset(fwes_full[fwes_features], label=fwes_full['zone_pd'], weight=fwes_w_full)
    fwes_final    = lgb.train(params, fwes_full_set, num_boost_round=fwes_model.best_iteration)
    fwes_final.save_model(str(MODELS_DIR / 'task1_zone_FWES.txt'))

    fwes_preds = np.clip(fwes_final.predict(fwes_test[fwes_features]), 0, None)
    fwes_test  = fwes_test.copy()
    fwes_test['zone_forecast'] = fwes_preds
    all_test_preds.append(fwes_test)

    # --- Other 7 zones: shared model ---
    print('\n--- Other 7 zones (shared model) ---')
    other_train = train[train['zone_name'] != 'FWES'].copy()
    other_val   = val[val['zone_name'] != 'FWES'].copy()
    other_test  = test[test['zone_name'] != 'FWES'].copy()

    other_w_train   = other_train['zone_pd'] / other_train['zone_pd'].mean()
    other_train_set = lgb.Dataset(other_train[shared_features], label=other_train['zone_pd'],
                                  categorical_feature=['zone_name'], weight=other_w_train)
    other_val_set   = lgb.Dataset(other_val[shared_features], label=other_val['zone_pd'],
                                  categorical_feature=['zone_name'], reference=other_train_set)

    other_model = lgb.train(
        params, other_train_set,
        num_boost_round=5000,
        valid_sets=[other_train_set, other_val_set],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(200)],
    )
    print(f'  Best iteration: {other_model.best_iteration}')

    other_full     = pd.concat([other_train, other_val])
    other_w_full   = other_full['zone_pd'] / other_full['zone_pd'].mean()
    other_full_set = lgb.Dataset(other_full[shared_features], label=other_full['zone_pd'],
                                  categorical_feature=['zone_name'], weight=other_w_full)
    other_final    = lgb.train(params, other_full_set, num_boost_round=other_model.best_iteration)
    other_final.save_model(str(MODELS_DIR / 'task1_zone_shared.txt'))

    other_preds = np.clip(other_final.predict(other_test[shared_features]), 0, None)
    other_test  = other_test.copy()
    other_test['zone_forecast'] = other_preds

    # Ensemble for problem zones: blend shared + dedicated model
    zone_params_override = {
        'NCEN': {**params, 'num_leaves': 255, 'learning_rate': 0.02, 'lambda_l2': 2.0},
        'SCEN': {**params, 'num_leaves': 255, 'learning_rate': 0.02, 'lambda_l2': 2.0},
        'SOUT': {**params, 'num_leaves': 191, 'learning_rate': 0.02, 'lambda_l2': 1.5},
    }

    # Captured so we can rebuild clean OOS predictions on val for problem-bus selection.
    # We can't use `other_final` / `z_final` for that — they were retrained on train+val.
    z_probes = {}

    for zone_code in ['NCEN', 'SCEN', 'SOUT']:
        z_params = zone_params_override.get(zone_code, params)
        print(f'\n--- {zone_code} ensemble (shared + dedicated blend) ---')
        z_train = other_train[other_train['zone_name'] == zone_code].copy()
        z_val   = other_val[other_val['zone_name'] == zone_code].copy()
        z_full  = pd.concat([z_train, z_val])

        # Recency weighting: exponential decay so recent data matters ~20x more than oldest
        _span_tr = max((z_train['timestamp'].max() - z_train['timestamp'].min()).days, 1)
        _days_tr = (z_train['timestamp'] - z_train['timestamp'].min()).dt.days.values
        recency_w_train = np.exp(3.0 * _days_tr / _span_tr)
        recency_w_train /= recency_w_train.mean()

        _span_fu = max((z_full['timestamp'].max() - z_full['timestamp'].min()).days, 1)
        _days_fu = (z_full['timestamp'] - z_full['timestamp'].min()).dt.days.values
        recency_w_full = np.exp(3.0 * _days_fu / _span_fu)
        recency_w_full /= recency_w_full.mean()

        z_probe_set = lgb.Dataset(z_train[fwes_features], label=z_train['zone_pd'], weight=recency_w_train)
        z_val_set   = lgb.Dataset(z_val[fwes_features],   label=z_val['zone_pd'], reference=z_probe_set)
        z_probe = lgb.train(
            z_params, z_probe_set, num_boost_round=3000,
            valid_sets=[z_probe_set, z_val_set],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)],
        )

        shared_val_pred    = np.clip(other_final.predict(z_val[shared_features]), 0, None)
        dedicated_val_pred = np.clip(z_probe.predict(z_val[fwes_features]), 0, None)

        best_alpha, best_wmape = 0.5, float('inf')
        for alpha in np.arange(0.0, 1.1, 0.1):
            blended = alpha * dedicated_val_pred + (1 - alpha) * shared_val_pred
            wmape = (np.sum(np.abs(z_val['zone_pd'].values - blended)) /
                     max(z_val['zone_pd'].sum(), 1e-6) * 100)
            if wmape < best_wmape:
                best_wmape, best_alpha = wmape, alpha
        print(f'  Optimal dedicated weight α={best_alpha:.1f}, val WMAPE: {best_wmape:.2f}%')

        z_full_set = lgb.Dataset(z_full[fwes_features], label=z_full['zone_pd'], weight=recency_w_full)
        z_final    = lgb.train(z_params, z_full_set, num_boost_round=z_probe.best_iteration)
        z_final.save_model(str(MODELS_DIR / f'task1_zone_{zone_code}.txt'))

        z_test_mask         = other_test['zone_name'] == zone_code
        shared_test_pred    = other_test.loc[z_test_mask, 'zone_forecast'].values
        z_test_rows         = other_test[z_test_mask]
        dedicated_test_pred = np.clip(z_final.predict(z_test_rows[fwes_features]), 0, None)
        blended_test_pred   = best_alpha * dedicated_test_pred + (1 - best_alpha) * shared_test_pred
        other_test.loc[z_test_mask, 'zone_forecast'] = blended_test_pred
        print(f'  Applied blend to {z_test_mask.sum():,} test rows')

        # Stash probe + chosen alpha so we can replay clean OOS predictions on val below
        z_probes[zone_code] = (z_probe, best_alpha)

    all_test_preds.append(other_test)

    # === Clean OOS val forecasts for problem-bus selection ===========================
    # Why: 06_idenitfy_problem_buses.py must NOT pick problem buses by looking at 2025
    # errors — that's selection bias on the test set. We build a parallel val-fold
    # forecast (2024-07-01 .. 2024-12-31) using only the probe models, which have
    # never seen val data. Disaggregation step (05) writes the bus-level version.
    val_pieces = []

    fwes_val_oos = fwes_val.copy()
    fwes_val_oos['zone_forecast'] = np.clip(
        fwes_model.predict(fwes_val[fwes_features]), 0, None
    )
    val_pieces.append(fwes_val_oos[['zone_name', 'date', 'he', 'timestamp', 'zone_pd', 'zone_forecast']])

    other_val_oos = other_val.copy()
    other_val_oos['zone_forecast'] = np.clip(
        other_model.predict(other_val[shared_features]), 0, None
    )

    # Overwrite ensemble-zone rows with the blended probe predictions
    for zone_code, (z_probe, alpha) in z_probes.items():
        mask = other_val_oos['zone_name'] == zone_code
        if not mask.any():
            continue
        rows = other_val_oos.loc[mask]
        ded  = np.clip(z_probe.predict(rows[fwes_features]),    0, None)
        shr  = np.clip(other_model.predict(rows[shared_features]), 0, None)
        other_val_oos.loc[mask, 'zone_forecast'] = alpha * ded + (1 - alpha) * shr

    val_pieces.append(other_val_oos[['zone_name', 'date', 'he', 'timestamp', 'zone_pd', 'zone_forecast']])

    val_full = pd.concat(val_pieces, ignore_index=True)
    val_full.to_parquet(DATA_DIR / 'task1_val_zone_forecast.parquet', index=False)
    print(f'\n  Saved OOS val forecasts (2024 H2, {len(val_full):,} rows) → '
          f'data/task1_val_zone_forecast.parquet')
    compute_metrics(val_full['zone_pd'].values, val_full['zone_forecast'].values,
                    label='Task 1 Val OOS (2024 H2 — used for problem-bus selection)')

    # --- Combine & score ---
    full_test = pd.concat(all_test_preds, ignore_index=True)
    compute_metrics(full_test['zone_pd'].values, full_test['zone_forecast'].values,
                    label='Task 1 Hybrid (2025)')

    print('\nPer-zone WMAPE:')
    by_zone = full_test.groupby('zone_name', observed=True).apply(
        lambda g: np.sum(np.abs(g['zone_pd'] - g['zone_forecast'])) / max(g['zone_pd'].sum(), 1e-6) * 100
    )
    for zone, wmape in by_zone.items():
        print(f"  {zone}: {wmape:.2f}%")

    full_test[['zone_name', 'date', 'he', 'timestamp', 'zone_pd', 'zone_forecast']].to_parquet(
        DATA_DIR / 'task1_zone_forecast.parquet', index=False
    )
    
def train_task2():
    print("\n" + "="*70)
    print("TASK 2 (zone level): NEXT-MONTH FORECAST")
    print("="*70)
    print("  Per spec: training period 2022-01-01 → 2024-12-31. 2025 is test only.")
    print("  One model is fit once on 2022-2024 and applied to all 12 months of 2025.")
    print("  No 2025 data enters fitting, climatology, or any feature on a train row.")

    df = load_zone_with_weather()
    df = add_calendar_feature(df)
    df = add_task2_features(df)

    # Climatology computed strictly from pre-2025 data (build_climatology now uses < cutoff).
    TRAIN_END = pd.Timestamp('2025-01-01')
    df = build_climatology(df, train_cutoff=TRAIN_END)
    df = df.dropna(subset=['lag_8760h']).copy()

    train = df[df['timestamp'] < TRAIN_END].copy()
    print(f"  Train rows: {len(train):,}  "
          f"({train['timestamp'].min()} → {train['timestamp'].max()})")

    features = get_features(df)
    print(f"  Features: {len(features)}")

    params = {
        'objective':        'regression_l1',
        'metric':           'mae',
        'num_leaves':       127,
        'learning_rate':    0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq':     5,
        'min_data_in_leaf': 50,
        'lambda_l2':        1.0,
        'verbosity':        -1,
        'n_jobs':           -1,
        'random_state':     42,
    }

    train_set = lgb.Dataset(train[features], label=train['zone_pd'],
                            categorical_feature=['zone_name'])
    model = lgb.train(params, train_set, num_boost_round=600)

    # Predict each calendar month of 2025 separately so we can label the
    # forecast_created_at field per the spec (first day of prior month).
    all_preds = []
    for month in range(1, 13):
        target_start = pd.Timestamp(year=2025, month=month, day=1)
        next_month   = target_start + pd.offsets.MonthBegin(1)
        # forecast_created_at = first day of the previous month (per spec).
        forecast_created_at = (target_start - pd.offsets.MonthBegin(1)).normalize()

        test = df[(df['timestamp'] >= target_start) & (df['timestamp'] < next_month)].copy()
        if len(test) == 0:
            print(f"    {target_start.strftime('%Y-%m')}: no test rows in zone data, skipping")
            continue

        test['zone_forecast']       = np.clip(model.predict(test[features]), 0, None)
        test['forecast_created_at'] = forecast_created_at

        all_preds.append(test[['zone_name', 'date', 'he', 'timestamp',
                               'forecast_created_at', 'zone_pd', 'zone_forecast']])
        compute_metrics(test['zone_pd'].values, test['zone_forecast'].values,
                        label=f"  {target_start.strftime('%Y-%m')}")

    if not all_preds:
        print("  No 2025 monthly test data — Task 2 evaluation skipped.")
        return

    full = pd.concat(all_preds, ignore_index=True)
    compute_metrics(full['zone_pd'].values, full['zone_forecast'].values,
                    label='Task 2 Zone OVERALL (2025)')
    full.to_parquet(DATA_DIR / 'task2_zone_forecasts.parquet', index=False)

def main():
    if not (DATA_DIR / "zone_load.parquet").exists():
        print("Missing data/zone_load.parquet")
        return
    if not (DATA_DIR / "weather.parquet").exists():
        print("Missing data/weather.parquet")
        return
 
    train_task1()
    train_task2()
 
    print("\n" + "="*70)
    print("STAGE 2 COMPLETE — next step: python 03_compute_shares.py")
    print("="*70)
 
 
if __name__ == "__main__":
    main()