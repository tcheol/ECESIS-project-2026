import requests
import pandas as pd
import polars as pl
import time
from pathlib import Path

ZONE_COORDS = {
    "COAS": (29.76,-95.37),
    "EAST": (32.35,-94.86),
    "FWES": (32.75,-97.33),
    "NCEN": (32.78,-96.80),
    "NOTH": (32.91,-98.49),   # was "NRTH" — zone_load.parquet uses "NOTH", join was silently NaN
    "SCEN": (30.27,-97.74),
    "SOUT": (29.42,-98.49),
    "WEST": (32.84,-102.36)
}

def fetch_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m", "relativehumidity_2m", "dewpoint_2m", "apparent_temperature", "windspeed_10m","cloudcover","precipitation",
        ],
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "timezone": "America/Chicago",
    }

    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date
    df['he'] = df['time'].dt.hour + 1

    return df

def heat_index(temp_f, humidity):
    hi = (
        -42.379 + 2.04901523 * temp_f + 10.14333127 * humidity
        - 0.22475541 * temp_f * humidity
        - 0.00683783 * temp_f**2 - 0.05481717 * humidity**2
        + 0.00122874 * temp_f**2 * humidity
        + 0.00085282 * temp_f * humidity**2
        - 0.00000199 * temp_f**2 * humidity**2
    )

    return pd.Series(
        [h if t >= 80 else t for h, t in zip(hi, temp_f)],
        index=temp_f.index,
    )

def main(start_date="2022-01-01", end_date="2025-12-31"):
    Path('data').mkdir(exist_ok=True)
    all_dfs = []

    for zone, (lat,lon) in ZONE_COORDS.items():
        print(f"Pulling {zone}...")
        df = fetch_weather(lat,lon,start_date, end_date)

        df["zone_name"] = zone
        df['heat_index'] = heat_index(df['temperature_2m'],df['relativehumidity_2m'])
        df['hdh'] = (65 - df['temperature_2m']).clip(lower=0)
        df['cdh'] = (df['temperature_2m'] - 65).clip(lower=0)
        all_dfs.append(df)
        time.sleep(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])

    pl.from_pandas(combined).write_parquet('data/weather.parquet')
    print(f"\nSaved {len(combined):,} rows to data/weather.parquet")

if __name__ == "__main__":
    main()


