"""
Fetches hourly weather (temperature, precipitation and wind) for 2022-2024 for the six locations from the
Open-Meteo historical API (ERA5 reanalysis). Free, no API key. Report Section 2.5, "Possible External Data".

Usage:
    python fetch_weather_data.py

Output: data/weather_2022_2024.csv (one row per location per hour, time in UTC). It is already in the repo, so this
only needs to run to fetch the data again.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

LOCATIONS = {
    "trondheim":   (63.4305, 10.3951),
    "tromsø":      (69.6496, 18.9560),
    "stavanger":   (58.9700, 5.7331),
    "bergen":      (60.3913, 5.3221),
    "oslo":        (59.9139, 10.7522),
    "helsingfors": (60.1699, 24.9384),
}

VARIABLES = [
    "temperature_2m",       # °C
    "precipitation",        # mm (rain + snow as water equivalent)
    "rain",                 # mm
    "snowfall",             # cm
    "wind_speed_10m",       # m/s
    "wind_gusts_10m",       # m/s
    "wind_direction_10m",   # degrees
]

URL = "https://archive-api.open-meteo.com/v1/archive"
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "weather_2022_2024.csv"


def fetch_location(name, lat, lon, start, end):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(VARIABLES),
        "wind_speed_unit": "ms",
        "timezone": "GMT",  # UTC avoids repeated/missing hours at daylight saving changes
    }
    for attempt in range(5):
        r = requests.get(URL, params=params, timeout=120)
        if r.status_code == 429:  # rate limit: wait and try again
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
        df = pd.DataFrame(r.json()["hourly"])
        df.insert(1, "location", name)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df
    raise RuntimeError(f"Gave up on {name} after repeated 429 responses")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--out", default=OUTPUT)
    a = p.parse_args()

    parts = []
    for name, (lat, lon) in LOCATIONS.items():
        print(f"Fetching {name} …")
        parts.append(fetch_location(name, lat, lon, a.start, a.end))
        time.sleep(2)
    df = pd.concat(parts, ignore_index=True)

    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df):,} rows to {a.out}")


if __name__ == "__main__":
    main()
