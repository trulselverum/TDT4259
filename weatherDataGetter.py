"""
Henter timesdata for vind og nedbør (og temperatur) 2022–2024 for seks byer
fra Open-Meteo sitt historiske API (ERA5-reanalyse). Gratis, ingen API-nøkkel.

Bruk:
    pip install requests pandas
    python hent_vaerdata.py
    # valgfritt: slå sammen med egen temperatur-CSV
    python hent_vaerdata.py --merge temperatur.csv --by-kolonne by --tid-kolonne tid

Resultat: vaerdata_2022_2024.csv (én rad per by per time, tid i UTC).
"""
import argparse
import time

import pandas as pd
import requests

BYER = {
    "trondheim":   (63.4305, 10.3951),
    "tromsø":      (69.6496, 18.9560),
    "stavanger":   (58.9700, 5.7331),
    "bergen":      (60.3913, 5.3221),
    "oslo":        (59.9139, 10.7522),
    "helsingfors": (60.1699, 24.9384),
}

VARIABLER = [
    "temperature_2m",       # °C
    "precipitation",        # mm (regn + snø som vannekvivalent)
    "rain",                 # mm
    "snowfall",             # cm
    "wind_speed_10m",       # m/s
    "wind_gusts_10m",       # m/s
    "wind_direction_10m",   # grader
]

URL = "https://archive-api.open-meteo.com/v1/archive"


def hent_by(navn, lat, lon, start, slutt):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": slutt,
        "hourly": ",".join(VARIABLER),
        "wind_speed_unit": "ms",
        "timezone": "GMT",  # UTC unngår doble/manglende timer ved sommertid
    }
    for forsok in range(5):
        r = requests.get(URL, params=params, timeout=120)
        if r.status_code == 429:  # rate limit – vent og prøv igjen
            time.sleep(30 * (forsok + 1))
            continue
        r.raise_for_status()
        df = pd.DataFrame(r.json()["hourly"])
        df.insert(1, "location", navn)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df
    raise RuntimeError(f"Ga opp {navn} etter gjentatte 429-svar")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--slutt", default="2024-12-31")
    p.add_argument("--ut", default="data/vaerdata_2022_2024.csv")
    p.add_argument("--merge", help="Egen CSV med temperatur å slå sammen med")
    p.add_argument("--by-kolonne", default="location")
    p.add_argument("--tid-kolonne", default="time")
    p.add_argument("--tid-er-lokal", action="store_true",
                   help="Sett hvis tidene i din CSV er lokal tid (Oslo/Helsinki), ikke UTC")
    a = p.parse_args()

    deler = []
    for navn, (lat, lon) in BYER.items():
        print(f"Henter {navn} …")
        deler.append(hent_by(navn, lat, lon, a.start, a.slutt))
        time.sleep(2)
    df = pd.concat(deler, ignore_index=True)

    if a.merge:
        egen = pd.read_csv(a.merge)
        tid = pd.to_datetime(egen[a.tid_kolonne])
        if a.tid_er_lokal:
            sone = egen[a.by_kolonne].map(
                lambda b: "Europe/Helsinki" if b == "helsingfors" else "Europe/Oslo")
            tid = pd.Series([
                t.tz_localize(z, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
                for t, z in zip(tid, sone)])
        elif tid.dt.tz is None:
            tid = tid.dt.tz_localize("UTC")
        else:
            tid = tid.dt.tz_convert("UTC")
        egen["time"] = tid
        egen = egen.rename(columns={a.by_kolonne: "location"})
        df = egen.merge(df.drop(columns="temperature_2m"),
                        on=["location", "time"], how="left")

    df.to_csv(a.ut, index=False)
    print(f"Skrev {len(df):,} rader til {a.ut}")


if __name__ == "__main__":
    main()