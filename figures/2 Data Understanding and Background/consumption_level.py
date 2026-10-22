"""Daily mean consumption per location (Section 2.3, "Consumption level and seasonality")."""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("consumption_level.png")

# Categorical slots 1,2,3,6,5,7 of the shared palette, fixed order per location
COLORS = {"oslo": "#2a78d6", "stavanger": "#eb6834", "trondheim": "#1baf7a",
          "helsingfors": "#008300", "bergen": "#e87ba4", "tromsø": "#6250d6"}
LABELS = {"oslo": "Oslo", "stavanger": "Stavanger", "trondheim": "Trondheim",
          "helsingfors": "Helsingfors", "bergen": "Bergen", "tromsø": "Tromsø"}

plt.rcParams.update({
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.5,
    "xtick.color": "#898781", "ytick.color": "#898781",
})

df = pd.read_csv(ROOT / "data" / "consumption_temp.csv", parse_dates=["time"])
# Helsingfors reports 0 MW for its first 32 hours: not delivered, so treated as missing
df["consumption"] = df["consumption"].mask(df["consumption"] <= 0)
daily = (df.set_index("time").groupby("location")["consumption"]
           .resample("D").mean().unstack(level=0))

fig, ax = plt.subplots(figsize=(9, 4.2))
for loc, color in COLORS.items():
    s = daily[loc].dropna()
    ax.plot(s.index, s, color=color, linewidth=1.5)
    ax.annotate(LABELS[loc], (s.index[-1], s.iloc[-7:].mean()), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color="#52514e")
ax.set_ylabel("daily mean consumption (MW)")
ax.set_ylim(bottom=0)
ax.margins(x=0.01)
ax.set_xlim(right=daily.index[-1] + pd.Timedelta(days=45))
ax.set_xticks(pd.date_range("2022-05-01", "2023-04-01", freq="2MS"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.tight_layout()
fig.savefig(OUT, dpi=300)
