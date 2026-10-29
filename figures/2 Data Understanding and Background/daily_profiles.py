"""Average daily load profile, weekday vs weekend (Section 2.3, "Daily and weekly patterns")."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("daily_profiles.png")
BLUE, ORANGE = "#2a78d6", "#eb6834"
ORDER = ["oslo", "stavanger", "trondheim", "tromsø", "bergen", "helsingfors"]
NAMES = {"oslo": "Oslo", "stavanger": "Stavanger", "trondheim": "Trondheim",
         "tromsø": "Tromsø", "bergen": "Bergen", "helsingfors": "Helsingfors"}
TZ = {"helsingfors": "Europe/Helsinki"}

plt.rcParams.update({
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.5,
    "xtick.color": "#898781", "ytick.color": "#898781",
})

df = pd.read_csv(ROOT / "data" / "consumption_temp.csv", parse_dates=["time"])
df = df[df["consumption"] > 0]

fig, axes = plt.subplots(2, 3, figsize=(9, 4.4), sharex=True, sharey=True)
for ax, loc in zip(axes.flat, ORDER):
    g = df[df["location"] == loc].copy()
    local = g["time"].dt.tz_localize("UTC").dt.tz_convert(TZ.get(loc, "Europe/Oslo"))
    g["hour"], g["weekend"] = local.dt.hour, local.dt.dayofweek >= 5
    # Divide by the location's mean so the shape is comparable across portfolio sizes
    g["rel"] = g["consumption"] / g["consumption"].mean()
    prof = g.groupby(["weekend", "hour"])["rel"].mean().unstack(0)
    ax.plot(prof.index, prof[False], color=BLUE, lw=1.8, label="weekday")
    ax.plot(prof.index, prof[True], color=ORANGE, lw=1.8, label="weekend")
    ax.set_title(NAMES[loc], fontsize=10, color="#52514e")
    ax.set_xticks([0, 6, 12, 18, 23])
axes[0, 0].legend(frameon=False, fontsize=8)
fig.supxlabel("hour of day (local time)", color="#52514e")
fig.supylabel("consumption relative to location mean", color="#52514e")
fig.tight_layout()
fig.savefig(OUT, dpi=300)
