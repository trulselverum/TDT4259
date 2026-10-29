"""Consumption against temperature per location (Section 2.3, "Consumption and temperature")."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("temperature_scatter.png")
BLUE = "#2a78d6"
ORDER = ["oslo", "stavanger", "trondheim", "tromsø", "bergen", "helsingfors"]
NAMES = {"oslo": "Oslo", "stavanger": "Stavanger", "trondheim": "Trondheim",
         "tromsø": "Tromsø", "bergen": "Bergen", "helsingfors": "Helsingfors"}

plt.rcParams.update({
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.5,
    "xtick.color": "#898781", "ytick.color": "#898781",
})

df = pd.read_csv(ROOT / "data" / "consumption_temp.csv")
df = df[df["consumption"] > 0]  # 0 MW hours were not delivered

fig, axes = plt.subplots(2, 3, figsize=(9, 4.4), sharex=True)
for ax, loc in zip(axes.flat, ORDER):
    g = df[df["location"] == loc]
    ax.scatter(g["temperature"], g["consumption"], s=3, alpha=0.2, color=BLUE, linewidths=0)
    ax.set_title(NAMES[loc], fontsize=10, color="#52514e")
fig.supxlabel("temperature forecast (°C)", color="#52514e")
fig.supylabel("consumption (MW)", color="#52514e")
fig.tight_layout()
fig.savefig(OUT, dpi=300)
