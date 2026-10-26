"""Pairwise correlation between attributes (Section 3.2, "Data Preparation and Feature Engineering")."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("attribute_correlation_heatmap.png")
COLS = ["temperature", "precipitation", "rain", "snowfall", "wind_speed_10m",
        "consumption_6days_ago", "consumption_7days_ago", "consumption_14days_ago"]
LABELS = ["Consumption", "Temperature", "Precipitation", "Rain", "Snowfall", "Wind speed",
          "Lag 6 d", "Lag 7 d", "Lag 14 d"]

df = pd.read_csv(ROOT / "data" / "dataset.csv")
df = df[df["consumption"] > 0]  # 0 MW hours were not delivered

# correlation within each location, then averaged over locations
corr = sum(g[["consumption"] + COLS].corr() for _, g in df.groupby("location")) / df["location"].nunique()
corr = corr.to_numpy()

# lower triangle without the diagonal
mask = np.triu(np.ones_like(corr, dtype=bool))
shown = np.ma.masked_array(corr, mask)[1:, :-1]

fig, ax = plt.subplots(figsize=(7.6, 6.0))
im = ax.imshow(shown, cmap="RdBu_r", vmin=-1, vmax=1)
for i in range(shown.shape[0]):
    for j in range(shown.shape[1]):
        if not shown.mask[i, j]:
            v = shown[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.6 else "#2b2a27")
ax.set_xticks(range(len(LABELS) - 1), LABELS[:-1], rotation=30, ha="right")
ax.set_yticks(range(len(LABELS) - 1), LABELS[1:])
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax)
cb.set_label("Pearson correlation $r$")
cb.outline.set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=300)
