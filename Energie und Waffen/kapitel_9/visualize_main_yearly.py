import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = f"sentence_sentiment_with_meta.csv"
YEAR_START, YEAR_END = 1980, 2024

df = pd.read_csv(CSV_PATH, sep=None, engine="python", encoding="utf-8-sig")

# Required columns
req = ["Category", "polarity", "Datum_standardisiert"]
missing = [c for c in req if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Parse dates + keep valid rows
df["Datum_standardisiert"] = pd.to_datetime(df["Datum_standardisiert"], errors="coerce")
df = df.dropna(subset=["Datum_standardisiert", "polarity"])

# Year
df["Year"] = df["Datum_standardisiert"].dt.year
df = df[(df["Year"] >= YEAR_START) & (df["Year"] <= YEAR_END)]

# Group by Year and Category, compute mean polarity
g = (
    df.groupby(["Year", "Category"])["polarity"]
      .mean()
      .reset_index()
)

# Pivot for plotting
pivot = g.pivot(index="Year", columns="Category", values="polarity")

years = np.arange(YEAR_START, YEAR_END + 1)
pivot = pivot.reindex(years)

E_vals = pivot.get("E", pd.Series(index=pivot.index, dtype=float)).to_numpy()
W_vals = pivot.get("W", pd.Series(index=pivot.index, dtype=float)).to_numpy()

x = np.arange(len(years))
bar_width = 0.4

xticklabels = [f"'{str(y)[-2:]}" for y in years]

# Plot
plt.figure(figsize=(20, 6))
plt.bar(x - bar_width/2, W_vals, width=bar_width, label="W")
plt.bar(x + bar_width/2, E_vals, width=bar_width, label="E")

plt.xticks(x, xticklabels, rotation=0)
plt.xlabel("Year")
plt.ylabel("Average Sentiment")
plt.title("Average Sentiment per Year (E vs W), 1980 - 2024")
plt.legend()
plt.tight_layout()
plt.savefig("sentiment_main_yearly.png", bbox_inches="tight", dpi=600)
plt.show()
