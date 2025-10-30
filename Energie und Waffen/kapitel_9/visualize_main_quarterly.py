import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = f"sentence_sentiment_with_meta.csv"  # adjust as needed
YEAR_START, YEAR_END = 1980, 2024

df = pd.read_csv(CSV_PATH, sep=None, engine="python", encoding="utf-8-sig")

# Required columns
req = ["Category", "polarity", "Datum_standardisiert"]
missing = [c for c in req if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["Datum_standardisiert"] = pd.to_datetime(df["Datum_standardisiert"], errors="coerce")
df = df.dropna(subset=["Datum_standardisiert", "polarity"])

df["Year"] = df["Datum_standardisiert"].dt.year
df["Quarter"] = df["Datum_standardisiert"].dt.quarter

df = df[(df["Year"] >= YEAR_START) & (df["Year"] <= YEAR_END)]

g = (
    df.groupby(["Year", "Quarter", "Category"])["polarity"]
      .mean()
      .reset_index()
)

pivot = g.pivot(index=["Year", "Quarter"], columns="Category", values="polarity")

years = np.arange(YEAR_START, YEAR_END + 1)
quarters = [1, 2, 3, 4]
full_index = pd.MultiIndex.from_product([years, quarters], names=["Year", "Quarter"])

pivot = pivot.reindex(full_index).sort_index()

E_vals = pivot.get("E", pd.Series(index=pivot.index, dtype=float)).to_numpy()
W_vals = pivot.get("W", pd.Series(index=pivot.index, dtype=float)).to_numpy()

x = np.arange(len(pivot))
bar_width = 0.4

year_for_slot = pivot.index.get_level_values("Year").to_numpy()
quarter_for_slot = pivot.index.get_level_values("Quarter").to_numpy()
xticklabels = [f"'{str(y)[-2:]}" if q == 1 else "" for y, q in zip(year_for_slot, quarter_for_slot)]

# Plot
plt.figure(figsize=(20, 6))
plt.bar(x - bar_width/2, W_vals, width=bar_width, label="W")
plt.bar(x + bar_width/2, E_vals, width=bar_width, label="E")

plt.xticks(x, xticklabels, rotation=0)
plt.xlabel("Year")
plt.ylabel("Average Sentiment")
plt.title("Average Sentiment per Quarter (E vs W), 1980 - 2024")
plt.legend()
plt.tight_layout()
plt.savefig("sentiment_main_quarterly.png")
plt.show()
