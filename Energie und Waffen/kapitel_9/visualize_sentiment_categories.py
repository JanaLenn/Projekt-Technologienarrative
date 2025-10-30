import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def read_auto(path, usecols=None):
    return pd.read_csv(
        path,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
        usecols=usecols
    )

p_in    = "sentence_sentiment_with_meta.csv"
p_plot  = "sentiment_E_W.png"

df = read_auto(p_in)

need = ["Category", "polarity"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in {p_in}: {missing}")

if "ID" in df.columns:
    df["ID"] = df["ID"].astype(str).str.strip()

df_E = df[df["Category"] == "E"]
df_W = df[df["Category"] == "W"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

axes[0].hist(df_E["polarity"].dropna(), bins=40, edgecolor="black")
axes[0].set_title("Sentiment for Category E")
axes[0].set_xlabel("P(positive) − P(negative)")
axes[0].set_ylabel("# documents")

axes[1].hist(df_W["polarity"].dropna(), bins=40, edgecolor="black")
axes[1].set_title("Sentiment for Category W")
axes[1].set_xlabel("P(positive) − P(negative)")

plt.tight_layout()
Path(p_plot).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(p_plot, bbox_inches="tight", dpi=300)
plt.show()

print(f"Loaded everything from {p_in}")
print(f"Saved plots to {p_plot}")
