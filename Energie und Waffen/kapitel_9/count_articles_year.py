import pandas as pd
from pathlib import Path

IN_PATH = Path("sentence_sentiment_with_meta.csv")
OUT_PIVOT = Path("articles_per_year_category.csv")
OUT_LONG  = Path("articles_per_year_category_long.csv")

df = pd.read_csv("sentence_sentiment_with_meta.csv", sep=",", engine="python", encoding="utf-8", on_bad_lines="skip")

date_col = "Datum_standardisiert"
cat_col  = "Category"

if date_col is None:
    raise ValueError(f"Could not find a date column. Found: {list(df.columns)}")
if cat_col is None:
    raise ValueError(f"Could not find a category column. Found: {list(df.columns)}")

df["__parsed_date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")

df["Year"] = df["__parsed_date"].dt.year
df = df.dropna(subset=["Year"]).copy()
df["Year"] = df["Year"].astype(int)

counts_long = (
    df.groupby(["Year", cat_col])
      .size()
      .reset_index(name="Count")
      .sort_values(["Year", cat_col])
)

# Pivot table
pivot_counts = (
    counts_long.pivot(index="Year", columns=cat_col, values="Count")
               .fillna(0)
               .astype(int)
               .sort_index()
)

# Save
pivot_counts.to_csv(OUT_PIVOT)
counts_long.to_csv(OUT_LONG, index=False)

# Print
print("Articles per Year x Category (pivot):")
print(pivot_counts.tail(15))  # show last years as a check
print(f"\nSaved wide table -> {OUT_PIVOT}")
print(f"Saved long table  -> {OUT_LONG}")
