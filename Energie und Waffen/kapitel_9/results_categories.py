import pandas as pd

# === Settings ===
CSV_PATH = f"sentence_sentiment_with_meta.csv"  # adjust to your file

# === Load CSV (auto-detect delimiter) ===
df = pd.read_csv(CSV_PATH, sep=None, engine="python", encoding="utf-8-sig")

# --- Check required columns ---
required = ["Category", "Datum_standardisiert"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# --- Convert dates ---
df["Datum_standardisiert"] = pd.to_datetime(df["Datum_standardisiert"], errors="coerce")

# --- Drop rows without category or date ---
df = df.dropna(subset=["Category", "Datum_standardisiert"]).copy()

# --- Extract Year ---
df["Year"] = df["Datum_standardisiert"].dt.year

# 1) Total articles per category
total_per_category = df["Category"].value_counts().reset_index()
total_per_category.columns = ["Category", "Article_Count"]

# 2) Articles per category per year
per_year_category = df.groupby(["Year", "Category"]).size().reset_index(name="Article_Count")

# === Output results ===
print("=== Total articles per category ===")
print(total_per_category.to_string(index=False))

print("\n=== Articles per category per year ===")
print(per_year_category.sort_values(["Year", "Category"]).to_string(index=False))

# === Optional: save to CSV ===
total_per_category.to_csv(f"total_per_category.csv", index=False)
per_year_category.to_csv(f"total_category_per_year.csv", index=False)
print("\n✓ Results saved to CSV files.")
