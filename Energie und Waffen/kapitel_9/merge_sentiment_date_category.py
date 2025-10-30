import pandas as pd

main_topic_path = f"main_topic.csv"
sentiment_path = f"polarity_results_main.csv"

df_main = pd.read_csv(main_topic_path, sep=None, engine="python")
df_sentiment = pd.read_csv(sentiment_path, sep=None, engine="python")

# Select only the needed columns from main_topic
df_main_reduced = df_main[["ID", "Datum_standardisiert", "Category"]]

if "doc_id" in df_sentiment.columns and "ID" not in df_sentiment.columns:
    df_sentiment = df_sentiment.rename(columns={"doc_id": "ID"})

# Merge on ID
merged_df = pd.merge(df_sentiment, df_main_reduced, on="ID", how="left")

# Save
merged_df.to_csv(f"sentence_sentiment_with_meta.csv", index=False)

print("Merge complete. File saved as sentence_sentiment_with_meta.csv")
print(merged_df.head())
