import pandas as pd

df = pd.read_csv(f"sentence_sentiment_with_meta.csv", sep=None, engine="python")

def classify_sentiment(score):
    if score < -0.25:
        return "negative"
    elif score > 0.25:
        return "positive"
    else:
        return "neutral"

df["sentiment_category"] = df["polarity"].apply(classify_sentiment)

counts_by_cat = df.groupby(["Category", "sentiment_category"]).size().unstack(fill_value=0)

print(counts_by_cat)
