import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

FILE = f"merged_polarity.csv"
df = pd.read_csv(FILE)

label2score = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
plot_df = pd.DataFrame({
    "Manual": df["sentiment_final"].str.lower().map(label2score),
    "VADER":  pd.to_numeric(df["VADER_Compound"], errors="coerce"),
    "RoBERTa-SieBERT":  pd.to_numeric(df["polarity_siebert"],  errors="coerce"),
    "RoBERTa-Twitter":  pd.to_numeric(df["polarity_twitter"],  errors="coerce"),
    "HIV4":   pd.to_numeric(df["HIV4_Polarity"],  errors="coerce"),
})


from sklearn.metrics import classification_report

# Define function to convert score to predicted label
def score_to_label(score):
    if pd.isnull(score):
        return None
    if score > 0.25:
        return "positive"
    elif score < -0.25:
        return "negative"
    else:
        return "neutral"

# Ground truth as string labels
label_map = {-1.0: "negative", 0.0: "neutral", 1.0: "positive"}
ground_truth = plot_df["Manual"].map(label_map)

# Evaluate each source
sources = ["VADER", "RoBERTa-SieBERT", "RoBERTa-Twitter", "HIV4"]
label_order = ["positive", "neutral", "negative"]

for source in sources:
    predictions = plot_df[source].apply(score_to_label)
    
    # Drop rows with NaNs in either column
    valid_mask = ground_truth.notnull() & predictions.notnull()
    y_true = ground_truth[valid_mask]
    y_pred = predictions[valid_mask]
    
    print(f"\n=== {source} ===")
    print(classification_report(
        y_true,
        y_pred,
        labels=label_order,
        zero_division=0
    ))
    
    # Compute confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=label_order
    )
    
    # Plot confusion matrix as heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_order,
        yticklabels=label_order
    )
    plt.title(f"Confusion Matrix for {source}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{source}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion_matrix_{source}.png")

