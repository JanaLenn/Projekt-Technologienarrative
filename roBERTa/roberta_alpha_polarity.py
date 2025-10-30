import numpy as np
import pandas as pd
import nltk, torch, matplotlib.pyplot as plt
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import sys

# ----------------------------- 0. NLTK tokenizer
nltk.download("punkt", quiet=True)

# ----------------------------- 1. Load CSV
print("Reading CSV…")
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH   = SCRIPT_DIR / "snippets_sentence_level.csv"   

try:
    df = pd.read_csv(
        CSV_PATH,
        sep=";",
        header=0,
        on_bad_lines="skip",
        keep_default_na=False,
        encoding="utf-8"
    )
except Exception as e:
    print(f"Failed to read CSV: {e}")
    sys.exit(1)

print("✓ CSV loaded. Columns:", list(df.columns))

TEXT_COL = "text"
ID_COL   = "ID"                   
if TEXT_COL not in df.columns:
    raise ValueError(f"Expected column “{TEXT_COL}” in CSV.")

if ID_COL not in df.columns:
    print(f"Column “{ID_COL}” not found. Falling back to row index as ID.")
    df[ID_COL] = df.index

# ----------------------------- 2. Load model
MODEL_NAME = "cardiffnlp/twitter-roberta-base"
print("Loading model…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
roberta   = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = 0 if torch.cuda.is_available() else -1
sent_pipe = pipeline(
    "sentiment-analysis",
    model=roberta,
    tokenizer=tokenizer,
    device=device,
    return_all_scores=True,
    truncation=True,
    max_length=256
)

label_index = {v.lower(): k for k, v in roberta.config.id2label.items()}
print("✓ Model ready. Labels:", label_index)

# ----------------------------- 3. Analyse sentences with checkpointing
print("Scoring sentences…")
records = []
CHECKPOINT_EVERY = 5000
CHECKPOINT_PATH = SCRIPT_DIR / "sentence_sentiment_partial.csv"

try:
    for doc_id, text in tqdm(zip(df[ID_COL], df[TEXT_COL]), total=len(df), desc="Documents"):
        if not text or not isinstance(text, str):
            continue

        sentences = nltk.sent_tokenize(text)
        if not sentences:
            continue

        outputs = sent_pipe(sentences, batch_size=16)

        for sent, out in zip(sentences, outputs):
            
            scores = {d["label"].lower(): d["score"] for d in out}
            p_pos = scores.get("positive", 0.0)
            p_neg = scores.get("negative", 0.0)

            
            records.append(
                dict(doc_id=doc_id, sentence=sent,
                     p_positive=p_pos, p_negative=p_neg)
            )

        # Save checkpoint periodically
        if len(records) % CHECKPOINT_EVERY < len(sentences):
            pd.DataFrame(records).to_csv(CHECKPOINT_PATH, index=False, encoding="utf-8")
            print(f"✓ Checkpoint saved with {len(records):,} records → {CHECKPOINT_PATH}")

except KeyboardInterrupt:
    print("\n Interrupted by user. Saving partial results…")
    if records:
        pd.DataFrame(records).to_csv(CHECKPOINT_PATH, index=False, encoding="utf-8")
    sys.exit(1)
except Exception as e:
    print(f"\n Error during processing: {e}")
    if records:
        pd.DataFrame(records).to_csv(CHECKPOINT_PATH, index=False, encoding="utf-8")
    sys.exit(1)

print(f"✓ Scored {len(records):,} sentences from {len(df):,} documents.")

# ----------------------------- 4. Sentence-level extras
sent_df = pd.DataFrame(records)
sent_df["sentence_polarity"] = (sent_df.p_positive - sent_df.p_negative) / (
    sent_df.p_positive + sent_df.p_negative + 1e-12
)

# ----------------------------- 5. Document-level polarity
agg = sent_df.groupby("doc_id")[["p_positive", "p_negative"]].sum()
agg["polarity"] = (agg.p_positive - agg.p_negative) / (
    agg.p_positive + agg.p_negative + 1e-12
)
doc_polarity_df = agg.reset_index()

print("Doc-level polarity stats:",
      f"mean={doc_polarity_df.polarity.mean():+.3f}, "
      f"sd={doc_polarity_df.polarity.std():.3f}, "
      f"min={doc_polarity_df.polarity.min():+.3f}, "
      f"max={doc_polarity_df.polarity.max():+.3f}")

# ----------------------------- 6. Histogram
doc_polarity_df.polarity.hist(bins=40, figsize=(6, 3))
plt.title("Document Polarity (RoBERTa)")
plt.xlabel("(Σpos - Σneg) / (Σpos + Σneg)")
plt.ylabel("# documents")
plt.tight_layout()
plt.savefig("RoBERTa_Twitter_Polarity.png", bbox_inches="tight", dpi=600)
plt.show()

# ----------------------------- 7. Save outputs
SENT_OUT = SCRIPT_DIR / "sentence_sentiment_twitter.csv"
DOC_OUT  = SCRIPT_DIR / "doc_polarity_results_twitter.csv"
sent_df.to_csv(SENT_OUT, index=False, encoding="utf-8")
doc_polarity_df.to_csv(DOC_OUT, index=False, encoding="utf-8")

print("✓ Sentence results →", SENT_OUT)
print("✓ Document polarity →", DOC_OUT)
