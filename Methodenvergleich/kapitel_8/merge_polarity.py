from pathlib import Path
import pandas as pd

ANNO_PATH    = Path("annotation_final.csv")
DICT_PATH    = Path("dictionary_results_200.csv")
SIEBERT_PATH = Path("doc_polarity_results_siebert.csv")
TWITTER_PATH = Path("doc_polarity_results_twitter.csv")

# Output CSV file
OUT_PATH     = Path("merged_polarity.csv")
MERGE_HOW    = "inner"


def read_annotation(path: Path) -> pd.DataFrame:
    """Read annotation_final.csv (semicolon-separated)."""
    df = pd.read_csv(path, sep=";")
    return df[["ID", "sentiment_final"]]


def read_dictionary(path: Path) -> pd.DataFrame:
    """Read dictionary_results_200.csv (comma-separated)."""
    df = pd.read_csv(path)
    return df[["ID", "HIV4_Polarity", "VADER_Compound"]]


def read_siebert(path: Path) -> pd.DataFrame:
    """Read doc_polarity_results_siebert.csv (comma-separated) and extract polarity."""
    df = pd.read_csv(path).rename(columns={"doc_id": "ID"}).rename(columns={"polarity": "polarity_siebert"})
    return df[["ID", "polarity_siebert"]]


def read_twitter(path: Path) -> pd.DataFrame:
    """Read doc_polarity_results_twitter.csv (comma-separated) and extract polarity."""
    df = pd.read_csv(path).rename(columns={"doc_id": "ID"}).rename(columns={"polarity": "polarity_twitter"})
    return df[["ID", "polarity_twitter"]]


def merge_sentiment_csv(
    anno_path: Path,
    dict_path: Path,
    siebert_path: Path,
    twitter_path: Path,
    out_path: Path,
    merge_how: str = "inner",
) -> pd.DataFrame:
    anno = read_annotation(anno_path)
    dictionary = read_dictionary(dict_path)
    siebert = read_siebert(siebert_path)
    twitter = read_twitter(twitter_path)

    merged = (
        anno.merge(dictionary, on="ID", how=merge_how)
        .merge(siebert, on="ID", how=merge_how)
        .merge(twitter, on="ID", how=merge_how)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Saved merged file to {out_path} (rows: {len(merged)})")
    return merged


def main() -> None:
    merge_sentiment_csv(
        ANNO_PATH, DICT_PATH, SIEBERT_PATH, TWITTER_PATH, OUT_PATH, MERGE_HOW
    )


if __name__ == "__main__":
    main()
