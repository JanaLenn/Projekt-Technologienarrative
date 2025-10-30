"""
Categorise articles by examining their **first three subjects** and assigning
one of four categories:

* **W**      Weapons-related dominates
* **E**      Energy-related dominates
* **SPLIT**  Both W and E appear and the highest percentages are equal
* **IRR**    Neither W nor E in the first three subjects (these rows are dropped)

Rules
-----
1. If only one of the three subjects is W *or* E → that category.
2. If two or more subjects for an article are W (or E) → that category.
3. If both W and E are present → choose the category whose *single highest percentage* wins; if tied → **SPLIT**.

Input  : `artikel_korpus_fin_updated.csv`  (semicolon-delimited)
Output : `articles_categorised_main.csv` (semicolon-delimited, sorted by Category)
"""
import re
import pandas as pd
from typing import List

WEAPON_KEYWORDS: List[str] = [
    "NUCLEAR WEAPONS",
]

ENERGY_KEYWORDS: List[str] = [
    "ENERGY",
]

PERCENT_RE = re.compile(r"\((\d{1,3})%\)")
BRACKET_TAIL_RE = re.compile(r"\s*\(.*?\)\s*$")
SPLIT_RE = re.compile(r'[;,]')


def extract_percentage(label: str) -> int:
    """Return percentage integer in label or 0 if absent."""
    m = PERCENT_RE.search(label)
    return int(m.group(1)) if m else 0


def base_subject(label: str) -> str:
    """Return uppercase subject string with brackets removed."""
    return BRACKET_TAIL_RE.sub('', label).strip().upper()


def classify(base: str) -> str:
    """Return 'W', 'E', or 'X'."""
    for kw in WEAPON_KEYWORDS:
        if kw in base:
            return 'W'
    for kw in ENERGY_KEYWORDS:
        if kw in base:
            return 'E'
    return 'X'


def decide_category(tokens):
    """Apply decision rules to first-three token list. Returns category or None."""
    first3 = tokens[:3]
    w_scores, e_scores = [], []

    for tok in first3:
        base = base_subject(tok)
        cat = classify(base)
        pct = extract_percentage(tok)
        if cat == 'W':
            w_scores.append(pct)
        elif cat == 'E':
            e_scores.append(pct)

    # Only one category present (rules 1 & 2)
    if w_scores and not e_scores:
        return 'W'
    if e_scores and not w_scores:
        return 'E'

    # Both present (rule 3)
    if w_scores and e_scores:
        max_w = max(w_scores)
        max_e = max(e_scores)
        if max_w > max_e:
            return 'W'
        elif max_e > max_w:
            return 'E'
        else:
            return 'SPLIT'

    # Irrelevant
    return None


def main():
    in_file = f'artikel_korpus_fin_updated.csv'
    out_file = f'main_topic.csv'

    try:
        df = pd.read_csv(
            in_file,
            sep=';',
            engine='python',
            on_bad_lines='skip',
            encoding='utf-8'
        )
    except Exception as e:
        print(f"Error reading {in_file}: {e}")
        return

    subj_col = next((c for c in df.columns if c.lower() == 'subject'), None)
    if not subj_col:
        print("No 'Subject' column found. Columns:", ', '.join(df.columns))
        return

    categories = []
    for subj_entry in df[subj_col].fillna(''):
        tokens = [s.strip() for s in SPLIT_RE.split(str(subj_entry)) if s.strip()]
        cat = decide_category(tokens)
        categories.append(cat if cat else 'IRR')

    df['Category'] = categories

    # Keep only W, E, SPLIT
    filtered_df = df[df['Category'].isin(['W', 'E', 'SPLIT'])].copy()

    if filtered_df.empty:
        print("No articles met the categorisation rules.")
        return

    # Sort by Category with explicit order
    cat_order = pd.CategoricalDtype(['W', 'E', 'SPLIT'], ordered=True)
    filtered_df['Category'] = filtered_df['Category'].astype(cat_order)
    filtered_df = filtered_df.sort_values('Category')

    try:
        filtered_df.to_csv(out_file, sep=';', index=False, encoding='utf-8')
        print(f"Saved {len(filtered_df)} categorised articles to '{out_file}'.")
    except Exception as e:
        print(f"Error writing {out_file}: {e}")


if __name__ == '__main__':
    main()
