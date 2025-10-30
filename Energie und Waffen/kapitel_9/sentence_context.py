import re
import sys
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

# Load CSV
csv_path = sys.argv[1] if len(sys.argv) > 1 else f'main_topic.csv'
print(f"Reading articles from: {csv_path}")

df = pd.read_csv(
    csv_path,
    sep=';', 
    engine='python',
    quotechar='"',
    encoding='utf-8'
)

pattern = re.compile(r'\b(nuclear|atom\w*)\b', flags=re.IGNORECASE)

def extract_context_blocks(text, gap_tolerance=1):
    """Extract non-overlapping context blocks around sentences with keyword hits."""
    if not isinstance(text, str):
        return []
    sents = sent_tokenize(text)
    hits  = [i for i, s in enumerate(sents) if pattern.search(s)]
    if not hits:
        return []

    ctxs = []
    block_start = hits[0]
    block_end   = hits[0]

    for idx in hits[1:]:
        if idx <= block_end + gap_tolerance:  
            block_end = idx
        else:
            start = max(block_start - 1, 0)
            end   = min(block_end + 1, len(sents)-1)
            ctxs.append(" ".join(sents[start:end+1]))
            block_start = block_end = idx

    start = max(block_start - 1, 0)
    end   = min(block_end + 1, len(sents)-1)
    ctxs.append(" ".join(sents[start:end+1]))

    return ctxs

df['extracted_text'] = df['Text'].apply(extract_context_blocks)

out = df.explode('extracted_text')

out = out[out['extracted_text'].str.contains(r'\b(?:nuclear|atom\w*)\b', case=False, na=False)]

grouped_texts = out.groupby('ID')['extracted_text'] \
    .apply(lambda x: '\n\n'.join(x)).rename('all_snippets')

non_snippet_cols = out.drop(columns=['extracted_text']).drop_duplicates(subset='ID').set_index('ID')

grouped = pd.concat([non_snippet_cols, grouped_texts], axis=1).reset_index()

if 'Category' in grouped.columns and 'category' not in grouped.columns:
    grouped = grouped.rename(columns={'Category': 'category'})

required_cols = ['ID', 'Datum_standardisiert', 'category', 'all_snippets']
missing = [c for c in required_cols if c not in grouped.columns]

if 'category' in missing and 'Category' in grouped.columns:
    grouped = grouped.rename(columns={'Category': 'category'})
    missing = [c for c in required_cols if c not in grouped.columns]

if missing:
    print(f"Warning: missing expected columns: {missing}")

grouped = grouped[[c for c in required_cols if c in grouped.columns]]

grouped.to_csv(f'snippets_main.csv', sep=';', index=False)
print(f"→ Grouped output saved with {len(grouped)} articles and columns {list(grouped.columns)}.")
