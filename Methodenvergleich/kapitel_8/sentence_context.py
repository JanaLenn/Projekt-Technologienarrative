# This code extracts only the sentences mentioning "atom" or "nuclear" as well as the sentence before and after. 

import re
import sys
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

csv_path = sys.argv[1] if len(sys.argv) > 1 else 'artikel_korpus_fin_updated.csv'
print(f"Reading articles from: {csv_path}")
df = pd.read_csv(
    csv_path,
    sep=';',
    engine='python',
    quotechar='"',
    encoding='utf-8'
)

df = df.reset_index().rename(columns={'index':'article_index'})

pattern = re.compile(r'\b(nuclear|atom\w*)\b', flags=re.IGNORECASE)

def extract_context_blocks(text):
    if not isinstance(text, str):
        return []
    sents = sent_tokenize(text)
    hits  = [i for i, s in enumerate(sents) if pattern.search(s)]
    ctxs  = []
    for idx in hits:
        start = max(idx-1, 0)
        end   = min(idx+1, len(sents)-1)
        while start>0 and pattern.search(sents[start]):
            start -= 1
        while end<len(sents)-1 and pattern.search(sents[end]):
            end += 1
        ctxs.append(" ".join(sents[start:end+1]))
    return ctxs

df['contexts'] = df['Text'].apply(extract_context_blocks)

out = (
    df
    .loc[:, ['ID', 'contexts']]
    .explode('contexts')
    .rename(columns={'contexts':'extracted_text'})
    .drop_duplicates()
)
out = out[out['extracted_text'].str.contains(r'\b(nuclear|atom\w*)\b', case=False, na=False)]

# 8) write with semicolons
out.to_csv('extracted_nuclear_atom_contexts_fin.csv', sep=';', index=False)
print(f"Processed {len(df)} unique articles; extracted {len(out)} context blocks")
