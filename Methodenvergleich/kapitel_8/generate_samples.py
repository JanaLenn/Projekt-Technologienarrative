# This code generates samples for annotation -> but 200_samples.csv is provided to make sure the sample isn't changed. 

import pandas as pd
import sys

# Output file path will be inside the kapitel_9 folder.
input_csv  = sys.argv[1] if len(sys.argv) > 1 else 'extracted_nuclear_atom_contexts.csv'
output_csv = sys.argv[2] if len(sys.argv) > 2 else '200_samples.csv'

df = pd.read_csv(input_csv, sep=';')

df_first = df.groupby('ID', as_index=False).first()

n_to_sample = min(200, len(df_first))
sample_df = df_first.sample(n=n_to_sample, random_state=42)

sample_df.to_csv(output_csv, sep=';', index=False)

print(f"Sampled {len(sample_df)} contexts from {len(df_first)} articles {output_csv}")
