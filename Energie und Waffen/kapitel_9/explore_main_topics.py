import pandas as pd

df = pd.read_csv(f'main_topic.csv', sep=';')

# Ensure datetime format
df['Datum_standardisiert'] = pd.to_datetime(df['Datum_standardisiert'], errors='coerce')
df['year'] = df['Datum_standardisiert'].dt.year
articles_per_year = df['year'].value_counts().sort_index()

print("\nArticles per Year:")
print(articles_per_year)

topic_col = 'Subject' if 'Subject' in df.columns else 'Themen'

# Handle multiple topics per article
df[topic_col] = df[topic_col].fillna('').astype(str)
all_topics = df[topic_col].str.split(';').explode().str.strip()
articles_per_topic = all_topics.value_counts()

print("\nArticles per Topic:")
print(articles_per_topic)
