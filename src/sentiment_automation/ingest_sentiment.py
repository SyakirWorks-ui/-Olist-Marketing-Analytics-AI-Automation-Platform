import pandas as pd
import spacy
import re

def run_ingestion(input_path, output_path):
    df = pd.read_csv(input_path)
    # Filtering and Cleaning
    df_clean = df.dropna(subset=['review_comment_message']).copy()
    
v    
    def fast_clean(text):
        text = re.sub(r'[^a-zA-Záàâãéèêíïóôõöúçñ\s]', '', text.lower())
        return " ".join([t.lemma_ for t in nlp(text) if not t.is_stop])

    df_clean['processed_text'] = df_clean['review_comment_message'].apply(fast_clean)
    
    # Save as Parquet for performance
    df_clean.to_parquet(output_path, compression='snappy')
    print(f"Ingestion Complete: {output_path}")

if __name__ == "__main__":
    run_ingestion('01_olist_master_join_cleaned.csv', 'sentiment_staging_data.parquet')