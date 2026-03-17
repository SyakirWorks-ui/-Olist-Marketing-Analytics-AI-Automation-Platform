import os
import duckdb

# --- DYNAMIC PATH CONFIGURATION (PROFESSIONAL SETUP) ---
# Menentukan lokasi script saat ini
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Menaikkan direktori sebanyak 3 tingkat untuk mencapai 'Olist_Ecommerce_Analytics_Portfolio'
# 1: 04_sentiment_analysis, 2: notebooks & py, 3: Olist_Ecommerce_Analytics_Portfolio
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))

# Mendefinisikan Path secara Absolut
INPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', '01_olist_master_join_cleaned.csv')
OUTPUT_STAGING = os.path.join(PROJECT_ROOT, 'data', 'production', 'sentiment', '04_sentiment_staging.parquet')

def run_targeted_ingestion():
    # Validasi keberadaan file sebelum running DuckDB
    if not os.path.exists(INPUT_PATH):
        print(f"❌ CRITICAL ERROR: File tidak ditemukan di:")
        print(f"   👉 {INPUT_PATH}")
        print("\n💡 Tips: Pastikan file CSV ada di folder 'data/processed' di dalam root project Anda.")
        return

    print("🚀 Initializing DuckDB In-Process Ingestion...")
    print(f"📂 Source: {INPUT_PATH}")
    
    # Inisialisasi koneksi DuckDB
    con = duckdb.connect(database=':memory:')
    
    # Gunakan path absolut yang sudah dibersihkan untuk SQL
    # replace('\\', '/') penting untuk kompatibilitas path Windows di DuckDB
    clean_input_path = INPUT_PATH.replace('\\', '/')
    clean_output_path = OUTPUT_STAGING.replace('\\', '/')

    ingestion_query = f"""
        SELECT 
            order_id, 
            customer_id, 
            review_id, 
            review_score, 
            review_comment_message, 
            review_creation_date, 
            product_category_name
        FROM read_csv_auto('{clean_input_path}')
        WHERE review_comment_message IS NOT NULL 
          AND trim(review_comment_message) != ''
    """
    
    try:
        # Eksekusi Ingestion & Export ke Parquet
        con.execute(f"COPY ({ingestion_query}) TO '{clean_output_path}' (FORMAT PARQUET)")
        
        # Audit Sederhana
        row_count = con.execute(f"SELECT COUNT(*) FROM '{clean_output_path}'").fetchone()[0]
        print(f"✅ Ingestion Success! {row_count:,} baris siap dianalisis.")
        print(f"💾 File Staging: {OUTPUT_STAGING}")
        
    except Exception as e:
        print(f"❌ Error saat proses DuckDB: {e}")
    finally:
        con.close()

if __name__ == "__main__":
    run_targeted_ingestion()