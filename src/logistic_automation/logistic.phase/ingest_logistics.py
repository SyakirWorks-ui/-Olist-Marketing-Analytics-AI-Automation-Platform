import duckdb
import pandas as pd
import time
from pathlib import Path
from datetime import datetime

# ==========================================
# CONFIGURATION & PATH MANAGEMENT (FIXED)
# ==========================================

# Menggunakan __file__ agar path selalu relatif terhadap lokasi script ini berada
# Script berada di: src/utils/ingest_logistics.py
# Kita perlu naik 2 level ke atas untuk mencapai root project (Olist_Ecommerce_Analytics_Portfolio)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Path disesuaikan dengan struktur folder VS Code Anda
DATA_RAW_PATH = BASE_DIR / "data" / "processed" / "01_olist_master_join_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "production"
OUTPUT_FILE = OUTPUT_DIR / "02_logistics_analytics_data.parquet"

# Pastikan output directory tersedia
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def ingest_and_process_data(source_path: Path, target_path: Path):
    """
    Ingests e-commerce data using DuckDB, filters for logistics columns,
    performs initial feature engineering, and saves to Parquet.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting Logistics Data Ingestion Pipeline...")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📂 Source: {source_path}")

    # ==========================================
    # DUCKDB QUERY ENGINE
    # ==========================================
    # We use DuckDB to:
    # 1. Read CSV efficiently (read_csv_auto)
    # 2. Select ONLY relevant columns (Column Pruning)
    # 3. Cast types immediately (Type Safety)
    # 4. Perform calculation in-engine (High Performance)
    
    query = f"""
    SELECT 
        -- 1. Identifiers
        order_id,
        customer_id,
        seller_id,
        product_id,
        
        -- 2. Order Status & Metrics
        order_status,
        freight_value,
        product_weight_g,
        product_length_cm,
        product_height_cm,
        product_width_cm,
        
        -- 3. Geolocation (Origin & Destination)
        customer_zip_code_prefix,
        customer_city,
        customer_state,
        seller_zip_code_prefix,
        seller_city,
        seller_state,
        
        -- 4. Time Series (Casting to TIMESTAMP for correct Date math)
        CAST(order_purchase_timestamp AS TIMESTAMP) as order_purchase_timestamp,
        CAST(order_approved_at AS TIMESTAMP) as order_approved_at,
        CAST(order_delivered_carrier_date AS TIMESTAMP) as order_delivered_carrier_date,
        CAST(order_delivered_customer_date AS TIMESTAMP) as order_delivered_customer_date,
        CAST(order_estimated_delivery_date AS TIMESTAMP) as order_estimated_delivery_date,
        CAST(shipping_limit_date AS TIMESTAMP) as shipping_limit_date,

        -- 5. Feature Engineering (SQL Based)
        
        -- Actual Lead Time (Days): Time from Purchase to Delivery
        -- Using epoch extraction to get fractional days for precision
        (date_part('epoch', CAST(order_delivered_customer_date AS TIMESTAMP)) - 
         date_part('epoch', CAST(order_purchase_timestamp AS TIMESTAMP))) / 86400.0 
         AS actual_lead_time_days,

        -- Delay Risk (Days): Estimate - Actual. 
        -- Positive value = Delivered Early/On time. Negative value = Late.
        (date_part('epoch', CAST(order_estimated_delivery_date AS TIMESTAMP)) - 
         date_part('epoch', CAST(order_delivered_customer_date AS TIMESTAMP))) / 86400.0 
         AS diff_estimated_vs_actual

    FROM read_csv_auto('{source_path.as_posix()}', header=True)
    
    -- Filter: For logistics modeling, we typically focus on completed deliveries 
    -- to train the model, or we handle 'shipped' items differently. 
    -- For now, we ingest everything but you can uncomment below to filter:
    -- WHERE order_status = 'delivered'
    """

    try:
        # Execute Query and convert to Pandas DataFrame
        # DuckDB -> Arrow -> Pandas is zero-copy in many cases (extremely fast)
        start_time = time.time()
        
        df = duckdb.query(query).to_df()
        
        duration = time.time() - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Ingestion Complete in {duration:.2f} seconds.")

        # ==========================================
        # POST-INGESTION VALIDATION
        # ==========================================
        print(f"\n📊 Data Shape: {df.shape}")
        print("-" * 30)
        print("First 5 rows of Logistics Data:")
        print(df[['order_id', 'actual_lead_time_days', 'diff_estimated_vs_actual', 'customer_state']].head().to_markdown(index=False))
        print("-" * 30)

        # ==========================================
        # SAVING ARTIFACTS
        # ==========================================
        # Parquet is the Gold Standard for Analytics (Columnar storage, preserves schema)
        df.to_parquet(target_path, index=False)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 Data saved to: {target_path}")
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error during ingestion: {e}")
        raise

if __name__ == "__main__":
    # Check if input file exists before running
    if DATA_RAW_PATH.exists():
        ingest_and_process_data(DATA_RAW_PATH, OUTPUT_FILE)
    else:
        print(f"❌ Input file not found at: {DATA_RAW_PATH}")
        print("Please check your file path configuration.")