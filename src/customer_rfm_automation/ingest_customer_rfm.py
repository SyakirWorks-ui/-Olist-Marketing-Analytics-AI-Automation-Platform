import duckdb
import pandas as pd
import logging
import sys
from pathlib import Path

# Konfigurasi Logging dengan encoding UTF-8 untuk kompatibilitas Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RFM_Ingestion")

class RFMIngestor:
    def __init__(self):
        self.root = self._find_project_root()
        self.source_path = self.root / "data" / "processed" / "01_olist_master_join_cleaned.csv"
        self.output_dir = self.root / "data" / "production"
        self.output_file = self.output_dir / "03_customer_rfm_features.parquet"

    def _find_project_root(self, marker='models'):
        """Mencari root project secara dinamis untuk fleksibilitas eksekusi"""
        current_path = Path(__file__).resolve()
        for parent in current_path.parents:
            if (parent / marker).exists():
                return parent
        return Path.cwd()

    def run_ingestion(self):
        """Menjalankan proses Ingestion dan RFM Calculation menggunakan DuckDB"""
        logger.info("--- MEMULAI PROSES INGESTION CUSTOMER RFM ---")
        
        if not self.source_path.exists():
            logger.error(f"File sumber tidak ditemukan: {self.source_path}")
            sys.exit(1)

        try:
            # Menggunakan DuckDB Relation API untuk efisiensi tinggi
            con = duckdb.connect()
            
            logger.info(f"Mengekstraksi data dari: {self.source_path.name}")
            
            # Query SQL untuk Recency, Frequency, dan Monetary (Platinum Logic)
            # Menangani NULL pada payment_value dan konversi timestamp secara otomatis
            rfm_query = f"""
                WITH base_data AS (
                    SELECT 
                        customer_unique_id,
                        order_id,
                        CAST(order_purchase_timestamp AS TIMESTAMP) as purchase_date,
                        COALESCE(payment_value, 0) as payment
                    FROM read_csv_auto('{self.source_path}')
                ),
                reference_date AS (
                    SELECT MAX(purchase_date) + INTERVAL '1 day' as target_date FROM base_data
                )
                SELECT 
                    customer_unique_id,
                    DATE_DIFF('day', MAX(purchase_date), (SELECT target_date FROM reference_date)) as recency,
                    COUNT(DISTINCT order_id) as frequency,
                    SUM(payment) as monetary
                FROM base_data
                GROUP BY customer_unique_id
            """

            # Eksekusi query dan konversi ke DataFrame
            rfm_df = con.execute(rfm_query).df()
            
            # Persistensi Data ke format Parquet (Gold Standard)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            rfm_df.to_parquet(self.output_file, index=False)
            
            self._generate_summary(rfm_df)
            
        except Exception as e:
            logger.error(f"Gagal dalam proses ingestion: {str(e)}")
            sys.exit(1)

    def _generate_summary(self, df):
        """Audit Summary untuk verifikasi integritas hasil"""
        total_customers = len(df)
        total_revenue = df['monetary'].sum()
        
        logger.info("--- SUMMARY AUDIT ---")
        logger.info(f"Total Customer Berhasil Diolah: {total_customers:,}")
        logger.info(f"Total Revenue Terakumulasi: R$ {total_revenue:,.2f}")
        logger.info(f"Output disimpan di: {self.output_file}")
        logger.info("--- INGESTION SELESAI DENGAN SUKSES ---")

if __name__ == "__main__":
    ingestor = RFMIngestor()
    ingestor.run_ingestion()