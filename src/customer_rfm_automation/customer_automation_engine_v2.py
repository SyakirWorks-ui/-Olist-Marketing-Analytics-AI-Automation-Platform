"""
DOCSTRING: Enterprise Automation Engine for Olist RFM Segmentation.
VERSION: 1.0.0 (Production)
AUTHOR: Principal MDA Specialist
"""

import logging  # Library untuk logging sistem
import sys  # Library untuk interaksi dengan interpreter Python
import json  # Library untuk memproses file JSON
import joblib  # Library untuk memuat model machine learning (.joblib)
import pandas as pd  # Library untuk manipulasi dataframe
import numpy as np  # Library untuk operasi numerik
import duckdb  # Library database OLAP untuk pemrosesan data cepat
from pathlib import Path  # Library untuk manajemen path sistem operasi
from datetime import datetime  # Library untuk manajemen waktu dan tanggal
from dataclasses import dataclass  # Library untuk struktur data konfigurasi
from sklearn.metrics import silhouette_score  # Library untuk evaluasi kualitas cluster
from statsmodels.stats.power import TTestIndPower  # Library untuk analisis statistik sample size

# --- CONFIGURATION LAYER ---
@dataclass
class AppConfig:
    """Kelas untuk menyimpan konfigurasi global secara absolut menggunakan pathlib."""
    # Menentukan root directory proyek secara dinamis agar aman di OS apapun
    ROOT_DIR: Path = Path(__file__).parent.resolve()
    
    # Path untuk file input (Data Master, Model, dan Katalog Strategi)
    MASTER_DATA: Path = ROOT_DIR / "data" / "processed" / "01_olist_master_join_cleaned.csv"
    MODEL_PATH: Path = ROOT_DIR / "models" / "customer_rfm" / "rfm_kmeans_model.joblib"
    CATALOG_PATH: Path = ROOT_DIR / "models" / "customer_rfm" / "segment_action_catalog.json"
    
    # Path untuk file output (Hasil Produksi dan Log Audit)
    PROD_OUTPUT_DIR: Path = ROOT_DIR / "data" / "production" / "customer_rfm"
    PROD_OUTPUT_FILE: Path = PROD_OUTPUT_DIR / "customer_segmentation_results.parquet"
    AUDIT_LOG_PATH: Path = ROOT_DIR / "logs" / "automation_history.csv"

# --- CORE ENGINE LAYER ---
class RFMProductionEngine:
    """Mesin utama untuk mengotomasi siklus hidup RFM secara end-to-end."""
    
    def __init__(self):
        """Inisialisasi logging dan koneksi database internal."""
        # Konfigurasi standar logging ke dalam stdout (konsol)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)  # Inisialisasi logger untuk kelas ini
        self.config = AppConfig()  # Memuat konfigurasi path
        self.db = duckdb.connect()  # Membuka koneksi DuckDB (in-memory)
        self.logger.info("Automation Engine Berhasil Diinisialisasi.")

    def ingest_data(self) -> pd.DataFrame:
        """Tier 1: Descriptive - Ingesti data cepat menggunakan SQL DuckDB."""
        self.logger.info("Tahap 1: Memulai Ingesti Data via DuckDB...")
        # Query SQL untuk agregasi RFM secara langsung dari file CSV
        query = f"""
            SELECT 
                customer_unique_id,
                CAST(MAX(order_purchase_timestamp) AS DATE) as last_purchase,
                COUNT(DISTINCT order_id) as Frequency,
                SUM(payment_value) as Monetary
            FROM read_csv_auto('{self.config.MASTER_DATA}')
            GROUP BY 1
        """
        df = self.db.execute(query).df()  # Menjalankan query dan mengonversi ke DataFrame
        self.logger.info(f"Ingesti Selesai. Total baris unik: {len(df)}")
        return df

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tier 2: Diagnostic - Pembersihan data dan kalkulasi fitur Recency."""
        self.logger.info("Tahap 2: Pembersihan Data dan Feature Engineering...")
        # Mengonversi kolom tanggal ke format datetime
        df['last_purchase'] = pd.to_datetime(df['last_purchase'])
        # Menentukan tanggal referensi (pembelian terbaru di database)
        reference_date = df['last_purchase'].max()
        # Menghitung selisih hari sebagai fitur Recency
        df['Recency'] = (reference_date - df['last_purchase']).dt.days
        
        # Penanganan data kosong (NaN) secara eksplisit
        initial_rows = len(df)
        df = df.dropna(subset=['Recency', 'Frequency', 'Monetary'])
        dropped_rows = initial_rows - len(df)
        
        # Memberikan peringatan jika ada data yang dibuang karena NaN
        if dropped_rows > 0:
            self.logger.warning(f"Ditemukan {dropped_rows} baris NaN yang telah dihapus.")
        
        return df

    def predict_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tier 3 & 4: Predictive & Prescriptive - Scoring dan Pemetaan Strategi."""
        self.logger.info("Tahap 3: Menjalankan Prediksi Cluster dan Pemetaan Strategi...")
        
        # Memuat model yang sudah dilatih (Joblib)
        model = joblib.load(self.config.MODEL_PATH)
        # Memuat katalog aksi strategi (JSON)
        with open(self.config.CATALOG_PATH, 'r') as f:
            catalog = json.load(f)
        
        # ALINEASI SKEMA: Menyamakan nama kolom dengan yang diharapkan model (R_scaled, dst)
        X = df[['Recency', 'Frequency', 'Monetary']].copy()
        X.columns = ['R_scaled', 'F_scaled', 'M_scaled']
        
        # Melakukan prediksi cluster menggunakan model KMeans
        df['Cluster'] = model.predict(X)
        
        # Mengambil label segmen dari kunci utama di katalog JSON
        segment_labels = list(catalog.keys())
        
        # Memetakan angka cluster ke nama segmen dan rencana aksi bisnis
        df['Segment'] = df['Cluster'].apply(lambda x: segment_labels[x] if x < len(segment_labels) else "Others")
        df['Action_Plan'] = df['Segment'].map(lambda x: catalog.get(x, {}).get('strategy', 'No Action Required'))
        
        return df

    def validate_quality(self, df: pd.DataFrame):
        """QA Gate: Melakukan audit stabilitas cluster dan kalkulasi sample size."""
        self.logger.info("Tahap 4: Menjalankan QA Gate & Analisis Statistik...")
        # Menyiapkan fitur untuk perhitungan skor silhouette
        features = df[['Recency', 'Frequency', 'Monetary']]
        # Menghitung Silhouette Score (Stabilitas Cluster)
        score = silhouette_score(features, df['Cluster'], sample_size=10000)
        
        # Kalkulasi Power Analysis untuk merekomendasikan jumlah sample A/B Testing
        power_analysis = TTestIndPower()
        req_sample = power_analysis.solve_power(effect_size=0.2, alpha=0.05, power=0.8)
        
        # Log hasil audit agar terekam di sistem
        self.logger.info(f"Audit Selesai. Stability Score: {score:.4f} | Req Sample Size: {int(np.ceil(req_sample))}")
        return score, int(np.ceil(req_sample))

    def persist_data(self, df: pd.DataFrame, score: float, sample_size: int):
        """Persistence Layer: Penyimpanan hasil ke format Parquet dan pembaruan log audit."""
        self.logger.info("Tahap 5: Menyimpan Hasil ke Storage Produksi...")
        # Memastikan direktori output sudah tersedia
        self.config.PROD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.config.AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Menyimpan dataframe ke format Parquet dengan kompresi Snappy untuk efisiensi
        df.to_parquet(self.config.PROD_OUTPUT_FILE, compression='snappy', index=False)
        
        # Menyiapkan data untuk log audit historis
        audit_entry = pd.DataFrame([{
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_records': len(df),
            'silhouette_stability': round(score, 4),
            'recommended_sample_size': sample_size,
            'status': 'SUCCESS'
        }])
        
        # Menambahkan baris baru ke file CSV log audit tanpa menghapus data lama
        audit_entry.to_csv(self.config.AUDIT_LOG_PATH, mode='a', header=not self.config.AUDIT_LOG_PATH.exists(), index=False)
        self.logger.info(f"Produksi Selesai. Data disimpan di: {self.config.PROD_OUTPUT_FILE}")

    def run_pipeline(self):
        """Orkestrator Utama: Menjalankan semua tahap secara berurutan."""
        try:
            raw_data = self.ingest_data()  # Jalankan Ingesti
            processed_data = self.process_features(raw_data)  # Jalankan Cleaning
            final_data = self.predict_segments(processed_data)  # Jalankan Prediksi
            stability_score, req_sample = self.validate_quality(final_data)  # Jalankan QA Audit
            self.persist_data(final_data, stability_score, req_sample)  # Jalankan Penyimpanan
            
            self.logger.info("PIPELINE RFM BERHASIL DIJALANKAN 100%.")
            sys.exit(0)  # Exit dengan status sukses
        except Exception as e:
            # Menangkap semua error dan mencatat traceback lengkap ke log
            self.logger.error(f"PIPELINE GAGAL: {str(e)}", exc_info=True)
            sys.exit(1)  # Exit dengan status error untuk trigger alert di Cron/Airflow

# --- EXECUTION ENTRY POINT ---
if __name__ == "__main__":
    # Instansiasi mesin dan jalankan seluruh pipeline
    engine = RFMProductionEngine()
    engine.run_pipeline()