import os
import sys
import json
import joblib
import logging
import re
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. DEFINISI DIRECTORY UNTUK LOGGING ---
# Mendapatkan path absolut dari folder script saat ini
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Membuat folder 'logs' di dalam direktori script tersebut
LOG_DIR = os.path.join(CURRENT_SCRIPT_DIR, 'logs')

# Pastikan folder logs dibuat sebelum logging dimulai
os.makedirs(LOG_DIR, exist_ok=True)

# --- 2. CONFIGURASI LOGGING (AUDIT TRAIL) ---
# PERBAIKAN: Mendefinisikan LOG_DIR terlebih dahulu agar tidak terjadi NameError
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        # Menggunakan encoding='utf-8' untuk mendukung emoji di Windows
        logging.FileHandler(os.path.join(LOG_DIR, 'execution.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("Sistem Logging berhasil diinisialisasi dengan encoding UTF-8.")

# --- PERBAIKAN: Menambahkan Custom Transformer yang dibutuhkan Model untuk Load ---
class ReviewCleaner(BaseEstimator, TransformerMixin):
    """
    Custom Transformer yang digunakan saat training model di Phase 03.
    Wajib didefinisikan ulang agar joblib bisa melakukan deserialisasi model.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Logika pembersihan teks yang konsisten dengan fase R&D
        if isinstance(X, pd.Series):
            return X.apply(self._clean_text)
        else:
            return [self._clean_text(str(text)) for text in X]

    def _clean_text(self, text):
        if not isinstance(text, str):
            return ""
        # 1. Lowercase
        text = text.lower()
        # 2. Hapus karakter non-alfabet (sesuai standar industri)
        text = re.sub(r'[^a-z\s]', '', text)
        # 3. Trim whitespace
        return text.strip()

# --- SETELAH INI BARU MASUK KE CLASS UTAMA ---

class SentimentRecoveryEngine:
    """
    ENGINE OTOMASI SENTIMEN & PEMULIHAN LAYANAN (PLATINUM TIER).
    
    Class ini bertanggung jawab untuk mengelola pipeline end-to-end:
    1. Ingest Data Baru (Transaction Data)
    2. Preprocessing Text & Cleaning
    3. AI Inference (Prediksi Sentimen)
    4. Prescriptive Logic (Penerapan Aturan Bisnis dari Fase 05)
    5. Export Actionable Insights & Reporting
    """

    def __init__(self):
        """
        Inisialisasi Engine.
        Menetapkan jalur dinamis (dynamic paths), memuat konfigurasi bisnis, 
        dan memuat model AI yang telah dilatih.
        """
        logging.info("🚀 Memulai Inisialisasi SentimentRecoveryEngine...")

        try:
            # 1. Definisi Dynamic Absolute Paths (Anti-Hardcode)
            self.current_dir = os.path.dirname(os.path.abspath(__file__))
            self.root_dir = os.path.abspath(os.path.join(self.current_dir, "..", "..", "..")) 
            
            # Asumsi struktur: root/data/research, root/models/sentiment, root/data/production
            self.paths = {
                "config": os.path.join(self.root_dir, "data", "production", "sentiment", "automation_config_refined.json"),
                "model": os.path.join(self.root_dir, "models", "sentiment", "sentiment_predictor.joblib"),
                "raw_data": os.path.join(self.root_dir, "data", "processed", "01_olist_master_join_cleaned.csv"),
                "output_data": os.path.join(self.root_dir, "data", "production", "sentiment", "final_actionable_list.parquet"),
                "output_report": os.path.join(self.root_dir, "data", "production", "sentiment", "daily_execution_summary.json")
            }

            # 2. Validasi & Muat Config JSON (Hasil Lab Phase 05)
            if not os.path.exists(self.paths["config"]):
                raise FileNotFoundError(f"Config file tidak ditemukan di: {self.paths['config']}")
            
            with open(self.paths["config"], 'r') as f:
                self.config = json.load(f)
            logging.info("✅ Konfigurasi Bisnis (Phase 05) berhasil dimuat.")

            # 3. Validasi & Muat Model AI (Hasil Phase 03)
            if not os.path.exists(self.paths["model"]):
                raise FileNotFoundError(f"Model file tidak ditemukan di: {self.paths['model']}")
            
            self.model = joblib.load(self.paths["model"])
            logging.info("✅ Model Sentiment Predictor (Phase 03) berhasil dimuat.")

            # Pastikan folder output tersedia
            os.makedirs(os.path.dirname(self.paths["output_data"]), exist_ok=True)

        except Exception as e:
            logging.critical(f"⛔ CRITICAL ERROR saat Init: {str(e)}")
            sys.exit(1) # Hentikan program jika init gagal

    def _preprocess_text(self, text):
        """
        Membersihkan teks ulasan pelanggan.
        Logika ini HARUS SAMA PERSIS dengan yang digunakan saat training model (Phase 02/03)
        untuk menjaga konsistensi akurasi.
        """
        if not isinstance(text, str):
            return ""
        
        # 1. Lowercase
        text = text.lower()
        # 2. Hapus karakter non-alfabet (tanda baca & angka)
        text = re.sub(r'[^a-z\s]', '', text)
        # 3. Hapus whitespace berlebih
        text = text.strip()
        
        return text

    def run_inference(self, df):
        """
        Menjalankan prediksi sentimen menggunakan model yang sudah dilatih.
        """
        logging.info("🧠 Menjalankan AI Inference (Prediksi Sentimen)...")
        
        try:
            # Pastikan kolom target ada, isi NaN dengan string kosong
            target_col = 'review_comment_message'
            if target_col not in df.columns:
                raise ValueError(f"Kolom '{target_col}' tidak ditemukan dalam data input.")
            
            df[target_col] = df[target_col].fillna("")

            # Proses Preprocessing (Vectorized apply)
            logging.info("   ...Preprocessing teks ulasan")
            df['clean_text'] = df[target_col].apply(self._preprocess_text)

            # Prediksi Label & Probabilitas
            logging.info("   ...Melakukan prediksi batch")
            df['predicted_sentiment'] = self.model.predict(df['clean_text'])
            
            # Mendapatkan probabilitas (confidence score)
            # Asumsi model memiliki method predict_proba
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(df['clean_text'])
                df['sentiment_confidence'] = np.max(probs, axis=1)
            else:
                df['sentiment_confidence'] = 1.0 # Default jika model tidak support probability

            return df

        except Exception as e:
            logging.error(f"❌ Error pada Inference Module: {str(e)}")
            raise

    def apply_prescriptive_logic(self, df):
        """
        Menerapkan 'Business Rules' yang telah divalidasi di Experiment Lab (Phase 05).
        Menentukan tindakan pemulihan (Recovery Action) dan estimasi dampak finansial.
        """
        logging.info("⚖️ Menerapkan Prescriptive Logic (Business Rules)...")

        try:
            # Ambil parameter dari JSON Config
            THRESHOLD_SCORE = self.config.get("optimal_threshold", 2) # Default 2 jika config gagal baca
            MIN_PRICE_VOUCHER = self.config.get("min_price_for_voucher", 100)
            
            # --- 1. Identifikasi High Priority Cases ---
            # Kondisi: Sentimen Negatif DAN Review Score <= Threshold (Biasanya 1 atau 2)
            # Kita gunakan numpy where untuk kecepatan (Vectorized Operation)
            
            # Normalisasi kolom review_score
            df['review_score'] = pd.to_numeric(df['review_score'], errors='coerce').fillna(5)
            df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)

            # Kondisi Logika
            is_negative_sentiment = df['predicted_sentiment'] == 'Negative'
            is_low_score = df['review_score'] <= THRESHOLD_SCORE
            is_high_value = df['price'] >= MIN_PRICE_VOUCHER

            # --- 2. Penetapan Action Plan ---
            conditions = [
                (is_negative_sentiment & is_low_score & is_high_value), # KASUS KRITIS & BERNILAI TINGGI
                (is_negative_sentiment & is_low_score),                 # KASUS KRITIS BIASA
                (is_negative_sentiment)                                 # KELUHAN RINGAN
            ]
            
            choices = [
                "Auto-Refund + 15% Voucher (Priority)",
                "Personal Apology + 5% Voucher",
                "Automated Apology Email"
            ]
            
            df['recovery_action'] = np.select(conditions, choices, default="No Action Needed")

            # --- 3. Estimasi Financial Impact (ROI Projection) ---
            # Revenue at Risk = Harga barang pada customer yang kecewa berat
            df['revenue_at_risk'] = np.where(
                (is_negative_sentiment & is_low_score), 
                df['price'], 
                0
            )

            return df

        except Exception as e:
            logging.error(f"❌ Error pada Prescriptive Module: {str(e)}")
            raise

    def execute_pipeline(self):
        """
        Orkestrator Utama: Menggabungkan semua tahapan menjadi satu aliran eksekusi.
        Data Ingest -> Preprocessing -> Inference -> Action -> Export.
        """
        logging.info("▶️ START PIPELINE EXECUTION")
        start_time = datetime.now()
        
        try:
            # STEP 1: LOAD DATA
            logging.info(f"📂 Membaca Raw Data dari: {self.paths['raw_data']}")
            if not os.path.exists(self.paths['raw_data']):
                raise FileNotFoundError("Raw data file tidak ditemukan.")
            
            # Membaca chunk kecil atau full dataset (disesuaikan kebutuhan memory)
            # Di sini kita load full karena analisis historis, tapi di real-time bisa batch
            df_raw = pd.read_csv(self.paths['raw_data'])
            logging.info(f"   Data dimuat: {len(df_raw):,} baris.")

            # STEP 2 & 3: INFERENCE
            df_scored = self.run_inference(df_raw)

            # STEP 4: BUSINESS LOGIC
            df_final = self.apply_prescriptive_logic(df_scored)

            # STEP 5: EXPORT HASIL (Production Grade)
            logging.info(f"💾 Menyimpan hasil ke Parquet: {self.paths['output_data']}")
            
            # Konversi kolom object ke string untuk kompatibilitas PyArrow
            for col in df_final.columns:
                if df_final[col].dtype == 'object':
                    df_final[col] = df_final[col].astype(str)
            
            df_final.to_parquet(self.paths['output_data'], index=False, engine='pyarrow')

            # --- TAMBAHAN UNTUK EXCEL USER (Update Disini) ---
            csv_path = self.paths['output_data'].replace('.parquet', '.csv')
            logging.info(f"📊 Menyimpan salinan Excel (CSV): {csv_path}")
            df_final.to_csv(csv_path, index=False)
            # -------------------------------------------------

            # STEP 6: SUMMARY REPORTING
            # Menghitung metrik ringkasan untuk monitoring harian
            summary_metrics = {
                "execution_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_processed": int(len(df_final)),
                "negative_sentiment_detected": int(df_final[df_final['predicted_sentiment'] == 'Negative'].shape[0]),
                "critical_recovery_actions": int(df_final[df_final['recovery_action'].str.contains('Priority')].shape[0]),
                "total_revenue_at_risk": float(df_final['revenue_at_risk'].sum()),
                "status": "SUCCESS"
            }

            # Simpan report JSON
            with open(self.paths['output_report'], 'w') as f:
                json.dump(summary_metrics, f, indent=4)
            
            logging.info("📊 DAILY SUMMARY REPORT:")
            logging.info(json.dumps(summary_metrics, indent=2))

        except Exception as e:
            logging.error(f"🔥 PIPELINE FAILED: {str(e)}")
            # Optional: Kirim notifikasi email/slack kepada tim data di sini
            sys.exit(1)
        
        finally:
            duration = datetime.now() - start_time
            logging.info(f"🏁 Execution Finished. Duration: {duration}")

# --- ENTRY POINT ---
if __name__ == "__main__":
    # Instansiasi Engine dan Jalankan Pipeline
    engine = SentimentRecoveryEngine()
    engine.execute_pipeline()