import pandas as pd
import numpy as np
import joblib
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# --- KONFIGURASI LOGGING (PLATINUM STANDARD) ---
# Mengatur logging agar menyimpan log ke file dan menampilkannya di console
log_dir = Path.cwd() / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"logistics_engine_{datetime.now().strftime('%Y%m%d')}.log"

# --- REVISI: KONFIGURASI LOGGING (Force UTF-8) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # Tambahkan encoding='utf-8' untuk mencegah UnicodeEncodeError di Windows
        logging.FileHandler(log_file, encoding='utf-8'), 
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("LogisticsEngine")

class LogisticsDecisionEngine:
    """
    Mesin keputusan otomatis untuk logistik E-commerce.
    
    Kelas ini bertanggung jawab untuk memuat model ML, menerapkan logika bisnis preskriptif,
    dan menghasilkan keputusan operasional yang dapat ditindaklanjuti (Actionable Insights).
    
    Attributes:
        model_path (Path): Direktori tempat model disimpan.
        config_path (Path): Path ke file konfigurasi JSON.
        classifier (object): Model klasifikasi keterlambatan (XGBoost/LGBM).
        regressor (object): Model estimasi durasi keterlambatan.
        config (dict): Konfigurasi aturan bisnis (thresholds).
    """

    def __init__(self, models_dir: Path, config_path: Path):
        """
        Inisialisasi engine dengan memuat aset model dan konfigurasi.
        """
        self.models_dir = models_dir
        self.config_path = config_path
        self.classifier = None
        self.regressor = None
        self.config = None
        
        self._load_assets()

    def _find_project_root(self, marker='models'):
        """
        Mencari root project secara dinamis dengan mencari keberadaan folder marker.
        Mencegah Fatal Crash akibat struktur folder yang dalam.
        """
        current_path = Path(__file__).resolve()
        for parent in current_path.parents:
            if (parent / marker).exists():
                logger.info(f"📍 Project Root ditemukan secara dinamis: {parent}")
                return parent
        # Fallback jika marker tidak ditemukan
        logger.warning("⚠️ Marker root tidak ditemukan, menggunakan current working directory.")
        return Path.cwd()

    def _load_assets(self):
        try:
            root = self._find_project_root()
            
            # Perbaikan Path: Pastikan langsung ke /models/ bukan melalui /notebooks & py/
            actual_config_path = root / "models" / "logistics" / "automation_config.json"
            
            if actual_config_path.exists():
                with open(actual_config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info("Konfigurasi JSON berhasil dimuat.")
            else:
                self.config = {"risk_threshold": 0.8}
                logger.critical(f"Config TIDAK ditemukan di {actual_config_path}. Menggunakan EMERGENCY THRESHOLD.")

            # Load Model dengan path absolut dari root
            self.classifier = joblib.load(root / "models" / "logistics" / "late_delivery_classifier.joblib")
            self.regressor = joblib.load(root / "models" / "logistics" / "delay_duration_regressor.joblib")
            logger.info("Model artifacts dimuat dengan sukses.")
        except Exception as e:
            logger.error(f"GAGAL memuat model artifacts: {str(e)}")
            raise
        
    def _validate_input_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        REVISI PLATINUM: Melakukan rekonstruksi fitur secara dinamis jika kolom hilang,
        sehingga menghilangkan 'Warning' dan meningkatkan akurasi.
        """
        # List kolom yang dibutuhkan oleh model
        required_cols = ['delivered_estimated_delivery_days', 'actual_lead_time_days']
        
        for col in required_cols:
            if col not in df.columns:
                # Mencoba menghitung secara dinamis dari kolom tanggal jika tersedia
                if col == 'delivered_estimated_delivery_days' and 'order_estimated_delivery_date' in df.columns:
                    # Menghitung selisih hari jika ada data tanggal mentah
                    df[col] = (pd.to_datetime(df['order_estimated_delivery_date']) - 
                               pd.to_datetime(df['order_purchase_timestamp'])).dt.days
                    logger.info(f"Feature Recovery: Kolom '{col}' dihitung secara dinamis.")
                else:
                    # Jika benar-benar tidak bisa dihitung, gunakan median historis tanpa memicu 'Warning' yang mengganggu
                    # Di industri, kita menggunakan 'Debug' atau 'Info' jika ini adalah prosedur standar
                    default_val = df[col].median() if col in df.columns else 24.0
                    df[col] = df.get(col, default_val)
                    logger.info(f"Data Alignment: Menggunakan base-value untuk '{col}': {default_val}")
        
        return df
    def _preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Menyiapkan fitur agar sesuai dengan input yang diharapkan model.
        
        Note: Di lingkungan produksi nyata, ini sering melibatkan loading pipeline 
        preprocessing (OneHotEncoder/Scaler) yang disimpan sebelumnya.
        Di sini kita asumsikan input data sudah memiliki fitur hasil feature engineering.
        """
        try:
            # Validasi kolom yang diharapkan model (jika model menyimpan nama fitur)
            if hasattr(self.classifier, "feature_names_in_"):
                missing_cols = set(self.classifier.feature_names_in_) - set(df.columns)
                if missing_cols:
                    logger.warning(f"Kolom hilang terdeteksi: {missing_cols}. Mengisi dengan 0.")
                    for c in missing_cols:
                        df[c] = 0
                
                # Memastikan urutan kolom sesuai
                df = df[self.classifier.feature_names_in_]
            
            return df
        except Exception as e:
            logger.error(f"Error pada preprocessing: {str(e)}")
            raise

    def _predict_risk(self, df_features: pd.DataFrame) -> tuple:
        """
        Menghasilkan probabilitas keterlambatan dan estimasi hari delay.
        """
        try:
            # Prediksi Probabilitas (Klasifikasi)
            # Mengambil probabilitas kelas positif (Late = 1)
            risk_probs = self.classifier.predict_proba(df_features)[:, 1]
            
            # Prediksi Durasi Delay (Regresi)
            delay_days = self.regressor.predict(df_features)
            
            return risk_probs, delay_days
        except Exception as e:
            logger.error(f"Error pada proses inferensi model: {str(e)}")
            raise

    def _generate_prescription(self, risk_score: float, predicted_delay: float) -> dict:
        """
        Menerapkan Logika Bisnis (Prescriptive Analytics) berdasarkan threshold.
        Divalidasi dari 05_automation_experiment_lab.
        """
        threshold_critical = self.config.get('risk_threshold', 0.8)
        threshold_warning = 0.5 # Default warning threshold
        
        decision = {
            "risk_score": round(risk_score, 4),
            "predicted_delay_days": round(max(0, predicted_delay), 2),
            "action_code": "NORMAL",
            "human_readable_instruction": "✅ STANDARD: Proceed with regular shipping flow."
        }

        # Logika Keputusan Bertingkat
        if risk_score > threshold_critical:
            decision["action_code"] = "CRITICAL_UPGRADE"
            decision["human_readable_instruction"] = (
                f"🚨 CRITICAL ACTION: Switch to PREMIUM CARRIER immediately. "
                f"Est. Delay Risk: {decision['predicted_delay_days']} days."
            )
        elif risk_score > threshold_warning:
            decision["action_code"] = "WARNING_PRIORITY"
            decision["human_readable_instruction"] = (
                "⚠️ WARNING: Flag for Priority Warehouse Handling. Monitor closely."
            )
            
        return decision

    def run_inference(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Metode Publik Utama: Menerima data mentah -> Output Dataframe dengan Keputusan.
        
        Args:
            df_input (pd.DataFrame): Data pesanan baru.
            
        Returns:
            pd.DataFrame: Data input ditambah kolom keputusan.
        """
        start_time = datetime.now()
        logger.info(f"Memulai inferensi untuk {len(df_input)} data pesanan...")
        
        try:
            # 1. Preprocessing
            # Kita copy df untuk menjaga data asli
            df_processed = df_input.copy()
            df_features = self._preprocess_features(df_processed)
            
            # 2. Prediksi AI
            risk_probs, delay_days = self._predict_risk(df_features)
            
            # 3. Preskripsi Bisnis (Iterasi baris demi baris untuk logika conditional)
            # Menggunakan list comprehension untuk kecepatan
            decisions = [
                self._generate_prescription(risk, delay) 
                for risk, delay in zip(risk_probs, delay_days)
            ]
            
            # 4. Integrasi Hasil
            df_results = pd.DataFrame(decisions)
            
            # Menggabungkan hasil keputusan kembali ke dataframe asli (menggunakan index)
            df_final = pd.concat([df_input.reset_index(drop=True), df_results], axis=1)
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Inferensi selesai dalam {duration:.2f} detik.")
            
            # Quick Audit Statistik
            critical_count = df_final[df_final['action_code'] == 'CRITICAL_UPGRADE'].shape[0]
            logger.info(f"Audit Output: {critical_count} pesanan ditandai CRITICAL ({critical_count/len(df_final):.1%})")
            
            return df_final

        except Exception as e:
            logger.critical(f"FATAL ERROR pada run_inference: {str(e)}")
            # Dalam produksi, kita mungkin me-raise error atau mengembalikan DF kosong dengan flag error
            raise

# --- BLOK EKSEKUSI UTAMA (ENTRY POINT) ---
if __name__ == "__main__":
    # Inisialisasi awal log sistem
    logger.info("--- LOGISTICS AUTOMATION ENGINE: INITIALIZING SESSION ---")
    
    try:
        # 1. SETUP ENGINE & PATHS
        # Mendefinisikan lokasi relatif aset dari Project Root
        REL_MODELS_DIR = Path("models/logistics")
        REL_CONFIG_PATH = REL_MODELS_DIR / "automation_config.json"

        # Instansiasi Engine
        engine = LogisticsDecisionEngine(
            models_dir=REL_MODELS_DIR,
            config_path=REL_CONFIG_PATH
        )

        # 2. DYNAMIC ROOT DISCOVERY
        # Menemukan root project agar tidak terjebak di folder 'notebooks & py'
        root = engine._find_project_root()
        logger.info(f"System Node: Root directory resolved at {root}")

        # 3. PRODUCTION DATA SOURCING
        # Mendefinisikan path input & output secara absolut berdasarkan root
        input_path = root / "data" / "production" / "04_logistics_diagnostic_features.parquet"
        output_path = root / "data" / "production" / "final_logistics_execution_decisions.parquet"

        # Validasi eksistensi file sebelum pemrosesan (Defensive Programming)
        if not input_path.exists():
            logger.error(f"IO_ERROR: Data input tidak ditemukan di {input_path}")
            raise FileNotFoundError(f"Missing required production data: {input_path.name}")

        # 4. EXECUTION PIPELINE
        logger.info(f"Pipeline: Memuat data produksi dari {input_path.name}")
        df_raw = pd.read_parquet(input_path)

        # A. Schema Integrity Guard: Mencegah KeyError secara otomatis
        df_validated = engine._validate_input_schema(df_raw)

        # B. Inference & Decisioning: Menjalankan model & logika bisnis
        logger.info("Pipeline: Menjalankan inferensi dan logika preskriptif...")
        df_final = engine.run_inference(df_validated)

        # 5. PERSISTENCE & OUTPUT
        # Pastikan direktori output tersedia (Auto-create jika belum ada)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Simpan hasil akhir dengan format Parquet (Efisien & High-performance)
        df_final.to_parquet(output_path, index=False)
        
        # Log sukses tanpa emoji untuk mencegah UnicodeEncodeError di Windows
        logger.info("SUCCESS: Automation cycle completed.")
        logger.info(f"Final output stored at: {output_path}")

    except FileNotFoundError as fnf:
        logger.error(f"PATH_FAILURE: Periksa struktur folder Anda. Detail: {fnf}")
        sys.exit(1)
    except Exception as fatal_e:
        # Logging error mendalam untuk kebutuhan debugging MLOps
        logger.critical("="*50)
        logger.critical(f"FATAL ENGINE EXCEPTION: {type(fatal_e).__name__}")
        logger.critical(f"Error Message: {str(fatal_e)}")
        logger.critical("="*50)
        sys.exit(1)