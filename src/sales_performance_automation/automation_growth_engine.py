import pandas as pd
import joblib
import os
import logging
import sys
from datetime import datetime

# =================================================================
# 1. WINDOWS COMPATIBILITY & UNICODE (Solusi image_27145b.png)
# =================================================================
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# =================================================================
# 2. PATH RESOLUTION (Solusi image_271cb2.png & image_27910f.png)
# =================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def safe_path(rel_path):
    return os.path.normpath(os.path.join(BASE_DIR, rel_path))

MODEL_PATH = safe_path('../../outputs/models/growth_predictive_model.pkl')
SCALER_PATH = safe_path('../../outputs/models/feature_scaler.pkl')
DATA_INPUT = safe_path('../../data/processed/01_olist_master_join_cleaned.csv')
REPORT_DIR = safe_path('../../outputs/reports/')
LOG_DIR = safe_path('../../logs/')

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"prod_run_{datetime.now().strftime('%Y%m%d')}.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class GrowthAnalyticsEngine:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_features = None
        self._load_assets()

    def _load_assets(self):
        """Memuat assets dan mengunci metadata fitur model."""
        try:
            if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
                raise FileNotFoundError(f"Model/Scaler tidak ditemukan di: {MODEL_PATH}")
            
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            # KUNCI: Mengambil 102 nama fitur asli dari model (Solusi image_262ff4.png)
            self.model_features = list(self.model.feature_names_in_)
            
            logging.info(f"✅ Assets Loaded: Model siap dengan {len(self.model_features)} fitur.")
        except Exception as e:
            logging.error(f"❌ Critical Load Error: {e}")
            sys.exit(1)

    def preprocess_and_align(self, df_raw):
        """Feature Engineering & Sinkronisasi Fitur 100% (Strict Alignment)."""
        try:
            logging.info("⚙️ Menjalankan Automated Feature Engineering...")
            df = df_raw.copy()

            # 1. Business Logic
            df['freight_ratio'] = df['freight_value'] / df['price']
            north_states = ['AM', 'RR', 'AP', 'PA', 'AC', 'RO', 'TO']
            df['is_north_region'] = df['customer_state'].apply(lambda x: 1 if x in north_states else 0)

            # 2. One-Hot Encoding
            df_enc = pd.get_dummies(df, columns=['customer_state', 'product_category_name_english'])

            # 3. KUNCI RATING 100%: Reindex & Alignment
            # Memastikan kolom tepat 102, mengisi kolom yang tidak ada dengan 0, 
            # dan menghapus kolom baru yang tidak dikenal model.
            X_final = df_enc.reindex(columns=self.model_features, fill_value=0)
            
            logging.info("✅ Feature Alignment selesai (102 kolom disinkronkan).")
            return df, X_final
        except Exception as e:
            logging.error(f"❌ Preprocessing Error: {e}")
            return df_raw, pd.DataFrame()

    def run_inference(self, df_base, X_input):
        """Menjalankan Prediksi & Prescriptive Analysis."""
        if X_input.empty: return pd.DataFrame()
        try:
            logging.info("🔮 Menjalankan Prescriptive Modeling...")
            # Scaling dengan fitur yang sudah disejajarkan
            X_scaled = self.scaler.transform(X_input)
            df_base['predicted_volume'] = self.model.predict(X_scaled)

            # Kalkulasi Dampak Bisnis
            df_base['revenue_upside'] = (df_base['predicted_volume'] * df_base['price']) - df_base['price']
            return df_base
        except Exception as e:
            logging.error(f"❌ Inference Error: {e}")
            return pd.DataFrame()

    def export_report(self, df):
        """Export laporan agregasi akhir."""
        if df.empty or 'revenue_upside' not in df.columns:
            logging.warning("⚠️ Data kosong/tidak valid. Export dibatalkan.")
            return
        try:
            os.makedirs(REPORT_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            path = os.path.join(REPORT_DIR, f"olist_growth_report_{ts}.csv")
            
            summary = df.groupby(['customer_state', 'product_category_name_english']).agg({
                'predicted_volume': 'sum',
                'revenue_upside': 'sum'
            }).reset_index()

            summary.to_csv(path, index=False)
            logging.info(f"🚀 SUCCESS: Laporan disimpan di {path}")
        except Exception as e:
            logging.error(f"❌ Export Error: {e}")

# =================================================================
# 3. MAIN EXECUTION
# =================================================================
if __name__ == "__main__":
    engine = GrowthAnalyticsEngine()
    
    if os.path.exists(DATA_INPUT):
        raw_data = pd.read_csv(DATA_INPUT)
        logging.info(f"📊 Membaca data: {len(raw_data)} baris.")
        
        df_p, X_i = engine.preprocess_and_align(raw_data)
        df_f = engine.run_inference(df_p, X_i)
        engine.export_report(df_f)
    else:
        logging.error(f"❌ Data Input tidak ditemukan: {DATA_INPUT}")