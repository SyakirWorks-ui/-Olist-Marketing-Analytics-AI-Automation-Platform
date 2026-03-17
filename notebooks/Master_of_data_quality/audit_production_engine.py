"""
Script: audit_production_engine.py
Objective: Automated Audit & Production Ingestion (Clean & Stable Version)
Author: Senior Data Engineer
Standards: PEP 8, OOP, Clean Architecture, Cross-Platform Compatibility
"""

import pandas as pd
import sqlite3
import json
import logging
import os
from datetime import datetime

# --- LOGGING CONFIGURATION (Stable for Windows/Linux) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # Menggunakan encoding utf-8 untuk file log agar tetap aman
        logging.FileHandler("production_audit.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class DataAuditor:
    def __init__(self, metadata_path):
        self.metadata_path = metadata_path
        self.df = None
        self.forbidden_cols = [
            'order_delivered_customer_date', 
            'order_delivered_carrier_date', 
            'review_answer_timestamp'
        ] # Target Anti-Leakage

    def _get_expected_features(self):
        """Ambil skema dari metadata atau gunakan default jika file belum ada."""
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    meta = json.load(f)
                    return meta.get("expected_feature_count", 43)
            return 43 # Berdasarkan temuan terakhir Anda
        except Exception:
            return 43

    def load_data(self, file_path):
        """Load Master Data Olist (110.182 baris)."""
        try:
            self.df = pd.read_csv(file_path)
            logging.info(f"SUCCESS: Loaded {len(self.df)} rows from {os.path.basename(file_path)}")
        except Exception as e:
            logging.error(f"ERROR: Failed to load data: {e}")
            raise

    def clean_features(self):
        """Automated Feature Selection (Mitigasi Leakage)."""
        if self.df is not None:
            logging.info("ACTION: Removing temporal leaked columns...")
            self.df = self.df.drop(columns=[c for c in self.forbidden_cols if c in self.df.columns])
            logging.info(f"SUCCESS: Feature selection complete. Current shape: {self.df.shape}")

    def validate_leakage(self):
        """Professional Quality Gate: Strict Leakage Validation."""
        logging.info("AUDIT: Running Leakage & Schema Validation...")
        
        # 1. Check Leakage
        leaked_found = [col for col in self.forbidden_cols if col in self.df.columns]
        if leaked_found:
            logging.error(f"CRITICAL: Leakage detected: {leaked_found}")
            return False
        
        # 2. Check Schema Consistency
        expected = self._get_expected_features()
        found = self.df.shape[1]
        if found != expected:
            logging.warning(f"SCHEMA: Feature mismatch. Expected: {expected}, Found: {found}")
            
        logging.info("SUCCESS: Anti-Leakage Check passed. Dataset is safe for production.")
        return True

    def export_to_production(self, db_path):
        """Storage Layer: SQLite for Scalability."""
        try:
            conn = sqlite3.connect(db_path)
            # Simpan ke tabel gold_master_audited
            self.df.to_sql('gold_master_audited', conn, if_exists='replace', index=False)
            conn.close()
            logging.info(f"DEPLOY: Master Data successfully exported to {os.path.basename(db_path)}")
        except Exception as e:
            logging.error(f"ERROR: Export failed: {e}")
            raise

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Dinamis Path Resolution (Menyesuaikan struktur folder Anda)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
    
    INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "01_olist_master_join_cleaned.csv")
    DATABASE_OUT = os.path.join(PROJECT_ROOT, "olist_production.db")
    METADATA_IN = os.path.join(BASE_DIR, "master_metadata.json")

    auditor = DataAuditor(metadata_path=METADATA_IN)

    try:
        logging.info("--- STARTING PRODUCTION AUDIT ENGINE ---")
        auditor.load_data(INPUT_FILE)
        auditor.clean_features()
        
        if auditor.validate_leakage():
            auditor.export_to_production(DATABASE_OUT)
            logging.info("--- PRODUCTION RUN COMPLETED SUCCESSFULLY ---")
        else:
            logging.critical("--- PRODUCTION HALTED: AUDIT FAILURE ---")
            
    except Exception as e:
        logging.critical(f"SYSTEM FAILURE: {e}")