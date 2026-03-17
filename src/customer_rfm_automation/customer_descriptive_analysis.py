import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
import sys
from pathlib import Path

# ==========================================
# 1. KONFIGURASI SISTEM & LOGGING
# ==========================================
# Mengatur logging profesional dengan standar UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RFM_Descriptive_Analysis")

class CustomerDescriptiveAnalysis:
    """
    Kelas untuk melakukan analisis deskriptif mendalam pada data RFM
    sebagai fondasi sebelum tahap modeling (Clustering).
    """
    def __init__(self):
        self.root = self._find_project_root()
        # Path data input hasil Ingestion DuckDB
        self.input_path = self.root / "data" / "production" / "customer_rfm" / "03_customer_rfm_features.parquet"
        # Path output untuk audit dan modeling
        self.output_stats = self.root / "data" / "logs" / "rfm_descriptive_stats.json"
        self.output_ready = self.root / "data" / "production" / "customer_rfm" / "01_customer_rfm_ready_to_model.parquet"
        
        # Setup visualisasi
        sns.set_theme(style="whitegrid")

    def _find_project_root(self, marker='models'):
        """Menemukan root direktori secara dinamis"""
        current_path = Path(__file__).resolve()
        for parent in current_path.parents:
            if (parent / marker).exists():
                return parent
        return Path.cwd()

    def load_data(self):
        """Memuat data Gold Layer RFM"""
        if not self.input_path.exists():
            logger.error(f"Data input tidak ditemukan di: {self.input_path}")
            sys.exit(1)
        logger.info(f"Memuat data RFM: {self.input_path.name}")
        return pd.read_parquet(self.input_path)

    # ==========================================
    # 2. CORE ANALYTICS FUNCTIONS
    # ==========================================
    def statistical_profiling(self, df):
        """Analisis statistik deskriptif & Deteksi Outlier (Whale Detection)"""
        logger.info("Menjalankan Statistical Profiling & Outlier Detection...")
        
        cols = ['recency', 'frequency', 'monetary']
        
        # Profiling statistik lengkap
        stats = df[cols].describe().T
        stats['skewness'] = df[cols].skew()
        
        # Identifikasi Outlier dengan IQR Method
        outlier_report = {}
        for col in cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
            outlier_report[f"{col}_outlier_pct"] = (len(outliers) / len(df)) * 100
        
        # Visualisasi Boxplot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, col in enumerate(cols):
            sns.boxplot(x=df[col], ax=axes[i], color='#4C72B0')
            axes[i].set_title(f'Outlier Distribution: {col.capitalize()}')
        plt.tight_layout()
        plt.show()

        return stats.to_dict(), outlier_report

    def distribution_and_correlation(self, df):
        """Analisis Distribusi (KDE) & Korelasi Spearman"""
        logger.info("Menganalisis Distribusi Variabel & Korelasi...")
        
        cols = ['recency', 'frequency', 'monetary']
        
        # Visualisasi Histogram & KDE
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, col in enumerate(cols):
            sns.histplot(df[col], kde=True, ax=axes[i], color='#55A868')
            axes[i].set_title(f'Distribution: {col.capitalize()}')
        plt.show()

        # Matriks Korelasi (Spearman untuk data non-normal)
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[cols].corr(method='spearman'), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Spearman Correlation Heatmap (RFM Variables)")
        plt.show()

    def baseline_segmentation(self, df):
        """Binning Persentil untuk Baseline Segmentasi (Standard Industry)"""
        logger.info("Membuat Baseline Segmentasi menggunakan Persentil Binning...")
        
        # Recency: Semakin kecil hari (pembelian baru), semakin baik [Score 5]
        df['R_Score'] = pd.qcut(df['recency'], 5, labels=[5, 4, 3, 2, 1])
        
        # Frequency & Monetary: Semakin besar, semakin baik [Score 5]
        # Menggunakan rank 'first' untuk menangani data frequency yang terkonsentrasi di angka 1
        df['F_Score'] = pd.qcut(df['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        df['M_Score'] = pd.qcut(df['monetary'], 5, labels=[1, 2, 3, 4, 5])
        
        # Gabungan RFM Score
        df['RFM_Score_Total'] = df['R_Score'].astype(int) + df['F_Score'].astype(int) + df['M_Score'].astype(int)
        
        # Analisis Dampak Bisnis (Pareto Principle)
        top_20_revenue = df.nlargest(int(len(df) * 0.2), 'monetary')['monetary'].sum()
        total_revenue = df['monetary'].sum()
        logger.info(f"Business Insight: Top 20% Pelanggan menyumbang {(top_20_revenue/total_revenue)*100:.2f}% Total Revenue.")
        
        return df

    # ==========================================
    # 3. EXECUTION PIPELINE
    # ==========================================
    def run_analysis(self):
        """Menjalankan seluruh R&D Workflow"""
        try:
            df = self.load_data()
            
            # Eksekusi Tahapan
            stats_dict, outliers = self.statistical_profiling(df)
            self.distribution_and_correlation(df)
            df_final = self.baseline_segmentation(df)
            
            # Persistensi Hasil Audit (JSON)
            audit_log = {"statistics": stats_dict, "outlier_analysis": outliers}
            with open(self.output_stats, 'w') as f:
                json.dump(audit_log, f, indent=4)
            
            # Simpan Data untuk Modeling (Parquet)
            df_final.to_parquet(self.output_ready, index=False)
            
            logger.info(f"SUKSES: Statistik deskriptif disimpan di {self.output_stats}")
            logger.info(f"SUKSES: Data siap model disimpan di {self.output_ready}")
            
        except Exception as e:
            logger.error(f"Kegagalan pada R&D Pipeline: {str(e)}")

if __name__ == "__main__":
    analysis = CustomerDescriptiveAnalysis()
    analysis.run_analysis()