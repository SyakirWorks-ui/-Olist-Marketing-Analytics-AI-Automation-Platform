import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os

# --- 1. KONFIGURASI HALAMAN (PAGE CONFIG) ---
# Mengatur layout menjadi 'wide' agar dashboard terlihat luas seperti Command Center
st.set_page_config(
    page_title="Olist Sentiment Command Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. FUNGSI LOAD DATA (CACHED) ---
# Menggunakan decorator @st.cache_data agar data tidak dimuat ulang setiap kali user melakukan interaksi (klik filter dll)
# Ini kunci performa tinggi (High Performance).
@st.cache_data
def load_data():
    """
    Memuat data dari Parquet (Detail) dan JSON (Summary).
    Dilengkapi error handling agar dashboard tidak crash jika file belum ada.
    """
    # Definisi Path (Sesuaikan dengan struktur folder project Anda)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Naik ke root jika script ini ada di dalam folder 'notebooks & py', sesuaikan path ini:
    # Asumsi script dijalankan dari root folder project untuk akses data yang benar
    
    # Path Relatif Standar (Sesuaikan jika perlu)
    PARQUET_PATH = os.path.join("data", "production", "sentiment", "final_actionable_list.parquet")
    JSON_PATH = os.path.join("data", "production", "sentiment", "daily_execution_summary.json")

    df = None
    summary = None

    try:
        # Load Main Data
        if os.path.exists(PARQUET_PATH):
            df = pd.read_parquet(PARQUET_PATH)
            # Pastikan tipe data benar
            df['review_score'] = df['review_score'].astype(int)
        
        # Load Summary Data
        if os.path.exists(JSON_PATH):
            with open(JSON_PATH, 'r') as f:
                summary = json.load(f)
                
    except Exception as e:
        st.error(f"Gagal memuat data: {str(e)}")
        
    return df, summary

# Memanggil fungsi load data
df_raw, summary_data = load_data()

# --- 3. SIDEBAR & FILTERS (CONTROL PANEL) ---
with st.sidebar:
    st.title("🎛️ Command Center")
    st.markdown("---")
    
    # Tombol Refresh manual (berguna di production real-time)
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.header("Filter Data")
    
    # Cek apakah data berhasil dimuat
    if df_raw is not None:
        # Filter 1: Sentimen
        sentiment_filter = st.multiselect(
            "Select Sentiment:",
            options=df_raw['predicted_sentiment'].unique(),
            default=df_raw['predicted_sentiment'].unique()
        )
        
        # Filter 2: Jenis Tindakan Pemulihan
        action_filter = st.multiselect(
            "Recovery Action Type:",
            options=df_raw['recovery_action'].unique(),
            default=df_raw['recovery_action'].unique()
        )
        
        # Filter 3: Skor Review
        score_filter = st.slider(
            "Review Score:",
            min_value=1, max_value=5, value=(1, 5)
        )
        
        # Terapkan Filter ke DataFrame
        df_filtered = df_raw[
            (df_raw['predicted_sentiment'].isin(sentiment_filter)) &
            (df_raw['recovery_action'].isin(action_filter)) &
            (df_raw['review_score'].between(score_filter[0], score_filter[1]))
        ]
    else:
        st.warning("Data source not found. Please run the Automation Engine first.")
        df_filtered = pd.DataFrame() # DataFrame kosong untuk mencegah crash

st.title("📊 Olist Sentiment Monitoring Dashboard")
st.markdown("### *Real-time Customer Sentiment & Recovery Tracking*")

# --- 4. KPI METRICS (HEADER SECTION) ---
# Menampilkan angka-angka kunci untuk eksekutif (CEO View)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

if summary_data and df_filtered is not None:
    # KPI 1: Total Volume
    kpi1.metric(
        label="Total Reviews Processed",
        value=f"{summary_data.get('total_processed', 0):,}",
        delta="Today's Batch"
    )
    
    # KPI 2: Negative Rate (Critical)
    total = summary_data.get('total_processed', 1)
    neg_count = summary_data.get('negative_sentiment_detected', 0)
    neg_rate = (neg_count / total) * 100
    kpi2.metric(
        label="Negative Sentiment Rate",
        value=f"{neg_rate:.1f}%",
        delta_color="inverse" # Merah jika naik (buruk)
    )
    
    # KPI 3: Financial Impact (Revenue at Risk)
    rev_risk = summary_data.get('total_revenue_at_risk', 0)
    kpi3.metric(
        label="Total Revenue at Risk",
        value=f"R$ {rev_risk:,.2f}",
        delta="Potential Loss"
    )
    
    # KPI 4: Pending Actions (Priority)
    # Menghitung real-time dari data yang difilter
    priority_count = df_filtered[df_filtered['recovery_action'].str.contains("Priority", case=False)].shape[0]
    kpi4.metric(
        label="Pending Priority Actions",
        value=priority_count,
        delta="Needs Attention",
        delta_color="inverse"
    )

st.markdown("---")

# --- 5. MAIN CHARTS AREA (VISUALISASI) ---
if df_filtered is not None and not df_filtered.empty:
    col_chart1, col_chart2, col_chart3 = st.columns([1, 1, 1])

    with col_chart1:
        st.subheader("Sentiment Distribution")
        # Donut Chart untuk proporsi Sentimen
        fig_pie = px.pie(
            df_filtered, 
            names='predicted_sentiment', 
            hole=0.4,
            color='predicted_sentiment',
            color_discrete_map={'Negative': '#FF4B4B', 'Positive': '#00CC96'} # Semantic Colors
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        st.subheader("Recovery Actions Required")
        # Bar Chart untuk melihat beban kerja tim CS
        action_counts = df_filtered['recovery_action'].value_counts().reset_index()
        action_counts.columns = ['Action', 'Count']
        fig_bar = px.bar(
            action_counts, 
            x='Count', 
            y='Action', 
            orientation='h',
            color='Action',
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart3:
        st.subheader("AI Confidence Level")
        # Gauge Chart sederhana untuk melihat keyakinan model AI
        avg_conf = df_filtered['sentiment_confidence'].mean() * 100
        fig_gauge = px.bar(
            x=[avg_conf], 
            y=["Avg Confidence"], 
            orientation='h', 
            range_x=[0, 100],
            text=[f"{avg_conf:.1f}%"],
            color_discrete_sequence=['#FFA500'] # Gold
        )
        fig_gauge.update_layout(xaxis_title="Confidence Score (%)", yaxis_title="")
        st.plotly_chart(fig_gauge, use_container_width=True)

    # --- 6. DETAILED ANALYSIS SECTION (THE HIT LIST) ---
    st.markdown("### 📋 The Hit List (Actionable Data)")
    
    # Styling DataFrame: Highlight baris Priority
    def highlight_priority(row):
        """Fungsi styling untuk mewarnai baris Priority menjadi kemerahan"""
        if "Priority" in row['recovery_action']:
            return ['background-color: #3d1010'] * len(row) # Merah gelap transparan (Dark mode friendly)
        return [''] * len(row)

    # Menampilkan DataFrame dengan Column Config yang cantik
    st.dataframe(
        df_filtered.style.apply(highlight_priority, axis=1),
        use_container_width=True,
        column_order=["order_id", "predicted_sentiment", "review_score", "recovery_action", "revenue_at_risk", "sentiment_confidence"],
        column_config={
            "order_id": "Order ID",
            "predicted_sentiment": st.column_config.TextColumn(
                "Sentiment",
                help="AI Predicted Sentiment",
                validate="^[a-zA-Z]+$"
            ),
            "review_score": st.column_config.ProgressColumn(
                "Score",
                format="%d ⭐",
                min_value=1,
                max_value=5,
            ),
            "recovery_action": "Recommended Action",
            "revenue_at_risk": st.column_config.NumberColumn(
                "Revenue at Risk",
                format="R$ %.2f"
            ),
            "sentiment_confidence": st.column_config.NumberColumn(
                "AI Conf.",
                format="%.2f"
            )
        },
        height=400
    )

    # --- 7. DRILL DOWN FEATURE (DEEP DIVE) ---
    st.markdown("---")
    st.subheader("🔍 Case Drill Down")
    
    # Selectbox untuk memilih Order ID
    selected_order = st.selectbox(
        "Search or Select Order ID to Investigate:",
        options=df_filtered['order_id'].unique(),
        index=0 if not df_filtered.empty else None
    )
    
    if selected_order:
        # Filter data untuk ID yang dipilih
        case_data = df_filtered[df_filtered['order_id'] == selected_order].iloc[0]
        
        # Tampilan detail menggunakan Container dan Columns
        with st.container(border=True):
            col_d1, col_d2 = st.columns([2, 1])
            
            with col_d1:
                st.markdown(f"#### 🆔 Order: {case_data['order_id']}")
                st.info(f"💬 **Customer Review:**\n\n_{case_data['review_comment_message']}_")
            
            with col_d2:
                st.markdown("#### AI Analysis")
                if case_data['predicted_sentiment'] == 'Negative':
                    st.error(f"⚠️ **Sentiment:** {case_data['predicted_sentiment']}")
                else:
                    st.success(f"✅ **Sentiment:** {case_data['predicted_sentiment']}")
                
                st.warning(f"🛠️ **Action:** {case_data['recovery_action']}")
                st.write(f"💰 **Risk Value:** R$ {case_data['revenue_at_risk']:,.2f}")

else:
    # Tampilan jika data kosong
    st.info("No data available based on current filters or dataset is empty.")

# --- FOOTER ---
st.markdown("---")
st.caption("Olist Sentiment Automation Engine | v1.0.0 Platinum Build | © 2026 Marketing Data Analyst Portfolio")