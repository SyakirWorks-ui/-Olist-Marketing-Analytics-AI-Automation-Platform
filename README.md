# 🛒 Olist Marketing Analytics & AI Automation Platform

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B.svg)
![Status](https://img.shields.io/badge/Project-Production--Ready-green.svg)

An end-to-end data science ecosystem for the Olist E-commerce dataset, transforming raw marketplace data into automated business actions. This repository covers the full lifecycle from **Research & Development (Notebooks)** to **Scalable Production (Python Scripts)**.

---

## 🚀 Project Overview

This monorepo contains four strategic analytical modules designed to optimize marketing performance and customer retention:

### 1. 📈 Sales Performance Analytics
* **Goal:** Tracking GMV growth and identifying high-performing product categories.
* **Output:** Comprehensive sales trend analysis and seasonality forecasting.

### 2. 🚚 Logistic Delivery Optimization
* **Goal:** Analyzing delivery lead times and carrier performance to reduce churn caused by delays.
* **Output:** Logistics efficiency reporting and bottleneck identification.

### 3. 👥 Customer RFM Segmentation
* **Goal:** Categorizing customers based on Recency, Frequency, and Monetary value.
* **Output:** Tailored marketing segments (Loyalists, At-Risk, New Customers) for targeted campaigns.

### 4. 🤖 AI Sentiment Analysis & Recovery Engine (Core Project)
* **Goal:** Automating customer recovery using NLP and Prescriptive Analytics.
* **Research Phase:** Deep EDA and Sentiment Modeling in Jupyter Notebooks.
* **Production Phase:** A robust Python Engine that predicts sentiment and triggers automated recovery actions.

---

## 📂 Repository Structure

The project is organized into two distinct phases for each module: **Research** and **Production**.

```text
olist-marketing-automation/
├── 01_sales_performance/
├── 02_logistic_delivery/
├── 03_customer_rfm/
├── 04_sentiment_analysis/
│   ├── data/                 # Raw & Processed (.parquet, .csv)
│   ├── notebooks/            # R&D: EDA & Model Training (.ipynb)
│   ├── src/                  # Production: Automation Engines (.py)
│   ├── dashboard/            # Executive Command Center (app.py)
```

---

## 💡 Technical Highlights (Project 04)

The **Sentiment Automation Engine** represents the "Platinum Tier" of this repository:

* **AI Inference:** Deploys a Machine Learning pipeline to analyze customer reviews with high confidence.
* **Prescriptive Automation:** Automatically assigns recovery actions (e.g., "Auto-Refund + 15% Voucher") based on churn risk.
* **Revenue Protection:** Quantifies **"Revenue at Risk"** to allow proactive financial interventions.
* **Real-time Monitoring:** A Streamlit-based dashboard providing an executive 5-second overview of business health.

---

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/olist-marketing-automation.git](https://github.com/yourusername/olist-marketing-automation.git)
   cd olist-marketing-automation
2. **InstallDependencies:**
   pip install -r requirements.txt
   
3. **Run the Sentiment Dashboard:**
   streamlit run 04_sentiment_analysis/dashboard/app.py
│   └── logs/                 # Operational Audit Trails
├── requirements.txt          # Shared dependencies
└── README.md                 # Main Portfolio documentation

## 📊 Dashboard & Analytics Preview

### 📈 Project 1: Sales Performance Highlights
To represent the business growth of Olist, the following key metrics were analyzed:

| Monthly Revenue & Growth | Top 10 Categories by GMV |
|---|---|
| ![Monthly Revenue](https://raw.githubusercontent.com/yourusername/olist-marketing-automation/main/reports/Monthly%20Revenue.png) | ![Top 10 Product](https://raw.githubusercontent.com/yourusername/olist-marketing-automation/main/reports/Top%2010%20Product%20by%20GMV.png) |
| *Visualizing a steady revenue climb with a peak in late 2017.* | *Health & Beauty leads the GMV contribution at 14.7%.* |

---

### 🚚 Project 2: Logistics & Delivery Excellence
Fokus pada optimalisasi rantai pasok dan pemetaan performa pengiriman di seluruh negara bagian Brasil untuk menekan angka keterlambatan.

| State Performance: Lead Time vs OTD% | Olist Business Priority Matrix |
|---|---|
| ![State Performance](https://raw.githubusercontent.com/yourusername/olist-analytics/main/reports/state_performance.png) | ![Priority Matrix](https://raw.githubusercontent.com/yourusername/olist-analytics/main/reports/olist%20business%20priority%20matrix.png) |
| *Analisis korelasi antara waktu pengiriman rata-rata dan persentase On-Time Delivery.* | *Pemetaan topik masalah untuk menentukan skala prioritas perbaikan operasional.* |

---

### 👥 Project 3: Customer Segmentation (RFM Analysis)
Mengklasifikasikan pelanggan berdasarkan perilaku transaksi untuk personalisasi strategi pemasaran dan peningkatan retensi.

| Customer Segmentation & Monetary Value | Strategic Recovery Actions |
|---|---|
| ![Customer Segmentation](https://raw.githubusercontent.com/yourusername/olist-analytics/main/reports/customer%20segmentation.png) | ![Recovery Actions](https://raw.githubusercontent.com/yourusername/olist-analytics/main/reports/image_3e4ca5.png) |
| *Visualisasi komposisi segmen 'Champions' hingga 'At Risk' berdasarkan kontribusi moneter.* | *Distribusi tindakan pemulihan otomatis untuk meminimalkan churn pada pelanggan kritis.* |

---

### 🤖 Project 4: AI Sentiment Command Center
Implementasi NLP (Natural Language Processing) untuk mendeteksi sentimen pelanggan secara real-time dan mengotomatisasi respon layanan pelanggan.

| AI Sentiment Command Center Overview | Keyword Impact Analysis |
|---|---|
| ![Sentiment Overview](https://raw.githubusercontent.com/yourusername/olist-analytics/main/reports/image_3ec182.png) | ![Keyword Impact](https://raw.githubusercontent.com/yourusername/olist-analytics/main/reports/image_3e4ce0.png) |
| *Monitoring sentimen negatif secara real-time dengan estimasi Revenue at Risk.* | *Top 20 kata kunci dalam bahasa Portugis yang paling memengaruhi prediksi model AI.* |

---
## 🤝 Contact & Contribution
I am a Data Analyst passionate about transforming complex datasets into clear business narratives. Feel free to reach out for collaboration or inquiries!

* **Author**: [Muhamad Syakirullah]
* **LinkedIn**: [https://www.linkedin.com/in/syakirworks/]
* **Email**: [syakirworksid@gmail.com]
* **Fiverr**: [https://www.fiverr.com/sellers/tajulmuluk/]

---
*Developed as part of the iFood Marketing Intelligence Project - 2026*

