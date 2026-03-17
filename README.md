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

### 📈 Project 2: Logistic Delivery Higlights
To represent the business growth of Olist, the following key metrics were analyzed:

| Monthly Revenue & Growth | Top 10 Categories by GMV |
|---|---|

### 📈 Project 3: Customer rfm Highlights
To represent the business growth of Olist, the following key metrics were analyzed:

| Monthly Revenue & Growth | Top 10 Categories by GMV |
|---|---|

### 📈 Project 4: Sentiment Highlights
To represent the business growth of Olist, the following key metrics were analyzed:

| Monthly Revenue & Growth | Top 10 Categories by GMV |
|---|---|

**Key Sales Insights:**
* **Revenue Trend:** Significant growth observed starting Jan 2017, reaching peaks above 1.1M BRL.
* **Monthly Active Users:** Consistent MAU growth peaking at over 7,000 unique customers.
* **Order Value:** Maintained a healthy Average Order Value (AOV) around 154.41 BRL.

---

### 🤖 Project 4: Sentiment AI Command Center
The "5-Second Rule" dashboard for executive decision-making:

![Sentiment Dashboard Preview](https://raw.githubusercontent.com/yourusername/olist-marketing-automation/main/reports/Sentiment_Dashboard.png)
*Automated monitoring of Revenue at Risk and Priority Tiers.*
