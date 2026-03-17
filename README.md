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
| ![Monthly Revenue](<img width="1246" height="667" alt="Monthly Revenue" src="https://github.com/user-attachments/assets/ecf6d711-793e-4247-9241-d9c46ab49e06" />
) | ![Top 10 Product](<img width="715" height="583" alt="Top 10 Product by GMV" src="https://github.com/user-attachments/assets/e2267c7d-ae03-4bc3-a142-dadcf033ddcd" />
) |
| *Visualizing a steady revenue climb with a peak in late 2017.* | *Health & Beauty leads the GMV contribution at 14.7%.* |

---

### 🚚 Project 2: Logistics & Delivery Excellence
Focuses on supply chain optimization and delivery performance mapping across Brazilian states to minimize latency and delays.

| State Performance: Lead Time vs OTD% | Olist Business Priority Matrix |
|---|---|
| ![State Performance](<img width="993" height="450" alt="state_performance" src="https://github.com/user-attachments/assets/3909133b-4f85-4285-93cf-ab78095f406e" />
) | ![Correlation Matrix](<img width="1077" height="450" alt="correlation matrix" src="https://github.com/user-attachments/assets/72e66b5f-1aa3-41bd-a16e-535a21a9ab1b" />
) |
| *Correlation analysis between average delivery time and On-Time Delivery percentage.* | *Mapping of problem topics to determine the priority scale of operational improvements.* |

---

### 👥 Project 3: Customer Segmentation (RFM Analysis)
Classifies customers based on transactional behavior to drive personalized marketing strategies and improve retention rates.

| Customer Segmentation & Monetary Value | Strategic Recovery Actions |
|---|---|
| ![Customer Segmentation](<img width="993" height="450" alt="customer segmentation" src="https://github.com/user-attachments/assets/5940de20-8af0-40bf-b6d4-b1068f1b1157" />
) | ![Recovery Actions](<img width="1077" height="450" alt="strategic recomendation" src="https://github.com/user-attachments/assets/98191ebd-7ef8-4120-bfa5-30a66c050e99" />

) |
| *Visualization of the composition of 'Champions' to 'At Risk' segments based on monetary contribution.* | *Automated distribution of recovery actions to minimize churn among critical customers.* |

---

### 🤖 Project 4: AI Sentiment Command Center
Implementation of Natural Language Processing (NLP) to detect customer sentiment in real-time and automate customer service responses.

| Priority Matrix | Keyword Impact Analysis |
|---|---|
| ![Confusion Matrix](<img width="1035" height="939" alt="confusion matrix" src="https://github.com/user-attachments/assets/26ab4a8b-17f4-4e42-a821-a7d48bdccd11" />
) | ![Keyword Impact](<img width="912" height="547" alt="image" src="https://github.com/user-attachments/assets/608b2d48-0d75-4cb9-a2e2-2b660ef357d2" />
) |
| - **Sentiment Analytics**: Leverages AI to categorize feedback and calculate an 'AI Confidence' score for high-precision monitoring.
  - **Operational Recovery**: Maps specific AI-driven recommendations (e.g., Vouchers, CS Priority) to critical customer cases to protect future revenue. |

---
## 🤝 Contact & Contribution
I am a Data Analyst passionate about transforming complex datasets into clear business narratives. Feel free to reach out for collaboration or inquiries!

* **Author**: [Muhamad Syakirullah]
* **LinkedIn**: [https://www.linkedin.com/in/syakirworks/]
* **Email**: [syakirworksid@gmail.com]
* **Fiverr**: [https://www.fiverr.com/sellers/tajulmuluk/]
* **Upwork**: [https://www.upwork.com/freelancers/~01dc67531f5441b58a]

---
*Developed as part of the iFood Marketing Intelligence Project - 2026*

