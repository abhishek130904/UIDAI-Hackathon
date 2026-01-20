<div align="center">

# 🚀 Lifecycle & Inclusivity Predictor
### UIDAI Hackathon Submission (Team UIDAI_8037)
**Analyzing enrollment data to identify patterns, detect fraud, and support policy decisions**

</div>

---

## 📌 About the Project

This project was built for the UIDAI Hackathon to transform raw Aadhaar logs into actionable intelligence. We process enrollment, biometric update, and demographic records to help identify operational bottlenecks, geographic patterns, and districts requiring infrastructure interventions.

The system handles millions of records across three datasets, merging and analyzing them to generate insights for India's identity infrastructure. While this was a learning project giving us hands-on experience with real-world data science challenges (messy data, geographic inconsistencies), the resulting architecture is a robust, end-to-end analytical pipeline.

The analysis outputs risk scores, flags anomalies using Machine Learning, generates demand forecasts, and creates visualizations to support data-driven policy decisions.

---

## 🎯 Objectives

- **Data Integration:** Merge fragmented CSV files (Enrolment, Biometric, Demographic) into a unified analytical timeline.
- **Predictive Planning:** Forecast biometric update demand 30 days in advance using **SARIMA** to handle weekly operational cycles.
- **Risk Assessment:** Flag districts with suspicious **Update-to-Enrolment Ratios (UER)** that may indicate fraud or operator error.
- **Inclusivity Auditing:** Identify districts where child enrolment (0-5 years) lags behind expected rates.
- **Visualization:** Create clear, report-ready visual representations of complex data patterns.

---

## ✨ Features

**1. Data Engineering & Normalization**
- Automated cleaning of district names to match Census 2011 shapefiles (e.g., handling *Allahabad* → *Prayagraj*).
- Temporal alignment to ensure continuous time-series data even with missing daily logs.

**2. Machine Learning: Anomaly Detection**
- Uses **Isolation Forest** (Unsupervised Learning) to detect outliers.
- Flags districts with:
  - Abnormal *Update-to-Enrolment Ratios* (UER > 2.0).
  - Suspiciously high biometric failure rates.

**3. Forecasting Engine (SARIMA)**
- We implemented **SARIMA (Seasonal ARIMA)** instead of basic linear models.
- Captures the 7-day operational cycle (weekend dips vs. weekday peaks) for realistic 30-day demand planning.

**4. Geospatial Intelligence**
- Generates district-level heatmaps for Biometric intensity and Pincode load density.
- Identifies "Infrastructure Gaps" (High Demand + Low Center Count).

**5. Automated Visualization Suite**
- One-click generation of **22 distinct visualizations**, including:
  - Pareto Charts (State Load Analysis)
  - Butterfly Charts (Child vs. Adult stats)
  - Correlation Scatter plots

---

## 🛠 Tech Stack

- **Python 3.8+**
- **Pandas & NumPy:** Heavy lifting for data aggregation and pivot tables.
- **GeoPandas:** Handling shapefiles and spatial joins for map visualizations.
- **Statsmodels (SARIMAX):** Time-series forecasting with seasonality.
- **Scikit-Learn (IsolationForest):** Anomaly detection logic.
- **Matplotlib & Seaborn:** Static report-ready plotting.

---

## 📂 Project Structure

```text
UIDAI-Hackathon/
│
├── master_aadhar_enrolment_data.csv   # Raw Source Data
├── master_biometric_data.csv
├── master_demographic_data.csv
│
├── visualization_dashboard.py         # MAIN ENGINE (Run this)
├── run.sh                             # Auto-setup script (Mac/Linux)
├── requirements.txt                   # Python dependencies
│
├── visualizations/                    # OUTPUT FOLDER (Auto-generated)
│   ├── 01_state_bio_map.png
│   ├── 13_sarima_forecast.png
│   ├── 21_anomaly_scatter.png
│   └── ... (22 files total)
│
└── notebooks/                         # Exploratory work
    ├── UIDIA-Adarsh.ipynb
    └── data_experiments.ipynb
```

---

## ⚙️ Installation & Run

We've included a helper script to automate the environment setup.

### Option 1: The Fast Way (Mac/Linux)

1.  **Clone the repo:**
    ```bash
    git clone https://github.com/your-username/UIDAI-Hackathon.git
    cd UIDAI-Hackathon
    ```

2.  **Run the auto-script:**
    ```bash
    chmod +x run.sh
    ./run.sh
    ```
    *This will create a virtual environment, install dependencies, and generate all graphs automatically.*

### Option 2: Manual Setup (Windows/Manual)

1.  **Create env:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

2.  **Install libs:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run analysis:**
    ```bash
    python visualization_dashboard.py
    ```

---

## ⚠️ Limitations

1.  **Shapefiles:** The code expects `India-States.shp` and `India-Districts-2011Census.shp` in the `../maps-master/` directory. If missing, map generation is skipped.
2.  **Memory:** Processing ~5M records requires about 4GB RAM.
3.  **Data Gaps:** The SARIMA model assumes missing dates are "zero activity" days (like holidays).

---

## 📚 Learning Outcomes

This project was a valuable learning experience that helped us develop several skills:

- **Data Engineering:** Working with large, messy datasets taught us about merging strategies and handling inconsistencies (e.g., date formats, district spellings).
- **Time-Series Modeling:** Implementing SARIMA introduced us to stationarity, seasonality, and differencing.
- **Unsupervised Learning:** Using Isolation Forest for anomaly detection gave us hands-on experience in identifying fraud patterns without labeled data.
- **Geospatial Analysis:** Merging operational data with shapefiles taught us about coordinate systems and spatial joins.
- **Pipeline Design:** We learned how to structure a project so that a single script (`visualization_dashboard.py`) can run the entire analysis end-to-end.

---