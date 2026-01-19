<div align="center">

# 🚀 UIDAI Aadhaar Data Analytics Platform
### Analyzing enrollment data to identify patterns and support policy decisions

</div>

---

## 📌 About the Project

This project was built for a UIDAI hackathon focused on analyzing large-scale Aadhaar data. It processes enrollment, biometric update, and demographic modification records to help identify operational bottlenecks, geographic patterns, and areas that need infrastructure improvements.

The system handles around 4.9 million records across three data types, merging and analyzing them to generate insights that could inform decision-making for India's identity infrastructure. It was a learning project that gave us hands-on experience with real-world data science challenges—handling messy data, geographic inconsistencies, and building end-to-end analytical pipelines.

The analysis outputs risk scores for districts, flags anomalies, generates forecasts, and creates visualizations that help understand enrollment patterns across India. It's designed to support data-driven policy decisions rather than replace human judgment.

---

## 🎯 Objectives

- **Data Integration:** Merge fragmented CSV files from multiple sources into unified datasets for analysis
- **Pattern Identification:** Discover geographic and temporal trends in enrollment and biometric update data
- **Risk Assessment:** Develop a scoring system to identify districts with high operational stress or resource gaps
- **Anomaly Detection:** Automatically flag unusual patterns that might indicate data quality issues or require investigation
- **Forecasting:** Predict future demand trends to support capacity planning
- **Visualization:** Create clear visual representations of complex data patterns for easier interpretation

---

## ✨ Features

**Data Processing & Merging**
- Combines enrollment, biometric, and demographic CSV files into master datasets
- Standardizes geographic names to handle inconsistencies (e.g., "Orissa" → "Odisha")
- Handles date parsing and data type conversions

**Risk Scoring System**
- Calculates district-level risk scores based on biometric stress, demographic update ratios, and mobility indices
- Classifies districts into Low, Medium, and High risk zones

**Anomaly Detection**
- Uses Isolation Forest algorithm to identify unusual pincode patterns
- Z-score based detection for temporal anomalies in time-series data

**Forecasting**
- ARIMA model generates 30-day ahead demand forecasts
- Analyzes day-of-week and monthly patterns

**Geospatial Analysis**
- Creates state and district-level maps showing enrollment intensity and update patterns
- Requires India shapefiles for map visualizations

**Policy Recommendations**
- Suggests actions like infrastructure upgrades, school-based enrollment drives, or temporary centers based on district characteristics

**Infrastructure Optimization**
- Identifies top 100 pincode locations where new enrollment centers would have maximum impact

**Visualization Dashboard**
- Generates 15+ visualizations including maps, time-series charts, comparative analyses, and statistical distributions

---

## 🛠 Technologies Used

**Core Libraries**
- Python 3.7+
- pandas - Data manipulation and analysis
- numpy - Numerical computations

**Visualization**
- matplotlib - Static plotting
- seaborn - Statistical visualizations
- plotly - Interactive charts (imported, though outputs are currently static PNGs)
- geopandas - Geospatial mapping

**Machine Learning & Statistics**
- scikit-learn - IsolationForest, KMeans, StandardScaler, MinMaxScaler
- statsmodels - ARIMA time-series forecasting
- scipy - Statistical functions

**Development**
- Jupyter Notebook - Interactive analysis and exploration

---

## 📂 Project Structure

```
UIDAI-Hackathon/
│
├── Dataset UIDAI/                    # Raw source data
│   ├── api_data_aadhar_biometric/   # 4 CSV files (~1.8M records)
│   ├── api_data_aadhar_demographic/ # 5 CSV files (~2M records)
│   └── api_data_aadhar_enrolment/   # 3 CSV files (~1M records)
│
├── visualizations/                   # Generated PNG files (15+ charts and maps)
│
├── master_*.csv                      # Processed master datasets
│   ├── master_aadhar_enrolment_data.csv
│   ├── master_biometric_data.csv
│   └── master_demographic_data.csv
│
├── UIDIA-Adarsh.ipynb               # Main analysis notebook
├── UIDIA.ipynb                      # Alternative workflow
├── UIDAI_Hackathon-Abhi.ipynb       # Exploratory analysis
│
├── analysis_script.py               # Complete analysis pipeline
│   └── Runs: data processing, feature engineering,
│       anomaly detection, forecasting, optimization
│
└── visualization_dashboard.py       # Visualization generator
    └── Creates: geospatial maps, time-series, 
        comparative charts, statistical plots
```

**Key Output Files:**
- `processed_master_table.csv` - Merged unified dataset
- `recommended_new_centers.csv` - Top 100 infrastructure recommendations
- `executive_dashboard.png` - 4-panel consolidated view
- `forecast_chart_v2.png` - 30-day demand forecast
- `visualizations/` - 15+ PNG files organized by analysis type

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step 1: Clone the repository

```bash
git clone https://github.com/abhishek130904/UIDAI-Hackathon.git
cd UIDAI-Hackathon
```

### Step 2: Install dependencies

```bash
pip install pandas numpy matplotlib seaborn plotly geopandas scikit-learn statsmodels scipy jupyter
```

**Note:** Installing geopandas can be tricky on some systems. If it fails, you might need to install GDAL and other geospatial dependencies separately. Check the geopandas documentation for your platform.

### Step 3: Prepare data files

The scripts expect master CSV files to be present. If you don't have them:
- Check if `master_aadhar_enrolment_data.csv`, `master_biometric_data.csv`, and `master_demographic_data.csv` exist in the project root
- If not, you'll need to run the data merging code from `UIDIA-Adarsh.ipynb` first (look for cells that use `glob.glob()` to merge CSV files)

### Step 4: Shapefiles (optional)

For geospatial map visualizations, you'll need India state and district shapefiles. If you have them, update the paths in `visualization_dashboard.py`:

```python
STATE_SHP = 'path/to/India-States.shp'
DISTRICT_SHP = 'path/to/India-Districts.shp'
```

The script will still work without shapefiles—it just won't generate the map visualizations.

---

## 🚀 How to Use

### Option 1: Run the complete analysis

This executes the full pipeline including data processing, feature engineering, anomaly detection, forecasting, and optimization:

```bash
python analysis_script.py
```

This generates:
- Processed master table CSV
- Recommended centers CSV
- Executive dashboard PNG
- Forecast and pulse charts
- Clustering visualizations

### Option 2: Generate visualizations only

If you already have processed data and just want to create charts:

```bash
python visualization_dashboard.py
```

This creates all 15+ visualization PNGs in the `visualizations/` folder. Expect this to take 10-20 minutes, especially if generating maps.

### Option 3: Interactive notebooks

For exploring the data step-by-step or customizing the analysis:

```bash
jupyter notebook UIDIA-Adarsh.ipynb
```

Notebooks let you run cells individually, see intermediate results, and modify parameters easily.

### Customizing parameters

You can adjust various parameters in the scripts:
- **Risk score weights** in notebooks (currently: 40% biometric stress, 30% demo update ratio, 20% mobility, 10% enrollments)
- **Forecast horizon** in `analysis_script.py` (change `steps=30` to different values)
- **Anomaly detection sensitivity** (modify `contamination=0.01` in Isolation Forest)

---

## 📸 Output / Results

The project generates several types of outputs:

**CSV Files:**
- `processed_master_table.csv` - Unified dataset with engineered features
- `recommended_new_centers.csv` - Prioritized list of 100 pincode locations for new centers

**Dashboard & Charts:**
- `executive_dashboard.png` - Four-panel view with live ticker, forecast, clustering matrix, and top anomalies
- `forecast_chart_v2.png` - 30-day ahead demand prediction with historical context
- `pulse_chart_v2.png` - System activity trends over time
- `inclusivity_clusters.png` - Demographic segmentation results

**Visualizations Folder:**
Contains 15+ PNG files organized into sections:
- Geospatial maps (state/district level)
- Time-series trends (daily, weekly, monthly)
- Comparative analysis (Pareto charts, butterfly charts, state rankings)
- Statistical plots (scatter plots, age distributions, correlations)

The visualizations use color coding and legends to make patterns easy to interpret. Maps show intensity through color gradients, while time-series charts include labels and trend lines.

---

## ⚠️ Limitations

**Data dependencies:** The scripts expect pre-merged master CSV files. If these don't exist, you need to run the merging code from notebooks first, which requires access to raw data in the `Dataset UIDAI/` folders.

**Shapefile requirement:** Geospatial maps need India shapefiles that aren't included. Without them, map generation is skipped but other visualizations still work.

**Memory usage:** Processing ~4.9M records needs significant RAM (4GB+ recommended). Large datasets might cause memory issues on systems with limited resources.

**Date format assumption:** Code assumes dates in `%d-%m-%Y` format. If your data uses different formats, you'll need to modify the date parsing sections.

**Geographic matching:** State/district name normalization tries to handle common variations but might miss edge cases. Manual verification helps ensure accuracy.

**Static processing:** Designed for batch processing of CSV files, not real-time analysis. No API integration or streaming data support currently.

**Model parameters:** ARIMA uses fixed parameters (5,1,0) that might not be optimal for all time-series patterns. The anomaly detection contamination rate (1%) is also a fixed assumption.

**Scalability:** Current implementation processes data sequentially. Very large datasets or multiple years of data might benefit from parallel processing, which isn't implemented yet.

---

## 🔮 Future Improvements

**Interactive dashboard:** Build a web-based dashboard (using Dash or Streamlit) where users can filter by region, adjust parameters, and explore results dynamically without modifying code.

**Database integration:** Move from CSV processing to database storage (PostgreSQL or MongoDB) to handle larger datasets more efficiently and support incremental updates.

**Enhanced anomaly detection:** Implement multiple algorithms (Local Outlier Factor, One-Class SVM) with ensemble voting to reduce false positives and improve detection accuracy.

**API development:** Create RESTful APIs to expose analytical results programmatically, allowing integration with other UIDAI systems or external applications.

**Automated reporting:** Generate formatted PDF or HTML reports with executive summaries and recommendations, making insights more accessible to non-technical stakeholders.

---

## 📚 Learning Outcomes

This project was a valuable learning experience that helped us develop several skills:

**Data engineering:** Working with large, messy datasets taught us about data cleaning, merging strategies, and handling inconsistencies. We learned to debug issues with geographic name mismatches and date parsing errors.

**Machine learning application:** Implementing Isolation Forest and K-Means gave hands-on experience with unsupervised learning. We learned about parameter tuning, feature scaling, and interpreting model outputs.

**Time-series analysis:** ARIMA forecasting introduced us to time-series modeling concepts like stationarity, differencing, and model selection. We gained appreciation for the challenges in making reliable predictions.

**Geospatial analysis:** Using geopandas to merge data with shapefiles and create maps taught us about coordinate systems, geographic data quality issues, and visualization techniques for spatial data.

**Visualization design:** Creating 15+ different visualizations helped us understand principles of effective data communication—choosing appropriate chart types, using color effectively, and making complex information accessible.

**Problem-solving:** Working through the full pipeline from raw data to insights taught us to think systematically, debug step-by-step, and balance exploratory analysis with production-ready code.

**Domain knowledge:** Understanding the context of Aadhaar operations helped us design more meaningful analyses and interpret results in ways that could actually inform decision-making.

---

## 📄 License

Not specified

---
