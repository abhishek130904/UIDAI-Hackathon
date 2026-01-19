# UIDAI Hackathon - Aadhaar Data Analysis & Insights

## Description

This project provides comprehensive analysis of Aadhaar enrolment, biometric updates, and demographic data to support UIDAI operations. It processes large-scale datasets from multiple sources, generates actionable insights through advanced analytics, and creates visualizations for decision-making. The system identifies operational bottlenecks, detects anomalies, recommends policy actions, and forecasts future demand. It is designed for data analysts, policy makers, and UIDAI stakeholders who need data-driven insights for improving service delivery and resource allocation.

## Features

- **Data Processing & Aggregation**: Merges multiple CSV files from enrolment, biometric, and demographic datasets into master files
- **Geospatial Visualizations**: Interactive maps showing state-wise and district-wise patterns across India
- **Time-Series Analysis**: Daily, weekly, and monthly trend analysis with ARIMA forecasting
- **Risk Scoring System**: District-level risk classification (Low/Medium/High) based on biometric stress, demographic update ratios, and mobility indices
- **Anomaly Detection**: Isolation Forest algorithm to identify unusual patterns in enrolment and update data
- **Policy Recommendations**: Automated suggestions for infrastructure upgrades, enrolment drives, and resource allocation
- **Clustering Analysis**: K-means clustering for demographic segmentation and inclusivity analysis
- **Comprehensive Dashboards**: 15+ visualization types including Pareto charts, butterfly charts, scatter plots, and heatmaps
- **Operational Efficiency Metrics**: Identifies under-served pincodes and recommends new enrollment centers

## Tech Stack

**Languages & Core Libraries:**
- Python 3.x
- pandas
- numpy

**Visualization:**
- matplotlib
- seaborn
- plotly
- geopandas

**Machine Learning & Analytics:**
- scikit-learn (IsolationForest, KMeans, MinMaxScaler, StandardScaler)
- statsmodels (ARIMA)
- scipy

**Tools:**
- Jupyter Notebook
- Python scripts

## Project Structure

```
UIDAI-Hackathon/
├── Dataset UIDAI/                    # Raw data files
│   ├── api_data_aadhar_biometric/    # Biometric update CSV files
│   ├── api_data_aadhar_demographic/  # Demographic update CSV files
│   └── api_data_aadhar_enrolment/    # Enrollment CSV files
├── visualizations/                   # Generated visualization PNG files
├── master_aadhar_enrolment_data.csv  # Processed enrollment master file
├── master_biometric_data.csv         # Processed biometric master file
├── master_demographic_data.csv       # Processed demographic master file
├── UIDIA-Adarsh.ipynb                # Main analysis notebook
├── UIDIA.ipynb                       # Alternative analysis notebook
├── UIDAI_Hackathon-Abhi.ipynb        # Additional analysis notebook
├── analysis_script.py                # Python script for data analysis
└── visualization_dashboard.py        # Python script for generating visualizations
```

## Installation & Setup

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/abhishek130904/UIDAI-Hackathon.git
cd UIDAI-Hackathon
```

### Step 2: Install Required Packages

```bash
pip install pandas numpy matplotlib seaborn plotly geopandas scikit-learn statsmodels scipy
```

### Step 3: Download India Shapefiles (Optional, for map visualizations)

The visualization dashboard uses India state and district shapefiles. Update the path in `visualization_dashboard.py` if you have custom shapefiles:

```python
STATE_SHP = 'path/to/India-States.shp'
DISTRICT_SHP = 'path/to/India-Districts.shp'
```

### Step 4: Verify Data Files

Ensure the following master CSV files are present (or generate them from raw data in `Dataset UIDAI/`):
- `master_aadhar_enrolment_data.csv`
- `master_biometric_data.csv`
- `master_demographic_data.csv`

## Usage

### Option 1: Run Analysis Script

Execute the comprehensive analysis script that performs data processing, feature engineering, anomaly detection, and optimization:

```bash
python analysis_script.py
```

This generates:
- `processed_master_table.csv` - Merged and processed data
- `recommended_new_centers.csv` - Top 100 recommended enrollment centers
- `executive_dashboard.png` - 4-panel executive dashboard
- Various analysis charts

### Option 2: Generate Visualizations

Run the visualization dashboard to create all geospatial and statistical visualizations:

```bash
python visualization_dashboard.py
```

This generates 15+ visualization files in the `visualizations/` folder, including:
- State and district maps
- Time-series trends
- Comparative analysis charts
- Statistical distributions

### Option 3: Use Jupyter Notebooks

Open any of the `.ipynb` files in Jupyter Notebook for interactive analysis:

```bash
jupyter notebook UIDIA-Adarsh.ipynb
```

Notebooks allow step-by-step execution, data exploration, and custom analysis.

### Example Output Files

- **Visualizations**: `visualizations/01_bio_updates_map.png`, `visualizations/04_daily_bio_trend.png`, etc.
- **Analysis Results**: `recommended_new_centers.csv`, `processed_master_table.csv`
- **Dashboard**: `executive_dashboard.png`

## Configuration

### Data File Paths

Update paths in scripts if your data structure differs:

```python
# In analysis_script.py
enrolment_df = pd.read_csv('master_aadhar_enrolment_data.csv')
biometric_df = pd.read_csv('master_biometric_data.csv')
demographic_df = pd.read_csv('master_demographic_data.csv')
```

### Visualization Settings

Customize visualization output directory in `visualization_dashboard.py`:

```python
OUTPUT_DIR = 'visualizations'  # Change output folder name
```

### Risk Scoring Weights

Adjust risk score calculation weights in notebooks:

```python
district_master_scaled['risk_score'] = (
    0.4 * district_master_scaled['biometric_stress'] +
    0.3 * district_master_scaled['demo_update_ratio'] +
    0.2 * district_master_scaled['mobility_index_norm'] +
    0.1 * district_master_scaled['total_enrolments']
)
```

### Forecast Parameters

Modify ARIMA model parameters in `analysis_script.py`:

```python
model = ARIMA(state_pulse, order=(5,1,0))  # (p, d, q) parameters
forecast = fit.forecast(steps=30)  # Forecast 30 days ahead
```

## Known Limitations / TODO

- **Shapefile Dependency**: Map visualizations require external India shapefiles that must be provided separately
- **Data Preprocessing**: Raw CSV files must be merged into master files before running analysis scripts
- **Memory Usage**: Processing large datasets may require significant RAM (4GB+ recommended)
- **Date Format**: Assumes date format `%d-%m-%Y`; may need adjustment for different formats
- **Geographic Matching**: State/district name normalization may need updates for better shapefile matching
- **Real-time Data**: Currently processes static CSV files; real-time API integration not implemented
- **Scalability**: Large-scale processing could benefit from parallel processing or chunking optimizations

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Make your changes and ensure code follows existing style
4. Test your changes with sample data
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature-name`)
7. Open a Pull Request

**Guidelines:**
- Keep code readable with clear variable names and comments
- Ensure visualizations are properly labeled and formatted
- Document any new analysis methods or features
- Test with sample data before submitting

## License

Not specified
