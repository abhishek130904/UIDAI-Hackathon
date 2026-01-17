import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings("ignore")

# Set plot style
plt.style.use('ggplot')
sns.set_palette("husl")

print("Starting Analysis...")

# Load Datasets
try:
    enrolment_df = pd.read_csv('master_aadhar_enrolment_data.csv')
    biometric_df = pd.read_csv('master_biometric_data.csv')
    demographic_df = pd.read_csv('master_demographic_data.csv')
    print("Loaded Master CSVs successfully.")
except FileNotFoundError:
    print("Master CSVs not found. Please run the consolidation script first.")
    exit(1)

# Date Parsing
for df in [enrolment_df, biometric_df, demographic_df]:
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')

# State Name Standardization Dictionary
state_corrections = {
    'Orissa': 'Odisha',
    'Uttaranchal': 'Uttarakhand',
    'Pondicherry': 'Puducherry',
    'Allahabad': 'Prayagraj',
}

def standardize_geo(df):
    df['state'] = df['state'].replace(state_corrections)
    return df

enrolment_df = standardize_geo(enrolment_df)
biometric_df = standardize_geo(biometric_df)
demographic_df = standardize_geo(demographic_df)

# Merge Dataframes
merge_keys = ['date', 'state', 'district', 'pincode']
master_df = pd.merge(enrolment_df, biometric_df, on=merge_keys, how='outer', suffixes=('_enr', '_bio'))
master_df = pd.merge(master_df, demographic_df, on=merge_keys, how='outer')

# Fill NaNs
num_cols = master_df.select_dtypes(include=[np.number]).columns
master_df[num_cols] = master_df[num_cols].fillna(0)

print(f"Master Table Shape: {master_df.shape}")

# --- Analysis 1: The Pulse Chart ---
print("Generating Pulse Chart...")
daily_pulse = master_df.groupby('date')[['age_0_5', 'bio_age_5_17', 'demo_age_5_17']].sum().reset_index()

plt.figure(figsize=(14, 7))
plt.plot(daily_pulse['date'], daily_pulse['age_0_5'], label='New Enrolments (0-5)', linewidth=2)
plt.plot(daily_pulse['date'], daily_pulse['bio_age_5_17'], label='Biometric Updates (5-17)', linestyle='--', alpha=0.8)
plt.plot(daily_pulse['date'], daily_pulse['demo_age_5_17'], label='Demographic Updates (5-17)', linestyle=':', alpha=0.8)

plt.title('The "Pulse" of Aadhaar: Enrolments vs Updates Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('pulse_chart.png')
print("Saved pulse_chart.png")

# --- Analysis 2: Lifecycle Predictor (Forecast) ---
print("Generating Forecast...")
# Prepare Time Series for Biometric Updates
daily_ts = daily_pulse.set_index('date')['bio_age_5_17'].asfreq('D').fillna(0)

try:
    model = ARIMA(daily_ts, order=(5,1,0))
    model_fit = model.fit()
    
    # Forecast next 30 days
    forecast = model_fit.forecast(steps=30)
    
    plt.figure(figsize=(10, 5))
    plt.plot(daily_ts.index[-90:], daily_ts[-90:], label='Historical Data (Last 90 Days)')
    plt.plot(forecast.index, forecast, label='Forecast (Next 30 Days)', color='green')
    plt.title('Biometric Update Demand Forecast (ARIMA)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('forecast_chart.png')
    print("Saved forecast_chart.png")
except Exception as e:
    print(f"Forecasting failed: {e}")

# --- Analysis 3: Anomaly Detection (Isolation Forest) ---
print("Running Anomaly Detection...")
district_features = master_df.groupby('district')[['age_0_5', 'bio_age_5_17', 'demo_age_5_17']].sum().reset_index()
district_features['update_ratio'] = district_features['bio_age_5_17'] / (district_features['age_0_5'] + 1)

iso_forest = IsolationForest(contamination=0.05, random_state=42)
district_features['anomaly'] = iso_forest.fit_predict(district_features[['bio_age_5_17', 'update_ratio']])

# Anomaly = -1, Normal = 1
anomalies = district_features[district_features['anomaly'] == -1]
print(f"Detected {len(anomalies)} anomalies.")
print("Top 5 Anomalous Districts:")
print(anomalies.sort_values(by='update_ratio', ascending=False).head(5)[['district', 'bio_age_5_17', 'update_ratio']])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=district_features, x='bio_age_5_17', y='update_ratio', hue='anomaly', palette={1: 'blue', -1: 'red'})
plt.title('Anomaly Detection: Districts with Unusual Update Patterns')
plt.xlabel('Total Biometric Updates')
plt.ylabel('Update Ratio')
plt.savefig('anomaly_detection.png')
print("Saved anomaly_detection.png")

# --- Analysis 4: Problem Map (District Bar Chart) ---
print("Generating Problem Map...")
top_problem_districts = district_summary.sort_values(by='update_ratio', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_problem_districts.index, y=top_problem_districts['update_ratio'], palette='viridis')
plt.xticks(rotation=45)
plt.title('Top 10 Districts by Biometric Update Ratio (Potential Hotspots)')
plt.ylabel('Update Ratio (Updates / Enrolments)')
plt.tight_layout()
plt.savefig('district_problem_map.png')
print("Saved district_problem_map.png")

print("Analysis Complete.")
