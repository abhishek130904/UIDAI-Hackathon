import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('ggplot')
sns.set_palette("husl")

print("Starting Analysis (V2 Systemic Optimization)...")

# --- 1. Data Ingestion & Cleaning ---
try:
    enrolment_df = pd.read_csv('master_aadhar_enrolment_data.csv')
    biometric_df = pd.read_csv('master_biometric_data.csv')
    demographic_df = pd.read_csv('master_demographic_data.csv')
    print("Loaded Master CSVs.")
except FileNotFoundError:
    print("Master CSVs not found.")
    exit(1)

# Date Parsing
for df in [enrolment_df, biometric_df, demographic_df]:
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')

# Standardization
state_corrections = {'orissa': 'odisha', 'uttaranchal': 'uttarakhand', 'pondicherry': 'puducherry', 'allahabad': 'prayagraj'}
district_corrections = {'medchal?malkajgiri': 'medchal-malkajgiri', 'yadadri.': 'yadadri', 'tumkur': 'tumakuru'}

def standardize_geo(df):
    df['state'] = df['state'].astype(str).str.strip().str.lower().replace(state_corrections)
    df['district'] = df['district'].astype(str).str.strip().str.lower().replace(district_corrections)
    df['district'] = df['district'].str.replace(r'[\*\.]+$', '', regex=True).str.strip()
    return df

enrolment_df = standardize_geo(enrolment_df)
biometric_df = standardize_geo(biometric_df)
demographic_df = standardize_geo(demographic_df)

# Merge
merge_keys = ['date', 'state', 'district', 'pincode']
master_df = pd.merge(enrolment_df, biometric_df, on=merge_keys, how='outer', suffixes=('_enr', '_bio'))
master_df = pd.merge(master_df, demographic_df, on=merge_keys, how='outer')
master_df.fillna(0, inplace=True)
master_df.to_csv('processed_master_table.csv', index=False)
print("Saved processed_master_table.csv")

# --- 2. Feature Engineering (Phase 1) ---
print("Running Feature Engineering...")
pincode_profile = master_df.groupby('pincode')[['age_0_5', 'age_5_17', 'bio_age_5_17']].sum().reset_index()

# Urban/Rural Proxy
pincode_profile['total_activity'] = pincode_profile['age_0_5'] + pincode_profile['bio_age_5_17']
urban_threshold = pincode_profile['total_activity'].quantile(0.80)
pincode_profile['is_urban'] = pincode_profile['total_activity'] > urban_threshold
print(f"Classified {pincode_profile['is_urban'].sum()} Pincodes as Urban.")

# Merge back
master_df = master_df.merge(pincode_profile[['pincode', 'is_urban']], on='pincode', how='left')

# Date Features
master_df['day_of_week'] = master_df['date'].dt.day_name()
master_df['is_weekend'] = master_df['date'].dt.dayofweek >= 5
master_df['month'] = master_df['date'].dt.month_name()

# Lags
master_df = master_df.sort_values(['pincode', 'date'])
master_df['lag_1m_enrolment'] = master_df.groupby('pincode')['age_0_5'].shift(30).fillna(0)
master_df['rolling_3m_enrolment'] = master_df.groupby('pincode')['age_0_5'].rolling(90).mean().reset_index(0, drop=True).fillna(0)

# --- 3. Operational Efficiency (Phase 2) ---
print("Analyzing Efficiency...")
under_served = pincode_profile[
    (pincode_profile['age_0_5'] > pincode_profile['age_0_5'].quantile(0.75)) & 
    (pincode_profile['bio_age_5_17'] == 0)
]
print(f"Found {len(under_served)} Under-Served Pincodes.")

pincode_profile['efficiency_score'] = pincode_profile['bio_age_5_17'] / (pincode_profile['age_0_5'] + 1)

# Forensics
iso = IsolationForest(contamination=0.01, random_state=42)
pincode_profile['anomaly'] = iso.fit_predict(pincode_profile[['age_0_5', 'bio_age_5_17', 'efficiency_score']].fillna(0))
print(f"Flagged {len(pincode_profile[pincode_profile['anomaly'] == -1])} Anomalies.")

# --- 4. Deep Demographics (Phase 3) ---
cluster_data = pincode_profile[['age_0_5', 'efficiency_score']].fillna(0)
scaler = StandardScaler()
kmeans = KMeans(n_clusters=3, random_state=42)
pincode_profile['cluster'] = kmeans.fit_predict(scaler.fit_transform(cluster_data))

plt.figure(figsize=(10,6))
sns.scatterplot(data=pincode_profile, x='age_0_5', y='efficiency_score', hue='cluster', palette='viridis')
plt.title('Inclusivity Clusters')
plt.savefig('inclusivity_clusters.png')
print("Saved inclusivity_clusters.png")

# --- 5. Optimization (Phase 4) ---
budget = 100
pincode_profile['infra_gap'] = pincode_profile['age_0_5'] - pincode_profile['bio_age_5_17']
recommended_centers = pincode_profile.sort_values(by='infra_gap', ascending=False).head(budget)
recommended_centers.to_csv('recommended_new_centers.csv', index=False)
print("Saved recommended_new_centers.csv")

# Forecast
state_pulse = master_df.groupby('date')['bio_age_5_17'].sum()
model = ARIMA(state_pulse, order=(5,1,0))
fit = model.fit()
forecast = fit.forecast(steps=30)

plt.figure(figsize=(12,5))
state_pulse.tail(90).plot(label='History')
forecast.plot(label='Forecast')
plt.title('National Forecast')
plt.legend()
plt.savefig('forecast_chart_v2.png')
print("Saved forecast_chart_v2.png")

# --- 6. Reporting (Phase 5) ---
daily_pulse = master_df.groupby('date')[['age_0_5', 'bio_age_5_17']].sum()
plt.figure(figsize=(14, 7))
plt.plot(daily_pulse.index, daily_pulse['age_0_5'], label='Enrolments')
plt.plot(daily_pulse.index, daily_pulse['bio_age_5_17'], label='Biometric Updates', linestyle='--')
plt.title('System Pulse')
plt.legend()
plt.savefig('pulse_chart_v2.png')
print("Saved pulse_chart_v2.png")

# --- Step 19: Executive Dashboard (Consolidated 4-Panel) ---
print("Generating Executive Dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('UIDAI 2026 Executive Command Center', fontsize=16)

# Panel 1: Ticker (Daily Pulse)
axes[0, 0].plot(daily_pulse.index, daily_pulse['age_0_5'], label='New Enrolments', color='blue')
axes[0, 0].plot(daily_pulse.index, daily_pulse['bio_age_5_17'], label='Biometric Updates', color='green', linestyle='--')
axes[0, 0].set_title('Live Ticker: Enrolment vs Updates')
axes[0, 0].legend()

# Panel 2: Forecast
axes[0, 1].plot(state_pulse.tail(90).index, state_pulse.tail(90), label='Historical', color='gray')
axes[0, 1].plot(forecast.index, forecast, label='30-Day Forecast', color='red')
axes[0, 1].set_title('Demand Forecast (Next 30 Days)')
axes[0, 1].legend()

# Panel 3: Inclusivity Clusters (Scatter)
sns.scatterplot(data=pincode_profile, x='age_0_5', y='efficiency_score', hue='cluster', palette='viridis', ax=axes[1, 0])
axes[1, 0].set_title('Inclusivity Matrix (Clusters)')

# Panel 4: Anomaly List (Bar Chart of Top Anomalies)
top_anomalies = pincode_profile[pincode_profile['anomaly'] == -1].sort_values(by='bio_age_5_17', ascending=False).head(5)
sns.barplot(data=top_anomalies, x='pincode', y='bio_age_5_17', ax=axes[1, 1], palette='Reds_d')
axes[1, 1].set_title('Top 5 Anomalous Pincodes (Possible Fraud)')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('executive_dashboard.png')
print("Saved executive_dashboard.png")

print("V2 Analysis Complete.")
