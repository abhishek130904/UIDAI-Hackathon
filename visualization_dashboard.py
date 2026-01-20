import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import geopandas as gpd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from matplotlib.ticker import FuncFormatter

# --- System Configuration ---
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Setup Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
MAPS_DIR = os.path.join(BASE_DIR, '../maps-master/Survey-of-India-Index-Maps/Boundaries')
OUTPUT_DIR = os.path.join(BASE_DIR, 'visualizations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(">>> Initializing UIDAI Analytics Dashboard...")

# 1. DATA PROCESSING UTILITIES

def clean_location_name(text):
    """Standardizes state and district names for shapefile matching."""
    if pd.isna(text): return 'unknown'
    txt = str(text).lower().strip()
    txt = txt.replace('&', ' and ')
    txt = ''.join(e if e.isalnum() else ' ' for e in txt)
    txt = ' '.join(txt.split())
    
    # Corrections dictionary
    mappings = {
        'uttaranchal': 'uttarakhand',
        'orissa': 'odisha',
        'pondicherry': 'puducherry',
        'allahabad': 'prayagraj',  # Normalized name
        'delhi': 'nct of delhi',
        'west bangal': 'west bengal',
        'westbengal': 'west bengal',
        'jammu and kashmir': 'jammu and kashmir',
        'ladakh': 'jammu and kashmir',
        'andaman and nicobar islands': 'andaman and nicobar island',
        'dadra and nagar haveli': 'dadra and nagar haveli and daman and diu',
        'daman and diu': 'dadra and nagar haveli and daman and diu'
    }
    return mappings.get(txt, txt)

# 2. DATA INGESTION

print("--> Loading CSV Datasets...")
try:
    df_enr = pd.read_csv(os.path.join(DATA_DIR, 'master_aadhar_enrolment_data.csv'))
    df_bio = pd.read_csv(os.path.join(DATA_DIR, 'master_biometric_data.csv'))
    df_demo = pd.read_csv(os.path.join(DATA_DIR, 'master_demographic_data.csv'))
except FileNotFoundError as e:
    print(f"CRITICAL ERROR: {e}")
    exit()

# Apply cleaning and date parsing
for df in [df_enr, df_bio, df_demo]:
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    df['state'] = df['state'].apply(clean_location_name)
    df['district'] = df['district'].apply(clean_location_name)

print("--> Loading Shapefiles...")
gdf_states = gpd.read_file(os.path.join(MAPS_DIR, 'India-States.shp'))
gdf_districts = gpd.read_file(os.path.join(MAPS_DIR, 'India-Districts-2011Census.shp'))

# Normalize shapefile names
gdf_states['state_norm'] = gdf_states['ST_NM'].apply(clean_location_name)
gdf_districts['state_norm'] = gdf_districts['ST_NM'].apply(clean_location_name)
gdf_districts['dist_norm'] = gdf_districts['DISTRICT'].apply(clean_location_name)

# 3. AGGREGATION ENGINE
print("--> Aggregating Operational Metrics...")

# --- State Level Aggregation ---
state_bio = df_bio.groupby('state').agg({
    'bio_age_5_17': 'sum', 'bio_age_17_': 'sum', 'pincode': 'nunique'
}).rename(columns={'pincode': 'num_pincodes'}).reset_index()

state_enr = df_enr.groupby('state').agg({
    'age_0_5': 'sum', 'age_5_17': 'sum', 'age_18_greater': 'sum'
}).reset_index()

state_demo = df_demo.groupby('state').agg({
    'demo_age_5_17': 'sum', 'demo_age_17_': 'sum'
}).reset_index()

# Merge State Data
state_stats = state_bio.merge(state_enr, on='state', how='outer') \
                       .merge(state_demo, on='state', how='outer').fillna(0)

# Calculate Totals (State)
state_stats['total_bio'] = state_stats['bio_age_5_17'] + state_stats['bio_age_17_']
state_stats['total_enr'] = state_stats['age_0_5'] + state_stats['age_5_17'] + state_stats['age_18_greater']
state_stats['total_demo'] = state_stats['demo_age_5_17'] + state_stats['demo_age_17_']

# --- District Level Aggregation ---
dist_bio = df_bio.groupby(['state', 'district']).agg({
    'bio_age_5_17': 'sum', 'bio_age_17_': 'sum', 'pincode': 'nunique'
}).rename(columns={'pincode': 'num_pincodes'}).reset_index()

dist_enr = df_enr.groupby(['state', 'district']).agg({
    'age_0_5': 'sum', 'age_5_17': 'sum', 'age_18_greater': 'sum'
}).reset_index()

dist_demo = df_demo.groupby(['state', 'district']).agg({
    'demo_age_5_17': 'sum', 'demo_age_17_': 'sum'
}).reset_index()

# Merge District Data
dist_stats = dist_bio.merge(dist_enr, on=['state', 'district'], how='outer') \
                     .merge(dist_demo, on=['state', 'district'], how='outer').fillna(0)

# Calculate Totals (District)
dist_stats['total_bio'] = dist_stats['bio_age_5_17'] + dist_stats['bio_age_17_']
dist_stats['total_enr'] = dist_stats['age_0_5'] + dist_stats['age_5_17'] + dist_stats['age_18_greater']
dist_stats['total_demo'] = dist_stats['demo_age_5_17'] + dist_stats['demo_age_17_']

# Advanced Features (UER, Load Intensity)
dist_stats['UER'] = dist_stats['total_bio'] / (dist_stats['total_enr'] + 1)
dist_stats['child_share'] = (dist_stats['age_0_5'] + dist_stats['age_5_17']) / (dist_stats['total_enr'] + 1)
dist_stats['load_per_pincode'] = dist_stats['total_bio'] / (dist_stats['num_pincodes'] + 1e-6)

# --- Geospatial Merging ---
print("--> Joining Data with Shapefiles...")
india_map = gdf_states.merge(state_stats, left_on='state_norm', right_on='state', how='left').fillna(0)
district_map = gdf_districts.merge(dist_stats, left_on=['state_norm', 'dist_norm'], 
                                   right_on=['state', 'district'], how='left').fillna(0)

# 4. VISUALIZATION SUITE

def save_viz(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {filename}")

def robust_map_plot(gdf, col, title, cmap, filename, label=''):
    fig, ax = plt.subplots(figsize=(10, 10))
    # Handle outliers for better color contrast
    vmax = np.percentile(gdf[col], 95)
    if vmax == 0: vmax = gdf[col].max()
    
    gdf.plot(column=col, ax=ax, cmap=cmap, legend=True, vmin=0, vmax=vmax,
             missing_kwds={'color': '#eeeeee'}, legend_kwds={'label': label, 'shrink': 0.5})
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    save_viz(filename)

# --- PART A: GEOSPATIAL MAPS ---
print("--> Generating Geospatial Maps...")

# National Level
robust_map_plot(india_map, 'bio_age_5_17', 'Biometric Updates (Age 5-17)', 'YlOrRd', '01_state_bio_map.png', 'Updates')
robust_map_plot(india_map, 'total_enr', 'Total Enrolments', 'Blues', '02_state_enr_map.png', 'Enrolments')
robust_map_plot(india_map, 'total_demo', 'Demographic Updates', 'Greens', '03_state_demo_map.png', 'Updates')

# District Level
robust_map_plot(district_map, 'bio_age_5_17', 'District Biometric Updates', 'YlOrRd', '04_district_bio_map.png', 'Updates')
robust_map_plot(district_map, 'total_enr', 'District Enrolment Intensity', 'Blues', '05_district_enr_map.png', 'Enrolments')
robust_map_plot(district_map, 'total_demo', 'District Demographic Updates', 'Greens', '06_district_demo_map.png', 'Updates')
robust_map_plot(district_map, 'load_per_pincode', 'Pincode Service Load Density', 'RdPu', '07_pincode_load_map.png', 'Avg Updates/Center')

# --- PART B: TIME SERIES & FORECASTING ---
print("--> Generating Time Series Analysis...")

# Daily Aggregations
daily_bio = df_bio.groupby('date')[['bio_age_5_17', 'bio_age_17_']].sum().sort_index()
daily_enr = df_enr.groupby('date')[['age_0_5', 'age_5_17', 'age_18_greater']].sum().sort_index()
daily_demo = df_demo.groupby('date')[['demo_age_5_17', 'demo_age_17_']].sum().sort_index()

# 8. Daily Biometric Trend
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(daily_bio.index, daily_bio['bio_age_5_17'], label='Age 5-17', color='#3498db', alpha=0.4)
ax.plot(daily_bio.index, daily_bio['bio_age_17_'], label='Age 17+', color='#e74c3c', alpha=0.3)

# 7-day Moving Average for clarity
ax.plot(daily_bio.index, daily_bio['bio_age_5_17'].rolling(7).mean(), color='#2980b9', lw=2, label='5-17 (7d Avg)')
ax.plot(daily_bio.index, daily_bio['bio_age_17_'].rolling(7).mean(), color='#c0392b', lw=2, label='17+ (7d Avg)')

ax.set_title('Daily Biometric Updates Trend (with 7-day Moving Average)', fontweight='bold')
ax.legend()
save_viz('08_daily_bio_trend.png')

# 9. Daily Enrolment Stackplot
fig, ax = plt.subplots(figsize=(12, 6))
ax.stackplot(daily_enr.index, daily_enr['age_0_5'], daily_enr['age_5_17'], daily_enr['age_18_greater'],
             labels=['0-5', '5-17', '18+'], colors=['#2ecc71', '#3498db', '#9b59b6'], alpha=0.8)
ax.set_title('Daily Enrolment by Age Group', fontweight='bold')
ax.legend(loc='upper left')
save_viz('09_daily_enr_trend.png')

# 10. Day of Week Pattern
df_bio['dow'] = df_bio['date'].dt.day_name()
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_stats = df_bio.groupby('dow')['bio_age_5_17'].sum().reindex(days_order)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(dow_stats.index, dow_stats.values, color=plt.cm.Set2(np.arange(7)), edgecolor='black')
ax.set_title('Biometric Updates by Day of Week', fontweight='bold')
save_viz('10_day_of_week.png')

# 11. Monthly Trend
df_bio['month'] = df_bio['date'].dt.to_period('M')
monthly_stats = df_bio.groupby('month')[['bio_age_5_17', 'bio_age_17_']].sum()
monthly_stats.index = monthly_stats.index.astype(str)

fig, ax = plt.subplots(figsize=(12, 6))
monthly_stats.plot(kind='bar', ax=ax, width=0.8, color=['#3498db', '#e74c3c'])
ax.set_title('Monthly Biometric Updates', fontweight='bold')
plt.xticks(rotation=45)
save_viz('11_monthly_trend.png')

# 12. Combined Trend (Dual Axis)
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel('Date')
ax1.set_ylabel('Enrolments', color='#2ecc71', fontweight='bold')
ax1.plot(daily_enr.index, daily_enr.sum(axis=1), color='#2ecc71', label='Total Enrolments')
ax1.tick_params(axis='y', labelcolor='#2ecc71')

ax2 = ax1.twinx()
ax2.set_ylabel('Biometric Updates', color='#3498db', fontweight='bold')
ax2.plot(daily_bio.index, daily_bio.sum(axis=1), color='#3498db', linestyle='--', label='Total Updates')
ax2.tick_params(axis='y', labelcolor='#3498db')
plt.title('Combined Trend: Enrolment vs Updates', fontsize=14, fontweight='bold')
save_viz('12_combined_trend.png')

# 13. Holt-Winters Forecasting
print("--> Running Robust Forecasting (Holt-Winters)...")

# Prepare Time Series - Handle daily gaps (Interpolate instead of fill 0 for smoother HW)
ts_data = daily_bio['bio_age_5_17'].asfreq('D').interpolate(method='linear')

# Holt-Winters Exponential Smoothing with Multiplicative Seasonality
# Seasonal period = 7 (weekly)
model = ExponentialSmoothing(ts_data, 
                             seasonal_periods=7, 
                             trend='add', 
                             seasonal='add').fit()

# Forecast
steps = 30
forecast = model.forecast(steps)

fig, ax = plt.subplots(figsize=(12, 6))

# Plot historical (Last 90 days)
ax.plot(ts_data.index[-90:], ts_data[-90:], label='Historical Data', color='#34495e', alpha=0.7)

# Plot Forecast
ax.plot(forecast.index, forecast, label='30-Day Forecast', color='#e67e22', linestyle='-', lw=2.5)

# Visual polish
ax.fill_between(forecast.index, forecast*0.85, forecast*1.15, color='#e67e22', alpha=0.1, label='Confidence Zone (±15%)')

ax.set_title('Aadhaar Biometric Demand Forecast (Holt-Winters Seasonal Smoothing)', fontweight='bold')
ax.legend()
save_viz('13_demand_forecast.png')

# --- PART C: COMPARATIVE ANALYSIS ---
print("--> Generating Comparative Charts...")

# 14. Top States Bar
top_states = state_stats.nlargest(15, 'bio_age_5_17').sort_values('bio_age_5_17')
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(top_states['state'], top_states['bio_age_5_17'], color=plt.cm.viridis(np.linspace(0.2, 0.8, 15)))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
ax.set_title('Top 15 States: Child Biometric Updates', fontweight='bold')
save_viz('14_state_comparison.png')

# 15. Butterfly Chart (Child vs Adult)
top_total = state_stats.nlargest(15, 'total_bio').sort_values('bio_age_5_17')
fig, ax = plt.subplots(figsize=(10, 8))
y = np.arange(len(top_total))
ax.barh(y, -top_total['bio_age_5_17']/1e6, color='#3498db', label='Children (5-17)')
ax.barh(y, top_total['bio_age_17_']/1e6, color='#e67e22', label='Adults (17+)')
ax.set_yticks(y)
ax.set_yticklabels(top_total['state'])
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Butterfly Chart: Child vs Adult Updates', fontweight='bold')
ax.legend()
save_viz('15_butterfly_chart.png')

# 16. Pareto Chart
sorted_load = state_stats.sort_values('total_bio', ascending=False)
sorted_load['cum_pct'] = sorted_load['total_bio'].cumsum() / sorted_load['total_bio'].sum() * 100
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(sorted_load['state'], sorted_load['total_bio'], color='#3498db')
ax1.tick_params(axis='x', rotation=90, labelsize=8)
ax2 = ax1.twinx()
ax2.plot(sorted_load['state'], sorted_load['cum_pct'], color='red', marker='o', ms=3)
ax2.axhline(80, color='k', linestyle='--', alpha=0.5)
ax1.set_title('Pareto Analysis: State Load', fontweight='bold')
save_viz('16_pareto_chart.png')

# 17. Top Districts
top_dists = dist_stats.nlargest(20, 'bio_age_5_17').sort_values('bio_age_5_17')
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(top_dists['district'], top_dists['bio_age_5_17'], color='#c0392b')
ax.set_title('Top 20 Districts: Biometric Updates', fontweight='bold')
save_viz('17_top_districts.png')

# --- PART D: STATISTICAL INSIGHTS ---
print("--> Generating Statistical Insights...")

# 18. Enrolment vs Bio Scatter
fig, ax = plt.subplots(figsize=(8, 8))
sns.regplot(data=state_stats, x='total_enr', y='total_bio', ax=ax, 
            scatter_kws={'s': 100, 'edgecolor': 'k', 'alpha': 0.7})
ax.set_title('Correlation: Enrolment vs Updates', fontweight='bold')
save_viz('18_correlation_scatter.png')

# 19. Age Distribution Pie
totals = [state_stats['age_0_5'].sum(), state_stats['age_5_17'].sum(), state_stats['age_18_greater'].sum()]
fig, ax = plt.subplots()
ax.pie(totals, labels=['0-5', '5-17', '18+'], autopct='%1.1f%%', colors=['#2ecc71', '#3498db', '#95a5a6'])
ax.set_title('Overall Enrolment Age Distribution', fontweight='bold')
save_viz('19_age_distribution.png')

# 20. Child vs Adult Scatter (Size = Pincodes)
fig, ax = plt.subplots(figsize=(12, 10))
sc = ax.scatter(state_stats['bio_age_5_17'], state_stats['bio_age_17_'], 
                s=state_stats['num_pincodes']/5, alpha=0.6, 
                c=state_stats['total_enr'], cmap='viridis', edgecolors='white')
plt.colorbar(sc, label='Total Enrolments')

# Annotate Top 10 States
top_10_states = state_stats.nlargest(10, 'total_bio')
for _, row in top_10_states.iterrows():
    ax.annotate(row['state'].title(), 
                (row['bio_age_5_17'], row['bio_age_17_']),
                fontsize=9, fontweight='bold', alpha=0.8,
                xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('Child Biometric Updates (5-17)')
ax.set_ylabel('Adult Biometric Updates (17+)')
ax.set_title('State-wise Child vs Adult Workload (Size = Unique Pincodes)', fontweight='bold')
save_viz('20_bubble_chart.png')

# --- PART E: MACHINE LEARNING & ANOMALY DETECTION ---
print("--> Running Anomaly Detection...")

# Feature Preparation
features = ['total_bio', 'UER', 'child_share']
X = dist_stats[features].fillna(0)

# Isolation Forest Model
iso = IsolationForest(contamination=0.02, random_state=42)
dist_stats['anomaly_score'] = iso.fit_predict(X)
anomalies = dist_stats[dist_stats['anomaly_score'] == -1]

print(f"   >> Identified {len(anomalies)} anomalies out of {len(dist_stats)} districts.")

# Save Report
anomalies[['state', 'district', 'total_bio', 'UER', 'child_share']].to_csv(
    os.path.join(OUTPUT_DIR, 'anomaly_detection_report.csv'), index=False
)
print("   Saved: anomaly_detection_report.csv")

# 21. Anomaly Scatter Plot
fig, ax = plt.subplots(figsize=(12, 7))
normal = dist_stats[dist_stats['anomaly_score'] == 1]
ax.scatter(normal['total_enr'], normal['total_bio'], c='blue', alpha=0.15, label='Normal', s=15)
ax.scatter(anomalies['total_enr'], anomalies['total_bio'], c='red', alpha=0.9, label='Anomaly Flag', s=60, edgecolors='black')

# Label Anomalous Districts (Filter for prominent ones to avoid clutter)
for _, row in anomalies.iterrows():
    ax.annotate(row['district'].title(), 
                (row['total_enr'], row['total_bio']),
                fontsize=8, xytext=(3, 3), textcoords='offset points', alpha=0.7)

ax.set_xlabel('Total Enrolments')
ax.set_ylabel('Total Biometric Updates')
ax.set_title('Anomaly Detection: High-Intensity Operational Irregularities', fontweight='bold')
ax.legend()
save_viz('21_anomaly_scatter.png')


# 22. UER Distribution (Stress Analysis)
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(dist_stats['UER'], bins=50, kde=True, color='#e67e22', ax=ax)
ax.axvline(2.0, color='red', linestyle='--', label='Anomaly Threshold (>2.0)')
ax.set_title('Distribution of Update-to-Enrolment Ratio (UER)', fontweight='bold')
ax.set_xlabel('UER (Updates per Enrolment)')
ax.legend()
save_viz('22_uer_distribution.png')

print("="*60)
print(f"✅ Dashboard Generation Complete. All 21 visualizations stored in: {OUTPUT_DIR}")
print("="*60)