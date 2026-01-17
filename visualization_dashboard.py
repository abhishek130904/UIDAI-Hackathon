"""
UIDAI Visualization Dashboard v3
Uses actual master CSV files with real date, state, district data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
import geopandas as gpd
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
STATE_SHP = os.path.join(SCRIPT_DIR, '../maps-master/Survey-of-India-Index-Maps/Boundaries/India-States.shp')
DISTRICT_SHP = os.path.join(SCRIPT_DIR, '../maps-master/Survey-of-India-Index-Maps/Boundaries/India-Districts-2011Census.shp')
OUTPUT_DIR = 'visualizations'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("UIDAI VISUALIZATION DASHBOARD v3")
print("=" * 60)

# ============================================================
# DATA LOADING
# ============================================================
print("\n[Loading] Master CSV files...")

enrolment_df = pd.read_csv('master_aadhar_enrolment_data.csv')
biometric_df = pd.read_csv('master_biometric_data.csv')
demographic_df = pd.read_csv('master_demographic_data.csv')

print(f"    Enrolment: {len(enrolment_df):,} rows")
print(f"    Biometric: {len(biometric_df):,} rows")
print(f"    Demographic: {len(demographic_df):,} rows")

# Parse dates
for df in [enrolment_df, biometric_df, demographic_df]:
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')

# Standardize state names (fix typos)
def clean_text(text):
    if pd.isna(text):
        return 'unknown'
    # Convert to string, lowercase, strip
    text = str(text).lower().strip()
    # Replace & with and
    text = text.replace('&', ' and ')
    # Replace special chars containing ? or - or others with space
    text = ''.join(e if e.isalnum() else ' ' for e in text)
    # Remove multiple spaces
    text = ' '.join(text.split())
    
    # Specific replacements for consistency
    corrections = {
        'uttaranchal': 'uttarakhand',
        'orissa': 'odisha',
        'pondicherry': 'puducherry',
        'dadra and nagar haveli': 'dadra and nagar haveli and daman and diu',
        'daman and diu': 'dadra and nagar haveli and daman and diu',
        'the dadra and nagar haveli and daman and diu': 'dadra and nagar haveli and daman and diu',
        'dadara and nagar havelli': 'dadra and nagar haveli and daman and diu',
        'delhi': 'nct of delhi',
        'jammu and kashmir': 'jammu and kashmir',
        'andaman and nicobar islands': 'andaman and nicobar island',
        'arunachal pradesh': 'arunanchal pradesh', # Match shapefile spelling
        'west bangal': 'west bengal',
        'westbengal': 'west bengal',
        'ladakh': 'jammu and kashmir', # Map to J&K if shapefile is old
    }
    return corrections.get(text, text)

print("    Cleaning text columns...")
for df in [enrolment_df, biometric_df, demographic_df]:
    df['state'] = df['state'].apply(clean_text)
    df['district'] = df['district'].apply(clean_text)
    df['state_normalized'] = df['state']

# Load shapefile
print("    Loading India shapefile...")
india_states = gpd.read_file(STATE_SHP)
# Apply same cleaning
india_states['state_normalized'] = india_states['ST_NM'].apply(clean_text)

print("    Loading India District shapefile...")
india_districts = gpd.read_file(DISTRICT_SHP)
india_districts['state_normalized'] = india_districts['ST_NM'].apply(clean_text)
india_districts['district_normalized'] = india_districts['DISTRICT'].apply(clean_text)

# ============================================================
# AGGREGATE DATA
# ============================================================
print("\n[Processing] Aggregating data...")

# State-level aggregations
state_bio = biometric_df.groupby('state_normalized').agg({
    'bio_age_5_17': 'sum',
    'bio_age_17_': 'sum',
    'pincode': 'nunique'
}).rename(columns={'pincode': 'num_pincodes'}).reset_index()

state_enr = enrolment_df.groupby('state_normalized').agg({
    'age_0_5': 'sum',
    'age_5_17': 'sum',
    'age_18_greater': 'sum'
}).reset_index()

state_demo = demographic_df.groupby('state_normalized').agg({
    'demo_age_5_17': 'sum',
    'demo_age_17_': 'sum'
}).reset_index()

# Merge state stats
state_stats = state_bio.merge(state_enr, on='state_normalized', how='outer')
state_stats = state_stats.merge(state_demo, on='state_normalized', how='outer')
state_stats.fillna(0, inplace=True)
state_stats['total_bio'] = state_stats['bio_age_5_17'] + state_stats['bio_age_17_']
state_stats['total_enr'] = state_stats['age_0_5'] + state_stats['age_5_17'] + state_stats['age_18_greater']

# Merge with shapefile
india_map = india_states.merge(state_stats, on='state_normalized', how='left')

# Time-series aggregations (daily)
daily_bio = biometric_df.groupby('date').agg({'bio_age_5_17': 'sum', 'bio_age_17_': 'sum'}).reset_index()
daily_enr = enrolment_df.groupby('date').agg({'age_0_5': 'sum', 'age_5_17': 'sum', 'age_18_greater': 'sum'}).reset_index()
daily_demo = demographic_df.groupby('date').agg({'demo_age_5_17': 'sum', 'demo_age_17_': 'sum'}).reset_index()

# District-level Aggregations
district_bio = biometric_df.groupby(['state', 'district']).agg({
    'bio_age_5_17': 'sum', 'bio_age_17_': 'sum',
    'pincode': 'nunique'
}).rename(columns={'pincode': 'num_pincodes'}).reset_index()

district_enr = enrolment_df.groupby(['state', 'district']).agg({
    'age_0_5': 'sum', 'age_5_17': 'sum', 'age_18_greater': 'sum'
}).reset_index()

district_demo = demographic_df.groupby(['state', 'district']).agg({
    'demo_age_5_17': 'sum', 'demo_age_17_': 'sum'
}).reset_index()

# Merge all district stats
district_stats = district_bio.merge(district_enr, on=['state', 'district'], how='outer')
district_stats = district_stats.merge(district_demo, on=['state', 'district'], how='outer')
district_stats = district_stats.fillna(0)

district_stats['district_normalized'] = district_stats['district'] # Already cleaned
district_stats['state_normalized'] = district_stats['state']       # Already cleaned

# Calculate totals
district_stats['total_enr'] = district_stats['age_0_5'] + district_stats['age_5_17'] + district_stats['age_18_greater']
district_stats['total_demo'] = district_stats['demo_age_5_17'] + district_stats['demo_age_17_']

# Merge District Data with Shapefile
print("    Merging District Data...")
india_districts_map = india_districts.merge(district_stats, on=['state_normalized', 'district_normalized'], how='left')
india_districts_map = india_districts_map.fillna(0)

viz_count = 0

# ============================================================
# SECTION A: GEOSPATIAL (INDIA MAPS)
# ============================================================
print("\n" + "=" * 60)
print("SECTION A: GEOSPATIAL MAPS")
print("=" * 60)

# Helper for robust plotting
def plot_robust_map(gdf, column, ax, title, cmap='YlOrRd', label=''):
    # Cap at 95th percentile to handle skew
    vmax = np.percentile(gdf[column], 95)
    if vmax == 0: vmax = gdf[column].max()
    gdf.plot(column=column, ax=ax, legend=True, cmap=cmap,
             missing_kwds={'color': 'lightgrey'}, vmin=0, vmax=vmax,
             legend_kwds={'label': label, 'shrink': 0.5})
    ax.set_xlim([68, 98]); ax.set_ylim([6, 38]); ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')

# 1. Biometric Updates Map
viz_count += 1
print(f"\n[{viz_count}] Biometric Updates Map...")
fig, ax = plt.subplots(figsize=(12, 14))
plot_robust_map(india_map, 'bio_age_5_17', ax, 'Biometric Updates (Age 5-17)', 'YlOrRd', 'Updates')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_bio_updates_map.png', bbox_inches='tight')
plt.close()
print(f"    Saved: 01_bio_updates_map.png")

# 2. Enrolment Map
viz_count += 1
print(f"\n[{viz_count}] Enrolment Map...")
fig, ax = plt.subplots(figsize=(12, 14))
plot_robust_map(india_map, 'total_enr', ax, 'Total Aadhaar Enrolments', 'Blues', 'Enrolments')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_enrolment_map.png', bbox_inches='tight')
plt.close()
print(f"    Saved: 02_enrolment_map.png")

# 3. Demographic Updates Map
viz_count += 1
print(f"\n[{viz_count}] Demographic Updates Map...")
fig, ax = plt.subplots(figsize=(12, 14))
india_map['total_demo'] = india_map['demo_age_5_17'].fillna(0) + india_map['demo_age_17_'].fillna(0)
plot_robust_map(india_map, 'total_demo', ax, 'Demographic Updates', 'Greens', 'Updates')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_demo_updates_map.png', bbox_inches='tight')
plt.close()
print(f"    Saved: 03_demo_updates_map.png")

# 3b. District-Wise Biometric Map (New)
viz_count += 1
print(f"\n[{viz_count}] District-Wise Biometric Map...")
fig, ax = plt.subplots(figsize=(12, 14))
plot_robust_map(india_districts_map, 'bio_age_5_17', ax, 'District-Wise Biometric Updates (5-17)', 'YlOrRd', 'Updates')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03b_district_bio_map.png', bbox_inches='tight')
plt.close()
print(f"    Saved: 03b_district_bio_map.png")

# 3c. District-Wise Enrolment Map (New)
viz_count += 1
print(f"\n[{viz_count}] District-Wise Enrolment Map...")
fig, ax = plt.subplots(figsize=(12, 14))
plot_robust_map(india_districts_map, 'total_enr', ax, 'District-Wise Total Enrolments', 'Blues', 'Enrolments')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03c_district_enr_map.png', bbox_inches='tight')
plt.close()
print(f"    Saved: 03c_district_enr_map.png")

# 3d. District-Wise Demographic Map (New)
viz_count += 1
print(f"\n[{viz_count}] District-Wise Demographic Map...")
fig, ax = plt.subplots(figsize=(12, 14))
plot_robust_map(india_districts_map, 'total_demo', ax, 'District-Wise Demographic Updates', 'Greens', 'Updates')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03d_district_demo_map.png', bbox_inches='tight')
plt.close()
print(f"    Saved: 03d_district_demo_map.png")

# ============================================================
# SECTION B: TIME-SERIES (REAL DATES)
# ============================================================
print("\n" + "=" * 60)
print("SECTION B: TIME-SERIES")
print("=" * 60)

# 4. Daily Biometric Trend
viz_count += 1
print(f"\n[{viz_count}] Daily Biometric Trend...")
daily_bio_sorted = daily_bio.dropna().sort_values('date')
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(daily_bio_sorted['date'], daily_bio_sorted['bio_age_5_17'], label='Age 5-17', color='#3498db', lw=1.5)
ax.plot(daily_bio_sorted['date'], daily_bio_sorted['bio_age_17_'], label='Age 17+', color='#e74c3c', lw=1.5, alpha=0.8)
ax.set_xlabel('Date'); ax.set_ylabel('Biometric Updates'); ax.legend()
ax.set_title('Daily Biometric Updates Over Time', fontsize=14, fontweight='bold')
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_daily_bio_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: 04_daily_bio_trend.png")

# 5. Daily Enrolment Trend
viz_count += 1
print(f"\n[{viz_count}] Daily Enrolment Trend...")
daily_enr_sorted = daily_enr.dropna().sort_values('date')
fig, ax = plt.subplots(figsize=(14, 6))
ax.stackplot(daily_enr_sorted['date'], 
             daily_enr_sorted['age_0_5'], daily_enr_sorted['age_5_17'], daily_enr_sorted['age_18_greater'],
             labels=['0-5 yrs', '5-17 yrs', '18+ yrs'], colors=['#2ecc71', '#3498db', '#9b59b6'], alpha=0.8)
ax.set_xlabel('Date'); ax.set_ylabel('Enrolments'); ax.legend(loc='upper left')
ax.set_title('Daily Enrolments by Age Group', fontsize=14, fontweight='bold')
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_daily_enrolment.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: 05_daily_enrolment.png")

# 6. Weekly Pattern (Day of Week)
viz_count += 1
print(f"\n[{viz_count}] Day-of-Week Pattern...")
biometric_df['day_of_week'] = biometric_df['date'].dt.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_bio = biometric_df.groupby('day_of_week')['bio_age_5_17'].sum().reindex(day_order)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(dow_bio.index, dow_bio.values, color=plt.cm.Set2(np.arange(7)), edgecolor='black')
ax.set_xlabel('Day of Week'); ax.set_ylabel('Total Biometric Updates')
ax.set_title('Biometric Updates by Day of Week', fontsize=14, fontweight='bold')
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_dayofweek_pattern.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: 06_dayofweek_pattern.png")

# 7. Monthly Aggregation
viz_count += 1
print(f"\n[{viz_count}] Monthly Trend...")
biometric_df['month'] = biometric_df['date'].dt.to_period('M')
monthly_bio = biometric_df.groupby('month').agg({'bio_age_5_17': 'sum', 'bio_age_17_': 'sum'}).reset_index()
monthly_bio['month'] = monthly_bio['month'].astype(str)

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(monthly_bio))
width = 0.35
ax.bar(x - width/2, monthly_bio['bio_age_5_17'], width, label='Age 5-17', color='#3498db')
ax.bar(x + width/2, monthly_bio['bio_age_17_'], width, label='Age 17+', color='#e74c3c')
ax.set_xticks(x); ax.set_xticklabels(monthly_bio['month'], rotation=45)
ax.set_xlabel('Month'); ax.set_ylabel('Biometric Updates'); ax.legend()
ax.set_title('Monthly Biometric Updates', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_monthly_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: 07_monthly_trend.png")

# ============================================================
# SECTION C: COMPARATIVE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("SECTION C: COMPARATIVE ANALYSIS")
print("=" * 60)

# 8. State Comparison Bar Chart
viz_count += 1
print(f"\n[{viz_count}] State Comparison...")
top_states = state_stats.nlargest(15, 'bio_age_5_17').sort_values('bio_age_5_17')
fig, ax = plt.subplots(figsize=(10, 10))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_states)))
ax.barh(top_states['state_normalized'], top_states['bio_age_5_17'], color=colors)
ax.set_xlabel('Biometric Updates (5-17)'); ax.set_ylabel('State')
ax.set_title('Top 15 States: Child Biometric Updates', fontsize=14, fontweight='bold')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_state_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: 08_state_comparison.png")

# 9. Butterfly Chart (Child vs Adult)
viz_count += 1
print(f"\n[{viz_count}] Butterfly Chart...")
state_age = state_stats.nlargest(15, 'total_bio').sort_values('bio_age_5_17')
fig, ax = plt.subplots(figsize=(10, 10))
y_pos = np.arange(len(state_age))
ax.barh(y_pos, -state_age['bio_age_5_17']/1e6, color='#3498db', label='Children (5-17)', alpha=0.8)
ax.barh(y_pos, state_age['bio_age_17_']/1e6, color='#e67e22', label='Adults (17+)', alpha=0.8)
ax.set_yticks(y_pos); ax.set_yticklabels(state_age['state_normalized'])
ax.axvline(0, color='black', lw=1); ax.set_xlabel('Updates (Millions)')
ax.set_title('Child vs Adult Biometric Updates', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/09_butterfly_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: 09_butterfly_chart.png")

# 10. Pareto Chart (Fixed Labels)
viz_count += 1
print(f"\n[{viz_count}] Pareto Chart...")
state_load = state_stats.sort_values('bio_age_5_17', ascending=False).copy()
state_load['cumulative_pct'] = state_load['bio_age_5_17'].cumsum() / state_load['bio_age_5_17'].sum() * 100

fig, ax1 = plt.subplots(figsize=(14, 8)) # Increased height for labels
ax1.bar(range(len(state_load)), state_load['bio_age_5_17'], color='#3498db', alpha=0.8)
ax1.set_ylabel('Updates', color='#3498db')
ax1.set_xlabel('State')
# Set x-ticks properly
ax1.set_xticks(range(len(state_load)))
ax1.set_xticklabels(state_load['state_normalized'], rotation=90, fontsize=10)

ax2 = ax1.twinx()
ax2.plot(range(len(state_load)), state_load['cumulative_pct'], color='#e74c3c', lw=2, marker='o', ms=3)
ax2.axhline(80, color='#e74c3c', ls='--', alpha=0.7)
ax2.set_ylabel('Cumulative %', color='#e74c3c')
ax1.set_title('Pareto Chart: State-wise Load Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/10_pareto_chart.png', bbox_inches='tight')
plt.close()
print(f"    Saved: 10_pareto_chart.png")

# 11. District Heatmap (Top Districts)
viz_count += 1
print(f"\n[{viz_count}] Top Districts...")
top_districts = district_bio.nlargest(20, 'bio_age_5_17')
fig, ax = plt.subplots(figsize=(10, 8))
top_districts_sorted = top_districts.sort_values('bio_age_5_17')
colors = plt.cm.YlOrRd(np.linspace(0.3, 1, len(top_districts_sorted)))
ax.barh(top_districts_sorted['district'], top_districts_sorted['bio_age_5_17'], color=colors)
ax.set_xlabel('Biometric Updates (5-17)')
ax.set_title('Top 20 Districts: Child Biometric Updates', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/11_top_districts.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: 11_top_districts.png")

# ============================================================
# SECTION D: CORRELATION & STATISTICS
# ============================================================
print("\n" + "=" * 60)
print("SECTION D: STATISTICS")
print("=" * 60)

# 12. Scatter: Enrolment vs Biometric
viz_count += 1
print(f"\n[{viz_count}] Enrolment vs Biometric Scatter...")
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(state_stats['total_enr'], state_stats['total_bio'], s=100, alpha=0.7, c='#3498db', edgecolors='black')
# Regression line
z = np.polyfit(state_stats['total_enr'], state_stats['total_bio'], 1)
p = np.poly1d(z)
x_line = np.linspace(state_stats['total_enr'].min(), state_stats['total_enr'].max(), 100)
ax.plot(x_line, p(x_line), 'r--', lw=2, label=f'Trend (slope={z[0]:.2f})')
ax.set_xlabel('Total Enrolments'); ax.set_ylabel('Total Biometric Updates')
ax.set_title('Enrolment vs Biometric Updates by State', fontsize=14, fontweight='bold')
ax.legend()
# Annotate top states
for _, row in state_stats.nlargest(3, 'total_bio').iterrows():
    ax.annotate(row['state_normalized'], (row['total_enr'], row['total_bio']), fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/12_enrol_vs_bio.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: 12_enrol_vs_bio.png")

# 13. Age Distribution Pie
viz_count += 1
print(f"\n[{viz_count}] Age Distribution Pie...")
total_enr = state_stats[['age_0_5', 'age_5_17', 'age_18_greater']].sum()
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['#2ecc71', '#3498db', '#9b59b6']
explode = (0.02, 0.02, 0.02)
ax.pie(total_enr, labels=['Age 0-5', 'Age 5-17', 'Age 18+'], autopct='%1.1f%%', 
       colors=colors, explode=explode, startangle=90, shadow=True)
ax.set_title('Enrolment Age Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/13_age_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: 13_age_distribution.png")

# 14. Child vs Adult Bio (Scatter per state)
viz_count += 1
print(f"\n[{viz_count}] Child vs Adult Biometric...")
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(state_stats['bio_age_5_17'], state_stats['bio_age_17_'], 
                     s=state_stats['num_pincodes']/10, alpha=0.6, c=state_stats['total_enr'], cmap='viridis')
plt.colorbar(scatter, label='Total Enrolments')
ax.set_xlabel('Child Bio Updates (5-17)'); ax.set_ylabel('Adult Bio Updates (17+)')
ax.set_title('Child vs Adult Biometric (Size = #Pincodes)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/14_child_vs_adult.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: 14_child_vs_adult.png")

# ============================================================
# SECTION E: OPERATIONAL INSIGHTS
# ============================================================
print("\n" + "=" * 60)
print("SECTION E: OPERATIONAL INSIGHTS")
print("=" * 60)

# 15. Pincode Load Density Map (District Level)
viz_count += 1
print(f"\n[{viz_count}] Pincode Load Density Map...")
# Calculate avg updates per pincode for each district
india_districts_map['load_per_pincode'] = india_districts_map['bio_age_5_17'] / (india_districts_map['num_pincodes'] + 1e-6)

fig, ax = plt.subplots(figsize=(12, 14))
plot_robust_map(india_districts_map, 'load_per_pincode', ax, 'Pincode Load Intensity (Avg Updates per Pincode)', 'RdPu', 'Avg Updates')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/15_pincode_load_map.png', bbox_inches='tight')
plt.close()
print(f"    Saved: 15_pincode_load_map.png")

# ============================================================
# COMPLETION
# ============================================================
print("\n" + "=" * 60)
print(f"✅ ALL {viz_count} VISUALIZATIONS GENERATED!")
print("=" * 60)
print(f"\nOutput: {os.path.abspath(OUTPUT_DIR)}/")
for i, f in enumerate(sorted(os.listdir(OUTPUT_DIR)), 1):
    print(f"  {i:2d}. {f}")
print("\n🎉 Complete!")
