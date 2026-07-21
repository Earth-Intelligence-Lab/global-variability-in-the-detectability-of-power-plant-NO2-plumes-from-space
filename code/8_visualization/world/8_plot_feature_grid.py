import pandas as pd
import geopandas
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings

# Ignore ignorable warnings
warnings.filterwarnings('ignore')

# Set aesthetic styling from reference code
plt.rcdefaults()
plt.rcParams.update({
    'font.family': 'Nimbus Roman',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 20,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 24,
    'axes.linewidth': 1.5,
    'axes.edgecolor': '#2c3e50',
    'axes.labelcolor': '#2c3e50',
    'text.color': '#2c3e50'
})

# Feature name mapping (unchanged)
feature_name_mapping = {
    'sensor_azimuth_angle': 'Sensor Azimuth Angle',
    'sensor_zenith_angle': 'Sensor Zenith Angle',
    'sensor_altitude': 'Sensor Altitude',
    'scaled_small_pixel_variance': 'Scaled Small-Pixel Variance',
    'annual_nox_emission': 'Annual NOx Emission',
    'hourly_emission_rate': 'Hourly NOx Emission',
    '10m_wind_speed': '10 m Wind Speed',
    'wind_speed_10m': '10 m Wind Speed',
    'wind_speed': 'Wind Speed',
    'cloud_albedo': 'Cloud Albedo',
    'cloud_pressure': 'Cloud Pressure',
    'cloud_fraction': 'Cloud Fraction',
    'cloud_albedo_crb': 'Cloud Albedo',
    'cloud_pressure_crb': 'Cloud Pressure',
    'cloud_fraction_crb': 'Cloud Fraction',
    'solar_zenith_angle': 'Solar Zenith Angle',
    'solar_azimuth_angle': 'Solar Azimuth Angle',
    'apparent_scene_pressure': 'Apparent Scene Pressure',
    'aerosol_index_354_388': 'Aerosol Index',
    't2m': '2 m Temperature',
    'tcwv': 'Total Column Water Vapour',
    'tisr': 'TOA Incident Solar Radiation',
    'surface_classification': 'Surface Classification',
    'surface_pressure': 'Surface Pressure',
    'snow_ice_flag': 'Snow/Ice Flag',
    'scene_albedo': 'Scene Albedo',
    'surface_albedo_nitrogendioxide_window': 'Surface Albedo Nitrogen-Dioxide Window',
    'tropospheric_NO2_column_number_density': 'Tropospheric NO2 Column Number Density',
    'surface_albedo': 'Surface Albedo',
    'surface_altitude': 'Surface Altitude',
    'surface_altitude_precision': 'Surface Altitude Precision',
    'nearby_cities_count_20km': 'Nearby Cities Count 20 km',
    'nearby_cities_count_50km': 'Nearby Cities Count 50 km',
    'nearby_cities_count_100km': 'Nearby Cities Count 100 km',
    'nearby_cities_pop_20km': 'Nearby Cities Pop 20 km',
    'nearby_cities_pop_50km': 'Nearby Cities Pop 50 km',
    'nearby_cities_pop_100km': 'Nearby Cities Pop 100 km',
    'nearby_plants_count_20km': 'Nearby Plants Count 20 km',
    'nearby_plants_count_50km': 'Nearby Plants Count 50 km',
    'nearby_plants_count_100km': 'Nearby Plants Count 100 km',
    'total_emission_20km': 'Total Emission 20 km',
    'total_emission_50km': 'Total Emission 50 km',
    'total_emission_100km': 'Total Emission 100 km',
    'percentage_emission_20km': 'Percentage Emission 20 km',
    'percentage_emission_50km': 'Percentage Emission 50 km',
    'percentage_emission_100km': 'Percentage Emission 100 km',
    'no2_std_radius': 'NO2 Std 50 km',
    'no2_mean_radius': 'NO2 Mean 50 km',
    'no2_frac_valid_radius': 'NO2 Frac Valid 50 km',
    'primary_fuel_type': 'Primary Fuel Type',
    'NOx Mass (lbs)': 'Hourly NOx Emission',
    'Nox Mass (Lbs)': 'Hourly NOx Emission',
}

# Feature groups and colors (unchanged)
feature_groups = {
    'sensor': {'features': ['sensor_azimuth_angle', 'sensor_zenith_angle', 'sensor_altitude', 'scaled_small_pixel_variance'], 'colors': ['#FF1493', '#FF69B4', '#FFB6C1', '#FFC0CB', '#FFE4E1']},
    'power_plant': {'features': ['annual_nox_emission', 'hourly_emission_rate', 'NOx Mass (lbs)', 'Nox Mass (Lbs)', 'primary_fuel_type'], 'colors': ['#00FF00', '#32CD32', '#228B22', '#006400', '#2E8B57']},
    'meteorology': {'features': ['10m_wind_speed', 'wind_speed_10m', 'wind_speed', 'cloud_albedo', 'cloud_pressure', 'cloud_fraction', 'cloud_albedo_crb', 'cloud_pressure_crb', 'cloud_fraction_crb', 'solar_zenith_angle', 'solar_azimuth_angle', 'apparent_scene_pressure', 'aerosol_index_354_388', 't2m', 'tcwv', 'tisr'], 'colors': ['#87CEEB', '#4682B4', '#4169E1', '#0000FF', '#000080']},
    'environment': {'features': ['surface_classification', 'surface_pressure', 'snow_ice_flag', 'scene_albedo', 'surface_albedo_nitrogendioxide_window', 'surface_albedo', 'surface_altitude', 'surface_altitude_precision', 'tropospheric_no2_column_number_density', 'nearby_cities_count_20km', 'nearby_cities_count_50km', 'nearby_cities_count_100km', 'nearby_cities_pop_20km', 'nearby_cities_pop_50km', 'nearby_cities_pop_100km', 'nearby_plants_count_20km', 'nearby_plants_count_50km', 'nearby_plants_count_100km', 'total_emission_20km', 'total_emission_50km', 'total_emission_100km', 'percentage_emission_20km', 'percentage_emission_50km', 'percentage_emission_100km'], 'colors': ['#FFFF00', '#FFE600', '#FFD700', '#FFC300', '#FFA500', '#FF8C00', '#FF6347', '#FF4500', '#FF2F00', '#FF0000']},
    'no2_statistics': {'features': ['no2_std_radius', 'no2_mean_radius', 'no2_frac_valid_radius'], 'colors': ['#9370DB', '#8B008B', '#4B0082', '#6A0DAD', '#9932CC']},
    'other': {'features': [], 'colors': ['#696969', '#808080', '#A9A9A9', '#C0C0C0', '#D3D3D3']}
}

# --- Utility Functions (Coloring) ---
def fade_color(hex_color, fade_factor=0.5):
    hex_code = hex_color.lstrip('#')
    rgb = [int(hex_code[i:i+2], 16)/255. for i in (0, 2, 4)]
    faded = [c + (1 - c) * fade_factor for c in rgb]
    return '#%02x%02x%02x' % tuple(int(v*255) for v in faded)

def assign_feature_group(feature_name):
    for group_name, group_info in feature_groups.items():
        if feature_name in group_info['features']:
            return group_name
    return 'other'

def create_feature_color_map(features, fade_factor=0.5):
    feature_to_group = {feature: assign_feature_group(feature) for feature in features}
    feature_colors = {}
    group_color_indices = {group: 0 for group in feature_groups}
    for feature in sorted(features):
        group = feature_to_group[feature]
        color_list = feature_groups[group]['colors']
        color_idx = group_color_indices[group] % len(color_list)
        feature_colors[feature] = fade_color(color_list[color_idx], fade_factor)
        group_color_indices[group] += 1
    return feature_colors

# --- Data Loading and Analysis (Unchanged) ---
print("Loading data...")
feature_analysis_us = pd.read_csv('/net/fs06/d3/rzhuang/TROPOMI_US/data/Run_20250623_203825/all_plants_contributions.csv')
feature_analysis_global = pd.read_csv('/net/fs06/d3/rzhuang/TROPOMI_world/data/Run_3/all_global_plants_contributions.csv')

nearby_features_to_exclude = [
    'nearby_plants_count_20km', 'total_emission_20km', 'percentage_emission_20km', 'nearby_plants_count_50km', 'total_emission_50km', 'percentage_emission_50km', 'nearby_plants_count_100km', 'total_emission_100km', 'percentage_emission_100km', 'nearby_cities_count_20km', 'nearby_cities_pop_20km', 'nearby_cities_count_50km', 'nearby_cities_pop_50km', 'nearby_cities_count_100km', 'nearby_cities_pop_100km'
]

def analyze_features(feature_df, grid_size, exclude_features=None):
    df_copy = feature_df.copy()
    df_copy['lon_bin'] = (df_copy['longitude'] // grid_size) * grid_size
    df_copy['lat_bin'] = (df_copy['latitude'] // grid_size) * grid_size
    def get_grid_summary(group):
        features = pd.melt(group, value_vars=[f'top_{i}_feature' for i in range(1, 5)])['value']
        directions = pd.melt(group, value_vars=[f'top_{i}_direction' for i in range(1, 5)])['value']
        combined = pd.DataFrame({'feature': features, 'direction': directions}).dropna()
        if exclude_features:
            combined = combined[~combined['feature'].isin(exclude_features)]
        top_feature_in_cell = combined['feature'].value_counts().nlargest(1)
        top_feat_name, top_feat_dir = None, None
        if not top_feature_in_cell.empty:
            top_feat_name = top_feature_in_cell.index[0]
            dominant_direction = combined[combined['feature'] == top_feat_name]['direction'].mode()
            if not dominant_direction.empty:
                top_feat_dir = '↑' if dominant_direction[0] == 'increases' else '↓'
        return pd.Series({'top_1_feat': top_feat_name, 'top_1_dir': top_feat_dir})
    grid_summary = df_copy.groupby(['lon_bin', 'lat_bin']).apply(get_grid_summary).reset_index()
    return grid_summary

print("Analyzing all scenarios...")
us_grid_with = analyze_features(feature_analysis_us, grid_size=2.0)
us_grid_without = analyze_features(feature_analysis_us, grid_size=2.0, exclude_features=nearby_features_to_exclude)
global_grid_with = analyze_features(feature_analysis_global, grid_size=4.0)
global_grid_without = analyze_features(feature_analysis_global, grid_size=4.0, exclude_features=nearby_features_to_exclude)

all_unique_features = pd.concat([
    us_grid_with['top_1_feat'], us_grid_without['top_1_feat'],
    global_grid_with['top_1_feat'], global_grid_without['top_1_feat']
]).dropna().unique()
color_map = create_feature_color_map(all_unique_features, fade_factor=0.5)


# --- New Plotting Setup ---
fig = plt.figure(figsize=(20, 26), facecolor='white')
fig.suptitle('Power Plant Feature Importance: Grid-based Spatial Analysis',
             fontsize=28, y=0.99, weight='bold', color="#000000")

gs = fig.add_gridspec(
    3, 2, height_ratios=[1, 1.2, 1.2], hspace=0.35, wspace=0.1,
    top=0.95, bottom=0.02, left=0.03, right=0.97
)

ax_us_with = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax_us_without = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax_global_with = fig.add_subplot(gs[1, :], projection=ccrs.PlateCarree())
ax_global_without = fig.add_subplot(gs[2, :], projection=ccrs.PlateCarree())

## ----------------------------------------------------------------
## Map Styling Functions (from reference)
## ----------------------------------------------------------------

def setup_us_map(ax, title):
    """Configures a Cartopy map with a U.S. focus."""
    ax.set_extent([-130, -65, 22, 52], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#fafafa', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='#e8f4ff', alpha=0.6)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2, color='#34495e', alpha=0.8)
    ax.add_feature(cfeature.STATES, linewidth=0.8, edgecolor='#7f8c8d', alpha=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=1.5, edgecolor='#2c3e50', alpha=0.7)
    gl = ax.gridlines(draw_labels=True, linestyle=':', alpha=0.3, color='#95a5a6')
    gl.top_labels, gl.right_labels = False, False
    gl.xlabel_style, gl.ylabel_style = {'size': 14, 'color': '#34495e'}, {'size': 14, 'color': '#34495e'}
    ax.set_title(title, fontsize=22, pad=10, weight='600', color="#000000")

def setup_world_map(ax, title):
    """Configures a Cartopy map with a global focus."""
    ax.set_extent([-180, 180, -60, 80], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#fafafa', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='#e8f4ff', alpha=0.6)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, color='#34495e', alpha=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='#7f8c8d', alpha=0.5)
    gl = ax.gridlines(draw_labels=True, linestyle=':', alpha=0.3, color='#95a5a6')
    gl.top_labels, gl.right_labels = False, False
    gl.xlabel_style, gl.ylabel_style = {'size': 14, 'color': '#34495e'}, {'size': 14, 'color': '#34495e'}
    ax.set_title(title, fontsize=22, pad=15, weight='600', color="#000000")

## ----------------------------------------------------------------
## Plotting and Legend Functions (Adapted for Grid Data)
## ----------------------------------------------------------------

def plot_grid_on_map(ax, grid_summary, grid_size):
    """Plots the grid data as rectangles on the Cartopy map."""
    for _, row in grid_summary.iterrows():
        if pd.notna(row['top_1_feat']):
            lon, lat = row['lon_bin'], row['lat_bin']
            color = color_map.get(row['top_1_feat'], 'lightgrey')
            
            # Add grid cell as a Rectangle patch
            rect = Rectangle((lon, lat), grid_size, grid_size,
                             facecolor=color, edgecolor='black', linewidth=0.3,
                             alpha=0.75, transform=ccrs.PlateCarree())
            ax.add_patch(rect)
            
            # Add direction arrow
            if pd.notna(row['top_1_dir']):
                arrow_color = 'darkgreen' if row['top_1_dir'] == '↑' else 'darkred'
                ax.text(lon + grid_size / 2, lat + grid_size / 2, row['top_1_dir'],
                        ha='center', va='center', fontsize=8, color=arrow_color, weight='bold',
                        transform=ccrs.PlateCarree(), zorder=10)

def create_enhanced_legend(ax, grid_summary, title_suffix, n_features, ncol, bbox_y):
    """Creates a styled legend for the top features in the grid summary."""
    top_features = grid_summary['top_1_feat'].value_counts().nlargest(n_features)
    legend_elements = []
    for feat, count in top_features.items():
        clean_name = feature_name_mapping.get(feat, feat)
        label = f"{clean_name} ({count:,} grids)"
        legend_elements.append(mpatches.Patch(color=color_map[feat], label=label, edgecolor='white', linewidth=1.5))

    legend = ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, bbox_y),
                       ncol=ncol, frameon=True, fancybox=True, shadow=True,
                       prop={'size': 11, 'weight': '500'}, title=f'Top Features {title_suffix}',
                       title_fontsize=13, columnspacing=2.0)
    legend.get_frame().set_facecolor('#ffffff')
    legend.get_frame().set_edgecolor('#e0e0e0')
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_alpha(0.95)
    legend.get_title().set_color("#000000")
    legend.get_title().set_fontweight('bold')

## ----------------------------------------------------------------
## Main Plotting Execution
## ----------------------------------------------------------------

print("Creating plots...")

# --- Plot 1: U.S. With Nearby ---
setup_us_map(ax_us_with, '(a) U.S. - With Nearby Statistics')
plot_grid_on_map(ax_us_with, us_grid_with, 2.0)
create_enhanced_legend(ax_us_with, us_grid_with, '(With Nearby)', 9, 3, -0.1)

# --- Plot 2: U.S. Without Nearby ---
setup_us_map(ax_us_without, '(b) U.S. - Without Nearby Statistics')
plot_grid_on_map(ax_us_without, us_grid_without, 2.0)
create_enhanced_legend(ax_us_without, us_grid_without, '(Without Nearby)', 9, 3, -0.1)

# --- Plot 3: Global With Nearby ---
setup_world_map(ax_global_with, '(c) Global - With Nearby Statistics')
plot_grid_on_map(ax_global_with, global_grid_with, 4.0)
create_enhanced_legend(ax_global_with, global_grid_with, '(With Nearby)', 9, 3, -0.08)

# --- Plot 4: Global Without Nearby ---
setup_world_map(ax_global_without, '(d) Global - Without Nearby Statistics')
plot_grid_on_map(ax_global_without, global_grid_without, 4.0)
create_enhanced_legend(ax_global_without, global_grid_without, '(Without Nearby)', 9, 3, -0.08)

# Add decorative frames and labels
for ax in fig.get_axes():
    rect = plt.Rectangle((0.01, 0.01), 0.98, 0.98, transform=ax.transAxes,
                         fill=False, edgecolor='#cccccc', linewidth=2, zorder=10, clip_on=False)
    ax.add_patch(rect)

print("Finalizing and saving figure...")
plt.savefig('updated_grid_feature_map.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("\nVisualization complete!")