"""
Map visualization showing F1 scores from ML models for US and Global power plants.
Replaces probability of detection with model F1 scores.
"""

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.colors as mcolors
import numpy as np

# Styling
plt.rcdefaults()
plt.rcParams['font.family'] = 'Nimbus Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['figure.facecolor'] = 'white'

print("="*60)
print("LOADING MODEL PERFORMANCE DATA")
print("="*60)

# Load US performance metrics
print("\nLoading US model performance...")
us_performance = pd.read_csv('/net/fs06/d3/rzhuang/TROPOMI_US/results/per_plant_performance_metrics.csv')
print(f"  US plants with performance metrics: {len(us_performance)}")

# Load Global performance metrics
print("\nLoading Global model performance...")
global_performance = pd.read_csv('/net/fs06/d3/rzhuang/TROPOMI_world/results/per_plant_performance_metrics.csv')
print(f"  Global plants with performance metrics: {len(global_performance)}")

# US: Extract key columns
us_gdf = gpd.GeoDataFrame(
    us_performance,
    geometry=gpd.points_from_xy(us_performance["Longitude"], us_performance["Latitude"]),
    crs="EPSG:4326"
)

# Scale US NOx emissions for marker sizes
short_ton_to_metric_ton = 0.907185
us_nox_scaling_factor = 0.01
if 'Total_NOx_Mass' in us_gdf.columns:
    us_gdf["NOx_mass_metric_tons"] = us_gdf["Total_NOx_Mass"] * short_ton_to_metric_ton
else:
    us_gdf["NOx_mass_metric_tons"] = 0.0
us_gdf["NOx_mass_scaled"] = us_gdf["NOx_mass_metric_tons"] * us_nox_scaling_factor / 6

# Global: Extract key columns
global_gdf = gpd.GeoDataFrame(
    global_performance,
    geometry=gpd.points_from_xy(global_performance["longitude"], global_performance["latitude"]),
    crs="EPSG:4326"
)

# Scale Global NOx emissions
global_nox_scaling_factor = 0.01
if 'nox_emis_ty' in global_gdf.columns:
    global_gdf["NOx_mass_scaled"] = global_gdf["nox_emis_ty"] * global_nox_scaling_factor
else:
    global_gdf["NOx_mass_scaled"] = 0.0

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nUS Plants:")
print(f"  Count: {len(us_gdf)}")
print(f"  F1 Score - Mean: {us_gdf['f1'].mean():.3f}, Median: {us_gdf['f1'].median():.3f}")
print(f"  F1 Score - Range: [{us_gdf['f1'].min():.3f}, {us_gdf['f1'].max():.3f}]")
print(f"  AUC - Mean: {us_gdf['auc'].mean():.3f}, Median: {us_gdf['auc'].median():.3f}")

print(f"\nGlobal Plants:")
print(f"  Count: {len(global_gdf)}")
print(f"  F1 Score - Mean: {global_gdf['f1'].mean():.3f}, Median: {global_gdf['f1'].median():.3f}")
print(f"  F1 Score - Range: [{global_gdf['f1'].min():.3f}, {global_gdf['f1'].max():.3f}]")
print(f"  AUC - Mean: {global_gdf['auc'].mean():.3f}, Median: {global_gdf['auc'].median():.3f}")

# =========================
# Plot Setup
# =========================
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# Normalization for colormaps
us_norm_f1 = mcolors.Normalize(vmin=0, vmax=1)
us_norm_auc = mcolors.Normalize(vmin=0, vmax=1)
global_norm_f1 = mcolors.Normalize(vmin=0, vmax=1)
global_norm_auc = mcolors.Normalize(vmin=0, vmax=1)

fig = plt.figure(figsize=(20, 24))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.5, 1.5], hspace=0.3, wspace=0.1,
                      top=0.94, bottom=0.08, left=0.05, right=0.82)

fig.suptitle('Power Plant ML Model Performance: F1 Score & AUC from MLP Models',
             fontsize=32, y=0.955, weight='bold', color="#000000")

def style_us_map(ax, title):
    ax.set_extent([-130, -65, 24, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#f8f8f8', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff')
    ax.add_feature(cfeature.STATES, linewidth=0.8, edgecolor='#666666', alpha=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=1.2, edgecolor='#333333')
    ax.coastlines(linewidth=1.0, color='#444444')
    gl = ax.gridlines(draw_labels=True, linestyle=':', alpha=0.4, linewidth=0.8, color='#888888')
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {'size': 18, 'color': '#444444'}
    gl.ylabel_style = {'size': 18, 'color': '#444444'}
    ax.set_title(title, fontsize=28, fontweight='bold', pad=20)

def style_global_map(ax, title):
    ax.set_extent([-180, 180, -60, 80], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#f8f8f8', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff')
    ax.add_feature(cfeature.BORDERS, linewidth=1.2, edgecolor='#333333')
    ax.coastlines(linewidth=1.0, color='#444444')
    gl = ax.gridlines(draw_labels=True, linestyle=':', alpha=0.4, linewidth=0.8, color='#888888')
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {'size': 18, 'color': '#444444'}
    gl.ylabel_style = {'size': 18, 'color': '#444444'}
    ax.set_title(title, fontsize=28, fontweight='bold', pad=20)

# Row 1: U.S.
print("\n1. Creating US maps...")
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
style_us_map(ax1, "(a) U.S. - Model F1 Score")
sc1 = ax1.scatter(
    us_gdf["Longitude"], us_gdf["Latitude"],
    s=us_gdf["NOx_mass_scaled"],
    c=us_gdf['f1'],
    cmap='RdYlGn', norm=us_norm_f1, alpha=0.9, marker='o', 
    edgecolors='black', linewidths=0.5,
    transform=ccrs.PlateCarree()
)

ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
style_us_map(ax2, "(b) U.S. - Model AUC")
sc2 = ax2.scatter(
    us_gdf["Longitude"], us_gdf["Latitude"],
    s=us_gdf["NOx_mass_scaled"],
    c=us_gdf['auc'],
    cmap='RdYlGn', norm=us_norm_auc, alpha=0.9, marker='o',
    edgecolors='black', linewidths=0.5,
    transform=ccrs.PlateCarree()
)

# Row 2: Global F1
print("2. Creating Global F1 map...")
ax3 = fig.add_subplot(gs[1, :], projection=ccrs.PlateCarree())
style_global_map(ax3, "(c) Global - Model F1 Score")
sc3 = ax3.scatter(
    global_gdf["longitude"], global_gdf["latitude"],
    s=global_gdf["NOx_mass_scaled"],
    c=global_gdf['f1'],
    cmap='RdYlGn', norm=global_norm_f1, alpha=0.9, marker='o',
    edgecolors='black', linewidths=0.3,
    transform=ccrs.PlateCarree()
)

# Row 3: Global AUC
print("3. Creating Global AUC map...")
ax4 = fig.add_subplot(gs[2, :], projection=ccrs.PlateCarree())
style_global_map(ax4, "(d) Global - Model AUC")
sc4 = ax4.scatter(
    global_gdf["longitude"], global_gdf["latitude"],
    s=global_gdf["NOx_mass_scaled"],
    c=global_gdf['auc'],
    cmap='RdYlGn', norm=global_norm_auc, alpha=0.9, marker='o',
    edgecolors='black', linewidths=0.3,
    transform=ccrs.PlateCarree()
)

# Colorbars
print("4. Adding colorbars and legends...")
cbar_ax1 = fig.add_axes([0.075, 0.74, 0.32, 0.015])
cbar1 = fig.colorbar(sc1, cax=cbar_ax1, orientation='horizontal')
cbar1.set_label('U.S. Model F1 Score (0 = Poor, 1 = Perfect)', fontsize=14, fontweight='bold')

cbar_ax2 = fig.add_axes([0.475, 0.74, 0.32, 0.015])
cbar2 = fig.colorbar(sc2, cax=cbar_ax2, orientation='horizontal')
cbar2.set_label('U.S. Model AUC (0 = Poor, 1 = Perfect)', fontsize=14, fontweight='bold')

cbar_ax3 = fig.add_axes([0.84, 0.43, 0.03, 0.25])
cbar3 = fig.colorbar(sc3, cax=cbar_ax3, orientation='vertical')
cbar3.set_label('Global Model F1 Score', fontsize=14, fontweight='bold')

cbar_ax4 = fig.add_axes([0.84, 0.09, 0.03, 0.25])
cbar4 = fig.colorbar(sc4, cax=cbar_ax4, orientation='vertical')
cbar4.set_label('Global Model AUC', fontsize=14, fontweight='bold')

# Legends for marker sizes
us_sizes = [50, 200, 500, 1000]
us_labels = [f'{int(s/0.008):,} t' for s in us_sizes]
us_handles = [plt.scatter([], [], s=s, c='gray', alpha=0.7, edgecolors='black', linewidths=0.5) for s in us_sizes]
us_legend = fig.legend(
    us_handles, us_labels, borderpad=0.8,
    title='U.S. NOx emissions\n(metric t/year)', ncol=1, labelspacing=1.5,
    loc='upper center', bbox_to_anchor=(0.875, 0.885),
    fontsize=12, title_fontsize=14, frameon=True, fancybox=True, shadow=True,
    handletextpad=1.2, columnspacing=2
)

glb_sizes = [50, 200, 500, 1000]
glb_labels = [f'{int(s/0.005):,} t' for s in glb_sizes]
glb_handles = [plt.scatter([], [], s=s, c='gray', alpha=0.7, edgecolors='black', linewidths=0.3) for s in glb_sizes]
global_legend = fig.legend(
    glb_handles, glb_labels, borderpad=0.8,
    title='Global NOx emissions (metric t/year)', labelspacing=1.2, ncol=len(glb_handles),
    loc='upper center', bbox_to_anchor=(0.435, 0.405),
    fontsize=12, title_fontsize=14, frameon=True, fancybox=True, shadow=True,
    handletextpad=1.2, columnspacing=2
)
for legend in [us_legend, global_legend]:
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

# Save
out_path_pdf = '/net/fs06/d3/rzhuang/TROPOMI_US/figure/Model_F1_AUC_Geographic_Map.pdf'
out_path_png = '/net/fs06/d3/rzhuang/TROPOMI_US/figure/Model_F1_AUC_Geographic_Map.png'

print("\n5. Saving figures...")
plt.savefig(out_path_pdf, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(out_path_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

print("\n" + "="*60)
print("VISUALIZATION COMPLETE!")
print("="*60)
print(f"\nOutput saved to:")
print(f"  {out_path_pdf}")
print(f"  {out_path_png}")
print("\n" + "="*60)

# Show the plot
plt.show()
