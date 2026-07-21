"""
Create visualizations for model performance analysis.

This script generates plots showing:
1. Per-plant performance distribution
2. Feature importance rankings
3. Correlations between features and performance
4. Geographic patterns in prediction accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def plot_plant_performance_distribution(plant_metrics_df, output_dir):
    """Plot distribution of performance metrics across plants."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Distribution of Model Performance Across Power Plants', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Histogram
        ax.hist(plant_metrics_df[metric], bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(plant_metrics_df[metric].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {plant_metrics_df[metric].mean():.3f}')
        ax.axvline(plant_metrics_df[metric].median(), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {plant_metrics_df[metric].median():.3f}')
        
        ax.set_xlabel(title)
        ax.set_ylabel('Number of Plants')
        ax.set_title(f'{title} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Number of observations distribution
    ax = axes[1, 2]
    ax.hist(plant_metrics_df['n_observations'], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Observations')
    ax.set_ylabel('Number of Plants')
    ax.set_title('Observations per Plant')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}plant_performance_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}plant_performance_distribution.pdf', bbox_inches='tight')
    print(f"Saved: plant_performance_distribution.png/pdf")
    plt.close()

def plot_feature_importance(gradient_importance, output_dir):
    """Plot feature importance from gradient analysis."""
    # Sort features by importance
    sorted_features = sorted(gradient_importance.items(), key=lambda x: x[1], reverse=True)
    features, importances = zip(*sorted_features[:20])  # Top 20
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create horizontal bar plot
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, alpha=0.8, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance (Gradient Magnitude)', fontweight='bold')
    ax.set_title('Top 20 Most Important Features\n(Gradient-Based Analysis)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}feature_importance_gradient.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}feature_importance_gradient.pdf', bbox_inches='tight')
    print(f"Saved: feature_importance_gradient.png/pdf")
    plt.close()

def plot_feature_comparison(feature_comparison, output_dir):
    """Plot feature differences between high and low performers."""
    # Sort by absolute relative difference
    sorted_features = sorted(
        feature_comparison.items(), 
        key=lambda x: abs(x[1]['relative_difference']), 
        reverse=True
    )[:15]  # Top 15
    
    features = [f[0] for f in sorted_features]
    rel_diffs = [f[1]['relative_difference'] for f in sorted_features]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['green' if x > 0 else 'red' for x in rel_diffs]
    y_pos = np.arange(len(features))
    
    ax.barh(y_pos, rel_diffs, alpha=0.8, color=colors, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Relative Difference (%)', fontweight='bold')
    ax.set_title('Feature Differences: High vs Low Performing Plants\n(Green = Higher in High Performers)', 
                 fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}feature_comparison_high_vs_low.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}feature_comparison_high_vs_low.pdf', bbox_inches='tight')
    print(f"Saved: feature_comparison_high_vs_low.png/pdf")
    plt.close()

def plot_performance_vs_observations(plant_metrics_df, output_dir):
    """Plot relationship between number of observations and performance."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance vs Number of Observations', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'f1', 'auc', 'precision']
    titles = ['Accuracy', 'F1 Score', 'AUC', 'Precision']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Scatter plot with log scale for observations
        ax.scatter(plant_metrics_df['n_observations'], 
                  plant_metrics_df[metric],
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Number of Observations (log scale)', fontweight='bold')
        ax.set_ylabel(title, fontweight='bold')
        ax.set_title(f'{title} vs Observations')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = plant_metrics_df[['n_observations', metric]].corr().iloc[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}performance_vs_observations.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}performance_vs_observations.pdf', bbox_inches='tight')
    print(f"Saved: performance_vs_observations.png/pdf")
    plt.close()

def plot_performance_vs_emissions(plant_metrics_df, output_dir):
    """Plot relationship between NOx emissions and performance."""
    # Check which NOx column is available
    nox_col = None
    for col in ['NOx Mass (lbs)', 'Total_NOx_Mass', 'NOx_Mass']:
        if col in plant_metrics_df.columns:
            nox_col = col
            break
    
    if nox_col is None:
        print("No NOx emission data found for plotting")
        return
    
    # Filter out missing emission data
    df_with_emissions = plant_metrics_df[plant_metrics_df[nox_col].notna()].copy()
    
    if len(df_with_emissions) < 10:
        print("Not enough emission data for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance vs NOx Emissions', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'f1', 'auc', 'recall']
    titles = ['Accuracy', 'F1 Score', 'AUC', 'Recall']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Scatter plot with log scale for emissions
        scatter = ax.scatter(df_with_emissions[nox_col], 
                           df_with_emissions[metric],
                           c=df_with_emissions['n_observations'],
                           cmap='viridis',
                           alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('NOx Emissions (log scale)', fontweight='bold')
        ax.set_ylabel(title, fontweight='bold')
        ax.set_title(f'{title} vs Emissions')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('# Observations', rotation=270, labelpad=20)
        
        # Add correlation coefficient
        corr = df_with_emissions[[nox_col, metric]].corr().iloc[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}performance_vs_emissions.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}performance_vs_emissions.pdf', bbox_inches='tight')
    print(f"Saved: performance_vs_emissions.png/pdf")
    plt.close()

def plot_geographic_performance(plant_metrics_df, output_dir):
    """Plot geographic distribution of model performance."""
    # Filter out missing location data
    df_with_location = plant_metrics_df[
        plant_metrics_df['Latitude'].notna() & plant_metrics_df['Longitude'].notna()
    ].copy()
    
    if len(df_with_location) < 10:
        print("Not enough location data for plotting")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Geographic Distribution of Model Performance', fontsize=16, fontweight='bold')
    
    # F1 Score map
    ax1 = axes[0]
    scatter1 = ax1.scatter(df_with_location['Longitude'], 
                          df_with_location['Latitude'],
                          c=df_with_location['f1'],
                          s=100, cmap='RdYlGn', vmin=0, vmax=1,
                          edgecolors='black', linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Longitude', fontweight='bold')
    ax1.set_ylabel('Latitude', fontweight='bold')
    ax1.set_title('F1 Score by Location')
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('F1 Score', rotation=270, labelpad=20)
    
    # AUC map
    ax2 = axes[1]
    scatter2 = ax2.scatter(df_with_location['Longitude'], 
                          df_with_location['Latitude'],
                          c=df_with_location['auc'],
                          s=100, cmap='RdYlGn', vmin=0.5, vmax=1,
                          edgecolors='black', linewidth=0.5, alpha=0.7)
    ax2.set_xlabel('Longitude', fontweight='bold')
    ax2.set_ylabel('Latitude', fontweight='bold')
    ax2.set_title('AUC by Location')
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('AUC', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}geographic_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}geographic_performance.pdf', bbox_inches='tight')
    print(f"Saved: geographic_performance.png/pdf")
    plt.close()

def main():
    print("="*60)
    print("VISUALIZATION GENERATION")
    print("="*60)
    
    # Paths
    results_dir = '/net/fs06/d3/rzhuang/TROPOMI_US/results/'
    figure_dir = '/net/fs06/d3/rzhuang/TROPOMI_US/figure/'
    
    # Create figure directory if needed
    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading analysis results...")
    plant_metrics_df = pd.read_csv(f'{results_dir}per_plant_performance_metrics.csv')
    
    with open(f'{results_dir}feature_importance_gradient.json', 'r') as f:
        gradient_importance = json.load(f)
    
    with open(f'{results_dir}feature_comparison_high_vs_low_performers.json', 'r') as f:
        feature_comparison = json.load(f)
    
    print(f"Loaded data for {len(plant_metrics_df)} plants")
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    print("\n1. Plant performance distribution...")
    plot_plant_performance_distribution(plant_metrics_df, figure_dir)
    
    print("\n2. Feature importance...")
    plot_feature_importance(gradient_importance, figure_dir)
    
    print("\n3. Feature comparison...")
    plot_feature_comparison(feature_comparison, figure_dir)
    
    print("\n4. Performance vs observations...")
    plot_performance_vs_observations(plant_metrics_df, figure_dir)
    
    print("\n5. Performance vs emissions...")
    plot_performance_vs_emissions(plant_metrics_df, figure_dir)
    
    print("\n6. Geographic performance...")
    plot_geographic_performance(plant_metrics_df, figure_dir)
    
    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print(f"Check: {figure_dir}")
    print("="*60)

if __name__ == '__main__':
    main()
