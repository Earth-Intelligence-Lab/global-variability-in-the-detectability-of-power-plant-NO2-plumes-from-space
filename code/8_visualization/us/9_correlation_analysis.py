"""
Additional correlation analysis between plant characteristics and model performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

def main():
    print("="*60)
    print("CORRELATION ANALYSIS: Plant Characteristics vs Performance")
    print("="*60)
    
    # Load data
    results_dir = '/net/fs06/d3/rzhuang/TROPOMI_US/results/'
    figure_dir = '/net/fs06/d3/rzhuang/TROPOMI_US/figure/'
    
    plant_metrics_df = pd.read_csv(f'{results_dir}per_plant_performance_metrics.csv')
    
    # Select numeric columns for correlation
    numeric_cols = [
        'n_observations', 'Total_NOx_Mass', 'Latitude', 'Longitude',
        'Total_Operating_Time_Sum', 'Total_Gross_Load_MWh',
        'Total_SO2_Mass', 'Total_CO2_Mass', 'Total_Heat_Input_mmBtu',
        'Avg_SO2_Rate', 'Avg_CO2_Rate', 'Avg_NOx_Rate'
    ]
    
    performance_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Filter to available columns
    available_plant_chars = [col for col in numeric_cols if col in plant_metrics_df.columns]
    
    print(f"\nAnalyzing {len(available_plant_chars)} plant characteristics")
    print(f"Against {len(performance_metrics)} performance metrics")
    
    # Calculate correlations
    correlation_results = []
    
    for metric in performance_metrics:
        for char in available_plant_chars:
            # Remove NaN values
            valid_data = plant_metrics_df[[metric, char]].dropna()
            
            if len(valid_data) < 10:
                continue
            
            # Pearson correlation
            pearson_r, pearson_p = pearsonr(valid_data[metric], valid_data[char])
            
            # Spearman correlation (rank-based, robust to outliers)
            spearman_r, spearman_p = spearmanr(valid_data[metric], valid_data[char])
            
            correlation_results.append({
                'performance_metric': metric,
                'plant_characteristic': char,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n_samples': len(valid_data)
            })
    
    corr_df = pd.DataFrame(correlation_results)
    
    # Save correlation results
    corr_df.to_csv(f'{results_dir}correlation_analysis.csv', index=False)
    print(f"\nSaved: correlation_analysis.csv")
    
    # Print significant correlations (p < 0.05, |r| > 0.3)
    print("\n" + "="*60)
    print("SIGNIFICANT CORRELATIONS (p < 0.05, |r| > 0.3)")
    print("="*60)
    
    significant = corr_df[
        (corr_df['pearson_p'] < 0.05) & 
        (abs(corr_df['pearson_r']) > 0.3)
    ].sort_values('pearson_r', key=abs, ascending=False)
    
    print(f"\nFound {len(significant)} significant correlations:\n")
    
    for _, row in significant.iterrows():
        sign = "+" if row['pearson_r'] > 0 else "-"
        print(f"{row['performance_metric']:<12s} vs {row['plant_characteristic']:<30s}: "
              f"{sign} {abs(row['pearson_r']):.3f} (p={row['pearson_p']:.4f})")
    
    # Create correlation heatmap
    print("\n" + "="*60)
    print("Creating correlation heatmap...")
    print("="*60)
    
    # Pivot for heatmap
    pivot_pearson = corr_df.pivot(
        index='plant_characteristic',
        columns='performance_metric',
        values='pearson_r'
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        pivot_pearson,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Pearson Correlation'},
        ax=ax
    )
    
    ax.set_title('Correlation: Plant Characteristics vs Model Performance', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Performance Metric', fontweight='bold')
    ax.set_ylabel('Plant Characteristic', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{figure_dir}correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{figure_dir}correlation_heatmap.pdf', bbox_inches='tight')
    print("Saved: correlation_heatmap.png/pdf")
    plt.close()
    
    # Create scatter plots for top correlations
    print("\nCreating scatter plots for top correlations...")
    
    top_correlations = significant.head(6)
    
    if len(top_correlations) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Top Correlations: Plant Characteristics vs Performance', 
                     fontsize=16, fontweight='bold')
        
        for idx, (_, row) in enumerate(top_correlations.iterrows()):
            if idx >= 6:
                break
            
            ax = axes[idx // 3, idx % 3]
            
            metric = row['performance_metric']
            char = row['plant_characteristic']
            
            valid_data = plant_metrics_df[[metric, char]].dropna()
            
            ax.scatter(valid_data[char], valid_data[metric], 
                      alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            
            # Add trend line
            z = np.polyfit(valid_data[char], valid_data[metric], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(valid_data[char].min(), valid_data[char].max(), 100)
            ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, alpha=0.8)
            
            ax.set_xlabel(char, fontweight='bold')
            ax.set_ylabel(metric, fontweight='bold')
            ax.set_title(f'r = {row["pearson_r"]:.3f}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{figure_dir}top_correlations_scatter.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{figure_dir}top_correlations_scatter.pdf', bbox_inches='tight')
        print("Saved: top_correlations_scatter.png/pdf")
        plt.close()
    
    # Summary statistics by performance quartiles
    print("\n" + "="*60)
    print("PLANT CHARACTERISTICS BY PERFORMANCE QUARTILES (F1 Score)")
    print("="*60)
    
    # Create quartiles based on F1 score
    plant_metrics_df['f1_quartile'] = pd.qcut(
        plant_metrics_df['f1'], 
        q=4, 
        labels=['Q1 (Worst)', 'Q2', 'Q3', 'Q4 (Best)']
    )
    
    for char in ['Total_NOx_Mass', 'n_observations', 'Latitude', 'Total_Gross_Load_MWh']:
        if char not in plant_metrics_df.columns:
            continue
        
        print(f"\n{char}:")
        summary = plant_metrics_df.groupby('f1_quartile')[char].agg(['mean', 'median', 'std', 'count'])
        print(summary.to_string())
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == '__main__':
    main()
