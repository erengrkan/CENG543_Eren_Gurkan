#!/usr/bin/env python3
"""
Clean and Simple Visualization Script

Creates publication-ready, easy-to-understand figures:
1. Model comparison per dataset (simple bar charts)
2. Alpha sensitivity analysis
3. Speed vs Quality tradeoff
4. Best hybrid configurations
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'minilm': '#3498db',
    'splade': '#9b59b6', 
    'bm25': '#e67e22',
    'word2vec': '#1abc9c',
    'bge-m3': '#2ecc71',
    'bge-m3-all': '#e74c3c',
    'hybrid': '#34495e'
}

def load_all_results(results_dir='results'):
    """Load all benchmark JSON files."""
    all_data = []
    for f in glob.glob(os.path.join(results_dir, 'benchmark_*.json')):
        with open(f) as fp:
            all_data.extend(json.load(fp))
    return pd.DataFrame(all_data)


def plot_recall_by_dataset(df, output_dir):
    """Simple bar chart: Recall@10 per model, separate subplot per dataset."""
    datasets = df['dataset'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = df[df['dataset'] == dataset]
        
        # Get best result per model (highest recall)
        best = subset.groupby('model')['recall_at_10'].max().sort_values(ascending=True)
        
        # Color bars
        colors = [COLORS.get(m.split('+')[0], COLORS['hybrid']) for m in best.index]
        
        bars = ax.barh(best.index, best.values, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Recall@10', fontsize=11)
        ax.set_title(f'{dataset.upper()}', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1.0)
        
        # Add value labels
        for bar, val in zip(bars, best.values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=9)
    
    plt.suptitle('Recall@10 Comparison by Dataset', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recall_by_dataset.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ recall_by_dataset.png")


def plot_alpha_sensitivity(df, output_dir):
    """Line chart showing how alpha affects performance per hybrid model."""
    hybrids = df[df['model_type'] == 'hybrid'].copy()
    if hybrids.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    datasets = df['dataset'].unique()
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = hybrids[hybrids['dataset'] == dataset]
        
        # Group by model and alpha
        for model in subset['model'].unique():
            model_data = subset[subset['model'] == model].sort_values('alpha')
            if len(model_data) > 1:
                ax.plot(model_data['alpha'], model_data['recall_at_10'], 
                       'o-', label=model, linewidth=2, markersize=8)
        
        ax.set_xlabel('Alpha (Dense Weight)', fontsize=11)
        ax.set_ylabel('Recall@10', fontsize=11)
        ax.set_title(f'{dataset.upper()}', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim(0, 1)
    
    plt.suptitle('Alpha Sensitivity: How Dense/Sparse Balance Affects Performance', 
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alpha_sensitivity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ alpha_sensitivity.png")


def plot_speed_vs_quality(df, output_dir):
    """Scatter plot: QPS vs Recall@10, color by model type."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Single models
    singles = df[df['model_type'] == 'single']
    hybrids = df[df['model_type'] == 'hybrid']
    
    # Plot singles
    for model in singles['model'].unique():
        model_data = singles[singles['model'] == model]
        color = COLORS.get(model, '#7f8c8d')
        ax.scatter(model_data['qps'], model_data['recall_at_10'], 
                  c=color, s=150, label=model, alpha=0.8, edgecolors='white', linewidth=1)
    
    # Plot best hybrid per dataset
    best_hybrids = hybrids.loc[hybrids.groupby('dataset')['recall_at_10'].idxmax()]
    ax.scatter(best_hybrids['qps'], best_hybrids['recall_at_10'], 
              c=COLORS['hybrid'], s=200, marker='s', label='Best Hybrid', 
              alpha=0.8, edgecolors='white', linewidth=2)
    
    ax.set_xlabel('Queries Per Second (QPS)', fontsize=12)
    ax.set_ylabel('Recall@10', fontsize=12)
    ax.set_title('Speed vs Quality Tradeoff\n(Higher right is better)', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speed_vs_quality.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ speed_vs_quality.png")


def plot_model_comparison_summary(df, output_dir):
    """Single clean summary: Average Recall@10 across all datasets."""
    # Best result per model-dataset combo
    best = df.groupby(['model', 'dataset'])['recall_at_10'].max().reset_index()
    avg_recall = best.groupby('model')['recall_at_10'].mean().sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = [COLORS.get(m.split('+')[0], COLORS['hybrid']) for m in avg_recall.index]
    
    bars = ax.barh(avg_recall.index, avg_recall.values, color=colors, 
                   edgecolor='white', linewidth=0.5, height=0.7)
    
    # Add value labels
    for bar, val in zip(bars, avg_recall.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Average Recall@10 (across all datasets)', fontsize=12)
    ax.set_title('Overall Model Performance Ranking', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_ranking.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ model_ranking.png")


def plot_bge_m3_all_comparison(df, output_dir):
    """Compare BGE-M3-ALL baseline with best hybrid per dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    datasets = df['dataset'].unique()
    
    # Get bge-m3-all and best hybrid per dataset
    bge_all = df[df['model'] == 'bge-m3-all'].groupby('dataset')['recall_at_10'].max()
    
    hybrids = df[df['model_type'] == 'hybrid']
    best_hybrid = hybrids.loc[hybrids.groupby('dataset')['recall_at_10'].idxmax()]
    best_hybrid_recall = best_hybrid.set_index('dataset')['recall_at_10']
    
    # Recall comparison
    ax1 = axes[0]
    x = range(len(datasets))
    width = 0.35
    
    bge_vals = [bge_all.get(d, 0) for d in datasets]
    hybrid_vals = [best_hybrid_recall.get(d, 0) for d in datasets]
    
    bars1 = ax1.bar([i - width/2 for i in x], bge_vals, width, label='BGE-M3-ALL', color=COLORS['bge-m3-all'])
    bars2 = ax1.bar([i + width/2 for i in x], hybrid_vals, width, label='Best Hybrid', color=COLORS['hybrid'])
    
    ax1.set_ylabel('Recall@10', fontsize=12)
    ax1.set_title('Recall: BGE-M3-ALL vs Best Hybrid', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=30, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # QPS comparison
    ax2 = axes[1]
    bge_qps = df[df['model'] == 'bge-m3-all'].groupby('dataset')['qps'].mean()
    best_hybrid_qps = best_hybrid.set_index('dataset')['qps']
    
    bge_qps_vals = [bge_qps.get(d, 0) for d in datasets]
    hybrid_qps_vals = [best_hybrid_qps.get(d, 0) for d in datasets]
    
    bars1 = ax2.bar([i - width/2 for i in x], bge_qps_vals, width, label='BGE-M3-ALL', color=COLORS['bge-m3-all'])
    bars2 = ax2.bar([i + width/2 for i in x], hybrid_qps_vals, width, label='Best Hybrid', color=COLORS['hybrid'])
    
    ax2.set_ylabel('Queries Per Second', fontsize=12)
    ax2.set_title('Speed: BGE-M3-ALL vs Best Hybrid', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=30, ha='right')
    ax2.legend()
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bge_m3_all_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ bge_m3_all_comparison.png")


def generate_clean_summary(df, output_dir):
    """Generate a clean markdown summary."""
    summary_path = os.path.join(output_dir, 'final_summary.md')
    
    with open(summary_path, 'w') as f:
        f.write("# Vector Retrieval Benchmark Results\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        # Overview
        f.write("## Overview\n\n")
        f.write(f"- **Datasets:** {', '.join(df['dataset'].unique())}\n")
        f.write(f"- **Models tested:** {len(df['model'].unique())}\n")
        f.write(f"- **Total experiments:** {len(df)}\n\n")
        
        # Best models table
        f.write("## Top Performing Models by Dataset\n\n")
        f.write("| Dataset | Best Model | Recall@10 | QPS |\n")
        f.write("|---------|------------|-----------|-----|\n")
        
        for dataset in df['dataset'].unique():
            subset = df[df['dataset'] == dataset]
            best_row = subset.loc[subset['recall_at_10'].idxmax()]
            f.write(f"| {dataset} | {best_row['model']} | {best_row['recall_at_10']:.4f} | {best_row['qps']:.1f} |\n")
        
        f.write("\n## Key Insights\n\n")
        
        # Hybrid vs Single comparison
        singles = df[df['model_type'] == 'single']['recall_at_10'].mean()
        hybrids = df[df['model_type'] == 'hybrid']['recall_at_10'].mean()
        improvement = (hybrids - singles) / singles * 100
        
        f.write(f"1. **Hybrid models improve recall by {improvement:.1f}%** over single models on average\n")
        
        # BGE-M3-ALL comparison
        bge_all = df[df['model'] == 'bge-m3-all']
        if not bge_all.empty:
            bge_recall = bge_all['recall_at_10'].mean()
            bge_qps = bge_all['qps'].mean()
            f.write(f"2. **BGE-M3-ALL baseline:** Average R@10={bge_recall:.4f}, QPS={bge_qps:.1f}\n")
        
        # Best overall
        best = df.loc[df['recall_at_10'].idxmax()]
        f.write(f"3. **Best overall:** {best['model']} on {best['dataset']} with R@10={best['recall_at_10']:.4f}\n")
        
    print(f"✓ {summary_path}")


def main():
    results_dir = 'results'
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    print("Loading results...")
    df = load_all_results(results_dir)
    print(f"Loaded {len(df)} experiments\n")
    
    print("Generating figures...")
    plot_recall_by_dataset(df, figures_dir)
    plot_alpha_sensitivity(df, figures_dir)
    plot_speed_vs_quality(df, figures_dir)
    plot_model_comparison_summary(df, figures_dir)
    plot_bge_m3_all_comparison(df, figures_dir)
    
    print("\nGenerating summary...")
    generate_clean_summary(df, results_dir)
    
    print("\n" + "="*50)
    print("✓ All figures generated!")
    print("="*50)


if __name__ == "__main__":
    main()
