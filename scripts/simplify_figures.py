#!/usr/bin/env python3
"""
Beautiful Speed vs Quality figures - Clean, informative, and visually appealing
"""

import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Modern color palette
COLORS = {
    'minilm': '#3498db',
    'splade': '#9b59b6', 
    'bm25': '#e67e22',
    'word2vec': '#1abc9c',
    'bge-m3': '#27ae60',
    'bge-m3-all': '#e74c3c',
    'minilm+splade': '#2c3e50',
    'minilm+bm25': '#34495e',
}

def load_all_results(results_dir='results'):
    all_data = []
    for f in glob.glob(os.path.join(results_dir, 'benchmark_*.json')):
        with open(f) as fp:
            all_data.extend(json.load(fp))
    return pd.DataFrame(all_data)


def plot_speed_vs_quality_beautiful(df, output_dir):
    """Beautiful scatter plot with gradient background and clear categories."""
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Set background gradient effect with regions
    ax.set_facecolor('#f8f9fa')
    
    # Add colored regions for interpretation
    ax.axhspan(0.8, 1.0, alpha=0.1, color='green', label='_High Quality Zone')
    ax.axvspan(100, 10000, alpha=0.08, color='blue', label='_High Speed Zone')
    
    # Get best result per model
    model_stats = df.groupby('model').agg({
        'recall_at_10': 'max',
        'qps': 'mean'
    }).reset_index()
    
    # Define model categories
    single_dense = ['minilm', 'word2vec', 'bge-m3', 'bge-m3-all']
    single_sparse = ['splade', 'bm25']
    
    # Plot each category separately
    for _, row in model_stats.iterrows():
        model = row['model']
        x, y = row['qps'], row['recall_at_10']
        
        if model in single_dense:
            color = COLORS.get(model, '#3498db')
            marker = 'o'
            size = 300
            edge = 'white'
        elif model in single_sparse:
            color = COLORS.get(model, '#9b59b6')
            marker = '^'
            size = 280
            edge = 'white'
        else:  # Hybrid
            color = COLORS.get(model, '#34495e')
            marker = 's'
            size = 250
            edge = 'white'
        
        ax.scatter(x, y, c=color, s=size, marker=marker, alpha=0.85,
                  edgecolors=edge, linewidth=2, zorder=10)
    
    # Add labels for key models only
    key_models = ['minilm', 'splade', 'bge-m3', 'bge-m3-all', 'minilm+splade', 'minilm+bm25']
    for _, row in model_stats.iterrows():
        model = row['model']
        if model in key_models:
            x, y = row['qps'], row['recall_at_10']
            
            # Smart label positioning
            if model == 'bge-m3-all':
                offset = (-80, 10)
            elif model == 'minilm+splade':
                offset = (15, -15)
            elif 'splade' in model:
                offset = (10, 10)
            else:
                offset = (12, 5)
            
            ax.annotate(model, (x, y), xytext=offset, textcoords='offset points',
                       fontsize=11, fontweight='bold', color='#2c3e50',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                alpha=0.8, edgecolor='none'))
    
    # Styling
    ax.set_xlabel('Queries Per Second (QPS)', fontsize=13, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('Recall@10', fontsize=13, fontweight='bold', color='#2c3e50')
    ax.set_title('Speed vs Quality Tradeoff\n', fontsize=16, fontweight='bold', color='#2c3e50')
    ax.set_xscale('log')
    ax.set_ylim(0.3, 1.0)
    ax.set_xlim(1, 10000)
    
    # Grid
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax.tick_params(labelsize=11)
    
    # Legend with categories
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', 
               markersize=14, label='Dense (MiniLM, BGE-M3)', markeredgecolor='white', markeredgewidth=2),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#9b59b6', 
               markersize=14, label='Sparse (SPLADE, BM25)', markeredgecolor='white', markeredgewidth=2),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#34495e', 
               markersize=14, label='Hybrid (Dense + Sparse)', markeredgecolor='white', markeredgewidth=2),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11, 
             framealpha=0.95, edgecolor='#bdc3c7')
    
    # Add annotation arrows for best models
    ax.annotate('← Faster', xy=(5000, 0.35), fontsize=10, color='#7f8c8d', ha='center')
    ax.annotate('Better ↑', xy=(2, 0.95), fontsize=10, color='#7f8c8d', rotation=90, va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speed_vs_quality.png'), dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    print("✓ speed_vs_quality.png")


def plot_latency_vs_recall_beautiful(df, output_dir):
    """Beautiful latency vs recall plot."""
    
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_facecolor('#f8f9fa')
    
    # Add colored regions
    ax.axhspan(0.8, 1.0, alpha=0.1, color='green')
    ax.axvspan(0.1, 10, alpha=0.08, color='blue')
    
    # Get best result per model
    model_stats = df.groupby('model').agg({
        'recall_at_10': 'max',
        'latency_p99_ms': 'mean'
    }).reset_index()
    
    # Filter low recall
    model_stats = model_stats[model_stats['recall_at_10'] > 0.25]
    
    single_dense = ['minilm', 'word2vec', 'bge-m3', 'bge-m3-all']
    single_sparse = ['splade', 'bm25']
    
    for _, row in model_stats.iterrows():
        model = row['model']
        x, y = row['latency_p99_ms'], row['recall_at_10']
        
        if model in single_dense:
            color = COLORS.get(model, '#3498db')
            marker = 'o'
            size = 300
        elif model in single_sparse:
            color = COLORS.get(model, '#9b59b6')
            marker = '^'
            size = 280
        else:
            color = COLORS.get(model, '#34495e')
            marker = 's'
            size = 250
        
        ax.scatter(x, y, c=color, s=size, marker=marker, alpha=0.85,
                  edgecolors='white', linewidth=2, zorder=10)
    
    # Labels for key models
    key_models = ['minilm', 'splade', 'bge-m3', 'bge-m3-all', 'minilm+splade', 'minilm+bm25']
    for _, row in model_stats.iterrows():
        model = row['model']
        if model in key_models:
            x, y = row['latency_p99_ms'], row['recall_at_10']
            
            if model == 'bge-m3-all':
                offset = (-80, 10)
            elif 'splade' in model:
                offset = (10, 10)
            else:
                offset = (12, 5)
            
            ax.annotate(model, (x, y), xytext=offset, textcoords='offset points',
                       fontsize=11, fontweight='bold', color='#2c3e50',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                alpha=0.8, edgecolor='none'))
    
    ax.set_xlabel('Latency P99 (ms)', fontsize=13, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('Recall@10', fontsize=13, fontweight='bold', color='#2c3e50')
    ax.set_title('Latency vs Recall Tradeoff\n', fontsize=16, fontweight='bold', color='#2c3e50')
    ax.set_xscale('log')
    ax.set_ylim(0.3, 1.0)
    
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax.tick_params(labelsize=11)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', 
               markersize=14, label='Dense (MiniLM, BGE-M3)', markeredgecolor='white', markeredgewidth=2),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#9b59b6', 
               markersize=14, label='Sparse (SPLADE, BM25)', markeredgecolor='white', markeredgewidth=2),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#34495e', 
               markersize=14, label='Hybrid (Dense + Sparse)', markeredgecolor='white', markeredgewidth=2),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11,
             framealpha=0.95, edgecolor='#bdc3c7')
    
    ax.annotate('← Lower Latency', xy=(0.5, 0.35), fontsize=10, color='#7f8c8d', ha='center')
    ax.annotate('Better ↑', xy=(0.3, 0.95), fontsize=10, color='#7f8c8d', rotation=90, va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_vs_recall.png'), dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    print("✓ latency_vs_recall.png")


def main():
    results_dir = 'results'
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    print("Loading results...")
    df = load_all_results(results_dir)
    print(f"Loaded {len(df)} experiments\n")
    
    print("Generating beautiful figures...")
    plot_speed_vs_quality_beautiful(df, figures_dir)
    plot_latency_vs_recall_beautiful(df, figures_dir)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
