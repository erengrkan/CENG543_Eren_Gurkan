"""
Results Analysis and Visualization Script

This script generates publication-ready plots and tables from benchmark results.

Outputs:
- figures/recall_comparison.png: Bar chart comparing Recall@10 across models
- figures/latency_vs_recall.png: Scatter plot of latency vs recall tradeoff
- figures/qps_comparison.png: Bar chart comparing QPS
- tables/main_results.tex: LaTeX table for paper
- tables/main_results.md: Markdown table for README

Usage:
    python -m src.vector_experiments.analyze_results --input results/benchmark_*.json
"""

import os
import json
import argparse
from typing import List, Dict
from datetime import datetime
import pandas as pd

# Try to import matplotlib, skip plotting if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, skipping plots")


def load_results(result_files: List[str]) -> pd.DataFrame:
    """Load benchmark results from JSON files into DataFrame."""
    all_results = []
    for f in result_files:
        with open(f, 'r') as fp:
            results = json.load(fp)
            all_results.extend(results)
    return pd.DataFrame(all_results)


def generate_main_table(df: pd.DataFrame, output_dir: str):
    """Generate main results table in LaTeX and Markdown."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Pivot for better readability
    # Group by dataset and model
    summary = df.groupby(['dataset', 'model']).agg({
        'recall_at_10': 'mean',
        'ndcg_at_10': 'mean',
        'latency_p99_ms': 'mean',
        'qps': 'mean'
    }).round(4)
    
    # LaTeX table
    latex_path = os.path.join(output_dir, "main_results.tex")
    latex_content = summary.to_latex(
        caption="Retrieval Performance Comparison",
        label="tab:main_results",
        float_format="%.4f"
    )
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"LaTeX table saved to: {latex_path}")
    
    # Markdown table
    md_path = os.path.join(output_dir, "main_results.md")
    md_content = summary.to_markdown()
    with open(md_path, 'w') as f:
        f.write("# Benchmark Results\n\n")
        f.write(md_content)
    print(f"Markdown table saved to: {md_path}")
    
    return summary


def generate_recall_comparison_plot(df: pd.DataFrame, output_dir: str):
    """Generate bar chart comparing Recall@10 across models."""
    if not HAS_MATPLOTLIB:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter to single models + best hybrid per dataset
    fig, axes = plt.subplots(1, len(df['dataset'].unique()), figsize=(16, 5))
    
    if len(df['dataset'].unique()) == 1:
        axes = [axes]
    
    for ax, dataset in zip(axes, df['dataset'].unique()):
        subset = df[df['dataset'] == dataset]
        
        # Get single models
        singles = subset[subset['model_type'] == 'single']
        
        # Get best hybrid (highest recall)
        hybrids = subset[subset['model_type'] == 'hybrid']
        if not hybrids.empty:
            best_hybrid = hybrids.loc[hybrids['recall_at_10'].idxmax()]
            # Add to singles for plotting
            singles = pd.concat([singles, best_hybrid.to_frame().T])
        
        # Sort by recall
        singles = singles.sort_values('recall_at_10', ascending=True)
        
        colors = ['#2ecc71' if 'bge-m3-all' in m else '#3498db' if '+' in m else '#95a5a6' 
                  for m in singles['model']]
        
        ax.barh(singles['model'], singles['recall_at_10'], color=colors)
        ax.set_xlabel('Recall@10')
        ax.set_title(f'{dataset}')
        ax.set_xlim(0, 0.6)
        
    plt.suptitle('Recall@10 Comparison Across Datasets', fontsize=14)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "recall_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")


def generate_latency_vs_recall_plot(df: pd.DataFrame, output_dir: str):
    """Generate scatter plot showing latency vs recall tradeoff."""
    if not HAS_MATPLOTLIB:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by model type
    colors = {'single': '#3498db', 'hybrid': '#e74c3c'}
    markers = {'single': 'o', 'hybrid': 's'}
    
    for model_type in ['single', 'hybrid']:
        subset = df[df['model_type'] == model_type]
        ax.scatter(
            subset['latency_p99_ms'], 
            subset['recall_at_10'],
            c=colors[model_type],
            marker=markers[model_type],
            label=model_type.capitalize(),
            s=100,
            alpha=0.7
        )
        
        # Annotate points
        for _, row in subset.iterrows():
            ax.annotate(
                row['model'][:10],  # Truncate long names
                (row['latency_p99_ms'], row['recall_at_10']),
                fontsize=8,
                alpha=0.7
            )
    
    ax.set_xlabel('Latency P99 (ms)', fontsize=12)
    ax.set_ylabel('Recall@10', fontsize=12)
    ax.set_title('Latency vs Recall Tradeoff', fontsize=14)
    ax.legend()
    ax.set_xscale('log')  # Log scale for latency
    ax.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, "latency_vs_recall.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")


def generate_qps_comparison_plot(df: pd.DataFrame, output_dir: str):
    """Generate bar chart comparing QPS (throughput)."""
    if not HAS_MATPLOTLIB:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Average across datasets
    avg_qps = df.groupby('model')['qps'].mean().sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#e74c3c' if 'bge-m3-all' in m else '#2ecc71' if '+' in m else '#3498db' 
              for m in avg_qps.index]
    
    ax.barh(avg_qps.index, avg_qps.values, color=colors)
    ax.set_xlabel('QPS (Queries Per Second)', fontsize=12)
    ax.set_title('Throughput Comparison', fontsize=14)
    ax.set_xscale('log')
    
    plot_path = os.path.join(output_dir, "qps_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")


def generate_summary_report(df: pd.DataFrame, output_dir: str):
    """Generate a summary text report."""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, "summary_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("BENCHMARK SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Datasets: {', '.join(df['dataset'].unique())}\n")
        f.write(f"Models: {', '.join(df['model'].unique())}\n\n")
        
        # Best performers
        f.write("TOP PERFORMERS (by Recall@10):\n")
        f.write("-" * 40 + "\n")
        top = df.nlargest(5, 'recall_at_10')[['dataset', 'model', 'recall_at_10', 'qps']]
        f.write(top.to_string(index=False))
        f.write("\n\n")
        
        # BGE-M3-ALL comparison
        bge_all = df[df['model'] == 'bge-m3-all']
        if not bge_all.empty:
            f.write("BGE-M3-ALL BASELINE:\n")
            f.write("-" * 40 + "\n")
            for _, row in bge_all.iterrows():
                f.write(f"  {row['dataset']}: R@10={row['recall_at_10']:.4f}, QPS={row['qps']:.1f}\n")
            f.write("\n")
        
        # Best hybrid comparison
        hybrids = df[df['model_type'] == 'hybrid']
        if not hybrids.empty:
            f.write("BEST HYBRID PER DATASET:\n")
            f.write("-" * 40 + "\n")
            for dataset in df['dataset'].unique():
                ds_hybrids = hybrids[hybrids['dataset'] == dataset]
                if not ds_hybrids.empty:
                    best = ds_hybrids.loc[ds_hybrids['recall_at_10'].idxmax()]
                    f.write(f"  {dataset}: {best['model']} (α={best['alpha']}) - R@10={best['recall_at_10']:.4f}, QPS={best['qps']:.1f}\n")
        
    print(f"Summary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--input", type=str, nargs="+", required=True,
                        help="Path(s) to benchmark JSON files")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for figures and tables")
    args = parser.parse_args()
    
    print("Loading results...")
    df = load_results(args.input)
    print(f"Loaded {len(df)} result entries")
    
    figures_dir = os.path.join(args.output, "figures")
    tables_dir = os.path.join(args.output, "tables")
    
    print("\nGenerating tables...")
    generate_main_table(df, tables_dir)
    
    print("\nGenerating plots...")
    generate_recall_comparison_plot(df, figures_dir)
    generate_latency_vs_recall_plot(df, figures_dir)
    generate_qps_comparison_plot(df, figures_dir)
    
    print("\nGenerating summary report...")
    generate_summary_report(df, args.output)
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
