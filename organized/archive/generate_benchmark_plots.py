#!/usr/bin/env python3
"""Generate Publication-Quality Benchmark Plots

This script creates comprehensive visualization and analysis plots for the
Classical IMHK vs Quantum Walk-Based MCMC benchmark results.

Author: Nicholas Zhao
Date: 2025-05-31
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

# Set up styling for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'lines.linewidth': 2,
    'lines.markersize': 8
})

warnings.filterwarnings('ignore')


def load_benchmark_data(results_dir: str = "results") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load benchmark results from CSV files.
    
    Returns:
        Tuple of (classical_df, quantum_df, summary_df)
    """
    classical_df = pd.read_csv(f"{results_dir}/benchmark_classical_results.csv")
    quantum_df = pd.read_csv(f"{results_dir}/benchmark_quantum_results.csv")
    summary_df = pd.read_csv(f"{results_dir}/benchmark_summary.csv")
    
    return classical_df, quantum_df, summary_df


def plot_convergence_comparison(classical_df: pd.DataFrame, 
                              quantum_df: pd.DataFrame,
                              output_dir: str = "results") -> None:
    """Create convergence comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Classical IMHK vs Quantum Walk MCMC: Convergence Analysis', fontsize=18, y=0.98)
    
    # Plot 1: TV Distance vs Iteration for different dimensions
    ax = axes[0, 0]
    
    # Select representative cases
    for dim in [1, 2, 3, 4]:
        if dim in classical_df['dimension'].values:
            classical_subset = classical_df[
                (classical_df['dimension'] == dim) & 
                (classical_df['sigma'] == 1.5)
            ]
            quantum_subset = quantum_df[
                (quantum_df['dimension'] == dim) & 
                (quantum_df['sigma'] == 1.5)
            ]
            
            if not classical_subset.empty:
                ax.semilogy(classical_subset['iteration'], classical_subset['tv_distance'], 
                           'o-', label=f'Classical d={dim}', alpha=0.8)
            
            if not quantum_subset.empty:
                ax.semilogy(quantum_subset['iteration'], quantum_subset['tv_distance'], 
                           's--', label=f'Quantum d={dim}', alpha=0.8)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Variation Distance')
    ax.set_title('Convergence Rate Comparison (σ=1.5)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: TV Distance vs Sigma for final iteration
    ax = axes[0, 1]
    
    classical_final = classical_df.groupby(['dimension', 'sigma']).last().reset_index()
    quantum_final = quantum_df.groupby(['dimension', 'sigma']).last().reset_index()
    
    for dim in [1, 2, 3, 4]:
        classical_dim = classical_final[classical_final['dimension'] == dim]
        quantum_dim = quantum_final[quantum_final['dimension'] == dim]
        
        if not classical_dim.empty:
            ax.plot(classical_dim['sigma'], classical_dim['tv_distance'], 
                   'o-', label=f'Classical d={dim}', alpha=0.8)
        
        if not quantum_dim.empty:
            ax.plot(quantum_dim['sigma'], quantum_dim['tv_distance'], 
                   's--', label=f'Quantum d={dim}', alpha=0.8)
    
    ax.set_xlabel('σ (Gaussian Parameter)')
    ax.set_ylabel('Final TV Distance')
    ax.set_title('Final Convergence vs Parameter')
    ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Resource Scaling
    ax = axes[1, 0]
    
    # Classical: Effective Sample Size vs Dimension
    classical_ess = classical_df.groupby('dimension')['effective_sample_size'].mean()
    quantum_qubits = quantum_df.groupby('dimension')['num_qubits'].mean()
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(classical_ess.index, classical_ess.values, 'o-', 
                   color='blue', label='Classical ESS', linewidth=3)
    line2 = ax2.plot(quantum_qubits.index, quantum_qubits.values, 's-', 
                    color='red', label='Quantum Qubits', linewidth=3)
    
    ax.set_xlabel('Lattice Dimension')
    ax.set_ylabel('Effective Sample Size', color='blue')
    ax2.set_ylabel('Number of Qubits', color='red')
    ax.set_title('Resource Scaling')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Runtime Comparison
    ax = axes[1, 1]
    
    classical_runtime = classical_df.groupby('dimension')['runtime_seconds'].sum()
    quantum_runtime = quantum_df.groupby('dimension')['runtime_seconds'].sum()
    
    x = np.arange(len(classical_runtime))
    width = 0.35
    
    ax.bar(x - width/2, classical_runtime.values, width, 
           label='Classical', alpha=0.8, color='skyblue')
    ax.bar(x + width/2, quantum_runtime.values, width, 
           label='Quantum', alpha=0.8, color='lightcoral')
    
    ax.set_xlabel('Lattice Dimension')
    ax.set_ylabel('Total Runtime (seconds)')
    ax.set_title('Computational Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(classical_runtime.index)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/benchmark_convergence_analysis.png", 
                dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/benchmark_convergence_analysis.pdf", 
                bbox_inches='tight')
    plt.show()


def plot_resource_analysis(classical_df: pd.DataFrame, 
                         quantum_df: pd.DataFrame,
                         output_dir: str = "results") -> None:
    """Create detailed resource analysis plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Resource Usage and Scaling Analysis', fontsize=18, y=0.98)
    
    # Plot 1: Quantum Circuit Complexity
    ax = axes[0, 0]
    
    quantum_final = quantum_df.groupby(['dimension', 'sigma']).last().reset_index()
    
    scatter = ax.scatter(quantum_final['num_qubits'], quantum_final['circuit_depth'], 
                        c=quantum_final['dimension'], s=quantum_final['controlled_w_calls']/10,
                        alpha=0.7, cmap='viridis')
    
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Circuit Depth')
    ax.set_title('Quantum Circuit Complexity\n(Color: dimension, Size: controlled-W calls)')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Lattice Dimension')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Controlled-W Calls vs Dimension
    ax = axes[0, 1]
    
    controlled_w_by_dim = quantum_df.groupby('dimension')['controlled_w_calls'].mean()
    circuit_depth_by_dim = quantum_df.groupby('dimension')['circuit_depth'].mean()
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(controlled_w_by_dim.index, controlled_w_by_dim.values, 
                   'o-', color='purple', label='Controlled-W Calls', linewidth=3)
    line2 = ax2.plot(circuit_depth_by_dim.index, circuit_depth_by_dim.values, 
                    's-', color='orange', label='Circuit Depth', linewidth=3)
    
    ax.set_xlabel('Lattice Dimension')
    ax.set_ylabel('Controlled-W Calls', color='purple')
    ax2.set_ylabel('Circuit Depth', color='orange')
    ax.set_title('Quantum Resource Scaling')
    ax.tick_params(axis='y', labelcolor='purple')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Acceptance Rate vs Dimension (Classical)
    ax = axes[1, 0]
    
    acceptance_by_dim_sigma = classical_df.groupby(['dimension', 'sigma'])['acceptance_rate'].mean().unstack()
    
    for sigma in acceptance_by_dim_sigma.columns:
        ax.plot(acceptance_by_dim_sigma.index, acceptance_by_dim_sigma[sigma], 
               'o-', label=f'σ={sigma:.1f}', linewidth=2, markersize=8)
    
    ax.set_xlabel('Lattice Dimension')
    ax.set_ylabel('Acceptance Rate')
    ax.set_title('Classical IMHK Acceptance Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 4: Quantum Fidelity vs Classical ESS
    ax = axes[1, 1]
    
    classical_metrics = classical_df.groupby(['dimension', 'sigma']).agg({
        'effective_sample_size': 'mean',
        'tv_distance': 'last'
    }).reset_index()
    
    quantum_metrics = quantum_df.groupby(['dimension', 'sigma']).agg({
        'overlap_fidelity': 'last',
        'tv_distance': 'last'
    }).reset_index()
    
    # Merge data
    merged_data = pd.merge(classical_metrics, quantum_metrics, 
                          on=['dimension', 'sigma'], suffixes=('_classical', '_quantum'))
    
    scatter = ax.scatter(merged_data['effective_sample_size'], 
                        merged_data['overlap_fidelity'], 
                        c=merged_data['dimension'], 
                        s=100, alpha=0.7, cmap='plasma')
    
    ax.set_xlabel('Classical Effective Sample Size')
    ax.set_ylabel('Quantum Overlap Fidelity')
    ax.set_title('Sampling Quality Comparison')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Lattice Dimension')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/benchmark_resource_analysis.png", 
                dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/benchmark_resource_analysis.pdf", 
                bbox_inches='tight')
    plt.show()


def plot_performance_summary(classical_df: pd.DataFrame, 
                           quantum_df: pd.DataFrame,
                           output_dir: str = "results") -> None:
    """Create performance summary plots suitable for publication."""
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Performance Summary: Classical IMHK vs Quantum Walk MCMC', 
                 fontsize=18, y=1.02)
    
    # Plot 1: Convergence Rate Comparison
    ax = axes[0]
    
    # Calculate convergence rates (negative slope of log(TV) vs iteration)
    convergence_rates_classical = []
    convergence_rates_quantum = []
    dimensions = []
    
    for dim in [1, 2, 3, 4]:
        for sigma in [1.0, 1.5, 2.0]:
            classical_subset = classical_df[
                (classical_df['dimension'] == dim) & 
                (classical_df['sigma'] == sigma)
            ]
            quantum_subset = quantum_df[
                (quantum_df['dimension'] == dim) & 
                (quantum_df['sigma'] == sigma)
            ]
            
            if len(classical_subset) > 2:
                # Fit linear regression to log(TV) vs iteration
                log_tv = np.log(classical_subset['tv_distance'] + 1e-10)
                iterations = classical_subset['iteration']
                slope_classical = np.polyfit(iterations, log_tv, 1)[0]
                convergence_rates_classical.append(-slope_classical)
            else:
                convergence_rates_classical.append(0)
            
            if len(quantum_subset) > 2:
                log_tv = np.log(quantum_subset['tv_distance'] + 1e-10)
                iterations = quantum_subset['iteration']
                slope_quantum = np.polyfit(iterations, log_tv, 1)[0]
                convergence_rates_quantum.append(-slope_quantum)
            else:
                convergence_rates_quantum.append(0)
            
            dimensions.append(dim)
    
    # Create bar plot
    x = np.arange(len(dimensions))
    width = 0.35
    
    classical_bars = ax.bar(x - width/2, convergence_rates_classical, width, 
                           label='Classical IMHK', alpha=0.8, color='steelblue')
    quantum_bars = ax.bar(x + width/2, convergence_rates_quantum, width, 
                         label='Quantum Walk', alpha=0.8, color='darkorange')
    
    ax.set_xlabel('Experiment Index')
    ax.set_ylabel('Convergence Rate (1/iteration)')
    ax.set_title('Convergence Rate Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in classical_bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Resource Efficiency
    ax = axes[1]
    
    # Define efficiency metric: 1 / (final_TV_distance * computational_cost)
    classical_final = classical_df.groupby(['dimension', 'sigma']).agg({
        'tv_distance': 'last',
        'runtime_seconds': 'sum',
        'effective_sample_size': 'mean'
    }).reset_index()
    
    quantum_final = quantum_df.groupby(['dimension', 'sigma']).agg({
        'tv_distance': 'last',
        'runtime_seconds': 'sum',
        'num_qubits': 'mean'
    }).reset_index()
    
    # Efficiency = ESS / (TV_distance * runtime)
    classical_efficiency = (classical_final['effective_sample_size'] / 
                           (classical_final['tv_distance'] * classical_final['runtime_seconds']))
    
    # Quantum efficiency = 1 / (TV_distance * qubits)
    quantum_efficiency = 1 / (quantum_final['tv_distance'] * quantum_final['num_qubits'])
    
    # Normalize for comparison
    classical_efficiency_norm = classical_efficiency / classical_efficiency.max()
    quantum_efficiency_norm = quantum_efficiency / quantum_efficiency.max()
    
    x = np.arange(len(classical_efficiency))
    ax.plot(x, classical_efficiency_norm, 'o-', label='Classical IMHK', 
           linewidth=3, markersize=8, color='steelblue')
    ax.plot(x, quantum_efficiency_norm, 's-', label='Quantum Walk', 
           linewidth=3, markersize=8, color='darkorange')
    
    ax.set_xlabel('Experiment Configuration')
    ax.set_ylabel('Normalized Efficiency')
    ax.set_title('Resource Efficiency Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Scaling Analysis
    ax = axes[2]
    
    dimensions_unique = sorted(classical_df['dimension'].unique())
    
    # Calculate average performance metrics by dimension
    classical_by_dim = classical_df.groupby('dimension').agg({
        'tv_distance': 'mean',
        'runtime_seconds': 'mean',
        'effective_sample_size': 'mean'
    })
    
    quantum_by_dim = quantum_df.groupby('dimension').agg({
        'tv_distance': 'mean',
        'runtime_seconds': 'mean',
        'num_qubits': 'mean'
    })
    
    # Plot TV distance scaling
    ax.semilogy(dimensions_unique, classical_by_dim.loc[dimensions_unique, 'tv_distance'], 
               'o-', label='Classical TV Distance', linewidth=3, markersize=8, color='blue')
    ax.semilogy(dimensions_unique, quantum_by_dim.loc[dimensions_unique, 'tv_distance'], 
               's-', label='Quantum TV Distance', linewidth=3, markersize=8, color='red')
    
    ax.set_xlabel('Lattice Dimension')
    ax.set_ylabel('Average TV Distance')
    ax.set_title('Scaling with Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/benchmark_performance_summary.png", 
                dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/benchmark_performance_summary.pdf", 
                bbox_inches='tight')
    plt.show()


def generate_summary_table(classical_df: pd.DataFrame, 
                         quantum_df: pd.DataFrame,
                         output_dir: str = "results") -> pd.DataFrame:
    """Generate summary table for publication."""
    
    # Calculate summary statistics
    summary_data = []
    
    for dim in sorted(classical_df['dimension'].unique()):
        for sigma in sorted(classical_df['sigma'].unique()):
            classical_subset = classical_df[
                (classical_df['dimension'] == dim) & 
                (classical_df['sigma'] == sigma)
            ]
            quantum_subset = quantum_df[
                (quantum_df['dimension'] == dim) & 
                (quantum_df['sigma'] == sigma)
            ]
            
            if classical_subset.empty or quantum_subset.empty:
                continue
            
            # Classical metrics
            classical_final_tv = classical_subset['tv_distance'].iloc[-1]
            classical_acceptance = classical_subset['acceptance_rate'].mean()
            classical_ess = classical_subset['effective_sample_size'].iloc[-1]
            classical_runtime = classical_subset['runtime_seconds'].sum()
            
            # Quantum metrics
            quantum_final_tv = quantum_subset['tv_distance'].iloc[-1]
            quantum_fidelity = quantum_subset['overlap_fidelity'].iloc[-1]
            quantum_qubits = quantum_subset['num_qubits'].iloc[-1]
            quantum_depth = quantum_subset['circuit_depth'].iloc[-1]
            quantum_runtime = quantum_subset['runtime_seconds'].sum()
            
            summary_data.append({
                'Dimension': dim,
                'Sigma': sigma,
                'Classical_Final_TV': classical_final_tv,
                'Classical_Acceptance': classical_acceptance,
                'Classical_ESS': classical_ess,
                'Classical_Runtime': classical_runtime,
                'Quantum_Final_TV': quantum_final_tv,
                'Quantum_Fidelity': quantum_fidelity,
                'Quantum_Qubits': quantum_qubits,
                'Quantum_Depth': quantum_depth,
                'Quantum_Runtime': quantum_runtime,
                'TV_Ratio': quantum_final_tv / classical_final_tv,
                'Runtime_Ratio': quantum_runtime / classical_runtime
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Format for publication
    summary_df_formatted = summary_df.copy()
    for col in ['Classical_Final_TV', 'Quantum_Final_TV', 'TV_Ratio']:
        summary_df_formatted[col] = summary_df_formatted[col].apply(lambda x: f'{x:.4f}')
    for col in ['Classical_Acceptance', 'Quantum_Fidelity']:
        summary_df_formatted[col] = summary_df_formatted[col].apply(lambda x: f'{x:.3f}')
    for col in ['Classical_Runtime', 'Quantum_Runtime', 'Runtime_Ratio']:
        summary_df_formatted[col] = summary_df_formatted[col].apply(lambda x: f'{x:.2f}')
    
    # Save to files
    summary_df.to_csv(f"{output_dir}/benchmark_summary_table.csv", index=False)
    summary_df_formatted.to_csv(f"{output_dir}/benchmark_summary_table_formatted.csv", index=False)
    
    # Generate LaTeX table
    latex_table = summary_df_formatted.to_latex(index=False, escape=False)
    with open(f"{output_dir}/benchmark_summary_table.tex", 'w') as f:
        f.write(latex_table)
    
    print("Summary table saved to:")
    print(f"  - {output_dir}/benchmark_summary_table.csv")
    print(f"  - {output_dir}/benchmark_summary_table_formatted.csv") 
    print(f"  - {output_dir}/benchmark_summary_table.tex")
    
    return summary_df


def main():
    """Generate all benchmark plots and analysis."""
    print("Generating benchmark plots and analysis...")
    
    # Load data
    try:
        classical_df, quantum_df, summary_df = load_benchmark_data()
        print(f"Loaded {len(classical_df)} classical and {len(quantum_df)} quantum results")
    except FileNotFoundError:
        print("Benchmark data not found. Please run 'python benchmark_imhk_vs_quantum.py' first.")
        return
    
    # Create output directory
    Path("results").mkdir(exist_ok=True)
    
    # Generate plots
    print("Creating convergence analysis plots...")
    plot_convergence_comparison(classical_df, quantum_df)
    
    print("Creating resource analysis plots...")
    plot_resource_analysis(classical_df, quantum_df)
    
    print("Creating performance summary plots...")
    plot_performance_summary(classical_df, quantum_df)
    
    # Generate summary table
    print("Generating summary table...")
    summary_table = generate_summary_table(classical_df, quantum_df)
    
    print("\nBenchmark analysis completed!")
    print("Generated files:")
    print("  - results/benchmark_convergence_analysis.png/.pdf")
    print("  - results/benchmark_resource_analysis.png/.pdf")
    print("  - results/benchmark_performance_summary.png/.pdf")
    print("  - results/benchmark_summary_table.csv/.tex")


if __name__ == "__main__":
    main()