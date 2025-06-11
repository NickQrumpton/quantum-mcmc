"""Create publication-quality plots for Theorem 6 validation results.

This script generates comprehensive analysis plots from the validation data.

Author: Nicholas Zhao
Date: 5/31/2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set publication-quality plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'legend.frameon': True,
    'legend.fancybox': True,
    'figure.dpi': 300
})

def load_and_clean_data():
    """Load validation results and clean data."""
    df = pd.read_csv('results/theorem_6_validation_results.csv')
    
    # Clean infinite values
    df = df[df['ratio'] < 20]  # Remove extreme outliers
    df = df[df['test_norm'] < 10]  # Remove invalid norms
    
    return df

def create_main_results_plot(df):
    """Create the main results plot showing error scaling."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Error scaling vs k
    ax = ax1
    colors = ['#2E86AB', '#A23B72']
    markers = ['o', 's']
    
    for i, enhanced in enumerate([False, True]):
        data = df[df['enhanced_precision'] == enhanced]
        
        # Calculate means and standard errors
        k_stats = data.groupby('k').agg({
            'ratio': ['mean', 'std', 'count']
        }).round(3)
        
        k_vals = k_stats.index
        means = k_stats[('ratio', 'mean')]
        stds = k_stats[('ratio', 'std')]
        counts = k_stats[('ratio', 'count')]
        
        # Standard error
        errors = stds / np.sqrt(counts)
        
        label = 'Enhanced Precision' if enhanced else 'Standard Precision'
        ax.errorbar(k_vals, means, yerr=errors, 
                   color=colors[i], marker=markers[i], 
                   markersize=8, linewidth=2, capsize=5,
                   label=label)
    
    # Theoretical bound
    k_theory = [1, 2, 3, 4]
    theory_bounds = [2**(1-k) for k in k_theory]
    ax.plot(k_theory, theory_bounds, 'r--', linewidth=3, 
           label='Theoretical 2^(1-k)', alpha=0.8)
    
    ax.set_xlabel('Repetitions (k)', fontsize=14)
    ax.set_ylabel('Ratio to Theoretical Bound', fontsize=14)
    ax.set_title('Error Scaling Performance', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim(0.1, 20)
    
    # Plot 2: Success rate vs ancilla count
    ax = ax2
    
    for i, enhanced in enumerate([False, True]):
        data = df[df['enhanced_precision'] == enhanced]
        s_success = data.groupby('s')['success'].mean()
        
        label = 'Enhanced' if enhanced else 'Standard'
        ax.plot(s_success.index, s_success.values * 100, 
               color=colors[i], marker=markers[i], 
               markersize=8, linewidth=2, label=label)
    
    ax.set_xlabel('Ancilla Qubits (s)', fontsize=14)
    ax.set_ylabel('Success Rate (%)', fontsize=14)
    ax.set_title('Success Rate vs Precision', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # Plot 3: Resource usage verification
    ax = ax3
    
    # Calculate resource ratio
    df['resource_bound'] = df['k'] * 2**(df['s']+1)
    df['resource_ratio'] = df['controlled_w_calls'] / df['resource_bound']
    
    # Plot resource usage by k
    enhanced_data = df[df['enhanced_precision'] == True]
    
    for k in [1, 2, 3, 4]:
        k_data = enhanced_data[enhanced_data['k'] == k]
        if not k_data.empty:
            ax.scatter(k_data['s'], k_data['resource_ratio'], 
                      label=f'k={k}', s=50, alpha=0.7)
    
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, 
               label='Theorem 6 Bound', alpha=0.8)
    ax.set_xlabel('Ancilla Qubits (s)', fontsize=14)
    ax.set_ylabel('Resource Usage Ratio', fontsize=14)
    ax.set_title('Resource Bound Verification', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.2)
    
    # Plot 4: Performance heatmap
    ax = ax4
    
    enhanced_data = df[df['enhanced_precision'] == True]
    pivot_data = enhanced_data.pivot_table(
        values='ratio', index='s', columns='k', aggfunc='mean'
    )
    
    im = ax.imshow(pivot_data.values, cmap='viridis_r', aspect='auto', 
                   vmin=0.5, vmax=3.0)
    
    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            if not np.isnan(pivot_data.iloc[i, j]):
                text = ax.text(j, i, f'{pivot_data.iloc[i, j]:.2f}',
                             ha="center", va="center", color="white", 
                             fontweight='bold')
    
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_xticklabels(pivot_data.columns)
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_yticklabels(pivot_data.index)
    ax.set_xlabel('Repetitions (k)', fontsize=14)
    ax.set_ylabel('Ancilla Qubits (s)', fontsize=14)
    ax.set_title('Enhanced Precision Performance', fontsize=16, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Ratio to Bound', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/theorem_6_main_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Main results plot saved to results/theorem_6_main_results.png")

def create_robustness_analysis(df):
    """Create robustness analysis across spectral gaps."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Performance across spectral gaps
    ax = ax1
    enhanced_data = df[df['enhanced_precision'] == True]
    
    colors = plt.cm.Set1(np.linspace(0, 1, 3))
    
    for i, delta in enumerate(sorted(enhanced_data['spectral_gap'].unique())):
        gap_data = enhanced_data[np.abs(enhanced_data['spectral_gap'] - delta) < 0.01]
        if not gap_data.empty:
            k_means = gap_data.groupby('k')['ratio'].mean()
            k_stds = gap_data.groupby('k')['ratio'].std()
            
            ax.errorbar(k_means.index, k_means.values, yerr=k_stds.values,
                       color=colors[i], marker='o', markersize=8,
                       linewidth=2, capsize=5, label=f'δ = {delta:.3f}')
    
    # Theoretical bound
    k_vals = [1, 2, 3, 4]
    theory_bounds = [2**(1-k) for k in k_vals]
    ax.plot(k_vals, theory_bounds, 'r--', linewidth=3, 
           label='Theoretical 2^(1-k)', alpha=0.8)
    
    ax.set_xlabel('Repetitions (k)', fontsize=14)
    ax.set_ylabel('Ratio to Bound', fontsize=14)
    ax.set_title('Robustness Across Spectral Gaps', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Success rate by spectral gap
    ax = ax2
    
    for enhanced in [False, True]:
        precision_data = df[df['enhanced_precision'] == enhanced]
        gap_success = precision_data.groupby('spectral_gap')['success'].mean()
        
        label = 'Enhanced' if enhanced else 'Standard'
        ax.plot(gap_success.index, gap_success.values * 100, 
               'o-', markersize=8, linewidth=2, label=label)
    
    ax.set_xlabel('Spectral Gap (δ)', fontsize=14)
    ax.set_ylabel('Success Rate (%)', fontsize=14)
    ax.set_title('Success vs Spectral Gap', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/theorem_6_robustness.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Robustness analysis saved to results/theorem_6_robustness.png")

def create_detailed_analysis(df):
    """Create detailed analysis plots."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Distribution of ratios
    ax = ax1
    
    enhanced_data = df[df['enhanced_precision'] == True]
    standard_data = df[df['enhanced_precision'] == False]
    
    ax.hist(standard_data['ratio'], bins=20, alpha=0.6, 
           label='Standard', color='#A23B72', density=True)
    ax.hist(enhanced_data['ratio'], bins=20, alpha=0.6, 
           label='Enhanced', color='#2E86AB', density=True)
    
    ax.axvline(x=1.2, color='red', linestyle='--', linewidth=2,
               label='Success Threshold')
    ax.set_xlabel('Ratio to Theoretical Bound', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title('Distribution of Performance Ratios', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Circuit complexity analysis
    ax = ax2
    
    enhanced_data = df[df['enhanced_precision'] == True]
    
    # Color by success
    successful = enhanced_data[enhanced_data['success']]
    failed = enhanced_data[~enhanced_data['success']]
    
    ax.scatter(failed['total_ancillas'], failed['circuit_depth'], 
              c='red', alpha=0.6, s=30, label='Failed')
    ax.scatter(successful['total_ancillas'], successful['circuit_depth'], 
              c='green', alpha=0.6, s=30, label='Successful')
    
    ax.set_xlabel('Total Ancilla Qubits', fontsize=14)
    ax.set_ylabel('Circuit Depth', fontsize=14)
    ax.set_title('Circuit Complexity vs Success', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Individual k performance
    ax = ax3
    
    enhanced_data = df[df['enhanced_precision'] == True]
    
    for k in [1, 2, 3, 4]:
        k_data = enhanced_data[enhanced_data['k'] == k]
        if not k_data.empty:
            s_means = k_data.groupby('s')['ratio'].mean()
            ax.plot(s_means.index, s_means.values, 'o-', 
                   markersize=6, linewidth=2, label=f'k={k}')
    
    ax.axhline(y=1.2, color='red', linestyle='--', linewidth=2, 
               alpha=0.7, label='Success Threshold')
    ax.set_xlabel('Ancilla Qubits (s)', fontsize=14)
    ax.set_ylabel('Ratio to Bound', fontsize=14)
    ax.set_title('Performance by k Value', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 4: Computation time analysis
    ax = ax4
    
    for enhanced in [False, True]:
        precision_data = df[df['enhanced_precision'] == enhanced]
        time_means = precision_data.groupby('k')['computation_time'].mean()
        
        label = 'Enhanced' if enhanced else 'Standard'
        ax.plot(time_means.index, time_means.values, 'o-', 
               markersize=8, linewidth=2, label=label)
    
    ax.set_xlabel('Repetitions (k)', fontsize=14)
    ax.set_ylabel('Computation Time (s)', fontsize=14)
    ax.set_title('Computational Cost Scaling', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/theorem_6_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Detailed analysis saved to results/theorem_6_detailed_analysis.png")

def print_summary_statistics(df):
    """Print comprehensive summary statistics."""
    
    print("\n" + "="*60)
    print("THEOREM 6 VALIDATION - STATISTICAL SUMMARY")
    print("="*60)
    
    # Overall statistics
    total_tests = len(df)
    enhanced_tests = len(df[df['enhanced_precision'] == True])
    standard_tests = len(df[df['enhanced_precision'] == False])
    
    print(f"\nOverall Test Statistics:")
    print(f"  Total configurations: {total_tests}")
    print(f"  Enhanced precision tests: {enhanced_tests}")
    print(f"  Standard precision tests: {standard_tests}")
    
    # Success rates
    enhanced_success = df[df['enhanced_precision'] == True]['success'].mean()
    standard_success = df[df['enhanced_precision'] == False]['success'].mean()
    
    print(f"\nSuccess Rates (ratio < 1.2):")
    print(f"  Enhanced precision: {enhanced_success:.1%}")
    print(f"  Standard precision: {standard_success:.1%}")
    print(f"  Improvement factor: {enhanced_success/standard_success:.1f}×")
    
    # Performance statistics
    enhanced_data = df[df['enhanced_precision'] == True]
    standard_data = df[df['enhanced_precision'] == False]
    
    print(f"\nPerformance Ratios:")
    print(f"  Enhanced - Mean: {enhanced_data['ratio'].mean():.3f}")
    print(f"  Enhanced - Median: {enhanced_data['ratio'].median():.3f}")
    print(f"  Enhanced - Best: {enhanced_data['ratio'].min():.3f}")
    print(f"  Standard - Mean: {standard_data['ratio'].mean():.3f}")
    print(f"  Standard - Best: {standard_data['ratio'].min():.3f}")
    
    # Best configurations
    best_configs = enhanced_data.nsmallest(5, 'ratio')
    print(f"\nTop 5 Configurations:")
    for i, (_, row) in enumerate(best_configs.iterrows()):
        print(f"  {i+1}. k={row['k']}, s={row['s']}, δ={row['spectral_gap']:.3f}, "
              f"ratio={row['ratio']:.3f}")
    
    # Resource verification
    violations = df[df['controlled_w_calls'] > df['k'] * 2**(df['s']+1)]
    print(f"\nResource Verification:")
    print(f"  Theorem 6 bound violations: {len(violations)}/{total_tests}")
    print(f"  Compliance rate: {100*(1-len(violations)/total_tests):.1f}%")
    
    # Scaling analysis
    print(f"\nScaling Analysis (Enhanced Precision):")
    k_analysis = enhanced_data.groupby('k')['ratio'].agg(['mean', 'std', 'min', 'max'])
    for k, stats in k_analysis.iterrows():
        print(f"  k={k}: mean={stats['mean']:.3f} ± {stats['std']:.3f}, "
              f"range=[{stats['min']:.3f}, {stats['max']:.3f}]")

def main():
    """Generate all validation plots and analysis."""
    
    print("Generating Theorem 6 validation plots and analysis...")
    
    # Load data
    df = load_and_clean_data()
    
    # Create plots
    create_main_results_plot(df)
    create_robustness_analysis(df)
    create_detailed_analysis(df)
    
    # Print statistics
    print_summary_statistics(df)
    
    print(f"\n✓ All validation plots generated successfully!")
    print(f"✓ Files saved to results/ directory")

if __name__ == "__main__":
    main()