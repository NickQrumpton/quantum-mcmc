#!/usr/bin/env python3
"""
Create publication-quality plots for Theorem 6 QPE benchmark results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set publication style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# Load data
df = pd.read_csv('theorem6_qpe_resource_benchmark.csv', comment='#')

print('📊 Creating publication-quality plots...')

# Figure 1: Main resource scaling plot
fig1, ax1 = plt.subplots(figsize=(12, 8))

# Create multiple y-axes for different resources
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Plot resources vs error norm (log-log scale)
line1 = ax1.loglog(df['error_norm'], df['total_qubits'], 
                  'o-', color=colors[0], linewidth=3, markersize=12, 
                  label='Total Qubits')

line2 = ax2.loglog(df['error_norm'], df['circuit_depth'], 
                  's-', color=colors[1], linewidth=3, markersize=12,
                  label='Circuit Depth')

line3 = ax3.loglog(df['error_norm'], df['controlled_W_calls'], 
                  '^-', color=colors[2], linewidth=3, markersize=12,
                  label='Controlled-W(P) Calls')

# Formatting
ax1.set_xlabel(r'Error Norm $\|(R(P) + I)|\psi\rangle\|$', fontsize=16)
ax1.set_ylabel('Total Qubits', color=colors[0], fontsize=16, fontweight='bold')
ax2.set_ylabel('Circuit Depth', color=colors[1], fontsize=16, fontweight='bold')
ax3.set_ylabel('Controlled-W(P) Calls', color=colors[2], fontsize=16, fontweight='bold')

ax1.tick_params(axis='y', labelcolor=colors[0], labelsize=14)
ax2.tick_params(axis='y', labelcolor=colors[1], labelsize=14)
ax3.tick_params(axis='y', labelcolor=colors[2], labelsize=14)
ax1.tick_params(axis='x', labelsize=14)

ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_title('Theorem 6 QPE Resource Scaling\n(IMHK Lattice Gaussian Sampling)', 
             fontsize=18, pad=20, fontweight='bold')

# Add k annotations
for _, row in df.iterrows():
    ax1.annotate(f'k={int(row["k_repetitions"])}', 
                (row['error_norm'], row['total_qubits']),
                xytext=(10, 10), textcoords='offset points', 
                fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         edgecolor='black', alpha=0.8))

# Legend
lines = [line1[0], line2[0], line3[0]]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=14, 
          frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
fig1.savefig('theorem6_resource_scaling.pdf', dpi=300, bbox_inches='tight')
fig1.savefig('theorem6_resource_scaling.png', dpi=300, bbox_inches='tight')

# Figure 2: Parameter analysis (2x2 subplots)
fig2, ((ax4, ax5), (ax6, ax7)) = plt.subplots(2, 2, figsize=(15, 10))

k_values = df['k_repetitions']

# Error norm vs k with theoretical line
ax4.semilogy(k_values, df['error_norm'], 'ro-', linewidth=3, 
            markersize=12, label='Measured', markerfacecolor='red',
            markeredgecolor='darkred', markeredgewidth=2)

# Add theoretical 2^(1-k) line
k_theory = np.linspace(k_values.min(), k_values.max(), 100)
C_fit = df['error_norm'].iloc[0] / (2**(1 - df['k_repetitions'].iloc[0]))
theory_line = C_fit * 2**(1 - k_theory)
ax4.plot(k_theory, theory_line, '--', color='gray', linewidth=3, alpha=0.8,
        label=r'Theoretical $2^{1-k}$')

ax4.set_xlabel('k (QPE Repetitions)', fontsize=14)
ax4.set_ylabel('Error Norm', fontsize=14)
ax4.set_title('Error Decay Validation', fontsize=16, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=12)

# Qubits vs k
ax5.plot(k_values, df['total_qubits'], 'bo-', linewidth=3, markersize=12,
         markerfacecolor='blue', markeredgecolor='darkblue', markeredgewidth=2)
ax5.set_xlabel('k (QPE Repetitions)', fontsize=14)
ax5.set_ylabel('Total Qubits', fontsize=14)
ax5.set_title('Qubit Scaling', fontsize=16, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Circuit depth vs k
ax6.semilogy(k_values, df['circuit_depth'], 'go-', linewidth=3, markersize=12,
             markerfacecolor='green', markeredgecolor='darkgreen', markeredgewidth=2)
ax6.set_xlabel('k (QPE Repetitions)', fontsize=14)
ax6.set_ylabel('Circuit Depth', fontsize=14)
ax6.set_title('Depth Scaling', fontsize=16, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Controlled-W calls vs k
ax7.semilogy(k_values, df['controlled_W_calls'], 'mo-', linewidth=3, markersize=12,
             markerfacecolor='magenta', markeredgecolor='darkmagenta', markeredgewidth=2)
ax7.set_xlabel('k (QPE Repetitions)', fontsize=14)
ax7.set_ylabel('Controlled-W(P) Calls', fontsize=14)
ax7.set_title('Gate Complexity', fontsize=16, fontweight='bold')
ax7.grid(True, alpha=0.3)

fig2.suptitle('Theorem 6 Parameter Analysis', fontsize=18, fontweight='bold')
plt.tight_layout()
fig2.savefig('theorem6_parameter_analysis.pdf', dpi=300, bbox_inches='tight')
fig2.savefig('theorem6_parameter_analysis.png', dpi=300, bbox_inches='tight')

print('✅ Plots saved:')
print('   📊 theorem6_resource_scaling.pdf')
print('   📊 theorem6_parameter_analysis.pdf')
print('   📊 PNG versions also created')