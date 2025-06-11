# Use Cases and Examples - Quantum MCMC

## Overview

This document provides detailed usage scenarios demonstrating the practical application of the quantum-mcmc package. Each use case includes the goal, step-by-step instructions, complete code examples, and descriptions of expected outputs.

## Table of Contents

1. [Classical Reversible Markov Chain Analysis](#use-case-1-classical-reversible-markov-chain-analysis)
2. [Quantum Walk Spectral Gap Estimation](#use-case-2-quantum-walk-spectral-gap-estimation)
3. [Lattice Gaussian Sampling](#use-case-3-lattice-gaussian-sampling)
4. [Metropolis-Hastings with Quantum Enhancement](#use-case-4-metropolis-hastings-with-quantum-enhancement)

---

## Use Case 1: Classical Reversible Markov Chain Analysis

### Goal

Analyze a classical reversible Markov chain to understand its mixing properties, stationary distribution, and prepare it for quantum enhancement.

### Step-by-Step Instructions

1. Construct a reversible Markov chain
2. Verify stochasticity and reversibility
3. Compute stationary distribution
4. Analyze spectral properties
5. Estimate mixing time
6. Visualize the chain structure

### Complete Code Example

```python
import numpy as np
import matplotlib.pyplot as plt
from quantum_mcmc.classical import (
    build_two_state_chain,
    build_metropolis_chain,
    is_stochastic,
    is_reversible,
    stationary_distribution,
    discriminant_matrix,
    spectral_gap,
    mixing_time_bound
)
from quantum_mcmc.utils import plot_markov_chain_graph, plot_singular_values

# Step 1: Construct a two-state Markov chain
print("=== Two-State Markov Chain Analysis ===")
alpha, beta = 0.2, 0.3  # Transition probabilities
P_2state = build_two_state_chain(alpha, beta)
print(f"Transition matrix:\n{P_2state}")

# Step 2: Verify properties
assert is_stochastic(P_2state), "Matrix is not stochastic!"
pi_2state = stationary_distribution(P_2state)
assert is_reversible(P_2state, pi_2state), "Chain is not reversible!"
print(f"Stationary distribution: {pi_2state}")
print(f"Expected: [{beta/(alpha+beta):.3f}, {alpha/(alpha+beta):.3f}]")

# Step 3: Analyze spectral properties
D_2state = discriminant_matrix(P_2state, pi_2state)
gap_2state = spectral_gap(D_2state)
print(f"Spectral gap: {gap_2state:.4f}")

# Step 4: More complex example - Gaussian target
print("\n=== Metropolis-Hastings Chain Analysis ===")
n_states = 50
x = np.linspace(-4, 4, n_states)
# Mixture of Gaussians target
target_dist = 0.6 * np.exp(-(x-1)**2/0.5) + 0.4 * np.exp(-(x+1)**2/0.5)
target_dist /= target_dist.sum()

# Build Metropolis chain
P_metro = build_metropolis_chain(target_dist, proposal_std=0.8)

# Verify properties
print(f"Chain dimension: {P_metro.shape}")
print(f"Is stochastic: {is_stochastic(P_metro)}")
pi_metro = stationary_distribution(P_metro)
print(f"Is reversible: {is_reversible(P_metro, pi_metro)}")

# Compare target and stationary distributions
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(x, target_dist, 'b-', label='Target', linewidth=2)
plt.plot(x, pi_metro, 'r--', label='Stationary', linewidth=2)
plt.xlabel('State')
plt.ylabel('Probability')
plt.title('Target vs Stationary Distribution')
plt.legend()

# Spectral analysis
D_metro = discriminant_matrix(P_metro, pi_metro)
gap_metro = spectral_gap(D_metro)
mixing_bound = mixing_time_bound(D_metro, epsilon=0.01)

plt.subplot(1, 2, 2)
singular_vals = np.linalg.svdvals(D_metro)[:10]
plt.semilogy(singular_vals, 'o-')
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.title(f'Spectral Analysis (gap = {gap_metro:.4f})')
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Spectral gap: {gap_metro:.4f}")
print(f"Mixing time bound (µ=0.01): {mixing_bound:.0f} steps")

# Step 5: Simulate chain evolution
initial_dist = np.zeros(n_states)
initial_dist[n_states//2] = 1.0  # Start at center

evolution_steps = [1, 5, 10, 50, 100]
plt.figure(figsize=(12, 8))

for i, t in enumerate(evolution_steps):
    dist_t = initial_dist @ np.linalg.matrix_power(P_metro, t)
    
    plt.subplot(2, 3, i+1)
    plt.plot(x, dist_t, 'g-', alpha=0.7, label=f't={t}')
    plt.plot(x, pi_metro, 'r--', alpha=0.5, label='Stationary')
    plt.fill_between(x, dist_t, alpha=0.3)
    plt.xlabel('State')
    plt.ylabel('Probability')
    plt.title(f'Distribution at t={t}')
    plt.legend()

plt.tight_layout()
plt.show()
```

### Expected Output

1. **Two-state chain analysis**: Exact stationary distribution matching theoretical values
2. **Metropolis chain**: Stationary distribution closely matching the target
3. **Spectral gap**: Typically 0.01-0.1 for well-mixed chains
4. **Mixing visualization**: Convergence to stationary distribution over time

---

## Use Case 2: Quantum Walk Spectral Gap Estimation

### Goal

Construct a quantum walk operator from a classical Markov chain and use quantum phase estimation to analyze its spectral properties, demonstrating the quantum approach to mixing time analysis.

### Step-by-Step Instructions

1. Build classical Markov chain
2. Construct quantum walk operator
3. Apply quantum phase estimation
4. Extract and analyze eigenphases
5. Compare classical and quantum mixing times

### Complete Code Example

```python
import numpy as np
import matplotlib.pyplot as plt
from quantum_mcmc.classical import build_metropolis_chain, discriminant_matrix, phase_gap
from quantum_mcmc.core import (
    prepare_walk_operator,
    quantum_phase_estimation,
    analyze_qpe_results,
    walk_eigenvalues
)
from quantum_mcmc.utils import prepare_uniform_superposition

# Step 1: Create a challenging Markov chain (slow mixing)
n = 20  # State space size
# Create bottleneck structure: two communities weakly connected
P = np.zeros((n, n))
comm_size = n // 2

# Strong connections within communities
for i in range(comm_size):
    for j in range(comm_size):
        if i != j:
            P[i, j] = 0.9 / (comm_size - 1)
            P[i + comm_size, j + comm_size] = 0.9 / (comm_size - 1)

# Weak connections between communities
epsilon = 0.01  # Bottleneck strength
P[comm_size-1, comm_size] = epsilon
P[comm_size, comm_size-1] = epsilon

# Self-loops to ensure stochasticity
for i in range(n):
    P[i, i] = 1.0 - P[i, :].sum()

print(f"Created bottleneck chain with {n} states")
print(f"Inter-community connection strength: {epsilon}")

# Step 2: Build quantum walk operator
W_circuit = prepare_walk_operator(P, backend="qiskit")
W_matrix = prepare_walk_operator(P, backend="matrix")

print(f"\nQuantum walk operator:")
print(f"  Circuit depth: {W_circuit.depth()}")
print(f"  Number of qubits: {W_circuit.num_qubits}")
print(f"  Operator dimension: {W_matrix.shape}")

# Step 3: Theoretical eigenvalue analysis
theoretical_eigenvals = walk_eigenvalues(P)
print(f"\nTheoretical eigenvalues (first 5): {theoretical_eigenvals[:5]}")

# Step 4: Quantum Phase Estimation
print("\n=== Quantum Phase Estimation ===")

# Prepare initial state (uniform superposition over edges)
initial_state = prepare_uniform_superposition(W_circuit.num_qubits)

# Run QPE with different precision levels
precisions = [4, 6, 8, 10]
results_by_precision = {}

for n_ancilla in precisions:
    print(f"\nRunning QPE with {n_ancilla} ancilla qubits...")
    
    qpe_results = quantum_phase_estimation(
        W_circuit,
        num_ancilla=n_ancilla,
        initial_state=initial_state,
        backend="statevector"
    )
    
    analysis = analyze_qpe_results(qpe_results)
    results_by_precision[n_ancilla] = analysis
    
    print(f"  Dominant phases found: {len(analysis['dominant_phases'])}")
    print(f"  Top 3 phases: {analysis['dominant_phases'][:3]}")
    print(f"  Phase resolution: {analysis['resolution']:.6f}")

# Step 5: Visualize QPE results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (n_ancilla, analysis) in enumerate(results_by_precision.items()):
    ax = axes[idx]
    
    # Plot phase histogram
    phases = analysis['dominant_phases'][:20]  # Top 20
    probs = analysis['dominant_probabilities'][:20]
    
    ax.bar(phases, probs, width=analysis['resolution'], 
           alpha=0.7, edgecolor='black')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Probability')
    ax.set_title(f'QPE Results ({n_ancilla} ancilla qubits)')
    ax.set_xlim(-0.1, 1.1)
    
    # Mark theoretical phases
    theoretical_phases = np.angle(theoretical_eigenvals) / (2 * np.pi)
    theoretical_phases = (theoretical_phases + 1) % 1  # Map to [0, 1)
    
    for t_phase in theoretical_phases[:10]:
        ax.axvline(t_phase, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Step 6: Extract spectral gap from QPE
best_analysis = results_by_precision[10]  # Use highest precision
measured_phases = best_analysis['dominant_phases']

# Find phase closest to 0 (stationary state)
stationary_phase_idx = np.argmin(np.minimum(measured_phases, 1 - measured_phases))
stationary_phase = measured_phases[stationary_phase_idx]

# Find second smallest phase (gap)
other_phases = np.delete(measured_phases, stationary_phase_idx)
if len(other_phases) > 0:
    phase_gaps = np.minimum(other_phases, 1 - other_phases)
    measured_phase_gap = np.min(phase_gaps[phase_gaps > 0.001])  # Ignore very small
else:
    measured_phase_gap = 0.5

print(f"\n=== Spectral Gap Analysis ===")
print(f"Measured stationary phase: {stationary_phase:.6f} (expect H 0)")
print(f"Measured phase gap: {measured_phase_gap:.6f}")

# Compare with classical
D = discriminant_matrix(P)
classical_phase_gap = phase_gap(D)
print(f"Theoretical phase gap: {classical_phase_gap:.6f}")
print(f"Relative error: {abs(measured_phase_gap - classical_phase_gap)/classical_phase_gap:.2%}")

# Step 7: Mixing time comparison
classical_mixing = int(np.ceil(2.0 / classical_phase_gap * np.log(n / 0.01)))
quantum_mixing = int(np.ceil(2.0 / np.sqrt(classical_phase_gap) * np.log(n / 0.01)))

print(f"\n=== Mixing Time Comparison ===")
print(f"Classical mixing time: {classical_mixing} steps")
print(f"Quantum mixing time: {quantum_mixing} steps")
print(f"Quantum speedup: {classical_mixing/quantum_mixing:.1f}x")

# Visualize speedup
gaps = np.logspace(-3, -0.3, 50)
classical_times = 2.0 / gaps * np.log(n / 0.01)
quantum_times = 2.0 / np.sqrt(gaps) * np.log(n / 0.01)

plt.figure(figsize=(8, 6))
plt.loglog(gaps, classical_times, 'b-', label='Classical', linewidth=2)
plt.loglog(gaps, quantum_times, 'r--', label='Quantum', linewidth=2)
plt.axvline(classical_phase_gap, color='green', linestyle=':', 
            label=f'This chain (gap={classical_phase_gap:.4f})')
plt.xlabel('Phase Gap')
plt.ylabel('Mixing Time')
plt.title('Quantum vs Classical Mixing Time Scaling')
plt.legend()
plt.grid(True)
plt.show()
```

### Expected Output

1. **QPE Phase Histograms**: Increasingly precise phase measurements with more ancilla qubits
2. **Spectral Gap Measurement**: Agreement between QPE and theoretical values within resolution
3. **Mixing Time Speedup**: Quadratic improvement for small spectral gaps
4. **Scaling Plot**: Clear separation between classical O(1/´) and quantum O(1/´) scaling

---

## Use Case 3: Lattice Gaussian Sampling

### Goal

Implement both classical and quantum approaches to sampling from a discrete Gaussian distribution on a lattice, demonstrating practical applications in cryptography and optimization.

### Step-by-Step Instructions

1. Define discrete Gaussian distribution on lattice
2. Build Metropolis-Hastings chain for the target
3. Construct quantum walk and reflection operators
4. Compare sampling efficiency
5. Analyze sample quality

### Complete Code Example

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from quantum_mcmc.classical import build_metropolis_chain, stationary_distribution
from quantum_mcmc.core import (
    prepare_walk_operator,
    approximate_reflection_operator,
    apply_reflection_operator
)
from quantum_mcmc.utils import (
    prepare_gaussian_state,
    total_variation_distance,
    effective_sample_size,
    sample_quality_metrics
)

# Step 1: Define lattice Gaussian parameters
lattice_size = 64  # Points on 1D lattice
center = lattice_size // 2
sigma = 5.0  # Standard deviation

# Create discrete Gaussian distribution
x = np.arange(lattice_size)
target_dist = np.exp(-(x - center)**2 / (2 * sigma**2))
target_dist /= target_dist.sum()

print(f"=== Lattice Gaussian Sampling ===")
print(f"Lattice size: {lattice_size}")
print(f"Gaussian center: {center}")
print(f"Standard deviation: {sigma}")
print(f"Effective support: ~{int(4*sigma)} points")

# Step 2: Classical Metropolis-Hastings sampling
print("\n=== Classical Approach ===")

# Build transition matrix with local moves
P_classical = build_metropolis_chain(target_dist, proposal_std=2.0)

# Run classical MCMC
n_samples = 10000
current_state = center
classical_samples = []

for _ in range(n_samples):
    # Sample next state
    probs = P_classical[current_state, :]
    current_state = np.random.choice(lattice_size, p=probs)
    classical_samples.append(current_state)

classical_samples = np.array(classical_samples)

# Analyze classical samples
burn_in = 1000
classical_samples_burned = classical_samples[burn_in:]
classical_metrics = sample_quality_metrics(
    classical_samples_burned,
    target_distribution=target_dist,
    n_bins=lattice_size
)

print(f"Classical sampling results:")
print(f"  Effective sample size: {classical_metrics['effective_sample_size']:.1f}")
print(f"  Autocorrelation time: {classical_metrics['autocorrelation_time']:.1f}")
print(f"  Total variation distance: {classical_metrics['tv_distance']:.4f}")

# Step 3: Quantum approach setup
print("\n=== Quantum Approach ===")

# Build quantum walk operator
W = prepare_walk_operator(P_classical, pi=target_dist, backend="qiskit")
print(f"Quantum walk operator constructed")
print(f"  Circuit qubits: {W.num_qubits}")
print(f"  Circuit depth: {W.depth()}")

# Build reflection operator
R = approximate_reflection_operator(
    W,
    num_ancilla=8,
    phase_threshold=0.05
)
print(f"Reflection operator constructed")
print(f"  Total qubits: {R.num_qubits}")
print(f"  Circuit depth: {R.depth()}")

# Step 4: Quantum state preparation and evolution
# Prepare approximate Gaussian state
n_qubits = int(np.ceil(np.log2(lattice_size)))
initial_state = prepare_gaussian_state(
    mean=center,
    std=sigma,
    num_qubits=n_qubits
)

print(f"\nQuantum evolution simulation...")
# Note: In practice, this would run on quantum hardware
# Here we simulate the expected behavior

# Apply reflection operator multiple times (amplitude amplification)
n_reflections = int(np.pi / (4 * np.sqrt(1/lattice_size)))
print(f"Applying {n_reflections} reflection operations")

# Simulate quantum sampling distribution (idealized)
# In reality, this comes from quantum measurements
quantum_probs = target_dist.copy()
# Add small perturbation to simulate finite precision
quantum_probs += 0.01 * np.random.randn(lattice_size)
quantum_probs = np.maximum(quantum_probs, 0)
quantum_probs /= quantum_probs.sum()

# Step 5: Visualization and comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Target distribution
ax = axes[0, 0]
ax.bar(x, target_dist, alpha=0.7, color='blue')
ax.set_title('Target Gaussian Distribution')
ax.set_xlabel('Lattice Point')
ax.set_ylabel('Probability')

# Plot 2: Classical samples histogram
ax = axes[0, 1]
classical_hist, _ = np.histogram(classical_samples_burned, bins=lattice_size, density=True)
classical_hist *= (x[1] - x[0])  # Normalize
ax.bar(x, classical_hist, alpha=0.7, color='green')
ax.plot(x, target_dist, 'r--', linewidth=2, label='Target')
ax.set_title(f'Classical Samples (ESS={classical_metrics["effective_sample_size"]:.0f})')
ax.set_xlabel('Lattice Point')
ax.set_ylabel('Probability')
ax.legend()

# Plot 3: Quantum distribution (simulated)
ax = axes[0, 2]
ax.bar(x, quantum_probs, alpha=0.7, color='purple')
ax.plot(x, target_dist, 'r--', linewidth=2, label='Target')
ax.set_title('Quantum Sampling Distribution')
ax.set_xlabel('Lattice Point')
ax.set_ylabel('Probability')
ax.legend()

# Plot 4: Autocorrelation function
ax = axes[1, 0]
max_lag = 100
acf = [1.0]
for lag in range(1, max_lag):
    if lag < len(classical_samples_burned) - 1:
        c0 = np.cov(classical_samples_burned[:-lag], classical_samples_burned[lag:])[0, 1]
        c_var = np.var(classical_samples_burned)
        acf.append(c0 / c_var if c_var > 0 else 0)

ax.plot(range(len(acf)), acf, 'b-', linewidth=2)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax.set_title('Classical Sample Autocorrelation')
ax.set_xlabel('Lag')
ax.set_ylabel('ACF')
ax.set_ylim(-0.2, 1.0)

# Plot 5: TV distance evolution
ax = axes[1, 1]
tv_distances = []
sample_sizes = np.logspace(2, np.log10(len(classical_samples_burned)), 20, dtype=int)

for n in sample_sizes:
    subsample = classical_samples_burned[:n]
    hist, _ = np.histogram(subsample, bins=lattice_size, density=True)
    hist *= (x[1] - x[0])
    tv = total_variation_distance(hist, target_dist)
    tv_distances.append(tv)

ax.loglog(sample_sizes, tv_distances, 'g-', linewidth=2, label='Classical')
ax.axhline(y=0.01, color='r', linestyle='--', label='Target accuracy')
ax.set_xlabel('Number of Samples')
ax.set_ylabel('Total Variation Distance')
ax.set_title('Convergence Analysis')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Resource comparison
ax = axes[1, 2]
categories = ['Classical\nSamples', 'Quantum\nCircuit Depth', 'Quantum\nQubits']
classical_cost = classical_metrics['autocorrelation_time'] * 10  # Samples for µ=0.01
quantum_depth = R.depth()
quantum_qubits = R.num_qubits

values = [classical_cost, quantum_depth, quantum_qubits]
colors = ['green', 'purple', 'orange']

bars = ax.bar(categories, values, color=colors, alpha=0.7)
ax.set_ylabel('Resource Count')
ax.set_title('Resource Requirements Comparison')
ax.set_yscale('log')

# Add value labels
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(value)}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Summary statistics
print(f"\n=== Summary Comparison ===")
print(f"Classical approach:")
print(f"  Samples needed (µ=0.01): ~{int(classical_cost)}")
print(f"  Autocorrelation time: {classical_metrics['autocorrelation_time']:.1f}")
print(f"  Efficiency: {classical_metrics['efficiency']:.3f}")

quantum_tv = total_variation_distance(quantum_probs, target_dist)
print(f"\nQuantum approach:")
print(f"  Circuit depth: {quantum_depth}")
print(f"  Total qubits: {quantum_qubits}")
print(f"  TV distance: {quantum_tv:.4f}")
print(f"  Speedup potential: {classical_cost/quantum_depth:.1f}x")
```

### Expected Output

1. **Distribution Plots**: Visual comparison of target, classical samples, and quantum distribution
2. **Autocorrelation**: Decay showing mixing behavior of classical chain
3. **Convergence**: TV distance decreasing with sample size
4. **Resource Comparison**: Quantum advantage in terms of circuit depth vs samples needed

---

## Use Case 4: Metropolis-Hastings with Quantum Enhancement

### Goal

Implement a complete quantum-enhanced Metropolis-Hastings algorithm for sampling from a multimodal distribution, showcasing the full pipeline from classical chain construction to quantum speedup.

### Step-by-Step Instructions

1. Define challenging multimodal target distribution
2. Build classical Metropolis-Hastings chain
3. Identify bottlenecks using spectral analysis
4. Apply quantum walk with amplitude amplification
5. Compare mixing times and sample quality

### Complete Code Example

```python
import numpy as np
import matplotlib.pyplot as plt
from quantum_mcmc import (
    build_metropolis_chain,
    discriminant_matrix,
    spectral_analysis,
    prepare_walk_operator,
    quantum_phase_estimation,
    approximate_reflection_operator,
    prepare_stationary_state,
    convergence_diagnostics
)

# Step 1: Define multimodal target (mixture of Gaussians in 2D)
def create_multimodal_target(n_points=50):
    """Create challenging multimodal distribution."""
    x = np.linspace(-5, 5, n_points)
    y = np.linspace(-5, 5, n_points)
    X, Y = np.meshgrid(x, y)
    
    # Three well-separated modes
    mode1 = 0.4 * np.exp(-((X - 2)**2 + (Y - 2)**2) / 0.5)
    mode2 = 0.4 * np.exp(-((X + 2)**2 + (Y - 2)**2) / 0.5)
    mode3 = 0.2 * np.exp(-((X)**2 + (Y + 3)**2) / 0.8)
    
    Z = mode1 + mode2 + mode3
    
    # Flatten to 1D for Markov chain
    target_dist = Z.flatten()
    target_dist /= target_dist.sum()
    
    return target_dist, X, Y, Z

# Create target distribution
n_grid = 30  # 30x30 grid
target_dist, X, Y, Z = create_multimodal_target(n_grid)
n_states = len(target_dist)

print(f"=== Quantum-Enhanced Metropolis-Hastings ===")
print(f"Target: Multimodal distribution on {n_grid}x{n_grid} grid")
print(f"Total states: {n_states}")

# Visualize target
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(label='Probability Density')
plt.title('Target Distribution')
plt.xlabel('X')
plt.ylabel('Y')

# Step 2: Build Metropolis-Hastings chain
print("\nBuilding Metropolis-Hastings chain...")
P = build_metropolis_chain(target_dist, proposal_std=3.0)

# Analyze classical chain
D = discriminant_matrix(P)
spectral_info = spectral_analysis(D)

print(f"Classical chain analysis:")
print(f"  Spectral gap: {spectral_info['spectral_gap']:.6f}")
print(f"  Phase gap: {spectral_info['phase_gap']:.6f}")
print(f"  Classical mixing time: {spectral_info['mixing_time']:.0f}")
print(f"  Condition number: {spectral_info['condition_number']:.1f}")

# Step 3: Quantum enhancement
print("\nConstructing quantum components...")

# Build quantum walk
W = prepare_walk_operator(P, pi=target_dist)
print(f"  Walk operator: {W.num_qubits} qubits")

# Phase estimation for spectral analysis
qpe_results = quantum_phase_estimation(
    W,
    num_ancilla=10,
    initial_state=prepare_stationary_state(target_dist, W.num_qubits//2),
    backend="statevector"
)

# Build reflection operator
R = approximate_reflection_operator(W, num_ancilla=8, phase_threshold=0.05)
print(f"  Reflection operator: {R.num_qubits} qubits, depth {R.depth()}")

# Step 4: Sampling comparison
print("\nSampling performance comparison...")

# Classical sampling (simplified simulation)
n_classical_steps = int(spectral_info['mixing_time'] * 10)
classical_samples = []
current = np.random.choice(n_states, p=target_dist)

for _ in range(n_classical_steps):
    current = np.random.choice(n_states, p=P[current, :])
    classical_samples.append(current)

# Quantum sampling (theoretical performance)
quantum_steps = int(np.sqrt(spectral_info['mixing_time']))
quantum_speedup = n_classical_steps / quantum_steps

print(f"\nResults:")
print(f"  Classical steps needed: {n_classical_steps}")
print(f"  Quantum steps needed: {quantum_steps}")
print(f"  Speedup factor: {quantum_speedup:.1f}x")

# Step 5: Visualize results
plt.subplot(1, 2, 2)

# Convert samples to 2D coordinates
classical_coords = [(s % n_grid, s // n_grid) for s in classical_samples[::10]]
x_coords, y_coords = zip(*classical_coords)

plt.scatter(x_coords, y_coords, alpha=0.5, s=10, c='red', label='Classical samples')
plt.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
plt.title(f'Classical Sampling (speedup potential: {quantum_speedup:.1f}x)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.tight_layout()
plt.show()

# Additional analysis
print("\n=== Detailed Performance Analysis ===")
print(f"Bottleneck identification:")
print(f"  Effective dimension: {spectral_info['effective_dimension']}")
print(f"  Number of modes: 3")
print(f"  Mode separation: High (requires global moves)")

# Resource requirements
print(f"\nResource requirements:")
print(f"  Classical: {n_classical_steps} matrix-vector products")
print(f"  Quantum: {quantum_steps} quantum circuits × {R.depth()} gates")
print(f"  Break-even at: ~{R.depth()} classical steps per quantum circuit")

# Convergence diagnostics
if len(classical_samples) > 1000:
    # Split into 4 chains for diagnostics
    chain_length = len(classical_samples) // 4
    chains = [classical_samples[i*chain_length:(i+1)*chain_length] 
              for i in range(4)]
    
    diagnostics = convergence_diagnostics(chains, target_dist)
    print(f"\nConvergence diagnostics:")
    print(f"  Gelman-Rubin R-hat: {diagnostics['gelman_rubin']:.3f}")
    print(f"  Average ESS: {np.mean(diagnostics['effective_sample_sizes']):.1f}")
```

### Expected Output

1. **Multimodal Target**: Visualization showing three well-separated modes
2. **Small Spectral Gap**: Due to bottlenecks between modes (typically 10{³ to 10{t)
3. **Large Quantum Speedup**: Quadratic improvement (10x-100x for challenging problems)
4. **Sampling Visualization**: Classical samples showing slow mode exploration
5. **Resource Analysis**: Break-even point for quantum advantage

---

## Additional Resources

### Code Repository Structure

```
quantum-mcmc/
   examples/
      simple_2state_mcmc.py
      imhk_lattice_gaussian.py
      benchmark_results.ipynb
   notebooks/
      tutorial_quantum_mcmc.ipynb
      qpe_walk_demo.ipynb
      spectral_analysis.ipynb
   tests/
       test_markov_chain.py
       test_quantum_walk.py
       integration/
```

### Performance Considerations

1. **Classical Baseline**: Always establish classical performance first
2. **Quantum Resources**: Consider circuit depth, qubit count, and gate fidelity
3. **Problem Structure**: Quantum advantage is greatest for slowly mixing chains
4. **Precision Requirements**: Balance between ancilla qubits and accuracy needs

### Troubleshooting

1. **Numerical Precision**: Use `tol` parameters for finite precision effects
2. **Memory Usage**: Large state spaces may require sparse matrix representations
3. **Convergence**: Check R-hat < 1.1 and ESS for reliable sampling
4. **Quantum Simulation**: Use `backend="statevector"` for exact results, `"qiskit"` for realistic noise