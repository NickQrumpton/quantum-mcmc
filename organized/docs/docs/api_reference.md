# API Reference - Quantum MCMC

## Overview

The quantum-mcmc package provides a comprehensive framework for quantum-enhanced Markov Chain Monte Carlo sampling. This document provides detailed API documentation for all public classes, functions, and modules.

## Table of Contents

1. [Classical Module](#classical-module)
   - [Markov Chain Functions](#markov-chain-functions)
   - [Discriminant Matrix Functions](#discriminant-matrix-functions)
2. [Core Module](#core-module)
   - [Quantum Walk Functions](#quantum-walk-functions)
   - [Phase Estimation Functions](#phase-estimation-functions)
   - [Reflection Operator Functions](#reflection-operator-functions)
3. [Utils Module](#utils-module)
   - [State Preparation Functions](#state-preparation-functions)
   - [Analysis Functions](#analysis-functions)
   - [Visualization Functions](#visualization-functions)

---

## Classical Module

### Markov Chain Functions

#### `build_two_state_chain(p: float, q: Optional[float] = None) -> np.ndarray`

Constructs a 2×2 stochastic matrix for a two-state Markov chain.

**Parameters:**
- `p` (float): Transition probability from state 0 to 1 (0 d p d 1)
- `q` (float, optional): Transition probability from state 1 to 0. If None, uses q = p

**Returns:**
- `np.ndarray`: 2×2 row-stochastic transition matrix

**Example:**
```python
from quantum_mcmc.classical import build_two_state_chain

# Symmetric chain
P = build_two_state_chain(0.3)
# P = [[0.7, 0.3],
#      [0.3, 0.7]]

# Asymmetric chain
P = build_two_state_chain(0.2, 0.8)
# P = [[0.8, 0.2],
#      [0.8, 0.2]]
```

#### `build_metropolis_chain(target_dist: np.ndarray, proposal_matrix: Optional[np.ndarray] = None, proposal_std: float = 1.0, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]`

Constructs a Metropolis-Hastings transition matrix for a given target distribution.

**Parameters:**
- `target_dist` (np.ndarray): Target stationary distribution (normalized probability vector)
- `proposal_matrix` (np.ndarray, optional): Row-stochastic proposal matrix Q
- `proposal_std` (float): Standard deviation for default Gaussian proposal
- `sparse` (bool): If True, returns sparse CSR matrix

**Returns:**
- `Union[np.ndarray, sp.csr_matrix]`: Metropolis-Hastings transition matrix

**Example:**
```python
from quantum_mcmc.classical import build_metropolis_chain
import numpy as np

# Target distribution: Gaussian
x = np.linspace(-3, 3, 50)
target = np.exp(-x**2/2)
target /= target.sum()

P = build_metropolis_chain(target, proposal_std=0.5)
```

#### `is_stochastic(P: np.ndarray, tol: float = 1e-10) -> bool`

Verifies if a matrix is row-stochastic.

**Parameters:**
- `P` (np.ndarray): Matrix to check
- `tol` (float): Absolute tolerance for numerical comparison

**Returns:**
- `bool`: True if P is row-stochastic

#### `stationary_distribution(P: np.ndarray, method: str = 'eigen', max_iter: int = 10000, tol: float = 1e-10) -> np.ndarray`

Computes the stationary distribution of a stochastic matrix.

**Parameters:**
- `P` (np.ndarray): Row-stochastic transition matrix
- `method` (str): 'eigen' for eigenvalue decomposition, 'power' for power iteration
- `max_iter` (int): Maximum iterations for power method
- `tol` (float): Convergence tolerance

**Returns:**
- `np.ndarray`: Stationary distribution À

**Example:**
```python
from quantum_mcmc.classical import stationary_distribution

pi = stationary_distribution(P, method='eigen')
# Verify: pi @ P H pi
```

#### `is_reversible(P: np.ndarray, pi: Optional[np.ndarray] = None, tol: float = 1e-10) -> bool`

Checks if a Markov chain satisfies detailed balance (reversibility).

**Parameters:**
- `P` (np.ndarray): Transition matrix
- `pi` (np.ndarray, optional): Stationary distribution
- `tol` (float): Tolerance for numerical comparison

**Returns:**
- `bool`: True if chain is reversible

#### `sample_random_reversible_chain(n: int, sparsity: float = 0.3, seed: Optional[int] = None, return_sparse: bool = False) -> Tuple[Union[np.ndarray, sp.csr_matrix], np.ndarray]`

Generates a random n-state reversible Markov chain.

**Parameters:**
- `n` (int): Number of states
- `sparsity` (float): Fraction of zero entries (0 to 1)
- `seed` (int, optional): Random seed
- `return_sparse` (bool): If True, returns sparse matrix

**Returns:**
- `Tuple[matrix, distribution]`: Transition matrix P and stationary distribution À

### Discriminant Matrix Functions

#### `discriminant_matrix(P: np.ndarray, pi: Optional[np.ndarray] = None) -> np.ndarray`

Computes the discriminant matrix D(P) for a reversible Markov chain.

**Mathematical Definition:**
```
D[x,y] = sqrt(P[x,y] * P[y,x] * À[y] / À[x])
```

**Parameters:**
- `P` (np.ndarray): n×n reversible transition matrix
- `pi` (np.ndarray, optional): Stationary distribution

**Returns:**
- `np.ndarray`: n×n discriminant matrix

#### `singular_values(D: np.ndarray) -> np.ndarray`

Computes sorted singular values of the discriminant matrix.

**Parameters:**
- `D` (np.ndarray): Discriminant matrix

**Returns:**
- `np.ndarray`: Singular values in descending order

#### `spectral_gap(D: np.ndarray) -> float`

Computes the spectral gap (Ã - Ã‚) of the discriminant matrix.

**Parameters:**
- `D` (np.ndarray): Discriminant matrix

**Returns:**
- `float`: Spectral gap value

#### `phase_gap(D: np.ndarray) -> float`

Computes the phase gap of the quantum walk operator.

**Mathematical Relation:**
```
´ H 2 * arcsin(spectral_gap(D) / 2)
```

**Parameters:**
- `D` (np.ndarray): Discriminant matrix

**Returns:**
- `float`: Phase gap in radians

---

## Core Module

### Quantum Walk Functions

#### `prepare_walk_operator(P: np.ndarray, pi: Optional[np.ndarray] = None, backend: str = "qiskit") -> Union[QuantumCircuit, np.ndarray]`

Constructs the Szegedy quantum walk operator W(P) for a Markov chain.

**Parameters:**
- `P` (np.ndarray): Reversible transition matrix
- `pi` (np.ndarray, optional): Stationary distribution
- `backend` (str): "qiskit" for QuantumCircuit, "matrix" for numpy array

**Returns:**
- `Union[QuantumCircuit, np.ndarray]`: Walk operator

**Example:**
```python
from quantum_mcmc.core import prepare_walk_operator

# As quantum circuit
W_circuit = prepare_walk_operator(P, backend="qiskit")

# As matrix
W_matrix = prepare_walk_operator(P, backend="matrix")
```

#### `walk_eigenvalues(P: np.ndarray, pi: Optional[np.ndarray] = None) -> np.ndarray`

Computes eigenvalues of the quantum walk operator.

**Parameters:**
- `P` (np.ndarray): Transition matrix
- `pi` (np.ndarray, optional): Stationary distribution

**Returns:**
- `np.ndarray`: Complex eigenvalues sorted by magnitude

#### `quantum_mixing_time(P: np.ndarray, epsilon: float = 0.01, pi: Optional[np.ndarray] = None) -> int`

Estimates quantum mixing time for given accuracy.

**Parameters:**
- `P` (np.ndarray): Transition matrix
- `epsilon` (float): Target accuracy
- `pi` (np.ndarray, optional): Stationary distribution

**Returns:**
- `int`: Estimated mixing time in walk steps

### Phase Estimation Functions

#### `quantum_phase_estimation(unitary: Union[QuantumCircuit, ControlledGate], num_ancilla: int, state_prep: Optional[Callable] = None, backend: str = "qiskit", shots: int = 1024, initial_state: Optional[QuantumCircuit] = None) -> Dict[str, any]`

Constructs and executes a quantum phase estimation circuit.

**Parameters:**
- `unitary` (Union[QuantumCircuit, ControlledGate]): Unitary operator
- `num_ancilla` (int): Number of ancilla qubits (precision)
- `state_prep` (Callable, optional): State preparation function
- `backend` (str): "qiskit" or "statevector"
- `shots` (int): Measurement shots
- `initial_state` (QuantumCircuit, optional): Initial state circuit

**Returns:**
- `Dict`: Contains 'counts', 'phases', 'probabilities', 'circuit'

**Example:**
```python
from quantum_mcmc.core import quantum_phase_estimation

results = quantum_phase_estimation(
    W_circuit,
    num_ancilla=8,
    backend="statevector"
)
print(f"Dominant phases: {results['phases'][:3]}")
```

#### `analyze_qpe_results(results: Dict[str, any]) -> Dict[str, any]`

Analyzes QPE output to extract eigenphases with error estimates.

**Parameters:**
- `results` (Dict): Output from quantum_phase_estimation()

**Returns:**
- `Dict`: Contains 'dominant_phases', 'uncertainties', 'confidence_intervals'

### Reflection Operator Functions

#### `approximate_reflection_operator(walk_operator: QuantumCircuit, num_ancilla: int, phase_threshold: float = 0.1, inverse: bool = False) -> QuantumCircuit`

Constructs approximate reflection operator about stationary state.

**Parameters:**
- `walk_operator` (QuantumCircuit): Quantum walk operator
- `num_ancilla` (int): Ancilla qubits for phase estimation
- `phase_threshold` (float): Threshold for identifying stationary eigenstates
- `inverse` (bool): If True, construct inverse reflection

**Returns:**
- `QuantumCircuit`: Reflection operator

**Example:**
```python
from quantum_mcmc.core import approximate_reflection_operator

R = approximate_reflection_operator(
    W_circuit,
    num_ancilla=6,
    phase_threshold=0.05
)
```

---

## Utils Module

### State Preparation Functions

#### `prepare_stationary_state(pi: np.ndarray, num_qubits: int, method: str = "exact", threshold: float = 1e-10) -> QuantumCircuit`

Prepares quantum state encoding stationary distribution.

**Parameters:**
- `pi` (np.ndarray): Probability distribution
- `num_qubits` (int): Number of qubits
- `method` (str): "exact" or "sparse"
- `threshold` (float): Minimum probability for sparse method

**Returns:**
- `QuantumCircuit`: State preparation circuit

#### `prepare_basis_state(index: int, num_qubits: int, little_endian: bool = True) -> QuantumCircuit`

Prepares a computational basis state |xé.

**Parameters:**
- `index` (int): Basis state index
- `num_qubits` (int): Number of qubits
- `little_endian` (bool): Bit ordering convention

**Returns:**
- `QuantumCircuit`: Basis state preparation

#### `prepare_uniform_superposition(num_qubits: int, num_states: Optional[int] = None) -> QuantumCircuit`

Prepares uniform superposition over computational basis states.

**Parameters:**
- `num_qubits` (int): Number of qubits
- `num_states` (int, optional): Number of states in superposition

**Returns:**
- `QuantumCircuit`: Uniform superposition circuit

### Analysis Functions

#### `total_variation_distance(dist1: np.ndarray, dist2: np.ndarray) -> float`

Computes total variation distance between probability distributions.

**Mathematical Definition:**
```
TV(P, Q) = (1/2) * £b |P(i) - Q(i)|
```

**Parameters:**
- `dist1` (np.ndarray): First distribution
- `dist2` (np.ndarray): Second distribution

**Returns:**
- `float`: Total variation distance in [0, 1]

#### `effective_sample_size(samples: np.ndarray, method: str = "autocorrelation") -> float`

Computes effective sample size for MCMC samples.

**Parameters:**
- `samples` (np.ndarray): Array of samples
- `method` (str): "autocorrelation" or "batch_means"

**Returns:**
- `float`: Effective sample size

#### `convergence_diagnostics(chain_samples: List[np.ndarray], target_distribution: Optional[np.ndarray] = None) -> Dict[str, Any]`

Comprehensive convergence diagnostics for MCMC chains.

**Parameters:**
- `chain_samples` (List[np.ndarray]): Multiple chain samples
- `target_distribution` (np.ndarray, optional): Known target

**Returns:**
- `Dict`: Contains 'gelman_rubin', 'geweke_scores', 'effective_sample_sizes'

### Visualization Functions

#### `plot_phase_histogram(phases: np.ndarray, counts: np.ndarray, theoretical_phases: Optional[np.ndarray] = None, ax: Optional[Axes] = None, **kwargs) -> Tuple[Figure, Axes]`

Plots histogram of measured phases from QPE.

**Parameters:**
- `phases` (np.ndarray): Phase values
- `counts` (np.ndarray): Counts or probabilities
- `theoretical_phases` (np.ndarray, optional): Expected phases
- `ax` (Axes, optional): Matplotlib axes

**Returns:**
- `Tuple[Figure, Axes]`: Figure and axes objects

#### `plot_singular_values(singular_values: np.ndarray, matrix_name: str = "Discriminant Matrix", ax: Optional[Axes] = None, log_scale: bool = True, **kwargs) -> Tuple[Figure, Axes]`

Plots singular value spectrum of a matrix.

**Parameters:**
- `singular_values` (np.ndarray): Singular values
- `matrix_name` (str): Name for plot title
- `ax` (Axes, optional): Matplotlib axes
- `log_scale` (bool): Use logarithmic scale

**Returns:**
- `Tuple[Figure, Axes]`: Figure and axes objects