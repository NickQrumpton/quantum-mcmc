"""
Visualization routines for quantum MCMC results.

This module provides plotting functions for visualizing quantum phase estimation results,
spectral properties, and convergence diagnostics.

Author: Nicholas Zhao
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from scipy import stats

# Set default plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_phase_histogram(
    phases: np.ndarray,
    counts: np.ndarray,
    theoretical_phases: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot histogram of measured phases from quantum phase estimation.
    
    Creates a publication-quality histogram showing the distribution of measured
    phases from QPE, optionally overlaying theoretical phase values for comparison.
    
    Parameters
    ----------
    phases : np.ndarray
        Array of measured phase values in [0, 1).
    counts : np.ndarray
        Count or probability for each phase value.
    theoretical_phases : np.ndarray, optional
        Array of theoretical phase values to mark on plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    **kwargs : dict
        Additional keyword arguments passed to plt.bar().
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plot.
    ax : matplotlib.axes.Axes
        Axes object containing the plot.
    
    Examples
    --------
    >>> phases = np.array([0.0, 0.25, 0.5, 0.75])
    >>> counts = np.array([100, 50, 150, 75])
    >>> fig, ax = plot_phase_histogram(phases, counts)
    
    Notes
    -----
    The phases are expected to be in the interval [0, 1), representing
    eigenphases ¸ where the eigenvalues are e^(2Ài¸).
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.get_figure()
    
    # Normalize counts to probabilities if needed
    if np.sum(counts) > 1.1:  # Assume counts if sum > 1.1
        probabilities = counts / np.sum(counts)
    else:
        probabilities = counts
    
    # Create histogram
    default_kwargs = {
        'alpha': 0.7,
        'edgecolor': 'black',
        'linewidth': 1.2
    }
    default_kwargs.update(kwargs)
    
    # Determine appropriate bar width
    if len(phases) > 1:
        width = 0.8 * min(np.diff(np.sort(phases)))
    else:
        width = 0.02
    
    bars = ax.bar(phases, probabilities, width=width, **default_kwargs)
    
    # Mark theoretical phases if provided
    if theoretical_phases is not None:
        for phase in theoretical_phases:
            ax.axvline(phase, color='red', linestyle='--', linewidth=2,
                      alpha=0.8, label='Theoretical' if phase == theoretical_phases[0] else "")
    
    # Formatting
    ax.set_xlabel('Phase ¸', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Quantum Phase Estimation Results', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.1 * np.max(probabilities))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend if theoretical phases were provided
    if theoretical_phases is not None:
        ax.legend(loc='best', fontsize=10)
    
    # Add text box with statistics
    n_measurements = int(np.sum(counts)) if np.sum(counts) > 1.1 else len(phases)
    textstr = f'Measurements: {n_measurements}\nPeaks: {len(phases)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig, ax


def plot_singular_values(
    singular_values: np.ndarray,
    matrix_name: str = "Discriminant Matrix",
    ax: Optional[Axes] = None,
    log_scale: bool = True,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot singular value spectrum of a matrix.
    
    Creates a publication-quality plot of singular values, useful for analyzing
    the spectral properties of discriminant matrices and understanding the
    quantum walk's behavior.
    
    Parameters
    ----------
    singular_values : np.ndarray
        Array of singular values in descending order.
    matrix_name : str, optional
        Name of the matrix for plot title. Default is "Discriminant Matrix".
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    log_scale : bool, optional
        Whether to use log scale for y-axis. Default is True.
    **kwargs : dict
        Additional keyword arguments passed to plt.plot().
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plot.
    ax : matplotlib.axes.Axes
        Axes object containing the plot.
    
    Examples
    --------
    >>> D = np.random.rand(10, 10)
    >>> _, s, _ = np.linalg.svd(D)
    >>> fig, ax = plot_singular_values(s)
    
    Notes
    -----
    The spectral gap (difference between largest and second-largest singular values)
    is automatically highlighted and annotated on the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.get_figure()
    
    # Ensure singular values are sorted in descending order
    singular_values = np.sort(singular_values)[::-1]
    n_values = len(singular_values)
    indices = np.arange(n_values)
    
    # Default plot parameters
    default_kwargs = {
        'marker': 'o',
        'markersize': 8,
        'linewidth': 2,
        'alpha': 0.8
    }
    default_kwargs.update(kwargs)
    
    # Plot singular values
    ax.plot(indices, singular_values, **default_kwargs, label='Singular values')
    
    # Highlight spectral gap if exists
    if n_values >= 2:
        gap = singular_values[0] - singular_values[1]
        # Draw shaded region for spectral gap
        ax.fill_between([0, 1], [singular_values[1], singular_values[1]], 
                       [singular_values[0], singular_values[0]], 
                       alpha=0.3, color='red', label=f'Spectral gap: {gap:.4f}')
        
        # Add annotation for gap
        ax.annotate('', xy=(0.5, singular_values[1]), xytext=(0.5, singular_values[0]),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.text(0.5, (singular_values[0] + singular_values[1])/2, f'” = {gap:.4f}',
               ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', 
                                                  facecolor='white', alpha=0.8))
    
    # Formatting
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Singular Value', fontsize=12)
    ax.set_title(f'Singular Value Spectrum of {matrix_name}', fontsize=14, fontweight='bold')
    
    if log_scale and np.all(singular_values > 0):
        ax.set_yscale('log')
        ax.set_ylabel('Singular Value (log scale)', fontsize=12)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Add statistics text box
    condition_number = singular_values[0] / singular_values[-1] if singular_values[-1] > 0 else np.inf
    rank = np.sum(singular_values > 1e-10)
    textstr = f'Rank: {rank}\nCondition #: {condition_number:.2e}\nMax: {singular_values[0]:.4f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    return fig, ax


def plot_total_variation(
    iterations: np.ndarray,
    tv_distances: np.ndarray,
    theoretical_bound: Optional[float] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot total variation distance over iterations.
    
    Creates a publication-quality convergence plot showing how the total variation
    distance between the current distribution and target distribution decreases
    over iterations.
    
    Parameters
    ----------
    iterations : np.ndarray
        Array of iteration numbers or time steps.
    tv_distances : np.ndarray
        Total variation distances at each iteration.
    theoretical_bound : float, optional
        Theoretical convergence bound to plot as reference.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    **kwargs : dict
        Additional keyword arguments passed to plt.plot().
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plot.
    ax : matplotlib.axes.Axes
        Axes object containing the plot.
    
    Examples
    --------
    >>> iters = np.arange(100)
    >>> tv_dist = 0.5 * np.exp(-0.1 * iters) + 0.01 * np.random.randn(100)
    >>> fig, ax = plot_total_variation(iters, tv_dist)
    
    Notes
    -----
    The total variation distance is defined as:
    TV(p, q) = 0.5 * £_i |p_i - q_i|
    
    A mixing time indicator is automatically added when the distance falls
    below common thresholds (0.25, 0.1, 0.01).
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.get_figure()
    
    # Default plot parameters
    default_kwargs = {
        'linewidth': 2.5,
        'alpha': 0.9,
        'color': 'blue'
    }
    default_kwargs.update(kwargs)
    
    # Main convergence plot
    ax.plot(iterations, tv_distances, **default_kwargs, label='TV Distance')
    
    # Add theoretical bound if provided
    if theoretical_bound is not None:
        ax.axhline(theoretical_bound, color='red', linestyle='--', linewidth=2,
                  alpha=0.7, label=f'Theoretical bound: {theoretical_bound:.4f}')
    
    # Add common mixing time thresholds
    thresholds = [0.25, 0.1, 0.01]
    threshold_colors = ['orange', 'green', 'purple']
    for thresh, color in zip(thresholds, threshold_colors):
        ax.axhline(thresh, color=color, linestyle=':', linewidth=1.5, alpha=0.5)
        
        # Find mixing time for this threshold
        mixing_idx = np.where(tv_distances <= thresh)[0]
        if len(mixing_idx) > 0:
            mixing_time = iterations[mixing_idx[0]]
            ax.plot(mixing_time, thresh, 'o', color=color, markersize=10)
            ax.text(mixing_time, thresh, f'  Ä_{thresh} = {mixing_time}', 
                   fontsize=9, va='center', color=color)
    
    # Formatting
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Total Variation Distance', fontsize=12)
    ax.set_title('Convergence to Stationary Distribution', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=10)
    
    # Set reasonable y-limits
    ax.set_ylim(max(1e-6, np.min(tv_distances) * 0.5), 1.2)
    ax.set_xlim(iterations[0], iterations[-1])
    
    # Add convergence rate estimate if possible
    if len(tv_distances) > 10:
        # Fit exponential decay to estimate rate
        log_tv = np.log(tv_distances[tv_distances > 0])
        valid_iters = iterations[:len(log_tv)]
        if len(valid_iters) > 5:
            try:
                slope, intercept, r_value, _, _ = stats.linregress(valid_iters[:len(log_tv)//2], 
                                                                   log_tv[:len(log_tv)//2])
                rate = -slope
                textstr = f'Convergence rate: {rate:.4f}\nR² = {r_value**2:.3f}'
                props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.7)
                ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)
            except:
                pass
    
    plt.tight_layout()
    return fig, ax


def plot_eigenvalue_spectrum(
    eigenvalues: np.ndarray,
    operator_name: str = "Quantum Walk Operator",
    ax: Optional[Axes] = None,
    highlight_unit: bool = True,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot eigenvalue spectrum on the complex plane.
    
    Visualizes the eigenvalues of quantum walk operators or other unitary matrices
    on the complex unit circle.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Complex eigenvalues to plot.
    operator_name : str, optional
        Name of the operator for plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    highlight_unit : bool, optional
        Whether to highlight eigenvalues close to 1. Default is True.
    **kwargs : dict
        Additional keyword arguments passed to plt.scatter().
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plot.
    ax : matplotlib.axes.Axes
        Axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.get_figure()
    
    # Extract real and imaginary parts
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)
    
    # Default scatter parameters
    default_kwargs = {
        's': 100,
        'alpha': 0.7,
        'edgecolors': 'black',
        'linewidth': 1
    }
    default_kwargs.update(kwargs)
    
    # Color eigenvalues by phase
    phases = np.angle(eigenvalues)
    scatter = ax.scatter(real_parts, imag_parts, c=phases, cmap='hsv',
                        vmin=-np.pi, vmax=np.pi, **default_kwargs)
    
    # Highlight eigenvalues close to 1 if requested
    if highlight_unit:
        unit_threshold = 0.01
        unit_mask = np.abs(eigenvalues - 1) < unit_threshold
        if np.any(unit_mask):
            ax.scatter(real_parts[unit_mask], imag_parts[unit_mask],
                      s=200, marker='*', color='red', edgecolors='darkred',
                      linewidth=2, label=f'Near unit (|»-1| < {unit_threshold})')
    
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, alpha=0.5)
    
    # Draw axes
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Real Part', fontsize=12)
    ax.set_ylabel('Imaginary Part', fontsize=12)
    ax.set_title(f'Eigenvalue Spectrum of {operator_name}', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar for phases
    cbar = plt.colorbar(scatter, ax=ax, label='Phase (radians)')
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels(['-À', '-À/2', '0', 'À/2', 'À'])
    
    if highlight_unit and np.any(unit_mask):
        ax.legend(loc='best', fontsize=10)
    
    # Add eigenvalue statistics
    spectral_radius = np.max(np.abs(eigenvalues))
    n_unit = np.sum(np.abs(np.abs(eigenvalues) - 1) < 1e-10)
    textstr = f'Spectral radius: {spectral_radius:.4f}\nUnit eigenvalues: {n_unit}'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig, ax


def plot_state_fidelity(
    times: np.ndarray,
    fidelities: np.ndarray,
    target_state_name: str = "Target State",
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot quantum state fidelity over time.
    
    Visualizes how closely a quantum state matches a target state during evolution.
    
    Parameters
    ----------
    times : np.ndarray
        Time points or iteration numbers.
    fidelities : np.ndarray
        Fidelity values at each time point (between 0 and 1).
    target_state_name : str, optional
        Name of the target state for plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    **kwargs : dict
        Additional keyword arguments passed to plt.plot().
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plot.
    ax : matplotlib.axes.Axes
        Axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.get_figure()
    
    # Default plot parameters
    default_kwargs = {
        'linewidth': 2.5,
        'alpha': 0.9,
        'color': 'darkgreen'
    }
    default_kwargs.update(kwargs)
    
    # Plot fidelity
    ax.plot(times, fidelities, **default_kwargs, label='Fidelity')
    
    # Add reference lines
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Perfect fidelity')
    ax.axhline(0.99, color='red', linestyle=':', alpha=0.5, label='99% threshold')
    
    # Shade high-fidelity region
    ax.fill_between(times, 0.99, 1.0, alpha=0.2, color='green')
    
    # Find first time reaching high fidelity
    high_fidelity_idx = np.where(fidelities >= 0.99)[0]
    if len(high_fidelity_idx) > 0:
        first_time = times[high_fidelity_idx[0]]
        ax.plot(first_time, 0.99, 'ro', markersize=10)
        ax.annotate(f't = {first_time:.2f}', xy=(first_time, 0.99),
                   xytext=(first_time + 0.1 * (times[-1] - times[0]), 0.985),
                   arrowprops=dict(arrowstyle='->', color='red'))
    
    # Formatting
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Fidelity', fontsize=12)
    ax.set_title(f'Quantum State Fidelity with {target_state_name}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(times[0], times[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Add average fidelity
    avg_fidelity = np.mean(fidelities)
    textstr = f'Average fidelity: {avg_fidelity:.4f}\nFinal fidelity: {fidelities[-1]:.4f}'
    props = dict(boxstyle='round', facecolor='lightcyan', alpha=0.7)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    return fig, ax


def plot_markov_chain_graph(
    transition_matrix: np.ndarray,
    state_labels: Optional[List[str]] = None,
    stationary_dist: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    threshold: float = 0.01,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Visualize Markov chain as a directed graph.
    
    Creates a graph visualization of the Markov chain structure with nodes
    representing states and edges representing transition probabilities.
    
    Parameters
    ----------
    transition_matrix : np.ndarray
        Row-stochastic transition matrix.
    state_labels : List[str], optional
        Labels for each state. If None, uses indices.
    stationary_dist : np.ndarray, optional
        Stationary distribution to size nodes.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    threshold : float, optional
        Minimum transition probability to display edge. Default is 0.01.
    **kwargs : dict
        Additional keyword arguments for graph layout.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plot.
    ax : matplotlib.axes.Axes
        Axes object containing the plot.
    
    Notes
    -----
    Requires networkx for graph visualization. Falls back to matrix heatmap
    if networkx is not available.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.get_figure()
    
    n_states = transition_matrix.shape[0]
    
    try:
        import networkx as nx
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        if state_labels is None:
            state_labels = [str(i) for i in range(n_states)]
        G.add_nodes_from(range(n_states))
        
        # Add edges with weights
        for i in range(n_states):
            for j in range(n_states):
                if transition_matrix[i, j] > threshold:
                    G.add_edge(i, j, weight=transition_matrix[i, j])
        
        # Layout
        pos = nx.spring_layout(G, k=2/np.sqrt(n_states), iterations=50)
        
        # Node sizes based on stationary distribution
        if stationary_dist is not None:
            node_sizes = 3000 * stationary_dist
        else:
            node_sizes = 1000
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue',
                              edgecolors='black', linewidths=2, ax=ax)
        
        # Draw edges with varying widths
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[5*w for w in weights],
                              alpha=0.6, edge_color='gray', arrows=True,
                              arrowsize=20, arrowstyle='->', ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, {i: state_labels[i] for i in range(n_states)},
                               font_size=12, font_weight='bold', ax=ax)
        
        # Draw edge labels for significant transitions
        edge_labels = {}
        for i, j in edges:
            if G[i][j]['weight'] > 0.1:  # Only label significant edges
                edge_labels[(i, j)] = f"{G[i][j]['weight']:.2f}"
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9, ax=ax)
        
        ax.set_title('Markov Chain Structure', fontsize=14, fontweight='bold')
        ax.axis('off')
        
    except ImportError:
        # Fallback to heatmap if networkx not available
        im = ax.imshow(transition_matrix, cmap='YlOrRd', aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Transition Probability')
        
        # Labels
        if state_labels is not None:
            ax.set_xticks(range(n_states))
            ax.set_yticks(range(n_states))
            ax.set_xticklabels(state_labels)
            ax.set_yticklabels(state_labels)
        
        ax.set_xlabel('To State', fontsize=12)
        ax.set_ylabel('From State', fontsize=12)
        ax.set_title('Markov Chain Transition Matrix', fontsize=14, fontweight='bold')
        
        # Add text annotations for values
        for i in range(n_states):
            for j in range(n_states):
                if transition_matrix[i, j] > threshold:
                    ax.text(j, i, f'{transition_matrix[i, j]:.2f}',
                           ha='center', va='center', color='black' if transition_matrix[i, j] < 0.5 else 'white')
    
    plt.tight_layout()
    return fig, ax


def create_multi_panel_figure(
    plot_functions: List[callable],
    plot_args: List[Dict[str, Any]],
    figsize: Tuple[float, float] = (15, 10),
    layout: Optional[Tuple[int, int]] = None
) -> Figure:
    """
    Create a multi-panel figure combining multiple plots.
    
    Utility function for creating publication-quality figures with multiple
    subplots arranged in a grid.
    
    Parameters
    ----------
    plot_functions : List[callable]
        List of plotting functions to call.
    plot_args : List[Dict[str, Any]]
        List of argument dictionaries for each plotting function.
    figsize : Tuple[float, float], optional
        Figure size in inches. Default is (15, 10).
    layout : Tuple[int, int], optional
        Grid layout (rows, cols). If None, automatically determined.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing all subplots.
    
    Examples
    --------
    >>> plot_funcs = [plot_phase_histogram, plot_singular_values]
    >>> plot_args = [
    ...     {'phases': phases, 'counts': counts},
    ...     {'singular_values': s_values}
    ... ]
    >>> fig = create_multi_panel_figure(plot_funcs, plot_args)
    """
    n_plots = len(plot_functions)
    
    # Determine layout if not provided
    if layout is None:
        cols = int(np.ceil(np.sqrt(n_plots)))
        rows = int(np.ceil(n_plots / cols))
    else:
        rows, cols = layout
    
    # Create figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Create each subplot
    for i, (func, args) in enumerate(zip(plot_functions, plot_args)):
        args['ax'] = axes[i]
        func(**args)
    
    # Hide extra axes
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def save_figure(
    fig: Figure,
    filename: str,
    dpi: int = 300,
    formats: List[str] = ['png', 'pdf']
) -> None:
    """
    Save figure in multiple formats for publication.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    filename : str
        Base filename without extension.
    dpi : int, optional
        Resolution for raster formats. Default is 300.
    formats : List[str], optional
        List of formats to save. Default is ['png', 'pdf'].
    """
    for fmt in formats:
        full_filename = f"{filename}.{fmt}"
        fig.savefig(full_filename, format=fmt, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to {full_filename}")