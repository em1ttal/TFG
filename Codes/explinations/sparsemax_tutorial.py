#!/usr/bin/env python3
"""
================================================================================
SPARSEMAX TUTORIAL: Understanding Sparse Attention Mechanisms
================================================================================

This script provides an interactive, educational exploration of sparsemax,
the sparse attention mechanism used in TabNet and other modern architectures.

Topics covered:
1. Softmax review and its limitations
2. Sparsemax algorithm step-by-step
3. Visual comparisons
4. Entmax generalization (α parameter)
5. Practical implications for feature selection
6. Connection to TabNet and machine unlearning

Author: Educational Tutorial for Machine Unlearning Thesis
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up nice plotting
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


# =============================================================================
# SECTION 1: SOFTMAX REVIEW
# =============================================================================

def softmax(z: np.ndarray) -> np.ndarray:
    """
    Standard softmax function.
    
    Formula: softmax(z_i) = exp(z_i) / Σ exp(z_j)
    
    Properties:
    - Output is always in (0, 1) - never exactly 0 or 1
    - All outputs are positive
    - Outputs sum to 1
    
    Args:
        z: Input scores (logits)
    
    Returns:
        Probability distribution (always dense, no zeros)
    """
    # Subtract max for numerical stability (doesn't change result)
    z_stable = z - np.max(z)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z)


def demonstrate_softmax():
    """Show softmax behavior with various inputs."""
    
    print("=" * 70)
    print("SECTION 1: SOFTMAX REVIEW")
    print("=" * 70)
    print("\nSoftmax converts scores to probabilities, but NEVER outputs zero.\n")
    
    # Example 1: Clear winner
    z1 = np.array([5.0, 1.0, 0.5, 0.1])
    p1 = softmax(z1)
    
    print("Example 1: Clear winner (score=5.0 vs others ~1)")
    print(f"  Input scores:  {z1}")
    print(f"  Softmax output: {np.round(p1, 4)}")
    print(f"  Minimum value:  {p1.min():.6f} (never zero!)")
    print()
    
    # Example 2: One negative score
    z2 = np.array([2.0, 1.0, -5.0, 0.5])
    p2 = softmax(z2)
    
    print("Example 2: One very negative score (-5.0)")
    print(f"  Input scores:  {z2}")
    print(f"  Softmax output: {np.round(p2, 4)}")
    print(f"  Score -5.0 gets: {p2[2]:.6f} (tiny but NOT zero!)")
    print()
    
    # Example 3: Extreme difference
    z3 = np.array([100.0, 1.0, 1.0, 1.0])
    p3 = softmax(z3)
    
    print("Example 3: Extreme difference (100 vs 1)")
    print(f"  Input scores:  {z3}")
    print(f"  Softmax output: {np.round(p3, 6)}")
    print(f"  Even with score=100, others get: {p3[1]:.2e} (still not zero!)")
    print()
    
    print("KEY INSIGHT: Softmax can make values very small, but NEVER exactly 0.")
    print("             This is problematic for feature SELECTION.\n")
    
    return z1, p1


# =============================================================================
# SECTION 2: SPARSEMAX ALGORITHM
# =============================================================================

def sparsemax(z: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Sparsemax: Projects input onto the probability simplex.
    
    Unlike softmax, sparsemax can output EXACT zeros, enabling true sparsity.
    
    Mathematical definition:
        sparsemax(z) = argmin_{p ∈ Δ} ||p - z||²
        
    Where Δ is the probability simplex: {p : p_i ≥ 0, Σp_i = 1}
    
    In words: "Find the valid probability distribution closest to z"
    
    Args:
        z: Input scores
        verbose: If True, print step-by-step computation
    
    Returns:
        Sparse probability distribution (can contain exact zeros)
    """
    n = len(z)
    
    # Step 1: Sort in descending order
    z_sorted = np.sort(z)[::-1]
    
    if verbose:
        print("\n  SPARSEMAX ALGORITHM STEP-BY-STEP:")
        print("  " + "-" * 50)
        print(f"  Input z = {z}")
        print(f"  Step 1: Sort descending → {z_sorted}")
    
    # Step 2: Compute cumulative sums
    cumsum = np.cumsum(z_sorted)
    
    if verbose:
        print(f"  Step 2: Cumulative sums → {cumsum}")
    
    # Step 3: Find k (number of non-zero elements in output)
    # k = max{j : z_sorted[j] > (cumsum[j] - 1) / j}
    k_array = np.arange(1, n + 1)
    threshold_candidates = (cumsum - 1) / k_array
    
    if verbose:
        print(f"  Step 3: Find support size k")
        print(f"          For each k, threshold τ_k = (cumsum - 1) / k")
        for j in range(n):
            comparison = ">" if z_sorted[j] > threshold_candidates[j] else "≤"
            print(f"          k={j+1}: z[{j}]={z_sorted[j]:.3f} {comparison} τ={threshold_candidates[j]:.3f}")
    
    # Find the largest k where condition holds
    support = z_sorted > threshold_candidates
    k = np.sum(support)
    
    if verbose:
        print(f"          → k = {k} (number of non-zero outputs)")
    
    # Step 4: Compute threshold τ
    tau = (cumsum[k - 1] - 1) / k
    
    if verbose:
        print(f"  Step 4: Compute threshold τ = (cumsum[{k-1}] - 1) / {k}")
        print(f"          τ = ({cumsum[k-1]:.3f} - 1) / {k} = {tau:.4f}")
    
    # Step 5: Apply soft thresholding (ReLU-like)
    # p_i = max(z_i - τ, 0)
    p = np.maximum(z - tau, 0)
    
    if verbose:
        print(f"  Step 5: Apply threshold: p_i = max(z_i - τ, 0)")
        for i in range(n):
            print(f"          p[{i}] = max({z[i]:.3f} - {tau:.4f}, 0) = {p[i]:.4f}")
        print(f"\n  Output: {np.round(p, 4)}")
        print(f"  Sum: {np.sum(p):.4f} (should be 1.0)")
        print(f"  Non-zeros: {np.sum(p > 0)} out of {n}")
    
    return p


def sparsemax_step_by_step_demo():
    """Demonstrate sparsemax with detailed step-by-step output."""
    
    print("\n" + "=" * 70)
    print("SECTION 2: SPARSEMAX ALGORITHM - STEP BY STEP")
    print("=" * 70)
    
    print("\nSparsemax finds the CLOSEST valid probability distribution to the input.")
    print("If a score is too low, the closest valid probability is exactly 0.\n")
    
    # Example with clear sparsity
    z = np.array([2.0, 1.5, 0.5, 0.1])
    print("Example: z = [2.0, 1.5, 0.5, 0.1]")
    p = sparsemax(z, verbose=True)
    
    print("\n" + "-" * 70)
    print("COMPARISON:")
    print("-" * 70)
    p_soft = softmax(z)
    print(f"  Softmax:   {np.round(p_soft, 4)} ← All positive!")
    print(f"  Sparsemax: {np.round(p, 4)} ← Has exact zeros!")
    
    return z, p


# =============================================================================
# SECTION 3: VISUAL COMPARISONS
# =============================================================================

def visualize_softmax_vs_sparsemax():
    """Create visual comparison plots."""
    
    print("\n" + "=" * 70)
    print("SECTION 3: VISUAL COMPARISON")
    print("=" * 70)
    print("\nGenerating comparison plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Test cases with increasing sparsity needs
    test_cases = [
        np.array([2.0, 1.8, 1.6, 1.4, 1.2]),  # Similar scores
        np.array([3.0, 1.5, 0.5, 0.3, 0.1]),  # Mixed
        np.array([5.0, 1.0, 0.1, 0.0, -1.0]), # Clear winner
    ]
    titles = ["Similar Scores", "Mixed Scores", "Clear Winner"]
    
    for col, (z, title) in enumerate(zip(test_cases, titles)):
        p_soft = softmax(z)
        p_sparse = sparsemax(z)
        
        x = np.arange(len(z))
        width = 0.35
        
        # Top row: Bar comparison
        ax = axes[0, col]
        bars1 = ax.bar(x - width/2, p_soft, width, label='Softmax', 
                       color='#3498DB', alpha=0.8)
        bars2 = ax.bar(x + width/2, p_sparse, width, label='Sparsemax',
                       color='#E74C3C', alpha=0.8)
        
        ax.set_xlabel('Element Index')
        ax.set_ylabel('Probability')
        ax.set_title(f'{title}\nInput: {z}', fontweight='bold')
        ax.set_xticks(x)
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 0.02:
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', 
                           fontsize=8, color='#3498DB')
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0.02:
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center',
                           fontsize=8, color='#E74C3C')
            elif height == 0:
                ax.annotate('0', xy=(bar.get_x() + bar.get_width()/2, 0.02),
                           ha='center', fontsize=8, color='#E74C3C', fontweight='bold')
        
        # Bottom row: Sparsity visualization
        ax = axes[1, col]
        
        # Create heatmap-style visualization
        data = np.vstack([p_soft, p_sparse])
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Softmax', 'Sparsemax'])
        ax.set_xticks(x)
        ax.set_xticklabels([f'z={z[i]:.1f}' for i in range(len(z))])
        ax.set_title('Probability Heatmap', fontweight='bold')
        
        # Add text annotations
        for i in range(2):
            for j in range(len(z)):
                val = data[i, j]
                color = 'white' if val > 0.5 else 'black'
                text = f'{val:.2f}' if val > 0 else '0'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)
        
        # Mark zeros with X
        for j in range(len(z)):
            if p_sparse[j] == 0:
                ax.plot(j, 1, 'wx', markersize=15, markeredgewidth=3)
    
    plt.tight_layout()
    plt.savefig('Codes/explinations/sparsemax_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Plot saved to: sparsemax_comparison.png")
    print("\nOBSERVATIONS:")
    print("  • Softmax (blue): All bars are positive, weight is spread")
    print("  • Sparsemax (red): Some bars are exactly 0 (marked with X)")
    print("  • As winner becomes clearer, sparsemax becomes more sparse")


def visualize_threshold_mechanism():
    """Visualize how the threshold τ determines sparsity."""
    
    print("\n" + "=" * 70)
    print("SECTION 3B: THRESHOLD MECHANISM VISUALIZATION")
    print("=" * 70)
    
    z = np.array([2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5])
    n = len(z)
    
    # Compute sparsemax
    z_sorted = np.sort(z)[::-1]
    cumsum = np.cumsum(z_sorted)
    k_array = np.arange(1, n + 1)
    thresholds = (cumsum - 1) / k_array
    
    # Find actual k
    support = z_sorted > thresholds
    k = np.sum(support)
    tau = (cumsum[k-1] - 1) / k
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Sorted values vs thresholds
    ax = axes[0]
    x = np.arange(1, n + 1)
    ax.plot(x, z_sorted, 'bo-', linewidth=2, markersize=10, label='Sorted z values')
    ax.plot(x, thresholds, 'r^--', linewidth=2, markersize=8, label='Threshold candidates τ_k')
    ax.axhline(y=tau, color='green', linestyle='-', linewidth=2, label=f'Final τ = {tau:.3f}')
    ax.axvline(x=k, color='purple', linestyle=':', linewidth=2, label=f'k = {k}')
    
    # Shade the "survive" region
    ax.fill_between(x[:k], z_sorted[:k], tau, alpha=0.3, color='green', label='Survives (z > τ)')
    
    ax.set_xlabel('Rank (sorted position)')
    ax.set_ylabel('Value')
    ax.set_title('Finding the Threshold τ', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Before and after threshold
    ax = axes[1]
    p_sparse = sparsemax(z)
    
    x = np.arange(len(z))
    width = 0.35
    
    ax.bar(x - width/2, z, width, label='Input z', color='#3498DB', alpha=0.7)
    ax.bar(x + width/2, p_sparse, width, label='Output p', color='#2ECC71', alpha=0.7)
    ax.axhline(y=tau, color='red', linestyle='--', linewidth=2, label=f'Threshold τ = {tau:.3f}')
    
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_title('Threshold Operation: p = max(z - τ, 0)', fontweight='bold')
    ax.legend()
    ax.set_xticks(x)
    
    # Plot 3: The projection interpretation
    ax = axes[2]
    
    # For 2D visualization, use first 2 elements
    z_2d = z[:2]
    p_2d = sparsemax(z_2d)
    
    # Draw simplex (line from (1,0) to (0,1))
    simplex_x = np.linspace(0, 1, 100)
    simplex_y = 1 - simplex_x
    ax.plot(simplex_x, simplex_y, 'b-', linewidth=3, label='Probability simplex Δ')
    ax.fill_between(simplex_x, 0, simplex_y, alpha=0.1, color='blue')
    
    # Plot original point and projection
    ax.plot(z_2d[0], z_2d[1], 'ro', markersize=15, label=f'Input z = {z_2d}')
    ax.plot(p_2d[0], p_2d[1], 'g*', markersize=20, label=f'Sparsemax p = {np.round(p_2d, 2)}')
    
    # Draw projection line
    ax.plot([z_2d[0], p_2d[0]], [z_2d[1], p_2d[1]], 'k--', linewidth=2, label='Euclidean projection')
    
    ax.set_xlabel('p₁')
    ax.set_ylabel('p₂')
    ax.set_title('Sparsemax = Projection onto Simplex', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(-0.5, 3)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Codes/explinations/sparsemax_threshold.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Plot saved to: sparsemax_threshold.png")
    print("\nKEY INSIGHTS:")
    print(f"  • Input: {z}")
    print(f"  • Threshold τ = {tau:.4f}")
    print(f"  • Elements with z_i > τ survive: {np.sum(z > tau)} out of {n}")
    print(f"  • Output: {np.round(p_sparse, 4)}")


# =============================================================================
# SECTION 4: ENTMAX - THE GENERALIZATION
# =============================================================================

def entmax(z: np.ndarray, alpha: float = 1.5, n_iter: int = 50) -> np.ndarray:
    """
    α-entmax: Generalization of softmax (α=1) and sparsemax (α=2).
    
    Uses bisection algorithm for general α.
    
    Args:
        z: Input scores
        alpha: Sparsity parameter (1=softmax, 2=sparsemax)
        n_iter: Number of bisection iterations
    
    Returns:
        Probability distribution with sparsity controlled by α
    """
    if alpha == 1.0:
        return softmax(z)
    elif alpha == 2.0:
        return sparsemax(z)
    
    # General case: bisection algorithm
    z = np.array(z, dtype=np.float64)
    n = len(z)
    
    # Initial bounds for threshold τ
    tau_min = z.min() - 1
    tau_max = z.max()
    
    for _ in range(n_iter):
        tau = (tau_min + tau_max) / 2
        
        # p_i = [(α-1)(z_i - τ)]_+^{1/(α-1)}
        p = np.maximum((alpha - 1) * (z - tau), 0) ** (1 / (alpha - 1))
        
        # Adjust bounds based on sum
        if p.sum() < 1:
            tau_max = tau
        else:
            tau_min = tau
    
    # Final computation
    p = np.maximum((alpha - 1) * (z - tau), 0) ** (1 / (alpha - 1))
    
    # Normalize to ensure sum = 1
    if p.sum() > 0:
        p = p / p.sum()
    
    return p


def demonstrate_entmax():
    """Show how α controls sparsity in entmax."""
    
    print("\n" + "=" * 70)
    print("SECTION 4: ENTMAX - THE α PARAMETER")
    print("=" * 70)
    print("\nEntmax generalizes softmax and sparsemax with parameter α:")
    print("  α = 1.0  →  Softmax (completely dense)")
    print("  α = 1.5  →  Moderate sparsity (TabNet default)")
    print("  α = 2.0  →  Sparsemax (maximum sparsity)")
    print()
    
    z = np.array([2.0, 1.5, 1.0, 0.5, 0.0, -0.5])
    alphas = [1.0, 1.25, 1.5, 1.75, 2.0]
    
    print(f"Input: z = {z}\n")
    print(f"{'α':^6} | {'Output':^45} | {'# Zeros':^8} | {'Max':^8}")
    print("-" * 75)
    
    results = {}
    for alpha in alphas:
        p = entmax(z, alpha)
        n_zeros = np.sum(p < 1e-6)
        results[alpha] = p
        print(f"{alpha:^6.2f} | {np.array2string(np.round(p, 3), separator=', '):^45} | {n_zeros:^8} | {p.max():^8.3f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Bar chart comparison
    ax = axes[0]
    x = np.arange(len(z))
    width = 0.15
    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))
    
    for i, (alpha, color) in enumerate(zip(alphas, colors)):
        offset = (i - len(alphas)/2 + 0.5) * width
        ax.bar(x + offset, results[alpha], width, label=f'α={alpha}', color=color, alpha=0.8)
    
    ax.set_xlabel('Element Index')
    ax.set_ylabel('Probability')
    ax.set_title('Entmax Output for Different α Values', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'z={z[i]:.1f}' for i in range(len(z))])
    ax.legend(title='α value')
    
    # Plot 2: Sparsity vs α
    ax = axes[1]
    
    alpha_range = np.linspace(1.0, 2.5, 30)
    sparsity = []
    max_prob = []
    
    for alpha in alpha_range:
        p = entmax(z, alpha)
        sparsity.append(np.sum(p < 1e-6) / len(z))
        max_prob.append(p.max())
    
    ax.plot(alpha_range, sparsity, 'b-', linewidth=2, label='Sparsity (fraction of zeros)')
    ax.plot(alpha_range, max_prob, 'r--', linewidth=2, label='Maximum probability')
    
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, label='α=1 (Softmax)')
    ax.axvline(x=1.5, color='green', linestyle=':', alpha=0.7, label='α=1.5 (TabNet)')
    ax.axvline(x=2.0, color='orange', linestyle=':', alpha=0.7, label='α=2 (Sparsemax)')
    
    ax.set_xlabel('α (sparsity parameter)')
    ax.set_ylabel('Value')
    ax.set_title('Effect of α on Sparsity', fontweight='bold')
    ax.legend(loc='center right')
    ax.set_xlim(1.0, 2.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Codes/explinations/entmax_alpha.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Plot saved to: entmax_alpha.png")
    print("\nKEY INSIGHT: Higher α → More sparsity → Fewer features selected")
    print("             TabNet uses α≈1.5 for balanced sparsity")


# =============================================================================
# SECTION 5: GRADIENTS AND BACKPROPAGATION
# =============================================================================

def demonstrate_gradients():
    """Show how gradients differ between softmax and sparsemax."""
    
    print("\n" + "=" * 70)
    print("SECTION 5: GRADIENTS - WHY SPARSITY MATTERS FOR LEARNING")
    print("=" * 70)
    
    print("\nThe Jacobian (gradient matrix) determines how errors backpropagate.\n")
    
    z = np.array([2.0, 1.0, 0.3, -0.5])
    
    # Softmax Jacobian: ∂p_i/∂z_j = p_i(δ_ij - p_j)
    p_soft = softmax(z)
    jacobian_soft = np.diag(p_soft) - np.outer(p_soft, p_soft)
    
    # Sparsemax Jacobian: ∂p_i/∂z_j = (1/|S|) if i,j ∈ S, else 0
    # where S is the support (non-zero elements)
    p_sparse = sparsemax(z)
    support = p_sparse > 0
    s = np.sum(support)  # Support size
    
    jacobian_sparse = np.zeros((len(z), len(z)))
    for i in range(len(z)):
        for j in range(len(z)):
            if support[i] and support[j]:
                if i == j:
                    jacobian_sparse[i, j] = 1 - 1/s
                else:
                    jacobian_sparse[i, j] = -1/s
    
    print(f"Input: z = {z}")
    print(f"\nSoftmax output:   {np.round(p_soft, 4)}")
    print(f"Sparsemax output: {np.round(p_sparse, 4)}")
    print(f"\nSparsemax support: indices {np.where(support)[0]} (size {s})")
    
    print("\n" + "-" * 50)
    print("SOFTMAX JACOBIAN (all elements contribute to gradient):")
    print("-" * 50)
    print(np.round(jacobian_soft, 4))
    print(f"Non-zero elements: {np.sum(np.abs(jacobian_soft) > 1e-6)}/{jacobian_soft.size}")
    
    print("\n" + "-" * 50)
    print("SPARSEMAX JACOBIAN (only support elements contribute):")
    print("-" * 50)
    print(np.round(jacobian_sparse, 4))
    print(f"Non-zero elements: {np.sum(np.abs(jacobian_sparse) > 1e-6)}/{jacobian_sparse.size}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot Jacobians as heatmaps
    for ax, jac, title in [(axes[0], jacobian_soft, 'Softmax Jacobian'),
                            (axes[1], jacobian_sparse, 'Sparsemax Jacobian')]:
        im = ax.imshow(jac, cmap='RdBu', vmin=-0.5, vmax=0.5)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('∂/∂z_j')
        ax.set_ylabel('∂p_i')
        
        for i in range(len(z)):
            for j in range(len(z)):
                color = 'white' if abs(jac[i,j]) > 0.25 else 'black'
                ax.text(j, i, f'{jac[i,j]:.2f}', ha='center', va='center', 
                       color=color, fontsize=10)
        
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Plot gradient flow comparison
    ax = axes[2]
    
    # Simulate gradient from output
    grad_output = np.array([1.0, 0.0, 0.0, 0.0])  # Gradient w.r.t first element
    
    grad_z_soft = jacobian_soft.T @ grad_output
    grad_z_sparse = jacobian_sparse.T @ grad_output
    
    x = np.arange(len(z))
    width = 0.35
    ax.bar(x - width/2, np.abs(grad_z_soft), width, label='Softmax', color='#3498DB')
    ax.bar(x + width/2, np.abs(grad_z_sparse), width, label='Sparsemax', color='#E74C3C')
    
    ax.set_xlabel('Input index')
    ax.set_ylabel('|Gradient| magnitude')
    ax.set_title('Gradient Backpropagation\n(from output element 0)', fontweight='bold')
    ax.set_xticks(x)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('Codes/explinations/sparsemax_gradients.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Plot saved to: sparsemax_gradients.png")
    print("\nKEY INSIGHT FOR UNLEARNING:")
    print("  • Softmax: Gradients flow to ALL inputs (dense updates)")
    print("  • Sparsemax: Gradients only flow to SUPPORT elements (sparse updates)")
    print("  • Sparse gradients = more targeted learning = potentially easier unlearning")


# =============================================================================
# SECTION 6: PRACTICAL FEATURE SELECTION SIMULATION
# =============================================================================

def simulate_tabnet_feature_selection():
    """Simulate how TabNet uses sparsemax for feature selection."""
    
    print("\n" + "=" * 70)
    print("SECTION 6: TABNET FEATURE SELECTION SIMULATION")
    print("=" * 70)
    
    print("\nSimulating 3 decision steps with 6 features...")
    print("Watch how the model selects different features at each step.\n")
    
    n_features = 6
    feature_names = ['Age', 'Income', 'Education', 'Hours', 'Occupation', 'Country']
    
    # Simulated attention scores (would be learned in real TabNet)
    attention_scores = [
        np.array([2.5, 1.8, 0.3, 0.1, 1.5, 0.0]),   # Step 1: Focus on Age, Income, Occupation
        np.array([0.5, 2.0, 1.9, 0.2, 0.0, 0.3]),   # Step 2: Focus on Income, Education
        np.array([1.0, 0.3, 0.2, 2.1, 0.8, 1.5]),   # Step 3: Focus on Hours, Country
    ]
    
    # Prior scales (simulate feature reuse penalty)
    gamma = 1.5  # TabNet default
    prior = np.ones(n_features)
    
    all_masks = []
    
    print("=" * 70)
    for step, scores in enumerate(attention_scores):
        # Apply prior scaling
        scaled_scores = scores * prior
        
        # Apply sparsemax
        mask = sparsemax(scaled_scores)
        all_masks.append(mask)
        
        print(f"\nSTEP {step + 1}:")
        print(f"  Raw attention scores:    {np.round(scores, 2)}")
        print(f"  Prior scales:            {np.round(prior, 2)}")
        print(f"  Scaled scores:           {np.round(scaled_scores, 2)}")
        print(f"  Sparsemax mask:          {np.round(mask, 3)}")
        print(f"  Selected features:       {[feature_names[i] for i in np.where(mask > 0.01)[0]]}")
        
        # Update prior for next step (penalize reused features)
        prior = prior * (gamma - mask)
        prior = np.clip(prior, 0.1, 1.0)  # Keep prior bounded
    
    print("\n" + "=" * 70)
    
    # Aggregate feature importance across all steps
    total_importance = np.sum(all_masks, axis=0)
    
    print("\nAGGREGATE FEATURE IMPORTANCE:")
    print("-" * 40)
    for i, (name, imp) in enumerate(sorted(zip(feature_names, total_importance), 
                                           key=lambda x: -x[1])):
        bar = '█' * int(imp * 20)
        print(f"  {name:12s}: {imp:.3f} {bar}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Masks per step
    ax = axes[0]
    mask_matrix = np.array(all_masks)
    im = ax.imshow(mask_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_yticks(range(3))
    ax.set_yticklabels(['Step 1', 'Step 2', 'Step 3'])
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_title('Attention Masks per Step', fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Add annotations
    for i in range(3):
        for j in range(n_features):
            val = mask_matrix[i, j]
            if val > 0.01:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       color='white' if val > 0.5 else 'black', fontsize=9)
    
    # Plot 2: Aggregate importance
    ax = axes[1]
    colors = plt.cm.RdYlGn(total_importance / total_importance.max())
    bars = ax.bar(feature_names, total_importance, color=colors)
    ax.set_ylabel('Total Attention')
    ax.set_title('Aggregate Feature Importance', fontweight='bold')
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    
    # Plot 3: Selection pattern
    ax = axes[2]
    for step in range(3):
        mask = all_masks[step]
        selected = mask > 0.01
        y = [step] * n_features
        sizes = mask * 500 + 10
        ax.scatter(range(n_features), y, s=sizes, c=[f'C{step}']*n_features, 
                  alpha=0.7, label=f'Step {step+1}')
        
        # Mark selected with ring
        for i, sel in enumerate(selected):
            if sel:
                ax.scatter(i, step, s=sizes[i] + 100, facecolors='none', 
                          edgecolors='black', linewidths=2)
    
    ax.set_yticks(range(3))
    ax.set_yticklabels(['Step 1', 'Step 2', 'Step 3'])
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_title('Feature Selection Pattern\n(size = attention weight)', fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('Codes/explinations/tabnet_feature_selection.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Plot saved to: tabnet_feature_selection.png")
    print("\nKEY INSIGHT: Different steps select DIFFERENT features!")
    print("             This is like a decision tree that learns which")
    print("             features to examine at each decision point.")


# =============================================================================
# SECTION 7: IMPLICATIONS FOR UNLEARNING
# =============================================================================

def discuss_unlearning_implications():
    """Discuss how sparsemax affects machine unlearning."""
    
    print("\n" + "=" * 70)
    print("SECTION 7: IMPLICATIONS FOR MACHINE UNLEARNING")
    print("=" * 70)
    
    print("""
    How Sparse Attention Affects Unlearning
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    DENSE ATTENTION (Softmax):
    ┌─────────────────────────────────────────────────────────────────┐
    │  All features → All weights → Prediction                        │
    │                                                                  │
    │  Feature 1 ──┬──→ Weight 1 ──┬──→                               │
    │  Feature 2 ──┼──→ Weight 2 ──┼──→  Prediction                   │
    │  Feature 3 ──┼──→ Weight 3 ──┼──→                               │
    │  Feature 4 ──┴──→ Weight 4 ──┴──→                               │
    │                                                                  │
    │  Problem: Information about "forget class" is spread EVERYWHERE │
    │           Need to inject noise into ALL weights                 │
    │           High risk of catastrophic forgetting                  │
    └─────────────────────────────────────────────────────────────────┘

    SPARSE ATTENTION (Sparsemax/Entmax):
    ┌─────────────────────────────────────────────────────────────────┐
    │  Selected features → Specific weights → Prediction              │
    │                                                                  │
    │  Feature 1 ──────→ Weight 1 ──────→                             │
    │  Feature 2 ──────→ Weight 2 ──────→  Prediction                 │
    │  Feature 3    ✗     (unused)                                    │
    │  Feature 4    ✗     (unused)                                    │
    │                                                                  │
    │  Advantage: Information is LOCALIZED                            │
    │             Can target specific weights for unlearning          │
    │             Lower risk of catastrophic forgetting               │
    └─────────────────────────────────────────────────────────────────┘

    UNLEARNING STRATEGIES FOR TABNET:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    1. ATTENTION-GUIDED NOISE INJECTION:
       - Identify which features the model uses for "forget class"
       - Inject noise primarily into those feature pathways
       - Preserve pathways used by "retain classes"

    2. MASK-BASED UNLEARNING:
       - For forget samples, find their attention masks
       - Add noise only to weights with high mask values
       - More targeted than uniform noise

    3. GRADIENT-BASED WITH ATTENTION:
       - Compute gradients on forget set
       - Weight gradient magnitude by attention scores
       - Parameters with high attention on forget set get more noise

    4. DECISION STEP TARGETING:
       - If a specific step is critical for forget class
       - Focus noise injection on that step's parameters
       - Preserve earlier feature extraction steps
    """)
    
    # Simple simulation
    print("\nSIMPLE DEMONSTRATION:")
    print("-" * 50)
    
    # Simulate model behavior
    np.random.seed(42)
    
    # "Model" with attention patterns for 2 classes
    class_0_attention = np.array([0.7, 0.3, 0.0, 0.0])  # Uses features 0, 1
    class_1_attention = np.array([0.0, 0.2, 0.5, 0.3])  # Uses features 2, 3
    
    print("Class 0 (RETAIN) uses features: 0, 1")
    print(f"  Attention: {class_0_attention}")
    print("\nClass 1 (FORGET) uses features: 2, 3")
    print(f"  Attention: {class_1_attention}")
    
    # If we want to forget Class 1
    print("\nTO UNLEARN CLASS 1:")
    print("  → Inject noise into features 2, 3 pathways")
    print("  → Preserve features 0, 1 pathways")
    print("  → This is TARGETED unlearning!")
    
    # Compute overlap
    overlap = np.sum(class_0_attention * class_1_attention)
    print(f"\nAttention overlap: {overlap:.3f}")
    print(f"  Low overlap = Easier unlearning (less collateral damage)")
    print(f"  High overlap = Harder unlearning (shared pathways)")


# =============================================================================
# SECTION 8: INTERACTIVE EXPERIMENTS
# =============================================================================

def interactive_experiments():
    """Allow user to experiment with different inputs."""
    
    print("\n" + "=" * 70)
    print("SECTION 8: INTERACTIVE EXPERIMENTS")
    print("=" * 70)
    
    test_cases = {
        'uniform': np.array([1.0, 1.0, 1.0, 1.0]),
        'linear': np.array([4.0, 3.0, 2.0, 1.0]),
        'one_hot': np.array([10.0, 0.0, 0.0, 0.0]),
        'negative': np.array([2.0, 1.0, -1.0, -2.0]),
        'close': np.array([1.01, 1.00, 0.99, 0.98]),
        'mixed': np.array([3.0, 0.5, 2.8, 0.1, 2.9]),
    }
    
    print("\nComparing softmax vs sparsemax on various inputs:\n")
    
    for name, z in test_cases.items():
        p_soft = softmax(z)
        p_sparse = sparsemax(z)
        
        n_zeros_sparse = np.sum(p_sparse == 0)
        entropy_soft = -np.sum(p_soft * np.log(p_soft + 1e-10))
        
        print(f"Case: {name}")
        print(f"  Input:      {z}")
        print(f"  Softmax:    {np.round(p_soft, 4)}")
        print(f"  Sparsemax:  {np.round(p_sparse, 4)}")
        print(f"  Zeros:      {n_zeros_sparse}/{len(z)}")
        print(f"  Softmax entropy: {entropy_soft:.3f}")
        print()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all tutorial sections."""
    
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "     SPARSEMAX TUTORIAL: Understanding Sparse Attention     ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # Section 1: Softmax Review
    demonstrate_softmax()
    
    # Section 2: Sparsemax Algorithm
    sparsemax_step_by_step_demo()
    
    # Section 3: Visual Comparisons
    #visualize_softmax_vs_sparsemax()
    #visualize_threshold_mechanism()
    
    # Section 4: Entmax
    demonstrate_entmax()
    
    # Section 5: Gradients
    demonstrate_gradients()
    
    # Section 6: TabNet Simulation
    simulate_tabnet_feature_selection()
    
    # Section 7: Unlearning Implications
    discuss_unlearning_implications()
    
    # Section 8: Interactive Experiments
    interactive_experiments()
    
    print("\n" + "=" * 70)
    print("TUTORIAL COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  • sparsemax_comparison.png")
    print("  • sparsemax_threshold.png")
    print("  • entmax_alpha.png")
    print("  • sparsemax_gradients.png")
    print("  • tabnet_feature_selection.png")
    print("\nKey Takeaways:")
    print("  1. Softmax: Dense output, all elements positive")
    print("  2. Sparsemax: Sparse output, exact zeros possible")
    print("  3. Entmax(α): Interpolates between them (α=1.5 for TabNet)")
    print("  4. Sparsity enables targeted feature selection")
    print("  5. For unlearning: sparse attention = more localized information")
    print("=" * 70)


if __name__ == "__main__":
    main()
