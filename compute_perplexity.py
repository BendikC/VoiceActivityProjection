"""
Compute perplexity and KL divergence between VAP probability distributions.

Usage:
    python compute_perplexity.py -j1 conv1.json -j2 conv2.json
    python compute_perplexity.py -j1 conv1.json -j2 conv2.json --plot
    python compute_perplexity.py -j1 conv1.json -j2 conv2.json --output results.csv --plot perplexity.png
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
import json

from vap.utils import read_json


def load_vap_output(json_path: str):
    """Load VAP model output from JSON file."""
    try:
        data = read_json(json_path)
    except json.JSONDecodeError as e:
        print(f"Warning: Standard JSON loading failed: {e}")
        print("Attempting to extract first valid JSON object...")
        
        with open(json_path, 'r') as f:
            content = f.read()
        
        decoder = json.JSONDecoder()
        data, idx = decoder.raw_decode(content)
        print(f"Successfully extracted JSON object (chars 0-{idx})")
    
    # Convert lists back to tensors
    output = {}
    for key, value in data.items():
        if isinstance(value, list):
            output[key] = torch.tensor(value)
        else:
            output[key] = value
    
    return output


def compute_perplexity(probs: torch.Tensor, epsilon: float = 1e-10):
    """
    Compute perplexity from probability distribution.
    
    Perplexity = 2^(entropy) where entropy = -sum(p * log2(p))
    
    Args:
        probs: Probability distribution (n_frames, n_classes)
        epsilon: Small value to avoid log(0)
        
    Returns:
        torch.Tensor: Perplexity per frame
    """
    # Add epsilon to avoid log(0)
    probs = probs + epsilon
    probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize
    
    # Compute entropy: -sum(p * log2(p))
    entropy = -(probs * torch.log2(probs)).sum(dim=-1)
    
    # Perplexity = 2^entropy
    perplexity = 2 ** entropy
    
    return perplexity


def compute_cross_entropy(p: torch.Tensor, q: torch.Tensor, epsilon: float = 1e-10):
    """
    Compute cross-entropy: H(p, q) = -sum(p * log2(q))
    
    Measures how well distribution q predicts distribution p.
    
    Args:
        p: True distribution (n_frames, n_classes)
        q: Predicted/model distribution (n_frames, n_classes)
        epsilon: Small value to avoid log(0)
        
    Returns:
        torch.Tensor: Cross-entropy per frame
    """
    q = q + epsilon
    q = q / q.sum(dim=-1, keepdim=True)
    
    cross_entropy = -(p * torch.log2(q)).sum(dim=-1)
    return cross_entropy


def compute_kl_divergence(p: torch.Tensor, q: torch.Tensor, epsilon: float = 1e-10):
    """
    Compute KL divergence: KL(p || q) = sum(p * log2(p/q))
    
    Measures how different distribution p is from q.
    KL divergence is asymmetric: KL(p||q) != KL(q||p)
    
    Args:
        p: Reference distribution (n_frames, n_classes)
        q: Comparison distribution (n_frames, n_classes)
        epsilon: Small value to avoid log(0) and division by zero
        
    Returns:
        torch.Tensor: KL divergence per frame
    """
    p = p + epsilon
    q = q + epsilon
    
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    
    kl = (p * torch.log2(p / q)).sum(dim=-1)
    return kl


def compute_js_divergence(p: torch.Tensor, q: torch.Tensor, epsilon: float = 1e-10):
    """
    Compute Jensen-Shannon divergence: JS(p, q) = 0.5 * KL(p||m) + 0.5 * KL(q||m)
    where m = 0.5 * (p + q)
    
    JS divergence is symmetric: JS(p, q) = JS(q, p)
    Range: [0, 1] bits
    
    Args:
        p: First distribution (n_frames, n_classes)
        q: Second distribution (n_frames, n_classes)
        epsilon: Small value to avoid numerical issues
        
    Returns:
        torch.Tensor: JS divergence per frame
    """
    p = p + epsilon
    q = q + epsilon
    
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    
    # Mixture distribution
    m = 0.5 * (p + q)
    
    # JS = 0.5 * KL(p||m) + 0.5 * KL(q||m)
    kl_pm = compute_kl_divergence(p, m, epsilon=0)  # epsilon already added
    kl_qm = compute_kl_divergence(q, m, epsilon=0)
    
    js = 0.5 * kl_pm + 0.5 * kl_qm
    return js


def align_sequences(probs1: torch.Tensor, probs2: torch.Tensor):
    """
    Align two probability sequences to same length by truncating the longer one.
    
    Args:
        probs1: First probability distribution (batch, n_frames1, n_classes)
        probs2: Second probability distribution (batch, n_frames2, n_classes)
        
    Returns:
        tuple: (aligned_probs1, aligned_probs2)
    """
    min_frames = min(probs1.shape[1], probs2.shape[1])
    
    probs1_aligned = probs1[:, :min_frames, :]
    probs2_aligned = probs2[:, :min_frames, :]
    
    return probs1_aligned, probs2_aligned


def plot_perplexity(
    perp1: torch.Tensor,
    perp2: torch.Tensor,
    kl_1_2: torch.Tensor,
    kl_2_1: torch.Tensor,
    js: torch.Tensor,
    frame_hz: int = 50,
    save_path: str = None,
    json1_name: str = "Conv1",
    json2_name: str = "Conv2"
):
    """
    Plot perplexity and divergence metrics over time.
    
    Args:
        perp1: Perplexity for conversation 1
        perp2: Perplexity for conversation 2
        kl_1_2: KL divergence from conv1 to conv2
        kl_2_1: KL divergence from conv2 to conv1
        js: JS divergence
        frame_hz: Frame rate
        save_path: Path to save figure (optional)
        json1_name: Name for conversation 1
        json2_name: Name for conversation 2
    """
    # Convert to numpy
    perp1_np = perp1.cpu().numpy()
    perp2_np = perp2.cpu().numpy()
    kl_1_2_np = kl_1_2.cpu().numpy()
    kl_2_1_np = kl_2_1.cpu().numpy()
    js_np = js.cpu().numpy()
    
    # Create time axis
    time = np.arange(len(perp1_np)) / frame_hz
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('VAP Probability Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Perplexity
    ax1 = axes[0]
    ax1.plot(time, perp1_np, label=json1_name, color='blue', alpha=0.7, linewidth=1.5)
    ax1.plot(time, perp2_np, label=json2_name, color='orange', alpha=0.7, linewidth=1.5)
    ax1.axhline(perp1_np.mean(), color='blue', linestyle='--', alpha=0.5, 
                label=f'{json1_name} mean: {perp1_np.mean():.2f}')
    ax1.axhline(perp2_np.mean(), color='orange', linestyle='--', alpha=0.5,
                label=f'{json2_name} mean: {perp2_np.mean():.2f}')
    ax1.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax1.set_title('Perplexity Over Time (lower = more predictable)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: KL Divergence
    ax2 = axes[1]
    ax2.plot(time, kl_1_2_np, label=f'KL({json1_name}||{json2_name})', 
             color='purple', alpha=0.7, linewidth=1.5)
    ax2.plot(time, kl_2_1_np, label=f'KL({json2_name}||{json1_name})', 
             color='green', alpha=0.7, linewidth=1.5)
    ax2.axhline(kl_1_2_np.mean(), color='purple', linestyle='--', alpha=0.5,
                label=f'Mean: {kl_1_2_np.mean():.3f} bits')
    ax2.axhline(kl_2_1_np.mean(), color='green', linestyle='--', alpha=0.5,
                label=f'Mean: {kl_2_1_np.mean():.3f} bits')
    ax2.set_ylabel('KL Divergence (bits)', fontsize=12, fontweight='bold')
    ax2.set_title('KL Divergence Over Time (asymmetric)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: JS Divergence
    ax3 = axes[2]
    ax3.plot(time, js_np, label='JS Divergence', color='red', alpha=0.7, linewidth=1.5)
    ax3.axhline(js_np.mean(), color='red', linestyle='--', alpha=0.5,
                label=f'Mean: {js_np.mean():.3f} bits')
    ax3.fill_between(time, 0, js_np, alpha=0.2, color='red')
    ax3.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('JS Divergence (bits)', fontsize=12, fontweight='bold')
    ax3.set_title('Jensen-Shannon Divergence Over Time (symmetric, 0 = identical)', fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
    
    plt.show()


def compare_conversations(
    json_path1: str,
    json_path2: str,
    frame_hz: int = 50,
    output_path: str = None,
    plot_path: str = None
):
    """
    Compare two conversations using perplexity and divergence metrics.
    
    Args:
        json_path1: Path to first JSON file
        json_path2: Path to second JSON file
        frame_hz: Frame rate
        output_path: Path to save results CSV (optional)
        plot_path: Path to save plot (optional, True for default name)
        
    Returns:
        pd.DataFrame: Results with metrics
    """
    print(f"Loading conversation 1: {json_path1}")
    out1 = load_vap_output(json_path1)
    
    print(f"Loading conversation 2: {json_path2}")
    out2 = load_vap_output(json_path2)
    
    # Extract probs
    probs1 = out1['probs']  # (batch, frames, 256)
    probs2 = out2['probs']
    
    print(f"Probs1 shape: {probs1.shape}")
    print(f"Probs2 shape: {probs2.shape}")
    
    # Align sequences
    probs1, probs2 = align_sequences(probs1, probs2)
    print(f"Aligned to {probs1.shape[1]} frames ({probs1.shape[1]/frame_hz:.1f}s)")
    
    # Remove batch dimension
    probs1 = probs1[0]  # (frames, 256)
    probs2 = probs2[0]
    
    # Compute metrics
    print("\nComputing metrics...")
    
    # Perplexity (intrinsic complexity)
    perp1 = compute_perplexity(probs1)
    perp2 = compute_perplexity(probs2)
    
    # KL divergence (asymmetric)
    kl_1_2 = compute_kl_divergence(probs1, probs2)
    kl_2_1 = compute_kl_divergence(probs2, probs1)
    
    # JS divergence (symmetric)
    js = compute_js_divergence(probs1, probs2)
    
    # Cross-entropy
    ce_1_2 = compute_cross_entropy(probs1, probs2)
    ce_2_1 = compute_cross_entropy(probs2, probs1)
    
    # Create results dataframe
    results = {
        'metric': [
            'perplexity_conv1',
            'perplexity_conv2',
            'kl_divergence_1→2',
            'kl_divergence_2→1',
            'js_divergence',
            'cross_entropy_1→2',
            'cross_entropy_2→1'
        ],
        'mean': [
            perp1.mean().item(),
            perp2.mean().item(),
            kl_1_2.mean().item(),
            kl_2_1.mean().item(),
            js.mean().item(),
            ce_1_2.mean().item(),
            ce_2_1.mean().item()
        ],
        'std': [
            perp1.std().item(),
            perp2.std().item(),
            kl_1_2.std().item(),
            kl_2_1.std().item(),
            js.std().item(),
            ce_1_2.std().item(),
            ce_2_1.std().item()
        ],
        'min': [
            perp1.min().item(),
            perp2.min().item(),
            kl_1_2.min().item(),
            kl_2_1.min().item(),
            js.min().item(),
            ce_1_2.min().item(),
            ce_2_1.min().item()
        ],
        'max': [
            perp1.max().item(),
            perp2.max().item(),
            kl_1_2.max().item(),
            kl_2_1.max().item(),
            js.max().item(),
            ce_1_2.max().item(),
            ce_2_1.max().item()
        ]
    }
    
    df = pd.DataFrame(results)
    
    # Print results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    
    print("\n📊 INTERPRETATION:")
    print(f"  Perplexity Conv1: {perp1.mean():.2f} (lower = more predictable)")
    print(f"  Perplexity Conv2: {perp2.mean():.2f}")
    print(f"  KL(Conv1||Conv2): {kl_1_2.mean():.3f} bits (how different Conv1 is from Conv2)")
    print(f"  KL(Conv2||Conv1): {kl_2_1.mean():.3f} bits (how different Conv2 is from Conv1)")
    print(f"  JS Divergence:    {js.mean():.3f} bits (symmetric difference, 0=identical)")
    
    # Save CSV if requested
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")
    
    # Plot if requested
    if plot_path is not None:
        print("\n📈 Creating plots...")
        
        # Generate default filename if plot_path is True
        if plot_path == True:
            plot_path = "perplexity_comparison.png"
        
        # Extract filenames for legend
        json1_name = Path(json_path1).stem
        json2_name = Path(json_path2).stem
        
        plot_perplexity(
            perp1=perp1,
            perp2=perp2,
            kl_1_2=kl_1_2,
            kl_2_1=kl_2_1,
            js=js,
            frame_hz=frame_hz,
            save_path=plot_path,
            json1_name=json1_name,
            json2_name=json2_name
        )
    
    return df


def get_args():
    parser = ArgumentParser(description="Compare VAP probability distributions")
    parser.add_argument(
        "-j1", "--json1",
        type=str,
        required=True,
        help="Path to first JSON file"
    )
    parser.add_argument(
        "-j2", "--json2",
        type=str,
        required=True,
        help="Path to second JSON file"
    )
    parser.add_argument(
        "--frame-hz",
        type=int,
        default=50,
        help="Frame rate (default: 50)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save results CSV (optional)"
    )
    parser.add_argument(
        "--plot",
        nargs='?',
        const=True,
        default=None,
        help="Create plots. Optionally specify output path (default: perplexity_comparison.png)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    # Validate paths
    json1 = Path(args.json1)
    json2 = Path(args.json2)
    
    if not json1.exists():
        raise FileNotFoundError(f"JSON file not found: {json1}")
    if not json2.exists():
        raise FileNotFoundError(f"JSON file not found: {json2}")
    
    # Compare
    compare_conversations(
        json_path1=str(json1),
        json_path2=str(json2),
        frame_hz=args.frame_hz,
        output_path=args.output,
        plot_path=args.plot
    )