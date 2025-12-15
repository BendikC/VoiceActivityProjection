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


def compare_conversations(
    json_path1: str,
    json_path2: str,
    frame_hz: int = 50,
    output_path: str = None
):
    """
    Compare two conversations using divergence metrics.
    
    Args:
        json_path1: Path to first JSON file
        json_path2: Path to second JSON file
        frame_hz: Frame rate
        output_path: Path to save results CSV (optional)
        
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
    
    # KL divergence (asymmetric)
    kl_1_2 = compute_kl_divergence(probs1, probs2)
    kl_2_1 = compute_kl_divergence(probs2, probs1)
    
    # JS divergence (symmetric)
    js = compute_js_divergence(probs1, probs2)
    
    # Create results dataframe
    results = {
        'metric': [
            'kl_divergence_1→2',
            'kl_divergence_2→1',
            'js_divergence'
        ],
        'mean': [
            kl_1_2.mean().item(),
            kl_2_1.mean().item(),
            js.mean().item(),
        ],
        'std': [
            kl_1_2.std().item(),
            kl_2_1.std().item(),
            js.std().item()
        ],
        'min': [
            kl_1_2.min().item(),
            kl_2_1.min().item(),
            js.min().item()
        ],
        'max': [
            kl_1_2.max().item(),
            kl_2_1.max().item(),
            js.max().item()
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
    print(f"  KL(Conv1||Conv2): {kl_1_2.mean():.3f} bits (how different Conv1 is from Conv2)")
    print(f"  KL(Conv2||Conv1): {kl_2_1.mean():.3f} bits (how different Conv2 is from Conv1)")
    print(f"  JS Divergence:    {js.mean():.3f} bits (symmetric difference, 0=identical)")
    
    # Save CSV if requested
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")
     
    
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