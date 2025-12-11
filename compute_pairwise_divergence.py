"""
Compute Jensen-Shannon divergence between conversational pairs.

Pairs up files with same task and delay but different speakers, computes
JS divergence over time, and saves results as JSON.

Usage:
    python compute_pairwise_divergence.py --input_dir outputs/ --output_dir divergence_results/
    python compute_pairwise_divergence.py --input_dir outputs/ --output divergence.json
"""

import torch
import numpy as np
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import re

from vap.utils import read_json


def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse filename to extract metadata.
    
    Example: 01_S01_boat_d1500_conversation.json
    Returns: {
        'pair': '01',
        'speaker': 'S01',
        'task': 'boat',
        'delay': 'd1500',
        'delay_ms': 1500
    }
    """
    name = Path(filename).stem
    
    # Pattern: {pair}_{speaker}_{task}_{delay}_conversation
    pattern = r'^(\d+)_([A-Z]\d+)_([a-z]+)_(d\d+)_conversation'
    match = re.match(pattern, name)
    
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    
    pair, speaker, task, delay = match.groups()
    delay_ms = int(delay[1:])  # Remove 'd' prefix
    
    return {
        'pair': pair,
        'speaker': speaker,
        'task': task,
        'delay': delay,
        'delay_ms': delay_ms,
        'filename': filename
    }


def get_speaker_pair(speaker: str) -> str:
    """
    Get the other speaker in the pair.
    Assumes S01 pairs with S02, S03 with S04, etc.
    """
    num = int(speaker[1:])
    if num % 2 == 1:
        return f"S{num+1:02d}"
    else:
        return f"S{num-1:02d}"


def find_conversation_pairs(input_dir: Path) -> List[Tuple[Path, Path, Dict]]:
    """
    Find all conversational pairs in directory.
    
    Args:
        input_dir: Directory containing JSON files
        
    Returns:
        List of tuples: (file1_path, file2_path, metadata)
        where metadata contains shared task, delay, and pair info
    """
    # Find all JSON files
    json_files = list(input_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    # Parse all filenames
    file_metadata = {}
    for json_file in json_files:
        try:
            metadata = parse_filename(json_file.name)
            file_metadata[json_file] = metadata
        except ValueError as e:
            print(f"Skipping {json_file.name}: {e}")
            continue
    
    # Find pairs: same pair number, task, and delay, but different speakers
    pairs = []
    processed = set()
    
    for file1, meta1 in file_metadata.items():
        if file1 in processed:
            continue
            
        # Find matching file
        expected_speaker = get_speaker_pair(meta1['speaker'])
        
        for file2, meta2 in file_metadata.items():
            if file2 in processed or file1 == file2:
                continue
            
            # Check if they match
            if (meta1['pair'] == meta2['pair'] and
                meta1['task'] == meta2['task'] and
                meta1['delay'] == meta2['delay'] and
                meta2['speaker'] == expected_speaker):
                
                pairs.append((
                    file1,
                    file2,
                    {
                        'pair': meta1['pair'],
                        'task': meta1['task'],
                        'delay': meta1['delay'],
                        'delay_ms': meta1['delay_ms'],
                        'speaker1': meta1['speaker'],
                        'speaker2': meta2['speaker']
                    }
                ))
                
                processed.add(file1)
                processed.add(file2)
                break
    
    return pairs


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
    
    Args:
        p: Reference distribution (n_frames, n_classes)
        q: Comparison distribution (n_frames, n_classes)
        epsilon: Small value to avoid log(0)
        
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
    
    JS divergence is symmetric and bounded: [0, 1] bits
    
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
    kl_pm = compute_kl_divergence(p, m, epsilon=0)
    kl_qm = compute_kl_divergence(q, m, epsilon=0)
    
    js = 0.5 * kl_pm + 0.5 * kl_qm
    return js


def align_sequences(probs1: torch.Tensor, probs2: torch.Tensor):
    """
    Align two probability sequences to same length by truncating longer one.
    
    Returns:
        tuple: (aligned_probs1, aligned_probs2)
    """
    min_frames = min(probs1.shape[1], probs2.shape[1])
    
    probs1_aligned = probs1[:, :min_frames, :]
    probs2_aligned = probs2[:, :min_frames, :]
    
    return probs1_aligned, probs2_aligned


def compute_pair_divergence(
    file1: Path,
    file2: Path,
    metadata: Dict,
    frame_hz: int = 50
) -> Dict:
    """
    Compute JS divergence between a conversational pair.
    
    Args:
        file1: Path to first JSON file
        file2: Path to second JSON file
        metadata: Shared metadata (pair, task, delay)
        frame_hz: Frame rate
        
    Returns:
        dict: Results including JS divergence array and statistics
    """
    # Load both conversations
    out1 = load_vap_output(str(file1))
    out2 = load_vap_output(str(file2))
    
    # Extract probability distributions
    probs1 = out1['probs']  # (batch, frames, 256)
    probs2 = out2['probs']
    
    # Align sequences
    probs1, probs2 = align_sequences(probs1, probs2)
    n_frames = probs1.shape[1]
    
    # Remove batch dimension
    probs1 = probs1[0]  # (frames, 256)
    probs2 = probs2[0]
    
    # Compute JS divergence per frame
    js = compute_js_divergence(probs1, probs2)
    
    # Convert to numpy for JSON serialization
    js_array = js.cpu().numpy().tolist()
    times = (np.arange(n_frames) / frame_hz).tolist()
    
    # Compute statistics
    result = {
        'metadata': metadata,
        'file1': file1.name,
        'file2': file2.name,
        'n_frames': n_frames,
        'duration_seconds': n_frames / frame_hz,
        'frame_hz': frame_hz,
        'js_divergence': {
            'values': js_array,
            'times': times,
            'statistics': {
                'mean': float(js.mean().item()),
                'std': float(js.std().item()),
                'min': float(js.min().item()),
                'max': float(js.max().item()),
                'median': float(js.median().item())
            }
        }
    }
    
    return result


def process_dataset(
    input_dir: str,
    output_path: str = None,
    frame_hz: int = 50
):
    """
    Process entire dataset and compute pairwise divergences.
    
    Args:
        input_dir: Directory containing JSON files
        output_path: Path to save results JSON (or directory for individual files)
        frame_hz: Frame rate
    """
    input_dir = Path(input_dir)
    
    print(f"🔍 Scanning directory: {input_dir}")
    
    # Find all conversational pairs
    pairs = find_conversation_pairs(input_dir)
    
    if len(pairs) == 0:
        print("❌ No conversational pairs found!")
        print("   Make sure files follow pattern: {pair}_{speaker}_{task}_{delay}_conversation.json")
        return
    
    print(f"📊 Found {len(pairs)} conversational pairs")
    
    # Determine output format
    if output_path:
        output_path = Path(output_path)
        if output_path.suffix == '.json':
            # Single JSON file with all results
            save_mode = 'single'
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Directory with individual JSON files per pair
            save_mode = 'multiple'
            output_path.mkdir(parents=True, exist_ok=True)
    else:
        save_mode = 'single'
        output_path = Path('divergence_results.json')
    
    print(f"💾 Output mode: {save_mode}")
    if save_mode == 'single':
        print(f"   Saving to: {output_path}")
    else:
        print(f"   Saving to directory: {output_path}")
    
    # Process each pair
    results = []
    
    for file1, file2, metadata in tqdm(pairs, desc="Computing divergences"):
        try:
            result = compute_pair_divergence(
                file1=file1,
                file2=file2,
                metadata=metadata,
                frame_hz=frame_hz
            )
            results.append(result)
            
            # Save individual file if in multiple mode
            if save_mode == 'multiple':
                pair_filename = f"pair{metadata['pair']}_{metadata['task']}_{metadata['delay']}_divergence.json"
                pair_path = output_path / pair_filename
                
                with open(pair_path, 'w') as f:
                    json.dump(result, f, indent=2)
            
        except Exception as e:
            print(f"\n❌ Error processing {file1.name} + {file2.name}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("DIVERGENCE ANALYSIS SUMMARY")
    print("="*80)
    print(f"Successfully processed: {len(results)}/{len(pairs)} pairs")
    
    if len(results) > 0:
        # Aggregate statistics by condition
        by_delay = {}
        by_task = {}
        
        for result in results:
            delay = result['metadata']['delay']
            task = result['metadata']['task']
            js_mean = result['js_divergence']['statistics']['mean']
            
            if delay not in by_delay:
                by_delay[delay] = []
            by_delay[delay].append(js_mean)
            
            if task not in by_task:
                by_task[task] = []
            by_task[task].append(js_mean)
        
        # Print by delay
        print(f"\n{'Delay':<10} {'Count':<8} {'Mean JS':<12} {'Std JS':<12}")
        print("-" * 50)
        for delay in sorted(by_delay.keys()):
            values = by_delay[delay]
            print(f"{delay:<10} {len(values):<8} {np.mean(values):.4f}       {np.std(values):.4f}")
        
        # Print by task
        print(f"\n{'Task':<15} {'Count':<8} {'Mean JS':<12} {'Std JS':<12}")
        print("-" * 50)
        for task in sorted(by_task.keys()):
            values = by_task[task]
            print(f"{task:<15} {len(values):<8} {np.mean(values):.4f}       {np.std(values):.4f}")
        
        # Overall statistics
        all_js_means = [r['js_divergence']['statistics']['mean'] for r in results]
        print(f"\nOverall JS Divergence:")
        print(f"  Mean: {np.mean(all_js_means):.4f} bits")
        print(f"  Std:  {np.std(all_js_means):.4f} bits")
        print(f"  Min:  {np.min(all_js_means):.4f} bits")
        print(f"  Max:  {np.max(all_js_means):.4f} bits")
    
    # Save results
    if save_mode == 'single':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ All results saved to: {output_path}")
    else:
        print(f"\n✅ Individual files saved to: {output_path}")
    
    print("="*80)
    
    return results


def get_args():
    parser = ArgumentParser(description="Compute pairwise JS divergence for conversations")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing JSON output files"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="divergence_results.json",
        help="Output path: .json file for single output, or directory for multiple files (default: divergence_results.json)"
    )
    parser.add_argument(
        "--frame-hz",
        type=int,
        default=50,
        help="Frame rate (default: 50)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    process_dataset(
        input_dir=args.input_dir,
        output_path=args.output,
        frame_hz=args.frame_hz
    )