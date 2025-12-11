"""
Evaluate VAP predictions against ground truth for entire dataset.

Usage:
    python evaluate_dataset.py --pred_dir outputs/ --gt_csv ch_solo.csv --output results.csv
    python evaluate_dataset.py --pred_dir outputs/ --gt_csv ch_solo.csv --output results.csv --threshold 1.0
"""

import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import re
from typing import Dict, List, Tuple
from tqdm import tqdm


def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse filename to extract metadata.
    
    Example: 01_S01_boat_d1500_conversation.csv
    Returns: {
        'pair': '01',
        'speaker': 'S01',
        'task': 'boat',
        'delay': 'd1500',
        'delay_ms': 1500
    }
    """
    # Remove extension
    name = Path(filename).stem
    
    # Pattern: {pair}_{speaker}_{task}_{delay}_conversation[_optional_suffix]
    pattern = r'^(\d+)_([A-Z]\d+)_([a-z]+)_(d\d+)_conversation'
    match = re.match(pattern, name)
    
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    
    pair, speaker, task, delay = match.groups()
    delay_ms = int(delay[1:])  # Remove 'd' prefix and convert to int
    
    return {
        'pair': pair,
        'speaker': speaker,
        'task': task,
        'delay': delay,
        'delay_ms': delay_ms,
        'filename': filename
    }


def read_prediction_csv(pred_csv_path: str) -> Dict[Tuple[float, float], str]:
    """Reads predicted turns from the csv and returns as a dict."""
    df = pd.read_csv(pred_csv_path)
    pred_turns = {}
    for _, row in df.iterrows():
        start = row['start_time']
        end = row['end_time']
        speaker = row['predicted_speaker']
        pred_turns[(start, end)] = speaker
    return pred_turns


def read_ground_truth_csv(gt_csv_path: str, speaker_1: str, speaker_2: str, 
                         task: str, level: str) -> Dict[Tuple[float, float], str]:
    """Reads ground truth turns from the csv and returns as a dict."""
    df = pd.read_csv(gt_csv_path)
    
    # Filter for correct task and level first
    df = df[(df['task'] == task) & (df['level'] == level)]
    
    # Then filter for either speaker
    df = df[(df['speaker'] == speaker_1) | (df['speaker'] == speaker_2)]

    # get the delay from the level, level is d0000, d0500, d1000, d1500
    delay_ms = int(level[1:])
    delay = delay_ms / 1000.0  # convert to seconds
    
    gt_turns = {}
    for _, row in df.iterrows():
        start = row['start_time']
        end = row['end_time']
        speaker = row['speaker']
        if speaker == speaker_1:
            gt_turns[(start, end)] = 'A'
        else:
            # we add the delay to the speaker 2 turn to simulate the delayed response
            gt_turns[(start + delay, end + delay)] = 'B'
    
    return gt_turns


def compute_turn_durations(turns: Dict[Tuple[float, float], str]) -> Dict:
    """
    Compute turn duration statistics.
    
    Args:
        turns: Dictionary of turns {(start, end): speaker}
    
    Returns:
        dict: Statistics including mean, median, std, min, max durations
    """
    if len(turns) == 0:
        return {
            'mean_duration': 0.0,
            'median_duration': 0.0,
            'std_duration': 0.0,
            'min_duration': 0.0,
            'max_duration': 0.0
        }
    
    durations = [end - start for (start, end) in turns.keys()]
    
    return {
        'mean_duration': sum(durations) / len(durations),
        'median_duration': sorted(durations)[len(durations) // 2],
        'std_duration': (sum((d - sum(durations) / len(durations)) ** 2 for d in durations) / len(durations)) ** 0.5,
        'min_duration': min(durations),
        'max_duration': max(durations)
    }


def compare_turns(pred_turns: Dict[Tuple[float, float], str], 
                 gt_turns: Dict[Tuple[float, float], str],
                 threshold: float = 1.0) -> Dict:
    """
    Compares the predicted turns with the ground truth turns.
    
    Returns:
        dict: Metrics including correct_shifts, false_positives, false_negatives, avg_time_difference, duration stats
    """
    correct_shifts = 0
    false_positives = 0
    matched_gt_shifts = set()
    time_differences = []
    
    # Iterate over predicted turns
    for (pred_start, pred_end), pred_speaker in pred_turns.items():
        pred_shift_time = pred_start
        matched = False
        
        for (gt_start, gt_end), gt_speaker in gt_turns.items():
            gt_shift_time = gt_start
            
            # Check if prediction matches ground truth
            if abs(pred_shift_time - gt_shift_time) <= threshold:
                if pred_shift_time <= gt_shift_time:  # Must be predictive
                    if pred_speaker == gt_speaker:
                        if (gt_start, gt_end) not in matched_gt_shifts:
                            correct_shifts += 1
                            matched_gt_shifts.add((gt_start, gt_end))
                            time_differences.append(abs(pred_shift_time - gt_shift_time))
                            matched = True
                            break
        
        if not matched:
            false_positives += 1
    
    # Count false negatives
    false_negatives = len(gt_turns) - len(matched_gt_shifts)
    avg_time_difference = sum(time_differences) / len(time_differences) if time_differences else 0

    # Compute turn duration statistics
    pred_duration_stats = compute_turn_durations(pred_turns)
    gt_duration_stats = compute_turn_durations(gt_turns)
    
    return {
        'correct_shifts': correct_shifts,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'avg_time_difference': avg_time_difference,
        'total_predictions': len(pred_turns),
        'total_ground_truth': len(gt_turns),
        'pred_mean_duration': pred_duration_stats['mean_duration'],
        'pred_median_duration': pred_duration_stats['median_duration'],
        'pred_std_duration': pred_duration_stats['std_duration'],
        'gt_mean_duration': gt_duration_stats['mean_duration'],
        'gt_median_duration': gt_duration_stats['median_duration'],
        'gt_std_duration': gt_duration_stats['std_duration']
    }


def compute_metrics(correct: int, false_pos: int, false_neg: int) -> Dict:
    """Compute precision, recall, and F1 score."""
    precision = correct / (correct + false_pos) if (correct + false_pos) > 0 else 0
    recall = correct / (correct + false_neg) if (correct + false_neg) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
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


def evaluate_file(pred_path: Path, gt_csv_path: str, threshold: float = 1.0) -> Dict:
    """
    Evaluate a single prediction file against ground truth.
    
    Args:
        pred_path: Path to prediction CSV
        gt_csv_path: Path to ground truth CSV
        threshold: Time threshold for matching turns (seconds)
    
    Returns:
        dict: Evaluation results including metadata and metrics
    """
    # Parse filename to get metadata
    try:
        metadata = parse_filename(pred_path.name)
    except ValueError as e:
        print(f"Skipping {pred_path.name}: {e}")
        return None
    
    # Get speaker pair
    speaker_1 = metadata['speaker']
    speaker_2 = get_speaker_pair(speaker_1)
    
    # Read predictions and ground truth
    try:
        pred_turns = read_prediction_csv(str(pred_path))
        gt_turns = read_ground_truth_csv(
            gt_csv_path, 
            speaker_1=speaker_1, 
            speaker_2=speaker_2,
            task=metadata['task'],
            level=metadata['delay']
        )
    except Exception as e:
        print(f"Error reading {pred_path.name}: {e}")
        return None
    
    # Compare turns
    comparison = compare_turns(pred_turns, gt_turns, threshold=threshold)
    
    # Compute metrics
    metrics = compute_metrics(
        comparison['correct_shifts'],
        comparison['false_positives'],
        comparison['false_negatives']
    )
    
    # Combine all results
    result = {
        **metadata,
        **comparison,
        **metrics
    }
    
    return result


def evaluate_dataset(pred_dir: str, gt_csv_path: str, output_path: str = None, 
                     threshold: float = 1.0, pattern: str = "*.csv"):
    """
    Evaluate all prediction files in a directory against ground truth.
    
    Args:
        pred_dir: Directory containing prediction CSV files
        gt_csv_path: Path to ground truth CSV
        output_path: Path to save results CSV (optional)
        threshold: Time threshold for matching turns (seconds)
        pattern: File pattern to match (default: *.csv)
    """
    pred_dir = Path(pred_dir)
    
    # Find all prediction files
    pred_files = list(pred_dir.glob(pattern))
    print(f"Found {len(pred_files)} prediction files in {pred_dir}")
    
    if len(pred_files) == 0:
        print(f"No files matching pattern '{pattern}' found in {pred_dir}")
        return
    
    # Evaluate each file
    results = []
    for pred_path in tqdm(pred_files, desc="Evaluating files"):
        result = evaluate_file(pred_path, gt_csv_path, threshold=threshold)
        if result is not None:
            results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    column_order = [
        'filename', 'pair', 'speaker', 'task', 'delay', 'delay_ms',
        'correct_shifts', 'false_positives', 'false_negatives',
        'total_predictions', 'total_ground_truth',
        'precision', 'recall', 'f1',
        'avg_time_difference',
        'pred_mean_duration', 'pred_median_duration', 'pred_std_duration',
        'gt_mean_duration', 'gt_median_duration', 'gt_std_duration'
    ]
    df = df[column_order]
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total files evaluated: {len(results)}")
    print(f"\nOverall Metrics:")
    print(f"  Mean Precision: {df['precision'].mean():.3f} (±{df['precision'].std():.3f})")
    print(f"  Mean Recall:    {df['recall'].mean():.3f} (±{df['recall'].std():.3f})")
    print(f"  Mean F1:        {df['f1'].mean():.3f} (±{df['f1'].std():.3f})")
    print(f"  Mean Time Diff: {df['avg_time_difference'].mean():.3f}s (±{df['avg_time_difference'].std():.3f}s)")
    print(f"\nTurn Duration:")
    print(f"  Predicted - Mean:   {df['pred_mean_duration'].mean():.3f}s (±{df['pred_mean_duration'].std():.3f}s)")
    print(f"  Predicted - Median: {df['pred_median_duration'].mean():.3f}s (±{df['pred_median_duration'].std():.3f}s)")
    print(f"  Ground Truth - Mean:   {df['gt_mean_duration'].mean():.3f}s (±{df['gt_mean_duration'].std():.3f}s)")
    print(f"  Ground Truth - Median: {df['gt_median_duration'].mean():.3f}s (±{df['gt_median_duration'].std():.3f}s)")
    
    # Group by delay
    print(f"\n{'Delay':<10} {'Count':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Pred Dur':<12} {'GT Dur':<12}")
    print("-" * 80)
    for delay, group in df.groupby('delay'):
        print(f"{delay:<10} {len(group):<8} "
              f"{group['precision'].mean():.3f} (±{group['precision'].std():.2f})  "
              f"{group['recall'].mean():.3f} (±{group['recall'].std():.2f})  "
              f"{group['f1'].mean():.3f} (±{group['f1'].std():.2f})  "
              f"{group['pred_mean_duration'].mean():.2f}s      "
              f"{group['gt_mean_duration'].mean():.2f}s")
    
    # Group by task
    print(f"\n{'Task':<15} {'Count':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Pred Dur':<12} {'GT Dur':<12}")
    print("-" * 90)
    for task, group in df.groupby('task'):
        print(f"{task:<15} {len(group):<8} "
              f"{group['precision'].mean():.3f} (±{group['precision'].std():.2f})  "
              f"{group['recall'].mean():.3f} (±{group['recall'].std():.2f})  "
              f"{group['f1'].mean():.3f} (±{group['f1'].std():.2f})  "
              f"{group['pred_mean_duration'].mean():.2f}s      "
              f"{group['gt_mean_duration'].mean():.2f}s")
    
    # Save results
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")
    
    return df


def aggregate_results(results_csv: str, group_by: str = 'delay'):
    """
    Aggregate results from evaluation CSV.
    
    Args:
        results_csv: Path to results CSV from evaluate_dataset
        group_by: Column to group by ('delay', 'task', 'speaker', 'pair')
    """
    df = pd.read_csv(results_csv)
    
    print(f"\n{'='*80}")
    print(f"AGGREGATED RESULTS BY {group_by.upper()}")
    print(f"{'='*80}")
    
    grouped = df.groupby(group_by).agg({
        'precision': ['mean', 'std', 'min', 'max'],
        'recall': ['mean', 'std', 'min', 'max'],
        'f1': ['mean', 'std', 'min', 'max'],
        'avg_time_difference': ['mean', 'std'],
        'pred_mean_duration': ['mean', 'std'],
        'gt_mean_duration': ['mean', 'std'],
        'correct_shifts': 'sum',
        'false_positives': 'sum',
        'false_negatives': 'sum',
        'filename': 'count'
    })
    
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'filename_count': 'n_files'})
    
    print(grouped.to_string())
    
    return grouped


def get_args():
    parser = ArgumentParser(description="Evaluate VAP predictions against ground truth")
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Directory containing prediction CSV files"
    )
    parser.add_argument(
        "--gt_csv",
        type=str,
        required=True,
        help="Path to ground truth CSV file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="evaluation_results.csv",
        help="Path to save results CSV (default: evaluation_results.csv)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Time threshold for matching turns in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="File pattern to match (default: *.csv)"
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        choices=['delay', 'task', 'speaker', 'pair'],
        help="Aggregate results by specified column (optional)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    # Evaluate dataset
    df = evaluate_dataset(
        pred_dir=args.pred_dir,
        gt_csv_path=args.gt_csv,
        output_path=args.output,
        threshold=args.threshold,
        pattern=args.pattern
    )
    
    # Optional aggregation
    if args.aggregate and df is not None:
        aggregate_results(args.output, group_by=args.aggregate)