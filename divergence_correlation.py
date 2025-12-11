"""
Analyze correlation between JS divergence and turn-taking events from ground truth.

Usage:
    python divergence_correlation.py --divergence divergence_results/ --gt_csv ch_solo.csv --output correlation_results.csv
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, List
from scipy.stats import pearsonr
from tqdm import tqdm


def load_divergence_results(divergence_path: str) -> List[Dict]:
    """Load JS divergence results from JSON file or directory."""
    divergence_path = Path(divergence_path)
    
    if divergence_path.is_file():
        with open(divergence_path, 'r') as f:
            return json.load(f)
    
    elif divergence_path.is_dir():
        print(f"📂 Loading from directory: {divergence_path}")
        json_files = list(divergence_path.glob("*.json"))
        print(f"   Found {len(json_files)} JSON files")
        
        results = []
        for json_file in tqdm(json_files, desc="Loading files"):
            try:
                with open(json_file, 'r') as f:
                    results.append(json.load(f))
            except Exception as e:
                print(f"   ⚠️  Error loading {json_file.name}: {e}")
        
        return results
    
    else:
        raise FileNotFoundError(f"Path not found: {divergence_path}")


def load_ground_truth(gt_csv_path: str, speaker_1: str, speaker_2: str, 
                     task: str, delay: str) -> pd.DataFrame:
    """Load ground truth turns for a conversation pair."""
    df = pd.read_csv(gt_csv_path)
    
    # Filter
    df = df[(df['task'] == task) & (df['level'] == delay)]
    df = df[(df['speaker'] == speaker_1) | (df['speaker'] == speaker_2)]
    
    # Apply delay to speaker 2
    delay_sec = int(delay[1:]) / 1000.0
    df = df.copy()
    df.loc[df['speaker'] == speaker_2, 'start_time'] += delay_sec
    df.loc[df['speaker'] == speaker_2, 'end_time'] += delay_sec
    
    return df.sort_values('start_time').reset_index(drop=True)


def extract_turn_events(gt_df: pd.DataFrame, duration: float, frame_hz: int = 50) -> Dict[str, np.ndarray]:
    """
    Extract simple binary indicators for turn-taking events.
    
    Returns 4 key event types:
    - turn_switch: Any speaker change
    - overlap: Simultaneous speech
    - gap: Silence between turns
    - turn_onset: Start of speech
    """
    n_frames = int(duration * frame_hz)
    
    events = {
        'turn_switch': np.zeros(n_frames, dtype=int),
        'overlap': np.zeros(n_frames, dtype=int),
        'gap': np.zeros(n_frames, dtype=int),
        'turn_onset': np.zeros(n_frames, dtype=int),
    }
    
    turns = []
    for _, row in gt_df.iterrows():
        start_frame = int(row['start_time'] * frame_hz)
        end_frame = int(row['end_time'] * frame_hz)
        turns.append({'start': start_frame, 'end': end_frame})
        
        # Mark turn onset
        if start_frame < n_frames:
            events['turn_onset'][start_frame] = 1
    
    # Detect overlaps and gaps
    for i in range(len(turns) - 1):
        curr = turns[i]
        next_turn = turns[i + 1]
        
        # Turn switch
        if next_turn['start'] < n_frames:
            events['turn_switch'][next_turn['start']] = 1
        
        # Overlap
        if next_turn['start'] < curr['end']:
            overlap_start = next_turn['start']
            overlap_end = min(curr['end'], next_turn['end'], n_frames)
            events['overlap'][overlap_start:overlap_end] = 1
        
        # Gap
        elif next_turn['start'] > curr['end']:
            gap_start = curr['end']
            gap_end = min(next_turn['start'], n_frames)
            events['gap'][gap_start:gap_end] = 1
    
    return events


def compute_correlation_at_lag(js_values: np.ndarray, event_values: np.ndarray, 
                               lag_frames: int) -> float:
    """Compute correlation with a specific lag."""
    if lag_frames >= 0:
        # Event happens AFTER JS spike
        js_shifted = js_values[:-lag_frames] if lag_frames > 0 else js_values
        ev_shifted = event_values[lag_frames:]
    else:
        # Event happens BEFORE JS spike
        js_shifted = js_values[-lag_frames:]
        ev_shifted = event_values[:lag_frames]
    
    min_len = min(len(js_shifted), len(ev_shifted))
    if min_len < 10 or ev_shifted[:min_len].sum() == 0:
        return np.nan
    
    try:
        r, _ = pearsonr(js_shifted[:min_len], ev_shifted[:min_len])
        return r
    except:
        return np.nan


def analyze_pair(divergence_result: Dict, gt_csv_path: str, 
                max_lag_sec: float = 5.0, frame_hz: int = 50) -> Dict:
    """Analyze one conversational pair."""
    metadata = divergence_result['metadata']
    
    # Load ground truth
    gt_df = load_ground_truth(
        gt_csv_path,
        speaker_1=metadata['speaker1'],
        speaker_2=metadata['speaker2'],
        task=metadata['task'],
        delay=metadata['delay']
    )
    
    # Get JS divergence
    js_values = np.array(divergence_result['js_divergence']['values'])
    duration = divergence_result['duration_seconds']
    
    # Extract events
    events = extract_turn_events(gt_df, duration, frame_hz)
    
    # Compute correlations at different lags
    max_lag_frames = int(max_lag_sec * frame_hz)
    lags = range(-max_lag_frames, max_lag_frames + 1)
    
    results = {
        'metadata': metadata,
        'events': {}
    }
    
    for event_name, event_values in events.items():
        min_len = min(len(js_values), len(event_values))
        
        if event_values[:min_len].sum() == 0:
            continue  # Skip if no events
        
        # Compute correlation at each lag
        correlations = [compute_correlation_at_lag(js_values[:min_len], 
                                                   event_values[:min_len], 
                                                   lag) 
                       for lag in lags]
        
        # Find peak
        correlations = np.array(correlations)
        valid_idx = ~np.isnan(correlations)
        
        if valid_idx.sum() == 0:
            continue
        
        peak_idx = np.nanargmax(np.abs(correlations))
        peak_lag = list(lags)[peak_idx] / frame_hz
        peak_corr = correlations[peak_idx]
        
        # Zero-lag correlation
        zero_idx = max_lag_frames
        zero_corr = correlations[zero_idx] if not np.isnan(correlations[zero_idx]) else 0
        
        results['events'][event_name] = {
            'event_count': int(event_values[:min_len].sum()),
            'zero_lag_r': float(zero_corr),
            'peak_lag_sec': float(peak_lag),
            'peak_r': float(peak_corr)
        }
    
    return results


def process_all_pairs(divergence_path: str, gt_csv_path: str, 
                     output_csv: str = None, max_lag_sec: float = 5.0):
    """Process all conversational pairs."""
    print(f"🔍 Loading divergence results...")
    divergence_results = load_divergence_results(divergence_path)
    print(f"📊 Processing {len(divergence_results)} pairs")
    
    all_results = []
    for div_result in tqdm(divergence_results, desc="Analyzing"):
        try:
            result = analyze_pair(div_result, gt_csv_path, max_lag_sec)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue
    
    # Create summary table
    rows = []
    for result in all_results:
        meta = result['metadata']
        for event_name, stats in result['events'].items():
            rows.append({
                'pair': meta['pair'],
                'task': meta['task'],
                'delay': meta['delay'],
                'event_type': event_name,
                'event_count': stats['event_count'],
                'correlation': stats['zero_lag_r'],
                'peak_lag_sec': stats['peak_lag_sec'],
                'peak_correlation': stats['peak_r']
            })
    
    df = pd.DataFrame(rows)
    
    # Print summary
    print("\n" + "="*70)
    print("CORRELATION SUMMARY")
    print("="*70)
    
    print(f"\n{'Event Type':<15} {'Pairs':<8} {'Avg Corr':<12} {'Avg Peak Lag':<15}")
    print("-" * 55)
    for event in df['event_type'].unique():
        subset = df[df['event_type'] == event]
        print(f"{event:<15} {len(subset):<8} "
              f"{subset['correlation'].mean():>6.3f}      "
              f"{subset['peak_lag_sec'].mean():>6.2f}s")
    
    # Strong correlations
    strong = df[df['correlation'].abs() > 0.2]
    if len(strong) > 0:
        print(f"\n📈 Strong correlations (|r| > 0.2):")
        print(strong[['pair', 'task', 'delay', 'event_type', 'correlation']].to_string(index=False))
    
    # Save
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Saved to: {output_csv}")
    
    print("="*70)
    return df


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--divergence", required=True, 
                       help="Path to divergence JSON file or directory")
    parser.add_argument("--gt_csv", required=True, 
                       help="Path to ground truth CSV")
    parser.add_argument("-o", "--output", default="correlation_results.csv",
                       help="Output CSV path")
    parser.add_argument("--max_lag", type=float, default=5.0,
                       help="Maximum lag in seconds (default: 5.0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    df = process_all_pairs(args.divergence, args.gt_csv, args.output, args.max_lag)