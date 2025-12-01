"""
Convert VAP JSON output to CSV with turn-taking analysis.

Usage:
    python json_to_csv.py -j output.json -o turns.csv
    python json_to_csv.py -j output.json -o turns.csv --threshold 0.6
    python json_to_csv.py -j output.json -o turns.csv --vad-threshold 0.5
"""

import torch
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

from vap.utils import read_json


def load_vap_output(json_path: str):
    """Load VAP model output from JSON file."""
    data = read_json(json_path)
    
    # Convert lists back to tensors
    output = {}
    for key, value in data.items():
        if isinstance(value, list):
            output[key] = torch.tensor(value)
        else:
            output[key] = value
    
    return output


def extract_turns(
    p_now: np.ndarray,
    vad: np.ndarray,
    frame_hz: int = 50,
    threshold: float = 0.5,
    vad_threshold: float = 0.5,
    min_turn_duration: float = 0.3
):
    """
    Extract turn-taking events from predictions.
    
    Args:
        p_now: Next speaker probabilities (n_frames,)
        vad: Voice activity detection (n_frames, 2)
        frame_hz: Frame rate
        threshold: Threshold for p_now (0.5 = equal probability)
        vad_threshold: Threshold for VAD detection
        min_turn_duration: Minimum turn duration in seconds
        
    Returns:
        DataFrame with turn information
    """
    # Convert frame indices to time
    times = np.arange(len(p_now)) / frame_hz
    
    
    predicted_speaker_numeric = (p_now > threshold).astype(int)  # 1 = A, 0 = B
    predicted_speaker = np.where(p_now > threshold, 'A', 'B')
    turn_changes = np.where(np.diff(predicted_speaker_numeric) != 0)[0] + 1
    
    turns = []
    
    # Add initial state
    if len(predicted_speaker) > 0:
        turns.append({
            'turn_id': 0,
            'start_time': 0.0,
            'end_time': times[turn_changes[0]] if len(turn_changes) > 0 else times[-1],
            'duration': (times[turn_changes[0]] if len(turn_changes) > 0 else times[-1]) - 0.0,
            'predicted_speaker': predicted_speaker[0],
            'confidence': abs(p_now[0] - 0.5) * 2
        })
    
    # Process each turn change
    for i, change_idx in enumerate(turn_changes):
        start_idx = change_idx
        end_idx = turn_changes[i + 1] if i + 1 < len(turn_changes) else len(p_now)
        
        start_time = times[start_idx]
        end_time = times[end_idx - 1] if end_idx > start_idx else times[start_idx]
        duration = end_time - start_time
        
        # Skip very short turns
        if duration < min_turn_duration:
            continue
        
        # Get average confidence during this turn
        segment_probs = p_now[start_idx:end_idx]
        avg_confidence = np.mean(np.abs(segment_probs - 0.5))
        
        
        turns.append({
            'turn_id': len(turns),
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'predicted_speaker': predicted_speaker[start_idx],
            'confidence': avg_confidence * 2
        })
    
    return pd.DataFrame(turns)


def create_frame_level_csv(
    p_now: np.ndarray,
    vad: np.ndarray,
    frame_hz: int = 50,
    threshold: float = 0.5,
    vad_threshold: float = 0.5
):
    """
    Create frame-by-frame CSV with all information.
    
    Args:
        p_now: Next speaker probabilities
        vad: Voice activity detection
        frame_hz: Frame rate
        threshold: Threshold for predictions
        vad_threshold: Threshold for VAD
        
    Returns:
        DataFrame with frame-level information
    """
    times = np.arange(len(p_now)) / frame_hz
    
    df = pd.DataFrame({
        'time': times,
        'p_now': p_now,
        'p_speaker_a': p_now,  # Probability Speaker A is next
        'p_speaker_b': 1 - p_now,  # Probability Speaker B is next
        'predicted_speaker': np.where(p_now > threshold, 'A', 'B'),
        'confidence': np.abs(p_now - 0.5),  # Distance from 0.5
        'vad_speaker_a': vad[:, 0],
        'vad_speaker_b': vad[:, 1],
        'speaker_a_active': vad[:, 0] > vad_threshold,
        'speaker_b_active': vad[:, 1] > vad_threshold,
    })
    
    # Add turn-taking interpretation
    df['interpretation'] = df.apply(lambda row: 
        'HOLD (A continues)' if row['p_now'] > threshold 
        else 'SHIFT (B takes turn)', axis=1)
    
    return df


def json_to_csv(
    json_path: str,
    output_path: str,
    frame_hz: int = 50,
    threshold: float = 0.5,
    vad_threshold: float = 0.5,
    min_turn_duration: float = 0.3,
    output_format: str = 'turns'
):
    """
    Convert VAP JSON output to CSV format.
    
    Args:
        json_path: Path to JSON file
        output_path: Path to save CSV file
        frame_hz: Frame rate of predictions
        threshold: Threshold for turn predictions (default 0.5)
        vad_threshold: Threshold for VAD detection (default 0.5)
        min_turn_duration: Minimum turn duration in seconds
        output_format: 'turns' or 'frames'
    """
    print(f"Loading VAP output from: {json_path}")
    output = load_vap_output(json_path)
    
    # Extract data
    p_now = output['p_now'][0, :, 0].cpu().numpy()
    vad = output['vad'][0].cpu().numpy()
    
    print(f"Data loaded - {len(p_now)} frames ({len(p_now)/frame_hz:.1f}s)")
    print(f"Using threshold: {threshold} (>= {threshold} = Speaker A, < {threshold} = Speaker B)")
    print(f"Using VAD threshold: {vad_threshold}")
    
    if output_format == 'turns':
        # Create turn-level CSV
        df = extract_turns(
            p_now=p_now,
            vad=vad,
            frame_hz=frame_hz,
            threshold=threshold,
            vad_threshold=vad_threshold,
            min_turn_duration=min_turn_duration
        )
        
        print(f"\nExtracted {len(df)} turns")
        print(f"  - Speaker A turns: {(df['predicted_speaker'] == 'A').sum()}")
        print(f"  - Speaker B turns: {(df['predicted_speaker'] == 'B').sum()}")
        
    else:  # frames
        # Create frame-level CSV
        df = create_frame_level_csv(
            p_now=p_now,
            vad=vad,
            frame_hz=frame_hz,
            threshold=threshold,
            vad_threshold=vad_threshold
        )
        
        print(f"\nCreated frame-by-frame data: {len(df)} frames")
    
    # Save to CSV
    df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\n✅ CSV saved to: {output_path}")
    
    # Print preview
    print("\n📊 Preview (first 5 rows):")
    print(df.head().to_string())
    
    # Print summary statistics
    if output_format == 'turns':
        print("\n📈 Summary Statistics:")
        print(f"  Total turns: {len(df)}")
        print(f"  Average turn duration: {df['duration'].mean():.2f}s")
        print(f"  Median confidence: {df['confidence'].median():.3f}")
    
    return df


def get_args():
    parser = ArgumentParser(description="Convert VAP JSON to CSV for linguistic analysis")
    parser.add_argument(
        "-j", "--json",
        type=str,
        required=True,
        help="Path to JSON file with VAP output"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Path to save CSV file"
    )
    parser.add_argument(
        "--frame-hz",
        type=int,
        default=50,
        help="Frame rate of predictions (default: 50)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for turn predictions (default: 0.5). Values >= threshold = Speaker A, < threshold = Speaker B"
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="Threshold for VAD detection (default: 0.5)"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.3,
        help="Minimum turn duration in seconds (default: 0.3)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['turns', 'frames'],
        default='turns',
        help="Output format: 'turns' for turn-level analysis, 'frames' for frame-by-frame data"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    # Validate paths
    json_path = Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    # Convert
    json_to_csv(
        json_path=str(json_path),
        output_path=args.output,
        frame_hz=args.frame_hz,
        threshold=args.threshold,
        vad_threshold=args.vad_threshold,
        min_turn_duration=args.min_duration,
        output_format=args.format
    )