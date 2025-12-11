"""
Convert VAP JSON output to CSV with turn-taking analysis.

Usage:
    # Single file
    python json_to_csv.py -j output.json -o turns.csv
    
    # Entire folder
    python json_to_csv.py -j outputs/ -o csv_outputs/
    
    # With custom parameters
    python json_to_csv.py -j outputs/ -o csv_outputs/ --threshold 0.6 --hysteresis 0.15
"""

import torch
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

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
    frame_hz: int = 50,
    threshold: float = 0.5,
    min_turn_duration: float = 0.07,
    hysteresis: float = 0.1
):
    """
    Extract turn-taking events from predictions.
    
    Args:
        p_now: Next speaker probabilities (n_frames,)
        frame_hz: Frame rate
        threshold: Threshold for p_now (0.5 = equal probability)
        min_turn_duration: Minimum turn duration in seconds
        hysteresis: Hysteresis margin to prevent rapid switching
        
    Returns:
        DataFrame with turn information
    """
    # Convert frame indices to time
    times = np.arange(len(p_now)) / frame_hz
    confidence = np.abs(p_now - 0.5) * 2

    # State machine with hysteresis
    current_speaker = 'A' if p_now[0] > threshold else 'B'
    speaker_sequence = []

    for prob in p_now:
        if current_speaker == 'A':
            # Need to drop below lower threshold to switch to B
            if prob < (threshold - hysteresis):
                current_speaker = 'B'
        else:  # current_speaker == 'B'
            # Need to rise above upper threshold to switch to A
            if prob > (threshold + hysteresis):
                current_speaker = 'A'
        
        speaker_sequence.append(current_speaker)
    
    speaker_sequence = np.array(speaker_sequence)
    
    predicted_speaker_numeric = (speaker_sequence == 'A').astype(int)  # 1 = A, 0 = B
    predicted_speaker = speaker_sequence
    turn_changes = np.where(np.diff(predicted_speaker_numeric) != 0)[0] + 1
    
    turns = []
    
    # Add initial state
    if len(predicted_speaker) > 0:
        initial_end_idx = turn_changes[0] if len(turn_changes) > 0 else len(times)
        initial_conf = np.mean(confidence[0:initial_end_idx])       

        turns.append({
            'turn_id': 0,
            'start_time': 0.0,
            'end_time': times[turn_changes[0]] if len(turn_changes) > 0 else times[-1],
            'duration': (times[turn_changes[0]] if len(turn_changes) > 0 else times[-1]) - 0.0,
            'predicted_speaker': predicted_speaker[0],
            'confidence': initial_conf
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
        
        segment_conf = np.mean(confidence[start_idx:end_idx])
        
        turns.append({
            'turn_id': len(turns),
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'predicted_speaker': predicted_speaker[start_idx],
            'confidence': segment_conf
        })
    
    return pd.DataFrame(turns)


def create_frame_level_csv(
    p_now: np.ndarray,
    frame_hz: int = 50,
    threshold: float = 0.5
):
    """
    Create frame-by-frame CSV with all information.
    
    Args:
        p_now: Next speaker probabilities
        frame_hz: Frame rate
        threshold: Threshold for predictions
        
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
        'confidence': np.abs(p_now - 0.5)  # Distance from 0.5
    })
    
    # Add turn-taking interpretation
    df['interpretation'] = df.apply(lambda row: 
        'HOLD (A continues)' if row['p_now'] > threshold 
        else 'SHIFT (B takes turn)', axis=1)
    
    return df


def process_single_file(
    json_path: Path,
    output_path: Path,
    frame_hz: int = 50,
    threshold: float = 0.5,
    min_turn_duration: float = 0.07,
    hysteresis: float = 0.1,
    output_format: str = 'turns',
    verbose: bool = True
):
    """
    Convert a single VAP JSON output to CSV format.
    
    Args:
        json_path: Path to JSON file
        output_path: Path to save CSV file
        frame_hz: Frame rate of predictions
        threshold: Threshold for turn predictions (default 0.5)
        min_turn_duration: Minimum turn duration in seconds
        hysteresis: Hysteresis margin for turn detection
        output_format: 'turns' or 'frames'
        verbose: Print detailed output
        
    Returns:
        DataFrame with results
    """
    try:
        # Load output
        output = load_vap_output(str(json_path))
        
        # Extract data
        p_now = output['p_now'][0, :, 0].cpu().numpy()
        
        if verbose:
            print(f"\n📁 Processing: {json_path.name}")
            print(f"   Data: {len(p_now)} frames ({len(p_now)/frame_hz:.1f}s)")
        
        if output_format == 'turns':
            # Create turn-level CSV
            df = extract_turns(
                p_now=p_now,
                frame_hz=frame_hz,
                threshold=threshold,
                min_turn_duration=min_turn_duration,
                hysteresis=hysteresis
            )
            
            if verbose:
                print(f"   Extracted {len(df)} turns")
                print(f"   - Speaker A: {(df['predicted_speaker'] == 'A').sum()}")
                print(f"   - Speaker B: {(df['predicted_speaker'] == 'B').sum()}")
            
        else:  # frames
            # Create frame-level CSV
            df = create_frame_level_csv(
                p_now=p_now,
                frame_hz=frame_hz,
                threshold=threshold
            )
            
            if verbose:
                print(f"   Created {len(df)} frames")
        
        # Save to CSV
        df.to_csv(output_path, index=False, float_format='%.4f')
        
        if verbose:
            print(f"   ✅ Saved to: {output_path.name}")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error processing {json_path.name}: {e}")
        return None


def json_to_csv(
    json_path: str,
    output_path: str,
    frame_hz: int = 50,
    threshold: float = 0.5,
    min_turn_duration: float = 0.07,
    hysteresis: float = 0.1,
    output_format: str = 'turns'
):
    """
    Convert VAP JSON output to CSV format.
    Handles both single files and directories.
    
    Args:
        json_path: Path to JSON file or directory
        output_path: Path to save CSV file or output directory
        frame_hz: Frame rate of predictions
        threshold: Threshold for turn predictions (default 0.5)
        min_turn_duration: Minimum turn duration in seconds
        hysteresis: Hysteresis margin for turn detection
        output_format: 'turns' or 'frames'
    """
    json_path = Path(json_path)
    output_path = Path(output_path)
    
    # Check if input is a directory
    if json_path.is_dir():
        # Process entire directory
        print(f"🔍 Scanning directory: {json_path}")
        
        # Find all JSON files
        json_files = list(json_path.glob("*.json"))
        
        if len(json_files) == 0:
            print(f"❌ No JSON files found in {json_path}")
            return
        
        print(f"📊 Found {len(json_files)} JSON files")
        print(f"📂 Output directory: {output_path}")
        print(f"⚙️  Settings: threshold={threshold}, hysteresis={hysteresis}, min_duration={min_turn_duration}s")
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each file
        results = []
        for json_file in tqdm(json_files, desc="Converting files"):
            # Generate output filename
            csv_file = output_path / json_file.with_suffix('.csv').name
            
            df = process_single_file(
                json_path=json_file,
                output_path=csv_file,
                frame_hz=frame_hz,
                threshold=threshold,
                min_turn_duration=min_turn_duration,
                hysteresis=hysteresis,
                output_format=output_format,
                verbose=False  # Suppress individual file output
            )
            
            if df is not None:
                results.append({
                    'filename': json_file.name,
                    'turns': len(df) if output_format == 'turns' else None,
                    'frames': len(df) if output_format == 'frames' else None,
                    'output': csv_file.name
                })
        
        # Print summary
        print("\n" + "="*70)
        print("CONVERSION SUMMARY")
        print("="*70)
        print(f"Successfully converted: {len(results)}/{len(json_files)} files")
        
        if output_format == 'turns':
            total_turns = sum(r['turns'] for r in results if r['turns'] is not None)
            print(f"Total turns extracted: {total_turns}")
            print(f"Average turns per file: {total_turns/len(results):.1f}")
        
        print(f"\n✅ All CSV files saved to: {output_path}")
        
    else:
        # Process single file
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        print(f"Loading VAP output from: {json_path}")
        print(f"Using threshold: {threshold} (>= {threshold} = Speaker A, < {threshold} = Speaker B)")
        print(f"Hysteresis: {hysteresis} (±{hysteresis} around threshold)")
        
        # Ensure output is a file path, not directory
        if output_path.is_dir():
            output_path = output_path / json_path.with_suffix('.csv').name
        
        df = process_single_file(
            json_path=json_path,
            output_path=output_path,
            frame_hz=frame_hz,
            threshold=threshold,
            min_turn_duration=min_turn_duration,
            hysteresis=hysteresis,
            output_format=output_format,
            verbose=True
        )
        
        if df is not None:
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
        help="Path to JSON file or directory with JSON files"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Path to save CSV file or output directory"
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
        "--hysteresis",
        type=float,
        default=0.1,
        help="Hysteresis margin to prevent rapid switching (default: 0.1). Higher = more conservative"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.07,
        help="Minimum turn duration in seconds (default: 0.07)"
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
    
    # Convert
    json_to_csv(
        json_path=args.json,
        output_path=args.output,
        frame_hz=args.frame_hz,
        threshold=args.threshold,
        min_turn_duration=args.min_duration,
        hysteresis=args.hysteresis,
        output_format=args.format
    )