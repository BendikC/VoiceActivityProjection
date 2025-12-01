"""
Visualize VAP model output with interactive audio playback.

Usage:
    python visualize_with_playback.py -j output.json -a audio.wav
    python visualize_with_playback.py -j output.json -a audio.wav --html output.html
"""

import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
import numpy as np

from vap.utils import read_json
from vap.audio import load_waveform
from vap.plot_utils import plot_stereo
from audio_playback import add_audio_playback


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

def visualize_with_playback(
    json_path: str,
    audio_path: str,
    sample_rate: int = 16000,
    frame_hz: int = 50,
    figsize=(16, 9),
    start_time: float = 0.0,
    duration: float = None
):
    """
    Load VAP output and create interactive visualization with audio playback.
    
    Args:
        json_path: Path to JSON file with VAP output
        audio_path: Path to audio file
        sample_rate: Audio sample rate
        frame_hz: Frame rate of predictions
        figsize: Figure size
        start_time: Start time in seconds for the snippet
        duration: Duration in seconds to visualize (None = full audio)
        html_output: Path to save HTML file (optional)
    """
    import sys
    import traceback
    
    try:
        # Load data
        print(f"Loading VAP output from: {json_path}")
        output = load_vap_output(json_path)
        print(f"Loaded keys: {list(output.keys())}")
        
        print(f"Loading audio from: {audio_path}")
        waveform, _ = load_waveform(audio_path, sample_rate=sample_rate)
        print(f"Waveform shape: {waveform.shape}")
        
        # Ensure stereo
        if waveform.shape[0] == 1:
            waveform = torch.cat((waveform, torch.zeros_like(waveform)))
            print(f"Converted to stereo: {waveform.shape}")
        
        # Extract data from output
        print("Extracting data from output...")
        p_now = output['p_now'][0, :, 0]  # Shape: (n_frames,)
        vad = output['vad'][0]  # Shape: (n_frames, 2)
        
        print(f"p_now shape: {p_now.shape}")
        print(f"vad shape: {vad.shape}")
        
        # Calculate time window
        if duration is not None:
            # Convert times to samples/frames
            start_sample = int(start_time * sample_rate)
            end_sample = int((start_time + duration) * sample_rate)
            start_frame = int(start_time * frame_hz)
            end_frame = int((start_time + duration) * frame_hz)
            
            # Clip to valid ranges
            start_sample = max(0, min(start_sample, waveform.shape[-1]))
            end_sample = max(0, min(end_sample, waveform.shape[-1]))
            start_frame = max(0, min(start_frame, len(p_now)))
            end_frame = max(0, min(end_frame, len(p_now)))
            
            # Extract snippets
            waveform = waveform[:, start_sample:end_sample]
            p_now = p_now[start_frame:end_frame]
            vad = vad[start_frame:end_frame]
            
            print(f"Extracted snippet: {start_time}s to {start_time + duration}s")
            print(f"  Waveform shape: {waveform.shape}")
            print(f"  p_now shape: {p_now.shape}")
            print(f"  vad shape: {vad.shape}")
        
        # Create matplotlib plot
        print("Creating visualization...")
        sys.stdout.flush()
        
        fig, ax = plot_stereo(
            waveform=waveform,
            p_ns=p_now,
            vad=vad,
            plot=False,
            figsize=figsize
        )
        
        print("Plot created successfully!")
        print(f"Type of ax: {type(ax)}")
        sys.stdout.flush()
        
        # Handle different return types from plot_stereo
        if isinstance(ax, np.ndarray):
            axes_list = ax.flatten().tolist()
            print(f"Converted numpy array to list: {len(axes_list)} axes")
        elif isinstance(ax, list):
            axes_list = ax
            print(f"Already a list: {len(axes_list)} axes")
        elif isinstance(ax, plt.Axes):
            axes_list = [ax]
            print(f"Single axis, wrapped in list")
        else:
            print(f"Unexpected type: {type(ax)}")
            try:
                axes_list = list(ax)
                print(f"Converted iterable to list: {len(axes_list)} axes")
            except:
                axes_list = [ax]
                print(f"Wrapped in list as fallback")
        
        # Add audio playback
        print("Adding audio playback controls...")
        sys.stdout.flush()
        
        player = add_audio_playback(
            fig=fig,
            axes=axes_list,
            waveform=waveform,
            sample_rate=sample_rate,
            frame_hz=frame_hz
        )
        
        print("Playback controls added!")
        print("Showing plot window...")
        sys.stdout.flush()
        
        plt.show()
        
        return fig, axes_list, player
        
    except Exception as e:
        print(f"\n!!! ERROR !!!")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.stdout.flush()
        raise


def get_args():
    parser = ArgumentParser(description="Visualize VAP output with audio playback")
    parser.add_argument(
        "-j", "--json",
        type=str,
        required=True,
        help="Path to JSON file with VAP output"
    )
    parser.add_argument(
        "-a", "--audio",
        type=str,
        required=True,
        help="Path to audio file"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Audio sample rate (default: 16000)"
    )
    parser.add_argument(
        "--frame_hz",
        type=int,
        default=50,
        help="Frame rate of predictions (default: 50)"
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[16, 9],
        help="Figure size as width height (default: 16 9)"
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time in seconds (default: 0.0)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration in seconds to visualize (default: 30.0, use 0 for full audio)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    # Validate paths
    json_path = Path(args.json)
    audio_path = Path(args.audio)
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Handle duration = 0 as "full audio"
    duration = None if args.duration == 0 else args.duration
    
    # Visualize
    visualize_with_playback(
        json_path=str(json_path),
        audio_path=str(audio_path),
        sample_rate=args.sample_rate,
        frame_hz=args.frame_hz,
        figsize=tuple(args.figsize),
        start_time=args.start,
        duration=duration
    )