"""
Combine paired mono audio files into stereo audio files to recreate conversations.

This script processes audio files with naming conventions:
- Old: {pair_id}_{speaker_id}_{task}_d{delay}_{org|dly}.wav
- New: {speaker_id}_{org|dly}_{task}_d{delay}.wav

For each conversation, it creates TWO stereo files:
1. Speaker 1's perspective: Left=S01_org, Right=S02_dly
2. Speaker 2's perspective: Left=S02_org, Right=S01_dly

This recreates the actual conversation as experienced by each participant.
"""

import os
import argparse
from pathlib import Path
import re
from typing import Dict, List, Tuple
import wave
import numpy as np


def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse audio filename to extract metadata.
    
    Supports two formats:
    - Old: {pair_id}_{speaker_id}_{task}_d{delay}_{org|dly}.wav
      Example: 01_S01_boat_d1500_dly.wav
    - New: {speaker_id}_{org|dly}_{task}_d{delay}.wav
      Example: S15_dly_beach_d0000.wav
    
    Returns:
        Dictionary with keys: pair_id, speaker_id, task, delay, perspective
    """
    # Try old format first
    pattern_old = r'(\d+)_(S\d+)_([^_]+)_d(\d+)_(org|dly)\.wav'
    match = re.match(pattern_old, filename)
    
    if match:
        return {
            'pair_id': match.group(1),
            'speaker_id': match.group(2),
            'task': match.group(3),
            'delay': match.group(4),
            'perspective': match.group(5)
        }
        
    # Try another new format with pair_id prefix
    pattern_new2 = r'(\d+)_(S\d+)_(org|dly)_([^_]+)_d(\d+)\.wav'
    match = re.match(pattern_new2, filename)

    if match:
        return {
            'pair_id': match.group(1),
            'speaker_id': match.group(2),
            'perspective': match.group(3),
            'task': match.group(4),
            'delay': match.group(5)
        }
    
    # Try new format
    pattern_new = r'(S\d+)_(org|dly)_([^_]+)_d(\d+)\.wav'
    match = re.match(pattern_new, filename)
    
    if match:
        speaker_id = match.group(1)
        # Extract pair_id from speaker_id (e.g., S15 -> pair 15, or S01 -> pair 01)
        pair_num = speaker_id[1:]  # Remove 'S' prefix
        # For pairing, we need to determine which pair this belongs to
        # Assuming speakers come in pairs: S01-S02, S03-S04, etc.
        pair_id = str((int(pair_num) + 1) // 2).zfill(2)
        
        return {
            'pair_id': pair_id,
            'speaker_id': speaker_id,
            'task': match.group(3),
            'delay': match.group(4),
            'perspective': match.group(2)
        }
    
    return None


def find_conversation_pairs(input_dir: Path) -> List[Tuple[Path, Path, Path, Path, Dict[str, str]]]:
    """
    Find all conversation pairs and return files needed for both perspectives.
    
    Returns:
        List of tuples (s1_org, s1_dly, s2_org, s2_dly, metadata_dict)
    """
    # Get all wav files
    wav_files = list(input_dir.glob('*.wav'))
    
    # Group files by conversation parameters (pair_id, task, delay)
    conversations = {}
    
    for wav_file in wav_files:
        metadata = parse_filename(wav_file.name)
        if metadata is None:
            print(f"Warning: Skipping file with unexpected format: {wav_file.name}")
            continue
        
        # Create a key for the conversation (pair_id, task, delay)
        conv_key = f"{metadata['pair_id']}_{metadata['task']}_d{metadata['delay']}"
        
        if conv_key not in conversations:
            conversations[conv_key] = {}
        
        # Store file by speaker and perspective
        speaker_persp_key = f"{metadata['speaker_id']}_{metadata['perspective']}"
        conversations[conv_key][speaker_persp_key] = (wav_file, metadata)
    
    # Match conversation pairs
    pairs = []
    for conv_key, files in conversations.items():
        # Find two different speakers
        speakers = set()
        for key in files.keys():
            speaker = key.split('_')[0]
            speakers.add(speaker)
        
        speakers = sorted(list(speakers))
        
        if len(speakers) < 2:
            print(f"Warning: Not enough speakers for conversation {conv_key}")
            continue
        
        # Get all 4 required files
        s1_org_key = f"{speakers[0]}_org"
        s1_dly_key = f"{speakers[0]}_dly"
        s2_org_key = f"{speakers[1]}_org"
        s2_dly_key = f"{speakers[1]}_dly"
        
        # Check if all 4 files exist
        required_keys = [s1_org_key, s1_dly_key, s2_org_key, s2_dly_key]
        missing = [key for key in required_keys if key not in files]
        
        if missing:
            print(f"Warning: Missing files for {conv_key}: {', '.join(missing)}")
            continue
        
        # Get the files
        s1_org_file, _ = files[s1_org_key]
        s1_dly_file, _ = files[s1_dly_key]
        s2_org_file, _ = files[s2_org_key]
        s2_dly_file, s2_metadata = files[s2_dly_key]
        
        # Create combined metadata
        combined_metadata = {
            'pair_id': s2_metadata['pair_id'],
            'task': s2_metadata['task'],
            'delay': s2_metadata['delay'],
            'speaker1': speakers[0],
            'speaker2': speakers[1]
        }
        
        pairs.append((s1_org_file, s1_dly_file, s2_org_file, s2_dly_file, combined_metadata))
    
    return pairs


def combine_to_stereo(left_file: Path, right_file: Path, output_file: Path):
    """
    Combine two mono audio files into one stereo file.
    
    Left channel: first audio file
    Right channel: second audio file
    """
    # Read the left channel file
    with wave.open(str(left_file), 'rb') as left_wav:
        left_params = left_wav.getparams()
        left_frames = left_wav.readframes(left_params.nframes)
        
        # Handle different sample widths
        if left_params.sampwidth == 2:
            left_audio = np.frombuffer(left_frames, dtype=np.int16)
        elif left_params.sampwidth == 3:
            # Convert 24-bit to 16-bit using vectorized NumPy operations
            left_audio_24 = np.frombuffer(left_frames, dtype=np.uint8)
            # Reshape to (n_samples, 3) and pad to 4 bytes
            n_samples = len(left_audio_24) // 3
            left_audio_24 = left_audio_24[:n_samples * 3].reshape(-1, 3)
            # Pad with sign extension byte and convert to int32
            left_audio_32 = np.zeros((n_samples, 4), dtype=np.uint8)
            left_audio_32[:, :3] = left_audio_24
            # Sign extend
            left_audio_32[:, 3] = np.where(left_audio_24[:, 2] >= 128, 255, 0)
            # Convert to int32 and shift down to int16 range
            left_audio = left_audio_32.view(np.int32).flatten() >> 8
            left_audio = left_audio.astype(np.int16)
        else:
            raise ValueError(f"Unsupported sample width: {left_params.sampwidth} in {left_file.name}")
    
    # Read the right channel file
    with wave.open(str(right_file), 'rb') as right_wav:
        right_params = right_wav.getparams()
        right_frames = right_wav.readframes(right_params.nframes)
        
        # Handle different sample widths
        if right_params.sampwidth == 2:
            right_audio = np.frombuffer(right_frames, dtype=np.int16)
        elif right_params.sampwidth == 3:
            # Convert 24-bit to 16-bit using vectorized NumPy operations
            right_audio_24 = np.frombuffer(right_frames, dtype=np.uint8)
            n_samples = len(right_audio_24) // 3
            right_audio_24 = right_audio_24[:n_samples * 3].reshape(-1, 3)
            right_audio_32 = np.zeros((n_samples, 4), dtype=np.uint8)
            right_audio_32[:, :3] = right_audio_24
            right_audio_32[:, 3] = np.where(right_audio_24[:, 2] >= 128, 255, 0)
            right_audio = right_audio_32.view(np.int32).flatten() >> 8
            right_audio = right_audio.astype(np.int16)
        else:
            raise ValueError(f"Unsupported sample width: {right_params.sampwidth} in {right_file.name}")
    
    # Verify both files have compatible parameters
    if left_params.framerate != right_params.framerate:
        raise ValueError(f"Sample rate mismatch: {left_file.name} ({left_params.framerate}) vs {right_file.name} ({right_params.framerate})")
    
    # Handle different lengths by padding the shorter one with silence
    max_length = max(len(left_audio), len(right_audio))
    if len(left_audio) < max_length:
        left_audio = np.pad(left_audio, (0, max_length - len(left_audio)), 'constant')
    if len(right_audio) < max_length:
        right_audio = np.pad(right_audio, (0, max_length - len(right_audio)), 'constant')
    
    # Interleave the two channels
    stereo_audio = np.empty((max_length * 2,), dtype=np.int16)
    stereo_audio[0::2] = left_audio   # Left channel
    stereo_audio[1::2] = right_audio  # Right channel
    
    # Write stereo file (always as 16-bit)
    with wave.open(str(output_file), 'wb') as stereo_wav:
        stereo_wav.setnchannels(2)  # Stereo
        stereo_wav.setsampwidth(2)  # Always use 16-bit output
        stereo_wav.setframerate(left_params.framerate)
        stereo_wav.writeframes(stereo_audio.tobytes())


def main():
    parser = argparse.ArgumentParser(
        description='Combine paired mono audio files into stereo conversation files'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing mono audio files'
    )
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        default=None,
        help='Output directory for stereo files (default: input_dir/stereo)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / 'stereo'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find conversation pairs
    print(f"Scanning for audio files in: {input_dir}")
    pairs = find_conversation_pairs(input_dir)
    print(f"Found {len(pairs)} complete conversations")
    print(f"Will create {len(pairs) * 2} stereo files (2 perspectives per conversation)")
    
    # Process each pair - create 2 stereo files per conversation
    file_count = 0
    for i, (s1_org, s1_dly, s2_org, s2_dly, metadata) in enumerate(pairs, 1):
        # Speaker 1's perspective: S01_org (left) + S02_dly (right)
        output_filename_s1 = f"{metadata['pair_id']}_{metadata['speaker1']}_{metadata['task']}_d{metadata['delay']}_conversation.wav"
        output_file_s1 = output_dir / output_filename_s1
        
        file_count += 1
        print(f"[{file_count}/{len(pairs) * 2}] Processing: {output_filename_s1}")
        print(f"  Left ({metadata['speaker1']}_org): {s1_org.name}")
        print(f"  Right ({metadata['speaker2']}_dly): {s2_dly.name}")
        
        try:
            combine_to_stereo(s1_org, s2_dly, output_file_s1)
            print(f"  ✓ Created: {output_file_s1}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        # Speaker 2's perspective: S02_org (left) + S01_dly (right)
        output_filename_s2 = f"{metadata['pair_id']}_{metadata['speaker2']}_{metadata['task']}_d{metadata['delay']}_conversation.wav"
        output_file_s2 = output_dir / output_filename_s2
        
        file_count += 1
        print(f"[{file_count}/{len(pairs) * 2}] Processing: {output_filename_s2}")
        print(f"  Left ({metadata['speaker2']}_org): {s2_org.name}")
        print(f"  Right ({metadata['speaker1']}_dly): {s1_dly.name}")
        
        try:
            combine_to_stereo(s2_org, s1_dly, output_file_s2)
            print(f"  ✓ Created: {output_file_s2}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print()  # Blank line between conversations
    
    print(f"Done! Stereo conversation files saved to: {output_dir}")


if __name__ == '__main__':
    main()