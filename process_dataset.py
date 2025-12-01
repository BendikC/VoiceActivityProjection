from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import torch

from run import load_vap_model, process_audio
from vap.model import VapConfig


def process_dataset(
    audio_dir,
    output_dir,
    state_dict_path,
    audio_extension=".wav",
    chunk=True,
    chunk_time=30,
    step_time=5,
    plot=False,
    conf=None
):
    """
    Process all audio files in a directory with VAP model.
    
    Args:
        audio_dir: Directory containing audio files
        output_dir: Directory to save outputs
        state_dict_path: Path to VAP model state dict
        audio_extension: Audio file extension to process
        chunk: Whether to process in chunks
        chunk_time: Duration of each chunk
        step_time: Step size for chunked processing
        plot: Whether to generate plots
        conf: VapConfig object
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all audio files
    audio_files = sorted(audio_dir.glob(f"*{audio_extension}"))
    print(f"Found {len(audio_files)} audio files")
    
    # Load model once
    print("Loading model...")
    model, device = load_vap_model(state_dict_path=state_dict_path, conf=conf)
    print(f"Model loaded on {device}")
    
    # Process each audio file
    for audio_path in tqdm(audio_files, desc="Processing dataset"):
        try:
            output_path = output_dir / f"{audio_path.stem}.json"
            
            # Skip if already processed
            if output_path.exists():
                print(f"Skipping {audio_path.name} (already processed)")
                continue
            
            process_audio(
                audio_path=str(audio_path),
                model=model,
                device=device,
                chunk=chunk,
                chunk_time=chunk_time,
                step_time=step_time,
                output_path=str(output_path),
                plot=plot,
                verbose=False
            )
            
        except Exception as e:
            print(f"Error processing {audio_path.name}: {e}")
            continue
    
    print(f"Processing complete. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Process a dataset of audio files with VAP")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing audio files"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save outputs"
    )
    parser.add_argument(
        "-sd",
        "--state_dict",
        type=str,
        default="example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt",
        help="Path to VAP model state dict"
    )
    parser.add_argument(
        "-ext",
        "--extension",
        type=str,
        default=".wav",
        help="Audio file extension (default: .wav)"
    )
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Process audio in chunks"
    )
    parser.add_argument(
        "--chunk_time",
        type=float,
        default=30,
        help="Duration of each chunk"
    )
    parser.add_argument(
        "--step_time",
        type=float,
        default=5,
        help="Step size for chunked processing"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots for each audio file"
    )
    
    parser, _ = VapConfig.add_argparse_args(parser)
    args = parser.parse_args()
    
    conf = VapConfig.args_to_conf(args)
    
    process_dataset(
        audio_dir=args.input_dir,
        output_dir=args.output_dir,
        state_dict_path=args.state_dict,
        audio_extension=args.extension,
        chunk=args.chunk,
        chunk_time=args.chunk_time,
        step_time=args.step_time,
        plot=args.plot,
        conf=conf
    )