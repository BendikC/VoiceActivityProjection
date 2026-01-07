from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import torch

from run import load_vap_model, process_audio
from vap.model import VapConfig


def process_single_dataset(
    audio_dir,
    output_dir,
    model,
    device,
    audio_extension=".wav",
    chunk=True,
    chunk_time=30,
    step_time=5,
    plot=False
):
    """
    Process all audio files in a single directory.
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all audio files
    audio_files = sorted(audio_dir.glob(f"*{audio_extension}"))
    
    if len(audio_files) == 0:
        print(f"  No audio files found in {audio_dir}")
        return 0
    
    processed_count = 0
    
    # Process each audio file
    for audio_path in audio_files:
        try:
            output_path = output_dir / f"{audio_path.stem}.json"
            
            # Skip if already processed
            if output_path.exists():
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
            
            processed_count += 1
            
        except Exception as e:
            print(f"  ❌ Error processing {audio_path.name}: {e}")
            continue
    
    return processed_count


def process_dataset(
    audio_dir,
    output_dir,
    state_dict_path,
    audio_extension=".wav",
    chunk=True,
    chunk_time=30,
    step_time=5,
    plot=False,
    conf=None,
    recursive=False
):
    """
    Process audio files with VAP model.
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model once
    print("🔧 Loading model...")
    model, device = load_vap_model(state_dict_path=state_dict_path, conf=conf)
    print(f"✅ Model loaded on {device}")
    
    if recursive:
        # Process subdirectories
        subdirs = [d for d in audio_dir.iterdir() if d.is_dir()]
        
        if len(subdirs) == 0:
            print(f"❌ No subdirectories found in {audio_dir}")
            print(f"   Tip: Use --recursive flag only when processing nested folders")
            return
        
        print(f"\n📁 Found {len(subdirs)} subdirectories to process")
        print(f"🎯 Input:  {audio_dir}")
        print(f"💾 Output: {output_dir}")
        print("="*70)
        
        total_processed = 0
        
        for subdir in subdirs:
            print(f"\n📂 Processing: {subdir.name}")
            
            # Create corresponding output subdirectory
            output_subdir = output_dir / subdir.name
            
            # Count files
            audio_files = list(subdir.glob(f"*{audio_extension}"))
            print(f"   Files: {len(audio_files)}")
            
            # Process this subdirectory
            processed = process_single_dataset(
                audio_dir=subdir,
                output_dir=output_subdir,
                model=model,
                device=device,
                audio_extension=audio_extension,
                chunk=chunk,
                chunk_time=chunk_time,
                step_time=step_time,
                plot=plot
            )
            
            total_processed += processed
            print(f"   ✅ Processed: {processed}/{len(audio_files)}")
        
        print("\n" + "="*70)
        print(f"🎉 Complete! Total files processed: {total_processed}")
        print(f"💾 Results saved to: {output_dir}")
        
    else:
        # Process single directory (original behavior)
        audio_files = sorted(audio_dir.glob(f"*{audio_extension}"))
        
        if len(audio_files) == 0:
            print(f"❌ No audio files found in {audio_dir}")
            print(f"   Tip: If you have subdirectories, use --recursive flag")
            return
        
        print(f"\n📊 Found {len(audio_files)} audio files")
        print(f"🎯 Input:  {audio_dir}")
        print(f"💾 Output: {output_dir}")
        print("="*70)
        
        # Process each audio file with progress bar
        processed_count = 0
        skipped_count = 0
        
        for audio_path in tqdm(audio_files, desc="Processing"):
            try:
                output_path = output_dir / f"{audio_path.stem}.json"
                
                # Skip if already processed
                if output_path.exists():
                    skipped_count += 1
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
                
                processed_count += 1
                
            except Exception as e:
                print(f"\n❌ Error processing {audio_path.name}: {e}")
                continue
        
        print("="*70)
        print(f"✅ Processed: {processed_count} files")
        if skipped_count > 0:
            print(f"⏭️  Skipped: {skipped_count} files (already processed)")
        print(f"💾 Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Process audio files with VAP")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing audio files or subdirectories"
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
        "--recursive",
        action="store_true",
        help="Process subdirectories recursively (maintains folder structure in output)"
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
        conf=conf,
        recursive=args.recursive
    )