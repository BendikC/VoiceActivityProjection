from argparse import ArgumentParser
from os.path import basename
import matplotlib.pyplot as plt
import torch
import torchaudio


from vap.model import VapGPT, VapConfig, load_older_state_dict
from vap.audio import load_waveform
from vap.utils import (
    batch_to_device,
    everything_deterministic,
    tensor_dict_to_json,
    write_json,
)
from vap.plot_utils import plot_stereo


everything_deterministic()
torch.manual_seed(0)


def step_extraction(
    waveform,
    model,
    device="cpu",
    context_time=20,
    step_time=5,
    vad_thresh=0.5,
    ipu_time=0.1,
    pbar=True,
    verbose=False,
):
    """
    Takes a waveform, the model, and extracts probability output in chunks with
    a specific context and step time. Concatenates the output accordingly and returns full waveform output.
    """

    n_samples = waveform.shape[-1]
    duration = round(n_samples / model.sample_rate, 2)

    chunk_time = context_time + step_time

    # Samples
    # context_samples = int(context_time * model.sample_rate)
    step_samples = int(step_time * model.sample_rate)
    chunk_samples = int(chunk_time * model.sample_rate)

    # Frames
    # context_frames = int(context_time * model.frame_hz)
    chunk_frames = int(chunk_time * model.frame_hz)
    step_frames = int(step_time * model.frame_hz)

    # Fold the waveform to get total chunks
    folds = waveform.unfold(
        dimension=-1, size=chunk_samples, step=step_samples
    ).permute(2, 0, 1, 3)
    print("folds: ", tuple(folds.shape))

    expected_frames = round(duration * model.frame_hz)
    n_folds = int((n_samples - chunk_samples) / step_samples + 1.0)
    total = (n_folds - 1) * step_samples + chunk_samples

    # First chunk
    # Use all extracted data. Does not overlap with anything prior.
    out = model.probs(folds[0].to(device))
    # OUT:
    # {
    #   "probs": probs,
    #   "vad": vad,
    #   "p_now": p_now,
    #   "p_future": p_future,
    #   "H": H,
    # }

    if pbar:
        from tqdm import tqdm

        pbar = tqdm(folds[1:], desc=f"Context: {context_time}s, step: {step_time}")
    else:
        pbar = folds[1:]
    # Iterate over all other folds
    # and add simply the new processed step
    for w in pbar:
        o = model.probs(w.to(device))
        out["vad"] = torch.cat([out["vad"], o["vad"][:, -step_frames:]], dim=1)
        out["p_now"] = torch.cat([out["p_now"], o["p_now"][:, -step_frames:]], dim=1)
        out["p_future"] = torch.cat(
            [out["p_future"], o["p_future"][:, -step_frames:]], dim=1
        )
        out["probs"] = torch.cat([out["probs"], o["probs"][:, -step_frames:]], dim=1)
        out["H"] = torch.cat([out["H"], o["H"][:, -step_frames:]], dim=1)
        # out["p_zero_shot"] = torch.cat([out["p_zero_shot"], o["p_zero_shot"][:, -step_frames:]], dim=1)

    processed_frames = out["p_now"].shape[1]

    ###################################################################
    # Handle LAST SEGMENT (not included in `unfold`)
    ###################################################################
    if expected_frames != processed_frames:
        omitted_frames = expected_frames - processed_frames

        omitted_samples = model.sample_rate * omitted_frames / model.frame_hz

        if verbose:
            print(f"Expected frames {expected_frames} != {processed_frames}")
            print(f"omitted frames: {omitted_frames}")
            print(f"omitted samples: {omitted_samples}")
            print(f"chunk_samples: {chunk_samples}")

        w = waveform[..., -chunk_samples:]
        o = model.probs(w.to(device))
        out["vad"] = torch.cat([out["vad"], o["vad"][:, -omitted_frames:]], dim=1)
        out["p_now"] = torch.cat([out["p_now"], o["p_now"][:, -omitted_frames:]], dim=1)
        out["p_future"] = torch.cat(
            [out["p_future"], o["p_future"][:, -omitted_frames:]], dim=1
        )
        out["probs"] = torch.cat([out["probs"], o["probs"][:, -omitted_frames:]], dim=1)
        out["H"] = torch.cat([out["H"], o["H"][:, -omitted_frames:]], dim=1)

    # ###################################################################
    # # Extract Vad-list over entire vad
    # ###################################################################
    # out["vad_list"] = vad_output_to_vad_list(
    #     out["vad"],
    #     frame_hz=model.frame_hz,
    #     vad_thresh=vad_thresh,
    #     ipu_thresh_time=ipu_time,
    # )
    out = batch_to_device(out, "cpu")  # to cpu for plot/save
    return out


def load_vap_model(state_dict_path=None, checkpoint_path=None, conf=None, device=None):
    """
    Load VAP model from state dict or checkpoint.
    Handles both:
    - Official pretrained .pt files (pure state_dict)
    - Lightning .ckpt files (contains 'state_dict' key)
    
    Args:
        state_dict_path: Path to state dict file (.pt or .ckpt)
        checkpoint_path: Path to Lightning checkpoint (deprecated, use state_dict_path)
        conf: VapConfig object (required)
        device: Device to load model on ('cpu', 'cuda', or None for auto-detect)
    
    Returns:
        tuple: (model, device)
    """
    if checkpoint_path is not None:
        print("WARNING: checkpoint_path is deprecated, use state_dict_path instead")
        state_dict_path = checkpoint_path
    
    if state_dict_path is None:
        raise ValueError("Must provide state_dict_path")
    
    print("From state-dict: ", state_dict_path)
    
    # Register OptConfig in case checkpoint references it
    try:
        from vap.train_config import OptConfig
        import sys
        sys.modules['__main__'].OptConfig = OptConfig
    except ImportError:
        # If train_config doesn't exist, create a dummy class
        from dataclasses import dataclass
        
        @dataclass
        class OptConfig:
            learning_rate: float = 3.63e-4
            find_learning_rate: bool = False
            betas = [0.9, 0.999]
            weight_decay: float = 0.001
            lr_scheduler_interval: str = "step"
            lr_scheduler_freq: int = 100
            lr_scheduler_tmax: int = 2500
            lr_scheduler_patience: int = 2
            lr_scheduler_factor: float = 0.5
            early_stopping: bool = True
            patience: int = 10
            monitor: str = "val_loss"
            mode: str = "min"
        
        import sys
        sys.modules['__main__'].OptConfig = OptConfig
    
    # Load checkpoint with proper settings
    ckpt = torch.load(
        state_dict_path, 
        map_location="cpu",  # Always load to CPU first
        weights_only=False    # Allow loading Lightning checkpoints
    )
    
    # Handle different checkpoint formats
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        # Lightning checkpoint format
        print("Detected Lightning checkpoint, extracting 'state_dict'")
        sd = ckpt["state_dict"]
    else:
        # Pure state_dict format (official pretrained models)
        sd = ckpt
    
    # Clean up state dict keys
    new_sd = {}
    for k, v in sd.items():
        # Remove "model." prefix if present (from Lightning)
        if k.startswith("model."):
            k = k[len("model."):]
        
        # Skip zero_shot parameters (used in training, not inference)
        if k.startswith("zero_shot."):
            continue
        
        new_sd[k] = v
    
    # Load model
    model = VapGPT(conf)
    
    # Load with strict=False to handle missing/unexpected keys gracefully
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    
    if len(missing) > 0:
        print(f"Warning: {len(missing)} missing keys in state_dict")
        if len(missing) < 10:  # Only print if not too many
            for k in missing:
                print(f"  - {k}")
    
    if len(unexpected) > 0:
        print(f"Warning: {len(unexpected)} unexpected keys in state_dict")
        if len(unexpected) < 10:
            for k in unexpected:
                print(f"  - {k}")
    
    # Move to device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    model = model.eval()
    
    print(f"Model loaded on {device}")
    
    return model, device


def process_audio(
    audio_path,
    model,
    device="cpu",
    chunk=False,
    chunk_time=30,
    step_time=5,
    output_path=None,
    plot=False,
    verbose=True
):
    """
    Process a single audio file with VAP model.
    
    Args:
        audio_path: Path to audio file
        model: VAP model
        device: Device to run inference on
        chunk: Whether to process in chunks
        chunk_time: Duration of each chunk
        step_time: Step size for chunked processing
        output_path: Path to save JSON output (optional)
        plot: Whether to generate and save plot
        verbose: Whether to print progress
        downsample_hz: Target sample rate for output (e.g., 100 for 100 samples/sec)
    
    Returns:
        dict: Model output containing 'vad', 'p_now', 'p_future', 'probs', 'H'
    """
    # Load audio
    waveform, _ = load_waveform(audio_path, sample_rate=model.sample_rate)
    duration = round(waveform.shape[-1] / model.sample_rate)
    
    if waveform.shape[0] == 1:
        waveform = torch.cat((waveform, torch.zeros_like(waveform)))
    waveform = waveform.unsqueeze(0)
    
    # Check if chunking is needed
    if duration > 160 and not chunk:
        if verbose:
            print(f"WARNING: Audio duration {duration}s > 160s, enabling chunking")
        chunk = True
    
    # Model forward
    if chunk:
        out = step_extraction(
            waveform, 
            model, 
            device, 
            context_time=chunk_time - step_time,
            step_time=step_time,
            pbar=True  # Force pbar=True
        )
    else:
        waveform = waveform.to(device)
        out = model.probs(waveform)
        out = batch_to_device(out, "cpu")
    
    
    # Print shapes
    if verbose:
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: ", tuple(v.shape))
    
    # Save output
    if output_path is not None:
        if not output_path.endswith(".json"):
            output_path += ".json"
        
        data_to_save = {k: v for k, v in out.items()}
        data = tensor_dict_to_json(data_to_save)
        write_json(data, output_path)
        if verbose:
            print("Saved output -> ", output_path)
    
    # Plot
    if plot:
        vad = out["vad"][0].cpu()
        p_ns = out["p_now"][0, :, 0].cpu()
        fig, ax = plot_stereo(
            waveform[0].cpu(), p_ns, vad, plot=False, figsize=(100, 6)
        )
        
        figpath = output_path.replace(".json", ".png") if output_path else "output.png"
        fig.savefig(figpath)
        if verbose:
            print(f"Saved figure as {figpath}")
        plt.close(fig)
    
    return out


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-a",
        "--audio",
        type=str,
        help="Path to waveform",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default=None,
        help="Path to waveform",
    )
    parser.add_argument(
        "-sd",
        "--state_dict",
        type=str,
        default="example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt",
        help="Path to state_dict",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model",
    )
    parser, _ = VapConfig.add_argparse_args(parser)
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Process the audio in chunks (longer > 164s on 24Gb GPU audio)",
    )
    parser.add_argument(
        "--chunk_time",
        type=float,
        default=30,
        help="Duration of each chunk processed by model",
    )
    parser.add_argument(
        "--step_time",
        type=float,
        default=5,
        help="Increment to process in a step",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Visualize output (matplotlib)"
    )
    args = parser.parse_args()

    conf = VapConfig.args_to_conf(args)
    return args, conf


if __name__ == "__main__":
    args, conf = get_args()

    # Load the model
    print("Load Model...")
    model, device = load_vap_model(
        state_dict_path=args.state_dict,
        checkpoint_path=args.checkpoint,
        conf=conf
    )

    # Generate output filename if not provided
    if args.filename is None:
        args.filename = basename(args.audio).replace(".wav", ".json")

    # Process audio
    out = process_audio(
        audio_path=args.audio,
        model=model,
        device=device,
        chunk=args.chunk,
        chunk_time=args.chunk_time,
        step_time=args.step_time,
        output_path=args.filename,
        plot=args.plot,
        verbose=True,
        downsample_hz=100
    )
    
    print("wavefile: ", args.audio)