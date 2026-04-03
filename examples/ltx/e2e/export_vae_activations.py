"""Export video VAE decoder activations for Zig validation.

Loads a video latent (from a previous Zig run or generates a small random one),
runs the video VAE decoder with hooks to capture intermediates at each layer
boundary, and saves all activations as safetensors.

Usage (on GPU server):
  cd /root/repos/LTX-2
  uv run /root/repos/zml/examples/ltx/e2e/export_vae_activations.py \
      --checkpoint /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
      --video-latent /root/e2e_demo/unified_out/video_latent.bin \
      --meta /root/e2e_demo/pipeline_meta.json \
      --output-dir /root/e2e_demo/vae_ref/

  # Or with a small random latent for unit testing:
  uv run /root/repos/zml/examples/ltx/e2e/export_vae_activations.py \
      --checkpoint /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
      --small-test \
      --output-dir /root/e2e_demo/vae_ref_small/
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from einops import rearrange
from safetensors.torch import save_file

from ltx_core.model.video_vae.model_configurator import (
    VideoDecoderConfigurator,
    VAE_DECODER_COMFY_KEYS_FILTER,
)
from ltx_core.model.video_vae.video_vae import UNetMidBlock3D, ResnetBlock3D
from ltx_pipelines.utils.blocks import Builder, DummyRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export VAE decoder activations")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to ltx-2.3-22b-distilled.safetensors")
    parser.add_argument("--video-latent", type=Path, default=None,
                        help="Path to video_latent.bin (patchified bf16 [1, T_v, 128])")
    parser.add_argument("--meta", type=Path, default=None,
                        help="Path to pipeline_meta.json (for latent dims)")
    parser.add_argument("--small-test", action="store_true",
                        help="Generate a small random latent (F'=2, H'=4, W'=4) for unit testing")
    parser.add_argument("--medium-test", action="store_true",
                        help="Generate a medium random latent (F'=8, H'=16, W'=16) for scale testing")
    parser.add_argument("--output-only", action="store_true",
                        help="Only capture input+output (skip intermediates) to reduce peak memory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for activation safetensors")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_decoder(checkpoint_path: Path, device, dtype):
    """Load the video VAE decoder from checkpoint using the standard Builder."""
    builder = Builder(
        model_path=str(checkpoint_path),
        model_class_configurator=VideoDecoderConfigurator,
        model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
        registry=DummyRegistry(),
    )
    decoder = builder.build(device=device, dtype=dtype).to(device).eval()

    # Extract per-channel statistics from the loaded decoder
    pcs = decoder.per_channel_statistics
    per_channel_stats = {
        "mean-of-means": getattr(pcs, "mean-of-means").to(device=device, dtype=dtype),
        "std-of-means": getattr(pcs, "std-of-means").to(device=device, dtype=dtype),
    }
    print(f"  Decoder loaded: {sum(p.numel() for p in decoder.parameters())} params")
    print(f"  timestep_conditioning: {decoder.timestep_conditioning}")
    print(f"  causal: {decoder.causal}")
    print(f"  up_blocks: {len(decoder.up_blocks)}")
    return decoder, per_channel_stats


def run_decoder_with_hooks(decoder, latent_5d, per_channel_stats, device, dtype) -> dict:
    """Run the decoder step-by-step, capturing intermediates."""
    activations = {}
    causal = decoder.causal

    # Save input
    activations["input_latent"] = latent_5d.clone()

    # Step 1: Denormalize (un_normalize)
    mean_of_means = per_channel_stats["mean-of-means"]
    std_of_means = per_channel_stats["std-of-means"]
    mean = mean_of_means.view(1, -1, 1, 1, 1)
    std_ = std_of_means.view(1, -1, 1, 1, 1)
    x = latent_5d * std_ + mean
    activations["after_denorm"] = x.clone()

    # Step 2: conv_in
    x = decoder.conv_in(x, causal=causal)
    activations["after_conv_in"] = x.clone()

    # Step 3-11: up_blocks.0 through up_blocks.8
    for i, up_block in enumerate(decoder.up_blocks):
        if isinstance(up_block, UNetMidBlock3D):
            x = up_block(x, causal=causal, timestep=None, generator=None)
        elif isinstance(up_block, ResnetBlock3D):
            x = up_block(x, causal=causal, generator=None)
        else:
            x = up_block(x, causal=causal)
        activations[f"after_up{i}"] = x.clone()

    # Step 12: conv_norm_out + conv_act (PixelNorm + SiLU)
    x = decoder.conv_norm_out(x)
    x = decoder.conv_act(x)
    activations["after_norm_silu"] = x.clone()

    # Step 13: conv_out
    x = decoder.conv_out(x, causal=causal)
    activations["after_conv_out"] = x.clone()

    # Step 14: unpatchify
    # "b (c p r q) f h w -> b c (f p) (h q) (w r)" with p=1, q=4, r=4
    x = rearrange(x, "b (c p r q) f h w -> b c (f p) (h q) (w r)", p=1, q=4, r=4)
    activations["output"] = x.clone()

    return activations


def run_decoder_output_only(decoder, latent_5d, per_channel_stats, device, dtype) -> dict:
    """Run the decoder end-to-end, capturing only input and output (low peak memory)."""
    activations = {}
    causal = decoder.causal

    activations["input_latent"] = latent_5d.clone()

    # Denormalize
    mean = per_channel_stats["mean-of-means"].view(1, -1, 1, 1, 1)
    std_ = per_channel_stats["std-of-means"].view(1, -1, 1, 1, 1)
    x = latent_5d * std_ + mean

    # conv_in
    x = decoder.conv_in(x, causal=causal)

    # up_blocks
    for up_block in decoder.up_blocks:
        if isinstance(up_block, UNetMidBlock3D):
            x = up_block(x, causal=causal, timestep=None, generator=None)
        elif isinstance(up_block, ResnetBlock3D):
            x = up_block(x, causal=causal, generator=None)
        else:
            x = up_block(x, causal=causal)

    # PixelNorm + SiLU + conv_out
    x = decoder.conv_norm_out(x)
    x = decoder.conv_act(x)
    x = decoder.conv_out(x, causal=causal)

    # unpatchify
    x = rearrange(x, "b (c p r q) f h w -> b c (f p) (h q) (w r)", p=1, q=4, r=4)
    activations["output"] = x

    return activations


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Device: {device}, dtype: {dtype}")

    # ========================================================================
    # Prepare input latent (5D: [B, C, F', H', W'])
    # ========================================================================
    if args.small_test:
        # Small random latent for fast iteration
        f_lat, h_lat, w_lat = 2, 4, 4
        print(f"Generating small random latent: F'={f_lat}, H'={h_lat}, W'={w_lat}")
        torch.manual_seed(args.seed)
        latent_5d = torch.randn(1, 128, f_lat, h_lat, w_lat, device=device, dtype=dtype)
    elif args.medium_test:
        # Medium random latent — produces [1, 3, 57, 512, 512] decoded video
        f_lat, h_lat, w_lat = 8, 16, 16
        print(f"Generating medium random latent: F'={f_lat}, H'={h_lat}, W'={w_lat}")
        torch.manual_seed(args.seed)
        latent_5d = torch.randn(1, 128, f_lat, h_lat, w_lat, device=device, dtype=dtype)
    else:
        if args.video_latent is None or args.meta is None:
            raise ValueError("--video-latent and --meta are required unless --small-test is used")

        # Load metadata
        with open(args.meta) as f:
            meta = json.load(f)
        f_lat = meta["stage2"]["f_lat"]
        h_lat = meta["stage2"]["h_lat"]
        w_lat = meta["stage2"]["w_lat"]
        v_tokens = f_lat * h_lat * w_lat

        print(f"Loading video latent: F'={f_lat}, H'={h_lat}, W'={w_lat} ({v_tokens} tokens)")

        # Load raw binary latent [1, T_v, 128] bf16
        video_bytes = args.video_latent.read_bytes()
        latent_patchified = torch.frombuffer(bytearray(video_bytes), dtype=torch.bfloat16)
        latent_patchified = latent_patchified.reshape(1, v_tokens, 128).to(device=device)

        # Unpatchify: [1, T_v, 128] → [1, 128, F', H', W']
        from einops import rearrange
        latent_5d = rearrange(
            latent_patchified,
            "b (f h w) c -> b c f h w",
            f=f_lat, h=h_lat, w=w_lat, c=128,
        )

    print(f"  latent_5d shape: {list(latent_5d.shape)}")

    # ========================================================================
    # Load decoder
    # ========================================================================
    print("Loading video VAE decoder...")
    decoder, per_channel_stats = load_decoder(args.checkpoint, device, dtype)
    print("  Decoder loaded.")

    # ========================================================================
    # Run with hooks
    # ========================================================================
    if args.output_only:
        print("Running decoder (output-only mode, no intermediates)...")
        activations = run_decoder_output_only(decoder, latent_5d, per_channel_stats, device, dtype)
    else:
        print("Running decoder with activation capture...")
        activations = run_decoder_with_hooks(decoder, latent_5d, per_channel_stats, device, dtype)

    # Print shapes
    print("\nCaptured activations:")
    for key, tensor in activations.items():
        size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
        print(f"  {key:20s}: {str(list(tensor.shape)):30s} ({size_mb:.1f} MB)")

    # ========================================================================
    # Save
    # ========================================================================
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "vae_activations.safetensors"

    # Move all to CPU for saving
    save_dict = {k: v.contiguous().cpu() for k, v in activations.items()}

    # Also save per-channel stats
    save_dict["per_channel_mean_of_means"] = per_channel_stats["mean-of-means"].cpu()
    save_dict["per_channel_std_of_means"] = per_channel_stats["std-of-means"].cpu()

    # Save metadata
    metadata = {
        "f_lat": str(f_lat),
        "h_lat": str(h_lat),
        "w_lat": str(w_lat),
    }

    save_file(save_dict, str(output_path), metadata=metadata)
    print(f"\nSaved activations to {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
