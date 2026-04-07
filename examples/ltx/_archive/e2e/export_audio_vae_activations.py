"""Export audio VAE decoder activations for Zig validation.

Runs the AudioDecoder on a known latent, saves input + decoded output as
safetensors for comparison with the Zig implementation.

Usage (on GPU server):
  cd /root/repos/LTX-2
  uv run examples/ltx/e2e/export_audio_vae_activations.py \
      --checkpoint /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
      --output-dir /root/e2e_demo/audio_vae_ref/

To use the real denoised audio latent:
  uv run examples/ltx/e2e/export_audio_vae_activations.py \
      --checkpoint /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
      --audio-latent /root/e2e_demo/unified_pipeline/unified/audio_latent.bin \
      --meta /root/e2e_demo/pipeline_meta.json \
      --output-dir /root/e2e_demo/audio_vae_ref/
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import numpy as np
from einops import rearrange
from safetensors.torch import save_file

from ltx_core.model.audio_vae import AudioDecoderConfigurator
from ltx_core.model.audio_vae.vocoder import Vocoder
from ltx_pipelines.utils.blocks import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    Builder,
    DummyRegistry,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Export audio VAE decoder activations")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--audio-latent", type=str, default=None,
                   help="Path to audio_latent.bin from Zig denoiser (bf16)")
    p.add_argument("--meta", type=str, default=None,
                   help="Path to pipeline_meta.json for t_aud dimension")
    p.add_argument("--small-test", action="store_true",
                   help="Use a small synthetic test input [1, 8, 4, 16]")
    p.add_argument("--medium-test", action="store_true",
                   help="Use a medium synthetic test input [1, 8, 32, 16]")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for synthetic tests (default: 42)")
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Build audio latent input
    # ========================================================================
    if args.small_test:
        torch.manual_seed(args.seed)
        log.info("Using small synthetic test input [1, 8, 4, 16]")
        audio_latent = torch.randn(1, 8, 4, 16, dtype=dtype, device=device)
    elif args.medium_test:
        torch.manual_seed(args.seed)
        log.info("Using medium synthetic test input [1, 8, 32, 16]")
        audio_latent = torch.randn(1, 8, 32, 16, dtype=dtype, device=device)
    elif args.audio_latent is not None:
        log.info(f"Loading audio latent from {args.audio_latent}")
        audio_bytes = Path(args.audio_latent).read_bytes()
        audio_flat = torch.frombuffer(bytearray(audio_bytes), dtype=torch.bfloat16)

        # Get t_aud from metadata
        if args.meta:
            with open(args.meta) as f:
                meta = json.load(f)
            t_aud = meta["stage2"]["t_audio"]
        else:
            # Infer from size: total elements = 1 * t_aud * 128
            t_aud = len(audio_flat) // 128
        log.info(f"  t_aud={t_aud}, total elements={len(audio_flat)}")

        # Unpatchify: [1, T, 128] → [1, 8, T, 16]
        audio_patchified = audio_flat.reshape(1, t_aud, 128)
        audio_latent = rearrange(audio_patchified, "b t (c f) -> b c t f", c=8, f=16)
        audio_latent = audio_latent.to(device=device)
        log.info(f"  audio_latent: {list(audio_latent.shape)}")
    else:
        raise ValueError("Provide --audio-latent or --small-test")

    # ========================================================================
    # Load audio VAE decoder
    # ========================================================================
    log.info("Loading AudioDecoder...")
    builder = Builder(
        model_path=args.checkpoint,
        model_class_configurator=AudioDecoderConfigurator,
        model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
        registry=DummyRegistry(),
    )
    decoder = builder.build(device=device, dtype=dtype).to(device).eval()

    # Save per-channel statistics
    pcs = decoder.per_channel_statistics
    mean_of_means = getattr(pcs, "mean-of-means").cpu()
    std_of_means = getattr(pcs, "std-of-means").cpu()
    log.info(f"  mean-of-means: {list(mean_of_means.shape)}")
    log.info(f"  std-of-means: {list(std_of_means.shape)}")

    # ========================================================================
    # Run decoder
    # ========================================================================
    log.info("Running audio VAE decoder...")
    decoded = decoder(audio_latent)
    log.info(f"  Decoded: {list(decoded.shape)} {decoded.dtype}")

    # ========================================================================
    # Save activations
    # ========================================================================
    tensors = {
        "input_latent": audio_latent.cpu().contiguous(),
        "decoded_output": decoded.cpu().contiguous(),
        "per_channel_mean_of_means": mean_of_means.contiguous(),
        "per_channel_std_of_means": std_of_means.contiguous(),
    }

    out_path = out_dir / "audio_vae_activations.safetensors"
    save_file(tensors, str(out_path))
    log.info(f"  Saved {out_path}")

    # Print summary
    log.info("Summary:")
    for name, t in tensors.items():
        log.info(f"  {name}: {list(t.shape)} {t.dtype}")


if __name__ == "__main__":
    main()
