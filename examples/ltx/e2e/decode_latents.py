"""Decode Zig-produced denoised latents into video + audio MP4.

Loads the raw binary latent files produced by denoise_e2e (Zig), unpatchifies
them back to standard latent-space layout, runs the video VAE decoder and
audio VAE decoder + vocoder, and muxes the result into an MP4 file.

Usage (on GPU server):
  cd /root/repos/LTX-2
  uv run examples/ltx/e2e/decode_latents.py \
      --inputs /root/e2e_demo/stage2_inputs.safetensors \
      --video-latent /root/e2e_demo/video_latent.bin \
      --audio-latent /root/e2e_demo/audio_latent.bin \
      --output /root/e2e_demo/output.mp4
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from safetensors import safe_open

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.types import Audio
from ltx_pipelines.utils import AudioDecoder, VideoDecoder, get_device
from ltx_pipelines.utils.media_io import encode_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode Zig latents to MP4")
    parser.add_argument("--inputs", type=Path, required=True,
                        help="Path to stage2_inputs.safetensors (for metadata)")
    parser.add_argument("--video-latent", type=Path, required=True,
                        help="Path to video_latent.bin from Zig denoiser")
    parser.add_argument("--audio-latent", type=Path, required=True,
                        help="Path to audio_latent.bin from Zig denoiser")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output MP4 path")
    # Model paths
    parser.add_argument("--checkpoint", type=str,
                        default=str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser()))
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for VAE decode (tiling noise)")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()

    device = get_device()
    dtype = torch.bfloat16

    # ========================================================================
    # Read metadata from the input safetensors
    # ========================================================================
    print("Reading metadata from input file...")
    with safe_open(str(args.inputs), framework="pt") as f:
        metadata = f.metadata()

    num_frames = int(metadata["num_frames"])
    height = int(metadata["height"])
    width = int(metadata["width"])
    fps = float(metadata["fps"])
    f_lat = int(metadata["f_lat"])
    h_lat = int(metadata["h_lat"])
    w_lat = int(metadata["w_lat"])
    t_aud = int(metadata["t_aud"])
    v_tokens = int(metadata["video_latent_tokens"])
    a_tokens = int(metadata["audio_latent_tokens"])

    print(f"  Resolution: {width}x{height}, {num_frames} frames @ {fps} fps")
    print(f"  Video latent: F={f_lat}, H={h_lat}, W={w_lat} → {v_tokens} tokens")
    print(f"  Audio latent: T={t_aud} tokens")

    # ========================================================================
    # Load raw binary latents from Zig
    # ========================================================================
    print("Loading Zig-produced latents...")

    # Video: bf16 [B=1, T_v, 128] → raw bytes = 1 * T_v * 128 * 2
    video_bytes = args.video_latent.read_bytes()
    video_flat = np.frombuffer(video_bytes, dtype=np.float16)  # bf16 read as fp16 for shape
    # Actually bf16 needs special handling — let's use torch
    video_latent_patchified = torch.frombuffer(bytearray(video_bytes), dtype=torch.bfloat16)
    video_latent_patchified = video_latent_patchified.reshape(1, v_tokens, 128)
    print(f"  video_latent_patchified: {list(video_latent_patchified.shape)}")

    audio_bytes = args.audio_latent.read_bytes()
    audio_latent_patchified = torch.frombuffer(bytearray(audio_bytes), dtype=torch.bfloat16)
    audio_latent_patchified = audio_latent_patchified.reshape(1, a_tokens, 128)
    print(f"  audio_latent_patchified: {list(audio_latent_patchified.shape)}")

    # ========================================================================
    # Unpatchify
    # ========================================================================
    print("Unpatchifying...")

    # Video: (B, F*H*W, 128) → (B, 128, F, H, W) — patch_size=(1,1,1)
    # rearrange: "b (f h w) (c p1 p2 p3) -> b c (f p1) (h p2) (w p3)"
    # with p1=p2=p3=1, c=128
    video_latent = rearrange(
        video_latent_patchified,
        "b (f h w) c -> b c f h w",
        f=f_lat, h=h_lat, w=w_lat, c=128,
    )
    print(f"  video_latent: {list(video_latent.shape)}")

    # Audio: (B, T_aud, 128) → (B, 8, T_aud, 16) — c=8, f=16
    audio_latent = rearrange(
        audio_latent_patchified,
        "b t (c f) -> b c t f",
        c=8, f=16,
    )
    print(f"  audio_latent: {list(audio_latent.shape)}")

    # Move to device
    video_latent = video_latent.to(device=device, dtype=dtype)
    audio_latent = audio_latent.to(device=device, dtype=dtype)

    # ========================================================================
    # Load VAE models and decode
    # ========================================================================
    print("Decoding video...")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    tiling_config = TilingConfig.default()

    video_decoder = VideoDecoder(args.checkpoint, dtype, device)
    decoded_video = video_decoder(video_latent, tiling_config, generator)
    print("  Video decoded.")

    print("Decoding audio...")
    audio_decoder = AudioDecoder(args.checkpoint, dtype, device)
    decoded_audio = audio_decoder(audio_latent)
    print(f"  Audio decoded: sample_rate={decoded_audio.sampling_rate}")

    # ========================================================================
    # Write MP4
    # ========================================================================
    print(f"Encoding MP4 to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
    encode_video(
        video=decoded_video,
        fps=fps,
        audio=decoded_audio,
        output_path=str(args.output),
        video_chunks_number=video_chunks_number,
    )
    print(f"Done! Output: {args.output}")


if __name__ == "__main__":
    main()
