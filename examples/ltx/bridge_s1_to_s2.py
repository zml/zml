#!/usr/bin/env python3
"""Bridge: Stage 1 (Zig) → Stage 2 (Zig).

Loads Zig Stage 1 output (.bin files), unpatchifies, upsamples the video
latent 2x, creates Stage 2 input state with pre-drawn noise, and exports
a safetensors file for the Zig Stage 2 driver (denoise_e2e).

Inputs:
  --stage1-video   Zig Stage 1 video_latent.bin (raw bf16 [1, T_v1, 128])
  --stage1-audio   Zig Stage 1 audio_latent.bin (raw bf16 [1, T_a1, 128])
  --stage2-noise   Pre-drawn Stage 2 noise (stage2_noise.safetensors from M0)
  --meta           Pipeline metadata (pipeline_meta.json from M0)
  --output         Output safetensors for Zig Stage 2 driver

Usage (on GPU server):
  cd /root/repos/LTX-2
  uv run examples/ltx/bridge_s1_to_s2.py \\
      --stage1-video /root/mixed/stage1_out/video_latent.bin \\
      --stage1-audio /root/mixed/stage1_out/audio_latent.bin \\
      --stage2-noise /root/mixed/stage2_noise.safetensors \\
      --meta /root/mixed/pipeline_meta.json \\
      --output /root/mixed/stage2_inputs.safetensors

Output safetensors keys (same format as export_stage2_inputs.py):
  video_noise          bf16  [B, T_v2, 128]
  audio_noise          bf16  [B, T_a2, 128]
  video_clean_latent   bf16  [B, T_v2, 128]
  audio_clean_latent   bf16  [B, T_a2, 128]
  video_denoise_mask   f32   [B, T_v2, 1]
  audio_denoise_mask   f32   [B, T_a2, 1]
  video_positions      bf16  [B, 3, T_v2, 2]
  audio_positions      f32   [B, 1, T_a2, 2]
  v_context            bf16  [B, S, 4096]
  a_context            bf16  [B, S, 2048]
  video_latent         bf16  [B, T_v2, 128]   (noised, for validation)
  audio_latent         bf16  [B, T_a2, 128]   (noised, for validation)
"""

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.types import (
    AudioLatentShape,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
    VideoPixelShape,
)
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_pipelines.utils import VideoUpsampler, cleanup_memory, get_device


VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()  # (8, 32, 32)
VIDEO_LATENT_CHANNELS = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge: Zig Stage 1 → Stage 2 inputs")
    parser.add_argument("--stage1-video", type=Path, required=True,
                        help="Zig Stage 1 video_latent.bin")
    parser.add_argument("--stage1-audio", type=Path, required=True,
                        help="Zig Stage 1 audio_latent.bin")
    parser.add_argument("--stage2-noise", type=Path, required=True,
                        help="Pre-drawn Stage 2 noise (stage2_noise.safetensors)")
    parser.add_argument("--meta", type=Path, required=True,
                        help="Pipeline metadata (pipeline_meta.json)")
    parser.add_argument("--stage1-inputs", type=Path, default=None,
                        help="Stage 1 inputs safetensors (for text contexts). "
                             "If not provided, attempts to find it relative to --meta.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output safetensors for Zig Stage 2 driver")
    # Model paths (for upsample)
    parser.add_argument("--checkpoint", type=str,
                        default=str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser()))
    parser.add_argument("--spatial-upsampler", type=str,
                        default=str(Path("~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors").expanduser()))
    return parser.parse_args()


def load_raw_bf16(path: Path, shape: list[int]) -> torch.Tensor:
    """Load a raw binary file as bf16 tensor with given shape."""
    raw = np.fromfile(str(path), dtype=np.uint16)
    t = torch.from_numpy(raw.copy()).view(torch.bfloat16)
    expected_numel = 1
    for s in shape:
        expected_numel *= s
    assert t.numel() == expected_numel, (
        f"Shape mismatch: file has {t.numel()} elements, expected {expected_numel} for shape {shape}"
    )
    return t.reshape(shape)


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()

    device = get_device()
    dtype = torch.bfloat16

    # ========================================================================
    # Load metadata
    # ========================================================================
    with open(args.meta) as f:
        meta = json.load(f)

    num_frames = meta["num_frames"]
    height = meta["height"]
    width = meta["width"]
    fps = meta["frame_rate"]
    seed = meta["seed"]
    prompt = meta["prompt"]
    neg_prompt = meta["negative_prompt"]

    s1 = meta["stage1"]
    s2 = meta["stage2"]
    sigma_0 = s2["sigma_0"]

    print(f"Pipeline: {width}x{height}, {num_frames} frames @ {fps} fps, seed={seed}")
    print(f"Stage 1 latent: F={s1['f_lat']} H={s1['h_lat']} W={s1['w_lat']} → T_v1={s1['t_video']}, T_a={s1['t_audio']}")
    print(f"Stage 2 latent: F={s2['f_lat']} H={s2['h_lat']} W={s2['w_lat']} → T_v2={s2['t_video']}, T_a={s2['t_audio']}")

    # ========================================================================
    # Load Zig Stage 1 outputs
    # ========================================================================
    print("\n=== Loading Zig Stage 1 outputs ===")
    video_patchified = load_raw_bf16(
        args.stage1_video,
        [1, s1["t_video"], VIDEO_LATENT_CHANNELS],
    ).to(device)
    audio_patchified = load_raw_bf16(
        args.stage1_audio,
        [1, s1["t_audio"], VIDEO_LATENT_CHANNELS],
    ).to(device)
    print(f"  video: {list(video_patchified.shape)} {video_patchified.dtype}")
    print(f"  audio: {list(audio_patchified.shape)} {audio_patchified.dtype}")

    # ========================================================================
    # Unpatchify Stage 1 video: [1, T_v1, 128] → [1, 128, F, H_s1, W_s1]
    # ========================================================================
    print("\n=== Unpatchifying Stage 1 video ===")
    f_lat = s1["f_lat"]
    h_s1 = s1["h_lat"]
    w_s1 = s1["w_lat"]

    # VideoLatentPatchifier(patch_size=1): patchify is "b c f h w -> b (f h w) c"
    # So unpatchify is "b (f h w) c -> b c f h w"
    video_5d = video_patchified.reshape(1, f_lat, h_s1, w_s1, VIDEO_LATENT_CHANNELS)
    video_5d = video_5d.permute(0, 4, 1, 2, 3).contiguous()
    print(f"  Unpatchified video: {list(video_5d.shape)}")

    # ========================================================================
    # Upsample video latent 2x
    # ========================================================================
    print("\n=== Upsampling video latent 2x ===")
    upsampler = VideoUpsampler(args.checkpoint, args.spatial_upsampler, dtype, device)
    upscaled_video = upsampler(video_5d)
    del upsampler
    torch.cuda.synchronize()
    cleanup_memory()

    print(f"  Upscaled video: {list(upscaled_video.shape)}")
    assert upscaled_video.shape == (1, VIDEO_LATENT_CHANNELS, s2["f_lat"], s2["h_lat"], s2["w_lat"]), (
        f"Unexpected upscaled shape: {upscaled_video.shape}, "
        f"expected [1, 128, {s2['f_lat']}, {s2['h_lat']}, {s2['w_lat']}]"
    )

    # Save upscaled video as a reference for Zig upsampler validation
    upscaled_ref_path = args.output.parent / "upsampled_ref.safetensors"
    save_file(
        {"upscaled_video_latent": upscaled_video.detach().cpu().contiguous()},
        str(upscaled_ref_path),
    )
    print(f"  Saved upsampler reference: {upscaled_ref_path}")

    # ========================================================================
    # Unpatchify Stage 1 audio: [1, T_a, 128] → [1, 8, T_a, 16]
    # ========================================================================
    print("\n=== Unpatchifying Stage 1 audio ===")
    audio_channels = 8
    audio_mel_bins = 16
    t_a = s1["t_audio"]

    # AudioPatchifier(patch_size=1): patchify is "b c t f -> b t (c f)"
    # So unpatchify is "b t (c f) -> b c t f"
    audio_4d = audio_patchified.reshape(1, t_a, audio_channels, audio_mel_bins)
    audio_4d = audio_4d.permute(0, 2, 1, 3).contiguous()
    print(f"  Unpatchified audio: {list(audio_4d.shape)}")

    # ========================================================================
    # Create Stage 2 latent states (positions, masks, clean latents)
    # ========================================================================
    print("\n=== Creating Stage 2 states ===")

    stage_2_output_shape = VideoPixelShape(
        batch=1, frames=num_frames, width=width, height=height, fps=fps,
    )

    # Video tools + initial state (handles patchification + position computation)
    video_latent_shape = VideoLatentShape.from_pixel_shape(
        shape=stage_2_output_shape,
        latent_channels=VIDEO_LATENT_CHANNELS,
        scale_factors=VIDEO_SCALE_FACTORS,
    )
    video_patchifier = VideoLatentPatchifier(patch_size=1)
    video_tools = VideoLatentTools(
        patchifier=video_patchifier,
        target_shape=video_latent_shape,
        fps=fps,
    )
    video_state = video_tools.create_initial_state(device, dtype, upscaled_video)

    # Audio tools + initial state
    audio_latent_shape = AudioLatentShape.from_video_pixel_shape(stage_2_output_shape)
    audio_patchifier_obj = AudioPatchifier(patch_size=1)
    audio_tools = AudioLatentTools(
        patchifier=audio_patchifier_obj,
        target_shape=audio_latent_shape,
    )
    audio_state = audio_tools.create_initial_state(device, dtype, audio_4d)

    print(f"  video_clean_latent (patchified): {list(video_state.clean_latent.shape)}")
    print(f"  video_positions:                 {list(video_state.positions.shape)}")
    print(f"  video_denoise_mask:              {list(video_state.denoise_mask.shape)}")
    print(f"  audio_clean_latent (patchified): {list(audio_state.clean_latent.shape)}")
    print(f"  audio_positions:                 {list(audio_state.positions.shape)}")
    print(f"  audio_denoise_mask:              {list(audio_state.denoise_mask.shape)}")

    # ========================================================================
    # Load pre-drawn Stage 2 noise and compute noised latents
    # ========================================================================
    print("\n=== Loading pre-drawn Stage 2 noise ===")
    noise_tensors = load_file(str(args.stage2_noise))
    video_noise_s2 = noise_tensors["video_noise_s2"].to(device)
    audio_noise_s2 = noise_tensors["audio_noise_s2"].to(device)
    print(f"  video_noise_s2: {list(video_noise_s2.shape)} {video_noise_s2.dtype}")
    print(f"  audio_noise_s2: {list(audio_noise_s2.shape)} {audio_noise_s2.dtype}")

    assert video_noise_s2.shape == video_state.clean_latent.shape, (
        f"Video noise shape {video_noise_s2.shape} != clean latent shape {video_state.clean_latent.shape}"
    )
    assert audio_noise_s2.shape == audio_state.clean_latent.shape, (
        f"Audio noise shape {audio_noise_s2.shape} != clean latent shape {audio_state.clean_latent.shape}"
    )

    # noised = clean * (1 - mask * sigma) + noise * mask * sigma
    print(f"  Computing noised latents (sigma_0={sigma_0})...")
    v_mask_sigma = video_state.denoise_mask.float() * sigma_0
    video_noised = (
        video_state.clean_latent.float() * (1.0 - v_mask_sigma)
        + video_noise_s2.float() * v_mask_sigma
    ).to(dtype)

    a_mask_sigma = audio_state.denoise_mask.float() * sigma_0
    audio_noised = (
        audio_state.clean_latent.float() * (1.0 - a_mask_sigma)
        + audio_noise_s2.float() * a_mask_sigma
    ).to(dtype)

    print(f"  video_noised: {list(video_noised.shape)} {video_noised.dtype}")
    print(f"  audio_noised: {list(audio_noised.shape)} {audio_noised.dtype}")

    # ========================================================================
    # Load text contexts from Stage 1 inputs
    # ========================================================================
    print("\n=== Loading text contexts ===")
    s1_inputs_path = args.stage1_inputs
    if s1_inputs_path is None:
        # Try to find it relative to --meta
        s1_inputs_path = args.meta.parent / "stage1_inputs.safetensors"

    s1_inputs = load_file(str(s1_inputs_path))
    # Stage 2 uses positive context only (distilled, no CFG)
    v_context = s1_inputs["v_context_pos"]
    a_context = s1_inputs["a_context_pos"]
    print(f"  v_context: {list(v_context.shape)} {v_context.dtype}")
    print(f"  a_context: {list(a_context.shape)} {a_context.dtype}")

    # ========================================================================
    # Export Stage 2 inputs (same format as export_stage2_inputs.py)
    # ========================================================================
    print("\n=== Exporting Stage 2 inputs ===")

    tensors = {
        "video_latent": video_noised.detach().cpu().contiguous(),
        "audio_latent": audio_noised.detach().cpu().contiguous(),
        "video_noise": video_noise_s2.detach().cpu().contiguous(),
        "audio_noise": audio_noise_s2.detach().cpu().contiguous(),
        "video_clean_latent": video_state.clean_latent.detach().cpu().contiguous(),
        "audio_clean_latent": audio_state.clean_latent.detach().cpu().contiguous(),
        "video_denoise_mask": video_state.denoise_mask.detach().cpu().contiguous(),
        "audio_denoise_mask": audio_state.denoise_mask.detach().cpu().contiguous(),
        "video_positions": video_state.positions.detach().cpu().contiguous(),
        "audio_positions": audio_state.positions.detach().cpu().contiguous(),
        "v_context": v_context.contiguous(),
        "a_context": a_context.contiguous(),
    }

    s2_metadata = {
        "num_frames": str(num_frames),
        "height": str(height),
        "width": str(width),
        "fps": str(fps),
        "seed": str(seed),
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "sigma_0": str(sigma_0),
        "f_lat": str(s2["f_lat"]),
        "h_lat": str(s2["h_lat"]),
        "w_lat": str(s2["w_lat"]),
        "t_aud": str(audio_state.clean_latent.shape[1]),
        "video_latent_tokens": str(video_state.clean_latent.shape[1]),
        "audio_latent_tokens": str(audio_state.clean_latent.shape[1]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata=s2_metadata)

    print(f"\nSaved: {args.output}")
    print(f"  {len(tensors)} tensors, metadata: {json.dumps(s2_metadata, indent=2)}")
    for key, t in sorted(tensors.items()):
        print(f"  {key:30s}  {list(t.shape)}  {t.dtype}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("BRIDGE COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {args.output}")
    print(f"\nNext: run Zig Stage 2 denoiser:")
    print(f"  bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:denoise_e2e -- \\")
    print(f"      <distilled_ckpt> {args.output} <output_dir>/")


if __name__ == "__main__":
    main()
