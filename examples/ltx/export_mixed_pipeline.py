#!/usr/bin/env python3
"""Export mixed pipeline: reference boundaries + Stage 1 inputs + Stage 2 noise.

Single Python run that executes the full two-stage pipeline end-to-end,
capturing tensors at every boundary for validation, AND exporting the
Stage 1 inputs and Stage 2 noise needed by the mixed Zig pipeline.

Flow:
  text enc → Stage 1 denoise (30 steps, full guidance) → upsample →
  Stage 2 denoise (3 steps, distilled) → done

Outputs:
  {out}/stage1_inputs.safetensors     — Stage 1 inputs for Zig driver (14 tensors)
  {out}/stage2_noise.safetensors      — Pre-drawn Stage 2 noise for bridge script
  {out}/ref/stage1_outputs.safetensors — Stage 1 denoised latents (reference)
  {out}/ref/upsampled.safetensors      — Upscaled video latent (reference)
  {out}/ref/stage2_inputs.safetensors  — Stage 2 full inputs (reference, 12 tensors)
  {out}/ref/stage2_outputs.safetensors — Stage 2 denoised latents (reference)
  {out}/pipeline_meta.json             — Pipeline config metadata
  {out}/python_reference.mp4           — Full-Python video (with --decode-video)

Usage (on GPU server):
  cd /root/repos/LTX-2
  uv run examples/ltx/export_mixed_pipeline.py \
      --output-dir /root/mixed/ \
      --prompt "A beautiful sunset over the ocean" \
      --seed 10 \
      --decode-video
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from safetensors.torch import save_file

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.transformer import X0Model
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.types import LatentState, VideoPixelShape
from ltx_pipelines.utils import (
    AudioDecoder,
    DiffusionStage,
    FactoryGuidedDenoiser,
    ImageConditioner,
    ModalitySpec,
    PromptEncoder,
    SimpleDenoiser,
    VideoDecoder,
    VideoUpsampler,
    cleanup_memory,
    combined_image_conditionings,
    euler_denoising_loop,
    get_device,
)
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import Denoiser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export mixed pipeline: reference boundaries + Zig inputs"
    )
    # Generation params
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over the ocean")
    parser.add_argument("--negative-prompt", type=str, default="blurry, out of focus")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    # Output
    parser.add_argument("--output-dir", type=Path, required=True)
    # Model paths
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser()),
        help="Base model checkpoint (used for both Stage 1 and Stage 2)",
    )
    parser.add_argument(
        "--spatial-upsampler", type=str,
        default=str(Path("~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors").expanduser()),
    )
    parser.add_argument(
        "--gemma-root", type=str,
        default=str(Path("~/models/gemma-3-12b-it").expanduser()),
    )
    # Decode options
    parser.add_argument(
        "--decode-video", action="store_true",
        help="Decode final latents to MP4 (full-Python reference video)",
    )
    return parser.parse_args()


def save_safetensors(tensors: dict[str, torch.Tensor], path: Path,
                     metadata: dict[str, str] | None = None) -> None:
    """Save tensors to safetensors file, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
    save_file(clean, str(path), metadata=metadata)
    print(f"  Saved {path.name}: {len(clean)} tensors")
    for k, v in sorted(clean.items()):
        print(f"    {k:30s}  {list(v.shape)}  {v.dtype}")


def recover_noise(noised: torch.Tensor, clean: torch.Tensor,
                  mask: torch.Tensor, sigma_0: float) -> torch.Tensor:
    """Back-compute noise: noise = (noised - clean * (1 - mask*s)) / (mask*s)."""
    noised_f32 = noised.float()
    clean_f32 = clean.float()
    mask_f32 = mask.float()
    mask_sigma = mask_f32 * sigma_0
    one_minus = 1.0 - mask_sigma
    noise_f32 = torch.zeros_like(noised_f32)
    nonzero = mask_sigma.squeeze(-1) > 0
    if nonzero.any():
        noise_f32[nonzero] = (
            (noised_f32[nonzero] - clean_f32[nonzero] * one_minus[nonzero])
            / mask_sigma[nonzero]
        )
    return noise_f32.to(noised.dtype)


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()

    device = get_device()
    dtype = torch.bfloat16
    out = args.output_dir
    ref = out / "ref"

    print(f"Prompt: {args.prompt!r}")
    print(f"Resolution: {args.width}x{args.height}, {args.num_frames} frames @ {args.frame_rate} fps")
    print(f"Seed: {args.seed}")
    print(f"Output: {out}")

    # ========================================================================
    # Model setup
    # ========================================================================
    print("\n=== Loading models ===")

    prompt_encoder = PromptEncoder(
        args.checkpoint, args.gemma_root, dtype, device,
    )
    image_conditioner = ImageConditioner(args.checkpoint, dtype, device)
    upsampler = VideoUpsampler(args.checkpoint, args.spatial_upsampler, dtype, device)

    stage_1_diffusion = DiffusionStage(args.checkpoint, dtype, device)
    stage_2_diffusion = DiffusionStage(args.checkpoint, dtype, device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    noiser = GaussianNoiser(generator=generator)

    # ========================================================================
    # Text encoding
    # ========================================================================
    print("\n=== Text encoding ===")
    ctx_p, ctx_n = prompt_encoder(
        [args.prompt, args.negative_prompt],
        enhance_first_prompt=False,
        enhance_prompt_image=None,      # no image-based prompt enhancement in this pipeline. Might add in the future as an option.
        enhance_prompt_seed=args.seed,
    )
    v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
    v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

    print(f"  v_context_pos: {list(v_context_p.shape)} {v_context_p.dtype}")
    print(f"  a_context_pos: {list(a_context_p.shape)} {a_context_p.dtype}")
    print(f"  v_context_neg: {list(v_context_n.shape)} {v_context_n.dtype}")
    print(f"  a_context_neg: {list(a_context_n.shape)} {a_context_n.dtype}")

    # ========================================================================
    # Stage 1 setup
    # ========================================================================
    print("\n=== Stage 1: half-resolution denoising ===")
    stage_1_output_shape = VideoPixelShape(
        batch=1,
        frames=args.num_frames,
        width=args.width // 2,
        height=args.height // 2,
        fps=args.frame_rate,
    )

    stage_1_conditionings = image_conditioner(
        lambda enc: combined_image_conditionings(
            images=[],                  # no image-based prompt enhancement in this pipeline. Might add in the future as an option.
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=enc,
            dtype=dtype,
            device=device,
        )
    )

    sigmas = LTX2Scheduler().execute(steps=args.num_inference_steps).to(
        dtype=torch.float32, device=device,
    )

    # LTX-2.3 guidance params (matching denoise_stage1.zig)
    video_guider_params = MultiModalGuiderParams(
        cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, skip_step=0, stg_blocks=[28],
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, skip_step=0, stg_blocks=[28],
    )

    # ========================================================================
    # Stage 1 denoising (with capture of initial states)
    # ========================================================================
    captured_s1 = {}

    def stage1_loop(
        sigmas: torch.Tensor,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        stepper: DiffusionStepProtocol,
        transformer: X0Model,
        denoiser: Denoiser,
    ) -> tuple[LatentState | None, LatentState | None]:
        # Capture the initial patchified states (before any denoising)
        captured_s1["video"] = video_state.clone()
        captured_s1["audio"] = audio_state.clone()
        print(f"  [capture] Stage 1 initial states:")
        print(f"    video_latent: {list(video_state.latent.shape)} {video_state.latent.dtype}")
        print(f"    audio_latent: {list(audio_state.latent.shape)} {audio_state.latent.dtype}")
        print(f"    video_mask:   {list(video_state.denoise_mask.shape)} {video_state.denoise_mask.dtype}")
        print(f"    video_pos:    {list(video_state.positions.shape)} {video_state.positions.dtype}")

        return euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            transformer=transformer,
            denoiser=denoiser,
        )

    print("  Running Stage 1 denoising (this will take a while)...")
    video_state_s1, audio_state_s1 = stage_1_diffusion(
        denoiser=FactoryGuidedDenoiser(
            v_context=v_context_p,
            a_context=a_context_p,
            video_guider_factory=create_multimodal_guider_factory(
                params=video_guider_params,
                negative_context=v_context_n,
            ),
            audio_guider_factory=create_multimodal_guider_factory(
                params=audio_guider_params,
                negative_context=a_context_n,
            ),
        ),
        sigmas=sigmas,
        noiser=noiser,
        width=stage_1_output_shape.width,
        height=stage_1_output_shape.height,
        frames=args.num_frames,
        fps=args.frame_rate,
        video=ModalitySpec(context=v_context_p, conditionings=stage_1_conditionings),
        audio=ModalitySpec(context=a_context_p),
        loop=stage1_loop,
    )

    # NOTE: video_state_s1.latent is UNPATCHIFIED [B, 128, F, H, W] after
    # denoise_audio_video returns (it calls unpatchify internally).
    print(f"  Stage 1 done.")
    print(f"    video_latent (unpatchified): {list(video_state_s1.latent.shape)}")
    print(f"    audio_latent (unpatchified): {list(audio_state_s1.latent.shape)}")

    # ========================================================================
    # Export: Stage 1 inputs (for Zig driver)
    # ========================================================================
    print("\n=== Exporting Stage 1 inputs ===")
    vs1 = captured_s1["video"]
    as1 = captured_s1["audio"]

    s1_inputs = {
        "video_latent": vs1.latent,
        "audio_latent": as1.latent,
        "video_denoise_mask": vs1.denoise_mask,
        "audio_denoise_mask": as1.denoise_mask,
        "video_clean_latent": vs1.clean_latent,
        "audio_clean_latent": as1.clean_latent,
        "video_positions": vs1.positions,
        "audio_positions": as1.positions,
        "v_context_pos": v_context_p,
        "a_context_pos": a_context_p,
        "v_context_neg": v_context_n,
        "a_context_neg": a_context_n,
    }
    save_safetensors(s1_inputs, out / "stage1_inputs.safetensors")

    # ========================================================================
    # Export: Stage 1 reference outputs
    # ========================================================================
    print("\n=== Exporting Stage 1 reference outputs ===")
    save_safetensors(
        {
            "video_latent_denoised": video_state_s1.latent,
            "audio_latent_denoised": audio_state_s1.latent,
        },
        ref / "stage1_outputs.safetensors",
    )

    # ========================================================================
    # Upsample video latent
    # ========================================================================
    print("\n=== Upsampling video latent 2x ===")
    upscaled_video_latent = upsampler(video_state_s1.latent[:1])
    print(f"  Upscaled: {list(upscaled_video_latent.shape)}")

    save_safetensors(
        {"upscaled_video_latent": upscaled_video_latent},
        ref / "upsampled.safetensors",
    )

    # ========================================================================
    # Stage 2 setup
    # ========================================================================
    print("\n=== Stage 2: full-resolution refinement ===")

    stage_2_conditionings = image_conditioner(
        lambda enc: combined_image_conditionings(
            images=[],
            height=args.height,
            width=args.width,
            video_encoder=enc,
            dtype=dtype,
            device=device,
        )
    )

    distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(device)

    # ========================================================================
    # Stage 2 denoising (with capture of initial states)
    # ========================================================================
    captured_s2 = {}

    def stage2_loop(
        sigmas: torch.Tensor,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        stepper: DiffusionStepProtocol,
        transformer: X0Model,
        denoiser: Denoiser,
    ) -> tuple[LatentState | None, LatentState | None]:
        # Capture the initial patchified states (before any denoising)
        captured_s2["video"] = video_state.clone()
        captured_s2["audio"] = audio_state.clone()
        print(f"  [capture] Stage 2 initial states:")
        print(f"    video_latent: {list(video_state.latent.shape)} {video_state.latent.dtype}")
        print(f"    audio_latent: {list(audio_state.latent.shape)} {audio_state.latent.dtype}")

        return euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            transformer=transformer,
            denoiser=denoiser,
        )

    print("  Running Stage 2 denoising...")
    video_state_s2, audio_state_s2 = stage_2_diffusion(
        denoiser=SimpleDenoiser(v_context=v_context_p, a_context=a_context_p),
        sigmas=distilled_sigmas,
        noiser=noiser,
        width=args.width,
        height=args.height,
        frames=args.num_frames,
        fps=args.frame_rate,
        video=ModalitySpec(
            context=v_context_p,
            conditionings=stage_2_conditionings,
            noise_scale=distilled_sigmas[0].item(),
            initial_latent=upscaled_video_latent,
        ),
        audio=ModalitySpec(
            context=a_context_p,
            noise_scale=distilled_sigmas[0].item(),
            initial_latent=audio_state_s1.latent,
        ),
        loop=stage2_loop,
    )

    print(f"  Stage 2 done.")
    print(f"    video_latent (unpatchified): {list(video_state_s2.latent.shape)}")
    print(f"    audio_latent (unpatchified): {list(audio_state_s2.latent.shape)}")

    # ========================================================================
    # Export: Stage 2 noise (for bridge script)
    # ========================================================================
    print("\n=== Exporting Stage 2 noise ===")
    vs2 = captured_s2["video"]
    as2 = captured_s2["audio"]
    sigma_0 = distilled_sigmas[0].item()

    video_noise_s2 = recover_noise(vs2.latent, vs2.clean_latent, vs2.denoise_mask, sigma_0)
    audio_noise_s2 = recover_noise(as2.latent, as2.clean_latent, as2.denoise_mask, sigma_0)

    save_safetensors(
        {
            "video_noise_s2": video_noise_s2,
            "audio_noise_s2": audio_noise_s2,
        },
        out / "stage2_noise.safetensors",
    )

    # ========================================================================
    # Export: Stage 2 reference inputs (same format as export_stage2_inputs.py)
    # ========================================================================
    print("\n=== Exporting Stage 2 reference inputs ===")

    f_lat = (args.num_frames - 1) // 8 + 1
    h_lat = args.height // 32
    w_lat = args.width // 32

    s2_ref_inputs = {
        "video_latent": vs2.latent,
        "audio_latent": as2.latent,
        "video_denoise_mask": vs2.denoise_mask,
        "audio_denoise_mask": as2.denoise_mask,
        "video_clean_latent": vs2.clean_latent,
        "audio_clean_latent": as2.clean_latent,
        "video_positions": vs2.positions,
        "audio_positions": as2.positions,
        "v_context": v_context_p,
        "a_context": a_context_p,
        "video_noise": video_noise_s2,
        "audio_noise": audio_noise_s2,
    }
    s2_metadata = {
        "num_frames": str(args.num_frames),
        "height": str(args.height),
        "width": str(args.width),
        "fps": str(args.frame_rate),
        "seed": str(args.seed),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "sigma_0": str(sigma_0),
        "f_lat": str(f_lat),
        "h_lat": str(h_lat),
        "w_lat": str(w_lat),
        "t_aud": str(as2.latent.shape[1]),
        "video_latent_tokens": str(vs2.latent.shape[1]),
        "audio_latent_tokens": str(as2.latent.shape[1]),
    }
    save_safetensors(s2_ref_inputs, ref / "stage2_inputs.safetensors", metadata=s2_metadata)

    # ========================================================================
    # Export: Stage 2 reference outputs
    # ========================================================================
    print("\n=== Exporting Stage 2 reference outputs ===")
    save_safetensors(
        {
            "video_latent_final": video_state_s2.latent,
            "audio_latent_final": audio_state_s2.latent,
        },
        ref / "stage2_outputs.safetensors",
    )

    # ========================================================================
    # Export: Pipeline metadata
    # ========================================================================
    print("\n=== Exporting pipeline metadata ===")

    # Stage 1 latent shapes
    f_lat_s1 = (args.num_frames - 1) // 8 + 1
    h_lat_s1 = (args.height // 2) // 32
    w_lat_s1 = (args.width // 2) // 32
    t_v1 = f_lat_s1 * h_lat_s1 * w_lat_s1
    t_a1 = captured_s1["audio"].latent.shape[1]

    # Stage 2 latent shapes
    t_v2 = f_lat * h_lat * w_lat
    t_a2 = captured_s2["audio"].latent.shape[1]

    meta = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "frame_rate": args.frame_rate,
        "num_inference_steps": args.num_inference_steps,
        "checkpoint": args.checkpoint,
        "spatial_upsampler": args.spatial_upsampler,
        "stage1": {
            "f_lat": f_lat_s1,
            "h_lat": h_lat_s1,
            "w_lat": w_lat_s1,
            "t_video": t_v1,
            "t_audio": t_a1,
            "sigmas": sigmas.cpu().tolist(),
        },
        "stage2": {
            "f_lat": f_lat,
            "h_lat": h_lat,
            "w_lat": w_lat,
            "t_video": t_v2,
            "t_audio": t_a2,
            "sigma_0": sigma_0,
            "sigmas": distilled_sigmas.cpu().tolist(),
        },
        "guidance": {
            "video": {
                "cfg_scale": 3.0, "stg_scale": 1.0, "rescale_scale": 0.7,
                "modality_scale": 3.0, "stg_blocks": [28],
            },
            "audio": {
                "cfg_scale": 7.0, "stg_scale": 1.0, "rescale_scale": 0.7,
                "modality_scale": 3.0, "stg_blocks": [28],
            },
        },
    }

    meta_path = out / "pipeline_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved {meta_path}")

    # ========================================================================
    # Decode to MP4 (optional, full-Python reference video)
    # ========================================================================
    python_video_path = None
    if args.decode_video:
        print("\n=== Decoding full-Python reference video ===")
        tiling_config = TilingConfig.default()
        decode_generator = torch.Generator(device=device).manual_seed(args.seed)

        video_decoder = VideoDecoder(args.checkpoint, dtype, device)
        decoded_video = video_decoder(video_state_s2.latent, tiling_config, decode_generator)
        print("  Video decoded.")

        audio_decoder = AudioDecoder(args.checkpoint, dtype, device)
        decoded_audio = audio_decoder(audio_state_s2.latent)
        print(f"  Audio decoded: sample_rate={decoded_audio.sampling_rate}")

        python_video_path = out / "python_reference.mp4"
        python_video_path.parent.mkdir(parents=True, exist_ok=True)
        video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
        encode_video(
            video=decoded_video,
            fps=args.frame_rate,
            audio=decoded_audio,
            output_path=str(python_video_path),
            video_chunks_number=video_chunks_number,
        )
        print(f"  Full-Python reference video: {python_video_path}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print(f"\nStage 1 inputs (for Zig):  {out / 'stage1_inputs.safetensors'}")
    print(f"Stage 2 noise (for bridge): {out / 'stage2_noise.safetensors'}")
    print(f"Pipeline metadata:          {out / 'pipeline_meta.json'}")
    if python_video_path:
        print(f"Full-Python reference video: {python_video_path}")
    print(f"\nReference tensors:")
    print(f"  Stage 1 outputs: {ref / 'stage1_outputs.safetensors'}")
    print(f"  Upsampled:       {ref / 'upsampled.safetensors'}")
    print(f"  Stage 2 inputs:  {ref / 'stage2_inputs.safetensors'}")
    print(f"  Stage 2 outputs: {ref / 'stage2_outputs.safetensors'}")
    print(f"\nNext steps:")
    print(f"  M1: bazel run //examples/ltx:denoise_stage1 -- \\")
    print(f"        <base_ckpt> {out / 'stage1_inputs.safetensors'} {out / 'stage1_out/'}")
    print(f"  M2: python bridge_s1_to_s2.py \\")
    print(f"        --stage1-out {out / 'stage1_out/'} \\")
    print(f"        --stage2-noise {out / 'stage2_noise.safetensors'} \\")
    print(f"        --meta {out / 'pipeline_meta.json'} \\")
    print(f"        --output {out / 'stage2_inputs.safetensors'}")


if __name__ == "__main__":
    main()
