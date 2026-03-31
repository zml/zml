#!/usr/bin/env python3
"""Export Stage 1 inputs for the Zig denoise_stage1 driver.

Captures the pre-denoising tensors from a real Stage 1 pipeline run:
  - video_noise, audio_noise (Gaussian noise)
  - video_denoise_mask, audio_denoise_mask
  - video_clean_latent, audio_clean_latent
  - video_positions, audio_positions
  - v_context_pos, a_context_pos (positive text context)
  - v_context_neg, a_context_neg (negative text context)

Saved as a single safetensors file.

Usage:
    python export_stage1_inputs.py [output.safetensors]

Requires: ltx_core, ltx_pipelines, torch, safetensors
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.types import VideoPixelShape
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils import (
    cleanup_memory,
    combined_image_conditionings,
    denoise_audio_video,
    encode_prompts,
)


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else "stage1_inputs.safetensors"

    # ── Pipeline setup ─────────────────────────────────────────────────────
    checkpoint_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser())
    distilled_lora_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled-lora-384.safetensors").expanduser())
    spatial_upsampler_path = str(Path("~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors").expanduser())
    gemma_root = str(Path("~/models/gemma-3-12b-it").expanduser())

    prompt = "A beautiful sunset over the ocean"
    negative_prompt = "blurry, out of focus"
    seed = 10
    height = 1024
    width = 1536
    num_frames = 121
    frame_rate = 24.0
    num_inference_steps = 30

    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=[
            LoraPathStrengthAndSDOps(
                path=distilled_lora_path,
                strength=0.8,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ],
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=[],
        quantization=None,
    )

    device = pipeline.device
    dtype = torch.bfloat16

    generator = torch.Generator(device=device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)
    stepper = EulerDiffusionStep()

    ctx_p, ctx_n = encode_prompts(
        [prompt, negative_prompt],
        pipeline.stage_1_model_ledger,
        enhance_first_prompt=False,
        enhance_prompt_image=None,
        enhance_prompt_seed=seed,
    )
    v_context_pos, a_context_pos = ctx_p.video_encoding, ctx_p.audio_encoding
    v_context_neg, a_context_neg = ctx_n.video_encoding, ctx_n.audio_encoding

    stage_1_output_shape = VideoPixelShape(
        batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=frame_rate
    )

    video_encoder = pipeline.stage_1_model_ledger.video_encoder()
    stage_1_conditionings = combined_image_conditionings(
        images=[],
        height=stage_1_output_shape.height,
        width=stage_1_output_shape.width,
        video_encoder=video_encoder,
        dtype=dtype,
        device=device,
    )
    del video_encoder
    torch.cuda.synchronize()
    cleanup_memory()

    sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=device)

    # ── Custom denoising_loop_fn that captures state objects and exits ────
    # We only need the initial state objects (noise, positions, masks, etc.),
    # NOT the denoised result. So our loop captures the states and returns
    # them immediately — no transformer needed, no denoising cost.
    captured = {}

    def capturing_denoising_loop(sigmas_arg, video_state, audio_state, stepper_arg):
        """Capture state objects and return immediately (no denoising)."""
        print(f"\n[capture] video_state type: {type(video_state).__name__}")
        print(f"[capture] video_state attrs: {[a for a in dir(video_state) if not a.startswith('_')]}")
        print(f"[capture] audio_state type: {type(audio_state).__name__}")
        print(f"[capture] audio_state attrs: {[a for a in dir(audio_state) if not a.startswith('_')]}")

        # Extract tensors from state objects.
        for attr_name in ["latent", "noise", "positions", "denoise_mask",
                          "clean_latent", "sample", "noised_latent"]:
            for prefix, state in [("video", video_state), ("audio", audio_state)]:
                val = getattr(state, attr_name, None)
                if val is not None and isinstance(val, torch.Tensor):
                    key = f"{prefix}_{attr_name}"
                    captured[key] = val.detach().cpu().clone()
                    print(f"[capture] {key}: {list(val.shape)} {val.dtype}")

        # Return states as-is — no denoising, just capture.
        return video_state, audio_state

    # ── Run the pipeline ──────────────────────────────────────────────────
    print(f"Running denoise_audio_video (capture-only, no denoising)...")

    video_state, audio_state = denoise_audio_video(
        output_shape=stage_1_output_shape,
        conditionings=stage_1_conditionings,
        noiser=noiser,
        sigmas=sigmas,
        stepper=stepper,
        denoising_loop_fn=capturing_denoising_loop,
        components=pipeline.pipeline_components,
        dtype=dtype,
        device=device,
    )

    torch.cuda.synchronize()
    cleanup_memory()

    # ── Assemble output tensors ───────────────────────────────────────────
    if not captured:
        print("ERROR: No tensors captured! Check the state object attributes above.")
        sys.exit(1)

    print(f"\nCaptured {len(captured)} tensors from state objects.")

    # Build the output dict with standardized names for the Zig driver.
    # The state object has `latent` (already noised = clean + noise * sigma),
    # NOT a separate `noise` tensor. So we export the noised latent directly
    # and the Zig driver can skip forwardNoiseInit.
    #
    # Exported tensors:
    #   video_latent, audio_latent          — initial noised latents (bf16)
    #   video_denoise_mask, audio_denoise_mask — denoise masks (f32)
    #   video_clean_latent, audio_clean_latent — clean latents for post-process (bf16)
    #   video_positions, audio_positions     — RoPE position indices
    #   v_context_pos, a_context_pos         — positive text context (bf16)
    #   v_context_neg, a_context_neg         — negative text context (bf16)

    tensors = {}

    # Noised latents (already combined: clean + noise * sigma)
    for prefix in ["video", "audio"]:
        latent_key = f"{prefix}_latent"
        if latent_key in captured:
            tensors[latent_key] = captured[latent_key]
        else:
            print(f"ERROR: {latent_key} not found in captured state!")

    # Denoise mask
    for prefix in ["video", "audio"]:
        mask_key = f"{prefix}_denoise_mask"
        if mask_key in captured:
            tensors[mask_key] = captured[mask_key]
        else:
            print(f"WARNING: {mask_key} not found in captured state!")

    # Clean latent (for post_process_latent in Euler step)
    for prefix in ["video", "audio"]:
        clean_key = f"{prefix}_clean_latent"
        if clean_key in captured:
            tensors[clean_key] = captured[clean_key]
        else:
            print(f"WARNING: {clean_key} not found in captured state!")

    # Positions
    for prefix in ["video", "audio"]:
        pos_key = f"{prefix}_positions"
        if pos_key in captured:
            tensors[pos_key] = captured[pos_key]
        else:
            print(f"WARNING: {pos_key} not found in captured state!")

    # Text contexts (from encode_prompts, not from state objects)
    tensors["v_context_pos"] = v_context_pos.detach().cpu().contiguous()
    tensors["a_context_pos"] = a_context_pos.detach().cpu().contiguous()
    tensors["v_context_neg"] = v_context_neg.detach().cpu().contiguous()
    tensors["a_context_neg"] = a_context_neg.detach().cpu().contiguous()

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\nSaving {len(tensors)} tensors to {output_path}:")
    for k, v in tensors.items():
        print(f"  {k}: {list(v.shape)} {v.dtype}")

    tensors = {k: v.contiguous() for k, v in tensors.items()}
    save_file(tensors, output_path)
    print(f"\nDone! Saved to {output_path}")

    # Also dump ALL captured keys for reference
    print(f"\nAll captured state keys: {sorted(captured.keys())}")


if __name__ == "__main__":
    main()
