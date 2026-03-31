#!/usr/bin/env python3
"""Export all 30 intermediate latents from a full Python denoising run.

Saves per-step input latents so the Zig reset test can feed Python's
trajectory into the Zig pipeline step-by-step and measure per-step error
in isolation (no accumulation).

Output tensor naming:
  v_lat_0, a_lat_0     = initial noised latent (input to step 0)
  v_lat_1, a_lat_1     = latent after step 0 Euler update (input to step 1)
  ...
  v_lat_30, a_lat_30   = final latent (after step 29 Euler update)

Usage:
    python export_all_step_latents.py [--inputs PATH] [output.safetensors]
"""

import gc
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

from ltx_core.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
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
    multi_modal_guider_factory_denoising_func,
)
from ltx_pipelines.utils.helpers import post_process_latent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("output", nargs="?", default="all_step_latents.safetensors")
    p.add_argument("--inputs", default="/root/e2e_demo/stage1_inputs.safetensors")
    return p.parse_args()


def main():
    args = parse_args()
    output_path = args.output

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

    # Phase 1: encode prompts + capture states
    print("Phase 1: encoding prompts + capturing initial states...")

    ctx_p, ctx_n = encode_prompts(
        [prompt, negative_prompt],
        pipeline.stage_1_model_ledger,
        enhance_first_prompt=False,
        enhance_prompt_image=None,
        enhance_prompt_seed=seed,
    )
    v_context_pos = ctx_p.video_encoding.detach().cpu().clone()
    a_context_pos = ctx_p.audio_encoding.detach().cpu().clone()
    v_context_neg = ctx_n.video_encoding.detach().cpu().clone()
    a_context_neg = ctx_n.audio_encoding.detach().cpu().clone()

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

    captured_states = {}
    def capture_loop(sigmas_arg, video_state, audio_state, stepper_arg):
        captured_states["video"] = video_state.clone()
        captured_states["audio"] = audio_state.clone()
        captured_states["sigmas"] = sigmas_arg.clone()
        return video_state, audio_state

    denoise_audio_video(
        output_shape=stage_1_output_shape,
        conditionings=stage_1_conditionings,
        noiser=noiser,
        sigmas=sigmas,
        stepper=stepper,
        denoising_loop_fn=capture_loop,
        components=pipeline.pipeline_components,
        dtype=dtype,
        device=device,
    )

    # Phase 2: free pipeline, load transformer
    print("\nPhase 2: freeing pipeline, loading transformer...")
    stage_1_ledger = pipeline.stage_1_model_ledger
    del pipeline, ctx_p, ctx_n, stage_1_conditionings
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    transformer = stage_1_ledger.transformer()
    print(f"  Transformer loaded: {type(transformer).__name__}")

    v_context_pos = v_context_pos.to(device=device)
    a_context_pos = a_context_pos.to(device=device)
    v_context_neg = v_context_neg.to(device=device)
    a_context_neg = a_context_neg.to(device=device)

    # Phase 3: run 30 steps, save all latents
    full_guidance_video = MultiModalGuiderParams(
        cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, skip_step=0, stg_blocks=[28],
    )
    full_guidance_audio = MultiModalGuiderParams(
        cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, skip_step=0, stg_blocks=[28],
    )

    denoise_fn = multi_modal_guider_factory_denoising_func(
        video_guider_factory=create_multimodal_guider_factory(
            params=full_guidance_video,
            negative_context=v_context_neg,
        ),
        audio_guider_factory=create_multimodal_guider_factory(
            params=full_guidance_audio,
            negative_context=a_context_neg,
        ),
        v_context=v_context_pos,
        a_context=a_context_pos,
        transformer=transformer,
    )

    video_state = captured_states["video"]
    audio_state = captured_states["audio"]
    sigmas_tensor = captured_states["sigmas"]
    num_steps = len(sigmas_tensor) - 1

    tensors = {}

    # Save initial latent (input to step 0)
    tensors["v_lat_0"] = video_state.latent.detach().cpu().clone().contiguous()
    tensors["a_lat_0"] = audio_state.latent.detach().cpu().clone().contiguous()

    print(f"\nRunning {num_steps} steps, saving all intermediate latents...")
    torch.cuda.empty_cache()

    with torch.inference_mode():
        for step_idx in range(num_steps):
            sigma = sigmas_tensor[step_idx]
            sigma_next = sigmas_tensor[step_idx + 1]
            print(f"  Step {step_idx + 1}/{num_steps}: sigma={sigma:.6f} -> {sigma_next:.6f}")

            result = denoise_fn(video_state, audio_state, sigmas_tensor, step_idx)
            denoised_v, denoised_a = result

            denoised_v = post_process_latent(denoised_v, video_state.denoise_mask, video_state.clean_latent)
            denoised_a = post_process_latent(denoised_a, audio_state.denoise_mask, audio_state.clean_latent)

            new_v = stepper.step(video_state.latent, denoised_v, sigmas_tensor, step_idx)
            new_a = stepper.step(audio_state.latent, denoised_a, sigmas_tensor, step_idx)

            object.__setattr__(video_state, "latent", new_v)
            object.__setattr__(audio_state, "latent", new_a)

            # Save latent after this step (= input to next step)
            tensors[f"v_lat_{step_idx + 1}"] = new_v.detach().cpu().clone().contiguous()
            tensors[f"a_lat_{step_idx + 1}"] = new_a.detach().cpu().clone().contiguous()

    print(f"\nSaving {len(tensors)} tensors to {output_path}:")
    for k in sorted(tensors.keys()):
        v = tensors[k]
        print(f"  {k}: {list(v.shape)} {v.dtype}")

    save_file(tensors, output_path)
    print(f"\nDone! Saved to {output_path}")


if __name__ == "__main__":
    main()
