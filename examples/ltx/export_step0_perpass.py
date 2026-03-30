#!/usr/bin/env python3
"""Export per-pass step-0 reference for detailed parity diagnosis.

Captures at step 0:
  - Pass 1 (conditional) raw velocity + x0
  - Pass 2 (negative/CFG) raw velocity + x0
  - Pass 3 (STG/perturbed) raw velocity + x0
  - Pass 4 (isolated modality) raw velocity + x0
  - Guided x0 (after guider combine, per modality)
  - Blended (after post_process_latent)
  - Step-1 latent (after Euler step)

Two-phase approach to avoid OOM (same as export_stage1_step0_reference.py):
  Phase 1: encode prompts + capture LatentStates, then free pipeline (gemma).
  Phase 2: load transformer, run 1 step with full guidance, capture everything.

Usage:
    python export_step0_perpass.py [--inputs PATH] [output.safetensors]

Requires: ltx_core, ltx_pipelines, torch, safetensors
"""

import sys
import gc
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

from ltx_core.components.guiders import (
    MultiModalGuider,
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
    p.add_argument("output", nargs="?", default="step0_perpass_reference.safetensors")
    p.add_argument("--inputs", default="/root/e2e_demo/stage1_inputs.safetensors",
                   help="Path to exported stage1 inputs (for sanity check)")
    return p.parse_args()


def main():
    args = parse_args()
    output_path = args.output
    inputs_path = args.inputs

    # ── Pipeline setup (identical to other export scripts) ──────────────
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

    # ── Phase 1: Encode prompts + capture LatentStates ──────────────────
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
    print(f"  Captured LatentStates: video={list(captured_states['video'].latent.shape)}, "
          f"audio={list(captured_states['audio'].latent.shape)}")

    # ── Phase 2: Free pipeline, load transformer ────────────────────────
    print("\nPhase 2: freeing pipeline, loading transformer...")
    stage_1_ledger = pipeline.stage_1_model_ledger
    del pipeline, ctx_p, ctx_n, stage_1_conditionings
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print(f"  GPU after cleanup: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    transformer = stage_1_ledger.transformer()
    print(f"  Transformer loaded: {type(transformer).__name__}")
    print(f"  GPU after transformer: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    v_context_pos = v_context_pos.to(device=device)
    a_context_pos = a_context_pos.to(device=device)
    v_context_neg = v_context_neg.to(device=device)
    a_context_neg = a_context_neg.to(device=device)

    # ── Phase 3: Run step 0 with per-pass capture ───────────────────────
    # Hook velocity_model.forward to capture raw velocities for each pass
    velocity_captures = []  # list of (video_vel, audio_vel) tuples
    original_vm_forward = transformer.velocity_model.forward

    def capturing_vm_forward(*args, **kwargs):
        result = original_vm_forward(*args, **kwargs)
        velocity_captures.append((
            result[0].detach().cpu().clone() if result[0] is not None else None,
            result[1].detach().cpu().clone() if result[1] is not None else None,
        ))
        return result

    transformer.velocity_model.forward = capturing_vm_forward

    # Hook guider.calculate to capture per-modality inputs and output
    guider_captures = {"video": {}, "audio": {}}
    original_calculate = MultiModalGuider.calculate

    def capturing_calculate(self, cond, uncond_text, uncond_perturbed, uncond_modality):
        result = original_calculate(self, cond, uncond_text, uncond_perturbed, uncond_modality)
        # Determine which modality this is by checking shape
        # video: [1, 6144, 128], audio: [1, 126, 128]
        if cond.shape[1] > 1000:
            mod = "video"
        else:
            mod = "audio"
        guider_captures[mod]["cond_x0"] = cond.detach().cpu().clone()
        guider_captures[mod]["neg_x0"] = uncond_text.detach().cpu().clone() if isinstance(uncond_text, torch.Tensor) else None
        guider_captures[mod]["ptb_x0"] = uncond_perturbed.detach().cpu().clone() if isinstance(uncond_perturbed, torch.Tensor) else None
        guider_captures[mod]["iso_x0"] = uncond_modality.detach().cpu().clone() if isinstance(uncond_modality, torch.Tensor) else None
        guider_captures[mod]["guided_x0"] = result.detach().cpu().clone()
        return result

    MultiModalGuider.calculate = capturing_calculate

    # Full guidance params
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

    # Sanity check
    if Path(inputs_path).exists():
        saved = load_file(inputs_path)
        print(f"\n── Sanity check ──")
        if "video_latent" in saved:
            diff = (saved["video_latent"].float() - video_state.latent.detach().cpu().float()).abs().max().item()
            print(f"  video_latent max_diff={diff:.6f}")
        if "audio_latent" in saved:
            diff = (saved["audio_latent"].float() - audio_state.latent.detach().cpu().float()).abs().max().item()
            print(f"  audio_latent max_diff={diff:.6f}")

    print(f"\nRunning step 0 with full guidance (4 passes)...")
    torch.cuda.empty_cache()

    with torch.inference_mode():
        step_idx = 0
        sigma = sigmas_tensor[step_idx]
        sigma_next = sigmas_tensor[step_idx + 1]
        print(f"  sigma={sigma:.6f} -> {sigma_next:.6f}")

        velocity_captures.clear()

        # denoise_fn returns guided x0 (after X0Model + guider combine)
        result = denoise_fn(video_state, audio_state, sigmas_tensor, step_idx)

        print(f"  velocity_model calls: {len(velocity_captures)}")
        for i, (vv, av) in enumerate(velocity_captures):
            pass_name = ["cond", "neg", "ptb", "iso"][i] if i < 4 else f"pass{i}"
            vshape = list(vv.shape) if vv is not None else None
            ashape = list(av.shape) if av is not None else None
            print(f"    {pass_name}: video={vshape} audio={ashape}")

        denoised_v, denoised_a = result

        # post_process_latent
        blended_v = post_process_latent(denoised_v, video_state.denoise_mask, video_state.clean_latent)
        blended_a = post_process_latent(denoised_a, audio_state.denoise_mask, audio_state.clean_latent)

        # Euler step
        new_v = stepper.step(video_state.latent, blended_v, sigmas_tensor, step_idx)
        new_a = stepper.step(audio_state.latent, blended_a, sigmas_tensor, step_idx)

    # Restore originals
    transformer.velocity_model.forward = original_vm_forward
    MultiModalGuider.calculate = original_calculate

    # ── Assemble output ───────────────────────────────────────────────────
    tensors = {}

    # Per-pass raw velocities (from velocity_model, before to_denoised)
    pass_names = ["cond", "neg", "ptb", "iso"]
    for i, name in enumerate(pass_names):
        if i < len(velocity_captures):
            vv, av = velocity_captures[i]
            if vv is not None:
                tensors[f"step0_{name}_video_vel"] = vv.contiguous()
            if av is not None:
                tensors[f"step0_{name}_audio_vel"] = av.contiguous()

    # Per-pass x0 (from guider.calculate inputs)
    for mod in ["video", "audio"]:
        for key in ["cond_x0", "neg_x0", "ptb_x0", "iso_x0", "guided_x0"]:
            val = guider_captures[mod].get(key)
            if val is not None:
                tensors[f"step0_{mod}_{key}"] = val.contiguous()

    # Blended (after post_process_latent)
    tensors["step0_video_blended"] = blended_v.detach().cpu().contiguous()
    tensors["step0_audio_blended"] = blended_a.detach().cpu().contiguous()

    # Step-1 latent (after Euler)
    tensors["step1_video_latent"] = new_v.detach().cpu().contiguous()
    tensors["step1_audio_latent"] = new_a.detach().cpu().contiguous()

    print(f"\nSaving {len(tensors)} tensors to {output_path}:")
    for k, v in sorted(tensors.items()):
        print(f"  {k}: {list(v.shape)} {v.dtype}")

    save_file(tensors, output_path)
    print(f"\nDone! Saved to {output_path}")


if __name__ == "__main__":
    main()
