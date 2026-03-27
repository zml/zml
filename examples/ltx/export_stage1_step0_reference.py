#!/usr/bin/env python3
"""Export Stage 1 step-0 velocity reference + optional full 30-step reference.

Two-phase approach to avoid OOM:
  Phase 1: Use full pipeline to encode prompts + capture initial LatentStates
           (no transformer loaded). Then free the pipeline to release gemma.
  Phase 2: Load transformer into freed VRAM, run 1 or 30 denoising steps,
           capture the step-0 conditional velocity for parity checking.

Usage:
    python export_stage1_step0_reference.py [--full] [output.safetensors]

    --full   Also run 30 steps with guidance disabled and save final latent.
             Without this flag, only step-0 velocity is captured (faster).

Requires: ltx_core, ltx_pipelines, torch, safetensors
"""

import sys
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
    euler_denoising_loop,
    multi_modal_guider_factory_denoising_func,
)
from ltx_pipelines.utils.helpers import post_process_latent


def cosine_sim(a, b):
    a_f, b_f = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("output", nargs="?", default="stage1_step0_reference.safetensors")
    p.add_argument("--full", action="store_true",
                   help="Run full 30-step cond-only loop and save final latent")
    p.add_argument("--inputs", default="/root/e2e_demo/stage1_inputs.safetensors",
                   help="Path to exported stage1 inputs (for sanity check)")
    return p.parse_args()


def main():
    args = parse_args()
    output_path = args.output
    run_full = args.full
    inputs_path = args.inputs

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

    # ======================================================================
    # Phase 1: Encode prompts + capture initial LatentStates
    #          (gemma loaded → ~24GB; no transformer yet)
    # ======================================================================
    print("Phase 1: encoding prompts + capturing initial states...")

    ctx_p, ctx_n = encode_prompts(
        [prompt, negative_prompt],
        pipeline.stage_1_model_ledger,
        enhance_first_prompt=False,
        enhance_prompt_image=None,
        enhance_prompt_seed=seed,
    )
    # Clone contexts to CPU so they survive pipeline deletion
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

    # Capture initial LatentStates via capture-only denoising loop
    # (no transformer loaded — denoise_audio_video with components just sets up states)
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
    print(f"  GPU before cleanup: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # ======================================================================
    # Phase 2: Free pipeline (releases gemma ~24GB), load transformer
    # ======================================================================
    print("\nPhase 2: freeing pipeline, loading transformer...")

    # Keep the model ledger alive — it has the transformer builder
    stage_1_ledger = pipeline.stage_1_model_ledger

    # Delete pipeline + all cached models
    del pipeline, ctx_p, ctx_n, stage_1_conditionings
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print(f"  GPU after cleanup: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Now load transformer (should fit — gemma is freed)
    transformer = stage_1_ledger.transformer()
    print(f"  Transformer loaded: {type(transformer).__name__}")
    print(f"  GPU after transformer: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Move contexts back to GPU
    v_context_pos = v_context_pos.to(device=device)
    a_context_pos = a_context_pos.to(device=device)
    v_context_neg = v_context_neg.to(device=device)
    a_context_neg = a_context_neg.to(device=device)

    # ======================================================================
    # Phase 3: Run denoising steps with captured states
    # ======================================================================
    # Guidance disabled: cfg=1, stg=0, mod=1 → combine = identity
    no_guidance_video = MultiModalGuiderParams(
        cfg_scale=1.0, stg_scale=0.0, rescale_scale=0.0,
        modality_scale=1.0, skip_step=0, stg_blocks=[28],
    )
    no_guidance_audio = MultiModalGuiderParams(
        cfg_scale=1.0, stg_scale=0.0, rescale_scale=0.0,
        modality_scale=1.0, skip_step=0, stg_blocks=[28],
    )

    # Full Stage 1 guidance (default pipeline parameters)
    full_guidance_video = MultiModalGuiderParams(
        cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, skip_step=0, stg_blocks=[28],
    )
    full_guidance_audio = MultiModalGuiderParams(
        cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, skip_step=0, stg_blocks=[28],
    )

    # Step-0 velocity capture uses no guidance (single conditional pass)
    # so we can compare raw velocity. Full 30-step uses real guidance.
    use_video_params = full_guidance_video if run_full else no_guidance_video
    use_audio_params = full_guidance_audio if run_full else no_guidance_audio

    denoise_fn = multi_modal_guider_factory_denoising_func(
        video_guider_factory=create_multimodal_guider_factory(
            params=use_video_params,
            negative_context=v_context_neg,
        ),
        audio_guider_factory=create_multimodal_guider_factory(
            params=use_audio_params,
            negative_context=a_context_neg,
        ),
        v_context=v_context_pos,
        a_context=a_context_pos,
        transformer=transformer,
    )

    # Hook velocity_model.forward to capture RAW velocity (before X0Model's
    # to_denoised conversion). This matches what Zig's forwardOutputProjection
    # produces: the velocity v, not the x0 prediction (x0 = sample - v*sigma).
    velocity_outputs = []
    original_vm_forward = transformer.velocity_model.forward

    def capturing_vm_forward(*args, **kwargs):
        result = original_vm_forward(*args, **kwargs)
        velocity_outputs.append(result)
        return result

    transformer.velocity_model.forward = capturing_vm_forward

    # Recover captured states
    video_state = captured_states["video"]
    audio_state = captured_states["audio"]
    sigmas_tensor = captured_states["sigmas"]

    # Sanity check: compare initial latent with saved inputs
    initial_video_latent = video_state.latent.detach().cpu().clone()
    initial_audio_latent = audio_state.latent.detach().cpu().clone()

    if Path(inputs_path).exists():
        print(f"\n── Sanity check: initial latent vs saved inputs ──")
        saved = load_file(inputs_path)
        if "video_latent" in saved:
            v_cos = cosine_sim(saved["video_latent"], initial_video_latent)
            v_max = (saved["video_latent"].float() - initial_video_latent.float()).abs().max().item()
            print(f"  video_latent: cos_sim={v_cos:.6f}  max_abs_diff={v_max:.6f}")
        if "audio_latent" in saved:
            a_cos = cosine_sim(saved["audio_latent"], initial_audio_latent)
            a_max = (saved["audio_latent"].float() - initial_audio_latent.float()).abs().max().item()
            print(f"  audio_latent: cos_sim={a_cos:.6f}  max_abs_diff={a_max:.6f}")

    # Run denoising steps (inference_mode to avoid autograd storing all 48 blocks' activations)
    num_steps = len(sigmas_tensor) - 1
    max_steps = num_steps if run_full else 1
    mode = "full 30-step" if run_full else "step-0 only"
    print(f"\nRunning {max_steps} denoising step(s) ({mode})...")

    torch.cuda.empty_cache()

    step0_video_vel = None
    step0_audio_vel = None

    intermediate_latents = {}
    with torch.inference_mode():
        for step_idx in range(max_steps):
            sigma = sigmas_tensor[step_idx]
            sigma_next = sigmas_tensor[step_idx + 1]
            print(f"  Step {step_idx + 1}/{max_steps}: sigma={sigma:.6f} -> {sigma_next:.6f}")

            velocity_outputs.clear()

            # Call denoise_fn — returns guided x0 predictions (denoised samples)
            result = denoise_fn(video_state, audio_state, sigmas_tensor, step_idx)

            # At step 0, capture the raw velocity from the 1st velocity_model call
            if step_idx == 0:
                print(f"    velocity_model calls this step: {len(velocity_outputs)}")
                if len(velocity_outputs) > 0:
                    first_output = velocity_outputs[0]
                    if isinstance(first_output, tuple) and len(first_output) >= 2:
                        step0_video_vel = first_output[0].detach().cpu().clone()
                        step0_audio_vel = first_output[1].detach().cpu().clone()
                        print(f"    Captured step-0 raw velocity: video={list(step0_video_vel.shape)} "
                              f"audio={list(step0_audio_vel.shape)}")
                    else:
                        print(f"    WARNING: unexpected velocity_model output type: {type(first_output)}")
                else:
                    print(f"    WARNING: No velocity_model calls captured!")

            # Apply the same pipeline as euler_denoising_loop:
            # 1. denoise_fn → guided x0 (already done above)
            # 2. post_process_latent → blend with clean using mask
            # 3. stepper.step → Euler update
            if isinstance(result, tuple) and len(result) == 2:
                denoised_v, denoised_a = result

                if step_idx < 3 or step_idx in (4, 14, 29):
                    _dv = denoised_v.float()
                    _da = denoised_a.float()
                    print(f"    denoise_fn x0: v range=[{_dv.min():.4f}, {_dv.max():.4f}] std={_dv.std():.4f}")
                    print(f"    denoise_fn x0: a range=[{_da.min():.4f}, {_da.max():.4f}] std={_da.std():.4f}")

                # post_process_latent: blend denoised with clean using mask
                denoised_v = post_process_latent(denoised_v, video_state.denoise_mask, video_state.clean_latent)
                denoised_a = post_process_latent(denoised_a, audio_state.denoise_mask, audio_state.clean_latent)

                # Euler step: sample + to_velocity(sample, sigma, denoised) * dt
                new_v_latent = stepper.step(video_state.latent, denoised_v, sigmas_tensor, step_idx)
                new_a_latent = stepper.step(audio_state.latent, denoised_a, sigmas_tensor, step_idx)

                object.__setattr__(video_state, "latent", new_v_latent)
                object.__setattr__(audio_state, "latent", new_a_latent)

                # Capture intermediate latents for per-step comparison
                if step_idx in (0, 4, 14):
                    step_label = {0: 1, 4: 5, 14: 15}[step_idx]
                    intermediate_latents[f"step{step_label}_video_latent"] = video_state.latent.detach().cpu().clone()
                    intermediate_latents[f"step{step_label}_audio_latent"] = audio_state.latent.detach().cpu().clone()

                if step_idx < 3 or step_idx in (4, 14, 29):
                    _vl = video_state.latent.float()
                    _al = audio_state.latent.float()
                    print(f"    after Euler: v_lat range=[{_vl.min():.4f}, {_vl.max():.4f}] std={_vl.std():.4f}")
                    print(f"    after Euler: a_lat range=[{_al.min():.4f}, {_al.max():.4f}] std={_al.std():.4f}")
            else:
                print(f"    WARNING: denoise_fn returned unexpected: {type(result)}")
                break

    # Capture final latent — handle both LatentState and plain Tensor
    def _get_latent(state):
        return state.latent if hasattr(state, "latent") else state
    final_video_latent = _get_latent(video_state).detach().cpu().clone()
    final_audio_latent = _get_latent(audio_state).detach().cpu().clone()

    transformer.velocity_model.forward = original_vm_forward

    # ── Save reference tensors ────────────────────────────────────────────
    tensors = {}
    tensors["initial_video_latent"] = initial_video_latent.contiguous()
    tensors["initial_audio_latent"] = initial_audio_latent.contiguous()

    if step0_video_vel is not None:
        tensors["step0_video_vel"] = step0_video_vel.contiguous()
        tensors["step0_audio_vel"] = step0_audio_vel.contiguous()

    if run_full:
        tensors["final_video_latent"] = final_video_latent.contiguous()
        tensors["final_audio_latent"] = final_audio_latent.contiguous()
        for k, v in intermediate_latents.items():
            tensors[k] = v.contiguous()

    print(f"\nSaving {len(tensors)} tensors to {output_path}:")
    for k, v in tensors.items():
        print(f"  {k}: {list(v.shape)} {v.dtype}")

    save_file(tensors, output_path)
    print(f"\nDone! Saved to {output_path}")


if __name__ == "__main__":
    main()
