"""Export Step 3 parity fixture — full 3-step denoising loop.

Captures the full stage-2 denoising loop (3 Euler steps) starting from
the initial noisy latent states, running velocity_model.forward at each
step, applying to_denoised + post_process_latent + Euler step, and saving
all intermediate and final latent states for parity validation.

Sigma schedule: [0.909375, 0.725, 0.421875, 0.0] → 3 Euler steps.

Saved keys:
  # Initial state (inputs to the denoising loop)
  init.video_latent              bf16  [B, T_v, 128]
  init.audio_latent              bf16  [B, T_a, 128]
  init.video_denoise_mask        f32   [B, T_v, 1]
  init.audio_denoise_mask        f32   [B, T_a, 1]
  init.video_clean_latent        bf16  [B, T_v, 128]
  init.audio_clean_latent        bf16  [B, T_a, 128]
  init.video_positions           bf16  [B, 3, T_v, 2]
  init.audio_positions           f32   [B, 1, T_a, 2]
  init.v_context                 bf16  [B, T_text, 4096]
  init.a_context                 bf16  [B, T_text, 2048]

  # Per-step intermediates (after each Euler step)
  step_N.sigma                   f32   []
  step_N.video_velocity          bf16  [B, T_v, 128]
  step_N.audio_velocity          bf16  [B, T_a, 128]
  step_N.video_denoised          bf16  [B, T_v, 128]  — after to_denoised
  step_N.audio_denoised          bf16  [B, T_a, 128]
  step_N.video_blended           bf16  [B, T_v, 128]  — after post_process_latent
  step_N.audio_blended           bf16  [B, T_a, 128]
  step_N.video_next_latent       bf16  [B, T_v, 128]  — Euler step output
  step_N.audio_next_latent       bf16  [B, T_a, 128]

  # Final denoised output (after all 3 steps)
  final.video_latent             bf16  [B, T_v, 128]
  final.audio_latent             bf16  [B, T_a, 128]

Usage:
  cd /root/repos/LTX-2
  uv run scripts/export_step3_fixture.py --token-limit 512
"""

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import save_file

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.types import LatentState
from ltx_core.utils import to_denoised
from ltx_pipelines.utils.helpers import modality_from_latent_state, post_process_latent
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline


TRACE_DIR = Path("trace_run")
SIGMAS = STAGE_2_DISTILLED_SIGMA_VALUES  # [0.909375, 0.725, 0.421875, 0.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Step 3 (denoising loop) fixture")
    parser.add_argument("--token-limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _slice_token_prefix(x, token_limit: int):
    if isinstance(x, torch.Tensor) and x.ndim >= 2:
        return x[:, :token_limit, ...].contiguous()
    return x


def _slice_positions_token_prefix(x, token_limit: int):
    if not isinstance(x, torch.Tensor):
        return x
    if x.ndim < 2:
        return x
    if x.ndim >= 3 and x.shape[1] <= 8 and x.shape[2] > x.shape[1]:
        return x[:, :, :token_limit, ...].contiguous()
    return x[:, :token_limit, ...].contiguous()


def main() -> None:
    args = parse_args()

    checkpoint_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser())
    distilled_lora_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled-lora-384.safetensors").expanduser())
    spatial_upsampler_path = str(
        Path("~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors").expanduser()
    )
    gemma_root = str(Path("~/models/gemma-3-12b-it").expanduser())

    print("Loading pipeline...")
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=[],
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=[],
        quantization=None,
    )

    device = pipeline.device
    dtype = torch.bfloat16

    # Load the saved replay data (step 0 = initial noisy state for the loop)
    print("Loading saved replay tensors...")
    contexts = torch.load(TRACE_DIR / "01_text_contexts.pt", map_location="cpu", weights_only=False)
    stage2_steps = torch.load(TRACE_DIR / "11_stage2_steps.pt", map_location="cpu", weights_only=False)

    # Step 0 is the initial state at sigma[0]
    step = stage2_steps[0]

    v_context_p = contexts["v_context_p"].to(device=device, dtype=dtype)
    a_context_p = contexts["a_context_p"].to(device=device, dtype=dtype)

    video_state = LatentState(
        latent=step["video_latent"].to(device=device, dtype=dtype),
        denoise_mask=step["video_denoise_mask"].to(device=device),
        positions=step["video_positions"].to(device=device),
        clean_latent=step["video_clean_latent"].to(device=device, dtype=dtype),
    )
    audio_state = LatentState(
        latent=step["audio_latent"].to(device=device, dtype=dtype),
        denoise_mask=step["audio_denoise_mask"].to(device=device),
        positions=step["audio_positions"].to(device=device),
        clean_latent=step["audio_clean_latent"].to(device=device, dtype=dtype),
    )

    if args.token_limit is not None:
        tl = args.token_limit
        v_context_p = _slice_token_prefix(v_context_p, tl)
        a_context_p = _slice_token_prefix(a_context_p, tl)
        video_state = LatentState(
            latent=_slice_token_prefix(video_state.latent, tl),
            denoise_mask=_slice_token_prefix(video_state.denoise_mask, tl),
            positions=_slice_positions_token_prefix(video_state.positions, tl),
            clean_latent=_slice_token_prefix(video_state.clean_latent, tl),
        )
        audio_state = LatentState(
            latent=_slice_token_prefix(audio_state.latent, tl),
            denoise_mask=_slice_token_prefix(audio_state.denoise_mask, tl),
            positions=_slice_positions_token_prefix(audio_state.positions, tl),
            clean_latent=_slice_token_prefix(audio_state.clean_latent, tl),
        )
        print(f"  token-limited: token_limit={tl}")

    # Get the transformer (X0Model wrapping velocity_model)
    transformer = pipeline.stage_2_model_ledger.transformer()
    stepper = EulerDiffusionStep()
    sigmas = torch.tensor(SIGMAS, device=device, dtype=torch.float32)

    print(f"Sigma schedule: {SIGMAS}")
    print(f"  video_latent: {list(video_state.latent.shape)}  dtype={video_state.latent.dtype}")
    print(f"  audio_latent: {list(audio_state.latent.shape)}  dtype={audio_state.latent.dtype}")
    print(f"  video_denoise_mask: {list(video_state.denoise_mask.shape)}  dtype={video_state.denoise_mask.dtype}")
    print(f"  audio_denoise_mask: {list(audio_state.denoise_mask.shape)}  dtype={audio_state.denoise_mask.dtype}")

    # ---- Capture initial state ----
    captured: dict[str, torch.Tensor] = {}
    captured["init.video_latent"] = video_state.latent.detach().cpu().contiguous()
    captured["init.audio_latent"] = audio_state.latent.detach().cpu().contiguous()
    captured["init.video_denoise_mask"] = video_state.denoise_mask.detach().cpu().contiguous()
    captured["init.audio_denoise_mask"] = audio_state.denoise_mask.detach().cpu().contiguous()
    captured["init.video_clean_latent"] = video_state.clean_latent.detach().cpu().contiguous()
    captured["init.audio_clean_latent"] = audio_state.clean_latent.detach().cpu().contiguous()
    captured["init.video_positions"] = video_state.positions.detach().cpu().contiguous()
    captured["init.audio_positions"] = audio_state.positions.detach().cpu().contiguous()
    captured["init.v_context"] = v_context_p.detach().cpu().contiguous()
    captured["init.a_context"] = a_context_p.detach().cpu().contiguous()

    # ---- Run denoising loop ----
    print("\nRunning 3-step denoising loop...")
    with torch.inference_mode():
        for step_idx in range(len(sigmas) - 1):  # 3 steps
            sigma = sigmas[step_idx]
            sigma_next = sigmas[step_idx + 1]
            prefix = f"step_{step_idx}"

            print(f"\n  Step {step_idx}: sigma={sigma.item():.6f} → {sigma_next.item():.6f}")

            captured[f"{prefix}.sigma"] = sigma.detach().cpu().contiguous()

            # 1. Create Modality objects
            pos_video = modality_from_latent_state(video_state, v_context_p, sigma)
            pos_audio = modality_from_latent_state(audio_state, a_context_p, sigma)

            # 2. Run velocity model (LTXModel.forward, NOT X0Model)
            vm = transformer.velocity_model
            vx, ax = vm(video=pos_video, audio=pos_audio, perturbations=None)

            captured[f"{prefix}.video_velocity"] = vx.detach().cpu().contiguous()
            captured[f"{prefix}.audio_velocity"] = ax.detach().cpu().contiguous()
            print(f"    velocity: video={list(vx.shape)} audio={list(ax.shape)}")

            # 3. to_denoised: denoised = sample - velocity * timesteps
            #    where timesteps = denoise_mask * sigma (per-token)
            denoised_video = to_denoised(video_state.latent, vx, pos_video.timesteps)
            denoised_audio = to_denoised(audio_state.latent, ax, pos_audio.timesteps)

            captured[f"{prefix}.video_denoised"] = denoised_video.detach().cpu().contiguous()
            captured[f"{prefix}.audio_denoised"] = denoised_audio.detach().cpu().contiguous()

            # 4. post_process_latent: blend with clean_latent via denoise_mask
            blended_video = post_process_latent(denoised_video, video_state.denoise_mask, video_state.clean_latent)
            blended_audio = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)

            captured[f"{prefix}.video_blended"] = blended_video.detach().cpu().contiguous()
            captured[f"{prefix}.audio_blended"] = blended_audio.detach().cpu().contiguous()

            # 5. Euler step: advance noisy latent
            next_video_latent = stepper.step(video_state.latent, blended_video, sigmas, step_idx)
            next_audio_latent = stepper.step(audio_state.latent, blended_audio, sigmas, step_idx)

            captured[f"{prefix}.video_next_latent"] = next_video_latent.detach().cpu().contiguous()
            captured[f"{prefix}.audio_next_latent"] = next_audio_latent.detach().cpu().contiguous()
            print(f"    next_latent: video={list(next_video_latent.shape)} audio={list(next_audio_latent.shape)}")

            # Update states for next iteration
            video_state = replace(video_state, latent=next_video_latent)
            audio_state = replace(audio_state, latent=next_audio_latent)

    # ---- Capture final output ----
    captured["final.video_latent"] = video_state.latent.detach().cpu().contiguous()
    captured["final.audio_latent"] = audio_state.latent.detach().cpu().contiguous()

    # ---- Summary ----
    print(f"\nTotal: {len(captured)} tensors")
    for key in sorted(captured.keys()):
        t = captured[key]
        print(f"  {key:50s}  shape={list(t.shape)}  dtype={t.dtype}")

    # ---- Save ----
    token_suffix = f"_t{args.token_limit}" if args.token_limit is not None else ""
    default_out = TRACE_DIR / f"step3_fixture{token_suffix}.safetensors"
    out_path = args.output if args.output is not None else default_out

    TRACE_DIR.mkdir(exist_ok=True)
    save_file(captured, str(out_path))
    print(f"\nSaved: {out_path}  ({len(captured)} tensors)")


if __name__ == "__main__":
    main()
