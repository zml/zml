"""Export Step 2 parity fixture for velocity_model.forward with raw inputs.

Captures the full boundary of velocity_model.forward starting from raw
LatentState + sigma + context (Step 2 boundary), plus intermediate outputs
for isolated component validation (RoPE generation, patchify, adaln).

This extends the Step 1 fixture (which starts from pre-computed SharedInputs)
by adding the raw pre-preprocessing inputs.

Saved keys — raw inputs (Step 2 boundary):
  raw.sigma                         f32   []           — noise level
  raw.video_latent                  bf16  [B, T_v, 128] — noisy video latent
  raw.video_denoise_mask            f32   [B, T_v, 1]   — video denoising mask
  raw.video_positions               bf16  [B, 3, T_v, 2] — video position grid (t,h,w)
  raw.video_clean_latent            bf16  [B, T_v, 128] — video clean reference
  raw.audio_latent                  bf16  [B, T_a, 128] — noisy audio latent
  raw.audio_denoise_mask            f32   [B, T_a, 1]   — audio denoising mask
  raw.audio_positions               f32   [B, 1, T_a, 2] — audio position grid (t)
  raw.audio_clean_latent            bf16  [B, T_a, 128] — audio clean reference
  raw.v_context                     bf16  [B, T_text, 4096] — video text context
  raw.a_context                     bf16  [B, T_text, 2048] — audio text context

Saved keys — intermediate outputs (for isolated validation):
  intermediate.v_pe_cos             bf16  [B, H, T_v, HD/2] — video self-attn RoPE cos
  intermediate.v_pe_sin             bf16  [B, H, T_v, HD/2] — video self-attn RoPE sin
  intermediate.a_pe_cos             bf16  [B, H, T_a, HD/2] — audio self-attn RoPE cos
  intermediate.a_pe_sin             bf16  [B, H, T_a, HD/2] — audio self-attn RoPE sin
  intermediate.a2v_pe_cos           bf16  — video cross-attn RoPE cos
  intermediate.a2v_pe_sin           bf16  — video cross-attn RoPE sin
  intermediate.v2a_pe_cos           bf16  — audio cross-attn RoPE cos
  intermediate.v2a_pe_sin           bf16  — audio cross-attn RoPE sin
  intermediate.vx_patchified        bf16  [B, T_v, 4096] — patchified video (pre-adaln)
  intermediate.ax_patchified        bf16  [B, T_a, 2048] — patchified audio (pre-adaln)

Saved keys — velocity model outputs (reference):
  output.video_velocity             bf16  [B, T_v, 128] — raw velocity output
  output.audio_velocity             bf16  [B, T_a, 128] — raw velocity output

Usage:
  cd /root/repos/LTX-2
  uv run scripts/export_step2_fixture.py --step-idx 0 --token-limit 512
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.types import LatentState
from ltx_pipelines.utils.helpers import modality_from_latent_state
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline


TRACE_DIR = Path("trace_run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Step 2 parity fixture")
    parser.add_argument("--step-idx", type=int, default=0)
    parser.add_argument("--token-limit", type=int, default=None)
    parser.add_argument("--distilled-lora-strength", type=float, default=0.0)
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

    lora_cfg = []
    if args.distilled_lora_strength != 0.0:
        lora_cfg = [
            LoraPathStrengthAndSDOps(
                path=distilled_lora_path,
                strength=args.distilled_lora_strength,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ]

    print(f"Loading pipeline (lora_strength={args.distilled_lora_strength})...")
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=lora_cfg,
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=[],
        quantization=None,
    )

    device = pipeline.device
    dtype = torch.bfloat16

    print("Loading saved replay tensors...")
    contexts = torch.load(TRACE_DIR / "01_text_contexts.pt", map_location="cpu", weights_only=False)
    stage2_steps = torch.load(TRACE_DIR / "11_stage2_steps.pt", map_location="cpu", weights_only=False)
    step = stage2_steps[args.step_idx]

    sigma = step["sigma"].to(device=device, dtype=torch.float32)
    print(f"  step_idx={args.step_idx}  sigma={sigma.item():.6f}")

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

    # ---- Capture raw inputs ----
    captured: dict[str, torch.Tensor] = {}

    captured["raw.sigma"] = sigma.detach().cpu().contiguous()
    captured["raw.video_latent"] = video_state.latent.detach().cpu().contiguous()
    captured["raw.video_denoise_mask"] = video_state.denoise_mask.detach().cpu().contiguous()
    captured["raw.video_positions"] = video_state.positions.detach().cpu().contiguous()
    captured["raw.video_clean_latent"] = video_state.clean_latent.detach().cpu().contiguous()
    captured["raw.audio_latent"] = audio_state.latent.detach().cpu().contiguous()
    captured["raw.audio_denoise_mask"] = audio_state.denoise_mask.detach().cpu().contiguous()
    captured["raw.audio_positions"] = audio_state.positions.detach().cpu().contiguous()
    captured["raw.audio_clean_latent"] = audio_state.clean_latent.detach().cpu().contiguous()
    captured["raw.v_context"] = v_context_p.detach().cpu().contiguous()
    captured["raw.a_context"] = a_context_p.detach().cpu().contiguous()

    print(f"  Raw inputs captured ({len(captured)} tensors)")
    for k, v in sorted(captured.items()):
        print(f"    {k:40s}  shape={list(v.shape)}  dtype={v.dtype}")

    # ---- Create Modality objects (as the pipeline does) ----
    pos_video = modality_from_latent_state(video_state, v_context_p, sigma)
    pos_audio = modality_from_latent_state(audio_state, a_context_p, sigma)

    # ---- Get the velocity model and its preprocessors ----
    transformer = pipeline.stage_2_model_ledger.transformer()
    vm = transformer.velocity_model

    # ---- Capture intermediate: RoPE cos/sin from preprocessors ----
    # Call the preprocessors to get TransformerArgs and capture PE
    print("\nRunning args preprocessors...")
    with torch.inference_mode():
        video_args = vm.video_args_preprocessor.prepare(pos_video, pos_audio)
        audio_args = vm.audio_args_preprocessor.prepare(pos_audio, pos_video)

    # Self-attention PE
    if isinstance(video_args.positional_embeddings, (tuple, list)):
        v_pe_cos, v_pe_sin = video_args.positional_embeddings
        captured["intermediate.v_pe_cos"] = v_pe_cos.detach().cpu().contiguous()
        captured["intermediate.v_pe_sin"] = v_pe_sin.detach().cpu().contiguous()
        print(f"  v_pe: cos={list(v_pe_cos.shape)} sin={list(v_pe_sin.shape)}  dtype={v_pe_cos.dtype}")
    if isinstance(audio_args.positional_embeddings, (tuple, list)):
        a_pe_cos, a_pe_sin = audio_args.positional_embeddings
        captured["intermediate.a_pe_cos"] = a_pe_cos.detach().cpu().contiguous()
        captured["intermediate.a_pe_sin"] = a_pe_sin.detach().cpu().contiguous()
        print(f"  a_pe: cos={list(a_pe_cos.shape)} sin={list(a_pe_sin.shape)}  dtype={a_pe_cos.dtype}")

    # Cross-attention PE
    if video_args.cross_positional_embeddings is not None:
        if isinstance(video_args.cross_positional_embeddings, (tuple, list)):
            c, s = video_args.cross_positional_embeddings
            captured["intermediate.a2v_pe_cos"] = c.detach().cpu().contiguous()
            captured["intermediate.a2v_pe_sin"] = s.detach().cpu().contiguous()
            print(f"  a2v_pe: cos={list(c.shape)} sin={list(s.shape)}  dtype={c.dtype}")
    if audio_args.cross_positional_embeddings is not None:
        if isinstance(audio_args.cross_positional_embeddings, (tuple, list)):
            c, s = audio_args.cross_positional_embeddings
            captured["intermediate.v2a_pe_cos"] = c.detach().cpu().contiguous()
            captured["intermediate.v2a_pe_sin"] = s.detach().cpu().contiguous()
            print(f"  v2a_pe: cos={list(c.shape)} sin={list(s.shape)}  dtype={c.dtype}")

    # Patchified outputs
    captured["intermediate.vx_patchified"] = video_args.x.detach().cpu().contiguous()
    captured["intermediate.ax_patchified"] = audio_args.x.detach().cpu().contiguous()
    print(f"  vx_patchified: {list(video_args.x.shape)}  dtype={video_args.x.dtype}")
    print(f"  ax_patchified: {list(audio_args.x.shape)}  dtype={audio_args.x.dtype}")

    # Timestep embeddings (for cross-check)
    captured["intermediate.video_timesteps"] = video_args.timesteps.detach().cpu().contiguous()
    captured["intermediate.audio_timesteps"] = audio_args.timesteps.detach().cpu().contiguous()
    captured["intermediate.v_embedded_ts"] = video_args.embedded_timestep.detach().cpu().contiguous()
    captured["intermediate.a_embedded_ts"] = audio_args.embedded_timestep.detach().cpu().contiguous()
    if video_args.prompt_timestep is not None:
        captured["intermediate.v_prompt_timestep"] = video_args.prompt_timestep.detach().cpu().contiguous()
    if audio_args.prompt_timestep is not None:
        captured["intermediate.a_prompt_timestep"] = audio_args.prompt_timestep.detach().cpu().contiguous()

    # Cross-attention timestep embeddings
    if video_args.cross_scale_shift_timestep is not None:
        captured["intermediate.v_cross_ss_ts"] = video_args.cross_scale_shift_timestep.detach().cpu().contiguous()
    if video_args.cross_gate_timestep is not None:
        captured["intermediate.v_cross_gate_ts"] = video_args.cross_gate_timestep.detach().cpu().contiguous()
    if audio_args.cross_scale_shift_timestep is not None:
        captured["intermediate.a_cross_ss_ts"] = audio_args.cross_scale_shift_timestep.detach().cpu().contiguous()
    if audio_args.cross_gate_timestep is not None:
        captured["intermediate.a_cross_gate_ts"] = audio_args.cross_gate_timestep.detach().cpu().contiguous()

    # Context (post-prepare, should match raw since no caption_projection)
    captured["intermediate.v_context"] = video_args.context.detach().cpu().contiguous()
    captured["intermediate.a_context"] = audio_args.context.detach().cpu().contiguous()

    print(f"\n  Intermediate tensors captured ({sum(1 for k in captured if k.startswith('intermediate.'))} tensors)")

    # ---- Run the velocity model forward pass ----
    print("\nRunning velocity_model forward...")
    with torch.inference_mode():
        # Use LTXModel.forward directly to get raw velocity (not X0Model which applies to_denoised)
        vx, ax = vm(video=pos_video, audio=pos_audio, perturbations=None)

    captured["output.video_velocity"] = vx.detach().cpu().contiguous()
    captured["output.audio_velocity"] = ax.detach().cpu().contiguous()
    print(f"  video_velocity: {list(vx.shape)}  dtype={vx.dtype}")
    print(f"  audio_velocity: {list(ax.shape)}  dtype={ax.dtype}")

    # ---- Summary ----
    print(f"\nTotal: {len(captured)} tensors")
    for key in sorted(captured.keys()):
        t = captured[key]
        print(f"  {key:50s}  shape={list(t.shape)}  dtype={t.dtype}")

    # ---- Save ----
    token_suffix = f"_t{args.token_limit}" if args.token_limit is not None else ""
    lora_suffix = f"_lora{args.distilled_lora_strength:.1f}" if args.distilled_lora_strength != 0.0 else ""
    default_out = TRACE_DIR / f"step2_fixture_step_{args.step_idx:03d}{token_suffix}{lora_suffix}.safetensors"
    out_path = args.output if args.output is not None else default_out

    TRACE_DIR.mkdir(exist_ok=True)
    save_file(captured, str(out_path))
    print(f"\nSaved: {out_path}  ({len(captured)} tensors)")


if __name__ == "__main__":
    main()
