"""Export a live-capture parity fixture for full velocity_model.forward validation.

Replays one stage-2 transformer step (from trace_run/11_stage2_steps.pt) and
captures *actual pipeline activations* flowing through velocity_model.forward:

  - All 8 AdaLayerNormSingle modules (sigma_scaled, modulation, embedded_timestep)
  - OutputProjection for both video and audio (x_in, embedded_timestep, output)
  - Block-level SharedInputs (post-patchify vx/ax, PE cos/sin, text contexts, masks)
  - Final denoised outputs (video_out, audio_out)

This fixture provides everything needed for:
  1. AdaLayerNormSingle parity checks (live_capture_check.zig)
  2. OutputProjection parity checks (live_capture_check.zig)
  3. Full velocity_model.forward parity check (full_step_check.zig)

Saved keys — AdaLayerNormSingle (for each module P in ADALN_ATTR_NAMES):
  P.sigma_scaled           f32  [B]        — timestep input (sigma × 1000)
  P.modulation             bf16 [B, N*D]   — shift/scale/gate coefficients
  P.embedded_timestep      bf16 [B, D]     — intermediate (used by OutputProjection)

Saved keys — OutputProjection:
  output_projection.video.x_in              bf16 [B, T_v, 4096]
  output_projection.video.embedded_timestep bf16 [B, T_v, 4096]  (expanded in live pipeline)
  output_projection.video.output            bf16 [B, T_v, 128]
  output_projection.audio.x_in             bf16 [B, T_a, 2048]
  output_projection.audio.embedded_timestep bf16 [B, T_a, 2048]
  output_projection.audio.output            bf16 [B, T_a, 128]

Saved keys — full-step SharedInputs (from block 0 pre-hook):
  full_step.vx_in                   bf16 [B, T_v, 4096]
  full_step.ax_in                   bf16 [B, T_a, 2048]
  full_step.video_timesteps         bf16 [B, 1, 9*4096]
  full_step.audio_timesteps         bf16 [B, 1, 9*2048]
  full_step.v_prompt_timestep       bf16 [B, 1, 2*4096]
  full_step.a_prompt_timestep       bf16 [B, 1, 2*2048]
  full_step.v_pe_cos/sin            bf16 [B, H, T_v, HD]
  full_step.a_pe_cos/sin            bf16 [B, H, T_a, HD]
  full_step.v_text_ctx              bf16 [B, T_text, D]
  full_step.a_text_ctx              bf16 [B, T_text, D]
  full_step.v_cross_ss_ts           bf16 [B, 1, 4*4096]
  full_step.v_cross_gate_ts         bf16 [B, 1, 4096]
  full_step.a_cross_ss_ts           bf16 [B, 1, 4*2048]
  full_step.a_cross_gate_ts         bf16 [B, 1, 2048]
  full_step.a2v_pe_cos/sin          (cross-attn PE pairs)
  full_step.a2v_k_pe_cos/sin
  full_step.v2a_pe_cos/sin
  full_step.v2a_k_pe_cos/sin
  full_step.video_out               bf16 [B, T_v, 128]
  full_step.audio_out               bf16 [B, T_a, 128]

Usage:
  cd /root/repos/LTX-2
  uv run ./scripts/export_live_capture_fixture.py --step-idx 0 --token-limit 512
  uv run ./scripts/export_live_capture_fixture.py --step-idx 0
"""

import argparse
import inspect
from pathlib import Path

import torch
from safetensors.torch import save_file

from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.types import LatentState
from ltx_pipelines.utils.helpers import modality_from_latent_state
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline


TRACE_DIR = Path("trace_run")

# Attribute names of the 8 AdaLayerNormSingle modules on velocity_model.
ADALN_ATTR_NAMES = [
    "adaln_single",
    "audio_adaln_single",
    "prompt_adaln_single",
    "audio_prompt_adaln_single",
    "av_ca_video_scale_shift_adaln_single",
    "av_ca_audio_scale_shift_adaln_single",
    "av_ca_a2v_gate_adaln_single",
    "av_ca_v2a_gate_adaln_single",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export live-capture parity fixture")
    parser.add_argument(
        "--step-idx",
        type=int,
        default=0,
        help="Index in 11_stage2_steps.pt (0 = first denoising step, sigma ≈ 1.0)",
    )
    parser.add_argument(
        "--distilled-lora-strength",
        type=float,
        default=0.0,
        help=(
            "LoRA strength to apply. Default 0.0 (base checkpoint only). "
            "adaln_single and proj_out weights are not LoRA-adapted in LTX-22b, "
            "so this has no effect on the captured adaln/output-projection tensors, "
            "but setting it matches the training setup for a more realistic forward pass."
        ),
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=None,
        help=(
            "Truncate video/audio token sequences to this length before the forward pass. "
            "Useful to reduce memory usage. E.g. --token-limit 512."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output safetensors path. Default: trace_run/live_capture_fixture_step_NNN.safetensors",
    )
    return parser.parse_args()


def load_pt(name: str):
    """Load a tensor/object from TRACE_DIR."""
    return torch.load(TRACE_DIR / name, map_location="cpu", weights_only=False)


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
    contexts = load_pt("01_text_contexts.pt")
    stage2_steps = load_pt("11_stage2_steps.pt")
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
        print(f"  token-limited replay: token_limit={tl}")

    pos_video = modality_from_latent_state(video_state, v_context_p, sigma)
    pos_audio = modality_from_latent_state(audio_state, a_context_p, sigma)

    transformer = pipeline.stage_2_model_ledger.transformer()
    vm = transformer.velocity_model

    # Print _process_output signature for diagnostics
    try:
        sig = inspect.signature(type(vm)._process_output)
        print(f"  _process_output signature: {sig}")
    except Exception as exc:
        print(f"  WARNING: could not inspect _process_output signature: {exc}")

    # ---- Capture machinery ----
    captured: dict[str, torch.Tensor] = {}
    handles: list = []

    # Forward hooks on each AdaLayerNormSingle module.
    # PyTorch forward hooks receive (module, inputs_tuple, outputs).
    # AdaLayerNormSingle.forward(timestep, ...) → (modulation, embedded_timestep).
    def make_adaln_hook(prefix: str):
        def hook(module, inputs, outputs):
            # inputs[0] = timestep (sigma × 1000), shape [B] or [B, 1]
            if inputs and isinstance(inputs[0], torch.Tensor):
                sigma_in = inputs[0].detach().float().cpu().squeeze()  # → 1-D [B]
                if sigma_in.ndim == 0:
                    sigma_in = sigma_in.unsqueeze(0)
                captured[f"{prefix}.sigma_scaled"] = sigma_in.contiguous()
            # outputs may be a (modulation, embedded_timestep) tuple
            if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
                mod, emb = outputs[0], outputs[1]
                if isinstance(mod, torch.Tensor):
                    captured[f"{prefix}.modulation"] = mod.detach().cpu().contiguous()
                if isinstance(emb, torch.Tensor):
                    captured[f"{prefix}.embedded_timestep"] = emb.detach().cpu().contiguous()
            elif isinstance(outputs, torch.Tensor):
                # Some adaln variants may only return modulation
                captured[f"{prefix}.modulation"] = outputs.detach().cpu().contiguous()
                print(f"  WARNING: {prefix} returned a single Tensor (no embedded_timestep)")
        return hook

    for attr in ADALN_ATTR_NAMES:
        mod = getattr(vm, attr, None)
        if mod is None:
            print(f"  WARNING: velocity_model.{attr} not found — skipping")
            continue
        h = mod.register_forward_hook(make_adaln_hook(attr))
        handles.append(h)
        print(f"  adaln hook registered: {attr}")

    # ---- Block-level hooks: capture SharedInputs from block 0 ----
    # The pre-hook (with_kwargs=True) fires with (module, args_tuple, kwargs_dict).
    # Block.forward(video=..., audio=..., perturbations=...).
    # video/audio are TransformerArgs objects whose fields become SharedInputs.

    def _detach(v):
        """Detach tensor to CPU if it's a Tensor, else return None."""
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().contiguous()
        return None

    def _extract_pe_pair(value):
        """Extract (cos, sin) from a tuple or structured PE object."""
        if isinstance(value, (tuple, list)) and len(value) == 2:
            c, s = value
            if isinstance(c, torch.Tensor) and isinstance(s, torch.Tensor):
                return _detach(c), _detach(s)
        if hasattr(value, "cos") and hasattr(value, "sin"):
            return _detach(value.cos), _detach(value.sin)
        return None, None

    block0 = vm.transformer_blocks[0]

    def block0_pre_hook(module, args_tuple, kwargs_dict):
        """Capture SharedInputs from the first transformer block's input."""
        try:
            video = kwargs_dict.get("video") if kwargs_dict else None
            if video is None and len(args_tuple) > 0:
                video = args_tuple[0]
            audio = kwargs_dict.get("audio") if kwargs_dict else None
            if audio is None and len(args_tuple) > 1:
                audio = args_tuple[1]

            if video is None or audio is None:
                print("  WARNING: block0 pre-hook: could not extract video/audio args")
                return None

            # Stream inputs (post-patchify)
            if hasattr(video, "x") and isinstance(video.x, torch.Tensor):
                captured["full_step.vx_in"] = _detach(video.x)
            if hasattr(audio, "x") and isinstance(audio.x, torch.Tensor):
                captured["full_step.ax_in"] = _detach(audio.x)

            # Timestep modulations (identical across all blocks)
            if hasattr(video, "timesteps"):
                captured["full_step.video_timesteps"] = _detach(video.timesteps)
            if hasattr(audio, "timesteps"):
                captured["full_step.audio_timesteps"] = _detach(audio.timesteps)
            if hasattr(video, "prompt_timestep"):
                captured["full_step.v_prompt_timestep"] = _detach(video.prompt_timestep)
            if hasattr(audio, "prompt_timestep"):
                captured["full_step.a_prompt_timestep"] = _detach(audio.prompt_timestep)

            # AV cross-attn timestep embeddings
            if hasattr(video, "cross_scale_shift_timestep"):
                captured["full_step.v_cross_ss_ts"] = _detach(video.cross_scale_shift_timestep)
            if hasattr(video, "cross_gate_timestep"):
                captured["full_step.v_cross_gate_ts"] = _detach(video.cross_gate_timestep)
            if hasattr(audio, "cross_scale_shift_timestep"):
                captured["full_step.a_cross_ss_ts"] = _detach(audio.cross_scale_shift_timestep)
            if hasattr(audio, "cross_gate_timestep"):
                captured["full_step.a_cross_gate_ts"] = _detach(audio.cross_gate_timestep)

            # Self-attention PE cos/sin
            if hasattr(video, "positional_embeddings"):
                c, s = _extract_pe_pair(video.positional_embeddings)
                if c is not None:
                    captured["full_step.v_pe_cos"] = c
                if s is not None:
                    captured["full_step.v_pe_sin"] = s
            if hasattr(audio, "positional_embeddings"):
                c, s = _extract_pe_pair(audio.positional_embeddings)
                if c is not None:
                    captured["full_step.a_pe_cos"] = c
                if s is not None:
                    captured["full_step.a_pe_sin"] = s

            # Text contexts
            if hasattr(video, "context") and isinstance(video.context, torch.Tensor):
                captured["full_step.v_text_ctx"] = _detach(video.context)
            if hasattr(audio, "context") and isinstance(audio.context, torch.Tensor):
                captured["full_step.a_text_ctx"] = _detach(audio.context)
            if hasattr(video, "context_mask") and isinstance(video.context_mask, torch.Tensor):
                captured["full_step.v_text_ctx_mask"] = _detach(video.context_mask)
            if hasattr(audio, "context_mask") and isinstance(audio.context_mask, torch.Tensor):
                captured["full_step.a_text_ctx_mask"] = _detach(audio.context_mask)

            # AV cross-attn PE
            if hasattr(video, "cross_positional_embeddings"):
                c, s = _extract_pe_pair(video.cross_positional_embeddings)
                if c is not None:
                    captured["full_step.a2v_pe_cos"] = c
                if s is not None:
                    captured["full_step.a2v_pe_sin"] = s
            if hasattr(video, "cross_k_positional_embeddings"):
                c, s = _extract_pe_pair(video.cross_k_positional_embeddings)
                if c is not None:
                    captured["full_step.a2v_k_pe_cos"] = c
                if s is not None:
                    captured["full_step.a2v_k_pe_sin"] = s
            if hasattr(audio, "cross_positional_embeddings"):
                c, s = _extract_pe_pair(audio.cross_positional_embeddings)
                if c is not None:
                    captured["full_step.v2a_pe_cos"] = c
                if s is not None:
                    captured["full_step.v2a_pe_sin"] = s
            if hasattr(audio, "cross_k_positional_embeddings"):
                c, s = _extract_pe_pair(audio.cross_k_positional_embeddings)
                if c is not None:
                    captured["full_step.v2a_k_pe_cos"] = c
                if s is not None:
                    captured["full_step.v2a_k_pe_sin"] = s

            fs_keys = [k for k in captured if k.startswith("full_step.")]
            print(f"  block0 pre-hook: captured {len(fs_keys)} full_step.* tensors")
        except Exception as exc:
            print(f"  WARNING: block0 pre-hook failed: {exc}")
        return None

    h = block0.register_forward_pre_hook(block0_pre_hook, with_kwargs=True)
    handles.append(h)
    print("  block0 pre-hook registered")

    # ---- Cross-attn submodule hooks: capture k_pe kwargs ----
    # k_pe is passed as a kwarg to audio_to_video_attn / video_to_audio_attn
    # inside the block forward, not on TransformerArgs. Hook those submodules.
    def make_cross_attn_hook(prefix):
        """Create a pre-hook that captures k_pe from cross-attn module kwargs."""
        def hook(module, args_tuple, kwargs_dict):
            try:
                k_pe = kwargs_dict.get("k_pe") if kwargs_dict else None
                if k_pe is not None:
                    c, s = _extract_pe_pair(k_pe)
                    if c is not None:
                        captured[f"{prefix}_k_pe_cos"] = c
                    if s is not None:
                        captured[f"{prefix}_k_pe_sin"] = s
                    print(f"  {prefix} k_pe hook: captured cos={c.shape if c is not None else None} sin={s.shape if s is not None else None}")
            except Exception as exc:
                print(f"  WARNING: {prefix} k_pe hook failed: {exc}")
            return None
        return hook

    if hasattr(block0, "audio_to_video_attn"):
        h = block0.audio_to_video_attn.register_forward_pre_hook(
            make_cross_attn_hook("full_step.a2v"), with_kwargs=True)
        handles.append(h)
        print("  a2v cross-attn k_pe hook registered")
    if hasattr(block0, "video_to_audio_attn"):
        h = block0.video_to_audio_attn.register_forward_pre_hook(
            make_cross_attn_hook("full_step.v2a"), with_kwargs=True)
        handles.append(h)
        print("  v2a cross-attn k_pe hook registered")

    # Monkeypatch velocity_model._process_output to capture OutputProjection I/O.
    # Signature (from source inspection): _process_output(self, scale_shift_table,
    #   norm_out, proj_out, x, embedded_timestep) → Tensor
    vm_cls = type(vm)

    try:
        original_process_output = vm_cls._process_output

        def patched_process_output(self, *args, **kwargs):
            out = original_process_output(self, *args, **kwargs)
            try:
                # Positional args: (scale_shift_table, norm_out, proj_out, x, embedded_timestep)
                if len(args) >= 5:
                    sst_arg = args[0]
                    x_arg = args[3]
                    emb_arg = args[4]
                elif len(args) >= 2:
                    # Fallback: try kwargs
                    sst_arg = kwargs.get("scale_shift_table", args[0] if args else None)
                    x_arg = kwargs.get("x", None)
                    emb_arg = kwargs.get("embedded_timestep", None)
                else:
                    print("  WARNING: unexpected _process_output args, skipping capture")
                    return out

                # Identify video vs audio by which scale_shift_table was passed
                if hasattr(self, "scale_shift_table") and sst_arg is self.scale_shift_table:
                    prefix = "output_projection.video"
                elif hasattr(self, "audio_scale_shift_table") and sst_arg is self.audio_scale_shift_table:
                    prefix = "output_projection.audio"
                else:
                    print("  WARNING: _process_output called with unknown scale_shift_table — skipping")
                    return out

                if isinstance(x_arg, torch.Tensor):
                    captured[f"{prefix}.x_in"] = x_arg.detach().cpu().contiguous()
                if isinstance(emb_arg, torch.Tensor):
                    captured[f"{prefix}.embedded_timestep"] = emb_arg.detach().cpu().contiguous()
                if isinstance(out, torch.Tensor):
                    captured[f"{prefix}.output"] = out.detach().cpu().contiguous()

                shapes = {
                    "x": tuple(x_arg.shape) if isinstance(x_arg, torch.Tensor) else "?",
                    "emb": tuple(emb_arg.shape) if isinstance(emb_arg, torch.Tensor) else "?",
                    "out": tuple(out.shape) if isinstance(out, torch.Tensor) else "?",
                }
                print(f"  {prefix}: x={shapes['x']} emb={shapes['emb']} out={shapes['out']}")
            except Exception as exc:
                print(f"  WARNING: capture in patched _process_output failed: {exc}")
            return out

        vm_cls._process_output = patched_process_output
        print("  _process_output monkeypatched")
        process_output_patched = True
    except Exception as exc:
        print(f"  WARNING: could not monkeypatch _process_output: {exc}")
        process_output_patched = False

    # ---- Run the forward pass ----
    print(f"\nRunning transformer forward pass (step_idx={args.step_idx})...")
    try:
        with torch.inference_mode():
            denoised_video, denoised_audio = transformer(
                video=pos_video,
                audio=pos_audio,
                perturbations=None,
            )
        print(f"  denoised_video: {tuple(denoised_video.shape)}")
        print(f"  denoised_audio: {tuple(denoised_audio.shape)}")

        # Capture final denoised outputs for full-step parity
        captured["full_step.video_out"] = denoised_video.detach().cpu().contiguous()
        captured["full_step.audio_out"] = denoised_audio.detach().cpu().contiguous()
    finally:
        # Always restore original method and remove hooks
        if process_output_patched:
            vm_cls._process_output = original_process_output
        for h in handles:
            h.remove()

    # ---- Summary ----
    print(f"\nCaptured {len(captured)} tensors:")
    for key in sorted(captured.keys()):
        t = captured[key]
        print(f"  {key:60s}  {str(tuple(t.shape)):25s}  {t.dtype}")

    # Verify all expected keys are present
    expected_adaln_keys = {
        f"{attr}.{field}"
        for attr in ADALN_ATTR_NAMES
        for field in ("sigma_scaled", "modulation", "embedded_timestep")
        if getattr(vm, attr, None) is not None
    }
    expected_op_keys = {
        f"output_projection.{stream}.{field}"
        for stream in ("video", "audio")
        for field in ("x_in", "embedded_timestep", "output")
    }
    expected_full_step_keys = {
        "full_step.vx_in", "full_step.ax_in",
        "full_step.video_timesteps", "full_step.audio_timesteps",
        "full_step.v_prompt_timestep", "full_step.a_prompt_timestep",
        "full_step.v_cross_ss_ts", "full_step.v_cross_gate_ts",
        "full_step.a_cross_ss_ts", "full_step.a_cross_gate_ts",
        "full_step.v_pe_cos", "full_step.v_pe_sin",
        "full_step.a_pe_cos", "full_step.a_pe_sin",
        "full_step.v_text_ctx", "full_step.a_text_ctx",
        "full_step.a2v_pe_cos", "full_step.a2v_pe_sin",
        "full_step.a2v_k_pe_cos", "full_step.a2v_k_pe_sin",
        "full_step.v2a_pe_cos", "full_step.v2a_pe_sin",
        "full_step.v2a_k_pe_cos", "full_step.v2a_k_pe_sin",
        "full_step.video_out", "full_step.audio_out",
    }
    missing = (expected_adaln_keys | expected_op_keys | expected_full_step_keys) - set(captured.keys())
    if missing:
        print(f"\n  WARNING: missing expected keys:")
        for k in sorted(missing):
            print(f"    {k}")
    else:
        print("\n  All expected keys captured.")

    # ---- Save ----
    token_suffix = f"_t{args.token_limit}" if args.token_limit is not None else ""
    lora_suffix = f"_lora{args.distilled_lora_strength:.1f}" if args.distilled_lora_strength != 0.0 else ""
    default_out = TRACE_DIR / f"live_capture_fixture_step_{args.step_idx:03d}{token_suffix}{lora_suffix}.safetensors"
    out_path = args.output if args.output is not None else default_out

    TRACE_DIR.mkdir(exist_ok=True)
    save_file(captured, str(out_path))
    print(f"\nSaved: {out_path}  ({len(captured)} tensors)")


if __name__ == "__main__":
    main()
