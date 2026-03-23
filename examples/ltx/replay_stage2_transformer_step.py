"""Replay one stage-2 transformer step and capture activations.

This script replays a single recorded step from `trace_run/11_stage2_steps.pt`,
rebuilds stage-2 inputs, runs the transformer forward pass, and saves activations
to a trace file.

Typical usage:

1) Full pass (outputs only):
     uv run ./scripts/replay_stage2_transformer_step.py --pass-label full

2) One block slice with inputs+outputs:
     uv run ./scripts/replay_stage2_transformer_step.py \
         --pass-label b00_07 \
         --capture-inputs \
         --include '^velocity_model\\.transformer_blocks\\.(0|1|2|3|4|5|6|7)(\\.|$)'

3) Another slice:
     uv run ./scripts/replay_stage2_transformer_step.py \
         --pass-label b08_15 \
         --capture-inputs \
         --include '^velocity_model\\.transformer_blocks\\.(8|9|10|11|12|13|14|15)(\\.|$)'

Notes:
- Use multiple `--include` flags to trace multiple regex groups in one pass.
- `--leaf-only` (default) captures only leaf modules and avoids container duplicates.
- `--max-capture-gib` limits in-memory capture and prevents OOM kills.
- Output file format:
    trace_run/acts_stage2_transformer_step_<step>_<pass-label>.pt
"""

import argparse
import types
from typing import Any
from pathlib import Path

import torch
import torch.nn.functional as F
import zml_utils

from ltx_core.guidance.perturbations import PerturbationType
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_pipelines.utils.helpers import modality_from_latent_state
from ltx_core.types import LatentState
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline


TRACE_DIR = Path("trace_run")


def parse_args() -> argparse.Namespace:
    """Parse CLI options for pass-based activation tracing."""
    parser = argparse.ArgumentParser(description="Replay one stage-2 transformer step and capture activations")
    parser.add_argument("--step-idx", type=int, default=0, help="Index in 11_stage2_steps.pt")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Regex of module names to trace. Can be repeated.",
    )
    parser.add_argument(
        "--pass-label",
        type=str,
        default="full",
        help="Label injected in output filename and metadata.",
    )
    parser.add_argument(
        "--capture-inputs",
        action="store_true",
        help="Also capture module inputs/kwargs in addition to outputs.",
    )
    parser.add_argument(
        "--all-modules",
        action="store_true",
        help="Capture all matched modules, including container modules.",
    )
    parser.add_argument(
        "--max-capture-gib",
        type=float,
        default=2.0,
        help="Capture budget in GiB before collector disables further captures.",
    )
    parser.add_argument(
        "--capture-kwargs",
        action="store_true",
        help=(
            "For modules matching --include patterns, also capture keyword arguments "
            "(e.g. pe, mask, context) via register_forward_pre_hook(with_kwargs=True). "
            "Stored in activations as '{module_name}.__kwargs__': {kwarg_name: tensor, ...}. "
            "Requires PyTorch >= 2.0."
        ),
    )
    parser.add_argument(
        "--distilled-lora-strength",
        type=float,
        default=0.8,
        help=(
            "Strength for the distilled LoRA used by TI2VidTwoStagesPipeline. "
            "Set to 0.0 to disable LoRA and capture activations that match the base checkpoint."
        ),
    )
    parser.add_argument(
        "--capture-sdpa-call-idx",
        type=int,
        default=None,
        help=(
            "If set, monkeypatch torch.nn.functional.scaled_dot_product_attention and capture "
            "q/k/v/out tensors for the specified call index (0-based) into activations."
        ),
    )
    parser.add_argument(
        "--log-sdpa-calls",
        action="store_true",
        help=(
            "If set, monkeypatch torch.nn.functional.scaled_dot_product_attention and log "
            "all observed call shapes (q/k/v/out). Useful to identify the correct call index "
            "for --capture-sdpa-call-idx."
        ),
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=None,
        help=(
            "Optional token prefix length applied at replay time to video/audio latent states "
            "and text contexts before transformer forward. "
            "Useful to generate context-consistent token-limited traces."
        ),
    )
    return parser.parse_args()


def _slice_token_prefix(x: Any, token_limit: int) -> Any:
    if isinstance(x, torch.Tensor) and x.ndim >= 2:
        return x[:, :token_limit, ...].contiguous()
    return x


def _slice_positions_token_prefix(x: Any, token_limit: int) -> Any:
    if not isinstance(x, torch.Tensor):
        return x
    if x.ndim < 2:
        return x

    if x.ndim >= 3:
        # Positions can be either [B, T, C] or [B, C, T].
        if x.shape[1] <= 8 and x.shape[2] > x.shape[1]:
            return x[:, :, :token_limit, ...].contiguous()

    return x[:, :token_limit, ...].contiguous()


def load_pt(name: str):
    """Load a trace tensor/object from TRACE_DIR."""
    return torch.load(TRACE_DIR / name, map_location="cpu", weights_only=False)


def main() -> None:
    """Build pipeline, replay one step, collect activations, and save trace."""
    args = parse_args()
    if args.token_limit is not None and args.token_limit <= 0:
        raise ValueError(f"--token-limit must be > 0, got {args.token_limit}")

    checkpoint_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser())
    distilled_lora_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled-lora-384.safetensors").expanduser())
    spatial_upsampler_path = str(Path("~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors").expanduser())
    gemma_root = str(Path("~/models/gemma-3-12b-it").expanduser())

    distilled_lora_cfg = []
    if args.distilled_lora_strength != 0.0:
        distilled_lora_cfg = [
            LoraPathStrengthAndSDOps(
                path=distilled_lora_path,
                strength=args.distilled_lora_strength,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ]

    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=distilled_lora_cfg,
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=[],
        quantization=None,
    )

    device = pipeline.device
    dtype = torch.bfloat16

    contexts = load_pt("01_text_contexts.pt")
    stage2_steps = load_pt("11_stage2_steps.pt")

    step_idx = args.step_idx
    step = stage2_steps[step_idx]

    v_context_p = contexts["v_context_p"].to(device=device, dtype=dtype)
    a_context_p = contexts["a_context_p"].to(device=device, dtype=dtype)

    sigma = step["sigma"].to(device=device, dtype=torch.float32)
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
        token_limit = args.token_limit
        v_context_p = _slice_token_prefix(v_context_p, token_limit)
        a_context_p = _slice_token_prefix(a_context_p, token_limit)

        video_state = LatentState(
            latent=_slice_token_prefix(video_state.latent, token_limit),
            denoise_mask=_slice_token_prefix(video_state.denoise_mask, token_limit),
            positions=_slice_positions_token_prefix(video_state.positions, token_limit),
            clean_latent=_slice_token_prefix(video_state.clean_latent, token_limit),
        )
        audio_state = LatentState(
            latent=_slice_token_prefix(audio_state.latent, token_limit),
            denoise_mask=_slice_token_prefix(audio_state.denoise_mask, token_limit),
            positions=_slice_positions_token_prefix(audio_state.positions, token_limit),
            clean_latent=_slice_token_prefix(audio_state.clean_latent, token_limit),
        )
        print(f"token-limited replay enabled: token_limit={token_limit}")

    pos_video = modality_from_latent_state(video_state, v_context_p, sigma)
    pos_audio = modality_from_latent_state(audio_state, a_context_p, sigma)

    transformer = pipeline.stage_2_model_ledger.transformer()

    # For debug
    # seen = {}

    # def dbg_hook(module, inputs, output):
    #     seen["called"] = True
    #     print("HOOK CALLED:", type(module))
    #     if isinstance(output, torch.Tensor):
    #         print("output shape:", output.shape)

    # handle = transformer.velocity_model.patchify_proj.register_forward_hook(dbg_hook)

    # denoised_video, denoised_audio = transformer(
    #     video=pos_video,
    #     audio=pos_audio,
    #     perturbations=None,
    # )

    # handle.remove()
    # print("manual hook fired?", seen.get("called", False))

    # -- End debug --

    inner = transformer.velocity_model
    print("inner type:", type(inner))
    print("inner named_modules:", len(list(inner.named_modules())))

    include_regexes = args.include
    if include_regexes:
        print("tracing with include regexes:", include_regexes)
    leaf_modules_only = not args.all_modules
    print("leaf_modules_only:", leaf_modules_only)
    max_capture_bytes = int(args.max_capture_gib * 1024**3)

    collector = zml_utils.ActivationCollector(
        transformer,
        stop_after_first_step=True,
        max_layers=5000,
        include_regexes=include_regexes,
        leaf_modules_only=leaf_modules_only,
        capture_inputs=args.capture_inputs,
        max_capture_bytes=max_capture_bytes,
    )

    # Intermediate diagnostic capture: for attn modules, also hook to_q, to_k, to_v outputs
    # to capture q, k, v post-projection (pre-rope). This helps validate head-split and rope application.
    # Stored as activations['{attn_name}.to_q.__output__'] = q_tensor (and same for k, v).
    captured_intermediates: dict[str, torch.Tensor] = {}
    intermediate_handles: list = []
    
    # Kwargs capture: for each module matching an include regex, register a
    # forward pre-hook (with_kwargs=True) to capture keyword arguments such as
    # pe, mask, and context that are NOT visible to standard forward hooks.
    # Results are stored as activations['{name}.__kwargs__'] = {kwarg: tensor}.
    captured_kwargs: dict[str, dict[str, Any]] = {}
    kwargs_handles: list = []
    captured_aux: dict[str, torch.Tensor] = {}
    aux_handles: list = []
    current_block_args: dict[str, dict[str, Any]] = {}
    text_ca_method_patches: list[tuple[torch.nn.Module, Any]] = []
    captured_sdpa: dict[str, torch.Tensor] = {}
    sdpa_call_shapes: list[dict[str, Any]] = []
    orig_sdpa = None
    sdpa_call_count = 0

    if args.capture_sdpa_call_idx is not None or args.log_sdpa_calls:
        target_idx = args.capture_sdpa_call_idx
        orig_sdpa = F.scaled_dot_product_attention

        def _sdpa_wrapper(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
            nonlocal sdpa_call_count
            out = orig_sdpa(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

            if args.log_sdpa_calls:
                sdpa_call_shapes.append(
                    {
                        "idx": sdpa_call_count,
                        "q": tuple(q.shape),
                        "k": tuple(k.shape),
                        "v": tuple(v.shape),
                        "out": tuple(out.shape),
                        "mask": tuple(attn_mask.shape) if isinstance(attn_mask, torch.Tensor) else None,
                    }
                )

            if target_idx is not None and sdpa_call_count == target_idx:
                captured_sdpa[f"__sdpa_call_{target_idx}__.q"] = q.detach().cpu().contiguous()
                captured_sdpa[f"__sdpa_call_{target_idx}__.k"] = k.detach().cpu().contiguous()
                captured_sdpa[f"__sdpa_call_{target_idx}__.v"] = v.detach().cpu().contiguous()
                captured_sdpa[f"__sdpa_call_{target_idx}__.out"] = out.detach().cpu().contiguous()
                if isinstance(attn_mask, torch.Tensor):
                    captured_sdpa[f"__sdpa_call_{target_idx}__.attn_mask"] = attn_mask.detach().cpu().contiguous()

            sdpa_call_count += 1
            return out

        F.scaled_dot_product_attention = _sdpa_wrapper
        if target_idx is not None:
            print(f"SDPA capture enabled for call index {target_idx}")
        if args.log_sdpa_calls:
            print("SDPA call logging enabled")
    if args.capture_kwargs and include_regexes:
        import re as _re

        def _normalize_kwarg_value(module: torch.nn.Module, key: str, value: Any, args_tup: tuple[Any, ...], kwargs: dict[str, Any]) -> Any | None:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()

            # LTX rotary kwargs are typically tuples: (cos, sin).
            if isinstance(value, (tuple, list)) and len(value) == 2 and all(isinstance(v, torch.Tensor) for v in value):
                return {
                    "cos": value[0].detach().cpu(),
                    "sin": value[1].detach().cpu(),
                }

            # xFormers may pass mask as an attention-bias object rather than a Tensor.
            # Materialize it to dense [B, H, Q, K] so parity checker can consume it.
            if key == "mask" and value is not None and hasattr(value, "materialize"):
                q_in = args_tup[0] if len(args_tup) > 0 else None
                if isinstance(q_in, torch.Tensor) and q_in.ndim >= 3:
                    b, q_len = int(q_in.shape[0]), int(q_in.shape[1])
                    ctx = kwargs.get("context")
                    k_len = int(ctx.shape[1]) if isinstance(ctx, torch.Tensor) and ctx.ndim >= 3 else q_len
                    heads = int(getattr(module, "heads", 1))
                    try:
                        dense_mask = value.materialize((b, heads, q_len, k_len), dtype=q_in.dtype, device=q_in.device)
                        if isinstance(dense_mask, torch.Tensor):
                            return dense_mask.detach().cpu()
                    except Exception as exc:
                        print(f"WARNING: could not materialize mask for {type(module).__name__}: {exc}")

            return None

        for name, mod in transformer.named_modules():
            if any(_re.search(rx, name) for rx in include_regexes):
                def _make_kwarg_hook(mod_name: str):
                    def _hook(module, args_tup, kwargs):
                        kw_values: dict[str, Any] = {}
                        for k, v in kwargs.items():
                            normalized = _normalize_kwarg_value(module, k, v, args_tup, kwargs)
                            if normalized is not None:
                                kw_values[k] = normalized
                        if kw_values:
                            captured_kwargs[mod_name] = kw_values
                        return None  # don't modify args
                    return _hook
                try:
                    handle = mod.register_forward_pre_hook(
                        _make_kwarg_hook(name), with_kwargs=True, prepend=True
                    )
                    kwargs_handles.append(handle)
                    print(f"kwargs hook registered for: {name}")
                except TypeError as exc:
                    print(f"WARNING: could not register kwargs hook for {name}: {exc}")

        # Also hook to_q, to_k, to_v outputs for modules matching the include regex.
        # This captures q, k, v tensors right after projection (pre-rope).
        # Only registers for modules whose parent attn module matches the include patterns.
        def _make_output_hook(output_key: str):
            def _hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    captured_intermediates[output_key] = output.detach().cpu().contiguous()
            return _hook

        for name, mod in transformer.named_modules():
            if not any(name.endswith(f".{proj}") for proj in ("to_q", "to_k", "to_v")):
                continue
            # The parent module name (e.g., "velocity_model.transformer_blocks.0.attn1")
            parent_name = name.rsplit(".", 1)[0]
            if not any(_re.search(rx, parent_name) for rx in include_regexes):
                continue
            proj = name.rsplit(".", 1)[1]  # "to_q", "to_k", or "to_v"
            try:
                h = mod.register_forward_hook(_make_output_hook(f"{name}.__output__"))
                intermediate_handles.append(h)
                print(f"{proj} hook registered for: {name}")
            except Exception as exc:
                print(f"WARNING: could not register {proj} hook for {name}: {exc}")

        # Capture block-level AdaLN gate directly from the model implementation.
        # This avoids reconstructing vgate_msa from heuristics on flattened inputs.
        for name, mod in transformer.named_modules():
            if name != "velocity_model.transformer_blocks.0":
                continue
            if not any(_re.search(rx, name) for rx in include_regexes):
                continue

            def _make_block0_aux_hook(mod_name: str):
                def _hook(module, args_tup, kwargs):
                    try:
                        video = kwargs.get("video") if isinstance(kwargs, dict) else None
                        if video is None and len(args_tup) > 0:
                            video = args_tup[0]

                        if video is None or not hasattr(video, "timesteps") or not hasattr(video, "x"):
                            return None

                        if not hasattr(module, "get_ada_values") or not hasattr(module, "scale_shift_table"):
                            return None

                        # M6: capture initial vx_in / ax_in at block entry.
                        if isinstance(video.x, torch.Tensor):
                            captured_aux[f"{mod_name}.__aux__.vx_in"] = video.x.detach().cpu().contiguous()

                        batch_size = int(video.x.shape[0])
                        _, _, vgate_msa = module.get_ada_values(
                            module.scale_shift_table,
                            batch_size,
                            video.timesteps,
                            slice(0, 3),
                        )
                        _, _, vgate_mlp = module.get_ada_values(
                            module.scale_shift_table,
                            batch_size,
                            video.timesteps,
                            slice(3, 6),
                        )
                        if isinstance(vgate_msa, torch.Tensor):
                            captured_aux[f"{mod_name}.__aux__.vgate_msa"] = vgate_msa.detach().cpu().contiguous()
                        if isinstance(vgate_mlp, torch.Tensor):
                            captured_aux[f"{mod_name}.__aux__.vgate_mlp"] = vgate_mlp.detach().cpu().contiguous()
                        if isinstance(video.timesteps, torch.Tensor):
                            captured_aux[f"{mod_name}.__aux__.video_timesteps"] = video.timesteps.detach().cpu().contiguous()
                        if hasattr(video, "prompt_timestep") and isinstance(video.prompt_timestep, torch.Tensor):
                            captured_aux[f"{mod_name}.__aux__.v_prompt_timestep"] = video.prompt_timestep.detach().cpu().contiguous()
                        if hasattr(video, "cross_scale_shift_timestep") and isinstance(video.cross_scale_shift_timestep, torch.Tensor):
                            captured_aux[f"{mod_name}.__aux__.v_cross_ss_ts"] = video.cross_scale_shift_timestep.detach().cpu().contiguous()
                        if hasattr(video, "cross_gate_timestep") and isinstance(video.cross_gate_timestep, torch.Tensor):
                            captured_aux[f"{mod_name}.__aux__.v_cross_gate_ts"] = video.cross_gate_timestep.detach().cpu().contiguous()

                        audio = kwargs.get("audio") if isinstance(kwargs, dict) else None
                        if audio is None and len(args_tup) > 1:
                            audio = args_tup[1]
                        perturbations = kwargs.get("perturbations") if isinstance(kwargs, dict) else None
                        if perturbations is None and len(args_tup) > 2:
                            perturbations = args_tup[2]

                        if audio is not None and hasattr(audio, "x") and isinstance(audio.x, torch.Tensor):
                            captured_aux[f"{mod_name}.__aux__.ax_in"] = audio.x.detach().cpu().contiguous()

                        current_block_args[mod_name] = {
                            "video": video,
                            "audio": audio,
                            "perturbations": perturbations,
                        }

                        if (
                            audio is not None
                            and hasattr(audio, "timesteps")
                            and hasattr(module, "audio_scale_shift_table")
                        ):
                            _, _, agate_msa = module.get_ada_values(
                                module.audio_scale_shift_table,
                                batch_size,
                                audio.timesteps,
                                slice(0, 3),
                            )
                            _, _, agate_mlp = module.get_ada_values(
                                module.audio_scale_shift_table,
                                batch_size,
                                audio.timesteps,
                                slice(3, 6),
                            )
                            if isinstance(agate_msa, torch.Tensor):
                                captured_aux[f"{mod_name}.__aux__.agate_msa"] = agate_msa.detach().cpu().contiguous()
                            if isinstance(agate_mlp, torch.Tensor):
                                captured_aux[f"{mod_name}.__aux__.agate_mlp"] = agate_mlp.detach().cpu().contiguous()
                            if isinstance(audio.timesteps, torch.Tensor):
                                captured_aux[f"{mod_name}.__aux__.audio_timesteps"] = audio.timesteps.detach().cpu().contiguous()
                            if hasattr(audio, "prompt_timestep") and isinstance(audio.prompt_timestep, torch.Tensor):
                                captured_aux[f"{mod_name}.__aux__.a_prompt_timestep"] = audio.prompt_timestep.detach().cpu().contiguous()
                            if hasattr(audio, "cross_scale_shift_timestep") and isinstance(audio.cross_scale_shift_timestep, torch.Tensor):
                                captured_aux[f"{mod_name}.__aux__.a_cross_ss_ts"] = audio.cross_scale_shift_timestep.detach().cpu().contiguous()
                            if hasattr(audio, "cross_gate_timestep") and isinstance(audio.cross_gate_timestep, torch.Tensor):
                                captured_aux[f"{mod_name}.__aux__.a_cross_gate_ts"] = audio.cross_gate_timestep.detach().cpu().contiguous()

                        # M5 gates/masks: AV cross-attn branches A->V and V->A.
                        if (
                            video is not None
                            and audio is not None
                            and hasattr(module, "get_av_ca_ada_values")
                            and hasattr(module, "scale_shift_table_a2v_ca_video")
                            and hasattr(module, "scale_shift_table_a2v_ca_audio")
                            and hasattr(video, "cross_scale_shift_timestep")
                            and hasattr(video, "cross_gate_timestep")
                            and hasattr(audio, "cross_scale_shift_timestep")
                            and hasattr(audio, "cross_gate_timestep")
                            and hasattr(video, "x")
                            and hasattr(audio, "x")
                        ):
                            _, _, gate_out_a2v = module.get_av_ca_ada_values(
                                module.scale_shift_table_a2v_ca_video,
                                video.x.shape[0],
                                video.cross_scale_shift_timestep,
                                video.cross_gate_timestep,
                                slice(0, 2),
                            )
                            _, _, gate_out_v2a = module.get_av_ca_ada_values(
                                module.scale_shift_table_a2v_ca_audio,
                                audio.x.shape[0],
                                audio.cross_scale_shift_timestep,
                                audio.cross_gate_timestep,
                                slice(2, 4),
                            )

                            if perturbations is not None:
                                try:
                                    a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, module.idx, video.x)
                                except Exception:
                                    a2v_mask = torch.ones_like(video.x)
                                try:
                                    v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, module.idx, audio.x)
                                except Exception:
                                    v2a_mask = torch.ones_like(audio.x)
                            else:
                                a2v_mask = torch.ones_like(video.x)
                                v2a_mask = torch.ones_like(audio.x)

                            if isinstance(gate_out_a2v, torch.Tensor):
                                captured_aux[f"{mod_name}.__aux__.a2v_gate"] = gate_out_a2v.detach().cpu().contiguous()
                            if isinstance(a2v_mask, torch.Tensor):
                                captured_aux[f"{mod_name}.__aux__.a2v_mask"] = a2v_mask.detach().cpu().contiguous()
                            if isinstance(gate_out_v2a, torch.Tensor):
                                captured_aux[f"{mod_name}.__aux__.v2a_gate"] = gate_out_v2a.detach().cpu().contiguous()
                            if isinstance(v2a_mask, torch.Tensor):
                                captured_aux[f"{mod_name}.__aux__.v2a_mask"] = v2a_mask.detach().cpu().contiguous()
                    except Exception as exc:
                        print(f"WARNING: failed to capture block0 aux tensors: {exc}")
                    return None

                return _hook

            try:
                h = mod.register_forward_pre_hook(
                    _make_block0_aux_hook(name), with_kwargs=True, prepend=True
                )
                aux_handles.append(h)
                print(f"aux hook registered for: {name}")
            except Exception as exc:
                print(f"WARNING: could not register aux hook for {name}: {exc}")

            # M6: post-hook to capture final vx_out / ax_out at block exit.
            def _make_block0_out_hook(mod_name: str):
                def _hook(module, args_tup, kwargs, out):
                    try:
                        video_out = out[0] if isinstance(out, (tuple, list)) else out
                        audio_out = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
                        if video_out is not None and hasattr(video_out, "x") and isinstance(video_out.x, torch.Tensor):
                            captured_aux[f"{mod_name}.__aux__.vx_out"] = video_out.x.detach().cpu().contiguous()
                        if audio_out is not None and hasattr(audio_out, "x") and isinstance(audio_out.x, torch.Tensor):
                            captured_aux[f"{mod_name}.__aux__.ax_out"] = audio_out.x.detach().cpu().contiguous()
                    except Exception as exc:
                        print(f"WARNING: failed to capture block0 output tensors: {exc}")
                    return None
                return _hook

            try:
                h = mod.register_forward_hook(
                    _make_block0_out_hook(name), with_kwargs=True
                )
                aux_handles.append(h)
                print(f"out hook registered for: {name}")
            except Exception as exc:
                print(f"WARNING: could not register out hook for {name}: {exc}")

            # M5 capture: AV cross-attn branch deltas and gates for A->V / V->A.
            # NOTE: `block_mod` is passed explicitly to avoid the Python late-binding
            # closure bug where `mod` would refer to whatever the loop variable holds
            # at hook invocation time (a later submodule that lacks get_av_ca_ada_values).
            def _make_av_attn_hook(mod_name: str, direction: str, block_mod):
                def _hook(attn_module, args_tup, kwargs, out):
                    try:
                        if not isinstance(out, torch.Tensor):
                            return None

                        x = args_tup[0] if len(args_tup) > 0 else None
                        if not isinstance(x, torch.Tensor):
                            return None

                        state = current_block_args.get(mod_name, {})
                        video = state.get("video")
                        audio = state.get("audio")
                        perturbations = state.get("perturbations")

                        gate = None
                        mask = None

                        if direction == "a2v":
                            if (
                                video is not None
                                and hasattr(block_mod, "get_av_ca_ada_values")
                                and hasattr(block_mod, "scale_shift_table_a2v_ca_video")
                                and hasattr(video, "cross_scale_shift_timestep")
                                and hasattr(video, "cross_gate_timestep")
                            ):
                                _, _, gate = block_mod.get_av_ca_ada_values(
                                    block_mod.scale_shift_table_a2v_ca_video,
                                    x.shape[0],
                                    video.cross_scale_shift_timestep,
                                    video.cross_gate_timestep,
                                    slice(0, 2),
                                )
                            if perturbations is not None:
                                try:
                                    mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, block_mod.idx, out)
                                except Exception:
                                    mask = None
                        else:
                            if (
                                audio is not None
                                and hasattr(block_mod, "get_av_ca_ada_values")
                                and hasattr(block_mod, "scale_shift_table_a2v_ca_audio")
                                and hasattr(audio, "cross_scale_shift_timestep")
                                and hasattr(audio, "cross_gate_timestep")
                            ):
                                _, _, gate = block_mod.get_av_ca_ada_values(
                                    block_mod.scale_shift_table_a2v_ca_audio,
                                    x.shape[0],
                                    audio.cross_scale_shift_timestep,
                                    audio.cross_gate_timestep,
                                    slice(2, 4),
                                )
                            if perturbations is not None:
                                try:
                                    mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, block_mod.idx, out)
                                except Exception:
                                    mask = None

                        if gate is None:
                            return None
                        if mask is None:
                            mask = torch.ones_like(out)

                        delta = out * gate * mask

                        if direction == "a2v":
                            captured_aux[f"{mod_name}.__aux__.a2v_attn_out"] = out.detach().cpu().contiguous()
                            captured_aux[f"{mod_name}.__aux__.a2v_gate"] = gate.detach().cpu().contiguous()
                            captured_aux[f"{mod_name}.__aux__.a2v_mask"] = mask.detach().cpu().contiguous()
                            captured_aux[f"{mod_name}.__aux__.a2v_delta"] = delta.detach().cpu().contiguous()
                        else:
                            captured_aux[f"{mod_name}.__aux__.v2a_attn_out"] = out.detach().cpu().contiguous()
                            captured_aux[f"{mod_name}.__aux__.v2a_gate"] = gate.detach().cpu().contiguous()
                            captured_aux[f"{mod_name}.__aux__.v2a_mask"] = mask.detach().cpu().contiguous()
                            captured_aux[f"{mod_name}.__aux__.v2a_delta"] = delta.detach().cpu().contiguous()
                    except Exception as exc:
                        print(f"WARNING: failed to capture AV cross-attn aux tensors: {exc}")
                    return None

                return _hook

            if hasattr(mod, "audio_to_video_attn"):
                try:
                    h = mod.audio_to_video_attn.register_forward_hook(
                        _make_av_attn_hook(name, "a2v", mod),
                        with_kwargs=True,
                    )
                    aux_handles.append(h)
                    print(f"a2v aux hook registered for: {name}.audio_to_video_attn")
                except Exception as exc:
                    print(f"WARNING: could not register a2v aux hook for {name}: {exc}")

            if hasattr(mod, "video_to_audio_attn"):
                try:
                    h = mod.video_to_audio_attn.register_forward_hook(
                        _make_av_attn_hook(name, "v2a", mod),
                        with_kwargs=True,
                    )
                    aux_handles.append(h)
                    print(f"v2a aux hook registered for: {name}.video_to_audio_attn")
                except Exception as exc:
                    print(f"WARNING: could not register v2a aux hook for {name}: {exc}")

            # M2 capture: wrap _apply_text_cross_attention to capture the exact
            # residual delta returned by Python block logic.
            if hasattr(mod, "_apply_text_cross_attention"):
                orig_text_ca = getattr(mod, "_apply_text_cross_attention")

                def _make_text_ca_wrapper(mod_name: str, orig_fn: Any):
                    def _wrapped(
                        self,
                        x,
                        context,
                        attn,
                        scale_shift_table,
                        prompt_scale_shift_table,
                        timestep,
                        prompt_timestep,
                        context_mask,
                        cross_attention_adaln=False,
                    ):
                        out = orig_fn(
                            x,
                            context,
                            attn,
                            scale_shift_table,
                            prompt_scale_shift_table,
                            timestep,
                            prompt_timestep,
                            context_mask,
                            cross_attention_adaln=cross_attention_adaln,
                        )
                        try:
                            is_video_text_ca = hasattr(self, "attn2") and (attn is self.attn2)
                            is_audio_text_ca = hasattr(self, "audio_attn2") and (attn is self.audio_attn2)

                            if isinstance(x, torch.Tensor):
                                if is_video_text_ca:
                                    captured_aux[f"{mod_name}.__aux__.text_ca_vx_in"] = x.detach().cpu().contiguous()
                                elif is_audio_text_ca:
                                    captured_aux[f"{mod_name}.__aux__.audio_text_ca_ax_in"] = x.detach().cpu().contiguous()
                            if isinstance(out, torch.Tensor):
                                if is_video_text_ca:
                                    captured_aux[f"{mod_name}.__aux__.text_ca_out"] = out.detach().cpu().contiguous()
                                elif is_audio_text_ca:
                                    captured_aux[f"{mod_name}.__aux__.audio_text_ca_out"] = out.detach().cpu().contiguous()
                            if isinstance(context, torch.Tensor):
                                if is_video_text_ca:
                                    captured_aux[f"{mod_name}.__aux__.text_ca_context"] = context.detach().cpu().contiguous()
                                elif is_audio_text_ca:
                                    captured_aux[f"{mod_name}.__aux__.audio_text_ca_context"] = context.detach().cpu().contiguous()
                            if isinstance(context_mask, torch.Tensor):
                                if is_video_text_ca:
                                    captured_aux[f"{mod_name}.__aux__.text_ca_context_mask"] = context_mask.detach().cpu().contiguous()
                                elif is_audio_text_ca:
                                    captured_aux[f"{mod_name}.__aux__.audio_text_ca_context_mask"] = context_mask.detach().cpu().contiguous()
                        except Exception as exc:
                            print(f"WARNING: failed to capture text-ca aux tensors: {exc}")
                        return out

                    return _wrapped

                try:
                    wrapped = types.MethodType(_make_text_ca_wrapper(name, orig_text_ca), mod)
                    setattr(mod, "_apply_text_cross_attention", wrapped)
                    text_ca_method_patches.append((mod, orig_text_ca))
                    print(f"text-ca wrapper installed for: {name}")
                except Exception as exc:
                    print(f"WARNING: could not install text-ca wrapper for {name}: {exc}")

    print("Starting transformer forward + activation collection...")
    try:
        (denoised_video, denoised_audio), activations = collector(
            video=pos_video,
            audio=pos_audio,
            perturbations=None,
        )
    finally:
        if orig_sdpa is not None:
            F.scaled_dot_product_attention = orig_sdpa
        for mod, orig_fn in text_ca_method_patches:
            try:
                setattr(mod, "_apply_text_cross_attention", orig_fn)
            except Exception as exc:
                print(f"WARNING: failed restoring text-ca wrapper: {exc}")

    # Remove kwargs pre-hooks and merge their captures into activations.
    for h in kwargs_handles:
        h.remove()
    for h in intermediate_handles:
        h.remove()
    for h in aux_handles:
        h.remove()
    for mod_name, kw_tensors in captured_kwargs.items():
        activations[f"{mod_name}.__kwargs__"] = kw_tensors
        print(f"kwargs captured for {mod_name}: {list(kw_tensors.keys())}")
    for mod_name, tensor in captured_intermediates.items():
        activations[mod_name] = tensor
        print(f"intermediate captured for {mod_name}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")
    for key, tensor in captured_aux.items():
        activations[key] = tensor
        print(f"aux captured for {key}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")
    for key, tensor in captured_sdpa.items():
        activations[key] = tensor
        print(f"sdpa captured for {key}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")
    if args.log_sdpa_calls:
        print("sdpa call shapes observed:")
        for rec in sdpa_call_shapes:
            print(
                "  idx={idx} q={q} k={k} v={v} out={out} mask={mask}".format(**rec)
            )
    if args.capture_sdpa_call_idx is not None or args.log_sdpa_calls:
        print(f"sdpa calls observed: {sdpa_call_count}")

    print("denoised_video shape:", denoised_video.shape)
    print("denoised_audio shape:", denoised_audio.shape)
    print("activation entries:", len(activations))

    token_suffix = f"_t{args.token_limit}" if args.token_limit is not None else ""
    output_path = TRACE_DIR / f"acts_stage2_transformer_step_{step_idx:03d}_{args.pass_label}{token_suffix}.pt"
    torch.save(
        {
            "step_idx": step_idx,
            "pass_label": args.pass_label,
            "token_limit": args.token_limit,
            "include_regexes": include_regexes,
            "leaf_modules_only": leaf_modules_only,
            "capture_inputs": args.capture_inputs,
            "max_capture_gib": args.max_capture_gib,
            "denoised_video": denoised_video.detach().cpu(),
            "denoised_audio": denoised_audio.detach().cpu(),
            "activations": activations,
        },
        output_path,
    )

    print("saved:", output_path)
    print("activation entries:", len(activations))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
