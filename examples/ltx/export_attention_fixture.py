import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from export_activation_fixture import resolve_activation_key


ACTIVATION_KEYS = {
    "attn1": "velocity_model.transformer_blocks.0.attn1",
    "attn2": "velocity_model.transformer_blocks.0.attn2",
    "audio_attn1": "velocity_model.transformer_blocks.0.audio_attn1",
    "audio_attn2": "velocity_model.transformer_blocks.0.audio_attn2",
    "audio_to_video_attn": "velocity_model.transformer_blocks.0.audio_to_video_attn",
    "video_to_audio_attn": "velocity_model.transformer_blocks.0.video_to_audio_attn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export attention activation fixture from replay .pt to safetensors")
    parser.add_argument("input_pt", type=Path, help="Path to acts_stage2_transformer_step_... .pt")
    parser.add_argument("output_st", type=Path, help="Output safetensors path")
    parser.add_argument(
        "--mode",
        required=True,
        choices=sorted(ACTIVATION_KEYS.keys()),
        help="Attention component to export",
    )
    parser.add_argument(
        "--activation-key",
        default=None,
        help="Override activation key inside obj['activations']",
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=None,
        help="Optional token prefix length to export for query-side tensors",
    )
    return parser.parse_args()


def _slice_query_prefix(name: str, tensor: torch.Tensor, token_limit: int) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor

    if name.endswith(".mask0") and tensor.ndim >= 4:
        # Typical attention mask layout: [B, 1, T, T].
        return tensor[..., :token_limit, :token_limit].contiguous()

    if name.endswith(".pe_cos0") or name.endswith(".pe_sin0"):
        # Rotary tensors can appear as [B, H, T, HD], [T, H*HD], or [T, ...].
        if tensor.ndim == 4:
            return tensor[:, :, :token_limit, :].contiguous()
        if tensor.ndim >= 2:
            return tensor[:token_limit, ...].contiguous()

    if name.endswith(".k_pe_cos0") or name.endswith(".k_pe_sin0"):
        # Key-side rotary tensors follow the same convention as pe tensors.
        if tensor.ndim == 4:
            return tensor[:, :, :token_limit, :].contiguous()
        if tensor.ndim >= 2:
            return tensor[:token_limit, ...].contiguous()

    if tensor.ndim >= 2:
        # Default attention activations are query-token-major on dim=1: [B, T, ...].
        return tensor[:, :token_limit, ...].contiguous()

    return tensor


def main() -> None:
    args = parse_args()
    activation_key = args.activation_key or ACTIVATION_KEYS[args.mode]

    obj = torch.load(args.input_pt, map_location="cpu", weights_only=False)
    acts = obj.get("activations", {})

    input0 = None
    output0 = None
    resolved_key = None

    # Preferred path: full attention module was hooked.
    if activation_key in acts:
        resolved_key = activation_key
        entry = acts[activation_key]
        if not isinstance(entry, dict):
            raise TypeError(f"Unexpected activation entry type: {type(entry)}")

        inputs = entry.get("input", [])
        outputs = entry.get("output", [])
        if len(inputs) < 1 or len(outputs) < 1:
            raise ValueError(
                f"Expected non-empty input/output for {activation_key}, got input={len(inputs)} output={len(outputs)}"
            )

        input0 = inputs[0].detach().cpu().contiguous()
        output0 = outputs[0].detach().cpu().contiguous()
    else:
        # Fallback path: capture usually contains leaf modules only.
        # Attention input is identical to to_q input, and attention output is to_out.0 output.
        q_key = resolve_activation_key(acts, activation_key + ".to_q", allow_proj_suffix=False)

        out_key = None
        out_err = None
        for candidate in (
            activation_key + ".to_out.0",
            activation_key + ".to_out",
            activation_key,
        ):
            try:
                out_key = resolve_activation_key(acts, candidate, allow_proj_suffix=False)
                break
            except KeyError as err:
                out_err = err

        if out_key is None:
            # Some traces only include q/k/v projections. That is insufficient for full attention parity,
            # because checker output is post-attention and post-to_out.
            available = sorted([k for k in acts.keys() if k.startswith(activation_key + ".")])
            raise ValueError(
                "Trace is missing required attention output hook for full parity. "
                f"Could not find any of: '{activation_key}.to_out.0', '{activation_key}.to_out', '{activation_key}'. "
                f"Available under prefix: {available}. "
                "Regenerate replay with --all-modules and an include regex that explicitly captures "
                "to_gate_logits and to_out(.0), e.g. "
                rf"^({activation_key}\\.(q_norm|k_norm|to_q|to_k|to_v|to_gate_logits|to_out(\\.0)?))(\\.|$)"
            ) from out_err

        q_entry = acts[q_key]
        out_entry = acts[out_key]
        if not isinstance(q_entry, dict) or not isinstance(out_entry, dict):
            raise TypeError("Unexpected activation entry type for attention fallback")

        # ActivationCollector stores input/output as tuples (from PyTorch hooks).
        # Normalize to list-like for consistent access.
        q_inputs = q_entry.get("input") or []
        out_outputs = out_entry.get("output") or []
        if not isinstance(q_inputs, (list, tuple)) or len(q_inputs) < 1:
            raise ValueError(
                f"Expected non-empty input for {q_key}, got type={type(q_inputs)} len={len(q_inputs) if isinstance(q_inputs, (list, tuple)) else '?'}"
            )
        if not isinstance(out_outputs, (list, tuple)) or len(out_outputs) < 1:
            raise ValueError(
                f"Expected non-empty output for {out_key}, got type={type(out_outputs)} len={len(out_outputs) if isinstance(out_outputs, (list, tuple)) else '?'}"
            )

        # Extract first input and output tensors (others are typically gradients or unused)
        input0 = q_inputs[0].detach().cpu().contiguous() if isinstance(q_inputs[0], torch.Tensor) else q_inputs[0][0].detach().cpu().contiguous()
        output0 = out_outputs[0].detach().cpu().contiguous() if isinstance(out_outputs[0], torch.Tensor) else out_outputs[0][0].detach().cpu().contiguous()
        resolved_key = f"{activation_key} [fallback: input={q_key}, output={out_key}]"

    input_name = f"{args.mode}.input0"
    output_name = f"{args.mode}.output0"
    tensors = {input_name: input0, output_name: output0}

    # Optional kwargs captured via --capture-kwargs in replay.
    # Stored in acts as '{activation_key}.__kwargs__'.
    # For rotary kwargs, we expect tuple-preserving capture under:
    #   {'pe': {'cos': tensor, 'sin': tensor}, ...}
    kwargs_key = activation_key + ".__kwargs__"
    captured_kw = acts.get(kwargs_key, {})
    if not isinstance(captured_kw, dict):
        captured_kw = {}

    kwarg_shapes: dict[str, str] = {}
    for kw_name in ("mask", "context"):
        kw_val = captured_kw.get(kw_name)
        if kw_val is not None and isinstance(kw_val, torch.Tensor):
            tensor_name = f"{args.mode}.{kw_name}0"
            tensors[tensor_name] = kw_val.detach().cpu().contiguous()
            kwarg_shapes[kw_name] = f"shape={tuple(kw_val.shape)} dtype={kw_val.dtype}"

    for kw_name in ("pe", "k_pe"):
        kw_val = captured_kw.get(kw_name)
        if isinstance(kw_val, dict):
            cos = kw_val.get("cos")
            sin = kw_val.get("sin")
            if isinstance(cos, torch.Tensor) and isinstance(sin, torch.Tensor):
                cos_name = f"{args.mode}.{kw_name}_cos0"
                sin_name = f"{args.mode}.{kw_name}_sin0"
                tensors[cos_name] = cos.detach().cpu().contiguous()
                tensors[sin_name] = sin.detach().cpu().contiguous()
                kwarg_shapes[f"{kw_name}_cos"] = f"shape={tuple(cos.shape)} dtype={cos.dtype}"
                kwarg_shapes[f"{kw_name}_sin"] = f"shape={tuple(sin.shape)} dtype={sin.dtype}"
        elif isinstance(kw_val, torch.Tensor):
            # Backward compatibility for older captures that flattened pe to one tensor.
            tensor_name = f"{args.mode}.{kw_name}0"
            tensors[tensor_name] = kw_val.detach().cpu().contiguous()
            kwarg_shapes[kw_name] = f"shape={tuple(kw_val.shape)} dtype={kw_val.dtype}"

    if kwarg_shapes:
        print(f"kwargs found under {kwargs_key}: {list(kwarg_shapes.keys())}")
    else:
        print(
            f"No kwargs found under {kwargs_key}. "
            "Re-run replay with --capture-kwargs (and refine --include so attn1 is matched) "
            "to capture pe_cos/pe_sin and mask for RoPE parity."
        )

    # Diagnostic intermediates: if captured, export q, k, v post-projection (pre-norm/pre-rope).
    # These are used for diagnostic comparison only.
    diagnostic_keys: dict[str, str] = {}
    for proj_type in ("to_q", "to_k", "to_v"):
        intermediate_key = f"{activation_key}.{proj_type}.__output__"
        if intermediate_key in acts:
            try:
                tensor = acts[intermediate_key]
                if isinstance(tensor, torch.Tensor):
                    tensor_name = f"{args.mode}.{proj_type}_diag0"
                    tensors[tensor_name] = tensor.detach().cpu().contiguous()
                    diagnostic_keys[proj_type] = f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"
                    print(f"diagnostic {proj_type} captured: shape={tuple(tensor.shape)} dtype={tensor.dtype}")
            except Exception as e:
                print(f"WARNING: could not extract diagnostic {proj_type}: {e}")

    # Also export q_norm and k_norm outputs (post-norm, pre-head-split, pre-rope).
    # These are the values that go into head-split + RoPE in LTX, so they are
    # the correct reference for validating ZML's head-split and rotary application.
    for norm_type in ("q_norm", "k_norm"):
        norm_key = f"{activation_key}.{norm_type}"
        if norm_key in acts:
            try:
                entry = acts[norm_key]
                if isinstance(entry, dict):
                    outputs = entry.get("output", [])
                    if outputs and isinstance(outputs[0], torch.Tensor):
                        tensor = outputs[0].detach().cpu().contiguous()
                        tensor_name = f"{args.mode}.{norm_type}_diag0"
                        tensors[tensor_name] = tensor
                        diagnostic_keys[norm_type] = f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"
                        print(f"diagnostic {norm_type} captured: shape={tuple(tensor.shape)} dtype={tensor.dtype}")
            except Exception as e:
                print(f"WARNING: could not extract diagnostic {norm_type}: {e}")

    # If replay captured a specific SDPA call (via --capture-sdpa-call-idx), export those tensors.
    # Expected captured layout from PyTorch SDPA is [B, H, T, HD]; convert to [B, T, H, HD]
    # for consistency with the rest of ZML diagnostics.
    sdpa_prefixes = [k.rsplit(".", 1)[0] for k in acts.keys() if k.startswith("__sdpa_call_") and k.endswith(".q")]
    if sdpa_prefixes:
        # If multiple captured calls exist, use the first in lexical order for determinism.
        sdpa_prefix = sorted(sdpa_prefixes)[0]

        def _to_bthd(t: torch.Tensor) -> torch.Tensor:
            if t.ndim == 4:
                return t.permute(0, 2, 1, 3).contiguous()
            return t

        for sdpa_name in ("q", "k", "v", "out"):
            key = f"{sdpa_prefix}.{sdpa_name}"
            t = acts.get(key)
            if isinstance(t, torch.Tensor):
                tensor_name = f"{args.mode}.sdpa_{sdpa_name}_diag0"
                tensors[tensor_name] = _to_bthd(t.detach().cpu().contiguous())
                diagnostic_keys[f"sdpa_{sdpa_name}"] = f"shape={tuple(tensors[tensor_name].shape)} dtype={tensors[tensor_name].dtype}"
                print(f"diagnostic sdpa_{sdpa_name} captured: shape={tuple(tensors[tensor_name].shape)} dtype={tensors[tensor_name].dtype}")

    # Export gate logits if captured: [B, T, H]
    gate_key = f"{activation_key}.to_gate_logits"
    if gate_key in acts:
        try:
            entry = acts[gate_key]
            if isinstance(entry, dict):
                outputs = entry.get("output", [])
                if outputs and isinstance(outputs[0], torch.Tensor):
                    tensor = outputs[0].detach().cpu().contiguous()
                    tensor_name = f"{args.mode}.to_gate_logits_diag0"
                    tensors[tensor_name] = tensor
                    diagnostic_keys["to_gate_logits"] = f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"
                    print(f"diagnostic to_gate_logits captured: shape={tuple(tensor.shape)} dtype={tensor.dtype}")
        except Exception as e:
            print(f"WARNING: could not extract diagnostic to_gate_logits: {e}")

    # Export to_out input if captured: this is the merged pre-to_out tensor [B, T, D_V]
    if resolved_key is not None and "fallback: input=" in resolved_key:
        try:
            # Extract out_key from fallback marker: "... output=<out_key>]"
            marker = "output="
            idx = resolved_key.rfind(marker)
            if idx != -1:
                out_key = resolved_key[idx + len(marker):].rstrip("]")
                out_entry = acts.get(out_key)
                if isinstance(out_entry, dict):
                    out_inputs = out_entry.get("input", [])
                    if out_inputs and isinstance(out_inputs[0], torch.Tensor):
                        tensor = out_inputs[0].detach().cpu().contiguous()
                        tensor_name = f"{args.mode}.to_out_input_diag0"
                        tensors[tensor_name] = tensor
                        diagnostic_keys["to_out_input"] = f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"
                        print(f"diagnostic to_out_input captured: shape={tuple(tensor.shape)} dtype={tensor.dtype}")
        except Exception as e:
            print(f"WARNING: could not extract diagnostic to_out_input: {e}")

    if args.token_limit is not None:
        token_limit = args.token_limit
        if token_limit <= 0:
            raise ValueError(f"--token-limit must be > 0, got {token_limit}")

        sliced_tensors: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            sliced_tensors[name] = _slice_query_prefix(name, tensor, token_limit)
        tensors = sliced_tensors
        print(f"applied token_limit={token_limit} to exported fixture tensors")

    metadata = {
        "source_pt": str(args.input_pt),
        "activation_key": resolved_key,
        "step_idx": str(obj.get("step_idx", "")),
        "pass_label": str(obj.get("pass_label", "")),
        "dtype": str(output0.dtype),
        "has_pe": str("pe_cos" in kwarg_shapes and "pe_sin" in kwarg_shapes),
        "has_mask": str("mask" in kwarg_shapes),
        "has_diagnostics": str(len(diagnostic_keys) > 0),
        "token_limit": "" if args.token_limit is None else str(args.token_limit),
    }

    args.output_st.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output_st), metadata=metadata)

    print(f"saved: {args.output_st}")
    print(f"activation_key={resolved_key}")
    print(f"{input_name} shape={tuple(input0.shape)} dtype={input0.dtype} nbytes={input0.nelement() * input0.element_size()}")
    print(f"{output_name} shape={tuple(output0.shape)} dtype={output0.dtype} nbytes={output0.nelement() * output0.element_size()}")
    for kw_name, shape_str in kwarg_shapes.items():
        if kw_name.endswith("_cos") or kw_name.endswith("_sin"):
            tensor_name = f"{args.mode}.{kw_name}0"
        else:
            tensor_name = f"{args.mode}.{kw_name}0"
        print(f"{tensor_name} {shape_str}")
    for proj_type, shape_str in diagnostic_keys.items():
        tensor_name = f"{args.mode}.{proj_type}_diag0"
        print(f"{tensor_name} {shape_str}")


if __name__ == "__main__":
    main()
