import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate pre-to_out attention semantics against fixture")
    p.add_argument("fixture_path", type=Path, help="Path to attn1_diag.safetensors")
    p.add_argument("--attn-name", default="attn1")
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--token-limit", type=int, default=256)
    p.add_argument("--abs-tol", type=float, default=0.1)
    p.add_argument("--rel-tol", type=float, default=0.01)
    return p.parse_args()


def apply_head_split_bthd(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    # [B,T,D] -> [B,T,H,HD]
    b, t, d = x.shape
    hd = d // num_heads
    return x.view(b, t, num_heads, hd).contiguous()


def apply_rope_split_bthd(x_bthd: torch.Tensor, cos_bhtd: torch.Tensor, sin_bhtd: torch.Tensor) -> torch.Tensor:
    # Convert to [B,H,T,HD] for easy elementwise math with cos/sin.
    x = x_bthd.permute(0, 2, 1, 3).contiguous()
    hd = x.shape[-1]
    half = hd // 2

    x_first = x[..., :half]
    x_second = x[..., half:]

    out_first = x_first * cos_bhtd - x_second * sin_bhtd
    out_second = x_second * cos_bhtd + x_first * sin_bhtd
    out = torch.cat([out_first, out_second], dim=-1)

    # Back to [B,T,H,HD]
    return out.permute(0, 2, 1, 3).contiguous()


def compare(computed: torch.Tensor, expected: torch.Tensor, abs_tol: float, rel_tol: float) -> tuple[float, float, float]:
    computed = computed.float()
    expected = expected.float()

    abs_err = (computed - expected).abs()
    rel_err = abs_err / expected.abs().clamp_min(1e-12)
    close = (abs_err <= abs_tol) | (rel_err <= rel_tol)

    return float(abs_err.max().item()), float(abs_err.mean().item()), float(close.float().mean().item())


def run_variant(
    name: str,
    q_rot_bthd: torch.Tensor,
    k_rot_bthd: torch.Tensor,
    v_bthd: torch.Tensor,
    gate_logits_bth: torch.Tensor,
    expected_pre_to_out_btd: torch.Tensor,
    abs_tol: float,
    rel_tol: float,
    gate_mode: str,
    use_sdpa: bool,
    head_first: bool,
) -> None:
    if head_first:
        # [B,T,H,HD] -> [B,H,T,HD]
        q = q_rot_bthd.permute(0, 2, 1, 3).contiguous()
        k = k_rot_bthd.permute(0, 2, 1, 3).contiguous()
        v = v_bthd.permute(0, 2, 1, 3).contiguous()
    else:
        # Keep [B,T,H,HD] and permute to [B,H,T,HD] anyway for torch attention ops.
        q = q_rot_bthd.permute(0, 2, 1, 3).contiguous()
        k = k_rot_bthd.permute(0, 2, 1, 3).contiguous()
        v = v_bthd.permute(0, 2, 1, 3).contiguous()

    if use_sdpa:
        # sdpa returns [B,H,T,HD]
        attn = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=None, dropout_p=0.0, is_causal=False)
    else:
        # Manual: softmax(q @ k^T / sqrt(hd)) @ v
        scale = 1.0 / (q.shape[-1] ** 0.5)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, v.float())

    # [B,H,T,HD] -> [B,T,H,HD]
    attn_bthd = attn.permute(0, 2, 1, 3).contiguous()

    gate = gate_logits_bth.float()
    if gate_mode == "none":
        gate_bth = torch.ones_like(gate)
    elif gate_mode == "sigmoid":
        gate_bth = torch.sigmoid(gate)
    elif gate_mode == "sigmoid_x2":
        gate_bth = torch.sigmoid(gate) * 2.0
    elif gate_mode == "raw":
        gate_bth = gate
    else:
        raise ValueError(f"Unknown gate mode: {gate_mode}")

    gated = attn_bthd * gate_bth.unsqueeze(-1)

    # Flatten [B,T,H,HD] -> [B,T,DV]
    pre_to_out_btd = gated.flatten(start_dim=2)

    mx, mean, frac = compare(pre_to_out_btd, expected_pre_to_out_btd, abs_tol, rel_tol)
    print(f"{name}: max_abs_error={mx:.4f}, mean_abs_error={mean:.4f}, close_fraction={frac:.4f}")


def main() -> None:
    args = parse_args()
    tensors = load_file(str(args.fixture_path))

    q_norm = tensors[f"{args.attn_name}.q_norm_diag0"]
    k_norm = tensors[f"{args.attn_name}.k_norm_diag0"]
    v = tensors[f"{args.attn_name}.to_v_diag0"]
    gate_logits = tensors[f"{args.attn_name}.to_gate_logits_diag0"]
    pre_to_out_expected = tensors[f"{args.attn_name}.to_out_input_diag0"]
    pe_cos = tensors[f"{args.attn_name}.pe_cos0"]
    pe_sin = tensors[f"{args.attn_name}.pe_sin0"]

    tlim = min(args.token_limit, q_norm.shape[1])
    q_norm = q_norm[:, :tlim, :].contiguous()
    k_norm = k_norm[:, :tlim, :].contiguous()
    v = v[:, :tlim, :].contiguous()
    gate_logits = gate_logits[:, :tlim, :].contiguous()
    pre_to_out_expected = pre_to_out_expected[:, :tlim, :].contiguous()
    pe_cos = pe_cos[:, :, :tlim, :].contiguous()
    pe_sin = pe_sin[:, :, :tlim, :].contiguous()

    qh = apply_head_split_bthd(q_norm, args.num_heads)
    kh = apply_head_split_bthd(k_norm, args.num_heads)
    vh = apply_head_split_bthd(v, args.num_heads)

    q_rot = apply_rope_split_bthd(qh, pe_cos, pe_sin)
    k_rot = apply_rope_split_bthd(kh, pe_cos, pe_sin)

    print("Comparing pre_to_out variants against fixture to_out_input_diag0")
    run_variant("sdpa_sigmoid", q_rot, k_rot, vh, gate_logits, pre_to_out_expected, args.abs_tol, args.rel_tol, "sigmoid", True, False)
    run_variant("sdpa_sigmoid_x2", q_rot, k_rot, vh, gate_logits, pre_to_out_expected, args.abs_tol, args.rel_tol, "sigmoid_x2", True, False)
    run_variant("sdpa_no_gate", q_rot, k_rot, vh, gate_logits, pre_to_out_expected, args.abs_tol, args.rel_tol, "none", True, False)
    run_variant("manual_sigmoid", q_rot, k_rot, vh, gate_logits, pre_to_out_expected, args.abs_tol, args.rel_tol, "sigmoid", False, False)
    run_variant("manual_sigmoid_x2", q_rot, k_rot, vh, gate_logits, pre_to_out_expected, args.abs_tol, args.rel_tol, "sigmoid_x2", False, False)
    run_variant("manual_no_gate", q_rot, k_rot, vh, gate_logits, pre_to_out_expected, args.abs_tol, args.rel_tol, "none", False, False)


if __name__ == "__main__":
    main()
