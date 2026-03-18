"""Diagnostic script to validate RoPE application between LTX and ZML.

This script loads intermediate tensors (q, k, v pre-rope) and rotary parameters
(pe_cos, pe_sin) from a fixture, applies RoPE manually in PyTorch, and outputs
the post-rope values for comparison with ZML Zig implementation.

Usage:
    python diagnostic_rope_validation.py \
        /path/to/fixture.safetensors \
        /output/diagnostics.safetensors \
        --attn-name attn1 \
        --token-limit 256
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def apply_head_split(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Split [B, T, D] -> [B, H, T, HD] for attention.
    
    Args:
        x: [B, T, D] tensor
        num_heads: number of attention heads
    
    Returns:
        [B, H, T, HD] tensor
    """
    b, t, d = x.shape
    hd = d // num_heads
    assert d == num_heads * hd, f"D={d} not divisible by num_heads={num_heads}"
    
    # [B, T, D] -> [B, T, H, HD] -> [B, H, T, HD]
    x_split = x.reshape(b, t, num_heads, hd)
    return x_split.permute(0, 2, 1, 3)


def apply_rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply interleaved RoPE: cos/sin have same dim as x's head dimension.
    
    Interleaved layout: [x0, x1, x2, x3, ...] with cos/sin of same size.
    Formula: out = x * cos + rotate(x) * sin
    where rotate(x) for interleaved = [-x1, x0, -x3, x2, ...]
    
    Args:
        x: [B, H, T, HD] tensor
        cos: [1, H, T, HD] or [T, HD] tensor (after broadcast)
        sin: [1, H, T, HD] or [T, HD] tensor
    
    Returns:
        [B, H, T, HD] tensor with rope applied
    """
    # Ensure cos/sin are broadcastable
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(0)
    while sin.ndim < x.ndim:
        sin = sin.unsqueeze(0)
    
    # Interleaved rotate: [-x1, x0, -x3, x2, ...]
    x_rotated = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
    
    cos_b = cos.expand_as(x)
    sin_b = sin.expand_as(x)
    
    return x * cos_b + x_rotated * sin_b


def apply_rope_split(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply split RoPE: cos/sin have half the dim of x's head dimension.
    
    Split layout: [a, b, c, d, ...] with cos/sin of size HD/2, e.g. [cos_ab, cos_cd, ...]
    Formula (for each pair):
        out_first = x_first * cos - x_second * sin
        out_second = x_second * cos + x_first * sin
    
    Args:
        x: [B, H, T, HD] tensor
        cos: [1, H, T, HD/2] or [T, HD/2] tensor
        sin: [1, H, T, HD/2] or [T, HD/2] tensor
    
    Returns:
        [B, H, T, HD] tensor with rope applied
    """
    # Ensure cos/sin are broadcastable
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(0)
    while sin.ndim < x.ndim:
        sin = sin.unsqueeze(0)
    
    hd = x.shape[-1]
    half = hd // 2
    
    x_first = x[..., :half]
    x_second = x[..., half:]
    
    cos_b = cos.expand(x_first.shape)
    sin_b = sin.expand(x_first.shape)
    
    out_first = x_first * cos_b - x_second * sin_b
    out_second = x_second * cos_b + x_first * sin_b
    
    return torch.cat([out_first, out_second], dim=-1)


def detect_rope_layout(hd_qkv: int, hd_rope: int) -> str:
    """Detect RoPE layout (interleaved vs split) by comparing dimensions.
    
    Args:
        hd_qkv: head dimension of q/k/v tensor
        hd_rope: head dimension of cos/sin tensor
    
    Returns:
        "interleaved" if hd_rope == hd_qkv, "split" if hd_rope * 2 == hd_qkv
    """
    if hd_rope == hd_qkv:
        return "interleaved"
    elif hd_rope * 2 == hd_qkv:
        return "split"
    else:
        raise ValueError(
            f"Unsupported RoPE shape: qkv_hd={hd_qkv} rope_hd={hd_rope}. "
            f"Expected rope_hd == qkv_hd (interleaved) or rope_hd * 2 == qkv_hd (split)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate RoPE application between LTX Python and ZML Zig"
    )
    parser.add_argument("fixture_path", type=Path, help="Path to fixture.safetensors with intermediates")
    parser.add_argument(
        "output_path", type=Path, help="Path to save diagnostic output safetensors"
    )
    parser.add_argument(
        "--attn-name", default="attn1", help="Attention component name (attn1, attn2, ...)"
    )
    parser.add_argument(
        "--num-heads", type=int, default=32, help="Number of attention heads"
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=None,
        help="Limit tokens for comparison (default: full sequence)",
    )

    args = parser.parse_args()

    print(f"Loading fixture from {args.fixture_path}...")
    tensors = load_file(str(args.fixture_path))

    # Extract diagnostic intermediates.
    # Prefer q_norm_diag0 / k_norm_diag0 (post-norm, pre-head-split) when available,
    # because that is what actually enters the head-split and RoPE in LTX.
    # Fall back to to_q_diag0 / to_k_diag0 (raw projection, pre-norm) otherwise.
    q_key = f"{args.attn_name}.q_norm_diag0"
    k_key = f"{args.attn_name}.k_norm_diag0"
    q_source = "q_norm (post-norm)"
    k_source = "k_norm (post-norm)"
    if q_key not in tensors:
        q_key = f"{args.attn_name}.to_q_diag0"
        q_source = "to_q (pre-norm)"
    if k_key not in tensors:
        k_key = f"{args.attn_name}.to_k_diag0"
        k_source = "to_k (pre-norm)"

    v_key = f"{args.attn_name}.to_v_diag0"
    pe_cos_key = f"{args.attn_name}.pe_cos0"
    pe_sin_key = f"{args.attn_name}.pe_sin0"

    if q_key not in tensors or k_key not in tensors or v_key not in tensors:
        print(f"ERROR: Missing intermediate diagnostics. Expected {q_key}, {k_key}, {v_key}")
        print(f"Available keys: {sorted(tensors.keys())}")
        return

    if pe_cos_key not in tensors or pe_sin_key not in tensors:
        print(f"ERROR: Missing RoPE parameters. Expected {pe_cos_key}, {pe_sin_key}")
        print(f"Available keys: {sorted(tensors.keys())}")
        return

    q = tensors[q_key]
    k = tensors[k_key]
    v = tensors[v_key]
    pe_cos = tensors[pe_cos_key]
    pe_sin = tensors[pe_sin_key]

    print(f"q source: {q_source}  shape: {q.shape} dtype: {q.dtype}")
    print(f"k source: {k_source}  shape: {k.shape} dtype: {k.dtype}")
    print(f"v shape: {v.shape} dtype: {v.dtype}")
    print(f"pe_cos shape: {pe_cos.shape} dtype: {pe_cos.dtype}")
    print(f"pe_sin shape: {pe_sin.shape} dtype: {pe_sin.dtype}")

    # Apply head-split: [B, T, D] -> [B, H, T, HD]
    # pe_cos is already head-split [B, H, T, HD_rope] for LTX attn1
    print("Applying head-split to q/k/v...")
    q_head_split = apply_head_split(q, args.num_heads)
    k_head_split = apply_head_split(k, args.num_heads)
    v_head_split = apply_head_split(v, args.num_heads)

    print(f"q_head_split shape: {q_head_split.shape}")
    print(f"k_head_split shape: {k_head_split.shape}")
    print(f"v_head_split shape: {v_head_split.shape}")

    # Token limit (if specified)
    if args.token_limit is not None:
        t = q_head_split.shape[2]  # T dimension
        limit = min(args.token_limit, t)
        print(f"Applying token limit: {limit} (of {t})")
        q_head_split = q_head_split[:, :, :limit, :]
        k_head_split = k_head_split[:, :, :limit, :]
        if pe_cos.ndim == 4:
            pe_cos = pe_cos[:, :, :limit, :]
            pe_sin = pe_sin[:, :, :limit, :]
        else:
            pe_cos = pe_cos[:, :limit, :]
            pe_sin = pe_sin[:, :limit, :]

    # Detect RoPE layout
    hd_qkv = q_head_split.shape[-1]
    hd_rope = pe_cos.shape[-1]
    rope_layout = detect_rope_layout(hd_qkv, hd_rope)
    print(f"Detected RoPE layout: {rope_layout} (qkv_hd={hd_qkv}, rope_hd={hd_rope})")

    # Apply RoPE
    print(f"Applying RoPE ({rope_layout})...")
    if rope_layout == "interleaved":
        q_rotated = apply_rope_interleaved(q_head_split, pe_cos, pe_sin)
        k_rotated = apply_rope_interleaved(k_head_split, pe_cos, pe_sin)
    else:  # split
        q_rotated = apply_rope_split(q_head_split, pe_cos, pe_sin)
        k_rotated = apply_rope_split(k_head_split, pe_cos, pe_sin)

    print(f"q_rotated shape: {q_rotated.shape} dtype: {q_rotated.dtype}")
    print(f"k_rotated shape: {k_rotated.shape} dtype: {k_rotated.dtype}")

    # Save diagnostics in ZML-native [B, T, H, HD] layout.
    # ZML's splitAxis produces [B, T, H, HD] (token first), while PyTorch apply_head_split
    # produces [B, H, T, HD]. Permute to ZML layout so comparison is byte-for-byte correct.
    # Also apply token limit to v_head_split consistently.
    if args.token_limit is not None:
        limit = min(args.token_limit, v_head_split.shape[2])
        v_head_split = v_head_split[:, :, :limit, :]

    def to_zml_layout(t: torch.Tensor) -> torch.Tensor:
        """Permute [B, H, T, HD] -> [B, T, H, HD] to match ZML's splitAxis output."""
        return t.permute(0, 2, 1, 3).contiguous()

    output_tensors = {
        # All saved in [B, T, H, HD] to match ZML's native splitAxis layout
        f"{args.attn_name}.q_head_split": to_zml_layout(q_head_split).cpu(),
        f"{args.attn_name}.k_head_split": to_zml_layout(k_head_split).cpu(),
        f"{args.attn_name}.v_head_split": to_zml_layout(v_head_split).cpu(),
        f"{args.attn_name}.q_rotated": to_zml_layout(q_rotated).cpu(),
        f"{args.attn_name}.k_rotated": to_zml_layout(k_rotated).cpu(),
    }

    metadata = {
        "rope_layout": rope_layout,
        "layout": "BTHD",  # ZML-native [B, T, H, HD] ordering
        "num_heads": str(args.num_heads),
        "token_limit": str(args.token_limit) if args.token_limit else "full",
        "fixture_path": str(args.fixture_path),
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(output_tensors, str(args.output_path), metadata=metadata)

    print(f"\nDiagnostics saved to {args.output_path}")
    print(f"Metadata: {metadata}")
    for key, tensor in output_tensors.items():
        print(f"  {key}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")


if __name__ == "__main__":
    main()
