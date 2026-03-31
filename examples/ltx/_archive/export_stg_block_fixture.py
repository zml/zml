"""Export STG (V-passthrough) block fixture for parity checking.

Self-contained: loads a single block from the base checkpoint, generates
synthetic block inputs with correct shapes, runs the block normally AND with
V-passthrough monkeypatch, and saves both outputs for the Zig STG checker.

The STG perturbation makes self-attention (attn1 / audio_attn1) skip Q·K·V and
compute to_out(to_v(x)) instead, with per-head gating still applied.

Saved keys:
  block0_native.{vx_in, ax_in, video_timesteps, audio_timesteps, ...}  (inputs)
  block0_native.{vx_out, ax_out}                                       (normal outputs)
  stg_block.{vx_out, ax_out}                                           (STG outputs)

Usage:
    uv run python export_stg_block_fixture.py \\
        --checkpoint /path/to/ltx-2.3-22b-dev.safetensors \\
        --output     /path/to/stg_block_fixture.safetensors \\
        [--block-index 0] [--video-tokens 64] [--audio-tokens 32] [--text-tokens 256]
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.attention import AttentionFunction


# ── LTX 2.3 block config (same for base and distilled) ──────────────────────
_VIDEO_CFG = TransformerConfig(
    dim=4096,
    heads=32,
    d_head=128,
    context_dim=4096,
    apply_gated_attention=True,
    cross_attention_adaln=True,
)
_AUDIO_CFG = TransformerConfig(
    dim=2048,
    heads=32,
    d_head=64,
    context_dim=2048,
    apply_gated_attention=True,
    cross_attention_adaln=True,
)
_NORM_EPS = 1e-6
_ROPE_TYPE = LTXRopeType.SPLIT


def parse_args():
    p = argparse.ArgumentParser(description="Export STG block fixture (self-contained)")
    p.add_argument("--checkpoint", type=Path, required=True,
                    help="Model checkpoint .safetensors (base, not distilled)")
    p.add_argument("--output", type=Path, required=True,
                    help="Output STG fixture .safetensors")
    p.add_argument("--block-index", type=int, default=0,
                    help="Block index to use (0-indexed, default=0)")
    p.add_argument("--video-tokens", type=int, default=64,
                    help="Number of video tokens (default=64)")
    p.add_argument("--audio-tokens", type=int, default=32,
                    help="Number of audio tokens (default=32)")
    p.add_argument("--text-tokens", type=int, default=256,
                    help="Number of text tokens (default=256)")
    return p.parse_args()


def build_and_load_block(checkpoint: Path, block_idx: int, device, dtype) -> BasicAVTransformerBlock:
    """Build one block and load its weights from the full checkpoint."""
    block = BasicAVTransformerBlock(
        idx=block_idx,
        video=_VIDEO_CFG,
        audio=_AUDIO_CFG,
        rope_type=_ROPE_TYPE,
        norm_eps=_NORM_EPS,
        attention_function=AttentionFunction.DEFAULT,
    )

    print(f"Loading weights for block {block_idx} from: {checkpoint}")
    full_sd = load_file(str(checkpoint), device="cpu")

    # Show sample keys to identify the prefix
    all_keys = sorted(full_sd.keys())
    print(f"  Checkpoint has {len(all_keys)} keys total")
    # Find keys containing "attn1" — uniquely identifies transformer blocks
    attn1_keys = [k for k in all_keys if "attn1" in k][:5]
    print(f"  Sample attn1 keys: {attn1_keys}")

    # Auto-detect prefix: find the part before "transformer_blocks.{idx}." or
    # before "attn1" in the first matching key
    block_sd = {}
    used_prefix = None

    # Strategy 1: look for "transformer_blocks.{idx}." with any prefix
    import re
    pattern = re.compile(rf"^(.*transformer_blocks\.{block_idx}\.)")
    for k in all_keys:
        m = pattern.match(k)
        if m:
            prefix = m.group(1)
            block_sd = {k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)}
            if block_sd:
                used_prefix = prefix
                break

    print(f"  Prefix: {used_prefix or 'NONE MATCHED'}")
    print(f"  Found {len(block_sd)} weight tensors for block {block_idx}")
    if not block_sd:
        raise RuntimeError(
            f"No weights found for block {block_idx}. "
            f"Sample attn1 keys: {attn1_keys}"
        )

    missing, unexpected = block.load_state_dict(block_sd, strict=False)
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected[:5]}")
    if missing:
        print(f"  WARNING: missing keys: {missing[:5]}")

    block.to(device=device, dtype=dtype)
    block.eval()
    return block


def generate_synthetic_inputs(B, T_v, T_a, T_text, device, dtype):
    """Generate synthetic block inputs with correct shapes and realistic magnitudes.

    Uses small random values (std=0.02) to mimic typical hidden state magnitudes
    after layer norm. RoPE values use cos/sin of random phases.
    """
    g = torch.Generator(device="cpu").manual_seed(42)

    def randn(*shape):
        return (torch.randn(*shape, generator=g) * 0.02).to(device=device, dtype=dtype)

    def rope_pair(heads, tokens, d_head_half):
        """Generate (cos, sin) RoPE pair: [1, heads, tokens, d_head/2]."""
        phase = torch.randn(1, heads, tokens, d_head_half, generator=g)
        return (phase.cos().to(device=device, dtype=dtype),
                phase.sin().to(device=device, dtype=dtype))

    V_DIM, A_DIM = 4096, 2048
    V_HEADS, A_HEADS = 32, 32
    V_DHEAD, A_DHEAD = 128, 64

    # AdaLN param counts per scale_shift_table:
    #   scale_shift_table:          9 params (3 MSA + 3 FF + 3 text_CA)
    #   prompt_scale_shift_table:   2 params (KV shift, scale for text CA)
    #   scale_shift_table_a2v_ca_*: 5 params (4 scale/shift + 1 gate)
    #   gate adaln:                 1 param  (single gate value)
    V_ADA, A_ADA = 9, 9
    V_PROMPT_ADA, A_PROMPT_ADA = 2, 2
    V_CROSS_SS_ADA, A_CROSS_SS_ADA = 4, 4

    inputs = {}
    inputs["vx_in"] = randn(B, T_v, V_DIM)
    inputs["ax_in"] = randn(B, T_a, A_DIM)
    # Timestep embeddings: [B, 1, num_ada_params * dim]
    inputs["video_timesteps"] = randn(B, 1, V_ADA * V_DIM)       # [1, 1, 36864]
    inputs["audio_timesteps"] = randn(B, 1, A_ADA * A_DIM)       # [1, 1, 18432]
    inputs["v_prompt_timestep"] = randn(B, 1, V_PROMPT_ADA * V_DIM)  # [1, 1, 8192]
    inputs["a_prompt_timestep"] = randn(B, 1, A_PROMPT_ADA * A_DIM)  # [1, 1, 4096]
    inputs["v_text_ctx"] = randn(B, T_text, V_DIM)
    inputs["a_text_ctx"] = randn(B, T_text, A_DIM)
    inputs["v_cross_ss_ts"] = randn(B, 1, V_CROSS_SS_ADA * V_DIM)  # [1, 1, 16384]
    inputs["v_cross_gate_ts"] = randn(B, 1, V_DIM)                  # [1, 1, 4096]
    inputs["a_cross_ss_ts"] = randn(B, 1, A_CROSS_SS_ADA * A_DIM)  # [1, 1, 8192]
    inputs["a_cross_gate_ts"] = randn(B, 1, A_DIM)                  # [1, 1, 2048]

    # Self-attention RoPE: [1, heads, tokens, d_head/2]
    inputs["v_pe_cos"], inputs["v_pe_sin"] = rope_pair(V_HEADS, T_v, V_DHEAD // 2)
    inputs["a_pe_cos"], inputs["a_pe_sin"] = rope_pair(A_HEADS, T_a, A_DHEAD // 2)

    # Cross-attention RoPE: BOTH A2V and V2A use inner_dim=2048, hd=64, half_hd=32.
    # In "split" mode, only the first 64 dims of a 128-dim head get rotated.
    CROSS_ROPE_HALF_HD = 32  # 2048 / 32 heads / 2

    # A2V cross-attn RoPE: q uses video tokens, k uses audio tokens
    inputs["a2v_pe_cos"], inputs["a2v_pe_sin"] = rope_pair(V_HEADS, T_v, CROSS_ROPE_HALF_HD)
    inputs["a2v_k_pe_cos"], inputs["a2v_k_pe_sin"] = rope_pair(V_HEADS, T_a, CROSS_ROPE_HALF_HD)

    # V2A cross-attn RoPE: q uses audio tokens, k uses video tokens
    inputs["v2a_pe_cos"], inputs["v2a_pe_sin"] = rope_pair(A_HEADS, T_a, CROSS_ROPE_HALF_HD)
    inputs["v2a_k_pe_cos"], inputs["v2a_k_pe_sin"] = rope_pair(A_HEADS, T_v, CROSS_ROPE_HALF_HD)

    return inputs


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    B = 1

    # ── Build block and load weights ───────────────────────────────────────
    block = build_and_load_block(args.checkpoint, args.block_index, device, dtype)

    # ── Generate synthetic inputs ──────────────────────────────────────────
    print(f"\nGenerating synthetic inputs: B={B}, T_v={args.video_tokens}, "
          f"T_a={args.audio_tokens}, T_text={args.text_tokens}")
    inputs = generate_synthetic_inputs(
        B, args.video_tokens, args.audio_tokens, args.text_tokens, device, dtype
    )

    def _pe(cos_key, sin_key):
        return (inputs[cos_key], inputs[sin_key])

    # ── Build TransformerArgs ──────────────────────────────────────────────
    video_args = TransformerArgs(
        x=inputs["vx_in"].clone(),
        context=inputs["v_text_ctx"],
        context_mask=None,
        timesteps=inputs["video_timesteps"],
        embedded_timestep=inputs["video_timesteps"],
        positional_embeddings=_pe("v_pe_cos", "v_pe_sin"),
        cross_positional_embeddings=_pe("a2v_pe_cos", "a2v_pe_sin"),
        cross_scale_shift_timestep=inputs["v_cross_ss_ts"],
        cross_gate_timestep=inputs["v_cross_gate_ts"],
        enabled=True,
        prompt_timestep=inputs["v_prompt_timestep"],
        self_attention_mask=None,
    )
    audio_args = TransformerArgs(
        x=inputs["ax_in"].clone(),
        context=inputs["a_text_ctx"],
        context_mask=None,
        timesteps=inputs["audio_timesteps"],
        embedded_timestep=inputs["audio_timesteps"],
        positional_embeddings=_pe("a_pe_cos", "a_pe_sin"),
        cross_positional_embeddings=_pe("a2v_k_pe_cos", "a2v_k_pe_sin"),
        cross_scale_shift_timestep=inputs["a_cross_ss_ts"],
        cross_gate_timestep=inputs["a_cross_gate_ts"],
        enabled=True,
        prompt_timestep=inputs["a_prompt_timestep"],
        self_attention_mask=None,
    )

    # ── Run 1: Normal (no perturbation) ────────────────────────────────────
    print("\nRunning normal forward (no perturbation)...")
    with torch.inference_mode():
        video_out_normal, audio_out_normal = block(
            video=video_args,
            audio=audio_args,
            perturbations=None,
        )
    vx_out_normal = video_out_normal.x.detach().cpu().contiguous()
    ax_out_normal = audio_out_normal.x.detach().cpu().contiguous()
    print(f"  Normal video out: shape={list(vx_out_normal.shape)} dtype={vx_out_normal.dtype}")
    print(f"  Normal audio out: shape={list(ax_out_normal.shape)} dtype={ax_out_normal.dtype}")

    # ── Run 2: STG (V-passthrough via manual monkeypatch) ──────────────────
    # Monkeypatch self-attention (attn1 / audio_attn1) to do V-only passthrough.
    # This is the ground-truth V-passthrough: to_v → heads → gate → to_out.
    from einops import rearrange

    def make_v_passthrough(attn_mod):
        """Create a V-passthrough forward function for the given attention module."""
        def v_passthrough_forward(x, *args, **kwargs):
            v = attn_mod.to_v(x)
            heads = attn_mod.heads
            v = rearrange(v, 'b t (h d) -> b h t d', h=heads)

            # Per-head gating (same as normal path)
            gate = attn_mod.to_gate_logits(x).sigmoid() * 2.0
            gate = rearrange(gate, 'b t h -> b h t 1')
            v = v * gate

            v = rearrange(v, 'b h t d -> b t (h d)')
            return attn_mod.to_out(v)
        return v_passthrough_forward

    # Rebuild fresh args (x may have been modified in-place by the block)
    video_args_stg = TransformerArgs(
        x=inputs["vx_in"].clone(),
        context=inputs["v_text_ctx"],
        context_mask=None,
        timesteps=inputs["video_timesteps"],
        embedded_timestep=inputs["video_timesteps"],
        positional_embeddings=_pe("v_pe_cos", "v_pe_sin"),
        cross_positional_embeddings=_pe("a2v_pe_cos", "a2v_pe_sin"),
        cross_scale_shift_timestep=inputs["v_cross_ss_ts"],
        cross_gate_timestep=inputs["v_cross_gate_ts"],
        enabled=True,
        prompt_timestep=inputs["v_prompt_timestep"],
        self_attention_mask=None,
    )
    audio_args_stg = TransformerArgs(
        x=inputs["ax_in"].clone(),
        context=inputs["a_text_ctx"],
        context_mask=None,
        timesteps=inputs["audio_timesteps"],
        embedded_timestep=inputs["audio_timesteps"],
        positional_embeddings=_pe("a_pe_cos", "a_pe_sin"),
        cross_positional_embeddings=_pe("a2v_k_pe_cos", "a2v_k_pe_sin"),
        cross_scale_shift_timestep=inputs["a_cross_ss_ts"],
        cross_gate_timestep=inputs["a_cross_gate_ts"],
        enabled=True,
        prompt_timestep=inputs["a_prompt_timestep"],
        self_attention_mask=None,
    )

    # Monkeypatch both self-attention modules
    orig_attn1_forward = block.attn1.forward
    orig_audio_attn1_forward = block.audio_attn1.forward
    block.attn1.forward = make_v_passthrough(block.attn1)
    block.audio_attn1.forward = make_v_passthrough(block.audio_attn1)

    print(f"\nRunning STG forward (V-passthrough on attn1 + audio_attn1)...")
    with torch.inference_mode():
        video_out_stg, audio_out_stg = block(
            video=video_args_stg,
            audio=audio_args_stg,
            perturbations=None,
        )

    # Restore originals
    block.attn1.forward = orig_attn1_forward
    block.audio_attn1.forward = orig_audio_attn1_forward

    vx_out_stg = video_out_stg.x.detach().cpu().contiguous()
    ax_out_stg = audio_out_stg.x.detach().cpu().contiguous()

    # Verify STG differs from normal
    cos_v = torch.nn.functional.cosine_similarity(
        vx_out_stg.float().flatten(), vx_out_normal.float().flatten(), dim=0
    ).item()
    cos_a = torch.nn.functional.cosine_similarity(
        ax_out_stg.float().flatten(), ax_out_normal.float().flatten(), dim=0
    ).item()
    v_diff = (vx_out_stg.float() - vx_out_normal.float()).abs()
    a_diff = (ax_out_stg.float() - ax_out_normal.float()).abs()
    print(f"  STG vs Normal — Video: cos_sim={cos_v:.6f} max_diff={v_diff.max():.6f} mean_diff={v_diff.mean():.6f}")
    print(f"  STG vs Normal — Audio: cos_sim={cos_a:.6f} max_diff={a_diff.max():.6f} mean_diff={a_diff.mean():.6f}")

    if cos_v > 0.9999 and cos_a > 0.9999:
        print("  ERROR: STG outputs nearly identical to normal — monkeypatch may not be working!")
        raise RuntimeError("V-passthrough did not change outputs — check monkeypatch logic")

    # ── Save fixture ───────────────────────────────────────────────────────
    tensors_out: dict[str, torch.Tensor] = {}

    # Save all inputs with block0_native.* prefix
    for key, val in inputs.items():
        tensors_out[f"block0_native.{key}"] = val.detach().cpu().contiguous()

    # Normal and STG outputs
    tensors_out["block0_native.vx_out"] = vx_out_normal
    tensors_out["block0_native.ax_out"] = ax_out_normal
    tensors_out["stg_block.vx_out"] = vx_out_stg
    tensors_out["stg_block.ax_out"] = ax_out_stg

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors_out, str(args.output))

    print(f"\nSaved STG fixture: {args.output}")
    max_key = max(len(k) for k in tensors_out)
    for k in sorted(tensors_out.keys()):
        v = tensors_out[k]
        print(f"  {k:<{max_key + 2}} shape={list(v.shape)} dtype={v.dtype}")


if __name__ == "__main__":
    main()
