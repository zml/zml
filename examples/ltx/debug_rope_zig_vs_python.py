"""Targeted diagnostic: replicate the Zig RoPE algorithm in Python and compare.

Identifies exactly where video PE diverges from Python reference.

Run from the LTX-2 repo:
  cd /root/repos/LTX-2 && uv run scripts/debug_rope_zig_vs_python.py
"""

import math
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file


def zig_generate_freq_grid(theta: float, n_pos_dims: int, dim: int) -> torch.Tensor:
    """Exact replica of Zig generateFreqGrid."""
    log_theta = math.log(theta)
    n_elem = 2 * n_pos_dims
    num_freqs = dim // n_elem  # floor division

    idx = torch.arange(num_freqs, dtype=torch.float32)
    step = 1.0 / (num_freqs - 1) if num_freqs > 1 else 0.0
    indices = torch.exp(idx * step * log_theta) * (math.pi / 2.0)
    return indices


def zig_get_fractional_positions(positions: torch.Tensor, max_pos: list[int]) -> list[torch.Tensor]:
    """Exact replica of Zig getFractionalPositions. positions: [B, C, T, 2]"""
    n_pos_dims = positions.shape[1]
    start = positions[:, :, :, 0].float()  # [B, C, T]
    end = positions[:, :, :, 1].float()    # [B, C, T]
    middle = (start + end) * 0.5           # [B, C, T]

    result = []
    for c in range(n_pos_dims):
        pos_c = middle[:, c, :]  # [B, T]
        result.append(pos_c / max_pos[c])
    return result


def zig_generate_freqs(freq_basis: torch.Tensor, frac_positions: list[torch.Tensor]) -> torch.Tensor:
    """Exact replica of Zig generateFreqs."""
    parts = []
    for c, frac in enumerate(frac_positions):
        scaled_frac = frac * 2.0 - 1.0  # map [0,1] → [-1,1]
        # outer product: [B, T] x [num_freqs] → [B, T, num_freqs]
        part = scaled_frac.unsqueeze(-1) * freq_basis.unsqueeze(0).unsqueeze(0)
        parts.append(part)

    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=-1)


def zig_split_freqs_cis(freqs: torch.Tensor, num_heads: int, head_dim: int):
    """Exact replica of Zig splitFreqsCis."""
    half_hd = head_dim // 2
    total_needed = num_heads * half_hd
    freq_dim = freqs.shape[-1]

    cos_freq = freqs.cos()
    sin_freq = freqs.sin()

    if freq_dim < total_needed:
        pad_size = total_needed - freq_dim
        cos_pad = torch.ones(*freqs.shape[:-1], pad_size)
        sin_pad = torch.zeros(*freqs.shape[:-1], pad_size)
        cos_freq = torch.cat([cos_freq, cos_pad], dim=-1)
        sin_freq = torch.cat([sin_freq, sin_pad], dim=-1)

    # Reshape [B, T, H*HD/2] → [B, T, H, HD/2] → transpose to [B, H, T, HD/2]
    B, T, _ = cos_freq.shape
    cos_r = cos_freq.reshape(B, T, num_heads, half_hd).permute(0, 2, 1, 3)
    sin_r = sin_freq.reshape(B, T, num_heads, half_hd).permute(0, 2, 1, 3)
    return cos_r, sin_r


def zig_precompute_freqs_cis(positions: torch.Tensor, theta: float, max_pos: list[int],
                              num_heads: int, inner_dim: int):
    """Exact replica of Zig precomputeFreqsCis pipeline."""
    n_pos_dims = positions.shape[1]
    head_dim = inner_dim // num_heads

    freq_basis = zig_generate_freq_grid(theta, n_pos_dims, inner_dim)
    frac_positions = zig_get_fractional_positions(positions, max_pos)
    freqs = zig_generate_freqs(freq_basis, frac_positions)
    return zig_split_freqs_cis(freqs, num_heads, head_dim)


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return (torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm() + 1e-10)).item()


def main():
    fixture_path = Path("trace_run/step2_fixture_step_000_t512.safetensors")
    if not fixture_path.exists():
        print(f"Fixture not found: {fixture_path}")
        return

    fix = load_file(str(fixture_path))

    v_positions = fix["raw.video_positions"]  # [1, 3, 512, 2] bf16
    a_positions = fix["raw.audio_positions"]  # [1, 1, 126, 2] f32

    print("=== Position tensor info ===")
    print(f"video_positions: shape={list(v_positions.shape)} dtype={v_positions.dtype}")
    print(f"audio_positions: shape={list(a_positions.shape)} dtype={a_positions.dtype}")

    # Position value analysis
    for c in range(v_positions.shape[1]):
        s = v_positions[0, c, :, 0].float()
        e = v_positions[0, c, :, 1].float()
        m = (s + e) / 2.0
        print(f"  video dim {c}: start=[{s.min():.1f}, {s.max():.1f}]  "
              f"end=[{e.min():.1f}, {e.max():.1f}]  "
              f"mid=[{m.min():.3f}, {m.max():.3f}]  "
              f"unique_mids={m.unique().numel()}")

    for c in range(a_positions.shape[1]):
        s = a_positions[0, c, :, 0].float()
        e = a_positions[0, c, :, 1].float()
        m = (s + e) / 2.0
        print(f"  audio dim {c}: start=[{s.min():.1f}, {s.max():.1f}]  "
              f"end=[{e.min():.1f}, {e.max():.1f}]  "
              f"mid=[{m.min():.3f}, {m.max():.3f}]  "
              f"unique_mids={m.unique().numel()}")

    # Reference PE from fixture
    ref_v_cos = fix["intermediate.v_pe_cos"].float()  # [1, 32, 512, 64]
    ref_v_sin = fix["intermediate.v_pe_sin"].float()
    ref_a_cos = fix["intermediate.a_pe_cos"].float()  # [1, 32, 126, 32]
    ref_a_sin = fix["intermediate.a_pe_sin"].float()
    print(f"\nref_v_pe_cos: shape={list(ref_v_cos.shape)} dtype={ref_v_cos.dtype}")
    print(f"ref_a_pe_cos: shape={list(ref_a_cos.shape)} dtype={ref_a_cos.dtype}")

    # ---- 1. Test audio PE (should match) ----
    print("\n=== Audio PE (Zig algorithm, should match) ===")
    a_pe_cos, a_pe_sin = zig_precompute_freqs_cis(
        a_positions, theta=10000.0, max_pos=[20],
        num_heads=32, inner_dim=2048,
    )
    print(f"  cos_sim(a_pe_cos): {cos_sim(a_pe_cos, ref_a_cos):.6f}")
    print(f"  cos_sim(a_pe_sin): {cos_sim(a_pe_sin, ref_a_sin):.6f}")

    # ---- 2. Test video PE with current Zig config ----
    print("\n=== Video PE (Zig algorithm, max_pos=[20, 2048, 2048]) ===")
    v_pe_cos, v_pe_sin = zig_precompute_freqs_cis(
        v_positions, theta=10000.0, max_pos=[20, 2048, 2048],
        num_heads=32, inner_dim=4096,
    )
    print(f"  cos_sim(v_pe_cos): {cos_sim(v_pe_cos, ref_v_cos):.6f}")
    print(f"  cos_sim(v_pe_sin): {cos_sim(v_pe_sin, ref_v_sin):.6f}")

    # ---- 3. Try different max_pos values ----
    print("\n=== Video PE: trying different max_pos ===")
    for mp in [
        [20, 2048, 2048],
        [128, 128, 128],
        [20, 80, 80],
        [20, 128, 128],
        [20, 64, 64],
        [20, 32, 32],
        [20, 16, 16],
        [20, 256, 256],
        [20, 512, 512],
        [128, 2048, 2048],
        [256, 256, 256],
        [32, 32, 32],
        [64, 64, 64],
    ]:
        c, s = zig_precompute_freqs_cis(
            v_positions, theta=10000.0, max_pos=mp,
            num_heads=32, inner_dim=4096,
        )
        print(f"  max_pos={mp}:  cos_sim={cos_sim(c, ref_v_cos):.6f}")

    # ---- 4. Try with Python preprocessor directly ----
    print("\n=== Video PE from Python preprocessor (ground truth) ===")
    try:
        from ltx_core.types import LatentState
        from ltx_pipelines.utils.helpers import modality_from_latent_state

        sigma = fix["raw.sigma"].to(dtype=torch.float32).cuda()

        v_state = LatentState(
            latent=fix["raw.video_latent"].to(dtype=torch.bfloat16).cuda(),
            denoise_mask=fix["raw.video_denoise_mask"].cuda(),
            positions=fix["raw.video_positions"].cuda(),
            clean_latent=fix["raw.video_clean_latent"].to(dtype=torch.bfloat16).cuda(),
        )
        a_state = LatentState(
            latent=fix["raw.audio_latent"].to(dtype=torch.bfloat16).cuda(),
            denoise_mask=fix["raw.audio_denoise_mask"].cuda(),
            positions=fix["raw.audio_positions"].cuda(),
            clean_latent=fix["raw.audio_clean_latent"].to(dtype=torch.bfloat16).cuda(),
        )
        v_ctx = fix["raw.v_context"].to(dtype=torch.bfloat16).cuda()

        pos_video = modality_from_latent_state(v_state, v_ctx, sigma)

        # Check if positions changed
        if torch.equal(fix["raw.video_positions"].cuda(), pos_video.positions):
            print("  Positions unchanged by modality_from_latent_state")
        else:
            maxd = (fix["raw.video_positions"].cuda().float() - pos_video.positions.float()).abs().max()
            print(f"  Positions CHANGED by modality_from_latent_state! max_diff={maxd}")
            print(f"  New positions shape: {list(pos_video.positions.shape)} dtype={pos_video.positions.dtype}")
            # Show position value ranges
            p = pos_video.positions
            for c_idx in range(p.shape[1]):
                s = p[0, c_idx, :, 0].float()
                e = p[0, c_idx, :, 1].float()
                m_val = (s + e) / 2.0
                print(f"    new dim {c_idx}: mid range=[{m_val.min():.3f}, {m_val.max():.3f}]  unique={m_val.unique().numel()}")

        # Now run the actual preprocessor
        checkpoint_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser())
        spatial_upsampler_path = str(Path("~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors").expanduser())
        gemma_root = str(Path("~/models/gemma-3-12b-it").expanduser())

        from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

        print("\n  Loading pipeline to inspect RoPE internals...")
        pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=[],
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=gemma_root,
            loras=[],
            quantization=None,
        )
        vm = pipeline.stage_2_model_ledger.transformer().velocity_model
        vap = vm.video_args_preprocessor

        # Look for the RoPE config
        print(f"\n  video_args_preprocessor type: {type(vap).__name__}")
        print(f"  module: {type(vap).__module__}")

        # Inspect source for RoPE parameters
        import inspect
        src = inspect.getsource(type(vap))
        # Find MAX_POS, theta, rope references
        for line_no, line in enumerate(src.split('\n'), 1):
            l = line.strip()
            if any(kw in l.lower() for kw in ['max_pos', 'rope_max', 'theta',
                                                'precompute_freq', 'inner_dim',
                                                'n_pos_dim', 'positional']):
                if not l.startswith('#'):
                    print(f"    L{line_no}: {l}")

        # Try to access rope config attributes
        for attr in sorted(dir(vap)):
            if attr.startswith('_'):
                continue
            try:
                val = getattr(vap, attr)
                if callable(val) and not isinstance(val, torch.nn.Module):
                    continue
                if isinstance(val, torch.Tensor) and val.numel() > 100:
                    print(f"    attr {attr}: Tensor shape={list(val.shape)}")
                elif isinstance(val, (int, float, str, bool, list, tuple)):
                    print(f"    attr {attr}: {val}")
                elif isinstance(val, torch.nn.Module):
                    print(f"    attr {attr}: {type(val).__name__}")
            except Exception:
                pass

        # Run prepare and inspect the PE
        a_ctx = fix["raw.a_context"].to(dtype=torch.bfloat16).cuda()
        pos_audio = modality_from_latent_state(a_state, a_ctx, sigma)

        with torch.inference_mode():
            video_args = vap.prepare(pos_video, pos_audio)

        py_v_cos, py_v_sin = video_args.positional_embeddings
        print(f"\n  Python v_pe_cos: shape={list(py_v_cos.shape)} dtype={py_v_cos.dtype}")
        print(f"  cos_sim(python_pe vs fixture): {cos_sim(py_v_cos.cpu(), ref_v_cos):.6f}")
        print(f"  cos_sim(zig_pe vs fixture):    {cos_sim(v_pe_cos, ref_v_cos):.6f}")

        # Extract the actual RoPE computation from the preprocessor
        # Look for precompute_freqs_cis in the source
        prepare_src = inspect.getsource(type(vap).prepare)
        print(f"\n  prepare() source ({len(prepare_src)} chars):")
        for i, line in enumerate(prepare_src.split('\n'), 1):
            print(f"    {i:3d}: {line}")

    except Exception as e:
        print(f"  Error loading pipeline: {e}")
        import traceback
        traceback.print_exc()

    # ---- 5. Cross-gate timestep analysis ----
    print("\n=== Cross-gate timestep analysis ===")
    ref_v_cross_gate = fix.get("intermediate.v_cross_gate_ts")
    if ref_v_cross_gate is not None:
        print(f"  ref v_cross_gate_ts: shape={list(ref_v_cross_gate.shape)} dtype={ref_v_cross_gate.dtype}")
        print(f"  ref values[:10]: {ref_v_cross_gate.flatten()[:10].float().tolist()}")
        print(f"  ref value range: [{ref_v_cross_gate.float().min():.6f}, {ref_v_cross_gate.float().max():.6f}]")

    ref_v_cross_ss = fix.get("intermediate.v_cross_ss_ts")
    if ref_v_cross_ss is not None:
        print(f"  ref v_cross_ss_ts: shape={list(ref_v_cross_ss.shape)} dtype={ref_v_cross_ss.dtype}")


if __name__ == "__main__":
    main()
