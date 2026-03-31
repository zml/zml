"""
Diagnose A→V Ada extraction: compare Zig's adaValueAt formula against Python's
get_ada_values for scale_v_a2v and shift_v_a2v, then verify the modulation output.

Usage:
    python diag_ada_extract.py <fixture.safetensors> <checkpoint.safetensors>
"""

import sys
import torch
from safetensors.torch import load_file


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python diag_ada_extract.py <fixture.safetensors> <checkpoint.safetensors>")
        sys.exit(1)

    fix_path = sys.argv[1]
    ckpt_path = sys.argv[2]

    print("Loading fixture …")
    fix = load_file(fix_path)
    print("Loading checkpoint …")
    ckpt = load_file(ckpt_path)

    # ── auto-detect checkpoint key prefix ───────────────────────────────────
    # Merged checkpoints may use "velocity_model.transformer_blocks.N.*" or
    # the reindexed form "transformer_blocks.N.*".
    _sst_suffix = "transformer_blocks.0.scale_shift_table_a2v_ca_video"
    if f"velocity_model.{_sst_suffix}" in ckpt:
        _ckpt_prefix = "velocity_model."
    elif _sst_suffix in ckpt:
        _ckpt_prefix = ""
    else:
        candidates = [k for k in ckpt if "scale_shift_table_a2v_ca_video" in k]
        print(f"[ERROR] Cannot find scale_shift_table_a2v_ca_video. Candidates: {candidates}")
        sys.exit(1)
    print(f"  checkpoint key prefix: '{_ckpt_prefix}'")

    def ckpt_key(block: int, name: str) -> str:
        return f"{_ckpt_prefix}transformer_blocks.{block}.{name}"

    # ── shapes ──────────────────────────────────────────────────────────────
    v_cross_ss_ts = fix["block_slice_native.v_cross_ss_ts"]
    v_cross_gate_ts = fix["block_slice_native.v_cross_gate_ts"]
    sst_video = ckpt[ckpt_key(0, "scale_shift_table_a2v_ca_video")]

    print(f"\nFixture tensors:")
    print(f"  v_cross_ss_ts    : {v_cross_ss_ts.shape}  dtype={v_cross_ss_ts.dtype}")
    print(f"  v_cross_gate_ts  : {v_cross_gate_ts.shape}  dtype={v_cross_gate_ts.dtype}")
    print(f"\nCheckpoint SST:")
    print(f"  scale_shift_table_a2v_ca_video : {sst_video.shape}  dtype={sst_video.dtype}")

    # ── derived dimensions ───────────────────────────────────────────────────
    D = sst_video.shape[1]           # 4096 for real model
    total_ts_width = v_cross_ss_ts.shape[-1]
    num_ada_params_inferred = total_ts_width // D
    print(f"\n  D={D}, ts_total={total_ts_width}, inferred num_ada_params={num_ada_params_inferred}")

    B = v_cross_ss_ts.shape[0]
    seq = v_cross_ss_ts.shape[1]
    ts_bf16 = v_cross_ss_ts.to(torch.bfloat16)
    sst_video_bf16 = sst_video.to(torch.bfloat16)  # matches adaValueAt .convert(ts.dtype())

    sst_ss = sst_video_bf16[:4, :]   # rows 0-3
    sst_gate = sst_video_bf16[4:, :] # row 4

    # ── ZIG formula: adaValueAt(sst, ts, idx) = sst[idx] + ts[:, :, idx*D:(idx+1)*D] ──
    def zig_ada(sst_sub, ts, idx: int) -> torch.Tensor:
        d = sst_sub.shape[1]
        ts_slice = ts[:, :, idx * d : (idx + 1) * d]
        sst_row = sst_sub[idx]  # shape [D]
        return sst_row.unsqueeze(0).unsqueeze(0).expand_as(ts_slice) + ts_slice

    # ── PYTHON formula from get_ada_values(sst_sub, B, ts, indices) ──────────
    def python_ada(sst_sub, ts, indices) -> torch.Tensor:
        num_ada_params = sst_sub.shape[0]
        ts_reshaped = ts.reshape(B, seq, num_ada_params, -1)
        table_part = sst_sub[indices].unsqueeze(0).unsqueeze(0).to(dtype=ts.dtype)
        ts_part = ts_reshaped[:, :, indices, :]
        result = (table_part + ts_part).squeeze(2)
        return result  # [B, seq, D]

    # A→V scale (idx=0) and shift (idx=1)
    zig_scale = zig_ada(sst_ss, ts_bf16, 0)
    zig_shift = zig_ada(sst_ss, ts_bf16, 1)
    py_scale = python_ada(sst_ss, ts_bf16, slice(0, 1))
    py_shift = python_ada(sst_ss, ts_bf16, slice(1, 2))

    def compare(name, a, b):
        diff = (a.float() - b.float()).abs()
        print(f"  {name}: max_abs={diff.max().item():.2e}  mean_abs={diff.mean().item():.2e}  "
              f"exact_match={torch.equal(a, b)}")

    def stats(name, t):
        f = t.float()
        print(f"  {name}: mean={f.mean().item():.4f}  std={f.std().item():.4f}  "
              f"abs_mean={f.abs().mean().item():.4f}  min={f.min().item():.4f}  max={f.max().item():.4f}")

    print("\n── Actual Ada value statistics (scale / shift for A→V) ─────────────")
    stats("scale_v_a2v (Zig)", zig_scale)
    stats("shift_v_a2v (Zig)", zig_shift)
    print()
    print("  NOTE: a2v_x error ≈ error_vx_norm3 × |1 + scale|")
    mean_amplification = (1.0 + zig_scale.float()).abs().mean().item()
    print(f"  mean |1 + scale| = {mean_amplification:.4f}  "
          f"→ expected a2v_x error from 0.0005 vx_norm3 error: {0.0005 * mean_amplification:.5f}")

    print("\n── Zig vs Python Ada extraction ────────────────────────────────────")
    compare("scale (idx=0)", zig_scale, py_scale)
    compare("shift (idx=1)", zig_shift, py_shift)

    # V→A uses idx=2,3
    zig_scale_v2a = zig_ada(sst_ss, ts_bf16, 2)
    zig_shift_v2a = zig_ada(sst_ss, ts_bf16, 3)
    if total_ts_width >= 4 * D:
        py_scale_v2a = python_ada(sst_ss, ts_bf16, slice(2, 3))
        py_shift_v2a = python_ada(sst_ss, ts_bf16, slice(3, 4))
        compare("scale_v2a (idx=2)", zig_scale_v2a, py_scale_v2a)
        compare("shift_v2a (idx=3)", zig_shift_v2a, py_shift_v2a)
    else:
        print(f"  [WARN] ts width {total_ts_width} < 4*D={4*D}: V→A params idx=2,3 are OUT OF BOUNDS in fixture!")

    # ── Now verify the modulation: a2v_x = vx_norm3 * (1+scale) + shift ──────
    if "block_slice_native.vx_norm3_block_0" in fix and "block_slice_native.a2v_x_block_0" in fix:
        vx_norm3 = fix["block_slice_native.vx_norm3_block_0"]
        a2v_x_ref = fix["block_slice_native.a2v_x_block_0"]
        print(f"\nvx_norm3 shape: {vx_norm3.shape}  dtype={vx_norm3.dtype}")
        print(f"a2v_x_ref shape: {a2v_x_ref.shape}  dtype={a2v_x_ref.dtype}")

        vn3 = vx_norm3.to(torch.bfloat16)
        # Zig formula
        a2v_x_zig = vn3 * (1.0 + zig_scale) + zig_shift
        # Python formula
        a2v_x_py = vn3 * (1.0 + py_scale) + py_shift

        print("\n── Modulation output vs fixture a2v_x ─────────────────────────────")
        compare("Zig-formula   vs fixture", a2v_x_zig, a2v_x_ref.to(torch.bfloat16))
        compare("Python-formula vs fixture", a2v_x_py, a2v_x_ref.to(torch.bfloat16))
        compare("Zig vs Python modulation", a2v_x_zig, a2v_x_py)

        # Check if errors are systematic (biased) vs random
        signed_err = (a2v_x_zig.float() - a2v_x_ref.float())
        print(f"\n  Signed error of Zig vs fixture a2v_x:")
        print(f"    mean (bias) = {signed_err.mean().item():.2e}  (non-zero = systematic bias)")
        print(f"    std         = {signed_err.std().item():.2e}")
        print(f"    |mean|/std  = {abs(signed_err.mean().item()) / (signed_err.std().item() + 1e-9):.3f}  (>0.1 suggests directional)")
        print(f"\n  Scale magnitude check: mean |1+scale| = {mean_amplification:.4f}")
        predicted_from_vxnorm3 = 0.0005 * mean_amplification
        print(f"  If vx_norm3 had 0.0005 mean_abs error → predicted a2v_x error ≈ {predicted_from_vxnorm3:.5f}")
        print(f"  Actual a2v_x mean_abs vs fixture = {(a2v_x_zig.float() - a2v_x_ref.float()).abs().mean().item():.5f}")
    else:
        print("\n[INFO] vx_norm3_block_0 or a2v_x_block_0 not in fixture — skipping modulation check")
        print("       Available fixture keys:")
        for k in sorted(fix.keys()):
            print(f"         {k}")


if __name__ == "__main__":
    main()
