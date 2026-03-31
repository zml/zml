"""Quick check: does _prepare_context modify the context?

cd /root/repos/LTX-2 && uv run scripts/check_context_diff.py
"""
import torch
from pathlib import Path
from safetensors.torch import load_file


def cos_sim(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return (torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm() + 1e-10)).item()


def main():
    fix = load_file("trace_run/step2_fixture_step_000_t512.safetensors")

    # Compare raw vs intermediate context
    raw_v = fix["raw.v_context"]
    int_v = fix["intermediate.v_context"]
    print(f"raw.v_context:          shape={list(raw_v.shape)} dtype={raw_v.dtype}")
    print(f"intermediate.v_context: shape={list(int_v.shape)} dtype={int_v.dtype}")
    if raw_v.shape == int_v.shape:
        diff = (raw_v.float() - int_v.float()).abs()
        print(f"  max_diff={diff.max():.6f}  mean_diff={diff.mean():.6f}  equal={torch.equal(raw_v, int_v)}")
        print(f"  cos_sim={cos_sim(raw_v, int_v):.6f}")
    else:
        print(f"  SHAPE MISMATCH!")

    raw_a = fix.get("raw.a_context")
    int_a = fix.get("intermediate.a_context")
    if raw_a is not None and int_a is not None:
        print(f"\nraw.a_context:          shape={list(raw_a.shape)} dtype={raw_a.dtype}")
        print(f"intermediate.a_context: shape={list(int_a.shape)} dtype={int_a.dtype}")
        if raw_a.shape == int_a.shape:
            diff = (raw_a.float() - int_a.float()).abs()
            print(f"  max_diff={diff.max():.6f}  mean_diff={diff.mean():.6f}  equal={torch.equal(raw_a, int_a)}")
        else:
            print(f"  SHAPE MISMATCH!")

    # Also check video_timesteps shape and first-token vs broadcast
    v_ts = fix["intermediate.video_timesteps"]
    print(f"\nvideo_timesteps: shape={list(v_ts.shape)} dtype={v_ts.dtype}")
    if v_ts.shape[1] > 1:
        t0 = v_ts[0, 0, :].float()
        t1 = v_ts[0, 1, :].float()
        t_last = v_ts[0, -1, :].float()
        print(f"  token 0 vs token 1: max_diff={((t0-t1).abs().max()):.6f}  equal={torch.equal(t0, t1)}")
        print(f"  token 0 vs last: max_diff={((t0-t_last).abs().max()):.6f}  equal={torch.equal(t0, t_last)}")
        all_same = all(
            torch.equal(v_ts[0, 0, :], v_ts[0, i, :])
            for i in range(v_ts.shape[1])
        )
        print(f"  all tokens identical: {all_same}")

    a_ts = fix["intermediate.audio_timesteps"]
    print(f"\naudio_timesteps: shape={list(a_ts.shape)} dtype={a_ts.dtype}")
    if a_ts.shape[1] > 1:
        t0 = a_ts[0, 0, :].float()
        t1 = a_ts[0, 1, :].float()
        print(f"  token 0 vs token 1: max_diff={((t0-t1).abs().max()):.6f}  equal={torch.equal(t0, t1)}")
        all_same = all(
            torch.equal(a_ts[0, 0, :], a_ts[0, i, :])
            for i in range(a_ts.shape[1])
        )
        print(f"  all tokens identical: {all_same}")

    # Compare our embedded_timestep shape
    v_emb = fix.get("intermediate.v_embedded_ts")
    a_emb = fix.get("intermediate.a_embedded_ts")
    if v_emb is not None:
        print(f"\nv_embedded_ts: shape={list(v_emb.shape)} dtype={v_emb.dtype}")
    if a_emb is not None:
        print(f"a_embedded_ts: shape={list(a_emb.shape)} dtype={a_emb.dtype}")

    # Check cross_ss_ts shapes
    for key in ["intermediate.v_cross_ss_ts", "intermediate.v_cross_gate_ts",
                 "intermediate.a_cross_ss_ts", "intermediate.a_cross_gate_ts"]:
        t = fix.get(key)
        if t is not None:
            print(f"{key}: shape={list(t.shape)} dtype={t.dtype}")


if __name__ == "__main__":
    main()
