#!/usr/bin/env python3
"""Export guider combine fixture for Zig parity checking.

Generates synthetic cond/neg/ptb/iso tensors and computes the Stage 1
guidance combine formula:
    pred = cond + (cfg-1)*(cond-neg) + stg*(cond-ptb) + (mod-1)*(cond-iso)
    pred *= rescale * (cond.std() / pred.std()) + (1 - rescale)

No model weights needed — this is pure tensor math.

Usage:
    python export_guider_combine_fixture.py [output.safetensors]
"""

import sys
import torch
from safetensors.torch import save_file


def guider_combine(cond, neg, ptb, iso, cfg, stg, mod, rescale):
    """Python reference: Stage 1 guidance combine per modality."""
    pred = (
        cond
        + (cfg - 1) * (cond - neg)
        + stg * (cond - ptb)
        + (mod - 1) * (cond - iso)
    )
    if rescale != 0:
        factor = rescale * (cond.std() / pred.std()) + (1 - rescale)
        pred = pred * factor
    return pred


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else "guider_combine_fixture.safetensors"

    torch.manual_seed(42)

    # Representative shapes matching Stage 1 half-resolution inference
    B = 1
    T_v = 3072   # video tokens at half-res
    T_a = 512    # audio tokens
    D = 128      # output projection dimension (patch channels)

    # Generate synthetic inputs with realistic-ish distributions
    # Use randn scaled to mimic typical denoised velocity outputs
    cond_v = torch.randn(B, T_v, D, dtype=torch.bfloat16) * 0.5
    neg_v = torch.randn(B, T_v, D, dtype=torch.bfloat16) * 0.5
    ptb_v = cond_v + torch.randn(B, T_v, D, dtype=torch.bfloat16) * 0.1  # STG: close to cond
    iso_v = torch.randn(B, T_v, D, dtype=torch.bfloat16) * 0.5

    cond_a = torch.randn(B, T_a, D, dtype=torch.bfloat16) * 0.5
    neg_a = torch.randn(B, T_a, D, dtype=torch.bfloat16) * 0.5
    ptb_a = cond_a + torch.randn(B, T_a, D, dtype=torch.bfloat16) * 0.1
    iso_a = torch.randn(B, T_a, D, dtype=torch.bfloat16) * 0.5

    # Default Stage 1 guidance parameters
    cfg_v, stg_v, mod_v, rescale_v = 3.0, 1.0, 3.0, 0.7
    cfg_a, stg_a, mod_a, rescale_a = 7.0, 1.0, 3.0, 0.7

    # Compute reference outputs in float32 (matching Zig's compute path)
    guided_v = guider_combine(
        cond_v.float(), neg_v.float(), ptb_v.float(), iso_v.float(),
        cfg_v, stg_v, mod_v, rescale_v,
    ).to(torch.bfloat16)

    guided_a = guider_combine(
        cond_a.float(), neg_a.float(), ptb_a.float(), iso_a.float(),
        cfg_a, stg_a, mod_a, rescale_a,
    ).to(torch.bfloat16)

    # Also compute a no-rescale reference (rescale=0) for diagnostic
    guided_v_norescale = guider_combine(
        cond_v.float(), neg_v.float(), ptb_v.float(), iso_v.float(),
        cfg_v, stg_v, mod_v, 0.0,
    ).to(torch.bfloat16)

    guided_a_norescale = guider_combine(
        cond_a.float(), neg_a.float(), ptb_a.float(), iso_a.float(),
        cfg_a, stg_a, mod_a, 0.0,
    ).to(torch.bfloat16)

    tensors = {
        # Inputs
        "cond_v": cond_v.contiguous(),
        "neg_v": neg_v.contiguous(),
        "ptb_v": ptb_v.contiguous(),
        "iso_v": iso_v.contiguous(),
        "cond_a": cond_a.contiguous(),
        "neg_a": neg_a.contiguous(),
        "ptb_a": ptb_a.contiguous(),
        "iso_a": iso_a.contiguous(),
        # Scalar guidance params (as 1-element tensors)
        "cfg_v": torch.tensor([cfg_v], dtype=torch.float32),
        "stg_v": torch.tensor([stg_v], dtype=torch.float32),
        "mod_v": torch.tensor([mod_v], dtype=torch.float32),
        "rescale_v": torch.tensor([rescale_v], dtype=torch.float32),
        "cfg_a": torch.tensor([cfg_a], dtype=torch.float32),
        "stg_a": torch.tensor([stg_a], dtype=torch.float32),
        "mod_a": torch.tensor([mod_a], dtype=torch.float32),
        "rescale_a": torch.tensor([rescale_a], dtype=torch.float32),
        # Reference outputs
        "guided_v": guided_v.contiguous(),
        "guided_a": guided_a.contiguous(),
        # Diagnostic: no-rescale reference
        "guided_v_norescale": guided_v_norescale.contiguous(),
        "guided_a_norescale": guided_a_norescale.contiguous(),
    }

    save_file(tensors, output_path)
    print(f"Saved fixture to {output_path}")

    # Print diagnostics
    for name, t in tensors.items():
        print(f"  {name}: shape={list(t.shape)} dtype={t.dtype}")

    # Basic sanity checks
    print(f"\nSanity checks:")
    v_rescale_effect = torch.nn.functional.cosine_similarity(
        guided_v.float().flatten(), guided_v_norescale.float().flatten(), dim=0
    )
    a_rescale_effect = torch.nn.functional.cosine_similarity(
        guided_a.float().flatten(), guided_a_norescale.float().flatten(), dim=0
    )
    print(f"  Video rescale vs no-rescale cos_sim: {v_rescale_effect:.6f}")
    print(f"  Audio rescale vs no-rescale cos_sim: {a_rescale_effect:.6f}")

    # Verify inputs differ enough to make guidance meaningful
    v_cond_neg_cos = torch.nn.functional.cosine_similarity(
        cond_v.float().flatten(), neg_v.float().flatten(), dim=0
    )
    print(f"  Video cond vs neg cos_sim: {v_cond_neg_cos:.6f} (should be near 0)")


if __name__ == "__main__":
    main()
