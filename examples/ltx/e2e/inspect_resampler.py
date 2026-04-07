#!/usr/bin/env python3
"""Inspect Python UpSample1d resampler parameters for Zig parity.

Dumps filter values, padding constants, and kernel_size so we can verify
the hardcoded values in forwardSincResample3x.

Usage:
  cd /root/repos/LTX-2 && uv run python /root/repos/zml/examples/ltx/e2e/inspect_resampler.py \
      --checkpoint /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors
"""

import argparse
import json
import sys

import torch
import numpy as np
import safetensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    sys.path.insert(0, "/root/repos/LTX-2/packages/ltx-core/src")
    sys.path.insert(0, "/root/repos/LTX-2/packages/ltx-pipelines/src")

    from ltx_core.model.audio_vae.vocoder import VocoderWithBWE, Vocoder, MelSTFT, UpSample1d

    # Load config
    with safetensors.safe_open(args.checkpoint, framework="pt") as f:
        config = json.loads(f.metadata()["config"])

    bwe_config = config["vocoder"]["bwe"]

    # Build resampler directly
    input_sr = bwe_config["input_sampling_rate"]
    output_sr = bwe_config["output_sampling_rate"]
    ratio = output_sr // input_sr
    print(f"input_sr={input_sr}, output_sr={output_sr}, ratio={ratio}")

    # Inspect UpSample1d class source
    import inspect
    print("\n=== UpSample1d source ===")
    print(inspect.getsource(UpSample1d))

    # Build it
    resampler = UpSample1d(ratio=ratio)

    print("\n=== Resampler attributes ===")
    for attr in sorted(dir(resampler)):
        if attr.startswith('_'):
            continue
        val = getattr(resampler, attr)
        if isinstance(val, (int, float, str, bool)):
            print(f"  {attr} = {val}")
        elif isinstance(val, torch.Tensor):
            print(f"  {attr} = Tensor shape={list(val.shape)} dtype={val.dtype}")
            if val.numel() <= 100:
                print(f"    values = {val.flatten().tolist()}")
        elif callable(val):
            pass
        else:
            print(f"  {attr} = {type(val).__name__}: {val}")

    # Extract filter
    if hasattr(resampler, 'filter'):
        filt = resampler.filter
        print(f"\n=== Filter ===")
        print(f"  shape: {list(filt.shape)}")
        print(f"  dtype: {filt.dtype}")
        flat = filt.flatten().tolist()
        print(f"  num_elements: {len(flat)}")
        print(f"  sum: {sum(flat):.10f}")
        print(f"  sum * ratio: {sum(flat) * ratio:.10f}")
        print(f"  values:")
        for i, v in enumerate(flat):
            if abs(v) > 1e-10:
                print(f"    [{i:3d}] = {v:.10e}")
    elif hasattr(resampler, 'kernel'):
        filt = resampler.kernel
        print(f"\n=== Kernel ===")
        print(f"  shape: {list(filt.shape)}")
        flat = filt.flatten().tolist()
        print(f"  values:")
        for i, v in enumerate(flat):
            if abs(v) > 1e-10:
                print(f"    [{i:3d}] = {v:.10e}")

    # Check padding attributes
    print(f"\n=== Padding constants ===")
    for name in ['pad', 'pad_left', 'pad_right', 'kernel_size', 'ratio',
                 'rolloff', 'lowpass_filter_width', 'width', 'stride',
                 'padding', 'output_padding']:
        if hasattr(resampler, name):
            print(f"  {name} = {getattr(resampler, name)}")

    # Also test with a simple input to verify output shape
    print(f"\n=== Forward test ===")
    test_input = torch.randn(1, 2, 1280)
    with torch.no_grad():
        test_output = resampler(test_input)
    print(f"  input:  {list(test_input.shape)}")
    print(f"  output: {list(test_output.shape)}")
    print(f"  expected output length: {1280 * ratio}")

    # Also dump the Zig hardcoded filter for comparison
    zig_filter = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        2.2237423e-04, -8.3113974e-04, 1.6642004e-03, -2.4983494e-03, 3.1095231e-03,
        3.3000001e-01,
        3.1095231e-03, -2.4983494e-03, 1.6642004e-03, -8.3113974e-04, 2.2237423e-04,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]

    if hasattr(resampler, 'filter'):
        py_flat = filt.flatten().tolist()
        if len(py_flat) == len(zig_filter):
            print(f"\n=== Filter comparison (Python vs Zig hardcoded) ===")
            max_diff = 0
            for i, (p, z) in enumerate(zip(py_flat, zig_filter)):
                d = abs(p - z)
                if d > 1e-8:
                    print(f"  [{i:3d}] python={p:.10e}  zig={z:.10e}  diff={d:.10e}")
                max_diff = max(max_diff, d)
            print(f"  Max filter diff: {max_diff:.10e}")
        else:
            print(f"\n  WARNING: filter size mismatch: Python={len(py_flat)} vs Zig={len(zig_filter)}")


if __name__ == "__main__":
    main()
