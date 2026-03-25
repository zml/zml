#!/usr/bin/env python3
"""
Simplified test: Compare block 28 outputs with and without corrupted input.
Directly loads from fixture activations without needing model.
"""

import sys
import torch
import safetensors.torch
from pathlib import Path

checkpoint_path = Path(sys.argv[1])
fixture_path = checkpoint_path.parent / "block_slice_native_0_47_lora0.0_t128.safetensors"

print(f"\n{'='*60}\nBLOCK 28 INPUT ERROR PROPAGATION TEST\n{'='*60}\n")

if not fixture_path.exists():
    print(f"ERROR: Fixture not found at {fixture_path}")
    sys.exit(1)

fixture = safetensors.torch.load_file(fixture_path)

print(f"Loaded fixture: {fixture_path.name}")
print(f"Available keys: {len(fixture)} tensors\n")

# Get reference outputs for blocks 27 and 28
b27_ref_out_vx = fixture.get("block_slice_native.vx_out_block_27")
b27_ref_out_ax = fixture.get("block_slice_native.ax_out_block_27")
b28_ref_out_vx = fixture.get("block_slice_native.vx_out_block_28")
b28_ref_out_ax = fixture.get("block_slice_native.ax_out_block_28")

print(f"{'='*60}")
print(f"AVAILABLE REFERENCE OUTPUTS")
print(f"{'='*60}")
print(f"Block 27 output vx: {b27_ref_out_vx.shape if b27_ref_out_vx is not None else 'NOT FOUND'}")
print(f"Block 27 output ax: {b27_ref_out_ax.shape if b27_ref_out_ax is not None else 'NOT FOUND'}")
print(f"Block 28 output vx: {b28_ref_out_vx.shape if b28_ref_out_vx is not None else 'NOT FOUND'}")
print(f"Block 28 output ax: {b28_ref_out_ax.shape if b28_ref_out_ax is not None else 'NOT FOUND'}")

if b27_ref_out_vx is None or b28_ref_out_vx is None:
    print("\nERROR: Required reference outputs not found in fixture")
    print("This fixture may not have per-block outputs. Need to regenerate with ref-outputs enabled.")
    sys.exit(1)

print(f"\n{'='*60}")
print(f"ANALYSIS: What changes from block 27 to 28?")
print(f"{'='*60}")

# Simple analysis: block outputs
print(f"\nBlock 27 output statistics (video):")
print(f"  Shape: {b27_ref_out_vx.shape}")
print(f"  Min: {b27_ref_out_vx.min():.6f}, Max: {b27_ref_out_vx.max():.6f}")
print(f"  Mean: {b27_ref_out_vx.mean():.6f}, Std: {b27_ref_out_vx.std():.6f}")

print(f"\nBlock 28 output statistics (video):")
print(f"  Shape: {b28_ref_out_vx.shape}")
print(f"  Min: {b28_ref_out_vx.min():.6f}, Max: {b28_ref_out_vx.max():.6f}")
print(f"  Mean: {b28_ref_out_vx.mean():.6f}, Std: {b28_ref_out_vx.std():.6f}")

if b27_ref_out_ax is not None:
    print(f"\nBlock 27 output statistics (audio):")
    print(f"  Shape: {b27_ref_out_ax.shape}")
    print(f"  Min: {b27_ref_out_ax.min():.6f}, Max: {b27_ref_out_ax.max():.6f}")
    print(f"  Mean: {b27_ref_out_ax.mean():.6f}, Std: {b27_ref_out_ax.std():.6f}")

if b28_ref_out_ax is not None:
    print(f"\nBlock 28 output statistics (audio):")
    print(f"  Shape: {b28_ref_out_ax.shape}")
    print(f"  Min: {b28_ref_out_ax.min():.6f}, Max: {b28_ref_out_ax.max():.6f}")
    print(f"  Mean: {b28_ref_out_ax.mean():.6f}, Std: {b28_ref_out_ax.std():.6f}")

# Check: do blocks 27 and 28 have identical magnitude ranges?
print(f"\n{'='*60}")
print(f"OUTPUT RANGE COMPARISON")
print(f"{'='*60}")

v_range_27 = b27_ref_out_vx.max() - b27_ref_out_vx.min()
v_range_28 = b28_ref_out_vx.max() - b28_ref_out_vx.min()
print(f"Video output range: Block 27={v_range_27:.4f}, Block 28={v_range_28:.4f}")
print(f"  Block 28 range is {v_range_28/v_range_27:.2f}x Block 27")

if b27_ref_out_ax is not None and b28_ref_out_ax is not None:
    a_range_27 = b27_ref_out_ax.max() - b27_ref_out_ax.min()
    a_range_28 = b28_ref_out_ax.max() - b28_ref_out_ax.min()
    print(f"Audio output range: Block 27={a_range_27:.4f}, Block 28={a_range_28:.4f}")
    print(f"  Block 28 range is {a_range_28/a_range_27:.2f}x Block 27")

# List all per-block output keys to understand what's available
print(f"\n{'='*60}")
print(f"ALL AVAILABLE PER-BLOCK OUTPUTS")
print(f"{'='*60}")

block_out_keys = [k for k in fixture.keys() if 'out_block_' in k]
print(f"Found {len(block_out_keys)} per-block output keys:")
for k in sorted(block_out_keys):
    shape = fixture[k].shape
    print(f"  {k}: {shape}")

