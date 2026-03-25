#!/usr/bin/env python3
"""
Block 28 Transition Diagnostic
====================================
Checks what changes betweenblocks 27 and 28.
Determines if issue is in model architecture, loading, or computation.
"""

import torch
import safetensors.torch as st
import sys

def diagnostic():
    if len(sys.argv) < 2:
        print("Usage: python block_28_diagnostic.py <checkpoint.safetensors>")
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    
    print("="*60)
    print("BLOCK 28 TRANSITION DIAGNOSTIC")
    print("="*60)
    
    # Load checkpoint
    ckpt = st.load_file(ckpt_path)
    
    # Get transformer blocks 27 and 28 keys
    keys_27 = [k for k in ckpt.keys() if 'transformer_blocks.27' in k]
    keys_28 = [k for k in ckpt.keys() if 'transformer_blocks.28' in k]
    
    print(f"\nBlock 27 keys ({len(keys_27)}):")
    for k in sorted(keys_27)[:5]:
        print(f"  {k}: shape={ckpt[k].shape} dtype={ckpt[k].dtype}")
    
    print(f"\nBlock 28 keys ({len(keys_28)}):")
    for k in sorted(keys_28)[:5]:
        print(f"  {k}: shape={ckpt[k].shape} dtype={ckpt[k].dtype}")
    
    # Check for structural changes
    print("\n" + "="*60)
    print("STRUCTURAL ANALYSIS")
    print("="*60)
    
    if len(keys_27) != len(keys_28):
        print(f"⚠️ STRUCTURE MISMATCH: Block 27 has {len(keys_27)} keys, Block 28 has {len(keys_28)}")
    else:
        print(f"✓ Structure matches: Both have {len(keys_27)} keys")
    
    # Extract architecture info from keys
    print("\nKey patterns in blocks:")
    
    patterns_27 = set()
    patterns_28 = set()
    
    for k in keys_27:
        subpart = k.split('transformer_blocks.27.')[-1]
        patterns_27.add(subpart.split('.')[0])
    
    for k in keys_28:
        subpart = k.split('transformer_blocks.28.')[-1]
        patterns_28.add(subpart.split('.')[0])
    
    print(f"  Block 27 sublayers: {sorted(patterns_27)}")
    print(f"  Block 28 sublayers: {sorted(patterns_28)}")
    
    if patterns_27 == patterns_28:
        print("  ✓ Sublayer structure identical")
    else:
        print(f"  ⚠️ Sublayer structure differs!")
        print(f"    In 27 only: {patterns_27 - patterns_28}")
        print(f"    In 28 only: {patterns_28 - patterns_27}")
    
    # Check value ranges to see if there's a saturation pattern
    print("\n" + "="*60)
    print("VALUE RANGE ANALYSIS (Weight Statistics)")
    print("="*60)
    
    def get_stats(keys):
        stats = {
            'min': float('inf'),
            'max': float('-inf'),
            'mean': 0,
            'std': 0,
            'num_zeros': 0,
            'num_inf': 0,
            'num_nan': 0,
        }
        all_vals = []
        for k in keys:
            t = ckpt[k]
            all_vals.append(t.flatten())
            if torch.isnan(t).any():
                stats['num_nan'] += torch.isnan(t).sum().item()
            if torch.isinf(t).any():
                stats['num_inf'] += torch.isinf(t).sum().item()
            stats['num_zeros'] += (t == 0).sum().item()
        
        if all_vals:
            combined = torch.cat(all_vals)
            finite_mask = torch.isfinite(combined)
            if finite_mask.any():
                finite = combined[finite_mask]
                stats['min'] = finite.min().item()
                stats['max'] = finite.max().item()
                stats['mean'] = finite.mean().item()
                stats['std'] = finite.std().item()
        
        return stats
    
    stats_27 = get_stats(keys_27)
    stats_28 = get_stats(keys_28)
    
    print("\nBlock 27 weights:")
    print(f"  Range: [{stats_27['min']:.6f}, {stats_27['max']:.6f}]")
    print(f"  Mean: {stats_27['mean']:.6f}, Std: {stats_27['std']:.6f}")
    print(f"  NaNs: {stats_27['num_nan']}, Infs: {stats_27['num_inf']}, Zeros: {stats_27['num_zeros']}")
    
    print("\nBlock 28 weights:")
    print(f"  Range: [{stats_28['min']:.6f}, {stats_28['max']:.6f}]")
    print(f"  Mean: {stats_28['mean']:.6f}, Std: {stats_28['std']:.6f}")
    print(f"  NaNs: {stats_28['num_nan']}, Infs: {stats_28['num_inf']}, Zeros: {stats_28['num_zeros']}")
    
    delta_max = stats_28['max'] - stats_27['max']
    delta_min = stats_28['min'] - stats_27['min']
    print(f"\nBlock 28 vs 27 deltas:")
    print(f"  Max range delta: {delta_max:+.6f}")
    print(f"  Min range delta: {delta_min:+.6f}")
    print(f"  Mean delta: {stats_28['mean'] - stats_27['mean']:+.6f}")
    
    # Summary
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    if stats_28['num_nan'] > 0 or stats_28['num_inf'] > 0:
        print("⚠️  Block 28 weights contain NaN/Inf!")
    
    if patterns_27 != patterns_28:
        print("⚠️  Block 28 has different architecture than Block 27!")
    else:
        print("✓  Block 28 structure matches Block 27")
    
    if abs(delta_max) > 0.1 or abs(delta_min) > 0.1:
        print(f"⚠️  Block 28 weight ranges significantly different from Block 27")
    else:
        print(f"✓  Block 28 weight ranges similar to Block 27")

if __name__ == '__main__':
    diagnostic()
