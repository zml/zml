#!/usr/bin/env python3
"""
Check if block 28 has any shape or dimension changes that only manifest in free-running mode.
"""

import sys
import torch
import safetensors.torch
from pathlib import Path

checkpoint_path = Path(sys.argv[1])

print(f"\n{'='*60}\nBLOCK 28 SHAPE & DIMENSION ANALYSIS\n{'='*60}\n")

# Load reference model
sys.path.insert(0, str(Path(__file__).parent))
from replay_stage2_transformer_step import load_reference_model

model = load_reference_model()

# Check model dimension config
print(f"Model config:")
print(f"  hidden_size (text): {model.config.hidden_size}")
print(f"  video_latent_channels: {model.config.video_latent_channels}")
print(f"  audio_latent_channels: {model.config.audio_latent_channels}")

# Get blocks 0, 27, 28 and check if they're identical in structure
block0 = model.transformer_blocks[0]
block27 = model.transformer_blocks[27]
block28 = model.transformer_blocks[28]

print(f"\n{'='*60}\nBLOCK STRUCTURE COMPARISON\n{'='*60}\n")

# Check attention dimensions
print(f"Block 0 attention heads:")
print(f"  attn1 (self): num_heads={block0.attn1.num_heads}, head_dim={block0.attn1.head_dim}")
print(f"  attn2 (cross): num_heads={block0.attn2.num_heads}, head_dim={block0.attn2.head_dim}")

print(f"\nBlock 27 attention heads:")
print(
    f"  attn1 (self): num_heads={block27.attn1.num_heads}, head_dim={block27.attn1.head_dim}"
)
print(
    f"  attn2 (cross): num_heads={block27.attn2.num_heads}, head_dim={block27.attn2.head_dim}"
)

print(f"\nBlock 28 attention heads:")
print(
    f"  attn1 (self): num_heads={block28.attn1.num_heads}, head_dim={block28.attn1.head_dim}"
)
print(
    f"  attn2 (cross): num_heads={block28.attn2.num_heads}, head_dim={block28.attn2.head_dim}"
)

# Check if there's a special audio attention at block 28
print(f"\n{'='*60}\nCRISS-CROSS ATTENTION (A2V/V2A) ANALYSIS\n{'='*60}\n")

print(f"Block 27:")
print(f"  audio_to_video_attn: {hasattr(block27, 'audio_to_video_attn')}")
print(f"  video_to_audio_attn: {hasattr(block27, 'video_to_audio_attn')}")

print(f"\nBlock 28:")
print(f"  audio_to_video_attn: {hasattr(block28, 'audio_to_video_attn')}")
print(f"  video_to_audio_attn: {hasattr(block28, 'video_to_audio_attn')}")

if hasattr(block28, 'audio_to_video_attn') and block28.audio_to_video_attn is not None:
    print(f"\n  audio_to_video_attn num_heads: {block28.audio_to_video_attn.num_heads}")
    print(f"  audio_to_video_attn head_dim: {block28.audio_to_video_attn.head_dim}")

if hasattr(block28, 'video_to_audio_attn') and block28.video_to_audio_attn is not None:
    print(f"\n  video_to_audio_attn num_heads: {block28.video_to_audio_attn.num_heads}")
    print(f"  video_to_audio_attn head_dim: {block28.video_to_audio_attn.head_dim}")

# Load fixture and check if there are per-block shape changes
fixture_path = checkpoint_path.parent / checkpoint_path.name.replace(
    "blocks_0_47", "block_slice_native_0_47"
).replace("_merged", "")

if fixture_path.exists():
    fixture = safetensors.torch.load_file(fixture_path)
    print(f"\n{'='*60}\nFIXTURE SHAPES\n{'='*60}\n")

    # Check block 27 and 28 input/output shapes
    for i in [27, 28]:
        key_in_vx = f"activations.transformer_block_{i}.input.vx"
        key_out_vx = f"activations.transformer_block_{i}.output.vx"
        key_in_ax = f"activations.transformer_block_{i}.input.ax"
        key_out_ax = f"activations.transformer_block_{i}.output.ax"

        in_vx = fixture.get(key_in_vx)
        out_vx = fixture.get(key_out_vx)
        in_ax = fixture.get(key_in_ax)
        out_ax = fixture.get(key_out_ax)

        print(f"\nBlock {i}:")
        if in_vx is not None:
            print(f"  Input vx: {in_vx.shape}")
        if out_vx is not None:
            print(f"  Output vx: {out_vx.shape}")
        if in_ax is not None:
            print(f"  Input ax: {in_ax.shape}")
        if out_ax is not None:
            print(f"  Output ax: {out_ax.shape}")
