#!/usr/bin/env python3
"""Export intermediate connector outputs for debugging Zig divergence.

Captures after each stage of the Embeddings1DConnector:
  1. After register replacement (before blocks)
  2. After transformer block 0
  3. After transformer block 1
  4. After final RMS norm (= connector output)

Usage (on GPU server):
  cd ~/repos/LTX-2
  uv run ~/repos/zml/examples/ltx/export_connector_intermediates.py \
      --hidden-states ~/gemma_export/pos_hidden_states.safetensors \
      --checkpoint ~/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
      --output ~/gemma_export/connector_intermediates.safetensors
"""

import argparse
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ltx_core.loader import DummyRegistry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.text_encoders.gemma.encoders.encoder_configurator import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    EmbeddingsProcessorConfigurator,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-states", type=str, required=True,
                        help="Path to pos_hidden_states.safetensors")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to ltx checkpoint")
    parser.add_argument("--output", type=str, required=True,
                        help="Output safetensors path")
    return parser.parse_args()


@torch.inference_mode()
def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Load hidden states
    hs_data = load_file(args.hidden_states)
    stacked_hs = hs_data["stacked_hidden_states"].to(device=device, dtype=dtype)
    attention_mask = hs_data["attention_mask"].to(device=device)

    print(f"stacked_hidden_states: {stacked_hs.shape} {stacked_hs.dtype}")
    print(f"attention_mask: {attention_mask.shape} {attention_mask.dtype}")
    print(f"attention_mask sum: {attention_mask.sum().item()}/{attention_mask.shape[-1]}")

    # Load embeddings processor
    builder = Builder(
        model_path=args.checkpoint,
        model_class_configurator=EmbeddingsProcessorConfigurator,
        model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
        registry=DummyRegistry(),
    )
    processor = builder.build(device=device, dtype=dtype).to(device).eval()

    # Print connector config
    for name in ["video_connector", "audio_connector"]:
        conn = getattr(processor, name)
        print(f"\n{name}:")
        print(f"  num_attention_heads: {conn.num_attention_heads}")
        print(f"  inner_dim: {conn.inner_dim}")
        print(f"  learnable_registers: {conn.learnable_registers.shape}")
        print(f"  num_blocks: {len(conn.transformer_1d_blocks)}")
        print(f"  rope_type: {conn.rope_type}")
        print(f"  positional_embedding_theta: {conn.positional_embedding_theta}")
        print(f"  positional_embedding_max_pos: {conn.positional_embedding_max_pos}")
        for i, blk in enumerate(conn.transformer_1d_blocks):
            attn = blk.attn1
            print(f"  block[{i}].attn1.to_q: {attn.to_q.weight.shape}")
            print(f"  block[{i}].attn1.to_gate_logits: {attn.to_gate_logits is not None}")
            if attn.to_gate_logits is not None:
                print(f"    gate weight: {attn.to_gate_logits.weight.shape}")
            print(f"  block[{i}].ff.net[0].proj: {blk.ff.net[0].proj.weight.shape}")

    # Step 1: Feature extraction
    hidden_states_list = [stacked_hs[:, :, :, i] for i in range(stacked_hs.shape[-1])]
    video_feats, audio_feats = processor.feature_extractor(
        hidden_states_list, attention_mask, padding_side="left"
    )
    print(f"\nvideo_features: {video_feats.shape} {video_feats.dtype}")
    print(f"audio_features: {audio_feats.shape} {audio_feats.dtype}")

    # Step 2: Additive mask (same as processor.process_hidden_states does internally)
    # The mask for the connector is the additive mask [B, 1, 1, S]
    mask_2d = attention_mask.float()  # [B, S]
    additive_mask = (mask_2d - 1.0) * torch.finfo(video_feats.dtype).max
    additive_mask = additive_mask.to(video_feats.dtype)
    additive_mask = additive_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]

    tensors = {}

    # Step 3: Run each connector with instrumentation
    for prefix, conn, feats in [
        ("video", processor.video_connector, video_feats),
        ("audio", processor.audio_connector, audio_feats),
    ]:
        print(f"\n=== {prefix} connector ===")
        hs = feats.clone()
        mask = additive_mask.clone()

        # Register replacement
        if conn.num_learnable_registers:
            hs, mask = conn._replace_padded_with_learnable_registers(hs, mask)
        tensors[f"{prefix}_after_replace"] = hs.detach().cpu().contiguous()
        tensors[f"{prefix}_mask_after_replace"] = mask.detach().cpu().contiguous()
        print(f"  after register replacement: {hs.shape}")

        # RoPE (computed inside forward, but we need it for the blocks)
        from ltx_core.model.transformer.rope import (
            precompute_freqs_cis,
            generate_freq_grid_pytorch,
            generate_freq_grid_np,
        )
        indices_grid = torch.arange(hs.shape[1], dtype=torch.float32, device=device)
        indices_grid = indices_grid[None, None, :]
        freq_gen = generate_freq_grid_np if conn.double_precision_rope else generate_freq_grid_pytorch
        freqs_cis = precompute_freqs_cis(
            indices_grid=indices_grid,
            dim=conn.inner_dim,
            out_dtype=hs.dtype,
            theta=conn.positional_embedding_theta,
            max_pos=conn.positional_embedding_max_pos,
            num_attention_heads=conn.num_attention_heads,
            rope_type=conn.rope_type,
            freq_grid_generator=freq_gen,
        )

        # Save RoPE cos/sin for debugging
        cos_freq, sin_freq = freqs_cis
        tensors[f"{prefix}_rope_cos"] = cos_freq.detach().cpu().contiguous()
        tensors[f"{prefix}_rope_sin"] = sin_freq.detach().cpu().contiguous()
        print(f"  rope cos: {cos_freq.shape} sin: {sin_freq.shape}")

        # Run blocks one by one
        for i, block in enumerate(conn.transformer_1d_blocks):
            hs = block(hs, attention_mask=mask, pe=freqs_cis)
            tensors[f"{prefix}_after_block{i}"] = hs.detach().cpu().contiguous()
            print(f"  after block {i}: {hs.shape}, "
                  f"mean={hs.float().mean().item():.6f}, "
                  f"std={hs.float().std().item():.6f}")

        # Final RMS norm
        hs = torch.nn.functional.rms_norm(hs, (hs.shape[-1],), weight=None, eps=1e-6)
        tensors[f"{prefix}_after_final_norm"] = hs.detach().cpu().contiguous()
        print(f"  after final norm: {hs.shape}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))
    print(f"\nSaved {len(tensors)} tensors to {out_path}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:40s}  {list(v.shape)}  {v.dtype}")


if __name__ == "__main__":
    main()
