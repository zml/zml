#!/usr/bin/env python3
"""Export Gemma hidden states and reference embeddings for Zig text embedding validation.

Runs Gemma forward pass only (no LTX projection/connector weights), saves:
  - 49 hidden states stacked as [1, S, 3840, 49] bf16
  - attention_mask [1, S] int64
  - Reference: final embeddings computed by the full Python EmbeddingsProcessor

Two files per prompt (positive + negative), plus one reference file.

Usage (on GPU server):
  cd /root/repos/LTX-2
  uv run examples/ltx/export_gemma_hidden_states.py \
      --prompt "A beautiful sunset over the ocean" \
      --negative-prompt "blurry, out of focus" \
      --checkpoint ~/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
      --gemma-root ~/models/gemma-3-12b-it \
      --output-dir /root/gemma_export/
"""

import argparse
import logging
from pathlib import Path

import torch
from safetensors.torch import save_file

from ltx_core.loader import DummyRegistry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.text_encoders.gemma.encoders.encoder_configurator import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    GEMMA_LLM_KEY_OPS,
    GEMMA_MODEL_OPS,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoderConfigurator,
)
from ltx_pipelines.utils.blocks import gpu_model, module_ops_from_gemma_root

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Gemma hidden states for Zig validation")
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over the ocean")
    parser.add_argument("--negative-prompt", type=str, default="blurry, out of focus")
    parser.add_argument("--checkpoint", type=str,
                        default=str(Path("~/models/ltx-2.3/ltx-2.3-22b-dev.safetensors").expanduser()))
    parser.add_argument("--gemma-root", type=str,
                        default=str(Path("~/models/gemma-3-12b-it").expanduser()))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def save_safetensors(tensors: dict[str, torch.Tensor], path: Path,
                     metadata: dict[str, str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
    save_file(clean, str(path), metadata=metadata)
    print(f"  Saved {path.name}: {len(clean)} tensors")
    for k, v in sorted(clean.items()):
        print(f"    {k:40s}  {list(v.shape)}  {v.dtype}")


def find_matching_file(root: str, pattern: str) -> Path:
    """Find a file matching a glob pattern under root."""
    root_path = Path(root)
    matches = list(root_path.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} under {root}")
    return matches[0]


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    out = args.output_dir

    print(f"Prompt:     {args.prompt!r}")
    print(f"Neg prompt: {args.negative_prompt!r}")
    print(f"Output:     {out}")

    # ========================================================================
    # Phase 1: Load Gemma, run forward pass, capture hidden states
    # ========================================================================
    print("\n=== Phase 1: Gemma forward pass ===")

    gemma_root = args.gemma_root
    module_ops = module_ops_from_gemma_root(gemma_root)
    model_folder = find_matching_file(gemma_root, "model*.safetensors").parent
    weight_paths = [str(p) for p in model_folder.rglob("*.safetensors")]

    text_encoder_builder = Builder(
        model_path=tuple(weight_paths),
        model_class_configurator=GemmaTextEncoderConfigurator,
        model_sd_ops=GEMMA_LLM_KEY_OPS,
        module_ops=(GEMMA_MODEL_OPS, *module_ops),
        registry=DummyRegistry(),
    )

    with gpu_model(text_encoder_builder.build(device=device, dtype=dtype).eval()) as text_encoder:
        prompts = [args.prompt, args.negative_prompt]
        raw_outputs = []
        for i, prompt in enumerate(prompts):
            hidden_states, attention_mask = text_encoder.encode(prompt)
            raw_outputs.append((hidden_states, attention_mask))
            label = "pos" if i == 0 else "neg"
            print(f"  {label}: {len(hidden_states)} layers, "
                  f"shape={list(hidden_states[0].shape)}, "
                  f"mask sum={attention_mask.sum().item()}/{attention_mask.shape[-1]}")

    # Save hidden states
    print("\n=== Saving hidden states ===")
    for i, (hidden_states, attention_mask) in enumerate(raw_outputs):
        label = "pos" if i == 0 else "neg"
        # Stack all hidden states into [B, S, D, L]
        stacked = torch.stack(list(hidden_states), dim=-1)  # [1, S, 3840, 49]
        tensors = {
            "stacked_hidden_states": stacked.to(dtype),
            "attention_mask": attention_mask,
        }
        save_safetensors(tensors, out / f"{label}_hidden_states.safetensors")

    # ========================================================================
    # Phase 2: Run EmbeddingsProcessor for reference outputs
    # ========================================================================
    print("\n=== Phase 2: EmbeddingsProcessor reference ===")

    embeddings_processor_builder = Builder(
        model_path=args.checkpoint,
        model_class_configurator=EmbeddingsProcessorConfigurator,
        model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
        registry=DummyRegistry(),
    )

    with gpu_model(
        embeddings_processor_builder.build(device=device, dtype=dtype).to(device).eval()
    ) as embeddings_processor:
        ref_tensors = {}
        feature_tensors = {}
        for i, (hidden_states, attention_mask) in enumerate(raw_outputs):
            label = "pos" if i == 0 else "neg"

            # Step A: Feature extraction only (for intermediate validation)
            video_feats, audio_feats = embeddings_processor.feature_extractor(
                hidden_states, attention_mask, padding_side="left"
            )
            feature_tensors[f"video_features_{label}"] = video_feats
            feature_tensors[f"audio_features_{label}"] = audio_feats
            print(f"  {label} features: video={list(video_feats.shape)}, audio={list(audio_feats.shape)}")

            # Step B: Full pipeline (feature extraction + connectors)
            result = embeddings_processor.process_hidden_states(
                hidden_states, attention_mask, padding_side="left"
            )
            ref_tensors[f"v_context_{label}"] = result.video_encoding
            ref_tensors[f"a_context_{label}"] = result.audio_encoding
            ref_tensors[f"binary_mask_{label}"] = result.attention_mask
            print(f"  {label} embeddings: video={list(result.video_encoding.shape)}, "
                  f"audio={list(result.audio_encoding.shape)}, "
                  f"mask={list(result.attention_mask.shape)}")

        save_safetensors(feature_tensors, out / "ref_features.safetensors")
        save_safetensors(ref_tensors, out / "ref_embeddings.safetensors")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"\nHidden states:")
    print(f"  Positive: {out / 'pos_hidden_states.safetensors'}")
    print(f"  Negative: {out / 'neg_hidden_states.safetensors'}")
    print(f"\nReferences:")
    print(f"  Features:   {out / 'ref_features.safetensors'}")
    print(f"  Embeddings: {out / 'ref_embeddings.safetensors'}")
    print(f"\nUsage with Zig validator:")
    print(f"  bazel run //examples/ltx:validate_text_embeddings -- \\")
    print(f"    --hidden-states {out / 'pos_hidden_states.safetensors'} \\")
    print(f"    --ref-embeddings {out / 'ref_embeddings.safetensors'} \\")
    print(f"    --ref-features {out / 'ref_features.safetensors'} \\")
    print(f"    --checkpoint {args.checkpoint}")


if __name__ == "__main__":
    main()
