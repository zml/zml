"""Export a LoRA-merged stage-2 block0 attention checkpoint for Zig parity checks.

This script builds TI2VidTwoStagesPipeline with optional distilled LoRA strength,
then saves only stage-2 block0 attention tensors into a compact safetensors file.

Typical usage:
  uv run python examples/ltx/export_stage2_block0_attention_checkpoint.py \
    --distilled-lora-strength 0.5 \
    --output /root/repos/LTX-2/trace_run/stage2_block0_attention_lora0.5_merged.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export stage-2 block0 attention tensors from TI2VidTwoStagesPipeline")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser(),
        help="Base LTX checkpoint",
    )
    parser.add_argument(
        "--distilled-lora-path",
        type=Path,
        default=Path("~/models/ltx-2.3/ltx-2.3-22b-distilled-lora-384.safetensors").expanduser(),
        help="Distilled LoRA safetensors",
    )
    parser.add_argument(
        "--distilled-lora-strength",
        type=float,
        default=0.5,
        help="Distilled LoRA strength applied when building stage-2 model",
    )
    parser.add_argument(
        "--spatial-upsampler-path",
        type=Path,
        default=Path("~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors").expanduser(),
        help="Spatial upsampler checkpoint",
    )
    parser.add_argument(
        "--gemma-root",
        type=Path,
        default=Path("~/models/gemma-3-12b-it").expanduser(),
        help="Gemma root for prompt encoding assets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output safetensors path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    distilled_lora_cfg: list[LoraPathStrengthAndSDOps] = []
    if args.distilled_lora_strength != 0.0:
        distilled_lora_cfg = [
            LoraPathStrengthAndSDOps(
                path=str(args.distilled_lora_path),
                strength=float(args.distilled_lora_strength),
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ]

    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=str(args.checkpoint_path),
        distilled_lora=distilled_lora_cfg,
        spatial_upsampler_path=str(args.spatial_upsampler_path),
        gemma_root=str(args.gemma_root),
        loras=[],
        quantization=None,
    )

    velocity_model = pipeline.stage_2_model_ledger.transformer().velocity_model
    state_dict = velocity_model.state_dict()

    keep_prefixes = (
        "transformer_blocks.0.attn1.",
        "transformer_blocks.0.attn2.",
        "transformer_blocks.0.audio_attn1.",
        "transformer_blocks.0.audio_attn2.",
        "transformer_blocks.0.audio_to_video_attn.",
        "transformer_blocks.0.video_to_audio_attn.",
    )

    tensors: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not any(key.startswith(prefix) for prefix in keep_prefixes):
            continue
        if not torch.is_tensor(value):
            continue
        tensors[f"velocity_model.{key}"] = value.detach().to("cpu").contiguous()

    if not tensors:
        raise RuntimeError("No block0 attention tensors were collected from stage-2 transformer")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(args.output),
        metadata={
            "source": "TI2VidTwoStagesPipeline stage_2_model_ledger.transformer().velocity_model",
            "distilled_lora_strength": str(args.distilled_lora_strength),
            "tensor_scope": "block0 attention modules (all 6 kinds)",
        },
    )

    print(f"saved: {args.output}")
    print(f"tensors: {len(tensors)}")
    print("sample keys:")
    for idx, key in enumerate(sorted(tensors.keys())):
        if idx >= 12:
            break
        print(f"  {key}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
