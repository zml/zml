"""Export a LoRA-merged stage-2 checkpoint for a contiguous transformer block slice.

The output safetensors is re-indexed to a local 0..N-1 block range so Zig slice
checkers can iterate params linearly without global block offsets.

Example:
  uv run python examples/ltx/export_stage2_block_slice_checkpoint.py \
    --start-block 0 --end-block 7 \
    --distilled-lora-strength 0.5 \
    --output trace_run/stage2_blocks_00_07_lora0.5_merged.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export stage-2 block-slice tensors for native Zig parity checks")
    parser.add_argument("--start-block", type=int, required=True, help="Inclusive global start block index")
    parser.add_argument("--end-block", type=int, required=True, help="Inclusive global end block index")
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
        default=0.0,
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
    parser.add_argument("--output", type=Path, required=True, help="Output safetensors path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_block < 0 or args.end_block < args.start_block:
        raise ValueError(f"Invalid block range: [{args.start_block}, {args.end_block}]")

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
        "attn1.",
        "attn2.",
        "ff.",
        "audio_attn1.",
        "audio_attn2.",
        "audio_ff.",
        "audio_to_video_attn.",
        "video_to_audio_attn.",
    )
    keep_exact = {
        "scale_shift_table",
        "audio_scale_shift_table",
        "scale_shift_table_a2v_ca_video",
        "scale_shift_table_a2v_ca_audio",
        "prompt_scale_shift_table",
        "audio_prompt_scale_shift_table",
    }

    tensors: dict[str, torch.Tensor] = {}
    for global_idx in range(args.start_block, args.end_block + 1):
        global_prefix = f"transformer_blocks.{global_idx}."
        local_idx = global_idx - args.start_block
        local_prefix = f"transformer_blocks.{local_idx}."

        for key, value in state_dict.items():
            if not key.startswith(global_prefix):
                continue
            if not torch.is_tensor(value):
                continue

            suffix = key[len(global_prefix) :]
            if not (any(suffix.startswith(p) for p in keep_prefixes) or suffix in keep_exact):
                continue

            out_key = f"velocity_model.{local_prefix}{suffix}"
            tensors[out_key] = value.detach().to("cpu").contiguous()

    if not tensors:
        raise RuntimeError(
            f"No tensors collected for block range [{args.start_block}, {args.end_block}]"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(args.output),
        metadata={
            "source": "TI2VidTwoStagesPipeline stage_2_model_ledger.transformer().velocity_model",
            "distilled_lora_strength": str(args.distilled_lora_strength),
            "block_range_global": f"{args.start_block}-{args.end_block}",
            "block_count": str(args.end_block - args.start_block + 1),
            "indexing": "reindexed to local transformer_blocks.0..N-1",
        },
    )

    print(f"saved: {args.output}")
    print(f"tensors: {len(tensors)}")
    print(f"range: [{args.start_block}, {args.end_block}] -> local [0, {args.end_block - args.start_block}]")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
