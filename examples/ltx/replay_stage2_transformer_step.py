"""Replay one stage-2 transformer step and capture activations.

This script replays a single recorded step from `trace_run/11_stage2_steps.pt`,
rebuilds stage-2 inputs, runs the transformer forward pass, and saves activations
to a trace file.

Typical usage:

1) Full pass (outputs only):
     uv run ./scripts/replay_stage2_transformer_step.py --pass-label full

2) One block slice with inputs+outputs:
     uv run ./scripts/replay_stage2_transformer_step.py \
         --pass-label b00_07 \
         --capture-inputs \
         --include '^velocity_model\\.transformer_blocks\\.(0|1|2|3|4|5|6|7)(\\.|$)'

3) Another slice:
     uv run ./scripts/replay_stage2_transformer_step.py \
         --pass-label b08_15 \
         --capture-inputs \
         --include '^velocity_model\\.transformer_blocks\\.(8|9|10|11|12|13|14|15)(\\.|$)'

Notes:
- Use multiple `--include` flags to trace multiple regex groups in one pass.
- `--leaf-only` (default) captures only leaf modules and avoids container duplicates.
- `--max-capture-gib` limits in-memory capture and prevents OOM kills.
- Output file format:
    trace_run/acts_stage2_transformer_step_<step>_<pass-label>.pt
"""

import argparse
from pathlib import Path

import torch
import zml_utils

from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_pipelines.utils.helpers import modality_from_latent_state
from ltx_core.types import LatentState
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline


TRACE_DIR = Path("trace_run")


def parse_args() -> argparse.Namespace:
    """Parse CLI options for pass-based activation tracing."""
    parser = argparse.ArgumentParser(description="Replay one stage-2 transformer step and capture activations")
    parser.add_argument("--step-idx", type=int, default=0, help="Index in 11_stage2_steps.pt")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Regex of module names to trace. Can be repeated.",
    )
    parser.add_argument(
        "--pass-label",
        type=str,
        default="full",
        help="Label injected in output filename and metadata.",
    )
    parser.add_argument(
        "--capture-inputs",
        action="store_true",
        help="Also capture module inputs/kwargs in addition to outputs.",
    )
    parser.add_argument(
        "--all-modules",
        action="store_true",
        help="Capture all matched modules, including container modules.",
    )
    parser.add_argument(
        "--max-capture-gib",
        type=float,
        default=2.0,
        help="Capture budget in GiB before collector disables further captures.",
    )
    return parser.parse_args()


def load_pt(name: str):
    """Load a trace tensor/object from TRACE_DIR."""
    return torch.load(TRACE_DIR / name, map_location="cpu", weights_only=False)


def main() -> None:
    """Build pipeline, replay one step, collect activations, and save trace."""
    args = parse_args()
    checkpoint_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser())
    distilled_lora_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled-lora-384.safetensors").expanduser())
    spatial_upsampler_path = str(Path("~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors").expanduser())
    gemma_root = str(Path("~/models/gemma-3-12b-it").expanduser())

    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=[
            LoraPathStrengthAndSDOps(
                path=distilled_lora_path,
                strength=0.8,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ],
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=[],
        quantization=None,
    )

    device = pipeline.device
    dtype = torch.bfloat16

    contexts = load_pt("01_text_contexts.pt")
    stage2_steps = load_pt("11_stage2_steps.pt")

    step_idx = args.step_idx
    step = stage2_steps[step_idx]

    v_context_p = contexts["v_context_p"].to(device=device, dtype=dtype)
    a_context_p = contexts["a_context_p"].to(device=device, dtype=dtype)

    sigma = step["sigma"].to(device=device, dtype=torch.float32)
    video_state = LatentState(
        latent=step["video_latent"].to(device=device, dtype=dtype),
        denoise_mask=step["video_denoise_mask"].to(device=device),
        positions=step["video_positions"].to(device=device),
        clean_latent=step["video_clean_latent"].to(device=device, dtype=dtype),
    )

    audio_state = LatentState(
        latent=step["audio_latent"].to(device=device, dtype=dtype),
        denoise_mask=step["audio_denoise_mask"].to(device=device),
        positions=step["audio_positions"].to(device=device),
        clean_latent=step["audio_clean_latent"].to(device=device, dtype=dtype),
    )

    pos_video = modality_from_latent_state(video_state, v_context_p, sigma)
    pos_audio = modality_from_latent_state(audio_state, a_context_p, sigma)

    transformer = pipeline.stage_2_model_ledger.transformer()

    # For debug
    # seen = {}

    # def dbg_hook(module, inputs, output):
    #     seen["called"] = True
    #     print("HOOK CALLED:", type(module))
    #     if isinstance(output, torch.Tensor):
    #         print("output shape:", output.shape)

    # handle = transformer.velocity_model.patchify_proj.register_forward_hook(dbg_hook)

    # denoised_video, denoised_audio = transformer(
    #     video=pos_video,
    #     audio=pos_audio,
    #     perturbations=None,
    # )

    # handle.remove()
    # print("manual hook fired?", seen.get("called", False))

    # -- End debug --

    inner = transformer.velocity_model
    print("inner type:", type(inner))
    print("inner named_modules:", len(list(inner.named_modules())))

    include_regexes = args.include
    if include_regexes:
        print("tracing with include regexes:", include_regexes)
    leaf_modules_only = not args.all_modules
    print("leaf_modules_only:", leaf_modules_only)
    max_capture_bytes = int(args.max_capture_gib * 1024**3)

    collector = zml_utils.ActivationCollector(
        transformer,
        stop_after_first_step=True,
        max_layers=5000,
        include_regexes=include_regexes,
        leaf_modules_only=leaf_modules_only,
        capture_inputs=args.capture_inputs,
        max_capture_bytes=max_capture_bytes,
    )

    print("Starting transformer forward + activation collection...")
    (denoised_video, denoised_audio), activations = collector(
        video=pos_video,
        audio=pos_audio,
        perturbations=None,
    )

    print("denoised_video shape:", denoised_video.shape)
    print("denoised_audio shape:", denoised_audio.shape)
    print("activation entries:", len(activations))

    output_path = TRACE_DIR / f"acts_stage2_transformer_step_{step_idx:03d}_{args.pass_label}.pt"
    torch.save(
        {
            "step_idx": step_idx,
            "pass_label": args.pass_label,
            "include_regexes": include_regexes,
            "leaf_modules_only": leaf_modules_only,
            "capture_inputs": args.capture_inputs,
            "max_capture_gib": args.max_capture_gib,
            "denoised_video": denoised_video.detach().cpu(),
            "denoised_audio": denoised_audio.detach().cpu(),
            "activations": activations,
        },
        output_path,
    )

    print("saved:", output_path)
    print("activation entries:", len(activations))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
