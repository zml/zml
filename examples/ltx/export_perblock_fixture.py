"""Export per-block intermediate outputs for error accumulation analysis.

Runs the velocity model block-by-block and saves vx/ax after selected blocks.
This lets us compare Zig vs Python block-by-block to see where error accumulates.

Saves:
  block_NNN.vx    — bf16 [B, T_v, 4096]  video hidden after block N
  block_NNN.ax    — bf16 [B, T_a, 2048]  audio hidden after block N

Usage:
  cd /root/repos/LTX-2
  uv run python export_perblock_fixture.py --step-idx 0 --token-limit 512
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.types import LatentState
from ltx_pipelines.utils.helpers import modality_from_latent_state
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline


TRACE_DIR = Path("trace_run")

# Blocks to capture outputs from (0-indexed)
# Dense sampling 23-47 to pinpoint error accumulation in later blocks
CAPTURE_BLOCKS = [0, 1, 2, 3, 7, 15] + list(range(23, 48))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-block intermediate outputs")
    parser.add_argument("--step-idx", type=int, default=0)
    parser.add_argument("--token-limit", type=int, default=512)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--distilled-lora-strength", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser())
    distilled_lora_path = str(
        Path("~/models/ltx-2.3/ltx-2.3-distilled-lora-v2.safetensors").expanduser()
    )
    spatial_upsampler_path = str(
        Path("~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors").expanduser()
    )
    gemma_root = str(Path("~/models/gemma-3-12b-it").expanduser())

    lora_cfg = []
    if args.distilled_lora_strength != 0.0:
        lora_cfg = [
            LoraPathStrengthAndSDOps(
                path=distilled_lora_path,
                strength=args.distilled_lora_strength,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ]

    print(f"Loading pipeline (lora_strength={args.distilled_lora_strength})...")
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=lora_cfg,
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=[],
        quantization=None,
    )

    device = pipeline.device
    dtype = torch.bfloat16

    print("Loading saved replay tensors...")
    contexts = torch.load(TRACE_DIR / "01_text_contexts.pt", map_location="cpu", weights_only=False)
    stage2_steps = torch.load(TRACE_DIR / "11_stage2_steps.pt", map_location="cpu", weights_only=False)
    step = stage2_steps[args.step_idx]

    sigma = step["sigma"].to(device=device, dtype=torch.float32)
    v_context_p = contexts["v_context_p"].to(device=device, dtype=dtype)
    a_context_p = contexts["a_context_p"].to(device=device, dtype=dtype)

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

    # Apply token limit (same as export_step2_fixture.py)
    if args.token_limit is not None:
        tl = args.token_limit
        v_context_p = v_context_p[:, :tl, ...].contiguous()
        a_context_p = a_context_p[:, :tl, ...].contiguous()
        video_state = LatentState(
            latent=video_state.latent[:, :tl, ...].contiguous(),
            denoise_mask=video_state.denoise_mask[:, :tl, ...].contiguous(),
            positions=video_state.positions[:, :, :tl, ...].contiguous() if video_state.positions.dim() > 2 else video_state.positions[:, :tl, ...].contiguous(),
            clean_latent=video_state.clean_latent[:, :tl, ...].contiguous(),
        )
        audio_state = LatentState(
            latent=audio_state.latent[:, :tl, ...].contiguous(),
            denoise_mask=audio_state.denoise_mask[:, :tl, ...].contiguous(),
            positions=audio_state.positions[:, :, :tl, ...].contiguous() if audio_state.positions.dim() > 2 else audio_state.positions[:, :tl, ...].contiguous(),
            clean_latent=audio_state.clean_latent[:, :tl, ...].contiguous(),
        )
        print(f"  token-limited: token_limit={tl}")

    pos_video = modality_from_latent_state(video_state, v_context_p, sigma)
    pos_audio = modality_from_latent_state(audio_state, a_context_p, sigma)

    transformer = pipeline.stage_2_model_ledger.transformer()
    vm = transformer.velocity_model

    # ---- Prepare TransformerArgs ----
    print("\nRunning args preprocessors...")
    with torch.inference_mode():
        video_args = vm.video_args_preprocessor.prepare(pos_video, pos_audio)
        audio_args = vm.audio_args_preprocessor.prepare(pos_audio, pos_video)

    print(f"  vx: {list(video_args.x.shape)}  dtype={video_args.x.dtype}")
    print(f"  ax: {list(audio_args.x.shape)}  dtype={audio_args.x.dtype}")
    print(f"  video_timesteps: {list(video_args.timesteps.shape)}  dtype={video_args.timesteps.dtype}")
    print(f"  audio_timesteps: {list(audio_args.timesteps.shape)}  dtype={audio_args.timesteps.dtype}")

    # ---- Print timestep/embedded_ts dtype diagnostics ----
    print(f"\n  embedded_timestep dtype: video={video_args.embedded_timestep.dtype}  audio={audio_args.embedded_timestep.dtype}")
    if video_args.prompt_timestep is not None:
        print(f"  prompt_timestep dtype: video={video_args.prompt_timestep.dtype}  audio={audio_args.prompt_timestep.dtype}")
    if video_args.cross_scale_shift_timestep is not None:
        print(f"  cross_ss_ts dtype: video={video_args.cross_scale_shift_timestep.dtype}  audio={audio_args.cross_scale_shift_timestep.dtype}")
    if video_args.cross_gate_timestep is not None:
        print(f"  cross_gate_ts dtype: video={video_args.cross_gate_timestep.dtype}  audio={audio_args.cross_gate_timestep.dtype}")

    # ---- Run block-by-block ----
    print(f"\nRunning {len(vm.transformer_blocks)} blocks...")
    captured = {}

    # Capture pre-block state
    captured["block_input.vx"] = video_args.x.detach().cpu().contiguous()
    captured["block_input.ax"] = audio_args.x.detach().cpu().contiguous()

    with torch.inference_mode():
        for i, block in enumerate(vm.transformer_blocks):
            video_args, audio_args = block(
                video=video_args,
                audio=audio_args,
                perturbations=None,
            )

            if i in CAPTURE_BLOCKS:
                captured[f"block_{i:03d}.vx"] = video_args.x.detach().cpu().contiguous()
                captured[f"block_{i:03d}.ax"] = audio_args.x.detach().cpu().contiguous()
                print(f"  block {i:3d}: vx {list(video_args.x.shape)} dtype={video_args.x.dtype}  "
                      f"ax {list(audio_args.x.shape)} dtype={audio_args.x.dtype}")

    # ---- Also capture final velocity output via vm() for sanity ----
    # Skip manual _process_output — the step2 fixture already has velocity refs.
    # Just save the final hidden states (block 47 already captured above).

    # ---- Summary ----
    print(f"\nTotal: {len(captured)} tensors")
    for key in sorted(captured.keys()):
        t = captured[key]
        print(f"  {key:40s}  shape={list(t.shape)}  dtype={t.dtype}")

    # ---- Save ----
    token_suffix = f"_t{args.token_limit}" if args.token_limit is not None else ""
    default_out = TRACE_DIR / f"perblock_fixture_step_{args.step_idx:03d}{token_suffix}.safetensors"
    out_path = args.output if args.output is not None else default_out
    TRACE_DIR.mkdir(exist_ok=True)
    save_file(captured, str(out_path))
    print(f"\nSaved: {out_path}  ({len(captured)} tensors)")


if __name__ == "__main__":
    main()
