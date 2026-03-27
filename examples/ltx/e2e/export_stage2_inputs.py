"""Export Stage 2 denoising inputs for Zig e2e demo.

Runs the full TI2VidTwoStagesPipeline through Stage 1 + upsample + Stage 2
noise initialization, then exports the patchified noised LatentState tensors
and text contexts to a safetensors file. The Zig denoiser consumes this file,
runs the 3-step Euler denoising loop, and writes the denoised patchified
latents to an output file for decode_latents.py to finish.

Usage (on GPU server):
  cd /root/repos/LTX-2
  uv run examples/ltx/e2e/export_stage2_inputs.py \
      --prompt "A cat playing piano" \
      --output /root/e2e_demo/stage2_inputs.safetensors

Saved keys:
  video_latent         bf16  [B, T_v, 128]   patchified noised video latent
  audio_latent         bf16  [B, T_a, 128]   patchified noised audio latent
  video_denoise_mask   f32   [B, T_v, 1]     per-token denoising strength
  audio_denoise_mask   f32   [B, T_a, 1]
  video_clean_latent   bf16  [B, T_v, 128]   conditioning reference
  audio_clean_latent   bf16  [B, T_a, 128]
  video_noise          bf16  [B, T_v, 128]   back-computed noise tensor
  audio_noise          bf16  [B, T_a, 128]
  video_positions      bf16  [B, 3, T_v, 2]  position coordinates
  audio_positions      f32   [B, 1, T_a, 2]
  v_context            bf16  [B, S, 4096]    text context (video)
  a_context            bf16  [B, S, 2048]    text context (audio)

Metadata (in safetensors header):
  num_frames, height, width, fps, video_latent_tokens, audio_latent_tokens,
  seed, prompt
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from safetensors.torch import save_file

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.upsampler import upsample_video
from ltx_core.types import LatentState, VideoPixelShape
from ltx_pipelines.utils import (
    ModelLedger,
    cleanup_memory,
    combined_image_conditionings,
    denoise_audio_video,
    encode_prompts,
    euler_denoising_loop,
    get_device,
    multi_modal_guider_factory_denoising_func,
)
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.helpers import noise_video_state, noise_audio_state
from ltx_pipelines.utils.types import PipelineComponents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage 2 inputs for Zig denoiser")
    parser.add_argument("--prompt", type=str, default="A cat playing piano in a sunny room")
    parser.add_argument("--negative-prompt", type=str, default="worst quality, blurry")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--frame-rate", type=float, default=25.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    # Model paths
    parser.add_argument("--checkpoint", type=str,
                        default=str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser()))
    parser.add_argument("--spatial-upsampler", type=str,
                        default=str(Path("~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors").expanduser()))
    parser.add_argument("--gemma-root", type=str,
                        default=str(Path("~/models/gemma-3-12b-it").expanduser()))
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()

    device = get_device()
    dtype = torch.bfloat16

    print(f"Prompt: {args.prompt!r}")
    print(f"Resolution: {args.width}x{args.height}, {args.num_frames} frames @ {args.frame_rate} fps")
    print(f"Seed: {args.seed}")

    # ========================================================================
    # Load pipeline (reuse TI2VidTwoStagesPipeline's model loading)
    # ========================================================================
    print("Loading models...")
    stage_1_ledger = ModelLedger(
        dtype=dtype,
        device=device,
        checkpoint_path=args.checkpoint,
        gemma_root_path=args.gemma_root,
        spatial_upsampler_path=args.spatial_upsampler,
        loras=[],
        quantization=None,
    )
    # Stage 2 uses the same weights (distilled LoRA would be applied here in
    # production, but for the e2e demo we skip it — the Zig denoiser uses the
    # base distilled checkpoint which already has LoRA merged).
    stage_2_ledger = stage_1_ledger

    pipeline_components = PipelineComponents(dtype=dtype, device=device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    noiser = GaussianNoiser(generator=generator)
    stepper = EulerDiffusionStep()

    # ========================================================================
    # Stage 0: Text encoding
    # ========================================================================
    print("Encoding text prompts...")
    ctx_p, ctx_n = encode_prompts(
        [args.prompt, args.negative_prompt],
        stage_1_ledger,
    )
    v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
    v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

    # ========================================================================
    # Stage 1: Half-resolution denoising with CFG
    # ========================================================================
    print("Running Stage 1 (half-resolution denoising)...")
    stage_1_output_shape = VideoPixelShape(
        batch=1,
        frames=args.num_frames,
        width=args.width // 2,
        height=args.height // 2,
        fps=args.frame_rate,
    )

    video_encoder = stage_1_ledger.video_encoder()
    stage_1_conditionings = combined_image_conditionings(
        images=[],
        height=stage_1_output_shape.height,
        width=stage_1_output_shape.width,
        video_encoder=video_encoder,
        dtype=dtype,
        device=device,
    )
    torch.cuda.synchronize()
    del video_encoder
    cleanup_memory()

    transformer = stage_1_ledger.transformer()
    from ltx_core.components.schedulers import LTX2Scheduler
    sigmas = LTX2Scheduler().execute(steps=args.num_inference_steps).to(dtype=torch.float32, device=device)

    video_guider_params = MultiModalGuiderParams(cfg_scale=3.0)
    audio_guider_params = MultiModalGuiderParams(cfg_scale=3.0)

    def first_stage_denoising_loop(sigmas, video_state, audio_state, stepper):
        return euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            denoise_fn=multi_modal_guider_factory_denoising_func(
                video_guider_factory=create_multimodal_guider_factory(
                    params=video_guider_params,
                    negative_context=v_context_n,
                ),
                audio_guider_factory=create_multimodal_guider_factory(
                    params=audio_guider_params,
                    negative_context=a_context_n,
                ),
                v_context=v_context_p,
                a_context=a_context_p,
                transformer=transformer,
            ),
        )

    video_state, audio_state = denoise_audio_video(
        output_shape=stage_1_output_shape,
        conditionings=stage_1_conditionings,
        noiser=noiser,
        sigmas=sigmas,
        stepper=stepper,
        denoising_loop_fn=first_stage_denoising_loop,
        components=pipeline_components,
        dtype=dtype,
        device=device,
    )

    torch.cuda.synchronize()
    del transformer
    cleanup_memory()
    print(f"  Stage 1 done. video_latent: {list(video_state.latent.shape)}")

    # ========================================================================
    # Upsample video latent 2x
    # ========================================================================
    print("Upsampling video latent 2x...")
    video_encoder = stage_1_ledger.video_encoder()
    upscaled_video_latent = upsample_video(
        latent=video_state.latent[:1],
        video_encoder=video_encoder,
        upsampler=stage_2_ledger.spatial_upsampler(),
    )
    print(f"  Upscaled: {list(upscaled_video_latent.shape)}")

    # ========================================================================
    # Stage 2: Prepare noise init inputs
    # ========================================================================
    stage_2_output_shape = VideoPixelShape(
        batch=1,
        frames=args.num_frames,
        width=args.width,
        height=args.height,
        fps=args.frame_rate,
    )
    stage_2_conditionings = combined_image_conditionings(
        images=[],
        height=stage_2_output_shape.height,
        width=stage_2_output_shape.width,
        video_encoder=video_encoder,
        dtype=dtype,
        device=device,
    )
    del video_encoder
    torch.cuda.synchronize()
    cleanup_memory()

    distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(device)

    # ========================================================================
    # Stage 2: Noise initialization (replicate denoise_audio_video setup)
    # ========================================================================
    print("Initializing Stage 2 noised latent states...")
    video_state_s2, video_tools = noise_video_state(
        output_shape=stage_2_output_shape,
        noiser=noiser,
        conditionings=stage_2_conditionings,
        components=pipeline_components,
        dtype=dtype,
        device=device,
        noise_scale=distilled_sigmas[0].item(),
        initial_latent=upscaled_video_latent,
    )
    audio_state_s2, audio_tools = noise_audio_state(
        output_shape=stage_2_output_shape,
        noiser=noiser,
        conditionings=[],
        components=pipeline_components,
        dtype=dtype,
        device=device,
        noise_scale=distilled_sigmas[0].item(),
        initial_latent=audio_state.latent,
    )

    print(f"  video_latent: {list(video_state_s2.latent.shape)}  dtype={video_state_s2.latent.dtype}")
    print(f"  audio_latent: {list(audio_state_s2.latent.shape)}  dtype={audio_state_s2.latent.dtype}")
    print(f"  video_denoise_mask: {list(video_state_s2.denoise_mask.shape)}")
    print(f"  audio_denoise_mask: {list(audio_state_s2.denoise_mask.shape)}")
    print(f"  video_positions: {list(video_state_s2.positions.shape)}")
    print(f"  audio_positions: {list(audio_state_s2.positions.shape)}")
    print(f"  v_context: {list(v_context_p.shape)}")
    print(f"  a_context: {list(a_context_p.shape)}")

    # ========================================================================
    # Back-compute noise tensors for Zig noise init validation
    # ========================================================================
    sigma_0 = distilled_sigmas[0].item()
    print(f"  sigma_0 = {sigma_0}")

    def recover_noise(noised, clean, mask, sigma_0):
        """Recover noise tensor: noise = (noised - clean * (1 - mask*s)) / (mask*s)."""
        noised_f32 = noised.float()
        clean_f32 = clean.float()
        mask_f32 = mask.float()
        mask_sigma = mask_f32 * sigma_0
        one_minus = 1.0 - mask_sigma
        noise_f32 = torch.zeros_like(noised_f32)
        nonzero = mask_sigma.squeeze(-1) > 0
        if nonzero.any():
            noise_f32[nonzero] = (noised_f32[nonzero] - clean_f32[nonzero] * one_minus[nonzero]) / mask_sigma[nonzero]
        return noise_f32.to(noised.dtype)

    video_noise = recover_noise(
        video_state_s2.latent, video_state_s2.clean_latent,
        video_state_s2.denoise_mask, sigma_0,
    )
    audio_noise = recover_noise(
        audio_state_s2.latent, audio_state_s2.clean_latent,
        audio_state_s2.denoise_mask, sigma_0,
    )
    print(f"  video_noise:  {list(video_noise.shape)} {video_noise.dtype}")
    print(f"  audio_noise:  {list(audio_noise.shape)} {audio_noise.dtype}")

    # ========================================================================
    # Export to safetensors
    # ========================================================================
    print("Exporting tensors...")
    tensors = {
        "video_latent": video_state_s2.latent.detach().cpu().contiguous(),
        "audio_latent": audio_state_s2.latent.detach().cpu().contiguous(),
        "video_denoise_mask": video_state_s2.denoise_mask.detach().cpu().contiguous(),
        "audio_denoise_mask": audio_state_s2.denoise_mask.detach().cpu().contiguous(),
        "video_clean_latent": video_state_s2.clean_latent.detach().cpu().contiguous(),
        "audio_clean_latent": audio_state_s2.clean_latent.detach().cpu().contiguous(),
        "video_positions": video_state_s2.positions.detach().cpu().contiguous(),
        "audio_positions": audio_state_s2.positions.detach().cpu().contiguous(),
        "v_context": v_context_p.detach().cpu().contiguous(),
        "a_context": a_context_p.detach().cpu().contiguous(),
        "video_noise": video_noise.detach().cpu().contiguous(),
        "audio_noise": audio_noise.detach().cpu().contiguous(),
    }

    # Compute unpatchified shapes for the decode script
    # Video: patchify is (B, 128, F, H, W) → (B, F*H*W, 128) with patch_size=1
    # so T_v = F_lat * H_lat * W_lat
    # Audio: patchify is (B, 8, T_aud, 16) → (B, T_aud, 128) with c=8, f=16
    #
    # From VideoLatentShape.from_pixel_shape:
    #   F_lat = (num_frames - 1) // 8 + 1
    #   H_lat = height // 32
    #   W_lat = width // 32
    f_lat = (args.num_frames - 1) // 8 + 1
    h_lat = args.height // 32
    w_lat = args.width // 32
    t_aud = audio_state_s2.latent.shape[1]

    metadata = {
        "num_frames": str(args.num_frames),
        "height": str(args.height),
        "width": str(args.width),
        "fps": str(args.frame_rate),
        "seed": str(args.seed),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "sigma_0": str(sigma_0),
        "f_lat": str(f_lat),
        "h_lat": str(h_lat),
        "w_lat": str(w_lat),
        "t_aud": str(t_aud),
        "video_latent_tokens": str(video_state_s2.latent.shape[1]),
        "audio_latent_tokens": str(audio_state_s2.latent.shape[1]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata=metadata)

    print(f"\nSaved: {args.output}")
    print(f"  {len(tensors)} tensors, metadata: {json.dumps(metadata, indent=2)}")
    for key, t in sorted(tensors.items()):
        print(f"  {key:30s}  shape={list(t.shape)}  dtype={t.dtype}")


if __name__ == "__main__":
    main()
