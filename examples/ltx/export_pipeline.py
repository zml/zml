#!/usr/bin/env python3
"""Export pipeline: run the full two-stage LTX-2.3 pipeline in Python and
capture all intermediate states for validation against the Zig inference binary.

Supports both text-to-video (default) and image-conditioned generation
(pass --image).  When --image is provided, the VAE encoder is also run and
activations are captured for validation.

The Zig inference binary only needs:
  - Gemma hidden states (pos + neg)
  - pipeline_meta.json (latent geometry)
All other outputs are reference data for debugging/validation.

Outputs (always):
  {out}/pos_hidden_states.safetensors       — Gemma hidden states (positive prompt) [used by Zig]
  {out}/neg_hidden_states.safetensors       — Gemma hidden states (negative prompt) [used by Zig]
  {out}/pipeline_meta.json               — Pipeline config metadata [used by Zig]
  {out}/stage2_noise.safetensors                — Pre-drawn Stage 2 noise (reference only)
  {out}/ref/stage1_outputs.safetensors           — Stage 1 denoised latents (reference only)
  {out}/ref/upsampled.safetensors                — Upscaled video latent (reference only)
  {out}/ref/stage2_outputs.safetensors           — Stage 2 denoised latents (reference only)
  {out}/python_reference.mp4             — Full-Python video (with --decode-video)

Usage (on GPU server):
  cd /root/repos/LTX-2
  uv run examples/ltx/export_pipeline.py \
      --image /path/to/reference_image.jpg \
      --output-dir /root/imgcond_ref/ \
      --prompt "A beautiful sunset over the ocean" \
      --seed 10 \
      --checkpoint ~/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
      --stage2-checkpoint ~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors
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
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.transformer import X0Model
from ltx_core.model.video_vae import TilingConfig, VideoEncoder, get_video_chunks_number
from ltx_core.types import LatentState, VideoPixelShape
from ltx_core.loader import DummyRegistry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.text_encoders.gemma.encoders.encoder_configurator import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    GEMMA_LLM_KEY_OPS,
    GEMMA_MODEL_OPS,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoderConfigurator,
)
from ltx_pipelines.utils import (
    AudioDecoder,
    DiffusionStage,
    FactoryGuidedDenoiser,
    ImageConditioner,
    ModalitySpec,
    SimpleDenoiser,
    VideoDecoder,
    VideoUpsampler,
    cleanup_memory,
    combined_image_conditionings,
    euler_denoising_loop,
    get_device,
)
from ltx_pipelines.utils.blocks import gpu_model, module_ops_from_gemma_root
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.media_io import (
    encode_video,
    load_image_and_preprocess,
)
from ltx_pipelines.utils.types import Denoiser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export reference activations for image-conditioned pipeline"
    )
    # Image conditioning
    parser.add_argument("--image", type=str, default=None, help="Path to conditioning image (omit for unconditioned generation)")
    parser.add_argument("--strength", type=float, default=1.0, help="Conditioning strength (1.0 = full)")
    # Generation params
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over the ocean")
    parser.add_argument("--negative-prompt", type=str, default="blurry, out of focus")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    # Output
    parser.add_argument("--output-dir", type=Path, required=True)
    # Model paths
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(Path("~/models/ltx-2.3/ltx-2.3-22b-dev.safetensors").expanduser()),
        help="Base model checkpoint",
    )
    parser.add_argument(
        "--stage2-checkpoint", type=str,
        default=str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser()),
        help="Distilled checkpoint for Stage 2",
    )
    parser.add_argument(
        "--spatial-upsampler", type=str,
        default=str(Path("~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors").expanduser()),
    )
    parser.add_argument(
        "--gemma-root", type=str,
        default=str(Path("~/models/gemma-3-12b-it").expanduser()),
    )
    # Decode options
    parser.add_argument(
        "--decode-video", action="store_true",
        help="Decode final latents to MP4 (full-Python reference video)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_safetensors(tensors: dict[str, torch.Tensor], path: Path,
                     metadata: dict[str, str] | None = None) -> None:
    """Save tensors to safetensors file, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
    save_file(clean, str(path), metadata=metadata)
    print(f"  Saved {path.name}: {len(clean)} tensors")
    for k, v in sorted(clean.items()):
        print(f"    {k:40s}  {list(v.shape)}  {v.dtype}")


def recover_noise(noised: torch.Tensor, clean: torch.Tensor,
                  mask: torch.Tensor, sigma_0: float) -> torch.Tensor:
    """Back-compute noise: noise = (noised - clean * (1 - mask*s)) / (mask*s)."""
    noised_f32 = noised.float()
    clean_f32 = clean.float()
    mask_f32 = mask.float()
    mask_sigma = mask_f32 * sigma_0
    one_minus = 1.0 - mask_sigma
    noise_f32 = torch.zeros_like(noised_f32)
    nonzero = mask_sigma.squeeze(-1) > 0
    if nonzero.any():
        noise_f32[nonzero] = (
            (noised_f32[nonzero] - clean_f32[nonzero] * one_minus[nonzero])
            / mask_sigma[nonzero]
        )
    return noise_f32.to(noised.dtype)


# ---------------------------------------------------------------------------
# Instrumented encoder: captures activations at every boundary
# ---------------------------------------------------------------------------

def encode_with_activations(
    encoder: VideoEncoder,
    image: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run encoder forward pass, capturing intermediate activations.

    Args:
        encoder: The VideoEncoder model instance.
        image: Preprocessed image tensor [1, 3, 1, H, W] in [-1, 1], bf16.

    Returns:
        (encoded_normalized, activations_dict)
        where activations_dict has keys like "after_patchify", "after_conv_in", etc.
    """
    from ltx_core.model.video_vae.ops import patchify

    acts = {}

    # Step 1: Patchify (pixel-space → channel-space)
    x = patchify(image, patch_size_hw=encoder.patch_size, patch_size_t=1)
    acts["after_patchify"] = x.clone()

    # Step 2: conv_in
    x = encoder.conv_in(x)
    acts["after_conv_in"] = x.clone()

    # Step 3: Down blocks
    for i, down_block in enumerate(encoder.down_blocks):
        x = down_block(x)
        acts[f"after_down_{i}"] = x.clone()

    # Step 4: conv_norm_out + conv_act
    x = encoder.conv_norm_out(x)
    x = encoder.conv_act(x)
    acts["after_norm_silu"] = x.clone()

    # Step 5: conv_out
    x = encoder.conv_out(x)
    acts["after_conv_out"] = x.clone()

    # Step 6: Extract means (UNIFORM log-var mode: last channel is log-var)
    means = x[:, :-1, ...]
    acts["encoded_means"] = means.clone()

    # Step 7: Normalize
    normalized = encoder.per_channel_statistics.normalize(means)
    acts["encoded_normalized"] = normalized.clone()

    return normalized, acts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()

    device = get_device()
    dtype = torch.bfloat16
    out = args.output_dir
    ref = out / "ref"

    has_image = args.image is not None
    print(f"Image:  {args.image if has_image else '(none — unconditioned)'}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Resolution: {args.width}x{args.height}, {args.num_frames} frames @ {args.frame_rate} fps")
    print(f"Seed: {args.seed}")
    if has_image:
        print(f"Strength: {args.strength}")
    print(f"Output: {out}")

    # Stage 1 resolution (half)
    s1_h, s1_w = args.height // 2, args.width // 2

    image_conditioner = None
    if has_image:
        # ====================================================================
        # Image preprocessing
        # ====================================================================
        print("\n=== Image preprocessing ===")

        image_s1 = load_image_and_preprocess(
            image_path=args.image,
            height=s1_h, width=s1_w,
            dtype=dtype, device=device,
        )
        print(f"  Stage 1 image (half-res): {list(image_s1.shape)} {image_s1.dtype}")
        print(f"    range: [{image_s1.min().item():.3f}, {image_s1.max().item():.3f}]")

        image_s2 = load_image_and_preprocess(
            image_path=args.image,
            height=args.height, width=args.width,
            dtype=dtype, device=device,
        )
        print(f"  Stage 2 image (full-res): {list(image_s2.shape)} {image_s2.dtype}")
        print(f"    range: [{image_s2.min().item():.3f}, {image_s2.max().item():.3f}]")

        save_safetensors(
            {
                "image_s1": image_s1,
                "image_s2": image_s2,
            },
            out / "image_preprocessed.safetensors",
        )

        # ====================================================================
        # VAE Encoder: capture activations at both resolutions
        # ====================================================================
        print("\n=== VAE Encoder activations ===")

        image_conditioner = ImageConditioner(args.checkpoint, dtype, device)

        def capture_encoder_activations(encoder: VideoEncoder):
            """Run encoder on both resolutions, capturing all activations."""
            print("  Encoding Stage 1 image (half-res)...")
            encoded_s1, acts_s1 = encode_with_activations(encoder, image_s1)
            print(f"    encoded: {list(encoded_s1.shape)}")

            print("  Encoding Stage 2 image (full-res)...")
            encoded_s2, acts_s2 = encode_with_activations(encoder, image_s2)
            print(f"    encoded: {list(encoded_s2.shape)}")

            return encoded_s1, acts_s1, encoded_s2, acts_s2

        encoded_s1, acts_s1, encoded_s2, acts_s2 = image_conditioner(
            capture_encoder_activations
        )

        # Save Stage 1 activations (half-res — primary validation target)
        s1_act_tensors = {f"s1/{k}": v for k, v in acts_s1.items()}
        s2_act_tensors = {f"s2/{k}": v for k, v in acts_s2.items()}
        save_safetensors(
            {**s1_act_tensors, **s2_act_tensors},
            out / "encoder_activations.safetensors",
        )

    # ========================================================================
    # Model setup
    # ========================================================================
    print("\n=== Loading models ===")

    upsampler = VideoUpsampler(args.checkpoint, args.spatial_upsampler, dtype, device)

    stage_1_diffusion = DiffusionStage(args.checkpoint, dtype, device)
    stage_2_diffusion = DiffusionStage(args.stage2_checkpoint, dtype, device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    noiser = GaussianNoiser(generator=generator)

    # ========================================================================
    # Text encoding: Gemma forward pass → hidden states → EmbeddingsProcessor
    # ========================================================================
    print("\n=== Text encoding: Gemma forward pass ===")

    def find_matching_file(root: str, pattern: str) -> Path:
        root_path = Path(root)
        matches = list(root_path.rglob(pattern))
        if not matches:
            raise FileNotFoundError(f"No file matching {pattern!r} under {root}")
        return matches[0]

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

    raw_outputs = []
    with gpu_model(text_encoder_builder.build(device=device, dtype=dtype).eval()) as text_encoder:
        for i, prompt in enumerate([args.prompt, args.negative_prompt]):
            hidden_states, attention_mask = text_encoder.encode(prompt)
            raw_outputs.append((hidden_states, attention_mask))
            label = "pos" if i == 0 else "neg"
            print(f"  {label}: {len(hidden_states)} layers, "
                  f"shape={list(hidden_states[0].shape)}, "
                  f"mask sum={attention_mask.sum().item()}/{attention_mask.shape[-1]}")

    # Save hidden states as sidecar files
    print("\n=== Saving Gemma hidden states ===")
    for i, (hidden_states, attention_mask) in enumerate(raw_outputs):
        label = "pos" if i == 0 else "neg"
        stacked = torch.stack(list(hidden_states), dim=-1)  # [1, S, 3840, 49]
        save_safetensors(
            {"stacked_hidden_states": stacked.to(dtype), "attention_mask": attention_mask},
            out / f"{label}_hidden_states.safetensors",
        )

    # Run EmbeddingsProcessor to get final contexts (needed internally for denoising)
    print("\n=== Text encoding: EmbeddingsProcessor ===")

    embeddings_processor_builder = Builder(
        model_path=args.checkpoint,
        model_class_configurator=EmbeddingsProcessorConfigurator,
        model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
        registry=DummyRegistry(),
    )

    with gpu_model(
        embeddings_processor_builder.build(device=device, dtype=dtype).to(device).eval()
    ) as embeddings_processor:
        contexts = []
        for i, (hidden_states, attention_mask) in enumerate(raw_outputs):
            label = "pos" if i == 0 else "neg"
            result = embeddings_processor.process_hidden_states(
                hidden_states, attention_mask, padding_side="left"
            )
            contexts.append(result)
            print(f"  {label}: video={list(result.video_encoding.shape)}, "
                  f"audio={list(result.audio_encoding.shape)}")

    v_context_p, a_context_p = contexts[0].video_encoding, contexts[0].audio_encoding
    v_context_n, a_context_n = contexts[1].video_encoding, contexts[1].audio_encoding

    print(f"  v_context_pos: {list(v_context_p.shape)} {v_context_p.dtype}")
    print(f"  a_context_pos: {list(a_context_p.shape)} {a_context_p.dtype}")

    # ========================================================================
    # Stage 1: image-conditioned half-resolution denoising
    # ========================================================================
    print("\n=== Stage 1: image-conditioned half-resolution denoising ===")

    if has_image:
        images_input = [ImageConditioningInput(
            path=args.image, strength=args.strength, frame_idx=0,
        )]

        stage_1_conditionings = image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images_input,
                height=args.height // 2,
                width=args.width // 2,
                video_encoder=enc,
                dtype=dtype,
                device=device,
            )
        )
        print(f"  Stage 1 conditionings: {len(stage_1_conditionings)} item(s)")
    else:
        stage_1_conditionings = []
        print(f"  Stage 1 conditionings: none (unconditioned)")

    sigmas = LTX2Scheduler().execute(steps=args.num_inference_steps).to(
        dtype=torch.float32, device=device,
    )

    # LTX-2.3 guidance params
    video_guider_params = MultiModalGuiderParams(
        cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, skip_step=0, stg_blocks=[28],
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, skip_step=0, stg_blocks=[28],
    )

    # Capture initial states
    captured_s1 = {}

    def stage1_loop(
        sigmas: torch.Tensor,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        stepper: DiffusionStepProtocol,
        transformer: X0Model,
        denoiser: Denoiser,
    ) -> tuple[LatentState | None, LatentState | None]:
        captured_s1["video"] = video_state.clone()
        captured_s1["audio"] = audio_state.clone()
        print(f"  [capture] Stage 1 conditioned initial states:")
        print(f"    video_latent:        {list(video_state.latent.shape)} {video_state.latent.dtype}")
        print(f"    video_denoise_mask:  {list(video_state.denoise_mask.shape)}")
        print(f"    video_clean_latent:  {list(video_state.clean_latent.shape)}")

        if has_image:
            # Check conditioning was applied: first-frame tokens should have mask != 1.0
            s1_h_lat = s1_h // 32
            s1_w_lat = s1_w // 32
            n_img_tokens = s1_h_lat * s1_w_lat
            mask_first = video_state.denoise_mask[0, :n_img_tokens]
            mask_rest = video_state.denoise_mask[0, n_img_tokens:]
            print(f"    conditioning check: n_img_tokens={n_img_tokens}")
            print(f"      mask[:n_img] mean={mask_first.mean().item():.4f}  (expect ~{1.0 - args.strength:.1f})")
            print(f"      mask[n_img:] mean={mask_rest.mean().item():.4f}  (expect ~1.0)")
        else:
            print(f"    (unconditioned — all mask values should be 1.0)")
            print(f"      mask mean={video_state.denoise_mask.mean().item():.4f}")

        return euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            transformer=transformer,
            denoiser=denoiser,
        )

    print("  Running Stage 1 denoising...")
    video_state_s1, audio_state_s1 = stage_1_diffusion(
        denoiser=FactoryGuidedDenoiser(
            v_context=v_context_p,
            a_context=a_context_p,
            video_guider_factory=create_multimodal_guider_factory(
                params=video_guider_params,
                negative_context=v_context_n,
            ),
            audio_guider_factory=create_multimodal_guider_factory(
                params=audio_guider_params,
                negative_context=a_context_n,
            ),
        ),
        sigmas=sigmas,
        noiser=noiser,
        width=args.width // 2,
        height=args.height // 2,
        frames=args.num_frames,
        fps=args.frame_rate,
        video=ModalitySpec(context=v_context_p, conditionings=stage_1_conditionings),
        audio=ModalitySpec(context=a_context_p),
        loop=stage1_loop,
    )

    print(f"  Stage 1 done.")
    print(f"    video_latent (unpatchified): {list(video_state_s1.latent.shape)}")

    # ========================================================================
    # Export: Stage 1 reference outputs
    # ========================================================================
    print("\n=== Exporting Stage 1 reference outputs ===")
    save_safetensors(
        {
            "video_latent_denoised": video_state_s1.latent,
            "audio_latent_denoised": audio_state_s1.latent,
        },
        ref / "stage1_outputs.safetensors",
    )

    # ========================================================================
    # Upsample video latent
    # ========================================================================
    print("\n=== Upsampling video latent 2x ===")
    upscaled_video_latent = upsampler(video_state_s1.latent[:1])
    print(f"  Upscaled: {list(upscaled_video_latent.shape)}")

    save_safetensors(
        {"upscaled_video_latent": upscaled_video_latent},
        ref / "upsampled.safetensors",
    )

    # ========================================================================
    # Stage 2: image-conditioned full-resolution refinement
    # ========================================================================
    print("\n=== Stage 2: full-resolution refinement ===")

    if has_image:
        stage_2_conditionings = image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images_input,
                height=args.height,
                width=args.width,
                video_encoder=enc,
                dtype=dtype,
                device=device,
            )
        )
        print(f"  Stage 2 conditionings: {len(stage_2_conditionings)} item(s)")
    else:
        stage_2_conditionings = []
        print(f"  Stage 2 conditionings: none (unconditioned)")

    distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(device)

    captured_s2 = {}

    def stage2_loop(
        sigmas: torch.Tensor,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        stepper: DiffusionStepProtocol,
        transformer: X0Model,
        denoiser: Denoiser,
    ) -> tuple[LatentState | None, LatentState | None]:
        captured_s2["video"] = video_state.clone()
        captured_s2["audio"] = audio_state.clone()
        print(f"  [capture] Stage 2 initial states:")
        print(f"    video_latent:        {list(video_state.latent.shape)} {video_state.latent.dtype}")
        print(f"    video_denoise_mask:  {list(video_state.denoise_mask.shape)}")

        if has_image:
            # Check conditioning
            s2_h_lat = args.height // 32
            s2_w_lat = args.width // 32
            n_img_tokens = s2_h_lat * s2_w_lat
            mask_first = video_state.denoise_mask[0, :n_img_tokens]
            mask_rest = video_state.denoise_mask[0, n_img_tokens:]
            print(f"    conditioning check: n_img_tokens={n_img_tokens}")
            print(f"      mask[:n_img] mean={mask_first.mean().item():.4f}  (expect ~{1.0 - args.strength:.1f})")
            print(f"      mask[n_img:] mean={mask_rest.mean().item():.4f}  (expect ~1.0)")
        else:
            print(f"    (unconditioned — all mask values should be 1.0)")
            print(f"      mask mean={video_state.denoise_mask.mean().item():.4f}")

        return euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            transformer=transformer,
            denoiser=denoiser,
        )

    print("  Running Stage 2 denoising...")
    video_state_s2, audio_state_s2 = stage_2_diffusion(
        denoiser=SimpleDenoiser(v_context=v_context_p, a_context=a_context_p),
        sigmas=distilled_sigmas,
        noiser=noiser,
        width=args.width,
        height=args.height,
        frames=args.num_frames,
        fps=args.frame_rate,
        video=ModalitySpec(
            context=v_context_p,
            conditionings=stage_2_conditionings,
            noise_scale=distilled_sigmas[0].item(),
            initial_latent=upscaled_video_latent,
        ),
        audio=ModalitySpec(
            context=a_context_p,
            noise_scale=distilled_sigmas[0].item(),
            initial_latent=audio_state_s1.latent,
        ),
        loop=stage2_loop,
    )

    print(f"  Stage 2 done.")
    print(f"    video_latent (unpatchified): {list(video_state_s2.latent.shape)}")

    # ========================================================================
    # Export: Conditioned Stage 2 inputs + noise
    # ========================================================================
    print("\n=== Exporting conditioned Stage 2 inputs ===")
    vs2 = captured_s2["video"]
    as2 = captured_s2["audio"]
    sigma_0 = distilled_sigmas[0].item()

    video_noise_s2 = recover_noise(vs2.latent, vs2.clean_latent, vs2.denoise_mask, sigma_0)
    audio_noise_s2 = recover_noise(as2.latent, as2.clean_latent, as2.denoise_mask, sigma_0)

    f_lat = (args.num_frames - 1) // 8 + 1
    h_lat = args.height // 32
    w_lat = args.width // 32

    s2_inputs = {
        "video_latent": vs2.latent,
        "audio_latent": as2.latent,
        "video_denoise_mask": vs2.denoise_mask,
        "audio_denoise_mask": as2.denoise_mask,
        "video_clean_latent": vs2.clean_latent,
        "audio_clean_latent": as2.clean_latent,
        "video_positions": vs2.positions,
        "audio_positions": as2.positions,
        "v_context": v_context_p,
        "a_context": a_context_p,
        "video_noise": video_noise_s2,
        "audio_noise": audio_noise_s2,
    }
    s2_metadata = {
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
    }
    if has_image:
        s2_metadata["strength"] = str(args.strength)
        s2_metadata["image"] = args.image
    save_safetensors(s2_inputs, out / "conditioned_stage2_inputs.safetensors", metadata=s2_metadata)

    # ========================================================================
    # Export: Stage 2 reference outputs
    # ========================================================================
    print("\n=== Exporting Stage 2 reference outputs ===")
    save_safetensors(
        {
            "video_latent_final": video_state_s2.latent,
            "audio_latent_final": audio_state_s2.latent,
        },
        ref / "stage2_outputs.safetensors",
    )

    # ========================================================================
    # Export: Stage 2 noise (for bridge script)
    # ========================================================================
    print("\n=== Exporting Stage 2 noise ===")
    save_safetensors(
        {
            "video_noise_s2": video_noise_s2,
            "audio_noise_s2": audio_noise_s2,
        },
        out / "stage2_noise.safetensors",
    )

    # ========================================================================
    # Export: Pipeline metadata
    # ========================================================================
    print("\n=== Exporting pipeline metadata ===")

    f_lat_s1 = (args.num_frames - 1) // 8 + 1
    h_lat_s1 = (args.height // 2) // 32
    w_lat_s1 = (args.width // 2) // 32
    t_v1 = f_lat_s1 * h_lat_s1 * w_lat_s1
    t_a1 = captured_s1["audio"].latent.shape[1]

    t_v2 = f_lat * h_lat * w_lat
    t_a2 = captured_s2["audio"].latent.shape[1]

    meta = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "frame_rate": args.frame_rate,
        "num_inference_steps": args.num_inference_steps,
        "checkpoint": args.checkpoint,
        "stage2_checkpoint": args.stage2_checkpoint,
        "spatial_upsampler": args.spatial_upsampler,
        "stage1": {
            "f_lat": f_lat_s1,
            "h_lat": h_lat_s1,
            "w_lat": w_lat_s1,
            "t_video": t_v1,
            "t_audio": t_a1,
        },
        "stage2": {
            "f_lat": f_lat,
            "h_lat": h_lat,
            "w_lat": w_lat,
            "t_video": t_v2,
            "t_audio": t_a2,
        },
        "guidance": {
            "video": {
                "cfg_scale": 3.0, "stg_scale": 1.0, "rescale_scale": 0.7,
                "modality_scale": 3.0, "stg_blocks": [28],
            },
            "audio": {
                "cfg_scale": 7.0, "stg_scale": 1.0, "rescale_scale": 0.7,
                "modality_scale": 3.0, "stg_blocks": [28],
            },
        },
    }
    if has_image:
        n_img_s1 = h_lat_s1 * w_lat_s1
        n_img_s2 = h_lat * w_lat
        meta["image"] = args.image
        meta["strength"] = args.strength
        meta["image_conditioning"] = {
            "stage1": {
                "image_resolution": [s1_h, s1_w],
                "n_image_tokens": n_img_s1,
                "h_lat": h_lat_s1,
                "w_lat": w_lat_s1,
            },
            "stage2": {
                "image_resolution": [args.height, args.width],
                "n_image_tokens": n_img_s2,
                "h_lat": h_lat,
                "w_lat": w_lat,
            },
        }

    meta_path = out / "pipeline_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved {meta_path}")

    # ========================================================================
    # Decode to MP4 (optional)
    # ========================================================================
    python_video_path = None
    if args.decode_video:
        print("\n=== Decoding full-Python reference video ===")
        tiling_config = TilingConfig.default()
        decode_generator = torch.Generator(device=device).manual_seed(args.seed)

        video_decoder = VideoDecoder(args.stage2_checkpoint, dtype, device)
        decoded_video = video_decoder(video_state_s2.latent, tiling_config, decode_generator)
        print("  Video decoded.")

        audio_decoder = AudioDecoder(args.stage2_checkpoint, dtype, device)
        decoded_audio = audio_decoder(audio_state_s2.latent)
        print(f"  Audio decoded: sample_rate={decoded_audio.sampling_rate}")

        python_video_path = out / "python_reference.mp4"
        python_video_path.parent.mkdir(parents=True, exist_ok=True)
        video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
        encode_video(
            video=decoded_video,
            fps=args.frame_rate,
            audio=decoded_audio,
            output_path=str(python_video_path),
            video_chunks_number=video_chunks_number,
        )
        print(f"  Full-Python reference video: {python_video_path}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE EXPORT COMPLETE")
    print("=" * 70)
    if has_image:
        print(f"\nImage:     {args.image}")
        print(f"Strength:  {args.strength}")
        print(f"\nEncoder activations:     {out / 'encoder_activations.safetensors'}")
        print(f"Preprocessed images:     {out / 'image_preprocessed.safetensors'}")
        n_img_s1 = h_lat_s1 * w_lat_s1
        n_img_s2 = h_lat * w_lat
        print(f"  → n_image_tokens: s1={n_img_s1} (h={h_lat_s1}, w={w_lat_s1}), s2={n_img_s2} (h={h_lat}, w={w_lat})")
        print(f"Conditioned Stage 2:     {out / 'conditioned_stage2_inputs.safetensors'}")
    else:
        print(f"\nMode: unconditioned (text-to-video)")
    print(f"Prompt:    {args.prompt!r}")
    print(f"\nGemma hidden states:")
    print(f"  Positive: {out / 'pos_hidden_states.safetensors'}")
    print(f"  Negative: {out / 'neg_hidden_states.safetensors'}")
    print(f"\nStage 2 noise:           {out / 'stage2_noise.safetensors'}")
    print(f"Pipeline metadata:       {out / 'pipeline_meta.json'}")
    if python_video_path:
        print(f"Full-Python ref video:   {python_video_path}")
    print(f"\nReference tensors:")
    print(f"  Stage 1 outputs: {ref / 'stage1_outputs.safetensors'}")
    print(f"  Upsampled:       {ref / 'upsampled.safetensors'}")
    print(f"  Stage 2 outputs: {ref / 'stage2_outputs.safetensors'}")
    if has_image:
        print(f"\nValidation keys in encoder_activations.safetensors:")
        all_act_tensors = {**s1_act_tensors, **s2_act_tensors}
        for key in sorted(all_act_tensors.keys()):
            t = all_act_tensors[key]
            print(f"  {key:40s}  {list(t.shape)}")


if __name__ == "__main__":
    main()
