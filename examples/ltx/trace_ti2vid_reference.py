from pathlib import Path
import os
import torch

from ltx_core.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio

from ltx_core.types import VideoPixelShape
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils import (
    cleanup_memory,
    combined_image_conditionings,
    denoise_audio_video,
    encode_prompts,
    euler_denoising_loop,
    multi_modal_guider_factory_denoising_func,
    simple_denoising_func
)
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES


TRACE_DIR = Path("trace_run")
TRACE_DIR.mkdir(exist_ok=True)


def save_pt(name: str, obj) -> None:
    torch.save(obj, TRACE_DIR / f"{name}.pt")


def main():
    checkpoint_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser())
    distilled_lora_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled-lora-384.safetensors").expanduser())
    spatial_upsampler_path = str(Path("~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors").expanduser())
    gemma_root = str(Path("~/models/gemma-3-12b-it").expanduser())

    prompt = "A beautiful sunset over the ocean"
    negative_prompt = "blurry, out of focus"
    seed = 10
    height = 1024
    width = 1536
    num_frames = 121
    frame_rate = 24.0
    num_inference_steps = 30
    images = []

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

    tiling_config = TilingConfig.default()

    device = pipeline.device
    dtype = torch.bfloat16

    video_guider_params = MultiModalGuiderParams(
        cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, skip_step=0, stg_blocks=[28],
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, skip_step=0, stg_blocks=[28],
    )

    generator = torch.Generator(device=device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)
    stepper = EulerDiffusionStep()

    ctx_p, ctx_n = encode_prompts(
        [prompt, negative_prompt],
        pipeline.stage_1_model_ledger,
        enhance_first_prompt=False,
        enhance_prompt_image=None,
        enhance_prompt_seed=seed,
    )
    v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
    v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

    save_pt("01_text_contexts", {
        "v_context_p": v_context_p.detach().cpu(),
        "a_context_p": a_context_p.detach().cpu(),
        "v_context_n": v_context_n.detach().cpu(),
        "a_context_n": a_context_n.detach().cpu(),
    })

    stage_1_output_shape = VideoPixelShape(
        batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=frame_rate
    )

    video_encoder = pipeline.stage_1_model_ledger.video_encoder()
    stage_1_conditionings = combined_image_conditionings(
        images=images,
        height=stage_1_output_shape.height,
        width=stage_1_output_shape.width,
        video_encoder=video_encoder,
        dtype=dtype,
        device=device,
    )
    save_pt("02_stage1_conditionings", stage_1_conditionings)

    del video_encoder
    torch.cuda.synchronize()
    cleanup_memory()

    transformer = pipeline.stage_1_model_ledger.transformer()
    sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=device)
    save_pt("03_stage1_sigmas", sigmas.detach().cpu())

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
        components=pipeline.pipeline_components,
        dtype=dtype,
        device=device,
    )

    save_pt("04_stage1_outputs", {
        "video_latent": video_state.latent.detach().cpu(),
        "audio_latent": audio_state.latent.detach().cpu(),
    })

    del transformer
    torch.cuda.synchronize()
    cleanup_memory()

    video_encoder = pipeline.stage_1_model_ledger.video_encoder()
    upscaled_video_latent = upsample_video(
        latent=video_state.latent[:1],
        video_encoder=video_encoder,
        upsampler=pipeline.stage_2_model_ledger.spatial_upsampler(),
    )
    save_pt("05_stage2_upsample_io", {
        "input_video_latent": video_state.latent[:1].detach().cpu(),
        "upscaled_video_latent": upscaled_video_latent.detach().cpu(),
    })

    stage_2_output_shape = VideoPixelShape(
        batch=1, frames=num_frames, width=width, height=height, fps=frame_rate
    )
    stage_2_conditionings = combined_image_conditionings(
        images=images,
        height=stage_2_output_shape.height,
        width=stage_2_output_shape.width,
        video_encoder=video_encoder,
        dtype=dtype,
        device=device,
    )
    save_pt("06_stage2_conditionings", stage_2_conditionings)

    del video_encoder
    torch.cuda.synchronize()
    cleanup_memory()

    transformer = pipeline.stage_2_model_ledger.transformer()
    distilled_sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, device=device)
    save_pt("07_stage2_sigmas", distilled_sigmas.detach().cpu())

    def second_stage_denoising_loop(
            sigmas: torch.Tensor, video_state, audio_state, stepper
        ):
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer,  # noqa: F821
                ),
            )
    
    video_state, audio_state = denoise_audio_video(
        output_shape=stage_2_output_shape,
        conditionings=stage_2_conditionings,
        noiser=noiser,
        sigmas=distilled_sigmas,
        stepper=stepper,
        denoising_loop_fn=second_stage_denoising_loop,
        components=pipeline.pipeline_components,
        dtype=dtype,
        device=device,
        noise_scale=distilled_sigmas[0],
        initial_video_latent=upscaled_video_latent,
        initial_audio_latent=audio_state.latent,
    )

    save_pt("08_stage2_outputs", {
        "video_latent": video_state.latent.detach().cpu(),
        "audio_latent": audio_state.latent.detach().cpu(),
    })

    del transformer
    torch.cuda.synchronize()
    cleanup_memory()

    decoded_video = vae_decode_video(
        video_state.latent, pipeline.stage_2_model_ledger.video_decoder(), tiling_config, generator
    )
    decoded_audio = vae_decode_audio(
        audio_state.latent, pipeline.stage_2_model_ledger.audio_decoder(), pipeline.stage_2_model_ledger.vocoder()
    )

    print(type(decoded_video))
    print(type(decoded_audio))
    # save_pt("09_decoded_outputs", {
    #     "decoded_video": decoded_video.detach().cpu(),
    #     "decoded_audio": decoded_audio.detach().cpu(),
    # })
    decoded_video_chunks = list(decoded_video)
    save_pt("09_decoded_outputs", {
        "decoded_video_chunks": [
            x.detach().cpu() if torch.is_tensor(x) else x
            for x in decoded_video_chunks
        ],
        "decoded_audio": decoded_audio.detach().cpu() if torch.is_tensor(decoded_audio) else decoded_audio,
    })


    print("Reference trace saved to", TRACE_DIR)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
