from pathlib import Path

import torch

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.media_io import encode_video


def main() -> None:
    checkpoint_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser())
    distilled_lora_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled-lora-384.safetensors").expanduser())
    spatial_upsampler_path = str(Path("~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors").expanduser())
    gemma_root = str(Path("~/models/gemma-3-12b-it").expanduser())
    output_path = "output-distilled.mp4"

    prompt = "A beautiful sunset over the ocean"
    negative_prompt = (
        "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
        "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, "
        "unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, "
        "extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, "
        "camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, "
        "harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, "
        "unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, "
        "wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, "
        "background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, "
        "awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
        "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
    )

    seed = 10
    height = 1024
    width = 1536
    num_frames = 121
    frame_rate = 24.0
    num_inference_steps = 30

    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=[LoraPathStrengthAndSDOps(path=distilled_lora_path, strength=0.8, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)],
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=[],
        quantization=None,
    )

    tiling_config = TilingConfig.default()

    video_guider_params = MultiModalGuiderParams(
        cfg_scale=3.0,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,   # a2v_guidance_scale
        skip_step=0,
        stg_blocks=[28],
    )

    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=7.0,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,   # v2a_guidance_scale
        skip_step=0,
        stg_blocks=[28],
    )

    video, audio = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        num_inference_steps=num_inference_steps,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        images=[],
        tiling_config=tiling_config,
    )

    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    encode_video(
        video=video,
        fps=frame_rate,
        audio=audio,
        output_path=output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
