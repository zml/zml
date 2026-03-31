from pathlib import Path

import torch

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.media_io import encode_video

import zml_utils

# def walk_for_modules(obj, prefix="root", seen=None, max_depth=8, depth=0):
#     if seen is None:
#         seen = set()
#     if id(obj) in seen or depth > max_depth:
#         return
#     seen.add(id(obj))

#     try:
#         if isinstance(obj, torch.nn.Module):
#             print(f"{prefix} -> {type(obj)}")
#             for name, mod in obj.named_modules():
#                 if name:
#                     print(f"  {prefix}.{name} -> {type(mod)}")
#             return
#     except Exception:
#         pass

#     # inspect normal attributes
#     for name in dir(obj):
#         if name.startswith("_"):
#             continue
#         try:
#             value = getattr(obj, name)
#         except Exception:
#             continue

#         if isinstance(value, (str, int, float, bool, bytes, Path, type(None))):
#             continue

#         if isinstance(value, torch.nn.Module):
#             print(f"{prefix}.{name} -> {type(value)}")
#             for child_name, mod in value.named_modules():
#                 if child_name:
#                     print(f"  {prefix}.{name}.{child_name} -> {type(mod)}")
#             continue

#         if isinstance(value, dict):
#             for k, v in value.items():
#                 walk_for_modules(v, f"{prefix}.{name}[{k!r}]", seen, max_depth, depth + 1)
#             continue

#         if isinstance(value, (list, tuple)):
#             for i, v in enumerate(value):
#                 walk_for_modules(v, f"{prefix}.{name}[{i}]", seen, max_depth, depth + 1)
#             continue

#         walk_for_modules(value, f"{prefix}.{name}", seen, max_depth, depth + 1)


def expose_lazy_modules_on_pipeline(pipeline):
    # Materialize the modules that will be used later inside __call__.
    # Attach them directly on the pipeline so zml_utils.named_modules(...)
    # can see them.
    pipeline._stage1_text_encoder = pipeline.stage_1_model_ledger.text_encoder()
    pipeline._stage1_video_encoder = pipeline.stage_1_model_ledger.video_encoder()
    pipeline._stage1_transformer = pipeline.stage_1_model_ledger.transformer()

    pipeline._stage2_spatial_upsampler = pipeline.stage_2_model_ledger.spatial_upsampler()
    pipeline._stage2_transformer = pipeline.stage_2_model_ledger.transformer()
    pipeline._stage2_video_decoder = pipeline.stage_2_model_ledger.video_decoder()
    pipeline._stage2_audio_decoder = pipeline.stage_2_model_ledger.audio_decoder()
    pipeline._stage2_vocoder = pipeline.stage_2_model_ledger.vocoder()

    # Force the ledger methods to return the exact same module objects
    # during pipeline(...) execution.
    pipeline.stage_1_model_ledger.text_encoder = lambda: pipeline._stage1_text_encoder
    pipeline.stage_1_model_ledger.video_encoder = lambda: pipeline._stage1_video_encoder
    pipeline.stage_1_model_ledger.transformer = lambda: pipeline._stage1_transformer

    pipeline.stage_2_model_ledger.spatial_upsampler = lambda: pipeline._stage2_spatial_upsampler
    pipeline.stage_2_model_ledger.transformer = lambda: pipeline._stage2_transformer
    pipeline.stage_2_model_ledger.video_decoder = lambda: pipeline._stage2_video_decoder
    pipeline.stage_2_model_ledger.audio_decoder = lambda: pipeline._stage2_audio_decoder
    pipeline.stage_2_model_ledger.vocoder = lambda: pipeline._stage2_vocoder

    return pipeline


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

    # Critical step.
    pipeline = expose_lazy_modules_on_pipeline(pipeline)

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

    # # inspect first
    # print(type(pipeline))
    # print("oboulant2")
    # walk_for_modules(pipeline)

    # return

    # Wrap the pipeline, and extract activations.
    # Activations files can be huge for big models,
    # so let's stop collecting after 1000 layers.
    pipeline = zml_utils.ActivationCollector(pipeline, 
                                             stop_after_first_step=True, 
                                             max_layers=1000)
    result, activations = pipeline(
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
    print(f"collected {len(activations)} activation entries")
    torch.save(activations, "ltx_2_3.activations.pt")

    if result is not None:
        video, audio = result
        
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        encode_video(
            video=video,
            fps=frame_rate,
            audio=audio,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )
        

    # video, audio = pipeline(
    #     prompt=prompt,
    #     negative_prompt=negative_prompt,
    #     seed=seed,
    #     height=height,
    #     width=width,
    #     num_frames=num_frames,
    #     frame_rate=frame_rate,
    #     num_inference_steps=num_inference_steps,
    #     video_guider_params=video_guider_params,
    #     audio_guider_params=audio_guider_params,
    #     images=[],
    #     tiling_config=tiling_config,
    # )




if __name__ == "__main__":
    # Use bf16 accumulation in matmuls to match XLA dot_precision=.fast used by ZML.
    # Without this, PyTorch defaults to TF32 accumulation for bf16 ops on CUDA,
    # which produces ~0.5% more elements outside the tolerance band when checked
    # against a ZML implementation using fast bf16 accumulation.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    with torch.inference_mode():
        main()
