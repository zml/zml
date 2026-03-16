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

import inspect


TRACE_DIR = Path("trace_run")
TRACE_DIR.mkdir(exist_ok=True)

PARAM_NAMES_TO_COMPARE = [
    # --- block 0, video FF / attn ---
    "velocity_model.transformer_blocks.0.attn1.to_q.weight",
    "velocity_model.transformer_blocks.0.attn1.to_q.bias",
    "velocity_model.transformer_blocks.0.ff.net.0.proj.weight",
    "velocity_model.transformer_blocks.0.ff.net.0.proj.bias",
    "velocity_model.transformer_blocks.0.ff.net.2.weight",
    "velocity_model.transformer_blocks.0.ff.net.2.bias",

    # --- another early block ---
    "velocity_model.transformer_blocks.1.attn1.to_q.weight",
    "velocity_model.transformer_blocks.1.ff.net.0.proj.weight",
    "velocity_model.transformer_blocks.1.ff.net.2.weight",

    # --- a late block ---
    "velocity_model.transformer_blocks.47.attn1.to_q.weight",
    "velocity_model.transformer_blocks.47.ff.net.0.proj.weight",
    "velocity_model.transformer_blocks.47.ff.net.2.weight",

    # --- audio-side ---
    "velocity_model.transformer_blocks.0.audio_attn1.to_q.weight",
    "velocity_model.transformer_blocks.0.audio_attn1.to_q.bias",
    "velocity_model.transformer_blocks.0.audio_ff.net.0.proj.weight",
    "velocity_model.transformer_blocks.0.audio_ff.net.2.weight",

    # --- cross-modal ---
    "velocity_model.transformer_blocks.0.audio_to_video_attn.to_q.weight",
    "velocity_model.transformer_blocks.0.audio_to_video_attn.to_out.0.weight",
    "velocity_model.transformer_blocks.0.video_to_audio_attn.to_k.weight",
    "velocity_model.transformer_blocks.0.video_to_audio_attn.to_out.0.weight",
]


def extract_named_params_to_cpu(model, names):
    sd = model.state_dict()
    out = {}
    for name in names:
        if name not in sd:
            print(f"[missing] {name}")
            continue
        out[name] = sd[name].detach().cpu().clone()
    return out

def compare_param_dicts(params_a, params_b):
    print("--------------------------------------------------------")
    print("---------------- Parameter comparison ------------------")
    print("--------------------------------------------------------")

    all_names = sorted(set(params_a.keys()) | set(params_b.keys()))
    changed = 0
    unchanged = 0

    for name in all_names:
        a = params_a.get(name)
        b = params_b.get(name)

        if a is None:
            print(f"[missing in A] {name}\n")
            continue
        if b is None:
            print(f"[missing in B] {name}\n")
            continue

        same_shape = a.shape == b.shape
        exact_equal = torch.equal(a, b)

        if same_shape and a.is_floating_point() and b.is_floating_point():
            max_abs_diff = (a.float() - b.float()).abs().max().item()
        elif same_shape:
            max_abs_diff = 0.0 if exact_equal else float("nan")
        else:
            max_abs_diff = float("nan")

        if exact_equal:
            unchanged += 1
        else:
            changed += 1

        print(f"name          : {name}")
        print(f"shape A / B   : {tuple(a.shape)} / {tuple(b.shape)}")
        print(f"exact equal   : {exact_equal}")
        print(f"max abs diff  : {max_abs_diff}")
        print()

    print("--------------------------------------------------------")
    print(f"changed   : {changed}")
    print(f"unchanged : {unchanged}")
    print("--------------------------------------------------------")


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

    # Debug model tree
    stage1_params = extract_named_params_to_cpu(transformer, PARAM_NAMES_TO_COMPARE)
    print("--------------------------------------------------------")
    print("------------------ Stage 1 Model Tree ------------------")
    print("--------------------------------------------------------")
    print("------------------ Transformer type: ------------------")
    print(type(transformer))
    print("------------------ Transformer : ------------------")
    print(transformer)
    print("------------------ Velocity Model type: ------------------")
    print(type(transformer.velocity_model))
    print("------------------ Velocity Model : ------------------")
    print(transformer.velocity_model)
    print("------------------ First Transformer Block type: ------------------")
    print(type(transformer.velocity_model.transformer_blocks[0]))
    print("------------------ First Transformer Block : ------------------")
    print(transformer.velocity_model.transformer_blocks[0])
    print("------------------ First Transformer Block FF type: ------------------")
    print(type(transformer.velocity_model.transformer_blocks[0].ff))
    print("------------------ First Transformer Block FF : ------------------")
    print(transformer.velocity_model.transformer_blocks[0].ff)

    stage1_steps = []
    stage1_base_denoise_fn = multi_modal_guider_factory_denoising_func(
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
    )

    def stage1_traced_denoise_fn(video_state, audio_state, sigmas, step_idx, *args, **kwargs):
        stage1_steps.append({
            "sigma": sigmas[step_idx].detach().cpu(),

            "video_latent": video_state.latent.detach().cpu(),
            "video_denoise_mask": video_state.denoise_mask.detach().cpu(),
            "video_positions": video_state.positions.detach().cpu(),
            "video_clean_latent": video_state.clean_latent.detach().cpu(),

            "audio_latent": audio_state.latent.detach().cpu(),
            "audio_denoise_mask": audio_state.denoise_mask.detach().cpu(),
            "audio_positions": audio_state.positions.detach().cpu(),
            "audio_clean_latent": audio_state.clean_latent.detach().cpu(),
        })
        return stage1_base_denoise_fn(video_state, audio_state, sigmas, step_idx, *args, **kwargs)

    def first_stage_denoising_loop(sigmas, video_state, audio_state, stepper):
        return euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            denoise_fn=stage1_traced_denoise_fn,
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
    save_pt("10_stage1_steps", stage1_steps)

    # Drop closures that capture the stage-1 transformer before stage-2 loads.
    del stage1_base_denoise_fn
    del stage1_traced_denoise_fn
    del first_stage_denoising_loop

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

    # Debug model tree
    stage2_params = extract_named_params_to_cpu(transformer, PARAM_NAMES_TO_COMPARE)
    print("--------------------------------------------------------")
    print("------------------ Stage 2 Model Tree ------------------")
    print("--------------------------------------------------------")
    print("------------------ Transformer type: ------------------")
    print(type(transformer))
    print("------------------ Transformer : ------------------")
    print(transformer)
    print("------------------ Velocity Model type: ------------------")
    print(type(transformer.velocity_model))
    print("------------------ Velocity Model : ------------------")
    print(transformer.velocity_model)
    print("------------------ First Transformer Block type: ------------------")
    print(type(transformer.velocity_model.transformer_blocks[0]))
    print("------------------ First Transformer Block : ------------------")
    print(transformer.velocity_model.transformer_blocks[0])
    print("------------------ First Transformer Block FF type: ------------------")
    print(type(transformer.velocity_model.transformer_blocks[0].ff))
    print("------------------ First Transformer Block FF : ------------------")
    print(transformer.velocity_model.transformer_blocks[0].ff)

    # Compare parameters of interest between stage 1 and stage 2 transformers.
    compare_param_dicts(stage1_params, stage2_params)

    # Still exploring
    ff = transformer.velocity_model.transformer_blocks[0].ff
    block = transformer.velocity_model.transformer_blocks[0]

    print("===== FeedForward source =====")
    print(inspect.getsource(ff.__class__))

    print("===== BasicAVTransformerBlock source =====")
    print(inspect.getsource(block.__class__))

    gelu_approx = ff.net[0]
    print("===== GELUApprox source =====")
    print(inspect.getsource(gelu_approx.__class__))

    mods = {
        "block0.ff": transformer.velocity_model.transformer_blocks[0].ff,
        "block0.ff.net.0": transformer.velocity_model.transformer_blocks[0].ff.net[0],
        "block0.ff.net.0.proj": transformer.velocity_model.transformer_blocks[0].ff.net[0].proj,
        "block0.ff.net.2": transformer.velocity_model.transformer_blocks[0].ff.net[2],
    }

    handles = []
    def hook(name):
        def _hook(module, inputs, output):
            print(f"\n--- {name} ---")
            if inputs:
                x = inputs[0]
                if torch.is_tensor(x):
                    print("input shape :", tuple(x.shape), x.dtype, x.device)
            if torch.is_tensor(output):
                print("output shape:", tuple(output.shape), output.dtype, output.device)
        return _hook

    for name, mod in mods.items():
        handles.append(mod.register_forward_hook(hook(name)))

    # End still exploring


    stage2_steps = []
    stage2_base_denoise_fn = simple_denoising_func(
        video_context=v_context_p,
        audio_context=a_context_p,
        transformer=transformer,
    )

    def stage2_traced_denoise_fn(video_state, audio_state, sigmas, step_idx, *args, **kwargs):
        stage2_steps.append({
            "sigma": sigmas[step_idx].detach().cpu(),

            "video_latent": video_state.latent.detach().cpu(),
            "video_denoise_mask": video_state.denoise_mask.detach().cpu(),
            "video_positions": video_state.positions.detach().cpu(),
            "video_clean_latent": video_state.clean_latent.detach().cpu(),

            "audio_latent": audio_state.latent.detach().cpu(),
            "audio_denoise_mask": audio_state.denoise_mask.detach().cpu(),
            "audio_positions": audio_state.positions.detach().cpu(),
            "audio_clean_latent": audio_state.clean_latent.detach().cpu(),
        })
        return stage2_base_denoise_fn(video_state, audio_state, sigmas, step_idx, *args, **kwargs)

    def second_stage_denoising_loop(
            sigmas: torch.Tensor, video_state, audio_state, stepper
        ):
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=stage2_traced_denoise_fn,
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

    # Removing the hooks before decoding to avoid cluttering the trace with decoder internals.
    for h in handles:
        h.remove()

    save_pt("08_stage2_outputs", {
        "video_latent": video_state.latent.detach().cpu(),
        "audio_latent": audio_state.latent.detach().cpu(),
    })
    save_pt("11_stage2_steps", stage2_steps)

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
