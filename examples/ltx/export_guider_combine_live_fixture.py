#!/usr/bin/env python3
"""Export guider combine fixture from a REAL Stage 1 pipeline run.

Hooks into the actual ltx_core guider to capture the 4 raw denoised
outputs (cond, neg, ptb, iso) and the combined guided result at a
single step, then saves as a safetensors fixture for Zig parity checking.

Usage:
    python export_guider_combine_live_fixture.py [output.safetensors] [--step N]

Requires: ltx_core, ltx_pipelines, torch, safetensors
"""

import sys
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from ltx_core.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.types import VideoPixelShape
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils import (
    cleanup_memory,
    combined_image_conditionings,
    denoise_audio_video,
    encode_prompts,
    euler_denoising_loop,
    multi_modal_guider_factory_denoising_func,
)


def parse_args():
    p = argparse.ArgumentParser(description="Capture guider combine from real pipeline")
    p.add_argument("output", nargs="?", default="guider_combine_live_fixture.safetensors",
                    help="Output safetensors path")
    p.add_argument("--step", type=int, default=0,
                    help="Which denoising step to capture (0-indexed, default=0)")
    return p.parse_args()


def main():
    args = parse_args()
    capture_step = args.step
    output_path = args.output

    # ── Pipeline setup (same as reference scripts) ─────────────────────────
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

    stage_1_output_shape = VideoPixelShape(
        batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=frame_rate
    )

    video_encoder = pipeline.stage_1_model_ledger.video_encoder()
    stage_1_conditionings = combined_image_conditionings(
        images=[],
        height=stage_1_output_shape.height,
        width=stage_1_output_shape.width,
        video_encoder=video_encoder,
        dtype=dtype,
        device=device,
    )
    del video_encoder
    torch.cuda.synchronize()
    cleanup_memory()

    transformer = pipeline.stage_1_model_ledger.transformer()
    sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=device)

    # ── Hook into the guider combine ───────────────────────────────────────
    # We'll monkeypatch the guider's combine method to capture the
    # 4 raw inputs and the combined output at the target step.

    captured = {"step_count": 0, "data": None}

    # Create the base denoise function
    base_denoise_fn = multi_modal_guider_factory_denoising_func(
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

    # Strategy: We can't easily hook inside the guider combine since it's
    # deep in ltx_core. Instead, we'll re-run the 4 forward passes manually
    # at the target step, then apply guide combine ourselves and compare.
    #
    # Actually, a simpler approach: monkeypatch the guider class's
    # combine_predictions method. Let's find what class it is.

    from ltx_core.components import guiders as guiders_module
    import inspect

    # Discover the combine method by inspecting the module
    print("Inspecting guiders module...")
    for name, obj in inspect.getmembers(guiders_module, inspect.isclass):
        methods = [m for m in dir(obj) if not m.startswith('_') and callable(getattr(obj, m, None))]
        if methods:
            print(f"  {name} methods: {methods}")

    # ── Hook MultiModalGuider.calculate to capture the pipeline's own output ──
    # MultiModalGuider.calculate is the method that:
    #   1. Runs the 4 forward passes via do_unconditional_generation etc.
    #   2. Collects deltas from sub-guiders (CFG, STG, modality)
    #   3. Applies the combine formula
    #   4. Returns the guided velocity
    # We monkeypatch it to capture its output at the target step.

    MultiModalGuider = guiders_module.MultiModalGuider
    original_calculate = MultiModalGuider.calculate
    pipeline_guided_outputs = {"video": [], "audio": []}

    def capturing_calculate(self, *args, **kwargs):
        result = original_calculate(self, *args, **kwargs)
        if captured["step_count"] == capture_step:
            # Result is the guided velocity for one modality.
            # MultiModalGuider.calculate is called once per modality per step
            # (video first, then audio). Capture both.
            pipeline_guided_outputs.setdefault("calls", []).append(
                result.detach().clone() if isinstance(result, torch.Tensor) else result
            )
        return result

    MultiModalGuider.calculate = capturing_calculate

    # Plan B: wrap the denoise_fn to do 4 separate calls ourselves
    # at the target step. This is more robust than trying to patch
    # internal guider methods which may vary across versions.

    # We'll intercept at the denoise_fn level:
    # On the target step, call the base_denoise_fn normally (which does
    # all 4 passes + combine internally) but ALSO manually re-run the
    # transformer 4 times to capture the raw outputs, then apply our
    # own guider_combine and compare against the pipeline's result.
    #
    # Wait — that's expensive (4 extra passes). Better approach:
    # monkeypatch the transformer.forward to capture outputs per-call.

    forward_outputs = []
    original_forward = transformer.forward

    def capturing_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        if captured["step_count"] == capture_step:
            # result is typically (video_out, audio_out) tuple
            if isinstance(result, tuple) and len(result) >= 2:
                forward_outputs.append((
                    result[0].detach().clone(),
                    result[1].detach().clone(),
                ))
            else:
                forward_outputs.append(result.detach().clone() if isinstance(result, torch.Tensor) else result)
        return result

    transformer.forward = capturing_forward

    def traced_denoise_fn(video_state, audio_state, sigmas_arg, step_idx, *a, **kw):
        current_step = captured["step_count"]
        if current_step == capture_step:
            forward_outputs.clear()
            print(f"\n[Step {current_step}] Capturing 4 forward passes...")

        result = base_denoise_fn(video_state, audio_state, sigmas_arg, step_idx, *a, **kw)

        if current_step == capture_step:
            print(f"[Step {current_step}] Captured {len(forward_outputs)} forward passes")
            # forward_outputs should have 4 entries:
            #   [0] = cond (positive prompt, normal)
            #   [1] = neg  (negative prompt, normal)
            #   [2] = ptb  (positive prompt, STG V-passthrough)
            #   [3] = iso  (positive prompt, no cross-modal)

            if len(forward_outputs) == 4:
                captured["data"] = {
                    "cond": forward_outputs[0],
                    "neg": forward_outputs[1],
                    "ptb": forward_outputs[2],
                    "iso": forward_outputs[3],
                    # The result from denoise_fn includes the guided+Euler-stepped
                    # output. We need the guided output BEFORE Euler step.
                    # We'll compute it ourselves from the 4 captures.
                }
                print(f"  cond video shape: {forward_outputs[0][0].shape}")
                print(f"  cond audio shape: {forward_outputs[0][1].shape}")
            else:
                print(f"  WARNING: Expected 4 forward passes, got {len(forward_outputs)}")
                for i, o in enumerate(forward_outputs):
                    if isinstance(o, tuple):
                        print(f"    [{i}] tuple of {len(o)}: shapes {[x.shape for x in o]}")
                    elif isinstance(o, torch.Tensor):
                        print(f"    [{i}] tensor shape {o.shape}")
                    else:
                        print(f"    [{i}] type {type(o)}")

        captured["step_count"] += 1
        return result

    def denoising_loop(sigmas_arg, video_state, audio_state, stepper_arg):
        return euler_denoising_loop(
            sigmas=sigmas_arg,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper_arg,
            denoise_fn=traced_denoise_fn,
        )

    # ── Run the pipeline ───────────────────────────────────────────────────
    print(f"Running Stage 1 denoising ({num_inference_steps} steps), capturing step {capture_step}...")
    video_state, audio_state = denoise_audio_video(
        output_shape=stage_1_output_shape,
        conditionings=stage_1_conditionings,
        noiser=noiser,
        sigmas=sigmas,
        stepper=stepper,
        denoising_loop_fn=denoising_loop,
        components=pipeline.pipeline_components,
        dtype=dtype,
        device=device,
    )

    # Restore original methods
    transformer.forward = original_forward
    MultiModalGuider.calculate = original_calculate

    # ── Process captured data ──────────────────────────────────────────────
    if captured["data"] is None:
        print(f"ERROR: No data captured at step {capture_step}!")
        sys.exit(1)

    data = captured["data"]
    cond_v, cond_a = data["cond"][0], data["cond"][1]
    neg_v, neg_a = data["neg"][0], data["neg"][1]
    ptb_v, ptb_a = data["ptb"][0], data["ptb"][1]
    iso_v, iso_a = data["iso"][0], data["iso"][1]

    print(f"\nCaptured tensors:")
    print(f"  cond_v: {cond_v.shape} {cond_v.dtype}")
    print(f"  cond_a: {cond_a.shape} {cond_a.dtype}")

    # Apply our guider_combine formula to the captured outputs
    def guider_combine(cond, neg, ptb, iso, cfg, stg, mod, rescale):
        pred = (
            cond
            + (cfg - 1) * (cond - neg)
            + stg * (cond - ptb)
            + (mod - 1) * (cond - iso)
        )
        if rescale != 0:
            factor = rescale * (cond.std() / pred.std()) + (1 - rescale)
            pred = pred * factor
        return pred

    # Compute in f32 to match Zig
    guided_v = guider_combine(
        cond_v.float(), neg_v.float(), ptb_v.float(), iso_v.float(),
        3.0, 1.0, 3.0, 0.7,
    ).to(torch.bfloat16)

    guided_a = guider_combine(
        cond_a.float(), neg_a.float(), ptb_a.float(), iso_a.float(),
        7.0, 1.0, 3.0, 0.7,
    ).to(torch.bfloat16)

    print(f"  guided_v: {guided_v.shape} {guided_v.dtype}")
    print(f"  guided_a: {guided_a.shape} {guided_a.dtype}")

    # ── Compare our formula against the pipeline's actual guider output ────
    pipeline_calls = pipeline_guided_outputs.get("calls", [])
    print(f"\nPipeline guider outputs captured: {len(pipeline_calls)} calls")
    if len(pipeline_calls) >= 2:
        # MultiModalGuider.calculate is called per-modality: video then audio
        pipeline_guided_v = pipeline_calls[0]
        pipeline_guided_a = pipeline_calls[1]
        print(f"  Pipeline guided_v: {pipeline_guided_v.shape} {pipeline_guided_v.dtype}")
        print(f"  Pipeline guided_a: {pipeline_guided_a.shape} {pipeline_guided_a.dtype}")

        # Compare our formula output vs the pipeline's actual output
        v_cos = torch.nn.functional.cosine_similarity(
            guided_v.float().flatten(), pipeline_guided_v.float().flatten(), dim=0
        )
        a_cos = torch.nn.functional.cosine_similarity(
            guided_a.float().flatten(), pipeline_guided_a.float().flatten(), dim=0
        )
        v_max_abs = (guided_v.float() - pipeline_guided_v.float()).abs().max().item()
        a_max_abs = (guided_a.float() - pipeline_guided_a.float()).abs().max().item()
        v_mean_abs = (guided_v.float() - pipeline_guided_v.float()).abs().mean().item()
        a_mean_abs = (guided_a.float() - pipeline_guided_a.float()).abs().mean().item()

        print(f"\n  === OUR FORMULA vs PIPELINE's ACTUAL OUTPUT ===")
        print(f"  Video: cos_sim={v_cos:.6f}  max_abs={v_max_abs:.6f}  mean_abs={v_mean_abs:.6f}")
        print(f"  Audio: cos_sim={a_cos:.6f}  max_abs={a_max_abs:.6f}  mean_abs={a_mean_abs:.6f}")

        if v_cos > 0.9999 and a_cos > 0.9999:
            print(f"  MATCH: Our formula matches the pipeline's guider implementation!")
        else:
            print(f"  MISMATCH: Our formula differs from the pipeline. Investigate!")
            # Also save the pipeline's actual outputs for analysis
            pipeline_guided_v_cpu = pipeline_guided_v.cpu().contiguous()
            pipeline_guided_a_cpu = pipeline_guided_a.cpu().contiguous()
            print(f"  Saving pipeline outputs for analysis...")
    elif len(pipeline_calls) == 0:
        print(f"  WARNING: No MultiModalGuider.calculate calls captured!")
        print(f"  The guider may use a different method or code path.")
    else:
        print(f"  WARNING: Expected >= 2 calculate calls (video+audio), got {len(pipeline_calls)}")
        for i, t in enumerate(pipeline_calls):
            if isinstance(t, torch.Tensor):
                print(f"    [{i}] shape={t.shape} dtype={t.dtype}")
            else:
                print(f"    [{i}] type={type(t)}")

    # ── Save fixture ───────────────────────────────────────────────────────
    tensors = {
        # Inputs (from real pipeline)
        "cond_v": cond_v.cpu().contiguous(),
        "neg_v": neg_v.cpu().contiguous(),
        "ptb_v": ptb_v.cpu().contiguous(),
        "iso_v": iso_v.cpu().contiguous(),
        "cond_a": cond_a.cpu().contiguous(),
        "neg_a": neg_a.cpu().contiguous(),
        "ptb_a": ptb_a.cpu().contiguous(),
        "iso_a": iso_a.cpu().contiguous(),
        # Scalar guidance params
        "cfg_v": torch.tensor([3.0], dtype=torch.float32),
        "stg_v": torch.tensor([1.0], dtype=torch.float32),
        "mod_v": torch.tensor([3.0], dtype=torch.float32),
        "rescale_v": torch.tensor([0.7], dtype=torch.float32),
        "cfg_a": torch.tensor([7.0], dtype=torch.float32),
        "stg_a": torch.tensor([1.0], dtype=torch.float32),
        "mod_a": torch.tensor([3.0], dtype=torch.float32),
        "rescale_a": torch.tensor([0.7], dtype=torch.float32),
        # Reference outputs (our formula applied to real intermediates)
        "guided_v": guided_v.cpu().contiguous(),
        "guided_a": guided_a.cpu().contiguous(),
    }

    # Also save the pipeline's actual guider outputs if captured
    if len(pipeline_calls) >= 2:
        tensors["pipeline_guided_v"] = pipeline_calls[0].cpu().contiguous()
        tensors["pipeline_guided_a"] = pipeline_calls[1].cpu().contiguous()

    save_file(tensors, output_path)
    print(f"\nSaved live fixture to {output_path}")
    for name, t in tensors.items():
        print(f"  {name}: shape={list(t.shape)} dtype={t.dtype}")

    # ── Diagnostic: check shapes match what Zig expects ────────────────────
    print(f"\nDiagnostics:")
    print(f"  Video tokens: {cond_v.shape[1]} (expected: half-res dependent)")
    print(f"  Audio tokens: {cond_a.shape[1]}")
    print(f"  Channel dim: {cond_v.shape[2]} (expected: 128)")

    # Verify cond differs from neg/ptb/iso
    for name, ref in [("neg", neg_v), ("ptb", ptb_v), ("iso", iso_v)]:
        cos = torch.nn.functional.cosine_similarity(
            cond_v.float().flatten(), ref.float().flatten(), dim=0
        )
        print(f"  Video cond vs {name} cos_sim: {cos:.6f}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
