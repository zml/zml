"""Diagnostic script: inspect RoPE config from the Python LTX-2 model.

Run from the LTX-2 repo directory:
  cd /root/repos/LTX-2
  uv run scripts/debug_rope_config.py
"""

import math
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file

from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline


def main() -> None:
    checkpoint_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser())
    distilled_lora_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled-lora-384.safetensors").expanduser())
    spatial_upsampler_path = str(
        Path("~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors").expanduser()
    )
    gemma_root = str(Path("~/models/gemma-3-12b-it").expanduser())

    print("Loading pipeline...")
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=[],
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=[],
        quantization=None,
    )

    vm = pipeline.stage_2_model_ledger.transformer().velocity_model

    # ---- Inspect video args preprocessor RoPE config ----
    print("\n=== Video Args Preprocessor ===")
    vap = vm.video_args_preprocessor
    print(f"  type: {type(vap)}")
    for attr in ['rope', 'positional_encoding', 'pe', 'rope_emb', 'freq_cis', 'rotary_emb']:
        if hasattr(vap, attr):
            rope_obj = getattr(vap, attr)
            print(f"  {attr}: {type(rope_obj)}")
            for sub in ['theta', 'max_pos', 'n_pos_dims', 'inner_dim', 'dim',
                        'rope_max_pos', 'base', 'inv_freq', 'freqs', 'config']:
                if hasattr(rope_obj, sub):
                    val = getattr(rope_obj, sub)
                    if isinstance(val, torch.Tensor) and val.numel() > 20:
                        print(f"    {sub}: shape={list(val.shape)} dtype={val.dtype}")
                    else:
                        print(f"    {sub}: {val}")

    # Try to find RoPE parameters by inspecting all attributes
    print("\n  All attributes of video_args_preprocessor:")
    for name in sorted(dir(vap)):
        if name.startswith('_'):
            continue
        try:
            val = getattr(vap, name)
            if callable(val):
                continue
            if isinstance(val, torch.Tensor) and val.numel() > 50:
                print(f"    {name}: Tensor shape={list(val.shape)} dtype={val.dtype}")
            else:
                print(f"    {name}: {val}")
        except Exception:
            pass

    # ---- Inspect audio args preprocessor ----
    print("\n=== Audio Args Preprocessor ===")
    aap = vm.audio_args_preprocessor
    print(f"  type: {type(aap)}")
    for name in sorted(dir(aap)):
        if name.startswith('_'):
            continue
        try:
            val = getattr(aap, name)
            if callable(val):
                continue
            if isinstance(val, torch.Tensor) and val.numel() > 50:
                print(f"    {name}: Tensor shape={list(val.shape)} dtype={val.dtype}")
            else:
                print(f"    {name}: {val}")
        except Exception:
            pass

    # ---- Load fixture and compare PE computation ----
    print("\n=== Fixture PE comparison ===")
    fixture_path = Path("trace_run/step2_fixture_step_000_t512.safetensors")
    if fixture_path.exists():
        fix = load_file(str(fixture_path))
        v_positions = fix["raw.video_positions"].to(device=pipeline.device)
        a_positions = fix["raw.audio_positions"].to(device=pipeline.device)

        print(f"\n  video_positions: shape={list(v_positions.shape)} dtype={v_positions.dtype}")
        print(f"  audio_positions: shape={list(a_positions.shape)} dtype={a_positions.dtype}")

        # Print position value ranges per dimension
        for c in range(v_positions.shape[1]):
            start_vals = v_positions[0, c, :, 0].float()
            end_vals = v_positions[0, c, :, 1].float()
            mid_vals = (start_vals + end_vals) / 2
            print(f"  video dim {c}: start range=[{start_vals.min():.1f}, {start_vals.max():.1f}]"
                  f"  end range=[{end_vals.min():.1f}, {end_vals.max():.1f}]"
                  f"  mid range=[{mid_vals.min():.3f}, {mid_vals.max():.3f}]"
                  f"  unique_mids={mid_vals.unique().numel()}")

        for c in range(a_positions.shape[1]):
            start_vals = a_positions[0, c, :, 0].float()
            end_vals = a_positions[0, c, :, 1].float()
            mid_vals = (start_vals + end_vals) / 2
            print(f"  audio dim {c}: start range=[{start_vals.min():.1f}, {start_vals.max():.1f}]"
                  f"  end range=[{end_vals.min():.1f}, {end_vals.max():.1f}]"
                  f"  mid range=[{mid_vals.min():.3f}, {mid_vals.max():.3f}]"
                  f"  unique_mids={mid_vals.unique().numel()}")

        # Run the Python preprocessor on fixture positions to check cross-PE
        print("\n  Running Python preprocessors on fixture positions...")
        from ltx_core.types import LatentState
        from ltx_pipelines.utils.helpers import modality_from_latent_state

        sigma = fix["raw.sigma"].to(device=pipeline.device, dtype=torch.float32)
        v_state = LatentState(
            latent=fix["raw.video_latent"].to(device=pipeline.device, dtype=torch.bfloat16),
            denoise_mask=fix["raw.video_denoise_mask"].to(device=pipeline.device),
            positions=v_positions,
            clean_latent=fix["raw.video_clean_latent"].to(device=pipeline.device, dtype=torch.bfloat16),
        )
        a_state = LatentState(
            latent=fix["raw.audio_latent"].to(device=pipeline.device, dtype=torch.bfloat16),
            denoise_mask=fix["raw.audio_denoise_mask"].to(device=pipeline.device),
            positions=a_positions,
            clean_latent=fix["raw.audio_clean_latent"].to(device=pipeline.device, dtype=torch.bfloat16),
        )
        v_ctx = fix["raw.v_context"].to(device=pipeline.device, dtype=torch.bfloat16)
        a_ctx = fix["raw.a_context"].to(device=pipeline.device, dtype=torch.bfloat16)

        pos_video = modality_from_latent_state(v_state, v_ctx, sigma)
        pos_audio = modality_from_latent_state(a_state, a_ctx, sigma)

        # Check if modality_from_latent_state transforms positions
        print(f"\n  modality video positions: shape={list(pos_video.positions.shape)} dtype={pos_video.positions.dtype}")
        print(f"  modality audio positions: shape={list(pos_audio.positions.shape)} dtype={pos_audio.positions.dtype}")

        # Compare positions before/after
        if torch.equal(v_positions, pos_video.positions):
            print("  video positions unchanged by modality_from_latent_state")
        else:
            diff = (v_positions.float() - pos_video.positions.float()).abs().max()
            print(f"  video positions CHANGED! max_diff={diff}")

        # Try to access the actual RoPE object
        print("\n  Checking video_args_preprocessor.prepare internals...")
        with torch.inference_mode():
            video_args = vap.prepare(pos_video, pos_audio)

        # Check if the preprocessor has a rope attribute we can introspect
        if hasattr(vap, 'prepare'):
            import inspect
            src = inspect.getsource(type(vap).prepare)
            # Look for RoPE-related patterns
            rope_lines = [l.strip() for l in src.split('\n')
                         if any(kw in l.lower() for kw in ['rope', 'freq', 'positional', 'cos', 'sin', 'pe', 'rotary'])]
            print(f"\n  RoPE-related lines in prepare():")
            for l in rope_lines[:20]:
                print(f"    {l}")

        # Print first few values of reference PE for comparison
        ref_v_pe_cos = fix.get("intermediate.v_pe_cos")
        if ref_v_pe_cos is not None:
            print(f"\n  Reference v_pe_cos: shape={list(ref_v_pe_cos.shape)} dtype={ref_v_pe_cos.dtype}")
            print(f"  First values [0, 0, 0, :8]: {ref_v_pe_cos[0, 0, 0, :8].float().tolist()}")
            print(f"  First values [0, 0, 1, :8]: {ref_v_pe_cos[0, 0, 1, :8].float().tolist()}")
            print(f"  First values [0, 1, 0, :8]: {ref_v_pe_cos[0, 1, 0, :8].float().tolist()}")
    else:
        print(f"  Fixture not found at {fixture_path}")


if __name__ == "__main__":
    main()
