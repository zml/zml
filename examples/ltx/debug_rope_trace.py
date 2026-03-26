"""Extract and trace the actual Python RoPE computation step by step.

Finds the real generate_freq_grid / split_freqs_cis functions and traces them.
Also checks the cross-gate timestep scaling.

Run: cd /root/repos/LTX-2 && uv run scripts/debug_rope_trace.py
"""

import inspect
import math
import torch
from pathlib import Path
from safetensors.torch import load_file


def cos_sim(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return (torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm() + 1e-10)).item()


def main():
    fixture_path = Path("trace_run/step2_fixture_step_000_t512.safetensors")
    fix = load_file(str(fixture_path))

    # Load pipeline
    from ltx_core.types import LatentState
    from ltx_pipelines.utils.helpers import modality_from_latent_state
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

    checkpoint_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser())
    spatial_upsampler_path = str(Path("~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors").expanduser())
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
    vap = vm.video_args_preprocessor
    sp = vap.simple_preprocessor

    # ---- 1. Print all relevant source code ----
    print("\n" + "=" * 80)
    print("=== simple_preprocessor type and source ===")
    print(f"type: {type(sp).__name__}  module: {type(sp).__module__}")

    # Print attributes
    for attr in sorted(dir(sp)):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(sp, attr)
            if callable(val) and not isinstance(val, torch.nn.Module):
                continue
            if isinstance(val, torch.Tensor) and val.numel() > 100:
                print(f"  {attr}: Tensor shape={list(val.shape)}")
            elif isinstance(val, (int, float, str, bool, list, tuple)):
                print(f"  {attr}: {val}")
            elif isinstance(val, torch.nn.Module):
                print(f"  {attr}: {type(val).__name__}")
        except Exception:
            pass

    # Print prepare source
    print("\n" + "=" * 80)
    print("=== simple_preprocessor.prepare() source ===")
    try:
        src = inspect.getsource(type(sp).prepare)
        for i, line in enumerate(src.split('\n'), 1):
            print(f"  {i:3d}: {line}")
    except Exception as e:
        print(f"  Error: {e}")

    # Print _prepare_positional_embeddings source
    print("\n" + "=" * 80)
    print("=== simple_preprocessor._prepare_positional_embeddings() source ===")
    try:
        src = inspect.getsource(type(sp)._prepare_positional_embeddings)
        for i, line in enumerate(src.split('\n'), 1):
            print(f"  {i:3d}: {line}")
    except Exception as e:
        print(f"  Error: {e}")

    # Find the precompute_freqs_cis / generate_freq_grid functions
    print("\n" + "=" * 80)
    print("=== Finding RoPE functions ===")

    # Search in related modules
    import ltx_core
    for mod_name in ['ltx_core.model', 'ltx_core.model.transformer',
                     'ltx_core.model.transformer.positional_encoding',
                     'ltx_core.model.positional_encoding',
                     'ltx_core.model.rope', 'ltx_core.rope',
                     'ltx_core.utils', 'ltx_core.model.transformer.rope']:
        try:
            mod = __import__(mod_name, fromlist=[''])
            for name in dir(mod):
                if any(kw in name.lower() for kw in ['freq', 'rope', 'positional', 'cis', 'rotary']):
                    obj = getattr(mod, name)
                    if callable(obj) and not isinstance(obj, type):
                        print(f"\n  Found: {mod_name}.{name}")
                        try:
                            fsrc = inspect.getsource(obj)
                            for j, line in enumerate(fsrc.split('\n'), 1):
                                print(f"    {j:3d}: {line}")
                        except Exception:
                            print(f"    (could not get source)")
        except ImportError:
            pass

    # Also look for the function by tracing _prepare_positional_embeddings imports
    print("\n" + "=" * 80)
    print("=== Tracing _prepare_positional_embeddings imports ===")
    try:
        src_file = inspect.getfile(type(sp))
        print(f"  Source file: {src_file}")
        with open(src_file) as f:
            file_src = f.read()
        # Print import lines and any function starting with generate_freq or split_freq
        for i, line in enumerate(file_src.split('\n'), 1):
            l = line.strip()
            if (l.startswith('import') or l.startswith('from')) and any(
                kw in l.lower() for kw in ['freq', 'rope', 'positional', 'cis', 'precompute']):
                print(f"  L{i}: {l}")
            if any(kw in l.lower() for kw in ['def generate_freq', 'def split_freq',
                                                'def precompute_freq', 'def get_fractional']):
                print(f"  L{i}: {l}")
    except Exception as e:
        print(f"  Error: {e}")

    # Try to find through the module that the preprocessor is in
    print("\n" + "=" * 80)
    print("=== Searching all ltx_core modules for frequency functions ===")
    import pkgutil
    for importer, modname, ispkg in pkgutil.walk_packages(
        ltx_core.__path__, prefix='ltx_core.'):
        try:
            mod = __import__(modname, fromlist=[''])
            for name in dir(mod):
                if 'freq_grid' in name.lower() or 'freqs_cis' in name.lower() or 'split_freq' in name.lower():
                    obj = getattr(mod, name)
                    if callable(obj):
                        print(f"\n  Found: {modname}.{name}")
                        try:
                            fsrc = inspect.getsource(obj)
                            for j, line in enumerate(fsrc.split('\n'), 1):
                                print(f"    {j:3d}: {line}")
                        except Exception:
                            print(f"    (could not get source)")
        except Exception:
            pass

    # ---- 2. Cross-gate timestep investigation ----
    print("\n" + "=" * 80)
    print("=== _prepare_cross_attention_timestep source ===")
    try:
        src = inspect.getsource(type(vap)._prepare_cross_attention_timestep)
        for i, line in enumerate(src.split('\n'), 1):
            print(f"  {i:3d}: {line}")
    except Exception as e:
        print(f"  Error: {e}")

    # Check the timestep_scale_multiplier
    print(f"\n  timestep_scale_multiplier: {sp.timestep_scale_multiplier}")
    print(f"  av_ca_timestep_scale_multiplier: {vap.av_ca_timestep_scale_multiplier}")

    # ---- 3. Run actual PE computation and trace ----
    print("\n" + "=" * 80)
    print("=== Running PE computation with tracing ===")

    sigma = fix["raw.sigma"].to(dtype=torch.float32).cuda()
    v_state = LatentState(
        latent=fix["raw.video_latent"].to(dtype=torch.bfloat16).cuda(),
        denoise_mask=fix["raw.video_denoise_mask"].cuda(),
        positions=fix["raw.video_positions"].cuda(),
        clean_latent=fix["raw.video_clean_latent"].to(dtype=torch.bfloat16).cuda(),
    )
    a_state = LatentState(
        latent=fix["raw.audio_latent"].to(dtype=torch.bfloat16).cuda(),
        denoise_mask=fix["raw.audio_denoise_mask"].cuda(),
        positions=fix["raw.audio_positions"].cuda(),
        clean_latent=fix["raw.audio_clean_latent"].to(dtype=torch.bfloat16).cuda(),
    )
    v_ctx = fix["raw.v_context"].to(dtype=torch.bfloat16).cuda()
    a_ctx = fix["raw.a_context"].to(dtype=torch.bfloat16).cuda()

    pos_video = modality_from_latent_state(v_state, v_ctx, sigma)
    pos_audio = modality_from_latent_state(a_state, a_ctx, sigma)

    # Call _prepare_positional_embeddings directly with video positions
    try:
        v_pe = sp._prepare_positional_embeddings(
            positions=pos_video.positions,
            inner_dim=4096,
            max_pos=[20, 2048, 2048],
            use_middle_indices_grid=True,
            num_attention_heads=32,
            x_dtype=torch.bfloat16,
        )
        v_pe_cos, v_pe_sin = v_pe
        ref_v_cos = fix["intermediate.v_pe_cos"].float()
        print(f"  PE with max_pos=[20,2048,2048]: cos_sim={cos_sim(v_pe_cos.cpu(), ref_v_cos):.6f}")
    except Exception as e:
        print(f"  Error calling _prepare_positional_embeddings: {e}")

    # Try to find actual max_pos from the config
    print("\n  Trying to find actual max_pos used by simple_preprocessor...")
    if hasattr(sp, 'max_pos'):
        print(f"  sp.max_pos = {sp.max_pos}")
        v_pe2 = sp._prepare_positional_embeddings(
            positions=pos_video.positions,
            inner_dim=4096,
            max_pos=sp.max_pos,
            use_middle_indices_grid=True,
            num_attention_heads=32,
            x_dtype=torch.bfloat16,
        )
        v_pe2_cos, _ = v_pe2
        print(f"  PE with sp.max_pos={sp.max_pos}: cos_sim={cos_sim(v_pe2_cos.cpu(), ref_v_cos):.6f}")


if __name__ == "__main__":
    main()
