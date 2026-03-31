"""Inspector script (part 2): dump remaining Step 2 source code.

Targets:
  - timesteps_from_mask (used by modality_from_latent_state)
  - RoPE frequency generation: generate_freqs, precompute_freqs_cis, split_freqs_cis, etc.
  - LTXMultiModalDiffusionTransformer.forward (velocity_model forward)
  - 11_stage2_steps.pt structure (fixed formatting)

Usage (on GPU machine):
  cd /root/repos/LTX-2
  uv run scripts/inspect_step2_sources_part2.py
"""

import inspect
import torch
from pathlib import Path


def section(title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def dump_source(obj, label: str):
    try:
        src = inspect.getsource(obj)
        print(f"--- {label} (from {inspect.getfile(obj)}) ---")
        print(src)
    except Exception as exc:
        print(f"--- {label}: could not get source: {exc} ---")


def main():
    # 1. timesteps_from_mask
    section("timesteps_from_mask")
    try:
        from ltx_pipelines.utils.helpers import timesteps_from_mask
        dump_source(timesteps_from_mask, "timesteps_from_mask")
    except ImportError:
        # Search in the helpers module
        try:
            import ltx_pipelines.utils.helpers as helpers_mod
            names = [n for n in dir(helpers_mod) if "timestep" in n.lower() or "mask" in n.lower()]
            print(f"  helpers module names matching timestep/mask: {names}")
            for n in names:
                dump_source(getattr(helpers_mod, n), f"helpers.{n}")
        except Exception as exc:
            print(f"  Could not inspect helpers: {exc}")

    # 2. RoPE frequency generation functions
    section("RoPE frequency generation functions")
    try:
        import ltx_core.model.transformer.rope as rope_mod

        for name in [
            "generate_freqs",
            "precompute_freqs_cis",
            "split_freqs_cis",
            "interleaved_freqs_cis",
            "generate_freq_grid_np",
            "generate_freq_grid_pytorch",
        ]:
            obj = getattr(rope_mod, name, None)
            if obj is not None:
                dump_source(obj, f"rope.{name}")
            else:
                print(f"  rope.{name}: not found")
    except ImportError as exc:
        print(f"  Could not import rope module: {exc}")

    # 3. LTXMultiModalDiffusionTransformer.forward (velocity_model)
    section("velocity_model forward")
    # Try multiple possible class locations
    found_vm = False
    for cls_path in [
        "ltx_core.model.transformer.ltx_model.LTXMultiModalDiffusionTransformer",
        "ltx_core.model.transformer.transformer.LTXMultiModalDiffusionTransformer",
        "ltx_core.model.ltx_model.LTXMultiModalDiffusionTransformer",
    ]:
        try:
            parts = cls_path.rsplit(".", 1)
            mod = __import__(parts[0], fromlist=[parts[1]])
            cls = getattr(mod, parts[1])
            dump_source(cls.forward, f"{parts[1]}.forward")
            # Also get __init__ to understand config params
            dump_source(cls.__init__, f"{parts[1]}.__init__")
            found_vm = True
            break
        except (ImportError, AttributeError):
            continue

    if not found_vm:
        # Fallback: find it via the pipeline
        print("  Direct import failed. Trying via pipeline...")
        try:
            from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
            # Just inspect the class, don't instantiate
            # Look at the pipeline source to find the model class
            dump_source(TI2VidTwoStagesPipeline, "TI2VidTwoStagesPipeline (full class)")
        except Exception as exc:
            print(f"  Fallback also failed: {exc}")

    # 4. X0Model / transformer wrapper forward
    section("X0Model (transformer wrapper)")
    for cls_path in [
        "ltx_core.guidance.x0_model.X0Model",
        "ltx_core.model.x0_model.X0Model",
    ]:
        try:
            parts = cls_path.rsplit(".", 1)
            mod = __import__(parts[0], fromlist=[parts[1]])
            cls = getattr(mod, parts[1])
            dump_source(cls.forward, f"{parts[1]}.forward")
            dump_source(cls.__init__, f"{parts[1]}.__init__")
            break
        except (ImportError, AttributeError):
            continue

    # 5. Search for _prepare_positional_embeddings / _get_rope etc.
    section("Positional embedding preparation methods")
    # These are often methods on the transformer class
    for cls_path in [
        "ltx_core.model.transformer.ltx_model.LTXMultiModalDiffusionTransformer",
        "ltx_core.model.transformer.transformer.LTXMultiModalDiffusionTransformer",
    ]:
        try:
            parts = cls_path.rsplit(".", 1)
            mod = __import__(parts[0], fromlist=[parts[1]])
            cls = getattr(mod, parts[1])
            # Look for PE-related methods
            for attr_name in dir(cls):
                if any(kw in attr_name.lower() for kw in ["position", "rope", "embed", "freq", "pe"]):
                    obj = getattr(cls, attr_name, None)
                    if callable(obj):
                        dump_source(obj, f"{parts[1]}.{attr_name}")
            break
        except (ImportError, AttributeError):
            continue

    # 6. 11_stage2_steps.pt structure (fixed)
    section("11_stage2_steps.pt structure")
    try:
        trace_dir = Path("trace_run")
        steps = torch.load(trace_dir / "11_stage2_steps.pt", map_location="cpu", weights_only=False)
        print(f"  Type: {type(steps)}, len: {len(steps)}")
        step0 = steps[0]
        print(f"  step[0] type: {type(step0)}")
        if isinstance(step0, dict):
            for k, v in step0.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k:30s}  shape={list(v.shape)}  dtype={v.dtype}")
                else:
                    print(f"    {k:30s}  type={type(v).__name__}  value={repr(v)}")
    except Exception as exc:
        print(f"  Could not load: {exc}")
        import traceback
        traceback.print_exc()

    # 7. Modality.forward or how Modality is consumed by velocity_model
    section("How Modality is consumed")
    try:
        from ltx_core.model.transformer.modality import Modality
        # Check if Modality has any methods beyond dataclass defaults
        custom_methods = [m for m in dir(Modality) if not m.startswith("_") and callable(getattr(Modality, m, None))]
        print(f"  Modality custom methods: {custom_methods}")
        for m in custom_methods:
            dump_source(getattr(Modality, m), f"Modality.{m}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
