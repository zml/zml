"""Inspector part 3: unwrap lru_cache'd RoPE functions and find velocity_model.forward.

Usage (on GPU machine):
  cd /root/repos/LTX-2
  uv run scripts/inspect_step2_sources_part3.py
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
    # 1. Unwrap lru_cache'd RoPE functions
    section("generate_freq_grid_pytorch (unwrapped)")
    try:
        from ltx_core.model.transformer.rope import generate_freq_grid_pytorch, generate_freq_grid_np
        # lru_cache wraps the function — the original is in __wrapped__
        unwrapped = getattr(generate_freq_grid_pytorch, "__wrapped__", None)
        if unwrapped is not None:
            dump_source(unwrapped, "generate_freq_grid_pytorch.__wrapped__")
        else:
            print("  No __wrapped__ attribute. Trying inspect on cache object...")
            # Try to read the source file directly
            try:
                src_file = inspect.getfile(generate_freq_grid_pytorch)
                print(f"  Source file: {src_file}")
                with open(src_file) as f:
                    content = f.read()
                # Find the function definition
                import re
                match = re.search(r'((?:@\w+.*\n)*def generate_freq_grid_pytorch\b.*?)(?=\n(?:@\w+|def |class |\Z))', content, re.DOTALL)
                if match:
                    print(f"--- generate_freq_grid_pytorch (from file) ---")
                    print(match.group(1))
                else:
                    print("  Could not find function in file")
            except Exception as exc2:
                print(f"  Fallback also failed: {exc2}")

        # Same for numpy variant
        unwrapped_np = getattr(generate_freq_grid_np, "__wrapped__", None)
        if unwrapped_np is not None:
            dump_source(unwrapped_np, "generate_freq_grid_np.__wrapped__")
    except ImportError as exc:
        print(f"  Import failed: {exc}")

    # 2. Find velocity_model forward via live pipeline
    section("velocity_model class and forward (via pipeline)")
    try:
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

        # Don't instantiate pipeline — just find the class
        # The velocity_model is created by ModelLedger.transformer()
        # Let's find ModelLedger
        from ltx_core.loader import ModelLedger
        print("ModelLedger found")

        # Check if we can find the transformer class it creates
        dump_source(ModelLedger.transformer, "ModelLedger.transformer")
    except ImportError:
        pass
    except Exception as exc:
        print(f"  ModelLedger error: {exc}")

    # 3. Find the velocity model class directly by searching common paths
    section("velocity_model class search")
    search_modules = [
        "ltx_core.model.transformer.ltx_model",
        "ltx_core.model.transformer.velocity_model",
        "ltx_core.model.transformer",
        "ltx_core.model.ltx_model",
        "ltx_core.model",
    ]
    for mod_name in search_modules:
        try:
            mod = __import__(mod_name, fromlist=[""])
            names = dir(mod)
            # Look for classes with 'Transformer' or 'Velocity' in name
            interesting = [n for n in names if any(kw in n for kw in ["Transformer", "Velocity", "Model"])]
            if interesting:
                print(f"\n  Module {mod_name}: {interesting}")
                for name in interesting:
                    cls = getattr(mod, name, None)
                    if cls is not None and inspect.isclass(cls):
                        if hasattr(cls, 'forward'):
                            dump_source(cls.forward, f"{name}.forward")
                            # Also check for _process methods
                            for attr in dir(cls):
                                if attr.startswith("_process") or attr.startswith("_prepare"):
                                    dump_source(getattr(cls, attr), f"{name}.{attr}")
                            break  # Found it
        except ImportError:
            continue

    # 4. simple_denoising_func — the wrapper that calls transformer()
    section("simple_denoising_func")
    try:
        from ltx_pipelines.utils import simple_denoising_func
        dump_source(simple_denoising_func, "simple_denoising_func")
    except ImportError:
        try:
            from ltx_pipelines.utils.helpers import simple_denoising_func
            dump_source(simple_denoising_func, "simple_denoising_func")
        except ImportError:
            # Search in utils module
            try:
                import ltx_pipelines.utils as utils_mod
                names = [n for n in dir(utils_mod) if "denois" in n.lower()]
                print(f"  utils names matching denois: {names}")
                for n in names:
                    dump_source(getattr(utils_mod, n), f"utils.{n}")
            except Exception as exc:
                print(f"  Could not find: {exc}")

    # 5. euler_denoising_loop — the outer loop
    section("euler_denoising_loop")
    try:
        from ltx_pipelines.utils import euler_denoising_loop
        dump_source(euler_denoising_loop, "euler_denoising_loop")
    except ImportError:
        try:
            from ltx_pipelines.utils.helpers import euler_denoising_loop
            dump_source(euler_denoising_loop, "euler_denoising_loop")
        except ImportError:
            print("  Could not find euler_denoising_loop")

    # 6. EulerDiffusionStep — the step rule
    section("EulerDiffusionStep")
    try:
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        dump_source(EulerDiffusionStep, "EulerDiffusionStep")
    except ImportError:
        print("  Could not find EulerDiffusionStep")

    # 7. STAGE_2_DISTILLED_SIGMA_VALUES
    section("STAGE_2_DISTILLED_SIGMA_VALUES")
    try:
        from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES
        print(f"  STAGE_2_DISTILLED_SIGMA_VALUES = {STAGE_2_DISTILLED_SIGMA_VALUES}")
        print(f"  len = {len(STAGE_2_DISTILLED_SIGMA_VALUES)}")
    except ImportError:
        print("  Could not find")

    # 8. Read the full rope.py file directly for completeness
    section("Full rope.py file")
    try:
        from ltx_core.model.transformer import rope as rope_mod
        rope_file = inspect.getfile(rope_mod)
        print(f"  File: {rope_file}")
        with open(rope_file) as f:
            print(f.read())
    except Exception as exc:
        print(f"  Could not read: {exc}")


if __name__ == "__main__":
    main()
