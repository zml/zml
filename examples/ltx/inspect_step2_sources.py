"""Inspector script: dump source code of modality_from_latent_state and RoPE generation.

Usage (on GPU machine):
  cd /root/repos/LTX-2
  uv run /root/repos/zml/examples/ltx/inspect_step2_sources.py

Or if symlinked into scripts/:
  uv run ./scripts/inspect_step2_sources.py
"""

import inspect
import torch
from pathlib import Path

from ltx_core.types import LatentState
from ltx_pipelines.utils.helpers import modality_from_latent_state


def section(title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def dump_source(obj, label: str):
    """Print source code of a function or class."""
    try:
        src = inspect.getsource(obj)
        print(f"--- {label} (from {inspect.getfile(obj)}) ---")
        print(src)
    except Exception as exc:
        print(f"--- {label}: could not get source: {exc} ---")


def main():
    # 1. modality_from_latent_state
    section("modality_from_latent_state")
    dump_source(modality_from_latent_state, "modality_from_latent_state")

    # 2. LatentState type
    section("LatentState type")
    dump_source(LatentState, "LatentState")

    # 3. Modality type (return type of modality_from_latent_state)
    # Try to find it
    try:
        from ltx_core.types import Modality
        dump_source(Modality, "Modality")
    except ImportError:
        # Try to infer from return annotation
        sig = inspect.signature(modality_from_latent_state)
        print(f"  Signature: {sig}")
        ret = sig.return_annotation
        if ret is not inspect.Parameter.empty:
            print(f"  Return annotation: {ret}")
            if hasattr(ret, '__module__'):
                dump_source(ret, f"Return type: {ret}")

    # 4. RoPE generation — look for it in the velocity model
    section("RoPE generation (velocity_model forward)")
    try:
        from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

        # We don't need to load full model — just inspect the class
        # Find the velocity_model class
        try:
            # Try direct import
            from ltx_core.model.transformer.ltx_model import LTXMultiModalDiffusionTransformer
            dump_source(LTXMultiModalDiffusionTransformer.forward, "LTXMultiModalDiffusionTransformer.forward")
        except ImportError:
            pass

        # Also try the X0 wrapper
        try:
            from ltx_core.guidance.x0_model import X0Model
            dump_source(X0Model.forward, "X0Model.forward")
        except ImportError:
            pass

        # The transformer wrapper that calls velocity_model
        try:
            from ltx_core.guidance.x0_model import X0Model
            dump_source(X0Model.__call__, "X0Model.__call__")
        except (ImportError, TypeError):
            pass

    except Exception as exc:
        print(f"  Could not inspect velocity model classes: {exc}")

    # 5. Look for RoPE/rotary embedding generation
    section("RoPE frequency/position computation")
    try:
        # Common locations for RoPE generation in diffusion transformers
        modules_to_check = [
            "ltx_core.model.transformer.ltx_model",
            "ltx_core.model.transformer.rotary_embedding",
            "ltx_core.model.transformer.embeddings",
            "ltx_core.model.transformer.rope",
            "ltx_core.rope",
            "ltx_core.model.rope",
        ]
        for mod_name in modules_to_check:
            try:
                mod = __import__(mod_name, fromlist=[""])
                print(f"\n--- Module: {mod_name} ---")
                print(f"  File: {inspect.getfile(mod)}")
                # List all public names
                names = [n for n in dir(mod) if not n.startswith("_")]
                print(f"  Public names: {names}")
                # Dump anything that looks like RoPE
                for name in names:
                    obj = getattr(mod, name)
                    if callable(obj) and any(kw in name.lower() for kw in ["rope", "rotary", "position", "embed"]):
                        dump_source(obj, f"{mod_name}.{name}")
            except ImportError:
                pass

    except Exception as exc:
        print(f"  Error searching for RoPE: {exc}")

    # 6. TransformerArgs — what modality_from_latent_state produces
    section("TransformerArgs (block input type)")
    try:
        from ltx_core.model.transformer.ltx_model import TransformerArgs
        dump_source(TransformerArgs, "TransformerArgs")
    except ImportError:
        try:
            from ltx_core.types import TransformerArgs
            dump_source(TransformerArgs, "TransformerArgs")
        except ImportError:
            print("  Could not find TransformerArgs")

    # 7. Examine what 11_stage2_steps.pt actually contains
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
                    print(f"    {k:30s}  shape={tuple(v.shape):30s}  dtype={v.dtype}")
                else:
                    print(f"    {k:30s}  type={type(v).__name__}  value={v}")
    except Exception as exc:
        print(f"  Could not load 11_stage2_steps.pt: {exc}")


if __name__ == "__main__":
    main()
