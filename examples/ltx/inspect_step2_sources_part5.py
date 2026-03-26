"""Inspector part 5: to_denoised + _init_preprocessors + _init_video/_init_audio.

Usage (on GPU machine):
  cd /root/repos/LTX-2
  uv run scripts/inspect_step2_sources_part5.py
"""

import inspect


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
    # 1. to_denoised (used by X0Model.forward)
    section("to_denoised")
    for mod_name in [
        "ltx_core.utils",
        "ltx_core.components.diffusion_steps",
        "ltx_core.model.transformer.model",
    ]:
        try:
            mod = __import__(mod_name, fromlist=[""])
            if hasattr(mod, "to_denoised"):
                dump_source(mod.to_denoised, f"{mod_name}.to_denoised")
                break
        except ImportError:
            continue

    # 2. _init_preprocessors (on LTXModel)
    section("LTXModel._init_preprocessors")
    try:
        from ltx_core.model.transformer.model import LTXModel
        dump_source(LTXModel._init_preprocessors, "LTXModel._init_preprocessors")
    except Exception as exc:
        print(f"  Error: {exc}")

    # 3. _init_video and _init_audio (for adaln / patchify setup)
    section("LTXModel._init_video")
    try:
        from ltx_core.model.transformer.model import LTXModel
        dump_source(LTXModel._init_video, "LTXModel._init_video")
    except Exception as exc:
        print(f"  Error: {exc}")

    section("LTXModel._init_audio")
    try:
        from ltx_core.model.transformer.model import LTXModel
        dump_source(LTXModel._init_audio, "LTXModel._init_audio")
    except Exception as exc:
        print(f"  Error: {exc}")

    section("LTXModel._init_audio_video")
    try:
        from ltx_core.model.transformer.model import LTXModel
        dump_source(LTXModel._init_audio_video, "LTXModel._init_audio_video")
    except Exception as exc:
        print(f"  Error: {exc}")

    # 4. AdaLayerNormSingle.__init__ and forward (to confirm signature)
    section("AdaLayerNormSingle (Python)")
    try:
        from ltx_core.model.transformer.model import LTXModel
        # Find it from the velocity model
        for mod_name in [
            "ltx_core.model.transformer.ada_layer_norm",
            "ltx_core.model.transformer.norm",
            "ltx_core.model.transformer",
        ]:
            try:
                mod = __import__(mod_name, fromlist=[""])
                if hasattr(mod, "AdaLayerNormSingle"):
                    dump_source(mod.AdaLayerNormSingle, f"{mod_name}.AdaLayerNormSingle")
                    break
            except ImportError:
                continue
    except Exception as exc:
        print(f"  Error: {exc}")

    # 5. Actual checkpoint config — inspect what configurator produces for 22b
    section("LTX-22b actual model config")
    try:
        from ltx_core.model.transformer.model import LTXModel
        from ltx_core.loader import ModelLedger
        # Check if there's a config file
        from pathlib import Path
        ckpt_path = Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser()
        # Try loading just the config metadata
        from safetensors import safe_open
        with safe_open(str(ckpt_path), framework="pt") as f:
            metadata = f.metadata()
            if metadata:
                print("  Checkpoint metadata keys:", list(metadata.keys())[:20])
                for k, v in sorted(metadata.items()):
                    if len(str(v)) < 200:
                        print(f"    {k}: {v}")
                    else:
                        print(f"    {k}: <{len(str(v))} chars>")
    except Exception as exc:
        print(f"  Could not read config: {exc}")


if __name__ == "__main__":
    main()
