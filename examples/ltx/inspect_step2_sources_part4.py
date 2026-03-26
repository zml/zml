"""Inspector part 4: args_preprocessor.prepare, TransformerArgs, X0Model, post_process_latent.

Usage (on GPU machine):
  cd /root/repos/LTX-2
  uv run scripts/inspect_step2_sources_part4.py
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
    # 1. TransformerArgs and ArgsPreprocessor
    section("TransformerArgs")
    search_modules = [
        "ltx_core.model.transformer.model",
        "ltx_core.model.transformer.types",
        "ltx_core.model.transformer.args",
        "ltx_core.model.transformer.block",
        "ltx_core.model.transformer",
    ]
    for mod_name in search_modules:
        try:
            mod = __import__(mod_name, fromlist=[""])
            if hasattr(mod, "TransformerArgs"):
                dump_source(mod.TransformerArgs, f"{mod_name}.TransformerArgs")
                break
        except ImportError:
            continue

    # 2. ArgsPreprocessor (video_args_preprocessor / audio_args_preprocessor)
    section("ArgsPreprocessor")
    for mod_name in search_modules:
        try:
            mod = __import__(mod_name, fromlist=[""])
            for name in dir(mod):
                if "preprocessor" in name.lower() or "ArgsPreprocessor" in name:
                    obj = getattr(mod, name)
                    if inspect.isclass(obj):
                        dump_source(obj, f"{mod_name}.{name}")
        except ImportError:
            continue

    # If not found, search more broadly
    section("ArgsPreprocessor (broader search)")
    broader = [
        "ltx_core.model.transformer.preprocessor",
        "ltx_core.model.transformer.args_preprocessor",
        "ltx_core.model.transformer.modality_preprocessor",
    ]
    for mod_name in broader:
        try:
            mod = __import__(mod_name, fromlist=[""])
            print(f"  Found module: {mod_name}")
            print(f"  Public names: {[n for n in dir(mod) if not n.startswith('_')]}")
            for name in dir(mod):
                if not name.startswith("_"):
                    obj = getattr(mod, name)
                    if inspect.isclass(obj) and hasattr(obj, "prepare"):
                        dump_source(obj, f"{mod_name}.{name}")
        except ImportError:
            continue

    # 3. LTXModel.__init__ to find preprocessor creation
    section("LTXModel.__init__")
    for mod_name in ["ltx_core.model.transformer.model", "ltx_core.model.transformer.ltx_model"]:
        try:
            mod = __import__(mod_name, fromlist=[""])
            for name in dir(mod):
                cls = getattr(mod, name, None)
                if inspect.isclass(cls) and hasattr(cls, "video_args_preprocessor"):
                    dump_source(cls.__init__, f"{name}.__init__")
                    break
                if inspect.isclass(cls) and name == "LTXModel":
                    dump_source(cls.__init__, f"{name}.__init__")
                    break
        except ImportError:
            continue

    # 4. X0Model
    section("X0Model")
    for mod_name in [
        "ltx_core.model.transformer.model",
        "ltx_core.model.transformer",
        "ltx_core.guidance.x0_model",
        "ltx_core.model.x0_model",
    ]:
        try:
            mod = __import__(mod_name, fromlist=[""])
            if hasattr(mod, "X0Model"):
                cls = mod.X0Model
                dump_source(cls, f"{mod_name}.X0Model (full class)")
                break
        except ImportError:
            continue

    # 5. post_process_latent
    section("post_process_latent")
    for mod_name in [
        "ltx_pipelines.utils.helpers",
        "ltx_pipelines.utils.samplers",
        "ltx_pipelines.utils",
        "ltx_core.components.diffusion_steps",
    ]:
        try:
            mod = __import__(mod_name, fromlist=[""])
            if hasattr(mod, "post_process_latent"):
                dump_source(mod.post_process_latent, f"{mod_name}.post_process_latent")
                break
        except ImportError:
            continue

    # 6. to_velocity (used in EulerDiffusionStep.step)
    section("to_velocity")
    for mod_name in [
        "ltx_core.components.diffusion_steps",
        "ltx_core.components",
    ]:
        try:
            mod = __import__(mod_name, fromlist=[""])
            if hasattr(mod, "to_velocity"):
                dump_source(mod.to_velocity, f"{mod_name}.to_velocity")
                break
        except ImportError:
            continue

    # 7. LTXModelConfigurator (might create the preprocessors)
    section("LTXModelConfigurator")
    for mod_name in ["ltx_core.model.transformer.model", "ltx_core.model.transformer"]:
        try:
            mod = __import__(mod_name, fromlist=[""])
            if hasattr(mod, "LTXModelConfigurator"):
                cls = mod.LTXModelConfigurator
                # Just the method that creates args preprocessors
                for attr in dir(cls):
                    if "preprocessor" in attr.lower() or "prepare" in attr.lower() or "create" in attr.lower():
                        dump_source(getattr(cls, attr), f"LTXModelConfigurator.{attr}")
                # Also __init__ / configure
                if hasattr(cls, "configure"):
                    dump_source(cls.configure, "LTXModelConfigurator.configure")
                if hasattr(cls, "__call__"):
                    dump_source(cls.__call__, "LTXModelConfigurator.__call__")
                dump_source(cls.__init__, "LTXModelConfigurator.__init__")
                break
        except ImportError:
            continue

    # 8. Find prepare method by searching LTXModel attributes at runtime
    section("LTXModel attribute types (introspect class)")
    for mod_name in ["ltx_core.model.transformer.model"]:
        try:
            mod = __import__(mod_name, fromlist=[""])
            cls = mod.LTXModel
            # Check type annotations
            if hasattr(cls, "__annotations__"):
                print(f"  Annotations: {cls.__annotations__}")
            # Check init signature
            sig = inspect.signature(cls.__init__)
            print(f"  __init__ signature: {sig}")
        except (ImportError, Exception) as exc:
            print(f"  Error: {exc}")


if __name__ == "__main__":
    main()
