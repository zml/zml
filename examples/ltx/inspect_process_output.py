"""Inspect _process_output source and model-level scale_shift_table.

Usage:
    uv run python scripts/inspect_process_output.py
"""

import inspect
from pathlib import Path
import torch
from safetensors.torch import load_file

from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

checkpoint_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser())
distilled_lora_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled-lora-384.safetensors").expanduser())
gemma_root = str(Path("~/models/gemma-3-12b-it").expanduser())
spatial_upsampler_path = str(Path("~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors").expanduser())

print("Loading pipeline...")
pipeline = TI2VidTwoStagesPipeline(
    checkpoint_path=checkpoint_path,
    distilled_lora=[LoraPathStrengthAndSDOps(
        path=distilled_lora_path, strength=0.8,
        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP
    )],
    spatial_upsampler_path=spatial_upsampler_path,
    gemma_root=gemma_root,
    loras=[],
    quantization=None,
)

transformer = pipeline.stage_2_model_ledger.transformer()
vm = transformer.velocity_model

# --- _process_output source ---
print("\n=== velocity_model._process_output source ===")
try:
    src = inspect.getsource(type(vm)._process_output)
    print(src)
except Exception as e:
    print(f"  Cannot get source: {e}")
    # Try parent class
    for cls in type(vm).__mro__:
        if hasattr(cls, '_process_output') and '_process_output' in cls.__dict__:
            print(f"  Found in {cls}")
            print(inspect.getsource(cls._process_output))
            break

# --- model-level buffers (scale_shift_table might be a buffer not a parameter) ---
print("\n=== velocity_model named_buffers (non-transformer_blocks) ===")
for name, buf in vm.named_buffers():
    if not name.startswith("transformer_blocks"):
        print(f"  {name}: {buf.shape}  {buf.dtype}")

# --- checkpoint keys: all non-transformer_blocks ---
print("\n=== checkpoint keys (non-transformer_blocks, non-adaln-single, non-av-ca) ===")
ckpt = load_file(str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser()))
for key in sorted(ckpt.keys()):
    if "transformer_blocks" in key:
        continue
    subkey = key.replace("model.diffusion_model.", "")
    if any(x in subkey for x in ["adaln_single", "av_ca", "audio_adaln", "prompt_adaln"]):
        continue
    print(f"  {key}  {ckpt[key].shape}  {ckpt[key].dtype}")
