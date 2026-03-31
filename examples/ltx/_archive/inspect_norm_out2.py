"""Inspect norm_out / proj_out - fixed version that prints forward source.

Usage:
    uv run python inspect_norm_out2.py /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors
"""

import inspect
import sys
import torch
from pathlib import Path
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

# --- norm_out details (safe) ---
print("\n=== norm_out ===")
no = vm.norm_out
print(f"  type:           {type(no).__name__}")
print(f"  repr:           {no}")
print(f"  _parameters:    {dict((k, (v.shape if v is not None else None)) for k, v in no._parameters.items())}")
print(f"  named_params:   {dict((k, v.shape) for k, v in no.named_parameters())}")

print("\n=== audio_norm_out ===")
ano = vm.audio_norm_out
print(f"  type:           {type(ano).__name__}")
print(f"  repr:           {ano}")
print(f"  _parameters:    {dict((k, (v.shape if v is not None else None)) for k, v in ano._parameters.items())}")

print("\n=== proj_out ===")
po = vm.proj_out
print(f"  type:           {type(po).__name__}")
print(f"  repr:           {po}")
print(f"  _parameters:    {dict((k, v.shape) for k, v in po.named_parameters())}")

print("\n=== audio_proj_out ===")
apo = vm.audio_proj_out
print(f"  type:           {type(apo).__name__}")
print(f"  _parameters:    {dict((k, v.shape) for k, v in apo.named_parameters())}")

# --- checkpoint keys for norm_out / proj_out ---
print("\n=== checkpoint keys matching norm_out / proj_out ===")
ckpt_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser())
ckpt = load_file(ckpt_path)
for key in sorted(ckpt.keys()):
    if "norm_out" in key or "proj_out" in key:
        print(f"  {key}  {ckpt[key].shape}  {ckpt[key].dtype}")

# --- velocity_model forward source ---
print("\n=== velocity_model forward source ===")
try:
    src = inspect.getsource(type(vm).forward)
    print(src)
except Exception as e:
    print(f"  Cannot get source: {e}")
    # Try finding the class source file
    try:
        import importlib
        print(f"  Module file: {inspect.getfile(type(vm))}")
    except Exception as e2:
        print(f"  Cannot find source file: {e2}")
