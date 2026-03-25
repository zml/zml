"""Inspect norm_out / proj_out structure in the ltx_core velocity model.

Usage:
    uv run python inspect_norm_out.py
"""

import inspect
import torch
import torch.nn as nn
from pathlib import Path

from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

checkpoint_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors").expanduser())
distilled_lora_path = str(Path("~/models/ltx-2.3/ltx-2.3-22b-distilled-lora-384.safetensors").expanduser())
gemma_root = str(Path("~/models/gemma-3-12b-it").expanduser())
spatial_upsampler_path = str(Path("~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors").expanduser())

print("Loading pipeline (this will take a while)...")
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

print("\n=== velocity_model named_modules (excluding transformer_blocks) ===")
for name, module in vm.named_modules():
    if not name.startswith("transformer_blocks") and name != "":
        print(f"  {name}: {type(module).__name__}  params={list(module._parameters.keys())}  children={list(k for k,_ in module.named_children())}")

print("\n=== norm_out submodule ===")
print(f"  type: {type(vm.norm_out).__name__}")
print(f"  params: {dict((k, v.shape) for k, v in vm.norm_out.named_parameters())}")
if hasattr(vm.norm_out, 'weight'):
    print(f"  weight: {vm.norm_out.weight.shape}")
if hasattr(vm.norm_out, 'bias') and vm.norm_out.bias is not None:
    print(f"  bias: {vm.norm_out.bias.shape}")
print(f"  repr: {vm.norm_out}")

print("\n=== audio_norm_out submodule ===")
if hasattr(vm, 'audio_norm_out'):
    print(f"  type: {type(vm.audio_norm_out).__name__}")
    print(f"  params: {dict((k, v.shape) for k, v in vm.audio_norm_out.named_parameters())}")

print("\n=== proj_out submodule ===")
print(f"  type: {type(vm.proj_out).__name__}")
if hasattr(vm.proj_out, 'weight'):
    print(f"  weight: {vm.proj_out.weight.shape}")
if hasattr(vm.proj_out, 'bias') and vm.proj_out.bias is not None:
    print(f"  bias: {vm.proj_out.bias.shape}")

print("\n=== velocity_model.forward source ===")
try:
    src = inspect.getsource(type(vm).forward)
    # Print only the final ~50 lines (after transformer blocks)
    lines = src.split('\n')
    print('\n'.join(lines))
except Exception as e:
    print(f"  Cannot get source: {e}")

print("\n=== velocity_model type ===")
print(f"  {type(vm)}")

print("\n=== ALL checkpoint keys for velocity_model (non-transformer_blocks) ===")
sd = vm.state_dict()
for key in sorted(sd.keys()):
    if not key.startswith("transformer_blocks"):
        print(f"  {key}  {sd[key].shape}  {sd[key].dtype}")
