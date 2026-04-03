#!/usr/bin/env python3
"""Export per-stage vocoder intermediate values for debugging.

Traces through the first few stages of the main vocoder and saves intermediates.

Usage:
  uv run python export_vocoder_stages.py \
      --checkpoint /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
      --activations /root/e2e_demo/vocoder_ref/vocoder_activations.safetensors \
      --output /root/e2e_demo/vocoder_ref/vocoder_stages.safetensors
"""

import argparse
import sys
import json

import torch
import einops
import safetensors
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sys.path.insert(0, "/root/repos/LTX-2/packages/ltx-core/src")
    sys.path.insert(0, "/root/repos/LTX-2/packages/ltx-pipelines/src")
    from ltx_core.model.audio_vae.vocoder import Vocoder

    # Load config
    with safetensors.safe_open(args.checkpoint, framework="pt") as f:
        config = json.loads(f.metadata()["config"])
        state_dict = {}
        for key in f.keys():
            if key.startswith("vocoder.vocoder."):
                local_key = key[len("vocoder.vocoder."):]
                state_dict[local_key] = f.get_tensor(key)

    voc_config = config["vocoder"]["vocoder"]
    vocoder = Vocoder(
        resblock_kernel_sizes=voc_config["resblock_kernel_sizes"],
        upsample_rates=voc_config["upsample_rates"],
        upsample_kernel_sizes=voc_config["upsample_kernel_sizes"],
        resblock_dilation_sizes=voc_config["resblock_dilation_sizes"],
        upsample_initial_channel=voc_config["upsample_initial_channel"],
        resblock=voc_config["resblock"],
        activation=voc_config["activation"],
        use_tanh_at_final=voc_config["use_tanh_at_final"],
        use_bias_at_final=voc_config["use_bias_at_final"],
    )
    vocoder.load_state_dict(state_dict, strict=True)
    vocoder = vocoder.cuda().eval()

    # Load input mel
    with safetensors.safe_open(args.activations, framework="pt") as f:
        input_mel = f.get_tensor("input_mel").cuda()

    print(f"Input mel: {list(input_mel.shape)} {input_mel.dtype}")

    tensors = {}

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float32):
        mel_f32 = input_mel.float()

        # Rearrange (same as Vocoder.forward)
        x = mel_f32.transpose(2, 3)  # [B, 2, 64, T]
        x = einops.rearrange(x, "b s c t -> b (s c) t")  # [B, 128, T]
        tensors["after_rearrange"] = x.cpu().float()
        print(f"After rearrange: {list(x.shape)}")
        print(f"  x[0,0,:8] = {x[0,0,:8].tolist()}")

        # conv_pre
        x = vocoder.conv_pre(x)
        tensors["after_conv_pre"] = x.cpu().float()
        print(f"After conv_pre: {list(x.shape)}")
        print(f"  range: [{x.min().item():.6f}, {x.max().item():.6f}]")
        print(f"  x[0,0,:8] = {x[0,0,:8].tolist()}")

        # First upsample (ups[0])
        x = vocoder.ups[0](x)
        tensors["after_ups0"] = x.cpu().float()
        print(f"After ups[0]: {list(x.shape)}")
        print(f"  range: [{x.min().item():.6f}, {x.max().item():.6f}]")
        print(f"  x[0,0,:8] = {x[0,0,:8].tolist()}")

        # First resblock stage (blocks 0,1,2 → mean)
        block_outputs = torch.stack(
            [vocoder.resblocks[idx](x) for idx in range(3)],
            dim=0,
        )
        x = block_outputs.mean(dim=0)
        tensors["after_stage0_resblocks"] = x.cpu().float()
        print(f"After stage0 resblocks mean: {list(x.shape)}")
        print(f"  range: [{x.min().item():.6f}, {x.max().item():.6f}]")
        print(f"  x[0,0,:8] = {x[0,0,:8].tolist()}")

    save_file(tensors, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
