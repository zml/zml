#!/usr/bin/env python3
"""Export BWE pipeline intermediate values for debugging Stage 2.

Runs through VocoderWithBWE step-by-step and exports:
  - bwe_input_padded: padded 16kHz waveform fed to BWE  [1, 2, T_padded]
  - bwe_mel:          log-mel from _compute_mel            [1, 2, T_frames, 64]
  - bwe_residual:     BWE generator output                 [1, 2, T_bwe]
  - bwe_skip:         sinc-resampled skip connection       [1, 2, T_skip]
  - bwe_output:       final 48kHz waveform                 [1, 2, T_out]

Also exports bwe_stft_magnitude for deeper STFT debugging.

Usage:
  cd /root/repos/LTX-2 && uv run python /root/repos/zml/examples/ltx/e2e/export_bwe_stages.py \
      --checkpoint /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
      --activations /root/e2e_demo/vocoder_ref/vocoder_activations.safetensors \
      --output /root/e2e_demo/vocoder_ref/bwe_stages.safetensors
"""

import argparse
import json
import sys

import torch
import numpy as np
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

    from ltx_core.model.audio_vae.vocoder import VocoderWithBWE, Vocoder, MelSTFT, UpSample1d

    # Load checkpoint config
    with safetensors.safe_open(args.checkpoint, framework="pt") as f:
        config = json.loads(f.metadata()["config"])
        state_dict = {}
        for key in f.keys():
            if key.startswith("vocoder."):
                local_key = key[len("vocoder."):]
                state_dict[local_key] = f.get_tensor(key)

    voc_config = config["vocoder"]["vocoder"]
    bwe_config = config["vocoder"]["bwe"]

    # Build model
    print("Building VocoderWithBWE model...")
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

    bwe_generator = Vocoder(
        resblock_kernel_sizes=bwe_config["resblock_kernel_sizes"],
        upsample_rates=bwe_config["upsample_rates"],
        upsample_kernel_sizes=bwe_config["upsample_kernel_sizes"],
        resblock_dilation_sizes=bwe_config["resblock_dilation_sizes"],
        upsample_initial_channel=bwe_config["upsample_initial_channel"],
        resblock=bwe_config["resblock"],
        activation=bwe_config["activation"],
        use_tanh_at_final=bwe_config.get("use_tanh_at_final", False),
        apply_final_activation=bwe_config.get("apply_final_activation", False),
        use_bias_at_final=bwe_config.get("use_bias_at_final", False),
    )

    mel_stft = MelSTFT(
        filter_length=bwe_config["n_fft"],
        hop_length=bwe_config["hop_length"],
        win_length=bwe_config["win_size"],
        n_mel_channels=bwe_config["num_mels"],
    )

    model = VocoderWithBWE(
        vocoder=vocoder,
        bwe_generator=bwe_generator,
        mel_stft=mel_stft,
        input_sampling_rate=bwe_config["input_sampling_rate"],
        output_sampling_rate=bwe_config["output_sampling_rate"],
        hop_length=bwe_config["hop_length"],
    )

    model.load_state_dict(state_dict, strict=True)
    model = model.cuda().eval()
    print("  Model loaded.")

    # Load input mel
    with safetensors.safe_open(args.activations, framework="pt") as f:
        input_mel = f.get_tensor("input_mel").cuda()
    print(f"Input mel: {list(input_mel.shape)} {input_mel.dtype}")

    tensors = {}

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float32):
        # --- Stage 1: Main vocoder (16kHz) ---
        waveform_16k = model.vocoder(input_mel.float())
        print(f"16kHz waveform: {list(waveform_16k.shape)} range=[{waveform_16k.min():.4f}, {waveform_16k.max():.4f}]")
        tensors["bwe_input_raw"] = waveform_16k.cpu().float()

        # --- Stage 2: BWE pipeline, step by step ---
        x = waveform_16k
        length_low_rate = x.shape[-1]
        sr_ratio = model.output_sampling_rate // model.input_sampling_rate  # 3
        output_length = length_low_rate * sr_ratio
        hop_length = model.hop_length  # 80

        # Step 2a: Pad to multiple of hop_length
        remainder = length_low_rate % hop_length
        if remainder:
            pad_amount = hop_length - remainder
            x = torch.nn.functional.pad(x, (0, pad_amount))
            print(f"  Padded by {pad_amount}: {list(x.shape)}")
        else:
            print(f"  No padding needed: {list(x.shape)}")
        tensors["bwe_input_padded"] = x.cpu().float()

        # Step 2b: Compute mel — manual to also capture STFT magnitude
        B, S, T = x.shape
        x_flat = x.reshape(B * S, T)

        # STFT — stft_fn returns a tuple; inspect and extract magnitude
        stft_result = model.mel_stft.stft_fn(x_flat)
        if isinstance(stft_result, tuple):
            stft_magnitude = stft_result[0]  # first element is magnitude
            print(f"  STFT returned tuple of {len(stft_result)} elements")
            for i, t in enumerate(stft_result):
                if hasattr(t, 'shape'):
                    print(f"    [{i}]: shape={list(t.shape)} range=[{t.min():.6f}, {t.max():.6f}]")
                else:
                    print(f"    [{i}]: {type(t)}")
        else:
            stft_magnitude = stft_result
        print(f"  STFT magnitude: {list(stft_magnitude.shape)} range=[{stft_magnitude.min():.6f}, {stft_magnitude.max():.6f}]")
        tensors["bwe_stft_magnitude"] = stft_magnitude.cpu().float()

        # Mel projection + log
        mel_proj = model.mel_stft.mel_basis @ stft_magnitude
        mel_log = torch.log(torch.clamp(mel_proj, min=1e-5))
        mel_log = mel_log.reshape(B, S, mel_log.shape[1], mel_log.shape[2])  # [B, 2, 64, T_frames]
        print(f"  log-mel (before transpose): {list(mel_log.shape)} range=[{mel_log.min():.4f}, {mel_log.max():.4f}]")
        tensors["bwe_mel_pre_transpose"] = mel_log.cpu().float()

        # _compute_mel returns [B, 2, n_mels, T_frames]; BWE generator expects [B, 2, T_frames, n_mels]
        mel_raw = model._compute_mel(x)  # [B, 2, 64, T_frames]
        mel_for_bwe = mel_raw.transpose(2, 3)  # [B, 2, T_frames, 64]
        print(f"  mel_for_bwe (after transpose): {list(mel_for_bwe.shape)} range=[{mel_for_bwe.min():.4f}, {mel_for_bwe.max():.4f}]")
        tensors["bwe_mel"] = mel_for_bwe.cpu().float().contiguous()

        # Step 2c: BWE generator — residual waveform
        residual = model.bwe_generator(mel_for_bwe)
        print(f"  BWE residual: {list(residual.shape)} range=[{residual.min():.6f}, {residual.max():.6f}]")
        tensors["bwe_residual"] = residual.cpu().float()

        # Step 2d: Sinc resample skip connection
        skip = model.resampler(x)
        print(f"  BWE skip: {list(skip.shape)} range=[{skip.min():.6f}, {skip.max():.6f}]")
        tensors["bwe_skip"] = skip.cpu().float()

        # Step 2e: Final = residual + skip, clamp, trim
        output = residual + skip
        output = output.clamp(-1, 1)
        output = output[..., :output_length]
        print(f"  BWE output: {list(output.shape)} range=[{output.min():.6f}, {output.max():.6f}]")
        tensors["bwe_output"] = output.cpu().float()

    # Save
    save_file(tensors, args.output)
    print(f"\nSaved {len(tensors)} tensors to {args.output}:")
    for k, v in sorted(tensors.items()):
        print(f"  {k}: {list(v.shape)} {v.dtype}")


if __name__ == "__main__":
    main()
