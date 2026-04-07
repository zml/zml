#!/usr/bin/env python3
"""Export vocoder reference activations for ZML validation.

Runs the VocoderWithBWE on a mel spectrogram (either from a saved audio VAE output
or synthesized) and saves:
  - input_mel: [1, 2, T, 64] bf16 mel spectrogram input
  - ref_waveform: [1, 2, T_audio] f32 reference waveform output

Usage:
  uv run python export_vocoder_activations.py \
      --checkpoint /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
      --audio-mel /root/e2e_demo/audio_vae_zig_out/decoded_audio.bin \
      --mel-shape 1,2,501,64 \
      --output /root/e2e_demo/vocoder_ref/vocoder_activations.safetensors

Or with a small synthetic test:
  uv run python export_vocoder_activations.py \
      --checkpoint /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
      --small-test \
      --output /root/e2e_demo/vocoder_ref/vocoder_activations.safetensors
"""

import argparse
import struct
import sys
from pathlib import Path

import torch
import numpy as np
from safetensors.torch import save_file


def load_bf16_bin(path: str, shape: list[int]) -> torch.Tensor:
    """Load raw bf16 binary file as torch tensor."""
    with open(path, "rb") as f:
        raw = f.read()
    n_elements = 1
    for s in shape:
        n_elements *= s
    expected = n_elements * 2  # bf16 = 2 bytes
    assert len(raw) == expected, f"Expected {expected} bytes, got {len(raw)}"
    # Convert bf16 raw bytes to float32 via uint16 shift
    arr_u16 = np.frombuffer(raw, dtype=np.uint16)
    arr_u32 = arr_u16.astype(np.uint32) << 16
    arr_f32 = arr_u32.view(np.float32)
    return torch.from_numpy(arr_f32.reshape(shape)).to(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser(description="Export vocoder activations for ZML validation")
    parser.add_argument("--checkpoint", required=True, help="Path to safetensors checkpoint")
    parser.add_argument("--audio-mel", help="Path to decoded audio mel .bin (bf16)")
    parser.add_argument("--mel-shape", help="Shape of mel, e.g. 1,2,501,64")
    parser.add_argument("--small-test", action="store_true", help="Use small synthetic mel (T=8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic test")
    parser.add_argument("--output", required=True, help="Output safetensors path")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Add LTX-2 packages to path
    sys.path.insert(0, "/root/repos/LTX-2/packages/ltx-core/src")
    sys.path.insert(0, "/root/repos/LTX-2/packages/ltx-pipelines/src")

    from ltx_core.model.audio_vae.vocoder import VocoderWithBWE, Vocoder, MelSTFT, UpSample1d
    import safetensors

    # Load checkpoint config
    with safetensors.safe_open(args.checkpoint, framework="pt") as f:
        meta = f.metadata()
    import json
    config = json.loads(meta["config"])
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

    # Load weights
    print("Loading weights from checkpoint...")
    with safetensors.safe_open(args.checkpoint, framework="pt") as f:
        state_dict = {}
        for key in f.keys():
            if key.startswith("vocoder."):
                local_key = key[len("vocoder."):]
                state_dict[local_key] = f.get_tensor(key)

    model.load_state_dict(state_dict, strict=True)
    model = model.cuda().eval()
    print(f"  Loaded {len(state_dict)} tensors")

    # Prepare input mel
    if args.small_test:
        T = 8
        input_mel = torch.randn(1, 2, T, 64, dtype=torch.bfloat16, device="cuda")
        print(f"  Using synthetic mel: {list(input_mel.shape)}")
    elif args.audio_mel:
        shape = [int(x) for x in args.mel_shape.split(",")]
        input_mel = load_bf16_bin(args.audio_mel, shape).cuda()
        print(f"  Loaded mel from {args.audio_mel}: {list(input_mel.shape)}")
    else:
        parser.error("Provide --audio-mel or --small-test")
        return

    # Run forward
    print("Running VocoderWithBWE forward...")
    with torch.no_grad():
        # Run main vocoder (stage 1) separately to capture intermediate
        # VocoderWithBWE.forward wraps in autocast(float32) and calls vocoder(mel.float())
        with torch.autocast(device_type=input_mel.device.type, dtype=torch.float32):
            waveform_16k = model.vocoder(input_mel.float())
        print(f"  Intermediate 16kHz waveform: {list(waveform_16k.shape)} {waveform_16k.dtype}")
        print(f"  16kHz range: [{waveform_16k.min().item():.4f}, {waveform_16k.max().item():.4f}]")

        # Run BWE pipeline (stage 2) via the full model
        waveform = model(input_mel)
    print(f"  Output waveform: {list(waveform.shape)} {waveform.dtype}")
    print(f"  Waveform range: [{waveform.min().item():.4f}, {waveform.max().item():.4f}]")

    # Save
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors = {
        "input_mel": input_mel.cpu().contiguous(),
        "ref_waveform_16k": waveform_16k.cpu().float().contiguous(),
        "ref_waveform": waveform.cpu().float().contiguous(),
    }
    save_file(tensors, args.output)
    print(f"  Saved to {args.output}")
    for k, v in tensors.items():
        print(f"    {k}: {list(v.shape)} {v.dtype}")


if __name__ == "__main__":
    main()
