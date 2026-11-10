"""Decode and play normalized audio latents produced by the Zig executable.

From the repository root:
    examples/phonon/reference/.venv/bin/python examples/phonon/reference/decode.py phonon.normalized.safetensors

Use --output phonon.wav to also save a WAV, or --no-play for decoding only.
"""

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoProcessor


# Same TTS checkpoint revision as MODULE.bazel; the stats must match Zig's weights.
MODEL_ID = "gradium/phonon-31m-en"
MODEL_REVISION = "92741c5b07d085ff840e26fea8ce43110b87852f"


def load_normalized_latents(path):
    """Read the exact-length, single-prompt output of Generation in main.zig."""
    with safe_open(str(path), framework="pt", device="cpu") as tensors:
        metadata = tensors.metadata() or {}
        if not metadata.get("latent_space", "").startswith("normalized TTS"):
            raise ValueError("Expected normalized TTS latents, as labelled by the Zig output metadata")
        latents = tensors.get_tensor("audio_latents")
    if latents.ndim != 3 or latents.shape[0] != 1 or latents.shape[2] != 32 or latents.shape[1] == 0:
        raise ValueError(f"Expected nonempty audio_latents [1,frames,32], got {tuple(latents.shape)}")
    if latents.dtype != torch.float32 or not torch.isfinite(latents).all():
        raise ValueError("Expected finite f32 audio latents")
    return latents


def denormalize_latents(latents, checkpoint):
    """Match the official model's denormalize_latents without loading the TTS network."""
    with safe_open(str(checkpoint), framework="pt", device="cpu") as tensors:
        # These are the original checkpoint names, before Hugging Face renaming.
        mean = tensors.get_tensor("flow_lm.emb_mean").float()
        std = tensors.get_tensor("flow_lm.emb_std").float()
    if mean.shape != (32,) or std.shape != (32,):
        raise ValueError(f"Incompatible latent statistics: mean {tuple(mean.shape)}, std {tuple(std.shape)}")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("Invalid checkpoint latent statistics")
    # [1,frames,32] * [32] + [32]: broadcast per-feature statistics over frames.
    return (latents * std + mean).to(latents.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("latents", type=Path, help="Zig's phonon.normalized.safetensors output")
    parser.add_argument("--checkpoint", type=Path, help="Use a local original model.safetensors for latent statistics")
    parser.add_argument("--output", type=Path, help="Also save the decoded waveform to this WAV path")
    parser.add_argument("--no-play", action="store_true", help="Decode without playing audio")
    parser.add_argument("--local-files-only", action="store_true", help="Require Hugging Face assets to be cached already")
    args = parser.parse_args()
    if args.no_play and args.output is None:
        parser.error("--no-play requires --output so the decoded waveform is retained")

    normalized = load_normalized_latents(args.latents)
    checkpoint = args.checkpoint or hf_hub_download(
        MODEL_ID, "model.safetensors", revision=MODEL_REVISION, local_files_only=args.local_files_only
    )
    latents = denormalize_latents(normalized, checkpoint)
    print(f"Decoding {latents.shape[1]} generated frames with the official audio decoder...", flush=True)
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=args.local_files_only
    )
    processor.audio_tokenizer.eval()
    # The file contains only valid frames, so every frame is included.
    mask = torch.ones(latents.shape[:2], dtype=torch.long)
    with torch.inference_mode():
        waveform = processor.decode(latents, mask)[0]
    sample_rate = processor.feature_extractor.sampling_rate
    print(f"Decoded {len(waveform)} samples at {sample_rate} Hz ({len(waveform) / sample_rate:.2f} seconds).")
    if args.output is not None:
        processor.save_audio(waveform, args.output)
        print(f"Saved {args.output}")
    if not args.no_play:
        import sounddevice as sd

        print("Playing generated speech...", flush=True)
        sd.play(waveform.detach().cpu().float().numpy(), samplerate=sample_rate)
        sd.wait()
        print("Playback finished.")


if __name__ == "__main__":
    main()
