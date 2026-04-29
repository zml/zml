#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
import wave
import torch
from diffusers import AutoencoderOobleck
from safetensors.torch import save_file

def load_wav_stereo_bct(wav_path: str) -> torch.Tensor:
    with wave.open(wav_path) as wav_file:
        num_channels = wav_file.getnchannels()
        num_frames = wav_file.getnframes()
        raw = wav_file.readframes(num_frames)

    audio_i16 = torch.frombuffer(raw, dtype=torch.int16).clone()
    audio_tc = audio_i16.view(num_frames, num_channels).to(torch.float32) / 32768.0
    audio_bct = audio_tc.transpose(0, 1).unsqueeze(0).contiguous()
    return audio_bct

def build_vae_from_local_files(model_path: str) -> AutoencoderOobleck:
    vae = AutoencoderOobleck.from_pretrained(model_path, local_files_only=True)
    vae.eval()
    return vae

def main() -> None:
    model_path = "//Users//sboulmier//zml//examples//acestep//models//Oobleck-vae//"
    wav_path = "//Users//sboulmier//zml//examples//acestep//models//Oobleck-vae//example.wav"
    output_path = "//Users//sboulmier//zml//examples//acestep//models//Oobleck-vae//example_encoded.safetensors"

    vae = build_vae_from_local_files(model_path)
    audio_bct = load_wav_stereo_bct(wav_path)
    print("audio dim", audio_bct.shape)
    audio_bct = audio_bct[:, :, 0:192000]
    print("audio dim", audio_bct.shape)

    with torch.no_grad():
        posterior = vae.encode(audio_bct).latent_dist
        latents_bct = posterior.mode().detach().cpu()

    latents_tc = latents_bct.squeeze(0).transpose(0,1).contiguous().clone()

    save_file(
        {
            "audio_bct": audio_bct.contiguous().detach().cpu(),
            "latents_bct": latents_bct.contiguous(),
            "latents_tc": latents_tc,
        },
        output_path,
    )

    print(f"Loaded VAE from: {model_path}")
    print(f"Loaded audio from: {wav_path}")
    print(f"Audio shape [B, C, T]: {tuple(audio_bct.shape)}")
    print(f"Encoded latents shape [B, C, T]: {tuple(latents_bct.shape)}")
    print(f"Encoded latents shape [T, C]: {tuple(latents_tc.shape)}")
    print(f"Saved encoded tensors to: {output_path}")


if __name__ == "__main__":
    main()
