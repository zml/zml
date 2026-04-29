#!/usr/bin/env python3

from __future__ import annotations
import soundfile as sf
import torch
from diffusers import AutoencoderOobleck
from safetensors.torch import load_file
import numpy as np

def write_float_wav_no_struct(path, audio_tensor, sample_rate):
    # arg is of shape [B, C, T]
    # convert to [T, C]
    audio_tensor = audio_tensor.squeeze(0).detach().cpu().contiguous()
    audio_tensor = audio_tensor.transpose(0, 1).contiguous().numpy()
    
    audio_data = audio_tensor.astype(np.float32).tobytes()
    num_channels = audio_tensor.shape[1]
    num_frames = audio_tensor.shape[0]

    # Manually defining header components using NumPy dtypes
    def p32(val): return np.array([val], dtype='<u4').tobytes() # 4-byte unsigned int
    def p16(val): return np.array([val], dtype='<u2').tobytes() # 2-byte unsigned int

    header = [
        b'RIFF',
        p32(36 + 12 + len(audio_data)),
        b'WAVE',
        b'fmt ', p32(16), p16(3), p16(num_channels), p32(sample_rate),
        p32(sample_rate * num_channels * 4), p16(num_channels * 4), p16(32),
        b'fact', p32(4), p32(num_frames),
        b'data', p32(len(audio_data))
    ]
    
    with open(path, 'wb') as f:
        f.write(b''.join(header))
        f.write(audio_data)

def save_stereo_wav(audio_bct: torch.Tensor, wav_path: str, sample_rate: int) -> None:
    if audio_bct.ndim != 3:
        raise ValueError(f"Expected audio tensor with shape [B, C, T], got {tuple(audio_bct.shape)}")
    if audio_bct.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got {audio_bct.shape[0]}")
    if audio_bct.shape[1] != 2:
        raise ValueError(f"Expected 2 audio channels, got {audio_bct.shape[1]}")
    audio_ct = audio_bct.squeeze(0).detach().cpu().contiguous()
    audio_tc = audio_ct.transpose(0, 1).contiguous().numpy()
    sf.write(wav_path, audio_tc, sample_rate, format="WAV", subtype="FLOAT")


def load_latents_bct(encoded_path: str) -> torch.Tensor:
    tensors = load_file(encoded_path)
    if "diffused_latents" in tensors:
        latents_bct = tensors["diffused_latents"]
    else:
        raise KeyError("Could not find 'diffused_latents' in example_encoded.safetensors")
    latents_bct = latents_bct.unsqueeze(0)
    print("shape", latents_bct.shape)
    if latents_bct.ndim != 3:
        raise ValueError(f"Expected latent tensor with shape [B, C, T], got {tuple(latents_bct.shape)}")
        
    if "decoded_latents" in tensors:
        decoded_bct = tensors["decoded_latents"]
    else:
        raise KeyError("Could not find 'decoded_latents' in example_encoded.safetensors")
        
    print("max", torch.max(latents_bct))
    print("min", torch.min(latents_bct))
    return latents_bct.to(torch.float32).contiguous(), decoded_bct.to(torch.float32).contiguous()

def build_vae(model_path: str) -> AutoencoderOobleck:
    vae = AutoencoderOobleck.from_pretrained(model_path, local_files_only=True)
    vae.eval()
    return vae

def main() -> None:
    model_path = "//Users//sboulmier//zml//examples//acestep//models//Oobleck-vae//"
    encoded_path = "//Users//sboulmier//zml//examples//acestep//models//Oobleck-vae//diffused_latents4.safetensors"
    output_path = "//Users//sboulmier//zml//examples//acestep//models//Oobleck-vae//diffused_decoded_py.wav"
    vae = build_vae(model_path)
    latents_bct, decoded_bct = load_latents_bct(encoded_path)
    with torch.no_grad():
        decoded = vae.decode(latents_bct).sample.detach().cpu()
    print("max", torch.max(decoded))
    print("min", torch.min(decoded))
    
    peak = decoded.abs().amax(dim=[1, 2], keepdim=True)
    if torch.any(peak > 1.0):
        decoded = decoded / peak.clamp(min=1.0)
        
    diff = decoded - decoded_bct
    diff = diff.abs()
    print("max diff", torch.max(diff))
    
    write_float_wav_no_struct(output_path, decoded, vae.config.sampling_rate)
    #save_stereo_wav(decoded, output_path, vae.config.sampling_rate)
    print(f"Loaded VAE from: {model_path}")
    print(f"Loaded latents from: {encoded_path}")
    print(f"Latents shape [B, C, T]: {tuple(latents_bct.shape)}")
    print(f"Decoded audio shape [B, C, T]: {tuple(decoded.shape)}")
    print(f"Saved decoded wav to: {output_path}")

if __name__ == "__main__":
    main()
