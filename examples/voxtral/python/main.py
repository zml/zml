#!/usr/bin/env python3
"""
Reference mel spectrogram implementation for Voxtral.
Outputs intermediate values for comparison with the Zig implementation.

Usage:
    pip install numpy soundfile
    python main.py <audio.wav>
"""

import sys

import numpy as np
import soundfile as sf
import torch

# Audio parameters (from Voxtral config / params.json)
SAMPLE_RATE = 16000
NUM_MEL_BINS = 128
HOP_LENGTH = 160
WINDOW_SIZE = 400  # also used as n_fft
GLOBAL_LOG_MEL_MAX = 1.5


# ==========================================================================
# Mel filter bank (Slaney-style, from mistral_common/audio.py)
# ==========================================================================
def hertz_to_mel(freq):
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0
    if isinstance(freq, np.ndarray):
        log_region = freq >= min_log_hertz
        mels[log_region] = (
            min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
        )
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep
    return mels


def mel_to_hertz(mels):
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    log_region = mels >= min_log_mel
    freq[log_region] = min_log_hertz * np.exp(
        logstep * (mels[log_region] - min_log_mel)
    )
    return freq


def compute_mel_filters():
    num_frequency_bins = 1 + WINDOW_SIZE // 2  # 201
    fft_freqs = np.linspace(0, SAMPLE_RATE // 2, num_frequency_bins)
    mel_min = hertz_to_mel(0.0)
    mel_max = hertz_to_mel(8000.0)
    mel_freqs = np.linspace(mel_min, mel_max, NUM_MEL_BINS + 2)
    filter_freqs = mel_to_hertz(mel_freqs)
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    fb = np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))
    enorm = 2.0 / (filter_freqs[2 : NUM_MEL_BINS + 2] - filter_freqs[:NUM_MEL_BINS])
    fb *= np.expand_dims(enorm, 0)
    return fb  # [201, 128]


# ============================================================================
# Mel spectrogram (from vLLM voxtral.py compute_whisper_melspec)
# ============================================================================


def compute_mel_spectrogram(audio, mel_filters):
    """audio: 1D tensor, mel_filters: [freq_bins, mel_bins] tensor"""
    window = torch.hann_window(WINDOW_SIZE)
    stft = torch.stft(
        audio, WINDOW_SIZE, HOP_LENGTH, window=window, return_complex=True
    )
    magnitudes = stft[..., :-1].abs() ** 2
    mel_spec = mel_filters.T @ magnitudes  # [mel_bins, frames]
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    # Use fixed global_log_mel_max from params.json (matching vLLM compute_whisper_melspec
    # which uses config.global_log_mel_max when set, falling back to dynamic max otherwise)
    log_spec = torch.maximum(log_spec, torch.tensor(GLOBAL_LOG_MEL_MAX) - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec  # [128, frames]


# ==========================================================================
# Main
# ==========================================================================


def execute(path):
    # Load audio
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import soxr

        audio = soxr.resample(audio, sr, SAMPLE_RATE, quality="HQ")

    print(
        f"Audio: {len(audio)} samples, {len(audio) / SAMPLE_RATE:.2f}s", file=sys.stderr
    )

    # Compute mel filters (numpy) and convert to torch
    mel_filters_np = compute_mel_filters()
    mel_filters = torch.tensor(mel_filters_np, dtype=torch.float32)
    print(f"Mel filters: {mel_filters.shape}", file=sys.stderr)
    print(
        f"  min={mel_filters.min():.6f}, max={mel_filters.max():.6f}", file=sys.stderr
    )

    # Compute mel spectrogram
    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    mel = compute_mel_spectrogram(audio_tensor, mel_filters)
    print(f"Mel spectrogram: {mel.shape}", file=sys.stderr)
    print(
        f"  min={mel.min():.6f}, max={mel.max():.6f}, mean={mel.mean():.6f}",
        file=sys.stderr,
    )

    # Dump first few values for comparison
    print("\n# First 10 mel filter values (row 0):", file=sys.stderr)
    print(mel_filters[0, :10].tolist(), file=sys.stderr)

    print("\n# First frame mel spectrogram (all 128 bins):", file=sys.stderr)
    print(mel[:, 0].tolist(), file=sys.stderr)

    print("\n# Mel spectrogram corner [0:4, 0:4]:", file=sys.stderr)
    print(mel[:4, :4].tolist(), file=sys.stderr)

    # Save as raw f32 binary (row-major) for loading in Zig
    # Layout: [num_mel_bins, num_frames] contiguous float32
    output_path = path.rsplit(".", 1)[0] + "_mel.bin"
    mel.numpy().astype(np.float32).tofile(output_path)
    print(f"\nSaved mel spectrogram ({mel.shape[0]}x{mel.shape[1]}, f32) to {output_path}", file=sys.stderr)


# def main():
#     if len(sys.argv) < 2:
#         print(f"Usage: {sys.argv[0]} <audio.wav>", file=sys.stderr)
#         sys.exit(1)

#     wav_path = sys.argv[1]
#     execute(wav_path)


# if __name__ == "__main__":
#     main()
