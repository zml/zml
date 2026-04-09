import torch

def stats(path, shape, name):
    data = open(path, 'rb').read()
    t = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(shape).float()
    kurt = ((t - t.mean())**4).mean() / t.std()**4
    print(f"{name}: mean={t.mean():.6f} std={t.std():.6f} min={t.min():.4f} max={t.max():.4f} kurtosis={kurt:.4f}")

# Adjust shapes to match what the logs printed for video_noise/audio_noise dims
# Stage 1 noise (check the dims from the log output)
stats("/root/ltx_run_oboulant3/unified/s1_video_noise.bin", (1, 6144, 128), "s1_video_noise")
stats("/root/ltx_run_oboulant3/unified/s1_audio_noise.bin", (1, 126, 128), "s1_audio_noise")

# Stage 2 noise
stats("/root/ltx_run_oboulant3/unified/s2_video_noise.bin", (1, 24576, 128), "s2_video_noise")
stats("/root/ltx_run_oboulant3/unified/s2_audio_noise.bin", (1, 126, 128), "s2_audio_noise")
