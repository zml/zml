pub const MelSpectrumConfig = struct {
    hop_length: u32 = 160,
    n_fft: u32 = 400,
    global_log_mel_max: f32 = 1.5,
};
