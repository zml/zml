pub const MelSpectrumConfig = struct {
    hop_length: u63 = 160,
    window_size: u63 = 400,
    global_log_mel_max: f32 = 1.5,
};
