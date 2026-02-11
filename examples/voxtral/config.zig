pub const MelSpectrumConfig = struct {
    hop_length: u63 = 160,
    window_size: u63 = 400,
    global_log_mel_max: f32 = 1.5,
};

pub const EncoderConfig = struct {
    dim: usize = 1280,
    n_layers: usize = 32,
    n_heads: usize = 32,
    head_dim: usize = 64,
    hidden_dim: usize = 5120,
    n_kv_heads: usize = 32,
    window_size: u32 = 750,
    norm_eps: f32 = 1e-5,
    rope_theta: f32 = 1_000_000.0,
    downsample_factor: usize = 4,
};
