const std = @import("std");

pub const Config = struct {
    dim: u32,
    n_layers: u32,
    head_dim: u32,
    hidden_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    use_biases: bool = false,
    causal: bool = true,
    rope_theta: f32,
    norm_eps: f32,
    vocab_size: u32,
    sliding_window: u32,
    tied_embeddings: bool = true,
    ada_rms_norm_t_cond: bool = true,
    ada_rms_norm_t_cond_dim: u32 = 32,
    multimodal: Multimodal,

    pub const Multimodal = struct {
        whisper_model_args: WhisperModelArgs,
    };

    pub const WhisperModelArgs = struct {
        encoder_args: EncoderArgs,
        downsample_args: DownsampleArgs,
    };

    pub const DownsampleArgs = struct {
        downsample_factor: u32,
    };

    pub const EncoderArgs = struct {
        audio_encoding_args: AudioEncodingArgs,
        dim: u32,
        n_layers: u32,
        head_dim: u32,
        hidden_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        use_biases: bool = true,
        rope_theta: f32,
        norm_eps: f32,
        sliding_window: u32,
    };

    pub const AudioEncodingArgs = struct {
        sampling_rate: u32 = 16000,
        frame_rate: f32 = 12.5,
        num_mel_bins: u32 = 128,
        hop_length: u32 = 160,
        window_size: u32 = 400,
        global_log_mel_max: f32 = 1.5,
    };
    
    // -- Shortcuts
    
    pub fn encoder(self: Config) EncoderArgs {
        return self.multimodal.whisper_model_args.encoder_args;
    }

    pub fn audio(self: Config) AudioEncodingArgs {
        return self.multimodal.whisper_model_args.encoder_args.audio_encoding_args;
    }

    pub fn downsample_factor(self: Config) u32 {
        return self.multimodal.whisper_model_args.downsample_args.downsample_factor;
    }
};

pub fn parseConfig(allocator: std.mem.Allocator, io: std.Io, model_dir: std.Io.Dir) !std.json.Parsed(Config) {
    const config_file = try model_dir.openFile(io, "params.json", .{}); // it is named params not config here
    defer config_file.close(io);
    
    var buffer: [4096]u8 = undefined;
    var file_reader = config_file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();
    
    return try std.json.parseFromTokenSource(Config, allocator, &reader, .{ .ignore_unknown_fields = true });
}
