const std = @import("std");

pub const AttentionConfig = struct {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    rope_theta: f32,
    sliding_window: u32,
};

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

        pub fn attentionConfig(self: EncoderArgs) AttentionConfig {
            return .{
                .n_heads = self.n_heads,
                .n_kv_heads = self.n_kv_heads,
                .head_dim = self.head_dim,
                .rope_theta = self.rope_theta,
                .sliding_window = self.sliding_window,
            };
        }
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

    pub fn attentionConfig(self: Config) AttentionConfig {
        return .{
            .n_heads = self.n_heads,
            .n_kv_heads = self.n_kv_heads,
            .head_dim = self.head_dim,
            .rope_theta = self.rope_theta,
            .sliding_window = self.sliding_window,
        };
    }

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

pub const StreamParams = struct {
    dsf: u32,
    mel_per_step: u32,
    chunk_mel: u32,
    chunk_audio: u32,
    raw_audio_length_per_tok: u32,
    _hop_length: u32,
    n_delay_tokens: u32,
    n_right_pad_tokens: u32,
    left_pad: u32,
    prompt_len: u32,

    pub fn init(config: Config, transcription_delay_ms: f32, n_left_pad_tokens: u32) StreamParams {
        const sample_rate: f32 = @floatFromInt(config.audio().sampling_rate);
        const frame_rate = config.audio().frame_rate;
        const raw_audio_length_per_tok: u32 = @intFromFloat(sample_rate / frame_rate);
        const hop_length = config.audio().hop_length;

        const delay_samples: u32 = @intFromFloat(transcription_delay_ms / 1000.0 * sample_rate);
        const audio_length_per_tok = raw_audio_length_per_tok / hop_length;
        const n_delay_tokens = std.math.divCeil(u32, delay_samples / hop_length, audio_length_per_tok) catch unreachable;

        const dsf = config.downsample_factor();
        const mel_per_step = dsf * 2;
        const window_size = config.audio().window_size;

        return .{
            .dsf = dsf,
            .mel_per_step = mel_per_step,
            .chunk_mel = mel_per_step,
            .chunk_audio = (mel_per_step - 1) * hop_length + window_size,
            .raw_audio_length_per_tok = raw_audio_length_per_tok,
            ._hop_length = hop_length,
            .n_delay_tokens = n_delay_tokens,
            .n_right_pad_tokens = (n_delay_tokens + 1) + 10,
            .left_pad = n_left_pad_tokens * raw_audio_length_per_tok,
            .prompt_len = 1 + n_left_pad_tokens + n_delay_tokens,
        };
    }

    pub fn numFrames(self: StreamParams, audio_len: usize) u32 {
        return @intCast(audio_len / self._hop_length);
    }

    pub fn totalSteps(self: StreamParams, audio_len: usize) u32 {
        const nf = self.numFrames(audio_len);
        const encoder_seq_len = (nf + 1) / 2;
        return (encoder_seq_len + self.dsf - 1) / self.dsf;
    }

    pub fn paddedMelFrames(self: StreamParams, audio_len: usize) u32 {
        return self.totalSteps(audio_len) * self.mel_per_step;
    }
};
