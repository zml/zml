const std = @import("std");

const zml = @import("zml");

// =============================================================================
// recipe/config.zig — H3 constants, tokenizer, frame math
//
// Official FL2VA clip is 17n+5 frames. Canvas multiple is 32.
// =============================================================================

pub const official_repo = "MiniMaxAI/MiniMax-H3";
pub const official_revision = "42ed227ee7df40d41602854ae760620d6eb651fe";
pub const task_dir = "FL2VA";

pub const modality_count: i64 = 3;
pub const video_fps: f32 = 24.0;
pub const audio_hz: f32 = 40.0;
pub const audio_sample_rate: u32 = 32_000;
pub const visual_spatial: u32 = 16;
pub const visual_temporal: u32 = 4;
/// VAE clip: `17 * n + 5` pixel frames, `5 * n + 2` latent frames.
pub const visual_clip_length: u32 = 17;
pub const visual_latents_per_chunk: u32 = 5;
pub const video_shift: f32 = 12.0;
pub const audio_shift: f32 = 3.0;
pub const encoder_layers_used: u32 = 50;
pub const canvas_multiple: u32 = 32;

pub fn officialTokenizerUri(buf: []u8) ![]u8 {
    return std.fmt.bufPrint(buf, "hf://{s}@{s}/{s}/tokenizer/tokenizer.json", .{
        official_repo,
        official_revision,
        task_dir,
    });
}

pub const Config = struct {
    hidden_size: i64 = 5376,
    num_layers: i64 = 50,
    num_refiner_layers: i64 = 2,
    num_attention_heads: i64 = 56,
    attention_head_dim: i64 = 128,
    ffn_dim: i64 = 14336,
    in_channels: i64 = 24,
    audio_in_channels: i64 = 32,
    patch_size: [3]i64 = .{ 1, 2, 2 },
    text_dim: i64 = 5120,
    freq_dim: i64 = 256,
    rope_freq_dim: i64 = 16,
    rope_theta: f32 = 10000.0,
    norm_eps: f32 = 1e-5,
    qk_norm_eps: f32 = 1e-5,
    final_norm_eps: f32 = 1e-5,

    pub fn official() Config {
        return .{};
    }

    pub fn videoPatchDim(self: Config) i64 {
        return self.in_channels * self.patch_size[0] * self.patch_size[1] * self.patch_size[2];
    }

    pub fn innerDim(self: Config) i64 {
        return self.num_attention_heads * self.attention_head_dim;
    }

    pub fn rotaryDim(self: Config) i64 {
        return 2 * 3 * self.rope_freq_dim;
    }

    pub fn adalnOutFeatures(self: Config) i64 {
        return 6 * self.hidden_size * modality_count;
    }

    pub fn finalAdalnOutFeatures(self: Config) i64 {
        return 2 * self.hidden_size;
    }
};

pub const EncoderConfig = struct {
    hidden_size: i64 = 5120,
    num_hidden_layers: i64 = 64,
    used_hidden_layers: i64 = encoder_layers_used,
    num_attention_heads: i64 = 64,
    num_key_value_heads: i64 = 8,
    intermediate_size: i64 = 25600,
    head_dim: i64 = 128,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 5_000_000.0,
    vocab_size: i64 = 151936,
    mrope_section: [3]i64 = .{ 24, 20, 20 },
    max_position_embeddings: i64 = 262144,

    pub fn official() EncoderConfig {
        return .{};
    }

    pub fn fromTextConfig(text: TextConfigJson) EncoderConfig {
        return .{
            .hidden_size = text.hidden_size,
            .num_hidden_layers = text.num_hidden_layers,
            .used_hidden_layers = @min(text.num_hidden_layers, encoder_layers_used),
            .num_attention_heads = text.num_attention_heads,
            .num_key_value_heads = text.num_key_value_heads,
            .intermediate_size = text.intermediate_size,
            .head_dim = text.head_dim orelse @divExact(text.hidden_size, text.num_attention_heads),
            .rms_norm_eps = text.rms_norm_eps,
            .rope_theta = text.rope_theta,
            .vocab_size = text.vocab_size,
            .mrope_section = if (text.rope_scaling) |s| s.mrope_section else EncoderConfig.official().mrope_section,
            .max_position_embeddings = text.max_position_embeddings,
        };
    }
};

pub const TextConfigJson = struct {
    hidden_size: i64,
    num_hidden_layers: i64,
    num_attention_heads: i64,
    num_key_value_heads: i64,
    intermediate_size: i64,
    head_dim: ?i64 = null,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 5_000_000.0,
    vocab_size: i64,
    max_position_embeddings: i64 = 262144,
    rope_scaling: ?struct {
        mrope_section: [3]i64 = .{ 24, 20, 20 },
    } = null,
};

pub const EncoderFileConfig = struct {
    text_config: ?TextConfigJson = null,
    hidden_size: ?i64 = null,
    num_hidden_layers: ?i64 = null,
    num_attention_heads: ?i64 = null,
    num_key_value_heads: ?i64 = null,
    intermediate_size: ?i64 = null,
    head_dim: ?i64 = null,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 5_000_000.0,
    vocab_size: ?i64 = null,
    max_position_embeddings: i64 = 262144,
    rope_scaling: ?struct {
        mrope_section: [3]i64 = .{ 24, 20, 20 },
    } = null,

    pub fn resolve(self: EncoderFileConfig) EncoderConfig {
        if (self.text_config) |text| return .fromTextConfig(text);
        const base = EncoderConfig.official();
        const layers = self.num_hidden_layers orelse base.num_hidden_layers;
        return .{
            .hidden_size = self.hidden_size orelse base.hidden_size,
            .num_hidden_layers = layers,
            .used_hidden_layers = @min(layers, encoder_layers_used),
            .num_attention_heads = self.num_attention_heads orelse base.num_attention_heads,
            .num_key_value_heads = self.num_key_value_heads orelse base.num_key_value_heads,
            .intermediate_size = self.intermediate_size orelse base.intermediate_size,
            .head_dim = self.head_dim orelse base.head_dim,
            .rms_norm_eps = self.rms_norm_eps,
            .rope_theta = self.rope_theta,
            .vocab_size = self.vocab_size orelse base.vocab_size,
            .mrope_section = if (self.rope_scaling) |s| s.mrope_section else base.mrope_section,
            .max_position_embeddings = self.max_position_embeddings,
        };
    }
};

pub fn minDeviceBytes(platform: *const zml.Platform) u64 {
    var min_b: u64 = 0;
    var any = false;
    for (platform.devices) |device| {
        if (device.memoryStats().bytes_limit) |bytes| {
            if (!any or bytes < min_b) min_b = bytes;
            any = true;
        }
    }
    return min_b;
}

pub fn parseJson(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8) !std.json.Parsed(T) {
    const file = try dir.openFile(io, name, .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    return try std.json.parseFromTokenSource(T, allocator, &reader, .{ .ignore_unknown_fields = true });
}

pub fn parseOptional(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8) !?std.json.Parsed(T) {
    return parseJson(T, allocator, io, dir, name) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
}

pub fn loadDitConfig(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !Config {
    const parsed = try parseOptional(Config, allocator, io, dir, "config.json") orelse return Config.official();
    defer parsed.deinit();
    return parsed.value;
}

pub fn loadEncoderConfig(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !EncoderConfig {
    const parsed = try parseOptional(EncoderFileConfig, allocator, io, dir, "config.json") orelse return EncoderConfig.official();
    defer parsed.deinit();
    return parsed.value.resolve();
}

pub const Size = struct { w: u32, h: u32 };

pub const FramePlan = struct {
    raw: u32,
    aligned: u32,

    pub fn seconds(self: FramePlan) f32 {
        return @as(f32, @floatFromInt(self.aligned)) / video_fps;
    }
};

pub fn checkDuration(seconds: f32) !void {
    if (!std.math.isFinite(seconds) or seconds < 5.0 or seconds > 15.0) return error.InvalidDuration;
}

pub fn resolveFrames(duration_s: f32, frames: u32) error{InvalidDuration}!FramePlan {
    const raw = if (frames != 0) frames else blk: {
        try checkDuration(duration_s);
        break :blk frameCount(duration_s);
    };
    if (frames != 0) {
        const min_frames = frameCount(5.0);
        const max_frames = frameCount(15.0);
        if (raw < min_frames or raw > max_frames) return error.InvalidDuration;
    }
    return .{
        .raw = raw,
        .aligned = alignFrameCount(raw),
    };
}

pub fn audioLatentFromFrames(frames: u32) u32 {
    return @intFromFloat(@round(@as(f32, @floatFromInt(frames)) / video_fps * audio_hz));
}

pub fn frameCount(duration_s: f32) u32 {
    if (!std.math.isFinite(duration_s) or duration_s < 0) return 0;
    return @intFromFloat(@round(duration_s * video_fps));
}

/// Snap up to the next `17 * n + 5` the video VAE can encode.
pub fn alignFrameCount(frames: u32) u32 {
    const n: u32 = if (frames < 1) 1 else frames;
    const rem = n % visual_clip_length;
    if (rem == visual_latents_per_chunk) return n;
    return n + (visual_latents_per_chunk + visual_clip_length - rem) % visual_clip_length;
}

/// Latent frames for an aligned `17 * n + 5` pixel count: `5 * n + 2`.
pub fn videoLatentFrames(aligned_frames: u32) u32 {
    std.debug.assert(aligned_frames % visual_clip_length == visual_latents_per_chunk);
    return (aligned_frames - visual_latents_per_chunk) / visual_clip_length * visual_latents_per_chunk + 2;
}

pub const LatentHw = struct { t: u32, h: u32, w: u32 };

pub fn visualLatentSize(pixel_h: u32, pixel_w: u32, frames: u32) LatentHw {
    return .{
        .t = videoLatentFrames(alignFrameCount(frames)),
        .h = pixel_h / visual_spatial,
        .w = pixel_w / visual_spatial,
    };
}

pub fn videoTokenCount(latent_t: u32, latent_h: u32, latent_w: u32, patch: [3]i64) u32 {
    const pt: u32 = @intCast(patch[0]);
    const ph: u32 = @intCast(patch[1]);
    const pw: u32 = @intCast(patch[2]);
    return (latent_t / pt) * (latent_h / ph) * (latent_w / pw);
}

pub fn openTaskDir(io: std.Io, repo: std.Io.Dir) !struct { dir: std.Io.Dir, owned: bool } {
    if (repo.openDir(io, task_dir, .{})) |dir| {
        return .{ .dir = dir, .owned = true };
    } else |_| {}
    return .{ .dir = repo, .owned = false };
}
