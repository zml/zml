const std = @import("std");

const zml = @import("zml");

pub const official_repo = "MiniMaxAI/MiniMax-H3";
pub const official_revision = "42ed227ee7df40d41602854ae760620d6eb651fe";

pub const modality_count: i64 = 3;
pub const video_fps: f32 = 24.0;
pub const audio_hz: f32 = 40.0;
pub const audio_sample_rate: u32 = 32_000;
pub const visual_spatial: u32 = 16;
pub const visual_temporal: u32 = 4;
/// VAE clip: `17 * n + 5` pixel frames, `5 * n + 2` latent frames.
pub const visual_clip_length: u32 = 17;
pub const visual_latents_per_chunk: u32 = 5;
pub const visual_cond_timestep: f32 = 0.999;
pub const default_short_side: u32 = 768;
pub const default_size: []const u8 = "1344x768";
pub const default_steps: u32 = 30;
/// 16:9 at 352 short-edge after snap-32. Memory floor uses this as "small".
pub const preview_short_side: u32 = 352;
pub const video_shift: f32 = 12.0;
pub const audio_shift: f32 = 3.0;
pub const encoder_layers_used: u32 = 50;
pub const qwen_video_fps: f32 = 2.0;
pub const canvas_multiple: u32 = 32;
pub const canvas_max_pixels: u32 = 768 * 1344;
/// Official `reference_image_short_edge`. Upscale allowed, no area cap.
pub const reference_image_short_edge: u32 = 2048;
pub const min_aspect: f32 = 0.25;
pub const max_aspect: f32 = 4.0;
pub const max_ref_files: u32 = 12;
pub const max_ref_images: u32 = 9;
pub const max_ref_videos: u32 = 3;
pub const max_ref_audios: u32 = 3;
pub const visual_encode_seed: u64 = 42;

pub const Variant = enum {
    t2va,
    fl2va,
    ref2va,

    pub fn taskFamily(self: Variant) TaskFamily {
        return switch (self) {
            .t2va, .fl2va => .fl2va,
            .ref2va => .ref2va,
        };
    }

    pub fn dirName(self: Variant) []const u8 {
        return taskDirName(self.taskFamily());
    }
};

pub const TaskFamily = enum { fl2va, ref2va };

pub fn taskDirName(family: TaskFamily) []const u8 {
    return switch (family) {
        .fl2va => "FL2VA",
        .ref2va => "Ref2VA",
    };
}

pub const official_task_dirs = [_][]const u8{ taskDirName(.fl2va), taskDirName(.ref2va) };

pub fn officialTokenizerUri(buf: []u8) ![]u8 {
    return std.fmt.bufPrint(buf, "hf://{s}@{s}/{s}/tokenizer/tokenizer.json", .{
        official_repo,
        official_revision,
        taskDirName(.fl2va),
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

/// Minimum measured device memory for official 768p.
pub const full_canvas_min_device_bytes: u64 = 40 * 1024 * 1024 * 1024;

pub fn checkDuration(seconds: f32) !void {
    if (seconds < 5.0 or seconds > 15.0) return error.InvalidDuration;
}

pub fn checkSteps(steps: u32) !void {
    if (steps < 2) return error.TooFewSteps;
}

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

pub fn checkDeviceForSize(width: u32, height: u32, device_bytes: u64) !void {
    if (device_bytes == 0) return;
    if (@min(width, height) > preview_short_side and device_bytes < full_canvas_min_device_bytes)
        return error.SizeTooLarge;
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

pub fn parseSize(text: []const u8) error{ InvalidSize, InvalidAspect, SizeTooLarge }!Size {
    const sep = std.mem.indexOfScalar(u8, text, 'x') orelse std.mem.indexOfScalar(u8, text, 'X') orelse return error.InvalidSize;
    if (sep == 0 or sep + 1 >= text.len) return error.InvalidSize;
    const width = std.fmt.parseInt(u32, text[0..sep], 10) catch return error.InvalidSize;
    const height = std.fmt.parseInt(u32, text[sep + 1 ..], 10) catch return error.InvalidSize;
    return snapSize(width, height);
}

pub fn snapSize(width: u32, height: u32) error{ InvalidSize, InvalidAspect, SizeTooLarge }!Size {
    if (width == 0 or height == 0) return error.InvalidSize;
    const ratio = @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(height));
    if (ratio < min_aspect or ratio > max_aspect) return error.InvalidAspect;
    const multiple: f32 = @floatFromInt(canvas_multiple);
    const w = @max(canvas_multiple, @as(u32, @intFromFloat(@round(@as(f32, @floatFromInt(width)) / multiple))) * canvas_multiple);
    const h = @max(canvas_multiple, @as(u32, @intFromFloat(@round(@as(f32, @floatFromInt(height)) / multiple))) * canvas_multiple);
    if (@as(u64, w) * @as(u64, h) > canvas_max_pixels) return error.SizeTooLarge;
    return .{ .w = w, .h = h };
}

/// Canvas: short edge, area cap `768*1344`, then nearest multiple of 32.
pub fn resolveCanvas(aspect_w: f32, aspect_h: f32, short_edge: u32, max_pixels: u32) error{InvalidAspect}!Size {
    if (aspect_w <= 0 or aspect_h <= 0) return error.InvalidAspect;
    const ratio = aspect_w / aspect_h;
    if (ratio < min_aspect or ratio > max_aspect) return error.InvalidAspect;

    var width: f32 = undefined;
    var height: f32 = undefined;
    if (ratio >= 1.0) {
        width = @as(f32, @floatFromInt(short_edge)) * ratio;
        height = @floatFromInt(short_edge);
    } else {
        width = @floatFromInt(short_edge);
        height = @as(f32, @floatFromInt(short_edge)) / ratio;
    }
    const area = width * height;
    if (area > @as(f32, @floatFromInt(max_pixels))) {
        const scale = @sqrt(@as(f32, @floatFromInt(max_pixels)) / area);
        width *= scale;
        height *= scale;
    }
    const multiple: f32 = @floatFromInt(canvas_multiple);
    return .{
        .w = @max(canvas_multiple, @as(u32, @intFromFloat(@round(width / multiple))) * canvas_multiple),
        .h = @max(canvas_multiple, @as(u32, @intFromFloat(@round(height / multiple))) * canvas_multiple),
    };
}

pub fn frameCount(duration_s: f32) u32 {
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

pub fn audioLatentLength(duration_s: f32) u32 {
    const frames = alignFrameCount(frameCount(duration_s));
    return @intFromFloat(@round(@as(f32, @floatFromInt(frames)) / video_fps * audio_hz));
}

pub fn videoTokenCount(latent_t: u32, latent_h: u32, latent_w: u32, patch: [3]i64) u32 {
    const pt: u32 = @intCast(patch[0]);
    const ph: u32 = @intCast(patch[1]);
    const pw: u32 = @intCast(patch[2]);
    return (latent_t / pt) * (latent_h / ph) * (latent_w / pw);
}

pub fn openTaskDir(io: std.Io, repo: std.Io.Dir, variant: Variant) !struct { dir: std.Io.Dir, owned: bool } {
    if (repo.openDir(io, variant.dirName(), .{})) |dir| {
        return .{ .dir = dir, .owned = true };
    } else |_| {}
    return .{ .dir = repo, .owned = false };
}
