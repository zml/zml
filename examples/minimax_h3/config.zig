const std = @import("std");

const zml = @import("zml");

pub const modality_count: i64 = 3;
pub const video_fps: f32 = 24.0;
pub const audio_hz: f32 = 40.0;
pub const audio_sample_rate: u32 = 32_000;
pub const visual_spatial: u32 = 16;
pub const visual_temporal: u32 = 4;
/// Official VAE clip: `17 * n + 5` pixel frames, `5 * n + 2` latent frames.
pub const visual_clip_length: u32 = 17;
pub const visual_latents_per_chunk: u32 = 5;
pub const visual_cond_timestep: f32 = 0.999;
pub const frame_rescale: f32 = 5.0 / 3.0;
pub const default_short_side: u32 = 768;
/// Community preview canvas (ComfyUI 0.2 MP path): 16:9 → 640×352 after snap-32.
pub const preview_short_side: u32 = 352;
pub const tiny_short_side: u32 = 128;
pub const preview_steps: u32 = 10;
pub const tiny_steps: u32 = 4;
pub const video_shift: f32 = 12.0;
pub const audio_shift: f32 = 3.0;
pub const encoder_layers_used: u32 = 50;

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
        return switch (self.taskFamily()) {
            .fl2va => "FL2VA",
            .ref2va => "Ref2VA",
        };
    }
};

pub const TaskFamily = enum { fl2va, ref2va };

pub const Aspect = enum {
    @"21:9",
    @"16:9",
    @"4:3",
    @"1:1",
    @"3:4",
    @"9:16",

    pub fn parse(text: []const u8) ?Aspect {
        return std.meta.stringToEnum(Aspect, text);
    }

    pub fn ratio(self: Aspect) struct { w: u32, h: u32 } {
        return switch (self) {
            .@"21:9" => .{ .w = 21, .h = 9 },
            .@"16:9" => .{ .w = 16, .h = 9 },
            .@"4:3" => .{ .w = 4, .h = 3 },
            .@"1:1" => .{ .w = 1, .h = 1 },
            .@"3:4" => .{ .w = 3, .h = 4 },
            .@"9:16" => .{ .w = 9, .h = 16 },
        };
    }
};

pub const Config = struct {
    hidden_size: i64 = 5376,
    num_layers: i64 = 50,
    num_refiner_layers: i64 = 2,
    token_refiner_num_layers: ?i64 = null,
    num_attention_heads: i64 = 56,
    attention_head_dim: i64 = 128,
    ffn_dim: i64 = 14336,
    ffn_hidden_size: ?i64 = null,
    in_channels: i64 = 24,
    latents_dim: ?i64 = null,
    audio_in_channels: i64 = 32,
    audio_latents_dim: ?i64 = null,
    patch_size: [3]i64 = .{ 1, 2, 2 },
    text_dim: i64 = 5120,
    freq_dim: i64 = 256,
    timestep_input_dim: ?i64 = null,
    time_embed_hidden_dim: i64 = 5376,
    time_embed_hidden_size: ?i64 = null,
    time_embed_dim: i64 = 2688,
    rope_freq_dim: i64 = 16,
    rope_inv_freq_len: ?i64 = null,
    rope_theta: f32 = 10000.0,
    norm_eps: f32 = 1e-5,
    qk_norm_eps: f32 = 1e-5,
    final_norm_eps: f32 = 1e-5,

    pub fn resolve(self: Config) Config {
        var out = self;
        if (self.token_refiner_num_layers) |n| out.num_refiner_layers = n;
        if (self.ffn_hidden_size) |n| out.ffn_dim = n;
        if (self.latents_dim) |n| out.in_channels = n;
        if (self.audio_latents_dim) |n| out.audio_in_channels = n;
        if (self.timestep_input_dim) |n| out.freq_dim = n;
        if (self.time_embed_hidden_size) |n| out.time_embed_hidden_dim = n;
        if (self.rope_inv_freq_len) |n| out.rope_freq_dim = n;
        return out;
    }

    pub fn official() Config {
        return (Config{}).resolve();
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
            .mrope_section = if (text.rope_scaling) |s| s.mrope_section else .{ 24, 20, 20 },
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
        return .{
            .hidden_size = self.hidden_size orelse 5120,
            .num_hidden_layers = self.num_hidden_layers orelse 64,
            .used_hidden_layers = @min(self.num_hidden_layers orelse 64, encoder_layers_used),
            .num_attention_heads = self.num_attention_heads orelse 64,
            .num_key_value_heads = self.num_key_value_heads orelse 8,
            .intermediate_size = self.intermediate_size orelse 25600,
            .head_dim = self.head_dim orelse 128,
            .rms_norm_eps = self.rms_norm_eps,
            .rope_theta = self.rope_theta,
            .vocab_size = self.vocab_size orelse 151936,
            .mrope_section = if (self.rope_scaling) |s| s.mrope_section else .{ 24, 20, 20 },
            .max_position_embeddings = self.max_position_embeddings,
        };
    }
};

/// Floor for official 768p. Below this, auto canvas is preview and `--full` is refused.
pub const full_canvas_min_device_bytes: u64 = 40 * 1024 * 1024 * 1024;

/// Preview + audio refs packs hundreds of extra tokens. Refused under 40 GiB.
pub fn audioRefsNeedTiny(short_side: u32, device_bytes: u64) bool {
    return short_side > tiny_short_side and device_bytes != 0 and device_bytes < full_canvas_min_device_bytes;
}

pub const Canvas = enum { auto, tiny, preview, full };

pub fn parseCanvas(tiny: bool, preview: bool, full: bool) error{ConflictingCanvas}!Canvas {
    const n = @as(u2, @intFromBool(tiny)) + @as(u2, @intFromBool(preview)) + @as(u2, @intFromBool(full));
    if (n > 1) return error.ConflictingCanvas;
    if (tiny) return .tiny;
    if (preview) return .preview;
    if (full) return .full;
    return .auto;
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

/// `device_bytes == 0` means unknown: CPU/Metal stay preview, accelerators stay official 768p.
pub fn canvasForTarget(target: zml.Target, canvas: Canvas, device_bytes: u64) struct { short_side: u32, steps: u32 } {
    return switch (canvas) {
        .tiny => .{ .short_side = tiny_short_side, .steps = tiny_steps },
        .preview => .{ .short_side = preview_short_side, .steps = preview_steps },
        .full => .{ .short_side = default_short_side, .steps = 30 },
        .auto => if (device_bytes != 0 and device_bytes < full_canvas_min_device_bytes)
            .{ .short_side = preview_short_side, .steps = preview_steps }
        else switch (target) {
            .cpu, .metal => .{ .short_side = preview_short_side, .steps = preview_steps },
            else => .{ .short_side = default_short_side, .steps = 30 },
        },
    };
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

pub fn parseConfig(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(Config) {
    return parseJson(Config, allocator, io, dir, "config.json");
}

pub fn snap32(value: u32) u32 {
    if (value == 0) return 32;
    return (value + 16) / 32 * 32;
}

pub fn pixelSize(aspect: Aspect, short_side: u32) struct { w: u32, h: u32 } {
    const r = aspect.ratio();
    const short = snap32(short_side);
    if (r.w >= r.h) {
        const long = snap32(@intCast((@as(u64, short) * r.w + r.h / 2) / r.h));
        return .{ .w = long, .h = short };
    }
    const long = snap32(@intCast((@as(u64, short) * r.h + r.w / 2) / r.w));
    return .{ .w = short, .h = long };
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
    return @intFromFloat(@round(duration_s * audio_hz));
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
