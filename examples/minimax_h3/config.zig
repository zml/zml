//! MiniMax-H3 geometry and layer sizes from the model repo JSON.
//!
//! `load` reads `text_encoder/`, `transformer/`, `vae/`, `audio_vae/`, and the
//! two scheduler files. Canvas flags `--width` / `--height` / `--duration` feed
//! `Geometry.init`:
//!   pixels  W×H, multiple of 32, area ≤ 768×1344
//!   frames  round(duration×24), snapped up to `clip_length·n + latents_per_chunk`
//!   latents W/16 × H/16 × (5n+2)
//!   DiT tokens  latent_t × (latent_h/2) × (latent_w/2)   (1×2×2 patch)
//!   audio tokens  2 × round(frames/24×40)

const std = @import("std");
const zml = @import("zml");

const log = std.log.scoped(.minimax_h3);

pub const video_fps: f32 = 24.0;
pub const audio_hz: f32 = 40.0;
/// VAE clip: `clip_length·n + latents_per_chunk` pixel frames, `5n+2` latent frames.
pub const visual_latents_per_chunk: u32 = 5;
pub const canvas_multiple: u32 = 32;
pub const canvas_max_pixels: u32 = 768 * 1344;
pub const min_duration_s: f32 = 5.0;
pub const max_duration_s: f32 = 15.0;
/// Packed-sequence modalities. Integer values are AdaLN table columns.
pub const Modality = enum(u8) { video = 0, text = 1, audio = 2 };
pub const modality_count: i64 = @intCast(std.meta.fields(Modality).len);
/// AdaLN / time-embed table width (checkpoint). A packed row usually has 2 unique
/// times (video/text vs audio); those two are ordered and padded to 4.
pub const timestep_slot_count: u32 = 4;

/// Qwen `text_config` in `text_encoder/config.json`.
/// `used_hidden_layers` is the MiniMax-H3 recipe (first 50 of Qwen's 64), not a JSON field.
pub const EncoderConfig = struct {
    hidden_size: i64,
    num_attention_heads: i64,
    num_key_value_heads: i64,
    head_dim: i64,
    rms_norm_eps: f32,
    rope_theta: f32,
    used_hidden_layers: i64 = 50,
};

const TextEncoderFile = struct {
    text_config: EncoderConfig,
};

/// DiT `transformer/config.json`.
pub const DitConfig = struct {
    hidden_size: i64,
    num_layers: i64,
    num_refiner_layers: i64,
    num_attention_heads: i64,
    attention_head_dim: i64,
    in_channels: i64,
    audio_in_channels: i64,
    patch_size: [3]i64,
    text_dim: i64,
    freq_dim: i64,
    rope_freq_dim: i64,
    rope_theta: f32,
    norm_eps: f32,
    qk_norm_eps: f32,
    final_norm_eps: f32,

    /// MM-RoPE width: 3 axes × `rope_freq_dim`, then duplicated (`ops.ropeCat3`).
    pub fn rotaryDim(self: DitConfig) i64 {
        return 2 * 3 * self.rope_freq_dim;
    }
};

/// Visual VAE `vae/config.json`.
pub const VisualConfig = struct {
    latent_channels: i64,
    out_channels: i64,
    decoder_num_layers: i64,
    decoder_num_attention_heads: i64,
    decoder_attention_head_dim: i64,
    decoder_num_register_tokens: i64,
    decoder_rope_theta: f32,
    decoder_rope_dim_ratio: f32,
    decoder_norm_eps: f32,
    spatial_downsample_factors: [6]i64,
    temporal_downsample_factors: [6]i64,
    clip_length: i64,
    token_drop: i64,
    latents_mean: [24]f32,
    latents_std: [24]f32,

    pub fn dim(self: VisualConfig) i64 {
        return self.decoder_num_attention_heads * self.decoder_attention_head_dim;
    }

    pub fn rotaryDim(self: VisualConfig) i64 {
        return @intFromFloat(@as(f32, @floatFromInt(self.decoder_attention_head_dim)) * self.decoder_rope_dim_ratio);
    }

    pub fn spatial(self: VisualConfig) u32 {
        return product(&self.spatial_downsample_factors);
    }

    pub fn temporal(self: VisualConfig) u32 {
        return product(&self.temporal_downsample_factors);
    }
};

/// Snake-beta upsample / downsample in the audio decoder.
pub const audio_activation_ratio: i64 = 2;
pub const audio_activation_kernel: i64 = 12;

/// Audio VAE `audio_vae/config.json`.
pub const AudioConfig = struct {
    latent_channels: i64,
    encoder_rates: [5]i64,
    decoder_rates: [7]i64,
    decoder_kernel_sizes: [7]i64,
    resblock_kernel_sizes: [3]i64,
    resblock_dilation_sizes: [3][3]i64,
    sampling_rate: u32,
    latents_mean: [32]f32,
    latents_std: [32]f32,

    pub fn hop(self: AudioConfig) u32 {
        return product(&self.encoder_rates);
    }
};

const SchedulerFile = struct {
    shift: f32,
};

pub const Configs = struct {
    encoder: EncoderConfig,
    dit: DitConfig,
    vae: VisualConfig,
    audio: AudioConfig,
    video_shift: f32,
    audio_shift: f32,
};

/// `CreatePhysicalMeshFn` has no extra args. `load` writes head counts here before `Platform.auto`.
var mesh_heads: struct {
    dit: i64 = 1,
    encoder: i64 = 1,
    encoder_kv: i64 = 1,
} = .{};

fn product(xs: []const i64) u32 {
    var p: u32 = 1;
    for (xs) |x| p *= @intCast(x);
    return p;
}

fn parseConfig(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8) !std.json.Parsed(T) {
    const file = try dir.openFile(io, name, .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    return try std.json.parseFromTokenSource(T, allocator, &reader, .{ .ignore_unknown_fields = true });
}

fn readJson(comptime T: type, allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, name: []const u8) !T {
    const parsed = try parseConfig(T, allocator, io, repo, name);
    defer parsed.deinit();
    return parsed.value;
}

pub fn load(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir) !Configs {
    const encoder_file = try readJson(TextEncoderFile, allocator, io, repo, "text_encoder/config.json");
    const dit = try readJson(DitConfig, allocator, io, repo, "transformer/config.json");
    const vae = try readJson(VisualConfig, allocator, io, repo, "vae/config.json");
    const audio = try readJson(AudioConfig, allocator, io, repo, "audio_vae/config.json");
    const video_sched = try readJson(SchedulerFile, allocator, io, repo, "scheduler/scheduler_config.json");
    const audio_sched = try readJson(SchedulerFile, allocator, io, repo, "audio_scheduler/scheduler_config.json");
    mesh_heads = .{
        .dit = dit.num_attention_heads,
        .encoder = encoder_file.text_config.num_attention_heads,
        .encoder_kv = encoder_file.text_config.num_key_value_heads,
    };
    return .{
        .encoder = encoder_file.text_config,
        .dit = dit,
        .vae = vae,
        .audio = audio,
        .video_shift = video_sched.shift,
        .audio_shift = audio_sched.shift,
    };
}

pub const Geometry = struct {
    pixel_w: u32,
    pixel_h: u32,
    frames: u32,
    latent_t: u32,
    latent_h: u32,
    latent_w: u32,
    video_tokens: u32,
    audio_t: u32,
    audio_tokens: u32,
    video_patch_dim: u32,
    audio_dim: u32,

    /// Build a canvas from pixel size and requested duration (seconds).
    /// Width/height must be multiples of 32; duration is 5–15 s. Frame count is
    /// rounded to 24 fps then snapped to a VAE-legal clip.
    pub fn init(width: u32, height: u32, duration_s: f32, dit: DitConfig, vae: VisualConfig) error{ InvalidCanvas, InvalidDuration }!Geometry {
        if (!std.math.isFinite(duration_s) or duration_s < min_duration_s or duration_s > max_duration_s)
            return error.InvalidDuration;
        if (width == 0 or height == 0 or
            width % canvas_multiple != 0 or height % canvas_multiple != 0 or
            @as(u64, width) * height > canvas_max_pixels)
            return error.InvalidCanvas;

        const pt: u32 = @intCast(dit.patch_size[0]);
        const ph: u32 = @intCast(dit.patch_size[1]);
        const pw: u32 = @intCast(dit.patch_size[2]);
        const spatial = vae.spatial();
        const clip_length: u32 = @intCast(vae.clip_length);
        const frames = alignFrameCount(frameCount(duration_s), clip_length);
        const latent_h = height / spatial;
        const latent_w = width / spatial;
        const latent_t = videoLatentFrames(frames, clip_length);
        const audio_t = audioLatentFromFrames(frames);
        return .{
            .pixel_w = width,
            .pixel_h = height,
            .frames = frames,
            .latent_t = latent_t,
            .latent_h = latent_h,
            .latent_w = latent_w,
            .video_tokens = (latent_t / pt) * (latent_h / ph) * (latent_w / pw),
            .audio_t = audio_t,
            .audio_tokens = audio_t * 2,
            .video_patch_dim = @as(u32, @intCast(dit.in_channels)) * pt * ph * pw,
            .audio_dim = @intCast(dit.audio_in_channels),
        };
    }
};

fn frameCount(duration_s: f32) u32 {
    return @intFromFloat(@round(duration_s * video_fps));
}

fn alignFrameCount(frames: u32, clip_length: u32) u32 {
    const n: u32 = if (frames < 1) 1 else frames;
    const rem = n % clip_length;
    if (rem == visual_latents_per_chunk) return n;
    return n + (visual_latents_per_chunk + clip_length - rem) % clip_length;
}

fn videoLatentFrames(aligned_frames: u32, clip_length: u32) u32 {
    return (aligned_frames - visual_latents_per_chunk) / clip_length * visual_latents_per_chunk + 2;
}

fn audioLatentFromFrames(frames: u32) u32 {
    return @intFromFloat(@round(@as(f32, @floatFromInt(frames)) / video_fps * audio_hz));
}

/// Head-wise tensor-parallel mesh on `.model`.
/// DiT and the text encoder shard `.h = .model`; the VAE and audio decoder stay replicated.
pub const Shardings = struct {
    model: zml.Sharding,

    fn tpCount(available: usize) usize {
        var n = available;
        while (n > 1) : (n -= 1) {
            const d: i64 = @intCast(n);
            if (@mod(mesh_heads.dit, d) == 0 and
                @mod(mesh_heads.encoder, d) == 0 and
                @mod(mesh_heads.encoder_kv, d) == 0)
                return n;
        }
        return 1;
    }

    pub fn physicalMesh(
        allocator: std.mem.Allocator,
        target: zml.Target,
        devices: []const zml.platform.Device,
    ) anyerror!zml.Sharding.PhysicalMesh {
        const n = tpCount(devices.len);
        if (n < devices.len) log.warn("tp={d} of {d} GPUs (head dims must divide)", .{ n, devices.len });
        return zml.Sharding.PhysicalMesh.auto(allocator, target, devices[0..n]);
    }

    pub fn init(platform: *zml.Platform) !Shardings {
        return .{
            .model = try platform.registerSharding("model", .mesh(.{ .model = .high_bandwidth })),
        };
    }
};

/// Visual VAE decode tiling (official recipe). Default 1344×768 is 4×7 = 28 tiles.
pub const vae_tile_px: u32 = 256;
pub const vae_tile_overlap_px: u32 = 64;
pub const vae_frame_pre: u32 = 3;
pub const vae_frame_overlap: u32 = 5;
pub const vae_latent_t: u32 = 7;
pub const vae_latent_h: u32 = 16;
pub const vae_latent_w: u32 = 16;
/// Tile batch compiled for the ViT decoder. Replicated on every device count.
pub const vae_tile_batch: u32 = 28;

test "downsample product" {
    try std.testing.expectEqual(@as(u32, 16), product(&.{ 2, 2, 2, 2, 1, 1 }));
    try std.testing.expectEqual(@as(u32, 4), product(&.{ 1, 2, 2, 1, 1, 1 }));
    try std.testing.expectEqual(@as(u32, 800), product(&.{ 2, 4, 4, 5, 5 }));
}

test "parse DitConfig" {
    const json =
        \\{"num_attention_heads":56,"attention_head_dim":128,"hidden_size":5376,"num_layers":50,"num_refiner_layers":2,"in_channels":24,"audio_in_channels":32,"patch_size":[1,2,2],"text_dim":5120,"freq_dim":256,"rope_freq_dim":16,"rope_theta":10000.0,"norm_eps":1e-5,"qk_norm_eps":1e-5,"final_norm_eps":1e-5,"_class_name":"x"}
    ;
    const parsed = try std.json.parseFromSlice(DitConfig, std.testing.allocator, json, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try std.testing.expectEqual(@as(i64, 56), parsed.value.num_attention_heads);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 2 }, &parsed.value.patch_size);
    try std.testing.expectEqual(@as(i64, 96), parsed.value.rotaryDim());
}

test "parse nested text_config" {
    const json =
        \\{"text_config":{"hidden_size":5120,"num_attention_heads":64,"num_key_value_heads":8,"head_dim":128,"rms_norm_eps":1e-6,"rope_theta":5000000}}
    ;
    const parsed = try std.json.parseFromSlice(TextEncoderFile, std.testing.allocator, json, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try std.testing.expectEqual(@as(i64, 5120), parsed.value.text_config.hidden_size);
    try std.testing.expectEqual(@as(i64, 50), parsed.value.text_config.used_hidden_layers);
    try std.testing.expectEqual(@as(f32, 5_000_000.0), parsed.value.text_config.rope_theta);
}

test "parse scheduler shift" {
    const parsed = try std.json.parseFromSlice(SchedulerFile, std.testing.allocator,
        \\{"shift":12.0,"_class_name":"MiniMaxH3Scheduler"}
    , .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try std.testing.expectEqual(@as(f32, 12.0), parsed.value.shift);
}

test "parse nested int arrays" {
    const T = struct { resblock_dilation_sizes: [3][3]i64 };
    const parsed = try std.json.parseFromSlice(T, std.testing.allocator,
        \\{"resblock_dilation_sizes":[[1,3,5],[1,3,5],[1,3,5]]}
    , .{});
    defer parsed.deinit();
    try std.testing.expectEqual(@as(i64, 5), parsed.value.resblock_dilation_sizes[2][2]);
}

test "parse latents_mean f32 array" {
    const T = struct { latents_mean: [2]f32 };
    const parsed = try std.json.parseFromSlice(T, std.testing.allocator,
        \\{"latents_mean":[0.5,-1.25]}
    , .{});
    defer parsed.deinit();
    try std.testing.expectEqual(@as(f32, 0.5), parsed.value.latents_mean[0]);
    try std.testing.expectEqual(@as(f32, -1.25), parsed.value.latents_mean[1]);
}
