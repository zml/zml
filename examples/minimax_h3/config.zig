//! Hardcoded 768P MiniMax-H3 geometry and layer sizes from `config.json`.
//!
//!   pixels  1344×768×124
//!   latents 84×48×37     (spatial /16, temporal /4, plus VAE padding)
//!   DiT tokens  37 × 24 × 42   (1×2×2 patchify of the latent grid)

const std = @import("std");
const zml = @import("zml");

const log = std.log.scoped(.minimax_h3);

pub const video_fps: f32 = 24.0;
pub const visual_spatial: u32 = 16;
pub const visual_temporal: u32 = 4;
pub const default_steps: u32 = 30;
/// Rectified-flow time-shift used by the official video scheduler.
pub const video_shift: f32 = 12.0;
/// Packed-sequence modalities in the checkpoint: video, text, unused audio.
pub const modality_count: i64 = 3;

pub const pixel_w: u32 = 1344;
pub const pixel_h: u32 = 768;
pub const duration_s: f32 = 5.0;
pub const canvas_frames: u32 = 124;
pub const latent_t: u32 = 37;
pub const latent_h: u32 = 48;
pub const latent_w: u32 = 84;

/// DiT (`transformer/config.json`). Python: `MiniMaxH3Transformer3DModel`.
pub const Config = struct {
    hidden_size: i64 = 5376,
    num_layers: i64 = 50,
    num_refiner_layers: i64 = 2,
    num_attention_heads: i64 = 56,
    attention_head_dim: i64 = 128,
    in_channels: i64 = 24,
    patch_size: [3]i64 = .{ 1, 2, 2 },
    text_dim: i64 = 5120,
    freq_dim: i64 = 256,
    rope_freq_dim: i64 = 16,
    rope_theta: f32 = 10000.0,
    norm_eps: f32 = 1e-5,
    qk_norm_eps: f32 = 1e-5,
    final_norm_eps: f32 = 1e-5,

    /// MM-RoPE width: 3 axes × `rope_freq_dim`, then duplicated (`ops.ropeCat3`).
    pub fn rotaryDim(self: Config) i64 {
        return 2 * 3 * self.rope_freq_dim;
    }
};

/// Qwen text tower is 64 layers; MiniMax-H3 uses the first 50 as `text_encoder`.
pub const EncoderConfig = struct {
    hidden_size: i64 = 5120,
    used_hidden_layers: i64 = 50,
    num_attention_heads: i64 = 64,
    num_key_value_heads: i64 = 8,
    head_dim: i64 = 128,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 5_000_000.0,
};

pub const Geometry = struct {
    pixel_w: u32,
    pixel_h: u32,
    frames: u32,
    latent_t: u32,
    latent_h: u32,
    latent_w: u32,
    video_tokens: u32,
    video_patch_dim: u32,
};

pub const geo: Geometry = .{
    .pixel_w = pixel_w,
    .pixel_h = pixel_h,
    .frames = canvas_frames,
    .latent_t = latent_t,
    .latent_h = latent_h,
    .latent_w = latent_w,
    .video_tokens = latent_t * (latent_h / 2) * (latent_w / 2),
    .video_patch_dim = 96, // in_channels (24) × 1 × 2 × 2
};

/// Head-wise tensor-parallel mesh on `.model`. GPU count must divide every
/// sharded head dimension (DiT 56, encoder 64/8, VAE 32).
pub const Shardings = struct {
    model: zml.Sharding,

    fn tpCount(available: usize) usize {
        const dit: Config = .{};
        const enc: EncoderConfig = .{};
        const vae: VisualConfig = .{};
        var n = available;
        while (n > 1) : (n -= 1) {
            const d: i64 = @intCast(n);
            if (@mod(dit.num_attention_heads, d) == 0 and
                @mod(enc.num_attention_heads, d) == 0 and
                @mod(enc.num_key_value_heads, d) == 0 and
                @mod(vae.decoder_num_attention_heads, d) == 0)
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
        var strategy: zml.Sharding.Strategy = .init;
        strategy.addBinding(.model, .link);
        return .{
            .model = try platform.registerShardingWithStrategy(
                "model",
                .mesh(.{ .model = .high_bandwidth }),
                strategy,
            ),
        };
    }

    pub fn all(self: Shardings) [1]zml.Sharding {
        return .{self.model};
    }
};

/// Channel-wise latent moments from `vae/config.json`. Applied before decode.
pub const visual_latents_mean = [24]f32{
    0.858090341091156,    -0.9606591463088989,  1.0661640167236328,   -0.5090325474739075,
    -0.2727581858634949,  -1.3675414323806763,  -0.2553254961967468,  -0.26907554268836975,
    -0.5376840829849243,  -0.0464097298681736,  0.6657370328903198,   0.19690127670764923,
    -0.5460608005523682,  -0.4035342037677765,  -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607,    -1.1206848621368408,  0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942,   0.22864092886447906,  0.7056031823158264,
};
pub const visual_latents_std = [24]f32{
    1.2223774194717407,  1.2767263650894165,  1.68317747116088865, 1.7549455165863037,
    1.5636216402053833,  2.194143533706665,   0.96531379222869875, 1.05698859691619875,
    0.841948926448822,   0.7729952931404114,  1.8955937623977661,  0.946841835975647,
    0.7996809482574463,  0.44988900423049925, 0.7197399735450745,  0.69362932443618775,
    2.961095094680786,   2.7694199085235595,  3.0496184825897215,  2.1088054180145265,
    3.276226282119751,   3.1627357006073,     2.28168129920959475, 2.6127843856811525,
};

/// VAE decoder (`vae/config.json`). Python: `AutoencoderKLMiniMaxH3`.
pub const VisualConfig = struct {
    latent_channels: i64 = 24,
    out_channels: i64 = 3,
    decoder_num_layers: i64 = 36,
    decoder_num_attention_heads: i64 = 32,
    decoder_attention_head_dim: i64 = 64,
    decoder_num_register_tokens: i64 = 4,
    decoder_rope_theta: f32 = 100.0,
    decoder_rope_dim_ratio: f32 = 0.75,
    decoder_norm_eps: f32 = 1e-5,
    latents_mean: [24]f32 = visual_latents_mean,
    latents_std: [24]f32 = visual_latents_std,

    pub fn dim(self: VisualConfig) i64 {
        return self.decoder_num_attention_heads * self.decoder_attention_head_dim;
    }

    pub fn rotaryDim(self: VisualConfig) i64 {
        return @intFromFloat(@as(f32, @floatFromInt(self.decoder_attention_head_dim)) * self.decoder_rope_dim_ratio);
    }
};
