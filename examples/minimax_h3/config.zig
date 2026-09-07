//! MiniMax-H3 geometry and pinned layer sizes (snapshot of repo `config.json`, not loaded at runtime).
//!
//! Canvas flags `--width` / `--height` / `--duration` feed `Geometry.init`:
//!   pixels  W×H, multiple of 32, area ≤ 768×1344
//!   frames  round(duration×24), snapped up to `17n+5` (VAE clip)
//!   latents W/16 × H/16 × (5n+2)
//!   DiT tokens  latent_t × (latent_h/2) × (latent_w/2)   (1×2×2 patch)
//!   audio tokens  2 × round(frames/24×40)

const std = @import("std");
const zml = @import("zml");

const log = std.log.scoped(.minimax_h3);

pub const video_fps: f32 = 24.0;
pub const audio_hz: f32 = 40.0;
pub const audio_sample_rate: u32 = 32_000;
pub const visual_spatial: u32 = 16;
pub const visual_temporal: u32 = 4;
/// VAE clip: `17n+5` pixel frames, `5n+2` latent frames.
pub const visual_clip_length: u32 = 17;
pub const visual_latents_per_chunk: u32 = 5;
pub const canvas_multiple: u32 = 32;
pub const canvas_max_pixels: u32 = 768 * 1344;
pub const min_duration_s: f32 = 5.0;
pub const max_duration_s: f32 = 15.0;
/// Rectified-flow time-shift for the video scheduler.
pub const video_shift: f32 = 12.0;
/// Rectified-flow time-shift for the audio scheduler.
pub const audio_shift: f32 = 3.0;
/// Packed-sequence modalities: video, text, audio.
pub const modality_count: i64 = 3;
/// AdaLN / time-embed table width (checkpoint). A packed row usually has 2 unique
/// times (video/text vs audio); `pack.writeRowPlan` unique-sorts and pads to 4.
pub const timestep_slot_count: u32 = 4;

/// Visual VAE decode tiling (official recipe). Default 1344×768 is 4×7 = 28 tiles.
pub const vae_tile_px: u32 = 256;
pub const vae_tile_overlap_px: u32 = 64;
pub const vae_token_drop: u32 = 3;
pub const vae_frame_pre: u32 = 3;
pub const vae_frame_overlap: u32 = 5;
pub const vae_latent_t: u32 = 7;
pub const vae_latent_h: u32 = 16;
pub const vae_latent_w: u32 = 16;
/// Tile batch compiled for the ViT decoder. `.b = .model` only when this divides TP
/// (2/4 GPUs). On 8 GPUs `28 % 8 != 0`, so the batch is replicated.
pub const vae_tile_batch: u32 = 28;

/// Snake-beta upsample / downsample in the audio decoder.
pub const audio_activation_ratio: i64 = 2;
pub const audio_activation_kernel: i64 = 12;

/// DiT (`transformer/config.json` snapshot).
pub const Config = struct {
    hidden_size: i64 = 5376,
    num_layers: i64 = 50,
    num_refiner_layers: i64 = 2,
    num_attention_heads: i64 = 56,
    attention_head_dim: i64 = 128,
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
    audio_t: u32,
    audio_tokens: u32,
    video_patch_dim: u32,
    audio_dim: u32,

    /// Build a canvas from pixel size and requested duration (seconds).
    /// Width/height must be multiples of 32; duration is 5–15 s. Frame count is
    /// rounded to 24 fps then snapped to a VAE-legal `17n+5`.
    pub fn init(width: u32, height: u32, duration_s: f32) error{ InvalidCanvas, InvalidDuration }!Geometry {
        if (!std.math.isFinite(duration_s) or duration_s < min_duration_s or duration_s > max_duration_s)
            return error.InvalidDuration;
        if (width == 0 or height == 0 or
            width % canvas_multiple != 0 or height % canvas_multiple != 0 or
            @as(u64, width) * height > canvas_max_pixels)
            return error.InvalidCanvas;

        const dit: Config = .{};
        const pt: u32 = @intCast(dit.patch_size[0]);
        const ph: u32 = @intCast(dit.patch_size[1]);
        const pw: u32 = @intCast(dit.patch_size[2]);
        const frames = alignFrameCount(frameCount(duration_s));
        const latent_h = height / visual_spatial;
        const latent_w = width / visual_spatial;
        const latent_t = videoLatentFrames(frames);
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

fn alignFrameCount(frames: u32) u32 {
    const n: u32 = if (frames < 1) 1 else frames;
    const rem = n % visual_clip_length;
    if (rem == visual_latents_per_chunk) return n;
    return n + (visual_latents_per_chunk + visual_clip_length - rem) % visual_clip_length;
}

fn videoLatentFrames(aligned_frames: u32) u32 {
    return (aligned_frames - visual_latents_per_chunk) / visual_clip_length * visual_latents_per_chunk + 2;
}

fn audioLatentFromFrames(frames: u32) u32 {
    return @intFromFloat(@round(@as(f32, @floatFromInt(frames)) / video_fps * audio_hz));
}

/// Head-wise tensor-parallel mesh on `.model`.
/// DiT and the text encoder shard `.h = .model`; the VAE is replicated (optional
/// `.b = .model` when the 28-tile batch divides TP). `tpCount` still requires the
/// VAE head count to divide so a later head-TP VAE would fit the same mesh.
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
        return .{
            .model = try platform.registerSharding("model", .mesh(.{ .model = .high_bandwidth })),
        };
    }

    pub fn all(self: Shardings) [1]zml.Sharding {
        return .{self.model};
    }
};

/// Channel-wise latent moments (`vae/config.json` snapshot). Applied before decode.
const visual_latents_mean = [24]f32{
    0.858090341091156,    -0.9606591463088989, 1.0661640167236328,   -0.5090325474739075,
    -0.2727581858634949,  -1.3675414323806763, -0.2553254961967468,  -0.26907554268836975,
    -0.5376840829849243,  -0.0464097298681736, 0.6657370328903198,   0.19690127670764923,
    -0.5460608005523682,  -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607,   -1.1206848621368408,  0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942,  0.22864092886447906,  0.7056031823158264,
};
const visual_latents_std = [24]f32{
    1.2223774194717407, 1.2767263650894165,  1.68317747116088865, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665,   0.96531379222869875, 1.05698859691619875,
    0.841948926448822,  0.7729952931404114,  1.8955937623977661,  0.946841835975647,
    0.7996809482574463, 0.44988900423049925, 0.7197399735450745,  0.69362932443618775,
    2.961095094680786,  2.7694199085235595,  3.0496184825897215,  2.1088054180145265,
    3.276226282119751,  3.1627357006073,     2.28168129920959475, 2.6127843856811525,
};

/// VAE decoder (`vae/config.json` snapshot).
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

/// Channel-wise latent moments (`audio_vae/config.json` snapshot). Applied before decode.
const audio_latents_mean = [32]f32{
    -0.020211687488382354, 0.3876466479950502,   -0.04398279799186767, -0.28591514936373,
    0.08179686214561671,   -0.35782641352446604, 0.040623809960919084, -0.01552534501956604,
    -0.223362481667332,    0.1821006842509091,   0.2941778783780663,   -0.07901167601970885,
    -0.056815072777201,    -0.3699028221860095,  -0.31616315591624855, 0.5905951377425391,
    -0.052139568068853864, 0.013673160263486295, -0.03691647864630577, 0.09732660653298163,
    -0.3394662328788498,   -0.30685677538541667, -0.24504598907458763, -0.034698524462007344,
    0.02868032184767538,   -0.21217779266454084, -0.1678263169941987,  0.3221287889040614,
    -0.1223055851554907,   0.4356604928128464,   -0.0502599202236253,  0.3979258376211797,
};
const audio_latents_std = [32]f32{
    1.6895524230479284, 2.76263727217653,   1.7945344281264435, 1.6801681847309828,
    1.6390226546605453, 2.7788298348882177, 1.7659090095747236, 1.6199757612137327,
    2.6336525640336896, 1.8539356672817833, 2.5056497896915633, 1.811019237886178,
    1.9579657790720237, 1.6685498243529284, 1.4922469314453364, 3.298670198067373,
    1.9491804496832168, 1.8720003270431442, 1.8334080103291832, 1.6488070416529093,
    1.6176957696319716, 1.9131449234774398, 1.5695245398428617, 1.6943659940415912,
    1.8318420762504692, 1.5540637421583379, 1.9344930328968526, 1.599198216109855,
    1.718045989838149,  1.6307219190837705, 1.8661226051202384, 1.5613768203168363,
};

/// Audio VAE decoder (`audio_vae/config.json` snapshot). Stereo, 32 kHz, hop 800.
pub const AudioConfig = struct {
    latent_channels: i64 = 32,
    hop: u32 = 800,
    upsample_rates: [7]i64 = .{ 5, 5, 2, 2, 2, 2, 2 },
    upsample_kernels: [7]i64 = .{ 9, 9, 4, 4, 4, 4, 4 },
    resblock_kernels: [3]i64 = .{ 3, 7, 11 },
    resblock_dilations: [3][3]i64 = .{ .{ 1, 3, 5 }, .{ 1, 3, 5 }, .{ 1, 3, 5 } },
    latents_mean: [32]f32 = audio_latents_mean,
    latents_std: [32]f32 = audio_latents_std,
};
