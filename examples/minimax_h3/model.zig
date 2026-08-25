// DiT, text encoder, vision, packing, noise, schedule, sampler.

// --- core/config.zig ---
pub const config = struct {
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
    pub const frame_rescale: f32 = 5.0 / 3.0;
    pub const default_short_side: u32 = 768;
    /// Preview canvas: 16:9 → 640×352 after snap-32.
    pub const preview_short_side: u32 = 352;
    pub const tiny_short_side: u32 = 128;
    pub const preview_steps: u32 = 10;
    pub const tiny_steps: u32 = 4;
    pub const video_shift: f32 = 12.0;
    pub const audio_shift: f32 = 3.0;
    pub const encoder_layers_used: u32 = 50;
    pub const qwen_video_fps: f32 = 2.0;
    pub const canvas_multiple: u32 = 32;
    pub const canvas_max_pixels: u32 = 768 * 1344;
    pub const min_aspect: f32 = 0.25;
    pub const max_aspect: f32 = 4.0;
    pub const max_ref_files: u32 = 12;
    pub const max_ref_images: u32 = 9;
    pub const max_ref_videos: u32 = 3;
    pub const max_ref_audios: u32 = 3;
    pub const visual_encode_seed: u64 = 42;

    pub const PosteriorPolicy = enum { sample_seed42, mean };
    pub const posterior: PosteriorPolicy = .mean;

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

    /// Minimum measured device memory for 768p (`--canvas=full`, and auto on large accelerators).
    pub const full_canvas_min_device_bytes: u64 = 40 * 1024 * 1024 * 1024;

    pub fn checkDuration(seconds: f32) !void {
        if (seconds < 4.0 or seconds > 15.0) return error.InvalidDuration;
    }

    /// FL2VA and Ref2VA at preview or larger need this budget; otherwise `--canvas=tiny`.
    pub fn conditionedPreviewNeedsTiny(variant: Variant, short_side: u32, device_bytes: u64) bool {
        if (variant == .t2va) return false;
        return short_side > tiny_short_side and device_bytes != 0 and device_bytes < full_canvas_min_device_bytes;
    }

    pub const Canvas = enum { auto, tiny, preview, full };

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

    /// Unreported device memory (`0`): CPU, Metal, and oneAPI use preview; CUDA, ROCm, TPU, and Neuron use 768p.
    pub fn checkCanvas(choice: Canvas, variant: Variant, short_side: u32, device_bytes: u64) !void {
        if (choice == .full and device_bytes != 0 and device_bytes < full_canvas_min_device_bytes)
            return error.FullCanvasTooLarge;
        if (conditionedPreviewNeedsTiny(variant, short_side, device_bytes))
            return error.ConditionedPreviewTooLarge;
    }

    pub fn canvasForTarget(target: zml.Target, canvas: Canvas, device_bytes: u64) struct { short_side: u32, steps: u32 } {
        return switch (canvas) {
            .tiny => .{ .short_side = tiny_short_side, .steps = tiny_steps },
            .preview => .{ .short_side = preview_short_side, .steps = preview_steps },
            .full => .{ .short_side = default_short_side, .steps = 30 },
            .auto => if (device_bytes != 0 and device_bytes < full_canvas_min_device_bytes)
                .{ .short_side = preview_short_side, .steps = preview_steps }
            else switch (target) {
                .cpu, .metal, .oneapi => .{ .short_side = preview_short_side, .steps = preview_steps },
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

    pub fn parseOptional(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8) !?std.json.Parsed(T) {
        return parseJson(T, allocator, io, dir, name) catch |err| switch (err) {
            error.FileNotFound => null,
            else => return err,
        };
    }

    pub fn loadDitConfig(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !Config {
        const parsed = try parseOptional(Config, allocator, io, dir, "config.json") orelse return Config.official();
        defer parsed.deinit();
        return parsed.value.resolve();
    }

    pub fn loadEncoderConfig(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !EncoderConfig {
        const parsed = try parseOptional(EncoderFileConfig, allocator, io, dir, "config.json") orelse return EncoderConfig.official();
        defer parsed.deinit();
        return parsed.value.resolve();
    }

    pub const Size = struct { w: u32, h: u32 };

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

    pub fn pixelSize(aspect: Aspect, short_side: u32) Size {
        const r = aspect.ratio();
        return resolveCanvas(@floatFromInt(r.w), @floatFromInt(r.h), short_side, canvas_max_pixels) catch unreachable;
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
};

// --- core/policy.zig ---
pub const policy = struct {
    const std = @import("std");

    const zml = @import("zml");

    pub const AttnKind = enum { vanilla, cuda_fa2 };

    pub const Decision = struct {
        attention: AttnKind,
        resident_blocks: u32,
        group_size: u32,
        tile_batch: u32,
        score_bytes: u64,
        fa2_scratch_bytes: u64,
        adaln_table_bytes: u64,
        activation_bytes: u64,
        block_core_bytes: u64,
    };

    pub const Query = struct {
        target: zml.Target,
        dtype: zml.DataType,
        head_dim: i64,
        heads: i64,
        seq: u64,
        causal: bool,
        tp: u32,
    };

    pub fn selectAttention(q: Query) AttnKind {
        if (q.target != .cuda) return .vanilla;
        if (q.dtype != .bf16 and q.dtype != .f16) return .vanilla;
        if (q.head_dim < 16 or q.head_dim > 256 or @rem(q.head_dim, 8) != 0) return .vanilla;
        if (q.heads <= 0 or @rem(q.heads, @as(i64, @max(1, q.tp))) != 0) return .vanilla;
        if (q.causal and q.seq < 2) return .vanilla;
        const heads_local: u64 = @intCast(@divExact(q.heads, @as(i64, @max(1, q.tp))));
        const quadratic = q.seq * q.seq * 4 * heads_local;
        const linear = q.seq * @as(u64, @intCast(q.head_dim)) * heads_local * 8;
        if (quadratic <= linear * 4) return .vanilla;
        return .cuda_fa2;
    }

    pub fn sdpaScoreBytes(seq: u64, heads: i64, tp: u32) u64 {
        const heads_local: u64 = @intCast(@divExact(@max(heads, 1), @as(i64, @max(1, tp))));
        return seq * seq * 4 * heads_local;
    }

    pub fn fa2ScratchBytes(seq: u64, heads: i64, head_dim: i64, tp: u32) u64 {
        const heads_local: u64 = @intCast(@divExact(@max(heads, 1), @as(i64, @max(1, tp))));
        const hd: u64 = @intCast(@max(head_dim, 1));
        const lse = seq * heads_local * 4;
        const lse_accum = heads_local * hd * 4;
        const out_accum = seq * heads_local * hd * 4;
        return lse + lse_accum + out_accum;
    }

    pub fn adalnTableBytes(steps: u32, hidden: i64, layers: i64, dtype_bytes: u32) u64 {
        const slots = packing.timestep_slot_count;
        const mods: u64 = @intCast(config.modality_count);
        const hid: u64 = @intCast(@max(hidden, 1));
        const per_block = @as(u64, steps) * slots * mods * 6 * hid * dtype_bytes;
        const final = @as(u64, steps) * slots * 2 * hid * dtype_bytes;
        return per_block * @as(u64, @intCast(@max(layers, 0))) + final;
    }

    pub fn groupSize(resident: u32) u32 {
        if (resident <= 1) return 1;
        if (resident < 4) return resident;
        return 4;
    }

    pub fn tileBatch(tile_count: u32, tile_act_bytes: u64, headroom: u64, devices: u32) u32 {
        if (tile_count == 0) return 1;
        const fit: u32 = if (tile_act_bytes == 0)
            tile_count
        else
            @intCast(@min(@as(u64, tile_count), @max(@as(u64, 1), headroom / tile_act_bytes)));
        const dev = @max(1, devices);
        if (dev == 1) return @max(1, fit);
        if (fit < dev) return 1;
        return (fit / dev) * dev;
    }

    pub fn decide(args: struct {
        target: zml.Target,
        seq: u64,
        hidden: i64,
        heads: i64,
        head_dim: i64,
        layers: u32,
        steps: u32,
        dtype: zml.DataType,
        device_bytes: u64,
        tp: u32,
        devices: u32,
        block_core_bytes: u64,
        dtype_bytes: u32,
        tile_count: u32 = 1,
        tile_act_bytes: u64 = 0,
    }) Decision {
        const attn = selectAttention(.{
            .target = args.target,
            .dtype = args.dtype,
            .head_dim = args.head_dim,
            .heads = args.heads,
            .seq = args.seq,
            .causal = false,
            .tp = args.tp,
        });
        const hid: u64 = @intCast(@max(args.hidden, 1));
        const act = args.seq * hid * args.dtype_bytes * 8;
        const scores = sdpaScoreBytes(args.seq, args.heads, args.tp);
        const scratch = if (attn == .cuda_fa2)
            fa2ScratchBytes(args.seq, args.heads, args.head_dim, args.tp)
        else
            scores;
        const tables = adalnTableBytes(args.steps, args.hidden, args.layers, args.dtype_bytes);
        const collective = act / 4;
        const reserved = act + scratch + collective + tables;
        const budget = if (args.device_bytes == 0)
            0
        else
            args.device_bytes * 85 / 100;
        const headroom = budget -| reserved;
        const per_core = if (args.block_core_bytes == 0) 0 else args.block_core_bytes;
        const resident: u32 = if (per_core == 0 or headroom < per_core)
            0
        else
            @intCast(@min(@as(u64, args.layers), headroom / per_core));
        const group = groupSize(resident);
        const tiles = tileBatch(args.tile_count, args.tile_act_bytes, headroom / 4, @max(1, args.devices));
        return .{
            .attention = attn,
            .resident_blocks = resident,
            .group_size = group,
            .tile_batch = tiles,
            .score_bytes = scores,
            .fa2_scratch_bytes = if (attn == .cuda_fa2) scratch else 0,
            .adaln_table_bytes = tables,
            .activation_bytes = act,
            .block_core_bytes = per_core,
        };
    }

    pub fn dtypeBytes(dt: zml.DataType) u32 {
        return @intCast(dt.sizeOf());
    }
};

// --- core/weights.zig ---
pub const weights = struct {
    const std = @import("std");

    const zml = @import("zml");

    /// Loader sized for one streamed transformer block (~768 MiB).
    pub const loader_opts: zml.io.Loader.Opts = .{
        .dma_chunks = 8,
        .dma_chunk_size = 64 * zml.MiB,
        .parallelism = 8,
    };

    pub fn initLoader(allocator: std.mem.Allocator, platform: *const zml.Platform) !zml.io.Loader {
        return .init(allocator, platform, loader_opts);
    }

    pub fn populate(
        loader: *zml.io.Loader,
        io: std.Io,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        comptime T: type,
        model: *const T,
        buffers: *zml.Bufferized(T),
        progress: *std.Progress.Node,
    ) !void {
        var span = zml.tracer.span("h3.weights.load", .{});
        defer span.end();
        loader.load(io, T, model, buffers, store, shardings, .{ .progress = progress });
        try loader.await(io);
    }

    pub fn modelBytes(model: anytype) u64 {
        const Ctx = struct {
            n: u64 = 0,
            fn add(ctx: *@This(), t: *const zml.Tensor) void {
                ctx.n += t.shape().byteSize();
            }
        };
        var ctx: Ctx = .{};
        zml.meta.visit(Ctx.add, &ctx, model);
        return ctx.n;
    }

    /// Rebind a compiled runner to the next streamed layer. `bake` is incremental;
    /// reset the count or the previous layer stays bound.
    pub fn rebake(runner: anytype, next: anytype) void {
        runner.args.baked_count = 0;
        runner.args.bake(next);
    }

    pub fn load(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        comptime T: type,
        model: *const T,
        progress: *std.Progress.Node,
        loader: ?*zml.io.Loader,
    ) !zml.Bufferized(T) {
        var buffers = try zml.mem.bufferize(allocator, T, model);
        if (loader) |shared| {
            try populate(shared, io, store, shardings, T, model, &buffers, progress);
            return buffers;
        }
        var owned = try initLoader(allocator, platform);
        defer owned.deinit();
        try populate(&owned, io, store, shardings, T, model, &buffers, progress);
        return buffers;
    }
};

// --- model/packing.zig ---
pub const packing = struct {
    const std = @import("std");

    pub const Modality = enum(u8) {
        video = 0,
        text = 1,
        audio = 2,
    };

    pub const SegmentKind = enum {
        text,
        condition_video,
        condition_audio,
        target_audio,
        target_video,
    };

    pub const Position = struct { t: f32, h: f32, w: f32 };

    pub const SequenceSegment = struct {
        start: u32,
        end: u32,
        kind: SegmentKind,
        source_index: i32 = -1,
    };

    pub const ConditionVideo = struct {
        latent_t: u32,
        latent_h: u32,
        latent_w: u32,
        /// 0 = first frame, last frame otherwise. Ignored for Ref2VA.
        keyframe_index: i32 = 0,
        /// When set, place this condition at that pixel frame (negative from the end).
        guide_frame: ?i32 = null,
    };

    pub const ConditionAudio = struct {
        latent_t: u32,
    };

    pub const ReferenceKind = enum { image, video, audio, video_audio };

    pub const ReferenceBlock = struct {
        kind: ReferenceKind,
        video_index: i32 = -1,
        audio_index: i32 = -1,
    };

    /// Fixed AdaLN / time-embed slots. Distinct rows stay valid when video and
    /// condition times diverge; equal values still produce equal `temb` rows.
    pub const timestep_slot_count: u32 = 4;
    pub const TimeSlot = enum(u32) {
        video = 0,
        audio = 1,
        visual_cond = 2,
        audio_cond = 3,
    };

    pub fn timestepValues(video_t: f32, audio_t: f32) [timestep_slot_count]f32 {
        return .{
            video_t,
            audio_t,
            @max(video_t, config.visual_cond_timestep),
            @max(audio_t, 1.0),
        };
    }

    /// `mask[i] != 0` keeps the row on `slot`; otherwise the row is unchanged.
    pub fn applyRowMask(timestep_indices: []u32, mask: []const u8, slot: u32) void {
        const n = @min(timestep_indices.len, mask.len);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            if (mask[i] != 0) timestep_indices[i] = slot;
        }
    }

    pub fn writeTimesteps(out: []f32, video_t: f32, audio_t: f32) void {
        const values = timestepValues(video_t, audio_t);
        const n = @min(out.len, values.len);
        @memcpy(out[0..n], values[0..n]);
        if (n > 0) {
            for (n..out.len) |i| out[i] = values[n - 1];
        }
    }

    pub const Layout = struct {
        positions: []Position,
        token_tags: []u8,
        timestep_indices: []u32,
        timesteps: []f32,
        segments: []SequenceSegment,
        text_indices: []u32,
        video_indices: []u32,
        audio_indices: []u32,
        target_video_start: u32,
        target_video_end: u32,
        target_audio_start: u32,
        target_audio_end: u32,

        pub fn deinit(self: Layout, allocator: std.mem.Allocator) void {
            allocator.free(self.positions);
            allocator.free(self.token_tags);
            allocator.free(self.timestep_indices);
            allocator.free(self.timesteps);
            allocator.free(self.segments);
            allocator.free(self.text_indices);
            allocator.free(self.video_indices);
            allocator.free(self.audio_indices);
        }

        pub fn seqLen(self: Layout) u32 {
            return @intCast(self.positions.len);
        }

        pub fn adalnIndex(self: Layout, row: usize) u32 {
            return self.timestep_indices[row] * @as(u32, @intCast(config.modality_count)) + self.token_tags[row];
        }
    };

    pub const BuildArgs = struct {
        text_len: u32,
        latent_t: u32,
        latent_h: u32,
        latent_w: u32,
        audio_t: u32,
        video_t: f32,
        audio_t_noise: f32,
        condition_videos: []const ConditionVideo = &.{},
        condition_audios: []const ConditionAudio = &.{},
        references: []const ReferenceBlock = &.{},
        text_tags: []const u8 = &.{},
        pixel_frames: u32 = 0,
    };

    const video_spans = [_]u32{ 1, 4, 4, 4, 4 };

    pub fn videoSpan(frame: u32) f32 {
        return config.frame_rescale * @as(f32, @floatFromInt(video_spans[frame % video_spans.len]));
    }

    fn guideStartT(text_len: u32, frame: i32, frames: u32, duration: f32) f32 {
        const base: f32 = @floatFromInt(text_len);
        if (frames == 0) return base;
        const last: i64 = @as(i64, frames) - 1;
        const idx: i64 = if (frame < 0) last + 1 + frame else frame;
        const clamped: u32 = @intCast(std.math.clamp(idx, 0, last));
        if (clamped == 0) return base;
        if (clamped == @as(u32, @intCast(last))) return base + duration - config.frame_rescale;
        const frac = @as(f32, @floatFromInt(clamped)) / @as(f32, @floatFromInt(last));
        return base + frac * duration;
    }

    pub fn videoDuration(latent_t: u32) f32 {
        var total: f32 = 0;
        for (0..latent_t) |t| total += videoSpan(@intCast(t));
        return total;
    }

    pub fn spatialAxis(dim: u32, sqrt_area: f32, out: []f32) []f32 {
        const count = dim / 2;
        std.debug.assert(out.len >= count);
        const ratio = @as(f32, @floatFromInt(dim)) / sqrt_area;
        const step = ratio / @as(f32, @floatFromInt(count));
        for (0..count) |i| {
            out[i] = (@as(f32, @floatFromInt(i)) * step + (1.0 - ratio) * 0.5) * 32.0;
        }
        return out[0..count];
    }

    const Builder = struct {
        allocator: std.mem.Allocator,
        positions: std.ArrayList(Position),
        token_tags: std.ArrayList(u8),
        timestep_indices: std.ArrayList(u32),
        timesteps: std.ArrayList(f32),
        segments: std.ArrayList(SequenceSegment),
        text_indices: std.ArrayList(u32),
        video_indices: std.ArrayList(u32),
        audio_indices: std.ArrayList(u32),

        fn init(allocator: std.mem.Allocator) Builder {
            return .{
                .allocator = allocator,
                .positions = .empty,
                .token_tags = .empty,
                .timestep_indices = .empty,
                .timesteps = .empty,
                .segments = .empty,
                .text_indices = .empty,
                .video_indices = .empty,
                .audio_indices = .empty,
            };
        }

        fn deinit(self: *Builder) void {
            self.positions.deinit(self.allocator);
            self.token_tags.deinit(self.allocator);
            self.timestep_indices.deinit(self.allocator);
            self.timesteps.deinit(self.allocator);
            self.segments.deinit(self.allocator);
            self.text_indices.deinit(self.allocator);
            self.video_indices.deinit(self.allocator);
            self.audio_indices.deinit(self.allocator);
        }

        fn row(self: *const Builder) u32 {
            return @intCast(self.positions.items.len);
        }

        fn appendRow(self: *Builder, pos: Position, tag: Modality, time_row: u32) !void {
            try self.positions.append(self.allocator, pos);
            try self.token_tags.append(self.allocator, @intFromEnum(tag));
            try self.timestep_indices.append(self.allocator, time_row);
        }

        fn finish(self: *Builder, target_video: struct { start: u32, end: u32 }, target_audio: struct { start: u32, end: u32 }) !Layout {
            return .{
                .positions = try self.positions.toOwnedSlice(self.allocator),
                .token_tags = try self.token_tags.toOwnedSlice(self.allocator),
                .timestep_indices = try self.timestep_indices.toOwnedSlice(self.allocator),
                .timesteps = try self.timesteps.toOwnedSlice(self.allocator),
                .segments = try self.segments.toOwnedSlice(self.allocator),
                .text_indices = try self.text_indices.toOwnedSlice(self.allocator),
                .video_indices = try self.video_indices.toOwnedSlice(self.allocator),
                .audio_indices = try self.audio_indices.toOwnedSlice(self.allocator),
                .target_video_start = target_video.start,
                .target_video_end = target_video.end,
                .target_audio_start = target_audio.start,
                .target_audio_end = target_audio.end,
            };
        }
    };

    fn appendText(b: *Builder, text_len: u32, text_tags: []const u8, time_row: u32) !void {
        const start = b.row();
        var run_start: u32 = 0;
        var current: u8 = if (text_tags.len == 0) @intFromEnum(Modality.text) else text_tags[0];
        for (0..text_len) |i| {
            const tag: u8 = if (i < text_tags.len) text_tags[i] else @intFromEnum(Modality.text);
            if (i > 0 and tag != current) {
                try b.segments.append(b.allocator, .{
                    .start = start + run_start,
                    .end = start + @as(u32, @intCast(i)),
                    .kind = .text,
                });
                run_start = @intCast(i);
                current = tag;
            }
            try b.appendRow(.{ .t = @floatFromInt(i), .h = 0, .w = 0 }, @enumFromInt(tag), time_row);
            try b.text_indices.append(b.allocator, start + @as(u32, @intCast(i)));
        }
        try b.segments.append(b.allocator, .{
            .start = start + run_start,
            .end = start + text_len,
            .kind = .text,
        });
    }

    fn appendVideoGrid(
        b: *Builder,
        h_axis: []const f32,
        w_axis: []const f32,
        latent_t: u32,
        start_t: f32,
        time_row: u32,
        kind: SegmentKind,
        source_index: i32,
    ) !struct { start: u32, end: u32, cursor: f32 } {
        const start = b.row();
        var cursor = start_t;
        for (0..latent_t) |t| {
            for (h_axis) |h| {
                for (w_axis) |w| {
                    const idx = b.row();
                    try b.appendRow(.{ .t = cursor, .h = h, .w = w }, .video, time_row);
                    try b.video_indices.append(b.allocator, idx);
                }
            }
            cursor += videoSpan(@intCast(t));
        }
        const end = b.row();
        try b.segments.append(b.allocator, .{
            .start = start,
            .end = end,
            .kind = kind,
            .source_index = source_index,
        });
        return .{ .start = start, .end = end, .cursor = cursor };
    }

    fn appendAudioRows(
        b: *Builder,
        length: u32,
        cursor: f32,
        w_low: f32,
        w_high: f32,
        time_row: u32,
        kind: SegmentKind,
        source_index: i32,
    ) !struct { start: u32, end: u32 } {
        const start = b.row();
        const widths = [_]f32{ w_low, w_high };
        for (widths) |w| {
            for (0..length) |t| {
                const idx = b.row();
                try b.appendRow(.{ .t = cursor + @as(f32, @floatFromInt(t)), .h = 0, .w = w }, .audio, time_row);
                try b.audio_indices.append(b.allocator, idx);
            }
        }
        const end = b.row();
        try b.segments.append(b.allocator, .{
            .start = start,
            .end = end,
            .kind = kind,
            .source_index = source_index,
        });
        return .{ .start = start, .end = end };
    }

    pub fn build(allocator: std.mem.Allocator, args: BuildArgs) !Layout {
        var b = Builder.init(allocator);
        errdefer b.deinit();

        const sqrt_area = @sqrt(@as(f32, @floatFromInt(args.latent_h * args.latent_w)));
        var h_buf: [256]f32 = undefined;
        var w_buf: [256]f32 = undefined;
        const h_axis = spatialAxis(args.latent_h, sqrt_area, &h_buf);
        const w_axis = spatialAxis(args.latent_w, sqrt_area, &w_buf);

        const times = timestepValues(args.video_t, args.audio_t_noise);
        try b.timesteps.appendSlice(b.allocator, &times);
        const video_time = @intFromEnum(TimeSlot.video);
        const audio_time = @intFromEnum(TimeSlot.audio);
        const cond_time = @intFromEnum(TimeSlot.visual_cond);
        const audio_cond_time = @intFromEnum(TimeSlot.audio_cond);

        try appendText(&b, args.text_len, args.text_tags, video_time);

        if (args.references.len == 0) {
            const duration = videoDuration(args.latent_t);
            for (args.condition_videos, 0..) |cond, index| {
                var ch_buf: [256]f32 = undefined;
                var cw_buf: [256]f32 = undefined;
                const area = @sqrt(@as(f32, @floatFromInt(cond.latent_h * cond.latent_w)));
                const ch = spatialAxis(cond.latent_h, area, &ch_buf);
                const cw = spatialAxis(cond.latent_w, area, &cw_buf);
                const is_first = cond.keyframe_index == 0;
                const keyframe_t = if (cond.guide_frame) |gf|
                    guideStartT(args.text_len, gf, args.pixel_frames, duration)
                else if (is_first)
                    @as(f32, @floatFromInt(args.text_len))
                else
                    @as(f32, @floatFromInt(args.text_len)) + duration - config.frame_rescale;
                _ = try appendVideoGrid(&b, ch, cw, cond.latent_t, keyframe_t, cond_time, .condition_video, @intCast(index));
            }
        } else {
            var cursor: f32 = @floatFromInt(args.text_len);
            for (args.references) |block| {
                var block_end = cursor;
                if (block.kind == .audio or block.kind == .video_audio) {
                    if (block.audio_index < 0) return error.MissingReferenceAudio;
                    const audio = args.condition_audios[@intCast(block.audio_index)];
                    var w_low = w_axis[0];
                    var w_high = w_axis[w_axis.len - 1];
                    if (block.video_index >= 0) {
                        const video = args.condition_videos[@intCast(block.video_index)];
                        var cw_buf: [256]f32 = undefined;
                        const area = @sqrt(@as(f32, @floatFromInt(video.latent_h * video.latent_w)));
                        const cw = spatialAxis(video.latent_w, area, &cw_buf);
                        w_low = cw[0];
                        w_high = cw[cw.len - 1];
                    }
                    _ = try appendAudioRows(&b, audio.latent_t, cursor, w_low, w_high, audio_cond_time, .condition_audio, block.audio_index);
                    block_end = @max(block_end, cursor + @as(f32, @floatFromInt(audio.latent_t)));
                }
                if (block.kind != .audio) {
                    if (block.video_index < 0) return error.MissingReferenceVideo;
                    const video = args.condition_videos[@intCast(block.video_index)];
                    var ch_buf: [256]f32 = undefined;
                    var cw_buf: [256]f32 = undefined;
                    const area = @sqrt(@as(f32, @floatFromInt(video.latent_h * video.latent_w)));
                    const ch = spatialAxis(video.latent_h, area, &ch_buf);
                    const cw = spatialAxis(video.latent_w, area, &cw_buf);
                    const placed = try appendVideoGrid(&b, ch, cw, video.latent_t, cursor, cond_time, .condition_video, block.video_index);
                    block_end = if (block.kind == .image)
                        @max(block_end, cursor + 1.0)
                    else
                        @max(block_end, placed.cursor);
                }
                cursor = block_end;
            }
        }

        var cursor: f32 = @floatFromInt(args.text_len);
        if (args.references.len != 0) {
            cursor = 0;
            for (b.positions.items) |pos| cursor = @max(cursor, pos.t);
        }

        const audio = try appendAudioRows(
            &b,
            args.audio_t,
            cursor,
            w_axis[0],
            w_axis[w_axis.len - 1],
            audio_time,
            .target_audio,
            -1,
        );
        const video = try appendVideoGrid(&b, h_axis, w_axis, args.latent_t, cursor, video_time, .target_video, -1);

        return b.finish(.{ .start = video.start, .end = video.end }, .{ .start = audio.start, .end = audio.end });
    }

    /// Video noise is `(C, T, H, W)`. Patchify consumes `{t,h,w,c}`.
    pub fn nchwToThwc(dst: []f32, src: []const f32, c: u32, t: u32, h: u32, w: u32) void {
        std.debug.assert(dst.len == src.len);
        std.debug.assert(src.len == @as(usize, c) * t * h * w);
        var ci: u32 = 0;
        while (ci < c) : (ci += 1) {
            var ti: u32 = 0;
            while (ti < t) : (ti += 1) {
                var hi: u32 = 0;
                while (hi < h) : (hi += 1) {
                    var wi: u32 = 0;
                    while (wi < w) : (wi += 1) {
                        const s = ((((ci * t) + ti) * h + hi) * w) + wi;
                        const d = ((((ti * h) + hi) * w + wi) * c) + ci;
                        dst[d] = src[s];
                    }
                }
            }
        }
    }

    /// Patchify `{t,h,w,c}` with `(pt,ph,pw)` into rows of `c*pt*ph*pw`.
    pub fn patchify(
        allocator: std.mem.Allocator,
        src: []const f32,
        t: u32,
        h: u32,
        w: u32,
        c: u32,
        patch: [3]i64,
    ) ![]f32 {
        const pt: u32 = @intCast(patch[0]);
        const ph: u32 = @intCast(patch[1]);
        const pw: u32 = @intCast(patch[2]);
        std.debug.assert(t % pt == 0 and h % ph == 0 and w % pw == 0);
        const rows = (t / pt) * (h / ph) * (w / pw);
        const width = c * pt * ph * pw;
        const out = try allocator.alloc(f32, rows * width);
        var row: usize = 0;
        var tt: u32 = 0;
        while (tt < t) : (tt += pt) {
            var hh: u32 = 0;
            while (hh < h) : (hh += ph) {
                var ww: u32 = 0;
                while (ww < w) : (ww += pw) {
                    var dst: usize = 0;
                    // Permute to (T',H',W',C,pt,ph,pw).
                    for (0..c) |ch| {
                        for (0..pt) |dt| {
                            for (0..ph) |dh| {
                                for (0..pw) |dw| {
                                    const src_t = tt + @as(u32, @intCast(dt));
                                    const src_h = hh + @as(u32, @intCast(dh));
                                    const src_w = ww + @as(u32, @intCast(dw));
                                    const base = (((@as(usize, src_t) * h + src_h) * w + src_w) * c) + ch;
                                    out[row * width + dst] = src[base];
                                    dst += 1;
                                }
                            }
                        }
                    }
                    row += 1;
                }
            }
        }
        return out;
    }

    pub fn unpatchify(
        allocator: std.mem.Allocator,
        src: []const f32,
        t: u32,
        h: u32,
        w: u32,
        c: u32,
        patch: [3]i64,
    ) ![]f32 {
        const pt: u32 = @intCast(patch[0]);
        const ph: u32 = @intCast(patch[1]);
        const pw: u32 = @intCast(patch[2]);
        const width = c * pt * ph * pw;
        const out = try allocator.alloc(f32, @as(usize, t) * h * w * c);
        var row: usize = 0;
        var tt: u32 = 0;
        while (tt < t) : (tt += pt) {
            var hh: u32 = 0;
            while (hh < h) : (hh += ph) {
                var ww: u32 = 0;
                while (ww < w) : (ww += pw) {
                    var src_i: usize = 0;
                    for (0..c) |ch| {
                        for (0..pt) |dt| {
                            for (0..ph) |dh| {
                                for (0..pw) |dw| {
                                    const dst_t = tt + @as(u32, @intCast(dt));
                                    const dst_h = hh + @as(u32, @intCast(dh));
                                    const dst_w = ww + @as(u32, @intCast(dw));
                                    const base = (((@as(usize, dst_t) * h + dst_h) * w + dst_w) * c) + ch;
                                    out[base] = src[row * width + src_i];
                                    src_i += 1;
                                }
                            }
                        }
                    }
                    row += 1;
                }
            }
        }
        return out;
    }
};

// --- model/scheduler.zig ---
pub const scheduler = struct {
    const std = @import("std");

    /// Rectified-flow Euler (`eta = 0`). Transformer predicts data-ward velocity:
    /// `x0 = x_t + sigma * v`, `t = 1 - sigma` in `[0, 1]`.
    pub const Schedule = struct {
        shift: f32,
        sigmas: []f32,
        timesteps: []f32,

        pub fn init(allocator: std.mem.Allocator, shift: f32, num_inference_steps: u32) !Schedule {
            if (shift <= 0) return error.InvalidShift;
            if (num_inference_steps < 2) return error.TooFewSteps;

            const raw = try allocator.alloc(f32, num_inference_steps);
            defer allocator.free(raw);
            for (raw, 0..) |*sigma, i| {
                const base = 1.0 - @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(num_inference_steps - 1));
                sigma.* = shiftSigma(base, shift);
            }

            var unique = try std.ArrayList(f32).initCapacity(allocator, raw.len);
            errdefer unique.deinit(allocator);
            unique.appendAssumeCapacity(raw[0]);
            for (raw[1..]) |sigma| {
                if (sigma != unique.items[unique.items.len - 1]) {
                    unique.appendAssumeCapacity(sigma);
                }
            }
            if (unique.items[unique.items.len - 1] != 0.0) {
                try unique.append(allocator, 0.0);
            }
            if (unique.items.len < 2) return error.DegenerateSchedule;

            const sigmas = try unique.toOwnedSlice(allocator);
            errdefer allocator.free(sigmas);

            const timesteps = try allocator.alloc(f32, sigmas.len - 1);
            for (timesteps, sigmas[0 .. sigmas.len - 1]) |*t, sigma| {
                t.* = 1.0 - sigma;
            }

            return .{
                .shift = shift,
                .sigmas = sigmas,
                .timesteps = timesteps,
            };
        }

        pub fn deinit(self: Schedule, allocator: std.mem.Allocator) void {
            allocator.free(self.sigmas);
            allocator.free(self.timesteps);
        }

        pub fn stepCount(self: Schedule) usize {
            return self.timesteps.len;
        }

        pub fn scaleNoise(t: f32, clean: f32, noisy: f32) f32 {
            return t * clean + (1.0 - t) * noisy;
        }
    };

    pub const DualSchedule = struct {
        video: Schedule,
        audio: Schedule,

        pub fn init(allocator: std.mem.Allocator, steps: u32, video_shift: f32, audio_shift: f32) !DualSchedule {
            const video = try Schedule.init(allocator, video_shift, steps);
            errdefer video.deinit(allocator);
            const audio = try Schedule.init(allocator, audio_shift, steps);
            return .{ .video = video, .audio = audio };
        }

        pub fn deinit(self: DualSchedule, allocator: std.mem.Allocator) void {
            self.video.deinit(allocator);
            self.audio.deinit(allocator);
        }
    };

    pub fn shiftSigma(sigma: f32, shift: f32) f32 {
        return shift * sigma / (1.0 + (shift - 1.0) * sigma);
    }

    /// Maps a video-schedule sigma onto the audio schedule (`from_shift` → `to_shift`).
    pub fn timeShiftSigma(sigma: f32, from_shift: f32, to_shift: f32) f32 {
        const base = sigma / (from_shift + sigma * (1.0 - from_shift));
        return to_shift * base / (1.0 + (to_shift - 1.0) * base);
    }

    pub fn timestepEmbedding(timesteps: []const f32, dim: usize, flip_sin_to_cos: bool, out: []f32) void {
        std.debug.assert(out.len == timesteps.len * dim);
        std.debug.assert(dim % 2 == 0);
        const half = dim / 2;
        for (timesteps, 0..) |t, row| {
            const dst = out[row * dim ..][0..dim];
            for (0..half) |i| {
                const freq = @exp(-@log(@as(f32, 10000.0)) * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half)));
                const angle = t * freq;
                if (flip_sin_to_cos) {
                    dst[i] = @cos(angle);
                    dst[half + i] = @sin(angle);
                } else {
                    dst[i] = @sin(angle);
                    dst[half + i] = @cos(angle);
                }
            }
        }
    }
};

// --- model/noise.zig ---
pub const noise = struct {
    const std = @import("std");

    /// mt19937, 16-wide Box-Muller. Draw order: condition NCHW, target `(C,T,H,W)`, audio `(2*T,C)`.
    const n = 624;
    const m = 397;
    const matrix_a: u32 = 0x9908b0df;
    const umask: u32 = 0x80000000;
    const lmask: u32 = 0x7fffffff;
    const two_pi: f32 = @floatCast(2.0 * std.math.pi);

    pub const Generator = struct {
        seed: u64,
        left: i32,
        next: u32,
        state: [n]u32,

        pub fn init(seed: u64) Generator {
            var self: Generator = .{
                .seed = seed,
                .left = 1,
                .next = 0,
                .state = undefined,
            };
            self.state[0] = @truncate(seed);
            var j: usize = 1;
            while (j < n) : (j += 1) {
                const prev = self.state[j - 1];
                self.state[j] = 1812433253 *% (prev ^ (prev >> 30)) +% @as(u32, @intCast(j));
            }
            return self;
        }

        pub fn reset(self: *Generator) void {
            self.* = init(self.seed);
        }

        pub fn random(self: *Generator) u32 {
            self.left -= 1;
            if (self.left == 0) self.nextState();
            var y = self.state[self.next];
            self.next += 1;
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c5680;
            y ^= (y << 15) & 0xefc60000;
            y ^= y >> 18;
            return y;
        }

        pub fn uniform01(self: *Generator) f32 {
            const mask: u32 = (1 << 24) - 1;
            const divisor: f32 = 1.0 / 16777216.0;
            return @as(f32, @floatFromInt(self.random() & mask)) * divisor;
        }

        fn mixBits(u: u32, v: u32) u32 {
            return (u & umask) | (v & lmask);
        }

        fn twist(u: u32, v: u32) u32 {
            return (mixBits(u, v) >> 1) ^ if (v & 1 != 0) matrix_a else @as(u32, 0);
        }

        fn nextState(self: *Generator) void {
            self.left = n;
            self.next = 0;
            var p: usize = 0;
            var j: i32 = n - m + 1;
            while (true) {
                j -= 1;
                if (j == 0) break;
                self.state[p] = self.state[p + m] ^ twist(self.state[p], self.state[p + 1]);
                p += 1;
            }
            j = m;
            while (true) {
                j -= 1;
                if (j == 0) break;
                self.state[p] = self.state[p + m - n] ^ twist(self.state[p], self.state[p + 1]);
                p += 1;
            }
            self.state[p] = self.state[p + m - n] ^ twist(self.state[p], self.state[0]);
        }
    };

    /// 16-wide Box-Muller when `numel >= 16`; serial otherwise.
    pub fn randn(gen: *Generator, out: []f32) void {
        if (out.len < 16) {
            randnSerial(gen, out);
            return;
        }
        for (out) |*x| x.* = gen.uniform01();
        var i: usize = 0;
        while (i + 16 <= out.len) : (i += 16) {
            boxMuller16(out[i..][0..16]);
        }
        if (out.len % 16 != 0) {
            const tail = out[out.len - 16 ..];
            for (tail) |*x| x.* = gen.uniform01();
            boxMuller16(tail[0..16]);
        }
    }

    fn boxMuller16(data: []f32) void {
        var j: usize = 0;
        while (j < 8) : (j += 1) {
            const unit_a = 1.0 - data[j];
            const unit_b = data[j + 8];
            const radius = @sqrt(-2.0 * @log(unit_a));
            const theta = two_pi * unit_b;
            data[j] = radius * @cos(theta);
            data[j + 8] = radius * @sin(theta);
        }
    }

    fn randnSerial(gen: *Generator, out: []f32) void {
        var cached: ?f32 = null;
        for (out) |*x| {
            if (cached) |c| {
                x.* = c;
                cached = null;
                continue;
            }
            const unit_a = gen.uniform01();
            const unit_b = gen.uniform01();
            const r = @sqrt(-2.0 * @log(1.0 - unit_b));
            const theta = two_pi * unit_a;
            x.* = r * @cos(theta);
            cached = r * @sin(theta);
        }
    }

    pub fn nchwRandn(allocator: std.mem.Allocator, gen: *Generator, c: u32, t: u32, h: u32, w: u32) ![]f32 {
        const out = try allocator.alloc(f32, @as(usize, c) * t * h * w);
        randn(gen, out);
        return out;
    }

    pub fn patchifyNchw(
        allocator: std.mem.Allocator,
        nchw: []const f32,
        c: u32,
        t: u32,
        h: u32,
        w: u32,
        patch: [3]i64,
    ) ![]f32 {
        const thwc = try allocator.alloc(f32, nchw.len);
        defer allocator.free(thwc);
        packing.nchwToThwc(thwc, nchw, c, t, h, w);
        return packing.patchify(allocator, thwc, t, h, w, c, patch);
    }

    pub fn drawVideo(
        allocator: std.mem.Allocator,
        gen: *Generator,
        videos: []const packing.ConditionVideo,
        clean_patches: []const f32,
        latent_t: u32,
        latent_h: u32,
        latent_w: u32,
        patch: [3]i64,
        reset_before_target: bool,
    ) ![]f32 {
        const channels: u32 = 24;
        const row_w = @as(usize, channels) * @as(usize, @intCast(patch[0] * patch[1] * patch[2]));
        var cond_len: usize = 0;
        for (videos) |v| {
            cond_len += @as(usize, config.videoTokenCount(v.latent_t, v.latent_h, v.latent_w, patch)) * row_w;
        }
        if (clean_patches.len != cond_len) return error.ConditionPatchSize;

        const target_rows = config.videoTokenCount(latent_t, latent_h, latent_w, patch);
        const out = try allocator.alloc(f32, cond_len + @as(usize, target_rows) * row_w);
        errdefer allocator.free(out);

        var off: usize = 0;
        for (videos) |v| {
            const nchw = try nchwRandn(allocator, gen, channels, v.latent_t, v.latent_h, v.latent_w);
            defer allocator.free(nchw);
            const noise_rows = try patchifyNchw(allocator, nchw, channels, v.latent_t, v.latent_h, v.latent_w, patch);
            defer allocator.free(noise_rows);
            if (off + noise_rows.len > out.len) return error.ConditionPatchSize;
            for (out[off..][0..noise_rows.len], clean_patches[off..][0..noise_rows.len], noise_rows) |*dst, clean, noisy| {
                dst.* = scheduler.Schedule.scaleNoise(config.visual_cond_timestep, clean, noisy);
            }
            off += noise_rows.len;
        }

        if (reset_before_target) gen.reset();

        const nchw = try nchwRandn(allocator, gen, channels, latent_t, latent_h, latent_w);
        defer allocator.free(nchw);
        const target = try patchifyNchw(allocator, nchw, channels, latent_t, latent_h, latent_w, patch);
        defer allocator.free(target);
        if (off + target.len != out.len) return error.TargetPatchSize;
        @memcpy(out[off..], target);
        return out;
    }

    pub fn drawAudio(
        allocator: std.mem.Allocator,
        gen: *Generator,
        clean_patches: []const f32,
        channels: u32,
        audio_t: u32,
    ) ![]f32 {
        const target_n = @as(usize, 2) * audio_t * channels;
        const out = try allocator.alloc(f32, clean_patches.len + target_n);
        errdefer allocator.free(out);
        if (clean_patches.len != 0) @memcpy(out[0..clean_patches.len], clean_patches);
        randn(gen, out[clean_patches.len..]);
        return out;
    }
};

// --- sampling/multistep.zig ---
pub const multistep = struct {
    const std = @import("std");

    const zml = @import("zml");

    /// Second-order multistep on one flow stream. Data-ward velocity:
    /// `x_next = x + (sigma - sigma_next) * (1.5 v - 0.5 v_prev)` after step 0.
    pub fn resMultistep(
        sigmas: []const f32,
        step_index: usize,
        sample: []f32,
        velocity: []const f32,
        prev_velocity: ?[]const f32,
    ) void {
        std.debug.assert(step_index + 1 < sigmas.len);
        std.debug.assert(sample.len == velocity.len);
        const sigma = sigmas[step_index];
        const sigma_next = sigmas[step_index + 1];
        const dt = sigma - sigma_next;
        if (step_index == 0 or prev_velocity == null or sigma_next == 0) {
            for (sample, velocity) |*x, v| x.* += dt * v;
            return;
        }
        const prev = prev_velocity.?;
        std.debug.assert(prev.len == sample.len);
        for (sample, velocity, prev) |*x, v, pv| {
            x.* += dt * (1.5 * v - 0.5 * pv);
        }
    }

    pub const State = struct {
        prev_video: ?[]f32 = null,
        prev_audio: ?[]f32 = null,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) State {
            return .{ .allocator = allocator };
        }

        pub fn deinit(self: *State) void {
            if (self.prev_video) |p| self.allocator.free(p);
            if (self.prev_audio) |p| self.allocator.free(p);
            self.prev_video = null;
            self.prev_audio = null;
        }

        pub fn remember(self: *State, video: []const f32, audio: []const f32) !void {
            if (self.prev_video) |p| self.allocator.free(p);
            if (self.prev_audio) |p| self.allocator.free(p);
            self.prev_video = try self.allocator.dupe(f32, video);
            self.prev_audio = try self.allocator.dupe(f32, audio);
        }
    };

    pub fn dualResMultistep(
        schedules: scheduler.DualSchedule,
        step_index: usize,
        video: []f32,
        audio: []f32,
        video_vel: []const f32,
        audio_vel: []const f32,
        state: *State,
    ) !void {
        resMultistep(schedules.video.sigmas, step_index, video, video_vel, state.prev_video);
        resMultistep(schedules.audio.sigmas, step_index, audio, audio_vel, state.prev_audio);
        try state.remember(video_vel, audio_vel);
    }

    pub const StepModel = struct {
        hold: i64,
    };

    pub const StepInput = struct {
        model: StepModel,
        sample: zml.Tensor,
        velocity: zml.Tensor,
        prev_velocity: zml.Tensor,
        dt: zml.Tensor,
        use_ab2: zml.Tensor,
    };

    pub const StepOutput = struct {
        sample: zml.Tensor,
        prev: zml.Tensor,
    };

    pub fn apply(input: StepInput) StepOutput {
        const vel = input.velocity.convert(.f32);
        const prev = input.prev_velocity.convert(.f32);
        const sample = input.sample.convert(.f32);
        const ab2 = vel.mul(zml.Tensor.scalar(1.5, .f32)).sub(prev.mul(zml.Tensor.scalar(0.5, .f32)));
        const use = input.use_ab2.cmp(.NE, zml.Tensor.scalar(0, input.use_ab2.dtype()));
        const v = zml.Tensor.select(use.broad(vel.shape()), ab2, vel);
        var next = sample.add(v.mul(input.dt.convert(.f32).broad(v.shape())));
        if (input.model.hold > 0) {
            const seq = next.dim(.s);
            const prefix = sample.slice1d(.s, .{ .start = 0, .end = input.model.hold });
            const rest = next.slice1d(.s, .{ .start = input.model.hold, .end = seq });
            next = zml.Tensor.concatenate(&.{ prefix, rest }, .s);
        }
        return .{
            .sample = next.reuseBuffer(input.sample),
            .prev = vel.reuseBuffer(input.prev_velocity),
        };
    }
};

// --- model/encoder.zig ---
pub const encoder = struct {
    const std = @import("std");

    const zml = @import("zml");

    const config_mod = @import("model.zig").config;

    const log = std.log.scoped(.minimax_h3_encoder);

    pub const Config = config_mod.EncoderConfig;

    const RmsNorm = struct {
        weight: zml.Tensor,
        eps: f32,

        pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
            return .{
                .weight = store.createTensor("weight", .{.d}, .replicated),
                .eps = eps,
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
            self.weight.deinit();
        }

        pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
            const x = input.withPartialTags(.{.d});
            const normalized = zml.nn.rmsNorm(x, .d, self.eps);
            return normalized.mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()));
        }
    };

    fn linear(store: zml.io.TensorStore.View, weight_name: []const u8, partitions: anytype) zml.nn.Linear {
        return .fromStore(store, weight_name, null, partitions, .replicated, .d);
    }

    const Mlp = struct {
        up_proj: zml.nn.Linear,
        gate_proj: zml.nn.Linear,
        down_proj: zml.nn.Linear,

        pub fn init(store: zml.io.TensorStore.View) Mlp {
            return .{
                .up_proj = linear(store, "up_proj.weight", .{ .dout = .model }),
                .gate_proj = linear(store, "gate_proj.weight", .{ .dout = .model }),
                .down_proj = linear(store, "down_proj.weight", .{ .d = .model }),
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
            zml.nn.Linear.unloadBuffers(&self.up_proj);
            zml.nn.Linear.unloadBuffers(&self.gate_proj);
            zml.nn.Linear.unloadBuffers(&self.down_proj);
        }

        pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
            const proj = self.up_proj.forward(x);
            var output = self.gate_proj.forward(x);
            output = output.silu().mul(proj).rename(.{ .dout = .d });
            return self.down_proj.forward(output);
        }
    };

    const SelfAttn = struct {
        q_proj: zml.nn.Linear,
        k_proj: zml.nn.Linear,
        v_proj: zml.nn.Linear,
        o_proj: zml.nn.Linear,
        q_norm: RmsNorm,
        k_norm: RmsNorm,
        num_heads: i64,
        num_kv_heads: i64,
        head_dim: i64,
        rope_opts: zml.nn.RopeOpts,
        attn_kind: policy.AttnKind = .vanilla,

        pub fn init(store: zml.io.TensorStore.View, cfg: Config) SelfAttn {
            return .{
                .q_proj = linear(store, "q_proj.weight", .{ .dout = .model }),
                .k_proj = linear(store, "k_proj.weight", .{ .dout = .model }),
                .v_proj = linear(store, "v_proj.weight", .{ .dout = .model }),
                .o_proj = linear(store, "o_proj.weight", .{ .d = .model }),
                .q_norm = .init(store.withPrefix("q_norm"), cfg.rms_norm_eps),
                .k_norm = .init(store.withPrefix("k_norm"), cfg.rms_norm_eps),
                .num_heads = cfg.num_attention_heads,
                .num_kv_heads = cfg.num_key_value_heads,
                .head_dim = cfg.head_dim,
                .rope_opts = .{
                    .layout = .real_im_pass,
                    .scaling = .{ .default = .{ .rope_theta = cfg.rope_theta } },
                },
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(SelfAttn)) void {
            zml.nn.Linear.unloadBuffers(&self.q_proj);
            zml.nn.Linear.unloadBuffers(&self.k_proj);
            zml.nn.Linear.unloadBuffers(&self.v_proj);
            zml.nn.Linear.unloadBuffers(&self.o_proj);
            RmsNorm.unloadBuffers(&self.q_norm);
            RmsNorm.unloadBuffers(&self.k_norm);
        }

        pub fn forward(self: SelfAttn, x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
            const x_qkv = x.withPartitioning(.{ .d = .replicated });
            var q = self.q_proj.forward(x_qkv).splitAxis(-1, .{ .h = self.num_heads, .hd = self.head_dim }).withPartitioning(.{ .h = .model });
            var k = self.k_proj.forward(x_qkv).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = self.head_dim }).withPartitioning(.{ .h = .model });
            var v = self.v_proj.forward(x_qkv).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = self.head_dim }).withPartitioning(.{ .h = .model });

            q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
            k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });
            q = applyRotary(q, cos, sin);
            k = applyRotary(k, cos, sin);

            const q_s = q.rename(.{ .s = .q });
            const k_s = k.rename(.{ .s = .k });
            const v_s = v.rename(.{ .s = .k });
            const attn = switch (self.attn_kind) {
                .cuda_fa2 => zml.attention.flashattn.fa2.dense(q_s, k_s, v_s, .{ .is_causal = true }),
                .vanilla => blk: {
                    const mask = zml.nn.causalAttnMask(.{ .q = q.dim(.s), .k = k.dim(.s) }, q.dtype(), null);
                    break :blk zml.nn.sdpa(q_s, k_s, v_s, .{ .attn_mask = mask });
                },
            }.rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } });
            return self.o_proj.forward(attn).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });
        }
    };

    fn applyRotary(x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const half = @divExact(x.dim(.hd), 2);
        const x1 = x.slice1d(.hd, .{ .start = 0, .end = half });
        const x2 = x.slice1d(.hd, .{ .start = half, .end = x.dim(.hd) });
        const rotated = zml.Tensor.concatenate(&.{ x2.negate(), x1 }, .hd);
        return x.mul(cos.broad(x.shape())).add(rotated.mul(sin.broad(x.shape())));
    }

    pub const TransformerLayer = struct {
        input_layernorm: RmsNorm,
        self_attn: SelfAttn,
        post_attention_layernorm: RmsNorm,
        mlp: Mlp,

        pub const Input = struct {
            layer: TransformerLayer,
            hidden: zml.Tensor,
            cos: zml.Tensor,
            sin: zml.Tensor,
            visual_delta: zml.Tensor,
        };

        pub const Output = struct {
            hidden: zml.Tensor,
        };

        pub fn init(store: zml.io.TensorStore.View, cfg: Config) TransformerLayer {
            return .{
                .input_layernorm = .init(store.withPrefix("input_layernorm"), cfg.rms_norm_eps),
                .self_attn = .init(store.withPrefix("self_attn"), cfg),
                .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm"), cfg.rms_norm_eps),
                .mlp = .init(store.withPrefix("mlp")),
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
            RmsNorm.unloadBuffers(&self.input_layernorm);
            SelfAttn.unloadBuffers(&self.self_attn);
            RmsNorm.unloadBuffers(&self.post_attention_layernorm);
            Mlp.unloadBuffers(&self.mlp);
        }

        pub fn forward(input: Input) Output {
            const self = input.layer;
            const residual = input.hidden.withPartitioning(.{ .d = .replicated });
            const attn = self.self_attn.forward(self.input_layernorm.forward(residual), input.cos, input.sin);
            const x1 = residual.add(attn).withPartitioning(.{ .d = .replicated });
            const mlp = self.mlp.forward(self.post_attention_layernorm.forward(x1)).rename(.{ .dout = .d });
            const hidden = x1.add(mlp).add(input.visual_delta.convert(x1.dtype())).withPartitioning(.{ .d = .replicated });
            return .{ .hidden = hidden.reuseBuffer(input.hidden) };
        }
    };

    pub const EmbedTokens = struct {
        embed_tokens: zml.nn.TokenEmbedding,

        pub const Input = struct {
            embedding: EmbedTokens,
            tokens: zml.Tensor,
        };

        pub const Output = struct {
            hidden: zml.Tensor,
        };

        pub fn unloadBuffers(self: *zml.Bufferized(EmbedTokens)) void {
            zml.nn.TokenEmbedding.unloadBuffers(&self.embed_tokens);
        }

        pub fn forward(input: Input) Output {
            const tokens = input.tokens.withPartialTags(.{.s});
            return .{ .hidden = input.embedding.embed_tokens.forward(tokens)
                .withPartialTags(.{.d})
                .withPartitioning(.{ .d = .replicated }) };
        }
    };

    pub const Model = struct {
        embed_tokens: zml.nn.TokenEmbedding,
        layers: []TransformerLayer,
        cfg: Config,

        pub fn init(allocator: std.mem.Allocator, store_: zml.io.TensorStore.View, cfg: Config) !Model {
            const store = rootView(store_);
            const used: usize = @intCast(cfg.used_hidden_layers);
            const layers = try allocator.alloc(TransformerLayer, used);
            errdefer allocator.free(layers);
            for (layers, 0..) |*layer, i| {
                layer.* = .init(store.withPrefix("layers").withLayer(i), cfg);
            }
            return .{
                .embed_tokens = embedTokens(store),
                .layers = layers,
                .cfg = cfg,
            };
        }

        pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
            allocator.free(self.layers);
        }

        pub fn applyBackend(self: *Model, kind: policy.AttnKind) void {
            for (self.layers) |*layer| layer.self_attn.attn_kind = kind;
        }
    };

    fn embedTokens(store: zml.io.TensorStore.View) zml.nn.TokenEmbedding {
        return .fromStore(store, "embed_tokens.weight", .{ .voc = .replicated, .d = .model });
    }

    const embed_roots = [_]struct { key: []const u8, prefix: []const u8 }{
        .{ .key = "model.language_model.embed_tokens.weight", .prefix = "model.language_model" },
        .{ .key = "model.embed_tokens.weight", .prefix = "model" },
    };

    fn rootView(store: zml.io.TensorStore.View) zml.io.TensorStore.View {
        for (embed_roots) |root| {
            if (store.hasKey(root.key)) return store.withPrefix(root.prefix);
        }
        return store;
    }

    pub const LoadedModel = struct {
        inner: Model,
        cfg: Config,

        pub fn init(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, store: zml.io.TensorStore.View) !LoadedModel {
            const cfg = try config_mod.loadEncoderConfig(allocator, io, repo);
            log.info("encoder: {d} layers hidden={d} heads={d}", .{
                cfg.used_hidden_layers,
                cfg.hidden_size,
                cfg.num_attention_heads,
            });
            return .{
                .inner = try .init(allocator, store, cfg),
                .cfg = cfg,
            };
        }

        pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
            self.inner.deinit(allocator);
        }

        pub fn loadEmbed(
            self: *const LoadedModel,
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            store: *zml.io.TensorStore,
            shardings: []const zml.Sharding,
            progress: *std.Progress.Node,
        ) !zml.Bufferized(EmbedTokens) {
            const part = EmbedTokens{ .embed_tokens = self.inner.embed_tokens };
            return weights.load(allocator, io, platform, store, shardings, EmbedTokens, &part, progress, null);
        }

        pub fn loadLayer(
            self: *const LoadedModel,
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            store: *zml.io.TensorStore,
            shardings: []const zml.Sharding,
            index: usize,
            progress: *std.Progress.Node,
            loader: ?*zml.io.Loader,
        ) !zml.Bufferized(TransformerLayer) {
            return weights.load(allocator, io, platform, store, shardings, TransformerLayer, &self.inner.layers[index], progress, loader);
        }
    };
};

// --- model/vision.zig ---
pub const vision = struct {
    const std = @import("std");

    const zml = @import("zml");

    const config_mod = @import("model.zig").config;

    const log = std.log.scoped(.minimax_h3_vision);

    pub const VISION_START: u32 = 151652;
    pub const VISION_END: u32 = 151653;
    pub const IMAGE_PAD: u32 = 151655;
    pub const VIDEO_PAD: u32 = 151656;

    pub const Config = struct {
        depth: i64 = 27,
        hidden_size: i64 = 1152,
        intermediate_size: i64 = 4304,
        num_heads: i64 = 16,
        patch_size: i64 = 16,
        temporal_patch_size: i64 = 2,
        spatial_merge_size: i64 = 2,
        out_hidden_size: i64 = 5120,
        num_position_embeddings: i64 = 2304,
        deepstack_visual_indexes: [3]i64 = .{ 8, 16, 24 },
        rms_norm_eps: f32 = 1e-6,

        pub fn headDim(self: Config) i64 {
            return @divExact(self.hidden_size, self.num_heads);
        }

        pub fn patchIn(self: Config) i64 {
            return 3 * self.temporal_patch_size * self.patch_size * self.patch_size;
        }

        pub fn mergeUnit(self: Config) i64 {
            return self.spatial_merge_size * self.spatial_merge_size;
        }
    };

    const FileConfig = struct {
        vision_config: ?struct {
            depth: ?i64 = null,
            hidden_size: ?i64 = null,
            intermediate_size: ?i64 = null,
            num_heads: ?i64 = null,
            patch_size: ?i64 = null,
            temporal_patch_size: ?i64 = null,
            spatial_merge_size: ?i64 = null,
            out_hidden_size: ?i64 = null,
            num_position_embeddings: ?i64 = null,
            deepstack_visual_indexes: ?[]const i64 = null,
        } = null,

        fn resolve(self: FileConfig, text_hidden: i64) Config {
            var out = Config{};
            out.out_hidden_size = text_hidden;
            if (self.vision_config) |v| {
                if (v.depth) |d| out.depth = d;
                if (v.hidden_size) |d| out.hidden_size = d;
                if (v.intermediate_size) |d| out.intermediate_size = d;
                if (v.num_heads) |d| out.num_heads = d;
                if (v.patch_size) |d| out.patch_size = d;
                if (v.temporal_patch_size) |d| out.temporal_patch_size = d;
                if (v.spatial_merge_size) |d| out.spatial_merge_size = d;
                if (v.out_hidden_size) |d| out.out_hidden_size = d;
                if (v.num_position_embeddings) |d| out.num_position_embeddings = d;
                if (v.deepstack_visual_indexes) |idx| {
                    for (0..@min(idx.len, out.deepstack_visual_indexes.len)) |i| out.deepstack_visual_indexes[i] = idx[i];
                }
            }
            return out;
        }
    };

    fn visionView(store: zml.io.TensorStore.View) zml.io.TensorStore.View {
        if (store.hasKey("model.visual.patch_embed.proj.weight")) return store.withPrefix("model.visual");
        return store;
    }

    pub fn ready(store: zml.io.TensorStore.View) bool {
        return store.hasKey("model.visual.patch_embed.proj.weight");
    }

    fn weightRank(store: zml.io.TensorStore.View, weight_name: []const u8) u8 {
        var buffer: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ store.prefix() orelse "", weight_name }) catch return 2;
        return if (store.store.getShape(key)) |s| s.rank() else 2;
    }

    fn linear(store: zml.io.TensorStore.View, weight_name: []const u8, bias_name: ?[]const u8) zml.nn.Linear {
        if (weightRank(store, weight_name) != 5)
            return .fromStore(store, weight_name, bias_name, .replicated, .replicated, .d);
        var layer: zml.nn.Linear = .init(
            store.createTensor(weight_name, .{ .dout, .d, .kt, .kh, .kw }, .replicated),
            if (bias_name) |n| store.maybeCreateTensor(n, .{.dout}, .replicated) else null,
            .d,
        );
        layer.attachQuant(store, weight_name);
        return layer;
    }

    fn asLinear(lin: zml.nn.Linear, x: zml.Tensor) zml.Tensor {
        var out = lin;
        if (out.weight.rank() == 5) {
            out.weight = out.weight.merge(.{ .d = .{ .d, .kt, .kh, .kw } });
        } else {
            while (out.weight.rank() > 2) out.weight = out.weight.squeeze(-1);
        }
        out.weight = out.weight.withTags(.{ .dout, .d });
        return out.forward(x.convert(out.weight.dtype()));
    }

    const LayerNorm = struct {
        weight: zml.Tensor,
        bias: ?zml.Tensor,

        pub fn init(store: zml.io.TensorStore.View) LayerNorm {
            return .{
                .weight = store.createTensor("weight", .{.d}, .replicated),
                .bias = store.maybeCreateTensor("bias", .{.d}, .replicated),
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(LayerNorm)) void {
            self.weight.deinit();
            if (self.bias) |*b| b.deinit();
        }

        pub fn forward(self: LayerNorm, x: zml.Tensor) zml.Tensor {
            const weight = self.weight.convert(.f32);
            const bias = if (self.bias) |b| b.convert(.f32) else null;
            return (zml.nn.LayerNorm{ .weight = weight, .bias = bias, .eps = 1e-6 }).forward(x.convert(.f32)).convert(x.dtype());
        }
    };

    fn applyRotary(x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const half = @divExact(x.dim(-1), 2);
        const x1 = x.slice1d(-1, .{ .start = 0, .end = half });
        const x2 = x.slice1d(-1, .{ .start = half, .end = x.dim(-1) });
        const rotated = zml.Tensor.concatenate(&.{ x2.negate(), x1 }, -1);
        const c = cos.broad(x.shape());
        const s = sin.broad(x.shape());
        return x.mul(c).add(rotated.mul(s));
    }

    pub const VisionBlock = struct {
        norm1: LayerNorm,
        qkv: zml.nn.Linear,
        proj: zml.nn.Linear,
        norm2: LayerNorm,
        fc1: zml.nn.Linear,
        fc2: zml.nn.Linear,
        num_heads: i64,
        head_dim: i64,

        pub const Input = struct {
            layer: VisionBlock,
            hidden: zml.Tensor,
            cos: zml.Tensor,
            sin: zml.Tensor,
        };
        pub const Output = struct { hidden: zml.Tensor };

        pub fn init(store: zml.io.TensorStore.View, cfg: Config) VisionBlock {
            const attn = store.withPrefix("attn");
            const mlp = store.withPrefix("mlp");
            return .{
                .norm1 = .init(store.withPrefix("norm1")),
                .qkv = linear(attn, "qkv.weight", "qkv.bias"),
                .proj = linear(attn, "proj.weight", "proj.bias"),
                .norm2 = .init(store.withPrefix("norm2")),
                .fc1 = linear(mlp, "linear_fc1.weight", "linear_fc1.bias"),
                .fc2 = linear(mlp, "linear_fc2.weight", "linear_fc2.bias"),
                .num_heads = cfg.num_heads,
                .head_dim = cfg.headDim(),
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(VisionBlock)) void {
            LayerNorm.unloadBuffers(&self.norm1);
            zml.nn.Linear.unloadBuffers(&self.qkv);
            zml.nn.Linear.unloadBuffers(&self.proj);
            LayerNorm.unloadBuffers(&self.norm2);
            zml.nn.Linear.unloadBuffers(&self.fc1);
            zml.nn.Linear.unloadBuffers(&self.fc2);
        }

        pub fn forward(input: Input) Output {
            const self = input.layer;
            const residual = input.hidden.withPartialTags(.{ .b, .s, .d });
            var qkv = asLinear(self.qkv, self.norm1.forward(residual));
            const parts = qkv.chunkExact(.dout, 3);
            var q = parts[0].rename(.{ .dout = .d }).splitAxis(.d, .{ .h = self.num_heads, .hd = self.head_dim });
            var k = parts[1].rename(.{ .dout = .d }).splitAxis(.d, .{ .h = self.num_heads, .hd = self.head_dim });
            const v = parts[2].rename(.{ .dout = .d }).splitAxis(.d, .{ .h = self.num_heads, .hd = self.head_dim });
            q = applyRotary(q, input.cos, input.sin);
            k = applyRotary(k, input.cos, input.sin);
            const attn = zml.nn.sdpa(q.rename(.{ .s = .q }), k.rename(.{ .s = .k }), v.rename(.{ .s = .k }), .{}).rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } });
            const x1 = residual.add(asLinear(self.proj, attn).rename(.{ .dout = .d }));
            const ff = asLinear(self.fc2, asLinear(self.fc1, self.norm2.forward(x1)).gelu().rename(.{ .dout = .d })).rename(.{ .dout = .d });
            return .{ .hidden = x1.add(ff).reuseBuffer(input.hidden) };
        }
    };

    pub const EmbedModel = struct {
        proj: zml.nn.Linear,

        pub fn unloadBuffers(self: *zml.Bufferized(EmbedModel)) void {
            zml.nn.Linear.unloadBuffers(&self.proj);
        }
    };

    pub const Merger = struct {
        norm: LayerNorm,
        fc1: zml.nn.Linear,
        fc2: zml.nn.Linear,
        merge: i64,
        postshuffle: bool,

        pub const Input = struct {
            model: Merger,
            hidden: zml.Tensor,
        };
        pub const Output = struct { tokens: zml.Tensor };

        pub fn init(store: zml.io.TensorStore.View, merge: i64, postshuffle: bool) Merger {
            return .{
                .norm = .init(store.withPrefix("norm")),
                .fc1 = linear(store, "linear_fc1.weight", "linear_fc1.bias"),
                .fc2 = linear(store, "linear_fc2.weight", "linear_fc2.bias"),
                .merge = merge,
                .postshuffle = postshuffle,
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(Merger)) void {
            LayerNorm.unloadBuffers(&self.norm);
            zml.nn.Linear.unloadBuffers(&self.fc1);
            zml.nn.Linear.unloadBuffers(&self.fc2);
        }

        pub fn forward(input: Input) Output {
            const self = input.model;
            var x = input.hidden.withPartialTags(.{ .b, .s, .d });
            const grouped = @divExact(x.dim(.s), self.merge);
            if (self.postshuffle) {
                x = x.splitAxis(.s, .{ .s = grouped, .m = self.merge }).merge(.{ .d = .{ .m, .d } });
                x = self.norm.forward(x);
            } else {
                x = self.norm.forward(x);
                x = x.splitAxis(.s, .{ .s = grouped, .m = self.merge }).merge(.{ .d = .{ .m, .d } });
            }
            x = asLinear(self.fc2, asLinear(self.fc1, x).gelu().rename(.{ .dout = .d })).rename(.{ .dout = .d });
            return .{ .tokens = x };
        }
    };

    pub const Model = struct {
        embed: EmbedModel,
        blocks: []VisionBlock,
        merger: Merger,
        deepstack: [3]Merger,
        pos_embed: zml.Tensor,
        cfg: Config,

        pub fn init(allocator: std.mem.Allocator, store_: zml.io.TensorStore.View, cfg: Config) !Model {
            const store = visionView(store_);
            const blocks = try allocator.alloc(VisionBlock, @intCast(cfg.depth));
            errdefer allocator.free(blocks);
            const block_store = store.withPrefix("blocks");
            for (blocks, 0..) |*block, i| block.* = .init(block_store.withLayer(i), cfg);
            var deepstack: [3]Merger = undefined;
            const ds = store.withPrefix("deepstack_merger_list");
            for (&deepstack, 0..) |*m, i| m.* = .init(ds.withLayer(i), cfg.mergeUnit(), true);
            return .{
                .embed = .{ .proj = linear(store.withPrefix("patch_embed.proj"), "weight", "bias") },
                .blocks = blocks,
                .merger = .init(store.withPrefix("merger"), cfg.mergeUnit(), false),
                .deepstack = deepstack,
                .pos_embed = store.createTensor("pos_embed.weight", .{ .s, .d }, .replicated),
                .cfg = cfg,
            };
        }

        pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
            allocator.free(self.blocks);
        }
    };

    pub const EmbedInput = struct {
        model: EmbedModel,
        patches: zml.Tensor,
        pos: zml.Tensor,
    };
    pub const EmbedOutput = struct { hidden: zml.Tensor };

    pub fn embed(input: EmbedInput) EmbedOutput {
        const tokens = asLinear(input.model.proj, input.patches.withPartialTags(.{ .b, .s, .d })).rename(.{ .dout = .d });
        return .{ .hidden = tokens.add(input.pos.convert(tokens.dtype())) };
    }

    pub const LoadedModel = struct {
        inner: Model,
        cfg: Config,

        pub fn init(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, store: zml.io.TensorStore.View, text_hidden: i64) !LoadedModel {
            const cfg = try configFromRepo(allocator, io, repo, text_hidden);
            return .{
                .inner = try .init(allocator, store, cfg),
                .cfg = cfg,
            };
        }

        pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
            self.inner.deinit(allocator);
        }

        pub fn loadEmbed(self: *const LoadedModel, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore, shardings: []const zml.Sharding, progress: *std.Progress.Node) !zml.Bufferized(EmbedModel) {
            return weights.load(allocator, io, platform, store, shardings, EmbedModel, &self.inner.embed, progress, null);
        }

        pub fn loadBlock(self: *const LoadedModel, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore, shardings: []const zml.Sharding, index: usize, progress: *std.Progress.Node, loader: ?*zml.io.Loader) !zml.Bufferized(VisionBlock) {
            return weights.load(allocator, io, platform, store, shardings, VisionBlock, &self.inner.blocks[index], progress, loader);
        }

        pub fn loadMerger(self: *const LoadedModel, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore, shardings: []const zml.Sharding, progress: *std.Progress.Node) !zml.Bufferized(Merger) {
            return weights.load(allocator, io, platform, store, shardings, Merger, &self.inner.merger, progress, null);
        }

        pub fn loadDeepstack(self: *const LoadedModel, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore, shardings: []const zml.Sharding, index: usize, progress: *std.Progress.Node) !zml.Bufferized(Merger) {
            return weights.load(allocator, io, platform, store, shardings, Merger, &self.inner.deepstack[index], progress, null);
        }

        pub fn loadPosEmbed(self: *const LoadedModel, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore, shardings: []const zml.Sharding, progress: *std.Progress.Node) !zml.Buffer {
            var part = self.inner.pos_embed;
            return weights.load(allocator, io, platform, store, shardings, zml.Tensor, &part, progress, null);
        }
    };

    pub const WeightCache = struct {
        embed: zml.Bufferized(EmbedModel),
        pos: zml.Buffer,
        blocks: []zml.Bufferized(VisionBlock),
        merger: zml.Bufferized(Merger),
        deepstack: [3]zml.Bufferized(Merger),

        pub fn deinit(self: *WeightCache, allocator: std.mem.Allocator) void {
            EmbedModel.unloadBuffers(&self.embed);
            self.pos.deinit();
            for (self.blocks) |*block| VisionBlock.unloadBuffers(block);
            allocator.free(self.blocks);
            Merger.unloadBuffers(&self.merger);
            for (&self.deepstack) |*m| Merger.unloadBuffers(m);
        }

        pub fn load(
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            loaded: *const LoadedModel,
            store: *zml.io.TensorStore,
            shardings: []const zml.Sharding,
            progress: *std.Progress.Node,
        ) !WeightCache {
            var embed_bufs = try loaded.loadEmbed(allocator, io, platform, store, shardings, progress);
            errdefer EmbedModel.unloadBuffers(&embed_bufs);
            var pos = try loaded.loadPosEmbed(allocator, io, platform, store, shardings, progress);
            errdefer pos.deinit();
            const blocks = try allocator.alloc(zml.Bufferized(VisionBlock), loaded.inner.blocks.len);
            errdefer allocator.free(blocks);
            var filled: usize = 0;
            errdefer {
                for (blocks[0..filled]) |*block| VisionBlock.unloadBuffers(block);
            }
            for (blocks, 0..) |*block, i| {
                block.* = try loaded.loadBlock(allocator, io, platform, store, shardings, i, progress, null);
                filled += 1;
            }
            var merger = try loaded.loadMerger(allocator, io, platform, store, shardings, progress);
            errdefer Merger.unloadBuffers(&merger);
            var deepstack: [3]zml.Bufferized(Merger) = undefined;
            var ds_filled: usize = 0;
            errdefer {
                for (deepstack[0..ds_filled]) |*m| Merger.unloadBuffers(m);
            }
            for (&deepstack, 0..) |*m, i| {
                m.* = try loaded.loadDeepstack(allocator, io, platform, store, shardings, i, progress);
                ds_filled += 1;
            }
            return .{
                .embed = embed_bufs,
                .pos = pos,
                .blocks = blocks,
                .merger = merger,
                .deepstack = deepstack,
            };
        }
    };

    pub const Grid = struct { h: u32, w: u32 };

    pub fn chooseGrid(cfg: Config, src_h: u32, src_w: u32, video: bool) struct { h: u32, w: u32 } {
        const factor: u32 = @intCast(cfg.patch_size * cfg.spatial_merge_size);
        var target_h = @max(factor, (src_h + factor / 2) / factor * factor);
        var target_w = @max(factor, (src_w + factor / 2) / factor * factor);
        const min_pixels: f32 = if (video) 4096.0 else 65536.0;
        const max_pixels: f32 = if (video) 25165824.0 else 16777216.0;
        const area = @as(f32, @floatFromInt(target_h)) * @as(f32, @floatFromInt(target_w));
        if (area > max_pixels) {
            const scale = @sqrt((@as(f32, @floatFromInt(src_h * src_w))) / max_pixels);
            target_h = @max(factor, @as(u32, @intFromFloat(@floor(@as(f32, @floatFromInt(src_h)) / scale / @as(f32, @floatFromInt(factor))))) * factor);
            target_w = @max(factor, @as(u32, @intFromFloat(@floor(@as(f32, @floatFromInt(src_w)) / scale / @as(f32, @floatFromInt(factor))))) * factor);
        } else if (area < min_pixels) {
            const scale = @sqrt(min_pixels / @as(f32, @floatFromInt(src_h * src_w)));
            target_h = @as(u32, @intFromFloat(@ceil(@as(f32, @floatFromInt(src_h)) * scale / @as(f32, @floatFromInt(factor))))) * factor;
            target_w = @as(u32, @intFromFloat(@ceil(@as(f32, @floatFromInt(src_w)) * scale / @as(f32, @floatFromInt(factor))))) * factor;
        }
        return .{ .h = target_h, .w = target_w };
    }

    pub fn patchifyRgb(allocator: std.mem.Allocator, rgb: []const u8, src_h: u32, src_w: u32, cfg: Config) !struct { patches: []f32, grid: Grid, seq: u32 } {
        const media = @import("generate.zig").media;
        const size = chooseGrid(cfg, src_h, src_w, false);
        const resized = try media.resizeRgb(allocator, rgb, src_w, src_h, size.w, size.h);
        defer allocator.free(resized);
        const patch: u32 = @intCast(cfg.patch_size);
        const merge: u32 = @intCast(cfg.spatial_merge_size);
        const gh = size.h / patch;
        const gw = size.w / patch;
        const seq = gh * gw;
        const width: u32 = @intCast(cfg.patchIn());
        const out = try allocator.alloc(f32, seq * width);
        var row: usize = 0;
        var ih: u32 = 0;
        while (ih < gh) : (ih += merge) {
            var iw: u32 = 0;
            while (iw < gw) : (iw += merge) {
                var di: u32 = 0;
                while (di < merge) : (di += 1) {
                    var dj: u32 = 0;
                    while (dj < merge) : (dj += 1) {
                        var dst: usize = 0;
                        var c: u32 = 0;
                        while (c < 3) : (c += 1) {
                            var t: u32 = 0;
                            while (t < 2) : (t += 1) {
                                var ph: u32 = 0;
                                while (ph < patch) : (ph += 1) {
                                    var pw: u32 = 0;
                                    while (pw < patch) : (pw += 1) {
                                        const y = (ih + di) * patch + ph;
                                        const x = (iw + dj) * patch + pw;
                                        const v = @as(f32, @floatFromInt(resized[(y * size.w + x) * 3 + c])) / 255.0;
                                        out[row * width + dst] = v * 2.0 - 1.0;
                                        dst += 1;
                                    }
                                }
                            }
                        }
                        row += 1;
                    }
                }
            }
        }
        return .{ .patches = out, .grid = .{ .h = gh, .w = gw }, .seq = seq };
    }

    pub fn patchifyVideo(
        allocator: std.mem.Allocator,
        rgb: []const u8,
        frames: u32,
        src_h: u32,
        src_w: u32,
        cfg: Config,
    ) !struct { patches: []f32, grid: Grid, seq: u32, temporal: u32 } {
        const media = @import("generate.zig").media;
        const size = chooseGrid(cfg, src_h, src_w, true);
        const even = frames + (frames % 2);
        const temporal = even / 2;
        const plane = @as(usize, src_w) * src_h * 3;
        const resized_plane = @as(usize, size.w) * size.h * 3;
        const stacked = try allocator.alloc(u8, even * resized_plane);
        defer allocator.free(stacked);
        var f: u32 = 0;
        while (f < even) : (f += 1) {
            const src_f = if (f < frames) f else frames - 1;
            const frame = try media.resizeRgb(allocator, rgb[src_f * plane ..][0..plane], src_w, src_h, size.w, size.h);
            defer allocator.free(frame);
            @memcpy(stacked[f * resized_plane ..][0..resized_plane], frame);
        }

        const patch: u32 = @intCast(cfg.patch_size);
        const merge: u32 = @intCast(cfg.spatial_merge_size);
        const gh = size.h / patch;
        const gw = size.w / patch;
        const seq = temporal * gh * gw;
        const width: u32 = @intCast(cfg.patchIn());
        const out = try allocator.alloc(f32, seq * width);
        var row: usize = 0;
        var tf: u32 = 0;
        while (tf < temporal) : (tf += 1) {
            var ih: u32 = 0;
            while (ih < gh) : (ih += merge) {
                var iw: u32 = 0;
                while (iw < gw) : (iw += merge) {
                    var di: u32 = 0;
                    while (di < merge) : (di += 1) {
                        var dj: u32 = 0;
                        while (dj < merge) : (dj += 1) {
                            var dst: u32 = 0;
                            var c: u32 = 0;
                            while (c < 3) : (c += 1) {
                                var t: u32 = 0;
                                while (t < 2) : (t += 1) {
                                    var ph: u32 = 0;
                                    while (ph < patch) : (ph += 1) {
                                        var pw: u32 = 0;
                                        while (pw < patch) : (pw += 1) {
                                            const y = (ih + di) * patch + ph;
                                            const x = (iw + dj) * patch + pw;
                                            const pix = (((tf * 2 + t) * size.h + y) * size.w + x) * 3 + c;
                                            const v = @as(f32, @floatFromInt(stacked[pix])) / 255.0;
                                            out[row * width + dst] = v * 2.0 - 1.0;
                                            dst += 1;
                                        }
                                    }
                                }
                            }
                            row += 1;
                        }
                    }
                }
            }
        }
        return .{ .patches = out, .grid = .{ .h = gh, .w = gw }, .seq = seq, .temporal = temporal };
    }

    pub fn interpolatePos(allocator: std.mem.Allocator, table: []const f32, table_side: u32, hidden: u32, gh: u32, gw: u32) ![]f32 {
        const out = try allocator.alloc(f32, @as(usize, gh) * gw * hidden);
        const merge: u32 = 2;
        var row: usize = 0;
        var ih: u32 = 0;
        while (ih < gh) : (ih += merge) {
            var iw: u32 = 0;
            while (iw < gw) : (iw += merge) {
                var di: u32 = 0;
                while (di < merge) : (di += 1) {
                    var dj: u32 = 0;
                    while (dj < merge) : (dj += 1) {
                        const yden = @max(gh, 2) - 1;
                        const xden = @max(gw, 2) - 1;
                        const y = @as(f32, @floatFromInt(ih + di)) * @as(f32, @floatFromInt(table_side - 1)) / @as(f32, @floatFromInt(yden));
                        const x = @as(f32, @floatFromInt(iw + dj)) * @as(f32, @floatFromInt(table_side - 1)) / @as(f32, @floatFromInt(xden));
                        const y0: u32 = @intFromFloat(@floor(y));
                        const x0: u32 = @intFromFloat(@floor(x));
                        const y1 = @min(table_side - 1, y0 + 1);
                        const x1 = @min(table_side - 1, x0 + 1);
                        const fy = y - @as(f32, @floatFromInt(y0));
                        const fx = x - @as(f32, @floatFromInt(x0));
                        var d: u32 = 0;
                        while (d < hidden) : (d += 1) {
                            const a = table[(y0 * table_side + x0) * hidden + d];
                            const b = table[(y0 * table_side + x1) * hidden + d];
                            const c = table[(y1 * table_side + x0) * hidden + d];
                            const e = table[(y1 * table_side + x1) * hidden + d];
                            out[row * hidden + d] = a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx + c * fy * (1 - fx) + e * fy * fx;
                        }
                        row += 1;
                    }
                }
            }
        }
        return out;
    }

    pub fn visionRope(allocator: std.mem.Allocator, gh: u32, gw: u32, head_dim: u32) !struct { cos: []f32, sin: []f32 } {
        const seq = gh * gw;
        const half = head_dim / 2;
        const cos = try allocator.alloc(f32, seq * head_dim);
        errdefer allocator.free(cos);
        const sin = try allocator.alloc(f32, seq * head_dim);
        const merge: u32 = 2;
        var row: usize = 0;
        var ih: u32 = 0;
        while (ih < gh) : (ih += merge) {
            var iw: u32 = 0;
            while (iw < gw) : (iw += merge) {
                var di: u32 = 0;
                while (di < merge) : (di += 1) {
                    var dj: u32 = 0;
                    while (dj < merge) : (dj += 1) {
                        const hpos: f32 = @floatFromInt(ih + di);
                        const wpos: f32 = @floatFromInt(iw + dj);
                        var f: u32 = 0;
                        while (f < half) : (f += 1) {
                            const freq = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(f)) / @as(f32, @floatFromInt(half)));
                            const ang = if (f < half / 2) hpos * freq else wpos * freq;
                            const c = @cos(ang);
                            const s = @sin(ang);
                            cos[row * head_dim + f] = c;
                            cos[row * head_dim + half + f] = c;
                            sin[row * head_dim + f] = s;
                            sin[row * head_dim + half + f] = s;
                        }
                        row += 1;
                    }
                }
            }
        }
        return .{ .cos = cos, .sin = sin };
    }

    pub fn hostInterleavedMrope(
        pos: []const f32,
        seq: u32,
        head_dim: u32,
        theta: f32,
        section: [3]i64,
        cos: []f32,
        sin: []f32,
    ) void {
        const half = head_dim / 2;
        std.debug.assert(pos.len >= seq * 3);
        std.debug.assert(cos.len >= seq * head_dim);
        var i: u32 = 0;
        while (i < seq) : (i += 1) {
            const pt = pos[i * 3 + 0];
            const ph = pos[i * 3 + 1];
            const pw = pos[i * 3 + 2];
            var f: u32 = 0;
            while (f < half) : (f += 1) {
                var p = pt;
                const h_end = @as(u32, @intCast(section[1] * 3));
                const w_end = @as(u32, @intCast(section[2] * 3));
                if (f < h_end and f % 3 == 1) p = ph;
                if (f < w_end and f % 3 == 2) p = pw;
                const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(f)) / @as(f32, @floatFromInt(half)));
                const ang = p * freq;
                const c = @cos(ang);
                const s = @sin(ang);
                cos[i * head_dim + f] = c;
                cos[i * head_dim + half + f] = c;
                sin[i * head_dim + f] = s;
                sin[i * head_dim + half + f] = s;
            }
        }
    }

    pub fn fillArangePositions(out: []f32, seq: u32) void {
        var i: u32 = 0;
        while (i < seq) : (i += 1) {
            const v: f32 = @floatFromInt(i);
            out[i * 3 + 0] = v;
            out[i * 3 + 1] = v;
            out[i * 3 + 2] = v;
        }
    }

    pub const EncodedVisual = struct {
        merged: []f32,
        deepstack: [3][]f32,
        tokens: u32,
        grid: Grid,
        temporal: u32 = 1,

        pub fn deinit(self: EncodedVisual, allocator: std.mem.Allocator) void {
            allocator.free(self.merged);
            for (self.deepstack) |d| allocator.free(d);
        }
    };

    fn runPatches(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        compiled: *const @import("generate.zig").pipeline.VisionCompiled,
        loaded: *const LoadedModel,
        cache: *const WeightCache,
        patches: []const f32,
        grid: Grid,
        seq: u32,
        temporal: u32,
    ) !EncodedVisual {
        if (seq != compiled.seq) return error.VisionSeqMismatch;

        const n_blocks = cache.blocks.len;
        const vision_start: std.Io.Timestamp = .now(io, .awake);
        log.info("vision: start seq={d} grid={d}x{d} blocks={d} temporal={d}", .{
            seq,
            grid.h,
            grid.w,
            n_blocks,
            temporal,
        });

        const table_host = try bufferToF32(allocator, io, cache.pos);
        defer allocator.free(table_host);
        const side: u32 = @intFromFloat(@sqrt(@as(f32, @floatFromInt(loaded.cfg.num_position_embeddings))));
        const spatial_pos = try interpolatePos(allocator, table_host, side, @intCast(loaded.cfg.hidden_size), grid.h, grid.w);
        defer allocator.free(spatial_pos);
        const spatial_rope = try visionRope(allocator, grid.h, grid.w, @intCast(loaded.cfg.headDim()));
        defer allocator.free(spatial_rope.cos);
        defer allocator.free(spatial_rope.sin);
        const pos = try tileTemporal(allocator, spatial_pos, temporal, @intCast(loaded.cfg.hidden_size));
        defer allocator.free(pos);
        const rope_cos = try tileTemporal(allocator, spatial_rope.cos, temporal, @intCast(loaded.cfg.headDim()));
        defer allocator.free(rope_cos);
        const rope_sin = try tileTemporal(allocator, spatial_rope.sin, temporal, @intCast(loaded.cfg.headDim()));
        defer allocator.free(rope_sin);

        var embed_runner = try zml.FnExe(embed).Runner(.{.model}).init(&compiled.embed, allocator, .{ .model = cache.embed });
        defer embed_runner.deinit(allocator);
        var patch_buf = try zml.Buffer.fromBytes(io, platform, .init(.{ .b = 1, .s = seq, .d = loaded.cfg.patchIn() }, .f32), .replicated, std.mem.sliceAsBytes(patches));
        defer patch_buf.deinit();
        var pos_buf = try zml.Buffer.fromBytes(io, platform, .init(.{ .b = 1, .s = seq, .d = loaded.cfg.hidden_size }, .f32), .replicated, std.mem.sliceAsBytes(pos));
        defer pos_buf.deinit();
        var hidden: zml.Buffer = undefined;
        embed_runner.run(io, .{
            .inputs = .{ .patches = patch_buf, .pos = pos_buf },
            .outputs = .{ .hidden = &hidden },
            .opts = .{ .wait = true },
        });
        defer hidden.deinit();

        var cos_buf = try bufferFromF32(allocator, io, platform, .init(.{ .s = seq, .hd = loaded.cfg.headDim() }, hidden.shape().dtype()), rope_cos);
        defer cos_buf.deinit();
        var sin_buf = try bufferFromF32(allocator, io, platform, .init(.{ .s = seq, .hd = loaded.cfg.headDim() }, hidden.shape().dtype()), rope_sin);
        defer sin_buf.deinit();

        var deepstack: [3][]f32 = .{ &.{}, &.{}, &.{} };
        errdefer {
            for (deepstack) |d| if (d.len != 0) allocator.free(d);
        }
        var ds_i: usize = 0;
        const BlockRunner = zml.FnExe(VisionBlock.forward).Runner(.{.layer});
        var block_runner: ?BlockRunner = null;
        defer if (block_runner) |*r| r.deinit(allocator);
        var block_i: usize = 0;
        while (block_i < n_blocks) : (block_i += 1) {
            if (block_runner) |*r| {
                weights.rebake(r, .{ .layer = cache.blocks[block_i] });
            } else {
                block_runner = try BlockRunner.init(&compiled.block, allocator, .{ .layer = cache.blocks[block_i] });
            }
            var next: zml.Buffer = undefined;
            block_runner.?.run(io, .{
                .inputs = .{ .hidden = hidden, .cos = cos_buf, .sin = sin_buf },
                .outputs = .{ .hidden = &next },
                .opts = .{ .wait = true },
            });
            hidden.deinit();
            hidden = next;
            if (ds_i < 3 and @as(i64, @intCast(block_i)) == loaded.cfg.deepstack_visual_indexes[ds_i]) {
                var ds_run = try zml.FnExe(Merger.forward).Runner(.{.model}).init(&compiled.deepstack, allocator, .{ .model = cache.deepstack[ds_i] });
                defer ds_run.deinit(allocator);
                var tokens: zml.Buffer = undefined;
                ds_run.run(io, .{ .inputs = .{ .hidden = hidden }, .outputs = .{ .tokens = &tokens }, .opts = .{ .wait = true } });
                defer tokens.deinit();
                const host = try bufferToF32(allocator, io, tokens);
                deepstack[ds_i] = host;
                ds_i += 1;
            }
        }
        var merge_run = try zml.FnExe(Merger.forward).Runner(.{.model}).init(&compiled.merger, allocator, .{ .model = cache.merger });
        defer merge_run.deinit(allocator);
        var merged_buf: zml.Buffer = undefined;
        merge_run.run(io, .{ .inputs = .{ .hidden = hidden }, .outputs = .{ .tokens = &merged_buf }, .opts = .{ .wait = true } });
        defer merged_buf.deinit();
        const merged = try bufferToF32(allocator, io, merged_buf);
        log.info("vision: ok merged={d} [{f}]", .{ merged.len, vision_start.untilNow(io, .awake) });
        return .{
            .merged = merged,
            .deepstack = deepstack,
            .tokens = compiled.merged,
            .grid = grid,
            .temporal = temporal,
        };
    }

    fn bufferFromF32(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        shape: zml.Shape,
        values: []const f32,
    ) !zml.Buffer {
        switch (shape.dtype()) {
            .f32 => return zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(values)),
            .bf16 => {
                const tmp = try allocator.alloc(zml.floats.BFloat16, values.len);
                defer allocator.free(tmp);
                for (tmp, values) |*dst, src| dst.* = .fromF32(src);
                return zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(tmp));
            },
            else => return error.UnsupportedEmbedDtype,
        }
    }

    fn bufferToF32(allocator: std.mem.Allocator, io: std.Io, buf: zml.Buffer) ![]f32 {
        const slice = try buf.toSliceAlloc(allocator, io);
        defer slice.free(allocator);
        switch (buf.shape().dtype()) {
            .f32 => return allocator.dupe(f32, slice.items(f32)),
            .bf16 => {
                const src = slice.items(zml.floats.BFloat16);
                const out = try allocator.alloc(f32, src.len);
                for (out, src) |*dst, v| dst.* = v.toF32();
                return out;
            },
            else => return error.UnsupportedEmbedDtype,
        }
    }

    fn tileTemporal(allocator: std.mem.Allocator, src: []const f32, temporal: u32, width: u32) ![]f32 {
        if (temporal <= 1) return allocator.dupe(f32, src);
        const out = try allocator.alloc(f32, src.len * temporal);
        var t: u32 = 0;
        while (t < temporal) : (t += 1) {
            @memcpy(out[t * src.len ..][0..src.len], src);
        }
        _ = width;
        return out;
    }

    pub fn runImage(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        compiled: *const @import("generate.zig").pipeline.VisionCompiled,
        loaded: *const LoadedModel,
        cache: *const WeightCache,
        rgb: []const u8,
        src_h: u32,
        src_w: u32,
    ) !EncodedVisual {
        const patched = try patchifyRgb(allocator, rgb, src_h, src_w, loaded.cfg);
        defer allocator.free(patched.patches);
        return runPatches(allocator, io, platform, compiled, loaded, cache, patched.patches, patched.grid, patched.seq, 1);
    }

    pub fn runVideo(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        compiled: *const @import("generate.zig").pipeline.VisionCompiled,
        loaded: *const LoadedModel,
        cache: *const WeightCache,
        rgb: []const u8,
        frames: u32,
        src_h: u32,
        src_w: u32,
    ) !EncodedVisual {
        const patched = try patchifyVideo(allocator, rgb, frames, src_h, src_w, loaded.cfg);
        defer allocator.free(patched.patches);
        return runPatches(allocator, io, platform, compiled, loaded, cache, patched.patches, patched.grid, patched.seq, patched.temporal);
    }

    pub fn applyVisionPositions(pos: []f32, start: u32, tokens: u32, grid_h: u32, grid_w: u32, temporal: u32, cursor: *f32) void {
        const rows = @max(grid_h / 2, 1);
        const cols = @max(grid_w / 2, 1);
        const time = @max(temporal, 1);
        const base = cursor.*;
        const spatial = rows * cols;
        var i: u32 = 0;
        while (i < tokens) : (i += 1) {
            const ti = if (time == 1) 0 else i / spatial;
            const rem = if (time == 1) i else i % spatial;
            const r = rem / cols;
            const c = rem % cols;
            pos[(start + i) * 3 + 0] = base + @as(f32, @floatFromInt(ti));
            pos[(start + i) * 3 + 1] = base + @as(f32, @floatFromInt(r));
            pos[(start + i) * 3 + 2] = base + @as(f32, @floatFromInt(c));
        }
        cursor.* = base + @as(f32, @floatFromInt(@max(@max(rows, cols), time)));
    }

    pub fn configFromRepo(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, text_hidden: i64) !Config {
        const parsed = try config_mod.parseJson(FileConfig, allocator, io, repo, "config.json");
        defer parsed.deinit();
        return parsed.value.resolve(text_hidden);
    }

    pub fn spatialTokens(cfg: Config, src_h: u32, src_w: u32, video: bool) struct { grid: Grid, seq: u32, merged: u32 } {
        const size = chooseGrid(cfg, src_h, src_w, video);
        const patch: u32 = @intCast(cfg.patch_size);
        const gh = size.h / patch;
        const gw = size.w / patch;
        const seq = gh * gw;
        return .{ .grid = .{ .h = gh, .w = gw }, .seq = seq, .merged = seq / @as(u32, @intCast(cfg.mergeUnit())) };
    }
};

// --- model/dit.zig ---
pub const dit = struct {
    const std = @import("std");

    const zml = @import("zml");

    const config_mod = @import("model.zig").config;

    const log = std.log.scoped(.minimax_h3);

    pub const Config = config_mod.Config;

    const RmsNorm = struct {
        weight: zml.Tensor,
        eps: f32,

        pub fn init(store: zml.io.TensorStore.View, tagz: anytype, eps: f32) RmsNorm {
            return .{
                .weight = store.createTensor("weight", tagz, .replicated),
                .eps = eps,
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
            self.weight.deinit();
        }

        pub fn forward(self: RmsNorm, input: zml.Tensor, axis: anytype) zml.Tensor {
            const normalized = zml.nn.rmsNorm(input, axis, self.eps);
            return normalized.mul(self.weight.convert(input.dtype()).broad(input.shape()));
        }
    };

    fn linear(store: zml.io.TensorStore.View, weight_name: []const u8, bias_name: ?[]const u8, partitions: anytype, bias_partitions: anytype) zml.nn.Linear {
        return .fromStore(store, weight_name, bias_name, partitions, bias_partitions, .d);
    }

    const SwiGlu = struct {
        fc1: zml.nn.Linear,
        fc2: zml.nn.Linear,

        pub fn init(store: zml.io.TensorStore.View) SwiGlu {
            return .{
                .fc1 = linear(store, "fc1.weight", null, .{ .dout = .model, .d = .replicated }, .replicated),
                .fc2 = linear(store, "fc2.weight", null, .{ .dout = .replicated, .d = .model }, .replicated),
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(SwiGlu)) void {
            zml.nn.Linear.unloadBuffers(&self.fc1);
            zml.nn.Linear.unloadBuffers(&self.fc2);
        }

        pub fn forward(self: SwiGlu, x: zml.Tensor) zml.Tensor {
            const uv = self.fc1.forward(x);
            const gate, const value = uv.chunkExact(-1, 2);
            return self.fc2.forward(gate.silu().mul(value).rename(.{ .dout = .d }));
        }
    };

    const Attention = struct {
        qkv: zml.nn.Linear,
        out: zml.nn.Linear,
        q_norm: RmsNorm,
        k_norm: RmsNorm,
        num_heads: i64,
        head_dim: i64,
        attn_kind: policy.AttnKind = .vanilla,

        pub fn init(store: zml.io.TensorStore.View, cfg: Config) Attention {
            const qkv_part = .{ .dout = .model, .d = .replicated };
            const out_part = .{ .dout = .replicated, .d = .model };
            return .{
                .qkv = linear(store, "qkv_proj.weight", null, qkv_part, .replicated),
                .out = linear(store, "out_proj.weight", null, out_part, .replicated),
                .q_norm = .init(store.withPrefix("q_norm"), .{.hd}, cfg.qk_norm_eps),
                .k_norm = .init(store.withPrefix("k_norm"), .{.hd}, cfg.qk_norm_eps),
                .num_heads = cfg.num_attention_heads,
                .head_dim = cfg.attention_head_dim,
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(Attention)) void {
            zml.nn.Linear.unloadBuffers(&self.qkv);
            zml.nn.Linear.unloadBuffers(&self.out);
            RmsNorm.unloadBuffers(&self.q_norm);
            RmsNorm.unloadBuffers(&self.k_norm);
        }

        pub fn forward(self: Attention, x: zml.Tensor, rotary: ?struct { zml.Tensor, zml.Tensor }) zml.Tensor {
            const x_qkv = x.withPartitioning(.{ .d = .replicated });
            // Fused `qkv_proj` is `(heads, 3, head_dim)`: per-head `[Q|K|V]`.
            const split = self.qkv.forward(x_qkv).splitAxis(.dout, .{ .h = self.num_heads, .p = 3, .hd = self.head_dim })
                .withPartitioning(.{ .h = .model });
            const parts = split.chunkExact(.p, 3);
            var q = parts[0].squeeze(.p);
            var k = parts[1].squeeze(.p);
            const v = parts[2].squeeze(.p);

            q = self.q_norm.forward(q, .hd);
            k = self.k_norm.forward(k, .hd);
            if (rotary) |pe| {
                q = applyRotary(q, pe[0], pe[1]);
                k = applyRotary(k, pe[0], pe[1]);
            }
            const q_s = q.rename(.{ .s = .q });
            const k_s = k.rename(.{ .s = .k });
            const v_s = v.rename(.{ .s = .k });
            const attn = switch (self.attn_kind) {
                .cuda_fa2 => zml.attention.flashattn.fa2.dense(q_s, k_s, v_s, .{ .is_causal = false }),
                .vanilla => zml.nn.sdpa(q_s, k_s, v_s, .{}),
            }.rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } });
            return self.out.forward(attn).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });
        }
    };

    fn rotateHalf(x: zml.Tensor) zml.Tensor {
        const half = @divExact(x.dim(-1), 2);
        const x1 = x.slice1d(-1, .{ .start = 0, .end = half });
        const x2 = x.slice1d(-1, .{ .start = half, .end = x.dim(-1) });
        return zml.Tensor.concatenate(&.{ x2.negate(), x1 }, -1);
    }

    fn applyRotary(x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const rotary_dim = cos.dim(-1);
        const x_rot = x.slice1d(-1, .{ .start = 0, .end = rotary_dim });
        const x_pass = x.slice1d(-1, .{ .start = rotary_dim, .end = x.dim(-1) });
        const cos_x = cos.rename(.{ .f = .hd }).broad(x_rot.shape());
        const sin_x = sin.rename(.{ .f = .hd }).broad(x_rot.shape());
        const rotated = x_rot.mul(cos_x).add(rotateHalf(x_rot).mul(sin_x));
        return zml.Tensor.concatenate(&.{ rotated, x_pass }, -1);
    }

    pub fn mmRope(position_ids: zml.Tensor, rope_freq_dim: i64, rope_theta: f32) struct { zml.Tensor, zml.Tensor } {
        const pos = position_ids.convert(.f32).withPartialTags(.{ .s, .ax });
        const inv = zml.nn.invFreq(2 * rope_freq_dim, .{
            .layout = .real_im_pass,
            .scaling = .{ .default = .{ .rope_theta = rope_theta } },
        }).withTags(.{.f});
        const freqs = pos.outer(inv);
        const parts = freqs.chunkExact(.ax, 3);
        const cat3 = zml.Tensor.concatenate(&.{ parts[0].squeeze(.ax), parts[1].squeeze(.ax), parts[2].squeeze(.ax) }, .f);
        const emb = zml.Tensor.concatenate(&.{ cat3, cat3 }, .f);
        return .{ emb.cos(), emb.sin() };
    }

    pub const TimeEmbedder = struct {
        table: ?zml.Tensor = null,
        proj_in: ?zml.nn.Linear = null,
        proj_out: ?zml.nn.Linear = null,

        pub fn init(store: zml.io.TensorStore.View) TimeEmbedder {
            if (store.getShape("time_embedder.proj_in.weight") != null) {
                const prefix = store.withPrefix("time_embedder");
                return .{
                    .proj_in = linear(prefix, "proj_in.weight", "proj_in.bias", .replicated, .replicated),
                    .proj_out = linear(prefix, "proj_out.weight", "proj_out.bias", .replicated, .replicated),
                };
            }
            if (store.maybeCreateTensor("adaln_t_table", .{ .t, .d }, .replicated)) |table| {
                return .{ .table = table };
            }
            std.debug.panic("DiT has neither time_embedder nor adaln_t_table", .{});
        }

        pub fn outDim(self: TimeEmbedder) i64 {
            if (self.table) |table| return table.dim(.d);
            return self.proj_out.?.weight.dim(.dout);
        }

        pub fn unloadBuffers(self: *zml.Bufferized(TimeEmbedder)) void {
            if (self.table) |*table| table.deinit();
            if (self.proj_in) |*layer| zml.nn.Linear.unloadBuffers(layer);
            if (self.proj_out) |*layer| zml.nn.Linear.unloadBuffers(layer);
        }

        fn forwardMlp(self: TimeEmbedder, features: zml.Tensor) zml.Tensor {
            return self.proj_out.?.forward(self.proj_in.?.forward(features).silu().rename(.{ .dout = .d })).rename(.{ .dout = .d });
        }

        fn activateAdaln(self: TimeEmbedder) bool {
            return self.proj_in != null;
        }

        pub fn forward(self: TimeEmbedder, t: zml.Tensor, freq_dim: i64) zml.Tensor {
            if (self.table) |table| return interpolateTable(table, t);
            return self.forwardMlp(timestepFeatures(t, freq_dim));
        }
    };

    /// Maps `t ∈ [0, 1]` onto a table with `rows` evenly spaced entries.
    pub fn tableCoord(t: f32, rows: u32) struct { i0: u32, i1: u32, frac: f32 } {
        std.debug.assert(rows >= 2);
        const last = @as(f32, @floatFromInt(rows - 1));
        const x = std.math.clamp(t, 0.0, 1.0) * last;
        const lo: u32 = @intFromFloat(@floor(x));
        const hi = @min(lo + 1, rows - 1);
        return .{
            .i0 = lo,
            .i1 = hi,
            .frac = x - @as(f32, @floatFromInt(lo)),
        };
    }

    fn interpolateTable(table: zml.Tensor, t: zml.Tensor) zml.Tensor {
        const last_i = table.dim(.t) - 1;
        const last = zml.Tensor.scalar(@as(f32, @floatFromInt(last_i)), .f32);
        const x = t.convert(.f32).mul(last).clamp(zml.Tensor.scalar(0, .f32), last);
        const lo = x.floor();
        const hi = lo.addConstant(1).minimum(last);
        const a = table.gather(.{ .t = lo.convert(.u32).withPartialTags(.{.n}) }, .{});
        const b = table.gather(.{ .t = hi.convert(.u32).withPartialTags(.{.n}) }, .{});
        const frac = x.sub(lo).withPartialTags(.{.n}).broad(a.shape());
        return a.mul(zml.Tensor.scalar(1, a.dtype()).sub(frac.convert(a.dtype()))).add(b.mul(frac.convert(b.dtype())));
    }

    pub fn timestepFeatures(t: zml.Tensor, dim: i64) zml.Tensor {
        const inv = zml.nn.invFreq(dim, .{
            .layout = .real_im_pass,
            .scaling = .{ .default = .{ .rope_theta = 10000.0 } },
        }).withTags(.{.f});
        const angles = t.convert(.f32).withPartialTags(.{.n}).outer(inv);
        return zml.Tensor.concatenate(&.{ angles.cos(), angles.sin() }, .f).rename(.{ .f = .d });
    }

    pub const AdaLn = struct {
        linear: zml.nn.Linear,
        hidden_size: i64,
        expand: i64,
        modalities: i64,
        activate: bool,

        pub fn init(store: zml.io.TensorStore.View, hidden_size: i64, expand: i64, modalities: i64, activate: bool) AdaLn {
            return .{
                .linear = linear(store, "linear.weight", "linear.bias", .replicated, .replicated),
                .hidden_size = hidden_size,
                .expand = expand,
                .modalities = modalities,
                .activate = activate,
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(AdaLn)) void {
            zml.nn.Linear.unloadBuffers(&self.linear);
        }

        pub fn forward(self: AdaLn, temb: zml.Tensor) zml.Tensor {
            const cond = if (self.activate) temb.silu() else temb;
            const raw = self.linear.forward(cond.convert(self.linear.weight.dtype()));
            if (self.modalities == 1) {
                return raw.splitAxis(.dout, .{ .k = self.expand, .d = self.hidden_size });
            }
            return raw.splitAxis(.dout, .{
                .mod = self.modalities,
                .k = self.expand,
                .d = self.hidden_size,
            });
        }
    };

    pub const BlockCore = struct {
        norm1: RmsNorm,
        attn: Attention,
        norm2: RmsNorm,
        mlp: SwiGlu,
        hidden_size: i64,

        pub const Input = struct {
            layer: BlockCore,
            hidden: zml.Tensor,
            mods: zml.Tensor,
            adaln_indices: zml.Tensor,
            cos: zml.Tensor,
            sin: zml.Tensor,
        };

        pub const Output = struct {
            hidden: zml.Tensor,
        };

        pub fn unloadBuffers(self: *zml.Bufferized(BlockCore)) void {
            RmsNorm.unloadBuffers(&self.norm1);
            Attention.unloadBuffers(&self.attn);
            RmsNorm.unloadBuffers(&self.norm2);
            SwiGlu.unloadBuffers(&self.mlp);
        }

        pub fn forward(input: Input) Output {
            const self = input.layer;
            const mods = if (input.mods.shape().hasTag(.mod)) |_|
                input.mods.merge(.{ .n = .{ .n, .mod } })
            else
                input.mods;
            const selected = mods.gather(.{ .n = input.adaln_indices }, .{});
            const parts = selected.chunkExact(.k, 6);
            const shift_msa = parts[0].squeeze(.k);
            const scale_msa = parts[1].squeeze(.k);
            const gate_msa = parts[2].squeeze(.k);
            const shift_mlp = parts[3].squeeze(.k);
            const scale_mlp = parts[4].squeeze(.k);
            const gate_mlp = parts[5].squeeze(.k);

            const residual0 = input.hidden.withPartitioning(.{ .d = .replicated });
            const n1 = self.norm1.forward(residual0, .d);
            const one = zml.Tensor.scalar(1.0, n1.dtype());
            const attn_in = n1.mul(one.add(scale_msa.convert(n1.dtype()).broad(n1.shape()))).add(shift_msa.convert(n1.dtype()).broad(n1.shape()));
            const attn_out = self.attn.forward(attn_in, .{ input.cos, input.sin });
            const x1 = residual0.add(gate_msa.convert(attn_out.dtype()).broad(attn_out.shape()).mul(attn_out)).withPartitioning(.{ .d = .replicated });

            const n2 = self.norm2.forward(x1, .d);
            const mlp_in = n2.mul(one.add(scale_mlp.convert(n2.dtype()).broad(n2.shape()))).add(shift_mlp.convert(n2.dtype()).broad(n2.shape()));
            const mlp_out = self.mlp.forward(mlp_in).rename(.{ .dout = .d });
            const x2 = x1.add(gate_mlp.convert(mlp_out.dtype()).broad(mlp_out.shape()).mul(mlp_out)).withPartitioning(.{ .d = .replicated });
            return .{ .hidden = x2.reuseBuffer(input.hidden) };
        }
    };

    pub const StepBlockInput = struct {
        layer: BlockCore,
        hidden: zml.Tensor,
        table: zml.Tensor,
        step: zml.Tensor,
        adaln_indices: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
    };

    pub fn stepBlock(input: StepBlockInput) BlockCore.Output {
        const mods = input.table.gather(.{ .t = input.step }, .{});
        return BlockCore.forward(.{
            .layer = input.layer,
            .hidden = input.hidden,
            .mods = mods,
            .adaln_indices = input.adaln_indices,
            .cos = input.cos,
            .sin = input.sin,
        });
    }

    pub const BlockGroup = struct {
        layers: []BlockCore,

        pub const Input = struct {
            group: BlockGroup,
            hidden: zml.Tensor,
            tables: []zml.Tensor,
            step: zml.Tensor,
            adaln_indices: zml.Tensor,
            cos: zml.Tensor,
            sin: zml.Tensor,
        };

        pub const Output = struct {
            hidden: zml.Tensor,
        };

        pub fn unloadBuffers(self: *zml.Bufferized(BlockGroup), allocator: std.mem.Allocator) void {
            for (self.layers) |*layer| BlockCore.unloadBuffers(layer);
            allocator.free(self.layers);
        }

        pub fn forward(input: Input) Output {
            var hidden = input.hidden;
            for (input.group.layers, input.tables) |layer, table| {
                const mods = table.gather(.{ .t = input.step }, .{});
                hidden = BlockCore.forward(.{
                    .layer = layer,
                    .hidden = hidden,
                    .mods = mods,
                    .adaln_indices = input.adaln_indices,
                    .cos = input.cos,
                    .sin = input.sin,
                }).hidden;
            }
            return .{ .hidden = hidden };
        }
    };

    pub const TransformerBlock = struct {
        norm1: RmsNorm,
        attn: Attention,
        norm2: RmsNorm,
        mlp: SwiGlu,
        adaln: AdaLn,
        hidden_size: i64,

        pub fn init(store: zml.io.TensorStore.View, cfg: Config, activate_adaln: bool) TransformerBlock {
            const attn_store = store.withPrefix("attn");
            const mlp_store = store.withPrefix("mlp");
            const adaln_store = store.withPrefix("adaln_proj");
            return .{
                .norm1 = .init(store.withPrefix("norm1"), .{.d}, cfg.norm_eps),
                .attn = .init(attn_store, cfg),
                .norm2 = .init(store.withPrefix("norm2"), .{.d}, cfg.norm_eps),
                .mlp = .init(mlp_store),
                .adaln = .init(adaln_store, cfg.hidden_size, 6, config_mod.modality_count, activate_adaln),
                .hidden_size = cfg.hidden_size,
            };
        }

        pub fn corePart(self: TransformerBlock) BlockCore {
            return .{
                .norm1 = self.norm1,
                .attn = self.attn,
                .norm2 = self.norm2,
                .mlp = self.mlp,
                .hidden_size = self.hidden_size,
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(TransformerBlock)) void {
            RmsNorm.unloadBuffers(&self.norm1);
            Attention.unloadBuffers(&self.attn);
            RmsNorm.unloadBuffers(&self.norm2);
            SwiGlu.unloadBuffers(&self.mlp);
            AdaLn.unloadBuffers(&self.adaln);
        }
    };

    const TokenRefinerBlock = struct {
        norm1: RmsNorm,
        attn: Attention,
        norm2: RmsNorm,
        mlp: SwiGlu,

        pub fn init(store: zml.io.TensorStore.View, cfg: Config) TokenRefinerBlock {
            const attn_store = store.withPrefix("attn");
            const mlp_store = store.withPrefix("mlp");
            return .{
                .norm1 = .init(store.withPrefix("norm1"), .{.d}, cfg.norm_eps),
                .attn = .init(attn_store, cfg),
                .norm2 = .init(store.withPrefix("norm2"), .{.d}, cfg.norm_eps),
                .mlp = .init(mlp_store),
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(TokenRefinerBlock)) void {
            RmsNorm.unloadBuffers(&self.norm1);
            Attention.unloadBuffers(&self.attn);
            RmsNorm.unloadBuffers(&self.norm2);
            SwiGlu.unloadBuffers(&self.mlp);
        }

        pub fn forward(self: TokenRefinerBlock, x: zml.Tensor) zml.Tensor {
            const residual = x.withPartitioning(.{ .d = .replicated });
            const x1 = residual.add(self.attn.forward(self.norm1.forward(residual, .d), null));
            return x1.add(self.mlp.forward(self.norm2.forward(x1, .d)).rename(.{ .dout = .d })).withPartitioning(.{ .d = .replicated }).reuseBuffer(x);
        }
    };

    const TokenRefiner = struct {
        blocks: []TokenRefinerBlock,
        final_norm: RmsNorm,

        pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, cfg: Config) !TokenRefiner {
            const block_store = store.withPrefix("blocks");
            const blocks = try allocator.alloc(TokenRefinerBlock, @intCast(cfg.num_refiner_layers));
            errdefer allocator.free(blocks);
            for (blocks, 0..) |*block, i| {
                block.* = .init(block_store.withLayer(i), cfg);
            }
            return .{
                .blocks = blocks,
                .final_norm = .init(store.withPrefix("final_norm"), .{.d}, cfg.final_norm_eps),
            };
        }

        pub fn deinit(self: TokenRefiner, allocator: std.mem.Allocator) void {
            allocator.free(self.blocks);
        }

        pub fn unloadBuffers(self: *zml.Bufferized(TokenRefiner), allocator: std.mem.Allocator) void {
            for (self.blocks) |*block| TokenRefinerBlock.unloadBuffers(block);
            allocator.free(self.blocks);
            RmsNorm.unloadBuffers(&self.final_norm);
        }

        pub fn forward(self: TokenRefiner, x: zml.Tensor) zml.Tensor {
            var hidden = x;
            for (self.blocks) |block| {
                hidden = block.forward(hidden);
            }
            return self.final_norm.forward(hidden, .d);
        }
    };

    const FinalLayer = struct {
        norm: RmsNorm,
        adaln: AdaLn,
        video_out: zml.nn.Linear,
        audio_out: zml.nn.Linear,

        pub fn init(store: zml.io.TensorStore.View, cfg: Config, activate_adaln: bool) FinalLayer {
            return .{
                .norm = .init(store.withPrefix("norm"), .{.d}, cfg.final_norm_eps),
                .adaln = .init(store.withPrefix("adaln_proj"), cfg.hidden_size, 2, 1, activate_adaln),
                .video_out = linear(store, "video_out.weight", "video_out.bias", .replicated, .replicated),
                .audio_out = linear(store, "audio_out.weight", "audio_out.bias", .replicated, .replicated),
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(FinalLayer)) void {
            RmsNorm.unloadBuffers(&self.norm);
            AdaLn.unloadBuffers(&self.adaln);
            zml.nn.Linear.unloadBuffers(&self.video_out);
            zml.nn.Linear.unloadBuffers(&self.audio_out);
        }
    };

    pub const Model = struct {
        video_proj: zml.nn.Linear,
        audio_proj: zml.nn.Linear,
        condition_proj: zml.nn.Linear,
        time_embedder: TimeEmbedder,
        token_refiner: TokenRefiner,
        blocks: []TransformerBlock,
        final_layer: FinalLayer,
        cfg: Config,

        pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, cfg: Config) !Model {
            const blocks_store = store.withPrefix("blocks");
            const blocks = try allocator.alloc(TransformerBlock, @intCast(cfg.num_layers));
            errdefer allocator.free(blocks);

            const token_refiner = try TokenRefiner.init(allocator, store.withPrefix("token_refiner"), cfg);
            errdefer token_refiner.deinit(allocator);

            const time_embedder: TimeEmbedder = .init(store);
            const activate_adaln = time_embedder.activateAdaln();
            for (blocks, 0..) |*block, i| {
                block.* = .init(blocks_store.withLayer(i), cfg, activate_adaln);
            }

            return .{
                .video_proj = linear(store, "video_patch_proj.weight", "video_patch_proj.bias", .replicated, .replicated),
                .audio_proj = linear(store, "audio_patch_proj.weight", "audio_patch_proj.bias", .replicated, .replicated),
                .condition_proj = linear(store, "condition_proj.weight", "condition_proj.bias", .replicated, .replicated),
                .time_embedder = time_embedder,
                .token_refiner = token_refiner,
                .blocks = blocks,
                .final_layer = FinalLayer.init(store.withPrefix("final_layer"), cfg, activate_adaln),
                .cfg = cfg,
            };
        }

        pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
            self.token_refiner.deinit(allocator);
            allocator.free(self.blocks);
        }

        pub fn textPrep(self: Model) TextPrep {
            return .{
                .condition_proj = self.condition_proj,
                .token_refiner = self.token_refiner,
            };
        }

        pub fn patchEmbed(self: Model) PatchEmbed {
            return .{
                .video_proj = self.video_proj,
                .audio_proj = self.audio_proj,
                .hidden_size = self.cfg.hidden_size,
                .seq = 0,
            };
        }

        pub fn finishCore(self: Model) FinishCore {
            return .{
                .norm = self.final_layer.norm,
                .video_out = self.final_layer.video_out,
                .audio_out = self.final_layer.audio_out,
            };
        }

        pub fn applyBackend(self: *Model, dit_kind: policy.AttnKind, refiner_kind: policy.AttnKind) void {
            for (self.blocks) |*block| block.attn.attn_kind = dit_kind;
            for (self.token_refiner.blocks) |*block| block.attn.attn_kind = refiner_kind;
        }
    };

    pub const TextPrep = struct {
        condition_proj: zml.nn.Linear,
        token_refiner: TokenRefiner,

        pub fn unloadBuffers(self: *zml.Bufferized(TextPrep), allocator: std.mem.Allocator) void {
            zml.nn.Linear.unloadBuffers(&self.condition_proj);
            TokenRefiner.unloadBuffers(&self.token_refiner, allocator);
        }
    };

    pub const PatchEmbed = struct {
        video_proj: zml.nn.Linear,
        audio_proj: zml.nn.Linear,
        hidden_size: i64,
        seq: i64 = 0,

        pub fn unloadBuffers(self: *zml.Bufferized(PatchEmbed)) void {
            zml.nn.Linear.unloadBuffers(&self.video_proj);
            zml.nn.Linear.unloadBuffers(&self.audio_proj);
        }
    };

    pub const FinishCore = struct {
        norm: RmsNorm,
        video_out: zml.nn.Linear,
        audio_out: zml.nn.Linear,

        pub fn unloadBuffers(self: *zml.Bufferized(FinishCore)) void {
            RmsNorm.unloadBuffers(&self.norm);
            zml.nn.Linear.unloadBuffers(&self.video_out);
            zml.nn.Linear.unloadBuffers(&self.audio_out);
        }
    };

    pub const TextPrepInput = struct {
        model: TextPrep,
        text: zml.Tensor,
    };

    pub const TextPrepOutput = struct {
        text: zml.Tensor,
    };

    pub fn prepareText(input: TextPrepInput) TextPrepOutput {
        const self = input.model;
        var text = self.condition_proj.forward(input.text.convert(self.condition_proj.weight.dtype())).rename(.{ .dout = .d });
        text = self.token_refiner.forward(text.convert(self.token_refiner.final_norm.weight.dtype()));
        return .{ .text = text };
    }

    pub const RopeModel = struct {
        rope_freq_dim: i64,
        rope_theta: f32,
        out_dtype: zml.DataType,
    };

    pub const RopeInput = struct {
        model: RopeModel,
        position_ids: zml.Tensor,
    };

    pub const RopeOutput = struct {
        cos: zml.Tensor,
        sin: zml.Tensor,
    };

    pub fn prepareRope(input: RopeInput) RopeOutput {
        const cos, const sin = mmRope(input.position_ids, input.model.rope_freq_dim, input.model.rope_theta);
        return .{
            .cos = cos.convert(input.model.out_dtype),
            .sin = sin.convert(input.model.out_dtype),
        };
    }

    pub const PatchInput = struct {
        model: PatchEmbed,
        video: zml.Tensor,
        audio: zml.Tensor,
        text: zml.Tensor,
        video_indices: zml.Tensor,
        audio_indices: zml.Tensor,
        text_indices: zml.Tensor,
    };

    pub const PatchOutput = struct {
        hidden: zml.Tensor,
    };

    pub fn embedPatches(input: PatchInput) PatchOutput {
        const self = input.model;
        const video = self.video_proj.forward(input.video.convert(self.video_proj.weight.dtype())).rename(.{ .dout = .d });
        const audio = self.audio_proj.forward(input.audio.convert(self.audio_proj.weight.dtype())).rename(.{ .dout = .d });
        const text = input.text;
        const batch = text.dim(.b);
        var hidden = zml.Tensor.zeroes(zml.Shape.init(.{ .b = batch, .s = self.seq, .d = self.hidden_size }, text.dtype()));
        hidden = hidden.scatterSlices(.{ .s = input.text_indices.withTags(.{.s}) }, text, .{ .update_fn = zml.Tensor.ScatterOpts.override });
        hidden = hidden.scatterSlices(.{ .s = input.video_indices.withTags(.{.s}) }, video.convert(text.dtype()), .{ .update_fn = zml.Tensor.ScatterOpts.override });
        hidden = hidden.scatterSlices(.{ .s = input.audio_indices.withTags(.{.s}) }, audio.convert(text.dtype()), .{ .update_fn = zml.Tensor.ScatterOpts.override });
        return .{ .hidden = hidden.withPartitioning(.{ .d = .replicated }) };
    }

    pub const TembInput = struct {
        model: TimeEmbedder,
        timestep: zml.Tensor,
        freq_dim: i64,
    };

    pub const TembOutput = struct {
        temb: zml.Tensor,
    };

    pub fn prepareTemb(input: TembInput) TembOutput {
        return .{ .temb = input.model.forward(input.timestep, input.freq_dim) };
    }

    pub const AdaLnPrep = struct {
        adaln: AdaLn,
        steps: i64,
        slots: i64,
    };

    pub const AdaLnPrepInput = struct {
        model: AdaLnPrep,
        temb: zml.Tensor,
    };

    pub const AdaLnPrepOutput = struct {
        table: zml.Tensor,
    };

    pub fn prepareAdaln(input: AdaLnPrepInput) AdaLnPrepOutput {
        const raw = input.model.adaln.forward(input.temb);
        return .{ .table = raw.splitAxis(.n, .{ .t = input.model.steps, .n = input.model.slots }) };
    }

    pub const ScatterInput = struct {
        hidden: zml.Tensor,
        values: zml.Tensor,
        indices: zml.Tensor,
    };

    pub const ScatterOutput = struct {
        hidden: zml.Tensor,
    };

    pub fn scatterRows(input: ScatterInput) ScatterOutput {
        const hidden = input.hidden.scatterSlices(
            .{ .s = input.indices.withTags(.{.s}) },
            input.values.convert(input.hidden.dtype()),
            .{ .update_fn = zml.Tensor.ScatterOpts.override },
        );
        return .{ .hidden = hidden };
    }

    pub const FinishInput = struct {
        model: FinishCore,
        hidden: zml.Tensor,
        table: zml.Tensor,
        step: zml.Tensor,
        timestep_indices: zml.Tensor,
        video_indices: zml.Tensor,
        audio_indices: zml.Tensor,
    };

    pub const FinishOutput = struct {
        video: zml.Tensor,
        audio: zml.Tensor,
    };

    fn modulateRows(norm: RmsNorm, hidden: zml.Tensor, mods: zml.Tensor, timestep_indices: zml.Tensor) zml.Tensor {
        const n = norm.forward(hidden.withPartitioning(.{ .d = .replicated }), .d);
        const selected = mods.gather(.{ .n = timestep_indices }, .{});
        const parts = selected.chunkExact(.k, 2);
        const shift = parts[0].squeeze(.k);
        const scale = parts[1].squeeze(.k);
        const one = zml.Tensor.scalar(1.0, n.dtype());
        return n.mul(one.add(scale.convert(n.dtype()).broad(n.shape()))).add(shift.convert(n.dtype()).broad(n.shape()));
    }

    pub fn finish(input: FinishInput) FinishOutput {
        const self = input.model;
        const mods = input.table.gather(.{ .t = input.step }, .{});
        const video_h = input.hidden.gather(.{ .s = input.video_indices.withTags(.{.idx}) }, .{}).rename(.{ .idx = .s });
        const audio_h = input.hidden.gather(.{ .s = input.audio_indices.withTags(.{.idx}) }, .{}).rename(.{ .idx = .s });
        const video_t = input.timestep_indices.gather(.{ .s = input.video_indices.withTags(.{.idx}) }, .{}).rename(.{ .idx = .s });
        const audio_t = input.timestep_indices.gather(.{ .s = input.audio_indices.withTags(.{.idx}) }, .{}).rename(.{ .idx = .s });
        const video_m = modulateRows(self.norm, video_h, mods, video_t);
        const audio_m = modulateRows(self.norm, audio_h, mods, audio_t);
        return .{
            .video = self.video_out.forward(video_m.convert(self.video_out.weight.dtype())),
            .audio = self.audio_out.forward(audio_m.convert(self.audio_out.weight.dtype())),
        };
    }

    pub const LoadedModel = struct {
        inner: Model,
        cfg: Config,

        pub fn init(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, store: zml.io.TensorStore.View) !LoadedModel {
            const cfg = try config_mod.loadDitConfig(allocator, io, repo);
            log.info("dit: {d} layers hidden={d} heads={d} text_dim={d}", .{
                cfg.num_layers,
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.text_dim,
            });
            return .{
                .inner = try .init(allocator, store, cfg),
                .cfg = cfg,
            };
        }

        pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
            self.inner.deinit(allocator);
        }

        pub fn loadCore(
            self: *const LoadedModel,
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            store: *zml.io.TensorStore,
            shardings: []const zml.Sharding,
            index: usize,
            progress: *std.Progress.Node,
            loader: ?*zml.io.Loader,
        ) !zml.Bufferized(BlockCore) {
            const core = self.inner.blocks[index].corePart();
            return weights.load(allocator, io, platform, store, shardings, BlockCore, &core, progress, loader);
        }

        pub fn loadAdaln(
            self: *const LoadedModel,
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            store: *zml.io.TensorStore,
            shardings: []const zml.Sharding,
            index: usize,
            progress: *std.Progress.Node,
            loader: ?*zml.io.Loader,
        ) !zml.Bufferized(AdaLn) {
            return weights.load(allocator, io, platform, store, shardings, AdaLn, &self.inner.blocks[index].adaln, progress, loader);
        }

        pub fn loadTextPrep(
            self: *const LoadedModel,
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            store: *zml.io.TensorStore,
            shardings: []const zml.Sharding,
            progress: *std.Progress.Node,
        ) !zml.Bufferized(TextPrep) {
            const part = self.inner.textPrep();
            return weights.load(allocator, io, platform, store, shardings, TextPrep, &part, progress, null);
        }

        pub fn loadPatchEmbed(
            self: *const LoadedModel,
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            store: *zml.io.TensorStore,
            shardings: []const zml.Sharding,
            progress: *std.Progress.Node,
        ) !zml.Bufferized(PatchEmbed) {
            const part = self.inner.patchEmbed();
            return weights.load(allocator, io, platform, store, shardings, PatchEmbed, &part, progress, null);
        }

        pub fn loadTimeEmbedder(
            self: *const LoadedModel,
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            store: *zml.io.TensorStore,
            shardings: []const zml.Sharding,
            progress: *std.Progress.Node,
        ) !zml.Bufferized(TimeEmbedder) {
            return weights.load(allocator, io, platform, store, shardings, TimeEmbedder, &self.inner.time_embedder, progress, null);
        }

        pub fn loadFinishCore(
            self: *const LoadedModel,
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            store: *zml.io.TensorStore,
            shardings: []const zml.Sharding,
            progress: *std.Progress.Node,
        ) !zml.Bufferized(FinishCore) {
            const part = self.inner.finishCore();
            return weights.load(allocator, io, platform, store, shardings, FinishCore, &part, progress, null);
        }

        pub fn loadFinalAdaln(
            self: *const LoadedModel,
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            store: *zml.io.TensorStore,
            shardings: []const zml.Sharding,
            progress: *std.Progress.Node,
        ) !zml.Bufferized(AdaLn) {
            return weights.load(allocator, io, platform, store, shardings, AdaLn, &self.inner.final_layer.adaln, progress, null);
        }
    };
};
