// Load weights, encode, compile, denoise, VAE, mux.

// --- core/checkpoint.zig ---
pub const checkpoint = struct {
    const std = @import("std");

    pub const Report = struct {
        has_adaln_proj: bool = false,
        has_time: bool = false,
    };

    fn hasKey(keys: []const []const u8, suffix: []const u8) bool {
        for (keys) |key| {
            if (std.mem.endsWith(u8, key, suffix) or std.mem.eql(u8, key, suffix)) return true;
        }
        return false;
    }

    pub fn inspect(keys: []const []const u8) Report {
        return .{
            .has_adaln_proj = hasKey(keys, "adaln_proj.linear.weight"),
            .has_time = hasKey(keys, "time_embedder.proj_in.weight") or hasKey(keys, "time_embedder.linear_1.weight") or hasKey(keys, "adaln_t_table"),
        };
    }

    pub fn refuseReason(report: Report) ?[]const u8 {
        if (!report.has_adaln_proj) return "AdaLN projection weights missing; not a recognized H3 DiT";
        if (!report.has_time) return "neither time_embedder nor adaln_t_table; not a recognized H3 DiT";
        return null;
    }

    pub const bundle_leaves = [_][]const u8{ "diffusion_models", "text_encoders", "vae" };

    fn containsIgnoreCase(hay: []const u8, needle: []const u8) bool {
        if (hay.len < needle.len) return false;
        var i: usize = 0;
        while (i + needle.len <= hay.len) : (i += 1) {
            if (std.ascii.eqlIgnoreCase(hay[i..][0..needle.len], needle)) return true;
        }
        return false;
    }

    /// `needles` are all required (AND). Empty `needles` matches any `.safetensors` file.
    pub fn safetensorsContains(name: []const u8, needles: []const []const u8) bool {
        if (!std.mem.endsWith(u8, name, ".safetensors")) return false;
        for (needles) |needle| {
            if (!containsIgnoreCase(name, needle)) return false;
        }
        return true;
    }

    pub fn isBundleLeaf(name: []const u8) bool {
        for (bundle_leaves) |leaf| {
            if (std.mem.eql(u8, name, leaf)) return true;
        }
        return false;
    }
};

// --- core/sharding.zig ---
pub const sharding = struct {
    const std = @import("std");

    const zml = @import("zml");

    const config = @import("model.zig").config;

    const log = std.log.scoped(.minimax_h3);

    /// Largest tensor-parallel degree that divides DiT heads (56), encoder heads (64), and GQA KV heads (8).
    pub const tensor_parallel_max: usize = 8;

    pub fn tensorParallelDegree(device_count: usize) usize {
        if (device_count == 0) return 0;
        if (device_count >= tensor_parallel_max) return tensor_parallel_max;
        if (device_count >= 4) return 4;
        if (device_count >= 2) return 2;
        return 1;
    }

    pub fn tensorParallelHeadsOk(degree: i64, dit_heads: i64, enc_heads: i64, kv_heads: i64) bool {
        if (degree <= 0) return false;
        return @rem(dit_heads, degree) == 0 and
            @rem(enc_heads, degree) == 0 and
            @rem(kv_heads, degree) == 0;
    }

    pub fn officialHeadsOk(degree: i64) bool {
        const dit = config.Config.official();
        const enc = config.EncoderConfig{};
        return tensorParallelHeadsOk(degree, dit.num_attention_heads, enc.num_attention_heads, enc.num_key_value_heads);
    }

    pub fn tensorParallelPrimaryAxis(target: zml.Target) zml.Sharding.PhysicalAxisTag {
        return switch (target) {
            .tpu => .link_x,
            .neuron => .link,
            .cuda, .rocm, .oneapi => .link,
            .cpu, .metal => .bus,
        };
    }

    pub fn presentShardableAxes(mesh: *const zml.Sharding.PhysicalMesh) zml.stdx.BoundedArray(zml.Sharding.PhysicalAxisTag, zml.Sharding.MAX_MESH_RANK) {
        var out: zml.stdx.BoundedArray(zml.Sharding.PhysicalAxisTag, zml.Sharding.MAX_MESH_RANK) = .empty;
        for (mesh.shardableAxes()) |tag| {
            if (mesh.hasAxis(tag)) out.appendAssumeCapacity(tag);
        }
        if (out.len == 0) {
            for (mesh.axisOrder().slice()) |tag| out.appendAssumeCapacity(tag);
        }
        return out;
    }

    /// Bind `.model` to the fastest present shardable axis and fold every other present axis into it.
    pub fn tensorParallelStrategy(mesh: *const zml.Sharding.PhysicalMesh) error{InvalidPhysicalMesh}!zml.Sharding.Strategy {
        const axes = presentShardableAxes(mesh);
        if (axes.len == 0) return error.InvalidPhysicalMesh;
        var strategy: zml.Sharding.Strategy = .init;
        strategy.addBinding(.model, axes.get(0));
        if (axes.len > 1) strategy.addFold(axes.get(0), axes.constSlice());
        return strategy;
    }

    /// Use all devices when the count is a legal H3 TP degree. Larger power-of-two
    /// meshes (16/32/64) keep the first 8 so head-parallel TP stays exact.
    pub fn physicalMesh(
        allocator: std.mem.Allocator,
        target: zml.Target,
        devices: []const zml.platform.Device,
    ) anyerror!zml.Sharding.PhysicalMesh {
        if (devices.len == 0) return error.MissingDevices;
        const degree = tensorParallelDegree(devices.len);
        if (degree == 0 or degree > devices.len) return error.MissingDevices;
        if (degree == devices.len) {
            return zml.Sharding.PhysicalMesh.auto(allocator, target, devices);
        }
        log.warn(
            "H3 tensor parallel uses {d} of {d} devices (DiT 56 heads and encoder GQA 8 require degree 1, 2, 4, or 8)",
            .{ degree, devices.len },
        );
        return tensorParallelLine(allocator, target, devices[0..degree]);
    }

    fn tensorParallelLine(
        allocator: std.mem.Allocator,
        target: zml.Target,
        devices: []const zml.platform.Device,
    ) !zml.Sharding.PhysicalMesh {
        const nodes = try allocator.alloc(zml.Sharding.PhysicalNode, devices.len);
        errdefer allocator.free(nodes);
        for (nodes, devices) |*node, device| node.* = .device(device);
        const root: zml.Sharding.PhysicalNode = .{
            .branch = .{
                .tag = tensorParallelPrimaryAxis(target),
                .geometry = switch (target) {
                    .tpu => .{ .mesh = .torus },
                    .neuron, .cuda, .rocm, .oneapi => .point_to_point,
                    .cpu, .metal => .tree,
                },
                .children = nodes,
            },
        };
        const mesh = try zml.Sharding.PhysicalMesh.fromTree(allocator, target, root);
        allocator.free(nodes);
        return mesh;
    }

    pub const Shardings = struct {
        model: zml.Sharding,

        pub fn init(platform: *zml.Platform) !Shardings {
            const strategy = try tensorParallelStrategy(&platform.physical_mesh);
            const model = try platform.registerShardingWithStrategy(
                "model",
                .mesh(.{ .model = .high_bandwidth }),
                strategy,
            );
            const degree = model.numPartitionsForLogicalAxis(.model);
            if (!officialHeadsOk(degree)) {
                log.err(
                    "H3 tensor parallel degree {d} does not divide DiT heads 56, encoder heads 64, and GQA KV heads 8. Use 1, 2, 4, or 8 devices.",
                    .{degree},
                );
                return error.IncompatibleSharding;
            }
            return .{ .model = model };
        }

        pub fn all(self: Shardings) [1]zml.Sharding {
            return .{self.model};
        }

        pub fn checkLoaded(self: Shardings, dit_cfg: config.Config, enc_cfg: config.EncoderConfig) !void {
            const degree = self.model.numPartitionsForLogicalAxis(.model);
            if (!tensorParallelHeadsOk(degree, dit_cfg.num_attention_heads, enc_cfg.num_attention_heads, enc_cfg.num_key_value_heads)) {
                log.err(
                    "Loaded heads dit={d} encoder={d} kv={d} do not divide by tensor-parallel degree {d}",
                    .{ dit_cfg.num_attention_heads, enc_cfg.num_attention_heads, enc_cfg.num_key_value_heads, degree },
                );
                return error.IncompatibleSharding;
            }
        }
    };
};

// --- core/request.zig ---
pub const request = struct {
    const std = @import("std");

    const config = @import("model.zig").config;
    const packing = @import("model.zig").packing;

    pub const max_ref_files = config.max_ref_files;
    pub const max_ref_images = config.max_ref_images;
    pub const max_ref_videos = config.max_ref_videos;
    pub const max_ref_audios = config.max_ref_audios;

    pub const Reference = struct {
        kind: packing.ReferenceKind,
        path: []const u8,
        soundtrack: []const u8 = "",
        source_fps: f32 = 0,
        source_rate: u32 = 0,
    };

    pub const Request = struct {
        variant: config.Variant = .t2va,
        prompt: []const u8,
        duration_s: f32 = 5.0,
        aspect: config.Aspect = .@"16:9",
        first_image: []const u8 = "",
        last_image: []const u8 = "",
        refs: []const Reference = &.{},
    };

    pub fn inferVariant(first_image: []const u8, last_image: []const u8, refs: []const Reference) !config.Variant {
        const has_keyframes = first_image.len != 0 or last_image.len != 0;
        if (refs.len != 0) {
            if (has_keyframes) return error.Ref2vaRejectsKeyframes;
            return .ref2va;
        }
        if (has_keyframes) return .fl2va;
        return .t2va;
    }

    pub fn refsToCsv(allocator: std.mem.Allocator, refs: []const Reference) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        for (refs, 0..) |r, i| {
            if (i != 0) try out.append(allocator, ',');
            try out.appendSlice(allocator, r.path);
            if (r.soundtrack.len != 0) {
                try out.append(allocator, ',');
                try out.appendSlice(allocator, r.soundtrack);
            }
        }
        return out.toOwnedSlice(allocator);
    }

    pub fn splitComma(allocator: std.mem.Allocator, text: []const u8) ![][]const u8 {
        if (text.len == 0) return &.{};
        var out: std.ArrayList([]const u8) = .empty;
        errdefer out.deinit(allocator);
        var it = std.mem.splitScalar(u8, text, ',');
        while (it.next()) |part| {
            const trimmed = std.mem.trim(u8, part, " \t");
            if (trimmed.len == 0) continue;
            try out.append(allocator, trimmed);
        }
        return out.toOwnedSlice(allocator);
    }

    pub fn refsFromComma(allocator: std.mem.Allocator, text: []const u8) ![]Reference {
        const paths = try splitComma(allocator, text);
        defer allocator.free(paths);
        return refsFromPaths(allocator, paths);
    }

    pub fn refsFromPaths(allocator: std.mem.Allocator, paths: []const []const u8) ![]Reference {
        var out: std.ArrayList(Reference) = .empty;
        errdefer out.deinit(allocator);
        var i: usize = 0;
        while (i < paths.len) : (i += 1) {
            const kind = media.guessKind(paths[i]);
            if (kind == .video and i + 1 < paths.len and media.guessKind(paths[i + 1]) == .audio) {
                try out.append(allocator, .{
                    .kind = .video_audio,
                    .path = paths[i],
                    .soundtrack = paths[i + 1],
                });
                i += 1;
                continue;
            }
            try out.append(allocator, .{ .kind = kind, .path = paths[i] });
        }
        return out.toOwnedSlice(allocator);
    }

    const ManifestEntry = struct {
        kind: []const u8,
        path: []const u8,
        soundtrack: []const u8 = "",
        fps: f32 = 0,
        sample_rate: u32 = 0,
    };

    pub fn refsFromManifest(allocator: std.mem.Allocator, bytes: []const u8) ![]Reference {
        const parsed = try std.json.parseFromSlice([]ManifestEntry, allocator, bytes, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const out = try allocator.alloc(Reference, parsed.value.len);
        for (parsed.value, out) |entry, *dst| {
            const kind = parseKind(entry.kind) orelse return error.UnknownRefKind;
            dst.* = .{
                .kind = if (kind == .video and entry.soundtrack.len != 0) .video_audio else kind,
                .path = try allocator.dupe(u8, entry.path),
                .soundtrack = try allocator.dupe(u8, entry.soundtrack),
                .source_fps = entry.fps,
                .source_rate = entry.sample_rate,
            };
        }
        return out;
    }

    pub fn freeRefs(allocator: std.mem.Allocator, refs: []Reference, owned_strings: bool) void {
        if (owned_strings) {
            for (refs) |r| {
                allocator.free(r.path);
                if (r.soundtrack.len != 0) allocator.free(r.soundtrack);
            }
        }
        allocator.free(refs);
    }

    pub fn validate(req: Request) !void {
        if (req.duration_s <= 0) return error.DurationInvalid;
        switch (req.variant) {
            .t2va => {
                if (req.first_image.len != 0 or req.last_image.len != 0 or req.refs.len != 0)
                    return error.T2vaRejectsMedia;
            },
            .fl2va => {
                if (req.first_image.len == 0 and req.last_image.len == 0) return error.Fl2vaNeedsImage;
                if (req.refs.len != 0) return error.Fl2vaRejectsRefs;
            },
            .ref2va => {
                if (req.refs.len == 0) return error.Ref2vaNeedsRefs;
                if (req.first_image.len != 0 or req.last_image.len != 0) return error.Ref2vaRejectsKeyframes;
            },
        }
        if (std.mem.trim(u8, req.prompt, " \t\r\n").len == 0) return error.IntentEmpty;
        try validateRefs(req.refs);
    }

    pub fn validateRefs(refs: []const Reference) !void {
        if (refs.len > max_ref_files) return error.TooManyRefs;
        var n_img: u32 = 0;
        var n_vid: u32 = 0;
        var n_aud: u32 = 0;
        for (refs) |r| {
            switch (r.kind) {
                .image => {
                    n_img += 1;
                    if (n_img > max_ref_images) return error.TooManyRefImages;
                },
                .video, .video_audio => {
                    n_vid += 1;
                    if (n_vid > max_ref_videos) return error.TooManyRefVideos;
                    if (r.kind == .video_audio) {
                        n_aud += 1;
                        if (n_aud > max_ref_audios) return error.TooManyRefAudios;
                    }
                },
                .audio => {
                    n_aud += 1;
                    if (n_aud > max_ref_audios) return error.TooManyRefAudios;
                },
            }
        }
        if (n_aud != 0 and n_img == 0 and n_vid == 0) return error.AudioRefNeedsVisual;
    }

    pub fn refsToManifest(allocator: std.mem.Allocator, refs: []const Reference) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        try out.appendSlice(allocator, "[");
        for (refs, 0..) |r, i| {
            if (i != 0) try out.appendSlice(allocator, ",");
            try out.appendSlice(allocator, "{\"kind\":\"");
            try out.appendSlice(allocator, @tagName(r.kind));
            try out.appendSlice(allocator, "\",\"path\":\"");
            try out.appendSlice(allocator, r.path);
            try out.appendSlice(allocator, "\"");
            if (r.soundtrack.len != 0) {
                try out.appendSlice(allocator, ",\"soundtrack\":\"");
                try out.appendSlice(allocator, r.soundtrack);
                try out.appendSlice(allocator, "\"");
            }
            if (r.source_fps != 0) {
                var buf: [32]u8 = undefined;
                const fps = try std.fmt.bufPrint(&buf, ",\"fps\":{d}", .{r.source_fps});
                try out.appendSlice(allocator, fps);
            }
            if (r.source_rate != 0) {
                var buf: [32]u8 = undefined;
                const rate = try std.fmt.bufPrint(&buf, ",\"sample_rate\":{d}", .{r.source_rate});
                try out.appendSlice(allocator, rate);
            }
            try out.appendSlice(allocator, "}");
        }
        try out.appendSlice(allocator, "]");
        return out.toOwnedSlice(allocator);
    }

    pub fn hasAudio(refs: []const Reference) bool {
        for (refs) |r| {
            if (r.kind == .audio or r.kind == .video_audio or r.soundtrack.len != 0) return true;
        }
        return false;
    }

    fn parseKind(text: []const u8) ?packing.ReferenceKind {
        if (std.mem.eql(u8, text, "image")) return .image;
        if (std.mem.eql(u8, text, "video")) return .video;
        if (std.mem.eql(u8, text, "audio")) return .audio;
        if (std.mem.eql(u8, text, "video_audio")) return .video_audio;
        return null;
    }
};

// --- core/memory.zig ---
pub const memory = struct {
    const std = @import("std");

    const zml = @import("zml");

    const config = @import("model.zig").config;
    const packing = @import("model.zig").packing;
    const policy = @import("model.zig").policy;
    const vae = @import("vae.zig").geom;

    pub const Plan = struct {
        activation_bytes: u64,
        streamed_block_bytes: u64,
        peak_bytes: u64,
        device_bytes: u64,
        score_bytes: u64,
        fa2_scratch_bytes: u64,
        adaln_table_bytes: u64,
        attention: policy.AttnKind,
        resident_blocks: u32,
        group_size: u32,
        tile_batch: u32,
        safe: bool,
        reason: []const u8,
    };

    /// Bytes reserved for one streamed transformer block.
    pub const streamed_block_bytes: u64 = 768 * 1024 * 1024;
    pub const safety_numer: u64 = 85;
    pub const safety_denom: u64 = 100;

    pub const Opts = struct {
        geo: pipeline.Geometry,
        layout: packing.Layout,
        hidden: i64,
        steps: u32,
        device_bytes: u64,
        tp: u32,
        heads: i64 = 56,
        head_dim: i64 = 128,
        layers: u32 = 50,
        dtype: zml.DataType = .bf16,
        target: zml.Target = .cpu,
        block_core_bytes: u64 = 0,
        devices: u32 = 1,
        tile_count: u32 = 0,
        tile_act_bytes: u64 = 0,
    };

    pub fn plan(
        geo: pipeline.Geometry,
        layout: packing.Layout,
        hidden: i64,
        steps: u32,
        device_bytes: u64,
        tp: u32,
    ) Plan {
        return planWith(.{
            .geo = geo,
            .layout = layout,
            .hidden = hidden,
            .steps = steps,
            .device_bytes = device_bytes,
            .tp = tp,
        });
    }

    pub fn planWith(opts: Opts) Plan {
        const seq: u64 = opts.layout.seqLen();
        const dtype_bytes = policy.dtypeBytes(opts.dtype);
        const spec = vae.official_visual;
        const tiles = if (opts.tile_count != 0) opts.tile_count else vae.tileCount(opts.geo.pixel_h, spec.tile_px, spec.tile_overlap_px, spec.spatial) *
            vae.tileCount(opts.geo.pixel_w, spec.tile_px, spec.tile_overlap_px, spec.spatial);
        const tile_lat = vae.decodeTileLatent(spec, opts.geo.latent_h, opts.geo.latent_w);
        const tile_t = vae.decodeClipTokens(spec, opts.geo.latent_t);
        const tile_seq = @as(u64, tile_t) * tile_lat.h * tile_lat.w + 5;
        const tile_act = if (opts.tile_act_bytes != 0) opts.tile_act_bytes else tile_seq * 2048 * dtype_bytes * 8;
        const decision = policy.decide(.{
            .target = opts.target,
            .seq = seq,
            .hidden = opts.hidden,
            .heads = opts.heads,
            .head_dim = opts.head_dim,
            .layers = opts.layers,
            .steps = opts.steps,
            .dtype = opts.dtype,
            .device_bytes = opts.device_bytes,
            .tp = opts.tp,
            .devices = opts.devices,
            .block_core_bytes = if (opts.block_core_bytes == 0)
                streamed_block_bytes / @max(1, opts.tp)
            else
                opts.block_core_bytes,
            .dtype_bytes = dtype_bytes,
            .tile_count = tiles,
            .tile_act_bytes = tile_act,
        });
        const host = (@as(u64, opts.geo.video_tokens) + opts.geo.audio_tokens) * 4 * 4;
        const block = if (opts.block_core_bytes == 0)
            streamed_block_bytes / @max(1, opts.tp)
        else
            opts.block_core_bytes;
        const attn_scratch = if (decision.attention == .cuda_fa2) decision.fa2_scratch_bytes else decision.score_bytes;
        const peak = decision.activation_bytes + host + block * 2 + attn_scratch + decision.adaln_table_bytes;
        const budget = if (opts.device_bytes == 0) std.math.maxInt(u64) else opts.device_bytes * safety_numer / safety_denom;
        const full_floor = config.full_canvas_min_device_bytes;
        const needs_full_floor = @min(opts.geo.pixel_w, opts.geo.pixel_h) > config.preview_short_side;

        var result: Plan = .{
            .activation_bytes = decision.activation_bytes,
            .streamed_block_bytes = block,
            .peak_bytes = peak,
            .device_bytes = opts.device_bytes,
            .score_bytes = decision.score_bytes,
            .fa2_scratch_bytes = decision.fa2_scratch_bytes,
            .adaln_table_bytes = decision.adaln_table_bytes,
            .attention = decision.attention,
            .resident_blocks = decision.resident_blocks,
            .group_size = decision.group_size,
            .tile_batch = decision.tile_batch,
            .safe = true,
            .reason = "ok",
        };
        if (needs_full_floor and opts.device_bytes != 0 and opts.device_bytes < full_floor) {
            result.safe = false;
            result.reason = "canvas above preview needs a measured 40 GiB-class device";
            return result;
        }
        if (opts.device_bytes != 0 and peak > budget) {
            result.safe = false;
            result.reason = "estimated peak exceeds 85% of device memory";
            return result;
        }
        return result;
    }
};

// --- conditioning/geom.zig ---
pub const cond_geom = struct {
    const std = @import("std");

    const config = @import("model.zig").config;

    pub const Size = config.Size;

    pub fn snapMultiple(value: u32, multiple: u32) u32 {
        if (value == 0) return multiple;
        return @max(multiple, @as(u32, @intFromFloat(@round(@as(f32, @floatFromInt(value)) / @as(f32, @floatFromInt(multiple))))) * multiple);
    }

    /// Official ref2va image geometry: short edge 2048, snap-32, upscale allowed, no area cap.
    pub fn refImageSize(src_w: u32, src_h: u32, canvas_w: u32, canvas_h: u32) error{InvalidAspect}!Size {
        _ = canvas_w;
        _ = canvas_h;
        if (src_w == 0 or src_h == 0) return error.InvalidAspect;
        const ratio = @as(f32, @floatFromInt(src_w)) / @as(f32, @floatFromInt(src_h));
        if (ratio < config.min_aspect or ratio > config.max_aspect) return error.InvalidAspect;
        const short = @min(src_w, src_h);
        const scale = @as(f32, @floatFromInt(config.reference_image_short_edge)) / @as(f32, @floatFromInt(short));
        const multiple = config.canvas_multiple;
        const w = @max(multiple, @as(u32, @intFromFloat(@round(@as(f32, @floatFromInt(src_w)) * scale / @as(f32, @floatFromInt(multiple))))) * multiple);
        const h = @max(multiple, @as(u32, @intFromFloat(@round(@as(f32, @floatFromInt(src_h)) * scale / @as(f32, @floatFromInt(multiple))))) * multiple);
        return .{ .w = w, .h = h };
    }

    /// Own-aspect canvas with 768-short-edge + area cap. Never upscale the source.
    pub fn videoCanvas(src_w: u32, src_h: u32) error{InvalidAspect}!Size {
        const adapted = try config.resolveCanvas(@floatFromInt(src_w), @floatFromInt(src_h), config.default_short_side, config.canvas_max_pixels);
        if (@as(u64, src_w) * src_h < @as(u64, adapted.w) * adapted.h) {
            return .{ .w = snapMultiple(src_w, config.canvas_multiple), .h = snapMultiple(src_h, config.canvas_multiple) };
        }
        return adapted;
    }

    pub fn fillVideoTimestamps(sample_count: u32, out: []f32) u32 {
        const n = @min(sample_count, @as(u32, @intCast(out.len)));
        var i: u32 = 0;
        while (i < n) : (i += 1) out[i] = @as(f32, @floatFromInt(i)) / 2.0;
        return n;
    }

    pub fn coverCropBox(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) struct { w: u32, h: u32, x: u32, y: u32 } {
        const scale = @max(
            @as(f32, @floatFromInt(dst_w)) / @as(f32, @floatFromInt(src_w)),
            @as(f32, @floatFromInt(dst_h)) / @as(f32, @floatFromInt(src_h)),
        );
        const rw = @max(dst_w, @as(u32, @intFromFloat(@round(@as(f32, @floatFromInt(src_w)) * scale))));
        const rh = @max(dst_h, @as(u32, @intFromFloat(@round(@as(f32, @floatFromInt(src_h)) * scale))));
        const x = @max(0, (rw - dst_w) / 2);
        const y = @max(0, (rh - dst_h) / 2);
        return .{ .w = rw, .h = rh, .x = x, .y = y };
    }

    fn sinc(x: f32) f32 {
        if (x == 0) return 1.0;
        const px = std.math.pi * x;
        return @sin(px) / px;
    }

    fn lanczos3(x: f32) f32 {
        if (x <= -3.0 or x >= 3.0) return 0;
        return sinc(x) * sinc(x / 3.0);
    }

    /// Torchvision Keys cubic, a=-0.75. Official Qwen2VL Fast processor.
    fn bicubicKeys(x: f32) f32 {
        const a = -0.75;
        const ax = @abs(x);
        if (ax <= 1.0) return ((a + 2.0) * ax - (a + 3.0)) * ax * ax + 1.0;
        if (ax < 2.0) return ((a * ax - 5.0 * a) * ax + 8.0 * a) * ax - 4.0 * a;
        return 0;
    }

    const ResizeKernel = enum { lanczos3, bicubic };

    fn kernelWeight(kernel: ResizeKernel, x: f32) f32 {
        return switch (kernel) {
            .lanczos3 => lanczos3(x),
            .bicubic => bicubicKeys(x),
        };
    }

    fn kernelSupport(kernel: ResizeKernel) f32 {
        return switch (kernel) {
            .lanczos3 => 3.0,
            .bicubic => 2.0,
        };
    }

    fn resize1d(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, dst_w: u32, horizontal: bool, kernel: ResizeKernel) ![]u8 {
        const out_w = if (horizontal) dst_w else src_w;
        const out_h = if (horizontal) src_h else dst_w;
        const out = try allocator.alloc(u8, @as(usize, out_w) * out_h * 3);
        const src_len: u32 = if (horizontal) src_w else src_h;
        const dst_len: u32 = dst_w;
        const scale = @as(f32, @floatFromInt(src_len)) / @as(f32, @floatFromInt(dst_len));
        const filterscale = @max(1.0, scale);
        const support = kernelSupport(kernel) * filterscale;

        var dy: u32 = 0;
        while (dy < out_h) : (dy += 1) {
            var dx: u32 = 0;
            while (dx < out_w) : (dx += 1) {
                const dst_i: f32 = @floatFromInt(if (horizontal) dx else dy);
                const center = (dst_i + 0.5) * @as(f32, @floatFromInt(src_len)) / @as(f32, @floatFromInt(dst_len));
                const xmin = @as(i32, @intFromFloat(@floor(center - support)));
                const xmax = @as(i32, @intFromFloat(@ceil(center + support)));
                var acc = [3]f32{ 0, 0, 0 };
                var wsum: f32 = 0;
                var xi = xmin;
                while (xi < xmax) : (xi += 1) {
                    const src_pos = std.math.clamp(xi, 0, @as(i32, @intCast(src_len - 1)));
                    const weight = kernelWeight(kernel, ((@as(f32, @floatFromInt(src_pos)) + 0.5) - center) / filterscale);
                    if (weight == 0) continue;
                    wsum += weight;
                    const sx: u32 = if (horizontal) @intCast(src_pos) else dx;
                    const sy: u32 = if (horizontal) dy else @intCast(src_pos);
                    const si = (@as(usize, sy) * src_w + sx) * 3;
                    inline for (0..3) |c| acc[c] += weight * @as(f32, @floatFromInt(src[si + c]));
                }
                const di = (@as(usize, dy) * out_w + dx) * 3;
                if (wsum == 0) wsum = 1;
                inline for (0..3) |c| {
                    out[di + c] = @intFromFloat(std.math.clamp(@round(acc[c] / wsum), 0, 255));
                }
            }
        }
        return out;
    }

    fn resizeKernel(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, dst_w: u32, dst_h: u32, kernel: ResizeKernel) ![]u8 {
        std.debug.assert(src.len == @as(usize, src_w) * src_h * 3);
        if (src_w == dst_w and src_h == dst_h) return allocator.dupe(u8, src);
        const mid = try resize1d(allocator, src, src_w, src_h, dst_w, true, kernel);
        defer allocator.free(mid);
        return resize1d(allocator, mid, dst_w, src_h, dst_h, false, kernel);
    }

    pub fn resizeLanczos(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) ![]u8 {
        return resizeKernel(allocator, src, src_w, src_h, dst_w, dst_h, .lanczos3);
    }

    pub fn resizeBicubic(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) ![]u8 {
        return resizeKernel(allocator, src, src_w, src_h, dst_w, dst_h, .bicubic);
    }

    pub fn cropRgb(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, x: u32, y: u32, dst_w: u32, dst_h: u32) ![]u8 {
        const out = try allocator.alloc(u8, @as(usize, dst_w) * dst_h * 3);
        var row: u32 = 0;
        while (row < dst_h) : (row += 1) {
            const sy = @min(src_h - 1, y + row);
            var col: u32 = 0;
            while (col < dst_w) : (col += 1) {
                const sx = @min(src_w - 1, x + col);
                const si = (@as(usize, sy) * src_w + sx) * 3;
                const di = (@as(usize, row) * dst_w + col) * 3;
                @memcpy(out[di..][0..3], src[si..][0..3]);
            }
        }
        return out;
    }

    pub fn stretchLanczos(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) ![]u8 {
        return resizeLanczos(allocator, src, src_w, src_h, dst_w, dst_h);
    }

    pub fn coverCropLanczos(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) ![]u8 {
        const box = coverCropBox(src_w, src_h, dst_w, dst_h);
        const resized = try resizeLanczos(allocator, src, src_w, src_h, box.w, box.h);
        defer allocator.free(resized);
        return cropRgb(allocator, resized, box.w, box.h, box.x, box.y, dst_w, dst_h);
    }

    /// 24 fps hold-resample: each source frame is held until the next slot.
    pub fn resampleFrameIndices(src_frames: u32, src_fps: f32, dst_fps: f32, allocator: std.mem.Allocator) ![]u32 {
        if (src_frames == 0) return error.EmptyVideo;
        if (src_fps <= 0 or dst_fps <= 0) return error.InvalidFps;
        if (src_fps == dst_fps) {
            const out = try allocator.alloc(u32, src_frames);
            for (out, 0..) |*d, i| d.* = @intCast(i);
            return out;
        }
        const scale = dst_fps / src_fps;
        const out_len_f = @floor(@as(f32, @floatFromInt(src_frames)) * scale + 0.5);
        const out_len: u32 = @intFromFloat(out_len_f);
        const out = try allocator.alloc(u32, out_len);
        var src: u32 = 0;
        var written: u32 = 0;
        while (src < src_frames) : (src += 1) {
            const slot: u32 = @intFromFloat(@floor(@as(f32, @floatFromInt(src)) * scale + 0.5));
            const next: u32 = if (src + 1 == src_frames)
                out_len
            else
                @intFromFloat(@floor(@as(f32, @floatFromInt(src + 1)) * scale + 0.5));
            const hold = if (next > slot) next - slot else 0;
            var h: u32 = 0;
            while (h < hold and written < out_len) : (h += 1) {
                out[written] = src;
                written += 1;
            }
        }
        if (written < out_len) {
            const last = if (src_frames == 0) 0 else src_frames - 1;
            while (written < out_len) : (written += 1) out[written] = last;
        }
        return out;
    }

    pub fn sampleVideoConditionFrames(frames: u32, fps: f32, sample_fps: f32, temporal_patch: u32) !struct { indices_len: u32, block_count: u32 } {
        if (frames == 0 or fps <= 0 or sample_fps <= 0) return error.EmptyVideo;
        const stride = fps / sample_fps;
        var count: u32 = 0;
        var last: i64 = -1;
        var cursor: f32 = 0;
        while (@round(cursor) < @as(f32, @floatFromInt(frames))) {
            const idx: i64 = @intFromFloat(@round(cursor));
            if (last < 0 or idx > last) {
                count += 1;
                last = idx;
            }
            cursor += stride;
        }
        if (count < temporal_patch) return error.VideoTooShort;
        const padded = count + (temporal_patch - (count % temporal_patch)) % temporal_patch;
        return .{ .indices_len = count, .block_count = padded / temporal_patch };
    }

    pub fn fillVideoConditionIndices(frames: u32, fps: f32, sample_fps: f32, out: []u32) u32 {
        const stride = fps / sample_fps;
        var n: u32 = 0;
        var last: i64 = -1;
        var cursor: f32 = 0;
        while (@round(cursor) < @as(f32, @floatFromInt(frames)) and n < out.len) {
            const idx: u32 = @intFromFloat(@round(cursor));
            if (last < 0 or @as(i64, idx) > last) {
                out[n] = @min(frames - 1, idx);
                n += 1;
                last = idx;
            }
            cursor += stride;
        }
        return n;
    }

    pub fn fillBlockTimestamps(sample_count: u32, sample_fps: f32, temporal_patch: u32, out: []f32) u32 {
        const padded = sample_count + (temporal_patch - (sample_count % temporal_patch)) % temporal_patch;
        const blocks = padded / temporal_patch;
        std.debug.assert(out.len >= blocks);
        var i: u32 = 0;
        while (i < blocks) : (i += 1) {
            const a = @as(f32, @floatFromInt(i * temporal_patch)) / sample_fps;
            const last_idx = @min(sample_count - 1, (i + 1) * temporal_patch - 1);
            const b = @as(f32, @floatFromInt(last_idx)) / sample_fps;
            out[i] = (a + b) / 2;
        }
        return blocks;
    }

    /// One decimal place, round half to even.
    pub fn formatSeconds1(value: f32, buf: []u8) []const u8 {
        const scaled = @as(f64, value) * 10.0;
        const whole = @floor(scaled);
        const frac = scaled - whole;
        var tenths: i64 = @intFromFloat(whole);
        if (frac > 0.5) {
            tenths += 1;
        } else if (frac == 0.5 and @mod(tenths, 2) != 0) {
            tenths += 1;
        }
        const ip = @divTrunc(tenths, 10);
        const frac_digit = @mod(tenths, 10);
        return std.fmt.bufPrint(buf, "{d}.{d}", .{ ip, if (frac_digit < 0) -frac_digit else frac_digit }) catch buf[0..0];
    }

    pub fn hopAlign(n: u32, hop: u32) u32 {
        if (hop == 0) return n;
        return n + (hop - (n % hop)) % hop;
    }

    pub fn applyRgb(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, indices: []const u32) ![]u8 {
        const plane = @as(usize, src_w) * src_h * 3;
        const out = try allocator.alloc(u8, indices.len * plane);
        for (indices, 0..) |src_i, i| {
            const si = @min(src_i, if (src.len == 0) 0 else @as(u32, @intCast(src.len / plane - 1)));
            @memcpy(out[i * plane ..][0..plane], src[si * plane ..][0..plane]);
        }
        return out;
    }

    pub fn truncateStereo(allocator: std.mem.Allocator, stereo: []const f32, max_samples: u32) ![]f32 {
        const have: u32 = @intCast(stereo.len / 2);
        const keep = @min(have, max_samples);
        const out = try allocator.alloc(f32, @as(usize, keep) * 2);
        @memcpy(out, stereo[0..out.len]);
        return out;
    }

    pub fn resampleLinear(allocator: std.mem.Allocator, stereo: []const f32, src_rate: u32, dst_rate: u32) ![]f32 {
        const src_n: u32 = @intCast(stereo.len / 2);
        if (src_rate == 0 or dst_rate == 0) return error.InvalidRate;
        if (src_rate == dst_rate) return allocator.dupe(f32, stereo);
        const dst_n: u32 = @intFromFloat(@round(@as(f64, src_n) * @as(f64, dst_rate) / @as(f64, src_rate)));
        const out = try allocator.alloc(f32, @as(usize, dst_n) * 2);
        if (src_n == 0 or dst_n == 0) {
            @memset(out, 0);
            return out;
        }
        if (dst_n == 1) {
            @memcpy(out[0..2], stereo[0..2]);
            return out;
        }
        var i: u32 = 0;
        while (i < dst_n) : (i += 1) {
            const src_pos = @as(f64, i) * @as(f64, src_n - 1) / @as(f64, dst_n -| 1);
            const lo: u32 = @intFromFloat(@floor(src_pos));
            const hi = @min(src_n - 1, lo + 1);
            const a: f32 = @floatCast(src_pos - @floor(src_pos));
            inline for (0..2) |c| {
                const a0 = stereo[@as(usize, lo) * 2 + c];
                const a1 = stereo[@as(usize, hi) * 2 + c];
                out[@as(usize, i) * 2 + c] = a0 * (1 - a) + a1 * a;
            }
        }
        return out;
    }

    pub fn monoToStereo(allocator: std.mem.Allocator, mono: []const f32) ![]f32 {
        const out = try allocator.alloc(f32, mono.len * 2);
        for (mono, 0..) |s, i| {
            out[i * 2] = s;
            out[i * 2 + 1] = s;
        }
        return out;
    }
};

// --- conditioning/presentation.zig ---
pub const presentation = struct {
    const std = @import("std");

    const config = @import("model.zig").config;
    const geom = @import("generate.zig").cond_geom;
    const packing = @import("model.zig").packing;
    const vision = @import("model.zig").vision;

    pub const VisionSpan = struct {
        start: u32,
        tokens: u32,
        grid_h: u32,
        grid_w: u32,
        temporal: u32,
    };

    pub const VisualSpec = struct {
        kind: packing.ReferenceKind,
        merged: u32,
        grid_h: u32,
        grid_w: u32,
        temporal: u32 = 1,
        timestamps: []const f32 = &.{},
        has_audio: bool = false,
    };

    pub const Assembled = struct {
        tokens: []u32,
        tags: []u8,
        spans: []VisionSpan,

        pub fn deinit(self: Assembled, allocator: std.mem.Allocator) void {
            allocator.free(self.tokens);
            allocator.free(self.tags);
            allocator.free(self.spans);
        }
    };

    const Builder = struct {
        allocator: std.mem.Allocator,
        tokens: std.ArrayList(u32),
        tags: std.ArrayList(u8),
        spans: std.ArrayList(VisionSpan),

        fn init(allocator: std.mem.Allocator) Builder {
            return .{
                .allocator = allocator,
                .tokens = .empty,
                .tags = .empty,
                .spans = .empty,
            };
        }

        fn deinit(self: *Builder) void {
            self.tokens.deinit(self.allocator);
            self.tags.deinit(self.allocator);
            self.spans.deinit(self.allocator);
        }

        fn emitText(self: *Builder, ids: []const u32) !void {
            try self.tokens.appendSlice(self.allocator, ids);
            try self.tags.appendNTimes(self.allocator, @intFromEnum(packing.Modality.text), ids.len);
        }

        fn emitVision(self: *Builder, pad: u32, count: u32, grid_h: u32, grid_w: u32, temporal: u32) !void {
            try self.tokens.append(self.allocator, vision.VISION_START);
            try self.tags.append(self.allocator, @intFromEnum(packing.Modality.video));
            const start: u32 = @intCast(self.tokens.items.len);
            var i: u32 = 0;
            while (i < count) : (i += 1) try self.tokens.append(self.allocator, pad);
            try self.tags.appendNTimes(self.allocator, @intFromEnum(packing.Modality.video), count);
            try self.tokens.append(self.allocator, vision.VISION_END);
            try self.tags.append(self.allocator, @intFromEnum(packing.Modality.video));
            try self.spans.append(self.allocator, .{
                .start = start,
                .tokens = count,
                .grid_h = grid_h,
                .grid_w = grid_w,
                .temporal = temporal,
            });
        }

        fn finish(self: *Builder) !Assembled {
            return .{
                .tokens = try self.tokens.toOwnedSlice(self.allocator),
                .tags = try self.tags.toOwnedSlice(self.allocator),
                .spans = try self.spans.toOwnedSlice(self.allocator),
            };
        }
    };

    fn encodeLabel(allocator: std.mem.Allocator, encode_text: anytype, comptime fmt: []const u8, args: anytype) ![]u32 {
        var buf: [64]u8 = undefined;
        const text = try std.fmt.bufPrint(&buf, fmt, args);
        return encode_text.encodeAlloc(allocator, text);
    }

    pub fn assembleT2va(allocator: std.mem.Allocator, encode_text: anytype, prompt: []const u8) !Assembled {
        var b = Builder.init(allocator);
        errdefer b.deinit();
        const ids = try encode_text.encodeAlloc(allocator, prompt);
        defer allocator.free(ids);
        try b.emitText(ids);
        return b.finish();
    }

    pub fn assembleFl2va(allocator: std.mem.Allocator, encode_text: anytype, visuals: []const VisualSpec, prompt: []const u8) !Assembled {
        var b = Builder.init(allocator);
        errdefer b.deinit();
        for (visuals, 0..) |vis, i| {
            const label = try encodeLabel(allocator, encode_text, "<Picture {d}>: ", .{i + 1});
            defer allocator.free(label);
            try b.emitText(label);
            try b.emitVision(vision.IMAGE_PAD, vis.merged, vis.grid_h, vis.grid_w, vis.temporal);
        }
        const ids = try encode_text.encodeAlloc(allocator, prompt);
        defer allocator.free(ids);
        try b.emitText(ids);
        return b.finish();
    }

    pub fn assembleRef2va(allocator: std.mem.Allocator, encode_text: anytype, visuals: []const VisualSpec, prompt: []const u8) !Assembled {
        var b = Builder.init(allocator);
        errdefer b.deinit();
        var n_pic: u32 = 0;
        var n_vid: u32 = 0;
        var n_aud: u32 = 0;
        for (visuals) |vis| {
            if (vis.has_audio or vis.kind == .audio or vis.kind == .video_audio) {
                n_aud += 1;
                const label = try encodeLabel(allocator, encode_text, "<Audio {d}>: ", .{n_aud});
                defer allocator.free(label);
                try b.emitText(label);
            }
            if (vis.kind == .image) {
                n_pic += 1;
                const label = try encodeLabel(allocator, encode_text, "<Picture {d}>: ", .{n_pic});
                defer allocator.free(label);
                try b.emitText(label);
                try b.emitVision(vision.IMAGE_PAD, vis.merged, vis.grid_h, vis.grid_w, vis.temporal);
            } else if (vis.kind == .video or vis.kind == .video_audio) {
                n_vid += 1;
                const label = try encodeLabel(allocator, encode_text, "<Video {d}>: ", .{n_vid});
                defer allocator.free(label);
                try b.emitText(label);
                for (vis.timestamps) |ts| {
                    var tbuf: [32]u8 = undefined;
                    const rendered = geom.formatSeconds1(ts, &tbuf);
                    var sbuf: [48]u8 = undefined;
                    const stamp = try std.fmt.bufPrint(&sbuf, "<{s} seconds>", .{rendered});
                    const ids = try encode_text.encodeAlloc(allocator, stamp);
                    defer allocator.free(ids);
                    try b.emitText(ids);
                    try b.emitVision(vision.VIDEO_PAD, vis.merged, vis.grid_h, vis.grid_w, 1);
                }
            }
        }
        const ids = try encode_text.encodeAlloc(allocator, prompt);
        defer allocator.free(ids);
        try b.emitText(ids);
        return b.finish();
    }

    pub fn assemble(
        allocator: std.mem.Allocator,
        encode_text: anytype,
        variant: config.Variant,
        visuals: []const VisualSpec,
        prompt: []const u8,
    ) !Assembled {
        return switch (variant) {
            .t2va => assembleT2va(allocator, encode_text, prompt),
            .fl2va => assembleFl2va(allocator, encode_text, visuals, prompt),
            .ref2va => assembleRef2va(allocator, encode_text, visuals, prompt),
        };
    }
};

// --- runtime/media.zig ---
pub const media = struct {
    const builtin = @import("builtin");
    const std = @import("std");

    const config = @import("model.zig").config;
    const geom = @import("generate.zig").cond_geom;
    const packing = @import("model.zig").packing;
    const vae = @import("vae.zig").geom;

    const log = std.log.scoped(.minimax_h3_media);

    pub const RgbImage = struct { w: u32, h: u32, rgb: []u8 };
    pub const Size = struct { w: u32, h: u32 };

    pub fn writePpm(
        io: std.Io,
        dir: std.Io.Dir,
        name: []const u8,
        width: u32,
        height: u32,
        rgb: []const u8,
    ) !void {
        std.debug.assert(rgb.len == @as(usize, width) * height * 3);
        const file = try dir.createFile(io, name, .{});
        defer file.close(io);
        var writer = file.writer(io, &.{});
        try writer.interface.print("P6\n{d} {d}\n255\n", .{ width, height });
        try writer.interface.writeAll(rgb);
    }

    pub fn writeWavS16(
        io: std.Io,
        dir: std.Io.Dir,
        name: []const u8,
        sample_rate: u32,
        channels: u16,
        pcm: []const i16,
    ) !void {
        const file = try dir.createFile(io, name, .{});
        defer file.close(io);
        var writer = file.writer(io, &.{});
        const data_bytes: u32 = @intCast(pcm.len * 2);
        const byte_rate = sample_rate * channels * 2;
        try writer.interface.writeAll("RIFF");
        try writer.interface.writeInt(u32, 36 + data_bytes, .little);
        try writer.interface.writeAll("WAVEfmt ");
        try writer.interface.writeInt(u32, 16, .little);
        try writer.interface.writeInt(u16, 1, .little);
        try writer.interface.writeInt(u16, channels, .little);
        try writer.interface.writeInt(u32, sample_rate, .little);
        try writer.interface.writeInt(u32, byte_rate, .little);
        try writer.interface.writeInt(u16, channels * 2, .little);
        try writer.interface.writeInt(u16, 16, .little);
        try writer.interface.writeAll("data");
        try writer.interface.writeInt(u32, data_bytes, .little);
        try writer.interface.writeAll(std.mem.sliceAsBytes(pcm));
    }

    pub fn rgbU8FromNchw(allocator: std.mem.Allocator, nchw: []const f32, frames: u32, height: u32, width: u32) ![]u8 {
        const plane = @as(usize, frames) * height * width;
        std.debug.assert(nchw.len >= plane * 3);
        const out = try allocator.alloc(u8, plane * 3);
        var i: usize = 0;
        while (i < plane) : (i += 1) {
            const r = std.math.clamp(nchw[i], 0, 1);
            const g = std.math.clamp(nchw[plane + i], 0, 1);
            const b = std.math.clamp(nchw[2 * plane + i], 0, 1);
            out[i * 3 + 0] = @intFromFloat(@round(r * 255.0));
            out[i * 3 + 1] = @intFromFloat(@round(g * 255.0));
            out[i * 3 + 2] = @intFromFloat(@round(b * 255.0));
        }
        return out;
    }

    pub const Output = struct {
        dir: []const u8,
        mp4_name: []const u8,

        pub fn parse(path: []const u8) Output {
            if (path.len == 0) return .{ .dir = "output", .mp4_name = "output.mp4" };
            if (path.len >= 4 and std.ascii.eqlIgnoreCase(path[path.len - 4 ..], ".mp4")) {
                return .{
                    .dir = std.fs.path.dirname(path) orelse ".",
                    .mp4_name = std.fs.path.basename(path),
                };
            }
            return .{ .dir = path, .mp4_name = "output.mp4" };
        }

        pub fn isCwd(self: Output) bool {
            return self.dir.len == 0 or std.mem.eql(u8, self.dir, ".");
        }
    };

    pub fn writeFrameSequence(
        allocator: std.mem.Allocator,
        io: std.Io,
        dir: std.Io.Dir,
        nchw: []const f32,
        frames: u32,
        height: u32,
        width: u32,
    ) !void {
        const rgb = try rgbU8FromNchw(allocator, nchw, frames, height, width);
        defer allocator.free(rgb);
        const stride = @as(usize, width) * height * 3;
        var f: u32 = 0;
        while (f < frames) : (f += 1) {
            var name_buf: [32]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "frame_{d:0>4}.ppm", .{f});
            try writePpm(io, dir, name, width, height, rgb[f * stride ..][0..stride]);
        }
    }

    pub fn f32ToS16(allocator: std.mem.Allocator, samples: []const f32) ![]i16 {
        const out = try allocator.alloc(i16, samples.len);
        for (samples, out) |s, *d| {
            const v = std.math.clamp(s, -1.0, 1.0);
            d.* = @intFromFloat(@round(v * 32767.0));
        }
        return out;
    }

    pub fn interleaveStereo(allocator: std.mem.Allocator, left: []const f32, right: []const f32) ![]f32 {
        std.debug.assert(left.len == right.len);
        const out = try allocator.alloc(f32, left.len * 2);
        for (left, right, 0..) |l, r, i| {
            out[i * 2] = l;
            out[i * 2 + 1] = r;
        }
        return out;
    }

    const ffmpeg_bin = "ffmpeg";

    var tmp_seq: u32 = 0;

    fn tmpId() u64 {
        tmp_seq += 1;
        return (@as(u64, @intFromPtr(&tmp_seq)) << 16) ^ tmp_seq;
    }

    fn envDir(name: [:0]const u8) ?[]const u8 {
        const raw = std.c.getenv(name) orelse return null;
        const path = std.mem.span(raw);
        return if (path.len == 0) null else path;
    }

    fn tempRoot() []const u8 {
        if (envDir("TMPDIR")) |p| return p;
        if (envDir("TEMP")) |p| return p;
        if (envDir("TMP")) |p| return p;
        return switch (builtin.os.tag) {
            .windows => "C:\\Windows\\Temp",
            else => "/tmp",
        };
    }

    pub const Scratch = struct {
        path: []u8,

        pub fn init(allocator: std.mem.Allocator) !Scratch {
            var threaded: std.Io.Threaded = .init_single_threaded;
            const io = threaded.io();
            const name = try std.fmt.allocPrint(allocator, "h3_{x}", .{tmpId()});
            defer allocator.free(name);
            const path = try std.fs.path.join(allocator, &.{ tempRoot(), name });
            errdefer allocator.free(path);
            try std.Io.Dir.cwd().createDirPath(io, path);
            return .{ .path = path };
        }

        pub fn join(self: Scratch, allocator: std.mem.Allocator, file: []const u8) ![]u8 {
            return std.fs.path.join(allocator, &.{ self.path, file });
        }

        pub fn deinit(self: *Scratch, allocator: std.mem.Allocator) void {
            var threaded: std.Io.Threaded = .init_single_threaded;
            const io = threaded.io();
            if (std.fs.path.isAbsolute(self.path)) {
                if (std.fs.path.dirname(self.path)) |parent_path| {
                    var parent = std.Io.Dir.openDirAbsolute(io, parent_path, .{}) catch {
                        allocator.free(self.path);
                        self.* = undefined;
                        return;
                    };
                    defer parent.close(io);
                    parent.deleteTree(io, std.fs.path.basename(self.path)) catch {};
                }
            } else {
                std.Io.Dir.cwd().deleteTree(io, self.path) catch {};
            }
            allocator.free(self.path);
            self.* = undefined;
        }
    };

    fn runFfmpeg(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8) !std.process.RunResult {
        return std.process.run(allocator, io, .{
            .argv = argv,
            .stdout_limit = .limited(4096),
            .stderr_limit = .limited(16 * 1024),
        });
    }

    pub fn muxMp4(
        allocator: std.mem.Allocator,
        io: std.Io,
        frames_dir: []const u8,
        audio_path: []const u8,
        mp4_path: []const u8,
    ) !bool {
        const frame_in = try std.fs.path.join(allocator, &.{ frames_dir, "frame_%04d.ppm" });
        defer allocator.free(frame_in);
        const result = runFfmpeg(allocator, io, &.{
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            "24",
            "-i",
            frame_in,
            "-i",
            audio_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            mp4_path,
        }) catch |err| {
            log.warn("ffmpeg {s}: {s}", .{ ffmpeg_bin, @errorName(err) });
            return false;
        };
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        switch (result.term) {
            .exited => |code| if (code == 0) {
                log.info("muxed {s} with {s}", .{ mp4_path, ffmpeg_bin });
                return true;
            } else {
                log.warn("{s} exited {d}: {s}", .{ ffmpeg_bin, code, result.stderr });
            },
            else => log.warn("{s} did not exit cleanly", .{ffmpeg_bin}),
        }
        return false;
    }

    pub fn openPath(io: std.Io, path: []const u8) !std.Io.Dir {
        if (std.fs.path.isAbsolute(path)) return std.Io.Dir.openDirAbsolute(io, path, .{});
        return std.Io.Dir.cwd().openDir(io, path, .{});
    }

    pub fn writeGeneratedVideo(
        allocator: std.mem.Allocator,
        io: std.Io,
        dest_dir: std.Io.Dir,
        dest_path: []const u8,
        mp4_name: []const u8,
        nchw: []const f32,
        frames: u32,
        height: u32,
        width: u32,
        pcm: []const i16,
        sample_rate: u32,
    ) !bool {
        var scratch = try Scratch.init(allocator);
        defer scratch.deinit(allocator);
        var scratch_dir = try openPath(io, scratch.path);
        defer scratch_dir.close(io);
        try writeFrameSequence(allocator, io, scratch_dir, nchw, frames, height, width);
        try writeWavS16(io, scratch_dir, "audio.wav", sample_rate, 2, pcm);
        const audio_in = try scratch.join(allocator, "audio.wav");
        defer allocator.free(audio_in);
        const mp4 = try std.fs.path.join(allocator, &.{ dest_path, mp4_name });
        defer allocator.free(mp4);
        if (try muxMp4(allocator, io, scratch.path, audio_in, mp4)) return true;

        const frames_path = try std.fs.path.join(allocator, &.{ dest_path, "frames" });
        defer allocator.free(frames_path);
        try std.Io.Dir.cwd().createDirPath(io, frames_path);
        var fallback = try openPath(io, frames_path);
        defer fallback.close(io);
        try writeFrameSequence(allocator, io, fallback, nchw, frames, height, width);
        try writeWavS16(io, dest_dir, "audio.wav", sample_rate, 2, pcm);
        return false;
    }

    pub fn readPpmRgb(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !RgbImage {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        defer allocator.free(bytes);
        const header = try parsePpmHeader(bytes);
        const need = @as(usize, header.w) * header.h * 3;
        if (bytes.len < header.data_off + need) return error.BadPpm;
        return .{ .w = header.w, .h = header.h, .rgb = try allocator.dupe(u8, bytes[header.data_off..][0..need]) };
    }

    pub fn resizeRgb(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) ![]u8 {
        return geom.resizeLanczos(allocator, src, src_w, src_h, dst_w, dst_h);
    }

    pub fn resizeRgbBicubic(allocator: std.mem.Allocator, src: []const u8, src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) ![]u8 {
        return geom.resizeBicubic(allocator, src, src_w, src_h, dst_w, dst_h);
    }

    pub fn ppmSize(io: std.Io, path: []const u8) !Size {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        var buf: [256]u8 = undefined;
        var reader = file.reader(io, &.{});
        const n = try reader.interface.readSliceShort(&buf);
        const img = try parsePpmHeader(buf[0..n]);
        return .{ .w = img.w, .h = img.h };
    }

    fn parsePpmHeader(bytes: []const u8) !struct { w: u32, h: u32, data_off: usize } {
        var rest = bytes;
        const magic_end = std.mem.indexOfScalar(u8, rest, '\n') orelse return error.BadPpm;
        if (!std.mem.eql(u8, std.mem.trim(u8, rest[0..magic_end], " \r"), "P6")) return error.UnsupportedImage;
        rest = rest[magic_end + 1 ..];
        var w: usize = 0;
        var h: usize = 0;
        var maxv: usize = 0;
        while (maxv == 0) {
            const line_end = std.mem.indexOfScalar(u8, rest, '\n') orelse return error.BadPpm;
            const line = std.mem.trim(u8, rest[0..line_end], " \r");
            rest = rest[line_end + 1 ..];
            if (line.len == 0 or line[0] == '#') continue;
            var it = std.mem.tokenizeScalar(u8, line, ' ');
            if (w == 0) w = try std.fmt.parseInt(usize, it.next() orelse return error.BadPpm, 10);
            if (h == 0) h = try std.fmt.parseInt(usize, it.next() orelse return error.BadPpm, 10);
            if (it.next()) |mv| maxv = try std.fmt.parseInt(usize, mv, 10);
        }
        if (maxv != 255) return error.UnsupportedPpmDepth;
        return .{ .w = @intCast(w), .h = @intCast(h), .data_off = bytes.len - rest.len };
    }

    pub fn imageSize(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Size {
        if (ppmSize(io, path)) |s| return s else |_| {}
        const img = try loadRgbRaw(allocator, io, path);
        defer allocator.free(img.rgb);
        return .{ .w = img.w, .h = img.h };
    }

    pub fn wavSampleCount(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !u32 {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024));
        defer allocator.free(bytes);
        const info = try parseWavHeader(bytes);
        return info.samples;
    }

    pub fn loadRgbRaw(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !RgbImage {
        if (readPpmRgb(allocator, io, path)) |img| return img else |_| {}
        var scratch = try Scratch.init(allocator);
        defer scratch.deinit(allocator);
        const tmp_name = try scratch.join(allocator, "in.ppm");
        defer allocator.free(tmp_name);
        const result = runFfmpeg(allocator, io, &.{
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-frames:v", "1", tmp_name,
        }) catch return error.FfmpegMissing;
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        switch (result.term) {
            .exited => |code| if (code != 0) return error.ImageLoadFailed,
            else => return error.ImageLoadFailed,
        }
        return readPpmRgb(allocator, io, tmp_name);
    }

    pub fn loadRgb(
        allocator: std.mem.Allocator,
        io: std.Io,
        path: []const u8,
        dst_w: u32,
        dst_h: u32,
    ) ![]u8 {
        const raw = try loadRgbRaw(allocator, io, path);
        defer allocator.free(raw.rgb);
        if (raw.w == dst_w and raw.h == dst_h) return allocator.dupe(u8, raw.rgb);
        return resizeRgb(allocator, raw.rgb, raw.w, raw.h, dst_w, dst_h);
    }

    pub fn loadRgbCover(
        allocator: std.mem.Allocator,
        io: std.Io,
        path: []const u8,
        dst_w: u32,
        dst_h: u32,
    ) ![]u8 {
        const raw = try loadRgbRaw(allocator, io, path);
        defer allocator.free(raw.rgb);
        return geom.coverCropLanczos(allocator, raw.rgb, raw.w, raw.h, dst_w, dst_h);
    }

    pub const VideoClip = struct {
        rgb: []u8,
        frames: u32,
        w: u32,
        h: u32,
        fps: f32,
        has_audio: bool = false,
    };

    pub const VideoMeta = struct { w: u32, h: u32, fps: f32, has_audio: bool };

    pub fn parseFfmpegProbe(text: []const u8) !VideoMeta {
        const video = std.mem.indexOf(u8, text, "Video:") orelse return error.VideoLoadFailed;
        const rest = text[video..];
        const size = findWxH(rest) orelse return error.VideoLoadFailed;
        const fps = findFps(rest) orelse return error.VideoLoadFailed;
        return .{
            .w = size.w,
            .h = size.h,
            .fps = fps,
            .has_audio = std.mem.indexOf(u8, text, "Audio:") != null,
        };
    }

    fn findWxH(text: []const u8) ?struct { w: u32, h: u32 } {
        var i: usize = 0;
        while (i + 3 < text.len) : (i += 1) {
            if (!std.ascii.isDigit(text[i])) continue;
            const x = std.mem.indexOfScalarPos(u8, text, i + 1, 'x') orelse return null;
            if (x == i) continue;
            var end = x + 1;
            while (end < text.len and std.ascii.isDigit(text[end])) end += 1;
            if (end == x + 1) continue;
            const w = std.fmt.parseInt(u32, text[i..x], 10) catch continue;
            const h = std.fmt.parseInt(u32, text[x + 1 .. end], 10) catch continue;
            if (w > 0 and h > 0) return .{ .w = w, .h = h };
        }
        return null;
    }

    fn findFps(text: []const u8) ?f32 {
        const needle = " fps";
        if (std.mem.indexOf(u8, text, needle)) |at| {
            var start = at;
            while (start > 0 and (std.ascii.isDigit(text[start - 1]) or text[start - 1] == '.')) start -= 1;
            if (start < at) {
                if (std.fmt.parseFloat(f32, text[start..at])) |v| {
                    if (v > 0) return v;
                } else |_| {}
            }
        }
        if (std.mem.indexOf(u8, text, " tbr")) |at| {
            var start = at;
            while (start > 0 and (std.ascii.isDigit(text[start - 1]) or text[start - 1] == '.')) start -= 1;
            if (start < at) {
                if (std.fmt.parseFloat(f32, text[start..at])) |v| {
                    if (v > 0) return v;
                } else |_| {}
            }
        }
        return null;
    }

    pub fn probeVideo(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !VideoMeta {
        const result = runFfmpeg(allocator, io, &.{
            ffmpeg_bin, "-hide_banner", "-i", path,
        }) catch return error.FfmpegMissing;
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        return parseFfmpegProbe(result.stderr);
    }

    fn parseRate(text: []const u8) ?f32 {
        const trimmed = std.mem.trim(u8, text, " \r\n");
        if (std.mem.indexOfScalar(u8, trimmed, '/')) |slash| {
            const num = std.fmt.parseFloat(f32, trimmed[0..slash]) catch return null;
            const den = std.fmt.parseFloat(f32, trimmed[slash + 1 ..]) catch return null;
            if (den == 0) return null;
            return num / den;
        }
        return std.fmt.parseFloat(f32, trimmed) catch null;
    }

    pub fn loadVideoNative(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !VideoClip {
        const meta = try probeVideo(allocator, io, path);
        var scratch = try Scratch.init(allocator);
        defer scratch.deinit(allocator);
        const tmp_pat = try scratch.join(allocator, "f_%04d.ppm");
        defer allocator.free(tmp_pat);
        const result = runFfmpeg(allocator, io, &.{
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-vsync", "0", tmp_pat,
        }) catch return error.FfmpegMissing;
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        switch (result.term) {
            .exited => |code| if (code != 0) return error.VideoLoadFailed,
            else => return error.VideoLoadFailed,
        }

        var frames_list: std.ArrayList([]u8) = .empty;
        defer {
            for (frames_list.items) |f| allocator.free(f);
            frames_list.deinit(allocator);
        }
        var loaded: u32 = 0;
        var fw: u32 = meta.w;
        var fh: u32 = meta.h;
        while (loaded < 4096) : (loaded += 1) {
            var name_buf: [16]u8 = undefined;
            const frame_name = try std.fmt.bufPrint(&name_buf, "f_{d:0>4}.ppm", .{loaded + 1});
            const name = try scratch.join(allocator, frame_name);
            defer allocator.free(name);
            const img = readPpmRgb(allocator, io, name) catch break;
            fw = img.w;
            fh = img.h;
            try frames_list.append(allocator, img.rgb);
        }
        if (frames_list.items.len == 0) return error.VideoLoadFailed;
        const plane = @as(usize, fw) * fh * 3;
        const out = try allocator.alloc(u8, frames_list.items.len * plane);
        for (frames_list.items, 0..) |frame, i| {
            if (frame.len != plane) {
                const resized = try resizeRgb(allocator, frame, fw, fh, fw, fh);
                defer allocator.free(resized);
                @memcpy(out[i * plane ..][0..plane], resized[0..plane]);
            } else {
                @memcpy(out[i * plane ..][0..plane], frame);
            }
        }
        return .{
            .rgb = out,
            .frames = @intCast(frames_list.items.len),
            .w = fw,
            .h = fh,
            .fps = meta.fps,
            .has_audio = meta.has_audio,
        };
    }

    pub fn loadVideoRgb(
        allocator: std.mem.Allocator,
        io: std.Io,
        path: []const u8,
        dst_w: u32,
        dst_h: u32,
        frames: u32,
    ) !struct { rgb: []u8, frames: u32, w: u32, h: u32 } {
        const clip = try loadVideoNative(allocator, io, path);
        defer allocator.free(clip.rgb);
        const indices = try geom.resampleFrameIndices(clip.frames, clip.fps, config.video_fps, allocator);
        defer allocator.free(indices);
        const keep = @min(frames, @as(u32, @intCast(indices.len)));
        const plane = @as(usize, dst_w) * dst_h * 3;
        const src_plane = @as(usize, clip.w) * clip.h * 3;
        const out = try allocator.alloc(u8, keep * plane);
        errdefer allocator.free(out);
        var i: u32 = 0;
        while (i < keep) : (i += 1) {
            const src_i = indices[i];
            const src = clip.rgb[src_i * src_plane ..][0..src_plane];
            const rgb = if (clip.w == dst_w and clip.h == dst_h)
                try allocator.dupe(u8, src)
            else
                try resizeRgb(allocator, src, clip.w, clip.h, dst_w, dst_h);
            defer allocator.free(rgb);
            @memcpy(out[i * plane ..][0..plane], rgb);
        }
        return .{ .rgb = out, .frames = keep, .w = dst_w, .h = dst_h };
    }

    pub fn rgbVideoToNchwImagenet(allocator: std.mem.Allocator, rgb: []const u8, frames: u32, height: u32, width: u32) ![]f32 {
        const plane = @as(usize, height) * width;
        const out = try allocator.alloc(f32, 3 * frames * plane);
        var f: u32 = 0;
        while (f < frames) : (f += 1) {
            var i: usize = 0;
            while (i < plane) : (i += 1) {
                inline for (0..3) |c| {
                    const v = @as(f32, @floatFromInt(rgb[(f * plane + i) * 3 + c])) / 255.0;
                    out[(c * frames + f) * plane + i] = (v - vae.imagenet_mean[c]) / vae.imagenet_std[c];
                }
            }
        }
        return out;
    }

    pub const Pcm = struct { stereo: []f32, rate: u32 };

    pub fn loadWavNative(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Pcm {
        if (readWavAny(allocator, io, path)) |pcm| return pcm else |_| {}
        var scratch = try Scratch.init(allocator);
        defer scratch.deinit(allocator);
        const tmp = try scratch.join(allocator, "native.wav");
        defer allocator.free(tmp);
        const result = runFfmpeg(allocator, io, &.{
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-ac", "2", tmp,
        }) catch return error.FfmpegMissing;
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        switch (result.term) {
            .exited => |code| if (code != 0) return error.AudioLoadFailed,
            else => return error.AudioLoadFailed,
        }
        return readWavAny(allocator, io, tmp);
    }

    fn readWavAny(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Pcm {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        defer allocator.free(bytes);
        const info = try parseWavHeader(bytes);
        const stereo = try decodeWavStereo(allocator, bytes, info);
        return .{ .stereo = stereo, .rate = info.rate };
    }

    pub fn loadAudioOfficial(allocator: std.mem.Allocator, io: std.Io, path: []const u8, duration_s: f32, dst_rate: u32) ![]f32 {
        const native = try loadWavNative(allocator, io, path);
        defer allocator.free(native.stereo);
        const max_pcm: u32 = @intFromFloat(@round(duration_s * @as(f32, @floatFromInt(native.rate))));
        const truncated = try geom.truncateStereo(allocator, native.stereo, max_pcm);
        defer allocator.free(truncated);
        return geom.resampleLinear(allocator, truncated, native.rate, dst_rate);
    }

    pub fn loadWavStereo(allocator: std.mem.Allocator, io: std.Io, path: []const u8, sample_rate: u32) ![]f32 {
        if (readWavStereo(allocator, io, path, sample_rate)) |pcm| return pcm else |_| {}
        var scratch = try Scratch.init(allocator);
        defer scratch.deinit(allocator);
        const tmp = try scratch.join(allocator, "in.wav");
        defer allocator.free(tmp);
        const rate = try std.fmt.allocPrint(allocator, "{d}", .{sample_rate});
        defer allocator.free(rate);
        const result = runFfmpeg(allocator, io, &.{
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-ac", "2", "-ar", rate, tmp,
        }) catch return error.FfmpegMissing;
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        switch (result.term) {
            .exited => |code| if (code != 0) return error.AudioLoadFailed,
            else => return error.AudioLoadFailed,
        }
        return readWavStereo(allocator, io, tmp, sample_rate);
    }

    const WavInfo = struct { samples: u32, ch: u16, rate: u32, bits: u16, data_off: usize };

    pub fn parseWavHeader(bytes: []const u8) !WavInfo {
        if (bytes.len < 44) return error.BadWav;
        if (!std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE")) return error.BadWav;
        var off: usize = 12;
        var data_off: usize = 0;
        var data_len: usize = 0;
        var ch: u16 = 0;
        var rate: u32 = 0;
        var bits: u16 = 0;
        while (off + 8 <= bytes.len) {
            const id = bytes[off..][0..4];
            const n = std.mem.readInt(u32, bytes[off + 4 ..][0..4], .little);
            off += 8;
            if (std.mem.eql(u8, id, "fmt ")) {
                if (n < 16 or off + 16 > bytes.len) return error.BadWav;
                ch = std.mem.readInt(u16, bytes[off + 2 ..][0..2], .little);
                rate = std.mem.readInt(u32, bytes[off + 4 ..][0..4], .little);
                bits = std.mem.readInt(u16, bytes[off + 14 ..][0..2], .little);
            } else if (std.mem.eql(u8, id, "data")) {
                data_off = off;
                data_len = n;
                break;
            }
            off += n;
        }
        if (data_off == 0 or ch == 0 or bits == 0) return error.BadWav;
        return .{
            .samples = @intCast(data_len / (ch * (bits / 8))),
            .ch = ch,
            .rate = rate,
            .bits = bits,
            .data_off = data_off,
        };
    }

    fn decodeWavStereo(allocator: std.mem.Allocator, bytes: []const u8, info: WavInfo) ![]f32 {
        const out = try allocator.alloc(f32, @as(usize, info.samples) * 2);
        var i: usize = 0;
        while (i < info.samples) : (i += 1) {
            var c: u16 = 0;
            while (c < 2) : (c += 1) {
                const src_c = if (c < info.ch) c else 0;
                const idx = info.data_off + (i * info.ch + src_c) * (info.bits / 8);
                const s: f32 = switch (info.bits) {
                    16 => @as(f32, @floatFromInt(std.mem.readInt(i16, bytes[idx..][0..2], .little))) / 32768.0,
                    32 => std.mem.bytesAsValue(f32, bytes[idx..][0..4]).*,
                    else => return error.UnsupportedWav,
                };
                out[i * 2 + c] = s;
            }
        }
        return out;
    }

    fn readWavStereo(allocator: std.mem.Allocator, io: std.Io, path: []const u8, sample_rate: u32) ![]f32 {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        defer allocator.free(bytes);
        const info = try parseWavHeader(bytes);
        if (info.rate != sample_rate) return error.WavRateMismatch;
        return decodeWavStereo(allocator, bytes, info);
    }

    pub fn refsContainAudio(refs: []const u8) bool {
        var it = std.mem.splitScalar(u8, refs, ',');
        while (it.next()) |part| {
            const path = std.mem.trim(u8, part, " \t");
            if (path.len != 0 and guessKind(path) == .audio) return true;
        }
        return false;
    }

    pub fn guessKind(path: []const u8) packing.ReferenceKind {
        const ext = std.fs.path.extension(path);
        if (std.ascii.eqlIgnoreCase(ext, ".wav") or std.ascii.eqlIgnoreCase(ext, ".mp3") or std.ascii.eqlIgnoreCase(ext, ".flac") or std.ascii.eqlIgnoreCase(ext, ".m4a"))
            return .audio;
        if (std.ascii.eqlIgnoreCase(ext, ".mp4") or std.ascii.eqlIgnoreCase(ext, ".mov") or std.ascii.eqlIgnoreCase(ext, ".mkv") or std.ascii.eqlIgnoreCase(ext, ".webm"))
            return .video;
        return .image;
    }

    pub fn rgbToNchwImagenet(allocator: std.mem.Allocator, rgb: []const u8, height: u32, width: u32) ![]f32 {
        return rgbVideoToNchwImagenet(allocator, rgb, 1, height, width);
    }
};

// --- runtime/repo.zig ---
pub const ckpt = struct {
    const std = @import("std");

    const zml = @import("zml");

    const audio_vae = @import("vae.zig").audio;
    const config = @import("model.zig").config;
    const dit = @import("model.zig").dit;
    const encoder = @import("model.zig").encoder;
    const visual_vae = @import("vae.zig").visual;

    const log = std.log.scoped(.minimax_h3);

    pub const Open = struct {
        model: []const u8,
        dit: []const u8 = "",
    };

    pub const FileSource = struct {
        dir: std.Io.Dir,
        dir_owned: bool,
        file: ?[]u8 = null,

        fn deinit(self: *FileSource, allocator: std.mem.Allocator, io: std.Io) void {
            if (self.file) |name| allocator.free(name);
            if (self.dir_owned) self.dir.close(io);
            self.file = null;
            self.dir_owned = false;
        }
    };

    const Search = struct {
        repo: std.Io.Dir,
        extra: ?std.Io.Dir,
        task: std.Io.Dir,
    };

    pub const Bundle = struct {
        task: std.Io.Dir,
        task_owned: bool,
        dit_src: FileSource,
        enc_src: FileSource,
        visual_src: FileSource,
        audio_src: FileSource,
        visual_source: ?std.Io.Dir,

        dit_registry: *zml.safetensors.TensorRegistry,
        dit_store: zml.io.TensorStore,
        enc_registry: *zml.safetensors.TensorRegistry,
        enc_store: zml.io.TensorStore,
        visual_registry: *zml.safetensors.TensorRegistry,
        visual_store: zml.io.TensorStore,
        audio_registry: *zml.safetensors.TensorRegistry,
        audio_store: zml.io.TensorStore,

        dit: dit.LoadedModel,
        enc: encoder.LoadedModel,
        visual: visual_vae.LoadedModel,
        audio: audio_vae.LoadedModel,

        pub fn open(
            allocator: std.mem.Allocator,
            io: std.Io,
            repo: std.Io.Dir,
            variant: config.Variant,
            shardings: sharding.Shardings,
            opts: Open,
        ) !Bundle {
            const task = try config.openTaskDir(io, repo, variant);
            errdefer if (task.owned) task.dir.close(io);

            var extra = openBundleRoot(io, opts.model);
            defer if (extra) |*dir| dir.close(io);
            const search: Search = .{ .repo = repo, .extra = extra, .task = task.dir };

            var dit_src = try resolveDit(allocator, io, search, variant, opts);
            errdefer dit_src.deinit(allocator, io);
            var enc_src = try resolveComponent(allocator, io, search, .{
                .official = "text_encoder",
                .scan = "text_encoders",
                .needles = &.{},
                .missing = error.EncoderMissing,
            });
            errdefer enc_src.deinit(allocator, io);
            var visual_src = try resolveComponent(allocator, io, search, .{
                .official = "video_vae",
                .aliases = &.{ "visual_vae", "vae" },
                .scan = "vae",
                .needles = &.{ "video", "vae" },
                .missing = error.VaeMissing,
            });
            errdefer visual_src.deinit(allocator, io);
            var audio_src = try resolveComponent(allocator, io, search, .{
                .official = "audio_vae",
                .scan = "vae",
                .needles = &.{ "audio", "vae" },
                .missing = error.VaeMissing,
            });
            errdefer audio_src.deinit(allocator, io);

            var visual_source = if (visual_src.file == null) openOptionalDir(io, visual_src.dir, "source") else null;
            errdefer if (visual_source) |*dir| dir.close(io);
            const visual_weights = visual_source orelse visual_src.dir;

            const dit_registry = try allocator.create(zml.safetensors.TensorRegistry);
            errdefer allocator.destroy(dit_registry);
            dit_registry.* = try openRegistry(allocator, io, dit_src);
            errdefer dit_registry.deinit();
            try refuseUnsupported(dit_registry, allocator);
            var dit_store: zml.io.TensorStore = .fromRegistry(allocator, dit_registry);
            errdefer dit_store.deinit();

            const enc_registry = try allocator.create(zml.safetensors.TensorRegistry);
            errdefer allocator.destroy(enc_registry);
            enc_registry.* = try openRegistry(allocator, io, enc_src);
            errdefer enc_registry.deinit();
            var enc_store: zml.io.TensorStore = .fromRegistry(allocator, enc_registry);
            errdefer enc_store.deinit();

            const visual_registry = try allocator.create(zml.safetensors.TensorRegistry);
            errdefer allocator.destroy(visual_registry);
            visual_registry.* = try openRegistry(allocator, io, .{
                .dir = visual_weights,
                .dir_owned = false,
                .file = visual_src.file,
            });
            errdefer visual_registry.deinit();
            var visual_store: zml.io.TensorStore = .fromRegistry(allocator, visual_registry);
            errdefer visual_store.deinit();

            const audio_registry = try allocator.create(zml.safetensors.TensorRegistry);
            errdefer allocator.destroy(audio_registry);
            audio_registry.* = try openRegistry(allocator, io, audio_src);
            errdefer audio_registry.deinit();
            var audio_store: zml.io.TensorStore = .fromRegistry(allocator, audio_registry);
            errdefer audio_store.deinit();
            if (!visual_vae.ready(visual_store.view()) or !audio_vae.decodeReady(audio_store.view()))
                return error.VaeSchemaMismatch;

            var loaded_dit = try dit.LoadedModel.init(allocator, io, dit_src.dir, dit_store.view());
            errdefer loaded_dit.deinit(allocator);
            var loaded_enc = try encoder.LoadedModel.init(allocator, io, enc_src.dir, enc_store.view());
            errdefer loaded_enc.deinit(allocator);
            try shardings.checkLoaded(loaded_dit.cfg, loaded_enc.cfg);

            var loaded_visual = try visual_vae.LoadedModel.init(allocator, io, visual_src.dir, visual_store.view());
            errdefer loaded_visual.deinit(allocator);
            var loaded_audio = try audio_vae.LoadedModel.init(allocator, io, audio_src.dir, audio_store.view());
            errdefer loaded_audio.deinit(allocator);
            log.info("vae: video+audio graphs ready", .{});

            return .{
                .task = task.dir,
                .task_owned = task.owned,
                .dit_src = dit_src,
                .enc_src = enc_src,
                .visual_src = visual_src,
                .audio_src = audio_src,
                .visual_source = visual_source,
                .dit_registry = dit_registry,
                .dit_store = dit_store,
                .enc_registry = enc_registry,
                .enc_store = enc_store,
                .visual_registry = visual_registry,
                .visual_store = visual_store,
                .audio_registry = audio_registry,
                .audio_store = audio_store,
                .dit = loaded_dit,
                .enc = loaded_enc,
                .visual = loaded_visual,
                .audio = loaded_audio,
            };
        }

        pub fn deinit(self: *Bundle, allocator: std.mem.Allocator, io: std.Io) void {
            self.audio.deinit(allocator);
            self.visual.deinit(allocator);
            self.enc.deinit(allocator);
            self.dit.deinit(allocator);
            self.audio_store.deinit();
            self.audio_registry.deinit();
            allocator.destroy(self.audio_registry);
            self.visual_store.deinit();
            self.visual_registry.deinit();
            allocator.destroy(self.visual_registry);
            self.enc_store.deinit();
            self.enc_registry.deinit();
            allocator.destroy(self.enc_registry);
            self.dit_store.deinit();
            self.dit_registry.deinit();
            allocator.destroy(self.dit_registry);
            if (self.visual_source) |*dir| dir.close(io);
            self.audio_src.deinit(allocator, io);
            self.visual_src.deinit(allocator, io);
            self.enc_src.deinit(allocator, io);
            self.dit_src.deinit(allocator, io);
            if (self.task_owned) self.task.close(io);
        }
    };

    fn refuseUnsupported(registry: *zml.safetensors.TensorRegistry, allocator: std.mem.Allocator) !void {
        var keys: std.ArrayList([]const u8) = .empty;
        defer keys.deinit(allocator);
        var it = registry.iterator();
        while (it.next()) |e| try keys.append(allocator, e.key_ptr.*);
        const report = checkpoint.inspect(keys.items);
        if (checkpoint.refuseReason(report)) |why| {
            log.err("{s}", .{why});
            return error.UnsupportedCheckpoint;
        }
    }

    fn openOptionalDir(io: std.Io, parent: std.Io.Dir, name: []const u8) ?std.Io.Dir {
        return parent.openDir(io, name, .{}) catch null;
    }

    fn openShared(io: std.Io, task_dir: std.Io.Dir, repo: std.Io.Dir, name: []const u8) ?std.Io.Dir {
        if (openOptionalDir(io, task_dir, name)) |dir| return dir;
        if (openOptionalDir(io, repo, name)) |dir| return dir;
        return openOfficialNested(io, repo, name);
    }

    fn openOfficialNested(io: std.Io, repo: std.Io.Dir, name: []const u8) ?std.Io.Dir {
        for (config.official_task_dirs) |task| {
            if (openNestedDir(io, repo, task, name)) |dir| return dir;
        }
        return null;
    }

    fn openRegistry(
        allocator: std.mem.Allocator,
        io: std.Io,
        src: FileSource,
    ) !zml.safetensors.TensorRegistry {
        if (src.file) |name| {
            const file = try src.dir.openFile(io, name, .{ .mode = .read_only });
            defer file.close(io);
            log.info("weights: {s}", .{name});
            return zml.safetensors.fetchRegistry(allocator, io, src.dir, file);
        }
        for (weight_entrypoints) |name| {
            if (src.dir.openFile(io, name, .{ .mode = .read_only })) |file| {
                defer file.close(io);
                log.info("weights: {s}", .{name});
                return zml.safetensors.fetchRegistry(allocator, io, src.dir, file);
            } else |_| {}
        }
        return error.FileNotFound;
    }

    fn fileInDir(io: std.Io, dir: std.Io.Dir, name: []const u8) bool {
        const file = dir.openFile(io, name, .{ .mode = .read_only }) catch return false;
        file.close(io);
        return true;
    }

    /// Official HF dumps use either Transformers (`model.safetensors*`) or
    /// Diffusers (`diffusion_pytorch_model*`) names. Empty task folders such as
    /// `FL2VA/transformer` exist and must not win over the real shard dir.
    pub const weight_entrypoints = [_][]const u8{
        "model.safetensors.index.json",
        "model.safetensors",
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors",
    };

    fn dirHasWeights(io: std.Io, dir: std.Io.Dir) bool {
        for (weight_entrypoints) |name| {
            if (fileInDir(io, dir, name)) return true;
        }
        return false;
    }

    fn takeWeightedDir(io: std.Io, dir: ?std.Io.Dir) ?std.Io.Dir {
        const opened = dir orelse return null;
        if (dirHasWeights(io, opened)) return opened;
        opened.close(io);
        return null;
    }

    fn resolveDit(
        allocator: std.mem.Allocator,
        io: std.Io,
        search: Search,
        variant: config.Variant,
        opts: Open,
    ) !FileSource {
        if (opts.dit.len != 0)
            return openFilePath(allocator, io, search, opts.dit, &.{"diffusion_models"}) catch
                return error.TransformerMissing;
        if (std.mem.endsWith(u8, opts.model, ".safetensors")) {
            return .{
                .dir = search.repo,
                .dir_owned = false,
                .file = try allocator.dupe(u8, std.fs.path.basename(opts.model)),
            };
        }
        if (openOfficialDit(io, search, variant)) |dir| {
            return .{ .dir = dir, .dir_owned = true, .file = null };
        }
        const needles: []const []const u8 = switch (variant.taskFamily()) {
            .fl2va => &.{"fl2va"},
            .ref2va => &.{"ref2va"},
        };
        const missing: anyerror = if (variant.taskFamily() == .ref2va)
            error.Ref2vaTransformerMissing
        else
            error.TransformerMissing;
        return takeScan(
            scanIn(allocator, io, search, "diffusion_models", needles, true),
            missing,
            error.AmbiguousDit,
        );
    }

    const ComponentSpec = struct {
        official: []const u8,
        aliases: []const []const u8 = &.{},
        scan: []const u8,
        needles: []const []const u8,
        missing: anyerror,
    };

    fn resolveComponent(
        allocator: std.mem.Allocator,
        io: std.Io,
        search: Search,
        spec: ComponentSpec,
    ) !FileSource {
        if (openShared(io, search.task, search.repo, spec.official)) |dir| {
            return .{ .dir = dir, .dir_owned = true, .file = null };
        }
        for (spec.aliases) |name| {
            if (openShared(io, search.task, search.repo, name)) |dir| {
                return .{ .dir = dir, .dir_owned = true, .file = null };
            }
        }
        const src = (try scanIn(allocator, io, search, spec.scan, spec.needles, false)) orelse return spec.missing;
        return src;
    }

    fn openOfficialDit(io: std.Io, search: Search, variant: config.Variant) ?std.Io.Dir {
        // Official dump has no Ref2VA/. openTaskDir then falls back to the repo, so
        // task/transformer is the fl2va DiT and must not win for ref2va.
        return switch (variant.taskFamily()) {
            .ref2va => takeWeightedDir(io, openOptionalDir(io, search.repo, "transformer_ref")) orelse
                takeWeightedDir(io, openNestedDir(io, search.repo, config.taskDirName(.ref2va), "transformer")),
            .fl2va => takeWeightedDir(io, openOptionalDir(io, search.task, "transformer")) orelse
                takeWeightedDir(io, openOptionalDir(io, search.repo, "transformer")) orelse
                takeWeightedDir(io, openNestedDir(io, search.repo, config.taskDirName(.fl2va), "transformer")),
        };
    }

    fn openFilePath(
        allocator: std.mem.Allocator,
        io: std.Io,
        search: Search,
        path: []const u8,
        folders: []const []const u8,
    ) !FileSource {
        const base = std.fs.path.basename(path);
        if (std.fs.path.dirname(path) != null and (std.mem.indexOfScalar(u8, path, '/') != null or std.mem.indexOfScalar(u8, path, '\\') != null)) {
            const dir = try zml.safetensors.resolveModelRepo(io, path);
            return .{ .dir = dir, .dir_owned = true, .file = try allocator.dupe(u8, base) };
        }
        if (fileInDir(io, search.repo, base)) {
            return .{ .dir = search.repo, .dir_owned = false, .file = try allocator.dupe(u8, base) };
        }
        if (try fileInFolders(allocator, io, search.repo, base, folders)) |src| return src;
        if (search.extra) |root| {
            if (try fileInFolders(allocator, io, root, base, folders)) |src| return src;
        }
        return error.FileNotFound;
    }

    fn takeScan(result: anytype, missing: anyerror, ambiguous: anyerror) !FileSource {
        const src = result catch |err| switch (err) {
            error.AmbiguousWeights => return ambiguous,
            else => |e| return e,
        };
        return src orelse return missing;
    }

    fn fileInFolders(
        allocator: std.mem.Allocator,
        io: std.Io,
        root: std.Io.Dir,
        base: []const u8,
        folders: []const []const u8,
    ) !?FileSource {
        for (folders) |folder| {
            if (openOptionalDir(io, root, folder)) |dir| {
                if (fileInDir(io, dir, base)) {
                    return .{ .dir = dir, .dir_owned = true, .file = try allocator.dupe(u8, base) };
                }
                dir.close(io);
            }
        }
        return null;
    }

    fn scanIn(
        allocator: std.mem.Allocator,
        io: std.Io,
        search: Search,
        folder: []const u8,
        needles: []const []const u8,
        unique: bool,
    ) !?FileSource {
        if (try scanFolder(allocator, io, search.repo, folder, needles, unique)) |src| return src;
        if (search.extra) |root| {
            if (try scanFolder(allocator, io, root, folder, needles, unique)) |src| return src;
        }
        return null;
    }

    fn scanFolder(
        allocator: std.mem.Allocator,
        io: std.Io,
        root: std.Io.Dir,
        folder: []const u8,
        needles: []const []const u8,
        unique: bool,
    ) !?FileSource {
        const dir = root.openDir(io, folder, .{ .iterate = true }) catch return null;
        if (scanFilename(allocator, io, dir, needles, unique)) |name| {
            if (name) |found| return .{ .dir = dir, .dir_owned = true, .file = found };
            dir.close(io);
            return null;
        } else |err| {
            dir.close(io);
            return err;
        }
    }

    fn scanFilename(
        allocator: std.mem.Allocator,
        io: std.Io,
        dir: std.Io.Dir,
        needles: []const []const u8,
        unique: bool,
    ) !?[]u8 {
        var it = dir.iterate();
        var found: ?[]u8 = null;
        errdefer if (found) |name| allocator.free(name);
        while (try it.next(io)) |entry| {
            if (entry.kind != .file) continue;
            if (!checkpoint.safetensorsContains(entry.name, needles)) continue;
            if (found != null) {
                if (unique) return error.AmbiguousWeights;
                continue;
            }
            found = try allocator.dupe(u8, entry.name);
        }
        return found;
    }

    fn openBundleRoot(io: std.Io, model_path: []const u8) ?std.Io.Dir {
        if (!std.mem.endsWith(u8, model_path, ".safetensors")) return null;
        const parent = std.fs.path.dirname(model_path) orelse return null;
        if (!checkpoint.isBundleLeaf(std.fs.path.basename(parent))) return null;
        const root = std.fs.path.dirname(parent) orelse ".";
        return std.Io.Dir.openDir(.cwd(), io, root, .{}) catch null;
    }

    pub const tokenizer_relpaths = [_][]const u8{
        "tokenizer/tokenizer.json",
        "processor/tokenizer.json",
        "text_encoder/tokenizer.json",
        "tokenizer.json",
    };

    pub fn loadTokenizer(
        allocator: std.mem.Allocator,
        io: std.Io,
        task_dir: std.Io.Dir,
        repo: std.Io.Dir,
        model: []const u8,
        progress: *std.Progress.Node,
    ) !zml.tokenizer.Tokenizer {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start("Loading tokenizer...", 1);
        defer node.end();

        const bytes = try readTokenizerBytes(allocator, io, task_dir, repo, model);
        defer allocator.free(bytes);
        log.info("tokenizer: {d} bytes", .{bytes.len});
        return try .fromBytes(allocator, bytes);
    }

    fn openNestedDir(io: std.Io, parent: std.Io.Dir, first: []const u8, second: []const u8) ?std.Io.Dir {
        var outer = parent.openDir(io, first, .{}) catch return null;
        const inner = outer.openDir(io, second, .{}) catch {
            outer.close(io);
            return null;
        };
        outer.close(io);
        return inner;
    }

    fn readTokenizerBytes(
        allocator: std.mem.Allocator,
        io: std.Io,
        task_dir: std.Io.Dir,
        repo: std.Io.Dir,
        model: []const u8,
    ) ![]u8 {
        if (readTokenizerAny(allocator, io, task_dir, repo, model)) |bytes| return bytes else |err| switch (err) {
            error.MissingTokenizer => {},
            else => return err,
        }
        return readOfficialTokenizer(allocator, io);
    }

    fn readTokenizerAny(
        allocator: std.mem.Allocator,
        io: std.Io,
        task_dir: std.Io.Dir,
        repo: std.Io.Dir,
        model_path: []const u8,
    ) ![]u8 {
        var extra = openBundleRoot(io, model_path);
        defer if (extra) |*dir| dir.close(io);

        const nearby = [_]?std.Io.Dir{ task_dir, repo, extra };
        for (nearby) |maybe| {
            const dir = maybe orelse continue;
            if (readTokenizer(allocator, io, dir)) |bytes| return bytes else |err| switch (err) {
                error.MissingTokenizer => {},
                else => return err,
            }
        }
        for (config.official_task_dirs) |name| {
            var dir = repo.openDir(io, name, .{}) catch continue;
            defer dir.close(io);
            if (readTokenizer(allocator, io, dir)) |bytes| return bytes else |err| switch (err) {
                error.MissingTokenizer => {},
                else => return err,
            }
        }
        return error.MissingTokenizer;
    }

    fn readOfficialTokenizer(allocator: std.mem.Allocator, io: std.Io) ![]u8 {
        var buf: [256]u8 = undefined;
        const path = config.officialTokenizerUri(&buf) catch return error.MissingTokenizer;
        const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return error.MissingTokenizer;
        defer file.close(io);
        log.info("tokenizer: {s}", .{path});
        return readTokenizerFile(allocator, io, file);
    }

    fn readTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) ![]u8 {
        for (tokenizer_relpaths) |name| {
            const file = dir.openFile(io, name, .{}) catch continue;
            defer file.close(io);
            return readTokenizerFile(allocator, io, file);
        }
        return error.MissingTokenizer;
    }

    fn readTokenizerFile(allocator: std.mem.Allocator, io: std.Io, file: std.Io.File) ![]u8 {
        var reader = file.reader(io, &.{});
        return try reader.interface.readAlloc(allocator, try file.length(io));
    }
};

// --- runtime/encode.zig ---
pub const encode = struct {
    const std = @import("std");

    const zml = @import("zml");

    const audio_vae = @import("vae.zig").audio;
    const config_mod = @import("model.zig").config;
    const packing = @import("model.zig").packing;
    const vae = @import("vae.zig").geom;
    const visual_enc = @import("vae.zig").visual_enc;

    const log = std.log.scoped(.minimax_h3_encode);

    fn bufferFromItems(io: std.Io, platform: *const zml.Platform, shape: zml.Shape, items: anytype) !zml.Buffer {
        return zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(items));
    }

    fn copyNchwTile(
        src: []const f32,
        channels: u32,
        t: u32,
        src_h: u32,
        src_w: u32,
        y0: u32,
        x0: u32,
        tile_h: u32,
        tile_w: u32,
        dst: []f32,
    ) void {
        @memset(dst, 0);
        const copy_h = @min(tile_h, src_h - y0);
        const copy_w = @min(tile_w, src_w - x0);
        var c: u32 = 0;
        while (c < channels) : (c += 1) {
            var tt: u32 = 0;
            while (tt < t) : (tt += 1) {
                var y: u32 = 0;
                while (y < copy_h) : (y += 1) {
                    const src_row = ((((c * t + tt) * src_h) + (y0 + y)) * src_w) + x0;
                    const dst_row = ((((c * t + tt) * tile_h) + y) * tile_w);
                    @memcpy(dst[dst_row..][0..copy_w], src[src_row..][0..copy_w]);
                }
            }
        }
    }

    fn blendNchw(
        acc: []f32,
        incoming: []const f32,
        channels: u32,
        t: u32,
        acc_h: u32,
        acc_w: u32,
        inc_h: u32,
        inc_w: u32,
        out_y: u32,
        out_x: u32,
        blend_h: u32,
        blend_w: u32,
    ) void {
        if (blend_h == 0 and blend_w == 0) {
            var c: u32 = 0;
            while (c < channels) : (c += 1) {
                var tt: u32 = 0;
                while (tt < t) : (tt += 1) {
                    var y: u32 = 0;
                    while (y < inc_h) : (y += 1) {
                        const si = ((((c * t + tt) * inc_h) + y) * inc_w);
                        const di = ((((c * t + tt) * acc_h) + (out_y + y)) * acc_w) + out_x;
                        @memcpy(acc[di..][0..inc_w], incoming[si..][0..inc_w]);
                    }
                }
            }
            return;
        }
        var c: u32 = 0;
        while (c < channels) : (c += 1) {
            var tt: u32 = 0;
            while (tt < t) : (tt += 1) {
                var y: u32 = 0;
                while (y < inc_h) : (y += 1) {
                    var x: u32 = 0;
                    while (x < inc_w) : (x += 1) {
                        const si = ((((c * t + tt) * inc_h) + y) * inc_w) + x;
                        const di = ((((c * t + tt) * acc_h) + (out_y + y)) * acc_w) + (out_x + x);
                        var w: f32 = 1.0;
                        if (blend_h > 0 and y < blend_h) {
                            w *= @as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(blend_h));
                        }
                        if (blend_w > 0 and x < blend_w) {
                            w *= @as(f32, @floatFromInt(x)) / @as(f32, @floatFromInt(blend_w));
                        }
                        acc[di] = acc[di] * (1.0 - w) + incoming[si] * w;
                    }
                }
            }
        }
    }

    fn runVisualClip(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        compiled: *const pipeline.EncodeCompiled,
        bufs: *const zml.Bufferized(visual_enc.Model),
        pixels_nchw: []const f32,
        frames: u32,
        height: u32,
        width: u32,
    ) ![]f32 {
        const spec = vae.official_visual;
        const tile_h = compiled.tile_h;
        const tile_w = compiled.tile_w;
        const y_plan = try vae.splitTiles(allocator, height, spec.tile_px, spec.tile_overlap_px, spec.spatial);
        defer y_plan.deinit(allocator);
        const x_plan = try vae.splitTiles(allocator, width, spec.tile_px, spec.tile_overlap_px, spec.spatial);
        defer x_plan.deinit(allocator);

        const exe = if (frames == 1)
            if (compiled.visual_t1) |*c| c else return error.VisualEncodeMissing
        else if (compiled.visual_clip) |*c| c else return error.VisualClipMissing;
        var runner = try zml.FnExe(visual_enc.encode).Runner(.{.model}).init(exe, allocator, .{ .model = bufs.* });
        defer runner.deinit(allocator);

        const latent_h = height / spec.spatial;
        const latent_w = width / spec.spatial;
        const moments_c: u32 = 48;
        const out_t = if (frames == 1) @as(u32, 1) else spec.tokensChunkSize();
        const canvas = try allocator.alloc(f32, moments_c * out_t * latent_h * latent_w);
        errdefer allocator.free(canvas);
        @memset(canvas, 0);

        const tile_px = try allocator.alloc(f32, 3 * frames * tile_h * tile_w);
        defer allocator.free(tile_px);
        const tile_lat_h = tile_h / spec.spatial;
        const tile_lat_w = tile_w / spec.spatial;
        const tile_mom = try allocator.alloc(f32, moments_c * out_t * tile_lat_h * tile_lat_w);
        defer allocator.free(tile_mom);

        var out_y: u32 = 0;
        for (y_plan.starts, y_plan.lengths, 0..) |y0, ylen, yi| {
            var out_x: u32 = 0;
            for (x_plan.starts, x_plan.lengths, 0..) |x0, xlen, xi| {
                copyNchwTile(pixels_nchw, 3, frames, height, width, y0, x0, tile_h, tile_w, tile_px);
                var pix = try bufferFromItems(io, platform, .init(.{
                    .b = 1,
                    .c = 3,
                    .t = frames,
                    .h = tile_h,
                    .w = tile_w,
                }, .f32), tile_px);
                defer pix.deinit();
                var moments: zml.Buffer = undefined;
                runner.run(io, .{
                    .inputs = .{ .pixels = pix },
                    .outputs = .{ .moments = &moments },
                    .opts = .{ .wait = true },
                });
                defer moments.deinit();
                try moments.toSlice(io, .init(zml.Shape.init(.{
                    .b = 1,
                    .c = moments_c,
                    .t = out_t,
                    .h = tile_lat_h,
                    .w = tile_lat_w,
                }, .f32), std.mem.sliceAsBytes(tile_mom)));

                const use_h = ylen / spec.spatial;
                const use_w = xlen / spec.spatial;
                const blend_h: u32 = if (yi == 0) 0 else y_plan.overlaps[yi - 1] / spec.spatial;
                const blend_w: u32 = if (xi == 0) 0 else x_plan.overlaps[xi - 1] / spec.spatial;
                blendNchw(canvas, tile_mom, moments_c, out_t, latent_h, latent_w, use_h, use_w, out_y, out_x, blend_h, blend_w);
                out_x += if (xi + 1 == x_plan.count()) use_w else use_w - (if (xi + 1 < x_plan.count()) x_plan.overlaps[xi] / spec.spatial else 0);
            }
            out_y += if (yi + 1 == y_plan.count()) ylen / spec.spatial else (ylen - (if (yi + 1 < y_plan.count()) y_plan.overlaps[yi] else 0)) / spec.spatial;
        }
        return canvas;
    }

    fn momentsToLatentThwc(
        allocator: std.mem.Allocator,
        moments_nchw: []const f32,
        t: u32,
        h: u32,
        w: u32,
        mean: []const f32,
        stddev: []const f32,
        policy: config_mod.PosteriorPolicy,
    ) ![]f32 {
        const sampled = try vae.sampleVisualPosteriorNchw(allocator, moments_nchw, t, h, w, policy);
        defer allocator.free(sampled);
        const out = try vae.nchwToThwc(allocator, sampled, 24, t, h, w);
        vae.applyLatentNorm(out, 24, mean, stddev, false);
        return out;
    }

    pub const VisualLatent = struct {
        thwc: []f32,
        latent_t: u32,
        latent_h: u32,
        latent_w: u32,
        keyframe_index: i32 = 0,
        guide_frame: ?i32 = null,

        pub fn deinit(self: VisualLatent, allocator: std.mem.Allocator) void {
            allocator.free(self.thwc);
        }
    };

    pub fn encodeKeyframe(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        compiled: *const pipeline.EncodeCompiled,
        loaded: *const visual_enc.LoadedModel,
        bufs: *const zml.Bufferized(visual_enc.Model),
        pixels_nchw: []const f32,
        height: u32,
        width: u32,
        policy: config_mod.PosteriorPolicy,
    ) !VisualLatent {
        const moments = try runVisualClip(allocator, io, platform, compiled, bufs, pixels_nchw, 1, height, width);
        defer allocator.free(moments);
        const lh = height / 16;
        const lw = width / 16;
        log.info("visual encode keyframe {d}x{d} -> latent 1x{d}x{d}", .{ width, height, lh, lw });
        return .{
            .thwc = try momentsToLatentThwc(allocator, moments, 1, lh, lw, &loaded.cfg.latents_mean, &loaded.cfg.latents_std, policy),
            .latent_t = 1,
            .latent_h = lh,
            .latent_w = lw,
        };
    }

    pub fn encodeVideo(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        compiled: *const pipeline.EncodeCompiled,
        loaded: *const visual_enc.LoadedModel,
        bufs: *const zml.Bufferized(visual_enc.Model),
        pixels_nchw: []const f32,
        frames: u32,
        height: u32,
        width: u32,
        policy: config_mod.PosteriorPolicy,
    ) !VisualLatent {
        const spec = vae.official_visual;
        const pad = (spec.clip_length - (frames % spec.clip_length)) % spec.clip_length;
        const padded_t = frames + pad;
        const plane = @as(usize, height) * width;
        const padded = try allocator.alloc(f32, 3 * padded_t * plane);
        defer allocator.free(padded);
        @memcpy(padded[0 .. 3 * frames * plane], pixels_nchw[0 .. 3 * frames * plane]);
        if (pad > 0) {
            var c: u32 = 0;
            while (c < 3) : (c += 1) {
                const last = pixels_nchw[(c * frames + (frames - 1)) * plane ..][0..plane];
                var p: u32 = 0;
                while (p < pad) : (p += 1) {
                    @memcpy(padded[(c * padded_t + frames + p) * plane ..][0..plane], last);
                }
            }
        }

        const encode_start: std.Io.Timestamp = .now(io, .awake);
        const clips = padded_t / spec.clip_length;
        const chunk = spec.tokensChunkSize();
        const lh = height / spec.spatial;
        const lw = width / spec.spatial;
        var acc_t: u32 = 0;
        const all = try allocator.alloc(f32, 48 * clips * chunk * lh * lw);
        defer allocator.free(all);

        var clip_i: u32 = 0;
        while (clip_i < clips) : (clip_i += 1) {
            const clip_px = try allocator.alloc(f32, 3 * spec.clip_length * plane);
            defer allocator.free(clip_px);
            var c: u32 = 0;
            while (c < 3) : (c += 1) {
                const src = (c * padded_t + clip_i * spec.clip_length) * plane;
                const dst = c * spec.clip_length * plane;
                @memcpy(clip_px[dst..][0 .. spec.clip_length * plane], padded[src..][0 .. spec.clip_length * plane]);
            }
            const moments = try runVisualClip(allocator, io, platform, compiled, bufs, clip_px, spec.clip_length, height, width);
            defer allocator.free(moments);
            const n = 48 * chunk * lh * lw;
            @memcpy(all[acc_t * 48 * lh * lw ..][0..n], moments[0..n]);
            acc_t += chunk;
            log.info("visual encode clip {d}/{d}", .{ clip_i + 1, clips });
        }

        const keep_t = if (spec.token_drop < acc_t) acc_t - spec.token_drop else acc_t;
        const kept = all[0 .. 48 * keep_t * lh * lw];
        log.info("visual encode video {d}x{d}x{d} -> {d}x{d}x{d} [{f}]", .{
            frames,
            height,
            width,
            keep_t,
            lh,
            lw,
            encode_start.untilNow(io, .awake),
        });
        return .{
            .thwc = try momentsToLatentThwc(allocator, kept, keep_t, lh, lw, &loaded.cfg.latents_mean, &loaded.cfg.latents_std, policy),
            .latent_t = keep_t,
            .latent_h = lh,
            .latent_w = lw,
        };
    }

    pub const AudioLatent = struct {
        values: []f32,
        latent_t: u32,

        pub fn deinit(self: AudioLatent, allocator: std.mem.Allocator) void {
            allocator.free(self.values);
        }
    };

    pub fn encodeAudio(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        compiled: *const pipeline.EncodeCompiled,
        loaded: *const audio_vae.LoadedEncoder,
        bufs: *const zml.Bufferized(audio_vae.EncoderModel),
        stereo: []const f32,
    ) !AudioLatent {
        const hop: u32 = @intCast(loaded.cfg.hop);
        const frames: u32 = @intCast(stereo.len / 2);
        const pad = (hop - (frames % hop)) % hop;
        const samples = frames + pad;
        const left = try allocator.alloc(f32, samples);
        defer allocator.free(left);
        const right = try allocator.alloc(f32, samples);
        defer allocator.free(right);
        @memset(left, 0);
        @memset(right, 0);
        var i: usize = 0;
        while (i < frames) : (i += 1) {
            left[i] = stereo[i * 2];
            right[i] = stereo[i * 2 + 1];
        }
        const batch = try allocator.alloc(f32, 2 * samples);
        defer allocator.free(batch);
        @memcpy(batch[0..samples], left);
        @memcpy(batch[samples..], right);

        const exe = if (compiled.audio) |*c| c else return error.AudioEncodeMissing;
        var runner = try zml.FnExe(audio_vae.encode).Runner(.{.model}).init(exe, allocator, .{ .model = bufs.* });
        defer runner.deinit(allocator);
        var wav = try bufferFromItems(io, platform, .init(.{ .b = 2, .c = 1, .t = samples }, .f32), batch);
        defer wav.deinit();
        var latents: zml.Buffer = undefined;
        runner.run(io, .{
            .inputs = .{ .wav = wav },
            .outputs = .{ .latents = &latents },
            .opts = .{ .wait = true },
        });
        defer latents.deinit();
        const latent_t = samples / hop;
        const channels: usize = @intCast(loaded.cfg.latent_channels);
        const host = try allocator.alloc(f32, 2 * channels * latent_t);
        errdefer allocator.free(host);
        try latents.toSlice(io, .init(zml.Shape.init(.{ .b = 2, .c = loaded.cfg.latent_channels, .t = latent_t }, .f32), std.mem.sliceAsBytes(host)));

        const packed_latents = try allocator.alloc(f32, host.len);
        errdefer allocator.free(packed_latents);
        vae.audioBctToRows(packed_latents, host, @intCast(channels), latent_t);
        allocator.free(host);
        vae.applyLatentNorm(packed_latents, @intCast(channels), &loaded.cfg.latents_mean, &loaded.cfg.latents_std, false);
        log.info("audio encode samples={d} latent_t={d} channels={d}", .{ samples, latent_t, channels });
        return .{ .values = packed_latents, .latent_t = latent_t };
    }

    pub const ConditionSet = struct {
        videos: []packing.ConditionVideo,
        video_patches: []f32,
        target_video_offset: u32,
        audios: []packing.ConditionAudio,
        audio_patches: []f32,
        target_audio_offset: u32,
        references: []packing.ReferenceBlock,

        pub fn empty() ConditionSet {
            return .{
                .videos = &.{},
                .video_patches = &.{},
                .target_video_offset = 0,
                .audios = &.{},
                .audio_patches = &.{},
                .target_audio_offset = 0,
                .references = &.{},
            };
        }

        pub fn deinit(self: ConditionSet, allocator: std.mem.Allocator) void {
            allocator.free(self.videos);
            allocator.free(self.video_patches);
            allocator.free(self.audios);
            allocator.free(self.audio_patches);
            allocator.free(self.references);
        }
    };

    pub fn packConditions(
        allocator: std.mem.Allocator,
        visuals: []const VisualLatent,
        audios: []const AudioLatent,
        references: []const packing.ReferenceBlock,
        patch: [3]i64,
    ) !ConditionSet {
        const vmeta = try allocator.alloc(packing.ConditionVideo, visuals.len);
        errdefer allocator.free(vmeta);
        var vlen: usize = 0;
        for (visuals, vmeta) |v, *m| {
            m.* = .{
                .latent_t = v.latent_t,
                .latent_h = v.latent_h,
                .latent_w = v.latent_w,
                .keyframe_index = v.keyframe_index,
                .guide_frame = v.guide_frame,
            };
            vlen += config_mod.videoTokenCount(v.latent_t, v.latent_h, v.latent_w, patch) * patchDim(patch);
        }
        const vpatches = try allocator.alloc(f32, vlen);
        errdefer allocator.free(vpatches);
        var off: usize = 0;
        for (visuals) |v| {
            const rows = try packing.patchify(allocator, v.thwc, v.latent_t, v.latent_h, v.latent_w, 24, patch);
            defer allocator.free(rows);
            @memcpy(vpatches[off..][0..rows.len], rows);
            off += rows.len;
        }

        const ameta = try allocator.alloc(packing.ConditionAudio, audios.len);
        errdefer allocator.free(ameta);
        var alen: usize = 0;
        for (audios, ameta) |a, *m| {
            m.* = .{ .latent_t = a.latent_t };
            alen += a.values.len;
        }
        const apatches = try allocator.alloc(f32, alen);
        errdefer allocator.free(apatches);
        off = 0;
        for (audios) |a| {
            @memcpy(apatches[off..][0..a.values.len], a.values);
            off += a.values.len;
        }
        const refs = try allocator.dupe(packing.ReferenceBlock, references);

        return .{
            .videos = vmeta,
            .video_patches = vpatches,
            .target_video_offset = @intCast(vlen / patchDim(patch)),
            .audios = ameta,
            .audio_patches = apatches,
            .target_audio_offset = @intCast(alen / 32),
            .references = refs,
        };
    }

    fn patchDim(patch: [3]i64) u32 {
        return 24 * @as(u32, @intCast(patch[0] * patch[1] * patch[2]));
    }
};

// --- runtime/conditions.zig ---
pub const conditions = struct {
    const std = @import("std");

    const zml = @import("zml");

    const audio_vae = @import("vae.zig").audio;
    const config_mod = @import("model.zig").config;
    const encode_mod = @import("generate.zig").encode;
    const geom = @import("generate.zig").cond_geom;
    const packing = @import("model.zig").packing;
    const request_mod = @import("generate.zig").request;
    const session_mod = @import("generate.zig").session;
    const sharding_mod = @import("generate.zig").sharding;
    const vae = @import("vae.zig").geom;
    const vision = @import("model.zig").vision;
    const visual_enc = @import("vae.zig").visual_enc;
    const visual_vae = @import("vae.zig").visual;

    const log = std.log.scoped(.minimax_h3_conditions);

    pub const Prepared = struct {
        tokens: []u32,
        tags: []u8,
        positions: ?[]f32 = null,
        deepstack: [3]?[]f32 = .{ null, null, null },
        vision_merged: ?[]f32 = null,
        vision_spans: []session_mod.VisionSpan = &.{},
        conds: encode_mod.ConditionSet = .empty(),

        pub fn deinit(self: Prepared, allocator: std.mem.Allocator) void {
            allocator.free(self.tokens);
            allocator.free(self.tags);
            if (self.positions) |p| allocator.free(p);
            for (self.deepstack) |d| if (d) |x| allocator.free(x);
            if (self.vision_merged) |m| allocator.free(m);
            if (self.vision_spans.len != 0) allocator.free(self.vision_spans);
            self.conds.deinit(allocator);
        }

        pub fn extras(self: Prepared) session_mod.TextExtras {
            return .{
                .positions = self.positions,
                .deepstack = self.deepstack,
                .vision_merged = self.vision_merged,
                .vision_spans = self.vision_spans,
            };
        }
    };

    pub fn tokenize(allocator: std.mem.Allocator, encode_text: anytype, text: []const u8) !Prepared {
        const tokens = try encode_text.encodeAlloc(allocator, text);
        errdefer allocator.free(tokens);
        const tags = try allocator.alloc(u8, tokens.len);
        @memset(tags, @intFromEnum(packing.Modality.text));
        return .{ .tokens = tokens, .tags = tags };
    }

    fn hasVideo(items: anytype) bool {
        for (items) |item| if (item.kind == .video or item.kind == .video_audio) return true;
        return false;
    }

    fn padStereo(allocator: std.mem.Allocator, stereo: []const f32, samples: u32) ![]f32 {
        const out = try allocator.alloc(f32, @as(usize, samples) * 2);
        @memset(out, 0);
        const n = @min(stereo.len, out.len);
        @memcpy(out[0..n], stereo[0..n]);
        return out;
    }

    pub const Request = struct {
        variant: config_mod.Variant,
        first_image: []const u8,
        last_image: []const u8,
        refs: []const request_mod.Reference,
        prompt: []const u8,
        geo: pipeline.Geometry,
        models: *ckpt.Bundle,
        shardings: sharding_mod.Shardings,
    };

    pub fn prepare(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        progress: *std.Progress.Node,
        encode_text: anytype,
        req: Request,
    ) !Prepared {
        const variant = req.variant;
        const first_image = req.first_image;
        const last_image = req.last_image;
        const refs = req.refs;
        const prompt = req.prompt;
        const geo = req.geo;
        const models = req.models;
        const shardings = req.shardings;
        const enc_dir = models.enc_src.dir;
        const visual_store = &models.visual_store;
        const audio_store = &models.audio_store;
        const enc_store = &models.enc_store;
        const loaded_visual = &models.visual;
        const loaded_audio = &models.audio;
        const patch = models.dit.cfg.patch_size;
        const text_hidden = models.enc.cfg.hidden_size;
        log.info(
            "conditions: {s} first={s} last={s} refs={d}",
            .{
                @tagName(variant),
                if (first_image.len == 0) "-" else first_image,
                if (last_image.len == 0) "-" else last_image,
                refs.len,
            },
        );
        try request_mod.validateRefs(refs);
        const VisualItem = struct {
            kind: packing.ReferenceKind,
            path: []const u8,
            keyframe_index: i32 = 0,
            guide_frame: ?i32 = null,
            rgb: []u8 = &.{},
            qwen_rgb: []u8 = &.{},
            frames: u32 = 1,
            w: u32 = 0,
            h: u32 = 0,
            nchw: ?[]f32 = null,
            latent_t: u32 = 1,
            latent_h: u32 = 0,
            latent_w: u32 = 0,
            grid_h: u32 = 1,
            grid_w: u32 = 1,
            temporal: u32 = 1,
            merged: u32 = 0,
            seq: u32 = 0,
            video_index: i32 = -1,
            timestamps: []f32 = &.{},
            has_audio: bool = false,
        };
        const AudioItem = struct {
            path: []const u8,
            stereo: []f32 = &.{},
            latent_t: u32 = 0,
            audio_index: i32 = -1,
        };

        var visuals: std.ArrayList(VisualItem) = .empty;
        defer {
            for (visuals.items) |item| {
                if (item.rgb.len != 0) allocator.free(item.rgb);
                if (item.qwen_rgb.len != 0) allocator.free(item.qwen_rgb);
                if (item.nchw) |n| allocator.free(n);
                if (item.timestamps.len != 0) allocator.free(item.timestamps);
            }
            visuals.deinit(allocator);
        }
        var audios: std.ArrayList(AudioItem) = .empty;
        defer {
            for (audios.items) |item| if (item.stereo.len != 0) allocator.free(item.stereo);
            audios.deinit(allocator);
        }
        var blocks: std.ArrayList(packing.ReferenceBlock) = .empty;
        defer blocks.deinit(allocator);

        if (variant == .fl2va) {
            if (first_image.len != 0) try visuals.append(allocator, .{ .kind = .image, .path = first_image, .keyframe_index = 0 });
            if (last_image.len != 0) try visuals.append(allocator, .{ .kind = .image, .path = last_image, .keyframe_index = 1 });
        } else {
            for (refs) |ref| {
                switch (ref.kind) {
                    .image => {
                        const vidx: i32 = @intCast(visuals.items.len);
                        try visuals.append(allocator, .{ .kind = .image, .path = ref.path, .video_index = vidx });
                        try blocks.append(allocator, .{ .kind = .image, .video_index = vidx });
                    },
                    .video, .video_audio => {
                        const vidx: i32 = @intCast(visuals.items.len);
                        var aidx: i32 = -1;
                        var has_audio = ref.kind == .video_audio or ref.soundtrack.len != 0;
                        var audio_path = ref.soundtrack;
                        if (!has_audio) {
                            const meta = try media.probeVideo(allocator, io, ref.path);
                            if (meta.has_audio) {
                                has_audio = true;
                                audio_path = ref.path;
                            }
                        }
                        try visuals.append(allocator, .{
                            .kind = if (has_audio) .video_audio else .video,
                            .path = ref.path,
                            .video_index = vidx,
                            .has_audio = has_audio,
                        });
                        if (has_audio) {
                            aidx = @intCast(audios.items.len);
                            try audios.append(allocator, .{ .path = if (audio_path.len != 0) audio_path else ref.path, .audio_index = aidx });
                        }
                        try blocks.append(allocator, .{
                            .kind = if (has_audio) .video_audio else .video,
                            .video_index = vidx,
                            .audio_index = aidx,
                        });
                    },
                    .audio => {
                        const aidx: i32 = @intCast(audios.items.len);
                        try audios.append(allocator, .{ .path = ref.path, .audio_index = aidx });
                        try visuals.append(allocator, .{ .kind = .audio, .path = ref.path, .has_audio = true });
                        try blocks.append(allocator, .{ .kind = .audio, .audio_index = aidx });
                    },
                }
            }
        }

        const vcfg = try vision.configFromRepo(allocator, io, enc_dir, text_hidden);
        const hidden_dim: u32 = @intCast(text_hidden);
        const spatial = vae.official_visual.spatial;

        var keyframe_i: usize = 0;
        for (visuals.items) |*item| {
            if (item.kind == .audio) continue;
            if (item.kind == .video or item.kind == .video_audio) {
                const clip = try media.loadVideoNative(allocator, io, item.path);
                defer allocator.free(clip.rgb);
                const fps = if (clip.fps > 0) clip.fps else config_mod.video_fps;
                const indices = try geom.resampleFrameIndices(clip.frames, fps, config_mod.video_fps, allocator);
                defer allocator.free(indices);
                const keep = @min(geo.frames, @as(u32, @intCast(indices.len)));
                const own = try geom.videoCanvas(clip.w, clip.h);
                item.w = own.w;
                item.h = own.h;
                item.frames = keep;
                const src_plane = @as(usize, clip.w) * clip.h * 3;
                const dst_plane = @as(usize, own.w) * own.h * 3;
                item.rgb = try allocator.alloc(u8, keep * dst_plane);
                var fi: u32 = 0;
                while (fi < keep) : (fi += 1) {
                    const src = clip.rgb[indices[fi] * src_plane ..][0..src_plane];
                    const rgb = try geom.resizeLanczos(allocator, src, clip.w, clip.h, own.w, own.h);
                    defer allocator.free(rgb);
                    @memcpy(item.rgb[fi * dst_plane ..][0..dst_plane], rgb);
                }
                item.nchw = try media.rgbVideoToNchwImagenet(allocator, item.rgb, item.frames, item.h, item.w);
                item.latent_t = vae.encodeVideoLatentT(vae.official_visual, item.frames);
            } else if (variant == .fl2va) {
                item.rgb = if (keyframe_i == 0)
                    try media.loadRgb(allocator, io, item.path, geo.pixel_w, geo.pixel_h)
                else
                    try media.loadRgbCover(allocator, io, item.path, geo.pixel_w, geo.pixel_h);
                keyframe_i += 1;
                item.w = geo.pixel_w;
                item.h = geo.pixel_h;
                item.nchw = try media.rgbToNchwImagenet(allocator, item.rgb, item.h, item.w);
                item.latent_t = 1;
            } else {
                const raw = try media.loadRgbRaw(allocator, io, item.path);
                defer allocator.free(raw.rgb);
                const dest = try geom.refImageSize(raw.w, raw.h, geo.pixel_w, geo.pixel_h);
                item.rgb = try geom.resizeLanczos(allocator, raw.rgb, raw.w, raw.h, dest.w, dest.h);
                item.w = dest.w;
                item.h = dest.h;
                item.nchw = try media.rgbToNchwImagenet(allocator, item.rgb, item.h, item.w);
                item.latent_t = 1;
            }
            item.latent_h = item.h / spatial;
            item.latent_w = item.w / spatial;
            const video = item.kind == .video or item.kind == .video_audio;
            const spec = vision.spatialTokens(vcfg, item.h, item.w, video);
            item.grid_h = spec.grid.h;
            item.grid_w = spec.grid.w;
            if (video) {
                const sampled = try geom.sampleVideoConditionFrames(item.frames, config_mod.video_fps, config_mod.qwen_video_fps, 2);
                item.temporal = sampled.block_count;
                item.seq = spec.seq * item.temporal;
                item.merged = spec.merged;
                item.timestamps = try allocator.alloc(f32, sampled.block_count);
                const idx_buf = try allocator.alloc(u32, sampled.indices_len);
                defer allocator.free(idx_buf);
                const nidx = geom.fillVideoConditionIndices(item.frames, config_mod.video_fps, config_mod.qwen_video_fps, idx_buf);
                _ = geom.fillVideoTimestamps(sampled.block_count, item.timestamps);
                if (item.rgb.len != 0) {
                    var qwen_idx = try allocator.alloc(u32, sampled.block_count * 2);
                    defer allocator.free(qwen_idx);
                    var qi: u32 = 0;
                    while (qi < sampled.block_count * 2) : (qi += 1) {
                        qwen_idx[qi] = idx_buf[@min(nidx - 1, qi)];
                    }
                    item.qwen_rgb = try geom.applyRgb(allocator, item.rgb, item.w, item.h, qwen_idx);
                }
            } else {
                item.seq = spec.seq;
                item.merged = spec.merged;
            }
        }

        const hop = loaded_audio.cfg.hop;
        const rate = loaded_audio.cfg.sample_rate;
        var max_audio_samples: u32 = 0;
        for (audios.items) |*item| {
            const duration_s = @as(f32, @floatFromInt(geo.frames)) / config_mod.video_fps;
            item.stereo = try media.loadAudioOfficial(allocator, io, item.path, duration_s, rate);
            const samples: u32 = @intCast(item.stereo.len / 2);
            const aligned = geom.hopAlign(samples, hop);
            item.latent_t = aligned / hop;
            max_audio_samples = @max(max_audio_samples, aligned);
        }

        var specs: std.ArrayList(presentation.VisualSpec) = .empty;
        defer specs.deinit(allocator);
        for (visuals.items) |item| {
            try specs.append(allocator, .{
                .kind = item.kind,
                .merged = item.merged,
                .grid_h = item.grid_h,
                .grid_w = item.grid_w,
                .temporal = item.temporal,
                .timestamps = item.timestamps,
                .has_audio = item.has_audio,
            });
        }
        var assembled = try presentation.assemble(allocator, encode_text, variant, specs.items, prompt);
        errdefer assembled.deinit(allocator);

        if (hasVisual(visuals.items)) {
            if (!visual_enc.ready(visual_store.view())) return error.VisualEncodeMissing;
            if (!vision.ready(enc_store.view())) return error.VisionWeightsMissing;
        }
        if (audios.items.len != 0) {
            if (!audio_vae.encodeReady(audio_store.view())) return error.AudioEncodeMissing;
        }

        var all = shardings.all();
        const session_spans = try sessionSpans(allocator, assembled.spans);
        errdefer allocator.free(session_spans);
        const positions = try allocator.alloc(f32, assembled.tokens.len * 3);
        errdefer allocator.free(positions);
        session_mod.fillEncoderPositions(positions, @intCast(assembled.tokens.len), session_spans);

        var merged_all: std.ArrayList(f32) = .empty;
        errdefer merged_all.deinit(allocator);
        var ds_host: [3][]f32 = .{ &.{}, &.{}, &.{} };
        errdefer for (ds_host) |d| if (d.len != 0) allocator.free(d);
        for (&ds_host) |*d| {
            d.* = try allocator.alloc(f32, assembled.tokens.len * hidden_dim);
            @memset(d.*, 0);
        }

        if (hasVisual(visuals.items)) {
            var loaded_vision = try vision.LoadedModel.init(allocator, io, enc_dir, enc_store.view(), text_hidden);
            defer loaded_vision.deinit(allocator);
            var vision_cache = try vision.WeightCache.load(allocator, io, platform, &loaded_vision, enc_store, &all, progress);
            defer vision_cache.deinit(allocator);
            var compiled_v: ?pipeline.VisionCompiled = null;
            defer if (compiled_v) |*c| c.deinit();
            var span_i: usize = 0;
            for (visuals.items) |item| {
                if (item.kind == .audio) continue;
                if (compiled_v == null or compiled_v.?.seq != item.seq) {
                    if (compiled_v) |*c| {
                        c.deinit();
                        compiled_v = null;
                    }
                    compiled_v = try pipeline.compileVision(allocator, io, platform, loaded_vision.inner, item.seq, shardings, progress);
                }
                const is_video = item.kind == .video or item.kind == .video_audio;
                var encoded = if (is_video) blk: {
                    const vis_frames = item.temporal * 2;
                    const src = if (item.qwen_rgb.len != 0) item.qwen_rgb else item.rgb;
                    break :blk try vision.runVideo(allocator, io, platform, &compiled_v.?, &loaded_vision, &vision_cache, src, vis_frames, item.h, item.w);
                } else try vision.runImage(allocator, io, platform, &compiled_v.?, &loaded_vision, &vision_cache, item.rgb, item.h, item.w);
                defer encoded.deinit(allocator);
                try merged_all.appendSlice(allocator, encoded.merged);
                const block_tokens = item.merged;
                const n_blocks: usize = if (is_video) item.temporal else 1;
                var bi: usize = 0;
                while (bi < n_blocks and span_i < session_spans.len) : (bi += 1) {
                    const span = session_spans[span_i];
                    span_i += 1;
                    for (0..3) |di| {
                        if (encoded.deepstack[di].len != 0) {
                            const src_off = bi * block_tokens * hidden_dim;
                            @memcpy(
                                ds_host[di][@as(usize, span.start) * hidden_dim ..][0 .. span.tokens * hidden_dim],
                                encoded.deepstack[di][src_off..][0 .. span.tokens * hidden_dim],
                            );
                        }
                    }
                }
            }
        }

        const n_visual_enc = countVisual(visuals.items);
        var encoded_visuals = try allocator.alloc(encode_mod.VisualLatent, n_visual_enc);
        var n_vis: usize = 0;
        errdefer {
            for (encoded_visuals[0..n_vis]) |v| v.deinit(allocator);
            allocator.free(encoded_visuals);
        }
        var encoded_audios = try allocator.alloc(encode_mod.AudioLatent, audios.items.len);
        var n_aud_enc: usize = 0;
        errdefer {
            for (encoded_audios[0..n_aud_enc]) |a| a.deinit(allocator);
            allocator.free(encoded_audios);
        }

        if (n_visual_enc != 0 or audios.items.len != 0) {
            const v_loaded: ?visual_enc.LoadedModel = if (n_visual_enc != 0)
                visual_enc.LoadedModel.init(visual_store.view(), loaded_visual.cfg)
            else
                null;
            var v_bufs: ?zml.Bufferized(visual_enc.Model) = if (v_loaded) |m|
                try m.loadBuffers(allocator, io, platform, visual_store, &all, progress)
            else
                null;
            defer if (v_bufs) |*b| visual_enc.Model.unloadBuffers(b);

            var a_loaded: ?audio_vae.LoadedEncoder = if (audios.items.len != 0)
                audio_vae.LoadedEncoder.init(audio_store.view(), loaded_audio.cfg)
            else
                null;
            var a_bufs: ?zml.Bufferized(audio_vae.EncoderModel) = if (a_loaded) |*m|
                try m.loadBuffers(allocator, io, platform, audio_store, &all, progress)
            else
                null;
            defer if (a_bufs) |*b| audio_vae.EncoderModel.unloadBuffers(b);

            const tile = encodeTileSize(visuals.items, vae.official_visual.tile_px);
            var compiled_e = try pipeline.compileEncode(
                allocator,
                io,
                platform,
                if (v_loaded) |m| m.inner else null,
                if (a_loaded) |m| m.inner else null,
                tile.h,
                tile.w,
                hasVideo(visuals.items),
                if (max_audio_samples == 0) hop else max_audio_samples,
                shardings,
                progress,
            );
            defer compiled_e.deinit();

            for (visuals.items) |item| {
                if (item.kind == .audio) continue;
                const policy = config_mod.posterior;
                encoded_visuals[n_vis] = if (item.kind == .video or item.kind == .video_audio)
                    try encode_mod.encodeVideo(allocator, io, platform, &compiled_e, &v_loaded.?, &v_bufs.?, item.nchw.?, item.frames, item.h, item.w, policy)
                else
                    try encode_mod.encodeKeyframe(allocator, io, platform, &compiled_e, &v_loaded.?, &v_bufs.?, item.nchw.?, item.h, item.w, policy);
                encoded_visuals[n_vis].keyframe_index = item.keyframe_index;
                encoded_visuals[n_vis].guide_frame = item.guide_frame;
                n_vis += 1;
            }
            for (audios.items, encoded_audios) |item, *out| {
                const padded = try padStereo(allocator, item.stereo, max_audio_samples);
                defer allocator.free(padded);
                out.* = try encode_mod.encodeAudio(allocator, io, platform, &compiled_e, &a_loaded.?, &a_bufs.?, padded);
                n_aud_enc += 1;
            }
        }

        const conds = try encode_mod.packConditions(allocator, encoded_visuals[0..n_vis], encoded_audios[0..n_aud_enc], blocks.items, patch);
        errdefer conds.deinit(allocator);
        if (try session.openDumpDir(io)) |dump_dir| {
            defer dump_dir.close(io);
            if (n_vis != 0) {
                const v0 = encoded_visuals[0];
                try session.dumpHostF32(io, dump_dir, "condition_latents", v0.thwc, &.{
                    @intCast(v0.latent_t),
                    @intCast(v0.latent_h),
                    @intCast(v0.latent_w),
                    24,
                });
            }
            if (conds.video_patches.len != 0) {
                try session.dumpHostF32(io, dump_dir, "condition_rows_clean", conds.video_patches, &.{
                    @intCast(conds.video_patches.len / 96),
                    96,
                });
            }
        }
        for (encoded_visuals[0..n_vis]) |v| v.deinit(allocator);
        allocator.free(encoded_visuals);
        for (encoded_audios[0..n_aud_enc]) |a| a.deinit(allocator);
        allocator.free(encoded_audios);

        const merged_out: ?[]f32 = if (merged_all.items.len == 0) null else try merged_all.toOwnedSlice(allocator);
        errdefer if (merged_out) |m| allocator.free(m);
        allocator.free(assembled.spans);
        log.info(
            "conditions: ok tokens={d} vision_spans={d} video_conds={d} audio_conds={d} refs={d}",
            .{ assembled.tokens.len, session_spans.len, conds.videos.len, conds.audios.len, conds.references.len },
        );
        return .{
            .tokens = assembled.tokens,
            .tags = assembled.tags,
            .positions = positions,
            .deepstack = .{ ds_host[0], ds_host[1], ds_host[2] },
            .vision_merged = merged_out,
            .vision_spans = session_spans,
            .conds = conds,
        };
    }

    fn sessionSpans(allocator: std.mem.Allocator, src: []const presentation.VisionSpan) ![]session_mod.VisionSpan {
        const out = try allocator.alloc(session_mod.VisionSpan, src.len);
        for (src, out) |s, *d| {
            d.* = .{
                .start = s.start,
                .tokens = s.tokens,
                .grid_h = s.grid_h,
                .grid_w = s.grid_w,
                .temporal = s.temporal,
            };
        }
        return out;
    }

    fn hasVisual(items: anytype) bool {
        for (items) |item| {
            if (item.kind != .audio) return true;
        }
        return false;
    }

    fn countVisual(items: anytype) usize {
        var n: usize = 0;
        for (items) |item| {
            if (item.kind != .audio) n += 1;
        }
        return n;
    }

    fn encodeTileSize(items: anytype, tile_px: u32) struct { h: u32, w: u32 } {
        var max_h: u32 = 0;
        var max_w: u32 = 0;
        for (items) |item| {
            if (item.kind == .audio) continue;
            max_h = @max(max_h, item.h);
            max_w = @max(max_w, item.w);
        }
        if (max_h == 0) return .{ .h = tile_px, .w = tile_px };
        return .{ .h = @min(tile_px, max_h), .w = @min(tile_px, max_w) };
    }
};

// --- runtime/decode.zig ---
pub const decode = struct {
    const std = @import("std");

    const zml = @import("zml");

    const audio_vae = @import("vae.zig").audio;
    const vae = @import("vae.zig").geom;
    const visual_vae = @import("vae.zig").visual;
    const weights = @import("model.zig").weights;
    const policy = @import("model.zig").policy;

    const log = std.log.scoped(.minimax_h3_decode);

    fn done(io: std.Io, start: std.Io.Timestamp, comptime msg: []const u8, args: anytype) void {
        log.info(msg ++ " [{f}]", args ++ .{start.untilNow(io, .awake)});
    }

    fn bufferFromItems(io: std.Io, platform: *const zml.Platform, shape: zml.Shape, items: anytype) !zml.Buffer {
        return bufferFromItemsSharded(io, platform, shape, .replicated, items);
    }

    fn bufferFromItemsSharded(io: std.Io, platform: *const zml.Platform, shape: zml.Shape, shard: zml.Sharding, items: anytype) !zml.Buffer {
        return zml.Buffer.fromBytes(io, platform, shape, shard, std.mem.sliceAsBytes(items));
    }

    fn copyLatentTile(
        src: []const f32,
        src_t: u32,
        src_h: u32,
        src_w: u32,
        channels: u32,
        t0: u32,
        h0: u32,
        w0: u32,
        tile: visual_vae.TileShape,
        dst: []f32,
    ) void {
        @memset(dst, 0);
        const copy_t = @min(tile.latent_t, src_t - t0);
        const copy_h = @min(tile.latent_h, src_h - h0);
        const copy_w = @min(tile.latent_w, src_w - w0);
        var tt: u32 = 0;
        while (tt < copy_t) : (tt += 1) {
            var hh: u32 = 0;
            while (hh < copy_h) : (hh += 1) {
                var ww: u32 = 0;
                while (ww < copy_w) : (ww += 1) {
                    const src_i = ((((t0 + tt) * src_h + (h0 + hh)) * src_w + (w0 + ww)) * channels);
                    const dst_i = (((tt * tile.latent_h + hh) * tile.latent_w + ww) * channels);
                    @memcpy(dst[dst_i..][0..channels], src[src_i..][0..channels]);
                }
            }
        }
    }

    const VisualCache = struct {
        embed: zml.Bufferized(visual_vae.EmbedModel),
        blocks: []zml.Bufferized(visual_vae.TransformerBlock),
        finish: zml.Bufferized(visual_vae.FinishModel),

        fn deinit(self: *VisualCache, allocator: std.mem.Allocator) void {
            visual_vae.EmbedModel.unloadBuffers(&self.embed);
            for (self.blocks) |*block| visual_vae.TransformerBlock.unloadBuffers(block);
            allocator.free(self.blocks);
            visual_vae.FinishModel.unloadBuffers(&self.finish);
        }
    };

    const load_window: usize = policy.vae_load_window;

    fn loadVisualEmbed(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        loaded: *const visual_vae.LoadedModel,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(visual_vae.EmbedModel) {
        const now: std.Io.Timestamp = .now(io, .awake);
        const bufs = try loaded.loadEmbed(allocator, io, platform, store, shardings, progress);
        log.info("visual embed: loaded [{f}]", .{now.untilNow(io, .awake)});
        return bufs;
    }

    fn loadVisualFinish(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        loaded: *const visual_vae.LoadedModel,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(visual_vae.FinishModel) {
        const now: std.Io.Timestamp = .now(io, .awake);
        const bufs = try loaded.loadFinish(allocator, io, platform, store, shardings, progress);
        log.info("visual finish: loaded [{f}]", .{now.untilNow(io, .awake)});
        return bufs;
    }

    fn loadVisualBlock(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        loaded: *const visual_vae.LoadedModel,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        index: usize,
        progress: *std.Progress.Node,
        loader: *zml.io.Loader,
    ) !zml.Bufferized(visual_vae.TransformerBlock) {
        const now: std.Io.Timestamp = .now(io, .awake);
        const bufs = try loaded.loadBlock(allocator, io, platform, store, shardings, index, progress, loader);
        log.debug("visual block {d}: loaded [{f}]", .{ index + 1, now.untilNow(io, .awake) });
        return bufs;
    }

    fn loadVisualCache(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        loaded: *const visual_vae.LoadedModel,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        n_blocks: usize,
        progress: *std.Progress.Node,
    ) !VisualCache {
        log.info("visual cache: load embed + finish + {d} blocks (window={d})", .{ n_blocks, load_window });
        var embed_f = try io.concurrent(loadVisualEmbed, .{ allocator, io, platform, loaded, store, shardings, progress });
        var embed_taken = false;
        errdefer if (!embed_taken) {
            if (embed_f.cancel(io)) |bufs_| {
                var bufs = bufs_;
                visual_vae.EmbedModel.unloadBuffers(&bufs);
            } else |_| {}
        };
        var finish_f = try io.concurrent(loadVisualFinish, .{ allocator, io, platform, loaded, store, shardings, progress });
        var finish_taken = false;
        errdefer if (!finish_taken) {
            if (finish_f.cancel(io)) |bufs_| {
                var bufs = bufs_;
                visual_vae.FinishModel.unloadBuffers(&bufs);
            } else |_| {}
        };

        const blocks = try allocator.alloc(zml.Bufferized(visual_vae.TransformerBlock), n_blocks);
        errdefer allocator.free(blocks);
        var filled: usize = 0;
        errdefer {
            for (blocks[0..filled]) |*block| visual_vae.TransformerBlock.unloadBuffers(block);
        }

        var loaders: [load_window]zml.io.Loader = undefined;
        var ready: usize = 0;
        defer for (loaders[0..ready]) |*loader| loader.deinit();
        while (ready < load_window) : (ready += 1) {
            loaders[ready] = try weights.initLoader(allocator, platform);
        }

        var start: usize = 0;
        while (start < n_blocks) {
            const batch = @min(load_window, n_blocks - start);
            var futs: [load_window]@TypeOf(try io.concurrent(loadVisualBlock, .{
                allocator, io, platform, loaded, store, shardings, start, progress, &loaders[0],
            })) = undefined;
            var spawned: usize = 0;
            while (spawned < batch) : (spawned += 1) {
                futs[spawned] = try io.concurrent(loadVisualBlock, .{
                    allocator, io, platform, loaded, store, shardings, start + spawned, progress, &loaders[spawned],
                });
            }
            var got: usize = 0;
            errdefer {
                while (got < spawned) : (got += 1) {
                    if (futs[got].cancel(io)) |bufs_| {
                        var bufs = bufs_;
                        visual_vae.TransformerBlock.unloadBuffers(&bufs);
                    } else |_| {}
                }
            }
            while (got < spawned) : (got += 1) {
                blocks[start + got] = try futs[got].await(io);
                filled += 1;
            }
            start += batch;
        }

        var embed = try embed_f.await(io);
        embed_taken = true;
        errdefer visual_vae.EmbedModel.unloadBuffers(&embed);
        var finish = try finish_f.await(io);
        finish_taken = true;
        errdefer visual_vae.FinishModel.unloadBuffers(&finish);
        return .{
            .embed = embed,
            .blocks = blocks,
            .finish = finish,
        };
    }

    const EmbedRunner = zml.FnExe(visual_vae.embed).Runner(.{.model});
    const BlockRunner = zml.FnExe(visual_vae.TransformerBlock.forward).Runner(.{.layer});
    const FinishRunner = zml.FnExe(visual_vae.finish).Runner(.{.model});

    const VisualRunners = struct {
        embed: EmbedRunner,
        block: BlockRunner,
        finish: FinishRunner,
        pos: zml.Buffer,

        fn deinit(self: *VisualRunners, allocator: std.mem.Allocator) void {
            self.embed.deinit(allocator);
            self.block.deinit(allocator);
            self.finish.deinit(allocator);
            self.pos.deinit();
        }
    };

    fn initVisualRunners(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        compiled: *const pipeline.VaeCompiled,
        loaded: *const visual_vae.LoadedModel,
        cache: *const VisualCache,
    ) !VisualRunners {
        if (cache.blocks.len == 0) return error.VisualBlocksMissing;
        const registers: u32 = @intCast(loaded.cfg.decoder_num_register_tokens);
        const seq = compiled.tile.seq(registers);
        const positions = try visual_vae.hostPositions(allocator, compiled.tile.latent_t, compiled.tile.latent_h, compiled.tile.latent_w, registers);
        defer allocator.free(positions);
        return .{
            .embed = try EmbedRunner.init(&compiled.embed, allocator, .{ .model = cache.embed }),
            .block = try BlockRunner.init(&compiled.block, allocator, .{ .layer = cache.blocks[0] }),
            .finish = try FinishRunner.init(&compiled.finish, allocator, .{ .model = cache.finish }),
            .pos = try bufferFromItems(io, platform, .init(.{ .s = seq, .ax = 3 }, .f32), positions),
        };
    }

    fn runVisualBatch(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        compiled: *const pipeline.VaeCompiled,
        loaded: *const visual_vae.LoadedModel,
        cache: *const VisualCache,
        runners: *VisualRunners,
        packed_latents: []const f32,
        shardings: []const zml.Sharding,
    ) ![]f32 {
        const tile = compiled.tile;
        const batch = compiled.tile_batch;
        var latent_shape: zml.Shape = .init(.{
            .b = batch,
            .s = tile.tokens(),
            .d = loaded.cfg.latent_channels,
        }, .f32);
        const latent_sharding: zml.Sharding = if (compiled.partition_b) blk: {
            latent_shape = latent_shape.withPartitioning(.{ .b = .model });
            break :blk shardings[0];
        } else .replicated;
        var latent_buf = try bufferFromItemsSharded(io, platform, latent_shape, latent_sharding, packed_latents);
        defer latent_buf.deinit();

        var hidden: zml.Buffer = undefined;
        var cos: zml.Buffer = undefined;
        var sin: zml.Buffer = undefined;
        var t: std.Io.Timestamp = .now(io, .awake);
        runners.embed.run(io, .{
            .inputs = .{ .latents = latent_buf, .position_ids = runners.pos },
            .outputs = .{ .hidden = &hidden, .cos = &cos, .sin = &sin },
            .opts = .{ .wait = true },
        });
        defer hidden.deinit();
        defer cos.deinit();
        defer sin.deinit();
        log.debug("visual embed: ran {f} [{f}]", .{ hidden.shape(), t.untilNow(io, .awake) });

        var i: usize = 0;
        while (i < cache.blocks.len) : (i += 1) {
            weights.rebake(&runners.block, .{ .layer = cache.blocks[i] });
            var next: zml.Buffer = undefined;
            t = .now(io, .awake);
            runners.block.run(io, .{
                .inputs = .{ .hidden = hidden, .cos = cos, .sin = sin },
                .outputs = .{ .hidden = &next },
                .opts = .{ .wait = true },
            });
            hidden.deinit();
            hidden = next;
            log.debug("visual block {d}/{d}: ran [{f}]", .{ i + 1, cache.blocks.len, t.untilNow(io, .awake) });
        }

        var patches: zml.Buffer = undefined;
        t = .now(io, .awake);
        runners.finish.run(io, .{
            .inputs = .{ .hidden = hidden },
            .outputs = .{ .patches = &patches },
            .opts = .{ .wait = true },
        });
        defer patches.deinit();

        const patch_dim: usize = @intCast(loaded.cfg.out_channels * 4 * 16 * 16);
        const host = try allocator.alloc(f32, @as(usize, batch) * tile.tokens() * patch_dim);
        errdefer allocator.free(host);
        try patches.toSlice(io, .init(patches.shape(), std.mem.sliceAsBytes(host)));
        done(io, t, "visual finish: toSlice ok", .{});
        return host;
    }

    pub fn decodeVideo(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        compiled: *const pipeline.VaeCompiled,
        loaded: *const visual_vae.LoadedModel,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        geo: pipeline.Geometry,
        video_thwc: []f32,
        progress: *std.Progress.Node,
    ) ![]f32 {
        const decode_start: std.Io.Timestamp = .now(io, .awake);
        const cfg = loaded.cfg;
        const spec = cfg.spec();
        vae.applyLatentNorm(video_thwc, @intCast(cfg.latent_channels), &cfg.latents_mean, &cfg.latents_std, true);
        log.info("visual decode: start {d}x{d} frames={d} latents {d}x{d}x{d}", .{
            geo.pixel_w,
            geo.pixel_h,
            geo.frames,
            geo.latent_t,
            geo.latent_h,
            geo.latent_w,
        });

        const channels: u32 = @intCast(cfg.latent_channels);
        const tile = compiled.tile;
        const y_plan = try vae.splitTiles(allocator, geo.pixel_h, spec.tile_px, spec.tile_overlap_px, spec.spatial);
        defer y_plan.deinit(allocator);
        const x_plan = try vae.splitTiles(allocator, geo.pixel_w, spec.tile_px, spec.tile_overlap_px, spec.spatial);
        defer x_plan.deinit(allocator);

        const chunk = spec.tokensChunkSize();
        const pad = vae.tokenDropPad(spec, geo.latent_t);
        const padded_t = geo.latent_t + pad;
        const padded = try allocator.alloc(f32, padded_t * geo.latent_h * geo.latent_w * channels);
        defer allocator.free(padded);
        const src_n = geo.latent_t * geo.latent_h * geo.latent_w * channels;
        @memcpy(padded[0..src_n], video_thwc[0..src_n]);
        if (pad > 0) {
            const last = padded[(geo.latent_t - 1) * geo.latent_h * geo.latent_w * channels ..][0 .. geo.latent_h * geo.latent_w * channels];
            var p: u32 = 0;
            while (p < pad) : (p += 1) {
                @memcpy(padded[(geo.latent_t + p) * geo.latent_h * geo.latent_w * channels ..][0..last.len], last);
            }
        }

        const num_tokens = geo.latent_t + spec.token_drop;
        const num_chunks = (num_tokens + pad) / chunk - @intFromBool(spec.token_drop > 0);
        const chunk_frames = chunk * spec.temporal;
        const pre = spec.framePrePadding();
        const frame_overlap = spec.frameOverlap();

        const out_frames = geo.frames;
        const out = try allocator.alloc(f32, 3 * out_frames * geo.pixel_h * geo.pixel_w);
        errdefer allocator.free(out);
        @memset(out, 0);

        var cache = try loadVisualCache(
            allocator,
            io,
            platform,
            loaded,
            store,
            shardings,
            loaded.inner.blocks.len,
            progress,
        );
        defer cache.deinit(allocator);
        var runners = try initVisualRunners(allocator, io, platform, compiled, loaded, &cache);
        defer runners.deinit(allocator);

        const plane = geo.pixel_h * geo.pixel_w;
        const overlap_n = 3 * frame_overlap * plane;
        const pending = try allocator.alloc(f32, overlap_n);
        defer allocator.free(pending);
        var has_overlap = false;

        var written: u32 = 0;
        var chunk_i: u32 = 0;
        while (chunk_i < num_chunks) : (chunk_i += 1) {
            const start_t = chunk_i * chunk;
            log.info("visual chunk {d}/{d} t0={d}", .{ chunk_i + 1, num_chunks, start_t });
            const tile_n = tile.tokens() * channels;
            const n_tiles: u32 = @intCast(y_plan.count() * x_plan.count());
            const tile_lats = try allocator.alloc(f32, n_tiles * tile_n);
            defer allocator.free(tile_lats);
            const jobs = try allocator.alloc(struct { y0: u32, x0: u32, ylen: u32, xlen: u32, yi: usize, xi: usize }, n_tiles);
            defer allocator.free(jobs);
            var job_i: usize = 0;
            for (y_plan.starts, y_plan.lengths, 0..) |y0, ylen, yi| {
                for (x_plan.starts, x_plan.lengths, 0..) |x0, xlen, xi| {
                    copyLatentTile(
                        padded,
                        padded_t,
                        geo.latent_h,
                        geo.latent_w,
                        channels,
                        start_t,
                        y0 / spec.spatial,
                        x0 / spec.spatial,
                        tile,
                        tile_lats[job_i * tile_n ..][0..tile_n],
                    );
                    jobs[job_i] = .{ .y0 = y0, .x0 = x0, .ylen = ylen, .xlen = xlen, .yi = yi, .xi = xi };
                    job_i += 1;
                }
            }

            const clip_t = tile.latent_t * spec.temporal;
            const clip = try allocator.alloc(f32, 3 * clip_t * geo.pixel_h * geo.pixel_w);
            defer allocator.free(clip);
            @memset(clip, 0);

            const batch = @max(1, compiled.tile_batch);
            const packed_lat = try allocator.alloc(f32, batch * tile_n);
            defer allocator.free(packed_lat);
            const patch_dim: usize = @intCast(loaded.cfg.out_channels * 4 * 16 * 16);
            const tile_patch = tile.tokens() * patch_dim;
            var off: usize = 0;
            while (off < jobs.len) {
                @memset(packed_lat, 0);
                const take = @min(batch, @as(u32, @intCast(jobs.len - off)));
                var b: u32 = 0;
                while (b < take) : (b += 1) {
                    @memcpy(packed_lat[b * tile_n ..][0..tile_n], tile_lats[(off + b) * tile_n ..][0..tile_n]);
                }
                const patches = try runVisualBatch(allocator, io, platform, compiled, loaded, &cache, &runners, packed_lat, shardings);
                defer allocator.free(patches);
                b = 0;
                while (b < take) : (b += 1) {
                    const job = jobs[off + b];
                    const pix = try visual_vae.unpackPatches(
                        allocator,
                        patches[b * tile_patch ..][0..tile_patch],
                        tile.latent_t,
                        tile.latent_h,
                        tile.latent_w,
                        spec.temporal,
                        spec.spatial,
                        3,
                    );
                    defer allocator.free(pix);
                    const blend_y: u32 = if (job.yi == 0) 0 else y_plan.overlaps[job.yi - 1];
                    const blend_x: u32 = if (job.xi == 0) 0 else x_plan.overlaps[job.xi - 1];
                    pasteNchw(clip, clip_t, geo.pixel_h, geo.pixel_w, pix, clip_t, job.ylen, job.xlen, job.y0, job.x0, blend_y, blend_x);
                }
                off += take;
            }

            // Each clip is two `chunk_frames` slices: write the first (minus pre-pad);
            // hold the second as the next overlap and append it after the last chunk.
            const take = @min(chunk_frames - pre, out_frames - written);
            var f: u32 = 0;
            while (f < take) : (f += 1) {
                if (has_overlap and f < frame_overlap) {
                    const w = @as(f32, @floatFromInt(f)) / @as(f32, @floatFromInt(frame_overlap));
                    var c: u32 = 0;
                    while (c < 3) : (c += 1) {
                        var p: usize = 0;
                        while (p < plane) : (p += 1) {
                            const oi = ((c * out_frames + written + f) * plane) + p;
                            const ci = ((c * clip_t + pre + f) * plane) + p;
                            const pi = ((c * frame_overlap + f) * plane) + p;
                            out[oi] = pending[pi] * (1.0 - w) + clip[ci] * w;
                        }
                    }
                } else {
                    var c: u32 = 0;
                    while (c < 3) : (c += 1) {
                        const oi = (c * out_frames + written + f) * plane;
                        const ci = (c * clip_t + pre + f) * plane;
                        @memcpy(out[oi..][0..plane], clip[ci..][0..plane]);
                    }
                }
            }
            written += take;

            const overlap_src = chunk_frames + pre;
            if (frame_overlap > 0 and overlap_src < clip_t) {
                const avail = @min(frame_overlap, clip_t - overlap_src);
                var c: u32 = 0;
                while (c < 3) : (c += 1) {
                    var of: u32 = 0;
                    while (of < avail) : (of += 1) {
                        const si = (c * clip_t + overlap_src + of) * plane;
                        const di = (c * frame_overlap + of) * plane;
                        @memcpy(pending[di..][0..plane], clip[si..][0..plane]);
                    }
                }
                has_overlap = true;
            }
            if (written >= out_frames) break;
        }
        if (has_overlap and written < out_frames) {
            const take = @min(frame_overlap, out_frames - written);
            var c: u32 = 0;
            while (c < 3) : (c += 1) {
                var f: u32 = 0;
                while (f < take) : (f += 1) {
                    const oi = (c * out_frames + written + f) * plane;
                    const pi = (c * frame_overlap + f) * plane;
                    @memcpy(out[oi..][0..plane], pending[pi..][0..plane]);
                }
            }
            written += take;
        }

        vae.denormImagenetRgb(out);
        log.info("visual decode: ok frames={d} [{f}]", .{ out_frames, decode_start.untilNow(io, .awake) });
        return out;
    }

    fn pasteNchw(
        dst: []f32,
        dst_t: u32,
        dst_h: u32,
        dst_w: u32,
        src: []const f32,
        src_t: u32,
        src_h: u32,
        src_w: u32,
        y0: u32,
        x0: u32,
        blend_y: u32,
        blend_x: u32,
    ) void {
        const copy_t = @min(dst_t, src_t);
        const copy_h = @min(src_h, dst_h - y0);
        const copy_w = @min(src_w, dst_w - x0);
        if (blend_y == 0 and blend_x == 0) {
            var c: u32 = 0;
            while (c < 3) : (c += 1) {
                var t: u32 = 0;
                while (t < copy_t) : (t += 1) {
                    var y: u32 = 0;
                    while (y < copy_h) : (y += 1) {
                        const si = (((c * src_t + t) * src_h + y) * src_w);
                        const di = (((c * dst_t + t) * dst_h + (y0 + y)) * dst_w + x0);
                        @memcpy(dst[di..][0..copy_w], src[si..][0..copy_w]);
                    }
                }
            }
            return;
        }
        var c: u32 = 0;
        while (c < 3) : (c += 1) {
            var t: u32 = 0;
            while (t < copy_t) : (t += 1) {
                var y: u32 = 0;
                while (y < copy_h) : (y += 1) {
                    var x: u32 = 0;
                    while (x < copy_w) : (x += 1) {
                        const si = (((c * src_t + t) * src_h + y) * src_w + x);
                        const di = (((c * dst_t + t) * dst_h + (y0 + y)) * dst_w + (x0 + x));
                        var w: f32 = 1.0;
                        if (blend_y > 0 and y < blend_y) {
                            w *= @as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(blend_y));
                        }
                        if (blend_x > 0 and x < blend_x) {
                            w *= @as(f32, @floatFromInt(x)) / @as(f32, @floatFromInt(blend_x));
                        }
                        dst[di] = dst[di] * (1.0 - w) + src[si] * w;
                    }
                }
            }
        }
    }

    pub fn decodeAudio(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        compiled: *pipeline.VaeCompiled,
        loaded: *const audio_vae.LoadedModel,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        geo: pipeline.Geometry,
        packed_audio: []f32,
        progress: *std.Progress.Node,
    ) ![]f32 {
        vae.applyLatentNorm(packed_audio, @intCast(loaded.cfg.latent_channels), &loaded.cfg.latents_mean, &loaded.cfg.latents_std, true);
        const channels: u32 = @intCast(loaded.cfg.latent_channels);
        const t = geo.audio_t;
        const batch = try allocator.alloc(f32, 2 * @as(usize, channels) * t);
        defer allocator.free(batch);
        vae.audioRowsToBct(batch, packed_audio, channels, t);

        log.info("audio: load", .{});
        var clock: std.Io.Timestamp = .now(io, .awake);
        var bufs = try loaded.loadBuffers(allocator, io, platform, store, shardings, progress);
        defer audio_vae.Model.unloadBuffers(&bufs, allocator);
        done(io, clock, "audio: loaded", .{});
        const audio_exe: *zml.FnExe(audio_vae.decode) = if (compiled.audio) |*exe| exe else return error.AudioDecodeMissing;
        var runner = try zml.FnExe(audio_vae.decode).Runner(.{.model}).init(audio_exe, allocator, .{ .model = bufs });
        defer runner.deinit(allocator);

        var latent_buf = try bufferFromItems(io, platform, .init(.{
            .b = 2,
            .c = loaded.cfg.latent_channels,
            .t = geo.audio_t,
        }, .f32), batch);
        defer latent_buf.deinit();

        var wav: zml.Buffer = undefined;
        clock = .now(io, .awake);
        log.info("audio: run t={d}", .{geo.audio_t});
        runner.run(io, .{
            .inputs = .{ .latents = latent_buf },
            .outputs = .{ .wav = &wav },
            .opts = .{ .wait = true },
        });
        defer wav.deinit();
        done(io, clock, "audio: ran {f}", .{wav.shape()});

        const samples = vae.official_audio.sampleCount(geo.audio_t);
        const host = try allocator.alloc(f32, 2 * samples);
        errdefer allocator.free(host);
        clock = .now(io, .awake);
        log.info("audio: toSlice samples={d}", .{samples});
        try wav.toSlice(io, .init(zml.Shape.init(.{ .b = 2, .c = 1, .t = samples }, .f32), std.mem.sliceAsBytes(host)));
        done(io, clock, "audio: toSlice ok", .{});

        const interleaved = try media.interleaveStereo(allocator, host[0..samples], host[samples..]);
        allocator.free(host);
        return interleaved;
    }

    pub fn writeOutputs(
        allocator: std.mem.Allocator,
        io: std.Io,
        dir: std.Io.Dir,
        out_path: []const u8,
        mp4_name: []const u8,
        geo: pipeline.Geometry,
        rgb_nchw: []const f32,
        stereo: []const f32,
    ) !void {
        const pcm = try media.f32ToS16(allocator, stereo);
        defer allocator.free(pcm);
        const muxed = try media.writeGeneratedVideo(
            allocator,
            io,
            dir,
            out_path,
            mp4_name,
            rgb_nchw,
            geo.frames,
            geo.pixel_h,
            geo.pixel_w,
            pcm,
            configSampleRate(),
        );
        if (muxed) {
            log.info("wrote {d}x{d} {d} frames → {s}/{s}", .{
                geo.pixel_w,
                geo.pixel_h,
                geo.frames,
                out_path,
                mp4_name,
            });
        } else {
            log.info("wrote {d}x{d} frames/ + audio.wav out={s} (ffmpeg missing)", .{
                geo.pixel_w,
                geo.pixel_h,
                out_path,
            });
        }
    }

    fn configSampleRate() u32 {
        return vae.official_audio.sample_rate;
    }
};

// --- runtime/pipeline.zig ---
pub const pipeline = struct {
    const std = @import("std");

    const zml = @import("zml");

    const audio_vae = @import("vae.zig").audio;
    const config_mod = @import("model.zig").config;
    const dit = @import("model.zig").dit;
    const encoder = @import("model.zig").encoder;
    const packing = @import("model.zig").packing;
    const policy = @import("model.zig").policy;
    const sharding_mod = @import("generate.zig").sharding;
    const scheduler_mod = @import("model.zig").scheduler;
    const multistep = @import("model.zig").multistep;
    const vae = @import("vae.zig").geom;
    const vision = @import("model.zig").vision;
    const visual_enc = @import("vae.zig").visual_enc;
    const visual_vae = @import("vae.zig").visual;

    const log = std.log.scoped(.minimax_h3);

    pub const Options = struct {
        variant: config_mod.Variant = .t2va,
        duration_s: f32 = 5.0,
        aspect: config_mod.Aspect = .@"16:9",
        short_side: u32 = config_mod.default_short_side,
        steps: u32 = 30,
        seed: u64 = 0,
        video_shift: f32 = config_mod.video_shift,
        audio_shift: f32 = config_mod.audio_shift,
    };

    pub const Geometry = struct {
        pixel_w: u32,
        pixel_h: u32,
        frames: u32,
        latent_t: u32,
        latent_h: u32,
        latent_w: u32,
        audio_t: u32,
        video_tokens: u32,
        audio_tokens: u32,
        target_video_tokens: u32,
        target_audio_tokens: u32,
        video_patch_dim: u32,
        audio_dim: u32,

        pub fn init(opts: Options, dit_cfg: config_mod.Config) Geometry {
            const px = config_mod.pixelSize(opts.aspect, opts.short_side);
            const frames = config_mod.alignFrameCount(config_mod.frameCount(opts.duration_s));
            const lat = config_mod.visualLatentSize(px.h, px.w, frames);
            const audio_t = config_mod.audioLatentLength(opts.duration_s);
            const vt = config_mod.videoTokenCount(lat.t, lat.h, lat.w, dit_cfg.patch_size);
            const at = vae.official_audio.tokenCount(audio_t);
            return .{
                .pixel_w = px.w,
                .pixel_h = px.h,
                .frames = frames,
                .latent_t = lat.t,
                .latent_h = lat.h,
                .latent_w = lat.w,
                .audio_t = audio_t,
                .video_tokens = vt,
                .audio_tokens = at,
                .target_video_tokens = vt,
                .target_audio_tokens = at,
                .video_patch_dim = @intCast(dit_cfg.videoPatchDim()),
                .audio_dim = @intCast(dit_cfg.audio_in_channels),
            };
        }

        pub fn withConditions(self: Geometry, extra_video: u32, extra_audio: u32) Geometry {
            var out = self;
            out.video_tokens = self.target_video_tokens + extra_video;
            out.audio_tokens = self.target_audio_tokens + extra_audio;
            return out;
        }
    };

    pub const CompilePolicy = struct {
        attention: policy.AttnKind = .vanilla,
        group_size: u32 = 1,
        steps: u32,
        hold_video: i64 = 0,
        hold_audio: i64 = 0,
        vision_tokens: u32 = 0,
    };

    pub const Compiled = struct {
        prepare_text: zml.FnExe(dit.prepareText),
        prepare_rope: zml.FnExe(dit.prepareRope),
        embed_patches: zml.FnExe(dit.embedPatches),
        prepare_temb: zml.FnExe(dit.prepareTemb),
        prepare_adaln: zml.FnExe(dit.prepareAdaln),
        prepare_final_adaln: zml.FnExe(dit.prepareAdaln),
        block: zml.FnExe(dit.stepBlock),
        block_group: ?zml.FnExe(dit.BlockGroup.forward) = null,
        group_size: u32 = 1,
        finish: zml.FnExe(dit.finish),
        apply_video: zml.FnExe(multistep.apply),
        apply_audio: zml.FnExe(multistep.apply),
        encode_embed: zml.FnExe(encoder.EmbedTokens.forward),
        encode_layer: zml.FnExe(encoder.TransformerLayer.forward),
        encode_scatter: ?zml.FnExe(dit.scatterRows) = null,

        pub fn deinit(self: *Compiled) void {
            self.prepare_text.deinit();
            self.prepare_rope.deinit();
            self.embed_patches.deinit();
            self.prepare_temb.deinit();
            self.prepare_adaln.deinit();
            self.prepare_final_adaln.deinit();
            self.block.deinit();
            if (self.block_group) |*g| g.deinit();
            self.finish.deinit();
            self.apply_video.deinit();
            self.apply_audio.deinit();
            self.encode_embed.deinit();
            self.encode_layer.deinit();
            if (self.encode_scatter) |*s| s.deinit();
        }
    };

    const CompileCtx = struct {
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        shardings: []const zml.Sharding,
        progress: *std.Progress.Node,
    };

    fn compileLogged(
        comptime function: anytype,
        comptime name: []const u8,
        ctx: CompileCtx,
        args: std.meta.ArgsTuple(@TypeOf(function)),
    ) !zml.FnExe(function) {
        ctx.progress.increaseEstimatedTotalItems(1);
        const now: std.Io.Timestamp = .now(ctx.io, .awake);
        const exe = try zml.FnExe(function).compile(ctx.allocator, ctx.io, ctx.platform, .{
            .shardings = ctx.shardings,
            .program_name = name,
        }, args);
        log.info("compile {s}: ok [{f}]", .{ name, now.untilNow(ctx.io, .awake) });
        return exe;
    }

    fn compilePrepareText(ctx: CompileCtx, dit_model: dit.Model, enc_dt: zml.DataType, text_len: u32) !zml.FnExe(dit.prepareText) {
        return compileLogged(dit.prepareText, "minimax_h3_prepare_text", ctx, .{.{
            .model = dit_model.textPrep(),
            .text = .init(.{ .b = 1, .s = text_len, .d = dit_model.cfg.text_dim }, enc_dt),
        }});
    }

    fn compilePrepareRope(ctx: CompileCtx, dit_model: dit.Model, seq_len: u32, out_dt: zml.DataType) !zml.FnExe(dit.prepareRope) {
        return compileLogged(dit.prepareRope, "minimax_h3_prepare_rope", ctx, .{.{
            .model = .{
                .rope_freq_dim = dit_model.cfg.rope_freq_dim,
                .rope_theta = dit_model.cfg.rope_theta,
                .out_dtype = out_dt,
            },
            .position_ids = .init(.{ .s = seq_len, .ax = 3 }, .f32),
        }});
    }

    fn compileEmbedPatches(ctx: CompileCtx, dit_model: dit.Model, geo: Geometry, text_len: u32, seq_len: u32, text_dt: zml.DataType) !zml.FnExe(dit.embedPatches) {
        var part = dit_model.patchEmbed();
        part.seq = seq_len;
        return compileLogged(dit.embedPatches, "minimax_h3_embed_patches", ctx, .{.{
            .model = part,
            .video = .init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32),
            .audio = .init(.{ .b = 1, .s = geo.audio_tokens, .d = geo.audio_dim }, .f32),
            .text = .init(.{ .b = 1, .s = text_len, .d = dit_model.cfg.hidden_size }, text_dt),
            .video_indices = .init(.{ .s = geo.video_tokens }, .u32),
            .audio_indices = .init(.{ .s = geo.audio_tokens }, .u32),
            .text_indices = .init(.{ .s = text_len }, .u32),
        }});
    }

    fn compilePrepareTemb(ctx: CompileCtx, dit_model: dit.Model, n_slots: u32) !zml.FnExe(dit.prepareTemb) {
        return compileLogged(dit.prepareTemb, "minimax_h3_prepare_temb", ctx, .{.{
            .model = dit_model.time_embedder,
            .timestep = .init(.{ .n = n_slots }, .f32),
            .freq_dim = dit_model.cfg.freq_dim,
        }});
    }

    fn compilePrepareBlockAdaln(ctx: CompileCtx, dit_model: dit.Model, n_slots: u32, steps: u32) !zml.FnExe(dit.prepareAdaln) {
        return compileLogged(dit.prepareAdaln, "minimax_h3_prepare_adaln", ctx, .{.{
            .model = .{
                .adaln = dit_model.blocks[0].adaln,
                .steps = steps,
                .slots = packing.timestep_slot_count,
            },
            .temb = .init(.{ .n = n_slots, .d = dit_model.time_embedder.outDim() }, .f32),
        }});
    }

    fn compilePrepareFinalAdaln(ctx: CompileCtx, dit_model: dit.Model, n_slots: u32, steps: u32) !zml.FnExe(dit.prepareAdaln) {
        return compileLogged(dit.prepareAdaln, "minimax_h3_prepare_final_adaln", ctx, .{.{
            .model = .{
                .adaln = dit_model.final_layer.adaln,
                .steps = steps,
                .slots = packing.timestep_slot_count,
            },
            .temb = .init(.{ .n = n_slots, .d = dit_model.time_embedder.outDim() }, .f32),
        }});
    }

    fn compileDitBlock(ctx: CompileCtx, dit_model: dit.Model, seq_len: u32, steps: u32) !zml.FnExe(dit.stepBlock) {
        const dt = dit_model.blocks[0].norm1.weight.dtype();
        const table = zml.Tensor.init(.{
            .t = steps,
            .n = packing.timestep_slot_count,
            .mod = config_mod.modality_count,
            .k = 6,
            .d = dit_model.cfg.hidden_size,
        }, dt);
        return compileLogged(dit.stepBlock, "minimax_h3_block", ctx, .{.{
            .layer = dit_model.blocks[0].corePart(),
            .hidden = zml.Tensor.init(.{ .b = 1, .s = seq_len, .d = dit_model.cfg.hidden_size }, dt),
            .table = table,
            .step = zml.Tensor.init(.{}, .u32),
            .adaln_indices = zml.Tensor.init(.{ .s = seq_len }, .u32),
            .cos = zml.Tensor.init(.{ .s = seq_len, .f = dit_model.cfg.rotaryDim() }, dt),
            .sin = zml.Tensor.init(.{ .s = seq_len, .f = dit_model.cfg.rotaryDim() }, dt),
        }});
    }

    fn compileDitGroup(ctx: CompileCtx, dit_model: dit.Model, seq_len: u32, steps: u32, group_size: u32) !zml.FnExe(dit.BlockGroup.forward) {
        const dt = dit_model.blocks[0].norm1.weight.dtype();
        const n: usize = group_size;
        const layers = try ctx.allocator.alloc(dit.BlockCore, n);
        defer ctx.allocator.free(layers);
        const tables = try ctx.allocator.alloc(zml.Tensor, n);
        defer ctx.allocator.free(tables);
        if (dit_model.blocks.len < n) return error.DitGroupTooLarge;
        for (layers, tables, 0..) |*layer, *tab, i| {
            layer.* = dit_model.blocks[i].corePart();
            tab.* = zml.Tensor.init(.{
                .t = steps,
                .n = packing.timestep_slot_count,
                .mod = config_mod.modality_count,
                .k = 6,
                .d = dit_model.cfg.hidden_size,
            }, dt);
        }
        return compileLogged(dit.BlockGroup.forward, "minimax_h3_block_group", ctx, .{.{
            .group = .{ .layers = layers },
            .hidden = zml.Tensor.init(.{ .b = 1, .s = seq_len, .d = dit_model.cfg.hidden_size }, dt),
            .tables = tables,
            .step = zml.Tensor.init(.{}, .u32),
            .adaln_indices = zml.Tensor.init(.{ .s = seq_len }, .u32),
            .cos = zml.Tensor.init(.{ .s = seq_len, .f = dit_model.cfg.rotaryDim() }, dt),
            .sin = zml.Tensor.init(.{ .s = seq_len, .f = dit_model.cfg.rotaryDim() }, dt),
        }});
    }

    fn compileDitFinish(ctx: CompileCtx, dit_model: dit.Model, geo: Geometry, seq_len: u32, steps: u32) !zml.FnExe(dit.finish) {
        const dt = dit_model.blocks[0].norm1.weight.dtype();
        return compileLogged(dit.finish, "minimax_h3_finish", ctx, .{.{
            .model = dit_model.finishCore(),
            .hidden = zml.Tensor.init(.{ .b = 1, .s = seq_len, .d = dit_model.cfg.hidden_size }, dt),
            .table = zml.Tensor.init(.{ .t = steps, .n = packing.timestep_slot_count, .k = 2, .d = dit_model.cfg.hidden_size }, dt),
            .step = zml.Tensor.init(.{}, .u32),
            .timestep_indices = .init(.{ .s = seq_len }, .u32),
            .video_indices = .init(.{ .s = geo.video_tokens }, .u32),
            .audio_indices = .init(.{ .s = geo.audio_tokens }, .u32),
        }});
    }

    fn compileApplyVideo(ctx: CompileCtx, tokens: u32, dim: u32, hold: i64) !zml.FnExe(multistep.apply) {
        return compileLogged(multistep.apply, "minimax_h3_apply_video", ctx, .{.{
            .model = .{ .hold = hold },
            .sample = .init(.{ .b = 1, .s = tokens, .d = dim }, .f32),
            .velocity = .init(.{ .b = 1, .s = tokens, .d = dim }, .f32),
            .sigma = .init(.{}, .f32),
            .sigma_next = .init(.{}, .f32),
            .sigma_t = .init(.{}, .f32),
        }});
    }

    fn compileApplyAudio(ctx: CompileCtx, tokens: u32, dim: u32, hold: i64) !zml.FnExe(multistep.apply) {
        return compileLogged(multistep.apply, "minimax_h3_apply_audio", ctx, .{.{
            .model = .{ .hold = hold },
            .sample = .init(.{ .b = 1, .s = tokens, .d = dim }, .f32),
            .velocity = .init(.{ .b = 1, .s = tokens, .d = dim }, .f32),
            .sigma = .init(.{}, .f32),
            .sigma_next = .init(.{}, .f32),
            .sigma_t = .init(.{}, .f32),
        }});
    }

    fn compileScatter(ctx: CompileCtx, seq: u32, hidden: i64, n: u32, dt: zml.DataType) !zml.FnExe(dit.scatterRows) {
        return compileLogged(dit.scatterRows, "minimax_h3_scatter", ctx, .{.{
            .hidden = .init(.{ .b = 1, .s = seq, .d = hidden }, dt),
            .values = .init(.{ .b = 1, .s = n, .d = hidden }, .f32),
            .indices = .init(.{ .s = n }, .u32),
        }});
    }

    fn compileEncEmbed(ctx: CompileCtx, enc_model: encoder.Model, text_len: u32) !zml.FnExe(encoder.EmbedTokens.forward) {
        return compileLogged(encoder.EmbedTokens.forward, "minimax_h3_encoder_embed", ctx, .{.{
            .embedding = .{ .embed_tokens = enc_model.embed_tokens },
            .tokens = .init(.{ .b = 1, .s = text_len }, .u32),
        }});
    }

    fn compileEncLayer(ctx: CompileCtx, enc_model: encoder.Model, text_len: u32) !zml.FnExe(encoder.TransformerLayer.forward) {
        const dt = enc_model.embed_tokens.weight.dtype();
        const hd = enc_model.cfg.head_dim;
        return compileLogged(encoder.TransformerLayer.forward, "minimax_h3_encoder_layer", ctx, .{.{
            .layer = enc_model.layers[0],
            .hidden = .init(.{ .b = 1, .s = text_len, .d = enc_model.cfg.hidden_size }, dt),
            .cos = .init(.{ .s = text_len, .hd = hd }, dt),
            .sin = .init(.{ .s = text_len, .hd = hd }, dt),
            .visual_delta = .init(.{ .b = 1, .s = text_len, .d = enc_model.cfg.hidden_size }, dt),
        }});
    }

    pub fn compile(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        dit_model: dit.Model,
        enc_model: encoder.Model,
        geo: Geometry,
        text_len: u32,
        seq_len: u32,
        compile_policy: CompilePolicy,
        shardings: sharding_mod.Shardings,
        progress: *std.Progress.Node,
    ) !Compiled {
        var model = dit_model;
        const tp: u32 = @intCast(shardings.model.numPartitionsForLogicalAxis(.model));
        const dit_dt = model.blocks[0].norm1.weight.dtype();
        const refiner_attn = policy.selectAttention(.{
            .target = platform.target,
            .dtype = dit_dt,
            .head_dim = model.cfg.attention_head_dim,
            .heads = model.cfg.num_attention_heads,
            .seq = text_len,
            .causal = false,
            .tp = tp,
        });
        model.applyBackend(compile_policy.attention, refiner_attn);
        var enc_work = enc_model;
        const enc_attn = policy.selectAttention(.{
            .target = platform.target,
            .dtype = enc_model.embed_tokens.weight.dtype(),
            .head_dim = enc_model.cfg.head_dim,
            .heads = enc_model.cfg.num_attention_heads,
            .seq = text_len,
            .causal = true,
            .tp = tp,
        });
        enc_work.applyBackend(enc_attn);

        var all = shardings.all();
        const ctx: CompileCtx = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .shardings = &all,
            .progress = progress,
        };
        const jobs: u32 = 12 + @as(u32, @intFromBool(compile_policy.group_size > 1)) + @as(u32, @intFromBool(compile_policy.vision_tokens > 0));
        var node = progress.start("Compiling MiniMax-H3", jobs);
        defer node.end();

        const dt = dit_dt;
        const enc_dt = enc_model.embed_tokens.weight.dtype();
        const n_flat = compile_policy.steps * packing.timestep_slot_count;
        log.info(
            "compile DiT+encoder: start seq={d} text={d} video_tokens={d} audio_tokens={d} attn={s} group={d} steps={d}",
            .{
                seq_len,
                text_len,
                geo.video_tokens,
                geo.audio_tokens,
                @tagName(compile_policy.attention),
                compile_policy.group_size,
                compile_policy.steps,
            },
        );
        const now: std.Io.Timestamp = .now(io, .awake);

        var text_f = try io.concurrent(compilePrepareText, .{ ctx, model, enc_dt, text_len });
        errdefer if (text_f.cancel(io)) |exe| exe.deinit() else |_| {};
        var rope_f = try io.concurrent(compilePrepareRope, .{ ctx, model, seq_len, dt });
        errdefer if (rope_f.cancel(io)) |exe| exe.deinit() else |_| {};
        var patch_f = try io.concurrent(compileEmbedPatches, .{ ctx, model, geo, text_len, seq_len, dt });
        errdefer if (patch_f.cancel(io)) |exe| exe.deinit() else |_| {};
        var temb_f = try io.concurrent(compilePrepareTemb, .{ ctx, model, n_flat });
        errdefer if (temb_f.cancel(io)) |exe| exe.deinit() else |_| {};
        var adaln_f = try io.concurrent(compilePrepareBlockAdaln, .{ ctx, model, n_flat, compile_policy.steps });
        errdefer if (adaln_f.cancel(io)) |exe| exe.deinit() else |_| {};
        var final_f = try io.concurrent(compilePrepareFinalAdaln, .{ ctx, model, n_flat, compile_policy.steps });
        errdefer if (final_f.cancel(io)) |exe| exe.deinit() else |_| {};
        var block_f = try io.concurrent(compileDitBlock, .{ ctx, model, seq_len, compile_policy.steps });
        errdefer if (block_f.cancel(io)) |exe| exe.deinit() else |_| {};
        var group_f: ?@TypeOf(try io.concurrent(compileDitGroup, .{ ctx, model, seq_len, compile_policy.steps, compile_policy.group_size })) = null;
        if (compile_policy.group_size > 1) {
            group_f = try io.concurrent(compileDitGroup, .{ ctx, model, seq_len, compile_policy.steps, compile_policy.group_size });
        }
        errdefer if (group_f) |*f| if (f.cancel(io)) |exe| exe.deinit() else |_| {};
        var finish_f = try io.concurrent(compileDitFinish, .{ ctx, model, geo, seq_len, compile_policy.steps });
        errdefer if (finish_f.cancel(io)) |exe| exe.deinit() else |_| {};
        var apply_v_f = try io.concurrent(compileApplyVideo, .{ ctx, geo.video_tokens, geo.video_patch_dim, compile_policy.hold_video });
        errdefer if (apply_v_f.cancel(io)) |exe| exe.deinit() else |_| {};
        var apply_a_f = try io.concurrent(compileApplyAudio, .{ ctx, geo.audio_tokens, geo.audio_dim, compile_policy.hold_audio });
        errdefer if (apply_a_f.cancel(io)) |exe| exe.deinit() else |_| {};
        var enc_embed_f = try io.concurrent(compileEncEmbed, .{ ctx, enc_work, text_len });
        errdefer if (enc_embed_f.cancel(io)) |exe| exe.deinit() else |_| {};
        var enc_layer_f = try io.concurrent(compileEncLayer, .{ ctx, enc_work, text_len });
        errdefer if (enc_layer_f.cancel(io)) |exe| exe.deinit() else |_| {};
        var scatter_f: ?@TypeOf(try io.concurrent(compileScatter, .{ ctx, text_len, enc_work.cfg.hidden_size, compile_policy.vision_tokens, enc_dt })) = null;
        if (compile_policy.vision_tokens > 0) {
            scatter_f = try io.concurrent(compileScatter, .{ ctx, text_len, enc_work.cfg.hidden_size, compile_policy.vision_tokens, enc_dt });
        }
        errdefer if (scatter_f) |*f| if (f.cancel(io)) |exe| exe.deinit() else |_| {};

        const prepare_text = try text_f.await(io);
        errdefer prepare_text.deinit();
        const prepare_rope = try rope_f.await(io);
        errdefer prepare_rope.deinit();
        const embed_patches = try patch_f.await(io);
        errdefer embed_patches.deinit();
        const prepare_temb = try temb_f.await(io);
        errdefer prepare_temb.deinit();
        const prepare_adaln = try adaln_f.await(io);
        errdefer prepare_adaln.deinit();
        const prepare_final_adaln = try final_f.await(io);
        errdefer prepare_final_adaln.deinit();
        const block_exe = try block_f.await(io);
        errdefer block_exe.deinit();
        const block_group = if (group_f) |*f| try f.await(io) else null;
        errdefer if (block_group) |exe| {
            var tmp = exe;
            tmp.deinit();
        };
        const finish_exe = try finish_f.await(io);
        errdefer finish_exe.deinit();
        const apply_video = try apply_v_f.await(io);
        errdefer apply_video.deinit();
        const apply_audio = try apply_a_f.await(io);
        errdefer apply_audio.deinit();
        const encode_embed = try enc_embed_f.await(io);
        errdefer encode_embed.deinit();
        const encode_layer = try enc_layer_f.await(io);
        errdefer encode_layer.deinit();
        const encode_scatter = if (scatter_f) |*f| try f.await(io) else null;
        errdefer if (encode_scatter) |exe| {
            var tmp = exe;
            tmp.deinit();
        };

        log.info("Compiled MiniMax-H3 [{f}] seq={d} video_tokens={d} audio_tokens={d} attn={s}", .{
            now.untilNow(io, .awake),
            seq_len,
            geo.video_tokens,
            geo.audio_tokens,
            @tagName(compile_policy.attention),
        });

        return .{
            .prepare_text = prepare_text,
            .prepare_rope = prepare_rope,
            .embed_patches = embed_patches,
            .prepare_temb = prepare_temb,
            .prepare_adaln = prepare_adaln,
            .prepare_final_adaln = prepare_final_adaln,
            .block = block_exe,
            .block_group = block_group,
            .group_size = compile_policy.group_size,
            .finish = finish_exe,
            .apply_video = apply_video,
            .apply_audio = apply_audio,
            .encode_embed = encode_embed,
            .encode_layer = encode_layer,
            .encode_scatter = encode_scatter,
        };
    }

    pub const VaeCompiled = struct {
        embed: zml.FnExe(visual_vae.embed),
        block: zml.FnExe(visual_vae.TransformerBlock.forward),
        finish: zml.FnExe(visual_vae.finish),
        audio: ?zml.FnExe(audio_vae.decode) = null,
        tile: visual_vae.TileShape,
        tile_batch: u32 = 1,
        partition_b: bool = false,

        pub fn deinit(self: *VaeCompiled) void {
            self.embed.deinit();
            self.block.deinit();
            self.finish.deinit();
            if (self.audio) |*exe| exe.deinit();
        }
    };

    fn vaeBatchShape(tags: anytype, dt: zml.DataType, partition_b: bool) zml.Tensor {
        const t = zml.Tensor.init(tags, dt);
        return if (partition_b) t.withPartitioning(.{ .b = .model }) else t;
    }

    fn compileVaeEmbed(ctx: CompileCtx, visual: visual_vae.Model, tile: visual_vae.TileShape, seq: u32, tile_batch: u32, partition_b: bool) !zml.FnExe(visual_vae.embed) {
        return compileLogged(visual_vae.embed, "minimax_h3_vae_embed", ctx, .{.{
            .model = visual.embed,
            .latents = vaeBatchShape(.{ .b = tile_batch, .s = tile.tokens(), .d = visual.cfg.latent_channels }, .f32, partition_b),
            .position_ids = .init(.{ .s = seq, .ax = 3 }, .f32),
        }});
    }

    fn compileVaeBlock(ctx: CompileCtx, visual: visual_vae.Model, seq: u32, tile_batch: u32, partition_b: bool) !zml.FnExe(visual_vae.TransformerBlock.forward) {
        const dt = visual.embed.proj.weight.dtype();
        return compileLogged(visual_vae.TransformerBlock.forward, "minimax_h3_vae_block", ctx, .{.{
            .layer = visual.blocks[0],
            .hidden = vaeBatchShape(.{ .b = tile_batch, .s = seq, .d = visual.cfg.dim() }, dt, partition_b),
            .cos = .init(.{ .s = seq, .f = visual.cfg.rotaryDim() }, dt),
            .sin = .init(.{ .s = seq, .f = visual.cfg.rotaryDim() }, dt),
        }});
    }

    fn compileVaeFinish(ctx: CompileCtx, visual: visual_vae.Model, seq: u32, tile_batch: u32, partition_b: bool) !zml.FnExe(visual_vae.finish) {
        return compileLogged(visual_vae.finish, "minimax_h3_vae_finish", ctx, .{.{
            .model = visual.finish,
            .hidden = vaeBatchShape(.{ .b = tile_batch, .s = seq, .d = visual.cfg.dim() }, visual.embed.proj.weight.dtype(), partition_b),
        }});
    }

    pub fn compileAudioDecode(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        audio: audio_vae.Model,
        geo: Geometry,
        shardings: []const zml.Sharding,
        progress: *std.Progress.Node,
    ) !zml.FnExe(audio_vae.decode) {
        const ctx: CompileCtx = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .shardings = shardings,
            .progress = progress,
        };
        return compileLogged(audio_vae.decode, "minimax_h3_audio_decode", ctx, .{.{
            .model = audio,
            .latents = .init(.{ .b = 2, .c = audio.cfg.latent_channels, .t = geo.audio_t }, .f32),
        }});
    }

    pub fn compileVae(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        visual: visual_vae.Model,
        geo: Geometry,
        tile_batch: u32,
        shardings: sharding_mod.Shardings,
        progress: *std.Progress.Node,
    ) !VaeCompiled {
        var all = shardings.all();
        const ctx: CompileCtx = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .shardings = &all,
            .progress = progress,
        };
        const tile = visual_vae.TileShape.fromGeometry(visual.cfg, geo.latent_t, geo.latent_h, geo.latent_w);
        const registers: u32 = @intCast(visual.cfg.decoder_num_register_tokens);
        const seq = tile.seq(registers);
        const batch = @max(1, tile_batch);
        // Off: `embed` rebuilds `.b` without `.model`, so outputs stay `[batch,…]`
        // while a sharded block/finish would expect `[batch/tp,…]`.
        const partition_b = false;
        var node = progress.start("Compiling MiniMax-H3 VAE", 3);
        defer node.end();

        log.info("compile VAE: start tile={d}x{d}x{d} audio_t={d} batch={d} shard_b={}", .{
            tile.latent_t,
            tile.latent_h,
            tile.latent_w,
            geo.audio_t,
            batch,
            partition_b,
        });
        const now: std.Io.Timestamp = .now(io, .awake);
        var embed_f = try io.concurrent(compileVaeEmbed, .{ ctx, visual, tile, seq, batch, partition_b });
        errdefer if (embed_f.cancel(io)) |exe| exe.deinit() else |_| {};
        var block_f = try io.concurrent(compileVaeBlock, .{ ctx, visual, seq, batch, partition_b });
        errdefer if (block_f.cancel(io)) |exe| exe.deinit() else |_| {};
        var finish_f = try io.concurrent(compileVaeFinish, .{ ctx, visual, seq, batch, partition_b });
        errdefer if (finish_f.cancel(io)) |exe| exe.deinit() else |_| {};

        const embed_exe = try embed_f.await(io);
        errdefer embed_exe.deinit();
        const block_exe = try block_f.await(io);
        errdefer block_exe.deinit();
        const finish_exe = try finish_f.await(io);
        errdefer finish_exe.deinit();

        log.info("Compiled MiniMax-H3 VAE tile={d}x{d}x{d} audio_t={d} [{f}]", .{
            tile.latent_t,
            tile.latent_h,
            tile.latent_w,
            geo.audio_t,
            now.untilNow(io, .awake),
        });

        return .{
            .embed = embed_exe,
            .block = block_exe,
            .finish = finish_exe,
            .tile = tile,
            .tile_batch = batch,
            .partition_b = partition_b,
        };
    }

    pub fn partitionsVaeBatch(batch: u32, tp: u32) bool {
        return batch > 1 and tp > 1 and batch % tp == 0;
    }

    pub fn adalnIndices(allocator: std.mem.Allocator, layout: packing.Layout) ![]u32 {
        const out = try allocator.alloc(u32, layout.seqLen());
        for (out, 0..) |*v, i| v.* = layout.adalnIndex(i);
        return out;
    }

    pub const EncodeCompiled = struct {
        visual_t1: ?zml.FnExe(visual_enc.encode) = null,
        visual_clip: ?zml.FnExe(visual_enc.encode) = null,
        audio: ?zml.FnExe(audio_vae.encode) = null,
        tile_h: u32,
        tile_w: u32,

        pub fn deinit(self: *EncodeCompiled) void {
            if (self.visual_t1) |*c| c.deinit();
            if (self.visual_clip) |*c| c.deinit();
            if (self.audio) |*c| c.deinit();
        }
    };

    fn compileVisualEncode(ctx: CompileCtx, model: visual_enc.Model, t: u32, h: u32, w: u32) !zml.FnExe(visual_enc.encode) {
        return compileLogged(visual_enc.encode, "minimax_h3_visual_encode", ctx, .{.{
            .model = model,
            .pixels = .init(.{ .b = 1, .c = 3, .t = t, .h = h, .w = w }, .f32),
        }});
    }

    fn compileAudioEncode(ctx: CompileCtx, model: audio_vae.EncoderModel, samples: u32) !zml.FnExe(audio_vae.encode) {
        return compileLogged(audio_vae.encode, "minimax_h3_audio_encode", ctx, .{.{
            .model = model,
            .wav = .init(.{ .b = 2, .c = 1, .t = samples }, .f32),
        }});
    }

    pub fn compileEncode(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        visual: ?visual_enc.Model,
        audio: ?audio_vae.EncoderModel,
        tile_h: u32,
        tile_w: u32,
        need_clip: bool,
        audio_samples: u32,
        shardings: sharding_mod.Shardings,
        progress: *std.Progress.Node,
    ) !EncodeCompiled {
        var all = shardings.all();
        const ctx: CompileCtx = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .shardings = &all,
            .progress = progress,
        };
        const t1 = if (visual) |m| try compileVisualEncode(ctx, m, 1, tile_h, tile_w) else null;
        errdefer if (t1) |exe| {
            var tmp = exe;
            tmp.deinit();
        };
        const clip = if (need_clip) blk: {
            const m = visual orelse return error.VisualEncodeMissing;
            break :blk try compileVisualEncode(ctx, m, 17, tile_h, tile_w);
        } else null;
        errdefer if (clip) |exe| {
            var tmp = exe;
            tmp.deinit();
        };
        const audio_exe = if (audio) |m| try compileAudioEncode(ctx, m, audio_samples) else null;
        errdefer if (audio_exe) |exe| {
            var tmp = exe;
            tmp.deinit();
        };
        return .{
            .visual_t1 = t1,
            .visual_clip = clip,
            .audio = audio_exe,
            .tile_h = tile_h,
            .tile_w = tile_w,
        };
    }

    pub const VisionCompiled = struct {
        embed: zml.FnExe(vision.embed),
        block: zml.FnExe(vision.VisionBlock.forward),
        merger: zml.FnExe(vision.Merger.forward),
        deepstack: zml.FnExe(vision.Merger.forward),
        seq: u32,
        merged: u32,

        pub fn deinit(self: *VisionCompiled) void {
            self.embed.deinit();
            self.block.deinit();
            self.merger.deinit();
            self.deepstack.deinit();
        }
    };

    pub fn compileVision(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        model: vision.Model,
        seq: u32,
        shardings: sharding_mod.Shardings,
        progress: *std.Progress.Node,
    ) !VisionCompiled {
        var all = shardings.all();
        const ctx: CompileCtx = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .shardings = &all,
            .progress = progress,
        };
        const cfg = model.cfg;
        const dt = model.embed.proj.weight.dtype();
        const merged: u32 = @intCast(@divExact(@as(i64, seq), cfg.mergeUnit()));
        const embed_exe = try compileLogged(vision.embed, "minimax_h3_vision_embed", ctx, .{.{
            .model = model.embed,
            .patches = .init(.{ .b = 1, .s = seq, .d = cfg.patchIn() }, .f32),
            .pos = .init(.{ .b = 1, .s = seq, .d = cfg.hidden_size }, .f32),
        }});
        errdefer embed_exe.deinit();
        const block_exe = try compileLogged(vision.VisionBlock.forward, "minimax_h3_vision_block", ctx, .{.{
            .layer = model.blocks[0],
            .hidden = .init(.{ .b = 1, .s = seq, .d = cfg.hidden_size }, dt),
            .cos = .init(.{ .s = seq, .hd = cfg.headDim() }, dt),
            .sin = .init(.{ .s = seq, .hd = cfg.headDim() }, dt),
        }});
        errdefer block_exe.deinit();
        const merger_exe = try compileLogged(vision.Merger.forward, "minimax_h3_vision_merger", ctx, .{.{
            .model = model.merger,
            .hidden = .init(.{ .b = 1, .s = seq, .d = cfg.hidden_size }, dt),
        }});
        errdefer merger_exe.deinit();
        const ds_exe = try compileLogged(vision.Merger.forward, "minimax_h3_vision_deepstack", ctx, .{.{
            .model = model.deepstack[0],
            .hidden = .init(.{ .b = 1, .s = seq, .d = cfg.hidden_size }, dt),
        }});
        errdefer ds_exe.deinit();
        return .{
            .embed = embed_exe,
            .block = block_exe,
            .merger = merger_exe,
            .deepstack = ds_exe,
            .seq = seq,
            .merged = merged,
        };
    }

    pub const Packed = struct {
        layout: packing.Layout,
        schedules: scheduler_mod.DualSchedule,

        pub fn deinit(self: *Packed, allocator: std.mem.Allocator) void {
            self.layout.deinit(allocator);
            self.schedules.deinit(allocator);
        }
    };

    pub fn pack(
        allocator: std.mem.Allocator,
        opts: Options,
        geo: Geometry,
        text_len: u32,
        text_tags: []const u8,
        videos: []const packing.ConditionVideo,
        audios: []const packing.ConditionAudio,
        references: []const packing.ReferenceBlock,
    ) !Packed {
        const schedules = try scheduler_mod.DualSchedule.init(allocator, opts.steps, opts.video_shift, opts.audio_shift);
        errdefer schedules.deinit(allocator);
        const video_t = schedules.video.timesteps[0];
        const audio_t = 1.0 - scheduler_mod.timeShiftSigma(1.0 - video_t, opts.video_shift, opts.audio_shift);
        const layout = try packing.build(allocator, .{
            .text_len = text_len,
            .latent_t = geo.latent_t,
            .latent_h = geo.latent_h,
            .latent_w = geo.latent_w,
            .audio_t = geo.audio_t,
            .video_t = video_t,
            .audio_t_noise = audio_t,
            .condition_videos = videos,
            .condition_audios = audios,
            .references = references,
            .text_tags = text_tags,
            .pixel_frames = geo.frames,
        });
        return .{ .layout = layout, .schedules = schedules };
    }

    pub fn describe(opts: Options, geo: Geometry, layout: packing.Layout) void {
        log.info(
            "layout {s} {d}x{d} {d} frames ({d:.1}s) latents {d}x{d}x{d} audio_t={d} seq={d} steps={d} seed={d}",
            .{
                @tagName(opts.variant),
                geo.pixel_w,
                geo.pixel_h,
                geo.frames,
                opts.duration_s,
                geo.latent_t,
                geo.latent_h,
                geo.latent_w,
                geo.audio_t,
                layout.seqLen(),
                opts.steps,
                opts.seed,
            },
        );
    }
};

// --- runtime/session.zig ---
pub const session = struct {
    const std = @import("std");

    const zml = @import("zml");

    const config = @import("model.zig").config;
    const dit = @import("model.zig").dit;
    const encoder = @import("model.zig").encoder;
    const noise = @import("model.zig").noise;
    const packing = @import("model.zig").packing;
    const policy = @import("model.zig").policy;
    const scheduler_mod = @import("model.zig").scheduler;
    const vision = @import("model.zig").vision;
    const weights = @import("model.zig").weights;
    const multistep = @import("model.zig").multistep;

    const log = std.log.scoped(.minimax_h3);

    pub const HostLayout = struct {
        positions: []f32,
        timesteps: []f32,
        timestep_indices: []u32,
        adaln_indices: []u32,
        text_indices: []u32,
        video_indices: []u32,
        audio_indices: []u32,

        pub fn fromLayout(allocator: std.mem.Allocator, layout: packing.Layout, timestep_slots: u32) !HostLayout {
            const positions = try allocator.alloc(f32, layout.positions.len * 3);
            errdefer allocator.free(positions);
            for (layout.positions, 0..) |pos, i| {
                positions[i * 3 + 0] = pos.t;
                positions[i * 3 + 1] = pos.h;
                positions[i * 3 + 2] = pos.w;
            }

            const timesteps = try allocator.alloc(f32, timestep_slots);
            errdefer allocator.free(timesteps);
            @memset(timesteps, 0);
            const n = @min(layout.timesteps.len, timesteps.len);
            if (n > 0) {
                @memcpy(timesteps[0..n], layout.timesteps[0..n]);
                for (n..timesteps.len) |i| timesteps[i] = layout.timesteps[n - 1];
            }

            const timestep_indices = try allocator.dupe(u32, layout.timestep_indices);
            errdefer allocator.free(timestep_indices);
            const adaln_indices = try pipeline.adalnIndices(allocator, layout);
            errdefer allocator.free(adaln_indices);
            const text_indices = try allocator.dupe(u32, layout.text_indices);
            errdefer allocator.free(text_indices);
            const video_indices = try allocator.dupe(u32, layout.video_indices);
            errdefer allocator.free(video_indices);
            return .{
                .positions = positions,
                .timesteps = timesteps,
                .timestep_indices = timestep_indices,
                .adaln_indices = adaln_indices,
                .text_indices = text_indices,
                .video_indices = video_indices,
                .audio_indices = try allocator.dupe(u32, layout.audio_indices),
            };
        }

        pub fn deinit(self: HostLayout, allocator: std.mem.Allocator) void {
            allocator.free(self.positions);
            allocator.free(self.timesteps);
            allocator.free(self.timestep_indices);
            allocator.free(self.adaln_indices);
            allocator.free(self.text_indices);
            allocator.free(self.video_indices);
            allocator.free(self.audio_indices);
        }
    };

    pub const Latents = struct {
        video: []f32,
        audio: []f32,

        pub fn deinit(self: Latents, allocator: std.mem.Allocator) void {
            allocator.free(self.video);
            allocator.free(self.audio);
        }
    };

    fn scatterVisionHidden(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        hidden: *zml.Buffer,
        merged: []const f32,
        spans: []const VisionSpan,
        hidden_dim: u32,
    ) !void {
        const slice = try hidden.toSliceAlloc(allocator, io);
        defer slice.free(allocator);
        var off: usize = 0;
        switch (hidden.shape().dtype()) {
            .f32 => {
                const host = slice.items(f32);
                for (spans) |span| {
                    const n = @as(usize, span.tokens) * hidden_dim;
                    @memcpy(host[@as(usize, span.start) * hidden_dim ..][0..n], merged[off..][0..n]);
                    off += n;
                }
            },
            .bf16 => {
                const host = slice.items(zml.floats.BFloat16);
                for (spans) |span| {
                    const n = @as(usize, span.tokens) * hidden_dim;
                    var i: usize = 0;
                    while (i < n) : (i += 1) {
                        host[@as(usize, span.start) * hidden_dim + i] = .fromF32(merged[off + i]);
                    }
                    off += n;
                }
            },
            else => return error.UnsupportedEmbedDtype,
        }
        const replacement = try zml.Buffer.fromBytes(io, platform, slice.shape, .replicated, slice.constData());
        hidden.deinit();
        hidden.* = replacement;
    }

    fn bufferFromItems(io: std.Io, platform: *const zml.Platform, shape: zml.Shape, items: anytype) !zml.Buffer {
        const bytes = std.mem.sliceAsBytes(items);
        return zml.Buffer.fromBytes(io, platform, shape, .replicated, bytes);
    }

    fn scalarU32(io: std.Io, platform: *const zml.Platform, value: u32) !zml.Buffer {
        var item: u32 = value;
        return zml.Buffer.fromBytes(io, platform, .init(.{}, .u32), .replicated, std.mem.asBytes(&item));
    }

    fn scalarF32(io: std.Io, platform: *const zml.Platform, value: f32) !zml.Buffer {
        var item: f32 = value;
        return zml.Buffer.fromBytes(io, platform, .init(.{}, .f32), .replicated, std.mem.asBytes(&item));
    }

    pub fn bufferFromF32(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        shape: zml.Shape,
        values: []const f32,
    ) !zml.Buffer {
        switch (shape.dtype()) {
            .f32 => return bufferFromItems(io, platform, shape, values),
            .bf16 => {
                const tmp = try allocator.alloc(zml.floats.BFloat16, values.len);
                defer allocator.free(tmp);
                for (tmp, values) |*dst, src| dst.* = .fromF32(src);
                return bufferFromItems(io, platform, shape, tmp);
            },
            else => return error.UnsupportedEmbedDtype,
        }
    }

    fn envPath(name: [:0]const u8) ?[]const u8 {
        const raw = std.c.getenv(name) orelse return null;
        const path = std.mem.span(raw);
        return if (path.len == 0) null else path;
    }

    pub fn openDumpDir(io: std.Io) !?std.Io.Dir {
        const path = envPath("H3_LAYER_DUMP") orelse return null;
        try std.Io.Dir.cwd().createDirPath(io, path);
        if (std.fs.path.isAbsolute(path)) return try std.Io.Dir.openDirAbsolute(io, path, .{});
        return try std.Io.Dir.cwd().openDir(io, path, .{});
    }

    fn writeDumpBytes(io: std.Io, dir: std.Io.Dir, name: []const u8, bytes: []const u8) !void {
        const file = try dir.createFile(io, name, .{});
        defer file.close(io);
        var writer = file.writer(io, &.{});
        try writer.interface.writeAll(bytes);
    }

    fn writeDumpShape(io: std.Io, dir: std.Io.Dir, name: []const u8, dims: []const i64) !void {
        var buf: [128]u8 = undefined;
        var used: usize = 0;
        for (dims, 0..) |d, i| {
            const part = if (i == 0)
                try std.fmt.bufPrint(buf[used..], "{d}", .{d})
            else
                try std.fmt.bufPrint(buf[used..], " {d}", .{d});
            used += part.len;
        }
        var path_buf: [160]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}.shape", .{name});
        try writeDumpBytes(io, dir, path, buf[0..used]);
    }

    pub fn dumpHostF32(io: std.Io, dir: std.Io.Dir, name: []const u8, values: []const f32, dims: []const i64) !void {
        var path_buf: [160]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}.f32", .{name});
        try writeDumpBytes(io, dir, path, std.mem.sliceAsBytes(values));
        try writeDumpShape(io, dir, name, dims);
    }

    fn dumpHostU32(io: std.Io, dir: std.Io.Dir, name: []const u8, values: []const u32, dims: []const i64) !void {
        var path_buf: [160]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}.u32", .{name});
        try writeDumpBytes(io, dir, path, std.mem.sliceAsBytes(values));
        try writeDumpShape(io, dir, name, dims);
    }

    fn dumpBuffer(
        allocator: std.mem.Allocator,
        io: std.Io,
        dir: std.Io.Dir,
        name: []const u8,
        buf: *zml.Buffer,
    ) !void {
        const slice = try buf.toSliceAlloc(allocator, io);
        defer slice.free(allocator);
        const out = try allocator.alloc(f32, slice.shape.count());
        defer allocator.free(out);
        switch (slice.shape.dtype()) {
            .f32 => @memcpy(out, slice.items(f32)),
            .bf16 => {
                const src = slice.items(zml.floats.BFloat16);
                for (out, src) |*d, s| d.* = s.toF32();
            },
            else => return error.UnsupportedEmbedDtype,
        }
        try dumpHostF32(io, dir, name, out, slice.shape.dims());
        log.info("dump {s} {any}", .{ name, slice.shape.dims() });
    }

    pub const VisionSpan = struct {
        start: u32,
        tokens: u32,
        grid_h: u32 = 1,
        grid_w: u32 = 1,
        temporal: u32 = 1,
    };

    pub const TextExtras = struct {
        positions: ?[]const f32 = null,
        deepstack: [3]?[]const f32 = .{ null, null, null },
        vision_merged: ?[]const f32 = null,
        vision_spans: []const VisionSpan = &.{},
    };

    pub fn fillEncoderPositions(pos: []f32, seq: u32, spans: []const VisionSpan) void {
        var cursor: f32 = 0;
        var i: u32 = 0;
        var span_i: usize = 0;
        while (i < seq) {
            if (span_i < spans.len and i == spans[span_i].start) {
                const span = spans[span_i];
                vision.applyVisionPositions(pos, span.start, span.tokens, span.grid_h, span.grid_w, span.temporal, &cursor);
                i += span.tokens;
                span_i += 1;
            } else {
                pos[i * 3 + 0] = cursor;
                pos[i * 3 + 1] = cursor;
                pos[i * 3 + 2] = cursor;
                cursor += 1;
                i += 1;
            }
        }
    }

    pub fn encodeText(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        compiled: *const pipeline.Compiled,
        loaded: *const encoder.LoadedModel,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        tokens: []const u32,
        extras: TextExtras,
        progress: *std.Progress.Node,
    ) !zml.Buffer {
        const seq: u32 = @intCast(tokens.len);
        const hidden_dim: u32 = @intCast(loaded.cfg.hidden_size);
        const head_dim: u32 = @intCast(loaded.cfg.head_dim);

        const token_shape = zml.Shape.init(.{ .b = 1, .s = tokens.len }, .u32);
        var token_buf = try bufferFromItems(io, platform, token_shape, tokens);
        defer token_buf.deinit();
        const encode_start: std.Io.Timestamp = .now(io, .awake);
        const n_layers = loaded.inner.layers.len;
        const layer_bytes: u64 = if (n_layers == 0) 0 else weights.modelBytes(&loaded.inner.layers[0]);
        const n_keep = policy.encKeepLayers(config.minDeviceBytes(platform), layer_bytes, @intCast(n_layers));
        const keep_all = n_keep == n_layers and n_layers > 0;
        log.info(
            "encoder: start tokens={d} layers={d} keep={d} prefetch={d} layer={d}MiB",
            .{ tokens.len, n_layers, n_keep, policy.enc_prefetch, layer_bytes / (1024 * 1024) },
        );
        var embed_bufs = try loaded.loadEmbed(allocator, io, platform, store, shardings, progress);
        defer encoder.EmbedTokens.unloadBuffers(&embed_bufs);
        var embed_runner = try zml.FnExe(encoder.EmbedTokens.forward).Runner(.{.embedding}).init(&compiled.encode_embed, allocator, .{
            .embedding = embed_bufs,
        });
        defer embed_runner.deinit(allocator);

        const prefetch = policy.enc_prefetch;
        var loaders: [prefetch]zml.io.Loader = undefined;
        var loaders_ready: u32 = 0;
        defer {
            var k: u32 = 0;
            while (k < loaders_ready) : (k += 1) loaders[k].deinit();
        }
        while (loaders_ready < prefetch) : (loaders_ready += 1) {
            loaders[loaders_ready] = try weights.initLoader(allocator, platform);
        }
        const EncFut = @TypeOf(try io.concurrent(loadEncoderLayer, .{
            allocator, io, platform, loaded, store, shardings, @as(usize, 0), progress, &loaders[0],
        }));
        var futs: [prefetch]?EncFut = .{null} ** prefetch;
        errdefer {
            for (&futs) |*f| cancelEnc(f, io);
        }
        var spawned: usize = 0;
        while (spawned < prefetch and spawned < n_layers) : (spawned += 1) {
            futs[spawned] = try io.concurrent(loadEncoderLayer, .{
                allocator, io, platform, loaded, store, shardings, spawned, progress, &loaders[spawned],
            });
        }

        var hidden: zml.Buffer = undefined;
        embed_runner.run(io, .{
            .inputs = .{ .tokens = token_buf },
            .outputs = .{ .hidden = &hidden },
            .opts = .{ .wait = true },
        });
        errdefer hidden.deinit();

        if (extras.vision_merged) |merged| {
            if (compiled.encode_scatter) |*scatter_exe| {
                const n_vis: u32 = @intCast(@divExact(merged.len, hidden_dim));
                const idx = try allocator.alloc(u32, n_vis);
                defer allocator.free(idx);
                var off: usize = 0;
                for (extras.vision_spans) |span| {
                    var t: u32 = 0;
                    while (t < span.tokens) : (t += 1) {
                        idx[off] = span.start + t;
                        off += 1;
                    }
                }
                var val_buf = try bufferFromItems(io, platform, .init(.{ .b = 1, .s = n_vis, .d = hidden_dim }, .f32), merged);
                defer val_buf.deinit();
                var idx_buf = try bufferFromItems(io, platform, .init(.{ .s = n_vis }, .u32), idx);
                defer idx_buf.deinit();
                var scatter_runner = try zml.FnExe(dit.scatterRows).Runner(.{}).init(scatter_exe, allocator, .{});
                defer scatter_runner.deinit(allocator);
                var next: zml.Buffer = undefined;
                scatter_runner.run(io, .{
                    .inputs = .{ .hidden = hidden, .values = val_buf, .indices = idx_buf },
                    .outputs = .{ .hidden = &next },
                    .opts = .{ .wait = true },
                });
                hidden.deinit();
                hidden = next;
            } else {
                try scatterVisionHidden(allocator, io, platform, &hidden, merged, extras.vision_spans, hidden_dim);
            }
            log.info("encoder: scattered vision spans={d}", .{extras.vision_spans.len});
        }

        const pos = try allocator.alloc(f32, seq * 3);
        defer allocator.free(pos);
        if (extras.positions) |p| {
            @memcpy(pos, p[0 .. seq * 3]);
        } else {
            vision.fillArangePositions(pos, seq);
        }
        const cos = try allocator.alloc(f32, seq * head_dim);
        defer allocator.free(cos);
        const sin = try allocator.alloc(f32, seq * head_dim);
        defer allocator.free(sin);
        vision.hostInterleavedMrope(pos, seq, head_dim, loaded.cfg.rope_theta, loaded.cfg.mrope_section, cos, sin);
        var cos_buf = try bufferFromF32(allocator, io, platform, .init(.{ .s = seq, .hd = head_dim }, loaded.inner.embed_tokens.weight.dtype()), cos);
        defer cos_buf.deinit();
        var sin_buf = try bufferFromF32(allocator, io, platform, .init(.{ .s = seq, .hd = head_dim }, loaded.inner.embed_tokens.weight.dtype()), sin);
        defer sin_buf.deinit();

        const zeros = try allocator.alloc(f32, seq * hidden_dim);
        defer allocator.free(zeros);
        @memset(zeros, 0);
        var zero_delta = try bufferFromF32(allocator, io, platform, .init(.{ .b = 1, .s = seq, .d = hidden_dim }, loaded.inner.embed_tokens.weight.dtype()), zeros);
        defer zero_delta.deinit();

        var kept = try allocator.alloc(?zml.Bufferized(encoder.TransformerLayer), if (keep_all) n_layers else 0);
        defer {
            for (kept) |*slot| {
                if (slot.*) |*bufs| encoder.TransformerLayer.unloadBuffers(bufs);
            }
            allocator.free(kept);
        }
        if (keep_all) {
            var fill_i: usize = 0;
            while (fill_i < n_layers) : (fill_i += 1) {
                const slot = fill_i % prefetch;
                kept[fill_i] = try futs[slot].?.await(io);
                futs[slot] = null;
                if (fill_i + prefetch < n_layers) {
                    futs[slot] = try io.concurrent(loadEncoderLayer, .{
                        allocator, io, platform, loaded, store, shardings, fill_i + prefetch, progress, &loaders[slot],
                    });
                }
            }
        }

        const LayerRunner = zml.FnExe(encoder.TransformerLayer.forward).Runner(.{.layer});
        var layer_runner: ?LayerRunner = null;
        defer if (layer_runner) |*r| r.deinit(allocator);
        var layer_i: usize = 0;
        while (layer_i < n_layers) : (layer_i += 1) {
            var streamed: ?zml.Bufferized(encoder.TransformerLayer) = null;
            defer if (streamed) |*bufs| encoder.TransformerLayer.unloadBuffers(bufs);
            const layer_bufs = if (keep_all) kept[layer_i].? else blk: {
                const slot = layer_i % prefetch;
                const bufs = try futs[slot].?.await(io);
                futs[slot] = null;
                if (layer_i + prefetch < n_layers) {
                    futs[slot] = try io.concurrent(loadEncoderLayer, .{
                        allocator, io, platform, loaded, store, shardings, layer_i + prefetch, progress, &loaders[slot],
                    });
                }
                streamed = bufs;
                break :blk bufs;
            };
            if (layer_runner) |*r| {
                weights.rebake(r, .{ .layer = layer_bufs });
            } else {
                layer_runner = try LayerRunner.init(&compiled.encode_layer, allocator, .{ .layer = layer_bufs });
            }

            var owned_delta: ?zml.Buffer = null;
            defer if (owned_delta) |*b| b.deinit();
            const delta = if (layer_i < 3) blk: {
                if (extras.deepstack[layer_i]) |host| {
                    owned_delta = try bufferFromF32(allocator, io, platform, .init(.{ .b = 1, .s = seq, .d = hidden_dim }, loaded.inner.embed_tokens.weight.dtype()), host);
                    break :blk owned_delta.?;
                }
                break :blk zero_delta;
            } else zero_delta;
            var next: zml.Buffer = undefined;
            layer_runner.?.run(io, .{
                .inputs = .{ .hidden = hidden, .cos = cos_buf, .sin = sin_buf, .visual_delta = delta },
                .outputs = .{ .hidden = &next },
                .opts = .{ .wait = true },
            });
            hidden.deinit();
            hidden = next;
        }
        log.info("encoder: ok tokens={d} layers={d} [{f}]", .{ tokens.len, n_layers, encode_start.untilNow(io, .awake) });
        if (try openDumpDir(io)) |dir| {
            var dump = dir;
            defer dump.close(io);
            try dumpBuffer(allocator, io, dump, "prompt_embeds", &hidden);
            try dumpHostU32(io, dump, "tokens", tokens, &.{@intCast(tokens.len)});
        }
        return hidden;
    }

    pub const DenoiseCond = struct {
        videos: []const packing.ConditionVideo = &.{},
        video_patches: []const f32 = &.{},
        audio_patches: []const f32 = &.{},
    };

    pub fn denoise(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        compiled: *const pipeline.Compiled,
        loaded: *const dit.LoadedModel,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        opts: pipeline.Options,
        geo: pipeline.Geometry,
        text: zml.Buffer,
        text_len: u32,
        layout: packing.Layout,
        schedules: scheduler_mod.DualSchedule,
        seed: u64,
        cond: DenoiseCond,
        progress: *std.Progress.Node,
    ) !Latents {
        var gen = noise.Generator.init(seed);
        var video = try noise.drawVideo(
            allocator,
            &gen,
            cond.videos,
            cond.video_patches,
            geo.latent_t,
            geo.latent_h,
            geo.latent_w,
            loaded.inner.cfg.patch_size,
            false,
        );
        errdefer allocator.free(video);
        var audio = try noise.drawAudio(allocator, &gen, cond.audio_patches, geo.audio_dim, geo.audio_t);
        errdefer allocator.free(audio);
        if (video.len != geo.video_tokens * geo.video_patch_dim) return error.VideoNoiseSize;
        if (audio.len != geo.audio_tokens * geo.audio_dim) return error.AudioNoiseSize;

        var dump_dir = try openDumpDir(io);
        defer if (dump_dir) |*d| d.close(io);
        if (dump_dir) |dir| {
            try dumpHostF32(io, dir, "video_noise", video, &.{ 1, @intCast(geo.video_tokens), @intCast(geo.video_patch_dim) });
            try dumpHostF32(io, dir, "audio_noise", audio, &.{ 1, @intCast(geo.audio_tokens), @intCast(geo.audio_dim) });
        }

        const video_shape = zml.Shape.init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32);
        const audio_shape = zml.Shape.init(.{ .b = 1, .s = geo.audio_tokens, .d = geo.audio_dim }, .f32);
        const seq = layout.seqLen();
        const steps = schedules.video.stepCount();
        const n_blocks = loaded.inner.blocks.len;

        var host = try HostLayout.fromLayout(allocator, layout, packing.timestep_slot_count);
        defer host.deinit(allocator);
        var pos_buf = try bufferFromItems(io, platform, .init(.{ .s = seq, .ax = 3 }, .f32), host.positions);
        defer pos_buf.deinit();
        var video_idx = try bufferFromItems(io, platform, .init(.{ .s = geo.video_tokens }, .u32), host.video_indices);
        defer video_idx.deinit();
        var audio_idx = try bufferFromItems(io, platform, .init(.{ .s = geo.audio_tokens }, .u32), host.audio_indices);
        defer audio_idx.deinit();
        var text_idx = try bufferFromItems(io, platform, .init(.{ .s = text_len }, .u32), host.text_indices);
        defer text_idx.deinit();
        var adaln_buf = try bufferFromItems(io, platform, .init(.{ .s = seq }, .u32), host.adaln_indices);
        defer adaln_buf.deinit();
        var time_idx = try bufferFromItems(io, platform, .init(.{ .s = seq }, .u32), host.timestep_indices);
        defer time_idx.deinit();
        if (dump_dir) |dir| {
            try dumpHostF32(io, dir, "positions", host.positions, &.{ @intCast(seq), 3 });
            try dumpHostU32(io, dir, "video_indices", host.video_indices, &.{@intCast(geo.video_tokens)});
            try dumpHostU32(io, dir, "audio_indices", host.audio_indices, &.{@intCast(geo.audio_tokens)});
            try dumpHostU32(io, dir, "text_indices", host.text_indices, &.{@intCast(text_len)});
            try dumpHostU32(io, dir, "adaln_indices", host.adaln_indices, &.{@intCast(seq)});
            try dumpHostU32(io, dir, "timestep_indices", host.timestep_indices, &.{@intCast(seq)});
            const tags = try allocator.alloc(u32, layout.token_tags.len);
            defer allocator.free(tags);
            for (layout.token_tags, tags) |tag, *dst| dst.* = tag;
            try dumpHostU32(io, dir, "token_tags", tags, &.{@intCast(seq)});
            try dumpHostF32(io, dir, "timesteps", schedules.video.timesteps, &.{@intCast(schedules.video.timesteps.len)});
            try dumpHostF32(io, dir, "audio_timesteps", schedules.audio.timesteps, &.{@intCast(schedules.audio.timesteps.len)});
        }

        var text_bufs = try loaded.loadTextPrep(allocator, io, platform, store, shardings, progress);
        defer dit.TextPrep.unloadBuffers(&text_bufs, allocator);
        var text_runner = try zml.FnExe(dit.prepareText).Runner(.{.model}).init(&compiled.prepare_text, allocator, .{ .model = text_bufs });
        defer text_runner.deinit(allocator);
        var refined_text: zml.Buffer = undefined;
        text_runner.run(io, .{
            .inputs = .{ .text = text },
            .outputs = .{ .text = &refined_text },
            .opts = .{ .wait = true },
        });
        defer refined_text.deinit();

        var rope_runner = try zml.FnExe(dit.prepareRope).Runner(.{}).init(&compiled.prepare_rope, allocator, .{});
        defer rope_runner.deinit(allocator);
        var cos: zml.Buffer = undefined;
        var sin: zml.Buffer = undefined;
        rope_runner.run(io, .{
            .inputs = .{ .position_ids = pos_buf },
            .outputs = .{ .cos = &cos, .sin = &sin },
            .opts = .{ .wait = true },
        });
        defer cos.deinit();
        defer sin.deinit();

        const flat_n = steps * packing.timestep_slot_count;
        const flat_t = try allocator.alloc(f32, flat_n);
        defer allocator.free(flat_t);
        for (0..steps) |i| {
            const video_t = schedules.video.timesteps[i];
            const audio_t = 1.0 - scheduler_mod.timeShiftSigma(1.0 - video_t, opts.video_shift, opts.audio_shift);
            packing.writeTimesteps(flat_t[i * packing.timestep_slot_count ..][0..packing.timestep_slot_count], video_t, audio_t);
        }
        var flat_buf = try bufferFromItems(io, platform, .init(.{ .n = flat_n }, .f32), flat_t);
        defer flat_buf.deinit();

        var time_bufs = try loaded.loadTimeEmbedder(allocator, io, platform, store, shardings, progress);
        var all_temb: zml.Buffer = undefined;
        {
            var temb_runner = try zml.FnExe(dit.prepareTemb).Runner(.{.model}).init(&compiled.prepare_temb, allocator, .{ .model = time_bufs });
            defer temb_runner.deinit(allocator);
            temb_runner.run(io, .{
                .inputs = .{ .timestep = flat_buf },
                .outputs = .{ .temb = &all_temb },
                .opts = .{ .wait = true },
            });
        }
        defer all_temb.deinit();
        dit.TimeEmbedder.unloadBuffers(&time_bufs);

        const core0 = loaded.inner.blocks[0].corePart();
        const core_bytes = weights.modelBytes(&core0);
        const tp: u32 = if (shardings.len > 0) @intCast(shardings[0].numPartitionsForLogicalAxis(.model)) else 1;
        const decision = policy.decide(.{
            .target = platform.target,
            .seq = seq,
            .hidden = loaded.cfg.hidden_size,
            .heads = loaded.cfg.num_attention_heads,
            .head_dim = loaded.cfg.attention_head_dim,
            .layers = @intCast(n_blocks),
            .steps = @intCast(steps),
            .dtype = loaded.inner.blocks[0].norm1.weight.dtype(),
            .device_bytes = config.minDeviceBytes(platform),
            .tp = tp,
            .devices = @intCast(platform.devices.len),
            .block_core_bytes = core_bytes / @max(1, tp),
            .dtype_bytes = policy.dtypeBytes(loaded.inner.blocks[0].norm1.weight.dtype()),
        });
        const n_resident = policy.ditKeepBlocks(decision.resident_blocks, @intCast(n_blocks));
        const group_size = @max(1, @min(compiled.group_size, n_resident));
        log.info(
            "denoise: prepare blocks={d} resident={d} keep={d} group={d} attn={s} core={d}MiB tables={d}MiB",
            .{
                n_blocks,
                decision.resident_blocks,
                n_resident,
                group_size,
                @tagName(decision.attention),
                core_bytes / (1024 * 1024),
                decision.adaln_table_bytes / (1024 * 1024),
            },
        );

        var tables = try allocator.alloc(zml.Buffer, n_blocks);
        var tables_filled: usize = 0;
        errdefer {
            for (tables[0..tables_filled]) |*t| t.deinit();
            allocator.free(tables);
        }
        var cores = try allocator.alloc(?zml.Bufferized(dit.BlockCore), n_blocks);
        @memset(cores, null);
        errdefer {
            for (cores) |*c| if (c.*) |*core| dit.BlockCore.unloadBuffers(core);
            allocator.free(cores);
        }

        var loaders = [2]zml.io.Loader{
            try weights.initLoader(allocator, platform),
            try weights.initLoader(allocator, platform),
        };
        defer loaders[0].deinit();
        defer loaders[1].deinit();

        const AdaLnRunner = zml.FnExe(dit.prepareAdaln).Runner(.{.model});
        var adaln_runner: ?AdaLnRunner = null;
        defer if (adaln_runner) |*r| r.deinit(allocator);
        var prev_adaln: ?zml.Bufferized(dit.AdaLn) = null;
        defer if (prev_adaln) |*a| dit.AdaLn.unloadBuffers(a);
        var block_i: usize = 0;
        while (block_i < n_blocks) : (block_i += 1) {
            const adaln_loader = &loaders[block_i % 2];
            const adaln_bufs = try loaded.loadAdaln(allocator, io, platform, store, shardings, block_i, progress, adaln_loader);
            if (adaln_runner) |*r| {
                weights.rebake(r, .{ .model = .{ .adaln = adaln_bufs } });
                if (prev_adaln) |*a| dit.AdaLn.unloadBuffers(a);
            } else {
                adaln_runner = try AdaLnRunner.init(&compiled.prepare_adaln, allocator, .{ .model = .{ .adaln = adaln_bufs } });
            }
            prev_adaln = adaln_bufs;
            var table: zml.Buffer = undefined;
            adaln_runner.?.run(io, .{
                .inputs = .{ .temb = all_temb },
                .outputs = .{ .table = &table },
                .opts = .{ .wait = true },
            });
            tables[block_i] = table;
            tables_filled += 1;
            if (block_i < n_resident) {
                cores[block_i] = try loaded.loadCore(allocator, io, platform, store, shardings, block_i, progress, &loaders[(block_i + 1) % 2]);
            }
        }
        if (adaln_runner) |*r| {
            r.deinit(allocator);
            adaln_runner = null;
        }
        if (prev_adaln) |*a| {
            dit.AdaLn.unloadBuffers(a);
            prev_adaln = null;
        }

        var final_table: zml.Buffer = undefined;
        {
            var final_adaln = try loaded.loadFinalAdaln(allocator, io, platform, store, shardings, progress);
            var final_runner = try AdaLnRunner.init(&compiled.prepare_final_adaln, allocator, .{
                .model = .{ .adaln = final_adaln },
            });
            defer final_runner.deinit(allocator);
            final_runner.run(io, .{
                .inputs = .{ .temb = all_temb },
                .outputs = .{ .table = &final_table },
                .opts = .{ .wait = true },
            });
            dit.AdaLn.unloadBuffers(&final_adaln);
        }
        defer final_table.deinit();

        var patch_bufs = try loaded.loadPatchEmbed(allocator, io, platform, store, shardings, progress);
        defer dit.PatchEmbed.unloadBuffers(&patch_bufs);
        var patch_runner = try zml.FnExe(dit.embedPatches).Runner(.{.model}).init(&compiled.embed_patches, allocator, .{ .model = patch_bufs });
        defer patch_runner.deinit(allocator);

        var finish_bufs = try loaded.loadFinishCore(allocator, io, platform, store, shardings, progress);
        defer dit.FinishCore.unloadBuffers(&finish_bufs);
        var finish_runner = try zml.FnExe(dit.finish).Runner(.{.model}).init(&compiled.finish, allocator, .{ .model = finish_bufs });
        defer finish_runner.deinit(allocator);

        const BlockRunner = zml.FnExe(dit.stepBlock).Runner(.{.layer});
        var block_runner: ?BlockRunner = null;
        defer if (block_runner) |*r| r.deinit(allocator);
        const GroupRunner = zml.FnExe(dit.BlockGroup.forward).Runner(.{.group});
        var group_runner: ?GroupRunner = null;
        defer if (group_runner) |*r| r.deinit(allocator);
        var group_layers: []zml.Bufferized(dit.BlockCore) = &.{};
        defer if (group_layers.len != 0) allocator.free(group_layers);
        var group_tables: []zml.Buffer = &.{};
        defer if (group_tables.len != 0) allocator.free(group_tables);
        const use_group = dump_dir == null and compiled.block_group != null and group_size > 1 and group_size == compiled.group_size;
        if (use_group) {
            group_layers = try allocator.alloc(zml.Bufferized(dit.BlockCore), group_size);
            group_tables = try allocator.alloc(zml.Buffer, group_size);
        }

        var apply_v = try zml.FnExe(multistep.apply).Runner(.{}).init(&compiled.apply_video, allocator, .{});
        defer apply_v.deinit(allocator);
        var apply_a = try zml.FnExe(multistep.apply).Runner(.{}).init(&compiled.apply_audio, allocator, .{});
        defer apply_a.deinit(allocator);

        var video_buf = try bufferFromItems(io, platform, video_shape, video);
        defer video_buf.deinit();
        var audio_buf = try bufferFromItems(io, platform, audio_shape, audio);
        defer audio_buf.deinit();

        const denoise_start: std.Io.Timestamp = .now(io, .awake);
        log.info(
            "denoise: start steps={d} blocks={d} video_tokens={d} audio_tokens={d} seed={d}",
            .{ steps, n_blocks, geo.video_tokens, geo.audio_tokens, seed },
        );

        var step_i: usize = 0;
        while (step_i < steps) : (step_i += 1) {
            const step_start: std.Io.Timestamp = .now(io, .awake);
            const video_t = schedules.video.timesteps[step_i];
            const audio_t = 1.0 - scheduler_mod.timeShiftSigma(1.0 - video_t, opts.video_shift, opts.audio_shift);
            var step_buf = try scalarU32(io, platform, @intCast(step_i));
            defer step_buf.deinit();

            var hidden: zml.Buffer = undefined;
            patch_runner.run(io, .{
                .inputs = .{
                    .video = video_buf,
                    .audio = audio_buf,
                    .text = refined_text,
                    .video_indices = video_idx,
                    .audio_indices = audio_idx,
                    .text_indices = text_idx,
                },
                .outputs = .{ .hidden = &hidden },
                .opts = .{ .wait = true },
            });
            defer hidden.deinit();
            if (step_i == 0) if (dump_dir) |dir| try dumpBuffer(allocator, io, dir, "step0_embed", &hidden);

            var i: usize = 0;
            if (use_group) {
                while (i + group_size <= n_resident) {
                    var g: usize = 0;
                    while (g < group_size) : (g += 1) {
                        group_layers[g] = cores[i + g].?;
                        group_tables[g] = tables[i + g];
                    }
                    if (group_runner) |*r| {
                        weights.rebake(r, .{ .group = .{ .layers = group_layers } });
                    } else if (compiled.block_group) |*exe| {
                        group_runner = try GroupRunner.init(exe, allocator, .{ .group = .{ .layers = group_layers } });
                    } else unreachable;
                    var next: zml.Buffer = undefined;
                    group_runner.?.run(io, .{
                        .inputs = .{
                            .hidden = hidden,
                            .tables = group_tables,
                            .step = step_buf,
                            .adaln_indices = adaln_buf,
                            .cos = cos,
                            .sin = sin,
                        },
                        .outputs = .{ .hidden = &next },
                        .opts = .{ .wait = true },
                    });
                    hidden.deinit();
                    hidden = next;
                    i += group_size;
                }
            }
            while (i < n_blocks) : (i += 1) {
                var owned_core: ?zml.Bufferized(dit.BlockCore) = null;
                defer if (owned_core) |*c| dit.BlockCore.unloadBuffers(c);
                const core = if (cores[i]) |c| c else blk: {
                    owned_core = try loaded.loadCore(allocator, io, platform, store, shardings, i, progress, &loaders[i % 2]);
                    break :blk owned_core.?;
                };
                if (block_runner) |*r| {
                    weights.rebake(r, .{ .layer = core });
                } else {
                    block_runner = try BlockRunner.init(&compiled.block, allocator, .{ .layer = core });
                }
                var next: zml.Buffer = undefined;
                block_runner.?.run(io, .{
                    .inputs = .{
                        .hidden = hidden,
                        .table = tables[i],
                        .step = step_buf,
                        .adaln_indices = adaln_buf,
                        .cos = cos,
                        .sin = sin,
                    },
                    .outputs = .{ .hidden = &next },
                    .opts = .{ .wait = true },
                });
                hidden.deinit();
                hidden = next;
            }

            var video_out: zml.Buffer = undefined;
            var audio_out: zml.Buffer = undefined;
            finish_runner.run(io, .{
                .inputs = .{
                    .hidden = hidden,
                    .table = final_table,
                    .step = step_buf,
                    .timestep_indices = time_idx,
                    .video_indices = video_idx,
                    .audio_indices = audio_idx,
                },
                .opts = .{ .wait = true },
                .outputs = .{ .video = &video_out, .audio = &audio_out },
            });
            if (step_i == 0) if (dump_dir) |dir| {
                try dumpBuffer(allocator, io, dir, "step0_video_vel", &video_out);
                try dumpBuffer(allocator, io, dir, "step0_audio_vel", &audio_out);
            };
            defer video_out.deinit();
            defer audio_out.deinit();

            var sigma_v = try scalarF32(io, platform, schedules.video.sigmas[step_i]);
            defer sigma_v.deinit();
            var sigma_v_next = try scalarF32(io, platform, schedules.video.sigmas[step_i + 1]);
            defer sigma_v_next.deinit();
            var sigma_v_t = try scalarF32(io, platform, 1.0 - schedules.video.timesteps[step_i]);
            defer sigma_v_t.deinit();
            var sigma_a = try scalarF32(io, platform, schedules.audio.sigmas[step_i]);
            defer sigma_a.deinit();
            var sigma_a_next = try scalarF32(io, platform, schedules.audio.sigmas[step_i + 1]);
            defer sigma_a_next.deinit();
            var sigma_a_t = try scalarF32(io, platform, 1.0 - schedules.audio.timesteps[step_i]);
            defer sigma_a_t.deinit();

            var next_video: zml.Buffer = undefined;
            apply_v.run(io, .{
                .inputs = .{
                    .sample = video_buf,
                    .velocity = video_out,
                    .sigma = sigma_v,
                    .sigma_next = sigma_v_next,
                    .sigma_t = sigma_v_t,
                },
                .outputs = .{ .sample = &next_video },
                .opts = .{ .wait = true },
            });
            video_buf.deinit();
            video_buf = next_video;

            var next_audio: zml.Buffer = undefined;
            apply_a.run(io, .{
                .inputs = .{
                    .sample = audio_buf,
                    .velocity = audio_out,
                    .sigma = sigma_a,
                    .sigma_next = sigma_a_next,
                    .sigma_t = sigma_a_t,
                },
                .outputs = .{ .sample = &next_audio },
                .opts = .{ .wait = true },
            });
            audio_buf.deinit();
            audio_buf = next_audio;

            log.info("denoise {d}/{d} t_video={d:.4} t_audio={d:.4} [{f}]", .{
                step_i + 1,
                steps,
                video_t,
                audio_t,
                step_start.untilNow(io, .awake),
            });
        }

        try video_buf.toSlice(io, .init(video_shape, std.mem.sliceAsBytes(video)));
        try audio_buf.toSlice(io, .init(audio_shape, std.mem.sliceAsBytes(audio)));

        for (tables) |*t| t.deinit();
        allocator.free(tables);
        for (cores) |*c| if (c.*) |*core| dit.BlockCore.unloadBuffers(core);
        allocator.free(cores);

        log.info("denoise: ok steps={d} [{f}]", .{ steps, denoise_start.untilNow(io, .awake) });

        if (cond.video_patches.len == 0 and cond.audio_patches.len == 0) {
            return .{ .video = video, .audio = audio };
        }
        const v_out = try allocator.dupe(f32, video[cond.video_patches.len..]);
        errdefer allocator.free(v_out);
        const a_out = try allocator.dupe(f32, audio[cond.audio_patches.len..]);
        allocator.free(video);
        allocator.free(audio);
        return .{ .video = v_out, .audio = a_out };
    }

    fn cancelLoad(comptime unload: anytype, fut: anytype, io: std.Io) void {
        if (fut.*) |*f| {
            if (f.cancel(io)) |bufs| {
                var b = bufs;
                unload(&b);
            } else |_| {}
            fut.* = null;
        }
    }

    fn cancelEnc(fut: anytype, io: std.Io) void {
        cancelLoad(encoder.TransformerLayer.unloadBuffers, fut, io);
    }

    fn loadEncoderLayer(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        loaded: *const encoder.LoadedModel,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        index: usize,
        progress: *std.Progress.Node,
        loader: *zml.io.Loader,
    ) !zml.Bufferized(encoder.TransformerLayer) {
        return loaded.loadLayer(allocator, io, platform, store, shardings, index, progress, loader);
    }

    pub const Request = struct {
        opts: pipeline.Options,
        geo: pipeline.Geometry,
        target: pipeline.Geometry,
        tokens: []const u32,
        extras: TextExtras,
        layout: packing.Layout,
        schedules: scheduler_mod.DualSchedule,
        cond: DenoiseCond,
        seed: u64,
        prompt: []const u8,
        out: []const u8,
    };

    pub fn generate(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        models: *ckpt.Bundle,
        compiled: *const pipeline.Compiled,
        compiled_vae: *pipeline.VaeCompiled,
        shardings: []const zml.Sharding,
        progress: *std.Progress.Node,
        req: Request,
    ) !void {
        const dest = media.Output.parse(req.out);
        if (!dest.isCwd()) try std.Io.Dir.cwd().createDirPath(io, dest.dir);
        var out_dir: std.Io.Dir = if (dest.isCwd())
            .cwd()
        else
            try std.Io.Dir.cwd().openDir(io, dest.dir, .{});
        defer if (!dest.isCwd()) out_dir.close(io);
        try writeText(io, out_dir, "prompt.txt", req.prompt);
        const enc_layers: u32 = @intCast(models.enc.inner.layers.len);
        const enc_layer_bytes: u64 = if (enc_layers == 0) 0 else weights.modelBytes(&models.enc.inner.layers[0]);
        const enc_keep = policy.encKeepLayers(config.minDeviceBytes(platform), enc_layer_bytes, enc_layers);
        log.info(
            "stream: encode keep={d}/{d} prefetch={d} vae_window={d}",
            .{ enc_keep, enc_layers, policy.enc_prefetch, policy.vae_load_window },
        );

        var audio_f = try io.concurrent(pipeline.compileAudioDecode, .{
            allocator,
            io,
            platform,
            models.audio.inner,
            req.target,
            shardings,
            progress,
        });
        var audio_taken = false;
        errdefer if (!audio_taken) {
            if (audio_f.cancel(io)) |exe| exe.deinit() else |_| {}
        };

        var text = try encodeText(
            allocator,
            io,
            platform,
            compiled,
            &models.enc,
            &models.enc_store,
            shardings,
            req.tokens,
            req.extras,
            progress,
        );
        defer text.deinit();

        var latents = try denoise(
            allocator,
            io,
            platform,
            compiled,
            &models.dit,
            &models.dit_store,
            shardings,
            req.opts,
            req.geo,
            text,
            @intCast(req.tokens.len),
            req.layout,
            req.schedules,
            req.seed,
            req.cond,
            progress,
        );
        defer latents.deinit(allocator);

        const channels: u32 = @intCast(models.dit.cfg.in_channels);
        const thwc = try packing.unpatchify(
            allocator,
            latents.video,
            req.target.latent_t,
            req.target.latent_h,
            req.target.latent_w,
            channels,
            models.dit.cfg.patch_size,
        );
        defer allocator.free(thwc);
        const rgb = try decode.decodeVideo(
            allocator,
            io,
            platform,
            compiled_vae,
            &models.visual,
            &models.visual_store,
            shardings,
            req.target,
            thwc,
            progress,
        );
        defer allocator.free(rgb);
        compiled_vae.audio = try audio_f.await(io);
        audio_taken = true;
        const wav = try decode.decodeAudio(
            allocator,
            io,
            platform,
            compiled_vae,
            &models.audio,
            &models.audio_store,
            shardings,
            req.target,
            latents.audio,
            progress,
        );
        defer allocator.free(wav);
        try decode.writeOutputs(allocator, io, out_dir, dest.dir, dest.mp4_name, req.target, rgb, wav);
    }

    fn writeText(io: std.Io, dir: std.Io.Dir, name: []const u8, text: []const u8) !void {
        const file = try dir.createFile(io, name, .{});
        defer file.close(io);
        var writer = file.writer(io, &.{});
        try writer.interface.writeAll(text);
        if (text.len == 0 or text[text.len - 1] != '\n')
            try writer.interface.writeByte('\n');
    }
};
