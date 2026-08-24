const std = @import("std");

const zml = @import("zml");

const dit = @import("dit.zig");
const encoder = @import("encoder.zig");
const packing = @import("packing.zig");
const pipeline = @import("pipeline.zig");
const scheduler_mod = @import("scheduler.zig");
const vision = @import("vision.zig");

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

fn bufferFromF32(
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

pub fn writeF32File(io: std.Io, dir: std.Io.Dir, name: []const u8, values: []const f32) !void {
    const file = try dir.createFile(io, name, .{});
    defer file.close(io);
    var writer = file.writer(io, &.{});
    try writer.interface.writeAll(std.mem.sliceAsBytes(values));
}

pub fn readF32File(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8, expected: usize) ![]f32 {
    const file = try dir.openFile(io, name, .{});
    defer file.close(io);
    const n = try file.length(io);
    if (n != expected * @sizeOf(f32)) return error.LatentSizeMismatch;
    const out = try allocator.alloc(f32, expected);
    errdefer allocator.free(out);
    var reader = file.reader(io, &.{});
    try reader.interface.readSliceAll(std.mem.sliceAsBytes(out));
    return out;
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
    log.info("encoder: start tokens={d} layers={d}", .{ tokens.len, loaded.inner.layers.len });
    log.debug("encoder embed: load", .{});
    var embed_bufs = try loaded.loadEmbed(allocator, io, platform, store, shardings, progress);
    defer encoder.EmbedTokens.unloadBuffers(&embed_bufs);
    var embed_runner = try zml.FnExe(encoder.EmbedTokens.forward).Runner(.{.embedding}).init(&compiled.encode_embed, allocator, .{
        .embedding = embed_bufs,
    });
    defer embed_runner.deinit(allocator);
    var hidden: zml.Buffer = undefined;
    log.debug("encoder embed: run tokens={d}", .{tokens.len});
    embed_runner.run(io, .{
        .inputs = .{ .tokens = token_buf },
        .outputs = .{ .hidden = &hidden },
        .opts = .{ .wait = true },
    });
    log.debug("encoder embed: ok {f}", .{hidden.shape()});
    errdefer hidden.deinit();

    if (extras.vision_merged) |merged| {
        try scatterVisionHidden(allocator, io, platform, &hidden, merged, extras.vision_spans, hidden_dim);
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

    const n_layers = loaded.inner.layers.len;
    const EncFut = @TypeOf(try io.concurrent(loadEncoderLayer, .{
        allocator, io, platform, loaded, store, shardings, @as(usize, 0), progress,
    }));
    var current_f: ?EncFut = if (n_layers > 0) try io.concurrent(loadEncoderLayer, .{
        allocator, io, platform, loaded, store, shardings, @as(usize, 0), progress,
    }) else null;
    var next_f: ?EncFut = if (n_layers > 1) try io.concurrent(loadEncoderLayer, .{
        allocator, io, platform, loaded, store, shardings, @as(usize, 1), progress,
    }) else null;
    errdefer cancelEnc(&current_f, io);
    errdefer cancelEnc(&next_f, io);
    var layer_i: usize = 0;
    while (layer_i < n_layers) : (layer_i += 1) {
        var layer_bufs = try current_f.?.await(io);
        current_f = null;
        defer encoder.TransformerLayer.unloadBuffers(&layer_bufs);
        current_f = next_f;
        next_f = if (layer_i + 2 < n_layers) try io.concurrent(loadEncoderLayer, .{
            allocator, io, platform, loaded, store, shardings, layer_i + 2, progress,
        }) else null;
        var layer_runner = try zml.FnExe(encoder.TransformerLayer.forward).Runner(.{.layer}).init(&compiled.encode_layer, allocator, .{
            .layer = layer_bufs,
        });
        defer layer_runner.deinit(allocator);

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
        log.debug("encoder layer {d}/{d}: run", .{ layer_i + 1, n_layers });
        layer_runner.run(io, .{
            .inputs = .{ .hidden = hidden, .cos = cos_buf, .sin = sin_buf, .visual_delta = delta },
            .outputs = .{ .hidden = &next },
            .opts = .{ .wait = true },
        });
        hidden.deinit();
        hidden = next;
    }
    log.info("encoder: ok tokens={d} layers={d} [{f}]", .{ tokens.len, n_layers, encode_start.untilNow(io, .awake) });
    return hidden;
}

pub const DenoiseCond = struct {
    video_patches: []const f32 = &.{},
    audio_patches: []const f32 = &.{},
    videos: []const packing.ConditionVideo = &.{},
    audios: []const packing.ConditionAudio = &.{},
    references: []const packing.ReferenceBlock = &.{},
    text_tags: []const u8 = &.{},
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
    var rng = std.Random.DefaultPrng.init(seed);
    const video_n = geo.video_tokens * geo.video_patch_dim;
    const audio_n = geo.audio_tokens * geo.audio_dim;
    const video = try allocator.alloc(f32, video_n);
    errdefer allocator.free(video);
    const audio = try allocator.alloc(f32, audio_n);
    errdefer allocator.free(audio);
    const video_vel = try allocator.alloc(f32, video_n);
    defer allocator.free(video_vel);
    const audio_vel = try allocator.alloc(f32, audio_n);
    defer allocator.free(audio_vel);
    fillUnitNormal(rng.random(), video);
    fillUnitNormal(rng.random(), audio);
    if (cond.video_patches.len != 0) {
        std.debug.assert(cond.video_patches.len <= video.len);
        @memcpy(video[0..cond.video_patches.len], cond.video_patches);
    }
    if (cond.audio_patches.len != 0) {
        std.debug.assert(cond.audio_patches.len <= audio.len);
        @memcpy(audio[0..cond.audio_patches.len], cond.audio_patches);
    }

    const video_shape = zml.Shape.init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32);
    const audio_shape = zml.Shape.init(.{ .b = 1, .s = geo.audio_tokens, .d = geo.audio_dim }, .f32);
    const seq = layout.seqLen();

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

    var embed_bufs = try loaded.loadEmbed(allocator, io, platform, store, shardings, progress);
    defer dit.EmbedModel.unloadBuffers(&embed_bufs, allocator);
    var embed_runner = try zml.FnExe(dit.embed).Runner(.{.model}).init(&compiled.embed, allocator, .{ .model = embed_bufs });
    defer embed_runner.deinit(allocator);

    var finish_bufs = try loaded.loadFinish(allocator, io, platform, store, shardings, progress);
    defer dit.FinishModel.unloadBuffers(&finish_bufs);
    var finish_runner = try zml.FnExe(dit.finish).Runner(.{.model}).init(&compiled.finish, allocator, .{ .model = finish_bufs });
    defer finish_runner.deinit(allocator);

    const steps = schedules.video.stepCount();
    const n_blocks = loaded.inner.blocks.len;
    const denoise_start: std.Io.Timestamp = .now(io, .awake);
    log.info(
        "denoise: start steps={d} blocks={d} video_tokens={d} audio_tokens={d} seed={d}",
        .{ steps, n_blocks, geo.video_tokens, geo.audio_tokens, seed },
    );
    var step_i: usize = 0;
    while (step_i < steps) : (step_i += 1) {
        const step_start: std.Io.Timestamp = .now(io, .awake);
        const video_t = schedules.video.timesteps[step_i];
        const video_sigma = 1.0 - video_t;
        const audio_sigma = scheduler_mod.timeShiftSigma(video_sigma, opts.video_shift, opts.audio_shift);
        const audio_t = 1.0 - audio_sigma;
        packing.writeTimesteps(host.timesteps, video_t, audio_t);

        var video_buf = try bufferFromItems(io, platform, video_shape, video);
        defer video_buf.deinit();
        var audio_buf = try bufferFromItems(io, platform, audio_shape, audio);
        defer audio_buf.deinit();
        var timestep_buf = try bufferFromItems(io, platform, .init(.{ .n = packing.timestep_slot_count }, .f32), host.timesteps);
        defer timestep_buf.deinit();

        var hidden: zml.Buffer = undefined;
        var temb: zml.Buffer = undefined;
        var cos: zml.Buffer = undefined;
        var sin: zml.Buffer = undefined;
        embed_runner.run(io, .{
            .inputs = .{
                .video = video_buf,
                .audio = audio_buf,
                .text = text,
                .timestep = timestep_buf,
                .position_ids = pos_buf,
                .video_indices = video_idx,
                .audio_indices = audio_idx,
                .text_indices = text_idx,
            },
            .outputs = .{ .hidden = &hidden, .temb = &temb, .cos = &cos, .sin = &sin },
            .opts = .{ .wait = true },
        });
        defer hidden.deinit();
        defer temb.deinit();
        defer cos.deinit();
        defer sin.deinit();

        const DitFut = @TypeOf(try io.concurrent(loadDitBlock, .{
            allocator, io, platform, loaded, store, shardings, @as(usize, 0), progress,
        }));
        var current_f: ?DitFut = if (n_blocks > 0) try io.concurrent(loadDitBlock, .{
            allocator, io, platform, loaded, store, shardings, @as(usize, 0), progress,
        }) else null;
        var next_f: ?DitFut = if (n_blocks > 1) try io.concurrent(loadDitBlock, .{
            allocator, io, platform, loaded, store, shardings, @as(usize, 1), progress,
        }) else null;
        errdefer cancelDit(&current_f, io);
        errdefer cancelDit(&next_f, io);
        var block_i: usize = 0;
        while (block_i < n_blocks) : (block_i += 1) {
            var block_bufs = try current_f.?.await(io);
            current_f = null;
            defer dit.TransformerBlock.unloadBuffers(&block_bufs);
            current_f = next_f;
            next_f = if (block_i + 2 < n_blocks) try io.concurrent(loadDitBlock, .{
                allocator, io, platform, loaded, store, shardings, block_i + 2, progress,
            }) else null;
            var block_runner = try zml.FnExe(dit.TransformerBlock.forward).Runner(.{.layer}).init(&compiled.block, allocator, .{
                .layer = block_bufs,
            });
            defer block_runner.deinit(allocator);

            var next: zml.Buffer = undefined;
            block_runner.run(io, .{
                .inputs = .{
                    .hidden = hidden,
                    .temb = temb,
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
                .temb = temb,
                .timestep_indices = time_idx,
                .video_indices = video_idx,
                .audio_indices = audio_idx,
            },
            .opts = .{ .wait = true },
            .outputs = .{ .video = &video_out, .audio = &audio_out },
        });
        defer video_out.deinit();
        defer audio_out.deinit();

        try video_out.toSlice(io, .init(video_shape, std.mem.sliceAsBytes(video_vel)));
        try audio_out.toSlice(io, .init(audio_shape, std.mem.sliceAsBytes(audio_vel)));
        if (cond.video_patches.len != 0) @memset(video_vel[0..cond.video_patches.len], 0);
        if (cond.audio_patches.len != 0) @memset(audio_vel[0..cond.audio_patches.len], 0);
        schedules.video.step(step_i, video, video_vel);
        schedules.audio.step(step_i, audio, audio_vel);
        if (cond.video_patches.len != 0) @memcpy(video[0..cond.video_patches.len], cond.video_patches);
        if (cond.audio_patches.len != 0) @memcpy(audio[0..cond.audio_patches.len], cond.audio_patches);
        log.info("denoise {d}/{d} t_video={d:.4} t_audio={d:.4} [{f}]", .{
            step_i + 1,
            steps,
            video_t,
            audio_t,
            step_start.untilNow(io, .awake),
        });
    }
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

fn cancelDit(fut: anytype, io: std.Io) void {
    cancelLoad(dit.TransformerBlock.unloadBuffers, fut, io);
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
) !zml.Bufferized(encoder.TransformerLayer) {
    log.debug("encoder layer {d}: load", .{index + 1});
    return loaded.loadLayer(allocator, io, platform, store, shardings, index, progress);
}

fn loadDitBlock(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    loaded: *const dit.LoadedModel,
    store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    index: usize,
    progress: *std.Progress.Node,
) !zml.Bufferized(dit.TransformerBlock) {
    return loaded.loadBlock(allocator, io, platform, store, shardings, index, progress);
}

fn fillUnitNormal(random: std.Random, out: []f32) void {
    var i: usize = 0;
    while (i + 1 < out.len) : (i += 2) {
        const unit_a = @max(random.float(f32), 1e-7);
        const unit_b = random.float(f32);
        const r = @sqrt(-2.0 * @log(unit_a));
        const theta = 2.0 * std.math.pi * unit_b;
        out[i] = r * @cos(theta);
        out[i + 1] = r * @sin(theta);
    }
    if (i < out.len) out[i] = random.floatNorm(f32);
}
