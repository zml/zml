const std = @import("std");

const zml = @import("zml");

const dit = @import("dit.zig");
const encoder = @import("encoder.zig");
const noise = @import("noise.zig");
const packing = @import("packing.zig");
const pipeline = @import("pipeline.zig");
const scheduler_mod = @import("scheduler.zig");
const vae = @import("vae.zig");
const vision = @import("vision.zig");
const weights = @import("weights.zig");

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

pub fn writeF32File(io: std.Io, dir: std.Io.Dir, name: []const u8, values: []const f32) !void {
    const file = try dir.createFile(io, name, .{});
    defer file.close(io);
    var writer = file.writer(io, &.{});
    try writer.interface.writeAll(std.mem.sliceAsBytes(values));
}

pub fn readF32File(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8, expected: usize) ![]f32 {
    const out = try readF32FileAll(allocator, io, dir, name);
    errdefer allocator.free(out);
    if (out.len != expected) return error.LatentSizeMismatch;
    return out;
}

pub fn readF32FileAll(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8) ![]f32 {
    const file = try dir.openFile(io, name, .{});
    defer file.close(io);
    const n = try file.length(io);
    if (n % @sizeOf(f32) != 0) return error.LatentSizeMismatch;
    const out = try allocator.alloc(f32, n / @sizeOf(f32));
    errdefer allocator.free(out);
    var reader = file.reader(io, &.{});
    try reader.interface.readSliceAll(std.mem.sliceAsBytes(out));
    return out;
}

pub fn readU8File(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8, expected: usize) ![]u8 {
    const file = try dir.openFile(io, name, .{});
    defer file.close(io);
    const n = try file.length(io);
    if (n != expected) return error.LatentSizeMismatch;
    const out = try allocator.alloc(u8, expected);
    errdefer allocator.free(out);
    var reader = file.reader(io, &.{});
    try reader.interface.readSliceAll(out);
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
    var embed_bufs = try loaded.loadEmbed(allocator, io, platform, store, shardings, progress);
    defer encoder.EmbedTokens.unloadBuffers(&embed_bufs);
    var embed_runner = try zml.FnExe(encoder.EmbedTokens.forward).Runner(.{.embedding}).init(&compiled.encode_embed, allocator, .{
        .embedding = embed_bufs,
    });
    defer embed_runner.deinit(allocator);

    var loaders = [2]zml.io.Loader{
        try weights.initLoader(allocator, platform),
        try weights.initLoader(allocator, platform),
    };
    defer loaders[0].deinit();
    defer loaders[1].deinit();

    const n_layers = loaded.inner.layers.len;
    const EncFut = @TypeOf(try io.concurrent(loadEncoderLayer, .{
        allocator, io, platform, loaded, store, shardings, @as(usize, 0), progress, &loaders[0],
    }));
    var current_f: ?EncFut = if (n_layers > 0) try io.concurrent(loadEncoderLayer, .{
        allocator, io, platform, loaded, store, shardings, @as(usize, 0), progress, &loaders[0],
    }) else null;
    var next_f: ?EncFut = if (n_layers > 1) try io.concurrent(loadEncoderLayer, .{
        allocator, io, platform, loaded, store, shardings, @as(usize, 1), progress, &loaders[1],
    }) else null;
    errdefer cancelEnc(&current_f, io);
    errdefer cancelEnc(&next_f, io);

    var hidden: zml.Buffer = undefined;
    embed_runner.run(io, .{
        .inputs = .{ .tokens = token_buf },
        .outputs = .{ .hidden = &hidden },
        .opts = .{ .wait = true },
    });
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

    const LayerRunner = zml.FnExe(encoder.TransformerLayer.forward).Runner(.{.layer});
    var layer_runner: ?LayerRunner = null;
    defer if (layer_runner) |*r| r.deinit(allocator);
    var layer_i: usize = 0;
    while (layer_i < n_layers) : (layer_i += 1) {
        var layer_bufs = try current_f.?.await(io);
        current_f = null;
        defer encoder.TransformerLayer.unloadBuffers(&layer_bufs);
        current_f = next_f;
        next_f = if (layer_i + 2 < n_layers) try io.concurrent(loadEncoderLayer, .{
            allocator, io, platform, loaded, store, shardings, layer_i + 2, progress, &loaders[(layer_i + 2) % 2],
        }) else null;
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
    return hidden;
}

pub const SharedInputs = struct {
    video_rows: []f32,
    audio_rows: []f32,
    embeds: []f32,
    tags: []u8,

    pub fn deinit(self: SharedInputs, allocator: std.mem.Allocator) void {
        allocator.free(self.video_rows);
        allocator.free(self.audio_rows);
        allocator.free(self.embeds);
        allocator.free(self.tags);
    }
};

pub fn loadShared(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    geo: pipeline.Geometry,
    text_dim: u32,
) !SharedInputs {
    const channels: u32 = 24;
    const patch = [_]i64{ 1, 2, 2 };
    const nchw_n = @as(usize, channels) * geo.latent_t * geo.latent_h * geo.latent_w;
    const nchw = try readF32File(allocator, io, dir, "video_noise.f32", nchw_n);
    defer allocator.free(nchw);
    const thwc = try allocator.alloc(f32, nchw_n);
    defer allocator.free(thwc);
    packing.nchwToThwc(thwc, nchw, channels, geo.latent_t, geo.latent_h, geo.latent_w);
    const video_rows = try packing.patchify(allocator, thwc, geo.latent_t, geo.latent_h, geo.latent_w, channels, patch);
    errdefer allocator.free(video_rows);

    const bct_n = @as(usize, 2) * geo.audio_dim * geo.audio_t;
    const bct = try readF32File(allocator, io, dir, "audio_noise.f32", bct_n);
    defer allocator.free(bct);
    const audio_rows = try allocator.alloc(f32, bct_n);
    errdefer allocator.free(audio_rows);
    vae.audioBctToRows(audio_rows, bct, geo.audio_dim, geo.audio_t);

    const embeds = try readF32FileAll(allocator, io, dir, "prompt_embeds.f32");
    errdefer allocator.free(embeds);
    if (embeds.len == 0 or embeds.len % text_dim != 0) return error.SharedEmbedSize;
    const text_len = embeds.len / text_dim;
    const tags = readU8File(allocator, io, dir, "text_tags.u8", text_len) catch |err| switch (err) {
        error.FileNotFound => blk: {
            const filled = try allocator.alloc(u8, text_len);
            @memset(filled, @intFromEnum(packing.Modality.text));
            break :blk filled;
        },
        else => return err,
    };

    log.info(
        "shared: video_rows={d} audio_rows={d} embeds={d}x{d}",
        .{ video_rows.len, audio_rows.len, text_len, text_dim },
    );
    return .{
        .video_rows = video_rows,
        .audio_rows = audio_rows,
        .embeds = embeds,
        .tags = tags,
    };
}

pub const DenoiseCond = struct {
    videos: []const packing.ConditionVideo = &.{},
    video_patches: []const f32 = &.{},
    audio_patches: []const f32 = &.{},
    video_noise: ?[]const f32 = null,
    audio_noise: ?[]const f32 = null,
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
    var video = if (cond.video_noise) |src| blk: {
        const copy = try allocator.dupe(f32, src);
        break :blk copy;
    } else try noise.drawVideo(
        allocator,
        &gen,
        cond.videos,
        cond.video_patches,
        geo.latent_t,
        geo.latent_h,
        geo.latent_w,
        loaded.inner.cfg.patch_size,
    );
    errdefer allocator.free(video);
    var audio = if (cond.audio_noise) |src| blk: {
        const copy = try allocator.dupe(f32, src);
        break :blk copy;
    } else try noise.drawAudio(allocator, &gen, cond.audio_patches, geo.audio_dim, geo.audio_t);
    errdefer allocator.free(audio);
    if (video.len != geo.video_tokens * geo.video_patch_dim) return error.VideoNoiseSize;
    if (audio.len != geo.audio_tokens * geo.audio_dim) return error.AudioNoiseSize;
    const video_vel = try allocator.alloc(f32, video.len);
    defer allocator.free(video_vel);
    const audio_vel = try allocator.alloc(f32, audio.len);
    defer allocator.free(audio_vel);
    const held_video = try allocator.dupe(f32, video[0..cond.video_patches.len]);
    defer allocator.free(held_video);
    const held_audio = try allocator.dupe(f32, audio[0..cond.audio_patches.len]);
    defer allocator.free(held_audio);

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

    var loaders = [2]zml.io.Loader{
        try weights.initLoader(allocator, platform),
        try weights.initLoader(allocator, platform),
    };
    defer loaders[0].deinit();
    defer loaders[1].deinit();
    const EmbedRunner = zml.FnExe(dit.embed).Runner(.{.model});
    var embed_runner: ?EmbedRunner = null;
    defer if (embed_runner) |*r| r.deinit(allocator);
    const FinishRunner = zml.FnExe(dit.finish).Runner(.{.model});
    var finish_runner: ?FinishRunner = null;
    defer if (finish_runner) |*r| r.deinit(allocator);
    const BlockRunner = zml.FnExe(dit.TransformerBlock.forward).Runner(.{.layer});
    var block_runner: ?BlockRunner = null;
    defer if (block_runner) |*r| r.deinit(allocator);

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
        {
            var embed_bufs = try loaded.loadEmbed(allocator, io, platform, store, shardings, progress);
            defer dit.EmbedModel.unloadBuffers(&embed_bufs, allocator);
            if (embed_runner) |*r| {
                weights.rebake(r, .{ .model = embed_bufs });
            } else {
                embed_runner = try EmbedRunner.init(&compiled.embed, allocator, .{ .model = embed_bufs });
            }
            embed_runner.?.run(io, .{
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
        }
        defer hidden.deinit();
        defer temb.deinit();
        defer cos.deinit();
        defer sin.deinit();

        const DitFut = @TypeOf(try io.concurrent(loadDitBlock, .{
            allocator, io, platform, loaded, store, shardings, @as(usize, 0), progress, &loaders[0],
        }));
        var current_f: ?DitFut = if (n_blocks > 0) try io.concurrent(loadDitBlock, .{
            allocator, io, platform, loaded, store, shardings, @as(usize, 0), progress, &loaders[0],
        }) else null;
        var next_f: ?DitFut = if (n_blocks > 1) try io.concurrent(loadDitBlock, .{
            allocator, io, platform, loaded, store, shardings, @as(usize, 1), progress, &loaders[1],
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
                allocator, io, platform, loaded, store, shardings, block_i + 2, progress, &loaders[(block_i + 2) % 2],
            }) else null;
            if (block_runner) |*r| {
                weights.rebake(r, .{ .layer = block_bufs });
            } else {
                block_runner = try BlockRunner.init(&compiled.block, allocator, .{ .layer = block_bufs });
            }

            var next: zml.Buffer = undefined;
            block_runner.?.run(io, .{
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
        {
            var finish_bufs = try loaded.loadFinish(allocator, io, platform, store, shardings, progress);
            defer dit.FinishModel.unloadBuffers(&finish_bufs);
            if (finish_runner) |*r| {
                weights.rebake(r, .{ .model = finish_bufs });
            } else {
                finish_runner = try FinishRunner.init(&compiled.finish, allocator, .{ .model = finish_bufs });
            }
            finish_runner.?.run(io, .{
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
        }
        defer video_out.deinit();
        defer audio_out.deinit();

        try video_out.toSlice(io, .init(video_shape, std.mem.sliceAsBytes(video_vel)));
        try audio_out.toSlice(io, .init(audio_shape, std.mem.sliceAsBytes(audio_vel)));
        if (cond.video_patches.len != 0) @memset(video_vel[0..cond.video_patches.len], 0);
        if (cond.audio_patches.len != 0) @memset(audio_vel[0..cond.audio_patches.len], 0);
        schedules.video.step(step_i, video, video_vel);
        schedules.audio.step(step_i, audio, audio_vel);
        if (held_video.len != 0) @memcpy(video[0..held_video.len], held_video);
        if (held_audio.len != 0) @memcpy(audio[0..held_audio.len], held_audio);
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
    loader: *zml.io.Loader,
) !zml.Bufferized(encoder.TransformerLayer) {
    return loaded.loadLayer(allocator, io, platform, store, shardings, index, progress, loader);
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
    loader: *zml.io.Loader,
) !zml.Bufferized(dit.TransformerBlock) {
    return loaded.loadBlock(allocator, io, platform, store, shardings, index, progress, loader);
}

