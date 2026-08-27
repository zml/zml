const std = @import("std");

const zml = @import("zml");

const repository = @import("repository.zig");
const config = @import("../core/config.zig");
const decode = @import("decode.zig");
const dit = @import("../model/dit.zig");
const encoder = @import("../model/encoder.zig");
const media = @import("media.zig");
const noise = @import("../model/noise.zig");
const packing = @import("../model/packing.zig");
const pipeline = @import("pipeline.zig");
const policy = @import("../core/policy.zig");
const presentation = @import("../conditioning/presentation.zig");
const scheduler_mod = @import("../model/scheduler.zig");
const vision = @import("../model/vision.zig");
const weights = @import("../core/weights.zig");

const log = std.log.scoped(.minimax_h3);

const HostLayout = struct {
    positions: []f32,
    text_indices: []u32,
    video_indices: []u32,
    audio_indices: []u32,

    fn fromLayout(allocator: std.mem.Allocator, layout: packing.Layout) !HostLayout {
        const positions = try allocator.alloc(f32, layout.positions.len * 3);
        errdefer allocator.free(positions);
        for (layout.positions, 0..) |pos, i| {
            positions[i * 3 + 0] = pos.t;
            positions[i * 3 + 1] = pos.h;
            positions[i * 3 + 2] = pos.w;
        }

        const text_indices = try allocator.dupe(u32, layout.text_indices);
        errdefer allocator.free(text_indices);
        const video_indices = try allocator.dupe(u32, layout.video_indices);
        errdefer allocator.free(video_indices);
        return .{
            .positions = positions,
            .text_indices = text_indices,
            .video_indices = video_indices,
            .audio_indices = try allocator.dupe(u32, layout.audio_indices),
        };
    }

    fn deinit(self: HostLayout, allocator: std.mem.Allocator) void {
        allocator.free(self.positions);
        allocator.free(self.text_indices);
        allocator.free(self.video_indices);
        allocator.free(self.audio_indices);
    }
};

const Latents = struct {
    video: []f32,
    audio: []f32,

    pub fn deinit(self: Latents, allocator: std.mem.Allocator) void {
        allocator.free(self.video);
        allocator.free(self.audio);
    }
};

fn scalarU32(io: std.Io, platform: *const zml.Platform, value: u32) !zml.Buffer {
    var item: u32 = value;
    return zml.Buffer.fromBytes(io, platform, .init(.{}, .u32), .replicated, std.mem.asBytes(&item));
}

fn scalarF32(io: std.Io, platform: *const zml.Platform, value: f32) !zml.Buffer {
    var item: f32 = value;
    return zml.Buffer.fromBytes(io, platform, .init(.{}, .f32), .replicated, std.mem.asBytes(&item));
}

pub const TextExtras = struct {
    positions: ?[]const f32 = null,
    deepstack: [3]?[]const f32 = .{ null, null, null },
    vision_merged: ?[]const f32 = null,
    vision_spans: []const presentation.VisionSpan = &.{},
};

fn encodeText(
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
    var token_buf = try weights.fromItems(io, platform, token_shape, tokens);
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
        const scatter_exe = if (compiled.encode_scatter) |*exe| exe else return error.VisionScatterUncompiled;
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
        var val_buf = try weights.fromItems(io, platform, .init(.{ .b = 1, .s = n_vis, .d = hidden_dim }, .f32), merged);
        defer val_buf.deinit();
        var idx_buf = try weights.fromItems(io, platform, .init(.{ .s = n_vis }, .u32), idx);
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
    var cos_buf = try weights.fromF32(allocator, io, platform, .init(.{ .s = seq, .hd = head_dim }, loaded.inner.embed_tokens.weight.dtype()), cos);
    defer cos_buf.deinit();
    var sin_buf = try weights.fromF32(allocator, io, platform, .init(.{ .s = seq, .hd = head_dim }, loaded.inner.embed_tokens.weight.dtype()), sin);
    defer sin_buf.deinit();

    const zeros = try allocator.alloc(f32, seq * hidden_dim);
    defer allocator.free(zeros);
    @memset(zeros, 0);
    var zero_delta = try weights.fromF32(allocator, io, platform, .init(.{ .b = 1, .s = seq, .d = hidden_dim }, loaded.inner.embed_tokens.weight.dtype()), zeros);
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
                owned_delta = try weights.fromF32(allocator, io, platform, .init(.{ .b = 1, .s = seq, .d = hidden_dim }, loaded.inner.embed_tokens.weight.dtype()), host);
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

pub const DenoiseCond = struct {
    videos: []const packing.ConditionVideo = &.{},
    video_patches: []const f32 = &.{},
    audio_patches: []const f32 = &.{},
};

fn denoise(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled: *const pipeline.Compiled,
    loaded: *const dit.LoadedModel,
    store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    geo: pipeline.Geometry,
    text: zml.Buffer,
    text_len: u32,
    layout: packing.Layout,
    schedules: scheduler_mod.DualSchedule,
    seed: u64,
    cond: DenoiseCond,
    resident_blocks: u32,
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
    );
    errdefer allocator.free(video);
    var audio = try noise.drawAudio(allocator, &gen, cond.audio_patches, geo.audio_dim, geo.audio_t);
    errdefer allocator.free(audio);
    if (video.len != geo.video_tokens * geo.video_patch_dim) return error.VideoNoiseSize;
    if (audio.len != geo.audio_tokens * geo.audio_dim) return error.AudioNoiseSize;

    const video_shape = zml.Shape.init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32);
    const audio_shape = zml.Shape.init(.{ .b = 1, .s = geo.audio_tokens, .d = geo.audio_dim }, .f32);
    const seq = layout.seqLen();
    const steps = schedules.video.stepCount();
    const n_blocks = loaded.inner.blocks.len;

    var host = try HostLayout.fromLayout(allocator, layout);
    defer host.deinit(allocator);

    const flat_n = steps * packing.timestep_slot_count;
    const flat_t = try allocator.alloc(f32, flat_n);
    defer allocator.free(flat_t);
    const all_tidx = try allocator.alloc(u32, steps * seq);
    defer allocator.free(all_tidx);
    const all_adaln = try allocator.alloc(u32, steps * seq);
    defer allocator.free(all_adaln);
    const row_ts = try allocator.alloc(f32, seq);
    defer allocator.free(row_ts);
    for (0..steps) |i| {
        const tidx = all_tidx[i * seq ..][0..seq];
        _ = packing.writeRowPlan(
            layout,
            schedules.video.timesteps[i],
            schedules.audio.timesteps[i],
            row_ts,
            tidx,
            flat_t[i * packing.timestep_slot_count ..][0..packing.timestep_slot_count],
        );
        packing.writeAdalnIndices(all_adaln[i * seq ..][0..seq], tidx, layout.token_tags);
    }

    var pos_buf = try weights.fromItems(io, platform, .init(.{ .s = seq, .ax = 3 }, .f32), host.positions);
    defer pos_buf.deinit();
    var video_idx = try weights.fromItems(io, platform, .init(.{ .s = geo.video_tokens }, .u32), host.video_indices);
    defer video_idx.deinit();
    var audio_idx = try weights.fromItems(io, platform, .init(.{ .s = geo.audio_tokens }, .u32), host.audio_indices);
    defer audio_idx.deinit();
    var text_idx = try weights.fromItems(io, platform, .init(.{ .s = text_len }, .u32), host.text_indices);
    defer text_idx.deinit();
    var adaln_buf = try weights.fromItems(io, platform, .init(.{ .s = seq }, .u32), all_adaln[0..seq]);
    defer adaln_buf.deinit();
    var time_idx = try weights.fromItems(io, platform, .init(.{ .s = seq }, .u32), all_tidx[0..seq]);
    defer time_idx.deinit();

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

    var flat_buf = try weights.fromItems(io, platform, .init(.{ .n = flat_n }, .f32), flat_t);
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
    const n_resident = policy.ditKeepBlocks(resident_blocks, @intCast(n_blocks));
    const group_size = @max(1, @min(compiled.group_size, n_resident));
    log.info(
        "denoise: prepare blocks={d} resident={d} keep={d} group={d} core={d}MiB",
        .{
            n_blocks,
            resident_blocks,
            n_resident,
            group_size,
            core_bytes / (1024 * 1024),
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
    const use_group = compiled.block_group != null and group_size > 1 and group_size == compiled.group_size;
    if (use_group) {
        group_layers = try allocator.alloc(zml.Bufferized(dit.BlockCore), group_size);
        group_tables = try allocator.alloc(zml.Buffer, group_size);
    }

    var apply_v = try zml.FnExe(scheduler_mod.apply).Runner(.{}).init(&compiled.apply_video, allocator, .{});
    defer apply_v.deinit(allocator);
    var apply_a = try zml.FnExe(scheduler_mod.apply).Runner(.{}).init(&compiled.apply_audio, allocator, .{});
    defer apply_a.deinit(allocator);

    var video_buf = try weights.fromItems(io, platform, video_shape, video);
    defer video_buf.deinit();
    var audio_buf = try weights.fromItems(io, platform, audio_shape, audio);
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
        const audio_t = schedules.audio.timesteps[step_i];
        if (step_i != 0) {
            adaln_buf.deinit();
            adaln_buf = try weights.fromItems(io, platform, .init(.{ .s = seq }, .u32), all_adaln[step_i * seq ..][0..seq]);
            time_idx.deinit();
            time_idx = try weights.fromItems(io, platform, .init(.{ .s = seq }, .u32), all_tidx[step_i * seq ..][0..seq]);
        }
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
        const DitFut = @TypeOf(try io.concurrent(loadDitCore, .{
            allocator, io, platform, loaded, store, shardings, @as(usize, 0), progress, &loaders[0],
        }));
        var prefetch_core: ?zml.Bufferized(dit.BlockCore) = null;
        defer if (prefetch_core) |*c| dit.BlockCore.unloadBuffers(c);
        while (i < n_blocks) : (i += 1) {
            var owned_core: ?zml.Bufferized(dit.BlockCore) = null;
            defer if (owned_core) |*c| dit.BlockCore.unloadBuffers(c);
            const core = if (cores[i]) |c| c else blk: {
                if (prefetch_core) |c| {
                    owned_core = c;
                    prefetch_core = null;
                    break :blk owned_core.?;
                }
                owned_core = try loaded.loadCore(allocator, io, platform, store, shardings, i, progress, &loaders[i % 2]);
                break :blk owned_core.?;
            };
            if (block_runner) |*r| {
                weights.rebake(r, .{ .layer = core });
            } else {
                block_runner = try BlockRunner.init(&compiled.block, allocator, .{ .layer = core });
            }
            var next_fut: ?DitFut = null;
            errdefer cancelDit(&next_fut, io);
            if (i + 1 < n_blocks and cores[i + 1] == null) {
                next_fut = try io.concurrent(loadDitCore, .{
                    allocator, io, platform, loaded, store, shardings, i + 1, progress, &loaders[(i + 1) % 2],
                });
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
            if (next_fut) |*f| {
                prefetch_core = try f.await(io);
                next_fut = null;
            }
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

fn cancelDit(fut: anytype, io: std.Io) void {
    cancelLoad(dit.BlockCore.unloadBuffers, fut, io);
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

fn loadDitCore(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    loaded: *const dit.LoadedModel,
    store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    index: usize,
    progress: *std.Progress.Node,
    loader: *zml.io.Loader,
) !zml.Bufferized(dit.BlockCore) {
    return loaded.loadCore(allocator, io, platform, store, shardings, index, progress, loader);
}

pub const Generate = struct {
    opts: pipeline.Options,
    geo: pipeline.Geometry,
    target: pipeline.Geometry,
    tokens: []const u32,
    extras: TextExtras,
    layout: packing.Layout,
    schedules: scheduler_mod.DualSchedule,
    cond: DenoiseCond,
    seed: u64,
    resident_blocks: u32,
    out: []const u8,
};

pub fn generate(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    models: *repository.Bundle,
    compiled: *const pipeline.Compiled,
    compiled_vae: *pipeline.VaeCompiled,
    shardings: []const zml.Sharding,
    progress: *std.Progress.Node,
    req: Generate,
) !void {
    const dest = media.Output.parse(req.out);
    if (!dest.isCwd()) try std.Io.Dir.cwd().createDirPath(io, dest.dir);
    var out_dir: std.Io.Dir = if (dest.isCwd())
        .cwd()
    else
        try std.Io.Dir.cwd().openDir(io, dest.dir, .{});
    defer if (!dest.isCwd()) out_dir.close(io);
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

    const device_bytes = config.minDeviceBytes(platform);
    const prefetch_vae = device_bytes == 0 or device_bytes >= config.full_canvas_min_device_bytes;
    const VaeCacheFut = @TypeOf(try io.concurrent(decode.loadVisualCache, .{
        allocator,
        io,
        platform,
        &models.visual,
        &models.visual_store,
        shardings,
        models.visual.inner.blocks.len,
        progress,
    }));
    var cache_f: ?VaeCacheFut = null;
    var cache_taken = false;
    errdefer if (!cache_taken) {
        if (cache_f) |*f| {
            if (f.cancel(io)) |c| {
                var cache = c;
                cache.deinit(allocator);
            } else |_| {}
        }
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

    if (prefetch_vae) {
        cache_f = try io.concurrent(decode.loadVisualCache, .{
            allocator,
            io,
            platform,
            &models.visual,
            &models.visual_store,
            shardings,
            models.visual.inner.blocks.len,
            progress,
        });
    }

    var latents = try denoise(
        allocator,
        io,
        platform,
        compiled,
        &models.dit,
        &models.dit_store,
        shardings,
        req.geo,
        text,
        @intCast(req.tokens.len),
        req.layout,
        req.schedules,
        req.seed,
        req.cond,
        req.resident_blocks,
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
    var owned_cache: ?decode.VisualCache = if (cache_f) |*f| try f.await(io) else null;
    cache_taken = true;
    defer if (owned_cache) |*c| c.deinit(allocator);
    const cache_arg: ?*decode.VisualCache = if (owned_cache) |*c| c else null;
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
        cache_arg,
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
