const std = @import("std");

const zml = @import("zml");

const repository = @import("../serve/repo.zig");
const audio_vae = @import("audio.zig");
const decode = @import("decode.zig");
const dit = @import("dit.zig");
const encoder = @import("encoder.zig");
const noise = @import("noise.zig");
const packing = @import("packing.zig");
const pipeline = @import("pipeline.zig");
const scheduler = @import("scheduler.zig");
const taeh3 = @import("taeh3.zig");
const rope = @import("rope.zig");
const weights = @import("../recipe/weights.zig");

const prepare = @import("../refine/handoff.zig");
const policy = @import("../recipe/policy.zig");

const log = std.log.scoped(.minimax_h3);

// =============================================================================
// draft/session.zig — one request: encode, denoise, TAEH3, audio
//
// Handoff RGB is produced on GPU. Audio VAE runs concurrently.
// =============================================================================

pub const JobProgress = struct {
    pub const Stage = enum(u32) {
        idle,
        text,
        draft,
        taeh3,
        vae,
        refine,
        taehv,
        remux,
        done,
    };

    stage: std.atomic.Value(u32) = .init(@intFromEnum(Stage.idle)),
    step: std.atomic.Value(u32) = .init(0),
    total: std.atomic.Value(u32) = .init(0),

    pub fn set(self: *JobProgress, stage: Stage, step: u32, total: u32) void {
        self.stage.store(@intFromEnum(stage), .release);
        self.step.store(step, .release);
        self.total.store(total, .release);
    }

    pub fn clear(self: *JobProgress) void {
        self.set(.idle, 0, 0);
    }
};

pub const Warm = struct {
    enc: []zml.Bufferized(encoder.TransformerLayer),
    embed: ?zml.Bufferized(encoder.EmbedTokens) = null,
    text_prep: ?zml.Bufferized(dit.TextPrep) = null,
    patch: ?zml.Bufferized(dit.PatchEmbed) = null,
    finish: ?zml.Bufferized(dit.FinishCore) = null,
    cores: []?zml.Bufferized(dit.BlockCore),
    audio_bufs: ?zml.Bufferized(audio_vae.Model) = null,

    pub fn deinit(self: *Warm, allocator: std.mem.Allocator) void {
        for (self.enc) |*layer| encoder.TransformerLayer.unloadBuffers(layer);
        if (self.enc.len != 0) allocator.free(self.enc);
        if (self.embed) |*e| encoder.EmbedTokens.unloadBuffers(e);
        if (self.text_prep) |*t| dit.TextPrep.unloadBuffers(t, allocator);
        if (self.patch) |*p| dit.PatchEmbed.unloadBuffers(p);
        if (self.finish) |*f| dit.FinishCore.unloadBuffers(f);
        for (self.cores) |*c| if (c.*) |*core| dit.BlockCore.unloadBuffers(core);
        if (self.cores.len != 0) allocator.free(self.cores);
        if (self.audio_bufs) |*b| audio_vae.Model.unloadBuffers(b, allocator);
        self.* = .{ .enc = &.{}, .cores = &.{} };
    }
};

pub const Bake = struct {
    tables: []zml.Buffer = &.{},
    final_table: ?zml.Buffer = null,
    audio_exe: ?zml.FnExe(audio_vae.decode) = null,
    owns_tables: bool = true,

    pub fn shareTables(self: *Bake, src: *const Bake) void {
        self.tables = src.tables;
        self.final_table = src.final_table;
        self.owns_tables = false;
    }

    pub fn deinit(self: *Bake, allocator: std.mem.Allocator) void {
        if (self.owns_tables) {
            for (self.tables) |*t| t.deinit();
            if (self.tables.len != 0) allocator.free(self.tables);
            if (self.final_table) |*t| t.deinit();
        }
        if (self.audio_exe) |*exe| exe.deinit();
        self.* = .{};
    }
};

pub const Draft = struct {
    pixels: zml.Buffer,
    wav: []f32,

    pub fn deinit(self: *Draft, allocator: std.mem.Allocator) void {
        self.pixels.deinit();
        allocator.free(self.wav);
    }
};

pub const DraftReq = struct {
    geo: pipeline.Geometry,
    tokens: []const u32,
    layout: packing.Layout,
    schedules: scheduler.DualSchedule,
    seed: u64,
    warm: *Warm,
    bake: *Bake,
    taeh3: *const taeh3.Compiled,
    handoff: *const prepare.Compiled,
    resident_blocks: u32 = 0,
    job: ?*JobProgress = null,
};

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

fn encodeText(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled: *const pipeline.Compiled,
    loaded: *const encoder.LoadedModel,
    tokens: []const u32,
    warm: *Warm,
) !zml.Buffer {
    const embed_bufs = warm.embed orelse return error.EncoderMissing;
    if (warm.enc.len != loaded.inner.layers.len) return error.EncoderMissing;
    const seq: u32 = @intCast(tokens.len);
    const head_dim: u32 = @intCast(loaded.cfg.head_dim);

    const token_shape = zml.Shape.init(.{ .b = 1, .s = tokens.len }, .u32);
    var token_buf = try weights.fromItems(io, platform, token_shape, tokens);
    defer token_buf.deinit();
    const encode_start: std.Io.Timestamp = .now(io, .awake);
    const n_layers = warm.enc.len;
    log.info("encoder: start tokens={d} layers={d} resident=true", .{ tokens.len, n_layers });
    var embed_runner = try zml.FnExe(encoder.EmbedTokens.forward).Runner(.{.embedding}).init(&compiled.encode_embed, allocator, .{
        .embedding = embed_bufs,
    });
    defer embed_runner.deinit(allocator);

    var hidden: zml.Buffer = undefined;
    embed_runner.run(io, .{
        .inputs = .{ .tokens = token_buf },
        .outputs = .{ .hidden = &hidden },
        .opts = .{ .wait = true },
    });
    errdefer hidden.deinit();
    const pos = try allocator.alloc(f32, seq * 3);
    defer allocator.free(pos);
    rope.fillArangePositions(pos, seq);
    const cos = try allocator.alloc(f32, seq * head_dim);
    defer allocator.free(cos);
    const sin = try allocator.alloc(f32, seq * head_dim);
    defer allocator.free(sin);
    rope.hostInterleavedMrope(pos, seq, head_dim, loaded.cfg.rope_theta, loaded.cfg.mrope_section, cos, sin);
    var cos_buf = try weights.fromF32(allocator, io, platform, .init(.{ .s = seq, .hd = head_dim }, loaded.inner.embed_tokens.weight.dtype()), cos);
    defer cos_buf.deinit();
    var sin_buf = try weights.fromF32(allocator, io, platform, .init(.{ .s = seq, .hd = head_dim }, loaded.inner.embed_tokens.weight.dtype()), sin);
    defer sin_buf.deinit();

    const LayerRunner = zml.FnExe(encoder.TransformerLayer.forward).Runner(.{.layer});
    var layer_runner: ?LayerRunner = null;
    defer if (layer_runner) |*r| r.deinit(allocator);
    var layer_i: usize = 0;
    while (layer_i < n_layers) : (layer_i += 1) {
        const layer_bufs = warm.enc[layer_i];
        if (layer_runner) |*r| {
            r.rebake(.{ .layer = layer_bufs });
        } else {
            layer_runner = try LayerRunner.init(&compiled.encode_layer, allocator, .{ .layer = layer_bufs });
        }

        var next: zml.Buffer = undefined;
        layer_runner.?.run(io, .{
            .inputs = .{ .hidden = hidden, .cos = cos_buf, .sin = sin_buf },
            .outputs = .{ .hidden = &next },
            .opts = .{ .wait = true },
        });
        hidden.deinit();
        hidden = next;
    }
    log.info("encoder: ok tokens={d} layers={d} resident=true [{f}]", .{
        tokens.len,
        n_layers,
        encode_start.untilNow(io, .awake),
    });
    return hidden;
}

fn cancelDit(comptime Fut: type, fut: *?Fut, io: std.Io) void {
    if (fut.*) |*f| {
        if (f.cancel(io)) |core| {
            var owned = core;
            dit.BlockCore.unloadBuffers(&owned);
        } else |_| {}
        fut.* = null;
    }
}

fn denoise(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled: *const pipeline.Compiled,
    loaded: *const dit.LoadedModel,
    models: *repository.Bundle,
    shardings: []const zml.Sharding,
    geo: pipeline.Geometry,
    text: zml.Buffer,
    text_len: u32,
    layout: packing.Layout,
    schedules: scheduler.DualSchedule,
    seed: u64,
    warm: *Warm,
    bake: *const Bake,
    resident_blocks: u32,
    progress: *std.Progress.Node,
    job: ?*JobProgress,
) !Latents {
    var gen = noise.Generator.init(seed);
    const video = try noise.drawVideo(
        allocator,
        &gen,
        geo.latent_t,
        geo.latent_h,
        geo.latent_w,
        loaded.inner.cfg.patch_size,
    );
    errdefer allocator.free(video);
    const audio = try noise.drawAudio(allocator, &gen, geo.audio_dim, geo.audio_t);
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

    log.info("denoise: host plan seq={d} steps={d}", .{ seq, steps });
    const text_bufs = warm.text_prep orelse return error.DitMissing;
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

    if (bake.tables.len != n_blocks or bake.final_table == null) return error.DitMissing;
    if (warm.cores.len != n_blocks) return error.DitMissing;
    const tables = bake.tables;
    const cores = warm.cores;
    const final_table = bake.final_table.?;
    const n_resident = if (resident_blocks == 0)
        @as(u32, @intCast(n_blocks))
    else
        policy.ditKeepBlocks(resident_blocks, @intCast(n_blocks));
    const group_size = @max(1, @min(compiled.group_size, n_resident));
    const use_group = compiled.block_group != null and group_size > 1 and group_size == compiled.group_size;
    const core0 = loaded.inner.blocks[0].corePart();
    log.info(
        "denoise: prepare blocks={d} resident={d} group={d} core={d}MiB baked=true",
        .{
            n_blocks,
            n_resident,
            group_size,
            weights.modelBytes(&core0) / (1024 * 1024),
        },
    );

    const patch_bufs = warm.patch orelse return error.DitMissing;
    var patch_runner = try zml.FnExe(dit.embedPatches).Runner(.{.model}).init(&compiled.embed_patches, allocator, .{ .model = patch_bufs });
    defer patch_runner.deinit(allocator);

    const finish_bufs = warm.finish orelse return error.DitMissing;
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
    if (use_group) {
        group_layers = try allocator.alloc(zml.Bufferized(dit.BlockCore), group_size);
        group_tables = try allocator.alloc(zml.Buffer, group_size);
    }
    var stream_loaders: [2]zml.io.Loader = undefined;
    var stream_loaders_init = false;
    defer if (stream_loaders_init) {
        stream_loaders[0].deinit();
        stream_loaders[1].deinit();
    };
    if (n_resident < n_blocks) {
        stream_loaders[0] = try weights.initLoader(allocator, platform);
        stream_loaders[1] = try weights.initLoader(allocator, platform);
        stream_loaders_init = true;
    }

    var apply_v = try zml.FnExe(scheduler.apply).Runner(.{}).init(&compiled.apply_video, allocator, .{});
    defer apply_v.deinit(allocator);
    var apply_a = try zml.FnExe(scheduler.apply).Runner(.{}).init(&compiled.apply_audio, allocator, .{});
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
            .opts = .{ .wait = false },
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
                    r.rebake(.{ .group = .{ .layers = group_layers } });
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
                    .opts = .{ .wait = false },
                });
                hidden.deinit();
                hidden = next;
                i += group_size;
            }
        }
        const DitFut = @TypeOf(try io.concurrent(loadDitCore, .{
            allocator,
            io,
            platform,
            loaded,
            &models.dit_store,
            shardings,
            @as(usize, 0),
            progress,
            &stream_loaders[0],
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
                owned_core = try loadDitCore(
                    allocator,
                    io,
                    platform,
                    loaded,
                    &models.dit_store,
                    shardings,
                    i,
                    progress,
                    &stream_loaders[i % 2],
                );
                break :blk owned_core.?;
            };
            if (block_runner) |*r| {
                r.rebake(.{ .layer = core });
            } else {
                block_runner = try BlockRunner.init(&compiled.block, allocator, .{ .layer = core });
            }
            var next_fut: ?DitFut = null;
            errdefer cancelDit(DitFut, &next_fut, io);
            if (i + 1 < n_blocks and cores[i + 1] == null) {
                next_fut = try io.concurrent(loadDitCore, .{
                    allocator,
                    io,
                    platform,
                    loaded,
                    &models.dit_store,
                    shardings,
                    i + 1,
                    progress,
                    &stream_loaders[(i + 1) % 2],
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
                .opts = .{ .wait = false },
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

        log.info("denoise {d}/{d} t_video={d:.4} t_audio={d:.4} sigma_v={d:.6} sigma_a={d:.6} [{f}]", .{
            step_i + 1,
            steps,
            video_t,
            audio_t,
            schedules.video.sigmas[step_i],
            schedules.audio.sigmas[step_i],
            step_start.untilNow(io, .awake),
        });
        if (job) |j| j.set(.draft, @intCast(step_i + 1), @intCast(steps));
    }

    try video_buf.toSlice(io, .init(video_shape, std.mem.sliceAsBytes(video)));
    try audio_buf.toSlice(io, .init(audio_shape, std.mem.sliceAsBytes(audio)));

    log.info("denoise: ok steps={d} [{f}]", .{ steps, denoise_start.untilNow(io, .awake) });
    return .{ .video = video, .audio = audio };
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

fn loadResidentEncoder(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    models: *repository.Bundle,
    shardings: []const zml.Sharding,
    progress: *std.Progress.Node,
) ![]zml.Bufferized(encoder.TransformerLayer) {
    const n = models.enc.inner.layers.len;
    const layers = try allocator.alloc(zml.Bufferized(encoder.TransformerLayer), n);
    var loaded: usize = 0;
    errdefer {
        for (layers[0..loaded]) |*layer| encoder.TransformerLayer.unloadBuffers(layer);
        allocator.free(layers);
    }
    var loader = try weights.initLoader(allocator, platform);
    defer loader.deinit();
    const load_start: std.Io.Timestamp = .now(io, .awake);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        layers[i] = try loadEncoderLayer(
            allocator,
            io,
            platform,
            &models.enc,
            &models.enc_store,
            shardings,
            i,
            progress,
            &loader,
        );
        loaded += 1;
    }
    log.info("encoder: resident {d} layers [{f}]", .{ n, load_start.untilNow(io, .awake) });
    return layers;
}

pub fn loadWarm(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    models: *repository.Bundle,
    shardings: []const zml.Sharding,
    progress: *std.Progress.Node,
    resident_blocks: u32,
) !Warm {
    const enc = try loadResidentEncoder(allocator, io, platform, models, shardings, progress);
    errdefer {
        for (enc) |*layer| encoder.TransformerLayer.unloadBuffers(layer);
        if (enc.len != 0) allocator.free(enc);
    }
    const n = models.dit.inner.blocks.len;
    const cores = try allocator.alloc(?zml.Bufferized(dit.BlockCore), n);
    @memset(cores, null);
    var loaded_n: usize = 0;
    errdefer {
        for (cores[0..loaded_n]) |*c| if (c.*) |*core| dit.BlockCore.unloadBuffers(core);
        allocator.free(cores);
    }
    var loader = try weights.initLoader(allocator, platform);
    defer loader.deinit();
    const load_start: std.Io.Timestamp = .now(io, .awake);
    const keep = if (resident_blocks == 0)
        n
    else
        policy.ditKeepBlocks(resident_blocks, @intCast(n));
    var i: usize = 0;
    while (i < keep) : (i += 1) {
        cores[i] = try loadDitCore(allocator, io, platform, &models.dit, &models.dit_store, shardings, i, progress, &loader);
        loaded_n += 1;
    }
    log.info("dit: resident {d}/{d} cores [{f}]", .{ keep, n, load_start.untilNow(io, .awake) });
    var embed = try models.enc.loadEmbed(allocator, io, platform, &models.enc_store, shardings, progress);
    errdefer encoder.EmbedTokens.unloadBuffers(&embed);
    var text_prep = try models.dit.loadTextPrep(allocator, io, platform, &models.dit_store, shardings, progress);
    errdefer dit.TextPrep.unloadBuffers(&text_prep, allocator);
    var patch = try models.dit.loadPatchEmbed(allocator, io, platform, &models.dit_store, shardings, progress);
    errdefer dit.PatchEmbed.unloadBuffers(&patch);
    var finish = try models.dit.loadFinishCore(allocator, io, platform, &models.dit_store, shardings, progress);
    errdefer dit.FinishCore.unloadBuffers(&finish);
    log.info("serve resident embed+text_prep+patch+finish", .{});
    return .{
        .enc = enc,
        .embed = embed,
        .text_prep = text_prep,
        .patch = patch,
        .finish = finish,
        .cores = cores,
    };
}

pub fn bakeDenoise(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    models: *repository.Bundle,
    compiled: *const pipeline.Compiled,
    shardings: []const zml.Sharding,
    progress: *std.Progress.Node,
    warm: *Warm,
    bake: *Bake,
    geo: pipeline.Geometry,
    layout: packing.Layout,
    schedules: scheduler.DualSchedule,
    reuse: ?*const Bake,
) !void {
    if (reuse) |src| {
        if (src.tables.len != 0) {
            bake.shareTables(src);
            try compileAudio(allocator, io, platform, models, shardings, progress, warm, bake, geo);
            log.info("serve reused AdaLN tables, compiled audio decode", .{});
            return;
        }
    }
    const bake_start: std.Io.Timestamp = .now(io, .awake);
    const n_blocks = models.dit.inner.blocks.len;
    const seq = layout.seqLen();
    const steps = schedules.video.stepCount();
    const flat_n = steps * packing.timestep_slot_count;
    const flat_t = try allocator.alloc(f32, flat_n);
    defer allocator.free(flat_t);
    const all_tidx = try allocator.alloc(u32, steps * seq);
    defer allocator.free(all_tidx);
    const row_ts = try allocator.alloc(f32, seq);
    defer allocator.free(row_ts);
    for (0..steps) |i| {
        _ = packing.writeRowPlan(
            layout,
            schedules.video.timesteps[i],
            schedules.audio.timesteps[i],
            row_ts,
            all_tidx[i * seq ..][0..seq],
            flat_t[i * packing.timestep_slot_count ..][0..packing.timestep_slot_count],
        );
    }
    var flat_buf = try weights.fromItems(io, platform, .init(.{ .n = flat_n }, .f32), flat_t);
    defer flat_buf.deinit();
    var time_bufs = try models.dit.loadTimeEmbedder(allocator, io, platform, &models.dit_store, shardings, progress);
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

    const tables = try allocator.alloc(zml.Buffer, n_blocks);
    var filled: usize = 0;
    errdefer {
        for (tables[0..filled]) |*t| t.deinit();
        allocator.free(tables);
    }
    var loader = try weights.initLoader(allocator, platform);
    defer loader.deinit();
    const AdaLnRunner = zml.FnExe(dit.prepareAdaln).Runner(.{.model});
    var adaln_runner: ?AdaLnRunner = null;
    defer if (adaln_runner) |*r| r.deinit(allocator);
    var prev_adaln: ?zml.Bufferized(dit.AdaLn) = null;
    defer if (prev_adaln) |*a| dit.AdaLn.unloadBuffers(a);
    var block_i: usize = 0;
    while (block_i < n_blocks) : (block_i += 1) {
        const adaln_bufs = try models.dit.loadAdaln(allocator, io, platform, &models.dit_store, shardings, block_i, progress, &loader);
        if (adaln_runner) |*r| {
            r.rebake(.{ .model = .{ .adaln = adaln_bufs } });
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
        filled += 1;
    }
    if (adaln_runner) |*r| {
        r.deinit(allocator);
        adaln_runner = null;
    }
    if (prev_adaln) |*a| {
        dit.AdaLn.unloadBuffers(a);
        prev_adaln = null;
    }
    var final_adaln = try models.dit.loadFinalAdaln(allocator, io, platform, &models.dit_store, shardings, progress);
    var final_runner = try AdaLnRunner.init(&compiled.prepare_final_adaln, allocator, .{
        .model = .{ .adaln = final_adaln },
    });
    defer final_runner.deinit(allocator);
    var final_table: zml.Buffer = undefined;
    final_runner.run(io, .{
        .inputs = .{ .temb = all_temb },
        .outputs = .{ .table = &final_table },
        .opts = .{ .wait = true },
    });
    dit.AdaLn.unloadBuffers(&final_adaln);

    bake.tables = tables;
    bake.final_table = final_table;
    try compileAudio(allocator, io, platform, models, shardings, progress, warm, bake, geo);
    log.info("serve baked AdaLN+audio [{f}]", .{bake_start.untilNow(io, .awake)});
}

fn compileAudio(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    models: *repository.Bundle,
    shardings: []const zml.Sharding,
    progress: *std.Progress.Node,
    warm: *Warm,
    bake: *Bake,
    geo: pipeline.Geometry,
) !void {
    bake.audio_exe = try pipeline.compileAudioDecode(allocator, io, platform, models.audio.inner, geo, shardings, progress);
    if (warm.audio_bufs == null) {
        warm.audio_bufs = try models.audio.loadBuffers(allocator, io, platform, &models.audio_store, shardings, progress);
    }
}

pub fn draft(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    models: *repository.Bundle,
    compiled: *const pipeline.Compiled,
    shardings: []const zml.Sharding,
    progress: *std.Progress.Node,
    req: DraftReq,
) !Draft {
    if (req.job) |j| j.set(.text, 1, 1);
    var text = try encodeText(
        allocator,
        io,
        platform,
        compiled,
        &models.enc,
        req.tokens,
        req.warm,
    );
    defer text.deinit();
    if (req.job) |j| j.set(.draft, 0, 1);
    var latents = try denoise(
        allocator,
        io,
        platform,
        compiled,
        &models.dit,
        models,
        shardings,
        req.geo,
        text,
        @intCast(req.tokens.len),
        req.layout,
        req.schedules,
        req.seed,
        req.warm,
        req.bake,
        req.resident_blocks,
        progress,
        req.job,
    );
    defer latents.deinit(allocator);

    const channels: u32 = @intCast(models.dit.cfg.in_channels);
    const thwc = try packing.unpatchify(
        allocator,
        latents.video,
        req.geo.latent_t,
        req.geo.latent_h,
        req.geo.latent_w,
        channels,
        models.dit.cfg.patch_size,
    );
    defer allocator.free(thwc);

    const audio_exe = if (req.bake.audio_exe) |*exe| exe else return error.AudioMissing;
    var audio_f = try io.concurrent(decode.decodeAudio, .{
        allocator,
        io,
        platform,
        audio_exe,
        &models.audio,
        &models.audio_store,
        shardings,
        req.geo,
        latents.audio,
        progress,
        if (req.warm.audio_bufs) |*b| b else null,
    });
    var audio_taken = false;
    errdefer if (!audio_taken) {
        if (audio_f.cancel(io)) |w| allocator.free(w) else |_| {}
    };

    if (req.job) |j| j.set(.taeh3, 1, 1);
    const vis_start: std.Io.Timestamp = .now(io, .awake);
    var rgb = try taeh3.runDevice(allocator, io, platform, req.taeh3, thwc);
    defer rgb.deinit();
    var pixels = try req.handoff.run(allocator, io, rgb);
    log.info("draft handoff [{f}]", .{vis_start.untilNow(io, .awake)});
    errdefer pixels.deinit();

    const wav = try audio_f.await(io);
    audio_taken = true;
    errdefer allocator.free(wav);
    return .{ .pixels = pixels, .wav = wav };
}
