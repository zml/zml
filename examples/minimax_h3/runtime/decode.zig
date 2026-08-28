const std = @import("std");

const zml = @import("zml");

const audio_vae = @import("../vae/audio.zig");
const config = @import("../core/config.zig");
const media = @import("media.zig");
const pipeline = @import("pipeline.zig");
const vae = @import("../vae/geometry.zig");
const visual_vae = @import("../vae/visual.zig");
const weights = @import("../core/weights.zig");
const policy = @import("../core/policy.zig");

const log = std.log.scoped(.minimax_h3_decode);

fn done(io: std.Io, start: std.Io.Timestamp, comptime msg: []const u8, args: anytype) void {
    log.info(msg ++ " [{f}]", args ++ .{start.untilNow(io, .awake)});
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
    const row_n = @as(usize, copy_w) * channels;
    var tt: u32 = 0;
    while (tt < copy_t) : (tt += 1) {
        var hh: u32 = 0;
        while (hh < copy_h) : (hh += 1) {
            const src_i = ((((t0 + tt) * src_h + (h0 + hh)) * src_w + w0) * channels);
            const dst_i = (((tt * tile.latent_h + hh) * tile.latent_w) * channels);
            @memcpy(dst[dst_i..][0..row_n], src[src_i..][0..row_n]);
        }
    }
}

pub const VisualCache = struct {
    embed: zml.Bufferized(visual_vae.EmbedModel),
    blocks: []zml.Bufferized(visual_vae.TransformerBlock),
    finish: zml.Bufferized(visual_vae.FinishModel),

    pub fn deinit(self: *VisualCache, allocator: std.mem.Allocator) void {
        visual_vae.EmbedModel.unloadBuffers(&self.embed);
        for (self.blocks) |*block| visual_vae.TransformerBlock.unloadBuffers(block);
        allocator.free(self.blocks);
        visual_vae.FinishModel.unloadBuffers(&self.finish);
    }
};

const load_window_cap: usize = policy.vae_load_window;

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

pub fn loadVisualCache(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    loaded: *const visual_vae.LoadedModel,
    store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    n_blocks: usize,
    progress: *std.Progress.Node,
) !VisualCache {
    const load_window = policy.vaeLoadWindow(@intCast(n_blocks));
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

    var loaders: [load_window_cap]zml.io.Loader = undefined;
    var ready: usize = 0;
    defer for (loaders[0..ready]) |*loader| loader.deinit();
    while (ready < load_window) : (ready += 1) {
        loaders[ready] = try weights.initLoader(allocator, platform);
    }

    var start: usize = 0;
    while (start < n_blocks) {
        const batch = @min(load_window, n_blocks - start);
        var futs: [load_window_cap]@TypeOf(try io.concurrent(loadVisualBlock, .{
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
        .pos = try weights.fromItems(io, platform, .init(.{ .s = seq, .ax = 3 }, .f32), positions),
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
    var latent_buf = try weights.fromItemsSharded(io, platform, latent_shape, latent_sharding, packed_latents);
    defer latent_buf.deinit();

    var hidden: zml.Buffer = undefined;
    var cos: zml.Buffer = undefined;
    var sin: zml.Buffer = undefined;
    var t: std.Io.Timestamp = .now(io, .awake);
    runners.embed.run(io, .{
        .inputs = .{ .latents = latent_buf, .position_ids = runners.pos },
        .outputs = .{ .hidden = &hidden, .cos = &cos, .sin = &sin },
    });
    defer cos.deinit();
    defer sin.deinit();
    var held: std.ArrayList(zml.Buffer) = .empty;
    defer {
        for (held.items) |*buf| buf.deinit();
        held.deinit(allocator);
    }
    try held.append(allocator, hidden);
    log.debug("visual embed: ran {f} [{f}]", .{ hidden.shape(), t.untilNow(io, .awake) });

    var i: usize = 0;
    while (i < cache.blocks.len) : (i += 1) {
        runners.block.rebake(.{ .layer = cache.blocks[i] });
        var next: zml.Buffer = undefined;
        t = .now(io, .awake);
        runners.block.run(io, .{
            .inputs = .{ .hidden = hidden, .cos = cos, .sin = sin },
            .outputs = .{ .hidden = &next },
        });
        hidden = next;
        try held.append(allocator, next);
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
    cache: ?*VisualCache,
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

    var owned_cache: ?VisualCache = null;
    defer if (owned_cache) |*c| c.deinit(allocator);
    const cache_ptr = cache orelse blk: {
        owned_cache = try loadVisualCache(
            allocator,
            io,
            platform,
            loaded,
            store,
            shardings,
            loaded.inner.blocks.len,
            progress,
        );
        break :blk &owned_cache.?;
    };
    var runners = try initVisualRunners(allocator, io, platform, compiled, loaded, cache_ptr);
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
        const jobs = try allocator.alloc(struct { yi: usize, xi: usize }, n_tiles);
        defer allocator.free(jobs);
        var job_i: usize = 0;
        for (y_plan.starts, y_plan.lengths, 0..) |y0, _, yi| {
            for (x_plan.starts, x_plan.lengths, 0..) |x0, _, xi| {
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
                jobs[job_i] = .{ .yi = yi, .xi = xi };
                job_i += 1;
            }
        }

        const clip_t = tile.latent_t * spec.temporal;
        const clip = try allocator.alloc(f32, 3 * clip_t * geo.pixel_h * geo.pixel_w);
        defer allocator.free(clip);
        @memset(clip, 0);
        const tile_px_h = tile.latent_h * spec.spatial;
        const tile_px_w = tile.latent_w * spec.spatial;
        var stitcher = try vae.NchwStitcher.init(
            allocator,
            clip,
            3,
            clip_t,
            geo.pixel_h,
            geo.pixel_w,
            tile_px_h,
            tile_px_w,
            @intCast(y_plan.count()),
            @intCast(x_plan.count()),
            y_plan.overlaps,
            x_plan.overlaps,
        );
        defer stitcher.deinit(allocator);

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
            const patches = try runVisualBatch(allocator, io, platform, compiled, loaded, cache_ptr, &runners, packed_lat, shardings);
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
                stitcher.push(@intCast(job.yi), @intCast(job.xi), pix);
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

    var latent_buf = try weights.fromItems(io, platform, .init(.{
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
    const play_s = @as(f32, @floatFromInt(geo.frames)) / config.video_fps;
    if (muxed) {
        log.info("wrote {s}/{s}  {d}x{d}  {d:.1}s", .{
            out_path,
            mp4_name,
            geo.pixel_w,
            geo.pixel_h,
            play_s,
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
