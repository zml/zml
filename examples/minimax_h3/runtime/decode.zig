const std = @import("std");

const zml = @import("zml");

const audio_vae = @import("../vae/audio.zig");
const media = @import("media.zig");
const pipeline = @import("pipeline.zig");
const vae = @import("../vae/geom.zig");
const visual_vae = @import("../vae/visual.zig");
const weights = @import("../core/weights.zig");

const log = std.log.scoped(.minimax_h3_decode);

fn done(io: std.Io, start: std.Io.Timestamp, comptime msg: []const u8, args: anytype) void {
    log.info(msg ++ " [{f}]", args ++ .{start.untilNow(io, .awake)});
}

fn bufferFromItems(io: std.Io, platform: *const zml.Platform, shape: zml.Shape, items: anytype) !zml.Buffer {
    return zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(items));
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

const load_window: usize = 4;

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

fn runVisualTile(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled: *const pipeline.VaeCompiled,
    loaded: *const visual_vae.LoadedModel,
    cache: *const VisualCache,
    tile_latents: []const f32,
) ![]f32 {
    const tile = compiled.tile;
    const registers: u32 = @intCast(loaded.cfg.decoder_num_register_tokens);
    const seq = tile.seq(registers);
    const positions = try visual_vae.hostPositions(allocator, tile.latent_t, tile.latent_h, tile.latent_w, registers);
    defer allocator.free(positions);

    var latent_buf = try bufferFromItems(io, platform, .init(.{
        .b = 1,
        .s = tile.tokens(),
        .d = loaded.cfg.latent_channels,
    }, .f32), tile_latents);
    defer latent_buf.deinit();
    var pos_buf = try bufferFromItems(io, platform, .init(.{ .s = seq, .ax = 3 }, .f32), positions);
    defer pos_buf.deinit();

    var embed_runner = try zml.FnExe(visual_vae.embed).Runner(.{.model}).init(&compiled.embed, allocator, .{ .model = cache.embed });
    defer embed_runner.deinit(allocator);

    var hidden: zml.Buffer = undefined;
    var cos: zml.Buffer = undefined;
    var sin: zml.Buffer = undefined;
    var t: std.Io.Timestamp = .now(io, .awake);
    log.debug("visual embed: run", .{});
    embed_runner.run(io, .{
        .inputs = .{ .latents = latent_buf, .position_ids = pos_buf },
        .outputs = .{ .hidden = &hidden, .cos = &cos, .sin = &sin },
        .opts = .{ .wait = true },
    });
    defer hidden.deinit();
    defer cos.deinit();
    defer sin.deinit();
    log.debug("visual embed: ran {f} [{f}]", .{ hidden.shape(), t.untilNow(io, .awake) });

    const n_blocks = cache.blocks.len;
    const BlockRunner = zml.FnExe(visual_vae.TransformerBlock.forward).Runner(.{.layer});
    var block_runner: ?BlockRunner = null;
    defer if (block_runner) |*r| r.deinit(allocator);
    var i: usize = 0;
    while (i < n_blocks) : (i += 1) {
        if (block_runner) |*r| {
            weights.rebake(r, .{ .layer = cache.blocks[i] });
        } else {
            block_runner = try BlockRunner.init(&compiled.block, allocator, .{ .layer = cache.blocks[i] });
        }
        var next: zml.Buffer = undefined;
        t = .now(io, .awake);
        log.debug("visual block {d}/{d}: run", .{ i + 1, n_blocks });
        block_runner.?.run(io, .{
            .inputs = .{ .hidden = hidden, .cos = cos, .sin = sin },
            .outputs = .{ .hidden = &next },
            .opts = .{ .wait = true },
        });
        hidden.deinit();
        hidden = next;
        log.debug("visual block {d}/{d}: ran [{f}]", .{ i + 1, n_blocks, t.untilNow(io, .awake) });
    }

    var finish_runner = try zml.FnExe(visual_vae.finish).Runner(.{.model}).init(&compiled.finish, allocator, .{ .model = cache.finish });
    defer finish_runner.deinit(allocator);

    var patches: zml.Buffer = undefined;
    t = .now(io, .awake);
    log.debug("visual finish: run", .{});
    finish_runner.run(io, .{
        .inputs = .{ .hidden = hidden },
        .outputs = .{ .patches = &patches },
        .opts = .{ .wait = true },
    });
    defer patches.deinit();
    log.debug("visual finish: ran {f} [{f}]", .{ patches.shape(), t.untilNow(io, .awake) });

    const patch_dim: usize = @intCast(loaded.cfg.out_channels * 4 * 16 * 16);
    const host = try allocator.alloc(f32, tile.tokens() * patch_dim);
    errdefer allocator.free(host);
    t = .now(io, .awake);
    log.debug("visual finish: toSlice tokens={d} patch_dim={d}", .{ tile.tokens(), patch_dim });
    try patches.toSlice(io, .init(zml.Shape.init(.{ .b = 1, .s = tile.tokens(), .d = @as(i64, @intCast(patch_dim)) }, .f32), std.mem.sliceAsBytes(host)));
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
        const tile_lat = try allocator.alloc(f32, tile.tokens() * channels);
        defer allocator.free(tile_lat);

        const clip_t = tile.latent_t * spec.temporal;
        const clip = try allocator.alloc(f32, 3 * clip_t * geo.pixel_h * geo.pixel_w);
        defer allocator.free(clip);
        @memset(clip, 0);

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
                    tile_lat,
                );
                const patches = try runVisualTile(allocator, io, platform, compiled, loaded, &cache, tile_lat);
                defer allocator.free(patches);
                const pix = try visual_vae.unpackPatches(allocator, patches, tile.latent_t, tile.latent_h, tile.latent_w, spec.temporal, spec.spatial, 3);
                defer allocator.free(pix);
                const blend_y: u32 = if (yi == 0) 0 else y_plan.overlaps[yi - 1];
                const blend_x: u32 = if (xi == 0) 0 else x_plan.overlaps[xi - 1];
                pasteNchw(clip, clip_t, geo.pixel_h, geo.pixel_w, pix, clip_t, ylen, xlen, y0, x0, blend_y, blend_x);
            }
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
    compiled: *const pipeline.VaeCompiled,
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
    var runner = try zml.FnExe(audio_vae.decode).Runner(.{.model}).init(&compiled.audio, allocator, .{ .model = bufs });
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
