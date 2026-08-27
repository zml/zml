const std = @import("std");

const zml = @import("zml");

const audio_vae = @import("../vae/audio.zig");
const buffers = @import("../core/buffers.zig");
const config_mod = @import("../core/config.zig");
const packing = @import("../model/packing.zig");
const pipeline = @import("pipeline.zig");
const vae = @import("../vae/geometry.zig");
const visual_enc = @import("../vae/visual_encoder.zig");

const log = std.log.scoped(.minimax_h3_encode);

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

    const n_y: u32 = @intCast(y_plan.count());
    const n_x: u32 = @intCast(x_plan.count());
    var stitcher: ?vae.NchwStitcher = null;
    defer if (stitcher) |*s| s.deinit(allocator);
    if (n_y > 1 or n_x > 1) {
        const y_ov = try allocator.alloc(u32, y_plan.overlaps.len);
        defer allocator.free(y_ov);
        const x_ov = try allocator.alloc(u32, x_plan.overlaps.len);
        defer allocator.free(x_ov);
        for (y_plan.overlaps, y_ov) |px, *lat| lat.* = px / spec.spatial;
        for (x_plan.overlaps, x_ov) |px, *lat| lat.* = px / spec.spatial;
        stitcher = try vae.NchwStitcher.init(
            allocator,
            canvas,
            moments_c,
            out_t,
            latent_h,
            latent_w,
            tile_lat_h,
            tile_lat_w,
            n_y,
            n_x,
            y_ov,
            x_ov,
        );
    }

    for (y_plan.starts, y_plan.lengths, 0..) |y0, ylen, yi| {
        for (x_plan.starts, x_plan.lengths, 0..) |x0, xlen, xi| {
            copyNchwTile(pixels_nchw, 3, frames, height, width, y0, x0, tile_h, tile_w, tile_px);
            var pix = try buffers.fromItems(io, platform, .init(.{
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

            if (stitcher) |*s| {
                s.push(@intCast(yi), @intCast(xi), tile_mom);
            } else {
                vae.copyNchwCrop(
                    canvas,
                    latent_h,
                    latent_w,
                    0,
                    0,
                    tile_mom,
                    tile_lat_h,
                    tile_lat_w,
                    ylen / spec.spatial,
                    xlen / spec.spatial,
                    moments_c,
                    out_t,
                );
            }
        }
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
    var wav = try buffers.fromItems(io, platform, .init(.{ .b = 2, .c = 1, .t = samples }, .f32), batch);
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
