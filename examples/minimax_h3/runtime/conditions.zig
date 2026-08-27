const std = @import("std");

const zml = @import("zml");

const audio_vae = @import("../vae/audio.zig");
const repository = @import("repository.zig");
const config_mod = @import("../core/config.zig");
const encode_mod = @import("encode.zig");
const geom = @import("../conditioning/geometry.zig");
const media = @import("media.zig");
const packing = @import("../model/packing.zig");
const pipeline = @import("pipeline.zig");
const presentation = @import("../conditioning/presentation.zig");
const request_mod = @import("../core/request.zig");
const session_mod = @import("session.zig");
const sharding_mod = @import("../core/sharding.zig");
const vae = @import("../vae/geometry.zig");
const vision = @import("../model/vision.zig");
const visual_enc = @import("../vae/visual_encoder.zig");

const log = std.log.scoped(.minimax_h3_conditions);

pub const Prepared = struct {
    tokens: []u32,
    tags: []u8,
    positions: ?[]f32 = null,
    deepstack: [3]?[]f32 = .{ null, null, null },
    vision_merged: ?[]f32 = null,
    vision_spans: []presentation.VisionSpan = &.{},
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

pub const Prepare = struct {
    variant: config_mod.Variant,
    first_image: []const u8,
    last_image: []const u8,
    refs: []const request_mod.Reference,
    prompt: []const u8,
    geo: pipeline.Geometry,
    models: *repository.Bundle,
    shardings: sharding_mod.Shardings,
};

pub fn prepare(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    progress: *std.Progress.Node,
    encode_text: anytype,
    req: Prepare,
) !Prepared {
    const enc_dir = req.models.enc_src.dir;
    const visual_store = &req.models.visual_store;
    const audio_store = &req.models.audio_store;
    const enc_store = &req.models.enc_store;
    const loaded_visual = &req.models.visual;
    const loaded_audio = &req.models.audio;
    const patch = req.models.dit.cfg.patch_size;
    const text_hidden = req.models.enc.cfg.hidden_size;
    log.info(
        "conditions: {s} first={s} last={s} refs={d}",
        .{
            @tagName(req.variant),
            if (req.first_image.len == 0) "-" else req.first_image,
            if (req.last_image.len == 0) "-" else req.last_image,
            req.refs.len,
        },
    );
    try request_mod.validateRefs(req.refs);
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

    if (req.variant == .fl2va) {
        if (req.first_image.len != 0) try visuals.append(allocator, .{ .kind = .image, .path = req.first_image, .keyframe_index = 0 });
        if (req.last_image.len != 0) try visuals.append(allocator, .{ .kind = .image, .path = req.last_image, .keyframe_index = 1 });
    } else {
        for (req.refs) |ref| {
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
            const keep = @min(req.geo.frames, @as(u32, @intCast(indices.len)));
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
        } else if (req.variant == .fl2va) {
            item.rgb = if (keyframe_i == 0)
                try media.loadRgb(allocator, io, item.path, req.geo.pixel_w, req.geo.pixel_h)
            else
                try media.loadRgbCover(allocator, io, item.path, req.geo.pixel_w, req.geo.pixel_h);
            keyframe_i += 1;
            item.w = req.geo.pixel_w;
            item.h = req.geo.pixel_h;
            item.nchw = try media.rgbToNchwImagenet(allocator, item.rgb, item.h, item.w);
            item.latent_t = 1;
        } else {
            const raw = try media.loadRgbRaw(allocator, io, item.path);
            defer allocator.free(raw.rgb);
            const dest = try geom.refImageSize(raw.w, raw.h, req.geo.pixel_w, req.geo.pixel_h);
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
        const duration_s = @as(f32, @floatFromInt(req.geo.frames)) / config_mod.video_fps;
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
    var assembled = try presentation.assemble(allocator, encode_text, req.variant, specs.items, req.prompt);
    errdefer assembled.deinit(allocator);

    if (hasVisual(visuals.items)) {
        if (!visual_enc.ready(visual_store.view())) return error.VisualEncodeMissing;
        if (!vision.ready(enc_store.view())) return error.VisionWeightsMissing;
    }
    if (audios.items.len != 0) {
        if (!audio_vae.encodeReady(audio_store.view())) return error.AudioEncodeMissing;
    }

    var all = req.shardings.all();
    const positions = try allocator.alloc(f32, assembled.tokens.len * 3);
    errdefer allocator.free(positions);
    presentation.fillEncoderPositions(positions, @intCast(assembled.tokens.len), assembled.spans);

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
        var compiled_v: ?vision.Compiled = null;
        defer if (compiled_v) |*c| c.deinit();
        var span_i: usize = 0;
        for (visuals.items) |item| {
            if (item.kind == .audio) continue;
            if (compiled_v == null or compiled_v.?.seq != item.seq) {
                if (compiled_v) |*c| {
                    c.deinit();
                    compiled_v = null;
                }
                compiled_v = try pipeline.compileVision(allocator, io, platform, loaded_vision.inner, item.seq, req.shardings, progress);
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
            while (bi < n_blocks and span_i < assembled.spans.len) : (bi += 1) {
                const span = assembled.spans[span_i];
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
            req.shardings,
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
    for (encoded_visuals[0..n_vis]) |v| v.deinit(allocator);
    allocator.free(encoded_visuals);
    for (encoded_audios[0..n_aud_enc]) |a| a.deinit(allocator);
    allocator.free(encoded_audios);

    const merged_out: ?[]f32 = if (merged_all.items.len == 0) null else try merged_all.toOwnedSlice(allocator);
    errdefer if (merged_out) |m| allocator.free(m);
    log.info(
        "conditions: ok tokens={d} vision_spans={d} video_conds={d} audio_conds={d} refs={d}",
        .{ assembled.tokens.len, assembled.spans.len, conds.videos.len, conds.audios.len, conds.references.len },
    );
    return .{
        .tokens = assembled.tokens,
        .tags = assembled.tags,
        .positions = positions,
        .deepstack = .{ ds_host[0], ds_host[1], ds_host[2] },
        .vision_merged = merged_out,
        .vision_spans = assembled.spans,
        .conds = conds,
    };
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
