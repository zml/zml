const std = @import("std");

const zml = @import("zml");

const audio_vae = @import("../vae/audio.zig");
const repository = @import("repository.zig");
const config = @import("../core/config.zig");
const encode_mod = @import("encode.zig");
const geom = @import("../conditioning/geometry.zig");
const media = @import("media.zig");
const packing = @import("../model/packing.zig");
const pipeline = @import("pipeline.zig");
const presentation = @import("../conditioning/presentation.zig");
const request_mod = @import("../core/request.zig");
const session_mod = @import("session.zig");
const sharding = @import("../core/sharding.zig");
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

pub const ReferenceIndexDomains = struct {
    encoded_visuals: i32 = 0,
    encoded_audios: i32 = 0,

    pub fn next(self: *ReferenceIndexDomains, kind: packing.ReferenceKind, has_audio: bool) packing.ReferenceBlock {
        return switch (kind) {
            .image => blk: {
                const index = self.encoded_visuals;
                self.encoded_visuals += 1;
                break :blk .{ .kind = .image, .video_index = index };
            },
            .video, .video_audio => blk: {
                const video_index = self.encoded_visuals;
                self.encoded_visuals += 1;
                var audio_index: i32 = -1;
                if (has_audio) {
                    audio_index = self.encoded_audios;
                    self.encoded_audios += 1;
                }
                break :blk .{
                    .kind = if (has_audio) .video_audio else .video,
                    .video_index = video_index,
                    .audio_index = audio_index,
                };
            },
            .audio => blk: {
                const index = self.encoded_audios;
                self.encoded_audios += 1;
                break :blk .{ .kind = .audio, .audio_index = index };
            },
        };
    }
};

pub const CollectedVisual = struct {
    kind: packing.ReferenceKind,
    path: []const u8,
    has_audio: bool = false,
};

pub const CollectedAudio = struct {
    path: []const u8,
};

pub const CollectedRefs = struct {
    visuals: []CollectedVisual,
    audios: []CollectedAudio,
    blocks: []packing.ReferenceBlock,
    encoded_visuals: i32 = 0,
    encoded_audios: i32 = 0,

    pub fn deinit(self: CollectedRefs, allocator: std.mem.Allocator) void {
        allocator.free(self.visuals);
        allocator.free(self.audios);
        allocator.free(self.blocks);
    }
};

/// Probe videos, then enforce the resolved audio-reference limit.
/// `probe` must be `fn (Allocator, Io, []const u8) !type` with a `has_audio` field.
pub fn collectRefs(
    allocator: std.mem.Allocator,
    io: std.Io,
    refs: []const request_mod.Reference,
    probe: anytype,
) !CollectedRefs {
    var visuals: std.ArrayList(CollectedVisual) = .empty;
    errdefer visuals.deinit(allocator);
    var audios: std.ArrayList(CollectedAudio) = .empty;
    errdefer audios.deinit(allocator);
    var blocks: std.ArrayList(packing.ReferenceBlock) = .empty;
    errdefer blocks.deinit(allocator);
    var reference_indices: ReferenceIndexDomains = .{};

    for (refs) |ref| {
        switch (ref.kind) {
            .image => {
                const block = reference_indices.next(.image, false);
                try visuals.append(allocator, .{ .kind = .image, .path = ref.path });
                try blocks.append(allocator, block);
            },
            .video, .video_audio => {
                const meta = try probe(allocator, io, ref.path);
                const has_audio = meta.has_audio;
                const block = reference_indices.next(.video, has_audio);
                try visuals.append(allocator, .{
                    .kind = if (has_audio) .video_audio else .video,
                    .path = ref.path,
                    .has_audio = has_audio,
                });
                if (has_audio) {
                    try audios.append(allocator, .{ .path = ref.path });
                }
                try blocks.append(allocator, block);
            },
            .audio => {
                const block = reference_indices.next(.audio, true);
                try audios.append(allocator, .{ .path = ref.path });
                try visuals.append(allocator, .{ .kind = .audio, .path = ref.path, .has_audio = true });
                try blocks.append(allocator, block);
            },
        }
    }
    try request_mod.validateResolvedAudioCount(audios.items.len);
    const visuals_s = try visuals.toOwnedSlice(allocator);
    errdefer allocator.free(visuals_s);
    const audios_s = try audios.toOwnedSlice(allocator);
    errdefer allocator.free(audios_s);
    const blocks_s = try blocks.toOwnedSlice(allocator);
    return .{
        .visuals = visuals_s,
        .audios = audios_s,
        .blocks = blocks_s,
        .encoded_visuals = reference_indices.encoded_visuals,
        .encoded_audios = reference_indices.encoded_audios,
    };
}

pub const VisualEncodeKey = struct {
    tile_h: u32,
    tile_w: u32,
    need_clip: bool,
};

pub fn visualEncodeKey(kind: packing.ReferenceKind, pixel_h: u32, pixel_w: u32) VisualEncodeKey {
    const tile_px = vae.official_visual.tile_px;
    return .{
        .tile_h = @min(tile_px, pixel_h),
        .tile_w = @min(tile_px, pixel_w),
        .need_clip = kind == .video or kind == .video_audio,
    };
}

pub const Prepare = struct {
    variant: config.Variant,
    first_frame: []const u8,
    last_frame: []const u8,
    refs: []const request_mod.Reference,
    prompt: []const u8,
    geo: pipeline.Geometry,
    models: *repository.Bundle,
    shardings: sharding.Shardings,
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
            if (req.first_frame.len == 0) "-" else req.first_frame,
            if (req.last_frame.len == 0) "-" else req.last_frame,
            req.refs.len,
        },
    );
    try request_mod.validateRefs(req.refs);
    const VisualItem = struct {
        kind: packing.ReferenceKind,
        path: []const u8,
        keyframe_index: i32 = 0,
        rgb: []u8 = &.{},
        qwen_rgb: []u8 = &.{},
        frames: u32 = 1,
        vae_frames: u32 = 1,
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
        timestamps: []f32 = &.{},
        has_audio: bool = false,
    };
    const AudioItem = struct {
        path: []const u8,
        stereo: []f32 = &.{},
        latent_t: u32 = 0,
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
    var reference_indices: ReferenceIndexDomains = .{};

    if (req.variant == .fl2va) {
        if (req.first_frame.len != 0) try visuals.append(allocator, .{ .kind = .image, .path = req.first_frame, .keyframe_index = 0 });
        if (req.last_frame.len != 0) try visuals.append(allocator, .{ .kind = .image, .path = req.last_frame, .keyframe_index = 1 });
    } else {
        const collected = try collectRefs(allocator, io, req.refs, media.probeVideo);
        defer collected.deinit(allocator);
        for (collected.visuals) |item| {
            try visuals.append(allocator, .{
                .kind = item.kind,
                .path = item.path,
                .has_audio = item.has_audio,
            });
        }
        for (collected.audios) |item| {
            try audios.append(allocator, .{ .path = item.path });
        }
        try blocks.appendSlice(allocator, collected.blocks);
        reference_indices = .{
            .encoded_visuals = collected.encoded_visuals,
            .encoded_audios = collected.encoded_audios,
        };
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
            const fps = if (clip.fps > 0) clip.fps else config.video_fps;
            const indices = try geom.resampleFrameIndices(clip.frames, fps, config.video_fps, allocator);
            defer allocator.free(indices);
            const keep = @min(req.geo.frames, @as(u32, @intCast(indices.len)));
            const own = try geom.videoCanvas(clip.w, clip.h);
            item.w = own.w;
            item.h = own.h;
            item.frames = keep;
            item.vae_frames = vae.referenceVideoFrameCount(vae.official_visual, keep);
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
            item.nchw = try media.rgbVideoToNchwImagenet(allocator, item.rgb, item.vae_frames, item.h, item.w);
            item.latent_t = vae.encodeVideoLatentT(vae.official_visual, item.vae_frames);
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
            const sampled = try geom.sampleVideoConditionFrames(item.frames, config.video_fps, config.qwen_video_fps, 2);
            item.temporal = sampled.block_count;
            item.seq = spec.seq * item.temporal;
            item.merged = spec.merged;
            item.timestamps = try allocator.alloc(f32, sampled.block_count);
            const idx_buf = try allocator.alloc(u32, sampled.indices_len);
            defer allocator.free(idx_buf);
            const nidx = geom.fillVideoConditionIndices(item.frames, config.video_fps, config.qwen_video_fps, idx_buf);
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
    for (audios.items) |*item| {
        const duration_s = @as(f32, @floatFromInt(req.geo.frames)) / config.video_fps;
        item.stereo = try media.loadAudioOfficial(allocator, io, item.path, duration_s, rate);
        const samples: u32 = @intCast(item.stereo.len / 2);
        const aligned = geom.hopAlign(samples, hop);
        item.latent_t = aligned / hop;
    }

    var specs: std.ArrayList(presentation.VisionClip) = .empty;
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

        const VisualExecutable = struct {
            tile_h: u32,
            tile_w: u32,
            need_clip: bool,
            exe: pipeline.EncodeCompiled,
        };
        var visual_executables: std.ArrayList(VisualExecutable) = .empty;
        defer {
            for (visual_executables.items) |*entry| entry.exe.deinit();
            visual_executables.deinit(allocator);
        }

        const AudioExecutable = struct {
            samples: u32,
            exe: zml.FnExe(audio_vae.encode),
        };
        var audio_executables: std.ArrayList(AudioExecutable) = .empty;
        defer {
            for (audio_executables.items) |*entry| entry.exe.deinit();
            audio_executables.deinit(allocator);
        }

        for (visuals.items) |item| {
            if (item.kind == .audio) continue;
            const need_clip = item.kind == .video or item.kind == .video_audio;
            const tile = encodeTileSize(item, vae.official_visual.tile_px);
            var executable_index: ?usize = null;
            for (visual_executables.items, 0..) |entry, index| {
                if (entry.tile_h == tile.h and entry.tile_w == tile.w and entry.need_clip == need_clip) {
                    executable_index = index;
                    break;
                }
            }
            if (executable_index == null) {
                const exe = try pipeline.compileEncode(
                    allocator,
                    io,
                    platform,
                    v_loaded.?.inner,
                    tile.h,
                    tile.w,
                    need_clip,
                    req.shardings,
                    progress,
                );
                try visual_executables.append(allocator, .{
                    .tile_h = tile.h,
                    .tile_w = tile.w,
                    .need_clip = need_clip,
                    .exe = exe,
                });
                executable_index = visual_executables.items.len - 1;
            }
            const compiled_e = &visual_executables.items[executable_index.?].exe;
            encoded_visuals[n_vis] = if (need_clip)
                try encode_mod.encodeVideo(allocator, io, platform, compiled_e, &v_loaded.?, &v_bufs.?, item.nchw.?, item.vae_frames, item.h, item.w)
            else
                try encode_mod.encodeKeyframe(allocator, io, platform, compiled_e, &v_loaded.?, &v_bufs.?, item.nchw.?, item.h, item.w);
            encoded_visuals[n_vis].keyframe_index = item.keyframe_index;
            n_vis += 1;
        }
        for (audios.items, encoded_audios) |item, *out| {
            const samples = item.latent_t * hop;
            var executable_index: ?usize = null;
            for (audio_executables.items, 0..) |entry, index| {
                if (entry.samples == samples) {
                    executable_index = index;
                    break;
                }
            }
            if (executable_index == null) {
                const exe = try pipeline.compileAudioEncode(
                    allocator,
                    io,
                    platform,
                    a_loaded.?.inner,
                    samples,
                    req.shardings,
                    progress,
                );
                try audio_executables.append(allocator, .{ .samples = samples, .exe = exe });
                executable_index = audio_executables.items.len - 1;
            }
            out.* = try encode_mod.encodeAudio(
                allocator,
                io,
                platform,
                &audio_executables.items[executable_index.?].exe,
                &a_loaded.?,
                &a_bufs.?,
                item.stereo,
            );
            std.debug.assert(out.latent_t == item.latent_t);
            n_aud_enc += 1;
        }
    }

    if (blocks.items.len != 0) {
        if (n_vis != @as(usize, @intCast(reference_indices.encoded_visuals)))
            return error.ReferenceVisualIndexMismatch;
        if (n_aud_enc != @as(usize, @intCast(reference_indices.encoded_audios)))
            return error.ReferenceAudioIndexMismatch;
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

fn encodeTileSize(item: anytype, tile_px: u32) struct { h: u32, w: u32 } {
    _ = tile_px;
    std.debug.assert(item.kind != .audio);
    const key = visualEncodeKey(item.kind, item.h, item.w);
    return .{ .h = key.tile_h, .w = key.tile_w };
}
