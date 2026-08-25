const std = @import("std");

const zml = @import("zml");

const audio_vae = @import("audio_vae.zig");
const config_mod = @import("config.zig");
const encode_mod = @import("encode.zig");
const media = @import("media.zig");
const packing = @import("packing.zig");
const pipeline = @import("pipeline.zig");
const session_mod = @import("session.zig");
const sharding_mod = @import("sharding.zig");
const vae = @import("vae.zig");
const vision = @import("vision.zig");
const visual_enc = @import("visual_enc.zig");
const visual_vae = @import("visual_vae.zig");

const log = std.log.scoped(.minimax_h3_conditions);

pub const Prepared = struct {
    tokens: []u32,
    tags: []u8,
    positions: ?[]f32 = null,
    deepstack: [3]?[]f32 = .{ null, null, null },
    vision_merged: ?[]f32 = null,
    vision_spans: []session_mod.VisionSpan = &.{},
    conds: encode_mod.ConditionSet,
};

pub const max_ref_files: u32 = 12;
pub const max_ref_images: u32 = 9;
pub const max_ref_videos: u32 = 3;
pub const max_ref_audios: u32 = 3;

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

fn snap16(n: u32) u32 {
    return @max(16, (n / 16) * 16);
}

fn hopAlign(n: u32, hop: u32) u32 {
    return n + (hop - (n % hop)) % hop;
}

fn hasVideo(items: anytype) bool {
    for (items) |item| if (item.kind == .video) return true;
    return false;
}

fn padStereo(allocator: std.mem.Allocator, stereo: []const f32, samples: u32) ![]f32 {
    const out = try allocator.alloc(f32, @as(usize, samples) * 2);
    @memset(out, 0);
    const n = @min(stereo.len, out.len);
    @memcpy(out[0..n], stereo[0..n]);
    return out;
}

fn sampleRgbFrames(allocator: std.mem.Allocator, rgb: []const u8, src_frames: u32, w: u32, h: u32, dst_frames: u32) ![]u8 {
    if (src_frames == 0 or dst_frames == 0) return error.EmptyVideo;
    const plane = @as(usize, w) * h * 3;
    const out = try allocator.alloc(u8, dst_frames * plane);
    var i: u32 = 0;
    while (i < dst_frames) : (i += 1) {
        const src = @min(src_frames - 1, i * src_frames / dst_frames);
        @memcpy(out[i * plane ..][0..plane], rgb[src * plane ..][0..plane]);
    }
    return out;
}

pub fn prepare(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    progress: *std.Progress.Node,
    variant: config_mod.Variant,
    first_image: []const u8,
    last_image: []const u8,
    ref_paths: []const []const u8,
    geo: pipeline.Geometry,
    patch: [3]i64,
    enc_dir: std.Io.Dir,
    visual_store: ?*zml.io.TensorStore,
    audio_store: ?*zml.io.TensorStore,
    enc_store: *zml.io.TensorStore,
    loaded_visual: ?*visual_vae.LoadedModel,
    loaded_audio: ?*audio_vae.LoadedModel,
    shardings: sharding_mod.Shardings,
    colon_tokens: []const u32,
    prompt_tokens: []const u32,
    text_hidden: i64,
    compile_only: bool,
) !Prepared {
    log.info(
        "conditions: {s} first={s} last={s} refs={d} compile_only={}",
        .{
            @tagName(variant),
            if (first_image.len == 0) "-" else first_image,
            if (last_image.len == 0) "-" else last_image,
            ref_paths.len,
            compile_only,
        },
    );
    const VisualItem = struct {
        kind: packing.ReferenceKind,
        path: []const u8,
        keyframe_index: i32 = 0,
        rgb: []u8 = &.{},
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
            if (item.nchw) |n| allocator.free(n);
        }
        visuals.deinit(allocator);
    }
    var audios: std.ArrayList(AudioItem) = .empty;
    defer {
        for (audios.items) |item| if (item.stereo.len != 0) allocator.free(item.stereo);
        audios.deinit(allocator);
    }
    var refs: std.ArrayList(packing.ReferenceBlock) = .empty;
    defer refs.deinit(allocator);

    if (variant == .fl2va) {
        if (first_image.len != 0) try visuals.append(allocator, .{ .kind = .image, .path = first_image, .keyframe_index = 0 });
        if (last_image.len != 0) try visuals.append(allocator, .{ .kind = .image, .path = last_image, .keyframe_index = 1 });
    } else {
        if (ref_paths.len > max_ref_files) return error.TooManyRefs;
        var n_img: u32 = 0;
        var n_vid: u32 = 0;
        var n_aud: u32 = 0;
        var i: usize = 0;
        while (i < ref_paths.len) : (i += 1) {
            const kind = media.guessKind(ref_paths[i]);
            switch (kind) {
                .image => {
                    n_img += 1;
                    if (n_img > max_ref_images) return error.TooManyRefImages;
                    const vidx: i32 = @intCast(visuals.items.len);
                    try visuals.append(allocator, .{ .kind = .image, .path = ref_paths[i], .video_index = vidx });
                    try refs.append(allocator, .{ .kind = .image, .video_index = vidx });
                },
                .video => {
                    n_vid += 1;
                    if (n_vid > max_ref_videos) return error.TooManyRefVideos;
                    const vidx: i32 = @intCast(visuals.items.len);
                    try visuals.append(allocator, .{ .kind = .video, .path = ref_paths[i], .video_index = vidx });
                    var aidx: i32 = -1;
                    var block_kind: packing.ReferenceKind = .video;
                    if (i + 1 < ref_paths.len and media.guessKind(ref_paths[i + 1]) == .audio) {
                        n_aud += 1;
                        if (n_aud > max_ref_audios) return error.TooManyRefAudios;
                        aidx = @intCast(audios.items.len);
                        try audios.append(allocator, .{ .path = ref_paths[i + 1], .audio_index = aidx });
                        block_kind = .video_audio;
                        i += 1;
                    }
                    try refs.append(allocator, .{ .kind = block_kind, .video_index = vidx, .audio_index = aidx });
                },
                .audio, .video_audio => {
                    n_aud += 1;
                    if (n_aud > max_ref_audios) return error.TooManyRefAudios;
                    const aidx: i32 = @intCast(audios.items.len);
                    try audios.append(allocator, .{ .path = ref_paths[i], .audio_index = aidx });
                    try refs.append(allocator, .{ .kind = .audio, .audio_index = aidx });
                },
            }
        }
    }

    const vcfg = vision.configFromRepo(allocator, io, enc_dir, text_hidden);
    const hidden_dim: u32 = @intCast(text_hidden);
    const spatial = vae.official_visual.spatial;

    for (visuals.items) |*item| {
        if (compile_only) {
            if (item.kind == .video or variant == .fl2va) {
                item.w = geo.pixel_w;
                item.h = geo.pixel_h;
                item.frames = if (item.kind == .video) geo.frames else 1;
            } else {
                const size = try media.imageSize(allocator, io, item.path);
                item.w = snap16(size.w);
                item.h = snap16(size.h);
                item.frames = 1;
            }
            item.latent_t = if (item.kind == .video) vae.encodeVideoLatentT(vae.official_visual, item.frames) else 1;
        } else if (item.kind == .video) {
            const vid = try media.loadVideoRgb(allocator, io, item.path, geo.pixel_w, geo.pixel_h, geo.frames);
            item.rgb = vid.rgb;
            item.frames = vid.frames;
            item.w = vid.w;
            item.h = vid.h;
            item.nchw = try media.rgbVideoToNchwImagenet(allocator, vid.rgb, vid.frames, vid.h, vid.w);
            item.latent_t = vae.encodeVideoLatentT(vae.official_visual, vid.frames);
        } else {
            const canvas = variant == .fl2va;
            if (canvas) {
                const rgb = try media.loadRgb(allocator, io, item.path, geo.pixel_w, geo.pixel_h);
                item.rgb = rgb;
                item.w = geo.pixel_w;
                item.h = geo.pixel_h;
            } else {
                const raw = try media.loadRgbRaw(allocator, io, item.path);
                const dw = snap16(raw.w);
                const dh = snap16(raw.h);
                item.rgb = if (dw == raw.w and dh == raw.h) raw.rgb else blk: {
                    defer allocator.free(raw.rgb);
                    break :blk try media.resizeRgb(allocator, raw.rgb, raw.w, raw.h, dw, dh);
                };
                item.w = dw;
                item.h = dh;
            }
            item.nchw = try media.rgbToNchwImagenet(allocator, item.rgb, item.h, item.w);
            item.latent_t = 1;
        }
        item.latent_h = item.h / spatial;
        item.latent_w = item.w / spatial;
        const video = item.kind == .video;
        const spec = vision.spatialTokens(vcfg, item.h, item.w, video);
        item.grid_h = spec.grid.h;
        item.grid_w = spec.grid.w;
        if (video) {
            var vis_frames = @max(2, (item.frames + 11) / 12);
            if (vis_frames % 2 != 0) vis_frames += 1;
            vis_frames = @min(vis_frames, 16);
            item.temporal = vis_frames / 2;
            item.seq = spec.seq * item.temporal;
            item.merged = spec.merged * item.temporal;
        } else {
            item.seq = spec.seq;
            item.merged = spec.merged;
        }
    }

    const hop: u32 = if (loaded_audio) |m| m.cfg.hop else 800;
    const rate: u32 = if (loaded_audio) |m| m.cfg.sample_rate else 32_000;
    var max_audio_samples: u32 = 0;
    for (audios.items) |*item| {
        if (compile_only) {
            const samples = media.wavSampleCount(allocator, io, item.path) catch hop;
            const aligned = hopAlign(samples, hop);
            item.latent_t = aligned / hop;
            max_audio_samples = @max(max_audio_samples, aligned);
            continue;
        }
        item.stereo = try media.loadWavStereo(allocator, io, item.path, rate);
        const samples: u32 = @intCast(item.stereo.len / 2);
        const aligned = hopAlign(samples, hop);
        item.latent_t = aligned / hop;
        max_audio_samples = @max(max_audio_samples, aligned);
    }

    var tokens: std.ArrayList(u32) = .empty;
    var tags: std.ArrayList(u8) = .empty;
    errdefer tokens.deinit(allocator);
    errdefer tags.deinit(allocator);
    try tokens.appendSlice(allocator, prompt_tokens);
    try tags.appendNTimes(allocator, @intFromEnum(packing.Modality.text), prompt_tokens.len);

    var spans: std.ArrayList(session_mod.VisionSpan) = .empty;
    errdefer spans.deinit(allocator);
    for (visuals.items) |item| {
        const pad = if (item.kind == .video) vision.VIDEO_PAD else vision.IMAGE_PAD;
        try tokens.appendSlice(allocator, colon_tokens);
        try tags.appendNTimes(allocator, @intFromEnum(packing.Modality.text), colon_tokens.len);
        try tokens.append(allocator, vision.VISION_START);
        try tags.append(allocator, @intFromEnum(packing.Modality.video));
        const start: u32 = @intCast(tokens.items.len);
        var p: u32 = 0;
        while (p < item.merged) : (p += 1) try tokens.append(allocator, pad);
        try tags.appendNTimes(allocator, @intFromEnum(packing.Modality.video), item.merged);
        try tokens.append(allocator, vision.VISION_END);
        try tags.append(allocator, @intFromEnum(packing.Modality.video));
        try spans.append(allocator, .{
            .start = start,
            .tokens = item.merged,
            .grid_h = item.grid_h,
            .grid_w = item.grid_w,
            .temporal = item.temporal,
        });
    }

    if (compile_only) {
        const dummy_v = try allocator.alloc(encode_mod.VisualLatent, visuals.items.len);
        var n_dv: usize = 0;
        defer {
            for (dummy_v[0..n_dv]) |d| allocator.free(d.thwc);
            allocator.free(dummy_v);
        }
        for (visuals.items, dummy_v) |item, *d| {
            d.* = .{
                .thwc = try allocator.alloc(f32, @as(usize, item.latent_t) * item.latent_h * item.latent_w * 24),
                .latent_t = item.latent_t,
                .latent_h = item.latent_h,
                .latent_w = item.latent_w,
                .keyframe_index = item.keyframe_index,
            };
            @memset(d.thwc, 0);
            n_dv += 1;
        }
        const dummy_a = try allocator.alloc(encode_mod.AudioLatent, audios.items.len);
        var n_da: usize = 0;
        defer {
            for (dummy_a[0..n_da]) |d| allocator.free(d.values);
            allocator.free(dummy_a);
        }
        for (audios.items, dummy_a) |item, *d| {
            d.* = .{ .values = try allocator.alloc(f32, @as(usize, item.latent_t) * 32), .latent_t = item.latent_t };
            @memset(d.values, 0);
            n_da += 1;
        }
        const tokens_out = try tokens.toOwnedSlice(allocator);
        errdefer allocator.free(tokens_out);
        const tags_out = try tags.toOwnedSlice(allocator);
        errdefer allocator.free(tags_out);
        const spans_out = try spans.toOwnedSlice(allocator);
        errdefer allocator.free(spans_out);
        return .{
            .tokens = tokens_out,
            .tags = tags_out,
            .vision_spans = spans_out,
            .conds = try encode_mod.packConditions(allocator, dummy_v, dummy_a, refs.items, patch),
        };
    }

    if (visuals.items.len != 0) {
        if (visual_store == null or loaded_visual == null) return error.VisualVaeMissing;
        if (!visual_enc.ready(visual_store.?.view())) return error.VisualEncodeMissing;
        if (!vision.ready(enc_store.view())) return error.VisionWeightsMissing;
    }
    if (audios.items.len != 0) {
        if (audio_store == null or loaded_audio == null) return error.AudioVaeMissing;
        if (!audio_vae.encodeReady(audio_store.?.view())) return error.AudioEncodeMissing;
    }

    var all = shardings.all();
    const positions = try allocator.alloc(f32, tokens.items.len * 3);
    errdefer allocator.free(positions);
    session_mod.fillEncoderPositions(positions, @intCast(tokens.items.len), spans.items);

    var merged_all: std.ArrayList(f32) = .empty;
    errdefer merged_all.deinit(allocator);
    var ds_host: [3][]f32 = .{ &.{}, &.{}, &.{} };
    errdefer for (ds_host) |d| if (d.len != 0) allocator.free(d);
    for (&ds_host) |*d| {
        d.* = try allocator.alloc(f32, tokens.items.len * hidden_dim);
        @memset(d.*, 0);
    }

    if (visuals.items.len != 0) {
        var loaded_vision = try vision.LoadedModel.init(allocator, io, enc_dir, enc_store.view(), text_hidden);
        defer loaded_vision.deinit(allocator);
        var compiled_v: ?pipeline.VisionCompiled = null;
        defer if (compiled_v) |*c| c.deinit();
        for (visuals.items, spans.items) |item, span| {
            if (compiled_v == null or compiled_v.?.seq != item.seq) {
                if (compiled_v) |*c| {
                    c.deinit();
                    compiled_v = null;
                }
                compiled_v = try pipeline.compileVision(allocator, io, platform, loaded_vision.inner, item.seq, shardings, progress);
            }
            var encoded = if (item.kind == .video) blk: {
                const vis_frames = item.temporal * 2;
                const sampled = try sampleRgbFrames(allocator, item.rgb, item.frames, item.w, item.h, vis_frames);
                defer allocator.free(sampled);
                break :blk try vision.runVideo(allocator, io, platform, &compiled_v.?, &loaded_vision, enc_store, &all, sampled, vis_frames, item.h, item.w, progress);
            } else try vision.runImage(allocator, io, platform, &compiled_v.?, &loaded_vision, enc_store, &all, item.rgb, item.h, item.w, progress);
            defer encoded.deinit(allocator);
            try merged_all.appendSlice(allocator, encoded.merged);
            for (0..3) |di| {
                if (encoded.deepstack[di].len != 0)
                    @memcpy(ds_host[di][@as(usize, span.start) * hidden_dim ..][0 .. span.tokens * hidden_dim], encoded.deepstack[di]);
            }
        }
    }

    var encoded_visuals = try allocator.alloc(encode_mod.VisualLatent, visuals.items.len);
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

    if (visuals.items.len != 0 or audios.items.len != 0) {
        const v_loaded: ?visual_enc.LoadedModel = if (visuals.items.len != 0)
            visual_enc.LoadedModel.init(visual_store.?.view(), loaded_visual.?.cfg)
        else
            null;
        var v_bufs: ?zml.Bufferized(visual_enc.Model) = if (v_loaded) |m|
            try m.loadBuffers(allocator, io, platform, visual_store.?, &all, progress)
        else
            null;
        defer if (v_bufs) |*b| visual_enc.Model.unloadBuffers(b);

        var a_loaded: ?audio_vae.LoadedEncoder = if (audios.items.len != 0)
            audio_vae.LoadedEncoder.init(audio_store.?.view(), loaded_audio.?.cfg)
        else
            null;
        var a_bufs: ?zml.Bufferized(audio_vae.EncoderModel) = if (a_loaded) |*m|
            try m.loadBuffers(allocator, io, platform, audio_store.?, &all, progress)
        else
            null;
        defer if (a_bufs) |*b| audio_vae.EncoderModel.unloadBuffers(b);

        var compiled_e = try pipeline.compileEncode(
            allocator,
            io,
            platform,
            if (v_loaded) |m| m.inner else null,
            if (a_loaded) |m| m.inner else null,
            vae.official_visual.tile_px,
            vae.official_visual.tile_px,
            hasVideo(visuals.items),
            if (max_audio_samples == 0) hop else max_audio_samples,
            shardings,
            progress,
        );
        defer compiled_e.deinit();

        for (visuals.items, encoded_visuals) |item, *out| {
            out.* = if (item.kind == .video)
                try encode_mod.encodeVideo(allocator, io, platform, &compiled_e, &v_loaded.?, &v_bufs.?, item.nchw.?, item.frames, item.h, item.w)
            else
                try encode_mod.encodeKeyframe(allocator, io, platform, &compiled_e, &v_loaded.?, &v_bufs.?, item.nchw.?, item.h, item.w);
            out.keyframe_index = item.keyframe_index;
            n_vis += 1;
        }
        for (audios.items, encoded_audios) |item, *out| {
            const padded = try padStereo(allocator, item.stereo, max_audio_samples);
            defer allocator.free(padded);
            out.* = try encode_mod.encodeAudio(allocator, io, platform, &compiled_e, &a_loaded.?, &a_bufs.?, padded);
            n_aud_enc += 1;
        }
    }

    const conds = try encode_mod.packConditions(allocator, encoded_visuals, encoded_audios, refs.items, patch);
    errdefer conds.deinit(allocator);
    for (encoded_visuals[0..n_vis]) |v| v.deinit(allocator);
    allocator.free(encoded_visuals);
    for (encoded_audios[0..n_aud_enc]) |a| a.deinit(allocator);
    allocator.free(encoded_audios);

    const tokens_out = try tokens.toOwnedSlice(allocator);
    errdefer allocator.free(tokens_out);
    const tags_out = try tags.toOwnedSlice(allocator);
    errdefer allocator.free(tags_out);
    const merged_out: ?[]f32 = if (merged_all.items.len == 0) null else try merged_all.toOwnedSlice(allocator);
    errdefer if (merged_out) |m| allocator.free(m);
    const vision_spans = try spans.toOwnedSlice(allocator);
    log.info(
        "conditions: ok tokens={d} vision_spans={d} video_conds={d} audio_conds={d} refs={d}",
        .{ tokens_out.len, vision_spans.len, conds.videos.len, conds.audios.len, conds.references.len },
    );
    return .{
        .tokens = tokens_out,
        .tags = tags_out,
        .positions = positions,
        .deepstack = .{ ds_host[0], ds_host[1], ds_host[2] },
        .vision_merged = merged_out,
        .vision_spans = vision_spans,
        .conds = conds,
    };
}
