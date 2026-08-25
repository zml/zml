const std = @import("std");

const zml = @import("zml");

const audio_vae = @import("audio_vae.zig");
const config_mod = @import("config.zig");
const dit = @import("dit.zig");
const encoder = @import("encoder.zig");
const packing = @import("packing.zig");
const sharding_mod = @import("sharding.zig");
const scheduler_mod = @import("scheduler.zig");
const vae = @import("vae.zig");
const vision = @import("vision.zig");
const visual_enc = @import("visual_enc.zig");
const visual_vae = @import("visual_vae.zig");

const log = std.log.scoped(.minimax_h3);

pub const Options = struct {
    variant: config_mod.Variant = .t2va,
    duration_s: f32 = 5.0,
    aspect: config_mod.Aspect = .@"16:9",
    short_side: u32 = config_mod.default_short_side,
    steps: u32 = 30,
    seed: u64 = 0,
    video_shift: f32 = config_mod.video_shift,
    audio_shift: f32 = config_mod.audio_shift,
};

pub const Geometry = struct {
    pixel_w: u32,
    pixel_h: u32,
    frames: u32,
    latent_t: u32,
    latent_h: u32,
    latent_w: u32,
    audio_t: u32,
    video_tokens: u32,
    audio_tokens: u32,
    target_video_tokens: u32,
    target_audio_tokens: u32,
    video_patch_dim: u32,
    audio_dim: u32,

    pub fn init(opts: Options, dit_cfg: config_mod.Config) Geometry {
        const px = config_mod.pixelSize(opts.aspect, opts.short_side);
        const frames = config_mod.alignFrameCount(config_mod.frameCount(opts.duration_s));
        const lat = config_mod.visualLatentSize(px.h, px.w, frames);
        const audio_t = config_mod.audioLatentLength(opts.duration_s);
        const vt = config_mod.videoTokenCount(lat.t, lat.h, lat.w, dit_cfg.patch_size);
        const at = vae.official_audio.tokenCount(audio_t);
        return .{
            .pixel_w = px.w,
            .pixel_h = px.h,
            .frames = frames,
            .latent_t = lat.t,
            .latent_h = lat.h,
            .latent_w = lat.w,
            .audio_t = audio_t,
            .video_tokens = vt,
            .audio_tokens = at,
            .target_video_tokens = vt,
            .target_audio_tokens = at,
            .video_patch_dim = @intCast(dit_cfg.videoPatchDim()),
            .audio_dim = @intCast(dit_cfg.audio_in_channels),
        };
    }

    pub fn withConditions(self: Geometry, extra_video: u32, extra_audio: u32) Geometry {
        var out = self;
        out.video_tokens = self.target_video_tokens + extra_video;
        out.audio_tokens = self.target_audio_tokens + extra_audio;
        return out;
    }
};

pub const Compiled = struct {
    embed: zml.FnExe(dit.embed),
    block: zml.FnExe(dit.TransformerBlock.forward),
    finish: zml.FnExe(dit.finish),
    encode_embed: zml.FnExe(encoder.EmbedTokens.forward),
    encode_layer: zml.FnExe(encoder.TransformerLayer.forward),

    pub fn deinit(self: *Compiled) void {
        self.embed.deinit();
        self.block.deinit();
        self.finish.deinit();
        self.encode_embed.deinit();
        self.encode_layer.deinit();
    }
};

const CompileCtx = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shardings: []const zml.Sharding,
    progress: *std.Progress.Node,
};

fn compileLogged(
    comptime function: anytype,
    comptime name: []const u8,
    ctx: CompileCtx,
    args: std.meta.ArgsTuple(@TypeOf(function)),
) !zml.FnExe(function) {
    ctx.progress.increaseEstimatedTotalItems(1);
    const now: std.Io.Timestamp = .now(ctx.io, .awake);
    const exe = try zml.FnExe(function).compile(ctx.allocator, ctx.io, ctx.platform, .{
        .shardings = ctx.shardings,
        .program_name = name,
    }, args);
    log.info("compile {s}: ok [{f}]", .{ name, now.untilNow(ctx.io, .awake) });
    return exe;
}

fn compileDitEmbed(ctx: CompileCtx, dit_model: dit.Model, enc_dt: zml.DataType, geo: Geometry, text_len: u32, seq_len: u32, num_timesteps: u32) !zml.FnExe(dit.embed) {
    return compileLogged(dit.embed, "minimax_h3_embed", ctx, .{.{
        .model = dit_model.embedPart(),
        .video = .init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32),
        .audio = .init(.{ .b = 1, .s = geo.audio_tokens, .d = geo.audio_dim }, .f32),
        .text = .init(.{ .b = 1, .s = text_len, .d = dit_model.cfg.text_dim }, enc_dt),
        .timestep = .init(.{ .n = num_timesteps }, .f32),
        .position_ids = .init(.{ .s = seq_len, .ax = 3 }, .f32),
        .video_indices = .init(.{ .s = geo.video_tokens }, .u32),
        .audio_indices = .init(.{ .s = geo.audio_tokens }, .u32),
        .text_indices = .init(.{ .s = text_len }, .u32),
    }});
}

fn compileDitBlock(ctx: CompileCtx, dit_model: dit.Model, seq_len: u32, num_timesteps: u32) !zml.FnExe(dit.TransformerBlock.forward) {
    const dt = dit_model.blocks[0].norm1.weight.dtype();
    return compileLogged(dit.TransformerBlock.forward, "minimax_h3_block", ctx, .{.{
        .layer = dit_model.blocks[0],
        .hidden = zml.Tensor.init(.{ .b = 1, .s = seq_len, .d = dit_model.cfg.hidden_size }, dt),
        .temb = zml.Tensor.init(.{ .n = num_timesteps, .d = dit_model.cfg.time_embed_dim }, .f32),
        .adaln_indices = zml.Tensor.init(.{ .s = seq_len }, .u32),
        .cos = zml.Tensor.init(.{ .s = seq_len, .f = dit_model.cfg.rotaryDim() }, dt),
        .sin = zml.Tensor.init(.{ .s = seq_len, .f = dit_model.cfg.rotaryDim() }, dt),
    }});
}

fn compileDitFinish(ctx: CompileCtx, dit_model: dit.Model, geo: Geometry, seq_len: u32, num_timesteps: u32) !zml.FnExe(dit.finish) {
    const dt = dit_model.blocks[0].norm1.weight.dtype();
    return compileLogged(dit.finish, "minimax_h3_finish", ctx, .{.{
        .model = dit_model.finishPart(),
        .hidden = zml.Tensor.init(.{ .b = 1, .s = seq_len, .d = dit_model.cfg.hidden_size }, dt),
        .temb = zml.Tensor.init(.{ .n = num_timesteps, .d = dit_model.cfg.time_embed_dim }, .f32),
        .timestep_indices = .init(.{ .s = seq_len }, .u32),
        .video_indices = .init(.{ .s = geo.video_tokens }, .u32),
        .audio_indices = .init(.{ .s = geo.audio_tokens }, .u32),
    }});
}

fn compileEncEmbed(ctx: CompileCtx, enc_model: encoder.Model, text_len: u32) !zml.FnExe(encoder.EmbedTokens.forward) {
    return compileLogged(encoder.EmbedTokens.forward, "minimax_h3_encoder_embed", ctx, .{.{
        .embedding = .{ .embed_tokens = enc_model.embed_tokens },
        .tokens = .init(.{ .b = 1, .s = text_len }, .u32),
    }});
}

fn compileEncLayer(ctx: CompileCtx, enc_model: encoder.Model, text_len: u32) !zml.FnExe(encoder.TransformerLayer.forward) {
    const dt = enc_model.embed_tokens.weight.dtype();
    const hd = enc_model.cfg.head_dim;
    return compileLogged(encoder.TransformerLayer.forward, "minimax_h3_encoder_layer", ctx, .{.{
        .layer = enc_model.layers[0],
        .hidden = .init(.{ .b = 1, .s = text_len, .d = enc_model.cfg.hidden_size }, dt),
        .cos = .init(.{ .s = text_len, .hd = hd }, dt),
        .sin = .init(.{ .s = text_len, .hd = hd }, dt),
        .visual_delta = .init(.{ .b = 1, .s = text_len, .d = enc_model.cfg.hidden_size }, dt),
    }});
}

pub fn compile(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    dit_model: dit.Model,
    enc_model: encoder.Model,
    geo: Geometry,
    text_len: u32,
    seq_len: u32,
    num_timesteps: u32,
    shardings: sharding_mod.Shardings,
    progress: *std.Progress.Node,
) !Compiled {
    var all = shardings.all();
    const ctx: CompileCtx = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .shardings = &all,
        .progress = progress,
    };
    var node = progress.start("Compiling MiniMax-H3", 5);
    defer node.end();

    log.info("compile DiT+encoder: start seq={d} text={d} video_tokens={d} audio_tokens={d}", .{
        seq_len,
        text_len,
        geo.video_tokens,
        geo.audio_tokens,
    });
    const now: std.Io.Timestamp = .now(io, .awake);
    var embed_f = try io.concurrent(compileDitEmbed, .{ ctx, dit_model, enc_model.embed_tokens.weight.dtype(), geo, text_len, seq_len, num_timesteps });
    errdefer if (embed_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var block_f = try io.concurrent(compileDitBlock, .{ ctx, dit_model, seq_len, num_timesteps });
    errdefer if (block_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var finish_f = try io.concurrent(compileDitFinish, .{ ctx, dit_model, geo, seq_len, num_timesteps });
    errdefer if (finish_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var enc_embed_f = try io.concurrent(compileEncEmbed, .{ ctx, enc_model, text_len });
    errdefer if (enc_embed_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var enc_layer_f = try io.concurrent(compileEncLayer, .{ ctx, enc_model, text_len });
    errdefer if (enc_layer_f.cancel(io)) |exe| exe.deinit() else |_| {};

    const embed_exe = try embed_f.await(io);
    errdefer embed_exe.deinit();
    const block_exe = try block_f.await(io);
    errdefer block_exe.deinit();
    const finish_exe = try finish_f.await(io);
    errdefer finish_exe.deinit();
    const encode_embed = try enc_embed_f.await(io);
    errdefer encode_embed.deinit();
    const encode_layer = try enc_layer_f.await(io);
    errdefer encode_layer.deinit();

    log.info("Compiled MiniMax-H3 [{f}] seq={d} video_tokens={d} audio_tokens={d}", .{
        now.untilNow(io, .awake),
        seq_len,
        geo.video_tokens,
        geo.audio_tokens,
    });

    return .{
        .embed = embed_exe,
        .block = block_exe,
        .finish = finish_exe,
        .encode_embed = encode_embed,
        .encode_layer = encode_layer,
    };
}

pub const VaeCompiled = struct {
    embed: zml.FnExe(visual_vae.embed),
    block: zml.FnExe(visual_vae.TransformerBlock.forward),
    finish: zml.FnExe(visual_vae.finish),
    audio: zml.FnExe(audio_vae.decode),
    tile: visual_vae.TileShape,

    pub fn deinit(self: *VaeCompiled) void {
        self.embed.deinit();
        self.block.deinit();
        self.finish.deinit();
        self.audio.deinit();
    }
};

fn compileVaeEmbed(ctx: CompileCtx, visual: visual_vae.Model, tile: visual_vae.TileShape, seq: u32) !zml.FnExe(visual_vae.embed) {
    return compileLogged(visual_vae.embed, "minimax_h3_vae_embed", ctx, .{.{
        .model = visual.embed,
        .latents = .init(.{ .b = 1, .s = tile.tokens(), .d = visual.cfg.latent_channels }, .f32),
        .position_ids = .init(.{ .s = seq, .ax = 3 }, .f32),
    }});
}

fn compileVaeBlock(ctx: CompileCtx, visual: visual_vae.Model, seq: u32) !zml.FnExe(visual_vae.TransformerBlock.forward) {
    const dt = visual.embed.proj.weight.dtype();
    return compileLogged(visual_vae.TransformerBlock.forward, "minimax_h3_vae_block", ctx, .{.{
        .layer = visual.blocks[0],
        .hidden = .init(.{ .b = 1, .s = seq, .d = visual.cfg.dim() }, dt),
        .cos = .init(.{ .s = seq, .f = visual.cfg.rotaryDim() }, dt),
        .sin = .init(.{ .s = seq, .f = visual.cfg.rotaryDim() }, dt),
    }});
}

fn compileVaeFinish(ctx: CompileCtx, visual: visual_vae.Model, seq: u32) !zml.FnExe(visual_vae.finish) {
    return compileLogged(visual_vae.finish, "minimax_h3_vae_finish", ctx, .{.{
        .model = visual.finish,
        .hidden = .init(.{ .b = 1, .s = seq, .d = visual.cfg.dim() }, visual.embed.proj.weight.dtype()),
    }});
}

fn compileAudioDecode(ctx: CompileCtx, audio: audio_vae.Model, geo: Geometry) !zml.FnExe(audio_vae.decode) {
    return compileLogged(audio_vae.decode, "minimax_h3_audio_decode", ctx, .{.{
        .model = audio,
        .latents = .init(.{ .b = 2, .c = audio.cfg.latent_channels, .t = geo.audio_t }, .f32),
    }});
}

pub fn compileVae(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    visual: visual_vae.Model,
    audio: audio_vae.Model,
    geo: Geometry,
    shardings: sharding_mod.Shardings,
    progress: *std.Progress.Node,
) !VaeCompiled {
    var all = shardings.all();
    const ctx: CompileCtx = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .shardings = &all,
        .progress = progress,
    };
    const tile = visual_vae.TileShape.fromGeometry(visual.cfg, geo.latent_t, geo.latent_h, geo.latent_w);
    const registers: u32 = @intCast(visual.cfg.decoder_num_register_tokens);
    const seq = tile.seq(registers);
    var node = progress.start("Compiling MiniMax-H3 VAE", 4);
    defer node.end();

    log.info("compile VAE: start tile={d}x{d}x{d} audio_t={d}", .{
        tile.latent_t,
        tile.latent_h,
        tile.latent_w,
        geo.audio_t,
    });
    const now: std.Io.Timestamp = .now(io, .awake);
    var embed_f = try io.concurrent(compileVaeEmbed, .{ ctx, visual, tile, seq });
    errdefer if (embed_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var block_f = try io.concurrent(compileVaeBlock, .{ ctx, visual, seq });
    errdefer if (block_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var finish_f = try io.concurrent(compileVaeFinish, .{ ctx, visual, seq });
    errdefer if (finish_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var audio_f = try io.concurrent(compileAudioDecode, .{ ctx, audio, geo });
    errdefer if (audio_f.cancel(io)) |exe| exe.deinit() else |_| {};

    const embed_exe = try embed_f.await(io);
    errdefer embed_exe.deinit();
    const block_exe = try block_f.await(io);
    errdefer block_exe.deinit();
    const finish_exe = try finish_f.await(io);
    errdefer finish_exe.deinit();
    const audio_exe = try audio_f.await(io);
    errdefer audio_exe.deinit();

    log.info("Compiled MiniMax-H3 VAE tile={d}x{d}x{d} audio_t={d} [{f}]", .{
        tile.latent_t,
        tile.latent_h,
        tile.latent_w,
        geo.audio_t,
        now.untilNow(io, .awake),
    });

    return .{
        .embed = embed_exe,
        .block = block_exe,
        .finish = finish_exe,
        .audio = audio_exe,
        .tile = tile,
    };
}

pub fn adalnIndices(allocator: std.mem.Allocator, layout: packing.Layout) ![]u32 {
    const out = try allocator.alloc(u32, layout.seqLen());
    for (out, 0..) |*v, i| v.* = layout.adalnIndex(i);
    return out;
}

pub const EncodeCompiled = struct {
    visual_t1: ?zml.FnExe(visual_enc.encode) = null,
    visual_clip: ?zml.FnExe(visual_enc.encode) = null,
    audio: ?zml.FnExe(audio_vae.encode) = null,
    tile_h: u32,
    tile_w: u32,

    pub fn deinit(self: *EncodeCompiled) void {
        if (self.visual_t1) |*c| c.deinit();
        if (self.visual_clip) |*c| c.deinit();
        if (self.audio) |*c| c.deinit();
    }
};

fn compileVisualEncode(ctx: CompileCtx, model: visual_enc.Model, t: u32, h: u32, w: u32) !zml.FnExe(visual_enc.encode) {
    return compileLogged(visual_enc.encode, "minimax_h3_visual_encode", ctx, .{.{
        .model = model,
        .pixels = .init(.{ .b = 1, .c = 3, .t = t, .h = h, .w = w }, .f32),
    }});
}

fn compileAudioEncode(ctx: CompileCtx, model: audio_vae.EncoderModel, samples: u32) !zml.FnExe(audio_vae.encode) {
    return compileLogged(audio_vae.encode, "minimax_h3_audio_encode", ctx, .{.{
        .model = model,
        .wav = .init(.{ .b = 2, .c = 1, .t = samples }, .f32),
    }});
}

pub fn compileEncode(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    visual: ?visual_enc.Model,
    audio: ?audio_vae.EncoderModel,
    tile_h: u32,
    tile_w: u32,
    need_clip: bool,
    audio_samples: u32,
    shardings: sharding_mod.Shardings,
    progress: *std.Progress.Node,
) !EncodeCompiled {
    var all = shardings.all();
    const ctx: CompileCtx = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .shardings = &all,
        .progress = progress,
    };
    const t1 = if (visual) |m| try compileVisualEncode(ctx, m, 1, tile_h, tile_w) else null;
    errdefer if (t1) |exe| {
        var tmp = exe;
        tmp.deinit();
    };
    const clip = if (need_clip) blk: {
        const m = visual orelse return error.VisualEncodeMissing;
        break :blk try compileVisualEncode(ctx, m, 17, tile_h, tile_w);
    } else null;
    errdefer if (clip) |exe| {
        var tmp = exe;
        tmp.deinit();
    };
    const audio_exe = if (audio) |m| try compileAudioEncode(ctx, m, audio_samples) else null;
    errdefer if (audio_exe) |exe| {
        var tmp = exe;
        tmp.deinit();
    };
    return .{
        .visual_t1 = t1,
        .visual_clip = clip,
        .audio = audio_exe,
        .tile_h = tile_h,
        .tile_w = tile_w,
    };
}

pub const VisionCompiled = struct {
    embed: zml.FnExe(vision.embed),
    block: zml.FnExe(vision.VisionBlock.forward),
    merger: zml.FnExe(vision.Merger.forward),
    deepstack: zml.FnExe(vision.Merger.forward),
    seq: u32,
    merged: u32,

    pub fn deinit(self: *VisionCompiled) void {
        self.embed.deinit();
        self.block.deinit();
        self.merger.deinit();
        self.deepstack.deinit();
    }
};

pub fn compileVision(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model: vision.Model,
    seq: u32,
    shardings: sharding_mod.Shardings,
    progress: *std.Progress.Node,
) !VisionCompiled {
    var all = shardings.all();
    const ctx: CompileCtx = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .shardings = &all,
        .progress = progress,
    };
    const cfg = model.cfg;
    const dt = model.embed.proj.weight.dtype();
    const merged: u32 = @intCast(@divExact(@as(i64, seq), cfg.mergeUnit()));
    const embed_exe = try compileLogged(vision.embed, "minimax_h3_vision_embed", ctx, .{.{
        .model = model.embed,
        .patches = .init(.{ .b = 1, .s = seq, .d = cfg.patchIn() }, .f32),
        .pos = .init(.{ .b = 1, .s = seq, .d = cfg.hidden_size }, .f32),
    }});
    errdefer embed_exe.deinit();
    const block_exe = try compileLogged(vision.VisionBlock.forward, "minimax_h3_vision_block", ctx, .{.{
        .layer = model.blocks[0],
        .hidden = .init(.{ .b = 1, .s = seq, .d = cfg.hidden_size }, dt),
        .cos = .init(.{ .s = seq, .hd = cfg.headDim() }, dt),
        .sin = .init(.{ .s = seq, .hd = cfg.headDim() }, dt),
    }});
    errdefer block_exe.deinit();
    const merger_exe = try compileLogged(vision.Merger.forward, "minimax_h3_vision_merger", ctx, .{.{
        .model = model.merger,
        .hidden = .init(.{ .b = 1, .s = seq, .d = cfg.hidden_size }, dt),
    }});
    errdefer merger_exe.deinit();
    const ds_exe = try compileLogged(vision.Merger.forward, "minimax_h3_vision_deepstack", ctx, .{.{
        .model = model.deepstack[0],
        .hidden = .init(.{ .b = 1, .s = seq, .d = cfg.hidden_size }, dt),
    }});
    errdefer ds_exe.deinit();
    return .{
        .embed = embed_exe,
        .block = block_exe,
        .merger = merger_exe,
        .deepstack = ds_exe,
        .seq = seq,
        .merged = merged,
    };
}

pub fn describe(opts: Options, geo: Geometry, layout: packing.Layout) void {
    log.info(
        "layout {s} {d}x{d} {d} frames ({d:.1}s) latents {d}x{d}x{d} audio_t={d} seq={d} steps={d} seed={d}",
        .{
            @tagName(opts.variant),
            geo.pixel_w,
            geo.pixel_h,
            geo.frames,
            opts.duration_s,
            geo.latent_t,
            geo.latent_h,
            geo.latent_w,
            geo.audio_t,
            layout.seqLen(),
            opts.steps,
            opts.seed,
        },
    );
}
