const std = @import("std");

const zml = @import("zml");

const audio_vae = @import("../vae/audio.zig");
const config = @import("../core/config.zig");
const dit = @import("../model/dit.zig");
const encoder = @import("../model/encoder.zig");
const packing = @import("../model/packing.zig");
const policy = @import("../core/policy.zig");
const scheduler = @import("../model/scheduler.zig");
const sharding = @import("../core/sharding.zig");
const vae = @import("../vae/geometry.zig");
const vision = @import("../model/vision.zig");
const visual_enc = @import("../vae/visual_encoder.zig");
const visual_vae = @import("../vae/visual.zig");

const log = std.log.scoped(.minimax_h3);

pub const Options = struct {
    duration_s: f32 = 5.0,
    width: u32 = 1344,
    height: u32 = 768,
    frames: u32 = 0,
    steps: u32 = 30,
    video_shift: f32 = config.video_shift,
    audio_shift: f32 = config.audio_shift,
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

    pub fn init(opts: Options, dit_cfg: config.Config) Geometry {
        const frames = if (opts.frames != 0) opts.frames else config.alignFrameCount(config.frameCount(opts.duration_s));
        const lat = config.visualLatentSize(opts.height, opts.width, frames);
        const audio_t = config.audioLatentFromFrames(frames);
        const vt = config.videoTokenCount(lat.t, lat.h, lat.w, dit_cfg.patch_size);
        const at = vae.official_audio.tokenCount(audio_t);
        return .{
            .pixel_w = opts.width,
            .pixel_h = opts.height,
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

pub const CompilePolicy = struct {
    attention: zml.attention.Backend = .vanilla,
    group_size: u32 = 1,
    steps: u32,
    hold_video: i64 = 0,
    hold_audio: i64 = 0,
    vision_tokens: u32 = 0,
};

pub const Compiled = struct {
    prepare_text: zml.FnExe(dit.prepareText),
    prepare_rope: zml.FnExe(dit.prepareRope),
    embed_patches: zml.FnExe(dit.embedPatches),
    prepare_temb: zml.FnExe(dit.prepareTemb),
    prepare_adaln: zml.FnExe(dit.prepareAdaln),
    prepare_final_adaln: zml.FnExe(dit.prepareAdaln),
    block: zml.FnExe(dit.stepBlock),
    block_group: ?zml.FnExe(dit.BlockGroup.forward) = null,
    group_size: u32 = 1,
    finish: zml.FnExe(dit.finish),
    apply_video: zml.FnExe(scheduler.apply),
    apply_audio: zml.FnExe(scheduler.apply),
    encode_embed: zml.FnExe(encoder.EmbedTokens.forward),
    encode_layer: zml.FnExe(encoder.TransformerLayer.forward),
    encode_scatter: ?zml.FnExe(dit.scatterRows) = null,

    pub fn deinit(self: *Compiled) void {
        self.prepare_text.deinit();
        self.prepare_rope.deinit();
        self.embed_patches.deinit();
        self.prepare_temb.deinit();
        self.prepare_adaln.deinit();
        self.prepare_final_adaln.deinit();
        self.block.deinit();
        if (self.block_group) |*g| g.deinit();
        self.finish.deinit();
        self.apply_video.deinit();
        self.apply_audio.deinit();
        self.encode_embed.deinit();
        self.encode_layer.deinit();
        if (self.encode_scatter) |*s| s.deinit();
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

fn compilePrepareText(ctx: CompileCtx, dit_model: dit.Model, enc_dt: zml.DataType, text_len: u32) !zml.FnExe(dit.prepareText) {
    return compileLogged(dit.prepareText, "minimax_h3_prepare_text", ctx, .{.{
        .model = dit_model.textPrep(),
        .text = .init(.{ .b = 1, .s = text_len, .d = dit_model.cfg.text_dim }, enc_dt),
    }});
}

fn compilePrepareRope(ctx: CompileCtx, dit_model: dit.Model, seq_len: u32, out_dt: zml.DataType) !zml.FnExe(dit.prepareRope) {
    return compileLogged(dit.prepareRope, "minimax_h3_prepare_rope", ctx, .{.{
        .model = .{
            .rope_freq_dim = dit_model.cfg.rope_freq_dim,
            .rope_theta = dit_model.cfg.rope_theta,
            .out_dtype = out_dt,
        },
        .position_ids = .init(.{ .s = seq_len, .ax = 3 }, .f32),
    }});
}

fn compileEmbedPatches(ctx: CompileCtx, dit_model: dit.Model, geo: Geometry, text_len: u32, seq_len: u32, text_dt: zml.DataType) !zml.FnExe(dit.embedPatches) {
    var part = dit_model.patchEmbed();
    part.seq = seq_len;
    return compileLogged(dit.embedPatches, "minimax_h3_embed_patches", ctx, .{.{
        .model = part,
        .video = .init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32),
        .audio = .init(.{ .b = 1, .s = geo.audio_tokens, .d = geo.audio_dim }, .f32),
        .text = .init(.{ .b = 1, .s = text_len, .d = dit_model.cfg.hidden_size }, text_dt),
        .video_indices = .init(.{ .s = geo.video_tokens }, .u32),
        .audio_indices = .init(.{ .s = geo.audio_tokens }, .u32),
        .text_indices = .init(.{ .s = text_len }, .u32),
    }});
}

fn compilePrepareTemb(ctx: CompileCtx, dit_model: dit.Model, n_slots: u32) !zml.FnExe(dit.prepareTemb) {
    return compileLogged(dit.prepareTemb, "minimax_h3_prepare_temb", ctx, .{.{
        .model = dit_model.time_embedder,
        .timestep = .init(.{ .n = n_slots }, .f32),
        .freq_dim = dit_model.cfg.freq_dim,
    }});
}

fn compilePrepareBlockAdaln(ctx: CompileCtx, dit_model: dit.Model, n_slots: u32, steps: u32) !zml.FnExe(dit.prepareAdaln) {
    return compileLogged(dit.prepareAdaln, "minimax_h3_prepare_adaln", ctx, .{.{
        .model = .{
            .adaln = dit_model.blocks[0].adaln,
            .steps = steps,
            .slots = packing.timestep_slot_count,
        },
        .temb = .init(.{ .n = n_slots, .d = dit_model.time_embedder.outDim() }, .f32),
    }});
}

fn compilePrepareFinalAdaln(ctx: CompileCtx, dit_model: dit.Model, n_slots: u32, steps: u32) !zml.FnExe(dit.prepareAdaln) {
    return compileLogged(dit.prepareAdaln, "minimax_h3_prepare_final_adaln", ctx, .{.{
        .model = .{
            .adaln = dit_model.final_layer.adaln,
            .steps = steps,
            .slots = packing.timestep_slot_count,
        },
        .temb = .init(.{ .n = n_slots, .d = dit_model.time_embedder.outDim() }, .f32),
    }});
}

fn compileDitBlock(ctx: CompileCtx, dit_model: dit.Model, seq_len: u32, steps: u32) !zml.FnExe(dit.stepBlock) {
    const dt = dit_model.blocks[0].norm1.weight.dtype();
    const table = zml.Tensor.init(.{
        .t = steps,
        .n = packing.timestep_slot_count,
        .mod = config.modality_count,
        .k = 6,
        .d = dit_model.cfg.hidden_size,
    }, dt);
    return compileLogged(dit.stepBlock, "minimax_h3_block", ctx, .{.{
        .layer = dit_model.blocks[0].corePart(),
        .hidden = zml.Tensor.init(.{ .b = 1, .s = seq_len, .d = dit_model.cfg.hidden_size }, dt),
        .table = table,
        .step = zml.Tensor.init(.{}, .u32),
        .adaln_indices = zml.Tensor.init(.{ .s = seq_len }, .u32),
        .cos = zml.Tensor.init(.{ .s = seq_len, .f = dit_model.cfg.rotaryDim() }, dt),
        .sin = zml.Tensor.init(.{ .s = seq_len, .f = dit_model.cfg.rotaryDim() }, dt),
    }});
}

fn compileDitGroup(ctx: CompileCtx, dit_model: dit.Model, seq_len: u32, steps: u32, group_size: u32) !zml.FnExe(dit.BlockGroup.forward) {
    const dt = dit_model.blocks[0].norm1.weight.dtype();
    const n: usize = group_size;
    const layers = try ctx.allocator.alloc(dit.BlockCore, n);
    defer ctx.allocator.free(layers);
    const tables = try ctx.allocator.alloc(zml.Tensor, n);
    defer ctx.allocator.free(tables);
    if (dit_model.blocks.len < n) return error.DitGroupTooLarge;
    for (layers, tables, 0..) |*layer, *tab, i| {
        layer.* = dit_model.blocks[i].corePart();
        tab.* = zml.Tensor.init(.{
            .t = steps,
            .n = packing.timestep_slot_count,
            .mod = config.modality_count,
            .k = 6,
            .d = dit_model.cfg.hidden_size,
        }, dt);
    }
    return compileLogged(dit.BlockGroup.forward, "minimax_h3_block_group", ctx, .{.{
        .group = .{ .layers = layers },
        .hidden = zml.Tensor.init(.{ .b = 1, .s = seq_len, .d = dit_model.cfg.hidden_size }, dt),
        .tables = tables,
        .step = zml.Tensor.init(.{}, .u32),
        .adaln_indices = zml.Tensor.init(.{ .s = seq_len }, .u32),
        .cos = zml.Tensor.init(.{ .s = seq_len, .f = dit_model.cfg.rotaryDim() }, dt),
        .sin = zml.Tensor.init(.{ .s = seq_len, .f = dit_model.cfg.rotaryDim() }, dt),
    }});
}

fn compileDitFinish(ctx: CompileCtx, dit_model: dit.Model, geo: Geometry, seq_len: u32, steps: u32) !zml.FnExe(dit.finish) {
    const dt = dit_model.blocks[0].norm1.weight.dtype();
    return compileLogged(dit.finish, "minimax_h3_finish", ctx, .{.{
        .model = dit_model.finishCore(),
        .hidden = zml.Tensor.init(.{ .b = 1, .s = seq_len, .d = dit_model.cfg.hidden_size }, dt),
        .table = zml.Tensor.init(.{ .t = steps, .n = packing.timestep_slot_count, .k = 2, .d = dit_model.cfg.hidden_size }, dt),
        .step = zml.Tensor.init(.{}, .u32),
        .timestep_indices = .init(.{ .s = seq_len }, .u32),
        .video_indices = .init(.{ .s = geo.video_tokens }, .u32),
        .audio_indices = .init(.{ .s = geo.audio_tokens }, .u32),
    }});
}

fn compileApplyVideo(ctx: CompileCtx, tokens: u32, dim: u32, hold: i64) !zml.FnExe(scheduler.apply) {
    return compileLogged(scheduler.apply, "minimax_h3_apply_video", ctx, .{.{
        .model = .{ .hold = hold },
        .sample = .init(.{ .b = 1, .s = tokens, .d = dim }, .f32),
        .velocity = .init(.{ .b = 1, .s = tokens, .d = dim }, .f32),
        .sigma = .init(.{}, .f32),
        .sigma_next = .init(.{}, .f32),
        .sigma_t = .init(.{}, .f32),
    }});
}

fn compileApplyAudio(ctx: CompileCtx, tokens: u32, dim: u32, hold: i64) !zml.FnExe(scheduler.apply) {
    return compileLogged(scheduler.apply, "minimax_h3_apply_audio", ctx, .{.{
        .model = .{ .hold = hold },
        .sample = .init(.{ .b = 1, .s = tokens, .d = dim }, .f32),
        .velocity = .init(.{ .b = 1, .s = tokens, .d = dim }, .f32),
        .sigma = .init(.{}, .f32),
        .sigma_next = .init(.{}, .f32),
        .sigma_t = .init(.{}, .f32),
    }});
}

fn compileScatter(ctx: CompileCtx, seq: u32, hidden: i64, n: u32, dt: zml.DataType) !zml.FnExe(dit.scatterRows) {
    return compileLogged(dit.scatterRows, "minimax_h3_scatter", ctx, .{.{
        .hidden = .init(.{ .b = 1, .s = seq, .d = hidden }, dt),
        .values = .init(.{ .b = 1, .s = n, .d = hidden }, .f32),
        .indices = .init(.{ .s = n }, .u32),
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
    compile_policy: CompilePolicy,
    shardings: sharding.Shardings,
    progress: *std.Progress.Node,
) !Compiled {
    var model = dit_model;
    const tp: u32 = @intCast(shardings.model.numPartitionsForLogicalAxis(.model));
    const dit_dt = model.blocks[0].norm1.weight.dtype();
    const flash = zml.attention.Backend.auto(platform);
    const refiner_attn = policy.selectAttention(.{
        .target = platform.target,
        .dtype = dit_dt,
        .head_dim = model.cfg.attention_head_dim,
        .heads = model.cfg.num_attention_heads,
        .seq = text_len,
        .causal = false,
        .tp = tp,
        .flash = flash,
    });
    model.applyBackend(compile_policy.attention, refiner_attn);
    var enc_work = enc_model;
    const enc_attn = policy.selectAttention(.{
        .target = platform.target,
        .dtype = enc_model.embed_tokens.weight.dtype(),
        .head_dim = enc_model.cfg.head_dim,
        .heads = enc_model.cfg.num_attention_heads,
        .seq = text_len,
        .causal = true,
        .tp = tp,
        .flash = flash,
    });
    enc_work.applyBackend(enc_attn);

    var all = shardings.all();
    const ctx: CompileCtx = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .shardings = &all,
        .progress = progress,
    };
    const jobs: u32 = 12 + @as(u32, @intFromBool(compile_policy.group_size > 1)) + @as(u32, @intFromBool(compile_policy.vision_tokens > 0));
    var node = progress.start("Compiling MiniMax-H3", jobs);
    defer node.end();

    const dt = dit_dt;
    const enc_dt = enc_model.embed_tokens.weight.dtype();
    const n_flat = compile_policy.steps * packing.timestep_slot_count;
    log.info(
        "compile DiT+encoder: start seq={d} text={d} video_tokens={d} audio_tokens={d} attn={s} group={d} steps={d}",
        .{
            seq_len,
            text_len,
            geo.video_tokens,
            geo.audio_tokens,
            @tagName(compile_policy.attention),
            compile_policy.group_size,
            compile_policy.steps,
        },
    );
    const now: std.Io.Timestamp = .now(io, .awake);

    var text_f = try io.concurrent(compilePrepareText, .{ ctx, model, enc_dt, text_len });
    errdefer if (text_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var rope_f = try io.concurrent(compilePrepareRope, .{ ctx, model, seq_len, dt });
    errdefer if (rope_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var patch_f = try io.concurrent(compileEmbedPatches, .{ ctx, model, geo, text_len, seq_len, dt });
    errdefer if (patch_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var temb_f = try io.concurrent(compilePrepareTemb, .{ ctx, model, n_flat });
    errdefer if (temb_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var adaln_f = try io.concurrent(compilePrepareBlockAdaln, .{ ctx, model, n_flat, compile_policy.steps });
    errdefer if (adaln_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var final_f = try io.concurrent(compilePrepareFinalAdaln, .{ ctx, model, n_flat, compile_policy.steps });
    errdefer if (final_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var block_f = try io.concurrent(compileDitBlock, .{ ctx, model, seq_len, compile_policy.steps });
    errdefer if (block_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var group_f: ?@TypeOf(try io.concurrent(compileDitGroup, .{ ctx, model, seq_len, compile_policy.steps, compile_policy.group_size })) = null;
    if (compile_policy.group_size > 1) {
        group_f = try io.concurrent(compileDitGroup, .{ ctx, model, seq_len, compile_policy.steps, compile_policy.group_size });
    }
    errdefer if (group_f) |*f| if (f.cancel(io)) |exe| exe.deinit() else |_| {};
    var finish_f = try io.concurrent(compileDitFinish, .{ ctx, model, geo, seq_len, compile_policy.steps });
    errdefer if (finish_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var apply_v_f = try io.concurrent(compileApplyVideo, .{ ctx, geo.video_tokens, geo.video_patch_dim, compile_policy.hold_video });
    errdefer if (apply_v_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var apply_a_f = try io.concurrent(compileApplyAudio, .{ ctx, geo.audio_tokens, geo.audio_dim, compile_policy.hold_audio });
    errdefer if (apply_a_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var enc_embed_f = try io.concurrent(compileEncEmbed, .{ ctx, enc_work, text_len });
    errdefer if (enc_embed_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var enc_layer_f = try io.concurrent(compileEncLayer, .{ ctx, enc_work, text_len });
    errdefer if (enc_layer_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var scatter_f: ?@TypeOf(try io.concurrent(compileScatter, .{ ctx, text_len, enc_work.cfg.hidden_size, compile_policy.vision_tokens, enc_dt })) = null;
    if (compile_policy.vision_tokens > 0) {
        scatter_f = try io.concurrent(compileScatter, .{ ctx, text_len, enc_work.cfg.hidden_size, compile_policy.vision_tokens, enc_dt });
    }
    errdefer if (scatter_f) |*f| if (f.cancel(io)) |exe| exe.deinit() else |_| {};

    const prepare_text = try text_f.await(io);
    errdefer prepare_text.deinit();
    const prepare_rope = try rope_f.await(io);
    errdefer prepare_rope.deinit();
    const embed_patches = try patch_f.await(io);
    errdefer embed_patches.deinit();
    const prepare_temb = try temb_f.await(io);
    errdefer prepare_temb.deinit();
    const prepare_adaln = try adaln_f.await(io);
    errdefer prepare_adaln.deinit();
    const prepare_final_adaln = try final_f.await(io);
    errdefer prepare_final_adaln.deinit();
    const block_exe = try block_f.await(io);
    errdefer block_exe.deinit();
    const block_group = if (group_f) |*f| try f.await(io) else null;
    errdefer if (block_group) |exe| {
        var tmp = exe;
        tmp.deinit();
    };
    const finish_exe = try finish_f.await(io);
    errdefer finish_exe.deinit();
    const apply_video = try apply_v_f.await(io);
    errdefer apply_video.deinit();
    const apply_audio = try apply_a_f.await(io);
    errdefer apply_audio.deinit();
    const encode_embed = try enc_embed_f.await(io);
    errdefer encode_embed.deinit();
    const encode_layer = try enc_layer_f.await(io);
    errdefer encode_layer.deinit();
    const encode_scatter = if (scatter_f) |*f| try f.await(io) else null;
    errdefer if (encode_scatter) |exe| {
        var tmp = exe;
        tmp.deinit();
    };

    log.info("Compiled MiniMax-H3 [{f}] seq={d} video_tokens={d} audio_tokens={d} attn={s}", .{
        now.untilNow(io, .awake),
        seq_len,
        geo.video_tokens,
        geo.audio_tokens,
        @tagName(compile_policy.attention),
    });

    return .{
        .prepare_text = prepare_text,
        .prepare_rope = prepare_rope,
        .embed_patches = embed_patches,
        .prepare_temb = prepare_temb,
        .prepare_adaln = prepare_adaln,
        .prepare_final_adaln = prepare_final_adaln,
        .block = block_exe,
        .block_group = block_group,
        .group_size = compile_policy.group_size,
        .finish = finish_exe,
        .apply_video = apply_video,
        .apply_audio = apply_audio,
        .encode_embed = encode_embed,
        .encode_layer = encode_layer,
        .encode_scatter = encode_scatter,
    };
}

pub const VaeCompiled = struct {
    embed: zml.FnExe(visual_vae.embed),
    block: zml.FnExe(visual_vae.TransformerBlock.forward),
    finish: zml.FnExe(visual_vae.finish),
    audio: ?zml.FnExe(audio_vae.decode) = null,
    tile: visual_vae.TileShape,
    tile_batch: u32 = 1,
    partition_b: bool = false,

    pub fn deinit(self: *VaeCompiled) void {
        self.embed.deinit();
        self.block.deinit();
        self.finish.deinit();
        if (self.audio) |*exe| exe.deinit();
    }
};

fn vaeBatchShape(tags: anytype, dt: zml.DataType, partition_b: bool) zml.Tensor {
    const t = zml.Tensor.init(tags, dt);
    return if (partition_b) t.withPartitioning(.{ .b = .model }) else t;
}

fn compileVaeEmbed(ctx: CompileCtx, visual: visual_vae.Model, tile: visual_vae.TileShape, seq: u32, tile_batch: u32, partition_b: bool) !zml.FnExe(visual_vae.embed) {
    return compileLogged(visual_vae.embed, "minimax_h3_vae_embed", ctx, .{.{
        .model = visual.embed,
        .latents = vaeBatchShape(.{ .b = tile_batch, .s = tile.tokens(), .d = visual.cfg.latent_channels }, .f32, partition_b),
        .position_ids = .init(.{ .s = seq, .ax = 3 }, .f32),
    }});
}

fn compileVaeBlock(ctx: CompileCtx, visual: visual_vae.Model, seq: u32, tile_batch: u32, partition_b: bool) !zml.FnExe(visual_vae.TransformerBlock.forward) {
    const dt = visual.embed.proj.weight.dtype();
    return compileLogged(visual_vae.TransformerBlock.forward, "minimax_h3_vae_block", ctx, .{.{
        .layer = visual.blocks[0],
        .hidden = vaeBatchShape(.{ .b = tile_batch, .s = seq, .d = visual.cfg.dim() }, dt, partition_b),
        .cos = .init(.{ .s = seq, .f = visual.cfg.rotaryDim() }, dt),
        .sin = .init(.{ .s = seq, .f = visual.cfg.rotaryDim() }, dt),
    }});
}

fn compileVaeFinish(ctx: CompileCtx, visual: visual_vae.Model, seq: u32, tile_batch: u32, partition_b: bool) !zml.FnExe(visual_vae.finish) {
    return compileLogged(visual_vae.finish, "minimax_h3_vae_finish", ctx, .{.{
        .model = visual.finish,
        .hidden = vaeBatchShape(.{ .b = tile_batch, .s = seq, .d = visual.cfg.dim() }, visual.embed.proj.weight.dtype(), partition_b),
    }});
}

pub fn compileAudioDecode(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    audio: audio_vae.Model,
    geo: Geometry,
    shardings: []const zml.Sharding,
    progress: *std.Progress.Node,
) !zml.FnExe(audio_vae.decode) {
    const ctx: CompileCtx = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .shardings = shardings,
        .progress = progress,
    };
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
    geo: Geometry,
    tile_batch: u32,
    shardings: sharding.Shardings,
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
    const batch = @max(1, tile_batch);
    const tp: u32 = @intCast(shardings.model.numPartitionsForLogicalAxis(.model));
    const partition_b = partitionsVaeBatch(batch, tp);
    var node = progress.start("Compiling MiniMax-H3 VAE", 3);
    defer node.end();

    log.info("compile VAE: start tile={d}x{d}x{d} audio_t={d} batch={d} shard_b={}", .{
        tile.latent_t,
        tile.latent_h,
        tile.latent_w,
        geo.audio_t,
        batch,
        partition_b,
    });
    const now: std.Io.Timestamp = .now(io, .awake);
    var embed_f = try io.concurrent(compileVaeEmbed, .{ ctx, visual, tile, seq, batch, partition_b });
    errdefer if (embed_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var block_f = try io.concurrent(compileVaeBlock, .{ ctx, visual, seq, batch, partition_b });
    errdefer if (block_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var finish_f = try io.concurrent(compileVaeFinish, .{ ctx, visual, seq, batch, partition_b });
    errdefer if (finish_f.cancel(io)) |exe| exe.deinit() else |_| {};

    const embed_exe = try embed_f.await(io);
    errdefer embed_exe.deinit();
    const block_exe = try block_f.await(io);
    errdefer block_exe.deinit();
    const finish_exe = try finish_f.await(io);
    errdefer finish_exe.deinit();

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
        .tile = tile,
        .tile_batch = batch,
        .partition_b = partition_b,
    };
}

/// Data-parallel `.b` only when every rank gets the same tile count.
pub fn partitionsVaeBatch(batch: u32, tp: u32) bool {
    return batch > 1 and tp > 1 and batch % tp == 0;
}

pub const EncodeCompiled = struct {
    visual_t1: ?zml.FnExe(visual_enc.encode) = null,
    visual_clip: ?zml.FnExe(visual_enc.encode) = null,
    tile_h: u32,
    tile_w: u32,

    pub fn deinit(self: *EncodeCompiled) void {
        if (self.visual_t1) |*c| c.deinit();
        if (self.visual_clip) |*c| c.deinit();
    }
};

fn compileVisualEncode(ctx: CompileCtx, model: visual_enc.Model, t: u32, h: u32, w: u32) !zml.FnExe(visual_enc.encode) {
    return compileLogged(visual_enc.encode, "minimax_h3_visual_encode", ctx, .{.{
        .model = model,
        .pixels = .init(.{ .b = 1, .c = 3, .t = t, .h = h, .w = w }, .f32),
    }});
}

fn compileAudioEncodeInner(ctx: CompileCtx, model: audio_vae.EncoderModel, samples: u32) !zml.FnExe(audio_vae.encode) {
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
    tile_h: u32,
    tile_w: u32,
    need_clip: bool,
    shardings: sharding.Shardings,
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
    return .{
        .visual_t1 = t1,
        .visual_clip = clip,
        .tile_h = tile_h,
        .tile_w = tile_w,
    };
}

pub fn compileAudioEncode(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model: audio_vae.EncoderModel,
    samples: u32,
    shardings: sharding.Shardings,
    progress: *std.Progress.Node,
) !zml.FnExe(audio_vae.encode) {
    var all = shardings.all();
    return compileAudioEncodeInner(.{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .shardings = &all,
        .progress = progress,
    }, model, samples);
}

pub fn compileVision(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model: vision.Model,
    seq: u32,
    shardings: sharding.Shardings,
    progress: *std.Progress.Node,
) !vision.Compiled {
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
        .cos = .init(.{ .s = seq, .hd = cfg.headDim() }, .f32),
        .sin = .init(.{ .s = seq, .hd = cfg.headDim() }, .f32),
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

pub const Packed = struct {
    layout: packing.Layout,
    schedules: scheduler.DualSchedule,

    pub fn deinit(self: *Packed, allocator: std.mem.Allocator) void {
        self.layout.deinit(allocator);
        self.schedules.deinit(allocator);
    }
};

pub fn pack(
    allocator: std.mem.Allocator,
    opts: Options,
    geo: Geometry,
    text_len: u32,
    text_tags: []const u8,
    videos: []const packing.ConditionVideo,
    audios: []const packing.ConditionAudio,
    references: []const packing.ReferenceBlock,
) !Packed {
    const schedules = try scheduler.DualSchedule.init(allocator, opts.steps, opts.video_shift, opts.audio_shift);
    errdefer schedules.deinit(allocator);
    const video_t = schedules.video.timesteps[0];
    const audio_t = schedules.audio.timesteps[0];
    const layout = try packing.build(allocator, .{
        .text_len = text_len,
        .latent_t = geo.latent_t,
        .latent_h = geo.latent_h,
        .latent_w = geo.latent_w,
        .audio_t = geo.audio_t,
        .video_t = video_t,
        .audio_t_noise = audio_t,
        .condition_videos = videos,
        .condition_audios = audios,
        .references = references,
        .text_tags = text_tags,
    });
    return .{ .layout = layout, .schedules = schedules };
}
