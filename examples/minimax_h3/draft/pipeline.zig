const std = @import("std");

const zml = @import("zml");

const audio_vae = @import("audio.zig");
const config = @import("../recipe/config.zig");
const dit = @import("dit.zig");
const encoder = @import("encoder.zig");
const packing = @import("packing.zig");
const policy = @import("../recipe/policy.zig");
const scheduler = @import("scheduler.zig");
const sharding = @import("../recipe/shard.zig");
const vae = @import("geometry.zig");

const log = std.log.scoped(.minimax_h3);

// =============================================================================
// draft/pipeline.zig — Stage 1 compile
//
// Encoder + DiT graphs for one SKU geometry. session.draft runs them.
// =============================================================================

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
    video_patch_dim: u32,
    audio_dim: u32,

    pub fn init(opts: Options, dit_cfg: config.Config) Geometry {
        if (opts.frames == 0) {
            std.debug.assert(std.math.isFinite(opts.duration_s));
            std.debug.assert(opts.duration_s >= 5.0 and opts.duration_s <= 15.0);
        }
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
            .video_patch_dim = @intCast(dit_cfg.videoPatchDim()),
            .audio_dim = @intCast(dit_cfg.audio_in_channels),
        };
    }
};

pub const CompilePolicy = struct {
    attention: zml.attention.Backend = .vanilla,
    group_size: u32 = 1,
    steps: u32,
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
    text_len: u32 = 0,
    n_slots: u32 = 0,
    owns_shared: bool = true,

    pub fn deinit(self: *Compiled) void {
        if (self.owns_shared) {
            self.prepare_text.deinit();
            self.prepare_temb.deinit();
            self.prepare_adaln.deinit();
            self.prepare_final_adaln.deinit();
            self.encode_embed.deinit();
            self.encode_layer.deinit();
        }
        self.prepare_rope.deinit();
        self.embed_patches.deinit();
        self.block.deinit();
        if (self.block_group) |*g| g.deinit();
        self.finish.deinit();
        self.apply_video.deinit();
        self.apply_audio.deinit();
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

fn compileApplyVideo(ctx: CompileCtx, tokens: u32, dim: u32) !zml.FnExe(scheduler.apply) {
    return compileLogged(scheduler.apply, "minimax_h3_apply_video", ctx, .{.{
        .sample = .init(.{ .b = 1, .s = tokens, .d = dim }, .f32),
        .velocity = .init(.{ .b = 1, .s = tokens, .d = dim }, .f32),
        .sigma = .init(.{}, .f32),
        .sigma_next = .init(.{}, .f32),
        .sigma_t = .init(.{}, .f32),
    }});
}

fn compileApplyAudio(ctx: CompileCtx, tokens: u32, dim: u32) !zml.FnExe(scheduler.apply) {
    return compileLogged(scheduler.apply, "minimax_h3_apply_audio", ctx, .{.{
        .sample = .init(.{ .b = 1, .s = tokens, .d = dim }, .f32),
        .velocity = .init(.{ .b = 1, .s = tokens, .d = dim }, .f32),
        .sigma = .init(.{}, .f32),
        .sigma_next = .init(.{}, .f32),
        .sigma_t = .init(.{}, .f32),
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
    shared: ?*const Compiled,
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

    var all = shardings.all();
    const ctx: CompileCtx = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .shardings = &all,
        .progress = progress,
    };
    const dt = dit_dt;
    const enc_dt = enc_model.embed_tokens.weight.dtype();
    const n_flat = compile_policy.steps * packing.timestep_slot_count;
    const share = if (shared) |s| s.text_len == text_len and s.n_slots == n_flat else false;
    const jobs: u32 = (if (share) @as(u32, 6) else 12) + @as(u32, @intFromBool(compile_policy.group_size > 1));
    var node = progress.start("Compiling MiniMax-H3", jobs);
    defer node.end();
    log.info(
        "compile DiT+encoder: start seq={d} text={d} video_tokens={d} audio_tokens={d} attn={s} group={d} steps={d}{s}",
        .{
            seq_len,
            text_len,
            geo.video_tokens,
            geo.audio_tokens,
            @tagName(compile_policy.attention),
            compile_policy.group_size,
            compile_policy.steps,
            if (share) " reuse encoder" else "",
        },
    );
    const now: std.Io.Timestamp = .now(io, .awake);

    var text_f: ?@TypeOf(try io.concurrent(compilePrepareText, .{ ctx, model, enc_dt, text_len })) = null;
    if (!share) text_f = try io.concurrent(compilePrepareText, .{ ctx, model, enc_dt, text_len });
    errdefer if (text_f) |*f| if (f.cancel(io)) |exe| exe.deinit() else |_| {};
    var rope_f = try io.concurrent(compilePrepareRope, .{ ctx, model, seq_len, dt });
    errdefer if (rope_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var patch_f = try io.concurrent(compileEmbedPatches, .{ ctx, model, geo, text_len, seq_len, dt });
    errdefer if (patch_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var temb_f: ?@TypeOf(try io.concurrent(compilePrepareTemb, .{ ctx, model, n_flat })) = null;
    if (!share) temb_f = try io.concurrent(compilePrepareTemb, .{ ctx, model, n_flat });
    errdefer if (temb_f) |*f| if (f.cancel(io)) |exe| exe.deinit() else |_| {};
    var adaln_f: ?@TypeOf(try io.concurrent(compilePrepareBlockAdaln, .{ ctx, model, n_flat, compile_policy.steps })) = null;
    if (!share) adaln_f = try io.concurrent(compilePrepareBlockAdaln, .{ ctx, model, n_flat, compile_policy.steps });
    errdefer if (adaln_f) |*f| if (f.cancel(io)) |exe| exe.deinit() else |_| {};
    var final_f: ?@TypeOf(try io.concurrent(compilePrepareFinalAdaln, .{ ctx, model, n_flat, compile_policy.steps })) = null;
    if (!share) final_f = try io.concurrent(compilePrepareFinalAdaln, .{ ctx, model, n_flat, compile_policy.steps });
    errdefer if (final_f) |*f| if (f.cancel(io)) |exe| exe.deinit() else |_| {};
    var block_f = try io.concurrent(compileDitBlock, .{ ctx, model, seq_len, compile_policy.steps });
    errdefer if (block_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var group_f: ?@TypeOf(try io.concurrent(compileDitGroup, .{ ctx, model, seq_len, compile_policy.steps, compile_policy.group_size })) = null;
    if (compile_policy.group_size > 1) {
        group_f = try io.concurrent(compileDitGroup, .{ ctx, model, seq_len, compile_policy.steps, compile_policy.group_size });
    }
    errdefer if (group_f) |*f| if (f.cancel(io)) |exe| exe.deinit() else |_| {};
    var finish_f = try io.concurrent(compileDitFinish, .{ ctx, model, geo, seq_len, compile_policy.steps });
    errdefer if (finish_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var apply_v_f = try io.concurrent(compileApplyVideo, .{ ctx, geo.video_tokens, geo.video_patch_dim });
    errdefer if (apply_v_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var apply_a_f = try io.concurrent(compileApplyAudio, .{ ctx, geo.audio_tokens, geo.audio_dim });
    errdefer if (apply_a_f.cancel(io)) |exe| exe.deinit() else |_| {};
    var enc_embed_f: ?@TypeOf(try io.concurrent(compileEncEmbed, .{ ctx, enc_model, text_len })) = null;
    if (!share) enc_embed_f = try io.concurrent(compileEncEmbed, .{ ctx, enc_model, text_len });
    errdefer if (enc_embed_f) |*f| if (f.cancel(io)) |exe| exe.deinit() else |_| {};
    var enc_layer_f: ?@TypeOf(try io.concurrent(compileEncLayer, .{ ctx, enc_model, text_len })) = null;
    if (!share) enc_layer_f = try io.concurrent(compileEncLayer, .{ ctx, enc_model, text_len });
    errdefer if (enc_layer_f) |*f| if (f.cancel(io)) |exe| exe.deinit() else |_| {};
    const prepare_text = if (share) shared.?.prepare_text else try text_f.?.await(io);
    errdefer if (!share) prepare_text.deinit();
    const prepare_rope = try rope_f.await(io);
    errdefer prepare_rope.deinit();
    const embed_patches = try patch_f.await(io);
    errdefer embed_patches.deinit();
    const prepare_temb = if (share) shared.?.prepare_temb else try temb_f.?.await(io);
    errdefer if (!share) prepare_temb.deinit();
    const prepare_adaln = if (share) shared.?.prepare_adaln else try adaln_f.?.await(io);
    errdefer if (!share) prepare_adaln.deinit();
    const prepare_final_adaln = if (share) shared.?.prepare_final_adaln else try final_f.?.await(io);
    errdefer if (!share) prepare_final_adaln.deinit();
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
    const encode_embed = if (share) shared.?.encode_embed else try enc_embed_f.?.await(io);
    errdefer if (!share) encode_embed.deinit();
    const encode_layer = if (share) shared.?.encode_layer else try enc_layer_f.?.await(io);
    errdefer if (!share) encode_layer.deinit();
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
        .text_len = text_len,
        .n_slots = n_flat,
        .owns_shared = !share,
    };
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
    });
    return .{ .layout = layout, .schedules = schedules };
}
