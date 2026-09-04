const std = @import("std");

const zml = @import("zml");

const config = @import("../recipe/config.zig");
const connector = @import("connector.zig");
const euler = @import("euler.zig");
const gemma4 = @import("gemma.zig");
const load = @import("load.zig");
const lora = @import("../recipe/lora.zig");
const ltx_dit = @import("ltx_dit.zig");
const ltx_rope = @import("ltx_rope.zig");
const ltx_upsampler = @import("ltx_up.zig");
const ltx_vae = @import("ltx_vae.zig");
const media = @import("../serve/media.zig");
const noise = @import("../draft/noise.zig");
const prepare = @import("handoff.zig");
const session = @import("../draft/session.zig");
const sku = @import("../recipe/sku.zig");
const taehv = @import("taehv.zig");
const weights = @import("../recipe/weights.zig");

const log = std.log.scoped(.minimax_h3);

// =============================================================================
// refine/run.zig — Stage 2 compile + infer
//
// VAE → upsample → 3 Euler steps → TAEHV. Gemma / connector / DiT / conv weights shared.
// =============================================================================

pub const WeightStores = struct {
    mu: std.Io.Mutex = .init,
    vae: ?load.Store = null,
    up: ?load.Store = null,
    dit: ?load.Store = null,
    tae: ?load.Store = null,

    pub fn deinit(self: *WeightStores) void {
        if (self.dit) |*s| s.deinit();
        if (self.vae) |*s| s.deinit();
        if (self.up) |*s| s.deinit();
        if (self.tae) |*s| s.deinit();
        self.* = .{};
    }

    fn open(
        self: *WeightStores,
        slot: *?load.Store,
        allocator: std.mem.Allocator,
        io: std.Io,
        paths: []const []const u8,
        missing: anyerror,
    ) !*load.Store {
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        if (slot.*) |*s| return s;
        const path = load.firstExisting(io, paths) orelse return missing;
        slot.* = try load.Store.open(allocator, io, path);
        return &slot.*.?;
    }
};

pub const Request = struct {
    prompt: []const u8,
    seed: u64,
    target_w: u32,
    target_h: u32,
    out: []const u8,
    audio: []const f32 = &.{},
    job: ?*session.JobProgress = null,
};

pub const Video = struct {
    nchw: []f32,
    frames: u32,
    height: u32,
    width: u32,

    pub fn deinit(self: Video, allocator: std.mem.Allocator) void {
        allocator.free(self.nchw);
    }
};

pub const Compiled = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    handoff: *prepare.Compiled,
    vae: ltx_vae.Compiled,
    up: ltx_upsampler.Compiled,
    gemma: *gemma4.Compiled,
    conn: connector.Compiled,
    dit: *ltx_dit.Compiled,
    apply: zml.FnExe(applyEuler),
    tae: taehv.Compiled,
    ctx: ?zml.Buffer = null,
    cos_v: zml.Buffer,
    sin_v: zml.Buffer,
    tau: zml.Buffer,
    time: u32,
    height: u32,
    width: u32,
    owns_text: bool = true,
    owns_tae: bool = true,

    pub fn deinit(self: *Compiled) void {
        if (self.ctx) |*c| c.deinit();
        self.handoff.deinit();
        self.allocator.destroy(self.handoff);
        self.vae.deinit();
        self.up.deinit();
        if (self.owns_text) {
            self.gemma.deinit();
            self.allocator.destroy(self.gemma);
            self.conn.deinit();
        }
        self.dit.deinit();
        self.allocator.destroy(self.dit);
        self.apply.deinit();
        if (self.owns_tae) self.tae.deinit();
        self.cos_v.deinit();
        self.sin_v.deinit();
        self.tau.deinit();
    }
};

pub fn compile(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shardings: []const zml.Sharding,
    rep_shardings: []const zml.Sharding,
    progress: *std.Progress.Node,
    src_t: u32,
    src_h: u32,
    src_w: u32,
    src_frames: u32,
    target_w: u32,
    target_h: u32,
    shared: ?*Compiled,
    tae_donors: []const *taehv.Compiled,
    stores: *WeightStores,
) !*Compiled {
    const out = try allocator.create(Compiled);
    errdefer allocator.destroy(out);
    log.info("stage2 compile start", .{});
    const handoff = try prepare.compile(
        allocator,
        io,
        platform,
        rep_shardings,
        progress,
        src_t,
        src_h,
        src_w,
        src_frames,
        target_w,
        target_h,
    );
    errdefer {
        handoff.deinit();
        allocator.destroy(handoff);
    }
    if (!handoff.matches_cpu) return error.HandoffMismatch;
    const enc = try sku.refineEncodeSize(target_w, target_h);
    const kept = 1 + ((src_frames - 1) / 8) * 8;
    log.info("stage2 compile encode {d}x{d} frames={d}", .{ enc.w, enc.h, kept });

    const vae_store = try stores.open(&stores.vae, allocator, io, &ltx_vae.weight_paths, error.LtxVaeMissing);
    var vae_bind = vae_store.bind(allocator);
    defer vae_bind.deinit();
    var vae = try ltx_vae.compile(
        allocator,
        io,
        platform,
        ltx_vae.Encoder.init(vae_bind.view()),
        kept,
        enc.h,
        enc.w,
        rep_shardings,
        &vae_bind,
        progress,
        if (shared) |src| &src.vae else null,
    );
    errdefer vae.deinit();

    const lat = ltx_vae.latentSize(kept, enc.h, enc.w);
    const up_store = try stores.open(&stores.up, allocator, io, &ltx_upsampler.weight_paths, error.LtxUpsamplerMissing);
    var up_bind = up_store.bind(allocator);
    defer up_bind.deinit();
    var up = try ltx_upsampler.compile(
        allocator,
        io,
        platform,
        ltx_upsampler.Model.init(up_bind.view()),
        lat.t,
        lat.h,
        lat.w,
        rep_shardings,
        &up_bind,
        progress,
        if (shared) |src| &src.up else null,
    );
    errdefer up.deinit();

    const gemma = if (shared) |src| src.gemma else blk: {
        const gemma_path = load.firstExisting(io, &gemma4.weight_paths) orelse return error.GemmaMissing;
        var gemma_store = try load.Store.open(allocator, io, gemma_path);
        defer gemma_store.deinit();
        break :blk try gemma4.compile(allocator, io, platform, &gemma_store.store, shardings, progress);
    };
    errdefer if (shared == null) {
        gemma.deinit();
        allocator.destroy(gemma);
    };

    const dit_path = load.firstExisting(io, &ltx_dit.weight_paths) orelse return error.LtxDitMissing;
    const dit_store = try stores.open(&stores.dit, allocator, io, &ltx_dit.weight_paths, error.LtxDitMissing);
    var dit_bind = dit_store.bind(allocator);
    defer dit_bind.deinit();
    var conn = if (shared) |src| src.conn else try connector.compile(allocator, io, platform, &dit_bind, shardings, progress, gemma4.keep_tokens);
    errdefer if (shared == null) conn.deinit();

    var lora_bundle: ?lora.Bundle = null;
    defer if (lora_bundle) |*b| b.deinit();
    const baked = std.mem.indexOf(u8, dit_path, "lora08") != null;
    if (baked) {
        log.info("stage2 using baked LoRA 0.8 {s}", .{dit_path});
    } else if (load.firstExisting(io, &ltx_dit.lora_paths)) |p| {
        lora_bundle = try lora.load(allocator, io, p, sku.ltx_lora_strength);
    }
    const up_h = lat.h * ltx_upsampler.spatial_factor;
    const up_w = lat.w * ltx_upsampler.spatial_factor;
    const dit = try ltx_dit.compile(
        allocator,
        io,
        platform,
        &dit_bind,
        shardings,
        progress,
        lat.t,
        up_h,
        up_w,
        if (shared == null) if (lora_bundle) |*b| b else null else null,
        if (shared) |src| src.dit else null,
    );
    errdefer {
        dit.deinit();
        allocator.destroy(dit);
    }

    const tokens = dit.tokens;
    const cos_v = try allocator.alloc(f32, @intCast(tokens * ltx_dit.heads * ltx_dit.head_dim));
    defer allocator.free(cos_v);
    const sin_v = try allocator.alloc(f32, cos_v.len);
    defer allocator.free(sin_v);
    ltx_rope.fillVideo(cos_v, sin_v, lat.t, up_h, up_w, 24);
    const rope_sh = zml.Shape.init(.{ .s = tokens, .h = ltx_dit.heads, .hd = ltx_dit.head_dim }, .bf16).withPartitioning(.{ .h = .model });
    const model_shard = if (shardings.len != 0) shardings[0] else zml.Sharding.replicated;
    var cos_vb = try weights.fromF32Sharded(allocator, io, platform, rope_sh, model_shard, cos_v);
    errdefer cos_vb.deinit();
    var sin_vb = try weights.fromF32Sharded(allocator, io, platform, rope_sh, model_shard, sin_v);
    errdefer sin_vb.deinit();
    var tau = try weights.fromItems(io, platform, .init(.{}, .f32), &[_]f32{0});
    errdefer tau.deinit();

    var apply = try zml.FnExe(applyEuler).compile(allocator, io, platform, .{
        .shardings = rep_shardings,
        .program_name = "minimax_h3_ltx_euler",
    }, .{.{
        .x = .init(.{ .n = 1, .c = 128, .t = lat.t, .h = up_h, .w = up_w }, .f32),
        .v = .init(.{ .n = 1, .c = 128, .t = lat.t, .h = up_h, .w = up_w }, .f32),
        .sigma = .init(.{}, .f32),
        .sigma_next = .init(.{}, .f32),
    }});
    errdefer apply.deinit();

    const tae_t = taehv.decodeTime(lat.t, up_h, sku.hdUpsampledH());
    const picked = pickTae(tae_donors, tae_t, up_h, up_w);
    var owns_tae = true;
    var tae: taehv.Compiled = undefined;
    if (picked.full) |src| {
        tae = src.*;
        tae.owns_bufs = false;
        owns_tae = false;
        log.info("TAEHV reuse {d}x{d} t={d}", .{ up_w, up_h, tae_t });
    } else {
        const tae_store = try stores.open(&stores.tae, allocator, io, &taehv.weight_paths, error.TaehvMissing);
        var tae_bind = tae_store.bind(allocator);
        defer tae_bind.deinit();
        const tae_m = taehv.Model.init(tae_bind.view(), 1, tae_t);
        tae = try taehv.compile(allocator, io, platform, tae_m, up_h, up_w, rep_shardings, &tae_bind, progress, picked.bufs);
    }
    errdefer if (owns_tae) tae.deinit();

    log.info(
        "compiled Stage 2 encode {d}x{d}x{d} latent {d}x{d}x{d} refine {d}x{d}x{d}",
        .{ enc.w, enc.h, kept, lat.w, lat.h, lat.t, up_w, up_h, lat.t },
    );
    out.* = .{
        .allocator = allocator,
        .io = io,
        .handoff = handoff,
        .vae = vae,
        .up = up,
        .gemma = gemma,
        .conn = conn,
        .dit = dit,
        .apply = apply,
        .tae = tae,
        .cos_v = cos_vb,
        .sin_v = sin_vb,
        .tau = tau,
        .time = lat.t,
        .height = up_h,
        .width = up_w,
        .owns_text = shared == null,
        .owns_tae = owns_tae,
    };
    return out;
}

pub fn refreshContext(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    exe: *Compiled,
    prompt: []const u8,
) !void {
    const tokens = gemma4.tokenizePrompt(allocator, io, prompt) catch |err| {
        log.err("stage2 gemma tokenize failed ({})", .{err});
        return error.GemmaMissing;
    };
    defer allocator.free(tokens);
    const proj = try gemma4.run(allocator, io, platform, exe.gemma, tokens);
    defer allocator.free(proj);
    const ctx = try runConnector(allocator, io, platform, &exe.conn, proj);
    if (exe.ctx) |*old| old.deinit();
    exe.ctx = ctx;
}

pub fn infer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    exe: *Compiled,
    pixels: zml.Buffer,
    req: Request,
) !Video {
    const job = req.job;
    if (job) |j| j.set(.vae, 1, 1);
    var vae_run = try zml.FnExe(ltx_vae.encode).Runner(.{.model}).init(&exe.vae.encode, allocator, .{ .model = exe.vae.bufs });
    defer vae_run.deinit(allocator);
    var vae_lat: zml.Buffer = undefined;
    vae_run.run(io, .{
        .inputs = .{ .pixels = pixels },
        .outputs = .{ .latent = &vae_lat },
        .opts = .{ .wait = true },
    });
    defer vae_lat.deinit();

    var up_run = try zml.FnExe(ltx_upsampler.forward).Runner(.{.model}).init(&exe.up.forward, allocator, .{ .model = exe.up.bufs });
    defer up_run.deinit(allocator);
    var up_initial: zml.Buffer = undefined;
    var up_pre: zml.Buffer = undefined;
    var up_mid: zml.Buffer = undefined;
    var up_post: zml.Buffer = undefined;
    var up_lat: zml.Buffer = undefined;
    up_run.run(io, .{
        .inputs = .{
            .latent = vae_lat,
            .mean = exe.vae.bufs.mean,
            .std = exe.vae.bufs.std,
        },
        .outputs = .{
            .after_initial = &up_initial,
            .after_pre = &up_pre,
            .after_up = &up_mid,
            .after_post = &up_post,
            .latent = &up_lat,
        },
        .opts = .{ .wait = true },
    });
    defer up_initial.deinit();
    defer up_pre.deinit();
    defer up_mid.deinit();
    defer up_post.deinit();
    defer up_lat.deinit();

    const ctx = exe.ctx orelse return error.GemmaMissing;
    var embed_run = try zml.FnExe(ltx_dit.Embed.forward).Runner(.{.model}).init(&exe.dit.embed, allocator, .{ .model = exe.dit.embed_bufs });
    defer embed_run.deinit(allocator);
    var fin = try zml.FnExe(ltx_dit.Finish.forward).Runner(.{.model}).init(&exe.dit.finish, allocator, .{ .model = exe.dit.finish_bufs });
    defer fin.deinit(allocator);
    const BlockRunner = zml.FnExe(ltx_dit.Block.forward).Runner(.{.layer});
    const block_runners = try allocator.alloc(BlockRunner, exe.dit.n);
    {
        var li: u32 = 0;
        while (li < exe.dit.n) : (li += 1) {
            block_runners[li] = try BlockRunner.init(exe.dit.blockExe(li), allocator, .{ .layer = exe.dit.blocks[li] });
        }
    }
    defer {
        var ri: u32 = 0;
        while (ri < exe.dit.n) : (ri += 1) block_runners[ri].deinit(allocator);
        allocator.free(block_runners);
    }
    var ar = try zml.FnExe(applyEuler).Runner(.{}).init(&exe.apply, allocator, .{});
    defer ar.deinit(allocator);
    var tae_run = try zml.FnExe(taehv.decode).Runner(.{.model}).init(&exe.tae.decode, allocator, .{ .model = exe.tae.bufs });
    defer tae_run.deinit(allocator);

    var x = try mixConstStart(allocator, io, platform, &up_lat, req.seed, euler.sigmaAt(0));
    var step: u32 = 0;
    while (step < euler.evals) : (step += 1) {
        if (job) |j| j.set(.refine, step + 1, euler.evals);
        const sigma = euler.sigmaAt(step);
        const sigma_next = euler.sigmaAt(step + 1);
        var tau_b = try weights.fromItems(io, platform, .init(.{}, .f32), &[_]f32{euler.tauAt(step)});
        defer tau_b.deinit();
        var ts = try weights.fromF32(allocator, io, platform, .init(.{ .n = 1 }, .f32), &[_]f32{sigma});
        defer ts.deinit();
        var hid: zml.Buffer = undefined;
        var ada: zml.Buffer = undefined;
        var pada: zml.Buffer = undefined;
        var emb: zml.Buffer = undefined;
        embed_run.run(io, .{
            .inputs = .{ .latent = x, .timestep = ts },
            .outputs = .{ .hidden = &hid, .ada = &ada, .prompt_ada = &pada, .embedded = &emb },
            .opts = .{ .wait = false },
        });
        defer ada.deinit();
        defer pada.deinit();
        defer emb.deinit();
        var layer_i: u32 = 0;
        while (layer_i < exe.dit.n) : (layer_i += 1) {
            var next: zml.Buffer = undefined;
            block_runners[layer_i].run(io, .{
                .inputs = .{
                    .hidden = hid,
                    .context = ctx,
                    .ada = ada,
                    .prompt_ada = pada,
                    .cos = exe.cos_v,
                    .sin = exe.sin_v,
                    .tau = tau_b,
                },
                .outputs = .{ .hidden = &next },
                .opts = .{ .wait = false },
            });
            hid.deinit();
            hid = next;
        }
        var vel: zml.Buffer = undefined;
        fin.run(io, .{
            .inputs = .{ .hidden = hid, .embedded = emb },
            .outputs = .{ .latent = &vel },
            .opts = .{ .wait = false },
        });
        hid.deinit();
        defer vel.deinit();
        var sig_b = try weights.fromItems(io, platform, .init(.{}, .f32), &[_]f32{sigma});
        defer sig_b.deinit();
        var sign_b = try weights.fromItems(io, platform, .init(.{}, .f32), &[_]f32{sigma_next});
        defer sign_b.deinit();
        var nxt: zml.Buffer = undefined;
        ar.run(io, .{
            .inputs = .{ .x = x, .v = vel, .sigma = sig_b, .sigma_next = sign_b },
            .outputs = .{ .x = &nxt },
            .opts = .{ .wait = step + 1 == euler.evals },
        });
        x.deinit();
        x = nxt;
    }

    if (job) |j| j.set(.taehv, 1, 1);
    const nchw = try taehv.decodeLatentWith(allocator, io, platform, &exe.tae, &tae_run, &x, exe.time, exe.height, exe.width);
    return .{
        .nchw = nchw,
        .frames = taehv.outFrames(exe.time),
        .height = exe.height * 32,
        .width = exe.width * 32,
    };
}

pub fn remux(
    allocator: std.mem.Allocator,
    io: std.Io,
    out_path: []const u8,
    rgb_nchw: []const f32,
    frames: u32,
    height: u32,
    width: u32,
    audio: []const f32,
) !void {
    const dest = media.Output.parse(out_path);
    if (!dest.isCwd()) try std.Io.Dir.cwd().createDirPath(io, dest.dir);
    var out_dir: std.Io.Dir = if (dest.isCwd()) .cwd() else try std.Io.Dir.cwd().openDir(io, dest.dir, .{});
    defer if (!dest.isCwd()) out_dir.close(io);
    const pcm = try media.f32ToS16(allocator, if (audio.len != 0) audio else &[_]f32{ 0, 0 });
    defer allocator.free(pcm);
    _ = try media.writeGeneratedVideo(
        allocator,
        io,
        out_dir,
        dest.dir,
        dest.mp4_name,
        rgb_nchw,
        frames,
        height,
        width,
        pcm,
        config.audio_sample_rate,
    );
}

const EulerIn = struct { x: zml.Tensor, v: zml.Tensor, sigma: zml.Tensor, sigma_next: zml.Tensor };
const EulerOut = struct { x: zml.Tensor };
fn applyEuler(input: EulerIn) EulerOut {
    const dt = input.sigma_next.sub(input.sigma);
    return .{ .x = input.x.add(input.v.mul(dt.broad(input.v.shape()))) };
}

fn mixConstStart(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    latent: *zml.Buffer,
    seed: u64,
    sigma: f32,
) !zml.Buffer {
    const sl = try latent.toSliceAlloc(allocator, io);
    defer sl.free(allocator);
    const clean = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, sl.data())));
    const mixed = try allocator.alloc(f32, clean.len);
    defer allocator.free(mixed);
    var gen = noise.Generator.init(seed);
    noise.randn(&gen, mixed);
    euler.mixConstInPlace(sigma, clean, mixed);
    return weights.fromItems(io, platform, latent.shape(), mixed);
}

fn pickTae(donors: []const *taehv.Compiled, time: i64, h: u32, w: u32) struct {
    full: ?*const taehv.Compiled,
    bufs: ?*const taehv.Compiled,
} {
    const bufs: ?*const taehv.Compiled = if (donors.len != 0) donors[0] else null;
    for (donors) |d| {
        if (d.matches(time, h, w)) return .{ .full = d, .bufs = d };
    }
    return .{ .full = null, .bufs = bufs };
}

fn runConnector(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    conn: *const connector.Compiled,
    proj: []const f32,
) !zml.Buffer {
    const keep = gemma4.keep_tokens;
    const vdim: usize = @intCast(connector.dim);
    const pdim: usize = @intCast(gemma4.proj_dim);
    const video_c = try allocator.alloc(f32, keep * vdim);
    defer allocator.free(video_c);
    var i: usize = 0;
    while (i < keep) : (i += 1) {
        @memcpy(video_c[i * vdim ..][0..vdim], proj[i * pdim ..][0..vdim]);
    }
    var text_b = try weights.fromF32(allocator, io, platform, .init(.{ .n = 1, .s = keep, .d = connector.dim }, .f32), video_c);
    defer text_b.deinit();
    var pad_run = try zml.FnExe(connector.Pad.forward).Runner(.{.model}).init(&conn.pad, allocator, .{ .model = conn.pad_bufs });
    defer pad_run.deinit(allocator);
    var ctx: zml.Buffer = undefined;
    pad_run.run(io, .{ .inputs = .{ .text = text_b }, .outputs = .{ .hidden = &ctx }, .opts = .{ .wait = true } });
    const cos_c = try allocator.alloc(f32, @as(usize, connector.min_tokens) * @as(usize, @intCast(connector.heads * connector.head_dim)));
    defer allocator.free(cos_c);
    const sin_c = try allocator.alloc(f32, cos_c.len);
    defer allocator.free(sin_c);
    ltx_rope.fillConnector(cos_c, sin_c, connector.min_tokens);
    const rope_sh = zml.Shape.init(.{ .s = connector.min_tokens, .h = connector.heads, .hd = connector.head_dim }, .bf16).withPartitioning(.{ .h = .model });
    const model_shard = platform.shardings.get("model") orelse zml.Sharding.replicated;
    var cos_cb = try weights.fromF32Sharded(allocator, io, platform, rope_sh, model_shard, cos_c);
    defer cos_cb.deinit();
    var sin_cb = try weights.fromF32Sharded(allocator, io, platform, rope_sh, model_shard, sin_c);
    defer sin_cb.deinit();
    var bi: u32 = 0;
    while (bi < conn.n) : (bi += 1) {
        var br = try zml.FnExe(connector.Block.forward).Runner(.{.layer}).init(&conn.block, allocator, .{ .layer = conn.blocks[bi] });
        defer br.deinit(allocator);
        var next: zml.Buffer = undefined;
        br.run(io, .{
            .inputs = .{ .hidden = ctx, .cos = cos_cb, .sin = sin_cb },
            .outputs = .{ .hidden = &next },
            .opts = .{ .wait = true },
        });
        ctx.deinit();
        ctx = next;
    }
    var fin_c = try zml.FnExe(connector.Finish.forward).Runner(.{}).init(&conn.finish, allocator, .{});
    defer fin_c.deinit(allocator);
    var ctx_n: zml.Buffer = undefined;
    fin_c.run(io, .{ .inputs = .{ .hidden = ctx }, .outputs = .{ .hidden = &ctx_n }, .opts = .{ .wait = true } });
    ctx.deinit();
    return ctx_n;
}
