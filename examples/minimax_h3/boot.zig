const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const config = @import("recipe/config.zig");
const lora = @import("recipe/lora.zig");
const memory = @import("recipe/memory.zig");
const pipeline = @import("draft/pipeline.zig");
const policy = @import("recipe/policy.zig");
const repo = @import("serve/repo.zig");
const refine = @import("refine/run.zig");
const load = @import("refine/load.zig");
const serve = @import("serve/http.zig");
const session = @import("draft/session.zig");
const sol_attn = @import("refine/sol_attn.zig");
const sharding = @import("recipe/shard.zig");
const sku = @import("recipe/sku.zig");
const taeh3 = @import("draft/taeh3.zig");
const taehv = @import("refine/taehv.zig");
const weights = @import("recipe/weights.zig");

const log = std.log.scoped(.minimax_h3);

// =============================================================================
// boot.zig — open GPUs, compile SKUs, hand the HTTP server a Runtime
//
// Flow: page (compiling) → VFS → peek heads → pin CUDA to TP → load weights
// → compile first SKU, then remaining SKUs one at a time → /generate.
// =============================================================================

const Args = struct {
    model: []const u8 = sku.default_model,
    port: u16 = 8080,
    devices: u32 = 0,
    dit: []const u8 = "",
    lora: []const u8 = "",
    taeh3: []const u8 = "",
    attn: []const u8 = "",

    pub const help =
        \\ Compile MiniMax-H3 Super once, then serve.
        \\ Tensor-parallel degree is the largest divisor of the attention-head gcd
        \\ that fits the visible GPU count (CUDA_VISIBLE_DEVICES, or --devices if
        \\ unset). Official H3 Super gcd is 8, so 1–8 GPUs map to 1/2/4/8-way TP;
        \\ extra GPUs stay idle. Leftover counts (3, 5–7, 9+) drop to the next
        \\ even split rather than an uneven head partition.
        \\
        \\   --model=<path>   Repository (default hf://MiniMaxAI/MiniMax-H3)
        \\   --port=<n>       HTTP port (default 8080)
        \\   --devices=<n>    Cap visible GPUs (default: all already visible)
        \\   --dit=<path>     Fused Turbo DiT overlay
        \\   --lora=<path>    Turbo LoRA if the fused overlay is missing
        \\   --taeh3=<path>   TAEH3 weights
        \\   --attn=auto|fa2|sdpa
        \\
    ;
};

fn reject(err: anyerror, comptime fmt: []const u8, args: anytype) anyerror {
    log.err(fmt, args);
    return err;
}

fn rejectUser(err: anyerror) anyerror {
    return switch (err) {
        error.TransformerMissing => reject(err, "transformer weights not found", .{}),
        error.EncoderMissing => reject(err, "text_encoder not found", .{}),
        error.VaeMissing => reject(err, "audio_vae not found", .{}),
        error.VaeSchemaMismatch => reject(err, "VAE weight names not recognized", .{}),
        error.UnsupportedCheckpoint => reject(err, "unsupported checkpoint", .{}),
        error.MissingTokenizer => reject(err, "tokenizer.json not found", .{}),
        error.MemoryPlanUnsafe => reject(err, "does not fit device memory", .{}),
        error.LoraName, error.LoraTensorMissing, error.UnsupportedLoraDtype => reject(err, "could not load Turbo LoRA", .{}),
        error.InvalidAttn => reject(err, "--attn must be auto, fa2, fa3, or sdpa", .{}),
        error.GemmaMissing => reject(err, "Stage 2 Gemma tokenizer or weights missing", .{}),
        error.LtxDitMissing => reject(err, "LTX DiT weights missing", .{}),
        error.LtxVaeMissing => reject(err, "LTX VAE weights missing", .{}),
        error.LtxUpsamplerMissing => reject(err, "LTX spatial upsampler weights missing", .{}),
        error.TaehvMissing => reject(err, "TAEHV weights missing", .{}),
        error.HandoffMismatch => reject(err, "GPU handoff failed CPU gate", .{}),
        else => err,
    };
}

pub fn run(init: std.process.Init) !void {
    const allocator = init.gpa;
    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |cwd| {
        var dir = try std.Io.Dir.openDirAbsolute(init.io, cwd, .{});
        defer dir.close(init.io);
        try std.process.setCurrentDir(init.io, dir);
    }

    const args = stdx.flags.parse(init.minimal.args, Args);
    if (args.model.len == 0) return reject(error.IntentEmpty, "--model is required", .{});

    var vfs_file: zml.io.VFS.File = .init(allocator, init.io, .{});
    defer vfs_file.deinit();
    var http_client: std.http.Client = .{ .allocator = allocator, .io = init.io };
    defer http_client.deinit();
    try http_client.initDefaultProxies(allocator, init.environ_map);
    var hf_vfs: zml.io.VFS.HF = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer hf_vfs.deinit();
    var s3_vfs: zml.io.VFS.S3 = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer s3_vfs.deinit();
    var gcs_vfs: zml.io.VFS.GCS = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer gcs_vfs.deinit();
    var https_vfs: zml.io.VFS.HTTP = try .init(allocator, init.io, &http_client, .https);
    defer https_vfs.deinit();
    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();
    try vfs.register("file", vfs_file.io());
    try vfs.register("gs", gcs_vfs.io());
    try vfs.register("hf", hf_vfs.io());
    try vfs.register("https", https_vfs.io());
    try vfs.register("s3", s3_vfs.io());
    const io = vfs.io();

    sku.applyCompileCache(io);
    sku.applyXlaAccelFlags();
    if (std.c.getenv("ZML_AUTOTUNE_CACHE_DIR")) |path| {
        log.info("xla autotune cache {s}", .{std.mem.span(path)});
    }

    var app: serve.App = .{
        .allocator = allocator,
        .io = io,
        .port = args.port,
    };
    app.setCompile("Starting", 2);
    var site = try io.concurrent(serve.run, .{&app});
    var serving = false;
    errdefer if (!serving) app.setFailed("compile failed");
    const model_repo = try zml.safetensors.resolveModelRepo(io, args.model);
    const heads = repo.peekHeadCounts(allocator, io, model_repo, .{
        .model = args.model,
        .dit = args.dit,
    }) catch |err| return rejectUser(err);
    const pinned = sku.narrowVisible(init.io, heads, args.devices);
    if (pinned != 0) {
        sharding.prepareDeviceCap(pinned);
    } else if (args.devices != 0) {
        sharding.prepareDeviceCap(args.devices);
    }
    sharding.preparePhysicalMesh(heads);

    app.setCompile("Opening GPUs", 8);
    const platform: *zml.Platform = try .auto(allocator, io, .{
        .physical_mesh = .{ .custom = sharding.physicalMesh },
        // Equal BFC arenas on every mesh GPU. `preallocate=false` was leaving
        // compile high-water on the primary device (~2× nvidia-smi skew).
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = true, .memory_fraction = 0.85 } } },
    });
    defer platform.deinit(allocator, io);
    try sol_attn.register(platform);
    log.info("\n{f}", .{platform.fmtVerbose()});

    const shardings: sharding.Shardings = try .init(platform, heads);
    const tp: u32 = @intCast(shardings.model.numPartitionsForLogicalAxis(.model));
    const device_bytes = config.minDeviceBytes(platform);
    const sku_n: u32 = @intCast(sku.enabledCount());
    log.info(
        "skus={d}/{d} tp={d} devices={d} {d}GiB (transformer TP only)",
        .{
            sku_n,
            sku.skus.len,
            tp,
            platform.devices.len,
            device_bytes / (1024 * 1024 * 1024),
        },
    );
    app.devices = @intCast(platform.devices.len);

    const fused_ready = sku.fusedDitPresent(init.io, sku.default_fused_dit);
    const paths: repo.Open = .{
        .model = args.model,
        .dit = sku.resolvedDit(args.dit, fused_ready),
    };
    if (paths.dit.len != 0) log.info("dit overlay {s}", .{paths.dit});
    app.setCompile("Loading weights", 16);
    var models = repo.Bundle.open(allocator, io, model_repo, shardings, paths) catch |err| return rejectUser(err);
    defer models.deinit(allocator, io);

    var lora_bundle: ?lora.Bundle = null;
    defer if (lora_bundle) |*b| b.deinit();
    if (sku.useRuntimeLora(args.dit, args.lora, fused_ready)) {
        const lora_path = if (args.lora.len != 0) args.lora else sku.default_lora_path;
        lora_bundle = lora.load(allocator, io, lora_path, sku.lora_strength) catch |err| return rejectUser(err);
        models.dit.lora = &lora_bundle.?;
    } else {
        log.info("fused dit (skip runtime LoRA)", .{});
    }

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    defer progress.end();

    var tokenizer = repo.loadTokenizer(allocator, io, models.task, model_repo, &progress) catch |err| return rejectUser(err);
    defer tokenizer.deinit();

    const text_len = serve.text_len;
    const all = shardings.all();
    const taeh3_path = if (args.taeh3.len != 0) args.taeh3 else load.firstExisting(io, &sku.taeh3_paths) orelse sku.default_taeh3_path;
    const core0 = models.dit.inner.blocks[0].corePart();
    const dit_dt = models.dit.inner.blocks[0].norm1.weight.dtype();
    const text_prep = models.dit.inner.textPrep();
    const patch_embed = models.dit.inner.patchEmbed();
    const finish_core = models.dit.inner.finishCore();
    var encoder_weight_bytes = weights.modelBytes(&models.enc.inner.embed_tokens);
    for (models.enc.inner.layers) |*layer| encoder_weight_bytes +|= weights.modelBytes(layer);
    const flash = (policy.parseAttnOverride(args.attn) catch |err| return rejectUser(err)) orelse zml.attention.Backend.auto(platform);
    const lane_plan = LanePlan{
        .flash = flash,
        .device_bytes = device_bytes,
        .tp = tp,
        .encoder_weight_bytes = encoder_weight_bytes,
        .dit_dt = dit_dt,
        .block_core_bytes = weights.modelBytes(&core0),
        .fixed_denoise_weight_bytes = weights.modelBytes(&text_prep) +|
            weights.modelBytes(&patch_embed) +| weights.modelBytes(&finish_core),
        .audio_vae_weight_bytes = weights.modelBytes(&models.audio.inner),
        .refine_weight_bytes = memory.refineWeightBytes(tp),
    };

    const boot_resident = try minResidentBlocks(allocator, &models, lane_plan, text_len, platform);
    log.info("boot resident DiT cores={d} tp={d}", .{ boot_resident, tp });

    app.setCompile("Loading resident weights", 20);
    var warm = try session.loadWarm(allocator, io, platform, &models, &all, &progress, boot_resident);
    defer warm.deinit(allocator);

    var stores: refine.WeightStores = .{};
    defer stores.deinit();
    var taeh3_loaded = try taeh3.open(allocator, io, taeh3_path);
    var taeh3_live = true;
    defer if (taeh3_live) taeh3_loaded.deinit(allocator);
    var share = CompileShare{
        .stores = &stores,
        .taeh3_store = &taeh3_loaded,
    };

    var built: [sku.skus.len]serve.Lane = undefined;
    var lane_n: usize = 0;
    var resident_blocks: u32 = boot_resident;
    var transferred = false;
    defer if (!transferred) {
        var i = lane_n;
        while (i > 0) {
            i -= 1;
            built[i].deinit(allocator);
        }
    };

    var spec_buf: [sku.skus.len]sku.Sku = undefined;
    const specs = sku.collectEnabled(&spec_buf);
    const ctx = LaneCtx{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .models = &models,
        .shardings = shardings,
        .all = &all,
        .progress = &progress,
        .text_len = text_len,
        .plan = lane_plan,
        .warm = &warm,
        .share = &share,
        .app = &app,
    };

    var next: usize = 0;
    while (next < specs.len) {
        const spec = specs[next];
        next += 1;
        var name_buf: [32]u8 = undefined;
        app.setCompile(skuCompileName(spec, &name_buf), skuPct(0, 0));
        app.setCompileSku(spec.id, .run);
        const lane = compileFrom(&ctx, spec, 0) catch |err| {
            try skipSku(&app, spec, err);
            continue;
        };
        takeLane(&built, &lane_n, &share, &app, spec, lane, &resident_blocks);
        break;
    }
    if (lane_n == 0) return rejectUser(error.MemoryPlanUnsafe);

    const extras = specs[next..];
    for (extras, 0..) |spec, ei| {
        var name_buf: [32]u8 = undefined;
        app.setCompile(skuCompileName(spec, &name_buf), skuPct(ei + 1, 0));
        app.setCompileSku(spec.id, .run);
        const lane = compileFrom(&ctx, spec, ei + 1) catch |err| {
            try skipSku(&app, spec, err);
            continue;
        };
        takeLane(&built, &lane_n, &share, &app, spec, lane, &resident_blocks);
    }
    if (extras.len != 0) app.setCompile("SKU matrix ready", skuPct(lane_n, 100));
    stores.deinit();
    taeh3_loaded.deinit(allocator);
    taeh3_live = false;

    const lanes = try allocator.alloc(serve.Lane, lane_n);
    @memcpy(lanes, built[0..lane_n]);
    transferred = true;
    defer {
        var i = lanes.len;
        while (i > 0) {
            i -= 1;
            lanes[i].deinit(allocator);
        }
        allocator.free(lanes);
    }

    var max_sol_tokens: u32 = 0;
    for (lanes) |lane| {
        if (lane.ltx.dit.block_sol != null) max_sol_tokens = @max(max_sol_tokens, lane.ltx.dit.tokens);
    }
    if (max_sol_tokens != 0) {
        sol_attn.reserveWorkspace(max_sol_tokens, @intCast(platform.devices.len)) catch |err| {
            log.warn("sol-attn workspace reserve failed ({s})", .{@errorName(err)});
        };
    }

    app.setCompile("Warming text encoder", 96);
    const boot_prompt = "A cinematic wide shot of waves at dusk.";
    try refine.refreshContext(allocator, io, platform, lanes[0].ltx, boot_prompt);
    var runtime: serve.Runtime = .{
        .platform = platform,
        .models = &models,
        .warm = &warm,
        .shardings = &all,
        .tokenizer = &tokenizer,
        .progress = &progress,
        .resident_blocks = resident_blocks,
        .lanes = lanes,
        .last_prompt = try allocator.dupe(u8, boot_prompt),
    };
    defer if (runtime.last_prompt.len != 0) allocator.free(runtime.last_prompt);
    app.setReady(&runtime, @intCast(platform.devices.len));
    serving = true;
    site.await(io) catch {};
}

const CompileShare = struct {
    ltx: ?*refine.Compiled = null,
    taeh3: ?*taeh3.Compiled = null,
    h3: ?*pipeline.Compiled = null,
    bake: ?*session.Bake = null,
    taes: [sku.skus.len]*taehv.Compiled = undefined,
    tae_n: usize = 0,
    stores: *refine.WeightStores,
    taeh3_store: *taeh3.Loaded,

    fn populated(self: *const CompileShare) bool {
        return self.ltx != null;
    }

    fn adopt(self: *CompileShare, lane: *serve.Lane) void {
        if (self.ltx == null) self.ltx = lane.ltx;
        if (self.h3 == null) self.h3 = lane.h3;
        if (self.taeh3 == null) self.taeh3 = lane.taeh3;
        if (self.bake == null) self.bake = lane.bake;
        self.addTae(&lane.ltx.tae);
    }

    fn addTae(self: *CompileShare, tae: *taehv.Compiled) void {
        if (self.tae_n >= self.taes.len) return;
        self.taes[self.tae_n] = tae;
        self.tae_n += 1;
    }

    fn taeDonors(self: *CompileShare) []const *taehv.Compiled {
        return self.taes[0..self.tae_n];
    }
};

const LanePlan = struct {
    flash: zml.attention.Backend,
    device_bytes: u64,
    tp: u32,
    encoder_weight_bytes: u64,
    dit_dt: zml.DataType,
    block_core_bytes: u64,
    fixed_denoise_weight_bytes: u64,
    audio_vae_weight_bytes: u64,
    refine_weight_bytes: u64,
};

const LaneCtx = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    models: *repo.Bundle,
    shardings: sharding.Shardings,
    all: []const zml.Sharding,
    progress: *std.Progress.Node,
    text_len: u32,
    plan: LanePlan,
    warm: *session.Warm,
    share: *CompileShare,
    app: *serve.App,
};

fn compileFrom(ctx: *const LaneCtx, spec: sku.Sku, index: usize) !serve.Lane {
    return compileLane(
        ctx.allocator,
        ctx.io,
        ctx.platform,
        ctx.models,
        ctx.shardings,
        ctx.all,
        ctx.progress,
        spec,
        ctx.text_len,
        ctx.plan,
        ctx.warm,
        ctx.share,
        ctx.app,
        index,
    );
}

fn skipSku(app: *serve.App, spec: sku.Sku, err: anyerror) !void {
    app.setCompileSku(spec.id, .skip);
    if (sku.isRequired(spec)) return rejectUser(err);
    log.warn("skip {s}: {s}", .{ spec.id, @errorName(err) });
}

fn takeLane(
    built: *[sku.skus.len]serve.Lane,
    n: *usize,
    share: *CompileShare,
    app: *serve.App,
    spec: sku.Sku,
    lane: serve.Lane,
    resident: *u32,
) void {
    built[n.*] = lane;
    share.adopt(&built[n.*]);
    resident.* = @min(resident.*, lane.resident_blocks);
    n.* += 1;
    app.setCompileSku(spec.id, .done);
    log.info("sku ready {s} {d}x{d} {d}s", .{ spec.id, spec.target_w, spec.target_h, sku.seconds(spec) });
}

fn skuCompileName(spec: sku.Sku, buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "Compiling {d}s {s}", .{
        sku.seconds(spec),
        sku.familyLabel(spec),
    }) catch spec.id;
}

fn skuPct(compiled_i: usize, inner: u32) u32 {
    const n = @max(sku.enabledCount(), 1);
    const start = 20 + (72 * compiled_i) / n;
    const end = 20 + (72 * (compiled_i + 1)) / n;
    return @intCast(start + (end - start) * @min(inner, 100) / 100);
}

fn compileLane(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    models: *repo.Bundle,
    shardings: sharding.Shardings,
    all: []const zml.Sharding,
    progress: *std.Progress.Node,
    spec: sku.Sku,
    text_len: u32,
    plan: LanePlan,
    warm: *session.Warm,
    share: *CompileShare,
    app: *serve.App,
    compiled_i: usize,
) !serve.Lane {
    const rep_arr = shardings.rep();
    const rep: []const zml.Sharding = &rep_arr;
    const frames = (config.resolveFrames(spec.duration_s, 0) catch |err| return rejectUser(err)).aligned;
    const opts = draftOpts(spec, frames);
    const geo = pipeline.Geometry.init(opts, models.dit.cfg);
    const packed_run = try allocator.create(pipeline.Packed);
    errdefer allocator.destroy(packed_run);
    packed_run.* = try pipeline.pack(allocator, opts, geo, text_len);
    errdefer packed_run.deinit(allocator);
    log.info(
        "sku {s} layout {d}x{d} {d} frames latents {d}x{d}x{d} seq={d} -> {d}x{d}",
        .{
            spec.id,
            geo.pixel_w,
            geo.pixel_h,
            geo.frames,
            geo.latent_t,
            geo.latent_h,
            geo.latent_w,
            packed_run.layout.seqLen(),
            spec.target_w,
            spec.target_h,
        },
    );

    const mem = memoryFor(models, plan, platform, geo, packed_run);
    if (!mem.safe) {
        log.err("{s} {s} (peak {d} MiB)", .{ spec.id, mem.reason, mem.peak_bytes / (1024 * 1024) });
        return error.MemoryPlanUnsafe;
    }
    log.info(
        "sku {s} memory peak={d}MiB resident={d} group={d} attn={s} refine={d}MiB/gpu",
        .{
            spec.id,
            mem.peak_bytes / (1024 * 1024),
            mem.resident_blocks,
            mem.group_size,
            @tagName(mem.attention),
            mem.refine_weight_bytes / (1024 * 1024),
        },
    );

    var name_buf: [40]u8 = undefined;
    app.setCompile(skuCompileName(spec, &name_buf), skuPct(compiled_i, 15));
    const graphs = try compileGraphs(
        allocator,
        io,
        platform,
        models,
        shardings,
        all,
        progress,
        spec,
        geo,
        text_len,
        packed_run.layout.seqLen(),
        .{
            .attention = mem.attention,
            .group_size = mem.group_size,
            .steps = @intCast(packed_run.schedules.video.stepCount()),
        },
        share,
        rep,
        compiled_i,
        app,
        &name_buf,
    );
    errdefer graphs.deinit(allocator);

    app.setCompile(skuCompileName(spec, &name_buf), skuPct(compiled_i, 85));
    const bake = try allocator.create(session.Bake);
    errdefer allocator.destroy(bake);
    bake.* = .{};
    errdefer bake.deinit(allocator);
    try session.bakeDenoise(
        allocator,
        io,
        platform,
        models,
        graphs.h3,
        all,
        progress,
        warm,
        bake,
        geo,
        packed_run.layout,
        packed_run.schedules,
        share.bake,
    );

    return .{
        .id = spec.id,
        .duration_s = spec.duration_s,
        .target_w = spec.target_w,
        .target_h = spec.target_h,
        .hd = sku.isHd(spec),
        .geo = geo,
        .packed_run = packed_run,
        .h3 = graphs.h3,
        .taeh3 = graphs.taeh3,
        .ltx = graphs.ltx,
        .bake = bake,
        .resident_blocks = mem.resident_blocks,
    };
}

const Graphs = struct {
    h3: *pipeline.Compiled,
    taeh3: *taeh3.Compiled,
    ltx: *refine.Compiled,

    fn deinit(self: Graphs, allocator: std.mem.Allocator) void {
        self.h3.deinit();
        allocator.destroy(self.h3);
        self.taeh3.deinit();
        allocator.destroy(self.taeh3);
        self.ltx.deinit();
        allocator.destroy(self.ltx);
    }
};

fn compileH3(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    models: *repo.Bundle,
    geo: pipeline.Geometry,
    text_len: u32,
    seq_len: u32,
    compile_policy: pipeline.CompilePolicy,
    shardings: sharding.Shardings,
    progress: *std.Progress.Node,
    shared_h3: ?*pipeline.Compiled,
) !*pipeline.Compiled {
    const h3 = try allocator.create(pipeline.Compiled);
    errdefer allocator.destroy(h3);
    h3.* = try pipeline.compile(
        allocator,
        io,
        platform,
        models.dit.inner,
        models.enc.inner,
        geo,
        text_len,
        seq_len,
        compile_policy,
        shardings,
        progress,
        shared_h3,
    );
    return h3;
}

fn compileTaeh3(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    loaded: *taeh3.Loaded,
    time: u32,
    latent_h: u32,
    latent_w: u32,
    shardings: []const zml.Sharding,
    progress: *std.Progress.Node,
    reuse: ?*taeh3.Compiled,
) !*taeh3.Compiled {
    const exe = try allocator.create(taeh3.Compiled);
    errdefer allocator.destroy(exe);
    exe.* = try taeh3.compile(
        allocator,
        io,
        platform,
        loaded,
        1,
        time,
        latent_h,
        latent_w,
        shardings,
        progress,
        reuse,
    );
    return exe;
}

fn compileGraphs(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    models: *repo.Bundle,
    shardings: sharding.Shardings,
    all: []const zml.Sharding,
    progress: *std.Progress.Node,
    spec: sku.Sku,
    geo: pipeline.Geometry,
    text_len: u32,
    seq_len: u32,
    compile_policy: pipeline.CompilePolicy,
    share: *CompileShare,
    rep: []const zml.Sharding,
    compiled_i: usize,
    app: *serve.App,
    name_buf: []u8,
) !Graphs {
    const h3_args = .{
        allocator,
        io,
        platform,
        models,
        geo,
        text_len,
        seq_len,
        compile_policy,
        shardings,
        progress,
        share.h3,
    };
    const tae_args = .{
        allocator,
        io,
        platform,
        share.taeh3_store,
        geo.latent_t,
        geo.latent_h,
        geo.latent_w,
        rep,
        progress,
        share.taeh3,
    };
    const ltx_args = .{
        allocator,
        io,
        platform,
        all,
        rep,
        progress,
        @as(u32, @intCast(taeh3.timeOut(geo.latent_t))),
        taeh3.pixelExtent(geo.latent_h),
        taeh3.pixelExtent(geo.latent_w),
        geo.frames,
        spec.target_w,
        spec.target_h,
        share.ltx,
        share.taeDonors(),
        share.stores,
    };

    if (share.populated()) {
        var h3_f = try io.concurrent(compileH3, h3_args);
        var h3_live = true;
        errdefer if (h3_live) {
            if (h3_f.cancel(io)) |p| {
                p.deinit();
                allocator.destroy(p);
            } else |_| {}
        };
        var tae_f = try io.concurrent(compileTaeh3, tae_args);
        var tae_live = true;
        errdefer if (tae_live) {
            if (tae_f.cancel(io)) |p| {
                p.deinit();
                allocator.destroy(p);
            } else |_| {}
        };
        var ltx_f = try io.concurrent(refine.compile, ltx_args);
        var ltx_live = true;
        errdefer if (ltx_live) {
            if (ltx_f.cancel(io)) |p| {
                p.deinit();
                allocator.destroy(p);
            } else |_| {}
        };
        const h3 = try h3_f.await(io);
        h3_live = false;
        errdefer {
            h3.deinit();
            allocator.destroy(h3);
        }
        const taeh3_exe = try tae_f.await(io);
        tae_live = false;
        errdefer {
            taeh3_exe.deinit();
            allocator.destroy(taeh3_exe);
        }
        const ltx = try ltx_f.await(io);
        ltx_live = false;
        return .{ .h3 = h3, .taeh3 = taeh3_exe, .ltx = ltx };
    }

    app.setCompile(skuCompileName(spec, name_buf), skuPct(compiled_i, 15));
    const h3 = try @call(.auto, compileH3, h3_args);
    errdefer {
        h3.deinit();
        allocator.destroy(h3);
    }
    app.setCompile(skuCompileName(spec, name_buf), skuPct(compiled_i, 45));
    const taeh3_exe = try @call(.auto, compileTaeh3, tae_args);
    errdefer {
        taeh3_exe.deinit();
        allocator.destroy(taeh3_exe);
    }
    app.setCompile(skuCompileName(spec, name_buf), skuPct(compiled_i, 55));
    const ltx = try @call(.auto, refine.compile, ltx_args);
    return .{ .h3 = h3, .taeh3 = taeh3_exe, .ltx = ltx };
}

fn minResidentBlocks(
    allocator: std.mem.Allocator,
    models: *repo.Bundle,
    plan: LanePlan,
    text_len: u32,
    platform: *const zml.Platform,
) !u32 {
    var min_blocks: u32 = std.math.maxInt(u32);
    var spec_buf: [sku.skus.len]sku.Sku = undefined;
    for (sku.collectEnabled(&spec_buf)) |spec| {
        const frames = (config.resolveFrames(spec.duration_s, 0) catch continue).aligned;
        const opts = draftOpts(spec, frames);
        const geo = pipeline.Geometry.init(opts, models.dit.cfg);
        var packed_run = pipeline.pack(allocator, opts, geo, text_len) catch continue;
        defer packed_run.deinit(allocator);
        const mem = memoryFor(models, plan, platform, geo, &packed_run);
        min_blocks = @min(min_blocks, mem.resident_blocks);
    }
    return if (min_blocks == std.math.maxInt(u32)) 0 else min_blocks;
}

fn draftOpts(spec: sku.Sku, frames: u32) pipeline.Options {
    return .{
        .duration_s = spec.duration_s,
        .width = spec.draft_w,
        .height = spec.draft_h,
        .frames = frames,
        .steps = sku.schedule_points,
        .video_shift = sku.turbo_video_shift,
        .audio_shift = sku.turbo_audio_shift,
    };
}

fn memoryFor(
    models: *repo.Bundle,
    plan: LanePlan,
    platform: *const zml.Platform,
    geo: pipeline.Geometry,
    packed_run: *const pipeline.Packed,
) memory.Plan {
    return memory.plan(.{
        .geo = .init(geo),
        .layout = packed_run.layout,
        .hidden = models.dit.cfg.hidden_size,
        .steps = @intCast(packed_run.schedules.video.stepCount()),
        .device_bytes = plan.device_bytes,
        .tp = plan.tp,
        .heads = models.dit.cfg.num_attention_heads,
        .head_dim = models.dit.cfg.attention_head_dim,
        .layers = @intCast(models.dit.cfg.num_layers),
        .dtype = plan.dit_dt,
        .target = platform.target,
        .block_core_bytes = plan.block_core_bytes,
        .flash = plan.flash,
        .fixed_denoise_weight_bytes = plan.fixed_denoise_weight_bytes,
        .encoder_weight_bytes = plan.encoder_weight_bytes,
        .audio_vae_weight_bytes = plan.audio_vae_weight_bytes,
        .refine_weight_bytes = plan.refine_weight_bytes,
    });
}
