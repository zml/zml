const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const generate = @import("generate.zig");

const conditions = generate.conditions;
const config = @import("model.zig").config;
const ir = generate.ir;
const memory = generate.memory;
const pipeline = generate.pipeline;
const repo = generate.ckpt;
const request = generate.request;
const session = generate.session;
const sharding = generate.sharding;
const telemetry = generate.telemetry;
const weights = @import("model.zig").weights;

const log = std.log.scoped(.minimax_h3);

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,
    prompt: []const u8 = "A cinematic wide shot of waves at dusk.",
    image: []const u8 = "",
    last_image: []const u8 = "",
    refs: []const u8 = "",
    duration: f32 = 5.0,
    ratio: config.Aspect = .@"16:9",
    canvas: config.Canvas = .auto,
    seed: u64 = 0,
    out: []const u8 = "output",
    dit: []const u8 = "",
    encoder: []const u8 = "",
    vae_video: []const u8 = "",
    vae_audio: []const u8 = "",
    tokenizer: []const u8 = "",
    profile: bool = false,

    pub const help =
        \\ Use minimax_h3 --model=<path> [options]
        \\
        \\ Joint video+audio from MiniMax-H3. Attachments pick the task:
        \\   none → t2va    --image/--last-image → fl2va    --refs → ref2va
        \\
        \\ Options:
        \\   --model=<path>      Model repository, DiT file, or hf:// (required)
        \\   --prompt=<string>   Intent (default: cinematic waves at dusk)
        \\   --image=<path>      First frame
        \\   --last-image=<path> Last frame
        \\   --refs=<paths>      Comma-separated images, videos, audio
        \\   --duration=<sec>    4–15 (default: 5)
        \\   --ratio=<aspect>    21:9 | 16:9 | 4:3 | 1:1 | 3:4 | 9:16
        \\   --canvas=<size>     auto | tiny | preview | full (auto: 768p or 640×352)
        \\   --seed=<n>          RNG seed
        \\   --out=<path>        Directory or .mp4 (default: output/)
        \\   --dit=<path>        DiT override when a folder has more than one match
        \\   --encoder=<path>    Text-encoder override
        \\   --vae-video=<path>  Video VAE override
        \\   --vae-audio=<path>  Audio VAE override
        \\   --tokenizer=<path>  tokenizer.json, or a directory that contains it
        \\   --profile           PJRT/TraceMe around generation
        \\
    ;
};

fn reject(err: anyerror, comptime fmt: []const u8, args: anytype) anyerror {
    log.err(fmt, args);
    return err;
}

fn rejectUser(err: anyerror) anyerror {
    return switch (err) {
        error.UnknownAspect => reject(err, "unknown --ratio (21:9|16:9|4:3|1:1|3:4|9:16)", .{}),
        error.InvalidDuration => reject(err, "--duration must be 4–15", .{}),
        error.Ref2vaRejectsKeyframes => reject(err, "pass --image/--last-image or --refs, not both", .{}),
        error.FullCanvasTooLarge => reject(err, "--canvas=full needs >={d} GiB/device", .{config.full_canvas_min_device_bytes / (1024 * 1024 * 1024)}),
        error.ConditionedPreviewTooLarge => reject(err, "preview + media needs --canvas=tiny under {d} GiB", .{config.full_canvas_min_device_bytes / (1024 * 1024 * 1024)}),
        error.Ref2vaTransformerMissing => reject(err, "ref2va needs Ref2VA/transformer", .{}),
        error.TransformerMissing => reject(err, "transformer weights not found", .{}),
        error.AmbiguousDit => reject(err, "multiple DiT files match; pass --dit=<file>", .{}),
        error.AmbiguousEncoder => reject(err, "multiple encoder files match; pass --encoder=<file>", .{}),
        error.AmbiguousVae => reject(err, "multiple VAE files match; pass --vae-video/--vae-audio", .{}),
        error.EncoderMissing => reject(err, "text_encoder not found", .{}),
        error.VaeMissing => reject(err, "video_vae or audio_vae not found", .{}),
        error.VaeSchemaMismatch => reject(err, "VAE weight names not recognized", .{}),
        error.UnsupportedCheckpoint => reject(err, "unsupported checkpoint", .{}),
        error.MissingTokenizer => reject(err, "tokenizer.json not found", .{}),
        error.MemoryPlanUnsafe => reject(err, "does not fit device memory", .{}),
        error.TooManyRefs => reject(err, "too many --refs (max 12)", .{}),
        error.TooManyRefImages => reject(err, "too many reference images (max 9)", .{}),
        error.TooManyRefVideos => reject(err, "too many reference videos (max 3)", .{}),
        error.TooManyRefAudios => reject(err, "too many reference audios (max 3)", .{}),
        error.IntentEmpty => reject(err, "needs a non-empty --prompt", .{}),
        else => err,
    };
}

fn hasMedia(args: Args) bool {
    return args.image.len != 0 or args.last_image.len != 0 or args.refs.len != 0;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
        var working_dir = try std.Io.Dir.openDirAbsolute(init.io, build_working_directory, .{});
        defer working_dir.close(init.io);
        try std.process.setCurrentDir(init.io, working_dir);
    }

    const args = stdx.flags.parse(init.minimal.args, Args);
    config.checkDuration(args.duration) catch |err| return rejectUser(err);
    const refs = try request.refsFromComma(allocator, args.refs);
    defer request.freeRefs(allocator, refs, false);
    const variant = request.inferVariant(args.image, args.last_image, refs) catch |err| return rejectUser(err);
    const paths: repo.Open = .{
        .model = args.model,
        .dit = args.dit,
        .encoder = args.encoder,
        .vae_video = args.vae_video,
        .vae_audio = args.vae_audio,
        .tokenizer = args.tokenizer,
    };

    var vfs_file: zml.io.VFS.File = .init(allocator, init.io, .{});
    defer vfs_file.deinit();
    var http_client: std.http.Client = .{ .allocator = allocator, .io = init.io };
    defer http_client.deinit();
    var hf_vfs: zml.io.VFS.HF = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer hf_vfs.deinit();
    var s3_vfs: zml.io.VFS.S3 = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer s3_vfs.deinit();
    var gcs_vfs: zml.io.VFS.GCS = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer gcs_vfs.deinit();
    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();
    try vfs.register("file", vfs_file.io());
    try vfs.register("gs", gcs_vfs.io());
    try vfs.register("hf", hf_vfs.io());
    try vfs.register("s3", s3_vfs.io());

    const io = vfs.io();

    const ir_args = .{ allocator, ir.Request{
        .prompt = args.prompt,
        .variant = variant,
        .duration_s = args.duration,
        .aspect = @tagName(args.ratio),
        .image = args.image,
        .last_image = args.last_image,
        .refs = args.refs,
    } };
    var ir_fut: ?@TypeOf(try init.io.concurrent(ir.compile, ir_args)) = try init.io.concurrent(ir.compile, ir_args);
    errdefer cancelBrief(&ir_fut, init.io, allocator);

    const platform: *zml.Platform = try .auto(allocator, io, .{
        .cpu = .{ .device_count = 1 },
        .physical_mesh = .{ .custom = sharding.physicalMesh },
    });
    defer platform.deinit(allocator, io);
    log.info("\n{f}", .{platform.fmtVerbose()});

    const device_bytes = config.minDeviceBytes(platform);
    const canvas = config.canvasForTarget(platform.target, args.canvas, device_bytes);
    config.checkCanvas(args.canvas, variant, canvas.short_side, device_bytes) catch |err| return rejectUser(err);

    const shardings: sharding.Shardings = try .init(platform);
    const px = config.pixelSize(args.ratio, canvas.short_side);
    const frames = config.alignFrameCount(config.frameCount(args.duration));
    log.info(
        "run model={s} variant={s} canvas={s} {d}x{d} frames={d} steps={d} seed={d} target={s} shard={d} devices={d}",
        .{
            args.model,
            @tagName(variant),
            @tagName(args.canvas),
            px.w,
            px.h,
            frames,
            canvas.steps,
            args.seed,
            @tagName(platform.target),
            shardings.model.numPartitionsForLogicalAxis(.model),
            platform.devices.len,
        },
    );

    const model_repo = try zml.safetensors.resolveModelRepo(io, args.model);
    var models = repo.Bundle.open(allocator, io, model_repo, variant, shardings, paths) catch |err| return rejectUser(err);
    defer models.deinit(allocator, io);

    const opts: pipeline.Options = .{
        .variant = variant,
        .duration_s = args.duration,
        .aspect = args.ratio,
        .short_side = canvas.short_side,
        .steps = canvas.steps,
        .seed = args.seed,
    };
    const geo = pipeline.Geometry.init(opts, models.dit.cfg);

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    defer progress.end();

    const brief = blk: {
        const result = ir_fut.?.await(init.io) catch |err| {
            ir_fut = null;
            return rejectUser(err);
        };
        ir_fut = null;
        break :blk result;
    };
    defer brief.deinit(allocator);

    var tokenizer = repo.loadTokenizer(allocator, io, models.task, model_repo, paths, &progress) catch |err| return rejectUser(err);
    defer tokenizer.deinit();
    var tok_enc = try tokenizer.encoder();
    defer tok_enc.deinit();

    var encoded = if (hasMedia(args))
        try conditions.prepare(allocator, io, platform, &progress, &tok_enc, .{
            .variant = variant,
            .first_image = args.image,
            .last_image = args.last_image,
            .refs = refs,
            .prompt = brief.text,
            .geo = geo,
            .models = &models,
            .shardings = shardings,
        })
    else
        try conditions.tokenize(allocator, &tok_enc, brief.text);
    defer encoded.deinit(allocator);

    const geo_work = if (hasMedia(args))
        geo.withConditions(encoded.conds.target_video_offset, encoded.conds.target_audio_offset)
    else
        geo;
    const extras = encoded.extras();
    const text_len: u32 = @intCast(encoded.tokens.len);
    log.info("prompt tokens={d} refs={d} cond_video={d} cond_audio={d}", .{
        text_len,
        encoded.conds.references.len,
        encoded.conds.videos.len,
        encoded.conds.audios.len,
    });

    var packed_run = try pipeline.pack(
        allocator,
        opts,
        geo_work,
        text_len,
        encoded.tags,
        encoded.conds.videos,
        encoded.conds.audios,
        encoded.conds.references,
    );
    defer packed_run.deinit(allocator);
    pipeline.describe(opts, geo_work, packed_run.layout);

    const core0 = models.dit.inner.blocks[0].corePart();
    const dit_dt = models.dit.inner.blocks[0].norm1.weight.dtype();
    const tp: u32 = @intCast(shardings.model.numPartitionsForLogicalAxis(.model));
    const mem = memory.planWith(.{
        .geo = geo_work,
        .layout = packed_run.layout,
        .hidden = models.dit.cfg.hidden_size,
        .steps = @intCast(packed_run.schedules.video.stepCount()),
        .device_bytes = device_bytes,
        .tp = tp,
        .heads = models.dit.cfg.num_attention_heads,
        .head_dim = models.dit.cfg.attention_head_dim,
        .layers = @intCast(models.dit.cfg.num_layers),
        .dtype = dit_dt,
        .target = platform.target,
        .block_core_bytes = weights.modelBytes(&core0) / @max(1, tp),
        .devices = @intCast(platform.devices.len),
    });
    if (!mem.safe) {
        log.err("{s} (peak {d} MiB)", .{ mem.reason, mem.peak_bytes / (1024 * 1024) });
        return rejectUser(error.MemoryPlanUnsafe);
    }
    log.info(
        "memory peak={d}MiB act={d}MiB block={d}MiB scores={d}MiB fa2={d}MiB tables={d}MiB resident={d} group={d} tile_batch={d} attn={s}",
        .{
            mem.peak_bytes / (1024 * 1024),
            mem.activation_bytes / (1024 * 1024),
            mem.streamed_block_bytes / (1024 * 1024),
            mem.score_bytes / (1024 * 1024),
            mem.fa2_scratch_bytes / (1024 * 1024),
            mem.adaln_table_bytes / (1024 * 1024),
            mem.resident_blocks,
            mem.group_size,
            mem.tile_batch,
            @tagName(mem.attention),
        },
    );

    const compile_policy: pipeline.CompilePolicy = .{
        .attention = mem.attention,
        .group_size = mem.group_size,
        .steps = @intCast(packed_run.schedules.video.stepCount()),
        .hold_video = if (geo.video_patch_dim == 0) 0 else @intCast(@divExact(encoded.conds.video_patches.len, geo.video_patch_dim)),
        .hold_audio = if (geo.audio_dim == 0) 0 else @intCast(@divExact(encoded.conds.audio_patches.len, geo.audio_dim)),
        .vision_tokens = blk: {
            var n: u32 = 0;
            for (extras.vision_spans) |span| n += span.tokens;
            break :blk n;
        },
    };

    const all = shardings.all();
    const compile_args = .{
        allocator,
        io,
        platform,
        models.dit.inner,
        models.enc.inner,
        geo_work,
        text_len,
        packed_run.layout.seqLen(),
        compile_policy,
        shardings,
        &progress,
    };
    var dit_compile: ?@TypeOf(try io.concurrent(pipeline.compile, compile_args)) =
        try io.concurrent(pipeline.compile, compile_args);
    errdefer cancelCompiled(&dit_compile, io);

    var compiled_vae = try pipeline.compileVae(
        allocator,
        io,
        platform,
        models.visual.inner,
        models.audio.inner,
        geo_work,
        mem.tile_batch,
        shardings,
        &progress,
    );
    defer compiled_vae.deinit();

    var compiled = dit_compile.?.await(io) catch |err| {
        dit_compile = null;
        return err;
    };
    dit_compile = null;
    defer compiled.deinit();

    var sink = telemetry.Sink.init(args.profile);
    var gen_span = zml.tracer.span("h3.generate", .{ .root = true });
    defer gen_span.end();
    try session.generate(allocator, io, platform, &models, &compiled, &compiled_vae, &all, &progress, .{
        .opts = opts,
        .geo = geo_work,
        .target = geo,
        .tokens = encoded.tokens,
        .extras = extras,
        .layout = packed_run.layout,
        .schedules = packed_run.schedules,
        .cond = .{
            .videos = encoded.conds.videos,
            .video_patches = encoded.conds.video_patches,
            .audio_patches = encoded.conds.audio_patches,
        },
        .seed = args.seed,
        .brief = brief.text,
        .out = args.out,
        .sink = &sink,
    });
}

fn cancelBrief(fut: anytype, io: std.Io, allocator: std.mem.Allocator) void {
    if (fut.*) |*f| {
        if (f.cancel(io)) |b| {
            var owned = b;
            owned.deinit(allocator);
        } else |_| {}
    }
}

fn cancelCompiled(fut: anytype, io: std.Io) void {
    if (fut.*) |*f| {
        if (f.cancel(io)) |exe| {
            var compiled = exe;
            compiled.deinit();
        } else |_| {}
    }
}
