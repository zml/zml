const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const conditions = @import("runtime/conditions.zig");
const config = @import("core/config.zig");
const ir = @import("ir/compile.zig");
const memory = @import("core/memory.zig");
const packing = @import("model/packing.zig");
const pipeline = @import("runtime/pipeline.zig");
const repo = @import("runtime/repo.zig");
const request = @import("core/request.zig");
const session = @import("runtime/session.zig");
const sharding = @import("core/sharding.zig");

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
    ratio: []const u8 = "16:9",
    steps: u32 = 0,
    seed: u64 = 0,
    out: []const u8 = "output",
    tiny: bool = false,
    preview: bool = false,
    full: bool = false,

    pub const help =
        \\ Use minimax_h3 --model=<path> [options]
        \\
        \\ Joint video+audio from a MiniMax-H3 repository.
        \\ Attachments pick the task: none, --image/--last-image, or --refs.
        \\
        \\ Options:
        \\   --model=<path>      Model repository (required)
        \\   --prompt=<string>   Text prompt
        \\   --image=<path>      First frame
        \\   --last-image=<path> Last frame
        \\   --refs=<paths>      Comma-separated images/videos/audio
        \\   --duration=<sec>    4–15 seconds (default: 5)
        \\   --ratio=<aspect>    21:9 | 16:9 | 4:3 | 1:1 | 3:4 | 9:16
        \\   --tiny|--preview|--full   Canvas (auto if omitted)
        \\   --steps=<n>         Denoising steps
        \\   --seed=<n>          RNG seed
        \\   --out=<path>        Directory or .mp4 (default: output/)
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
        error.ConflictingCanvas => reject(err, "use only one of --tiny, --preview, --full", .{}),
        error.Ref2vaRejectsKeyframes => reject(err, "pass --image/--last-image or --refs, not both", .{}),
        error.FullCanvasTooLarge => reject(
            err,
            "--full needs at least {d} GiB per device. Use --preview.",
            .{config.full_canvas_min_device_bytes / (1024 * 1024 * 1024)},
        ),
        error.ConditionedPreviewTooLarge => reject(
            err,
            "preview + images/refs needs --tiny on devices under {d} GiB",
            .{config.full_canvas_min_device_bytes / (1024 * 1024 * 1024)},
        ),
        error.Ref2vaTransformerMissing => reject(
            err,
            "ref2va needs Ref2VA/transformer (hf download MiniMaxAI/MiniMax-H3 --include \"Ref2VA/*\")",
            .{},
        ),
        error.TransformerMissing => reject(err, "transformer weights not found", .{}),
        error.EncoderMissing => reject(err, "text_encoder not found", .{}),
        error.VaeMissing => reject(err, "video_vae or audio_vae not found", .{}),
        error.VaeSchemaMismatch => reject(err, "VAE weight names not recognized", .{}),
        error.MissingTokenizer => reject(err, "tokenizer.json not found under the task dir, repo root, or FL2VA/", .{}),
        error.MemoryPlanUnsafe => reject(err, "run does not fit device memory; try --tiny", .{}),
        error.H3irLlmMissing => reject(err, "H3IR_LLM_URL is empty; unset it to use the local draft", .{}),
        error.H3irLlmFailed => reject(err, "IR LLM call failed. Check H3IR_LLM_URL / H3IR_LLM_MODEL.", .{}),
        error.H3irEmpty => reject(err, "IR LLM returned an empty brief", .{}),
        error.H3irHttpMissing => reject(err, "IR HTTP client missing", .{}),
        error.TooManyRefs => reject(err, "too many --refs (max 12 files)", .{}),
        error.TooManyRefImages => reject(err, "too many reference images (max 9)", .{}),
        error.TooManyRefVideos => reject(err, "too many reference videos (max 3)", .{}),
        error.TooManyRefAudios => reject(err, "too many reference audios (max 3)", .{}),
        error.IntentEmpty => reject(err, "needs a non-empty --prompt", .{}),
        error.CompilerInvariant => reject(err, "IR draft failed its own validator", .{}),
        else => err,
    };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // `bazel run` executes binaries from Bazel's runfiles tree by default.
    // If available, switch back to the shell's original working directory.
    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
        var working_dir = try std.Io.Dir.openDirAbsolute(init.io, build_working_directory, .{});
        defer working_dir.close(init.io);
        try std.process.setCurrentDir(init.io, working_dir);
    }

    const args = stdx.flags.parse(init.minimal.args, Args);
    const aspect = config.Aspect.parse(args.ratio) orelse return rejectUser(error.UnknownAspect);
    if (args.duration < 4.0 or args.duration > 15.0) return rejectUser(error.InvalidDuration);
    const refs = try request.refsFromComma(allocator, args.refs);
    defer request.freeRefs(allocator, refs, false);
    const variant = request.inferVariant(args.image, args.last_image, refs) catch |err| return rejectUser(err);
    const canvas_choice = config.parseCanvas(args.tiny, args.preview, args.full) catch |err| return rejectUser(err);

    //
    // Virtual File Systems
    //
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

    var ir_http: std.http.Client = .{ .allocator = allocator, .io = init.io };
    defer ir_http.deinit();
    const ir_args = .{ allocator, init.io, ir.Request{
        .prompt = args.prompt,
        .variant = variant,
        .duration_s = args.duration,
        .aspect = args.ratio,
        .llm_url = init.environ_map.get("H3IR_LLM_URL"),
        .llm_model = init.environ_map.get("H3IR_LLM_MODEL"),
        .image = args.image,
        .last_image = args.last_image,
        .refs = args.refs,
        .seed = args.seed,
        .http = &ir_http,
    } };
    var ir_fut: ?@TypeOf(try init.io.concurrent(ir.compile, ir_args)) = try init.io.concurrent(ir.compile, ir_args);
    errdefer cancelBrief(&ir_fut, init.io, allocator);

    //
    // Platform
    //
    const platform: *zml.Platform = try .auto(allocator, io, .{
        .cpu = .{ .device_count = 1 },
        .physical_mesh = .{ .custom = sharding.physicalMesh },
    });
    defer platform.deinit(allocator, io);
    log.info("\n{f}", .{platform.fmtVerbose()});

    const device_bytes = config.minDeviceBytes(platform);
    const canvas = config.canvasForTarget(platform.target, canvas_choice, device_bytes);
    const steps = if (args.steps == 0) canvas.steps else args.steps;
    config.checkCanvas(canvas_choice, variant, canvas.short_side, device_bytes) catch |err| return rejectUser(err);

    const shardings: sharding.Shardings = try .init(platform);
    const px = config.pixelSize(aspect, canvas.short_side);
    const frames = config.alignFrameCount(config.frameCount(args.duration));
    log.info(
        "run model={s} variant={s} canvas={s} {d}x{d} frames={d} steps={d} seed={d} target={s} shard={d} devices={d}",
        .{
            args.model,
            @tagName(variant),
            @tagName(canvas_choice),
            px.w,
            px.h,
            frames,
            steps,
            args.seed,
            @tagName(platform.target),
            shardings.model.numPartitionsForLogicalAxis(.model),
            platform.devices.len,
        },
    );

    //
    // Model
    //
    const model_repo = try zml.safetensors.resolveModelRepo(io, args.model);
    var models = repo.Bundle.open(allocator, io, model_repo, variant, shardings) catch |err| return rejectUser(err);
    defer models.deinit(allocator, io);

    const opts: pipeline.Options = .{
        .variant = variant,
        .duration_s = args.duration,
        .aspect = aspect,
        .short_side = canvas.short_side,
        .steps = steps,
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

    var tokenizer = repo.loadTokenizer(allocator, io, models.task, model_repo, &progress) catch |err| return rejectUser(err);
    defer tokenizer.deinit();
    var tok_enc = try tokenizer.encoder();
    defer tok_enc.deinit();

    const has_media = args.image.len != 0 or args.last_image.len != 0 or refs.len != 0;
    var encoded = if (has_media)
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

    const geo_work = if (has_media)
        geo.withConditions(encoded.conds.target_video_offset, encoded.conds.target_audio_offset)
    else
        geo;
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

    const mem = memory.plan(
        geo_work,
        packed_run.layout,
        models.dit.cfg.hidden_size,
        steps,
        device_bytes,
        @intCast(shardings.model.numPartitionsForLogicalAxis(.model)),
    );
    if (!mem.safe) {
        log.err("{s} (peak {d} MiB)", .{ mem.reason, mem.peak_bytes / (1024 * 1024) });
        return rejectUser(error.MemoryPlanUnsafe);
    }
    log.info("memory peak={d}MiB act={d}MiB block={d}MiB safe={}", .{
        mem.peak_bytes / (1024 * 1024),
        mem.activation_bytes / (1024 * 1024),
        mem.streamed_block_bytes / (1024 * 1024),
        mem.safe,
    });

    //
    // Compile
    //
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
        packing.timestep_slot_count,
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

    //
    // Generate
    //
    try session.generate(allocator, io, platform, &models, &compiled, &compiled_vae, &all, &progress, .{
        .opts = opts,
        .geo = geo_work,
        .target = geo,
        .tokens = encoded.tokens,
        .extras = encoded.extras(),
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
