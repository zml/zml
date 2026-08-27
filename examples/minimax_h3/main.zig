const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const conditions = @import("runtime/conditions.zig");
const config = @import("core/config.zig");
const memory = @import("core/memory.zig");
const pipeline = @import("runtime/pipeline.zig");
const policy = @import("core/policy.zig");
const repo = @import("runtime/repository.zig");
const request = @import("core/request.zig");
const session = @import("runtime/session.zig");
const sharding = @import("core/sharding.zig");
const vision = @import("model/vision.zig");
const weights = @import("core/weights.zig");

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
    size: []const u8 = config.default_size,
    steps: u32 = config.default_steps,
    seed: u64 = 0,
    out: []const u8 = "output",
    dit: []const u8 = "",

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
        \\   --duration=<sec>    5–15 (default: 5)
        \\   --size=<WxH>        Pixels (default: 1344x768)
        \\   --steps=<n>         Denoise steps (default: 30)
        \\   --seed=<n>          RNG seed
        \\   --out=<path>        Directory or .mp4 (default: output/)
        \\   --dit=<path>        Transformer weights (size / quant). Encoder and VAE stay with --model
        \\
    ;
};

fn reject(err: anyerror, comptime fmt: []const u8, args: anytype) anyerror {
    log.err(fmt, args);
    return err;
}

fn rejectUser(err: anyerror) anyerror {
    return switch (err) {
        error.InvalidSize => reject(err, "--size must be WxH (example 1344x768)", .{}),
        error.InvalidAspect => reject(err, "--size aspect must be between 1:4 and 4:1", .{}),
        error.SizeTooLarge => reject(err, "--size area exceeds 768×1344 or needs >={d} GiB/device", .{config.full_canvas_min_device_bytes / (1024 * 1024 * 1024)}),
        error.InvalidDuration => reject(err, "--duration must be 5–15", .{}),
        error.TooFewSteps => reject(err, "--steps must be >= 2", .{}),
        error.AudioRefNeedsVisual => reject(err, "audio --refs need at least one image or video", .{}),
        error.Ref2vaRejectsKeyframes => reject(err, "pass --image/--last-image or --refs, not both", .{}),
        error.Ref2vaTransformerMissing => reject(err, "ref2va needs transformer_ref/", .{}),
        error.TransformerMissing => reject(err, "transformer weights not found", .{}),
        error.AmbiguousDit => reject(err, "multiple transformer files match; pass --dit=<file>", .{}),
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
        error.Fl2vaNeedsImage => reject(err, "fl2va needs --image and/or --last-image", .{}),
        error.Ref2vaNeedsRefs => reject(err, "ref2va needs --refs", .{}),
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
    config.checkSteps(args.steps) catch |err| return rejectUser(err);
    const px = config.parseSize(args.size) catch |err| return rejectUser(err);
    const refs = try request.refsFromComma(allocator, args.refs);
    defer request.freeRefs(allocator, refs, false);
    const variant = request.inferVariant(args.image, args.last_image, refs) catch |err| return rejectUser(err);
    const encode_prompt = std.mem.trimEnd(u8, args.prompt, "\n");
    request.validate(.{
        .variant = variant,
        .prompt = encode_prompt,
        .first_image = args.image,
        .last_image = args.last_image,
        .refs = refs,
    }) catch |err| return rejectUser(err);
    const paths: repo.Open = .{
        .model = args.model,
        .dit = args.dit,
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

    const platform: *zml.Platform = try .auto(allocator, io, .{
        .cpu = .{ .device_count = 1 },
        .physical_mesh = .{ .custom = sharding.physicalMesh },
        // Grow to what the run uses. Default BFC preallocate grabs 90% of every GPU.
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false, .memory_fraction = 0.90 } } },
    });
    defer platform.deinit(allocator, io);
    try vision.register(platform);
    log.info("\n{f}", .{platform.fmtVerbose()});

    const device_bytes = config.minDeviceBytes(platform);
    config.checkDeviceForSize(px.w, px.h, device_bytes) catch |err| return rejectUser(err);

    const shardings: sharding.Shardings = try .init(platform);
    const frames = config.alignFrameCount(config.frameCount(args.duration));
    log.info(
        "run model={s} variant={s} {d}x{d} frames={d} steps={d} seed={d} target={s} shard={d} devices={d} device={d}GiB",
        .{
            args.model,
            @tagName(variant),
            px.w,
            px.h,
            frames,
            args.steps,
            args.seed,
            @tagName(platform.target),
            shardings.model.numPartitionsForLogicalAxis(.model),
            platform.devices.len,
            device_bytes / (1024 * 1024 * 1024),
        },
    );

    const model_repo = try zml.safetensors.resolveModelRepo(io, args.model);
    var models = repo.Bundle.open(allocator, io, model_repo, variant, shardings, paths) catch |err| return rejectUser(err);
    defer models.deinit(allocator, io);

    const opts: pipeline.Options = .{
        .variant = variant,
        .duration_s = args.duration,
        .width = px.w,
        .height = px.h,
        .steps = args.steps,
        .seed = args.seed,
    };
    const geo = pipeline.Geometry.init(opts, models.dit.cfg);

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    defer progress.end();

    var tokenizer = repo.loadTokenizer(allocator, io, models.task, model_repo, args.model, &progress) catch |err| return rejectUser(err);
    defer tokenizer.deinit();
    var tok_enc = try tokenizer.encoder();
    defer tok_enc.deinit();

    var encoded = if (hasMedia(args))
        try conditions.prepare(allocator, io, platform, &progress, &tok_enc, .{
            .variant = variant,
            .first_image = args.image,
            .last_image = args.last_image,
            .refs = refs,
            .prompt = encode_prompt,
            .geo = geo,
            .models = &models,
            .shardings = shardings,
        })
    else
        try conditions.tokenize(allocator, &tok_enc, encode_prompt);
    defer encoded.deinit(allocator);

    const geo_work = geo.withConditions(encoded.conds.target_video_offset, encoded.conds.target_audio_offset);
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
    log.info(
        "layout {s} {d}x{d} {d} frames ({d:.1}s) latents {d}x{d}x{d} audio_t={d} seq={d} steps={d} seed={d}",
        .{
            @tagName(opts.variant),
            geo_work.pixel_w,
            geo_work.pixel_h,
            geo_work.frames,
            opts.duration_s,
            geo_work.latent_t,
            geo_work.latent_h,
            geo_work.latent_w,
            geo_work.audio_t,
            packed_run.layout.seqLen(),
            opts.steps,
            opts.seed,
        },
    );

    const core0 = models.dit.inner.blocks[0].corePart();
    const dit_dt = models.dit.inner.blocks[0].norm1.weight.dtype();
    const tp: u32 = @intCast(shardings.model.numPartitionsForLogicalAxis(.model));
    const mem = memory.plan(.{
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
        .flash = .auto(platform),
    });
    if (!mem.safe) {
        log.err("{s} (peak {d} MiB)", .{ mem.reason, mem.peak_bytes / (1024 * 1024) });
        return rejectUser(error.MemoryPlanUnsafe);
    }
    log.info(
        "memory peak={d}MiB act={d}MiB block={d}MiB scores={d}MiB fa2={d}MiB tables={d}MiB resident={d} keep={d} group={d} tile_batch={d} attn={s}",
        .{
            mem.peak_bytes / (1024 * 1024),
            mem.activation_bytes / (1024 * 1024),
            mem.streamed_block_bytes / (1024 * 1024),
            mem.score_bytes / (1024 * 1024),
            mem.fa2_scratch_bytes / (1024 * 1024),
            mem.adaln_table_bytes / (1024 * 1024),
            mem.resident_blocks,
            policy.ditKeepBlocks(mem.resident_blocks, @intCast(models.dit.cfg.num_layers)),
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
    var compiled = try pipeline.compile(
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
    );
    defer compiled.deinit();

    var compiled_vae = try pipeline.compileVae(
        allocator,
        io,
        platform,
        models.visual.inner,
        geo_work,
        mem.tile_batch,
        shardings,
        &progress,
    );
    defer compiled_vae.deinit();

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
        .resident_blocks = mem.resident_blocks,
        .out = args.out,
    });
}
