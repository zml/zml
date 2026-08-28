const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const conditions = @import("runtime/conditions.zig");
const config = @import("core/config.zig");
const media = @import("runtime/media.zig");
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
    first_frame: []const u8 = "",
    last_frame: []const u8 = "",
    refs: []const u8 = "",
    duration: f32 = 5.0,
    frames: u32 = 0,
    ratio: []const u8 = "",
    resolution: []const u8 = "768P",
    size: []const u8 = "",
    short_edge: u32 = config.default_short_side,
    max_pixels: u32 = config.canvas_max_pixels,
    steps: u32 = config.default_steps,
    seed: u64 = 0,
    out: []const u8 = "output.mp4",
    dit: []const u8 = "",

    pub const help =
        \\ Use minimax_h3 --model=<path> [options]
        \\
        \\ MiniMax-H3: video + audio. Same modes as Hailuo / the MiniMax API.
        \\
        \\   text-to-video          --prompt='...'
        \\   image-to-video         --first-frame=first.png
        \\   last-frame             --last-frame=last.png
        \\   first-and-last-frame   --first-frame=a.png --last-frame=b.png
        \\   reference-to-video     --refs=char.png,motion.mp4,voice.wav
        \\
        \\ Options:
        \\   --model=<path>         Repository (required). Local or hf://MiniMaxAI/MiniMax-H3
        \\   --prompt=<string>      What happens in the shot
        \\   --first-frame=<path>   First frame
        \\   --last-frame=<path>    Last frame
        \\   --refs=<paths>         Reference images, videos, audio. Comma-separated, order matters.
        \\                          A video keeps its own soundtrack; a following wav is a separate ref.
        \\   --duration=<sec>       5–15 (default: 5)
        \\   --ratio=<spec>         Hailuo ratio: adaptive | 16:9 | 9:16 | 1:1 | 4:3 | 3:4 | 21:9
        \\                          Default: 16:9 for text-to-video, adaptive from the first visual otherwise
        \\   --resolution=768P      Open weights are 768P. 2K is hosted-only
        \\   --out=<path>           .mp4 or directory (default: output.mp4)
        \\   --steps=<n>            Denoise steps (default: 30)
        \\   --seed=<n>             RNG seed (default: 0)
        \\
        \\ Advanced:
        \\   --frames=<n>           Frame count instead of --duration
        \\   --size=<WxH>           Exact pixels (overrides --ratio)
        \\   --short-edge=<n>       Adaptive/ratio short edge (default: 768)
        \\   --max-pixels=<n>       Area cap (default: 768*1344)
        \\   --dit=<path>           Transformer weights only
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
        error.InvalidCanvas => reject(err, "--ratio must be adaptive, 16:9, 9:16, 1:1, 4:3, 3:4, or 21:9", .{}),
        error.ConflictingCanvas => reject(err, "pass --ratio or --size, not both", .{}),
        error.InvalidResolution => reject(err, "--resolution must be 768P (open weights)", .{}),
        error.OpenWeightsAre768P => reject(err, "this checkpoint is 768P; 2K is the hosted MiniMax API only", .{}),
        error.InvalidAspect => reject(err, "aspect must be between 1:4 and 4:1", .{}),
        error.SizeTooLarge => reject(err, "canvas exceeds --max-pixels or needs >={d} GiB/device", .{config.full_canvas_min_device_bytes / (1024 * 1024 * 1024)}),
        error.InvalidDuration => reject(err, "--duration must be 5–15 seconds", .{}),
        error.T2vaRejectsAdaptive => reject(err, "text-to-video needs a ratio (16:9, 9:16, …); omit --ratio for 16:9", .{}),
        error.AdaptiveNeedsVisual => reject(err, "adaptive ratio needs --first-frame, --last-frame, or a visual --refs", .{}),
        error.ImageLoadFailed => reject(err, "could not read the first image or video size", .{}),
        error.FfmpegMissing => reject(err, "ffmpeg not found; needed to read image/video size", .{}),
        error.TooFewSteps => reject(err, "--steps must be >= 2", .{}),
        error.AudioRefNeedsVisual => reject(err, "audio --refs need at least one image or video", .{}),
        error.T2vaRejectsMedia => reject(err, "text-to-video is prompt-only; drop --first-frame/--last-frame/--refs", .{}),
        error.Fl2vaRejectsRefs => reject(err, "image-to-video cannot take --refs; use reference-to-video instead", .{}),
        error.Ref2vaRejectsKeyframes => reject(err, "use --first-frame/--last-frame or --refs, not both", .{}),
        error.Ref2vaTransformerMissing => reject(err, "reference-to-video needs transformer_ref/", .{}),
        error.TransformerMissing => reject(err, "transformer weights not found", .{}),
        error.EncoderMissing => reject(err, "text_encoder not found", .{}),
        error.VaeMissing => reject(err, "vae or audio_vae not found", .{}),
        error.VaeSchemaMismatch => reject(err, "VAE weight names not recognized", .{}),
        error.UnsupportedCheckpoint => reject(err, "unsupported checkpoint", .{}),
        error.MissingTokenizer => reject(err, "tokenizer.json not found", .{}),
        error.MemoryPlanUnsafe => reject(err, "does not fit device memory", .{}),
        error.TooManyRefs => reject(err, "too many --refs (max 12)", .{}),
        error.TooManyRefImages => reject(err, "too many reference images (max 9)", .{}),
        error.TooManyRefVideos => reject(err, "too many reference videos (max 3)", .{}),
        error.TooManyRefAudios => reject(err, "too many reference audios (max 3)", .{}),
        error.IntentEmpty => reject(err, "needs a non-empty --prompt", .{}),
        error.Fl2vaNeedsImage => reject(err, "image-to-video needs --first-frame and/or --last-frame", .{}),
        error.Ref2vaNeedsRefs => reject(err, "reference-to-video needs --refs", .{}),
        else => err,
    };
}

fn hasMedia(first: []const u8, last: []const u8, refs: []const u8) bool {
    return first.len != 0 or last.len != 0 or refs.len != 0;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // `bazel run` starts in the runfiles tree. Hop back to the shell cwd.
    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
        var working_dir = try std.Io.Dir.openDirAbsolute(init.io, build_working_directory, .{});
        defer working_dir.close(init.io);
        try std.process.setCurrentDir(init.io, working_dir);
    }

    const args = stdx.flags.parse(init.minimal.args, Args);
    if (args.frames == 0) config.checkDuration(args.duration) catch |err| return rejectUser(err);
    config.checkSteps(args.steps) catch |err| return rejectUser(err);
    config.parseResolution(args.resolution) catch |err| return rejectUser(err);
    const first = args.first_frame;
    const last = args.last_frame;
    const refs = try request.refsFromComma(allocator, args.refs);
    defer request.freeRefs(allocator, refs);
    const variant = request.inferVariant(first, last, refs) catch |err| return rejectUser(err);
    const encode_prompt = std.mem.trimEnd(u8, args.prompt, "\n");
    request.validate(.{
        .variant = variant,
        .prompt = encode_prompt,
        .first_frame = first,
        .last_frame = last,
        .refs = refs,
    }) catch |err| return rejectUser(err);
    const canvas = config.pickCanvas(args.size, args.ratio) catch |err| return rejectUser(err);
    const need_src = canvas.kind == .adaptive or (canvas.kind == .default and variant != .t2va);
    var src_w: u32 = 0;
    var src_h: u32 = 0;
    const anchor = request.canvasAnchor(first, last, refs);
    if (need_src) {
        if (anchor.len == 0) return rejectUser(error.AdaptiveNeedsVisual);
        const src = media.probeSize(allocator, init.io, anchor) catch |err| return rejectUser(err);
        src_w = src.w;
        src_h = src.h;
    }
    const px = config.resolveCanvasSpec(canvas, variant, src_w, src_h, args.short_edge, args.max_pixels) catch |err| return rejectUser(err);
    const frame_plan = config.resolveFrames(args.duration, args.frames) catch |err| return rejectUser(err);
    const mode = config.modeLabel(variant, first, last);
    const play_s = frame_plan.seconds();
    log.info("{s}  {d}x{d}  {d:.1}s ({d} frames)  {d} steps  seed {d}", .{
        mode,
        px.w,
        px.h,
        play_s,
        frame_plan.aligned,
        args.steps,
        args.seed,
    });
    if (need_src)
        log.info("from {s} ({d}x{d})", .{ std.fs.path.basename(anchor), src_w, src_h });
    const paths: repo.Open = .{
        .model = args.model,
        .dit = args.dit,
    };

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
    const model_repo = try zml.safetensors.resolveModelRepo(io, args.model);
    const heads = repo.peekHeadCounts(allocator, io, model_repo, variant, paths) catch |err| return rejectUser(err);
    sharding.preparePhysicalMesh(heads);

    //
    // Platform
    //
    const platform: *zml.Platform = try .auto(allocator, io, .{
        .physical_mesh = .{ .custom = sharding.physicalMesh },
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false } } },
    });
    defer platform.deinit(allocator, io);
    try vision.register(platform);
    log.info("\n{f}", .{platform.fmtVerbose()});

    const device_bytes = config.minDeviceBytes(platform);
    config.checkDeviceForSize(px.w, px.h, device_bytes) catch |err| return rejectUser(err);

    const shardings: sharding.Shardings = try .init(platform, heads);
    if (frame_plan.raw != frame_plan.aligned)
        log.info("frames {d} → {d} (VAE 17n+5)", .{ frame_plan.raw, frame_plan.aligned });
    log.info(
        "{s}  shard={d}  devices={d}  {d}GiB",
        .{
            @tagName(platform.target),
            shardings.model.numPartitionsForLogicalAxis(.model),
            platform.devices.len,
            device_bytes / (1024 * 1024 * 1024),
        },
    );

    //
    // Load the model
    //
    var models = repo.Bundle.open(allocator, io, model_repo, variant, shardings, paths) catch |err| return rejectUser(err);
    defer models.deinit(allocator, io);

    const opts: pipeline.Options = .{
        .duration_s = args.duration,
        .width = px.w,
        .height = px.h,
        .frames = frame_plan.aligned,
        .steps = args.steps,
    };
    const out_geo = pipeline.Geometry.init(opts, models.dit.cfg);

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    defer progress.end();

    var tokenizer = repo.loadTokenizer(allocator, io, models.task, model_repo, &progress) catch |err| return rejectUser(err);
    defer tokenizer.deinit();
    var tok_enc = try tokenizer.encoder();
    defer tok_enc.deinit();

    var encoded = if (hasMedia(first, last, args.refs))
        try conditions.prepare(allocator, io, platform, &progress, &tok_enc, .{
            .variant = variant,
            .first_frame = first,
            .last_frame = last,
            .refs = refs,
            .prompt = encode_prompt,
            .geo = out_geo,
            .models = &models,
            .shardings = shardings,
        })
    else
        try conditions.tokenize(allocator, &tok_enc, encode_prompt);
    defer encoded.deinit(allocator);

    const geo = out_geo.withConditions(encoded.conds.target_video_offset, encoded.conds.target_audio_offset);
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
        geo,
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
            mode,
            geo.pixel_w,
            geo.pixel_h,
            geo.frames,
            play_s,
            geo.latent_t,
            geo.latent_h,
            geo.latent_w,
            geo.audio_t,
            packed_run.layout.seqLen(),
            opts.steps,
            args.seed,
        },
    );

    const core0 = models.dit.inner.blocks[0].corePart();
    const dit_dt = models.dit.inner.blocks[0].norm1.weight.dtype();
    const tp: u32 = @intCast(shardings.model.numPartitionsForLogicalAxis(.model));
    const mem = memory.plan(.{
        .geo = .init(geo),
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

    //
    // Compile
    //
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
        geo,
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
        out_geo,
        mem.tile_batch,
        shardings,
        &progress,
    );
    defer compiled_vae.deinit();

    //
    // Generate
    //
    try session.generate(allocator, io, platform, &models, &compiled, &compiled_vae, &all, &progress, .{
        .geo = geo,
        .canvas = out_geo,
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
