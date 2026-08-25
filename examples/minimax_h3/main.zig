const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const audio_vae = @import("audio_vae.zig");
const conditions = @import("conditions.zig");
const config_mod = @import("config.zig");
const decode_mod = @import("decode.zig");
const dit = @import("dit.zig");
const encode_mod = @import("encode.zig");
const encoder = @import("encoder.zig");
const ir_mod = @import("ir.zig");
const media = @import("media.zig");
const packing = @import("packing.zig");
const pipeline = @import("pipeline.zig");
const scheduler_mod = @import("scheduler.zig");
const session_mod = @import("session.zig");
const sharding_mod = @import("sharding.zig");
const visual_vae = @import("visual_vae.zig");

const log = std.log.scoped(.minimax_h3);

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8 = "hf://MiniMaxAI/MiniMax-H3",
    prompt: []const u8 = "A cinematic wide shot of waves at dusk.",
    variant: []const u8 = "t2va",
    duration: f32 = 5.0,
    ratio: []const u8 = "16:9",
    steps: u32 = 0,
    seed: u64 = 0,
    short_side: u32 = 0,
    ir: []const u8 = "auto",
    preview: bool = false,
    full: bool = false,
    tiny: bool = false,
    compile_only: bool = false,
    decode_only: bool = false,
    probe: bool = false,
    max_vae_blocks: u32 = 0,
    max_vae_chunks: u32 = 0,
    image: []const u8 = "",
    last_image: []const u8 = "",
    refs: []const u8 = "",
    out: []const u8 = ".",
    shared: []const u8 = "",

    pub const help =
        \\ Use minimax_h3 --model=<path> [options]
        \\
        \\ MiniMax-H3 joint video+audio. Pick one canvas: --tiny, --preview, or --full.
        \\ Auto is preview on CPU/Metal and on devices under 40 GiB; official 768p
        \\ otherwise. --full, --short-side above 352, and preview + audio --refs
        \\ are refused below 40 GiB (use --tiny for image+audio refs).
        \\
        \\ Options:
        \\   --model=<path>      Repository (hf://MiniMaxAI/MiniMax-H3 or local)
        \\   --prompt=<string>   Text prompt (OpenH3-IR or Prompting Guidance wrap)
        \\   --ir=<mode>         auto | prompt | h3ir | off (default: auto)
        \\   --variant=<name>    t2va | fl2va | ref2va (default: t2va)
        \\   --image=<path>      FL2VA first frame (ppm/jpg/png)
        \\   --last-image=<path> FL2VA last frame
        \\   --refs=<paths>      Ref2VA comma-separated images/videos/audio (max 12)
        \\   --duration=<sec>    4–15 seconds (default: 5)
        \\   --ratio=<aspect>    21:9 | 16:9 | 4:3 | 1:1 | 3:4 | 9:16
        \\   --steps=<n>         Denoising steps (0 = canvas default)
        \\   --short-side=<px>   Short edge before snap-32 (0 = canvas default)
        \\   --preview           352 short side, 10 steps
        \\   --tiny              128 short side, 4 steps
        \\   --full              768 short side, 30 steps
        \\   --compile-only      Stop after compile; skip weight load
        \\   --decode-only       Decode video_latents.f32 and audio_latents.f32 from --out
        \\   --probe             Run each VAE executable once
        \\   --max-vae-blocks=n  Decode at most n visual blocks (0 = all)
        \\   --max-vae-chunks=n  Decode at most n temporal chunks (0 = all)
        \\   --out=<dir>         Frames, audio.wav, output.mp4, and *.f32 latents
        \\   --shared=<dir>      Official video_noise.f32, audio_noise.f32, prompt_embeds.f32
        \\   --seed=<n>          RNG seed
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |cwd| {
        var working_dir = try std.Io.Dir.openDirAbsolute(init.io, cwd, .{});
        defer working_dir.close(init.io);
        try std.process.setCurrentDir(init.io, working_dir);
    }

    const args = stdx.flags.parse(init.minimal.args, Args);
    const variant = std.meta.stringToEnum(config_mod.Variant, args.variant) orelse
        return reject(error.UnknownVariant, "unknown --variant={s} (t2va|fl2va|ref2va)", .{args.variant});
    const aspect = config_mod.Aspect.parse(args.ratio) orelse
        return reject(error.UnknownAspect, "unknown --ratio={s} (21:9|16:9|4:3|1:1|3:4|9:16)", .{args.ratio});
    const ir_mode = ir_mod.Mode.parse(args.ir) orelse
        return reject(error.UnknownIrMode, "unknown --ir={s} (auto|prompt|h3ir|off)", .{args.ir});
    if (args.duration < 4.0 or args.duration > 15.0)
        return reject(error.InvalidDuration, "--duration must be 4–15, got {d}", .{args.duration});
    if (variant == .fl2va and args.image.len == 0 and args.last_image.len == 0 and !args.decode_only and !args.probe)
        return reject(error.Fl2vaNeedsImage, "fl2va requires --image and/or --last-image", .{});
    if (variant == .ref2va and args.refs.len == 0 and !args.decode_only and !args.probe)
        return reject(error.Ref2vaNeedsRefs, "ref2va requires --refs", .{});
    if (variant == .t2va and (args.image.len != 0 or args.last_image.len != 0 or args.refs.len != 0))
        return reject(error.T2vaRejectsMedia, "t2va does not take --image, --last-image, or --refs", .{});
    const canvas_choice = config_mod.parseCanvas(args.tiny, args.preview, args.full) catch
        return reject(error.ConflictingCanvas, "use only one of --tiny, --preview, --full", .{});

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
        .physical_mesh = .{ .custom = sharding_mod.physicalMesh },
    });
    defer platform.deinit(allocator, io);
    log.info("\n{f}", .{platform.fmtVerbose()});

    const device_bytes = config_mod.minDeviceBytes(platform);
    if (canvas_choice == .full and device_bytes != 0 and device_bytes < config_mod.full_canvas_min_device_bytes) {
        return reject(
            error.FullCanvasTooLarge,
            "--full needs at least {d} GiB per device; this platform reports {d} GiB. Use --preview.",
            .{ config_mod.full_canvas_min_device_bytes / (1024 * 1024 * 1024), device_bytes / (1024 * 1024 * 1024) },
        );
    }
    const canvas = config_mod.canvasForTarget(platform.target, canvas_choice, device_bytes);
    const short_side = if (args.short_side == 0) canvas.short_side else args.short_side;
    if (device_bytes != 0 and device_bytes < config_mod.full_canvas_min_device_bytes and short_side > config_mod.preview_short_side) {
        return reject(
            error.CanvasTooLarge,
            "short side {d} needs at least {d} GiB per device; this platform reports {d} GiB. Use --preview or --tiny.",
            .{ short_side, config_mod.full_canvas_min_device_bytes / (1024 * 1024 * 1024), device_bytes / (1024 * 1024 * 1024) },
        );
    }
    const steps = if (args.steps == 0) canvas.steps else args.steps;
    if (variant == .ref2va and !args.decode_only and !args.probe and config_mod.audioRefsNeedTiny(short_side, device_bytes) and media.refsContainAudio(args.refs)) {
        return reject(
            error.AudioRefTooLarge,
            "preview + audio --refs needs --tiny on devices under {d} GiB; this platform reports {d} GiB",
            .{ config_mod.full_canvas_min_device_bytes / (1024 * 1024 * 1024), device_bytes / (1024 * 1024 * 1024) },
        );
    }

    const shardings: sharding_mod.Shardings = try .init(platform);
    const px = config_mod.pixelSize(aspect, short_side);
    log.info(
        "run model={s} variant={s} canvas={s} ir={s} {d}x{d} frames={d} steps={d} seed={d} target={s} shard={d} devices={d}",
        .{
            args.model,
            @tagName(variant),
            @tagName(canvas_choice),
            @tagName(ir_mode),
            px.w,
            px.h,
            config_mod.alignFrameCount(config_mod.frameCount(args.duration)),
            steps,
            args.seed,
            @tagName(platform.target),
            shardings.model.numPartitionsForLogicalAxis(.model),
            platform.devices.len,
        },
    );
    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    const task = try config_mod.openTaskDir(io, repo, variant);
    var task_dir = task.dir;
    defer if (task.owned) task_dir.close(io);

    const opts: pipeline.Options = .{
        .variant = variant,
        .duration_s = args.duration,
        .aspect = aspect,
        .short_side = short_side,
        .steps = steps,
        .seed = args.seed,
    };

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    defer progress.end();

    var transformer_dir = openTransformer(io, task_dir, repo, variant) catch |err| switch (err) {
        error.Ref2vaTransformerMissing => return reject(
            err,
            "ref2va needs Ref2VA/transformer or transformer_ref (hf download MiniMaxAI/MiniMax-H3 --include \"Ref2VA/*\")",
            .{},
        ),
        error.TransformerMissing => return reject(err, "transformer weights not found under {s}", .{args.model}),
    };
    defer transformer_dir.close(io);
    const transformer = transformer_dir;

    var encoder_dir = openSharedComponent(io, task_dir, repo, "text_encoder");
    defer if (encoder_dir) |*dir| dir.close(io);
    const enc_dir = encoder_dir orelse task_dir;

    var visual_cfg_dir = openSharedComponent(io, task_dir, repo, "video_vae") orelse
        openSharedComponent(io, task_dir, repo, "vae");
    defer if (visual_cfg_dir) |*dir| dir.close(io);
    var visual_dir = if (visual_cfg_dir) |dir| openOptionalDir(io, dir, "source") else null;
    defer if (visual_dir) |*dir| dir.close(io);
    const visual_weights = visual_dir orelse visual_cfg_dir;
    const visual_cfg = visual_cfg_dir orelse visual_dir;

    var audio_dir = openSharedComponent(io, task_dir, repo, "audio_vae");
    defer if (audio_dir) |*dir| dir.close(io);

    var dit_registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, transformer);
    defer dit_registry.deinit();
    var dit_store: zml.io.TensorStore = .fromRegistry(allocator, &dit_registry);
    defer dit_store.deinit();

    var enc_registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, enc_dir);
    defer enc_registry.deinit();
    var enc_store: zml.io.TensorStore = .fromRegistry(allocator, &enc_registry);
    defer enc_store.deinit();

    var loaded_dit = try dit.LoadedModel.init(allocator, io, transformer, dit_store.view());
    defer loaded_dit.deinit(allocator);
    const dit_cfg = loaded_dit.parsed_config.value.resolve();

    var loaded_enc = try encoder.LoadedModel.init(allocator, io, enc_dir, enc_store.view());
    defer loaded_enc.deinit(allocator);
    try shardings.checkLoaded(dit_cfg, loaded_enc.cfg);

    var loaded_visual: ?visual_vae.LoadedModel = null;
    var visual_store: ?zml.io.TensorStore = null;
    var visual_registry: ?zml.safetensors.TensorRegistry = null;
    var compiled_vae: ?pipeline.VaeCompiled = null;
    var loaded_audio: ?audio_vae.LoadedModel = null;
    var audio_store: ?zml.io.TensorStore = null;
    var audio_registry: ?zml.safetensors.TensorRegistry = null;
    defer {
        if (compiled_vae) |*c| c.deinit();
        if (loaded_visual) |*m| m.deinit(allocator);
        if (visual_store) |*s| s.deinit();
        if (visual_registry) |*r| r.deinit();
        if (loaded_audio) |*m| m.deinit(allocator);
        if (audio_store) |*s| s.deinit();
        if (audio_registry) |*r| r.deinit();
    }
    if (visual_weights != null and audio_dir != null) {
        visual_registry = try .fromRepo(allocator, io, visual_weights.?);
        visual_store = .fromRegistry(allocator, &visual_registry.?);
        audio_registry = try .fromRepo(allocator, io, audio_dir.?);
        audio_store = .fromRegistry(allocator, &audio_registry.?);
        const vview = visual_store.?.view();
        const aview = audio_store.?.view();
        if (visual_vae.ready(vview) and audio_vae.decodeReady(aview)) {
            loaded_visual = try visual_vae.LoadedModel.init(allocator, io, visual_cfg.?, vview);
            loaded_audio = try audio_vae.LoadedModel.init(allocator, io, audio_dir.?, aview);
            log.info("vae: video+audio graphs ready", .{});
        } else {
            log.warn("vae: weight names not recognized; decode skipped", .{});
        }
    } else {
        log.info("vae: video_vae or audio_vae dir missing; latents-only", .{});
    }

    const geo = pipeline.Geometry.init(opts, dit_cfg);

    var brief = try ir_mod.compile(allocator, io, .{
        .prompt = args.prompt,
        .variant = variant,
        .duration_s = args.duration,
        .mode = ir_mode,
        .llm_url = init.environ_map.get("H3IR_LLM_URL"),
    });
    defer brief.deinit(allocator);

    var tokenizer = try loadTokenizer(allocator, io, task_dir, repo, &progress);
    defer tokenizer.deinit();
    var tok_enc = try tokenizer.encoder();
    defer tok_enc.deinit();
    const prompt_tokens = try tok_enc.encodeAlloc(allocator, brief.text);
    defer allocator.free(prompt_tokens);
    const colon_tokens = try tok_enc.encodeAlloc(allocator, ": ");
    defer allocator.free(colon_tokens);

    const ref_paths = try conditions.splitComma(allocator, args.refs);
    defer allocator.free(ref_paths);

    var geo_work = geo;
    var tokens = try allocator.dupe(u32, prompt_tokens);
    defer allocator.free(tokens);
    var text_tags = try allocator.alloc(u8, tokens.len);
    defer allocator.free(text_tags);
    @memset(text_tags, @intFromEnum(packing.Modality.text));
    var text_extras: session_mod.TextExtras = .{};
    var vision_merged: ?[]f32 = null;
    var vision_pos: ?[]f32 = null;
    var vision_ds: [3]?[]f32 = .{ null, null, null };
    var vision_spans: []session_mod.VisionSpan = &.{};
    defer {
        if (vision_merged) |e| allocator.free(e);
        if (vision_pos) |p| allocator.free(p);
        for (vision_ds) |d| if (d) |x| allocator.free(x);
        if (vision_spans.len != 0) allocator.free(vision_spans);
    }

    var cond_set: encode_mod.ConditionSet = .{
        .videos = &.{},
        .video_patches = &.{},
        .target_video_offset = 0,
        .audios = &.{},
        .audio_patches = &.{},
        .target_audio_offset = 0,
        .references = &.{},
    };
    var cond_owned = false;
    defer if (cond_owned) cond_set.deinit(allocator);

    if (!args.decode_only and !args.probe and (args.image.len != 0 or args.last_image.len != 0 or ref_paths.len != 0)) {
        const prepared = try conditions.prepare(
            allocator,
            io,
            platform,
            &progress,
            variant,
            args.image,
            args.last_image,
            ref_paths,
            geo,
            dit_cfg.patch_size,
            enc_dir,
            if (visual_store) |*s| s else null,
            if (audio_store) |*s| s else null,
            &enc_store,
            if (loaded_visual) |*m| m else null,
            if (loaded_audio) |*m| m else null,
            shardings,
            colon_tokens,
            prompt_tokens,
            loaded_enc.cfg.hidden_size,
            args.compile_only,
        );
        allocator.free(tokens);
        allocator.free(text_tags);
        tokens = prepared.tokens;
        text_tags = prepared.tags;
        vision_merged = prepared.vision_merged;
        vision_pos = prepared.positions;
        vision_ds = prepared.deepstack;
        vision_spans = prepared.vision_spans;
        cond_set = prepared.conds;
        cond_owned = true;
        geo_work = geo.withConditions(cond_set.target_video_offset, cond_set.target_audio_offset);
        text_extras = .{
            .positions = vision_pos,
            .deepstack = vision_ds,
            .vision_merged = vision_merged,
            .vision_spans = vision_spans,
        };
    }

    var shared_inputs: ?session_mod.SharedInputs = null;
    defer if (shared_inputs) |s| s.deinit(allocator);
    if (args.shared.len != 0 and !args.decode_only and !args.probe) {
        var shared_dir = try std.Io.Dir.cwd().openDir(io, args.shared, .{});
        defer shared_dir.close(io);
        const loaded = try session_mod.loadShared(allocator, io, shared_dir, geo_work, @intCast(dit_cfg.text_dim));
        shared_inputs = loaded;
        allocator.free(text_tags);
        text_tags = try allocator.dupe(u8, loaded.tags);
    }

    const text_len: u32 = if (shared_inputs) |s| blk: {
        const dim: usize = @intCast(dit_cfg.text_dim);
        break :blk @intCast(s.embeds.len / dim);
    } else @intCast(tokens.len);
    log.info("prompt tokens={d} refs={d} cond_video={d} cond_audio={d} shared={s}", .{
        text_len,
        cond_set.references.len,
        cond_set.videos.len,
        cond_set.audios.len,
        if (args.shared.len == 0) "-" else args.shared,
    });

    const schedules = try scheduler_mod.DualSchedule.init(allocator, opts.steps, opts.video_shift, opts.audio_shift);
    defer schedules.deinit(allocator);
    const video_t = schedules.video.timesteps[0];
    const audio_sigma = scheduler_mod.timeShiftSigma(1.0 - video_t, opts.video_shift, opts.audio_shift);
    const audio_t = 1.0 - audio_sigma;

    var layout = try packing.build(allocator, .{
        .text_len = text_len,
        .latent_t = geo_work.latent_t,
        .latent_h = geo_work.latent_h,
        .latent_w = geo_work.latent_w,
        .audio_t = geo_work.audio_t,
        .video_t = video_t,
        .audio_t_noise = audio_t,
        .condition_videos = cond_set.videos,
        .condition_audios = cond_set.audios,
        .references = cond_set.references,
        .text_tags = text_tags,
    });
    defer layout.deinit(allocator);
    pipeline.describe(opts, geo_work, layout);

    const all = shardings.all();
    var compiled: ?pipeline.Compiled = null;
    defer if (compiled) |*c| c.deinit();
    var dit_compile = if (args.probe or args.decode_only) null else try io.concurrent(pipeline.compile, .{
        allocator,
        io,
        platform,
        loaded_dit.inner,
        loaded_enc.inner,
        geo_work,
        text_len,
        layout.seqLen(),
        packing.timestep_slot_count,
        shardings,
        &progress,
    });
    errdefer if (dit_compile) |*f| if (f.cancel(io)) |exe| {
        var c = exe;
        c.deinit();
    } else |_| {} else {};
    if (args.probe or args.decode_only) {
        log.info("{s}: skip DiT/encoder compile", .{if (args.probe) "probe" else "decode-only"});
    }

    if (loaded_visual != null and loaded_audio != null) {
        compiled_vae = try pipeline.compileVae(
            allocator,
            io,
            platform,
            loaded_visual.?.inner,
            loaded_audio.?.inner,
            geo_work,
            shardings,
            &progress,
        );
    }

    if (dit_compile) |*f| {
        compiled = try f.await(io);
        dit_compile = null;
    }

    if (args.compile_only) {
        log.info("Compiled. Canvas {d}x{d} seq={d} steps={d}. Weight load skipped (--compile-only).", .{
            geo.pixel_w,
            geo.pixel_h,
            layout.seqLen(),
            opts.steps,
        });
        return;
    }

    if (args.probe) {
        if (compiled_vae == null) return reject(error.VaeMissing, "probe needs video_vae and audio_vae weights", .{});
        try decode_mod.probe(
            allocator,
            io,
            platform,
            &compiled_vae.?,
            &loaded_visual.?,
            &visual_store.?,
            &loaded_audio.?,
            &audio_store.?,
            &all,
            geo,
            &progress,
        );
        return;
    }

    var out_owned = false;
    var out_dir: std.Io.Dir = std.Io.Dir.cwd();
    if (!std.mem.eql(u8, args.out, ".")) {
        try std.Io.Dir.cwd().createDirPath(io, args.out);
        out_dir = try std.Io.Dir.cwd().openDir(io, args.out, .{});
        out_owned = true;
    }
    defer if (out_owned) out_dir.close(io);

    const video_n = geo.video_tokens * geo.video_patch_dim;
    const audio_n = geo.audio_tokens * geo.audio_dim;
    var latents: session_mod.Latents = if (args.decode_only) blk: {
        const video = try session_mod.readF32File(allocator, io, out_dir, "video_latents.f32", video_n);
        const audio = try session_mod.readF32File(allocator, io, out_dir, "audio_latents.f32", audio_n);
        log.info("decode-only: loaded video_latents.f32 ({d}) audio_latents.f32 ({d})", .{ video.len, audio.len });
        break :blk .{ .video = video, .audio = audio };
    } else blk: {
        var text = if (shared_inputs) |s|
            try session_mod.bufferFromF32(
                allocator,
                io,
                platform,
                .init(.{ .b = 1, .s = text_len, .d = dit_cfg.text_dim }, loaded_enc.inner.embed_tokens.weight.dtype()),
                s.embeds,
            )
        else
            try session_mod.encodeText(allocator, io, platform, &compiled.?, &loaded_enc, &enc_store, &all, tokens, text_extras, &progress);
        defer text.deinit();
        const denoised = try session_mod.denoise(
            allocator,
            io,
            platform,
            &compiled.?,
            &loaded_dit,
            &dit_store,
            &all,
            opts,
            geo_work,
            text,
            text_len,
            layout,
            schedules,
            args.seed,
            .{
                .video_patches = cond_set.video_patches,
                .audio_patches = cond_set.audio_patches,
                .video_noise = if (shared_inputs) |s| s.video_rows else null,
                .audio_noise = if (shared_inputs) |s| s.audio_rows else null,
            },
            &progress,
        );
        try session_mod.writeF32File(io, out_dir, "video_latents.f32", denoised.video);
        try session_mod.writeF32File(io, out_dir, "audio_latents.f32", denoised.audio);
        log.info("wrote video_latents.f32 ({d}) audio_latents.f32 ({d}) out={s}", .{
            denoised.video.len,
            denoised.audio.len,
            args.out,
        });
        break :blk denoised;
    };
    defer latents.deinit(allocator);

    if (compiled_vae) |*vae_exe| {
        const thwc = try packing.unpatchify(allocator, latents.video, geo.latent_t, geo.latent_h, geo.latent_w, 24, .{ 1, 2, 2 });
        defer allocator.free(thwc);
        const rgb = try decode_mod.decodeVideo(allocator, io, platform, vae_exe, &loaded_visual.?, &visual_store.?, &all, geo, thwc, .{
            .max_blocks = args.max_vae_blocks,
            .max_chunks = args.max_vae_chunks,
        }, &progress);
        defer allocator.free(rgb);
        const wav = try decode_mod.decodeAudio(allocator, io, platform, vae_exe, &loaded_audio.?, &audio_store.?, &all, geo, latents.audio, &progress);
        defer allocator.free(wav);
        try decode_mod.writeOutputs(allocator, io, out_dir, args.out, geo, rgb, wav);
    } else {
        log.info("VAE dirs missing; wrote patchified latents only", .{});
    }
}

fn openOptionalDir(io: std.Io, parent: std.Io.Dir, name: []const u8) ?std.Io.Dir {
    return parent.openDir(io, name, .{}) catch null;
}

fn openSharedComponent(io: std.Io, task_dir: std.Io.Dir, repo: std.Io.Dir, name: []const u8) ?std.Io.Dir {
    if (openOptionalDir(io, task_dir, name)) |dir| return dir;
    if (openOptionalDir(io, repo, name)) |dir| return dir;
    return openNestedDir(io, repo, "FL2VA", name);
}

fn openTransformer(io: std.Io, task_dir: std.Io.Dir, repo: std.Io.Dir, variant: config_mod.Variant) !std.Io.Dir {
    if (openOptionalDir(io, task_dir, "transformer")) |dir| return dir;
    if (variant == .ref2va) {
        if (openOptionalDir(io, repo, "transformer_ref")) |dir| return dir;
        if (openNestedDir(io, repo, "Ref2VA", "transformer")) |dir| return dir;
        return error.Ref2vaTransformerMissing;
    }
    if (openOptionalDir(io, repo, "transformer")) |dir| return dir;
    if (openNestedDir(io, repo, "FL2VA", "transformer")) |dir| return dir;
    return error.TransformerMissing;
}

fn openNestedDir(io: std.Io, parent: std.Io.Dir, first: []const u8, second: []const u8) ?std.Io.Dir {
    var outer = parent.openDir(io, first, .{}) catch return null;
    const inner = outer.openDir(io, second, .{}) catch {
        outer.close(io);
        return null;
    };
    outer.close(io);
    return inner;
}

fn loadTokenizer(
    allocator: std.mem.Allocator,
    io: std.Io,
    task_dir: std.Io.Dir,
    repo: std.Io.Dir,
    progress: *std.Progress.Node,
) !zml.tokenizer.Tokenizer {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Loading tokenizer...", 1);
    defer node.end();

    const bytes = try readTokenizerAny(allocator, io, task_dir, repo);
    defer allocator.free(bytes);
    log.info("tokenizer: {d} bytes", .{bytes.len});
    return try .fromBytes(allocator, bytes);
}

fn reject(err: anyerror, comptime fmt: []const u8, args: anytype) anyerror {
    log.err(fmt, args);
    return err;
}

fn readTokenizerAny(allocator: std.mem.Allocator, io: std.Io, task_dir: std.Io.Dir, repo: std.Io.Dir) ![]u8 {
    if (readTokenizer(allocator, io, task_dir)) |bytes| return bytes else |err| switch (err) {
        error.MissingTokenizer => {},
        else => return err,
    }
    if (readTokenizer(allocator, io, repo)) |bytes| return bytes else |err| switch (err) {
        error.MissingTokenizer => {},
        else => return err,
    }
    var fl = repo.openDir(io, "FL2VA", .{}) catch
        return reject(error.MissingTokenizer, "tokenizer.json not found under the task dir, repo root, or FL2VA/", .{});
    defer fl.close(io);
    return readTokenizer(allocator, io, fl) catch |err| switch (err) {
        error.MissingTokenizer => return reject(error.MissingTokenizer, "tokenizer.json not found under the task dir, repo root, or FL2VA/", .{}),
        else => return err,
    };
}

fn readTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) ![]u8 {
    const names = [_][]const u8{ "tokenizer/tokenizer.json", "tokenizer.json" };
    for (names) |name| {
        const file = dir.openFile(io, name, .{}) catch continue;
        defer file.close(io);
        var reader = file.reader(io, &.{});
        return try reader.interface.readAlloc(allocator, try file.length(io));
    }
    return error.MissingTokenizer;
}
