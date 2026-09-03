//! MiniMax-H3 text-to-video (768P, video only):
//! tokenize → text encoder → pack → DiT denoise → unpatchify → visual VAE → `video.rgb`.

const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const config = @import("config.zig");
const model = @import("model.zig");

const log = std.log.scoped(.minimax_h3);

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,
    prompt: []const u8 = "A cinematic wide shot of waves at dusk.",
    seed: u64 = 0,
    out: []const u8 = "out",
    steps: u32 = config.default_steps,
    dit_only: bool = false,

    pub const help =
        \\minimax_h3 --model=<path> [options]
        \\
        \\Prompt in, silent video.rgb out. 1344x768, 5s, 30 Euler steps.
        \\
        \\Options:
        \\  --model=<path>     Path to the MiniMax-H3 repository (required)
        \\  --prompt=<string>  Text prompt (default: a dusk waves shot)
        \\  --out=<dir>        Output directory for video.rgb (default: out)
        \\  --seed=<number>    Noise seed (default: 0)
        \\  --steps=<number>   Sigma points including terminal 0 (default: 30)
        \\  --dit-only         Stop after DiT; skip VAE compile and decode
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = stdx.flags.parse(init.minimal.args, Args);

    // `bazel run` executes from the runfiles tree. Write next to the workspace
    // when BUILD_WORKING_DIRECTORY is set, otherwise use --out as given.
    const out = if (std.fs.path.isAbsolute(args.out))
        try allocator.dupe(u8, args.out)
    else if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |wd|
        try std.fs.path.join(allocator, &.{ wd, args.out })
    else
        try allocator.dupe(u8, args.out);
    defer allocator.free(out);

    //
    // Platform and sharding
    //
    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    const platform: *zml.Platform = try .auto(allocator, io, .{
        .physical_mesh = .{ .custom = config.Shardings.physicalMesh },
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false } } },
    });
    defer platform.deinit(allocator, io);
    log.info("\n{f}", .{platform.fmtVerbose()});

    // Tensor-parallel on .model. Head counts must divide the GPU count.
    const shardings: config.Shardings = try .init(platform);
    const all = shardings.all();
    const geo = config.geo;
    log.info("t2v  {d}x{d}  {d:.1}s  {d} steps  seed {d}  devices={d}", .{
        geo.pixel_w, geo.pixel_h, config.duration_s, args.steps, args.seed, platform.devices.len,
    });

    //
    // Official repo layout (same names as HuggingFace MiniMaxAI/MiniMax-H3)
    //
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;

    // Qwen text tower (50 of 64 layers)
    const enc_path = try std.fmt.bufPrint(&path_buf, "{s}/text_encoder/model.safetensors.index.json", .{args.model});
    var enc_reg: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, enc_path);
    defer enc_reg.deinit();
    var enc_store: zml.io.TensorStore = .fromRegistry(allocator, &enc_reg);
    defer enc_store.deinit();

    // MiniMaxH3Transformer3DModel
    const dit_path = try std.fmt.bufPrint(&path_buf, "{s}/transformer/diffusion_pytorch_model.safetensors.index.json", .{args.model});
    var dit_reg: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, dit_path);
    defer dit_reg.deinit();
    var dit_store: zml.io.TensorStore = .fromRegistry(allocator, &dit_reg);
    defer dit_store.deinit();

    // AutoencoderKLMiniMaxH3
    const vae_path = try std.fmt.bufPrint(&path_buf, "{s}/vae/diffusion_pytorch_model.safetensors.index.json", .{args.model});
    var vae_reg: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, vae_path);
    defer vae_reg.deinit();
    var vae_store: zml.io.TensorStore = .fromRegistry(allocator, &vae_reg);
    defer vae_store.deinit();

    var enc_model = try model.Encoder.init(allocator, enc_store.view());
    defer enc_model.deinit(allocator);
    var dit_model = try model.Dit.init(allocator, dit_store.view());
    defer dit_model.deinit(allocator);
    var visual = try model.VisualModel.init(allocator, vae_store.view());
    defer visual.deinit(allocator);

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    defer progress.end();

    //
    // Tokenizer
    //
    const tok_file = try repo.openFile(io, "tokenizer/tokenizer.json", .{});
    defer tok_file.close(io);
    var tok_reader = tok_file.reader(io, &.{});
    const tok_bytes = try tok_reader.interface.readAlloc(allocator, try tok_file.length(io));
    defer allocator.free(tok_bytes);
    var tokenizer = try zml.tokenizer.Tokenizer.fromBytes(allocator, tok_bytes);
    defer tokenizer.deinit();
    var tok_enc = try tokenizer.encoder();
    defer tok_enc.deinit();
    const tokens = try tok_enc.encodeAlloc(allocator, args.prompt);
    defer allocator.free(tokens);
    log.info("prompt tokens={d}", .{tokens.len});

    // One packed sequence: text tokens, then the video patch grid. Also the σ schedule.
    var packed_run = try model.pack(allocator, geo, @intCast(tokens.len), args.steps);
    defer packed_run.deinit(allocator);

    //
    // Compile kernels (weights load later, at run time)
    //
    const compile_start: std.Io.Timestamp = .now(io, .awake);
    var compiled_enc = try model.EncoderCompiled.compile(allocator, io, platform, enc_model, @intCast(tokens.len), shardings, &progress);
    defer compiled_enc.deinit();
    var compiled = try model.compileDit(
        allocator,
        io,
        platform,
        dit_model,
        geo,
        @intCast(tokens.len),
        packed_run.layout.seqLen(),
        @intCast(packed_run.video.stepCount()),
        shardings,
        enc_model.embed_tokens.weight.dtype(),
        &progress,
    );
    defer compiled.deinit();
    var compiled_vae: ?model.VaeCompiled = if (args.dit_only) null else try model.compileVae(allocator, io, platform, visual, geo, shardings, &progress);
    defer if (compiled_vae) |*c| c.deinit();
    log.info("compile all: ok [{f}]", .{compile_start.untilNow(io, .awake)});

    //
    // prompt → text hidden → DiT latents → pixels
    //
    var text = try compiled_enc.encodeText(allocator, io, platform, &enc_model, &enc_store, &all, tokens, &progress);
    defer text.deinit();
    const video_tokens = try model.denoise(
        allocator,
        io,
        platform,
        &compiled,
        &dit_model,
        &dit_store,
        &all,
        geo,
        text,
        @intCast(tokens.len),
        packed_run,
        args.seed,
        &progress,
    );
    defer allocator.free(video_tokens);
    if (args.dit_only) {
        log.info("dit-only: skip vae tokens={d}", .{video_tokens.len});
        return;
    }
    const thwc = try model.unpatchify(
        allocator,
        video_tokens,
        geo.latent_t,
        geo.latent_h,
        geo.latent_w,
        @intCast(dit_model.cfg.in_channels),
        dit_model.cfg.patch_size,
    );
    defer allocator.free(thwc);
    const rgb = try model.decodeVideo(allocator, io, platform, &compiled_vae.?, &visual, &vae_store, &all, geo, thwc, &progress);
    defer allocator.free(rgb);

    //
    // Write raw RGB24. ffmpeg muxes.
    //
    try writeRgb(allocator, io, out, geo, rgb);
}

fn writeRgb(allocator: std.mem.Allocator, io: std.Io, out: []const u8, geo: config.Geometry, rgb: []const f32) !void {
    var out_dir: std.Io.Dir = if (std.fs.path.isAbsolute(out)) blk: {
        var root = try std.Io.Dir.openDirAbsolute(io, std.fs.path.dirname(out).?, .{});
        defer root.close(io);
        try root.createDirPath(io, std.fs.path.basename(out));
        break :blk try root.openDir(io, std.fs.path.basename(out), .{});
    } else blk: {
        try std.Io.Dir.cwd().createDirPath(io, out);
        break :blk try std.Io.Dir.cwd().openDir(io, out, .{});
    };
    defer out_dir.close(io);

    // VAE output is NCHW planar [0,1]; write packed RGB24 frames.
    const plane = @as(usize, geo.frames) * geo.pixel_h * geo.pixel_w;
    const rgb8 = try allocator.alloc(u8, plane * 3);
    defer allocator.free(rgb8);
    var i: usize = 0;
    while (i < plane) : (i += 1) {
        rgb8[i * 3 + 0] = @intFromFloat(@round(std.math.clamp(rgb[i], 0, 1) * 255.0));
        rgb8[i * 3 + 1] = @intFromFloat(@round(std.math.clamp(rgb[plane + i], 0, 1) * 255.0));
        rgb8[i * 3 + 2] = @intFromFloat(@round(std.math.clamp(rgb[2 * plane + i], 0, 1) * 255.0));
    }
    {
        const file = try out_dir.createFile(io, "video.rgb", .{});
        defer file.close(io);
        var writer = file.writer(io, &.{});
        try writer.interface.writeAll(rgb8);
    }

    log.info("wrote {s}/video.rgb", .{out});
    log.info(
        "ffmpeg -y -f rawvideo -pix_fmt rgb24 -s {d}x{d} -r {d} -i {s}/video.rgb -an -pix_fmt yuv420p -c:v libx264 {s}/out.mp4",
        .{ geo.pixel_w, geo.pixel_h, @as(u32, @intFromFloat(config.video_fps)), out, out },
    );
}
