//! MiniMax-H3 text-to-video.
//!
//! tokenize → encode → pack → denoise → unpatchify → visual VAE + audio VAE → rgb + wav
//!
//! Weights (`--model`):
//!   text_encoder/   transformer/   vae/   audio_vae/

const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const config = @import("config.zig");
const ops = @import("ops.zig");
const encoder = @import("encoder.zig");
const pack = @import("pack.zig");
const dit = @import("dit.zig");
const vae = @import("vae.zig");
const audio = @import("audio.zig");

const log = std.log.scoped(.minimax_h3);

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,
    prompt: []const u8 = "A cinematic wide shot of waves at dusk.",
    seed: u64 = 0,
    out: []const u8 = "out",
    steps: u32 = 30,
    width: u32 = 1344,
    height: u32 = 768,
    duration: f32 = 5,

    pub const help =
        \\minimax_h3 --model=<path> [options]
        \\
        \\Prompt in, video.rgb + audio.wav out. Default 1344x768, 5s, 30 Euler steps.
        \\
        \\Options:
        \\  --model=<path>       Path to the MiniMax-H3 repository (required)
        \\  --prompt=<string>    Text prompt (default: a dusk waves shot)
        \\  --out=<dir>          Output directory (default: out)
        \\  --seed=<number>      Noise seed (default: 0)
        \\  --steps=<number>     Sigma points including terminal 0 (default: 30)
        \\  --width=<pixels>     Canvas width, multiple of 32 (default: 1344)
        \\  --height=<pixels>    Canvas height, multiple of 32 (default: 768)
        \\  --duration=<seconds> Clip length 5–15 (default: 5)
        \\
    ;
};

const Checkpoint = struct {
    reg: zml.safetensors.TensorRegistry,
    store: zml.io.TensorStore,

    fn open(self: *Checkpoint, allocator: std.mem.Allocator, io: std.Io, path: []const u8) !void {
        self.reg = try .fromPath(allocator, io, path);
        self.store = .fromRegistry(allocator, &self.reg);
    }

    fn deinit(self: *Checkpoint) void {
        self.store.deinit();
        self.reg.deinit();
    }
};

fn u8fromUnit(x: f32) u8 {
    return @intFromFloat(@round(std.math.clamp(x, 0, 1) * 255.0));
}

/// Visual VAE output is NCHW planar RGB in `[0, 1]`. Audio is interleaved stereo f32 in `[-1, 1]`.
fn writeOutputs(
    allocator: std.mem.Allocator,
    io: std.Io,
    out: []const u8,
    geo: config.Geometry,
    rgb: []const f32,
    pcm_f32: []const f32,
) !void {
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

    const plane = @as(usize, geo.frames) * geo.pixel_h * geo.pixel_w;
    const rgb8 = try allocator.alloc(u8, plane * 3);
    defer allocator.free(rgb8);
    for (0..plane) |i| {
        rgb8[i * 3 + 0] = u8fromUnit(rgb[i]);
        rgb8[i * 3 + 1] = u8fromUnit(rgb[plane + i]);
        rgb8[i * 3 + 2] = u8fromUnit(rgb[2 * plane + i]);
    }
    {
        const file = try out_dir.createFile(io, "video.rgb", .{});
        defer file.close(io);
        var writer = file.writer(io, &.{});
        try writer.interface.writeAll(rgb8);
    }
    log.info("wrote {s}/video.rgb", .{out});

    const pcm = try allocator.alloc(i16, pcm_f32.len);
    defer allocator.free(pcm);
    for (pcm, pcm_f32) |*d, s| {
        d.* = @intFromFloat(@round(std.math.clamp(s, -1.0, 1.0) * 32767.0));
    }
    {
        const file = try out_dir.createFile(io, "audio.wav", .{});
        defer file.close(io);
        var writer = file.writer(io, &.{});
        const data_bytes: u32 = @intCast(pcm.len * 2);
        const channels: u16 = 2;
        try writer.interface.writeAll("RIFF");
        try writer.interface.writeInt(u32, 36 + data_bytes, .little);
        try writer.interface.writeAll("WAVEfmt ");
        try writer.interface.writeInt(u32, 16, .little);
        try writer.interface.writeInt(u16, 1, .little);
        try writer.interface.writeInt(u16, channels, .little);
        try writer.interface.writeInt(u32, config.audio_sample_rate, .little);
        try writer.interface.writeInt(u32, config.audio_sample_rate * channels * 2, .little);
        try writer.interface.writeInt(u16, channels * 2, .little);
        try writer.interface.writeInt(u16, 16, .little);
        try writer.interface.writeAll("data");
        try writer.interface.writeInt(u32, data_bytes, .little);
        try writer.interface.writeAll(std.mem.sliceAsBytes(pcm));
    }
    log.info("wrote {s}/audio.wav", .{out});
    log.info(
        "ffmpeg -y -f rawvideo -pix_fmt rgb24 -s {d}x{d} -r {d} -i {s}/video.rgb -i {s}/audio.wav -pix_fmt yuv420p -c:v libx264 -c:a aac {s}/out.mp4",
        .{ geo.pixel_w, geo.pixel_h, @as(u32, @intFromFloat(config.video_fps)), out, out, out },
    );
}

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

    // =============================================================================
    // Platform
    // =============================================================================

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    const platform: *zml.Platform = try .auto(allocator, io, .{
        .physical_mesh = .{ .custom = config.Shardings.physicalMesh },
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false } } },
    });
    defer platform.deinit(allocator, io);
    log.info("\n{f}", .{platform.fmtVerbose()});

    const shardings: config.Shardings = try .init(platform);
    const geo = config.Geometry.init(args.width, args.height, args.duration) catch |err| switch (err) {
        error.InvalidCanvas => stdx.flags.fatal(
            "--width/--height must be positive multiples of {d} with area at most {d}",
            .{ config.canvas_multiple, config.canvas_max_pixels },
        ),
        error.InvalidDuration => stdx.flags.fatal(
            "--duration must be between {d:.0} and {d:.0} seconds",
            .{ config.min_duration_s, config.max_duration_s },
        ),
    };
    if (args.steps < 2) stdx.flags.fatal("--steps must be at least 2", .{});
    log.info("t2v  {d}x{d}  {d} frames ({d:.1}s)  audio_t={d}  {d} steps  seed {d}  devices={d}", .{
        geo.pixel_w,
        geo.pixel_h,
        geo.frames,
        @as(f32, @floatFromInt(geo.frames)) / config.video_fps,
        geo.audio_t,
        args.steps,
        args.seed,
        platform.devices.len,
    });

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    defer progress.end();
    const run = ops.Run.init(allocator, io, platform, shardings, &progress);

    // =============================================================================
    // 1. Tokenize + pack
    // =============================================================================

    const tok_file = try repo.openFile(io, "tokenizer/tokenizer.json", .{});
    defer tok_file.close(io);
    var tok_reader = tok_file.reader(io, &.{});
    const tok_bytes = try tok_reader.interface.readAlloc(allocator, try tok_file.length(io));
    defer allocator.free(tok_bytes);
    var tokenizer = try zml.tokenizer.Tokenizer.fromBytes(allocator, tok_bytes);
    defer tokenizer.deinit();
    var tok_enc = try tokenizer.encoder();
    defer tok_enc.deinit();
    const tokens = try tok_enc.encodeAlloc(allocator, std.mem.trimEnd(u8, args.prompt, "\r\n"));
    defer allocator.free(tokens);
    log.info("prompt tokens={d}", .{tokens.len});

    var packed_run = try pack.pack(allocator, geo, @intCast(tokens.len), args.steps);
    defer packed_run.deinit(allocator);

    // =============================================================================
    // Weights
    // =============================================================================

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;

    var enc_ckpt: Checkpoint = undefined;
    try enc_ckpt.open(allocator, io, try std.fmt.bufPrint(&path_buf, "{s}/text_encoder/model.safetensors.index.json", .{args.model}));
    defer enc_ckpt.deinit();

    var dit_ckpt: Checkpoint = undefined;
    try dit_ckpt.open(allocator, io, try std.fmt.bufPrint(&path_buf, "{s}/transformer/diffusion_pytorch_model.safetensors.index.json", .{args.model}));
    defer dit_ckpt.deinit();

    var vae_ckpt: Checkpoint = undefined;
    try vae_ckpt.open(allocator, io, try std.fmt.bufPrint(&path_buf, "{s}/vae/diffusion_pytorch_model.safetensors.index.json", .{args.model}));
    defer vae_ckpt.deinit();

    var audio_ckpt: Checkpoint = undefined;
    try audio_ckpt.open(allocator, io, try std.fmt.bufPrint(&path_buf, "{s}/audio_vae/diffusion_pytorch_model.safetensors", .{args.model}));
    defer audio_ckpt.deinit();

    var enc_model = try encoder.Encoder.init(allocator, enc_ckpt.store.view());
    defer enc_model.deinit(allocator);
    var dit_model = try dit.Dit.init(allocator, dit_ckpt.store.view());
    defer dit_model.deinit(allocator);
    var vae_model = try vae.Vae.init(allocator, vae_ckpt.store.view());
    defer vae_model.deinit(allocator);
    var audio_model = try audio.AudioVae.init(allocator, audio_ckpt.store.view());
    defer audio_model.deinit(allocator);

    // =============================================================================
    // Compile  (weights load later, at run time)
    // =============================================================================

    const compile_start: std.Io.Timestamp = .now(io, .awake);
    try enc_model.compile(&run, @intCast(tokens.len));
    try dit_model.compile(&run, geo, @intCast(tokens.len), packed_run, enc_model.embed_tokens.weight.dtype());
    try vae_model.compile(&run);
    try audio_model.compile(&run, geo);
    log.info("compile all: ok [{f}]", .{compile_start.untilNow(io, .awake)});

    // =============================================================================
    // 2–6. Encode → denoise → unpatchify → decode
    // =============================================================================

    var text = try enc_model.encodeText(&run, &enc_ckpt.store, tokens);
    defer text.deinit();
    const latents = try dit_model.denoise(&run, &dit_ckpt.store, geo, text, @intCast(tokens.len), packed_run, args.seed);
    defer latents.deinit(allocator);
    const thwc = try pack.unpatchify(
        allocator,
        latents.video,
        geo.latent_t,
        geo.latent_h,
        geo.latent_w,
        @intCast(dit_model.cfg.in_channels),
        dit_model.cfg.patch_size,
    );
    defer allocator.free(thwc);
    const rgb = try vae_model.decodeVideo(&run, &vae_ckpt.store, geo, thwc);
    defer allocator.free(rgb);
    const pcm_f32 = try audio_model.decodeAudio(&run, &audio_ckpt.store, geo, latents.audio);
    defer allocator.free(pcm_f32);

    try writeOutputs(allocator, io, out, geo, rgb, pcm_f32);
}
