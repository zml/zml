//! MiniMax-H3 text-to-video (768P, video only).
//!
//! tokenize → encode → pack → denoise → unpatchify → VAE → rgb
//!
//! Weights (`--model`): HuggingFace `MiniMaxAI/MiniMax-H3`
//!   text_encoder/   transformer/   vae/

const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const config = @import("config.zig");
const ops = @import("ops.zig");
const encoder = @import("encoder.zig");
const pack = @import("pack.zig");
const dit = @import("dit.zig");
const vae = @import("vae.zig");

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
    const geo = config.geo;
    log.info("t2v  {d}x{d}  {d:.1}s  {d} steps  seed {d}  devices={d}", .{
        geo.pixel_w, geo.pixel_h, config.duration_s, args.steps, args.seed, platform.devices.len,
    });

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    defer progress.end();
    const run = ops.Run.init(allocator, io, platform, shardings, &progress);

    // =============================================================================
    // Checkpoints  (HuggingFace MiniMax-H3 layout)
    // =============================================================================

    var enc_ckpt: ops.Checkpoint = undefined;
    try enc_ckpt.open(allocator, io, args.model, "text_encoder/model.safetensors.index.json");
    defer enc_ckpt.deinit();
    var dit_ckpt: ops.Checkpoint = undefined;
    try dit_ckpt.open(allocator, io, args.model, "transformer/diffusion_pytorch_model.safetensors.index.json");
    defer dit_ckpt.deinit();
    var vae_ckpt: ops.Checkpoint = undefined;
    try vae_ckpt.open(allocator, io, args.model, "vae/diffusion_pytorch_model.safetensors.index.json");
    defer vae_ckpt.deinit();

    var enc_model = try encoder.Encoder.init(allocator, enc_ckpt.view());
    defer enc_model.deinit(allocator);
    var dit_model = try dit.Dit.init(allocator, dit_ckpt.view());
    defer dit_model.deinit(allocator);
    var vae_model = try vae.Vae.init(allocator, vae_ckpt.view());
    defer vae_model.deinit(allocator);

    // =============================================================================
    // 1. Tokenize
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
    const tokens = try tok_enc.encodeAlloc(allocator, args.prompt);
    defer allocator.free(tokens);
    log.info("prompt tokens={d}", .{tokens.len});

    var packed_run = try pack.pack(allocator, geo, @intCast(tokens.len), args.steps);
    defer packed_run.deinit(allocator);

    // =============================================================================
    // Compile  (weights load later, at run time)
    // =============================================================================

    const compile_start: std.Io.Timestamp = .now(io, .awake);
    try enc_model.compile(&run, @intCast(tokens.len));
    try dit_model.compile(&run, geo, @intCast(tokens.len), packed_run, enc_model.embed_tokens.weight.dtype());
    if (!args.dit_only) try vae_model.compile(&run);
    log.info("compile all: ok [{f}]", .{compile_start.untilNow(io, .awake)});

    // =============================================================================
    // 2–6. Encode → denoise → unpatchify → decode
    // =============================================================================

    var text = try enc_model.encodeText(&run, &enc_ckpt.store, tokens);
    defer text.deinit();
    const video_tokens = try dit_model.denoise(&run, &dit_ckpt.store, geo, text, @intCast(tokens.len), packed_run, args.seed);
    defer allocator.free(video_tokens);
    if (args.dit_only) {
        log.info("dit-only: skip vae tokens={d}", .{video_tokens.len});
        return;
    }
    const thwc = try pack.unpatchify(
        allocator,
        video_tokens,
        geo.latent_t,
        geo.latent_h,
        geo.latent_w,
        @intCast(dit_model.cfg.in_channels),
        dit_model.cfg.patch_size,
    );
    defer allocator.free(thwc);
    const rgb = try vae_model.decodeVideo(&run, &vae_ckpt.store, geo, thwc);
    defer allocator.free(rgb);

    try writeRgb(allocator, io, out, geo, rgb);
}

/// VAE output is NCHW planar RGB in `[0, 1]`. Write packed RGB24 frames.
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
