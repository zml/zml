//! MiniMax-H3 and Pi-H3 text-to-video with audio.
//!
//! Pi-H3 uses a pruned --transformer plus a matching --adapter.
//! It reuses the base tokenizer, text encoder, audio VAE and configs,
//! including vae/config.json, but does not load video VAE weights
//! or run the video VAE decoder.

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
const adapter = @import("adapter.zig");
const dwd = @import("dwd.zig");

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
    variant: pack.Variant = .h3,
    transformer: ?[]const u8 = null,
    adapter: ?[]const u8 = null,
    width: u32 = 1344,
    height: u32 = 768,
    duration: f32 = 5,

    pub const help =
        \\minimax_h3 --model=<path> [options]
        \\
        \\Prompt in, video.rgb + audio.wav out. Default 1344x768, 5s, 30 sigma points (29 steps) for H3.
        \\
        \\Options:
        \\  --model=<path>       Path to the MiniMax-H3 repository (required)
        \\  --prompt=<string>    Text prompt (default: a dusk waves shot)
        \\  --out=<dir>          Output directory (default: out)
        \\  --seed=<number>      Noise seed (default: 0)
        \\  --steps=<number>     Sigma points including terminal 0 (default: 30, H3 only)
        \\  --variant=<name>     h3 (default), pi2 (2-step), or pi4 (4-step)
        \\  --transformer=<dir>  Pinned pruned transformer directory (Pi-H3 only)
        \\  --adapter=<dir>      Matching standard DWD adapter directory (Pi-H3 only)
        \\  --width=<pixels>     Canvas width, multiple of 32 (default: 1344)
        \\  --height=<pixels>    Canvas height, multiple of 32 (default: 768)
        \\  --duration=<seconds> Clip length 5–15 (default: 5)
        \\
    ;
};

fn validateArgs(args: Args) !void {
    if (args.variant == .h3) {
        if (args.transformer != null or args.adapter != null) return error.PiFlagsRequireNamedVariant;
        if (args.steps < 2) return error.TooFewSigmaPoints;
    } else {
        if (args.transformer == null or args.adapter == null) return error.PiRequiresTransformerAndAdapter;
    }
}

/// Allocator statistics, unlike nvidia-smi, exclude unused BFC pool regions.
fn logPeakMemory(platform: *const zml.Platform) void {
    for (platform.devices, 0..) |device, i| {
        const stats = device.memoryStats();
        log.info("device {d} memory: peak in use {d} MiB, peak pool {d} MiB", .{
            i,
            (stats.peak_bytes_in_use orelse 0) >> 20,
            (stats.peak_pool_bytes orelse 0) >> 20,
        });
    }
}

const Checkpoint = struct {
    reg: zml.safetensors.TensorRegistry,
    store: zml.io.TensorStore,

    fn open(self: *Checkpoint, allocator: std.mem.Allocator, io: std.Io, path: []const u8, meshes: []const *const zml.Mesh) !void {
        self.reg = try .fromPath(allocator, io, path);
        self.store = .fromRegistry(allocator, &self.reg, meshes);
    }

    fn openPi(self: *Checkpoint, allocator: std.mem.Allocator, io: std.Io, transformer_dir: []const u8, adapter_dir: []const u8, meshes: []const *const zml.Mesh) !void {
        self.reg = try adapter.open(allocator, io, transformer_dir, adapter_dir);
        self.store = .fromRegistry(allocator, &self.reg, meshes);
    }

    fn deinit(self: *Checkpoint) void {
        self.store.deinit();
        self.reg.deinit();
    }
};

const VideoDecoder = union(enum) {
    h3: vae.Vae,
    pi: dwd.Dwd,

    fn deinit(self: *VideoDecoder, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .h3 => |*model| model.deinit(allocator),
            .pi => |*model| model.deinit(),
        }
    }
};

fn u8fromUnit(x: f32) u8 {
    return @intFromFloat(@round(std.math.clamp(x, 0, 1) * 255.0));
}

/// Video is planar RGB (C, F, H, W) (channels, frames, height, width) in `[0, 1]`.
/// Audio is interleaved stereo f32 in `[-1, 1]`.
fn writeOutputs(
    allocator: std.mem.Allocator,
    io: std.Io,
    out: []const u8,
    geo: config.Geometry,
    rgb: []const f32,
    pcm_f32: []const f32,
    sample_rate: u32,
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
        try writer.interface.writeInt(u32, sample_rate, .little);
        try writer.interface.writeInt(u32, sample_rate * channels * 2, .little);
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
    validateArgs(args) catch |err| stdx.flags.fatal("invalid variant options: {s} (see --help)", .{@errorName(err)});

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
    defer repo.close(io);
    var pruned_dir = if (args.transformer) |p| try std.Io.Dir.cwd().openDir(io, p, .{}) else null;
    defer if (pruned_dir) |*dir| dir.close(io);
    const cfgs = try config.loadSelected(allocator, io, repo, pruned_dir);
    const platform: *zml.Platform = try .auto(allocator, io, .{
        .physical_mesh = .{ .custom = config.Shardings.physicalMesh },
        .cuda = .{ .allocator = .{ .bfc = .{ .preallocate = false } } },
        .rocm = .{ .allocator = .{ .bfc = .{ .preallocate = false } } },
    });
    defer platform.deinit(allocator, io);
    defer logPeakMemory(platform);
    log.info("\n{f}", .{platform.fmtVerbose()});

    const shardings: config.Shardings = try .init(platform);
    const geo = config.Geometry.init(args.width, args.height, args.duration, cfgs.dit, cfgs.vae) catch |err| switch (err) {
        error.InvalidCanvas => stdx.flags.fatal(
            "--width/--height must be positive multiples of {d} with area at most {d}",
            .{ config.canvas_multiple, config.canvas_max_pixels },
        ),
        error.InvalidDuration => stdx.flags.fatal(
            "--duration must be between {d:.0} and {d:.0} seconds",
            .{ config.min_duration_s, config.max_duration_s },
        ),
    };
    const schedule: pack.ScheduleSpec = switch (args.variant) {
        .h3 => .{ .h3 = .{ .steps = args.steps, .video_shift = cfgs.video_shift, .audio_shift = cfgs.audio_shift } },
        .pi2 => .pi2,
        .pi4 => .pi4,
    };
    log.info("{s} t2v  {d}x{d}  {d} frames ({d:.1}s)  audio_t={d}  {d} evaluations  seed {d}  devices={d}", .{
        @tagName(args.variant),
        geo.pixel_w,
        geo.pixel_h,
        geo.frames,
        @as(f32, @floatFromInt(geo.frames)) / config.video_fps,
        geo.audio_t,
        schedule.evaluations(),
        args.seed,
        platform.devices.len,
    });

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    defer progress.end();
    const run = ops.Run.init(allocator, io, platform, shardings, &progress);

    // =============================================================================
    // Tokenize + pack
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

    var packed_run: pack.Packed = try .init(
        allocator,
        geo,
        @intCast(tokens.len),
        schedule,
    );
    defer packed_run.deinit(allocator);

    // =============================================================================
    // Weights
    // =============================================================================

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;

    var enc_ckpt: Checkpoint = undefined;
    try enc_ckpt.open(allocator, io, try std.fmt.bufPrint(&path_buf, "{s}/text_encoder/model.safetensors.index.json", .{args.model}), &.{shardings.model});
    defer enc_ckpt.deinit();

    var audio_ckpt: Checkpoint = undefined;
    try audio_ckpt.open(allocator, io, try std.fmt.bufPrint(&path_buf, "{s}/audio_vae/diffusion_pytorch_model.safetensors", .{args.model}), &.{shardings.model});
    defer audio_ckpt.deinit();

    var enc_model = try encoder.Encoder.init(allocator, enc_ckpt.store.view(), cfgs.encoder);
    defer enc_model.deinit(allocator);
    var audio_model = try audio.AudioVae.init(allocator, audio_ckpt.store.view(), cfgs.audio);
    defer audio_model.deinit(allocator);

    var dit_ckpt: Checkpoint = undefined;
    switch (args.variant) {
        .h3 => try dit_ckpt.open(
            allocator,
            io,
            try std.fmt.bufPrint(&path_buf, "{s}/transformer/diffusion_pytorch_model.safetensors.index.json", .{args.model}),
            &.{shardings.model},
        ),
        .pi2, .pi4 => try dit_ckpt.openPi(
            allocator,
            io,
            args.transformer.?,
            args.adapter.?,
            &.{shardings.model},
        ),
    }
    defer dit_ckpt.deinit();

    var dit_model = try dit.Dit.init(allocator, dit_ckpt.store.view(), cfgs.dit);
    defer dit_model.deinit(allocator);

    var vae_ckpt: Checkpoint = undefined;
    if (args.variant == .h3) {
        try vae_ckpt.open(
            allocator,
            io,
            try std.fmt.bufPrint(&path_buf, "{s}/vae/diffusion_pytorch_model.safetensors.index.json", .{args.model}),
            &.{shardings.model},
        );
    }
    defer if (args.variant == .h3) vae_ckpt.deinit();

    var video_decoder: VideoDecoder = switch (args.variant) {
        .h3 => .{ .h3 = try vae.Vae.init(allocator, vae_ckpt.store.view(), cfgs.vae) },
        .pi2, .pi4 => .{ .pi = try dwd.Dwd.init(dit_ckpt.store.view()) },
    };
    defer video_decoder.deinit(allocator);

    // =============================================================================
    // Compile  (weights load later, at run time)
    // =============================================================================

    const compile_start: std.Io.Timestamp = .now(io, .awake);
    try enc_model.compile(&run, @intCast(tokens.len));
    try dit_model.compile(&run, geo, @intCast(tokens.len), packed_run, enc_model.embed_tokens.weight.dtype());
    switch (args.variant) {
        .h3 => try video_decoder.h3.compile(&run, geo, dit_model.cfg.patch_size),
        .pi2, .pi4 => try video_decoder.pi.compile(&run, geo),
    }
    try audio_model.compile(&run, geo);
    log.info("compile all: ok [{f}]", .{compile_start.untilNow(io, .awake)});

    // =============================================================================
    // Encode → denoise → decode
    // =============================================================================

    const vae_loaded = if (args.variant == .h3)
        try video_decoder.h3.startLoad(&run, &vae_ckpt.store)
    else
        null;
    defer if (vae_loaded) |v| v.deinit(allocator, io);

    const audio_loaded = try audio_model.startLoad(&run, &audio_ckpt.store);
    defer audio_loaded.deinit(allocator, io);

    const text_start: std.Io.Timestamp = .now(io, .awake);
    var text = try enc_model.encodeText(&run, &enc_ckpt.store, tokens);
    defer text.deinit();
    log.info("encode text (including weight loads): ok [{f}]", .{text_start.untilNow(io, .awake)});

    var result = try dit_model.denoise(&run, &dit_ckpt.store, geo, text, @intCast(tokens.len), packed_run, args.seed);
    defer result.deinit();

    const rgb = switch (video_decoder) {
        .h3 => |*model| try model.decodeVideo(&run, geo, result.video, vae_loaded.?),
        .pi => |*model| blk: {
            const head_start: std.Io.Timestamp = .now(io, .awake);
            const pixels = try model.decode(&run, &dit_ckpt.store, result.video);
            log.info("DWD decode (including weight loads and transfer): ok [{f}]", .{head_start.untilNow(io, .awake)});
            break :blk pixels;
        },
    };
    defer allocator.free(rgb);

    const pcm_f32 = try audio_model.decodeAudio(&run, result.audio, audio_loaded);
    defer allocator.free(pcm_f32);

    try writeOutputs(allocator, io, out, geo, rgb, pcm_f32, cfgs.audio.sampling_rate);
}
