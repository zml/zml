//! MiniMax-H3 text-to-video.
//!
//! tokenize → encode → pack → denoise → visual VAE + audio VAE → rgb + wav
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
const mode = @import("mode.zig");
const vision = @import("vision.zig");
const visual_enc = @import("visual_encoder.zig");

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
    first_frame: []const u8 = "",
    last_frame: []const u8 = "",
    refs: []const u8 = "",

    pub const help =
        \\minimax_h3 --model=<path> [options]
        \\
        \\Prompt in, video.rgb + audio.wav out. Default 1344x768, 5s, 30 Euler steps.
        \\
        \\  text-to-video          --prompt=...
        \\  image-to-video        --first-frame=first.png
        \\  last-frame            --last-frame=last.png
        \\  first-and-last-frame  --first-frame=a.png --last-frame=b.png
        \\  reference-to-video    --refs=a.png,b.png
        \\  video/audio refs      --refs=clip.mp4,ref.mp3
        \\
        \\Options:
        \\  --model=<path>       Path to the MiniMax-H3 repository (required)
        \\  --prompt=<string>    Text prompt (default: a dusk waves shot)
        \\  --first-frame=<path> First-frame image
        \\  --last-frame=<path>  Last-frame image
        \\  --refs=<paths>       Comma-separated images, videos, or audio
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
    const cfgs = try config.load(allocator, io, repo);
    const platform: *zml.Platform = try .auto(allocator, io, .{
        .physical_mesh = .{ .custom = config.Shardings.physicalMesh },
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false } } },
    });
    defer platform.deinit(allocator, io);
    try vision.register(platform);
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
    if (args.steps < 2) stdx.flags.fatal("--steps must be at least 2", .{});
    const mode_name = if (args.refs.len != 0) "ref2v" else if (args.first_frame.len != 0 or args.last_frame.len != 0) "i2v" else "t2v";
    log.info("{s}  {d}x{d}  {d} frames ({d:.1}s)  audio_t={d}  {d} steps  seed {d}  devices={d}", .{
        mode_name,
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
    const vis_cfg = if (args.first_frame.len != 0 or args.last_frame.len != 0 or args.refs.len != 0)
        try vision.configFromRepo(allocator, io, repo, cfgs.encoder.hidden_size)
    else
        vision.Config{};
    const planned = try mode.plan(allocator, io, &tok_enc, args.prompt, args.first_frame, args.last_frame, args.refs, vis_cfg, geo, cfgs.vae, cfgs.audio);
    defer if (planned) |p| p.deinit(allocator);
    const tokens = if (planned) |p| p.tokens else try tok_enc.encodeAlloc(allocator, std.mem.trimEnd(u8, args.prompt, "\r\n"));
    defer if (planned == null) allocator.free(tokens);
    log.info("prompt tokens={d}", .{tokens.len});

    var conds: []pack.CondClip = &[_]pack.CondClip{};
    defer if (planned != null) allocator.free(conds);
    var audios: []pack.CondAudio = &.{};
    defer if (planned != null) allocator.free(audios);
    if (planned) |p| {
        conds = try p.clips(allocator, geo, cfgs.vae.spatial());
        audios = try p.audioClips(allocator);
    }
    const ref_blocks = if (planned) |p| p.refs else &[_]pack.RefBlock{};
    var packed_run = if (planned != null)
        try pack.packCond(allocator, geo, @intCast(tokens.len), args.steps, cfgs.video_shift, cfgs.audio_shift, conds, audios, ref_blocks, planned.?.tags)
    else
        try pack.pack(allocator, geo, @intCast(tokens.len), args.steps, cfgs.video_shift, cfgs.audio_shift);
    defer packed_run.deinit(allocator);
    var dit_geo = geo;
    dit_geo.video_tokens = packed_run.layout.video_len;
    dit_geo.audio_tokens = packed_run.layout.audio_len;
    if (planned != null) {
        log.info("pack cond_video={d} cond_audio={d} refs={d} seq={d} video_tokens={d} audio_tokens={d}", .{
            packed_run.layout.cond_video_len,
            packed_run.layout.cond_audio_len,
            ref_blocks.len,
            packed_run.layout.seqLen(),
            packed_run.layout.video_len,
            packed_run.layout.audio_len,
        });
    }

    // =============================================================================
    // Weights
    // =============================================================================

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;

    var enc_ckpt: Checkpoint = undefined;
    try enc_ckpt.open(allocator, io, try std.fmt.bufPrint(&path_buf, "{s}/text_encoder/model.safetensors.index.json", .{args.model}));
    defer enc_ckpt.deinit();

    var dit_ckpt: Checkpoint = undefined;
    try dit_ckpt.open(allocator, io, try std.fmt.bufPrint(&path_buf, "{s}/{s}/diffusion_pytorch_model.safetensors.index.json", .{
        args.model,
        if (planned != null and planned.?.ref2va) "transformer_ref" else "transformer",
    }));
    defer dit_ckpt.deinit();

    var vae_ckpt: Checkpoint = undefined;
    try vae_ckpt.open(allocator, io, try std.fmt.bufPrint(&path_buf, "{s}/vae/diffusion_pytorch_model.safetensors.index.json", .{args.model}));
    defer vae_ckpt.deinit();

    var audio_ckpt: Checkpoint = undefined;
    try audio_ckpt.open(allocator, io, try std.fmt.bufPrint(&path_buf, "{s}/audio_vae/diffusion_pytorch_model.safetensors", .{args.model}));
    defer audio_ckpt.deinit();

    var enc_model = try encoder.Encoder.init(allocator, enc_ckpt.store.view(), cfgs.encoder);
    defer enc_model.deinit(allocator);
    var dit_model = try dit.Dit.init(allocator, dit_ckpt.store.view(), cfgs.dit);
    defer dit_model.deinit(allocator);
    var vae_model = try vae.Vae.init(allocator, vae_ckpt.store.view(), cfgs.vae);
    defer vae_model.deinit(allocator);
    var audio_model = try audio.AudioVae.init(allocator, audio_ckpt.store.view(), cfgs.audio);
    defer audio_model.deinit(allocator);

    // Visual / audio encode first so Qwen+VAE-enc kernels are gone before DiT compile.
    var encoded_mode: ?mode.Encoded = null;
    defer if (encoded_mode) |*e| e.deinit(allocator);
    if (planned) |p| {
        var vis_loaded = try vision.LoadedModel.init(allocator, enc_ckpt.store.view(), vis_cfg);
        defer vis_loaded.deinit(allocator);
        var vis_cache = try vision.WeightCache.load(&run, &vis_loaded, &enc_ckpt.store);
        defer vis_cache.deinit(allocator);
        const ve_loaded = visual_enc.LoadedModel.init(vae_ckpt.store.view(), cfgs.vae);
        var ve_compiled = try visual_enc.compile(&run, ve_loaded.inner);
        defer ve_compiled.deinit();
        if (p.hasVideo()) try visual_enc.compileClip(&run, &ve_compiled, ve_loaded.inner, cfgs.vae);
        var ve_bufs = try ve_loaded.loadBuffers(&run, &vae_ckpt.store);
        defer zml.Buffer.deinitAll(visual_enc.Model, &ve_bufs);
        var audio_enc: ?audio.EncoderModel = null;
        var audio_bufs: ?zml.Bufferized(audio.EncoderModel) = null;
        defer if (audio_bufs) |*b| audio.EncoderModel.unloadBuffers(b);
        if (p.audios.len != 0) {
            audio_enc = audio.EncoderModel.init(audio_ckpt.store.view(), cfgs.audio);
            audio_bufs = try ops.load(&run, &audio_ckpt.store, audio.EncoderModel, &audio_enc.?, null);
        }
        encoded_mode = try mode.encode(&run, geo, cfgs.vae, p, &vis_loaded, &vis_cache, &ve_compiled, &ve_bufs, audio_enc, if (audio_bufs) |*b| b else null);
        packed_run.cond_video = encoded_mode.?.patches;
        packed_run.cond_audio = encoded_mode.?.audio_patches;
    }

    // =============================================================================
    // Compile  (weights load later, at run time)
    // =============================================================================

    const compile_start: std.Io.Timestamp = .now(io, .awake);
    try enc_model.compile(&run, @intCast(tokens.len));
    log.info("compile encoder: ok [{f}]", .{compile_start.untilNow(io, .awake)});

    var text = if (encoded_mode) |e|
        try enc_model.encodeTextVision(&run, &enc_ckpt.store, .{
            .tokens = tokens,
            .spans = planned.?.spans,
            .merged = e.merged,
            .deepstack = .{ e.deepstack[0], e.deepstack[1], e.deepstack[2] },
        })
    else
        try enc_model.encodeText(&run, &enc_ckpt.store, tokens);
    defer text.deinit();
    log.info("encode text: ok tokens={d}", .{tokens.len});
    enc_model.dropCompiled();

    const dit_start: std.Io.Timestamp = .now(io, .awake);
    try dit_model.compile(&run, dit_geo, @intCast(tokens.len), packed_run, enc_model.embed_tokens.weight.dtype());
    log.info("compile dit: ok [{f}]", .{dit_start.untilNow(io, .awake)});

    var latents = try dit_model.denoise(&run, &dit_ckpt.store, dit_geo, text, @intCast(tokens.len), packed_run, args.seed);
    defer latents.deinit();
    dit_model.dropCompiled();

    const vae_start: std.Io.Timestamp = .now(io, .awake);
    try vae_model.compile(&run, geo, dit_model.cfg.patch_size);
    try audio_model.compile(&run, geo);
    log.info("compile vae+audio: ok [{f}]", .{vae_start.untilNow(io, .awake)});

    const vae_loaded = try vae_model.startLoad(&run, &vae_ckpt.store);
    defer vae_loaded.deinit(allocator, io);
    const audio_loaded = try audio_model.startLoad(&run, &audio_ckpt.store);
    defer audio_loaded.deinit(allocator, io);

    const rgb = try vae_model.decodeVideo(&run, geo, latents.video, vae_loaded);
    defer allocator.free(rgb);
    const pcm_f32 = try audio_model.decodeAudio(&run, latents.audio, audio_loaded);
    defer allocator.free(pcm_f32);

    try writeOutputs(allocator, io, out, geo, rgb, pcm_f32, cfgs.audio.sampling_rate);
}
