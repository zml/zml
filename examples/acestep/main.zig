const std = @import("std");
const log = std.log;
const zml = @import("zml");
const stdx = zml.stdx;

const acellm_ = @import("acellm.zig");
const aceemb_ = @import("aceemb.zig");
const aceenc_ = @import("aceenc.zig");
const acedit_ = @import("acedit.zig");
const acevae_ = @import("acevae.zig");
const inference = @import("inference.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub const Shardings = struct {
    replicated: zml.sharding.Sharding,
    model: zml.sharding.Sharding,

    pub fn init(platform: *zml.Platform) !Shardings {
        const model_mesh: zml.sharding.LogicalMesh = try .init("model", .{ .model = .high_bandwidth });
        const model_sharding_strategy: zml.sharding.Strategy = try .suggest(model_mesh, platform.physical_mesh);
        return .{
            .replicated = try zml.sharding.replicatedSharding(platform),
            .model = try .initFromStrategy(platform, model_mesh, model_sharding_strategy),
        };
    }

    pub fn all(self: *const Shardings) [2]zml.sharding.Sharding {
        return .{ self.replicated, self.model };
    }
};

pub const Zml_handler = struct {
    allocator: std.mem.Allocator,
    arena: *std.heap.ArenaAllocator,
    platform: *zml.Platform,
    uris: Uri_handler,
    io: std.Io,
    local_io: std.Io,
    progress: std.Progress.Node,
    args: Args,
    timers: Timing_handler,

    pub fn fromInit(init: std.process.Init, io: std.Io) !Zml_handler {
        if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
            var working_dir = try std.Io.Dir.openDirAbsolute(init.io, build_working_directory, .{});
            defer working_dir.close(init.io);
            try std.process.setCurrentDir(init.io, working_dir);
        }
        const platform = try zml.Platform.auto(init.gpa, io, .{});
        errdefer platform.deinit(init.gpa);

        const args = stdx.flags.parse(init.minimal.args, Args);
        
        return .{
            .allocator = init.gpa,
            .arena = init.arena,
            .platform = platform,
            .uris = if (args.local_files) .fromLocal(args) else .fromHf(args),
            .io = io,
            .local_io = init.io,
            .progress = std.Progress.start(io, .{}),
            .args = args,
            .timers = .{},
        };
    }

    pub fn deinit(self: *Zml_handler) void {
        self.progress.end();
        self.platform.deinit(self.allocator);
    }

    pub fn tic(self: *Zml_handler, target: *Timing_handler.Module_timer.Field_timer) void {
        target.init = std.Io.Timestamp.now(self.io, .awake);
    }
    
    pub fn toc(self: *Zml_handler, target: *Timing_handler.Module_timer.Field_timer) void {
        const end = std.Io.Timestamp.now(self.io, .awake);
        const start: std.Io.Timestamp = target.init;
        const duration = std.Io.Timestamp.durationTo(start, end);
        target.nanoseconds += duration.nanoseconds;
    }
    
};

pub const Uri_handler = struct {
    acellm: []const u8,
    aceemb: []const u8,
    acedit: []const u8,
    is_xl: bool,
    acevae: []const u8,
    silence: []const u8 = "file://acestep//models//acestep-v15-turbo",

    pub fn fromLocal(args: Args) Uri_handler {
        return .{
            .acellm = if (args.llm_size == 0) "file://acestep//models//acestep-5Hz-lm-0.6B"
                             else if (args.llm_size == 1) "file://acestep//models//acestep-5Hz-lm-1.7B"
                                                     else "file://acestep//models//acestep-5Hz-lm-4B",
            .aceemb = "file://acestep//models//Qwen3-Embedding-0.6B",
            .acedit = if (args.dit_size == 0) "file://acestep//models//acestep-v15-turbo"
                                                     else "file://acestep//models//acestep-v15-xl-turbo",
            .is_xl = args.dit_size == 1,
            .acevae = "file://acestep//models//Oobleck-vae",
        };
    }

    pub fn fromHf(args: Args) Uri_handler {
        // official default models are in hf://ACE-Step/Ace-Step1.5/
        // additional models are in hf://ACE-Step/
        return .{
            .acellm = if (args.llm_size == 0) "hf://ACE-Step/acestep-5Hz-lm-0.6B"
                             else if (args.llm_size == 1) "hf://ACE-Step/Ace-Step1.5/acestep-5Hz-lm-1.7B"
                                                     else "hf://ACE-Step/acestep-5Hz-lm-4B",
            .aceemb = "hf://ACE-Step/Ace-Step1.5/Qwen3-Embedding-0.6B",
            .acedit = if (args.dit_size == 0) "hf://ACE-Step/Ace-Step1.5/acestep-v15-turbo"
                                                     else "hf://ACE-Step/acestep-v15-turbo-xl",
            .is_xl = args.dit_size == 1,
            .acevae = "hf://ACE-Step/Ace-Step1.5/vae/",
        };
    }

};

pub const Timing_handler = struct {
    
    pub const Module_timer = struct {
        
        pub const Field_timer = struct {
            nanoseconds: i96 = 0,
            init: std.Io.Timestamp = std.Io.Timestamp.zero,
        };
        
        init: Field_timer = .{},
        load: Field_timer = .{},
        compile: Field_timer = .{},
        prefill: Field_timer = .{},
        decode: Field_timer = .{},
        total: Field_timer = .{},
        
        pub fn print(self: Module_timer, name: []const u8) void {
            std.log.info("{s}  {d:>6.2}s  {d:>6.2}s  {d:>6.2}s  {d:>6.2}s  {d:>6.2}s  {d:>6.2}s", .{
                name,
                @as(f64, @floatFromInt(self.init.nanoseconds)) / 1e9,
                @as(f64, @floatFromInt(self.compile.nanoseconds)) / 1e9,
                @as(f64, @floatFromInt(self.load.nanoseconds)) / 1e9,
                @as(f64, @floatFromInt(self.prefill.nanoseconds)) / 1e9,
                @as(f64, @floatFromInt(self.decode.nanoseconds)) / 1e9,
                @as(f64, @floatFromInt(self.total.nanoseconds)) / 1e9,
            });
        }
    };
    
    llm: Module_timer = .{},
    cfg: Module_timer = .{},
    emb: Module_timer = .{},
    enc: Module_timer = .{},
    dit: Module_timer = .{},
    vae: Module_timer = .{},

    wav: Module_timer.Field_timer = .{},
    total: Module_timer.Field_timer = .{},

    pub fn print(self: Timing_handler) void {
        std.log.info("Module    init  compile     load  prefill   decode    total", .{});
        self.llm.print("  llm");
        self.cfg.print("  cfg");
        self.emb.print("  emb");
        self.enc.print("  enc");
        self.dit.print("  dit");
        self.vae.print("  vae");
        std.log.info("  wav                                               {d:>6.2}s", .{@as(f64, @floatFromInt(self.wav.nanoseconds)) / 1e9});
        std.log.info("total                                               {d:>6.2}s", .{@as(f64, @floatFromInt(self.total.nanoseconds)) / 1e9});
    }
};


const Args = struct {
    prompt: []const u8,
    instru: bool = false,
    local_files: bool = false,
    llm_size: u8 = 0,
    dit_size: u8 = 0,
    skip_cfg: bool = false,
    duration: i64 = -1,
    n: u8 = 1,

    pub const help =
        \\ Use acestep --prompt=<...> [options]
        \\
        \\ Run audio generation with the selected models.
        \\
        \\ Options:
        \\   --prompt=<string>     Prompt to use for generation (required)
        \\   --instru              Ask for an instrumental audio
        \\   --local-files         Optional, use local model paths, see README to setup
        \\   --llm_size=<int>      Size of the 5Hz LLM (0/1/2 for 0.6B/1.7B/4B, default: 0)
        \\   --dit_size=<int>      Size of the DiT model (0/1 for turbo/turbo-xl, default: 0)
        \\   --skip-cfg            Optional, disable CFG phase
        \\   --duration=<number>   Constrains the duration in seconds (required if CFG is disabled, default: -1)
        \\   --n=<int>             Number of audio files to generate with different seeds for the diffusion (default: 1)
        \\
    ;
};

// 4090
// --prompt='a peak-time dark techno track'
// --llm-size=2 --dit-size=1
// --duration=180 --n=3
// bazel run --config=release acestep --//platforms:cuda=true -- --instru --local-files 
// info: Module    init  compile     load  prefill   decode    total
// info:   llm    0.27s    1.81s    0.88s    0.09s    2.05s    5.10s
// info:   cfg    0.00s    0.79s    0.00s    0.13s   21.53s   22.45s
// info:   emb    0.00s    1.26s    0.55s    0.00s    0.00s    2.03s
// info:   enc    0.00s    2.74s    0.60s    0.01s    0.00s    3.35s
// info:   dit    0.00s    2.07s    0.81s    4.44s    0.00s    7.36s
// info:   vae    0.00s    1.40s    0.56s    3.97s    0.00s    5.96s
// info:   wav                                                 9.86s
// info: total                                                56.18s

// TODO: why does it goes oom during diffusion ??
// TODO: accelerate wav export, have a look at cfg, microtune vae decode_t
// TODO: reference audio
// TODO: reference timbre

// TODO: move model related code from inference to Exes struct inside models

pub fn main(init: std.process.Init) !void {
    var http_client: std.http.Client = .{ .allocator = init.gpa, .io = init.io };
    defer http_client.deinit();

    var vfs_file: zml.io.VFS.File = .init(init.gpa, init.io, .{});
    defer vfs_file.deinit();

    var hf_vfs: zml.io.VFS.HF = try .auto(init.gpa, init.io, &http_client, init.environ_map);
    defer hf_vfs.deinit();

    var vfs: zml.io.VFS = try .init(init.gpa, init.io);
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());
    try vfs.register("hf", hf_vfs.io());

    const io = vfs.io();

    var zml_handler: Zml_handler = try .fromInit(init, io);
    defer zml_handler.deinit();

    try printZmlLogo(zml_handler.io);

    zml_handler.tic(&zml_handler.timers.total);
    try runFullPipeline(&zml_handler);
    zml_handler.toc(&zml_handler.timers.total);

    zml_handler.timers.print();
}

pub fn runFullPipeline(zml_handler: *Zml_handler) !void {
    
    // ------------------------------------------------
    // Thinking/Inspiration phase : 5Hz LLM model
    // ------------------------------------------------

    // TODO: set seq_len higher than default 1024 if target duration is very long
    
    zml_handler.tic(&zml_handler.timers.llm.total);
    
    var acellm = try acellm_.AceLlm_handler.init(zml_handler);
    defer acellm.deinit(zml_handler.allocator);
    
    const inspi_tokens = try inference.tokenizeInspirationPrompt(zml_handler, acellm.tokenizer);
    defer zml_handler.allocator.free(inspi_tokens);
    
    const inspi_result = try inference.generateInspirationText(zml_handler, &acellm, inspi_tokens);
    defer zml_handler.allocator.free(inspi_result);
    var audio_metadata: inference.AudioMetadata = try .initFromString(zml_handler.allocator, inspi_result);
    defer audio_metadata.deinit(zml_handler.allocator);
    
    if (zml_handler.args.duration > 0) try audio_metadata.setDuration(zml_handler.allocator, zml_handler.args.duration);
    const duration = try audio_metadata.duration_s();

    zml_handler.toc(&zml_handler.timers.llm.total);

    var audio_codes: inference.AudioCodes = try .empty(zml_handler.allocator);
    defer audio_codes.deinit(zml_handler.allocator);
    
    if (!zml_handler.args.skip_cfg) {
        zml_handler.tic(&zml_handler.timers.cfg.total);
        
        const cond_tok, const uncond_tok = try inference.tokenizeGenerationPrompt(zml_handler.allocator, acellm.tokenizer, audio_metadata);
        defer zml_handler.allocator.free(cond_tok);
        defer zml_handler.allocator.free(uncond_tok);
        
        var acecfg = try acellm_.AceCfg_handler.initFromLlm(zml_handler, &acellm);
        defer acecfg.deinit(zml_handler.allocator);
        defer acecfg.unloadBuffers();
        
        audio_codes.deinit(zml_handler.allocator);
        audio_codes = try inference.generateAudioCodes(zml_handler, &acecfg, cond_tok, uncond_tok, audio_metadata);

        zml_handler.toc(&zml_handler.timers.cfg.total);
    }
    
    acellm.unloadBuffers(zml_handler.allocator);

    // ------------------------------------------------
    // The text inputs of the DiT need to be embedded
    // using the AceEmb model embedding, not 5Hz
    // ------------------------------------------------

    zml_handler.tic(&zml_handler.timers.emb.total);

    const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.aceemb);
    var tokenizer = try aceemb_.AceEmb_handler.loadTokenizer(zml_handler, repo);
    defer tokenizer.deinit();

    const caption_tok = try inference.tokenizeInputCaption(zml_handler.allocator, tokenizer, audio_metadata);
    const lyric_tok = try inference.tokenizeInputLyrics(zml_handler.allocator, tokenizer, audio_metadata);
    defer zml_handler.allocator.free(caption_tok);
    defer zml_handler.allocator.free(lyric_tok);

    var aceemb = try aceemb_.AceEmb_handler.init(zml_handler, @intCast(caption_tok.len), @intCast(lyric_tok.len));
    defer aceemb.deinit(zml_handler.allocator);

    const text_emb = try inference.generateTextEmbedding(zml_handler, &aceemb, tokenizer, caption_tok, lyric_tok);
    defer text_emb.deinit(zml_handler.allocator);

    aceemb.unloadBuffers(zml_handler.allocator);

    zml_handler.toc(&zml_handler.timers.emb.total);

    // ------------------------------------------------
    // Encoding phase : prepare input latents and
    // encoded conditions for diffusion
    // ------------------------------------------------

    zml_handler.tic(&zml_handler.timers.enc.total);
    
    const int_codes = try audio_codes.getIntCodes(zml_handler.allocator);
    defer zml_handler.allocator.free(int_codes);

    var aceenc = try aceenc_.AceEnc_handler.init(zml_handler, text_emb.textLen(), text_emb.lyricLen(), duration, int_codes.len);
    defer aceenc.deinit(zml_handler.allocator);

    const diffuse_args: inference.InitialLatents = try inference.prepareLatents(zml_handler, &aceenc, text_emb, int_codes, duration);
    defer diffuse_args.deinit(zml_handler.allocator);

    aceenc.unloadBuffers(zml_handler.allocator);

    zml_handler.toc(&zml_handler.timers.enc.total);

    // ------------------------------------------------
    // Tiled generation : compile DiT and VAE models
    // ------------------------------------------------

    zml_handler.tic(&zml_handler.timers.dit.total);
    
    var acedit = try acedit_.AceDit_handler.init(zml_handler, duration, diffuse_args.encoder_conditions.shape.dim(.s_enc));
    defer acedit.deinit(zml_handler.allocator);
    
    zml_handler.toc(&zml_handler.timers.dit.total);
    zml_handler.tic(&zml_handler.timers.vae.total);
    
    const decode_t: u32 = 1;
    var acevae = try acevae_.AceVae_handler.init(zml_handler, decode_t + 2);
    defer acevae.deinit(zml_handler.allocator);
    
    zml_handler.toc(&zml_handler.timers.vae.total);
    
    for (0..zml_handler.args.n) |i| {
        
        // ------------------------------------------------
        // Generation phase : diffusion with DiT model
        // ------------------------------------------------
    
        zml_handler.tic(&zml_handler.timers.dit.total);
    
        const diffused_latents: inference.DiffusedLatents = try inference.runDiffusion(zml_handler, &acedit, diffuse_args, i);
        defer diffused_latents.deinit(zml_handler.allocator);
    
        zml_handler.toc(&zml_handler.timers.dit.total);
        
        // ------------------------------------------------
        // Decode diffused latents with the VAE model
        // ------------------------------------------------
    
        zml_handler.tic(&zml_handler.timers.vae.total);
    
        const decoded_audio: inference.DecodedAudio = try inference.decodeAudioLatentsTiled(zml_handler, &acevae, diffused_latents, decode_t);
        defer decoded_audio.deinit(zml_handler.allocator);
    
        zml_handler.toc(&zml_handler.timers.vae.total);
    
        // ------------------------------------------------
        // Export decoded audio as WAV
        // ------------------------------------------------
    
         zml_handler.tic(&zml_handler.timers.wav);
         
         try exportDecodedAudioAsWav(zml_handler, decoded_audio, i);
    
         zml_handler.toc(&zml_handler.timers.wav);
    }

    acedit.unloadBuffers(zml_handler.allocator);
    acevae.unloadBuffers(zml_handler.allocator);
    
}

pub fn exportDecodedAudioAsWav(zml_handler: *Zml_handler, decoded_audio: inference.DecodedAudio, index: usize) !void {
    const io = zml_handler.local_io;
    const indexed_output_path = try std.fmt.allocPrint(zml_handler.allocator, "decoded_audio_{d}.wav", .{index});
    defer zml_handler.allocator.free(indexed_output_path);

    const audio = decoded_audio.audio;
    const shape = audio.shape;

    if (shape.rank() != 2) return error.InvalidAudioRank;
    if (audio.dtype() != .f32) return error.UnsupportedAudioType;

    const num_channels_i64 = shape.dim(0);
    const num_frames_i64 = shape.dim(1);

    const num_frames: u32 = std.math.cast(u32, num_frames_i64) orelse return error.AudioTooLong;
    const num_channels: u16 = std.math.cast(u16, num_channels_i64) orelse return error.InvalidChannelCount;
    const sample_rate: u32 = 48_000;
    const bytes_per_sample: u16 = @sizeOf(f32);
    const bits_per_sample: u16 = bytes_per_sample * 8;
    const block_align: u16 = num_channels * bytes_per_sample;
    const byte_rate: u32 = sample_rate * block_align;

    const samples = audio.constItems(f32);
    const expected_samples_u64 = @as(u64, num_frames) * num_channels;
    const expected_samples: usize = std.math.cast(usize, expected_samples_u64) orelse return error.AudioTooLarge;
    if (samples.len != expected_samples) return error.InvalidAudioBufferSize;

    const data_chunk_size_u64 = @as(u64, samples.len) * @sizeOf(f32);
    const data_chunk_size: u32 = std.math.cast(u32, data_chunk_size_u64) orelse return error.AudioTooLarge;
    const fact_chunk_size: u32 = 4;
    const riff_chunk_size_u64: u64 = 36 + 12 + data_chunk_size;
    const riff_chunk_size: u32 = std.math.cast(u32, riff_chunk_size_u64) orelse return error.AudioTooLarge;

    var file = try std.Io.Dir.createFile(.cwd(), io, indexed_output_path, .{ .truncate = true });
    defer file.close(io);

    var file_writer = file.writer(io, &.{});
    const writer: *std.Io.Writer = &file_writer.interface;

    try writer.writeAll("RIFF");
    try writer.writeInt(u32, riff_chunk_size, .little);
    try writer.writeAll("WAVE");

    try writer.writeAll("fmt ");
    try writer.writeInt(u32, 16, .little);
    try writer.writeInt(u16, 3, .little);
    try writer.writeInt(u16, num_channels, .little);
    try writer.writeInt(u32, sample_rate, .little);
    try writer.writeInt(u32, byte_rate, .little);
    try writer.writeInt(u16, block_align, .little);
    try writer.writeInt(u16, bits_per_sample, .little);

    try writer.writeAll("fact");
    try writer.writeInt(u32, fact_chunk_size, .little);
    try writer.writeInt(u32, num_frames, .little);

    try writer.writeAll("data");
    try writer.writeInt(u32, data_chunk_size, .little);

    for (0..num_frames) |frame_idx| {
        for (0..num_channels) |channel_idx| {
            const sample = samples[channel_idx * num_frames + frame_idx];
            try writer.writeInt(u32, @bitCast(sample), .little);
        }
    }

    try writer.flush();
    log.info("Exported decoded audio to {s}", .{ indexed_output_path });
}


pub fn printSafetensors(registry: zml.safetensors.TensorRegistry) !void {
    const tensors: zml.safetensors.Tensors = registry.tensors;
    const data = tensors.entries;
    for (0..data.len) |i| {
        const entry = data.get(i);
        const tensor: zml.safetensors.Tensor = tensors.get(entry.key).?;
        std.log.info("Tensor(name={s} shape={f} size={d})", .{
            tensor.name,
            tensor.shape,
            tensor.byteSize(),
        });
    }
}

pub fn parseConfig(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(T) {
    const file = try dir.openFile(io, "config.json", .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    return try std.json.parseFromTokenSource(T, allocator, &reader, .{ .ignore_unknown_fields = true });
}


pub fn printZmlLogo(io: std.Io) !void {
    const LOGO =
        \\
        \\
        \\ ███████╗███╗   ███╗██╗
        \\ ╚══███╔╝████╗ ████║██║
        \\   ███╔╝ ██╔████╔██║██║
        \\  ███╔╝  ██║╚██╔╝██║██║  .ai
        \\ ███████╗██║ ╚═╝ ██║███████╗
        \\ ╚══════╝╚═╝     ╚═╝╚══════╝
        \\
        \\
        \\
    ;
    var writer = std.Io.File.stdout().writer(io, &.{});
    try writer.interface.writeAll(LOGO);
    try writer.interface.flush();
}