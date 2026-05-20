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
    mem: MemoryChecker,

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
            .mem = .{ .platform = platform },
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

    pub fn tokensPerSecond(duration: std.Io.Duration, tokens: u64) f64 {
        if (tokens == 0) return 0;
        const seconds = @as(f64, @floatFromInt(duration.toNanoseconds())) / 1e9;
        if (seconds <= 0) return 0;
        return @as(f64, @floatFromInt(tokens)) / seconds;
    }

};

pub const Uri_handler = struct {
    acellm: []const u8,
    aceemb: []const u8,
    acedit: []const u8,
    is_xl: bool,
    acevae: []const u8,
    silence: []const u8 = "file://examples//acestep//models//acestep-v15-turbo",
    audio_ref: []const u8 = "file://examples//acestep//references//source.wav",
    style_ref: []const u8 = "file://examples//acestep//references//style.wav",

    pub fn fromLocal(args: Args) Uri_handler {
        return .{
            .acellm = if (args.llm_size == 0) "file://examples//acestep//models//acestep-5Hz-lm-0.6B"
                             else if (args.llm_size == 1) "file://examples//acestep//models//acestep-5Hz-lm-1.7B"
                                                     else "file://examples//acestep//models//acestep-5Hz-lm-4B",
            .aceemb = "file://examples//acestep//models//Qwen3-Embedding-0.6B",
            .acedit = if (args.dit_size == 0) "file://examples//acestep//models//acestep-v15-turbo"
                                                     else "file://examples//acestep//models//acestep-v15-xl-turbo",
            .is_xl = args.dit_size == 1,
            .acevae = "file://examples//acestep//models//Oobleck-vae",
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

pub const MemoryChecker = struct {
    platform: *zml.Platform,
    bytes_before: i64 = 0,
    bytes_after: i64 = 0,

    pub fn start(self: *MemoryChecker, id: usize) void {
        self.bytes_before = @intCast(self.platform.devices[id].memoryStats().bytes_in_use);
    }

    pub fn check(self: *MemoryChecker, id: usize) void {
        self.bytes_after = @intCast(self.platform.devices[id].memoryStats().bytes_in_use);
        const leaked = @abs(self.bytes_after - self.bytes_before);
        if (leaked != 0) {
            std.log.info("memory usage: before={d} after={d} leaked={d}", .{ self.bytes_before, self.bytes_after, leaked });
        }
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
    use_audio_ref: bool = false,
    use_style_ref: bool = false,
    match_level: u8 = 0,
    cover_strength: u8 = 0,
    seed: u32 = 0,

    pub const help =
        \\ Use acestep --prompt=<...> [options]
        \\
        \\ Run audio generation with the selected models.
        \\
        \\ Options:
        \\   --prompt=<string>      Prompt to use for generation (required)
        \\   --instru               Ask for an instrumental audio (default: 0)
        \\   --local-files          Use local model paths, see README to setup (default: 0)
        \\   --llm-size=<int>       Size of the 5Hz LLM (0/1/2 for 0.6B/1.7B/4B, default: 0)
        \\   --dit-size=<int>       Size of the DiT model (0/1 for turbo/turbo-xl, default: 0)
        \\   --skip-cfg             Optional, disable CFG phase (default: 0)
        \\   --duration=<number>    Constrains the duration in seconds (required if CFG is disabled, default: -1)
        \\   --n=<int>              Number of audio files to generate with different seeds for the diffusion (default: 1)
        \\   --use-audio-ref        Use references/source.wav as source to remix (default: 0)
        \\   --use-style-ref        Use references/style.wav as style/timbre reference (default: 0)
        \\   --match-level=<int>    Between 0 (pure noise) and 8 (exact audio ref) to init the diffusion (default: 0)
        \\   --cover-strength=<int> Between 0 (no cover) and 8 (full cover) iterations in cover mode before switching to non-cover mode (default: 0)
        \\   --seed=<int>           Random seed (default: 0)
        \\
    ;
};

// 4090
// --prompt='a peak-time dark techno track'
// --llm-size=2 --dit-size=1
// --duration=180 --n=3
// bazel run --config=release examples/acestep --//platforms:cuda=true -- --instru --local-files
// info: Module    init  compile     load  prefill   decode    total
// info:   llm    0.26s    1.88s    0.92s    0.17s    2.69s    5.93s
// info:   cfg    0.00s    0.80s    0.00s    0.29s   20.32s   21.42s
// info:   emb    0.00s    1.37s    0.57s    0.00s    0.00s    2.15s
// info:   enc    0.00s    2.92s    0.57s    0.01s    0.00s    3.51s
// info:   dit    0.00s    2.03s    0.83s    4.36s    0.00s    7.25s
// info:   vae    0.00s    1.82s    0.58s    2.24s    0.00s    4.67s
// info:   wav                                                 0.24s
// info: total                                                45.24s

// param1 : match_level: dimention iter, in 0-8, initial noise level matches the scheduled noised at iter match_level
// param2 : cover_strength: dimension iter, in 0-8, has to be >= match_level, how many iters we do in cover mode before switching to non cover

// TODO: faire remix basique
// - prepare no cover branch
// - smooth transitions between cover and non cover modes ?
// - if needed, context_latents from dequantized quantized source latents instead of source latents
// - ajouter variance
// TODO: essayer de faire lyric remix (flow-edit...)
// - embed old and new lyric
// - switch/blend/dice between the two during the diffusion
// TODO: batch cfg

// TODO: move model related code from inference to Exes struct inside models
// TODO: load in parallel as compile

// TODO: investigate audio level normalization
// TODO: sft/base model for max quality
// TODO: post process VAE output to improve audio quality (AudioSR/Vocos BWE + Music De-limiter Network/DSP Transient Shaper)
// TODO: to remove voice distortion, extract voice with MelBand RoFormer / BS-RoFormer, instru with HTDemucs v4
//       then recover audio (ex: DDDM-VC) voice, then recombine. Or: frequency cutoff with AudioSR

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
    try runText2MusicPipeline(&zml_handler);
    zml_handler.toc(&zml_handler.timers.total);

    zml_handler.timers.print();
}

pub fn runText2MusicPipeline(zml_handler: *Zml_handler) !void {

    // ------------------------------------------------
    // Thinking/Inspiration phase : 5Hz LLM model
    // ------------------------------------------------

    zml_handler.tic(&zml_handler.timers.llm.total);

    var acellm = try acellm_.AceLlm_handler.init(zml_handler);
    defer acellm.deinit(zml_handler.allocator);

    const inspi_tokens = try inference.tokenizeInspirationPrompt(zml_handler, acellm.tokenizer);
    defer zml_handler.allocator.free(inspi_tokens);

    zml_handler.mem.start(0);
    const inspi_result = try inference.generateInspirationText(zml_handler, &acellm, inspi_tokens);
    defer zml_handler.allocator.free(inspi_result);
    zml_handler.mem.check(0);
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
        zml_handler.mem.start(0);
        audio_codes = try inference.generateAudioCodes(zml_handler, &acecfg, cond_tok, uncond_tok, audio_metadata);
        zml_handler.mem.check(0);
        
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

    const caption_tok = try inference.tokenizeInputCaption(zml_handler.allocator, tokenizer, audio_metadata, !zml_handler.args.skip_cfg);
    const lyric_tok = try inference.tokenizeInputLyrics(zml_handler.allocator, tokenizer, audio_metadata);
    defer zml_handler.allocator.free(caption_tok);
    defer zml_handler.allocator.free(lyric_tok);

    var aceemb = try aceemb_.AceEmb_handler.init(zml_handler);
    defer aceemb.deinit(zml_handler.allocator);

    zml_handler.mem.start(0);
    const caption_emb = try inference.embedText(zml_handler, &aceemb, caption_tok);
    const lyric_emb = try inference.embedLyric(zml_handler, &aceemb, lyric_tok);
    zml_handler.mem.check(0);
    defer caption_emb.free(zml_handler.allocator);
    defer lyric_emb.free(zml_handler.allocator);

    aceemb.unloadBuffers(zml_handler.allocator);

    zml_handler.toc(&zml_handler.timers.emb.total);

    // ------------------------------------------------
    // Encoding phase : prepare input latents and
    // encoded conditions for diffusion
    // ------------------------------------------------

    zml_handler.tic(&zml_handler.timers.enc.total);

    const int_codes = if (zml_handler.args.skip_cfg) null else try audio_codes.getIntCodes(zml_handler.allocator);
    defer if (int_codes) |codes| zml_handler.allocator.free(codes);

    var aceenc = try aceenc_.AceEnc_handler.init(zml_handler);
    defer aceenc.deinit(zml_handler.allocator);

    zml_handler.mem.start(0);
    const diffusion_args: inference.ContextLatents = .{
        .latents = try inference.prepareDiffusionLatents(zml_handler, &aceenc, duration, int_codes, null),
        .conditions = try inference.prepareDiffusionConditions(zml_handler, &aceenc, caption_emb, lyric_emb, null),
    };
    defer diffusion_args.deinit(zml_handler.allocator);
    zml_handler.mem.check(0);

    aceenc.unloadBuffers(zml_handler.allocator);

    zml_handler.toc(&zml_handler.timers.enc.total);

    // ------------------------------------------------
    // Tiled generation : compile DiT and VAE models
    // ------------------------------------------------

    zml_handler.tic(&zml_handler.timers.dit.total);

    var acedit = try acedit_.AceDit_handler.init(zml_handler, duration, diffusion_args.conditions.shape.dim(.s));
    defer acedit.deinit(zml_handler.allocator);

    zml_handler.toc(&zml_handler.timers.dit.total);
    zml_handler.tic(&zml_handler.timers.vae.total);

    var acevae = try acevae_.AceVaeDecoder_handler.init(zml_handler);
    defer acevae.deinit(zml_handler.allocator);

    zml_handler.toc(&zml_handler.timers.vae.total);

    for (0..zml_handler.args.n) |i| {

        // ------------------------------------------------
        // Generation phase : diffusion with DiT model
        // ------------------------------------------------

        zml_handler.tic(&zml_handler.timers.dit.total);
        zml_handler.mem.start(0);

        const diffused_latents = try inference.runCoverDiffusion(zml_handler, &acedit, diffusion_args, i);
        defer diffused_latents.deinit(zml_handler.allocator);

        zml_handler.mem.check(0);
        zml_handler.toc(&zml_handler.timers.dit.total);

        // ------------------------------------------------
        // Decode diffused latents with the VAE model
        // ------------------------------------------------

        zml_handler.tic(&zml_handler.timers.vae.total);
        zml_handler.mem.start(0);

        const decoded_audio = try inference.decodeAudioLatents(zml_handler, &acevae, diffused_latents);
        defer decoded_audio.deinit(zml_handler.allocator);

        zml_handler.mem.check(0);
        zml_handler.toc(&zml_handler.timers.vae.total);

        // ------------------------------------------------
        // Export decoded audio as WAV
        // ------------------------------------------------

         zml_handler.tic(&zml_handler.timers.wav);

         try exportAudio(zml_handler, decoded_audio, i);

         zml_handler.toc(&zml_handler.timers.wav);
    }

    acedit.unloadBuffers(zml_handler.allocator);
    acevae.unloadBuffers(zml_handler.allocator);

}

pub fn runRemixPipeline(zml_handler: *Zml_handler) !void {

    // ------------------------------------------------
    // Read/encode audio and style references
    // ------------------------------------------------

    zml_handler.tic(&zml_handler.timers.wav);

    var input_audio = try importAudio(zml_handler, zml_handler.uris.audio_ref);
    defer input_audio.deinit(zml_handler.allocator);

    var input_style: ?inference.AudioFrames = null;
    defer if (input_style) |style| style.deinit(zml_handler.allocator);
    if (zml_handler.args.use_style_ref) {
        const style = try importAudio(zml_handler, zml_handler.uris.style_ref);
        defer style.deinit(zml_handler.allocator);
        input_style = try extractSummary(zml_handler, style);
    }

    zml_handler.toc(&zml_handler.timers.wav);
    zml_handler.tic(&zml_handler.timers.vae.total);

    var acevae_encoder = try acevae_.AceVaeEncoder_handler.init(zml_handler);
    defer acevae_encoder.deinit(zml_handler.allocator);

    var audio_latent = try inference.encodeAudioLatents(zml_handler, &acevae_encoder, input_audio);
    defer audio_latent.deinit(zml_handler.allocator);

    const style_latent = if (input_style) try inference.encodeAudioLatents(zml_handler, &acevae_encoder, input_style) else null;
    defer if (style_latent) |latent| latent.deinit(zml_handler.allocator);

    acevae_encoder.unloadBuffers(zml_handler.allocator);

    zml_handler.toc(&zml_handler.timers.vae.total);

    // ------------------------------------------------
    // No 5Hz task in remix mode
    // ------------------------------------------------

    const audio_metadata = inference.AudioMetadata.empty(zml_handler.allocator);
    defer audio_metadata.deinit(zml_handler.allocator);

    audio_metadata.setDuration(zml_handler.allocator, audio_latent.duration_s());
    const duration = audio_latent.duration_s();

    // ------------------------------------------------
    // Embed text inputs for diffusion conditions
    // ------------------------------------------------

    zml_handler.mem.start(0);
    zml_handler.tic(&zml_handler.timers.emb.total);

    const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.aceemb);
    var tokenizer = try aceemb_.AceEmb_handler.loadTokenizer(zml_handler, repo);
    defer tokenizer.deinit();

    const caption_tok_non_cover = try inference.tokenizeInputCaption(zml_handler.allocator, tokenizer, audio_metadata, false);
    const caption_tok_cover = try inference.tokenizeInputCaption(zml_handler.allocator, tokenizer, audio_metadata, true);
    const lyric_tok = try inference.tokenizeInputLyrics(zml_handler.allocator, tokenizer, audio_metadata);
    defer caption_tok_non_cover.free(zml_handler.allocator);
    defer caption_tok_cover.free(zml_handler.allocator);
    defer lyric_tok.free(zml_handler.allocator);

    std.log.info("caption_tok_cover: {d}, caption_tok_non_cover: {d}", .{caption_tok_cover.len, caption_tok_non_cover.len});

    var aceemb = try aceemb_.AceEmb_handler.init(zml_handler);
    defer aceemb.deinit(zml_handler.allocator);

    const caption_emb_non_cover = try inference.embedText(zml_handler, &aceemb, caption_tok_non_cover);
    const caption_emb_cover = try inference.embedText(zml_handler, &aceemb, caption_tok_cover);
    const lyric_emb = try inference.embedLyric(zml_handler, &aceemb, lyric_tok);
    defer zml_handler.allocator.free(caption_emb_non_cover);
    defer zml_handler.allocator.free(caption_emb_cover);
    defer zml_handler.allocator.free(lyric_emb);

    aceemb.unloadBuffers(zml_handler.allocator);

    zml_handler.toc(&zml_handler.timers.emb.total);
    zml_handler.mem.check(0);

    // ------------------------------------------------
    // Encode context and conditions for diffusion
    // ------------------------------------------------

    zml_handler.mem.start(0);
    zml_handler.tic(&zml_handler.timers.enc.total);

    var aceenc = try aceenc_.AceEnc_handler.init(zml_handler);
    defer aceenc.deinit(zml_handler.allocator);

    const diffusion_args_cover: inference.ContextLatents = .{
        .latents = try inference.prepareDiffusionLatents(zml_handler, &aceenc, duration, null, audio_latent),
        .encoder_conditions = try inference.prepareDiffusionConditions(zml_handler, &aceenc, caption_emb_cover, lyric_emb, style_latent),
    };
    defer diffusion_args_cover.deinit(zml_handler.allocator);
    
    const diffusion_args_non_cover: inference.ContextLatents = .{
        .latents = try inference.prepareDiffusionLatents(zml_handler, &aceenc, duration, null, null),
        .conditions = try inference.prepareDiffusionConditions(zml_handler, &aceenc, caption_emb_non_cover, lyric_emb, style_latent),
    };
    defer diffusion_args_non_cover.deinit(zml_handler.allocator);

    aceenc.unloadBuffers(zml_handler.allocator);

    zml_handler.toc(&zml_handler.timers.enc.total);
    zml_handler.mem.check(0);

    // ------------------------------------------------
    // Tiled generation : compile DiT and VAE models
    // ------------------------------------------------

    zml_handler.tic(&zml_handler.timers.dit.total);

    var acedit = try acedit_.AceDit_handler.init(zml_handler, duration, diffusion_args_cover.conditions.shape.dim(.s));
    defer acedit.deinit(zml_handler.allocator);

    zml_handler.toc(&zml_handler.timers.dit.total);
    zml_handler.tic(&zml_handler.timers.vae.total);

    var acevae = try acevae_.AceVaeDecoder_handler.init(zml_handler);
    defer acevae.deinit(zml_handler.allocator);

    zml_handler.toc(&zml_handler.timers.vae.total);

    for (0..zml_handler.args.n) |i| {

        // ------------------------------------------------
        // Generation phase : diffusion with DiT model
        // ------------------------------------------------

        zml_handler.tic(&zml_handler.timers.dit.total);
        zml_handler.mem.start(0);

        const diffused_latents = try inference.runRemixDiffusion(zml_handler, &acedit, audio_latent, diffusion_args_cover, diffusion_args_non_cover, i, zml_handler.args.match_level);
        defer diffused_latents.deinit(zml_handler.allocator);

        zml_handler.mem.check(0);
        zml_handler.toc(&zml_handler.timers.dit.total);

        // ------------------------------------------------
        // Decode diffused latents with the VAE model
        // ------------------------------------------------

        zml_handler.tic(&zml_handler.timers.vae.total);
        zml_handler.mem.start(0);

        const decoded_audio = try inference.decodeAudioLatents(zml_handler, &acevae, diffused_latents);
        defer decoded_audio.deinit(zml_handler.allocator);

        zml_handler.mem.check(0);
        zml_handler.toc(&zml_handler.timers.vae.total);

        // ------------------------------------------------
        // Export decoded audio as WAV
        // ------------------------------------------------

         zml_handler.tic(&zml_handler.timers.wav);

         try exportAudio(zml_handler, decoded_audio, i);

         zml_handler.toc(&zml_handler.timers.wav);
    }

    acedit.unloadBuffers(zml_handler.allocator);
    acevae.unloadBuffers(zml_handler.allocator);

}


pub fn importAudio(zml_handler: *Zml_handler, input_path: []const u8) !inference.AudioFrames {
    const io = zml_handler.io;

    const file = try std.Io.Dir.openFile(.cwd(), io, input_path, .{});
    defer file.close(io);

    var reader = file.reader(io, &.{});
    const bytes = try reader.interface.readAlloc(zml_handler.allocator, try file.length(io));
    errdefer zml_handler.allocator.free(bytes);

    if (bytes.len < 12) return error.InvalidWavFile;
    if (!std.mem.eql(u8, bytes[0..4], "RIFF")) return error.InvalidWavFile;
    if (!std.mem.eql(u8, bytes[8..12], "WAVE")) return error.InvalidWavFile;

    var offset: usize = 12;
    var num_channels: u16 = 0;
    var bits_per_sample: u16 = 0;
    var audio_format: u16 = 0;
    var data_chunk: ?[]const u8 = null;

    while (offset + 8 <= bytes.len) {
        const chunk_id = bytes[offset .. offset + 4];
        const chunk_size = std.mem.readInt(u32, bytes[offset + 4 .. offset + 8][0..4], .little);
        offset += 8;

        const chunk_end = offset + chunk_size;
        if (chunk_end > bytes.len) return error.InvalidWavFile;

        if (std.mem.eql(u8, chunk_id, "fmt ")) {
            if (chunk_size < 16) return error.InvalidWavFile;
            audio_format = std.mem.readInt(u16, bytes[offset + 0 .. offset + 2][0..2], .little);
            num_channels = std.mem.readInt(u16, bytes[offset + 2 .. offset + 4][0..2], .little);
            bits_per_sample = std.mem.readInt(u16, bytes[offset + 14 .. offset + 16][0..2], .little);
        } else if (std.mem.eql(u8, chunk_id, "data")) {
            data_chunk = bytes[offset..chunk_end];
        }

        offset = chunk_end + (chunk_size % 2);
    }

    if (audio_format != 3) return error.UnsupportedWavFormat;
    if (bits_per_sample != 32) return error.UnsupportedAudioType;
    if (num_channels == 0) return error.InvalidChannelCount;

    const data = data_chunk orelse return error.InvalidWavFile;
    const bytes_per_sample = @sizeOf(f32);
    const block_align = @as(usize, num_channels) * bytes_per_sample;
    if (block_align == 0 or data.len % block_align != 0) return error.InvalidAudioBufferSize;

    const num_frames = data.len / block_align;
    const samples = std.mem.bytesAsSlice(u32, @constCast(data));

    // make sure audio has an integer duration in seconds
    const pad_frames = (48_000 - (num_frames % 48_000)) % 48_000;
    const effective_num_frames = num_frames + pad_frames;

    const audio_slice: zml.Slice = try .alloc(zml_handler.allocator, zml.Shape.init(.{ .a = num_channels, .t = effective_num_frames }, .f32));
    errdefer audio_slice.free(zml_handler.allocator);

    for (0..num_frames) |frame_idx| {
        for (0..num_channels) |channel_idx| {
            const src_index = frame_idx * num_channels + channel_idx;
            const dst_index = channel_idx * effective_num_frames + frame_idx;
            audio_slice.items(f32)[dst_index] = @bitCast(samples[src_index]);
        }
    }
    for (num_frames..effective_num_frames) |frame_idx| {
        for (0..num_channels) |channel_idx| {
            const dst_index = channel_idx * effective_num_frames + frame_idx;
            audio_slice.items(f32)[dst_index] = 0.0;
        }
    }

    zml_handler.allocator.free(bytes);
    log.info("Imported {d}s of audio from {s}", .{ @divExact(effective_num_frames, 48_000), input_path });
    return .{ .audio = audio_slice };
}

pub fn exportAudio(zml_handler: *Zml_handler, decoded_audio: inference.AudioFrames, index: usize) !void {
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

    const interleaved_bytes = try zml_handler.allocator.alloc(u8, data_chunk_size);
    defer zml_handler.allocator.free(interleaved_bytes);

    const interleaved_words = std.mem.bytesAsSlice(u32, interleaved_bytes);

    for (0..num_frames) |frame_idx| {
        for (0..num_channels) |channel_idx| {
            const src_index = channel_idx * num_frames + frame_idx;
            const dst_index = frame_idx * num_channels + channel_idx;
            interleaved_words[dst_index] = @bitCast(samples[src_index]);
        }
    }

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
    try writer.writeAll(interleaved_bytes);

    try writer.flush();
    log.info("Exported decoded audio to {s}", .{indexed_output_path});
}

pub fn extractSummary(zml_handler: *Zml_handler, audio: inference.AudioFrames) !inference.AudioFrames {
    // audio.x is a 2 x num_frames @48kHz zml.Slice containing a style/timbre reference audio file.
    // to condition the diffusion, we need a 30s summary of the audio

    const sample_rate: usize = 48_000;
    const summary_duration_s: usize = 30;
    const chunk_duration_s: usize = 10;
    const summary_frames: usize = sample_rate * summary_duration_s;
    const chunk_frames: usize = sample_rate * chunk_duration_s;

    const num_channels: usize = @intCast(audio.audio.shape.dim(.a));
    const num_frames: usize = @intCast(audio.audio.shape.dim(.t));
    if (num_channels != 2) return error.InvalidChannelCount;
    if (num_frames < summary_frames) return error.AudioTooShort;

    const summary_audio: zml.Slice = try .alloc(zml_handler.allocator, zml.Shape.init(.{ .a = num_channels, .t = summary_frames }, .f32));
    errdefer summary_audio.free(zml_handler.allocator);

    const third_frames = num_frames / 3;
    if (third_frames < chunk_frames) return error.AudioTooShort;

    var prng = std.Random.DefaultPrng.init(@intCast(zml_handler.args.seed));
    const random = prng.random();

    const src = audio.audio.items(f32);
    const dst = summary_audio.items(f32);

    for (0..3) |third_idx| {
        const third_start = third_idx * third_frames;
        const third_end = if (third_idx == 2) num_frames else (third_idx + 1) * third_frames;
        const available_start_count = third_end - third_start - chunk_frames + 1;
        const chunk_start = third_start + random.uintLessThan(usize, available_start_count);
        const chunk_dst_start = third_idx * chunk_frames;

        for (0..num_channels) |channel_idx| {
            const src_offset = channel_idx * num_frames + chunk_start;
            const dst_offset = channel_idx * summary_frames + chunk_dst_start;
            @memcpy(dst[dst_offset .. dst_offset + chunk_frames], src[src_offset .. src_offset + chunk_frames]);
        }
    }

    return .{ .audio = summary_audio };
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
