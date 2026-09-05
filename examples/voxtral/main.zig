const std = @import("std");
const builtin = @import("builtin");
const log = std.log;
const cfg = @import("config.zig");
const Config = cfg.Config;
const StreamParams = cfg.StreamParams;

const zml = @import("zml");
const stdx = zml.stdx;
const Tensor = zml.Tensor;

const mel = @import("mel_spectrogram.zig");
const LogMelSpectrogram = mel.LogMelSpectrogram;

const enc = @import("encoder.zig");
const Encoder = enc.Encoder;

const dec = @import("decoder.zig");
const Decoder = dec.Decoder;
const Adapter = dec.Adapter;

const common = @import("common.zig");
const KvCache = common.KvCache;

const voxtral = @import("voxtral.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const CliArgs = struct {
    input: ?[]const u8 = null,
    model: []const u8,
    transcription_delay_ms: f32 = 480.0,
    backend: ?zml.attention.Backend = null,
    enc_kv_size: ?u32 = null,
    dec_kv_size: ?u32 = null,
};

const n_left_pad_tokens: u32 = 32;

pub fn main(init: std.process.Init) !void {
    log.info("Start of Voxtral", .{});

    var dbg = std.heap.DebugAllocator(.{ .thread_safe = true }).init;
    defer if (builtin.mode == .Debug) std.debug.assert(dbg.deinit() == .ok);

    const allocator = switch (builtin.mode) {
        .Debug => dbg.allocator(),
        else => std.heap.smp_allocator,
    };

    const args = stdx.flags.parse(init.minimal.args, CliArgs);
    const io = init.io;
    const input = if (args.input) |path| try std.Io.Dir.cwd().openFile(io, path, .{}) else std.Io.File.stdin();
    defer if (args.input != null) input.close(io);

    var progress = std.Progress.start(io, .{ .root_name = "Voxtral" });

    const model_dir = try zml.safetensors.resolveModelRepo(io, args.model);
    defer model_dir.close(io);

    // The repository also ships HF-renamed model.safetensors; use native Mistral keys.
    const model_file = try model_dir.openFile(io, if (std.mem.endsWith(u8, args.model, ".safetensors")) std.fs.path.basename(args.model) else "consolidated.safetensors", .{});
    defer model_file.close(io);
    var model_registry = try zml.safetensors.fetchRegistry(allocator, io, model_dir, model_file);
    defer model_registry.deinit();

    var model_store: zml.io.TensorStore = .fromRegistry(allocator, &model_registry);
    defer model_store.deinit();

    var parsed_config = try cfg.parseConfig(allocator, io, model_dir);
    defer parsed_config.deinit();
    const config = parsed_config.value;

    const delay = args.transcription_delay_ms;
    if (!std.math.isFinite(delay) or delay < 80 or delay > 2400 or @mod(delay, 80) != 0) {
        log.err("transcription_delay_ms must be a multiple of 80 in range [80, 2400], got {d}", .{delay});
        return error.InvalidTranscriptionDelay;
    }
    const sp: StreamParams = .init(config, delay, n_left_pad_tokens);

    // Build prompt_tokens: [BOS] ++ [STREAMING_PAD] * (n_left_pad_tokens + n_delay_tokens)
    const prompt_tokens = try allocator.alloc(u32, sp.prompt_len);
    defer allocator.free(prompt_tokens);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    log.info("Selected platform {f}\n", .{platform.fmtVerbose()});

    const backend = args.backend orelse b: {
        const selected = zml.attention.Backend.auto(platform);
        log.info("Selected backend: {}", .{selected});
        break :b selected;
    };

    var melspectro_model: LogMelSpectrogram = .init(config);

    var encoder_model: Encoder = .init(allocator, model_store.view(), config);
    defer encoder_model.deinit(allocator);

    const adapter: Adapter = .init(model_store.view(), config);

    var decoder_model: Decoder = .init(allocator, model_store.view(), config);
    defer decoder_model.deinit(allocator);

    const enc_cfg = config.encoder();
    // Preserve the oldest history needed by the first query of a multi-frame step.
    const enc_kv_size = args.enc_kv_size orelse (enc_cfg.sliding_window + sp.dsf - 1);
    const dec_kv_size = args.dec_kv_size orelse config.sliding_window;
    if (enc_kv_size < @max(enc_cfg.sliding_window + sp.dsf - 1, sp.prompt_len * sp.dsf) or dec_kv_size < @max(config.sliding_window, sp.prompt_len)) {
        log.err("KV caches must cover the sliding window and prefill", .{});
        return error.InvalidCacheSize;
    }

    // KV cache shapes for encoder and decoder, sized to sliding_window for memory efficiency
    const enc_dtype = encoder_model.norm.dtype();
    const enc_kv_cache: KvCache = .init(.init(.{
        .layer = enc_cfg.n_layers,
        .k = enc_kv_size,
        .h = enc_cfg.n_kv_heads,
        .hd = enc_cfg.head_dim,
    }, enc_dtype));

    const dec_dtype = decoder_model.tok_embeddings.dtype();
    const dec_kv_cache: KvCache = .init(.init(.{
        .layer = decoder_model.layers.len,
        .k = dec_kv_size,
        .h = config.n_kv_heads,
        .hd = config.head_dim,
    }, dec_dtype));

    const enc_attention_metadata: zml.attention.Metadata = .init(.fromBackend(backend, @intCast(enc_kv_size), config.encoder().n_heads));
    const dec_attention_metadata: zml.attention.Metadata = .init(.fromBackend(backend, @intCast(dec_kv_size), config.n_heads));
    const attention_parameters: zml.attention.Parameters = .init(.fromBackend(backend));

    // Load tokenizer and look up special token IDs.
    var tokenizer = try voxtral.loadTokenizer(allocator, io, model_dir, &progress);
    defer tokenizer.deinit();

    const token_bos = tokenizer.tokenToId("<s>") orelse @panic("tokenizer missing <s> token");
    const token_streaming_pad = tokenizer.tokenToId("[STREAMING_PAD]") orelse @panic("tokenizer missing [STREAMING_PAD] token");
    prompt_tokens[0] = token_bos;
    @memset(prompt_tokens[1..], token_streaming_pad);

    // Compile before loading weights to leave room for compiler/autotune allocations.
    var compiled_mel_step = try voxtral.compileMelStep(allocator, io, platform, melspectro_model, sp, &progress);
    defer compiled_mel_step.deinit();

    var compiled_mel_prefill = try voxtral.compileMelPrefill(allocator, io, platform, melspectro_model, sp, &progress);
    defer compiled_mel_prefill.deinit();

    var compiled_conv_stem_prefill = try voxtral.compileConvStemPrefill(allocator, io, platform, encoder_model, sp.prompt_len * sp.mel_per_step, &progress);
    defer compiled_conv_stem_prefill.deinit();

    var compiled_conv_stem_step = try voxtral.compileConvStemStep(allocator, io, platform, encoder_model, sp, &progress);
    defer compiled_conv_stem_step.deinit();

    var compiled_encoder_prefill = try voxtral.compileEncoderPrefill(allocator, io, platform, encoder_model, sp.prompt_len, enc_kv_cache, enc_attention_metadata, attention_parameters, &progress);
    defer compiled_encoder_prefill.deinit();

    var compiled_encoder_step = try voxtral.compileEncoderStep(allocator, io, platform, encoder_model, enc_kv_cache, enc_attention_metadata, attention_parameters, &progress);
    defer compiled_encoder_step.deinit();

    var compiled_adapter = try voxtral.compileAdapter(allocator, io, platform, adapter, sp.prompt_len, config, &progress);
    defer compiled_adapter.deinit();

    var compiled_adapter_step = try voxtral.compileAdapterStep(allocator, io, platform, adapter, config, &progress);
    defer compiled_adapter_step.deinit();

    var compiled_decoder_prefill, var compiled_decoder_decode = try voxtral.compileDecoder(allocator, io, platform, decoder_model, sp.prompt_len, dec_kv_cache, dec_attention_metadata, attention_parameters, &progress);
    defer compiled_decoder_prefill.deinit();
    defer compiled_decoder_decode.deinit();

    var mel_spectrum_buffers = try LogMelSpectrogram.load(&melspectro_model, io, platform);
    defer LogMelSpectrogram.unload(&mel_spectrum_buffers);

    var encoder_buffers = try Encoder.load(&encoder_model, allocator, io, platform, &model_store, &progress);
    defer Encoder.unload(&encoder_buffers, allocator);

    var adapter_buffers = try Adapter.load(&adapter, allocator, io, platform, &model_store, &progress);
    defer Adapter.unload(&adapter_buffers);

    var decoder_buffers = try Decoder.load(&decoder_model, allocator, io, platform, &model_store, &progress);
    defer Decoder.unload(&decoder_buffers, allocator);

    var enc_attention_metadata_buffers: zml.Bufferized(zml.attention.Metadata) = try enc_attention_metadata.initBuffer(io, platform, .replicated);
    defer zml.attention.Metadata.deinitBuffer(&enc_attention_metadata_buffers);

    var dec_attention_metadata_buffers: zml.Bufferized(zml.attention.Metadata) = try dec_attention_metadata.initBuffer(io, platform, .replicated);
    defer zml.attention.Metadata.deinitBuffer(&dec_attention_metadata_buffers);

    progress.end();

    const conv_state: enc.Encoder.ConvState = .{
        .conv1 = Tensor.init(.{ .batch = 1, .channels = config.audio().num_mel_bins, .time = 2 }, enc_dtype),
        .conv2 = Tensor.init(.{ .batch = 1, .channels = enc_cfg.dim, .time = 2 }, enc_dtype),
    };

    try voxtral.runPipeline(
        allocator,
        io,
        platform,
        config,
        &tokenizer,
        prompt_tokens,
        sp,
        .{
            .mel_step = &compiled_mel_step,
            .mel_prefill = &compiled_mel_prefill,
            .conv_stem_prefill = &compiled_conv_stem_prefill,
            .conv_stem_step = &compiled_conv_stem_step,
            .encoder_prefill = &compiled_encoder_prefill,
            .encoder_step = &compiled_encoder_step,
            .adapter = &compiled_adapter,
            .adapter_step = &compiled_adapter_step,
            .decoder_prefill = &compiled_decoder_prefill,
            .decoder_decode = &compiled_decoder_decode,
        },
        &mel_spectrum_buffers,
        &encoder_buffers,
        &adapter_buffers,
        &decoder_buffers,
        enc_kv_cache,
        dec_kv_cache,
        conv_state,
        &enc_attention_metadata_buffers,
        &dec_attention_metadata_buffers,
        input,
    );
}
