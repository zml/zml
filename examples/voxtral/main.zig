const std = @import("std");
const builtin = @import("builtin");
const log = std.log;
const wav_utils = @import("wav.zig");
const cfg = @import("config.zig");
const Config = cfg.Config;

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

const CliArgs = struct {
    input: []const u8,
    model: []const u8,
    transcription_delay_ms: f32 = 480.0,
    backend: ?zml.attention.Backend = null,
};

const n_left_pad_tokens: u32 = 32;

pub fn main() !void {
    log.info("Start of Voxtral", .{});

    var dbg = std.heap.DebugAllocator(.{ .thread_safe = true }).init;
    defer if (builtin.mode == .Debug) std.debug.assert(dbg.deinit() == .ok);

    const allocator = switch (builtin.mode) {
        .Debug => dbg.allocator(),
        else => std.heap.c_allocator,
    };

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    const args = stdx.flags.parseProcessArgs(CliArgs);

    var vfs: zml.io.VFS = try .init(allocator, threaded.io());
    defer vfs.deinit();

    var vfs_file: zml.io.VFS.File = .init(allocator, threaded.io(), .{});
    defer vfs_file.deinit();
    try vfs.register("file", vfs_file.io());

    const io = vfs.io();

    var progress = std.Progress.start(io, .{ .root_name = "Voxtral" });

    const file = try std.Io.Dir.openFile(.cwd(), io, args.input, .{});
    defer file.close(io);

    const model_dir = try zml.safetensors.resolveModelRepo(io, args.model);

    var model_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.model);
    defer model_registry.deinit();

    var model_store: zml.io.TensorStore = .fromRegistry(allocator, &model_registry);
    defer model_store.deinit();

    var parsed_config = try cfg.parseConfig(allocator, io, model_dir);
    defer parsed_config.deinit();
    const config = parsed_config.value;

    const sample_rate: f32 = @floatFromInt(config.audio().sampling_rate);
    const frame_rate = config.audio().frame_rate;
    const raw_audio_length_per_tok: u32 = @intFromFloat(sample_rate / frame_rate);
    const hop_length = config.audio().hop_length;

    const delay_samples: u32 = @intFromFloat(args.transcription_delay_ms / 1000.0 * sample_rate);
    const audio_length_per_tok = raw_audio_length_per_tok / hop_length;
    const n_delay_tokens = std.math.divCeil(u32, delay_samples / hop_length, audio_length_per_tok) catch unreachable;
    const n_right_pad_tokens = (n_delay_tokens + 1) + 10;

    // Build tokens: [BOS] ++ [STREAMING_PAD] * (n_left_pad_tokens + n_delay_tokens)
    const n_prompt_tokens = 1 + n_left_pad_tokens + n_delay_tokens;
    const tokens = try allocator.alloc(u32, n_prompt_tokens);
    defer allocator.free(tokens);

    var wav_buffer: [4096]u8 = undefined;
    var reader = file.reader(io, &wav_buffer);

    const left_pad = n_left_pad_tokens * raw_audio_length_per_tok;
    const padded_wav = try wav_utils.loadAndPadWav(allocator, &reader.interface, left_pad, n_right_pad_tokens, raw_audio_length_per_tok);
    defer allocator.free(padded_wav);
    const audio_len = padded_wav.len;

    var platform: *zml.Platform = try .auto(allocator, io, .{
        .cuda = .{ .allocator = .{ .bfc = .{ .memory_fraction = 0.75 } } },
    });
    defer platform.deinit(allocator);
    log.info("Selected platform {f}\n", .{platform.fmtVerbose()});

    const backend = args.backend orelse b: {
        const selected = zml.attention.Backend.auto(platform);
        log.info("Selected backend: {}", .{selected});
        break :b selected;
    };

    var melspectro_model: LogMelSpectrogram = .init(config);
    const encoder_prefix = "mm_streams_embeddings.embedding_module.whisper_encoder";
    var encoder_model: Encoder = .init(allocator, model_store.view().withPrefix(encoder_prefix), config);
    defer encoder_model.deinit(allocator);

    const adapter: Adapter = .init(model_store.view(), config);

    var decoder_model: Decoder = .init(allocator, model_store.view(), config);
    defer decoder_model.deinit(allocator);

    // KV cache shapes for encoder and decoder
    const enc_cfg = config.encoder();
    const enc_dtype = encoder_model.norm.dtype();
    const enc_kv_cache: KvCache = .init(.init(.{
        .layer = enc_cfg.n_layers,
        .k = enc_cfg.sliding_window,
        .h = enc_cfg.n_kv_heads,
        .hd = enc_cfg.head_dim,
    }, enc_dtype));

    const dec_dtype = decoder_model.tok_embeddings.dtype();
    const dec_kv_cache: KvCache = .init(.init(.{
        .layer = decoder_model.layers.len,
        .k = config.sliding_window,
        .h = config.n_kv_heads,
        .hd = config.head_dim,
    }, dec_dtype));

    const prompt_len: u32 = @intCast(tokens.len);

    const enc_attention_metadata: zml.attention.Metadata = .init(.fromBackend(backend, @intCast(enc_cfg.sliding_window)));
    const dec_attention_metadata: zml.attention.Metadata = .init(.fromBackend(backend, @intCast(config.sliding_window)));
    const attention_parameters: zml.attention.Parameters = .init(.fromBackend(backend));

    // Launch concurrent compilation and buffer loading
    var tokenizer_future = try io.concurrent(voxtral.loadTokenizer, .{ allocator, io, model_dir, &progress });

    // -- Fixed len for now
    var compiled_mel_spectrum_future = try io.concurrent(voxtral.compileMelSpectrum, .{ allocator, io, platform, melspectro_model, audio_len, &progress });
    const mel_per_step: u32 = config.downsample_factor() * 2;
    var compiled_conv_stem_future = try io.concurrent(voxtral.compileConvStem, .{ allocator, io, platform, encoder_model, prompt_len * mel_per_step, &progress });
    var compiled_conv_stem_step_future = try io.concurrent(voxtral.compileConvStemStep, .{ allocator, io, platform, encoder_model, &progress });
    // --

    var compiled_encoder_prefill_future = try io.concurrent(voxtral.compileEncoderPrefill, .{ allocator, io, platform, encoder_model, prompt_len, enc_kv_cache, enc_attention_metadata, attention_parameters, &progress });
    var compiled_encoder_step_future = try io.concurrent(voxtral.compileEncoderStep, .{ allocator, io, platform, encoder_model, enc_kv_cache, enc_attention_metadata, attention_parameters, &progress });
    var compiled_adapter_future = try io.concurrent(voxtral.compileAdapter, .{ allocator, io, platform, adapter, prompt_len, config, &progress });
    var compiled_adapter_step_future = try io.concurrent(voxtral.compileAdapterStep, .{ allocator, io, platform, adapter, config, &progress });
    var compiled_decoder_future = try io.concurrent(voxtral.compileDecoder, .{ allocator, io, platform, decoder_model, prompt_len, dec_kv_cache, dec_attention_metadata, attention_parameters, &progress });

    var mel_spectrum_buffers_future = try io.concurrent(LogMelSpectrogram.load, .{ &melspectro_model, io, platform });
    var encoder_buffers_future = try io.concurrent(Encoder.load, .{ &encoder_model, allocator, io, platform, &model_store, &progress });
    var adapter_buffers_future = try io.concurrent(Adapter.load, .{ &adapter, allocator, io, platform, &model_store, &progress });
    var decoder_buffers_future = try io.concurrent(Decoder.load, .{ &decoder_model, allocator, io, platform, &model_store, &progress });

    // Await tokenizer and look up special token IDs
    var tokenizer = try tokenizer_future.await(io);
    defer tokenizer.deinit();

    const token_bos = tokenizer.tokenToId("<s>") orelse @panic("tokenizer missing <s> token");
    const token_streaming_pad = tokenizer.tokenToId("[STREAMING_PAD]") orelse @panic("tokenizer missing [STREAMING_PAD] token");
    tokens[0] = token_bos;
    @memset(tokens[1..], token_streaming_pad);

    // -- Compiled models
    var compiled_mel_spectrum = try compiled_mel_spectrum_future.await(io);
    defer compiled_mel_spectrum.deinit();

    var compiled_conv_stem = try compiled_conv_stem_future.await(io);
    defer compiled_conv_stem.deinit();

    var compiled_conv_stem_step = try compiled_conv_stem_step_future.await(io);
    defer compiled_conv_stem_step.deinit();

    var compiled_encoder_prefill = try compiled_encoder_prefill_future.await(io);
    defer compiled_encoder_prefill.deinit();

    var compiled_encoder_step = try compiled_encoder_step_future.await(io);
    defer compiled_encoder_step.deinit();

    var compiled_adapter = try compiled_adapter_future.await(io);
    defer compiled_adapter.deinit();

    var compiled_adapter_step = try compiled_adapter_step_future.await(io);
    defer compiled_adapter_step.deinit();

    var compiled_decoder_prefill, var compiled_decoder_decode = try compiled_decoder_future.await(io);
    defer compiled_decoder_prefill.deinit();
    defer compiled_decoder_decode.deinit();

    // -- Buffers
    var mel_spectrum_buffers = try mel_spectrum_buffers_future.await(io);
    defer LogMelSpectrogram.unload(&mel_spectrum_buffers);

    var encoder_buffers = try encoder_buffers_future.await(io);
    defer Encoder.unload(&encoder_buffers, allocator);

    var adapter_buffers = try adapter_buffers_future.await(io);
    defer Adapter.unload(&adapter_buffers);

    var decoder_buffers = try decoder_buffers_future.await(io);
    defer Decoder.unload(&decoder_buffers, allocator);

    var enc_attention_metadata_buffers: zml.Bufferized(zml.attention.Metadata) = try enc_attention_metadata.initBuffer(io, platform);
    defer zml.attention.Metadata.deinitBuffer(&enc_attention_metadata_buffers);

    var dec_attention_metadata_buffers: zml.Bufferized(zml.attention.Metadata) = try dec_attention_metadata.initBuffer(io, platform);
    defer zml.attention.Metadata.deinitBuffer(&dec_attention_metadata_buffers);

    progress.end();

    try voxtral.runPipeline(
        allocator,
        io,
        platform,
        config,
        &tokenizer,
        padded_wav,
        audio_len,
        tokens,
        n_delay_tokens,
        &compiled_mel_spectrum,
        &compiled_conv_stem,
        &compiled_conv_stem_step,
        &compiled_encoder_prefill,
        &compiled_encoder_step,
        &compiled_adapter,
        &compiled_adapter_step,
        &compiled_decoder_prefill,
        &compiled_decoder_decode,
        &mel_spectrum_buffers,
        &encoder_buffers,
        &adapter_buffers,
        &decoder_buffers,
        enc_kv_cache,
        dec_kv_cache,
        &enc_attention_metadata_buffers,
        &dec_attention_metadata_buffers,
    );
}
