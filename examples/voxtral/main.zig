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
const Embedder = dec.Embedder;

const voxtral = @import("voxtral.zig");

const CliArgs = struct {
    input: []const u8,
    model: []const u8,
    transcription_delay_ms: f32 = 480.0,
};

// Streaming protocol constants
const token_bos: u32 = 1;
const token_streaming_pad: u32 = 32;
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
    tokens[0] = token_bos;
    @memset(tokens[1..], token_streaming_pad);

    var wav_buffer: [4096]u8 = undefined;
    var reader = file.reader(io, &wav_buffer);

    const left_pad = n_left_pad_tokens * raw_audio_length_per_tok;
    const padded_wav = try wav_utils.loadAndPadWav(allocator, &reader.interface, left_pad, n_right_pad_tokens, raw_audio_length_per_tok);
    defer allocator.free(padded_wav);
    const audio_len = padded_wav.len;

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    log.info("Selected platform {f}\n", .{platform.fmtVerbose()});

    const num_frames = audio_len / config.audio().hop_length;
    const encoder_seq_len: u32 = @intCast((num_frames / 2) - (num_frames / 2) % config.downsample_factor());
    const adapter_seq_len = encoder_seq_len / config.downsample_factor();

    var melspectro_model: LogMelSpectrogram = .init(config);
    const encoder_prefix = "mm_streams_embeddings.embedding_module.whisper_encoder";
    var encoder_model: Encoder = .init(allocator, model_store.view().withPrefix(encoder_prefix), config);
    defer encoder_model.deinit(allocator);

    const adapter: Adapter = .init(model_store.view());
    const embedder: Embedder = .init(model_store.view());

    var model: Decoder = .init(allocator, model_store.view(), config);
    defer model.deinit(allocator);

    const dtype = model.tok_embeddings.dtype();
    const model_params: voxtral.VoxtralParameters = .{
        .prefill_embeds = .init(.{ .s = @as(u32, @intCast(tokens.len)), .d = config.dim }, .bf16),
        .decode_embeds = .init(.{ .s = 1, .d = config.dim }, .bf16),
        .token_index = .init(.{}, .u32),
        .kv_cache = .init(.init(.{
            .layer = model.layers.len,
            .k = config.sliding_window,
            .h = config.n_kv_heads,
            .hd = config.head_dim,
        }, dtype)),
        .t_cond = Tensor.init(.{ .d = config.dim }, .f32),
    };

    // Launch concurrent compilation and buffer loading
    var tokenizer_future = try io.concurrent(voxtral.loadTokenizer, .{ allocator, io, model_dir, &progress });
    var compiled_mel_spectrum_future = try io.concurrent(voxtral.compileMelSpectrum, .{ allocator, io, platform, melspectro_model, audio_len, &progress });
    var compiled_encoder_future = try io.concurrent(voxtral.compileEncoder, .{ allocator, io, platform, encoder_model, audio_len, &progress });
    var compiled_adapter_future = try io.concurrent(voxtral.compileAdapter, .{ allocator, io, platform, adapter, encoder_seq_len, config, &progress });
    var compiled_embedder_future = try io.concurrent(voxtral.compileEmbedder, .{ allocator, io, platform, embedder, adapter_seq_len, @as(u32, @intCast(tokens.len)), config, &progress });
    var compiled_decode_embedder_future = try io.concurrent(voxtral.compileEmbedder, .{ allocator, io, platform, embedder, adapter_seq_len, @as(u32, 1), config, &progress });
    var compiled_decoder_future = try io.concurrent(voxtral.compileDecoder, .{ allocator, io, platform, model, model_params, &progress });

    var mel_spectrum_buffers_future = try io.concurrent(LogMelSpectrogram.load, .{ &melspectro_model, io, platform });
    var encoder_buffers_future = try io.concurrent(Encoder.load, .{ &encoder_model, allocator, io, platform, &model_store, &progress });
    var adapter_buffers_future = try io.concurrent(Adapter.load, .{ &adapter, allocator, io, platform, &model_store, &progress });
    var embedder_buffers_future = try io.concurrent(Embedder.load, .{ &embedder, allocator, io, platform, &model_store, &progress });
    var decoder_buffers_future = try io.concurrent(Decoder.load, .{ &model, allocator, io, platform, &model_store, &progress });

    // Await tokenizer
    var tokenizer = try tokenizer_future.await(io);
    defer tokenizer.deinit();

    // -- Compiled models
    var compiled_mel_spectrum = try compiled_mel_spectrum_future.await(io);
    defer compiled_mel_spectrum.deinit();

    var compiled_encoder = try compiled_encoder_future.await(io);
    defer compiled_encoder.deinit();

    var compiled_adapter = try compiled_adapter_future.await(io);
    defer compiled_adapter.deinit();

    var compiled_embedder = try compiled_embedder_future.await(io);
    defer compiled_embedder.deinit();

    var compiled_decode_embedder = try compiled_decode_embedder_future.await(io);
    defer compiled_decode_embedder.deinit();

    var compiled_prefill, var compiled_decoder = try compiled_decoder_future.await(io);
    defer compiled_prefill.deinit();
    defer compiled_decoder.deinit();

    // -- Buffers
    var mel_spectrum_buffers = try mel_spectrum_buffers_future.await(io);
    defer LogMelSpectrogram.unload(&mel_spectrum_buffers);

    var encoder_buffers = try encoder_buffers_future.await(io);
    defer Encoder.unload(&encoder_buffers, allocator);

    var adapter_buffers = try adapter_buffers_future.await(io);
    defer Adapter.unload(&adapter_buffers);

    var embedder_buffers = try embedder_buffers_future.await(io);
    defer Embedder.unload(&embedder_buffers);

    var decoder_buffers = try decoder_buffers_future.await(io);
    defer Decoder.unload(&decoder_buffers, allocator);

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
        &compiled_encoder,
        &compiled_adapter,
        &compiled_embedder,
        &compiled_decode_embedder,
        &compiled_prefill,
        &compiled_decoder,
        &mel_spectrum_buffers,
        &encoder_buffers,
        &adapter_buffers,
        &embedder_buffers,
        &decoder_buffers,
        model_params,
    );
}
