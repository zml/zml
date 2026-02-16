const std = @import("std");
const log = std.log;
const cfg = @import("config.zig");
const Config = cfg.Config;

const zml = @import("zml");
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

pub fn compileMelSpectrum(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: LogMelSpectrogram, padded_audio_len: usize, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling mel spectrogram...", 1);
    defer node.end();

    return try platform.compile(allocator, io, model, .forward, .{Tensor.init(.{padded_audio_len}, .f32).withTags(.{.samples})});
}

pub fn compileConvStem(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Encoder, padded_audio_len: usize, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling conv stem...", 1);
    defer node.end();

    const num_frames = padded_audio_len / model.config.audio().hop_length;

    return try platform.compile(allocator, io, model, .convStem, .{
        Tensor.init(.{ .channels = 128, .time = num_frames }, .f32),
    });
}

/// Encoder prefill: processes prompt_len * dsf frames at once, populating the KV cache.
pub fn compileEncoderPrefill(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Encoder, prompt_len: u32, enc_kv_cache: KvCache, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling encoder prefill...", 1);
    defer node.end();

    const enc_cfg = model.config.encoder();
    const dsf = model.config.downsample_factor();

    return try platform.compile(allocator, io, model, .transformerStep, .{
        Tensor.init(.{ .s = prompt_len * dsf, .d = enc_cfg.dim }, .bf16),
        Tensor.init(.{}, .u32),
        enc_kv_cache,
    });
}

/// Encoder step: processes dsf frames with KV cache (for streaming decode).
pub fn compileEncoderStep(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Encoder, enc_kv_cache: KvCache, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling encoder step...", 1);
    defer node.end();

    const enc_cfg = model.config.encoder();
    const dsf = model.config.downsample_factor();

    return try platform.compile(allocator, io, model, .transformerStep, .{
        Tensor.init(.{ .s = dsf, .d = enc_cfg.dim }, .bf16),
        Tensor.init(.{}, .u32),
        enc_kv_cache,
    });
}

/// Adapter batch: processes prompt_len * dsf encoder frames → prompt_len embeddings.
pub fn compileAdapter(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Adapter, prompt_len: u32, config: Config, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling adapter...", 1);
    defer node.end();

    const dsf = config.downsample_factor();

    return try platform.compile(allocator, io, model, .forward, .{
        Tensor.init(.{ .s = prompt_len * dsf, .d = config.encoder().dim }, .bf16),
    });
}

/// Adapter step: processes dsf encoder frames → 1 embedding (for streaming decode).
pub fn compileAdapterStep(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Adapter, config: Config, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling adapter step...", 1);
    defer node.end();

    const dsf = config.downsample_factor();

    return try platform.compile(allocator, io, model, .forwardStep, .{
        Tensor.init(.{ .s = dsf, .d = config.encoder().dim }, .bf16),
    });
}

/// Decoder prefill (s=prompt_len) + decode (s=1), compiled concurrently.
pub fn compileDecoder(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Decoder, prompt_len: u32, dec_kv_cache: KvCache, progress: *std.Progress.Node) !struct { zml.Exe, zml.Exe } {
    const dim = model.config.dim;

    const compilePrefill = struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: *const zml.Platform, model_: Decoder, s: u32, kv_cache_: KvCache, dim_: u32, progress_: *std.Progress.Node) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling decoder prefill...", 1);
            defer node_.end();

            return try platform_.compile(allocator_, io_, model_, .forward, .{
                Tensor.init(.{ .s = s }, .u32),
                Tensor.init(.{ .s = s, .d = dim_ }, .bf16),
                Tensor.init(.{}, .u32),
                kv_cache_,
                Tensor.init(.{ .d = dim_ }, .f32),
                Tensor.Rng.init(),
            });
        }
    }.call;

    const compileDecode = struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: *const zml.Platform, model_: Decoder, kv_cache_: KvCache, dim_: u32, progress_: *std.Progress.Node) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling decoder decode...", 1);
            defer node_.end();

            return try platform_.compile(allocator_, io_, model_, .forward, .{
                Tensor.init(.{ .s = 1 }, .u32),
                Tensor.init(.{ .s = 1, .d = dim_ }, .bf16),
                Tensor.init(.{}, .u32),
                kv_cache_,
                Tensor.init(.{ .d = dim_ }, .f32),
                Tensor.Rng.init(),
            });
        }
    }.call;

    var prefill_future = try io.concurrent(compilePrefill, .{ allocator, io, platform, model, prompt_len, dec_kv_cache, dim, progress });
    var decode_future = try io.concurrent(compileDecode, .{ allocator, io, platform, model, dec_kv_cache, dim, progress });

    return .{
        try prefill_future.await(io),
        try decode_future.await(io),
    };
}

pub fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, progress: *std.Progress.Node) !zml.tokenizer.Tokenizer {
    progress.increaseEstimatedTotalItems(1);

    var node = progress.start("Loading tokenizer...", 1);
    defer node.end();

    const bytes = b: {
        const file = try dir.openFile(io, "tokenizer.json", .{});
        defer file.close(io);

        var reader = file.reader(io, &.{});
        break :b try reader.interface.readAlloc(allocator, try file.length(io));
    };
    defer allocator.free(bytes);

    return try .fromBytes(allocator, io, bytes);
}

/// Sinusoidal time embedding: encodes a scalar t into a [dim]-dimensional vector.
/// out[i] = cos(t * inv_freq[i]) for i < half_dim
/// out[i] = sin(t * inv_freq[i-half_dim]) for i >= half_dim
/// where inv_freq[i] = exp(-log(theta) * i / half_dim)
pub fn computeTimeEmbedding(allocator: std.mem.Allocator, t_value: f32, dim: u32) []f32 {
    const half_dim = dim / 2;
    const result = allocator.alloc(f32, dim) catch unreachable;

    for (0..half_dim) |i| {
        const freq = @exp(-@log(@as(f32, 10000.0)) * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half_dim)));
        const angle = t_value * freq;
        result[i] = @cos(angle);
        result[i + half_dim] = @sin(angle);
    }

    return result;
}

pub fn runPipeline(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    config: Config,
    tokenizer: *zml.tokenizer.Tokenizer,
    padded_audio: []const f32,
    audio_len: usize,
    tokens: []const u32,
    n_delay_tokens: u32,
    compiled_mel_spectrum: *zml.Exe,
    compiled_conv_stem: *zml.Exe,
    compiled_encoder_prefill: *zml.Exe,
    compiled_encoder_step: *zml.Exe,
    compiled_adapter: *zml.Exe,
    compiled_adapter_step: *zml.Exe,
    compiled_decoder_prefill: *zml.Exe,
    compiled_decoder_decode: *zml.Exe,
    mel_spectrum_buffers: *zml.Bufferized(LogMelSpectrogram),
    encoder_buffers: *zml.Bufferized(Encoder),
    adapter_buffers: *zml.Bufferized(Adapter),
    decoder_buffers: *zml.Bufferized(Decoder),
    enc_kv_cache: KvCache,
    dec_kv_cache: KvCache,
) !void {
    log.info("Running inference pipeline...", .{});

    const num_frames = audio_len / config.audio().hop_length;
    const dsf: u32 = config.downsample_factor();
    const encoder_seq_len: u32 = @intCast((num_frames + 1) / 2);
    const total_steps = (encoder_seq_len + dsf - 1) / dsf;
    const enc_dim: u32 = config.encoder().dim;
    const prompt_len: u32 = @intCast(tokens.len);

    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    // 1. Mel spectrogram
    log.info("Running mel spectrogram...", .{});
    var mel_output: zml.Buffer = undefined;
    {
        var mel_args = try compiled_mel_spectrum.args(allocator);
        defer mel_args.deinit(allocator);
        var mel_results = try compiled_mel_spectrum.results(allocator);
        defer mel_results.deinit(allocator);

        const audio_slice: zml.Slice = .init(
            .init(.{audio_len}, .f32),
            std.mem.sliceAsBytes(padded_audio),
        );
        var audio_buffer: zml.Buffer = try .fromSlice(io, platform, audio_slice);
        defer audio_buffer.deinit();

        mel_output = try .uninitialized(io, platform, .init(.{ .channels = 128, .time = num_frames }, .f32), .{});

        mel_args.set(.{ mel_spectrum_buffers, audio_buffer });
        compiled_mel_spectrum.call(mel_args, &mel_results);
        mel_results.fill(.{&mel_output});
    }
    defer mel_output.deinit();
    log.info("Mel spectrogram done.", .{});

    // 2. Conv stem
    log.info("Running conv stem...", .{});
    var conv_output: zml.Buffer = undefined;
    {
        var args = try compiled_conv_stem.args(allocator);
        defer args.deinit(allocator);
        var results = try compiled_conv_stem.results(allocator);
        defer results.deinit(allocator);

        conv_output = try .uninitialized(io, platform, .init(.{ .s = encoder_seq_len, .d = enc_dim }, .bf16), .{});

        args.set(.{ encoder_buffers, mel_output });
        compiled_conv_stem.call(args, &results);
        results.fill(.{&conv_output});
    }
    defer conv_output.deinit();
    log.info("Conv stem done.", .{});

    // 3. Download conv output to host (for per-step slicing)
    const bf16_bytes: usize = 2;
    const frame_bytes: usize = @as(usize, enc_dim) * bf16_bytes;
    const chunk_bytes: usize = @as(usize, dsf) * frame_bytes;
    const padded_enc_len: u32 = total_steps * dsf;

    const conv_host = try allocator.alloc(u8, @as(usize, padded_enc_len) * frame_bytes);
    defer allocator.free(conv_host);
    @memset(conv_host, 0); // zero-pad for last chunk if encoder_seq_len % dsf != 0

    {
        var download = try conv_output.toSliceAlloc(allocator, io);
        defer download.free(allocator);
        const src = download.constData();
        @memcpy(conv_host[0..src.len], src);
    }
    log.info("Conv output downloaded to host ({} frames, {} steps).", .{ encoder_seq_len, total_steps });

    // 4. Init KV caches + t_cond + RNG
    var enc_kv_buffers = try enc_kv_cache.initBuffer(io, platform);
    defer KvCache.deinitBuffer(&enc_kv_buffers);

    var dec_kv_buffers = try dec_kv_cache.initBuffer(io, platform);
    defer KvCache.deinitBuffer(&dec_kv_buffers);

    const t_cond_data = computeTimeEmbedding(allocator, @as(f32, @floatFromInt(n_delay_tokens)), config.dim);
    defer allocator.free(t_cond_data);
    const t_cond_slice: zml.Slice = .init(
        .init(.{config.dim}, .f32),
        std.mem.sliceAsBytes(t_cond_data),
    );
    var t_cond_buffer: zml.Buffer = try .fromSlice(io, platform, t_cond_slice);
    defer t_cond_buffer.deinit();

    var rng_buffers = try Tensor.Rng.initBuffer(platform, 0, io);
    defer Tensor.Rng.deinitBuffer(&rng_buffers);

    const eos_token = tokenizer.tokenToId("</s>") orelse @panic("tokenizer missing </s> token");
    const streaming_pad_token = tokenizer.tokenToId("[STREAMING_PAD]") orelse @panic("tokenizer missing [STREAMING_PAD] token");

    // ===== Prefill phase =====
    // 5. Encoder prefill: first prompt_len*dsf frames through transformer with KV cache
    log.info("Running encoder prefill ({} frames)...", .{prompt_len * dsf});
    var encoder_prefill_output: zml.Buffer = undefined;
    {
        const prefill_frames = prompt_len * dsf;
        const prefill_conv_bytes = @as(usize, prefill_frames) * frame_bytes;
        const prefill_conv_slice: zml.Slice = .init(
            .init(.{ .s = prefill_frames, .d = enc_dim }, .bf16),
            conv_host[0..prefill_conv_bytes],
        );
        var prefill_conv_buffer: zml.Buffer = try .fromSlice(io, platform, prefill_conv_slice);
        defer prefill_conv_buffer.deinit();

        var enc_token_index: zml.Buffer = try .scalar(io, platform, @as(u32, 0), .u32);
        defer enc_token_index.deinit();

        encoder_prefill_output = try .uninitialized(io, platform, .init(.{ .s = prefill_frames, .d = enc_dim }, .bf16), .{});

        var args = try compiled_encoder_prefill.args(allocator);
        defer args.deinit(allocator);
        var results = try compiled_encoder_prefill.results(allocator);
        defer results.deinit(allocator);

        args.set(.{ encoder_buffers, prefill_conv_buffer, enc_token_index, &enc_kv_buffers });
        compiled_encoder_prefill.call(args, &results);
        results.fill(.{ &encoder_prefill_output, &enc_kv_buffers });
    }
    defer encoder_prefill_output.deinit();
    log.info("Encoder prefill done.", .{});

    // 6. Adapter batch: encoder prefill output → prompt_len audio embeddings
    log.info("Running adapter...", .{});
    var adapter_prefill_output: zml.Buffer = undefined;
    {
        adapter_prefill_output = try .uninitialized(io, platform, .init(.{ .s = prompt_len, .d = config.dim }, .bf16), .{});

        var args = try compiled_adapter.args(allocator);
        defer args.deinit(allocator);
        var results = try compiled_adapter.results(allocator);
        defer results.deinit(allocator);

        args.set(.{ adapter_buffers, encoder_prefill_output });
        compiled_adapter.call(args, &results);
        results.fill(.{&adapter_prefill_output});
    }
    defer adapter_prefill_output.deinit();
    log.info("Adapter done.", .{});

    // 7. Decoder prefill: all prompt tokens + audio embeddings → first generated token
    log.info("Running decoder prefill...", .{});
    var generated_token_slice: zml.Slice = try .alloc(allocator, .init(.{@as(u32, 1)}, .u32));
    defer generated_token_slice.free(allocator);
    {
        var args = try compiled_decoder_prefill.args(allocator);
        defer args.deinit(allocator);
        var results = try compiled_decoder_prefill.results(allocator);
        defer results.deinit(allocator);

        const token_data_slice: zml.Slice = .init(
            .init(.{prompt_len}, .u32),
            std.mem.sliceAsBytes(tokens),
        );
        var token_buffer: zml.Buffer = try .fromSlice(io, platform, token_data_slice);
        defer token_buffer.deinit();

        var token_index_buffer: zml.Buffer = try .scalar(io, platform, @as(u32, 0), .u32);
        defer token_index_buffer.deinit();

        var prefill_output_slice: zml.Slice = try .alloc(allocator, .init(.{prompt_len}, .u32));
        defer prefill_output_slice.free(allocator);
        var prefill_output: zml.Buffer = try .fromSlice(io, platform, prefill_output_slice);
        defer prefill_output.deinit();

        args.set(.{ decoder_buffers, token_buffer, adapter_prefill_output, token_index_buffer, &dec_kv_buffers, t_cond_buffer, rng_buffers });
        compiled_decoder_prefill.call(args, &results);
        results.fill(.{ &prefill_output, &dec_kv_buffers, &rng_buffers });

        // Extract last token from prefill output (first generated token)
        try prefill_output.toSlice(io, prefill_output_slice);
        log.info("Prefill output tokens: {any}", .{prefill_output_slice.items(u32)[0..prompt_len]});
        generated_token_slice.items(u32)[0] = prefill_output_slice.items(u32)[prompt_len - 1];
        log.info("First generated token (from prefill): {}", .{generated_token_slice.items(u32)[0]});
    }
    log.info("Decoder prefill done.", .{});

    // ===== Decode phase: step-by-step encoder → adapter → decoder =====
    log.info("Running decode loop (steps {}..{})...", .{ prompt_len, total_steps });

    var enc_step_args = try compiled_encoder_step.args(allocator);
    defer enc_step_args.deinit(allocator);
    var enc_step_results = try compiled_encoder_step.results(allocator);
    defer enc_step_results.deinit(allocator);

    var adp_step_args = try compiled_adapter_step.args(allocator);
    defer adp_step_args.deinit(allocator);
    var adp_step_results = try compiled_adapter_step.results(allocator);
    defer adp_step_results.deinit(allocator);

    var decode_args = try compiled_decoder_decode.args(allocator);
    defer decode_args.deinit(allocator);
    var decode_results = try compiled_decoder_decode.results(allocator);
    defer decode_results.deinit(allocator);

    var enc_step_output: zml.Buffer = try .uninitialized(io, platform, .init(.{ .s = dsf, .d = enc_dim }, .bf16), .{});
    defer enc_step_output.deinit();

    var adapter_step_output: zml.Buffer = try .uninitialized(io, platform, .init(.{ .s = 1, .d = config.dim }, .bf16), .{});
    defer adapter_step_output.deinit();

    var current_token_buffer: zml.Buffer = try .fromSlice(io, platform, generated_token_slice);
    defer current_token_buffer.deinit();

    var num_generated: usize = 0;
    for (prompt_len..total_steps) |t| {
        const generated_token = generated_token_slice.items(u32)[0];
        num_generated += 1;

        if (generated_token != streaming_pad_token) {
            if (try tokenizer_decoder.next(generated_token)) |chunk| {
                std.debug.print("{s}", .{chunk});
            }
        }

        if (generated_token == eos_token) {
            log.info("  EOS reached at step={}", .{t});
            break;
        }

        // Encoder step: dsf frames → encoded chunk
        const byte_offset = t * chunk_bytes;
        const chunk_slice: zml.Slice = .init(
            .init(.{ .s = dsf, .d = enc_dim }, .bf16),
            conv_host[byte_offset .. byte_offset + chunk_bytes],
        );
        var chunk_buffer: zml.Buffer = try .fromSlice(io, platform, chunk_slice);
        defer chunk_buffer.deinit();

        var enc_token_index: zml.Buffer = try .scalar(io, platform, @as(u32, @intCast(t)) * dsf, .u32);
        defer enc_token_index.deinit();

        enc_step_args.set(.{ encoder_buffers, chunk_buffer, enc_token_index, &enc_kv_buffers });
        compiled_encoder_step.call(enc_step_args, &enc_step_results);
        enc_step_results.fill(.{ &enc_step_output, &enc_kv_buffers });

        // Adapter step: encoded chunk → 1 audio embedding
        adp_step_args.set(.{ adapter_buffers, enc_step_output });
        compiled_adapter_step.call(adp_step_args, &adp_step_results);
        adp_step_results.fill(.{&adapter_step_output});

        // Decoder decode: 1 token + 1 audio embedding → next token
        var dec_token_index: zml.Buffer = try .scalar(io, platform, @as(u32, @intCast(t)), .u32);
        defer dec_token_index.deinit();

        decode_args.set(.{ decoder_buffers, current_token_buffer, adapter_step_output, dec_token_index, &dec_kv_buffers, t_cond_buffer, rng_buffers });
        compiled_decoder_decode.call(decode_args, &decode_results);
        decode_results.fill(.{ &current_token_buffer, &dec_kv_buffers, &rng_buffers });

        try current_token_buffer.toSlice(io, generated_token_slice);
    }
    std.debug.print("\n", .{});
    log.info("Decode done. Generated {} tokens total.", .{num_generated});
}
