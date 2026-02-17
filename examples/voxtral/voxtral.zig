const std = @import("std");
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

pub fn compileMelStep(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: LogMelSpectrogram, sp: StreamParams, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling mel step...", 1);
    defer node.end();

    return try platform.compile(allocator, io, model, .melStep, .{Tensor.init(.{sp.chunk_audio}, .f32).withTags(.{.samples})});
}

pub fn compileConvStem(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Encoder, num_mel_frames: u32, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling conv stem...", 1);
    defer node.end();

    return try platform.compile(allocator, io, model, .convStem, .{
        Tensor.init(.{ .channels = 128, .time = num_mel_frames }, .f32),
    });
}

pub fn compileConvStemStep(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Encoder, sp: StreamParams, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling conv stem step...", 1);
    defer node.end();

    return try platform.compile(allocator, io, model, .convStemStep, .{
        Tensor.init(.{ .channels = 128, .time = sp.chunk_mel }, .f32),
    });
}

/// Encoder prefill: processes prompt_len * dsf frames at once, populating the KV cache.
pub fn compileEncoderPrefill(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Encoder, prompt_len: u32, enc_kv_cache: KvCache, attention_metadata: zml.attention.Metadata, attention_parameters: zml.attention.Parameters, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling encoder prefill...", 1);
    defer node.end();

    const enc_cfg = model.config.encoder();
    const dsf = model.config.downsample_factor();

    return try platform.compile(allocator, io, model, .transformer, .{
        Tensor.init(.{ .s = prompt_len * dsf, .d = enc_cfg.dim }, .bf16),
        Tensor.init(.{}, .u32),
        enc_kv_cache,
        attention_metadata,
        attention_parameters,
    });
}

/// Encoder step: processes dsf frames with KV cache (for streaming decode).
pub fn compileEncoderStep(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Encoder, enc_kv_cache: KvCache, attention_metadata: zml.attention.Metadata, attention_parameters: zml.attention.Parameters, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling encoder step...", 1);
    defer node.end();

    const enc_cfg = model.config.encoder();
    const dsf = model.config.downsample_factor();

    return try platform.compile(allocator, io, model, .transformer, .{
        Tensor.init(.{ .s = dsf, .d = enc_cfg.dim }, .bf16),
        Tensor.init(.{}, .u32),
        enc_kv_cache,
        attention_metadata,
        attention_parameters,
    });
}

/// Adapter batch: processes prompt_len * dsf encoder frames -> prompt_len embeddings.
pub fn compileAdapter(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Adapter, prompt_len: u32, config: Config, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling adapter...", 1);
    defer node.end();

    const dsf = config.downsample_factor();

    return try platform.compile(allocator, io, model, .forward, .{
        Tensor.init(.{ .s = prompt_len * dsf, .d = config.encoder().dim }, .bf16),
    });
}

/// Adapter step: processes dsf encoder frames -> 1 embedding (for streaming decode).
pub fn compileAdapterStep(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Adapter, config: Config, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling adapter step...", 1);
    defer node.end();

    const dsf = config.downsample_factor();

    return try platform.compile(allocator, io, model, .forward, .{
        Tensor.init(.{ .s = dsf, .d = config.encoder().dim }, .bf16),
    });
}

/// Decoder prefill (s=prompt_len) + decode (s=1), compiled concurrently.
pub fn compileDecoder(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Decoder, prompt_len: u32, dec_kv_cache: KvCache, attention_metadata: zml.attention.Metadata, attention_parameters: zml.attention.Parameters, progress: *std.Progress.Node) !struct { zml.Exe, zml.Exe } {
    const dim = model.config.dim;

    const compilePrefill = struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: *const zml.Platform, model_: Decoder, s: u32, kv_cache_: KvCache, dim_: u32, attention_metadata_: zml.attention.Metadata, attention_parameters_: zml.attention.Parameters, progress_: *std.Progress.Node) !zml.Exe {
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
                attention_metadata_,
                attention_parameters_,
            });
        }
    }.call;

    const compileDecode = struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: *const zml.Platform, model_: Decoder, kv_cache_: KvCache, dim_: u32, attention_metadata_: zml.attention.Metadata, attention_parameters_: zml.attention.Parameters, progress_: *std.Progress.Node) !zml.Exe {
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
                attention_metadata_,
                attention_parameters_,
            });
        }
    }.call;

    var prefill_future = try io.concurrent(compilePrefill, .{ allocator, io, platform, model, prompt_len, dec_kv_cache, dim, attention_metadata, attention_parameters, progress });
    var decode_future = try io.concurrent(compileDecode, .{ allocator, io, platform, model, dec_kv_cache, dim, attention_metadata, attention_parameters, progress });

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

/// Prepend n_fft/2 reflected samples at the start and append n_fft/2 zeros at the end.
/// Left reflect skips boundary sample (index 0) to match Tensor reflect padding in forward().
/// Right zero-padding ensures STFT windows at the tail have enough samples.
pub fn reflectPadAudio(allocator: std.mem.Allocator, audio: []const f32, pad: usize) ![]f32 {
    const result = try allocator.alloc(f32, pad + audio.len + pad);
    // Reflect: result[i] = audio[pad - i] for i in 0..pad
    // i=0 → audio[pad], i=1 → audio[pad-1], ..., i=pad-1 → audio[1]
    for (0..pad) |i| {
        result[i] = audio[pad - i];
    }
    @memcpy(result[pad..][0..audio.len], audio);
    @memset(result[pad + audio.len ..], 0);
    return result;
}

/// Run mel spectrogram chunk-by-chunk and download result to host memory.
/// Returns mel data as bytes (channels-major: [128][prompt_len * mel_per_step]). Caller owns the memory.
pub fn runMelPrefill(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sp: StreamParams,
    padded_audio: []const f32,
    compiled_mel_step: *zml.Exe,
    mel_spectrum_buffers: *zml.Bufferized(LogMelSpectrogram),
) ![]u8 {
    const prompt_len = sp.prompt_len;
    const f32_bytes: usize = @sizeOf(f32);
    const prefill_mel_frames: u32 = prompt_len * sp.mel_per_step;

    log.info("Running mel prefill ({} chunks of {} frames, keeping {} per chunk)...", .{ prompt_len, sp.chunk_mel, sp.mel_per_step });

    // Allocate host mel buffer: [128][prefill_mel_frames] zero-initialized
    const mel_host = try allocator.alloc(u8, @as(usize, 128) * prefill_mel_frames * f32_bytes);
    @memset(mel_host, 0);

    var mel_step_args = try compiled_mel_step.args(allocator);
    defer mel_step_args.deinit(allocator);
    var mel_step_results = try compiled_mel_step.results(allocator);
    defer mel_step_results.deinit(allocator);

    var mel_step_output: zml.Buffer = try .uninitialized(io, platform, .init(.{ .channels = 128, .time = sp.chunk_mel }, .f32), .{});
    defer mel_step_output.deinit();

    for (0..prompt_len) |t| {
        // Audio chunk starts at t * mel_per_step * hop_length in the reflect-padded audio
        const audio_start: usize = t * sp.mel_per_step * sp._hop_length;
        const audio_chunk = padded_audio[audio_start .. audio_start + sp.chunk_audio];

        const audio_slice: zml.Slice = .init(
            .init(.{sp.chunk_audio}, .f32),
            std.mem.sliceAsBytes(audio_chunk),
        );
        var audio_buffer: zml.Buffer = try .fromSlice(io, platform, audio_slice);
        defer audio_buffer.deinit();

        mel_step_args.set(.{ mel_spectrum_buffers, audio_buffer });
        compiled_mel_step.call(mel_step_args, &mel_step_results);
        mel_step_results.fill(.{&mel_step_output});

        // Download chunk and keep first mel_per_step frames for conv stem prefill
        var download = try mel_step_output.toSliceAlloc(allocator, io);
        defer download.free(allocator);
        const src = download.constData();

        const src_stride = @as(usize, sp.chunk_mel) * f32_bytes;
        const dst_stride = @as(usize, prefill_mel_frames) * f32_bytes;
        const copy_len = @as(usize, sp.mel_per_step) * f32_bytes;
        const dst_offset = t * sp.mel_per_step * f32_bytes;
        for (0..128) |c| {
            @memcpy(mel_host[c * dst_stride + dst_offset ..][0..copy_len], src[c * src_stride ..][0..copy_len]);
        }
    }
    log.info("Mel prefill done ({} frames).", .{prefill_mel_frames});

    return mel_host;
}

/// Run the prefill phase: conv stem -> encoder -> adapter -> decoder prefill.
/// Populates KV caches and produces the first generated token into generated_token_slice.
pub fn runPrefill(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    config: Config,
    sp: StreamParams,
    mel_host: []const u8,
    tokens: []const u32,
    compiled_conv_stem: *zml.Exe,
    compiled_encoder_prefill: *zml.Exe,
    compiled_adapter: *zml.Exe,
    compiled_decoder_prefill: *zml.Exe,
    encoder_buffers: *zml.Bufferized(Encoder),
    adapter_buffers: *zml.Bufferized(Adapter),
    decoder_buffers: *zml.Bufferized(Decoder),
    enc_kv_buffers: *zml.Bufferized(KvCache),
    dec_kv_buffers: *zml.Bufferized(KvCache),
    t_cond_buffer: zml.Buffer,
    rng_buffers: *zml.Bufferized(Tensor.Rng),
    enc_attention_metadata_buffers: *zml.Bufferized(zml.attention.Metadata),
    dec_attention_metadata_buffers: *zml.Bufferized(zml.attention.Metadata),
    generated_token_slice: zml.Slice,
) !void {
    const enc_dim: u32 = config.encoder().dim;
    const prompt_len = sp.prompt_len;
    const f32_bytes: usize = @sizeOf(f32);

    // Conv stem prefill: mel_host is already [128][prompt_len*mel_per_step]
    log.info("Running conv stem prefill ({} mel frames)...", .{prompt_len * sp.mel_per_step});
    var conv_prefill_output: zml.Buffer = undefined;
    {
        const prefill_mel_time: u32 = prompt_len * sp.mel_per_step;
        const mel_slice: zml.Slice = .init(
            .init(.{ .channels = 128, .time = prefill_mel_time }, .f32),
            mel_host[0 .. @as(usize, 128) * prefill_mel_time * f32_bytes],
        );
        var mel_buffer: zml.Buffer = try .fromSlice(io, platform, mel_slice);
        defer mel_buffer.deinit();

        const prefill_enc_frames = prompt_len * sp.dsf;
        conv_prefill_output = try .uninitialized(io, platform, .init(.{ .s = prefill_enc_frames, .d = enc_dim }, .bf16), .{});

        var args = try compiled_conv_stem.args(allocator);
        defer args.deinit(allocator);
        var results = try compiled_conv_stem.results(allocator);
        defer results.deinit(allocator);

        args.set(.{ encoder_buffers, mel_buffer });
        compiled_conv_stem.call(args, &results);
        results.fill(.{&conv_prefill_output});
    }
    defer conv_prefill_output.deinit();
    log.info("Conv stem prefill done.", .{});

    // Encoder prefill: first prompt_len*dsf frames through transformer with KV cache
    log.info("Running encoder prefill ({} frames)...", .{prompt_len * sp.dsf});
    var encoder_prefill_output: zml.Buffer = undefined;
    {
        var enc_token_index: zml.Buffer = try .scalar(io, platform, @as(u32, 0), .u32);
        defer enc_token_index.deinit();

        const prefill_frames = prompt_len * sp.dsf;
        encoder_prefill_output = try .uninitialized(io, platform, .init(.{ .s = prefill_frames, .d = enc_dim }, .bf16), .{});

        var args = try compiled_encoder_prefill.args(allocator);
        defer args.deinit(allocator);
        var results = try compiled_encoder_prefill.results(allocator);
        defer results.deinit(allocator);

        args.set(.{ encoder_buffers, conv_prefill_output, enc_token_index, enc_kv_buffers, enc_attention_metadata_buffers });
        compiled_encoder_prefill.call(args, &results);
        results.fill(.{ &encoder_prefill_output, enc_kv_buffers });
    }
    defer encoder_prefill_output.deinit();
    log.info("Encoder prefill done.", .{});

    // Adapter batch: encoder prefill output -> prompt_len audio embeddings
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

    // Decoder prefill: all prompt tokens + audio embeddings -> first generated token
    log.info("Running decoder prefill...", .{});
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

        args.set(.{ decoder_buffers, token_buffer, adapter_prefill_output, token_index_buffer, dec_kv_buffers, t_cond_buffer, rng_buffers.*, dec_attention_metadata_buffers });
        compiled_decoder_prefill.call(args, &results);
        results.fill(.{ &prefill_output, dec_kv_buffers, rng_buffers });

        // Extract last token from prefill output (first generated token)
        try prefill_output.toSlice(io, prefill_output_slice);
        log.info("Prefill output tokens: {any}", .{prefill_output_slice.items(u32)[0..prompt_len]});
        generated_token_slice.items(u32)[0] = prefill_output_slice.items(u32)[prompt_len - 1];
        log.info("First generated token (from prefill): {}", .{generated_token_slice.items(u32)[0]});
    }
    log.info("Decoder prefill done.\n", .{});
}

/// Run the streaming decode loop: step-by-step conv_stem -> encoder -> adapter -> decoder.
pub fn runGenerationLoop(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    config: Config,
    sp: StreamParams,
    tokenizer: *zml.tokenizer.Tokenizer,
    initial_audio_history: []const f32,
    stdin_reader: *std.Io.Reader,
    compiled_mel_step: *zml.Exe,
    compiled_conv_stem_step: *zml.Exe,
    compiled_encoder_step: *zml.Exe,
    compiled_adapter_step: *zml.Exe,
    compiled_decoder_decode: *zml.Exe,
    mel_spectrum_buffers: *zml.Bufferized(LogMelSpectrogram),
    encoder_buffers: *zml.Bufferized(Encoder),
    adapter_buffers: *zml.Bufferized(Adapter),
    decoder_buffers: *zml.Bufferized(Decoder),
    enc_kv_buffers: *zml.Bufferized(KvCache),
    dec_kv_buffers: *zml.Bufferized(KvCache),
    t_cond_buffer: zml.Buffer,
    rng_buffers: *zml.Bufferized(Tensor.Rng),
    enc_attention_metadata_buffers: *zml.Bufferized(zml.attention.Metadata),
    dec_attention_metadata_buffers: *zml.Bufferized(zml.attention.Metadata),
    generated_token_slice: zml.Slice,
) !void {
    const enc_dim: u32 = config.encoder().dim;
    const prompt_len = sp.prompt_len;
    const new_audio_per_step: usize = @as(usize, sp.mel_per_step) * sp._hop_length;
    const audio_overlap: usize = sp.chunk_audio - new_audio_per_step;

    // Sliding audio buffer: [overlap from previous step | new samples]
    const audio_buf = try allocator.alloc(f32, sp.chunk_audio);
    defer allocator.free(audio_buf);
    @memcpy(audio_buf[0..audio_overlap], initial_audio_history);
    @memset(audio_buf[audio_overlap..], 0);

    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    const eos_token = tokenizer.tokenToId("</s>") orelse @panic("tokenizer missing </s> token");
    const streaming_pad_token = tokenizer.tokenToId("[STREAMING_PAD]") orelse @panic("tokenizer missing [STREAMING_PAD] token");
    const streaming_word_token = tokenizer.tokenToId("[STREAMING_WORD]");

    // log.info("Running decode loop (steps {}..{})...", .{ prompt_len, total_steps });
    const decode_start: std.Io.Timestamp = .now(io, .awake);

    var mel_step_args = try compiled_mel_step.args(allocator);
    defer mel_step_args.deinit(allocator);
    var mel_step_results = try compiled_mel_step.results(allocator);
    defer mel_step_results.deinit(allocator);

    var conv_step_args = try compiled_conv_stem_step.args(allocator);
    defer conv_step_args.deinit(allocator);
    var conv_step_results = try compiled_conv_stem_step.results(allocator);
    defer conv_step_results.deinit(allocator);

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

    var mel_step_output: zml.Buffer = try .uninitialized(io, platform, .init(.{ .channels = 128, .time = sp.chunk_mel }, .f32), .{});
    defer mel_step_output.deinit();

    var conv_step_output: zml.Buffer = try .uninitialized(io, platform, .init(.{ .s = sp.dsf, .d = enc_dim }, .bf16), .{});
    defer conv_step_output.deinit();

    var enc_step_output: zml.Buffer = try .uninitialized(io, platform, .init(.{ .s = sp.dsf, .d = enc_dim }, .bf16), .{});
    defer enc_step_output.deinit();

    var adapter_step_output: zml.Buffer = try .uninitialized(io, platform, .init(.{ .s = 1, .d = config.dim }, .bf16), .{});
    defer adapter_step_output.deinit();

    var current_token_buffer: zml.Buffer = try .fromSlice(io, platform, generated_token_slice);
    defer current_token_buffer.deinit();

    log.info("Start of transcription:\n", .{});
    var num_generated: usize = 0;
    var t: usize = prompt_len;
    while (true) : (t += 1) {
        const generated_token = generated_token_slice.items(u32)[0];
        num_generated += 1;

        if (generated_token != streaming_pad_token and (streaming_word_token == null or generated_token != streaming_word_token.?)) {
            if (try tokenizer_decoder.next(generated_token)) |chunk| {
                std.debug.print("{s}", .{chunk});
            }
        }

        if (generated_token == eos_token) {
            log.info("  EOS reached at step={}", .{t});
            break;
        }

        // Mel step: read new audio, slide history buffer, upload full chunk
        const new_samples = readStdinSamples(allocator, stdin_reader, new_audio_per_step) catch |err| switch (err) {
            error.EndOfStream => {
                std.debug.print("\n\n", .{});
                log.info("Audio stream ended at step={}", .{t});
                break;
            },
            else => return err,
        };
        defer allocator.free(new_samples);

        // Shift history left, append new samples
        std.mem.copyForwards(f32, audio_buf[0..audio_overlap], audio_buf[new_audio_per_step..]);
        @memcpy(audio_buf[audio_overlap..], new_samples);

        const audio_slice: zml.Slice = .init(
            .init(.{sp.chunk_audio}, .f32),
            std.mem.sliceAsBytes(audio_buf),
        );
        var audio_buffer: zml.Buffer = try .fromSlice(io, platform, audio_slice);
        defer audio_buffer.deinit();

        mel_step_args.set(.{ mel_spectrum_buffers, audio_buffer });
        compiled_mel_step.call(mel_step_args, &mel_step_results);
        mel_step_results.fill(.{&mel_step_output});

        // Conv stem step: mel step output stays on device → conv stem
        conv_step_args.set(.{ encoder_buffers, mel_step_output });
        compiled_conv_stem_step.call(conv_step_args, &conv_step_results);
        conv_step_results.fill(.{&conv_step_output});

        // Encoder step: conv stem output -> encoded chunk
        var enc_token_index: zml.Buffer = try .scalar(io, platform, @as(u32, @intCast(t)) * sp.dsf, .u32);
        defer enc_token_index.deinit();

        enc_step_args.set(.{ encoder_buffers, conv_step_output, enc_token_index, enc_kv_buffers, enc_attention_metadata_buffers });
        compiled_encoder_step.call(enc_step_args, &enc_step_results);
        enc_step_results.fill(.{ &enc_step_output, enc_kv_buffers });

        // Adapter step: encoded chunk -> 1 audio embedding
        adp_step_args.set(.{ adapter_buffers, enc_step_output });
        compiled_adapter_step.call(adp_step_args, &adp_step_results);
        adp_step_results.fill(.{&adapter_step_output});

        // Decoder decode: 1 token + 1 audio embedding -> next token
        var dec_token_index: zml.Buffer = try .scalar(io, platform, @as(u32, @intCast(t)), .u32);
        defer dec_token_index.deinit();

        decode_args.set(.{ decoder_buffers, current_token_buffer, adapter_step_output, dec_token_index, dec_kv_buffers, t_cond_buffer, rng_buffers.*, dec_attention_metadata_buffers });
        compiled_decoder_decode.call(decode_args, &decode_results);
        decode_results.fill(.{ &current_token_buffer, dec_kv_buffers, rng_buffers });

        try current_token_buffer.toSlice(io, generated_token_slice);
    }
    const decode_duration = decode_start.untilNow(io, .awake);
    std.debug.print("\n", .{});
    log.info("Decode done. Generated {} tokens in {D}: {:.3}tok/s", .{
        num_generated,
        stdx.fmt.fmtDuration(decode_duration),
        stdx.Io.Duration.hzFloat(stdx.Io.Duration.div(decode_duration, num_generated)),
    });
}

pub fn runPipeline(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    config: Config,
    tokenizer: *zml.tokenizer.Tokenizer,
    // padded_audio: []const f32,
    // audio_len: usize,
    tokens: []const u32,
    sp: StreamParams,
    compiled_mel_step: *zml.Exe,
    compiled_conv_stem: *zml.Exe,
    compiled_conv_stem_step: *zml.Exe,
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
    enc_attention_metadata_buffers: *zml.Bufferized(zml.attention.Metadata),
    dec_attention_metadata_buffers: *zml.Bufferized(zml.attention.Metadata),
) !void {
    log.info("Running inference pipeline...", .{});

    var stdin_buff: [2560 * 4]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(io, &stdin_buff);
    const stdin_reader_interface = &stdin_reader.interface;

    // 0. Read initial audio from stdin and build padded buffer for prefill (like loadAndPadWav)
    const n_fft: usize = config.audio().window_size;
    const reflect_pad: usize = n_fft / 2;

    const n_prefill_stdin: usize = @as(usize, sp.n_delay_tokens + 1) * sp.raw_audio_length_per_tok;
    const stdin_audio = try readStdinSamples(allocator, stdin_reader_interface, n_prefill_stdin);
    defer allocator.free(stdin_audio);

    // Pad: [left_pad zeros][stdin audio][right_pad zeros]
    const right_pad: usize = @as(usize, sp.n_right_pad_tokens) * sp.raw_audio_length_per_tok;
    const prefill_audio_len: usize = @as(usize, sp.left_pad) + stdin_audio.len + right_pad;
    const prefill_audio = try allocator.alloc(f32, prefill_audio_len);
    defer allocator.free(prefill_audio);
    @memset(prefill_audio, 0);
    @memcpy(prefill_audio[@as(usize, sp.left_pad)..][0..stdin_audio.len], stdin_audio);

    const reflect_padded_audio = try reflectPadAudio(allocator, prefill_audio, reflect_pad);
    defer allocator.free(reflect_padded_audio);

    // 1. Mel prefill (chunk-by-chunk)
    const mel_host = try runMelPrefill(allocator, io, platform, sp, reflect_padded_audio, compiled_mel_step, mel_spectrum_buffers);
    defer allocator.free(mel_host);

    // 2. Init KV caches + t_cond + RNG
    var enc_kv_buffers = try enc_kv_cache.initBuffer(io, platform);
    defer KvCache.deinitBuffer(&enc_kv_buffers);

    var dec_kv_buffers = try dec_kv_cache.initBuffer(io, platform);
    defer KvCache.deinitBuffer(&dec_kv_buffers);

    const t_cond_data = computeTimeEmbedding(allocator, @as(f32, @floatFromInt(sp.n_delay_tokens)), config.dim);
    defer allocator.free(t_cond_data);
    const t_cond_slice: zml.Slice = .init(
        .init(.{config.dim}, .f32),
        std.mem.sliceAsBytes(t_cond_data),
    );
    var t_cond_buffer: zml.Buffer = try .fromSlice(io, platform, t_cond_slice);
    defer t_cond_buffer.deinit();

    var rng_buffers = try Tensor.Rng.initBuffer(platform, 0, io);
    defer Tensor.Rng.deinitBuffer(&rng_buffers);

    var generated_token_slice: zml.Slice = try .alloc(allocator, .init(.{@as(u32, 1)}, .u32));
    defer generated_token_slice.free(allocator);

    // 3. Prefill
    try runPrefill(
        allocator,
        io,
        platform,
        config,
        sp,
        mel_host,
        tokens,
        compiled_conv_stem,
        compiled_encoder_prefill,
        compiled_adapter,
        compiled_decoder_prefill,
        encoder_buffers,
        adapter_buffers,
        decoder_buffers,
        &enc_kv_buffers,
        &dec_kv_buffers,
        t_cond_buffer,
        &rng_buffers,
        enc_attention_metadata_buffers,
        dec_attention_metadata_buffers,
        generated_token_slice,
    );

    // 4. Generation loop — extract audio history for sliding buffer
    const new_audio_per_step: usize = @as(usize, sp.mel_per_step) * sp._hop_length;
    const audio_overlap: usize = sp.chunk_audio - new_audio_per_step;
    const history_start: usize = @as(usize, sp.prompt_len) * sp.mel_per_step * sp._hop_length;
    const initial_audio_history = reflect_padded_audio[history_start .. history_start + audio_overlap];

    try runGenerationLoop(
        allocator,
        io,
        platform,
        config,
        sp,
        tokenizer,
        initial_audio_history,
        stdin_reader_interface,
        compiled_mel_step,
        compiled_conv_stem_step,
        compiled_encoder_step,
        compiled_adapter_step,
        compiled_decoder_decode,
        mel_spectrum_buffers,
        encoder_buffers,
        adapter_buffers,
        decoder_buffers,
        &enc_kv_buffers,
        &dec_kv_buffers,
        t_cond_buffer,
        &rng_buffers,
        enc_attention_metadata_buffers,
        dec_attention_metadata_buffers,
        generated_token_slice,
    );
}

fn readStdinSamples(allocator: std.mem.Allocator, reader: *std.Io.Reader, n_samples: usize) ![]f32 {
    const byte_count = n_samples * 2;
    const raw = try allocator.alloc(u8, byte_count);
    defer allocator.free(raw);
    try reader.readSliceAll(raw);

    const samples = try allocator.alloc(f32, n_samples);
    for (0..n_samples) |i| {
        const offset = i * 2;
        samples[i] = @as(f32, @floatFromInt(std.mem.bytesToValue(i16, raw[offset..][0..2]))) / 32768.0;
    }
    return samples;
}
