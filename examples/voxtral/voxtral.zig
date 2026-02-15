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

pub fn compileAdapterStep(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Adapter, config: Config, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling adapter step...", 1);
    defer node.end();

    const dsf = config.downsample_factor();

    return try platform.compile(allocator, io, model, .forwardStep, .{
        Tensor.init(.{ .s = dsf, .d = config.encoder().dim }, .bf16),
    });
}

pub fn compileDecoder(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Decoder, dec_kv_cache: KvCache, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling decoder...", 1);
    defer node.end();

    const dim = model.config.dim;

    return try platform.compile(allocator, io, model, .forward, .{
        Tensor.init(.{ .s = 1 }, .u32),
        Tensor.init(.{ .s = 1, .d = dim }, .bf16),
        Tensor.init(.{}, .u32),
        dec_kv_cache,
        Tensor.init(.{ .d = dim }, .f32),
        Tensor.Rng.init(),
    });
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
    compiled_encoder_step: *zml.Exe,
    compiled_adapter_step: *zml.Exe,
    compiled_decoder: *zml.Exe,
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

    // 5. Step-by-step loop
    log.info("Starting step-by-step loop ({} steps, {} prompt tokens)...", .{ total_steps, tokens.len });
    const eos_token = tokenizer.tokenToId("</s>") orelse @panic("tokenizer missing </s> token");
    const streaming_pad_token = tokenizer.tokenToId("[STREAMING_PAD]") orelse @panic("tokenizer missing [STREAMING_PAD] token");

    // Pre-allocate exe args/results (reused across iterations)
    var enc_step_args = try compiled_encoder_step.args(allocator);
    defer enc_step_args.deinit(allocator);
    var enc_step_results = try compiled_encoder_step.results(allocator);
    defer enc_step_results.deinit(allocator);

    var adp_step_args = try compiled_adapter_step.args(allocator);
    defer adp_step_args.deinit(allocator);
    var adp_step_results = try compiled_adapter_step.results(allocator);
    defer adp_step_results.deinit(allocator);

    var dec_args = try compiled_decoder.args(allocator);
    defer dec_args.deinit(allocator);
    var dec_results = try compiled_decoder.results(allocator);
    defer dec_results.deinit(allocator);

    // Pre-allocate step output buffers (reused across iterations)
    var enc_step_output: zml.Buffer = try .uninitialized(io, platform, .init(.{ .s = dsf, .d = enc_dim }, .bf16), .{});
    defer enc_step_output.deinit();

    var adapter_output: zml.Buffer = try .uninitialized(io, platform, .init(.{ .s = 1, .d = config.dim }, .bf16), .{});
    defer adapter_output.deinit();

    var token_slice: zml.Slice = try .alloc(allocator, .init(.{@as(u32, 1)}, .u32));
    defer token_slice.free(allocator);
    var token_output: zml.Buffer = try .uninitialized(io, platform, .init(.{@as(u32, 1)}, .u32), .{});
    defer token_output.deinit();

    var num_generated: usize = 0;

    for (0..total_steps) |t| {
        const t_u32: u32 = @intCast(t);

        // --- Encoder step: process dsf frames through transformer with KV cache ---
        const byte_offset = t * chunk_bytes;
        const chunk_slice: zml.Slice = .init(
            .init(.{ .s = dsf, .d = enc_dim }, .bf16),
            conv_host[byte_offset .. byte_offset + chunk_bytes],
        );
        var chunk_buffer: zml.Buffer = try .fromSlice(io, platform, chunk_slice);
        defer chunk_buffer.deinit();

        var enc_token_index: zml.Buffer = try .scalar(io, platform, t_u32 * dsf, .u32);
        defer enc_token_index.deinit();

        enc_step_args.set(.{ encoder_buffers, chunk_buffer, enc_token_index, &enc_kv_buffers });
        compiled_encoder_step.call(enc_step_args, &enc_step_results);
        enc_step_results.fill(.{ &enc_step_output, &enc_kv_buffers });

        // --- Adapter step: dsf encoded frames → 1 audio embedding ---
        adp_step_args.set(.{ adapter_buffers, enc_step_output });
        compiled_adapter_step.call(adp_step_args, &adp_step_results);
        adp_step_results.fill(.{&adapter_output});

        // --- Decoder step: 1 token + 1 audio embedding → 1 output token ---
        // During prompt: use prompt token. After prompt: use previous output.
        if (t < tokens.len) {
            token_slice.items(u32)[0] = tokens[t];
        }

        var text_token_buffer: zml.Buffer = try .fromSlice(io, platform, token_slice);
        defer text_token_buffer.deinit();

        var dec_token_index: zml.Buffer = try .scalar(io, platform, t_u32, .u32);
        defer dec_token_index.deinit();

        dec_args.set(.{ decoder_buffers, text_token_buffer, adapter_output, dec_token_index, &dec_kv_buffers, t_cond_buffer, rng_buffers });
        compiled_decoder.call(dec_args, &dec_results);
        dec_results.fill(.{ &token_output, &dec_kv_buffers, &rng_buffers });

        // Download output token
        try token_output.toSlice(io, token_slice);
        const generated_token = token_slice.items(u32)[0];

        // Print tokens after prompt phase
        if (t >= tokens.len) {
            num_generated += 1;

            if (generated_token != streaming_pad_token) {
                if (try tokenizer_decoder.next(generated_token)) |chunk| {
                    std.debug.print("{s}", .{chunk});
                }
            }

            if (generated_token == eos_token) {
                log.info("EOS reached at step={}", .{t});
                break;
            }
        }
    }

    std.debug.print("\n", .{});
    log.info("Done. Generated {} tokens.", .{num_generated});
}
