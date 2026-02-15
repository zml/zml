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

pub const VoxtralParameters = struct {
    prefill_tokens: Tensor,
    decode_tokens: Tensor,
    adapter_out: Tensor,
    token_index: Tensor,
    kv_cache: dec.KvCache,
    t_cond: Tensor,
};

pub fn compileMelSpectrum(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: LogMelSpectrogram, padded_audio_len: usize, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling mel spectrogram...", 1);
    defer node.end();

    return try platform.compile(allocator, io, model, .forward, .{Tensor.init(.{padded_audio_len}, .f32).withTags(.{.samples})});
}

pub fn compileEncoder(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Encoder, padded_audio_len: usize, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling encoder...", 1);
    defer node.end();

    const num_frames = padded_audio_len / model.config.audio().hop_length;

    return try platform.compile(allocator, io, model, .forward, .{
        Tensor.init(.{ .channels = 128, .time = num_frames }, .f32),
    });
}

pub fn compileAdapter(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Adapter, encoder_seq_len: u32, config: Config, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);

    var node = progress.start("Compiling adapter...", 1);
    defer node.end();

    return try platform.compile(allocator, io, model, .forward, .{
        Tensor.init(.{ .s = encoder_seq_len, .d = config.encoder().dim }, .bf16),
    });
}

pub fn compileDecoder(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Decoder, params: VoxtralParameters, progress: *std.Progress.Node) !struct { zml.Exe, zml.Exe } {
    const compilePrefill = struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: *const zml.Platform, model_: Decoder, tokens_: Tensor, params_: VoxtralParameters, progress_: *std.Progress.Node) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);

            var node_ = progress_.start("Compiling decoder prefill...", 1);
            defer node_.end();

            return try platform_.compile(allocator_, io_, model_, .forward, .{
                tokens_,
                params_.adapter_out,
                params_.token_index,
                params_.kv_cache,
                params_.t_cond,
            });
        }
    }.call;

    const compileDecode = struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: *const zml.Platform, model_: Decoder, tokens_: Tensor, params_: VoxtralParameters, progress_: *std.Progress.Node) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);

            var node_ = progress_.start("Compiling decoder decode...", 1);
            defer node_.end();

            return try platform_.compile(allocator_, io_, model_, .forward, .{
                tokens_,
                params_.adapter_out,
                params_.token_index,
                params_.kv_cache,
                params_.t_cond,
            });
        }
    }.call;

    var prefill_future = try io.concurrent(compilePrefill, .{ allocator, io, platform, model, params.prefill_tokens, params, progress });
    var decode_future = try io.concurrent(compileDecode, .{ allocator, io, platform, model, params.decode_tokens, params, progress });

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
    compiled_encoder: *zml.Exe,
    compiled_adapter: *zml.Exe,
    compiled_prefill: *zml.Exe,
    compiled_decoder: *zml.Exe,
    mel_spectrum_buffers: *zml.Bufferized(LogMelSpectrogram),
    encoder_buffers: *zml.Bufferized(Encoder),
    adapter_buffers: *zml.Bufferized(Adapter),
    decoder_buffers: *zml.Bufferized(Decoder),
    model_params: VoxtralParameters,
) !void {
    log.info("Running inference pipeline...", .{});

    const num_frames = audio_len / config.audio().hop_length;
    const dsf: u32 = config.downsample_factor();
    // Natural encoder output length after stride-2 causal conv (adapter pads to multiple of dsf)
    const encoder_seq_len: u32 = @intCast((num_frames + 1) / 2);
    const adapter_seq_len = (encoder_seq_len + dsf - 1) / dsf;

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

    // 2. Encoder
    log.info("Running encoder...", .{});
    var encoder_output: zml.Buffer = undefined;
    {
        var enc_args = try compiled_encoder.args(allocator);
        defer enc_args.deinit(allocator);
        var enc_results = try compiled_encoder.results(allocator);
        defer enc_results.deinit(allocator);

        encoder_output = try .uninitialized(io, platform, .init(.{ .s = encoder_seq_len, .d = config.encoder().dim }, .bf16), .{});

        enc_args.set(.{ encoder_buffers, mel_output });
        compiled_encoder.call(enc_args, &enc_results);
        enc_results.fill(.{&encoder_output});
    }
    defer encoder_output.deinit();
    log.info("Encoder done.", .{});

    // 3. Adapter
    log.info("Running adapter...", .{});
    var adapter_output: zml.Buffer = undefined;
    {
        var adp_args = try compiled_adapter.args(allocator);
        defer adp_args.deinit(allocator);
        var adp_results = try compiled_adapter.results(allocator);
        defer adp_results.deinit(allocator);

        adapter_output = try .uninitialized(io, platform, .init(.{ .s = adapter_seq_len, .d = config.dim }, .bf16), .{});

        adp_args.set(.{ adapter_buffers, encoder_output });
        compiled_adapter.call(adp_args, &adp_results);
        adp_results.fill(.{&adapter_output});
    }
    defer adapter_output.deinit();
    log.info("Adapter done.", .{});

    // 4. Init KV cache + t_cond buffer
    var kv_cache_buffers = try model_params.kv_cache.initBuffer(io, platform);
    defer dec.KvCache.deinitBuffer(&kv_cache_buffers);

    const t_cond_data = computeTimeEmbedding(allocator, @as(f32, @floatFromInt(n_delay_tokens)), config.dim);
    defer allocator.free(t_cond_data);
    const t_cond_slice: zml.Slice = .init(
        .init(.{config.dim}, .f32),
        std.mem.sliceAsBytes(t_cond_data),
    );
    var t_cond_buffer: zml.Buffer = try .fromSlice(io, platform, t_cond_slice);
    defer t_cond_buffer.deinit();

    // 5. Decoder prefill
    log.info("Running decoder prefill...", .{});
    var generated_token_slice: zml.Slice = try .alloc(allocator, .init(.{@as(u32, 1)}, .u32));
    defer generated_token_slice.free(allocator);
    {
        var prefill_args = try compiled_prefill.args(allocator);
        defer prefill_args.deinit(allocator);
        var prefill_results = try compiled_prefill.results(allocator);
        defer prefill_results.deinit(allocator);

        const token_slice: zml.Slice = .init(
            .init(.{@as(u32, @intCast(tokens.len))}, .u32),
            std.mem.sliceAsBytes(tokens),
        );
        var token_buffer: zml.Buffer = try .fromSlice(io, platform, token_slice);
        defer token_buffer.deinit();

        var token_index_buffer: zml.Buffer = try .scalar(io, platform, @as(u32, 0), .u32);
        defer token_index_buffer.deinit();

        var prefill_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{@as(u32, @intCast(tokens.len))}, .u32));
        defer prefill_tokens_slice.free(allocator);
        var prefill_output: zml.Buffer = try .fromSlice(io, platform, prefill_tokens_slice);
        defer prefill_output.deinit();

        prefill_args.set(.{ decoder_buffers, token_buffer, adapter_output, token_index_buffer, &kv_cache_buffers, t_cond_buffer });
        compiled_prefill.call(prefill_args, &prefill_results);
        prefill_results.fill(.{ &prefill_output, &kv_cache_buffers });

        // Extract last token from prefill output (first generated token)
        try prefill_output.toSlice(io, prefill_tokens_slice);
        log.info("Prefill output tokens: {any}", .{prefill_tokens_slice.items(u32)[0..tokens.len]});
        generated_token_slice.items(u32)[0] = prefill_tokens_slice.items(u32)[tokens.len - 1];
        log.info("First generated token (from prefill): {}", .{generated_token_slice.items(u32)[0]});
    }
    log.info("Prefill done.", .{});

    // 6. Autoregressive decode loop
    log.info("Running decoder autoregressive...", .{});
    const eos_token = tokenizer.tokenToId("</s>") orelse @panic("tokenizer missing </s> token");
    const streaming_pad_token = tokenizer.tokenToId("[STREAMING_PAD]") orelse @panic("tokenizer missing [STREAMING_PAD] token");
    {
        var decode_args = try compiled_decoder.args(allocator);
        defer decode_args.deinit(allocator);
        var decode_results = try compiled_decoder.results(allocator);
        defer decode_results.deinit(allocator);

        var current_token_buffer: zml.Buffer = try .fromSlice(io, platform, generated_token_slice);
        defer current_token_buffer.deinit();

        const max_pos = adapter_seq_len;
        log.info("Decode loop: tokens.len={}, max_pos={}", .{ tokens.len, max_pos });
        var num_generated: usize = 0;
        for (tokens.len..max_pos) |i| {
            const generated_token = generated_token_slice.items(u32)[0];
            num_generated += 1;

            if (generated_token != streaming_pad_token) {
                if (try tokenizer_decoder.next(generated_token)) |chunk| {
                    std.debug.print("{s}", .{chunk});
                }
            }

            if (generated_token == eos_token) {
                log.info("  EOS reached at pos={}", .{i});
                break;
            }

            var token_index_buffer: zml.Buffer = try .scalar(io, platform, @as(u32, @intCast(i)), .u32);
            defer token_index_buffer.deinit();

            decode_args.set(.{ decoder_buffers, current_token_buffer, adapter_output, token_index_buffer, &kv_cache_buffers, t_cond_buffer });
            compiled_decoder.call(decode_args, &decode_results);
            decode_results.fill(.{ &current_token_buffer, &kv_cache_buffers });

            try current_token_buffer.toSlice(io, generated_token_slice);
        }
        std.debug.print("\n", .{});
        log.info("Decode done. Generated {} tokens total.", .{num_generated});
    }
}
