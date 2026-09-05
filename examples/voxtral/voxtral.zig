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

const terminalwave = @import("terminalwave/terminalwave.zig");
const tesc = terminalwave.esc;
pub const Tokenizer = @import("tokenizer.zig").Tokenizer;

pub const CompiledExes = struct {
    mel_step: *zml.Exe,
    mel_prefill: *zml.Exe,
    conv_stem_prefill: *zml.Exe,
    conv_stem_step: *zml.Exe,
    encoder_prefill: *zml.Exe,
    encoder_step: *zml.Exe,
    adapter: *zml.Exe,
    adapter_step: *zml.Exe,
    decoder_prefill: *zml.Exe,
    decoder_decode: *zml.Exe,
};

pub const ModelBuffers = struct {
    mel_spectrum: *zml.Bufferized(LogMelSpectrogram),
    encoder: *zml.Bufferized(Encoder),
    adapter: *zml.Bufferized(Adapter),
    decoder: *zml.Bufferized(Decoder),
    enc_kv: *zml.Bufferized(KvCache),
    dec_kv: *zml.Bufferized(KvCache),
    conv_state: *zml.Bufferized(Encoder.ConvState),
    t_cond: zml.Buffer,
    rng: *zml.Bufferized(Tensor.Rng),
    enc_attention_metadata: *zml.Bufferized(zml.attention.Metadata),
    dec_attention_metadata: *zml.Bufferized(zml.attention.Metadata),
};

pub const PipelineContext = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    config: Config,
    sp: StreamParams,
    exes: CompiledExes,
    buffers: ModelBuffers,
};

fn compileStep(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: anytype, comptime method: anytype, inputs: anytype, progress: *std.Progress.Node, comptime label: []const u8) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling " ++ label ++ "...", 1);
    defer node.end();
    log.info("Compiling " ++ label ++ "...", .{});
    return try platform.compile(allocator, io, model, method, inputs, .{ .program_name = label });
}

/// Compiles LogMelSpectrogram.melStep for streaming (one chunk at a time).
/// Input:  [samples=chunk_audio] f32
/// Output: [channels=128, time=mel_per_step] f32
pub fn compileMelStep(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: LogMelSpectrogram, sp: StreamParams, progress: *std.Progress.Node) !zml.Exe {
    return compileStep(allocator, io, platform, model, .melStep, .{Tensor.init(.{sp.chunk_audio}, .f32).withTags(.{.samples})}, progress, "mel step");
}

/// Compiles LogMelSpectrogram.melStep for the full prefill audio.
/// Input:  [samples=(prompt_len-1)*mel_per_step*hop_length + chunk_audio] f32
/// Output: [channels=128, time=prompt_len*mel_per_step] f32
pub fn compileMelPrefill(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: LogMelSpectrogram, sp: StreamParams, progress: *std.Progress.Node) !zml.Exe {
    const full_audio: u32 = (sp.prompt_len - 1) * sp.mel_per_step * sp.hop_length + sp.chunk_audio;
    return compileStep(allocator, io, platform, model, .melStep, .{
        Tensor.init(.{full_audio}, .f32).withTags(.{.samples}),
    }, progress, "mel prefill");
}

/// Compiles Encoder.convStemPrefill for the full prefill mel spectrogram.
/// Input:  [channels=128, time=num_mel_frames] f32  (num_mel_frames = prompt_len*mel_per_step)
/// Output: ([s=num_mel_frames/2=prompt_len*dsf, d=enc_dim] bf16, ConvState)
pub fn compileConvStemPrefill(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Encoder, num_mel_frames: u32, progress: *std.Progress.Node) !zml.Exe {
    return compileStep(allocator, io, platform, model, .convStemPrefill, .{
        Tensor.init(.{ .channels = 128, .time = num_mel_frames }, .f32),
    }, progress, "conv stem prefill");
}

/// Compiles Encoder.convStemStep for one streaming step.
/// Input:  [channels=128, time=mel_per_step] f32, ConvState
/// Output: ([s=dsf, d=enc_dim] bf16, ConvState)
pub fn compileConvStemStep(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Encoder, sp: StreamParams, progress: *std.Progress.Node) !zml.Exe {
    const dtype = model.conv0_weight.dtype();
    const enc_dim = model.config.encoder().dim;
    return compileStep(allocator, io, platform, model, .convStemStep, .{
        Tensor.init(.{ .channels = 128, .time = sp.mel_per_step }, .f32),
        Encoder.ConvState{
            .conv1 = Tensor.init(.{ .batch = 1, .channels = 128, .time = 2 }, dtype),
            .conv2 = Tensor.init(.{ .batch = 1, .channels = enc_dim, .time = 2 }, dtype),
        },
    }, progress, "conv stem step");
}

/// Compiles Encoder.transformer for prefill: processes prompt_len*dsf frames, populating the KV cache.
/// Input:  x [s=prompt_len*dsf, d=enc_dim] bf16, token_index scalar u32
/// Output: ([s=prompt_len*dsf, d=enc_dim] bf16, KvCache)
pub fn compileEncoderPrefill(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Encoder, prompt_len: u32, enc_kv_cache: KvCache, attention_metadata: zml.attention.Metadata, attention_parameters: zml.attention.Parameters, progress: *std.Progress.Node) !zml.Exe {
    const enc_cfg = model.config.encoder();
    const dsf = model.config.downsample_factor();
    return compileStep(allocator, io, platform, model, .transformerStep, .{
        Tensor.init(.{ .s = prompt_len * dsf, .d = enc_cfg.dim }, .bf16),
        Tensor.init(.{}, .u32),
        enc_kv_cache,
        attention_metadata,
        attention_parameters,
    }, progress, "encoder prefill");
}

/// Compiles Encoder.transformerStep for streaming: processes dsf frames using the KV cache.
/// Input:  x [s=dsf, d=enc_dim] bf16, token_index scalar u32
/// Output: ([s=dsf, d=enc_dim] bf16, KvCache, token_index scalar u32)
pub fn compileEncoderStep(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Encoder, enc_kv_cache: KvCache, attention_metadata: zml.attention.Metadata, attention_parameters: zml.attention.Parameters, progress: *std.Progress.Node) !zml.Exe {
    const enc_cfg = model.config.encoder();
    const dsf = model.config.downsample_factor();
    return compileStep(allocator, io, platform, model, .transformerStep, .{
        Tensor.init(.{ .s = dsf, .d = enc_cfg.dim }, .bf16),
        Tensor.init(.{}, .u32),
        enc_kv_cache,
        attention_metadata,
        attention_parameters,
    }, progress, "encoder step");
}

/// Compiles Adapter.forward for prefill: maps prompt_len*dsf encoder frames to prompt_len decoder embeddings.
/// Input:  [s=prompt_len*dsf, d=enc_dim] bf16
/// Output: [s=prompt_len, d=dim] bf16
pub fn compileAdapter(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Adapter, prompt_len: u32, config: Config, progress: *std.Progress.Node) !zml.Exe {
    const dsf = config.downsample_factor();
    return compileStep(allocator, io, platform, model, .forward, .{
        Tensor.init(.{ .s = prompt_len * dsf, .d = config.encoder().dim }, .bf16),
    }, progress, "adapter");
}

/// Compiles Adapter.forward for streaming: maps dsf encoder frames to 1 decoder embedding.
/// Input:  [s=dsf, d=enc_dim] bf16
/// Output: [s=1, d=dim] bf16
pub fn compileAdapterStep(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Adapter, config: Config, progress: *std.Progress.Node) !zml.Exe {
    const dsf = config.downsample_factor();
    return compileStep(allocator, io, platform, model, .forward, .{
        Tensor.init(.{ .s = dsf, .d = config.encoder().dim }, .bf16),
    }, progress, "adapter step");
}

/// Compiles Decoder.forward (prefill) and Decoder.forwardStep (decode step) concurrently.
/// Prefill: text_tokens [s=prompt_len] u32, audio_embed [s=prompt_len, d=dim] → (tokens [s=prompt_len] u32, KvCache, Rng)
/// Decode:  text_tokens [s=1] u32,          audio_embed [s=1, d=dim]          → (token [s=1] u32, KvCache, Rng, token_index scalar u32)
pub fn compileDecoder(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Decoder, prompt_len: u32, dec_kv_cache: KvCache, attention_metadata: zml.attention.Metadata, attention_parameters: zml.attention.Parameters, progress: *std.Progress.Node) !struct { zml.Exe, zml.Exe } {
    const dim = model.config.dim;

    const compilePrefill = struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: *const zml.Platform, model_: Decoder, s: u32, kv_cache_: KvCache, dim_: u32, attention_metadata_: zml.attention.Metadata, attention_parameters_: zml.attention.Parameters, progress_: *std.Progress.Node) !zml.Exe {
            return compileStep(allocator_, io_, platform_, model_, .forwardStep, .{
                Tensor.init(.{ .s = s }, .u32),
                Tensor.init(.{ .s = s, .d = dim_ }, .bf16),
                Tensor.init(.{}, .u32),
                kv_cache_,
                Tensor.init(.{ .d = dim_ }, .f32),
                Tensor.Rng.init(),
                attention_metadata_,
                attention_parameters_,
            }, progress_, "decoder prefill");
        }
    }.call;

    const compileDecode = struct {
        fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: *const zml.Platform, model_: Decoder, kv_cache_: KvCache, dim_: u32, attention_metadata_: zml.attention.Metadata, attention_parameters_: zml.attention.Parameters, progress_: *std.Progress.Node) !zml.Exe {
            return compileStep(allocator_, io_, platform_, model_, .forwardStep, .{
                Tensor.init(.{ .s = 1 }, .u32),
                Tensor.init(.{ .s = 1, .d = dim_ }, .bf16),
                Tensor.init(.{}, .u32),
                kv_cache_,
                Tensor.init(.{ .d = dim_ }, .f32),
                Tensor.Rng.init(),
                attention_metadata_,
                attention_parameters_,
            }, progress_, "decoder decode");
        }
    }.call;

    const prefill = try compilePrefill(allocator, io, platform, model, prompt_len, dec_kv_cache, dim, attention_metadata, attention_parameters, progress);
    errdefer prefill.deinit();
    const decode = try compileDecode(allocator, io, platform, model, dec_kv_cache, dim, attention_metadata, attention_parameters, progress);
    return .{ prefill, decode };
}

pub fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, progress: *std.Progress.Node) !Tokenizer {
    progress.increaseEstimatedTotalItems(1);

    var node = progress.start("Loading tokenizer...", 1);
    defer node.end();

    const bytes = b: {
        const file = try dir.openFile(io, "tekken.json", .{});
        defer file.close(io);

        var reader = file.reader(io, &.{});
        break :b try reader.interface.readAlloc(allocator, try file.length(io));
    };
    defer allocator.free(bytes);

    return try .fromBytes(allocator, bytes);
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

/// Run the prefill phase: mel -> conv stem -> encoder -> adapter -> decoder prefill.
/// Populates KV caches and produces the first generated token into generated_token_slice.
fn runPrefill(ctx: *PipelineContext, padded_audio: []const f32, prompt_tokens: []const u32, generated_token_slice: zml.Slice) !void {
    const allocator = ctx.allocator;
    const io = ctx.io;
    const platform = ctx.platform;
    const sp = ctx.sp;

    const prompt_len = sp.prompt_len;

    // Mel prefill: single kernel, produces [channels=128, time=prompt_len*mel_per_step] on device
    const full_audio_len = sp.prefill_audio_len;
    log.info("Running mel prefill ({} audio samples -> {} mel frames)...", .{ full_audio_len, sp.prompt_len * sp.mel_per_step });
    var mel_prefill_output: zml.Buffer = undefined;
    {
        const audio_slice: zml.Slice = .init(
            .init(.{full_audio_len}, .f32),
            std.mem.sliceAsBytes(padded_audio[0..full_audio_len]),
        );
        var audio_buffer: zml.Buffer = try .fromSlice(io, platform, audio_slice, .replicated);
        defer audio_buffer.deinit();

        var args = try ctx.exes.mel_prefill.args(allocator);
        defer args.deinit(allocator);
        var results = try ctx.exes.mel_prefill.results(allocator);
        defer results.deinit(allocator);

        args.set(.{ ctx.buffers.mel_spectrum, audio_buffer });
        ctx.exes.mel_prefill.call(args, &results);
        results.fill(.{&mel_prefill_output});
    }
    defer mel_prefill_output.deinit();
    log.info("Mel prefill done ({} frames).", .{sp.prompt_len * sp.mel_per_step});

    // Conv stem prefill: [channels=128, time=prompt_len*mel_per_step=prompt_len*(dsf*2)] -> [s=prompt_len*dsf, d=enc_dim]
    // Also extracts conv states (last 2 frames of each intermediate) for streaming
    log.info("Running conv stem prefill ({} mel frames)...", .{prompt_len * sp.mel_per_step});
    var conv_prefill_output: zml.Buffer = undefined;
    {
        var args = try ctx.exes.conv_stem_prefill.args(allocator);
        defer args.deinit(allocator);
        var results = try ctx.exes.conv_stem_prefill.results(allocator);
        defer results.deinit(allocator);

        args.set(.{ ctx.buffers.encoder, mel_prefill_output });
        ctx.exes.conv_stem_prefill.call(args, &results);
        Encoder.ConvState.deinitBuffer(ctx.buffers.conv_state);
        results.fill(.{ &conv_prefill_output, ctx.buffers.conv_state });
    }
    defer conv_prefill_output.deinit();
    log.info("Conv stem prefill done.", .{});

    // Encoder prefill: first prompt_len*dsf frames through transformer with KV cache
    log.info("Running encoder prefill ({} frames)...", .{prompt_len * sp.dsf});
    var encoder_prefill_output: zml.Buffer = undefined;
    {
        var enc_token_index: zml.Buffer = try .scalar(io, platform, @as(u32, 0), .u32);
        defer enc_token_index.deinit();

        var args = try ctx.exes.encoder_prefill.args(allocator);
        defer args.deinit(allocator);
        var results = try ctx.exes.encoder_prefill.results(allocator);
        defer results.deinit(allocator);

        args.set(.{ ctx.buffers.encoder, conv_prefill_output, enc_token_index, ctx.buffers.enc_kv, ctx.buffers.enc_attention_metadata });
        ctx.exes.encoder_prefill.call(args, &results);
        KvCache.deinitBuffer(ctx.buffers.enc_kv);
        enc_token_index.deinit();
        results.fill(.{ &encoder_prefill_output, ctx.buffers.enc_kv, &enc_token_index });
    }
    defer encoder_prefill_output.deinit();
    log.info("Encoder prefill done.", .{});

    // Adapter batch: encoder prefill output -> prompt_len audio embeddings
    log.info("Running adapter...", .{});
    var adapter_prefill_output: zml.Buffer = undefined;
    {
        var args = try ctx.exes.adapter.args(allocator);
        defer args.deinit(allocator);
        var results = try ctx.exes.adapter.results(allocator);
        defer results.deinit(allocator);

        args.set(.{ ctx.buffers.adapter, encoder_prefill_output });
        ctx.exes.adapter.call(args, &results);
        results.fill(.{&adapter_prefill_output});
    }
    defer adapter_prefill_output.deinit();
    log.info("Adapter done.", .{});

    // Decoder prefill: all prompt tokens + audio embeddings -> first generated token
    log.info("Running decoder prefill...", .{});
    {
        var args = try ctx.exes.decoder_prefill.args(allocator);
        defer args.deinit(allocator);
        var results = try ctx.exes.decoder_prefill.results(allocator);
        defer results.deinit(allocator);

        const token_data_slice: zml.Slice = .init(
            .init(.{prompt_len}, .u32),
            std.mem.sliceAsBytes(prompt_tokens),
        );
        var token_buffer: zml.Buffer = try .fromSlice(io, platform, token_data_slice, .replicated);
        defer token_buffer.deinit();

        var token_index_buffer: zml.Buffer = try .scalar(io, platform, @as(u32, 0), .u32);
        defer token_index_buffer.deinit();

        var prefill_output_slice: zml.Slice = try .alloc(allocator, .init(.{prompt_len}, .u32));
        defer prefill_output_slice.free(allocator);
        var prefill_output: zml.Buffer = undefined;

        args.set(.{ ctx.buffers.decoder, token_buffer, adapter_prefill_output, token_index_buffer, ctx.buffers.dec_kv, ctx.buffers.t_cond, ctx.buffers.rng.*, ctx.buffers.dec_attention_metadata });
        ctx.exes.decoder_prefill.call(args, &results);
        KvCache.deinitBuffer(ctx.buffers.dec_kv);
        Tensor.Rng.deinitBuffer(ctx.buffers.rng);
        token_index_buffer.deinit();
        results.fill(.{ &prefill_output, ctx.buffers.dec_kv, ctx.buffers.rng, &token_index_buffer });
        defer prefill_output.deinit();

        // Extract last token from prefill output (first generated token)
        try prefill_output.toSlice(io, prefill_output_slice);
        generated_token_slice.items(u32)[0] = prefill_output_slice.items(u32)[prompt_len - 1];
    }
    log.info("Decoder prefill done.\n", .{});
}

/// Run the streaming decode loop: step-by-step conv_stem -> encoder -> adapter -> decoder.
fn runGenerationLoop(ctx: *PipelineContext, tokenizer: *Tokenizer, initial_overlap: []const f32, stdin_reader: *std.Io.Reader, generated_token_slice: zml.Slice) !void {
    const allocator = ctx.allocator;
    const io = ctx.io;
    const platform = ctx.platform;
    const sp = ctx.sp;

    const prompt_len = sp.prompt_len;
    const new_audio_per_step = sp.new_audio_per_step;
    const audio_overlap = sp.audio_overlap;

    // Sliding audio buffer: [overlap from previous step | new samples]
    const audio_buf = try allocator.alloc(f32, sp.chunk_audio);
    defer allocator.free(audio_buf);
    @memcpy(audio_buf[0..audio_overlap], initial_overlap);
    @memset(audio_buf[audio_overlap..], 0);

    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    const eos_token = tokenizer.tokenToId("</s>") orelse @panic("tokenizer missing </s> token");
    const streaming_pad_token = tokenizer.tokenToId("[STREAMING_PAD]") orelse @panic("tokenizer missing [STREAMING_PAD] token");
    const streaming_word_token = tokenizer.tokenToId("[STREAMING_WORD]");

    const decode_start: std.Io.Timestamp = .now(io, .awake);

    var mel_step_args = try ctx.exes.mel_step.args(allocator);
    defer mel_step_args.deinit(allocator);
    var mel_step_results = try ctx.exes.mel_step.results(allocator);
    defer mel_step_results.deinit(allocator);

    var conv_step_args = try ctx.exes.conv_stem_step.args(allocator);
    defer conv_step_args.deinit(allocator);
    var conv_step_results = try ctx.exes.conv_stem_step.results(allocator);
    defer conv_step_results.deinit(allocator);

    var enc_step_args = try ctx.exes.encoder_step.args(allocator);
    defer enc_step_args.deinit(allocator);
    var enc_step_results = try ctx.exes.encoder_step.results(allocator);
    defer enc_step_results.deinit(allocator);

    var adp_step_args = try ctx.exes.adapter_step.args(allocator);
    defer adp_step_args.deinit(allocator);
    var adp_step_results = try ctx.exes.adapter_step.results(allocator);
    defer adp_step_results.deinit(allocator);

    var decode_args = try ctx.exes.decoder_decode.args(allocator);
    defer decode_args.deinit(allocator);
    var decode_results = try ctx.exes.decoder_decode.results(allocator);
    defer decode_results.deinit(allocator);

    var current_token_buffer: zml.Buffer = try .fromSlice(io, platform, generated_token_slice, .replicated);
    defer current_token_buffer.deinit();

    // Pre-allocate reusable stdin sample buffers
    const raw_buf = try allocator.alloc(u8, new_audio_per_step * 2);
    defer allocator.free(raw_buf);
    const sample_buf = try allocator.alloc(f32, new_audio_per_step);
    defer allocator.free(sample_buf);

    // Persistent on-device token index buffers
    var enc_token_index: zml.Buffer = try .scalar(io, platform, @as(u32, prompt_len) * sp.dsf, .u32);
    defer enc_token_index.deinit();
    var dec_token_index: zml.Buffer = try .scalar(io, platform, @as(u32, prompt_len), .u32);
    defer dec_token_index.deinit();

    // Wave visualization + transcript accumulation
    var stdout = std.Io.File.stdout().writer(io, &.{});
    const interactive = try std.Io.File.stdout().isTty(io);
    var wave: terminalwave.State = if (interactive) .init(.{
        .title = "Voxtral Realtime",
        .sensitivity = 5.0,
        .num_bars = 50,
        .half_height = 20,
    }, &stdout.interface) else undefined;
    errdefer if (interactive) wave.deinit();

    var transcript: std.ArrayListUnmanaged(u8) = .empty;
    defer transcript.deinit(allocator);

    // Set up scroll region: wave is fixed at top, transcript scrolls below
    if (interactive) {
        const title_rows: u16 = if (wave.config.show_title) 3 else 0;
        const transcript_start_row: u16 = title_rows + wave.config.half_height + 4;

        // Render initial empty wave + separator, set scroll region
        _ = wave.render(0);
        wave.renderSeparator();
        wave.fmt(tesc.set_scroll_region_fmt, .{transcript_start_row});
        wave.fmt(tesc.move_cursor_fmt, .{ transcript_start_row, wave.config.padding_left + 1 });
        wave.put(tesc.clear_to_end);
        wave.flush();
    }

    var last_transcript_len: usize = 0;
    var num_generated: usize = 0;
    var flush_steps = sp.n_delay_tokens + 2;
    while (true) {
        const generated_token = generated_token_slice.items(u32)[0];
        num_generated += 1;

        if (generated_token != streaming_pad_token and (streaming_word_token == null or generated_token != streaming_word_token.?)) {
            if (try tokenizer_decoder.next(generated_token)) |chunk| {
                try transcript.appendSlice(allocator, chunk);
            }
        }

        if (generated_token == eos_token) break;

        // Read new audio from stdin
        const samples_read = try readStdinSamplesInto(stdin_reader, raw_buf, sample_buf);
        if (samples_read == 0) {
            if (flush_steps == 0) break;
            flush_steps -= 1;
        }

        if (interactive) {
            // Save cursor position (in transcript scroll region)
            wave.writer.writeAll(tesc.save_cursor) catch {};

            // Render wave with current audio level (fixed area above scroll region)
            const rms = terminalwave.computeRms(sample_buf);
            _ = wave.render(rms);
            wave.renderSeparator();

            // Restore cursor and print only new transcript text
            wave.put(tesc.restore_cursor);
            if (transcript.items.len > last_transcript_len) {
                wave.put(tesc.bright_white);
                wave.put(transcript.items[last_transcript_len..]);
                wave.put(tesc.reset);
                last_transcript_len = transcript.items.len;
            }
            wave.flush();
        }

        // Shift history left, append new samples
        @memmove(audio_buf[0..audio_overlap], audio_buf[new_audio_per_step..]);
        @memcpy(audio_buf[audio_overlap..], sample_buf);

        const audio_slice: zml.Slice = .init(
            .init(.{sp.chunk_audio}, .f32),
            std.mem.sliceAsBytes(audio_buf),
        );
        var audio_buffer: zml.Buffer = try .fromSliceOpts(io, platform, audio_slice, .replicated, .{ .wait = false });
        defer audio_buffer.deinit();

        var mel_step_output: zml.Buffer = undefined;
        mel_step_args.set(.{ ctx.buffers.mel_spectrum, audio_buffer });
        ctx.exes.mel_step.call(mel_step_args, &mel_step_results);
        mel_step_results.fill(.{&mel_step_output});
        defer mel_step_output.deinit();

        // Conv stem step: mel + conv states → output + updated conv states
        var conv_step_output: zml.Buffer = undefined;
        conv_step_args.set(.{ ctx.buffers.encoder, mel_step_output, ctx.buffers.conv_state });
        ctx.exes.conv_stem_step.call(conv_step_args, &conv_step_results);
        Encoder.ConvState.deinitBuffer(ctx.buffers.conv_state);
        conv_step_results.fill(.{ &conv_step_output, ctx.buffers.conv_state });
        defer conv_step_output.deinit();

        // Encoder step: conv stem output -> encoded chunk
        var enc_step_output: zml.Buffer = undefined;
        enc_step_args.set(.{ ctx.buffers.encoder, conv_step_output, enc_token_index, ctx.buffers.enc_kv, ctx.buffers.enc_attention_metadata });
        ctx.exes.encoder_step.call(enc_step_args, &enc_step_results);
        KvCache.deinitBuffer(ctx.buffers.enc_kv);
        enc_token_index.deinit();
        enc_step_results.fill(.{ &enc_step_output, ctx.buffers.enc_kv, &enc_token_index });
        defer enc_step_output.deinit();

        // Adapter step: encoded chunk -> 1 audio embedding
        var adapter_step_output: zml.Buffer = undefined;
        adp_step_args.set(.{ ctx.buffers.adapter, enc_step_output });
        ctx.exes.adapter_step.call(adp_step_args, &adp_step_results);
        adp_step_results.fill(.{&adapter_step_output});
        defer adapter_step_output.deinit();

        // Decoder decode: 1 token + 1 audio embedding -> next token
        decode_args.set(.{ ctx.buffers.decoder, current_token_buffer, adapter_step_output, dec_token_index, ctx.buffers.dec_kv, ctx.buffers.t_cond, ctx.buffers.rng.*, ctx.buffers.dec_attention_metadata });
        ctx.exes.decoder_decode.call(decode_args, &decode_results);
        current_token_buffer.deinit();
        KvCache.deinitBuffer(ctx.buffers.dec_kv);
        Tensor.Rng.deinitBuffer(ctx.buffers.rng);
        dec_token_index.deinit();
        decode_results.fill(.{ &current_token_buffer, ctx.buffers.dec_kv, ctx.buffers.rng, &dec_token_index });

        try current_token_buffer.toSlice(io, generated_token_slice);
    }

    // Reset scroll region and exit wave visualization (returns to main screen)
    if (interactive) wave.deinit();

    const decode_duration = decode_start.untilNow(io, .awake);
    try stdout.interface.print("{s}\n", .{transcript.items});
    log.info("Decode done. Generated {} tokens in {f}: {:.3}tok/s", .{
        num_generated,
        decode_duration,
        @as(f64, @floatFromInt(num_generated)) * std.time.ns_per_s / @as(f64, @floatFromInt(@max(1, decode_duration.nanoseconds))),
    });
}

pub fn runPipeline(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    config: Config,
    tokenizer: *Tokenizer,
    prompt_tokens: []const u32,
    sp: StreamParams,
    exes: CompiledExes,
    mel_spectrum_buffers: *zml.Bufferized(LogMelSpectrogram),
    encoder_buffers: *zml.Bufferized(Encoder),
    adapter_buffers: *zml.Bufferized(Adapter),
    decoder_buffers: *zml.Bufferized(Decoder),
    enc_kv_cache: KvCache,
    dec_kv_cache: KvCache,
    conv_state: Encoder.ConvState,
    enc_attention_metadata_buffers: *zml.Bufferized(zml.attention.Metadata),
    dec_attention_metadata_buffers: *zml.Bufferized(zml.attention.Metadata),
    input: std.Io.File,
) !void {
    log.info("Running inference pipeline...", .{});

    var stdin_buff: [2560 * 4]u8 = undefined;
    var stdin_reader = input.reader(io, &stdin_buff);
    const stdin_reader_interface = &stdin_reader.interface;

    // 0. Read initial audio from stdin and build padded buffer for prefill
    // We read the full mel buffer's worth of audio (minus left_pad) so the STFT
    // overlap region at the tail has real audio instead of silence.
    const full_audio_len = sp.prefill_audio_len;
    const n_prefill_stdin: usize = full_audio_len - sp.left_pad;
    const prefill_raw_buf = try allocator.alloc(u8, n_prefill_stdin * 2);
    defer allocator.free(prefill_raw_buf);
    const stdin_audio = try allocator.alloc(f32, n_prefill_stdin);
    defer allocator.free(stdin_audio);
    if (try readStdinSamplesInto(stdin_reader_interface, prefill_raw_buf, stdin_audio) == 0) return;

    // Pad: [left_pad zeros | stdin audio]
    const prefill_audio = try allocator.alloc(f32, full_audio_len);
    defer allocator.free(prefill_audio);
    @memset(prefill_audio[0..@as(usize, sp.left_pad)], 0);
    @memcpy(prefill_audio[@as(usize, sp.left_pad)..], stdin_audio);

    // full_audio_len: 50160 | n_prefill_stdin: 9200

    // 1. Init KV caches + t_cond + RNG
    var enc_kv_buffers = try enc_kv_cache.initBuffer(io, platform);
    defer KvCache.deinitBuffer(&enc_kv_buffers);

    var dec_kv_buffers = try dec_kv_cache.initBuffer(io, platform);
    defer KvCache.deinitBuffer(&dec_kv_buffers);

    var conv_state_buffers = try conv_state.initBuffer(io, platform);
    defer Encoder.ConvState.deinitBuffer(&conv_state_buffers);

    const t_cond_data = computeTimeEmbedding(allocator, @as(f32, @floatFromInt(sp.n_delay_tokens)), config.dim);
    defer allocator.free(t_cond_data);
    const t_cond_slice: zml.Slice = .init(
        .init(.{config.dim}, .f32),
        std.mem.sliceAsBytes(t_cond_data),
    );
    var t_cond_buffer: zml.Buffer = try .fromSlice(io, platform, t_cond_slice, .replicated);
    defer t_cond_buffer.deinit();

    var rng_buffers = try Tensor.Rng.initBuffer(io, platform, .replicated, 0);
    defer Tensor.Rng.deinitBuffer(&rng_buffers);

    // 2. Assemble pipeline context
    var ctx = PipelineContext{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .config = config,
        .sp = sp,
        .exes = exes,
        .buffers = .{
            .mel_spectrum = mel_spectrum_buffers,
            .encoder = encoder_buffers,
            .adapter = adapter_buffers,
            .decoder = decoder_buffers,
            .enc_kv = &enc_kv_buffers,
            .dec_kv = &dec_kv_buffers,
            .conv_state = &conv_state_buffers,
            .t_cond = t_cond_buffer,
            .rng = &rng_buffers,
            .enc_attention_metadata = enc_attention_metadata_buffers,
            .dec_attention_metadata = dec_attention_metadata_buffers,
        },
    };

    // 3. Prefill (mel -> conv stem -> encoder -> adapter -> decoder)
    var generated_token_slice: zml.Slice = try .alloc(allocator, .init(.{@as(u32, 1)}, .u32));
    defer generated_token_slice.free(allocator);

    try runPrefill(&ctx, prefill_audio, prompt_tokens, generated_token_slice);

    // 4. Generation loop — seed the overlap with the tail of the prefill audio
    const audio_overlap = sp.audio_overlap;
    try runGenerationLoop(&ctx, tokenizer, prefill_audio[full_audio_len - audio_overlap ..], stdin_reader_interface, generated_token_slice);
}

fn readStdinSamplesInto(reader: *std.Io.Reader, raw_buf: []u8, samples: []f32) !usize {
    const bytes_read = try reader.readSliceShort(raw_buf);
    if (bytes_read % 2 != 0) return error.IncompleteAudioSample;
    @memset(raw_buf[bytes_read..], 0);
    for (0..samples.len) |i| {
        samples[i] = @as(f32, @floatFromInt(std.mem.readInt(i16, raw_buf[i * 2 ..][0..2], .little))) / 32768.0;
    }
    return bytes_read / 2;
}
