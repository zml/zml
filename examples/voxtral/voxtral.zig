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

const CliArgs = struct {
    input: []const u8,
    model: []const u8,
};

const prompt = "";
const tokens_array = [_]u32{1, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32};
const tokens: []const u32 = &tokens_array;
const audio_len = 293699;

const n_delay_tokens = 6;

pub fn main() !void {
    log.info("Start of Voxtral", .{});

    var dbg = std.heap.DebugAllocator(.{}).init;
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

    const arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();

    const file = try std.Io.Dir.openFile(.cwd(), io, args.input, .{});
    defer file.close(io);

    var wav_buffer: [4096]u8 = undefined;
    var reader = file.reader(io, &wav_buffer);

    const wav_file = try loadWav(allocator, &reader.interface);
    defer allocator.free(wav_file);

    
    const model_dir = try zml.safetensors.resolveModelRepo(io, args.model);

    var model_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.model);
    defer model_registry.deinit();
    
    var model_store: zml.io.TensorStore = .fromRegistry(allocator, &model_registry);
    defer model_store.deinit();

    
    var parsed_config = try cfg.parseConfig(allocator, io, model_dir);
    defer parsed_config.deinit();
    const config = parsed_config.value;

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
    const model_params: VoxtralParameters = .{
	.prefill_embeds = .init(.{.s = adapter_seq_len, .d = config.dim}, .bf16),
	.decode_embeds = .init(.{.s = 1, .d = config.dim}, .bf16),
	.token_index = .init(.{}, .u32),
	.kv_cache = .init(.init(.{
	    .layer = model.layers.len,
	    .k = config.sliding_window,
	    .h = config.n_kv_heads,
	    .hd = config.head_dim,
	}, dtype)),
	.t_cond = Tensor.init(.{.d = config.dim}, dtype),
    };
    
    var tokenizer_future = try io.concurrent(loadTokenizer, .{allocator, io, model_dir});
    var compiled_mel_spectrum_future = try io.concurrent(compileMelSpectrum, .{allocator, io, platform, melspectro_model});
    var compiled_encoder_future = try io.concurrent(compileEncoder, .{allocator, io, platform, encoder_model});
    var compiled_adapter_future = try io.concurrent(compileAdapter, .{allocator, io, platform, adapter, encoder_seq_len, config});
    var compiled_embedder_future = try io.concurrent(compileEmbedder, .{allocator, io, platform, embedder, adapter_seq_len, config});
    var compiled_decode_embedder_future = try io.concurrent(compileEmbedder, .{allocator, io, platform, embedder, @as(u32, 1), config});
    var compiled_decoder_future = try io.concurrent(compileDecoder, .{allocator, io, platform, model, model_params});
    
    var mel_spectrum_buffers_future = try io.concurrent(LogMelSpectrogram.load, .{&melspectro_model, io, platform});
    var encoder_buffers_future = try io.concurrent(Encoder.load, .{&encoder_model,allocator,io, platform, &model_store, &progress});
    var adapter_buffers_future = try io.concurrent(Adapter.load, .{&adapter, allocator, io, platform, &model_store, &progress});
    var embedder_buffers_future = try io.concurrent(Embedder.load, .{&embedder, allocator, io, platform, &model_store, &progress});
    var decoder_buffers_future = try io.concurrent(Decoder.load, .{&model, allocator, io, platform, &model_store, &progress});

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

    var compiled_model = try compiled_decoder_future.await(io);
    defer compiled_model.prefill.deinit();
    defer compiled_model.decode.deinit();
   
    // -- Buffers
    
    var mel_spectrum_buffers = try mel_spectrum_buffers_future.await(io);
    defer LogMelSpectrogram.unload(&mel_spectrum_buffers);

    var encoder_buffers = try encoder_buffers_future.await(io);
    defer Encoder.unloadBuffers(&encoder_buffers, allocator);

    var adapter_buffers = try adapter_buffers_future.await(io);
    defer Adapter.unloadBuffers(&adapter_buffers);

    var embedder_buffers = try embedder_buffers_future.await(io);
    defer Embedder.unloadBuffers(&embedder_buffers);
    
    var decoder_buffers = try decoder_buffers_future.await(io);
    defer Decoder.unloadBuffers(&decoder_buffers, allocator);
    
    progress.end();

    // --- Execution pipeline ---
    log.info("Running inference pipeline...", .{});

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
            std.mem.sliceAsBytes(wav_file),
        );
        var audio_buffer: zml.Buffer = try .fromSlice(io, platform, audio_slice);
        defer audio_buffer.deinit();

        mel_output = try .uninitialized(io, platform, .init(.{ .channels = 128, .time = num_frames }, .f32), .{});

        mel_args.set(.{ &mel_spectrum_buffers, audio_buffer });
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

        enc_args.set(.{ &encoder_buffers, mel_output });
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

        adp_args.set(.{ &adapter_buffers, encoder_output });
        compiled_adapter.call(adp_args, &adp_results);
        adp_results.fill(.{&adapter_output});
    }
    defer adapter_output.deinit();
    log.info("Adapter done.", .{});

    // Download adapter_out to CPU for per-position slicing in the decode loop
    const adapter_out_slice: zml.Slice = try .alloc(allocator, .init(.{ .s = adapter_seq_len, .d = config.dim }, .bf16));
    defer adapter_out_slice.free(allocator);
    try adapter_output.toSlice(io, adapter_out_slice);
    const adapter_out_cpu = adapter_out_slice.constItems(u16); // bf16 stored as u16

    // 4. Embedder: combine audio + token embeddings for the prefix (39 positions)
    log.info("Running prefix embedder...", .{});
    var combined_embeds: zml.Buffer = undefined;
    {
        var emb_args = try compiled_embedder.args(allocator);
        defer emb_args.deinit(allocator);
        var emb_results = try compiled_embedder.results(allocator);
        defer emb_results.deinit(allocator);

        const padded_tokens = try allocator.alloc(u32, adapter_seq_len);
        defer allocator.free(padded_tokens);
        @memcpy(padded_tokens[0..tokens.len], tokens);
        @memset(padded_tokens[tokens.len..], 32); // STREAMING_PAD

        const tokens_slice: zml.Slice = .init(
            .init(.{ .s = adapter_seq_len }, .u32),
            std.mem.sliceAsBytes(padded_tokens),
        );
        var tokens_buffer: zml.Buffer = try .fromSlice(io, platform, tokens_slice);
        defer tokens_buffer.deinit();

        combined_embeds = try .uninitialized(io, platform, .init(.{ .s = adapter_seq_len, .d = config.dim }, .bf16), .{});

        emb_args.set(.{ &embedder_buffers, adapter_output, tokens_buffer });
        compiled_embedder.call(emb_args, &emb_results);
        emb_results.fill(.{&combined_embeds});
    }
    defer combined_embeds.deinit();
    log.info("Prefix embedder done.", .{});

    // 5. Compute time embedding on CPU and upload
    const t_cond_data = computeTimeEmbedding(allocator, @floatFromInt(n_delay_tokens), config.dim);
    defer allocator.free(t_cond_data);
    const t_cond_bf16 = try allocator.alloc(u16, config.dim);
    defer allocator.free(t_cond_bf16);
    for (0..config.dim) |i| {
        t_cond_bf16[i] = @bitCast(zml.floats.BFloat16.fromF32(t_cond_data[i]));
    }
    const t_cond_slice: zml.Slice = .init(
        .init(.{ .d = config.dim }, .bf16),
        std.mem.sliceAsBytes(t_cond_bf16),
    );
    var t_cond_buffer: zml.Buffer = try .fromSlice(io, platform, t_cond_slice);
    defer t_cond_buffer.deinit();
    log.info("Time embedding computed (t={d}).", .{n_delay_tokens});

    // 6. Init KV cache
    var kv_cache_buffers = try model_params.kv_cache.initBuffer(io, platform);
    defer dec.KvCache.deinitBuffer(&kv_cache_buffers);

    // 7. Decoder prefill: process all adapter_seq_len positions at once
    //    (fills KV cache; we take the token at position tokens.len-1 as the first generated token)
    log.info("Running decoder prefill ({d} positions)...", .{adapter_seq_len});
    var prefill_tokens_buffer: zml.Buffer = undefined;
    {
        var dec_args = try compiled_model.prefill.args(allocator);
        defer dec_args.deinit(allocator);
        var dec_results = try compiled_model.prefill.results(allocator);
        defer dec_results.deinit(allocator);

        var token_index_buffer = try zml.Buffer.scalar(io, platform, @as(u32, 0), .u32);
        defer token_index_buffer.deinit();

        prefill_tokens_buffer = try .uninitialized(io, platform, .init(.{ .s = adapter_seq_len }, .u32), .{});

        dec_args.set(.{ &decoder_buffers, combined_embeds, token_index_buffer, &kv_cache_buffers, t_cond_buffer });
        compiled_model.prefill.call(dec_args, &dec_results);
        dec_results.fill(.{ &prefill_tokens_buffer, &kv_cache_buffers });
    }
    defer prefill_tokens_buffer.deinit();
    log.info("Decoder prefill done.", .{});

    // Extract the first generated token from prefill (at position tokens.len - 1)
    const prefill_out_slice: zml.Slice = try .alloc(allocator, .init(.{ .s = adapter_seq_len }, .u32));
    defer prefill_out_slice.free(allocator);
    try prefill_tokens_buffer.toSlice(io, prefill_out_slice);
    var prev_token: u32 = prefill_out_slice.items(u32)[tokens.len - 1];
    log.info("First generated token (from prefill pos {d}): {d}", .{ tokens.len - 1, prev_token });

    // 8. Autoregressive decode loop
    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    var stdout = std.Io.File.stdout().writer(io, &.{});
    const d: usize = @intCast(config.dim);

    // Prepare reusable args/results for decode embedder and decoder
    var demb_args = try compiled_decode_embedder.args(allocator);
    defer demb_args.deinit(allocator);
    var demb_results = try compiled_decode_embedder.results(allocator);
    defer demb_results.deinit(allocator);

    var decode_args = try compiled_model.decode.args(allocator);
    defer decode_args.deinit(allocator);
    var decode_results = try compiled_model.decode.results(allocator);
    defer decode_results.deinit(allocator);

    // Reusable token output slice
    var gen_token_slice: zml.Slice = try .alloc(allocator, .init(.{ .s = 1 }, .u32));
    defer gen_token_slice.free(allocator);
    gen_token_slice.items(u32)[0] = prev_token;
    var gen_token_buffer: zml.Buffer = try .fromSlice(io, platform, gen_token_slice);
    defer gen_token_buffer.deinit();

    // Reusable combined embed buffer
    var combined_embed_buffer: zml.Buffer = try .uninitialized(io, platform, .init(.{ .s = 1, .d = config.dim }, .bf16), .{});
    defer combined_embed_buffer.deinit();

    const n_audio: usize = @intCast(adapter_seq_len);
    log.info("Running decoder decode ({d} steps)...", .{n_audio - tokens.len});

    for (tokens.len..n_audio) |pos| {
        // Check EOS (token_id=2)
        if (prev_token == 2) break;

        // Print decoded text for the current token
        if (try tokenizer_decoder.next(prev_token)) |chunk| {
	    try stdout.interface.print("{d}: {s}\n", .{pos, chunk});
            try stdout.interface.flush();
        }

        // Create audio embed [1, d] from adapter_out_cpu[pos]
        const pos_offset = pos * d;
        const audio_embed_slice: zml.Slice = .init(
            .init(.{ .s = 1, .d = config.dim }, .bf16),
            std.mem.sliceAsBytes(adapter_out_cpu[pos_offset .. pos_offset + d]),
        );
        var audio_embed_buf: zml.Buffer = try .fromSlice(io, platform, audio_embed_slice);
        defer audio_embed_buf.deinit();

        // Create token [1] buffer
        const token_data = [_]u32{prev_token};
        const tok_slice: zml.Slice = .init(
            .init(.{ .s = 1 }, .u32),
            std.mem.sliceAsBytes(&token_data),
        );
        var tok_buf: zml.Buffer = try .fromSlice(io, platform, tok_slice);
        defer tok_buf.deinit();

        // Run decode embedder: adapter_out[pos] + tok_embed(prev_token) → [1, d]
        demb_args.set(.{ &embedder_buffers, audio_embed_buf, tok_buf });
        compiled_decode_embedder.call(demb_args, &demb_results);
        demb_results.fill(.{&combined_embed_buffer});

        // Run decoder: combined_embed, token_index=pos, kv_cache, t_cond → next token
        const pos_data = [_]u32{@intCast(pos)};
        const pos_slice: zml.Slice = .init(.init(.{}, .u32), std.mem.sliceAsBytes(&pos_data));
        var pos_buf: zml.Buffer = try .fromSlice(io, platform, pos_slice);
        defer pos_buf.deinit();

        decode_args.set(.{ &decoder_buffers, combined_embed_buffer, pos_buf, &kv_cache_buffers, t_cond_buffer });
        compiled_model.decode.call(decode_args, &decode_results);
        decode_results.fill(.{ &gen_token_buffer, &kv_cache_buffers });

        // Download generated token
        try gen_token_buffer.toSlice(io, gen_token_slice);
        prev_token = gen_token_slice.items(u32)[0];
    }

    // Print final token
    if (try tokenizer_decoder.next(prev_token)) |chunk| {
	try stdout.interface.print("{s}", .{chunk});
    }
    
    try stdout.interface.writeAll("\n");
    try stdout.interface.flush();

    log.info("Decode complete.", .{});
}


fn loadWav(allocator: std.mem.Allocator, reader: *std.Io.Reader) ![]const f32 {
    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    var sample_list: std.ArrayList(u8) = .empty;

    const wav_fmt = try wav_utils.readPcmWav(arena, reader, &sample_list);
    const byte_per_sample = wav_fmt.bits_per_sample / 8;
    const sample_count = sample_list.items.len / (byte_per_sample * wav_fmt.num_channels);

    const samples = try allocator.alloc(f32, sample_count);
    for (0..sample_count) |i| {
	const offset = i * byte_per_sample * wav_fmt.num_channels;

	const sample = switch(byte_per_sample) {
	    1 => (@as(f32, @floatFromInt(std.mem.bytesToValue(u8, sample_list.items[offset .. offset + 1]))) - 128.0) / 128.0,
	    2 => @as(f32, @floatFromInt(std.mem.bytesToValue(i16, sample_list.items[offset .. offset + 2]))) / 32768.0,
	    3 => @as(f32, @floatFromInt(std.mem.bytesToValue(i24, sample_list.items[offset .. offset + 3]))) / 8388608.0,
	    4 => @as(f32, @floatFromInt(std.mem.bytesToValue(i32, sample_list.items[offset .. offset + 4]))) / 2147483648.0,
	    else => unreachable,
	};

	samples[i] = sample;
    }

    return samples;
}

pub fn compileMelSpectrum(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: LogMelSpectrogram) !zml.Exe {
    return try platform.compile(allocator, io, model, .forward, .{Tensor.init(.{audio_len}, .f32).withTags(.{.samples})});
}

pub fn compileEncoder(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Encoder) !zml.Exe {
    const num_frames = audio_len / model.config.audio().hop_length;
    
      return try platform.compile(allocator, io, model, .forward, .{
          Tensor.init(.{ .channels = 128, .time = num_frames }, .f32),
      });
}

pub fn compileAdapter(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Adapter, encoder_seq_len: u32, config: Config) !zml.Exe {
    return try platform.compile(allocator, io, model, .forward, .{
        Tensor.init(.{ .s = encoder_seq_len, .d = config.encoder().dim }, .bf16),
    });
}

pub fn compileEmbedder(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Embedder, seq_len: u32, config: Config) !zml.Exe {
    return try platform.compile(allocator, io, model, .forward, .{
        Tensor.init(.{ .s = seq_len, .d = config.dim }, .bf16),
        Tensor.init(.{ .s = seq_len }, .u32),
    });
}

const CompileModelResult = struct {
    prefill: zml.Exe,
    decode: zml.Exe,
};

const VoxtralParameters = struct {
    prefill_embeds: Tensor,
    decode_embeds: Tensor,
    token_index: Tensor,
    kv_cache: dec.KvCache,
    t_cond: Tensor,
};

pub fn compileDecoder(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Decoder, params: VoxtralParameters) !CompileModelResult {
    var prefill_future = try io.concurrent(struct {
	fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: *const zml.Platform, model_: Decoder, params_: VoxtralParameters) !zml.Exe {
	    return try platform_.compile(allocator_, io_, model_, .forward, .{
		params_.prefill_embeds,
		params_.token_index,
		params_.kv_cache,
		params_.t_cond,
	    });
	}
    }.call, .{allocator, io, platform, model, params});

    var decode_future = try io.concurrent(struct {
	fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: *const zml.Platform, model_: Decoder, params_: VoxtralParameters) !zml.Exe {
	    return try platform_.compile(allocator_, io_, model_, .forward, .{
		params_.decode_embeds,
		params_.token_index,
		params_.kv_cache,
		params_.t_cond,
	    });
	}
    }.call, .{allocator, io, platform, model, params});
    

    const prefill_exe = try prefill_future.await(io);
    const decode_exe = try decode_future.await(io);

    return .{ .prefill = prefill_exe, .decode = decode_exe };
}

/// Sinusoidal time embedding: encodes a scalar t into a [dim]-dimensional vector.
/// out[i] = cos(t * inv_freq[i]) for i < half_dim
/// out[i] = sin(t * inv_freq[i-half_dim]) for i >= half_dim
/// where inv_freq[i] = exp(-log(theta) * i / half_dim)
fn computeTimeEmbedding(allocator: std.mem.Allocator, t_value: f32, dim: u32) []f32 {
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

pub fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !zml.tokenizer.Tokenizer {
    const bytes = b: {
        const file = try dir.openFile(io, "tokenizer.json", .{});
        defer file.close(io);
        var reader = file.reader(io, &.{});
        break :b try reader.interface.readAlloc(allocator, try file.length(io));
    };
    defer allocator.free(bytes);

    return try .fromBytes(allocator, io, bytes);
}
