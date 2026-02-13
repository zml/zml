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
const n_delay_tokens = 6;
const n_left_pad_tokens = 32;
const n_right_pad_tokens = (n_delay_tokens + 1) + 10; // 17
const raw_audio_length_per_tok = 1280; // SAMPLE_RATE / FRAME_RATE = 16000 / 12.5

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

    const wav_file = try wav_utils.loadWav(allocator, &reader.interface);
    defer allocator.free(wav_file);

    const left_pad = n_left_pad_tokens * raw_audio_length_per_tok; // 32 * 1280 = 40960
    const align_pad = (raw_audio_length_per_tok - (wav_file.len % raw_audio_length_per_tok)) % raw_audio_length_per_tok;
    const right_pad = align_pad + n_right_pad_tokens * raw_audio_length_per_tok; // align + 17 * 1280
    const audio_len = left_pad + wav_file.len + right_pad;

    const padded_audio = try allocator.alloc(f32, audio_len);
    defer allocator.free(padded_audio);

    @memset(padded_audio, 0);
    @memcpy(padded_audio[left_pad .. left_pad + wav_file.len], wav_file);

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
	.prefill_embeds = .init(.{.s = @as(u32, @intCast(tokens.len)), .d = config.dim}, .bf16),
	.decode_embeds = .init(.{.s = 1, .d = config.dim}, .bf16),
	.token_index = .init(.{}, .u32),
	.kv_cache = .init(.init(.{
	    .layer = model.layers.len,
	    .k = config.sliding_window,
	    .h = config.n_kv_heads,
	    .hd = config.head_dim,
	}, dtype)),
	.t_cond = Tensor.init(.{.d = config.dim}, .f32),
    };
    
    var tokenizer_future = try io.concurrent(loadTokenizer, .{allocator, io, model_dir});
    var compiled_mel_spectrum_future = try io.concurrent(compileMelSpectrum, .{allocator, io, platform, melspectro_model, audio_len});
    var compiled_encoder_future = try io.concurrent(compileEncoder, .{allocator, io, platform, encoder_model, audio_len});
    var compiled_adapter_future = try io.concurrent(compileAdapter, .{allocator, io, platform, adapter, encoder_seq_len, config});
    var compiled_embedder_future = try io.concurrent(compileEmbedder, .{allocator, io, platform, embedder, adapter_seq_len, @as(u32, @intCast(tokens.len)), config});
    var compiled_decode_embedder_future = try io.concurrent(compileEmbedder, .{allocator, io, platform, embedder, adapter_seq_len, @as(u32, 1), config});
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

    const compiled_prefill, const compiled_decoder = try compiled_decoder_future.await(io);
    errdefer compiled_prefill.deinit();
    errdefer compiled_decoder.deinit();
   
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

    // 4. Embedder - combine audio embeddings with token embeddings
    log.info("Running embedder...", .{});
    var embedder_output: zml.Buffer = undefined;
    {
        var emb_args = try compiled_embedder.args(allocator);
        defer emb_args.deinit(allocator);
        var emb_results = try compiled_embedder.results(allocator);
        defer emb_results.deinit(allocator);

        const token_slice: zml.Slice = .init(
            .init(.{@as(u32, @intCast(tokens.len))}, .u32),
            std.mem.sliceAsBytes(tokens),
        );
        var token_buffer: zml.Buffer = try .fromSlice(io, platform, token_slice);
        defer token_buffer.deinit();

        var pos_buffer: zml.Buffer = try .scalar(io, platform, @as(u32, 0), .u32);
        defer pos_buffer.deinit();

        embedder_output = try .uninitialized(io, platform,
            .init(.{ .s = @as(u32, @intCast(tokens.len)), .d = config.dim }, .bf16), .{});

        emb_args.set(.{ &embedder_buffers, adapter_output, token_buffer, pos_buffer });
        compiled_embedder.call(emb_args, &emb_results);
        emb_results.fill(.{&embedder_output});
    }
    defer embedder_output.deinit();
    log.info("Embedder done.", .{});

    // 5. Init KV cache + t_cond buffer
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

    // 6. Decoder prefill
    log.info("Running decoder prefill...", .{});
    var generated_token_slice: zml.Slice = try .alloc(allocator, .init(.{@as(u32, 1)}, .u32));
    defer generated_token_slice.free(allocator);
    {
	defer compiled_prefill.deinit();
	
        var prefill_args = try compiled_prefill.args(allocator);
        defer prefill_args.deinit(allocator);
        var prefill_results = try compiled_prefill.results(allocator);
        defer prefill_results.deinit(allocator);

        var token_index_buffer: zml.Buffer = try .scalar(io, platform, @as(u32, 0), .u32);
        defer token_index_buffer.deinit();

        var prefill_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{@as(u32, @intCast(tokens.len))}, .u32));
        defer prefill_tokens_slice.free(allocator);
        var prefill_output: zml.Buffer = try .fromSlice(io, platform, prefill_tokens_slice);
        defer prefill_output.deinit();

        prefill_args.set(.{ &decoder_buffers, embedder_output, token_index_buffer, &kv_cache_buffers, t_cond_buffer });
        compiled_prefill.call(prefill_args, &prefill_results);
        prefill_results.fill(.{ &prefill_output, &kv_cache_buffers });

        // Extract last token from prefill output (first generated token)
        try prefill_output.toSlice(io, prefill_tokens_slice);
        log.info("Prefill output tokens: {any}", .{prefill_tokens_slice.items(u32)[0..tokens.len]});
        generated_token_slice.items(u32)[0] = prefill_tokens_slice.items(u32)[tokens.len - 1];
        log.info("First generated token (from prefill): {}", .{generated_token_slice.items(u32)[0]});
    }
    log.info("Prefill done.", .{});

    // 7. Autoregressive decode loop
    log.info("Running decoder autoregressive...", .{});
    const eos_token: u32 = 2;
    {
	defer compiled_decoder.deinit();
	
        var decode_args = try compiled_decoder.args(allocator);
        defer decode_args.deinit(allocator);
        var decode_results = try compiled_decoder.results(allocator);
        defer decode_results.deinit(allocator);

        var emb_args = try compiled_decode_embedder.args(allocator);
        defer emb_args.deinit(allocator);
        var emb_results = try compiled_decode_embedder.results(allocator);
        defer emb_results.deinit(allocator);

        var current_token_buffer: zml.Buffer = try .fromSlice(io, platform, generated_token_slice);
        defer current_token_buffer.deinit();

        var decode_embed: zml.Buffer = try .uninitialized(io, platform,
            .init(.{ .s = @as(u32, 1), .d = config.dim }, .bf16), .{});
        defer decode_embed.deinit();

        const max_pos = adapter_seq_len;
        log.info("Decode loop: tokens.len={}, max_pos={}", .{tokens.len, max_pos});
        var num_generated: usize = 0;
        for (tokens.len..max_pos) |i| {
            const generated_token = generated_token_slice.items(u32)[0];
            num_generated += 1;

            if (generated_token != 32) {
                if (try tokenizer_decoder.next(generated_token)) |chunk| {
                    std.debug.print("{s}", .{chunk});
		}
            }

            
            if (generated_token == eos_token) {
                log.info("  EOS reached at pos={}", .{i});
                break;
            }

            // 1. Embedder: combine adapter_output[pos] + tok_embed(token)
            const token_slice: zml.Slice = .init(.init(.{@as(u32, 1)}, .u32),
                std.mem.sliceAsBytes(generated_token_slice.items(u32)[0..1]));
            var token_buf: zml.Buffer = try .fromSlice(io, platform, token_slice);
            defer token_buf.deinit();

            var pos_buffer: zml.Buffer = try .scalar(io, platform, @as(u32, @intCast(i)), .u32);
            defer pos_buffer.deinit();

            emb_args.set(.{ &embedder_buffers, adapter_output, token_buf, pos_buffer });
            compiled_decode_embedder.call(emb_args, &emb_results);
            emb_results.fill(.{&decode_embed});

            // 2. Decoder: get next token
            var token_index_buffer: zml.Buffer = try .scalar(io, platform, @as(u32, @intCast(i)), .u32);
            defer token_index_buffer.deinit();

            decode_args.set(.{ &decoder_buffers, decode_embed, token_index_buffer, &kv_cache_buffers, t_cond_buffer });
            compiled_decoder.call(decode_args, &decode_results);
            decode_results.fill(.{ &current_token_buffer, &kv_cache_buffers });

            try current_token_buffer.toSlice(io, generated_token_slice);
        }
        std.debug.print("\n", .{});
        log.info("Decode done. Generated {} tokens total.", .{num_generated});
    }
}

pub fn compileMelSpectrum(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: LogMelSpectrogram, padded_audio_len: usize) !zml.Exe {
    return try platform.compile(allocator, io, model, .forward, .{Tensor.init(.{padded_audio_len}, .f32).withTags(.{.samples})});
}

pub fn compileEncoder(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Encoder, padded_audio_len: usize) !zml.Exe {
    const num_frames = padded_audio_len / model.config.audio().hop_length;
    
      return try platform.compile(allocator, io, model, .forward, .{
          Tensor.init(.{ .channels = 128, .time = num_frames }, .f32),
      });
}

pub fn compileAdapter(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Adapter, encoder_seq_len: u32, config: Config) !zml.Exe {
    return try platform.compile(allocator, io, model, .forward, .{
        Tensor.init(.{ .s = encoder_seq_len, .d = config.encoder().dim }, .bf16),
    });
}

pub fn compileEmbedder(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Embedder, adapter_seq_len_: u32, seq_len: u32, config: Config) !zml.Exe {
    return try platform.compile(allocator, io, model, .forward, .{
        Tensor.init(.{ .s = adapter_seq_len_, .d = config.dim }, .bf16),
        Tensor.init(.{ .s = seq_len }, .u32),
        Tensor.init(.{}, .u32),
    });
}

const VoxtralParameters = struct {
    prefill_embeds: Tensor,
    decode_embeds: Tensor,
    token_index: Tensor,
    kv_cache: dec.KvCache,
    t_cond: Tensor,
};

pub fn compileDecoder(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Decoder, params: VoxtralParameters) !struct { zml.Exe, zml.Exe} {
    const compile = struct {
	fn call(allocator_: std.mem.Allocator, io_: std.Io, platform_: *const zml.Platform, model_: Decoder, embeds_: Tensor, params_: VoxtralParameters) !zml.Exe {
	    return try platform_.compile(allocator_, io_, model_, .forward, .{
		embeds_,
		params_.token_index,
		params_.kv_cache,
		params_.t_cond,
	    });
	}
    }.call;
    
    var prefill_future = try io.concurrent(compile, .{ allocator, io, platform, model, params.prefill_embeds, params });
    var decode_future = try io.concurrent(compile, .{ allocator, io, platform, model, params.decode_embeds, params });

    return .{
        try prefill_future.await(io),
        try decode_future.await(io),
    };
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
