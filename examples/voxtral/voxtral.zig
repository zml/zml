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
const tokens: []u32 = .{1, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32};
const audio_len = 293699;

const n_delay_tokens = 6;
const max_seq_len = 1024;
const seq_len = 1024;

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
	.prefill_embeds = .init(.{.s = max_seq_len, .d = config.dim}, .bf16),
	.decode_embeds = .init(.{.s = 1, .d = config.dim}, .bf16),
	.token_index = .init(.{}, .u32),
	.kv_cache = .init(.init(.{
	    .layer = model.layers.len,
	    .k = max_seq_len,
	    .h = config.n_kv_heads,
	    .hd = config.head_dim,
	}, dtype)),
	.t_cond = Tensor.init(.{.d = config.dim}, dtype),
    };
    
    var tokenizer_future = try io.concurrent(loadTokenizer, .{allocator, io, model_dir});
    var compiled_mel_spectrum_future = try io.concurrent(compileMelSpectrum, .{allocator, io, platform, melspectro_model});
    var compiled_encoder_future = try io.concurrent(compileEncoder, .{allocator, io, platform, encoder_model});
    const num_frames = audio_len / config.audio().hop_length;
    const encoder_seq_len: u32 = @intCast((num_frames / 2) - (num_frames / 2) % config.downsample_factor());
    var compiled_adapter_future = try io.concurrent(compileAdapter, .{allocator, io, platform, adapter, encoder_seq_len});
    const adapter_seq_len = encoder_seq_len / config.downsample_factor();
    var compiled_embedder_future = try io.concurrent(compileEmbedder, .{allocator, io, platform, embedder, adapter_seq_len});
    var compiled_decoder_future = try io.concurrent(compileDecoder, .{allocator, io, platform, model, model_params});
    
    var mel_spectrum_buffers_future = try io.concurrent(LogMelSpectrogram.load, .{&melspectro_model, io, platform});
    var encoder_buffers_future = try io.concurrent(Encoder.load, .{&encoder_model,allocator,io, platform, &model_store, &progress});
    var adapter_buffers_future = try io.concurrent(Adapter.load, .{&adapter, allocator, io, platform, &model_store, &progress});

    var tokenizer = try tokenizer_future.await(io);
    tokenizer.deinit();

    // -- Compiled models
 
    var compiled_mel_spectrum = try compiled_mel_spectrum_future.await(io);
    defer compiled_mel_spectrum.deinit();
    
    var compiled_encoder = try compiled_encoder_future.await(io);
    defer compiled_encoder.deinit();
    _ = &compiled_encoder;

    var compiled_adapter = try compiled_adapter_future.await(io);
    defer compiled_adapter.deinit();
    _ = &compiled_adapter;

    var compiled_embedder = try compiled_embedder_future.await(io);
    defer compiled_embedder.deinit();
    _ = &compiled_embedder;

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
    progress.end();
    
    // try output_buffer.toSlice(io, output_slice);

    // const outfile = try std.Io.Dir.createFile(.cwd(), io, args.output, .{});
    // defer outfile.close(io);

    // var outbuff: [4096]u8 = undefined;
    // var writer = outfile.writer(io, &outbuff);
    // const writer_interface = &writer.interface;

    // try writer_interface.writeAll(output_slice.data());
}

fn generateTranscript(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, config: Config, wav_file: []f32, mel_spectogram: zml.Exe) !void {
        // Execute
    var mel_args = try mel_spectogram.args(allocator);
    defer mel_args.deinit(allocator);
    var mel_results = try mel_spectogram.results(allocator);
    defer mel_results.deinit(allocator);

    const num_frames = audio_len / config.audio().hop_length;
    const input_slice = zml.Slice.init(.init(.{ .freq = wav_file.len }, .f32), std.mem.sliceAsBytes(wav_file));
    var input_buffer: zml.Buffer = try .fromSlice(io, platform, input_slice);
    defer input_buffer.deinit();

    const output_shape = zml.Shape.init(.{ 128, num_frames }, .f32);
    
    const output_slice: zml.Slice = try zml.Slice.alloc(allocator, output_shape);
    defer output_slice.free(allocator);
    
    var output_buffer: zml.Buffer = try .fromSlice(io, platform, output_slice);
    defer output_buffer.deinit();

    // mel_args.set(.{mel_spectrum_buffers, input_buffer});
    // compiled_mel_spectrum.call(exec_args, &results);
    // results.fill(.{&output_buffer});    

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

pub fn compileAdapter(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Adapter, encoder_seq_len: u32) !zml.Exe {
    return try platform.compile(allocator, io, model, .forward, .{
        Tensor.init(.{ .s = encoder_seq_len, .d = 1280 }, .bf16),
    });
}

pub fn compileEmbedder(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Embedder, adapter_seq_len: u32) !zml.Exe {
    return try platform.compile(allocator, io, model, .forward, .{
        Tensor.init(.{ .s = adapter_seq_len, .d = 3072 }, .bf16),
        Tensor.init(.{ .s = adapter_seq_len }, .u32),
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
// fn computeTimeEmbedding(allocator: std.mem.Allocator, t_value: f32, dim: u32) []f32 {
//     const half_dim = dim / 2;
//     const result = allocator.alloc(f32, dim) catch unreachable;
//     for (0..half_dim) |i| {
//         const freq = @exp(-@log(@as(f32, 10000.0)) * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half_dim)));
//         const angle = t_value * freq;
//         result[i] = @cos(angle);
//         result[i + half_dim] = @sin(angle);
//     }
//     return result;
// }

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
