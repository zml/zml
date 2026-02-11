const std = @import("std");
const builtin = @import("builtin");
const log = std.log;
const wav_utils = @import("wav.zig");
const cfg = @import("config.zig");
const MelSpectrumConfig = cfg.MelSpectrumConfig;

const zml = @import("zml");
const Tensor = zml.Tensor;

const mel = @import("mel_spectrogram.zig");
const LogMelSpectrogram = mel.LogMelSpectrogram;

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

    var vfs: zml.io.VFS = try .init(allocator, threaded.io());
    defer vfs.deinit();
    
    var vfs_file: zml.io.VFS.File = .init(allocator, threaded.io(), .{});
    defer vfs_file.deinit();
    try vfs.register("file", vfs_file.io());

    const io = vfs.io();

    const arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();

    const file = try std.Io.Dir.openFile(.cwd(), io, "/Users/raph/Documents/Git-Repos/zml/examples/voxtral/inputs/harvard_16k.wav", .{});
    defer file.close(io);

    var wav_buffer: [4096]u8 = undefined;
    var reader = file.reader(io, &wav_buffer);

    const wav_file = try loadWav(allocator, &reader.interface);
    std.debug.print("LEN: {d}\n", .{wav_file.len});
    defer allocator.free(wav_file);
    std.debug.print("{any}\n", .{wav_file[0..100]});

    const melspectrum_config = MelSpectrumConfig{};

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform .deinit(allocator);
    log.info("Selected platform {f}\n", .{platform.fmtVerbose()});

    var melspectro_model: LogMelSpectrogram = .init(melspectrum_config);
    var compiled_mel_spectrum_future = try io.concurrent(compileMelSpectrum, .{allocator, io, platform, melspectro_model});
    var mel_spectrum_buffers_future = try io.concurrent(LogMelSpectrogram.load, .{&melspectro_model, io, platform});

    var compiled_mel_spectrum = try compiled_mel_spectrum_future.await(io);
    defer compiled_mel_spectrum.deinit();
    var mel_spectrum_buffers = try mel_spectrum_buffers_future.await(io);
    defer LogMelSpectrogram.unload(&mel_spectrum_buffers);

    // Execute
    var args = try compiled_mel_spectrum.args(allocator);
    defer args.deinit(allocator);
    var results = try compiled_mel_spectrum.results(allocator);
    defer results.deinit(allocator);

    const input_slice = zml.Slice.init(.init(.{ .freq = wav_file.len }, .f32), std.mem.sliceAsBytes(wav_file));
    var input_buffer: zml.Buffer = try .fromSlice(io, platform, input_slice);
    defer input_buffer.deinit();

    // const output_shape = zml.Shape.init(.{ 128, melspectro_model.force_num_frames }, .f32);
    const num_frames = wav_file.len / melspectrum_config.hop_length;
    const output_shape = zml.Shape.init(.{ 128, num_frames }, .f32);
    const output_slice: zml.Slice = try zml.Slice.alloc(allocator, output_shape);
    defer output_slice.free(allocator);
    var output_buffer: zml.Buffer = try .fromSlice(io, platform, output_slice);
    defer output_buffer.deinit();

    args.set(.{mel_spectrum_buffers, input_buffer});
    compiled_mel_spectrum.call(args, &results);
    results.fill(.{&output_buffer});


    try output_buffer.toSlice(io, output_slice);

    const outfile = try std.Io.Dir.createFile(.cwd(), io, "/Users/raph/Documents/Git-Repos/zml/examples/voxtral/outputs/out.bin", .{});
    defer outfile.close(io);

    var outbuff: [4096]u8 = undefined;
    var writer = outfile.writer(io, &outbuff);
    const writer_interface = &writer.interface;

    try writer_interface.writeAll(output_slice.data());

    // std.debug.print("Here is the results: {f}\n", .{first});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, "/Users/raph/Documents/Git-Repos/zml/examples/voxtral/outputs/voxtral_activations.safetensors");
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    // try zml.testing.testlayer(allocator, io, platform.*, melspectro_model, .forward, store.view(), "mel", mel_spectrum_buffers, 1e-3);
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
    // return try platform.compile(allocator, io, model, .forward, .{Tensor.init(.{model.force_num_frames * model.hop_len}, .f32).withTags(.{.freq})});
    return try platform.compile(allocator, io, model, .forward, .{Tensor.init(.{293699}, .f32).withTags(.{.samples})});
}
