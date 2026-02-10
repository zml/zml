const std = @import("std");
const builtin = @import("builtin");
const log = std.log;
const wav_utils = @import("wav.zig");
const cfg = @import("config.zig");
const MelSpectrumConfig = cfg.MelSpectrumConfig;

const zml = @import("zml");
const Tensor = zml.Tensor;

pub const LogMelSpectrogram = struct {
    window: AudioWindow,
    
    n_fft: u63 = 400, // Must match window size
    mel_filters: Tensor, // Shape must match (201, 128)
    
    hop_len: u63 = 160,
    global_log_mel_max: f32 = 1.5,
    mel_floor: f32 = 1e-10,
    force_num_frames: u32 = 3000,
    
    precision: zml.DataType = .f32,
    
    pub fn init(config: MelSpectrumConfig) LogMelSpectrogram {
	return .{
	    .window = .hann,
	    .mel_filters = Tensor.init(.{201, 128}, .f32).withTags(.{.freq, ._ }),
	    .hop_len = config.hop_length,
	    .n_fft = config.n_fft,
	    .global_log_mel_max = 1.5,
	};
    }
    
    pub fn forward(self: LogMelSpectrogram, waveform: Tensor) Tensor {
	const dtype = waveform.dtype();
        const rank: u63 = @intCast(waveform.shape().rank());
        const window_weight = self.window.getWeights(self.n_fft, dtype);
        const fft_len = window_weight.dim(rank - 1);
        var num_frames: u63 = @intCast(@divFloor(waveform.dim(rank - 1), self.hop_len));

        var wav = waveform;
	const force_num_frames = self.force_num_frames;
        const force_num_samples = force_num_frames * self.hop_len;
        if (num_frames > force_num_frames) {
            wav = wav.slice1d(-1, .{ .end = force_num_samples });
            num_frames = force_num_frames;
        } else if (num_frames < force_num_frames) {
	    const tagged = wav.withTags(.{.t});
	    wav = tagged.pad(0.0, .{ .t = Tensor.Pad{ .high = force_num_samples - tagged.dim(.t) } });
        }
	
        num_frames = @intCast(@divFloor(wav.dim(0), self.hop_len));
        std.debug.assert(num_frames == force_num_frames);

        // Reflect padding
        const padded_wav = blk: {
            const l = wav.slice1d(-1, .{ .start = 1, .end = @divExact(fft_len, 2) + 1 }).reverse(.{-1});
            const r = wav.slice1d(-1, .{ .start = -@divExact(fft_len, 2) - 1, .end = -1 }).reverse(.{-1});
            break :blk zml.Tensor.concatenate(&.{ l, wav, r }, -1);
        };

        // Use Short Time Fourier Transform to compute features.
        // Generate num_frames+1 (matching torch.stft center=True frame count),
        // then drop the last frame to match Whisper convention (stft[..., :-1]).
        var spectrogram = stft(padded_wav, window_weight, num_frames + 1, self.hop_len, self.precision);
        spectrogram = spectrogram.slice1d(0, .{ .end = -1 });
        spectrogram = spectrogram.convert(dtype);
        // Re-weight frequencies for speech
        spectrogram = spectrogram.dot(self.mel_filters, .freq);

        spectrogram = spectrogram.maximum(Tensor.constant(dtype.constant(self.mel_floor)));
        var log_spec = spectrogram.log().scale(1.0 / @log(10.0));

        const log_spec_min = Tensor.constant(dtype.constant(self.global_log_mel_max - 8.0));
        log_spec = log_spec.maximum(log_spec_min);
        const log_spec_rank = log_spec.shape().rank();
	
        // "center" the distribution
        return log_spec.addConstant(4).scale(1.0 / 4.0).transpose(.{ log_spec_rank - 1, log_spec_rank - 2 });
    }

    pub fn load(self: *LogMelSpectrogram, io: std.Io, platform: *zml.Platform) !zml.Bufferized(LogMelSpectrogram) {
	const mel_filters_data = @embedFile("assets/voxtral_mel_filter.data");
	const slice = zml.Slice.init(self.mel_filters.shape(), mel_filters_data);
	
	return .{
	    .mel_filters = try zml.Buffer.fromSlice(io, platform, slice), 
	};
    }

    pub fn unload(self: *zml.Bufferized(LogMelSpectrogram)) void {
	self.mel_filters.deinit();
    }
};

pub fn stft(waveform: Tensor, weight: Tensor, num_frames: usize, stride: u63, precision: zml.DataType) Tensor {
    const fft_len = weight.dim(0);

    const num_samples = waveform.dim(0);

    // const num_frames = 3000;
    const indices = Tensor.arange(.{ .end = @intCast(num_frames * stride), .step = stride }, .i32);
    std.log.warn("num_samples: {d}, num_frames: {d}, end: {d}, stride: {d}, indices: {}", .{ num_samples, num_frames, num_frames * stride, stride, indices });
    var windows = waveform.gatherSlices(.{fft_len}, indices.appendAxes(.{.coord }), .{ .indices_are_sorted = true });

    windows = windows.mul(weight.broadcastLeft(windows.shape()));

    var fft = windows.convert(precision).fft(.{ .kind = .RFFT, .length = &.{fft_len} });
    const spectrogram = fft.abs();
    
    const ret = spectrogram.mul(spectrogram).convert(waveform.dtype()).withTags(.{.temp, .freq});
    log.info("{f}", .{ret.shape()});
    
    return ret;
}


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
    
    const io = threaded.io();
    
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

    const output_shape = zml.Shape.init(.{ 128, melspectro_model.force_num_frames }, .f32);
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
    return try platform.compile(allocator, io, model, .forward, .{Tensor.init(.{293699}, .f32).withTags(.{.freq})});
}

pub const AudioWindow = enum {
    /// https://numpy.org/doc/stable/reference/generated/numpy.hanning.html
    hann,
    boxcar,
    // Other possible windows: https://en.wikipedia.org/wiki/Window_function

    pub fn getWeights(self: AudioWindow, len: i64, dtype: zml.DataType) Tensor {
        return switch (self) {
            .boxcar => Tensor.constant(dtype.one()),
            .hann => {
                if (len <= 1) return Tensor.constant(dtype.one());
                const flen: f64 = @floatFromInt(len);
                const freq = Tensor.constant(dtype.constant(std.math.pi / flen));
                const steps = Tensor.arange(.{ .start = -len, .end = len, .step = 2 }, dtype);
                return steps.mul(freq).cos().scale(0.5).addConstant(0.5).convert(dtype);
            },
        };
    }
};
