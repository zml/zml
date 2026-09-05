const std = @import("std");
const cfg = @import("config.zig");
const Config = cfg.Config;

const zml = @import("zml");
const Tensor = zml.Tensor;

pub const LogMelSpectrogram = struct {
    window: AudioWindow,

    n_fft: u63, // Must match window size
    mel_filters: Tensor, // Shape must match (201, 128)
    mel_floor: f32 = 1e-10,

    hop_len: u63,
    global_log_mel_max: f32,

    precision: zml.DataType = .f32,

    pub fn init(config: Config) LogMelSpectrogram {
        const audio = config.audio();

        return .{
            .window = .hann,
            .mel_filters = Tensor.init(.{ 201, 128 }, .f32).withTags(.{ .freq_bins, .mel }),
            .hop_len = audio.hop_length,
            .n_fft = audio.window_size,
            .global_log_mel_max = audio.global_log_mel_max,
        };
    }

    /// Computes the log mel spectrogram for a fixed-size audio chunk.
    /// Prefill: audio_chunk [samples=(prompt_len-1)*mel_per_step*hop_length + chunk_audio] → [channels=128, time=prompt_len*mel_per_step] f32
    /// Step:    audio_chunk [samples=chunk_audio]                                          → [channels=128, time=mel_per_step] f32
    pub fn melStep(self: LogMelSpectrogram, audio_chunk: Tensor) Tensor {
        const dtype = audio_chunk.dtype();

        const window_weight = self.window.getWeights(self.n_fft, dtype);
        const fft_len = window_weight.dim(.samples);
        const audio_len: u63 = @intCast(audio_chunk.dim(.samples));
        const num_frames: u63 = @intCast(@divFloor(audio_len - fft_len, self.hop_len) + 1);

        // No reflect padding — done on host
        const spectrogram = stft(audio_chunk, window_weight, num_frames, self.hop_len, self.precision);

        return self.postProcess(spectrogram, dtype).withTags(.{ .channels, .time });
    }

    /// Shared post-STFT processing: mel filter dot product, log scale, clamp, normalize.
    fn postProcess(self: LogMelSpectrogram, raw_spectrogram: Tensor, dtype: zml.DataType) Tensor {
        var spectrogram = raw_spectrogram.convert(dtype);

        // Re-weight frequencies for speech
        spectrogram = spectrogram.dot(self.mel_filters, .freq_bins);

        spectrogram = spectrogram.maximum(Tensor.constant(dtype.constant(self.mel_floor)));
        var log_spec = spectrogram.log().scale(1.0 / @log(10.0));

        const log_spec_min = Tensor.constant(dtype.constant(self.global_log_mel_max - 8.0));
        log_spec = log_spec.maximum(log_spec_min);

        // "center" the distribution
        return log_spec.addConstant(4).scale(1.0 / 4.0).transpose(.{ .mel, .frames });
    }

    pub fn load(self: *LogMelSpectrogram, io: std.Io, platform: *zml.Platform) !zml.Bufferized(LogMelSpectrogram) {
        const mel_filters_data = @embedFile("assets/voxtral_mel_filter.data");
        const slice = zml.Slice.init(self.mel_filters.shape(), mel_filters_data);

        return .{
            .mel_filters = try zml.Buffer.fromSlice(io, platform, slice, .replicated),
        };
    }

    pub fn unload(self: *zml.Bufferized(LogMelSpectrogram)) void {
        self.mel_filters.deinit();
    }
};

pub fn stft(waveform: Tensor, weight: Tensor, num_frames: usize, stride: u63, precision: zml.DataType) Tensor {
    const fft_len = weight.dim(0);

    const indices = Tensor.arange(.{ .end = @intCast(num_frames * stride), .step = stride }, .i32);
    var windows = waveform.gatherSlices(.{fft_len}, indices.appendAxes(.{.coord}), .{ .indices_are_sorted = true });

    windows = windows.mul(weight.broadcastLeft(windows.shape()));

    // The Metal PJRT backend does not implement FFT. A 400-point real DFT
    // is small enough to express as GPU multiply/reduce operations instead.
    if (zml.Compiler.current().platform.target == .metal) {
        return dftPower(windows.convert(precision)).convert(waveform.dtype());
    }

    var fft = windows.convert(precision).fft(.{ .kind = .RFFT, .length = &.{fft_len} });
    const spectrogram = fft.abs();

    return spectrogram.mul(spectrogram).convert(waveform.dtype()).withTags(.{ .frames, .freq_bins });
}

pub fn dftPower(windows: Tensor) Tensor {
    const n: usize = @intCast(windows.dim(-1));
    const bins = n / 2 + 1;
    const allocator = zml.Compiler.current().allocator;
    const real = allocator.alloc(f32, n * bins) catch @panic("out of memory");
    defer allocator.free(real);
    const imag = allocator.alloc(f32, n * bins) catch @panic("out of memory");
    defer allocator.free(imag);
    for (0..n) |sample| {
        for (0..bins) |bin| {
            const angle = -2.0 * std.math.pi * @as(f64, @floatFromInt(sample * bin)) / @as(f64, @floatFromInt(n));
            real[sample * bins + bin] = @floatCast(@cos(angle));
            imag[sample * bins + bin] = @floatCast(@sin(angle));
        }
    }
    const shape = zml.Shape.init(.{ .samples = n, .freq_bins = bins }, .f32);
    const input = windows.withTags(.{ .frames, .samples });
    const full_shape = zml.Shape.init(.{ .frames = windows.dim(0), .samples = n, .freq_bins = bins }, input.dtype());
    // Tensor.dot currently requests fast (reduced) precision. Cancellation in
    // the Fourier sums needs f32 products, especially for low-energy bins.
    const re = input.broad(full_shape).mul(Tensor.constantTensor(shape, std.mem.sliceAsBytes(real)).convert(input.dtype()).broad(full_shape)).sum(.samples);
    const im = input.broad(full_shape).mul(Tensor.constantTensor(shape, std.mem.sliceAsBytes(imag)).convert(input.dtype()).broad(full_shape)).sum(.samples);
    return re.mul(re).add(im.mul(im)).squeeze(.samples);
}

pub const AudioWindow = enum {
    /// https://numpy.org/doc/stable/reference/generated/numpy.hanning.html
    hann,
    boxcar,
    // Other possible windows: https://en.wikipedia.org/wiki/Window_function

    pub fn getWeights(self: AudioWindow, len: i64, dtype: zml.DataType) Tensor {
        return switch (self) {
            .boxcar => Tensor.constant(dtype.one()).withTags(.{.samples}),
            .hann => {
                if (len <= 1) return Tensor.constant(dtype.one());

                const flen: f64 = @floatFromInt(len);
                const freq = Tensor.constant(dtype.constant(std.math.pi / flen));
                const steps = Tensor.arange(.{ .start = -len, .end = len, .step = 2 }, dtype);

                return steps.mul(freq).cos().scale(0.5).addConstant(0.5).convert(dtype).withTags(.{.samples});
            },
        };
    }
};
