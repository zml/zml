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
	    .mel_filters = Tensor.init(.{201, 128}, .f32).withTags(.{.freq_bins, .mel}),
	    .hop_len = audio.hop_length,
	    .n_fft = audio.window_size,
	    .global_log_mel_max = audio.global_log_mel_max,
	};
    }

    pub fn forward(self: LogMelSpectrogram, waveform: Tensor) Tensor {
	const dtype = waveform.dtype();

        const window_weight = self.window.getWeights(self.n_fft, dtype);
        const fft_len = window_weight.dim(.samples);
        const num_frames: u63 = @intCast(@divFloor(waveform.dim(.samples), self.hop_len));

        // Reflect padding
        const padded_wav = blk: {
            const l = waveform.slice1d(.samples, .{ .start = 1, .end = @divExact(fft_len, 2) + 1 }).reverse(.{.samples});
            const r = waveform.slice1d(.samples, .{ .start = -@divExact(fft_len, 2) - 1, .end = -1 }).reverse(.{.samples});
            break :blk zml.Tensor.concatenate(&.{ l, waveform, r }, .samples);
        };

        // Use Short Time Fourier Transform to compute features.
        // Generate num_frames+1 (matching torch.stft center=True frame count),
        // then drop the last frame to match Whisper convention (stft[..., :-1]).
        var spectrogram = stft(padded_wav, window_weight, num_frames + 1, self.hop_len, self.precision);
        spectrogram = spectrogram.slice1d(.frames, .{ .end = -1 });
        spectrogram = spectrogram.convert(dtype);
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
	    .mel_filters = try zml.Buffer.fromSlice(io, platform, slice),
	};
    }

    pub fn unload(self: *zml.Bufferized(LogMelSpectrogram)) void {
	self.mel_filters.deinit();
    }
};

pub fn stft(waveform: Tensor, weight: Tensor, num_frames: usize, stride: u63, precision: zml.DataType) Tensor {
    const fft_len = weight.dim(0);

    const indices = Tensor.arange(.{ .end = @intCast(num_frames * stride), .step = stride }, .i32);
    var windows = waveform.gatherSlices(.{fft_len}, indices.appendAxes(.{.coord }), .{ .indices_are_sorted = true });

    windows = windows.mul(weight.broadcastLeft(windows.shape()));

    var fft = windows.convert(precision).fft(.{ .kind = .RFFT, .length = &.{fft_len} });
    const spectrogram = fft.abs();

    return spectrogram.mul(spectrogram).convert(waveform.dtype()).withTags(.{.frames, .freq_bins});
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
