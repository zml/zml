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

    /// Like forward(), but for fixed-size audio chunks with reflect padding done on host.
    /// Produces exactly (audio_len - n_fft) / hop_len + 1 mel frames.
    /// Output tags: .channels and .time (matching conv stem input).
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
    var windows = waveform.gatherSlices(.{fft_len}, indices.appendAxes(.{.coord}), .{ .indices_are_sorted = true });

    windows = windows.mul(weight.broadcastLeft(windows.shape()));

    var fft = windows.convert(precision).fft(.{ .kind = .RFFT, .length = &.{fft_len} });
    const spectrogram = fft.abs();

    return spectrogram.mul(spectrogram).convert(waveform.dtype()).withTags(.{ .frames, .freq_bins });
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
