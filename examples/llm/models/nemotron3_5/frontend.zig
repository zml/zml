const std = @import("std");

const audio = @import("audio.zig");

/// Nemotron's preprocessor JSON names and defaults. Unspecified numerical
/// conventions follow NeMo's AudioToMelSpectrogramPreprocessor / FilterbankFeatures.
pub const Config = struct {
    sample_rate: u32 = 16_000,
    window_size: ?f64 = 0.025,
    window_stride: ?f64 = 0.01,
    n_window_size: ?usize = null,
    n_window_stride: ?usize = null,
    window: enum { hann, hamming, blackman, bartlett } = .hann,
    normalize: ?[]const u8 = null,
    features: usize = 128,
    n_fft: ?usize = 512,
    preemph: ?f32 = 0.97,
    lowfreq: f64 = 0,
    highfreq: ?f64 = null,
    log: bool = true,
    log_zero_guard_type: enum { add, clamp } = .add,
    log_zero_guard_value: f32 = 0x1p-24,
    /// Eval never applies dither, even if the checkpoint specifies it.
    dither: f32 = 1e-5,
    pad_to: usize = 0,
    pad_value: f32 = 0,
    frame_splicing: usize = 1,
    mag_power: f32 = 2,
    mel_norm: ?[]const u8 = "slaney",
    mel_scale: enum { slaney, htk } = .slaney,
    periodic_window: bool = false,
    exact_pad: bool = false,
    snip_edges: bool = false,
    log_base: ?f64 = null,
    /// NeMo main explicitly uses constant padding. Older versions used reflect.
    /// This compatibility option is local, not a NeMo preprocessor JSON field.
    pad_mode: enum { constant, reflect } = .constant,
};

pub const Features = struct {
    /// Owned dense [frameCount, binCount], indexed as data[t * binCount + bin].
    data: []f32,
    frameCount: usize,
    binCount: usize,
    /// NeMo's returned sequence length, excluding its masked trailing frame
    /// and any pad_to alignment. Pass this length to the audio encoder.
    validFrameCount: usize,

    pub fn deinit(self: Features, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }
};

pub const ComputeError = std.mem.Allocator.Error || error{
    InvalidAudioFormat,
    EmptyAudio,
    NonFiniteAudioSample,
    InvalidFrontendConfig,
    UnsupportedFrontendConfig,
    AudioTooShortForReflectPadding,
    NonFiniteFeatures,
};

/// Borrows mono prepared audio without modifying it. Computes deterministic eval
/// features, including NeMo's dense output masking and optional pad_to alignment.
/// Reference: NVIDIA/NeMo main, nemo/collections/asr/parts/preprocessing/features.py.
pub fn compute(allocator: std.mem.Allocator, prepared: audio.DecodedAudio, config: Config) ComputeError!Features {
    @setFloatMode(.strict);
    if (prepared.channels != 1 or prepared.sample_rate != config.sample_rate or config.sample_rate != audio.targetSampleRate) return error.InvalidAudioFormat;
    if (prepared.samples.len == 0) return error.EmptyAudio;
    for (prepared.samples) |sample| {
        if (!std.math.isFinite(sample)) return error.NonFiniteAudioSample;
    }

    const winLength = try windowSamples(config.window_size, config.n_window_size, config.sample_rate);
    const hopLength = try windowSamples(config.window_stride, config.n_window_stride, config.sample_rate);
    const fftSize = config.n_fft orelse (std.math.ceilPowerOfTwo(usize, winLength) catch return error.InvalidFrontendConfig);
    const nyquist = @as(f64, @floatFromInt(config.sample_rate)) / 2;
    // NeMo treats both None and zero highfreq as Nyquist.
    const highfreq = if (config.highfreq == null or config.highfreq.? == 0) nyquist else config.highfreq.?;
    if (fftSize < 2 or !std.math.isPowerOfTwo(fftSize) or fftSize < winLength or config.features == 0) return error.InvalidFrontendConfig;
    if (!std.math.isFinite(config.lowfreq) or !std.math.isFinite(highfreq) or config.lowfreq < 0 or highfreq <= config.lowfreq or highfreq > nyquist) return error.InvalidFrontendConfig;
    if (!std.math.isFinite(config.mag_power) or config.mag_power <= 0 or !std.math.isFinite(config.pad_value)) return error.InvalidFrontendConfig;
    if (!std.math.isFinite(config.log_zero_guard_value) or config.log_zero_guard_value <= 0) return error.InvalidFrontendConfig;
    if (!std.math.isFinite(config.dither) or config.dither < 0) return error.InvalidFrontendConfig;
    if (config.preemph) |coefficient| {
        if (!std.math.isFinite(coefficient)) return error.InvalidFrontendConfig;
    }
    if (config.log_base) |base| {
        if (!std.math.isFinite(base) or base <= 0 or base == 1) return error.InvalidFrontendConfig;
    }
    if (config.exact_pad or config.snip_edges or config.frame_splicing != 1) return error.UnsupportedFrontendConfig;
    if (config.mel_norm) |norm| {
        if (!std.mem.eql(u8, norm, "slaney")) return error.UnsupportedFrontendConfig;
    }
    const normalization = try normalizationMode(config.normalize);
    if (config.pad_mode == .reflect and prepared.samples.len <= fftSize / 2) return error.AudioTooShortForReflectPadding;

    // Centered even-sized STFT produces floor(N / hop) + 1 frames, but NeMo
    // get_seq_len omits the last one and forward masks it to pad_value.
    const validFrames = prepared.samples.len / hopLength;
    var frameCount = std.math.add(usize, validFrames, 1) catch return error.OutOfMemory;
    if (config.pad_to > 0 and frameCount % config.pad_to != 0) {
        frameCount = std.math.add(usize, frameCount, config.pad_to - frameCount % config.pad_to) catch return error.OutOfMemory;
    }
    const data = try allocator.alloc(f32, std.math.mul(usize, frameCount, config.features) catch return error.OutOfMemory);
    errdefer allocator.free(data);
    @memset(data, config.pad_value);

    const emphasized = try allocator.alloc(f32, prepared.samples.len);
    defer allocator.free(emphasized);
    emphasized[0] = prepared.samples[0];
    for (emphasized[1..], 1..) |*sample, i| {
        sample.* = if (config.preemph) |coefficient| prepared.samples[i] - coefficient * prepared.samples[i - 1] else prepared.samples[i];
        if (!std.math.isFinite(sample.*)) return error.NonFiniteFeatures;
    }

    const window = try allocator.alloc(f32, winLength);
    defer allocator.free(window);
    fillWindow(window, config);
    const filterbank = try makeFilterbank(allocator, config, fftSize, highfreq);
    defer allocator.free(filterbank);
    const spectrum = try allocator.alloc(Complex, fftSize);
    defer allocator.free(spectrum);
    const twiddles = try allocator.alloc(Complex, fftSize / 2);
    defer allocator.free(twiddles);
    for (twiddles, 0..) |*twiddle, i| {
        const phase = -2 * std.math.pi * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(fftSize));
        twiddle.* = .{ .re = @cos(phase), .im = @sin(phase) };
    }
    const fftBins = fftSize / 2 + 1;
    const powers = try allocator.alloc(f32, fftBins);
    defer allocator.free(powers);
    const windowOffset = (fftSize - winLength) / 2;
    const logScale: f32 = if (config.log_base) |base| @floatCast(1 / @log(base)) else 1;

    for (0..validFrames) |frame| {
        @memset(spectrum, .{ .re = 0, .im = 0 });
        const center = frame * hopLength;
        for (window, 0..) |weight, i| {
            const position = @as(i128, @intCast(center)) + @as(i128, @intCast(windowOffset + i)) - @as(i128, @intCast(fftSize / 2));
            const sample = centeredSample(emphasized, position, config.pad_mode);
            spectrum[windowOffset + i].re = sample * weight;
        }
        fft(spectrum, twiddles);
        for (powers, spectrum[0..fftBins]) |*power, bin| {
            // Round the FFT to complex64 before NeMo's magnitude/power operations.
            const re: f32 = @floatCast(bin.re);
            const im: f32 = @floatCast(bin.im);
            const magnitude = @sqrt(re * re + im * im);
            power.* = if (config.mag_power == 1) magnitude else if (config.mag_power == 2) magnitude * magnitude else std.math.pow(f32, magnitude, config.mag_power);
            if (!std.math.isFinite(power.*)) return error.NonFiniteFeatures;
        }
        const output = data[frame * config.features ..][0..config.features];
        for (output, 0..) |*value, mel| {
            var energy: f32 = 0;
            for (filterbank[mel * fftBins ..][0..fftBins], powers) |weight, power| energy += weight * power;
            value.* = if (config.log) @log(switch (config.log_zero_guard_type) {
                .add => energy + config.log_zero_guard_value,
                .clamp => @max(energy, config.log_zero_guard_value),
            }) * logScale else energy;
            if (!std.math.isFinite(value.*)) return error.NonFiniteFeatures;
        }
    }
    normalize(data[0 .. validFrames * config.features], validFrames, config.features, normalization);
    for (data) |value| {
        if (!std.math.isFinite(value)) return error.NonFiniteFeatures;
    }
    return .{ .data = data, .frameCount = frameCount, .binCount = config.features, .validFrameCount = validFrames };
}

fn windowSamples(seconds: ?f64, samples: ?usize, rate: u32) ComputeError!usize {
    if (seconds) |duration| {
        if (samples != null or !std.math.isFinite(duration) or duration <= 0) return error.InvalidFrontendConfig;
        const count = duration * @as(f64, @floatFromInt(rate));
        if (!std.math.isFinite(count) or count < 1 or count >= @as(f64, @floatFromInt(std.math.maxInt(usize)))) return error.InvalidFrontendConfig;
        return @intFromFloat(count); // Python int(window_size * sample_rate) truncates.
    }
    const count = samples orelse return error.InvalidFrontendConfig;
    if (count == 0) return error.InvalidFrontendConfig;
    return count;
}

fn fillWindow(window: []f32, config: Config) void {
    if (window.len == 1) {
        window[0] = 1;
        return;
    }
    const denominator: f32 = @floatFromInt(if (config.periodic_window) window.len else window.len - 1);
    for (window, 0..) |*weight, i| {
        const phase = 2 * @as(f32, std.math.pi) * @as(f32, @floatFromInt(i)) / denominator;
        weight.* = switch (config.window) {
            .hann => 0.5 - 0.5 * @cos(phase),
            .hamming => 0.54 - 0.46 * @cos(phase),
            .blackman => 0.42 - 0.5 * @cos(phase) + 0.08 * @cos(2 * phase),
            .bartlett => 1 - @abs(2 * @as(f32, @floatFromInt(i)) / denominator - 1),
        };
    }
}

fn centeredSample(samples: []const f32, position: i128, mode: @FieldType(Config, "pad_mode")) f32 {
    const length: i128 = @intCast(samples.len);
    if (position >= 0 and position < length) return samples[@intCast(position)];
    if (mode == .constant) return 0;
    // PyTorch reflection excludes the boundary sample: x[-1] = x[1].
    const reflected = if (position < 0) -position else 2 * length - 2 - position;
    return samples[@intCast(reflected)];
}

fn hzToMel(hz: f64, scale: @FieldType(Config, "mel_scale")) f64 {
    if (scale == .htk) return 2595 * @log(1 + hz / 700) / @log(@as(f64, 10));
    return if (hz < 1000) hz / (200.0 / 3.0) else 15 + @log(hz / 1000) / (@log(@as(f64, 6.4)) / 27);
}

fn melToHz(mel: f64, scale: @FieldType(Config, "mel_scale")) f64 {
    if (scale == .htk) return 700 * (std.math.pow(f64, 10, mel / 2595) - 1);
    return if (mel < 15) mel * (200.0 / 3.0) else 1000 * @exp((@log(@as(f64, 6.4)) / 27) * (mel - 15));
}

fn makeFilterbank(allocator: std.mem.Allocator, config: Config, fftSize: usize, highfreq: f64) ComputeError![]f32 {
    const fftBins = fftSize / 2 + 1;
    const edges = try allocator.alloc(f64, std.math.add(usize, config.features, 2) catch return error.OutOfMemory);
    defer allocator.free(edges);
    const lowMel = hzToMel(config.lowfreq, config.mel_scale);
    const highMel = hzToMel(highfreq, config.mel_scale);
    const step = (highMel - lowMel) / @as(f64, @floatFromInt(edges.len - 1));
    for (edges, 0..) |*edge, i| edge.* = melToHz(lowMel + @as(f64, @floatFromInt(i)) * step, config.mel_scale);
    edges[edges.len - 1] = melToHz(highMel, config.mel_scale);
    const weights = try allocator.alloc(f32, std.math.mul(usize, config.features, fftBins) catch return error.OutOfMemory);
    for (0..config.features) |mel| {
        const lowerWidth = edges[mel + 1] - edges[mel];
        const upperWidth = edges[mel + 2] - edges[mel + 1];
        const area = if (config.mel_norm != null) 2 / (edges[mel + 2] - edges[mel]) else 1;
        for (weights[mel * fftBins ..][0..fftBins], 0..) |*weight, bin| {
            const hz = @as(f64, @floatFromInt(bin)) * @as(f64, @floatFromInt(config.sample_rate)) / @as(f64, @floatFromInt(fftSize));
            const triangle: f32 = @floatCast(@max(0, @min((hz - edges[mel]) / lowerWidth, (edges[mel + 2] - hz) / upperWidth)));
            // librosa stores triangles as float32 before applying float64 area scaling.
            weight.* = @floatCast(@as(f64, triangle) * area);
        }
    }
    return weights;
}

const Complex = struct { re: f64, im: f64 };

/// Unnormalized forward radix-2 FFT; scratch and twiddles are reused across frames.
fn fft(values: []Complex, twiddles: []const Complex) void {
    var reversed: usize = 0;
    for (1..values.len) |i| {
        var bit = values.len / 2;
        while (reversed & bit != 0) : (bit /= 2) reversed ^= bit;
        reversed ^= bit;
        if (i < reversed) std.mem.swap(Complex, &values[i], &values[reversed]);
    }
    var width: usize = 2;
    while (true) {
        const half = width / 2;
        const stride = values.len / width;
        var start: usize = 0;
        while (start < values.len) : (start += width) {
            for (0..half) |i| {
                const a = values[start + i];
                const b = values[start + i + half];
                const w = twiddles[i * stride];
                const re = b.re * w.re - b.im * w.im;
                const im = b.re * w.im + b.im * w.re;
                values[start + i] = .{ .re = a.re + re, .im = a.im + im };
                values[start + i + half] = .{ .re = a.re - re, .im = a.im - im };
            }
        }
        if (width == values.len) break;
        width *= 2;
    }
}

const Normalization = enum { none, per_feature, all_features };

fn normalizationMode(name: ?[]const u8) ComputeError!Normalization {
    const value = name orelse return .none;
    if (value.len == 0 or std.mem.eql(u8, value, "NA") or std.mem.eql(u8, value, "None")) return .none;
    if (std.mem.eql(u8, value, "per_feature")) return .per_feature;
    if (std.mem.eql(u8, value, "all_features")) return .all_features;
    return error.UnsupportedFrontendConfig;
}

fn normalize(data: []f32, frames: usize, bins: usize, mode: Normalization) void {
    if (mode == .none or frames == 0) return;
    const groups = if (mode == .per_feature) bins else 1;
    const stride = if (mode == .per_feature) bins else 1;
    const count = if (mode == .per_feature) frames else data.len;
    for (0..groups) |group| {
        const reference: f64 = data[group];
        var sum: f64 = 0;
        for (0..count) |i| sum += @as(f64, data[group + i * stride]) - reference;
        const mean = reference + sum / @as(f64, @floatFromInt(count));
        var variance: f64 = 0;
        for (0..count) |i| {
            const delta = @as(f64, data[group + i * stride]) - mean;
            variance += delta * delta;
        }
        const deviation = (if (count > 1) @sqrt(variance / @as(f64, @floatFromInt(count - 1))) else 0) + 1e-5;
        for (0..count) |i| data[group + i * stride] = @floatCast((@as(f64, data[group + i * stride]) - mean) / deviation);
    }
}
