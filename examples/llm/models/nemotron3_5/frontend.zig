const std = @import("std");

const zml = @import("zml");

const audio = @import("audio.zig");

const PadMode = enum {
    constant,
    reflect,
};

pub const Config = struct {
    sample_rate: u32 = 16_000,
    window_size: ?f64 = 0.025,
    window_stride: ?f64 = 0.01,
    n_window_size: ?usize = null,
    n_window_stride: ?usize = null,
    normalize: ?[]const u8 = null,
    features: usize = 128,
    n_fft: ?usize = 512,
    preemph: ?f32 = 0.97,
    lowfreq: f64 = 0,
    highfreq: ?f64 = null,
    log: bool = true,
    log_zero_guard_type: enum { add, clamp } = .add,
    log_zero_guard_value: f32 = 0x1p-24,
    dither: f32 = 1e-5,
    pad_to: usize = 0,
    pad_value: f32 = 0,
    frame_splicing: usize = 1,
    mag_power: f32 = 2,
    mel_norm: ?[]const u8 = "slaney",
    exact_pad: bool = false,
    snip_edges: bool = false,
    log_base: ?f64 = null,
    pad_mode: PadMode = .constant,
};

pub const Features = struct {
    data: []f32,
    frame_count: usize,
    bin_count: usize,
    valid_frame_count: usize,

    pub fn deinit(self: Features, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }
};

pub const Coefficients = struct {
    window: []const f32,
    fb: []const f32,

    pub fn load(allocator: std.mem.Allocator, io: std.Io, store: zml.io.TensorStore.View, config: Config) !Coefficients {
        const dimensions = try Dimensions.init(config);
        const window_shape = store.getShape("window") orelse return error.MissingFrontendCoefficients;
        const fb_shape = store.getShape("fb") orelse return error.MissingFrontendCoefficients;

        if (window_shape.rank() != 1 or window_shape.dim(0) != dimensions.win_length or fb_shape.rank() != 3 or fb_shape.dim(0) != 1 or fb_shape.dim(1) != config.features or fb_shape.dim(2) != dimensions.fft_size / 2 + 1)
            return error.InvalidFrontendCoefficients;

        const window = try readFloats(allocator, io, store, "window", window_shape);
        errdefer allocator.free(window);
        const fb = try readFloats(allocator, io, store, "fb", fb_shape);
        errdefer allocator.free(fb);

        const result: Coefficients = .{ .window = window, .fb = fb };
        try result.validate(dimensions, config.features);
        return result;
    }

    pub fn deinit(self: Coefficients, allocator: std.mem.Allocator) void {
        allocator.free(self.window);
        allocator.free(self.fb);
    }

    fn readFloats(allocator: std.mem.Allocator, io: std.Io, store: zml.io.TensorStore.View, key: []const u8, shape: zml.Shape) ![]f32 {
        const element_size: usize = switch (shape.dtype()) {
            .bf16, .f16 => 2,
            .f32 => 4,
            else => return error.UnsupportedFrontendCoefficientType,
        };

        var buffer: [8 * 1024]u8 = undefined;
        var reader = try store.getReader(key, io, &buffer);
        defer reader.deinit();
        const bytes = try allocator.alloc(u8, shape.byteSize());
        defer allocator.free(bytes);
        try reader.interface.readSliceAll(bytes);

        const values = try allocator.alloc(f32, bytes.len / element_size);
        for (values, 0..) |*value, i| {
            const encoded = bytes[i * element_size ..];
            value.* = switch (shape.dtype()) {
                .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, encoded[0..2], .little)) << 16),
                .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, encoded[0..2], .little)))),
                .f32 => @bitCast(std.mem.readInt(u32, encoded[0..4], .little)),
                else => unreachable,
            };
        }

        return values;
    }

    fn validate(self: Coefficients, dimensions: Dimensions, features: usize) ComputeError!void {
        const fb_length = std.math.mul(usize, features, dimensions.fft_size / 2 + 1) catch return error.InvalidFrontendCoefficients;
        if (self.window.len != dimensions.win_length or self.fb.len != fb_length)
            return error.InvalidFrontendCoefficients;

        for (self.window) |value| {
            if (!std.math.isFinite(value))
                return error.InvalidFrontendCoefficients;
        }

        for (self.fb) |value| {
            if (!std.math.isFinite(value))
                return error.InvalidFrontendCoefficients;
        }
    }
};

const Dimensions = struct {
    win_length: usize,
    hop_length: usize,
    fft_size: usize,

    fn init(config: Config) ComputeError!Dimensions {
        const win_length = try windowSamples(config.window_size, config.n_window_size, config.sample_rate);
        const hop_length = try windowSamples(config.window_stride, config.n_window_stride, config.sample_rate);
        const fft_size = config.n_fft orelse (std.math.ceilPowerOfTwo(usize, win_length) catch return error.InvalidFrontendConfig);

        if (fft_size < 2 or !std.math.isPowerOfTwo(fft_size) or fft_size < win_length or config.features == 0)
            return error.InvalidFrontendConfig;

        return .{
            .win_length = win_length,
            .hop_length = hop_length,
            .fft_size = fft_size,
        };
    }
};

const Complex = struct {
    re: f64,
    im: f64,
};

const Normalization = enum {
    none,
    per_feature,
    all_features,
};

pub const ComputeError = std.mem.Allocator.Error || error{
    AudioTooShortForReflectPadding,
    EmptyAudio,
    InvalidAudioFormat,
    InvalidFrontendCoefficients,
    InvalidFrontendConfig,
    MathFault,
    NonFiniteAudioSample,
    NonFiniteFeatures,
    UnsupportedFrontendConfig,
};

pub fn compute(allocator: std.mem.Allocator, prepared: audio.DecodedAudio, config: Config, coefficients: Coefficients) ComputeError!Features {
    @setFloatMode(.strict);

    if (prepared.channels != 1 or prepared.sample_rate != config.sample_rate or config.sample_rate != audio.target_sample_rate)
        return error.InvalidAudioFormat;

    if (prepared.samples.len == 0)
        return error.EmptyAudio;

    for (prepared.samples) |sample| {
        if (!std.math.isFinite(sample))
            return error.NonFiniteAudioSample;
    }

    const dimensions = try Dimensions.init(config);
    const win_length = dimensions.win_length;
    const hop_length = dimensions.hop_length;
    const fft_size = dimensions.fft_size;
    try coefficients.validate(dimensions, config.features);

    // https://en.wikipedia.org/wiki/Nyquist_frequency
    const nyquist = @as(f64, @floatFromInt(config.sample_rate)) / 2;

    // NeMo treats both none and zero highfreq as Nyquist
    const highfreq = if (config.highfreq == null or config.highfreq.? == 0) nyquist else config.highfreq.?;
    if (!std.math.isFinite(config.lowfreq) or !std.math.isFinite(highfreq) or config.lowfreq < 0 or highfreq <= config.lowfreq or highfreq > nyquist)
        return error.InvalidFrontendConfig;

    if (!std.math.isFinite(config.mag_power) or config.mag_power <= 0 or !std.math.isFinite(config.pad_value))
        return error.InvalidFrontendConfig;

    if (!std.math.isFinite(config.log_zero_guard_value) or config.log_zero_guard_value <= 0)
        return error.InvalidFrontendConfig;

    if (!std.math.isFinite(config.dither) or config.dither < 0)
        return error.InvalidFrontendConfig;

    if (config.preemph) |coefficient| {
        if (!std.math.isFinite(coefficient))
            return error.InvalidFrontendConfig;
    }

    if (config.log_base) |base| {
        if (!std.math.isFinite(base) or base <= 0 or base == 1)
            return error.InvalidFrontendConfig;
    }

    if (config.exact_pad or config.snip_edges or config.frame_splicing != 1)
        return error.UnsupportedFrontendConfig;

    if (config.mel_norm) |norm| {
        if (!std.mem.eql(u8, norm, "slaney"))
            return error.UnsupportedFrontendConfig;
    }

    const normalization = try normalizationMode(config.normalize);
    if (config.pad_mode == .reflect and prepared.samples.len <= fft_size / 2)
        return error.AudioTooShortForReflectPadding;

    // Centered even-sized STFT produces floor(N / hop) + 1 frames but NeMo omits the last one and forward masks it to pad_value
    const valid_frames = prepared.samples.len / hop_length;
    var frame_count = std.math.add(usize, valid_frames, 1) catch return error.MathFault;
    if (config.pad_to > 0 and frame_count % config.pad_to != 0) {
        frame_count = std.math.add(usize, frame_count, config.pad_to - frame_count % config.pad_to) catch return error.MathFault;
    }

    const data = try allocator.alloc(f32, std.math.mul(usize, frame_count, config.features) catch return error.MathFault);
    errdefer allocator.free(data);
    @memset(data, config.pad_value);

    const emphasized = try allocator.alloc(f32, prepared.samples.len);
    defer allocator.free(emphasized);
    emphasized[0] = prepared.samples[0];
    for (emphasized[1..], 1..) |*sample, i| {
        sample.* = if (config.preemph) |coefficient|
            prepared.samples[i] - coefficient * prepared.samples[i - 1]
        else
            prepared.samples[i];

        if (!std.math.isFinite(sample.*))
            return error.NonFiniteFeatures;
    }

    const window = coefficients.window;
    const filterbank = coefficients.fb;
    const spectrum = try allocator.alloc(Complex, fft_size);
    defer allocator.free(spectrum);
    const twiddles = try allocator.alloc(Complex, fft_size / 2);
    defer allocator.free(twiddles);

    for (twiddles, 0..) |*twiddle, i| {
        const phase = -2 * std.math.pi * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(fft_size));
        twiddle.* = .{ .re = @cos(phase), .im = @sin(phase) };
    }

    const fft_bins = fft_size / 2 + 1;
    const powers = try allocator.alloc(f32, fft_bins);
    defer allocator.free(powers);
    const window_offset = (fft_size - win_length) / 2;
    const log_scale: f32 = if (config.log_base) |base| @floatCast(1 / @log(base)) else 1;

    for (0..valid_frames) |frame| {
        @memset(spectrum, .{ .re = 0, .im = 0 });
        const center = frame * hop_length;

        for (window, 0..) |weight, i| {
            const position = @as(i128, @intCast(center)) + @as(i128, @intCast(window_offset + i)) - @as(i128, @intCast(fft_size / 2));
            const sample = centeredSample(emphasized, position, config.pad_mode);
            spectrum[window_offset + i].re = sample * weight;
        }

        fft(spectrum, twiddles);

        for (powers, spectrum[0..fft_bins]) |*power, bin| {
            // Round the FFT before NeMo's operations
            const re: f32 = @floatCast(bin.re);
            const im: f32 = @floatCast(bin.im);
            const magnitude = @sqrt(re * re + im * im);

            power.* = if (config.mag_power == 1)
                magnitude
            else if (config.mag_power == 2)
                magnitude * magnitude
            else
                std.math.pow(f32, magnitude, config.mag_power);

            if (!std.math.isFinite(power.*))
                return error.NonFiniteFeatures;
        }

        const output = data[frame * config.features ..][0..config.features];
        for (output, 0..) |*value, mel| {
            var energy: f32 = 0;
            for (filterbank[mel * fft_bins ..][0..fft_bins], powers) |weight, power|
                energy += weight * power;

            value.* = if (config.log) @log(switch (config.log_zero_guard_type) {
                .add => energy + config.log_zero_guard_value,
                .clamp => @max(energy, config.log_zero_guard_value),
            }) * log_scale else energy;

            if (!std.math.isFinite(value.*))
                return error.NonFiniteFeatures;
        }
    }

    normalize(data[0 .. valid_frames * config.features], valid_frames, config.features, normalization);

    for (data) |value| {
        if (!std.math.isFinite(value))
            return error.NonFiniteFeatures;
    }

    return .{
        .data = data,
        .frame_count = frame_count,
        .bin_count = config.features,
        .valid_frame_count = valid_frames,
    };
}

fn windowSamples(seconds: ?f64, samples: ?usize, rate: u32) ComputeError!usize {
    if (seconds) |duration| {
        if (samples != null or !std.math.isFinite(duration) or duration <= 0)
            return error.InvalidFrontendConfig;

        const count = duration * @as(f64, @floatFromInt(rate));
        if (!std.math.isFinite(count) or count < 1 or count >= @as(f64, @floatFromInt(std.math.maxInt(usize))))
            return error.InvalidFrontendConfig;

        return @intFromFloat(count); // Python int(window_size * sample_rate) truncates.
    }

    const count = samples orelse return error.InvalidFrontendConfig;
    if (count == 0)
        return error.InvalidFrontendConfig;

    return count;
}

fn centeredSample(samples: []const f32, position: i128, mode: PadMode) f32 {
    const length: i128 = @intCast(samples.len);

    if (position >= 0 and position < length)
        return samples[@intCast(position)];

    if (mode == .constant)
        return 0;

    const reflected = if (position < 0) -position else 2 * length - 2 - position;
    return samples[@intCast(reflected)];
}

/// Unnormalized forward radix-2 FFT
fn fft(values: []Complex, twiddles: []const Complex) void {
    var reversed: usize = 0;
    for (1..values.len) |i| {
        var bit = values.len / 2;
        while (reversed & bit != 0) : (bit /= 2)
            reversed ^= bit;

        reversed ^= bit;
        if (i < reversed)
            std.mem.swap(Complex, &values[i], &values[reversed]);
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

        if (width == values.len)
            break;

        width *= 2;
    }
}

fn normalizationMode(name: ?[]const u8) ComputeError!Normalization {
    const value = name orelse return .none;
    if (value.len == 0 or std.mem.eql(u8, value, "NA") or std.mem.eql(u8, value, "None"))
        return .none;

    if (std.mem.eql(u8, value, "per_feature"))
        return .per_feature;

    if (std.mem.eql(u8, value, "all_features"))
        return .all_features;

    return error.UnsupportedFrontendConfig;
}

fn normalize(data: []f32, frames: usize, bins: usize, mode: Normalization) void {
    if (mode == .none or frames == 0)
        return;

    const groups = if (mode == .per_feature) bins else 1;
    const stride = if (mode == .per_feature) bins else 1;
    const count = if (mode == .per_feature) frames else data.len;

    for (0..groups) |group| {
        const reference: f64 = data[group];
        var sum: f64 = 0;
        for (0..count) |i|
            sum += @as(f64, data[group + i * stride]) - reference;

        const mean = reference + sum / @as(f64, @floatFromInt(count));
        var variance: f64 = 0;
        for (0..count) |i| {
            const delta = @as(f64, data[group + i * stride]) - mean;
            variance += delta * delta;
        }

        const deviation = (if (count > 1) @sqrt(variance / @as(f64, @floatFromInt(count - 1))) else 0) + 1e-5;
        for (0..count) |i|
            data[group + i * stride] = @floatCast((@as(f64, data[group + i * stride]) - mean) / deviation);
    }
}
