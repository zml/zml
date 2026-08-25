const std = @import("std");

const config = @import("../core/config.zig");
const packing = @import("packing.zig");
const scheduler = @import("scheduler.zig");

/// PyTorch CPU `Generator` + `randn` (`at::mt19937`, 16-wide Box-Muller).
/// Official H3 draws condition NCHW, then target `(C,T,H,W)`, then audio `(2*T,C)`.
const n = 624;
const m = 397;
const matrix_a: u32 = 0x9908b0df;
const umask: u32 = 0x80000000;
const lmask: u32 = 0x7fffffff;
const two_pi: f32 = @floatCast(2.0 * std.math.pi);

pub const Generator = struct {
    seed: u64,
    left: i32,
    next: u32,
    state: [n]u32,

    pub fn init(seed: u64) Generator {
        var self: Generator = .{
            .seed = seed,
            .left = 1,
            .next = 0,
            .state = undefined,
        };
        self.state[0] = @truncate(seed);
        var j: usize = 1;
        while (j < n) : (j += 1) {
            const prev = self.state[j - 1];
            self.state[j] = 1812433253 *% (prev ^ (prev >> 30)) +% @as(u32, @intCast(j));
        }
        return self;
    }

    pub fn reset(self: *Generator) void {
        self.* = init(self.seed);
    }

    pub fn random(self: *Generator) u32 {
        self.left -= 1;
        if (self.left == 0) self.nextState();
        var y = self.state[self.next];
        self.next += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        return y;
    }

    pub fn uniform01(self: *Generator) f32 {
        const mask: u32 = (1 << 24) - 1;
        const divisor: f32 = 1.0 / 16777216.0;
        return @as(f32, @floatFromInt(self.random() & mask)) * divisor;
    }

    fn mixBits(u: u32, v: u32) u32 {
        return (u & umask) | (v & lmask);
    }

    fn twist(u: u32, v: u32) u32 {
        return (mixBits(u, v) >> 1) ^ if (v & 1 != 0) matrix_a else @as(u32, 0);
    }

    fn nextState(self: *Generator) void {
        self.left = n;
        self.next = 0;
        var p: usize = 0;
        var j: i32 = n - m + 1;
        while (true) {
            j -= 1;
            if (j == 0) break;
            self.state[p] = self.state[p + m] ^ twist(self.state[p], self.state[p + 1]);
            p += 1;
        }
        j = m;
        while (true) {
            j -= 1;
            if (j == 0) break;
            self.state[p] = self.state[p + m - n] ^ twist(self.state[p], self.state[p + 1]);
            p += 1;
        }
        self.state[p] = self.state[p + m - n] ^ twist(self.state[p], self.state[0]);
    }
};

/// Matches ATen `normal_fill` for `numel >= 16`. Smaller tensors use serial Box-Muller.
pub fn randn(gen: *Generator, out: []f32) void {
    if (out.len < 16) {
        randnSerial(gen, out);
        return;
    }
    for (out) |*x| x.* = gen.uniform01();
    var i: usize = 0;
    while (i + 16 <= out.len) : (i += 16) {
        boxMuller16(out[i..][0..16]);
    }
    if (out.len % 16 != 0) {
        const tail = out[out.len - 16 ..];
        for (tail) |*x| x.* = gen.uniform01();
        boxMuller16(tail[0..16]);
    }
}

fn boxMuller16(data: []f32) void {
    var j: usize = 0;
    while (j < 8) : (j += 1) {
        const unit_a = 1.0 - data[j];
        const unit_b = data[j + 8];
        const radius = @sqrt(-2.0 * @log(unit_a));
        const theta = two_pi * unit_b;
        data[j] = radius * @cos(theta);
        data[j + 8] = radius * @sin(theta);
    }
}

fn randnSerial(gen: *Generator, out: []f32) void {
    var cached: ?f32 = null;
    for (out) |*x| {
        if (cached) |c| {
            x.* = c;
            cached = null;
            continue;
        }
        const unit_a = gen.uniform01();
        const unit_b = gen.uniform01();
        const r = @sqrt(-2.0 * @log(1.0 - unit_b));
        const theta = two_pi * unit_a;
        x.* = r * @cos(theta);
        cached = r * @sin(theta);
    }
}

pub fn nchwRandn(allocator: std.mem.Allocator, gen: *Generator, c: u32, t: u32, h: u32, w: u32) ![]f32 {
    const out = try allocator.alloc(f32, @as(usize, c) * t * h * w);
    randn(gen, out);
    return out;
}

pub fn patchifyNchw(
    allocator: std.mem.Allocator,
    nchw: []const f32,
    c: u32,
    t: u32,
    h: u32,
    w: u32,
    patch: [3]i64,
) ![]f32 {
    const thwc = try allocator.alloc(f32, nchw.len);
    defer allocator.free(thwc);
    packing.nchwToThwc(thwc, nchw, c, t, h, w);
    return packing.patchify(allocator, thwc, t, h, w, c, patch);
}

pub fn drawVideo(
    allocator: std.mem.Allocator,
    gen: *Generator,
    videos: []const packing.ConditionVideo,
    clean_patches: []const f32,
    latent_t: u32,
    latent_h: u32,
    latent_w: u32,
    patch: [3]i64,
    reset_before_target: bool,
) ![]f32 {
    const channels: u32 = 24;
    const row_w = @as(usize, channels) * @as(usize, @intCast(patch[0] * patch[1] * patch[2]));
    var cond_len: usize = 0;
    for (videos) |v| {
        cond_len += @as(usize, config.videoTokenCount(v.latent_t, v.latent_h, v.latent_w, patch)) * row_w;
    }
    if (clean_patches.len != cond_len) return error.ConditionPatchSize;

    const target_rows = config.videoTokenCount(latent_t, latent_h, latent_w, patch);
    const out = try allocator.alloc(f32, cond_len + @as(usize, target_rows) * row_w);
    errdefer allocator.free(out);

    var off: usize = 0;
    for (videos) |v| {
        const nchw = try nchwRandn(allocator, gen, channels, v.latent_t, v.latent_h, v.latent_w);
        defer allocator.free(nchw);
        const noise_rows = try patchifyNchw(allocator, nchw, channels, v.latent_t, v.latent_h, v.latent_w, patch);
        defer allocator.free(noise_rows);
        if (off + noise_rows.len > out.len) return error.ConditionPatchSize;
        for (out[off..][0..noise_rows.len], clean_patches[off..][0..noise_rows.len], noise_rows) |*dst, clean, noise| {
            dst.* = scheduler.Schedule.scaleNoise(config.visual_cond_timestep, clean, noise);
        }
        off += noise_rows.len;
    }

    if (reset_before_target) gen.reset();

    const nchw = try nchwRandn(allocator, gen, channels, latent_t, latent_h, latent_w);
    defer allocator.free(nchw);
    const target = try patchifyNchw(allocator, nchw, channels, latent_t, latent_h, latent_w, patch);
    defer allocator.free(target);
    if (off + target.len != out.len) return error.TargetPatchSize;
    @memcpy(out[off..], target);
    return out;
}

pub fn drawAudio(
    allocator: std.mem.Allocator,
    gen: *Generator,
    clean_patches: []const f32,
    channels: u32,
    audio_t: u32,
) ![]f32 {
    const target_n = @as(usize, 2) * audio_t * channels;
    const out = try allocator.alloc(f32, clean_patches.len + target_n);
    errdefer allocator.free(out);
    if (clean_patches.len != 0) @memcpy(out[0..clean_patches.len], clean_patches);
    randn(gen, out[clean_patches.len..]);
    return out;
}
