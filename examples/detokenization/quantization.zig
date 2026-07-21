const std = @import("std");
const zml = @import("zml");
const builtin = @import("builtin");
const main = @import("main.zig");
const model_ = @import("model.zig");
const llm_ = @import("llm.zig");
const tokens = @import("tokens.zig");
const algebra = @import("algebra.zig");

const Zml_handler = main.Zml_handler;
const Tokenizer = zml.tokenizer.Tokenizer;
const LmHeadMatrix = algebra.LmHeadMatrix;

const simd_len_f32: comptime_int = 16;
const simd_len_i8: comptime_int = 16;
pub const hidden_dim_log2: comptime_int = 12;

pub const hidden_dim: comptime_int = 4096;
pub const inv_hidden_dim: comptime_float = 1.0 / @as(f32, @floatFromInt(hidden_dim));
pub const std_dev: comptime_float = @sqrt(inv_hidden_dim);

// applies the normalized in-place fast Walsh-Hadamard transform
// the vector needs to be of dimension 2^k
// the inverse transform is the transform itself (f o f = Id)
// TODO: Rademacher randomisation
pub fn walshHadamard(v: []f32, comptime k: comptime_int) void {
    const n: comptime_int = 1 << k;

    inline for (0..k) |stage| {
        const h: usize = 1 << stage;
        const step: usize = 2 * h;
        const simd_len: comptime_int = if ((1 << stage) < simd_len_f32) (1 << stage) else simd_len_f32;
        const Vec = @Vector(simd_len, f32);

        for (0..(n / step)) |block| {
            const i: usize = block * step;
            var offset: usize = 0;
            while (offset < h) : (offset += simd_len) {
                const left = i + offset;
                const right = left + h;
                const x: Vec = v[left..][0..simd_len].*;
                const y: Vec = v[right..][0..simd_len].*;
                v[left..][0..simd_len].* = x + y;
                v[right..][0..simd_len].* = x - y;
            }
        }
    }

    const Vec = @Vector(simd_len_f32, f32);
    const scale_scalar: comptime_float = 1.0 / @sqrt(@as(f32, @floatFromInt(n)));
    const scale: Vec = @splat(scale_scalar);
    var i: usize = 0;
    while (i + simd_len_f32 <= n) : (i += simd_len_f32) {
        const values: Vec = v[i..][0..simd_len_f32].*;
        v[i..][0..simd_len_f32].* = values * scale;
    }
}

pub inline fn normL2(v: []const f32) f32 {
    var res: f32 = 0.0;
    for (v) |val| {
        res += val * val;
    }
    return @sqrt(res);
}

const useNeonSdot = builtin.cpu.arch == .aarch64 and
    builtin.zig_backend != .stage2_c and
    (builtin.os.tag == .macos or builtin.cpu.has(.aarch64, .dotprod));

const Vec4i32 = @Vector(4, i32);
const Vec16i8 = @Vector(16, i8);
const Vec16i32 = @Vector(16, i32);
const int4_block_len: comptime_int = 32;
const int4_packed_block_len: comptime_int = int4_block_len / 2;

inline fn neonSdot(acc: Vec4i32, a: Vec16i8, b: Vec16i8) Vec4i32 {
    var result = acc;
    asm volatile (
        \\ sdot %[result].4s, %[a].16b, %[b].16b
        : [result] "+w" (result),
        : [a] "w" (a),
          [b] "w" (b),
    );
    return result;
}

pub const QuantizationInt8 = struct {
    allocator: std.mem.Allocator,
    lm_head: *LmHeadMatrix,
    d: usize,
    vocab_size: usize,
    lm_head_quantized: []i8,
    row_quant_scale: []f32,
    buffer: []f32,

    pub fn init(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix) !QuantizationInt8 {
        const v: usize = lm_head.n;
        const d: usize = lm_head.d;
        const lm_head_quantized = try zml_handler.allocator.alloc(i8, v * d);
        const row_quant_scale = try zml_handler.allocator.alloc(f32, v);
        const buffer = try zml_handler.allocator.alloc(f32, d);
        return .{
            .allocator = zml_handler.allocator,
            .lm_head = lm_head,
            .d = d,
            .vocab_size = v,
            .lm_head_quantized = lm_head_quantized,
            .row_quant_scale = row_quant_scale,
            .buffer = buffer,
        };
    }

    pub fn deinit(self: *QuantizationInt8) void {
        self.allocator.free(self.lm_head_quantized);
        self.allocator.free(self.row_quant_scale);
        self.allocator.free(self.buffer);
    }

    pub fn quantize(self: *QuantizationInt8) !void {
        std.debug.assert(self.d == hidden_dim);
        for (0..self.vocab_size) |row| {
            const src = self.lm_head.data[row * self.d ..][0..self.d];
            const dst = self.lm_head_quantized[row * self.d ..][0..self.d];
            self.row_quant_scale[row] = quantizeVector(src, self.buffer, dst, self.lm_head.row_norms[row]);
        }
    }

    pub fn quantizeVector(src: []const f32, buff: []f32, dst: []i8, src_norm: f32) f32 {
        @memcpy(buff, src);
        walshHadamard(buff, hidden_dim_log2);
        for (0..hidden_dim) |coord| {
            buff[coord] /= src_norm;
        }
        // buffer is the rotated vector with unit norm:
        // its coefficients follow the gaussian distribution N(0, std_dev), std = 1/sqrt(d)
        var max_dev: f32 = 0.0;
        for (0..hidden_dim) |coord| {
            max_dev = @max(max_dev, @abs(buff[coord]) / std_dev);
        }
        //std.log.info("max_dev {d}", .{row, max_dev});
        const max_abs = max_dev * std_dev;
        var quant_norm2: f32 = 0.0;
        for (0..hidden_dim) |coord| {
            const v_scaled: f32 = 127.0 * buff[coord] / max_abs;
            const v_clamped: f32 = std.math.clamp(v_scaled, -127.0, 127.0);
            dst[coord] = @intFromFloat(@round(v_clamped));
            const v_dequant: f32 = @floatFromInt(dst[coord]);
            quant_norm2 += v_dequant * v_dequant;
        }
        return src_norm / @sqrt(quant_norm2);
    }

    inline fn int8DotProduct(a: []const i8, b: []const i8) i32 {
        if (comptime useNeonSdot) {
            const unrollLen = 4 * simd_len_i8;
            std.debug.assert(a.len == b.len and a.len >= unrollLen and a.len % unrollLen == 0);

            var sum0: Vec4i32 = @splat(0);
            var sum1: Vec4i32 = @splat(0);
            var sum2: Vec4i32 = @splat(0);
            var sum3: Vec4i32 = @splat(0);
            var i: usize = 0;
            while (i < a.len) : (i += unrollLen) {
                const a0: Vec16i8 = a[i..][0..simd_len_i8].*;
                const a1: Vec16i8 = a[i + simd_len_i8 ..][0..simd_len_i8].*;
                const a2: Vec16i8 = a[i + 2 * simd_len_i8 ..][0..simd_len_i8].*;
                const a3: Vec16i8 = a[i + 3 * simd_len_i8 ..][0..simd_len_i8].*;
                const b0: Vec16i8 = b[i..][0..simd_len_i8].*;
                const b1: Vec16i8 = b[i + simd_len_i8 ..][0..simd_len_i8].*;
                const b2: Vec16i8 = b[i + 2 * simd_len_i8 ..][0..simd_len_i8].*;
                const b3: Vec16i8 = b[i + 3 * simd_len_i8 ..][0..simd_len_i8].*;
                sum0 = neonSdot(sum0, a0, b0);
                sum1 = neonSdot(sum1, a1, b1);
                sum2 = neonSdot(sum2, a2, b2);
                sum3 = neonSdot(sum3, a3, b3);
            }
            return @reduce(.Add, sum0 + sum1 + sum2 + sum3);
        }

        var acc: Vec16i32 = @splat(0);
        var i: usize = 0;
        while (i + simd_len_i8 <= a.len) : (i += simd_len_i8) {
            const a_vec: Vec16i8 = a[i..][0..simd_len_i8].*;
            const b_vec: Vec16i8 = b[i..][0..simd_len_i8].*;
            const a_i32: Vec16i32 = @intCast(a_vec);
            const b_i32: Vec16i32 = @intCast(b_vec);
            acc += a_i32 * b_i32;
        }
        return @reduce(.Add, acc);
    }
};

pub const QuantizationInt4 = struct {
    allocator: std.mem.Allocator,
    lm_head: *LmHeadMatrix,
    d: usize,
    vocab_size: usize,
    lm_head_quantized: []i8, // stored as 2 packed elems per i8
    row_quant_scale: []f32,
    buffer: []f32,

    pub fn init(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix) !QuantizationInt4 {
        const v: usize = lm_head.n;
        const d: usize = lm_head.d;
        const lm_head_quantized = try zml_handler.allocator.alloc(i8, v * d / 2);
        const row_quant_scale = try zml_handler.allocator.alloc(f32, v);
        const buffer = try zml_handler.allocator.alloc(f32, d);
        return .{
            .allocator = zml_handler.allocator,
            .lm_head = lm_head,
            .d = d,
            .vocab_size = v,
            .lm_head_quantized = lm_head_quantized,
            .row_quant_scale = row_quant_scale,
            .buffer = buffer,
        };
    }

    pub fn deinit(self: *QuantizationInt4) void {
        self.allocator.free(self.lm_head_quantized);
        self.allocator.free(self.row_quant_scale);
        self.allocator.free(self.buffer);
    }

    pub fn quantize(self: *QuantizationInt4) !void {
        std.debug.assert(self.d == hidden_dim);
        std.debug.assert(self.d % int4_block_len == 0);
        const packed_d = self.d / 2;
        for (0..self.vocab_size) |row| {
            const src = self.lm_head.data[row * self.d ..][0..self.d];
            const dst = self.lm_head_quantized[row * packed_d ..][0..packed_d];
            self.row_quant_scale[row] = quantizeVector(src, self.buffer, dst, self.lm_head.row_norms[row]);
        }
    }

    pub fn quantizeVector(src: []const f32, buff: []f32, dst: []i8, src_norm: f32) f32 {
        std.debug.assert(dst.len == src.len / 2);
        @memcpy(buff, src);
        walshHadamard(buff, hidden_dim_log2);
        for (0..hidden_dim) |coord| {
            buff[coord] /= src_norm;
        }
        // buffer is the rotated vector with unit norm:
        // its coefficients follow the gaussian distribution N(0, std_dev), std = 1/sqrt(d)
        var max_dev: f32 = 0.0;
        for (0..hidden_dim) |coord| {
            max_dev = @max(max_dev, @abs(buff[coord]) / std_dev);
        }
        //std.log.info("max_dev {d}", .{row, max_dev});
        const max_abs = max_dev * std_dev;
        var quant_norm2: f32 = 0.0;
        for (0..hidden_dim / int4_block_len) |block| {
            const coord_base = block * int4_block_len;
            const packed_base = block * int4_packed_block_len;
            for (0..int4_packed_block_len) |lane| {
                const coord1 = coord_base + lane;
                const coord2 = coord1 + int4_packed_block_len;
                const v_scaled1: f32 = 7.0 * buff[coord1] / max_abs;
                const v_scaled2: f32 = 7.0 * buff[coord2] / max_abs;
                const v_round1: f32 = @round(std.math.clamp(v_scaled1, -7.0, 7.0));
                const v_round2: f32 = @round(std.math.clamp(v_scaled2, -7.0, 7.0));
                const v_quant1: i8 = @intFromFloat(v_round1);
                const v_quant2: i8 = @intFromFloat(v_round2);
                const bits1: u8 = @bitCast(v_quant1);
                const bits2: u8 = @bitCast(v_quant2);
                dst[packed_base + lane] = @bitCast((bits1 << 4) | (bits2 & 0x0F));
                quant_norm2 += v_round1 * v_round1 + v_round2 * v_round2;
            }
        }
        return src_norm / @sqrt(quant_norm2);
    }

    const Int8x4DotAccumulators = struct {
        sum0: Vec4i32,
        sum1: Vec4i32,
        sum2: Vec4i32,
        sum3: Vec4i32,
    };

    const Int4DotAccumulators = struct {
        sum0: Vec4i32,
        sum1: Vec4i32,
        sum2: Vec4i32,
        sum3: Vec4i32,
    };

    inline fn signExtendInt4(nibble: u8) i8 {
        const shifted: i8 = @bitCast(nibble << 4);
        return shifted >> 4;
    }

    /// Unpacks two packed 32-coordinate blocks and accumulates their four SDOTs in
    /// one assembly block, keeping all intermediate vectors in SIMD registers.
    inline fn neonSdotInt8x4x64(
        sum0: Vec4i32,
        sum1: Vec4i32,
        sum2: Vec4i32,
        sum3: Vec4i32,
        a0: Vec16i8,
        a1: Vec16i8,
        a2: Vec16i8,
        a3: Vec16i8,
        packed0: Vec16i8,
        packed1: Vec16i8,
    ) Int8x4DotAccumulators {
        var result0 = sum0;
        var result1 = sum1;
        var result2 = sum2;
        var result3 = sum3;
        var high: Vec16i8 = undefined;
        var low: Vec16i8 = undefined;
        asm volatile (
            \\ sshr %[high].16b, %[packed0].16b, #4
            \\ shl %[low].16b, %[packed0].16b, #4
            \\ sshr %[low].16b, %[low].16b, #4
            \\ sdot %[result0].4s, %[a0].16b, %[high].16b
            \\ sdot %[result1].4s, %[a1].16b, %[low].16b
            \\ sshr %[high].16b, %[packed1].16b, #4
            \\ shl %[low].16b, %[packed1].16b, #4
            \\ sshr %[low].16b, %[low].16b, #4
            \\ sdot %[result2].4s, %[a2].16b, %[high].16b
            \\ sdot %[result3].4s, %[a3].16b, %[low].16b
            : [result0] "+&w" (result0),
              [result1] "+&w" (result1),
              [result2] "+&w" (result2),
              [result3] "+&w" (result3),
              [high] "=&w" (high),
              [low] "=&w" (low),
            : [a0] "w" (a0),
              [a1] "w" (a1),
              [a2] "w" (a2),
              [a3] "w" (a3),
              [packed0] "w" (packed0),
              [packed1] "w" (packed1),
        );
        return .{ .sum0 = result0, .sum1 = result1, .sum2 = result2, .sum3 = result3 };
    }

    /// Computes a dot product between a full int8 vector and an int4 vector packed
    /// two coordinates per byte.
    inline fn int8x4DotProduct(a: []const i8, b: []const i8) i32 {
        std.debug.assert(a.len == b.len * 2);
        std.debug.assert(a.len >= 64 and a.len % 64 == 0);

        if (comptime useNeonSdot) {
            var sum0: Vec4i32 = @splat(0);
            var sum1: Vec4i32 = @splat(0);
            var sum2: Vec4i32 = @splat(0);
            var sum3: Vec4i32 = @splat(0);

            var i: usize = 0;
            while (i < a.len) : (i += 64) {
                const a0: Vec16i8 = a[i..][0..16].*;
                const a1: Vec16i8 = a[i + 16 ..][0..16].*;
                const a2: Vec16i8 = a[i + 32 ..][0..16].*;
                const a3: Vec16i8 = a[i + 48 ..][0..16].*;

                const packed0: Vec16i8 = b[i / 2 ..][0..int4_packed_block_len].*;
                const packed1: Vec16i8 = b[i / 2 + int4_packed_block_len ..][0..int4_packed_block_len].*;
                const sums = neonSdotInt8x4x64(sum0, sum1, sum2, sum3, a0, a1, a2, a3, packed0, packed1);
                sum0 = sums.sum0;
                sum1 = sums.sum1;
                sum2 = sums.sum2;
                sum3 = sums.sum3;
            }
            return @reduce(.Add, sum0 + sum1 + sum2 + sum3);
        }

        var sum: i32 = 0;
        for (0..a.len / int4_block_len) |block| {
            const coord_base = block * int4_block_len;
            const packed_base = block * int4_packed_block_len;
            for (0..int4_packed_block_len) |lane| {
                const byte: u8 = @bitCast(b[packed_base + lane]);
                const high: i32 = signExtendInt4(byte >> 4);
                const low: i32 = signExtendInt4(byte & 0x0f);
                sum += @as(i32, a[coord_base + lane]) * high;
                sum += @as(i32, a[coord_base + int4_packed_block_len + lane]) * low;
            }
        }
        return sum;
    }

    /// Computes four SDOTs from two packed int4 query blocks and two packed int4
    /// weight blocks without materializing either operand as int8 in memory.
    inline fn neonSdotInt4x4x64(
        sum0: Vec4i32,
        sum1: Vec4i32,
        sum2: Vec4i32,
        sum3: Vec4i32,
        packedA0: Vec16i8,
        packedA1: Vec16i8,
        packedB0: Vec16i8,
        packedB1: Vec16i8,
    ) Int4DotAccumulators {
        var result0 = sum0;
        var result1 = sum1;
        var result2 = sum2;
        var result3 = sum3;
        var aHigh: Vec16i8 = undefined;
        var aLow: Vec16i8 = undefined;
        var bHigh: Vec16i8 = undefined;
        var bLow: Vec16i8 = undefined;
        asm volatile (
            \\ sshr %[a_high].16b, %[packed_a0].16b, #4
            \\ shl %[a_low].16b, %[packed_a0].16b, #4
            \\ sshr %[a_low].16b, %[a_low].16b, #4
            \\ sshr %[b_high].16b, %[packed_b0].16b, #4
            \\ shl %[b_low].16b, %[packed_b0].16b, #4
            \\ sshr %[b_low].16b, %[b_low].16b, #4
            \\ sdot %[result0].4s, %[a_high].16b, %[b_high].16b
            \\ sdot %[result1].4s, %[a_low].16b, %[b_low].16b
            \\ sshr %[a_high].16b, %[packed_a1].16b, #4
            \\ shl %[a_low].16b, %[packed_a1].16b, #4
            \\ sshr %[a_low].16b, %[a_low].16b, #4
            \\ sshr %[b_high].16b, %[packed_b1].16b, #4
            \\ shl %[b_low].16b, %[packed_b1].16b, #4
            \\ sshr %[b_low].16b, %[b_low].16b, #4
            \\ sdot %[result2].4s, %[a_high].16b, %[b_high].16b
            \\ sdot %[result3].4s, %[a_low].16b, %[b_low].16b
            : [result0] "+&w" (result0),
              [result1] "+&w" (result1),
              [result2] "+&w" (result2),
              [result3] "+&w" (result3),
              [a_high] "=&w" (aHigh),
              [a_low] "=&w" (aLow),
              [b_high] "=&w" (bHigh),
              [b_low] "=&w" (bLow),
            : [packed_a0] "w" (packedA0),
              [packed_a1] "w" (packedA1),
              [packed_b0] "w" (packedB0),
              [packed_b1] "w" (packedB1),
        );
        return .{ .sum0 = result0, .sum1 = result1, .sum2 = result2, .sum3 = result3 };
    }

    /// Computes a dot product between two vectors using the shared packed int4
    /// nibble-plane layout.
    inline fn int4DotProduct(a: []const i8, b: []const i8) i32 {
        std.debug.assert(a.len == b.len);
        std.debug.assert(a.len >= 2 * int4_packed_block_len and a.len % (2 * int4_packed_block_len) == 0);

        if (comptime useNeonSdot) {
            var sum0: Vec4i32 = @splat(0);
            var sum1: Vec4i32 = @splat(0);
            var sum2: Vec4i32 = @splat(0);
            var sum3: Vec4i32 = @splat(0);

            var i: usize = 0;
            while (i < a.len) : (i += 2 * int4_packed_block_len) {
                const packedA0: Vec16i8 = a[i..][0..int4_packed_block_len].*;
                const packedA1: Vec16i8 = a[i + int4_packed_block_len ..][0..int4_packed_block_len].*;
                const packedB0: Vec16i8 = b[i..][0..int4_packed_block_len].*;
                const packedB1: Vec16i8 = b[i + int4_packed_block_len ..][0..int4_packed_block_len].*;
                const sums = neonSdotInt4x4x64(sum0, sum1, sum2, sum3, packedA0, packedA1, packedB0, packedB1);
                sum0 = sums.sum0;
                sum1 = sums.sum1;
                sum2 = sums.sum2;
                sum3 = sums.sum3;
            }
            return @reduce(.Add, sum0 + sum1 + sum2 + sum3);
        }

        var sum: i32 = 0;
        for (a, b) |packedA, packedB| {
            const aByte: u8 = @bitCast(packedA);
            const bByte: u8 = @bitCast(packedB);
            const aHigh: i32 = signExtendInt4(aByte >> 4);
            const aLow: i32 = signExtendInt4(aByte & 0x0f);
            const bHigh: i32 = signExtendInt4(bByte >> 4);
            const bLow: i32 = signExtendInt4(bByte & 0x0f);
            sum += aHigh * bHigh + aLow * bLow;
        }
        return sum;
    }
};

pub const qjl_word = u64;
pub const qjl_word_length: comptime_int = @bitSizeOf(qjl_word);
pub const qjl_nb_words: comptime_int = hidden_dim / qjl_word_length;

pub const VectorQJL1 = [qjl_nb_words]qjl_word;
pub fn makeVectorQJL1() VectorQJL1 {
    return [_]qjl_word{0} ** qjl_nb_words;
}

pub inline fn setBit(v: *VectorQJL1, pos: usize) void {
    const shift: std.math.Log2Int(qjl_word) = @intCast(pos % qjl_word_length);
    v[pos / qjl_word_length] |= @as(qjl_word, 1) << shift;
}

pub const QuantizationQJL1 = struct {
    allocator: std.mem.Allocator,
    lm_head: *LmHeadMatrix,
    d: usize,
    vocab_size: usize,
    lm_head_quantized: []VectorQJL1,
    row_quant_scale: []f32,
    buffer: []f32,

    pub fn init(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix) !QuantizationQJL1 {
        const v: usize = lm_head.n;
        const d: usize = lm_head.d;
        const lm_head_quantized = try zml_handler.allocator.alloc(VectorQJL1, v);
        const row_quant_scale = try zml_handler.allocator.alloc(f32, v);
        const buffer = try zml_handler.allocator.alloc(f32, d);
        return .{
            .allocator = zml_handler.allocator,
            .lm_head = lm_head,
            .d = d,
            .vocab_size = v,
            .lm_head_quantized = lm_head_quantized,
            .row_quant_scale = row_quant_scale,
            .buffer = buffer,
        };
    }

    pub fn deinit(self: *QuantizationQJL1) void {
        self.allocator.free(self.lm_head_quantized);
        self.allocator.free(self.row_quant_scale);
        self.allocator.free(self.buffer);
    }

    pub fn quantize(self: *QuantizationQJL1) !void {
        for (0..self.vocab_size) |row| {
            const src = self.lm_head.data[row * self.d ..][0..self.d];
            const dst = &self.lm_head_quantized[row];
            self.row_quant_scale[row] = quantizeVector(src, self.buffer, dst, self.lm_head.row_norms[row]);
        }
    }

    pub fn quantizeVector(src: []const f32, buff: []f32, dst: *VectorQJL1, src_norm: f32) f32 {
        @memset(dst, 0);
        @memcpy(buff, src);
        walshHadamard(buff, hidden_dim_log2);
        var l1_norm: f32 = 0.0;
        for (0..hidden_dim) |coord| {
            buff[coord] /= src_norm;
            l1_norm = @max(l1_norm, @abs(buff[coord]));
        }
        for (0..hidden_dim) |coord| {
            if (buff[coord] > 0) setBit(dst, coord);
        }
        return l1_norm / @as(f32, @floatFromInt(hidden_dim));
    }

    pub inline fn qjl1dot(a: *const VectorQJL1, b: *const VectorQJL1) f32 {
        // dot(u,v) approx = ||u|| * ||v|| * sin(pi/2 * dot(q(u), q(v)) / d)
        // dot(q(u), q(v)) = d - 2 * bit_mismatches,
        // XOR returns 1 where bits differ, 0 where they match
        // this returns sin(pi/2 * dot(q(u), q(v)) / d)
        // we use a LUT for f(x) = sin(pi / 2 * (d - 2 * x) / d),
        // for x integer in [-d, d], as sin is about as expensive as
        // counting the mismatches
        const mismatches = popcountXor(a, b);
        return qjl_dot_lut[mismatches];
    }

    fn makeQjlDotLut(comptime coord_count: usize) [coord_count + 1]f32 {
        @setEvalBranchQuota(8192);
        var lut: [coord_count + 1]f32 = undefined;
        for (0..(coord_count + 1)) |mismatch_count| {
            const dot: i32 = @as(i32, @intCast(coord_count)) - 2 * @as(i32, @intCast(mismatch_count));
            const w = @as(f32, @floatFromInt(dot)) / @as(f32, @floatFromInt(coord_count));
            lut[mismatch_count] = @sin(@as(f32, @floatCast(0.5 * std.math.pi)) * w);
        }
        return lut;
    }

    const qjl_dot_lut = makeQjlDotLut(hidden_dim);

    inline fn popcountXor(a: *const VectorQJL1, b: *const VectorQJL1) u32 {
        // Cast the arrays directly to raw byte pointers
        const ptr_a = @as([*]const u8, @ptrCast(a));
        const ptr_b = @as([*]const u8, @ptrCast(b));

        // Define a 128-bit vector type
        const VecType = @Vector(16, u8);
        const vec_a = @as([*]const VecType, @ptrCast(@alignCast(ptr_a)));
        const vec_b = @as([*]const VecType, @ptrCast(@alignCast(ptr_b)));

        // 4 independent 16-bit accumulators to prevent overflow
        var sum0: @Vector(16, u16) = @splat(0);
        var sum1: @Vector(16, u16) = @splat(0);
        var sum2: @Vector(16, u16) = @splat(0);
        var sum3: @Vector(16, u16) = @splat(0);

        var i: usize = 0;
        var vec_idx: usize = 0;

        // Process 512 bits per iteration
        while (i + 64 <= @sizeOf(VectorQJL1)) {

            // Inner 8-bit accumulators
            var inner_sum0: VecType = @splat(0);
            var inner_sum1: VecType = @splat(0);
            var inner_sum2: VecType = @splat(0);
            var inner_sum3: VecType = @splat(0);

            var inner_iters: usize = 0;

            // Accumulate up to 30 times safely without 8-bit overflow (30 * 8 = 240 < 255)
            while (inner_iters < 30 and i + 64 <= @sizeOf(VectorQJL1)) {
                const a0 = vec_a[vec_idx + 0];
                const a1 = vec_a[vec_idx + 1];
                const a2 = vec_a[vec_idx + 2];
                const a3 = vec_a[vec_idx + 3];

                const b0 = vec_b[vec_idx + 0];
                const b1 = vec_b[vec_idx + 1];
                const b2 = vec_b[vec_idx + 2];
                const b3 = vec_b[vec_idx + 3];

                inner_sum0 += @popCount(a0 ^ b0);
                inner_sum1 += @popCount(a1 ^ b1);
                inner_sum2 += @popCount(a2 ^ b2);
                inner_sum3 += @popCount(a3 ^ b3);

                inner_iters += 1;
                vec_idx += 4;
                i += 64;
            }

            // 3. Widen the inner sums to 16-bit and add to main accumulators
            // This only executes once every 1,920 bytes, saving CPU time
            sum0 += @as(@Vector(16, u16), inner_sum0);
            sum1 += @as(@Vector(16, u16), inner_sum1);
            sum2 += @as(@Vector(16, u16), inner_sum2);
            sum3 += @as(@Vector(16, u16), inner_sum3);
        }

        // 4. Horizontal reduction of the 16-bit vector accumulators
        const total_vec = sum0 + sum1 + sum2 + sum3;
        return @reduce(.Add, total_vec);
    }

    pub inline fn qjlNx1dot(query_lut: []const f32, b: *const VectorQJL1, query_sum: f32) f32 {
        // dot(q, v) ~= ||rot(v)||_1 / D * dot(q, quant(rot(v)))
        // dot(q, quant(rot(v))) is hard to compute, the current
        // version is about 30% slower than the f32xf32 dot product.
        // with "usine à gaz" simd/masking/lut/assembly injection,
        // we can only get on par with the baseline.
        // TODO: on AVX512 CPUs, there is bitmasking available that
        // would allow to solve the issue at the hardware level,
        // effectively making this being a SIMD vectorized sum,
        // saving the multiplications of the f32xf32.
        var positive_sum: f32 = 0.0;
        const b_bytes: *const [hidden_dim / 8]u8 = @ptrCast(b);
        for (b_bytes, 0..) |byte, byte_i| {
            positive_sum += query_lut[byte_i * 256 + byte];
        }
        return 2.0 * positive_sum - query_sum;
    }
};

pub const VectorQJL2 = struct {
    msb: [hidden_dim / 8]u8,
    lsb: [hidden_dim / 8]u8,
};
pub fn makeVectorQJL2() VectorQJL2 {
    return .{
        .msb = [_]u8{0} ** (hidden_dim / 8),
        .lsb = [_]u8{0} ** (hidden_dim / 8),
    };
}

pub const QuantizationQJL2 = struct {
    allocator: std.mem.Allocator,
    lm_head: *LmHeadMatrix,
    d: usize,
    vocab_size: usize,
    lm_head_quantized: []VectorQJL2,
    row_quant_scale: []f32,
    buffer: []f32,

    pub fn init(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix) !QuantizationQJL2 {
        const v: usize = lm_head.n;
        const d: usize = lm_head.d;
        const lm_head_quantized = try zml_handler.allocator.alloc(VectorQJL2, v);
        const row_quant_scale = try zml_handler.allocator.alloc(f32, v);
        const buffer = try zml_handler.allocator.alloc(f32, d);
        return .{
            .allocator = zml_handler.allocator,
            .lm_head = lm_head,
            .d = d,
            .vocab_size = v,
            .lm_head_quantized = lm_head_quantized,
            .row_quant_scale = row_quant_scale,
            .buffer = buffer,
        };
    }

    pub fn deinit(self: *QuantizationQJL2) void {
        self.allocator.free(self.lm_head_quantized);
        self.allocator.free(self.row_quant_scale);
        self.allocator.free(self.buffer);
    }

    pub fn quantize(self: *QuantizationQJL2) !void {
        for (0..self.vocab_size) |row| {
            const src = self.lm_head.data[row * self.d ..][0..self.d];
            const dst = &self.lm_head_quantized[row];
            self.row_quant_scale[row] = quantizeVector(src, self.buffer, dst, self.lm_head.row_norms[row]);
        }
    }

    pub fn quantizeVector(src: []const f32, buff: []f32, dst: *VectorQJL2, src_norm: f32) f32 {
        @memset(&dst.msb, 0);
        @memset(&dst.lsb, 0);
        @memcpy(buff, src);
        walshHadamard(buff, hidden_dim_log2);

        // For the 2 bits QJL, we use the statistically optimal Lloyd codebook.
        // For a normal distribution N(0, 1/sqrt(d)), the codebook is appxox:
        // { -1.5 / sqrt(d), -0.5 / sqrt(d), +0.5 / sqrt(d), +1.5 / sqrt(d) }
        // See for example the TurboQuant paper for derivation.
        // We rounded the values, this is only a few % deviation to the optimum,
        // but allows us to do more computations in integer arithmetic.
        // The stored/dot-product representation is scaled to [-3, -1, +1, +3],
        // the quantization scale factors automatically correct any rescaling.
        // A quantized value is represented as 2 * sign(msb[i]) + 1 * sign(lsb[i])
        const unscaled_threshold: f32 = 1.0;
        const threshold: f32 = src_norm * unscaled_threshold / 64.0;

        const Vec8f = @Vector(8, f32);
        const Vec8u = @Vector(8, u32);
        const Vec8b = @Vector(8, bool);

        // Used to map a vector of booleans into a single packed byte
        const bit_values: Vec8u = .{ 1, 2, 4, 8, 16, 32, 64, 128 };
        const zeros_u32: Vec8u = @splat(0);
        const zeros_f32: Vec8f = @splat(0.0);
        const true_vec: Vec8b = @splat(true);
        const false_vec: Vec8b = @splat(false);

        const t_vec: Vec8f = @splat(threshold);
        const neg_t_vec: Vec8f = @splat(-threshold);

        // we quantize 8 coords at a time
        for (0..hidden_dim / 8) |byte_i| {
            const float_idx = byte_i * 8;
            const floats: Vec8f = buff[float_idx..][0..8].*;

            // ---------------------------------------------------------
            // MSB Logic: 1 if x > 0, else 0
            // ---------------------------------------------------------
            const m_mask = floats > zeros_f32;

            // ---------------------------------------------------------
            // LSB Logic:
            // If x > 0,  LSB = 1 if x > +T
            // If x <= 0, LSB = 1 if x > -T
            // ---------------------------------------------------------
            const gt_t = floats > t_vec;
            const le_zero = floats <= zeros_f32;
            const gt_neg_t = floats > neg_t_vec;

            // Branchless composition of the LSB condition using SIMD selects
            const in_middle_negative = @select(bool, le_zero, gt_neg_t, false_vec);
            const l_mask = @select(bool, gt_t, true_vec, in_middle_negative);

            // ---------------------------------------------------------
            // SIMD Bit Packing
            // ---------------------------------------------------------
            // This takes the boolean mask, maps true->bit_value and false->0,
            // and horizontally sums them into a single integer.
            const m_packed = @reduce(.Add, @select(u32, m_mask, bit_values, zeros_u32));
            const l_packed = @reduce(.Add, @select(u32, l_mask, bit_values, zeros_u32));

            dst.msb[byte_i] = @intCast(m_packed);
            dst.lsb[byte_i] = @intCast(l_packed);
        }

        // for two vectors a and b, we have:
        // dot(a, b) = dot(rot(a), rot(b))
        // dot(rot(a), rot(b)) ~= scale(q_a) * scale(q_b) * dot(q_a, q_b)
        // with q_a = quant(rot(a)) and q_b = quant(rot(b)),
        // and scale(q_a) = dot(rot(a), q_a) / norm(q_a)²
        // this is why when we scale q_a by a constant factor, the scale(q_a)
        // grows by 1 / factor, leaving the dot product estimator invariant.
        var dot: f32 = 0.0;
        var norm_squared: f32 = 0.0;
        for (0..hidden_dim / 8) |byte_i| {
            const m_byte = dst.msb[byte_i];
            const l_byte = dst.lsb[byte_i];
            inline for (0..8) |bit_i| {
                const mask: u8 = @as(u8, 1) << bit_i;
                const m: f32 = if ((m_byte & mask) != 0) 1.0 else -1.0;
                const l: f32 = if ((l_byte & mask) != 0) 1.0 else -1.0;
                const quant_value = 2.0 * m + 1.0 * l;
                dot += buff[byte_i * 8 + bit_i] * quant_value;
                norm_squared += quant_value * quant_value;
            }
        }
        return dot / norm_squared;
    }

    pub inline fn qjl2dot(a: *const VectorQJL2, b: *const VectorQJL2) i32 {
        // A quantized value is represented as 2 * sign(msb[i]) + 1 * sign(lsb[i])
        // dot(a, b) = dot(2 * sign(a.msb[i]) + 1 * sign(a.lsb[i]), 2 * sign(b.msb[i]) + 1 * sign(b.lsb[i]))
        // dot(a,b) = 4 * dot1(a.msb, b.msb) + 2 * dot1(a.msb, b.lsb) + 2 * dot1(a.lsb, b.msb) + dot1(a.lsb, b.lsb)

        const pop_mm = QuantizationQJL1.popcountXor(&@bitCast(a.msb), &@bitCast(b.msb));
        const pop_ml = QuantizationQJL1.popcountXor(&@bitCast(a.msb), &@bitCast(b.lsb));
        const pop_lm = QuantizationQJL1.popcountXor(&@bitCast(a.lsb), &@bitCast(b.msb));
        const pop_ll = QuantizationQJL1.popcountXor(&@bitCast(a.lsb), &@bitCast(b.lsb));

        // dot(a,b) = 4 * (d - 2 * pop_mm) + 2 * (d - 2 * pop_ml + d - 2 * pop_lm) + (d - 2 * pop_ll)
        // dot(a,b) = 9 * d - 8 * pop_mm - 4 * pop_ml - 4 * pop_lm - 2 * pop_ll

        const pos: i32 = 9 * hidden_dim;
        const neg = (pop_mm << 3) + ((pop_ml + pop_lm) << 2) + (pop_ll << 1);
        return pos - @as(i32, @intCast(neg));
    }

    pub inline fn qjl2x1dot(a: *const VectorQJL2, b: *const VectorQJL1) i32 {
        // A 2 bits quantized value is represented as 2 * sign(msb[i]) + 1 * sign(lsb[i])
        // dot(a, b) = dot(2 * sign(a.msb[i]) + 1 * sign(a.lsb[i]), sign(b))
        // dot(a,b) = 2 * dot1(a.msb, b) + 2 * dot1(a.lsb, b)

        const pop_m = QuantizationQJL1.popcountXor(&@bitCast(a.msb), b);
        const pop_l = QuantizationQJL1.popcountXor(&@bitCast(a.lsb), b);

        // dot(a,b) = 2 * (d - 2 * pop_m) + 2 * (d - 2 * pop_l)
        // dot(a,b) = 4 * d - 4 * pop_m - 4 * pop_l

        const pos: i32 = 4 * hidden_dim;
        const neg = (pop_m + pop_l) << 2;
        return pos - @as(i32, @intCast(neg));
    }
};
