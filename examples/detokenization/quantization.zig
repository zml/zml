const std = @import("std");
const zml = @import("zml");
const main = @import("main.zig");
const model_ = @import("model.zig");
const llm_ = @import("llm.zig");
const tokens = @import("tokens.zig");
const algebra = @import("algebra.zig");

const Zml_handler = main.Zml_handler;
const Tokenizer = zml.tokenizer.Tokenizer;
const LmHeadMatrix = algebra.LmHeadMatrix;

const simd_len_f32: comptime_int = 16;
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
    const res: f32 = 0.0;
    for (0..v) |val| { res += val * val; }
    return @sqrt(res);
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
};

const int4_block_len: comptime_int = 32;
const int4_packed_block_len: comptime_int = int4_block_len / 2;

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
};

pub const VectorQJL2 = struct {
    msb: [hidden_dim/8]u8,
    lsb: [hidden_dim/8]u8,
};
pub fn makeVectorQJL2() VectorQJL2 {
    return .{
        .msb = [_]u8{0} ** (hidden_dim/8),
        .lsb = [_]u8{0} ** (hidden_dim/8),
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
        for (0..hidden_dim/8) |byte_i| {
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
        for (0..hidden_dim/8) |byte_i| {
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

};
