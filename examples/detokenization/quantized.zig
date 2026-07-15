const std = @import("std");
const zml = @import("zml");
const main = @import("main.zig");
const tokens = @import("tokens.zig");
const algebra = @import("algebra.zig");

const LmHeadMatrix = algebra.LmHeadMatrix;
const Tokenizer = zml.tokenizer.Tokenizer;

pub const comptime_sliced_quant_bits: comptime_int = 3;
pub const comptime_sliced_quant_scale: comptime_int = (1 << comptime_sliced_quant_bits) - 1;
pub const comptime_sliced_nb_bits: comptime_int = comptime_sliced_quant_bits + 1;
pub const comptime_sliced_quant_scale_f32: f32 = @floatFromInt(comptime_sliced_quant_scale);

pub const Logit = struct {
    index: usize,
    logit_f32: f32,
    logit_i8: f32,
    logit_sliced: f32,
    logit_qjl: f32,
    lower_bound: f32,
    upper_bound: f32,
    proba: f32,
    quantized_rank: usize,
    sliced_rank: usize,
    qjl_rank: usize,

    pub fn decreasingOrder(_: void, lhs: Logit, rhs: Logit) bool {
        return (lhs.logit_f32 > rhs.logit_f32) or (lhs.logit_f32 == rhs.logit_f32 and lhs.index < rhs.index);
    }

    pub fn decreasingQuantizedOrder(_: void, lhs: Logit, rhs: Logit) bool {
        return (lhs.logit_i8 > rhs.logit_i8) or (lhs.logit_i8 == rhs.logit_i8 and lhs.index < rhs.index);
    }

    pub fn decreasingSlicedOrder(_: void, lhs: Logit, rhs: Logit) bool {
        return (lhs.logit_sliced > rhs.logit_sliced) or (lhs.logit_sliced == rhs.logit_sliced and lhs.index < rhs.index);
    }

    pub fn decreasingQjlOrder(_: void, lhs: Logit, rhs: Logit) bool {
        return (lhs.logit_qjl > rhs.logit_qjl) or (lhs.logit_qjl == rhs.logit_qjl and lhs.index < rhs.index);
    }

    fn boundViolation(logit: Logit) f32 {
        if (logit.logit_f32 < logit.lower_bound) return logit.lower_bound - logit.logit_f32;
        if (logit.logit_f32 > logit.upper_bound) return logit.logit_f32 - logit.upper_bound;
        return 0.0;
    }

    fn boundMargin(logit: Logit) f32 {
        return @min(@abs(logit.logit_f32 - logit.lower_bound), @abs(logit.upper_bound - logit.logit_f32));
    }

    pub fn boundPressureOrder(_: void, lhs: Logit, rhs: Logit) bool {
        const lhs_violation = boundViolation(lhs);
        const rhs_violation = boundViolation(rhs);
        if (lhs_violation != rhs_violation) return lhs_violation > rhs_violation;

        const lhs_margin = boundMargin(lhs);
        const rhs_margin = boundMargin(rhs);
        return lhs_margin < rhs_margin or (lhs_margin == rhs_margin and lhs.index < rhs.index);
    }
};

pub const top_k_sliced: usize = 16;
pub const default_syntax_token_id_threshold: usize = 2048;
pub const qjl_micro_coord_count: usize = 1024;
pub const qjl_micro_u128_word_count: usize = qjl_micro_coord_count / 128;
pub const qjl_half_coord_count: usize = 2 * qjl_micro_coord_count;
pub const qjl_half_to_full_coord_count: usize = 4096 - qjl_half_coord_count;
pub const qjl_half_to_full_u128_word_count: usize = qjl_half_to_full_coord_count / 128;
pub const partial_dot_block_size: usize = 128;
pub const partial_dot_block_count: usize = 4096 / partial_dot_block_size;
pub const partial_dot_phase_exact_top_k: usize = 16;
pub const partial_dot_prefix_coord_count: usize = 128;

comptime {
    std.debug.assert(4096 % partial_dot_block_size == 0);
    std.debug.assert(partial_dot_block_size % 16 == 0);
}

fn makeQjlSignDotLut(comptime coord_count: usize) [coord_count + 1]f32 {
    @setEvalBranchQuota(8192);
    var lut: [coord_count + 1]f32 = undefined;
    for (0..(coord_count + 1)) |mismatch_count| {
        const dot: i32 = @as(i32, @intCast(coord_count)) - 2 * @as(i32, @intCast(mismatch_count));
        const w = @as(f32, @floatFromInt(dot)) / @as(f32, @floatFromInt(coord_count));
        lut[mismatch_count] = @sin(@as(f32, @floatCast(0.5 * std.math.pi)) * w);
    }
    return lut;
}

const qjl_micro_sign_dot_lut = makeQjlSignDotLut(qjl_micro_coord_count);
const qjl_half_sign_dot_lut = makeQjlSignDotLut(qjl_half_coord_count);
const qjl_full_sign_dot_lut = makeQjlSignDotLut(4096);

pub const DenseSample = struct {
    rows: [top_k_sliced]usize,
    logits: [top_k_sliced]f32,
    probas: [top_k_sliced]f32,
    max_logit: f32,
    total_exp: f32,

    pub inline fn probaFromLogit(self: DenseSample, logit: f32) f32 {
        return @exp(logit - self.max_logit) / self.total_exp;
    }
};

pub const TwoPhaseSample = struct {
    rows: [top_k_sliced]usize,
    nb_scored: usize,
    nb_1bit_scored: usize = 0,
    nb_half_1bit_scored: usize = 0,
    nb_2bit_scored: usize = 0,
    nb_partial_dot_scored: usize = 0,
    nb_partial_dot_pruned: usize = 0,
    partial_dot_pruned_by_phase: [partial_dot_block_count]usize = [_]usize{0} ** partial_dot_block_count,
};

pub const U4096 = struct {
    words: [32]u128 = [_]u128{0} ** 32,

    pub fn set(self: *U4096, bit: usize) void {
        std.debug.assert(bit < 4096);
        self.words[bit / 128] |= @as(u128, 1) << @as(u7, @intCast(bit % 128));
    }

    pub inline fn popCount(self: *const U4096) u16 {
        var count: u16 = 0;
        inline for (0..32) |word_i| {
            count += @popCount(self.words[word_i]);
        }
        return count;
    }
};

pub const vector_U4096 = struct {
    bits: [comptime_sliced_nb_bits]U4096 = [_]U4096{.{}} ** comptime_sliced_nb_bits,

    pub fn dot(a: *const vector_U4096, b: *const vector_U4096) u32 {
        var total: u32 = 0;
        for (0..32) |word_i| {
            inline for (0..comptime_sliced_nb_bits) |b_bit| {
                const b_word = b.bits[b_bit].words[word_i];
                var partial: u32 = 0;
                inline for (0..comptime_sliced_nb_bits) |a_bit| {
                    const count: u32 = @intCast(@popCount(a.bits[a_bit].words[word_i] & b_word));
                    partial += count << @as(u8, @intCast(a_bit));
                }
                total += partial << @as(u8, @intCast(b_bit));
            }
        }
        return total;
    }
};

pub const Quantized = struct {
    zml_handler: *main.Zml_handler,
    allocator: std.mem.Allocator,
    lm_head: *LmHeadMatrix,

    vocab_size: usize,
    d: usize,
    d_inv_norm: f32,

    lm_head_quantized: []i8,
    row_scale: []f32,
    row_scale_sliced: []f32,
    f32_buffer: []f32,
    dense_logits: []f32,
    partial_logits: []f32,
    partial_dot_residual_logits: []f32,
    active_rows: []u32,
    i8_buffer: []i8,

    lm_head_sliced: []vector_U4096,
    row_offset: []i32,

    lm_head_qjl: []U4096,
    qjl_micro_rows: [][qjl_micro_u128_word_count]u128,
    qjl_micro_to_half_rows: [][qjl_micro_u128_word_count]u128,
    qjl_half_to_full_rows: [][qjl_half_to_full_u128_word_count]u128,
    qjl_micro_mismatches: []u16,
    qjl_row_scale: []f32,
    qjl_query_lut: []f32,
    qjl_inv_dim: f32 = 0.0,

    partial_dot_blocks: []f32,

    lm_head_qjl_2bits: [][64]u128,
    qjl_row_scale_2bits: []f32,

    pub fn init(zml_handler: *main.Zml_handler, lm_head: *LmHeadMatrix) !Quantized {
        const allocator = zml_handler.allocator;

        const vocab_size: usize = lm_head.n;
        const d: usize = lm_head.d;
        std.debug.assert(d == 4096);

        const d_inv_norm: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));

        const f32_buffer = try allocator.alloc(f32, d);
        const dense_logits = try allocator.alloc(f32, vocab_size);
        const partial_logits = try allocator.alloc(f32, vocab_size);
        const partial_dot_residual_logits = try allocator.alloc(f32, vocab_size);
        const active_rows = try allocator.alloc(u32, vocab_size);

        const lm_head_quantized = try allocator.alloc(i8, d * vocab_size);
        const lm_head_sliced = try allocator.alloc(vector_U4096, vocab_size);

        const row_offset = try allocator.alloc(i32, vocab_size);
        const row_scale = try allocator.alloc(f32, vocab_size);
        const row_scale_sliced = try allocator.alloc(f32, vocab_size);
        const i8_buffer = try allocator.alloc(i8, d);

        const lm_head_qjl = try allocator.alloc(U4096, vocab_size);
        const qjl_micro_rows = try allocator.alloc([qjl_micro_u128_word_count]u128, vocab_size);
        const qjl_micro_to_half_rows = try allocator.alloc([qjl_micro_u128_word_count]u128, vocab_size);
        const qjl_half_to_full_rows = try allocator.alloc([qjl_half_to_full_u128_word_count]u128, vocab_size);
        const qjl_micro_mismatches = try allocator.alloc(u16, vocab_size);
        const qjl_row_scale = try allocator.alloc(f32, vocab_size);
        const qjl_query_lut = try allocator.alloc(f32, 512 * 256);
        const partial_dot_blocks = try allocator.alloc(f32, vocab_size * d);
        const lm_head_qjl_2bits = try allocator.alloc([64]u128, vocab_size);
        const qjl_row_scale_2bits = try allocator.alloc(f32, vocab_size);

        return .{
            .zml_handler = zml_handler,
            .allocator = allocator,
            .lm_head = lm_head,
            .vocab_size = vocab_size,
            .d = d,
            .d_inv_norm = d_inv_norm,
            .row_scale = row_scale,
            .row_scale_sliced = row_scale_sliced,
            .f32_buffer = f32_buffer,
            .dense_logits = dense_logits,
            .partial_logits = partial_logits,
            .partial_dot_residual_logits = partial_dot_residual_logits,
            .active_rows = active_rows,
            .lm_head_quantized = lm_head_quantized,
            .lm_head_sliced = lm_head_sliced,
            .row_offset = row_offset,
            .i8_buffer = i8_buffer,
            .lm_head_qjl = lm_head_qjl,
            .qjl_micro_rows = qjl_micro_rows,
            .qjl_micro_to_half_rows = qjl_micro_to_half_rows,
            .qjl_half_to_full_rows = qjl_half_to_full_rows,
            .qjl_micro_mismatches = qjl_micro_mismatches,
            .qjl_row_scale = qjl_row_scale,
            .qjl_query_lut = qjl_query_lut,
            .partial_dot_blocks = partial_dot_blocks,
            .lm_head_qjl_2bits = lm_head_qjl_2bits,
            .qjl_row_scale_2bits = qjl_row_scale_2bits,
        };
    }

    pub fn deinit(self: *Quantized) void {
        self.allocator.free(self.row_scale);
        self.allocator.free(self.row_scale_sliced);
        self.allocator.free(self.f32_buffer);
        self.allocator.free(self.dense_logits);
        self.allocator.free(self.partial_logits);
        self.allocator.free(self.partial_dot_residual_logits);
        self.allocator.free(self.active_rows);
        self.allocator.free(self.lm_head_quantized);
        self.allocator.free(self.lm_head_sliced);
        self.allocator.free(self.row_offset);
        self.allocator.free(self.i8_buffer);
        self.allocator.free(self.lm_head_qjl);
        self.allocator.free(self.qjl_micro_rows);
        self.allocator.free(self.qjl_micro_to_half_rows);
        self.allocator.free(self.qjl_half_to_full_rows);
        self.allocator.free(self.qjl_micro_mismatches);
        self.allocator.free(self.qjl_row_scale);
        self.allocator.free(self.qjl_query_lut);
        self.allocator.free(self.partial_dot_blocks);
        self.allocator.free(self.lm_head_qjl_2bits);
        self.allocator.free(self.qjl_row_scale_2bits);
    }

    pub inline fn quantizeValue(value: f32, max_abs: f32) i8 {
        return quantizeValueScaled(value, max_abs, 127.0);
    }

    pub inline fn quantizeValueScaled(value: f32, max_abs: f32, scale: f32) i8 {
        if (max_abs == 0.0) return 0;
        const scaled = scale * value / max_abs;
        return @as(i8, @intFromFloat(scaled));
    }

    pub inline fn dequantizeValue(value: i8) f32 {
        return @as(f32, @floatFromInt(value));
    }

    pub fn walshHadamardTo(dst: []f32, v: []const f32, comptime k: comptime_int) void {
        const n: usize = 1 << k;
        std.debug.assert(v.len == n);
        std.debug.assert(dst.len >= n);
        if (@intFromPtr(dst.ptr) != @intFromPtr(v.ptr)) {
            @memcpy(dst[0..n], v[0..n]);
        }

        inline for (0..k) |stage| {
            const h: usize = 1 << stage;
            const step: usize = 2 * h;
            for (0..(n / step)) |block| {
                const i: usize = block * step;
                for (0..h) |offset| {
                    const j: usize = i + offset;
                    const x = dst[j];
                    const y = dst[j + h];
                    dst[j] = x + y;
                    dst[j + h] = x - y;
                }
            }
        }

        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(n)));
        for (0..n) |i| {
            dst[i] *= scale;
        }
    }

    pub fn walshHadamard(self: *Quantized, v: []const f32, comptime k: comptime_int) !void {
        self.zml_handler.tic(&self.zml_handler.timers.quant_walsh);
        walshHadamardTo(self.f32_buffer, v, k);
        self.zml_handler.toc(&self.zml_handler.timers.quant_walsh);
    }

    pub fn walshHadamardSimdTo(dst: []f32, v: []const f32, comptime k: comptime_int) void {
        const n: usize = 1 << k;
        std.debug.assert(v.len == n);
        std.debug.assert(dst.len >= n);
        if (@intFromPtr(dst.ptr) != @intFromPtr(v.ptr)) {
            @memcpy(dst[0..n], v[0..n]);
        }

        const max_simd_len: comptime_int = 8;

        inline for (0..k) |stage| {
            const h: usize = 1 << stage;
            const step: usize = 2 * h;
            const simd_len: comptime_int = if ((1 << stage) < max_simd_len) (1 << stage) else max_simd_len;
            const Vec = @Vector(simd_len, f32);

            for (0..(n / step)) |block| {
                const i: usize = block * step;
                var offset: usize = 0;
                while (offset < h) : (offset += simd_len) {
                    const left = i + offset;
                    const right = left + h;
                    const x: Vec = dst[left..][0..simd_len].*;
                    const y: Vec = dst[right..][0..simd_len].*;
                    dst[left..][0..simd_len].* = x + y;
                    dst[right..][0..simd_len].* = x - y;
                }
            }
        }

        const Vec = @Vector(max_simd_len, f32);
        const scale_scalar: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(n)));
        const scale: Vec = @splat(scale_scalar);
        var i: usize = 0;
        while (i + max_simd_len <= n) : (i += max_simd_len) {
            const values: Vec = dst[i..][0..max_simd_len].*;
            dst[i..][0..max_simd_len].* = values * scale;
        }
        while (i < n) : (i += 1) {
            dst[i] *= scale_scalar;
        }
    }

    pub fn walshHadamardSimd(self: *Quantized, v: []const f32, comptime k: comptime_int) !void {
        self.zml_handler.tic(&self.zml_handler.timers.quant_walsh);
        walshHadamardSimdTo(self.f32_buffer, v, k);
        self.zml_handler.toc(&self.zml_handler.timers.quant_walsh);
    }

    pub inline fn normL2(a: []const f32) f32 {
        var norm2: f32 = 0.0;
        for (a) |x| {
            norm2 += x * x;
        }
        return @sqrt(norm2);
    }
    
    fn l2Error(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);
        var squared_error: f32 = 0.0;
        for (a, b) |x, y| {
            const diff = x - y;
            squared_error += diff * diff;
        }
        return @sqrt(squared_error);
    }

    pub fn testWalshHadamardRoundtrip(self: *Quantized) !void {
        std.log.info("Test Walsh-Hadamard roundtrip", .{});

        var prng = std.Random.DefaultPrng.init(2);
        const random = prng.random();

        const original = try self.allocator.alloc(f32, self.d);
        defer self.allocator.free(original);
        const reference = try self.allocator.alloc(f32, self.d);
        defer self.allocator.free(reference);
        const transformed = try self.allocator.alloc(f32, self.d);
        defer self.allocator.free(transformed);

        var total_error: f64 = 0.0;
        var max_error: f32 = 0.0;
        var count: usize = 0;

        for (0..1000) |_| {
            for (original) |*value| {
                value.* = 2.0 * random.float(f32) - 1.0;
            }
            @memcpy(reference, original);
            try self.walshHadamard(original, 12);
            @memcpy(transformed, self.f32_buffer);
            try self.walshHadamard(transformed, 12);

            const err = l2Error(reference, self.f32_buffer);
            total_error += @as(f64, err);
            max_error = @max(max_error, err);
            count += 1;
        }

        for (0..1000) |_| {
            const row = random.uintLessThan(usize, self.vocab_size);
            const original_row = self.lm_head.data[row * self.d .. (row + 1) * self.d];
            @memcpy(original, original_row);
            @memcpy(reference, original);
            try self.walshHadamard(original, 12);
            @memcpy(transformed, self.f32_buffer);
            try self.walshHadamard(transformed, 12);

            const err = l2Error(reference, self.f32_buffer);
            total_error += @as(f64, err);
            max_error = @max(max_error, err);
            count += 1;
        }

        const avg_error = total_error / @as(f64, @floatFromInt(count));
        std.log.info(
            "Walsh-Hadamard roundtrip: vectors={d} avg_l2_error={d:.8} max_l2_error={d:.8}",
            .{ count, avg_error, max_error },
        );
    }

    pub fn quantizeVector(self: *Quantized, query: []const f32) !f32 {
        return self.quantizeVectorScaled(query, 127.0, self.i8_buffer);
    }

    pub fn quantizeVectorScaled(self: *Quantized, query: []const f32, scale: f32, out: []i8) !f32 {
        std.debug.assert(out.len == self.d);
        try self.walshHadamard(query, 12);
        var norm: f32 = 0.0;
        var max_abs: f32 = 0;
        for (self.f32_buffer) |val| {
            norm += val * val;
            max_abs = @max(max_abs, @abs(val));
        }
        norm = @sqrt(norm);
        var norm_quantized: f32 = 0.0;
        for (0..self.d) |j| {
            const q = quantizeValueScaled(self.f32_buffer[j], max_abs, scale);
            out[j] = q;
            const u = dequantizeValue(q);
            norm_quantized += u * u;
        }
        norm_quantized = @sqrt(norm_quantized);
        return norm / norm_quantized;
    }

    pub fn quantizeLmHead(self: *Quantized) !void {
        std.log.info("Quantizing lm_head", .{});
        const lm_head = self.lm_head.data;
        for (0..self.vocab_size) |i| {
            const row = lm_head[i * self.d .. (i + 1) * self.d];
            self.row_scale[i] = try self.quantizeVector(row);
            @memcpy(self.lm_head_quantized[i * self.d .. (i + 1) * self.d], self.i8_buffer[0..self.d]);
            //std.log.info("Quantized row {d}: norm={d}, max_abs={d}, scale={d}", .{ i, norm, max_abs, self.row_scale[i] });
        }
    }

    pub fn sample(self: *Quantized, query: []const f32) !usize {
        const query_scale = try self.quantizeVector(query);
        var best_row: usize = 0;
        var best_score: f32 = -1e20;
        for (0..self.vocab_size) |i| {
            if (self.lm_head.is_junk[i]) continue;
            const quantized_row = self.lm_head_quantized[i * self.d .. (i + 1) * self.d];
            const score = quantizedDot(self, self.i8_buffer, quantized_row);
            var score_f32 = @as(f32, @floatFromInt(score));
            score_f32 *= query_scale;
            score_f32 *= self.row_scale[i];
            if (score_f32 > best_score) {
                best_row = i;
                best_score = score_f32;
            }
        }
        return best_row;
    }

    pub fn quantizedDot(self: *Quantized, query: []const i8, row: []const i8) i32 {
        self.zml_handler.tic(&self.zml_handler.timers.quant_dot);
        std.debug.assert(query.len == row.len);

        const simd_len = 16;
        const VecI8 = @Vector(simd_len, i8);
        const VecI32 = @Vector(simd_len, i32);

        var acc: VecI32 = @splat(0);
        var i: usize = 0;
        while (i + simd_len <= query.len) : (i += simd_len) {
            const query_i8: VecI8 = query[i..][0..simd_len].*;
            const row_i8: VecI8 = row[i..][0..simd_len].*;
            const query_i32: VecI32 = @as(VecI32, @intCast(query_i8));
            const row_i32: VecI32 = @as(VecI32, @intCast(row_i8));
            acc += query_i32 * row_i32;
        }
        const res = @reduce(.Add, acc);
        self.zml_handler.toc(&self.zml_handler.timers.quant_dot);
        return res;
    }

    pub fn realLogit(_: *Quantized, query: []const f32, row: []const f32) f32 {
        const simd_len = 16;
        const Vec = @Vector(simd_len, f32);
        var acc: Vec = @splat(0);
        var i: usize = 0;
        while (i + simd_len <= query.len) : (i += simd_len) {
            const query_v: Vec = query[i..][0..simd_len].*;
            const row_v: Vec = row[i..][0..simd_len].*;
            acc = @mulAdd(Vec, query_v, row_v, acc);
        }
        return @reduce(.Add, acc);
    }

    pub fn logSampling(self: *Quantized, query: []const f32, tokenizer: *Tokenizer, log_basic_quantize: bool, log_slice: bool, log_qjl: bool) !void {
        const logits = try self.allocator.alloc(Logit, self.vocab_size);
        defer self.allocator.free(logits);

        const query_scale = if (log_basic_quantize) try self.quantizeVector(query) else 0.0;
        for (0..self.vocab_size) |i| {
            const row = self.lm_head.data[i * self.d .. (i + 1) * self.d];
            var score_f32: f32 = 0.0;
            if (log_basic_quantize) {
                const quantized_row = self.lm_head_quantized[i * self.d .. (i + 1) * self.d];
                const score = quantizedDot(self, self.i8_buffer, quantized_row);
                score_f32 = @as(f32, @floatFromInt(score));
                score_f32 *= query_scale;
                score_f32 *= self.row_scale[i];
            }
            logits[i] = .{
                .index = i,
                .logit_f32 = self.realLogit(query, row),
                .logit_i8 = score_f32,
                .logit_sliced = 0.0,
                .logit_qjl = 0.0,
                .lower_bound = 0.0,
                .upper_bound = 0.0,
                .proba = 0.0,
                .quantized_rank = 0,
                .sliced_rank = 0,
                .qjl_rank = 0,
            };
        }

        if (log_slice) {
            const query_scale_sliced = try self.quantizeVectorScaled(query, comptime_sliced_quant_scale_f32, self.i8_buffer);
            const query_sliced = try sliceVectorBits(self.i8_buffer);
            var query_sum: i32 = 0;
            for (self.i8_buffer) |coef| {
                query_sum += @as(i32, coef);
            }
            const query_offset = comptime_sliced_quant_scale * query_sum;
            for (0..self.vocab_size) |i| {
                const score = self.slicedDot(&query_sliced, query_offset, i);
                var score_f32 = @as(f32, @floatFromInt(score));
                score_f32 *= query_scale_sliced;
                score_f32 *= self.row_scale_sliced[i];
                logits[i].logit_sliced = score_f32;
            }
        }

        if (log_qjl) {
            const quantized_query, _ = try self.quantizeQjlVector(query);
            var query_l2_norm: f32 = 0.0;
            for (0..self.d) |i| {
                query_l2_norm += query[i] * query[i];
            }
            query_l2_norm = @sqrt(query_l2_norm);
            for (0..self.vocab_size) |i| {
                logits[i].logit_qjl = self.dotQjlSymmetric(i, quantized_query, query_l2_norm);
            }
        }

        if (log_basic_quantize) {
            std.mem.sort(Logit, logits, {}, Logit.decreasingQuantizedOrder);
            for (logits, 0..) |*logit, rank| {
                logit.quantized_rank = rank + 1;
            }
        }
        if (log_slice) {
            std.mem.sort(Logit, logits, {}, Logit.decreasingSlicedOrder);
            for (logits, 0..) |*logit, rank| {
                logit.sliced_rank = rank + 1;
            }
        }
        if (log_qjl) {
            std.mem.sort(Logit, logits, {}, Logit.decreasingQjlOrder);
            for (logits, 0..) |*logit, rank| {
                logit.qjl_rank = rank + 1;
            }
        }
        std.mem.sort(Logit, logits, {}, Logit.decreasingOrder);

        const nb_logged = @min(16, logits.len);
        const max_logit = logits[0].logit_f32;
        var total: f32 = 0.0;
        for (logits[0..nb_logged]) |*logit| {
            logit.proba = @exp(logit.logit_f32 - max_logit);
            total += logit.proba;
        }
        for (logits[0..nb_logged]) |*logit| {
            logit.proba /= total;
        }

        std.log.info("Top {d} quantized sampling logits", .{nb_logged});
        var header = try std.fmt.allocPrint(self.allocator, "{s:>4} {s:>8} {s:>14} {s:>14}", .{ "rank", "token", "proba", "real" });
        defer self.allocator.free(header);
        if (log_basic_quantize) {
            const next = try std.fmt.allocPrint(self.allocator, "{s} {s:>20}", .{ header, "quantized(pos)" });
            self.allocator.free(header);
            header = next;
        }
        if (log_slice) {
            const next = try std.fmt.allocPrint(self.allocator, "{s} {s:>20}", .{ header, "sliced(pos)" });
            self.allocator.free(header);
            header = next;
        }
        if (log_qjl) {
            const next = try std.fmt.allocPrint(self.allocator, "{s} {s:>20}", .{ header, "qjl(pos)" });
            self.allocator.free(header);
            header = next;
        }
        const header_with_text = try std.fmt.allocPrint(self.allocator, "{s}  {s}", .{ header, "text" });
        defer self.allocator.free(header_with_text);
        std.log.info("{s}", .{header_with_text});

        for (logits[0..nb_logged], 0..) |logit, rank| {
            const token_str = try tokens.tokenString(tokenizer.*, logit.index, self.allocator);
            defer self.allocator.free(token_str);
            const real_str = try std.fmt.allocPrint(self.allocator, "{d:.5}", .{logit.logit_f32});
            defer self.allocator.free(real_str);

            var line = try std.fmt.allocPrint(self.allocator, "{d:>4} {d:>8} {d:>14.8} {s:>14}", .{ rank + 1, logit.index, logit.proba, real_str });
            defer self.allocator.free(line);
            if (log_basic_quantize) {
                const quant_str = try std.fmt.allocPrint(self.allocator, "{d:.5} ({d})", .{ logit.logit_i8, logit.quantized_rank });
                defer self.allocator.free(quant_str);
                const next = try std.fmt.allocPrint(self.allocator, "{s} {s:>20}", .{ line, quant_str });
                self.allocator.free(line);
                line = next;
            }
            if (log_slice) {
                const sliced_str = try std.fmt.allocPrint(self.allocator, "{d:.5} ({d})", .{ logit.logit_sliced, logit.sliced_rank });
                defer self.allocator.free(sliced_str);
                const next = try std.fmt.allocPrint(self.allocator, "{s} {s:>20}", .{ line, sliced_str });
                self.allocator.free(line);
                line = next;
            }
            if (log_qjl) {
                const qjl_str = try std.fmt.allocPrint(self.allocator, "{d:.5} ({d})", .{ logit.logit_qjl, logit.qjl_rank });
                defer self.allocator.free(qjl_str);
                const next = try std.fmt.allocPrint(self.allocator, "{s} {s:>20}", .{ line, qjl_str });
                self.allocator.free(line);
                line = next;
            }
            const line_with_text = try std.fmt.allocPrint(self.allocator, "{s}  {s}", .{ line, token_str });
            defer self.allocator.free(line_with_text);
            std.log.info("{s}", .{line_with_text});
        }
    }

    pub fn sliceVectorBits(row: []const i8) !vector_U4096 {
        var sliced: vector_U4096 = .{};
        for (row, 0..) |value, coord| {
            std.debug.assert(value >= -comptime_sliced_quant_scale and value <= comptime_sliced_quant_scale);
            const shifted: i16 = @as(i16, value) + comptime_sliced_quant_scale;
            std.debug.assert(shifted >= 0 and shifted < (1 << comptime_sliced_nb_bits));
            const unsigned_value: u8 = @intCast(shifted);
            inline for (0..comptime_sliced_nb_bits) |bit| {
                if ((unsigned_value & (@as(u8, 1) << bit)) != 0) {
                    sliced.bits[bit].set(coord);
                }
            }
        }
        return sliced;
    }

    pub fn sliceLmHead(self: *Quantized) !void {
        std.log.info("Slicing lm_head", .{});
        for (0..self.vocab_size) |i| {
            const row = self.lm_head.data[i * self.d .. (i + 1) * self.d];
            self.row_scale_sliced[i] = try self.quantizeVectorScaled(row, comptime_sliced_quant_scale_f32, self.i8_buffer);
            var row_sum: i32 = 0;
            for (self.i8_buffer) |coef| {
                row_sum += @as(i32, coef);
            }
            self.row_offset[i] = comptime_sliced_quant_scale * row_sum;
            self.lm_head_sliced[i] = try sliceVectorBits(self.i8_buffer);
        }
    }

    pub inline fn slicedDot(self: *Quantized, query_sliced: *const vector_U4096, query_offset: i32, row: usize) i32 {
        //self.zml_handler.tic(&self.zml_handler.timers.quant_slice_dot);
        const constant_offset: i32 = comptime_sliced_quant_scale * comptime_sliced_quant_scale * @as(i32, @intCast(self.d));
        const shifted_score = vector_U4096.dot(query_sliced, &self.lm_head_sliced[row]);
        const res = @as(i32, @intCast(shifted_score)) - query_offset - self.row_offset[row] - constant_offset;
        //self.zml_handler.toc(&self.zml_handler.timers.quant_slice_dot);
        return res;
    }

    pub fn sampleSlice(self: *Quantized, query: []const f32) ![top_k_sliced]usize {
        const query_scale = try self.quantizeVectorScaled(query, comptime_sliced_quant_scale_f32, self.i8_buffer);
        const query_sliced = try sliceVectorBits(self.i8_buffer);
        var query_sum: i32 = 0;
        for (self.i8_buffer) |coef| {
            query_sum += @as(i32, coef);
        }
        const query_offset = comptime_sliced_quant_scale * query_sum;

        var top_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var top_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;
        for (0..self.vocab_size) |i| {
            if (self.lm_head.is_junk[i]) continue;
            const score = self.slicedDot(&query_sliced, query_offset, i);
            var score_f32 = @as(f32, @floatFromInt(score));
            score_f32 *= query_scale;
            score_f32 *= self.row_scale_sliced[i];
            if (score_f32 > top_scores[top_k_sliced - 1]) {
                var insert_pos = top_k_sliced - 1;
                while (insert_pos > 0 and score_f32 > top_scores[insert_pos - 1]) {
                    top_scores[insert_pos] = top_scores[insert_pos - 1];
                    top_rows[insert_pos] = top_rows[insert_pos - 1];
                    insert_pos -= 1;
                }
                top_scores[insert_pos] = score_f32;
                top_rows[insert_pos] = i;
            }
        }
        return top_rows;
    }

    pub fn testQuantizeSlice(self: *Quantized) !void {
        std.log.info("Test quantized bit slicing", .{});

        var prng = std.Random.DefaultPrng.init(0);
        const random = prng.random();
        const query_buffer = try self.allocator.alloc(i8, self.d);
        defer self.allocator.free(query_buffer);
        const row_buffer = try self.allocator.alloc(i8, self.d);
        defer self.allocator.free(row_buffer);

        for (0..1000) |test_i| {
            const query_row = random.uintLessThan(usize, self.vocab_size);
            const lm_head_row = random.uintLessThan(usize, self.vocab_size);
            const query_row_f32 = self.lm_head.data[query_row * self.d .. (query_row + 1) * self.d];
            const row_f32 = self.lm_head.data[lm_head_row * self.d .. (lm_head_row + 1) * self.d];

            _ = try self.quantizeVectorScaled(query_row_f32, comptime_sliced_quant_scale_f32, query_buffer);
            _ = try self.quantizeVectorScaled(row_f32, comptime_sliced_quant_scale_f32, row_buffer);

            const quantized_score = self.quantizedDot(query_buffer, row_buffer);
            var query_sum: i32 = 0;
            for (query_buffer) |coef| {
                query_sum += @as(i32, coef);
            }
            const query_offset = comptime_sliced_quant_scale * query_sum;
            const query_sliced = try sliceVectorBits(query_buffer);
            const sliced_score = self.slicedDot(&query_sliced, query_offset, lm_head_row);

            if (quantized_score != sliced_score) {
                std.log.err(
                    "Quantized slice mismatch test={d} query_row={d} row={d} quantized={d} sliced={d}",
                    .{ test_i, query_row, lm_head_row, quantized_score, sliced_score },
                );
                return error.QuantizedSliceMismatch;
            }
        }

        std.log.info("Quantized bit slicing test passed: 1000 random row pairs", .{});
    }

    inline fn qjlSliceU128Words(comptime start_coord: usize, comptime coord_count: usize, quantized: *const U4096) *const [coord_count / 128]u128 {
        if (coord_count % 128 != 0) {
            @compileError("QJL slice length must contain a whole number of u128 words");
        }
        if (start_coord % 128 != 0) {
            @compileError("QJL slice start must be u128-aligned");
        }
        if (start_coord + coord_count > 4096) {
            @compileError("QJL slice is out of bounds");
        }

        const start_word = start_coord / 128;
        return quantized.words[start_word..][0 .. coord_count / 128];
    }

    inline fn qjlMicroWords(quantized: *const U4096) *const [qjl_micro_u128_word_count]u128 {
        return qjlSliceU128Words(0, qjl_micro_coord_count, quantized);
    }

    inline fn qjlMicroToHalfWords(quantized: *const U4096) *const [qjl_micro_u128_word_count]u128 {
        return qjlSliceU128Words(qjl_micro_coord_count, qjl_micro_coord_count, quantized);
    }

    inline fn qjlHalfToFullWords(quantized: *const U4096) *const [qjl_half_to_full_u128_word_count]u128 {
        return qjlSliceU128Words(qjl_half_coord_count, qjl_half_to_full_coord_count, quantized);
    }

    inline fn qjlMismatchU128Loaded(comptime word_count: usize, row_words: *const [word_count]u128, query_vec: @Vector(word_count, u128)) u16 {
        const Words = @Vector(word_count, u128);
        const Counts = @Vector(word_count, u16);
        const row_vec: Words = row_words.*;
        const mismatches: Counts = @intCast(@popCount(row_vec ^ query_vec));
        return @reduce(.Add, mismatches);
    }

    inline fn qjlMismatchU128(comptime word_count: usize, row_words: *const [word_count]u128, query_words: *const [word_count]u128) u16 {
        const Words = @Vector(word_count, u128);
        const query_vec: Words = query_words.*;
        return qjlMismatchU128Loaded(word_count, row_words, query_vec);
    }

    pub fn quantizeQjlLmHead(self: *Quantized) !void {
        std.log.info("Quantizing QJL lm_head", .{});
        self.qjl_inv_dim = 1.0 / @as(f32, @floatFromInt(self.d));
        const data = self.lm_head.data;
        for (0..self.vocab_size) |i| {
            const quantized_row, const row_scale, _ = try self.quantizeQjlVector(data[i * self.d .. (i + 1) * self.d]);
            self.lm_head_qjl[i] = quantized_row;
            self.qjl_micro_rows[i] = qjlMicroWords(&quantized_row).*;
            self.qjl_micro_to_half_rows[i] = qjlMicroToHalfWords(&quantized_row).*;
            self.qjl_half_to_full_rows[i] = qjlHalfToFullWords(&quantized_row).*;
            self.qjl_row_scale[i] = row_scale;
        }
    }

    pub fn precomputePartialDotLmHeadBlocks(self: *Quantized) !void {
        try self.precomputePartialDotLmHeadBlocksWithBlockSize(partial_dot_block_size);
    }

    pub fn precomputePartialDotLmHeadBlocksWithBlockSize(self: *Quantized, comptime block_size: usize) !void {
        if (block_size != partial_dot_block_size) {
            @compileError("partial dot storage is currently specialized for partial_dot_block_size");
        }
        std.log.info("Precomputing partial-dot lm_head blocks block_size={d} blocks={d}", .{ partial_dot_block_size, partial_dot_block_count });

        const data = self.lm_head.data;
        for (0..self.vocab_size) |row_i| {
            const row = data[row_i * self.d ..][0..self.d];
            walshHadamardSimdTo(self.f32_buffer[0..self.d], row, 12);
            for (0..partial_dot_block_count) |block_i| {
                const block_start = block_i * partial_dot_block_size;
                const block_offset = block_i * partial_dot_block_size * self.vocab_size;
                const row_offset = block_offset + row_i * partial_dot_block_size;
                @memcpy(
                    self.partial_dot_blocks[row_offset..][0..partial_dot_block_size],
                    self.f32_buffer[block_start..][0..partial_dot_block_size],
                );
            }
        }
    }

    pub fn quantizeQjlVector(self: *Quantized, vector: []const f32) !struct { U4096, f32, f32 } {
        try self.walshHadamardSimd(vector, 12);
        var l1_norm: f32 = 0.0;
        var l2_norm: f32 = 0.0;
        for (self.f32_buffer) |coef| {
            l1_norm += @abs(coef);
            l2_norm += coef * coef;
        }
        l2_norm = @sqrt(l2_norm);
        var quantized_vector: U4096 = .{};
        for (0..self.d) |i| {
            if (self.f32_buffer[i] > 0) quantized_vector.set(i);
        }
        return .{ quantized_vector, l1_norm * self.qjl_inv_dim, l2_norm };
    }

    pub fn dequantizeQjlVector(self: *Quantized, quantized_vector: *U4096, scale: f32) !void {
        const bytes = std.mem.asBytes(&quantized_vector.words);
        std.debug.assert(self.d == 4096);
        std.debug.assert(bytes.len == 512);
        std.debug.assert(self.f32_buffer.len >= self.d);
        for (bytes, 0..) |byte, byte_i| {
            inline for (0..8) |bit_i| {
                const mask: u8 = @as(u8, 1) << bit_i;
                self.f32_buffer[byte_i * 8 + bit_i] = if ((byte & mask) != 0) scale else -scale;
            }
        }
        try self.walshHadamardSimd(self.f32_buffer[0..self.d], 12);
    }

    pub fn dotQjlSymmetric(self: *Quantized, row: usize, query_quantized: U4096, query_l2_norm: f32) f32 {
        // dot(u,v) approx = ||u|| * ||v|| * sin(pi/2 * dot(q(u), q(v)) / d)
        // dot(q(u), q(v)) = d - 2 * bit_mismatches,
        // XOR returns 1 where bits differ, 0 where they match
        var mismatch_count: u16 = 0;
        mismatch_count += qjlMismatchU128(qjl_micro_u128_word_count, &self.qjl_micro_rows[row], qjlMicroWords(&query_quantized));
        mismatch_count += qjlMismatchU128(qjl_micro_u128_word_count, &self.qjl_micro_to_half_rows[row], qjlMicroToHalfWords(&query_quantized));
        mismatch_count += qjlMismatchU128(qjl_half_to_full_u128_word_count, &self.qjl_half_to_full_rows[row], qjlHalfToFullWords(&query_quantized));
        return self.dotQjlSymmetricFromMismatchCount(row, query_l2_norm, mismatch_count);
    }

    inline fn dotQjlSymmetricFromMismatchCount(self: *Quantized, row: usize, query_l2_norm: f32, mismatch_count: u16) f32 {
        const row_l2_norm = self.lm_head.row_norms[row];
        return query_l2_norm * row_l2_norm * qjlSignDotCoeff(mismatch_count, 4096);
    }

    pub inline fn dotQjlSymmetricWithPrefix(
        self: *Quantized,
        row: usize,
        query_quantized: U4096,
        query_l2_norm: f32,
        prefix_mismatches: u16,
        comptime prefix_coord_count: usize,
    ) f32 {
        if (prefix_coord_count % 128 != 0) {
            @compileError("QJL prefix coordinate count must be a multiple of 128 to reuse u128 prefix popcounts");
        }

        var mismatch_count: u16 = prefix_mismatches;
        if (prefix_coord_count == 0) {
            mismatch_count += qjlMismatchU128(qjl_micro_u128_word_count, &self.qjl_micro_rows[row], qjlMicroWords(&query_quantized));
            mismatch_count += qjlMismatchU128(qjl_micro_u128_word_count, &self.qjl_micro_to_half_rows[row], qjlMicroToHalfWords(&query_quantized));
            mismatch_count += qjlMismatchU128(qjl_half_to_full_u128_word_count, &self.qjl_half_to_full_rows[row], qjlHalfToFullWords(&query_quantized));
        } else if (prefix_coord_count == qjl_micro_coord_count) {
            mismatch_count += qjlMismatchU128(qjl_micro_u128_word_count, &self.qjl_micro_to_half_rows[row], qjlMicroToHalfWords(&query_quantized));
            mismatch_count += qjlMismatchU128(qjl_half_to_full_u128_word_count, &self.qjl_half_to_full_rows[row], qjlHalfToFullWords(&query_quantized));
        } else if (prefix_coord_count == qjl_half_coord_count) {
            mismatch_count += qjlMismatchU128(qjl_half_to_full_u128_word_count, &self.qjl_half_to_full_rows[row], qjlHalfToFullWords(&query_quantized));
        } else {
            @compileError("unsupported QJL prefix coordinate count");
        }
        return self.dotQjlSymmetricFromMismatchCount(row, query_l2_norm, mismatch_count);
    }

    pub inline fn dotQjlSymmetricWithHalfPrefix(
        self: *Quantized,
        row: usize,
        query_quantized: U4096,
        query_l2_norm: f32,
        half_mismatches: u16,
    ) f32 {
        var mismatch_count = half_mismatches;
        mismatch_count += qjlMismatchU128(qjl_half_to_full_u128_word_count, &self.qjl_half_to_full_rows[row], qjlHalfToFullWords(&query_quantized));
        return self.dotQjlSymmetricFromMismatchCount(row, query_l2_norm, mismatch_count);
    }

    pub fn sampleQjlSymmetric(self: *Quantized, query: []const f32) ![top_k_sliced]usize {
        const quantized_query, _, _ = try self.quantizeQjlVector(query);
        var query_l2_norm: f32 = 0.0;
        for (0..self.d) |i| {
            query_l2_norm += query[i] * query[i];
        }
        query_l2_norm = @sqrt(query_l2_norm);

        var top_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var top_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;
        for (0..self.vocab_size) |i| {
            if (self.lm_head.is_junk[i]) continue;
            const score = self.dotQjlSymmetric(i, quantized_query, query_l2_norm);
            if (score > top_scores[top_k_sliced - 1]) {
                var insert_pos = top_k_sliced - 1;
                while (insert_pos > 0 and score > top_scores[insert_pos - 1]) {
                    top_scores[insert_pos] = top_scores[insert_pos - 1];
                    top_rows[insert_pos] = top_rows[insert_pos - 1];
                    insert_pos -= 1;
                }
                top_scores[insert_pos] = score;
                top_rows[insert_pos] = i;
            }
        }
        return top_rows;
    }

    pub fn dotQjlAsymmetric(self: *Quantized, row: usize, query_lut: []const f32, query_total_sum: f32) f32 {
        const row_quantized = &self.lm_head_qjl[row];
        const row_qjl_scale = self.qjl_row_scale[row];

        // dot(q,v) approx = ||rot(v)||_1 / D * dot(rot(q), quant(v))
        // since quant(v) is all ones and minus ones,
        // dot = row_qjl_scale * (sum {rot(q)_i, quant(v)_i == 1} - sum {rot(q)_i, quant(v)_i == 0})
        std.debug.assert(query_lut.len >= 512 * 256);
        var positive_sum: f32 = 0.0;
        const row_bytes = std.mem.asBytes(&row_quantized.words)[0..512];
        for (row_bytes, 0..) |byte, byte_i| {
            const lut_i = byte_i * 256 + @as(usize, byte);
            positive_sum += query_lut[lut_i];
        }

        return row_qjl_scale * (2.0 * positive_sum - query_total_sum);
    }

    pub fn dotQjlAsymmetricSimd(self: *Quantized, row: usize, query_lut: []const f32, query_total_sum: f32) f32 {
        const row_quantized = &self.lm_head_qjl[row];
        const row_qjl_scale = self.qjl_row_scale[row];
        const row_bytes = std.mem.asBytes(&row_quantized.words)[0..512];

        const Vec4 = @Vector(4, f32);
        var acc_vec: Vec4 = @splat(0.0);

        var i: usize = 0;
        while (i < 512) : (i += 4) {
            const b0 = row_bytes[i];
            const b1 = row_bytes[i + 1];
            const b2 = row_bytes[i + 2];
            const b3 = row_bytes[i + 3];

            const lut_i0 = (i) * 256 + @as(usize, b0);
            const lut_i1 = (i + 1) * 256 + @as(usize, b1);
            const lut_i2 = (i + 2) * 256 + @as(usize, b2);
            const lut_i3 = (i + 3) * 256 + @as(usize, b3);

            const loaded_values: Vec4 = .{
                query_lut[lut_i0],
                query_lut[lut_i1],
                query_lut[lut_i2],
                query_lut[lut_i3],
            };

            acc_vec += loaded_values;
        }

        const positive_sum = @reduce(.Add, acc_vec);
        return row_qjl_scale * (2.0 * positive_sum - query_total_sum);
    }

    pub fn precomputeQjlAsymmetricLut(self: *Quantized, rotated_query: []const f32) f32 {
        std.debug.assert(rotated_query.len >= 4096);
        std.debug.assert(self.qjl_query_lut.len >= 512 * 256);

        var query_total_sum: f32 = 0.0;
        for (0..512) |block_i| {
            const coord = 8 * block_i;
            const values = rotated_query[coord..][0..8];
            const lut = self.qjl_query_lut[block_i * 256 ..][0..256];

            lut[0] = 0.0;
            for (values) |value| {
                query_total_sum += value;
            }

            for (1..256) |mask| {
                const previous_mask = mask & (mask - 1);
                const bit_i: usize = @ctz(mask);
                lut[mask] = lut[previous_mask] + values[bit_i];
            }
        }
        return query_total_sum;
    }

    pub fn sampleQjlAsymmetric(self: *Quantized, query: []const f32) ![top_k_sliced]usize {
        try self.walshHadamardSimd(query, 12);
        const query_total_sum = self.precomputeQjlAsymmetricLut(self.f32_buffer);

        var top_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var top_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;
        for (0..self.vocab_size) |i| {
            if (self.lm_head.is_junk[i]) continue;
            const score = self.dotQjlAsymmetricSimd(i, self.qjl_query_lut, query_total_sum);
            if (score > top_scores[top_k_sliced - 1]) {
                var insert_pos = top_k_sliced - 1;
                while (insert_pos > 0 and score > top_scores[insert_pos - 1]) {
                    top_scores[insert_pos] = top_scores[insert_pos - 1];
                    top_rows[insert_pos] = top_rows[insert_pos - 1];
                    insert_pos -= 1;
                }
                top_scores[insert_pos] = score;
                top_rows[insert_pos] = i;
            }
        }
        return top_rows;
    }

    pub fn dotQjlAsymmetric2Bits(self: *Quantized, row: usize, query_msb: *const [512]u8, query_lsb: *const [512]u8, query_scale: f32) f32 {
        const row_bytes = std.mem.asBytes(&self.lm_head_qjl[row].words);
        const scale = self.qjl_row_scale[row] * query_scale;

        // The codebook for 2 bits is -1.51, -0.453, +0.453, +1.51, divided by sqrt(d)
        // A quantized value is then represented as q_i = alpha * sign(msb_i) + beta * sign(lsb_i)
        // alpha = (1.51 + 0.453) / 2 = 0.9815
        // beta  = (1.51 - 0.453) / 2 = 0.5285
        const alpha: f32 = 0.9815;
        const beta: f32 = 0.5285;
        const D: f32 = 4096.0;
        const C: f32 = D * (alpha + beta);

        // Cast the 512-byte arrays to 64-element arrays of u64 to process 64 bits at once
        const row_u128: *const [32]u128 = @ptrCast(@alignCast(row_bytes));
        const msb_u128: *const [32]u128 = @ptrCast(@alignCast(query_msb));
        const lsb_u128: *const [32]u128 = @ptrCast(@alignCast(query_lsb));

        var mismatches_msb: u32 = 0;
        var mismatches_lsb: u32 = 0;

        for (0..32) |i| {
            const v = row_u128[i];
            mismatches_msb += @popCount(msb_u128[i] ^ v);
            mismatches_lsb += @popCount(lsb_u128[i] ^ v);
        }

        const h_m: f32 = @floatFromInt(mismatches_msb);
        const h_l: f32 = @floatFromInt(mismatches_lsb);

        return scale * (C - (2.0 * alpha * h_m) - (2.0 * beta * h_l));
    }

    pub fn quantize2BitsAsymmatric(self: *Quantized, norm: f32) struct { [512]u8, [512]u8 } {
        var msb: [512]u8 = undefined;
        var lsb: [512]u8 = undefined;

        // Scale the threshold for 4096 dimensions (1 / sqrt(4096) = 1/64)
        const unscaled_threshold: f32 = (1.51 + 0.453) / 2.0; // 0.9815
        const threshold: f32 = norm * unscaled_threshold / 64.0;

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

        // Standard loop (processes 8 dimensions per iteration)
        for (0..512) |byte_i| {
            const float_idx = byte_i * 8;
            const floats: Vec8f = self.f32_buffer[float_idx..][0..8].*;

            // ---------------------------------------------------------
            // MSB Logic: 1 if x > 0, else 0
            // ---------------------------------------------------------
            const m_mask = floats > zeros_f32;

            // ---------------------------------------------------------
            // LSB Logic:
            // If x > 0,  LSB = 1 if x > T
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

            msb[byte_i] = @intCast(m_packed);
            lsb[byte_i] = @intCast(l_packed);
        }

        return .{ msb, lsb };
    }

    pub fn sampleQjlAsymmetric2Bits(self: *Quantized, query: []const f32) ![top_k_sliced]usize {
        try self.walshHadamardSimd(query, 12);
        // we compute the L2 norm and pass it to the quantization alg so that it can scale its thesholds
        // this avoids division during a classic normalization, while producing the exact same bits for msb and lsb
        var norm: f32 = 0.0;
        for (self.f32_buffer) |v| {
            norm += v * v;
        }
        norm = @sqrt(norm);
        const query_msb, const query_lsb = self.quantize2BitsAsymmetric(norm);
        const query_scale = scaleFactor2BitsAsymmetric(self.f32_buffer[0..4096], &query_msb, &query_lsb);

        var top_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var top_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;
        for (0..self.vocab_size) |i| {
            if (self.lm_head.is_junk[i]) continue;
            const score = self.dotQjlAsymmetric2Bits(i, &query_msb, &query_lsb, query_scale);
            if (score > top_scores[top_k_sliced - 1]) {
                var insert_pos = top_k_sliced - 1;
                while (insert_pos > 0 and score > top_scores[insert_pos - 1]) {
                    top_scores[insert_pos] = top_scores[insert_pos - 1];
                    top_rows[insert_pos] = top_rows[insert_pos - 1];
                    insert_pos -= 1;
                }
                top_scores[insert_pos] = score;
                top_rows[insert_pos] = i;
            }
        }
        return top_rows;
    }

    pub fn quantize2BitsSymmetric(self: *Quantized, norm: f32) struct { [32]u128, [32]u128 } {
        var msb_words: [32]u128 = undefined;
        var lsb_words: [32]u128 = undefined;
        const msb = std.mem.asBytes(&msb_words);
        const lsb = std.mem.asBytes(&lsb_words);

        // Bits are assigned with the rounded Lloyd codebook
        // [-1.5, -0.5, +0.5, +1.5], whose magnitude threshold is 1.0.
        // The stored/dot-product representation is doubled to [-3, -1, +1, +3]
        // so the hot path stays integer; the fitted scale factors absorb this 2x.
        const unscaled_threshold: f32 = 1.0;
        const threshold: f32 = norm * unscaled_threshold / 64.0;

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

        // Standard loop (processes 8 dimensions per iteration)
        for (0..512) |byte_i| {
            const float_idx = byte_i * 8;
            const floats: Vec8f = self.f32_buffer[float_idx..][0..8].*;

            // ---------------------------------------------------------
            // MSB Logic: 1 if x > 0, else 0
            // ---------------------------------------------------------
            const m_mask = floats > zeros_f32;

            // ---------------------------------------------------------
            // LSB Logic:
            // If x > 0,  LSB = 1 if x > T
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

            msb[byte_i] = @intCast(m_packed);
            lsb[byte_i] = @intCast(l_packed);
        }

        return .{ msb_words, lsb_words };
    }

    fn scaleFactor2BitsWeighted(
        rot_v: *const [4096]f32,
        quant_v_msb: anytype,
        quant_v_lsb: anytype,
        msb_weight: f32,
        lsb_weight: f32,
    ) f32 {
        // scale_factor(v) = dot(rot(v), quant(rot(v))) / norm(quant(rot(v)))²
        const msb = std.mem.asBytes(quant_v_msb);
        const lsb = std.mem.asBytes(quant_v_lsb);
        std.debug.assert(msb.len == 512);
        std.debug.assert(lsb.len == 512);

        var dot: f32 = 0.0;
        var norm_squared: f32 = 0.0;
        for (0..512) |byte_i| {
            const m_byte = msb[byte_i];
            const l_byte = lsb[byte_i];
            inline for (0..8) |bit_i| {
                const mask: u8 = @as(u8, 1) << bit_i;
                const m: f32 = if ((m_byte & mask) != 0) 1.0 else -1.0;
                const l: f32 = if ((l_byte & mask) != 0) 1.0 else -1.0;
                const quant_value = msb_weight * m + lsb_weight * l;
                dot += rot_v[byte_i * 8 + bit_i] * quant_value;
                norm_squared += quant_value * quant_value;
            }
        }

        if (norm_squared == 0.0) return 0.0;
        return dot / norm_squared;
    }

    pub fn scaleFactor2BitsAsymmetric(v: *const [4096]f32, quant_v_msb: anytype, quant_v_lsb: anytype) f32 {
        return scaleFactor2BitsWeighted(v, quant_v_msb, quant_v_lsb, 0.9815, 0.5285);
    }

    pub fn scaleFactor2BitsSymmetric(v: *const [4096]f32, quant_v_msb: anytype, quant_v_lsb: anytype) f32 {
        return scaleFactor2BitsWeighted(v, quant_v_msb, quant_v_lsb, 2.0, 1.0);
    }

    pub fn dequantize2Bits(self: *Quantized, msb: *[512]u8, lsb: *[512]u8, scale: f32) !void {
        std.debug.assert(self.d == 4096);
        std.debug.assert(self.f32_buffer.len >= self.d);
        for (0..512) |byte_i| {
            const m_byte = msb[byte_i];
            const l_byte = lsb[byte_i];
            inline for (0..8) |bit_i| {
                const mask: u8 = @as(u8, 1) << bit_i;
                const m: f32 = if ((m_byte & mask) != 0) 1.0 else -1.0;
                const l: f32 = if ((l_byte & mask) != 0) 1.0 else -1.0;
                self.f32_buffer[byte_i * 8 + bit_i] = scale * (2.0 * m + l);
            }
        }
        try self.walshHadamardSimd(self.f32_buffer[0..self.d], 12);
    }

    pub fn quantizeQjlLmHead2Bits(self: *Quantized) !void {
        std.log.info("Quantizing QJL lm_head 2 bits", .{});
        const data = self.lm_head.data;
        for (0..self.vocab_size) |i| {
            const v = data[i * self.d .. (i + 1) * self.d];
            try self.walshHadamardSimd(v, 12);
            const msb, const lsb = self.quantize2BitsSymmetric(self.lm_head.row_norms[i]);
            @memcpy(self.lm_head_qjl_2bits[i][0..32], &msb);
            @memcpy(self.lm_head_qjl_2bits[i][32..64], &lsb);
            self.qjl_row_scale_2bits[i] = scaleFactor2BitsSymmetric(self.f32_buffer[0..4096], &msb, &lsb);
        }
    }

    pub inline fn dotSymmetric2Bit(q_msb: *const [32]u128, q_lsb: *const [32]u128, row_msb: *const [32]u128, row_lsb: *const [32]u128) i32 {
        var h_mm: u16 = 0;
        var h_ml: u16 = 0;
        var h_lm: u16 = 0;
        var h_ll: u16 = 0;

        for (0..32) |i| {
            const qm = q_msb[i];
            const ql = q_lsb[i];
            const vm = row_msb[i];
            const vl = row_lsb[i];

            h_mm += @popCount(qm ^ vm);
            h_ml += @popCount(qm ^ vl);
            h_lm += @popCount(ql ^ vm);
            h_ll += @popCount(ql ^ vl);
        }

        const term_mm: i32 = @as(i32, @intCast(h_mm)) << 3;
        const term_cross: i32 = @as(i32, @intCast(h_ml + h_lm)) << 2;
        const term_ll: i32 = @as(i32, @intCast(h_ll)) << 1;

        return 36864 - term_mm - term_cross - term_ll;
    }

    pub inline fn dotQjlSymmetric2Bits(self: *Quantized, row: usize, query_msb: *const [32]u128, query_lsb: *const [32]u128) i32 {
        const row_msb = self.lm_head_qjl_2bits[row][0..32];
        const row_lsb = self.lm_head_qjl_2bits[row][32..64];
        const dot = dotSymmetric2Bit(query_msb, query_lsb, row_msb, row_lsb);
        return dot;
    }

    pub fn sampleQjlSymmetric2Bits(self: *Quantized, query: []const f32) ![top_k_sliced]usize {
        try self.walshHadamardSimd(query, 12);
        // we compute the L2 norm and pass it to the quantization alg so that it can scale its thesholds
        // this avoids division during a classic normalization, while producing the exact same bits for msb and lsb
        var norm: f32 = 0.0;
        for (self.f32_buffer) |v| {
            norm += v * v;
        }
        norm = @sqrt(norm);

        const query_msb, const query_lsb = self.quantize2BitsSymmetric(norm);
        const query_scale = scaleFactor2BitsSymmetric(self.f32_buffer[0..4096], &query_msb, &query_lsb);

        var top_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var top_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;
        for (0..self.vocab_size) |i| {
            if (self.lm_head.is_junk[i]) continue;
            const dot: f32 = @floatFromInt(self.dotQjlSymmetric2Bits(i, &query_msb, &query_lsb));
            const score = query_scale * self.qjl_row_scale_2bits[i] * dot;
            if (score > top_scores[top_k_sliced - 1]) {
                var insert_pos = top_k_sliced - 1;
                while (insert_pos > 0 and score > top_scores[insert_pos - 1]) {
                    top_scores[insert_pos] = top_scores[insert_pos - 1];
                    top_rows[insert_pos] = top_rows[insert_pos - 1];
                    insert_pos -= 1;
                }
                top_scores[insert_pos] = score;
                top_rows[insert_pos] = i;
            }
        }
        return top_rows;
    }

    pub fn testQjlReconstructionError(self: *Quantized) !void {
        const RowErrors = struct {
            abs_1bit: f32,
            rel_1bit: f32,
            abs_2bits: f32,
            rel_2bits: f32,

            fn compute(quantizer: *Quantized, row_i: usize) !@This() {
                const row = quantizer.lm_head.data[row_i * quantizer.d .. (row_i + 1) * quantizer.d];
                const row_norm = quantizer.lm_head.row_norms[row_i];

                try quantizer.dequantizeQjlVector(&quantizer.lm_head_qjl[row_i], quantizer.qjl_row_scale[row_i]);
                var err2_1bit: f32 = 0.0;
                var i: usize = 0;
                while (i < quantizer.d) : (i += 1) {
                    const diff = quantizer.f32_buffer[i] - row[i];
                    err2_1bit += diff * diff;
                }
                const abs_1bit = @sqrt(err2_1bit);
                const rel_1bit = if (row_norm == 0.0) 0.0 else abs_1bit / row_norm;

                const packed_bytes = std.mem.sliceAsBytes(quantizer.lm_head_qjl_2bits[row_i][0..64]);
                var msb: [512]u8 = undefined;
                var lsb: [512]u8 = undefined;
                @memcpy(msb[0..], packed_bytes[0..512]);
                @memcpy(lsb[0..], packed_bytes[512..1024]);

                try quantizer.dequantize2Bits(&msb, &lsb, quantizer.qjl_row_scale_2bits[row_i]);
                var err2_2bits: f32 = 0.0;
                i = 0;
                while (i < quantizer.d) : (i += 1) {
                    const diff = quantizer.f32_buffer[i] - row[i];
                    err2_2bits += diff * diff;
                }
                const abs_2bits = @sqrt(err2_2bits);
                const rel_2bits = if (row_norm == 0.0) 0.0 else abs_2bits / row_norm;

                return .{
                    .abs_1bit = abs_1bit,
                    .rel_1bit = rel_1bit,
                    .abs_2bits = abs_2bits,
                    .rel_2bits = rel_2bits,
                };
            }
        };

        try self.quantizeQjlLmHead();
        try self.quantizeQjlLmHead2Bits();

        var prng = std.Random.DefaultPrng.init(0);
        const random = prng.random();

        var total_abs_1bit: f32 = 0.0;
        var total_rel_1bit: f32 = 0.0;
        var total_abs_2bits: f32 = 0.0;
        var total_rel_2bits: f32 = 0.0;

        std.log.info("QJL reconstruction error sample:", .{});
        std.log.info("{s:>4} {s:>10} {s:>14} {s:>14} {s:>14} {s:>14} {s:>14}", .{ "rank", "row", "row_norm", "abs_1bit", "rel_1bit", "abs_2bits", "rel_2bits" });

        var sample_i: usize = 0;
        while (sample_i < 100) : (sample_i += 1) {
            const row_i = random.uintLessThan(usize, self.vocab_size);
            const row_norm = self.lm_head.row_norms[row_i];
            const errors = try RowErrors.compute(self, row_i);

            total_abs_1bit += errors.abs_1bit;
            total_rel_1bit += errors.rel_1bit;
            total_abs_2bits += errors.abs_2bits;
            total_rel_2bits += errors.rel_2bits;

            std.log.info("{d:>4} {d:>10} {d:>14.6} {d:>14.6} {d:>14.6} {d:>14.6} {d:>14.6}", .{
                sample_i + 1,
                row_i,
                row_norm,
                errors.abs_1bit,
                errors.rel_1bit,
                errors.abs_2bits,
                errors.rel_2bits,
            });
        }

        const inv_count: f32 = 1.0 / 100.0;
        std.log.info("QJL reconstruction error avg: abs_1bit={d:.6} rel_1bit={d:.6} abs_2bits={d:.6} rel_2bits={d:.6}", .{
            total_abs_1bit * inv_count,
            total_rel_1bit * inv_count,
            total_abs_2bits * inv_count,
            total_rel_2bits * inv_count,
        });

        var min_abs_1bit: f32 = std.math.inf(f32);
        var max_abs_1bit: f32 = 0.0;
        var min_rel_1bit: f32 = std.math.inf(f32);
        var max_rel_1bit: f32 = 0.0;
        var min_abs_2bits: f32 = std.math.inf(f32);
        var max_abs_2bits: f32 = 0.0;
        var min_rel_2bits: f32 = std.math.inf(f32);
        var max_rel_2bits: f32 = 0.0;

        var row_i: usize = 0;
        while (row_i < self.vocab_size) : (row_i += 1) {
            const errors = try RowErrors.compute(self, row_i);
            min_abs_1bit = @min(min_abs_1bit, errors.abs_1bit);
            max_abs_1bit = @max(max_abs_1bit, errors.abs_1bit);
            min_rel_1bit = @min(min_rel_1bit, errors.rel_1bit);
            max_rel_1bit = @max(max_rel_1bit, errors.rel_1bit);
            min_abs_2bits = @min(min_abs_2bits, errors.abs_2bits);
            max_abs_2bits = @max(max_abs_2bits, errors.abs_2bits);
            min_rel_2bits = @min(min_rel_2bits, errors.rel_2bits);
            max_rel_2bits = @max(max_rel_2bits, errors.rel_2bits);
        }

        std.log.info("QJL reconstruction error full rows:", .{});
        std.log.info("1-bit abs min={d:.6} max={d:.6} rel min={d:.6} max={d:.6}", .{ min_abs_1bit, max_abs_1bit, min_rel_1bit, max_rel_1bit });
        std.log.info("2-bit abs min={d:.6} max={d:.6} rel min={d:.6} max={d:.6}", .{ min_abs_2bits, max_abs_2bits, min_rel_2bits, max_rel_2bits });
    }

    pub fn sampleDense(self: *Quantized, query: []const f32) DenseSample {
        var top_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var top_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;

        var i: usize = 0;
        while (i < self.vocab_size) : (i += 1) {
            if (self.lm_head.is_junk[i]) {
                self.dense_logits[i] = -std.math.floatMax(f32);
                continue;
            }
            const score = self.realLogit(query, self.lm_head.data[i * self.d .. (i + 1) * self.d]);
            self.dense_logits[i] = score;

            if (score > top_scores[top_k_sliced - 1]) {
                var insert_pos = top_k_sliced - 1;
                while (insert_pos > 0 and score > top_scores[insert_pos - 1]) {
                    top_scores[insert_pos] = top_scores[insert_pos - 1];
                    top_rows[insert_pos] = top_rows[insert_pos - 1];
                    insert_pos -= 1;
                }
                top_scores[insert_pos] = score;
                top_rows[insert_pos] = i;
            }
        }

        const simd_len = 16;
        const Vec = @Vector(simd_len, f32);
        var max_vec: Vec = @splat(-std.math.inf(f32));
        var total_vec: Vec = @splat(0.0);

        i = 0;
        while (i + simd_len <= self.vocab_size) : (i += simd_len) {
            const logits: Vec = self.dense_logits[i..][0..simd_len].*;
            const new_max = @max(max_vec, logits);
            total_vec = total_vec * @exp(max_vec - new_max) + @exp(logits - new_max);
            max_vec = new_max;
        }

        var max_score = @reduce(.Max, max_vec);
        var total: f32 = @reduce(.Add, total_vec * @exp(max_vec - @as(Vec, @splat(max_score))));

        while (i < self.vocab_size) : (i += 1) {
            const score = self.dense_logits[i];
            if (score > max_score) {
                total = total * @exp(max_score - score) + 1.0;
                max_score = score;
            } else {
                total += @exp(score - max_score);
            }
        }

        var top_probas: [top_k_sliced]f32 = undefined;
        i = 0;
        while (i < top_k_sliced) : (i += 1) {
            top_probas[i] = @exp(top_scores[i] - max_score) / total;
        }

        return .{ .rows = top_rows, .logits = top_scores, .probas = top_probas, .max_logit = max_score, .total_exp = total };
    }

    fn log2PhaseRows(self: *Quantized, label: []const u8, title: []const u8, logits: []const Logit, count: usize, tokenizer: *Tokenizer) !void {
        std.log.info("{s} {s}:", .{ label, title });
        std.log.info("{s:>4} {s:>10} {s:>14} {s:>14} {s:>14} {s:>14}  {s}", .{ "rank", "token", "approx", "lower", "upper", "real", "text" });
        var rank: usize = 0;
        while (rank < count) : (rank += 1) {
            const logit = logits[rank];
            const token_str = try tokens.tokenString(tokenizer.*, logit.index, self.allocator);
            defer self.allocator.free(token_str);
            std.log.info("{d:>4} {d:>10} {d:>14.6} {d:>14.6} {d:>14.6} {d:>14.6}  {s}", .{
                rank + 1,
                logit.index,
                logit.logit_qjl,
                logit.lower_bound,
                logit.upper_bound,
                logit.logit_f32,
                token_str,
            });
        }
    }

    fn log2PhaseBounds(self: *Quantized, label: []const u8, logits: []Logit, tokenizer: *Tokenizer) !void {
        std.mem.sort(Logit, logits, {}, Logit.decreasingQjlOrder);

        var best_lower_bound: f32 = -std.math.inf(f32);
        for (logits) |logit| {
            best_lower_bound = @max(best_lower_bound, logit.lower_bound);
        }
        var pruned_count: usize = 0;
        for (logits) |logit| {
            if (logit.upper_bound < best_lower_bound) pruned_count += 1;
        }

        try self.log2PhaseRows(label, "2-phase approx top16", logits, top_k_sliced, tokenizer);

        std.mem.sort(Logit, logits, {}, Logit.boundPressureOrder);
        try self.log2PhaseRows(label, "2-phase closest/exceeded bounds top10", logits, 10, tokenizer);

        std.mem.sort(Logit, logits, {}, Logit.decreasingOrder);
        try self.log2PhaseRows(label, "2-phase true top16", logits, top_k_sliced, tokenizer);

        std.log.info("{s} 2-phase prune: best_lower_bound={d:.6} pruned={d}/{d}", .{ label, best_lower_bound, pruned_count, logits.len });
    }

    fn log2PhaseSeparator(self: *Quantized, true_token: usize, tokenizer: *Tokenizer) !void {
        const token_str = try tokens.tokenString(tokenizer.*, true_token, self.allocator);
        defer self.allocator.free(token_str);
        std.log.info("\n********** Token = {d} {s}", .{ true_token, token_str });
    }

    pub fn sample2Phase1BitLog(self: *Quantized, query: []const f32, tokenizer: *Tokenizer) !void {
        const logits = try self.allocator.alloc(Logit, self.vocab_size);
        defer self.allocator.free(logits);

        const quantized_query, _, const query_l2_norm = try self.quantizeQjlVector(query);
        const z_score: f32 = 5.0;
        const err_const: f32 = 0.92;

        var i: usize = 0;
        while (i < self.vocab_size) : (i += 1) {
            const row = self.lm_head.data[i * self.d .. (i + 1) * self.d];
            const approx = self.dotQjlSymmetric(i, quantized_query, query_l2_norm);
            const err = z_score * err_const * query_l2_norm * self.lm_head.row_norms[i] * self.d_inv_norm;
            logits[i] = .{
                .index = i,
                .logit_f32 = self.realLogit(query, row),
                .logit_i8 = 0.0,
                .logit_sliced = 0.0,
                .logit_qjl = approx,
                .lower_bound = approx - err,
                .upper_bound = approx + err,
                .proba = 0.0,
                .quantized_rank = 0,
                .sliced_rank = 0,
                .qjl_rank = 0,
            };
        }

        std.mem.sort(Logit, logits, {}, Logit.decreasingOrder);
        try self.log2PhaseSeparator(logits[0].index, tokenizer);
        try self.log2PhaseBounds("1-bit", logits, tokenizer);
    }

    pub fn sample2Phase2BitsLog(self: *Quantized, query: []const f32, tokenizer: *Tokenizer) !void {
        const logits = try self.allocator.alloc(Logit, self.vocab_size);
        defer self.allocator.free(logits);

        try self.walshHadamardSimd(query, 12);
        var query_l2_norm: f32 = 0.0;
        for (self.f32_buffer) |v| {
            query_l2_norm += v * v;
        }
        query_l2_norm = @sqrt(query_l2_norm);

        const query_msb, const query_lsb = self.quantize2BitsSymmetric(query_l2_norm);
        const query_scale = scaleFactor2BitsSymmetric(self.f32_buffer[0..4096], &query_msb, &query_lsb);
        const z_score: f32 = 5.0;
        const err_const: f32 = 0.49;

        var i: usize = 0;
        while (i < self.vocab_size) : (i += 1) {
            const row = self.lm_head.data[i * self.d .. (i + 1) * self.d];
            const dot: f32 = @floatFromInt(self.dotQjlSymmetric2Bits(i, &query_msb, &query_lsb));
            const approx = query_scale * self.qjl_row_scale_2bits[i] * dot;
            const err = z_score * err_const * query_l2_norm * self.lm_head.row_norms[i] * self.d_inv_norm;
            logits[i] = .{
                .index = i,
                .logit_f32 = self.realLogit(query, row),
                .logit_i8 = 0.0,
                .logit_sliced = 0.0,
                .logit_qjl = approx,
                .lower_bound = approx - err,
                .upper_bound = approx + err,
                .proba = 0.0,
                .quantized_rank = 0,
                .sliced_rank = 0,
                .qjl_rank = 0,
            };
        }

        std.mem.sort(Logit, logits, {}, Logit.decreasingOrder);
        try self.log2PhaseSeparator(logits[0].index, tokenizer);
        try self.log2PhaseBounds("2-bit", logits, tokenizer);
    }

    inline fn insertTopK(comptime k: usize, row: usize, score: f32, rows: *[k]usize, scores: *[k]f32) void {
        if (score <= scores[k - 1]) return;

        var insert_pos = k - 1;
        while (insert_pos > 0 and score > scores[insert_pos - 1]) {
            scores[insert_pos] = scores[insert_pos - 1];
            rows[insert_pos] = rows[insert_pos - 1];
            insert_pos -= 1;
        }
        scores[insert_pos] = score;
        rows[insert_pos] = row;
    }

    inline fn insertTop16(row: usize, score: f32, rows: *[top_k_sliced]usize, scores: *[top_k_sliced]f32) void {
        insertTopK(top_k_sliced, row, score, rows, scores);
    }

    inline fn insertTop8(row: usize, score: f32, rows: *[8]usize, scores: *[8]f32) void {
        insertTopK(8, row, score, rows, scores);
    }

    inline fn inTop16(row: usize, rows: *const [top_k_sliced]usize) bool {
        for (rows) |top_row| {
            if (top_row == row) return true;
        }
        return false;
    }

    pub fn sample2Phase1Bit(self: *Quantized, query: []const f32) !TwoPhaseSample {
        var approx_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var approx_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;

        if (true) return .{ .rows = approx_rows, .nb_scored = 0 };

        const quantized_query, _, const query_l2_norm = try self.quantizeQjlVector(query);
        const z_score: f32 = 5.0;
        const err_const: f32 = 0.92;
        var best_lower_bound: f32 = -std.math.inf(f32);

        // Phase 1: score every row with the 1-bit QJL approximation and store an upper bound.
        var i: usize = 0;
        while (i < self.vocab_size) : (i += 1) {
            const approx = self.dotQjlSymmetric(i, quantized_query, query_l2_norm);
            const err = z_score * err_const * query_l2_norm * self.lm_head.row_norms[i] * self.d_inv_norm;
            const lower_bound = approx - err;
            self.dense_logits[i] = approx + err;
            best_lower_bound = @max(best_lower_bound, lower_bound);
            insertTop16(i, approx, &approx_rows, &approx_scores);
        }

        var top_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var top_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;
        var nb_scored: usize = 0;
        var prune_threshold = best_lower_bound;

        // Phase 2a: exact-score the approximate top16 to tighten the pruning threshold.
        i = 0;
        while (i < top_k_sliced) : (i += 1) {
            const row_i = approx_rows[i];
            const row = self.lm_head.data[row_i * self.d .. (row_i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            prune_threshold = @max(prune_threshold, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
            self.dense_logits[row_i] = -std.math.inf(f32);
        }

        // Phase 2b: exact-score only rows whose 1-bit upper bound can still beat the threshold.
        i = 0;
        while (i < self.vocab_size) : (i += 1) {
            if (self.dense_logits[i] < prune_threshold) continue;
            const row = self.lm_head.data[i * self.d .. (i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            prune_threshold = @max(prune_threshold, real);
            insertTop16(i, real, &top_rows, &top_scores);
        }
        return .{ .rows = top_rows, .nb_scored = nb_scored };
    }

    pub fn sample2Phase2Bits(self: *Quantized, query: []const f32) !TwoPhaseSample {
        var approx_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var approx_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;

        if (true) return .{ .rows = approx_rows, .nb_scored = 0 };

        try self.walshHadamardSimd(query, 12);
        var query_l2_norm: f32 = 0.0;
        for (self.f32_buffer) |v| {
            query_l2_norm += v * v;
        }
        query_l2_norm = @sqrt(query_l2_norm);

        const query_msb, const query_lsb = self.quantize2BitsSymmetric(query_l2_norm);
        const query_scale = scaleFactor2BitsSymmetric(self.f32_buffer[0..4096], &query_msb, &query_lsb);
        const z_score: f32 = 5.0;
        const err_const: f32 = 0.49;
        var best_lower_bound: f32 = -std.math.inf(f32);

        // Phase 1: score every row with the 2-bit QJL approximation and store an upper bound.
        var i: usize = 0;
        while (i < self.vocab_size) : (i += 1) {
            const dot: f32 = @floatFromInt(self.dotQjlSymmetric2Bits(i, &query_msb, &query_lsb));
            const approx = query_scale * self.qjl_row_scale_2bits[i] * dot;
            const err = z_score * err_const * query_l2_norm * self.lm_head.row_norms[i] * self.d_inv_norm;
            const lower_bound = approx - err;
            self.dense_logits[i] = approx + err;
            best_lower_bound = @max(best_lower_bound, lower_bound);
            insertTop16(i, approx, &approx_rows, &approx_scores);
        }

        var top_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var top_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;
        var nb_scored: usize = 0;
        var prune_threshold = best_lower_bound;

        // Phase 2a: exact-score the approximate top16 to tighten the pruning threshold.
        i = 0;
        while (i < top_k_sliced) : (i += 1) {
            const row_i = approx_rows[i];
            const row = self.lm_head.data[row_i * self.d .. (row_i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            prune_threshold = @max(prune_threshold, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
            self.dense_logits[row_i] = -std.math.inf(f32);
        }

        // Phase 2b: exact-score only rows whose 2-bit upper bound can still beat the threshold.
        i = 0;
        while (i < self.vocab_size) : (i += 1) {
            if (self.dense_logits[i] < prune_threshold) continue;
            const row = self.lm_head.data[i * self.d .. (i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            prune_threshold = @max(prune_threshold, real);
            insertTop16(i, real, &top_rows, &top_scores);
        }
        return .{ .rows = top_rows, .nb_scored = nb_scored };
    }

    pub fn sample3Phases(self: *Quantized, query: []const f32) !TwoPhaseSample {
        var phase1_rows: [8]usize = [_]usize{0} ** 8;
        var phase1_scores: [8]f32 = [_]f32{-std.math.inf(f32)} ** 8;
        var top_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var top_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;

        if (true) return .{ .rows = top_rows, .nb_scored = 0, .nb_2bit_scored = 0 };

        const quantized_query, _, const query_l2_norm = try self.quantizeQjlVector(query);
        const z_score: f32 = 5.0;
        const err_const_1bit: f32 = 0.92;
        const err_const_2bits: f32 = 0.49;
        var best_lower_bound: f32 = -std.math.inf(f32);

        // Phase 1: score all rows with full 1-bit QJL and keep only loose upper bounds.
        var i: usize = 0;
        while (i < self.vocab_size) : (i += 1) {
            const approx = self.dotQjlSymmetric(i, quantized_query, query_l2_norm);
            const err = z_score * err_const_1bit * query_l2_norm * self.lm_head.row_norms[i] * self.d_inv_norm;
            self.dense_logits[i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTop8(i, approx, &phase1_rows, &phase1_scores);
        }

        var nb_scored: usize = 0;
        // Phase 1b: exact-score the 1-bit top8 to raise the global lower bound.
        i = 0;
        while (i < 8) : (i += 1) {
            if (phase1_scores[i] == -std.math.inf(f32)) break;
            const row_i = phase1_rows[i];
            const row = self.lm_head.data[row_i * self.d .. (row_i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
            self.dense_logits[row_i] = -std.math.inf(f32);
        }

        const query_msb, const query_lsb = self.quantize2BitsSymmetric(query_l2_norm);
        const query_scale = scaleFactor2BitsSymmetric(self.f32_buffer[0..4096], &query_msb, &query_lsb);
        var phase2_rows: [8]usize = [_]usize{0} ** 8;
        var phase2_scores: [8]f32 = [_]f32{-std.math.inf(f32)} ** 8;
        var nb_2bit_scored: usize = 0;

        // Phase 2: rescore rows surviving the 1-bit bound with the tighter 2-bit QJL approximation.
        i = 0;
        while (i < self.vocab_size) : (i += 1) {
            if (self.dense_logits[i] < best_lower_bound) {
                self.dense_logits[i] = -std.math.inf(f32);
                continue;
            }

            nb_2bit_scored += 1;
            const dot: f32 = @floatFromInt(self.dotQjlSymmetric2Bits(i, &query_msb, &query_lsb));
            const approx = query_scale * self.qjl_row_scale_2bits[i] * dot;
            const err = z_score * err_const_2bits * query_l2_norm * self.lm_head.row_norms[i] * self.d_inv_norm;
            self.dense_logits[i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTop8(i, approx, &phase2_rows, &phase2_scores);
        }

        // Phase 2b: exact-score the 2-bit top8 to tighten the final dense threshold.
        i = 0;
        while (i < 8) : (i += 1) {
            if (phase2_scores[i] == -std.math.inf(f32)) break;
            const row_i = phase2_rows[i];
            const row = self.lm_head.data[row_i * self.d .. (row_i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
            self.dense_logits[row_i] = -std.math.inf(f32);
        }

        // Phase 3: exact-score the remaining rows whose 2-bit upper bound still survives.
        i = 0;
        while (i < self.vocab_size) : (i += 1) {
            if (self.dense_logits[i] < best_lower_bound) continue;
            const row = self.lm_head.data[i * self.d .. (i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(i, real, &top_rows, &top_scores);
        }

        return .{ .rows = top_rows, .nb_scored = nb_scored, .nb_2bit_scored = nb_2bit_scored };
    }

    pub fn sample4Phases(self: *Quantized, query: []const f32) !TwoPhaseSample {
        std.debug.assert(query.len == self.d);
        const prefix_coord_count = 4;

        var prefix_coords: [prefix_coord_count]usize = [_]usize{0} ** prefix_coord_count;
        var prefix_scores: [prefix_coord_count]f32 = [_]f32{-std.math.inf(f32)} ** prefix_coord_count;
        var phase1_rows: [8]usize = [_]usize{0} ** 8;
        var phase1_scores: [8]f32 = [_]f32{-std.math.inf(f32)} ** 8;
        var top_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var top_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;

        if (true) return .{ .rows = top_rows, .nb_scored = 0, .nb_2bit_scored = 0 };

        var coord: usize = 0;
        // Phase 0a: find the largest raw query coordinates for exact prefix scoring.
        while (coord < self.d) : (coord += 1) {
            insertTopK(prefix_coord_count, coord, @abs(query[coord]), &prefix_coords, &prefix_scores);
        }

        @memset(self.partial_logits, 0.0);
        const simd_len = 16;
        const Vec = @Vector(simd_len, f32);

        var prefix_i: usize = 0;
        // Phase 0b: compute the exact lm_head dot contribution from the raw coordinate prefix.
        while (prefix_i < prefix_coord_count) : (prefix_i += 1) {
            if (prefix_scores[prefix_i] == -std.math.inf(f32)) break;
            const query_coord = prefix_coords[prefix_i];
            const query_value = query[query_coord];
            const coord_values = self.lm_head.data_t[query_coord * self.vocab_size ..][0..self.vocab_size];
            const query_vec: Vec = @splat(query_value);

            var row_i: usize = 0;
            while (row_i + simd_len <= self.vocab_size) : (row_i += simd_len) {
                const coord_vec: Vec = coord_values[row_i..][0..simd_len].*;
                var logits_vec: Vec = self.partial_logits[row_i..][0..simd_len].*;
                logits_vec = @mulAdd(Vec, coord_vec, query_vec, logits_vec);
                self.partial_logits[row_i..][0..simd_len].* = logits_vec;
            }
            while (row_i < self.vocab_size) : (row_i += 1) {
                self.partial_logits[row_i] += coord_values[row_i] * query_value;
            }
        }

        // Phase 0c: zero the exact prefix coordinates before quantizing the residual query.
        @memcpy(self.f32_buffer[0..self.d], query);
        prefix_i = 0;
        while (prefix_i < prefix_coord_count) : (prefix_i += 1) {
            if (prefix_scores[prefix_i] == -std.math.inf(f32)) break;
            self.f32_buffer[prefix_coords[prefix_i]] = 0.0;
        }

        const quantized_query, _, const query_l2_norm = try self.quantizeQjlVector(self.f32_buffer[0..self.d]);
        const z_score: f32 = 5.0;
        const err_const_1bit: f32 = 0.92;
        const err_const_2bits: f32 = 0.49;
        var best_lower_bound: f32 = -std.math.inf(f32);

        // Phase 1: score all rows as exact prefix plus full 1-bit QJL residual bounds.
        var i: usize = 0;
        while (i < self.vocab_size) : (i += 1) {
            const approx = self.partial_logits[i] + self.dotQjlSymmetric(i, quantized_query, query_l2_norm);
            const err = z_score * err_const_1bit * query_l2_norm * self.lm_head.row_norms[i] * self.d_inv_norm;
            self.dense_logits[i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTop8(i, approx, &phase1_rows, &phase1_scores);
        }

        var nb_scored: usize = 0;
        // Phase 1b: exact-score the 1-bit top8 to tighten the lower bound.
        i = 0;
        while (i < 8) : (i += 1) {
            if (phase1_scores[i] == -std.math.inf(f32)) break;
            const row_i = phase1_rows[i];
            const row = self.lm_head.data[row_i * self.d .. (row_i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
            self.dense_logits[row_i] = -std.math.inf(f32);
        }

        // Phase 1c: compact rows whose 1-bit upper bound survives into the 2-bit phase.
        var active_count: usize = 0;
        i = 0;
        while (i < self.vocab_size) : (i += 1) {
            if (self.dense_logits[i] < best_lower_bound) {
                self.dense_logits[i] = -std.math.inf(f32);
                continue;
            }
            self.active_rows[active_count] = @intCast(i);
            active_count += 1;
        }

        const query_msb, const query_lsb = self.quantize2BitsSymmetric(query_l2_norm);
        const query_scale = scaleFactor2BitsSymmetric(self.f32_buffer[0..4096], &query_msb, &query_lsb);
        var phase2_rows: [8]usize = [_]usize{0} ** 8;
        var phase2_scores: [8]f32 = [_]f32{-std.math.inf(f32)} ** 8;
        var nb_2bit_scored: usize = 0;

        // Phase 2: rescore surviving rows as exact prefix plus 2-bit QJL residual bounds.
        i = 0;
        while (i < active_count) : (i += 1) {
            const row_i: usize = @intCast(self.active_rows[i]);
            nb_2bit_scored += 1;
            const dot: f32 = @floatFromInt(self.dotQjlSymmetric2Bits(row_i, &query_msb, &query_lsb));
            const approx = self.partial_logits[row_i] + query_scale * self.qjl_row_scale_2bits[row_i] * dot;
            const err = z_score * err_const_2bits * query_l2_norm * self.lm_head.row_norms[row_i] * self.d_inv_norm;
            self.dense_logits[row_i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTop8(row_i, approx, &phase2_rows, &phase2_scores);
        }

        // Phase 2b: exact-score the 2-bit top8 to tighten the final dense threshold.
        i = 0;
        while (i < 8) : (i += 1) {
            if (phase2_scores[i] == -std.math.inf(f32)) break;
            const row_i = phase2_rows[i];
            const row = self.lm_head.data[row_i * self.d .. (row_i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
            self.dense_logits[row_i] = -std.math.inf(f32);
        }

        // Phase 3: exact-score remaining rows whose 2-bit upper bound still survives.
        i = 0;
        while (i < active_count) : (i += 1) {
            const row_i: usize = @intCast(self.active_rows[i]);
            if (self.dense_logits[row_i] < best_lower_bound) continue;
            const row = self.lm_head.data[row_i * self.d .. (row_i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
        }

        return .{ .rows = top_rows, .nb_scored = nb_scored, .nb_2bit_scored = nb_2bit_scored };
    }

    inline fn qjlSignDotCoeff(mismatch_count: u16, comptime coord_count: usize) f32 {
        const mismatch_i: usize = @intCast(mismatch_count);
        std.debug.assert(mismatch_i <= coord_count);
        return switch (coord_count) {
            qjl_micro_coord_count => qjl_micro_sign_dot_lut[mismatch_i],
            qjl_half_coord_count => qjl_half_sign_dot_lut[mismatch_i],
            4096 => qjl_full_sign_dot_lut[mismatch_i],
            else => @compileError("unsupported QJL sign-dot LUT size"),
        };
    }

    inline fn qjlPartialSignDot(query_l2_norm: f32, row_norm: f32, mismatch_count: u16, comptime coord_count: usize) f32 {
        return query_l2_norm * row_norm * qjlSignDotCoeff(mismatch_count, coord_count);
    }

    inline fn resetTopK(comptime k: usize, rows: *[k]usize, scores: *[k]f32) void {
        rows.* = [_]usize{0} ** k;
        scores.* = [_]f32{-std.math.inf(f32)} ** k;
    }

    inline fn dotPartialDotBlock(
        query_block: *const [partial_dot_block_size]f32,
        row_block: *const [partial_dot_block_size]f32,
    ) f32 {
        const simd_len = 16;
        const Vec = @Vector(simd_len, f32);
        var acc: Vec = @splat(0);
        var i: usize = 0;
        while (i + simd_len <= partial_dot_block_size) : (i += simd_len) {
            const query_v: Vec = query_block[i..][0..simd_len].*;
            const row_v: Vec = row_block[i..][0..simd_len].*;
            acc = @mulAdd(Vec, query_v, row_v, acc);
        }
        return @reduce(.Add, acc);
    }

    inline fn exactScorePhaseTopK(
        comptime k: usize,
        self: *Quantized,
        query: []const f32,
        phase_rows: *const [k]usize,
        phase_scores: *const [k]f32,
        top_rows: *[top_k_sliced]usize,
        top_scores: *[top_k_sliced]f32,
        best_lower_bound: *f32,
        nb_scored: *usize,
    ) void {
        self.zml_handler.tic(&self.zml_handler.timers.quant_5p_dense_top8);
        defer self.zml_handler.toc(&self.zml_handler.timers.quant_5p_dense_top8);
        for (0..k) |top_i| {
            if (phase_scores[top_i] == -std.math.inf(f32)) break;
            const row_i = phase_rows[top_i];
            if (self.lm_head.is_junk[row_i]) continue;
            if (self.dense_logits[row_i] == -std.math.inf(f32)) continue;
            const row = self.lm_head.data[row_i * self.d ..][0..self.d];
            const real = self.realLogit(query, row);
            nb_scored.* += 1;
            best_lower_bound.* = @max(best_lower_bound.*, real);
            insertTop16(row_i, real, top_rows, top_scores);
            self.dense_logits[row_i] = -std.math.inf(f32);
        }
    }

    pub fn sampleMultiPhasePartialDotEstimator(self: *Quantized, query: []const f32) !TwoPhaseSample {
        return self.sampleMultiPhasePartialDot(query);
    }

    pub fn sampleMultiPhasePartialDot(self: *Quantized, query: []const f32) !TwoPhaseSample {
        std.debug.assert(query.len == self.d);

        const phase_exact_top_k = partial_dot_phase_exact_top_k;
        const vocab_size = self.vocab_size;
        const d = self.d;
        const lm_head = self.lm_head;
        const lm_data = lm_head.data;
        const lm_data_t = lm_head.data_t;
        const is_junk = lm_head.is_junk;
        const active_rows = self.active_rows;
        const partial_logits = self.partial_logits;
        const residual_logits = self.partial_dot_residual_logits;
        const dense_logits = self.dense_logits;
        const prefix_coord_count = partial_dot_prefix_coord_count;

        var prefix_coords: [prefix_coord_count]usize = [_]usize{0} ** prefix_coord_count;
        var prefix_scores: [prefix_coord_count]f32 = [_]f32{-std.math.inf(f32)} ** prefix_coord_count;

        // Phase 0a: find the largest raw query coordinates for exact prefix scoring.
        for (0..d) |coord| {
            insertTopK(prefix_coord_count, coord, @abs(query[coord]), &prefix_coords, &prefix_scores);
        }

        // Phase 0b: compute the exact lm_head contribution of the raw coordinate prefix.
        @memset(partial_logits, 0.0);
        const simd_len = 16;
        const Vec = @Vector(simd_len, f32);
        for (0..prefix_coord_count) |prefix_i| {
            if (prefix_scores[prefix_i] == -std.math.inf(f32)) break;
            const query_coord = prefix_coords[prefix_i];
            const query_value = query[query_coord];
            const coord_values = lm_data_t[query_coord * vocab_size ..][0..vocab_size];
            const query_vec: Vec = @splat(query_value);

            var row_i: usize = 0;
            while (row_i + simd_len <= vocab_size) : (row_i += simd_len) {
                const coord_vec: Vec = coord_values[row_i..][0..simd_len].*;
                var logits_vec: Vec = partial_logits[row_i..][0..simd_len].*;
                logits_vec = @mulAdd(Vec, coord_vec, query_vec, logits_vec);
                partial_logits[row_i..][0..simd_len].* = logits_vec;
            }
            while (row_i < vocab_size) : (row_i += 1) {
                partial_logits[row_i] += coord_values[row_i] * query_value;
            }
        }

        // Phase 0c: zero the exact prefix coordinates before residual rotation.
        @memcpy(self.f32_buffer[0..d], query);
        for (0..prefix_coord_count) |prefix_i| {
            if (prefix_scores[prefix_i] == -std.math.inf(f32)) break;
            self.f32_buffer[prefix_coords[prefix_i]] = 0.0;
        }

        // Phase 1 setup: rotate the residual query globally. Exact partial blocks are sampled from this rotated space.
        try self.walshHadamardSimd(self.f32_buffer[0..d], 12);
        var residual_query_norm2: f32 = 0.0;
        for (self.f32_buffer[0..d]) |value| {
            residual_query_norm2 += value * value;
        }
        const residual_query_norm = @sqrt(residual_query_norm2);

        var active_count: usize = 0;
        for (0..vocab_size) |row_i| {
            if (is_junk[row_i]) {
                dense_logits[row_i] = -std.math.inf(f32);
                continue;
            }
            active_rows[active_count] = @intCast(row_i);
            active_count += 1;
            residual_logits[row_i] = 0.0;
            dense_logits[row_i] = std.math.inf(f32);
        }

        var phase_rows: [phase_exact_top_k]usize = [_]usize{0} ** phase_exact_top_k;
        var phase_scores: [phase_exact_top_k]f32 = [_]f32{-std.math.inf(f32)} ** phase_exact_top_k;
        var top_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var top_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;
        var pruned_by_phase: [partial_dot_block_count]usize = [_]usize{0} ** partial_dot_block_count;

        const z_score: f32 = 5.0;
        const d_f32: f32 = @floatFromInt(d);

        var best_lower_bound: f32 = -std.math.inf(f32);
        var nb_scored: usize = 0;
        var nb_partial_dot_scored: usize = 0;
        var nb_partial_dot_pruned: usize = 0;

        for (0..partial_dot_block_count) |phase_i| {
            if (active_count == 0) break;

            const block_i = phase_i;
            const block_start = block_i * partial_dot_block_size;
            const block_offset = block_i * partial_dot_block_size * vocab_size;
            const query_block = self.f32_buffer[block_start..][0..partial_dot_block_size];
            const scored_dim = (phase_i + 1) * partial_dot_block_size;
            const scored_dim_f32: f32 = @floatFromInt(scored_dim);
            const remaining_dim = d - scored_dim;
            const remaining_dim_f32: f32 = @floatFromInt(remaining_dim);
            const estimator_scale = d_f32 / scored_dim_f32;
            const err_scale = if (remaining_dim == 0)
                0.0
            else
                z_score * @sqrt(remaining_dim_f32 / d_f32) / @sqrt(scored_dim_f32);

            resetTopK(phase_exact_top_k, &phase_rows, &phase_scores);
            for (active_rows[0..active_count]) |row_u32| {
                const row_i: usize = @intCast(row_u32);
                const row_block = self.partial_dot_blocks[block_offset + row_i * partial_dot_block_size ..][0..partial_dot_block_size];
                residual_logits[row_i] += dotPartialDotBlock(query_block, row_block);
                nb_partial_dot_scored += 1;
                const approx = partial_logits[row_i] + estimator_scale * residual_logits[row_i];
                const err = err_scale * residual_query_norm * lm_head.row_norms[row_i];
                dense_logits[row_i] = approx + err;
                insertTopK(phase_exact_top_k, row_i, approx, &phase_rows, &phase_scores);
            }

            exactScorePhaseTopK(phase_exact_top_k, self, query, &phase_rows, &phase_scores, &top_rows, &top_scores, &best_lower_bound, &nb_scored);

            var next_active_count: usize = 0;
            for (active_rows[0..active_count]) |row_u32| {
                const row_i: usize = @intCast(row_u32);
                active_rows[next_active_count] = row_u32;
                next_active_count += @intFromBool(dense_logits[row_i] >= best_lower_bound);
            }
            const phase_pruned = active_count - next_active_count;
            pruned_by_phase[phase_i] = phase_pruned;
            nb_partial_dot_pruned += phase_pruned;
            active_count = next_active_count;
        }

        for (active_rows[0..active_count]) |row_u32| {
            const row_i: usize = @intCast(row_u32);
            if (dense_logits[row_i] < best_lower_bound) continue;
            const row = lm_data[row_i * d ..][0..d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
        }

        return .{
            .rows = top_rows,
            .nb_scored = nb_scored,
            .nb_partial_dot_scored = nb_partial_dot_scored,
            .nb_partial_dot_pruned = nb_partial_dot_pruned,
            .partial_dot_pruned_by_phase = pruned_by_phase,
        };
    }

    pub fn sample5PhasesOptimized(self: *Quantized, query: []const f32) !TwoPhaseSample {
        std.debug.assert(query.len == self.d);
        const prefix_coord_count = 12;
        const phase_exact_top_k = 16;
        const vocab_size = self.vocab_size;
        const d = self.d;
        const lm_head = self.lm_head;
        const lm_data_t = lm_head.data_t;
        const row_norms = lm_head.row_norms;
        const partial_logits = self.partial_logits;
        const dense_logits = self.dense_logits;
        const active_rows = self.active_rows;
        const qjl_micro_rows = self.qjl_micro_rows;
        const qjl_micro_to_half_rows = self.qjl_micro_to_half_rows;
        const qjl_micro_mismatches = self.qjl_micro_mismatches;

        var prefix_coords: [prefix_coord_count]usize = [_]usize{0} ** prefix_coord_count;
        var prefix_scores: [prefix_coord_count]f32 = [_]f32{-std.math.inf(f32)} ** prefix_coord_count;
        var phase_rows: [phase_exact_top_k]usize = [_]usize{0} ** phase_exact_top_k;
        var phase_scores: [phase_exact_top_k]f32 = [_]f32{-std.math.inf(f32)} ** phase_exact_top_k;
        var top_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var top_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;

        if (true) return .{
            .rows = top_rows,
            .nb_scored = 0,
            .nb_1bit_scored = 0,
            .nb_half_1bit_scored = 0,
            .nb_2bit_scored = 0,
        };
        
        // Phase 0a: find the largest raw query coordinates for exact prefix scoring.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_prefix_top);
        for (0..d) |coord| {
            insertTopK(prefix_coord_count, coord, @abs(query[coord]), &prefix_coords, &prefix_scores);
        }
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_prefix_top);

        // Phase 0b: compute the exact lm_head dot contribution from the raw coordinate prefix.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_prefix_dot);
        @memset(partial_logits, 0.0);
        const simd_len = 16;
        const Vec = @Vector(simd_len, f32);
        for (0..prefix_coord_count) |prefix_i| {
            if (prefix_scores[prefix_i] == -std.math.inf(f32)) break;
            const query_coord = prefix_coords[prefix_i];
            const query_value = query[query_coord];
            const coord_values = lm_data_t[query_coord * vocab_size ..][0..vocab_size];
            const query_vec: Vec = @splat(query_value);

            var row_i: usize = 0;
            while (row_i + simd_len <= vocab_size) : (row_i += simd_len) {
                const coord_vec: Vec = coord_values[row_i..][0..simd_len].*;
                var logits_vec: Vec = partial_logits[row_i..][0..simd_len].*;
                logits_vec = @mulAdd(Vec, coord_vec, query_vec, logits_vec);
                partial_logits[row_i..][0..simd_len].* = logits_vec;
            }
            while (row_i < vocab_size) : (row_i += 1) {
                partial_logits[row_i] += coord_values[row_i] * query_value;
            }
        }
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_prefix_dot);

        // Phase 0c: zero the exact prefix coordinates before quantizing the residual query.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_quantize);
        @memcpy(self.f32_buffer[0..d], query);
        for (0..prefix_coord_count) |prefix_i| {
            if (prefix_scores[prefix_i] == -std.math.inf(f32)) break;
            self.f32_buffer[prefix_coords[prefix_i]] = 0.0;
        }

        const quantized_query, _, const query_l2_norm = try self.quantizeQjlVector(self.f32_buffer[0..d]);
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_quantize);
        const z_score: f32 = 5.0;
        const err_const_1bit: f32 = 0.92;
        const err_const_2bits: f32 = 0.49;
        const qjl_micro_err_scale: f32 = @sqrt(@as(f32, @floatFromInt(d)) / @as(f32, @floatFromInt(qjl_micro_coord_count)));
        const qjl_half_err_scale: f32 = @sqrt(@as(f32, @floatFromInt(d)) / @as(f32, @floatFromInt(qjl_half_coord_count)));
        const base_err_scale_1bit = z_score * err_const_1bit * query_l2_norm * self.d_inv_norm;
        const err_scale_micro = base_err_scale_1bit * qjl_micro_err_scale;
        const err_scale_half = base_err_scale_1bit * qjl_half_err_scale;
        const err_scale_2bits = z_score * err_const_2bits * query_l2_norm * self.d_inv_norm;
        var best_lower_bound: f32 = -std.math.inf(f32);

        // Phase 0.5: score the first 1024 rotated sign bits with widened 1-bit bounds.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_micro);
        resetTopK(phase_exact_top_k, &phase_rows, &phase_scores);
        const query_micro_words = qjlMicroWords(&quantized_query);
        const query_micro_vec: @Vector(qjl_micro_u128_word_count, u128) = query_micro_words.*;
        for (0..vocab_size) |row_i| {
            if (lm_head.is_junk[row_i]) {
                dense_logits[row_i] = -std.math.inf(f32);
                continue;
            }
            const row_norm = row_norms[row_i];

            const row_words = &qjl_micro_rows[row_i];
            var mismatch_count_u16: u16 = 0;
            mismatch_count_u16 += qjlMismatchU128Loaded(qjl_micro_u128_word_count, row_words, query_micro_vec);
            qjl_micro_mismatches[row_i] = mismatch_count_u16;

            const approx = partial_logits[row_i] + qjlPartialSignDot(query_l2_norm, row_norm, mismatch_count_u16, qjl_micro_coord_count);
            const err = err_scale_micro * row_norm;
            dense_logits[row_i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTopK(phase_exact_top_k, row_i, approx, &phase_rows, &phase_scores);
        }
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_micro);

        var nb_scored: usize = 0;
        exactScorePhaseTopK(phase_exact_top_k, self, query, &phase_rows, &phase_scores, &top_rows, &top_scores, &best_lower_bound, &nb_scored);

        // Phase 0.5 prune: compact rows whose micro upper bound survives.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_prune);
        var active_count: usize = 0;
        for (0..vocab_size) |row_i| {
            active_rows[active_count] = @intCast(row_i);
            active_count += @intFromBool(dense_logits[row_i] >= best_lower_bound);
        }
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_prune);

        // Phase 0.75: score the next 1024 QJL bits, reusing the micro mismatch counts.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_half);
        resetTopK(phase_exact_top_k, &phase_rows, &phase_scores);
        const query_micro_to_half_words = qjlMicroToHalfWords(&quantized_query);
        const query_micro_to_half_vec: @Vector(qjl_micro_u128_word_count, u128) = query_micro_to_half_words.*;
        var nb_half_1bit_scored: usize = 0;
        for (active_rows[0..active_count]) |row_u32| {
            const row_i: usize = @intCast(row_u32);
            nb_half_1bit_scored += 1;

            const row_words = &qjl_micro_to_half_rows[row_i];
            var mismatch_count_u16: u16 = qjl_micro_mismatches[row_i];
            mismatch_count_u16 += qjlMismatchU128Loaded(qjl_micro_u128_word_count, row_words, query_micro_to_half_vec);
            qjl_micro_mismatches[row_i] = mismatch_count_u16;

            const row_norm = row_norms[row_i];
            const approx = partial_logits[row_i] + qjlPartialSignDot(query_l2_norm, row_norm, mismatch_count_u16, qjl_half_coord_count);
            const err = err_scale_half * row_norm;
            dense_logits[row_i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTopK(phase_exact_top_k, row_i, approx, &phase_rows, &phase_scores);
        }
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_half);

        exactScorePhaseTopK(phase_exact_top_k, self, query, &phase_rows, &phase_scores, &top_rows, &top_scores, &best_lower_bound, &nb_scored);

        // Phase 0.75 prune: compact rows whose half-dot upper bound survives.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_prune);
        var next_active_count: usize = 0;
        for (active_rows[0..active_count]) |row_u32| {
            const row_i: usize = @intCast(row_u32);
            active_rows[next_active_count] = row_u32;
            next_active_count += @intFromBool(dense_logits[row_i] >= best_lower_bound);
        }
        active_count = next_active_count;
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_prune);

        // Phase 1: rescore survivors with full 1-bit QJL, reusing the first 2048-bit popcount.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_full_1bit);
        resetTopK(phase_exact_top_k, &phase_rows, &phase_scores);
        var nb_1bit_scored: usize = 0;
        const query_half_to_full_words = qjlHalfToFullWords(&quantized_query);
        const query_half_to_full_vec: @Vector(qjl_half_to_full_u128_word_count, u128) = query_half_to_full_words.*;
        for (active_rows[0..active_count]) |row_u32| {
            const row_i: usize = @intCast(row_u32);
            nb_1bit_scored += 1;
            var mismatch_count_u16 = qjl_micro_mismatches[row_i];
            mismatch_count_u16 += qjlMismatchU128Loaded(
                qjl_half_to_full_u128_word_count,
                &self.qjl_half_to_full_rows[row_i],
                query_half_to_full_vec,
            );
            const approx = partial_logits[row_i] + self.dotQjlSymmetricFromMismatchCount(row_i, query_l2_norm, mismatch_count_u16);
            const err = base_err_scale_1bit * row_norms[row_i];
            dense_logits[row_i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTopK(phase_exact_top_k, row_i, approx, &phase_rows, &phase_scores);
        }
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_full_1bit);

        exactScorePhaseTopK(phase_exact_top_k, self, query, &phase_rows, &phase_scores, &top_rows, &top_scores, &best_lower_bound, &nb_scored);

        // Phase 1 prune: compact rows whose full 1-bit upper bound survives into 2-bit scoring.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_prune);
        next_active_count = 0;
        for (active_rows[0..active_count]) |row_u32| {
            const row_i: usize = @intCast(row_u32);
            active_rows[next_active_count] = row_u32;
            next_active_count += @intFromBool(dense_logits[row_i] >= best_lower_bound);
        }
        active_count = next_active_count;
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_prune);

        // Phase 2: score survivors with the tighter 2-bit QJL residual approximation.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_2bit);
        const query_msb, const query_lsb = self.quantize2BitsSymmetric(query_l2_norm);
        const query_scale = scaleFactor2BitsSymmetric(self.f32_buffer[0..4096], &query_msb, &query_lsb);
        resetTopK(phase_exact_top_k, &phase_rows, &phase_scores);
        var nb_2bit_scored: usize = 0;
        for (active_rows[0..active_count]) |row_u32| {
            const row_i: usize = @intCast(row_u32);
            nb_2bit_scored += 1;
            const dot: f32 = @floatFromInt(self.dotQjlSymmetric2Bits(row_i, &query_msb, &query_lsb));
            const approx = partial_logits[row_i] + query_scale * self.qjl_row_scale_2bits[row_i] * dot;
            const err = err_scale_2bits * row_norms[row_i];
            dense_logits[row_i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTopK(phase_exact_top_k, row_i, approx, &phase_rows, &phase_scores);
        }
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_2bit);

        exactScorePhaseTopK(phase_exact_top_k, self, query, &phase_rows, &phase_scores, &top_rows, &top_scores, &best_lower_bound, &nb_scored);

        // Phase 3: exact-score remaining rows whose 2-bit upper bound still survives.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_dense_final);
        for (active_rows[0..active_count]) |row_u32| {
            const row_i: usize = @intCast(row_u32);
            if (dense_logits[row_i] < best_lower_bound) continue;
            const row = lm_head.data[row_i * d ..][0..d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
        }
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_dense_final);

        return .{
            .rows = top_rows,
            .nb_scored = nb_scored,
            .nb_1bit_scored = nb_1bit_scored,
            .nb_half_1bit_scored = nb_half_1bit_scored,
            .nb_2bit_scored = nb_2bit_scored,
        };
    }

    pub fn sample5PhasesBaseline(self: *Quantized, query: []const f32, comptime phase_exact_top_k: usize) !TwoPhaseSample {
        std.debug.assert(query.len == self.d);
        std.debug.assert(phase_exact_top_k > 0);
        const prefix_coord_count = 4;

        var prefix_coords: [prefix_coord_count]usize = [_]usize{0} ** prefix_coord_count;
        var prefix_scores: [prefix_coord_count]f32 = [_]f32{-std.math.inf(f32)} ** prefix_coord_count;
        var phase05_rows: [phase_exact_top_k]usize = [_]usize{0} ** phase_exact_top_k;
        var phase05_scores: [phase_exact_top_k]f32 = [_]f32{-std.math.inf(f32)} ** phase_exact_top_k;
        var phase075_rows: [phase_exact_top_k]usize = [_]usize{0} ** phase_exact_top_k;
        var phase075_scores: [phase_exact_top_k]f32 = [_]f32{-std.math.inf(f32)} ** phase_exact_top_k;
        var phase1_rows: [phase_exact_top_k]usize = [_]usize{0} ** phase_exact_top_k;
        var phase1_scores: [phase_exact_top_k]f32 = [_]f32{-std.math.inf(f32)} ** phase_exact_top_k;
        var phase2_rows: [phase_exact_top_k]usize = [_]usize{0} ** phase_exact_top_k;
        var phase2_scores: [phase_exact_top_k]f32 = [_]f32{-std.math.inf(f32)} ** phase_exact_top_k;
        var top_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var top_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;

        var coord: usize = 0;
        // Phase 0a: find the largest raw query coordinates for exact prefix scoring.
        while (coord < self.d) : (coord += 1) {
            insertTopK(prefix_coord_count, coord, @abs(query[coord]), &prefix_coords, &prefix_scores);
        }

        @memset(self.partial_logits, 0.0);
        const simd_len = 16;
        const Vec = @Vector(simd_len, f32);

        var prefix_i: usize = 0;
        // Phase 0b: compute the exact lm_head dot contribution from the raw coordinate prefix.
        while (prefix_i < prefix_coord_count) : (prefix_i += 1) {
            if (prefix_scores[prefix_i] == -std.math.inf(f32)) break;
            const query_coord = prefix_coords[prefix_i];
            const query_value = query[query_coord];
            const coord_values = self.lm_head.data_t[query_coord * self.vocab_size ..][0..self.vocab_size];
            const query_vec: Vec = @splat(query_value);

            var row_i: usize = 0;
            while (row_i + simd_len <= self.vocab_size) : (row_i += simd_len) {
                const coord_vec: Vec = coord_values[row_i..][0..simd_len].*;
                var logits_vec: Vec = self.partial_logits[row_i..][0..simd_len].*;
                logits_vec = @mulAdd(Vec, coord_vec, query_vec, logits_vec);
                self.partial_logits[row_i..][0..simd_len].* = logits_vec;
            }
            while (row_i < self.vocab_size) : (row_i += 1) {
                self.partial_logits[row_i] += coord_values[row_i] * query_value;
            }
        }

        // Phase 0c: zero the exact prefix coordinates before quantizing the residual query.
        @memcpy(self.f32_buffer[0..self.d], query);
        prefix_i = 0;
        while (prefix_i < prefix_coord_count) : (prefix_i += 1) {
            if (prefix_scores[prefix_i] == -std.math.inf(f32)) break;
            self.f32_buffer[prefix_coords[prefix_i]] = 0.0;
        }

        const quantized_query, _, const query_l2_norm = try self.quantizeQjlVector(self.f32_buffer[0..self.d]);
        const z_score: f32 = 5.0;
        const err_const_1bit: f32 = 0.92;
        const err_const_2bits: f32 = 0.49;
        const qjl_micro_err_scale: f32 = @sqrt(@as(f32, @floatFromInt(self.d)) / @as(f32, @floatFromInt(qjl_micro_coord_count)));
        const qjl_half_err_scale: f32 = @sqrt(@as(f32, @floatFromInt(self.d)) / @as(f32, @floatFromInt(qjl_half_coord_count)));
        var best_lower_bound: f32 = -std.math.inf(f32);

        // Phase 0.5a: take the first rotated residual sign bits; no rotated top-k selection.
        const query_micro_words = qjlMicroWords(&quantized_query);

        var i: usize = 0;
        // Phase 0.5b: score the precomputed first-bit QJL subslice and use its widened bound.
        while (i < self.vocab_size) : (i += 1) {
            if (self.lm_head.is_junk[i]) {
                self.dense_logits[i] = -std.math.inf(f32);
                continue;
            }
            const row_words = &self.qjl_micro_rows[i];
            var mismatch_count_u16: u16 = 0;
            mismatch_count_u16 += qjlMismatchU128(qjl_micro_u128_word_count, row_words, query_micro_words);
            self.qjl_micro_mismatches[i] = mismatch_count_u16;
            const row_norm = self.lm_head.row_norms[i];
            const approx = self.partial_logits[i] + qjlPartialSignDot(query_l2_norm, row_norm, mismatch_count_u16, qjl_micro_coord_count);
            const err = z_score * err_const_1bit * qjl_micro_err_scale * query_l2_norm * row_norm * self.d_inv_norm;
            self.dense_logits[i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTopK(phase_exact_top_k, i, approx, &phase05_rows, &phase05_scores);
        }

        var nb_scored: usize = 0;
        // Phase 0.5c: exact-score the micro-estimator top-k to raise the lower bound early.
        i = 0;
        while (i < phase_exact_top_k) : (i += 1) {
            if (phase05_scores[i] == -std.math.inf(f32)) break;
            const row_i = phase05_rows[i];
            const row = self.lm_head.data[row_i * self.d .. (row_i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
            self.dense_logits[row_i] = -std.math.inf(f32);
        }

        // Phase 0.5d: compact rows whose micro upper bound survives into full 1-bit scoring.
        var active_count: usize = 0;
        i = 0;
        while (i < self.vocab_size) : (i += 1) {
            if (self.dense_logits[i] < best_lower_bound) {
                self.dense_logits[i] = -std.math.inf(f32);
                continue;
            }
            self.active_rows[active_count] = @intCast(i);
            active_count += 1;
        }

        // Phase 0.75a: score the next 1024 QJL bits, reusing the phase 0.5 mismatch count.
        const query_micro_to_half_words = qjlMicroToHalfWords(&quantized_query);
        var nb_half_1bit_scored: usize = 0;
        i = 0;
        while (i < active_count) : (i += 1) {
            const row_i: usize = @intCast(self.active_rows[i]);
            nb_half_1bit_scored += 1;

            const row_words = &self.qjl_micro_to_half_rows[row_i];
            var mismatch_count_u16: u16 = self.qjl_micro_mismatches[row_i];
            mismatch_count_u16 += qjlMismatchU128(qjl_micro_u128_word_count, row_words, query_micro_to_half_words);
            self.qjl_micro_mismatches[row_i] = mismatch_count_u16;

            const row_norm = self.lm_head.row_norms[row_i];
            const approx = self.partial_logits[row_i] + qjlPartialSignDot(query_l2_norm, row_norm, mismatch_count_u16, qjl_half_coord_count);
            const err = z_score * err_const_1bit * qjl_half_err_scale * query_l2_norm * row_norm * self.d_inv_norm;
            self.dense_logits[row_i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTopK(phase_exact_top_k, row_i, approx, &phase075_rows, &phase075_scores);
        }

        // Phase 0.75b: exact-score the half-dot top-k to tighten the lower bound.
        i = 0;
        while (i < phase_exact_top_k) : (i += 1) {
            if (phase075_scores[i] == -std.math.inf(f32)) break;
            const row_i = phase075_rows[i];
            const row = self.lm_head.data[row_i * self.d .. (row_i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
            self.dense_logits[row_i] = -std.math.inf(f32);
        }

        // Phase 0.75c: compact rows whose half-dot upper bound survives into full 1-bit scoring.
        var half_active_count: usize = 0;
        i = 0;
        while (i < active_count) : (i += 1) {
            const row_i: usize = @intCast(self.active_rows[i]);
            if (self.dense_logits[row_i] < best_lower_bound) {
                self.dense_logits[row_i] = -std.math.inf(f32);
                continue;
            }
            self.active_rows[half_active_count] = @intCast(row_i);
            half_active_count += 1;
        }
        active_count = half_active_count;

        var nb_1bit_scored: usize = 0;
        // Phase 1: rescore surviving rows with full 1-bit QJL, reusing the half-dot popcounts.
        i = 0;
        while (i < active_count) : (i += 1) {
            const row_i: usize = @intCast(self.active_rows[i]);
            nb_1bit_scored += 1;
            const approx = self.partial_logits[row_i] + self.dotQjlSymmetricWithPrefix(row_i, quantized_query, query_l2_norm, self.qjl_micro_mismatches[row_i], qjl_half_coord_count);
            const err = z_score * err_const_1bit * query_l2_norm * self.lm_head.row_norms[row_i] * self.d_inv_norm;
            self.dense_logits[row_i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTopK(phase_exact_top_k, row_i, approx, &phase1_rows, &phase1_scores);
        }

        // Phase 1b: exact-score the full 1-bit top-k to tighten the lower bound.
        i = 0;
        while (i < phase_exact_top_k) : (i += 1) {
            if (phase1_scores[i] == -std.math.inf(f32)) break;
            const row_i = phase1_rows[i];
            const row = self.lm_head.data[row_i * self.d .. (row_i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
            self.dense_logits[row_i] = -std.math.inf(f32);
        }

        // Phase 1c: compact rows whose full 1-bit upper bound survives into 2-bit scoring.
        var next_active_count: usize = 0;
        i = 0;
        while (i < active_count) : (i += 1) {
            const row_i: usize = @intCast(self.active_rows[i]);
            if (self.dense_logits[row_i] < best_lower_bound) {
                self.dense_logits[row_i] = -std.math.inf(f32);
                continue;
            }
            self.active_rows[next_active_count] = @intCast(row_i);
            next_active_count += 1;
        }
        active_count = next_active_count;

        const query_msb, const query_lsb = self.quantize2BitsSymmetric(query_l2_norm);
        const query_scale = scaleFactor2BitsSymmetric(self.f32_buffer[0..4096], &query_msb, &query_lsb);
        var nb_2bit_scored: usize = 0;

        // Phase 2: rescore surviving rows with the tighter 2-bit QJL residual approximation.
        i = 0;
        while (i < active_count) : (i += 1) {
            const row_i: usize = @intCast(self.active_rows[i]);
            nb_2bit_scored += 1;
            const dot: f32 = @floatFromInt(self.dotQjlSymmetric2Bits(row_i, &query_msb, &query_lsb));
            const approx = self.partial_logits[row_i] + query_scale * self.qjl_row_scale_2bits[row_i] * dot;
            const err = z_score * err_const_2bits * query_l2_norm * self.lm_head.row_norms[row_i] * self.d_inv_norm;
            self.dense_logits[row_i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTopK(phase_exact_top_k, row_i, approx, &phase2_rows, &phase2_scores);
        }

        // Phase 2b: exact-score the 2-bit top-k to tighten the final dense threshold.
        i = 0;
        while (i < phase_exact_top_k) : (i += 1) {
            if (phase2_scores[i] == -std.math.inf(f32)) break;
            const row_i = phase2_rows[i];
            const row = self.lm_head.data[row_i * self.d .. (row_i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
            self.dense_logits[row_i] = -std.math.inf(f32);
        }

        // Phase 3: exact-score remaining rows whose 2-bit upper bound still survives.
        i = 0;
        while (i < active_count) : (i += 1) {
            const row_i: usize = @intCast(self.active_rows[i]);
            if (self.dense_logits[row_i] < best_lower_bound) continue;
            const row = self.lm_head.data[row_i * self.d .. (row_i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
        }

        return .{ .rows = top_rows, .nb_scored = nb_scored, .nb_1bit_scored = nb_1bit_scored, .nb_half_1bit_scored = nb_half_1bit_scored, .nb_2bit_scored = nb_2bit_scored };
    }
};
