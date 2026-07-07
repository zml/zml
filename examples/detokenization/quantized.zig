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
    nb_2bit_scored: usize = 0,
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
    i8_buffer: []i8,

    lm_head_sliced: []vector_U4096,
    row_offset: []i32,

    lm_head_qjl: []U4096,
    qjl_row_scale: []f32,
    qjl_query_lut: []f32,
    qjl_inv_dim: f32 = 0.0,
    qjl_dim_shift: i32 = 0,

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
        const lm_head_quantized = try allocator.alloc(i8, d * vocab_size);
        const lm_head_sliced = try allocator.alloc(vector_U4096, vocab_size);

        const row_offset = try allocator.alloc(i32, vocab_size);
        const row_scale = try allocator.alloc(f32, vocab_size);
        const row_scale_sliced = try allocator.alloc(f32, vocab_size);
        const i8_buffer = try allocator.alloc(i8, d);

        const lm_head_qjl = try allocator.alloc(U4096, vocab_size);
        const qjl_row_scale = try allocator.alloc(f32, vocab_size);
        const qjl_query_lut = try allocator.alloc(f32, 512 * 256);
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
            .lm_head_quantized = lm_head_quantized,
            .lm_head_sliced = lm_head_sliced,
            .row_offset = row_offset,
            .i8_buffer = i8_buffer,
            .lm_head_qjl = lm_head_qjl,
            .qjl_row_scale = qjl_row_scale,
            .qjl_query_lut = qjl_query_lut,
            .lm_head_qjl_2bits = lm_head_qjl_2bits,
            .qjl_row_scale_2bits = qjl_row_scale_2bits,
        };
    }

    pub fn deinit(self: *Quantized) void {
        self.allocator.free(self.row_scale);
        self.allocator.free(self.row_scale_sliced);
        self.allocator.free(self.f32_buffer);
        self.allocator.free(self.dense_logits);
        self.allocator.free(self.lm_head_quantized);
        self.allocator.free(self.lm_head_sliced);
        self.allocator.free(self.row_offset);
        self.allocator.free(self.i8_buffer);
        self.allocator.free(self.lm_head_qjl);
        self.allocator.free(self.qjl_row_scale);
        self.allocator.free(self.qjl_query_lut);
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


    pub fn walshHadamard(self: *Quantized, v: []const f32, comptime k: comptime_int) !void {
        self.zml_handler.tic(&self.zml_handler.timers.quant_walsh);
        if (v.ptr != self.f32_buffer.ptr) {
            @memcpy(self.f32_buffer[0..v.len], v);
        }
        const n: usize = 1 << k;
        std.debug.assert(v.len == n);
        std.debug.assert(self.f32_buffer.len >= n);

        inline for (0..k) |stage| {
            const h: usize = 1 << stage;
            const step: usize = 2 * h;
            for (0..(n / step)) |block| {
                const i: usize = block * step;
                for (0..h) |offset| {
                    const j: usize = i + offset;
                    const x = self.f32_buffer[j];
                    const y = self.f32_buffer[j + h];
                    self.f32_buffer[j] = x + y;
                    self.f32_buffer[j + h] = x - y;
                }
            }
        }

        for (0..n) |i| {
            self.f32_buffer[i] *= self.d_inv_norm;
        }
        self.zml_handler.toc(&self.zml_handler.timers.quant_walsh);
    }

    pub fn walshHadamardSimd(self: *Quantized, v: []const f32, comptime k: comptime_int) !void {
        self.zml_handler.tic(&self.zml_handler.timers.quant_walsh);
        if (v.ptr != self.f32_buffer.ptr) {
            @memcpy(self.f32_buffer[0..v.len], v);
        }
        const n: usize = 1 << k;
        std.debug.assert(v.len == n);
        std.debug.assert(self.f32_buffer.len >= n);

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
                    const x: Vec = self.f32_buffer[left..][0..simd_len].*;
                    const y: Vec = self.f32_buffer[right..][0..simd_len].*;
                    self.f32_buffer[left..][0..simd_len].* = x + y;
                    self.f32_buffer[right..][0..simd_len].* = x - y;
                }
            }
        }

        const Vec = @Vector(max_simd_len, f32);
        const scale: Vec = @splat(self.d_inv_norm);
        var i: usize = 0;
        while (i + max_simd_len <= n) : (i += max_simd_len) {
            const values: Vec = self.f32_buffer[i..][0..max_simd_len].*;
            self.f32_buffer[i..][0..max_simd_len].* = values * scale;
        }
        self.zml_handler.toc(&self.zml_handler.timers.quant_walsh);
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
            acc += query_v * row_v;
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


    pub fn quantizeQjlLmHead(self: *Quantized) !void {
        std.log.info("Quantizing QJL lm_head", .{});
        self.qjl_inv_dim = 1.0 / @as(f32, @floatFromInt(self.d));
        self.qjl_dim_shift = @intCast(self.d);
        const data = self.lm_head.data;
        for (0..self.vocab_size) |i| {
            const quantized_row, const row_scale, _ = try self.quantizeQjlVector(data[i * self.d .. (i + 1) * self.d]);
            self.lm_head_qjl[i] = quantized_row;
            self.qjl_row_scale[i] = row_scale;
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
        const row_quantized = &self.lm_head_qjl[row];
        const row_l2_norm = self.lm_head.row_norms[row];

        // dot(u,v) approx = ||u|| * ||v|| * sin(pi/2 * dot(q(u), q(v)) / d)
        // dot(q(u), q(v)) = d - 2 * bit_mismatches,
        // XOR returns 1 where bits differ, 0 where they match
        const Vec32x128 = @Vector(32, u128);
        const Vec32x16 = @Vector(32, u16);
        const query_vec: Vec32x128 = query_quantized.words[0..32].*;
        const row_vec: Vec32x128 = row_quantized.words[0..32].*;
        const mismatches: Vec32x16 = @intCast(@popCount(query_vec ^ row_vec));
        const mismatch_count: i32 = @intCast(@reduce(.Add, mismatches));
        const dot = self.qjl_dim_shift - 2 * mismatch_count;

        // sin(pi/2 w) is approx = y * (0.775 + 0.225 |y|)
        // with y = w * (2.0 - |w|)
        const w = @as(f32, @floatFromInt(dot)) * self.qjl_inv_dim;
        std.debug.assert(-1.0 <= w and w <= 1.0);
        const y = w * (2.0 - @abs(w));
        const sin_approx = y * (0.775 + 0.225 * @abs(y));

        return query_l2_norm * row_l2_norm * sin_approx;
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
            const lut = self.qjl_query_lut[block_i * 256..][0..256];

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
            const score = self.realLogit(query, self.lm_head.data[i * self.d..(i + 1) * self.d]);
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

    inline fn insertTop16(row: usize, score: f32, rows: *[top_k_sliced]usize, scores: *[top_k_sliced]f32) void {
        if (score <= scores[top_k_sliced - 1]) return;

        var insert_pos = top_k_sliced - 1;
        while (insert_pos > 0 and score > scores[insert_pos - 1]) {
            scores[insert_pos] = scores[insert_pos - 1];
            rows[insert_pos] = rows[insert_pos - 1];
            insert_pos -= 1;
        }
        scores[insert_pos] = score;
        rows[insert_pos] = row;
    }

    inline fn insertTop8(row: usize, score: f32, rows: *[8]usize, scores: *[8]f32) void {
        if (score <= scores[7]) return;

        var insert_pos: usize = 7;
        while (insert_pos > 0 and score > scores[insert_pos - 1]) {
            scores[insert_pos] = scores[insert_pos - 1];
            rows[insert_pos] = rows[insert_pos - 1];
            insert_pos -= 1;
        }
        scores[insert_pos] = score;
        rows[insert_pos] = row;
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

        const quantized_query, _, const query_l2_norm = try self.quantizeQjlVector(query);
        const z_score: f32 = 5.0;
        const err_const_1bit: f32 = 0.92;
        const err_const_2bits: f32 = 0.49;
        var best_lower_bound: f32 = -std.math.inf(f32);

        var i: usize = 0;
        while (i < self.vocab_size) : (i += 1) {
            const approx = self.dotQjlSymmetric(i, quantized_query, query_l2_norm);
            const err = z_score * err_const_1bit * query_l2_norm * self.lm_head.row_norms[i] * self.d_inv_norm;
            self.dense_logits[i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTop8(i, approx, &phase1_rows, &phase1_scores);
        }

        var nb_scored: usize = 0;
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
};
