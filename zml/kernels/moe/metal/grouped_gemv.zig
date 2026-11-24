//! NVFP4 and MXFP8 expert projections over routes sorted by expert: each
//! threadgroup reads one expert's weights for a few output columns once,
//! dequantizes them in registers and applies them to every route of that expert.

const std = @import("std");

const metal = @import("zml/core").kernel.metal;

const Builder = metal.Builder;
const Type = metal.Type;
const Value = metal.Value;

const simd_width = 32;
/// Weight values in one 32-bit word of FP4 codes, or in two of FP8 codes.
const values_per_word = 8;

pub const Format = enum {
    /// E2M1 values, two per byte, low nibble first, and an E4M3 scale per 16.
    nvfp4,
    /// E4M3 values and an E8M0 scale per 32.
    mxfp8,

    /// Values sharing one scale.
    fn group(self: Format) i32 {
        return switch (self) {
            .nvfp4 => 16,
            .mxfp8 => 32,
        };
    }

    /// The load of `words` words of 8 values.
    fn unitType(self: Format, words: u32) Type {
        return switch (self) {
            .nvfp4 => if (words == 1) .uint else .uint2,
            .mxfp8 => if (words == 1) .uint2 else .uint4,
        };
    }
};

/// `out[r, n] = x[r, :] . W[ids[r], n, :]` for routes sorted by expert. The first route
/// of each expert computes all routes of that expert; a route without an expert
/// (negative id) gets zeros.
pub const GroupedGemv = struct {
    pub const Cfg = struct {
        format: Format,
        /// Routes a simdgroup accumulates per pass over its weights.
        rows: u32,
        /// Output columns per simdgroup.
        columns: u32,
        /// Words of 8 values a lane loads per column and step: 1 or 2.
        words: u32 = 1,
    };

    pub const Kernel = metal.Kernel(Cfg, .{
        .name = "moe_grouped_gemv",
        .inputs = &.{ "x", "w", "scales", "ids" },
        .outputs = &.{"out"},
        .constants = &.{ "k", "n", "routes" },
        .run = run,
    });

    fn run(b: *Builder, cfg: Cfg) metal.FinishError!void {
        const a = try b.declareArgs(.{
            .x = .{ .buffer = .{ .type = .bfloat, .const_ = true, .restrict = true } },
            .w = .{ .buffer = .{ .type = .uchar, .const_ = true, .restrict = true } },
            .scales = .{ .buffer = .{ .type = .uchar, .const_ = true, .restrict = true } },
            .ids = .{ .buffer = .{ .type = .int, .const_ = true, .restrict = true } },
            .out = .{ .buffer = .{ .type = .bfloat, .restrict = true } },
            .k = .{ .constant = .int },
            .n = .{ .constant = .int },
            .routes = .{ .constant = .int },
            .tg = .{ .threadgroup_position_in_grid = .uint3 },
            .sg = .{ .simdgroup_index_in_threadgroup = .uint },
            .simdgroups = .{ .simdgroups_per_threadgroup = .uint },
            .lane = .{ .thread_index_in_simdgroup = .uint },
        });
        const C = cfg.columns;
        const R = cfg.rows;

        const start = a.tg.y().to(.int);
        const column = a.tg.x().mul(a.simdgroups).add(a.sg).mul(C).to(.int);
        b.returnIf(column.ge(a.n));
        const lane = a.lane.to(.int);
        const expert = a.ids.load(start);

        var zero = b.openIf(expert.lt(0));
        {
            var store = b.openIf(lane.lt(@as(i32, @intCast(C))));
            {
                a.out.store(start.to(.long).mul(a.n).add(column).add(lane), b.constant(.bfloat, 0));
                store.close();
            }
            zero.close();
        }
        b.returnIf(expert.lt(0));
        // Only the first route of each expert works.
        var later = b.openIf(start.gt(0));
        {
            b.returnIf(a.ids.load(start.sub(1)).eq(expert));
            later.close();
        }
        const end = b.local(.int, a.routes);
        var scan = b.openFor(start.add(1), a.routes, 1);
        {
            var other = b.openIf(a.ids.load(scan.iv).ne(expert));
            {
                end.assign(scan.iv);
                scan.breakLoop();
                other.close();
            }
            scan.close();
        }

        // Weight rows of this simdgroup's columns, as units of 8 or 16 values, and their scales.
        const format = cfg.format;
        const W = cfg.words;
        const unit_values: i32 = @intCast(values_per_word * W);
        const units = a.k.div(unit_values);
        const row_bytes = if (format == .nvfp4) a.k.div(2) else a.k;
        var weights: [16]Value = undefined;
        var scales: [16]Value = undefined;
        for (0..C) |c| {
            const row = expert.to(.long).mul(a.n).add(column).add(@as(i32, @intCast(c)));
            weights[c] = a.w.add(row.mul(row_bytes)).to(format.unitType(W));
            scales[c] = a.scales.add(row.mul(a.k.div(format.group())));
        }

        var pass = b.openFor(start, end.rvalue(), @as(i32, @intCast(R)));
        {
            const first = pass.iv;
            const live = b.min(end.sub(first), @as(i32, @intCast(R)));
            var acc: [16 * 16]Value = undefined;
            for (0..R * C) |i| acc[i] = b.local(.float, 0.0);

            var steps = b.openFor(lane, units, simd_width);
            {
                const unit = steps.iv;
                // Each column's values as float4s, scaled by their decoding factor.
                var values: [16][4]Value = undefined;
                var scale: [16]Value = undefined;
                for (0..C) |c| {
                    const code = scales[c].load(unit.mul(unit_values).div(format.group()));
                    const loaded = weights[c].load(unit);
                    for (0..W) |w| {
                        switch (format) {
                            .nvfp4 => {
                                values[c][2 * w], values[c][2 * w + 1] = decodeE2m1(b, if (W == 1) loaded else loaded.at(@as(i32, @intCast(w))));
                            },
                            .mxfp8 => {
                                values[c][2 * w] = decodeE4m3(b, loaded.at(@as(i32, @intCast(2 * w))));
                                values[c][2 * w + 1] = decodeE4m3(b, loaded.at(@as(i32, @intCast(2 * w + 1))));
                            },
                        }
                    }
                    scale[c] = switch (format) {
                        .nvfp4 => decodeE4m3Scale(b, code),
                        .mxfp8 => decodeE8m0Scale(b, code),
                    };
                }
                for (0..R) |r| {
                    var live_row = b.openIf(b.constant(.int, @as(i32, @intCast(r))).lt(live));
                    {
                        const xs = a.x.add(first.add(@as(i32, @intCast(r))).to(.long).mul(a.k).add(unit.mul(unit_values))).to(.bfloat4);
                        var x: [4]Value = undefined;
                        for (0..2 * W) |i| x[i] = xs.load(@as(i32, @intCast(i))).to(.float4);
                        for (0..C) |c| {
                            var dot = b.dot(values[c][0], x[0]);
                            for (1..2 * W) |i| dot = dot.add(b.dot(values[c][i], x[i]));
                            const slot = acc[r * C + c];
                            slot.assign(b.fma(dot, scale[c], slot.rvalue()));
                        }
                        live_row.close();
                    }
                }
                steps.close();
            }

            for (0..R) |r| {
                var live_row = b.openIf(b.constant(.int, @as(i32, @intCast(r))).lt(live));
                {
                    const route = first.add(@as(i32, @intCast(r))).to(.long);
                    for (0..C) |c| {
                        const total = b.simdSum(acc[r * C + c].rvalue());
                        var owner = b.openIf(lane.eq(@as(i32, @intCast(c))));
                        {
                            a.out.store(route.mul(a.n).add(column).add(@as(i32, @intCast(c))), total.to(.bfloat));
                            owner.close();
                        }
                    }
                    live_row.close();
                }
            }
            pass.close();
        }
    }

    /// The 8 FP4 values of `unit`, low nibble first, as two `float4`s scaled by
    /// `2^-14`. An E2M1 code shifted into a half's exponent and mantissa is that
    /// value times `2^-14`, subnormals included.
    fn decodeE2m1(b: *Builder, unit: Value) struct { Value, Value } {
        var pairs: [4]Value = undefined;
        for (&pairs, 0..) |*pair, i| {
            // Nibbles i and i + 4 in the two 16-bit halves.
            const codes = unit.shr(@as(u32, @intCast(4 * i))).bitAnd(@as(u32, 0x000f000f));
            const bits = codes.bitAnd(@as(u32, 0x00070007)).shl(@as(u32, 9)).bitOr(codes.bitAnd(@as(u32, 0x00080008)).shl(@as(u32, 12)));
            pair.* = bits.asType(.half2).to(.float2);
        }
        return .{
            b.vec(.float4, .{ pairs[0].x(), pairs[1].x(), pairs[2].x(), pairs[3].x() }),
            b.vec(.float4, .{ pairs[0].y(), pairs[1].y(), pairs[2].y(), pairs[3].y() }),
        };
    }

    /// An E4M3 scale times `2^22`: shifted into a half, its value is `2^-8` times too
    /// small, and the FP4 values are `2^-14` times too small.
    fn decodeE4m3Scale(b: *Builder, code: Value) Value {
        const bits = code.to(.uint).bitAnd(@as(u32, 0x7f)).shl(@as(u32, 7)).to(.ushort);
        return bits.asType(.half).to(.float).mul(b.constant(.float, 0x1p22));
    }

    /// The 4 E4M3 values of `word`, first byte first, scaled by `2^-8`: an E4M3 code
    /// shifted into a half's exponent and mantissa is that value times `2^-8`,
    /// subnormals included.
    fn decodeE4m3(b: *Builder, word: Value) Value {
        var pairs: [2]Value = undefined;
        for (&pairs, 0..) |*pair, i| {
            // Bytes i and i + 2 in the two 16-bit halves.
            const codes = word.shr(@as(u32, @intCast(8 * i))).bitAnd(@as(u32, 0x00ff00ff));
            const bits = codes.bitAnd(@as(u32, 0x00800080)).shl(@as(u32, 8)).bitOr(codes.bitAnd(@as(u32, 0x007f007f)).shl(@as(u32, 7)));
            pair.* = bits.asType(.half2).to(.float2);
        }
        return b.vec(.float4, .{ pairs[0].x(), pairs[1].x(), pairs[0].y(), pairs[1].y() });
    }

    /// An E8M0 scale, `2^(code - 127)`, times `2^8` for the E4M3 values' factor.
    fn decodeE8m0Scale(b: *Builder, code: Value) Value {
        return code.to(.uint).shl(@as(u32, 23)).asType(.float).mul(b.constant(.float, 0x1p8));
    }

    /// Threadgroups and threads for `routes` sorted routes producing `n` columns.
    pub fn launch(cfg: Cfg, n: i64, routes: i64, simdgroups: u32) struct { grid: [3]u32, threadgroup: [3]u32 } {
        const columns_per_threadgroup = cfg.columns * simdgroups;
        return .{
            .grid = .{ @intCast(@divExact(n, columns_per_threadgroup)), @intCast(routes), 1 },
            .threadgroup = .{ simd_width * simdgroups, 1, 1 },
        };
    }
};

test "moe_grouped_gemv emits verified modules" {
    const allocator = std.testing.allocator;
    inline for (.{ Format.nvfp4, Format.mxfp8 }) |format| {
        inline for (.{ 1, 8 }) |rows| inline for (.{ 1, 2 }) |words| {
            allocator.free(try GroupedGemv.Kernel.emit(allocator, .{ .format = format, .rows = rows, .columns = 4, .words = words }));
        };
    }
}
