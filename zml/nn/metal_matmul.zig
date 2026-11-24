//! Metal matrix products of a few activation rows with a large BF16 weight: each
//! simdgroup reads a few weight rows once and applies them to every activation row.
//! XLA's Metal GEMM would instead convert the weight to FP32 in memory first.

const std = @import("std");

const zml = @import("zml/core");
const metal = zml.kernel.metal;
const Builder = metal.Builder;
const Shape = zml.Shape;
const Tensor = zml.Tensor;
const Type = metal.Type;
const Value = metal.Value;

const simd_width = 32;
/// Contracted values a lane reads per step: one 16-byte load of BF16 weights.
const values_per_unit = 8;
const columns = 4;
const rows = 8;
const simdgroups = 2;

/// Activation rows above which XLA's GEMM is used.
const max_rows = 64;

/// `out[m, n] = sum_k x[m, k] * w[n, k]`, accumulated and stored in FP32.
pub const SkinnyMatmul = struct {
    pub const Cfg = struct {
        /// `float` or `bfloat` activations.
        x: Type,
    };

    pub const Kernel = metal.Kernel(Cfg, .{
        .name = "skinny_matmul_f32",
        .inputs = &.{ "x", "w" },
        .outputs = &.{"out"},
        .constants = &.{ "m", "n", "k" },
        .run = run,
    });

    fn run(b: *Builder, cfg: Cfg) metal.FinishError!void {
        const a = try b.declareArgs(.{
            .x = .{ .buffer = .{ .type = cfg.x, .const_ = true, .restrict = true } },
            .w = .{ .buffer = .{ .type = .bfloat, .const_ = true, .restrict = true } },
            .out = .{ .buffer = .{ .type = .float, .restrict = true } },
            .m = .{ .constant = .int },
            .n = .{ .constant = .int },
            .k = .{ .constant = .int },
            .tg = .{ .threadgroup_position_in_grid = .uint3 },
            .sg = .{ .simdgroup_index_in_threadgroup = .uint },
            .lane = .{ .thread_index_in_simdgroup = .uint },
        });
        const x4: Type = .vec(cfg.x.scalar, 4);
        const column = a.tg.x().mul(simdgroups).add(a.sg).mul(columns).to(.int);
        b.returnIf(column.ge(a.n));
        const lane = a.lane.to(.int);
        const units = a.k.div(values_per_unit);
        var weights: [columns]Value = undefined;
        for (&weights, 0..) |*w, c| w.* = a.w.add(column.add(@as(i32, @intCast(c))).to(.long).mul(a.k)).to(.bfloat4);

        var pass = b.openFor(0, a.m, rows);
        {
            const first = pass.iv;
            const live = b.min(a.m.sub(first), rows);
            var acc: [rows * columns]Value = undefined;
            for (&acc) |*slot| slot.* = b.local(.float, 0.0);
            var steps = b.openFor(lane, units, simd_width);
            {
                const unit = steps.iv;
                var low: [columns]Value = undefined;
                var high: [columns]Value = undefined;
                for (weights, 0..) |w, c| {
                    low[c] = w.load(unit.mul(2)).to(.float4);
                    high[c] = w.load(unit.mul(2).add(1)).to(.float4);
                }
                for (0..rows) |r| {
                    var live_row = b.openIf(b.constant(.int, @as(i32, @intCast(r))).lt(live));
                    {
                        const xs = a.x.add(first.add(@as(i32, @intCast(r))).to(.long).mul(a.k).add(unit.mul(values_per_unit))).to(x4);
                        const x0 = xs.load(0).to(.float4);
                        const x1 = xs.load(1).to(.float4);
                        for (0..columns) |c| {
                            const slot = acc[r * columns + c];
                            slot.assign(slot.add(b.dot(low[c], x0).add(b.dot(high[c], x1))));
                        }
                        live_row.close();
                    }
                }
                steps.close();
            }
            for (0..rows) |r| {
                var live_row = b.openIf(b.constant(.int, @as(i32, @intCast(r))).lt(live));
                {
                    const row = first.add(@as(i32, @intCast(r))).to(.long);
                    for (0..columns) |c| {
                        const total = b.simdSum(acc[r * columns + c].rvalue());
                        var owner = b.openIf(lane.eq(@as(i32, @intCast(c))));
                        {
                            a.out.store(row.mul(a.n).add(column).add(@as(i32, @intCast(c))), total);
                            owner.close();
                        }
                    }
                    live_row.close();
                }
            }
            pass.close();
        }
    }
};

/// Whether `lhs . rhs` over `tag` runs as a skinny Metal product: a BF16 weight `{n, tag}`,
/// at most `max_rows` rows of BF16 or FP32 activations, and the contraction last on both.
/// One row stays with XLA, which reduces it at full bandwidth.
pub fn supports(lhs: Tensor, rhs: Tensor, tag: Shape.Tag) bool {
    if (zml.Compiler.current().platform.target != .metal) return false;
    if (rhs.dtype() != .bf16 or (lhs.dtype() != .bf16 and lhs.dtype() != .f32)) return false;
    const lhs_axis = lhs.shape().hasTag(tag) orelse return false;
    if (rhs.rank() != 2 or lhs_axis + 1 != lhs.rank() or rhs.shape().hasTag(tag) != 1) return false;
    const k = lhs.dim(tag);
    const m = @divExact(@as(i64, @intCast(lhs.count())), k);
    return m >= 2 and m <= max_rows and @rem(k, values_per_unit) == 0 and @rem(rhs.dim(0), columns) == 0;
}

/// `lhs . rhs` over `tag`, accumulated and returned in FP32; see `supports`.
pub fn matmul(lhs: Tensor, rhs: Tensor, tag: Shape.Tag) Tensor {
    const k = lhs.dim(tag);
    const m = @divExact(@as(i64, @intCast(lhs.count())), k);
    const n = rhs.dim(0);
    const out_shape = lhs.shape().remove(tag).appendDim(n, rhs.shape().tag(0)).withDtype(.f32);
    const per_threadgroup = columns * simdgroups;
    const out = SkinnyMatmul.Kernel.call(.{ .x = lhs.reshape(.{ m, k }), .w = rhs }, .{ .out = Shape.init(.{ m, n }, .f32) }, .{
        .cfg = .{ .x = metal.from(lhs.dtype()) },
        .grid = .{ @intCast(@divFloor(n + per_threadgroup - 1, per_threadgroup)), 1, 1 },
        .threadgroup = .{ simd_width * simdgroups, 1, 1 },
        .constants = .{
            .m = .of(@as(i32, @intCast(m))),
            .n = .of(@as(i32, @intCast(n))),
            .k = .of(@as(i32, @intCast(k))),
        },
    }).out;
    return out.reshape(out_shape);
}

test "skinny_matmul_f32 emits verified modules" {
    const allocator = std.testing.allocator;
    inline for (.{ Type.bfloat, Type.float }) |x| allocator.free(try SkinnyMatmul.Kernel.emit(allocator, .{ .x = x }));
}
