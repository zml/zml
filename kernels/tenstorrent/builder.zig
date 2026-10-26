//! Builder for pure TTIR tensor programs embedded in `__tt$ttmlir` calls.
const std = @import("std");
const stdx = @import("stdx");
const mlir = @import("mlir");
const func = @import("mlir/dialects/func");
const ttir = @import("mlir/dialects/tenstorrent");

pub const gated_delta_net = @import("gated_delta_net.zig");

pub const dialects_needed = ttir.dialects_needed;
pub const FinishError = error{ InvalidMlir, OutOfMemory } || std.Io.Writer.Error;
pub const Shape = stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK);

pub const Value = struct {
    inner: *const mlir.Value,

    pub fn type_(self: Value) *const mlir.Type {
        return self.inner.type_();
    }
    pub fn tensorType(self: Value) *const mlir.ShapedType {
        return self.type_().isA(mlir.ShapedType).?;
    }
    pub fn shape(self: Value) Shape {
        const t = self.tensorType();
        var dims: Shape = .empty;
        for (0..t.rank()) |i| dims.appendAssumeCapacity(t.dimension(i));
        return dims;
    }
};

pub const Builder = struct {
    allocator: std.mem.Allocator,
    ctx: *mlir.Context,
    module: *mlir.Module,
    block: *mlir.Block,

    pub fn init(allocator: std.mem.Allocator, ctx: *mlir.Context, name: []const u8, inputs: []const *const mlir.Type, outputs: []const *const mlir.Type) !Builder {
        const location = mlir.Location.unknown(ctx);
        const locations = try allocator.alloc(*const mlir.Location, inputs.len);
        defer allocator.free(locations);
        @memset(locations, location);
        const module = mlir.Module.init(location);
        const block = mlir.Block.init(inputs, locations);
        _ = func.func(ctx, .{ .name = name, .args = inputs, .results = outputs, .block = block, .location = location, .verify = false }).appendTo(module.body());
        return .{ .allocator = allocator, .ctx = ctx, .module = module, .block = block };
    }
    pub fn deinit(self: *Builder) void {
        self.module.deinit();
    }
    pub fn loc(self: *const Builder) *const mlir.Location {
        return .unknown(self.ctx);
    }
    pub fn arg(self: *const Builder, index: usize) Value {
        return .{ .inner = self.block.argument(index) };
    }
    fn emit(self: *Builder, op: *mlir.Operation) Value {
        return .{ .inner = op.appendTo(self.block).result(0) };
    }
    pub fn binary(self: *Builder, kind: ttir.Binary, lhs: Value, rhs: Value) Value {
        const ls = lhs.shape();
        const rs = rhs.shape();
        const rank = @max(ls.len, rs.len);
        var dims: Shape = .empty;
        for (0..rank) |i| {
            const l = if (i + ls.len >= rank) ls.get(i + ls.len - rank) else 1;
            const r = if (i + rs.len >= rank) rs.get(i + rs.len - rank) else 1;
            std.debug.assert(l == r or l == 1 or r == 1);
            dims.appendAssumeCapacity(@max(l, r));
        }
        const elem = switch (kind) {
            .gt, .ge, .lt, .le, .eq, .ne => mlir.Type.int(self.ctx, .i1),
            else => lhs.tensorType().elementType(),
        };
        const t = mlir.Type.rankedTensor(dims.constSlice(), elem);
        return self.emit(ttir.binary(self.ctx, kind, lhs.inner, rhs.inner, t, self.loc()));
    }
    pub fn reshape(self: *Builder, input: Value, dims: []const i64) Value {
        var shape32: stdx.BoundedArray(i32, mlir.ShapedType.MAX_RANK) = .empty;
        for (dims) |d| shape32.appendAssumeCapacity(@intCast(d));
        const t = mlir.Type.rankedTensor(dims, input.tensorType().elementType());
        return self.emit(ttir.reshape(self.ctx, input.inner, shape32.constSlice(), t, self.loc()));
    }
    pub fn sum(self: *Builder, input: Value, axis: usize, keep_dim: bool) Value {
        const shape = input.shape();
        std.debug.assert(axis < shape.len);
        var out: Shape = .empty;
        for (shape.constSlice(), 0..) |d, i| {
            if (i != axis) out.appendAssumeCapacity(d) else if (keep_dim) out.appendAssumeCapacity(1);
        }
        const t = mlir.Type.rankedTensor(out.constSlice(), input.tensorType().elementType());
        return self.emit(ttir.sum(self.ctx, input.inner, &.{@intCast(axis)}, keep_dim, t, self.loc()));
    }
    pub fn matmul(self: *Builder, lhs: Value, rhs: Value, transpose_lhs: bool, transpose_rhs: bool) Value {
        const ls = lhs.shape();
        const rs = rhs.shape();
        std.debug.assert(ls.len >= 2 and rs.len == ls.len);
        var dims = ls;
        const m = ls.get(ls.len - (if (transpose_lhs) @as(usize, 1) else 2));
        const lk = ls.get(ls.len - (if (transpose_lhs) @as(usize, 2) else 1));
        const rk = rs.get(rs.len - (if (transpose_rhs) @as(usize, 1) else 2));
        const n = rs.get(rs.len - (if (transpose_rhs) @as(usize, 2) else 1));
        std.debug.assert(lk == rk);
        for (0..ls.len - 2) |i| {
            std.debug.assert(ls.get(i) == rs.get(i) or ls.get(i) == 1 or rs.get(i) == 1);
            dims.set(i, @max(ls.get(i), rs.get(i)));
        }
        dims.set(ls.len - 2, m);
        dims.set(ls.len - 1, n);
        const t = mlir.Type.rankedTensor(dims.constSlice(), lhs.tensorType().elementType());
        return self.emit(ttir.matmul(self.ctx, lhs.inner, rhs.inner, t, transpose_lhs, transpose_rhs, self.loc()));
    }
    /// Select one position from the leading sequence dimension and remove it.
    pub fn sequenceStep(self: *Builder, input: Value, step: usize) Value {
        var dims = input.shape();
        std.debug.assert(dims.len > 0 and step < dims.get(0));
        var begins: stdx.BoundedArray(i32, mlir.ShapedType.MAX_RANK) = .empty;
        var ends = begins;
        var steps = begins;
        for (dims.constSlice(), 0..) |d, i| {
            begins.appendAssumeCapacity(if (i == 0) @intCast(step) else 0);
            ends.appendAssumeCapacity(if (i == 0) @intCast(step + 1) else @intCast(d));
            steps.appendAssumeCapacity(1);
        }
        dims.set(0, 1);
        const t = mlir.Type.rankedTensor(dims.constSlice(), input.tensorType().elementType());
        const slice = self.emit(ttir.sliceStatic(self.ctx, input.inner, begins.constSlice(), ends.constSlice(), steps.constSlice(), t, self.loc()));
        return self.reshape(slice, dims.constSlice()[1..]);
    }
    pub fn concat(self: *Builder, inputs: []const Value, axis: usize) !Value {
        std.debug.assert(inputs.len > 0);
        var dims = inputs[0].shape();
        dims.set(axis, 0);
        const values = try self.allocator.alloc(*const mlir.Value, inputs.len);
        defer self.allocator.free(values);
        for (inputs, 0..) |input, i| {
            const shape = input.shape();
            std.debug.assert(shape.len == dims.len);
            for (shape.constSlice(), 0..) |d, j| {
                if (j == axis) dims.set(j, dims.get(j) + d) else std.debug.assert(d == dims.get(j));
            }
            values[i] = input.inner;
        }
        const t = mlir.Type.rankedTensor(dims.constSlice(), inputs[0].tensorType().elementType());
        return self.emit(ttir.concat(self.ctx, values, @intCast(axis), t, self.loc()));
    }
    pub fn where(self: *Builder, predicate: Value, on_true: Value, on_false: Value) Value {
        std.debug.assert(on_true.type_().eql(on_false.type_()));
        return self.emit(ttir.where(self.ctx, predicate.inner, on_true.inner, on_false.inner, on_true.type_(), self.loc()));
    }
    /// `value` is the storage representation of one element (u16 for bf16).
    pub fn splat(self: *Builder, tensor_type: *const mlir.Type, value: anytype) Value {
        const attr = mlir.Attribute.denseElements(tensor_type.isA(mlir.ShapedType).?, &[_]@TypeOf(value){value});
        return self.emit(.make(self.ctx, "arith.constant", .{
            .results = .{ .flat = &.{tensor_type} },
            .attributes = &.{.named(self.ctx, "value", attr)},
            .location = self.loc(),
        }));
    }
    pub fn returns(self: *Builder, results: []const Value) void {
        var values: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
        for (results) |v| values.appendAssumeCapacity(v.inner);
        _ = func.returns(self.ctx, values.constSlice(), self.loc()).appendTo(self.block);
    }
    pub fn serialize(self: *Builder) FinishError![:0]const u8 {
        if (!self.module.operation().verify()) return error.InvalidMlir;
        var text: std.Io.Writer.Allocating = .init(self.allocator);
        defer text.deinit();
        try text.writer.print("{f}", .{self.module.operation()});
        return self.allocator.dupeSentinel(u8, text.written(), 0);
    }
    pub fn finish(self: *Builder, results: []const Value) FinishError![:0]const u8 {
        self.returns(results);
        return self.serialize();
    }
};

test "batched recurrence primitives produce verified TTIR" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (dialects_needed) |name| mlir.DialectHandle.fromString(name).insertDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    const f32_ty = mlir.Type.float(ctx, .f32);
    const state_ty = mlir.Type.rankedTensor(&.{ 2, 32, 32 }, f32_ty);
    const key_ty = mlir.Type.rankedTensor(&.{ 1, 2, 32 }, f32_ty);
    const output_ty = mlir.Type.rankedTensor(&.{ 2, 32, 1 }, f32_ty);
    var b = try Builder.init(std.testing.allocator, ctx, "recurrent_step", &.{ state_ty, key_ty }, &.{output_ty});
    defer b.deinit();
    const key = b.reshape(b.sequenceStep(b.arg(1), 0), &.{ 2, 32, 1 });
    const output = b.matmul(b.arg(0), key, false, false);
    const ir = try b.finish(&.{output});
    defer std.testing.allocator.free(ir);
    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
    try std.testing.expect(std.mem.indexOf(u8, ir, "ttir.matmul") != null);
}

test "gated DeltaNet emission round-trips for decode and prefill" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (dialects_needed) |name| mlir.DialectHandle.fromString(name).insertDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    const f32_ty = mlir.Type.float(ctx, .f32);
    for ([_]i64{ 1, 3 }) |seq| {
        const q = mlir.Type.rankedTensor(&.{ seq, 2, 32 }, f32_ty);
        const gates = mlir.Type.rankedTensor(&.{ seq, 2 }, f32_ty);
        const state = mlir.Type.rankedTensor(&.{ 2, 32, 32 }, f32_ty);
        const active = mlir.Type.rankedTensor(&.{}, mlir.Type.int(ctx, .i32));
        var b = try Builder.init(std.testing.allocator, ctx, "gated_delta_net", &.{ q, q, q, gates, gates, state, active }, &.{ state, q });
        defer b.deinit();
        try gated_delta_net.run(&b, {});
        const ir = try b.serialize();
        defer std.testing.allocator.free(ir);
        const parsed = try mlir.Module.parse(ctx, ir);
        defer parsed.deinit();
        try std.testing.expect(parsed.operation().verify());
        try std.testing.expect(std.mem.indexOf(u8, ir, "ttir.where") != null);
        if (seq > 1) try std.testing.expect(std.mem.indexOf(u8, ir, "ttir.concat") != null);
    }
}
