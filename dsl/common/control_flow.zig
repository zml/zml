//! `scf` scope wrappers, generic over the frontend's `BuilderT` and `ValueT`,
//! and `BlockStack`, the blocks a builder emits into.
//!
//! `BuilderT` must expose `ctx`, `loc()`, `pushBlock`, `popBlock` and
//! `currentBlock`. The openers also call its `lift` on literal inits and its
//! `scalarTy` on dtype result types, and `forBounds` its `constantLike` on
//! literal bounds.
//! `ValueT` must have an `inner: *const mlir.Value` field and a `kernel: ?*BuilderT` field.

const std = @import("std");
const mlir = @import("mlir");
const scf = @import("mlir/dialects/scf");

/// The blocks a builder emits into: the innermost scope's on top of the
/// function's entry block.
pub const BlockStack = struct {
    blocks: std.ArrayList(*mlir.Block) = .empty,

    pub fn push(self: *BlockStack, allocator: std.mem.Allocator, block: *mlir.Block) void {
        self.blocks.append(allocator, block) catch @panic("BlockStack.push: out of memory");
    }

    pub fn pop(self: *BlockStack) void {
        _ = self.blocks.pop();
    }

    /// Drops every block, keeping the memory for the next function.
    pub fn clear(self: *BlockStack) void {
        self.blocks.clearRetainingCapacity();
    }

    /// The innermost block, else `entry`.
    pub fn current(self: *const BlockStack, entry: ?*mlir.Block) *mlir.Block {
        return self.blocks.getLastOrNull() orelse entry orelse @panic("BlockStack.current: no block, call declareArgs first");
    }

    /// Continues in `block` in place of the innermost block, or of the entry
    /// block when no scope is open.
    pub fn replaceTop(self: *BlockStack, allocator: std.mem.Allocator, block: *mlir.Block) void {
        if (self.blocks.items.len == 0) return self.push(allocator, block);
        self.blocks.items[self.blocks.items.len - 1] = block;
    }
};

/// The length of `T`, a tuple like `.{ v1, v2 }` or an array.
pub fn arity(comptime T: type, comptime what: []const u8) comptime_int {
    return switch (@typeInfo(T)) {
        .array => |a| a.len,
        .@"struct" => |s| if (s.is_tuple) s.field_names.len else @compileError(what ++ " must be a tuple like `.{ v1, v2 }` or an array"),
        else => @compileError(what ++ " must be a tuple like `.{ v1, v2 }` or an array"),
    };
}

/// The MLIR values of `values`, `N` `ValueT`s.
pub fn innerValues(comptime ValueT: type, comptime N: usize, values: anytype, comptime what: []const u8) [N]*const mlir.Value {
    if (arity(@TypeOf(values), what) != N)
        @compileError(what ++ ": the arity must match the scope's");
    var buf: [N]*const mlir.Value = undefined;
    inline for (0..N) |i| {
        if (@TypeOf(values[i]) != ValueT) @compileError(what ++ ": every element must be a Value");
        buf[i] = values[i].inner;
    }
    return buf;
}

/// `values` as `ValueT`s: values as they are, anything else through `kernel.lift`.
fn liftValues(kernel: anytype, comptime ValueT: type, values: anytype, comptime what: []const u8) [arity(@TypeOf(values), what)]ValueT {
    var out: [arity(@TypeOf(values), what)]ValueT = undefined;
    inline for (0..out.len) |i| out[i] = if (@TypeOf(values[i]) == ValueT) values[i] else kernel.lift(values[i]);
    return out;
}

/// The types `types` names: `*const mlir.Type`s, `ValueT`s for their types,
/// or dtypes (`.f32`) for `kernel.scalarTy`.
pub fn resultTypes(kernel: anytype, comptime ValueT: type, types: anytype, comptime what: []const u8) [arity(@TypeOf(types), what)]*const mlir.Type {
    var out: [arity(@TypeOf(types), what)]*const mlir.Type = undefined;
    inline for (0..out.len) |i| {
        const T = @TypeOf(types[i]);
        out[i] = if (T == *const mlir.Type) types[i] else if (T == ValueT) types[i].inner.type_() else kernel.scalarTy(types[i]);
    }
    return out;
}

fn scfYield(comptime ValueT: type, comptime N: usize, kernel: anytype, block: *mlir.Block, values: anytype, comptime what: []const u8) void {
    const buf = innerValues(ValueT, N, values, what);
    _ = scf.yield(kernel.ctx, &buf, kernel.loc()).appendTo(block);
}

pub fn ForScope(comptime BuilderT: type, comptime ValueT: type, comptime N: usize) type {
    return struct {
        kernel: *BuilderT,
        body: *mlir.Block,
        lb_inner: *const mlir.Value,
        ub_inner: *const mlir.Value,
        step_inner: *const mlir.Value,
        inits_inner: [N]*const mlir.Value,
        iv: ValueT,
        carried: [N]ValueT,
        results: [N]ValueT = undefined,
        opts: scf.ForOpts = .{},

        const Self = @This();

        pub fn yield(self: *Self, values: anytype) void {
            if (N == 0) @compileError("ForScope.yield: the loop carries nothing, end it with close()");
            self.end(values);
        }

        pub fn close(self: *Self) void {
            if (N != 0) @compileError("ForScope.close: the loop carries values, end it with yield()");
            self.end(.{});
        }

        fn end(self: *Self, values: anytype) void {
            const k = self.kernel;
            scfYield(ValueT, N, k, self.body, values, "ForScope.yield");
            k.popBlock();
            const for_op = scf.for_(
                k.ctx,
                self.lb_inner,
                self.ub_inner,
                self.step_inner,
                &self.inits_inner,
                self.body,
                self.opts,
                k.loc(),
            );
            _ = for_op.appendTo(k.currentBlock());
            for (0..N) |i| self.results[i] = .{ .inner = for_op.result(i), .kernel = k };
        }
    };
}

pub fn IfOnlyScope(comptime BuilderT: type, comptime ValueT: type) type {
    return struct {
        kernel: *BuilderT,
        cond_inner: *const mlir.Value,
        then_block: *mlir.Block,

        const Self = @This();

        pub fn close(self: *Self) void {
            const k = self.kernel;
            scfYield(ValueT, 0, k, self.then_block, .{}, "IfOnlyScope.close");
            k.popBlock();
            const if_op = scf.if_(k.ctx, self.cond_inner, &.{}, self.then_block, .{}, k.loc());
            _ = if_op.appendTo(k.currentBlock());
        }
    };
}

pub fn IfScope(comptime BuilderT: type, comptime ValueT: type, comptime N: usize) type {
    return struct {
        kernel: *BuilderT,
        cond_inner: *const mlir.Value,
        then_block: *mlir.Block,
        else_block: *mlir.Block,
        result_types: [N]*const mlir.Type,
        results: [N]ValueT = undefined,

        const Self = @This();

        pub fn yieldThen(self: *Self, values: anytype) void {
            if (N == 0) @compileError("IfScope.yieldThen: the if has no results, switch branches with elseBranch()");
            self.endThen(values);
        }

        pub fn yieldElse(self: *Self, values: anytype) void {
            if (N == 0) @compileError("IfScope.yieldElse: the if has no results, end it with close()");
            self.endElse(values);
        }

        pub fn elseBranch(self: *Self) void {
            if (N != 0) @compileError("IfScope.elseBranch: the if has results, switch branches with yieldThen()");
            self.endThen(.{});
        }

        pub fn close(self: *Self) void {
            if (N != 0) @compileError("IfScope.close: the if has results, end it with yieldElse()");
            self.endElse(.{});
        }

        fn endThen(self: *Self, values: anytype) void {
            const k = self.kernel;
            scfYield(ValueT, N, k, self.then_block, values, "IfScope.yieldThen");
            k.popBlock();
            k.pushBlock(self.else_block);
        }

        fn endElse(self: *Self, values: anytype) void {
            const k = self.kernel;
            scfYield(ValueT, N, k, self.else_block, values, "IfScope.yieldElse");
            k.popBlock();
            const if_op = scf.if_(k.ctx, self.cond_inner, &self.result_types, self.then_block, .{ .else_block = self.else_block }, k.loc());
            _ = if_op.appendTo(k.currentBlock());
            for (0..N) |i| self.results[i] = .{ .inner = if_op.result(i), .kernel = k };
        }
    };
}

pub fn WhileScope(comptime BuilderT: type, comptime ValueT: type, comptime N: usize, comptime M: usize) type {
    return struct {
        kernel: *BuilderT,
        before_block: *mlir.Block,
        after_block: *mlir.Block,
        inits_inner: [N]*const mlir.Value,
        after_types: [M]*const mlir.Type,
        before_carried: [N]ValueT,
        after_carried: [M]ValueT = undefined,
        results: [M]ValueT = undefined,

        const Self = @This();

        pub fn yieldBefore(self: *Self, cond: ValueT, forwarded: anytype) void {
            const k = self.kernel;
            const buf = innerValues(ValueT, M, forwarded, "WhileScope.yieldBefore");
            _ = scf.condition(k.ctx, cond.inner, &buf, k.loc()).appendTo(self.before_block);
            k.popBlock();
            k.pushBlock(self.after_block);
            for (0..M) |i| self.after_carried[i] = .{ .inner = self.after_block.argument(i), .kernel = k };
        }

        pub fn yieldAfter(self: *Self, values: anytype) void {
            const k = self.kernel;
            scfYield(ValueT, N, k, self.after_block, values, "WhileScope.yieldAfter");
            k.popBlock();
            const w = scf.while_(
                k.ctx,
                &self.inits_inner,
                &self.after_types,
                self.before_block,
                self.after_block,
                k.loc(),
            );
            _ = w.appendTo(k.currentBlock());
            for (0..M) |i| self.results[i] = .{ .inner = w.result(i), .kernel = k };
        }
    };
}

/// `openFor`'s bounds as values. A bound that isn't a value takes the type of
/// the first one that is (`kernel.constantLike`); with none, `lower` becomes
/// `defaultLower(kernel, lower)` and the others take its type.
pub fn forBounds(
    kernel: anytype,
    comptime ValueT: type,
    lower: anytype,
    upper: anytype,
    step: anytype,
    comptime defaultLower: anytype,
) [3]ValueT {
    const bounds: [3]ValueT = if (@TypeOf(lower) != ValueT and @TypeOf(upper) != ValueT and @TypeOf(step) != ValueT) blk: {
        const lb = defaultLower(kernel, lower);
        break :blk .{ lb, kernel.constantLike(upper, lb), kernel.constantLike(step, lb) };
    } else blk: {
        const first: ValueT = if (@TypeOf(lower) == ValueT) lower else if (@TypeOf(upper) == ValueT) upper else step;
        break :blk .{ kernel.constantLike(lower, first), kernel.constantLike(upper, first), kernel.constantLike(step, first) };
    };
    const t = bounds[0].inner.type_();
    if (!bounds[1].inner.type_().eql(t) or !bounds[2].inner.type_().eql(t))
        std.debug.panic("openFor: bounds of types {f}, {f} and {f}", .{ t, bounds[1].inner.type_(), bounds[2].inner.type_() });
    return bounds;
}

/// Opens `for iv in [bounds[0], bounds[1]) step bounds[2]` carrying `inits`,
/// values or literals `kernel.lift` takes.
pub fn openFor(
    kernel: anytype,
    comptime ValueT: type,
    bounds: [3]ValueT,
    inits: anytype,
    opts: scf.ForOpts,
) ForScope(@TypeOf(kernel.*), ValueT, arity(@TypeOf(inits), "openFor: inits")) {
    const N = comptime arity(@TypeOf(inits), "openFor: inits");
    const values = liftValues(kernel, ValueT, inits, "openFor: inits");
    var types: [N + 1]*const mlir.Type = undefined;
    types[0] = bounds[0].inner.type_();
    var inits_inner: [N]*const mlir.Value = undefined;
    for (values, 0..) |v, i| {
        types[i + 1] = v.inner.type_();
        inits_inner[i] = v.inner;
    }
    const locs: [N + 1]*const mlir.Location = @splat(kernel.loc());
    const body = mlir.Block.init(&types, &locs);
    kernel.pushBlock(body);
    var carried: [N]ValueT = undefined;
    for (&carried, 0..) |*c, i| c.* = .{ .inner = body.argument(i + 1), .kernel = kernel };
    return .{
        .kernel = kernel,
        .body = body,
        .lb_inner = bounds[0].inner,
        .ub_inner = bounds[1].inner,
        .step_inner = bounds[2].inner,
        .inits_inner = inits_inner,
        .iv = .{ .inner = body.argument(0), .kernel = kernel },
        .carried = carried,
        .opts = opts,
    };
}

pub fn openIf(kernel: anytype, comptime ValueT: type, cond: ValueT) IfOnlyScope(@TypeOf(kernel.*), ValueT) {
    const then_block = mlir.Block.init(&.{}, &.{});
    kernel.pushBlock(then_block);
    return .{ .kernel = kernel, .cond_inner = cond.inner, .then_block = then_block };
}

/// Opens an `if` with an `else`, of results of `result_types` (see `resultTypes`).
pub fn openIfElse(
    kernel: anytype,
    comptime ValueT: type,
    cond: ValueT,
    result_types: anytype,
) IfScope(@TypeOf(kernel.*), ValueT, arity(@TypeOf(result_types), "openIfElse: result_types")) {
    const then_block = mlir.Block.init(&.{}, &.{});
    const else_block = mlir.Block.init(&.{}, &.{});
    kernel.pushBlock(then_block);
    return .{
        .kernel = kernel,
        .cond_inner = cond.inner,
        .then_block = then_block,
        .else_block = else_block,
        .result_types = resultTypes(kernel, ValueT, result_types, "openIfElse: result_types"),
    };
}

/// Opens an `scf.while` carrying `inits` into its condition region, which
/// forwards values of `after_types` (see `resultTypes`) to its body.
pub fn openWhile(
    kernel: anytype,
    comptime ValueT: type,
    inits: anytype,
    after_types: anytype,
) WhileScope(@TypeOf(kernel.*), ValueT, arity(@TypeOf(inits), "openWhile: inits"), arity(@TypeOf(after_types), "openWhile: after_types")) {
    const N = comptime arity(@TypeOf(inits), "openWhile: inits");
    const M = comptime arity(@TypeOf(after_types), "openWhile: after_types");
    const values = liftValues(kernel, ValueT, inits, "openWhile: inits");
    var before_types: [N]*const mlir.Type = undefined;
    var inits_inner: [N]*const mlir.Value = undefined;
    for (values, 0..) |v, i| {
        before_types[i] = v.inner.type_();
        inits_inner[i] = v.inner;
    }
    const before_locs: [N]*const mlir.Location = @splat(kernel.loc());
    const before_block = mlir.Block.init(&before_types, &before_locs);
    const after_types_ = resultTypes(kernel, ValueT, after_types, "openWhile: after_types");
    const after_locs: [M]*const mlir.Location = @splat(kernel.loc());
    const after_block = mlir.Block.init(&after_types_, &after_locs);
    kernel.pushBlock(before_block);
    var before_carried: [N]ValueT = undefined;
    for (&before_carried, 0..) |*c, i| c.* = .{ .inner = before_block.argument(i), .kernel = kernel };
    return .{
        .kernel = kernel,
        .before_block = before_block,
        .after_block = after_block,
        .inits_inner = inits_inner,
        .after_types = after_types_,
        .before_carried = before_carried,
    };
}

test {
    std.testing.refAllDecls(@This());
}
