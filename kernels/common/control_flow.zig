//! `scf` scope wrappers, generic over the frontend's `BuilderT` and `ValueT`.
//!
//! `BuilderT` must expose `ctx`, `loc()`, `pushBlock`, `popBlock`, `currentBlock`.
//! `ValueT` must have an `inner: *const mlir.Value` field and a `kernel: ?*BuilderT` field.

const std = @import("std");
const mlir = @import("mlir");
const dialects = @import("mlir/dialects");

const scf = dialects.scf;

pub fn tupleArity(comptime T: type, comptime what: []const u8) comptime_int {
    const info = @typeInfo(T);
    if (info != .@"struct" or !info.@"struct".is_tuple)
        @compileError(what ++ " must be a tuple literal like `.{ v1, v2 }`");
    return info.@"struct".fields.len;
}

pub fn emitScfYield(
    comptime ValueT: type,
    kernel: anytype,
    block: *mlir.Block,
    comptime N: usize,
    values: anytype,
    comptime what: []const u8,
) void {
    const info = @typeInfo(@TypeOf(values));
    if (info != .@"struct" or !info.@"struct".is_tuple)
        @compileError(what ++ " expects a tuple literal");
    if (info.@"struct".fields.len != N)
        @compileError(what ++ ": yield arity must match the scope's declared arity");
    var buf: [N]*const mlir.Value = undefined;
    inline for (info.@"struct".fields, 0..) |f, i| {
        if (f.type != ValueT)
            @compileError(what ++ ": every tuple element must be a Value");
        buf[i] = @field(values, f.name).inner;
    }
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

        const Self = @This();

        pub fn yield(self: *Self, values: anytype) void {
            const k = self.kernel;
            emitScfYield(ValueT, k, self.body, N, values, "ForScope.yield");
            k.popBlock();
            const for_op = scf.for_(
                k.ctx,
                self.lb_inner,
                self.ub_inner,
                self.step_inner,
                &self.inits_inner,
                self.body,
                .{},
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

        pub fn yieldThen(self: *Self, values: anytype) void {
            const k = self.kernel;
            emitScfYield(ValueT, k, self.then_block, 0, values, "IfOnlyScope.yieldThen");
            k.popBlock();
            const if_op = scf.if_(
                k.ctx,
                self.cond_inner,
                &.{},
                self.then_block,
                null,
                k.loc(),
            );
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
            const k = self.kernel;
            emitScfYield(ValueT, k, self.then_block, N, values, "IfScope.yieldThen");
            k.popBlock();
            k.pushBlock(self.else_block);
        }

        pub fn yieldElse(self: *Self, values: anytype) void {
            const k = self.kernel;
            emitScfYield(ValueT, k, self.else_block, N, values, "IfScope.yieldElse");
            k.popBlock();
            const if_op = scf.if_(
                k.ctx,
                self.cond_inner,
                &self.result_types,
                self.then_block,
                self.else_block,
                k.loc(),
            );
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
            const info = @typeInfo(@TypeOf(forwarded));
            if (info != .@"struct" or !info.@"struct".is_tuple)
                @compileError("WhileScope.yieldBefore: forwarded must be a tuple literal");
            if (info.@"struct".fields.len != M)
                @compileError("WhileScope.yieldBefore: forwarded arity must match after_types arity");
            const k = self.kernel;
            var buf: [M]*const mlir.Value = undefined;
            inline for (info.@"struct".fields, 0..) |f, i| {
                if (f.type != ValueT)
                    @compileError("WhileScope.yieldBefore: every forwarded element must be a Value");
                buf[i] = @field(forwarded, f.name).inner;
            }
            _ = scf.condition(k.ctx, cond.inner, &buf, k.loc()).appendTo(self.before_block);
            k.popBlock();
            k.pushBlock(self.after_block);
            for (0..M) |i| self.after_carried[i] = .{ .inner = self.after_block.argument(i), .kernel = k };
        }

        pub fn yieldAfter(self: *Self, values: anytype) void {
            const k = self.kernel;
            emitScfYield(ValueT, k, self.after_block, N, values, "WhileScope.yieldAfter");
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

test {
    std.testing.refAllDecls(@This());
}
