//! Structured control flow with loop-carried state of any size, for the MegaMoE device code.
//!
//! DeepGEMM's kernel mutates locals inside loops and branches (`stage_idx`, `phase`, the
//! scheduler's pipeline, the dispatch warps' expert cursor, ...). The builder only has `scf`, so
//! that state is threaded through `scf.for` / `scf.while` / `scf.if` as block arguments and
//! results. Here a state is any Zig struct whose fields are `V`, `[N]V`, `[]V` or nested such
//! structs (plain Zig scalars pass through unchanged); it is flattened to a list of `V` and
//! rebuilt with the same shape (a `[]V` keeps the length of the template it was built from).
//! The builder's own scopes take tuples of a comptime arity; these take a run-time number of
//! values, so a config only known when the program is emitted (BLOCK_M, the number of ranks,
//! ...) can size the state.
//!
//!   var loop = flow.While(S, F).open(b, init);    // before region: `loop.state` (S)
//!   const after = loop.cond(c, forwarded);        // after region: `after` (F)
//!   const results = loop.end(next);               // after the loop: `results` (F)
const std = @import("std");

const mlir = @import("mlir");
const dialects = @import("mlir/dialects");
const cute = @import("kernels/cute/builder");

const B = cute.Builder;
const V = cute.Value;
const scf = dialects.scf;

// ==================== flattening ====================

fn countLeaves(s: anytype) usize {
    const T = @TypeOf(s);
    if (T == V) return 1;
    switch (@typeInfo(T)) {
        .array => {
            var n: usize = 0;
            for (s) |x| n += countLeaves(x);
            return n;
        },
        .pointer => |p| {
            comptime std.debug.assert(p.size == .slice);
            var n: usize = 0;
            for (s) |x| n += countLeaves(x);
            return n;
        },
        .@"struct" => |st| {
            var n: usize = 0;
            inline for (st.field_names) |f| n += countLeaves(@field(s, f));
            return n;
        },
        .void, .int, .bool, .float, .@"enum" => return 0,
        else => @compileError("flow: unsupported state field type " ++ @typeName(T)),
    }
}

fn collect(s: anytype, out: []V, i: *usize) void {
    const T = @TypeOf(s);
    if (T == V) {
        out[i.*] = s;
        i.* += 1;
        return;
    }
    switch (@typeInfo(T)) {
        .array, .pointer => for (s) |x| collect(x, out, i),
        .@"struct" => |st| inline for (st.field_names) |f| collect(@field(s, f), out, i),
        .void, .int, .bool, .float, .@"enum" => {},
        else => @compileError("flow: unsupported state field type " ++ @typeName(T)),
    }
}

fn rebuild(comptime T: type, b: *B, template: T, vals: []const V, i: *usize) T {
    if (T == V) {
        const v = vals[i.*];
        i.* += 1;
        return v;
    }
    switch (@typeInfo(T)) {
        .array => |a| {
            var out: T = undefined;
            for (&out, template) |*o, t| o.* = rebuild(a.child, b, t, vals, i);
            return out;
        },
        .pointer => |p| {
            const out = b.arena.allocator().alloc(p.child, template.len) catch @panic("OOM");
            for (out, template) |*o, t| o.* = rebuild(p.child, b, t, vals, i);
            return out;
        },
        .@"struct" => |st| {
            var out: T = template;
            inline for (st.field_names, st.field_types) |f, F| @field(out, f) = rebuild(F, b, @field(template, f), vals, i);
            return out;
        },
        // Plain Zig values (configuration) pass through unchanged.
        .void, .int, .bool, .float, .@"enum" => return template,
        else => @compileError("flow: unsupported state field type " ++ @typeName(T)),
    }
}

/// The values of a state, in field order.
pub fn flatten(b: *B, s: anytype) []V {
    const out = b.arena.allocator().alloc(V, countLeaves(s)) catch @panic("OOM");
    var i: usize = 0;
    collect(s, out, &i);
    return out;
}

/// A state of the shape of `template` from `vals` (as `flatten` orders them).
pub fn unflatten(comptime T: type, b: *B, template: T, vals: []const V) T {
    var i: usize = 0;
    const out = rebuild(T, b, template, vals, &i);
    std.debug.assert(i == vals.len);
    return out;
}

fn inners(b: *B, vals: []const V) []*const mlir.Value {
    const out = b.arena.allocator().alloc(*const mlir.Value, vals.len) catch @panic("OOM");
    for (out, vals) |*o, v| o.* = v.inner;
    return out;
}

fn typesOf(b: *B, vals: []const V) []*const mlir.Type {
    const out = b.arena.allocator().alloc(*const mlir.Type, vals.len) catch @panic("OOM");
    for (out, vals) |*o, v| o.* = v.type_();
    return out;
}

fn locsOf(b: *B, n: usize) []*const mlir.Location {
    const out = b.arena.allocator().alloc(*const mlir.Location, n) catch @panic("OOM");
    for (out) |*o| o.* = b.loc();
    return out;
}

fn blockArgs(b: *B, block: *mlir.Block, first: usize, n: usize) []V {
    const out = b.arena.allocator().alloc(V, n) catch @panic("OOM");
    for (out, 0..) |*o, i| o.* = .{ .inner = block.argument(first + i), .kernel = b };
    return out;
}

fn yieldTo(b: *B, block: *mlir.Block, vals: []const V) void {
    _ = scf.yield(b.ctx, inners(b, vals), b.loc()).appendTo(block);
}

// `scf.condition` and `scf.for` without the dialect helpers' 64-operand bound (the combine's
// reduction loop carries 84 values).

fn conditionOp(b: *B, c: V, args: []const V) *mlir.Operation {
    const operands = b.arena.allocator().alloc(*const mlir.Value, args.len + 1) catch @panic("OOM");
    operands[0] = c.inner;
    for (operands[1..], args) |*o, a| o.* = a.inner;
    return mlir.Operation.make(b.ctx, "scf.condition", .{
        .operands = .{ .flat = operands },
        .verify = false,
        .location = b.loc(),
    });
}

fn forOp(b: *B, lower: V, upper: V, step: V, inits: []const *const mlir.Value, body: *mlir.Block) *mlir.Operation {
    const a = b.arena.allocator();
    const operands = a.alloc(*const mlir.Value, inits.len + 3) catch @panic("OOM");
    operands[0] = lower.inner;
    operands[1] = upper.inner;
    operands[2] = step.inner;
    @memcpy(operands[3..], inits);
    const result_types = a.alloc(*const mlir.Type, inits.len) catch @panic("OOM");
    for (result_types, inits) |*t, v| t.* = v.type_();
    return mlir.Operation.make(b.ctx, "scf.for", .{
        .operands = .{ .flat = operands },
        .results = .{ .flat = result_types },
        .blocks = &.{body},
        // As `scf.for_`: operands defined outside become reachable once appended.
        .verify = false,
        .location = b.loc(),
    });
}

// ==================== scf.while ====================

/// `while (cond(state)) state = body(forwarded)`: the before region gets the carried state `S`
/// and forwards `F` (to the after region and as the results).
pub fn While(comptime S: type, comptime F: type) type {
    return struct {
        b: *B,
        before: *mlir.Block,
        after: *mlir.Block = undefined,
        inits: []*const mlir.Value,
        init_template: S,
        fwd_template: F = undefined,
        fwd_types: []*const mlir.Type = &.{},
        /// The carried state inside the before region.
        state: S,

        const Self = @This();

        pub fn open(b: *B, init: S) Self {
            const vals = flatten(b, init);
            const before = mlir.Block.init(typesOf(b, vals), locsOf(b, vals.len));
            b.pushBlock(before);
            return .{
                .b = b,
                .before = before,
                .inits = inners(b, vals),
                .init_template = init,
                .state = unflatten(S, b, init, blockArgs(b, before, 0, vals.len)),
            };
        }

        /// Ends the before region: continue with the after region while `c` (i1) holds.
        /// Returns the forwarded state inside the after region.
        pub fn cond(self: *Self, c: V, fwd: F) F {
            const b = self.b;
            const vals = flatten(b, fwd);
            _ = conditionOp(b, c, vals).appendTo(self.before);
            b.popBlock();
            self.fwd_template = fwd;
            self.fwd_types = typesOf(b, vals);
            self.after = mlir.Block.init(self.fwd_types, locsOf(b, vals.len));
            b.pushBlock(self.after);
            return unflatten(F, b, fwd, blockArgs(b, self.after, 0, vals.len));
        }

        /// Ends the after region with the next carried state; returns the loop results (the
        /// state the before region forwarded when it stopped).
        pub fn end(self: *Self, next: S) F {
            const b = self.b;
            const vals = flatten(b, next);
            std.debug.assert(vals.len == self.inits.len);
            yieldTo(b, self.after, vals);
            b.popBlock();
            const op = scf.while_(b.ctx, self.inits, self.fwd_types, self.before, self.after, b.loc());
            _ = op.appendTo(b.currentBlock());
            const out = b.arena.allocator().alloc(V, self.fwd_types.len) catch @panic("OOM");
            for (out, 0..) |*o, i| o.* = .{ .inner = op.result(i), .kernel = b };
            return unflatten(F, b, self.fwd_template, out);
        }
    };
}

// ==================== scf.for ====================

/// `for (iv = lower; iv < upper; iv += step)` (signed bounds, like `scf.for`) with carried `S`.
pub fn For(comptime S: type) type {
    return struct {
        b: *B,
        body: *mlir.Block,
        lower: V,
        upper: V,
        step: V,
        inits: []*const mlir.Value,
        template: S,
        iv: V,
        state: S,

        const Self = @This();

        pub fn open(b: *B, lower: anytype, upper: anytype, step: anytype, init: S) Self {
            const lb = b.lift(lower);
            const ub = b.liftAs(upper, lb.dtype());
            const st = b.liftAs(step, lb.dtype());
            const vals = flatten(b, init);
            const types = b.arena.allocator().alloc(*const mlir.Type, vals.len + 1) catch @panic("OOM");
            types[0] = lb.type_();
            for (types[1..], vals) |*t, v| t.* = v.type_();
            const body = mlir.Block.init(types, locsOf(b, vals.len + 1));
            b.pushBlock(body);
            return .{
                .b = b,
                .body = body,
                .lower = lb,
                .upper = ub,
                .step = st,
                .inits = inners(b, vals),
                .template = init,
                .iv = .{ .inner = body.argument(0), .kernel = b },
                .state = unflatten(S, b, init, blockArgs(b, body, 1, vals.len)),
            };
        }

        pub fn end(self: *Self, next: S) S {
            const b = self.b;
            const vals = flatten(b, next);
            std.debug.assert(vals.len == self.inits.len);
            yieldTo(b, self.body, vals);
            b.popBlock();
            const op = forOp(b, self.lower, self.upper, self.step, self.inits, self.body);
            _ = op.appendTo(b.currentBlock());
            const out = b.arena.allocator().alloc(V, vals.len) catch @panic("OOM");
            for (out, 0..) |*o, i| o.* = .{ .inner = op.result(i), .kernel = b };
            return unflatten(S, b, self.template, out);
        }
    };
}

// ==================== scf.if with results ====================

/// `r = c ? then : else` where both branches emit code: `open`, then-branch code,
/// `orElse(then_result)`, else-branch code, `end(else_result)`.
pub fn IfElse(comptime S: type) type {
    return struct {
        b: *B,
        c: V,
        then_block: *mlir.Block,
        else_block: *mlir.Block,
        types: []*const mlir.Type = &.{},
        template: S = undefined,

        const Self = @This();

        pub fn open(b: *B, c: V) Self {
            const then_block = mlir.Block.init(&.{}, &.{});
            const else_block = mlir.Block.init(&.{}, &.{});
            b.pushBlock(then_block);
            return .{ .b = b, .c = c, .then_block = then_block, .else_block = else_block };
        }

        pub fn orElse(self: *Self, then_result: S) void {
            const b = self.b;
            const vals = flatten(b, then_result);
            self.template = then_result;
            self.types = typesOf(b, vals);
            yieldTo(b, self.then_block, vals);
            b.popBlock();
            b.pushBlock(self.else_block);
        }

        pub fn end(self: *Self, else_result: S) S {
            const b = self.b;
            const vals = flatten(b, else_result);
            std.debug.assert(vals.len == self.types.len);
            yieldTo(b, self.else_block, vals);
            b.popBlock();
            const op = scf.if_(b.ctx, self.c.inner, self.types, self.then_block, self.else_block, b.loc());
            _ = op.appendTo(b.currentBlock());
            const out = b.arena.allocator().alloc(V, vals.len) catch @panic("OOM");
            for (out, 0..) |*o, i| o.* = .{ .inner = op.result(i), .kernel = b };
            return unflatten(S, b, self.template, out);
        }
    };
}

/// `if (c) { ... }` without results (the builder's `openIf`).
pub const If = struct {
    scope: cute.IfOnlyScope,

    pub fn open(b: *B, c: V) If {
        return .{ .scope = b.openIf(c) };
    }

    pub fn end(self: *If) void {
        self.scope.yieldThen(.{});
    }
};

/// `while (cond.pending(b)) {}`: a spin loop without timeout (DeepGEMM's bare `while (...);`).
pub fn spin(b: *B, cond: anytype) void {
    var loop = b.openWhile(.{}, .{});
    loop.yieldBefore(cond.pending(b), .{});
    loop.yieldAfter(.{});
}

/// `for (i = start; i < end; i += step) body(i)` without carried state (a run-time loop).
pub fn forRange(b: *B, start: V, end_: anytype, step: anytype) For(void) {
    return For(void).open(b, start, end_, step, {});
}
