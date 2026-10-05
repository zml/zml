const std = @import("std");

const dialects = @import("mlir/dialects");
const arith = dialects.arith;
const func = dialects.func;
const mlir = @import("mlir");
const tcl = @import("mlir/dialects/tcl");

pub const dtypes = @import("dtype.zig");

pub const DType = dtypes.DType;
pub const Tactic = tcl.Tactic;
pub const ReduceMode = tcl.ReduceMode;
pub const Predicate = tcl.Predicate;

pub const dialects_needed = tcl.dialects_needed;

pub const FinishError = error{ InvalidMlir, OutOfMemory } || std.Io.Writer.Error;

test {
    std.testing.refAllDecls(@This());
}

/// A named TCL axis with its extent, from `Builder.axis`.
pub const Axis = struct {
    name: []const u8,
    size: i64,
};

/// A TCL expression over axes and integers, for composite axes
/// (`C: Axis = (A * B)`) and DRAM mappings (`D @ Dp`, `F / 2`, `Broadcast`).
/// `Builder.expr` makes the nodes.
pub const Expr = union(enum) {
    int: i64,
    axis: Axis,
    broadcast,
    node: *const Node,

    pub const Kind = enum { add, sub, mul, div, rem, exact_div, stride, modulo, padding, resize, pair };
    pub const Node = struct { kind: Kind, lhs: Expr, rhs: Expr };

    /// The value of integer arithmetic over axes; null for mappings.
    pub fn value(self: Expr) ?i64 {
        return switch (self) {
            .int => |n| n,
            .axis => |a| a.size,
            .broadcast => null,
            .node => |n| {
                const l = n.lhs.value() orelse return null;
                const r = n.rhs.value() orelse return null;
                return switch (n.kind) {
                    .add => l + r,
                    .sub => l - r,
                    .mul => l * r,
                    .div, .exact_div => if (r == 0) null else @divFloor(l, r),
                    .rem => if (r == 0) null else @mod(l, r),
                    else => null,
                };
            },
        };
    }
};

/// `dram(chip | inner)/T`: a tensor in an explicit DRAM mapping, Python's
/// `tcl.Dram[T, chip, inner, tcl.original[...]]`. `inner` defaults to the
/// tensor's axes.
pub const Dram = struct {
    chip: []const Expr = &.{.broadcast},
    inner: ?[]const Expr = null,
    original: []const Axis = &.{},
};

/// `@ context(operator = {...}, heuristic_hint = {...})`, Python's
/// `tcl.context(layout=..., heuristic_hint=...)`.
pub const Context = struct {
    layout: ?Layout = null,
    heuristic_hint: []const Hint = &.{},

    /// `{Chip: ..., Cluster: ..., Split: [...]}`: `chip` and `cluster` are an
    /// `.axis` or `.broadcast`.
    pub const Layout = struct {
        chip: Expr,
        cluster: ?Expr = null,
        split: []const Axis = &.{},
    };

    pub const Hint = struct {
        name: []const u8,
        value: union(enum) { int: i64, float: f64, boolean: bool, axis: Axis },
    };
};

/// The VE (vector engine) instructions, named like `furiosa.tcl.operation`.
pub const VeOp = enum {
    exp,
    neg_exp,
    sqrt,
    tanh,
    sigmoid,
    erf,
    log,
    sin,
    cos,
    negf,
    absf,
    addi,
    addi_sat,
    subi,
    subi_sat,
    mul_q31,
    muli,
    shl,
    shl_sat,
    shr_logical,
    shr_arith,
    shr_arith_round,
    bit_andi,
    bit_ori,
    bit_xori,
    mini,
    maxi,
    absmini,
    absmaxi,
    addf,
    subf,
    mulf,
    divf,
    bit_andf,
    bit_orf,
    bit_xorf,
    minf,
    maxf,
    absminf,
    absmaxf,
    reinterpret_as_float,
    reinterpret_as_int,
    to_float,
    to_int,

    pub fn arity(self: VeOp) usize {
        return switch (self) {
            .exp, .neg_exp, .sqrt, .tanh, .sigmoid, .erf, .log, .sin, .cos, .negf, .absf, .reinterpret_as_float, .reinterpret_as_int, .to_float, .to_int => 1,
            else => 2,
        };
    }

    /// VE registers are f32 or i32, according to the instruction.
    pub fn result(self: VeOp) DType {
        return switch (self) {
            .exp, .neg_exp, .sqrt, .tanh, .sigmoid, .erf, .log, .sin, .cos, .negf, .absf, .addf, .subf, .mulf, .divf, .bit_andf, .bit_orf, .bit_xorf, .minf, .maxf, .absminf, .absmaxf, .reinterpret_as_float, .to_float => .f32,
            else => .i32,
        };
    }
};

/// A graph-level tensor: a kernel argument, or the result of a tensor or
/// graph operation (`x: [A, B]/bf16` in TCL).
pub const Tensor = struct {
    inner: *const mlir.Value,
    dtype: DType,
    axes: []const Axis,
};

/// A value inside a tensor operation (`%N` in TCL): the result of a read,
/// the contraction or a VE instruction.
pub const Value = struct {
    op: *TensorOperation,
    inner: *const mlir.Value,
    dtype: DType,
    axes: []const Axis,

    pub fn unary(self: Value, comptime opcode: VeOp) Value {
        return self.op.ve(opcode, .{self});
    }

    /// `rhs` is a `Value`, a `Tensor` (read on use) or a scalar literal.
    pub fn binary(self: Value, comptime opcode: VeOp, rhs: anytype) Value {
        return self.op.ve(opcode, .{ self, rhs });
    }

    pub fn reduce(self: Value, axes: []const Axis, mode: ReduceMode) Value {
        return self.op.reduce(self, axes, mode);
    }

    pub fn addf(self: Value, rhs: anytype) Value {
        return self.binary(.addf, rhs);
    }
    pub fn subf(self: Value, rhs: anytype) Value {
        return self.binary(.subf, rhs);
    }
    pub fn mulf(self: Value, rhs: anytype) Value {
        return self.binary(.mulf, rhs);
    }
    pub fn divf(self: Value, rhs: anytype) Value {
        return self.binary(.divf, rhs);
    }
    pub fn maxf(self: Value, rhs: anytype) Value {
        return self.binary(.maxf, rhs);
    }
    pub fn minf(self: Value, rhs: anytype) Value {
        return self.binary(.minf, rhs);
    }
    pub fn addi(self: Value, rhs: anytype) Value {
        return self.binary(.addi, rhs);
    }
    pub fn subi(self: Value, rhs: anytype) Value {
        return self.binary(.subi, rhs);
    }
    pub fn muli(self: Value, rhs: anytype) Value {
        return self.binary(.muli, rhs);
    }
    pub fn exp(self: Value) Value {
        return self.unary(.exp);
    }
    pub fn sqrt(self: Value) Value {
        return self.unary(.sqrt);
    }
    pub fn tanh(self: Value) Value {
        return self.unary(.tanh);
    }
    pub fn sigmoid(self: Value) Value {
        return self.unary(.sigmoid);
    }
    pub fn log(self: Value) Value {
        return self.unary(.log);
    }
    pub fn negf(self: Value) Value {
        return self.unary(.negf);
    }
    pub fn toFloat(self: Value) Value {
        return self.unary(.to_float);
    }
    pub fn toInt(self: Value) Value {
        return self.unary(.to_int);
    }
    pub fn toFp(self: Value, int_width: u5) Value {
        return self.op.toFp(self, int_width);
    }
    pub fn toFxp(self: Value, int_width: u5) Value {
        return self.op.toFxp(self, int_width);
    }
};

/// One TCL tensor operation, `tk.<Tactic>(...)`: reads, at most one
/// contraction, VE instructions, then `commit` writes its result tensor.
/// Like `@tcl.tensor_operation`, every instruction after the reads is
/// pipelined: `commit` takes the last one.
pub const TensorOperation = struct {
    pub const Options = struct {
        /// Guessed from the instructions like the Python DSL when null.
        tactic: ?Tactic = null,
        context: ?Context = null,
    };

    pub const FetchOptions = struct {
        typecast_to: ?DType = null,
        broadcast_to: ?[]const Axis = null,
        subtraction: ?Tensor = null,
        table_lookup: ?Tensor = null,
        /// Padding along an axis, applied by a `slide` over it.
        pad: []const Pad = &.{},
        /// Sliding windows: each replaces its axis by `window_axis, frame_axis`.
        slide: []const Slide = &.{},
    };

    pub const Slide = struct {
        axis: Axis,
        frame_axis: Axis,
        window_axis: Axis,
        /// The window before dilation; `window_axis.size` by default.
        undilated_window: ?i64 = null,
        stride: i64 = 1,
        dilation: i64 = 1,
    };

    pub const Pad = struct {
        axis: Axis,
        left: i64 = 0,
        right: i64 = 0,
        fill: f64 = 0,
    };

    pub const CommitOptions = struct {
        dtype: ?DType = null,
        /// A permutation of the committed value's axes.
        axes: ?[]const Axis = null,
    };

    b: *Builder,
    tactic: ?Tactic,
    context: ?Context,
    body: *mlir.Block,
    inputs: std.ArrayList(Tensor) = .empty,
    plain_reads: std.ArrayList(struct { tensor: *const mlir.Value, value: Value }) = .empty,
    first_compute: ?*mlir.Operation = null,
    last: ?*const mlir.Value = null,
    has_dpe: bool = false,
    has_ve: bool = false,
    has_reduce: bool = false,
    has_select: bool = false,

    /// `read(t, typecast_to=..., broadcast_to=..., ...)`.
    pub fn fetch(self: *TensorOperation, t: Tensor, opts: FetchOptions) Value {
        const b = self.b;
        const ctx = b.ctx;
        var operands: [3]*const mlir.Value = .{ self.input(t), undefined, undefined };
        var n: usize = 1;
        var fields: [6]mlir.NamedAttribute = undefined;
        var nfields: usize = 0;
        var dtype = t.dtype;
        var axes = t.axes;
        if (opts.subtraction) |s| {
            operands[n] = self.input(s);
            fields[nfields] = .named(ctx, "subtraction", .int(ctx, .i64, n));
            n += 1;
            nfields += 1;
        }
        if (opts.table_lookup) |table| {
            operands[n] = self.input(table);
            fields[nfields] = .named(ctx, "table_lookup", .int(ctx, .i64, n));
            dtype = table.dtype;
            n += 1;
            nfields += 1;
        }
        if (opts.typecast_to) |to| {
            fields[nfields] = .named(ctx, "typecast_to", .typeAttr(to.toMlir(ctx)));
            dtype = to;
            nfields += 1;
        }
        if (opts.pad.len > 0) {
            const pads = b.arena.allocator().alloc(mlir.NamedAttribute, opts.pad.len) catch @panic("OOM");
            for (pads, opts.pad) |*dst, p| dst.* = .named(ctx, p.axis.name, .array(ctx, &.{
                .int(ctx, .i64, p.left),
                .int(ctx, .i64, p.right),
                .float(ctx, .f64, p.fill),
            }));
            fields[nfields] = .named(ctx, "pad", .dict(ctx, pads));
            nfields += 1;
        }
        if (opts.slide.len > 0) {
            const slides = b.arena.allocator().alloc(mlir.NamedAttribute, opts.slide.len) catch @panic("OOM");
            var slid: std.ArrayList(Axis) = .empty;
            slid.appendSlice(b.arena.allocator(), axes) catch @panic("OOM");
            for (slides, opts.slide) |*dst, sl| {
                dst.* = .named(ctx, sl.axis.name, .dict(ctx, &.{
                    .named(ctx, "undilated_window", .int(ctx, .i64, sl.undilated_window orelse sl.window_axis.size)),
                    .named(ctx, "frame_axis", b.axisAttr(sl.frame_axis)),
                    .named(ctx, "window_axis", b.axisAttr(sl.window_axis)),
                    .named(ctx, "stride", .int(ctx, .i64, sl.stride)),
                    .named(ctx, "dilation", .int(ctx, .i64, sl.dilation)),
                }));
                const at = for (slid.items, 0..) |a, i| {
                    if (std.mem.eql(u8, a.name, sl.axis.name)) break i;
                } else std.debug.panic("tcl: {s}: slide axis {s} is not read", .{ b.name, sl.axis.name });
                slid.items[at] = sl.window_axis;
                slid.insert(b.arena.allocator(), at + 1, sl.frame_axis) catch @panic("OOM");
            }
            fields[nfields] = .named(ctx, "slide", .dict(ctx, slides));
            nfields += 1;
            axes = slid.items;
        }
        if (opts.broadcast_to) |to| {
            fields[nfields] = .named(ctx, "broadcast_to", b.axesAttr(to));
            axes = b.dupeAxes(to);
            nfields += 1;
        }
        const options = if (nfields == 0) null else tcl.readOptions(ctx, fields[0..nfields]) catch
            std.debug.panic("tcl: invalid read options for {s}", .{b.name});
        const op = tcl.read(ctx, operands[0..n], b.logicalType(dtype, axes), options, b.loc());
        return self.emit(op, dtype, axes, true);
    }

    /// `dpe.exec(lhs, rhs -> [to])`: shared axes missing from `to` are
    /// contracted. Accumulates in f32 (float inputs) or i32.
    pub fn contract(self: *TensorOperation, lhs: anytype, rhs: anytype, to: []const Axis) Value {
        if (self.has_dpe or self.has_ve) std.debug.panic("tcl: one contraction per tensor operation, before VE instructions", .{});
        const l = self.lift(lhs, .f32);
        const r = self.lift(rhs, .f32);
        const dtype: DType = if (l.dtype.isFloat()) .f32 else .i32;
        self.has_dpe = true;
        const op = tcl.dpe(self.b.ctx, l.inner, r.inner, self.b.logicalType(dtype, to), self.b.loc());
        return self.emit(op, dtype, self.b.dupeAxes(to), false);
    }

    /// `ve.exec(...)` of a unary or binary instruction. Operands are
    /// `Value`s, `Tensor`s (read on use) or scalar literals.
    pub fn ve(self: *TensorOperation, comptime opcode: VeOp, operands: anytype) Value {
        if (comptime operands.len != opcode.arity()) @compileError("tcl: wrong operand count for " ++ @tagName(opcode));
        return self.veNamed(@tagName(opcode), opcode.result(), operands);
    }

    /// `ve.exec(x to_f<int_width>)`: Q-format fixed point to f32, Python's
    /// `operation.to_fp`.
    pub fn toFp(self: *TensorOperation, x: anytype, int_width: u5) Value {
        var buf: [8]u8 = undefined;
        return self.veNamed(std.fmt.bufPrint(&buf, "to_f{d}", .{int_width}) catch unreachable, .f32, .{x});
    }

    /// `ve.exec(x to_i<int_width>)`: f32 to Q-format fixed point, Python's
    /// `operation.to_fxp`.
    pub fn toFxp(self: *TensorOperation, x: anytype, int_width: u5) Value {
        var buf: [8]u8 = undefined;
        return self.veNamed(std.fmt.bufPrint(&buf, "to_i{d}", .{int_width}) catch unreachable, .i32, .{x});
    }

    fn veNamed(self: *TensorOperation, opcode: []const u8, scalar: DType, operands: anytype) Value {
        var values: [2]*const mlir.Value = undefined;
        var axes: []const Axis = &.{};
        inline for (operands, 0..) |o, i| {
            const v = self.lift(o, scalar);
            values[i] = v.inner;
            axes = self.b.unionAxes(axes, v.axes);
        }
        self.has_ve = true;
        const ctx = self.b.ctx;
        const code = tcl.veOpcode(ctx, opcode) catch std.debug.panic("tcl: unknown VE instruction {s}", .{opcode});
        const op = tcl.ve(ctx, values[0..operands.len], code, self.b.logicalType(scalar, axes), self.b.loc());
        return self.emit(op, scalar, axes, false);
    }

    /// `ve.exec(<mode> t[axes] x)`: removes `axes`.
    pub fn reduce(self: *TensorOperation, x: anytype, axes: []const Axis, mode: ReduceMode) Value {
        const v = self.lift(x, .f32);
        var kept: std.ArrayList(Axis) = .empty;
        for (v.axes) |a| {
            if (!containsAxis(axes, a)) kept.append(self.b.arena.allocator(), a) catch @panic("OOM");
        }
        self.has_ve = true;
        self.has_reduce = true;
        const op = tcl.ve_reduce(self.b.ctx, v.inner, mode, self.b.axisAttrs(axes), self.b.logicalType(v.dtype, kept.items), self.b.loc());
        return self.emit(op, v.dtype, kept.items, false);
    }

    /// `ve.exec(if cond <pred> threshold { yes } else { no })`, like
    /// `operation.where(cond <pred> threshold, yes, no)`.
    pub fn where(self: *TensorOperation, cond: anytype, pred: Predicate, threshold: anytype, yes: anytype, no: anytype) Value {
        const ctx = self.b.ctx;
        const c = self.lift(cond, .f32);
        const float = isFloatOperand(yes) or isFloatOperand(no);
        const scalar: DType = if (float) .f32 else .i32;
        const t = self.lift(yes, scalar);
        const f = self.lift(no, scalar);
        const threshold_attr: *const mlir.Attribute = if (c.dtype.isFloat())
            .float(ctx, .f32, threshold)
        else
            .int(ctx, .i32, threshold);
        const axes = self.b.unionAxes(self.b.unionAxes(c.axes, t.axes), f.axes);
        self.has_ve = true;
        self.has_select = true;
        const op = tcl.ve_select(ctx, c.inner, pred, threshold_attr, t.inner, f.inner, self.b.logicalType(scalar, axes), self.b.loc());
        return self.emit(op, scalar, axes, false);
    }

    /// `write.to(...)`: closes the tensor operation and returns its result.
    pub fn commit(self: *TensorOperation, v: Value, opts: CommitOptions) FinishError!Tensor {
        const b = self.b;
        if (self.last != v.inner) std.debug.panic("tcl: {s}: commit must take the last instruction", .{b.name});
        const axes = if (opts.axes) |a| b.dupeAxes(a) else v.axes;
        if (!sameAxisSet(axes, v.axes)) std.debug.panic("tcl: {s}: commit can only permute axes", .{b.name});
        const dtype = opts.dtype orelse v.dtype;
        _ = tcl.write(b.ctx, v.inner, b.loc()).appendTo(self.body);

        const inputs = try b.arena.allocator().alloc(*const mlir.Value, self.inputs.items.len);
        for (inputs, self.inputs.items) |*dst, t| dst.* = t.inner;
        const context = if (self.context) |c| b.contextAttr(c) else null;
        const kernel = tcl.kernel(b.ctx, inputs, b.logicalType(dtype, axes), self.tactic orelse self.guessTactic(axes), context, self.body, b.loc());
        try b.append(kernel);
        return .{ .inner = kernel.result(0), .dtype = dtype, .axes = axes };
    }

    /// Python's `BuilderHistory.guess_tactic_kind`.
    fn guessTactic(self: *const TensorOperation, output: []const Axis) Tactic {
        if (self.has_dpe) return .einsum_by_dpe;
        if (!self.has_ve) return .tensor_operation;
        if (self.has_reduce) return .reduce_by_ve;
        const in = self.inputs.items;
        if (in.len > 0 and in[0].axes.len < output.len) return .einsum_by_ve;
        if (!self.has_select and in.len == 2 and sameAxisSet(in[0].axes, in[1].axes)) return .interleaving;
        return .elementwise;
    }

    const Operand = struct { inner: *const mlir.Value, dtype: DType, axes: []const Axis };

    fn lift(self: *TensorOperation, x: anytype, scalar: DType) Operand {
        const T = @TypeOf(x);
        if (T == Value) return .{ .inner = x.inner, .dtype = x.dtype, .axes = x.axes };
        if (T == Tensor) {
            const r = self.plainRead(x);
            return .{ .inner = r.inner, .dtype = r.dtype, .axes = r.axes };
        }
        const ctx = self.b.ctx;
        const op = switch (@typeInfo(T)) {
            .comptime_int, .int => if (scalar == .f32)
                arith.constant_float(ctx, @floatFromInt(x), .f32, self.b.loc())
            else
                arith.constant_int(ctx, x, .int(ctx, .i32), self.b.loc()),
            .comptime_float, .float => blk: {
                if (scalar != .f32) std.debug.panic("tcl: float literal for an integer VE instruction", .{});
                break :blk arith.constant_float(ctx, x, .f32, self.b.loc());
            },
            else => @compileError("tcl: expected a Value, a Tensor or a scalar, got " ++ @typeName(T)),
        };
        _ = op.appendTo(self.body);
        return .{ .inner = op.result(0), .dtype = scalar, .axes = &.{} };
    }

    /// Registers `t` as a kernel operand.
    fn input(self: *TensorOperation, t: Tensor) *const mlir.Value {
        if (t.inner.type_().isA(tcl.LogicalType) == null)
            std.debug.panic("tcl: {s}: tensor operations read logical tensors, not DRAM ones", .{self.b.name});
        for (self.inputs.items) |known| {
            if (known.inner == t.inner) return t.inner;
        }
        self.inputs.append(self.b.arena.allocator(), t) catch @panic("OOM");
        return t.inner;
    }

    fn plainRead(self: *TensorOperation, t: Tensor) Value {
        for (self.plain_reads.items) |r| if (r.tensor == t.inner) return r.value;
        const r = self.fetch(t, .{});
        self.plain_reads.append(self.b.arena.allocator(), .{ .tensor = t.inner, .value = r }) catch @panic("OOM");
        return r;
    }

    /// Reads go before every other instruction, as TCL lists them.
    fn emit(self: *TensorOperation, op: *mlir.Operation, dtype: DType, axes: []const Axis, is_read: bool) Value {
        if (is_read and self.first_compute != null) {
            self.body.insertOwnedOperationBefore(self.first_compute.?, op);
        } else {
            _ = op.appendTo(self.body);
            if (!is_read and self.first_compute == null) self.first_compute = op;
            self.last = op.result(0);
        }
        return .{ .op = self, .inner = op.result(0), .dtype = dtype, .axes = axes };
    }
};

fn isFloatOperand(x: anytype) bool {
    return switch (@typeInfo(@TypeOf(x))) {
        .comptime_float, .float => true,
        .comptime_int, .int => false,
        else => x.dtype.isFloat(),
    };
}

fn containsAxis(axes: []const Axis, a: Axis) bool {
    for (axes) |x| if (std.mem.eql(u8, x.name, a.name)) return true;
    return false;
}

fn sameAxisSet(a: []const Axis, b: []const Axis) bool {
    if (a.len != b.len) return false;
    for (a) |x| if (!containsAxis(b, x)) return false;
    return true;
}

/// Builds one TCL function, `def name(axes..., args...) -> results:`, as a
/// `func.func` of the native `tcl` dialect.
pub const Builder = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    ctx: *mlir.Context,
    module: *mlir.Module,
    name: []const u8,
    axes: std.ArrayList(Declared) = .empty,
    compiler_config: ?*const mlir.Attribute = null,
    auto_config: ?*const mlir.Attribute = null,
    func_op: ?*mlir.Operation = null,
    entry: ?*mlir.Block = null,
    args: []const Tensor = &.{},
    results: ?[]const Tensor = null,

    pub fn open(allocator: std.mem.Allocator, ctx: *mlir.Context, name: []const u8) !Builder {
        _ = tcl.symbol(ctx, name) catch std.debug.panic("tcl: `{s}` is not a TCL function name", .{name});
        return .{
            .allocator = allocator,
            .arena = .init(allocator),
            .ctx = ctx,
            .module = .init(.unknown(ctx)),
            .name = name,
        };
    }

    pub fn deinit(self: *Builder) void {
        self.module.deinit();
        self.arena.deinit();
    }

    pub fn loc(self: *const Builder) *const mlir.Location {
        return .unknown(self.ctx);
    }

    const Declared = struct { axis: Axis, def: ?Expr };

    /// `A: Axis = size`, where `size` is an integer or an `Expr` over other
    /// axes (`b.axis("C", b.expr(.mul, A, B))`). Axis names start with an
    /// uppercase letter.
    pub fn axis(self: *Builder, name: []const u8, size: anytype) Axis {
        const def: ?Expr, const n: i64 = if (comptime @TypeOf(size) == Expr)
            .{ size, size.value() orelse std.debug.panic("tcl: axis {s} needs an integer size", .{name}) }
        else
            .{ null, size };
        if (name.len == 0 or !std.ascii.isUpper(name[0]) or n <= 0)
            std.debug.panic("tcl: invalid axis {s} = {d}", .{ name, n });
        _ = tcl.symbol(self.ctx, name) catch std.debug.panic("tcl: invalid axis name {s}", .{name});
        for (self.axes.items) |d| if (std.mem.eql(u8, d.axis.name, name)) {
            if (d.axis.size != n) std.debug.panic("tcl: axis {s} redeclared as {d} (was {d})", .{ name, n, d.axis.size });
            return d.axis;
        };
        const a: Axis = .{ .name = self.arena.allocator().dupe(u8, name) catch @panic("OOM"), .size = n };
        self.axes.append(self.arena.allocator(), .{ .axis = a, .def = def }) catch @panic("OOM");
        return a;
    }

    /// An `Expr` node: `lhs` and `rhs` are axes, integers or expressions.
    pub fn expr(self: *Builder, kind: Expr.Kind, lhs: anytype, rhs: anytype) Expr {
        const node = self.arena.allocator().create(Expr.Node) catch @panic("OOM");
        node.* = .{ .kind = kind, .lhs = toExpr(lhs), .rhs = toExpr(rhs) };
        return .{ .node = node };
    }

    fn toExpr(x: anytype) Expr {
        return switch (@TypeOf(x)) {
            Expr => x,
            Axis => .{ .axis = x },
            else => .{ .int = x },
        };
    }

    /// `@compiler_config({...})`, Python's `@tcl.kernel(config=...)`: an
    /// anonymous struct written out as JSON, `.{ .lowering_mode = "Heuristic" }`.
    pub fn compilerConfig(self: *Builder, config: anytype) void {
        self.compiler_config = tcl.config(self.ctx, self.jsonFields(config)) catch
            std.debug.panic("tcl: {s}: invalid compiler config", .{self.name});
    }

    /// `@auto_config({...})`, Python's `@tcl.kernel(auto_config=...)`.
    pub fn autoConfig(self: *Builder, config: anytype) void {
        self.auto_config = tcl.config(self.ctx, self.jsonFields(config)) catch
            std.debug.panic("tcl: {s}: invalid auto config", .{self.name});
    }

    /// One field per argument: `.{ .x = .{ .dtype = .bf16, .axes = &.{ M, K } } }`.
    /// With `.dram = .{...}` the argument is DRAM-mapped and the field is
    /// its `as_logical` view.
    pub fn declareArgs(self: *Builder, spec: anytype) FinishError!ArgsOf(@TypeOf(spec)) {
        std.debug.assert(self.entry == null);
        const fields = @typeInfo(@TypeOf(spec)).@"struct".fields;
        const scratch = self.arena.allocator();
        var types: [fields.len]*const mlir.Type = undefined;
        var locs: [fields.len]*const mlir.Location = undefined;
        const args = try scratch.alloc(Tensor, fields.len);
        inline for (fields, 0..) |f, i| {
            const s = @field(spec, f.name);
            const axes = self.dupeAxes(s.axes);
            types[i] = if (@hasField(@TypeOf(s), "dram")) self.mappedType(s.dtype, axes, toDram(s.dram)) else self.logicalType(s.dtype, axes);
            locs[i] = self.loc();
            args[i] = .{ .inner = undefined, .dtype = s.dtype, .axes = axes };
        }
        const entry = mlir.Block.init(&types, &locs);
        // In place from the start: the kernel verifier walks to its inputs'
        // regions. `finish` sets the function type.
        const func_op = func.func(self.ctx, .{ .name = self.name, .block = entry, .results = &.{}, .location = self.loc(), .visibility = null, .verify = false });
        _ = func_op.appendTo(self.module.body());
        self.func_op = func_op;
        self.entry = entry;
        var named: ArgsOf(@TypeOf(spec)) = undefined;
        inline for (fields, 0..) |f, i| {
            args[i].inner = entry.argument(i);
            if (@hasField(@TypeOf(@field(spec, f.name)), "dram"))
                args[i].inner = try self.graphOp(.as_logical, &.{entry.argument(i)}, self.logicalType(args[i].dtype, args[i].axes), &.{}, null);
            @field(named, f.name) = args[i];
        }
        self.args = args;
        return named;
    }

    fn toDram(d: anytype) Dram {
        const D = @TypeOf(d);
        if (D == Dram) return d;
        return .{
            .chip = if (@hasField(D, "chip")) d.chip else &.{.broadcast},
            .inner = if (@hasField(D, "inner")) d.inner else null,
            .original = if (@hasField(D, "original")) d.original else &.{},
        };
    }

    fn ArgsOf(comptime Spec: type) type {
        const in = @typeInfo(Spec).@"struct".fields;
        comptime var names: [in.len][]const u8 = undefined;
        inline for (in, 0..) |f, i| names[i] = f.name;
        return @Struct(.auto, null, &names, &@splat(Tensor), &@splat(.{}));
    }

    /// Opens a `tk.<Tactic>(...)` tensor operation; `commit` closes it.
    pub fn tensorOperation(self: *Builder, opts: TensorOperation.Options) *TensorOperation {
        _ = self.entryBlock();
        const op = self.arena.allocator().create(TensorOperation) catch @panic("OOM");
        op.* = .{ .b = self, .tactic = opts.tactic, .context = opts.context, .body = mlir.Block.init(&.{}, &.{}) };
        return op;
    }

    pub const GraphOptions = struct { context: ?Context = null };

    /// `reshape(t)` to `to`, same element count.
    pub fn reshape(self: *Builder, t: Tensor, to: []const Axis, opts: GraphOptions) FinishError!Tensor {
        if (elements(t.axes) != elements(to)) std.debug.panic("tcl: reshape changes the element count", .{});
        return self.graph(.reshape, &.{t}, t.dtype, to, &.{}, opts.context);
    }

    /// `transmute(t)`: the same bits as `dtype` over `to`.
    pub fn transmute(self: *Builder, t: Tensor, dtype: DType, to: []const Axis, opts: GraphOptions) FinishError!Tensor {
        return self.graph(.transmute, &.{t}, dtype, to, &.{}, opts.context);
    }

    /// `concat([inputs])` along the one axis of `to` the inputs lack.
    pub fn concat(self: *Builder, inputs: []const Tensor, to: []const Axis, opts: GraphOptions) FinishError!Tensor {
        return self.graph(.concat, inputs, inputs[0].dtype, to, &.{}, opts.context);
    }

    /// `slice(t, axis, offset)`: the window of `to` starting at `offset` along `axis`.
    pub fn slice(self: *Builder, t: Tensor, along: Axis, offset: i64, to: []const Axis, opts: GraphOptions) FinishError!Tensor {
        return self.graph(.slice, &.{t}, t.dtype, to, &.{
            .named(self.ctx, "axis", self.axisAttr(along)),
            .named(self.ctx, "offset", .int(self.ctx, .i64, offset)),
        }, opts.context);
    }

    pub const GatherOptions = struct {
        batch_axis: ?Axis = null,
        valid_length: ?Tensor = null,
        context: ?Context = null,
    };

    /// `gather(table, indices, axis)`: rows of `table` along `axis`.
    pub fn gather(self: *Builder, table: Tensor, indices: Tensor, along: Axis, to: []const Axis, opts: GatherOptions) FinishError!Tensor {
        var options: [2]mlir.NamedAttribute = .{ .named(self.ctx, "axis", self.axisAttr(along)), undefined };
        if (opts.batch_axis) |a| options[1] = .named(self.ctx, "batch_axis", self.axisAttr(a));
        var inputs: [3]Tensor = .{ table, indices, opts.valid_length orelse undefined };
        const n_inputs = @as(usize, 2) + @intFromBool(opts.valid_length != null);
        return self.graph(.gather, inputs[0..n_inputs], table.dtype, to, options[0 .. @as(usize, 1) + @intFromBool(opts.batch_axis != null)], opts.context);
    }

    pub const ScatterOptions = struct {
        /// The tensor scattered into; zeros otherwise.
        init: ?Tensor = null,
        batch_axis: ?Axis = null,
        context: ?Context = null,
    };

    /// `scatter(updates, indices, axis, init=...)`: `updates` written at
    /// `indices` along `axis` of the result.
    pub fn scatter(self: *Builder, updates: Tensor, indices: Tensor, along: Axis, to: []const Axis, opts: ScatterOptions) FinishError!Tensor {
        var options: [2]mlir.NamedAttribute = .{ .named(self.ctx, "axis", self.axisAttr(along)), undefined };
        if (opts.batch_axis) |a| options[1] = .named(self.ctx, "batch_axis", self.axisAttr(a));
        var inputs: [3]Tensor = .{ updates, indices, opts.init orelse undefined };
        const n_inputs = @as(usize, 2) + @intFromBool(opts.init != null);
        return self.graph(.scatter, inputs[0..n_inputs], updates.dtype, to, options[0 .. @as(usize, 1) + @intFromBool(opts.batch_axis != null)], opts.context);
    }

    pub const ArangeOptions = struct {
        start: ?i64 = null,
        step: ?i64 = null,
        context: ?Context = null,
    };

    /// `arange(start=, end, step=)` over the one axis of `to`.
    pub fn arange(self: *Builder, end: i64, dtype: DType, to: []const Axis, opts: ArangeOptions) FinishError!Tensor {
        var options: [3]mlir.NamedAttribute = undefined;
        var n: usize = 0;
        if (opts.start) |v| {
            options[n] = .named(self.ctx, "start", .int(self.ctx, .i64, v));
            n += 1;
        }
        options[n] = .named(self.ctx, "end", .int(self.ctx, .i64, end));
        n += 1;
        if (opts.step) |v| {
            options[n] = .named(self.ctx, "step", .int(self.ctx, .i64, v));
            n += 1;
        }
        return self.graph(.arange, &.{}, dtype, to, options[0..n], opts.context);
    }

    pub const VectorOptions = struct {
        /// Repeats `values` cyclically along this axis.
        repeat_to: ?Axis = null,
        context: ?Context = null,
    };

    /// `vector([values], repeat_to=)`: a constant tensor.
    pub fn vector(self: *Builder, values: []const f64, dtype: DType, to: []const Axis, opts: VectorOptions) FinishError!Tensor {
        const attrs = try self.arena.allocator().alloc(*const mlir.Attribute, values.len);
        for (attrs, values) |*dst, v| dst.* = if (dtype.isFloat()) .float(self.ctx, .f64, v) else .int(self.ctx, .i64, @as(i64, @intFromFloat(v)));
        var options: [2]mlir.NamedAttribute = .{ .named(self.ctx, "values", .array(self.ctx, attrs)), undefined };
        if (opts.repeat_to) |a| options[1] = .named(self.ctx, "repeat_to", self.axisAttr(a));
        return self.graph(.vector, &.{}, dtype, to, options[0 .. @as(usize, 1) + @intFromBool(opts.repeat_to != null)], opts.context);
    }

    /// `all_gather(t, axis)` across chips.
    pub fn allGather(self: *Builder, t: Tensor, along: Axis, opts: GraphOptions) FinishError!Tensor {
        return self.graph(.all_gather, &.{t}, t.dtype, t.axes, &.{
            .named(self.ctx, "axis", self.axisAttr(along)),
        }, opts.context);
    }

    /// `as_dram(t)`: `t` in the DRAM mapping `dram`, to return it as such.
    pub fn asDram(self: *Builder, t: Tensor, dram: Dram) FinishError!Tensor {
        const out = try self.graphOp(.as_dram, &.{t.inner}, self.mappedType(t.dtype, t.axes, dram), &.{}, null);
        return .{ .inner = out, .dtype = t.dtype, .axes = t.axes };
    }

    /// Any `tcl.graph.*` operation; `options` are its TCL keyword fields.
    pub fn graph(self: *Builder, comptime op: tcl.GraphOp, inputs: []const Tensor, dtype: DType, to: []const Axis, options: []const mlir.NamedAttribute, context: ?Context) FinishError!Tensor {
        const values = try self.arena.allocator().alloc(*const mlir.Value, inputs.len);
        for (values, inputs) |*v, t| v.* = t.inner;
        const axes = self.dupeAxes(to);
        const out = try self.graphOp(op, values, self.logicalType(dtype, axes), options, context);
        return .{ .inner = out, .dtype = dtype, .axes = axes };
    }

    fn graphOp(self: *Builder, comptime op: tcl.GraphOp, inputs: []const *const mlir.Value, output: *const mlir.Type, options: []const mlir.NamedAttribute, context: ?Context) FinishError!*const mlir.Value {
        const g = tcl.graph(self.ctx, op, inputs, output, options, if (context) |c| self.contextAttr(c) else null, self.loc());
        try self.append(g);
        return g.result(0);
    }

    /// `return results...`.
    pub fn ret(self: *Builder, results: []const Tensor) void {
        self.results = self.arena.allocator().dupe(Tensor, results) catch @panic("OOM");
    }

    /// The verified module: one `func.func` carrying its axes in `tcl.axes`.
    pub fn finish(self: *Builder) FinishError![:0]const u8 {
        const ctx = self.ctx;
        const f = self.func_op orelse return error.InvalidMlir;
        const results = self.results orelse return error.InvalidMlir;
        const scratch = self.arena.allocator();
        const values = try scratch.alloc(*const mlir.Value, results.len);
        const result_types = try scratch.alloc(*const mlir.Type, results.len);
        for (values, result_types, results) |*v, *t, r| {
            v.* = r.inner;
            t.* = r.inner.type_();
        }
        _ = func.returns(ctx, values, self.loc()).appendTo(self.entry.?);

        const entry = self.entry.?;
        const arg_types = try scratch.alloc(*const mlir.Type, entry.numArguments());
        for (arg_types, 0..) |*t, i| t.* = entry.argument(i).type_();
        f.setAttributeByName("function_type", .typeAttr(.function(ctx, arg_types, result_types)));

        const axes = try scratch.alloc(mlir.NamedAttribute, self.axes.items.len);
        for (axes, self.axes.items) |*dst, d| {
            dst.* = .named(ctx, d.axis.name, if (d.def) |e| self.exprAttr(e) else .int(ctx, .i64, d.axis.size));
        }
        f.setAttributeByName("tcl.axes", .dict(ctx, axes));
        if (self.compiler_config) |c| f.setAttributeByName("tcl.compiler_config", c);
        if (self.auto_config) |c| f.setAttributeByName("tcl.auto_config", c);
        if (!self.module.operation().verify()) return error.InvalidMlir;

        var printed: std.Io.Writer.Allocating = .init(self.allocator);
        defer printed.deinit();
        try printed.writer.print("{f}", .{self.module.operation()});
        return try self.allocator.dupeZ(u8, printed.written());
    }

    fn append(self: *Builder, op: *mlir.Operation) FinishError!void {
        _ = op.appendTo(self.entryBlock());
        if (!op.verify()) {
            std.log.err("tcl: {s}: invalid operation:\n{f}", .{ self.name, op.fmt(.{ .print_generic_op_form = true }) });
            return error.InvalidMlir;
        }
    }

    fn entryBlock(self: *const Builder) *mlir.Block {
        return self.entry orelse std.debug.panic("tcl: {s}: declareArgs first", .{self.name});
    }

    pub fn axisAttr(self: *const Builder, a: Axis) *const mlir.Attribute {
        return tcl.symbol(self.ctx, a.name) catch unreachable;
    }

    fn axisAttrs(self: *Builder, axes: []const Axis) []const *const mlir.Attribute {
        const attrs = self.arena.allocator().alloc(*const mlir.Attribute, axes.len) catch @panic("OOM");
        for (attrs, axes) |*dst, a| dst.* = self.axisAttr(a);
        return attrs;
    }

    fn axesAttr(self: *Builder, axes: []const Axis) *const mlir.Attribute {
        return .array(self.ctx, self.axisAttrs(axes));
    }

    pub fn exprAttr(self: *Builder, e: Expr) *const mlir.Attribute {
        return switch (e) {
            .int => |n| .int(self.ctx, .i64, n),
            .axis => |a| self.axisAttr(a),
            .broadcast => tcl.expr(self.ctx, "broadcast", &.{}) catch unreachable,
            .node => |n| tcl.expr(self.ctx, @tagName(n.kind), &.{ self.exprAttr(n.lhs), self.exprAttr(n.rhs) }) catch
                std.debug.panic("tcl: {s}: invalid {s} expression", .{ self.name, @tagName(n.kind) }),
        };
    }

    fn exprAttrs(self: *Builder, exprs: []const Expr) []const *const mlir.Attribute {
        const attrs = self.arena.allocator().alloc(*const mlir.Attribute, exprs.len) catch @panic("OOM");
        for (attrs, exprs) |*dst, e| dst.* = self.exprAttr(e);
        return attrs;
    }

    pub fn contextAttr(self: *Builder, c: Context) *const mlir.Attribute {
        const ctx = self.ctx;
        var fields: [2]mlir.NamedAttribute = undefined;
        var n: usize = 0;
        if (c.layout) |l| {
            var layout: [3]mlir.NamedAttribute = undefined;
            var m: usize = 0;
            layout[m] = .named(ctx, "Chip", self.exprAttr(l.chip));
            m += 1;
            if (l.cluster) |cl| {
                layout[m] = .named(ctx, "Cluster", self.exprAttr(cl));
                m += 1;
            }
            if (l.split.len > 0) {
                layout[m] = .named(ctx, "Split", self.axesAttr(l.split));
                m += 1;
            }
            fields[n] = .named(ctx, "operator", .dict(ctx, layout[0..m]));
            n += 1;
        }
        if (c.heuristic_hint.len > 0) {
            const hints = self.arena.allocator().alloc(mlir.NamedAttribute, c.heuristic_hint.len) catch @panic("OOM");
            for (hints, c.heuristic_hint) |*dst, h| dst.* = .named(ctx, h.name, switch (h.value) {
                .int => |v| .int(ctx, .i64, v),
                .float => |v| .float(ctx, .f64, v),
                .boolean => |v| .boolean(ctx, v),
                .axis => |a| self.axisAttr(a),
            });
            fields[n] = .named(ctx, "heuristic_hint", .dict(ctx, hints));
            n += 1;
        }
        return tcl.context(ctx, fields[0..n]) catch std.debug.panic("tcl: {s}: invalid context", .{self.name});
    }

    fn jsonFields(self: *Builder, config: anytype) []const mlir.NamedAttribute {
        const fields = @typeInfo(@TypeOf(config)).@"struct".fields;
        const out = self.arena.allocator().alloc(mlir.NamedAttribute, fields.len) catch @panic("OOM");
        inline for (fields, 0..) |f, i| out[i] = .named(self.ctx, f.name, self.json(@field(config, f.name)));
        return out;
    }

    fn json(self: *Builder, v: anytype) *const mlir.Attribute {
        const T = @TypeOf(v);
        return switch (@typeInfo(T)) {
            .bool => .boolean(self.ctx, v),
            .comptime_int, .int => .int(self.ctx, .i64, v),
            .comptime_float, .float => .float(self.ctx, .f64, v),
            .enum_literal, .@"enum" => .string(self.ctx, @tagName(v)),
            .@"struct" => |info| if (info.is_tuple) blk: {
                var items: [info.fields.len]*const mlir.Attribute = undefined;
                inline for (info.fields, 0..) |f, i| items[i] = self.json(@field(v, f.name));
                break :blk .array(self.ctx, &items);
            } else .dict(self.ctx, self.jsonFields(v)),
            .pointer => if (std.meta.Elem(T) == u8) .string(self.ctx, v) else blk: {
                const items = self.arena.allocator().alloc(*const mlir.Attribute, v.len) catch @panic("OOM");
                for (items, v) |*dst, x| dst.* = self.json(x);
                break :blk .array(self.ctx, items);
            },
            else => @compileError("tcl: config values are JSON: bool, numbers, strings, structs, tuples"),
        };
    }

    fn mappedType(self: *Builder, dtype: DType, axes: []const Axis, dram: Dram) *const mlir.Type {
        for (axes) |a| _ = self.axis(a.name, a.size);
        const inner = if (dram.inner) |e| self.exprAttrs(e) else self.axisAttrs(axes);
        const mapping = tcl.dramMapping(self.ctx, self.exprAttrs(dram.chip), inner, self.axisAttrs(dram.original)) catch
            std.debug.panic("tcl: {s}: invalid DRAM mapping", .{self.name});
        return tcl.mappedType(self.ctx, dtype.toMlir(self.ctx), mapping) catch unreachable;
    }

    pub fn logicalType(self: *Builder, dtype: DType, axes: []const Axis) *const mlir.Type {
        for (axes) |a| _ = self.axis(a.name, a.size);
        return tcl.logicalType(self.ctx, dtype.toMlir(self.ctx), self.axisAttrs(axes)) catch
            std.debug.panic("tcl: {s}: invalid tensor type (repeated axis?)", .{self.name});
    }

    fn dupeAxes(self: *Builder, axes: []const Axis) []const Axis {
        return self.arena.allocator().dupe(Axis, axes) catch @panic("OOM");
    }

    fn unionAxes(self: *Builder, a: []const Axis, b: []const Axis) []const Axis {
        var out: std.ArrayList(Axis) = .empty;
        const long, const short = if (a.len >= b.len) .{ a, b } else .{ b, a };
        out.appendSlice(self.arena.allocator(), long) catch @panic("OOM");
        for (short) |x| if (!containsAxis(out.items, x)) out.append(self.arena.allocator(), x) catch @panic("OOM");
        return out.items;
    }
};

fn elements(axes: []const Axis) i64 {
    var n: i64 = 1;
    for (axes) |a| n *= a.size;
    return n;
}

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    tcl.registerDialects(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

fn expectContains(haystack: []const u8, needle: []const u8) !void {
    if (std.mem.indexOf(u8, haystack, needle) == null) {
        std.debug.print("missing {s} in:\n{s}\n", .{ needle, haystack });
        return error.TestExpectedContains;
    }
}

test "softmax matches furiosa.tcl.examples.softmax" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = try Builder.open(std.testing.allocator, ctx, "softmax_kernel");
    defer b.deinit();

    const B = b.axis("B", 2);
    const S = b.axis("S", 16);
    const D = b.axis("D", 1024);
    const t = try b.declareArgs(.{ .input = .{ .dtype = .bf16, .axes = &.{ B, S, D } } });

    const op = b.tensorOperation(.{});
    const x = op.fetch(t.input, .{ .typecast_to = .f32 });
    const centered = x.subf(x.reduce(&.{D}, .maxf));
    const exped = centered.exp();
    const out = try op.commit(exped.divf(exped.reduce(&.{D}, .addf)), .{ .dtype = .bf16 });
    b.ret(&.{out});

    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    try expectContains(ir, "tcl.axes = {B = 2 : i64, D = 1024 : i64, S = 16 : i64}");
    try expectContains(ir, "options = #tcl.read_options<{typecast_to = f32}>");
    try expectContains(ir, "mode = #tcl.reduce_mode<\"Maxf\">");
    try expectContains(ir, "opcode = #tcl.ve_opcode<\"divf\">");
    try expectContains(ir, "tactic = #tcl.tactic<\"ReduceByVe\">");
    try expectContains(ir, "-> !tcl.logical<bf16, [#tcl.symbol<\"B\">, #tcl.symbol<\"S\">, #tcl.symbol<\"D\">]>");
    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "batched matmul, scalar operands and graph reshape" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = try Builder.open(std.testing.allocator, ctx, "bmm_scaled");
    defer b.deinit();

    const B = b.axis("B", 2);
    const M = b.axis("M", 128);
    const K = b.axis("K", 64);
    const N = b.axis("N", 256);
    const t = try b.declareArgs(.{
        .lhs = .{ .dtype = .bf16, .axes = &.{ B, M, K } },
        .rhs = .{ .dtype = .bf16, .axes = &.{ B, K, N } },
    });

    const mm = b.tensorOperation(.{});
    const acc = mm.contract(t.lhs, t.rhs, &.{ B, M, N });
    const prod = try mm.commit(acc.mulf(0.5), .{ .dtype = .bf16 });
    const flat = try b.reshape(prod, &.{ B, b.axis("MN", 128 * 256) }, .{});
    b.ret(&.{ prod, flat });

    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    try expectContains(ir, "tactic = #tcl.tactic<\"EinsumByDpe\">");
    try expectContains(ir, "\"tcl.dpe\"");
    try expectContains(ir, "arith.constant 5.000000e-01 : f32");
    try expectContains(ir, "\"tcl.graph.reshape\"");
}

test "conv1d matches furiosa.kernels.convolution" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = try Builder.open(std.testing.allocator, ctx, "conv");
    defer b.deinit();

    const N = b.axis("N", 2);
    const C = b.axis("C", 64);
    const L = b.axis("L", 128);
    const K = b.axis("K", 128);
    const Lw = b.axis("Lw", 3);
    const Lf = b.axis("Lf", 128);
    const t = try b.declareArgs(.{
        .inp = .{ .dtype = .bf16, .axes = &.{ N, C, L } },
        .flt = .{ .dtype = .bf16, .axes = &.{ K, C, Lw } },
        .bias = .{ .dtype = .bf16, .axes = &.{K} },
    });

    const op = b.tensorOperation(.{});
    const windows = op.fetch(t.inp, .{
        .pad = &.{.{ .axis = L, .left = 1, .right = 1 }},
        .slide = &.{.{ .axis = L, .frame_axis = Lf, .window_axis = Lw }},
    });
    try std.testing.expectEqualStrings("Lw", windows.axes[2].name);
    const out = try op.commit(op.contract(windows, t.flt, &.{ N, K, Lf }).addf(t.bias), .{ .dtype = .bf16 });
    b.ret(&.{out});

    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    try expectContains(ir, "slide = {L = {dilation = 1 : i64, frame_axis = #tcl.symbol<\"Lf\">, stride = 1 : i64, undilated_window = 3 : i64, window_axis = #tcl.symbol<\"Lw\">}}");
    try expectContains(ir, "tactic = #tcl.tactic<\"EinsumByDpe\">");
}

test "composite axes, DRAM, fixed point, context, config and graph ops" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = try Builder.open(std.testing.allocator, ctx, "features");
    defer b.deinit();

    const M = b.axis("M", 64);
    const N = b.axis("N", 128);
    const G = b.axis("G", 8);
    const MN = b.axis("MN", b.expr(.mul, M, N));
    try std.testing.expectEqual(64 * 128, MN.size);
    b.compilerConfig(.{ .lowering_mode = "Heuristic", .max_num_partitioning_axes = 1 });
    const t = try b.declareArgs(.{
        .x = .{ .dtype = .f32, .axes = &.{ M, N }, .dram = .{} },
        .idx = .{ .dtype = .i32, .axes = &.{G} },
    });

    const op = b.tensorOperation(.{ .context = .{
        .layout = .{ .chip = .broadcast, .split = &.{M} },
        .heuristic_hint = &.{.{ .name = "max_num_partitioning_axes", .value = .{ .int = 2 } }},
    } });
    const sum = try op.commit(op.fetch(t.x, .{}).reduce(&.{N}, .addf).toFxp(15).toFp(15), .{});

    const rows = try b.gather(t.x, t.idx, M, &.{ G, N }, .{});
    const back = try b.scatter(rows, t.idx, M, &.{ M, N }, .{ .init = t.x });
    const iota = try b.arange(N.size, .i32, &.{N}, .{});
    const pattern = try b.vector(&.{ 1, 2 }, .f32, &.{N}, .{ .repeat_to = N });
    const bits = try b.transmute(back, .i32, &.{ M, N }, .{});
    const flat = try b.asDram(try b.reshape(bits, &.{MN}, .{}), .{});
    b.ret(&.{ sum, iota, pattern, flat });

    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    try expectContains(ir, "MN = #tcl.expr<\"mul\", [#tcl.symbol<\"M\">, #tcl.symbol<\"N\">]>");
    try expectContains(ir, "tcl.compiler_config = #tcl.config<{lowering_mode = \"Heuristic\", max_num_partitioning_axes = 1 : i64}>");
    try expectContains(ir, "%arg0: !tcl.mapped<f32, <[#tcl.expr<\"broadcast\", []>], [#tcl.symbol<\"M\">, #tcl.symbol<\"N\">], []>>");
    try expectContains(ir, "\"tcl.graph.as_logical\"(%arg0)");
    try expectContains(ir, "opcode = #tcl.ve_opcode<\"to_i15\">");
    try expectContains(ir, "operator = {Chip = #tcl.expr<\"broadcast\", []>, Split = [#tcl.symbol<\"M\">]}");
    inline for (.{ "gather", "scatter", "arange", "vector", "transmute", "reshape", "as_dram" }) |g|
        try expectContains(ir, "\"tcl.graph." ++ g ++ "\"");
}
