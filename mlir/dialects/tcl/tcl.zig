const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");

pub const dialects_needed = [_][]const u8{ "func", "arith", "tcl" };

pub const Error = error{InvalidMlir};

fn stringRef(s: []const u8) c.MlirStringRef {
    return .{ .data = s.ptr, .length = s.len };
}

fn attribute(a: c.MlirAttribute) Error!*const mlir.Attribute {
    return if (a.ptr) |p| @ptrCast(p) else error.InvalidMlir;
}

fn type_(t: c.MlirType) Error!*const mlir.Type {
    return if (t.ptr) |p| @ptrCast(p) else error.InvalidMlir;
}

// =============================================================================
// Attributes
// =============================================================================

pub const Tactic = enum {
    einsum_by_dpe,
    reduce_by_ve,
    interleaving,
    elementwise,
    tensor_operation,
    einsum_by_ve,
    filter_compaction,

    pub fn name(self: Tactic) []const u8 {
        return switch (self) {
            .einsum_by_dpe => "EinsumByDpe",
            .reduce_by_ve => "ReduceByVe",
            .interleaving => "Interleaving",
            .elementwise => "Elementwise",
            .tensor_operation => "TensorOperation",
            .einsum_by_ve => "EinsumByVe",
            .filter_compaction => "FilterCompaction",
        };
    }
};

pub const ReduceMode = enum {
    addi,
    addf,
    maxi,
    maxf,
    mini,
    minf,
    /// A running i32 sum that keeps its axes.
    cumsum,

    pub fn name(self: ReduceMode) []const u8 {
        return switch (self) {
            .addi => "Addi",
            .addf => "Addf",
            .maxi => "Maxi",
            .maxf => "Maxf",
            .mini => "Mini",
            .minf => "Minf",
            .cumsum => "Cumsum",
        };
    }
};

pub const Predicate = enum { eq, ne, lt, le, gt, ge };

/// An axis or another TCL identifier: `#tcl.symbol<"A">`.
pub fn symbol(ctx: *mlir.Context, name: []const u8) Error!*const mlir.Attribute {
    return attribute(c.mlirTclSymbolAttrGet(ctx.ptr(), stringRef(name)));
}

/// `#tcl.expr<kind, [args]>`: `add`, `mul`, `stride`, `broadcast`, `arg`, ...
pub fn expr(ctx: *mlir.Context, kind: []const u8, args: []const *const mlir.Attribute) Error!*const mlir.Attribute {
    return attribute(c.mlirTclExprAttrGet(ctx.ptr(), stringRef(kind), mlir.Attribute.array(ctx, args).ptr()));
}

pub fn tactic(ctx: *mlir.Context, value: Tactic) *const mlir.Attribute {
    return attribute(c.mlirTclTacticAttrGet(ctx.ptr(), stringRef(value.name()))) catch unreachable;
}

/// A VE opcode of the SDK vocabulary (`exp`, `addf`, `to_f16`, ...).
pub fn veOpcode(ctx: *mlir.Context, opcode: []const u8) Error!*const mlir.Attribute {
    return attribute(c.mlirTclVeOpcodeAttrGet(ctx.ptr(), stringRef(opcode)));
}

pub fn reduceMode(ctx: *mlir.Context, mode: ReduceMode) *const mlir.Attribute {
    return attribute(c.mlirTclReduceModeAttrGet(ctx.ptr(), stringRef(mode.name()))) catch unreachable;
}

pub fn predicate(ctx: *mlir.Context, value: Predicate) *const mlir.Attribute {
    return attribute(c.mlirTclPredicateAttrGet(ctx.ptr(), stringRef(@tagName(value)))) catch unreachable;
}

/// `@ context(operator = {Chip: ...}, heuristic_hint = {...})`.
pub fn context(ctx: *mlir.Context, fields: []const mlir.NamedAttribute) Error!*const mlir.Attribute {
    return attribute(c.mlirTclContextAttrGet(ctx.ptr(), mlir.Attribute.dict(ctx, fields).ptr()));
}

/// `read(x, subtraction=, table_lookup=, broadcast_to=, typecast_to=, pad=)`.
pub fn readOptions(ctx: *mlir.Context, fields: []const mlir.NamedAttribute) Error!*const mlir.Attribute {
    return attribute(c.mlirTclReadOptionsAttrGet(ctx.ptr(), mlir.Attribute.dict(ctx, fields).ptr()));
}

/// `@compiler_config({...})` / `@auto_config({...})`, JSON-compatible fields.
pub fn config(ctx: *mlir.Context, fields: []const mlir.NamedAttribute) Error!*const mlir.Attribute {
    return attribute(c.mlirTclConfigAttrGet(ctx.ptr(), mlir.Attribute.dict(ctx, fields).ptr()));
}

/// `dram(chip | inner)`, with the original axes of a padded mapping.
pub fn dramMapping(ctx: *mlir.Context, chip: []const *const mlir.Attribute, inner: []const *const mlir.Attribute, original: []const *const mlir.Attribute) Error!*const mlir.Attribute {
    return attribute(c.mlirTclDramMappingAttrGet(
        ctx.ptr(),
        mlir.Attribute.array(ctx, chip).ptr(),
        mlir.Attribute.array(ctx, inner).ptr(),
        mlir.Attribute.array(ctx, original).ptr(),
    ));
}

// =============================================================================
// Types
// =============================================================================

/// `[A, B]/bf16`: a logical tensor over named axes.
pub fn logicalType(ctx: *mlir.Context, element: *const mlir.Type, axes: []const *const mlir.Attribute) Error!*const mlir.Type {
    return type_(c.mlirTclLogicalTypeGet(ctx.ptr(), element.ptr(), mlir.Attribute.array(ctx, axes).ptr()));
}

/// `dram(...)/bf16`: a tensor in an explicit DRAM mapping.
pub fn mappedType(ctx: *mlir.Context, element: *const mlir.Type, mapping: *const mlir.Attribute) Error!*const mlir.Type {
    return type_(c.mlirTclMappedTypeGet(ctx.ptr(), element.ptr(), mapping.ptr()));
}

pub const LogicalType = opaque {
    const M = mlir.Methods(LogicalType, c.MlirType);

    pub const isAFn = c.mlirTypeIsATclLogical;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);

    pub fn elementType(self: *const LogicalType) *const mlir.Type {
        return @ptrCast(c.mlirTclLogicalTypeGetElementType(self.ptr()).ptr);
    }

    pub fn axes(self: *const LogicalType) *const mlir.ArrayAttribute {
        return @ptrCast(c.mlirTclLogicalTypeGetAxes(self.ptr()).ptr);
    }
};

// =============================================================================
// Operations
// =============================================================================
//
// Builders do not verify: kernel instructions only verify inside a
// `tcl.kernel`, and a kernel or graph operation only once its operands
// dominate it. Verify each graph-level operation after inserting it.

pub fn kernel(ctx: *mlir.Context, inputs: []const *const mlir.Value, output: *const mlir.Type, tactic_: Tactic, context_: ?*const mlir.Attribute, body: *mlir.Block, location: *const mlir.Location) *mlir.Operation {
    var attrs: [2]mlir.NamedAttribute = undefined;
    attrs[0] = .named(ctx, "tactic", tactic(ctx, tactic_));
    if (context_) |a| attrs[1] = .named(ctx, "context", a);
    return mlir.Operation.make(ctx, "tcl.kernel", .{
        .operands = .{ .flat = inputs },
        .results = .{ .flat = &.{output} },
        .attributes = attrs[0 .. @as(usize, 1) + @intFromBool(context_ != null)],
        .blocks = &.{body},
        .verify = false,
        .location = location,
    });
}

pub fn read(ctx: *mlir.Context, inputs: []const *const mlir.Value, output: *const mlir.Type, options: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: [1]mlir.NamedAttribute = undefined;
    if (options) |o| attrs[0] = .named(ctx, "options", o);
    return mlir.Operation.make(ctx, "tcl.read", .{
        .operands = .{ .flat = inputs },
        .results = .{ .flat = &.{output} },
        .attributes = attrs[0..@intFromBool(options != null)],
        .verify = false,
        .location = location,
    });
}

pub fn dpe(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, output: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "tcl.dpe", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{output} },
        .verify = false,
        .location = location,
    });
}

pub fn ve(ctx: *mlir.Context, inputs: []const *const mlir.Value, opcode: *const mlir.Attribute, output: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "tcl.ve", .{
        .operands = .{ .flat = inputs },
        .results = .{ .flat = &.{output} },
        .attributes = &.{.named(ctx, "opcode", opcode)},
        .verify = false,
        .location = location,
    });
}

pub fn ve_reduce(ctx: *mlir.Context, input: *const mlir.Value, mode: ReduceMode, axes: []const *const mlir.Attribute, output: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "tcl.ve_reduce", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{output} },
        .attributes = &.{
            .named(ctx, "mode", reduceMode(ctx, mode)),
            .named(ctx, "axes", .array(ctx, axes)),
        },
        .verify = false,
        .location = location,
    });
}

pub fn ve_select(ctx: *mlir.Context, condition: *const mlir.Value, predicate_: Predicate, threshold: *const mlir.Attribute, true_value: *const mlir.Value, false_value: *const mlir.Value, output: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "tcl.ve_select", .{
        .operands = .{ .flat = &.{ condition, true_value, false_value } },
        .results = .{ .flat = &.{output} },
        .attributes = &.{
            .named(ctx, "predicate", predicate(ctx, predicate_)),
            .named(ctx, "threshold", threshold),
        },
        .verify = false,
        .location = location,
    });
}

pub fn write(ctx: *mlir.Context, input: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "tcl.write", .{
        .operands = .{ .flat = &.{input} },
        .verify = false,
        .location = location,
    });
}

pub const GraphOp = enum { gather, all_gather, scatter, reshape, transmute, concat, slice, arange, vector, as_logical, as_dram, index_read, index_write, scratchpad, full, reduce_max_i32, sym_expr };

/// `tcl.graph.<op>`: a graph-level TCL operation whose fields go in `options`.
pub fn graph(ctx: *mlir.Context, comptime op: GraphOp, inputs: []const *const mlir.Value, output: *const mlir.Type, options: []const mlir.NamedAttribute, context_: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: [2]mlir.NamedAttribute = undefined;
    attrs[0] = .named(ctx, "options", .dict(ctx, options));
    if (context_) |a| attrs[1] = .named(ctx, "context", a);
    return mlir.Operation.make(ctx, "tcl.graph." ++ @tagName(op), .{
        .operands = .{ .flat = inputs },
        .results = .{ .flat = &.{output} },
        .attributes = attrs[0 .. @as(usize, 1) + @intFromBool(context_ != null)],
        .verify = false,
        .location = location,
    });
}

/// `tcl.graph.for`: runs `body` up to `limit` (a scalar i32 tensor) or `bound`
/// (an integer or axis attribute) times. The body's arguments are the index,
/// then one accumulator per `inits`; it ends with `tcl.graph.yield`.
pub fn loop(ctx: *mlir.Context, limit: ?*const mlir.Value, bound: ?*const mlir.Attribute, inits: []const *const mlir.Value, result_types: []const *const mlir.Type, body: *mlir.Block, location: *const mlir.Location) *mlir.Operation {
    const limits: []const *const mlir.Value = if (limit) |l| &.{l} else &.{};
    var attrs: [1]mlir.NamedAttribute = undefined;
    if (bound) |b| attrs[0] = .named(ctx, "bound", b);
    return mlir.Operation.make(ctx, "tcl.graph.for", .{
        .operands = .{ .variadic = &.{ limits, inits } },
        .results = .{ .flat = result_types },
        .attributes = attrs[0..@intFromBool(bound != null)],
        .blocks = &.{body},
        .verify = false,
        .location = location,
    });
}

pub fn yield(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "tcl.graph.yield", .{
        .operands = .{ .flat = values },
        .verify = false,
        .location = location,
    });
}

pub fn registerDialects(registry: *mlir.DialectRegistry) void {
    inline for (dialects_needed) |d| mlir.DialectHandle.fromString(d).insertDialect(registry);
}

test "types and attributes verify" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    registerDialects(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();

    const a = try symbol(ctx, "A");
    const t = try logicalType(ctx, .float(ctx, .bf16), &.{a});
    try std.testing.expect(t.isA(LogicalType) != null);
    try std.testing.expectError(error.InvalidMlir, symbol(ctx, "not an axis"));
    try std.testing.expectError(error.InvalidMlir, logicalType(ctx, .float(ctx, .bf16), &.{ a, a }));
    try std.testing.expectError(error.InvalidMlir, veOpcode(ctx, "unknown"));
    _ = try veOpcode(ctx, "to_f16");
}
