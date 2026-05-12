const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

// =============================================================================
// Predicate enums (match ArithBase.td exactly)
// =============================================================================

/// Arith cmpi predicates. Values match mlir::arith::CmpIPredicate (I64EnumAttr).
pub const CmpIPredicate = enum(i64) {
    eq = 0,
    ne = 1,
    slt = 2,
    sle = 3,
    sgt = 4,
    sge = 5,
    ult = 6,
    ule = 7,
    ugt = 8,
    uge = 9,
};

/// Arith cmpf predicates. Values match mlir::arith::CmpFPredicate.
pub const CmpFPredicate = enum(i64) {
    always_false = 0,
    oeq = 1,
    ogt = 2,
    oge = 3,
    olt = 4,
    ole = 5,
    one = 6,
    ord = 7,
    ueq = 8,
    ugt = 9,
    uge = 10,
    ult = 11,
    ule = 12,
    une = 13,
    uno = 14,
    always_true = 15,
};

// =============================================================================
// Constants
// =============================================================================

/// arith.constant for an integer scalar. `value` is reinterpreted to `int_type`'s bit width.
pub fn constant_int(
    ctx: *mlir.Context,
    value: i64,
    int_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    const attr: *const mlir.Attribute = @ptrCast(c.mlirIntegerAttrGet(int_type.ptr(), value).ptr);
    return mlir.Operation.make(ctx, "arith.constant", .{
        .results = .{ .flat = &.{int_type} },
        .attributes = &.{.named(ctx, "value", attr)},
        .location = location,
    });
}

/// arith.constant for a float scalar of the given FloatTypes kind.
pub fn constant_float(
    ctx: *mlir.Context,
    value: f64,
    comptime ft: mlir.FloatTypes,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "arith.constant", .{
        .results = .{ .flat = &.{mlir.floatType(ctx, ft)} },
        .attributes = &.{.named(ctx, "value", mlir.floatAttribute(ctx, ft, value))},
        .location = location,
    });
}

/// arith.constant for an `index`-typed integer.
pub fn constant_index(ctx: *mlir.Context, value: i64, location: *const mlir.Location) *mlir.Operation {
    const idx_ty = mlir.indexType(ctx);
    const attr: *const mlir.Attribute = @ptrCast(c.mlirIntegerAttrGet(idx_ty.ptr(), value).ptr);
    return mlir.Operation.make(ctx, "arith.constant", .{
        .results = .{ .flat = &.{idx_ty} },
        .attributes = &.{.named(ctx, "value", attr)},
        .location = location,
    });
}

// =============================================================================
// Binary ops (generated from shared template)
// =============================================================================

fn binaryOp(comptime op_name: []const u8) type {
    return struct {
        pub fn call(
            ctx: *mlir.Context,
            lhs: *const mlir.Value,
            rhs: *const mlir.Value,
            location: *const mlir.Location,
        ) *mlir.Operation {
            return mlir.Operation.make(ctx, op_name, .{
                .operands = .{ .flat = &.{ lhs, rhs } },
                .result_type_inference = true,
                .location = location,
            });
        }
    };
}

// Integer arithmetic
pub const addi = binaryOp("arith.addi").call;
pub const subi = binaryOp("arith.subi").call;
pub const muli = binaryOp("arith.muli").call;
pub const divsi = binaryOp("arith.divsi").call;
pub const divui = binaryOp("arith.divui").call;
pub const remsi = binaryOp("arith.remsi").call;
pub const remui = binaryOp("arith.remui").call;

// Integer bitwise / shifts
pub const andi = binaryOp("arith.andi").call;
pub const ori = binaryOp("arith.ori").call;
pub const xori = binaryOp("arith.xori").call;
pub const shli = binaryOp("arith.shli").call;
pub const shrsi = binaryOp("arith.shrsi").call;
pub const shrui = binaryOp("arith.shrui").call;

// Integer min/max
pub const maxsi = binaryOp("arith.maxsi").call;
pub const maxui = binaryOp("arith.maxui").call;
pub const minsi = binaryOp("arith.minsi").call;
pub const minui = binaryOp("arith.minui").call;

// Ceil/floor div
pub const ceildivsi = binaryOp("arith.ceildivsi").call;
pub const ceildivui = binaryOp("arith.ceildivui").call;
pub const floordivsi = binaryOp("arith.floordivsi").call;

// Float arithmetic
pub const addf = binaryOp("arith.addf").call;
pub const subf = binaryOp("arith.subf").call;
pub const mulf = binaryOp("arith.mulf").call;
pub const divf = binaryOp("arith.divf").call;
pub const remf = binaryOp("arith.remf").call;

// Float min/max
pub const maximumf = binaryOp("arith.maximumf").call;
pub const maxnumf = binaryOp("arith.maxnumf").call;
pub const minimumf = binaryOp("arith.minimumf").call;
pub const minnumf = binaryOp("arith.minnumf").call;

// =============================================================================
// Unary ops
// =============================================================================

pub fn negf(ctx: *mlir.Context, src: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "arith.negf", .{
        .operands = .{ .flat = &.{src} },
        .result_type_inference = true,
        .location = location,
    });
}

/// arith.flush_denormals — clamp subnormals to zero (SameOperandsAndResultType).
pub fn flush_denormals(ctx: *mlir.Context, src: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "arith.flush_denormals", .{
        .operands = .{ .flat = &.{src} },
        .result_type_inference = true,
        .location = location,
    });
}

/// arith.constant for a dense splat: builds `arith.constant dense<value> : T`
/// where `T` is a ranked tensor type. Useful for tensor-typed zero/one constants.
/// For float splats, use `constant_dense_splat_f`; for ints, `constant_dense_splat_i`.
pub fn constant_dense_splat_f(
    ctx: *mlir.Context,
    value: f64,
    tensor_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    const elem_c = c.mlirShapedTypeGetElementType(tensor_type.ptr());
    const elem_attr = c.mlirFloatAttrDoubleGet(ctx.ptr(), elem_c, value);
    const dense_c = c.mlirDenseElementsAttrSplatGet(tensor_type.ptr(), elem_attr);
    const dense_attr: *const mlir.Attribute = @ptrCast(dense_c.ptr);
    return mlir.Operation.make(ctx, "arith.constant", .{
        .results = .{ .flat = &.{tensor_type} },
        .attributes = &.{.named(ctx, "value", dense_attr)},
        .location = location,
    });
}

pub fn constant_dense_splat_i(
    ctx: *mlir.Context,
    value: i64,
    tensor_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    const elem_c = c.mlirShapedTypeGetElementType(tensor_type.ptr());
    const elem_attr = c.mlirIntegerAttrGet(elem_c, value);
    const dense_c = c.mlirDenseElementsAttrSplatGet(tensor_type.ptr(), elem_attr);
    const dense_attr: *const mlir.Attribute = @ptrCast(dense_c.ptr);
    return mlir.Operation.make(ctx, "arith.constant", .{
        .results = .{ .flat = &.{tensor_type} },
        .attributes = &.{.named(ctx, "value", dense_attr)},
        .location = location,
    });
}

// =============================================================================
// Extended-result binary ops — return two values
// =============================================================================

/// arith.addui_extended — unsigned add, returns (sum, overflow: i1 / i1-tensor).
/// The overflow type is typically i1 (scalar) or tensor<...xi1> matching lhs shape.
pub fn addui_extended(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    overflow_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "arith.addui_extended", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{ lhs.type_(), overflow_type } },
        .location = location,
    });
}

/// arith.mulsi_extended — signed multiply, returns (low, high) both same type as lhs.
pub fn mulsi_extended(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "arith.mulsi_extended", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{ lhs.type_(), lhs.type_() } },
        .location = location,
    });
}

/// arith.mului_extended — unsigned multiply, returns (low, high) both same type as lhs.
pub fn mului_extended(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "arith.mului_extended", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{ lhs.type_(), lhs.type_() } },
        .location = location,
    });
}

// =============================================================================
// Cast ops (1 operand, explicit result type)
// =============================================================================

fn castOp(comptime op_name: []const u8) type {
    return struct {
        pub fn call(
            ctx: *mlir.Context,
            value: *const mlir.Value,
            result_type: *const mlir.Type,
            location: *const mlir.Location,
        ) *mlir.Operation {
            return mlir.Operation.make(ctx, op_name, .{
                .operands = .{ .flat = &.{value} },
                .results = .{ .flat = &.{result_type} },
                .location = location,
            });
        }
    };
}

pub const extsi = castOp("arith.extsi").call;
pub const extui = castOp("arith.extui").call;
pub const extf = castOp("arith.extf").call;
pub const trunci = castOp("arith.trunci").call;
pub const truncf = castOp("arith.truncf").call;
pub const sitofp = castOp("arith.sitofp").call;
pub const uitofp = castOp("arith.uitofp").call;
pub const fptosi = castOp("arith.fptosi").call;
pub const fptoui = castOp("arith.fptoui").call;
pub const bitcast = castOp("arith.bitcast").call;
pub const index_cast = castOp("arith.index_cast").call;
pub const index_castui = castOp("arith.index_castui").call;

// =============================================================================
// convertf — same-bitwidth float cast (e.g. f16↔bf16). Not expressible via
// arith.extf or arith.truncf.
// =============================================================================

/// Matches `Arith_RoundingModeAttr` (I32EnumAttr; see arith_dialect.td).
pub const RoundingMode = enum(i32) {
    to_nearest_even = 0,
    downward = 1,
    upward = 2,
    toward_zero = 3,
    to_nearest_away = 4,
};

pub const ConvertFOpts = struct {
    rounding: ?RoundingMode = null,
};

pub fn convertf(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    opts: ConvertFOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.rounding) |rm| {
        attrs.appendAssumeCapacity(.named(ctx, "roundingmode", mlir.integerAttribute(ctx, .i32, @intFromEnum(rm))));
    }
    return mlir.Operation.make(ctx, "arith.convertf", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// arith.scaling_truncf — OCP MXFP downcast with per-block scale factors.
pub fn scaling_truncf(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    scale: *const mlir.Value,
    result_type: *const mlir.Type,
    opts: ConvertFOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.rounding) |rm| {
        attrs.appendAssumeCapacity(.named(ctx, "roundingmode", mlir.integerAttribute(ctx, .i32, @intFromEnum(rm))));
    }
    return mlir.Operation.make(ctx, "arith.scaling_truncf", .{
        .operands = .{ .flat = &.{ src, scale } },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// =============================================================================
// Compare
// =============================================================================

pub fn cmpi(
    ctx: *mlir.Context,
    predicate: CmpIPredicate,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "arith.cmpi", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .result_type_inference = true,
        .attributes = &.{
            .named(ctx, "predicate", mlir.integerAttribute(ctx, .i64, @intFromEnum(predicate))),
        },
        .location = location,
    });
}

pub fn cmpf(
    ctx: *mlir.Context,
    predicate: CmpFPredicate,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "arith.cmpf", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .result_type_inference = true,
        .attributes = &.{
            .named(ctx, "predicate", mlir.integerAttribute(ctx, .i64, @intFromEnum(predicate))),
        },
        .location = location,
    });
}

// =============================================================================
// Select
// =============================================================================

/// arith.select — result type matches the true/false operand type.
pub fn select(
    ctx: *mlir.Context,
    cond: *const mlir.Value,
    t: *const mlir.Value,
    f: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "arith.select", .{
        .operands = .{ .flat = &.{ cond, t, f } },
        .results = .{ .flat = &.{t.type_()} },
        .location = location,
    });
}

test {
    std.testing.refAllDecls(@This());
}
