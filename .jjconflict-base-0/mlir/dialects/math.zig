const std = @import("std");

const mlir = @import("mlir");

// =============================================================================
// Float unary — SameOperandsAndResultType. Result type inferred from operand.
// =============================================================================

fn floatUnaryOp(comptime op_name: []const u8) type {
    return struct {
        pub fn call(
            ctx: *mlir.Context,
            src: *const mlir.Value,
            location: *const mlir.Location,
        ) *mlir.Operation {
            return mlir.Operation.make(ctx, op_name, .{
                .operands = .{ .flat = &.{src} },
                .result_type_inference = true,
                .location = location,
            });
        }
    };
}

// Exponentials / logs
pub const exp = floatUnaryOp("math.exp").call;
pub const exp2 = floatUnaryOp("math.exp2").call;
pub const expm1 = floatUnaryOp("math.expm1").call;
pub const log = floatUnaryOp("math.log").call;
pub const log2 = floatUnaryOp("math.log2").call;
pub const log10 = floatUnaryOp("math.log10").call;
pub const log1p = floatUnaryOp("math.log1p").call;

// Trig / hyperbolic
pub const sin = floatUnaryOp("math.sin").call;
pub const cos = floatUnaryOp("math.cos").call;
pub const tan = floatUnaryOp("math.tan").call;
pub const asin = floatUnaryOp("math.asin").call;
pub const acos = floatUnaryOp("math.acos").call;
pub const atan = floatUnaryOp("math.atan").call;
pub const sinh = floatUnaryOp("math.sinh").call;
pub const cosh = floatUnaryOp("math.cosh").call;
pub const tanh = floatUnaryOp("math.tanh").call;
pub const asinh = floatUnaryOp("math.asinh").call;
pub const acosh = floatUnaryOp("math.acosh").call;
pub const atanh = floatUnaryOp("math.atanh").call;

// Roots / errors / abs
pub const sqrt = floatUnaryOp("math.sqrt").call;
pub const rsqrt = floatUnaryOp("math.rsqrt").call;
pub const cbrt = floatUnaryOp("math.cbrt").call;
pub const erf = floatUnaryOp("math.erf").call;
pub const erfc = floatUnaryOp("math.erfc").call;
pub const absf = floatUnaryOp("math.absf").call;

// Rounding
pub const floor = floatUnaryOp("math.floor").call;
pub const ceil = floatUnaryOp("math.ceil").call;
pub const round = floatUnaryOp("math.round").call;
pub const roundeven = floatUnaryOp("math.roundeven").call;
pub const trunc = floatUnaryOp("math.trunc").call;

// =============================================================================
// Float binary — SameOperandsAndResultType.
// =============================================================================

fn floatBinaryOp(comptime op_name: []const u8) type {
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

pub const powf = floatBinaryOp("math.powf").call;
pub const atan2 = floatBinaryOp("math.atan2").call;
pub const copysign = floatBinaryOp("math.copysign").call;

// =============================================================================
// Float ternary — SameOperandsAndResultType.
// =============================================================================

pub fn fma(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "math.fma", .{
        .operands = .{ .flat = &.{ a, b, c } },
        .result_type_inference = true,
        .location = location,
    });
}

/// math.clampf value, min, max — clamp to [min, max], same type throughout.
/// Distinct from `tt.clampf` (which has an explicit `propagateNan` attr); the
/// math version uses fastmath for NaN behaviour.
pub fn clampf(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    min: *const mlir.Value,
    max: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "math.clampf", .{
        .operands = .{ .flat = &.{ value, min, max } },
        .result_type_inference = true,
        .location = location,
    });
}

// =============================================================================
// Sincos — operand → (sin, cos), both same type as operand.
// =============================================================================

pub fn sincos(
    ctx: *mlir.Context,
    operand: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "math.sincos", .{
        .operands = .{ .flat = &.{operand} },
        .results = .{ .flat = &.{ operand.type_(), operand.type_() } },
        .location = location,
    });
}

// =============================================================================
// FPowI — float base, signless integer exponent → float (same type as base).
// =============================================================================

pub fn fpowi(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    power: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "math.fpowi", .{
        .operands = .{ .flat = &.{ base, power } },
        .results = .{ .flat = &.{base.type_()} },
        .location = location,
    });
}

// =============================================================================
// Integer unary
// =============================================================================

fn intUnaryOp(comptime op_name: []const u8) type {
    return struct {
        pub fn call(
            ctx: *mlir.Context,
            src: *const mlir.Value,
            location: *const mlir.Location,
        ) *mlir.Operation {
            return mlir.Operation.make(ctx, op_name, .{
                .operands = .{ .flat = &.{src} },
                .result_type_inference = true,
                .location = location,
            });
        }
    };
}

pub const absi = intUnaryOp("math.absi").call;
pub const ctlz = intUnaryOp("math.ctlz").call;
pub const cttz = intUnaryOp("math.cttz").call;
pub const ctpop = intUnaryOp("math.ctpop").call;

/// math.ipowi(base, exp) — integer power. Result type matches base.
pub const ipowi = floatBinaryOp("math.ipowi").call;

// =============================================================================
// Float classification — operand is float-like, result is i1-like.
// =============================================================================

fn classifyOp(comptime op_name: []const u8) type {
    return struct {
        pub fn call(
            ctx: *mlir.Context,
            src: *const mlir.Value,
            result_type: *const mlir.Type,
            location: *const mlir.Location,
        ) *mlir.Operation {
            return mlir.Operation.make(ctx, op_name, .{
                .operands = .{ .flat = &.{src} },
                .results = .{ .flat = &.{result_type} },
                .location = location,
            });
        }
    };
}

pub const isfinite = classifyOp("math.isfinite").call;
pub const isinf = classifyOp("math.isinf").call;
pub const isnan = classifyOp("math.isnan").call;
pub const isnormal = classifyOp("math.isnormal").call;

test {
    std.testing.refAllDecls(@This());
}
