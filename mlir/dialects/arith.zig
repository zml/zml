const std = @import("std");
const mlir = @import("mlir");

pub fn constant(ctx: mlir.Context, value: mlir.Attribute, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "arith.constant", .{
        .attributes = &.{
            .{ "value", value },
        },
        .result_type_inference = true,
        .location = location,
    });
}

fn binary_fn(comptime op_name: [:0]const u8) fn (mlir.Context, mlir.Value, mlir.Value, mlir.Location) mlir.Operation {
    return struct {
        pub fn call(ctx: mlir.Context, lhs: mlir.Value, rhs: mlir.Value, location: mlir.Location) mlir.Operation {
            return mlir.Operation.make(ctx, op_name, .{
                .operands = &.{ lhs, rhs },
                .result_type_inference = true,
                .location = location,
            });
        }
    }.call;
}

fn cast_fn(comptime op_name: [:0]const u8) fn (mlir.Context, mlir.Value, mlir.Type, mlir.Location) mlir.Operation {
    return struct {
        pub fn call(ctx: mlir.Context, value: mlir.Value, new_type: mlir.Type, location: mlir.Location) mlir.Operation {
            return mlir.Operation.make(ctx, op_name, .{
                .operands = &.{value},
                .results = &.{new_type},
                .location = location,
            });
        }
    }.call;
}

pub const addi = binary_fn("arith.addi");
pub const addf = binary_fn("arith.addf");
pub const subi = binary_fn("arith.subi");
pub const subf = binary_fn("arith.subf");
pub const muli = binary_fn("arith.muli");
pub const mulf = binary_fn("arith.mulf");
pub const divsi = binary_fn("arith.divsi");
pub const divui = binary_fn("arith.divui");
pub const divf = binary_fn("arith.divf");
pub const extsi = cast_fn("arith.extsi");
pub const extui = cast_fn("arith.extui");
pub const extf = cast_fn("arith.extf");
pub const trunci = cast_fn("arith.trunci");
pub const truncf = cast_fn("arith.truncf");
pub const fptosi = cast_fn("arith.fptosi");
pub const fptoui = cast_fn("arith.fptoui");
pub const sitofp = cast_fn("arith.sitofp");
pub const uitofp = cast_fn("arith.uitofp");

pub const CmpIPredicate = enum {
    eq,
    ne,
    slt,
    sle,
    sgt,
    sge,
    ult,
    ule,
    ugt,
    uge,
};

pub fn cmpi(ctx: mlir.Context, predicate: CmpIPredicate, lhs: mlir.Value, rhs: mlir.Value, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "arith.cmpi", .{
        .operands = &.{ lhs, rhs },
        .result_type_inference = true,
        .attributes = &.{
            .{ "predicate", mlir.IntegerAttribute(.i64).init(ctx, @intFromEnum(predicate)).as(mlir.Attribute).? },
        },
        .location = location,
    });
}

pub const CmpFPredicate = enum {
    false,
    oeq,
    ogt,
    oge,
    olt,
    ole,
    one,
    ord,
    ueq,
    ugt,
    uge,
    ult,
    ule,
    une,
    uno,
    true,
};

pub fn cmpf(ctx: mlir.Context, predicate: CmpFPredicate, lhs: mlir.Value, rhs: mlir.Value, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "arith.cmpf", .{
        .operands = &.{ lhs, rhs },
        .result_type_inference = true,
        .attributes = &.{
            .{ "predicate", mlir.IntegerAttribute(.i64).init(ctx, @intFromEnum(predicate)).as(mlir.Attribute).? },
        },
        .location = location,
    });
}
