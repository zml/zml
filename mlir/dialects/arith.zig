const std = @import("std");

const mlir = @import("mlir2");

pub fn constant(ctx: *mlir.Context, value: *const mlir.Attribute, location: *const mlir.Location) *const mlir.Operation {
    return mlir.Operation.make(ctx, "arith.constant", .{
        .attributes = &.{
            .named(ctx, "value", value),
        },
        .result_type_inference = true,
        .location = location,
    });
}

fn binaryOp(comptime op_name: []const u8) type {
    return struct {
        pub fn call(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *const mlir.Operation {
            return mlir.Operation.make(ctx, op_name, .{
                .operands = .{ .flat = &.{ lhs, rhs } },
                .result_type_inference = true,
                .location = location,
            });
        }
    };
}

fn castOp(comptime op_name: []const u8) type {
    return struct {
        pub fn call(ctx: mlir.Context, value: *const mlir.Value, new_type: *const mlir.Type, location: *const mlir.Location) *const mlir.Operation {
            return mlir.Operation.make(ctx, op_name, .{
                .operands = .{ .float = &.{value} },
                .results = .{ .flat = &.{new_type} },
                .location = location,
            });
        }
    };
}

pub const addi = binaryOp("arith.addi").call;
pub const addf = binaryOp("arith.addf").call;
pub const subi = binaryOp("arith.subi").call;
pub const subf = binaryOp("arith.subf").call;
pub const muli = binaryOp("arith.muli").call;
pub const mulf = binaryOp("arith.mulf").call;
pub const divsi = binaryOp("arith.divsi").call;
pub const divui = binaryOp("arith.divui").call;
pub const divf = binaryOp("arith.divf").call;
pub const maxnumf = binaryOp("arith.maxnumf").call;
pub const maxnumi = binaryOp("arith.maxnumi").call;
pub const extsi = castOp("arith.extsi").call;
pub const extui = castOp("arith.extui").call;
pub const extf = castOp("arith.extf").call;
pub const trunci = castOp("arith.trunci").call;
pub const truncf = castOp("arith.truncf").call;
pub const fptosi = castOp("arith.fptosi").call;
pub const fptoui = castOp("arith.fptoui").call;
pub const sitofp = castOp("arith.sitofp").call;
pub const uitofp = castOp("arith.uitofp").call;

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

pub fn cmpi(ctx: mlir.Context, predicate: CmpIPredicate, lhs: *const mlir.Value, rhs: *const mlir.Value, location: mlir.Location) *const mlir.Operation {
    return mlir.Operation.make(ctx, "arith.cmpi", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .result_type_inference = true,
        .attributes = &.{
            .{ "predicate", mlir.integerAttribute(ctx, .i64, @intFromEnum(predicate)) },
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

pub fn cmpf(ctx: mlir.Context, predicate: CmpFPredicate, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *const mlir.Operation {
    return mlir.Operation.make(ctx, "arith.cmpf", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .result_type_inference = true,
        .attributes = &.{
            .named(ctx, "predicate", mlir.integerAttribute(ctx, .i64, @intFromEnum(predicate))),
        },
        .location = location,
    });
}
