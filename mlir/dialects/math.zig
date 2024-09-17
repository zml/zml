const std = @import("std");
const mlir = @import("mlir");

const namespace = "math";

fn unary_fn(comptime op_name: [:0]const u8) type {
    return struct {
        pub fn call(ctx: mlir.Context, value: mlir.Value, location: mlir.Location) mlir.Operation {
            return mlir.Operation.make(ctx, namespace ++ "." ++ op_name, .{
                .operands = &.{value},
                .results = &.{},
                .location = location,
            });
        }
    };
}

fn binary_fn(comptime op_name: [:0]const u8) type {
    return struct {
        pub fn call(ctx: mlir.Context, lhs: mlir.Value, rhs: mlir.Value, location: mlir.Location) mlir.Operation {
            return mlir.Operation.make(ctx, namespace ++ "." ++ op_name, .{
                .operands = &.{ lhs, rhs },
                .results = &.{},
                .location = location,
            });
        }
    };
}

pub const ipowi = binary_fn("ipowi").call;
pub const fpowi = binary_fn("fpowi").call;
pub const tanh = unary_fn("tanh").call;
pub const sqrt = unary_fn("sqrt").call;
pub const exp = unary_fn("exp").call;
pub const log = unary_fn("log").call;
