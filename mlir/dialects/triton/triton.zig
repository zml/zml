const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");

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

pub const CacheModifier = enum(i32) {
    none = 1,
    ca = 2,
    cg = 3,
    wb = 4,
    cs = 5,
    wt = 6,
    cv = 7,
};

pub const MemSemantic = enum(i32) {
    relaxed = 1,
    acquire = 2,
    release = 3,
    acquire_release = 4,
};

pub const EvictionPolicy = enum(i32) {
    normal = 1,
    evict_first = 2,
    evict_last = 3,
};

pub const PaddingOption = enum(i32) {
    pad_zero = 1,
    pad_nan = 2,
};

pub const AtomicRMWOp = enum(i32) {
    and_ = 1,
    or_ = 2,
    xor = 3,
    add = 4,
    fadd = 5,
    max = 6,
    min = 7,
    umax = 8,
    umin = 9,
    xchg = 10,
};

pub const MemSyncScope = enum(i32) {
    gpu = 1,
    cta = 2,
    system = 3,
};

pub const ProgramIDDim = enum(i32) {
    x = 0,
    y = 1,
    z = 2,
};

pub const RoundingMode = enum(i32) {
    rtz = 0,
    rtne = 1,
};

pub const PropagateNan = enum(i32) {
    none = 0,
    all = 0xFFFF,
};

pub const InputPrecision = enum(i32) {
    tf32 = 0,
    tf32x3 = 1,
    ieee = 2,
};

pub const ScaleDotElemType = enum(i32) {
    e4m3 = 0,
    e5m2 = 1,
    e2m3 = 2,
    e3m2 = 3,
    e2m1 = 4,
    bf16 = 5,
    fp16 = 6,
};
