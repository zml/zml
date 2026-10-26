const std = @import("std");

const mlir = @import("mlir");
const stablehlo = @import("mlir/dialects/stablehlo");

const operation_names: []const []const u8 = &.{
    "stablehlo.abs",
    "stablehlo.add",
    "stablehlo.all_reduce",
    "stablehlo.and",
    "stablehlo.atan2",
    "stablehlo.bitcast_convert",
    "stablehlo.broadcast_in_dim",
    "stablehlo.cbrt",
    "stablehlo.ceil",
    "stablehlo.cholesky",
    "stablehlo.clamp",
    "stablehlo.compare",
    "stablehlo.complex",
    "stablehlo.composite",
    "stablehlo.concatenate",
    "stablehlo.constant",
    "stablehlo.convert",
    "stablehlo.convolution",
    "stablehlo.cosine",
    "stablehlo.count_leading_zeros",
    "stablehlo.custom_call",
    "stablehlo.divide",
    "stablehlo.dot_general",
    "stablehlo.dynamic_slice",
    "stablehlo.dynamic_update_slice",
    "stablehlo.exponential",
    "stablehlo.exponential_minus_one",
    "stablehlo.fft",
    "stablehlo.floor",
    "stablehlo.gather",
    "stablehlo.get_tuple_element",
    "stablehlo.imag",
    "stablehlo.iota",
    "stablehlo.is_finite",
    "stablehlo.log",
    "stablehlo.log_plus_one",
    "stablehlo.logistic",
    "stablehlo.maximum",
    "stablehlo.minimum",
    "stablehlo.multiply",
    "stablehlo.negate",
    "stablehlo.not",
    "stablehlo.optimization_barrier",
    "stablehlo.or",
    "stablehlo.pad",
    "stablehlo.partition_id",
    "stablehlo.popcnt",
    "stablehlo.power",
    "stablehlo.real",
    "stablehlo.reduce_precision",
    "stablehlo.remainder",
    "stablehlo.reshape",
    "stablehlo.return",
    "stablehlo.reverse",
    "stablehlo.rng",
    "stablehlo.rng_bit_generator",
    "stablehlo.round_nearest_afz",
    "stablehlo.round_nearest_even",
    "stablehlo.rsqrt",
    "stablehlo.scatter",
    "stablehlo.select",
    "stablehlo.shift_left",
    "stablehlo.shift_right_arithmetic",
    "stablehlo.shift_right_logical",
    "stablehlo.sign",
    "stablehlo.sine",
    "stablehlo.slice",
    "stablehlo.sqrt",
    "stablehlo.subtract",
    "stablehlo.tanh",
    "stablehlo.transpose",
    "stablehlo.triangular_solve",
    "stablehlo.tuple",
    "stablehlo.while",
    "stablehlo.xor",
};

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(stablehlo);
    inline for (comptime std.meta.declarations(stablehlo)) |decl| {
        const T = @field(stablehlo, decl);
        if (@TypeOf(T) == type and @typeInfo(T) == .@"opaque") std.testing.refAllDecls(T);
    }
}

test "every bound operation is registered" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    stablehlo.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    for (operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}
