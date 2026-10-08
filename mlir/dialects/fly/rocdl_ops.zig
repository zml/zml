//! `fly_rocdl` operation builders, generated from FlyDSL's `include/flydsl/Dialect/FlyROCDL/IR/Ops.td` at
//! 1941889400621f1fc3d8f03bada1a4e7380cdf7e; do not edit, see REGENERATING.md.
//!
//! Builders take the operands, then the result types (only for operations
//! that do not infer them), then the attributes, then the location.
//! Parameters are the ODS names in snake case, with a trailing underscore
//! where needed. Pass `null` for absent optional operands or attributes and
//! `&.{}` for empty variadic ones; variable-length operand groups of
//! `AttrSizedOperandSegments` operations get their segment sizes from
//! `Operation.make`. Attributes use `mlir.Attribute`.

const std = @import("std");

const mlir = @import("mlir");
const stdx = @import("stdx");

const make = @import("fly.zig").make;

/// Every operation bound here, for the registration test.
pub const names = [_][]const u8{
    "fly_rocdl.get_buffer_rsrc",
    "fly_rocdl.make_tiled_tdm_load_atom",
    "fly_rocdl.make_tiled_tdm_store_atom",
};

/// `fly_rocdl.get_buffer_rsrc`. Extract the raw ROCDL buffer resource from a buffer-descriptor pointer. Result types are inferred.
pub fn get_buffer_rsrc(ctx: *mlir.Context, ptr: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "fly_rocdl.get_buffer_rsrc", .{
        .operands = .{ .flat = &.{ptr} },
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly_rocdl.make_tiled_tdm_load_atom`. Build a CDNA5 TDM load atom and the coordinate tensor that addresses it. Result types are inferred.
pub fn make_tiled_tdm_load_atom(ctx: *mlir.Context, tensor: *const mlir.Value, smem_layout: *const mlir.Value, tiler: *const mlir.Value, num_warps: ?*const mlir.Attribute, init_boundary_check: ?*const mlir.Attribute, cache_modifier: ?*const mlir.Attribute, atomic_barrier: ?*const mlir.Attribute, internal_type: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (num_warps) |present| attributes.appendAssumeCapacity(.named(ctx, "numWarps", present));
    if (init_boundary_check) |present| attributes.appendAssumeCapacity(.named(ctx, "initBoundaryCheck", present));
    if (cache_modifier) |present| attributes.appendAssumeCapacity(.named(ctx, "cacheModifier", present));
    if (atomic_barrier) |present| attributes.appendAssumeCapacity(.named(ctx, "atomicBarrier", present));
    if (internal_type) |present| attributes.appendAssumeCapacity(.named(ctx, "internalType", present));
    return make(ctx, "fly_rocdl.make_tiled_tdm_load_atom", .{
        .operands = .{ .flat = &.{ tensor, smem_layout, tiler } },
        .attributes = attributes.constSlice(),
        .result_type_inference = true,
        .location = location,
    });
}

/// `fly_rocdl.make_tiled_tdm_store_atom`. Build a CDNA5 TDM store atom and the coordinate tensor that addresses it. Result types are inferred.
pub fn make_tiled_tdm_store_atom(ctx: *mlir.Context, tensor: *const mlir.Value, smem_layout: *const mlir.Value, tiler: *const mlir.Value, num_warps: ?*const mlir.Attribute, init_boundary_check: ?*const mlir.Attribute, cache_modifier: ?*const mlir.Attribute, atomic_barrier: ?*const mlir.Attribute, internal_type: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    if (num_warps) |present| attributes.appendAssumeCapacity(.named(ctx, "numWarps", present));
    if (init_boundary_check) |present| attributes.appendAssumeCapacity(.named(ctx, "initBoundaryCheck", present));
    if (cache_modifier) |present| attributes.appendAssumeCapacity(.named(ctx, "cacheModifier", present));
    if (atomic_barrier) |present| attributes.appendAssumeCapacity(.named(ctx, "atomicBarrier", present));
    if (internal_type) |present| attributes.appendAssumeCapacity(.named(ctx, "internalType", present));
    return make(ctx, "fly_rocdl.make_tiled_tdm_store_atom", .{
        .operands = .{ .flat = &.{ tensor, smem_layout, tiler } },
        .attributes = attributes.constSlice(),
        .result_type_inference = true,
        .location = location,
    });
}

test {
    std.testing.refAllDecls(@This());
}
