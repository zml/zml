//! `ttcore` operation builders, generated from tt-lang's `include/ttlang/Dialect/TTCore/IR/TTCoreOps.td` at 9b7ad24d1bd6a89dc1f8f885686bc4db86ac7244; do not edit, see REGENERATING.md.
//!
//! Builders take the operands, then the result types (only for operations
//! that do not infer them), then the regions' entry blocks, then the attributes,
//! then the location. Parameters are the ODS names in snake case, with a trailing
//! underscore where needed. Pass `null` for absent optional operands or
//! attributes and `&.{}` for empty variadic ones.

const std = @import("std");

const mlir = @import("mlir");
const stdx = @import("stdx");

const helpers = @import("ttkernel.zig");
const make = helpers.make;

/// Every operation bound here, for the registration test.
pub const names = [_][]const u8{
    "ttcore.composite",
    "ttcore.cpu_module",
    "ttcore.device",
    "ttcore.device_module",
    "ttcore.get_global",
    "ttcore.get_key_value",
    "ttcore.get_tuple_element",
    "ttcore.global",
    "ttcore.load_cached",
    "ttcore.optimization_barrier",
    "ttcore.set_key_value",
    "ttcore.tuple",
};

/// `ttcore.composite`. A named composite operation with an inlineable decomposition. Result types are explicit.
pub fn composite(ctx: *mlir.Context, inputs: []const *const mlir.Value, results_types: []const *const mlir.Type, composite_name: *const mlir.Attribute, decomposition: *const mlir.Attribute, composite_attributes: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendSliceAssumeCapacity(inputs);
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "composite_name", composite_name));
    attrs.appendAssumeCapacity(.named(ctx, "decomposition", decomposition));
    if (composite_attributes) |present| attrs.appendAssumeCapacity(.named(ctx, "composite_attributes", present));
    return make(ctx, "ttcore.composite", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = results_types },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttcore.cpu_module`. Module-wrapper operation for CPU ops. No results.
pub fn cpu_module(ctx: *mlir.Context, body_region_block: *mlir.Block, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttcore.cpu_module", .{
        .blocks = &.{body_region_block},
        .location = location,
    });
}

/// `ttcore.device`. Named device. No results.
pub fn device(ctx: *mlir.Context, sym_name: *const mlir.Attribute, sym_visibility: ?*const mlir.Attribute, device_attr: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "sym_name", sym_name));
    if (sym_visibility) |present| attrs.appendAssumeCapacity(.named(ctx, "sym_visibility", present));
    attrs.appendAssumeCapacity(.named(ctx, "device_attr", device_attr));
    return make(ctx, "ttcore.device", .{
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttcore.device_module`. Module-wrapper operation for device ops. No results.
pub fn device_module(ctx: *mlir.Context, body_region_block: *mlir.Block, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttcore.device_module", .{
        .blocks = &.{body_region_block},
        .location = location,
    });
}

/// `ttcore.get_global`. Named global. Result types are explicit.
pub fn get_global(ctx: *mlir.Context, result_type: *const mlir.Type, sym_name: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttcore.get_global", .{
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{.named(ctx, "sym_name", sym_name)},
        .location = location,
    });
}

/// `ttcore.get_key_value`. Get dictionary value. Result types are explicit.
pub fn get_key_value(ctx: *mlir.Context, dict: *const mlir.Value, results_types: []const *const mlir.Type, key: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttcore.get_key_value", .{
        .operands = .{ .flat = &.{dict} },
        .results = .{ .flat = results_types },
        .attributes = &.{.named(ctx, "key", key)},
        .location = location,
    });
}

/// `ttcore.get_tuple_element`. GetTupleElement operation. Result types are inferred.
pub fn get_tuple_element(ctx: *mlir.Context, operand: *const mlir.Value, index: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttcore.get_tuple_element", .{
        .operands = .{ .flat = &.{operand} },
        .result_type_inference = true,
        .attributes = &.{.named(ctx, "index", index)},
        .location = location,
    });
}

/// `ttcore.global`. Named global. No results.
pub fn global(ctx: *mlir.Context, sym_name: *const mlir.Attribute, sym_visibility: ?*const mlir.Attribute, type_: *const mlir.Attribute, index: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "sym_name", sym_name));
    if (sym_visibility) |present| attrs.appendAssumeCapacity(.named(ctx, "sym_visibility", present));
    attrs.appendAssumeCapacity(.named(ctx, "type", type_));
    if (index) |present| attrs.appendAssumeCapacity(.named(ctx, "index", present));
    return make(ctx, "ttcore.global", .{
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttcore.load_cached`. Load cached results from a previously computed function. Result types are explicit.
pub fn load_cached(ctx: *mlir.Context, inputs: []const *const mlir.Value, results_types: []const *const mlir.Type, callee: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendSliceAssumeCapacity(inputs);
    return make(ctx, "ttcore.load_cached", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = results_types },
        .attributes = &.{.named(ctx, "callee", callee)},
        .location = location,
    });
}

/// `ttcore.optimization_barrier`. Optimization barrier operation. Result types are explicit.
pub fn optimization_barrier(ctx: *mlir.Context, inputs: []const *const mlir.Value, results_types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendSliceAssumeCapacity(inputs);
    return make(ctx, "ttcore.optimization_barrier", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = results_types },
        .location = location,
    });
}

/// `ttcore.set_key_value`. Set or update dictionary value. No results.
pub fn set_key_value(ctx: *mlir.Context, dict: *const mlir.Value, values: []const *const mlir.Value, key: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(dict);
    operands.appendSliceAssumeCapacity(values);
    return make(ctx, "ttcore.set_key_value", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = &.{.named(ctx, "key", key)},
        .location = location,
    });
}

/// `ttcore.tuple`. Tuple operation. Result types are inferred.
pub fn tuple(ctx: *mlir.Context, operands_: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendSliceAssumeCapacity(operands_);
    return make(ctx, "ttcore.tuple", .{
        .operands = .{ .flat = operands.constSlice() },
        .result_type_inference = true,
        .location = location,
    });
}

/// Results of each builder's operation; -1 when variadic or optional.
pub const num_results = struct {
    pub const composite: comptime_int = -1;
    pub const cpu_module: comptime_int = 0;
    pub const device: comptime_int = 0;
    pub const device_module: comptime_int = 0;
    pub const get_global: comptime_int = 1;
    pub const get_key_value: comptime_int = -1;
    pub const get_tuple_element: comptime_int = 1;
    pub const global: comptime_int = 0;
    pub const load_cached: comptime_int = -1;
    pub const optimization_barrier: comptime_int = -1;
    pub const set_key_value: comptime_int = 0;
    pub const tuple: comptime_int = 1;
};
