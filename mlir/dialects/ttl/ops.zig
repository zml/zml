//! `ttl` operation builders, generated from tt-lang's `include/ttlang/Dialect/TTL/IR/TTLOps.td` at 9b7ad24d1bd6a89dc1f8f885686bc4db86ac7244; do not edit, see REGENERATING.md.
//!
//! Builders take the operands, then the result types (only for operations
//! that do not infer them), then the regions' entry blocks, then the attributes,
//! then the location. Parameters are the ODS names in snake case, with a trailing
//! underscore where needed. Pass `null` for absent optional operands or
//! attributes and `&.{}` for empty variadic ones.

const std = @import("std");

const mlir = @import("mlir");
const stdx = @import("stdx");

const helpers = @import("ttl.zig");
const make = helpers.make;

/// Every operation bound here, for the registration test.
pub const names = [_][]const u8{
    "ttl.abs",
    "ttl.accumulation_scope",
    "ttl.acos",
    "ttl.add",
    "ttl.asin",
    "ttl.atan",
    "ttl.attach_cb",
    "ttl.bind_cb",
    "ttl.block.broadcast",
    "ttl.cb_pop",
    "ttl.cb_push",
    "ttl.cb_reserve",
    "ttl.cb_wait",
    "ttl.ceil",
    "ttl.compute",
    "ttl.copy",
    "ttl.copy_dst",
    "ttl.copy_tile",
    "ttl.core_x",
    "ttl.core_y",
    "ttl.cos",
    "ttl.create_pipe",
    "ttl.current_device_index",
    "ttl.dfb_reconfiguration",
    "ttl.div",
    "ttl.dprint",
    "ttl.dst_index",
    "ttl.dst_section",
    "ttl.eq",
    "ttl.exp",
    "ttl.exp2",
    "ttl.expm1",
    "ttl.fill",
    "ttl.floor",
    "ttl.frac",
    "ttl.gelu",
    "ttl.get_dfb_id",
    "ttl.gt",
    "ttl.hardsigmoid",
    "ttl.if_dst",
    "ttl.if_src",
    "ttl.init_sfpu",
    "ttl.is_active",
    "ttl.is_device",
    "ttl.is_device_in_range",
    "ttl.is_dst",
    "ttl.is_src",
    "ttl.iter_index",
    "ttl.log",
    "ttl.lt",
    "ttl.matmul",
    "ttl.max",
    "ttl.min",
    "ttl.mul",
    "ttl.mul_unary_const",
    "ttl.ne",
    "ttl.neg",
    "ttl.opaque_call",
    "ttl.pipe_transfer.create",
    "ttl.pipe_transfer.post",
    "ttl.pipe_transfer.send",
    "ttl.pipe_transfer.wait",
    "ttl.pipe_transfer.wait_any",
    "ttl.pipenet_destination_count",
    "ttl.pipenet_foreach_dst",
    "ttl.pipenet_foreach_src",
    "ttl.pipenet_scope",
    "ttl.raw_addr",
    "ttl.raw_element_read",
    "ttl.raw_element_write",
    "ttl.read_index",
    "ttl.ready_receive_index",
    "ttl.recip",
    "ttl.reduce",
    "ttl.relu",
    "ttl.reset_all_dfbs",
    "ttl.reset_dfbs",
    "ttl.rsqrt",
    "ttl.select_pipe_dst",
    "ttl.select_pipe_src",
    "ttl.selected_pipe_destination_coordinates",
    "ttl.selected_pipe_destination_device_index",
    "ttl.selected_pipe_source_coordinates",
    "ttl.selected_pipe_source_device_index",
    "ttl.sigmoid",
    "ttl.sign",
    "ttl.signbit",
    "ttl.signpost",
    "ttl.silu",
    "ttl.sin",
    "ttl.softsign",
    "ttl.sqrt",
    "ttl.square",
    "ttl.store",
    "ttl.sub",
    "ttl.tan",
    "ttl.tanh",
    "ttl.tensor_slice",
    "ttl.tile_abs",
    "ttl.tile_accumulate",
    "ttl.tile_acos",
    "ttl.tile_add",
    "ttl.tile_asin",
    "ttl.tile_atan",
    "ttl.tile_bcast",
    "ttl.tile_ceil",
    "ttl.tile_cos",
    "ttl.tile_div",
    "ttl.tile_eq",
    "ttl.tile_exp",
    "ttl.tile_exp2",
    "ttl.tile_expm1",
    "ttl.tile_fill",
    "ttl.tile_floor",
    "ttl.tile_frac",
    "ttl.tile_gelu",
    "ttl.tile_gt",
    "ttl.tile_hardsigmoid",
    "ttl.tile_log",
    "ttl.tile_lt",
    "ttl.tile_matmul_block",
    "ttl.tile_max",
    "ttl.tile_min",
    "ttl.tile_mul",
    "ttl.tile_mul_unary_const",
    "ttl.tile_ne",
    "ttl.tile_neg",
    "ttl.tile_recip",
    "ttl.tile_reduce",
    "ttl.tile_regs_acquire",
    "ttl.tile_regs_commit",
    "ttl.tile_regs_release",
    "ttl.tile_regs_wait",
    "ttl.tile_relu",
    "ttl.tile_row_normalization_block",
    "ttl.tile_rsqrt",
    "ttl.tile_sigmoid",
    "ttl.tile_sign",
    "ttl.tile_signbit",
    "ttl.tile_silu",
    "ttl.tile_sin",
    "ttl.tile_softsign",
    "ttl.tile_sqrt",
    "ttl.tile_square",
    "ttl.tile_store",
    "ttl.tile_sub",
    "ttl.tile_tan",
    "ttl.tile_tanh",
    "ttl.tile_transpose",
    "ttl.tile_trunc",
    "ttl.tile_typecast",
    "ttl.transpose",
    "ttl.trunc",
    "ttl.typecast",
    "ttl.wait",
    "ttl.wait_any",
    "ttl.yield",
};

/// `ttl.abs`. Unary elementwise operation. Result types are explicit.
pub fn abs(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.abs", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.accumulation_scope`. Structured accumulation scope. No results.
pub fn accumulation_scope(ctx: *mlir.Context, outputs: []const *const mlir.Value, inits: []const *const mlir.Value, body_block: *mlir.Block, initial_modes: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.accumulation_scope", .{
        .operands = .{ .variadic = &.{ outputs, inits } },
        .blocks = &.{body_block},
        .attributes = &.{.named(ctx, "initial_modes", initial_modes)},
        .location = location,
    });
}

/// `ttl.acos`. Unary elementwise operation. Result types are explicit.
pub fn acos(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.acos", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.add`. Binary elementwise operation. Result types are explicit.
pub fn add(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.add", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.asin`. Unary elementwise operation. Result types are explicit.
pub fn asin(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.asin", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.atan`. Unary elementwise operation. Result types are explicit.
pub fn atan(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.atan", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.attach_cb`. Associate a tensor SSA value with a circular buffer. Result types are explicit.
pub fn attach_cb(ctx: *mlir.Context, tensor: *const mlir.Value, cb: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.attach_cb", .{
        .operands = .{ .flat = &.{ tensor, cb } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.bind_cb`. Bind to a hardware circular buffer slot. Result types are explicit.
pub fn bind_cb(ctx: *mlir.Context, result_type: *const mlir.Type, cb_index: *const mlir.Attribute, block_count: ?*const mlir.Attribute, dfb_id: ?*const mlir.Attribute, tensor_backing: ?*const mlir.Attribute, allocation_group: ?*const mlir.Attribute, address_scope: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 6) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "cb_index", cb_index));
    if (block_count) |present| attrs.appendAssumeCapacity(.named(ctx, "block_count", present));
    if (dfb_id) |present| attrs.appendAssumeCapacity(.named(ctx, "dfb_id", present));
    if (tensor_backing) |present| attrs.appendAssumeCapacity(.named(ctx, "tensor_backing", present));
    if (allocation_group) |present| attrs.appendAssumeCapacity(.named(ctx, "allocation_group", present));
    if (address_scope) |present| attrs.appendAssumeCapacity(.named(ctx, "address_scope", present));
    return make(ctx, "ttl.bind_cb", .{
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.block.broadcast`. Broadcast a block over specified dimensions to a target shape. Result types are explicit.
pub fn block_broadcast(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, dims: *const mlir.Attribute, shape: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.block.broadcast", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{ .named(ctx, "dims", dims), .named(ctx, "shape", shape) },
        .location = location,
    });
}

/// `ttl.cb_pop`. Release pages from the circular buffer after reading. No results.
pub fn cb_pop(ctx: *mlir.Context, cb: *const mlir.Value, num_tiles: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (num_tiles) |present| attrs.appendAssumeCapacity(.named(ctx, "num_tiles", present));
    return make(ctx, "ttl.cb_pop", .{
        .operands = .{ .flat = &.{cb} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.cb_push`. Signal that tiles have been written to the dataflow buffer. No results.
pub fn cb_push(ctx: *mlir.Context, cb: *const mlir.Value, num_tiles: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (num_tiles) |present| attrs.appendAssumeCapacity(.named(ctx, "num_tiles", present));
    return make(ctx, "ttl.cb_push", .{
        .operands = .{ .flat = &.{cb} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.cb_reserve`. Reserve tiles in a dataflow buffer for writing. Result types are explicit.
pub fn cb_reserve(ctx: *mlir.Context, cb: *const mlir.Value, result_type: *const mlir.Type, num_tiles: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (num_tiles) |present| attrs.appendAssumeCapacity(.named(ctx, "num_tiles", present));
    return make(ctx, "ttl.cb_reserve", .{
        .operands = .{ .flat = &.{cb} },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.cb_wait`. Wait for pages to be available in the circular buffer. Result types are explicit.
pub fn cb_wait(ctx: *mlir.Context, cb: *const mlir.Value, result_type: *const mlir.Type, num_tiles: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (num_tiles) |present| attrs.appendAssumeCapacity(.named(ctx, "num_tiles", present));
    return make(ctx, "ttl.cb_wait", .{
        .operands = .{ .flat = &.{cb} },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.ceil`. Unary elementwise operation. Result types are explicit.
pub fn ceil(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.ceil", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.compute`. Structured tile-level compute operation. Result types are explicit.
pub fn compute(ctx: *mlir.Context, inputs: []const *const mlir.Value, outputs: []const *const mlir.Value, results_types: []const *const mlir.Type, body_block: *mlir.Block, indexing_maps: *const mlir.Attribute, iterator_types: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.compute", .{
        .operands = .{ .variadic = &.{ inputs, outputs } },
        .results = .{ .flat = results_types },
        .blocks = &.{body_block},
        .attributes = &.{ .named(ctx, "indexing_maps", indexing_maps), .named(ctx, "iterator_types", iterator_types) },
        .location = location,
    });
}

/// `ttl.copy`. Asynchronous copy involving a tensor, DFB, or pipe. Result types are explicit.
pub fn copy(ctx: *mlir.Context, src: *const mlir.Value, dst: *const mlir.Value, xf_type: *const mlir.Type, byte_count: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (byte_count) |present| attrs.appendAssumeCapacity(.named(ctx, "byte_count", present));
    return make(ctx, "ttl.copy", .{
        .operands = .{ .flat = &.{ src, dst } },
        .results = .{ .flat = &.{xf_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.copy_dst`. Copy a tile from one DST register to another. Result types are explicit.
pub fn copy_dst(ctx: *mlir.Context, src_tile: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.copy_dst", .{
        .operands = .{ .flat = &.{ src_tile, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.copy_tile`. Copy a tile from a circular buffer into a DST register. Result types are explicit.
pub fn copy_tile(ctx: *mlir.Context, src: *const mlir.Value, src_indices: []const *const mlir.Value, dst_index_: *const mlir.Value, dst_tile_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(src);
    operands.appendSliceAssumeCapacity(src_indices);
    operands.appendAssumeCapacity(dst_index_);
    return make(ctx, "ttl.copy_tile", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{ helpers.parseType(ctx, "!ttl.dst"), dst_tile_type } },
        .location = location,
    });
}

/// `ttl.core_x`. Get the X coordinate of the current core. Result types are fixed.
pub fn core_x(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.core_x", .{
        .results = .{ .flat = &.{mlir.Type.index(ctx)} },
        .location = location,
    });
}

/// `ttl.core_y`. Get the Y coordinate of the current core. Result types are fixed.
pub fn core_y(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.core_y", .{
        .results = .{ .flat = &.{mlir.Type.index(ctx)} },
        .location = location,
    });
}

/// `ttl.cos`. Unary elementwise operation. Result types are explicit.
pub fn cos(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.cos", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.create_pipe`. Create a pipe for core-to-core communication. Result types are explicit.
pub fn create_pipe(ctx: *mlir.Context, result_type: *const mlir.Type, src_x: *const mlir.Attribute, src_y: *const mlir.Attribute, dst_start_x: *const mlir.Attribute, dst_start_y: *const mlir.Attribute, dst_end_x: *const mlir.Attribute, dst_end_y: *const mlir.Attribute, pipe_net_id: *const mlir.Attribute, pipe_net_name: ?*const mlir.Attribute, is_collective: ?*const mlir.Attribute, device_transfer: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 10) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "srcX", src_x));
    attrs.appendAssumeCapacity(.named(ctx, "srcY", src_y));
    attrs.appendAssumeCapacity(.named(ctx, "dstStartX", dst_start_x));
    attrs.appendAssumeCapacity(.named(ctx, "dstStartY", dst_start_y));
    attrs.appendAssumeCapacity(.named(ctx, "dstEndX", dst_end_x));
    attrs.appendAssumeCapacity(.named(ctx, "dstEndY", dst_end_y));
    attrs.appendAssumeCapacity(.named(ctx, "pipeNetId", pipe_net_id));
    if (pipe_net_name) |present| attrs.appendAssumeCapacity(.named(ctx, "pipeNetName", present));
    if (is_collective) |present| attrs.appendAssumeCapacity(.named(ctx, "isCollective", present));
    if (device_transfer) |present| attrs.appendAssumeCapacity(.named(ctx, "deviceTransfer", present));
    return make(ctx, "ttl.create_pipe", .{
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.current_device_index`. Get the current logical device index. Result types are fixed.
pub fn current_device_index(ctx: *mlir.Context, domain: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.current_device_index", .{
        .results = .{ .flat = &.{mlir.Type.index(ctx)} },
        .attributes = &.{.named(ctx, "domain", domain)},
        .location = location,
    });
}

/// `ttl.dfb_reconfiguration`. Enter the next worker-local DFB configuration epoch. No results.
pub fn dfb_reconfiguration(ctx: *mlir.Context, boundary: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.dfb_reconfiguration", .{
        .attributes = &.{.named(ctx, "boundary", boundary)},
        .location = location,
    });
}

/// `ttl.div`. Binary elementwise operation. Result types are explicit.
pub fn div(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.div", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.dprint`. Debug print from device kernel. No results.
pub fn dprint(ctx: *mlir.Context, argv: []const *const mlir.Value, fmt: *const mlir.Attribute, mode: *const mlir.Attribute, thread: ?*const mlir.Attribute, num_pages: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendSliceAssumeCapacity(argv);
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "fmt", fmt));
    attrs.appendAssumeCapacity(.named(ctx, "mode", mode));
    if (thread) |present| attrs.appendAssumeCapacity(.named(ctx, "thread", present));
    if (num_pages) |present| attrs.appendAssumeCapacity(.named(ctx, "num_pages", present));
    return make(ctx, "ttl.dprint", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.dst_index`. Index a tile value resident in a specific DST slot. Result types are explicit.
pub fn dst_index(ctx: *mlir.Context, source: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.dst_index", .{
        .operands = .{ .flat = &.{ source, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.dst_section`. DST register acquisition scope. No results.
pub fn dst_section(ctx: *mlir.Context, body_block: *mlir.Block, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.dst_section", .{
        .blocks = &.{body_block},
        .location = location,
    });
}

/// `ttl.eq`. Binary elementwise operation. Result types are explicit.
pub fn eq(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.eq", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.exp`. Unary elementwise operation. Result types are explicit.
pub fn exp(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, approx: ?*const mlir.Attribute, scale: ?*const mlir.Attribute, input_clamping: ?*const mlir.Attribute, iterations: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (approx) |present| attrs.appendAssumeCapacity(.named(ctx, "approx", present));
    if (scale) |present| attrs.appendAssumeCapacity(.named(ctx, "scale", present));
    if (input_clamping) |present| attrs.appendAssumeCapacity(.named(ctx, "input_clamping", present));
    if (iterations) |present| attrs.appendAssumeCapacity(.named(ctx, "iterations", present));
    return make(ctx, "ttl.exp", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.exp2`. Unary elementwise operation. Result types are explicit.
pub fn exp2(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.exp2", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.expm1`. Unary elementwise operation. Result types are explicit.
pub fn expm1(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.expm1", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.fill`. Fill a tensor with a constant value. Result types are explicit.
pub fn fill(ctx: *mlir.Context, result_type: *const mlir.Type, value: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.fill", .{
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{.named(ctx, "value", value)},
        .location = location,
    });
}

/// `ttl.floor`. Unary elementwise operation. Result types are explicit.
pub fn floor(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.floor", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.frac`. Unary elementwise operation. Result types are explicit.
pub fn frac(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.frac", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.gelu`. Unary elementwise operation. Result types are explicit.
pub fn gelu(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.gelu", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.get_dfb_id`. Get the integer index for a dataflow buffer. Result types are fixed.
pub fn get_dfb_id(ctx: *mlir.Context, dfb: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.get_dfb_id", .{
        .operands = .{ .flat = &.{dfb} },
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i32)} },
        .location = location,
    });
}

/// `ttl.gt`. Binary elementwise operation. Result types are explicit.
pub fn gt(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.gt", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.hardsigmoid`. Unary elementwise operation. Result types are explicit.
pub fn hardsigmoid(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.hardsigmoid", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.if_dst`. Execute region if current core is a destination of a pipe. No results.
pub fn if_dst(ctx: *mlir.Context, pipe: *const mlir.Value, body_block: *mlir.Block, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.if_dst", .{
        .operands = .{ .flat = &.{pipe} },
        .blocks = &.{body_block},
        .location = location,
    });
}

/// `ttl.if_src`. Execute region if current core is the source of a pipe. No results.
pub fn if_src(ctx: *mlir.Context, pipe: *const mlir.Value, body_block: *mlir.Block, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.if_src", .{
        .operands = .{ .flat = &.{pipe} },
        .blocks = &.{body_block},
        .location = location,
    });
}

/// `ttl.init_sfpu`. Initialize SFPU unpacker/packer for CB data formats. No results.
pub fn init_sfpu(ctx: *mlir.Context, icb: *const mlir.Value, ocb: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.init_sfpu", .{
        .operands = .{ .flat = &.{ icb, ocb } },
        .location = location,
    });
}

/// `ttl.is_active`. Return whether the current node participates in a PipeNet. Result types are fixed.
pub fn is_active(ctx: *mlir.Context, pipe_net_id: *const mlir.Attribute, records: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "pipe_net_id", pipe_net_id));
    if (records) |present| attrs.appendAssumeCapacity(.named(ctx, "records", present));
    return make(ctx, "ttl.is_active", .{
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i1)} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.is_device`. Test whether the kernel executes on a logical device. Result types are fixed.
pub fn is_device(ctx: *mlir.Context, domain: *const mlir.Attribute, device: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.is_device", .{
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i1)} },
        .attributes = &.{ .named(ctx, "domain", domain), .named(ctx, "device", device) },
        .location = location,
    });
}

/// `ttl.is_device_in_range`. Test whether the kernel executes in a logical device range. Result types are fixed.
pub fn is_device_in_range(ctx: *mlir.Context, domain: *const mlir.Attribute, range: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.is_device_in_range", .{
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i1)} },
        .attributes = &.{ .named(ctx, "domain", domain), .named(ctx, "range", range) },
        .location = location,
    });
}

/// `ttl.is_dst`. Return whether the current node is a PipeNet destination. Result types are fixed.
pub fn is_dst(ctx: *mlir.Context, pipe_net_id: *const mlir.Attribute, records: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "pipe_net_id", pipe_net_id));
    if (records) |present| attrs.appendAssumeCapacity(.named(ctx, "records", present));
    return make(ctx, "ttl.is_dst", .{
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i1)} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.is_src`. Return whether the current node is a PipeNet source. Result types are fixed.
pub fn is_src(ctx: *mlir.Context, pipe_net_id: *const mlir.Attribute, records: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "pipe_net_id", pipe_net_id));
    if (records) |present| attrs.appendAssumeCapacity(.named(ctx, "records", present));
    return make(ctx, "ttl.is_src", .{
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i1)} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.iter_index`. Get the current iteration index for a dimension. Result types are fixed.
pub fn iter_index(ctx: *mlir.Context, dim: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.iter_index", .{
        .results = .{ .flat = &.{mlir.Type.index(ctx)} },
        .attributes = &.{.named(ctx, "dim", dim)},
        .location = location,
    });
}

/// `ttl.log`. Unary elementwise operation. Result types are explicit.
pub fn log(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.log", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.lt`. Binary elementwise operation. Result types are explicit.
pub fn lt(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.lt", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.matmul`. Matrix multiply two tensors of tiles. Result types are explicit.
pub fn matmul(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, transpose_rhs: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (transpose_rhs) |present| attrs.appendAssumeCapacity(.named(ctx, "transpose_rhs", present));
    return make(ctx, "ttl.matmul", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.max`. Binary elementwise operation. Result types are explicit.
pub fn max(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.max", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.min`. Binary elementwise operation. Result types are explicit.
pub fn min(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.min", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.mul`. Binary elementwise operation. Result types are explicit.
pub fn mul(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.mul", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.mul_unary_const`. Multiply a tensor of tiles by a compile-time scalar constant. Result types are explicit.
pub fn mul_unary_const(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, value: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.mul_unary_const", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{.named(ctx, "value", value)},
        .location = location,
    });
}

/// `ttl.ne`. Binary elementwise operation. Result types are explicit.
pub fn ne(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.ne", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.neg`. Unary elementwise operation. Result types are explicit.
pub fn neg(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.neg", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.opaque_call`. Call an external C/C++ function by name. Result types are explicit.
pub fn opaque_call(ctx: *mlir.Context, arg_operands: []const *const mlir.Value, template_dfb_operands: []const *const mlir.Value, dependency_dfb_operands: []const *const mlir.Value, result_types: []const *const mlir.Type, callee: *const mlir.Attribute, header: *const mlir.Attribute, template_args: ?*const mlir.Attribute, unsigned_arg_indices: ?*const mlir.Attribute, fabric_manager_effects: ?*const mlir.Attribute, dfb_effects: ?*const mlir.Attribute, dfb_accesses: ?*const mlir.Attribute, unknown_dfb_access: ?*const mlir.Attribute, condition_result: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 9) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "callee", callee));
    attrs.appendAssumeCapacity(.named(ctx, "header", header));
    if (template_args) |present| attrs.appendAssumeCapacity(.named(ctx, "template_args", present));
    if (unsigned_arg_indices) |present| attrs.appendAssumeCapacity(.named(ctx, "unsigned_arg_indices", present));
    if (fabric_manager_effects) |present| attrs.appendAssumeCapacity(.named(ctx, "fabric_manager_effects", present));
    if (dfb_effects) |present| attrs.appendAssumeCapacity(.named(ctx, "dfb_effects", present));
    if (dfb_accesses) |present| attrs.appendAssumeCapacity(.named(ctx, "dfb_accesses", present));
    if (unknown_dfb_access) |present| attrs.appendAssumeCapacity(.named(ctx, "unknown_dfb_access", present));
    if (condition_result) |present| attrs.appendAssumeCapacity(.named(ctx, "condition_result", present));
    return make(ctx, "ttl.opaque_call", .{
        .operands = .{ .variadic = &.{ arg_operands, template_dfb_operands, dependency_dfb_operands } },
        .results = .{ .flat = result_types },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.pipe_transfer.create`. Create a pipe transfer reference. Result types are fixed.
pub fn pipe_transfer_create(ctx: *mlir.Context, pipe: *const mlir.Value, kind: *const mlir.Attribute, device_transfer: ?*const mlir.Attribute, block_span: ?*const mlir.Attribute, destination_group_depth: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (device_transfer) |present| attrs.appendAssumeCapacity(.named(ctx, "deviceTransfer", present));
    if (block_span) |present| attrs.appendAssumeCapacity(.named(ctx, "block_span", present));
    if (destination_group_depth) |present| attrs.appendAssumeCapacity(.named(ctx, "destination_group_depth", present));
    return make(ctx, "ttl.pipe_transfer.create", .{
        .operands = .{ .flat = &.{pipe} },
        .results = .{ .flat = &.{helpers.parseType(ctx, "!ttl.pipe_transfer")} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.pipe_transfer.post`. Post a pipe receive destination. Result types are explicit.
pub fn pipe_transfer_post(ctx: *mlir.Context, transfer: *const mlir.Value, dst: *const mlir.Value, token_type: *const mlir.Type, byte_count: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (byte_count) |present| attrs.appendAssumeCapacity(.named(ctx, "byte_count", present));
    return make(ctx, "ttl.pipe_transfer.post", .{
        .operands = .{ .flat = &.{ transfer, dst } },
        .results = .{ .flat = &.{token_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.pipe_transfer.send`. Send a pipe transfer payload. Result types are explicit.
pub fn pipe_transfer_send(ctx: *mlir.Context, transfer: *const mlir.Value, src: *const mlir.Value, xf_type: *const mlir.Type, byte_count: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (byte_count) |present| attrs.appendAssumeCapacity(.named(ctx, "byte_count", present));
    return make(ctx, "ttl.pipe_transfer.send", .{
        .operands = .{ .flat = &.{ transfer, src } },
        .results = .{ .flat = &.{xf_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.pipe_transfer.wait`. Wait for pipe receive completion. No results.
pub fn pipe_transfer_wait(ctx: *mlir.Context, token: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.pipe_transfer.wait", .{
        .operands = .{ .flat = &.{token} },
        .location = location,
    });
}

/// `ttl.pipe_transfer.wait_any`. Wait for one pipe receive token. Result types are fixed.
pub fn pipe_transfer_wait_any(ctx: *mlir.Context, tokens: []const *const mlir.Value, start: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendSliceAssumeCapacity(tokens);
    operands.appendAssumeCapacity(start);
    return make(ctx, "ttl.pipe_transfer.wait_any", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{helpers.parseType(ctx, "!ttl.ready_receive")} },
        .location = location,
    });
}

/// `ttl.pipenet_destination_count`. Count PipeNet records targeting the current node. Result types are fixed.
pub fn pipenet_destination_count(ctx: *mlir.Context, pipe_net_id: *const mlir.Attribute, records: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.pipenet_destination_count", .{
        .results = .{ .flat = &.{mlir.Type.index(ctx)} },
        .attributes = &.{ .named(ctx, "pipe_net_id", pipe_net_id), .named(ctx, "records", records) },
        .location = location,
    });
}

/// `ttl.pipenet_foreach_dst`. Iterate over destination pipe records in a PipeNet. No results.
pub fn pipenet_foreach_dst(ctx: *mlir.Context, body_block: *mlir.Block, records: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.pipenet_foreach_dst", .{
        .blocks = &.{body_block},
        .attributes = &.{.named(ctx, "records", records)},
        .location = location,
    });
}

/// `ttl.pipenet_foreach_src`. Iterate over source pipe records in a PipeNet. No results.
pub fn pipenet_foreach_src(ctx: *mlir.Context, body_block: *mlir.Block, records: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.pipenet_foreach_src", .{
        .blocks = &.{body_block},
        .attributes = &.{.named(ctx, "records", records)},
        .location = location,
    });
}

/// `ttl.pipenet_scope`. Mark a DFB region restricted to PipeNet endpoints. No results.
pub fn pipenet_scope(ctx: *mlir.Context, body_block: *mlir.Block, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.pipenet_scope", .{
        .blocks = &.{body_block},
        .location = location,
    });
}

/// `ttl.raw_addr`. Get raw i32 base address for a tensor argument. Result types are fixed.
pub fn raw_addr(ctx: *mlir.Context, tensor: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.raw_addr", .{
        .operands = .{ .flat = &.{tensor} },
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i32)} },
        .location = location,
    });
}

/// `ttl.raw_element_read`. Read a scalar element from a block at flat coordinates. Result types are explicit.
pub fn raw_element_read(ctx: *mlir.Context, block: *const mlir.Value, coords: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(block);
    operands.appendSliceAssumeCapacity(coords);
    return make(ctx, "ttl.raw_element_read", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.raw_element_write`. Write a scalar value to a block at flat coordinates. No results.
pub fn raw_element_write(ctx: *mlir.Context, block: *const mlir.Value, coords: []const *const mlir.Value, value: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(block);
    operands.appendSliceAssumeCapacity(coords);
    operands.appendAssumeCapacity(value);
    return make(ctx, "ttl.raw_element_write", .{
        .operands = .{ .flat = operands.constSlice() },
        .location = location,
    });
}

/// `ttl.read_index`. Read a nonnegative scalar element as an index. Result types are fixed.
pub fn read_index(ctx: *mlir.Context, block: *const mlir.Value, coords: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(block);
    operands.appendSliceAssumeCapacity(coords);
    return make(ctx, "ttl.read_index", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{mlir.Type.index(ctx)} },
        .location = location,
    });
}

/// `ttl.ready_receive_index`. Return the selected receive-request index. Result types are fixed.
pub fn ready_receive_index(ctx: *mlir.Context, ready: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.ready_receive_index", .{
        .operands = .{ .flat = &.{ready} },
        .results = .{ .flat = &.{mlir.Type.index(ctx)} },
        .location = location,
    });
}

/// `ttl.recip`. Unary elementwise operation. Result types are explicit.
pub fn recip(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.recip", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.reduce`. Reduce a tensor along specified dimensions with a scaling factor. Result types are explicit.
pub fn reduce(ctx: *mlir.Context, input: *const mlir.Value, scaler: *const mlir.Value, result_type: *const mlir.Type, reduce_type: *const mlir.Attribute, dims: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.reduce", .{
        .operands = .{ .flat = &.{ input, scaler } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{ .named(ctx, "reduce_type", reduce_type), .named(ctx, "dims", dims) },
        .location = location,
    });
}

/// `ttl.relu`. Unary elementwise operation. Result types are explicit.
pub fn relu(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.relu", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.reset_all_dfbs`. Synchronize DFB owners and reset every local interface. No results.
pub fn reset_all_dfbs(ctx: *mlir.Context, preserved_dfbs: []const *const mlir.Value, reset: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendSliceAssumeCapacity(preserved_dfbs);
    return make(ctx, "ttl.reset_all_dfbs", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = &.{.named(ctx, "reset", reset)},
        .location = location,
    });
}

/// `ttl.reset_dfbs`. Synchronize DFB owners and reset selected interfaces. No results.
pub fn reset_dfbs(ctx: *mlir.Context, dfbs: []const *const mlir.Value, reset: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendSliceAssumeCapacity(dfbs);
    return make(ctx, "ttl.reset_dfbs", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = &.{.named(ctx, "reset", reset)},
        .location = location,
    });
}

/// `ttl.rsqrt`. Unary elementwise operation. Result types are explicit.
pub fn rsqrt(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.rsqrt", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.select_pipe_dst`. Represent a record selected by destination iteration. Result types are fixed.
pub fn select_pipe_dst(ctx: *mlir.Context, record_index: *const mlir.Value, src_x: *const mlir.Value, src_y: *const mlir.Value, dst_start_x: *const mlir.Value, dst_start_y: *const mlir.Value, dst_end_x: *const mlir.Value, dst_end_y: *const mlir.Value, num_dests: *const mlir.Value, src_in_dst_range: *const mlir.Value, source_device_index: *const mlir.Value, destination_device_index: *const mlir.Value, records: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.select_pipe_dst", .{
        .operands = .{ .flat = &.{ record_index, src_x, src_y, dst_start_x, dst_start_y, dst_end_x, dst_end_y, num_dests, src_in_dst_range, source_device_index, destination_device_index } },
        .results = .{ .flat = &.{helpers.parseType(ctx, "!ttl.selected_pipe_dst")} },
        .attributes = &.{.named(ctx, "records", records)},
        .location = location,
    });
}

/// `ttl.select_pipe_src`. Represent a record selected by source iteration. Result types are fixed.
pub fn select_pipe_src(ctx: *mlir.Context, record_index: *const mlir.Value, src_x: *const mlir.Value, src_y: *const mlir.Value, dst_start_x: *const mlir.Value, dst_start_y: *const mlir.Value, dst_end_x: *const mlir.Value, dst_end_y: *const mlir.Value, num_dests: *const mlir.Value, src_in_dst_range: *const mlir.Value, source_device_index: *const mlir.Value, destination_device_index: *const mlir.Value, records: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.select_pipe_src", .{
        .operands = .{ .flat = &.{ record_index, src_x, src_y, dst_start_x, dst_start_y, dst_end_x, dst_end_y, num_dests, src_in_dst_range, source_device_index, destination_device_index } },
        .results = .{ .flat = &.{helpers.parseType(ctx, "!ttl.selected_pipe_src")} },
        .attributes = &.{.named(ctx, "records", records)},
        .location = location,
    });
}

/// `ttl.selected_pipe_destination_coordinates`. Get the selected pipe record's destination node coordinates. Result types are fixed.
pub fn selected_pipe_destination_coordinates(ctx: *mlir.Context, pipe: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.selected_pipe_destination_coordinates", .{
        .operands = .{ .flat = &.{pipe} },
        .results = .{ .flat = &.{ mlir.Type.index(ctx), mlir.Type.index(ctx), mlir.Type.index(ctx), mlir.Type.index(ctx) } },
        .location = location,
    });
}

/// `ttl.selected_pipe_destination_device_index`. Get the selected pipe record's logical destination device index. Result types are fixed.
pub fn selected_pipe_destination_device_index(ctx: *mlir.Context, pipe: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.selected_pipe_destination_device_index", .{
        .operands = .{ .flat = &.{pipe} },
        .results = .{ .flat = &.{mlir.Type.index(ctx)} },
        .location = location,
    });
}

/// `ttl.selected_pipe_source_coordinates`. Get the selected pipe record's source node coordinates. Result types are fixed.
pub fn selected_pipe_source_coordinates(ctx: *mlir.Context, pipe: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.selected_pipe_source_coordinates", .{
        .operands = .{ .flat = &.{pipe} },
        .results = .{ .flat = &.{ mlir.Type.index(ctx), mlir.Type.index(ctx) } },
        .location = location,
    });
}

/// `ttl.selected_pipe_source_device_index`. Get the selected pipe record's logical source device index. Result types are fixed.
pub fn selected_pipe_source_device_index(ctx: *mlir.Context, pipe: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.selected_pipe_source_device_index", .{
        .operands = .{ .flat = &.{pipe} },
        .results = .{ .flat = &.{mlir.Type.index(ctx)} },
        .location = location,
    });
}

/// `ttl.sigmoid`. Unary elementwise operation. Result types are explicit.
pub fn sigmoid(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.sigmoid", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.sign`. Unary elementwise operation. Result types are explicit.
pub fn sign(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.sign", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.signbit`. Unary elementwise operation. Result types are explicit.
pub fn signbit(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.signbit", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.signpost`. Profiling marker for device tracing. No results.
pub fn signpost(ctx: *mlir.Context, name: *const mlir.Attribute, is_end: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "name", name));
    if (is_end) |present| attrs.appendAssumeCapacity(.named(ctx, "is_end", present));
    return make(ctx, "ttl.signpost", .{
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.silu`. Unary elementwise operation. Result types are explicit.
pub fn silu(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.silu", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.sin`. Unary elementwise operation. Result types are explicit.
pub fn sin(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.sin", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.softsign`. Unary elementwise operation. Result types are explicit.
pub fn softsign(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.softsign", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.sqrt`. Unary elementwise operation. Result types are explicit.
pub fn sqrt(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.sqrt", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.square`. Unary elementwise operation. Result types are explicit.
pub fn square(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.square", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.store`. Store a tensor to an acquired dataflow buffer view. No results.
pub fn store(ctx: *mlir.Context, tensor: *const mlir.Value, view: *const mlir.Value, accumulate: ?*const mlir.Attribute, row_prefix: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (accumulate) |present| attrs.appendAssumeCapacity(.named(ctx, "accumulate", present));
    if (row_prefix) |present| attrs.appendAssumeCapacity(.named(ctx, "row_prefix", present));
    return make(ctx, "ttl.store", .{
        .operands = .{ .flat = &.{ tensor, view } },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.sub`. Binary elementwise operation. Result types are explicit.
pub fn sub(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.sub", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tan`. Unary elementwise operation. Result types are explicit.
pub fn tan(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tan", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tanh`. Unary elementwise operation. Result types are explicit.
pub fn tanh(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tanh", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tensor_slice`. Create a view into a tensor at specific tile indices. Result types are explicit.
pub fn tensor_slice(ctx: *mlir.Context, tensor: *const mlir.Value, indices: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(tensor);
    operands.appendSliceAssumeCapacity(indices);
    return make(ctx, "ttl.tensor_slice", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_abs`. Unary tile operation. Result types are explicit.
pub fn tile_abs(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_abs", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_accumulate`. In-DST tile recurrence. Result types are explicit.
pub fn tile_accumulate(ctx: *mlir.Context, accumulator: *const mlir.Value, contribution: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, combiner: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_accumulate", .{
        .operands = .{ .flat = &.{ accumulator, contribution, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{.named(ctx, "combiner", combiner)},
        .location = location,
    });
}

/// `ttl.tile_acos`. Unary tile operation. Result types are explicit.
pub fn tile_acos(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_acos", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_add`. Binary tile operation. Result types are explicit.
pub fn tile_add(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_add", .{
        .operands = .{ .flat = &.{ lhs, rhs, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_asin`. Unary tile operation. Result types are explicit.
pub fn tile_asin(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_asin", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_atan`. Unary tile operation. Result types are explicit.
pub fn tile_atan(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_atan", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_bcast`. Tile-level broadcast operation. Result types are explicit.
pub fn tile_bcast(ctx: *mlir.Context, input: *const mlir.Value, output: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, bcast_type: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_bcast", .{
        .operands = .{ .flat = &.{ input, output, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{.named(ctx, "bcast_type", bcast_type)},
        .location = location,
    });
}

/// `ttl.tile_ceil`. Unary tile operation. Result types are explicit.
pub fn tile_ceil(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_ceil", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_cos`. Unary tile operation. Result types are explicit.
pub fn tile_cos(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_cos", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_div`. Binary tile operation. Result types are explicit.
pub fn tile_div(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_div", .{
        .operands = .{ .flat = &.{ lhs, rhs, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_eq`. Binary tile operation. Result types are explicit.
pub fn tile_eq(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_eq", .{
        .operands = .{ .flat = &.{ lhs, rhs, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_exp`. Unary tile operation. Result types are explicit.
pub fn tile_exp(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, approx: ?*const mlir.Attribute, scale: ?*const mlir.Attribute, input_clamping: ?*const mlir.Attribute, iterations: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (approx) |present| attrs.appendAssumeCapacity(.named(ctx, "approx", present));
    if (scale) |present| attrs.appendAssumeCapacity(.named(ctx, "scale", present));
    if (input_clamping) |present| attrs.appendAssumeCapacity(.named(ctx, "input_clamping", present));
    if (iterations) |present| attrs.appendAssumeCapacity(.named(ctx, "iterations", present));
    return make(ctx, "ttl.tile_exp", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.tile_exp2`. Unary tile operation. Result types are explicit.
pub fn tile_exp2(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_exp2", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_expm1`. Unary tile operation. Result types are explicit.
pub fn tile_expm1(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_expm1", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_fill`. Tile-level fill operation. Result types are explicit.
pub fn tile_fill(ctx: *mlir.Context, dst_index_: *const mlir.Value, result_type: *const mlir.Type, value: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_fill", .{
        .operands = .{ .flat = &.{dst_index_} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{.named(ctx, "value", value)},
        .location = location,
    });
}

/// `ttl.tile_floor`. Unary tile operation. Result types are explicit.
pub fn tile_floor(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_floor", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_frac`. Unary tile operation. Result types are explicit.
pub fn tile_frac(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_frac", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_gelu`. Unary tile operation. Result types are explicit.
pub fn tile_gelu(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_gelu", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_gt`. Binary tile operation. Result types are explicit.
pub fn tile_gt(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_gt", .{
        .operands = .{ .flat = &.{ lhs, rhs, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_hardsigmoid`. Unary tile operation. Result types are explicit.
pub fn tile_hardsigmoid(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_hardsigmoid", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_log`. Unary tile operation. Result types are explicit.
pub fn tile_log(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_log", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_lt`. Binary tile operation. Result types are explicit.
pub fn tile_lt(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_lt", .{
        .operands = .{ .flat = &.{ lhs, rhs, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_matmul_block`. Block-level matrix multiply-accumulate (CB inputs, DST output). Result types are explicit.
pub fn tile_matmul_block(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, accumulator: ?*const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, transpose_rhs: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(lhs);
    operands.appendAssumeCapacity(rhs);
    if (accumulator) |present| operands.appendAssumeCapacity(present);
    operands.appendAssumeCapacity(dst_index_);
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (transpose_rhs) |present| attrs.appendAssumeCapacity(.named(ctx, "transpose_rhs", present));
    return make(ctx, "ttl.tile_matmul_block", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.tile_max`. Binary tile operation. Result types are explicit.
pub fn tile_max(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_max", .{
        .operands = .{ .flat = &.{ lhs, rhs, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_min`. Binary tile operation. Result types are explicit.
pub fn tile_min(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_min", .{
        .operands = .{ .flat = &.{ lhs, rhs, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_mul`. Binary tile operation. Result types are explicit.
pub fn tile_mul(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_mul", .{
        .operands = .{ .flat = &.{ lhs, rhs, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_mul_unary_const`. Tile-level multiply by a compile-time scalar constant. Result types are explicit.
pub fn tile_mul_unary_const(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, value: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_mul_unary_const", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{.named(ctx, "value", value)},
        .location = location,
    });
}

/// `ttl.tile_ne`. Binary tile operation. Result types are explicit.
pub fn tile_ne(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_ne", .{
        .operands = .{ .flat = &.{ lhs, rhs, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_neg`. Unary tile operation. Result types are explicit.
pub fn tile_neg(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_neg", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_recip`. Unary tile operation. Result types are explicit.
pub fn tile_recip(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_recip", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_reduce`. Tile-level reduction with scaling. Result types are explicit.
pub fn tile_reduce(ctx: *mlir.Context, input: *const mlir.Value, scaler: *const mlir.Value, output: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, reduce_type: *const mlir.Attribute, reduce_dim: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_reduce", .{
        .operands = .{ .flat = &.{ input, scaler, output, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{ .named(ctx, "reduce_type", reduce_type), .named(ctx, "reduce_dim", reduce_dim) },
        .location = location,
    });
}

/// `ttl.tile_regs_acquire`. Acquire the DST register lock. No results.
pub fn tile_regs_acquire(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_regs_acquire", .{
        .location = location,
    });
}

/// `ttl.tile_regs_commit`. Commit the DST register lock (MATH thread). No results.
pub fn tile_regs_commit(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_regs_commit", .{
        .location = location,
    });
}

/// `ttl.tile_regs_release`. Release DST register lock (PACK thread). No results.
pub fn tile_regs_release(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_regs_release", .{
        .location = location,
    });
}

/// `ttl.tile_regs_wait`. Wait for DST register availability (PACK thread). No results.
pub fn tile_regs_wait(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_regs_wait", .{
        .location = location,
    });
}

/// `ttl.tile_relu`. Unary tile operation. Result types are explicit.
pub fn tile_relu(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_relu", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_row_normalization_block`. Normalize one static row in one DST acquisition. Result types are explicit.
pub fn tile_row_normalization_block(ctx: *mlir.Context, input: *const mlir.Value, gamma: *const mlir.Value, output: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, scale: *const mlir.Attribute, epsilon: *const mlir.Attribute, has_gamma: *const mlir.Attribute, num_tiles: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_row_normalization_block", .{
        .operands = .{ .flat = &.{ input, gamma, output, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{ .named(ctx, "scale", scale), .named(ctx, "epsilon", epsilon), .named(ctx, "has_gamma", has_gamma), .named(ctx, "num_tiles", num_tiles) },
        .location = location,
    });
}

/// `ttl.tile_rsqrt`. Unary tile operation. Result types are explicit.
pub fn tile_rsqrt(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_rsqrt", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_sigmoid`. Unary tile operation. Result types are explicit.
pub fn tile_sigmoid(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_sigmoid", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_sign`. Unary tile operation. Result types are explicit.
pub fn tile_sign(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_sign", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_signbit`. Unary tile operation. Result types are explicit.
pub fn tile_signbit(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_signbit", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_silu`. Unary tile operation. Result types are explicit.
pub fn tile_silu(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_silu", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_sin`. Unary tile operation. Result types are explicit.
pub fn tile_sin(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_sin", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_softsign`. Unary tile operation. Result types are explicit.
pub fn tile_softsign(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_softsign", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_sqrt`. Unary tile operation. Result types are explicit.
pub fn tile_sqrt(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_sqrt", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_square`. Unary tile operation. Result types are explicit.
pub fn tile_square(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_square", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_store`. Store a tile into an acquired dataflow buffer view. No results.
pub fn tile_store(ctx: *mlir.Context, tile: *const mlir.Value, view: *const mlir.Value, indices: []const *const mlir.Value, dst_index_: *const mlir.Value, store_kind: ?*const mlir.Attribute, row_prefix: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(tile);
    operands.appendAssumeCapacity(view);
    operands.appendSliceAssumeCapacity(indices);
    operands.appendAssumeCapacity(dst_index_);
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (store_kind) |present| attrs.appendAssumeCapacity(.named(ctx, "store_kind", present));
    if (row_prefix) |present| attrs.appendAssumeCapacity(.named(ctx, "row_prefix", present));
    return make(ctx, "ttl.tile_store", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttl.tile_sub`. Binary tile operation. Result types are explicit.
pub fn tile_sub(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_sub", .{
        .operands = .{ .flat = &.{ lhs, rhs, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_tan`. Unary tile operation. Result types are explicit.
pub fn tile_tan(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_tan", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_tanh`. Unary tile operation. Result types are explicit.
pub fn tile_tanh(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_tanh", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_transpose`. Tile-level 32x32 transpose (dataflow buffer input, DST output). Result types are explicit.
pub fn tile_transpose(ctx: *mlir.Context, input: *const mlir.Value, output: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_transpose", .{
        .operands = .{ .flat = &.{ input, output, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_trunc`. Unary tile operation. Result types are explicit.
pub fn tile_trunc(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_trunc", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.tile_typecast`. Tile-level typecast operation. Result types are explicit.
pub fn tile_typecast(ctx: *mlir.Context, input: *const mlir.Value, dst_index_: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.tile_typecast", .{
        .operands = .{ .flat = &.{ input, dst_index_ } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.transpose`. Transpose a 2D tensor of tiles. Result types are explicit.
pub fn transpose(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.transpose", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.trunc`. Unary elementwise operation. Result types are explicit.
pub fn trunc(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.trunc", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.typecast`. Elementwise typecast on tensors. Result types are explicit.
pub fn typecast(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.typecast", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttl.wait`. Wait on a transfer handle. No results.
pub fn wait(ctx: *mlir.Context, xf: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttl.wait", .{
        .operands = .{ .flat = &.{xf} },
        .location = location,
    });
}

/// `ttl.wait_any`. Wait for one posted PipeNet receive. Result types are fixed.
pub fn wait_any(ctx: *mlir.Context, requests: []const *const mlir.Value, start: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendSliceAssumeCapacity(requests);
    operands.appendAssumeCapacity(start);
    return make(ctx, "ttl.wait_any", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{helpers.parseType(ctx, "!ttl.ready_receive")} },
        .location = location,
    });
}

/// `ttl.yield`. Terminate a TTL region body. No results.
pub fn yield(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendSliceAssumeCapacity(values);
    return make(ctx, "ttl.yield", .{
        .operands = .{ .flat = operands.constSlice() },
        .location = location,
    });
}

/// Results of each builder's operation; -1 when variadic or optional.
pub const num_results = struct {
    pub const abs: comptime_int = 1;
    pub const accumulation_scope: comptime_int = 0;
    pub const acos: comptime_int = 1;
    pub const add: comptime_int = 1;
    pub const asin: comptime_int = 1;
    pub const atan: comptime_int = 1;
    pub const attach_cb: comptime_int = 1;
    pub const bind_cb: comptime_int = 1;
    pub const block_broadcast: comptime_int = 1;
    pub const cb_pop: comptime_int = 0;
    pub const cb_push: comptime_int = 0;
    pub const cb_reserve: comptime_int = 1;
    pub const cb_wait: comptime_int = 1;
    pub const ceil: comptime_int = 1;
    pub const compute: comptime_int = -1;
    pub const copy: comptime_int = 1;
    pub const copy_dst: comptime_int = 1;
    pub const copy_tile: comptime_int = 2;
    pub const core_x: comptime_int = 1;
    pub const core_y: comptime_int = 1;
    pub const cos: comptime_int = 1;
    pub const create_pipe: comptime_int = 1;
    pub const current_device_index: comptime_int = 1;
    pub const dfb_reconfiguration: comptime_int = 0;
    pub const div: comptime_int = 1;
    pub const dprint: comptime_int = 0;
    pub const dst_index: comptime_int = 1;
    pub const dst_section: comptime_int = 0;
    pub const eq: comptime_int = 1;
    pub const exp: comptime_int = 1;
    pub const exp2: comptime_int = 1;
    pub const expm1: comptime_int = 1;
    pub const fill: comptime_int = 1;
    pub const floor: comptime_int = 1;
    pub const frac: comptime_int = 1;
    pub const gelu: comptime_int = 1;
    pub const get_dfb_id: comptime_int = 1;
    pub const gt: comptime_int = 1;
    pub const hardsigmoid: comptime_int = 1;
    pub const if_dst: comptime_int = 0;
    pub const if_src: comptime_int = 0;
    pub const init_sfpu: comptime_int = 0;
    pub const is_active: comptime_int = 1;
    pub const is_device: comptime_int = 1;
    pub const is_device_in_range: comptime_int = 1;
    pub const is_dst: comptime_int = 1;
    pub const is_src: comptime_int = 1;
    pub const iter_index: comptime_int = 1;
    pub const log: comptime_int = 1;
    pub const lt: comptime_int = 1;
    pub const matmul: comptime_int = 1;
    pub const max: comptime_int = 1;
    pub const min: comptime_int = 1;
    pub const mul: comptime_int = 1;
    pub const mul_unary_const: comptime_int = 1;
    pub const ne: comptime_int = 1;
    pub const neg: comptime_int = 1;
    pub const opaque_call: comptime_int = -1;
    pub const pipe_transfer_create: comptime_int = 1;
    pub const pipe_transfer_post: comptime_int = 1;
    pub const pipe_transfer_send: comptime_int = 1;
    pub const pipe_transfer_wait: comptime_int = 0;
    pub const pipe_transfer_wait_any: comptime_int = 1;
    pub const pipenet_destination_count: comptime_int = 1;
    pub const pipenet_foreach_dst: comptime_int = 0;
    pub const pipenet_foreach_src: comptime_int = 0;
    pub const pipenet_scope: comptime_int = 0;
    pub const raw_addr: comptime_int = 1;
    pub const raw_element_read: comptime_int = 1;
    pub const raw_element_write: comptime_int = 0;
    pub const read_index: comptime_int = 1;
    pub const ready_receive_index: comptime_int = 1;
    pub const recip: comptime_int = 1;
    pub const reduce: comptime_int = 1;
    pub const relu: comptime_int = 1;
    pub const reset_all_dfbs: comptime_int = 0;
    pub const reset_dfbs: comptime_int = 0;
    pub const rsqrt: comptime_int = 1;
    pub const select_pipe_dst: comptime_int = 1;
    pub const select_pipe_src: comptime_int = 1;
    pub const selected_pipe_destination_coordinates: comptime_int = 4;
    pub const selected_pipe_destination_device_index: comptime_int = 1;
    pub const selected_pipe_source_coordinates: comptime_int = 2;
    pub const selected_pipe_source_device_index: comptime_int = 1;
    pub const sigmoid: comptime_int = 1;
    pub const sign: comptime_int = 1;
    pub const signbit: comptime_int = 1;
    pub const signpost: comptime_int = 0;
    pub const silu: comptime_int = 1;
    pub const sin: comptime_int = 1;
    pub const softsign: comptime_int = 1;
    pub const sqrt: comptime_int = 1;
    pub const square: comptime_int = 1;
    pub const store: comptime_int = 0;
    pub const sub: comptime_int = 1;
    pub const tan: comptime_int = 1;
    pub const tanh: comptime_int = 1;
    pub const tensor_slice: comptime_int = 1;
    pub const tile_abs: comptime_int = 1;
    pub const tile_accumulate: comptime_int = 1;
    pub const tile_acos: comptime_int = 1;
    pub const tile_add: comptime_int = 1;
    pub const tile_asin: comptime_int = 1;
    pub const tile_atan: comptime_int = 1;
    pub const tile_bcast: comptime_int = 1;
    pub const tile_ceil: comptime_int = 1;
    pub const tile_cos: comptime_int = 1;
    pub const tile_div: comptime_int = 1;
    pub const tile_eq: comptime_int = 1;
    pub const tile_exp: comptime_int = 1;
    pub const tile_exp2: comptime_int = 1;
    pub const tile_expm1: comptime_int = 1;
    pub const tile_fill: comptime_int = 1;
    pub const tile_floor: comptime_int = 1;
    pub const tile_frac: comptime_int = 1;
    pub const tile_gelu: comptime_int = 1;
    pub const tile_gt: comptime_int = 1;
    pub const tile_hardsigmoid: comptime_int = 1;
    pub const tile_log: comptime_int = 1;
    pub const tile_lt: comptime_int = 1;
    pub const tile_matmul_block: comptime_int = 1;
    pub const tile_max: comptime_int = 1;
    pub const tile_min: comptime_int = 1;
    pub const tile_mul: comptime_int = 1;
    pub const tile_mul_unary_const: comptime_int = 1;
    pub const tile_ne: comptime_int = 1;
    pub const tile_neg: comptime_int = 1;
    pub const tile_recip: comptime_int = 1;
    pub const tile_reduce: comptime_int = 1;
    pub const tile_regs_acquire: comptime_int = 0;
    pub const tile_regs_commit: comptime_int = 0;
    pub const tile_regs_release: comptime_int = 0;
    pub const tile_regs_wait: comptime_int = 0;
    pub const tile_relu: comptime_int = 1;
    pub const tile_row_normalization_block: comptime_int = 1;
    pub const tile_rsqrt: comptime_int = 1;
    pub const tile_sigmoid: comptime_int = 1;
    pub const tile_sign: comptime_int = 1;
    pub const tile_signbit: comptime_int = 1;
    pub const tile_silu: comptime_int = 1;
    pub const tile_sin: comptime_int = 1;
    pub const tile_softsign: comptime_int = 1;
    pub const tile_sqrt: comptime_int = 1;
    pub const tile_square: comptime_int = 1;
    pub const tile_store: comptime_int = 0;
    pub const tile_sub: comptime_int = 1;
    pub const tile_tan: comptime_int = 1;
    pub const tile_tanh: comptime_int = 1;
    pub const tile_transpose: comptime_int = 1;
    pub const tile_trunc: comptime_int = 1;
    pub const tile_typecast: comptime_int = 1;
    pub const transpose: comptime_int = 1;
    pub const trunc: comptime_int = 1;
    pub const typecast: comptime_int = 1;
    pub const wait: comptime_int = 0;
    pub const wait_any: comptime_int = 1;
    pub const yield: comptime_int = 0;
};
