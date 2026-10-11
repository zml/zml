//! `ttkernel` operation builders, generated from tt-lang's `include/ttlang/Dialect/TTKernel/IR/TTKernelOps.td` at 9b7ad24d1bd6a89dc1f8f885686bc4db86ac7244; do not edit, see REGENERATING.md.
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
    "ttkernel.LocalTensorAccessor",
    "ttkernel.TensorAccessor",
    "ttkernel.TensorAccessorArgs",
    "ttkernel.abs_tile",
    "ttkernel.abs_tile_init",
    "ttkernel.abs_tile_int32",
    "ttkernel.acos_tile",
    "ttkernel.acos_tile_init",
    "ttkernel.add_binary_tile",
    "ttkernel.add_binary_tile_init",
    "ttkernel.add_int_tile",
    "ttkernel.add_int_tile_init",
    "ttkernel.add_tiles",
    "ttkernel.add_tiles_init",
    "ttkernel.add_unary_tile",
    "ttkernel.add_unary_tile_int32",
    "ttkernel.asin_tile",
    "ttkernel.asin_tile_init",
    "ttkernel.atan2_binary_tile",
    "ttkernel.atan2_binary_tile_init",
    "ttkernel.atan_tile",
    "ttkernel.atan_tile_init",
    "ttkernel.bfloat16_greater",
    "ttkernel.binary_bitwise_tile_init",
    "ttkernel.binary_dest_reuse_tiles",
    "ttkernel.binary_dest_reuse_tiles_init",
    "ttkernel.binary_left_shift_tile",
    "ttkernel.binary_logical_right_shift_tile",
    "ttkernel.binary_max_int32_tile",
    "ttkernel.binary_max_int32_tile_init",
    "ttkernel.binary_max_tile",
    "ttkernel.binary_max_tile_init",
    "ttkernel.binary_min_int32_tile",
    "ttkernel.binary_min_int32_tile_init",
    "ttkernel.binary_min_tile",
    "ttkernel.binary_min_tile_init",
    "ttkernel.binary_op_init_common",
    "ttkernel.binary_right_shift_tile",
    "ttkernel.binary_shift_tile_init",
    "ttkernel.binop_with_scalar_tile_init",
    "ttkernel.bitcast",
    "ttkernel.bitwise_and_binary_tile",
    "ttkernel.bitwise_not_tile",
    "ttkernel.bitwise_not_tile_init",
    "ttkernel.bitwise_or_binary_tile",
    "ttkernel.bitwise_xor_binary_tile",
    "ttkernel.cast_to_l1_addr",
    "ttkernel.cb_pop_front",
    "ttkernel.cb_push_back",
    "ttkernel.cb_reserve_back",
    "ttkernel.cb_wait_front",
    "ttkernel.ceil_tile",
    "ttkernel.clamp_tile",
    "ttkernel.clamp_tile_init",
    "ttkernel.clamp_tile_int32",
    "ttkernel.compute_kernel_hw_startup",
    "ttkernel.copy_block_matmul_partials",
    "ttkernel.copy_dest_values",
    "ttkernel.copy_dest_values_init",
    "ttkernel.copy_tile",
    "ttkernel.copy_tile_init",
    "ttkernel.cos_tile",
    "ttkernel.cos_tile_init",
    "ttkernel.div_binary_tile",
    "ttkernel.div_binary_tile_init",
    "ttkernel.div_unary_tile",
    "ttkernel.dprint",
    "ttkernel.eq_binary_tile",
    "ttkernel.eq_binary_tile_init",
    "ttkernel.eqz_tile",
    "ttkernel.eqz_tile_init",
    "ttkernel.eqz_tile_int32",
    "ttkernel.erf_tile",
    "ttkernel.erf_tile_init",
    "ttkernel.erfc_tile",
    "ttkernel.erfc_tile_init",
    "ttkernel.exp2_tile",
    "ttkernel.exp2_tile_init",
    "ttkernel.exp_tile",
    "ttkernel.exp_tile_init",
    "ttkernel.experimental.close_fabric_connections",
    "ttkernel.experimental.constant_table_lookup",
    "ttkernel.experimental.convert_logical_x_to_translated",
    "ttkernel.experimental.convert_logical_y_to_translated",
    "ttkernel.experimental.create_fabric_connection_manager",
    "ttkernel.experimental.fabric_fast_write_any_len",
    "ttkernel.experimental.fabric_mcast_fast_write_any_len",
    "ttkernel.experimental.fabric_mcast_sem_inc",
    "ttkernel.experimental.fabric_sem_inc",
    "ttkernel.experimental.fill_arange_tile",
    "ttkernel.experimental.get_device_id_from_logical_mesh_position",
    "ttkernel.experimental.get_my_device_id",
    "ttkernel.experimental.get_my_logical_mesh_position",
    "ttkernel.experimental.matmul_block",
    "ttkernel.experimental.pack_untilize_block",
    "ttkernel.experimental.semaphore_reached",
    "ttkernel.experimental.semaphore_wait",
    "ttkernel.experimental.semaphore_wait_min",
    "ttkernel.experimental.setup_fabric_connections",
    "ttkernel.experimental.tilize_block",
    "ttkernel.experimental.unpack_stall_on_pack",
    "ttkernel.experimental.untilize_block",
    "ttkernel.experimental.write_col_mask_tile",
    "ttkernel.experimental.write_row_mask_tile",
    "ttkernel.experimental_row_normalization_block",
    "ttkernel.expm1_tile",
    "ttkernel.expm1_tile_init",
    "ttkernel.fill_tile",
    "ttkernel.fill_tile_init",
    "ttkernel.fill_tile_int",
    "ttkernel.float32_greater",
    "ttkernel.floor_tile",
    "ttkernel.frac_tile",
    "ttkernel.ge_binary_tile",
    "ttkernel.ge_binary_tile_init",
    "ttkernel.gelu_tile",
    "ttkernel.gelu_tile_init",
    "ttkernel.get_arg_val",
    "ttkernel.get_common_arg_val",
    "ttkernel.get_compile_time_arg_val",
    "ttkernel.get_dataformat",
    "ttkernel.get_dfb_id",
    "ttkernel.get_noc_addr",
    "ttkernel.get_noc_addr_from_bank_id",
    "ttkernel.get_noc_multicast_addr",
    "ttkernel.get_read_ptr",
    "ttkernel.get_semaphore",
    "ttkernel.get_tile_size",
    "ttkernel.get_write_ptr",
    "ttkernel.gez_tile",
    "ttkernel.gez_tile_init",
    "ttkernel.gez_tile_int32",
    "ttkernel.gt_binary_tile",
    "ttkernel.gt_binary_tile_init",
    "ttkernel.gtz_tile",
    "ttkernel.gtz_tile_init",
    "ttkernel.gtz_tile_int32",
    "ttkernel.hardsigmoid_tile",
    "ttkernel.hardsigmoid_tile_init",
    "ttkernel.init_sfpu",
    "ttkernel.invoke_sfpi",
    "ttkernel.le_binary_tile",
    "ttkernel.le_binary_tile_init",
    "ttkernel.lez_tile",
    "ttkernel.lez_tile_init",
    "ttkernel.lez_tile_int32",
    "ttkernel.load_from_l1",
    "ttkernel.log1p_tile",
    "ttkernel.log1p_tile_init",
    "ttkernel.log_tile",
    "ttkernel.log_tile_init",
    "ttkernel.logical_not_tile",
    "ttkernel.logical_not_tile_init",
    "ttkernel.lt_binary_tile",
    "ttkernel.lt_binary_tile_init",
    "ttkernel.ltz_tile",
    "ttkernel.ltz_tile_init",
    "ttkernel.ltz_tile_int32",
    "ttkernel.matmul_block",
    "ttkernel.matmul_tiles",
    "ttkernel.mem_zeros_base",
    "ttkernel.mem_zeros_size",
    "ttkernel.mm_block_init",
    "ttkernel.mm_block_init_short",
    "ttkernel.mm_init",
    "ttkernel.mm_init_short",
    "ttkernel.mul_binary_tile",
    "ttkernel.mul_binary_tile_init",
    "ttkernel.mul_int_tile",
    "ttkernel.mul_int_tile_init",
    "ttkernel.mul_tiles",
    "ttkernel.mul_tiles_init",
    "ttkernel.mul_unary_tile",
    "ttkernel.my_logical_x_",
    "ttkernel.my_logical_y_",
    "ttkernel.my_x",
    "ttkernel.my_y",
    "ttkernel.ne_binary_tile",
    "ttkernel.ne_binary_tile_init",
    "ttkernel.negative_tile",
    "ttkernel.negative_tile_init",
    "ttkernel.negative_tile_int32",
    "ttkernel.nez_tile",
    "ttkernel.nez_tile_init",
    "ttkernel.nez_tile_int32",
    "ttkernel.noc_async_atomic_barrier",
    "ttkernel.noc_async_read",
    "ttkernel.noc_async_read_barrier",
    "ttkernel.noc_async_read_barrier_with_trid",
    "ttkernel.noc_async_read_one_packet_set_state",
    "ttkernel.noc_async_read_one_packet_with_state",
    "ttkernel.noc_async_read_tile",
    "ttkernel.noc_async_write",
    "ttkernel.noc_async_write_barrier",
    "ttkernel.noc_async_write_barrier_with_trid",
    "ttkernel.noc_async_write_multicast",
    "ttkernel.noc_async_write_multicast_loopback_src",
    "ttkernel.noc_async_write_multicast_one_packet",
    "ttkernel.noc_async_write_one_packet_set_state",
    "ttkernel.noc_async_write_one_packet_with_state",
    "ttkernel.noc_async_write_one_packet_with_trid",
    "ttkernel.noc_async_write_tile",
    "ttkernel.noc_async_writes_flushed",
    "ttkernel.noc_inline_dw_write",
    "ttkernel.noc_semaphore_inc",
    "ttkernel.noc_semaphore_inc_multicast",
    "ttkernel.noc_semaphore_set",
    "ttkernel.noc_semaphore_set_multicast",
    "ttkernel.noc_semaphore_set_multicast_loopback_src",
    "ttkernel.opaque_call",
    "ttkernel.pack_reconfig_data_format",
    "ttkernel.pack_reconfig_l1_acc",
    "ttkernel.pack_tile",
    "ttkernel.pack_tile_block",
    "ttkernel.pack_untilize_init",
    "ttkernel.pack_untilize_uninit",
    "ttkernel.pack_waited_tile",
    "ttkernel.power_binary_tile",
    "ttkernel.power_binary_tile_init",
    "ttkernel.power_tile",
    "ttkernel.power_tile_init",
    "ttkernel.rand_tile",
    "ttkernel.rand_tile_init",
    "ttkernel.recip_tile",
    "ttkernel.recip_tile_init",
    "ttkernel.reduce_init",
    "ttkernel.reduce_tile",
    "ttkernel.reduce_uninit",
    "ttkernel.reinterpret_cast",
    "ttkernel.relu_tile",
    "ttkernel.relu_tile_init",
    "ttkernel.relu_tile_int32",
    "ttkernel.remote_sram_write_u32",
    "ttkernel.rounding_op_tile_init",
    "ttkernel.routing_plane.atomic_inc",
    "ttkernel.routing_plane.close_connections",
    "ttkernel.routing_plane.create_connection_manager",
    "ttkernel.routing_plane.fused_write_atomic_inc",
    "ttkernel.routing_plane.open_connections",
    "ttkernel.routing_plane.scatter_write",
    "ttkernel.routing_plane.write",
    "ttkernel.rsqrt_tile",
    "ttkernel.rsqrt_tile_init",
    "ttkernel.selu_tile",
    "ttkernel.selu_tile_init",
    "ttkernel.sfpu_reduce",
    "ttkernel.sfpu_reduce_init",
    "ttkernel.sigmoid_tile",
    "ttkernel.sigmoid_tile_init",
    "ttkernel.sign_tile",
    "ttkernel.sign_tile_init",
    "ttkernel.signbit_tile",
    "ttkernel.signbit_tile_init",
    "ttkernel.silu_tile",
    "ttkernel.silu_tile_init",
    "ttkernel.sin_tile",
    "ttkernel.sin_tile_init",
    "ttkernel.softsign_tile",
    "ttkernel.softsign_tile_init",
    "ttkernel.sqrt_tile",
    "ttkernel.sqrt_tile_init",
    "ttkernel.square_tile",
    "ttkernel.square_tile_init",
    "ttkernel.store_to_l1",
    "ttkernel.sub_binary_tile",
    "ttkernel.sub_binary_tile_init",
    "ttkernel.sub_int_tile",
    "ttkernel.sub_int_tile_init",
    "ttkernel.sub_tiles",
    "ttkernel.sub_tiles_init",
    "ttkernel.sub_unary_tile",
    "ttkernel.sub_unary_tile_int32",
    "ttkernel.tan_tile",
    "ttkernel.tan_tile_init",
    "ttkernel.tanh_tile",
    "ttkernel.tanh_tile_init",
    "ttkernel.tensor_accessor.get_bank_and_offset",
    "ttkernel.tensor_accessor.get_noc_addr",
    "ttkernel.tensor_accessor.get_shard_noc_addr",
    "ttkernel.tensor_accessor.is_local_addr",
    "ttkernel.tensor_accessor.is_local_bank",
    "ttkernel.tensor_accessor.is_local_page",
    "ttkernel.tensor_accessor.is_local_shard",
    "ttkernel.tile_regs_acquire",
    "ttkernel.tile_regs_commit",
    "ttkernel.tile_regs_release",
    "ttkernel.tile_regs_wait",
    "ttkernel.tilize_block",
    "ttkernel.tilize_init",
    "ttkernel.tilize_uninit",
    "ttkernel.topk_local_sort",
    "ttkernel.topk_merge",
    "ttkernel.topk_rebuild",
    "ttkernel.topk_tile_init",
    "ttkernel.transpose_wh_init",
    "ttkernel.transpose_wh_tile",
    "ttkernel.trunc_tile",
    "ttkernel.typecast_tile",
    "ttkernel.typecast_tile_init",
    "ttkernel.unary_bcast",
    "ttkernel.unary_bcast_init",
    "ttkernel.unary_op_init_common",
    "ttkernel.unreachable",
    "ttkernel.untilize_block",
    "ttkernel.untilize_init",
    "ttkernel.untilize_uninit",
    "ttkernel.where_tile",
    "ttkernel.where_tile_init",
};

/// `ttkernel.LocalTensorAccessor`. Construct a local SRAM byte accessor. Result types are fixed.
pub fn LocalTensorAccessor(ctx: *mlir.Context, bank_base_address_in: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.LocalTensorAccessor", .{
        .operands = .{ .flat = &.{bank_base_address_in} },
        .results = .{ .flat = &.{helpers.parseType(ctx, "!ttkernel.LocalTensorAccessor")} },
        .location = location,
    });
}

/// `ttkernel.TensorAccessor`. Construct a distributed tensor accessor. Result types are fixed.
pub fn TensorAccessor(ctx: *mlir.Context, args: *const mlir.Value, bank_base_address_in: *const mlir.Value, page_size_in: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(args);
    operands.appendAssumeCapacity(bank_base_address_in);
    if (page_size_in) |present| operands.appendAssumeCapacity(present);
    return make(ctx, "ttkernel.TensorAccessor", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{helpers.parseType(ctx, "!ttkernel.TensorAccessor")} },
        .location = location,
    });
}

/// `ttkernel.TensorAccessorArgs`. TensorAccessorArgs. Result types are fixed.
pub fn TensorAccessorArgs(ctx: *mlir.Context, cta_base: ?*const mlir.Value, crta_base: ?*const mlir.Value, prev_args: ?*const mlir.Value, cta_expr: ?*const mlir.Attribute, crta_expr: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (cta_expr) |present| attrs.appendAssumeCapacity(.named(ctx, "cta_expr", present));
    if (crta_expr) |present| attrs.appendAssumeCapacity(.named(ctx, "crta_expr", present));
    return make(ctx, "ttkernel.TensorAccessorArgs", .{
        .operands = .{ .variadic = &.{ if (cta_base) |present| &.{present} else &.{}, if (crta_base) |present| &.{present} else &.{}, if (prev_args) |present| &.{present} else &.{} } },
        .results = .{ .flat = &.{helpers.parseType(ctx, "!ttkernel.TensorAccessorArgs")} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttkernel.abs_tile`. Absolute value tile in the DST at specified index. No results.
pub fn abs_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.abs_tile", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.abs_tile_init`. Init function for abs_tile operation. Refer to documentation for any init function. No results.
pub fn abs_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.abs_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.abs_tile_int32`. Absolute value i32 tile in the DST at specified index. No results.
pub fn abs_tile_int32(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.abs_tile_int32", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.acos_tile`. Arccosine tile in the DST at specified index. No results.
pub fn acos_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.acos_tile", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.acos_tile_init`. Init function for acos_tile operation. Refer to documentation for any init function. No results.
pub fn acos_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.acos_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.add_binary_tile`. Addition operation between two tiles. No results.
pub fn add_binary_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.add_binary_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .location = location,
    });
}

/// `ttkernel.add_binary_tile_init`. Short init function. No results.
pub fn add_binary_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.add_binary_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.add_int_tile`. Integer addition operation between two tiles. No results.
pub fn add_int_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, dtype: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.add_int_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .attributes = &.{.named(ctx, "dtype", dtype)},
        .location = location,
    });
}

/// `ttkernel.add_int_tile_init`. Short init function. No results.
pub fn add_int_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.add_int_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.add_tiles`. Add operation. No results.
pub fn add_tiles(ctx: *mlir.Context, in0_cb: *const mlir.Value, in1_cb: *const mlir.Value, in0_tile_index: *const mlir.Value, in1_tile_index: *const mlir.Value, dst_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.add_tiles", .{
        .operands = .{ .flat = &.{ in0_cb, in1_cb, in0_tile_index, in1_tile_index, dst_index } },
        .location = location,
    });
}

/// `ttkernel.add_tiles_init`. Short init function. No results.
pub fn add_tiles_init(ctx: *mlir.Context, in0_cb: *const mlir.Value, in1_cb: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.add_tiles_init", .{
        .operands = .{ .flat = &.{ in0_cb, in1_cb } },
        .location = location,
    });
}

/// `ttkernel.add_unary_tile`. Add by scalar operation. No results.
pub fn add_unary_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, param: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.add_unary_tile", .{
        .operands = .{ .flat = &.{ dst0_index, param } },
        .location = location,
    });
}

/// `ttkernel.add_unary_tile_int32`. Add by int32 scalar operation. No results.
pub fn add_unary_tile_int32(ctx: *mlir.Context, dst0_index: *const mlir.Value, param: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.add_unary_tile_int32", .{
        .operands = .{ .flat = &.{ dst0_index, param } },
        .location = location,
    });
}

/// `ttkernel.asin_tile`. Arcsine tile in the DST at specified index. No results.
pub fn asin_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.asin_tile", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.asin_tile_init`. Init function for asin_tile operation. Refer to documentation for any init function. No results.
pub fn asin_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.asin_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.atan2_binary_tile`. Elementwise atan2 operation between two tiles. No results.
pub fn atan2_binary_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.atan2_binary_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .location = location,
    });
}

/// `ttkernel.atan2_binary_tile_init`. Short init function. No results.
pub fn atan2_binary_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.atan2_binary_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.atan_tile`. Arctangent tile in the DST at specified index. No results.
pub fn atan_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.atan_tile", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.atan_tile_init`. Init function for atan_tile operation. Refer to documentation for any init function. No results.
pub fn atan_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.atan_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.bfloat16_greater`. Compare two bfloat16 values using integer arithmetic. Result types are fixed.
pub fn bfloat16_greater(ctx: *mlir.Context, bf16_a: *const mlir.Value, bf16_b: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.bfloat16_greater", .{
        .operands = .{ .flat = &.{ bf16_a, bf16_b } },
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i1)} },
        .location = location,
    });
}

/// `ttkernel.binary_bitwise_tile_init`. Init function for binary bitwise operations (AND, OR, XOR). Refer to documentation for any init function. No results.
pub fn binary_bitwise_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.binary_bitwise_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.binary_dest_reuse_tiles`. Binary op with one operand from DST. No results.
pub fn binary_dest_reuse_tiles(ctx: *mlir.Context, in_cb: *const mlir.Value, in_tile_index: *const mlir.Value, dst_tile_index: *const mlir.Value, eltwise_binary_type: *const mlir.Attribute, reuse_type: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.binary_dest_reuse_tiles", .{
        .operands = .{ .flat = &.{ in_cb, in_tile_index, dst_tile_index } },
        .attributes = &.{ .named(ctx, "eltwise_binary_type", eltwise_binary_type), .named(ctx, "reuse_type", reuse_type) },
        .location = location,
    });
}

/// `ttkernel.binary_dest_reuse_tiles_init`. Init for binary op with dest reuse. No results.
pub fn binary_dest_reuse_tiles_init(ctx: *mlir.Context, in_cb: *const mlir.Value, eltwise_binary_type: *const mlir.Attribute, reuse_type: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.binary_dest_reuse_tiles_init", .{
        .operands = .{ .flat = &.{in_cb} },
        .attributes = &.{ .named(ctx, "eltwise_binary_type", eltwise_binary_type), .named(ctx, "reuse_type", reuse_type) },
        .location = location,
    });
}

/// `ttkernel.binary_left_shift_tile`. Elementwise left shift between two tiles. No results.
pub fn binary_left_shift_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, dtype: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.binary_left_shift_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .attributes = &.{.named(ctx, "dtype", dtype)},
        .location = location,
    });
}

/// `ttkernel.binary_logical_right_shift_tile`. Elementwise logical right shift between two tiles. No results.
pub fn binary_logical_right_shift_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, dtype: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.binary_logical_right_shift_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .attributes = &.{.named(ctx, "dtype", dtype)},
        .location = location,
    });
}

/// `ttkernel.binary_max_int32_tile`. Integer elementwise maximum operation between two tiles. No results.
pub fn binary_max_int32_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.binary_max_int32_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .location = location,
    });
}

/// `ttkernel.binary_max_int32_tile_init`. Short init function. No results.
pub fn binary_max_int32_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.binary_max_int32_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.binary_max_tile`. Elementwise maximum operation. No results.
pub fn binary_max_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.binary_max_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .location = location,
    });
}

/// `ttkernel.binary_max_tile_init`. Short init function. No results.
pub fn binary_max_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.binary_max_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.binary_min_int32_tile`. Integer elementwise minimum operation between two tiles. No results.
pub fn binary_min_int32_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.binary_min_int32_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .location = location,
    });
}

/// `ttkernel.binary_min_int32_tile_init`. Short init function. No results.
pub fn binary_min_int32_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.binary_min_int32_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.binary_min_tile`. Elementwise minimum operation. No results.
pub fn binary_min_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.binary_min_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .location = location,
    });
}

/// `ttkernel.binary_min_tile_init`. Short init function. No results.
pub fn binary_min_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.binary_min_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.binary_op_init_common`. Init function for all binary ops. No results.
pub fn binary_op_init_common(ctx: *mlir.Context, in0_cb: *const mlir.Value, in1_cb: *const mlir.Value, out_cb: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.binary_op_init_common", .{
        .operands = .{ .flat = &.{ in0_cb, in1_cb, out_cb } },
        .location = location,
    });
}

/// `ttkernel.binary_right_shift_tile`. Elementwise arithmetic right shift between two tiles. No results.
pub fn binary_right_shift_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, dtype: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.binary_right_shift_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .attributes = &.{.named(ctx, "dtype", dtype)},
        .location = location,
    });
}

/// `ttkernel.binary_shift_tile_init`. Init function for binary shift tile operations (left / right / logical right). Refer to documentation for any init function. No results.
pub fn binary_shift_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.binary_shift_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.binop_with_scalar_tile_init`. Init function for binary operations with scalar tile operations. No results.
pub fn binop_with_scalar_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.binop_with_scalar_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.bitcast`. Reinterpret a ui32 compile-time arg value as a scalar type. Result types are explicit.
pub fn bitcast(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.bitcast", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `ttkernel.bitwise_and_binary_tile`. Bitwise AND operation between two tiles. No results.
pub fn bitwise_and_binary_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, dtype: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.bitwise_and_binary_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .attributes = &.{.named(ctx, "dtype", dtype)},
        .location = location,
    });
}

/// `ttkernel.bitwise_not_tile`. Bitwise Not operation on tile in the DST at specified index. No results.
pub fn bitwise_not_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.bitwise_not_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.bitwise_not_tile_init`. Init function for bitwise_not_tile operation. Refer to documentation for any init function. No results.
pub fn bitwise_not_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.bitwise_not_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.bitwise_or_binary_tile`. Bitwise OR operation between two tiles. No results.
pub fn bitwise_or_binary_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, dtype: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.bitwise_or_binary_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .attributes = &.{.named(ctx, "dtype", dtype)},
        .location = location,
    });
}

/// `ttkernel.bitwise_xor_binary_tile`. Bitwise XOR operation between two tiles. No results.
pub fn bitwise_xor_binary_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, dtype: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.bitwise_xor_binary_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .attributes = &.{.named(ctx, "dtype", dtype)},
        .location = location,
    });
}

/// `ttkernel.cast_to_l1_addr`. Normalize a value to an L1 address. Result types are fixed.
pub fn cast_to_l1_addr(ctx: *mlir.Context, addr: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.cast_to_l1_addr", .{
        .operands = .{ .flat = &.{addr} },
        .results = .{ .flat = &.{helpers.parseType(ctx, "!ttkernel.l1_addr")} },
        .location = location,
    });
}

/// `ttkernel.cb_pop_front`. CBPopFront call. No results.
pub fn cb_pop_front(ctx: *mlir.Context, cb: *const mlir.Value, num_pages: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.cb_pop_front", .{
        .operands = .{ .flat = &.{ cb, num_pages } },
        .location = location,
    });
}

/// `ttkernel.cb_push_back`. CBPushBack call. No results.
pub fn cb_push_back(ctx: *mlir.Context, cb: *const mlir.Value, num_pages: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.cb_push_back", .{
        .operands = .{ .flat = &.{ cb, num_pages } },
        .location = location,
    });
}

/// `ttkernel.cb_reserve_back`. CBReserveBack call. No results.
pub fn cb_reserve_back(ctx: *mlir.Context, cb: *const mlir.Value, num_pages: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.cb_reserve_back", .{
        .operands = .{ .flat = &.{ cb, num_pages } },
        .location = location,
    });
}

/// `ttkernel.cb_wait_front`. CBWaitFront call. No results.
pub fn cb_wait_front(ctx: *mlir.Context, cb: *const mlir.Value, num_pages: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.cb_wait_front", .{
        .operands = .{ .flat = &.{ cb, num_pages } },
        .location = location,
    });
}

/// `ttkernel.ceil_tile`. Ceil tile in the DST at specified index. No results.
pub fn ceil_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.ceil_tile", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.clamp_tile`. Clamp tile elements to scalar range. No results.
pub fn clamp_tile(ctx: *mlir.Context, dst_index: *const mlir.Value, min_param: *const mlir.Value, max_param: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.clamp_tile", .{
        .operands = .{ .flat = &.{ dst_index, min_param, max_param } },
        .location = location,
    });
}

/// `ttkernel.clamp_tile_init`. Init function for clamp_scalar_tile operation. No results.
pub fn clamp_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.clamp_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.clamp_tile_int32`. Clamp int32 tile elements to scalar range. No results.
pub fn clamp_tile_int32(ctx: *mlir.Context, dst_index: *const mlir.Value, min_param: *const mlir.Value, max_param: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.clamp_tile_int32", .{
        .operands = .{ .flat = &.{ dst_index, min_param, max_param } },
        .location = location,
    });
}

/// `ttkernel.compute_kernel_hw_startup`. compute_kernel_hw_startup. No results.
pub fn compute_kernel_hw_startup(ctx: *mlir.Context, icb0: *const mlir.Value, icb1: ?*const mlir.Value, ocb: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(icb0);
    if (icb1) |present| operands.appendAssumeCapacity(present);
    operands.appendAssumeCapacity(ocb);
    return make(ctx, "ttkernel.compute_kernel_hw_startup", .{
        .operands = .{ .flat = operands.constSlice() },
        .location = location,
    });
}

/// `ttkernel.copy_block_matmul_partials`. CopyBlockMatmulPartials op. No results.
pub fn copy_block_matmul_partials(ctx: *mlir.Context, cb: *const mlir.Value, start_tile_index: *const mlir.Value, start_dst_index: *const mlir.Value, ntiles: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.copy_block_matmul_partials", .{
        .operands = .{ .flat = &.{ cb, start_tile_index, start_dst_index, ntiles } },
        .location = location,
    });
}

/// `ttkernel.copy_dest_values`. Copies all values from the tile at idst_in to the tile at idst_out in the DST register buffer. No results.
pub fn copy_dest_values(ctx: *mlir.Context, idst_in: *const mlir.Value, idst_out: *const mlir.Value, data_format: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.copy_dest_values", .{
        .operands = .{ .flat = &.{ idst_in, idst_out } },
        .attributes = &.{.named(ctx, "data_format", data_format)},
        .location = location,
    });
}

/// `ttkernel.copy_dest_values_init`. Short init function. No results.
pub fn copy_dest_values_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.copy_dest_values_init", .{
        .location = location,
    });
}

/// `ttkernel.copy_tile`. Copy tile from specified CB to DST. No results.
pub fn copy_tile(ctx: *mlir.Context, cb0: *const mlir.Value, tile_index_cb: *const mlir.Value, tile_index_dst: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.copy_tile", .{
        .operands = .{ .flat = &.{ cb0, tile_index_cb, tile_index_dst } },
        .location = location,
    });
}

/// `ttkernel.copy_tile_init`. Perform the init for copy tile. This does not reconfigure the unpacker data types. No results.
pub fn copy_tile_init(ctx: *mlir.Context, cb0: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.copy_tile_init", .{
        .operands = .{ .flat = &.{cb0} },
        .location = location,
    });
}

/// `ttkernel.cos_tile`. Cos operation. No results.
pub fn cos_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.cos_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.cos_tile_init`. Short init function which configures compute unit for execution of cos_tile. No results.
pub fn cos_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.cos_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.div_binary_tile`. Divide operation between two tiles. No results.
pub fn div_binary_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.div_binary_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .location = location,
    });
}

/// `ttkernel.div_binary_tile_init`. Short init function. No results.
pub fn div_binary_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.div_binary_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.div_unary_tile`. Divide by scalar operation. No results.
pub fn div_unary_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, param: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.div_unary_tile", .{
        .operands = .{ .flat = &.{ dst0_index, param } },
        .location = location,
    });
}

/// `ttkernel.dprint`. Print to output stream from kernel. No results.
pub fn dprint(ctx: *mlir.Context, argv: []const *const mlir.Value, fmt: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendSliceAssumeCapacity(argv);
    return make(ctx, "ttkernel.dprint", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = &.{.named(ctx, "fmt", fmt)},
        .location = location,
    });
}

/// `ttkernel.eq_binary_tile`. Equality operation between two tiles. No results.
pub fn eq_binary_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.eq_binary_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .location = location,
    });
}

/// `ttkernel.eq_binary_tile_init`. Short init function. No results.
pub fn eq_binary_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.eq_binary_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.eqz_tile`. Equal to zero tile in the DST at specified index. No results.
pub fn eqz_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.eqz_tile", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.eqz_tile_init`. Init function for eqz() operation. Refer to documentation for any init function. No results.
pub fn eqz_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.eqz_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.eqz_tile_int32`. Equal to zero tile in the DST at specified index. No results.
pub fn eqz_tile_int32(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.eqz_tile_int32", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.erf_tile`. Erf operation. No results.
pub fn erf_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.erf_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.erf_tile_init`. Short init function which configures compute unit for execution of erf_tile. No results.
pub fn erf_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.erf_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.erfc_tile`. Erfc operation. No results.
pub fn erfc_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.erfc_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.erfc_tile_init`. Short init function which configures compute unit for execution of erfc_tile. No results.
pub fn erfc_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.erfc_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.exp2_tile`. Exp2 operation. No results.
pub fn exp2_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.exp2_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.exp2_tile_init`. Short init function which configures compute unit for execution of exp2_tile. No results.
pub fn exp2_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.exp2_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.exp_tile`. Exp operation. No results.
pub fn exp_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, approx: ?*const mlir.Attribute, input_clamping: ?*const mlir.Attribute, iterations: ?*const mlir.Attribute, scale: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    if (approx) |present| attrs.appendAssumeCapacity(.named(ctx, "approx", present));
    if (input_clamping) |present| attrs.appendAssumeCapacity(.named(ctx, "input_clamping", present));
    if (iterations) |present| attrs.appendAssumeCapacity(.named(ctx, "iterations", present));
    if (scale) |present| attrs.appendAssumeCapacity(.named(ctx, "scale", present));
    return make(ctx, "ttkernel.exp_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttkernel.exp_tile_init`. Short init function which configures compute unit for execution of exp_tile. No results.
pub fn exp_tile_init(ctx: *mlir.Context, approx: ?*const mlir.Attribute, scale: ?*const mlir.Attribute, input_clamping: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    if (approx) |present| attrs.appendAssumeCapacity(.named(ctx, "approx", present));
    if (scale) |present| attrs.appendAssumeCapacity(.named(ctx, "scale", present));
    if (input_clamping) |present| attrs.appendAssumeCapacity(.named(ctx, "input_clamping", present));
    return make(ctx, "ttkernel.exp_tile_init", .{
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttkernel.experimental.close_fabric_connections`. CloseFabricConnections. No results.
pub fn experimental_close_fabric_connections(ctx: *mlir.Context, fabric_connection_manager: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.close_fabric_connections", .{
        .operands = .{ .flat = &.{fabric_connection_manager} },
        .location = location,
    });
}

/// `ttkernel.experimental.constant_table_lookup`. Look up an element in an immutable compile-time table. Result types are fixed.
pub fn experimental_constant_table_lookup(ctx: *mlir.Context, index: *const mlir.Value, values: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.constant_table_lookup", .{
        .operands = .{ .flat = &.{index} },
        .results = .{ .flat = &.{mlir.Type.index(ctx)} },
        .attributes = &.{.named(ctx, "values", values)},
        .location = location,
    });
}

/// `ttkernel.experimental.convert_logical_x_to_translated`. ConvertLogicalToTranslatedX. Result types are explicit.
pub fn experimental_convert_logical_x_to_translated(ctx: *mlir.Context, logical_x: *const mlir.Value, virtual_x_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.convert_logical_x_to_translated", .{
        .operands = .{ .flat = &.{logical_x} },
        .results = .{ .flat = &.{virtual_x_type} },
        .location = location,
    });
}

/// `ttkernel.experimental.convert_logical_y_to_translated`. ConvertLogicalToTranslatedY. Result types are explicit.
pub fn experimental_convert_logical_y_to_translated(ctx: *mlir.Context, logical_y: *const mlir.Value, virtual_y_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.convert_logical_y_to_translated", .{
        .operands = .{ .flat = &.{logical_y} },
        .results = .{ .flat = &.{virtual_y_type} },
        .location = location,
    });
}

/// `ttkernel.experimental.create_fabric_connection_manager`. CreateFabricConnectionManager. Result types are fixed.
pub fn experimental_create_fabric_connection_manager(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.create_fabric_connection_manager", .{
        .results = .{ .flat = &.{helpers.parseType(ctx, "!ttkernel.fabric_connection_manager")} },
        .location = location,
    });
}

/// `ttkernel.experimental.fabric_fast_write_any_len`. FabricWriteOp. No results.
pub fn experimental_fabric_fast_write_any_len(ctx: *mlir.Context, fabric_connection_manager: *const mlir.Value, dst_mesh_id: *const mlir.Value, dst_dev_id: *const mlir.Value, dst_addr: *const mlir.Value, src_addr: *const mlir.Value, len_bytes: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.fabric_fast_write_any_len", .{
        .operands = .{ .flat = &.{ fabric_connection_manager, dst_mesh_id, dst_dev_id, dst_addr, src_addr, len_bytes } },
        .location = location,
    });
}

/// `ttkernel.experimental.fabric_mcast_fast_write_any_len`. FabricMulticastWriteOp. No results.
pub fn experimental_fabric_mcast_fast_write_any_len(ctx: *mlir.Context, fabric_connection_manager: *const mlir.Value, dst_mesh_id: *const mlir.Value, dst_dev_id_start: *const mlir.Value, dst_dev_id_end: *const mlir.Value, dst_addr: *const mlir.Value, src_addr: *const mlir.Value, len_bytes: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.fabric_mcast_fast_write_any_len", .{
        .operands = .{ .flat = &.{ fabric_connection_manager, dst_mesh_id, dst_dev_id_start, dst_dev_id_end, dst_addr, src_addr, len_bytes } },
        .location = location,
    });
}

/// `ttkernel.experimental.fabric_mcast_sem_inc`. FabricMulticastSemIncOp. No results.
pub fn experimental_fabric_mcast_sem_inc(ctx: *mlir.Context, fabric_connection_manager: *const mlir.Value, dst_mesh_id: *const mlir.Value, dst_dev_id_start: *const mlir.Value, dst_dev_id_end: *const mlir.Value, semaphore_addr: *const mlir.Value, incr: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.fabric_mcast_sem_inc", .{
        .operands = .{ .flat = &.{ fabric_connection_manager, dst_mesh_id, dst_dev_id_start, dst_dev_id_end, semaphore_addr, incr } },
        .location = location,
    });
}

/// `ttkernel.experimental.fabric_sem_inc`. FabricSemIncOp. No results.
pub fn experimental_fabric_sem_inc(ctx: *mlir.Context, fabric_connection_manager: *const mlir.Value, dst_mesh_id: *const mlir.Value, dst_dev_id: *const mlir.Value, semaphore_addr: *const mlir.Value, incr: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.fabric_sem_inc", .{
        .operands = .{ .flat = &.{ fabric_connection_manager, dst_mesh_id, dst_dev_id, semaphore_addr, incr } },
        .location = location,
    });
}

/// `ttkernel.experimental.fill_arange_tile`. Experimental Write Full Linear Index Tile Op. No results.
pub fn experimental_fill_arange_tile(ctx: *mlir.Context, cb: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.fill_arange_tile", .{
        .operands = .{ .flat = &.{cb} },
        .location = location,
    });
}

/// `ttkernel.experimental.get_device_id_from_logical_mesh_position`. GetDeviceIdFromLogicalMeshPosition. Result types are fixed.
pub fn experimental_get_device_id_from_logical_mesh_position(ctx: *mlir.Context, fcm: *const mlir.Value, position_indices: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(fcm);
    operands.appendSliceAssumeCapacity(position_indices);
    return make(ctx, "ttkernel.experimental.get_device_id_from_logical_mesh_position", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i16)} },
        .location = location,
    });
}

/// `ttkernel.experimental.get_my_device_id`. GetMyDeviceId. Result types are fixed.
pub fn experimental_get_my_device_id(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.get_my_device_id", .{
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i16)} },
        .location = location,
    });
}

/// `ttkernel.experimental.get_my_logical_mesh_position`. GetMyLogicalMeshPosition. Result types are fixed.
pub fn experimental_get_my_logical_mesh_position(ctx: *mlir.Context, fcm: *const mlir.Value, dim: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.get_my_logical_mesh_position", .{
        .operands = .{ .flat = &.{fcm} },
        .results = .{ .flat = &.{mlir.Type.index(ctx)} },
        .attributes = &.{.named(ctx, "dim", dim)},
        .location = location,
    });
}

/// `ttkernel.experimental.matmul_block`. Matmul tiles operation. No results.
pub fn experimental_matmul_block(ctx: *mlir.Context, in0_cb_id: *const mlir.Value, in1_cb_id: *const mlir.Value, in0_tile_idx: *const mlir.Value, in1_tile_idx: *const mlir.Value, dst_tile_idx: *const mlir.Value, transpose: *const mlir.Value, ct_dim: *const mlir.Value, rt_dim: *const mlir.Value, kt_dim: *const mlir.Value, in1_k_stride: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.matmul_block", .{
        .operands = .{ .flat = &.{ in0_cb_id, in1_cb_id, in0_tile_idx, in1_tile_idx, dst_tile_idx, transpose, ct_dim, rt_dim, kt_dim, in1_k_stride } },
        .location = location,
    });
}

/// `ttkernel.experimental.pack_untilize_block`. Experimental PackUntilizeBlockOp call. No results.
pub fn experimental_pack_untilize_block(ctx: *mlir.Context, cb_in: *const mlir.Value, cb_out: *const mlir.Value, block_r: *const mlir.Value, block_c: *const mlir.Value, cols_per_dst_pass: *const mlir.Attribute, total_col_tiles: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.pack_untilize_block", .{
        .operands = .{ .flat = &.{ cb_in, cb_out, block_r, block_c } },
        .attributes = &.{ .named(ctx, "colsPerDstPass", cols_per_dst_pass), .named(ctx, "totalColTiles", total_col_tiles) },
        .location = location,
    });
}

/// `ttkernel.experimental.semaphore_reached`. Test a semaphore completion threshold. Result types are fixed.
pub fn experimental_semaphore_reached(ctx: *mlir.Context, sem_addr: *const mlir.Value, val: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.semaphore_reached", .{
        .operands = .{ .flat = &.{ sem_addr, val } },
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i1)} },
        .location = location,
    });
}

/// `ttkernel.experimental.semaphore_wait`. SemaphoreWait. No results.
pub fn experimental_semaphore_wait(ctx: *mlir.Context, sem_addr: *const mlir.Value, val: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.semaphore_wait", .{
        .operands = .{ .flat = &.{ sem_addr, val } },
        .location = location,
    });
}

/// `ttkernel.experimental.semaphore_wait_min`. SemaphoreWaitMin. No results.
pub fn experimental_semaphore_wait_min(ctx: *mlir.Context, sem_addr: *const mlir.Value, val: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.semaphore_wait_min", .{
        .operands = .{ .flat = &.{ sem_addr, val } },
        .location = location,
    });
}

/// `ttkernel.experimental.setup_fabric_connections`. SetupFabricConnections. No results.
pub fn experimental_setup_fabric_connections(ctx: *mlir.Context, fabric_connection_manager: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.setup_fabric_connections", .{
        .operands = .{ .flat = &.{fabric_connection_manager} },
        .location = location,
    });
}

/// `ttkernel.experimental.tilize_block`. Experimental TilizeBlockOp call. No results.
pub fn experimental_tilize_block(ctx: *mlir.Context, cb_in: *const mlir.Value, cb_out: *const mlir.Value, block_r: *const mlir.Value, block_c: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.tilize_block", .{
        .operands = .{ .flat = &.{ cb_in, cb_out, block_r, block_c } },
        .location = location,
    });
}

/// `ttkernel.experimental.unpack_stall_on_pack`. Stall UNPACK thread until previous PACK write is committed to L1. No results.
pub fn experimental_unpack_stall_on_pack(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.unpack_stall_on_pack", .{
        .location = location,
    });
}

/// `ttkernel.experimental.untilize_block`. Experimental UntilizeBlockOp call. No results.
pub fn experimental_untilize_block(ctx: *mlir.Context, cb_in: *const mlir.Value, cb_out: *const mlir.Value, block_r: *const mlir.Value, block_c: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.untilize_block", .{
        .operands = .{ .flat = &.{ cb_in, cb_out, block_r, block_c } },
        .location = location,
    });
}

/// `ttkernel.experimental.write_col_mask_tile`. Experimental Write Col Mask Tile Op. No results.
pub fn experimental_write_col_mask_tile(ctx: *mlir.Context, valid_cols: *const mlir.Value, cb: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.write_col_mask_tile", .{
        .operands = .{ .flat = &.{ valid_cols, cb } },
        .location = location,
    });
}

/// `ttkernel.experimental.write_row_mask_tile`. Experimental Write Row Mask Tile Op. No results.
pub fn experimental_write_row_mask_tile(ctx: *mlir.Context, valid_rows: *const mlir.Value, cb: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental.write_row_mask_tile", .{
        .operands = .{ .flat = &.{ valid_rows, cb } },
        .location = location,
    });
}

/// `ttkernel.experimental_row_normalization_block`. Normalize one row in a single DST acquisition. No results.
pub fn experimental_row_normalization_block(ctx: *mlir.Context, input_cb: *const mlir.Value, gamma_cb: *const mlir.Value, output_cb: *const mlir.Value, num_tiles: *const mlir.Attribute, scale: *const mlir.Attribute, epsilon: *const mlir.Attribute, has_gamma: *const mlir.Attribute, dtype: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.experimental_row_normalization_block", .{
        .operands = .{ .flat = &.{ input_cb, gamma_cb, output_cb } },
        .attributes = &.{ .named(ctx, "num_tiles", num_tiles), .named(ctx, "scale", scale), .named(ctx, "epsilon", epsilon), .named(ctx, "has_gamma", has_gamma), .named(ctx, "dtype", dtype) },
        .location = location,
    });
}

/// `ttkernel.expm1_tile`. Expm1 operation. No results.
pub fn expm1_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.expm1_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.expm1_tile_init`. Short init function which configures compute unit for execution of expm1_tile. No results.
pub fn expm1_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.expm1_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.fill_tile`. Fill tile with specified value. No results.
pub fn fill_tile(ctx: *mlir.Context, dst_index: *const mlir.Value, value: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.fill_tile", .{
        .operands = .{ .flat = &.{ dst_index, value } },
        .location = location,
    });
}

/// `ttkernel.fill_tile_init`. Init function for fill_tile operation. Refer to documentation for any init function. No results.
pub fn fill_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.fill_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.fill_tile_int`. Fill tile with specified int32 value. No results.
pub fn fill_tile_int(ctx: *mlir.Context, dst_index: *const mlir.Value, value: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.fill_tile_int", .{
        .operands = .{ .flat = &.{ dst_index, value } },
        .location = location,
    });
}

/// `ttkernel.float32_greater`. Compare two float32 values using integer arithmetic. Result types are fixed.
pub fn float32_greater(ctx: *mlir.Context, f32_a: *const mlir.Value, f32_b: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.float32_greater", .{
        .operands = .{ .flat = &.{ f32_a, f32_b } },
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i1)} },
        .location = location,
    });
}

/// `ttkernel.floor_tile`. Floor tile in the DST at specified index. No results.
pub fn floor_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.floor_tile", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.frac_tile`. Frac operation. No results.
pub fn frac_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.frac_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.ge_binary_tile`. Greater-than-or-equal operation between two tiles. No results.
pub fn ge_binary_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.ge_binary_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .location = location,
    });
}

/// `ttkernel.ge_binary_tile_init`. Short init function. No results.
pub fn ge_binary_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.ge_binary_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.gelu_tile`. GELU operation. No results.
pub fn gelu_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.gelu_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.gelu_tile_init`. Short init function which configures compute unit for execution of gelu_tile. No results.
pub fn gelu_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.gelu_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.get_arg_val`. Get runtime arg value. Result types are explicit.
pub fn get_arg_val(ctx: *mlir.Context, arg_index: *const mlir.Value, arg_val_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.get_arg_val", .{
        .operands = .{ .flat = &.{arg_index} },
        .results = .{ .flat = &.{arg_val_type} },
        .location = location,
    });
}

/// `ttkernel.get_common_arg_val`. Get common runtime arg value. Result types are explicit.
pub fn get_common_arg_val(ctx: *mlir.Context, arg_index: *const mlir.Value, arg_val_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.get_common_arg_val", .{
        .operands = .{ .flat = &.{arg_index} },
        .results = .{ .flat = &.{arg_val_type} },
        .location = location,
    });
}

/// `ttkernel.get_compile_time_arg_val`. Get compile-time arg value. Result types are explicit.
pub fn get_compile_time_arg_val(ctx: *mlir.Context, arg_val_type: *const mlir.Type, arg_index: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.get_compile_time_arg_val", .{
        .results = .{ .flat = &.{arg_val_type} },
        .attributes = &.{.named(ctx, "arg_index", arg_index)},
        .location = location,
    });
}

/// `ttkernel.get_dataformat`. Get the data format of a given CB. Result types are fixed.
pub fn get_dataformat(ctx: *mlir.Context, cb: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.get_dataformat", .{
        .operands = .{ .flat = &.{cb} },
        .results = .{ .flat = &.{helpers.parseType(ctx, "!ttkernel.DataFormat")} },
        .location = location,
    });
}

/// `ttkernel.get_dfb_id`. Get the integer index for a dataflow buffer. Result types are fixed.
pub fn get_dfb_id(ctx: *mlir.Context, dfb: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.get_dfb_id", .{
        .operands = .{ .flat = &.{dfb} },
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i32)} },
        .location = location,
    });
}

/// `ttkernel.get_noc_addr`. GetNocAddr. Result types are fixed.
pub fn get_noc_addr(ctx: *mlir.Context, x: *const mlir.Value, y: *const mlir.Value, l1_address: *const mlir.Value, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(x);
    operands.appendAssumeCapacity(y);
    operands.appendAssumeCapacity(l1_address);
    if (noc) |present| operands.appendAssumeCapacity(present);
    return make(ctx, "ttkernel.get_noc_addr", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{helpers.parseType(ctx, "!ttkernel.noc_addr")} },
        .location = location,
    });
}

/// `ttkernel.get_noc_addr_from_bank_id`. GetNocAddrFromBankID. Result types are fixed.
pub fn get_noc_addr_from_bank_id(ctx: *mlir.Context, bank_id: *const mlir.Value, bank_address_offset: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.get_noc_addr_from_bank_id", .{
        .operands = .{ .flat = &.{ bank_id, bank_address_offset } },
        .results = .{ .flat = &.{helpers.parseType(ctx, "!ttkernel.noc_addr")} },
        .location = location,
    });
}

/// `ttkernel.get_noc_multicast_addr`. GetNocMulticastAddr. Result types are fixed.
pub fn get_noc_multicast_addr(ctx: *mlir.Context, noc_x_start: *const mlir.Value, noc_y_start: *const mlir.Value, noc_x_end: *const mlir.Value, noc_y_end: *const mlir.Value, addr: *const mlir.Value, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(noc_x_start);
    operands.appendAssumeCapacity(noc_y_start);
    operands.appendAssumeCapacity(noc_x_end);
    operands.appendAssumeCapacity(noc_y_end);
    operands.appendAssumeCapacity(addr);
    if (noc) |present| operands.appendAssumeCapacity(present);
    return make(ctx, "ttkernel.get_noc_multicast_addr", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{helpers.parseType(ctx, "!ttkernel.noc_addr")} },
        .location = location,
    });
}

/// `ttkernel.get_read_ptr`. GetReadPtr. Result types are fixed.
pub fn get_read_ptr(ctx: *mlir.Context, cb: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.get_read_ptr", .{
        .operands = .{ .flat = &.{cb} },
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i32)} },
        .location = location,
    });
}

/// `ttkernel.get_semaphore`. GetSemaphoreOp. Result types are fixed.
pub fn get_semaphore(ctx: *mlir.Context, semaphore: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.get_semaphore", .{
        .operands = .{ .flat = &.{semaphore} },
        .results = .{ .flat = &.{helpers.parseType(ctx, "!ttkernel.local_semaphore")} },
        .location = location,
    });
}

/// `ttkernel.get_tile_size`. Get the tile size in bytes of a given CB. Result types are fixed.
pub fn get_tile_size(ctx: *mlir.Context, cb: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.get_tile_size", .{
        .operands = .{ .flat = &.{cb} },
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i32)} },
        .location = location,
    });
}

/// `ttkernel.get_write_ptr`. GetWritePtr. Result types are fixed.
pub fn get_write_ptr(ctx: *mlir.Context, cb: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.get_write_ptr", .{
        .operands = .{ .flat = &.{cb} },
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i32)} },
        .location = location,
    });
}

/// `ttkernel.gez_tile`. Greater than or equal to zero tile in the DST at specified index. No results.
pub fn gez_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.gez_tile", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.gez_tile_init`. Init function for gez() operation. Refer to documentation for any init function. No results.
pub fn gez_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.gez_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.gez_tile_int32`. Greater than or equal to zero tile in the DST at specified index. No results.
pub fn gez_tile_int32(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.gez_tile_int32", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.gt_binary_tile`. Greater-than operation between two tiles. No results.
pub fn gt_binary_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.gt_binary_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .location = location,
    });
}

/// `ttkernel.gt_binary_tile_init`. Short init function. No results.
pub fn gt_binary_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.gt_binary_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.gtz_tile`. Greater than zero tile in the DST at specified index. No results.
pub fn gtz_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.gtz_tile", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.gtz_tile_init`. Init function for gtz() operation. Refer to documentation for any init function. No results.
pub fn gtz_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.gtz_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.gtz_tile_int32`. Greater than zero tile in the DST at specified index. No results.
pub fn gtz_tile_int32(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.gtz_tile_int32", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.hardsigmoid_tile`. Hardsigmoid operation. No results.
pub fn hardsigmoid_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.hardsigmoid_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.hardsigmoid_tile_init`. Short init function which configures compute unit for execution of hardsigmoid_tile. No results.
pub fn hardsigmoid_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.hardsigmoid_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.init_sfpu`. Initialization function for SFPU operations. No results.
pub fn init_sfpu(ctx: *mlir.Context, icb: *const mlir.Value, ocb: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.init_sfpu", .{
        .operands = .{ .flat = &.{ icb, ocb } },
        .location = location,
    });
}

/// `ttkernel.invoke_sfpi`. No results.
pub fn invoke_sfpi(ctx: *mlir.Context, region_block: *mlir.Block, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.invoke_sfpi", .{
        .blocks = &.{region_block},
        .location = location,
    });
}

/// `ttkernel.le_binary_tile`. Less-than-or-equal operation between two tiles. No results.
pub fn le_binary_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.le_binary_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .location = location,
    });
}

/// `ttkernel.le_binary_tile_init`. Short init function. No results.
pub fn le_binary_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.le_binary_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.lez_tile`. Less than or equal to zero tile in the DST at specified index. No results.
pub fn lez_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.lez_tile", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.lez_tile_init`. Init function for lez() operation. Refer to documentation for any init function. No results.
pub fn lez_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.lez_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.lez_tile_int32`. Less than or equal to zero tile in the DST at specified index. No results.
pub fn lez_tile_int32(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.lez_tile_int32", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.load_from_l1`. Load a value from local L1. Result types are explicit.
pub fn load_from_l1(ctx: *mlir.Context, l1_ptr: *const mlir.Value, offset: *const mlir.Value, value_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.load_from_l1", .{
        .operands = .{ .flat = &.{ l1_ptr, offset } },
        .results = .{ .flat = &.{value_type} },
        .location = location,
    });
}

/// `ttkernel.log1p_tile`. Log1p operation. No results.
pub fn log1p_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.log1p_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.log1p_tile_init`. Short init function which configures compute unit for execution of log1p_tile. No results.
pub fn log1p_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.log1p_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.log_tile`. Log operation. No results.
pub fn log_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.log_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.log_tile_init`. Short init function which configures compute unit for execution of log_tile. No results.
pub fn log_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.log_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.logical_not_tile`. Logical negation tile in the DST at specified index. No results.
pub fn logical_not_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dtype: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.logical_not_tile", .{
        .operands = .{ .flat = &.{dst0_index} },
        .attributes = &.{.named(ctx, "dtype", dtype)},
        .location = location,
    });
}

/// `ttkernel.logical_not_tile_init`. Init function for logical_not_tile operation. Refer to documentation for any init function. No results.
pub fn logical_not_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.logical_not_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.lt_binary_tile`. Less-than operation between two tiles. No results.
pub fn lt_binary_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.lt_binary_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .location = location,
    });
}

/// `ttkernel.lt_binary_tile_init`. Short init function. No results.
pub fn lt_binary_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.lt_binary_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.ltz_tile`. Less than zero tile in the DST at specified index. No results.
pub fn ltz_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.ltz_tile", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.ltz_tile_init`. Init function for ltz() operation. Refer to documentation for any init function. No results.
pub fn ltz_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.ltz_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.ltz_tile_int32`. Less than zero tile in the DST at specified index. No results.
pub fn ltz_tile_int32(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.ltz_tile_int32", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.matmul_block`. Block matmul operation. No results.
pub fn matmul_block(ctx: *mlir.Context, in0_cb_id: *const mlir.Value, in1_cb_id: *const mlir.Value, in0_tile_idx: *const mlir.Value, in1_tile_idx: *const mlir.Value, dst_tile_idx: *const mlir.Value, transpose: *const mlir.Value, ct_dim: *const mlir.Value, rt_dim: *const mlir.Value, kt_dim: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.matmul_block", .{
        .operands = .{ .flat = &.{ in0_cb_id, in1_cb_id, in0_tile_idx, in1_tile_idx, dst_tile_idx, transpose, ct_dim, rt_dim, kt_dim } },
        .location = location,
    });
}

/// `ttkernel.matmul_tiles`. Matmul tiles operation. No results.
pub fn matmul_tiles(ctx: *mlir.Context, in0_cb_id: *const mlir.Value, in1_cb_id: *const mlir.Value, in0_tile_idx: *const mlir.Value, in1_tile_idx: *const mlir.Value, dst_tile_idx: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.matmul_tiles", .{
        .operands = .{ .flat = &.{ in0_cb_id, in1_cb_id, in0_tile_idx, in1_tile_idx, dst_tile_idx } },
        .location = location,
    });
}

/// `ttkernel.mem_zeros_base`. Op corresponding to MEM_ZEROS_BASE macro in kernels. Result types are fixed.
pub fn mem_zeros_base(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.mem_zeros_base", .{
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i32)} },
        .location = location,
    });
}

/// `ttkernel.mem_zeros_size`. Op corresponding to MEM_ZEROS_SIZE macro in kernels. Result types are fixed.
pub fn mem_zeros_size(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.mem_zeros_size", .{
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i32)} },
        .location = location,
    });
}

/// `ttkernel.mm_block_init`. Matmul init function. No results.
pub fn mm_block_init(ctx: *mlir.Context, in0_cb: *const mlir.Value, in1_cb: *const mlir.Value, out_cb: *const mlir.Value, transpose: *const mlir.Value, ct_dim: *const mlir.Value, rt_dim: *const mlir.Value, kt_dim: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.mm_block_init", .{
        .operands = .{ .flat = &.{ in0_cb, in1_cb, out_cb, transpose, ct_dim, rt_dim, kt_dim } },
        .location = location,
    });
}

/// `ttkernel.mm_block_init_short`. Matmul short init function. No results.
pub fn mm_block_init_short(ctx: *mlir.Context, in0_cb: *const mlir.Value, in1_cb: *const mlir.Value, transpose: *const mlir.Value, ct_dim: *const mlir.Value, rt_dim: *const mlir.Value, kt_dim: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.mm_block_init_short", .{
        .operands = .{ .flat = &.{ in0_cb, in1_cb, transpose, ct_dim, rt_dim, kt_dim } },
        .location = location,
    });
}

/// `ttkernel.mm_init`. Matmul init function. No results.
pub fn mm_init(ctx: *mlir.Context, in0_cb: *const mlir.Value, in1_cb: *const mlir.Value, out_cb: *const mlir.Value, transpose: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.mm_init", .{
        .operands = .{ .flat = &.{ in0_cb, in1_cb, out_cb, transpose } },
        .location = location,
    });
}

/// `ttkernel.mm_init_short`. Matmul short init function. No results.
pub fn mm_init_short(ctx: *mlir.Context, in0_cb: *const mlir.Value, in1_cb: *const mlir.Value, transpose: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.mm_init_short", .{
        .operands = .{ .flat = &.{ in0_cb, in1_cb, transpose } },
        .location = location,
    });
}

/// `ttkernel.mul_binary_tile`. Multiplication operation between two tiles. No results.
pub fn mul_binary_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.mul_binary_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .location = location,
    });
}

/// `ttkernel.mul_binary_tile_init`. Short init function. No results.
pub fn mul_binary_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.mul_binary_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.mul_int_tile`. Integer multiplication operation between two tiles. No results.
pub fn mul_int_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, dtype: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.mul_int_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .attributes = &.{.named(ctx, "dtype", dtype)},
        .location = location,
    });
}

/// `ttkernel.mul_int_tile_init`. Short init function. No results.
pub fn mul_int_tile_init(ctx: *mlir.Context, dtype: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.mul_int_tile_init", .{
        .attributes = &.{.named(ctx, "dtype", dtype)},
        .location = location,
    });
}

/// `ttkernel.mul_tiles`. Mul operation. No results.
pub fn mul_tiles(ctx: *mlir.Context, in0_cb: *const mlir.Value, in1_cb: *const mlir.Value, in0_tile_index: *const mlir.Value, in1_tile_index: *const mlir.Value, dst_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.mul_tiles", .{
        .operands = .{ .flat = &.{ in0_cb, in1_cb, in0_tile_index, in1_tile_index, dst_index } },
        .location = location,
    });
}

/// `ttkernel.mul_tiles_init`. Short init function. No results.
pub fn mul_tiles_init(ctx: *mlir.Context, in0_cb: *const mlir.Value, in1_cb: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.mul_tiles_init", .{
        .operands = .{ .flat = &.{ in0_cb, in1_cb } },
        .location = location,
    });
}

/// `ttkernel.mul_unary_tile`. Multiply by scalar operation. No results.
pub fn mul_unary_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, param: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.mul_unary_tile", .{
        .operands = .{ .flat = &.{ dst0_index, param } },
        .location = location,
    });
}

/// `ttkernel.my_logical_x_`. MyLogicalX. Result types are fixed.
pub fn my_logical_x_(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.my_logical_x_", .{
        .results = .{ .flat = &.{mlir.Type.index(ctx)} },
        .location = location,
    });
}

/// `ttkernel.my_logical_y_`. MyLogicalY. Result types are fixed.
pub fn my_logical_y_(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.my_logical_y_", .{
        .results = .{ .flat = &.{mlir.Type.index(ctx)} },
        .location = location,
    });
}

/// `ttkernel.my_x`. MyX. Result types are fixed.
pub fn my_x(ctx: *mlir.Context, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    if (noc) |present| operands.appendAssumeCapacity(present);
    return make(ctx, "ttkernel.my_x", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{mlir.Type.index(ctx)} },
        .location = location,
    });
}

/// `ttkernel.my_y`. MyY. Result types are fixed.
pub fn my_y(ctx: *mlir.Context, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    if (noc) |present| operands.appendAssumeCapacity(present);
    return make(ctx, "ttkernel.my_y", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{mlir.Type.index(ctx)} },
        .location = location,
    });
}

/// `ttkernel.ne_binary_tile`. Inequality operation between two tiles. No results.
pub fn ne_binary_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.ne_binary_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .location = location,
    });
}

/// `ttkernel.ne_binary_tile_init`. Short init function. No results.
pub fn ne_binary_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.ne_binary_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.negative_tile`. Negative operation. No results.
pub fn negative_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.negative_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.negative_tile_init`. Short init function which configures compute unit for execution of negative_tile. No results.
pub fn negative_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.negative_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.negative_tile_int32`. Negative i32 tile in the DST at specified index. No results.
pub fn negative_tile_int32(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.negative_tile_int32", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.nez_tile`. Not equal to zero tile in the DST at specified index. No results.
pub fn nez_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.nez_tile", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.nez_tile_init`. Init function for nez() operation. Refer to documentation for any init function. No results.
pub fn nez_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.nez_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.nez_tile_int32`. Not equal to zero tile in the DST at specified index. No results.
pub fn nez_tile_int32(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.nez_tile_int32", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.noc_async_atomic_barrier`. NocAsyncAtomicBarrier. No results.
pub fn noc_async_atomic_barrier(ctx: *mlir.Context, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    if (noc) |present| operands.appendAssumeCapacity(present);
    return make(ctx, "ttkernel.noc_async_atomic_barrier", .{
        .operands = .{ .flat = operands.constSlice() },
        .location = location,
    });
}

/// `ttkernel.noc_async_read`. NocAsyncRead. No results.
pub fn noc_async_read(ctx: *mlir.Context, src_core_xy: []const *const mlir.Value, src_bank_id: []const *const mlir.Value, src_address: *const mlir.Value, dst_local_l1_addr: *const mlir.Value, size: *const mlir.Value, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.noc_async_read", .{
        .operands = .{ .variadic = &.{ src_core_xy, src_bank_id, &.{src_address}, &.{dst_local_l1_addr}, &.{size}, if (noc) |present| &.{present} else &.{} } },
        .location = location,
    });
}

/// `ttkernel.noc_async_read_barrier`. NocAsyncReadBarrier. No results.
pub fn noc_async_read_barrier(ctx: *mlir.Context, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    if (noc) |present| operands.appendAssumeCapacity(present);
    return make(ctx, "ttkernel.noc_async_read_barrier", .{
        .operands = .{ .flat = operands.constSlice() },
        .location = location,
    });
}

/// `ttkernel.noc_async_read_barrier_with_trid`. NocAsyncReadBarrierWithTrid. No results.
pub fn noc_async_read_barrier_with_trid(ctx: *mlir.Context, trid: *const mlir.Value, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(trid);
    if (noc) |present| operands.appendAssumeCapacity(present);
    return make(ctx, "ttkernel.noc_async_read_barrier_with_trid", .{
        .operands = .{ .flat = operands.constSlice() },
        .location = location,
    });
}

/// `ttkernel.noc_async_read_one_packet_set_state`. NocAsyncReadOnePacketSetState. No results.
pub fn noc_async_read_one_packet_set_state(ctx: *mlir.Context, src_core_xy: []const *const mlir.Value, src_bank_id: []const *const mlir.Value, src_address: *const mlir.Value, size: *const mlir.Value, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.noc_async_read_one_packet_set_state", .{
        .operands = .{ .variadic = &.{ src_core_xy, src_bank_id, &.{src_address}, &.{size}, if (noc) |present| &.{present} else &.{} } },
        .location = location,
    });
}

/// `ttkernel.noc_async_read_one_packet_with_state`. NocAsyncReadOnePacketWithState. No results.
pub fn noc_async_read_one_packet_with_state(ctx: *mlir.Context, src_core_xy: []const *const mlir.Value, src_bank_id: []const *const mlir.Value, src_address: *const mlir.Value, dst_local_l1_addr: *const mlir.Value, size: *const mlir.Value, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.noc_async_read_one_packet_with_state", .{
        .operands = .{ .variadic = &.{ src_core_xy, src_bank_id, &.{src_address}, &.{dst_local_l1_addr}, &.{size}, if (noc) |present| &.{present} else &.{} } },
        .location = location,
    });
}

/// `ttkernel.noc_async_read_tile`. NocAsyncReadTile. No results.
pub fn noc_async_read_tile(ctx: *mlir.Context, id: *const mlir.Value, addr_gen_struct: *const mlir.Value, dst_local_l1_addr: *const mlir.Value, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(id);
    operands.appendAssumeCapacity(addr_gen_struct);
    operands.appendAssumeCapacity(dst_local_l1_addr);
    if (noc) |present| operands.appendAssumeCapacity(present);
    return make(ctx, "ttkernel.noc_async_read_tile", .{
        .operands = .{ .flat = operands.constSlice() },
        .location = location,
    });
}

/// `ttkernel.noc_async_write`. NocAsyncWrite. No results.
pub fn noc_async_write(ctx: *mlir.Context, src_local_l1_addr: *const mlir.Value, dst_core_xy: []const *const mlir.Value, dst_bank_id: []const *const mlir.Value, dst_address: *const mlir.Value, size: *const mlir.Value, noc: ?*const mlir.Value, posted: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (posted) |present| attrs.appendAssumeCapacity(.named(ctx, "posted", present));
    return make(ctx, "ttkernel.noc_async_write", .{
        .operands = .{ .variadic = &.{ &.{src_local_l1_addr}, dst_core_xy, dst_bank_id, &.{dst_address}, &.{size}, if (noc) |present| &.{present} else &.{} } },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttkernel.noc_async_write_barrier`. NocAsyncWriteBarrier. No results.
pub fn noc_async_write_barrier(ctx: *mlir.Context, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    if (noc) |present| operands.appendAssumeCapacity(present);
    return make(ctx, "ttkernel.noc_async_write_barrier", .{
        .operands = .{ .flat = operands.constSlice() },
        .location = location,
    });
}

/// `ttkernel.noc_async_write_barrier_with_trid`. NocAsyncWriteBarrierWithTrid. No results.
pub fn noc_async_write_barrier_with_trid(ctx: *mlir.Context, trid: *const mlir.Value, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(trid);
    if (noc) |present| operands.appendAssumeCapacity(present);
    return make(ctx, "ttkernel.noc_async_write_barrier_with_trid", .{
        .operands = .{ .flat = operands.constSlice() },
        .location = location,
    });
}

/// `ttkernel.noc_async_write_multicast`. NocAsyncWriteMulticast. No results.
pub fn noc_async_write_multicast(ctx: *mlir.Context, src_local_l1_addr: *const mlir.Value, size: *const mlir.Value, num_dests: *const mlir.Value, dst_noc_x_start: *const mlir.Value, dst_noc_y_start: *const mlir.Value, dst_noc_x_end: *const mlir.Value, dst_noc_y_end: *const mlir.Value, dst_local_l1_addr: *const mlir.Value, noc: ?*const mlir.Value, linked: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(src_local_l1_addr);
    operands.appendAssumeCapacity(size);
    operands.appendAssumeCapacity(num_dests);
    operands.appendAssumeCapacity(dst_noc_x_start);
    operands.appendAssumeCapacity(dst_noc_y_start);
    operands.appendAssumeCapacity(dst_noc_x_end);
    operands.appendAssumeCapacity(dst_noc_y_end);
    operands.appendAssumeCapacity(dst_local_l1_addr);
    if (noc) |present| operands.appendAssumeCapacity(present);
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (linked) |present| attrs.appendAssumeCapacity(.named(ctx, "linked", present));
    return make(ctx, "ttkernel.noc_async_write_multicast", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttkernel.noc_async_write_multicast_loopback_src`. NocAsyncWriteMulticastLoopbackSrc. No results.
pub fn noc_async_write_multicast_loopback_src(ctx: *mlir.Context, src_local_l1_addr: *const mlir.Value, size: *const mlir.Value, num_dests: *const mlir.Value, dst_noc_x_start: *const mlir.Value, dst_noc_y_start: *const mlir.Value, dst_noc_x_end: *const mlir.Value, dst_noc_y_end: *const mlir.Value, dst_local_l1_addr: *const mlir.Value, noc: ?*const mlir.Value, linked: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(src_local_l1_addr);
    operands.appendAssumeCapacity(size);
    operands.appendAssumeCapacity(num_dests);
    operands.appendAssumeCapacity(dst_noc_x_start);
    operands.appendAssumeCapacity(dst_noc_y_start);
    operands.appendAssumeCapacity(dst_noc_x_end);
    operands.appendAssumeCapacity(dst_noc_y_end);
    operands.appendAssumeCapacity(dst_local_l1_addr);
    if (noc) |present| operands.appendAssumeCapacity(present);
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (linked) |present| attrs.appendAssumeCapacity(.named(ctx, "linked", present));
    return make(ctx, "ttkernel.noc_async_write_multicast_loopback_src", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttkernel.noc_async_write_multicast_one_packet`. NocAsyncWriteMulticastOnePacket. No results.
pub fn noc_async_write_multicast_one_packet(ctx: *mlir.Context, src_local_l1_addr: *const mlir.Value, size: *const mlir.Value, num_dests: *const mlir.Value, dst_noc_x_start: *const mlir.Value, dst_noc_y_start: *const mlir.Value, dst_noc_x_end: *const mlir.Value, dst_noc_y_end: *const mlir.Value, dst_local_l1_addr: *const mlir.Value, noc: ?*const mlir.Value, linked: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(src_local_l1_addr);
    operands.appendAssumeCapacity(size);
    operands.appendAssumeCapacity(num_dests);
    operands.appendAssumeCapacity(dst_noc_x_start);
    operands.appendAssumeCapacity(dst_noc_y_start);
    operands.appendAssumeCapacity(dst_noc_x_end);
    operands.appendAssumeCapacity(dst_noc_y_end);
    operands.appendAssumeCapacity(dst_local_l1_addr);
    if (noc) |present| operands.appendAssumeCapacity(present);
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (linked) |present| attrs.appendAssumeCapacity(.named(ctx, "linked", present));
    return make(ctx, "ttkernel.noc_async_write_multicast_one_packet", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttkernel.noc_async_write_one_packet_set_state`. NocAsyncWriteOnePacketSetState. No results.
pub fn noc_async_write_one_packet_set_state(ctx: *mlir.Context, dst_noc_addr: *const mlir.Value, size: *const mlir.Value, noc: ?*const mlir.Value, posted: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(dst_noc_addr);
    operands.appendAssumeCapacity(size);
    if (noc) |present| operands.appendAssumeCapacity(present);
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (posted) |present| attrs.appendAssumeCapacity(.named(ctx, "posted", present));
    return make(ctx, "ttkernel.noc_async_write_one_packet_set_state", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttkernel.noc_async_write_one_packet_with_state`. NocAsyncWriteOnePacketWithState. No results.
pub fn noc_async_write_one_packet_with_state(ctx: *mlir.Context, src_local_l1_addr: *const mlir.Value, dst_local_l1_addr: *const mlir.Value, noc: ?*const mlir.Value, posted: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(src_local_l1_addr);
    operands.appendAssumeCapacity(dst_local_l1_addr);
    if (noc) |present| operands.appendAssumeCapacity(present);
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (posted) |present| attrs.appendAssumeCapacity(.named(ctx, "posted", present));
    return make(ctx, "ttkernel.noc_async_write_one_packet_with_state", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttkernel.noc_async_write_one_packet_with_trid`. NocAsyncWriteOnePacketWithTrid. No results.
pub fn noc_async_write_one_packet_with_trid(ctx: *mlir.Context, src_local_l1_addr: *const mlir.Value, dst_core_xy: []const *const mlir.Value, dst_bank_id: []const *const mlir.Value, dst_address: *const mlir.Value, size: *const mlir.Value, trid: *const mlir.Value, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.noc_async_write_one_packet_with_trid", .{
        .operands = .{ .variadic = &.{ &.{src_local_l1_addr}, dst_core_xy, dst_bank_id, &.{dst_address}, &.{size}, &.{trid}, if (noc) |present| &.{present} else &.{} } },
        .location = location,
    });
}

/// `ttkernel.noc_async_write_tile`. NocAsyncWriteTile. No results.
pub fn noc_async_write_tile(ctx: *mlir.Context, id: *const mlir.Value, addr_gen_struct: *const mlir.Value, src_local_l1_addr: *const mlir.Value, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(id);
    operands.appendAssumeCapacity(addr_gen_struct);
    operands.appendAssumeCapacity(src_local_l1_addr);
    if (noc) |present| operands.appendAssumeCapacity(present);
    return make(ctx, "ttkernel.noc_async_write_tile", .{
        .operands = .{ .flat = operands.constSlice() },
        .location = location,
    });
}

/// `ttkernel.noc_async_writes_flushed`. NocAsyncWritesFlushed. No results.
pub fn noc_async_writes_flushed(ctx: *mlir.Context, noc: ?*const mlir.Value, posted: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    if (noc) |present| operands.appendAssumeCapacity(present);
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (posted) |present| attrs.appendAssumeCapacity(.named(ctx, "posted", present));
    return make(ctx, "ttkernel.noc_async_writes_flushed", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttkernel.noc_inline_dw_write`. NocInlineDwWrite. No results.
pub fn noc_inline_dw_write(ctx: *mlir.Context, dst_noc_x: *const mlir.Value, dst_noc_y: *const mlir.Value, dst_address: *const mlir.Value, val: *const mlir.Value, byte_enable: *const mlir.Value, noc: ?*const mlir.Value, posted: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(dst_noc_x);
    operands.appendAssumeCapacity(dst_noc_y);
    operands.appendAssumeCapacity(dst_address);
    operands.appendAssumeCapacity(val);
    operands.appendAssumeCapacity(byte_enable);
    if (noc) |present| operands.appendAssumeCapacity(present);
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (posted) |present| attrs.appendAssumeCapacity(.named(ctx, "posted", present));
    return make(ctx, "ttkernel.noc_inline_dw_write", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttkernel.noc_semaphore_inc`. NocSemaphoreInc. No results.
pub fn noc_semaphore_inc(ctx: *mlir.Context, addr: *const mlir.Value, incr: *const mlir.Value, noc: ?*const mlir.Value, posted: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(addr);
    operands.appendAssumeCapacity(incr);
    if (noc) |present| operands.appendAssumeCapacity(present);
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (posted) |present| attrs.appendAssumeCapacity(.named(ctx, "posted", present));
    return make(ctx, "ttkernel.noc_semaphore_inc", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttkernel.noc_semaphore_inc_multicast`. NocSemaphoreIncMulticast. No results.
pub fn noc_semaphore_inc_multicast(ctx: *mlir.Context, addr: *const mlir.Value, incr: *const mlir.Value, num_dests: *const mlir.Value, noc: ?*const mlir.Value, posted: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(addr);
    operands.appendAssumeCapacity(incr);
    operands.appendAssumeCapacity(num_dests);
    if (noc) |present| operands.appendAssumeCapacity(present);
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (posted) |present| attrs.appendAssumeCapacity(.named(ctx, "posted", present));
    return make(ctx, "ttkernel.noc_semaphore_inc_multicast", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttkernel.noc_semaphore_set`. NocSemaphoreSet. No results.
pub fn noc_semaphore_set(ctx: *mlir.Context, sem_addr: *const mlir.Value, val: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.noc_semaphore_set", .{
        .operands = .{ .flat = &.{ sem_addr, val } },
        .location = location,
    });
}

/// `ttkernel.noc_semaphore_set_multicast`. NocSemaphoreSetMulticast. No results.
pub fn noc_semaphore_set_multicast(ctx: *mlir.Context, src_local_l1_addr: *const mlir.Value, dst_noc_addr_multicast: *const mlir.Value, num_dests: *const mlir.Value, linked: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (linked) |present| attrs.appendAssumeCapacity(.named(ctx, "linked", present));
    return make(ctx, "ttkernel.noc_semaphore_set_multicast", .{
        .operands = .{ .flat = &.{ src_local_l1_addr, dst_noc_addr_multicast, num_dests } },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttkernel.noc_semaphore_set_multicast_loopback_src`. NocSemaphoreSetMulticastLoopback. No results.
pub fn noc_semaphore_set_multicast_loopback_src(ctx: *mlir.Context, src_local_l1_addr: *const mlir.Value, dst_noc_addr_multicast: *const mlir.Value, num_dests: *const mlir.Value, linked: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (linked) |present| attrs.appendAssumeCapacity(.named(ctx, "linked", present));
    return make(ctx, "ttkernel.noc_semaphore_set_multicast_loopback_src", .{
        .operands = .{ .flat = &.{ src_local_l1_addr, dst_noc_addr_multicast, num_dests } },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttkernel.opaque_call`. Call an external C/C++ function declared in a user header. Result types are explicit.
pub fn opaque_call(ctx: *mlir.Context, arg_operands: []const *const mlir.Value, result_types: []const *const mlir.Type, callee: *const mlir.Attribute, header: *const mlir.Attribute, template_args: ?*const mlir.Attribute, unsigned_arg_indices: ?*const mlir.Attribute, dfb_resource_indices: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendSliceAssumeCapacity(arg_operands);
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "callee", callee));
    attrs.appendAssumeCapacity(.named(ctx, "header", header));
    if (template_args) |present| attrs.appendAssumeCapacity(.named(ctx, "template_args", present));
    if (unsigned_arg_indices) |present| attrs.appendAssumeCapacity(.named(ctx, "unsigned_arg_indices", present));
    if (dfb_resource_indices) |present| attrs.appendAssumeCapacity(.named(ctx, "dfb_resource_indices", present));
    return make(ctx, "ttkernel.opaque_call", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = result_types },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttkernel.pack_reconfig_data_format`. Reconfigure packer data format for the given output CB. No results.
pub fn pack_reconfig_data_format(ctx: *mlir.Context, out_cb: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.pack_reconfig_data_format", .{
        .operands = .{ .flat = &.{out_cb} },
        .location = location,
    });
}

/// `ttkernel.pack_reconfig_l1_acc`. Reconfigure packer to L1 accumulation mode. No results.
pub fn pack_reconfig_l1_acc(ctx: *mlir.Context, l1_acc_en: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.pack_reconfig_l1_acc", .{
        .operands = .{ .flat = &.{l1_acc_en} },
        .location = location,
    });
}

/// `ttkernel.pack_tile`. PackTile op. No results.
pub fn pack_tile(ctx: *mlir.Context, dst_index: *const mlir.Value, out_cb: *const mlir.Value, out_index: *const mlir.Value, out_of_order: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (out_of_order) |present| attrs.appendAssumeCapacity(.named(ctx, "out_of_order", present));
    return make(ctx, "ttkernel.pack_tile", .{
        .operands = .{ .flat = &.{ dst_index, out_cb, out_index } },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttkernel.pack_tile_block`. PackTileBlock op. No results.
pub fn pack_tile_block(ctx: *mlir.Context, dst_index: *const mlir.Value, out_cb: *const mlir.Value, ntiles: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.pack_tile_block", .{
        .operands = .{ .flat = &.{ dst_index, out_cb, ntiles } },
        .location = location,
    });
}

/// `ttkernel.pack_untilize_init`. PackUntilizeInitOp call. No results.
pub fn pack_untilize_init(ctx: *mlir.Context, cb_in: *const mlir.Value, cb_out: *const mlir.Value, cols_per_dst_pass: ?*const mlir.Attribute, total_col_tiles: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (cols_per_dst_pass) |present| attrs.appendAssumeCapacity(.named(ctx, "colsPerDstPass", present));
    if (total_col_tiles) |present| attrs.appendAssumeCapacity(.named(ctx, "totalColTiles", present));
    return make(ctx, "ttkernel.pack_untilize_init", .{
        .operands = .{ .flat = &.{ cb_in, cb_out } },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttkernel.pack_untilize_uninit`. PackUntilizeUninitOp call. No results.
pub fn pack_untilize_uninit(ctx: *mlir.Context, cb_out: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.pack_untilize_uninit", .{
        .operands = .{ .flat = &.{cb_out} },
        .location = location,
    });
}

/// `ttkernel.pack_waited_tile`. Replace a tile in a waited full-ring DFB transaction. No results.
pub fn pack_waited_tile(ctx: *mlir.Context, dst_index: *const mlir.Value, out_cb: *const mlir.Value, out_index: *const mlir.Value, out_of_order: ?*const mlir.Attribute, acquired_tiles: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (out_of_order) |present| attrs.appendAssumeCapacity(.named(ctx, "out_of_order", present));
    attrs.appendAssumeCapacity(.named(ctx, "acquired_tiles", acquired_tiles));
    return make(ctx, "ttkernel.pack_waited_tile", .{
        .operands = .{ .flat = &.{ dst_index, out_cb, out_index } },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `ttkernel.power_binary_tile`. Power operation between two tiles. No results.
pub fn power_binary_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.power_binary_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .location = location,
    });
}

/// `ttkernel.power_binary_tile_init`. Short init function which configures compute unit for execution of power_binary_tile. No results.
pub fn power_binary_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.power_binary_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.power_tile`. Power by scalar operation. No results.
pub fn power_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, param: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.power_tile", .{
        .operands = .{ .flat = &.{ dst0_index, param } },
        .location = location,
    });
}

/// `ttkernel.power_tile_init`. Init function for power_tile operation. No results.
pub fn power_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.power_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.rand_tile`. Rand tile operation. No results.
pub fn rand_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, from: *const mlir.Value, scale: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.rand_tile", .{
        .operands = .{ .flat = &.{ tile_index, from, scale } },
        .location = location,
    });
}

/// `ttkernel.rand_tile_init`. Init function for rand_tile operation. Refer to documentation for any init function. No results.
pub fn rand_tile_init(ctx: *mlir.Context, seed: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.rand_tile_init", .{
        .operands = .{ .flat = &.{seed} },
        .location = location,
    });
}

/// `ttkernel.recip_tile`. Recip tile in the DST at specified index. No results.
pub fn recip_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.recip_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.recip_tile_init`. Init function for recip_tile operation. Refer to documentation for any init function. No results.
pub fn recip_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.recip_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.reduce_init`. Init function. No results.
pub fn reduce_init(ctx: *mlir.Context, in_cb: *const mlir.Value, scaling_cb: *const mlir.Value, out_cb: *const mlir.Value, reduce_type: *const mlir.Attribute, reduce_dim: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.reduce_init", .{
        .operands = .{ .flat = &.{ in_cb, scaling_cb, out_cb } },
        .attributes = &.{ .named(ctx, "reduce_type", reduce_type), .named(ctx, "reduce_dim", reduce_dim) },
        .location = location,
    });
}

/// `ttkernel.reduce_tile`. Reduce operation. No results.
pub fn reduce_tile(ctx: *mlir.Context, in_cb: *const mlir.Value, scaling_cb: *const mlir.Value, in_tile_index: *const mlir.Value, scaling_tile_index: *const mlir.Value, dst_index: *const mlir.Value, reduce_type: *const mlir.Attribute, reduce_dim: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.reduce_tile", .{
        .operands = .{ .flat = &.{ in_cb, scaling_cb, in_tile_index, scaling_tile_index, dst_index } },
        .attributes = &.{ .named(ctx, "reduce_type", reduce_type), .named(ctx, "reduce_dim", reduce_dim) },
        .location = location,
    });
}

/// `ttkernel.reduce_uninit`. Init function for reduce_uninit operation. No results.
pub fn reduce_uninit(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.reduce_uninit", .{
        .location = location,
    });
}

/// `ttkernel.reinterpret_cast`. CastToL1Ptr. Result types are explicit.
pub fn reinterpret_cast(ctx: *mlir.Context, addr: *const mlir.Value, l1_ptr_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.reinterpret_cast", .{
        .operands = .{ .flat = &.{addr} },
        .results = .{ .flat = &.{l1_ptr_type} },
        .location = location,
    });
}

/// `ttkernel.relu_tile`. Relu operation. No results.
pub fn relu_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.relu_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.relu_tile_init`. Short init function which configures compute unit for execution of relu_tile(_int32). No results.
pub fn relu_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.relu_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.relu_tile_int32`. Relu operation (for int32 type). No results.
pub fn relu_tile_int32(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.relu_tile_int32", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.remote_sram_write_u32`. RemoteSramWriteU32. No results.
pub fn remote_sram_write_u32(ctx: *mlir.Context, src_sram_addr: *const mlir.Value, dst_remote_addr: *const mlir.Value, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(src_sram_addr);
    operands.appendAssumeCapacity(dst_remote_addr);
    if (noc) |present| operands.appendAssumeCapacity(present);
    return make(ctx, "ttkernel.remote_sram_write_u32", .{
        .operands = .{ .flat = operands.constSlice() },
        .location = location,
    });
}

/// `ttkernel.rounding_op_tile_init`. Init function for ceil/floor/round_tile operation. Refer to documentation for any init function. No results.
pub fn rounding_op_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.rounding_op_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.routing_plane.atomic_inc`. Increment a remote semaphore through one fabric connection. No results.
pub fn routing_plane_atomic_inc(ctx: *mlir.Context, manager: *const mlir.Value, route_id: *const mlir.Value, connection_index: *const mlir.Value, destination_device_id: *const mlir.Value, destination_mesh_id: *const mlir.Value, destination_hop_count: *const mlir.Value, semaphore_address: *const mlir.Value, increment: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.routing_plane.atomic_inc", .{
        .operands = .{ .flat = &.{ manager, route_id, connection_index, destination_device_id, destination_mesh_id, destination_hop_count, semaphore_address, increment } },
        .location = location,
    });
}

/// `ttkernel.routing_plane.close_connections`. Close routing-plane connections. No results.
pub fn routing_plane_close_connections(ctx: *mlir.Context, manager: *const mlir.Value, connection_count: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.routing_plane.close_connections", .{
        .operands = .{ .flat = &.{ manager, connection_count } },
        .location = location,
    });
}

/// `ttkernel.routing_plane.create_connection_manager`. Create a routing-plane connection manager. Result types are fixed.
pub fn routing_plane_create_connection_manager(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.routing_plane.create_connection_manager", .{
        .results = .{ .flat = &.{helpers.parseType(ctx, "!ttkernel.routing_plane_connection_manager")} },
        .location = location,
    });
}

/// `ttkernel.routing_plane.fused_write_atomic_inc`. Write a payload and increment a remote semaphore. No results.
pub fn routing_plane_fused_write_atomic_inc(ctx: *mlir.Context, manager: *const mlir.Value, route_id: *const mlir.Value, connection_index: *const mlir.Value, destination_device_id: *const mlir.Value, destination_mesh_id: *const mlir.Value, destination_hop_count: *const mlir.Value, source_address: *const mlir.Value, size_bytes: *const mlir.Value, destination_address: *const mlir.Value, semaphore_address: *const mlir.Value, increment: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.routing_plane.fused_write_atomic_inc", .{
        .operands = .{ .flat = &.{ manager, route_id, connection_index, destination_device_id, destination_mesh_id, destination_hop_count, source_address, size_bytes, destination_address, semaphore_address, increment } },
        .location = location,
    });
}

/// `ttkernel.routing_plane.open_connections`. Open routing-plane connections from runtime arguments. Result types are fixed.
pub fn routing_plane_open_connections(ctx: *mlir.Context, manager: *const mlir.Value, connection_count: *const mlir.Value, runtime_arg_base: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.routing_plane.open_connections", .{
        .operands = .{ .flat = &.{ manager, connection_count, runtime_arg_base } },
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i32)} },
        .location = location,
    });
}

/// `ttkernel.routing_plane.scatter_write`. Scatter contiguous source chunks through one fabric connection. No results.
pub fn routing_plane_scatter_write(ctx: *mlir.Context, manager: *const mlir.Value, route_id: *const mlir.Value, connection_index: *const mlir.Value, destination_device_id: *const mlir.Value, destination_mesh_id: *const mlir.Value, destination_hop_count: *const mlir.Value, source_address: *const mlir.Value, chunk_size_bytes: *const mlir.Value, chunk_count: *const mlir.Value, destination_address0: *const mlir.Value, destination_address1: *const mlir.Value, destination_address2: *const mlir.Value, destination_address3: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.routing_plane.scatter_write", .{
        .operands = .{ .flat = &.{ manager, route_id, connection_index, destination_device_id, destination_mesh_id, destination_hop_count, source_address, chunk_size_bytes, chunk_count, destination_address0, destination_address1, destination_address2, destination_address3 } },
        .location = location,
    });
}

/// `ttkernel.routing_plane.write`. Write a payload through one fabric connection. No results.
pub fn routing_plane_write(ctx: *mlir.Context, manager: *const mlir.Value, route_id: *const mlir.Value, connection_index: *const mlir.Value, destination_device_id: *const mlir.Value, destination_mesh_id: *const mlir.Value, destination_hop_count: *const mlir.Value, source_address: *const mlir.Value, size_bytes: *const mlir.Value, destination_address: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.routing_plane.write", .{
        .operands = .{ .flat = &.{ manager, route_id, connection_index, destination_device_id, destination_mesh_id, destination_hop_count, source_address, size_bytes, destination_address } },
        .location = location,
    });
}

/// `ttkernel.rsqrt_tile`. Rsqrt operation. No results.
pub fn rsqrt_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.rsqrt_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.rsqrt_tile_init`. Short init function which configures compute unit for execution of rsqrt_tile. No results.
pub fn rsqrt_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.rsqrt_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.selu_tile`. Selu operation. No results.
pub fn selu_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, scale: *const mlir.Value, alpha: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.selu_tile", .{
        .operands = .{ .flat = &.{ tile_index, scale, alpha } },
        .location = location,
    });
}

/// `ttkernel.selu_tile_init`. Short init function which configures compute unit for execution of selu_tile. No results.
pub fn selu_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.selu_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.sfpu_reduce`. SFPU reduce operation on a tile in DST. No results.
pub fn sfpu_reduce(ctx: *mlir.Context, dst_index: *const mlir.Value, reduce_type: *const mlir.Attribute, data_format: *const mlir.Attribute, reduce_dim: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.sfpu_reduce", .{
        .operands = .{ .flat = &.{dst_index} },
        .attributes = &.{ .named(ctx, "reduce_type", reduce_type), .named(ctx, "data_format", data_format), .named(ctx, "reduce_dim", reduce_dim) },
        .location = location,
    });
}

/// `ttkernel.sfpu_reduce_init`. Init for sfpu_reduce. No results.
pub fn sfpu_reduce_init(ctx: *mlir.Context, reduce_type: *const mlir.Attribute, data_format: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.sfpu_reduce_init", .{
        .attributes = &.{ .named(ctx, "reduce_type", reduce_type), .named(ctx, "data_format", data_format) },
        .location = location,
    });
}

/// `ttkernel.sigmoid_tile`. Sigmoid operation. No results.
pub fn sigmoid_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.sigmoid_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.sigmoid_tile_init`. Short init function which configures compute unit for execution of sigmoid_tile. No results.
pub fn sigmoid_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.sigmoid_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.sign_tile`. Sign operation. No results.
pub fn sign_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.sign_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.sign_tile_init`. Init function for sign_tile operation. Refer to documentation for any init function. No results.
pub fn sign_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.sign_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.signbit_tile`. Signbit operation. No results.
pub fn signbit_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.signbit_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.signbit_tile_init`. Short init function which configures compute unit for execution of signbit_tile. No results.
pub fn signbit_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.signbit_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.silu_tile`. Silu operation. No results.
pub fn silu_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.silu_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.silu_tile_init`. Short init function which configures compute unit for execution of silu_tile. No results.
pub fn silu_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.silu_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.sin_tile`. Sine tile in the DST at specified index. No results.
pub fn sin_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.sin_tile", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.sin_tile_init`. Init function for sin_tile operation. Refer to documentation for any init function. No results.
pub fn sin_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.sin_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.softsign_tile`. Softsign operation. No results.
pub fn softsign_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.softsign_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.softsign_tile_init`. Short init function which configures compute unit for execution of softsign_tile. No results.
pub fn softsign_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.softsign_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.sqrt_tile`. Sqrt operation. No results.
pub fn sqrt_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.sqrt_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.sqrt_tile_init`. Short init function which configures compute unit for execution of sqrt_tile. No results.
pub fn sqrt_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.sqrt_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.square_tile`. Square operation. No results.
pub fn square_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.square_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.square_tile_init`. Short init function which configures compute unit for execution of square_tile. No results.
pub fn square_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.square_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.store_to_l1`. StoreToL1. No results.
pub fn store_to_l1(ctx: *mlir.Context, value: *const mlir.Value, l1_ptr: *const mlir.Value, offset: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.store_to_l1", .{
        .operands = .{ .flat = &.{ value, l1_ptr, offset } },
        .location = location,
    });
}

/// `ttkernel.sub_binary_tile`. Subtraction operation between two tiles. No results.
pub fn sub_binary_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.sub_binary_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .location = location,
    });
}

/// `ttkernel.sub_binary_tile_init`. Short init function. No results.
pub fn sub_binary_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.sub_binary_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.sub_int_tile`. Integer subtraction operation between two tiles. No results.
pub fn sub_int_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, dst1_index: *const mlir.Value, odst_index: *const mlir.Value, dtype: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.sub_int_tile", .{
        .operands = .{ .flat = &.{ dst0_index, dst1_index, odst_index } },
        .attributes = &.{.named(ctx, "dtype", dtype)},
        .location = location,
    });
}

/// `ttkernel.sub_int_tile_init`. Short init function. No results.
pub fn sub_int_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.sub_int_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.sub_tiles`. Sub operation. No results.
pub fn sub_tiles(ctx: *mlir.Context, in0_cb: *const mlir.Value, in1_cb: *const mlir.Value, in0_tile_index: *const mlir.Value, in1_tile_index: *const mlir.Value, dst_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.sub_tiles", .{
        .operands = .{ .flat = &.{ in0_cb, in1_cb, in0_tile_index, in1_tile_index, dst_index } },
        .location = location,
    });
}

/// `ttkernel.sub_tiles_init`. Short init function. No results.
pub fn sub_tiles_init(ctx: *mlir.Context, in0_cb: *const mlir.Value, in1_cb: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.sub_tiles_init", .{
        .operands = .{ .flat = &.{ in0_cb, in1_cb } },
        .location = location,
    });
}

/// `ttkernel.sub_unary_tile`. Subtract by scalar operation. No results.
pub fn sub_unary_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, param: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.sub_unary_tile", .{
        .operands = .{ .flat = &.{ dst0_index, param } },
        .location = location,
    });
}

/// `ttkernel.sub_unary_tile_int32`. Subtract by int32 scalar operation. No results.
pub fn sub_unary_tile_int32(ctx: *mlir.Context, dst0_index: *const mlir.Value, param: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.sub_unary_tile_int32", .{
        .operands = .{ .flat = &.{ dst0_index, param } },
        .location = location,
    });
}

/// `ttkernel.tan_tile`. Tan operation. No results.
pub fn tan_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.tan_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.tan_tile_init`. Short init function which configures compute unit for execution of tan_tile. No results.
pub fn tan_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.tan_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.tanh_tile`. Tanh operation. No results.
pub fn tanh_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.tanh_tile", .{
        .operands = .{ .flat = &.{dst0_index} },
        .location = location,
    });
}

/// `ttkernel.tanh_tile_init`. Short init function which configures compute unit for execution of tanh_tile. No results.
pub fn tanh_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.tanh_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.tensor_accessor.get_bank_and_offset`. TensorAccessor's get_bank_and_offset. Result types are fixed.
pub fn tensor_accessor_get_bank_and_offset(ctx: *mlir.Context, tensor_accessor: *const mlir.Value, page_id: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.tensor_accessor.get_bank_and_offset", .{
        .operands = .{ .flat = &.{ tensor_accessor, page_id } },
        .results = .{ .flat = &.{helpers.parseType(ctx, "!ttkernel.PageMapping")} },
        .location = location,
    });
}

/// `ttkernel.tensor_accessor.get_noc_addr`. TensorAccessor's get_noc_addr. Result types are fixed.
pub fn tensor_accessor_get_noc_addr(ctx: *mlir.Context, tensor_accessor: *const mlir.Value, id: *const mlir.Value, offset: *const mlir.Value, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(tensor_accessor);
    operands.appendAssumeCapacity(id);
    operands.appendAssumeCapacity(offset);
    if (noc) |present| operands.appendAssumeCapacity(present);
    return make(ctx, "ttkernel.tensor_accessor.get_noc_addr", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{helpers.parseType(ctx, "!ttkernel.noc_addr")} },
        .location = location,
    });
}

/// `ttkernel.tensor_accessor.get_shard_noc_addr`. TensorAccessor's get_shard_noc_addr. Result types are fixed.
pub fn tensor_accessor_get_shard_noc_addr(ctx: *mlir.Context, tensor_accessor: *const mlir.Value, shard_id: *const mlir.Value, offset: *const mlir.Value, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(tensor_accessor);
    operands.appendAssumeCapacity(shard_id);
    operands.appendAssumeCapacity(offset);
    if (noc) |present| operands.appendAssumeCapacity(present);
    return make(ctx, "ttkernel.tensor_accessor.get_shard_noc_addr", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i32)} },
        .location = location,
    });
}

/// `ttkernel.tensor_accessor.is_local_addr`. TensorAccessor's is_local_addr. Result types are fixed.
pub fn tensor_accessor_is_local_addr(ctx: *mlir.Context, tensor_accessor: *const mlir.Value, virtual_x: *const mlir.Value, virtual_y: *const mlir.Value, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(tensor_accessor);
    operands.appendAssumeCapacity(virtual_x);
    operands.appendAssumeCapacity(virtual_y);
    if (noc) |present| operands.appendAssumeCapacity(present);
    return make(ctx, "ttkernel.tensor_accessor.is_local_addr", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i1)} },
        .location = location,
    });
}

/// `ttkernel.tensor_accessor.is_local_bank`. TensorAccessor's is_local_bank. Result types are fixed.
pub fn tensor_accessor_is_local_bank(ctx: *mlir.Context, tensor_accessor: *const mlir.Value, id: *const mlir.Value, offset: *const mlir.Value, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(tensor_accessor);
    operands.appendAssumeCapacity(id);
    operands.appendAssumeCapacity(offset);
    if (noc) |present| operands.appendAssumeCapacity(present);
    return make(ctx, "ttkernel.tensor_accessor.is_local_bank", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i1)} },
        .location = location,
    });
}

/// `ttkernel.tensor_accessor.is_local_page`. TensorAccessor's is_local_page. Result types are fixed.
pub fn tensor_accessor_is_local_page(ctx: *mlir.Context, tensor_accessor: *const mlir.Value, page_id: *const mlir.Value, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(tensor_accessor);
    operands.appendAssumeCapacity(page_id);
    if (noc) |present| operands.appendAssumeCapacity(present);
    return make(ctx, "ttkernel.tensor_accessor.is_local_page", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i1)} },
        .location = location,
    });
}

/// `ttkernel.tensor_accessor.is_local_shard`. TensorAccessor's is_local_shard. Result types are fixed.
pub fn tensor_accessor_is_local_shard(ctx: *mlir.Context, tensor_accessor: *const mlir.Value, shard_id: *const mlir.Value, noc: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendAssumeCapacity(tensor_accessor);
    operands.appendAssumeCapacity(shard_id);
    if (noc) |present| operands.appendAssumeCapacity(present);
    return make(ctx, "ttkernel.tensor_accessor.is_local_shard", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{mlir.Type.int(ctx, .i1)} },
        .location = location,
    });
}

/// `ttkernel.tile_regs_acquire`. tile_regs_acquire. No results.
pub fn tile_regs_acquire(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.tile_regs_acquire", .{
        .location = location,
    });
}

/// `ttkernel.tile_regs_commit`. tile_regs_commit. No results.
pub fn tile_regs_commit(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.tile_regs_commit", .{
        .location = location,
    });
}

/// `ttkernel.tile_regs_release`. tile_regs_release. No results.
pub fn tile_regs_release(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.tile_regs_release", .{
        .location = location,
    });
}

/// `ttkernel.tile_regs_wait`. tile_regs_wait. No results.
pub fn tile_regs_wait(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.tile_regs_wait", .{
        .location = location,
    });
}

/// `ttkernel.tilize_block`. TilizeBlockOp call. No results.
pub fn tilize_block(ctx: *mlir.Context, cb_in: *const mlir.Value, num_tiles: *const mlir.Value, cb_out: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.tilize_block", .{
        .operands = .{ .flat = &.{ cb_in, num_tiles, cb_out } },
        .location = location,
    });
}

/// `ttkernel.tilize_init`. TilizeInitOp call. No results.
pub fn tilize_init(ctx: *mlir.Context, cb_in: *const mlir.Value, num_tiles: *const mlir.Value, cb_out: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.tilize_init", .{
        .operands = .{ .flat = &.{ cb_in, num_tiles, cb_out } },
        .location = location,
    });
}

/// `ttkernel.tilize_uninit`. TilizeUninitOp call. No results.
pub fn tilize_uninit(ctx: *mlir.Context, cb_i: *const mlir.Value, cb_out: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.tilize_uninit", .{
        .operands = .{ .flat = &.{ cb_i, cb_out } },
        .location = location,
    });
}

/// `ttkernel.topk_local_sort`. TopK local sort operation. No results.
pub fn topk_local_sort(ctx: *mlir.Context, idst: *const mlir.Value, idir: *const mlir.Value, i_end_phase: *const mlir.Value, i_start_phase: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.topk_local_sort", .{
        .operands = .{ .flat = &.{ idst, idir, i_end_phase, i_start_phase } },
        .location = location,
    });
}

/// `ttkernel.topk_merge`. TopK merge operation. No results.
pub fn topk_merge(ctx: *mlir.Context, idst: *const mlir.Value, m_iter: *const mlir.Value, k: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.topk_merge", .{
        .operands = .{ .flat = &.{ idst, m_iter, k } },
        .location = location,
    });
}

/// `ttkernel.topk_rebuild`. TopK rebuild operation. No results.
pub fn topk_rebuild(ctx: *mlir.Context, idst: *const mlir.Value, idir: *const mlir.Value, m_iter: *const mlir.Value, k: *const mlir.Value, logk: *const mlir.Value, skip_second: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.topk_rebuild", .{
        .operands = .{ .flat = &.{ idst, idir, m_iter, k, logk, skip_second } },
        .location = location,
    });
}

/// `ttkernel.topk_tile_init`. TopK tile init function. No results.
pub fn topk_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.topk_tile_init", .{
        .location = location,
    });
}

/// `ttkernel.transpose_wh_init`. TransposeInitOp call. No results.
pub fn transpose_wh_init(ctx: *mlir.Context, cb_in: *const mlir.Value, cb_out: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.transpose_wh_init", .{
        .operands = .{ .flat = &.{ cb_in, cb_out } },
        .location = location,
    });
}

/// `ttkernel.transpose_wh_tile`. Transpose WH tile operation. No results.
pub fn transpose_wh_tile(ctx: *mlir.Context, icb: *const mlir.Value, itile: *const mlir.Value, idst: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.transpose_wh_tile", .{
        .operands = .{ .flat = &.{ icb, itile, idst } },
        .location = location,
    });
}

/// `ttkernel.trunc_tile`. Trunc operation. No results.
pub fn trunc_tile(ctx: *mlir.Context, tile_index: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.trunc_tile", .{
        .operands = .{ .flat = &.{tile_index} },
        .location = location,
    });
}

/// `ttkernel.typecast_tile`. Cast the dataformat of the tile in the DST at specified index. No results.
pub fn typecast_tile(ctx: *mlir.Context, dst0_index: *const mlir.Value, in_dtype: *const mlir.Attribute, out_dtype: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.typecast_tile", .{
        .operands = .{ .flat = &.{dst0_index} },
        .attributes = &.{ .named(ctx, "in_dtype", in_dtype), .named(ctx, "out_dtype", out_dtype) },
        .location = location,
    });
}

/// `ttkernel.typecast_tile_init`. Init function for typecast_tile operation. Refer to documentation for any init function. No results.
pub fn typecast_tile_init(ctx: *mlir.Context, in_dtype: *const mlir.Attribute, out_dtype: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.typecast_tile_init", .{
        .attributes = &.{ .named(ctx, "in_dtype", in_dtype), .named(ctx, "out_dtype", out_dtype) },
        .location = location,
    });
}

/// `ttkernel.unary_bcast`. Broadcast operation. No results.
pub fn unary_bcast(ctx: *mlir.Context, in_cb: *const mlir.Value, in_tile_index: *const mlir.Value, dst_tile_index: *const mlir.Value, bcast_type: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.unary_bcast", .{
        .operands = .{ .flat = &.{ in_cb, in_tile_index, dst_tile_index } },
        .attributes = &.{.named(ctx, "bcast_type", bcast_type)},
        .location = location,
    });
}

/// `ttkernel.unary_bcast_init`. Init function. No results.
pub fn unary_bcast_init(ctx: *mlir.Context, in_cb: *const mlir.Value, out_cb: *const mlir.Value, bcast_type: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.unary_bcast_init", .{
        .operands = .{ .flat = &.{ in_cb, out_cb } },
        .attributes = &.{.named(ctx, "bcast_type", bcast_type)},
        .location = location,
    });
}

/// `ttkernel.unary_op_init_common`. Initialization function for unary operations. No results.
pub fn unary_op_init_common(ctx: *mlir.Context, icb: *const mlir.Value, ocb: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.unary_op_init_common", .{
        .operands = .{ .flat = &.{ icb, ocb } },
        .location = location,
    });
}

/// `ttkernel.unreachable`. Unreachable op. No results.
pub fn unreachable_(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.unreachable", .{
        .location = location,
    });
}

/// `ttkernel.untilize_block`. UntilizeBlockOp call. No results.
pub fn untilize_block(ctx: *mlir.Context, cb_in: *const mlir.Value, num_tiles: *const mlir.Value, cb_out: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.untilize_block", .{
        .operands = .{ .flat = &.{ cb_in, num_tiles, cb_out } },
        .location = location,
    });
}

/// `ttkernel.untilize_init`. UntilizeInitOp call. No results.
pub fn untilize_init(ctx: *mlir.Context, cb_in: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.untilize_init", .{
        .operands = .{ .flat = &.{cb_in} },
        .location = location,
    });
}

/// `ttkernel.untilize_uninit`. UntilizeUninitOp call. No results.
pub fn untilize_uninit(ctx: *mlir.Context, cb_in: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.untilize_uninit", .{
        .operands = .{ .flat = &.{cb_in} },
        .location = location,
    });
}

/// `ttkernel.where_tile`. Conditional selection operation. No results.
pub fn where_tile(ctx: *mlir.Context, dst_cond_index: *const mlir.Value, dst_true_index: *const mlir.Value, dst_false_index: *const mlir.Value, odst_index: *const mlir.Value, dtype: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.where_tile", .{
        .operands = .{ .flat = &.{ dst_cond_index, dst_true_index, dst_false_index, odst_index } },
        .attributes = &.{.named(ctx, "dtype", dtype)},
        .location = location,
    });
}

/// `ttkernel.where_tile_init`. Short init function. No results.
pub fn where_tile_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, "ttkernel.where_tile_init", .{
        .location = location,
    });
}

/// Results of each builder's operation; -1 when variadic or optional.
pub const num_results = struct {
    pub const LocalTensorAccessor: comptime_int = 1;
    pub const TensorAccessor: comptime_int = 1;
    pub const TensorAccessorArgs: comptime_int = 1;
    pub const abs_tile: comptime_int = 0;
    pub const abs_tile_init: comptime_int = 0;
    pub const abs_tile_int32: comptime_int = 0;
    pub const acos_tile: comptime_int = 0;
    pub const acos_tile_init: comptime_int = 0;
    pub const add_binary_tile: comptime_int = 0;
    pub const add_binary_tile_init: comptime_int = 0;
    pub const add_int_tile: comptime_int = 0;
    pub const add_int_tile_init: comptime_int = 0;
    pub const add_tiles: comptime_int = 0;
    pub const add_tiles_init: comptime_int = 0;
    pub const add_unary_tile: comptime_int = 0;
    pub const add_unary_tile_int32: comptime_int = 0;
    pub const asin_tile: comptime_int = 0;
    pub const asin_tile_init: comptime_int = 0;
    pub const atan2_binary_tile: comptime_int = 0;
    pub const atan2_binary_tile_init: comptime_int = 0;
    pub const atan_tile: comptime_int = 0;
    pub const atan_tile_init: comptime_int = 0;
    pub const bfloat16_greater: comptime_int = 1;
    pub const binary_bitwise_tile_init: comptime_int = 0;
    pub const binary_dest_reuse_tiles: comptime_int = 0;
    pub const binary_dest_reuse_tiles_init: comptime_int = 0;
    pub const binary_left_shift_tile: comptime_int = 0;
    pub const binary_logical_right_shift_tile: comptime_int = 0;
    pub const binary_max_int32_tile: comptime_int = 0;
    pub const binary_max_int32_tile_init: comptime_int = 0;
    pub const binary_max_tile: comptime_int = 0;
    pub const binary_max_tile_init: comptime_int = 0;
    pub const binary_min_int32_tile: comptime_int = 0;
    pub const binary_min_int32_tile_init: comptime_int = 0;
    pub const binary_min_tile: comptime_int = 0;
    pub const binary_min_tile_init: comptime_int = 0;
    pub const binary_op_init_common: comptime_int = 0;
    pub const binary_right_shift_tile: comptime_int = 0;
    pub const binary_shift_tile_init: comptime_int = 0;
    pub const binop_with_scalar_tile_init: comptime_int = 0;
    pub const bitcast: comptime_int = 1;
    pub const bitwise_and_binary_tile: comptime_int = 0;
    pub const bitwise_not_tile: comptime_int = 0;
    pub const bitwise_not_tile_init: comptime_int = 0;
    pub const bitwise_or_binary_tile: comptime_int = 0;
    pub const bitwise_xor_binary_tile: comptime_int = 0;
    pub const cast_to_l1_addr: comptime_int = 1;
    pub const cb_pop_front: comptime_int = 0;
    pub const cb_push_back: comptime_int = 0;
    pub const cb_reserve_back: comptime_int = 0;
    pub const cb_wait_front: comptime_int = 0;
    pub const ceil_tile: comptime_int = 0;
    pub const clamp_tile: comptime_int = 0;
    pub const clamp_tile_init: comptime_int = 0;
    pub const clamp_tile_int32: comptime_int = 0;
    pub const compute_kernel_hw_startup: comptime_int = 0;
    pub const copy_block_matmul_partials: comptime_int = 0;
    pub const copy_dest_values: comptime_int = 0;
    pub const copy_dest_values_init: comptime_int = 0;
    pub const copy_tile: comptime_int = 0;
    pub const copy_tile_init: comptime_int = 0;
    pub const cos_tile: comptime_int = 0;
    pub const cos_tile_init: comptime_int = 0;
    pub const div_binary_tile: comptime_int = 0;
    pub const div_binary_tile_init: comptime_int = 0;
    pub const div_unary_tile: comptime_int = 0;
    pub const dprint: comptime_int = 0;
    pub const eq_binary_tile: comptime_int = 0;
    pub const eq_binary_tile_init: comptime_int = 0;
    pub const eqz_tile: comptime_int = 0;
    pub const eqz_tile_init: comptime_int = 0;
    pub const eqz_tile_int32: comptime_int = 0;
    pub const erf_tile: comptime_int = 0;
    pub const erf_tile_init: comptime_int = 0;
    pub const erfc_tile: comptime_int = 0;
    pub const erfc_tile_init: comptime_int = 0;
    pub const exp2_tile: comptime_int = 0;
    pub const exp2_tile_init: comptime_int = 0;
    pub const exp_tile: comptime_int = 0;
    pub const exp_tile_init: comptime_int = 0;
    pub const experimental_close_fabric_connections: comptime_int = 0;
    pub const experimental_constant_table_lookup: comptime_int = 1;
    pub const experimental_convert_logical_x_to_translated: comptime_int = 1;
    pub const experimental_convert_logical_y_to_translated: comptime_int = 1;
    pub const experimental_create_fabric_connection_manager: comptime_int = 1;
    pub const experimental_fabric_fast_write_any_len: comptime_int = 0;
    pub const experimental_fabric_mcast_fast_write_any_len: comptime_int = 0;
    pub const experimental_fabric_mcast_sem_inc: comptime_int = 0;
    pub const experimental_fabric_sem_inc: comptime_int = 0;
    pub const experimental_fill_arange_tile: comptime_int = 0;
    pub const experimental_get_device_id_from_logical_mesh_position: comptime_int = 1;
    pub const experimental_get_my_device_id: comptime_int = 1;
    pub const experimental_get_my_logical_mesh_position: comptime_int = 1;
    pub const experimental_matmul_block: comptime_int = 0;
    pub const experimental_pack_untilize_block: comptime_int = 0;
    pub const experimental_semaphore_reached: comptime_int = 1;
    pub const experimental_semaphore_wait: comptime_int = 0;
    pub const experimental_semaphore_wait_min: comptime_int = 0;
    pub const experimental_setup_fabric_connections: comptime_int = 0;
    pub const experimental_tilize_block: comptime_int = 0;
    pub const experimental_unpack_stall_on_pack: comptime_int = 0;
    pub const experimental_untilize_block: comptime_int = 0;
    pub const experimental_write_col_mask_tile: comptime_int = 0;
    pub const experimental_write_row_mask_tile: comptime_int = 0;
    pub const experimental_row_normalization_block: comptime_int = 0;
    pub const expm1_tile: comptime_int = 0;
    pub const expm1_tile_init: comptime_int = 0;
    pub const fill_tile: comptime_int = 0;
    pub const fill_tile_init: comptime_int = 0;
    pub const fill_tile_int: comptime_int = 0;
    pub const float32_greater: comptime_int = 1;
    pub const floor_tile: comptime_int = 0;
    pub const frac_tile: comptime_int = 0;
    pub const ge_binary_tile: comptime_int = 0;
    pub const ge_binary_tile_init: comptime_int = 0;
    pub const gelu_tile: comptime_int = 0;
    pub const gelu_tile_init: comptime_int = 0;
    pub const get_arg_val: comptime_int = 1;
    pub const get_common_arg_val: comptime_int = 1;
    pub const get_compile_time_arg_val: comptime_int = 1;
    pub const get_dataformat: comptime_int = 1;
    pub const get_dfb_id: comptime_int = 1;
    pub const get_noc_addr: comptime_int = 1;
    pub const get_noc_addr_from_bank_id: comptime_int = 1;
    pub const get_noc_multicast_addr: comptime_int = 1;
    pub const get_read_ptr: comptime_int = 1;
    pub const get_semaphore: comptime_int = 1;
    pub const get_tile_size: comptime_int = 1;
    pub const get_write_ptr: comptime_int = 1;
    pub const gez_tile: comptime_int = 0;
    pub const gez_tile_init: comptime_int = 0;
    pub const gez_tile_int32: comptime_int = 0;
    pub const gt_binary_tile: comptime_int = 0;
    pub const gt_binary_tile_init: comptime_int = 0;
    pub const gtz_tile: comptime_int = 0;
    pub const gtz_tile_init: comptime_int = 0;
    pub const gtz_tile_int32: comptime_int = 0;
    pub const hardsigmoid_tile: comptime_int = 0;
    pub const hardsigmoid_tile_init: comptime_int = 0;
    pub const init_sfpu: comptime_int = 0;
    pub const invoke_sfpi: comptime_int = 0;
    pub const le_binary_tile: comptime_int = 0;
    pub const le_binary_tile_init: comptime_int = 0;
    pub const lez_tile: comptime_int = 0;
    pub const lez_tile_init: comptime_int = 0;
    pub const lez_tile_int32: comptime_int = 0;
    pub const load_from_l1: comptime_int = 1;
    pub const log1p_tile: comptime_int = 0;
    pub const log1p_tile_init: comptime_int = 0;
    pub const log_tile: comptime_int = 0;
    pub const log_tile_init: comptime_int = 0;
    pub const logical_not_tile: comptime_int = 0;
    pub const logical_not_tile_init: comptime_int = 0;
    pub const lt_binary_tile: comptime_int = 0;
    pub const lt_binary_tile_init: comptime_int = 0;
    pub const ltz_tile: comptime_int = 0;
    pub const ltz_tile_init: comptime_int = 0;
    pub const ltz_tile_int32: comptime_int = 0;
    pub const matmul_block: comptime_int = 0;
    pub const matmul_tiles: comptime_int = 0;
    pub const mem_zeros_base: comptime_int = 1;
    pub const mem_zeros_size: comptime_int = 1;
    pub const mm_block_init: comptime_int = 0;
    pub const mm_block_init_short: comptime_int = 0;
    pub const mm_init: comptime_int = 0;
    pub const mm_init_short: comptime_int = 0;
    pub const mul_binary_tile: comptime_int = 0;
    pub const mul_binary_tile_init: comptime_int = 0;
    pub const mul_int_tile: comptime_int = 0;
    pub const mul_int_tile_init: comptime_int = 0;
    pub const mul_tiles: comptime_int = 0;
    pub const mul_tiles_init: comptime_int = 0;
    pub const mul_unary_tile: comptime_int = 0;
    pub const my_logical_x_: comptime_int = 1;
    pub const my_logical_y_: comptime_int = 1;
    pub const my_x: comptime_int = 1;
    pub const my_y: comptime_int = 1;
    pub const ne_binary_tile: comptime_int = 0;
    pub const ne_binary_tile_init: comptime_int = 0;
    pub const negative_tile: comptime_int = 0;
    pub const negative_tile_init: comptime_int = 0;
    pub const negative_tile_int32: comptime_int = 0;
    pub const nez_tile: comptime_int = 0;
    pub const nez_tile_init: comptime_int = 0;
    pub const nez_tile_int32: comptime_int = 0;
    pub const noc_async_atomic_barrier: comptime_int = 0;
    pub const noc_async_read: comptime_int = 0;
    pub const noc_async_read_barrier: comptime_int = 0;
    pub const noc_async_read_barrier_with_trid: comptime_int = 0;
    pub const noc_async_read_one_packet_set_state: comptime_int = 0;
    pub const noc_async_read_one_packet_with_state: comptime_int = 0;
    pub const noc_async_read_tile: comptime_int = 0;
    pub const noc_async_write: comptime_int = 0;
    pub const noc_async_write_barrier: comptime_int = 0;
    pub const noc_async_write_barrier_with_trid: comptime_int = 0;
    pub const noc_async_write_multicast: comptime_int = 0;
    pub const noc_async_write_multicast_loopback_src: comptime_int = 0;
    pub const noc_async_write_multicast_one_packet: comptime_int = 0;
    pub const noc_async_write_one_packet_set_state: comptime_int = 0;
    pub const noc_async_write_one_packet_with_state: comptime_int = 0;
    pub const noc_async_write_one_packet_with_trid: comptime_int = 0;
    pub const noc_async_write_tile: comptime_int = 0;
    pub const noc_async_writes_flushed: comptime_int = 0;
    pub const noc_inline_dw_write: comptime_int = 0;
    pub const noc_semaphore_inc: comptime_int = 0;
    pub const noc_semaphore_inc_multicast: comptime_int = 0;
    pub const noc_semaphore_set: comptime_int = 0;
    pub const noc_semaphore_set_multicast: comptime_int = 0;
    pub const noc_semaphore_set_multicast_loopback_src: comptime_int = 0;
    pub const opaque_call: comptime_int = -1;
    pub const pack_reconfig_data_format: comptime_int = 0;
    pub const pack_reconfig_l1_acc: comptime_int = 0;
    pub const pack_tile: comptime_int = 0;
    pub const pack_tile_block: comptime_int = 0;
    pub const pack_untilize_init: comptime_int = 0;
    pub const pack_untilize_uninit: comptime_int = 0;
    pub const pack_waited_tile: comptime_int = 0;
    pub const power_binary_tile: comptime_int = 0;
    pub const power_binary_tile_init: comptime_int = 0;
    pub const power_tile: comptime_int = 0;
    pub const power_tile_init: comptime_int = 0;
    pub const rand_tile: comptime_int = 0;
    pub const rand_tile_init: comptime_int = 0;
    pub const recip_tile: comptime_int = 0;
    pub const recip_tile_init: comptime_int = 0;
    pub const reduce_init: comptime_int = 0;
    pub const reduce_tile: comptime_int = 0;
    pub const reduce_uninit: comptime_int = 0;
    pub const reinterpret_cast: comptime_int = 1;
    pub const relu_tile: comptime_int = 0;
    pub const relu_tile_init: comptime_int = 0;
    pub const relu_tile_int32: comptime_int = 0;
    pub const remote_sram_write_u32: comptime_int = 0;
    pub const rounding_op_tile_init: comptime_int = 0;
    pub const routing_plane_atomic_inc: comptime_int = 0;
    pub const routing_plane_close_connections: comptime_int = 0;
    pub const routing_plane_create_connection_manager: comptime_int = 1;
    pub const routing_plane_fused_write_atomic_inc: comptime_int = 0;
    pub const routing_plane_open_connections: comptime_int = 1;
    pub const routing_plane_scatter_write: comptime_int = 0;
    pub const routing_plane_write: comptime_int = 0;
    pub const rsqrt_tile: comptime_int = 0;
    pub const rsqrt_tile_init: comptime_int = 0;
    pub const selu_tile: comptime_int = 0;
    pub const selu_tile_init: comptime_int = 0;
    pub const sfpu_reduce: comptime_int = 0;
    pub const sfpu_reduce_init: comptime_int = 0;
    pub const sigmoid_tile: comptime_int = 0;
    pub const sigmoid_tile_init: comptime_int = 0;
    pub const sign_tile: comptime_int = 0;
    pub const sign_tile_init: comptime_int = 0;
    pub const signbit_tile: comptime_int = 0;
    pub const signbit_tile_init: comptime_int = 0;
    pub const silu_tile: comptime_int = 0;
    pub const silu_tile_init: comptime_int = 0;
    pub const sin_tile: comptime_int = 0;
    pub const sin_tile_init: comptime_int = 0;
    pub const softsign_tile: comptime_int = 0;
    pub const softsign_tile_init: comptime_int = 0;
    pub const sqrt_tile: comptime_int = 0;
    pub const sqrt_tile_init: comptime_int = 0;
    pub const square_tile: comptime_int = 0;
    pub const square_tile_init: comptime_int = 0;
    pub const store_to_l1: comptime_int = 0;
    pub const sub_binary_tile: comptime_int = 0;
    pub const sub_binary_tile_init: comptime_int = 0;
    pub const sub_int_tile: comptime_int = 0;
    pub const sub_int_tile_init: comptime_int = 0;
    pub const sub_tiles: comptime_int = 0;
    pub const sub_tiles_init: comptime_int = 0;
    pub const sub_unary_tile: comptime_int = 0;
    pub const sub_unary_tile_int32: comptime_int = 0;
    pub const tan_tile: comptime_int = 0;
    pub const tan_tile_init: comptime_int = 0;
    pub const tanh_tile: comptime_int = 0;
    pub const tanh_tile_init: comptime_int = 0;
    pub const tensor_accessor_get_bank_and_offset: comptime_int = 1;
    pub const tensor_accessor_get_noc_addr: comptime_int = 1;
    pub const tensor_accessor_get_shard_noc_addr: comptime_int = 1;
    pub const tensor_accessor_is_local_addr: comptime_int = 1;
    pub const tensor_accessor_is_local_bank: comptime_int = 1;
    pub const tensor_accessor_is_local_page: comptime_int = 1;
    pub const tensor_accessor_is_local_shard: comptime_int = 1;
    pub const tile_regs_acquire: comptime_int = 0;
    pub const tile_regs_commit: comptime_int = 0;
    pub const tile_regs_release: comptime_int = 0;
    pub const tile_regs_wait: comptime_int = 0;
    pub const tilize_block: comptime_int = 0;
    pub const tilize_init: comptime_int = 0;
    pub const tilize_uninit: comptime_int = 0;
    pub const topk_local_sort: comptime_int = 0;
    pub const topk_merge: comptime_int = 0;
    pub const topk_rebuild: comptime_int = 0;
    pub const topk_tile_init: comptime_int = 0;
    pub const transpose_wh_init: comptime_int = 0;
    pub const transpose_wh_tile: comptime_int = 0;
    pub const trunc_tile: comptime_int = 0;
    pub const typecast_tile: comptime_int = 0;
    pub const typecast_tile_init: comptime_int = 0;
    pub const unary_bcast: comptime_int = 0;
    pub const unary_bcast_init: comptime_int = 0;
    pub const unary_op_init_common: comptime_int = 0;
    pub const unreachable_: comptime_int = 0;
    pub const untilize_block: comptime_int = 0;
    pub const untilize_init: comptime_int = 0;
    pub const untilize_uninit: comptime_int = 0;
    pub const where_tile: comptime_int = 0;
    pub const where_tile_init: comptime_int = 0;
};
