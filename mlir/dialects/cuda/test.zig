const std = @import("std");

const cuda = @import("mlir/dialects/cuda");
const mlir = @import("mlir");

const operation_names: []const []const u8 = &.{
    "cuda.addressof",
    "cuda.binary",
    "cuda.cast",
    "cuda.compose",
    "cuda.decompose",
    "cuda.device.sync",
    "cuda.driver.event.create",
    "cuda.driver.event.destroy",
    "cuda.driver.event.elapsed_time",
    "cuda.driver.event.query",
    "cuda.driver.event.record",
    "cuda.driver.event.record_with_flags",
    "cuda.driver.event.synchronize",
    "cuda.driver.library.get_global",
    "cuda.driver.library.get_kernel",
    "cuda.driver.library.get_kernel_count",
    "cuda.driver.library.get_managed",
    "cuda.driver.library.get_module",
    "cuda.driver.library.get_unified_function",
    "cuda.driver.library.load_data",
    "cuda.driver.library.load_from_file",
    "cuda.driver.library.unload",
    "cuda.driver.tensor_map.encode_im2col",
    "cuda.driver.tensor_map.encode_im2col_wide",
    "cuda.driver.tensor_map.replace_address",
    "cuda.event.create",
    "cuda.event.destroy",
    "cuda.event.elapsed_time",
    "cuda.event.record",
    "cuda.func",
    "cuda.func.get_name",
    "cuda.graph.destroy",
    "cuda.graph.instantiate",
    "cuda.graph.launch",
    "cuda.graph_exec.destroy",
    "cuda.kernel",
    "cuda.launch",
    "cuda.launch_cfg.access_policy_window",
    "cuda.launch_cfg.cluster_dim",
    "cuda.launch_cfg.cluster_scheduling_policy",
    "cuda.launch_cfg.cooperative",
    "cuda.launch_cfg.create",
    "cuda.launch_cfg.create_from_stream",
    "cuda.launch_cfg.device_updatable_kernel_node",
    "cuda.launch_cfg.grid_dimensions",
    "cuda.launch_cfg.launch_completion_event",
    "cuda.launch_cfg.mem_sync_domain",
    "cuda.launch_cfg.mem_sync_domain_map",
    "cuda.launch_cfg.nvlink_util_centric_scheduling",
    "cuda.launch_cfg.preferred_cluster_dim",
    "cuda.launch_cfg.print",
    "cuda.launch_cfg.priority",
    "cuda.launch_cfg.programmatic_event",
    "cuda.launch_cfg.programmatic_stream_serialization_allowed",
    "cuda.launch_cfg.shared_mem_carveout",
    "cuda.launch_cfg.sync_policy",
    "cuda.launch_ex",
    "cuda.launch_kernel",
    "cuda.library.get_kernel",
    "cuda.library.load_data",
    "cuda.library.unload",
    "cuda.memory.alloc",
    "cuda.memory.alloc_async",
    "cuda.memory.alloc_host",
    "cuda.memory.alloc_managed",
    "cuda.memory.copy",
    "cuda.memory.copy_async",
    "cuda.memory.free",
    "cuda.memory.free_async",
    "cuda.memory.free_host",
    "cuda.memory.set",
    "cuda.memory.set_async",
    "cuda.occupancy_max_active_blocks_per_multiprocessor",
    "cuda.profiler.start",
    "cuda.profiler.stop",
    "cuda.result.assert_success",
    "cuda.result.failed",
    "cuda.result.print_on_error",
    "cuda.result.succeeded",
    "cuda.return",
    "cuda.return_if_error",
    "cuda.runtime.device.properties",
    "cuda.runtime.device_prop.get_access_policy_max_window_size",
    "cuda.runtime.device_prop.get_async_engine_count",
    "cuda.runtime.device_prop.get_can_map_host_memory",
    "cuda.runtime.device_prop.get_can_use_host_pointer_for_registered_mem",
    "cuda.runtime.device_prop.get_cluster_launch",
    "cuda.runtime.device_prop.get_compute_preemption_supported",
    "cuda.runtime.device_prop.get_concurrent_kernels",
    "cuda.runtime.device_prop.get_concurrent_managed_access",
    "cuda.runtime.device_prop.get_cooperative_launch",
    "cuda.runtime.device_prop.get_deferred_mapping_cuda_array_supported",
    "cuda.runtime.device_prop.get_device_numa_config",
    "cuda.runtime.device_prop.get_device_numa_id",
    "cuda.runtime.device_prop.get_direct_managed_mem_access_from_host",
    "cuda.runtime.device_prop.get_ecc_enabled",
    "cuda.runtime.device_prop.get_global_l1cache_supported",
    "cuda.runtime.device_prop.get_gpu_direct_rdma_flush_writes_options",
    "cuda.runtime.device_prop.get_gpu_direct_rdma_supported",
    "cuda.runtime.device_prop.get_gpu_direct_rdma_writes_ordering",
    "cuda.runtime.device_prop.get_gpu_pci_device_id",
    "cuda.runtime.device_prop.get_gpu_pci_subsystem_id",
    "cuda.runtime.device_prop.get_host_native_atomic_supported",
    "cuda.runtime.device_prop.get_host_numa_id",
    "cuda.runtime.device_prop.get_host_numa_multinode_ipc_supported",
    "cuda.runtime.device_prop.get_host_register_read_only_supported",
    "cuda.runtime.device_prop.get_host_register_supported",
    "cuda.runtime.device_prop.get_integrated",
    "cuda.runtime.device_prop.get_ipc_event_supported",
    "cuda.runtime.device_prop.get_is_multi_gpu_board",
    "cuda.runtime.device_prop.get_l_2cache_size",
    "cuda.runtime.device_prop.get_local_l1cache_supported",
    "cuda.runtime.device_prop.get_luid",
    "cuda.runtime.device_prop.get_luid_device_node_mask",
    "cuda.runtime.device_prop.get_major",
    "cuda.runtime.device_prop.get_managed_memory",
    "cuda.runtime.device_prop.get_max_blocks_per_multi_processor",
    "cuda.runtime.device_prop.get_max_grid_size",
    "cuda.runtime.device_prop.get_max_surface_1d",
    "cuda.runtime.device_prop.get_max_surface_1d_layered",
    "cuda.runtime.device_prop.get_max_surface_2d",
    "cuda.runtime.device_prop.get_max_surface_2d_layered",
    "cuda.runtime.device_prop.get_max_surface_3d",
    "cuda.runtime.device_prop.get_max_surface_cubemap",
    "cuda.runtime.device_prop.get_max_surface_cubemap_layered",
    "cuda.runtime.device_prop.get_max_texture_1d",
    "cuda.runtime.device_prop.get_max_texture_1d_layered",
    "cuda.runtime.device_prop.get_max_texture_1d_mipmap",
    "cuda.runtime.device_prop.get_max_texture_2d",
    "cuda.runtime.device_prop.get_max_texture_2d_gather",
    "cuda.runtime.device_prop.get_max_texture_2d_layered",
    "cuda.runtime.device_prop.get_max_texture_2d_linear",
    "cuda.runtime.device_prop.get_max_texture_2d_mipmap",
    "cuda.runtime.device_prop.get_max_texture_3d",
    "cuda.runtime.device_prop.get_max_texture_3d_alt",
    "cuda.runtime.device_prop.get_max_texture_cubemap",
    "cuda.runtime.device_prop.get_max_texture_cubemap_layered",
    "cuda.runtime.device_prop.get_max_threads_dim",
    "cuda.runtime.device_prop.get_max_threads_per_block",
    "cuda.runtime.device_prop.get_max_threads_per_multi_processor",
    "cuda.runtime.device_prop.get_mem_pitch",
    "cuda.runtime.device_prop.get_memory_bus_width",
    "cuda.runtime.device_prop.get_memory_pool_supported_handle_types",
    "cuda.runtime.device_prop.get_memory_pools_supported",
    "cuda.runtime.device_prop.get_minor",
    "cuda.runtime.device_prop.get_mps_enabled",
    "cuda.runtime.device_prop.get_multi_gpu_board_group_id",
    "cuda.runtime.device_prop.get_multi_processor_count",
    "cuda.runtime.device_prop.get_name",
    "cuda.runtime.device_prop.get_pageable_memory_access",
    "cuda.runtime.device_prop.get_pageable_memory_access_uses_host_page_tables",
    "cuda.runtime.device_prop.get_pci_bus_id",
    "cuda.runtime.device_prop.get_pci_device_id",
    "cuda.runtime.device_prop.get_pci_domain_id",
    "cuda.runtime.device_prop.get_persisting_l2cache_max_size",
    "cuda.runtime.device_prop.get_regs_per_block",
    "cuda.runtime.device_prop.get_regs_per_multiprocessor",
    "cuda.runtime.device_prop.get_reserved_shared_mem_per_block",
    "cuda.runtime.device_prop.get_shared_mem_per_block",
    "cuda.runtime.device_prop.get_shared_mem_per_block_optin",
    "cuda.runtime.device_prop.get_shared_mem_per_multiprocessor",
    "cuda.runtime.device_prop.get_sparse_cuda_array_supported",
    "cuda.runtime.device_prop.get_stream_priorities_supported",
    "cuda.runtime.device_prop.get_surface_alignment",
    "cuda.runtime.device_prop.get_tcc_driver",
    "cuda.runtime.device_prop.get_texture_alignment",
    "cuda.runtime.device_prop.get_texture_pitch_alignment",
    "cuda.runtime.device_prop.get_timeline_semaphore_interop_supported",
    "cuda.runtime.device_prop.get_total_const_mem",
    "cuda.runtime.device_prop.get_total_global_mem",
    "cuda.runtime.device_prop.get_unified_addressing",
    "cuda.runtime.device_prop.get_unified_function_pointers",
    "cuda.runtime.device_prop.get_uuid",
    "cuda.runtime.device_prop.get_warp_size",
    "cuda.runtime.dim_3.get_x",
    "cuda.runtime.dim_3.get_y",
    "cuda.runtime.dim_3.get_z",
    "cuda.runtime.event.create_with_flags",
    "cuda.runtime.event.query",
    "cuda.runtime.event.record_with_flags",
    "cuda.runtime.event.synchronize",
    "cuda.runtime.extent.get_depth",
    "cuda.runtime.extent.get_height",
    "cuda.runtime.extent.get_width",
    "cuda.runtime.graph.add_kernel_node",
    "cuda.runtime.graph.create",
    "cuda.runtime.graph.kernel_node_get_attribute",
    "cuda.runtime.graph.kernel_node_set_attribute",
    "cuda.runtime.kernel_node_params.get_block_dim",
    "cuda.runtime.kernel_node_params.get_extra",
    "cuda.runtime.kernel_node_params.get_func",
    "cuda.runtime.kernel_node_params.get_grid_dim",
    "cuda.runtime.kernel_node_params.get_kernel_params",
    "cuda.runtime.kernel_node_params.get_shared_mem_bytes",
    "cuda.runtime.library.get_global",
    "cuda.runtime.library.get_kernel",
    "cuda.runtime.library.get_kernel_count",
    "cuda.runtime.library.get_managed",
    "cuda.runtime.library.get_unified_function",
    "cuda.runtime.library.load_data",
    "cuda.runtime.library.load_from_file",
    "cuda.runtime.library.unload",
    "cuda.runtime.memcpy_3d_params.get_dst_array",
    "cuda.runtime.memcpy_3d_params.get_dst_pos",
    "cuda.runtime.memcpy_3d_params.get_dst_ptr",
    "cuda.runtime.memcpy_3d_params.get_extent",
    "cuda.runtime.memcpy_3d_params.get_kind",
    "cuda.runtime.memcpy_3d_params.get_src_array",
    "cuda.runtime.memcpy_3d_params.get_src_pos",
    "cuda.runtime.memcpy_3d_params.get_src_ptr",
    "cuda.runtime.memory.cpy3d",
    "cuda.runtime.memory.cpy3d_async",
    "cuda.runtime.pitched_ptr.get_pitch",
    "cuda.runtime.pitched_ptr.get_ptr",
    "cuda.runtime.pitched_ptr.get_xsize",
    "cuda.runtime.pitched_ptr.get_ysize",
    "cuda.runtime.pos.get_x",
    "cuda.runtime.pos.get_y",
    "cuda.runtime.pos.get_z",
    "cuda.runtime.uuid.get_bytes",
    "cuda.stream.add_callback",
    "cuda.stream.begin_capture",
    "cuda.stream.create",
    "cuda.stream.destroy",
    "cuda.stream.end_capture",
    "cuda.stream.synchronize",
    "cuda.tensor_map_encode_tiled",
};

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(cuda);
    std.testing.refAllDecls(cuda.ops);
    inline for (comptime std.meta.declarations(cuda.ops)) |decl| {
        const T = @field(cuda.ops, decl);
        if (@TypeOf(T) == type) std.testing.refAllDecls(T);
    }
}

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    cuda.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

test "every bound operation is registered" {
    const ctx = try testContext();
    defer ctx.deinit();
    for (operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}

test "cuda types and attributes print as the compiler does" {
    @setEvalBranchQuota(100_000);
    const ctx = try testContext();
    defer ctx.deinit();
    // The compiler's printed forms, which these must print byte for byte.
    inline for (.{
        .{ cuda.ops.AssumeKernelAttr, "#cuda.assume_kernel_attr<true>" },
        .{ cuda.ops.ComputeTargetAttr, "#cuda.compute_target<sass, portable, [sm_90, sm_100]>" },
        .{ cuda.ops.DevMaxSharedMemoryOptinAttr, "#cuda.dev_max_shared_memory_optin" },
        .{ cuda.ops.ExecutableAttr, "#cuda.executable<elf>" },
        .{ cuda.ops.LaunchConfigType, "!cuda.launch_cfg<max_attrs = 3>" },
        .{ cuda.ops.StreamType, "!cuda.stream" },
        .{ cuda.ops.ResultType, "!cuda.result" },
        .{ cuda.ops.IntType, "!cuda.int" },
        .{ cuda.ops.RuntimeDim3Type, "!cuda.runtime.dim_3" },
        .{ cuda.ops.RuntimeLaunchAttributeValueSyncPolicyType, "!cuda.runtime.launch_attribute_value.sync_policy" },
        .{ cuda.ops.RuntimeLaunchAttributeValueClusterSchedulingPolicyPreferenceType, "!cuda.runtime.launch_attribute_value.cluster_scheduling_policy_preference" },
        .{ cuda.ops.RuntimeLaunchAttributeValueNvlinkUtilCentricSchedulingType, "!cuda.runtime.launch_attribute_value.nvlink_util_centric_scheduling" },
        .{ cuda.ops.RuntimeLaunchAttributeValueProgrammaticStreamSerializationAllowedType, "!cuda.runtime.launch_attribute_value.programmatic_stream_serialization_allowed" },
    }) |example| {
        const T = example[0];
        const is_type = @hasDecl(T, "type_");
        const parsed = if (is_type) try mlir.Type.parse(ctx, example[1]) else try mlir.Attribute.parse(ctx, example[1]);
        const value = parsed.isA(T) orelse return error.TestUnexpectedResult;
        try std.testing.expect((try rebuildEnums(T, ctx, value)).eql(value));
        var buf: [512]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        try w.print("{f}", .{value});
        try std.testing.expectEqualStrings(example[1], w.buffered());
    }
    // The compiler prints these without the space before the dictionary; it
    // parses both.
    inline for (.{
        .{ cuda.ops.DeviceAttributesAttr, "#cuda<device_attributes{}>" },
        .{ cuda.ops.FuncAttributesAttr, "#cuda<func_attributes{}>" },
    }) |example| {
        const parsed = try mlir.Attribute.parse(ctx, example[1]);
        const value = parsed.isA(example[0]) orelse return error.TestUnexpectedResult;
        try std.testing.expect((try rebuildEnums(example[0], ctx, value)).eql(value));
    }
}

/// `rebuild` for parameters that are slices of enums.
fn rebuildEnums(comptime T: type, ctx: *mlir.Context, value: *const T) !*const T {
    var args: T.InitArgs = undefined;
    var buffers: [4][16]u32 = undefined;
    inline for (@typeInfo(T.InitArgs).@"struct".field_names, 0..) |field, i| {
        const name = [_]u8{comptime std.ascii.toUpper(field[0])} ++ field[1..];
        const noun = [_]u8{comptime std.ascii.toLower(field[0])} ++ field[1..];
        const accessor = if (@hasDecl(T, noun ++ "_")) noun ++ "_" else noun;
        const info = @typeInfo(@FieldType(T.InitArgs, field));
        if (info == .pointer and info.pointer.size == .slice and @typeInfo(info.pointer.child) == .@"enum") {
            const E = info.pointer.child;
            const count = @field(T, "num" ++ name)(value);
            const elements: []E = @ptrCast(buffers[i][0..count]);
            for (elements, 0..) |*element, pos| element.* = @field(T, noun[0 .. noun.len - 1])(value, pos);
            @field(args, field) = elements;
        } else {
            @field(args, field) = @field(T, accessor)(value);
        }
    }
    return T.get(ctx, args);
}
