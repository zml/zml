const std = @import("std");

const cuda = @import("mlir/dialects/cuda");
const mlir = @import("mlir");

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
    for (cuda.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
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
        const info = @typeInfo(@FieldType(T.InitArgs, field));
        if (info == .pointer and info.pointer.size == .slice and @typeInfo(info.pointer.child) == .@"enum") {
            const E = info.pointer.child;
            const count = @field(T, "getNum" ++ name)(value);
            const elements: []E = @ptrCast(buffers[i][0..count]);
            for (elements, 0..) |*element, pos| element.* = @field(T, "get" ++ name[0 .. name.len - 1])(value, pos);
            @field(args, field) = elements;
        } else {
            @field(args, field) = @field(T, "get" ++ name)(value);
        }
    }
    return T.get(ctx, args);
}
