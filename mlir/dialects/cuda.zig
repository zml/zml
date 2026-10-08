//! Builders for the CUDA host-program operations CuTe kernels emit, over the
//! bindings of NVIDIA's `cuda` dialect (`mlir/dialects/cute_ir`, `cute.cuda`),
//! which the CuTe-DSL compiler registers. Each builder fixes the attributes the
//! kernels do not choose.

const std = @import("std");

const cute = @import("mlir/dialects/cute_ir");
const mlir = @import("mlir");
const stdx = @import("stdx");

pub const ops = cute.cuda;

pub const dialect_namespace = ops.dialect_namespace;

pub fn streamType(ctx: *mlir.Context) *const mlir.Type {
    return (ops.StreamType.get(ctx, .{}) catch unreachable).type_();
}

pub fn resultType(ctx: *mlir.Context) *const mlir.Type {
    return (ops.ResultType.get(ctx, .{}) catch unreachable).type_();
}

pub fn launchConfigType(ctx: *mlir.Context, max_attrs: u32) *const mlir.Type {
    return (ops.LaunchConfigType.get(ctx, .{ .maxNumAttrs = max_attrs }) catch unreachable).type_();
}

/// `#cuda.assume_kernel_attr<value>`, whether `cuda.launch_ex` may assume its
/// callee is a kernel.
pub fn assumeKernelAttr(ctx: *mlir.Context, value: bool) *const mlir.Attribute {
    return (ops.AssumeKernelAttr.get(ctx, .{ .value = value }) catch unreachable).attribute();
}

pub const KernelArgs = struct {
    name: []const u8,
    function_type: *const mlir.Type,
    body: *mlir.Block,
    arg_attrs: ?*const mlir.Attribute = null,
    extra_attributes: []const mlir.NamedAttribute = &.{},
    location: *const mlir.Location,
};

/// `cuda.kernel`; `body` is filled afterwards.
pub fn kernel(ctx: *mlir.Context, args: KernelArgs) *mlir.Operation {
    const op = ops.kernel(ctx, .string(ctx, args.name), .typeAttr(args.function_type), args.arg_attrs, null, null, .{args.body}, args.location);
    for (args.extra_attributes) |attr| op.setAttributeByName(attr.name().str(), attr.attribute());
    return op;
}

pub fn return_(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return ops.@"return"(ctx, values, location);
}

pub fn cast(ctx: *mlir.Context, dst: *const mlir.Type, src: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return ops.cast(ctx, src, dst, location);
}

pub const LaunchConfigCreateArgs = struct {
    max_attrs: u32,
    block: [3]*const mlir.Value,
    dynamic_smem: *const mlir.Value,
    grid: [3]*const mlir.Value,
    stream: *const mlir.Value,
    location: *const mlir.Location,
};

pub fn launch_cfg_create(ctx: *mlir.Context, args: LaunchConfigCreateArgs) *mlir.Operation {
    return ops.launch_cfg_create(
        ctx,
        args.block[0],
        args.block[1],
        args.block[2],
        args.dynamic_smem,
        args.grid[0],
        args.grid[1],
        args.grid[2],
        args.stream,
        launchConfigType(ctx, args.max_attrs),
        .int(ctx, .i32, args.max_attrs),
        args.location,
    );
}

pub fn launch_cfg_cluster_dim(ctx: *mlir.Context, config: *const mlir.Value, dims: [3]*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return ops.launch_cfg_cluster_dim(ctx, config, dims[0], dims[1], dims[2], location);
}

pub fn launch_cfg_cooperative(ctx: *mlir.Context, config: *const mlir.Value, enabled: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return ops.launch_cfg_cooperative(ctx, config, enabled, location);
}

pub fn launch_cfg_programmatic_stream_serialization_allowed(ctx: *mlir.Context, config: *const mlir.Value, enabled: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return ops.launch_cfg_programmatic_stream_serialization_allowed(ctx, config, enabled, location);
}

pub const LaunchExArgs = struct {
    config: *const mlir.Value,
    inputs: []const *const mlir.Value,
    callee: *const mlir.Attribute,
    assume_kernel_attr: ?*const mlir.Attribute = null,
    location: *const mlir.Location,
};

pub fn launch_ex(ctx: *mlir.Context, args: LaunchExArgs) *mlir.Operation {
    return ops.launch_ex(ctx, args.config, args.inputs, resultType(ctx), args.callee, null, null, args.assume_kernel_attr, args.location);
}

test {
    std.testing.refAllDecls(@This());
}
