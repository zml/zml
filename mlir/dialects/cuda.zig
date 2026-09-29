//! Builders for the NVIDIA CUDA host-program dialect used by CuTe DSL.
//!
//! The dialect itself is registered by NVIDIA's runtime compiler. ZML builds
//! these operations in an allow-unregistered context, so constructors mirror
//! the ODS schemas from NVIDIA CUTLASS DSL's generated `_cuda_ops_gen.py`.
const std = @import("std");

const mlir = @import("mlir");
const stdx = @import("stdx");

pub const dialect_namespace = "cuda";

pub fn streamType(ctx: *mlir.Context) *const mlir.Type {
    return mlir.Type.parse(ctx, "!cuda.stream") catch @panic("CUDA stream type is unavailable");
}

pub fn resultType(ctx: *mlir.Context) *const mlir.Type {
    return mlir.Type.parse(ctx, "!cuda.result") catch @panic("CUDA result type is unavailable");
}

pub fn launchConfigType(ctx: *mlir.Context, max_attrs: u32) *const mlir.Type {
    var buffer: [64]u8 = undefined;
    const text = std.fmt.bufPrint(&buffer, "!cuda.launch_cfg<max_attrs = {d}>", .{max_attrs}) catch @panic("CUDA launch-config type overflow");
    return mlir.Type.parse(ctx, text) catch @panic("CUDA launch-config type is unavailable");
}

pub const KernelArgs = struct {
    name: []const u8,
    function_type: *const mlir.Type,
    body: *mlir.Block,
    arg_attrs: ?*const mlir.Attribute = null,
    extra_attributes: []const mlir.NamedAttribute = &.{},
    location: *const mlir.Location,
};

pub fn kernel(ctx: *mlir.Context, args: KernelArgs) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 16) = .empty;
    attrs.appendSliceAssumeCapacity(&.{
        .named(ctx, "sym_name", .string(ctx, args.name)),
        .named(ctx, "function_type", .typeAttr(args.function_type)),
    });
    if (args.arg_attrs) |attr| attrs.appendAssumeCapacity(.named(ctx, "arg_attrs", attr));
    attrs.appendSliceAssumeCapacity(args.extra_attributes);
    return mlir.Operation.make(ctx, "cuda.kernel", .{
        .blocks = &.{args.body},
        .attributes = attrs.constSlice(),
        .location = args.location,
        .verify = false,
    });
}

pub fn return_(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.return", .{
        .operands = .{ .flat = values },
        .location = location,
        .verify = false,
    });
}

pub fn cast(ctx: *mlir.Context, dst: *const mlir.Type, src: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.cast", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst} },
        .location = location,
        .verify = false,
    });
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
    return mlir.Operation.make(ctx, "cuda.launch_cfg.create", .{
        .operands = .{ .flat = &.{
            args.block[0], args.block[1], args.block[2], args.dynamic_smem,
            args.grid[0],  args.grid[1],  args.grid[2],  args.stream,
        } },
        .results = .{ .flat = &.{launchConfigType(ctx, args.max_attrs)} },
        .attributes = &.{.named(ctx, "maxNumAttrs", .int(ctx, .i32, args.max_attrs))},
        .location = args.location,
        .verify = false,
    });
}

pub fn launch_cfg_cluster_dim(ctx: *mlir.Context, config: *const mlir.Value, dims: [3]*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.launch_cfg.cluster_dim", .{
        .operands = .{ .flat = &.{ config, dims[0], dims[1], dims[2] } },
        .location = location,
        .verify = false,
    });
}

fn launchCfgFlag(ctx: *mlir.Context, comptime name: []const u8, config: *const mlir.Value, enabled: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, name, .{
        .operands = .{ .flat = &.{ config, enabled } },
        .location = location,
        .verify = false,
    });
}

pub fn launch_cfg_cooperative(ctx: *mlir.Context, config: *const mlir.Value, enabled: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return launchCfgFlag(ctx, "cuda.launch_cfg.cooperative", config, enabled, location);
}

pub fn launch_cfg_programmatic_stream_serialization_allowed(ctx: *mlir.Context, config: *const mlir.Value, enabled: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return launchCfgFlag(ctx, "cuda.launch_cfg.programmatic_stream_serialization_allowed", config, enabled, location);
}

pub const LaunchExArgs = struct {
    config: *const mlir.Value,
    inputs: []const *const mlir.Value,
    callee: *const mlir.Attribute,
    assume_kernel_attr: ?*const mlir.Attribute = null,
    location: *const mlir.Location,
};

pub fn launch_ex(ctx: *mlir.Context, args: LaunchExArgs) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 128) = .empty;
    operands.appendAssumeCapacity(args.config);
    operands.appendSliceAssumeCapacity(args.inputs);
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "callee", args.callee));
    if (args.assume_kernel_attr) |attr| attrs.appendAssumeCapacity(.named(ctx, "assume_kernel_attr", attr));
    return mlir.Operation.make(ctx, "cuda.launch_ex", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{resultType(ctx)} },
        .attributes = attrs.constSlice(),
        .location = args.location,
        .verify = false,
    });
}

test {
    std.testing.refAllDecls(@This());
}
