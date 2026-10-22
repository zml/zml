const std = @import("std");

const mlir = @import("mlir");

/// Discardable ROCDL attributes for an `llvm.func` operation.
pub const FunctionOptions = struct {
    kernel: bool = false,
    reqdWorkGroupSize: ?[3]i32 = null,
    flatWorkGroupSize: ?[]const u8 = null,
    maxFlatWorkGroupSize: ?i32 = null,
    wavesPerEu: ?WavesPerEu = null,
    unsafeFpAtomics: ?bool = null,

    pub const WavesPerEu = struct {
        min: u32,
        max: ?u32 = null,
    };
};

pub fn annotateFunction(ctx: *mlir.Context, op: *mlir.Operation, opts: FunctionOptions) void {
    if (opts.kernel) op.setAttributeByName("rocdl.kernel", .unit(ctx));
    if (opts.reqdWorkGroupSize) |size| op.setAttributeByName("rocdl.reqd_work_group_size", .denseArray(ctx, .i32, &size));
    if (opts.flatWorkGroupSize) |size| op.setAttributeByName("rocdl.flat_work_group_size", .string(ctx, size));
    if (opts.maxFlatWorkGroupSize) |size| op.setAttributeByName("rocdl.max_flat_work_group_size", .int(ctx, .i32, size));
    if (opts.wavesPerEu) |waves| {
        var buffer: [24]u8 = undefined;
        const value = if (waves.max) |max| std.fmt.bufPrint(&buffer, "{d},{d}", .{ waves.min, max }) else std.fmt.bufPrint(&buffer, "{d}", .{waves.min});
        op.setAttributeByName("rocdl.waves_per_eu", .string(ctx, value catch unreachable));
    }
    if (opts.unsafeFpAtomics) |enabled| op.setAttributeByName("rocdl.unsafe_fp_atomics", .boolean(ctx, enabled));
}

/// A missing flag preserves the target's default; false explicitly disables it.
pub const ModuleOptions = struct {
    xnack: ?bool = null,
    sramecc: ?bool = null,
};

pub fn annotateModule(ctx: *mlir.Context, op: *mlir.Operation, opts: ModuleOptions) void {
    if (opts.xnack) |enabled| op.setAttributeByName("rocdl.xnack", .boolean(ctx, enabled));
    if (opts.sramecc) |enabled| op.setAttributeByName("rocdl.sramecc", .boolean(ctx, enabled));
}

pub const MemoryOptions = struct {
    lastUse: bool = false,
    noRemoteMemory: bool = false,
    noFineGrainedMemory: bool = false,
    ignoreDenormalMode: bool = false,
};

pub fn annotateMemory(ctx: *mlir.Context, op: *mlir.Operation, opts: MemoryOptions) void {
    if (opts.lastUse) op.setAttributeByName("rocdl.last_use", .unit(ctx));
    if (opts.noRemoteMemory) op.setAttributeByName("rocdl.no_remote_memory", .unit(ctx));
    if (opts.noFineGrainedMemory) op.setAttributeByName("rocdl.no_fine_grained_memory", .unit(ctx));
    if (opts.ignoreDenormalMode) op.setAttributeByName("rocdl.ignore_denormal_mode", .unit(ctx));
}
