const std = @import("std");

const c = @import("c");
const cuda = @import("platforms/cuda");
const zml = @import("zml");

test "NCCL preload preserves primary contexts, restores the current context, and is idempotent" {
    if (comptime !zml.Target.cuda.isEnabled()) return error.SkipZigTest;
    std.testing.log_level = .info;
    defer std.testing.log_level = .warn;
    const io = std.testing.io;
    const platform = try zml.Platform.init(std.testing.allocator, io, .cuda, .{
        .cuda = .{ .allocator = .platform },
    });
    defer platform.deinit(std.testing.allocator, io);

    var driver = try std.DynLib.open("libcuda.so.1");
    defer driver.close();
    const get_current = driver.lookup(@TypeOf(&c.cuCtxGetCurrent), "cuCtxGetCurrent").?;
    const retain = driver.lookup(@TypeOf(&c.cuDevicePrimaryCtxRetain), "cuDevicePrimaryCtxRetain").?;
    const release = driver.lookup(@TypeOf(&c.cuDevicePrimaryCtxRelease), "cuDevicePrimaryCtxRelease_v2").?;
    const push = driver.lookup(@TypeOf(&c.cuCtxPushCurrent), "cuCtxPushCurrent_v2").?;
    const pop = driver.lookup(@TypeOf(&c.cuCtxPopCurrent), "cuCtxPopCurrent_v2").?;

    const device: c.CUdevice = @intCast(platform.devices[0].localHardwareId());
    var expected: c.CUcontext = null;
    try std.testing.expectEqual(@as(c.CUresult, c.CUDA_SUCCESS), retain(&expected, device));
    defer _ = release(device);
    try std.testing.expectEqual(@as(c.CUresult, c.CUDA_SUCCESS), push(expected));
    defer {
        var popped: c.CUcontext = null;
        _ = pop(&popped);
    }

    try platform.preloadNccl(io);
    var actual: c.CUcontext = null;
    try std.testing.expectEqual(@as(c.CUresult, c.CUDA_SUCCESS), get_current(&actual));
    try std.testing.expectEqual(expected, actual);

    // Create fresh temporary communicators after the first ones were destroyed.
    // A cold-cache profile should show no further finalization in this phase.
    var ordinals: [zml.Platform.MAX_NUM_DEVICES]i32 = undefined;
    for (platform.devices, 0..) |gpu, i| ordinals[i] = @intCast(gpu.localHardwareId());
    var additional = try cuda.preloadNccl(io, ordinals[0..platform.devices.len]);
    defer additional.deinit();
    try std.testing.expectEqual(@as(c.CUresult, c.CUDA_SUCCESS), get_current(&actual));
    try std.testing.expectEqual(expected, actual);

    var retained_again: c.CUcontext = null;
    try std.testing.expectEqual(@as(c.CUresult, c.CUDA_SUCCESS), retain(&retained_again, device));
    defer _ = release(device);
    try std.testing.expectEqual(expected, retained_again);

    const library_handle = platform.state.cuda.nccl_preload.?.nccl.inner.handle;
    try platform.preloadNccl(io);
    try std.testing.expectEqual(library_handle, platform.state.cuda.nccl_preload.?.nccl.inner.handle);
    try std.testing.expectEqual(@as(c.CUresult, c.CUDA_SUCCESS), get_current(&actual));
    try std.testing.expectEqual(expected, actual);
}
