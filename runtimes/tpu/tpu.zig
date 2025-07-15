const builtin = @import("builtin");
const std = @import("std");

const asynk = @import("async");
const pjrt = @import("pjrt");
const c = @import("c");
const stdx = @import("stdx");
const bazel_builtin = @import("bazel_builtin");
const runfiles = @import("runfiles");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_TPU");
}

/// Check if running on Google Compute Engine, because TPUs will poll the
/// metadata server, hanging the process. So only do it on GCP.
/// Do it using the official method at:
/// https://cloud.google.com/compute/docs/instances/detect-compute-engine?hl=en#use_operating_system_tools_to_detect_if_a_vm_is_running_in
fn isOnGCP() !bool {
    // TODO: abstract that in the client and fail init
    const GoogleComputeEngine = "Google Compute Engine";

    var f = try asynk.File.open("/sys/devices/virtual/dmi/id/product_name", .{ .mode = .read_only });
    defer f.close() catch {};

    var buf = [_]u8{0} ** GoogleComputeEngine.len;
    _ = try f.reader().readAll(&buf);

    return std.mem.eql(u8, &buf, GoogleComputeEngine);
}

pub fn load() !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }
    if (!(isOnGCP() catch false)) {
        return error.Unavailable;
    }

    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    var r_ = try runfiles.Runfiles.create(.{ .allocator = arena.allocator() }) orelse {
        stdx.debug.panic("Unable to find CUDA directory", .{});
    };

    const source_repo = bazel_builtin.current_repository;
    const r = r_.withSourceRepo(source_repo);
    const cuda_data_dir = (try r.rlocationAlloc(arena.allocator(), "libpjrt_tpu/sandbox")).?;

    const library = try std.fmt.allocPrintZ(arena.allocator(), "{s}/lib/libpjrt_tpu.so", .{cuda_data_dir});

    return try asynk.callBlocking(pjrt.Api.loadFrom, .{library});
}
