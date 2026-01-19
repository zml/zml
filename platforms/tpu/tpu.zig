const std = @import("std");
const builtin = @import("builtin");

const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const runfiles = @import("runfiles");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/tpu");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_TPU");
}

/// Check if running on Google Compute Engine, because TPUs will poll the
/// metadata server, hanging the process. So only do it on GCP.
/// Do it using the official method at:
/// https://cloud.google.com/compute/docs/instances/detect-compute-engine?hl=en#use_operating_system_tools_to_detect_if_a_vm_is_running_in
fn isOnGCP(io: std.Io) !bool {
    const GoogleComputeEngine = "Google Compute Engine";
    var buffer: [GoogleComputeEngine.len]u8 = undefined;
    return std.mem.eql(
        u8,
        GoogleComputeEngine,
        try std.Io.Dir.readFile(
            .cwd(),
            io,
            "/sys/devices/virtual/dmi/id/product_name",
            &buffer,
        ),
    );
}

pub fn load(io: std.Io) !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }
    if (!(isOnGCP(io) catch false)) {
        return error.Unavailable;
    }

    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();

    var r_ = try runfiles.Runfiles.create(.{ .allocator = arena.allocator() }) orelse {
        stdx.debug.panic("Unable to find runfiles", .{});
    };

    const source_repo = bazel_builtin.current_repository;
    const r = r_.withSourceRepo(source_repo);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libpjrt_tpu/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for TPU runtime", .{});
        return error.FileNotFound;
    };

    return blk: {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libpjrt_tpu.so" });
        var future = io.async(struct {
            fn call(path_: [:0]const u8) !*const pjrt.Api {
                return pjrt.Api.loadFrom(path_);
            }
        }.call, .{path});
        break :blk future.await(io);
    };
}
