const builtin = @import("builtin");
const std = @import("std");

const asynk = @import("async");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const runfiles = @import("runfiles");
const stdx = @import("stdx");

const ROCmEnvEntry = struct {
    name: [:0]const u8,
    rpath: []const u8,
    dirname: bool,
    mandatory: bool,
};

const rocm_env_entries: []const ROCmEnvEntry = &.{
    .{ .name = "HIPBLASLT_EXT_OP_LIBRARY_PATH", .rpath = "hipblaslt/lib/hipblaslt/library/hipblasltExtOpLibrary.dat", .dirname = false, .mandatory = false },
    .{ .name = "HIPBLASLT_TENSILE_LIBPATH", .rpath = "hipblaslt/lib/hipblaslt/library/TensileManifest.txt", .dirname = true, .mandatory = false },
    .{ .name = "ROCBLAS_TENSILE_LIBPATH", .rpath = "rocblas/lib/rocblas/library/TensileManifest.txt", .dirname = true, .mandatory = true },
    .{ .name = "ROCM_PATH", .rpath = "libpjrt_rocm/sandbox", .dirname = false, .mandatory = true },
};

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_ROCM");
}

fn hasRocmDevices() bool {
    inline for (&.{ "/dev/kfd", "/dev/dri" }) |path| {
        asynk.File.access(path, .{ .mode = .read_only }) catch return false;
    }
    return true;
}

fn setupRocmEnv() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    const r = blk: {
        var r_ = try runfiles.Runfiles.create(.{ .allocator = arena.allocator() }) orelse {
            stdx.debug.panic("Unable to find Runfiles directory", .{});
        };
        const source_repo = bazel_builtin.current_repository;
        break :blk r_.withSourceRepo(source_repo);
    };

    for (rocm_env_entries) |entry| {
        var real_path = r.rlocationAlloc(arena.allocator(), entry.rpath) catch null orelse {
            if (entry.mandatory) {
                stdx.debug.panic("Unable to find {s} in {s}", .{ entry.name, bazel_builtin.current_repository });
            }
            continue;
        };

        if (entry.dirname) {
            real_path = std.fs.path.dirname(real_path) orelse {
                stdx.debug.panic("Unable to dirname on {s}", .{real_path});
            };
        }

        _ = c.setenv(entry.name, try arena.allocator().dupeZ(u8, real_path), 1);
    }
}

pub fn load() !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }
    if (!hasRocmDevices()) {
        return error.Unavailable;
    }

    try setupRocmEnv();

    return try asynk.callBlocking(pjrt.Api.loadFrom, .{"libpjrt_rocm.so"});
}
