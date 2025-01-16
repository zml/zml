const builtin = @import("builtin");
const std = @import("std");

const asynk = @import("async");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const runfiles = @import("runfiles");
const stdx = @import("stdx");

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

    const paths = .{
        .{ "HIPBLASLT_EXT_OP_LIBRARY_PATH", "hipblaslt/lib/hipblaslt/library/hipblasltExtOpLibrary.dat", false },
        .{ "HIPBLASLT_TENSILE_LIBPATH", "hipblaslt/lib/hipblaslt/library/TensileManifest.txt", true },
        .{ "ROCBLAS_TENSILE_LIBPATH", "rocblas/lib/rocblas/library/TensileManifest.txt", true },
        .{ "ROCM_PATH", "libpjrt_rocm/sandbox", false },
    };

    const r = blk: {
        var r_ = try runfiles.Runfiles.create(.{ .allocator = arena.allocator() }) orelse {
            stdx.debug.panic("Unable to find Runfiles directory", .{});
        };
        const source_repo = bazel_builtin.current_repository;
        break :blk r_.withSourceRepo(source_repo);
    };

    inline for (paths) |path| {
        const name = path[0];
        const rpath = path[1];
        const dirname = path[2];

        var real_path = r.rlocationAlloc(arena.allocator(), rpath) catch null orelse {
            stdx.debug.panic("Unable to find " ++ name ++ " in " ++ bazel_builtin.current_repository, .{});
        };

        if (dirname) {
            real_path = std.fs.path.dirname(real_path) orelse {
                stdx.debug.panic("Unable to dirname on {s}", .{real_path});
            };
        }

        _ = c.setenv(name, try arena.allocator().dupeZ(u8, real_path), 1);
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

    return try pjrt.Api.loadFrom("libpjrt_rocm.so");
}
