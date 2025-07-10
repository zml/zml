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
    .{ .name = "HIPBLASLT_EXT_OP_LIBRARY_PATH", .rpath = "/lib/hipblaslt/library/hipblasltExtOpLibrary.dat", .dirname = false, .mandatory = false },
    .{ .name = "HIPBLASLT_TENSILE_LIBPATH", .rpath = "/lib/hipblaslt/library/TensileManifest.txt", .dirname = true, .mandatory = false },
    .{ .name = "ROCBLAS_TENSILE_LIBPATH", .rpath = "/lib/rocblas/library/TensileManifest.txt", .dirname = true, .mandatory = true },
    .{ .name = "ROCM_PATH", .rpath = "/", .dirname = false, .mandatory = true },
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

fn setupRocmEnv(allocator: std.mem.Allocator, rocm_data_dir: []const u8) !void {
    for (rocm_env_entries) |entry| {
        var real_path: []const u8 = std.fmt.allocPrintZ(allocator, "{s}/{s}", .{rocm_data_dir, entry.rpath}) catch null orelse {
            if (entry.mandatory) {
                stdx.debug.panic("Unable to find {s} in {s}\n", .{ entry.name, bazel_builtin.current_repository });
            }
            continue;
        };

        if (entry.dirname) {
            real_path = std.fs.path.dirname(real_path) orelse {
                stdx.debug.panic("Unable to dirname on {s}", .{real_path});
            };
        }

        _ = c.setenv(entry.name, try allocator.dupeZ(u8, real_path), 1);
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


    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    const r = blk: {
        var r_ = try runfiles.Runfiles.create(.{ .allocator = arena.allocator() }) orelse {
            stdx.debug.panic("Unable to find Runfiles directory", .{});
        };
        const source_repo = bazel_builtin.current_repository;
        break :blk r_.withSourceRepo(source_repo);
    };
    const rocm_data_dir = (try r.rlocationAlloc(arena.allocator(), "libpjrt_rocm/sandbox")).?;
    std.debug.print("found {s}\n", .{rocm_data_dir});

    try setupRocmEnv(arena.allocator(), rocm_data_dir);

    const library = try std.fmt.allocPrintZ(arena.allocator(), "{s}/lib/libpjrt_rocm.so", .{rocm_data_dir});

    // We must load the PJRT plugin from the main thread.
    //
    // This is because libamdhip64.so use thread local storage as part of the static destructors...
    //
    // This destructor accesses a thread-local variable. If the destructor is
    // executed in a different thread than the one that originally called dlopen()
    // on the library, the thread-local storage (TLS) offset may be resolved
    // relative to the TLS base of the main thread, rather than the thread actually
    // executing the destructor. Accessing this variable results in a segmentation fault...
    return try pjrt.Api.loadFrom(library);
}
