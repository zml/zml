const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const platforms_options = @import("platforms/options");
const runfiles = @import("runfiles");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/rocm");

pub fn isEnabled() bool {
    return platforms_options.rocm_enabled;
}

fn hasRocmDevices(io: std.Io) bool {
    inline for (&.{ "/dev/kfd", "/dev/dri" }) |path| {
        std.Io.Dir.accessAbsolute(io, path, .{ .read = true }) catch return false;
    }
    return true;
}

fn setupRocmEnv(rocm_data_dir: []const u8) !void {
    var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    _ = c.setenv("ROCM_PATH", try stdx.Io.Dir.path.bufJoinZ(&buf, &.{rocm_data_dir}), 1); // must be zero terminated

    // Use one HIP graph stream to reduce cross-stream synchronization overhead.
    _ = c.setenv("DEBUG_HIP_FORCE_GRAPH_QUEUES", "1", 0);
    // Share one hardware queue per device so auxiliary barriers stay on the kernel queue.
    _ = c.setenv("GPU_MAX_HW_QUEUES", "1", 0);
    // Disable rocprofiler queue interposition to avoid its dispatch overhead outside profiling.
    _ = c.setenv("ROCPROFILER_QUEUE_INTERPOSITION", "0", 0);

    if (c.getenv("ROCPROFILER_LOG_LEVEL") == null) {
        // rocprofiler defaults to the `warning` log level, which can produce unnecessary log output.
        // Set the log level to `fatal` to suppress all but fatal error messages.
        _ = c.setenv("ROCPROFILER_LOG_LEVEL", "error", 0);
    }
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    _ = allocator;
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }
    if (!hasRocmDevices(io)) {
        return error.Unavailable;
    }

    const r = try bazel.runfiles(bazel_builtin.current_repository);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libzml_rocm/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for ROCm runtime", .{});
        return error.FileNotFound;
    };

    try setupRocmEnv(sandbox_path);

    // We must load the PJRT plugin from the main thread.
    //
    // This is because libamdhip64.so use thread local storage as part of the static destructors...
    //
    // This destructor accesses a thread-local variable. If the destructor is
    // executed in a different thread than the one that originally called dlopen()
    // on the library, the thread-local storage (TLS) offset may be resolved
    // relative to the TLS base of the main thread, rather than the thread actually
    // executing the destructor. Accessing this variable results in a segmentation fault...
    return blk: {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const lib_path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libzml_rocm.so" });
        const api = try pjrt.Api.loadFrom(lib_path);
        loaded_hip = Hip.loaded();
        break :blk api;
    };
}

/// Whether `releaseRuntimeDeviceMemory` asks the devices for their whole
/// memory. The first call that finds the HIP runtime missing, or a device
/// that does not refuse, logs it and clears this for good. It lives here,
/// not with a PJRT client: the HIP runtime the plugin loaded, and the
/// memory it keeps, belong to the process and serve every client.
var releases_runtime_memory: std.atomic.Value(bool) = .init(true);
/// The HIP calls of `releaseRuntimeDeviceMemory`, resolved by `load`, which
/// zml calls once per process; null when `libamdhip64.so.7` is not loaded
/// or lacks one of them.
var loaded_hip: ?Hip = null;

/// Makes the HIP runtime the plugin loaded give back the device memory it
/// keeps outside the plugin's allocator, chiefly the scratch of kernels
/// that spill registers, which ROCr keeps bound to its queues and in its
/// caches until the driver refuses one of its allocations. It asks every
/// device in turn for its whole memory, rounded down to 2 MiB: the driver
/// refuses, ROCr reclaims the scratch of the device's queues and frees its
/// caches, and HIP answers `hipErrorOutOfMemory`. Kernels that spill later
/// map their scratch again. Every queue in the process that holds scratch
/// on the device is suspended and resumed, and the call waits for its
/// kernels using scratch while holding the device's ROCr memory lock:
/// running work and allocations on that device stall meanwhile, a stall,
/// not a deadlock. Restores the thread's HIP device, and clears any HIP
/// error left on the thread, earlier ones included.
///
/// Returns whether every device refused. A missing `libamdhip64.so.7` or
/// symbol, or a device that does not refuse (a grant is freed at once), is
/// logged once and ends the trim for good: the remaining devices are not
/// asked, nor any device in later calls.
pub fn releaseRuntimeDeviceMemory(hip_ordinals: []const c_int) bool {
    if (comptime !isEnabled() or builtin.os.tag != .linux) return false;
    if (!releases_runtime_memory.load(.monotonic)) return false;
    const hip = loaded_hip orelse {
        if (releases_runtime_memory.swap(false, .monotonic)) {
            log.warn("The HIP runtime (libamdhip64.so.7) is not loaded or lacks a symbol; the device memory it keeps is no longer given back", .{});
        }
        return false;
    };

    var previous: c_int = 0;
    const has_previous = hip.hipGetDevice(&previous) == hip_success;
    defer {
        if (has_previous) _ = hip.hipSetDevice(previous);
        _ = hip.hipGetLastError();
    }

    for (hip_ordinals) |ordinal| {
        const answer = hip.requestWholeDevice(ordinal);
        if (answer == .refused) continue;
        if (releases_runtime_memory.swap(false, .monotonic)) {
            log.warn("HIP device {d} did not refuse a request for its whole memory: {f}; the device memory the HIP runtime keeps is no longer given back", .{ ordinal, answer });
        }
        return false;
    }
    return true;
}

/// `hipError_t` values (hip_runtime_api.h).
const hip_success: c_int = 0;
const hip_error_out_of_memory: c_int = 2;

/// The HIP calls `releaseRuntimeDeviceMemory` makes, from the HIP runtime
/// already in the process.
const Hip = struct {
    hipGetDevice: *const fn (device: *c_int) callconv(.c) c_int,
    hipSetDevice: *const fn (device: c_int) callconv(.c) c_int,
    hipMemGetInfo: *const fn (free: *usize, total: *usize) callconv(.c) c_int,
    hipMalloc: *const fn (ptr: *?*anyopaque, size: usize) callconv(.c) c_int,
    hipFree: *const fn (ptr: ?*anyopaque) callconv(.c) c_int,
    hipGetLastError: *const fn () callconv(.c) c_int,

    /// Only finds the library: it never loads it, nor unloads it (see the
    /// note on thread-local storage in `load`). `libzml_rocm.so` needs
    /// `libamdhip64.so.7`, and the plugin's own dlopen of `libamdhip64.so`
    /// resolves to the same file (`zmlxrocm_dlopen`).
    fn loaded() ?Hip {
        var lib: std.DynLib = .{ .inner = .{
            .handle = std.c.dlopen("libamdhip64.so.7", .{ .LAZY = true, .NOLOAD = true }) orelse return null,
        } };
        var hip: Hip = undefined;
        inline for (@typeInfo(Hip).@"struct".field_names) |name| {
            @field(hip, name) = lib.lookup(@FieldType(Hip, name), name) orelse return null;
        }
        return hip;
    }

    /// Asks HIP for the whole memory of the device, made the current one,
    /// when less than that is free, so that the driver cannot grant it.
    /// Frees what HIP grants anyway.
    fn requestWholeDevice(hip: Hip, ordinal: c_int) Answer {
        const set_device = hip.hipSetDevice(ordinal);
        if (set_device != hip_success) return .{ .set_device = set_device };
        var free: usize = 0;
        var total: usize = 0;
        const mem_get_info = hip.hipMemGetInfo(&free, &total);
        if (mem_get_info != hip_success) return .{ .mem_get_info = mem_get_info };
        // ROCr rounds a request up to 2 MiB and refuses one beyond the
        // device's size itself, without the driver's refusal that trims.
        const request = std.mem.alignBackward(usize, total, 2 << 20);
        if (free >= request) return .{ .not_asked = .{ .free = free, .request = request } };
        var ptr: ?*anyopaque = null;
        const answer = hip.hipMalloc(&ptr, request);
        if (answer == hip_error_out_of_memory) return .refused;
        if (answer == hip_success) _ = hip.hipFree(ptr);
        return .{ .malloc = answer };
    }
};

/// How a device answered `Hip.requestWholeDevice`.
const Answer = union(enum) {
    /// hipMalloc answered `hipErrorOutOfMemory`, as expected.
    refused,
    /// So much is free that the driver could grant the request.
    not_asked: struct { free: usize, request: usize },
    /// hipMalloc's other answers: `hip_success` for a grant, freed at once.
    malloc: c_int,
    /// The `hipError_t` of hipSetDevice or hipMemGetInfo, which failed.
    set_device: c_int,
    mem_get_info: c_int,

    pub fn format(answer: Answer, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        switch (answer) {
            .refused => try writer.writeAll("refused"),
            .not_asked => |memory| try writer.print("not asked, {d} bytes being free for a request of {d}", .{ memory.free, memory.request }),
            .malloc => |code| if (code == hip_success)
                try writer.writeAll("granted, then freed")
            else
                try writer.print("hipMalloc returned hipError_t {d}", .{code}),
            .set_device => |code| try writer.print("hipSetDevice returned hipError_t {d}", .{code}),
            .mem_get_info => |code| try writer.print("hipMemGetInfo returned hipError_t {d}", .{code}),
        }
    }
};
