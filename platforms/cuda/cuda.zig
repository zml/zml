const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const gpu_event = @import("platforms/gpu_event");
const pjrt = @import("pjrt");
const stdx = @import("stdx");

const compat_probe = @import("compat_probe.zig");

const nvidiaLibsPath = "/cuda/";

const log = std.log.scoped(.@"zml/platforms/cuda");

const EventFns = struct {
    create: *const fn (event: *?*anyopaque) callconv(.c) c_int,
    record: *const fn (event: *anyopaque, stream: ?*anyopaque) callconv(.c) c_int,
    synchronize: *const fn (event: *anyopaque) callconv(.c) c_int,
    elapsed_time: *const fn (milliseconds: *f32, start: *anyopaque, stop: *anyopaque) callconv(.c) c_int,
    destroy: *const fn (event: *anyopaque) callconv(.c) c_int,
    memset_async: *const fn (device_pointer: ?*anyopaque, value: c_int, count: usize, stream: ?*anyopaque) callconv(.c) c_int,
    get_error_string: *const fn (status: c_int) callconv(.c) ?[*:0]const u8,
};

const EventState = struct {
    functions: EventFns,
};

var event_state: EventState = undefined;
var event_provider: ?gpu_event.Provider = null;

const event_vtable: gpu_event.Provider.VTable = .{
    .create = createEvent,
    .marker_init_async = markerInitAsync,
    .record = recordEvent,
    .synchronize = synchronizeEvent,
    .elapsed_ms = elapsedMs,
    .destroy = destroyEvent,
    .error_string = errorString,
};

pub fn eventProvider() ?*const gpu_event.Provider {
    return if (event_provider) |*provider| provider else null;
}

fn eventState(context: *const anyopaque) *const EventState {
    return @ptrCast(@alignCast(context));
}

fn createEvent(context: *const anyopaque, stream: *pjrt.Stream) gpu_event.Result(gpu_event.Event) {
    _ = stream;
    var event: ?*anyopaque = null;
    const status = eventState(context).functions.create(&event);
    if (status != 0) return .{ .err = .{ .operation = .create, .status = status } };
    return .{ .ok = event orelse return .{ .err = .{ .operation = .create, .status = -1 } } };
}

fn markerInitAsync(context: *const anyopaque, marker: *anyopaque, stream: *pjrt.Stream) gpu_event.Result(void) {
    const status = eventState(context).functions.memset_async(marker, 0, 1, @ptrCast(stream));
    if (status != 0) return .{ .err = .{ .operation = .marker_init, .status = status } };
    return .{ .ok = {} };
}

fn recordEvent(context: *const anyopaque, event: gpu_event.Event, stream: *pjrt.Stream) gpu_event.Result(void) {
    const status = eventState(context).functions.record(event, @ptrCast(stream));
    if (status != 0) return .{ .err = .{ .operation = .record, .status = status } };
    return .{ .ok = {} };
}

fn synchronizeEvent(context: *const anyopaque, event: gpu_event.Event) gpu_event.Result(void) {
    const status = eventState(context).functions.synchronize(event);
    if (status != 0) return .{ .err = .{ .operation = .synchronize, .status = status } };
    return .{ .ok = {} };
}

fn elapsedMs(context: *const anyopaque, start: gpu_event.Event, stop: gpu_event.Event) gpu_event.Result(f32) {
    var milliseconds: f32 = undefined;
    const status = eventState(context).functions.elapsed_time(&milliseconds, start, stop);
    if (status != 0) return .{ .err = .{ .operation = .elapsed, .status = status } };
    return .{ .ok = milliseconds };
}

fn destroyEvent(context: *const anyopaque, event: gpu_event.Event) gpu_event.Result(void) {
    const status = eventState(context).functions.destroy(event);
    if (status != 0) return .{ .err = .{ .operation = .destroy, .status = status } };
    return .{ .ok = {} };
}

fn errorString(context: *const anyopaque, status: c_int) ?[*:0]const u8 {
    return eventState(context).functions.get_error_string(status);
}

fn lookup(lib: *std.DynLib, comptime T: type, name: [:0]const u8) !T {
    return lib.lookup(T, name) orelse {
        log.err("Missing CUDA runtime symbol {s}", .{name});
        return error.MissingSymbol;
    };
}

fn initEventProvider(sandbox_path: []const u8) !void {
    if (event_provider != null) return;

    var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libcudart.so.13" });
    const handle = std.c.dlopen(path, .{ .NOW = true, .GLOBAL = true, .NODELETE = true }) orelse {
        log.err("Unable to load CUDA event runtime {s}: {s}", .{ path, std.c.dlerror() orelse "unknown error" });
        return error.EventRuntimeUnavailable;
    };

    var lib: std.DynLib = .{ .inner = .{ .handle = handle } };
    event_state = .{ .functions = .{
        .create = try lookup(&lib, @FieldType(EventFns, "create"), "cudaEventCreate"),
        .record = try lookup(&lib, @FieldType(EventFns, "record"), "cudaEventRecord"),
        .synchronize = try lookup(&lib, @FieldType(EventFns, "synchronize"), "cudaEventSynchronize"),
        .elapsed_time = try lookup(&lib, @FieldType(EventFns, "elapsed_time"), "cudaEventElapsedTime"),
        .destroy = try lookup(&lib, @FieldType(EventFns, "destroy"), "cudaEventDestroy"),
        .memset_async = try lookup(&lib, @FieldType(EventFns, "memset_async"), "cudaMemsetAsync"),
        .get_error_string = try lookup(&lib, @FieldType(EventFns, "get_error_string"), "cudaGetErrorString"),
    } };
    event_provider = .{
        .backend = .cuda,
        .library_handle = handle,
        .context = &event_state,
        .vtable = &event_vtable,
    };
}

fn findCudaSandbox(
    r: anytype,
    buffer: *[std.Io.Dir.max_path_bytes]u8,
) !?[]const u8 {
    const candidate = switch (builtin.cpu.arch) {
        .aarch64 => "libpjrt_cuda_linux_arm64/sandbox",
        .x86_64 => "libpjrt_cuda_linux_amd64/sandbox",
        else => return null,
    };
    return try r.rlocation(candidate, buffer);
}

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_CUDA");
}

pub fn needsCudaCompat(io: std.Io, sandbox_path: []const u8) !bool {
    var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const nvidia_compat_path = try stdx.Io.Dir.path.bufJoinZ(&buf, &.{ sandbox_path, "bin", "compat_probe" });

    var child = try std.process.spawn(io, .{
        .argv = &[_][]const u8{nvidia_compat_path},
        .cwd = .{ .path = sandbox_path },
        .stdin = .ignore,
        .stdout = .inherit,
        .stderr = .inherit,
    });
    defer child.kill(io);

    const res = child.wait(io) catch |err| {
        log.err("Failed to run CUDA compatibility probe: {any}", .{err});
        return err;
    };
    const result: compat_probe.ExitCode = @enumFromInt(res.exited);

    return switch (result) {
        .Success => true,
        .SystemDriverMismatch, .CompatNotSupportedOnDevice => false,
        .UnexpectedError => blk: {
            log.err("CUDA compatibility probe returned unexpected error code", .{});
            break :blk false;
        },
    };
}

fn hasNvidiaDevice(io: std.Io) bool {
    for (&[_][]const u8{ "/dev/nvidiactl", "/dev/dxg" }) |dev| {
        std.Io.Dir.accessAbsolute(io, dev, .{ .read = true }) catch continue;
        return true;
    }
    return false;
}

fn hasCudaPathInLDPath() bool {
    const ldLibraryPath = std.c.getenv("LD_LIBRARY_PATH") orelse return false;
    return std.ascii.indexOfIgnoreCase(std.mem.span(ldLibraryPath), nvidiaLibsPath) != null;
}

fn setupXlaGpuCudaDirFlag(allocator: std.mem.Allocator, sandbox: []const u8) !void {
    const xla_flags = std.c.getenv("XLA_FLAGS") orelse "";
    const new_xla_flagsZ = try std.fmt.allocPrintSentinel(allocator, "{s} --xla_gpu_cuda_data_dir={s}", .{ xla_flags, sandbox }, 0);
    _ = c.setenv("XLA_FLAGS", new_xla_flagsZ, 1);
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }
    if (!hasNvidiaDevice(io)) {
        return error.Unavailable;
    }
    if (hasCudaPathInLDPath()) {
        log.warn("Detected {s} in LD_LIBRARY_PATH. This can lead to undefined behaviors and crashes", .{nvidiaLibsPath});
    }

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const r = try bazel.runfiles(bazel_builtin.current_repository);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try findCudaSandbox(r, &path_buf) orelse {
        log.err("Failed to find sandbox path for CUDA runtime", .{});
        return error.FileNotFound;
    };

    // CUDA path has to be set _before_ loading the PJRT plugin.
    // See https://github.com/openxla/xla/issues/21428
    try setupXlaGpuCudaDirFlag(arena.allocator(), sandbox_path);

    {
        const cudaCompat = needsCudaCompat(io, sandbox_path) catch |err| blk: {
            log.err("Unable to determine wether or not to use CUDA Compat, disabling: {any}", .{err});
            break :blk false;
        };

        if (cudaCompat) {
            log.warn("Detected NVIDIA GPU that requires CUDA compatibility libraries.", .{});
            var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
            const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "compat", "libcuda.so.1" });
            _ = std.c.dlopen(path, .{ .NOW = true }) orelse {
                log.warn("Failed to load CUDA compatibility library from {s}: {any}", .{ path, std.mem.span(std.c.dlerror()) });
            };
            log.info("Loaded CUDA compatibility libraries.", .{});
        }
    }

    initEventProvider(sandbox_path) catch |err| {
        log.warn("Native CUDA event timing is unavailable: {}", .{err});
    };

    return blk: {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libpjrt_cuda.so" });
        break :blk .loadFrom(path);
    };
}
