const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const gpu_event = @import("platforms/gpu_event");
const pjrt = @import("pjrt");
const runfiles = @import("runfiles");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/rocm");

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
        log.err("Missing ROCm runtime symbol {s}", .{name});
        return error.MissingSymbol;
    };
}

fn initEventProvider(sandbox_path: []const u8) !void {
    if (event_provider != null) return;

    var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libamdhip64.so.7" });
    const handle = std.c.dlopen(path, .{ .NOW = true, .GLOBAL = true, .NODELETE = true }) orelse {
        log.err("Unable to load ROCm event runtime {s}: {s}", .{ path, std.c.dlerror() orelse "unknown error" });
        return error.EventRuntimeUnavailable;
    };

    var lib: std.DynLib = .{ .inner = .{ .handle = handle } };
    event_state = .{ .functions = .{
        .create = try lookup(&lib, @FieldType(EventFns, "create"), "hipEventCreate"),
        .record = try lookup(&lib, @FieldType(EventFns, "record"), "hipEventRecord"),
        .synchronize = try lookup(&lib, @FieldType(EventFns, "synchronize"), "hipEventSynchronize"),
        .elapsed_time = try lookup(&lib, @FieldType(EventFns, "elapsed_time"), "hipEventElapsedTime"),
        .destroy = try lookup(&lib, @FieldType(EventFns, "destroy"), "hipEventDestroy"),
        .memset_async = try lookup(&lib, @FieldType(EventFns, "memset_async"), "hipMemsetAsync"),
        .get_error_string = try lookup(&lib, @FieldType(EventFns, "get_error_string"), "hipGetErrorString"),
    } };
    event_provider = .{
        .backend = .rocm,
        .library_handle = handle,
        .context = &event_state,
        .vtable = &event_vtable,
    };
}

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_ROCM");
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
    const sandbox_path = try r.rlocation("libpjrt_rocm/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for ROCm runtime", .{});
        return error.FileNotFound;
    };

    try setupRocmEnv(sandbox_path);
    initEventProvider(sandbox_path) catch |err| {
        log.warn("Native ROCm event timing is unavailable: {}", .{err});
    };

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
        const lib_path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libpjrt_rocm.so" });
        break :blk .loadFrom(lib_path);
    };
}
