const std = @import("std");

const c = @import("c");

const log = std.log.scoped(.@"zml/platforms/cuda/nccl");

/// Owns library references, not communicators or CUDA contexts. PJRT keeps the
/// primary contexts alive after the temporary communicators have been destroyed.
pub const Preload = struct {
    nccl: std.DynLib,
    cuda: std.DynLib,

    pub fn init(io: std.Io, path: [:0]const u8, devices: []const i32) !Preload {
        log.info("NCCL preload/JIT begin: devices={any}; loading kernels including any CUDA JIT/finalization", .{devices});
        const begin = std.Io.Timestamp.now(io, .awake);
        errdefer log.err("NCCL preload/JIT failed", .{});
        var nccl = try std.DynLib.open(path);
        errdefer nccl.close();
        // PJRT has already selected and loaded the system or compatibility driver.
        // NOLOAD avoids loading a second driver if that assumption is violated.
        const driver_handle = std.c.dlopen("libcuda.so.1", .{ .NOW = true, .NOLOAD = true }) orelse {
            log.err("The PJRT CUDA driver must be loaded before preloading NCCL", .{});
            return error.CudaDriverNotLoaded;
        };
        var cuda: std.DynLib = .{ .inner = .{ .handle = driver_handle } };
        errdefer cuda.close();
        const api = try Api.init(&nccl, &cuda);
        var version: c_int = undefined;
        try api.checkNccl(api.getVersion(&version), "ncclGetVersion");
        log.info("NCCL version={d}, library={s}", .{ version, path });
        for (devices) |ordinal| {
            log.info("NCCL kernel initialization/JIT begin: CUDA device {d}", .{ordinal});
            const device_begin = std.Io.Timestamp.now(io, .awake);
            api.initializeDevice(ordinal) catch |err| {
                log.err("NCCL kernel initialization/JIT failed on CUDA device {d}: {}", .{ ordinal, err });
                return err;
            };
            const elapsed_ns = std.Io.Timestamp.now(io, .awake).nanoseconds - device_begin.nanoseconds;
            log.info("NCCL kernel initialization/JIT end: CUDA device {d}, {d} ms", .{ ordinal, @divTrunc(elapsed_ns, std.time.ns_per_ms) });
        }
        const elapsed_ns = std.Io.Timestamp.now(io, .awake).nanoseconds - begin.nanoseconds;
        log.info("NCCL preload/JIT end: {d} ms", .{@divTrunc(elapsed_ns, std.time.ns_per_ms)});
        return .{ .nccl = nccl, .cuda = cuda };
    }

    pub fn deinit(self: *Preload) void {
        self.nccl.close();
        self.cuda.close();
    }
};

// Public NCCL ABI: result codes are C enums and communicators are opaque pointers.
const Result = c_uint;
const Comm = ?*anyopaque;
const InitAll = *const fn ([*]Comm, c_int, [*]const c_int) callconv(.c) Result;
const CommOp = *const fn (Comm) callconv(.c) Result;

const Api = struct {
    getVersion: *const fn (*c_int) callconv(.c) Result,
    initAll: InitAll,
    destroy: CommOp,
    abort: CommOp,
    getAsyncError: *const fn (Comm, *Result) callconv(.c) Result,
    ncclErrorString: *const fn (Result) callconv(.c) [*:0]const u8,
    getCurrent: @TypeOf(&c.cuCtxGetCurrent),
    setCurrent: @TypeOf(&c.cuCtxSetCurrent),
    cudaErrorString: @TypeOf(&c.cuGetErrorString),

    fn init(nccl: *std.DynLib, cuda: *std.DynLib) !Api {
        return .{
            .getVersion = try lookup(nccl, *const fn (*c_int) callconv(.c) Result, "ncclGetVersion"),
            .initAll = try lookup(nccl, InitAll, "ncclCommInitAll"),
            .destroy = try lookup(nccl, CommOp, "ncclCommDestroy"),
            .abort = try lookup(nccl, CommOp, "ncclCommAbort"),
            .getAsyncError = try lookup(nccl, *const fn (Comm, *Result) callconv(.c) Result, "ncclCommGetAsyncError"),
            .ncclErrorString = try lookup(nccl, *const fn (Result) callconv(.c) [*:0]const u8, "ncclGetErrorString"),
            .getCurrent = try lookup(cuda, @TypeOf(&c.cuCtxGetCurrent), "cuCtxGetCurrent"),
            .setCurrent = try lookup(cuda, @TypeOf(&c.cuCtxSetCurrent), "cuCtxSetCurrent"),
            .cudaErrorString = try lookup(cuda, @TypeOf(&c.cuGetErrorString), "cuGetErrorString"),
        };
    }

    fn initializeDevice(self: Api, ordinal: i32) !void {
        // NCCL selects the device's primary context, already retained by PJRT.
        // Restore the caller's context, including when it was initially null.
        // Keep the whole scope synchronous because current contexts are thread-local.
        var previous: c.CUcontext = null;
        try self.checkCuda(self.getCurrent(&previous), "cuCtxGetCurrent");
        const init_result = self.initializeCommunicator(ordinal);
        const restore_result = self.checkCuda(self.setCurrent(previous), "cuCtxSetCurrent");
        try init_result;
        try restore_result;
    }

    fn initializeCommunicator(self: Api, ordinal: i32) !void {
        var comm: Comm = null;
        errdefer if (comm != null) {
            self.checkNccl(self.abort(comm), "ncclCommAbort") catch {};
        };
        var result = self.initAll(@ptrCast(&comm), 1, @ptrCast(&ordinal));
        // NCCL_COMM_BLOCKING can override the default blocking initialization.
        while (result == 7) { // ncclInProgress
            const pause: std.c.timespec = .{ .sec = 0, .nsec = std.time.ns_per_ms };
            _ = std.c.nanosleep(&pause, null);
            try self.checkNccl(self.getAsyncError(comm, &result), "ncclCommGetAsyncError");
        }
        try self.checkNccl(result, "ncclCommInitAll");
        const destroy_result = self.destroy(comm);
        // The communicator must not be accessed after ncclCommDestroy returns.
        comm = null;
        if (destroy_result != 7) try self.checkNccl(destroy_result, "ncclCommDestroy");
    }

    fn checkNccl(self: Api, result: Result, operation: []const u8) !void {
        if (result == 0) return;
        log.err("{s} failed: {s} ({d})", .{ operation, self.ncclErrorString(result), result });
        return error.NcclInitializationFailed;
    }

    fn checkCuda(self: Api, result: c.CUresult, operation: []const u8) !void {
        if (result == c.CUDA_SUCCESS) return;
        var message: [*c]const u8 = null;
        _ = self.cudaErrorString(result, &message);
        log.err("{s} failed: {s} ({d})", .{ operation, if (message != null) std.mem.span(message) else "unknown CUDA error", result });
        return error.CudaInitializationFailed;
    }
};

fn lookup(library: *std.DynLib, comptime T: type, name: [:0]const u8) !T {
    return library.lookup(T, name) orelse {
        log.err("Missing NCCL preload symbol: {s}", .{name});
        return error.SymbolNotFound;
    };
}
