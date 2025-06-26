const builtin = @import("builtin");

const c = @import("c");

pub const Tracer = switch (builtin.os.tag) {
    .macos => MacOsTracer,
    .linux => if (@hasDecl(c, "ZML_RUNTIME_CUDA")) CudaTracer else FakeTracer,
    else => FakeTracer,
};

const CudaTracer = struct {

    // Those symbols are defined in cudaProfiler.h but their implementation is in libcuda.so
    // They will be bound at call time after libcuda.so is loaded (as a needed dependency of libpjrt_cuda.so).
    const cuProfilerStart = @extern(*const fn () callconv(.C) c_int, .{ .name = "cuProfilerStart", .linkage = .weak }) orelse unreachable;
    const cuProfilerStop = @extern(*const fn () callconv(.C) c_int, .{ .name = "cuProfilerStop", .linkage = .weak }) orelse unreachable;

    // Those symbols are defined in nvToolsExt.h which we don't want to provide.
    // However, we link with libnvToolsExt.so which provides them.
    // They will be bound at call time after libnvToolsExt.so is loaded (manually dlopen'ed by us).
    const nvtxMarkA = @extern(*const fn ([*:0]const u8) callconv(.C) void, .{ .name = "nvtxMarkA", .linkage = .weak }) orelse unreachable;
    const nvtxRangeStartA = @extern(*const fn ([*:0]const u8) callconv(.C) c_int, .{ .name = "nvtxRangeStartA", .linkage = .weak }) orelse unreachable;
    const nvtxRangeEnd = @extern(*const fn (c_int) callconv(.C) void, .{ .name = "nvtxRangeEnd", .linkage = .weak }) orelse unreachable;

    pub fn init(name: [:0]const u8) CudaTracer {
        _ = name;
        _ = cuProfilerStart();
        return .{};
    }

    pub fn deinit(self: *const CudaTracer) void {
        _ = self;
        _ = cuProfilerStop();
    }

    pub fn event(self: *const CudaTracer, message: [:0]const u8) void {
        _ = self;
        nvtxMarkA(message.ptr);
    }

    pub fn frameStart(self: *const CudaTracer, message: [:0]const u8) u64 {
        _ = self;
        return @intCast(nvtxRangeStartA(message.ptr));
    }

    pub fn frameEnd(self: *const CudaTracer, interval_id: u64, message: [:0]const u8) void {
        _ = self;
        _ = message;
        nvtxRangeEnd(@intCast(interval_id));
        return;
    }
};

const MacOsTracer = struct {
    logger: c.os_log_t,

    pub fn init(name: [:0]const u8) MacOsTracer {
        const logger = c.os_log_create(name.ptr, c.OS_LOG_CATEGORY_POINTS_OF_INTEREST);
        return .{
            .logger = logger,
        };
    }

    pub fn event(self: *const MacOsTracer, message: [:0]const u8) void {
        const interval_id = c.os_signpost_id_generate(self.logger);
        c.zml_os_signpost_event(self.logger, interval_id, message);
    }

    pub fn frameStart(self: *const MacOsTracer, message: [:0]const u8) c.os_signpost_id_t {
        const interval_id = c.os_signpost_id_generate(self.logger);
        c.zml_os_signpost_interval_begin(self.logger, interval_id, message);
        return interval_id;
    }

    pub fn frameEnd(self: *const MacOsTracer, interval_id: c.os_signpost_id_t, message: [:0]const u8) void {
        c.zml_os_signpost_interval_end(self.logger, interval_id, message);
    }
};

/// Mock tracer for OS which don't have an impl.
const FakeTracer = struct {
    pub fn init(name: [:0]const u8) FakeTracer {
        _ = name;
        return .{};
    }

    pub fn deinit(self: *const FakeTracer) void {
        _ = self;
    }

    pub fn event(self: *const FakeTracer, message: [:0]const u8) void {
        _ = self;
        _ = message;
        return;
    }

    pub fn frameStart(self: *const FakeTracer, message: [:0]const u8) u64 {
        _ = self;
        _ = message;
        return 0;
    }

    pub fn frameEnd(self: *const FakeTracer, interval_id: u64, message: [:0]const u8) void {
        _ = self;
        _ = interval_id;
        _ = message;
        return;
    }
};
