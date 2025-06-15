const builtin = @import("builtin");

const c = @import("c");

pub const Tracer = switch (builtin.os.tag) {
    .macos => MacOsTracer,
    // .linux => if (@hasDecl(c, "ZML_RUNTIME_CUDA")) CudaTracer else FakeTracer,
    .linux => FakeTracer,
    else => FakeTracer,
};

const CudaTracer = struct {
    extern fn cudaProfilerStart() c_int;
    extern fn cudaProfilerStop() c_int;

    extern fn nvtxMarkA(message: [*:0]const u8) void;
    extern fn nvtxRangeStartA(message: [*:0]const u8) c_int;
    extern fn nvtxRangeEnd(id: c_int) void;

    pub fn init(name: [:0]const u8) CudaTracer {
        _ = name;
        _ = cudaProfilerStart();
        return .{};
    }

    pub fn deinit(self: *const CudaTracer) void {
        _ = self;
        _ = cudaProfilerStop();
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
