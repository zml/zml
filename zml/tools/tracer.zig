const builtin = @import("builtin");

pub const Tracer = switch (builtin.os.tag) {
    .macos => MacOsTracer,
    else => FakeTracer,
};

const MacOsTracer = struct {
    const c = @import("c");

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

    pub fn event(self: *const MacOsTracer, message: [:0]const u8) void {
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
