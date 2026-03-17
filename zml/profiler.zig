const std = @import("std");
const log = std.log;

const c = @import("c");
const pjrt = @import("pjrt");
const upb = @import("upb");
const zffi = @import("ffi");

pub fn profiler(api: *const pjrt.Api, allocator: std.mem.Allocator, io: std.Io, options: ProfilerOptions) !Profiler {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var upb_alloc: upb.Allocator = .init(arena.allocator());
    const upb_arena = c.upb_Arena_Init(null, 0, upb_alloc.inner());
    defer c.upb_Arena_Free(upb_arena);

    const profile_options = try upb.new(c.tensorflow_ProfileOptions, upb_arena);
    try options.writeProto(upb_arena, profile_options);

    const serialized_options = try upb.serialize(profile_options, upb_arena);
    const inner = try api.profiler(serialized_options);

    return try .init(allocator, io, inner, options);
}

pub const ProfilerOptions = struct {
    pub const default_version: u32 = 1;

    pub const DeviceType = enum(i32) {
        unspecified = c.tensorflow_ProfileOptions_UNSPECIFIED,
        cpu = c.tensorflow_ProfileOptions_CPU,
        gpu = c.tensorflow_ProfileOptions_GPU,
        tpu = c.tensorflow_ProfileOptions_TPU,
        pluggable_device = c.tensorflow_ProfileOptions_PLUGGABLE_DEVICE,
    };

    pub const TraceOptions = struct {
        host_traceme_filter_mask: ?u64 = null,

        fn isEmpty(self: TraceOptions) bool {
            return self.host_traceme_filter_mask == null;
        }

        fn writeProto(self: TraceOptions, proto: *c.tensorflow_ProfileOptions_TraceOptions) void {
            if (self.host_traceme_filter_mask) |value| {
                c.tensorflow_ProfileOptions_TraceOptions_set_host_traceme_filter_mask(proto, value);
            }
        }
    };

    pub const AdvancedConfigValue = union(enum) {
        string: []const u8,
        boolean: bool,
        int64: i64,

        fn writeProto(self: AdvancedConfigValue, proto: *c.tensorflow_ProfileOptions_AdvancedConfigValue) void {
            switch (self) {
                .string => |value| c.tensorflow_ProfileOptions_AdvancedConfigValue_set_string_value(proto, upb.stringView(value)),
                .boolean => |value| c.tensorflow_ProfileOptions_AdvancedConfigValue_set_bool_value(proto, value),
                .int64 => |value| c.tensorflow_ProfileOptions_AdvancedConfigValue_set_int64_value(proto, value),
            }
        }
    };

    pub const AdvancedConfiguration = struct {
        key: []const u8,
        value: AdvancedConfigValue,
    };

    pub const defaults: ProfilerOptions = .{};

    version: u32 = default_version,
    repository_path: []const u8 = "/tmp/xprof",
    session_id: []const u8 = "profiling",
    device_type: ?DeviceType = .unspecified,
    include_dataset_ops: ?bool = true,
    host_tracer_level: ?u32 = 3,
    device_tracer_level: ?u32 = 3,
    python_tracer_level: ?u32 = 1,
    enable_hlo_proto: ?bool = true,
    start_timestamp_ns: ?u64 = null,
    duration_ms: ?u64 = null,
    trace_options: ?TraceOptions = null,
    advanced_configuration: []const AdvancedConfiguration = &.{},
    raise_error_on_start_failure: ?bool = null,
    override_hostname: ?[]const u8 = null,

    pub fn writeProto(self: ProfilerOptions, upb_arena: *c.upb_Arena, proto: *c.tensorflow_ProfileOptions) !void {
        c.tensorflow_ProfileOptions_set_version(proto, self.version);
        c.tensorflow_ProfileOptions_set_session_id(proto, upb.stringView(self.session_id));
        c.tensorflow_ProfileOptions_set_repository_path(proto, upb.stringView(self.repository_path));

        if (self.device_type) |value| {
            c.tensorflow_ProfileOptions_set_device_type(proto, @intFromEnum(value));
        }
        if (self.include_dataset_ops) |value| {
            c.tensorflow_ProfileOptions_set_include_dataset_ops(proto, value);
        }
        if (self.host_tracer_level) |value| {
            c.tensorflow_ProfileOptions_set_host_tracer_level(proto, value);
        }
        if (self.device_tracer_level) |value| {
            c.tensorflow_ProfileOptions_set_device_tracer_level(proto, value);
        }
        if (self.python_tracer_level) |value| {
            c.tensorflow_ProfileOptions_set_python_tracer_level(proto, value);
        }
        if (self.enable_hlo_proto) |value| {
            c.tensorflow_ProfileOptions_set_enable_hlo_proto(proto, value);
        }
        if (self.start_timestamp_ns) |value| {
            c.tensorflow_ProfileOptions_set_start_timestamp_ns(proto, value);
        }
        if (self.duration_ms) |value| {
            c.tensorflow_ProfileOptions_set_duration_ms(proto, value);
        }
        if (self.trace_options) |trace_options| {
            if (!trace_options.isEmpty()) {
                const trace_proto = try upb.new(c.tensorflow_ProfileOptions_TraceOptions, upb_arena);
                trace_options.writeProto(trace_proto);
                c.tensorflow_ProfileOptions_set_trace_options(proto, trace_proto);
            }
        }
        for (self.advanced_configuration) |entry| {
            const value_proto = try upb.new(c.tensorflow_ProfileOptions_AdvancedConfigValue, upb_arena);
            entry.value.writeProto(value_proto);
            if (!c.tensorflow_ProfileOptions_advanced_configuration_set(proto, upb.stringView(entry.key), value_proto, upb_arena)) {
                return error.OutOfMemory;
            }
        }
        if (self.raise_error_on_start_failure) |value| {
            c.tensorflow_ProfileOptions_set_raise_error_on_start_failure(proto, value);
        }
        if (self.override_hostname) |value| {
            c.tensorflow_ProfileOptions_set_override_hostname(proto, upb.stringView(value));
        }
    }
};

pub const Profiler = struct {
    arena: std.heap.ArenaAllocator,
    allocator: std.mem.Allocator,
    io: std.Io,
    inner: ?pjrt.Profiler,
    session_dir: []const u8,
    profile: Profile,

    pub const Profile = struct {
        protobuf_path: []const u8,
        perfetto_path: []const u8,
    };

    pub fn init(allocator: std.mem.Allocator, io: std.Io, inner: ?pjrt.Profiler, options: ProfilerOptions) !Profiler {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        const session_dir = try std.Io.Dir.path.join(allocator, &.{ options.repository_path, "plugins", "profile", options.session_id });
        errdefer allocator.free(session_dir);

        try std.Io.Dir.cwd().createDirPath(io, session_dir);

        const protobuf_path = try std.Io.Dir.path.join(allocator, &.{ session_dir, "profiling.xplane.pb" });
        errdefer allocator.free(protobuf_path);

        const perfetto_path = try std.Io.Dir.path.join(allocator, &.{ session_dir, "profiling.trace.json" });
        errdefer allocator.free(perfetto_path);

        return .{
            .arena = arena,
            .allocator = allocator,
            .io = io,
            .inner = inner,
            .session_dir = session_dir,
            .profile = .{
                .protobuf_path = protobuf_path,
                .perfetto_path = perfetto_path,
            },
        };
    }

    pub fn deinit(self: *Profiler) void {
        self.allocator.free(self.profile.protobuf_path);
        self.allocator.free(self.profile.perfetto_path);
        self.allocator.free(self.session_dir);
        self.arena.deinit();
    }

    pub fn start(self: *Profiler) !void {
        if (self.inner) |*inner| {
            try inner.start();
        }
    }

    pub fn stop(self: *Profiler) !?Profile {
        var inner = self.inner orelse return null;
        self.inner = null;
        try inner.stop();

        const protobuf = try inner.collectData(self.arena.allocator());

        try self.writeFile(self.profile.protobuf_path, protobuf);

        const conversion_error = c.zml_xspace_to_perfetto_dump(
            zffi.ZigSlice.from(protobuf),
            zffi.ZigSlice.from(self.profile.perfetto_path),
        );
        defer if (conversion_error.len != 0) {
            c.zml_xspace_to_perfetto_str_free(conversion_error);
        };
        if (conversion_error.len != 0) {
            log.err("Failed to convert profile protobuf to Perfetto trace: {s}", .{zffi.ZigSlice.to(u8, conversion_error)});
            return error.ProfileTraceConversionFailed;
        }

        return self.profile;
    }

    fn writeFile(self: *const Profiler, path: []const u8, contents: []const u8) !void {
        const file = try std.Io.Dir.createFile(.cwd(), self.io, path, .{});
        defer file.close(self.io);

        try file.writePositionalAll(self.io, contents, 0);
    }
};
