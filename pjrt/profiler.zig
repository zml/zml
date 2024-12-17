const std = @import("std");
const c = @import("c");
const tsl_proto = @import("//tsl:profiler_options_proto");

const log = std.log.scoped(.@"pjrt/profiler");
const TraceContainer = @import("convert/trace_container.zig").TraceContainer;

/// Pjrt Profiler extension
pub const Profiler = struct {
    api: ?c.PLUGIN_Profiler_Api,
    inner: *c.PLUGIN_Profiler,
    last_error: ?*Error = null,
    status: Status = .ready,

    pub const Status = enum { ready, started, stopped, done };
    pub const Error = c.PLUGIN_Profiler_Error;
    pub const Options = tsl_proto.ProfileOptions;

    pub const default_options: Options = .{
        .version = 1,
        .device_type = .UNSPECIFIED, // profile all devices
        .include_dataset_ops = false, // tensorflow specific
        .host_tracer_level = 2,
        .device_tracer_level = 1,
        .python_tracer_level = 0,
        .enable_hlo_proto = true,
        .start_timestamp_ns = 0,
        .duration_ms = 0,
        .repository_path = .Empty,
    };

    pub fn init(api: ?c.PLUGIN_Profiler_Api, options: ?Options) Profiler {
        if (api == null) {
            return .{ .api = null, .inner = undefined };
        }
        var options_with_timestamp = options orelse default_options;
        options_with_timestamp.start_timestamp_ns = @truncate(@max(0, std.time.nanoTimestamp()));

        var buffer: [std.fs.max_path_bytes + @sizeOf(Options) * 4]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&buffer);
        const byte_options = options_with_timestamp.encode(fba.allocator()) catch unreachable;
        var args: c.PLUGIN_Profiler_Create_Args = .{
            .options = byte_options.ptr,
            .options_size = byte_options.len,
            .profiler = undefined, // out
        };
        var res: Profiler = .{ .api = api, .inner = undefined };
        res.check(api.?.create.?(&args)) catch unreachable;

        res.inner = args.profiler.?;
        return res;
    }

    fn transition(self: *Profiler, fn_name: []const u8, expected: Status, next: Status) void {
        if (self.status == expected) {
            self.status = next;
            return;
        }
        std.debug.panic("Profiler can't `{s}()`. Current status: {}, expected: {}", .{ fn_name, self.status, expected });
    }

    pub fn start(self: *Profiler) void {
        self.transition("start", .ready, .started);
        if (self.api == null) return;
        var args: c.PLUGIN_Profiler_Start_Args = .{ .profiler = self.inner };
        self.check(self.api.?.start.?(&args)) catch unreachable;
    }

    pub fn stop(self: *Profiler) void {
        self.transition("stop", .started, .stopped);
        if (self.api == null) return;

        var args: c.PLUGIN_Profiler_Stop_Args = .{ .profiler = self.inner };
        self.check(self.api.?.stop.?(&args)) catch unreachable;
    }

    pub fn collectData(self: *Profiler, allocator: std.mem.Allocator) !ProfilingData {
        self.transition("collect_data", .stopped, .done);
        if (self.api == null) return .{ .external = &.{} };

        var args: c.PLUGIN_Profiler_CollectData_Args = .{
            .struct_size = c.PLUGIN_Profiler_CollectData_Args_STRUCT_SIZE,
            .profiler = self.inner,
            .buffer = null,
            .buffer_size_in_bytes = 0,
        };
        try self.check(self.api.?.collect_data.?(&args));
        std.debug.assert(args.buffer_size_in_bytes > 0);
        return if (args.buffer == null) blk: {
            log.debug("Plugin profiler wants us to allocate {d} bytes for profile data", .{args.buffer_size_in_bytes});
            // The plugin want us to allocate memory for it:
            const buffer = try allocator.alloc(u8, args.buffer_size_in_bytes);
            args.buffer = buffer.ptr;
            try self.check(self.api.?.collect_data.?(&args));
            break :blk .{ .owned = buffer };
        } else blk: {
            log.debug("Plugin profiler has {d} bytes of profile data", .{args.buffer_size_in_bytes});
            // Drop sentinel. The profiler plugin returns a null terminated string.
            // But this is creating issues if we save the sentinel on disk,
            // because it will trip up protobuf readers.
            var data = args.buffer[0..args.buffer_size_in_bytes];
            data = if (data.len > 0 and data[data.len - 1] == 0) data[0 .. data.len - 1] else data;
            break :blk .{ .external = data };
        };
    }

    pub fn dumpDataTo(
        self: *Profiler,
        allocator: std.mem.Allocator,
        dir: std.fs.Dir,
        file_name: []const u8,
    ) !void {
        const profile_data = try self.collectData(allocator);
        defer profile_data.free(allocator);

        if (profile_data.items().len == 0) return;

        const file = try dir.createFile(file_name, .{ .truncate = true });
        defer file.close();
        log.info("Writing profiling data to {s} ({} bytes)", .{ file_name, profile_data.items().len });
        return try file.writeAll(profile_data.items());
    }

    pub fn dumpAsJsonTo(
        self: *Profiler,
        allocator: std.mem.Allocator,
        dir: std.fs.Dir,
        file_name: []const u8,
    ) !void {
        const profile_data = try self.collectData(allocator);
        defer profile_data.free(allocator);

        if (profile_data.items().len == 0) {
            log.warn("No profile data was collected: {}", .{self});
            return;
        }

        var converter = try TraceContainer.init(allocator, profile_data.items(), null);
        defer converter.deinit();

        var output_file = try dir.createFile(file_name, .{});
        defer output_file.close();
        var buffered_writer = std.io.bufferedWriter(output_file.writer());
        log.info("Writing profiling data to {s}", .{file_name});
        try converter.toJson(buffered_writer.writer());
        try buffered_writer.flush();
    }

    fn check(self: *Profiler, c_error: ?*Error) !void {
        if (c_error) |err| {
            self.last_error = err;
            return error.PjrtProfilerError;
        }
    }

    pub fn deinit(self: Profiler) void {
        switch (self.status) {
            .started => log.warn("Profiler was never stopped", .{}),
            .stopped => log.warn("Profiler data was never collected", .{}),
            else => {},
        }
        if (self.api == null) return;

        var args: c.PLUGIN_Profiler_Destroy_Args = .{ .profiler = self.inner };
        _ = self.api.?.destroy.?(&args);
    }
};

const ProfilingData = union(enum) {
    owned: []const u8,
    external: []const u8,

    pub fn items(self: ProfilingData) []const u8 {
        return switch (self) {
            inline else => |x| x,
        };
    }

    pub fn free(self: ProfilingData, allocator: std.mem.Allocator) void {
        switch (self) {
            .owned => |data| allocator.free(data),
            .external => {},
        }
    }
};
