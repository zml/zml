const std = @import("std");
const c = @import("c");
const tsl_proto = @import("//tsl:profiler_options_proto");

const log = std.log.scoped(.zml_profiler);

/// Pjrt Profiler extension
pub const Profiler = struct {
    api: ?c.PLUGIN_Profiler_Api,
    inner: *c.PLUGIN_Profiler,
    last_error: ?*Error = null,
    status: Status = .ready,

    pub const Status = enum { ready, started, stopped, done };
    pub const Error = c.PLUGIN_Profiler_Error;
    pub const Options = tsl_proto.ProfileOptions;

    pub fn init(api: ?c.PLUGIN_Profiler_Api, options: Options) Profiler {
        if (api == null) {
            return .{ .api = null, .inner = undefined };
        }

        var buffer: [std.fs.max_path_bytes + @sizeOf(Options) * 4]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&buffer);
        const byte_options = options.encode(fba.allocator()) catch unreachable;
        var res: Profiler = .{ .api = api, .inner = undefined };
        var args: c.PLUGIN_Profiler_Create_Args = .{
            .options = byte_options.ptr,
            .options_size = byte_options.len,
            .profiler = undefined, // out
        };
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
        const buffer: ProfilingData = if (args.buffer == null) blk: {
            std.log.debug("Plugin profiler wants us to allocate {d} bytes for profile data", .{args.buffer_size_in_bytes});
            // The plugin want us to allocate memory for it:
            const buffer = try allocator.alloc(u8, args.buffer_size_in_bytes);
            args.buffer = buffer.ptr;
            try self.check(self.api.?.collect_data.?(&args));
            break :blk .{ .owned = buffer };
        } else blk: {
            std.log.debug("Plugin profiler has {d} bytes of profile data", .{args.buffer_size_in_bytes});
            // Drop sentinel. The profiler plugin returns a null terminated string.
            // But this is creating issues if we save the sentinel on disk,
            // because it will trip up protobuf readers.
            var data = args.buffer[0..args.buffer_size_in_bytes];
            data = if (data.len > 0 and data[data.len - 1] == 0) data[0 .. data.len - 1] else data;
            break :blk .{ .external = data };
        };

        // printDataAsXSpace(allocator, buffer.items());
        return buffer;
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

// If this was working it would be a good alternative to xspace_to_json.cc
// const xspace = @import("xspace.pb.zig");
// pub fn printDataAsXSpace(allocator: std.mem.Allocator, data: []const u8) void {
//     var arena = std.heap.ArenaAllocator.init(allocator);
//     defer arena.deinit();
//
//     const space = xspace.XSpace.decode(data, arena.allocator()) catch |e| {
//         std.log.err("Couldn't load profiling data: {}", .{e});
//         return;
//     };
//
//     for (space.errors.items) |err| {
//         std.log.err("{s}", .{err.getSlice()});
//     }
//     for (space.warnings.items) |warning| {
//         std.log.warn("{s}", .{warning.getSlice()});
//     }
//     for (space.hostnames.items) |host| {
//         std.log.info("Profiled host {s}", .{host.getSlice()});
//     }
//     for (space.planes.items) |plane| {
//         var event_metadata = std.hash_map.AutoHashMap(i64, xspace.XEventMetadata).init(arena.allocator());
//         event_metadata.ensureTotalCapacity(@intCast(plane.event_metadata.items.len)) catch return;
//         defer event_metadata.deinit();
//         for (plane.event_metadata.items) |event_meta_entry| {
//             if (event_meta_entry.value) |event_meta| {
//                 event_metadata.putAssumeCapacity(event_meta.id, event_meta);
//             }
//         }
//         std.log.info("Profiled device {s}", .{plane.name.getSlice()});

//         for (plane.lines.items) |line| {
//             std.log.info(
//                 "{d} -> {d} xline {s} ({d} events)",
//                 .{ line.timestamp_ns, line.duration_ps, line.name.getSlice(), line.events.items.len },
//             );
//             const ps_per_ns: i64 = 1000;
//             var duration_ns: i64 = 0;
//             var last_metadata_id: i64 = 0;
//             for (line.events.items) |event| {
//                 if (event.metadata_id != last_metadata_id and duration_ns != 0) {
//                     const duration_us = @as(f32, @floatFromInt(duration_ns)) / std.time.ns_per_us;
//                     const meta = event_metadata.get(event.metadata_id).?;
//                     std.log.info("event {s}: {d:.1}μs", .{ meta.name.getSlice(), duration_us });

//                     last_metadata_id = event.metadata_id;
//                     duration_ns = 0;
//                 }
//                 duration_ns += @divFloor(event.duration_ps, ps_per_ns);

//                 const duration_us = @as(f32, @floatFromInt(duration_ns)) / std.time.ns_per_us;
//                 const meta = event_metadata.get(event.metadata_id).?;
//                 std.log.info("event {s}: {d:.1}μs", .{ meta.name.getSlice(), duration_us });
//             }
//         }
//     }
// }

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
