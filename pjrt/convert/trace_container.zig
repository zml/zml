const std = @import("std");
const trace_events_proto = @import("//tsl:trace_events_proto");
const xplane_proto = @import("//tsl:xplane_proto");
const xplane_schema = @import("xplane_schema.zig");
const xplane_visitor = @import("xplane_visitor.zig");

// Constants used as trace_viewer PID (device_id in trace_events.proto).
// PID 0 is unused.
// Support up to 500 accelerator devices.
const first_device_id = 1;
const last_device_id = 500;
// Support Upto 200 custom planes as fake devices (i.e., planes with a
// "/custom:" prefix). See `<project_name>::custom_plane_prefix` for more
// information
const first_custom_plane_device_id = last_device_id + 1;
const max_custom_plane_devices_per_host = 200;
const last_custom_plane_device_id = first_custom_plane_device_id + max_custom_plane_devices_per_host - 1;

// Host threads are shown as a single fake device.
pub const host_threads_device_id = last_custom_plane_device_id + 1;

pub const xla_async_op_line_name = "Async XLA Ops";

pub const host_threads_plane_name = "/host:CPU";
pub const gpu_plane_prefix = "/device:GPU:";
pub const tpu_plane_prefix = "/device:TPU:";
pub const custom_plane_prefix = "/device:CUSTOM:";

pub const TraceContainer = struct {
    arena: std.heap.ArenaAllocator,
    events: std.ArrayListUnmanaged(TraceEvent) = .{},
    devices: std.AutoArrayHashMapUnmanaged(u32, Device) = .{},

    pub const Device = struct {
        name: []const u8 = &[_]u8{},
        device_id: u32 = 0,
        resources: std.AutoArrayHashMapUnmanaged(u32, trace_events_proto.Resource) = .{},
    };

    pub const TraceEvent = struct {
        device_id: u32 = 0,
        resource_id: u32 = 0,
        name: []const u8 = &[_]u8{},
        timestamp_ps: u128 = 0,
        duration_ps: u64 = 0,
        args: std.StringArrayHashMapUnmanaged([]const u8) = .{},
    };

    pub fn init(allocator: std.mem.Allocator, pb_buffer: []const u8, max_events: ?usize) !TraceContainer {
        var self: TraceContainer = .{
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
        const arena = self.arena.allocator();

        const xspace = try xplane_proto.XSpace.decode(pb_buffer, arena);
        try self.fromXSpace(arena, &xspace, max_events);
        return self;
    }

    pub fn deinit(self: *TraceContainer) void {
        self.arena.deinit();
    }

    fn picoToMicro(p: anytype) f64 {
        return @as(f64, @floatFromInt(p)) / 1E6;
    }

    fn xLineDisplayId(xline: *const xplane_proto.XLine) i64 {
        return if (xline.display_id != 0) xline.display_id else xline.id;
    }

    fn xLineDisplayName(xline: *const xplane_proto.XLine) []const u8 {
        return switch (xline.display_name) {
            .Empty => xline.name.getSlice(),
            else => xline.display_name.getSlice(),
        };
    }

    fn xstatValueToString(stat: *const xplane_proto.XStat, plane: *const xplane_visitor.XPlaneVisitor, writer: anytype) !void {
        if (stat.value) |val| {
            switch (val) {
                inline .int64_value, .uint64_value, .double_value => |v| try writer.print("{d}", .{v}),
                .str_value => |*v| try writer.writeAll(v.getSlice()),
                .bytes_value => try writer.writeAll("<opaque bytes>"),
                .ref_value => |v| try writer.writeAll(plane.getStatMetadataName(@intCast(v))),
            }
        } else return;
    }

    fn findPlaneWithName(space: *const xplane_proto.XSpace, name: []const u8) ?*xplane_proto.XPlane {
        for (space.planes.items) |*v| {
            if (std.mem.eql(u8, v.name.getSlice(), name)) return v;
        }
        return null;
    }

    fn findPlanesWithPrefix(
        allocator: std.mem.Allocator,
        space: *const xplane_proto.XSpace,
        prefix: []const u8,
    ) ![]*const xplane_proto.XPlane {
        var res = std.ArrayList(*const xplane_proto.XPlane).init(allocator);
        for (space.planes.items) |*p| {
            if (std.mem.startsWith(u8, p.name.getSlice(), prefix)) try res.append(p);
        }
        return res.toOwnedSlice();
    }

    fn buildDeviceAndResources(allocator: std.mem.Allocator, device_id: u32, plane: *const xplane_visitor.XPlaneVisitor, device: *Device) !void {
        device.name = plane.name();
        device.device_id = device_id;
        const sort_by_ordinal = (device_id == host_threads_device_id);
        var ordinal: u32 = 0;
        for (plane.plane.lines.items) |*xline| {
            const resource_id: u32 = @intCast(xLineDisplayId(xline));
            var resource: trace_events_proto.Resource = .{
                .resource_id = resource_id,
                .name = .{ .Const = xLineDisplayName(xline) },
            };

            if (sort_by_ordinal) {
                ordinal += 1;
                resource.sort_index = ordinal;
            }
            try device.resources.put(allocator, resource_id, resource);
        }
    }

    fn xplaneToTraceEvents(self: *TraceContainer, allocator: std.mem.Allocator, device_id: u32, xplane: *const xplane_visitor.XPlaneVisitor) !void {
        // Convert devices and resources.
        const device_entry = try self.devices.getOrPut(allocator, device_id);
        if (!device_entry.found_existing) device_entry.value_ptr.* = .{};

        try buildDeviceAndResources(allocator, device_id, xplane, device_entry.value_ptr);

        // Convert events.
        for (xplane.plane.lines.items) |*xline| {
            const resource_id: u32 = @intCast(xLineDisplayId(xline));

            if (std.mem.eql(u8, xLineDisplayName(xline), xla_async_op_line_name)) continue;
            for (xline.events.items) |*xevent| {
                const event_type = xplane.getEventType(xevent.metadata_id);
                if (event_type.isInternalEvent()) continue;
                var event = try self.createEvent(allocator);
                event.device_id = device_id;
                event.resource_id = resource_id;

                if (xplane.event_metadata_by_id.get(xevent.metadata_id)) |metadata| {
                    if (metadata.display_name != .Empty) {
                        event.name = metadata.display_name.getSlice();
                        try event.args.put(allocator, "long_name", metadata.name.getSlice());
                    } else {
                        event.name = metadata.name.getSlice();
                    }

                    event.timestamp_ps = (@as(u128, @intCast(xline.timestamp_ns)) * 1000) + @as(u128, @intCast(xevent.data.?.offset_ps));
                    event.duration_ps = @intCast(xevent.duration_ps);

                    for (metadata.stats.items) |*xstat| {
                        if (xstat.value == null) continue;
                        var stat_buffer = std.ArrayList(u8).init(allocator);
                        try xstatValueToString(xstat, xplane, stat_buffer.writer().any());
                        const stat_str = try stat_buffer.toOwnedSlice();
                        const stat_type = xplane.getStatType(xstat.metadata_id);
                        if (stat_type.isInternalStat()) continue;
                        if (stat_type == .step_name) event.name = stat_str;
                        try event.args.put(allocator, xplane.getStatMetadataName(xstat.metadata_id), stat_str);
                    }
                }

                for (xevent.stats.items) |*xstat| {
                    if (xstat.value == null) continue;
                    var stat_buffer = std.ArrayList(u8).init(allocator);
                    try xstatValueToString(xstat, xplane, stat_buffer.writer().any());
                    const stat_str = try stat_buffer.toOwnedSlice();
                    const stat_type = xplane.getStatType(xstat.metadata_id);
                    if (stat_type.isInternalStat()) continue;
                    if (stat_type == .step_name) event.name = stat_str;
                    try event.args.put(allocator, xplane.getStatMetadataName(xstat.metadata_id), stat_str);
                }
            }
        }
    }

    fn fromXSpace(self: *TraceContainer, allocator: std.mem.Allocator, xspace: *const xplane_proto.XSpace, max_events: ?usize) !void {
        if (findPlaneWithName(xspace, host_threads_plane_name)) |hp| {
            const xplane = try xplane_visitor.XPlaneVisitor.init(allocator, hp);
            try self.xplaneToTraceEvents(allocator, host_threads_device_id, &xplane);
        }

        var device_planes = try findPlanesWithPrefix(allocator, xspace, gpu_plane_prefix);

        // We don't expect GPU and TPU planes and custom devices to be present in the
        // same XSpace.
        if (device_planes.len == 0) {
            device_planes = try findPlanesWithPrefix(allocator, xspace, tpu_plane_prefix);
        }
        if (device_planes.len == 0) {
            device_planes = try findPlanesWithPrefix(allocator, xspace, custom_plane_prefix);
        }

        for (device_planes) |dp| {
            const xplane = try xplane_visitor.XPlaneVisitor.init(allocator, dp);
            const device_id: u32 = first_device_id + @as(u32, @intCast(xplane.plane.id));
            try self.xplaneToTraceEvents(allocator, device_id, &xplane);
        }

        // Trace viewer (non-streaming) has scalability issues, we need to drop
        // events to avoid loading failure for trace viewer.
        if (max_events) |limit| self.capEvents(limit);
    }

    pub fn createEvent(self: *TraceContainer, allocator: std.mem.Allocator) !*TraceEvent {
        try self.events.append(allocator, .{});
        return &self.events.items[self.events.items.len - 1];
    }

    pub fn capEvents(self: *TraceContainer, max_count: u64) void {
        const total_count = self.events.items.len;
        if (total_count <= max_count) {
            // Nothing to do. Events are not known sorted after return.
            return;
        }
        // sort the events according to start time.
        // TODO: partial sort would improve performance.
        std.mem.sort(TraceEvent, self.events.items, {}, struct {
            pub fn call(_: void, lhs: TraceEvent, rhs: TraceEvent) bool {
                return lhs.timestamp_ps < rhs.timestamp_ps;
            }
        }.call);
        self.events.shrinkRetainingCapacity(max_count);
    }

    pub fn toJson(self: *TraceContainer, writer: anytype) !void {
        try writer.writeAll(
            \\{"displayTimeUnit":"ns","metadata":{"highres-ticks":true},"traceEvents":[
        );

        self.devices.sort(struct {
            keys: []const u32,
            pub fn lessThan(ctx: @This(), lhs: usize, rhs: usize) bool {
                return ctx.keys[lhs] < ctx.keys[rhs];
            }
        }{ .keys = self.devices.keys() });

        for (self.devices.keys(), self.devices.values()) |device_id, *device| {
            if (device.name.len != 0) {
                try writer.print(
                    \\{{"ph":"M","pid":{d},"name":"process_name","args":{{"name":"{s}"}}}},
                , .{ device_id, device.name });
            }
            try writer.print(
                \\{{"ph":"M","pid":{d},"name":"process_sort_index","args":{{"sort_index":{d}}}}},
            , .{
                device_id,
                device_id,
            });

            device.resources.sort(struct {
                keys: []const u32,
                pub fn lessThan(ctx: @This(), lhs: usize, rhs: usize) bool {
                    return ctx.keys[lhs] < ctx.keys[rhs];
                }
            }{ .keys = device.resources.keys() });

            for (device.resources.keys(), device.resources.values()) |resource_id, resource| {
                if (resource.name.getSlice().len != 0) {
                    try writer.print(
                        \\{{"ph":"M","pid":{d},"tid":{d},"name":"thread_name","args":{{"name":"{s}"}}}},
                    , .{
                        device_id,
                        resource_id,
                        resource.name.getSlice(),
                    });
                }
                const sort_index = if (resource.sort_index != 0) resource.sort_index else resource_id;
                try writer.print(
                    \\{{"ph":"M","pid":{d},"tid":{d},"name":"thread_sort_index","args":{{"sort_index":{d}}}}},
                , .{ device_id, resource_id, sort_index });
            }
        }

        for (self.events.items) |*event| {
            const duration_ps = @max(event.duration_ps, 1);
            try writer.print(
                \\{{"ph":"X","pid":{d},"tid":{d},"ts":{d:.17},"dur":{d:.17},"name":"{s}"
            , .{
                event.device_id,
                event.resource_id,
                picoToMicro(event.timestamp_ps),
                picoToMicro(duration_ps),
                event.name,
            });
            if (event.args.count() != 0) {
                try writer.writeAll(
                    \\,"args":{
                );
                event.args.sort(struct {
                    keys: []const []const u8,

                    pub fn lessThan(ctx: @This(), lhs: usize, rhs: usize) bool {
                        return std.mem.order(u8, ctx.keys[lhs], ctx.keys[rhs]).compare(std.math.CompareOperator.lt);
                    }
                }{ .keys = event.args.keys() });

                for (event.args.keys(), event.args.values(), 0..) |key, value, i| {
                    if (i < event.args.count() - 1) {
                        try writer.print(
                            \\"{s}":"{s}",
                        , .{ key, value });
                    } else {
                        // Last item has closing bracket rather than trailing comma.
                        try writer.print(
                            \\"{s}":"{s}"}}
                        , .{ key, value });
                    }
                }
            }
            try writer.writeAll("},");
        }
        try writer.writeAll("{}]}");
    }
};
