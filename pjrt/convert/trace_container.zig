const std = @import("std");
const xplane_proto = @import("//tsl:xplane_proto");
const xplane_schema = @import("utils/xplane_schema.zig");
const xplane_utils = @import("utils/xplane_utils.zig");
const xplane_visitor = @import("utils/xplane_visitor.zig");

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
    metadata: Trace = .{},
    events: std.ArrayListUnmanaged(*TraceEvent) = .{},

    pub const Trace = struct {
        devices: std.AutoHashMapUnmanaged(u32, Device) = .{},
        trace_events: std.ArrayListUnmanaged(TraceEvent) = .{},
    };

    pub const TraceEvent = struct {
        device_id: u32 = 0,
        resource_id: u32 = 0,
        name: []const u8 = &[_]u8{},
        timestamp_ps: u128 = 0,
        duration_ps: u64 = 0,
        args: std.StringHashMapUnmanaged([]const u8) = .{},
    };

    pub const Device = struct {
        name: []const u8 = &[_]u8{},
        device_id: u32 = 0,
        resources: std.AutoHashMapUnmanaged(u32, Resource) = .{},

        pub fn mutableResource(self: *Device, allocator: std.mem.Allocator, resource_id: u32) !*Resource {
            const entry = try self.resources.getOrPut(allocator, resource_id);
            if (!entry.found_existing) {
                entry.value_ptr.* = .{};
            }
            return entry.value_ptr;
        }
    };

    pub const Resource = struct {
        name: []const u8 = &[_]u8{},
        resource_id: u32 = 0,
        sort_index: u32 = 0,
    };

    const BuilderCtx = struct {
        ordinal: u32 = 0,
        sort_by_ordinal: bool,
        device: *Device,
    };

    fn buildDeviceAndResources(allocator: std.mem.Allocator, device_id: u32, plane: *const xplane_visitor.XPlaneVisitor, device: *Device) !void {
        device.name = plane.name();
        device.device_id = device_id;
        const sort_by_ordinal = (device_id == host_threads_device_id);
        var ordinal: u32 = 0;
        for (plane.plane.lines.items) |*l| {
            var line = xplane_visitor.XLineVisitor.init(plane, l);
            const resource_id: u32 = @intCast(line.displayId());
            const resource = try device.mutableResource(allocator, resource_id);
            resource.resource_id = resource_id;
            resource.name = line.displayName();
            if (sort_by_ordinal) {
                ordinal += 1;
                resource.sort_index = ordinal;
            }
        }
    }

    fn xplaneToTraceEvents(self: *TraceContainer, allocator: std.mem.Allocator, device_id: u32, xplane: *const xplane_visitor.XPlaneVisitor) !void {
        // Convert devices and resources.
        try buildDeviceAndResources(allocator, device_id, xplane, try self.mutableDevice(allocator, device_id));

        // Convert events.
        for (xplane.plane.lines.items) |*l| {
            const xline = xplane_visitor.XLineVisitor.init(xplane, l);
            const resource_id: u32 = @intCast(xline.displayId());
            if (std.mem.eql(u8, xline.displayName(), xla_async_op_line_name)) return;
            for (xline.line.events.items) |*e| {
                const xevent = xplane_visitor.XEventVisitor.init(xline.plane, xline.line, e);
                const event_type: xplane_schema.HostEventType = xevent
                    .type_ orelse .UnknownHostEventType;
                if (event_type.isInternalEvent()) return;
                var event = try self.createEvent(allocator);
                event.device_id = device_id;
                event.resource_id = resource_id;
                if (xevent.hasDisplayName()) {
                    event.name = xevent.displayName();
                    try event.args.put(allocator, "long_name", xevent.name());
                } else {
                    event.name = xevent.name();
                }
                event.timestamp_ps = @intCast(xevent.timestampPs());
                event.duration_ps = @intCast(xevent.durationPs());

                const xevent_md_visitor = xevent.metadataVisitor();
                for (xevent_md_visitor.stats_owner.stats.items) |*s| {
                    const xstat = xplane_visitor.XStatVisitor.init(xevent_md_visitor.plane, s);
                    if (xstat.stat.value == null) return;
                    if (xstat.type()) |t| {
                        if (t.isInternalStat()) return;
                        if (t == .step_name) {
                            event.name = try xstat.toString(allocator);
                        }
                    }
                    try event.args.put(allocator, xstat.name(), try xstat.toString(allocator));
                }

                for (xevent.event.stats.items) |*s| {
                    const xstat = xplane_visitor.XStatVisitor.init(xevent.plane, s);
                    if (xstat.stat.value == null) return;
                    if (xstat.type()) |t| {
                        if (t.isInternalStat()) return;
                        if (t == .step_name) {
                            event.name = try xstat.toString(allocator);
                        }
                    }
                    try event.args.put(allocator, xstat.name(), try xstat.toString(allocator));
                }
            }
        }
    }

    fn getTraceViewerMaxEvents() !u64 {
        const kMaxEvents = 1000000;
        if (std.posix.getenv("TF_PROFILER_TRACE_VIEWER_MAX_EVENTS")) |max_events| {
            return std.fmt.parseInt(u64, max_events, 10);
        } else return kMaxEvents;
    }

    pub fn fromXSpace(allocator: std.mem.Allocator, xspace: *const xplane_proto.XSpace) !TraceContainer {
        var self: TraceContainer = .{};
        if (xplane_utils.findPlaneWithName(xspace, host_threads_plane_name)) |hp| {
            const xplane = try xplane_visitor.XPlaneVisitor.init(
                allocator,
                hp,
                &.{ xplane_schema.HostEventType.fromString, xplane_schema.HostEventType.fromTfOpEventType },
                &.{xplane_schema.StatType.fromString},
            );
            try self.xplaneToTraceEvents(allocator, host_threads_device_id, &xplane);
        }

        var device_planes = try xplane_utils.findPlanesWithPrefix(allocator, xspace, gpu_plane_prefix);

        // We don't expect GPU and TPU planes and custom devices to be present in the
        // same XSpace.
        if (device_planes.len == 0) {
            device_planes = try xplane_utils.findPlanesWithPrefix(allocator, xspace, tpu_plane_prefix);
        }
        if (device_planes.len == 0) {
            device_planes = try xplane_utils.findPlanesWithPrefix(allocator, xspace, custom_plane_prefix);
        }

        for (device_planes) |dp| {
            const xplane = try xplane_visitor.XPlaneVisitor.init(
                allocator,
                dp,
                &.{ xplane_schema.HostEventType.fromString, xplane_schema.HostEventType.fromTfOpEventType },
                &.{xplane_schema.StatType.fromString},
            );
            const device_id: u32 = first_device_id + @as(u32, @intCast(xplane.plane.id));
            try self.xplaneToTraceEvents(allocator, device_id, &xplane);
        }

        // Trace viewer (non-streaming) has scalability issues, we need to drop
        // events to avoid loading failure for trace viewer.
        const viewer_max_events = try getTraceViewerMaxEvents();
        self.capEvents(viewer_max_events);
        return self;
    }

    pub fn mutableDevice(self: *TraceContainer, allocator: std.mem.Allocator, device_id: u32) !*Device {
        const entry = try self.metadata.devices.getOrPut(allocator, device_id);
        if (!entry.found_existing) {
            entry.value_ptr.* = .{};
        }
        return entry.value_ptr;
    }

    pub fn createEvent(self: *TraceContainer, allocator: std.mem.Allocator) !*TraceEvent {
        const event = try allocator.create(TraceEvent);
        event.* = .{};
        try self.events.append(allocator, event);
        return event;
    }

    pub fn capEvents(self: *TraceContainer, max_count: u64) void {
        const total_count = self.events.items.len;
        if (total_count <= max_count) {
            // Nothing to do. Events are not known sorted after return.
            return;
        }
        // sort the events according to start time.
        // TODO: partial sort would improve performance.
        std.mem.sort(*TraceEvent, self.events.items, {}, struct {
            pub fn call(_: void, lhs: *TraceEvent, rhs: *TraceEvent) bool {
                return lhs.timestamp_ps < rhs.timestamp_ps;
            }
        }.call);
        self.events.shrinkRetainingCapacity(max_count);
    }
};
