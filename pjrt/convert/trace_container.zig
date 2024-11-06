const std = @import("std");
const trace_utils = @import("utils/trace_utils.zig");
const xplane_proto = @import("//tsl:xplane_proto");
const xplane_schema = @import("utils/xplane_schema.zig");
const xplane_utils = @import("utils/xplane_utils.zig");
const xplane_visitor = @import("utils/xplane_visitor.zig");

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
        timestamp_ps: u64 = 0,
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
        var context: BuilderCtx = .{
            .sort_by_ordinal = (device_id == trace_utils.kHostThreadsDeviceId),
            .device = device,
        };
        try plane.forEachLine(allocator, struct {
            pub fn call(a: std.mem.Allocator, line: xplane_visitor.XLineVisitor, ctx: ?*anyopaque) !void {
                const ctx_: *BuilderCtx = @ptrCast(@alignCast(ctx));
                const resource_id: u32 = @intCast(line.displayId());
                const resource = try ctx_.device.mutableResource(a, resource_id);
                resource.resource_id = resource_id;
                resource.name = line.displayName();
                if (ctx_.sort_by_ordinal) {
                    // When sort_index is absent (i.e. 0), resource id will be used.
                    // Therefore sort_index starts with 1.
                    ctx_.ordinal += 1;
                    resource.sort_index = ctx_.ordinal;
                }
            }
        }.call, &context);
    }

    fn xplaneToTraceEvents(self: *TraceContainer, allocator: std.mem.Allocator, device_id: u32, xplane: *const xplane_visitor.XPlaneVisitor) !void {
        // Convert devices and resources.
        try buildDeviceAndResources(allocator, device_id, xplane, try self.mutableDevice(allocator, device_id));

        // Convert events.

        const PerLineCtx = struct {
            device_id: u32,
            container: *TraceContainer,
        };
        var line_ctx: PerLineCtx = .{ .device_id = device_id, .container = self };
        try xplane.forEachLine(allocator, struct {
            pub fn call(a1: std.mem.Allocator, xline: xplane_visitor.XLineVisitor, l_ctx: ?*anyopaque) !void {
                const parsed_line_ctx: *PerLineCtx = @ptrCast(@alignCast(l_ctx));
                const resource_id: u32 = @intCast(xline.displayId());
                if (std.mem.eql(u8, xline.displayName(), xplane_schema.kXlaAsyncOpLineName)) return;
                const PerEachEventCtx = struct {
                    device_id: u32,
                    resource_id: u32,
                    container: *TraceContainer,
                };

                var evt_ctx: PerEachEventCtx = .{
                    .device_id = parsed_line_ctx.device_id,
                    .resource_id = resource_id,
                    .container = parsed_line_ctx.container,
                };
                try xline.forEachEvent(a1, struct {
                    pub fn call(a2: std.mem.Allocator, xevent: xplane_visitor.XEventVisitor, e_ctx: ?*anyopaque) !void {
                        const parsed_event_ctx: *PerEachEventCtx = @ptrCast(@alignCast(e_ctx));
                        const event_type: xplane_schema.HostEventType = xevent
                            .type_ orelse .kUnknownHostEventType;
                        if (event_type.isInternalEvent()) return;
                        var event = try parsed_event_ctx.container.createEvent(a2);
                        event.device_id = parsed_event_ctx.device_id;
                        event.resource_id = parsed_event_ctx.resource_id;
                        if (xevent.hasDisplayName()) {
                            event.name = xevent.displayName();
                            try event.args.put(a2, "long_name", xevent.name());
                        } else {
                            event.name = xevent.name();
                        }
                        event.timestamp_ps = @intCast(xevent.timestampPs());
                        event.duration_ps = @intCast(xevent.durationPs());

                        const PerStatCtx = struct {
                            event: *TraceEvent,
                        };

                        var stat_ctx: PerStatCtx = .{ .event = event };
                        const for_each_stat = struct {
                            pub fn call(a3: std.mem.Allocator, xstat: xplane_visitor.XStatVisitor, s_ctx: ?*anyopaque) !void {
                                const parsed_stat_ctx: *PerStatCtx = @ptrCast(@alignCast(s_ctx));
                                if (xstat.stat.value == null) return;
                                if (xstat.type()) |t| {
                                    if (t.isInternalStat()) return;
                                    if (t == .kStepName) {
                                        parsed_stat_ctx.event.name = try xstat.toString(a3);
                                    }
                                }
                                try parsed_stat_ctx.event.args.put(a3, xstat.name(), try xstat.toString(a3));
                            }
                        }.call;
                        try xevent.metadataVisitor().forEachStat(a2, for_each_stat, &stat_ctx);
                        try xevent.forEachStat(a2, for_each_stat, &stat_ctx);
                    }
                }.call, &evt_ctx);
            }
        }.call, &line_ctx);
    }

    fn getTraceViewerMaxEvents() !u64 {
        const kMaxEvents = 1000000;
        if (std.posix.getenv("TF_PROFILER_TRACE_VIEWER_MAX_EVENTS")) |max_events| {
            return std.fmt.parseInt(u64, max_events, 10);
        } else return kMaxEvents;
    }

    pub fn fromXSpace(allocator: std.mem.Allocator, xspace: *const xplane_proto.XSpace) !TraceContainer {
        var self: TraceContainer = .{};
        if (xplane_utils.findPlaneWithName(xspace, xplane_schema.kHostThreadsPlaneName)) |hp| {
            const xplane = try xplane_visitor.XPlaneVisitor.init(
                allocator,
                hp,
                &.{ xplane_schema.HostEventType.fromString, xplane_schema.HostEventType.fromTfOpEventType },
                &.{xplane_schema.StatType.fromString},
            );
            try self.xplaneToTraceEvents(allocator, trace_utils.kHostThreadsDeviceId, &xplane);
        }

        var device_planes = try xplane_utils.findPlanesWithPrefix(allocator, xspace, xplane_schema.kGpuPlanePrefix);

        // We don't expect GPU and TPU planes and custom devices to be present in the
        // same XSpace.
        if (device_planes.len == 0) {
            device_planes = try xplane_utils.findPlanesWithPrefix(allocator, xspace, xplane_schema.kTpuPlanePrefix);
        }
        if (device_planes.len == 0) {
            device_planes = try xplane_utils.findPlanesWithPrefix(allocator, xspace, xplane_schema.kCustomPlanePrefix);
        }

        for (device_planes) |dp| {
            const xplane = try xplane_visitor.XPlaneVisitor.init(
                allocator,
                dp,
                &.{ xplane_schema.HostEventType.fromString, xplane_schema.HostEventType.fromTfOpEventType },
                &.{xplane_schema.StatType.fromString},
            );
            const device_id: u32 = trace_utils.kFirstDeviceId + @as(u32, @intCast(xplane.plane.id));
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
