const std = @import("std");

const c = @import("c");
const upb = @import("upb");

pub const Args = struct {
    positional: struct {
        input: []const u8,
        output: ?[]const u8 = null,
    },

    pub const help =
        \\Usage: xspace_to_perfetto <input_xspace.pb> [output_trace.json]
        \\
        \\Converts an XSpace protobuf profile into trace-viewer JSON.
        \\
    ;
};

// XPlane schema source:
// ../../xla/third_party/tsl/tsl/profiler/protobuf/xplane.proto
const kHostThreadsPlaneName = "/host:CPU";
const kGpuPlanePrefix = "/device:GPU:";
const kTpuPlanePrefix = "/device:TPU:";
const kCustomPlanePrefix = "/device:CUSTOM:";
const kXlaAsyncOpLineName = "Async XLA Ops";
const kOpaqueBytesText = "<opaque bytes>";

const kPicosecondsPerNanosecond: u64 = 1_000;
const kPicosecondsPerMicrosecond: f64 = 1_000_000.0;
const kOutputBufferSizeBytes = 1024 * 1024;
const kHexDigits = "0123456789abcdef";

const kFirstDeviceId: u32 = 1;
const kLastDeviceId: u32 = 500;
const kFirstCustomPlaneDeviceId: u32 = kLastDeviceId + 1;
const kMaxCustomPlaneDevicesPerHost: u32 = 200;
const kLastCustomPlaneDeviceId: u32 = kFirstCustomPlaneDeviceId + kMaxCustomPlaneDevicesPerHost - 1;
const kHostThreadsDeviceId: u32 = kLastCustomPlaneDeviceId + 1;

const StringValue = union(enum) {
    text: []const u8,
    int64: i64,
    uint64: u64,
    double: f64,
    opaque_bytes,
};

const Arg = struct {
    name: []const u8,
    value: StringValue,
};

const StatBehavior = enum {
    emit,
    step_name,
    skip,
};

const JsonArrayWriter = struct {
    needs_comma: bool = false,

    fn nextItem(self: *JsonArrayWriter, writer: *std.Io.Writer) !void {
        if (self.needs_comma) {
            try writer.writeByte(',');
        }
        self.needs_comma = true;
    }
};

const EventScratch = struct {
    args: std.ArrayListUnmanaged(Arg) = .empty,

    fn reset(self: *EventScratch) void {
        self.args.clearRetainingCapacity();
    }

    fn deinit(self: *EventScratch, allocator: std.mem.Allocator) void {
        self.args.deinit(allocator);
    }
};

const PlaneContext = struct {
    plane: *const c.tensorflow_profiler_XPlane,
    device_id: u32,
    is_host: bool,
    event_metadata_by_id: std.AutoHashMapUnmanaged(i64, *const c.tensorflow_profiler_XEventMetadata) = .{},
    stat_metadata_by_id: std.AutoHashMapUnmanaged(i64, *const c.tensorflow_profiler_XStatMetadata) = .{},
    event_is_internal_by_id: std.AutoHashMapUnmanaged(i64, bool) = .{},
    stat_behavior_by_id: std.AutoHashMapUnmanaged(i64, StatBehavior) = .{},

    // Metadata caching mirrors the role of XPlaneVisitor lookups in:
    // ../../xla/xla/tsl/profiler/utils/xplane_visitor.h
    // ../../xla/xla/tsl/profiler/utils/xplane_visitor.cc
    fn init(plane: *const c.tensorflow_profiler_XPlane, device_id: u32, is_host: bool) PlaneContext {
        return .{
            .plane = plane,
            .device_id = device_id,
            .is_host = is_host,
        };
    }

    fn deinit(self: *PlaneContext, allocator: std.mem.Allocator) void {
        self.event_metadata_by_id.deinit(allocator);
        self.stat_metadata_by_id.deinit(allocator);
        self.event_is_internal_by_id.deinit(allocator);
        self.stat_behavior_by_id.deinit(allocator);
    }

    fn getEventMetadata(
        self: *PlaneContext,
        allocator: std.mem.Allocator,
        metadata_id: i64,
    ) !?*const c.tensorflow_profiler_XEventMetadata {
        if (self.event_metadata_by_id.get(metadata_id)) |metadata| {
            return metadata;
        }

        var metadata: ?*c.tensorflow_profiler_XEventMetadata = null;
        if (!c.tensorflow_profiler_XPlane_event_metadata_get(self.plane, metadata_id, &metadata)) {
            return null;
        }
        const resolved = metadata orelse return null;
        const result: *const c.tensorflow_profiler_XEventMetadata = resolved;
        try self.event_metadata_by_id.put(allocator, metadata_id, result);
        return result;
    }

    fn getStatMetadata(
        self: *PlaneContext,
        allocator: std.mem.Allocator,
        metadata_id: i64,
    ) !?*const c.tensorflow_profiler_XStatMetadata {
        if (self.stat_metadata_by_id.get(metadata_id)) |metadata| {
            return metadata;
        }

        var metadata: ?*c.tensorflow_profiler_XStatMetadata = null;
        if (!c.tensorflow_profiler_XPlane_stat_metadata_get(self.plane, metadata_id, &metadata)) {
            return null;
        }
        const resolved = metadata orelse return null;
        const result: *const c.tensorflow_profiler_XStatMetadata = resolved;
        try self.stat_metadata_by_id.put(allocator, metadata_id, result);
        return result;
    }

    fn isInternalEvent(
        self: *PlaneContext,
        allocator: std.mem.Allocator,
        metadata_id: i64,
    ) !bool {
        if (self.event_is_internal_by_id.get(metadata_id)) |is_internal| {
            return is_internal;
        }

        const is_internal = if (try self.getEventMetadata(allocator, metadata_id)) |metadata|
            isInternalEventName(stringView(c.tensorflow_profiler_XEventMetadata_name(metadata)))
        else
            false;
        try self.event_is_internal_by_id.put(allocator, metadata_id, is_internal);
        return is_internal;
    }

    fn statBehavior(
        self: *PlaneContext,
        allocator: std.mem.Allocator,
        metadata_id: i64,
    ) !StatBehavior {
        if (self.stat_behavior_by_id.get(metadata_id)) |behavior| {
            return behavior;
        }

        const behavior = if (try self.getStatMetadata(allocator, metadata_id)) |metadata|
            classifyStatName(stringView(c.tensorflow_profiler_XStatMetadata_name(metadata)))
        else
            .emit;
        try self.stat_behavior_by_id.put(allocator, metadata_id, behavior);
        return behavior;
    }
};

pub fn dumpXSpaceProtoToTraceJsonFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    xspace_proto: []const u8,
    output_path: []const u8,
) !void {
    var upb_backing_arena = std.heap.ArenaAllocator.init(allocator);
    defer upb_backing_arena.deinit();

    var upb_allocator: upb.Allocator = .init(upb_backing_arena.allocator());
    const upb_arena = c.upb_Arena_Init(null, 0, upb_allocator.inner());
    defer c.upb_Arena_Free(upb_arena);

    const xspace = c.tensorflow_profiler_XSpace_parse(xspace_proto.ptr, xspace_proto.len, upb_arena) orelse
        return error.MalformedXSpace;
    try writeTraceJsonFile(allocator, io, xspace, output_path);
}

fn writeTraceJsonFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    xspace: *const c.tensorflow_profiler_XSpace,
    output_path: []const u8,
) !void {
    var output_file = try std.Io.Dir.createFile(.cwd(), io, output_path, .{});
    defer output_file.close(io);

    const output_buffer = try allocator.alloc(u8, kOutputBufferSizeBytes);
    defer allocator.free(output_buffer);

    var file_writer = output_file.writer(io, output_buffer);
    defer file_writer.interface.flush() catch {};

    try streamXSpaceToTraceJson(allocator, xspace, &file_writer.interface);
    try file_writer.interface.flush();
}

fn streamXSpaceToTraceJson(
    allocator: std.mem.Allocator,
    xspace: *const c.tensorflow_profiler_XSpace,
    writer: *std.Io.Writer,
) !void {
    // Keep the same trace JSON envelope as the streaming C++ implementation in:
    // ../../xla/xla/tsl/profiler/convert/trace_events_to_json.cc
    var planes = try collectSelectedPlanes(allocator, xspace);
    defer {
        for (planes.items) |*plane| {
            plane.deinit(allocator);
        }
        planes.deinit(allocator);
    }

    try writer.writeAll("{\"displayTimeUnit\":\"ns\",\"metadata\":{\"highres-ticks\":true},\"traceEvents\":[");
    var array_writer: JsonArrayWriter = .{};

    for (planes.items) |*plane| {
        try streamPlaneMetadata(writer, &array_writer, plane);
    }
    for (planes.items) |*plane| {
        try streamPlaneEvents(allocator, writer, &array_writer, plane);
    }

    try streamFlows(allocator, writer, &array_writer, planes.items);

    try writer.writeAll("]}");
}

fn collectSelectedPlanes(
    allocator: std.mem.Allocator,
    xspace: *const c.tensorflow_profiler_XSpace,
) !std.ArrayListUnmanaged(PlaneContext) {
    // Plane selection intentionally follows:
    // ../../xla/xla/tsl/profiler/convert/xplane_to_trace_events.cc
    var result: std.ArrayListUnmanaged(PlaneContext) = .empty;
    errdefer {
        for (result.items) |*plane| {
            plane.deinit(allocator);
        }
        result.deinit(allocator);
    }

    var plane_count: usize = 0;
    const planes = c.tensorflow_profiler_XSpace_planes(xspace, &plane_count) orelse return result;
    const plane_slice = planes[0..plane_count];

    var host_plane: ?*const c.tensorflow_profiler_XPlane = null;
    for (plane_slice) |plane_c| {
        const plane: *const c.tensorflow_profiler_XPlane = plane_c orelse continue;
        if (std.mem.eql(u8, stringView(c.tensorflow_profiler_XPlane_name(plane)), kHostThreadsPlaneName)) {
            host_plane = plane;
            break;
        }
    }

    if (host_plane) |plane| {
        try result.append(allocator, PlaneContext.init(plane, kHostThreadsDeviceId, true));
    }

    if (try appendPlanesWithPrefix(allocator, &result, plane_slice, kGpuPlanePrefix)) {
        return result;
    }
    if (try appendPlanesWithPrefix(allocator, &result, plane_slice, kTpuPlanePrefix)) {
        return result;
    }
    _ = try appendPlanesWithPrefix(allocator, &result, plane_slice, kCustomPlanePrefix);
    return result;
}

fn appendPlanesWithPrefix(
    allocator: std.mem.Allocator,
    result: *std.ArrayListUnmanaged(PlaneContext),
    planes: []const [*c]const c.tensorflow_profiler_XPlane,
    prefix: []const u8,
) !bool {
    var found = false;
    for (planes) |plane_c| {
        const plane: *const c.tensorflow_profiler_XPlane = plane_c orelse continue;
        const name = stringView(c.tensorflow_profiler_XPlane_name(plane));
        if (!std.mem.startsWith(u8, name, prefix)) continue;
        found = true;
        const device_ordinal = wrapI64ToU32(c.tensorflow_profiler_XPlane_id(plane));
        try result.append(allocator, PlaneContext.init(plane, kFirstDeviceId + device_ordinal, false));
    }
    return found;
}

fn streamPlaneMetadata(
    writer: *std.Io.Writer,
    array_writer: *JsonArrayWriter,
    plane: *const PlaneContext,
) !void {
    // Metadata event shape matches the resource/process metadata emitted in:
    // ../../xla/xla/tsl/profiler/convert/xplane_to_trace_events.cc
    const plane_name = stringView(c.tensorflow_profiler_XPlane_name(plane.plane));
    if (plane_name.len != 0) {
        try array_writer.nextItem(writer);
        try writer.writeAll("{\"ph\":\"M\",\"pid\":");
        try writer.print("{d}", .{plane.device_id});
        try writer.writeAll(",\"name\":\"process_name\",\"args\":{\"name\":");
        try writeJsonString(writer, plane_name);
        try writer.writeAll("}}");
    }

    try array_writer.nextItem(writer);
    try writer.writeAll("{\"ph\":\"M\",\"pid\":");
    try writer.print("{d}", .{plane.device_id});
    try writer.writeAll(",\"name\":\"process_sort_index\",\"args\":{\"sort_index\":");
    try writer.print("{d}", .{plane.device_id});
    try writer.writeAll("}}");

    var ordinal: u32 = 0;
    var line_count: usize = 0;
    const lines = c.tensorflow_profiler_XPlane_lines(plane.plane, &line_count) orelse return;
    for (lines[0..line_count]) |line_c| {
        const line: *const c.tensorflow_profiler_XLine = line_c orelse continue;
        const resource_id = try lineResourceId(line);
        const resource_name = lineDisplayName(line);
        const sort_index = if (plane.is_host) blk: {
            ordinal += 1;
            break :blk ordinal;
        } else resource_id;
        try streamResourceMetadata(writer, array_writer, plane.device_id, resource_id, resource_name, sort_index);
    }
}

fn streamResourceMetadata(
    writer: *std.Io.Writer,
    array_writer: *JsonArrayWriter,
    device_id: u32,
    resource_id: u32,
    resource_name: []const u8,
    sort_index: u32,
) !void {
    if (resource_name.len != 0) {
        try array_writer.nextItem(writer);
        try writer.writeAll("{\"ph\":\"M\",\"pid\":");
        try writer.print("{d}", .{device_id});
        try writer.writeAll(",\"tid\":");
        try writer.print("{d}", .{resource_id});
        try writer.writeAll(",\"name\":\"thread_name\",\"args\":{\"name\":");
        try writeJsonString(writer, resource_name);
        try writer.writeAll("}}");
    }

    try array_writer.nextItem(writer);
    try writer.writeAll("{\"ph\":\"M\",\"pid\":");
    try writer.print("{d}", .{device_id});
    try writer.writeAll(",\"tid\":");
    try writer.print("{d}", .{resource_id});
    try writer.writeAll(",\"name\":\"thread_sort_index\",\"args\":{\"sort_index\":");
    try writer.print("{d}", .{sort_index});
    try writer.writeAll("}}");
}

fn streamPlaneEvents(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    array_writer: *JsonArrayWriter,
    plane: *PlaneContext,
) !void {
    var scratch: EventScratch = .{};
    defer scratch.deinit(allocator);

    var line_count: usize = 0;
    const lines = c.tensorflow_profiler_XPlane_lines(plane.plane, &line_count) orelse return;
    for (lines[0..line_count]) |line_c| {
        const line: *const c.tensorflow_profiler_XLine = line_c orelse continue;
        if (std.mem.eql(u8, lineDisplayName(line), kXlaAsyncOpLineName)) continue;

        const resource_id = try lineResourceId(line);
        var event_count: usize = 0;
        const events = c.tensorflow_profiler_XLine_events(line, &event_count) orelse continue;
        for (events[0..event_count]) |event_c| {
            const event: *const c.tensorflow_profiler_XEvent = event_c orelse continue;
            try streamEvent(allocator, writer, array_writer, plane, line, event, resource_id, &scratch);
        }
    }
}

fn streamEvent(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    array_writer: *JsonArrayWriter,
    plane: *PlaneContext,
    line: *const c.tensorflow_profiler_XLine,
    event: *const c.tensorflow_profiler_XEvent,
    resource_id: u32,
    scratch: *EventScratch,
) !void {
    const metadata_id = c.tensorflow_profiler_XEvent_metadata_id(event);
    if (try plane.isInternalEvent(allocator, metadata_id)) return;

    // Display name, long_name, stat merge order, and step_name override mirror:
    // ../../xla/xla/tsl/profiler/convert/xplane_to_trace_events.cc
    const metadata = try plane.getEventMetadata(allocator, metadata_id);
    const metadata_name = if (metadata) |value| stringView(c.tensorflow_profiler_XEventMetadata_name(value)) else "";
    const metadata_display_name = if (metadata) |value| stringView(c.tensorflow_profiler_XEventMetadata_display_name(value)) else "";

    var event_name: StringValue = .{
        .text = if (metadata_display_name.len != 0) metadata_display_name else metadata_name,
    };

    scratch.reset();
    const args = &scratch.args;

    if (metadata_display_name.len != 0) {
        try putArg(args, allocator, "long_name", .{ .text = metadata_name });
    }

    if (metadata) |value| {
        try appendStats(allocator, plane, args, &event_name, metadataStats(value));
    }
    try appendStats(allocator, plane, args, &event_name, eventStats(event));

    std.mem.sort(Arg, args.items, {}, argLessThan);

    try array_writer.nextItem(writer);
    try writer.writeAll("{\"ph\":\"X\",\"pid\":");
    try writer.print("{d}", .{plane.device_id});
    try writer.writeAll(",\"tid\":");
    try writer.print("{d}", .{resource_id});
    try writer.writeAll(",\"ts\":");
    var number_buf: [64]u8 = undefined;
    try writer.writeAll(formatPicosToMicros(&number_buf, eventTimestampPs(line, event)));
    try writer.writeAll(",\"dur\":");
    try writer.writeAll(formatPicosToMicros(&number_buf, eventDurationPs(event)));
    try writer.writeAll(",\"name\":");
    try writeJsonStringValue(writer, event_name);

    if (args.items.len != 0) {
        try writer.writeAll(",\"args\":{");
        for (args.items, 0..) |arg, index| {
            if (index != 0) try writer.writeByte(',');
            try writeJsonString(writer, arg.name);
            try writer.writeByte(':');
            try writeJsonStringValue(writer, arg.value);
        }
        try writer.writeByte('}');
    }

    try writer.writeByte('}');
}

const FlowKey = struct {
    context_type: u64,
    context_id: u64,
};

const FlowEndpoint = struct {
    key: FlowKey,
    zml_origin: bool,
    role: enum { producer, consumer },
    timestamp_ps: u64,
    device_id: u32,
    resource_id: u32,
    event: *const c.tensorflow_profiler_XEvent,

    fn lessThan(_: void, a: FlowEndpoint, b: FlowEndpoint) bool {
        if (a.key.context_type != b.key.context_type) return a.key.context_type < b.key.context_type;
        if (a.key.context_id != b.key.context_id) return a.key.context_id < b.key.context_id;
        if (a.timestamp_ps != b.timestamp_ps) return a.timestamp_ps < b.timestamp_ps;
        if (a.role != b.role) return a.role == .producer;
        if (a.device_id != b.device_id) return a.device_id < b.device_id;
        return a.resource_id < b.resource_id;
    }
};

const FlowContext = struct {
    zml_origin: bool = false,
    producer_type: ?u64 = null,
    producer_id: ?u64 = null,
    consumer_type: ?u64 = null,
    consumer_id: ?u64 = null,

    fn readStats(self: *FlowContext, allocator: std.mem.Allocator, plane: *PlaneContext, stats: []const [*c]const c.tensorflow_profiler_XStat) !void {
        for (stats) |stat_c| {
            const stat = stat_c orelse continue;
            const metadata = (try plane.getStatMetadata(allocator, c.tensorflow_profiler_XStat_metadata_id(stat))) orelse continue;
            const name = stringView(c.tensorflow_profiler_XStatMetadata_name(metadata));
            if (std.mem.eql(u8, name, "_zml_flow")) {
                self.zml_origin = switch (c.tensorflow_profiler_XStat_value_case(stat)) {
                    c.tensorflow_profiler_XStat_value_uint64_value => c.tensorflow_profiler_XStat_uint64_value(stat) != 0,
                    c.tensorflow_profiler_XStat_value_int64_value => c.tensorflow_profiler_XStat_int64_value(stat) != 0,
                    else => false,
                };
                continue;
            }
            const field: *?u64 = if (std.mem.eql(u8, name, "_pt"))
                &self.producer_type
            else if (std.mem.eql(u8, name, "_p"))
                &self.producer_id
            else if (std.mem.eql(u8, name, "_ct"))
                &self.consumer_type
            else if (std.mem.eql(u8, name, "_c"))
                &self.consumer_id
            else
                continue;
            field.* = switch (c.tensorflow_profiler_XStat_value_case(stat)) {
                c.tensorflow_profiler_XStat_value_uint64_value => c.tensorflow_profiler_XStat_uint64_value(stat),
                c.tensorflow_profiler_XStat_value_int64_value => @bitCast(c.tensorflow_profiler_XStat_int64_value(stat)),
                else => null,
            };
        }
    }
};

fn streamFlows(allocator: std.mem.Allocator, writer: *std.Io.Writer, array_writer: *JsonArrayWriter, planes: []PlaneContext) !void {
    // Index only correlation endpoints; the rest of the trace remains streamed.
    // A second pass makes matching independent of the order of XPlane threads.
    var endpoints: std.ArrayListUnmanaged(FlowEndpoint) = .empty;
    defer endpoints.deinit(allocator);
    for (planes) |*plane| {
        var line_count: usize = 0;
        const lines = c.tensorflow_profiler_XPlane_lines(plane.plane, &line_count) orelse continue;
        for (lines[0..line_count]) |line_c| {
            const line = line_c orelse continue;
            if (std.mem.eql(u8, lineDisplayName(line), kXlaAsyncOpLineName)) continue;
            const resource_id = try lineResourceId(line);
            var event_count: usize = 0;
            const events = c.tensorflow_profiler_XLine_events(line, &event_count) orelse continue;
            for (events[0..event_count]) |event_c| {
                const event = event_c orelse continue;
                const metadata_id = c.tensorflow_profiler_XEvent_metadata_id(event);
                if (try plane.isInternalEvent(allocator, metadata_id)) continue;
                var context: FlowContext = .{};
                if (try plane.getEventMetadata(allocator, metadata_id)) |metadata| {
                    try context.readStats(allocator, plane, metadataStats(metadata));
                }
                try context.readStats(allocator, plane, eventStats(event));
                var endpoint: FlowEndpoint = .{
                    .key = undefined,
                    .zml_origin = context.zml_origin,
                    .role = .producer,
                    .timestamp_ps = eventTimestampPs(line, event),
                    .device_id = plane.device_id,
                    .resource_id = resource_id,
                    .event = event,
                };
                if (context.producer_type) |context_type| {
                    if (context.producer_id) |context_id| {
                        endpoint.key = .{ .context_type = context_type, .context_id = context_id };
                        try endpoints.append(allocator, endpoint);
                    }
                }
                if (context.consumer_type) |context_type| {
                    if (context.consumer_id) |context_id| {
                        endpoint.key = .{ .context_type = context_type, .context_id = context_id };
                        endpoint.role = .consumer;
                        try endpoints.append(allocator, endpoint);
                    }
                }
            }
        }
    }
    try streamFlowEndpoints(writer, array_writer, endpoints.items);
}

fn streamFlowEndpoints(writer: *std.Io.Writer, array_writer: *JsonArrayWriter, endpoints: []FlowEndpoint) !void {
    std.mem.sort(FlowEndpoint, endpoints, {}, FlowEndpoint.lessThan);
    var producer: ?FlowEndpoint = null;
    var flow_id: u64 = 0;
    for (endpoints) |endpoint| {
        if (producer) |previous| {
            if (!std.meta.eql(previous.key, endpoint.key)) producer = null;
        }
        if (endpoint.role == .producer) {
            producer = endpoint;
        } else if (producer) |previous| {
            if (previous.event == endpoint.event) continue;
            // A fresh flow per consumer preserves fan-out without inventing
            // consumer-to-consumer dependencies. Reused IDs select the latest producer.
            flow_id += 1;
            // Use one category at both endpoints, including ZML/backend handoffs.
            const category = if (previous.zml_origin or endpoint.zml_origin) "zml.flow" else "xla.flow";
            try streamFlowPoint(writer, array_writer, previous, flow_id, 's', category);
            try streamFlowPoint(writer, array_writer, endpoint, flow_id, 'f', category);
        }
    }
}

fn streamFlowPoint(writer: *std.Io.Writer, array_writer: *JsonArrayWriter, endpoint: FlowEndpoint, flow_id: u64, phase: u8, category: []const u8) !void {
    try array_writer.nextItem(writer);
    try writer.print("{{\"ph\":\"{c}\",\"cat\":\"{s}\",\"name\":\"producer_consumer\",\"id\":\"{x}\",\"pid\":{d},\"tid\":{d},\"ts\":", .{ phase, category, flow_id, endpoint.device_id, endpoint.resource_id });
    var number_buf: [64]u8 = undefined;
    try writer.writeAll(formatPicosToMicros(&number_buf, endpoint.timestamp_ps));
    if (phase == 'f') try writer.writeAll(",\"bp\":\"e\"");
    try writer.writeByte('}');
}

fn appendStats(
    allocator: std.mem.Allocator,
    plane: *PlaneContext,
    args: *std.ArrayListUnmanaged(Arg),
    event_name: *StringValue,
    stats: []const [*c]const c.tensorflow_profiler_XStat,
) !void {
    // Stat filtering and name lookup correspond to:
    // ../../xla/xla/tsl/profiler/utils/xplane_schema.cc
    for (stats) |stat_c| {
        const stat: *const c.tensorflow_profiler_XStat = stat_c orelse continue;
        const value_case = c.tensorflow_profiler_XStat_value_case(stat);
        if (value_case == c.tensorflow_profiler_XStat_value_NOT_SET) continue;

        const metadata_id = c.tensorflow_profiler_XStat_metadata_id(stat);
        const behavior = try plane.statBehavior(allocator, metadata_id);
        if (behavior == .skip) continue;

        const stat_value = try stringValueFromStat(allocator, plane, stat);
        if (behavior == .step_name) {
            event_name.* = stat_value;
        }

        const stat_name = if (try plane.getStatMetadata(allocator, metadata_id)) |metadata|
            stringView(c.tensorflow_profiler_XStatMetadata_name(metadata))
        else
            "";
        try putArg(args, allocator, stat_name, stat_value);
    }
}

fn putArg(
    args: *std.ArrayListUnmanaged(Arg),
    allocator: std.mem.Allocator,
    name: []const u8,
    value: StringValue,
) !void {
    for (args.items) |*arg| {
        if (std.mem.eql(u8, arg.name, name)) {
            arg.value = value;
            return;
        }
    }
    try args.append(allocator, .{ .name = name, .value = value });
}

fn metadataStats(metadata: *const c.tensorflow_profiler_XEventMetadata) []const [*c]const c.tensorflow_profiler_XStat {
    var stat_count: usize = 0;
    const stats = c.tensorflow_profiler_XEventMetadata_stats(metadata, &stat_count) orelse return &.{};
    return stats[0..stat_count];
}

fn eventStats(event: *const c.tensorflow_profiler_XEvent) []const [*c]const c.tensorflow_profiler_XStat {
    var stat_count: usize = 0;
    const stats = c.tensorflow_profiler_XEvent_stats(event, &stat_count) orelse return &.{};
    return stats[0..stat_count];
}

fn stringValueFromStat(
    allocator: std.mem.Allocator,
    plane: *PlaneContext,
    stat: *const c.tensorflow_profiler_XStat,
) !StringValue {
    // Stringification intentionally matches XStatVisitor::ToString():
    // ../../xla/xla/tsl/profiler/utils/xplane_visitor.cc
    return switch (c.tensorflow_profiler_XStat_value_case(stat)) {
        c.tensorflow_profiler_XStat_value_int64_value => .{ .int64 = c.tensorflow_profiler_XStat_int64_value(stat) },
        c.tensorflow_profiler_XStat_value_uint64_value => .{ .uint64 = c.tensorflow_profiler_XStat_uint64_value(stat) },
        c.tensorflow_profiler_XStat_value_double_value => .{ .double = c.tensorflow_profiler_XStat_double_value(stat) },
        c.tensorflow_profiler_XStat_value_str_value => .{ .text = stringView(c.tensorflow_profiler_XStat_str_value(stat)) },
        c.tensorflow_profiler_XStat_value_bytes_value => .opaque_bytes,
        c.tensorflow_profiler_XStat_value_ref_value => blk: {
            const ref_metadata_id = std.math.cast(i64, c.tensorflow_profiler_XStat_ref_value(stat)) orelse return error.InvalidStatMetadataId;
            const text = if (try plane.getStatMetadata(allocator, ref_metadata_id)) |metadata|
                stringView(c.tensorflow_profiler_XStatMetadata_name(metadata))
            else
                "";
            break :blk .{ .text = text };
        },
        else => unreachable,
    };
}

fn writeJsonStringValue(writer: *std.Io.Writer, value: StringValue) !void {
    switch (value) {
        .text => |text| try writeJsonString(writer, text),
        .opaque_bytes => try writeJsonString(writer, kOpaqueBytesText),
        .int64 => |number| {
            var buffer: [64]u8 = undefined;
            const text = try std.fmt.bufPrint(&buffer, "{d}", .{number});
            try writeJsonString(writer, text);
        },
        .uint64 => |number| {
            var buffer: [64]u8 = undefined;
            const text = try std.fmt.bufPrint(&buffer, "{d}", .{number});
            try writeJsonString(writer, text);
        },
        .double => |number| {
            var buffer: [64]u8 = undefined;
            try writeJsonString(writer, formatDouble(&buffer, "%g", number));
        },
    }
}

fn writeJsonString(writer: *std.Io.Writer, text: []const u8) !void {
    try writer.writeByte('"');

    var chunk_start: usize = 0;
    for (text, 0..) |byte, index| {
        const escape = switch (byte) {
            '"' => "\\\"",
            '\\' => "\\\\",
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            0x08 => "\\b",
            0x0c => "\\f",
            else => null,
        };
        if (escape) |escaped| {
            if (chunk_start < index) {
                try writer.writeAll(text[chunk_start..index]);
            }
            try writer.writeAll(escaped);
            chunk_start = index + 1;
            continue;
        }
        if (byte < 0x20) {
            if (chunk_start < index) {
                try writer.writeAll(text[chunk_start..index]);
            }
            try writeAsciiControlEscape(writer, byte);
            chunk_start = index + 1;
        }
    }

    if (chunk_start < text.len) {
        try writer.writeAll(text[chunk_start..]);
    }

    try writer.writeByte('"');
}

fn lineResourceId(line: *const c.tensorflow_profiler_XLine) !u32 {
    const display_id = c.tensorflow_profiler_XLine_display_id(line);
    const effective_id = if (display_id != 0) display_id else c.tensorflow_profiler_XLine_id(line);
    return wrapI64ToU32(effective_id);
}

fn lineDisplayName(line: *const c.tensorflow_profiler_XLine) []const u8 {
    const display_name = stringView(c.tensorflow_profiler_XLine_display_name(line));
    if (display_name.len != 0) return display_name;
    return stringView(c.tensorflow_profiler_XLine_name(line));
}

fn eventTimestampPs(line: *const c.tensorflow_profiler_XLine, event: *const c.tensorflow_profiler_XEvent) u64 {
    const line_timestamp_ns = reinterpretSignedAsUnsigned64(c.tensorflow_profiler_XLine_timestamp_ns(line));
    const event_offset_ps = reinterpretSignedAsUnsigned64(c.tensorflow_profiler_XEvent_offset_ps(event));
    return line_timestamp_ns *% kPicosecondsPerNanosecond +% event_offset_ps;
}

fn eventDurationPs(event: *const c.tensorflow_profiler_XEvent) u64 {
    const duration_ps = reinterpretSignedAsUnsigned64(c.tensorflow_profiler_XEvent_duration_ps(event));
    return @max(duration_ps, 1);
}

fn formatPicosToMicros(buffer: *[64]u8, picoseconds: u64) []const u8 {
    // The conversion and "%.17g" formatting are kept identical to:
    // ../../xla/xla/tsl/profiler/utils/math_utils.h
    // ../../xla/xla/tsl/profiler/utils/format_utils.h
    const microseconds = @as(f64, @floatFromInt(picoseconds)) / kPicosecondsPerMicrosecond;
    return formatDouble(buffer, "%.17g", microseconds);
}

fn formatDouble(buffer: *[64]u8, comptime fmt: [*:0]const u8, value: f64) []const u8 {
    const written = c.snprintf(buffer.ptr, buffer.len, fmt, value);
    if (written <= 0 or written >= buffer.len) unreachable;
    return buffer[0..@intCast(written)];
}

fn stringView(value: c.upb_StringView) []const u8 {
    const text = upb.slice(value);
    return text orelse "";
}

fn reinterpretSignedAsUnsigned64(value: i64) u64 {
    const same_bits: u64 = @bitCast(value);
    return same_bits;
}

fn wrapI64ToU32(value: i64) u32 {
    const unsigned_value = reinterpretSignedAsUnsigned64(value);
    return @truncate(unsigned_value);
}

fn writeAsciiControlEscape(writer: *std.Io.Writer, byte: u8) !void {
    const high_nibble = @as(usize, byte / 16);
    const low_nibble = @as(usize, byte % 16);

    try writer.writeAll("\\u00");
    try writer.writeByte(kHexDigits[high_nibble]);
    try writer.writeByte(kHexDigits[low_nibble]);
}

fn argLessThan(_: void, lhs: Arg, rhs: Arg) bool {
    return std.mem.order(u8, lhs.name, rhs.name) == .lt;
}

fn isInternalEventName(name: []const u8) bool {
    // Internal host-event filtering is derived from:
    // ../../xla/xla/tsl/profiler/utils/xplane_schema.cc
    inline for ([_][]const u8{
        "MemoryAllocation",
        "MemoryDeallocation",
        "PrefetchProduce",
        "PrefetchConsume",
        "ParallelInterleaveProduce",
        "ParallelInterleaveConsume",
        "ParallelInterleaveInitializeInput",
        "ParallelMapProduce",
        "ParallelMapConsume",
        "MapAndBatchProduce",
        "MapAndBatchConsume",
        "ParseExampleProduce",
        "ParseExampleConsume",
    }) |candidate| {
        if (std.mem.eql(u8, name, candidate)) return true;
    }
    return false;
}

fn classifyStatName(name: []const u8) StatBehavior {
    // Internal stat filtering and step_name handling are derived from:
    // ../../xla/xla/tsl/profiler/utils/xplane_schema.cc
    if (std.mem.eql(u8, name, "step_name")) return .step_name;
    inline for ([_][]const u8{
        "_pt",
        "_ct",
        "_p",
        "_c",
        "_zml_flow",
        "_r",
        "flops",
        "program_id",
        "symbol_id",
    }) |candidate| {
        if (std.mem.eql(u8, name, candidate)) return .skip;
    }
    return .emit;
}

// Deliberately store consumers before producers in XPlane order. Metadata holds
// the correlation fields, and individual events override the IDs/types.
test "flows preserve fan-out, namespaces, reused IDs and ZML categories" {
    const allocator = std.testing.allocator;
    const arena = c.upb_Arena_New() orelse return error.OutOfMemory;
    defer c.upb_Arena_Free(arena);
    const space = try upb.new(c.tensorflow_profiler_XSpace, arena);
    const plane = c.tensorflow_profiler_XSpace_add_planes(space, arena) orelse return error.OutOfMemory;
    c.tensorflow_profiler_XPlane_set_name(plane, upb.stringView(kHostThreadsPlaneName));
    for ([_][]const u8{ "_pt", "_p", "_ct", "_c", "_zml_flow" }, 1..) |name, index| {
        const metadata = try upb.new(c.tensorflow_profiler_XStatMetadata, arena);
        c.tensorflow_profiler_XStatMetadata_set_name(metadata, upb.stringView(name));
        try std.testing.expect(c.tensorflow_profiler_XPlane_stat_metadata_set(plane, @intCast(index), metadata, arena));
    }
    const id: u64 = std.math.maxInt(u64);
    for ([_][]const u8{ "producer", "consumer" }, 1..) |name, index| {
        const metadata = try upb.new(c.tensorflow_profiler_XEventMetadata, arena);
        c.tensorflow_profiler_XEventMetadata_set_name(metadata, upb.stringView(name));
        const context_type = c.tensorflow_profiler_XEventMetadata_add_stats(metadata, arena) orelse return error.OutOfMemory;
        c.tensorflow_profiler_XStat_set_metadata_id(context_type, if (index == 1) 1 else 3);
        c.tensorflow_profiler_XStat_set_uint64_value(context_type, 1);
        const context_id = c.tensorflow_profiler_XEventMetadata_add_stats(metadata, arena) orelse return error.OutOfMemory;
        c.tensorflow_profiler_XStat_set_metadata_id(context_id, if (index == 1) 2 else 4);
        c.tensorflow_profiler_XStat_set_uint64_value(context_id, id);
        try std.testing.expect(c.tensorflow_profiler_XPlane_event_metadata_set(plane, @intCast(index), metadata, arena));
    }
    try addFlowTestEvent(plane, arena, 2, 20, null, false); // A -> consumer 1.
    try addFlowTestEvent(plane, arena, 1, 10, null, false); // Producer A.
    try addFlowTestEvent(plane, arena, 2, 30, null, false); // A -> consumer 2.
    try addFlowTestEvent(plane, arena, 1, 15, 2, true); // Same ID, different context.
    try addFlowTestEvent(plane, arena, 2, 35, 2, false);
    try addFlowTestEvent(plane, arena, 1, 40, null, false); // Reuse the ID for producer B.
    try addFlowTestEvent(plane, arena, 2, 50, null, true); // B -> consumer.
    try addFlowTestEvent(plane, arena, 2, 5, null, false); // Consumer before any producer.
    try addFlowTestEvent(plane, arena, 2, 70, 3, false); // Unmatched context.

    var writer: std.Io.Writer.Allocating = .init(allocator);
    defer writer.deinit();
    try streamXSpaceToTraceJson(allocator, space, &writer.writer);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, writer.written(), .{});
    defer parsed.deinit();
    const events = parsed.value.object.get("traceEvents").?.array.items;
    var sources: std.StringHashMapUnmanaged(i64) = .empty;
    defer sources.deinit(allocator);
    var source_categories: std.StringHashMapUnmanaged([]const u8) = .empty;
    defer source_categories.deinit(allocator);
    var consumers: usize = 0;
    for (events) |event| {
        const fields = event.object;
        const phase = fields.get("ph").?.string;
        if (fields.get("args")) |args| try std.testing.expect(!args.object.contains("_zml_flow"));
        if (std.mem.eql(u8, phase, "s")) {
            try sources.put(allocator, fields.get("id").?.string, fields.get("ts").?.integer);
            try source_categories.put(allocator, fields.get("id").?.string, fields.get("cat").?.string);
        } else if (std.mem.eql(u8, phase, "f")) {
            const start = sources.get(fields.get("id").?.string).?;
            const end = fields.get("ts").?.integer;
            const category = if (end == 35 or end == 50) "zml.flow" else "xla.flow";
            try std.testing.expectEqualStrings(category, fields.get("cat").?.string);
            try std.testing.expectEqualStrings(category, source_categories.get(fields.get("id").?.string).?);
            try std.testing.expectEqual(@as(i64, switch (end) {
                20, 30 => 10,
                35 => 15,
                50 => 40,
                else => return error.UnexpectedConsumer,
            }), start);
            try std.testing.expectEqualStrings("e", fields.get("bp").?.string);
            consumers += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 4), sources.count());
    try std.testing.expectEqual(@as(usize, 4), consumers);
}

fn addFlowTestEvent(plane: *c.tensorflow_profiler_XPlane, arena: *c.upb_Arena, metadata_id: i64, time_us: i64, context_type: ?u64, zml_origin: bool) !void {
    const line = c.tensorflow_profiler_XPlane_add_lines(plane, arena) orelse return error.OutOfMemory;
    c.tensorflow_profiler_XLine_set_id(line, metadata_id);
    const event = c.tensorflow_profiler_XLine_add_events(line, arena) orelse return error.OutOfMemory;
    c.tensorflow_profiler_XEvent_set_metadata_id(event, metadata_id);
    c.tensorflow_profiler_XEvent_set_offset_ps(event, time_us * 1_000_000);
    c.tensorflow_profiler_XEvent_set_duration_ps(event, 1_000_000);
    if (zml_origin) {
        const stat = c.tensorflow_profiler_XEvent_add_stats(event, arena) orelse return error.OutOfMemory;
        c.tensorflow_profiler_XStat_set_metadata_id(stat, 5);
        if (metadata_id == 1) {
            c.tensorflow_profiler_XStat_set_int64_value(stat, 1);
        } else {
            c.tensorflow_profiler_XStat_set_uint64_value(stat, 1);
        }
    }
    if (context_type) |value| {
        const stat = c.tensorflow_profiler_XEvent_add_stats(event, arena) orelse return error.OutOfMemory;
        c.tensorflow_profiler_XStat_set_metadata_id(stat, if (metadata_id == 1) 1 else 3);
        c.tensorflow_profiler_XStat_set_uint64_value(stat, value);
    }
}
