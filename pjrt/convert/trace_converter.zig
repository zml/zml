const std = @import("std");
const trace_events_proto = @import("//tsl:trace_events_proto");
const xplane_proto = @import("//tsl:xplane_proto");

const TraceContainer = @import("trace_container.zig").TraceContainer;

pub const TraceConverter = struct {
    arena: std.heap.ArenaAllocator,
    container: TraceContainer = .{},
    xspace: xplane_proto.XSpace = .{},

    pub fn init(allocator: std.mem.Allocator, pb_buffer: []const u8) !TraceConverter {
        var res: TraceConverter = .{
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
        const arena = res.arena.allocator();
        res.xspace = try xplane_proto.XSpace.decode(pb_buffer, arena);
        res.container = try TraceContainer.fromXSpace(arena, &res.xspace);
        return res;
    }

    pub fn deinit(self: *TraceConverter) void {
        self.arena.deinit();
    }

    pub fn sortByKey(
        allocator: std.mem.Allocator,
        comptime T: type,
        a: std.ArrayListUnmanaged(T),
        comptime lt_fn: fn (ctx: void, lhs: *const T, rhs: *const T) bool,
    ) ![]const *const T {
        const pairs = try allocator.alloc(*const T, a.items.len);
        for (a.items, 0..) |*pair, i| {
            pairs[i] = pair;
        }
        std.mem.sort(
            *const T,
            pairs,
            {},
            lt_fn,
        );
        return pairs;
    }

    fn picoToMicro(p: anytype) f64 {
        return @as(f64, @floatFromInt(p)) / 1E6;
    }

    pub fn toJson(self: *TraceConverter, writer: std.io.AnyWriter) !void {
        try writer.writeAll(
            \\{"displayTimeUnit":"ns","metadata":{"highres-ticks":true},"traceEvents":[
        );

        self.container.metadata.devices.sort(struct {
            keys: []const u32,
            pub fn lessThan(ctx: @This(), lhs: usize, rhs: usize) bool {
                return ctx.keys[lhs] < ctx.keys[rhs];
            }
        }{ .keys = self.container.metadata.devices.keys() });

        for (self.container.metadata.devices.keys(), self.container.metadata.devices.values()) |device_id, *device| {
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

        for (self.container.events.items) |event| {
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
