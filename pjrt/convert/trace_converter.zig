const std = @import("std");
const xplane_proto = @import("//tsl:xplane_proto");

const TraceContainer = @import("trace_container.zig").TraceContainer;

pub const TraceConverter = struct {
    arena: std.heap.ArenaAllocator,
    container: TraceContainer = .{},
    xspace: xplane_proto.XSpace = .{},

    pub fn init(allocator: std.mem.Allocator, path: []const u8) !TraceConverter {
        var res: TraceConverter = .{
            .arena = std.heap.ArenaAllocator.init(allocator),
        };

        var fd = try std.fs.openFileAbsolute(path, .{});
        defer fd.close();

        const arena = res.arena.allocator();

        const pb_buffer = try fd.readToEndAlloc(arena, (try fd.stat()).size);
        if (pb_buffer.len == 0) return error.EmptyBuffer;

        res.xspace = try xplane_proto.XSpace.decode(pb_buffer, arena);

        var events: usize = 0;
        for (res.xspace.planes.items) |plane| {
            for (plane.lines.items) |line| {
                events += line.events.items.len;
            }
        }

        std.debug.print("Found {d} events across {d} spaces.\n", .{ events, res.xspace.planes.items.len });
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
    ) ![]const *const T {
        const pairs = try allocator.alloc(*const T, a.items.len);
        for (a.items, 0..) |*pair, i| {
            pairs[i] = pair;
        }
        std.mem.sort(
            *const T,
            pairs,
            {},
            struct {
                pub fn call(_: void, lhs: *const T, rhs: *const T) bool {
                    return lhs.key < rhs.key;
                }
            }.call,
        );
        return pairs;
    }

    fn picoToMicro(p: anytype) f64 {
        return @as(f64, @floatFromInt(p)) / 1E6;
    }

    pub fn toJson(self: *TraceConverter, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        errdefer buffer.deinit();
        var writer = buffer.writer();
        try writer.writeAll(
            \\{"displayTimeUnit":"ns","metadata":{"highres-ticks":true},"traceEvents":[
        );

        // TODO: finish implementing
        const Entry = std.AutoHashMapUnmanaged(u32, TraceContainer.Device).Entry;
        const pairs = try allocator.alloc(Entry, self.container.metadata.devices.count());
        defer allocator.free(pairs);
        var iter = self.container.metadata.devices.iterator();
        var idx: usize = 0;
        while (iter.next()) |entry| : (idx += 1) {
            pairs[idx] = entry;
        }

        std.mem.sort(
            Entry,
            pairs,
            {},
            struct {
                pub fn call(_: void, lhs: Entry, rhs: Entry) bool {
                    return lhs.key_ptr.* < rhs.key_ptr.*;
                }
            }.call,
        );

        for (pairs) |id_and_device| {
            const device_id = id_and_device.key_ptr.*;
            const device = id_and_device.value_ptr.*;
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

            const ResourceEntry = std.AutoHashMapUnmanaged(u32, TraceContainer.Resource).Entry;
            const resources = try allocator.alloc(ResourceEntry, device.resources.count());
            defer allocator.free(resources);

            var resource_iter = device.resources.iterator();
            idx = 0;
            while (resource_iter.next()) |entry| : (idx += 1) {
                resources[idx] = entry;
            }

            std.mem.sort(
                ResourceEntry,
                resources,
                {},
                struct {
                    pub fn call(_: void, lhs: ResourceEntry, rhs: ResourceEntry) bool {
                        return lhs.key_ptr.* < rhs.key_ptr.*;
                    }
                }.call,
            );

            for (resources) |id_and_resource| {
                const resource_id = id_and_resource.key_ptr.*;
                const resource = id_and_resource.value_ptr.*;
                if (resource.name.len != 0) {
                    try writer.print(
                        \\{{"ph":"M","pid":{d},"tid":{d},"name":"thread_name","args":{{"name":"{s}"}}}},
                    , .{
                        device_id,
                        resource_id,
                        resource.name,
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
                const ArgsEntry = std.StringHashMapUnmanaged([]const u8).Entry;
                const sorted_args = try allocator.alloc(ArgsEntry, event.args.count());
                defer allocator.free(sorted_args);
                var args_iter = event.args.iterator();
                idx = 0;
                while (args_iter.next()) |entry| : (idx += 1) {
                    sorted_args[idx] = entry;
                }

                std.mem.sort(ArgsEntry, sorted_args, {}, struct {
                    pub fn call(_: void, lhs: ArgsEntry, rhs: ArgsEntry) bool {
                        return std.mem.order(u8, lhs.key_ptr.*, rhs.key_ptr.*).compare(std.math.CompareOperator.lt);
                    }
                }.call);
                for (sorted_args) |arg| {
                    try writer.print(
                        \\"{s}":"{s}",
                    , .{ arg.key_ptr.*, arg.value_ptr.* });
                }

                // Replace trailing comma with closing brace.
                buffer.items[buffer.items.len - 1] = '}';
            }
            try writer.writeAll("},");
        }
        try writer.writeAll("{}]}");
        return buffer.toOwnedSlice();
    }
};
