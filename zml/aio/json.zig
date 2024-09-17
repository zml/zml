const asynk = @import("async");
const std = @import("std");
const utils = @import("utils.zig");
const zml = @import("../zml.zig");

const StringBuilder = std.ArrayListUnmanaged(u8);
const Allocator = std.mem.Allocator;

pub fn open(allocator: std.mem.Allocator, path: []const u8) !zml.aio.BufferStore {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    var res: zml.aio.BufferStore = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
    };
    errdefer res.arena.deinit();
    const arena = res.arena.allocator();

    const json_data = try file.reader().readAllAlloc(arena, (try file.metadata()).size());
    const metadata = try std.json.parseFromSliceLeaky(std.json.Value, allocator, json_data, .{ .allocate = .alloc_if_needed });

    var it = metadata.object.iterator();
    while (it.next()) |entry| {
        var prefix_buf: [1024]u8 = undefined;
        try parseMetadata(allocator, &res, StringBuilder.initBuffer(&prefix_buf), entry.value_ptr.*);
    }

    return res;
}

pub fn parseMetadata(allocator: Allocator, store: *zml.aio.BufferStore, key: StringBuilder, val: std.json.Value) !void {
    const metadata = &store._metadata;
    switch (val) {
        .null => try metadata.put(allocator, try allocator.dupe(u8, key.items), .{ .null = {} }),
        .bool => |v| try metadata.put(allocator, try allocator.dupe(u8, key.items), .{ .boolval = v }),
        .integer => |v| try metadata.put(allocator, try allocator.dupe(u8, key.items), .{ .int64 = v }),
        .float => |v| try metadata.put(allocator, try allocator.dupe(u8, key.items), .{ .float64 = v }),
        .number_string, .string => |v| try metadata.put(allocator, try allocator.dupe(u8, key.items), .{ .string = try allocator.dupe(u8, v) }),
        .array => |v| switch (validSlice(v)) {
            true => {
                if (v.items.len == 0) return;
                switch (v.items[0]) {
                    .bool => {
                        const values = try allocator.alloc(bool, v.items.len);
                        errdefer allocator.free(values);
                        for (v.items, 0..) |item, i| values[i] = item.bool;
                        try metadata.put(allocator, try allocator.dupe(u8, key.items), .{ .array = .{ .item_type = .boolval, .data = std.mem.sliceAsBytes(values) } });
                    },
                    .integer => {
                        const values = try allocator.alloc(i64, v.items.len);
                        errdefer allocator.free(values);
                        for (v.items, 0..) |item, i| values[i] = item.integer;
                        try metadata.put(allocator, try allocator.dupe(u8, key.items), .{ .array = .{ .item_type = .int64, .data = std.mem.sliceAsBytes(values) } });
                    },
                    .float => {
                        const values = try allocator.alloc(f64, v.items.len);
                        errdefer allocator.free(values);
                        for (v.items, 0..) |item, i| values[i] = item.float;
                        try metadata.put(allocator, try allocator.dupe(u8, key.items), .{ .array = .{ .item_type = .float64, .data = std.mem.sliceAsBytes(values) } });
                    },
                    inline .string, .number_string => |_, tag| {
                        const values = try allocator.alloc([]const u8, v.items.len);
                        errdefer allocator.free(values);
                        for (v.items, 0..) |item, i| {
                            values[i] = @field(item, @tagName(tag));
                        }
                        try metadata.put(allocator, try allocator.dupe(u8, key.items), .{ .array = .{ .item_type = .string, .data = std.mem.sliceAsBytes(values) } });
                    },
                    else => unreachable,
                }
            },
            false => for (v.items, 0..) |item, i| {
                var new_key = key;
                if (key.items.len > 0)
                    new_key.appendAssumeCapacity('.');
                new_key.items.len += std.fmt.formatIntBuf(new_key.unusedCapacitySlice(), i, 10, .lower, .{});
                try parseMetadata(allocator, store, new_key, item);
            },
        },
        .object => |v| {
            var obj_iter = v.iterator();
            while (obj_iter.next()) |entry| {
                var new_key = key;
                if (key.items.len > 0)
                    new_key.appendAssumeCapacity('.');
                new_key.appendSliceAssumeCapacity(entry.key_ptr.*);
                try parseMetadata(allocator, store, new_key, entry.value_ptr.*);
            }
        },
    }
}

fn validSlice(v: std.json.Array) bool {
    const item_type = std.meta.activeTag(v.items[0]);
    switch (item_type) {
        .null, .array, .object => return false,
        else => {},
    }

    for (v.items[1..]) |item|
        if (item_type != std.meta.activeTag(item)) return false;

    return true;
}
