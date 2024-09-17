const std = @import("std");
const utils = @import("utils.zig");
const yaml = @import("zig-yaml");
const zml = @import("../zml.zig");

const Allocator = std.mem.Allocator;

const StringBuilder = std.ArrayListUnmanaged(u8);

pub fn open(allocator: Allocator, path: []const u8) !zml.aio.BufferStore {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    var res: zml.aio.BufferStore = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
    };
    errdefer res.arena.deinit();
    const arena = res.arena.allocator();

    const yaml_data = try file.reader().readAllAlloc(arena, (try file.metadata()).size());
    const parsed = try yaml.Yaml.load(arena, yaml_data);

    const map = parsed.docs.items[0].map;
    var map_iter = map.iterator();
    while (map_iter.next()) |entry| {
        var prefix_buf: [1024]u8 = undefined;
        try parseMetadata(arena, &res, StringBuilder.initBuffer(&prefix_buf), entry.key, entry.value);
    }
    return res;
}

pub fn parseMetadata(allocator: Allocator, store: *zml.aio.BufferStore, key: StringBuilder, val: yaml.Value) !void {
    switch (val) {
        .int => |v| try store.metadata.put(allocator, try allocator.dupe(u8, key.items), .{ .int64 = v }),
        .float => |v| try store.metadata.put(allocator, try allocator.dupe(u8, key.items), .{ .float64 = v }),
        .string => |v| try store.metadata.put(allocator, try allocator.dupe(u8, key.items), .{ .string = v }),
        .list => |v| switch (validSlice(v)) {
            true => {
                if (v.len == 0) return;
                switch (v[0]) {
                    .int => {
                        const values = try allocator.alloc(i64, v.len);
                        errdefer allocator.free(values);
                        for (v, 0..) |item, i| values[i] = item.int;
                        try store.metadata.put(allocator, try allocator.dupe(u8, key.items), .{ .array = .{ .item_type = .int64, .data = utils.toVoidSlice(values) } });
                    },
                    .float => {
                        const values = try allocator.alloc(f64, v.len);
                        errdefer allocator.free(values);
                        for (v, 0..) |item, i| values[i] = item.float;
                        try store.metadata.put(allocator, try allocator.dupe(u8, key.items), .{ .array = .{ .item_type = .float64, .data = utils.toVoidSlice(values) } });
                    },
                    .string => {
                        const values = try allocator.alloc([]const u8, v.len);
                        errdefer allocator.free(values);
                        for (v, 0..) |item, i| {
                            values[i] = try allocator.dupe(u8, item.string);
                        }
                        try store.metadata.put(allocator, try allocator.dupe(u8, key.items), .{ .array = .{ .item_type = .string, .data = utils.toVoidSlice(values) } });
                    },
                    .list => unreachable,
                    else => {},
                }
            },
            false => for (v, 0..) |item, i| {
                var new_key = key;
                if (key.items.len > 0)
                    new_key.appendAssumeCapacity('.');
                new_key.items.len += std.fmt.formatIntBuf(new_key.unusedCapacitySlice(), i, 10, .lower, .{});
                try parseMetadata(allocator, store, new_key, item);
            },
        },
        .map => {
            var map_iter = val.map.iterator();
            while (map_iter.next()) |entry| {
                var new_prefix = key;
                if (key.items.len > 0)
                    new_prefix.appendAssumeCapacity('.');
                new_prefix.appendSliceAssumeCapacity(entry.key_ptr.*);
                try parseMetadata(allocator, store, new_prefix, entry.value_ptr.*);
            }
        },
        else => {},
    }
}

fn validSlice(v: []yaml.Value) bool {
    if (v.len == 0) return false;
    const item_type = std.meta.activeTag(v[0]);
    switch (item_type) {
        .empty, .list, .map => return false,
        else => {},
    }

    for (v[1..]) |item|
        if (item_type != std.meta.activeTag(item)) return false;

    return true;
}
