const std = @import("std");
const zml = @import("zml");

comptime {
    // ZML's link dependencies retain this initializer even without a Platform.
    _ = zml.bazel;
}

pub const Node = struct {
    name: []const u8,
    fullName: []const u8,
    parent: ?*Node,
    depth: usize,
    children: std.StringArrayHashMapUnmanaged(*Node) = .empty,
    tensor: ?zml.safetensors.Tensor = null,
    expanded: bool = true,
    searchCollapsed: bool = false,
    matches: bool = true,
};

pub const Tree = struct {
    arena: std.heap.ArenaAllocator,
    allocator: std.mem.Allocator,
    root: *Node,
    visible: std.ArrayList(*Node) = .empty,
    selected: usize = 0,
    searching: bool = false,

    pub fn init(allocator: std.mem.Allocator, registry: *zml.safetensors.TensorRegistry) !Tree {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        errdefer arena.deinit();
        const a = arena.allocator();
        const root = try a.create(Node);
        root.* = .{ .name = "", .fullName = "", .parent = null, .depth = 0 };
        var it = registry.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            var walk = root;
            var parts = std.mem.splitScalar(u8, name, '.');
            while (parts.next()) |part| {
                const gop = try walk.children.getOrPut(a, part);
                if (!gop.found_existing) {
                    const child = try a.create(Node);
                    child.* = .{
                        .name = part,
                        .fullName = name[0 .. @intFromPtr(part.ptr) - @intFromPtr(name.ptr) + part.len],
                        .parent = walk,
                        .depth = walk.depth + 1,
                    };
                    gop.value_ptr.* = child;
                }
                walk = gop.value_ptr.*;
            }
            walk.tensor = entry.value_ptr.*;
        }
        sort(root);
        var self: Tree = .{ .arena = arena, .allocator = allocator, .root = root };
        errdefer self.visible.deinit(allocator);
        try self.rebuild();
        return self;
    }

    pub fn deinit(self: *Tree) void {
        self.visible.deinit(self.allocator);
        self.arena.deinit();
    }

    fn sort(node: *Node) void {
        const Context = struct {
            nodes: []*Node,
            pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                const aName = ctx.nodes[a].name;
                const bName = ctx.nodes[b].name;
                const ln = std.fmt.parseInt(usize, aName, 10) catch null;
                const rn = std.fmt.parseInt(usize, bName, 10) catch null;
                if (ln != null and rn != null and ln.? != rn.?) return ln.? < rn.?;
                return std.mem.lessThan(u8, aName, bName);
            }
        };
        node.children.sort(Context{ .nodes = node.children.values() });
        for (node.children.values()) |child| sort(child);
    }

    pub fn current(self: *const Tree) ?*Node {
        return if (self.visible.items.len == 0) null else self.visible.items[self.selected];
    }

    pub fn isExpanded(self: *const Tree, node: *Node) bool {
        return if (self.searching) !node.searchCollapsed else node.expanded;
    }

    pub fn move(self: *Tree, down: bool) void {
        const count = self.visible.items.len;
        if (count == 0) return;
        self.selected = if (down) (self.selected + 1) % count else (self.selected + count - 1) % count;
    }

    pub fn toggle(self: *Tree) !void {
        const node = self.current() orelse return;
        if (node.children.count() == 0) return;
        if (self.searching) node.searchCollapsed = !node.searchCollapsed else node.expanded = !node.expanded;
        try self.rebuild();
    }

    pub fn left(self: *Tree) !void {
        const node = self.current() orelse return;
        if (node.children.count() > 0 and self.isExpanded(node)) return self.toggle();
        for (self.visible.items, 0..) |candidate, i| {
            if (candidate == node.parent) {
                self.selected = i;
                return;
            }
        }
    }

    pub fn right(self: *Tree) !void {
        const node = self.current() orelse return;
        if (node.children.count() == 0) return;
        if (!self.isExpanded(node)) return self.toggle();
        if (self.selected + 1 < self.visible.items.len) self.selected += 1;
    }

    pub fn filter(self: *Tree, query: []const u8) !void {
        self.searching = query.len != 0;
        _ = markMatches(self.root, query);
        self.selected = 0;
        try self.rebuild();
    }

    fn markMatches(node: *Node, query: []const u8) bool {
        node.searchCollapsed = false;
        node.matches = node.tensor != null and std.ascii.indexOfIgnoreCase(node.fullName, query) != null;
        for (node.children.values()) |child| {
            const childMatches = markMatches(child, query);
            node.matches = node.matches or childMatches;
        }
        return node.matches;
    }

    fn rebuild(self: *Tree) !void {
        self.visible.clearRetainingCapacity();
        try self.appendChildren(self.root);
        self.selected = @min(self.selected, self.visible.items.len -| 1);
    }

    fn appendChildren(self: *Tree, node: *Node) std.mem.Allocator.Error!void {
        for (node.children.values()) |child| {
            if (self.searching and !child.matches) continue;
            try self.visible.append(self.allocator, child);
            if (self.isExpanded(child)) try self.appendChildren(child);
        }
    }
};

test "natural ordering, parent navigation, wrapping, and filtered ancestors" {
    var registry: zml.safetensors.TensorRegistry = .init(std.testing.allocator);
    defer registry.deinit();
    for ([_][]const u8{ "model.layers.10.weight", "model.layers.2.weight", "model.norm" }) |name| {
        try registry.registerTensor(.{ .name = name, .file_uri = "shard.safetensors", .shape = .init(.{2}, .f32), .offset = 128 });
    }
    var tree = try Tree.init(std.testing.allocator, &registry);
    defer tree.deinit();
    try std.testing.expectEqualStrings("model.layers.2", tree.visible.items[2].fullName);
    tree.move(false);
    try std.testing.expectEqualStrings("model.norm", tree.current().?.fullName);
    tree.move(true);
    try tree.toggle();
    try std.testing.expectEqual(@as(usize, 1), tree.visible.items.len);
    try tree.filter("LAYERS.10.WEIGHT");
    try std.testing.expectEqual(@as(usize, 4), tree.visible.items.len);
    tree.selected = 3;
    try tree.left();
    try std.testing.expectEqualStrings("model.layers.10", tree.current().?.fullName);
    try tree.toggle();
    try std.testing.expectEqual(@as(usize, 3), tree.visible.items.len);
    try tree.filter("");
    try std.testing.expectEqual(@as(usize, 1), tree.visible.items.len);
    try tree.right();
    try std.testing.expectEqual(@as(usize, 7), tree.visible.items.len);
    try tree.filter("missing");
    tree.move(true);
    try tree.left();
    try tree.right();
    try std.testing.expect(tree.current() == null);
}

test "empty registry and tensor that is also a parent" {
    var registry: zml.safetensors.TensorRegistry = .init(std.testing.allocator);
    defer registry.deinit();
    var empty = try Tree.init(std.testing.allocator, &registry);
    defer empty.deinit();
    try std.testing.expect(empty.current() == null);
    for ([_][]const u8{ "a", "a.b" }) |name| {
        try registry.registerTensor(.{ .name = name, .file_uri = "model.safetensors", .shape = .init(.{}, .f32), .offset = 80 });
    }
    var tree = try Tree.init(std.testing.allocator, &registry);
    defer tree.deinit();
    try std.testing.expect(tree.current().?.tensor != null);
    try std.testing.expectEqual(@as(usize, 1), tree.current().?.children.count());
}
