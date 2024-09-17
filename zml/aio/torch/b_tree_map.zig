const std = @import("std");

/// BTreeMap Node implementation.
pub fn NodeType(comptime K: type, comptime V: type, comptime B: u32) type {
    return struct {
        const Self = @This();
        keys: [2 * B - 1]K = [_]K{undefined} ** (2 * B - 1),
        values: [2 * B - 1]V = [_]V{undefined} ** (2 * B - 1),
        len: usize = 0,
        edges: [2 * B]?*Self = [_]?*Self{null} ** (2 * B),

        pub const KV = struct { key: K, value: V };
        const KVE = struct { key: K, value: V, edge: ?*Self };
        const Entry = struct { key_ptr: *K, value_ptr: *V };

        /// Initializes an empty Node.
        pub fn initEmpty(allocator: std.mem.Allocator) !*Self {
            const res: *Self = try allocator.create(Self);
            res.* = .{};
            return res;
        }

        /// Initializes a Node with a single Entry.
        pub fn initKeyValue(allocator: std.mem.Allocator, entry: struct { K, V }) !*Self {
            const key, const value = entry;
            var res = try Self.initEmpty(allocator);
            res.keys[0] = key;
            res.values[0] = value;
            res.len = 1;
            return res;
        }

        fn initFromSplit(allocator: std.mem.Allocator, keys: []K, values: []V, edges: []?*Self) !*Self {
            var out = try Self.initEmpty(allocator);
            std.mem.copyBackwards(K, out.keys[0..], keys);
            std.mem.copyBackwards(V, out.values[0..], values);
            std.mem.copyBackwards(?*Self, out.edges[0..], edges);
            out.len = keys.len;
            return out;
        }

        pub fn count(self: Self) usize {
            var len: usize = self.len;
            for (0..self.len + 1) |i| {
                if (!self.isLeaf()) {
                    len += self.edges[i].?.count();
                }
            }
            return len;
        }

        // Searches the Node for a key.
        pub fn search(self: Self, key: K) std.meta.Tuple(&.{ bool, usize }) {
            var i: usize = 0;
            while (i < self.len) : (i += 1) {
                if (eql(key, self.keys[i])) {
                    return .{ true, i };
                } else if (lt(key, self.keys[i])) {
                    return .{ false, i };
                }
            }
            return .{ false, self.len };
        }

        pub fn insertOrSplit(
            self: *Self,
            allocator: std.mem.Allocator,
            index: usize,
            key: K,
            value: V,
            edge: ?*Self,
        ) !?KVE {
            if (self.isFull()) {
                var split_result = try self.split(allocator);
                switch (index < B) {
                    true => self.insert(index, key, value, edge),
                    false => split_result.edge.?.insert(index - B, key, value, edge),
                }
                return split_result;
            }
            self.insert(index, key, value, edge);
            return null;
        }

        pub fn swapValue(self: *Self, index: usize, value: V) V {
            const out = self.values[index];
            self.values[index] = value;
            return out;
        }

        pub fn swapKeyValue(self: *Self, index: usize, key: K, value: V) KV {
            const out = .{ .key = self.keys[index], .value = self.values[index] };
            self.values[index] = value;
            self.keys[index] = key;
            return out;
        }

        pub fn orderedRemove(self: *Self, index: usize) KVE {
            const out: KVE = .{
                .key = self.keys[index],
                .value = self.values[index],
                .edge = self.edges[index + 1],
            };
            std.mem.copyForwards(K, self.keys[index..], self.keys[index + 1 .. self.len]);
            std.mem.copyForwards(V, self.values[index..], self.values[index + 1 .. self.len]);
            self.keys[self.len - 1] = undefined;
            self.values[self.len - 1] = undefined;
            if (!self.isLeaf()) {
                std.mem.copyForwards(?*Self, self.edges[index + 1 ..], self.edges[index + 2 .. self.len + 1]);
                self.edges[self.len] = null;
            }
            self.len -= 1;
            return out;
        }

        fn pop(self: *Self) KVE {
            return self.orderedRemove(self.len - 1);
        }

        fn shift(self: *Self) KVE {
            const out: KVE = .{
                .key = self.keys[0],
                .value = self.values[0],
                .edge = self.edges[0],
            };
            std.mem.copyForwards(K, self.keys[0..], self.keys[1..self.len]);
            std.mem.copyForwards(V, self.values[0..], self.values[1..self.len]);
            self.keys[self.len - 1] = undefined;
            self.values[self.len - 1] = undefined;
            if (!self.isLeaf()) {
                std.mem.copyForwards(
                    ?*Self,
                    self.edges[0..],
                    self.edges[1 .. self.len + 1],
                );
                self.edges[self.len] = null;
            }
            self.len -= 1;
            return out;
        }

        fn insert(self: *Self, index: usize, key: K, value: V, edge: ?*Self) void {
            std.mem.copyBackwards(
                K,
                self.keys[index + 1 .. self.len + 1],
                self.keys[index..self.len],
            );
            self.keys[index] = key;
            std.mem.copyBackwards(V, self.values[index + 1 .. self.len + 1], self.values[index..self.len]);
            self.values[index] = value;
            if (!self.isLeaf()) {
                std.mem.copyBackwards(?*Self, self.edges[index + 2 .. self.len + 2], self.edges[index + 1 .. self.len + 1]);
                self.edges[index + 1] = edge;
            }
            self.len += 1;
        }

        fn append(self: *Self, key: K, value: V, edge: ?*Self) void {
            self.keys[self.len] = key;
            self.values[self.len] = value;
            self.edges[self.len + 1] = edge;
            self.len += 1;
        }

        fn unshift(self: *Self, key: K, value: V, edge: ?*Self) void {
            std.mem.copyBackwards(K, self.keys[1 .. self.len + 1], self.keys[0..self.len]);
            self.keys[0] = key;
            std.mem.copyBackwards(V, self.values[1 .. self.len + 1], self.values[0..self.len]);
            self.values[0] = value;
            if (!self.isLeaf()) {
                std.mem.copyBackwards(?*Self, self.edges[1 .. self.len + 2], self.edges[0 .. self.len + 1]);
                self.edges[0] = edge;
            }
            self.len += 1;
        }

        pub fn borrowRight(self: *Self, index: usize) bool {
            if (index == self.len) return false;
            var from = self.edges[index + 1].?;
            if (from.len > B - 1) {
                var to = self.edges[index].?;
                const borrowed = from.shift();
                to.append(self.keys[index], self.values[index], borrowed.edge);
                _ = self.swapKeyValue(index, borrowed.key, borrowed.value);
                return true;
            }
            return false;
        }

        pub fn borrowLeft(self: *Self, index: usize) bool {
            if (index == 0) return false;
            var from = self.edges[index - 1].?;
            if (from.len > B - 1) {
                var to = self.edges[index].?;
                const borrowed = from.pop();
                to.unshift(self.keys[index - 1], self.values[index - 1], borrowed.edge);
                _ = self.swapKeyValue(index - 1, borrowed.key, borrowed.value);
                return true;
            }
            return false;
        }

        pub fn mergeEdges(self: *Self, allocator: std.mem.Allocator, left_edge_index: usize) void {
            var left = self.edges[left_edge_index].?;
            const removed = self.orderedRemove(left_edge_index);
            left.append(removed.key, removed.value, null);
            std.mem.copyBackwards(K, left.keys[left.len..], removed.edge.?.keys[0..removed.edge.?.len]);
            std.mem.copyBackwards(V, left.values[left.len..], removed.edge.?.values[0..removed.edge.?.len]);
            std.mem.copyBackwards(?*Self, left.edges[left.len..], removed.edge.?.edges[0 .. removed.edge.?.len + 1]);
            left.len += removed.edge.?.len;
            allocator.destroy(removed.edge.?);
        }

        fn split(self: *Self, allocator: std.mem.Allocator) !KVE {
            const median = B - 1;
            const new_key = self.keys[median];
            const new_value = self.values[median];
            const new_node = try Self.initFromSplit(
                allocator,
                self.keys[median + 1 .. self.len],
                self.values[median + 1 .. self.len],
                self.edges[median + 1 .. self.len + 1],
            );
            @memset(self.keys[median..], undefined);
            @memset(self.values[median..], undefined);
            @memset(self.edges[median + 1 ..], null);
            self.len = median;
            return .{ .key = new_key, .value = new_value, .edge = new_node };
        }

        pub fn isLeaf(self: Self) bool {
            return self.edges[0] == null;
        }

        pub fn isFull(self: Self) bool {
            return self.len == 2 * B - 1;
        }

        pub fn isLacking(self: Self) bool {
            return self.len < B - 1;
        }
    };
}

pub fn BTreeMap(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();

        const B = 6;
        const Node = NodeType(K, V, B);
        const KV = Node.KV;
        const SearchResult = std.meta.Tuple(&.{ bool, usize });
        const StackEntry = struct { node: *Node, index: usize };

        allocator: std.mem.Allocator,
        root: ?*Node = null,

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .allocator = allocator };
        }

        pub fn deinit(self: Self) !void {
            if (self.root == null) return;
            var stack = std.ArrayList(*Node).init(self.allocator);
            defer stack.deinit();
            if (self.root) |root| {
                try stack.append(root);
            }
            while (stack.popOrNull()) |node| {
                if (!node.isLeaf()) {
                    for (0..node.len + 1) |i| {
                        try stack.append(node.edges[i].?);
                    }
                }
                self.allocator.destroy(node);
            }
        }

        pub fn count(self: Self) usize {
            if (self.root == null) return 0;
            var len: usize = 0;
            if (self.root) |node| {
                len += node.count();
            }
            return len;
        }

        pub fn isEmpty(self: *const Self) bool {
            if (self.root == null) return true;
            return self.root.?.len == 0;
        }

        pub fn get(self: Self, key: K) ?V {
            var current = self.root;
            while (current) |node| {
                const found, const index = node.search(key);
                switch (found) {
                    true => return node.values[index],
                    false => current = node.edges[index],
                }
            }
            return null;
        }

        pub fn getPtr(self: Self, key: K) ?*V {
            var current = self.root;
            while (current) |node| {
                const found, const index = node.search(key);
                switch (found) {
                    true => return &node.values[index],
                    false => current = node.edges[index],
                }
            }
            return null;
        }

        pub fn fetchPut(self: *Self, key: K, value: V) !?KV {
            if (self.root == null) {
                self.root = try Node.initKeyValue(self.allocator, .{ key, value });
                return null;
            }
            var stack = std.ArrayList(StackEntry).init(self.allocator);
            defer stack.deinit();
            var current = self.root;
            var search_result: SearchResult = undefined;
            while (current) |node| {
                search_result = node.search(key);
                if (search_result[0]) {
                    return .{ .key = key, .value = node.swapValue(search_result[1], value) };
                }
                current = node.edges[search_result[1]];
                try stack.append(.{ .node = node, .index = search_result[1] });
            }
            var stack_next: ?StackEntry = stack.pop();
            var split_result = try stack_next.?.node.insertOrSplit(
                self.allocator,
                stack_next.?.index,
                key,
                value,
                null,
            );
            if (split_result == null) {
                return null;
            }
            stack_next = stack.popOrNull();
            while (split_result) |split_result_unwrapped| {
                if (stack_next) |stack_next_unwrapped| {
                    split_result = try stack_next_unwrapped.node.insertOrSplit(
                        self.allocator,
                        stack_next_unwrapped.index,
                        split_result_unwrapped.key,
                        split_result_unwrapped.value,
                        split_result_unwrapped.edge,
                    );
                    stack_next = stack.popOrNull();
                } else {
                    var new_root = try Node.initKeyValue(
                        self.allocator,
                        .{ split_result_unwrapped.key, split_result_unwrapped.value },
                    );
                    new_root.edges[0] = self.root;
                    new_root.edges[1] = split_result_unwrapped.edge;
                    self.root = new_root;
                    return null;
                }
            } else return null;
        }

        pub fn fetchRemove(self: *Self, key: K) !?KV {
            var stack = std.ArrayList(StackEntry).init(self.allocator);
            defer stack.deinit();
            var current = self.root;
            var search_result: SearchResult = undefined;
            var found_key_ptr: ?*K = null;
            var found_value_ptr: ?*V = null;
            while (current) |node| {
                search_result = node.search(key);
                if (search_result[0]) {
                    found_key_ptr = &node.keys[search_result[1]];
                    found_value_ptr = &node.values[search_result[1]];
                    if (!node.isLeaf()) search_result[1] += 1;
                }
                try stack.append(.{
                    .node = node,
                    .index = search_result[1],
                });
                current = node.edges[search_result[1]];
                if (search_result[0]) break;
            } else return null;
            while (current) |node| {
                try stack.append(.{ .node = node, .index = 0 });
                current = node.edges[0];
            }
            var current_stack = stack.pop();
            const out: KV = .{ .key = found_key_ptr.?.*, .value = found_value_ptr.?.* };
            found_key_ptr.?.* = current_stack.node.keys[current_stack.index];
            found_value_ptr.?.* = current_stack.node.values[current_stack.index];
            _ = current_stack.node.orderedRemove(current_stack.index);
            if (current_stack.node == self.root) return out;
            while (current_stack.node.isLacking()) {
                current_stack = stack.pop();
                if (current_stack.node.borrowRight(current_stack.index)) return out;
                if (current_stack.node.borrowLeft(current_stack.index)) return out;
                if (current_stack.index == current_stack.node.len) {
                    current_stack.node.mergeEdges(self.allocator, current_stack.index - 1);
                } else {
                    current_stack.node.mergeEdges(self.allocator, current_stack.index);
                }
                if (current_stack.node == self.root) {
                    if (self.root.?.len == 0) {
                        const new_root = current_stack.node.edges[0].?;
                        self.allocator.destroy(self.root.?);
                        self.root.? = new_root;
                    }
                    break;
                }
            }
            return out;
        }

        const Iterator = struct {
            stack: std.ArrayList(StackEntry),
            backwards: bool,

            pub fn deinit(it: Iterator) void {
                it.stack.deinit();
            }

            pub fn next(it: *Iterator) ?Node.Entry {
                while (it.topStackItem()) |item| {
                    if (!item.node.isLeaf() and !it.backwards) {
                        const child = item.node.edges[item.index].?;
                        it.stack.append(StackEntry{ .node = child, .index = 0 }) catch unreachable;
                    } else {
                        if (item.index < item.node.len) {
                            const out: Node.Entry = .{ .key_ptr = &item.node.keys[item.index], .value_ptr = &item.node.values[item.index] };
                            item.index += 1;
                            it.backwards = false;
                            return out;
                        } else {
                            _ = it.stack.popOrNull();
                            it.backwards = true;
                        }
                    }
                } else return null;
            }

            fn topStackItem(it: *Iterator) ?*StackEntry {
                return switch (it.stack.items.len) {
                    0 => null,
                    else => &it.stack.items[it.stack.items.len - 1],
                };
            }
        };

        pub fn iterator(self: *const Self) Iterator {
            var new_stack = std.ArrayList(StackEntry).init(self.allocator);
            if (self.root) |root| {
                new_stack.append(.{ .node = root, .index = 0 }) catch unreachable;
            }
            return Iterator{
                .stack = new_stack,
                .backwards = false,
            };
        }
    };
}

/// Compares two of any type for equality. Containers are compared on a field-by-field basis,
/// where possible. Pointers are followed if the addresses are not equal.
fn eql(a: anytype, b: @TypeOf(a)) bool {
    const T = @TypeOf(a);
    switch (@typeInfo(T)) {
        .Struct => |info| {
            inline for (info.fields) |field_info| {
                if (!eql(@field(a, field_info.name), @field(b, field_info.name))) return false;
            }
            return true;
        },
        .ErrorUnion => {
            if (a) |a_p| {
                if (b) |b_p| return eql(a_p, b_p) else |_| return false;
            } else |a_e| {
                if (b) |_| return false else |b_e| return a_e == b_e;
            }
        },
        .Union => |info| {
            if (info.tag_type) |UnionTag| {
                const tag_a = std.meta.activeTag(a);
                const tag_b = std.meta.activeTag(b);
                if (tag_a != tag_b) return false;

                inline for (info.fields) |field_info| {
                    if (@field(UnionTag, field_info.name) == tag_a) {
                        return eql(@field(a, field_info.name), @field(b, field_info.name));
                    }
                }
                return false;
            }

            @compileError("Cannot compare untagged union type " ++ @typeName(T));
        },
        .Array => {
            if (a.len != b.len) return false;
            for (a, 0..) |e, i|
                if (!eql(e, b[i])) return false;
            return true;
        },
        .Vector => |info| {
            var i: usize = 0;
            while (i < info.len) : (i += 1) {
                if (!eql(a[i], b[i])) return false;
            }
            return true;
        },
        .Pointer => |info| {
            return switch (info.size) {
                .One => if (a == b) true else eql(a.*, b.*),
                .Many => if (a == b) true else {
                    if (info.sentinel) {
                        if (std.mem.len(a) != std.mem.len(b)) return false;
                        var i: usize = 0;
                        while (i < std.mem.len(a)) : (i += 1)
                            if (!eql(a[i], b[i])) return false;
                        return true;
                    }
                    @compileError("Cannot compare many-item Pointers without sentinel value");
                },
                .C => if (a == b) true else @compileError("Cannot compare C pointers"),
                .Slice => if (a.ptr == b.ptr and a.len == b.len) true else {
                    if (a.len != b.len) return false;
                    for (a, 0..) |_, i|
                        if (!eql(a[i], b[i])) return false;
                    return true;
                },
            };
        },
        .Optional => {
            if (a == null and b == null) return true;
            if (a == null or b == null) return false;
            return eql(a.?, b.?);
        },
        else => return a == b,
    }
}

fn lt(a: anytype, b: @TypeOf(a)) bool {
    const T = @TypeOf(a);

    switch (@typeInfo(T)) {
        .Int, .ComptimeInt, .Float, .ComptimeFloat => {
            return a < b;
        },
        .Struct => {
            if (!@hasDecl(T, "lt")) {
                @compileError("Type `" ++ @typeName(T) ++ "` must implement a `lt` comparison method.");
            }
            return T.lt(a, b);
        },
        .Union => |info| {
            if (info.tag_type) |UnionTag| {
                const tag_a = std.meta.activeTag(a);
                const tag_b = std.meta.activeTag(b);
                // if tags are not equal, perform comparison based on tag
                if (tag_a != tag_b) {
                    return std.ascii.lessThanIgnoreCase(@tagName(tag_a), @tagName(tag_b));
                }
                // if tags are equal, compare based on the active field
                inline for (info.fields) |field_info| {
                    if (@field(UnionTag, field_info.name) == tag_a) {
                        return lt(@field(a, field_info.name), @field(b, field_info.name));
                    }
                }
                return false;
            }

            @compileError("Cannot perform `lt` check on untagged union type " ++ @typeName(T));
        },
        .Array => {
            for (a, 0..) |_, i| {
                if (lt(a[i], b[i])) {
                    return true;
                } else if (eql(a[i], b[i])) {
                    continue;
                } else {
                    return false;
                }
            }
            return false;
        },
        .Vector => |info| {
            var i: usize = 0;
            while (i < info.len) : (i += 1) {
                if (lt(a[i], b[i])) {
                    return true;
                } else if (eql(a[i], b[i])) {
                    continue;
                } else {
                    return false;
                }
            }
            return false;
        },
        .Pointer => |info| {
            switch (info.size) {
                .One => return lt(a.*, b.*),
                .Slice => {
                    const n = @min(a.len, b.len);
                    for (a[0..n], 0..) |_, i| {
                        if (lt(a[i], b[i])) {
                            return true;
                        } else if (eql(a[i], b[i])) {
                            continue;
                        } else {
                            return false;
                        }
                    }
                    return lt(a.len, b.len);
                },
                .Many => {
                    if (info.sentinel) {
                        const n = @min(std.mem.len(a), std.mem.len(b));
                        var i: usize = 0;
                        while (i < n) : (i += 1) {
                            if (lt(a[i], b[i])) {
                                return true;
                            } else if (eql(a[i], b[i])) {
                                continue;
                            } else {
                                return false;
                            }
                        }
                        return lt(std.mem.len(a), std.mem.len(b));
                    }
                    @compileError("Cannot compare many-item pointer to unknown number of items without sentinel value");
                },
                .C => @compileError("Cannot compare C pointers"),
            }
        },
        .Optional => {
            if (a == null or b == null) return false;
            return lt(a.?, b.?);
        },
        else => {
            @compileError("Cannot compare type '" ++ @typeName(T) ++ "'");
        },
    }
}

pub fn gt(a: anytype, b: @TypeOf(a)) bool {
    return !lt(a, b) and !eql(a, b);
}
