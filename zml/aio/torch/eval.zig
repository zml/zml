const std = @import("std");
const stdx = @import("stdx");

const py = @import("py.zig");
const pickle = @import("pickle.zig");

const MAX_DEPTH: usize = 250;
const MAX_PROTOCOL: u8 = 5;

pub const PickleMemo = struct {
    map: std.AutoHashMap(u32, py.Any),

    pub fn init(allocator: std.mem.Allocator) PickleMemo {
        return .{
            .map = std.AutoHashMap(u32, py.Any).init(allocator),
        };
    }

    pub fn deinit(self: *PickleMemo) void {
        const allocator = self.map.allocator;
        var iterator = self.map.iterator();
        defer iterator.deinit();
        while (iterator.next()) |entry| {
            entry.value_ptr.deinit(allocator);
        }
        self.map.deinit() catch unreachable;
        self.* = undefined;
    }

    pub fn resolve(self: *PickleMemo, allocator: std.mem.Allocator, op: py.Any, recursive: bool) !py.Any {
        var used_op = op;
        while (used_op == .ref) {
            var count: usize = 0;
            const val = self.map.get(op.ref) orelse {
                return error.BadMemoRef;
            };
            if (!recursive) {
                return val.clone(allocator);
            }

            count += 1;
            if (count >= MAX_DEPTH or val != .ref) {
                used_op = try val.clone(allocator);
                break;
            }
            used_op = val;
        }
        if (used_op.containsRef()) {
            switch (used_op) {
                .app, .object, .global => |v| {
                    if (v.member.containsRef()) {
                        v.member = try self.resolve(allocator, v.member, recursive);
                    }
                    for (v.args) |*item| {
                        if (item.containsRef()) {
                            item.* = try self.resolve(allocator, item.*, recursive);
                        }
                    }
                },
                .set_state => |v| {
                    if (v.obj.containsRef()) {
                        v.obj = try self.resolve(allocator, v.obj, recursive);
                    }
                    if (v.state.containsRef()) {
                        v.state = try self.resolve(allocator, v.state, recursive);
                    }
                },
                .pers_id => |v| {
                    if (v.ref.containsRef()) {
                        v.ref = try self.resolve(allocator, v.ref, recursive);
                    }
                },
                .seq => |*v| {
                    for (v.values) |*item| {
                        if (item.containsRef()) {
                            item.* = try self.resolve(allocator, item.*, recursive);
                        }
                    }
                },
                else => {},
            }
        }
        return used_op;
    }

    pub fn insert(self: *PickleMemo, mid: u32, val: py.Any) !void {
        _ = try self.map.fetchPut(mid, val);
    }

    pub fn resolveMut(self: *PickleMemo, op: *py.Any, recursive: bool) !*py.Any {
        if (op.* != .ref) return op;
        var lastmid = op.ref;
        var count: usize = 0;
        var val = self.map.get(lastmid) orelse {
            return error.BadMemoRef;
        };
        while (val == .ref) {
            lastmid = val.ref;
            if (!recursive) {
                break;
            }
            count += 1;
            if (count >= MAX_DEPTH) {
                break;
            }
            val = self.map.get(lastmid) orelse {
                return error.BadMemoRef;
            };
        }
        return (self.map.getPtr(lastmid) orelse {
            return error.BadMemoRef;
        });
    }

    const MemoError = py.Any.UnpickleError || error{BadMemoRef};

    pub fn resolveAllRefsIter(self: *PickleMemo, allocator: std.mem.Allocator, depth: usize, vals: []py.Any, fix_values: bool) MemoError![]py.Any {
        if (depth >= MAX_DEPTH) {
            return vals;
        }
        const res = try allocator.alloc(py.Any, vals.len);
        for (vals, 0..) |v, i| {
            res[i] = try self.resolveAllRefs(allocator, depth + 1, v, fix_values);
        }
        return res;
    }

    pub fn resolveAllRefs(self: *PickleMemo, allocator: std.mem.Allocator, depth: usize, val: py.Any, fix_values: bool) !py.Any {
        var output: py.Any = switch (val) {
            .ref => try self.resolve(allocator, val, true),
            inline .app, .object, .global => |v, tag| @unionInit(py.Any, @tagName(tag), try py.Object.init(
                allocator,
                try self.resolveAllRefs(allocator, depth + 1, v.member, fix_values),
                try self.resolveAllRefsIter(allocator, depth + 1, v.args, fix_values),
                try self.resolveAllRefsIter(allocator, depth + 1, v.kwargs, fix_values),
            )),
            .set_state => |v| .{ .set_state = try py.SetState.init(
                allocator,
                try self.resolveAllRefs(allocator, depth + 1, v.obj, fix_values),
                try self.resolveAllRefs(allocator, depth + 1, v.state, fix_values),
            ) },
            .seq => |v| .{ .seq = .{ .type = v.type, .values = try self.resolveAllRefsIter(allocator, depth + 1, v.values, fix_values) } },
            .pers_id => |v| .{ .pers_id = try py.PersId.init(allocator, try self.resolveAllRefs(allocator, depth + 1, v.ref, fix_values)) },
            else => try val.clone(allocator),
        };
        if (fix_values) {
            output = try output.coerceFromRaw(allocator);
        }
        return output;
    }
};

pub fn evaluate(arena: std.mem.Allocator, x: []const pickle.Op, resolve_refs: bool) ![]const py.Any {
    var stack = std.ArrayList(py.Any).init(arena);
    var memo = PickleMemo.init(arena);

    for (x) |op| {
        switch (op) {
            .mark => try stack.append(.{ .raw = op }),
            .frame => {},
            .stop => break,
            .pop => _ = try pop(&stack),
            .pop_mark => _ = try popMark(&stack),
            .dup => if (stack.getLastOrNull()) |item|
                try stack.append(try item.clone(arena))
            else
                return error.CannotDupEmptyStack,
            .persid => |v| try stack.append(.{ .pers_id = try py.PersId.init(arena, .{ .string = try arena.dupe(u8, v) }) }),
            .binpersid => try stack.append(.{ .pers_id = try py.PersId.init(arena, try pop(&stack)) }),
            .reduce => try stack.append(.{ .global = blk: {
                var args = try pop(&stack);
                args = try memo.resolve(arena, args, true);
                if (args != .seq) return error.InvalidInput;
                var func = try pop(&stack);
                func = try memo.resolve(arena, func, true);
                break :blk try py.Object.init(arena, func, args.seq.values, &.{});
            } }),
            .build => try stack.append(blk: {
                const args = try memo.resolve(arena, try pop(&stack), true);
                const member = try memo.resolve(arena, try pop(&stack), true);
                break :blk .{ .set_state = try py.SetState.init(arena, member, args) };
            }),
            .empty_dict => try stack.append(.{ .seq = .{ .type = .dict, .values = &[_]py.Any{} } }),
            .get => |v| try stack.append(.{ .ref = v }),
            .empty_list => try stack.append(.{ .seq = .{ .type = .list, .values = &[_]py.Any{} } }),
            .put => |v| {
                try memo.insert(v, try pop(&stack));
                try stack.append(.{ .ref = v });
            },
            .tuple => try stack.append(blk: {
                const popped = try popMark(&stack);
                break :blk .{ .seq = .{ .type = .tuple, .values = try arena.dupe(py.Any, popped) } };
            }),
            .empty_tuple => try stack.append(.{ .seq = .{ .type = .tuple, .values = &[_]py.Any{} } }),
            .setitem => {
                const v = try memo.resolve(arena, try pop(&stack), true);
                const k = try memo.resolve(arena, try pop(&stack), true);
                const top = try lastMut(&stack);
                const rtop = try memo.resolveMut(top, true);
                switch (rtop.*) {
                    .global => |obj| {
                        try append(arena, &obj.kwargs, &.{ k, v });
                    },
                    .seq => |*dict| {
                        if (dict.type != .dict) return error.BadStackTopForSetItem;
                        try append(arena, &dict.values, &.{ k, v });
                    },
                    else => {
                        return error.BadStackTopForSetItem;
                    },
                }
            },
            .setitems => {
                const popped = try memo.resolveAllRefsIter(arena, 0, try popMark(&stack), true);
                const top = try lastMut(&stack);
                const rtop = try memo.resolveMut(top, true);
                switch (rtop.*) {
                    .global => |obj| {
                        try append(arena, &obj.kwargs, popped);
                    },
                    .seq => |*dict| {
                        if (dict.type != .dict) return error.BadStackTopForSetItems;
                        try append(arena, &dict.values, popped);
                    },
                    else => {
                        return error.BadStackTopForSetItems;
                    },
                }
            },
            .proto => |proto| stdx.debug.assert(proto <= MAX_PROTOCOL, "Unsupported protocol {d}", .{proto}),
            .tuple1 => try stack.append(blk: {
                const tup_values = try arena.alloc(py.Any, 1);
                tup_values[0] = try pop(&stack);
                break :blk .{ .seq = .{ .type = .tuple, .values = tup_values } };
            }),
            .tuple2 => try stack.append(blk: {
                const tup_values = try arena.alloc(py.Any, 2);
                inline for (0..2) |i| tup_values[(tup_values.len - 1) - i] = try pop(&stack);
                break :blk .{ .seq = .{ .type = .tuple, .values = tup_values } };
            }),
            .tuple3 => try stack.append(blk: {
                const tup_values = try arena.alloc(py.Any, 3);
                inline for (0..3) |i| tup_values[(tup_values.len - 1) - i] = try pop(&stack);
                break :blk .{ .seq = .{ .type = .tuple, .values = tup_values } };
            }),
            .append => {
                const v = try pop(&stack);
                const top = try lastMut(&stack);
                const rtop = try memo.resolveMut(top, true);
                switch (rtop.*) {
                    .global => |obj| {
                        // can this happen ?
                        try append(arena, &obj.args, &.{v});
                    },
                    .seq => |*seq| {
                        if (seq.type != .list) return error.BadStackTopForAppend;
                        try append(arena, &seq.values, &.{v});
                    },
                    else => {
                        return error.BadStackTopForAppend;
                    },
                }
            },
            .appends => {
                const postmark = try popMark(&stack);
                const top = try lastMut(&stack);
                const rtop = try memo.resolveMut(top, true);
                switch (rtop.*) {
                    .global => try append(arena, &rtop.global.args, postmark),
                    .seq => |*seq| {
                        if (seq.type != .list) return error.BadStackTopForAppend;
                        try append(arena, &seq.values, postmark);
                    },
                    else => {
                        return error.BadStackTopForAppends;
                    },
                }
            },
            .dict => try stack.append(.{ .seq = .{
                .type = .dict,
                .values = try arena.dupe(py.Any, try popMark(&stack)),
            } }),
            .list => try stack.append(.{ .seq = .{
                .type = .list,
                .values = try arena.dupe(py.Any, try popMark(&stack)),
            } }),
            .inst => |v| try stack.append(.{ .object = try py.Object.init(
                arena,
                try py.tuple(&.{ .{ .string = v.module }, .{ .string = v.class } }).clone(arena),
                try arena.dupe(py.Any, try popMark(&stack)),
                &.{},
            ) }),
            .obj => try stack.append(blk: {
                const mark = try findMark(&stack);
                const args = try arena.dupe(py.Any, stack.items[mark + 2 ..]);
                const member = stack.items[mark + 1];
                break :blk .{ .object = try py.Object.init(arena, member, args, &.{}) };
            }),
            .newobj => try stack.append(blk: {
                const args = try arena.alloc(py.Any, 1);
                args[0] = try pop(&stack);
                break :blk .{ .object = try py.Object.init(arena, try pop(&stack), args, &.{}) };
            }),
            .empty_set => try stack.append(.{ .seq = .{ .type = .set, .values = &[_]py.Any{} } }),
            .additems => {
                const postmark = try popMark(&stack);
                const top = try lastMut(&stack);
                const rtop = try memo.resolveMut(top, true);
                switch (rtop.*) {
                    .seq => |*seq| {
                        if (seq.type != .set) return error.BadStackTopForAppend;
                        try append(arena, &seq.values, postmark);
                    },
                    else => {
                        return error.BadStackTopForAppends;
                    },
                }
            },
            .frozenset => try stack.append(.{ .seq = .{
                .type = .frozen_set,
                .values = try arena.dupe(py.Any, try popMark(&stack)),
            } }),
            .newobj_ex => try stack.append(blk: {
                const kwargs, const args, const cls = .{ try pop(&stack), try pop(&stack), try pop(&stack) };
                break :blk .{ .object = try py.Object.init(arena, cls, args.seq.values, kwargs.seq.values) };
            }),
            .stack_global => try stack.append(blk: {
                const gn, const mn = .{
                    try memo.resolve(arena, try pop(&stack), true),
                    try memo.resolve(arena, try pop(&stack), true),
                };
                const new_seq: py.Sequence = .{ .type = .tuple, .values = try arena.dupe(py.Any, &.{ gn, mn }) };
                break :blk .{ .object = try py.Object.init(arena, .{ .seq = new_seq }, &.{}, &.{}) };
            }),
            .memoize => {
                const item = stack.getLastOrNull() orelse {
                    return error.StackUnderrun;
                };
                try memo.insert(@intCast(memo.map.count()), try item.clone(arena));
            },
            else => try stack.append(.{ .raw = try op.clone(arena) }),
        }
    }
    if (resolve_refs) {
        return try memo.resolveAllRefsIter(arena, 0, stack.items, true);
    }
    return stack.toOwnedSlice();
}

fn append(allocator: std.mem.Allocator, current: *[]py.Any, values: []const py.Any) !void {
    var array_list = std.ArrayListUnmanaged(py.Any).fromOwnedSlice(current.*);
    try array_list.appendSlice(allocator, values);
    current.* = array_list.items;
}

test evaluate {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const file = try std.fs.cwd().openFile("zml/aio/torch/simple_test_4.pickle", .{ .mode = .read_only });
    var buffered_reader = std.io.bufferedReader(file.reader());
    const ops = try pickle.parse(allocator, buffered_reader.reader(), 4096);

    const vals = try evaluate(allocator, ops, true);
    defer allocator.free(vals);

    try std.testing.expect(vals.len == 1);
    try std.testing.expect(vals[0] == .seq);
    try std.testing.expect(vals[0].seq.type == .dict);
    const entries = vals[0].seq.values;
    const expected: []const py.Any = &.{
        // Key, followed by its value
        .{ .string = "hello" }, .{ .string = "world" },
        .{ .string = "int" },   .{ .int64 = 1 },
        .{ .string = "float" }, .{ .float64 = 3.141592 },
        .{ .string = "list" },
        .{
            .seq = .{
                .type = .list,
                .values = @constCast(&[_]py.Any{
                    .{ .int64 = 255 },
                    .{ .int64 = 1234 },
                    .{ .int64 = -123 },
                    .{ .int64 = 1_000_000_000 },
                    .{ .int64 = 999_000_000_000 },
                    .{ .bigint = (try std.math.big.int.Managed.initSet(allocator, 999_000_000_000_000_000_000_000_000_000)).toConst() },
                }),
            },
        },
        .{ .string = "bool" },  .{ .boolval = false },
        .{ .string = "tuple" },
        .{ .seq = .{
            .type = .tuple,
            .values = @constCast(&[_]py.Any{
                .{ .string = "a" },
                .{ .int64 = 10 },
            }),
        } },
    };

    try std.testing.expectEqualDeep(expected, entries);
}

pub fn pop(values: *std.ArrayList(py.Any)) error{StackUnderrun}!py.Any {
    if (values.items.len == 0) {
        return error.StackUnderrun;
    }
    return values.pop().?;
}

fn popMark(values: *std.ArrayList(py.Any)) ![]py.Any {
    const mark = try findMark(values);
    const popping = values.items[mark + 1 ..];
    values.shrinkRetainingCapacity(mark);
    return popping;
}

fn lastMut(values: *std.ArrayList(py.Any)) !*py.Any {
    if (values.items.len == 0) {
        return error.UnexpectedEmptyStack;
    }
    return &values.items[values.items.len - 1];
}

fn findMark(values: *std.ArrayList(py.Any)) !usize {
    const len = values.items.len;
    for (0..len) |i| {
        const idx = (len - 1) - i;
        const val = values.items[idx];
        if (val == .raw and val.raw == .mark) {
            return idx;
        }
    }
    return error.MarkNotFound;
}
