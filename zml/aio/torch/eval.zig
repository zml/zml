const std = @import("std");
const zml = @import("../../zml.zig");
const meta = zml.meta;

const value = @import("value.zig");
const pickle = @import("pickle.zig");
const BTreeMap = @import("b_tree_map.zig").BTreeMap;

const Build = value.Build;
const Object = value.Object;
const PersId = value.PersId;
const Sequence = value.Sequence;
const SequenceType = value.SequenceType;
const Value = value.Value;

const MAX_DEPTH: usize = 250;
const MAX_PROTOCOL: u8 = 5;

pub const PickleMemo = struct {
    allocator: std.mem.Allocator,
    map: BTreeMap(u32, Value),

    pub fn init(allocator: std.mem.Allocator) PickleMemo {
        return .{
            .allocator = allocator,
            .map = BTreeMap(u32, Value).init(allocator),
        };
    }

    pub fn deinit(self: *PickleMemo) void {
        var iterator = self.map.iterator();
        defer iterator.deinit();
        while (iterator.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.map.deinit() catch unreachable;
        self.* = undefined;
    }

    pub fn resolve(self: *PickleMemo, allocator: std.mem.Allocator, op: Value, recursive: bool) !Value {
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
                .build => |v| {
                    if (v.member.containsRef()) {
                        v.member = try self.resolve(allocator, v.member, recursive);
                    }
                    if (v.args.containsRef()) {
                        v.args = try self.resolve(allocator, v.args, recursive);
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

    pub fn insert(self: *PickleMemo, mid: u32, val: Value) !void {
        _ = try self.map.fetchPut(mid, val);
    }

    pub fn resolveMut(self: *PickleMemo, op: *Value, recursive: bool) !*Value {
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

    const MemoError = std.math.big.int.Managed.ConvertError || std.mem.Allocator.Error || error{BadMemoRef};

    pub fn resolveAllRefsIter(self: *PickleMemo, allocator: std.mem.Allocator, depth: usize, vals: []Value, fix_values: bool) MemoError![]Value {
        if (depth >= MAX_DEPTH) {
            return vals;
        }
        const res = try allocator.alloc(Value, vals.len);
        for (vals, 0..) |v, i| {
            res[i] = try self.resolveAllRefs(allocator, depth + 1, v, fix_values);
        }
        return res;
    }

    pub fn resolveAllRefs(self: *PickleMemo, allocator: std.mem.Allocator, depth: usize, val: Value, fix_values: bool) !Value {
        var output: Value = switch (val) {
            .ref => try self.resolve(allocator, val, true),
            inline .app, .object, .global => |v, tag| @unionInit(Value, @tagName(tag), try Object.init(
                allocator,
                try self.resolveAllRefs(allocator, depth + 1, v.member, fix_values),
                try self.resolveAllRefsIter(allocator, depth + 1, v.args, fix_values),
            )),
            .build => |v| .{ .build = try Build.init(
                allocator,
                try self.resolveAllRefs(allocator, depth + 1, v.member, fix_values),
                try self.resolveAllRefs(allocator, depth + 1, v.args, fix_values),
            ) },
            .seq => |v| .{ .seq = .{ .type = v.type, .values = try self.resolveAllRefsIter(allocator, depth + 1, v.values, fix_values) } },
            .pers_id => |v| .{ .pers_id = try PersId.init(allocator, try self.resolveAllRefs(allocator, depth + 1, v.ref, fix_values)) },
            else => try val.clone(allocator),
        };
        if (fix_values) {
            output = try output.coerceFromRaw(allocator);
        }
        return output;
    }
};

pub fn evaluate(arena: std.mem.Allocator, x: []const pickle.Op, resolve_refs: bool) ![]const Value {
    var stack = std.ArrayList(Value).init(arena);
    var memo = PickleMemo.init(arena);
    errdefer memo.deinit();

    const makeKVList = (struct {
        pub fn call(alloc: std.mem.Allocator, items: []const Value) ![]Value {
            meta.assert(items.len & 1 == 0, "Bad value for setitems", .{});
            var kv_items = try std.ArrayList(Value).initCapacity(alloc, items.len);
            errdefer kv_items.deinit();
            var idx: usize = 0;
            while (idx < items.len) : (idx += 2) {
                if (idx + 1 >= items.len) {
                    return error.MissingValueItem;
                }
                const kv = try alloc.alloc(Value, 2);
                kv[0] = items[idx];
                kv[1] = items[idx + 1];
                kv_items.appendAssumeCapacity(.{ .seq = .{ .type = .kv_tuple, .values = kv } });
            }
            return kv_items.toOwnedSlice();
        }
    }).call;

    for (x) |op| {
        switch (op) {
            .mark => try stack.append(.{ .raw = op }),
            .frame => {},
            .stop => break,
            .pop => _ = try pop(&stack),
            .pop_mark => try popMarkDiscard(&stack),
            .dup => if (stack.getLastOrNull()) |item|
                try stack.append(try item.clone(arena))
            else
                return error.CannotDupEmptyStack,
            .persid => |v| try stack.append(.{ .pers_id = try PersId.init(arena, .{ .string = try arena.dupe(u8, v) }) }),
            .binpersid => try stack.append(.{ .pers_id = try PersId.init(arena, try pop(&stack)) }),
            .reduce => try stack.append(.{ .global = blk: {
                const values = try arena.alloc(Value, 1);
                values[0] = try memo.resolve(arena, try pop(&stack), true);
                break :blk try Object.init(arena, try memo.resolve(arena, try pop(&stack), true), values);
            } }),
            .build => try stack.append(blk: {
                const args = try memo.resolve(arena, try pop(&stack), true);
                const member = try memo.resolve(arena, try pop(&stack), true);
                break :blk .{ .build = try Build.init(arena, member, args) };
            }),
            .empty_dict => try stack.append(.{ .seq = .{ .type = .dict, .values = &[_]Value{} } }),
            .get => |v| try stack.append(.{ .ref = v }),
            .empty_list => try stack.append(.{ .seq = .{ .type = .list, .values = &[_]Value{} } }),
            .put => |v| {
                try memo.insert(v, try pop(&stack));
                try stack.append(.{ .ref = v });
            },
            .tuple => try stack.append(blk: {
                const popped = try popMark(&stack, arena);
                break :blk .{ .seq = .{ .type = .tuple, .values = popped } };
            }),
            .empty_tuple => try stack.append(.{ .seq = .{ .type = .tuple, .values = &[_]Value{} } }),
            .setitem => {
                const v, const k = .{ try pop(&stack), try pop(&stack) };
                const top = try lastMut(&stack);
                const rtop = try memo.resolveMut(top, true);
                switch (rtop.*) {
                    .global => |obj| {
                        obj.args = try assuredResize(Value, arena, obj.args, obj.args.len + 1);
                        obj.args[obj.args.len - 1] = .{ .seq = .{ .type = .tuple, .values = try arena.dupe(Value, &.{ k, v }) } };
                    },
                    .seq => |*tup| {
                        tup.values = try assuredResize(Value, arena, tup.values, tup.values.len + 1);
                        tup.values[tup.values.len - 1] = .{ .seq = .{ .type = .tuple, .values = try arena.dupe(Value, &.{ k, v }) } };
                    },
                    else => {
                        return error.BadStackTopForSetItem;
                    },
                }
            },
            .setitems => {
                const popped = try popMark(&stack, arena);
                defer arena.free(popped);
                const kv_items = try makeKVList(arena, popped);
                const top = try lastMut(&stack);
                const rtop = try memo.resolveMut(top, true);
                switch (rtop.*) {
                    .global => |obj| {
                        obj.args = try assuredResize(Value, arena, obj.args, obj.args.len + 1);
                        obj.args[obj.args.len - 1] = .{ .seq = .{ .type = .tuple, .values = kv_items } };
                    },
                    .seq => |*tup| {
                        tup.values = try assuredResize(Value, arena, tup.values, tup.values.len + 1);
                        tup.values[tup.values.len - 1] = .{ .seq = .{ .type = .tuple, .values = kv_items } };
                    },
                    else => {
                        defer arena.free(kv_items);
                        return error.BadStackTopForSetItems;
                    },
                }
            },
            .proto => |proto| meta.assert(proto <= MAX_PROTOCOL, "Unsupported protocol {d}", .{proto}),
            .tuple1 => try stack.append(blk: {
                const tup_values = try arena.alloc(Value, 1);
                tup_values[0] = try pop(&stack);
                break :blk .{ .seq = .{ .type = .tuple, .values = tup_values } };
            }),
            .tuple2 => try stack.append(blk: {
                const tup_values = try arena.alloc(Value, 2);
                inline for (0..2) |i| tup_values[(tup_values.len - 1) - i] = try pop(&stack);
                break :blk .{ .seq = .{ .type = .tuple, .values = tup_values } };
            }),
            .tuple3 => try stack.append(blk: {
                const tup_values = try arena.alloc(Value, 3);
                inline for (0..3) |i| tup_values[(tup_values.len - 1) - i] = try pop(&stack);
                break :blk .{ .seq = .{ .type = .tuple, .values = tup_values } };
            }),
            .append => {
                const v = try pop(&stack);
                const top = try lastMut(&stack);
                const rtop = try memo.resolveMut(top, true);
                switch (rtop.*) {
                    .global => |obj| {
                        obj.args = try assuredResize(Value, arena, obj.args, obj.args.len + 1);
                        obj.args[obj.args.len - 1] = v;
                    },
                    .seq => |*tup| {
                        tup.values = try assuredResize(Value, arena, tup.values, tup.values.len + 1);
                        tup.values[tup.values.len - 1] = v;
                    },
                    else => {
                        return error.BadStackTopForAppend;
                    },
                }
            },
            .appends => {
                const postmark = try popMark(&stack, arena);
                defer arena.free(postmark);
                const top = try lastMut(&stack);
                const rtop = try memo.resolveMut(top, true);
                switch (rtop.*) {
                    .global => |obj| {
                        const obj_len = obj.args.len;
                        obj.args = try assuredResize(Value, arena, obj.args, obj_len + postmark.len);
                        @memcpy(obj.args[obj_len..], postmark);
                    },
                    .seq => |*tup| {
                        const tup_len = tup.values.len;
                        tup.values = try assuredResize(Value, arena, tup.values, tup_len + postmark.len);
                        @memcpy(tup.values[tup_len..], postmark);
                    },
                    else => {
                        return error.BadStackTopForAppends;
                    },
                }
            },
            .dict => try stack.append(blk: {
                const popped = try popMark(&stack, arena);
                defer arena.free(popped);
                const kv_items = try makeKVList(arena, popped);
                break :blk .{ .seq = .{ .type = .dict, .values = kv_items } };
            }),
            .list => try stack.append(.{ .seq = .{ .type = .list, .values = try popMark(&stack, arena) } }),
            .inst => |v| try stack.append(blk: {
                const tup_items = try arena.dupe(Value, &.{ .{ .string = v.module }, .{ .string = v.class } });
                break :blk .{ .object = try Object.init(arena, .{ .seq = .{ .type = .tuple, .values = tup_items } }, try popMark(&stack, arena)) };
            }),
            .obj => try stack.append(blk: {
                const mark = try findMark(&stack);
                const args = try arena.dupe(Value, stack.items[mark + 2 ..]);
                const member = stack.items[mark + 1];
                break :blk .{ .object = try Object.init(arena, member, args) };
            }),
            .newobj => try stack.append(blk: {
                const args = try arena.alloc(Value, 1);
                args[0] = try pop(&stack);
                break :blk .{ .object = try Object.init(arena, try pop(&stack), args) };
            }),
            .empty_set => try stack.append(.{ .seq = .{ .type = .set, .values = &[_]Value{} } }),
            .additems => {
                const postmark = try popMark(&stack, arena);
                defer arena.free(postmark);
                const top = try lastMut(&stack);
                const rtop = try memo.resolveMut(top, true);
                switch (rtop.*) {
                    .global => |obj| {
                        const obj_len = obj.args.len;
                        obj.args = try assuredResize(Value, arena, obj.args, obj_len + postmark.len);
                        @memcpy(obj.args[obj_len..], postmark);
                    },
                    .seq => |*tup| {
                        const tup_len = tup.values.len;
                        tup.values = try assuredResize(Value, arena, tup.values, tup_len + postmark.len);
                        @memcpy(tup.values[tup_len..], postmark);
                    },
                    else => {
                        return error.BadStackTopForSetItem;
                    },
                }
            },
            .frozenset => try stack.append(.{ .seq = .{ .type = .frozen_set, .values = try popMark(&stack, arena) } }),
            .newobj_ex => try stack.append(blk: {
                const kwargs, const args, const cls = .{ try pop(&stack), try pop(&stack), try pop(&stack) };
                const new_seq: Sequence = .{ .type = .tuple, .values = try arena.dupe(Value, &.{ args, kwargs }) };
                break :blk .{ .object = try Object.init(arena, cls, try arena.dupe(Value, &.{.{ .seq = new_seq }})) };
            }),
            .stack_global => try stack.append(blk: {
                const gn, const mn = .{
                    try memo.resolve(arena, try pop(&stack), true),
                    try memo.resolve(arena, try pop(&stack), true),
                };
                const new_seq: Sequence = .{ .type = .tuple, .values = try arena.dupe(Value, &.{ gn, mn }) };
                break :blk .{ .object = try Object.init(arena, .{ .seq = new_seq }, &[_]Value{}) };
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

// TODO: this is a unmanaged array list, minus the optimisation. We should use that instead
fn assuredResize(comptime T: type, allocator: std.mem.Allocator, old: []T, new_length: usize) ![]T {
    if (allocator.resize(old, new_length)) {
        return old;
    } else {
        defer allocator.free(old);
        const new = try allocator.alloc(T, new_length);
        @memcpy(new[0..old.len], old);
        return new;
    }
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
    const entries = vals[0].seq.values[0].seq.values;
    const expected: []const Value = &.{
        .{ .seq = .{ .type = .kv_tuple, .values = @constCast(&[_]Value{ .{ .string = "hello" }, .{ .string = "world" } }) } },
        .{ .seq = .{ .type = .kv_tuple, .values = @constCast(&[_]Value{ .{ .string = "int" }, .{ .int64 = 1 } }) } },
        .{ .seq = .{ .type = .kv_tuple, .values = @constCast(&[_]Value{ .{ .string = "float" }, .{ .float64 = 3.141592 } }) } },
        .{
            .seq = .{ .type = .kv_tuple, .values = @constCast(&[_]Value{ .{ .string = "list" }, .{
                .seq = .{
                    .type = .list,
                    .values = @constCast(&[_]Value{
                        .{ .int64 = 255 },
                        .{ .int64 = 1234 },
                        .{ .int64 = -123 },
                        .{ .int64 = 1_000_000_000 },
                        .{ .int64 = 999_000_000_000 },
                        .{ .bigint = (try std.math.big.int.Managed.initSet(allocator, 999_000_000_000_000_000_000_000_000_000)).toConst() },
                    }),
                },
            } }) },
        },
        .{ .seq = .{ .type = .kv_tuple, .values = @constCast(&[_]Value{ .{ .string = "bool" }, .{ .boolval = false } }) } },
        .{ .seq = .{ .type = .kv_tuple, .values = @constCast(&[_]Value{
            .{ .string = "tuple" },
            .{ .seq = .{
                .type = .tuple,
                .values = @constCast(&[_]Value{
                    .{ .string = "a" },
                    .{ .int64 = 10 },
                }),
            } },
        }) } },
    };

    try std.testing.expectEqualDeep(expected, entries);
}

pub fn pop(values: *std.ArrayList(Value)) !Value {
    if (values.items.len == 0) {
        return error.StackUnderrun;
    }
    return values.pop();
}

fn popMarkDiscard(values: *std.ArrayList(Value)) !void {
    const mark = try findMark(values);
    values.shrinkRetainingCapacity(mark);
}

fn popMark(values: *std.ArrayList(Value), allocator: std.mem.Allocator) ![]Value {
    const mark = try findMark(values);
    const popping = values.items[mark + 1 ..];
    values.shrinkRetainingCapacity(mark);
    return try allocator.dupe(Value, popping);
}

fn lastMut(values: *std.ArrayList(Value)) !*Value {
    if (values.items.len == 0) {
        return error.UnexpectedEmptyStack;
    }
    return &values.items[values.items.len - 1];
}

fn findMark(values: *std.ArrayList(Value)) !usize {
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
