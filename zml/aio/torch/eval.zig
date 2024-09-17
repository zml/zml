const std = @import("std");
const zml = @import("../../zml.zig");
const meta = zml.meta;

const value = @import("value.zig");
const BTreeMap = @import("b_tree_map.zig").BTreeMap;
const PickleOp = @import("ops.zig").PickleOp;

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
                    for (v[1]) |*item| {
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
            .seq => |v| .{ .seq = .{ v[0], try self.resolveAllRefsIter(allocator, depth + 1, v[1], fix_values) } },
            .pers_id => |v| .{ .pers_id = try PersId.init(allocator, try self.resolveAllRefs(allocator, depth + 1, v.ref, fix_values)) },
            else => try val.clone(allocator),
        };
        if (fix_values) {
            output = try output.coerceFromRaw(allocator);
        }
        return output;
    }
};

pub const InternalStack = struct {
    allocator: std.mem.Allocator,
    values: std.ArrayList(Value),

    pub fn init(allocator: std.mem.Allocator) InternalStack {
        return .{
            .allocator = allocator,
            .values = std.ArrayList(Value).init(allocator),
        };
    }

    pub fn deinit(self: *InternalStack) void {
        for (0..self.values.items.len) |i| self.values.items[i].deinit(self.allocator);
        self.values.deinit();
        self.* = undefined;
    }

    pub fn pop(self: *InternalStack) !Value {
        if (self.values.items.len == 0) {
            return error.StackUnderrun;
        }
        return self.values.pop();
    }

    pub fn popMark(self: *InternalStack, allocator: ?std.mem.Allocator) ![]Value {
        const markidx = try self.findMark();
        var postmark: []Value = &[_]Value{};
        if (allocator) |a| {
            postmark = try a.alloc(Value, self.values.items.len - (markidx + 1));
            @memcpy(postmark, self.values.items[markidx + 1 ..]);
        }
        self.values.shrinkAndFree(markidx);
        return postmark;
    }

    pub fn lastMut(self: *InternalStack) !*Value {
        if (self.values.items.len == 0) {
            return error.UnexpectedEmptyStack;
        }
        return &self.values.items[self.values.items.len - 1];
    }

    pub fn findMark(self: *InternalStack) !usize {
        const len = self.values.items.len;
        for (0..len) |i| {
            const idx = (len - 1) - i;
            const val = self.values.items[idx];
            if (val == .raw and val.raw == .mark) {
                return idx;
            }
        }
        zml.log.warn("pytorch loader: missing mark", .{});
        return 0;
    }

    pub fn toPickleStack(self: *InternalStack) !PickleStack {
        return .{ .stack = try self.values.toOwnedSlice(), .allocator = self.allocator };
    }
};

pub const PickleStack = struct {
    stack: []Value,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, values: []Value) PickleStack {
        return .{ .allocator = allocator, .stack = values };
    }

    pub fn deinit(self: *PickleStack) void {
        for (self.stack) |*v| v.deinit(self.allocator);
        self.allocator.free(self.stack);
    }
};

pub fn evaluate(allocator: std.mem.Allocator, x: []const PickleOp, resolve_refs: bool) !struct { PickleStack, PickleMemo } {
    var stack = InternalStack.init(allocator);
    defer stack.deinit();
    var memo = PickleMemo.init(allocator);
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
                kv_items.appendAssumeCapacity(.{ .seq = .{ .kv_tuple, kv } });
            }
            return kv_items.toOwnedSlice();
        }
    }).call;

    outer: for (x) |op| {
        switch (op) {
            .mark => try stack.values.append(.{ .raw = op }),
            .stop => break :outer,
            .pop => _ = try stack.pop(),
            .pop_mark => _ = try stack.popMark(allocator),
            .dup => {
                if (stack.values.getLastOrNull()) |item| {
                    try stack.values.append(try item.clone(allocator));
                } else {
                    return error.CannotDupEmptyStack;
                }
            },
            .persid => |v| try stack.values.append(.{ .pers_id = try PersId.init(allocator, .{ .string = try allocator.dupe(u8, v) }) }),
            .binpersid => try stack.values.append(.{ .pers_id = try PersId.init(allocator, try stack.pop()) }),
            .reduce => try stack.values.append(.{ .global = blk: {
                const values = try allocator.alloc(Value, 1);
                values[0] = try memo.resolve(allocator, try stack.pop(), true);
                break :blk try Object.init(allocator, try memo.resolve(allocator, try stack.pop(), true), values);
            } }),
            .build => try stack.values.append(blk: {
                const args = try memo.resolve(allocator, try stack.pop(), true);
                const member = try memo.resolve(allocator, try stack.pop(), true);
                break :blk .{ .build = try Build.init(allocator, member, args) };
            }),
            .empty_dict => try stack.values.append(.{ .seq = .{ .dict, &[_]Value{} } }),
            .get => |v| try stack.values.append(.{ .ref = try std.fmt.parseInt(u32, v, 10) }),
            inline .binget, .long_binget => |v| try stack.values.append(.{ .ref = v }),
            .empty_list => try stack.values.append(.{ .seq = .{ .list, &[_]Value{} } }),
            .binput, .long_binput => |v| {
                try memo.insert(v, try stack.pop());
                try stack.values.append(.{ .ref = v });
            },
            .tuple => try stack.values.append(blk: {
                const popped = try stack.popMark(allocator);
                break :blk .{ .seq = .{ .tuple, popped } };
            }),
            .empty_tuple => try stack.values.append(.{ .seq = .{ .tuple, &[_]Value{} } }),
            .setitem => {
                const v, const k = .{ try stack.pop(), try stack.pop() };
                const top = try stack.lastMut();
                const rtop = try memo.resolveMut(top, true);
                switch (rtop.*) {
                    .global => |obj| {
                        obj.args = try assuredResize(Value, allocator, obj.args, obj.args.len + 1);
                        obj.args[obj.args.len - 1] = .{ .seq = .{ .tuple, try allocator.dupe(Value, &.{ k, v }) } };
                    },
                    .seq => |*tup| {
                        tup[1] = try assuredResize(Value, allocator, tup[1], tup[1].len + 1);
                        tup[1][tup[1].len - 1] = .{ .seq = .{ .tuple, try allocator.dupe(Value, &.{ k, v }) } };
                    },
                    else => {
                        return error.BadStackTopForSetItem;
                    },
                }
            },
            .setitems => {
                const popped = try stack.popMark(allocator);
                defer allocator.free(popped);
                const kv_items = try makeKVList(allocator, popped);
                const top = try stack.lastMut();
                const rtop = try memo.resolveMut(top, true);
                switch (rtop.*) {
                    .global => |obj| {
                        obj.args = try assuredResize(Value, allocator, obj.args, obj.args.len + 1);
                        obj.args[obj.args.len - 1] = .{ .seq = .{ .tuple, kv_items } };
                    },
                    .seq => |*tup| {
                        tup[1] = try assuredResize(Value, allocator, tup[1], tup[1].len + 1);
                        tup[1][tup[1].len - 1] = .{ .seq = .{ .tuple, kv_items } };
                    },
                    else => {
                        defer allocator.free(kv_items);
                        return error.BadStackTopForSetItems;
                    },
                }
            },
            .proto => |proto| meta.assert(proto <= MAX_PROTOCOL, "Unsupported protocol {d}", .{proto}),
            .tuple1 => try stack.values.append(blk: {
                const tup_values = try allocator.alloc(Value, 1);
                tup_values[0] = try stack.pop();
                break :blk .{ .seq = .{ .tuple, tup_values } };
            }),
            .tuple2 => try stack.values.append(blk: {
                const tup_values = try allocator.alloc(Value, 2);
                inline for (0..2) |i| tup_values[(tup_values.len - 1) - i] = try stack.pop();
                break :blk .{ .seq = .{ .tuple, tup_values } };
            }),
            .tuple3 => try stack.values.append(blk: {
                const tup_values = try allocator.alloc(Value, 3);
                inline for (0..3) |i| tup_values[(tup_values.len - 1) - i] = try stack.pop();
                break :blk .{ .seq = .{ .tuple, tup_values } };
            }),
            .append => {
                const v = try stack.pop();
                const top = try stack.lastMut();
                const rtop = try memo.resolveMut(top, true);
                switch (rtop.*) {
                    .global => |obj| {
                        obj.args = try assuredResize(Value, allocator, obj.args, obj.args.len + 1);
                        obj.args[obj.args.len - 1] = v;
                    },
                    .seq => |*tup| {
                        tup[1] = try assuredResize(Value, allocator, tup[1], tup[1].len + 1);
                        tup[1][tup[1].len - 1] = v;
                    },
                    else => {
                        return error.BadStackTopForAppend;
                    },
                }
            },
            .appends => {
                const postmark = try stack.popMark(allocator);
                defer allocator.free(postmark);
                const top = try stack.lastMut();
                const rtop = try memo.resolveMut(top, true);
                switch (rtop.*) {
                    .global => |obj| {
                        const obj_len = obj.args.len;
                        obj.args = try assuredResize(Value, allocator, obj.args, obj_len + postmark.len);
                        @memcpy(obj.args[obj_len..], postmark);
                    },
                    .seq => |*tup| {
                        const tup_len = tup[1].len;
                        tup[1] = try assuredResize(Value, allocator, tup[1], tup_len + postmark.len);
                        @memcpy(tup[1][tup_len..], postmark);
                    },
                    else => {
                        return error.BadStackTopForAppends;
                    },
                }
            },
            .dict => try stack.values.append(blk: {
                const popped = try stack.popMark(allocator);
                defer allocator.free(popped);
                const kv_items = try makeKVList(allocator, popped);
                break :blk .{ .seq = .{ .dict, kv_items } };
            }),
            .list => try stack.values.append(.{ .seq = .{ .list, try stack.popMark(allocator) } }),
            .inst => |v| try stack.values.append(blk: {
                const tup_items = try allocator.dupe(Value, &.{ .{ .string = v[0] }, .{ .string = v[1] } });
                break :blk .{ .object = try Object.init(allocator, .{ .seq = .{ .tuple, tup_items } }, try stack.popMark(allocator)) };
            }),
            .obj => try stack.values.append(blk: {
                const markidx = try stack.findMark();
                const args = try allocator.alloc(Value, stack.values.items.len - (markidx + 2));
                @memcpy(args, stack.values.items[markidx + 2 ..]);
                const member = stack.values.items[markidx + 1];
                break :blk .{ .object = try Object.init(allocator, member, args) };
            }),
            .put => |v| {
                const mid = try std.fmt.parseInt(u32, v, 10);
                try memo.insert(mid, try stack.pop());
                try stack.values.append(.{ .ref = mid });
            },
            .newobj => try stack.values.append(blk: {
                const args = try allocator.alloc(Value, 1);
                args[0] = try stack.pop();
                break :blk .{ .object = try Object.init(allocator, try stack.pop(), args) };
            }),
            .empty_set => try stack.values.append(.{ .seq = .{ .set, &[_]Value{} } }),
            .additems => {
                const postmark = try stack.popMark(allocator);
                defer allocator.free(postmark);
                const top = try stack.lastMut();
                const rtop = try memo.resolveMut(top, true);
                switch (rtop.*) {
                    .global => |obj| {
                        const obj_len = obj.args.len;
                        obj.args = try assuredResize(Value, allocator, obj.args, obj_len + postmark.len);
                        @memcpy(obj.args[obj_len..], postmark);
                    },
                    .seq => |*tup| {
                        const tup_len = tup[1].len;
                        tup[1] = try assuredResize(Value, allocator, tup[1], tup_len + postmark.len);
                        @memcpy(tup[1][tup_len..], postmark);
                    },
                    else => {
                        return error.BadStackTopForSetItem;
                    },
                }
            },
            .frozenset => try stack.values.append(.{ .seq = .{ .frozen_set, try stack.popMark(allocator) } }),
            .newobj_ex => try stack.values.append(blk: {
                const kwargs, const args, const cls = .{ try stack.pop(), try stack.pop(), try stack.pop() };
                const new_seq: Sequence = .{ .tuple, try allocator.dupe(Value, &.{ args, kwargs }) };
                break :blk .{ .object = try Object.init(allocator, cls, try allocator.dupe(Value, &.{.{ .seq = new_seq }})) };
            }),
            .stack_global => try stack.values.append(blk: {
                const gn, const mn = .{
                    try memo.resolve(allocator, try stack.pop(), true),
                    try memo.resolve(allocator, try stack.pop(), true),
                };
                const new_seq: Sequence = .{ .tuple, try allocator.dupe(Value, &.{ gn, mn }) };
                break :blk .{ .object = try Object.init(allocator, .{ .seq = new_seq }, &[_]Value{}) };
            }),
            .memoize => {
                const item = stack.values.getLastOrNull() orelse {
                    return error.StackUnderrun;
                };
                try memo.insert(@intCast(memo.map.count()), try item.clone(allocator));
            },
            else => try stack.values.append(.{ .raw = try op.clone(allocator) }),
        }
    }
    if (!resolve_refs) {
        return .{ try stack.toPickleStack(), memo };
    }
    return .{
        PickleStack.init(allocator, try memo.resolveAllRefsIter(allocator, 0, stack.values.items, true)),
        memo,
    };
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
