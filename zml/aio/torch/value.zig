const std = @import("std");
const utils = @import("utils.zig");

const PickleOp = @import("ops.zig").PickleOp;

const big_int = std.math.big.int;

/// The types of sequences that exist.
pub const SequenceType = enum {
    list,
    dict,
    kv_tuple,
    tuple,
    set,
    frozen_set,
};

pub const Object = struct {
    allocator: std.mem.Allocator,
    member: Value,
    args: []Value,

    pub fn init(allocator: std.mem.Allocator, member: Value, args: []Value) !*Object {
        const self = try allocator.create(Object);
        self.* = .{ .allocator = allocator, .member = member, .args = args };
        return self;
    }

    pub fn clone(self: *Object, allocator: std.mem.Allocator) std.mem.Allocator.Error!*Object {
        const res = try allocator.create(Object);
        res.* = .{ .allocator = allocator, .member = try self.member.clone(allocator), .args = try allocator.alloc(Value, self.args.len) };
        for (self.args, 0..) |v, i| res.args[i] = try v.clone(allocator);
        return res;
    }

    pub fn deinit(self: *Object) void {
        self.member.deinit(self.allocator);
        for (self.args) |*v| v.deinit(self.allocator);
        self.allocator.free(self.args);
        self.allocator.destroy(self);
    }
};

pub const Build = struct {
    allocator: std.mem.Allocator,
    member: Value,
    args: Value,

    pub fn init(allocator: std.mem.Allocator, member: Value, args: Value) !*Build {
        const self = try allocator.create(Build);
        self.* = .{ .allocator = allocator, .member = member, .args = args };
        return self;
    }

    pub fn clone(self: *Build, allocator: std.mem.Allocator) std.mem.Allocator.Error!*Build {
        const res = try allocator.create(Build);
        res.* = .{ .allocator = allocator, .member = try self.member.clone(allocator), .args = try self.args.clone(allocator) };
        return res;
    }

    pub fn deinit(self: *Build) void {
        self.member.deinit(self.allocator);
        self.args.deinit(self.allocator);
        self.allocator.destroy(self);
    }
};

pub const Sequence = struct { SequenceType, []Value };

pub const PersId = struct {
    allocator: std.mem.Allocator,
    ref: Value,

    pub fn init(allocator: std.mem.Allocator, ref: Value) !*PersId {
        const self = try allocator.create(PersId);
        self.* = .{ .allocator = allocator, .ref = ref };
        return self;
    }

    pub fn clone(self: *PersId, allocator: std.mem.Allocator) std.mem.Allocator.Error!*PersId {
        const res = try allocator.create(PersId);
        res.* = .{ .allocator = allocator, .ref = try self.ref.clone(allocator) };
        return res;
    }

    pub fn deinit(self: *PersId) void {
        self.ref.deinit(self.allocator);
        self.allocator.destroy(self);
    }
};

pub const ValueType = enum {
    raw,
    ref,
    app,
    object,
    build,
    pers_id,
    global,
    seq,
    string,
    bytes,
    int,
    bigint,
    float,
    raw_num,
    bool,
    none,
};

/// A processed value.
pub const Value = union(ValueType) {
    /// Types that we can't handle or just had to give up on processing.
    raw: PickleOp,

    /// A reference. You might be able to look it up in the memo map
    /// unless there's something weird going on like recursive references.
    /// You generally shouldn't see this in the result unless bad things
    /// are going on...
    ref: u32,

    /// The result of applying a thing to another thing. We're not
    /// Python so we don't really know what a "thing" is.
    app: *Object,

    /// An object or something. The first tuple member is the
    /// thing, the second one is the arguments it got applied to.
    object: *Object,

    /// Something we tried to build. The first tuple member is the
    /// thing, the second one is the arguments it got applied to.
    build: *Build,

    /// References to persistant storage. They basically could be anything.
    /// You kind of have to know what the thing you're trying to
    /// interface wants to use as keys for persistant storage.
    /// Good luck.
    pers_id: *PersId,

    /// A global value of some kind. The first tuple member is
    /// the thing, the second one is the arguments it got applied to.
    global: *Object,

    /// A sequence. We don't really distinguish between them
    /// much. The one exception is when the SequenceType is
    /// Dict we try to split the flat list of `[k, v, k, v, k, v]`
    /// into a list of tuples with the key and value.
    seq: Sequence,

    /// A string, but not the crazy strings that have to be
    /// unescaped as if they were Python strings. If you
    /// need one of those, look for it inside a `Value.raw`.
    string: []const u8,

    /// Some bytes. It might be a byte array or a binary
    /// string that couldn't get UTF8 decoded. We do the best
    /// we can.
    bytes: []const u8,

    /// An integer, but not the crazy kind that comes as a string
    /// that has to be parsed. You can look in `Value.raw_num` for
    /// those.
    int: i64,

    /// An integer that can't fit in i64.
    bigint: big_int.Managed,

    /// An float, but not the crazy kind that comes as a string
    /// that has to be parsed. You can look in `Value.raw_num` for
    /// those.
    float: f64,

    /// Some kind of weird number we can't handle.
    raw_num: PickleOp,

    /// A boolean value.
    bool: bool,

    /// Python `None`.
    none: void,

    pub fn deinit(self: *Value, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .raw, .raw_num => |v| v.deinit(allocator),
            inline .app, .object, .global, .build, .pers_id => |v| v.deinit(),
            .seq => |v| {
                for (v[1]) |*val| val.deinit(allocator);
                allocator.free(v[1]);
            },
            .string, .bytes => |v| allocator.free(v),
            .bigint => self.bigint.deinit(),
            else => {},
        }
        self.* = undefined;
    }

    inline fn writeIndents(indents: usize, writer: anytype) !void {
        try writer.writeBytesNTimes("  ", indents); // resolve tab = 2 spaces
        // try writer.writeByteNTimes('\t');
    }

    fn internalFormat(value: Value, indents: usize, writer: anytype) !void {
        try writeIndents(indents, writer);
        try writer.writeAll(".{\n");
        try writeIndents(indents + 1, writer);
        try writer.print(".{s} = ", .{@tagName(std.meta.activeTag(value))});
        switch (value) {
            inline .ref, .int, .float => |v| try writer.print("{d} ", .{v}),
            .app, .object, .global => |v| {
                try writer.writeAll(".{\n");
                try internalFormat(v.member, indents + 2, writer);
                try writer.writeAll(",\n");
                try writeIndents(indents + 2, writer);
                if (v.args.len > 0) {
                    try writer.writeAll(".{\n");
                    for (v.args, 0..) |arg, i| {
                        try internalFormat(arg, indents + 3, writer);
                        if (i < v.args.len - 1) try writer.writeAll(",");
                        try writer.writeByte('\n');
                    }
                    try writeIndents(indents + 2, writer);
                    try writer.writeAll("}\n");
                } else {
                    try writer.writeAll(".{}\n");
                }
                try writeIndents(indents + 1, writer);
                try writer.writeAll("}");
            },
            .build => |v| {
                try writer.writeAll(".{\n");
                try internalFormat(v.member, indents + 2, writer);
                try writer.writeAll(",\n");
                try internalFormat(v.args, indents + 2, writer);
                try writer.writeAll(",\n");
                try writeIndents(indents + 1, writer);
                try writer.writeAll("}");
            },
            inline .pers_id => |v| {
                try writer.writeByte('\n');
                try internalFormat(v.ref, indents + 2, writer);
            },
            .seq => |v| {
                try writer.writeAll(".{\n");
                try writeIndents(indents + 2, writer);
                try writer.print(".{s},\n", .{@tagName(v[0])});
                try writeIndents(indents + 2, writer);
                if (v[1].len > 0) {
                    try writer.writeAll(".{\n");
                    for (v[1], 0..) |arg, i| {
                        try internalFormat(arg, indents + 3, writer);
                        if (i < v[1].len - 1) try writer.writeAll(",");
                        try writer.writeByte('\n');
                    }
                    try writeIndents(indents + 2, writer);
                    try writer.writeAll("}\n");
                } else {
                    try writer.writeAll(".{}\n");
                }

                try writeIndents(indents + 1, writer);
                try writer.writeAll("}");
            },
            .string => |v| try writer.print("\"{s}\"", .{v}),
            .raw => |v| switch (v) {
                .global => |raw_global| try writer.print("\"{s}\", \"{s}\"", .{ raw_global[0], raw_global[1] }),
                else => try writer.print("{any}", .{v}),
            },
            inline else => |v| {
                try writer.print("{any}", .{v});
            },
        }
        try writer.writeByte('\n');
        try writeIndents(indents, writer);
        try writer.writeByte('}');
    }

    pub fn format(self: Value, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        return internalFormat(self, 0, writer);
    }

    pub fn clone(self: Value, allocator: std.mem.Allocator) !Value {
        return switch (self) {
            inline .raw, .raw_num => |v, tag| @unionInit(Value, @tagName(tag), try v.clone(allocator)),
            inline .app, .object, .global, .build, .pers_id => |v, tag| @unionInit(Value, @tagName(tag), try v.clone(allocator)),
            .seq => |seq| blk: {
                const new_val: Sequence = .{ seq[0], try allocator.alloc(Value, seq[1].len) };
                for (seq[1], 0..) |v, i| new_val[1][i] = try v.clone(allocator);
                break :blk .{ .seq = new_val };
            },
            inline .string, .bytes => |v, tag| @unionInit(Value, @tagName(tag), try allocator.dupe(u8, v)),
            .bigint => |v| .{ .bigint = try v.clone() },
            else => self,
        };
    }

    pub fn isPrimitive(self: Value) bool {
        return switch (self) {
            .int, .bigint, .float, .string, .bytes, .bool, .none => true,
            .seq => |seq| utils.allTrue(seq[1], Value.isPrimitive),
            else => false,
        };
    }

    pub fn containsRef(self: Value) bool {
        switch (self) {
            .ref => return true,
            .app, .object, .global => |v| {
                if (v.member.containsRef()) return true;
                for (v.args) |arg| if (arg.containsRef()) return true;
                return false;
            },
            .build => |v| {
                if (v.member.containsRef()) return true;
                if (v.args.containsRef()) return true;
                return false;
            },
            .pers_id => |v| return v.ref.containsRef(),
            .seq => |v| {
                for (v[1]) |val| if (val.containsRef()) return true;
                return false;
            },
            else => return false,
        }
    }

    const BI64MIN = big_int.Const{
        .limbs = &.{@intCast(@abs(std.math.minInt(i64)))},
        .positive = false,
    };

    const BI64MAX = big_int.Const{
        .limbs = &.{@intCast(std.math.maxInt(i64))},
        .positive = true,
    };

    pub fn coerceFromRaw(self: Value, allocator: std.mem.Allocator) !Value {
        return switch (self) {
            .raw => |raw_val| switch (raw_val) {
                .binint, .binint1, .binint2 => |val| .{ .int = val },
                .long1, .long4 => |b| if (b.len != 0) {
                    var bint = try big_int.Managed.initCapacity(allocator, std.math.big.int.calcTwosCompLimbCount(b.len));
                    var mutable = bint.toMutable();
                    mutable.readTwosComplement(b, b.len, .little, .signed);
                    const min_comp = bint.toConst().order(BI64MIN);
                    const max_comp = bint.toConst().order(BI64MAX);
                    if ((min_comp == .gt or min_comp == .eq) and (max_comp == .lt or max_comp == .eq)) {
                        defer bint.deinit();
                        return .{ .int = try bint.to(i64) };
                    } else return .{ .bigint = bint };
                } else .{ .raw_num = raw_val },
                .binfloat => |val| .{ .float = val },
                .binunicode, .binunicode8, .short_binunicode => |s| .{ .string = s },
                .binbytes, .binbytes8, .short_binbytes, .bytearray8 => |b| .{ .bytes = b },
                // This isn't how Pickle actually works but we just try to UTF8 decode the
                // string and if it fails, we make it a bytes value instead. If anyone
                // actually cares they can just fix values themselves or recover the raw bytes
                // from the UTF8 string (it's guaranteed to be reversible, as far as I know).
                .binstring, .short_binstring => |b| if (std.unicode.utf8ValidateSlice(b)) .{ .string = b } else .{ .bytes = b },
                .newtrue => .{ .bool = true },
                .newfalse => .{ .bool = false },
                .none => .{ .none = {} },
                inline .int,
                .float,
                .long,
                => |v, tag| {
                    if (tag == .int and std.mem.eql(u8, v, "01")) {
                        return .{ .bool = true };
                    } else if (tag == .int and std.mem.eql(u8, v, "00")) {
                        return .{ .bool = false };
                    } else {
                        return .{ .raw_num = raw_val };
                    }
                },
                else => self,
            },
            .app, .object, .global => |v| blk: {
                v.member = try v.member.coerceFromRaw(allocator);
                for (v.args) |*arg| {
                    arg.* = try arg.coerceFromRaw(allocator);
                }
                break :blk self;
            },
            .build => |v| blk: {
                v.member = try v.member.coerceFromRaw(allocator);
                v.args = try v.args.coerceFromRaw(allocator);
                break :blk self;
            },
            .pers_id => |v| blk: {
                v.ref = try v.ref.coerceFromRaw(allocator);
                break :blk self;
            },
            .seq => |*v| blk: {
                for (v[1]) |*val| {
                    val.* = try val.coerceFromRaw(allocator);
                }
                break :blk self;
            },
            else => self,
        };
    }
};
