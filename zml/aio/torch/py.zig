const std = @import("std");
const math = std.math;

const pickle = @import("pickle.zig");

const log = std.log.scoped(.@"zml/aio");

/// Correspond to a function/constructor call
pub const Object = struct {
    member: Any,
    args: []Any,
    kwargs: []Any,

    pub fn init(allocator: std.mem.Allocator, member: Any, args: []Any, kwargs: []Any) !*Object {
        const self = try allocator.create(Object);
        self.* = .{ .member = member, .args = args, .kwargs = kwargs };
        return self;
    }

    pub fn clone(self: *Object, allocator: std.mem.Allocator) std.mem.Allocator.Error!*Object {
        const res = try allocator.create(Object);
        res.* = .{
            .member = try self.member.clone(allocator),
            .args = try allocator.alloc(Any, self.args.len),
            .kwargs = try allocator.alloc(Any, self.kwargs.len),
        };
        for (self.args, 0..) |v, i| res.args[i] = try v.clone(allocator);
        for (self.kwargs, 0..) |v, i| res.kwargs[i] = try v.clone(allocator);
        return res;
    }

    pub fn deinit(self: *Object, allocator: std.mem.Allocator) void {
        self.member.deinit(allocator);
        for (self.args) |*v| v.deinit(allocator);
        allocator.free(self.args);
        allocator.destroy(self);
    }
};

/// Correspond to the __set_state__ call when pickle finishes building an object.
pub const SetState = struct {
    obj: Any,
    state: Any,

    pub fn init(allocator: std.mem.Allocator, obj: Any, state: Any) !*SetState {
        const res = try allocator.create(SetState);
        res.* = .{ .obj = obj, .state = state };
        return res;
    }

    pub fn clone(self: *SetState, allocator: std.mem.Allocator) std.mem.Allocator.Error!*SetState {
        const res = try allocator.create(SetState);
        res.* = .{ .obj = try self.obj.clone(allocator), .state = try self.state.clone(allocator) };
        return res;
    }

    pub fn deinit(self: *SetState, allocator: std.mem.Allocator) void {
        self.obj.deinit(allocator);
        self.state.deinit(allocator);
        self.allocator.destroy(self);
    }
};

/// The types of sequences that exist.
pub const SequenceType = enum {
    list,
    dict,
    tuple,
    set,
    frozen_set,
};

pub const Sequence = struct {
    type: SequenceType,
    values: []Any,
};

pub fn tuple(values: []const Any) Any {
    // tuple are readonly, but sequence in general aren't
    return .{ .seq = .{ .type = .tuple, .values = @constCast(values) } };
}

pub const PersId = struct {
    ref: Any,

    pub fn init(allocator: std.mem.Allocator, ref: Any) !*PersId {
        const self = try allocator.create(PersId);
        self.* = .{ .ref = ref };
        return self;
    }

    pub fn clone(self: *PersId, allocator: std.mem.Allocator) std.mem.Allocator.Error!*PersId {
        const res = try allocator.create(PersId);
        res.* = .{ .ref = try self.ref.clone(allocator) };
        return res;
    }

    pub fn deinit(self: *PersId, allocator: std.mem.Allocator) void {
        self.ref.deinit(allocator);
        allocator.destroy(self);
    }
};

pub const Kind = enum {
    raw,
    ref,
    app,
    object,
    set_state,
    pers_id,
    global,
    seq,
    string,
    bytes,
    int64,
    bigint,
    float64,
    raw_num,
    boolval,
    none,
};

/// A pickle operator that has been interpreted.
pub const Any = union(Kind) {
    /// Types that we can't handle or just had to give up on processing.
    raw: pickle.Op,

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

    /// Correspond to the __set_state__ call when pickle finishes building an object.
    /// The first tuple member is the target object,
    /// the second one is the "state" argument
    set_state: *SetState,

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
    int64: i64,

    /// An integer that can't fit in i64.
    bigint: math.big.int.Const,

    /// An float, but not the crazy kind that comes as a string
    /// that has to be parsed. You can look in `Value.raw_num` for
    /// those.
    float64: f64,

    /// Some kind of weird number we can't handle.
    raw_num: pickle.Op,

    /// A boolean value.
    boolval: bool,

    /// Python `None`.
    none: void,

    pub fn deinit(self: *Any, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .raw, .raw_num => |v| v.deinit(allocator),
            inline .app, .object, .global, .build, .pers_id => |v| v.deinit(),
            .seq => |v| {
                for (v.values) |*val| val.deinit(allocator);
                allocator.free(v.values);
            },
            .string, .bytes => |v| allocator.free(v),
            .bigint => |big| allocator.free(big.limbs),
            else => {},
        }
        self.* = undefined;
    }

    inline fn writeIndents(indents: usize, writer: *std.Io.Writer) !void {
        try writer.writeBytesNTimes("  ", indents); // resolve tab = 2 spaces
        // try writer.writeByteNTimes('\t');
    }

    pub fn format(self: Any, writer: *std.Io.Writer) !void {
        return internalFormat(self, 0, writer);
    }

    fn internalFormat(value: Any, indents: usize, writer: *std.Io.Writer) !void {
        try writeIndents(indents, writer);
        try writer.writeAll(".{\n");
        try writeIndents(indents + 1, writer);
        try writer.print(".{s} = ", .{@tagName(std.meta.activeTag(value))});
        switch (value) {
            inline .ref, .int64, .float64 => |v| try writer.print("{d} ", .{v}),
            .app, .object, .global => |v| {
                try writer.writeAll(".{\n");
                try writeIndents(indents + 2, writer);
                try writer.writeAll(".fn =");
                try internalFormat(v.member, indents + 2, writer);
                try writer.writeAll(",\n");
                try writeIndents(indents + 2, writer);
                try writer.writeAll(".args = ");
                if (v.args.len > 0) {
                    try writer.writeAll(".{\n");
                    for (v.args, 0..) |arg, i| {
                        try internalFormat(arg, indents + 3, writer);
                        if (i < v.args.len - 1) try writer.writeAll(",");
                        try writer.writeByte('\n');
                    }
                    try writeIndents(indents + 2, writer);
                    try writer.writeAll("},\n");
                } else {
                    try writer.writeAll(".{},\n");
                }
                try writeIndents(indents + 2, writer);
                try writer.writeAll(".kwargs =");
                if (v.kwargs.len > 0) {
                    try writer.writeAll(".{\n");
                    for (v.kwargs, 0..) |arg, i| {
                        try internalFormat(arg, indents + 3, writer);
                        if (i < v.kwargs.len - 1) try writer.writeAll(",");
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
            .set_state => |v| {
                try writer.writeAll(".{\n");
                try internalFormat(v.obj, indents + 2, writer);
                try writer.writeAll(",\n");
                try internalFormat(v.state, indents + 2, writer);
                try writer.writeAll(",\n");
                try writeIndents(indents + 1, writer);
                try writer.writeAll("}");
            },
            .pers_id => |v| {
                try writer.writeByte('\n');
                try internalFormat(v.ref, indents + 2, writer);
            },
            .seq => |v| {
                try writer.writeAll(".{\n");
                try writeIndents(indents + 2, writer);
                try writer.print(".{s},\n", .{@tagName(v.type)});
                try writeIndents(indents + 2, writer);
                if (v.values.len > 0) {
                    try writer.writeAll(".{\n");
                    for (v.values, 0..) |arg, i| {
                        try internalFormat(arg, indents + 3, writer);
                        if (i < v.values.len - 1) try writer.writeAll(",");
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
                .global => |py_type| try writer.print("\"{s}\", \"{s}\"", .{ py_type.module, py_type.class }),
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

    pub fn clone(self: Any, allocator: std.mem.Allocator) !Any {
        return switch (self) {
            inline .raw, .raw_num => |v, tag| @unionInit(Any, @tagName(tag), try v.clone(allocator)),
            inline .app, .object, .global, .set_state, .pers_id => |v, tag| @unionInit(Any, @tagName(tag), try v.clone(allocator)),
            .seq => |seq| {
                const values = try allocator.alloc(Any, seq.values.len);
                for (seq.values, 0..) |v, i| values[i] = try v.clone(allocator);
                return .{ .seq = .{ .type = seq.type, .values = values } };
            },
            inline .string, .bytes => |v, tag| @unionInit(Any, @tagName(tag), try allocator.dupe(u8, v)),
            .bigint => |v| .{ .bigint = (try v.toManaged(allocator)).toConst() },
            else => self,
        };
    }

    pub fn isPrimitive(self: Any) bool {
        return switch (self) {
            .int64, .bigint, .float64, .string, .bytes, .boolval, .none => true,
            .seq => |seq| {
                for (seq.values) |v| {
                    if (!v.isPrimitive()) return false;
                }
                return true;
            },
            else => false,
        };
    }

    pub fn containsRef(self: Any) bool {
        switch (self) {
            .ref => return true,
            .app, .object, .global => |v| {
                if (v.member.containsRef()) return true;
                for (v.args) |arg| if (arg.containsRef()) return true;
                return false;
            },
            .set_state => |v| {
                if (v.obj.containsRef()) return true;
                if (v.state.containsRef()) return true;
                return false;
            },
            .pers_id => |v| return v.ref.containsRef(),
            .seq => |v| {
                for (v.values) |val| if (val.containsRef()) return true;
                return false;
            },
            else => return false,
        }
    }

    pub const UnpickleError = error{ InvalidCharacter, OutOfMemory };

    pub fn coerceFromRaw(self: Any, allocator: std.mem.Allocator) UnpickleError!Any {
        return switch (self) {
            .raw => |raw_val| switch (raw_val) {
                .none => .none,
                .bool => |b| .{ .boolval = b },
                .float => |b| .{ .float64 = std.fmt.parseFloat(f64, b) catch std.math.nan(f64) },
                .int => |val| .{ .int64 = val },
                .long => |digits| {
                    const n = std.fmt.parseInt(i64, digits[0 .. digits.len - 1], 10) catch |err| {
                        switch (err) {
                            error.Overflow => {
                                log.warn("Not parsing long integer: {s}", .{digits});
                                return self;
                            },
                            error.InvalidCharacter => return error.InvalidCharacter,
                        }
                    };
                    return .{ .int64 = n };
                },
                .binlong => |bytes| if (bytes.len <= 8)
                    .{ .int64 = std.mem.readVarInt(i64, bytes, .little) }
                else {
                    // Note: we need to copy here, because Zig big int limbs are usize aligned,
                    // whereas pickle big int are byte aligned.
                    const n_limbs = std.math.divCeil(usize, bytes.len, @sizeOf(math.big.Limb)) catch unreachable;
                    var big = (try math.big.int.Managed.initCapacity(allocator, n_limbs)).toMutable();
                    big.readTwosComplement(bytes, bytes.len * 8, .little, .signed);

                    return .{ .bigint = big.toConst() };
                },
                .binfloat => |val| .{ .float64 = val },
                .unicode => |s| .{ .string = s },
                inline .bytes, .bytearray => |b| .{ .bytes = b },
                // This isn't how Pickle actually works but we just try to UTF8 decode the
                // string and if it fails, we make it a bytes value instead. If anyone
                // actually cares they can just fix values themselves or recover the raw bytes
                // from the UTF8 string (it's guaranteed to be reversible, as far as I know).
                .string => |b| if (std.unicode.utf8ValidateSlice(b))
                    .{ .string = b }
                else
                    .{ .bytes = b },
                else => self,
            },
            .app, .object, .global => |v| blk: {
                v.member = try v.member.coerceFromRaw(allocator);
                for (v.args) |*arg| {
                    arg.* = try arg.coerceFromRaw(allocator);
                }
                break :blk self;
            },
            .set_state => |v| blk: {
                v.obj = try v.obj.coerceFromRaw(allocator);
                v.state = try v.state.coerceFromRaw(allocator);
                break :blk self;
            },
            .pers_id => |v| blk: {
                v.ref = try v.ref.coerceFromRaw(allocator);
                break :blk self;
            },
            .seq => |v| blk: {
                for (v.values) |*val| {
                    val.* = try val.coerceFromRaw(allocator);
                }
                break :blk self;
            },
            else => self,
        };
    }
};
