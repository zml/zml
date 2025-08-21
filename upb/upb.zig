const std = @import("std");

const c = @import("c");

export fn _upb_MiniTable_StrongReference_dont_copy_me__upb_internal_use_only(mt: *c.upb_MiniTable) callconv(.c) void {
    const unused: *volatile c.upb_MiniTable = mt;
    _ = &unused;
}

pub const SerializeOptions = packed struct(c_int) {
    deterministic: bool = false,
    skip_unknown: bool = false,
    check_required: bool = false,
    _ignored: u29 = 0,
};

pub const SerializeError = error{
    MaxDepthExceeded,
    MissingRequired,
    Unknown,
} || std.mem.Allocator.Error;

pub const ParseOptions = packed struct {
    alias_string: bool = false,
    check_required: bool = false,
    experimental_allow_unlinked: bool = false,
    always_validate_utf8: bool = false,
    disable_fast_table: bool = false,
    _ignored: u27 = 0,
};

pub const ParseError = error{
    Malformed,
    BadUtf8,
    MaxDepthExceeded,
    MissingRequired,
    UnlinkedSubMessage,
    Unknown,
} || std.mem.Allocator.Error;

pub fn deepClone(comptime UpbType: type, arena: *c.upb_Arena, msg: *const UpbType) std.mem.Allocator.Error!*UpbType {
    return @ptrCast(c.upb_Message_DeepClone(@ptrCast(msg), Minitable(UpbType), arena) orelse return std.mem.Allocator.Error.OutOfMemory);
}

pub fn shallowClone(comptime UpbType: type, arena: *c.upb_Arena, msg: *const UpbType) std.mem.Allocator.Error!*UpbType {
    return @ptrCast(c.upb_Message_ShallowClone(@ptrCast(msg), Minitable(UpbType), arena) orelse return std.mem.Allocator.Error.OutOfMemory);
}

pub fn stringView(data: ?[]const u8) c.upb_StringView {
    return if (data) |d| c.upb_StringView_FromDataAndSize(d.ptr, d.len) else .{};
}

pub fn slice(sv: c.upb_StringView) ?[]const u8 {
    return if (sv.data) |d| d[0..sv.size] else null;
}

fn ProtoName(comptime UpbType: type) []const u8 {
    const needle = ".struct_";
    const type_name = @typeName(UpbType);
    const idx = std.mem.indexOf(u8, type_name, needle) orelse @compileError("Type name is invalid");
    return type_name[idx + needle.len ..];
}

fn Minitable(comptime UpbType: type) *const c.upb_MiniTable {
    const field_name = comptime blk: {
        const name = ProtoName(UpbType);
        var it = std.mem.tokenizeScalar(u8, name, '_');
        while (it.next()) |_| {
            const new_name = name[0..it.index] ++ "_" ++ name[it.index..] ++ "_msg_init";
            if (@hasDecl(c, new_name)) {
                break :blk new_name;
            }
        } else {
            @compileError("Unable to find minitable for type:" ++ @typeName(UpbType));
        }
    };
    return &@field(c, field_name);
}

pub fn serializeEx(ptr: anytype, arena: *c.upb_Arena, opts: SerializeOptions) SerializeError![]const u8 {
    var buf: [*c]u8 = undefined;
    var size: usize = undefined;
    return switch (c.upb_Encode(@ptrCast(ptr), Minitable(@TypeOf(ptr.*)), @bitCast(opts), arena, &buf, &size)) {
        c.kUpb_EncodeStatus_Ok => buf[0..size],
        c.kUpb_EncodeStatus_OutOfMemory => std.mem.Allocator.Error.OutOfMemory,
        c.kUpb_EncodeStatus_MaxDepthExceeded => SerializeError.MaxDepthExceeded,
        c.kUpb_EncodeStatus_MissingRequired => SerializeError.MissingRequired,
        else => return SerializeError.Unknown,
    };
}

pub fn serialize(ptr: anytype, arena: *c.upb_Arena) SerializeError![]const u8 {
    return try serializeEx(ptr, arena, .{});
}

pub fn parseEx(comptime UpbType: type, arena: *c.upb_Arena, data: []const u8, opts: ParseOptions) ParseError!*UpbType {
    const obj = try new(UpbType, arena);
    return switch (c.upb_Decode(@ptrCast(@constCast(data)), data.len, @alignCast(@ptrCast(obj)), Minitable(UpbType), null, @bitCast(opts), arena)) {
        c.kUpb_DecodeStatus_Ok => obj,
        c.kUpb_DecodeStatus_Malformed => ParseError.Malformed,
        c.kUpb_DecodeStatus_OutOfMemory => std.mem.Allocator.Error.OutOfMemory,
        c.kUpb_DecodeStatus_BadUtf8 => ParseError.BadUtf8,
        c.kUpb_DecodeStatus_MaxDepthExceeded => ParseError.MaxDepthExceeded,
        c.kUpb_DecodeStatus_MissingRequired => ParseError.MissingRequired,
        c.kUpb_DecodeStatus_UnlinkedSubMessage => ParseError.UnlinkedSubMessage,
        else => ParseError.Unknown,
    };
}

pub fn parse(comptime UpbType: type, arena: *c.upb_Arena, data: []const u8) ParseError!*UpbType {
    return parseEx(UpbType, arena, data, .{});
}

pub fn new(comptime UpbType: type, arena: *c.upb_Arena) std.mem.Allocator.Error!*UpbType {
    const new_fn = @field(c, ProtoName(UpbType) ++ "_new");
    return @ptrCast(new_fn(arena) orelse return std.mem.Allocator.Error.OutOfMemory);
}

pub const Allocator = struct {
    upb_alloc: c.upb_alloc,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Allocator {
        return .{
            .upb_alloc = .{
                .func = &alloc_impl,
            },
            .allocator = allocator,
        };
    }

    pub fn inner(self: *Allocator) *c.upb_alloc {
        return &self.upb_alloc;
    }

    fn alloc_impl(alloc: [*c]c.upb_alloc, ptr: ?*anyopaque, oldsize: usize, size: usize, actual_size: [*c]usize) callconv(.c) ?*anyopaque {
        const PointerAlignedSlice = [*c]align(@alignOf(*anyopaque)) u8;
        const upb_alloc: *c.upb_alloc = alloc orelse return null;
        const self: *Allocator = @fieldParentPtr("upb_alloc", upb_alloc);
        defer {
            if (actual_size) |as| {
                as.* = size;
            }
        }
        if (ptr) |ptr_| {
            const ptr_as_slice: []u8 = @as(PointerAlignedSlice, @ptrCast(@alignCast(ptr_)))[0..oldsize];
            if (size == 0) {
                self.allocator.free(ptr_as_slice);
                return null;
            } else if (size != oldsize) {
                return (self.allocator.realloc(ptr_as_slice, size) catch return null).ptr;
            }
            @panic("Unsupported case");
        }

        return @ptrCast(self.allocator.alignedAlloc(u8, @alignOf(*anyopaque), size) catch return null);
    }
};
