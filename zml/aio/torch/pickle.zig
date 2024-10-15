const std = @import("std");

/// A decoded Pickle operation in its natural state.
pub const Op = union(OpCode) {
    mark,
    stop,
    pop,
    pop_mark,
    dup,
    float: []const u8,
    int: []const u8,
    binint: i32,
    binint1: u8,
    long: []const u8,
    binint2: u16,
    none,
    persid: []const u8,
    binpersid,
    reduce,
    string: []const u8,
    binstring: []const u8,
    short_binstring: []const u8,
    unicode: []const u8,
    binunicode: []const u8,
    append,
    build,
    global: PyType,
    dict,
    empty_dict,
    appends,
    get: []const u8,
    binget: u8,
    inst: PyType,
    long_binget: u32,
    list,
    empty_list,
    obj,
    put: []const u8,
    binput: u8,
    long_binput: u32,
    setitem,
    tuple,
    empty_tuple,
    setitems,
    binfloat: f64,
    proto: u8,
    newobj,
    ext1: u8,
    ext2: i16,
    ext4: i32,
    tuple1,
    tuple2,
    tuple3,
    newtrue,
    newfalse,
    long1: []const u8,
    long4: []const u8,
    binbytes: []const u8,
    short_binbytes: []const u8,
    short_binunicode: []const u8,
    binunicode8: []const u8,
    binbytes8: []const u8,
    empty_set,
    additems,
    frozenset,
    newobj_ex,
    stack_global,
    memoize,
    frame: u64,
    bytearray8: []const u8,
    next_buffer,
    readonly_buffer,

    pub const PyType = struct { module: []const u8, class: []const u8 };

    pub fn deinit(self: Op, allocator: std.mem.Allocator) void {
        switch (self) {
            .float,
            .int,
            .long,
            .persid,
            .string,
            .binstring,
            .short_binstring,
            .unicode,
            .binunicode,
            .get,
            .put,
            .long1,
            .long4,
            .binbytes,
            .short_binbytes,
            .short_binunicode,
            .binunicode8,
            .binbytes8,
            .bytearray8,
            => |v| allocator.free(v),
            .global, .inst => |py_type| {
                allocator.free(py_type.module);
                allocator.free(py_type.class);
            },
            else => {},
        }
    }

    pub fn clone(self: Op, allocator: std.mem.Allocator) !Op {
        var res = self;
        return switch (self) {
            inline .float,
            .int,
            .long,
            .persid,
            .string,
            .binstring,
            .short_binstring,
            .unicode,
            .binunicode,
            .get,
            .put,
            .long1,
            .long4,
            .binbytes,
            .short_binbytes,
            .short_binunicode,
            .binunicode8,
            .binbytes8,
            .bytearray8,
            => |v, tag| {
                const cloned = try allocator.alloc(u8, v.len);
                @memcpy(cloned, v);
                @field(res, @tagName(tag)) = cloned;
                return res;
            },
            inline .global, .inst => |v, tag| {
                @field(res, @tagName(tag)) = PyType{
                    .module = try allocator.dupe(u8, v.module),
                    .class = try allocator.dupe(u8, v.class),
                };
                return res;
            },
            else => self,
        };
    }
};

/// The values for the possible opcodes are in this enum.
/// Reference: https://github.com/python/cpython/blob/3.13/Lib/pickletools.py
pub const OpCode = enum(u8) {
    mark = '(', // push special markobject on stack
    stop = '.', // every pickle ends with stop
    pop = '0', // discard topmost stack item
    pop_mark = '1', // discard stack top through topmost markobject
    dup = '2', // duplicate top stack item
    float = 'F', // push float object; decimal string argument
    int = 'I', // push integer or bool; decimal string argument
    binint = 'J', // push four-byte signed int
    binint1 = 'K', // push 1-byte unsigned int
    long = 'L', // push long; decimal string argument
    binint2 = 'M', // push 2-byte unsigned int
    none = 'N', // push None
    persid = 'P', // push persistent object; id is taken from string arg
    binpersid = 'Q', //  "       "         "  ;  "  "   "     "  stack
    reduce = 'R', // apply callable to argtuple, both on stack
    string = 'S', // push string; NL-terminated string argument
    binstring = 'T', // push string; counted binary string argument
    short_binstring = 'U', //  "     "   ;    "      "       "      " < 256 bytes
    unicode = 'V', // push Unicode string; raw-unicode-escaped'd argument
    binunicode = 'X', //   "     "       "  ; counted UTF-8 string argument
    append = 'a', // append stack top to list below it
    build = 'b', // call __setstate__ or __dict__.update()
    global = 'c', // push self.find_class(modname, name); 2 string args
    dict = 'd', // build a dict from stack items
    empty_dict = '}', // push empty dict
    appends = 'e', // extend list on stack by topmost stack slice
    get = 'g', // push item from memo on stack; index is string arg
    binget = 'h', //   "    "    "    "   "   "  ;   "    " 1-byte arg
    inst = 'i', // build & push class instance
    long_binget = 'j', // push item from memo on stack; index is 4-byte arg
    list = 'l', // build list from topmost stack items
    empty_list = ']', // push empty list
    obj = 'o', // build & push class instance
    put = 'p', // store stack top in memo; index is string arg
    binput = 'q', //   "     "    "   "   " ;   "    " 1-byte arg
    long_binput = 'r', //   "     "    "   "   " ;   "    " 4-byte arg
    setitem = 's', // add key+value pair to dict
    tuple = 't', // build tuple from topmost stack items
    empty_tuple = ')', // push empty tuple
    setitems = 'u', // modify dict by adding topmost key+value pairs
    binfloat = 'G', // push float; arg is 8-byte float encoding

    // Protocol 2
    proto = '\x80', // identify pickle protocol
    newobj = '\x81', // build object by applying cls.__new__ to argtuple
    ext1 = '\x82', // push object from extension registry; 1-byte index
    ext2 = '\x83', // ditto, but 2-byte index
    ext4 = '\x84', // ditto, but 4-byte index
    tuple1 = '\x85', // build 1-tuple from stack top
    tuple2 = '\x86', // build 2-tuple from two topmost stack items
    tuple3 = '\x87', // build 3-tuple from three topmost stack items
    newtrue = '\x88', // push True
    newfalse = '\x89', // push False
    long1 = '\x8a', // push long from < 256 bytes
    long4 = '\x8b', // push really big long

    // Protocol 3
    binbytes = 'B', // push bytes; counted binary string argument
    short_binbytes = 'C', //  "     "   ;    "      "       "      " < 256 bytes

    // Protocol 4
    short_binunicode = '\x8c', // push short string; UTF-8 length < 256 bytes
    binunicode8 = '\x8d', // push very long string
    binbytes8 = '\x8e', // push very long bytes string
    empty_set = '\x8f', // push empty set on the stack
    additems = '\x90', // modify set by adding topmost stack items
    frozenset = '\x91', // build frozenset from topmost stack items
    newobj_ex = '\x92', // like newobj but work with keyword only arguments
    stack_global = '\x93', // same as GLOBAL but using names on the stacks
    memoize = '\x94', // store top of the stack in memo
    frame = '\x95', // indicate the beginning of a new frame

    // Protocol 5
    bytearray8 = '\x96', // push bytearray
    next_buffer = '\x97', // push next out-of-band buffer
    readonly_buffer = '\x98', // make top of stack readonly
    _,
};

pub fn parse(allocator: std.mem.Allocator, reader: anytype, max_line_len: usize) ![]const Op {
    var results = std.ArrayList(Op).init(allocator);
    errdefer results.deinit();
    const len = max_line_len;

    while (true) {
        const b = try reader.readByte();
        const code: OpCode = @enumFromInt(b);
        switch (code) {
            .stop => {
                //
                try results.append(.{ .stop = {} });
                break;
            },
            .mark => try results.append(.{ .mark = {} }),
            .pop => try results.append(.{ .pop = {} }),
            .pop_mark => try results.append(.{ .pop_mark = {} }),
            .dup => try results.append(.{ .dup = {} }),
            .float => {
                const buf = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                errdefer allocator.free(buf);
                try results.append(.{ .float = buf });
            },
            .int => {
                const buf = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                errdefer allocator.free(buf);
                try results.append(.{ .int = buf });
            },
            .binint => try results.append(.{ .binint = try reader.readInt(i32, .little) }),
            .binint1 => try results.append(.{ .binint1 = try reader.readByte() }),
            .long => {
                const buf = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                errdefer allocator.free(buf);
                try results.append(.{ .long = buf });
            },
            .binint2 => try results.append(.{ .binint2 = try reader.readInt(u16, .little) }),
            .none => try results.append(.{ .none = {} }),
            .persid => {
                const buf = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                errdefer allocator.free(buf);
                try results.append(.{ .persid = buf });
            },
            .binpersid => try results.append(.{ .binpersid = {} }),
            .reduce => try results.append(.{ .reduce = {} }),
            .string => {
                const buf = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                errdefer allocator.free(buf);
                try results.append(.{ .string = buf });
            },
            .binstring => {
                const str_len = try reader.readInt(u32, .little);
                const buf = try allocator.alloc(u8, str_len);
                errdefer allocator.free(buf);
                _ = try reader.read(buf);
                try results.append(.{ .binstring = buf });
            },
            .short_binstring => {
                const str_len = try reader.readByte();
                const buf = try allocator.alloc(u8, str_len);
                errdefer allocator.free(buf);
                _ = try reader.read(buf);
                try results.append(.{ .short_binstring = buf });
            },
            .unicode => {
                const buf = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                errdefer allocator.free(buf);
                try results.append(.{ .unicode = buf });
            },
            .binunicode => {
                const str_len = try reader.readInt(u32, .little);
                const buf = try allocator.alloc(u8, str_len);
                errdefer allocator.free(buf);
                _ = try reader.read(buf);
                try results.append(.{ .binunicode = buf });
            },
            .append => try results.append(.{ .append = {} }),
            .build => try results.append(.{ .build = {} }),
            .global, .inst => {
                const module = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                errdefer allocator.free(module);
                const class = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                errdefer allocator.free(class);
                try results.append(.{ .global = .{ .module = module, .class = class } });
            },
            .dict => try results.append(.{ .dict = {} }),
            .empty_dict => try results.append(.{ .empty_dict = {} }),
            .appends => try results.append(.{ .appends = {} }),
            .get => {
                const buf = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                errdefer allocator.free(buf);
                try results.append(.{ .get = buf });
            },
            .binget => try results.append(.{ .binget = try reader.readByte() }),
            .long_binget => try results.append(.{ .long_binget = try reader.readInt(u32, .little) }),
            .list => try results.append(.{ .list = {} }),
            .empty_list => try results.append(.{ .empty_list = {} }),
            .obj => try results.append(.{ .obj = {} }),
            .put => {
                const buf = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                errdefer allocator.free(buf);
                try results.append(.{ .put = buf });
            },
            .binput => {
                try results.append(.{ .binput = try reader.readByte() });
            },
            .long_binput => {
                try results.append(.{ .long_binput = try reader.readInt(u32, .little) });
            },
            .setitem => try results.append(.{ .setitem = {} }),
            .tuple => try results.append(.{ .tuple = {} }),
            .empty_tuple => try results.append(.{ .empty_tuple = {} }),
            .setitems => try results.append(.{ .setitems = {} }),
            .binfloat => try results.append(.{ .binfloat = @bitCast(try reader.readInt(u64, .big)) }),
            .proto => try results.append(.{ .proto = try reader.readByte() }),
            .newobj => try results.append(.{ .newobj = {} }),
            .ext1 => try results.append(.{ .ext1 = try reader.readByte() }),
            .ext2 => try results.append(.{ .ext2 = try reader.readInt(i16, .little) }),
            .ext4 => try results.append(.{ .ext4 = try reader.readInt(i32, .little) }),
            .tuple1 => try results.append(.{ .tuple1 = {} }),
            .tuple2 => try results.append(.{ .tuple2 = {} }),
            .tuple3 => try results.append(.{ .tuple3 = {} }),
            .newtrue => try results.append(.{ .newtrue = {} }),
            .newfalse => try results.append(.{ .newfalse = {} }),
            .long1 => {
                const str_len = try reader.readByte();
                const buf = try allocator.alloc(u8, str_len);
                errdefer allocator.free(buf);
                _ = try reader.read(buf);
                try results.append(.{ .long1 = buf });
            },
            .long4 => {
                const str_len = try reader.readInt(u32, .little);
                const buf = try allocator.alloc(u8, str_len);
                errdefer allocator.free(buf);
                _ = try reader.read(buf);
                try results.append(.{ .long4 = buf });
            },
            .binbytes => {
                const str_len = try reader.readInt(u32, .little);
                const buf = try allocator.alloc(u8, str_len);
                errdefer allocator.free(buf);
                _ = try reader.read(buf);
                try results.append(.{ .binbytes = buf });
            },
            .binbytes8 => {
                const str_len = try reader.readInt(u64, .little);
                const buf = try allocator.alloc(u8, str_len);
                errdefer allocator.free(buf);
                _ = try reader.read(buf);
                try results.append(.{ .binbytes8 = buf });
            },
            .short_binbytes => {
                const str_len = try reader.readByte();
                const buf = try allocator.alloc(u8, str_len);
                errdefer allocator.free(buf);
                _ = try reader.read(buf);
                try results.append(.{ .short_binbytes = buf });
            },
            .binunicode8 => {
                const str_len = try reader.readInt(u64, .little);
                const buf = try allocator.alloc(u8, str_len);
                errdefer allocator.free(buf);
                _ = try reader.read(buf);
                try results.append(.{ .binunicode8 = buf });
            },
            .short_binunicode => {
                const str_len = try reader.readByte();
                const buf = try allocator.alloc(u8, str_len);
                errdefer allocator.free(buf);
                _ = try reader.read(buf);
                try results.append(.{ .short_binunicode = buf });
            },
            .empty_set => try results.append(.{ .empty_set = {} }),
            .additems => try results.append(.{ .additems = {} }),
            .frozenset => try results.append(.{ .frozenset = {} }),
            .newobj_ex => try results.append(.{ .newobj_ex = {} }),
            .stack_global => try results.append(.{ .stack_global = {} }),
            .memoize => try results.append(.{ .memoize = {} }),
            .frame => try results.append(.{ .frame = try reader.readInt(u64, .little) }),
            .bytearray8 => {
                const str_len = try reader.readInt(u64, .little);
                const buf = try allocator.alloc(u8, str_len);
                errdefer allocator.free(buf);
                _ = try reader.read(buf);
                try results.append(.{ .bytearray8 = buf });
            },
            .next_buffer => try results.append(.{ .next_buffer = {} }),
            .readonly_buffer => try results.append(.{ .readonly_buffer = {} }),
            _ => {},
        }
    }
    return results.toOwnedSlice();
}

test parse {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const file = try std.fs.cwd().openFile("zml/aio/torch/simple_test.pickle", .{ .mode = .read_only });
    var buffered_reader = std.io.bufferedReader(file.reader());
    const ops = try parse(allocator, buffered_reader.reader(), 4096);

    try std.testing.expect(ops.len == 35);
    // this can be obtained by running: `python -m pickletools simple_test.pickle`
    const expected = [_]Op{
        .{ .proto = 4 },
        .{ .frame = 83 },
        .empty_dict,
        .memoize,
        .mark,
        .{ .short_binunicode = "hello" },
        .memoize,
        .{ .short_binunicode = "world" },
        .memoize,
        .{ .short_binunicode = "int" },
        .memoize,
        .{ .binint1 = 1 },
        .{ .short_binunicode = "float" },
        .memoize,
        .{ .binfloat = 3.141592 },
        .{ .short_binunicode = "list" },
        .memoize,
        .empty_list,
        .memoize,
        .mark,
        .{ .binint1 = 0 },
        .{ .binint1 = 1 },
        .{ .binint1 = 2 },
        .{ .binint1 = 3 },
        .{ .binint1 = 4 },
        .appends,
        .{ .short_binunicode = "tuple" },
        .memoize,
        .{ .short_binunicode = "a" },
        .memoize,
        .{ .binint1 = 10 },
        .tuple2,
        .memoize,
        .setitems,
        .stop,
    };
    try std.testing.expectEqualDeep(&expected, ops);
}
