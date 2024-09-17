const std = @import("std");

/// A decoded Pickle operation in its natural state.
pub const PickleOp = union(RawPickleOp) {
    mark,
    stop,
    pop,
    pop_mark,
    dup,
    float: []u8,
    int: []u8,
    binint: i32,
    binint1: u8,
    long: []u8,
    binint2: u16,
    none,
    persid: []u8,
    binpersid,
    reduce,
    string: []u8,
    binstring: []u8,
    short_binstring: []u8,
    unicode: []u8,
    binunicode: []u8,
    append,
    build,
    global: [2][]u8,
    dict,
    empty_dict,
    appends,
    get: []u8,
    binget: u8,
    inst: [2][]u8,
    long_binget: u32,
    list,
    empty_list,
    obj,
    put: []u8,
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
    long1: []u8,
    long4: []u8,
    binbytes: []u8,
    short_binbytes: []u8,
    short_binunicode: []u8,
    binunicode8: []u8,
    binbytes8: []u8,
    empty_set,
    additems,
    frozenset,
    newobj_ex,
    stack_global,
    memoize,
    frame: u64,
    bytearray8: []u8,
    next_buffer,
    readonly_buffer,

    pub fn deinit(self: PickleOp, allocator: std.mem.Allocator) void {
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
            .global, .inst => |fields| {
                inline for (fields) |field| {
                    allocator.free(field);
                }
            },
            else => {},
        }
    }

    pub fn clone(self: PickleOp, allocator: std.mem.Allocator) !PickleOp {
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
                var out: std.meta.Tuple(&.{ []u8, []u8 }) = undefined;
                inline for (0..2) |i| {
                    out[i] = try allocator.alloc(u8, v[i].len);
                    @memcpy(out[i], v[i]);
                }
                @field(res, @tagName(tag)) = out;
                return res;
            },
            else => self,
        };
    }
};

/// The values for the possible opcodes are in this enum.
pub const RawPickleOp = enum(u8) {
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
