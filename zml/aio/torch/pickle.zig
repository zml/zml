const std = @import("std");

const log = std.log.scoped(.@"zml/aio");

/// All possible pickle operators.
/// Reference: https://github.com/python/cpython/blob/3.13/Lib/pickletools.py
pub const OpCode = enum(u8) {
    /// Push an integer or bool.
    ///
    /// The argument is a newline-terminated decimal literal string.
    ///
    /// The intent may have been that this always fit in a short Python int,
    /// but INT can be generated in pickles written on a 64-bit box that
    /// require a Python long on a 32-bit box. The difference between this
    /// and LONG then is that INT skips a trailing 'L', and produces a short
    /// int whenever possible.
    ///
    /// Another difference is due to that, when bool was introduced as a
    /// distinct type in 2.3, builtin names True and False were also added to
    /// 2.2.2, mapping to ints 1 and 0. For compatibility in both directions,
    /// True gets pickled as INT + "I01\n", and False as INT + "I00\n".
    /// Leading zeroes are never produced for a genuine integer. The 2.3
    /// (and later) unpicklers special-case these and return bool instead;
    /// earlier unpicklers ignore the leading "0" and return the int.
    int = 'I',

    /// Push a four-byte signed integer.
    /// Introduced in protocol 1.
    ///
    /// This handles the full range of Python (short) integers on a 32-bit
    /// box, directly as binary bytes (1 for the opcode and 4 for the integer).
    /// If the integer is non-negative and fits in 1 or 2 bytes, pickling via
    /// BININT1 or BININT2 saves space.
    binint = 'J',

    /// Push a one-byte unsigned integer.
    /// Introduced in protocol 1.
    ///
    /// This is a space optimization for pickling very small non-negative ints,
    /// in range(256).
    binint1 = 'K',

    /// Push a two-byte unsigned integer.
    /// Introduced in protocol 1.
    ///
    /// This is a space optimization for pickling small positive ints, in
    /// range(256, 2**16). Integers in range(256) can also be pickled via
    /// BININT2, but BININT1 instead saves a byte.
    binint2 = 'M',

    /// Push a long integer.
    ///
    /// The same as INT, except that the literal ends with 'L', and always
    /// unpickles to a Python long. There doesn't seem a real purpose to the
    /// trailing 'L'.
    ///
    /// Note that LONG takes time quadratic in the number of digits when
    /// unpickling (this is simply due to the nature of decimal->binary
    /// conversion). Proto 2 added linear-time (in C; still quadratic-time
    /// in Python) LONG1 and LONG4 opcodes.
    long = 'L',

    /// Long integer using one-byte length.
    /// Introduced in protocol 2.
    ///
    /// A more efficient encoding of a Python long; the long1 encoding
    long1 = 0x8a,

    /// Long integer using four-byte length.
    /// Introduced in protocol 2.
    ///
    /// A more efficient encoding of a Python long; the long4 encoding
    long4 = 0x8b,

    /// Push a Python string object.
    ///
    /// The argument is a repr-style string, with bracketing quote characters,
    /// and perhaps embedded escapes. The argument extends until the next
    /// newline character. These are usually decoded into a str instance
    /// using the encoding given to the Unpickler constructor. or the default,
    /// 'ASCII'. If the encoding given was 'bytes' however, they will be
    /// decoded as bytes object instead.
    string = 'S',

    /// Push a Python string object.
    /// Introduced in protocol 1.
    ///
    /// There are two arguments: the first is a 4-byte little-endian
    /// signed int giving the number of bytes in the string, and the
    /// second is that many bytes, which are taken literally as the string
    /// content. These are usually decoded into a str instance using the
    /// encoding given to the Unpickler constructor. or the default,
    /// 'ASCII'. If the encoding given was 'bytes' however, they will be
    /// decoded as bytes object instead.
    binstring = 'T',

    /// Push a Python string object.
    /// Introduced in protocol 1.
    ///
    /// There are two arguments: the first is a 1-byte unsigned int giving
    /// the number of bytes in the string, and the second is that many
    /// bytes, which are taken literally as the string content. These are
    /// usually decoded into a str instance using the encoding given to
    /// the Unpickler constructor. or the default, 'ASCII'. If the
    /// encoding given was 'bytes' however, they will be decoded as bytes
    /// object instead.
    short_binstring = 'U',

    /// Push a Python bytes object.
    /// Introduced in protocol 3.
    ///
    /// There are two arguments:  the first is a 4-byte little-endian unsigned int
    /// giving the number of bytes, and the second is that many bytes, which are
    /// taken literally as the bytes content.
    binbytes = 'B',

    /// Push a Python bytes object.
    /// Introduced in protocol 3.
    ///
    /// There are two arguments:  the first is a 1-byte unsigned int giving
    /// the number of bytes, and the second is that many bytes, which are taken
    /// literally as the string content.
    short_binbytes = 'C',

    /// Push a Python bytes object.
    /// Introduced in protocol 4.
    ///
    /// There are two arguments:  the first is an 8-byte unsigned int giving
    /// the number of bytes in the string, and the second is that many bytes,
    /// which are taken literally as the string content.
    binbytes8 = 0x8e,

    /// Push a Python bytearray object.
    /// Introduced in protocol 5.
    ///
    /// There are two arguments:  the first is an 8-byte unsigned int giving
    /// the number of bytes in the bytearray, and the second is that many bytes,
    /// which are taken literally as the bytearray content.
    bytearray8 = 0x96,

    /// Introduced in protocol 5.
    next_buffer = 0x97,

    /// Introduced in protocol 5.
    readonly_buffer = 0x98,

    none = 'N',

    /// Introduced in protocol 2.
    newtrue = 0x88,

    /// Introduced in protocol 2.
    newfalse = 0x89,

    /// Push a Python Unicode string object.
    ///
    /// The argument is a raw-unicode-escape encoding of a Unicode string,
    /// and so may contain embedded escape sequences. The argument extends
    /// until the next newline character.
    unicode = 'V',

    /// Push a Python Unicode string object.
    /// Introduced in protocol 4.
    ///
    /// There are two arguments:  the first is a 1-byte little-endian signed int
    /// giving the number of bytes in the string. The second is that many
    /// bytes, and is the UTF-8 encoding of the Unicode string.
    short_binunicode = 0x8c,

    /// Push a Python Unicode string object.
    /// Introduced in protocol 1.
    ///
    /// There are two arguments:  the first is a 4-byte little-endian unsigned int
    /// giving the number of bytes in the string. The second is that many
    /// bytes, and is the UTF-8 encoding of the Unicode string.
    binunicode = 'X',

    /// Push a Python Unicode string object.
    /// Introduced in protocol 4.
    ///
    /// There are two arguments:  the first is an 8-byte little-endian signed int
    /// giving the number of bytes in the string. The second is that many
    /// bytes, and is the UTF-8 encoding of the Unicode string.
    binunicode8 = 0x8d,

    /// Newline-terminated decimal float literal.
    ///
    /// The argument is repr(a_float), and in general requires 17 significant
    /// digits for roundtrip conversion to be an identity (this is so for
    /// IEEE-754 double precision values, which is what Python float maps to
    /// on most boxes).
    ///
    /// In general, FLOAT cannot be used to transport infinities, NaNs, or
    /// minus zero across boxes (or even on a single box, if the platform C
    /// library can't read the strings it produces for such things -- Windows
    /// is like that), but may do less damage than BINFLOAT on boxes with
    /// greater precision or dynamic range than IEEE-754 double.
    float = 'F',

    /// Float stored in binary form, with 8 bytes of data.
    /// Introduced in protocol 1.
    ///
    /// This generally requires less than half the space of FLOAT encoding.
    /// In general, BINFLOAT cannot be used to transport infinities, NaNs, or
    /// minus zero, raises an exception if the exponent exceeds the range of
    /// an IEEE-754 double, and retains no more than 53 bits of precision (if
    /// there are more than that, "add a half and chop" rounding is used to
    /// cut it back to 53 significant bits).
    binfloat = 'G',

    /// Introduced in protocol 1.
    empty_list = ']',

    /// Append an object to a list.
    ///
    /// Stack before:  ... pylist anyobject
    /// Stack after:   ... pylist+[anyobject]
    ///
    /// although pylist is really extended in-place.
    append = 'a',

    /// Extend a list by a slice of stack objects.
    /// Introduced in protocol 1.
    ///
    /// Stack before:  ... pylist markobject stackslice
    /// Stack after:   ... pylist+stackslice
    ///
    /// although pylist is really extended in-place.
    appends = 'e',

    /// Build a list out of the topmost stack slice, after markobject.
    ///
    /// All the stack entries following the topmost markobject are placed into
    /// a single Python list, which single list object replaces all of the
    /// stack from the topmost markobject onward. For example,
    ///
    /// Stack before: ... markobject 1 2 3 'abc'
    /// Stack after:  ... [1, 2, 3, 'abc']
    list = 'l',

    /// Introduced in protocol 1.
    empty_tuple = ')',

    /// Build a tuple out of the topmost stack slice, after markobject.
    ///
    /// All the stack entries following the topmost markobject are placed into
    /// a single Python tuple, which single tuple object replaces all of the
    /// stack from the topmost markobject onward. For example,
    ///
    /// Stack before: ... markobject 1 2 3 'abc'
    /// Stack after:  ... (1, 2, 3, 'abc')
    tuple = 't',

    /// Build a one-tuple out of the topmost item on the stack.
    /// Introduced in protocol 2.
    ///
    /// This code pops one value off the stack and pushes a tuple of
    /// length 1 whose one item is that value back onto it. In other
    /// words:
    ///
    ///     stack[-1] = tuple(stack[-1:])
    tuple1 = 0x85,

    /// Build a two-tuple out of the top two items on the stack.
    /// Introduced in protocol 2.
    ///
    /// This code pops two values off the stack and pushes a tuple of
    /// length 2 whose items are those values back onto it. In other
    /// words:
    ///
    ///     stack[-2:] = [tuple(stack[-2:])]
    tuple2 = 0x86,

    /// Build a three-tuple out of the top three items on the stack.
    /// Introduced in protocol 2.
    ///
    /// This code pops three values off the stack and pushes a tuple of
    /// length 3 whose items are those values back onto it. In other
    /// words:
    ///
    ///     stack[-3:] = [tuple(stack[-3:])]
    tuple3 = 0x87,

    /// Introduced in protocol 1.
    empty_dict = '}',

    /// Build a dict out of the topmost stack slice, after markobject.
    ///
    /// All the stack entries following the topmost markobject are placed into
    /// a single Python dict, which single dict object replaces all of the
    /// stack from the topmost markobject onward. The stack slice alternates
    /// key, value, key, value, .... For example,
    ///
    /// Stack before: ... markobject 1 2 3 'abc'
    /// Stack after:  ... {1: 2, 3: 'abc'}
    dict = 'd',

    /// Add a key+value pair to an existing dict.
    ///
    /// Stack before:  ... pydict key value
    /// Stack after:   ... pydict
    ///
    /// where pydict has been modified via pydict[key] = value.
    setitem = 's',

    /// Add an arbitrary number of key+value pairs to an existing dict.
    /// Introduced in protocol 1.
    ///
    /// The slice of the stack following the topmost markobject is taken as
    /// an alternating sequence of keys and values, added to the dict
    /// immediately under the topmost markobject. Everything at and after the
    /// topmost markobject is popped, leaving the mutated dict at the top
    /// of the stack.
    ///
    /// Stack before:  ... pydict markobject key_1 value_1 ... key_n value_n
    /// Stack after:   ... pydict
    ///
    /// where pydict has been modified via pydict[key_i] = value_i for i in
    /// 1, 2, ..., n, and in that order.
    setitems = 'u',

    /// Introduced in protocol 4.
    empty_set = 0x8f,

    /// Add an arbitrary number of items to an existing set.
    /// Introduced in protocol 4.
    ///
    /// The slice of the stack following the topmost markobject is taken as
    /// a sequence of items, added to the set immediately under the topmost
    /// markobject. Everything at and after the topmost markobject is popped,
    /// leaving the mutated set at the top of the stack.
    ///
    /// Stack before:  ... pyset markobject item_1 ... item_n
    /// Stack after:   ... pyset
    ///
    /// where pyset has been modified via pyset.add(item_i) = item_i for i in
    /// 1, 2, ..., n, and in that order.
    additems = 0x90,

    /// Build a frozenset out of the topmost slice, after markobject.
    /// Introduced in protocol 4.
    ///
    /// All the stack entries following the topmost markobject are placed into
    /// a single Python frozenset, which single frozenset object replaces all
    /// of the stack from the topmost markobject onward. For example,
    ///
    /// Stack before: ... markobject 1 2 3
    /// Stack after:  ... frozenset({1, 2, 3})
    frozenset = 0x91,

    pop = '0',

    dup = '2',

    /// Push markobject onto the stack.
    ///
    /// markobject is a unique object, used by other opcodes to identify a
    /// region of the stack containing a variable number of objects for them
    /// to work on. See markobject.doc for more detail.
    mark = '(',

    /// Pop all the stack objects at and above the topmost markobject.
    /// Introduced in protocol 1.
    ///
    /// When an opcode using a variable number of stack objects is done,
    /// POP_MARK is used to remove those objects, and to remove the markobject
    /// that delimited their starting position on the stack.
    pop_mark = '1',

    /// Read an object from the memo and push it on the stack.
    ///
    /// The index of the memo object to push is given by the newline-terminated
    /// decimal string following. BINGET and LONG_BINGET are space-optimized
    /// versions.
    get = 'g',

    /// Read an object from the memo and push it on the stack.
    /// Introduced in protocol 1.
    ///
    /// The index of the memo object to push is given by the 1-byte unsigned
    /// integer following.
    binget = 'h',

    /// Read an object from the memo and push it on the stack.
    /// Introduced in protocol 1.
    ///
    /// The index of the memo object to push is given by the 4-byte unsigned
    /// little-endian integer following.
    long_binget = 'j',

    /// Store the stack top into the memo. The stack is not popped.
    ///
    /// The index of the memo location to write into is given by the newline-
    /// terminated decimal string following. BINPUT and LONG_BINPUT are
    /// space-optimized versions.
    put = 'p',

    /// Store the stack top into the memo. The stack is not popped.
    /// Introduced in protocol 1.
    ///
    /// The index of the memo location to write into is given by the 1-byte
    /// unsigned integer following.
    binput = 'q',

    /// Store the stack top into the memo. The stack is not popped.
    /// Introduced in protocol 1.
    ///
    /// The index of the memo location to write into is given by the 4-byte
    /// unsigned little-endian integer following.
    long_binput = 'r',

    /// Store the stack top into the memo. The stack is not popped.
    /// Introduced in protocol 4.
    ///
    /// The index of the memo location to write is the number of
    /// elements currently present in the memo.
    memoize = 0x94,

    /// Extension code.
    /// Introduced in protocol 2.
    ///
    /// This code and the similar EXT2 and EXT4 allow using a registry
    /// of popular objects that are pickled by name, typically classes.
    /// It is envisioned that through a global negotiation and
    /// registration process, third parties can set up a mapping between
    /// ints and object names.
    ///
    /// In order to guarantee pickle interchangeability, the extension
    /// code registry ought to be global, although a range of codes may
    /// be reserved for private use.
    ///
    /// EXT1 has a 1-byte integer argument. This is used to index into the
    /// extension registry, and the object at that index is pushed on the stack.
    ext1 = 0x82,

    /// Extension code.
    /// Introduced in protocol 2.
    ///
    /// See EXT1. EXT2 has a two-byte integer argument.
    ext2 = 0x83,

    /// Extension code.
    /// Introduced in protocol 2.
    ///
    /// See EXT1. EXT4 has a four-byte integer argument.
    ext4 = 0x84,

    /// Push a global object (module.attr) on the stack.
    ///
    /// Two newline-terminated strings follow the GLOBAL opcode. The first is
    /// taken as a module name, and the second as a class name. The class
    /// object module.class is pushed on the stack. More accurately, the
    /// object returned by self.find_class(module, class) is pushed on the
    /// stack, so unpickling subclasses can override this form of lookup.
    global = 'c',

    /// Push a global object (module.attr) on the stack.
    /// Introduced in protocol 4.
    stack_global = 0x93,

    /// Push an object built from a callable and an argument tuple.
    ///
    /// The opcode is named to remind of the __reduce__() method.
    ///
    /// Stack before: ... callable pytuple
    /// Stack after:  ... callable(*pytuple)
    ///
    /// The callable and the argument tuple are the first two items returned
    /// by a __reduce__ method. Applying the callable to the argtuple is
    /// supposed to reproduce the original object, or at least get it started.
    /// If the __reduce__ method returns a 3-tuple, the last component is an
    /// argument to be passed to the object's __setstate__, and then the REDUCE
    /// opcode is followed by code to create setstate's argument, and then a
    /// BUILD opcode to apply  __setstate__ to that argument.
    ///
    /// If not isinstance(callable, type), REDUCE complains unless the
    /// callable has been registered with the copyreg module's
    /// safe_constructors dict, or the callable has a magic
    /// '__safe_for_unpickling__' attribute with a true value. I'm not sure
    /// why it does this, but I've sure seen this complaint often enough when
    /// I didn't want to <wink>.
    reduce = 'R',

    /// Finish building an object, via __setstate__ or dict update.
    ///
    /// Stack before: ... anyobject argument
    /// Stack after:  ... anyobject
    ///
    /// where anyobject may have been mutated, as follows:
    ///
    /// If the object has a __setstate__ method,
    ///
    ///     anyobject.__setstate__(argument)
    ///
    /// is called.
    ///
    /// Else the argument must be a dict, the object must have a __dict__, and
    /// the object is updated via
    ///
    ///     anyobject.__dict__.update(argument)
    build = 'b',

    /// Build a class instance.
    ///
    /// This is the protocol 0 version of protocol 1's OBJ opcode.
    /// INST is followed by two newline-terminated strings, giving a
    /// module and class name, just as for the GLOBAL opcode (and see
    /// GLOBAL for more details about that). self.find_class(module, name)
    /// is used to get a class object.
    ///
    /// In addition, all the objects on the stack following the topmost
    /// markobject are gathered into a tuple and popped (along with the
    /// topmost markobject), just as for the TUPLE opcode.
    ///
    /// Now it gets complicated. If all of these are true:
    ///
    ///   + The argtuple is empty (markobject was at the top of the stack
    ///     at the start).
    ///
    ///   + The class object does not have a __getinitargs__ attribute.
    ///
    /// then we want to create an old-style class instance without invoking
    /// its __init__() method (pickle has waffled on this over the years; not
    /// calling __init__() is current wisdom). In this case, an instance of
    /// an old-style dummy class is created, and then we try to rebind its
    /// __class__ attribute to the desired class object. If this succeeds,
    /// the new instance object is pushed on the stack, and we're done.
    ///
    /// Else (the argtuple is not empty, it's not an old-style class object,
    /// or the class object does have a __getinitargs__ attribute), the code
    /// first insists that the class object have a __safe_for_unpickling__
    /// attribute. Unlike as for the __safe_for_unpickling__ check in REDUCE,
    /// it doesn't matter whether this attribute has a true or false value, it
    /// only matters whether it exists (XXX this is a bug). If
    /// __safe_for_unpickling__ doesn't exist, UnpicklingError is raised.
    ///
    /// Else (the class object does have a __safe_for_unpickling__ attr),
    /// the class object obtained from INST's arguments is applied to the
    /// argtuple obtained from the stack, and the resulting instance object
    /// is pushed on the stack.
    ///
    /// NOTE:  checks for __safe_for_unpickling__ went away in Python 2.3.
    /// NOTE:  the distinction between old-style and new-style classes does
    ///        not make sense in Python 3.
    inst = 'i',

    /// Build a class instance.
    /// Introduced in protocol 1.
    ///
    /// This is the protocol 1 version of protocol 0's INST opcode, and is
    /// very much like it. The major difference is that the class object
    /// is taken off the stack, allowing it to be retrieved from the memo
    /// repeatedly if several instances of the same class are created. This
    /// can be much more efficient (in both time and space) than repeatedly
    /// embedding the module and class names in INST opcodes.
    ///
    /// Unlike INST, OBJ takes no arguments from the opcode stream. Instead
    /// the class object is taken off the stack, immediately above the
    /// topmost markobject:
    ///
    /// Stack before: ... markobject classobject stackslice
    /// Stack after:  ... new_instance_object
    ///
    /// As for INST, the remainder of the stack above the markobject is
    /// gathered into an argument tuple, and then the logic seems identical,
    /// except that no __safe_for_unpickling__ check is done (XXX this is
    /// a bug). See INST for the gory details.
    ///
    /// NOTE:  In Python 2.3, INST and OBJ are identical except for how they
    /// get the class object. That was always the intent; the implementations
    /// had diverged for accidental reasons.
    obj = 'o',

    /// Build an object instance.
    /// Introduced in protocol 2.
    ///
    /// The stack before should be thought of as containing a class
    /// object followed by an argument tuple (the tuple being the stack
    /// top). Call these cls and args. They are popped off the stack,
    /// and the value returned by cls.__new__(cls, *args) is pushed back
    /// onto the stack.
    newobj = 0x81,

    /// Build an object instance.
    /// Introduced in protocol 4.
    ///
    /// The stack before should be thought of as containing a class
    /// object followed by an argument tuple and by a keyword argument dict
    /// (the dict being the stack top). Call these cls and args. They are
    /// popped off the stack, and the value returned by
    /// cls.__new__(cls, *args, *kwargs) is  pushed back  onto the stack.
    newobj_ex = 0x92,

    /// Protocol version indicator.
    /// Introduced in protocol 2.
    ///
    /// For protocol 2 and above, a pickle must start with this opcode.
    /// The argument is the protocol version, an int in range(2, 256).
    proto = 0x80,

    /// Stop the unpickling machine.
    ///
    /// Every pickle ends with this opcode. The object at the top of the stack
    /// is popped, and that's the result of unpickling. The stack should be
    /// empty then.
    stop = '.',

    /// Indicate the beginning of a new frame.
    /// Introduced in protocol 4.
    ///
    /// The unpickler may use this opcode to safely prefetch data from its
    /// underlying stream.
    frame = 0x95,

    /// Push an object identified by a persistent ID.
    ///
    /// The pickle module doesn't define what a persistent ID means. PERSID's
    /// argument is a newline-terminated str-style (no embedded escapes, no
    /// bracketing quote characters) string, which *is* "the persistent ID".
    /// The unpickler passes this string to self.persistent_load(). Whatever
    /// object that returns is pushed on the stack. There is no implementation
    /// of persistent_load() in Python's unpickler:  it must be supplied by an
    /// unpickler subclass.
    persid = 'P',

    /// Push an object identified by a persistent ID.
    /// Introduced in protocol 1.
    ///
    /// Like PERSID, except the persistent ID is popped off the stack (instead
    /// of being a string embedded in the opcode bytestream). The persistent
    /// ID is passed to self.persistent_load(), and whatever object that
    /// returns is pushed on the stack. See PERSID for more detail.
    binpersid = 'Q',

    _,
};

// The above enum was generated with the following Python code,
// run inside pickletools.py
//
// def generate_zig():
//     print("""/// All possible pickle operators.
// /// Reference: https://github.com/python/cpython/blob/3.13/Lib/pickletools.py
// pub const OpCode = enum(u8) {
// """)
//     for op in opcodes:
//         lines = [_cleanup(l) for l in op.doc.split("\n")[:-1]]
//         if op.proto > 0:
//             lines.insert(1, _cleanup(f"Introduced in protocol {op.proto}."))
//         doc = "\n".join(lines)
//         op_code = op.code.__repr__()
//         if op_code.startswith("'\\x"):
//             op_code = "0x" + op_code[3:-1]
//         print(f"""{doc}
//     {op.name.lower()} = {op_code},
// """)
//
//     print("    _,")
//     print("};")
//
// def _cleanup(line: str) -> str:
//     indent = "      "
//     if (line.startswith(indent)):
//         line = line[len(indent):]
//     line = line.replace(".  ", ". ")
//     if line:
//         line = " " + line
//     line = "    ///" + line
//     return line

/// A decoded Pickle operation in its natural state.
/// This is a bit different from Op enum,
/// because operators having same semantics, but different encoding have been merged.
/// ex: string, binstring, short_binstring -> string.
pub const Op = union(enum) {
    int: i32,
    // Python can represent arbitrary long integers
    long: []const u8,
    binlong: []const u8,
    string: []const u8,
    bytes: []const u8,
    bytearray: []u8,
    next_buffer,
    readonly_buffer,
    none,
    bool: bool,
    unicode: []const u8,
    float: []const u8,
    binfloat: f64,
    empty_list,
    append,
    appends,
    list,
    empty_tuple,
    tuple,
    tuple1,
    tuple2,
    tuple3,
    empty_dict,
    dict,
    setitem,
    setitems,
    empty_set,
    additems,
    frozenset,
    pop,
    dup,
    mark,
    pop_mark,
    get: u32,
    put: u32,
    memoize,
    ext1: u8,
    ext2: i16,
    ext4: i32,
    global: PyType,
    stack_global,
    reduce,
    build,
    inst: PyType,
    obj,
    newobj,
    newobj_ex,
    proto: u8,
    stop,
    frame: u64, // new frame and its size
    persid: []const u8,
    binpersid,

    pub const PyType = struct { module: []const u8, class: []const u8 };

    pub fn deinit(self: Op, allocator: std.mem.Allocator) void {
        switch (self) {
            // Use a switch on the type of the stored data,
            // this is easier than listing every opcode.
            inline else => |v| switch (@TypeOf(v)) {
                void, bool, u8, u16, u32, u64, i16, i32, f64 => {},
                []const u8, []u8 => allocator.free(v),
                PyType => {
                    allocator.free(v.module);
                    allocator.free(v.class);
                },
                else => @compileError("please explicit how to free this new opcode: " ++ @typeName(@TypeOf(v))),
            },
        }
    }

    pub fn clone(self: Op, allocator: std.mem.Allocator) !Op {
        return switch (self) {
            // Use a switch on the type of the stored data,
            // this is easier than listing every opcode.
            inline else => |v, tag| switch (@TypeOf(v)) {
                void, bool, u8, u16, u32, u64, i16, i32, f64 => self,
                []const u8, []u8 => @unionInit(Op, @tagName(tag), try allocator.dupe(u8, v)),
                PyType => @unionInit(Op, @tagName(tag), .{
                    .module = try allocator.dupe(u8, v.module),
                    .class = try allocator.dupe(u8, v.class),
                }),
                else => @compileError("please explicit how to close this new opcode: " ++ @typeName(@TypeOf(v))),
            },
        };
    }
};

/// Read a stream of bytes, and interpret it as a stream of Pickle operators.
pub fn parse(allocator: std.mem.Allocator, reader: anytype, max_line_len: usize) ![]const Op {
    var results = std.ArrayList(Op).init(allocator);
    errdefer results.deinit();
    const len = max_line_len;
    var _buf: std.BoundedArray(u8, 12) = .{};

    while (true) {
        const b = try reader.readByte();
        const code: OpCode = @enumFromInt(b);
        const op: Op = switch (code) {
            .int => blk: {
                _buf.len = 0;
                try reader.streamUntilDelimiter(_buf.writer(), '\n', _buf.capacity() + 1);
                const buf = _buf.constSlice();
                // Legacy hack, see OpCode.int documentation
                // We do this parsing right away to simplify downstream code.
                break :blk if (std.mem.eql(u8, "00", buf))
                    .{ .bool = false }
                else if (std.mem.eql(u8, "01", buf))
                    .{ .bool = true }
                else
                    .{ .int = try std.fmt.parseInt(i32, buf, 10) };
            },
            .binint => .{ .int = try reader.readInt(i32, .little) },
            .binint1 => .{ .int = try reader.readByte() },
            .binint2 => .{ .int = try reader.readInt(u16, .little) },
            // TODO: long should handle the trailing 'L' -> add a test.
            .long => .{ .long = try reader.readUntilDelimiterAlloc(allocator, '\n', len) },
            .long1 => .{ .binlong = try _readSlice(reader, allocator, 1) },
            .long4 => .{ .binlong = try _readSlice(reader, allocator, 4) },
            .string => .{ .string = try reader.readUntilDelimiterAlloc(allocator, '\n', len) },
            .binstring => .{ .string = try _readSlice(reader, allocator, 4) },
            .short_binstring => .{ .string = try _readSlice(reader, allocator, 1) },
            .binbytes => .{ .bytes = try _readSlice(reader, allocator, 4) },
            .binbytes8 => .{ .bytes = try _readSlice(reader, allocator, 8) },
            .short_binbytes => .{ .bytes = try _readSlice(reader, allocator, 1) },
            .bytearray8 => .{ .bytearray = try _readSlice(reader, allocator, 8) },
            .next_buffer => .next_buffer,
            .readonly_buffer => .readonly_buffer,
            .none => .none,
            .newtrue => .{ .bool = true },
            .newfalse => .{ .bool = false },
            .unicode => .{ .unicode = try reader.readUntilDelimiterAlloc(allocator, '\n', len) },
            .short_binunicode => .{ .unicode = try _readSlice(reader, allocator, 1) },
            .binunicode => .{ .unicode = try _readSlice(reader, allocator, 4) },
            .binunicode8 => .{ .unicode = try _readSlice(reader, allocator, 8) },
            .float => .{ .float = try reader.readUntilDelimiterAlloc(allocator, '\n', len) },
            .binfloat => .{ .binfloat = @bitCast(try reader.readInt(u64, .big)) },
            .empty_list => .empty_list,
            .append => .append,
            .appends => .appends,
            .list => .list,
            .empty_tuple => .empty_tuple,
            .tuple => .tuple,
            .tuple1 => .tuple1,
            .tuple2 => .tuple2,
            .tuple3 => .tuple3,
            .empty_dict => .empty_dict,
            .dict => .dict,
            .setitem => .setitem,
            .setitems => .setitems,
            .empty_set => .empty_set,
            .additems => .additems,
            .frozenset => .frozenset,
            .pop => .pop,
            .dup => .dup,
            .mark => .mark,
            .pop_mark => .pop_mark,
            // If we fail to parse delay the error to the evaluation.
            .get => .{
                .get = _readDigits(u32, reader, &_buf) catch std.math.maxInt(u32),
            },
            .binget => .{ .get = try reader.readByte() },
            .long_binget => .{ .get = try reader.readInt(u32, .little) },
            .put => blk: {
                const buf = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                defer allocator.free(buf);
                const n = std.fmt.parseInt(u32, buf, 10) catch std.math.maxInt(u32);
                break :blk .{ .put = n };
            },
            .binput => .{ .put = try reader.readByte() },
            .long_binput => .{ .put = try reader.readInt(u32, .little) },
            .memoize => .memoize,
            .ext1 => .{ .ext1 = try reader.readByte() },
            .ext2 => .{ .ext2 = try reader.readInt(i16, .little) },
            .ext4 => .{ .ext4 = try reader.readInt(i32, .little) },
            .global => .{ .global = .{
                .module = try reader.readUntilDelimiterAlloc(allocator, '\n', len),
                .class = try reader.readUntilDelimiterAlloc(allocator, '\n', len),
            } },
            .stack_global => .stack_global,
            .reduce => .reduce,
            .build => .build,
            .inst => .{ .inst = .{
                .module = try reader.readUntilDelimiterAlloc(allocator, '\n', len),
                .class = try reader.readUntilDelimiterAlloc(allocator, '\n', len),
            } },
            .obj => .obj,
            .newobj => .newobj,
            .newobj_ex => .newobj_ex,
            .proto => blk: {
                const version = try reader.readByte();
                if (version > 5) log.warn("zml.aio.torch.pickle.parse expects a Python pickle object of version <=5, got version {}. Will try to interpret anyway, but this may lead to more errors.", .{version});
                break :blk .{ .proto = version };
            },
            .stop => .stop,
            // This is not documented in pickletools but in https://peps.python.org/pep-3154/
            // The frame size is stored right after the frame header.
            // The loader is allowed to prefetch framesize from the underlying reader,
            // and ops are not allowed to cross a frame boundary.
            // We don't prefetch because we assume the reader is going to use some kind of buffered reader.
            // We could try to enforce frame boundaries, but we would need to track
            // how many bytes we are reading from the stream.
            .frame => .{ .frame = try reader.readInt(u64, .little) },
            .persid => .{ .persid = try reader.readUntilDelimiterAlloc(allocator, '\n', len) },
            .binpersid => .binpersid,
            _ => |unk_tag| {
                log.err("Unknow pickle operator {}, note we are only supporting pickle protocol up to version 5.", .{unk_tag});
                return error.NotSupported;
            },
        };
        try results.append(op);
        if (op == .stop) break;
    }
    return results.toOwnedSlice();
}

test "parse protocol 4" {
    const allocator = std.testing.allocator;
    const file = try std.fs.cwd().openFile("zml/aio/torch/simple_test_4.pickle", .{ .mode = .read_only });
    var buffered_reader = std.io.bufferedReader(file.reader());
    const ops = try parse(allocator, buffered_reader.reader(), 4096);
    defer {
        // Test we are correctly freeing every allocation.
        for (ops) |op| op.deinit(allocator);
        allocator.free(ops);
    }

    // this can be obtained by running: `python -m pickletools simple_test_4.pickle`
    var expected = [_]Op{
        .{ .proto = 4 },
        .{ .frame = 119 },
        .empty_dict,
        .memoize,
        .mark,
        .{ .unicode = "hello" },
        .memoize,
        .{ .unicode = "world" },
        .memoize,
        .{ .unicode = "int" },
        .memoize,
        .{ .int = 1 },
        .{ .unicode = "float" },
        .memoize,
        .{ .binfloat = 3.141592 },
        .{ .unicode = "list" },
        .memoize,
        .empty_list,
        .memoize,
        .mark,
        .{ .int = 255 },
        .{ .int = 1234 },
        .{ .int = -123 },
        .{ .int = 1_000_000_000 },
        .{ .binlong = &writeIntBuff(u48, 999_000_000_000) },
        .{ .binlong = &writeIntBuff(u104, 999_000_000_000_000_000_000_000_000_000) },
        .appends,
        .{ .unicode = "bool" },
        .memoize,
        .{ .bool = false },
        .{ .unicode = "tuple" },
        .memoize,
        .{ .unicode = "a" },
        .memoize,
        .{ .int = 10 },
        .tuple2,
        .memoize,
        .setitems,
        .stop,
    };
    try std.testing.expectEqualDeep(&expected, ops);
}

test "parse protocol 0" {
    // We also test protocol 0, cause it's more text oriented.
    const allocator = std.testing.allocator;
    const pickle_0 =
        \\(dp0
        \\Vhello
        \\p1
        \\Vworld
        \\p2
        \\sVint
        \\p3
        \\I1
        \\sVfloat
        \\p4
        \\F3.141592
        \\sVlist
        \\p5
        \\(lp6
        \\I255
        \\aI1234
        \\aI-123
        \\aI1000000000
        \\aL999000000000L
        \\aL999000000000000000000000000000L
        \\asVbool
        \\p7
        \\I00
        \\sVtuple
        \\p8
        \\(Va
        \\p9
        \\I10
        \\tp10
        \\s.
    ;

    var stream = std.io.fixedBufferStream(pickle_0);
    const ops = try parse(allocator, stream.reader(), 4096);
    defer {
        // Test we are correctly freeing every allocation.
        for (ops) |op| op.deinit(allocator);
        allocator.free(ops);
    }

    var expected = [_]Op{
        .mark,
        .dict,
        .{ .put = 0 },
        .{ .unicode = "hello" },
        .{ .put = 1 },
        .{ .unicode = "world" },
        .{ .put = 2 },
        .setitem,
        .{ .unicode = "int" },
        .{ .put = 3 },
        .{ .int = 1 },
        .setitem,
        .{ .unicode = "float" },
        .{ .put = 4 },
        .{ .float = "3.141592" },
        .setitem,
        .{ .unicode = "list" },
        .{ .put = 5 },
        .mark,
        .list,
        .{ .put = 6 },
        .{ .int = 255 },
        .append,
        .{ .int = 1234 },
        .append,
        .{ .int = -123 },
        .append,
        .{ .int = 1_000_000_000 },
        .append,
        .{ .long = "999000000000L" },
        .append,
        .{ .long = "999000000000000000000000000000L" },
        .append,
        .setitem,
        .{ .unicode = "bool" },
        .{ .put = 7 },
        .{ .bool = false },
        .setitem,
        .{ .unicode = "tuple" },
        .{ .put = 8 },
        .mark,
        .{ .unicode = "a" },
        .{ .put = 9 },
        .{ .int = 10 },
        .tuple,
        .{ .put = 10 },
        .setitem,
        .stop,
    };
    try std.testing.expectEqualDeep(&expected, ops);
}

fn _readDigits(comptime T: type, reader: anytype, buffer: *std.BoundedArray(u8, 12)) !T {
    buffer.len = 0;
    try reader.streamUntilDelimiter(buffer.writer(), '\n', 13);
    return std.fmt.parseInt(T, buffer.constSlice(), 10);
}

fn _readSlice(reader: anytype, allocator: std.mem.Allocator, comptime len_bytes: u8) ![]u8 {
    const T = std.meta.Int(.unsigned, 8 * len_bytes);
    const str_len: u64 = try reader.readInt(T, .little);
    const buf = try allocator.alloc(u8, str_len);
    errdefer allocator.free(buf);
    _ = try reader.read(buf);
    return buf;
}

fn writeIntBuff(comptime T: type, value: T) [@divExact(@typeInfo(T).int.bits, 8)]u8 {
    var res: [@divExact(@typeInfo(T).int.bits, 8)]u8 = undefined;
    std.mem.writeInt(T, &res, value, .little);
    return res;
}
