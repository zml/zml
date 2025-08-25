const std = @import("std");

const c = @import("c");

const log = std.log.scoped(.mlir);

test {
    std.testing.refAllDeclsRecursive(@This());
}

const Error = error{
    /// Invalid Mlir was created.
    InvalidMlir,
    /// Another Mlir error. Check the log for more context.
    MlirUnexpected,
    /// A resource/executor was not found.
    NotFound,
} || std.mem.Allocator.Error;

pub inline fn stringRef(str: []const u8) c.MlirStringRef {
    return .{ .data = str.ptr, .length = str.len };
}

pub inline fn string(str: c.MlirStringRef) []const u8 {
    // Note: mlir.StringRef need not to be null terminated.
    return str.data[0..str.length];
}

pub const StringCallbackCtx = struct {
    writer: *std.Io.Writer,
    err: ?std.Io.Writer.Error = null,
};

pub fn stringCallback(str: c.MlirStringRef, ctx_: ?*anyopaque) callconv(.c) void {
    const ctx: *StringCallbackCtx = @ptrCast(@alignCast(ctx_));
    ctx.writer.writeAll(string(str)) catch |err| {
        ctx.err = err;
    };
}

pub fn registerPasses(comptime passes: []const u8) void {
    @field(c, "mlirRegister" ++ passes ++ "Passes")();
}

fn isPtrConst(comptime T: type) bool {
    return @typeInfo(T).pointer.is_const;
}

fn CastedPtr(comptime T1: type, comptime T2: type) type {
    return if (@typeInfo(T1).pointer.is_const) *const T2 else *T2;
}

pub fn StdMethods(comptime T: type, comptime NativeT: type) type {
    return struct {
        pub const IsPtrConst = @typeInfo(@typeInfo(@FieldType(NativeT, "ptr")).optional.child).pointer.is_const;
        pub const Ptr = if (IsPtrConst) *const T else *T;

        pub fn ptr(self: anytype) NativeT {
            return .{ .ptr = @ptrCast(@constCast(self)) };
        }

        fn format(f: anytype) fn (self: *const T, writer: *std.Io.Writer) std.Io.Writer.Error!void {
            return struct {
                fn format(self: *const T, writer: *std.Io.Writer) std.Io.Writer.Error!void {
                    var ctx: StringCallbackCtx = .{ .writer = writer };
                    f(self.ptr(), stringCallback, &ctx);
                    if (ctx.err) |err| {
                        return err;
                    }
                }
            }.format;
        }

        pub fn eql(f: anytype) fn (self: *const T, other: *const T) bool {
            return struct {
                fn eql(self: *const T, other: *const T) bool {
                    return f(self.ptr(), other.ptr());
                }
            }.eql;
        }

        pub fn isA(self: *const T, OtherType: type) ?CastedPtr(@TypeOf(self), OtherType) {
            return if (OtherType.isAFn(self.ptr())) @ptrCast(self) else null;
        }
    };
}

pub const DialectRegistry = opaque {
    const M = StdMethods(DialectRegistry, c.MlirDialectRegistry);

    pub const ptr = M.ptr;

    pub fn init() std.mem.Allocator.Error!*DialectRegistry {
        return @ptrCast(c.mlirDialectRegistryCreate().ptr orelse return std.mem.Allocator.Error.OutOfMemory);
    }

    pub fn deinit(self: *DialectRegistry) void {
        c.mlirDialectRegistryDestroy(self.ptr());
    }

    pub fn registerDialect(self: *DialectRegistry, comptime name: []const u8) void {
        DialectHandle.fromString(name).insertDialect(self);
    }
};

pub const Context = opaque {
    const M = StdMethods(Context, c.MlirContext);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirContextEqual);

    pub const InitOptions = struct {
        registry: ?*const DialectRegistry = null,
        threading: bool = false,
    };

    pub fn init(opts: InitOptions) std.mem.Allocator.Error!*Context {
        const ctx = if (opts.registry) |r| c.mlirContextCreateWithRegistry(
            r.ptr(),
            opts.threading,
        ) else c.mlirContextCreateWithThreading(opts.threading);
        return @ptrCast(ctx.ptr orelse return std.mem.Allocator.Error.OutOfMemory);
    }

    pub fn deinit(self: *Context) void {
        c.mlirContextDestroy(self.ptr());
    }

    pub fn setMultiThreading(self: *Context, enabled: bool) void {
        c.mlirContextEnableMultithreading(self.ptr(), enabled);
    }

    pub fn appendDialectRegistry(self: *Context, registry: *const DialectRegistry) void {
        c.mlirContextAppendDialectRegistry(self.ptr(), registry.ptr());
    }

    pub fn loadAllAvailableDialects(self: *Context) void {
        c.mlirContextLoadAllAvailableDialects(self.ptr());
    }

    pub fn numRegisteredDialects(self: *const Context) usize {
        return @intCast(c.mlirContextGetNumRegisteredDialects(self.ptr()));
    }

    pub fn numLoadedDialects(self: *const Context) usize {
        return @intCast(c.mlirContextGetNumLoadedDialects(self.ptr()));
    }

    pub fn numThreads(self: *const Context) usize {
        return @intCast(c.mlirContextGetNumThreads(self.ptr()));
    }

    pub fn isRegisteredOperation(self: *const Context, op: []const u8) bool {
        return c.mlirContextIsRegisteredOperation(self.ptr(), stringRef(op));
    }

    pub fn format(self: *const Context, writer: *std.Io.Writer) !void {
        try writer.print("{s}{{ .numThreads = {d}, .numRegisteredDialects = {d}, .numLoadedDialects = {d} }}", .{
            @typeName(Context),
            self.numThreads(),
            self.numRegisteredDialects(),
            self.numLoadedDialects(),
        });
    }
};

pub const Location = opaque {
    const M = StdMethods(Location, c.MlirLocation);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirLocationEqual);
    pub const format = M.format(c.mlirLocationPrint);

    pub fn fromSrc(ctx: *Context, src: std.builtin.SourceLocation) *const Location {
        return fileLineCol(
            ctx,
            src.file,
            src.line,
            src.column,
        );
    }

    pub fn fileLineCol(ctx: *Context, file: []const u8, line: usize, column: usize) *const Location {
        return @ptrCast(c.mlirLocationFileLineColGet(
            ctx.ptr(),
            stringRef(file),
            @intCast(line),
            @intCast(column),
        ).ptr);
    }

    pub fn callSite(callee: *const Location, caller: *const Location) *const Location {
        return @ptrCast(c.mlirLocationCallSiteGet(callee.ptr(), caller.ptr()).ptr);
    }

    pub fn fused(ctx: *Context, locs: []const *const Location, metadata: *const Attribute) *const Location {
        return @ptrCast(c.mlirLocationFusedGet(
            ctx.ptr(),
            @intCast(locs.len),
            @ptrCast(locs),
            metadata.ptr(),
        ).ptr);
    }

    pub fn named(self: *const Location, ctx: *Context, loc_name: []const u8) *const Location {
        return @ptrCast(c.mlirLocationNameGet(ctx.ptr(), stringRef(loc_name), self.ptr()).ptr);
    }

    pub fn namedFmt(loc: *const Location, ctx: *Context, comptime fmt: []const u8, args: anytype) *const Location {
        var buf: [256]u8 = undefined;
        return loc.named(loc, ctx, std.fmt.bufPrint(&buf, fmt, args) catch blk: {
            buf[buf.len - 3 ..].* = "...".*;
            break :blk &buf;
        });
    }

    pub fn fromAttribute(attribute: *const Attribute) *const Location {
        return @ptrCast(c.mlirLocationFromAttribute(attribute.ptr()).ptr);
    }

    pub fn unknown(ctx: *Context) *const Location {
        return @ptrCast(c.mlirLocationUnknownGet(ctx.ptr()).ptr);
    }
};

pub const Type = opaque {
    const M = StdMethods(Type, c.MlirType);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    pub fn parse(ctx: *Context, str: []const u8) Error!*const Type {
        return @ptrCast(c.mlirTypeParseGet(ctx.ptr(), stringRef(str)).ptr orelse return Error.InvalidMlir);
    }
};

pub const IndexType = opaque {
    const M = StdMethods(IndexType, c.MlirType);

    pub const isAFn = c.mlirTypeIsAIndex;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);

    pub fn get(ctx: *Context) *const IndexType {
        return @ptrCast(c.mlirIndexTypeGet(ctx.ptr()).ptr);
    }
};

pub const Signedness = enum {
    signless,
    signed,
    unsigned,
};

pub const IntegerType = opaque {
    const M = StdMethods(IntegerType, c.MlirType);

    pub const isAFn = c.mlirTypeIsAInteger;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);

    fn get(ctx: *Context, bitwidth: usize, sign: Signedness) *const IntegerType {
        return @ptrCast(switch (sign) {
            .signless => c.mlirIntegerTypeGet(ctx.ptr(), @intCast(bitwidth)).ptr,
            .signed => c.mlirIntegerTypeSignedGet(ctx.ptr(), @intCast(bitwidth)).ptr,
            .unsigned => c.mlirIntegerTypeUnsignedGet(ctx.ptr(), @intCast(bitwidth)).ptr,
        });
    }
};

pub fn integerType(ctx: *Context, bitwidth: usize, sign: Signedness) *const Type {
    return @ptrCast(IntegerType.get(ctx, bitwidth, sign));
}

pub const FloatTypes = enum {
    f4e2m1fn,
    f6e2m3fn,
    f6e3m2fn,
    f8e3m4,
    f8e4m3,
    f8e4m3b11fnuz,
    f8e4m3fn,
    f8e4m3fnuz,
    f8e5m2,
    f8e5m2fnuz,
    f8e8m0fnu,
    bf16,
    f16,
    f32,
    f64,
    // f80,
    // f128,
    tf32,
};

pub fn FloatType(comptime ft: FloatTypes) type {
    const Config = switch (ft) {
        .f4e2m1fn => .{ c.mlirTypeIsAFloat4E2M1FN, c.mlirFloat4E2M1FNTypeGet },
        .f6e2m3fn => .{ c.mlirTypeIsAFloat6E2M3FN, c.mlirFloat6E2M3FNTypeGet },
        .f6e3m2fn => .{ c.mlirTypeIsAFloat6E3M2FN, c.mlirFloat6E3M2FNTypeGet },
        .f8e3m4 => .{ c.mlirTypeIsAFloat8E3M4, c.mlirFloat8E3M4TypeGet },
        .f8e4m3 => .{ c.mlirTypeIsAFloat8E4M3, c.mlirFloat8E4M3TypeGet },
        .f8e4m3b11fnuz => .{ c.mlirTypeIsAFloat8E4M3B11FNUZ, c.mlirFloat8E4M3B11FNUZTypeGet },
        .f8e4m3fn => .{ c.mlirTypeIsAFloat8E4M3FN, c.mlirFloat8E4M3FNTypeGet },
        .f8e4m3fnuz => .{ c.mlirTypeIsAFloat8E4M3FNUZ, c.mlirFloat8E4M3FNUZTypeGet },
        .f8e5m2 => .{ c.mlirTypeIsAFloat8E5M2, c.mlirFloat8E5M2TypeGet },
        .f8e5m2fnuz => .{ c.mlirTypeIsAFloat8E5M2FNUZ, c.mlirFloat8E5M2FNUZTypeGet },
        .f8e8m0fnu => .{ c.mlirTypeIsAFloat8E8M0FNU, c.mlirFloat8E8M0FNUTypeGet },
        .bf16 => .{ c.mlirTypeIsABF16, c.mlirBF16TypeGet },
        .f16 => .{ c.mlirTypeIsAF16, c.mlirF16TypeGet },
        .f32 => .{ c.mlirTypeIsAF32, c.mlirF32TypeGet },
        .f64 => .{ c.mlirTypeIsAF64, c.mlirF64TypeGet },
        .tf32 => .{ c.mlirTypeIsATF32, c.mlirTF32TypeGet },
    };

    return opaque {
        const Self = @This();
        const M = StdMethods(Self, c.MlirType);

        pub const isAFn = Config[0];
        pub const ptr = M.ptr;
        pub const eql = M.eql(c.mlirTypeEqual);
        pub const format = M.format(c.mlirTypePrint);

        pub fn get(ctx: *Context) *const Self {
            const getter = Config[1];
            return @ptrCast(getter(ctx.ptr()).ptr);
        }
    };
}

pub fn floatType(comptime ft: FloatTypes, ctx: *Context) *const Type {
    return @ptrCast(FloatType(ft).get(ctx));
}

pub const Attribute = opaque {
    const M = StdMethods(Attribute, c.MlirAttribute);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub fn parse(ctx: *Context, attr: []const u8) Error!*const Attribute {
        return @ptrCast(c.mlirAttributeParseGet(ctx.ptr(), stringRef(attr)).ptr orelse return Error.InvalidMlir);
    }
};

pub const StringAttribute = opaque {
    const M = StdMethods(StringAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsAString;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub fn init(ctx: *Context, str: []const u8) *const StringAttribute {
        return @ptrCast(c.mlirStringAttrGet(ctx.ptr(), stringRef(str)).ptr);
    }

    pub fn value(self: *const StringAttribute) []const u8 {
        return string(c.mlirStringAttrGetValue(self.ptr()));
    }
};

pub fn stringAttribute(ctx: *Context, str: []const u8) *const Attribute {
    return @ptrCast(StringAttribute.init(ctx, str));
}

pub const IntegerAttribute = opaque {
    const M = StdMethods(IntegerAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsAInteger;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub fn init(ctx: *Context, value_: anytype) *const IntegerAttribute {
        const ti = @typeInfo(@TypeOf(value_)).int;
        const sign: Signedness = switch (ti.signedness) {
            .signed => .signless,
            .unsigned => .unsigned,
        };
        return @ptrCast(c.mlirIntegerAttrGet(
            IntegerType.get(ctx, @intCast(ti.bits), sign).ptr(),
            @intCast(value_),
        ).ptr);
    }

    pub fn value(self: *const IntegerAttribute, comptime T: type) T {
        const getValue = switch (@typeInfo(T).int.signedness) {
            .signed => c.mlirIntegerAttrGetValueInt,
            .unsigned => c.mlirIntegerAttrGetValueUInt,
        };
        return @intCast(getValue(self.ptr()));
    }
};

pub fn integerAttribute(ctx: *Context, value_: anytype) *const Attribute {
    return @ptrCast(IntegerAttribute.init(ctx, value_));
}

pub const BoolAttribute = opaque {
    const M = StdMethods(BoolAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsABool;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub fn init(ctx: *Context, value_: bool) *const BoolAttribute {
        return @ptrCast(c.mlirBoolAttrGet(ctx.ptr(), @intFromBool(value_)).ptr);
    }

    pub fn value(self: *const BoolAttribute) bool {
        return c.mlirBoolAttrGetValue(self.ptr());
    }
};

pub fn boolAttribute(ctx: *Context, value: bool) *const Attribute {
    return @ptrCast(BoolAttribute.init(ctx, value));
}

pub const TypeAttribute = opaque {
    const M = StdMethods(TypeAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsAType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub fn init(type_: *const Type) *const TypeAttribute {
        return @ptrCast(c.mlirTypeAttrGet(type_.ptr()).ptr);
    }

    pub fn value(self: *const TypeAttribute) *const Type {
        return @ptrCast(c.mlirTypeAttrGetValue(self.ptr()).ptr);
    }
};

pub fn typeAttribute(type_: *const Type) *const TypeAttribute {
    return @ptrCast(TypeAttribute.init(type_));
}

pub const Dialect = struct {
    const M = StdMethods(Dialect, c.MlirDialect);

    pub const ptr = M.ptr;

    pub fn context(self: *const Dialect) *Context {
        return @ptrCast(c.mlirDialectGetContext(self.ptr()).ptr);
    }

    pub fn namespace(self: *const Dialect) []const u8 {
        return string(c.mlirDialectGetNamespace(self.ptr()));
    }

    pub fn format(self: *const Dialect, writer: *std.Io.Writer) !void {
        try writer.print("{s}{{ .namespace = \"{s}\" }}", .{ @typeName(Dialect), self.namespace() });
    }
};

pub const DialectHandle = opaque {
    const M = StdMethods(DialectHandle, c.MlirDialectHandle);

    pub const ptr = M.ptr;

    pub fn format(self: *const DialectHandle, writer: *std.Io.Writer) !void {
        try writer.print("{s}{{ .namespace = \"{s}\" }}", .{ @typeName(DialectHandle), self.namespace() });
    }

    pub fn namespace(self: *const DialectHandle) []const u8 {
        return string(c.mlirDialectHandleGetNamespace(self.ptr()));
    }

    pub fn insertDialect(self: *const DialectHandle, registry: *DialectRegistry) void {
        c.mlirDialectHandleInsertDialect(self.ptr(), registry.ptr());
    }

    pub fn registerDialect(self: *const DialectHandle, ctx: *Context) void {
        c.mlirDialectHandleRegisterDialect(self.ptr(), ctx.ptr());
    }

    pub fn loadDialect(self: *const DialectHandle, ctx: *Context) *Dialect {
        return @ptrCast(c.mlirDialectHandleLoadDialect(self.ptr(), ctx.ptr()).ptr);
    }

    pub fn fromString(comptime namespace_: []const u8) *const DialectHandle {
        return @ptrCast(@field(c, "mlirGetDialectHandle__" ++ namespace_ ++ "__")().ptr);
    }
};

pub const Identifier = opaque {
    const M = StdMethods(Identifier, c.MlirIdentifier);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirIdentifierEqual);

    pub fn context(self: *const Identifier) *Context {
        return @ptrCast(c.mlirIdentifierGetContext(self.ptr()).ptr);
    }

    pub fn get(ctx: *Context, str_: []const u8) *const Identifier {
        return @ptrCast(c.mlirIdentifierGet(ctx.ptr(), stringRef(str_)).ptr);
    }

    pub fn str(self: *const Identifier) []const u8 {
        return string(c.mlirIdentifierStr(self.ptr()));
    }

    pub fn format(self: *const Identifier, writer: *std.Io.Writer) !void {
        try writer.print("{s}{{ .str = \"{s}\" }}", .{ @typeName(Identifier), self.str() });
    }
};

pub const Module = opaque {
    const M = StdMethods(Module, c.MlirModule);

    const ptr = M.ptr;

    pub fn init(loc: *const Location) *const Module {
        return @ptrCast(c.mlirModuleCreateEmpty(loc.ptr()).ptr);
    }

    pub fn deinit(self: *Module) void {
        c.mlirModuleDestroy(self.ptr());
    }

    pub fn fromOperation(op: *Operation) *const Module {
        return @ptrCast(c.mlirModuleFromOperation(op.ptr()).ptr);
    }

    pub fn operation(self: *const Module) *Operation {
        return @ptrCast(c.mlirModuleGetOperation(self.ptr()).ptr);
    }

    pub fn parse(ctx: *Context, source: []const u8) Error!*const Module {
        return @ptrCast(c.mlirModuleCreateParse(ctx.ptr(), stringRef(source)).ptr orelse return Error.InvalidMlir);
    }

    pub fn context(self: *const Module) *Context {
        return @ptrCast(c.mlirModuleGetContext(self.ptr()).ptr);
    }

    pub fn body(self: *const Module) *Block {
        return @ptrCast(c.mlirModuleGetBody(self.ptr()).ptr);
    }
};

pub const Block = opaque {
    const M = StdMethods(Block, c.MlirBlock);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirBlockEqual);
    pub const format = M.format(c.mlirBlockPrint);

    pub fn init(args: []const *const Type, locs: []const *const Location) *Block {
        return @ptrCast(c.mlirBlockCreate(@intCast(args.len), @ptrCast(args), @ptrCast(locs)).ptr);
    }

    pub fn deinit(self: *Block) void {
        std.debug.assert(self.parentOperation() == null);
        c.mlirBlockDestroy(self.ptr());
    }

    pub fn argument(self: *const Block, index: usize) *const Value {
        return @ptrCast(c.mlirBlockGetArgument(self.ptr(), @intCast(index)).ptr);
    }

    pub fn numArguments(self: *const Block) usize {
        return @intCast(c.mlirBlockGetNumArguments(self.ptr()));
    }

    pub fn addArgument(self: *Block, type_: *const Type, loc: *const Location) *const Value {
        return @ptrCast(c.mlirBlockAddArgument(self.ptr(), type_.ptr(), loc.ptr()).ptr);
    }

    pub fn insertArgument(self: *Block, index: usize, type_: *const Type, loc: *const Location) *const Value {
        return @ptrCast(c.mlirBlockInsertArgument(self.ptr(), @intCast(index), type_.ptr(), loc.ptr()).ptr);
    }

    pub fn appendOwnedOperation(self: *Block, op: *Operation) void {
        c.mlirBlockAppendOwnedOperation(self.ptr(), op.ptr());
    }

    pub fn appendOwnedOperations(self: *Block, ops: []const *Operation) void {
        for (ops) |op| {
            self.appendOwnedOperation(op);
        }
    }

    pub fn parentOperation(self: *const Block) ?*Operation {
        return @ptrCast(c.mlirBlockGetParentOperation(self.ptr()).ptr);
    }

    pub fn detach(self: *Block) void {
        c.mlirBlockDetach(self.ptr());
    }
};

pub const NamedAttribute = struct {
    inner: c.MlirNamedAttribute,

    pub fn ptr(self: NamedAttribute) c.MlirNamedAttribute {
        return self.inner;
    }

    pub fn name(self: NamedAttribute) *const Identifier {
        return @ptrCast(self.inner.name.ptr);
    }

    pub fn attribute(self: NamedAttribute) *const Attribute {
        return @ptrCast(self.inner.attribute.ptr);
    }

    pub fn named(ctx: *Context, name_: []const u8, attr: *const Attribute) NamedAttribute {
        return .{
            .inner = .{
                .name = Identifier.get(ctx, name_).ptr(),
                .attribute = attr.ptr(),
            },
        };
    }

    pub fn init(name_: *const Identifier, attr: *const Attribute) NamedAttribute {
        return .{
            .inner = .{
                .name = name_.ptr(),
                .attribute = attr.ptr(),
            },
        };
    }

    pub fn format(self: NamedAttribute, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("{s}{{ .{s} = \"{f}\" }}", .{
            @typeName(NamedAttribute),
            self.name().str(),
            self.attribute(),
        });
    }
};

pub const DictionaryAttribute = opaque {
    const M = StdMethods(DictionaryAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsADictionary;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub fn init(ctx: *Context, attributes: []const NamedAttribute) *const DictionaryAttribute {
        return @ptrCast(c.mlirDictionaryAttrGet(
            ctx.ptr(),
            @intCast(attributes.len),
            @ptrCast(attributes.ptr),
        ).ptr);
    }

    pub fn size(self: *const DictionaryAttribute) usize {
        return @intCast(c.mlirDictionaryAttrGetNumElements(self.ptr()));
    }

    pub fn get(self: *const DictionaryAttribute, pos: usize) NamedAttribute {
        return .{ .inner = c.mlirDictionaryAttrGetElement(self.ptr(), @bitCast(pos)) };
    }

    pub fn getByName(self: *const DictionaryAttribute, name: []const u8) ?*const Attribute {
        return @ptrCast(c.mlirDictionaryAttrGetElementByName(self.ptr(), stringRef(name)).ptr);
    }
};

pub const Value = opaque {
    const M = StdMethods(Value, c.MlirValue);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirValueEqual);
    pub const format = M.format(c.mlirValuePrint);
    pub const isA = M.isA;

    pub fn type_(val: *const Value) *const Type {
        return @ptrCast(c.mlirValueGetType(val.ptr()).ptr);
    }

    pub fn setType(self: *Value, type__: *const Type) void {
        c.mlirValueSetType(self.ptr(), type__.ptr());
    }

    pub fn firstUse(self: *const Value) *const OpOperand {
        return @ptrCast(c.mlirValueGetFirstUse(self.ptr()).ptr);
    }

    pub fn replaceAllUsesWith(self: *const Value, with: *const Value) void {
        c.mlirValueReplaceAllUsesOfWith(self.ptr(), with.ptr());
    }
};

pub const OpOperand = opaque {
    const M = StdMethods(OpOperand, c.MlirOpOperand);

    pub const ptr = M.ptr;

    pub fn owner(self: *const OpOperand) *Operation {
        return @ptrCast(c.mlirOpOperandGetOwner(self.ptr()).ptr);
    }

    pub fn number(self: *const OpOperand) usize {
        return @intCast(c.mlirOpOperandGetOperandNumber(self.ptr()));
    }

    pub fn nextUse(self: *const OpOperand) ?*const OpOperand {
        return @ptrCast(c.mlirOpOperandGetNextUse(self.ptr()).ptr);
    }
};

pub const OpResult = opaque {
    const M = StdMethods(OpResult, c.MlirValue);

    pub const isAFn = c.mlirValueIsAOpResult;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirValueEqual);
    pub const format = M.format(c.mlirValuePrint);

    pub fn owner(self: *const OpResult) *Operation {
        return @ptrCast(c.mlirOpResultGetOwner(self.ptr()).ptr);
    }

    pub fn resultNumber(self: *const OpResult) usize {
        return @intCast(c.mlirOpResultGetResultNumber(self.ptr()));
    }
};

pub const BlockArgument = opaque {
    const M = StdMethods(BlockArgument, c.MlirValue);

    pub const isAFn = c.mlirValueIsABlockArgument;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirValueEqual);
    pub const format = M.format(c.mlirValuePrint);

    pub fn block(arg: *const BlockArgument) *Block {
        return @ptrCast(c.mlirBlockArgumentGetOwner(arg.ptr()).ptr);
    }

    pub fn number(self: *const BlockArgument) usize {
        return @intCast(c.mlirBlockArgumentGetArgNumber(self.ptr()));
    }

    pub fn setType(self: *BlockArgument, type_: *const Type) void {
        c.mlirBlockArgumentSetType(self.ptr(), type_.ptr());
    }
};

pub const OperationState = struct {
    const M = StdMethods(OperationState, c.MlirOperationState);

    name: []const u8,
    location: *const Location,
    results: []const *const Type = &.{},
    operands: []const *const Value = &.{},
    regions: []*const Region = &.{},
    successors: []*const Block = &.{},
    attributes: []NamedAttribute = &.{},
    result_type_inference: bool = false,

    pub const ptr = M.ptr;

    pub fn inner(self: OperationState) c.MlirOperationState {
        return .{
            .name = stringRef(self.name),
            .nResults = @intCast(self.results.len),
            .results = @constCast(@ptrCast(self.results)),
            .nOperands = @intCast(self.operands.len),
            .operands = @constCast(@ptrCast(self.operands)),
            .nRegions = @intCast(self.regions.len),
            .regions = @constCast(@ptrCast(self.regions)),
            .nSuccessors = @intCast(self.successors.len),
            .successors = @constCast(@ptrCast(self.successors)),
            .nAttributes = @intCast(self.attributes.len),
            .attributes = @constCast(@ptrCast(self.attributes)),
            .enableResultTypeInference = self.result_type_inference,
        };
    }
};

pub const Operation = opaque {
    const M = StdMethods(Operation, c.MlirOperation);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirOperationEqual);
    pub const format = M.format(c.mlirOperationPrint);

    pub fn init(state: *OperationState) !*Operation {
        var os = state.inner();
        return @ptrCast(c.mlirOperationCreate(&os).ptr orelse return Error.InvalidMlir);
    }

    pub fn deinit(self: *Operation) void {
        std.debug.assert(self.block() == null);
        c.mlirOperationDestroy(self.ptr());
    }

    pub fn parse(ctx: *Context, source: []const u8, name: []const u8) Error!*Operation {
        return @ptrCast(c.mlirOperationCreateParse(ctx.ptr(), stringRef(source), stringRef(name)).ptr orelse return Error.InvalidMlir);
    }

    pub fn clone(self: *const Operation) *Operation {
        return @ptrCast(c.mlirOperationClone(self.ptr()).ptr);
    }

    pub fn block(self: *const Operation) ?*Block {
        return @ptrCast(c.mlirOperationGetBlock(self.ptr()).ptr);
    }

    pub fn parent(self: *const Operation) ?*Operation {
        return @ptrCast(c.mlirOperationGetParentOperation(self.ptr()).ptr);
    }

    pub const PrintFlags = struct {
        elide_large_elements_attrs: ?usize = null,
        debug_info: bool = false,
        debug_info_pretty_form: bool = true,
        print_generic_op_form: bool = false,
        use_local_scope: bool = false,
        assume_verified: bool = false,

        const FmtCtx = struct {
            self: *const Operation,
            flags: PrintFlags,
        };
    };

    fn formatWithFlags(fctx: PrintFlags.FmtCtx, writer: *std.io.Writer) std.Io.Writer.Error!void {
        const flags = c.mlirOpPrintingFlagsCreate();
        defer c.mlirOpPrintingFlagsDestroy(flags);

        if (fctx.flags.elide_large_elements_attrs) |v| {
            c.mlirOpPrintingFlagsElideLargeElementsAttrs(flags, @intCast(v));
        }
        c.mlirOpPrintingFlagsEnableDebugInfo(
            flags,
            fctx.flags.debug_info,
            fctx.flags.debug_info_pretty_form,
        );
        if (fctx.flags.print_generic_op_form) {
            c.mlirOpPrintingFlagsPrintGenericOpForm(flags);
        }
        if (fctx.flags.use_local_scope) {
            c.mlirOpPrintingFlagsUseLocalScope(flags);
        }
        if (fctx.flags.assume_verified) {
            c.mlirOpPrintingFlagsAssumeVerified(flags);
        }
        var sctx: StringCallbackCtx = .{ .writer = writer };
        c.mlirOperationPrintWithFlags(fctx.self.ptr(), flags, stringCallback, &sctx);
        if (sctx.err) |err| {
            return err;
        }
    }

    pub fn fmt(self: *const Operation, flags: PrintFlags) std.fmt.Alt(PrintFlags.FmtCtx, formatWithFlags) {
        return .{ .data = .{
            .self = self,
            .flags = flags,
        } };
    }

    pub const MakeArgs = struct {
        pub const AttrTuple = struct { []const u8, *const Attribute };

        operands: []const *const Value,
        operands_variadic: ?[]const usize = null,
        results: []const *const Type,
        results_variadic: ?[]const usize = null,
        result_type_inference: bool = false,
        n_regions: usize = 0,
        attributes: ?[]const AttrTuple = null,
        blocks: ?[]const *const Block = null,
        verify: bool = true,
        location: *const Location,
    };

    pub fn try_make(ctx: *Context, op_name: []const u8, args: MakeArgs) !*Operation {
        _ = ctx; // autofix
        var state: OperationState = .{
            .name = op_name,
            .location = args.location,
            .result_type_inference = args.result_type_inference,
        };
        // std.debug.assert(!(args.operands != null and args.variadic_operands != null));
        state.addOperands(operands);

        // if (args.operands) |operands| {
        //     state.addOperands(operands);
        //     // } else if (args.variadic_operands) |operands_segments| {
        //     //     const MAX_SEGMENTS = 32;
        //     //     var segments: std.BoundedArray(i32, MAX_SEGMENTS) = .{};

        //     //     for (operands_segments) |operands| {
        //     //         state.addOperands(operands);
        //     //         segments.appendAssumeCapacity(@intCast(operands.len));
        //     //     }
        //     //     state.addAttribute(ctx, "operandSegmentSizes", .denseElements(ctx, &.{@intCast(segments.len)}, .i32, segments.constSlice()));
        //     // } else if (args.tt_variadic_operands) |operands_segments| {
        //     //     // stablehlo and triton seems to disagree on the expected type of operandSegmentSizes, let's fix that.
        //     //     const MAX_SEGMENTS = 32;
        //     //     var segments: std.BoundedArray(i32, MAX_SEGMENTS) = .{};

        //     //     for (operands_segments) |operands| {
        //     //         state.addOperands(operands);
        //     //         segments.appendAssumeCapacity(@intCast(operands.len));
        //     //     }
        //     //     state.addAttribute(ctx, "operandSegmentSizes", .dense(ctx, .i32, segments.constSlice()));
        // }
        // std.debug.assert(!(args.results != null and args.variadic_results != null));
        // if (args.results) |results| {
        //     state.addResults(results);
        //     // } else if (args.variadic_results) |result_segments| {
        //     //     for (result_segments) |results| {
        //     //         state.addResults(results);
        //     //     }
        // }
        // for (0..args.n_regions) |_| {
        //     var region_ = Region.init() catch {
        //         @panic("Failed to create MLIR region");
        //     };
        //     state.addRegion(&region_);
        // }
        // if (args.attributes) |attrs| {
        //     for (attrs) |attr| {
        //         state.addAttributeRaw(
        //             Identifier.get(ctx, attr[0]),
        //             attr[1],
        //         );
        //     }
        // }
        // if (args.blocks) |blocks_| {
        //     for (blocks_) |block_| {
        //         var region_ = Region.init() catch {
        //             @panic("Failed to create MLIR region");
        //         };
        //         region_.appendBlock(block_);
        //         state.addRegion(&region_);
        //     }
        // }

        const new_op = try Operation.init(&state);
        if (args.verify and new_op.verify() == false) {
            log.err("Failed to verify MLIR operation:\n{f}", .{
                new_op.fmt(.{
                    .print_generic_op_form = true,
                    .debug_info = true,
                }),
            });
            return Error.InvalidMlir;
        }

        return new_op;
    }

    pub fn make(ctx: *Context, op_name: []const u8, args: MakeArgs) *Operation {
        return try_make(ctx, op_name, args) catch {
            @panic("Failed to create MLIR operation");
        };
    }

    pub fn verify(self: *const Operation) bool {
        return c.mlirOperationVerify(self.ptr());
    }

    pub fn location(self: *const Operation) *const Location {
        return @ptrCast(c.mlirOperationGetLocation(self.ptr()).ptr);
    }

    pub const WalkOrder = enum(c.MlirWalkOrder) {
        pre_order = c.MlirWalkPreOrder,
        post_order = c.MlirWalkPostOrder,
    };

    pub const WalkResult = enum(c.MlirWalkResult) {
        advance = c.MlirWalkResultAdvance,
        interrupt = c.MlirWalkResultInterrupt,
        skip = c.MlirWalkResultSkip,
    };

    pub fn walk(self: *Operation, order: WalkOrder, ctx: anytype, walkfn: fn (ctx_: @TypeOf(ctx), op: *Operation) WalkResult) void {
        var inner_ctx = .{ .ctx = ctx };
        const ContextType = @TypeOf(inner_ctx);

        c.mlirOperationWalk(
            self._inner,
            (struct {
                pub fn callback(op: c.MlirOperation, ctx_: ?*anyopaque) callconv(.c) c.MlirWalkResult {
                    const inner_ctx_: *ContextType = @ptrCast(@alignCast(ctx_));
                    return @intFromEnum(walkfn(inner_ctx_.ctx, @ptrCast(op.ptr)));
                }
            }).callback,
            &inner_ctx,
            @intFromEnum(order),
        );
    }
};

pub const Region = opaque {
    const M = StdMethods(Region, c.MlirRegion);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirRegionEqual);

    pub fn init() Error!*Region {
        return @ptrCast(c.mlirRegionCreate().ptr orelse return Error.InvalidMlir);
    }

    pub fn deinit(self: *Region) void {
        c.mlirRegionDestroy(self.ptr());
    }

    pub fn appendOwnedBlock(self: *Region, block: *Block) void {
        c.mlirRegionAppendOwnedBlock(self.ptr(), block.ptr());
    }

    pub fn insertOwnedBlock(self: *Region, index: usize, block: *Block) void {
        c.mlirRegionInsertOwnedBlock(self.ptr(), @intCast(index), block.ptr());
    }

    pub fn insertOwnedBlockBefore(self: *Region, reference: *const Block, block: *Block) void {
        c.mlirRegionInsertOwnedBlockBefore(self.ptr(), reference.ptr(), block.ptr());
    }

    pub fn insertOwnedBlockAfter(self: *Region, reference: *const Block, block: *Block) void {
        c.mlirRegionInsertOwnedBlockAfter(self.ptr(), reference.ptr(), block.ptr());
    }

    pub fn firstBlock(self: *const Region) ?*Block {
        return @ptrCast(c.mlirRegionGetFirstBlock(self.ptr()).ptr);
    }

    pub fn takeBody(self: *Region, from: *Region) void {
        c.mlirRegionTakeBody(self.ptr(), from.ptr());
    }

    pub fn nextInOperation(self: *const Region) *Region {
        return @ptrCast(c.mlirRegionGetNextInOperation(self.ptr()).ptr);
    }
};

// pub const Block = opaque {
//     const M = StdMethods(Block, c.MlirBlock);
//     // pub const wrapOr = helpers.wrapOr(Block, c.mlirBlockIsNull);
//     // pub const deinit = helpers.deinit(Block, c.mlirBlockDestroy);

//     // pub const eql = helpers.eql(Block, c.mlirBlockEqual);

//     pub fn init(args: []*const Type, locs: []const *const Location) !Block {
//         return @ptrCast(c.mlirBlockCreate(@intCast(args.len), @ptrCast(args.ptr), @ptrCast(locs.ptr)) orelse return Error.InvalidMlir);
//     }

//     pub fn argument(self: Block, index: usize) Value {
//         return .{ ._inner = c.mlirBlockGetArgument(self._inner, @intCast(index)) };
//     }

//     pub fn numArguments(self: Block) usize {
//         return @intCast(c.mlirBlockGetNumArguments(self._inner));
//     }

//     pub fn addArgument(self: *Block, typ: Type, loc: Location) Value {
//         return .{ ._inner = c.mlirBlockAddArgument(self._inner, typ._inner, loc._inner) };
//     }

//     pub fn insertArgument(self: *Block, index: usize, typ: Type, loc: Location) Value {
//         return .{ ._inner = c.mlirBlockInsertArgument(self._inner, @intCast(index), typ._inner, loc._inner) };
//     }

//     pub fn equals(self: Block, other: Block) bool {
//         return c.mlirBlockEqual(self._inner, other._inner);
//     }

//     pub fn appendOperation(self: Block, op: Operation) void {
//         c.mlirBlockAppendOwnedOperation(self._inner, op._inner);
//     }

//     pub fn appendOperations(self: *Block, ops: []const Operation) void {
//         for (ops) |op| {
//             c.mlirBlockAppendOwnedOperation(self._inner, op._inner);
//         }
//     }

//     pub const RecursiveOpts = enum { open, hermetic };

//     pub fn appendValueRecursive(self: Block, value: Value, opt: RecursiveOpts) void {
//         switch (value.kind()) {
//             .op_result => |parent_op| self.appendOperationRecursive(parent_op, opt),
//             .block_argument => |arg| {
//                 // Hermetic blocks are not allowed to use arguments from other blocks.
//                 stdx.debug.assert(opt == .open or self.eql(arg.block()), "Can't add {} from {?x} block to {?x} block", .{ arg, arg.block()._inner.ptr, self._inner.ptr });
//             },
//             .null => @panic("InvalidMlir"),
//         }
//     }

//     pub fn appendOperationRecursive(self: Block, op: Operation, opt: RecursiveOpts) void {
//         if (op.block()) |prev_block| {
//             // Hermetic blocks are not allowed to reference values from other blocks.
//             stdx.debug.assert(opt == .open or self.equals(prev_block), "Can't add {} from {?x} block to {?x} block", .{ op, prev_block._inner.ptr, self._inner.ptr });
//             return;
//         }
//         for (0..op.numOperands()) |i| {
//             self.appendValueRecursive(op.operand(i), opt);
//         }
//         self.appendOperation(op);
//     }
// };
