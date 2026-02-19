const std = @import("std");

const c = @import("c");
const stdx = @import("stdx");

const log = std.log.scoped(.mlir);

test {
    std.testing.refAllDecls(@This());
}

pub const Error = error{
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

fn stringFromStream(buf: []u8, streamFn: anytype, args: anytype) []const u8 {
    var writer = std.Io.Writer.fixed(buf);
    var sctx: StringCallbackCtx = .{ .writer = &writer };
    _ = @call(.auto, streamFn, args ++ .{ stringCallback, &sctx });
    return writer.buffered();
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

pub fn Methods(comptime T: type, comptime NativeT: type) type {
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
    const M = Methods(DialectRegistry, c.MlirDialectRegistry);

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

pub const PassManager = opaque {
    const M = Methods(PassManager, c.MlirPassManager);

    pub const ptr = M.ptr;

    pub fn init(ctx: *Context) *PassManager {
        return @ptrCast(c.mlirPassManagerCreate(ctx.ptr()).ptr);
    }

    pub fn deinit(self: *const PassManager) void {
        c.mlirPassManagerDestroy(self.ptr());
    }

    pub fn asOpPassManager(self: *PassManager) *OpPassManager {
        return @ptrCast(c.mlirPassManagerGetAsOpPassManager(self.ptr()).ptr);
    }

    pub fn runOnOp(self: *const PassManager, op: *Operation) !void {
        const result = c.mlirPassManagerRunOnOp(self.ptr(), op.ptr());
        if (result.value == 0) {
            return error.MlirUnexpected;
        }
    }

    pub const EnableIRPrintingOpts = struct {
        printBeforeAll: bool = false,
        printAfterAll: bool = false,
        printModuleScope: bool = false,
        printAfterOnlyOnChange: bool = false,
        printAfterOnlyOnFailure: bool = false,
        flags: Operation.PrintFlags = .{},
        treePrintingPath: []const u8 = "",
    };

    pub fn enableIRPrinting(self: *PassManager, opts: EnableIRPrintingOpts) void {
        const flags = opts.flags.createNative();
        defer c.mlirOpPrintingFlagsDestroy(flags);

        c.mlirPassManagerEnableIRPrinting(
            self.ptr(),
            opts.printBeforeAll,
            opts.printAfterAll,
            opts.printModuleScope,
            opts.printAfterOnlyOnChange,
            opts.printAfterOnlyOnFailure,
            flags,
            stringRef(opts.treePrintingPath),
        );
    }
};

pub const OpPassManager = opaque {
    const M = Methods(OpPassManager, c.MlirOpPassManager);

    pub const ptr = M.ptr;
    pub const format = M.format(c.mlirPrintPassPipeline);

    pub fn addPass(self: *OpPassManager, pass: *Pass) void {
        c.mlirOpPassManagerAddOwnedPass(self.ptr(), pass.ptr());
    }

    pub fn addPasses(self: *OpPassManager, passes: []const *Pass) void {
        for (passes) |pass| {
            self.addPass(pass);
        }
    }

    pub fn addPipeline(self: *OpPassManager, pipeline_name: []const u8) !void {
        var buffer: [1024]u8 = undefined;
        const result = stringFromStream(&buffer, c.mlirOpPassManagerAddPipeline, .{ self.ptr(), stringRef(pipeline_name) });
        if (result.len != 0) {
            log.err("mlir error: {s}", .{result});
            return error.MlirUnexpected;
        }
    }
};

pub const Pass = opaque {
    const M = Methods(Pass, c.MlirPass);

    pub const ptr = M.ptr;

    pub fn create(comptime name: @EnumLiteral()) *Pass {
        return @ptrCast(@field(c, "mlirCreate" ++ @tagName(name))().ptr);
    }
};

pub const Context = opaque {
    const M = Methods(Context, c.MlirContext);

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

    pub fn allowUnregisteredDialects(self: *Context, allow: bool) void {
        c.mlirContextSetAllowUnregisteredDialects(self.ptr(), allow);
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
    const M = Methods(Location, c.MlirLocation);

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
    const M = Methods(Type, c.MlirType);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    pub fn parse(ctx: *Context, str: []const u8) Error!*const Type {
        return @ptrCast(c.mlirTypeParseGet(ctx.ptr(), stringRef(str)).ptr orelse return Error.InvalidMlir);
    }
};

pub const IndexType = opaque {
    const M = Methods(IndexType, c.MlirType);

    pub const isAFn = c.mlirTypeIsAIndex;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);

    pub fn get(ctx: *Context) *const IndexType {
        return @ptrCast(c.mlirIndexTypeGet(ctx.ptr()).ptr);
    }
};

pub fn indexType(ctx: *Context) *const Type {
    return @ptrCast(IndexType.get(ctx));
}

pub const IntegerTypes = enum {
    i1,
    i2,
    i4,
    i8,
    i16,
    i32,
    i64,
    i128,
    u1,
    u2,
    u4,
    u8,
    u16,
    u32,
    u64,
    u128,
    si1,
    si4,
    si8,
    si16,
    si32,
    si64,
    si128,

    pub fn bitwidth(self: IntegerTypes) usize {
        return switch (self) {
            .i1, .u1, .si1 => 1,
            .i2, .u2 => 2,
            .i4, .u4, .si4 => 4,
            .i8, .u8, .si8 => 8,
            .i16, .u16, .si16 => 16,
            .i32, .u32, .si32 => 32,
            .i64, .u64, .si64 => 64,
            .i128, .u128, .si128 => 128,
        };
    }

    pub fn signedness(self: IntegerTypes) IntegerType.Signedness {
        return switch (self) {
            .i1, .i2, .i4, .i8, .i16, .i32, .i64, .i128 => .signless,
            .si1, .si4, .si8, .si16, .si32, .si64, .si128 => .signed,
            .u1, .u2, .u4, .u8, .u16, .u32, .u64, .u128 => .unsigned,
        };
    }
};

pub const IntegerType = opaque {
    pub const Signedness = enum {
        signless,
        signed,
        unsigned,
    };

    const M = Methods(IntegerType, c.MlirType);

    pub const isAFn = c.mlirTypeIsAInteger;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);

    fn get(ctx: *Context, it: IntegerTypes) *const IntegerType {
        return exact(ctx, it.bitwidth(), it.signedness());
    }

    fn exact(ctx: *Context, bitwidth: usize, sign: Signedness) *const IntegerType {
        return @ptrCast(switch (sign) {
            .signless => c.mlirIntegerTypeGet(ctx.ptr(), @intCast(bitwidth)).ptr,
            .signed => c.mlirIntegerTypeSignedGet(ctx.ptr(), @intCast(bitwidth)).ptr,
            .unsigned => c.mlirIntegerTypeUnsignedGet(ctx.ptr(), @intCast(bitwidth)).ptr,
        });
    }
};

pub fn integerType(ctx: *Context, it: IntegerTypes) *const Type {
    return @ptrCast(IntegerType.get(ctx, it));
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
    const _isAFn, const getter = switch (ft) {
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
        const M = Methods(Self, c.MlirType);

        pub const isAFn = _isAFn;
        pub const ptr = M.ptr;
        pub const eql = M.eql(c.mlirTypeEqual);
        pub const format = M.format(c.mlirTypePrint);

        pub fn get(ctx: *Context) *const Self {
            return @ptrCast(getter(ctx.ptr()).ptr);
        }
    };
}

pub fn floatType(ctx: *Context, ft: FloatTypes) *const Type {
    return switch (ft) {
        inline else => |v| @ptrCast(FloatType(v).get(ctx)),
    };
}

pub const ComplexTypes = enum {
    c64,
    c128,
};

pub const ComplexType = opaque {
    const M = Methods(ComplexType, c.MlirType);

    pub const isAFn = c.mlirTypeIsAComplex;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);

    pub fn get(ctx: *Context, complex_type: ComplexTypes) *const ComplexType {
        return @ptrCast(c.mlirComplexTypeGet(switch (complex_type) {
            .c64 => floatType(ctx, .f32).ptr(),
            .c128 => floatType(ctx, .f64).ptr(),
        }).ptr);
    }
};

pub fn complexType(ctx: *Context, complex_type: ComplexTypes) *const Type {
    return @ptrCast(ComplexType.get(ctx, complex_type));
}

pub const Attribute = opaque {
    const M = Methods(Attribute, c.MlirAttribute);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub fn parse(ctx: *Context, attr: []const u8) Error!*const Attribute {
        return @ptrCast(c.mlirAttributeParseGet(ctx.ptr(), stringRef(attr)).ptr orelse return Error.InvalidMlir);
    }

    pub fn context(self: *const Context) *Context {
        return @ptrCast(c.mlirAttributeGetContext(self.ptr()).ptr);
    }

    pub fn dialect(self: *const Context) *const Dialect {
        return @ptrCast(c.mlirAttributeGetDialect(self.ptr()).ptr);
    }
};

pub const StringAttribute = opaque {
    const M = Methods(StringAttribute, c.MlirAttribute);

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
    const M = Methods(IntegerAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsAInteger;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub fn init(ctx: *Context, it: IntegerTypes, value_: anytype) *const IntegerAttribute {
        return @ptrCast(c.mlirIntegerAttrGet(
            IntegerType.get(ctx, it).ptr(),
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

pub fn integerAttribute(ctx: *Context, it: IntegerTypes, value_: anytype) *const Attribute {
    return @ptrCast(IntegerAttribute.init(ctx, it, value_));
}

pub const FloatAttribute = opaque {
    const M = Methods(FloatAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsAFloat;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub fn init(ctx: *Context, comptime ft: FloatTypes, value_: f64) *const FloatAttribute {
        return @ptrCast(c.mlirFloatAttrDoubleGet(
            ctx.ptr(),
            FloatType(ft).get(ctx).ptr(),
            value_,
        ).ptr);
    }
};

pub fn floatAttribute(ctx: *Context, comptime ft: FloatTypes, value_: anytype) *const Attribute {
    return @ptrCast(FloatAttribute.init(ctx, ft, value_));
}

pub const BoolAttribute = opaque {
    const M = Methods(BoolAttribute, c.MlirAttribute);

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
    const M = Methods(TypeAttribute, c.MlirAttribute);

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

pub fn typeAttribute(type_: *const Type) *const Attribute {
    return @ptrCast(TypeAttribute.init(type_));
}

pub const Dialect = struct {
    const M = Methods(Dialect, c.MlirDialect);

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
    const M = Methods(DialectHandle, c.MlirDialectHandle);

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
    const M = Methods(Identifier, c.MlirIdentifier);

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
    const M = Methods(Module, c.MlirModule);

    pub const ptr = M.ptr;

    pub fn init(loc: *const Location) *Module {
        return @ptrCast(@constCast(c.mlirModuleCreateEmpty(loc.ptr()).ptr));
    }

    pub fn deinit(self: *Module) void {
        c.mlirModuleDestroy(self.ptr());
    }

    pub fn format(self: *const Module, writer: *std.Io.Writer) !void {
        return self.operation().format(writer);
    }

    pub fn fromOperation(op: *Operation) *Module {
        return @ptrCast(@constCast(c.mlirModuleFromOperation(op.ptr()).ptr));
    }

    pub fn operation(self: *const Module) *Operation {
        return @ptrCast(c.mlirModuleGetOperation(self.ptr()).ptr);
    }

    pub fn parse(ctx: *Context, source: []const u8) Error!*Module {
        return @ptrCast(@constCast(c.mlirModuleCreateParse(ctx.ptr(), stringRef(source)).ptr orelse return Error.InvalidMlir));
    }

    pub fn context(self: *const Module) *Context {
        return @ptrCast(c.mlirModuleGetContext(self.ptr()).ptr);
    }

    pub fn body(self: *const Module) *Block {
        return @ptrCast(c.mlirModuleGetBody(self.ptr()).ptr);
    }
};

pub const Block = opaque {
    const M = Methods(Block, c.MlirBlock);

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

    pub fn terminator(self: *const Block) ?*Operation {
        return @ptrCast(c.mlirBlockGetTerminator(self.ptr()).ptr);
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
    const M = Methods(DictionaryAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsADictionary;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub fn init(ctx: *Context, attributes: []const NamedAttribute) *const DictionaryAttribute {
        return @ptrCast(c.mlirDictionaryAttrGet(
            ctx.ptr(),
            @intCast(attributes.len),
            @ptrCast(attributes),
        ).ptr);
    }

    pub fn size(self: *const DictionaryAttribute) usize {
        return @intCast(c.mlirDictionaryAttrGetNumElements(self.ptr()));
    }

    pub fn get(self: *const DictionaryAttribute, pos: usize) NamedAttribute {
        return .{ .inner = c.mlirDictionaryAttrGetElement(self.ptr(), @intCast(pos)) };
    }

    pub fn getByName(self: *const DictionaryAttribute, name: []const u8) ?*const Attribute {
        return @ptrCast(c.mlirDictionaryAttrGetElementByName(self.ptr(), stringRef(name)).ptr);
    }
};

pub fn dictionaryAttribute(ctx: *Context, attributes: []const NamedAttribute) *const Attribute {
    return @ptrCast(DictionaryAttribute.init(ctx, attributes));
}

pub const Value = opaque {
    const M = Methods(Value, c.MlirValue);

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

    pub fn owner(self: *const Value) *Operation {
        return @ptrCast(c.mlirOpResultGetOwner(self.ptr()).ptr);
    }
};

pub const OpOperand = opaque {
    const M = Methods(OpOperand, c.MlirOpOperand);

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
    const M = Methods(OpResult, c.MlirValue);

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
    const M = Methods(BlockArgument, c.MlirValue);

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
    inner: c.MlirOperationState,

    pub fn init(name: []const u8, location: *const Location) OperationState {
        return .{ .inner = c.mlirOperationStateGet(stringRef(name), location.ptr()) };
    }

    pub fn addResults(self: *OperationState, results: []const *const Type) void {
        c.mlirOperationStateAddResults(&self.inner, @intCast(results.len), @ptrCast(results));
    }

    pub fn addOperands(self: *OperationState, operands: []const *const Value) void {
        c.mlirOperationStateAddOperands(&self.inner, @intCast(operands.len), @ptrCast(operands));
    }

    pub fn addOwnedRegions(self: *OperationState, regions: []const *Region) void {
        c.mlirOperationStateAddOwnedRegions(&self.inner, @intCast(regions.len), @ptrCast(regions));
    }

    pub fn addSuccessors(self: *OperationState, successors: []*const Block) void {
        c.mlirOperationStateAddSuccessors(&self.inner, @intCast(successors.len), @ptrCast(successors));
    }

    pub fn addAttributes(self: *OperationState, attributes: []const NamedAttribute) void {
        c.mlirOperationStateAddAttributes(&self.inner, @intCast(attributes.len), @ptrCast(attributes.ptr));
    }

    pub fn enableResultTypeInference(self: *OperationState) void {
        c.mlirOperationStateEnableResultTypeInference(&self.inner);
    }
};

pub const Operation = opaque {
    const M = Methods(Operation, c.MlirOperation);

    const MAX_SEGMENTS = 32;

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirOperationEqual);
    pub const format = M.format(c.mlirOperationPrint);

    pub fn init(state: *OperationState) !*Operation {
        defer state.* = undefined;
        return @ptrCast(c.mlirOperationCreate(&state.inner).ptr orelse return Error.InvalidMlir);
    }

    pub fn deinit(self: *Operation) void {
        std.debug.assert(self.block() == null);
        c.mlirOperationDestroy(self.ptr());
    }

    pub fn parse(ctx: *Context, source: []const u8, name_: []const u8) Error!*Operation {
        return @ptrCast(c.mlirOperationCreateParse(ctx.ptr(), stringRef(source), stringRef(name_)).ptr orelse return Error.InvalidMlir);
    }

    pub fn name(self: *const Operation) []const u8 {
        return string(c.mlirOperationGetName(self.ptr()));
    }

    pub fn clone(self: *const Operation) *Operation {
        return @ptrCast(c.mlirOperationClone(self.ptr()).ptr);
    }

    pub fn block(self: *const Operation) ?*Block {
        return @ptrCast(c.mlirOperationGetBlock(self.ptr()).ptr);
    }

    pub fn appendTo(self: *Operation, block_: *Block) *Operation {
        block_.appendOwnedOperation(self);
        return self;
    }

    pub fn parent(self: *const Operation) ?*Operation {
        return @ptrCast(c.mlirOperationGetParentOperation(self.ptr()).ptr);
    }

    pub fn removeFromParent(self: *Operation) void {
        c.mlirOperationRemoveFromParent(self.ptr());
    }

    pub fn numOperands(self: *const Operation) usize {
        return @intCast(c.mlirOperationGetNumOperands(self.ptr()));
    }

    pub fn operand(self: *const Operation, index: usize) *const Value {
        return @ptrCast(c.mlirOperationGetOperand(self.ptr(), @intCast(index)).ptr);
    }

    pub fn setOperand(self: *Operation, index: usize, value: *const Value) void {
        c.mlirOperationSetOperand(self.ptr(), @intCast(index), value.ptr());
    }

    pub fn numResults(self: *const Operation) usize {
        return @intCast(c.mlirOperationGetNumResults(self.ptr()));
    }

    pub fn result(self: *const Operation, index: usize) *const Value {
        return @ptrCast(c.mlirOperationGetResult(self.ptr(), @intCast(index)).ptr);
    }

    pub fn nextInBlock(self: *const Operation) *const Operation {
        return @ptrCast(c.mlirOperationGetNextInBlock(self.ptr()).ptr);
    }

    pub fn context(self: *const Operation) *Context {
        return @ptrCast(c.mlirOperationGetContext(self.ptr()).ptr);
    }

    pub fn setAttributeByName(self: *Operation, name_: []const u8, attribute: *const Attribute) void {
        c.mlirOperationSetAttributeByName(self.ptr(), stringRef(name_), attribute.ptr());
    }

    pub const PrintFlags = struct {
        elide_large_elements_attrs: ?usize = null,
        debug_info: bool = false,
        debug_info_pretty_form: bool = true,
        print_generic_op_form: bool = false,
        use_local_scope: bool = false,
        assume_verified: bool = false,

        const Ctx = struct {
            self: *const Operation,
            flags: PrintFlags,
        };

        fn createNative(self: PrintFlags) c.MlirOpPrintingFlags {
            const flags = c.mlirOpPrintingFlagsCreate();
            if (self.elide_large_elements_attrs) |v| {
                c.mlirOpPrintingFlagsElideLargeElementsAttrs(flags, @intCast(v));
            }
            c.mlirOpPrintingFlagsEnableDebugInfo(
                flags,
                self.debug_info,
                self.debug_info_pretty_form,
            );
            if (self.print_generic_op_form) {
                c.mlirOpPrintingFlagsPrintGenericOpForm(flags);
            }
            if (self.use_local_scope) {
                c.mlirOpPrintingFlagsUseLocalScope(flags);
            }
            if (self.assume_verified) {
                c.mlirOpPrintingFlagsAssumeVerified(flags);
            }
            return flags;
        }
    };

    fn formatWithFlags(fctx: PrintFlags.Ctx, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        const flags = fctx.flags.createNative();
        defer c.mlirOpPrintingFlagsDestroy(flags);
        var sctx: StringCallbackCtx = .{ .writer = writer };
        c.mlirOperationPrintWithFlags(fctx.self.ptr(), flags, stringCallback, &sctx);
        if (sctx.err) |err| {
            return err;
        }
    }

    pub fn fmt(self: *const Operation, flags: PrintFlags) std.fmt.Alt(PrintFlags.Ctx, formatWithFlags) {
        return .{ .data = .{
            .self = self,
            .flags = flags,
        } };
    }

    pub const MakeArgs = struct {
        pub const AttrTuple = struct { []const u8, *const Attribute };

        pub const Operands = union(enum) {
            flat: []const *const Value,
            variadic: []const []const *const Value,
        };

        pub const ResultTypes = union(enum) {
            flat: []const *const Type,
            variadic: []const []const *const Type,
        };

        operands: ?Operands = null,
        results: ?ResultTypes = null,
        result_type_inference: bool = false,
        attributes: ?[]const NamedAttribute = null,
        blocks: ?[]const *Block = null,
        verify: bool = true,
        location: ?*const Location = null,
    };

    pub fn try_make(ctx: *Context, name_: []const u8, args: MakeArgs) !*Operation {
        var state: OperationState = .init(name_, args.location orelse .unknown(ctx));
        if (args.result_type_inference) {
            state.enableResultTypeInference();
        }
        if (args.operands) |operands| switch (operands) {
            .flat => |v| state.addOperands(v),
            .variadic => |segments| {
                var sizes: stdx.BoundedArray(i32, MAX_SEGMENTS) = .{};
                for (segments) |segment_operands| {
                    state.addOperands(segment_operands);
                    sizes.appendAssumeCapacity(@intCast(segment_operands.len));
                }
                state.addAttributes(&.{
                    .named(ctx, "operandSegmentSizes", denseElementsAttribute(RankedTensorType.get(&.{@intCast(sizes.len)}, integerType(ctx, .i32), null).shaped(), sizes.constSlice())),
                });
            },
        };
        if (args.results) |results| switch (results) {
            .flat => |v| state.addResults(v),
            .variadic => |segments| {
                var sizes: stdx.BoundedArray(i32, MAX_SEGMENTS) = .{};
                for (segments) |segment_results| {
                    state.addResults(segment_results);
                    sizes.appendAssumeCapacity(@intCast(segment_results.len));
                }
                state.addAttributes(&.{
                    .named(ctx, "resultSegmentSizes", denseElementsAttribute(RankedTensorType.get(&.{@intCast(sizes.len)}, integerType(ctx, .i32), null).shaped(), sizes.constSlice())),
                });
            },
        };
        if (args.attributes) |attrs| {
            state.addAttributes(attrs);
        }
        if (args.blocks) |blocks| {
            const region = Region.init();
            for (blocks) |block_| {
                region.appendOwnedBlock(block_);
            }
            state.addOwnedRegions(&.{region});
        }
        const new_op = try Operation.init(&state);
        errdefer new_op.deinit();
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

    pub const BytecodeWriterConfig = struct {
        desired_emit_version: ?usize = null,

        const Ctx = struct {
            self: *const Operation,
            config: ?BytecodeWriterConfig,
        };
    };

    const WriteBytecodeError = error{InvalidMlirBytecodeVersion} || std.Io.Writer.Error;

    pub fn writeBytecode(self: *const Operation, config: ?BytecodeWriterConfig, writer: *std.Io.Writer) WriteBytecodeError!void {
        if (config) |config_| {
            const bc_config = c.mlirBytecodeWriterConfigCreate();
            defer c.mlirBytecodeWriterConfigDestroy(bc_config);
            if (config_.desired_emit_version) |version| {
                c.mlirBytecodeWriterConfigDesiredEmitVersion(bc_config, @intCast(version));
            }
            var sctx: StringCallbackCtx = .{ .writer = writer };
            const logical_result = c.mlirOperationWriteBytecodeWithConfig(self.ptr(), bc_config, stringCallback, &sctx);

            if (logical_result.value == 0) {
                return WriteBytecodeError.InvalidMlirBytecodeVersion;
            }

            if (sctx.err) |err| {
                return err;
            }
            return;
        }

        var sctx: StringCallbackCtx = .{ .writer = writer };
        c.mlirOperationWriteBytecode(self.ptr(), stringCallback, &sctx);
        if (sctx.err) |err| {
            return err;
        }
    }

    fn fmtBytecodeImpl(ctx: BytecodeWriterConfig.Ctx, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try ctx.self.writeBytecode(ctx.config, writer);
    }

    pub fn fmtBytecode(self: *const Operation, config: ?BytecodeWriterConfig) std.fmt.Alt(BytecodeWriterConfig.Ctx, fmtBytecodeImpl) {
        return .{ .data = .{
            .self = self,
            .config = config,
        } };
    }
};

pub const Region = opaque {
    const M = Methods(Region, c.MlirRegion);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirRegionEqual);

    pub fn init() *Region {
        return @ptrCast(c.mlirRegionCreate().ptr);
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

pub const ShapedType = opaque {
    const M = Methods(ShapedType, c.MlirType);

    pub const MAX_RANK = 16;

    pub const isAFn = c.mlirTypeIsAShaped;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    pub fn elementType(self: *const ShapedType) *const Type {
        return @ptrCast(c.mlirShapedTypeGetElementType(self.ptr()).ptr);
    }

    pub fn rank(self: *const ShapedType) usize {
        return @intCast(c.mlirShapedTypeGetRank(self.ptr()));
    }

    pub fn dimension(self: *const ShapedType, dim: usize) i64 {
        return c.mlirShapedTypeGetDimSize(self.ptr(), @intCast(dim));
    }
};

pub const RankedTensorType = opaque {
    const M = Methods(RankedTensorType, c.MlirType);

    pub const isAFn = c.mlirTypeIsARankedTensor;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);

    pub fn get(dimensions: []const i64, elemType: *const Type, encoding: ?*Attribute) *const RankedTensorType {
        return @ptrCast(c.mlirRankedTensorTypeGet(
            @intCast(dimensions.len),
            @ptrCast(dimensions),
            elemType.ptr(),
            if (encoding) |e| e.ptr() else c.mlirAttributeGetNull(),
        ).ptr);
    }

    pub fn fromShaped(other: *const ShapedType) *const RankedTensorType {
        var dims: stdx.BoundedArray(i64, ShapedType.MAX_RANK) = .{};
        for (0..other.rank()) |i| {
            dims.appendAssumeCapacity(other.dimension(i));
        }
        return get(
            dims.constSlice(),
            other.elementType(),
        );
    }

    pub fn elementType(self: *const RankedTensorType) *const Type {
        return @ptrCast(c.mlirShapedTypeGetElementType(self.ptr()).ptr);
    }

    pub fn rank(self: *const RankedTensorType) usize {
        return @intCast(c.mlirShapedTypeGetRank(self.ptr()));
    }

    pub fn dimension(self: *const RankedTensorType, dim: usize) i64 {
        return c.mlirShapedTypeGetDimSize(self.ptr(), @intCast(dim));
    }

    pub fn shaped(self: *const RankedTensorType) *const ShapedType {
        return @ptrCast(self);
    }
};

pub fn rankedTensorType(dimensions: []const i64, elem_type: *const Type) *const Type {
    return @ptrCast(RankedTensorType.get(dimensions, elem_type, null));
}

pub fn rankedTensorTypeWithEncoding(dimensions: []const i64, elem_type: *const Type, encoding: *Attribute) *const Type {
    return @ptrCast(RankedTensorType.get(dimensions, elem_type, encoding));
}

pub const DenseElementsAttribute = opaque {
    const M = Methods(DenseElementsAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsADenseElements;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub fn init(shaped_type: *const ShapedType, values: anytype) *const DenseElementsAttribute {
        const bytes = std.mem.sliceAsBytes(values);
        return @ptrCast(c.mlirDenseElementsAttrRawBufferGet(
            shaped_type.ptr(),
            @intCast(bytes.len),
            @ptrCast(bytes),
        ).ptr orelse unreachable);
    }
};

pub fn denseElementsAttribute(shaped_type: *const ShapedType, values: anytype) *const Attribute {
    return @ptrCast(DenseElementsAttribute.init(shaped_type, values));
}

pub const DenseArrayTypes = enum {
    bool,
    i8,
    i16,
    i32,
    i64,
    f32,
    f64,

    fn ZigType(comptime ElementType: @This()) type {
        return switch (ElementType) {
            .bool => i32,
            .i8 => i8,
            .i16 => i16,
            .i32 => i32,
            .i64 => i64,
            .f32 => f32,
            .f64 => f64,
        };
    }
};

pub fn DenseArrayAttribute(comptime ElementType: DenseArrayTypes) type {
    const ElementTypeZig = ElementType.ZigType();

    const _isAFn, const getFn, const getElementFn = switch (ElementType) {
        .bool => .{ c.mlirAttributeIsADenseBoolArray, c.mlirDenseBoolArrayGet, c.mlirDenseBoolArrayGetElement },
        .i8 => .{ c.mlirAttributeIsADenseI8Array, c.mlirDenseI8ArrayGet, c.mlirDenseI8ArrayGetElement },
        .i16 => .{ c.mlirAttributeIsADenseI16Array, c.mlirDenseI16ArrayGet, c.mlirDenseI16ArrayGetElement },
        .i32 => .{ c.mlirAttributeIsADenseI32Array, c.mlirDenseI32ArrayGet, c.mlirDenseI32ArrayGetElement },
        .i64 => .{ c.mlirAttributeIsADenseI64Array, c.mlirDenseI64ArrayGet, c.mlirDenseI64ArrayGetElement },
        .f32 => .{ c.mlirAttributeIsADenseF32Array, c.mlirDenseF32ArrayGet, c.mlirDenseF32ArrayGetElement },
        .f64 => .{ c.mlirAttributeIsADenseF64Array, c.mlirDenseF64ArrayGet, c.mlirDenseF64ArrayGetElement },
    };

    return struct {
        const Self = @This();

        const M = Methods(@This(), c.MlirAttribute);

        pub const isAFn = _isAFn;
        pub const ptr = M.ptr;
        pub const eql = M.eql(c.mlirAttributeEqual);
        pub const format = M.format(c.mlirAttributePrint);

        pub fn init(ctx: *Context, values: []const ElementTypeZig) *const Self {
            return @ptrCast(getFn(ctx.ptr(), @intCast(values.len), @ptrCast(values)).ptr);
        }

        pub fn element(self: *const Self, pos: usize) ElementTypeZig {
            return getElementFn(self.ptr(), @intCast(pos));
        }

        pub fn numElements(self: *const Self) usize {
            return @intCast(c.mlirDenseArrayGetNumElements(self.ptr()));
        }
    };
}

pub fn denseArrayAttribute(ctx: *Context, comptime ElementType: DenseArrayTypes, values: []const ElementType.ZigType()) *const Attribute {
    return @ptrCast(DenseArrayAttribute(ElementType).init(ctx, @ptrCast(values)));
}

pub const ArrayAttribute = opaque {
    const M = Methods(ArrayAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsAArray;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub fn init(ctx: *Context, attrs: []const *const Attribute) *const ArrayAttribute {
        return @ptrCast(c.mlirArrayAttrGet(ctx.ptr(), @intCast(attrs.len), @ptrCast(attrs)).ptr);
    }

    pub fn element(self: *const ArrayAttribute, index: usize) Attribute {
        return @ptrCast(c.mlirArrayAttrGetElement(self.ptr(), @intCast(index)).ptr);
    }

    pub fn numElements(self: *const ArrayAttribute) usize {
        return @intCast(c.mlirArrayAttrGetNumElements(self.ptr()));
    }
};

pub fn arrayAttribute(ctx: *Context, attrs: []const *const Attribute) *const Attribute {
    return @ptrCast(ArrayAttribute.init(ctx, attrs));
}

pub const FunctionType = opaque {
    const M = Methods(FunctionType, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFunction;

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);

    pub fn init(ctx: *Context, args: []const *const Type, results: []const *const Type) *const FunctionType {
        return @ptrCast(c.mlirFunctionTypeGet(
            ctx.ptr(),
            @intCast(args.len),
            @ptrCast(args.ptr),
            @intCast(results.len),
            @ptrCast(results.ptr),
        ).ptr);
    }
};

pub fn functionType(ctx: *Context, args: []const *const Type, results: []const *const Type) *const Type {
    return @ptrCast(FunctionType.init(ctx, args, results));
}

pub const FlatSymbolRefAttribute = opaque {
    const M = Methods(FlatSymbolRefAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsAFlatSymbolRef;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub fn init(ctx: *Context, symbol_: []const u8) *const FlatSymbolRefAttribute {
        return @ptrCast(c.mlirFlatSymbolRefAttrGet(ctx.ptr(), stringRef(symbol_)).ptr);
    }

    pub fn symbol(self: *const FlatSymbolRefAttribute) []const u8 {
        return string(c.mlirFlatSymbolRefAttrGetValue(self.ptr()));
    }
};

pub fn flatSymbolRefAttribute(ctx: *Context, symbol: []const u8) *const Attribute {
    return @ptrCast(FlatSymbolRefAttribute.init(ctx, symbol));
}

pub const UnitAttribute = opaque {
    const M = Methods(UnitAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsAUnit;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub fn get(ctx: *Context) *const UnitAttribute {
        return @ptrCast(c.mlirUnitAttrGet(ctx.ptr()).ptr);
    }
};

pub fn unitAttribute(ctx: *Context) *const Attribute {
    return @ptrCast(UnitAttribute.get(ctx));
}

pub const MemRefType = opaque {
    const M = Methods(MemRefType, c.MlirType);

    pub const isAFn = c.mlirTypeIsAInteger;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);

    pub fn init(element_type: *const Type, shape: []const i64, layout: ?*const Attribute, memory_space: ?*const Attribute) *const MemRefType {
        return @ptrCast(c.mlirMemRefTypeGet(
            element_type.ptr(),
            @intCast(shape.len),
            @ptrCast(shape),
            if (layout) |l| l.ptr() else .{},
            if (memory_space) |m| m.ptr() else .{},
        ).ptr);
    }

    pub fn fromShaped(other: *const ShapedType) *const MemRefType {
        var dims: stdx.BoundedArray(i64, ShapedType.MAX_RANK) = .{};
        for (0..other.rank()) |i| {
            dims.appendAssumeCapacity(other.dimension(i));
        }
        return init(
            other.elementType(),
            dims.constSlice(),
            null,
            null,
        );
    }

    pub fn shaped(self: *const MemRefType) *const ShapedType {
        return @ptrCast(self);
    }
};

pub fn memRefType(element_type: *const Type, shape: []const i64, layout: ?*const Attribute, memory_space: ?*const Attribute) *const Type {
    return @ptrCast(MemRefType.init(element_type, shape, layout, memory_space));
}
