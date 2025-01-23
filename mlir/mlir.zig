const builtin = @import("builtin");
const std = @import("std");
const log = std.log.scoped(.mlir);

const c = @import("c");

test {
    std.testing.refAllDecls(@This());
}

const Error = error{
    /// Invalid Mlir was created.
    InvalidMlir,
    /// Another Mlir error. Check the log for more context.
    MlirUnexpected,
    /// A resource/executor was not found.
    NotFound,
    OutOfMemory,
};

pub inline fn stringRef(str: []const u8) c.MlirStringRef {
    return .{ .data = str.ptr, .length = str.len };
}

pub inline fn fromStringRef(str: c.MlirStringRef) []const u8 {
    // Note: mlir.StringRef need not to be null terminated.
    return str.data[0..str.length];
}

pub fn registerPasses(comptime passes: []const u8) void {
    @field(c, "mlirRegister" ++ passes ++ "Passes")();
}

pub fn successOr(res: c.MlirLogicalResult, err: anytype) !void {
    return if (res.value == 0) err;
}

pub fn MlirTypeMethods(comptime InnerT: type) type {
    return struct {
        is_null_fn: ?fn (InnerT) callconv(.C) bool = null,
        is_a_fn: ?fn (InnerT) callconv(.C) bool = null,
        equal_fn: ?fn (InnerT, InnerT) callconv(.C) bool = null,
        dump_fn: ?fn (InnerT) callconv(.C) void = null,
        deinit_fn: ?fn (InnerT) callconv(.C) void = null,
    };
}

/// Alternative to MlirWrapperType
pub const MlirStrCallback = fn (c.MlirStringRef, ?*anyopaque) callconv(.C) void;

fn MlirHelpersMethods(OuterT: type) type {
    switch (@typeInfo(OuterT)) {
        .Struct => |info| {
            if (info.fields.len != 1) @compileError("Mlir wrapper type can only wrap one Mlir value. Received: " ++ @typeName(OuterT));
        },
        else => @compileError("MlirHelpersMethods is only available on an Mlir wrapper struct. Received: " ++ @typeName(OuterT)),
    }

    return struct {
        pub const InnerT = std.meta.FieldType(OuterT, ._inner);
        comptime {
            std.debug.assert(@sizeOf(InnerT) == @sizeOf(OuterT));
        }

        is_null_fn: ?fn (InnerT) callconv(.C) bool = null,
        is_a_fn: ?fn (InnerT) callconv(.C) bool = null,
        equal_fn: ?fn (InnerT, InnerT) callconv(.C) bool = null,
        deinit_fn: ?fn (InnerT) callconv(.C) void = null,
        dump_fn: ?fn (InnerT) callconv(.C) void = null,
        print_fn: ?fn (InnerT, ?*const MlirStrCallback, ?*anyopaque) callconv(.C) void = null,
    };
}

pub fn MlirHelpers(comptime OuterT: type, comptime methods: MlirHelpersMethods(OuterT)) type {
    const InnerT = @TypeOf(methods).InnerT;
    return struct {
        pub const Methods = methods;

        pub inline fn wrap(raw: InnerT) OuterT {
            return .{ ._inner = raw };
        }

        pub inline fn inner(self: OuterT) InnerT {
            return self._inner;
        }

        pub inline fn innerPtr(self: *OuterT) *InnerT {
            return &self._inner;
        }

        pub inline fn is_a(self: OuterT, comptime otherT: type) bool {
            if (otherT.Methods.is_a_fn) |is_a_fn| {
                return is_a_fn(self.inner());
            }
            return false;
        }

        pub inline fn as(self: OuterT, comptime OtherT: type) ?OtherT {
            if (OtherT.Methods.is_a_fn) |is_a_fn| {
                return if (is_a_fn(self.inner())) OtherT.wrap(self.inner()) else null;
            }
            // if the other type doesn't have an is_a_fn, try.
            return OtherT.wrap(self.inner());
        }

        pub usingnamespace if (Methods.is_null_fn) |is_null| struct {
            pub inline fn wrapOr(raw: InnerT) ?OuterT {
                return if (is_null(raw)) null else OuterT.wrap(raw);
            }
        } else struct {};

        pub usingnamespace if (Methods.equal_fn) |equal| struct {
            pub inline fn eql(self: OuterT, other: OuterT) bool {
                return equal(self.inner(), other.inner());
            }
        } else struct {};

        pub usingnamespace if (Methods.deinit_fn) |_deinit| struct {
            pub inline fn deinit(self: *OuterT) void {
                _deinit(self.inner());
                self.* = undefined;
            }
        } else struct {};

        pub usingnamespace if (Methods.dump_fn) |_dump| struct {
            pub inline fn dump(self: OuterT) void {
                return _dump(self.inner());
            }
        } else struct {};

        pub usingnamespace if (Methods.print_fn) |print| struct {
            pub fn format(
                self: OuterT,
                comptime fmt: []const u8,
                options: std.fmt.FormatOptions,
                writer: anytype,
            ) !void {
                _ = fmt;
                _ = options;

                const Writer = struct {
                    writer: @TypeOf(writer),
                    err: ?@TypeOf(writer).Error = null,
                    fn printCallback(mlir_str: c.MlirStringRef, opaque_ctx: ?*anyopaque) callconv(.C) void {
                        var ctx: *@This() = @alignCast(@ptrCast(opaque_ctx));
                        if (ctx.err) |_| return;
                        _ = ctx.writer.write(mlir_str.data[0..mlir_str.length]) catch |err| {
                            ctx.err = err;
                            return;
                        };
                    }
                };

                var context: Writer = .{ .writer = writer };
                print(self.inner(), &Writer.printCallback, &context);
                if (context.err) |err| return err;
            }
        } else struct {};
    };
}

pub const Registry = struct {
    _inner: c.MlirDialectRegistry,
    pub usingnamespace MlirHelpers(Registry, .{
        .is_null_fn = c.mlirDialectRegistryIsNull,
        .deinit_fn = c.mlirDialectRegistryDestroy,
    });
    const Self = Registry;

    pub fn init() !Self {
        return Self.wrapOr(c.mlirDialectRegistryCreate()) orelse Error.MlirUnexpected;
    }
};

pub const Context = struct {
    _inner: c.MlirContext,
    pub usingnamespace MlirHelpers(Context, .{
        .is_null_fn = c.mlirContextIsNull,
        .deinit_fn = c.mlirContextDestroy,
    });
    const Self = Context;

    pub fn init() !Self {
        return Self.wrapOr(c.mlirContextCreate()) orelse Error.MlirUnexpected;
    }

    pub fn initWithRegistry(registry: Registry, threadingEnabled: bool) !Self {
        return Self.wrapOr(
            c.mlirContextCreateWithRegistry(registry.inner(), threadingEnabled),
        ) orelse Error.InvalidMlir;
    }

    pub fn setMultiThreading(self: *Self, enabled: bool) void {
        c.mlirContextEnableMultithreading(self.inner(), enabled);
    }

    pub fn appendDialectRegistry(self: *Self, registry: Registry) void {
        c.mlirContextAppendDialectRegistry(self.inner(), registry.inner());
    }

    pub fn loadAllAvailableDialects(self: *Self) void {
        c.mlirContextLoadAllAvailableDialects(self.inner());
    }

    pub fn numRegisteredDialects(self: Self) usize {
        return @intCast(c.mlirContextGetNumRegisteredDialects(self.inner()));
    }

    pub fn numLoadedDialects(self: Self) usize {
        return @intCast(c.mlirContextGetNumLoadedDialects(self.inner()));
    }

    pub fn isRegisteredOperation(self: Self, op: [:0]const u8) bool {
        return c.mlirContextIsRegisteredOperation(self.inner(), stringRef(op));
    }

    pub fn location(self: Self, src: std.builtin.SourceLocation) Location {
        return Location.fromSrc(self, src);
    }
};

pub const Module = struct {
    _inner: c.MlirModule,
    pub usingnamespace MlirHelpers(Module, .{
        .is_null_fn = c.mlirModuleIsNull,
        .deinit_fn = c.mlirModuleDestroy,
    });
    const Self = Module;

    pub fn init(loc: Location) Self {
        return Self.wrap(c.mlirModuleCreateEmpty(loc.inner()));
    }

    pub fn parse(ctx: Context, source: [:0]const u8) !Module {
        return Module.wrapOr(
            c.mlirModuleCreateParse(ctx.inner(), stringRef(source)),
        ) orelse Error.InvalidMlir;
    }

    pub fn fromOperation(operation: Operation) Module {
        return Module.wrap(c.mlirModuleFromOperation(operation.inner()));
    }

    pub fn context(self: Module) Context {
        return Context.wrap(c.mlirModuleGetContext(self.inner()));
    }

    pub fn getBody(self: Module) Block {
        return Block.wrap(c.mlirModuleGetBody(self.inner()));
    }

    pub fn op(self: Module) Operation {
        return Operation.wrap(c.mlirModuleGetOperation(self.inner()));
    }
};

pub const PassManager = struct {
    _inner: c.MlirPassManager,

    pub usingnamespace MlirHelpers(PassManager, .{
        .is_null_fn = c.mlirPassManagerIsNull,
        .deinit_fn = c.mlirPassManagerDestroy,
    });
    const Self = PassManager;

    pub fn init(ctx: Context) !Self {
        return Self.wrapOr(
            c.mlirPassManagerCreate(ctx.inner()),
        ) orelse Error.MlirUnexpected;
    }

    pub fn initOnOperation(ctx: Context, op: [:0]const u8) !Self {
        return Self.wrapOr(
            c.mlirPassManagerCreateOnOperation(ctx.inner(), stringRef(op)),
        ) orelse Error.MlirUnexpected;
    }

    pub fn asOpPassManager(self: Self) OpPassManager {
        return OpPassManager.wrap(c.mlirPassManagerGetAsOpPassManager(self.inner()));
    }

    pub fn enableIRPrinting(self: *Self) void {
        c.mlirPassManagerEnableIRPrinting(self.inner());
    }

    pub fn runOnOp(self: *Self, op: Operation) error{InvalidMlir}!void {
        if (c.mlirPassManagerRunOnOp(self.inner(), op.inner()).value == 0) {
            return Error.InvalidMlir;
        }
    }
};

fn _mlir_passpipeline_error(err: c.MlirStringRef, ctx: ?*anyopaque) callconv(.C) void {
    _ = ctx;
    std.debug.print(">>ERROR: {s}\n", .{err.data});
}

pub const OpPassManager = struct {
    _inner: c.MlirOpPassManager,
    pub usingnamespace MlirHelpers(OpPassManager, .{});

    pub fn addPipeline(self: *OpPassManager, pipeline: [:0]const u8) error{OutOfMemory}!void {
        if (c.mlirOpPassManagerAddPipeline(
            self.inner(),
            stringRef(pipeline),
            &_mlir_passpipeline_error,
            null,
        ).value == 0) {
            return Error.OutOfMemory;
        }
    }
};

pub const Identifier = struct {
    _inner: c.MlirIdentifier,
    pub usingnamespace MlirHelpers(Identifier, .{});
    const Self = Identifier;

    pub fn get(ctx: Context, str_: [:0]const u8) Self {
        return Self.wrap(c.mlirIdentifierGet(ctx.inner(), stringRef(str_)));
    }

    pub fn context(self: Self) Context {
        return Context.wrap(c.mlirIdentifierGetContext(self.inner()));
    }

    pub fn str(self: Self) []const u8 {
        return fromStringRef(c.mlirIdentifierStr(self.inner()));
    }

    pub fn equals(self: Self, other: Self) bool {
        return c.mlirIdentifierEqual(self.inner(), other.inner());
    }
};

pub const AttrTuple = struct { [:0]const u8, Attribute };

pub const Attribute = struct {
    _inner: c.MlirAttribute,
    pub usingnamespace MlirHelpers(Attribute, .{
        .is_null_fn = c.mlirAttributeIsNull,
        .dump_fn = c.mlirAttributeDump,
        .equal_fn = c.mlirAttributeEqual,
    });
    const Self = Attribute;

    pub fn parse(ctx: Context, attr: [:0]const u8) !Attribute {
        return Attribute.wrapOr(
            c.mlirAttributeParseGet(ctx.inner(), stringRef(attr)),
        ) orelse Error.InvalidMlir;
    }

    pub fn getNull() Self {
        return Self.wrap(c.mlirAttributeGetNull());
    }
};

pub const NamedAttribute = struct {
    _inner: c.MlirNamedAttribute,
    pub usingnamespace MlirHelpers(NamedAttribute, .{});
    const Self = NamedAttribute;

    pub fn init(name: Identifier, attr: Attribute) Self {
        return Self.wrap(.{
            .name = name.inner(),
            .attribute = attr.inner(),
        });
    }
};

pub const StringAttribute = struct {
    _inner: c.MlirAttribute,
    pub usingnamespace MlirHelpers(StringAttribute, .{
        .is_a_fn = c.mlirAttributeIsAString,
        .is_null_fn = c.mlirAttributeIsNull,
        .dump_fn = c.mlirAttributeDump,
        .equal_fn = c.mlirAttributeEqual,
    });
    const Self = StringAttribute;

    pub fn init(ctx: Context, str: []const u8) Self {
        return Self.wrap(c.mlirStringAttrGet(ctx.inner(), stringRef(str)));
    }

    pub fn value(self: Self) []const u8 {
        return fromStringRef(c.mlirStringAttrGetValue(self.inner()));
    }

    pub fn asAttr(self: StringAttribute) Attribute {
        return .{ ._inner = self._inner };
    }
};

pub const UnitAttribute = struct {
    _inner: c.MlirAttribute,
    pub usingnamespace MlirHelpers(UnitAttribute, .{
        .is_a_fn = c.mlirAttributeIsAUnit,
        .is_null_fn = c.mlirAttributeIsNull,
        .dump_fn = c.mlirAttributeDump,
        .equal_fn = c.mlirAttributeEqual,
    });
    const Self = UnitAttribute;

    pub fn init(ctx: Context) Self {
        return Self.wrap(c.mlirUnitAttrGet(ctx.inner()));
    }
};

pub const BoolAttribute = struct {
    _inner: c.MlirAttribute,
    pub usingnamespace MlirHelpers(BoolAttribute, .{
        .is_a_fn = c.mlirAttributeIsABool,
        .is_null_fn = c.mlirAttributeIsNull,
        .dump_fn = c.mlirAttributeDump,
        .equal_fn = c.mlirAttributeEqual,
    });
    const Self = BoolAttribute;

    pub fn init(ctx: Context, value_: bool) Self {
        return Self.wrap(c.mlirBoolAttrGet(ctx.inner(), if (value_) 1 else 0));
    }

    pub fn value(self: Self) bool {
        return c.mlirBoolAttrGetValue(self.inner());
    }

    pub fn asAttr(self: Self) Attribute {
        return self.as(Attribute).?;
    }
};

pub const TypeAttribute = struct {
    _inner: c.MlirAttribute,
    pub usingnamespace MlirHelpers(TypeAttribute, .{
        .is_a_fn = c.mlirAttributeIsAType,
        .is_null_fn = c.mlirAttributeIsNull,
        .dump_fn = c.mlirAttributeDump,
        .equal_fn = c.mlirAttributeEqual,
    });
    pub fn init(type_: Type) TypeAttribute {
        return TypeAttribute.wrap(c.mlirTypeAttrGet(type_.inner()));
    }

    pub fn typ(self: TypeAttribute) Type {
        return Type.wrap(c.mlirAttributeGetType(self.inner()));
    }

    pub fn asAttr(self: TypeAttribute) Attribute {
        return self.as(Attribute).?;
    }
};

pub const ArrayAttribute = struct {
    _inner: c.MlirAttribute,
    pub usingnamespace MlirHelpers(ArrayAttribute, .{
        .is_a_fn = c.mlirAttributeIsAArray,
        .is_null_fn = c.mlirAttributeIsNull,
        .dump_fn = c.mlirAttributeDump,
        .equal_fn = c.mlirAttributeEqual,
    });
    const Self = ArrayAttribute;

    pub fn init(ctx: Context, attrs: []const Attribute) Self {
        return Self.wrap(c.mlirArrayAttrGet(ctx.inner(), @intCast(attrs.len), @ptrCast(attrs.ptr)));
    }

    pub fn size(self: Self) usize {
        return @intCast(c.mlirArrayAttrGetNumElements(self.inner()));
    }

    pub fn get(self: Self, index: usize) Attribute {
        return Attribute.wrap(c.mlirArrayAttrGetElement(self.inner(), @intCast(index)));
    }

    pub fn asAttr(self: Self) Attribute {
        return .{ ._inner = self._inner };
    }
};

pub fn IntegerAttribute(comptime it: IntegerTypes) type {
    const ZigType, const getter = comptime switch (it) {
        .i1, .i4, .i8, .i16, .i32, .i64 => .{ u64, c.mlirIntegerAttrGetValueInt },
        .si4, .si8, .si16, .si32, .si64 => .{ u64, c.mlirIntegerAttrGetValueSInt },
        .u4, .u8, .u16, .u32, .u64 => .{ u64, c.mlirIntegerAttrGetValueUInt },
        .unknown => @compileError("IntegerAttribute(unknown)"),
    };

    return struct {
        _inner: c.MlirAttribute,
        pub usingnamespace MlirHelpers(@This(), .{
            .is_a_fn = c.mlirAttributeIsAInteger,
            .is_null_fn = c.mlirAttributeIsNull,
            .dump_fn = c.mlirAttributeDump,
            .equal_fn = c.mlirAttributeEqual,
        });
        pub const IntegerTypeType = IntegerType(it);
        const IntAttr = @This();

        pub fn init(ctx: Context, value: i64) IntAttr {
            return IntAttr.wrap(c.mlirIntegerAttrGet(
                IntegerType(it).init(ctx).inner(),
                value,
            ));
        }

        pub fn get(value: IntAttr) ZigType {
            return @intCast(getter(value.inner()));
        }

        pub fn asAttr(self: IntAttr) Attribute {
            return .{ ._inner = self._inner };
        }
    };
}

pub fn FloatAttribute(comptime ft: FloatTypes) type {
    return struct {
        _inner: c.MlirAttribute,
        pub usingnamespace MlirHelpers(@This(), .{
            .is_a_fn = c.mlirAttributeIsAFloat,
            .is_null_fn = c.mlirAttributeIsNull,
            .dump_fn = c.mlirAttributeDump,
            .equal_fn = c.mlirAttributeEqual,
        });
        const FloatAttr = @This();
        pub fn init(ctx: Context, value: f64) FloatAttr {
            return FloatAttr.wrap(c.mlirFloatAttrDoubleGet(
                ctx.inner(),
                FloatType(ft).init(ctx).inner(),
                value,
            ));
        }

        pub fn get(value: FloatAttr) f64 {
            return c.mlirFloatAttrGetValueDouble(value.inner());
        }
    };
}

pub const DenseArrayTypes = enum {
    bool,
    i8,
    i16,
    i32,
    i64,
    f32,
    f64,
};

pub fn DenseArrayAttribute(comptime dt: DenseArrayTypes) type {
    const Config = switch (dt) {
        .bool => .{ i32, c.mlirAttributeIsADenseBoolArray, c.mlirDenseBoolArrayGet, c.mlirDenseBoolArrayGetElement },
        .i8 => .{ i8, c.mlirAttributeIsADenseI8Array, c.mlirDenseI8ArrayGet, c.mlirDenseI8ArrayGetElement },
        .i16 => .{ i16, c.mlirAttributeIsADenseI16Array, c.mlirDenseI16ArrayGet, c.mlirDenseI16ArrayGetElement },
        .i32 => .{ i32, c.mlirAttributeIsADenseI32Array, c.mlirDenseI32ArrayGet, c.mlirDenseI32ArrayGetElement },
        .i64 => .{ i64, c.mlirAttributeIsADenseI64Array, c.mlirDenseI64ArrayGet, c.mlirDenseI64ArrayGetElement },
        .f32 => .{ f32, c.mlirAttributeIsADenseF32Array, c.mlirDenseF32ArrayGet, c.mlirDenseF32ArrayGetElement },
        .f64 => .{ f64, c.mlirAttributeIsADenseF64Array, c.mlirDenseF64ArrayGet, c.mlirDenseF64ArrayGetElement },
    };

    return struct {
        _inner: c.MlirAttribute,
        pub usingnamespace MlirHelpers(@This(), .{
            .is_a_fn = Config[1],
            .is_null_fn = c.mlirAttributeIsNull,
            .dump_fn = c.mlirAttributeDump,
            .equal_fn = c.mlirAttributeEqual,
        });
        const Attr = @This();
        const ElementType = dt;
        const ElementTypeZig = Config[0];

        pub fn init(ctx: Context, values: []const ElementTypeZig) Attr {
            const get_fn = Config[2];
            return Attr.wrap(get_fn(ctx.inner(), @intCast(values.len), @ptrCast(values.ptr)));
        }

        pub fn get(self: Attr, pos: usize) ElementTypeZig {
            const get_element_fn = Config[3];
            return get_element_fn(self.inner(), @intCast(pos));
        }

        pub fn len(self: Attr) usize {
            return @intCast(c.mlirDenseArrayGetNumElements(self.inner()));
        }

        pub usingnamespace switch (dt) {
            .bool, .i64 => struct {
                const DenseArray = DenseArrayAttribute(switch (dt) {
                    .bool => .bool,
                    .i64 => .i64,
                    else => @compileError("DenseArrayAttribute: unreachable"),
                });

                pub fn toElements(self: Attr) DenseArray {
                    return DenseArray.wrap(c.mlirDenseArrayToElements(self.inner()));
                }
            },
            else => struct {},
        };
    };
}

pub const DenseElementsAttributeTypes = enum {
    bool,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    bf16,
    f16,
    f32,
    f64,
};

pub fn DenseIntOrFPElementsAttribute(comptime dt: DenseElementsAttributeTypes) type {
    const ZigInDataType, const ZigOutDataType, const initFn, const getValue = switch (dt) {
        .bool => .{ bool, bool, c.mlirDenseElementsAttrBoolGet, c.mlirDenseElementsAttrGetBoolValue },
        .i8 => .{ i8, i8, c.mlirDenseElementsAttrInt8Get, c.mlirDenseElementsAttrGetInt8Value },
        .i16 => .{ i16, i16, c.mlirDenseElementsAttrInt16Get, c.mlirDenseElementsAttrGetInt16Value },
        .i32 => .{ i32, i32, c.mlirDenseElementsAttrInt32Get, c.mlirDenseElementsAttrGetInt32Value },
        .i64 => .{ i64, i64, c.mlirDenseElementsAttrInt64Get, c.mlirDenseElementsAttrGetInt64Value },
        .u8 => .{ u8, u8, c.mlirDenseElementsAttrUInt8Get, c.mlirDenseElementsAttrGetUInt8Value },
        .u16 => .{ u16, u16, c.mlirDenseElementsAttrUInt16Get, c.mlirDenseElementsAttrGetUInt16Value },
        .u32 => .{ u32, u32, c.mlirDenseElementsAttrUInt32Get, c.mlirDenseElementsAttrGetUInt32Value },
        .u64 => .{ u64, u64, c.mlirDenseElementsAttrUInt64Get, c.mlirDenseElementsAttrGetUInt64Value },
        .bf16 => .{ u16, f32, c.mlirDenseElementsAttrBFloat16Get, c.mlirDenseElementsAttrGetFloatValue },
        .f16 => .{ f16, f32, c.mlirDenseElementsAttrFloat16Get, c.mlirDenseElementsAttrGetFloatValue },
        .f32 => .{ f32, f32, c.mlirDenseElementsAttrFloatGet, c.mlirDenseElementsAttrGetFloatValue },
        .f64 => .{ f64, f64, c.mlirDenseElementsAttrDoubleGet, c.mlirDenseElementsAttrGetDoubleValue },
    };

    return struct {
        _inner: c.MlirAttribute,
        const Attr = @This();
        pub usingnamespace MlirHelpers(Attr, .{
            .is_a_fn = c.mlirAttributeIsADenseElements,
            .is_null_fn = c.mlirAttributeIsNull,
            .dump_fn = c.mlirAttributeDump,
            .equal_fn = c.mlirAttributeEqual,
        });

        pub fn init(shaped_type: Type, raw_values: []const u8) Attr {
            const values = std.mem.bytesAsSlice(ZigInDataType, raw_values);
            return Attr.wrap(initFn(shaped_type.inner(), @intCast(values.len), @ptrCast(@alignCast(values.ptr))));
        }

        pub fn get(self: Attr, pos: usize) ZigOutDataType {
            return getValue(self.inner(), @intCast(pos));
        }
    };
}

pub const FlatSymbolRefAttribute = struct {
    _inner: c.MlirAttribute,
    pub usingnamespace MlirHelpers(FlatSymbolRefAttribute, .{
        .is_a_fn = c.mlirAttributeIsAFlatSymbolRef,
        .is_null_fn = c.mlirAttributeIsNull,
        .dump_fn = c.mlirAttributeDump,
        .equal_fn = c.mlirAttributeEqual,
    });

    const Self = FlatSymbolRefAttribute;

    pub fn init(ctx: Context, str: [:0]const u8) Self {
        return Self.wrap(c.mlirFlatSymbolRefAttrGet(ctx.inner(), stringRef(str)));
    }

    pub fn value(self: Self) []const u8 {
        return fromStringRef(c.mlirFlatSymbolRefAttrGetValue(self.inner()));
    }
};

pub const OperationState = struct {
    _inner: c.MlirOperationState,
    pub usingnamespace MlirHelpers(
        OperationState,
        .{},
    );

    const Self = OperationState;

    pub fn init(name: [:0]const u8, loc: Location) Self {
        return Self.wrap(c.mlirOperationStateGet(stringRef(name), loc.inner()));
    }

    pub fn addResult(self: *Self, type_: Type) void {
        c.mlirOperationStateAddResults(self.innerPtr(), 1, &[_]c.MlirType{type_.inner()});
    }

    pub fn addResults(self: *Self, types: []const Type) void {
        c.mlirOperationStateAddResults(self.innerPtr(), @intCast(types.len), @ptrCast(types.ptr));
    }

    pub fn addOperand(self: *Self, value: Value) void {
        c.mlirOperationStateAddOperands(self.innerPtr(), 1, &[_]c.MlirValue{value.inner()});
    }

    pub fn addOperands(self: *Self, values: []const Value) void {
        c.mlirOperationStateAddOperands(self.innerPtr(), @intCast(values.len), @ptrCast(values.ptr));
    }

    pub fn addRegion(self: *Self, region: *Region) void {
        c.mlirOperationStateAddOwnedRegions(self.innerPtr(), 1, &[_]c.MlirRegion{region.inner()});
    }

    pub fn addRegions(self: *Self, regions: []const Region) void {
        c.mlirOperationStateAddOwnedRegions(self.innerPtr(), @intCast(regions.len), @ptrCast(regions.ptr));
    }

    pub fn addAttribute(self: *Self, ctx: Context, name: [:0]const u8, attr: Attribute) void {
        c.mlirOperationStateAddAttributes(self.innerPtr(), 1, @ptrCast(&.{
            .{
                .name = Identifier.get(ctx, name).inner(),
                .attribute = attr.inner(),
            },
        }));
    }

    pub fn addAttributeRaw(self: *Self, name: Identifier, attr: Attribute) void {
        c.mlirOperationStateAddAttributes(self.innerPtr(), 1, @ptrCast(&.{
            .{
                .name = name.inner(),
                .attribute = attr.inner(),
            },
        }));
    }

    pub fn addAttributes(self: *Self, attributes: []const NamedAttribute) void {
        c.mlirOperationStateAddAttributes(self.innerPtr(), @intCast(attributes.len), @ptrCast(attributes.ptr));
    }

    pub fn resultTypeInference(self: *Self, enabled: bool) void {
        self.innerPtr().enableResultTypeInference = enabled;
    }
};

pub const DictionaryAttribute = struct {
    _inner: c.MlirAttribute,
    pub usingnamespace MlirHelpers(DictionaryAttribute, .{
        .is_a_fn = c.mlirAttributeIsADictionary,
        .is_null_fn = c.mlirAttributeIsNull,
        .dump_fn = c.mlirAttributeDump,
        .equal_fn = c.mlirAttributeEqual,
    });

    pub fn init(ctx: Context, attributes: []const NamedAttribute) DictionaryAttribute {
        return DictionaryAttribute.wrap(c.mlirDictionaryAttrGet(
            ctx.inner(),
            @intCast(attributes.len),
            @ptrCast(attributes.ptr),
        ));
    }

    pub fn size(self: DictionaryAttribute) usize {
        return @intCast(c.mlirDictionaryAttrGetNumElements(self.inner()));
    }

    pub fn get(self: DictionaryAttribute, pos: usize) NamedAttribute {
        return NamedAttribute.wrap(c.mlirDictionaryAttrGetElement(self.inner(), @intCast(pos)));
    }

    pub fn getByName(self: DictionaryAttribute, name: [:0]const u8) ?NamedAttribute {
        return NamedAttribute.wrapOr(c.mlirDictionaryAttrGetElementByName(self.inner(), name));
    }

    pub fn asAttr(self: DictionaryAttribute) Attribute {
        return .{ ._inner = self._inner };
    }
};

pub const Operation = struct {
    const Self = Operation;
    _inner: c.MlirOperation,

    pub usingnamespace MlirHelpers(
        Operation,
        .{
            .is_null_fn = c.mlirOperationIsNull,
            .deinit_fn = c.mlirOperationDestroy,
            .dump_fn = c.mlirOperationDump,
            .equal_fn = c.mlirOperationEqual,
        },
    );

    pub fn init(state: *OperationState) !Self {
        return Self.wrapOr(
            c.mlirOperationCreate(state.innerPtr()),
        ) orelse Error.InvalidMlir;
    }

    pub fn make(ctx: Context, op_name: [:0]const u8, args: struct {
        operands: ?[]const Value = null,
        variadic_operands: ?[]const []const Value = null,
        results: ?[]const Type = null,
        variadic_results: ?[]const []const Type = null,
        result_type_inference: ?bool = null,
        n_regions: usize = 0,
        attributes: ?[]const AttrTuple = null,
        blocks: ?[]const Block = null,
        verify: bool = true,
        location: Location,
    }) Self {
        var state = OperationState.init(op_name, args.location);
        std.debug.assert(!(args.operands != null and args.variadic_operands != null));
        if (args.operands) |operands| {
            state.addOperands(operands);
        } else if (args.variadic_operands) |operands_segments| {
            for (operands_segments) |operands| {
                state.addOperands(operands);
            }
        }
        if (args.result_type_inference) |enable| {
            state.resultTypeInference(enable);
        }
        std.debug.assert(!(args.results != null and args.variadic_results != null));
        if (args.results) |results| {
            state.addResults(results);
        } else if (args.variadic_results) |result_segments| {
            for (result_segments) |results| {
                state.addResults(results);
            }
        }
        for (0..args.n_regions) |_| {
            var region_ = Region.init() catch {
                @panic("Failed to create MLIR region");
            };
            state.addRegion(&region_);
        }
        if (args.attributes) |attrs| {
            for (attrs) |attr| {
                state.addAttributeRaw(
                    Identifier.get(ctx, attr[0]),
                    attr[1],
                );
            }
        }
        if (args.blocks) |blocks_| {
            for (blocks_) |block_| {
                var region_ = Region.init() catch {
                    @panic("Failed to create MLIR region");
                };
                region_.appendBlock(block_);
                state.addRegion(&region_);
            }
        }

        const new_op = Operation.init(&state) catch {
            @panic("Failed to create MLIR operation");
        };
        if (args.verify and new_op.verify() == false) {
            log.err("Failed to verify MLIR operation:\n{}", .{new_op.mlirFormatter(.{ .debug_info = true })});
            @panic("Failed to verify MLIR operation");
        }
        return new_op;
    }

    pub fn initParse(ctx: Context, str: [:0]const u8) !Self {
        return Self.wrapOr(
            c.mlirOperationCreateParse(ctx.inner(), stringRef(str), stringRef("pouet")),
        ) orelse Error.InvalidMlir;
    }

    pub fn clone(self: Self) !Self {
        return Self.wrapOr(
            c.mlirOperationClone(self.inner()),
        ) orelse Error.InvalidMlir;
    }

    pub fn name(self: Self) Identifier {
        return Identifier.wrap(c.mlirOperationGetName(self.inner()));
    }

    pub fn removeFromParent(self: *Self) void {
        c.mlirOperationRemoveFromParent(self.inner());
    }

    pub fn numOperands(self: Self) usize {
        return @intCast(c.mlirOperationGetNumOperands(self.inner()));
    }

    pub fn operand(self: Self, index: usize) Value {
        return Value.wrap(c.mlirOperationGetOperand(self.inner(), @intCast(index)));
    }

    pub fn setOperand(self: *Self, index: usize, value: Value) void {
        c.mlirOperationSetOperand(self.inner(), @intCast(index), value.inner());
    }

    pub fn numResults(self: Self) usize {
        return @intCast(c.mlirOperationGetNumResults(self.inner()));
    }

    pub fn result(self: Self, index: usize) Value {
        return Value.wrap(c.mlirOperationGetResult(self.inner(), @intCast(index)));
    }

    pub fn nextInBlock(self: Self) Self {
        return Self.wrap(c.mlirOperationGetNextInBlock(self.inner()));
    }

    // pub fn previousInBlock(self: Self) Self {
    //     return Self.wrap(c.mlirOperationGetPrevInBlock(self.inner()));
    // }

    pub fn block(self: Self) ?Block {
        return Block.wrapOr(c.mlirOperationGetBlock(self.inner()));
    }

    pub fn parent(self: Self) ?Self {
        return Self.wrapOr(c.mlirOperationGetParentOperation(self.inner()));
    }

    pub fn region(self: Self, index: usize) Region {
        return Region.wrap(c.mlirOperationGetRegion(self.inner(), @intCast(index)));
    }

    pub fn context(self: Self) Context {
        return Context.wrap(c.mlirOperationGetContext(self.inner()));
    }

    pub fn writeBytecode(self: Self, writer: anytype) void {
        var writer_context = .{ .writer = writer };
        const WriterContext = @TypeOf(writer_context);

        c.mlirOperationWriteBytecode(
            self.inner(),
            (struct {
                pub fn callback(str: c.MlirStringRef, ctx_: ?*anyopaque) callconv(.C) void {
                    const inner_writer_context: *WriterContext = @ptrCast(@alignCast(ctx_));
                    _ = inner_writer_context.writer.write(str.data[0..str.length]) catch unreachable;
                }
            }).callback,
            &writer_context,
        );
    }

    pub fn writeBytecodeWithConfig(self: Self, writer: anytype, config: struct {
        desiredEmitedVersion: ?i64 = null,
    }) !void {
        const cfg = c.mlirBytecodeWriterConfigCreate();
        defer c.mlirBytecodeWriterConfigDestroy(cfg);
        if (config.desiredEmitedVersion) |v| {
            c.mlirBytecodeWriterConfigDesiredEmitVersion(cfg, v);
        }

        const WriterContext = struct {
            writer: @TypeOf(writer),
            write_error: ?@TypeOf(writer).Error = null,
        };
        var writer_context: WriterContext = .{ .writer = writer };

        try successOr(c.mlirOperationWriteBytecodeWithConfig(
            self.inner(),
            cfg,
            (struct {
                pub fn callback(str: c.MlirStringRef, ctx_: ?*anyopaque) callconv(.C) void {
                    const inner_writer_context: *WriterContext = @ptrCast(@alignCast(ctx_));
                    _ = inner_writer_context.writer.write(str.data[0..str.length]) catch |err| {
                        inner_writer_context.write_error = err;
                    };
                }
            }).callback,
            &writer_context,
        ), error.InvalidMlirBytecodeVersion);

        if (writer_context.write_error) |err| return err;
    }

    /// Enable a full dump of the IR.
    /// Usage `std.log.debug("{}", .{ module.op().mlirFormatter(.{}) });
    pub fn mlirFormatter(self: Operation, flags: OpPrintingFlags) MlirFormatter {
        return .{ .op = self, .flags = flags };
    }

    const MlirFormatter = struct {
        op: Operation,
        flags: OpPrintingFlags,

        pub fn format(self: @This(), comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;
            self.op.print(writer, self.flags);
        }
    };

    pub fn print(self: Self, writer: anytype, flags: OpPrintingFlags) void {
        const pflags = flags.create();
        defer c.mlirOpPrintingFlagsDestroy(pflags);

        var writer_context = .{ .writer = writer };
        const WriterContext = @TypeOf(writer_context);
        c.mlirOperationPrintWithFlags(
            self.inner(),
            pflags,
            (struct {
                pub fn callback(str: c.MlirStringRef, ctx_: ?*anyopaque) callconv(.C) void {
                    const inner_writer_context: *WriterContext = @ptrCast(@alignCast(ctx_));
                    _ = inner_writer_context.writer.write(str.data[0..str.length]) catch unreachable;
                }
            }).callback,
            &writer_context,
        );
    }

    pub fn verify(self: Self) bool {
        return c.mlirOperationVerify(self.inner());
    }

    pub fn getLocation(self: Self) Location {
        return Location.wrap(c.mlirOperationGetLocation(self.inner()));
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

    pub fn walk(self: Self, order: WalkOrder, ctx: anytype, walkfn: fn (ctx: anytype, op: Operation) WalkResult) void {
        var inner_ctx = .{ .ctx = ctx };
        const ContextType = @TypeOf(inner_ctx);

        c.mlirOperationWalk(
            self.inner(),
            (struct {
                pub fn callback(op: c.MlirOperation, ctx_: ?*anyopaque) callconv(.C) c.MlirWalkResult {
                    const inner_ctx_: *ContextType = @ptrCast(@alignCast(ctx_));
                    return @intFromEnum(walkfn(inner_ctx_.ctx, Operation.wrap(op)));
                }
            }).callback,
            &inner_ctx,
            @intFromEnum(order),
        );
    }

    pub fn getAttribute(self: Self, pos: usize) NamedAttribute {
        return NamedAttribute.wrap(c.mlirOperationGetAttribute(self.inner(), @intCast(pos)));
    }

    pub fn getAttributeByName(self: Self, name_: [:0]const u8) ?Attribute {
        return Attribute.wrapOr(c.mlirOperationGetAttributeByName(self.inner(), stringRef(name_)));
    }

    pub fn setAttributeByName(self: Self, name_: [:0]const u8, attr: Attribute) void {
        c.mlirOperationSetAttributeByName(self.inner(), stringRef(name_), attr.inner());
    }

    pub fn removeAttributeByName(self: Self, name_: [:0]const u8) bool {
        return c.mlirOperationRemoveAttributeByName(self.inner(), stringRef(name_));
    }
};

pub const OpPrintingFlags = struct {
    elide_large_elements_attrs: ?usize = null,
    debug_info: bool = false,
    debug_info_pretty_form: bool = true,
    print_generic_op_form: bool = false,
    use_local_scope: bool = false,
    assume_verified: bool = false,

    pub fn create(self: OpPrintingFlags) c.MlirOpPrintingFlags {
        const pflags = c.mlirOpPrintingFlagsCreate();
        if (self.elide_large_elements_attrs) |v| {
            c.mlirOpPrintingFlagsElideLargeElementsAttrs(pflags, @intCast(v));
        }
        c.mlirOpPrintingFlagsEnableDebugInfo(pflags, self.debug_info, self.debug_info_pretty_form);
        if (self.print_generic_op_form) {
            c.mlirOpPrintingFlagsPrintGenericOpForm(pflags);
        }
        if (self.use_local_scope) {
            c.mlirOpPrintingFlagsUseLocalScope(pflags);
        }
        if (self.assume_verified) {
            c.mlirOpPrintingFlagsAssumeVerified(pflags);
        }
        return pflags;
    }
};

pub const OpOperand = struct {
    _inner: c.MlirOpOperand,
    pub usingnamespace MlirHelpers(OpOperand, .{
        .is_null_fn = c.mlirOpOperandIsNull,
    });

    const Self = OpOperand;

    pub fn owner(self: Self) Operation {
        return Operation.wrap(c.mlirOpOperandGetOwner(self.inner()));
    }

    pub fn number(self: Self) usize {
        return @intCast(c.mlirOpOperandGetOperandNumber(self.inner()));
    }

    pub fn nextUse(self: Self) ?Self {
        return Self.wrapOr(
            c.mlirOpOperandGetNextUse(self.inner()),
        );
    }
};

pub const Region = struct {
    _inner: c.MlirRegion,
    pub usingnamespace MlirHelpers(Region, .{
        .is_null_fn = c.mlirRegionIsNull,
        .deinit_fn = c.mlirRegionDestroy,
        .equal_fn = c.mlirRegionEqual,
    });

    const Self = Region;

    pub fn init() !Self {
        return Self.wrapOr(
            c.mlirRegionCreate(),
        ) orelse Error.InvalidMlir;
    }

    pub fn appendBlock(self: *Self, block: Block) void {
        c.mlirRegionAppendOwnedBlock(self.inner(), block.inner());
    }

    pub fn insertBlock(self: *Self, index: isize, block: Block) void {
        c.mlirRegionInsertOwnedBlock(self.inner(), index, block.inner());
    }

    pub fn insertBlockBefore(self: *Self, reference: Block, block: Block) void {
        c.mlirRegionInsertOwnedBlockBefore(self.inner(), reference.inner(), block.inner());
    }

    pub fn insertBlockAfter(self: *Self, reference: Block, block: Block) void {
        c.mlirRegionInsertOwnedBlockAfter(self.inner(), reference.inner(), block.inner());
    }

    pub fn firstBlock(self: Self) Block {
        return Block.wrap(c.mlirRegionGetFirstBlock(self.inner()));
    }
};

pub const Value = struct {
    _inner: c.MlirValue,

    pub usingnamespace MlirHelpers(Value, .{
        .is_null_fn = c.mlirValueIsNull,
        .equal_fn = c.mlirValueEqual,
        .dump_fn = c.mlirValueDump,
        .print_fn = c.mlirValuePrint,
    });

    pub fn getType(val: Value) Type {
        return Type.wrap(c.mlirValueGetType(val.inner()));
    }

    pub fn setType(val: *Value, typ: Type) void {
        c.mlirValueSetType(val.inner(), typ.inner());
    }

    pub fn firstUse(val: Value) OpOperand {
        return OpOperand.wrap(c.mlirValueGetFirstUse(val.inner()));
    }

    pub fn replaceAllUsesWith(val: Value, with: Value) void {
        c.mlirValueReplaceAllUsesOfWith(val.inner(), with.inner());
    }

    pub fn owner(val: Value) Operation {
        return Operation.wrap(c.mlirOpResultGetOwner(val.inner()));
    }

    pub fn isABlockArgument(val: Value) bool {
        return c.mlirValueIsABlockArgument(val.inner());
    }

    pub fn isAOpResult(val: Value) bool {
        return c.mlirValueIsAOpResult(val.inner());
    }

    pub const Kind = union(enum) {
        block_argument: BlockArgument,
        op_result: Operation,
        null,
    };

    pub fn kind(val: Value) Kind {
        if (val.isAOpResult()) {
            return .{ .op_result = val.owner() };
        }
        if (val.isABlockArgument()) {
            return .{ .block_argument = .{ ._inner = val._inner } };
        }
        // From MLIR docs:
        // https://mlir.llvm.org/doxygen/classmlir_1_1Value.html#details
        // > An SSA value is either a BlockArgument or the result of an operation.
        return .null;
    }
};

pub const BlockArgument = struct {
    _inner: c.MlirValue,

    pub fn block(arg: BlockArgument) Block {
        return Block.wrap(c.mlirBlockArgumentGetOwner(arg._inner));
    }

    pub fn number(arg: BlockArgument) usize {
        return @bitCast(c.mlirBlockArgumentGetArgNumber(arg._inner));
    }

    pub fn format(self: BlockArgument, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        const value = Value{ ._inner = self._inner };
        return value.format(fmt, options, writer);
    }
};

pub const Type = struct {
    _inner: c.MlirType,

    pub usingnamespace MlirHelpers(Type, .{
        .is_null_fn = c.mlirTypeIsNull,
        .dump_fn = c.mlirTypeDump,
        .equal_fn = c.mlirTypeEqual,
        .print_fn = c.mlirTypePrint,
    });

    pub fn parse(ctx: Context, str: [:0]const u8) !Type {
        return Type.wrapOr(
            c.mlirTypeParseGet(ctx.inner(), stringRef(str)),
        ) orelse Error.InvalidMlir;
    }
};

pub const IntegerTypes = enum {
    i1,
    i4,
    i8,
    i16,
    i32,
    i64,
    si4,
    si8,
    si16,
    si32,
    si64,
    u4,
    u8,
    u16,
    u32,
    u64,

    unknown,
};

pub fn IntegerType(comptime it: IntegerTypes) type {
    const Config = switch (it) {
        .i1 => .{ 1, c.mlirIntegerTypeGet, c.mlirIntegerTypeIsSignless },
        .i4 => .{ 4, c.mlirIntegerTypeGet, c.mlirIntegerTypeIsSignless },
        .i8 => .{ 8, c.mlirIntegerTypeGet, c.mlirIntegerTypeIsSignless },
        .i16 => .{ 16, c.mlirIntegerTypeGet, c.mlirIntegerTypeIsSignless },
        .i32 => .{ 32, c.mlirIntegerTypeGet, c.mlirIntegerTypeIsSignless },
        .i64 => .{ 64, c.mlirIntegerTypeGet, c.mlirIntegerTypeIsSignless },
        .si4 => .{ 4, c.mlirIntegerTypeSignedGet, c.mlirIntegerTypeIsSigned },
        .si8 => .{ 8, c.mlirIntegerTypeSignedGet, c.mlirIntegerTypeIsSigned },
        .si16 => .{ 16, c.mlirIntegerTypeSignedGet, c.mlirIntegerTypeIsSigned },
        .si32 => .{ 32, c.mlirIntegerTypeSignedGet, c.mlirIntegerTypeIsSigned },
        .si64 => .{ 64, c.mlirIntegerTypeSignedGet, c.mlirIntegerTypeIsSigned },
        .u4 => .{ 4, c.mlirIntegerTypeUnsignedGet, c.mlirIntegerTypeIsUnsigned },
        .u8 => .{ 8, c.mlirIntegerTypeUnsignedGet, c.mlirIntegerTypeIsUnsigned },
        .u16 => .{ 16, c.mlirIntegerTypeUnsignedGet, c.mlirIntegerTypeIsUnsigned },
        .u32 => .{ 32, c.mlirIntegerTypeUnsignedGet, c.mlirIntegerTypeIsUnsigned },
        .u64 => .{ 64, c.mlirIntegerTypeUnsignedGet, c.mlirIntegerTypeIsUnsigned },
        .unknown => .{ 0, null, null },
    };

    return struct {
        _inner: c.MlirType,

        const Int = @This();
        pub usingnamespace MlirHelpers(Int, .{
            .is_a_fn = switch (it) {
                .unknown => c.mlirTypeIsAInteger,
                else => typeIsAIntegerExact,
            },
            .is_null_fn = c.mlirTypeIsNull,
            .dump_fn = c.mlirTypeDump,
            .equal_fn = c.mlirTypeEqual,
        });
        const IntegerTypeType = it;

        fn typeIsAIntegerExact(typ: c.MlirType) callconv(.C) bool {
            const bit_width = Config[0];
            const is_sign = Config[2];
            return c.mlirTypeIsAInteger(typ) and (c.mlirIntegerTypeGetWidth(typ) == bit_width) and is_sign(typ);
        }
        pub usingnamespace if (it != .unknown) struct {
            pub const BitWidth = Config[0];

            pub fn init(ctx: Context) Int {
                const type_get = Config[1];
                return Int.wrap(type_get(ctx.inner(), BitWidth));
            }
        } else struct {};
    };
}

pub const FloatTypes = enum {
    f8e4m3b11fnuz,
    f8e4m3fn,
    f8e4m3fnuz,
    f8e5m2,
    f8e5m2fnuz,
    bf16,
    f16,
    f32,
    f64,

    unknown,

    pub fn asType(self: FloatTypes, ctx: Context) ?Type {
        return switch (self) {
            .unknown => null,
            inline else => |ft| FloatType(ft).init(ctx).asType(),
        };
    }
};

pub fn FloatType(comptime ft: FloatTypes) type {
    const Config = switch (ft) {
        .f8e4m3b11fnuz => .{ c.mlirTypeIsAFloat8E4M3B11FNUZ, c.mlirFloat8E4M3B11FNUZTypeGet },
        .f8e4m3fn => .{ c.mlirTypeIsAFloat8E4M3FN, c.mlirFloat8E4M3FNTypeGet },
        .f8e4m3fnuz => .{ c.mlirTypeIsAFloat8E4M3FNUZ, c.mlirFloat8E4M3FNUZTypeGet },
        .f8e5m2 => .{ c.mlirTypeIsAFloat8E5M2, c.mlirFloat8E5M2TypeGet },
        .f8e5m2fnuz => .{ c.mlirTypeIsAFloat8E5M2FNUZ, c.mlirFloat8E5M2FNUZTypeGet },
        .bf16 => .{ c.mlirTypeIsABF16, c.mlirBF16TypeGet },
        .f16 => .{ c.mlirTypeIsAF16, c.mlirF16TypeGet },
        .f32 => .{ c.mlirTypeIsAF32, c.mlirF32TypeGet },
        .f64 => .{ c.mlirTypeIsAF64, c.mlirF64TypeGet },
        .unknown => .{ null, null },
    };

    return struct {
        _inner: c.MlirType,
        const Float = @This();
        pub usingnamespace MlirHelpers(Float, .{
            .is_a_fn = switch (ft) {
                .unknown => typeIsAUnknownFloat,
                else => Config[0],
            },
            .is_null_fn = c.mlirTypeIsNull,
            .dump_fn = c.mlirTypeDump,
            .equal_fn = c.mlirTypeEqual,
        });

        pub usingnamespace if (ft != .unknown) struct {
            pub const FloatTypeType = ft;

            pub fn init(ctx: Context) Float {
                const type_get = Config[1];
                return Float.wrap(type_get(ctx.inner()));
            }
        } else struct {};

        fn typeIsAUnknownFloat(typ: c.MlirType) callconv(.C) bool {
            const is_a_fns = .{
                c.mlirTypeIsABF16,
                c.mlirTypeIsAF16,
                c.mlirTypeIsAF32,
                c.mlirTypeIsF64,
            };
            inline for (is_a_fns) |is_a_fn| {
                if (is_a_fn(typ)) {
                    return true;
                }
            }
            return false;
        }

        pub fn asType(self: Float) Type {
            return self.as(Type).?;
        }
    };
}

pub const ComplexTypes = enum {
    c64,
    c128,

    unknown,
};

pub fn ComplexType(comptime ct: ComplexTypes) type {
    return struct {
        _inner: c.MlirType,
        const Complex = @This();

        fn mlirC64TypeGet(ctx: c.MlirContext) callconv(.C) c.MlirType {
            return c.mlirComplexTypeGet(c.mlirF32TypeGet(ctx));
        }

        fn mlirC128TypeGet(ctx: c.MlirContext) callconv(.C) c.MlirType {
            return c.mlirComplexTypeGet(c.mlirF64TypeGet(ctx));
        }

        fn mlirTypeIsAC64(typ: c.MlirType) callconv(.C) bool {
            const element_type: c.MlirType = c.mlirComplexTypeGetElementType(typ);
            return c.mlirTypeIsAF32(element_type);
        }

        fn mlirTypeIsAC128(typ: c.MlirType) callconv(.C) bool {
            const element_type: c.MlirType = c.mlirComplexTypeGetElementType(typ);
            return c.mlirTypeIsAF64(element_type);
        }

        const Config = switch (ct) {
            .c64 => .{ mlirTypeIsAC64, mlirC64TypeGet },
            .c128 => .{ mlirTypeIsAC128, mlirC128TypeGet },
            .unknown => .{ c.mlirTypeIsAComplex, null },
        };

        fn typeIsAUnknownComplex(typ: c.MlirType) callconv(.C) bool {
            return c.mlirTypeIsAComplex(typ);
        }

        pub usingnamespace MlirHelpers(@This(), .{
            .is_a_fn = Config[0],
            .is_null_fn = c.mlirTypeIsNull,
            .dump_fn = c.mlirTypeDump,
            .equal_fn = c.mlirTypeEqual,
        });

        pub usingnamespace if (ct != .unknown) struct {
            pub const ComplexTypeType = ct;

            pub fn init(ctx: Context) Complex {
                const type_get = Config[1];
                return Complex.wrap(type_get(ctx.inner()));
            }
        } else struct {};
    };
}

pub const TupleType = struct {
    _inner: c.MlirType,
    pub usingnamespace MlirHelpers(TupleType, .{
        .is_a_fn = c.mlirTypeIsATuple,
        .is_null_fn = c.mlirTypeIsNull,
        .dump_fn = c.mlirTypeDump,
        .equal_fn = c.mlirTypeEqual,
    });

    const Self = TupleType;

    pub fn init(ctx: Context, elements: []const Type) !Self {
        return Self.wrapOr(c.mlirTupleTypeGet(
            ctx.inner(),
            @intCast(elements.len),
            @ptrCast(elements.ptr),
        )) orelse Error.InvalidMlir;
    }

    pub fn getNumTypes(self: Self) usize {
        return @intCast(c.mlirTupleTypeGetNumTypes(self.inner()));
    }

    pub fn getElementType(self: Self, index: usize) Type {
        return Type.wrap(c.mlirTupleTypeGetType(self.inner(), @intCast(index)));
    }
};

pub const FunctionType = struct {
    _inner: c.MlirType,
    pub usingnamespace MlirHelpers(FunctionType, .{
        .is_a_fn = c.mlirTypeIsAFunction,
        .is_null_fn = c.mlirTypeIsNull,
        .dump_fn = c.mlirTypeDump,
        .equal_fn = c.mlirTypeEqual,
    });

    const Self = FunctionType;

    pub fn init(ctx: Context, args: []const Type, results: []const Type) !Self {
        return Self.wrapOr(c.mlirFunctionTypeGet(
            ctx.inner(),
            @intCast(args.len),
            @ptrCast(args.ptr),
            @intCast(results.len),
            @ptrCast(results.ptr),
        )) orelse Error.InvalidMlir;
    }
};

pub const RankedTensorType = struct {
    _inner: c.MlirType,
    pub usingnamespace MlirHelpers(RankedTensorType, .{
        .is_a_fn = c.mlirTypeIsARankedTensor,
        .is_null_fn = c.mlirTypeIsNull,
        .dump_fn = c.mlirTypeDump,
        .equal_fn = c.mlirTypeEqual,
        .print_fn = c.mlirTypePrint,
    });

    pub fn init(dimensions: []const i64, elemType: Type) RankedTensorType {
        return RankedTensorType.wrap(
            c.mlirRankedTensorTypeGet(
                @intCast(dimensions.len),
                @ptrCast(dimensions.ptr),
                elemType.inner(),
                c.mlirAttributeGetNull(),
            ),
        );
    }

    pub fn getElementType(self: RankedTensorType) Type {
        return Type.wrap(c.mlirShapedTypeGetElementType(self.inner()));
    }

    pub fn getRank(self: RankedTensorType) usize {
        return @intCast(c.mlirShapedTypeGetRank(self.inner()));
    }

    pub fn getDimension(self: RankedTensorType, dim: usize) i64 {
        return c.mlirShapedTypeGetDimSize(self.inner(), @intCast(dim));
    }

    pub fn asType(self: RankedTensorType) Type {
        return self.as(Type).?;
    }
};

pub const Dialect = struct {
    _inner: c.MlirDialect,
    pub usingnamespace MlirHelpers(Dialect, .{
        .equal_fn = c.mlirDialectEqual,
        .is_null_fn = c.mlirDialectIsNull,
    });

    const Self = Dialect;

    pub fn getContext(self: Self) Context {
        return Context.wrap(c.mlirDialectGetContext(self.inner()));
    }

    pub fn getNamespace(self: Self) []const u8 {
        return fromStringRef(c.mlirDialectGetNamespace(self.inner()));
    }
};

pub const DialectHandle = struct {
    _inner: c.MlirDialectHandle,
    pub usingnamespace MlirHelpers(
        DialectHandle,
        .{},
    );

    pub fn getNamespace(self: DialectHandle) []const u8 {
        return fromStringRef(c.mlirDialectHandleGetNamespace(self.inner()));
    }

    pub fn insertDialect(self: DialectHandle, registry: Registry) void {
        c.mlirDialectHandleInsertDialect(self.inner(), registry.inner());
    }

    pub fn registerDialect(self: DialectHandle, ctx: Context) void {
        c.mlirDialectHandleRegisterDialect(self.inner(), ctx.inner());
    }

    pub fn loadDialect(self: DialectHandle, ctx: Context) Dialect {
        return Dialect.wrap(c.mlirDialectHandleLoadDialect(self.inner(), ctx.inner()));
    }

    pub fn fromString(comptime namespace: []const u8) DialectHandle {
        return DialectHandle.wrap(@field(c, "mlirGetDialectHandle__" ++ namespace ++ "__")());
    }
};

pub const ShapedType = struct {
    _inner: c.MlirType,
    pub usingnamespace MlirHelpers(ShapedType, .{
        .is_a_fn = c.mlirTypeIsAShaped,
        .is_null_fn = c.mlirTypeIsNull,
        .dump_fn = c.mlirTypeDump,
        .equal_fn = c.mlirTypeEqual,
    });
    const Self = ShapedType;

    pub fn rank(self: Self) usize {
        return @intCast(c.mlirShapedTypeGetRank(self.inner()));
    }

    pub fn elementType(self: Self) Type {
        return Type.wrap(c.mlirShapedTypeGetElementType(self.inner()));
    }

    pub fn dimension(self: Self, dim: usize) usize {
        return @intCast(c.mlirShapedTypeGetDimSize(self.inner(), @intCast(dim)));
    }
};

pub const Location = struct {
    _inner: c.MlirLocation,

    pub usingnamespace MlirHelpers(Location, .{
        .is_null_fn = c.mlirLocationIsNull,
        .equal_fn = c.mlirLocationEqual,
        .print_fn = c.mlirLocationPrint,
    });

    pub fn fromSrc(ctx: Context, src: std.builtin.SourceLocation) Location {
        return Location.wrap(c.mlirLocationFileLineColGet(
            ctx.inner(),
            stringRef(src.file),
            @intCast(src.line),
            @intCast(src.column),
        ));
    }

    pub fn fileLineCol(ctx: Context, file: []const u8, line: usize, column: usize) Location {
        return Location.wrap(c.mlirLocationFileLineColGet(
            ctx.inner(),
            stringRef(file),
            @intCast(line),
            @intCast(column),
        ));
    }

    pub fn callSite(callee: Location, caller: Location) Location {
        return Location.wrap(c.mlirLocationCallSiteGet(callee.inner(), caller.inner()));
    }

    pub fn fused(ctx: Context, locations: []const Location, metadata: Attribute) Location {
        return Location.wrap(c.mlirLocationFusedGet(
            ctx.inner(),
            @intCast(locations.len),
            @ptrCast(locations.ptr),
            metadata.inner(),
        ));
    }

    pub fn named(loc: Location, ctx: Context, loc_name: [:0]const u8) Location {
        return Location.wrap(c.mlirLocationNameGet(ctx.inner(), stringRef(loc_name), loc.inner()));
    }

    pub fn namedFmt(loc: Location, ctx: Context, comptime fmt: [:0]const u8, args: anytype) Location {
        var buf: [256]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buf);
        std.fmt.format(stream.writer(), fmt, args) catch {
            buf[256 - 3 ..].* = "...".*;
        };
        return loc.named(ctx, @ptrCast(stream.getWritten()));
    }

    pub fn unknown(ctx: Context) Location {
        return Location.wrap(c.mlirLocationUnknownGet(ctx.inner()));
    }
};

pub const Block = struct {
    _inner: c.MlirBlock,
    pub usingnamespace MlirHelpers(Block, .{
        .is_null_fn = c.mlirBlockIsNull,
        .deinit_fn = c.mlirBlockDestroy,
        .equal_fn = c.mlirBlockEqual,
    });

    pub fn init(args: []const Type, locs: []const Location) !Block {
        const block = Block.wrapOr(
            c.mlirBlockCreate(@intCast(args.len), @ptrCast(args.ptr), @ptrCast(locs.ptr)),
        );
        return block orelse error.InvalidMlir;
    }

    pub fn argument(self: Block, index: usize) Value {
        return Value.wrap(c.mlirBlockGetArgument(self.inner(), @intCast(index)));
    }

    pub fn numArguments(self: Block) usize {
        return @intCast(c.mlirBlockGetNumArguments(self.inner()));
    }

    pub fn addArgument(self: *Block, typ: Type, loc: Location) Value {
        return Value.wrap(c.mlirBlockAddArgument(self.inner(), typ.inner(), loc.inner()));
    }

    pub fn insertArgument(self: *Block, index: usize, typ: Type, loc: Location) Value {
        return Value.wrap(c.mlirBlockInsertArgument(self.inner(), @intCast(index), typ.inner(), loc.inner()));
    }

    pub fn equals(self: Block, other: Block) bool {
        return c.mlirBlockEqual(self.inner(), other.inner());
    }

    pub fn appendOperation(self: Block, op: Operation) void {
        c.mlirBlockAppendOwnedOperation(self.inner(), op.inner());
    }

    pub fn appendOperations(self: *Block, ops: []const Operation) void {
        for (ops) |op| {
            c.mlirBlockAppendOwnedOperation(self.inner(), op.inner());
        }
    }
};
