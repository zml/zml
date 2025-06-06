const std = @import("std");
const builtin = @import("builtin");

const c = @import("c");
const stdx = @import("stdx");

const log = std.log.scoped(.mlir);

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

/// Alternative to MlirWrapperType
pub const MlirStrCallback = fn (c.MlirStringRef, ?*anyopaque) callconv(.C) void;

pub const Registry = struct {
    _inner: c.MlirDialectRegistry,

    pub const deinit = helpers.deinit(Registry, c.mlirDialectRegistryDestroy);

    pub fn init() !Registry {
        return helpers.init(Registry, c.mlirDialectRegistryCreate(), c.mlirDialectRegistryIsNull) orelse Error.MlirUnexpected;
    }
};

pub const Context = struct {
    _inner: c.MlirContext,
    const Self = Context;
    pub const deinit = helpers.deinit(Context, c.mlirContextDestroy);
    pub const wrapOr = helpers.wrapOr(Context, c.mlirContextIsNull);

    pub fn init() !Self {
        return Self.wrapOr(c.mlirContextCreate()) orelse Error.MlirUnexpected;
    }

    pub fn initWithRegistry(registry: Registry, threadingEnabled: bool) !Self {
        return Self.wrapOr(
            c.mlirContextCreateWithRegistry(registry._inner, threadingEnabled),
        ) orelse Error.InvalidMlir;
    }

    pub fn setMultiThreading(self: *Self, enabled: bool) void {
        c.mlirContextEnableMultithreading(self._inner, enabled);
    }

    pub fn appendDialectRegistry(self: *Self, registry: Registry) void {
        c.mlirContextAppendDialectRegistry(self._inner, registry._inner);
    }

    pub fn loadAllAvailableDialects(self: *Self) void {
        c.mlirContextLoadAllAvailableDialects(self._inner);
    }

    pub fn numRegisteredDialects(self: Self) usize {
        return @intCast(c.mlirContextGetNumRegisteredDialects(self._inner));
    }

    pub fn numLoadedDialects(self: Self) usize {
        return @intCast(c.mlirContextGetNumLoadedDialects(self._inner));
    }

    pub fn isRegisteredOperation(self: Self, op: [:0]const u8) bool {
        return c.mlirContextIsRegisteredOperation(self._inner, stringRef(op));
    }

    pub fn location(self: Self, src: std.builtin.SourceLocation) Location {
        return Location.fromSrc(self, src);
    }
};

pub const Module = struct {
    _inner: c.MlirModule,

    pub const deinit = helpers.deinit(Module, c.mlirModuleDestroy);
    pub const wrapOr = helpers.wrapOr(Module, c.mlirModuleIsNull);

    const Self = Module;

    pub fn init(loc: Location) Self {
        return .{ ._inner = c.mlirModuleCreateEmpty(loc._inner) };
    }

    pub fn parse(ctx: Context, source: [:0]const u8) !Module {
        return Module.wrapOr(
            c.mlirModuleCreateParse(ctx._inner, stringRef(source)),
        ) orelse Error.InvalidMlir;
    }

    pub fn fromOperation(operation: Operation) Module {
        return .{ ._inner = c.mlirModuleFromOperation(operation._inner) };
    }

    pub fn context(self: Module) Context {
        return .{ ._inner = c.mlirModuleGetContext(self._inner) };
    }

    pub fn getBody(self: Module) Block {
        return .{ ._inner = c.mlirModuleGetBody(self._inner) };
    }

    pub fn op(self: Module) Operation {
        return .{ ._inner = c.mlirModuleGetOperation(self._inner) };
    }

    pub fn hash(self: Module, hasher: *std.hash.XxHash64) void {
        return self.op().hash(hasher);
    }
};

pub const PassManager = struct {
    _inner: c.MlirPassManager,

    pub const deinit = helpers.deinit(PassManager, c.mlirPassManagerDestroy);
    pub const wrapOr = helpers.wrapOr(PassManager, c.mlirPassManagerIsNull);

    const Self = PassManager;

    pub fn init(ctx: Context) !Self {
        return Self.wrapOr(
            c.mlirPassManagerCreate(ctx._inner),
        ) orelse Error.MlirUnexpected;
    }

    pub fn initOnOperation(ctx: Context, op: [:0]const u8) !Self {
        return Self.wrapOr(
            c.mlirPassManagerCreateOnOperation(ctx._inner, stringRef(op)),
        ) orelse Error.MlirUnexpected;
    }

    pub fn asOpPassManager(self: Self) OpPassManager {
        return .{ ._inner = c.mlirPassManagerGetAsOpPassManager(self._inner) };
    }

    pub fn enableIRPrinting(self: *Self) void {
        c.mlirPassManagerEnableIRPrinting(self._inner);
    }

    pub fn runOnOp(self: *Self, op: Operation) error{InvalidMlir}!void {
        if (c.mlirPassManagerRunOnOp(self._inner, op._inner).value == 0) {
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

    pub fn addPipeline(self: *OpPassManager, pipeline: [:0]const u8) error{OutOfMemory}!void {
        if (c.mlirOpPassManagerAddPipeline(
            self._inner,
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
    const Self = Identifier;

    pub fn get(ctx: Context, str_: [:0]const u8) Self {
        return .{ ._inner = c.mlirIdentifierGet(ctx._inner, stringRef(str_)) };
    }

    pub fn context(self: Self) Context {
        return .{ ._inner = c.mlirIdentifierGetContext(self._inner) };
    }

    pub fn str(self: Self) []const u8 {
        return fromStringRef(c.mlirIdentifierStr(self._inner));
    }

    pub fn equals(self: Self, other: Self) bool {
        return c.mlirIdentifierEqual(self._inner, other._inner);
    }
};

pub const AttrTuple = struct { [:0]const u8, Attribute };

pub const Attribute = struct {
    _inner: c.MlirAttribute,

    pub const dump = helpers.dump(Attribute, c.mlirAttributeDump);
    pub const eql = helpers.eql(Attribute, c.mlirAttributeEqual);
    pub const format = helpers.format(Attribute, c.mlirAttributePrint);
    pub const wrapOr = helpers.wrapOr(Attribute, c.mlirAttributeIsNull);

    pub fn wrap(c_attr: c.MlirAttribute) Attribute {
        return .{ ._inner = c_attr };
    }

    pub fn parse(ctx: Context, attr: [:0]const u8) !Attribute {
        return Attribute.wrapOr(
            c.mlirAttributeParseGet(ctx._inner, stringRef(attr)),
        ) orelse Error.InvalidMlir;
    }

    pub fn fromAny(SpecificAttr: type) fn (x: SpecificAttr) Attribute {
        return struct {
            fn cast(x: SpecificAttr) Attribute {
                return .{ ._inner = x._inner };
            }
        }.cast;
    }

    pub fn isA(self: Attribute, SpecificAttr: type) bool {
        return SpecificAttr.is_a_fn(self._inner);
    }

    // utilities function to built common attributes.
    // All attributes are upcasted to the Attribute type, making it easier to chain construct,
    // but losing type information.

    pub fn null_() Attribute {
        return .wrap(c.mlirAttributeGetNull());
    }

    pub fn string(ctx: Context, str: []const u8) Attribute {
        return StringAttribute.init(ctx, str).asAttr();
    }

    pub fn type_(t: Type) Attribute {
        return TypeAttribute.init(t).asAttr();
    }

    pub fn unit(ctx: Context) Attribute {
        return .wrap(c.mlirUnitAttrGet(ctx._inner));
    }

    pub fn boolean(ctx: Context, value: bool) Attribute {
        return BoolAttribute.init(ctx, value).asAttr();
    }

    pub fn i1FromBool(ctx: Context, value: bool) Attribute {
        return IntegerAttribute(.i1).init(ctx, @intFromBool(value)).asAttr();
    }

    pub fn int(ctx: Context, comptime int_type: IntegerTypes, value: i64) Attribute {
        return IntegerAttribute(int_type).init(ctx, value).asAttr();
    }

    pub fn float(ctx: Context, comptime float_type: FloatTypes, value: f64) Attribute {
        return FloatAttribute(float_type).init(ctx, value).asAttr();
    }

    pub fn array(ctx: Context, attrs: []const Attribute) Attribute {
        return ArrayAttribute.init(ctx, attrs).asAttr();
    }

    pub fn dense(ctx: Context, comptime dt: DenseArrayTypes, values: []const dt.ZigType()) Attribute {
        return DenseArrayAttribute(dt).init(ctx, values).asAttr();
    }

    /// Use a tensor as an attribute.
    /// The tensor is specified by dims, dtype and a flat slice of values.
    pub fn denseElements(ctx: Context, dims: []const i64, comptime dt: DenseElementsAttributeTypes, values: []const dt.ZigType()) Attribute {
        return DenseElementsAttribute(dt).init(.tensor(dims, dt.mlirType(ctx)), values).asAttr();
    }

    pub fn denseElementsFromBytes(ctx: Context, dims: []const i64, dt: DenseElementsAttributeTypes, raw_bytes: []const u8) Attribute {
        const shape: Type = .tensor(dims, dt.mlirType(ctx));
        return .{ ._inner = c.mlirDenseElementsAttrRawBufferGet(
            shape._inner,
            @intCast(raw_bytes.len),
            raw_bytes.ptr,
        ) };
    }

    pub fn symbol(ctx: Context, flat_name: [:0]const u8) Attribute {
        return FlatSymbolRefAttribute.init(ctx, flat_name).asAttr();
    }

    pub fn named(attr: Attribute, ctx: Context, name: [:0]const u8) NamedAttribute {
        return NamedAttribute.named(ctx, name, attr);
    }

    pub fn dict(ctx: Context, named_attrs: []const AttrTuple) Attribute {
        var attr_buf: [32]NamedAttribute = undefined;
        stdx.debug.assert(named_attrs.len <= attr_buf.len, ".dict helper only support up to {} attribute, got {}. You will need to call mlir.DictionaryAttribute directly", .{ attr_buf.len, named_attrs.len });

        const attrs = attr_buf[0..named_attrs.len];
        for (attrs, named_attrs) |*attr, tuple| {
            attr.* = .named(ctx, tuple[0], tuple[1]);
        }

        return DictionaryAttribute.init(ctx, attrs).asAttr();
    }
};

pub const NamedAttribute = extern struct {
    _inner: c.MlirNamedAttribute,

    pub fn wrap(c_named_attribute: c.MlirNamedAttribute) NamedAttribute {
        return @bitCast(c_named_attribute);
    }

    pub fn named(ctx: Context, name: [:0]const u8, attr: Attribute) NamedAttribute {
        return .{ ._inner = .{
            .name = c.mlirIdentifierGet(ctx._inner, stringRef(name)),
            .attribute = attr._inner,
        } };
    }

    pub fn init(name: Identifier, attr: Attribute) NamedAttribute {
        return .{ ._inner = .{
            .name = name._inner,
            .attribute = attr._inner,
        } };
    }
};

pub const StringAttribute = struct {
    _inner: c.MlirAttribute,
    pub const is_a_fn = c.mlirAttributeIsAString;
    const Self = StringAttribute;
    pub const asAttr = Attribute.fromAny(Self);
    pub const eql = Attribute.eqlAny(Self);

    pub fn init(ctx: Context, str: []const u8) Self {
        return .{ ._inner = c.mlirStringAttrGet(ctx._inner, stringRef(str)) };
    }

    pub fn value(self: Self) []const u8 {
        return fromStringRef(c.mlirStringAttrGetValue(self._inner));
    }
};

pub const BoolAttribute = struct {
    _inner: c.MlirAttribute,
    pub const is_a_fn = c.mlirAttributeIsABool;
    const Self = BoolAttribute;
    pub const asAttr = Attribute.fromAny(Self);
    pub const eql = Attribute.eqlAny(Self);

    pub fn init(ctx: Context, value_: bool) Self {
        return .{ ._inner = c.mlirBoolAttrGet(ctx._inner, if (value_) 1 else 0) };
    }

    pub fn value(self: Self) bool {
        return c.mlirBoolAttrGetValue(self._inner);
    }
};

pub const TypeAttribute = struct {
    _inner: c.MlirAttribute,
    pub const is_a_fn = c.mlirAttributeIsAType;
    pub const eql = Attribute.eqlAny(TypeAttribute);

    pub fn init(type_: Type) TypeAttribute {
        return .{ ._inner = c.mlirTypeAttrGet(type_._inner) };
    }

    pub fn typ(self: TypeAttribute) Type {
        return .{ ._inner = c.mlirAttributeGetType(self._inner) };
    }

    pub const asAttr = Attribute.fromAny(TypeAttribute);
};

pub const ArrayAttribute = struct {
    _inner: c.MlirAttribute,
    pub const is_a_fn = c.mlirAttributeIsAArray;
    const Self = ArrayAttribute;
    pub const asAttr = Attribute.fromAny(Self);
    pub const eql = Attribute.eqlAny(Self);

    pub fn init(ctx: Context, attrs: []const Attribute) Self {
        return .{ ._inner = c.mlirArrayAttrGet(ctx._inner, @intCast(attrs.len), @ptrCast(attrs.ptr)) };
    }

    pub fn size(self: Self) usize {
        return @intCast(c.mlirArrayAttrGetNumElements(self._inner));
    }

    pub fn get(self: Self, index: usize) Attribute {
        return .{ ._inner = c.mlirArrayAttrGetElement(self._inner, @intCast(index)) };
    }
};

pub fn IntegerAttribute(comptime it: IntegerTypes) type {
    const ZigType, const getter = comptime switch (it) {
        .i1, .i4, .i8, .i16, .i32, .i64 => .{ i64, c.mlirIntegerAttrGetValueInt },
        .si4, .si8, .si16, .si32, .si64 => .{ i64, c.mlirIntegerAttrGetValueSInt },
        .u4, .u8, .u16, .u32, .u64 => .{ u64, c.mlirIntegerAttrGetValueUInt },
        .unknown => @compileError("IntegerAttribute(unknown)"),
    };

    return struct {
        _inner: c.MlirAttribute,
        pub const is_a_fn = c.mlirAttributeIsAInteger;

        pub const IntegerTypeType = IntegerType(it);
        const IntAttr = @This();

        pub const asAttr = Attribute.fromAny(IntAttr);
        pub const eql = Attribute.eqlAny(IntAttr);

        pub fn init(ctx: Context, value: i64) IntAttr {
            return .{ ._inner = c.mlirIntegerAttrGet(
                IntegerType(it).init(ctx)._inner,
                value,
            ) };
        }

        pub fn get(value: IntAttr) ZigType {
            return @intCast(getter(value._inner));
        }
    };
}

pub fn FloatAttribute(comptime ft: FloatTypes) type {
    return struct {
        _inner: c.MlirAttribute,
        pub const is_a_fn = c.mlirAttributeIsAFloat;
        const FloatAttr = @This();
        pub const asAttr = Attribute.fromAny(FloatAttr);

        pub fn init(ctx: Context, value: f64) FloatAttr {
            return .{ ._inner = c.mlirFloatAttrDoubleGet(
                ctx._inner,
                FloatType(ft).init(ctx)._inner,
                value,
            ) };
        }

        pub fn get(value: FloatAttr) f64 {
            return c.mlirFloatAttrGetValueDouble(value._inner);
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

    pub fn ZigType(comptime dt: DenseArrayTypes) type {
        return switch (dt) {
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

pub fn DenseArrayAttribute(comptime dt: DenseArrayTypes) type {
    const _is_a_fn, const get_fn, const get_element_fn = switch (dt) {
        .bool => .{ c.mlirAttributeIsADenseBoolArray, c.mlirDenseBoolArrayGet, c.mlirDenseBoolArrayGetElement },
        .i8 => .{ c.mlirAttributeIsADenseI8Array, c.mlirDenseI8ArrayGet, c.mlirDenseI8ArrayGetElement },
        .i16 => .{ c.mlirAttributeIsADenseI16Array, c.mlirDenseI16ArrayGet, c.mlirDenseI16ArrayGetElement },
        .i32 => .{ c.mlirAttributeIsADenseI32Array, c.mlirDenseI32ArrayGet, c.mlirDenseI32ArrayGetElement },
        .i64 => .{ c.mlirAttributeIsADenseI64Array, c.mlirDenseI64ArrayGet, c.mlirDenseI64ArrayGetElement },
        .f32 => .{ c.mlirAttributeIsADenseF32Array, c.mlirDenseF32ArrayGet, c.mlirDenseF32ArrayGetElement },
        .f64 => .{ c.mlirAttributeIsADenseF64Array, c.mlirDenseF64ArrayGet, c.mlirDenseF64ArrayGetElement },
    };

    return struct {
        _inner: c.MlirAttribute,
        const Attr = @This();
        const ElementType = dt;
        const ElementTypeZig = dt.ZigType();

        pub const asAttr = Attribute.fromAny(Attr);
        pub const eql = Attribute.eqlAny(Attr);
        pub const is_a_fn = _is_a_fn;

        pub fn init(ctx: Context, values: []const ElementTypeZig) Attr {
            return .{ ._inner = get_fn(ctx._inner, @intCast(values.len), @ptrCast(values.ptr)) };
        }

        pub fn get(self: Attr, pos: usize) ElementTypeZig {
            return get_element_fn(self._inner, @intCast(pos));
        }

        pub fn len(self: Attr) usize {
            return @intCast(c.mlirDenseArrayGetNumElements(self._inner));
        }
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
    index,

    pub fn ZigType(comptime dt: DenseElementsAttributeTypes) type {
        return switch (dt) {
            .bool => bool,
            .i8 => i8,
            .i16 => i16,
            .i32 => i32,
            .i64 => i64,
            .u8 => u8,
            .u16 => u16,
            .u32 => u32,
            .u64 => u64,
            .bf16 => u16,
            .f16 => f16,
            .f32 => f32,
            .f64 => f64,
            .index => usize,
        };
    }

    pub fn mlirType(dt: DenseElementsAttributeTypes, ctx: Context) Type {
        return switch (dt) {
            .bool => .int(ctx, .i1),
            .i8 => .int(ctx, .i8),
            .i16 => .int(ctx, .i16),
            .i32 => .int(ctx, .i32),
            .i64 => .int(ctx, .i64),
            .u8 => .int(ctx, .u8),
            .u16 => .int(ctx, .u16),
            .u32 => .int(ctx, .u32),
            .u64 => .int(ctx, .u64),
            .bf16 => .float(ctx, .bf16),
            .f16 => .float(ctx, .f16),
            .f32 => .float(ctx, .f32),
            .f64 => .float(ctx, .f64),
            .index => .index(ctx),
        };
    }
};

pub fn DenseElementsAttribute(comptime dt: DenseElementsAttributeTypes) type {
    return struct {
        _inner: c.MlirAttribute,

        const Attr = @This();

        pub const is_a_fn = c.mlirAttributeIsADenseElements;
        pub const asAttr = Attribute.fromAny(Attr);
        pub const eql = Attribute.eqlAny(Attr);

        pub fn init(shaped_type: Type, slice: []const dt.ZigType()) Attr {
            const raw_bytes = std.mem.sliceAsBytes(slice);
            const res: Attr = .{ ._inner = c.mlirDenseElementsAttrRawBufferGet(
                shaped_type._inner,
                @intCast(raw_bytes.len),
                @ptrCast(raw_bytes.ptr),
            ) };
            std.debug.assert(Attribute.wrapOr(res._inner) != null);
            return res;
        }

        pub fn len(self: Attr) usize {
            return @intCast(c.mlirElementsAttrGetNumElements(self._inner));
        }

        pub fn items(self: Attr) []const dt.ZigType() {
            const raw_bytes: [*]const u8 = c.mlirDenseElementsAttrGetRawData(self._inner) orelse unreachable;
            const ptr: [*]const dt.ZigType() = @alignCast(@ptrCast(raw_bytes));
            // Note the mlir API returns us the number of elements, not the number of bytes,
            // that's why we track the element type at comptime to allow items to work.
            return ptr[0..self.len()];
        }

        pub fn bytes(self: Attr) []const u8 {
            return std.mem.sliceAsBytes(self.items());
        }
    };
}

pub const FlatSymbolRefAttribute = struct {
    _inner: c.MlirAttribute,
    pub const is_a_fn = c.mlirAttributeIsAFlatSymbolRef;
    const Self = FlatSymbolRefAttribute;
    pub const eql = Attribute.eqlAny(Self);

    pub fn init(ctx: Context, str: [:0]const u8) Self {
        return .{ ._inner = c.mlirFlatSymbolRefAttrGet(ctx._inner, stringRef(str)) };
    }

    pub fn value(self: Self) []const u8 {
        return fromStringRef(c.mlirFlatSymbolRefAttrGetValue(self._inner));
    }

    pub const asAttr = Attribute.fromAny(Self);
};

pub const OperationState = struct {
    _inner: c.MlirOperationState,

    const Self = OperationState;

    pub fn init(name: [:0]const u8, loc: Location) Self {
        return .{ ._inner = c.mlirOperationStateGet(stringRef(name), loc._inner) };
    }

    pub fn addResult(self: *Self, type_: Type) void {
        c.mlirOperationStateAddResults(&self._inner, 1, &[_]c.MlirType{type_._inner});
    }

    pub fn addResults(self: *Self, types: []const Type) void {
        c.mlirOperationStateAddResults(&self._inner, @intCast(types.len), @ptrCast(types.ptr));
    }

    pub fn addOperand(self: *Self, value: Value) void {
        c.mlirOperationStateAddOperands(&self._inner, 1, &[_]c.MlirValue{value._inner});
    }

    pub fn addOperands(self: *Self, values: []const Value) void {
        c.mlirOperationStateAddOperands(&self._inner, @intCast(values.len), @ptrCast(values.ptr));
    }

    pub fn addRegion(self: *Self, region: *Region) void {
        c.mlirOperationStateAddOwnedRegions(&self._inner, 1, &[_]c.MlirRegion{region._inner});
    }

    pub fn addRegions(self: *Self, regions: []const Region) void {
        c.mlirOperationStateAddOwnedRegions(&self._inner, @intCast(regions.len), @ptrCast(regions.ptr));
    }

    pub fn addAttribute(self: *Self, ctx: Context, name: [:0]const u8, attr: Attribute) void {
        c.mlirOperationStateAddAttributes(&self._inner, 1, @ptrCast(&.{
            .{
                .name = Identifier.get(ctx, name)._inner,
                .attribute = attr._inner,
            },
        }));
    }

    pub fn addAttributeRaw(self: *Self, name: Identifier, attr: Attribute) void {
        c.mlirOperationStateAddAttributes(&self._inner, 1, @ptrCast(&.{
            .{
                .name = name._inner,
                .attribute = attr._inner,
            },
        }));
    }

    pub fn addAttributes(self: *Self, attributes: []const NamedAttribute) void {
        c.mlirOperationStateAddAttributes(&self._inner, @intCast(attributes.len), @ptrCast(attributes.ptr));
    }

    pub fn resultTypeInference(self: *Self, enabled: bool) void {
        self._inner.enableResultTypeInference = enabled;
    }
};

pub const DictionaryAttribute = struct {
    _inner: c.MlirAttribute,
    pub const is_a_fn = c.mlirAttributeIsADictionary;
    pub const asAttr = Attribute.fromAny(DictionaryAttribute);
    pub const eql = Attribute.eqlAny(DictionaryAttribute);

    pub fn init(ctx: Context, attributes: []const NamedAttribute) DictionaryAttribute {
        return .{ ._inner = c.mlirDictionaryAttrGet(
            ctx._inner,
            @intCast(attributes.len),
            @ptrCast(attributes.ptr),
        ) };
    }

    pub fn size(self: DictionaryAttribute) usize {
        return @intCast(c.mlirDictionaryAttrGetNumElements(self._inner));
    }

    pub fn get(self: DictionaryAttribute, pos: usize) NamedAttribute {
        return .wrap(c.mlirDictionaryAttrGetElement(self._inner, @bitCast(pos)));
    }

    pub fn getByName(self: DictionaryAttribute, name: [:0]const u8) ?Attribute {
        return Attribute.wrapOr(c.mlirDictionaryAttrGetElementByName(self._inner, name));
    }
};

pub const Operation = struct {
    const Self = Operation;
    _inner: c.MlirOperation,

    pub const dump = helpers.dump(Operation, c.mlirOperationDestroy);
    pub const deinit = helpers.deinit(Operation, c.mlirOperationDestroy);
    pub const wrapOr = helpers.wrapOr(Operation, c.mlirOperationIsNull);

    pub const eql = Attribute.eqlAny(Self);

    pub fn init(state: *OperationState) !Self {
        return Self.wrapOr(c.mlirOperationCreate(&state._inner)) orelse Error.InvalidMlir;
    }

    pub fn make(ctx: Context, op_name: [:0]const u8, args: struct {
        operands: ?[]const Value = null,
        variadic_operands: ?[]const []const Value = null,
        tt_variadic_operands: ?[]const []const Value = null,
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
            const MAX_SEGMENTS = 32;
            var segments: std.BoundedArray(i32, MAX_SEGMENTS) = .{};

            for (operands_segments) |operands| {
                state.addOperands(operands);
                segments.appendAssumeCapacity(@intCast(operands.len));
            }
            state.addAttribute(ctx, "operandSegmentSizes", .denseElements(ctx, &.{@intCast(segments.len)}, .i32, segments.constSlice()));
        } else if (args.tt_variadic_operands) |operands_segments| {
            // stablehlo and triton seems to disagree on the expected type of operandSegmentSizes, let's fix that.
            const MAX_SEGMENTS = 32;
            var segments: std.BoundedArray(i32, MAX_SEGMENTS) = .{};

            for (operands_segments) |operands| {
                state.addOperands(operands);
                segments.appendAssumeCapacity(@intCast(operands.len));
            }
            state.addAttribute(ctx, "operandSegmentSizes", .dense(ctx, .i32, segments.constSlice()));
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
            c.mlirOperationCreateParse(ctx._inner, stringRef(str), stringRef("pouet")),
        ) orelse Error.InvalidMlir;
    }

    pub fn clone(self: Self) !Self {
        return Self.wrapOr(
            c.mlirOperationClone(self._inner),
        ) orelse Error.InvalidMlir;
    }

    pub fn name(self: Self) Identifier {
        return .{ ._inner = c.mlirOperationGetName(self._inner) };
    }

    pub fn removeFromParent(self: *Self) void {
        c.mlirOperationRemoveFromParent(self._inner);
    }

    pub fn numOperands(self: Self) usize {
        return @intCast(c.mlirOperationGetNumOperands(self._inner));
    }

    pub fn operand(self: Self, index: usize) Value {
        return .{ ._inner = c.mlirOperationGetOperand(self._inner, @intCast(index)) };
    }

    pub fn setOperand(self: *Self, index: usize, value: Value) void {
        c.mlirOperationSetOperand(self._inner, @intCast(index), value._inner);
    }

    pub fn numResults(self: Self) usize {
        return @intCast(c.mlirOperationGetNumResults(self._inner));
    }

    pub fn result(self: Self, index: usize) Value {
        return .{ ._inner = c.mlirOperationGetResult(self._inner, @intCast(index)) };
    }

    pub fn nextInBlock(self: Self) Self {
        return .{ ._inner = c.mlirOperationGetNextInBlock(self._inner) };
    }

    // pub fn previousInBlock(self: Self) Self {
    //     return .{ ._inner = c.mlirOperationGetPrevInBlock(self._inner) };
    // }

    pub fn block(self: Self) ?Block {
        return Block.wrapOr(c.mlirOperationGetBlock(self._inner));
    }

    pub fn parent(self: Self) ?Self {
        return Self.wrapOr(c.mlirOperationGetParentOperation(self._inner));
    }

    pub fn region(self: Self, index: usize) Region {
        return .{ ._inner = c.mlirOperationGetRegion(self._inner, @intCast(index)) };
    }

    pub fn context(self: Self) Context {
        return .{ ._inner = c.mlirOperationGetContext(self._inner) };
    }

    pub fn writeBytecode(self: Self, writer: anytype) void {
        var writer_context = .{ .writer = writer };
        const WriterContext = @TypeOf(writer_context);

        c.mlirOperationWriteBytecode(
            self._inner,
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
            self._inner,
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
            self._inner,
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
        return c.mlirOperationVerify(self._inner);
    }

    pub fn getLocation(self: Self) Location {
        return .{ ._inner = c.mlirOperationGetLocation(self._inner) };
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
            self._inner,
            (struct {
                pub fn callback(op: c.MlirOperation, ctx_: ?*anyopaque) callconv(.C) c.MlirWalkResult {
                    const inner_ctx_: *ContextType = @ptrCast(@alignCast(ctx_));
                    return @intFromEnum(walkfn(inner_ctx_.ctx, .{ ._inner = op }));
                }
            }).callback,
            &inner_ctx,
            @intFromEnum(order),
        );
    }

    pub fn getAttribute(self: Self, pos: usize) NamedAttribute {
        return .{ ._inner = c.mlirOperationGetAttribute(self._inner, @intCast(pos)) };
    }

    pub fn getAttributeByName(self: Self, name_: [:0]const u8) ?Attribute {
        return Attribute.wrapOr(c.mlirOperationGetAttributeByName(self._inner, stringRef(name_)));
    }

    pub fn setAttributeByName(self: Self, name_: [:0]const u8, attr: Attribute) void {
        c.mlirOperationSetAttributeByName(self._inner, stringRef(name_), attr._inner);
    }

    pub fn removeAttributeByName(self: Self, name_: [:0]const u8) bool {
        return c.mlirOperationRemoveAttributeByName(self._inner, stringRef(name_));
    }

    pub fn hash(op: Operation, hasher: *std.hash.XxHash64) void {
        const NoError = error{};
        const write = struct {
            fn write(hasher_: *std.hash.XxHash64, bytes: []const u8) NoError!usize {
                hasher_.update(bytes);
                return bytes.len;
            }
        }.write;
        const HashWriter = std.io.Writer(*std.hash.XxHash64, NoError, write);
        const writer: HashWriter = .{ .context = hasher };

        // Hash the canonicalized IR, without debug information that can change across builds.
        // Note: before we where using op.writeBytecode(writer),
        // but it crashes on some inputs, notably for unused variables.
        // So we use the text representation of the mlir.
        // See https://github.com/zml/zml/issues/97.
        // Writes can't fail because we are writing to a hasher.
        op.print(writer, .{ .debug_info = false });
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
    const Self = OpOperand;

    pub fn owner(self: Self) Operation {
        return .{ ._inner = c.mlirOpOperandGetOwner(self._inner) };
    }

    pub fn number(self: Self) usize {
        return @intCast(c.mlirOpOperandGetOperandNumber(self._inner));
    }

    pub fn nextUse(self: Self) ?Self {
        return Self.wrapOr(
            c.mlirOpOperandGetNextUse(self._inner),
        );
    }
};

pub const Region = struct {
    _inner: c.MlirRegion,

    pub const eql = helpers.eql(Region, c.mlirBlockEqual);
    pub const deinit = helpers.deinit(Region, c.mlirRegionDestroy);
    pub const wrapOr = helpers.wrapOr(Region, c.mlirRegionIsNull);

    const Self = Region;

    pub fn init() !Self {
        return Self.wrapOr(c.mlirRegionCreate()) orelse Error.InvalidMlir;
    }

    pub fn appendBlock(self: *Self, block: Block) void {
        c.mlirRegionAppendOwnedBlock(self._inner, block._inner);
    }

    pub fn insertBlock(self: *Self, index: isize, block: Block) void {
        c.mlirRegionInsertOwnedBlock(self._inner, index, block._inner);
    }

    pub fn insertBlockBefore(self: *Self, reference: Block, block: Block) void {
        c.mlirRegionInsertOwnedBlockBefore(self._inner, reference._inner, block._inner);
    }

    pub fn insertBlockAfter(self: *Self, reference: Block, block: Block) void {
        c.mlirRegionInsertOwnedBlockAfter(self._inner, reference._inner, block._inner);
    }

    pub fn firstBlock(self: Self) Block {
        return .{ ._inner = c.mlirRegionGetFirstBlock(self._inner) };
    }
};

pub const Value = struct {
    _inner: c.MlirValue,

    pub const dump = helpers.dump(Value, c.mlirValueDump);
    pub const eql = helpers.eql(Value, c.mlirValueEqual);
    pub const format = helpers.format(Value, c.mlirValuePrint).format;
    pub const wrapOr = helpers.wrapOr(Value, c.mlirValueIsNull);

    pub fn getType(val: Value) Type {
        return .{ ._inner = c.mlirValueGetType(val._inner) };
    }

    pub fn setType(val: *Value, typ: Type) void {
        c.mlirValueSetType(val._inner, typ._inner);
    }

    pub fn firstUse(val: Value) OpOperand {
        return .{ ._inner = c.mlirValueGetFirstUse(val._inner) };
    }

    pub fn replaceAllUsesWith(val: Value, with: Value) void {
        c.mlirValueReplaceAllUsesOfWith(val._inner, with._inner);
    }

    pub fn owner(val: Value) Operation {
        return .{ ._inner = c.mlirOpResultGetOwner(val._inner) };
    }

    pub fn isABlockArgument(val: Value) bool {
        return c.mlirValueIsABlockArgument(val._inner);
    }

    pub fn isAOpResult(val: Value) bool {
        return c.mlirValueIsAOpResult(val._inner);
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
        return .{ ._inner = c.mlirBlockArgumentGetOwner(arg._inner) };
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

    pub const dump = helpers.eql(Type, c.mlirTypeDump);
    pub const eql = helpers.eql(Type, c.mlirTypeEqual);
    pub const format = helpers.format(Type, c.mlirTypePrint);
    pub const wrapOr = helpers.wrapOr(Type, c.mlirTypeIsNull);

    pub fn parse(ctx: Context, str: [:0]const u8) !Type {
        return Type.wrapOr(
            c.mlirTypeParseGet(ctx._inner, stringRef(str)),
        ) orelse Error.InvalidMlir;
    }

    pub fn as(generic: Type, SpecificType: type) ?SpecificType {
        if (@hasDecl(SpecificType, "is_a_fn")) {
            return if (SpecificType.is_a_fn(generic._inner))
                .{ ._inner = generic._inner }
            else
                null;
        }
        @compileError("Mlir subclass of type need `is_a_fn` attribute: " ++ @typeName(SpecificType));
    }

    pub fn fromAny(SpecificType: type) fn (x: SpecificType) Type {
        stdx.debug.assertComptime(@hasDecl(SpecificType, "asType"), "Type.fromAny expects a type subclass, got: {}. Missing `asAttr` declaration.", .{SpecificType});
        return struct {
            fn cast(x: SpecificType) Type {
                return .{ ._inner = x._inner };
            }
        }.cast;
    }

    pub fn eqlAny(SpecificType: type) fn (SpecificType, SpecificType) bool {
        return struct {
            fn eql(a: SpecificType, b: SpecificType) bool {
                return a.asType().eql(b.asType());
            }
        }.eql;
    }

    pub fn formatAny(SpecificType: type) fn (SpecificType, SpecificType) type {
        return struct {
            pub fn format(self: SpecificType, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
                return try Type.format(self.asType(), fmt, options, writer);
            }
        };
    }

    pub fn index(ctx: Context) Type {
        return IndexType.init(ctx).asType();
    }

    pub fn int(ctx: Context, int_type: IntegerTypes) Type {
        return switch (int_type) {
            .unknown => @panic("Unknown integer type"),
            inline else => |t| IntegerType(t).init(ctx).asType(),
        };
    }

    pub fn float(ctx: Context, float_type: FloatTypes) Type {
        return switch (float_type) {
            inline else => |t| FloatType(t).init(ctx).asType(),
        };
    }

    pub fn complex(ctx: Context, complex_type: ComplexTypes) Type {
        return switch (complex_type) {
            .unknown => @panic("Unknown complex type can't be created like this"), // What's the point ?
            inline else => |t| ComplexType(t).init(ctx).asType(),
        };
    }

    pub fn tuple(ctx: Context, types: []const Type) Type {
        return (TupleType.init(ctx, types) catch unreachable).asType();
    }

    pub fn function(ctx: Context, args: []const Type, results: []const Type) Type {
        return (FunctionType.init(ctx, args, results) catch unreachable).asType();
    }

    pub fn tensor(dimensions: []const i64, elem_type: Type) Type {
        return RankedTensorType.init(dimensions, elem_type).asType();
    }
};

pub const IndexType = struct {
    _inner: c.MlirType,

    pub const asType = Type.fromAny(IndexType);
    pub const eql = Type.eqlAny(IndexType);
    pub const format = Type.formatAny(IndexType).format;

    pub fn init(ctx: Context) IndexType {
        return .{ ._inner = c.mlirIndexTypeGet(ctx._inner) };
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
        pub const is_a_fn = switch (it) {
            .unknown => c.mlirTypeIsAInteger,
            else => typeIsAIntegerExact,
        };

        pub const asType = Type.fromAny(Int);
        pub const eql = Type.eqlAny(Int);
        pub const format = helpers.format(Int, c.mlirTypePrint);

        fn typeIsAIntegerExact(typ: c.MlirType) callconv(.C) bool {
            const bit_width = Config[0];
            const is_sign = Config[2];
            return c.mlirTypeIsAInteger(typ) and (c.mlirIntegerTypeGetWidth(typ) == bit_width) and is_sign(typ);
        }

        pub const BitWidth = Config[0];

        pub const init = if (it != .unknown) struct {
            pub fn init(ctx: Context) Int {
                const type_get = Config[1];
                return .{ ._inner = type_get(ctx._inner, BitWidth) };
            }
        }.init else {};
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

    pub fn asType(self: FloatTypes, ctx: Context) Type {
        return switch (self) {
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
    };

    return struct {
        _inner: c.MlirType,

        const Self = @This();

        pub const is_a_fn = Config[0];

        pub const asType = Type.fromAny(Self);
        pub const eql = Type.eqlAny(Self);
        pub const format = helpers.format(Self, c.mlirTypePrint);

        pub fn init(ctx: Context) Self {
            const type_get = Config[1];
            return .{ ._inner = type_get(ctx._inner) };
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

        pub const is_a_fn = Config[0];

        pub const asType = Type.fromAny(Complex);
        pub const eql = Type.eqlAny(Complex);
        pub const format = Type.formatAny(Complex).format;
        pub const ComplexTypeType: ComplexTypes = ct;

        pub const init = if (ct != .unknown) struct {
            pub fn init(ctx: Context) Complex {
                const type_get = Config[1];
                return .{ ._inner = type_get(ctx._inner) };
            }
        }.init else {};
    };
}

pub const TupleType = struct {
    _inner: c.MlirType,
    pub const is_a_fn = c.mlirTypeIsATuple;

    const Self = TupleType;

    pub fn init(ctx: Context, elements: []const Type) !Self {
        return Self.wrapOr(c.mlirTupleTypeGet(
            ctx._inner,
            @intCast(elements.len),
            @ptrCast(elements.ptr),
        )) orelse Error.InvalidMlir;
    }

    pub fn getNumTypes(self: Self) usize {
        return @intCast(c.mlirTupleTypeGetNumTypes(self._inner));
    }

    pub fn getElementType(self: Self, index: usize) Type {
        return .{ ._inner = c.mlirTupleTypeGetType(self._inner, @intCast(index)) };
    }

    pub const asType = Type.fromAny(Self);
};

pub const FunctionType = struct {
    _inner: c.MlirType,
    pub const is_a_fn = c.mlirTypeIsAFunction;
    const Self = FunctionType;
    pub const asType = Type.fromAny(Self);
    pub const eql = Type.eqlAny(IndexType);

    pub fn init(ctx: Context, args: []const Type, results: []const Type) !Self {
        const func = Type.wrapOr(c.mlirFunctionTypeGet(
            ctx._inner,
            @intCast(args.len),
            @ptrCast(args.ptr),
            @intCast(results.len),
            @ptrCast(results.ptr),
        )) orelse return Error.InvalidMlir;
        return func.as(Self).?;
    }
};

pub const RankedTensorType = struct {
    _inner: c.MlirType,
    pub const is_a_fn = c.mlirTypeIsARankedTensor;
    pub const eql = Type.eqlAny(RankedTensorType);
    pub const format = helpers.format(Type, c.mlirTypePrint);

    pub fn init(dimensions: []const i64, elemType: Type) RankedTensorType {
        return .{ ._inner = c.mlirRankedTensorTypeGet(
            @intCast(dimensions.len),
            @ptrCast(dimensions.ptr),
            elemType._inner,
            c.mlirAttributeGetNull(),
        ) };
    }

    pub fn getElementType(self: RankedTensorType) Type {
        return .{ ._inner = c.mlirShapedTypeGetElementType(self._inner) };
    }

    pub fn getRank(self: RankedTensorType) usize {
        return @intCast(c.mlirShapedTypeGetRank(self._inner));
    }

    pub fn getDimension(self: RankedTensorType, dim: usize) i64 {
        return c.mlirShapedTypeGetDimSize(self._inner, @intCast(dim));
    }

    pub const asType = Type.fromAny(RankedTensorType);
};

pub const Dialect = struct {
    _inner: c.MlirDialect,

    const Self = Dialect;

    pub fn getContext(self: Self) Context {
        return .{ ._inner = c.mlirDialectGetContext(self._inner) };
    }

    pub fn getNamespace(self: Self) []const u8 {
        return fromStringRef(c.mlirDialectGetNamespace(self._inner));
    }
};

pub const DialectHandle = struct {
    _inner: c.MlirDialectHandle,

    pub fn getNamespace(self: DialectHandle) []const u8 {
        return fromStringRef(c.mlirDialectHandleGetNamespace(self._inner));
    }

    pub fn insertDialect(self: DialectHandle, registry: Registry) void {
        c.mlirDialectHandleInsertDialect(self._inner, registry._inner);
    }

    pub fn registerDialect(self: DialectHandle, ctx: Context) void {
        c.mlirDialectHandleRegisterDialect(self._inner, ctx._inner);
    }

    pub fn loadDialect(self: DialectHandle, ctx: Context) Dialect {
        return .{ ._inner = c.mlirDialectHandleLoadDialect(self._inner, ctx._inner) };
    }

    pub fn fromString(comptime namespace: []const u8) DialectHandle {
        return .{ ._inner = @field(c, "mlirGetDialectHandle__" ++ namespace ++ "__")() };
    }
};

pub const Location = struct {
    _inner: c.MlirLocation,

    pub const eql = helpers.eql(Type, c.mlirLocationEqual);
    pub const format = helpers.format(Location, c.mlirLocationPrint);

    pub fn fromSrc(ctx: Context, src: std.builtin.SourceLocation) Location {
        return .{ ._inner = c.mlirLocationFileLineColGet(
            ctx._inner,
            stringRef(src.file),
            @intCast(src.line),
            @intCast(src.column),
        ) };
    }

    pub fn fileLineCol(ctx: Context, file: []const u8, line: usize, column: usize) Location {
        return .{ ._inner = c.mlirLocationFileLineColGet(
            ctx._inner,
            stringRef(file),
            @intCast(line),
            @intCast(column),
        ) };
    }

    pub fn callSite(callee: Location, caller: Location) Location {
        return .{ ._inner = c.mlirLocationCallSiteGet(callee._inner, caller._inner) };
    }

    pub fn fused(ctx: Context, locations: []const Location, metadata: Attribute) Location {
        return .{ ._inner = c.mlirLocationFusedGet(
            ctx._inner,
            @intCast(locations.len),
            @ptrCast(locations.ptr),
            metadata._inner,
        ) };
    }

    pub fn named(loc: Location, ctx: Context, loc_name: [:0]const u8) Location {
        return .{ ._inner = c.mlirLocationNameGet(ctx._inner, stringRef(loc_name), loc._inner) };
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
        return .{ ._inner = c.mlirLocationUnknownGet(ctx._inner) };
    }
};

pub const Block = struct {
    _inner: c.MlirBlock,

    pub const wrapOr = helpers.wrapOr(Block, c.mlirBlockIsNull);
    pub const deinit = helpers.deinit(Block, c.mlirBlockDestroy);

    pub const eql = helpers.eql(Block, c.mlirBlockEqual);

    pub fn init(args: []const Type, locs: []const Location) !Block {
        const block = Block.wrapOr(
            c.mlirBlockCreate(@intCast(args.len), @ptrCast(args.ptr), @ptrCast(locs.ptr)),
        );
        return block orelse error.InvalidMlir;
    }

    pub fn argument(self: Block, index: usize) Value {
        return .{ ._inner = c.mlirBlockGetArgument(self._inner, @intCast(index)) };
    }

    pub fn numArguments(self: Block) usize {
        return @intCast(c.mlirBlockGetNumArguments(self._inner));
    }

    pub fn addArgument(self: *Block, typ: Type, loc: Location) Value {
        return .{ ._inner = c.mlirBlockAddArgument(self._inner, typ._inner, loc._inner) };
    }

    pub fn insertArgument(self: *Block, index: usize, typ: Type, loc: Location) Value {
        return .{ ._inner = c.mlirBlockInsertArgument(self._inner, @intCast(index), typ._inner, loc._inner) };
    }

    pub fn equals(self: Block, other: Block) bool {
        return c.mlirBlockEqual(self._inner, other._inner);
    }

    pub fn appendOperation(self: Block, op: Operation) void {
        c.mlirBlockAppendOwnedOperation(self._inner, op._inner);
    }

    pub fn appendOperations(self: *Block, ops: []const Operation) void {
        for (ops) |op| {
            c.mlirBlockAppendOwnedOperation(self._inner, op._inner);
        }
    }

    pub const RecursiveOpts = enum { open, hermetic };

    pub fn appendValueRecursive(self: Block, value: Value, opt: RecursiveOpts) void {
        switch (value.kind()) {
            .op_result => |parent_op| self.appendOperationRecursive(parent_op, opt),
            .block_argument => |arg| {
                // Hermetic blocks are not allowed to use arguments from other blocks.
                stdx.debug.assert(opt == .open or self.eql(arg.block()), "Can't add {} from {?x} block to {?x} block", .{ arg, arg.block()._inner.ptr, self._inner.ptr });
            },
            .null => @panic("InvalidMlir"),
        }
    }

    pub fn appendOperationRecursive(self: Block, op: Operation, opt: RecursiveOpts) void {
        if (op.block()) |prev_block| {
            // Hermetic blocks are not allowed to reference values from other blocks.
            stdx.debug.assert(opt == .open or self.equals(prev_block), "Can't add {} from {?x} block to {?x} block", .{ op, prev_block._inner.ptr, self._inner.ptr });
            return;
        }
        for (0..op.numOperands()) |i| {
            self.appendValueRecursive(op.operand(i), opt);
        }
        self.appendOperation(op);
    }
};

pub const helpers = struct {
    pub fn eql(T: type, equal_fn: fn (@FieldType(T, "_inner"), @FieldType(T, "_inner")) callconv(.C) bool) fn (T, T) bool {
        return struct {
            fn eql(a: T, b: T) bool {
                return equal_fn(a._inner, b._inner);
            }
        }.eql;
    }

    pub fn deinit(T: type, deinit_fn: fn (@FieldType(T, "_inner")) callconv(.C) void) fn (*T) void {
        return struct {
            fn deinit(a: *T) void {
                deinit_fn(a._inner);
                a.* = undefined;
            }
        }.deinit;
    }

    pub fn dump(T: type, dump_fn: fn (@FieldType(T, "_inner")) callconv(.C) void) fn (T) void {
        return struct {
            fn dump(a: T) void {
                return dump_fn(a._inner);
            }
        }.dump;
    }

    pub fn isNull(T: type, is_null_fn: fn (@FieldType(T, "_inner")) callconv(.C) bool) fn (T) bool {
        return struct {
            fn isNull(a: T) bool {
                return is_null_fn(a._inner);
            }
        }.isNull;
    }

    pub fn format(Any: type, print_fn: fn (@FieldType(Any, "_inner"), ?*const MlirStrCallback, ?*anyopaque) callconv(.C) void) type {
        return struct {
            pub fn format(
                self: Any,
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
                print_fn(self._inner, &Writer.printCallback, &context);
                if (context.err) |err| return err;
            }
        };
    }

    pub fn wrapOr(T: type, is_null_fn: fn (@FieldType(T, "_inner")) callconv(.C) bool) fn (@FieldType(T, "_inner")) ?T {
        return struct {
            fn wrapOr(inner: @FieldType(T, "_inner")) ?T {
                if (is_null_fn(inner)) return null;
                return .{ ._inner = inner };
            }
        }.wrapOr;
    }

    pub fn init(T: type, inner: @FieldType(T, "_inner"), is_null_fn: fn (@FieldType(T, "_inner")) callconv(.C) bool) ?T {
        if (is_null_fn(inner)) return null;
        return .{ ._inner = inner };
    }
};
