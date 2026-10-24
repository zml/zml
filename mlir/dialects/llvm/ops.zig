const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

pub fn registerDialect(registry: *mlir.DialectRegistry) void {
    registry.registerDialect("llvm");
}

// =============================================================================
// Types
// =============================================================================

/// Address spaces of `!llvm.ptr<N>` on NVPTX.
pub const NVPTXAddressSpace = enum(u32) {
    generic = 0,
    global = 1,
    shared = 3,
    constant = 4,
    local = 5,
    /// Tensor memory (SM100 TMEM).
    tensor = 6,
    /// Shared memory of any CTA of the cluster (`shared::cluster`).
    shared_cluster = 7,
};

/// `!llvm.ptr<space>`, an opaque pointer. Loads the LLVM dialect into `ctx`.
pub fn pointerType(ctx: *mlir.Context, space: anytype) *const mlir.Type {
    _ = c.mlirContextGetOrLoadDialect(ctx.ptr(), mlir.stringRef("llvm"));
    return @ptrCast(c.mlirLLVMPointerTypeGet(ctx.ptr(), @intCast(@backingInt(space))).ptr);
}

/// `!llvm.struct<(fields...)>`, a literal struct.
pub fn structType(ctx: *mlir.Context, fields: []const *const mlir.Type) *const mlir.Type {
    _ = c.mlirContextGetOrLoadDialect(ctx.ptr(), mlir.stringRef("llvm"));
    return @ptrCast(c.mlirLLVMStructTypeLiteralGet(ctx.ptr(), @intCast(fields.len), @ptrCast(fields), false).ptr);
}

/// `!llvm.struct<"name", (fields...)>`, an identified struct; null if `name`
/// already names a struct of other fields.
pub fn identifiedStructType(ctx: *mlir.Context, name: []const u8, fields: []const *const mlir.Type) ?*const mlir.Type {
    _ = c.mlirContextGetOrLoadDialect(ctx.ptr(), mlir.stringRef("llvm"));
    const t = c.mlirLLVMStructTypeIdentifiedGet(ctx.ptr(), mlir.stringRef(name));
    if (c.mlirLLVMStructTypeSetBody(t, @intCast(fields.len), @ptrCast(fields), false).value == 0) return null;
    return @ptrCast(t.ptr);
}

/// `!llvm.array<count x element_type>`.
pub fn arrayType(element_type: *const mlir.Type, count: u32) *const mlir.Type {
    return @ptrCast(c.mlirLLVMArrayTypeGet(element_type.ptr(), count).ptr);
}

// =============================================================================
// Attributes
// =============================================================================

fn optionalBool(ctx: *mlir.Context, value: ?bool) c.MlirAttribute {
    return if (value) |v| mlir.Attribute.boolean(ctx, v).ptr() else .{ .ptr = null };
}

/// `#llvm.loop_unroll`: the `llvm.loop.unroll.*` loop metadata.
pub const LoopUnrollAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsALLVMLoopUnroll;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        disable: ?bool = null,
        count: ?u32 = null,
        runtimeDisable: ?bool = null,
        full: ?bool = null,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) *const Self {
        const count: c.MlirAttribute = if (args.count) |n| mlir.Attribute.int(ctx, .i32, n).ptr() else .{ .ptr = null };
        return @ptrCast(c.mlirLLVMLoopUnrollAttrGet(ctx.ptr(), optionalBool(ctx, args.disable), count, optionalBool(ctx, args.runtimeDisable), optionalBool(ctx, args.full)).ptr);
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

/// A global's linkage, as `#llvm.linkage` spells it.
pub const Linkage = enum(c_uint) {
    external = c.MlirLLVMLinkageExternal,
    available_externally = c.MlirLLVMLinkageAvailableExternally,
    linkonce = c.MlirLLVMLinkageLinkonce,
    linkonce_odr = c.MlirLLVMLinkageLinkonceODR,
    weak = c.MlirLLVMLinkageWeak,
    weak_odr = c.MlirLLVMLinkageWeakODR,
    appending = c.MlirLLVMLinkageAppending,
    internal = c.MlirLLVMLinkageInternal,
    private = c.MlirLLVMLinkagePrivate,
    extern_weak = c.MlirLLVMLinkageExternWeak,
    common = c.MlirLLVMLinkageCommon,
};

/// `#llvm.linkage<linkage>`; `llvm.linkage` on a `func.func` sets its
/// linkage once lowered to LLVM. Loads the LLVM dialect into `ctx`, which
/// must have it registered.
pub fn linkageAttr(ctx: *mlir.Context, linkage: Linkage) *const mlir.Attribute {
    _ = c.mlirContextGetOrLoadDialect(ctx.ptr(), mlir.stringRef("llvm"));
    return @ptrCast(c.mlirLLVMLinkageAttrGet(ctx.ptr(), @backingInt(linkage)).ptr);
}

/// `#llvm.loop_annotation`: the `llvm.loop` metadata of a loop.
pub const LoopAnnotationAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsALLVMLoopAnnotation;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        disableNonforced: ?bool = null,
        unroll: ?*const LoopUnrollAttr = null,
        mustProgress: ?bool = null,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) *const Self {
        const unroll: c.MlirAttribute = if (args.unroll) |u| u.ptr() else .{ .ptr = null };
        return @ptrCast(c.mlirLLVMLoopAnnotationAttrGet(ctx.ptr(), optionalBool(ctx, args.disableNonforced), unroll, optionalBool(ctx, args.mustProgress)).ptr);
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

// =============================================================================
// Casts and arithmetic
// =============================================================================

/// llvm.inttoptr — an integer address as a pointer of type `result_type`.
pub fn inttoptr(ctx: *mlir.Context, address: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "llvm.inttoptr", .{
        .operands = .{ .flat = &.{address} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// llvm.bitcast — reinterpret `value` as `result_type` (same bit width).
pub fn bitcast(ctx: *mlir.Context, value: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "llvm.bitcast", .{
        .operands = .{ .flat = &.{value} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// llvm.fmul — floating-point (or floating-point vector) product of type `result_type`.
pub fn fmul(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "llvm.fmul", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// =============================================================================
// Memory
// =============================================================================

pub const LoadOpts = struct {
    /// Alignment in bytes (none: the natural alignment of the type).
    alignment: ?u64 = null,
    /// The memory is not written while the kernel runs (`!invariant.load`; on NVPTX
    /// global memory this selects the non-coherent `ld.global.nc`).
    invariant: bool = false,
};

/// llvm.load — a value of type `result_type` through `pointer`.
pub fn load(ctx: *mlir.Context, pointer: *const mlir.Value, result_type: *const mlir.Type, opts: LoadOpts, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (opts.alignment) |alignment| {
        attrs.appendAssumeCapacity(.named(ctx, "alignment", .int(ctx, .i64, alignment)));
    }
    if (opts.invariant) {
        attrs.appendAssumeCapacity(.named(ctx, "invariant", .unit(ctx)));
    }
    return mlir.Operation.make(ctx, "llvm.load", .{
        .operands = .{ .flat = &.{pointer} },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

pub const StoreOpts = struct {
    /// Alignment in bytes (none: the natural alignment of the type).
    alignment: ?u64 = null,
};

/// llvm.store — `value` through `pointer`.
pub fn store(ctx: *mlir.Context, value: *const mlir.Value, pointer: *const mlir.Value, opts: StoreOpts, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.alignment) |alignment| {
        attrs.appendAssumeCapacity(.named(ctx, "alignment", .int(ctx, .i64, alignment)));
    }
    return mlir.Operation.make(ctx, "llvm.store", .{
        .operands = .{ .flat = &.{ value, pointer } },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// An index of `getelementptr`: a constant, or an SSA integer.
pub const GEPIndex = union(enum) {
    constant: i32,
    value: *const mlir.Value,
};

/// llvm.getelementptr — `base` offset by `indices` into `elem_type`, with
/// `base`'s pointer type.
pub fn getelementptr(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const GEPIndex,
    elem_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    const dynamic_index = std.math.minInt(i32);
    var operands: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    var raw: stdx.BoundedArray(i32, 16) = .empty;
    operands.appendAssumeCapacity(base);
    for (indices) |index| switch (index) {
        .constant => |c_| raw.appendAssumeCapacity(c_),
        .value => |v| {
            raw.appendAssumeCapacity(dynamic_index);
            operands.appendAssumeCapacity(v);
        },
    };
    return mlir.Operation.make(ctx, "llvm.getelementptr", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{base.type_()} },
        .attributes = &.{
            .named(ctx, "rawConstantIndices", .denseArray(ctx, .i32, raw.constSlice())),
            .named(ctx, "elem_type", .typeAttr(elem_type)),
        },
        .location = location,
    });
}

/// llvm.alloca — stack storage for `array_size` elements of `elem_type`, as a
/// pointer of type `result_type`.
pub fn alloca(
    ctx: *mlir.Context,
    array_size: *const mlir.Value,
    result_type: *const mlir.Type,
    elem_type: *const mlir.Type,
    opts: struct {
        alignment: ?u64 = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "elem_type", .typeAttr(elem_type)));
    if (opts.alignment) |alignment| attrs.appendAssumeCapacity(.named(ctx, "alignment", .int(ctx, .i64, alignment)));
    return mlir.Operation.make(ctx, "llvm.alloca", .{
        .operands = .{ .flat = &.{array_size} },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// =============================================================================
// Intrinsics and inline assembly
// =============================================================================

/// llvm.call_intrinsic — call the LLVM intrinsic `name` (e.g. `llvm.nvvm.*`), with an
/// optional result of type `result_type`.
pub fn call_intrinsic(
    ctx: *mlir.Context,
    name: []const u8,
    operands: []const *const mlir.Value,
    opts: struct {
        result_type: ?*const mlir.Type = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "llvm.call_intrinsic", .{
        .operands = .{ .flat = operands },
        .results = .{ .flat = if (opts.result_type) |r| &.{r} else &.{} },
        .attributes = &.{
            .named(ctx, "intrin", .string(ctx, name)),
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &.{ @intCast(operands.len), 0 })),
            .named(ctx, "op_bundle_sizes", .denseArray(ctx, .i32, &.{})),
        },
        .location = location,
    });
}

pub const InlineAsmOpts = struct {
    result_type: ?*const mlir.Type = null,

    /// LLVM constraint string, e.g. `"=r,r"` (outputs first).
    constraints: []const u8 = "",
    /// The assembly has effects beyond its outputs (it is never removed or merged).
    has_side_effects: bool = false,
};

/// llvm.inline_asm — the PTX (or other target) assembly `asm_string`, with `$i`
/// placeholders bound to the result then `operands`.
pub fn inline_asm(
    ctx: *mlir.Context,
    asm_string: []const u8,
    operands: []const *const mlir.Value,
    opts: InlineAsmOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "asm_string", .string(ctx, asm_string)));
    attrs.appendAssumeCapacity(.named(ctx, "constraints", .string(ctx, opts.constraints)));
    if (opts.has_side_effects) {
        attrs.appendAssumeCapacity(.named(ctx, "has_side_effects", .unit(ctx)));
    }
    return mlir.Operation.make(ctx, "llvm.inline_asm", .{
        .operands = .{ .flat = operands },
        .results = .{ .flat = if (opts.result_type) |r| &.{r} else &.{} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}
