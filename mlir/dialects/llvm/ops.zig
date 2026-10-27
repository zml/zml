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
pub fn linkageAttr(ctx: *mlir.Context, value: Linkage) *const mlir.Attribute {
    _ = c.mlirContextGetOrLoadDialect(ctx.ptr(), mlir.stringRef("llvm"));
    return @ptrCast(c.mlirLLVMLinkageAttrGet(ctx.ptr(), @backingInt(value)).ptr);
}

/// `llvm.emit_c_interface`: has the lowering of a `func.func` emit a
/// `_mlir_ciface_` wrapper that takes its memrefs by pointer.
pub fn emitCInterface(ctx: *mlir.Context) mlir.NamedAttribute {
    return .named(ctx, "llvm.emit_c_interface", .unit(ctx));
}

/// `llvm.linkage = #llvm.linkage<linkage>` on a `func.func`.
pub fn linkage(ctx: *mlir.Context, value: Linkage) mlir.NamedAttribute {
    return .named(ctx, "llvm.linkage", linkageAttr(ctx, value));
}

/// `llvm.inline_hint` on a `func.func`: `inlinehint` once lowered.
pub fn inlineHint(ctx: *mlir.Context) mlir.NamedAttribute {
    return .named(ctx, "llvm.inline_hint", .unit(ctx));
}

/// `llvm.noundef` on an argument: never undef or poison.
pub fn noundef(ctx: *mlir.Context) mlir.NamedAttribute {
    return .named(ctx, "llvm.noundef", .unit(ctx));
}

/// `llvm.zeroext` on an argument: a narrow integer passed zero-extended.
pub fn zeroext(ctx: *mlir.Context) mlir.NamedAttribute {
    return .named(ctx, "llvm.zeroext", .unit(ctx));
}

/// `llvm.signext` on an argument: a narrow integer passed sign-extended.
pub fn signext(ctx: *mlir.Context) mlir.NamedAttribute {
    return .named(ctx, "llvm.signext", .unit(ctx));
}

/// `llvm.align = bytes` on a pointer argument.
pub fn alignment(ctx: *mlir.Context, bytes: u64) mlir.NamedAttribute {
    return .named(ctx, "llvm.align", .int(ctx, .i64, @as(i64, @intCast(bytes))));
}

/// `llvm.dereferenceable = bytes` on a pointer argument: that many bytes are
/// readable.
pub fn dereferenceable(ctx: *mlir.Context, bytes: u64) mlir.NamedAttribute {
    return .named(ctx, "llvm.dereferenceable", .int(ctx, .i64, @as(i64, @intCast(bytes))));
}

/// `llvm.elementtype = type` on a pointer argument: its pointee, which
/// typed-pointer IR prints.
pub fn elementtype(ctx: *mlir.Context, pointee: *const mlir.Type) mlir.NamedAttribute {
    return .named(ctx, "llvm.elementtype", .typeAttr(pointee));
}

/// The operation of `llvm.atomicrmw`, as the LLVM dialect numbers it.
pub const AtomicBinOp = enum(i64) {
    xchg = 0,
    add = 1,
    sub = 2,
    _and = 3,
    nand = 4,
    _or = 5,
    _xor = 6,
    max = 7,
    min = 8,
    umax = 9,
    umin = 10,
    fadd = 11,
    fsub = 12,
    fmax = 13,
    fmin = 14,
    uinc_wrap = 15,
    udec_wrap = 16,
    usub_cond = 17,
    usub_sat = 18,
    fmaximum = 19,
    fminimum = 20,
    fmaximumnum = 21,
    fminimumnum = 22,
};

/// The memory ordering of an atomic operation, as the LLVM dialect numbers it.
pub const AtomicOrdering = enum(i64) {
    not_atomic = 0,
    unordered = 1,
    monotonic = 2,
    acquire = 4,
    release = 5,
    acq_rel = 6,
    seq_cst = 7,
};

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
// Aggregates
// =============================================================================

/// llvm.extractvalue — the element of type `result_type` at `position` in the
/// struct or array `container`.
pub fn extractvalue(ctx: *mlir.Context, container: *const mlir.Value, result_type: *const mlir.Type, position: []const i64, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "llvm.extractvalue", .{
        .operands = .{ .flat = &.{container} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{.named(ctx, "position", .denseArray(ctx, .i64, position))},
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
    if (opts.alignment) |bytes| {
        attrs.appendAssumeCapacity(.named(ctx, "alignment", .int(ctx, .i64, bytes)));
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
    if (opts.alignment) |bytes| {
        attrs.appendAssumeCapacity(.named(ctx, "alignment", .int(ctx, .i64, bytes)));
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
    if (opts.alignment) |bytes| attrs.appendAssumeCapacity(.named(ctx, "alignment", .int(ctx, .i64, bytes)));
    return mlir.Operation.make(ctx, "llvm.alloca", .{
        .operands = .{ .flat = &.{array_size} },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// llvm.atomicrmw — combine `val` into the value at `ptr` with `bin_op`,
/// atomically; the result is the previous value, of `val`'s type.
pub fn atomicrmw(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    val: *const mlir.Value,
    bin_op: AtomicBinOp,
    ordering: AtomicOrdering,
    opts: struct {
        /// Synchronization scope, e.g. `"agent"` (none: the whole system).
        syncscope: ?[]const u8 = null,
        /// Alignment in bytes (none: the natural alignment of the type).
        alignment: ?u64 = null,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attrs.appendSliceAssumeCapacity(&.{
        .named(ctx, "bin_op", .int(ctx, .i64, @backingInt(bin_op))),
        .named(ctx, "ordering", .int(ctx, .i64, @backingInt(ordering))),
    });
    if (opts.syncscope) |syncscope| attrs.appendAssumeCapacity(.named(ctx, "syncscope", .string(ctx, syncscope)));
    if (opts.alignment) |bytes| attrs.appendAssumeCapacity(.named(ctx, "alignment", .int(ctx, .i64, bytes)));
    return mlir.Operation.make(ctx, "llvm.atomicrmw", .{
        .operands = .{ .flat = &.{ ptr, val } },
        .results = .{ .flat = &.{val.type_()} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// =============================================================================
// Intrinsics and inline assembly
// =============================================================================

/// llvm.intr.assume — let the optimizer assume that `cond` (i1) holds.
pub fn intr_assume(ctx: *mlir.Context, cond: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "llvm.intr.assume", .{
        .operands = .{ .flat = &.{cond} },
        .attributes = &.{
            .named(ctx, "op_bundle_sizes", .denseArray(ctx, .i32, &.{})),
            .named(ctx, "op_bundle_tags", .array(ctx, &.{})),
        },
        .location = location,
    });
}

pub fn intr_lifetime_start(ctx: *mlir.Context, ptr: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "llvm.intr.lifetime.start", .{
        .operands = .{ .flat = &.{ptr} },
        .location = location,
    });
}

pub fn intr_lifetime_end(ctx: *mlir.Context, ptr: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "llvm.intr.lifetime.end", .{
        .operands = .{ .flat = &.{ptr} },
        .location = location,
    });
}

pub fn extractelement(ctx: *mlir.Context, vector: *const mlir.Value, position: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "llvm.extractelement", .{
        .operands = .{ .flat = &.{ vector, position } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn insertelement(ctx: *mlir.Context, vector: *const mlir.Value, value: *const mlir.Value, position: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "llvm.insertelement", .{
        .operands = .{ .flat = &.{ vector, value, position } },
        .results = .{ .flat = &.{vector.type_()} },
        .location = location,
    });
}

pub fn shufflevector(ctx: *mlir.Context, v1: *const mlir.Value, v2: *const mlir.Value, result_type: *const mlir.Type, mask: []const i32, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "llvm.shufflevector", .{
        .operands = .{ .flat = &.{ v1, v2 } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{.named(ctx, "mask", .denseArray(ctx, .i32, mask))},
        .location = location,
    });
}

pub fn mlir_undef(ctx: *mlir.Context, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "llvm.mlir.undef", .{
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn mlir_poison(ctx: *mlir.Context, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "llvm.mlir.poison", .{
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn freeze(ctx: *mlir.Context, value: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "llvm.freeze", .{
        .operands = .{ .flat = &.{value} },
        .results = .{ .flat = &.{value.type_()} },
        .location = location,
    });
}

pub const BrOpts = struct {
    loop_annotation: ?*const LoopAnnotationAttr = null,
};

/// `llvm.br`. Not verified: a terminator only verifies inside its block.
pub fn br(ctx: *mlir.Context, dest: *mlir.Block, dest_operands: []const *const mlir.Value, opts: BrOpts, location: *const mlir.Location) *mlir.Operation {
    var state: mlir.OperationState = .init("llvm.br", location);
    state.addOperands(dest_operands);
    var successors: [1]*const mlir.Block = .{dest};
    state.addSuccessors(&successors);
    if (opts.loop_annotation) |annotation| state.addAttributes(&.{.named(ctx, "loop_annotation", annotation.attribute())});
    return mlir.Operation.init(&state) catch @panic("llvm.br: failed to create the operation");
}

/// `llvm.unreachable`. Not verified: a terminator only verifies inside its
/// block.
pub fn unreachable_(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "llvm.unreachable", .{ .location = location, .verify = false });
}

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
