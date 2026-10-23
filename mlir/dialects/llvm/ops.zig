const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

pub fn registerDialect(registry: *mlir.DialectRegistry) void {
    registry.registerDialect("llvm");
}

pub const operation_names: []const []const u8 = &.{
    "llvm.bitcast",
    "llvm.call_intrinsic",
    "llvm.fmul",
    "llvm.inline_asm",
    "llvm.inttoptr",
    "llvm.load",
    "llvm.store",
};

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
