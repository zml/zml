const std = @import("std");

const mlir = @import("mlir");

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

/// `!llvm.ptr<space>`, an opaque pointer.
///
/// Parsed rather than built with `mlirLLVMPointerTypeGet`: the type must come from the
/// LLVM dialect registered in `ctx`, which may be another MLIR build than the C API's
/// (e.g. in the CuTe compiler context), and loading the C API's dialect there would
/// clash with it.
pub fn pointerType(ctx: *mlir.Context, space: anytype) *const mlir.Type {
    var buf: [32]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, "!llvm.ptr<{d}>", .{@backingInt(space)}) catch unreachable;
    return mlir.Type.parse(ctx, text) catch std.debug.panic("failed to parse LLVM pointer type '{s}'", .{text});
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

pub const LoadOptions = struct {
    /// Alignment in bytes (none: the natural alignment of the type).
    alignment: ?u64 = null,
    /// The memory is not written while the kernel runs (`!invariant.load`; on NVPTX
    /// global memory this selects the non-coherent `ld.global.nc`).
    invariant: bool = false,
};

/// llvm.load — a value of type `result_type` through `pointer`.
pub fn load(ctx: *mlir.Context, pointer: *const mlir.Value, result_type: *const mlir.Type, opts: LoadOptions, location: *const mlir.Location) *mlir.Operation {
    var attrs: [2]mlir.NamedAttribute = undefined;
    var len: usize = 0;
    if (opts.alignment) |alignment| {
        attrs[len] = .named(ctx, "alignment", .int(ctx, .i64, alignment));
        len += 1;
    }
    if (opts.invariant) {
        attrs[len] = .named(ctx, "invariant", .unit(ctx));
        len += 1;
    }
    return mlir.Operation.make(ctx, "llvm.load", .{
        .operands = .{ .flat = &.{pointer} },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs[0..len],
        .location = location,
    });
}

pub const StoreOptions = struct {
    /// Alignment in bytes (none: the natural alignment of the type).
    alignment: ?u64 = null,
};

/// llvm.store — `value` through `pointer`.
pub fn store(ctx: *mlir.Context, value: *const mlir.Value, pointer: *const mlir.Value, opts: StoreOptions, location: *const mlir.Location) *mlir.Operation {
    var attrs: [1]mlir.NamedAttribute = undefined;
    var len: usize = 0;
    if (opts.alignment) |alignment| {
        attrs[len] = .named(ctx, "alignment", .int(ctx, .i64, alignment));
        len += 1;
    }
    return mlir.Operation.make(ctx, "llvm.store", .{
        .operands = .{ .flat = &.{ value, pointer } },
        .attributes = attrs[0..len],
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
    result_type: ?*const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "llvm.call_intrinsic", .{
        .operands = .{ .flat = operands },
        .results = .{ .flat = if (result_type) |r| &.{r} else &.{} },
        .attributes = &.{
            .named(ctx, "intrin", .string(ctx, name)),
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &.{ @intCast(operands.len), 0 })),
            .named(ctx, "op_bundle_sizes", .denseArray(ctx, .i32, &.{})),
        },
        .location = location,
    });
}

pub const InlineAsmOptions = struct {
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
    result_type: ?*const mlir.Type,
    opts: InlineAsmOptions,
    location: *const mlir.Location,
) *mlir.Operation {
    var attrs: [3]mlir.NamedAttribute = undefined;
    attrs[0] = .named(ctx, "asm_string", .string(ctx, asm_string));
    attrs[1] = .named(ctx, "constraints", .string(ctx, opts.constraints));
    var len: usize = 2;
    if (opts.has_side_effects) {
        attrs[len] = .named(ctx, "has_side_effects", .unit(ctx));
        len += 1;
    }
    return mlir.Operation.make(ctx, "llvm.inline_asm", .{
        .operands = .{ .flat = operands },
        .results = .{ .flat = if (result_type) |r| &.{r} else &.{} },
        .attributes = attrs[0..len],
        .location = location,
    });
}

test {
    std.testing.refAllDecls(@This());
}
