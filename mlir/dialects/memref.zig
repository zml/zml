//! Layer A builders for the upstream `memref` dialect.
//!
//! Covers the ops a Mosaic / Pallas-style kernel typically emits or that a
//! lowering pipeline needs to reason about. The dialect is registered via the
//! `mlirGetDialectHandle__memref__` symbol shipped by upstream
//! `@llvm-project//mlir:CAPIMemRef`.
//!
//! Intentionally omitted (add on demand): `generic_atomic_rmw` (region-bearing,
//! rarely used outside lowerings), `dma_start` / `dma_wait` (TPU kernels go via
//! the `tpu` dialect's DMAs), `extract_strided_metadata` (lowering-only).
//!
//! Mixed static/dynamic sizes/offsets/strides — the ops that consume them
//! (`subview`, `reinterpret_cast`, `expand_shape`, `collapse_shape`,
//! `reshape`) follow MLIR's convention: pass `static_*` arrays with the
//! sentinel `kDynamic` for entries that come from the SSA value lists.

const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

/// Sentinel matching `mlir::ShapedType::kDynamic` (== INT64_MIN).
/// Use this in static-size/offset/stride arrays to mark a slot as dynamic.
pub const kDynamic: i64 = std.math.minInt(i64);

// =============================================================================
// Allocation — alloc / alloca / dealloc / realloc
// =============================================================================

pub const AllocOpts = struct {
    alignment: ?i64 = null,
};

/// memref.alloc — heap allocation. `dynamic_sizes` provides one Index value per
/// dynamic dim of `result_type` (left-to-right). `symbol_operands` binds to
/// the symbols of the layout map (rare; pass &.{} when none).
pub fn alloc(
    ctx: *mlir.Context,
    result_type: *const mlir.Type,
    dynamic_sizes: []const *const mlir.Value,
    symbol_operands: []const *const mlir.Value,
    opts: AllocOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands_buf.appendSliceAssumeCapacity(dynamic_sizes);
    operands_buf.appendSliceAssumeCapacity(symbol_operands);

    const seg_sizes = [2]i32{ @intCast(dynamic_sizes.len), @intCast(symbol_operands.len) };
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "operandSegmentSizes", mlir.denseArrayAttribute(ctx, .i32, &seg_sizes)));
    if (opts.alignment) |a| {
        attrs.appendAssumeCapacity(.named(ctx, "alignment", mlir.integerAttribute(ctx, .i64, a)));
    }

    return mlir.Operation.make(ctx, "memref.alloc", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// memref.alloca — stack allocation; same operand/attribute layout as alloc.
pub fn alloca(
    ctx: *mlir.Context,
    result_type: *const mlir.Type,
    dynamic_sizes: []const *const mlir.Value,
    symbol_operands: []const *const mlir.Value,
    opts: AllocOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands_buf.appendSliceAssumeCapacity(dynamic_sizes);
    operands_buf.appendSliceAssumeCapacity(symbol_operands);

    const seg_sizes = [2]i32{ @intCast(dynamic_sizes.len), @intCast(symbol_operands.len) };
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "operandSegmentSizes", mlir.denseArrayAttribute(ctx, .i32, &seg_sizes)));
    if (opts.alignment) |a| {
        attrs.appendAssumeCapacity(.named(ctx, "alignment", mlir.integerAttribute(ctx, .i64, a)));
    }

    return mlir.Operation.make(ctx, "memref.alloca", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

pub fn dealloc(
    ctx: *mlir.Context,
    memref: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "memref.dealloc", .{
        .operands = .{ .flat = &.{memref} },
        .location = location,
    });
}

/// memref.realloc — grow/shrink an existing allocation. `dynamic_size` is
/// optional; pass null to keep the size encoded in `result_type`.
pub fn realloc(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    dynamic_size: ?*const mlir.Value,
    result_type: *const mlir.Type,
    opts: AllocOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 2) = .empty;
    operands_buf.appendAssumeCapacity(src);
    if (dynamic_size) |s| operands_buf.appendAssumeCapacity(s);

    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.alignment) |a| {
        attrs.appendAssumeCapacity(.named(ctx, "alignment", mlir.integerAttribute(ctx, .i64, a)));
    }

    return mlir.Operation.make(ctx, "memref.realloc", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// =============================================================================
// Load / Store — plain (non-affine) memref load/store
// =============================================================================

pub const LoadOpts = struct {
    /// Hint that the lowering may use a non-temporal load.
    nontemporal: bool = false,
};

/// memref.load — load a single element from a memref.
pub fn load(
    ctx: *mlir.Context,
    memref: *const mlir.Value,
    indices: []const *const mlir.Value,
    opts: LoadOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    buf.appendAssumeCapacity(memref);
    buf.appendSliceAssumeCapacity(indices);

    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.nontemporal) {
        attrs.appendAssumeCapacity(.named(ctx, "nontemporal", mlir.boolAttribute(ctx, true)));
    }

    return mlir.Operation.make(ctx, "memref.load", .{
        .operands = .{ .flat = buf.constSlice() },
        .result_type_inference = true,
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

pub const StoreOpts = struct {
    nontemporal: bool = false,
};

/// memref.store — store a single element into a memref.
pub fn store(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    memref: *const mlir.Value,
    indices: []const *const mlir.Value,
    opts: StoreOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    buf.appendSliceAssumeCapacity(&.{ value, memref });
    buf.appendSliceAssumeCapacity(indices);

    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.nontemporal) {
        attrs.appendAssumeCapacity(.named(ctx, "nontemporal", mlir.boolAttribute(ctx, true)));
    }

    return mlir.Operation.make(ctx, "memref.store", .{
        .operands = .{ .flat = buf.constSlice() },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// =============================================================================
// Atomic RMW
// =============================================================================

/// Matches `mlir::arith::AtomicRMWKind` (I64EnumAttr) in upstream `Arith` td.
/// memref.atomic_rmw and affine.parallel both encode reductions through this.
pub const AtomicRMWKind = enum(i64) {
    addf = 0,
    addi = 1,
    assign = 2,
    maximumf = 3,
    maxs = 4,
    maxu = 5,
    minimumf = 6,
    mins = 7,
    minu = 8,
    mulf = 9,
    muli = 10,
    ori = 11,
    andi = 12,
    maxnumf = 13,
    minnumf = 14,
};

/// memref.atomic_rmw — atomic read-modify-write.
pub fn atomic_rmw(
    ctx: *mlir.Context,
    kind: AtomicRMWKind,
    value: *const mlir.Value,
    memref: *const mlir.Value,
    indices: []const *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    var buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    buf.appendSliceAssumeCapacity(&.{ value, memref });
    buf.appendSliceAssumeCapacity(indices);

    return mlir.Operation.make(ctx, "memref.atomic_rmw", .{
        .operands = .{ .flat = buf.constSlice() },
        .results = .{ .flat = &.{value.type_()} },
        .attributes = &.{
            .named(ctx, "kind", mlir.integerAttribute(ctx, .i64, @intFromEnum(kind))),
        },
        .location = location,
    });
}

// =============================================================================
// Casts / metadata-only ops
// =============================================================================

/// memref.cast — change layout/element-shape compatibility (e.g. unranked ↔
/// ranked, dynamic ↔ static). Element type and total memory must match.
pub fn cast(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "memref.cast", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// memref.memory_space_cast — change the memory-space attribute on a memref.
pub fn memory_space_cast(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "memref.memory_space_cast", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// memref.assume_alignment — pure, returns the same memref with an alignment
/// hint attached.
pub fn assume_alignment(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    alignment: i32,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "memref.assume_alignment", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{src.type_()} },
        .attributes = &.{
            .named(ctx, "alignment", mlir.integerAttribute(ctx, .i32, alignment)),
        },
        .location = location,
    });
}

// =============================================================================
// Shape introspection
// =============================================================================

/// memref.dim — returns the size of dim `index` of `memref` as `index`.
pub fn dim(
    ctx: *mlir.Context,
    memref: *const mlir.Value,
    index: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "memref.dim", .{
        .operands = .{ .flat = &.{ memref, index } },
        .results = .{ .flat = &.{mlir.indexType(ctx)} },
        .location = location,
    });
}

/// memref.rank — returns rank of `memref` as `index`.
pub fn rank(
    ctx: *mlir.Context,
    memref: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "memref.rank", .{
        .operands = .{ .flat = &.{memref} },
        .results = .{ .flat = &.{mlir.indexType(ctx)} },
        .location = location,
    });
}

/// memref.extract_aligned_pointer_as_index — get the aligned base pointer of
/// a memref as an `index`.
pub fn extract_aligned_pointer_as_index(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "memref.extract_aligned_pointer_as_index", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{mlir.indexType(ctx)} },
        .location = location,
    });
}

// =============================================================================
// View / SubView / Reinterpret
// =============================================================================

/// memref.view — type-changing view of a flat 1-D `memref<?xi8>` as a higher-
/// rank memref. `byte_shift` is an `index` byte offset; `dynamic_sizes`
/// supplies one operand per dynamic dim of `result_type`.
pub fn view(
    ctx: *mlir.Context,
    source: *const mlir.Value,
    byte_shift: *const mlir.Value,
    dynamic_sizes: []const *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    buf.appendSliceAssumeCapacity(&.{ source, byte_shift });
    buf.appendSliceAssumeCapacity(dynamic_sizes);

    return mlir.Operation.make(ctx, "memref.view", .{
        .operands = .{ .flat = buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// Mixed static/dynamic offsets-sizes-strides — `static_*` slots holding
/// `kDynamic` consume one entry from the corresponding `dynamic_*` slice.
pub const SubViewArgs = struct {
    static_offsets: []const i64,
    static_sizes: []const i64,
    static_strides: []const i64,
    dynamic_offsets: []const *const mlir.Value = &.{},
    dynamic_sizes: []const *const mlir.Value = &.{},
    dynamic_strides: []const *const mlir.Value = &.{},
};

/// memref.subview — strided view into a source memref. The result type must
/// be precomputed by the caller (the dialect provides `inferResultType`,
/// but we don't expose that yet — pass an explicit `result_type`).
pub fn subview(
    ctx: *mlir.Context,
    source: *const mlir.Value,
    args: SubViewArgs,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands_buf.appendAssumeCapacity(source);
    operands_buf.appendSliceAssumeCapacity(args.dynamic_offsets);
    operands_buf.appendSliceAssumeCapacity(args.dynamic_sizes);
    operands_buf.appendSliceAssumeCapacity(args.dynamic_strides);

    const seg_sizes = [4]i32{
        1,
        @intCast(args.dynamic_offsets.len),
        @intCast(args.dynamic_sizes.len),
        @intCast(args.dynamic_strides.len),
    };

    return mlir.Operation.make(ctx, "memref.subview", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", mlir.denseArrayAttribute(ctx, .i32, &seg_sizes)),
            .named(ctx, "static_offsets", mlir.denseArrayAttribute(ctx, .i64, args.static_offsets)),
            .named(ctx, "static_sizes", mlir.denseArrayAttribute(ctx, .i64, args.static_sizes)),
            .named(ctx, "static_strides", mlir.denseArrayAttribute(ctx, .i64, args.static_strides)),
        },
        .location = location,
    });
}

pub const ReinterpretCastArgs = struct {
    static_offsets: []const i64,
    static_sizes: []const i64,
    static_strides: []const i64,
    dynamic_offsets: []const *const mlir.Value = &.{},
    dynamic_sizes: []const *const mlir.Value = &.{},
    dynamic_strides: []const *const mlir.Value = &.{},
};

/// memref.reinterpret_cast — changes offset/sizes/strides of a memref without
/// touching the underlying memory.
pub fn reinterpret_cast(
    ctx: *mlir.Context,
    source: *const mlir.Value,
    args: ReinterpretCastArgs,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands_buf.appendAssumeCapacity(source);
    operands_buf.appendSliceAssumeCapacity(args.dynamic_offsets);
    operands_buf.appendSliceAssumeCapacity(args.dynamic_sizes);
    operands_buf.appendSliceAssumeCapacity(args.dynamic_strides);

    const seg_sizes = [4]i32{
        1,
        @intCast(args.dynamic_offsets.len),
        @intCast(args.dynamic_sizes.len),
        @intCast(args.dynamic_strides.len),
    };

    return mlir.Operation.make(ctx, "memref.reinterpret_cast", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", mlir.denseArrayAttribute(ctx, .i32, &seg_sizes)),
            .named(ctx, "static_offsets", mlir.denseArrayAttribute(ctx, .i64, args.static_offsets)),
            .named(ctx, "static_sizes", mlir.denseArrayAttribute(ctx, .i64, args.static_sizes)),
            .named(ctx, "static_strides", mlir.denseArrayAttribute(ctx, .i64, args.static_strides)),
        },
        .location = location,
    });
}

// =============================================================================
// Reshape — expand_shape / collapse_shape / reshape
// =============================================================================

/// memref.reshape — reshape a memref using a separate `shape` memref of `index`
/// elements. `result_type` carries the new shape.
pub fn reshape(
    ctx: *mlir.Context,
    source: *const mlir.Value,
    shape: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "memref.reshape", .{
        .operands = .{ .flat = &.{ source, shape } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `reassociation[i]` lists the source dimensions that map into result dim i
/// (for collapse_shape) or the result dimensions that come from source dim i
/// (for expand_shape). Each inner slice is stored as an `IntegerAttr` array.
pub const ReassociationArgs = struct {
    /// Index groupings.
    reassociation: []const []const i64,
    /// For expand_shape only: per-result-dim sizes, with `kDynamic` for slots
    /// resolved from `dynamic_output_shape`. Pass &.{} for collapse_shape.
    static_output_shape: []const i64 = &.{},
    /// For expand_shape only: the dynamic dim values referenced by `kDynamic`
    /// entries in `static_output_shape`.
    dynamic_output_shape: []const *const mlir.Value = &.{},
};

fn buildReassociationAttr(ctx: *mlir.Context, reassociation: []const []const i64) *const mlir.Attribute {
    var outer: stdx.BoundedArray(*const mlir.Attribute, 32) = .empty;
    var inner_buf: stdx.BoundedArray(*const mlir.Attribute, 64) = .empty;
    for (reassociation) |group| {
        const start = inner_buf.len;
        for (group) |idx| {
            inner_buf.appendAssumeCapacity(mlir.integerAttribute(ctx, .i64, idx));
        }
        const inner_attr = mlir.arrayAttribute(ctx, inner_buf.constSlice()[start..]);
        outer.appendAssumeCapacity(inner_attr);
    }
    return mlir.arrayAttribute(ctx, outer.constSlice());
}

/// memref.expand_shape — split source dimensions into multiple result
/// dimensions. `static_output_shape` must have one entry per result dim;
/// dynamic entries are filled from `dynamic_output_shape` in order.
pub fn expand_shape(
    ctx: *mlir.Context,
    source: *const mlir.Value,
    args: ReassociationArgs,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 32) = .empty;
    operands_buf.appendAssumeCapacity(source);
    operands_buf.appendSliceAssumeCapacity(args.dynamic_output_shape);

    return mlir.Operation.make(ctx, "memref.expand_shape", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "reassociation", buildReassociationAttr(ctx, args.reassociation)),
            .named(ctx, "static_output_shape", mlir.denseArrayAttribute(ctx, .i64, args.static_output_shape)),
        },
        .location = location,
    });
}

/// memref.collapse_shape — merge consecutive source dimensions into single
/// result dimensions. Reassociation map is canonical.
pub fn collapse_shape(
    ctx: *mlir.Context,
    source: *const mlir.Value,
    reassociation: []const []const i64,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "memref.collapse_shape", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "reassociation", buildReassociationAttr(ctx, reassociation)),
        },
        .location = location,
    });
}

// =============================================================================
// Transpose — metadata-only permutation. Permutation is an AffineMapAttr.
// =============================================================================

/// memref.transpose — permute the dimensions of `src` according to
/// `permutation` (which must be a permutation of `(d0, ..., dN-1)`). Build
/// the map with `mlir.AffineMap.get` or parse via
/// `mlir.parseAffineMapAttribute`.
pub fn transpose(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    permutation: *const mlir.Attribute,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "memref.transpose", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "permutation", permutation),
        },
        .location = location,
    });
}

// =============================================================================
// Copy / Prefetch
// =============================================================================

/// memref.copy — copy source memref into target memref (same element type,
/// same total element count).
pub fn copy(
    ctx: *mlir.Context,
    source: *const mlir.Value,
    target: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "memref.copy", .{
        .operands = .{ .flat = &.{ source, target } },
        .location = location,
    });
}

pub const PrefetchOpts = struct {
    /// True for a write prefetch, false for a read prefetch.
    is_write: bool = false,
    /// 0 = no locality, 3 = extreme locality (keep in cache).
    locality_hint: i32 = 3,
    /// True for data cache, false for instruction cache.
    is_data_cache: bool = true,
};

/// memref.prefetch — non-binding hint to bring `memref[indices]` into cache.
pub fn prefetch(
    ctx: *mlir.Context,
    memref: *const mlir.Value,
    indices: []const *const mlir.Value,
    opts: PrefetchOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    buf.appendAssumeCapacity(memref);
    buf.appendSliceAssumeCapacity(indices);

    return mlir.Operation.make(ctx, "memref.prefetch", .{
        .operands = .{ .flat = buf.constSlice() },
        .attributes = &.{
            .named(ctx, "isWrite", mlir.boolAttribute(ctx, opts.is_write)),
            .named(ctx, "localityHint", mlir.integerAttribute(ctx, .i32, opts.locality_hint)),
            .named(ctx, "isDataCache", mlir.boolAttribute(ctx, opts.is_data_cache)),
        },
        .location = location,
    });
}

// =============================================================================
// Globals
// =============================================================================

pub const GlobalArgs = struct {
    pub const Visibility = enum { public, private, nested };

    name: []const u8,
    type_: *const mlir.Type,
    /// Optional dense initial value. `null` ⇒ uninitialised global.
    initial_value: ?*const mlir.Attribute = null,
    visibility: Visibility = .public,
    constant: bool = false,
    alignment: ?i64 = null,
};

/// memref.global — declare a module-scope memref global (matches the symbol
/// referenced by `memref.get_global`).
pub fn global(ctx: *mlir.Context, args: GlobalArgs, location: *const mlir.Location) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 6) = .empty;
    attrs.appendSliceAssumeCapacity(&.{
        .named(ctx, "sym_name", mlir.stringAttribute(ctx, args.name)),
        .named(ctx, "sym_visibility", mlir.stringAttribute(ctx, @tagName(args.visibility))),
        .named(ctx, "type", mlir.typeAttribute(args.type_)),
    });
    if (args.initial_value) |iv| {
        attrs.appendAssumeCapacity(.named(ctx, "initial_value", iv));
    }
    if (args.constant) {
        attrs.appendAssumeCapacity(.named(ctx, "constant", mlir.unitAttribute(ctx)));
    }
    if (args.alignment) |a| {
        attrs.appendAssumeCapacity(.named(ctx, "alignment", mlir.integerAttribute(ctx, .i64, a)));
    }

    return mlir.Operation.make(ctx, "memref.global", .{
        .attributes = attrs.constSlice(),
        .verify = false,
        .location = location,
    });
}

/// memref.get_global — fetch the value of a `memref.global` by symbol name.
pub fn get_global(
    ctx: *mlir.Context,
    name: []const u8,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "memref.get_global", .{
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "name", mlir.flatSymbolRefAttribute(ctx, name)),
        },
        .location = location,
    });
}

test {
    std.testing.refAllDecls(@This());
}
