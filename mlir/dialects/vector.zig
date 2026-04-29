const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

// =============================================================================
// Enum attributes — parsed from their textual `#vector.<kind><...>` form.
// =============================================================================

/// Matches `mlir::vector::CombiningKind` (I32EnumAttr in VectorAttributes.td).
pub const CombiningKind = enum {
    add,
    mul,
    minui,
    minsi,
    minnumf,
    maxui,
    maxsi,
    maxnumf,
    @"and",
    @"or",
    xor,
    minimumf,
    maximumf,

    pub fn attribute(self: CombiningKind, ctx: *mlir.Context) *const mlir.Attribute {
        var buf: [64]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, "#vector.kind<{s}>", .{@tagName(self)}) catch unreachable;
        return mlir.Attribute.parse(ctx, text) catch
            std.debug.panic("failed to parse CombiningKind '{s}'", .{text});
    }
};

/// Matches `mlir::vector::IteratorType` — used by `vector.contract`.
pub const IteratorType = enum {
    parallel,
    reduction,

    pub fn attribute(self: IteratorType, ctx: *mlir.Context) *const mlir.Attribute {
        const text = switch (self) {
            .parallel => "#vector.iterator_type<parallel>",
            .reduction => "#vector.iterator_type<reduction>",
        };
        return mlir.Attribute.parse(ctx, text) catch
            std.debug.panic("failed to parse IteratorType", .{});
    }
};

// =============================================================================
// Element / lane access — extract / insert / from_elements / to_elements
// =============================================================================

/// vector.extract — pull a single element or sub-vector out of `src`. Static
/// position entries equal to `kPoison` (== `INT64_MIN` per VectorOps.td) come
/// from `dynamic_position`. The result type is provided by the caller.
pub const kPoison: i64 = std.math.minInt(i64);

pub fn extract(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    static_position: []const i64,
    dynamic_position: []const *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    operands_buf.appendAssumeCapacity(src);
    operands_buf.appendSliceAssumeCapacity(dynamic_position);

    return mlir.Operation.make(ctx, "vector.extract", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "static_position", mlir.denseArrayAttribute(ctx, .i64, static_position)),
        },
        .location = location,
    });
}

/// vector.insert — write `value` into `dest` at the given position. The result
/// has the same type as `dest`.
pub fn insert(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    dest: *const mlir.Value,
    static_position: []const i64,
    dynamic_position: []const *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 17) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ value, dest });
    operands_buf.appendSliceAssumeCapacity(dynamic_position);

    return mlir.Operation.make(ctx, "vector.insert", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{dest.type_()} },
        .attributes = &.{
            .named(ctx, "static_position", mlir.denseArrayAttribute(ctx, .i64, static_position)),
        },
        .location = location,
    });
}

/// vector.from_elements — build a vector from an exact-size list of scalars.
pub fn from_elements(
    ctx: *mlir.Context,
    elements: []const *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.from_elements", .{
        .operands = .{ .flat = elements },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// vector.to_elements — destructure a vector into its scalar lanes.
pub fn to_elements(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_types: []const *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.to_elements", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = result_types },
        .location = location,
    });
}

// =============================================================================
// Strided slice — extract / insert with offsets/sizes/strides
// =============================================================================

pub const StridedSliceArgs = struct {
    offsets: []const i64,
    sizes: []const i64,
    strides: []const i64,
};

pub fn extract_strided_slice(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    args: StridedSliceArgs,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.extract_strided_slice", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "offsets", mlir.denseArrayAttribute(ctx, .i64, args.offsets)),
            .named(ctx, "sizes", mlir.denseArrayAttribute(ctx, .i64, args.sizes)),
            .named(ctx, "strides", mlir.denseArrayAttribute(ctx, .i64, args.strides)),
        },
        .location = location,
    });
}

pub fn insert_strided_slice(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    dest: *const mlir.Value,
    offsets: []const i64,
    strides: []const i64,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.insert_strided_slice", .{
        .operands = .{ .flat = &.{ src, dest } },
        .results = .{ .flat = &.{dest.type_()} },
        .attributes = &.{
            .named(ctx, "offsets", mlir.denseArrayAttribute(ctx, .i64, offsets)),
            .named(ctx, "strides", mlir.denseArrayAttribute(ctx, .i64, strides)),
        },
        .location = location,
    });
}

// =============================================================================
// Shape manipulation
// =============================================================================

/// vector.broadcast — broadcast a scalar / lower-rank vector to a wider one.
pub fn broadcast(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.broadcast", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// vector.shape_cast — reshape a vector preserving total number of lanes.
pub fn shape_cast(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.shape_cast", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// vector.bitcast — same total bits, possibly different element type / shape.
pub fn bitcast(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.bitcast", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// vector.type_cast — cast a `memref<vector<...>>` to `memref<...>` (and vice-
/// versa). Used in vector lowering / unrolling.
pub fn type_cast(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.type_cast", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// vector.transpose — permute dims of a vector.
pub fn transpose(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    permutation: []const i64,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.transpose", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "permutation", mlir.denseArrayAttribute(ctx, .i64, permutation)),
        },
        .location = location,
    });
}

// =============================================================================
// Lane shuffle — shuffle / interleave / deinterleave
// =============================================================================

/// vector.shuffle — concat lhs and rhs into a logical vector (rank 1) and
/// pick `mask`-listed lanes to form the result.
pub fn shuffle(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    mask: []const i64,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.shuffle", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "mask", mlir.denseArrayAttribute(ctx, .i64, mask)),
        },
        .location = location,
    });
}

/// vector.interleave — interleave even/odd lanes from lhs/rhs.
pub fn interleave(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.interleave", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// vector.deinterleave — split a vector into (even-lanes, odd-lanes).
pub fn deinterleave(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.deinterleave", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{ result_type, result_type } },
        .location = location,
    });
}

// =============================================================================
// Compute — fma / step / vscale
// =============================================================================

/// vector.fma — element-wise fused multiply-add `lhs * rhs + acc`.
pub fn fma(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    acc: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.fma", .{
        .operands = .{ .flat = &.{ lhs, rhs, acc } },
        .result_type_inference = true,
        .location = location,
    });
}

/// vector.step — produce `<0, 1, ..., N-1>` of the result type.
pub fn step(
    ctx: *mlir.Context,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.step", .{
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// vector.vscale — runtime scalable-vector size factor (index-typed).
pub fn vscale(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.vscale", .{
        .results = .{ .flat = &.{mlir.indexType(ctx)} },
        .location = location,
    });
}

/// vector.yield — terminator for vector.mask / vector.warp regions.
pub fn yield(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.yield", .{
        .operands = .{ .flat = values },
        .verify = false,
        .location = location,
    });
}

// =============================================================================
// Reductions — reduction / multi_reduction / contract
// =============================================================================

pub const ReductionOpts = struct {
    /// Optional accumulator (scalar of result type). When provided the op is
    /// `result = combine(reduce(src), acc)`.
    acc: ?*const mlir.Value = null,
};

/// vector.reduction — reduce a 1-D vector to a scalar with `kind`.
pub fn reduction(
    ctx: *mlir.Context,
    kind: CombiningKind,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    opts: ReductionOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 2) = .empty;
    operands_buf.appendAssumeCapacity(src);
    if (opts.acc) |a| operands_buf.appendAssumeCapacity(a);

    return mlir.Operation.make(ctx, "vector.reduction", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "kind", kind.attribute(ctx)),
        },
        .location = location,
    });
}

/// vector.multi_reduction — reduce a multi-D vector along `reduction_dims`.
pub fn multi_reduction(
    ctx: *mlir.Context,
    kind: CombiningKind,
    src: *const mlir.Value,
    acc: *const mlir.Value,
    reduction_dims: []const i64,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.multi_reduction", .{
        .operands = .{ .flat = &.{ src, acc } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "kind", kind.attribute(ctx)),
            .named(ctx, "reduction_dims", mlir.denseArrayAttribute(ctx, .i64, reduction_dims)),
        },
        .location = location,
    });
}

pub const ContractOpts = struct {
    kind: CombiningKind = .add,
};

/// vector.contract — generalized contraction (matmul, dot, batched matmul,
/// reductions). `indexing_maps` is one AffineMapAttr per (lhs, rhs, acc).
/// `iterator_types` is one IteratorType per dim of the iteration space.
pub fn contract(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    acc: *const mlir.Value,
    indexing_maps: []const *const mlir.Attribute,
    iterator_types: []const IteratorType,
    result_type: *const mlir.Type,
    opts: ContractOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var iter_attrs: stdx.BoundedArray(*const mlir.Attribute, 16) = .empty;
    for (iterator_types) |it| iter_attrs.appendAssumeCapacity(it.attribute(ctx));

    return mlir.Operation.make(ctx, "vector.contract", .{
        .operands = .{ .flat = &.{ lhs, rhs, acc } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "indexing_maps", mlir.arrayAttribute(ctx, indexing_maps)),
            .named(ctx, "iterator_types", mlir.arrayAttribute(ctx, iter_attrs.constSlice())),
            .named(ctx, "kind", opts.kind.attribute(ctx)),
        },
        .location = location,
    });
}

// =============================================================================
// Memory transfers — load / store / masked / gather / scatter / transfer_*
// =============================================================================

pub const LoadStoreOpts = struct {
    nontemporal: bool = false,
    alignment: ?i64 = null,
};

fn loadStoreAttrs(ctx: *mlir.Context, opts: LoadStoreOpts, dst: *stdx.BoundedArray(mlir.NamedAttribute, 2)) void {
    if (opts.nontemporal) {
        dst.appendAssumeCapacity(.named(ctx, "nontemporal", mlir.boolAttribute(ctx, true)));
    }
    if (opts.alignment) |a| {
        dst.appendAssumeCapacity(.named(ctx, "alignment", mlir.integerAttribute(ctx, .i64, a)));
    }
}

/// vector.load — strided contiguous load from `base[indices...]` into the
/// result vector.
pub fn load(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    result_type: *const mlir.Type,
    opts: LoadStoreOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);

    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    loadStoreAttrs(ctx, opts, &attrs);

    return mlir.Operation.make(ctx, "vector.load", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// vector.store — strided contiguous store of `value` to `base[indices...]`.
pub fn store(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    opts: LoadStoreOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 17) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ value, base });
    operands_buf.appendSliceAssumeCapacity(indices);

    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    loadStoreAttrs(ctx, opts, &attrs);

    return mlir.Operation.make(ctx, "vector.store", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// vector.maskedload — like load but lanes whose `mask` bit is 0 are read
/// from `passthru` instead.
pub fn masked_load(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    mask: *const mlir.Value,
    passthru: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 18) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);
    operands_buf.appendSliceAssumeCapacity(&.{ mask, passthru });

    return mlir.Operation.make(ctx, "vector.maskedload", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn masked_store(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    mask: *const mlir.Value,
    value: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 18) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);
    operands_buf.appendSliceAssumeCapacity(&.{ mask, value });

    return mlir.Operation.make(ctx, "vector.maskedstore", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .location = location,
    });
}

/// vector.gather — gather scalar elements from `base[indices...]` plus per-lane
/// `index_vec` offsets.
pub fn gather(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    index_vec: *const mlir.Value,
    mask: *const mlir.Value,
    passthru: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 20) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);
    operands_buf.appendSliceAssumeCapacity(&.{ index_vec, mask, passthru });

    return mlir.Operation.make(ctx, "vector.gather", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn scatter(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    index_vec: *const mlir.Value,
    mask: *const mlir.Value,
    value: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 20) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);
    operands_buf.appendSliceAssumeCapacity(&.{ index_vec, mask, value });

    return mlir.Operation.make(ctx, "vector.scatter", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .location = location,
    });
}

pub fn expand_load(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    mask: *const mlir.Value,
    passthru: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 18) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);
    operands_buf.appendSliceAssumeCapacity(&.{ mask, passthru });

    return mlir.Operation.make(ctx, "vector.expandload", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn compress_store(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    mask: *const mlir.Value,
    value: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 18) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);
    operands_buf.appendSliceAssumeCapacity(&.{ mask, value });

    return mlir.Operation.make(ctx, "vector.compressstore", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .location = location,
    });
}

// =============================================================================
// Transfer ops — vector.transfer_read / vector.transfer_write
// =============================================================================

pub const TransferReadArgs = struct {
    /// `affine_map<(d0,...) -> (...)>` attribute. Build with
    /// `mlir.parseAffineMapAttribute` or `mlir.affineMapAttribute(...)`.
    permutation_map: *const mlir.Attribute,
    /// One bool per result-vector dim. Length must equal vector rank.
    in_bounds: []const bool,
    /// Optional i1 vector mask.
    mask: ?*const mlir.Value = null,
};

/// vector.transfer_read — supervector load with permutation map and optional
/// mask. AttrSizedOperandSegments: (base, indices, padding, [mask]).
pub fn transfer_read(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    padding: *const mlir.Value,
    args: TransferReadArgs,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 20) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);
    operands_buf.appendAssumeCapacity(padding);
    const mask_len: i32 = if (args.mask) |m| blk: {
        operands_buf.appendAssumeCapacity(m);
        break :blk 1;
    } else 0;

    const seg_sizes = [4]i32{ 1, @intCast(indices.len), 1, mask_len };

    var in_bounds_attrs: stdx.BoundedArray(*const mlir.Attribute, 16) = .empty;
    for (args.in_bounds) |b| in_bounds_attrs.appendAssumeCapacity(mlir.boolAttribute(ctx, b));

    return mlir.Operation.make(ctx, "vector.transfer_read", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", mlir.denseArrayAttribute(ctx, .i32, &seg_sizes)),
            .named(ctx, "permutation_map", args.permutation_map),
            .named(ctx, "in_bounds", mlir.arrayAttribute(ctx, in_bounds_attrs.constSlice())),
        },
        .location = location,
    });
}

pub const TransferWriteArgs = struct {
    permutation_map: *const mlir.Attribute,
    in_bounds: []const bool,
    mask: ?*const mlir.Value = null,
    /// When `dest` is a tensor, the op produces a new tensor — pass its type
    /// here. For memref destinations leave null.
    result_type: ?*const mlir.Type = null,
};

/// vector.transfer_write — supervector store with permutation map and optional
/// mask. AttrSizedOperandSegments: (vector, base, indices, [mask]).
pub fn transfer_write(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    args: TransferWriteArgs,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 20) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ value, base });
    operands_buf.appendSliceAssumeCapacity(indices);
    const mask_len: i32 = if (args.mask) |m| blk: {
        operands_buf.appendAssumeCapacity(m);
        break :blk 1;
    } else 0;

    const seg_sizes = [4]i32{ 1, 1, @intCast(indices.len), mask_len };

    var in_bounds_attrs: stdx.BoundedArray(*const mlir.Attribute, 16) = .empty;
    for (args.in_bounds) |b| in_bounds_attrs.appendAssumeCapacity(mlir.boolAttribute(ctx, b));

    var result_types: stdx.BoundedArray(*const mlir.Type, 1) = .empty;
    if (args.result_type) |rt| result_types.appendAssumeCapacity(rt);

    return mlir.Operation.make(ctx, "vector.transfer_write", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = result_types.constSlice() },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", mlir.denseArrayAttribute(ctx, .i32, &seg_sizes)),
            .named(ctx, "permutation_map", args.permutation_map),
            .named(ctx, "in_bounds", mlir.arrayAttribute(ctx, in_bounds_attrs.constSlice())),
        },
        .location = location,
    });
}

// =============================================================================
// Masks — constant_mask / create_mask
// =============================================================================

/// vector.constant_mask — constant boolean mask. Each entry of
/// `mask_dim_sizes` is the number of leading true lanes along that dim.
pub fn constant_mask(
    ctx: *mlir.Context,
    mask_dim_sizes: []const i64,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.constant_mask", .{
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "mask_dim_sizes", mlir.denseArrayAttribute(ctx, .i64, mask_dim_sizes)),
        },
        .location = location,
    });
}

/// vector.create_mask — runtime mask from per-dim count operands. `operands`
/// length must equal the rank of `result_type`.
pub fn create_mask(
    ctx: *mlir.Context,
    operands: []const *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.create_mask", .{
        .operands = .{ .flat = operands },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

test {
    std.testing.refAllDecls(@This());
}
