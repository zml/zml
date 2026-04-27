//! Layer A builders for the upstream `affine` dialect.
//!
//! Upstream MLIR does not ship `mlir-c/Dialect/Affine.h`, so dialect
//! registration goes through a small CAPI shim (`affine_capi.cc`) that
//! exposes `mlirGetDialectHandle__affine__`. The non-dialect-specific
//! affine machinery (`AffineMap`, `IntegerSet`, `AffineExpr`) is part of the
//! core `mlir-c/AffineMap.h` / `mlir-c/IntegerSet.h` headers that ship with
//! `@llvm-project//mlir:CAPIIR` and is wrapped in `mlir/mlir.zig`
//! (`mlir.AffineMap`, `mlir.AffineExpr`, `mlir.parseAffineMapAttribute`, …).
//!
//! AffineMap and IntegerSet attributes can be either built programmatically
//! (`mlir.AffineMap.get(ctx, ...)` and friends) or parsed from text
//! (`mlir.parseAffineMapAttribute(ctx, "affine_map<(d0) -> (d0 + 3)>")`).
//!
//! Intentionally omitted (add on demand): `dma_start` / `dma_wait` (TPU
//! kernels go via the `tpu` dialect's DMAs), `prefetch` (rarely needed in
//! emitted IR).

const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

// =============================================================================
// affine.apply / affine.min / affine.max
// =============================================================================

/// affine.apply — apply a 1-result affine map to a list of dim+symbol
/// operands. Result is `index`.
pub fn apply(
    ctx: *mlir.Context,
    map: *const mlir.AffineMap,
    map_operands: []const *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "affine.apply", .{
        .operands = .{ .flat = map_operands },
        .results = .{ .flat = &.{mlir.indexType(ctx)} },
        .attributes = &.{
            .named(ctx, "map", mlir.affineMapAttribute(map)),
        },
        .location = location,
    });
}

/// affine.min — minimum of all results of `map` evaluated at `map_operands`.
pub fn min(
    ctx: *mlir.Context,
    map: *const mlir.AffineMap,
    map_operands: []const *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "affine.min", .{
        .operands = .{ .flat = map_operands },
        .results = .{ .flat = &.{mlir.indexType(ctx)} },
        .attributes = &.{
            .named(ctx, "map", mlir.affineMapAttribute(map)),
        },
        .location = location,
    });
}

/// affine.max — maximum of all results of `map`.
pub fn max(
    ctx: *mlir.Context,
    map: *const mlir.AffineMap,
    map_operands: []const *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "affine.max", .{
        .operands = .{ .flat = map_operands },
        .results = .{ .flat = &.{mlir.indexType(ctx)} },
        .attributes = &.{
            .named(ctx, "map", mlir.affineMapAttribute(map)),
        },
        .location = location,
    });
}

// =============================================================================
// affine.for / affine.yield
// =============================================================================

pub const ForArgs = struct {
    /// Constant lower bound. Pass `null` when supplying a `lower_bound_map`.
    lower_bound: ?i64 = null,
    /// Constant upper bound. Pass `null` when supplying an `upper_bound_map`.
    upper_bound: ?i64 = null,
    /// Optional map-based bounds. When non-null they take precedence over the
    /// constant counterpart.
    lower_bound_map: ?*const mlir.AffineMap = null,
    upper_bound_map: ?*const mlir.AffineMap = null,
    /// Operands feeding `lower_bound_map` (in dim+symbol order).
    lower_bound_operands: []const *const mlir.Value = &.{},
    /// Operands feeding `upper_bound_map`.
    upper_bound_operands: []const *const mlir.Value = &.{},
    /// Loop step (positive, defaults to 1).
    step: i64 = 1,
    /// Loop-carried initial values.
    inits: []const *const mlir.Value = &.{},
};

/// affine.for — `for %i = lower to upper step S iter_args(...)`. The body
/// block takes `index, *init_arg_types` and must terminate with `affine.yield`.
pub fn for_(
    ctx: *mlir.Context,
    args: ForArgs,
    body: *mlir.Block,
    location: *const mlir.Location,
) *mlir.Operation {
    // Resolve final maps.
    const lb_map = args.lower_bound_map orelse blk: {
        const cst = mlir.AffineExpr.constant(ctx, args.lower_bound orelse 0);
        break :blk mlir.AffineMap.get(ctx, 0, 0, &.{cst});
    };
    const ub_map = args.upper_bound_map orelse blk: {
        const cst = mlir.AffineExpr.constant(ctx, args.upper_bound orelse 0);
        break :blk mlir.AffineMap.get(ctx, 0, 0, &.{cst});
    };

    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendSliceAssumeCapacity(args.lower_bound_operands);
    operands.appendSliceAssumeCapacity(args.upper_bound_operands);
    operands.appendSliceAssumeCapacity(args.inits);

    const seg_sizes = [3]i32{
        @intCast(args.lower_bound_operands.len),
        @intCast(args.upper_bound_operands.len),
        @intCast(args.inits.len),
    };

    var result_types: stdx.BoundedArray(*const mlir.Type, 64) = .empty;
    for (args.inits) |v| result_types.appendAssumeCapacity(v.type_());

    return mlir.Operation.make(ctx, "affine.for", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = result_types.constSlice() },
        .blocks = &.{body},
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", mlir.denseArrayAttribute(ctx, .i32, &seg_sizes)),
            .named(ctx, "lowerBoundMap", mlir.affineMapAttribute(lb_map)),
            .named(ctx, "upperBoundMap", mlir.affineMapAttribute(ub_map)),
            .named(ctx, "step", mlir.integerAttribute(ctx, .i64, args.step)),
        },
        // The for body may reference outer SSA values that only become
        // reachable once it's appended to its parent — defer to module-level
        // verify (same trick used by scf.for).
        .verify = false,
        .location = location,
    });
}

/// affine.yield — terminator for affine.for / affine.if / affine.parallel.
pub fn yield(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "affine.yield", .{
        .operands = .{ .flat = values },
        .verify = false,
        .location = location,
    });
}

// =============================================================================
// affine.if
// =============================================================================

/// affine.if — `if condition[ ... ] { then } else { else }`. `condition` is an
/// IntegerSetAttribute (a conjunction of affine constraints). The result types
/// are inferred from the yielded values; pass `&.{}` for the no-result form.
pub fn if_(
    ctx: *mlir.Context,
    condition: *const mlir.IntegerSet,
    operands: []const *const mlir.Value,
    result_types: []const *const mlir.Type,
    then_block: *mlir.Block,
    else_block: ?*mlir.Block,
    location: *const mlir.Location,
) *mlir.Operation {
    var state: mlir.OperationState = .init("affine.if", location);
    state.addOperands(operands);
    state.addResults(result_types);
    state.addAttributes(&.{
        .named(ctx, "condition", mlir.integerSetAttribute(condition)),
    });

    const then_region = mlir.Region.init();
    then_region.appendOwnedBlock(then_block);
    const else_region = mlir.Region.init();
    if (else_block) |b| else_region.appendOwnedBlock(b);
    state.addOwnedRegions(&.{ then_region, else_region });

    return mlir.Operation.init(&state) catch @panic("Failed to create affine.if");
}

// =============================================================================
// affine.load / affine.store / affine.vector_load / affine.vector_store
// =============================================================================

/// affine.load — load with affine indexing.
pub fn load(
    ctx: *mlir.Context,
    memref: *const mlir.Value,
    map: *const mlir.AffineMap,
    map_operands: []const *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    operands_buf.appendAssumeCapacity(memref);
    operands_buf.appendSliceAssumeCapacity(map_operands);

    return mlir.Operation.make(ctx, "affine.load", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "map", mlir.affineMapAttribute(map)),
        },
        .location = location,
    });
}

pub fn store(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    memref: *const mlir.Value,
    map: *const mlir.AffineMap,
    map_operands: []const *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 17) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ value, memref });
    operands_buf.appendSliceAssumeCapacity(map_operands);

    return mlir.Operation.make(ctx, "affine.store", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = &.{
            .named(ctx, "map", mlir.affineMapAttribute(map)),
        },
        .location = location,
    });
}

pub fn vector_load(
    ctx: *mlir.Context,
    memref: *const mlir.Value,
    map: *const mlir.AffineMap,
    map_operands: []const *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    operands_buf.appendAssumeCapacity(memref);
    operands_buf.appendSliceAssumeCapacity(map_operands);

    return mlir.Operation.make(ctx, "affine.vector_load", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "map", mlir.affineMapAttribute(map)),
        },
        .location = location,
    });
}

pub fn vector_store(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    memref: *const mlir.Value,
    map: *const mlir.AffineMap,
    map_operands: []const *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 17) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ value, memref });
    operands_buf.appendSliceAssumeCapacity(map_operands);

    return mlir.Operation.make(ctx, "affine.vector_store", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = &.{
            .named(ctx, "map", mlir.affineMapAttribute(map)),
        },
        .location = location,
    });
}

// =============================================================================
// affine.parallel
// =============================================================================

pub const ParallelArgs = struct {
    /// Per-loop-dim reductions, one entry per result. Use the same kinds as
    /// `mlir.dialects.memref.AtomicRMWKind` (the dialect uses
    /// `arith::AtomicRMWKind`). Pass `&.{}` for parallel loops with no
    /// loop-carried reductions.
    reductions: []const i64 = &.{},
    /// Map producing all lower-bound values; a single dim's bound is the
    /// max of the results in its `lower_bounds_groups[i]` group.
    lower_bounds_map: *const mlir.AffineMap,
    /// One i32 per loop dim — number of map results that contribute to the
    /// lower bound of that dim. Sum must equal `lower_bounds_map.numResults()`.
    lower_bounds_groups: []const i32,
    upper_bounds_map: *const mlir.AffineMap,
    upper_bounds_groups: []const i32,
    /// Per-dim integer steps (default 1).
    steps: []const i64,
    /// Operands feeding both bound maps.
    map_operands: []const *const mlir.Value = &.{},
};

/// affine.parallel — multi-dim parallel band. `body_block` takes one `index`
/// argument per loop dim and must terminate with `affine.yield`.
pub fn parallel(
    ctx: *mlir.Context,
    args: ParallelArgs,
    body_block: *mlir.Block,
    result_types: []const *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    // I32ElementsAttr — dense<[...]> : tensor<NxI32>.
    const lb_groups_ty = mlir.RankedTensorType.get(
        &.{@intCast(args.lower_bounds_groups.len)},
        mlir.integerType(ctx, .i32),
        null,
    ).shaped();
    const ub_groups_ty = mlir.RankedTensorType.get(
        &.{@intCast(args.upper_bounds_groups.len)},
        mlir.integerType(ctx, .i32),
        null,
    ).shaped();

    // I64SmallVectorArrayAttr — built as ArrayAttr<IntegerAttr<I64>>.
    var steps_attrs: stdx.BoundedArray(*const mlir.Attribute, 16) = .empty;
    for (args.steps) |s| {
        steps_attrs.appendAssumeCapacity(mlir.integerAttribute(ctx, .i64, s));
    }

    // Reductions are AtomicRMWKindAttr (I64EnumAttr). We let callers pass
    // raw int values to avoid coupling against the enum table.
    var reductions_attrs: stdx.BoundedArray(*const mlir.Attribute, 16) = .empty;
    for (args.reductions) |k| {
        reductions_attrs.appendAssumeCapacity(mlir.integerAttribute(ctx, .i64, k));
    }

    return mlir.Operation.make(ctx, "affine.parallel", .{
        .operands = .{ .flat = args.map_operands },
        .results = .{ .flat = result_types },
        .blocks = &.{body_block},
        .attributes = &.{
            .named(ctx, "reductions", mlir.arrayAttribute(ctx, reductions_attrs.constSlice())),
            .named(ctx, "lowerBoundsMap", mlir.affineMapAttribute(args.lower_bounds_map)),
            .named(ctx, "lowerBoundsGroups", mlir.denseElementsAttribute(lb_groups_ty, args.lower_bounds_groups)),
            .named(ctx, "upperBoundsMap", mlir.affineMapAttribute(args.upper_bounds_map)),
            .named(ctx, "upperBoundsGroups", mlir.denseElementsAttribute(ub_groups_ty, args.upper_bounds_groups)),
            .named(ctx, "steps", mlir.arrayAttribute(ctx, steps_attrs.constSlice())),
        },
        .verify = false,
        .location = location,
    });
}

// =============================================================================
// affine.delinearize_index / affine.linearize_index
// =============================================================================

pub const DelinearizeArgs = struct {
    static_basis: []const i64,
    dynamic_basis: []const *const mlir.Value = &.{},
};

pub fn delinearize_index(
    ctx: *mlir.Context,
    linear_index: *const mlir.Value,
    args: DelinearizeArgs,
    result_types: []const *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    operands_buf.appendAssumeCapacity(linear_index);
    operands_buf.appendSliceAssumeCapacity(args.dynamic_basis);

    return mlir.Operation.make(ctx, "affine.delinearize_index", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = result_types },
        .attributes = &.{
            .named(ctx, "static_basis", mlir.denseArrayAttribute(ctx, .i64, args.static_basis)),
        },
        .location = location,
    });
}

pub const LinearizeArgs = struct {
    multi_index: []const *const mlir.Value,
    static_basis: []const i64,
    dynamic_basis: []const *const mlir.Value = &.{},
    disjoint: bool = false,
};

pub fn linearize_index(
    ctx: *mlir.Context,
    args: LinearizeArgs,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 32) = .empty;
    operands_buf.appendSliceAssumeCapacity(args.multi_index);
    operands_buf.appendSliceAssumeCapacity(args.dynamic_basis);

    const seg_sizes = [2]i32{
        @intCast(args.multi_index.len),
        @intCast(args.dynamic_basis.len),
    };

    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attrs.appendSliceAssumeCapacity(&.{
        .named(ctx, "operandSegmentSizes", mlir.denseArrayAttribute(ctx, .i32, &seg_sizes)),
        .named(ctx, "static_basis", mlir.denseArrayAttribute(ctx, .i64, args.static_basis)),
    });
    if (args.disjoint) {
        attrs.appendAssumeCapacity(.named(ctx, "disjoint", mlir.unitAttribute(ctx)));
    }

    return mlir.Operation.make(ctx, "affine.linearize_index", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

test {
    std.testing.refAllDecls(@This());
}
