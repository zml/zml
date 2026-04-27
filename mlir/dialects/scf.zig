//! Layer A builders for the `scf` dialect.
//!
//! `scf.for` exposes `unsignedCmp` via `ForOpts`; all other optional attrs
//! (none currently required for TTIR emission) default to unset and are added
//! on demand.

const std = @import("std");

const mlir = @import("mlir");
const stdx = @import("stdx");

pub const ForOpts = struct {
    /// Per td: `scf.for` uses signed comparison by default. Setting this emits
    /// the `unsignedCmp` UnitAttr so the induction variable advances with
    /// unsigned comparison instead.
    unsigned_cmp: bool = false,
};

/// scf.for — operands are (lower, upper, step, init_args...) as a flat list.
/// Result arity equals init_args arity; result types must match init_args types.
///
/// The body block must have arguments `[iv_type, *init_arg_types]` and be
/// terminated with `scf.yield`.
pub fn for_(
    ctx: *mlir.Context,
    lower: *const mlir.Value,
    upper: *const mlir.Value,
    step: *const mlir.Value,
    init_args: []const *const mlir.Value,
    body: *mlir.Block,
    opts: ForOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendSliceAssumeCapacity(&.{ lower, upper, step });
    operands.appendSliceAssumeCapacity(init_args);

    var result_types: stdx.BoundedArray(*const mlir.Type, 64) = .empty;
    for (init_args) |v| result_types.appendAssumeCapacity(v.type_());

    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.unsigned_cmp) {
        attrs.appendAssumeCapacity(.named(ctx, "unsignedCmp", mlir.unitAttribute(ctx)));
    }

    return mlir.Operation.make(ctx, "scf.for", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = result_types.constSlice() },
        .blocks = &.{body},
        .attributes = attrs.constSlice(),
        // Per-op verify fails with "operand #0 does not dominate this use" when
        // the scf.for references outer SSA values that only become reachable
        // once it's appended to the parent block. Defer to module-level verify.
        .verify = false,
        .location = location,
    });
}

/// scf.if — cond is i1; then/else blocks must be terminated with `scf.yield`
/// (or implicitly for no-result forms).
///
/// Built via raw OperationState because it has two separate regions.
pub fn if_(
    ctx: *mlir.Context,
    cond: *const mlir.Value,
    result_types: []const *const mlir.Type,
    then_block: *mlir.Block,
    else_block: ?*mlir.Block,
    location: *const mlir.Location,
) *mlir.Operation {
    _ = ctx;
    var state: mlir.OperationState = .init("scf.if", location);
    state.addOperands(&.{cond});
    state.addResults(result_types);

    const then_region = mlir.Region.init();
    then_region.appendOwnedBlock(then_block);
    const else_region = mlir.Region.init();
    if (else_block) |b| else_region.appendOwnedBlock(b);
    state.addOwnedRegions(&.{ then_region, else_region });

    return mlir.Operation.init(&state) catch @panic("Failed to create scf.if");
}

/// scf.yield — variadic result terminator for scf.for / scf.if regions.
pub fn yield(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "scf.yield", .{
        .operands = .{ .flat = values },
        .verify = false,
        .location = location,
    });
}

/// scf.while — two regions. The `before` region computes the condition and
/// terminates with `scf.condition`. The `after` region runs when the condition
/// is true and terminates with `scf.yield`. The arguments of `before` have the
/// types of `inits`. The trailing operands of `scf.condition` feed both the
/// `after` region's arguments and the `scf.while` op's results.
pub fn while_(
    ctx: *mlir.Context,
    inits: []const *const mlir.Value,
    result_types: []const *const mlir.Type,
    before_block: *mlir.Block,
    after_block: *mlir.Block,
    location: *const mlir.Location,
) *mlir.Operation {
    _ = ctx;
    var state: mlir.OperationState = .init("scf.while", location);
    state.addOperands(inits);
    state.addResults(result_types);

    const before_region = mlir.Region.init();
    before_region.appendOwnedBlock(before_block);
    const after_region = mlir.Region.init();
    after_region.appendOwnedBlock(after_block);
    state.addOwnedRegions(&.{ before_region, after_region });

    return mlir.Operation.init(&state) catch @panic("Failed to create scf.while");
}

/// scf.condition — terminator of the `before` region of scf.while.
/// First operand is i1 (the continuation condition). The remaining operands
/// are forwarded to either the `after` region's arguments (if true) or the
/// scf.while's results (if false).
pub fn condition(
    ctx: *mlir.Context,
    cond: *const mlir.Value,
    args: []const *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    var buf: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    buf.appendAssumeCapacity(cond);
    buf.appendSliceAssumeCapacity(args);
    return mlir.Operation.make(ctx, "scf.condition", .{
        .operands = .{ .flat = buf.constSlice() },
        .verify = false,
        .location = location,
    });
}

pub const ExecuteRegionOpts = struct {
    no_inline: bool = false,
};

/// scf.execute_region — runs `body` exactly once, returns the yielded values.
/// `body` has no arguments but may span multiple blocks (unlike for/if).
pub fn execute_region(
    ctx: *mlir.Context,
    result_types: []const *const mlir.Type,
    body: *mlir.Block,
    opts: ExecuteRegionOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (opts.no_inline) {
        attrs.appendAssumeCapacity(.named(ctx, "no_inline", mlir.unitAttribute(ctx)));
    }
    return mlir.Operation.make(ctx, "scf.execute_region", .{
        .results = .{ .flat = result_types },
        .blocks = &.{body},
        .attributes = attrs.constSlice(),
        // execute_region has its own region-verify; defer to the module-level
        // verify so unresolved SSA uses don't block construction.
        .verify = false,
        .location = location,
    });
}

/// scf.parallel — multi-dim parallel for loop with optional reductions.
/// Operands are (lbs, ubs, steps, inits). `body_block` takes one `index` arg
/// per loop dim and must be terminated with `scf.reduce` (or no-op if inits is
/// empty — the parser auto-inserts an empty reduce).
pub fn parallel(
    ctx: *mlir.Context,
    lbs: []const *const mlir.Value,
    ubs: []const *const mlir.Value,
    steps: []const *const mlir.Value,
    inits: []const *const mlir.Value,
    body_block: *mlir.Block,
    result_types: []const *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    std.debug.assert(lbs.len == ubs.len);
    std.debug.assert(lbs.len == steps.len);

    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendSliceAssumeCapacity(lbs);
    operands.appendSliceAssumeCapacity(ubs);
    operands.appendSliceAssumeCapacity(steps);
    operands.appendSliceAssumeCapacity(inits);

    const seg_sizes = [4]i32{
        @intCast(lbs.len),
        @intCast(ubs.len),
        @intCast(steps.len),
        @intCast(inits.len),
    };
    return mlir.Operation.make(ctx, "scf.parallel", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = result_types },
        .blocks = &.{body_block},
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", mlir.denseArrayAttribute(ctx, .i32, &seg_sizes)),
        },
        .verify = false,
        .location = location,
    });
}

/// scf.reduce — terminator of `scf.parallel`. Each region is a reduction body
/// taking two args (lhs, rhs) matching the corresponding operand's type,
/// terminated with `scf.reduce.return`.
pub fn reduce(
    ctx: *mlir.Context,
    operands: []const *const mlir.Value,
    reduction_blocks: []const *mlir.Block,
    location: *const mlir.Location,
) *mlir.Operation {
    _ = ctx;
    std.debug.assert(operands.len == reduction_blocks.len);
    var state: mlir.OperationState = .init("scf.reduce", location);
    state.addOperands(operands);

    var regions: stdx.BoundedArray(*mlir.Region, 16) = .empty;
    for (reduction_blocks) |b| {
        const r = mlir.Region.init();
        r.appendOwnedBlock(b);
        regions.appendAssumeCapacity(r);
    }
    state.addOwnedRegions(regions.constSlice());

    return mlir.Operation.init(&state) catch @panic("Failed to create scf.reduce");
}

pub fn reduce_return(ctx: *mlir.Context, value: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "scf.reduce.return", .{
        .operands = .{ .flat = &.{value} },
        .verify = false,
        .location = location,
    });
}

/// scf.forall — multi-dim parallel region, similar to `scf.parallel` but using
/// tensor-based `shared_outs` instead of init values. This wrapper builds the
/// fully-dynamic form (all lb/ub/step are runtime values).
///
/// `body_block` must have `rank + shared_outs.len` arguments (one index per
/// loop dim, plus one per shared_out) and must be terminated with
/// `scf.forall.in_parallel`.
pub fn forall(
    ctx: *mlir.Context,
    lbs: []const *const mlir.Value,
    ubs: []const *const mlir.Value,
    steps: []const *const mlir.Value,
    shared_outs: []const *const mlir.Value,
    mapping: ?*const mlir.Attribute,
    body_block: *mlir.Block,
    result_types: []const *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    std.debug.assert(lbs.len == ubs.len);
    std.debug.assert(lbs.len == steps.len);

    var operands: stdx.BoundedArray(*const mlir.Value, 64) = .empty;
    operands.appendSliceAssumeCapacity(lbs);
    operands.appendSliceAssumeCapacity(ubs);
    operands.appendSliceAssumeCapacity(steps);
    operands.appendSliceAssumeCapacity(shared_outs);

    const seg_sizes = [4]i32{
        @intCast(lbs.len),
        @intCast(ubs.len),
        @intCast(steps.len),
        @intCast(shared_outs.len),
    };

    // Fully-dynamic form: the static arrays use the kDynamic sentinel
    // (mlir::ShapedType::kDynamic == INT64_MIN). We build one slot per loop
    // dim, all dynamic. This matches how MLIR printers emit
    // `scf.forall (%iv) = (%lb) to (%ub) step (%step)` when no static
    // constants are folded in.
    const k_dynamic: i64 = std.math.minInt(i64);
    var sentinel: stdx.BoundedArray(i64, 16) = .empty;
    for (0..lbs.len) |_| sentinel.appendAssumeCapacity(k_dynamic);
    const sentinel_slice = sentinel.constSlice();

    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attrs.appendSliceAssumeCapacity(&.{
        .named(ctx, "operandSegmentSizes", mlir.denseArrayAttribute(ctx, .i32, &seg_sizes)),
        .named(ctx, "staticLowerBound", mlir.denseArrayAttribute(ctx, .i64, sentinel_slice)),
        .named(ctx, "staticUpperBound", mlir.denseArrayAttribute(ctx, .i64, sentinel_slice)),
        .named(ctx, "staticStep", mlir.denseArrayAttribute(ctx, .i64, sentinel_slice)),
    });
    if (mapping) |m| attrs.appendAssumeCapacity(.named(ctx, "mapping", m));

    return mlir.Operation.make(ctx, "scf.forall", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = result_types },
        .blocks = &.{body_block},
        .attributes = attrs.constSlice(),
        .verify = false,
        .location = location,
    });
}

/// scf.forall.in_parallel — terminator for scf.forall. The inner region is
/// a single block containing a flat list of ops (typically
/// tensor.parallel_insert_slice). `body_block` must be built by the caller.
pub fn forall_in_parallel(
    ctx: *mlir.Context,
    body_block: *mlir.Block,
    location: *const mlir.Location,
) *mlir.Operation {
    _ = ctx;
    var state: mlir.OperationState = .init("scf.forall.in_parallel", location);
    const region = mlir.Region.init();
    region.appendOwnedBlock(body_block);
    state.addOwnedRegions(&.{region});
    return mlir.Operation.init(&state) catch @panic("Failed to create scf.forall.in_parallel");
}

/// scf.index_switch — branches to case_blocks[i] if arg == cases[i], else
/// falls through to default_block. Blocks must be terminated with scf.yield.
/// The op's results have the types yielded by each region.
pub fn index_switch(
    ctx: *mlir.Context,
    arg: *const mlir.Value,
    cases: []const i64,
    default_block: *mlir.Block,
    case_blocks: []const *mlir.Block,
    result_types: []const *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    std.debug.assert(cases.len == case_blocks.len);
    var state: mlir.OperationState = .init("scf.index_switch", location);
    state.addOperands(&.{arg});
    state.addResults(result_types);
    state.addAttributes(&.{
        .named(ctx, "cases", mlir.denseArrayAttribute(ctx, .i64, cases)),
    });

    // First region is the default, then one per case — this is how
    // `VariadicRegion<SizedRegion<1>>:$caseRegions` is positioned in the td.
    const default_region = mlir.Region.init();
    default_region.appendOwnedBlock(default_block);

    var regions: stdx.BoundedArray(*mlir.Region, 64) = .empty;
    regions.appendAssumeCapacity(default_region);
    for (case_blocks) |b| {
        const r = mlir.Region.init();
        r.appendOwnedBlock(b);
        regions.appendAssumeCapacity(r);
    }
    state.addOwnedRegions(regions.constSlice());

    return mlir.Operation.init(&state) catch @panic("Failed to create scf.index_switch");
}

test {
    std.testing.refAllDecls(@This());
}
