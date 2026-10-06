const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

/// Mosaic IR version stored as `stable_mosaic.version`, matching
/// `xla/mosaic/dialect/tpu/transforms/serde.h:kVersion`. This is distinct from
/// the custom-call JSON `serialization_format`, which remains 1.
pub const SERDE_VERSION: i32 = 18;

/// Register the `mosaic-serde` pass so `PassManager.parse` can pick it up by
/// name. Idempotent — safe to call repeatedly. Required before running the
/// `mosaic-serde{serialize=true}` pipeline that produces the bytecode embedded
/// in `tpu_custom_call`'s backend_config.
pub fn registerMosaicSerdePass() void {
    c.mlirTpuRegisterMosaicSerdePass();
}

// =============================================================================
// Types — !tpu.semaphore, !tpu.dma_semaphore, !tpu.float8_exmy<...>
// =============================================================================

/// `!tpu.semaphore` element type.
pub fn semaphoreType(ctx: *mlir.Context) *const mlir.Type {
    return mlir.Type.parse(ctx, "!tpu.semaphore") catch
        @panic("failed to parse !tpu.semaphore — is the tpu dialect registered?");
}

/// Build `!tpu.dma_semaphore`.
pub fn dmaSemaphoreType(ctx: *mlir.Context) *const mlir.Type {
    return mlir.Type.parse(ctx, "!tpu.dma_semaphore") catch
        @panic("failed to parse !tpu.dma_semaphore — is the tpu dialect registered?");
}

/// `!tpu.float8_exmy<UnderlyingType>` — Mosaic Float8 type.
pub const Float8EXMYType = opaque {
    const M = mlir.Methods(Float8EXMYType, c.MlirType);

    pub const isAFn = c.mlirTpuIsAFloat8EXMYType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    pub fn get(ctx: *mlir.Context, underlying: *const mlir.Type) *const Float8EXMYType {
        return @ptrCast(c.mlirTpuFloat8EXMYTypeGet(ctx.ptr(), underlying.ptr()).ptr);
    }

    pub fn underlyingType(self: *const Float8EXMYType) *const mlir.Type {
        return @ptrCast(c.mlirTpuFloat8EXMYTypeGetUnderlyingType(self.ptr()).ptr);
    }
};

pub fn float8ExmyType(ctx: *mlir.Context, underlying: *const mlir.Type) *const mlir.Type {
    return @ptrCast(Float8EXMYType.get(ctx, underlying));
}

// =============================================================================
// Enum / dialect-attribute helpers — parsed from their textual form.
// =============================================================================

/// Reduction kind for `tpu.all_reduce`, `tpu.reduce_index`, `tpu.scan`.
pub const ReductionKind = enum {
    sum,
    max,
    min,
    maxf,
    minf,
    maxsi,
    minsi,
    maxui,
    minui,
    arg_max,
    arg_min,
    find_first_set,
    @"and",
    @"or",
    xor,

    pub fn attribute(self: ReductionKind, ctx: *mlir.Context) *const mlir.Attribute {
        var buf: [64]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, "#tpu.reduction_kind<{s}>", .{@tagName(self)}) catch unreachable;
        return mlir.Attribute.parse(ctx, text) catch
            std.debug.panic("failed to parse tpu.reduction_kind '{s}'", .{text});
    }

    fn forInput(self: ReductionKind, input: *const mlir.Value, unsigned_integer: bool) ReductionKind {
        if (self != .max and self != .min) return self;
        const element_type = input.type_().isA(mlir.ShapedType).?.elementType();
        if (element_type.isFloat()) return if (self == .max) .maxf else .minf;
        // Match Mosaic's serde upgrade: scans used unsigned integer extrema,
        // while all-reduce used signed integer extrema.
        return if (unsigned_integer)
            (if (self == .max) .maxui else .minui)
        else
            (if (self == .max) .maxsi else .minsi);
    }
};

/// `tpu.contract_precision` — matmul precision setting.
pub const ContractPrecision = enum {
    bf16,
    fp32,

    pub fn attribute(self: ContractPrecision, ctx: *mlir.Context) *const mlir.Attribute {
        const text = switch (self) {
            .bf16 => "#tpu.contract_precision<bf16>",
            .fp32 => "#tpu.contract_precision<fp32>",
        };
        return mlir.Attribute.parse(ctx, text) catch @panic("contract_precision parse");
    }
};

/// `tpu.rounding_mode` — used by tpu.fptosi / tpu.fptoui / etc.
pub const RoundingMode = enum {
    to_nearest_even,
    to_nearest_away,
    upward,
    downward,
    toward_zero,

    pub fn attribute(self: RoundingMode, ctx: *mlir.Context) *const mlir.Attribute {
        var buf: [64]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, "#tpu.rounding_mode<{s}>", .{@tagName(self)}) catch unreachable;
        return mlir.Attribute.parse(ctx, text) catch
            std.debug.panic("failed to parse tpu.rounding_mode '{s}'", .{text});
    }
};

/// `#tpu.memory_space<...>` — passed as the `memory_space` of a memref's type.
pub const MemorySpace = enum {
    vmem,
    smem,
    cmem,
    hbm,
    semaphore_mem,
    vmem_shared,
    any,

    pub fn attribute(self: MemorySpace, ctx: *mlir.Context) *const mlir.Attribute {
        var buf: [64]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, "#tpu.memory_space<{s}>", .{@tagName(self)}) catch unreachable;
        return mlir.Attribute.parse(ctx, text) catch
            std.debug.panic("failed to parse tpu.memory_space '{s}'", .{text});
    }
};

/// `#tpu.core_type<...>` — used by remote DMAs / sem_signal target hints.
pub const CoreType = enum {
    tc,
    sc_scalar_subcore,
    sc_vector_subcore,

    pub fn attribute(self: CoreType, ctx: *mlir.Context) *const mlir.Attribute {
        var buf: [64]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, "#tpu.core_type<{s}>", .{@tagName(self)}) catch unreachable;
        return mlir.Attribute.parse(ctx, text) catch
            std.debug.panic("failed to parse tpu.core_type '{s}'", .{text});
    }
};

/// `#tpu.dimension_semantics<...>` — grid dimension semantics annotation.
pub const DimensionSemantics = enum {
    parallel,
    arbitrary,
    sequential,

    pub fn attribute(self: DimensionSemantics, ctx: *mlir.Context) *const mlir.Attribute {
        var buf: [64]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, "#tpu.dimension_semantics<{s}>", .{@tagName(self)}) catch unreachable;
        return mlir.Attribute.parse(ctx, text) catch
            std.debug.panic("failed to parse tpu.dimension_semantics '{s}'", .{text});
    }
};

/// `#tpu.pipeline_mode<...>` — double-buffering mode in `window_params`.
pub const PipelineMode = enum {
    synchronous,
    double_buffered,

    pub fn attribute(self: PipelineMode, ctx: *mlir.Context) *const mlir.Attribute {
        var buf: [64]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, "#tpu.pipeline_mode<{s}>", .{@tagName(self)}) catch unreachable;
        return mlir.Attribute.parse(ctx, text) catch
            std.debug.panic("failed to parse tpu.pipeline_mode '{s}'", .{text});
    }
};

/// `#tpu.revisit_mode<...>` — controls block revisit for multi-pass outputs.
pub const RevisitMode = enum {
    immediate,
    any,

    pub fn attribute(self: RevisitMode, ctx: *mlir.Context) *const mlir.Attribute {
        var buf: [64]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, "#tpu.revisit_mode<{s}>", .{@tagName(self)}) catch unreachable;
        return mlir.Attribute.parse(ctx, text) catch
            std.debug.panic("failed to parse tpu.revisit_mode '{s}'", .{text});
    }
};

/// `#tpu.element_window<[pad_low], [pad_high]>` — per-arg window padding.
pub fn elementWindowAttribute(
    ctx: *mlir.Context,
    pad_low: []const i64,
    pad_high: []const i64,
) *const mlir.Attribute {
    var buf: [256]u8 = undefined;
    var len: usize = 0;
    len += (std.fmt.bufPrint(buf[len..], "#tpu.element_window<[", .{}) catch unreachable).len;
    for (pad_low, 0..) |p, i| {
        const sep = if (i == 0) "" else ", ";
        len += (std.fmt.bufPrint(buf[len..], "{s}{d}", .{ sep, p }) catch unreachable).len;
    }
    len += (std.fmt.bufPrint(buf[len..], "], [", .{}) catch unreachable).len;
    for (pad_high, 0..) |p, i| {
        const sep = if (i == 0) "" else ", ";
        len += (std.fmt.bufPrint(buf[len..], "{s}{d}", .{ sep, p }) catch unreachable).len;
    }
    len += (std.fmt.bufPrint(buf[len..], "]>", .{}) catch unreachable).len;
    const text = buf[0..len];
    return mlir.Attribute.parse(ctx, text) catch
        std.debug.panic("failed to parse tpu.element_window '{s}'", .{text});
}

// =============================================================================
// Reductions / scan / sort
// =============================================================================

pub fn all_reduce(
    ctx: *mlir.Context,
    input: *const mlir.Value,
    dim: i64,
    kind: ReductionKind,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.all_reduce", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "dim", .int(ctx, .i64, dim)),
            .named(ctx, "kind", kind.forInput(input, false).attribute(ctx)),
        },
        .location = location,
    });
}

pub fn reduce_index(
    ctx: *mlir.Context,
    input: *const mlir.Value,
    axis: i32,
    kind: ReductionKind,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.reduce_index", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "axis", .int(ctx, .i32, axis)),
            .named(ctx, "kind", kind.attribute(ctx)),
        },
        .location = location,
    });
}

/// Scan the last dimension; an optional rank-one mask spans that dimension.
pub fn scan(
    ctx: *mlir.Context,
    input: *const mlir.Value,
    kind: ReductionKind,
    mask: ?*const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 2) = .empty;
    operands_buf.appendAssumeCapacity(input);
    if (mask) |m| operands_buf.appendAssumeCapacity(m);

    return mlir.Operation.make(ctx, "tpu.scan", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "kind", kind.forInput(input, true).attribute(ctx)),
            .named(ctx, "dimension", .int(ctx, .i64, @as(i64, @intCast(input.type_().isA(mlir.ShapedType).?.rank())) - 1)),
        },
        .location = location,
    });
}

pub const SortOpts = struct {
    descending: bool = false,
};

pub fn sort(
    ctx: *mlir.Context,
    keys: *const mlir.Value,
    values: *const mlir.Value,
    mask: ?*const mlir.Value,
    opts: SortOpts,
    output_mask_type: *const mlir.Type,
    sorted_keys_type: *const mlir.Type,
    sorted_values_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 3) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ keys, values });
    if (mask) |m| operands_buf.appendAssumeCapacity(m);

    return mlir.Operation.make(ctx, "tpu.sort", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{ output_mask_type, sorted_keys_type, sorted_values_type } },
        .attributes = &.{
            .named(ctx, "descending", .boolean(ctx, opts.descending)),
        },
        .location = location,
    });
}

// =============================================================================
// Memory — load / store / vector_load / vector_store / strided_*
// =============================================================================

pub const LoadOpts = struct {
    sublane_mask: []const bool,
    sublane_stride: i32 = 1,
};

/// tpu.load — sublane-aware load into a vreg.
pub fn load(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    opts: LoadOpts,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);

    return mlir.Operation.make(ctx, "tpu.load", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "sublane_mask", denseBoolArrayAttribute(ctx, opts.sublane_mask)),
            .named(ctx, "sublane_stride", .int(ctx, .i32, opts.sublane_stride)),
        },
        .location = location,
    });
}

pub const StoreOpts = struct {
    sublane_mask: []const bool,
    sublane_stride: i32 = 1,
    add: bool = false,
};

/// tpu.store — sublane-aware store from a vreg. Optional elementwise mask.
pub fn store(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    mask: ?*const mlir.Value,
    opts: StoreOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 18) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ value, base });
    operands_buf.appendSliceAssumeCapacity(indices);
    const mask_len: i32 = if (mask) |m| blk: {
        operands_buf.appendAssumeCapacity(m);
        break :blk 1;
    } else 0;

    const seg_sizes = [4]i32{ 1, 1, @intCast(indices.len), mask_len };
    return mlir.Operation.make(ctx, "tpu.store", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
            .named(ctx, "sublane_mask", denseBoolArrayAttribute(ctx, opts.sublane_mask)),
            .named(ctx, "sublane_stride", .int(ctx, .i32, opts.sublane_stride)),
            .named(ctx, "add", .boolean(ctx, opts.add)),
        },
        .location = location,
    });
}

pub const VectorLoadOpts = struct {
    /// Per-dim load stride. Pass `&.{}` for unit-strided.
    strides: []const i32 = &.{},
};

/// tpu.vector_load — multi-dim strided load with optional elementwise mask.
pub fn vector_load(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    mask: ?*const mlir.Value,
    opts: VectorLoadOpts,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 17) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);
    const mask_len: i32 = if (mask) |m| blk: {
        operands_buf.appendAssumeCapacity(m);
        break :blk 1;
    } else 0;

    const seg_sizes = [3]i32{ 1, @intCast(indices.len), mask_len };
    return mlir.Operation.make(ctx, "tpu.vector_load", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
            .named(ctx, "strides", .denseArray(ctx, .i32, opts.strides)),
        },
        .location = location,
    });
}

pub const VectorStoreOpts = struct {
    strides: []const i32 = &.{},
    add: bool = false,
};

pub fn vector_store(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    mask: ?*const mlir.Value,
    opts: VectorStoreOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 18) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ value, base });
    operands_buf.appendSliceAssumeCapacity(indices);
    const mask_len: i32 = if (mask) |m| blk: {
        operands_buf.appendAssumeCapacity(m);
        break :blk 1;
    } else 0;

    const seg_sizes = [4]i32{ 1, 1, @intCast(indices.len), mask_len };
    return mlir.Operation.make(ctx, "tpu.vector_store", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
            .named(ctx, "strides", .denseArray(ctx, .i32, opts.strides)),
            .named(ctx, "add", .boolean(ctx, opts.add)),
        },
        .location = location,
    });
}

pub fn strided_load(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    strides: []const i32,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);

    return mlir.Operation.make(ctx, "tpu.strided_load", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "strides", .denseArray(ctx, .i32, strides)),
        },
        .location = location,
    });
}

pub fn strided_store(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    strides: []const i32,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 17) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ value, base });
    operands_buf.appendSliceAssumeCapacity(indices);

    return mlir.Operation.make(ctx, "tpu.strided_store", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = &.{
            .named(ctx, "strides", .denseArray(ctx, .i32, strides)),
        },
        .location = location,
    });
}

// =============================================================================
// Compute — matmul, iota, reciprocal, casts
// =============================================================================

pub const MatmulOpts = struct {
    /// Deprecated when `dimension_numbers` is provided.
    transpose_lhs: bool = false,
    transpose_rhs: bool = false,
    transpose_lhs_hint: bool = false,
    /// Optional precision; pass null to omit.
    precision: ?ContractPrecision = null,
    /// Optional `#tpu.dot_dimension_numbers<...>` attribute. When omitted the
    /// canonicalizer derives one. Build with `mlir.Attribute.parse`.
    dimension_numbers: ?*const mlir.Attribute = null,
};

pub fn matmul(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    acc: *const mlir.Value,
    opts: MatmulOpts,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attrs.appendSliceAssumeCapacity(&.{
        .named(ctx, "transpose_lhs", .boolean(ctx, opts.transpose_lhs)),
        .named(ctx, "transpose_rhs", .boolean(ctx, opts.transpose_rhs)),
        .named(ctx, "transpose_lhs_hint", .boolean(ctx, opts.transpose_lhs_hint)),
    });
    if (opts.precision) |p| {
        attrs.appendAssumeCapacity(.named(ctx, "precision", p.attribute(ctx)));
    }
    if (opts.dimension_numbers) |dn| {
        attrs.appendAssumeCapacity(.named(ctx, "dimension_numbers", dn));
    }

    return mlir.Operation.make(ctx, "tpu.matmul", .{
        .operands = .{ .flat = &.{ lhs, rhs, acc } },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

pub fn iota(
    ctx: *mlir.Context,
    dimensions: ?[]const i32,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (dimensions) |d| {
        attrs.appendAssumeCapacity(.named(ctx, "dimensions", .denseArray(ctx, .i32, d)));
    }
    return mlir.Operation.make(ctx, "tpu.iota", .{
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

pub const ReciprocalOpts = struct {
    approx: bool = false,
    full_range: bool = false,
};

pub fn reciprocal(
    ctx: *mlir.Context,
    input: *const mlir.Value,
    opts: ReciprocalOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.reciprocal", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{input.type_()} },
        .attributes = &.{
            .named(ctx, "approx", .boolean(ctx, opts.approx)),
            .named(ctx, "full_range", .boolean(ctx, opts.full_range)),
        },
        .location = location,
    });
}

/// Generic 1-operand → 1-result builder for the various TPU casts.
fn castOp(comptime op_name: []const u8) type {
    return struct {
        pub fn call(
            ctx: *mlir.Context,
            src: *const mlir.Value,
            result_type: *const mlir.Type,
            location: *const mlir.Location,
        ) *mlir.Operation {
            return mlir.Operation.make(ctx, op_name, .{
                .operands = .{ .flat = &.{src} },
                .results = .{ .flat = &.{result_type} },
                .location = location,
            });
        }
    };
}

pub const fptosi = castOp("tpu.fptosi").call;
pub const fptoui = castOp("tpu.fptoui").call;
pub const sitofp = castOp("tpu.sitofp").call;
pub const uitofp = castOp("tpu.uitofp").call;
pub const extf = castOp("tpu.extf").call;
pub const truncf = castOp("tpu.truncf").call;
pub const bitcast = castOp("tpu.bitcast").call;
pub const bitcast_vreg = castOp("tpu.bitcast_vreg").call;
pub const mask_cast = castOp("tpu.mask_cast").call;

// =============================================================================
// Shape — reshape / repeat / concatenate / transpose / broadcast_in_sublanes
// =============================================================================

pub fn reshape(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.reshape", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn repeat(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    dimension: i32,
    times: i32,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.repeat", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "dimension", .int(ctx, .i32, dimension)),
            .named(ctx, "times", .int(ctx, .i32, times)),
        },
        .location = location,
    });
}

pub fn concatenate(
    ctx: *mlir.Context,
    sources: []const *const mlir.Value,
    dimension: i32,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.concatenate", .{
        .operands = .{ .flat = sources },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "dimension", .int(ctx, .i32, dimension)),
        },
        .location = location,
    });
}

pub fn transpose(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    permutation: []const i64,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.transpose", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "permutation", .denseArray(ctx, .i64, permutation)),
        },
        .location = location,
    });
}

pub fn broadcast_in_sublanes(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    lane: i32,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.broadcast_in_sublanes", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "lane", .int(ctx, .i32, lane)),
        },
        .location = location,
    });
}

pub const DynamicRotateOpts = struct {
    dimension: i32,
    stride: ?i32 = null,
    stride_dimension: ?i32 = null,
};

pub fn dynamic_rotate(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    amount: *const mlir.Value,
    opts: DynamicRotateOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attrs.appendSliceAssumeCapacity(&.{
        .named(ctx, "dimension", .int(ctx, .si32, opts.dimension)),
    });
    if (opts.stride) |s| attrs.appendAssumeCapacity(.named(ctx, "stride", .int(ctx, .si32, s)));
    if (opts.stride_dimension) |sd| attrs.appendAssumeCapacity(.named(ctx, "stride_dimension", .int(ctx, .si32, sd)));

    return mlir.Operation.make(ctx, "tpu.dynamic_rotate", .{
        .operands = .{ .flat = &.{ value, amount } },
        .results = .{ .flat = &.{value.type_()} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// =============================================================================
// Memref-shaped ops — slice / squeeze / reshape / bitcast / reinterpret_cast
// =============================================================================

pub const MemRefSliceArgs = struct {
    /// `base_idx` count must equal source memref rank.
    base_idx: []const *const mlir.Value,
    /// Optional dynamic sizes; left empty when target shape is fully static.
    dynamic_sizes: []const *const mlir.Value = &.{},
};

pub fn memref_slice(
    ctx: *mlir.Context,
    mem_ref: *const mlir.Value,
    args: MemRefSliceArgs,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 32) = .empty;
    operands_buf.appendAssumeCapacity(mem_ref);
    operands_buf.appendSliceAssumeCapacity(args.base_idx);
    operands_buf.appendSliceAssumeCapacity(args.dynamic_sizes);

    const seg_sizes = [3]i32{ 1, @intCast(args.base_idx.len), @intCast(args.dynamic_sizes.len) };
    return mlir.Operation.make(ctx, "tpu.memref_slice", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
        },
        .location = location,
    });
}

pub fn memref_squeeze(
    ctx: *mlir.Context,
    mem_ref: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.memref_squeeze", .{
        .operands = .{ .flat = &.{mem_ref} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn memref_reshape(
    ctx: *mlir.Context,
    mem_ref: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.memref_reshape", .{
        .operands = .{ .flat = &.{mem_ref} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn memref_bitcast(
    ctx: *mlir.Context,
    mem_ref: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.memref_bitcast", .{
        .operands = .{ .flat = &.{mem_ref} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn reinterpret_cast(
    ctx: *mlir.Context,
    mem_ref: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.reinterpret_cast", .{
        .operands = .{ .flat = &.{mem_ref} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{.named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &.{ 1, 0, 0, 0 }))},
        .location = location,
    });
}

pub fn assume_layout(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.assume_layout", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn erase_memref_layout(
    ctx: *mlir.Context,
    mem_ref: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.erase_memref_layout", .{
        .operands = .{ .flat = &.{mem_ref} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// =============================================================================
// Mask building — create_mask / create_subelement_mask / mask_cast
// =============================================================================

/// `tpu.create_mask` — two equal-length operand groups (low, high) per dim.
pub fn create_mask(
    ctx: *mlir.Context,
    low_bounds: []const *const mlir.Value,
    high_bounds: []const *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    std.debug.assert(low_bounds.len == high_bounds.len);
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 32) = .empty;
    operands_buf.appendSliceAssumeCapacity(low_bounds);
    operands_buf.appendSliceAssumeCapacity(high_bounds);

    return mlir.Operation.make(ctx, "tpu.create_mask", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// =============================================================================
// Semaphores / DMA / barriers
// =============================================================================

pub fn sem_alloc(
    ctx: *mlir.Context,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.sem_alloc", .{
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn sem_barrier(
    ctx: *mlir.Context,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.sem_barrier", .{
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn sem_read(
    ctx: *mlir.Context,
    semaphore: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.sem_read", .{
        .operands = .{ .flat = &.{semaphore} },
        .results = .{ .flat = &.{.int(ctx, .i32)} },
        .location = location,
    });
}

pub fn sem_wait(
    ctx: *mlir.Context,
    semaphore: *const mlir.Value,
    amount: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.sem_wait", .{
        .operands = .{ .flat = &.{ semaphore, amount } },
        .location = location,
    });
}

pub const SemSignalOpts = struct {
    device_id: ?*const mlir.Value = null,
    core_id: ?*const mlir.Value = null,
    subcore_id: ?*const mlir.Value = null,
};

pub fn sem_signal(
    ctx: *mlir.Context,
    semaphore: *const mlir.Value,
    amount: *const mlir.Value,
    opts: SemSignalOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 5) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ semaphore, amount });
    const dev_len: i32 = if (opts.device_id) |d| blk: {
        operands_buf.appendAssumeCapacity(d);
        break :blk 1;
    } else 0;
    const core_len: i32 = if (opts.core_id) |c_| blk: {
        operands_buf.appendAssumeCapacity(c_);
        break :blk 1;
    } else 0;

    const subcore_len: i32 = if (opts.subcore_id) |subcore| blk: {
        operands_buf.appendAssumeCapacity(subcore);
        break :blk 1;
    } else 0;

    const seg_sizes = [5]i32{ 1, 1, dev_len, core_len, subcore_len };
    return mlir.Operation.make(ctx, "tpu.sem_signal", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
        },
        .location = location,
    });
}

pub fn barrier(
    ctx: *mlir.Context,
    barrier_id: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.barrier", .{
        .operands = .{ .flat = &.{barrier_id} },
        .location = location,
    });
}

pub const EnqueueDmaOpts = struct {
    source_semaphore: ?*const mlir.Value = null,
    device_id: ?*const mlir.Value = null,
    core_id: ?*const mlir.Value = null,
    subcore_id: ?*const mlir.Value = null,
    priority: i32 = 0,
    strict_ordering: bool = false,
};

pub fn enqueue_dma(
    ctx: *mlir.Context,
    source: *const mlir.Value,
    target: *const mlir.Value,
    target_semaphore: *const mlir.Value,
    opts: EnqueueDmaOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 7) = .empty;
    operands_buf.appendAssumeCapacity(source);
    const src_sem_len: i32 = if (opts.source_semaphore) |s| blk: {
        operands_buf.appendAssumeCapacity(s);
        break :blk 1;
    } else 0;
    operands_buf.appendAssumeCapacity(target);
    operands_buf.appendAssumeCapacity(target_semaphore);
    const dev_len: i32 = if (opts.device_id) |d| blk: {
        operands_buf.appendAssumeCapacity(d);
        break :blk 1;
    } else 0;
    const core_len: i32 = if (opts.core_id) |c_| blk: {
        operands_buf.appendAssumeCapacity(c_);
        break :blk 1;
    } else 0;

    // Source / source_semaphore? / target / target_semaphore / device_id? / core_id? / subcore_id?
    const subcore_len: i32 = if (opts.subcore_id) |subcore| blk: {
        operands_buf.appendAssumeCapacity(subcore);
        break :blk 1;
    } else 0;

    const seg_sizes = [7]i32{ 1, src_sem_len, 1, 1, dev_len, core_len, subcore_len };
    return mlir.Operation.make(ctx, "tpu.enqueue_dma", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
            .named(ctx, "priority", .int(ctx, .i32, opts.priority)),
            .named(ctx, "strict_ordering", .boolean(ctx, opts.strict_ordering)),
        },
        .location = location,
    });
}

pub const WaitDma2Opts = struct {
    device_id: ?*const mlir.Value = null,
    core_id: ?*const mlir.Value = null,
    strict_ordering: bool = false,
};

pub fn wait_dma2(
    ctx: *mlir.Context,
    semaphore: *const mlir.Value,
    src: *const mlir.Value,
    dst: *const mlir.Value,
    opts: WaitDma2Opts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 5) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ semaphore, src, dst });
    const dev_len: i32 = if (opts.device_id) |d| blk: {
        operands_buf.appendAssumeCapacity(d);
        break :blk 1;
    } else 0;
    const core_len: i32 = if (opts.core_id) |c_| blk: {
        operands_buf.appendAssumeCapacity(c_);
        break :blk 1;
    } else 0;

    const seg_sizes = [5]i32{ 1, 1, 1, dev_len, core_len };
    return mlir.Operation.make(ctx, "tpu.wait_dma2", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
            .named(ctx, "strict_ordering", .boolean(ctx, opts.strict_ordering)),
        },
        .location = location,
    });
}

// =============================================================================
// Region-bearing — region / trace / yield
// =============================================================================

/// tpu.region — single-block region with an implicit `tpu.yield` terminator.
pub fn region(
    ctx: *mlir.Context,
    body: *mlir.Block,
    result_types: []const *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.region", .{
        .results = .{ .flat = result_types },
        .blocks = &.{body},
        .verify = false,
        .location = location,
    });
}

pub fn trace(
    ctx: *mlir.Context,
    body: *mlir.Block,
    result_types: []const *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.trace", .{
        .results = .{ .flat = result_types },
        .blocks = &.{body},
        .verify = false,
        .location = location,
    });
}

pub fn yield(
    ctx: *mlir.Context,
    values: []const *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.yield", .{
        .operands = .{ .flat = values },
        .verify = false,
        .location = location,
    });
}

pub fn trace_start(
    ctx: *mlir.Context,
    level: i32,
    message: []const u8,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.trace_start", .{
        .attributes = &.{
            .named(ctx, "level", .int(ctx, .i32, level)),
            .named(ctx, "message", .string(ctx, message)),
        },
        .verify = false,
        .location = location,
    });
}

pub fn trace_stop(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.trace_stop", .{
        .verify = false,
        .location = location,
    });
}

/// Build a `#tpu.dot_dimension_numbers<...>` attribute. Mirrors
/// `tpu_ops.td:TPU_DotDimensionNumbersAttr`. Pass empty slices for the
/// fields you don't use (most dot operations leave batch dims empty).
pub fn dotDimensionNumbers(
    ctx: *mlir.Context,
    lhs_contracting: []const i64,
    rhs_contracting: []const i64,
    lhs_non_contracting: []const i64,
    rhs_non_contracting: []const i64,
    output_dim_order: []const i64,
    lhs_batch: []const i64,
    rhs_batch: []const i64,
) *const mlir.Attribute {
    var buf: [512]u8 = undefined;
    var len: usize = 0;
    len += (std.fmt.bufPrint(buf[len..], "#tpu.dot_dimension_numbers<", .{}) catch unreachable).len;
    inline for (.{
        lhs_contracting,
        rhs_contracting,
        lhs_non_contracting,
        rhs_non_contracting,
        output_dim_order,
        lhs_batch,
        rhs_batch,
    }, 0..) |arr, i| {
        if (i != 0) len += (std.fmt.bufPrint(buf[len..], ", ", .{}) catch unreachable).len;
        len += (std.fmt.bufPrint(buf[len..], "[", .{}) catch unreachable).len;
        for (arr, 0..) |v, j| {
            const sep = if (j == 0) "" else ", ";
            len += (std.fmt.bufPrint(buf[len..], "{s}{d}", .{ sep, v }) catch unreachable).len;
        }
        len += (std.fmt.bufPrint(buf[len..], "]", .{}) catch unreachable).len;
    }
    len += (std.fmt.bufPrint(buf[len..], ">", .{}) catch unreachable).len;
    return mlir.Attribute.parse(ctx, buf[0..len]) catch
        std.debug.panic("failed to parse tpu.dot_dimension_numbers '{s}'", .{buf[0..len]});
}

// =============================================================================
// Misc — device_id / delay / assume_multiple
// =============================================================================

pub fn device_id(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.device_id", .{
        .results = .{ .flat = &.{.int(ctx, .i32)} },
        .location = location,
    });
}

pub fn delay(
    ctx: *mlir.Context,
    cycles: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.delay", .{
        .operands = .{ .flat = &.{cycles} },
        .location = location,
    });
}

pub fn assume_multiple(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    multiple: i32,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.assume_multiple", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{src.type_()} },
        .attributes = &.{
            .named(ctx, "multiple", .int(ctx, .i32, multiple)),
        },
        .location = location,
    });
}

// =============================================================================
// Internal helpers
// =============================================================================

/// `tpu.DenseBoolArrayAttr` stores one C `int` (i32) per bool, see `mlirDenseBoolArrayGet`
fn denseBoolArrayAttribute(ctx: *mlir.Context, values: []const bool) *const mlir.Attribute {
    var ints: stdx.BoundedArray(i32, 64) = .empty;
    for (values) |b| ints.appendAssumeCapacity(@intFromBool(b));
    return .denseArray(ctx, .bool, ints.constSlice());
}

test {
    std.testing.refAllDecls(@This());
}

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    registry.registerDialect("tpu");
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

fn expectSegments(op: *const mlir.Operation, expected: []const i32) !void {
    const segments = op.attributeByName("operandSegmentSizes").?.isA(mlir.DenseArrayAttribute(.i32)).?;
    try std.testing.expectEqual(expected.len, segments.numElements());
    for (expected, 0..) |size, i| try std.testing.expectEqual(size, segments.element(i));
}

test "DMA and semaphore wrappers satisfy current operand groups" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc = mlir.Location.unknown(ctx);
    const args = mlir.Block.init(&.{}, &.{});
    defer args.deinit();
    const source = args.addArgument(try mlir.Type.parse(ctx, "memref<8x128xbf16, #tpu.memory_space<hbm>>"), loc);
    const target = args.addArgument(try mlir.Type.parse(ctx, "memref<8x128xbf16, #tpu.memory_space<vmem>>"), loc);
    const dma_sem = args.addArgument(try mlir.Type.parse(ctx, "memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>"), loc);
    const sem = args.addArgument(try mlir.Type.parse(ctx, "memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>"), loc);
    const amount = args.addArgument(.int(ctx, .i32), loc);

    const dma = enqueue_dma(ctx, source, target, dma_sem, .{}, loc);
    defer dma.deinit();
    try expectSegments(dma, &.{ 1, 0, 1, 1, 0, 0, 0 });
    try std.testing.expect(dma.verify());
    const signal = sem_signal(ctx, sem, amount, .{}, loc);
    defer signal.deinit();
    try expectSegments(signal, &.{ 1, 1, 0, 0, 0 });
    try std.testing.expect(signal.verify());

    const sc_sem = args.addArgument(try mlir.Type.parse(ctx, "memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem, sc_vector_subcore>>"), loc);
    const sc_dma_sem = args.addArgument(try mlir.Type.parse(ctx, "memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem, sc_vector_subcore>>"), loc);
    const core = args.addArgument(.int(ctx, .i32), loc);
    const subcore = args.addArgument(.int(ctx, .i32), loc);
    const remote_signal = sem_signal(ctx, sc_sem, amount, .{ .core_id = core, .subcore_id = subcore }, loc);
    defer remote_signal.deinit();
    try expectSegments(remote_signal, &.{ 1, 1, 0, 1, 1 });
    try std.testing.expect(remote_signal.operand(3).eql(subcore));
    const remote_dma = enqueue_dma(ctx, source, target, sc_dma_sem, .{
        .source_semaphore = dma_sem,
        .core_id = core,
        .subcore_id = subcore,
    }, loc);
    defer remote_dma.deinit();
    try expectSegments(remote_dma, &.{ 1, 1, 1, 1, 0, 1, 1 });
    try std.testing.expect(remote_dma.operand(5).eql(subcore));
}

test "matmul and reinterpret cast satisfy current required properties" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc = mlir.Location.unknown(ctx);
    const args = mlir.Block.init(&.{}, &.{});
    defer args.deinit();
    const vector_type = mlir.Type.vector(&.{ 8, 8 }, .float(ctx, .f32));
    const lhs = args.addArgument(vector_type, loc);
    const rhs = args.addArgument(vector_type, loc);
    const acc = args.addArgument(vector_type, loc);
    const product = matmul(ctx, lhs, rhs, acc, .{}, vector_type, loc);
    defer product.deinit();
    try std.testing.expect(product.verify());
    try std.testing.expect(!product.attributeByName("transpose_lhs_hint").?.isA(mlir.BoolAttribute).?.value());

    const ref_type = try mlir.Type.parse(ctx, "memref<8x8xf32, #tpu.memory_space<vmem>>");
    const ref = args.addArgument(ref_type, loc);
    const cast = reinterpret_cast(ctx, ref, ref_type, loc);
    defer cast.deinit();
    try expectSegments(cast, &.{ 1, 0, 0, 0 });
    try std.testing.expect(cast.verify());
}

test "legacy reduction extrema and scan axis preserve their semantics" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc = mlir.Location.unknown(ctx);
    const args = mlir.Block.init(&.{}, &.{});
    defer args.deinit();
    const float_type = mlir.Type.vector(&.{ 2, 8 }, .float(ctx, .f32));
    const int_type = mlir.Type.vector(&.{ 2, 8 }, .int(ctx, .i32));
    const floats = args.addArgument(float_type, loc);
    const integers = args.addArgument(int_type, loc);
    const mask = args.addArgument(mlir.Type.vector(&.{8}, .int(ctx, .i1)), loc);
    const float_min = all_reduce(ctx, floats, 1, .min, float_type, loc);
    defer float_min.deinit();
    const int_max = all_reduce(ctx, integers, 1, .max, int_type, loc);
    defer int_max.deinit();
    try std.testing.expect(float_min.attributeByName("kind").?.eql(try mlir.Attribute.parse(ctx, "#tpu.reduction_kind<minf>")));
    try std.testing.expect(int_max.attributeByName("kind").?.eql(try mlir.Attribute.parse(ctx, "#tpu.reduction_kind<maxsi>")));

    const int_scan = scan(ctx, integers, .max, mask, int_type, loc);
    defer int_scan.deinit();
    try std.testing.expect(int_scan.verify());
    try std.testing.expect(int_scan.attributeByName("kind").?.eql(try mlir.Attribute.parse(ctx, "#tpu.reduction_kind<maxui>")));
    try std.testing.expectEqual(@as(i64, 1), int_scan.attributeByName("dimension").?.isA(mlir.IntegerAttribute).?.value(i64));
    const float_scan = scan(ctx, floats, .min, null, float_type, loc);
    defer float_scan.deinit();
    try std.testing.expect(float_scan.attributeByName("kind").?.eql(try mlir.Attribute.parse(ctx, "#tpu.reduction_kind<minf>")));
}
