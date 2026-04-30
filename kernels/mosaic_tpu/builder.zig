//! ZML Mosaic TPU DSL — a Zig builder for Mosaic TPU IR (the `tpu` dialect).
//!
//! Args are memrefs (`!memref<S x dtype, mem_space>`) plus optional scalars,
//! iteration-index, and semaphore args. The dialect must be registered into
//! the MLIR context — see `mlir/dialects/mosaic_tpu`. The entire Mosaic stack is
//! `manual`-tagged in Bazel until the JAX repo is wired into `MODULE.bazel`.

const std = @import("std");

const mlir = @import("mlir");
const dialects = @import("mlir/dialects");
const dsl = @import("kernels/common");
const dtypes = @import("dtype.zig");
const stdx = @import("stdx");
const tpu = @import("mlir/dialects/mosaic_tpu");

const arith = dialects.arith;
const cf = dialects.cf;
const func = dialects.func;
const math = dialects.math;
const memref = dialects.memref;
const scf = dialects.scf;
const vector = dialects.vector;

pub const DType = dtypes.DType;

const isFloatDtype = dtypes.isFloatDtype;
const dtypeBitwidth = dtypes.dtypeBitwidth;
const intBitwidth = dtypes.intBitwidth;

/// `tpu`-dialect enums re-exported for convenience.
pub const MemorySpace = tpu.MemorySpace;
pub const ReductionKind = tpu.ReductionKind;
pub const ContractPrecision = tpu.ContractPrecision;
pub const RoundingMode = tpu.RoundingMode;
pub const DimensionSemantics = tpu.DimensionSemantics;
pub const CoreType = tpu.CoreType;
pub const PipelineMode = tpu.PipelineMode;
pub const RevisitMode = tpu.RevisitMode;

/// `vector`-dialect enums re-exported for convenience.
/// Used by `vectorReduction` / `multiReduction`.
pub const CombiningKind = vector.CombiningKind;

// =============================================================================
// Value — a handle to an MLIR value inside the kernel body.
// =============================================================================

/// MLIR SSA value with an optional back-pointer to its `Builder` for fluent ops.
pub const Value = struct {
    inner: *const mlir.Value,
    kernel: ?*Builder = null,

    pub const Shape = stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK);

    pub fn type_(self: Value) *const mlir.Type {
        return self.inner.type_();
    }

    fn kern(self: Value) *Builder {
        return self.kernel orelse @panic("Value has no owning kernel; use Builder.* helpers instead");
    }

    /// Element type of a vector / memref value, or the value's own type if
    /// scalar.
    pub fn elemType(self: Value) *const mlir.Type {
        if (self.type_().isA(mlir.ShapedType)) |shaped| return shaped.elementType();
        return self.type_();
    }

    /// True when this value is a vector (rank ≥ 1).
    pub fn isVector(self: Value) bool {
        return self.type_().isA(mlir.VectorType) != null;
    }

    /// True when this value is a memref.
    pub fn isMemRef(self: Value) bool {
        return self.type_().isA(mlir.MemRefType) != null;
    }

    /// True when the value is shaped (vector / memref / tensor).
    pub fn isShaped(self: Value) bool {
        return self.type_().isA(mlir.ShapedType) != null;
    }

    /// Rank of this value. Scalars have rank 0.
    pub fn rank(self: Value) usize {
        if (self.type_().isA(mlir.ShapedType)) |shaped| return shaped.rank();
        return 0;
    }

    /// `i`-th dim size of a shaped value; traps on scalar.
    pub fn dim(self: Value, i: usize) i64 {
        return self.type_().isA(mlir.ShapedType).?.dimension(i);
    }

    /// Shape as a stack-allocated BoundedArray. Use `.constSlice()` for `[]const i64`.
    pub fn shape(self: Value) Shape {
        var out: Shape = .empty;
        if (self.type_().isA(mlir.ShapedType)) |shaped| {
            const r = shaped.rank();
            for (0..r) |i| out.appendAssumeCapacity(shaped.dimension(i));
        }
        return out;
    }

    /// True if the element type is a floating-point type (f16/bf16/f32/f64/fp8/…).
    pub fn isFloatElem(self: Value) bool {
        const et = self.elemType();
        inline for (std.meta.fields(mlir.FloatTypes)) |f| {
            if (et.isA(mlir.FloatType(@field(mlir.FloatTypes, f.name))) != null) return true;
        }
        return false;
    }

    /// True if the element type is an integer type.
    pub fn isIntElem(self: Value) bool {
        return self.elemType().isA(mlir.IntegerType) != null;
    }

    // -------- fluent arithmetic (int/float auto-dispatch from elem type) --------

    pub fn add(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.addf(l, r) else k.addi(l, r);
    }

    pub fn sub(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.subf(l, r) else k.subi(l, r);
    }

    pub fn mul(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.mulf(l, r) else k.muli(l, r);
    }

    pub fn div(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.divf(l, r) else k.divsi(l, r);
    }

    pub fn rem(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.remf(l, r) else k.remsi(l, r);
    }

    pub fn bitAnd(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.andi(l, r);
    }

    pub fn bitOr(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.ori(l, r);
    }

    pub fn minimum(self: Value, rhs: anytype) Value {
        return self.kern().minimum(self, rhs);
    }
    pub fn maximum(self: Value, rhs: anytype) Value {
        return self.kern().maximum(self, rhs);
    }

    // -------- fluent comparisons (int predicates default; float uses cmpf) --------

    pub fn lt(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.olt, l, r) else k.cmpi(.slt, l, r);
    }
    pub fn le(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.ole, l, r) else k.cmpi(.sle, l, r);
    }
    pub fn gt(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.ogt, l, r) else k.cmpi(.sgt, l, r);
    }
    pub fn ge(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.oge, l, r) else k.cmpi(.sge, l, r);
    }
    pub fn eq(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.oeq, l, r) else k.cmpi(.eq, l, r);
    }
    pub fn ne(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.one, l, r) else k.cmpi(.ne, l, r);
    }

    // -------- casts --------

    /// Numeric cast preserving shape (vector or scalar). Auto-dispatches on
    /// source / destination float-vs-int.
    pub fn to(self: Value, dtype: DType) Value {
        return self.kern().cast(self, dtype);
    }

    /// `tpu.bitcast` — same-bitwidth element-type swap.
    pub fn bitcast(self: Value, dtype: DType) Value {
        return self.kern().bitcastTo(self, dtype);
    }

    /// Absolute value (float→absf, int→absi).
    pub fn abs(self: Value) Value {
        return self.kern().abs(self);
    }
};

// =============================================================================
// Arg spec — describes one function parameter.
// =============================================================================

/// Per-arg specification for building `func.func`'s signature.
pub const ArgSpec = struct {
    name: []const u8,
    kind: Kind,

    pub const Kind = union(enum) {
        /// Plain scalar — typically used for grid sizes / dynamic dims passed
        /// in by the runtime. `.{ .scalar = .i32 }`.
        scalar: DType,
        /// Loop / grid iteration index — `index` type with optional
        /// `tpu.dimension_semantics` annotation.
        /// `.{ .iter = .{ .semantics = .parallel } }`.
        iter: IterOpts,
        /// Memref (the bread-and-butter kernel arg). The arg is typed as
        /// `memref<shape x dtype, #tpu.memory_space<...>>`.
        /// `.{ .ref = .{ .shape = &.{128, 256}, .dtype = .bf16, .memory_space = .vmem } }`.
        ref: RefSpec,
        /// `memref<shape x !tpu.(dma_)semaphore, semaphore_mem>` — the
        /// scratch-array form Pallas uses for semaphore pools. Pallas
        /// classifies it as `.scratch`, so it never appears in
        /// `window_params` / `transform_N`.
        sem_array: SemArraySpec,
        /// `!tpu.semaphore` or `!tpu.dma_semaphore`.
        /// `.{ .sem = .regular }` / `.{ .sem = .dma }`.
        sem: SemKind,
    };

    pub const IterOpts = struct {
        semantics: ?DimensionSemantics = null,
    };

    pub const RefSpec = struct {
        shape: []const i64,
        dtype: DType,
        memory_space: MemorySpace = .vmem,
        /// Block window. `null` ⇒ trivial window (block == operand shape);
        /// VMEM trivial windows auto-get `pipeline_mode = synchronous`.
        window: ?WindowSpec = null,
        /// Pallas role of this operand. Only `.input`/`.output` get a
        /// `window_params` entry and a `transform_N` stub.
        /// `.scalar_prefetch` refs become trailing args of every
        /// `transform_N` body (Pallas's index_map sees them).
        /// `.scratch` refs are not visible to transforms or window_params.
        role: Role = .input,
    };

    pub const Role = enum { input, output, scratch, scalar_prefetch };

    pub const WindowSpec = struct {
        /// `null` ⇒ use the operand's full shape (trivial window).
        /// Determines `window_bounds` and the `transform_N` arity.
        block_shape: ?[]const i64 = null,
        /// `#tpu.pipeline_mode<…>`. `null` ⇒ auto-`synchronous` for VMEM trivial windows, omitted otherwise.
        pipeline_mode: ?PipelineMode = null,
        /// `#tpu.revisit_mode<…>`. `null` ⇒ omit.
        revisit_mode: ?RevisitMode = null,
        /// `#tpu.element_window<[pad_low], [pad_high]>`. `null` ⇒ omit.
        element_window: ?ElementWindowSpec = null,
        /// What `transform_N` returns. `null` ⇒ all-zero `block_rank` i32s
        /// (matching a trivial index_map). One entry per result; length
        /// must match `block_shape.len` (or operand rank if `block_shape`
        /// is `null`).
        transform_returns: ?[]const TransformReturn = null,
    };

    /// One return slot of a `transform_N` body. Mirrors what
    /// `lower_jaxpr_to_transform_func` produces — typically a select of
    /// program-ids, scalar-prefetch loads, or constants. We only encode
    /// the most common forms; drop to a custom transform body for anything
    /// fancier.
    pub const TransformReturn = union(enum) {
        /// Return `arith.constant 0 : i32`.
        zero,
        /// Return the Nth program-id (`%argN`).
        program_id: usize,
    };

    pub const ElementWindowSpec = struct {
        pad_low: []const i64,
        pad_high: []const i64,
    };

    pub const SemKind = enum { regular, dma };

    pub const SemArraySpec = struct {
        shape: []const i64,
        kind: SemKind = .dma,
    };
};

pub const FinishError = error{ InvalidMlir, MlirUnexpected } || std.mem.Allocator.Error || std.Io.Writer.Error;

// =============================================================================
// Named-param structs.
// =============================================================================

pub const VectorLoadOpts = struct {
    /// Per-dim load stride. Pass `&.{}` for unit-strided.
    strides: []const i32 = &.{},
    /// Optional element-wise mask. Must be a `vector<...xi1>` matching the
    /// load result shape.
    mask: ?Value = null,
};

pub const VectorStoreOpts = struct {
    strides: []const i32 = &.{},
    mask: ?Value = null,
    /// If true emits `tpu.vector_store ... add = true` (atomic-add semantics
    /// where supported).
    add: bool = false,
};

pub const MatmulOpts = struct {
    transpose_lhs: bool = false,
    transpose_rhs: bool = false,
    precision: ?ContractPrecision = null,
    /// Optional `#tpu.dot_dimension_numbers<...>` attribute. When omitted the
    /// canonicalizer derives one. Build with `mlir.Attribute.parse`.
    dimension_numbers: ?*const mlir.Attribute = null,
};

pub const ReduceOpts = struct {
    /// Reduction dim. Defaults to 0.
    dim: i64 = 0,
};

pub const SortOpts = struct {
    descending: bool = false,
};

pub const SemSignalOpts = struct {
    device_id: ?Value = null,
    core_id: ?Value = null,
};

pub const EnqueueDmaOpts = struct {
    source_semaphore: ?Value = null,
    device_id: ?Value = null,
    core_id: ?Value = null,
    priority: i32 = 0,
    strict_ordering: bool = false,
};

pub const WaitDma2Opts = struct {
    device_id: ?Value = null,
    core_id: ?Value = null,
    strict_ordering: bool = false,
};

pub const ReciprocalOpts = struct {
    approx: bool = false,
    full_range: bool = false,
};

pub const RotateOpts = struct {
    amount: i32,
    dimension: i32,
    stride: ?i32 = null,
    stride_dimension: ?i32 = null,
};

pub const CastOpts = struct {
    /// Forwarded to `tpu.fptosi` / `tpu.fptoui` / etc.
    rounding: ?RoundingMode = null,
    /// If true, perform a same-bitwidth `tpu.bitcast` instead of a numeric cast.
    bitcast: bool = false,
};

pub const MinMaxOpts = struct {
    /// IEEE NaN-propagation toggle. Default is non-propagating (maxnumf/minnumf).
    propagate_nan: bool = false,
};

// =============================================================================
// Builder — the main DSL builder.
// =============================================================================

/// The main DSL builder. Create with `init`, populate the body with helpers,
/// then call `finish` to get the IR string.
pub const Builder = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    ctx: *mlir.Context,
    module: *mlir.Module,
    /// Name of the kernel — used when `declareArgs` creates the `func.func`.
    name: []const u8,
    /// `func.func` op for this kernel. `null` between `open` and the first
    /// `declareArgs` call; populated thereafter.
    func_op: ?*mlir.Operation,
    /// Entry block for the `func.func`. `null` between `open` and the first
    /// `declareArgs` call; populated thereafter.
    entry_block: ?*mlir.Block,
    block_stack: std.ArrayList(*mlir.Block),
    /// Cached `arith.constant 0 : index` Value (and the block it lives
    /// in) so repeat no-index loads / stores share an SSA value. See
    /// `zeroIndex` for the lookup logic.
    zero_index: ?Value = null,
    zero_index_block: ?*mlir.Block = null,
    /// Snapshot of the kernel's ref-args saved at `declareArgsLowOpts`
    /// time. `finish` walks this list to emit `transform_N` stubs when
    /// `pallas_window_params=true`. Allocated in `arena`.
    ref_args: []RefArgSnapshot = &.{},
    /// SMEM scalar-prefetch ref types — appended to every `transform_N`
    /// body's argument list. Empty when no `.scalar_prefetch` refs were
    /// declared.
    prefetch_args: []const *const mlir.Type = &.{},
    /// Number of grid program-ids (i32) at the head of every `transform_N`
    /// body. Equals `Opts.dimension_semantics.len`.
    grid_dim_count: usize = 0,
    /// Whether to emit Pallas-style `transform_N` stubs at `finish` time.
    /// Set by `declareArgsLowOpts` from `Opts.pallas_window_params`.
    emit_transform_stubs: bool = false,

    /// Per-ref-arg snapshot for `finish`-time `transform_N` stub emission.
    /// `skipped` = ANY/HBM/SEMAPHORE memory space (empty window, no transform).
    pub const RefArgSnapshot = struct {
        block_rank: i64,
        skipped: bool,
        role: ArgSpec.Role,
        transform_returns: ?[]const ArgSpec.TransformReturn = null,
    };

    /// Snapshot of the SMEM scalar-prefetch ref types (after the program-ids)
    /// — repeated as trailing args of every `transform_N` body to mirror
    /// Pallas's `lower_jaxpr_to_transform_func`.
    pub const PrefetchSnapshot = struct {
        /// Pre-built MLIR memref types (cached pointers from declareArgs time).
        types: []const *const mlir.Type,
    };

    /// Builder-level options. `noinline` is a Zig keyword — write `.{ .@"noinline" = true }`.
    pub const Opts = struct {
        @"noinline": bool = false,
        /// `tpu.core_type` on the `func.func`. `null` skips the attr.
        core_type: ?CoreType = .tc,
        /// One entry per grid dim. Empty list emits an empty array attr
        /// (matching Pallas's `dimension_semantics = []`).
        dimension_semantics: []const DimensionSemantics = &.{},
        /// Number of leading scalar-prefetch operands.
        scalar_prefetch: i64 = 0,
        /// Number of trailing scratch refs (memrefs holding kernel-local
        /// scratch state).
        scratch_operands: i64 = 0,
        /// Pallas's `iteration_bounds = array<i64: …>` — the grid shape
        /// (one entry per `dimension_semantics`). `null` omits the attr.
        iteration_bounds: ?[]const i64 = null,
        /// When `true`, auto-emit `window_params = [...]` and trailing
        /// `transform_N` stub funcs at `finish` time. Per-arg behavior
        /// matches `lowering.py:1014–1131`: ANY/HBM/SEMAPHORE refs get
        /// an empty `{}` and no transform; VMEM trivial windows
        /// auto-force `pipeline_mode = synchronous`. Override per-arg
        /// via `RefSpec.window`.
        pallas_window_params: bool = false,
    };

    /// Create a sub-module containing a single public `func.func` of the
    /// given name, signature, and result types.
    pub fn init(
        allocator: std.mem.Allocator,
        ctx: *mlir.Context,
        name: []const u8,
        args: []const ArgSpec,
        result_types: []const *const mlir.Type,
    ) !Builder {
        return initOpts(allocator, ctx, name, args, result_types, .{});
    }

    pub fn initOpts(
        allocator: std.mem.Allocator,
        ctx: *mlir.Context,
        name: []const u8,
        args: []const ArgSpec,
        result_types: []const *const mlir.Type,
        opts: Opts,
    ) !Builder {
        var b = try open(allocator, ctx, name);
        errdefer b.deinit();
        try b.declareArgsLowOpts(args, result_types, opts);
        return b;
    }

    /// Create an empty `Builder` with no `func.func` yet.
    pub fn open(
        allocator: std.mem.Allocator,
        ctx: *mlir.Context,
        name: []const u8,
    ) !Builder {
        const unknown_loc: *const mlir.Location = .unknown(ctx);
        const module: *mlir.Module = .init(unknown_loc);
        errdefer module.deinit();

        var arena: std.heap.ArenaAllocator = .init(allocator);
        errdefer arena.deinit();

        return .{
            .allocator = allocator,
            .arena = arena,
            .ctx = ctx,
            .module = module,
            .name = name,
            .func_op = null,
            .entry_block = null,
            .block_stack = .empty,
        };
    }

    /// Declare the kernel's args from a named-field struct literal and
    /// return a `NamedArgs(Spec)`.
    pub fn declareArgs(self: *Builder, spec: anytype) !dsl.NamedArgs(@TypeOf(spec), Value) {
        return self.declareArgsOpts(spec, &.{}, .{});
    }

    pub fn declareArgsOpts(
        self: *Builder,
        spec: anytype,
        result_types: []const *const mlir.Type,
        opts: Opts,
    ) !dsl.NamedArgs(@TypeOf(spec), Value) {
        const Spec = @TypeOf(spec);
        const fields = @typeInfo(Spec).@"struct".fields;

        var arg_specs: [fields.len]ArgSpec = undefined;
        inline for (fields, 0..) |f, i| {
            const raw = @field(spec, f.name);
            const kind: ArgSpec.Kind = if (@TypeOf(raw) == ArgSpec.Kind) raw else blk: {
                const variant = @typeInfo(@TypeOf(raw)).@"struct".fields[0].name;
                const tag = @field(std.meta.Tag(ArgSpec.Kind), variant);
                const inner = @field(raw, variant);
                break :blk switch (tag) {
                    .scalar => .{ .scalar = inner },
                    .iter => .{ .iter = .{
                        .semantics = if (@hasField(@TypeOf(inner), "semantics")) inner.semantics else null,
                    } },
                    .ref => .{ .ref = .{
                        .shape = inner.shape,
                        .dtype = inner.dtype,
                        .memory_space = if (@hasField(@TypeOf(inner), "memory_space")) inner.memory_space else .vmem,
                        .window = if (@hasField(@TypeOf(inner), "window")) inner.window else null,
                        .role = if (@hasField(@TypeOf(inner), "role")) inner.role else .input,
                    } },
                    .sem_array => .{ .sem_array = .{
                        .shape = inner.shape,
                        .kind = if (@hasField(@TypeOf(inner), "kind")) inner.kind else .dma,
                    } },
                    .sem => .{ .sem = inner },
                };
            };
            arg_specs[i] = .{ .name = f.name, .kind = kind };
        }

        try self.declareArgsLowOpts(&arg_specs, result_types, opts);

        var named: dsl.NamedArgs(Spec, Value) = undefined;
        inline for (fields, 0..) |f, i| {
            @field(named, f.name) = self.arg(i);
        }
        return named;
    }

    fn declareArgsLowOpts(
        self: *Builder,
        args: []const ArgSpec,
        result_types: []const *const mlir.Type,
        opts: Opts,
    ) !void {
        std.debug.assert(self.entry_block == null);
        const ctx = self.ctx;
        const unknown_loc: *const mlir.Location = .unknown(ctx);
        const scratch = self.arena.allocator();

        const arg_types = try scratch.alloc(*const mlir.Type, args.len);
        const arg_locs = try scratch.alloc(*const mlir.Location, args.len);
        for (args, 0..) |a, i| {
            arg_types[i] = switch (a.kind) {
                .scalar => |dt| dt.toMlir(ctx),
                .iter => mlir.indexType(ctx),
                .ref => |r| mlir.memRefType(
                    r.dtype.toMlir(ctx),
                    r.shape,
                    null,
                    r.memory_space.attribute(ctx),
                ),
                .sem => |s| switch (s) {
                    .regular => tpu.semaphoreType(ctx),
                    .dma => tpu.dmaSemaphoreType(ctx),
                },
                .sem_array => |sa| mlir.memRefType(
                    switch (sa.kind) {
                        .regular => tpu.semaphoreType(ctx),
                        .dma => tpu.dmaSemaphoreType(ctx),
                    },
                    sa.shape,
                    null,
                    MemorySpace.semaphore_mem.attribute(ctx),
                ),
            };
            arg_locs[i] = unknown_loc.named(ctx, a.name);
        }

        const entry = mlir.Block.init(arg_types, arg_locs);

        // Build per-arg attribute dicts. The only ones we currently emit are
        // `tpu.dimension_semantics` on iter args.
        const arg_attrs = try scratch.alloc(*const mlir.Attribute, args.len);
        var any_arg_attr = false;
        for (args, 0..) |a, i| {
            const empty_dict: *const mlir.Attribute = mlir.dictionaryAttribute(ctx, &.{});
            switch (a.kind) {
                .iter => |io| {
                    if (io.semantics) |sem| {
                        arg_attrs[i] = mlir.dictionaryAttribute(ctx, &.{
                            .named(ctx, "tpu.dimension_semantics", sem.attribute(ctx)),
                        });
                        any_arg_attr = true;
                    } else {
                        arg_attrs[i] = empty_dict;
                    }
                },
                else => arg_attrs[i] = empty_dict,
            }
        }

        // Kernel-level func.func attributes. MLIR sorts attrs alphabetically on print.
        var extra_attrs: stdx.BoundedArray(mlir.NamedAttribute, 8) = .empty;
        const dim_sem_buf = try scratch.alloc(*const mlir.Attribute, opts.dimension_semantics.len);
        for (opts.dimension_semantics, 0..) |s, i| dim_sem_buf[i] = s.attribute(ctx);
        extra_attrs.appendAssumeCapacity(.named(ctx, "dimension_semantics", mlir.arrayAttribute(ctx, dim_sem_buf)));
        if (opts.iteration_bounds) |bounds| {
            extra_attrs.appendAssumeCapacity(.named(ctx, "iteration_bounds", mlir.denseArrayAttribute(ctx, .i64, bounds)));
        }
        extra_attrs.appendAssumeCapacity(.named(ctx, "scalar_prefetch", mlir.integerAttribute(ctx, .i64, opts.scalar_prefetch)));
        extra_attrs.appendAssumeCapacity(.named(ctx, "scratch_operands", mlir.integerAttribute(ctx, .i64, opts.scratch_operands)));
        if (opts.core_type) |ct| {
            extra_attrs.appendAssumeCapacity(.named(ctx, "tpu.core_type", ct.attribute(ctx)));
        }

        // Snapshot the ref args for `finish`-time stub emission. Only
        // `.input` / `.output` refs participate in `window_params` and
        // `transform_N` — `.scalar_prefetch` and `.scratch` are ignored
        // (they're tracked separately for transform-arg expansion).
        const arena_alloc = self.arena.allocator();
        var ref_args_buf: std.ArrayList(RefArgSnapshot) = .empty;
        var prefetch_types_buf: std.ArrayList(*const mlir.Type) = .empty;
        for (args, 0..) |a, i| switch (a.kind) {
            .ref => |r| {
                if (r.role == .input or r.role == .output) {
                    const block_shape = if (r.window) |w|
                        (w.block_shape orelse r.shape)
                    else
                        r.shape;
                    try ref_args_buf.append(arena_alloc, .{
                        .block_rank = @intCast(block_shape.len),
                        .skipped = isSkippedWindowSpace(r.memory_space),
                        .role = r.role,
                        .transform_returns = if (r.window) |w| w.transform_returns else null,
                    });
                } else if (r.role == .scalar_prefetch) {
                    try prefetch_types_buf.append(arena_alloc, arg_types[i]);
                }
            },
            else => {},
        };
        self.ref_args = try arena_alloc.dupe(RefArgSnapshot, ref_args_buf.items);
        self.prefetch_args = try arena_alloc.dupe(*const mlir.Type, prefetch_types_buf.items);
        self.grid_dim_count = opts.dimension_semantics.len;
        self.emit_transform_stubs = opts.pallas_window_params;

        // window_params: one entry per `.input` / `.output` ref.
        //   ANY/HBM/SEMAPHORE → empty dict, no transform_N.
        //   Otherwise → {transform_indices, window_bounds, [pipeline_mode], [revisit_mode], [window_kind]}.
        //   VMEM trivial-window → auto-force pipeline_mode = synchronous.
        if (opts.pallas_window_params) {
            var idx: usize = 0;
            const wp_buf = try scratch.alloc(*const mlir.Attribute, self.ref_args.len);
            for (args) |a| switch (a.kind) {
                .ref => |r| {
                    if (r.role != .input and r.role != .output) continue;
                    if (isSkippedWindowSpace(r.memory_space)) {
                        wp_buf[idx] = mlir.dictionaryAttribute(ctx, &.{});
                        idx += 1;
                        continue;
                    }

                    const w = r.window orelse ArgSpec.WindowSpec{};
                    const block_shape = w.block_shape orelse r.shape;
                    // `bm.has_trivial_window()` in pallas — block_shape ==
                    // operand_shape AND the index_map is the identity
                    // (returns all zeros).
                    const has_trivial_index_map = if (w.transform_returns) |trs| blk: {
                        for (trs) |tr| {
                            if (tr != .zero) break :blk false;
                        }
                        break :blk true;
                    } else true;
                    const trivial = std.mem.eql(i64, block_shape, r.shape) and has_trivial_index_map;

                    const resolved_pipeline: ?PipelineMode =
                        w.pipeline_mode orelse blk: {
                            // VMEM + trivial window ⇒ Buffered(1) ⇒ synchronous.
                            if (r.memory_space == .vmem and trivial) break :blk .synchronous;
                            break :blk null;
                        };

                    var sym_buf: [32]u8 = undefined;
                    const sym_name = std.fmt.bufPrint(&sym_buf, "transform_{d}", .{idx}) catch unreachable;

                    var inner: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
                    if (resolved_pipeline) |pm| {
                        inner.appendAssumeCapacity(.named(ctx, "pipeline_mode", pm.attribute(ctx)));
                    }
                    if (w.revisit_mode) |rm| {
                        inner.appendAssumeCapacity(.named(ctx, "revisit_mode", rm.attribute(ctx)));
                    }
                    inner.appendAssumeCapacity(.named(ctx, "transform_indices", mlir.flatSymbolRefAttribute(ctx, sym_name)));
                    inner.appendAssumeCapacity(.named(ctx, "window_bounds", mlir.denseArrayAttribute(ctx, .i64, block_shape)));
                    if (w.element_window) |ew| {
                        inner.appendAssumeCapacity(.named(ctx, "window_kind", tpu.elementWindowAttribute(ctx, ew.pad_low, ew.pad_high)));
                    }
                    wp_buf[idx] = mlir.dictionaryAttribute(ctx, inner.constSlice());
                    idx += 1;
                },
                else => {},
            };
            extra_attrs.appendAssumeCapacity(.named(ctx, "window_params", mlir.arrayAttribute(ctx, wp_buf)));
        }

        const func_op = func.func(ctx, .{
            .name = self.name,
            .args = arg_types,
            .args_attributes = if (any_arg_attr) arg_attrs else null,
            .results = result_types,
            .block = entry,
            .location = unknown_loc,
            .no_inline = opts.@"noinline",
            // Body is empty at declare time — verify only at `finish`.
            .verify = false,
            .extra_attributes = extra_attrs.constSlice(),
            // sym_visibility omitted — defaults to public, elided when unset.
            .visibility = null,
        });
        _ = func_op.appendTo(self.module.body());

        self.func_op = func_op;
        self.entry_block = entry;
    }

    pub fn deinit(self: *Builder) void {
        self.module.deinit();
        self.arena.deinit();
    }

    /// The i-th entry block argument.
    pub fn arg(self: *Builder, i: usize) Value {
        const eb = self.entry_block orelse @panic("Builder.arg called before declareArgs");
        return .{ .inner = eb.argument(i), .kernel = self };
    }

    pub fn pushBlock(self: *Builder, b: *mlir.Block) void {
        self.block_stack.append(self.arena.allocator(), b) catch @panic("Builder.pushBlock OOM");
    }

    pub fn popBlock(self: *Builder) void {
        _ = self.block_stack.pop();
    }

    pub fn currentBlock(self: *Builder) *mlir.Block {
        if (self.block_stack.items.len > 0) {
            return self.block_stack.items[self.block_stack.items.len - 1];
        }
        return self.entry_block orelse @panic("Builder has no current block — call declareArgs first");
    }

    /// Unpack a slice of `Value`s into an arena-allocated slice of raw MLIR
    /// value pointers.
    fn innerSlice(self: *Builder, values: []const Value) []const *const mlir.Value {
        const out = self.arena.allocator().alloc(*const mlir.Value, values.len) catch @panic("Builder.innerSlice OOM");
        for (values, 0..) |v, i| out[i] = v.inner;
        return out;
    }

    /// Emit `op` into the current block and return its first result as a Value.
    pub fn emit(self: *Builder, op: *mlir.Operation) Value {
        _ = op.appendTo(self.currentBlock());
        return .{ .inner = op.result(0), .kernel = self };
    }

    pub fn emitMulti(self: *Builder, op: *mlir.Operation, n: usize) []Value {
        _ = op.appendTo(self.currentBlock());
        const out = self.arena.allocator().alloc(Value, n) catch @panic("Builder.emitMulti OOM");
        for (0..n) |i| out[i] = .{ .inner = op.result(i), .kernel = self };
        return out;
    }

    pub fn yield(self: *Builder, values: anytype) []const Value {
        const T = @TypeOf(values);
        const info = @typeInfo(T);
        if (info != .@"struct" or !info.@"struct".is_tuple)
            @compileError("Builder.yield expects a tuple literal like `.{ v1, v2 }`");
        const n = info.@"struct".fields.len;
        const out = self.arena.allocator().alloc(Value, n) catch @panic("Builder.yield OOM");
        inline for (info.@"struct".fields, 0..) |f, i| {
            if (f.type != Value)
                @compileError("Builder.yield: every tuple element must be a Value; got " ++ @typeName(f.type));
            out[i] = @field(values, f.name);
        }
        return out;
    }

    pub fn loc(self: *const Builder) *const mlir.Location {
        return .unknown(self.ctx);
    }

    // ==================== type helpers ====================

    pub fn scalarTy(self: *const Builder, dtype: DType) *const mlir.Type {
        return dtype.toMlir(self.ctx);
    }

    pub fn vectorTy(self: *const Builder, shape: []const i64, dtype: DType) *const mlir.Type {
        return mlir.vectorType(shape, dtype.toMlir(self.ctx));
    }

    pub fn memRefTy(self: *const Builder, shape: []const i64, dtype: DType, memory_space: MemorySpace) *const mlir.Type {
        return mlir.memRefType(dtype.toMlir(self.ctx), shape, null, memory_space.attribute(self.ctx));
    }

    fn mlirElemToDType(self: *const Builder, elem: *const mlir.Type) DType {
        return dtypes.mlirElemToDType(self.ctx, elem);
    }

    /// Replace `src`'s element type with `out_dtype`, preserving rank/shape.
    fn swapElem(self: *const Builder, src: Value, out_dtype: DType) *const mlir.Type {
        const out_elem = out_dtype.toMlir(self.ctx);
        if (src.isVector()) return mlir.vectorType(src.shape().constSlice(), out_elem);
        if (src.isMemRef()) {
            const mr = src.type_().isA(mlir.MemRefType).?;
            return mlir.memRefType(out_elem, src.shape().constSlice(), null, mr.memorySpace());
        }
        return out_elem;
    }

    // ==================== constants ====================

    fn emitInt(self: *Builder, dtype: DType, value: i64) Value {
        return self.emit(arith.constant_int(self.ctx, value, dtype.toMlir(self.ctx), self.loc()));
    }

    fn emitFloat(self: *Builder, dtype: DType, value: f64) Value {
        return switch (dtype) {
            .f16 => self.emit(arith.constant_float(self.ctx, value, .f16, self.loc())),
            .bf16 => self.emit(arith.constant_float(self.ctx, value, .bf16, self.loc())),
            .f32 => self.emit(arith.constant_float(self.ctx, value, .f32, self.loc())),
            .f64 => self.emit(arith.constant_float(self.ctx, value, .f64, self.loc())),
            // fp8 has no direct arith.constant; emit f32 then truncf.
            else => self.arithTruncf(self.emitFloat(.f32, value), dtype),
        };
    }

    /// Constant `index`-typed value (for memref / loop bounds).
    pub fn cIndex(self: *Builder, value: i64) Value {
        return self.emit(arith.constant_index(self.ctx, value, self.loc()));
    }

    /// Lift a Zig scalar (or pass-through Value) to a DSL constant.
    pub fn lift(self: *Builder, value: anytype) Value {
        const T = @TypeOf(value);
        if (T == Value) return value;
        return switch (@typeInfo(T)) {
            .comptime_int => blk: {
                if (value >= std.math.minInt(i32) and value <= std.math.maxInt(i32))
                    break :blk self.emitInt(.i32, value);
                if (value >= std.math.minInt(i64) and value <= std.math.maxInt(i64))
                    break :blk self.emitInt(.i64, value);
                if (value >= 0 and value <= std.math.maxInt(u64))
                    break :blk self.emitInt(.i64, @bitCast(@as(u64, value)));
                @compileError("Builder.lift: integer literal out of 64-bit range");
            },
            .int => |info| blk: {
                const signed = info.signedness == .signed;
                if (info.bits > 32) {
                    const v: i64 = if (signed) @intCast(value) else @bitCast(@as(u64, @intCast(value)));
                    break :blk self.emitInt(.i64, v);
                }
                const v32: i32 = if (signed) @intCast(value) else @bitCast(@as(u32, @intCast(value)));
                break :blk self.emitInt(.i32, v32);
            },
            .comptime_float => self.emitFloat(.f32, @floatCast(value)),
            .float => |info| switch (info.bits) {
                16 => self.emitFloat(.f16, @floatCast(value)),
                32 => self.emitFloat(.f32, @floatCast(value)),
                64 => self.emitFloat(.f64, value),
                else => @compileError("Builder.lift: unsupported float bitwidth"),
            },
            else => @compileError("Builder.lift: unsupported type " ++ @typeName(T)),
        };
    }

    /// Lift `value` and pin its dtype to `dtype`. Unsigned ints bit-cast
    /// preserving the source bit-pattern.
    pub fn liftAs(self: *Builder, value: anytype, dtype: DType) Value {
        const T = @TypeOf(value);
        if (T == Value) return value;
        if (isFloatDtype(dtype)) {
            const v_f64: f64 = switch (@typeInfo(T)) {
                .comptime_int, .int => @floatFromInt(value),
                .comptime_float, .float => @floatCast(value),
                else => @compileError("Builder.liftAs: unsupported scalar " ++ @typeName(T)),
            };
            return self.emitFloat(dtype, v_f64);
        }
        const v_i64: i64 = switch (@typeInfo(T)) {
            .comptime_int => blk: {
                if (value >= std.math.minInt(i64) and value <= std.math.maxInt(i64))
                    break :blk @intCast(value);
                if (value >= 0 and value <= std.math.maxInt(u64))
                    break :blk @bitCast(@as(u64, value));
                @compileError("Builder.liftAs: integer literal out of 64-bit range");
            },
            .int => |info| if (info.signedness == .signed) @intCast(value) else @bitCast(@as(u64, @intCast(value))),
            .comptime_float, .float => @intFromFloat(value),
            else => @compileError("Builder.liftAs: unsupported scalar " ++ @typeName(T)),
        };
        return self.emitInt(dtype, v_i64);
    }

    /// Build an arith.constant of the given MLIR element type from a Zig
    /// numeric. Used for matching the type of an existing reference value.
    pub fn constMatching(self: *Builder, value: anytype, elem: *const mlir.Type) Value {
        if (elem.eql(mlir.indexType(self.ctx))) return self.cIndex(@intCast(value));
        return self.liftAs(value, self.mlirElemToDType(elem));
    }

    fn liftMatching(self: *Builder, value: anytype, ref_elem: *const mlir.Type) Value {
        const T = @TypeOf(value);
        if (T == Value) return value;
        return switch (@typeInfo(T)) {
            .comptime_int, .comptime_float => self.constMatching(value, ref_elem),
            else => self.lift(value),
        };
    }

    /// Splat a scalar to a vector. Zig scalars take the dense-splat fast path
    /// (`arith.constant dense<v>`); a `Value` falls back to `vector.broadcast`.
    pub fn splat(self: *Builder, value: anytype, shape: []const i64, dtype: DType) Value {
        const T = @TypeOf(value);
        const ty = mlir.vectorType(shape, dtype.toMlir(self.ctx));
        if (T == Value) return self.emit(vector.broadcast(self.ctx, value.inner, ty, self.loc()));
        if (isFloatDtype(dtype)) {
            const v_f64: f64 = switch (@typeInfo(T)) {
                .comptime_int, .int => @floatFromInt(value),
                .comptime_float, .float => @floatCast(value),
                else => @compileError("Builder.splat: unsupported scalar " ++ @typeName(T)),
            };
            return self.emit(arith.constant_dense_splat_f(self.ctx, v_f64, ty, self.loc()));
        }
        const v_i64: i64 = switch (@typeInfo(T)) {
            .comptime_int => @intCast(value),
            .int => |info| if (info.signedness == .signed) @intCast(value) else @bitCast(@as(u64, @intCast(value))),
            .comptime_float, .float => @intFromFloat(value),
            else => @compileError("Builder.splat: unsupported scalar " ++ @typeName(T)),
        };
        return self.emit(arith.constant_dense_splat_i(self.ctx, v_i64, ty, self.loc()));
    }

    /// Zero-valued vector of `shape x dtype`.
    pub fn zeros(self: *Builder, shape: []const i64, dtype: DType) Value {
        return self.splat(@as(i64, 0), shape, dtype);
    }

    pub fn ones(self: *Builder, shape: []const i64, dtype: DType) Value {
        return self.splat(@as(i64, 1), shape, dtype);
    }

    pub fn full(self: *Builder, shape: []const i64, value: anytype, dtype: DType) Value {
        return self.splat(value, shape, dtype);
    }

    /// Lift scalars and broadcast scalar → vector to match the other operand.
    pub fn coerce(self: *Builder, a: anytype, b: anytype) struct { Value, Value } {
        const a_ref: ?*const mlir.Type = if (@TypeOf(b) == Value) b.elemType() else null;
        const b_ref: ?*const mlir.Type = if (@TypeOf(a) == Value) a.elemType() else null;
        var av = if (a_ref) |t| self.liftMatching(a, t) else self.lift(a);
        var bv = if (b_ref) |t| self.liftMatching(b, t) else self.lift(b);

        // integer-width promotion (int family only).
        if (av.isIntElem() and bv.isIntElem()) {
            const a_dt = self.mlirElemToDType(av.elemType());
            const b_dt = self.mlirElemToDType(bv.elemType());
            const aw = intBitwidth(a_dt);
            const bw = intBitwidth(b_dt);
            if (aw < bw) av = av.to(b_dt);
            if (bw < aw) bv = bv.to(a_dt);
        }

        // scalar↔vector splat.
        if (av.isVector() and !bv.isVector()) {
            const dt = self.mlirElemToDType(av.elemType());
            bv = self.splat(bv, av.shape().constSlice(), dt);
        }
        if (!av.isVector() and bv.isVector()) {
            const dt = self.mlirElemToDType(bv.elemType());
            av = self.splat(av, bv.shape().constSlice(), dt);
        }

        return .{ av, bv };
    }

    // ==================== arith: binary ====================

    pub fn addi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.addi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn subi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.subi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn muli(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.muli(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn addf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.addf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn subf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.subf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn mulf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.mulf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn divf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.divf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn divsi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.divsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn divui(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.divui(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    /// Mirrors `jax.numpy.floor_divide` (`jax/_src/numpy/ufuncs.py:2487`):
    /// truncating `divsi` plus the sign-correcting expansion
    ///   `q - 1 if (sign(a) != sign(b) and rem != 0) else q`.
    /// `lhs` must be a vector of i32. `rhs` is `anytype` — passing a
    /// scalar comptime int produces the dense<...> splat Pallas emits.
    /// JAX's MLIR canonicalizer does not fold the expansion away even
    /// when one operand is a constant, so this helper is needed for
    /// IR-equivalent floor-div with `pallas_call`.
    pub fn divFloor(self: *Builder, lhs: Value, rhs: anytype) Value {
        const shape = lhs.shape().constSlice();
        const rhs_v: Value = if (@TypeOf(rhs) == Value) rhs else self.splat(rhs, shape, .i32);
        const zero_v = self.zeros(shape, .i32);
        const one_v = self.ones(shape, .i32);
        const q = self.divsi(lhs, rhs_v);
        const sign_a = self.subi(
            self.extui(self.cmpi(.sgt, lhs, zero_v), .i32),
            self.extui(self.cmpi(.slt, lhs, zero_v), .i32),
        );
        const sign_b = self.subi(
            self.extui(self.cmpi(.sgt, rhs_v, zero_v), .i32),
            self.extui(self.cmpi(.slt, rhs_v, zero_v), .i32),
        );
        const sign_diff = self.cmpi(.ne, sign_a, sign_b);
        const rem_nz = self.cmpi(.ne, self.remsi(lhs, rhs_v), zero_v);
        const cond = self.andi(sign_diff, rem_nz);
        return self.select(cond, self.subi(q, one_v), q);
    }

    pub fn remsi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.remsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn remui(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.remui(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn remf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.remf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn andi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.andi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn ori(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.ori(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn xori(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.xori(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn shli(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.shli(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn shrsi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.shrsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn shrui(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.shrui(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn maxsi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.maxsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn minsi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.minsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn maximumf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.maximumf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn minimumf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.minimumf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn maxnumf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.maxnumf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn minnumf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.minnumf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn negf(self: *Builder, src: Value) Value {
        return self.emit(arith.negf(self.ctx, src.inner, self.loc()));
    }
    pub fn cmpi(self: *Builder, predicate: arith.CmpIPredicate, lhs: Value, rhs: Value) Value {
        return self.emit(arith.cmpi(self.ctx, predicate, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn cmpf(self: *Builder, predicate: arith.CmpFPredicate, lhs: Value, rhs: Value) Value {
        return self.emit(arith.cmpf(self.ctx, predicate, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn select(self: *Builder, cond: Value, t: Value, f: Value) Value {
        return self.emit(arith.select(self.ctx, cond.inner, t.inner, f.inner, self.loc()));
    }
    pub fn where(self: *Builder, condition: Value, x: Value, y: Value) Value {
        const xv, const yv = self.coerce(x, y);
        const c = if (condition.isVector() == xv.isVector()) condition else blk: {
            // Splat a scalar predicate to match a vector x/y.
            if (xv.isVector()) {
                break :blk self.splat(condition, xv.shape().constSlice(), .i1);
            }
            break :blk condition;
        };
        return self.select(c, xv, yv);
    }

    /// Elementwise max with auto-broadcast scalar→vector. Use `maxOpts`
    /// for NaN-propagation control.
    pub fn maximum(self: *Builder, a: anytype, b: anytype) Value {
        return self.maximumOpts(a, b, .{});
    }
    pub fn maximumOpts(self: *Builder, a: anytype, b: anytype, opts: MinMaxOpts) Value {
        const l, const r = self.coerce(a, b);
        if (l.isFloatElem()) {
            return if (opts.propagate_nan) self.maximumf(l, r) else self.maxnumf(l, r);
        }
        return self.maxsi(l, r);
    }
    pub fn minimum(self: *Builder, a: anytype, b: anytype) Value {
        return self.minimumOpts(a, b, .{});
    }
    pub fn minimumOpts(self: *Builder, a: anytype, b: anytype, opts: MinMaxOpts) Value {
        const l, const r = self.coerce(a, b);
        if (l.isFloatElem()) {
            return if (opts.propagate_nan) self.minimumf(l, r) else self.minnumf(l, r);
        }
        return self.minsi(l, r);
    }

    // ==================== arith casts ====================

    pub fn extsi(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(arith.extsi(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    pub fn extui(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(arith.extui(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    pub fn arithExtf(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(arith.extf(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    pub fn trunci(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(arith.trunci(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    pub fn arithTruncf(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(arith.truncf(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    pub fn arithSitofp(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(arith.sitofp(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    pub fn arithUitofp(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(arith.uitofp(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    pub fn arithFptosi(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(arith.fptosi(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    pub fn arithFptoui(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(arith.fptoui(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    pub fn arithBitcast(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(arith.bitcast(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }

    /// `arith.index_cast` — cast between integer and `index` types.
    pub fn indexCast(self: *Builder, src: Value, target: *const mlir.Type) Value {
        return self.emit(arith.index_cast(self.ctx, src.inner, target, self.loc()));
    }

    /// Cast a scalar integer (i32 / i64 / …) to the `index` type. Common
    /// pattern for using i32 program-ids and scalar-prefetch values as
    /// memref offsets.
    pub fn toIndex(self: *Builder, src: Value) Value {
        return self.indexCast(src, mlir.indexType(self.ctx));
    }

    // ==================== TPU casts ====================

    /// `tpu.bitcast` — same-bitwidth element-type swap. Shape preserved.
    pub fn bitcastTo(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(tpu.bitcast(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    pub fn extf(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(tpu.extf(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    pub fn truncf(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(tpu.truncf(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    pub fn sitofp(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(tpu.sitofp(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    pub fn uitofp(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(tpu.uitofp(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    pub fn fptosi(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(tpu.fptosi(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    pub fn fptoui(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(tpu.fptoui(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }

    /// Numeric cast with auto-dispatch. Routes through `arith.*` ops for the
    /// common float / int conversions (which don't carry a rounding-mode
    /// attribute). The `tpu.*` cast variants (`tpuTruncf`, `tpuFptosi`, …)
    /// are exposed separately for callers that need rounding-mode control —
    /// pass `opts.rounding` to opt in.
    pub fn cast(self: *Builder, src: Value, dt: DType) Value {
        return self.castOpts(src, dt, .{});
    }
    pub fn castOpts(self: *Builder, src: Value, dt: DType, opts: CastOpts) Value {
        if (opts.bitcast) return self.bitcastTo(src, dt);
        const cur_elem = src.elemType();
        const tgt_elem = dt.toMlir(self.ctx);
        if (cur_elem.eql(tgt_elem)) return src;
        const cur_dtype = self.mlirElemToDType(cur_elem);
        const cur_is_float = src.isFloatElem();
        const tgt_is_float = isFloatDtype(dt);
        if (cur_is_float and tgt_is_float) {
            const cur_bw = dtypeBitwidth(cur_dtype);
            const tgt_bw = dtypeBitwidth(dt);
            if (tgt_bw > cur_bw) return self.arithExtf(src, dt);
            return self.arithTruncf(src, dt);
        }
        if (cur_is_float and !tgt_is_float) return self.arithFptosi(src, dt);
        if (!cur_is_float and tgt_is_float) {
            if (cur_dtype == .i1) return self.arithUitofp(src, dt);
            return self.arithSitofp(src, dt);
        }
        const cur_bw = dtypeBitwidth(cur_dtype);
        const tgt_bw = dtypeBitwidth(dt);
        if (tgt_bw > cur_bw) return self.extsi(src, dt);
        if (tgt_bw < cur_bw) return self.trunci(src, dt);
        return self.arithBitcast(src, dt);
    }

    // ==================== math dialect ====================

    pub fn exp(self: *Builder, x: Value) Value {
        return self.emit(math.exp(self.ctx, x.inner, self.loc()));
    }
    /// `exp2(x)` → `exp(x * ln(2))`. Matches Pallas's TPU lowering rule
    /// (`jax/_src/pallas/mosaic/lowering.py:_exp2_lowering_rule`), which
    /// expands `lax.exp2_p` into the same `x * ln2` pattern in forward-compat
    /// mode. Materializes as an `arith.constant` (matching x's element type)
    /// + `arith.mulf x, ln2` + `math.exp`. For the raw `math.exp2` op
    /// (post-2025-07-26 cloud TPUs), use `mathExp2`.
    pub fn exp2(self: *Builder, x: Value) Value {
        const ln2 = self.liftAs(@as(f64, 0.6931471805599453), self.mlirElemToDType(x.elemType()));
        const ln2_bc = if (x.isShaped()) self.broadcastTo(ln2, x.shape().constSlice()) else ln2;
        return self.exp(self.mulf(x, ln2_bc));
    }
    pub fn mathExp2(self: *Builder, x: Value) Value {
        return self.emit(math.exp2(self.ctx, x.inner, self.loc()));
    }
    pub fn log(self: *Builder, x: Value) Value {
        return self.emit(math.log(self.ctx, x.inner, self.loc()));
    }
    pub fn log2(self: *Builder, x: Value) Value {
        return self.emit(math.log2(self.ctx, x.inner, self.loc()));
    }
    pub fn sqrt(self: *Builder, x: Value) Value {
        return self.emit(math.sqrt(self.ctx, x.inner, self.loc()));
    }
    pub fn rsqrt(self: *Builder, x: Value) Value {
        return self.emit(math.rsqrt(self.ctx, x.inner, self.loc()));
    }
    pub fn sin(self: *Builder, x: Value) Value {
        return self.emit(math.sin(self.ctx, x.inner, self.loc()));
    }
    pub fn cos(self: *Builder, x: Value) Value {
        return self.emit(math.cos(self.ctx, x.inner, self.loc()));
    }
    pub fn tanh(self: *Builder, x: Value) Value {
        return self.emit(math.tanh(self.ctx, x.inner, self.loc()));
    }
    pub fn erf(self: *Builder, x: Value) Value {
        return self.emit(math.erf(self.ctx, x.inner, self.loc()));
    }
    pub fn absf(self: *Builder, x: Value) Value {
        return self.emit(math.absf(self.ctx, x.inner, self.loc()));
    }
    pub fn absi(self: *Builder, x: Value) Value {
        return self.emit(math.absi(self.ctx, x.inner, self.loc()));
    }
    pub fn abs(self: *Builder, x: Value) Value {
        if (x.isFloatElem()) return self.absf(x);
        return self.absi(x);
    }
    pub fn floor(self: *Builder, x: Value) Value {
        return self.emit(math.floor(self.ctx, x.inner, self.loc()));
    }
    pub fn ceil(self: *Builder, x: Value) Value {
        return self.emit(math.ceil(self.ctx, x.inner, self.loc()));
    }
    pub fn powf(self: *Builder, x: Value, y: Value) Value {
        return self.emit(math.powf(self.ctx, x.inner, y.inner, self.loc()));
    }
    pub fn fma(self: *Builder, a: Value, b: Value, c_: Value) Value {
        return self.emit(math.fma(self.ctx, a.inner, b.inner, c_.inner, self.loc()));
    }

    // ==================== TPU memory: vector_load / vector_store / scalar load/store ====================

    /// Build the result vector type for a full-shape `tpu.vector_load` from
    /// the input ref's shape + element type.
    fn refVectorType(self: *Builder, ref: Value) *const mlir.Type {
        _ = self;
        const mr = ref.type_().isA(mlir.MemRefType) orelse @panic("Builder.refVectorType: not a memref");
        var s: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (0..mr.rank()) |i| s.appendAssumeCapacity(mr.dimension(i));
        return mlir.vectorType(s.constSlice(), mr.elementType());
    }

    /// Full-shape `vector.load`. Use `tpuVectorLoad` for sublane stride / masking.
    pub fn refLoad(self: *Builder, ref: Value) Value {
        const result_ty = self.refVectorType(ref);
        const idx_inner = self.indicesOrZero(ref, &.{});
        return self.emit(vector.load(self.ctx, ref.inner, idx_inner, result_ty, .{}, self.loc()));
    }

    /// `vector.load(ref, indices)` — explicit `index`-typed offsets.
    pub fn vectorLoadAt(self: *Builder, ref: Value, indices: []const Value) Value {
        const result_ty = self.refVectorType(ref);
        const idx_inner = self.indicesOrZero(ref, indices);
        return self.emit(vector.load(self.ctx, ref.inner, idx_inner, result_ty, .{}, self.loc()));
    }

    /// `tpu.vector_load` — sublane stride + optional element-wise mask.
    pub fn tpuVectorLoad(self: *Builder, ref: Value, indices: []const Value, opts: VectorLoadOpts) Value {
        const ctx = self.ctx;
        const result_ty = self.refVectorType(ref);
        const idx_inner = self.indicesOrZero(ref, indices);
        const mask_inner: ?*const mlir.Value = if (opts.mask) |m| m.inner else null;
        return self.emit(tpu.vector_load(
            ctx,
            ref.inner,
            idx_inner,
            mask_inner,
            .{ .strides = opts.strides },
            result_ty,
            self.loc(),
        ));
    }

    /// `tpu.vector_store(value, ref)` — full-shape vector store.
    pub fn refStore(self: *Builder, ref: Value, value: Value) void {
        self.refStoreOpts(ref, value, &.{}, .{});
    }

    pub fn vectorStoreAt(self: *Builder, ref: Value, value: Value, indices: []const Value) void {
        self.refStoreOpts(ref, value, indices, .{});
    }

    pub fn refStoreOpts(self: *Builder, ref: Value, value: Value, indices: []const Value, opts: VectorStoreOpts) void {
        const idx_inner = self.indicesOrZero(ref, indices);
        const mask_inner: ?*const mlir.Value = if (opts.mask) |m| m.inner else null;
        _ = tpu.vector_store(
            self.ctx,
            value.inner,
            ref.inner,
            idx_inner,
            mask_inner,
            .{ .strides = opts.strides, .add = opts.add },
            self.loc(),
        ).appendTo(self.currentBlock());
    }

    /// `memref.load` — scalar load. Indices must be `index`-typed.
    pub fn scalarLoad(self: *Builder, ref: Value, indices: []const Value) Value {
        const idx_inner = self.indicesOrZero(ref, indices);
        return self.emit(memref.load(self.ctx, ref.inner, idx_inner, .{}, self.loc()));
    }

    /// `memref.store` — scalar store.
    pub fn scalarStore(self: *Builder, ref: Value, value: Value, indices: []const Value) void {
        const idx_inner = self.indicesOrZero(ref, indices);
        _ = memref.store(self.ctx, value.inner, ref.inner, idx_inner, .{}, self.loc()).appendTo(self.currentBlock());
    }

    /// Returns a `*const mlir.Value` slice of length `ref.rank()`, padding with
    /// a cached `arith.constant 0 : index` when `indices` is empty.
    fn indicesOrZero(self: *Builder, ref: Value, indices: []const Value) []const *const mlir.Value {
        const r = ref.rank();
        if (indices.len == r) return self.innerSlice(indices);
        std.debug.assert(indices.len == 0);
        const zero = self.zeroIndex();
        const out = self.arena.allocator().alloc(*const mlir.Value, r) catch @panic("Builder.indicesOrZero OOM");
        for (out) |*p| p.* = zero.inner;
        return out;
    }

    /// Return a cached `arith.constant 0 : index` Value, lazily emitting
    /// it the first time it's needed. The cache is invalidated whenever
    /// the insertion block changes — `index` constants are scoped to the
    /// block they're emitted in.
    fn zeroIndex(self: *Builder) Value {
        const block = self.currentBlock();
        if (self.zero_index) |z| {
            if (self.zero_index_block == block) return z;
        }
        const z = self.cIndex(0);
        self.zero_index = z;
        self.zero_index_block = block;
        return z;
    }

    /// `tpu.strided_load` — multi-dim strided load. Result shape is derived
    /// from `ref.shape / strides` (per-dim ceil divide).
    pub fn stridedLoad(self: *Builder, ref: Value, indices: []const Value, strides: []const i32) Value {
        const mr = ref.type_().isA(mlir.MemRefType) orelse @panic("stridedLoad: not a memref");
        var s: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (0..mr.rank()) |i| {
            const stride: i64 = if (i < strides.len) strides[i] else 1;
            const dim = mr.dimension(i);
            s.appendAssumeCapacity(@divTrunc(dim + stride - 1, stride));
        }
        const result_ty = mlir.vectorType(s.constSlice(), mr.elementType());
        return self.emit(tpu.strided_load(self.ctx, ref.inner, self.innerSlice(indices), strides, result_ty, self.loc()));
    }

    /// `tpu.strided_load` with an explicit result shape — when you know the
    /// load shape doesn't match the simple `ref.shape / strides` form (e.g.
    /// inner-dim crop).
    pub fn stridedLoadShape(self: *Builder, ref: Value, indices: []const Value, strides: []const i32, result_shape: []const i64) Value {
        const mr = ref.type_().isA(mlir.MemRefType) orelse @panic("stridedLoadShape: not a memref");
        const result_ty = mlir.vectorType(result_shape, mr.elementType());
        return self.emit(tpu.strided_load(self.ctx, ref.inner, self.innerSlice(indices), strides, result_ty, self.loc()));
    }

    /// `tpu.strided_store` — multi-dim strided store.
    pub fn stridedStore(self: *Builder, ref: Value, value: Value, indices: []const Value, strides: []const i32) void {
        _ = tpu.strided_store(self.ctx, value.inner, ref.inner, self.innerSlice(indices), strides, self.loc()).appendTo(self.currentBlock());
    }

    // ==================== TPU compute: matmul / iota / reciprocal ====================

    /// `tpu.matmul(lhs, rhs, acc)`. Result type is taken from `acc`. Use
    /// `matmulOpts` to set transposes / precision / dimension_numbers.
    pub fn matmul(self: *Builder, lhs: Value, rhs: Value, acc: Value) Value {
        return self.matmulOpts(lhs, rhs, acc, .{});
    }

    pub fn matmulOpts(self: *Builder, lhs: Value, rhs: Value, acc: Value, opts: MatmulOpts) Value {
        return self.emit(tpu.matmul(
            self.ctx,
            lhs.inner,
            rhs.inner,
            acc.inner,
            .{
                .transpose_lhs = opts.transpose_lhs,
                .transpose_rhs = opts.transpose_rhs,
                .precision = opts.precision,
                .dimension_numbers = opts.dimension_numbers,
            },
            acc.type_(),
            self.loc(),
        ));
    }

    /// `tpu.iota` over the given shape / dimensions. Pass `null` for
    /// `dimensions` to use the default.
    pub fn iota(self: *Builder, shape: []const i64, dtype: DType, dimensions: ?[]const i32) Value {
        const ty = mlir.vectorType(shape, dtype.toMlir(self.ctx));
        return self.emit(tpu.iota(self.ctx, dimensions, ty, self.loc()));
    }

    /// 1-D `arange(0, n)`. TPU layout pass requires rank ≥ 2 with the lane dim
    /// trailing, so this emits a `<1xN>` `tpu.iota` then `shape_cast` to `<N>`.
    pub fn arange(self: *Builder, n: i64, dtype: DType) Value {
        const elem = dtype.toMlir(self.ctx);
        const ty_2d = mlir.vectorType(&.{ 1, n }, elem);
        const iota_2d = self.emit(tpu.iota(self.ctx, &.{1}, ty_2d, self.loc()));
        return self.shapeCast(iota_2d, &.{n});
    }

    /// `tpu.reciprocal(x)` with default options.
    pub fn reciprocal(self: *Builder, x: Value) Value {
        return self.reciprocalOpts(x, .{});
    }

    pub fn reciprocalOpts(self: *Builder, x: Value, opts: ReciprocalOpts) Value {
        return self.emit(tpu.reciprocal(self.ctx, x.inner, .{
            .approx = opts.approx,
            .full_range = opts.full_range,
        }, self.loc()));
    }

    // ==================== Vector dialect helpers ====================
    //
    // The `tpu.*` shape ops (`reshape`, `repeat`, `transpose`,
    // `broadcastInSublanes`) are Mosaic-specific layout-aware variants.

    /// `vector.broadcast` to `shape`. For Zig scalar literals prefer `splat` (dense-splat fast path).
    pub fn broadcastTo(self: *Builder, src: Value, shape: []const i64) Value {
        const ty = mlir.vectorType(shape, src.elemType());
        return self.emit(vector.broadcast(self.ctx, src.inner, ty, self.loc()));
    }

    /// `vector.shape_cast` — reshape preserving total lanes. `reshape` emits `tpu.reshape` instead.
    pub fn shapeCast(self: *Builder, src: Value, new_shape: []const i64) Value {
        const ty = mlir.vectorType(new_shape, src.elemType());
        return self.emit(vector.shape_cast(self.ctx, src.inner, ty, self.loc()));
    }

    /// `vector.extract` — extract a scalar (or lower-rank vector) at a
    /// static position. Result shape = `src.shape[position.len..]`.
    pub fn vectorExtract(self: *Builder, src: Value, position: []const i64) Value {
        const sh = src.shape();
        var out: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (position.len..sh.len) |i| out.appendAssumeCapacity(sh.get(i));
        const ty: *const mlir.Type = if (out.len == 0)
            src.elemType()
        else
            mlir.vectorType(out.constSlice(), src.elemType());
        return self.emit(vector.extract(self.ctx, src.inner, position, &.{}, ty, self.loc()));
    }

    /// `vector.reduction <kind>` — reduce a 1-D vector to a scalar.
    /// Emits the bare upstream op; on real TPU the layout pass may reject
    /// 1-D vectors, so prefer `reduceToScalar` for kernel-level reductions.
    pub fn vectorReductionFlat(self: *Builder, kind: CombiningKind, src: Value) Value {
        return self.emit(vector.reduction(self.ctx, kind, src.inner, src.elemType(), .{}, self.loc()));
    }

    /// 1-D vector → scalar. TPU vectors are 2-D, so this adds a leading
    /// singleton before reducing: shape_cast<N>→<1xN>, multi_reduction[1]→<1>, extract[0].
    /// `acc` is the reduction identity (e.g. `k.zeros(&.{1}, .f32)` for sum).
    pub fn reduceToScalar(self: *Builder, kind: CombiningKind, src: Value, acc: Value) Value {
        std.debug.assert(src.rank() == 1);
        const n = src.shape().get(0);
        const src_2d = self.shapeCast(src, &.{ 1, n });
        const reduced = self.multiReduction(kind, src_2d, acc, &.{1});
        return self.vectorExtract(reduced, &.{0});
    }

    /// `vector.multi_reduction <kind>` — reduce a multi-D vector along
    /// the given axes with an explicit accumulator. Result rank =
    /// `src.rank - axes.len`.
    pub fn multiReduction(
        self: *Builder,
        kind: CombiningKind,
        src: Value,
        acc: Value,
        axes: []const i64,
    ) Value {
        const sh = src.shape();
        var out: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        outer: for (0..sh.len) |i| {
            for (axes) |a| if (a == @as(i64, @intCast(i))) continue :outer;
            out.appendAssumeCapacity(sh.get(i));
        }
        const ty: *const mlir.Type = if (out.len == 0)
            src.elemType()
        else
            mlir.vectorType(out.constSlice(), src.elemType());
        return self.emit(vector.multi_reduction(self.ctx, kind, src.inner, acc.inner, axes, ty, self.loc()));
    }

    /// `vector.load` with an explicit result shape — for `ref[i, j, :, k:k+BLOCK]`-style sliced loads.
    pub fn vectorLoadShape(self: *Builder, ref: Value, indices: []const Value, shape: []const i64) Value {
        const idx_inner = self.indicesOrZero(ref, indices);
        const ty = mlir.vectorType(shape, ref.elemType());
        return self.emit(vector.load(self.ctx, ref.inner, idx_inner, ty, .{}, self.loc()));
    }

    // ==================== TPU shape ====================

    pub fn reshape(self: *Builder, src: Value, new_shape: []const i64) Value {
        const ty = mlir.vectorType(new_shape, src.elemType());
        return self.emit(tpu.reshape(self.ctx, src.inner, ty, self.loc()));
    }

    pub fn repeat(self: *Builder, src: Value, dimension: i32, times: i32) Value {
        const in_shape = src.shape();
        var out: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (0..in_shape.len) |i| {
            const d = in_shape.get(i);
            out.appendAssumeCapacity(if (i == @as(usize, @intCast(dimension))) d * times else d);
        }
        const ty = mlir.vectorType(out.constSlice(), src.elemType());
        return self.emit(tpu.repeat(self.ctx, src.inner, dimension, times, ty, self.loc()));
    }

    pub fn concatenate(self: *Builder, sources: []const Value, dimension: i32) Value {
        std.debug.assert(sources.len > 0);
        const ax: usize = @intCast(dimension);
        var sum: i64 = 0;
        for (sources) |s| sum += s.dim(ax);
        const head = sources[0].shape();
        var out: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (0..head.len) |i| out.appendAssumeCapacity(if (i == ax) sum else head.get(i));
        const ty = mlir.vectorType(out.constSlice(), sources[0].elemType());
        return self.emit(tpu.concatenate(self.ctx, self.innerSlice(sources), dimension, ty, self.loc()));
    }

    pub fn transpose(self: *Builder, src: Value, permutation: []const i32) Value {
        const in_shape = src.shape();
        std.debug.assert(permutation.len == in_shape.len);
        var out: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (permutation) |p| out.appendAssumeCapacity(in_shape.get(@intCast(p)));
        const ty = mlir.vectorType(out.constSlice(), src.elemType());
        return self.emit(tpu.transpose(self.ctx, src.inner, permutation, ty, self.loc()));
    }

    pub fn broadcastInSublanes(self: *Builder, src: Value, lane: i32) Value {
        // Result shape = src shape (the op replicates within sublanes).
        return self.emit(tpu.broadcast_in_sublanes(self.ctx, src.inner, lane, src.type_(), self.loc()));
    }

    pub fn rotate(self: *Builder, value: Value, opts: RotateOpts) Value {
        return self.emit(tpu.rotate(self.ctx, value.inner, .{
            .amount = opts.amount,
            .dimension = opts.dimension,
            .stride = opts.stride,
            .stride_dimension = opts.stride_dimension,
        }, self.loc()));
    }

    // ==================== TPU reductions / scan / sort ====================

    /// `tpu.all_reduce(input, dim, kind)` over a single dim.
    pub fn allReduce(self: *Builder, input: Value, kind: ReductionKind, dim: i64) Value {
        const in_shape = input.shape();
        var out: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (0..in_shape.len) |i| out.appendAssumeCapacity(if (i == @as(usize, @intCast(dim))) 1 else in_shape.get(i));
        const ty = mlir.vectorType(out.constSlice(), input.elemType());
        return self.emit(tpu.all_reduce(self.ctx, input.inner, dim, kind, ty, self.loc()));
    }

    /// `tpu.reduce_index(input, kind, axis)` — drop axis, replace with
    /// `index`-valued indices (integer).
    pub fn reduceIndex(self: *Builder, input: Value, kind: ReductionKind, axis: i32) Value {
        const in_shape = input.shape();
        var out: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (in_shape.constSlice(), 0..) |d, i| {
            if (@as(i32, @intCast(i)) != axis) out.appendAssumeCapacity(d);
        }
        const ty = mlir.vectorType(out.constSlice(), mlir.integerType(self.ctx, .i32));
        return self.emit(tpu.reduce_index(self.ctx, input.inner, axis, kind, ty, self.loc()));
    }

    /// `tpu.scan(input, kind, mask?)` — same shape as input.
    pub fn scan(self: *Builder, input: Value, kind: ReductionKind, mask: ?Value) Value {
        const m_inner: ?*const mlir.Value = if (mask) |m| m.inner else null;
        return self.emit(tpu.scan(self.ctx, input.inner, kind, m_inner, input.type_(), self.loc()));
    }

    /// `tpu.sort(keys, values, mask?)` — returns (output_mask, sorted_keys, sorted_values).
    pub fn sort(self: *Builder, keys: Value, values: Value, mask: ?Value, opts: SortOpts) [3]Value {
        const m_inner: ?*const mlir.Value = if (mask) |m| m.inner else null;
        const out_mask_ty = if (mask) |m| m.type_() else blk: {
            // Default: vector<keys.shape x i1>.
            break :blk mlir.vectorType(keys.shape().constSlice(), mlir.integerType(self.ctx, .i1));
        };
        const op = tpu.sort(
            self.ctx,
            keys.inner,
            values.inner,
            m_inner,
            .{ .descending = opts.descending },
            out_mask_ty,
            keys.type_(),
            values.type_(),
            self.loc(),
        );
        _ = op.appendTo(self.currentBlock());
        return .{
            .{ .inner = op.result(0), .kernel = self },
            .{ .inner = op.result(1), .kernel = self },
            .{ .inner = op.result(2), .kernel = self },
        };
    }

    // ==================== TPU memref-shape ops ====================

    pub fn memRefSlice(
        self: *Builder,
        mem_ref: Value,
        base_idx: []const Value,
        result_shape: []const i64,
        dynamic_sizes: []const Value,
    ) Value {
        const mr = mem_ref.type_().isA(mlir.MemRefType) orelse @panic("memRefSlice: not a memref");
        const result_ty = mlir.memRefType(mr.elementType(), result_shape, null, mr.memorySpace());
        return self.emit(tpu.memref_slice(self.ctx, mem_ref.inner, .{
            .base_idx = self.innerSlice(base_idx),
            .dynamic_sizes = self.innerSlice(dynamic_sizes),
        }, result_ty, self.loc()));
    }

    pub fn memRefSqueeze(self: *Builder, mem_ref: Value, result_shape: []const i64) Value {
        const mr = mem_ref.type_().isA(mlir.MemRefType) orelse @panic("memRefSqueeze: not a memref");
        const result_ty = mlir.memRefType(mr.elementType(), result_shape, null, mr.memorySpace());
        return self.emit(tpu.memref_squeeze(self.ctx, mem_ref.inner, result_ty, self.loc()));
    }

    pub fn memRefReshape(self: *Builder, mem_ref: Value, result_shape: []const i64) Value {
        const mr = mem_ref.type_().isA(mlir.MemRefType) orelse @panic("memRefReshape: not a memref");
        const result_ty = mlir.memRefType(mr.elementType(), result_shape, null, mr.memorySpace());
        return self.emit(tpu.memref_reshape(self.ctx, mem_ref.inner, result_ty, self.loc()));
    }

    pub fn memRefBitcast(self: *Builder, mem_ref: Value, dt: DType) Value {
        const mr = mem_ref.type_().isA(mlir.MemRefType) orelse @panic("memRefBitcast: not a memref");
        var s: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (0..mr.rank()) |i| s.appendAssumeCapacity(mr.dimension(i));
        const result_ty = mlir.memRefType(dt.toMlir(self.ctx), s.constSlice(), null, mr.memorySpace());
        return self.emit(tpu.memref_bitcast(self.ctx, mem_ref.inner, result_ty, self.loc()));
    }

    /// `tpu.memref_bitcast` with an explicit result shape — needed for the
    /// `bf16 → i32` packing trick in ragged-paged-attention's
    /// `strided_load_kv` path (`<64x128xbf16>` → `<32x128xi32>`).
    pub fn memRefBitcastShape(self: *Builder, mem_ref: Value, result_shape: []const i64, dt: DType) Value {
        const mr = mem_ref.type_().isA(mlir.MemRefType) orelse @panic("memRefBitcastShape: not a memref");
        const result_ty = mlir.memRefType(dt.toMlir(self.ctx), result_shape, null, mr.memorySpace());
        return self.emit(tpu.memref_bitcast(self.ctx, mem_ref.inner, result_ty, self.loc()));
    }

    pub fn reinterpretCast(self: *Builder, mem_ref: Value, result_shape: []const i64, dt: DType) Value {
        const mr = mem_ref.type_().isA(mlir.MemRefType) orelse @panic("reinterpretCast: not a memref");
        const result_ty = mlir.memRefType(dt.toMlir(self.ctx), result_shape, null, mr.memorySpace());
        return self.emit(tpu.reinterpret_cast(self.ctx, mem_ref.inner, result_ty, self.loc()));
    }

    pub fn assumeLayout(self: *Builder, src: Value) Value {
        return self.emit(tpu.assume_layout(self.ctx, src.inner, src.type_(), self.loc()));
    }

    pub fn eraseMemRefLayout(self: *Builder, mem_ref: Value) Value {
        const mr = mem_ref.type_().isA(mlir.MemRefType) orelse @panic("eraseMemRefLayout: not a memref");
        var s: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (0..mr.rank()) |i| s.appendAssumeCapacity(mr.dimension(i));
        // Result drops the layout but keeps shape, element type, memory space.
        const result_ty = mlir.memRefType(mr.elementType(), s.constSlice(), null, mr.memorySpace());
        return self.emit(tpu.erase_memref_layout(self.ctx, mem_ref.inner, result_ty, self.loc()));
    }

    // ==================== Mask building ====================

    /// `tpu.create_mask` — 2-sided mask `[low, high)` per dim.
    pub fn createMask(self: *Builder, low_bounds: []const Value, high_bounds: []const Value, shape: []const i64) Value {
        const ty = mlir.vectorType(shape, mlir.integerType(self.ctx, .i1));
        return self.emit(tpu.create_mask(self.ctx, self.innerSlice(low_bounds), self.innerSlice(high_bounds), ty, self.loc()));
    }

    /// Convenience: 1-sided range mask `(0 <= idx < limit)` — emits two
    /// `index` constants and a `tpu.create_mask`.
    pub fn rangeMask(self: *Builder, limits: []const Value, shape: []const i64) Value {
        const zero = self.cIndex(0);
        const lows = self.arena.allocator().alloc(Value, limits.len) catch @panic("rangeMask OOM");
        for (lows) |*l| l.* = zero;
        return self.createMask(lows, limits, shape);
    }

    // ==================== Semaphores / DMA / barriers ====================

    pub fn semAlloc(self: *Builder, kind: ArgSpec.SemKind) Value {
        const ty = switch (kind) {
            .regular => tpu.semaphoreType(self.ctx),
            .dma => tpu.dmaSemaphoreType(self.ctx),
        };
        return self.emit(tpu.sem_alloc(self.ctx, ty, self.loc()));
    }

    pub fn semBarrier(self: *Builder) Value {
        const ty = tpu.dmaSemaphoreType(self.ctx);
        return self.emit(tpu.sem_barrier(self.ctx, ty, self.loc()));
    }

    pub fn semRead(self: *Builder, semaphore: Value) Value {
        return self.emit(tpu.sem_read(self.ctx, semaphore.inner, self.loc()));
    }

    pub fn semWait(self: *Builder, semaphore: Value, amount: anytype) void {
        const a = if (@TypeOf(amount) == Value) amount else self.lift(amount);
        _ = tpu.sem_wait(self.ctx, semaphore.inner, a.inner, self.loc()).appendTo(self.currentBlock());
    }

    pub fn semSignal(self: *Builder, semaphore: Value, amount: anytype) void {
        self.semSignalOpts(semaphore, amount, .{});
    }

    pub fn semSignalOpts(self: *Builder, semaphore: Value, amount: anytype, opts: SemSignalOpts) void {
        const a = if (@TypeOf(amount) == Value) amount else self.lift(amount);
        const dev: ?*const mlir.Value = if (opts.device_id) |d| d.inner else null;
        const core: ?*const mlir.Value = if (opts.core_id) |c_| c_.inner else null;
        _ = tpu.sem_signal(self.ctx, semaphore.inner, a.inner, .{
            .device_id = dev,
            .core_id = core,
        }, self.loc()).appendTo(self.currentBlock());
    }

    pub fn barrier(self: *Builder, barrier_id: Value) void {
        _ = tpu.barrier(self.ctx, barrier_id.inner, self.loc()).appendTo(self.currentBlock());
    }

    pub fn enqueueDma(self: *Builder, source: Value, target: Value, target_semaphore: Value, opts: EnqueueDmaOpts) void {
        const src_sem: ?*const mlir.Value = if (opts.source_semaphore) |s| s.inner else null;
        const dev: ?*const mlir.Value = if (opts.device_id) |d| d.inner else null;
        const core: ?*const mlir.Value = if (opts.core_id) |c_| c_.inner else null;
        _ = tpu.enqueue_dma(self.ctx, source.inner, target.inner, target_semaphore.inner, .{
            .source_semaphore = src_sem,
            .device_id = dev,
            .core_id = core,
            .priority = opts.priority,
            .strict_ordering = opts.strict_ordering,
        }, self.loc()).appendTo(self.currentBlock());
    }

    pub fn waitDma2(self: *Builder, semaphore: Value, src: Value, dst: Value, opts: WaitDma2Opts) void {
        const dev: ?*const mlir.Value = if (opts.device_id) |d| d.inner else null;
        const core: ?*const mlir.Value = if (opts.core_id) |c_| c_.inner else null;
        _ = tpu.wait_dma2(self.ctx, semaphore.inner, src.inner, dst.inner, .{
            .device_id = dev,
            .core_id = core,
            .strict_ordering = opts.strict_ordering,
        }, self.loc()).appendTo(self.currentBlock());
    }

    pub fn deviceId(self: *Builder) Value {
        return self.emit(tpu.device_id(self.ctx, self.loc()));
    }

    /// `tpu.trace_start` — pair with `traceStop`. Used by Pallas to mark
    /// einsum boundaries (e.g. `"nd,md->nm"`).
    pub fn traceStart(self: *Builder, level: i32, message: []const u8) void {
        _ = tpu.trace_start(self.ctx, level, message, self.loc()).appendTo(self.currentBlock());
    }

    pub fn traceStop(self: *Builder) void {
        _ = tpu.trace_stop(self.ctx, self.loc()).appendTo(self.currentBlock());
    }

    /// Build a `#tpu.dot_dimension_numbers<...>` attribute for `matmulOpts`.
    pub fn dotDimensionNumbers(
        self: *Builder,
        lhs_contracting: []const i64,
        rhs_contracting: []const i64,
        lhs_non_contracting: []const i64,
        rhs_non_contracting: []const i64,
        output_dim_order: []const i64,
        lhs_batch: []const i64,
        rhs_batch: []const i64,
    ) *const mlir.Attribute {
        return tpu.dotDimensionNumbers(
            self.ctx,
            lhs_contracting,
            rhs_contracting,
            lhs_non_contracting,
            rhs_non_contracting,
            output_dim_order,
            lhs_batch,
            rhs_batch,
        );
    }

    pub fn delay(self: *Builder, cycles: anytype) void {
        const c_ = if (@TypeOf(cycles) == Value) cycles else self.lift(cycles);
        _ = tpu.delay(self.ctx, c_.inner, self.loc()).appendTo(self.currentBlock());
    }

    // ==================== SCF regions ====================

    /// Open a scoped `scf.for`. `lower`/`upper`/`step` accept Values, comptime
    /// ints, or runtime Zig ints — coerced to `index`. `inits` is `.{v1, v2, ...}`.
    pub fn openFor(
        self: *Builder,
        lower: anytype,
        upper: anytype,
        step: anytype,
        inits: anytype,
    ) dsl.ForScope(Builder, Value, dsl.tupleArity(@TypeOf(inits), "openFor: inits")) {
        const N = comptime dsl.tupleArity(@TypeOf(inits), "openFor: inits");
        const fields = @typeInfo(@TypeOf(inits)).@"struct".fields;

        // scf.for bounds default to `index`.
        const lb_v: Value = if (@TypeOf(lower) == Value) lower else self.cIndex(@intCast(lower));
        const ub_v: Value = if (@TypeOf(upper) == Value) upper else self.cIndex(@intCast(upper));
        const step_v: Value = if (@TypeOf(step) == Value) step else self.cIndex(@intCast(step));

        var block_types: [N + 1]*const mlir.Type = undefined;
        var block_locs: [N + 1]*const mlir.Location = undefined;
        block_types[0] = lb_v.type_();
        block_locs[0] = self.loc();
        var inits_inner: [N]*const mlir.Value = undefined;
        inline for (fields, 0..) |f, i| {
            const raw = @field(inits, f.name);
            const v: Value = if (f.type == Value) raw else self.lift(raw);
            block_types[i + 1] = v.type_();
            block_locs[i + 1] = self.loc();
            inits_inner[i] = v.inner;
        }

        const body = mlir.Block.init(&block_types, &block_locs);
        self.pushBlock(body);

        var carried: [N]Value = undefined;
        for (0..N) |i| carried[i] = .{ .inner = body.argument(i + 1), .kernel = self };

        return .{
            .kernel = self,
            .body = body,
            .lb_inner = lb_v.inner,
            .ub_inner = ub_v.inner,
            .step_inner = step_v.inner,
            .inits_inner = inits_inner,
            .iv = .{ .inner = body.argument(0), .kernel = self },
            .carried = carried,
        };
    }

    pub fn openIf(self: *Builder, cond: Value) dsl.IfOnlyScope(Builder, Value) {
        const then_block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(then_block);
        return .{
            .kernel = self,
            .cond_inner = cond.inner,
            .then_block = then_block,
        };
    }

    pub fn openIfElse(
        self: *Builder,
        cond: Value,
        result_types: anytype,
    ) dsl.IfScope(Builder, Value, dsl.tupleArity(@TypeOf(result_types), "openIfElse: result_types")) {
        const N = comptime dsl.tupleArity(@TypeOf(result_types), "openIfElse: result_types");
        const fields = @typeInfo(@TypeOf(result_types)).@"struct".fields;

        var types: [N]*const mlir.Type = undefined;
        inline for (fields, 0..) |f, i| {
            if (f.type != *const mlir.Type)
                @compileError("openIfElse: every result_type must be *const mlir.Type (use k.scalarTy/vectorTy)");
            types[i] = @field(result_types, f.name);
        }

        const then_block = mlir.Block.init(&.{}, &.{});
        const else_block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(then_block);

        return .{
            .kernel = self,
            .cond_inner = cond.inner,
            .then_block = then_block,
            .else_block = else_block,
            .result_types = types,
        };
    }

    pub fn openWhile(
        self: *Builder,
        inits: anytype,
        after_types: anytype,
    ) dsl.WhileScope(
        Builder,
        Value,
        dsl.tupleArity(@TypeOf(inits), "openWhile: inits"),
        dsl.tupleArity(@TypeOf(after_types), "openWhile: after_types"),
    ) {
        const N = comptime dsl.tupleArity(@TypeOf(inits), "openWhile: inits");
        const M = comptime dsl.tupleArity(@TypeOf(after_types), "openWhile: after_types");
        const init_fields = @typeInfo(@TypeOf(inits)).@"struct".fields;
        const type_fields = @typeInfo(@TypeOf(after_types)).@"struct".fields;

        var before_types: [N]*const mlir.Type = undefined;
        var before_locs: [N]*const mlir.Location = undefined;
        var inits_inner: [N]*const mlir.Value = undefined;
        inline for (init_fields, 0..) |f, i| {
            if (f.type != Value)
                @compileError("openWhile: every init must be a Value");
            const v: Value = @field(inits, f.name);
            before_types[i] = v.type_();
            before_locs[i] = self.loc();
            inits_inner[i] = v.inner;
        }
        const before_block = mlir.Block.init(&before_types, &before_locs);

        var after_tys: [M]*const mlir.Type = undefined;
        var after_locs: [M]*const mlir.Location = undefined;
        inline for (type_fields, 0..) |f, i| {
            if (f.type != *const mlir.Type)
                @compileError("openWhile: every after_type must be *const mlir.Type");
            after_tys[i] = @field(after_types, f.name);
            after_locs[i] = self.loc();
        }
        const after_block = mlir.Block.init(&after_tys, &after_locs);

        self.pushBlock(before_block);
        var before_carried: [N]Value = undefined;
        for (0..N) |i| before_carried[i] = .{ .inner = before_block.argument(i), .kernel = self };

        return .{
            .kernel = self,
            .before_block = before_block,
            .after_block = after_block,
            .inits_inner = inits_inner,
            .after_types = after_tys,
            .before_carried = before_carried,
        };
    }

    // ==================== finalization ====================

    /// Append `func.return` with the given results, verify the module, and
    /// serialize to a NUL-terminated MLIR string owned by the kernel's
    /// allocator. Caller must `defer allocator.free(ir)`.
    pub fn finish(self: *Builder, results: []const Value) FinishError![:0]const u8 {
        return self.finishOpts(results, .{});
    }

    pub const FinishOpts = struct {
        /// Run `canonicalize,cse` on the module before serializing — lets
        /// the user code stay one-line-per-Pallas-line and lean on MLIR
        /// to dedupe duplicated constants / identical subexpressions, the
        /// same way Pallas does (its lowering doesn't dedupe either; the
        /// IR you read in `debug=True` is post-canonicalize).
        canonicalize: bool = false,
    };

    pub fn finishOpts(self: *Builder, results: []const Value, opts: FinishOpts) FinishError![:0]const u8 {
        const current = self.currentBlock();
        _ = func.returns(self.ctx, self.innerSlice(results), self.loc()).appendTo(current);

        if (self.emit_transform_stubs) self.appendTransformStubs();

        if (!self.module.operation().verify()) {
            return error.InvalidMlir;
        }

        if (opts.canonicalize) {
            const pm = mlir.PassManager.init(self.ctx);
            defer pm.deinit();
            // `canonicalize` folds `x * 1`, `x + 0`, etc.; `cse` then
            // dedupes identical constants and subexpressions. Same
            // pipeline `zml/module.zig` runs on every compiled module.
            const opm = pm.asOpPassManager();
            inline for (.{ "canonicalize", "cse", "canonicalize" }) |pass| {
                try opm.addPipeline(pass);
            }
            try pm.runOnOp(self.module.operation());
        }

        var al: std.Io.Writer.Allocating = .init(self.allocator);
        defer al.deinit();
        try al.writer.print("{f}", .{self.module.operation().fmt(.{
            .debug_info = false,
        })});

        return try self.allocator.dupeZ(u8, al.written());
    }

    /// Emit `transform_N` stub funcs — one per non-skipped i/o ref-arg.
    /// Each takes `(i32 ×grid_dim_count, prefetch_refs...)` and returns
    /// `block_rank` i32 values per `RefArgSnapshot.transform_returns`
    /// (default: all zeros, matching a trivial index_map).
    fn appendTransformStubs(self: *Builder) void {
        const ctx = self.ctx;
        const unknown_loc: *const mlir.Location = .unknown(ctx);
        const i32_ty = mlir.integerType(ctx, .i32);

        const arena_alloc = self.arena.allocator();
        const total_args = self.grid_dim_count + self.prefetch_args.len;
        const arg_types = arena_alloc.alloc(*const mlir.Type, total_args) catch @panic("OOM");
        const arg_locs = arena_alloc.alloc(*const mlir.Location, total_args) catch @panic("OOM");
        for (0..self.grid_dim_count) |i| {
            arg_types[i] = i32_ty;
            arg_locs[i] = unknown_loc;
        }
        for (self.prefetch_args, 0..) |ty, i| {
            arg_types[self.grid_dim_count + i] = ty;
            arg_locs[self.grid_dim_count + i] = unknown_loc;
        }

        for (self.ref_args, 0..) |ra, idx| {
            if (ra.skipped) continue;

            const rank: usize = @intCast(ra.block_rank);
            const result_types = arena_alloc.alloc(*const mlir.Type, rank) catch @panic("OOM");
            for (result_types) |*r| r.* = i32_ty;

            const block = mlir.Block.init(arg_types, arg_locs);

            const c0 = arith.constant_int(ctx, 0, i32_ty, unknown_loc);
            _ = c0.appendTo(block);

            const ret_vals = arena_alloc.alloc(*const mlir.Value, rank) catch @panic("OOM");
            const returns = ra.transform_returns;
            for (ret_vals, 0..) |*v, i| {
                if (returns) |r_list| {
                    if (i < r_list.len) {
                        v.* = switch (r_list[i]) {
                            .zero => c0.result(0),
                            .program_id => |pid| block.argument(pid),
                        };
                        continue;
                    }
                }
                v.* = c0.result(0);
            }
            _ = func.returns(ctx, ret_vals, unknown_loc).appendTo(block);

            var name_buf: [32]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "transform_{d}", .{idx}) catch unreachable;

            const stub_op = func.func(ctx, .{
                .name = name,
                .args = arg_types,
                .results = result_types,
                .block = block,
                .location = unknown_loc,
                .visibility = null,
                .verify = false,
            });
            _ = stub_op.appendTo(self.module.body());
        }
    }
};

/// Memory spaces that don't support windowing (empty window_params, no transform_N).
fn isSkippedWindowSpace(ms: MemorySpace) bool {
    return switch (ms) {
        .hbm, .any, .semaphore_mem => true,
        else => false,
    };
}

// ==================== scope-type aliases pinned to Mosaic's Builder ====================

const tupleArity = dsl.tupleArity;

pub fn ForScope(comptime N: usize) type {
    return dsl.ForScope(Builder, Value, N);
}

pub const IfOnlyScope = dsl.IfOnlyScope(Builder, Value);

pub fn IfScope(comptime N: usize) type {
    return dsl.IfScope(Builder, Value, N);
}

pub fn WhileScope(comptime N: usize, comptime M: usize) type {
    return dsl.WhileScope(Builder, Value, N, M);
}

test {
    std.testing.refAllDecls(@This());
}
