//! Layer B — high-level DSL for building TTIR kernels from Zig.
//!
//! A `Kernel` owns its own `mlir.Module` and `tt.func` entry block. Consumers
//! build the body with typed `Value` handles via the `programId`/`makeRange`/
//! `addptr`/`load`/`store`/`addi`/... helpers, plus `forLoop`/`ifThenElse`
//! for SCF regions. Call `finish(...)` to terminate with `tt.return`, verify,
//! and serialize the module to a TTIR string suitable for `zml.ops.triton(...)`.
//!
//! The `mlir.Context` is provided by the caller; it must have the `tt`, `scf`,
//! and `arith` dialects loaded (the main ZML context loads them all — see
//! `zml/module.zig` registration).

const std = @import("std");

const mlir = @import("mlir");
const dialects = @import("mlir/dialects");
const stdx = @import("stdx");

const arith = dialects.arith;
const math = dialects.math;
const scf = dialects.scf;
const ttir = dialects.ttir;

/// A handle to an MLIR value inside the kernel body. Carries an optional
/// back-pointer to its owning `Kernel` so fluent methods (`a.add(b)`, `a.lt(b)`,
/// `a.to(.f32)`, …) can emit new ops without a second argument. `kernel` is
/// populated by `emit`/`emitMulti`/`arg`; values constructed by hand may leave
/// it `null` and must use the explicit `Kernel.*` builders.
pub const Value = struct {
    inner: *const mlir.Value,
    kernel: ?*Kernel = null,

    pub const Shape = stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK);

    pub fn type_(self: Value) *const mlir.Type {
        return self.inner.type_();
    }

    fn kern(self: Value) *Kernel {
        return self.kernel orelse @panic("Value has no owning kernel; use Kernel.* helpers instead");
    }

    /// Element type of a tensor value, or the value's own type if scalar.
    pub fn elemType(self: Value) *const mlir.Type {
        if (self.type_().isA(mlir.ShapedType)) |shaped| return shaped.elementType();
        return self.type_();
    }

    /// True when this value is a ranked tensor (rank ≥ 0), false for scalars.
    pub fn isTensor(self: Value) bool {
        return self.type_().isA(mlir.ShapedType) != null;
    }

    /// Rank of this value. Scalars have rank 0.
    pub fn rank(self: Value) usize {
        if (self.type_().isA(mlir.ShapedType)) |shaped| return shaped.rank();
        return 0;
    }

    /// `i`-th dim size of a tensor value; traps on scalar.
    pub fn dim(self: Value, i: usize) i64 {
        return self.type_().isA(mlir.ShapedType).?.dimension(i);
    }

    /// Shape as a stack-allocated BoundedArray. Use `.constSlice()` for `[]const i64`.
    pub fn shape(self: Value) Shape {
        var out: Shape = .{};
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
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.addf(l, r) else k.addi(l, r);
    }

    pub fn sub(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.subf(l, r) else k.subi(l, r);
    }

    pub fn mul(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.mulf(l, r) else k.muli(l, r);
    }

    pub fn div(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.divf(l, r) else k.divsi(l, r);
    }

    pub fn rem(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.remf(l, r) else k.remsi(l, r);
    }

    pub fn cdiv(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return k.ceildivsi(l, r);
    }

    pub fn bitAnd(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return k.andi(l, r);
    }

    pub fn bitOr(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return k.ori(l, r);
    }

    /// Elementwise min — `tl.minimum(a, b)`. Name distinguished from the reduce
    /// `min` (Python `tl.min`). Use `minimumOpts` for `propagate_nan`.
    pub fn minimum(self: Value, rhs: anytype) Value {
        return self.kern().minimum(self, rhs);
    }
    /// `tl.minimum(self, rhs, propagate_nan=...)`.
    pub fn minimumOpts(self: Value, rhs: anytype, opts: MinMaxOpts) Value {
        return self.kern().minimumOpts(self, rhs, opts);
    }

    /// Elementwise max — `tl.maximum(a, b)`. Name distinguished from the reduce
    /// `max` (Python `tl.max`). Use `maximumOpts` for `propagate_nan`.
    pub fn maximum(self: Value, rhs: anytype) Value {
        return self.kern().maximum(self, rhs);
    }
    /// `tl.maximum(self, rhs, propagate_nan=...)`.
    pub fn maximumOpts(self: Value, rhs: anytype, opts: MinMaxOpts) Value {
        return self.kern().maximumOpts(self, rhs, opts);
    }

    // -------- fluent comparisons (int predicates default; float uses cmpf) --------

    pub fn lt(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.olt, l, r) else k.cmpi(.slt, l, r);
    }
    pub fn le(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.ole, l, r) else k.cmpi(.sle, l, r);
    }
    pub fn gt(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.ogt, l, r) else k.cmpi(.sgt, l, r);
    }
    pub fn ge(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.oge, l, r) else k.cmpi(.sge, l, r);
    }
    pub fn eq(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.oeq, l, r) else k.cmpi(.eq, l, r);
    }
    pub fn ne(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.one, l, r) else k.cmpi(.ne, l, r);
    }

    // -------- fluent casts (`to`) --------

    /// `tl.cast(self, dtype)` / `self.to(dtype)` — numeric cast, preserving
    /// shape. Auto-picks extsi/trunci/extf/truncf/sitofp/fptosi/fpToFp. Use
    /// `toOpts` for `fp_downcast_rounding` / `bitcast=True`.
    pub fn to(self: Value, dtype: DType) Value {
        return self.kern().cast(self, dtype);
    }
    /// `tl.cast(self, dtype, fp_downcast_rounding=..., bitcast=...)`.
    pub fn toOpts(self: Value, dtype: DType, opts: CastOpts) Value {
        return self.kern().castOpts(self, dtype, opts);
    }

    // -------- shape manipulation --------

    /// Expand this 1-D vector to a 2-D `[m, n]` tensor by broadcasting along
    /// the axis opposite to `axis`. `axis=0` keeps the value as a column
    /// (replicates across rows); `axis=1` keeps it as a row. Matches Python's
    /// `vec[:, None]` (axis=1) / `vec[None, :]` (axis=0) idioms.
    pub fn broadcast2d(self: Value, axis: i32, m: i64, n: i64) Value {
        const k = self.kern();
        const exp = k.expandDims(self, axis);
        const target: [2]i64 = .{ m, n };
        return k.broadcastTo(exp, &target);
    }

    /// Splat this scalar value to a tensor of the given shape.
    pub fn splatTo(self: Value, shape_: []const i64) Value {
        return self.kern().splat(self, shape_);
    }

    // -------- fluent reductions / scans / gather --------

    /// Reduce sum — `tl.sum(self)` over all dims (Python's axis=None default).
    pub fn sum(self: Value) Value {
        return self.kern().sum(self);
    }
    /// Reduce sum with opts — `tl.sum(self, axis=..., keep_dims=...)`.
    pub fn sumOpts(self: Value, opts: ReduceOpts) Value {
        return self.kern().sumOpts(self, opts);
    }

    /// Reduce max — `tl.max(self)` over all dims. For elementwise max, use `.maximum(rhs)`.
    pub fn max(self: Value) Value {
        return self.kern().max(self);
    }
    /// Reduce max with opts — `tl.max(self, axis=..., keep_dims=...)`.
    pub fn maxOpts(self: Value, opts: ReduceOpts) Value {
        return self.kern().maxOpts(self, opts);
    }

    /// Reduce min — `tl.min(self)` over all dims. For elementwise min, use `.minimum(rhs)`.
    pub fn min(self: Value) Value {
        return self.kern().min(self);
    }
    /// Reduce min with opts — `tl.min(self, axis=..., keep_dims=...)`.
    pub fn minOpts(self: Value, opts: ReduceOpts) Value {
        return self.kern().minOpts(self, opts);
    }

    /// Cumulative sum — `tl.cumsum(self, axis=0)`.
    pub fn cumsum(self: Value) Value {
        return self.kern().cumsum(self);
    }
    /// Cumulative sum with opts — `tl.cumsum(self, axis=..., reverse=...)`.
    pub fn cumsumOpts(self: Value, opts: ScanOpts) Value {
        return self.kern().cumsumOpts(self, opts);
    }

    /// Cumulative product — `tl.cumprod(self, axis=0)`.
    pub fn cumprod(self: Value) Value {
        return self.kern().cumprod(self);
    }
    /// Cumulative product with opts — `tl.cumprod(self, axis=..., reverse=...)`.
    pub fn cumprodOpts(self: Value, opts: ScanOpts) Value {
        return self.kern().cumprodOpts(self, opts);
    }

    /// `tl.gather(self, indices, axis)` — gather elements from self.
    pub fn gather(self: Value, indices: Value, axis: i32) Value {
        return self.kern().gather(self, indices, axis);
    }

    /// `tl.abs(self)` — absolute value (dispatches float/int on element type).
    pub fn abs(self: Value) Value {
        return self.kern().abs(self);
    }

    /// `addptr(self, offset)` where offset can be a Value or comptime int.
    /// If `self` is a scalar pointer and `offset` is a tensor, the pointer
    /// is auto-splatted to match (Triton-style `x_ptr + offs`).
    pub fn addPtr(self: Value, offset: anytype) Value {
        const k = self.kern();
        const off: Value = if (@TypeOf(offset) == Value) offset else k.lift(offset);
        const ptr = if (!self.isTensor() and off.isTensor()) k.splat(self, off.shape().constSlice()) else self;
        return k.addptr(ptr, off);
    }
};

fn isFloatDtype(dt: DType) bool {
    return switch (dt) {
        .f16, .bf16, .f32, .f64, .f8e4m3fn, .f8e5m2 => true,
        else => false,
    };
}

fn dtypeBitwidth(dt: DType) usize {
    return switch (dt) {
        .i1 => 1,
        .i8, .f8e4m3fn, .f8e5m2 => 8,
        .i16, .f16, .bf16 => 16,
        .i32, .f32 => 32,
        .i64, .f64 => 64,
    };
}

/// Compute the result type of `reduce(src, axis)`: drop dim `axis` from `src_shape`.
/// When the source is 1-D, the result is a scalar (element type only).
fn computeReducedType(src_shape: []const i64, axis: i32, elem: *const mlir.Type) *const mlir.Type {
    if (src_shape.len <= 1) return elem;
    var out: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .{};
    for (src_shape, 0..) |d, i| {
        if (@as(i32, @intCast(i)) != axis) out.appendAssumeCapacity(d);
    }
    return mlir.rankedTensorType(out.constSlice(), elem);
}

/// Element type on a kernel parameter: integer, float, or an index-friendly int.
pub const DType = enum {
    i1,
    i8,
    i16,
    i32,
    i64,
    f16,
    bf16,
    f32,
    f64,
    f8e4m3fn,
    f8e5m2,

    pub fn toMlir(self: DType, ctx: *mlir.Context) *const mlir.Type {
        return switch (self) {
            .i1 => mlir.integerType(ctx, .i1),
            .i8 => mlir.integerType(ctx, .i8),
            .i16 => mlir.integerType(ctx, .i16),
            .i32 => mlir.integerType(ctx, .i32),
            .i64 => mlir.integerType(ctx, .i64),
            .f16 => mlir.floatType(ctx, .f16),
            .bf16 => mlir.floatType(ctx, .bf16),
            .f32 => mlir.floatType(ctx, .f32),
            .f64 => mlir.floatType(ctx, .f64),
            .f8e4m3fn => mlir.floatType(ctx, .f8e4m3fn),
            .f8e5m2 => mlir.floatType(ctx, .f8e5m2),
        };
    }
};

/// Per-arg specification for building `tt.func`'s signature.
pub const ArgSpec = struct {
    name: []const u8,
    kind: Kind,

    pub const Kind = union(enum) {
        /// `!tt.ptr<T, addr_space>` — a pointer to a scalar. Optional
        /// `tt.divisibility` arg-attr hint (default 32, used by Triton to
        /// enable vectorized loads).
        ptr: struct {
            dtype: DType,
            address_space: i32 = 1,
            divisibility: ?u32 = 32,
        },
        /// Plain scalar argument (e.g. strides, sizes).
        scalar: DType,
        /// Ranked tensor argument (rare in TTIR kernels, but supported).
        tensor: struct {
            shape: []const i64,
            dtype: DType,
        },
    };

    /// Terse constructor: `arg.ptr("a_ptr", .f32)` for a `!tt.ptr<f32>` arg.
    pub fn ptr(name: []const u8, dtype: DType) ArgSpec {
        return .{ .name = name, .kind = .{ .ptr = .{ .dtype = dtype } } };
    }

    /// Terse constructor: `arg.scalar("n", .i64)` for a plain scalar arg.
    pub fn scalar(name: []const u8, dtype: DType) ArgSpec {
        return .{ .name = name, .kind = .{ .scalar = dtype } };
    }

    /// Terse constructor: `arg.tensor("x", &.{512}, .f32)` for a ranked tensor arg.
    pub fn tensor(name: []const u8, shape: []const i64, dtype: DType) ArgSpec {
        return .{ .name = name, .kind = .{ .tensor = .{ .shape = shape, .dtype = dtype } } };
    }
};

pub const FinishError = error{InvalidMlir} || std.mem.Allocator.Error || std.Io.Writer.Error;

/// Named-param struct for `Kernel.clampfOpts` — mirrors
/// `tl.clamp(x, min, max, propagate_nan=...)`.
pub const ClampOpts = struct {
    propagate_nan: ttir.PropagateNan = .none,
};

/// Named-param struct for `Kernel.histogramOpts` —
/// `tl.histogram(src, num_bins, mask=...)`.
pub const HistogramOpts = struct {
    mask: ?Value = null,
};

/// Named-param struct for `Kernel.printOpts` —
/// `tl.device_print(prefix, *args, hex=...)`.
pub const PrintOpts = struct {
    hex: bool = false,
};

/// Named-param struct for `Kernel.sumOpts` / `Kernel.maxOpts` — mirrors
/// `tl.sum(src, axis=None, keep_dims=False)` / `tl.max(src, axis=None, keep_dims=False)`.
/// `axis = null` flattens over all dims (Python's `axis=None` default).
pub const ReduceOpts = struct {
    axis: ?i32 = null,
    keep_dims: bool = false,
};

/// Named-param struct for `Kernel.cumsumOpts` —
/// `tl.cumsum(src, axis=0, reverse=False)`.
pub const ScanOpts = struct {
    axis: i32 = 0,
    reverse: bool = false,
};

/// Named-param struct for `Kernel.loadOpts` — mirrors
/// `tl.load(ptr, mask=..., other=..., cache_modifier=..., eviction_policy=..., volatile=...)`.
pub const LoadOpts = struct {
    mask: ?Value = null,
    other: ?Value = null,
    cache_modifier: ttir.CacheModifier = .none,
    eviction_policy: ttir.EvictionPolicy = .normal,
    @"volatile": bool = false,
};

/// Named-param struct for `Kernel.storeOpts` — mirrors
/// `tl.store(ptr, value, mask=..., cache_modifier=..., eviction_policy=...)`.
pub const StoreOpts = struct {
    mask: ?Value = null,
    cache_modifier: ttir.CacheModifier = .none,
    eviction_policy: ttir.EvictionPolicy = .normal,
};

/// Named-param struct for `Kernel.reshapeOpts` —
/// `tl.reshape(src, shape, can_reorder=...)` plus the `efficient_layout`
/// escape hatch.
pub const ReshapeOpts = struct {
    can_reorder: bool = false,
    efficient_layout: bool = false,
};

/// Named-param struct for `Kernel.dotOpts` — mirrors
/// `tl.dot(a, b, acc, input_precision=..., max_num_imprecise_acc=...)`.
pub const DotOpts = struct {
    input_precision: ttir.InputPrecision = .ieee,
    max_num_imprecise_acc: i32 = 0,
};

/// Named-param struct for `Kernel.fpToFpOpts`. If `rounding` is null, no
/// rounding-mode attribute is emitted.
pub const FpToFpOpts = struct {
    rounding: ?ttir.RoundingMode = null,
};

/// Named-param struct for `Kernel.atomicRmwOpts` — mirrors
/// `tl.atomic_add(ptr, val, mask=..., sem=..., scope=...)`.
pub const AtomicRMWOpts = struct {
    mask: ?Value = null,
    sem: ttir.MemSemantic = .acq_rel,
    scope: ttir.MemSyncScope = .gpu,
};

/// Named-param struct for `Kernel.atomicCasOpts`.
pub const AtomicCasOpts = struct {
    sem: ttir.MemSemantic = .acq_rel,
    scope: ttir.MemSyncScope = .gpu,
};

/// Named-param struct for `Kernel.dotScaledOpts`.
pub const DotScaledOpts = struct {
    fast_math: bool = false,
    lhs_k_pack: bool = true,
    rhs_k_pack: bool = true,
};

/// Named-param struct for `Kernel.externElementwiseOpts` — mirrors
/// `tl.extern_elementwise(..., is_pure=...)`.
pub const ExternElementwiseOpts = struct {
    is_pure: bool = true,
};

/// Named-param struct for `Kernel.descriptorLoadOpts`.
pub const DescriptorLoadOpts = struct {
    cache_modifier: ttir.CacheModifier = .none,
    eviction_policy: ttir.EvictionPolicy = .normal,
};

/// Named-param struct for `Kernel.inlineAsmElementwiseOpts` — mirrors
/// `tl.inline_asm_elementwise(asm, constraints, args, dtype, is_pure=..., pack=...)`.
pub const InlineAsmOpts = struct {
    is_pure: bool = true,
    pack: i32 = 1,
};

/// Named-param struct for `Kernel.catOpts` — mirrors
/// `tl.cat(input, other, can_reorder=False, dim=0)`.
pub const CatOpts = struct {
    can_reorder: bool = false,
    dim: i32 = 0,
};

/// Named-param struct for `Kernel.castOpts` / `Value.toOpts` — mirrors
/// `tl.cast(input, dtype, fp_downcast_rounding=None, bitcast=False)`.
pub const CastOpts = struct {
    fp_downcast_rounding: ?ttir.RoundingMode = null,
    bitcast: bool = false,
};

/// Named-param struct for `Kernel.maximumOpts`/`minimumOpts` / `Value.maximumOpts`/`minimumOpts` —
/// mirrors `tl.maximum(a, b, propagate_nan=...)` / `tl.minimum(a, b, propagate_nan=...)`.
pub const MinMaxOpts = struct {
    propagate_nan: ttir.PropagateNan = .none,
};

/// Named-param struct for `Kernel.deviceAssertOpts` — mirrors
/// `tl.device_assert(cond, msg="", mask=None)`.
pub const DeviceAssertOpts = struct {
    mask: ?Value = null,
};

/// Typed arg struct for `Kernel.forLoop`. The ctx type is inferred from the
/// caller-supplied `ctx` value at the call site, so fields like `body` can
/// carry a concrete function-pointer type instead of `anytype`. `lower`,
/// `upper`, and `step` are passed positionally to `forLoop` so they retain
/// the polymorphic lift (Value / comptime int / runtime int).
pub fn ForArgs(comptime CtxT: type) type {
    return struct {
        inits: []const Value = &.{},
        body: *const fn (*Kernel, Value, []const Value, CtxT) []const Value,
    };
}

/// Typed arg struct for `Kernel.ifThenElse`.
pub fn IfArgs(comptime CtxT: type) type {
    return struct {
        cond: Value,
        results: []const *const mlir.Type = &.{},
        then_: *const fn (*Kernel, CtxT) []const Value,
        else_: ?*const fn (*Kernel, CtxT) []const Value = null,
    };
}

/// Return type for `whileLoop`'s `before` callback: the i1 loop condition plus
/// the values forwarded to the `after` region via `scf.condition`.
pub const WhileBefore = struct {
    cond: Value,
    forwarded: []const Value,
};

/// Typed arg struct for `Kernel.whileLoop`.
pub fn WhileArgs(comptime CtxT: type) type {
    return struct {
        inits: []const Value,
        after_types: []const *const mlir.Type,
        before: *const fn (*Kernel, []const Value, CtxT) WhileBefore,
        after: *const fn (*Kernel, []const Value, CtxT) []const Value,
    };
}

/// Typed arg struct for `Kernel.reduce`.
pub fn ReduceArgs(comptime CtxT: type) type {
    return struct {
        src: Value,
        axis: i32,
        elem: *const mlir.Type,
        result: *const mlir.Type,
        combine: *const fn (*Kernel, Value, Value, CtxT) Value,
    };
}

/// Typed arg struct for `Kernel.scan`.
pub fn ScanArgs(comptime CtxT: type) type {
    return struct {
        src: Value,
        axis: i32,
        reverse: bool = false,
        elem: *const mlir.Type,
        result: *const mlir.Type,
        combine: *const fn (*Kernel, Value, Value, CtxT) Value,
    };
}

/// The main DSL builder. Create with `init`, populate the body with helpers,
/// then call `finish` to get the IR string.
pub const Kernel = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    ctx: *mlir.Context,
    module: *mlir.Module,
    func_op: *mlir.Operation,
    entry_block: *mlir.Block,
    block_stack: stdx.BoundedArray(*mlir.Block, 16) = .{},

    /// Create a sub-module containing a single public `tt.func` of the given
    /// name, signature, and result types. The entry block is populated with
    /// one argument per `ArgSpec`. Subsequent ops emitted via the kernel
    /// helpers are appended to the entry block by default.
    pub fn init(
        allocator: std.mem.Allocator,
        ctx: *mlir.Context,
        name: []const u8,
        args: []const ArgSpec,
        result_types: []const *const mlir.Type,
    ) !Kernel {
        const unknown_loc: *const mlir.Location = .unknown(ctx);
        const module: *mlir.Module = .init(unknown_loc);
        errdefer module.deinit();

        var arg_types: stdx.BoundedArray(*const mlir.Type, 64) = .{};
        var arg_locs: stdx.BoundedArray(*const mlir.Location, 64) = .{};
        for (args) |a| {
            arg_types.appendAssumeCapacity(switch (a.kind) {
                .ptr => |p| ttir.pointerType(p.dtype.toMlir(ctx), p.address_space),
                .scalar => |dt| dt.toMlir(ctx),
                .tensor => |t| mlir.rankedTensorType(t.shape, t.dtype.toMlir(ctx)),
            });
            arg_locs.appendAssumeCapacity(unknown_loc);
        }

        const entry = mlir.Block.init(arg_types.constSlice(), arg_locs.constSlice());

        var arg_attrs: stdx.BoundedArray(*const mlir.Attribute, 64) = .{};
        var any_arg_attr = false;
        for (args) |a| {
            const empty_dict: *const mlir.Attribute = mlir.dictionaryAttribute(ctx, &.{});
            switch (a.kind) {
                .ptr => |p| {
                    if (p.divisibility) |v| {
                        const div_attr = mlir.dictionaryAttribute(ctx, &.{
                            .named(ctx, "tt.divisibility", mlir.integerAttribute(ctx, .i32, v)),
                        });
                        arg_attrs.appendAssumeCapacity(div_attr);
                        any_arg_attr = true;
                    } else {
                        arg_attrs.appendAssumeCapacity(empty_dict);
                    }
                },
                else => arg_attrs.appendAssumeCapacity(empty_dict),
            }
        }

        const func_op = ttir.func(ctx, .{
            .name = name,
            .args = arg_types.constSlice(),
            .args_attributes = if (any_arg_attr) arg_attrs.constSlice() else null,
            .results = result_types,
            .block = entry,
            .location = unknown_loc,
        });
        _ = func_op.appendTo(module.body());

        return .{
            .allocator = allocator,
            .arena = std.heap.ArenaAllocator.init(allocator),
            .ctx = ctx,
            .module = module,
            .func_op = func_op,
            .entry_block = entry,
        };
    }

    pub fn deinit(self: *Kernel) void {
        self.module.deinit();
        self.arena.deinit();
    }

    /// The i-th entry block argument (i.e. `%argN`).
    pub fn arg(self: *Kernel, i: usize) Value {
        return .{ .inner = self.entry_block.argument(i), .kernel = self };
    }

    pub fn pushBlock(self: *Kernel, b: *mlir.Block) void {
        self.block_stack.appendAssumeCapacity(b);
    }

    pub fn popBlock(self: *Kernel) void {
        _ = self.block_stack.pop();
    }

    pub fn currentBlock(self: *Kernel) *mlir.Block {
        return if (self.block_stack.len == 0)
            self.entry_block
        else
            self.block_stack.get(self.block_stack.len - 1);
    }

    /// Emit `op` into the current block and return its first result as a Value.
    pub fn emit(self: *Kernel, op: *mlir.Operation) Value {
        _ = op.appendTo(self.currentBlock());
        return .{ .inner = op.result(0), .kernel = self };
    }

    /// Emit `op` and return the first `n` results. Allocated in the kernel's
    /// per-frame arena.
    pub fn emitMulti(self: *Kernel, op: *mlir.Operation, n: usize) []Value {
        _ = op.appendTo(self.currentBlock());
        const out = self.arena.allocator().alloc(Value, n) catch @panic("Kernel.emitMulti OOM");
        for (0..n) |i| out[i] = .{ .inner = op.result(i), .kernel = self };
        return out;
    }

    /// Pack a tuple of Values into an arena-allocated slice for returning from
    /// a loop / if body. Use instead of the `arena.allocator().alloc(...) +
    /// out[i] = ...` boilerplate.
    ///
    ///     return kk.yield(.{ new_acc });
    ///     return kk.yield(.{ new_acc, new_a_ptrs, new_b_ptrs });
    pub fn yield(self: *Kernel, values: anytype) []const Value {
        const T = @TypeOf(values);
        const info = @typeInfo(T);
        if (info != .@"struct" or !info.@"struct".is_tuple)
            @compileError("Kernel.yield expects a tuple literal like `.{ v1, v2 }`");
        const n = info.@"struct".fields.len;
        const out = self.arena.allocator().alloc(Value, n) catch @panic("Kernel.yield OOM");
        inline for (info.@"struct".fields, 0..) |f, i| {
            if (f.type != Value)
                @compileError("Kernel.yield: every tuple element must be a Value; got " ++ @typeName(f.type));
            out[i] = @field(values, f.name);
        }
        return out;
    }

    fn loc(self: *const Kernel) *const mlir.Location {
        return .unknown(self.ctx);
    }

    // ==================== type helpers ====================

    /// Scalar MLIR type for the given DSL element type. Public because
    /// `whileLoop.after_types` and other advanced ops still accept raw MLIR
    /// types; the common load / reshape / dot / etc. paths infer internally
    /// and never expose this to users.
    pub fn scalarTy(self: *const Kernel, dtype: DType) *const mlir.Type {
        return dtype.toMlir(self.ctx);
    }

    /// Ranked tensor MLIR type `tensor<shape x dtype>`. See `scalarTy`.
    pub fn tensorTy(self: *const Kernel, shape: []const i64, dtype: DType) *const mlir.Type {
        return mlir.rankedTensorType(shape, dtype.toMlir(self.ctx));
    }

    /// MLIR type produced by loading through `ptr`. Scalar `!tt.ptr<T>` →
    /// scalar `T`; `tensor<... x !tt.ptr<T>>` → `tensor<... x T>`.
    fn loadResultType(self: *const Kernel, ptr: Value) *const mlir.Type {
        _ = self;
        const pt = ptr.type_();
        if (pt.isA(mlir.ShapedType)) |shaped| {
            const elem_ptr_ty = shaped.elementType();
            const pointee = elem_ptr_ty.isA(ttir.PointerType).?.pointee();
            var shape: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .{};
            for (0..shaped.rank()) |i| shape.appendAssumeCapacity(shaped.dimension(i));
            return mlir.rankedTensorType(shape.constSlice(), pointee);
        }
        return pt.isA(ttir.PointerType).?.pointee();
    }

    /// Replace `src`'s element type with `out_dtype`, preserving rank/shape
    /// (scalars stay scalars, tensors keep their shape).
    fn swapElem(self: *const Kernel, src: Value, out_dtype: DType) *const mlir.Type {
        const out_elem = out_dtype.toMlir(self.ctx);
        if (src.isTensor()) return mlir.rankedTensorType(src.shape().constSlice(), out_elem);
        return out_elem;
    }

    // ==================== SPMD ====================

    pub fn programId(self: *Kernel, dim: ttir.ProgramIDDim) Value {
        return self.emit(ttir.get_program_id(self.ctx, dim, self.loc()));
    }

    pub fn numPrograms(self: *Kernel, dim: ttir.ProgramIDDim) Value {
        return self.emit(ttir.get_num_programs(self.ctx, dim, self.loc()));
    }

    // ==================== constants ====================
    //
    // Public surface:
    //   - `lift(value)`                 — dtype inferred from source Zig type.
    //   - `liftAs(value, dtype)`        — explicit DSL `DType`.
    //   - `constMatching(value, elem)`  — same as `liftAs` but takes an MLIR
    //                                     element type (e.g. `ref.elemType()`).
    // Binary ops, `splat`, `addPtr`, and `forLoop` all take `anytype` so in
    // practice you rarely need to build a constant by hand.

    fn emitInt(self: *Kernel, dtype: DType, value: i64) Value {
        return self.emit(arith.constant_int(self.ctx, value, dtype.toMlir(self.ctx), self.loc()));
    }

    fn emitFloat(self: *Kernel, dtype: DType, value: f64) Value {
        return switch (dtype) {
            .f32 => self.emit(arith.constant_float(self.ctx, value, .f32, self.loc())),
            .f64 => self.emit(arith.constant_float(self.ctx, value, .f64, self.loc())),
            else => blk: {
                const f32_scalar = self.emit(arith.constant_float(self.ctx, value, .f32, self.loc()));
                break :blk self.fpToFp(f32_scalar, dtype);
            },
        };
    }

    // ==================== ranges and shape manipulation ====================

    /// `tt.make_range` — produces a 1D `tensor<Nxi32>` with values `[start, end)`.
    pub fn makeRange(self: *Kernel, start: i32, end: i32) Value {
        std.debug.assert(end >= start);
        const len: i64 = @intCast(end - start);
        const ty = mlir.rankedTensorType(&.{len}, mlir.integerType(self.ctx, .i32));
        return self.emit(ttir.make_range(self.ctx, start, end, ty, self.loc()));
    }

    /// `tl.arange(start, end)` with optional dtype promotion — mirrors Python
    /// Triton's two-argument signature. Pass `.i32` for the native range type;
    /// any other dtype triggers an extsi/trunci/sitofp as appropriate.
    pub fn arange(self: *Kernel, start: i32, end: i32, dtype: DType) Value {
        const r = self.makeRange(start, end);
        if (dtype == .i32) return r;
        return switch (dtype) {
            .i64 => self.extsi(r, dtype),
            .i16, .i8 => self.trunci(r, dtype),
            .f16, .bf16, .f32, .f64, .f8e4m3fn, .f8e5m2 => self.sitofp(r, dtype),
            else => @panic("Kernel.arange: unsupported dtype"),
        };
    }

    /// `tt.splat` with polymorphic input: accepts a Value, a `comptime_int`, or
    /// a `comptime_float`. Comptime scalars are auto-lifted to a constant of
    /// the matching dtype (i32 for int, f32 for float) before splatting.
    pub fn splat(self: *Kernel, value: anytype, shape: []const i64) Value {
        const v: Value = if (@TypeOf(value) == Value) value else self.lift(value);
        const ty = mlir.rankedTensorType(shape, v.type_());
        return self.emit(ttir.splat(self.ctx, v.inner, ty, self.loc()));
    }

    /// Lift a Zig scalar (or pass-through Value) to a scalar DSL constant.
    /// MLIR integers are signless; unsigned sources are bit-cast so the MLIR
    /// constant carries the same bit pattern (e.g. `u32(0xFFFF_FFFF)` → i32
    /// with bit pattern `-1`). The resulting DSL dtype follows the source:
    /// - `comptime_int`: `i32` if it fits in i32; else `i64` for values up to
    ///   the full u64 bit range; larger → `@compileError`.
    /// - Runtime int `i8/i16/i32`, `u8/…/u32`: `i32` (signed preserved, unsigned
    ///   bit-cast).
    /// - Runtime int `i64/u64`: `i64` (signed preserved, unsigned bit-cast).
    /// - `comptime_float`: `f32` (Triton's common default).
    /// - Runtime `f16` → `f16`; `f32` → `f32`; `f64` → `f64`.
    ///
    /// To target a specific dtype regardless of source, call `liftAs(value, dtype)`.
    pub fn lift(self: *Kernel, value: anytype) Value {
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
                @compileError("Kernel.lift: integer literal out of 64-bit range");
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
                else => @compileError("Kernel.lift: unsupported float bitwidth"),
            },
            else => @compileError("Kernel.lift: unsupported type " ++ @typeName(T)),
        };
    }

    /// `lift(value)` but comptime scalars match `ref_elem` instead of the
    /// default i32/f32 lift. Pass-through for existing Values; runtime Zig
    /// scalars preserve their source width (cast at the call site to change).
    fn liftMatching(self: *Kernel, value: anytype, ref_elem: *const mlir.Type) Value {
        const T = @TypeOf(value);
        if (T == Value) return value;
        return switch (@typeInfo(T)) {
            .comptime_int, .comptime_float => self.constMatching(value, ref_elem),
            else => self.lift(value),
        };
    }

    /// Lift `value` and — if `ref` is a tensor — splat it to match `ref`'s
    /// shape. Comptime scalars pick up `ref`'s element type.
    pub fn broadcastLike(self: *Kernel, value: anytype, ref: Value) Value {
        const v = self.liftMatching(value, ref.elemType());
        if (ref.isTensor() and !v.isTensor()) return self.splat(v, ref.shape().constSlice());
        return v;
    }

    /// `tl.broadcast(a, b)` — lifts both sides and splats the scalar one to
    /// match the tensor's shape. Used by fluent ops (`.add`, `.lt`, …) so
    /// `tensor.op(s)` and `s.op(tensor)` both work. No NumPy rank unification.
    pub fn broadcast(self: *Kernel, a: anytype, b: anytype) struct { Value, Value } {
        const a_ref: ?*const mlir.Type = if (@TypeOf(b) == Value) b.elemType() else null;
        const b_ref: ?*const mlir.Type = if (@TypeOf(a) == Value) a.elemType() else null;
        const av = if (a_ref) |t| self.liftMatching(a, t) else self.lift(a);
        const bv = if (b_ref) |t| self.liftMatching(b, t) else self.lift(b);
        if (av.isTensor() and !bv.isTensor()) return .{ av, self.splat(bv, av.shape().constSlice()) };
        if (!av.isTensor() and bv.isTensor()) return .{ self.splat(av, bv.shape().constSlice()), bv };
        return .{ av, bv };
    }

    /// Reverse lookup: scalar MLIR type → DSL `DType`.
    fn mlirElemToDType(self: *const Kernel, elem: *const mlir.Type) DType {
        inline for (std.meta.fields(DType)) |f| {
            const dt = @field(DType, f.name);
            if (elem.eql(dt.toMlir(self.ctx))) return dt;
        }
        @panic("element type not a recognized DSL DType");
    }

    /// Lift a Zig scalar to a DSL constant of the given `DType`. Unlike
    /// `lift`, the target dtype is explicit, so source-width is ignored.
    /// Unsigned ints are bit-cast (bit pattern preserved, not value).
    /// Float→int casts via `@intFromFloat`, int→float via `@floatFromInt`.
    pub fn liftAs(self: *Kernel, value: anytype, dtype: DType) Value {
        const T = @TypeOf(value);
        if (T == Value) return value;
        if (isFloatDtype(dtype)) {
            const v_f64: f64 = switch (@typeInfo(T)) {
                .comptime_int, .int => @floatFromInt(value),
                .comptime_float, .float => @floatCast(value),
                else => @compileError("Kernel.liftAs: unsupported scalar " ++ @typeName(T)),
            };
            return self.emitFloat(dtype, v_f64);
        }
        const v_i64: i64 = switch (@typeInfo(T)) {
            .comptime_int => blk: {
                if (value >= std.math.minInt(i64) and value <= std.math.maxInt(i64))
                    break :blk @intCast(value);
                if (value >= 0 and value <= std.math.maxInt(u64))
                    break :blk @bitCast(@as(u64, value));
                @compileError("Kernel.liftAs: integer literal out of 64-bit range");
            },
            .int => |info| if (info.signedness == .signed) @intCast(value) else @bitCast(@as(u64, @intCast(value))),
            .comptime_float, .float => @intFromFloat(value),
            else => @compileError("Kernel.liftAs: unsupported scalar " ++ @typeName(T)),
        };
        return self.emitInt(dtype, v_i64);
    }

    /// Build an arith.constant of the given MLIR element type from a Zig
    /// numeric. Private — used by `broadcastLike`, `forLoop` bounds, and
    /// `lift` when matching an existing Value's element type. Users should
    /// call `liftAs(value, dtype)` instead.
    fn constMatching(self: *Kernel, value: anytype, elem: *const mlir.Type) Value {
        return self.liftAs(value, self.mlirElemToDType(elem));
    }

    /// `tl.zeros(shape, dtype)` — zero-valued tensor of the given shape and dtype.
    pub fn zeros(self: *Kernel, shape: []const i64, dtype: DType) Value {
        const scalar = if (isFloatDtype(dtype)) self.emitFloat(dtype, 0.0) else self.emitInt(dtype, 0);
        return self.splat(scalar, shape);
    }

    /// One-valued tensor — non-Python convenience (Python only has `tl.zeros` +
    /// `tl.full`). Prefer `full(shape, 1, dtype)` when strictly mirroring Python.
    pub fn ones(self: *Kernel, shape: []const i64, dtype: DType) Value {
        const scalar = if (isFloatDtype(dtype)) self.emitFloat(dtype, 1.0) else self.emitInt(dtype, 1);
        return self.splat(scalar, shape);
    }

    /// `tl.full(shape, value, dtype)` — tensor filled with the scalar `value`
    /// for the given `shape` and `dtype`. `value` is auto-lifted via `liftAs`.
    pub fn full(self: *Kernel, shape: []const i64, value: anytype, dtype: DType) Value {
        const scalar = self.liftAs(value, dtype);
        return self.splat(scalar, shape);
    }

    /// `tt.load` of a tensor, with a mask and an implicit zero "other". The
    /// "other" is auto-created in the right dtype and splat to match the ptr's
    /// shape. Matches Python `tl.load(ptr, mask=mask, other=0)`.
    pub fn loadMasked(self: *Kernel, ptr: Value, mask: Value) Value {
        const result_ty = self.loadResultType(ptr);
        const shaped = result_ty.isA(mlir.ShapedType).?;
        const dtype = self.mlirElemToDType(shaped.elementType());
        var shape: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .{};
        for (0..shaped.rank()) |i| shape.appendAssumeCapacity(shaped.dimension(i));
        const zero = self.zeros(shape.constSlice(), dtype);
        return self.loadOpts(ptr, .{ .mask = mask, .other = zero });
    }

    /// 2D broadcast helper. Matches `vec[:, None]` (axis=1) and `vec[None, :]`
    /// (axis=0) Python idioms. Grows a 1-D vector into a `[m, n]` tensor.
    pub fn broadcast2d(_: *Kernel, vec: Value, axis: i32, m: i64, n: i64) Value {
        return vec.broadcast2d(axis, m, n);
    }

    /// 2-D mask from two 1-D conditions. Equivalent to
    /// `cond_m[:, None] & cond_n[None, :]` in Python. `cond_m` has length `m`,
    /// `cond_n` has length `n`; result is `[m, n] x i1`.
    pub fn mask2d(self: *Kernel, cond_m: Value, cond_n: Value, m: i64, n: i64) Value {
        const cm2 = cond_m.broadcast2d(1, m, n);
        const cn2 = cond_n.broadcast2d(0, m, n);
        return self.andi(cm2, cn2);
    }

    /// `reduce` with an internal adder — `tl.sum(src)` reducing over all dims
    /// (Python's `axis=None` default). Use `sumOpts` to target a specific axis
    /// or set `keep_dims`.
    pub fn sum(self: *Kernel, src: Value) Value {
        return self.sumOpts(src, .{});
    }

    /// `reduce` with an internal adder — `tl.sum(src, axis=..., keep_dims=...)`.
    /// `opts.axis = null` flattens over all dims.
    pub fn sumOpts(self: *Kernel, src: Value, opts: ReduceOpts) Value {
        return self.reduceDispatch(src, opts, .sum);
    }

    /// `reduce` with an internal maximum — `tl.max(src)` reducing over all dims
    /// (Python's `axis=None` default). Use `maxOpts` to target a specific axis
    /// or set `keep_dims`.
    pub fn max(self: *Kernel, src: Value) Value {
        return self.maxOpts(src, .{});
    }

    /// `reduce` with an internal maximum — `tl.max(src, axis=..., keep_dims=...)`.
    /// `opts.axis = null` flattens over all dims.
    pub fn maxOpts(self: *Kernel, src: Value, opts: ReduceOpts) Value {
        return self.reduceDispatch(src, opts, .max);
    }

    /// `reduce` with an internal minimum — `tl.min(src)` reducing over all dims
    /// (Python's `axis=None` default). Use `minOpts` to target a specific axis
    /// or set `keep_dims`.
    pub fn min(self: *Kernel, src: Value) Value {
        return self.minOpts(src, .{});
    }

    /// `reduce` with an internal minimum — `tl.min(src, axis=..., keep_dims=...)`.
    pub fn minOpts(self: *Kernel, src: Value, opts: ReduceOpts) Value {
        return self.reduceDispatch(src, opts, .min);
    }

    fn reduceDispatch(self: *Kernel, src: Value, opts: ReduceOpts, comptime kind: enum { sum, max, min }) Value {
        const src_shape = src.shape();
        // axis=null ⇒ flatten to 1-D, then reduce axis 0.
        const input: Value, const axis: i32 = if (opts.axis) |a| .{ src, a } else blk: {
            var total: i64 = 1;
            for (0..src_shape.len) |i| total *= src_shape.get(i);
            break :blk .{ self.reshape(src, &.{total}), 0 };
        };

        const elem = input.elemType();
        const in_shape = input.shape();
        const result_ty: *const mlir.Type = computeReducedType(in_shape.constSlice(), axis, elem);
        const is_float = input.isFloatElem();
        const Ctx = struct { is_float: bool, kind: @TypeOf(kind) };
        const combine = struct {
            fn c(kk: *Kernel, lhs: Value, rhs: Value, cc: Ctx) Value {
                return switch (cc.kind) {
                    .sum => if (cc.is_float) kk.addf(lhs, rhs) else kk.addi(lhs, rhs),
                    .max => if (cc.is_float) kk.maximumf(lhs, rhs) else kk.maxsi(lhs, rhs),
                    .min => if (cc.is_float) kk.minimumf(lhs, rhs) else kk.minsi(lhs, rhs),
                };
            }
        }.c;
        const out = self.reduce(Ctx{ .is_float = is_float, .kind = kind }, .{
            .src = input,
            .axis = axis,
            .elem = elem,
            .result = result_ty,
            .combine = combine,
        });

        if (!opts.keep_dims) return out;
        // keep_dims=true: restore original rank with 1 in every reduced axis.
        // axis=null collapsed everything, so all original dims become 1;
        // otherwise expand_dims re-inserts a size-1 slot at `axis`.
        if (opts.axis == null) {
            var ones_shape: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .{};
            for (0..src_shape.len) |_| ones_shape.appendAssumeCapacity(1);
            return self.splat(out, ones_shape.constSlice());
        }
        if (in_shape.len <= 1) return self.splat(out, &.{1});
        return self.expandDims(out, axis);
    }

    /// `scan` with addition — `tl.cumsum(src, axis=0)`. Use `cumsumOpts`
    /// to override `axis` / `reverse`.
    pub fn cumsum(self: *Kernel, src: Value) Value {
        return self.cumsumOpts(src, .{});
    }

    /// `scan` with addition — `tl.cumsum(src, axis=..., reverse=...)`.
    pub fn cumsumOpts(self: *Kernel, src: Value, opts: ScanOpts) Value {
        return self.scanDispatch(src, opts, .sum);
    }

    /// `scan` with multiplication — `tl.cumprod(src, axis=0)`. Use `cumprodOpts`
    /// to override `axis` / `reverse`.
    pub fn cumprod(self: *Kernel, src: Value) Value {
        return self.cumprodOpts(src, .{});
    }

    /// `scan` with multiplication — `tl.cumprod(src, axis=..., reverse=...)`.
    pub fn cumprodOpts(self: *Kernel, src: Value, opts: ScanOpts) Value {
        return self.scanDispatch(src, opts, .prod);
    }

    fn scanDispatch(self: *Kernel, src: Value, opts: ScanOpts, comptime kind: enum { sum, prod }) Value {
        const elem = src.elemType();
        const is_float = src.isFloatElem();
        const Ctx = struct { is_float: bool, kind: @TypeOf(kind) };
        const combine = struct {
            fn c(kk: *Kernel, lhs: Value, rhs: Value, cc: Ctx) Value {
                return switch (cc.kind) {
                    .sum => if (cc.is_float) kk.addf(lhs, rhs) else kk.addi(lhs, rhs),
                    .prod => if (cc.is_float) kk.mulf(lhs, rhs) else kk.muli(lhs, rhs),
                };
            }
        }.c;
        return self.scan(Ctx{ .is_float = is_float, .kind = kind }, .{
            .src = src,
            .axis = opts.axis,
            .reverse = opts.reverse,
            .elem = elem,
            .result = src.type_(),
            .combine = combine,
        });
    }

    /// `tt.expand_dims` — insert a size-1 axis at `axis`. Result shape derived
    /// from `value.shape()` + `axis`.
    pub fn expandDims(self: *Kernel, value: Value, axis: i32) Value {
        const in_shape = value.shape();
        var out_shape: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .{};
        const ax: usize = @intCast(axis);
        for (0..in_shape.len + 1) |i| {
            if (i == ax) out_shape.appendAssumeCapacity(1) else out_shape.appendAssumeCapacity(in_shape.get(if (i < ax) i else i - 1));
        }
        const ty = mlir.rankedTensorType(out_shape.constSlice(), value.elemType());
        return self.emit(ttir.expand_dims(self.ctx, value.inner, axis, ty, self.loc()));
    }

    /// `tl.broadcast_to(value, shape)` / `tt.broadcast` — broadcast size-1
    /// dims up to the target shape. Element type is taken from the input value.
    pub fn broadcastTo(self: *Kernel, value: Value, result_shape: []const i64) Value {
        const elem = mlir.RankedTensorType.fromShaped(value.inner.type_().isA(mlir.ShapedType).?).elementType();
        const ty = mlir.rankedTensorType(result_shape, elem);
        return self.emit(ttir.broadcast(self.ctx, value.inner, ty, self.loc()));
    }

    // ==================== pointer arithmetic ====================

    pub fn addptr(self: *Kernel, ptr: Value, offset: Value) Value {
        return self.emit(ttir.addptr(self.ctx, ptr.inner, offset.inner, self.loc()));
    }

    /// `tt.load` from a scalar (or tensor of) pointer with defaulted options —
    /// `tl.load(ptr)` in Python. The result type is inferred from `ptr`:
    /// scalar `!tt.ptr<T>` loads a scalar `T`, `tensor<... x !tt.ptr<T>>`
    /// loads a `tensor<... x T>`. Use `loadOpts` for mask / other / cache /
    /// eviction / volatile.
    pub fn load(self: *Kernel, ptr: Value) Value {
        return self.loadOpts(ptr, .{});
    }

    /// `tt.load` with the full named-param set — `tl.load(ptr, mask=..., other=..., cache_modifier=..., eviction_policy=..., is_volatile=...)`.
    /// Result type is inferred from `ptr` the same way `load` does.
    pub fn loadOpts(self: *Kernel, ptr: Value, opts: LoadOpts) Value {
        const mask_inner: ?*const mlir.Value = if (opts.mask) |m| m.inner else null;
        const other_inner: ?*const mlir.Value = if (opts.other) |o| o.inner else null;
        return self.emit(ttir.load(self.ctx, ptr.inner, self.loadResultType(ptr), mask_inner, other_inner, opts.cache_modifier, opts.eviction_policy, opts.@"volatile", self.loc()));
    }

    /// `tt.store` with defaulted options — `tl.store(ptr, value)` in Python.
    /// Use `storeOpts` for mask / cache / eviction.
    pub fn store(self: *Kernel, ptr: Value, value: Value) void {
        self.storeOpts(ptr, value, .{});
    }

    /// `tt.store` with the full named-param set — `tl.store(ptr, value, mask=..., cache_modifier=..., eviction_policy=...)`.
    pub fn storeOpts(self: *Kernel, ptr: Value, value: Value, opts: StoreOpts) void {
        const mask_inner: ?*const mlir.Value = if (opts.mask) |m| m.inner else null;
        _ = ttir.store(self.ctx, ptr.inner, value.inner, mask_inner, opts.cache_modifier, opts.eviction_policy, self.loc()).appendTo(self.currentBlock());
    }

    // ==================== arith: binary ====================

    pub fn addi(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.addi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn subi(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.subi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn muli(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.muli(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn addf(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.addf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn subf(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.subf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn mulf(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.mulf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn divf(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.divf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }

    pub fn cmpi(self: *Kernel, predicate: arith.CmpIPredicate, lhs: Value, rhs: Value) Value {
        return self.emit(arith.cmpi(self.ctx, predicate, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn cmpf(self: *Kernel, predicate: arith.CmpFPredicate, lhs: Value, rhs: Value) Value {
        return self.emit(arith.cmpf(self.ctx, predicate, lhs.inner, rhs.inner, self.loc()));
    }

    /// `arith.select` — low-level backend op. Prefer `where(cond, x, y)` for
    /// the Python-named entry point (`tl.where`).
    pub fn select(self: *Kernel, cond: Value, t: Value, f: Value) Value {
        return self.emit(arith.select(self.ctx, cond.inner, t.inner, f.inner, self.loc()));
    }

    /// `tl.where(condition, x, y)` — select `x` where condition is true, `y`
    /// elsewhere. Same as `select(condition, x, y)`; this is the Python-named
    /// alias.
    pub fn where(self: *Kernel, condition: Value, x: Value, y: Value) Value {
        return self.select(condition, x, y);
    }

    // Integer div / mod
    pub fn divsi(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.divsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn divui(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.divui(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn remsi(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.remsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn remui(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.remui(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn ceildivsi(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.ceildivsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn floordivsi(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.floordivsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }

    // Integer bitwise / shifts
    pub fn andi(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.andi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn ori(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.ori(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn xori(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.xori(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn shli(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.shli(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn shrsi(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.shrsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn shrui(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.shrui(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }

    // Integer min/max
    pub fn maxsi(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.maxsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn minsi(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.minsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }

    // Float ops
    pub fn remf(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.remf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn maximumf(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.maximumf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn minimumf(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.emit(arith.minimumf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn negf(self: *Kernel, src: Value) Value {
        return self.emit(arith.negf(self.ctx, src.inner, self.loc()));
    }

    /// `tl.maximum(a, b)` — elementwise max with symmetric scalar↔tensor
    /// auto-broadcast, dispatches to `maximumf` for floats, `maxsi` for ints.
    /// Use `maximumOpts` for `propagate_nan`.
    pub fn maximum(self: *Kernel, a: anytype, b: anytype) Value {
        return self.maximumOpts(a, b, .{});
    }

    /// `tl.maximum(a, b, propagate_nan=...)`. `propagate_nan` only applies to
    /// floats; ignored for integer operands.
    pub fn maximumOpts(self: *Kernel, a: anytype, b: anytype, opts: MinMaxOpts) Value {
        const l, const r = self.broadcast(a, b);
        _ = opts.propagate_nan; // TODO: thread through when ttir.maximumf supports it
        return if (l.isFloatElem()) self.maximumf(l, r) else self.maxsi(l, r);
    }

    /// `tl.minimum(a, b)` — elementwise min, see `maximum`.
    pub fn minimum(self: *Kernel, a: anytype, b: anytype) Value {
        return self.minimumOpts(a, b, .{});
    }

    /// `tl.minimum(a, b, propagate_nan=...)`.
    pub fn minimumOpts(self: *Kernel, a: anytype, b: anytype, opts: MinMaxOpts) Value {
        const l, const r = self.broadcast(a, b);
        _ = opts.propagate_nan;
        return if (l.isFloatElem()) self.minimumf(l, r) else self.minsi(l, r);
    }

    // Casts — each preserves the src shape and swaps the element type to `out_dtype`.
    pub fn extsi(self: *Kernel, src: Value, out_dtype: DType) Value {
        return self.emit(arith.extsi(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }
    pub fn extui(self: *Kernel, src: Value, out_dtype: DType) Value {
        return self.emit(arith.extui(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }
    pub fn extf(self: *Kernel, src: Value, out_dtype: DType) Value {
        return self.emit(arith.extf(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }
    pub fn trunci(self: *Kernel, src: Value, out_dtype: DType) Value {
        return self.emit(arith.trunci(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }
    pub fn truncf(self: *Kernel, src: Value, out_dtype: DType) Value {
        return self.emit(arith.truncf(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }
    pub fn sitofp(self: *Kernel, src: Value, out_dtype: DType) Value {
        return self.emit(arith.sitofp(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }
    pub fn uitofp(self: *Kernel, src: Value, out_dtype: DType) Value {
        return self.emit(arith.uitofp(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }
    pub fn fptosi(self: *Kernel, src: Value, out_dtype: DType) Value {
        return self.emit(arith.fptosi(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }
    pub fn fptoui(self: *Kernel, src: Value, out_dtype: DType) Value {
        return self.emit(arith.fptoui(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }
    pub fn arithBitcast(self: *Kernel, src: Value, out_dtype: DType) Value {
        return self.emit(arith.bitcast(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }

    // ==================== Triton-specific ops ====================

    /// `tt.bitcast` — same-bitwidth cast. Shape preserved from `src`; elem → `out_dtype`.
    /// For ptr ↔ ptr bitcasts use `bitcastTo` with an explicit pointer type.
    pub fn bitcast(self: *Kernel, src: Value, out_dtype: DType) Value {
        return self.emit(ttir.bitcast(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }

    /// `tl.cast(src, dtype)` — numeric cast with auto-dispatch
    /// (extsi/trunci/sitofp/fptosi/fpToFp). Shape preserved. Use `castOpts`
    /// for `fp_downcast_rounding` / `bitcast=True`.
    pub fn cast(self: *Kernel, src: Value, dtype: DType) Value {
        return self.castOpts(src, dtype, .{});
    }

    /// `tl.cast(src, dtype, fp_downcast_rounding=..., bitcast=...)`.
    pub fn castOpts(self: *Kernel, src: Value, dtype: DType, opts: CastOpts) Value {
        if (opts.bitcast) return self.bitcast(src, dtype);
        const cur_elem = src.elemType();
        const tgt_elem = dtype.toMlir(self.ctx);
        if (cur_elem.eql(tgt_elem)) return src;
        const cur_is_float = src.isFloatElem();
        const tgt_is_float = isFloatDtype(dtype);
        if (cur_is_float and tgt_is_float) {
            return self.fpToFpOpts(src, dtype, .{ .rounding = opts.fp_downcast_rounding orelse .rtne });
        }
        if (cur_is_float and !tgt_is_float) return self.fptosi(src, dtype);
        if (!cur_is_float and tgt_is_float) return self.sitofp(src, dtype);
        // int → int
        const cur_bw = dtypeBitwidth(self.mlirElemToDType(cur_elem));
        const tgt_bw = dtypeBitwidth(dtype);
        if (tgt_bw > cur_bw) return self.extsi(src, dtype);
        if (tgt_bw < cur_bw) return self.trunci(src, dtype);
        return self.arithBitcast(src, dtype);
    }
    /// `tt.int_to_ptr` — cast integer (scalar or tensor) to `!tt.ptr<pointee, address_space>`,
    /// preserving shape.
    pub fn intToPtr(self: *Kernel, src: Value, pointee: DType, address_space: i32) Value {
        const ptr_elem = ttir.pointerType(pointee.toMlir(self.ctx), address_space);
        const result_type: *const mlir.Type = if (src.isTensor())
            mlir.rankedTensorType(src.shape().constSlice(), ptr_elem)
        else
            ptr_elem;
        return self.emit(ttir.int_to_ptr(self.ctx, src.inner, result_type, self.loc()));
    }
    /// `tt.ptr_to_int` — cast pointer (scalar or tensor) to integer, preserving shape.
    pub fn ptrToInt(self: *Kernel, src: Value, out_dtype: DType) Value {
        return self.emit(ttir.ptr_to_int(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }
    /// `tt.fp_to_fp` with default IEEE downcast rounding (`rtne`). Shape is
    /// preserved from `src`; only the element type changes to `out_dtype`.
    /// Use `fpToFpOpts` to choose a different `rounding`.
    pub fn fpToFp(self: *Kernel, src: Value, out_dtype: DType) Value {
        return self.fpToFpOpts(src, out_dtype, .{ .rounding = .rtne });
    }
    /// `tt.fp_to_fp` with the full named-param set (currently just
    /// `rounding`). `tl.cast(x, dtype, fp_downcast_rounding=...)` in Python.
    pub fn fpToFpOpts(self: *Kernel, src: Value, out_dtype: DType, opts: FpToFpOpts) Value {
        return self.emit(ttir.fp_to_fp(self.ctx, src.inner, self.swapElem(src, out_dtype), opts.rounding, self.loc()));
    }
    /// `tt.clampf` with default `propagate_nan = .none` — `tl.clamp(x, min, max)`.
    /// Use `clampfOpts` to override NaN propagation.
    pub fn clampf(self: *Kernel, x: Value, lo: Value, hi: Value) Value {
        return self.clampfOpts(x, lo, hi, .{});
    }
    /// `tt.clampf` with the full named-param set —
    /// `tl.clamp(x, min, max, propagate_nan=...)`.
    pub fn clampfOpts(self: *Kernel, x: Value, lo: Value, hi: Value, opts: ClampOpts) Value {
        return self.emit(ttir.clampf(self.ctx, x.inner, lo.inner, hi.inner, opts.propagate_nan, self.loc()));
    }
    /// `tl.sqrt_rn(x)` — precise square root (IEEE round-to-nearest).
    pub fn sqrtRn(self: *Kernel, x: Value) Value {
        return self.emit(ttir.precise_sqrt(self.ctx, x.inner, self.loc()));
    }
    /// `tl.div_rn(x, y)` — precise division (IEEE round-to-nearest).
    pub fn divRn(self: *Kernel, x: Value, y: Value) Value {
        return self.emit(ttir.precise_divf(self.ctx, x.inner, y.inner, self.loc()));
    }
    /// `tl.umulhi(x, y)` — most-significant N bits of the 2N-bit unsigned product.
    pub fn umulhi(self: *Kernel, x: Value, y: Value) Value {
        return self.emit(ttir.mulhiui(self.ctx, x.inner, y.inner, self.loc()));
    }

    /// `tt.dot` — matmul with accumulator, default options (ieee precision).
    /// `a`, `b`, `c_acc` are tensors. The result type follows `c_acc`'s type.
    /// Use `dotOpts` for `input_precision` / `max_num_imprecise_acc` /
    /// `allow_tf32`.
    pub fn dot(self: *Kernel, a: Value, b: Value, c_acc: Value) Value {
        return self.dotOpts(a, b, c_acc, .{});
    }

    /// `tt.dot` with the full named-param set — matches `tl.dot(a, b, acc,
    /// input_precision=..., max_num_imprecise_acc=..., out_dtype=...)`.
    /// Result type = `c_acc.type_()`.
    pub fn dotOpts(self: *Kernel, a: Value, b: Value, c_acc: Value, opts: DotOpts) Value {
        return self.emit(ttir.dot(self.ctx, a.inner, b.inner, c_acc.inner, c_acc.type_(), opts.input_precision, opts.max_num_imprecise_acc, self.loc()));
    }

    /// `tt.reshape` with default options — `tl.reshape(src, shape)`. Element
    /// type is preserved from `src`. Use `reshapeOpts` for `can_reorder` /
    /// `efficient_layout`.
    pub fn reshape(self: *Kernel, src: Value, new_shape: []const i64) Value {
        return self.reshapeOpts(src, new_shape, .{});
    }

    /// `tt.reshape` with the full named-param set —
    /// `tl.reshape(src, shape, can_reorder=...)` plus the `efficient_layout`
    /// escape hatch. Element type preserved from `src`.
    pub fn reshapeOpts(self: *Kernel, src: Value, new_shape: []const i64, opts: ReshapeOpts) Value {
        const result_ty = mlir.rankedTensorType(new_shape, src.elemType());
        return self.emit(ttir.reshape(self.ctx, src.inner, result_ty, opts.can_reorder, opts.efficient_layout, self.loc()));
    }

    /// `tl.trans(src, *dims)` / `tt.trans` — permute dimensions according to
    /// `order`. Result shape is `src.shape` permuted by `order`; element type
    /// preserved.
    pub fn trans(self: *Kernel, src: Value, order: []const i32) Value {
        const src_shape = src.shape();
        std.debug.assert(order.len == src_shape.len);
        var out_shape: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .{};
        for (order) |i| out_shape.appendAssumeCapacity(src_shape.get(@intCast(i)));
        const result_type = mlir.rankedTensorType(out_shape.constSlice(), src.elemType());
        return self.emit(ttir.trans(self.ctx, src.inner, order, result_type, self.loc()));
    }

    /// `tl.permute(src, *dims)` — alias for `trans`.
    pub fn permute(self: *Kernel, src: Value, order: []const i32) Value {
        return self.trans(src, order);
    }

    /// `tl.cat(input, other)` — concatenate two tensors along dim 0 (Python
    /// default). Both operands must share rank, element type, and dims ≥ 1;
    /// result dim 0 = sum of the operand dim 0s. Use `catOpts` for
    /// `can_reorder` / `dim`.
    pub fn cat(self: *Kernel, lhs: Value, rhs: Value) Value {
        return self.catOpts(lhs, rhs, .{});
    }

    /// `tl.cat(input, other, can_reorder=..., dim=...)` — concat with kwargs.
    /// Currently `dim != 0` is unsupported (Python decomposes via
    /// `join + permute + reshape`; we would need to mirror that).
    pub fn catOpts(self: *Kernel, lhs: Value, rhs: Value, opts: CatOpts) Value {
        std.debug.assert(opts.dim == 0); // TODO: general dim via join+permute+reshape
        _ = opts.can_reorder; // forwarded to ttir.cat when supported
        const ls = lhs.shape();
        const rs = rhs.shape();
        std.debug.assert(ls.len == rs.len);
        var out_shape: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .{};
        out_shape.appendAssumeCapacity(ls.get(0) + rs.get(0));
        for (1..ls.len) |i| out_shape.appendAssumeCapacity(ls.get(i));
        const result_type = mlir.rankedTensorType(out_shape.constSlice(), lhs.elemType());
        return self.emit(ttir.cat(self.ctx, lhs.inner, rhs.inner, result_type, self.loc()));
    }

    /// `tt.join` — stack two equal-shape tensors along a new minor dim of size 2.
    pub fn join(self: *Kernel, lhs: Value, rhs: Value) Value {
        const ls = lhs.shape();
        var out_shape: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .{};
        for (0..ls.len) |i| out_shape.appendAssumeCapacity(ls.get(i));
        out_shape.appendAssumeCapacity(2);
        const result_type = mlir.rankedTensorType(out_shape.constSlice(), lhs.elemType());
        return self.emit(ttir.join(self.ctx, lhs.inner, rhs.inner, result_type, self.loc()));
    }

    /// `tt.split` — split a tensor whose last dim is 2 into two tensors with
    /// that last dim dropped. Returns (lhs, rhs).
    pub fn split(self: *Kernel, src: Value) [2]Value {
        const ss = src.shape();
        std.debug.assert(ss.len >= 1 and ss.get(ss.len - 1) == 2);
        var out_shape: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .{};
        for (0..ss.len - 1) |i| out_shape.appendAssumeCapacity(ss.get(i));
        const result_type = mlir.rankedTensorType(out_shape.constSlice(), src.elemType());
        const op = ttir.split(self.ctx, src.inner, result_type, self.loc());
        _ = op.appendTo(self.currentBlock());
        return .{
            .{ .inner = op.result(0), .kernel = self },
            .{ .inner = op.result(1), .kernel = self },
        };
    }

    /// `tl.item(src)` / `tt.unsplat` — 1-element tensor → scalar of the same element type.
    pub fn item(self: *Kernel, src: Value) Value {
        return self.emit(ttir.unsplat(self.ctx, src.inner, src.elemType(), self.loc()));
    }

    /// `tt.gather` — `tl.gather(src, index, axis)`. All positional, no opts
    /// (mirrors Python: no kwargs). Output shape matches `indices`, element
    /// type matches `src`.
    pub fn gather(self: *Kernel, src: Value, indices: Value, axis: i32) Value {
        const result_ty = mlir.rankedTensorType(indices.shape().constSlice(), src.elemType());
        return self.emit(ttir.gather(self.ctx, src.inner, indices.inner, axis, result_ty, false, self.loc()));
    }

    /// `tt.histogram` without a mask — `tl.histogram(src, num_bins)` in Python.
    /// Output is `tensor<num_bins x i32>`. Use `histogramOpts` to pass a mask.
    pub fn histogram(self: *Kernel, src: Value, num_bins: i64) Value {
        return self.histogramOpts(src, num_bins, .{});
    }

    /// `tt.histogram` with the full named-param set —
    /// `tl.histogram(src, num_bins, mask=...)`.
    pub fn histogramOpts(self: *Kernel, src: Value, num_bins: i64, opts: HistogramOpts) Value {
        const result_ty = mlir.rankedTensorType(&.{num_bins}, DType.i32.toMlir(self.ctx));
        const mask_inner: ?*const mlir.Value = if (opts.mask) |m| m.inner else null;
        return self.emit(ttir.histogram(self.ctx, src.inner, mask_inner, result_ty, self.loc()));
    }

    /// `tl.device_assert(cond, msg="")` / `tt.assert` — device-side assert on a
    /// condition (i1 or tensor<...xi1>). Use `deviceAssertOpts` for a mask.
    pub fn deviceAssert(self: *Kernel, condition: Value, message: []const u8) void {
        self.deviceAssertOpts(condition, message, .{});
    }

    /// `tl.device_assert(cond, msg="", mask=...)`.
    pub fn deviceAssertOpts(self: *Kernel, condition: Value, message: []const u8, opts: DeviceAssertOpts) void {
        // TODO: threading mask through ttir.assert_ when supported.
        _ = opts.mask;
        _ = ttir.assert_(self.ctx, condition.inner, message, self.loc()).appendTo(self.currentBlock());
    }

    /// `tt.atomic_rmw` with default options — `tl.atomic_add(ptr, val)` (no
    /// mask, default sem/scope). Use `atomicRmwOpts` for mask / sem / scope.
    /// Returns the old value at ptr before the RMW.
    pub fn atomicRmw(self: *Kernel, rmw: ttir.RMWOp, ptr: Value, val: Value) Value {
        return self.atomicRmwOpts(rmw, ptr, val, .{});
    }

    /// `tt.atomic_rmw` with the full named-param set —
    /// `tl.atomic_add(ptr, val, mask=..., sem=..., scope=...)`.
    pub fn atomicRmwOpts(self: *Kernel, rmw: ttir.RMWOp, ptr: Value, val: Value, opts: AtomicRMWOpts) Value {
        const mask_inner: ?*const mlir.Value = if (opts.mask) |m| m.inner else null;
        return self.emit(ttir.atomic_rmw(self.ctx, rmw, ptr.inner, val.inner, mask_inner, opts.sem, opts.scope, self.loc()));
    }

    /// `tt.atomic_cas` with default sem/scope — `tl.atomic_cas(ptr, cmp, val)`.
    /// Returns the old value at ptr. Use `atomicCasOpts` for sem / scope.
    pub fn atomicCas(self: *Kernel, ptr: Value, cmp: Value, val: Value) Value {
        return self.atomicCasOpts(ptr, cmp, val, .{});
    }

    /// `tt.atomic_cas` with the full named-param set —
    /// `tl.atomic_cas(ptr, cmp, val, sem=..., scope=...)`.
    pub fn atomicCasOpts(self: *Kernel, ptr: Value, cmp: Value, val: Value, opts: AtomicCasOpts) Value {
        return self.emit(ttir.atomic_cas(self.ctx, ptr.inner, cmp.inner, val.inner, opts.sem, opts.scope, self.loc()));
    }

    /// `tt.call` — call another tt.func in this module by symbol name.
    pub fn call(self: *Kernel, callee: []const u8, operands: []const Value, result_types: []const *const mlir.Type) []Value {
        var buf: stdx.BoundedArray(*const mlir.Value, 32) = .{};
        for (operands) |v| buf.appendAssumeCapacity(v.inner);
        const op = ttir.call(self.ctx, callee, buf.constSlice(), result_types, null, null, self.loc());
        return self.emitMulti(op, result_types.len);
    }

    /// `tt.dot_scaled` — dot with microscaling factors; defaulted options.
    /// Result type follows `c_acc`'s type. Use `dotScaledOpts` for `fast_math`
    /// / `lhs_k_pack` / `rhs_k_pack`.
    pub fn dotScaled(
        self: *Kernel,
        a: Value,
        b: Value,
        c_acc: Value,
        a_scale: ?Value,
        b_scale: ?Value,
        a_elem_type: ttir.ScaleDotElemType,
        b_elem_type: ttir.ScaleDotElemType,
    ) Value {
        return self.dotScaledOpts(a, b, c_acc, a_scale, b_scale, a_elem_type, b_elem_type, .{});
    }

    /// `tt.dot_scaled` with the full named-param set. Result type = `c_acc.type_()`.
    pub fn dotScaledOpts(
        self: *Kernel,
        a: Value,
        b: Value,
        c_acc: Value,
        a_scale: ?Value,
        b_scale: ?Value,
        a_elem_type: ttir.ScaleDotElemType,
        b_elem_type: ttir.ScaleDotElemType,
        opts: DotScaledOpts,
    ) Value {
        const a_s: ?*const mlir.Value = if (a_scale) |s| s.inner else null;
        const b_s: ?*const mlir.Value = if (b_scale) |s| s.inner else null;
        return self.emit(ttir.dot_scaled(
            self.ctx,
            a.inner,
            b.inner,
            c_acc.inner,
            a_s,
            b_s,
            a_elem_type,
            b_elem_type,
            c_acc.type_(),
            opts.fast_math,
            opts.lhs_k_pack,
            opts.rhs_k_pack,
            self.loc(),
        ));
    }

    /// `tt.extern_elementwise` — call a library symbol pointwise with default
    /// options. Use `externElementwiseOpts` for `pure` and the rest of the
    /// named-param set.
    /// `tt.extern_elementwise` — result is `tensor<result_shape x result_dtype>`.
    /// For a scalar result, pass `&.{}` as `result_shape`.
    pub fn externElementwise(
        self: *Kernel,
        srcs: []const Value,
        result_shape: []const i64,
        result_dtype: DType,
        libname: []const u8,
        libpath: []const u8,
        symbol: []const u8,
    ) Value {
        return self.externElementwiseOpts(srcs, result_shape, result_dtype, libname, libpath, symbol, .{});
    }

    /// `tt.extern_elementwise` with the full named-param set.
    pub fn externElementwiseOpts(
        self: *Kernel,
        srcs: []const Value,
        result_shape: []const i64,
        result_dtype: DType,
        libname: []const u8,
        libpath: []const u8,
        symbol: []const u8,
        opts: ExternElementwiseOpts,
    ) Value {
        var buf: stdx.BoundedArray(*const mlir.Value, 8) = .{};
        for (srcs) |s| buf.appendAssumeCapacity(s.inner);
        const result_type: *const mlir.Type = if (result_shape.len == 0)
            result_dtype.toMlir(self.ctx)
        else
            mlir.rankedTensorType(result_shape, result_dtype.toMlir(self.ctx));
        return self.emit(ttir.extern_elementwise(
            self.ctx,
            buf.constSlice(),
            result_type,
            libname,
            libpath,
            symbol,
            opts.is_pure,
            self.loc(),
        ));
    }

    /// `tl.device_print(prefix, *args)` / `tt.print` — device-side print with
    /// default options (non-hex). `is_signed` must have one entry per `args`
    /// entry (Python infers this from Python types; we need it explicit). Use
    /// `devicePrintOpts` to set `hex = true`.
    pub fn devicePrint(
        self: *Kernel,
        prefix: []const u8,
        args: []const Value,
        is_signed: []const i32,
    ) void {
        self.devicePrintOpts(prefix, args, is_signed, .{});
    }

    /// `tl.device_print(prefix, *args, hex=...)` with the full named-param set.
    pub fn devicePrintOpts(
        self: *Kernel,
        prefix: []const u8,
        args: []const Value,
        is_signed: []const i32,
        opts: PrintOpts,
    ) void {
        var buf: stdx.BoundedArray(*const mlir.Value, 16) = .{};
        for (args) |a| buf.appendAssumeCapacity(a.inner);
        _ = ttir.print(self.ctx, prefix, opts.hex, buf.constSlice(), is_signed, self.loc()).appendTo(self.currentBlock());
    }

    /// `tt.make_tensor_descriptor` — build a TMA descriptor.
    /// `shape` / `strides` are runtime Values; `block_shape` / `dtype` describe
    /// the compile-time tile shape baked into the descriptor type.
    pub fn makeTensorDescriptor(
        self: *Kernel,
        base: Value,
        shape: []const Value,
        strides: []const Value,
        padding: ttir.PaddingOption,
        block_shape: []const i64,
        dtype: DType,
    ) Value {
        var shape_buf: stdx.BoundedArray(*const mlir.Value, 16) = .{};
        for (shape) |s| shape_buf.appendAssumeCapacity(s.inner);
        var strides_buf: stdx.BoundedArray(*const mlir.Value, 16) = .{};
        for (strides) |s| strides_buf.appendAssumeCapacity(s.inner);
        const result_type = ttir.tensorDescType(block_shape, dtype.toMlir(self.ctx), null);
        return self.emit(ttir.make_tensor_descriptor(
            self.ctx,
            base.inner,
            shape_buf.constSlice(),
            strides_buf.constSlice(),
            padding,
            result_type,
            self.loc(),
        ));
    }

    /// `tt.descriptor_load` — TMA load with default options. `shape` and
    /// `dtype` describe the tile being loaded. Use `descriptorLoadOpts` for
    /// cache / eviction modifiers.
    pub fn descriptorLoad(
        self: *Kernel,
        desc: Value,
        indices: []const Value,
        shape: []const i64,
        dtype: DType,
    ) Value {
        return self.descriptorLoadOpts(desc, indices, shape, dtype, .{});
    }

    /// `tt.descriptor_load` with the full named-param set.
    pub fn descriptorLoadOpts(
        self: *Kernel,
        desc: Value,
        indices: []const Value,
        shape: []const i64,
        dtype: DType,
        opts: DescriptorLoadOpts,
    ) Value {
        var idx_buf: stdx.BoundedArray(*const mlir.Value, 8) = .{};
        for (indices) |i| idx_buf.appendAssumeCapacity(i.inner);
        const result_ty = mlir.rankedTensorType(shape, dtype.toMlir(self.ctx));
        return self.emit(ttir.descriptor_load(self.ctx, desc.inner, idx_buf.constSlice(), result_ty, opts.cache_modifier, opts.eviction_policy, self.loc()));
    }

    /// `tt.descriptor_store` — TMA store.
    pub fn descriptorStore(self: *Kernel, desc: Value, src: Value, indices: []const Value) void {
        var idx_buf: stdx.BoundedArray(*const mlir.Value, 8) = .{};
        for (indices) |i| idx_buf.appendAssumeCapacity(i.inner);
        _ = ttir.descriptor_store(self.ctx, desc.inner, src.inner, idx_buf.constSlice(), self.loc()).appendTo(self.currentBlock());
    }

    /// `tt.descriptor_reduce` — TMA reducing store.
    pub fn descriptorReduce(
        self: *Kernel,
        kind: ttir.DescriptorReduceKind,
        desc: Value,
        src: Value,
        indices: []const Value,
    ) void {
        var idx_buf: stdx.BoundedArray(*const mlir.Value, 8) = .{};
        for (indices) |i| idx_buf.appendAssumeCapacity(i.inner);
        _ = ttir.descriptor_reduce(self.ctx, kind, desc.inner, src.inner, idx_buf.constSlice(), self.loc()).appendTo(self.currentBlock());
    }

    /// `tt.descriptor_gather` — TMA gather (rows by x_offsets + single y_offset).
    /// `shape` / `dtype` describe the gathered tile.
    pub fn descriptorGather(
        self: *Kernel,
        desc: Value,
        x_offsets: Value,
        y_offset: Value,
        shape: []const i64,
        dtype: DType,
    ) Value {
        const result_ty = mlir.rankedTensorType(shape, dtype.toMlir(self.ctx));
        return self.emit(ttir.descriptor_gather(self.ctx, desc.inner, x_offsets.inner, y_offset.inner, result_ty, self.loc()));
    }

    /// `tt.descriptor_scatter` — TMA scatter.
    pub fn descriptorScatter(self: *Kernel, desc: Value, x_offsets: Value, y_offset: Value, src: Value) void {
        _ = ttir.descriptor_scatter(self.ctx, desc.inner, x_offsets.inner, y_offset.inner, src.inner, self.loc()).appendTo(self.currentBlock());
    }

    // ==================== math dialect ====================

    pub fn exp(self: *Kernel, x: Value) Value {
        return self.emit(math.exp(self.ctx, x.inner, self.loc()));
    }
    pub fn exp2(self: *Kernel, x: Value) Value {
        return self.emit(math.exp2(self.ctx, x.inner, self.loc()));
    }
    pub fn log(self: *Kernel, x: Value) Value {
        return self.emit(math.log(self.ctx, x.inner, self.loc()));
    }
    pub fn log2(self: *Kernel, x: Value) Value {
        return self.emit(math.log2(self.ctx, x.inner, self.loc()));
    }
    pub fn sqrt(self: *Kernel, x: Value) Value {
        return self.emit(math.sqrt(self.ctx, x.inner, self.loc()));
    }
    pub fn rsqrt(self: *Kernel, x: Value) Value {
        return self.emit(math.rsqrt(self.ctx, x.inner, self.loc()));
    }
    pub fn sin(self: *Kernel, x: Value) Value {
        return self.emit(math.sin(self.ctx, x.inner, self.loc()));
    }
    pub fn cos(self: *Kernel, x: Value) Value {
        return self.emit(math.cos(self.ctx, x.inner, self.loc()));
    }
    pub fn tan(self: *Kernel, x: Value) Value {
        return self.emit(math.tan(self.ctx, x.inner, self.loc()));
    }
    pub fn tanh(self: *Kernel, x: Value) Value {
        return self.emit(math.tanh(self.ctx, x.inner, self.loc()));
    }
    pub fn erf(self: *Kernel, x: Value) Value {
        return self.emit(math.erf(self.ctx, x.inner, self.loc()));
    }
    pub fn absf(self: *Kernel, x: Value) Value {
        return self.emit(math.absf(self.ctx, x.inner, self.loc()));
    }
    pub fn absi(self: *Kernel, x: Value) Value {
        return self.emit(math.absi(self.ctx, x.inner, self.loc()));
    }
    /// `tl.abs(x)` — absolute value, auto-dispatch by element type (float→`absf`,
    /// signed int→`absi`, unsigned int→no-op).
    pub fn abs(self: *Kernel, x: Value) Value {
        if (x.isFloatElem()) return self.absf(x);
        return self.absi(x);
    }
    pub fn floor(self: *Kernel, x: Value) Value {
        return self.emit(math.floor(self.ctx, x.inner, self.loc()));
    }
    pub fn ceil(self: *Kernel, x: Value) Value {
        return self.emit(math.ceil(self.ctx, x.inner, self.loc()));
    }
    pub fn powf(self: *Kernel, x: Value, y: Value) Value {
        return self.emit(math.powf(self.ctx, x.inner, y.inner, self.loc()));
    }
    pub fn fma(self: *Kernel, a: Value, b: Value, c_: Value) Value {
        return self.emit(math.fma(self.ctx, a.inner, b.inner, c_.inner, self.loc()));
    }
    /// `math.clampf` — distinct from `tt.clampf` (no propagateNan attr).
    pub fn mathClampf(self: *Kernel, value: Value, lo: Value, hi: Value) Value {
        return self.emit(math.clampf(self.ctx, value.inner, lo.inner, hi.inner, self.loc()));
    }
    /// `math.sincos` — returns (sin, cos).
    pub fn sincos(self: *Kernel, x: Value) [2]Value {
        const op = math.sincos(self.ctx, x.inner, self.loc());
        _ = op.appendTo(self.currentBlock());
        return .{ .{ .inner = op.result(0), .kernel = self }, .{ .inner = op.result(1), .kernel = self } };
    }
    /// `math.fpowi` — float base, integer exponent.
    pub fn fpowi(self: *Kernel, base: Value, power: Value) Value {
        return self.emit(math.fpowi(self.ctx, base.inner, power.inner, self.loc()));
    }
    pub fn ipowi(self: *Kernel, base: Value, power: Value) Value {
        return self.emit(math.ipowi(self.ctx, base.inner, power.inner, self.loc()));
    }
    pub fn atan2(self: *Kernel, y: Value, x: Value) Value {
        return self.emit(math.atan2(self.ctx, y.inner, x.inner, self.loc()));
    }
    pub fn copysign(self: *Kernel, mag: Value, sign: Value) Value {
        return self.emit(math.copysign(self.ctx, mag.inner, sign.inner, self.loc()));
    }
    pub fn mathRound(self: *Kernel, x: Value) Value {
        return self.emit(math.round(self.ctx, x.inner, self.loc()));
    }
    pub fn mathRoundEven(self: *Kernel, x: Value) Value {
        return self.emit(math.roundeven(self.ctx, x.inner, self.loc()));
    }
    pub fn mathTrunc(self: *Kernel, x: Value) Value {
        return self.emit(math.trunc(self.ctx, x.inner, self.loc()));
    }

    /// `arith.convertf` — same-bitwidth float cast (e.g. f16 ↔ bf16) with
    /// default rounding. Shape preserved from `src`. Use `convertfOpts` for
    /// `fastMath`.
    pub fn convertf(self: *Kernel, src: Value, out_dtype: DType) Value {
        return self.convertfOpts(src, out_dtype, .{});
    }
    /// `arith.convertf` with the full named-param set.
    pub fn convertfOpts(self: *Kernel, src: Value, out_dtype: DType, opts: arith.ConvertFOpts) Value {
        return self.emit(arith.convertf(self.ctx, src.inner, self.swapElem(src, out_dtype), opts, self.loc()));
    }
    /// `arith.scaling_truncf` — MXFP downcast with per-block scales, default
    /// options. Shape preserved from `src`. Use `scalingTruncfOpts` for
    /// `fastMath`.
    pub fn scalingTruncf(self: *Kernel, src: Value, scale: Value, out_dtype: DType) Value {
        return self.scalingTruncfOpts(src, scale, out_dtype, .{});
    }
    /// `arith.scaling_truncf` with the full named-param set.
    pub fn scalingTruncfOpts(self: *Kernel, src: Value, scale: Value, out_dtype: DType, opts: arith.ConvertFOpts) Value {
        return self.emit(arith.scaling_truncf(self.ctx, src.inner, scale.inner, self.swapElem(src, out_dtype), opts, self.loc()));
    }

    // ==================== tt.map_elementwise / elementwise_inline_asm ====================

    /// `tt.map_elementwise` — body builder receives one scalar Value per
    /// `srcs` tensor (repeated `pack` times, flattened), and must return the
    /// scalar results to feed back via `tt.map_elementwise.return`.
    pub fn mapElementwise(
        self: *Kernel,
        srcs: []const Value,
        pack: i32,
        scalar_arg_types: []const *const mlir.Type,
        result_tensor_types: []const *const mlir.Type,
        comptime body_fn: anytype,
        body_ctx: anytype,
    ) []Value {
        var locs: stdx.BoundedArray(*const mlir.Location, 32) = .{};
        for (scalar_arg_types) |_| locs.appendAssumeCapacity(self.loc());
        const body_block = mlir.Block.init(scalar_arg_types, locs.constSlice());

        self.pushBlock(body_block);
        var block_args: stdx.BoundedArray(Value, 32) = .{};
        for (0..scalar_arg_types.len) |i| block_args.appendAssumeCapacity(.{ .inner = body_block.argument(i), .kernel = self });
        const out_vals: []const Value = body_fn(self, block_args.constSlice(), body_ctx);
        var ret_vals: stdx.BoundedArray(*const mlir.Value, 16) = .{};
        for (out_vals) |v| ret_vals.appendAssumeCapacity(v.inner);
        _ = ttir.map_elementwise_return(self.ctx, ret_vals.constSlice(), self.loc()).appendTo(body_block);
        self.popBlock();

        var src_vals: stdx.BoundedArray(*const mlir.Value, 16) = .{};
        for (srcs) |s| src_vals.appendAssumeCapacity(s.inner);

        const op = ttir.map_elementwise(
            self.ctx,
            src_vals.constSlice(),
            pack,
            body_block,
            result_tensor_types,
            self.loc(),
        );
        return self.emitMulti(op, result_tensor_types.len);
    }

    /// `tl.inline_asm_elementwise(asm, constraints, args, dtype)` — inline asm
    /// pointwise op with default options. Use `inlineAsmElementwiseOpts` for
    /// `is_pure` / `pack`.
    pub fn inlineAsmElementwise(
        self: *Kernel,
        asm_string: []const u8,
        constraints: []const u8,
        args: []const Value,
        result_types: []const *const mlir.Type,
    ) []Value {
        return self.inlineAsmElementwiseOpts(asm_string, constraints, args, result_types, .{});
    }

    /// `tl.inline_asm_elementwise(asm, constraints, args, dtype, is_pure=..., pack=...)`
    /// with the full named-param set.
    pub fn inlineAsmElementwiseOpts(
        self: *Kernel,
        asm_string: []const u8,
        constraints: []const u8,
        args: []const Value,
        result_types: []const *const mlir.Type,
        opts: InlineAsmOpts,
    ) []Value {
        var buf: stdx.BoundedArray(*const mlir.Value, 16) = .{};
        for (args) |a| buf.appendAssumeCapacity(a.inner);
        const op = ttir.elementwise_inline_asm(self.ctx, asm_string, constraints, buf.constSlice(), result_types, opts.is_pure, opts.pack, self.loc());
        return self.emitMulti(op, result_types.len);
    }

    // ==================== scf.parallel / scf.reduce / scf.forall ====================

    /// `scf.parallel` with optional reductions.
    ///
    /// `body_fn(k, ivs, body_ctx) []const Value` — returns the per-reduction
    /// operand values. Must match `reduction_result_types` in length.
    ///
    /// `reduction_combiners` — one combiner per reduction; each combiner
    /// `fn(k: *Kernel, lhs: Value, rhs: Value, body_ctx) Value` returns the
    /// combined scalar, which becomes `scf.reduce.return`'s operand.
    pub fn parallel(
        self: *Kernel,
        lbs: []const Value,
        ubs: []const Value,
        steps: []const Value,
        inits: []const Value,
        reduction_result_types: []const *const mlir.Type,
        comptime body_fn: anytype,
        comptime combiners: anytype,
        body_ctx: anytype,
    ) []Value {
        std.debug.assert(lbs.len == ubs.len);
        std.debug.assert(inits.len == reduction_result_types.len);

        var iv_types: stdx.BoundedArray(*const mlir.Type, 16) = .{};
        var iv_locs: stdx.BoundedArray(*const mlir.Location, 16) = .{};
        const idx_ty = mlir.indexType(self.ctx);
        for (lbs) |_| {
            iv_types.appendAssumeCapacity(idx_ty);
            iv_locs.appendAssumeCapacity(self.loc());
        }
        const body_block = mlir.Block.init(iv_types.constSlice(), iv_locs.constSlice());

        self.pushBlock(body_block);
        var iv_args: stdx.BoundedArray(Value, 16) = .{};
        for (0..lbs.len) |i| iv_args.appendAssumeCapacity(.{ .inner = body_block.argument(i), .kernel = self });
        const reduce_operands: []const Value = body_fn(self, iv_args.constSlice(), body_ctx);
        std.debug.assert(reduce_operands.len == reduction_result_types.len);

        // Build scf.reduce with one region per reduction.
        var red_operand_raw: stdx.BoundedArray(*const mlir.Value, 16) = .{};
        for (reduce_operands) |v| red_operand_raw.appendAssumeCapacity(v.inner);

        var red_blocks: stdx.BoundedArray(*mlir.Block, 16) = .{};
        inline for (combiners, 0..) |comb, i| {
            const arg_ty = reduce_operands[i].type_();
            const arg_locs: [2]*const mlir.Location = .{ self.loc(), self.loc() };
            const red_block = mlir.Block.init(&.{ arg_ty, arg_ty }, &arg_locs);
            self.pushBlock(red_block);
            const lhs_v: Value = .{ .inner = red_block.argument(0), .kernel = self };
            const rhs_v: Value = .{ .inner = red_block.argument(1), .kernel = self };
            const combined: Value = comb(self, lhs_v, rhs_v, body_ctx);
            _ = scf.reduce_return(self.ctx, combined.inner, self.loc()).appendTo(red_block);
            self.popBlock();
            red_blocks.appendAssumeCapacity(red_block);
        }

        _ = scf.reduce(self.ctx, red_operand_raw.constSlice(), red_blocks.constSlice(), self.loc()).appendTo(body_block);
        self.popBlock();

        var lb_raw: stdx.BoundedArray(*const mlir.Value, 16) = .{};
        var ub_raw: stdx.BoundedArray(*const mlir.Value, 16) = .{};
        var step_raw: stdx.BoundedArray(*const mlir.Value, 16) = .{};
        var init_raw: stdx.BoundedArray(*const mlir.Value, 16) = .{};
        for (lbs) |v| lb_raw.appendAssumeCapacity(v.inner);
        for (ubs) |v| ub_raw.appendAssumeCapacity(v.inner);
        for (steps) |v| step_raw.appendAssumeCapacity(v.inner);
        for (inits) |v| init_raw.appendAssumeCapacity(v.inner);

        const op = scf.parallel(
            self.ctx,
            lb_raw.constSlice(),
            ub_raw.constSlice(),
            step_raw.constSlice(),
            init_raw.constSlice(),
            body_block,
            reduction_result_types,
            self.loc(),
        );
        _ = op.appendTo(self.currentBlock());

        const out = self.arena.allocator().alloc(Value, reduction_result_types.len) catch @panic("Kernel.parallel OOM");
        for (0..reduction_result_types.len) |i| out[i] = .{ .inner = op.result(i), .kernel = self };
        return out;
    }

    // ==================== reduce / scan ====================

    /// `tt.reduce` along `axis`, single-input. See `ReduceArgs` for fields.
    /// `ctx` is separate from `args` so `ReduceArgs(@TypeOf(ctx))` can be
    /// inferred.
    pub fn reduce(self: *Kernel, ctx: anytype, args: ReduceArgs(@TypeOf(ctx))) Value {
        const region = mlir.Block.init(&.{ args.elem, args.elem }, &.{ self.loc(), self.loc() });

        self.pushBlock(region);
        const lhs: Value = .{ .inner = region.argument(0), .kernel = self };
        const rhs: Value = .{ .inner = region.argument(1), .kernel = self };
        const combined: Value = args.combine(self, lhs, rhs, ctx);
        _ = ttir.reduce_return(self.ctx, &.{combined.inner}, self.loc()).appendTo(region);
        self.popBlock();

        const op = ttir.reduce(self.ctx, &.{args.src.inner}, args.axis, region, &.{args.result}, self.loc());
        _ = op.appendTo(self.currentBlock());
        return .{ .inner = op.result(0), .kernel = self };
    }

    /// `tt.scan` along `axis`, single-input. See `ScanArgs` for fields.
    pub fn scan(self: *Kernel, ctx: anytype, args: ScanArgs(@TypeOf(ctx))) Value {
        const region = mlir.Block.init(&.{ args.elem, args.elem }, &.{ self.loc(), self.loc() });

        self.pushBlock(region);
        const lhs: Value = .{ .inner = region.argument(0), .kernel = self };
        const rhs: Value = .{ .inner = region.argument(1), .kernel = self };
        const combined: Value = args.combine(self, lhs, rhs, ctx);
        _ = ttir.scan_return(self.ctx, &.{combined.inner}, self.loc()).appendTo(region);
        self.popBlock();

        const op = ttir.scan(self.ctx, &.{args.src.inner}, args.axis, args.reverse, region, &.{args.result}, self.loc());
        _ = op.appendTo(self.currentBlock());
        return .{ .inner = op.result(0), .kernel = self };
    }

    // ==================== SCF regions ====================

    /// `scf.for` with iter_args. `lower`, `upper`, and `step` accept Values,
    /// comptime ints, or runtime Zig ints (see `Kernel.lift` / comptime-adapt
    /// rules). See `ForArgs` for the rest of the fields. Returns the iter_args'
    /// post-loop values (same length as `args.inits`).
    pub fn forLoop(
        self: *Kernel,
        ctx: anytype,
        lower: anytype,
        upper: anytype,
        step: anytype,
        args: ForArgs(@TypeOf(ctx)),
    ) []Value {
        const iter_args = args.inits;
        std.debug.assert(iter_args.len <= 31);

        const lb_v: Value = if (@TypeOf(lower) == Value) lower else self.lift(lower);
        const ub_v: Value = if (@TypeOf(upper) == Value) upper else self.constMatching(upper, lb_v.type_());
        const step_v: Value = if (@TypeOf(step) == Value) step else self.constMatching(step, lb_v.type_());

        var block_types: stdx.BoundedArray(*const mlir.Type, 32) = .{};
        var block_locs: stdx.BoundedArray(*const mlir.Location, 32) = .{};
        block_types.appendAssumeCapacity(lb_v.type_());
        block_locs.appendAssumeCapacity(self.loc());
        for (iter_args) |ia| {
            block_types.appendAssumeCapacity(ia.type_());
            block_locs.appendAssumeCapacity(self.loc());
        }

        const body = mlir.Block.init(block_types.constSlice(), block_locs.constSlice());

        self.pushBlock(body);
        const iv: Value = .{ .inner = body.argument(0), .kernel = self };
        var barg_buf: stdx.BoundedArray(Value, 32) = .{};
        for (0..iter_args.len) |i| {
            barg_buf.appendAssumeCapacity(.{ .inner = body.argument(i + 1), .kernel = self });
        }

        const yielded: []const Value = args.body(self, iv, barg_buf.constSlice(), ctx);

        var yield_values: stdx.BoundedArray(*const mlir.Value, 32) = .{};
        for (yielded) |y| yield_values.appendAssumeCapacity(y.inner);
        _ = scf.yield(self.ctx, yield_values.constSlice(), self.loc()).appendTo(body);

        // Pop BEFORE appending the scf.for — otherwise currentBlock()
        // still returns the body we just built, and appending scf.for
        // into its own body creates a cycle that hangs the IR printer.
        self.popBlock();

        var init_values: stdx.BoundedArray(*const mlir.Value, 32) = .{};
        for (iter_args) |ia| init_values.appendAssumeCapacity(ia.inner);

        const for_op = scf.for_(
            self.ctx,
            lb_v.inner,
            ub_v.inner,
            step_v.inner,
            init_values.constSlice(),
            body,
            .{},
            self.loc(),
        );
        _ = for_op.appendTo(self.currentBlock());

        const out = self.arena.allocator().alloc(Value, iter_args.len) catch @panic("Kernel.forLoop OOM");
        for (0..iter_args.len) |i| out[i] = .{ .inner = for_op.result(i), .kernel = self };
        return out;
    }

    /// `scf.if` — build a conditional with optional results. See `IfArgs` for
    /// fields. `else_` is optional; omitting it builds an empty else-block
    /// (requires `results = &.{}`). To express "if not cond: do X", invert the
    /// condition rather than providing an empty `then_`.
    pub fn ifThenElse(self: *Kernel, ctx: anytype, args: IfArgs(@TypeOf(ctx))) []Value {
        const then_block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(then_block);
        const then_vals: []const Value = args.then_(self, ctx);
        var then_yield: stdx.BoundedArray(*const mlir.Value, 32) = .{};
        for (then_vals) |v| then_yield.appendAssumeCapacity(v.inner);
        _ = scf.yield(self.ctx, then_yield.constSlice(), self.loc()).appendTo(then_block);
        self.popBlock();

        const else_block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(else_block);
        if (args.else_) |else_fn| {
            const else_vals: []const Value = else_fn(self, ctx);
            var else_yield: stdx.BoundedArray(*const mlir.Value, 32) = .{};
            for (else_vals) |v| else_yield.appendAssumeCapacity(v.inner);
            _ = scf.yield(self.ctx, else_yield.constSlice(), self.loc()).appendTo(else_block);
        } else {
            std.debug.assert(args.results.len == 0);
            _ = scf.yield(self.ctx, &.{}, self.loc()).appendTo(else_block);
        }
        self.popBlock();

        const if_op = scf.if_(
            self.ctx,
            args.cond.inner,
            args.results,
            then_block,
            else_block,
            self.loc(),
        );
        _ = if_op.appendTo(self.currentBlock());

        const out = self.arena.allocator().alloc(Value, args.results.len) catch @panic("Kernel.ifThenElse OOM");
        for (0..args.results.len) |i| out[i] = .{ .inner = if_op.result(i), .kernel = self };
        return out;
    }

    /// `scf.while` — generic while loop with separate "before" (cond) and
    /// "after" (body) regions. See `WhileArgs` / `WhileBefore` for fields.
    pub fn whileLoop(self: *Kernel, ctx: anytype, args: WhileArgs(@TypeOf(ctx))) []Value {
        const inits = args.inits;
        const arg_types_after = args.after_types;
        // Before-region arg types == init types.
        var before_types: stdx.BoundedArray(*const mlir.Type, 32) = .{};
        var before_locs: stdx.BoundedArray(*const mlir.Location, 32) = .{};
        for (inits) |v| {
            before_types.appendAssumeCapacity(v.type_());
            before_locs.appendAssumeCapacity(self.loc());
        }
        const before_block = mlir.Block.init(before_types.constSlice(), before_locs.constSlice());

        var after_locs: stdx.BoundedArray(*const mlir.Location, 32) = .{};
        for (arg_types_after) |_| after_locs.appendAssumeCapacity(self.loc());
        const after_block = mlir.Block.init(arg_types_after, after_locs.constSlice());

        // Populate "before".
        self.pushBlock(before_block);
        var before_args: stdx.BoundedArray(Value, 32) = .{};
        for (0..inits.len) |i| before_args.appendAssumeCapacity(.{ .inner = before_block.argument(i), .kernel = self });
        const before_out = args.before(self, before_args.constSlice(), ctx);
        // `before_out` is expected to be a struct { cond: Value, forwarded: []const Value }
        var cond_args: stdx.BoundedArray(*const mlir.Value, 32) = .{};
        for (before_out.forwarded) |v| cond_args.appendAssumeCapacity(v.inner);
        _ = scf.condition(self.ctx, before_out.cond.inner, cond_args.constSlice(), self.loc()).appendTo(before_block);
        self.popBlock();

        // Populate "after".
        self.pushBlock(after_block);
        var after_args: stdx.BoundedArray(Value, 32) = .{};
        for (0..arg_types_after.len) |i| after_args.appendAssumeCapacity(.{ .inner = after_block.argument(i), .kernel = self });
        const after_out = args.after(self, after_args.constSlice(), ctx);
        var yield_vals: stdx.BoundedArray(*const mlir.Value, 32) = .{};
        for (after_out) |v| yield_vals.appendAssumeCapacity(v.inner);
        _ = scf.yield(self.ctx, yield_vals.constSlice(), self.loc()).appendTo(after_block);
        self.popBlock();

        var init_values: stdx.BoundedArray(*const mlir.Value, 32) = .{};
        for (inits) |v| init_values.appendAssumeCapacity(v.inner);

        const w = scf.while_(
            self.ctx,
            init_values.constSlice(),
            arg_types_after,
            before_block,
            after_block,
            self.loc(),
        );
        _ = w.appendTo(self.currentBlock());

        const out = self.arena.allocator().alloc(Value, arg_types_after.len) catch @panic("Kernel.whileLoop OOM");
        for (0..arg_types_after.len) |i| out[i] = .{ .inner = w.result(i), .kernel = self };
        return out;
    }

    // ==================== finalization ====================

    /// Append `tt.return` with the given results, verify the module, and
    /// serialize to a NUL-terminated TTIR string owned by the kernel's
    /// allocator (passed to `init`). Caller must `defer allocator.free(ir)`.
    pub fn finish(self: *Kernel, results: []const Value) FinishError![:0]const u8 {
        var return_values: stdx.BoundedArray(*const mlir.Value, 16) = .{};
        for (results) |r| return_values.appendAssumeCapacity(r.inner);
        _ = ttir.return_(self.ctx, return_values.constSlice(), self.loc()).appendTo(self.entry_block);

        if (!self.module.operation().verify()) {
            return error.InvalidMlir;
        }

        var al: std.Io.Writer.Allocating = .init(self.allocator);
        defer al.deinit();
        try al.writer.print("{f}", .{self.module.operation()});

        return try self.allocator.dupeZ(u8, al.written());
    }
};

test {
    std.testing.refAllDecls(@This());
}

fn setupTestContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    inline for (.{ "func", "arith", "scf", "math", "tt" }) |d| {
        mlir.DialectHandle.fromString(d).insertDialect(registry);
    }
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

test "Kernel builds a trivial tt.func round-trip" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    var kernel = try Kernel.init(std.testing.allocator, ctx, "add_one", &.{
        .{ .name = "a_ptr", .kind = .{ .ptr = .{ .dtype = .f32 } } },
        .{ .name = "b_ptr", .kind = .{ .ptr = .{ .dtype = .f32 } } },
    }, &.{});
    defer kernel.deinit();

    const a_ptr = kernel.arg(0);
    const b_ptr = kernel.arg(1);

    const loaded = kernel.load(a_ptr);
    const one = kernel.lift(@as(f32, 1.0));
    const summed = kernel.addf(loaded, one);
    kernel.store(b_ptr, summed);

    const ir = try kernel.finish(&.{});
    defer std.testing.allocator.free(ir);

    // Text must contain the expected operators.
    try std.testing.expect(std.mem.indexOf(u8, ir, "tt.func") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "tt.load") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "arith.addf") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "tt.store") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "tt.return") != null);

    // Round-trip: re-parse as a module and verify.
    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "Kernel with scf.for iter_args" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    // tt.func @sum_range(%base: !tt.ptr<f32>) {
    //   %lb = arith.constant 0 : i32
    //   %ub = arith.constant 8 : i32
    //   %step = arith.constant 1 : i32
    //   %init = arith.constant 0.0 : f32
    //   %sum = scf.for %iv = %lb to %ub step %step
    //       iter_args(%acc = %init) -> f32 : i32 {
    //     %v = tt.load %base : !tt.ptr<f32>
    //     %next = arith.addf %acc, %v : f32
    //     scf.yield %next : f32
    //   }
    //   tt.store %base, %sum
    //   tt.return
    // }
    var kernel = try Kernel.init(std.testing.allocator, ctx, "sum_range", &.{
        .{ .name = "base", .kind = .{ .ptr = .{ .dtype = .f32 } } },
    }, &.{});
    defer kernel.deinit();

    const base = kernel.arg(0);
    const lb = kernel.lift(@as(i32, 0));
    const ub = kernel.lift(@as(i32, 8));
    const step = kernel.lift(@as(i32, 1));
    const init_acc = kernel.lift(@as(f32, 0.0));

    const Ctx = struct { base: Value };
    const body = struct {
        fn call(k: *Kernel, iv: Value, iter: []const Value, c: Ctx) []const Value {
            _ = iv;
            const v = k.load(c.base);
            return k.yield(.{k.addf(iter[0], v)});
        }
    }.call;

    const loop_results = kernel.forLoop(Ctx{ .base = base }, lb, ub, step, .{
        .inits = &.{init_acc},
        .body = body,
    });
    kernel.store(base, loop_results[0]);

    const ir = try kernel.finish(&.{});
    defer std.testing.allocator.free(ir);

    try std.testing.expect(std.mem.indexOf(u8, ir, "scf.for") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "iter_args") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "scf.yield") != null);

    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "Kernel with scf.if" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    // tt.func @branch(%a: !tt.ptr<f32>) {
    //   %c1 = arith.constant 1 : i32
    //   %c2 = arith.constant 2 : i32
    //   %eq = arith.cmpi eq, %c1, %c2 : i32
    //   %v = scf.if %eq -> f32 {
    //     %one = arith.constant 1.0 : f32
    //     scf.yield %one : f32
    //   } else {
    //     %two = arith.constant 2.0 : f32
    //     scf.yield %two : f32
    //   }
    //   tt.store %a, %v
    //   tt.return
    // }
    var kernel = try Kernel.init(std.testing.allocator, ctx, "branch", &.{
        .{ .name = "a", .kind = .{ .ptr = .{ .dtype = .f32 } } },
    }, &.{});
    defer kernel.deinit();

    const a_ptr = kernel.arg(0);
    const c1 = kernel.lift(@as(i32, 1));
    const c2 = kernel.lift(@as(i32, 2));
    const eq = kernel.cmpi(.eq, c1, c2);

    const then_fn = struct {
        fn call(k: *Kernel, _: void) []const Value {
            return k.yield(.{k.lift(@as(f32, 1.0))});
        }
    }.call;
    const else_fn = struct {
        fn call(k: *Kernel, _: void) []const Value {
            return k.yield(.{k.lift(@as(f32, 2.0))});
        }
    }.call;

    const f32_ty = mlir.floatType(ctx, .f32);
    const results = kernel.ifThenElse({}, .{
        .cond = eq,
        .results = &.{f32_ty},
        .then_ = then_fn,
        .else_ = else_fn,
    });
    kernel.store(a_ptr, results[0]);

    const ir = try kernel.finish(&.{});
    defer std.testing.allocator.free(ir);

    try std.testing.expect(std.mem.indexOf(u8, ir, "scf.if") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "else") != null);

    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "tt.bitcast / tt.int_to_ptr / tt.fp_to_fp round-trip" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    var kernel = try Kernel.init(std.testing.allocator, ctx, "cast_kernel", &.{
        .{ .name = "iptr", .kind = .{ .ptr = .{ .dtype = .i64 } } },
        .{ .name = "fptr", .kind = .{ .ptr = .{ .dtype = .f32 } } },
        .{ .name = "half_ptr", .kind = .{ .ptr = .{ .dtype = .f16 } } },
    }, &.{});
    defer kernel.deinit();

    const iptr = kernel.arg(0);
    const fptr = kernel.arg(1);
    const half_ptr = kernel.arg(2);

    // Round-trip: i64 → ptr<f32> → int → ptr<f32>.
    const raw = kernel.load(iptr);
    const cast_ptr = kernel.intToPtr(raw, .f32, 1);
    const back_int = kernel.ptrToInt(cast_ptr, .i64);
    _ = back_int;

    // tt.bitcast f32 → i32 scalar.
    const fval = kernel.load(fptr);
    const bits = kernel.bitcast(fval, .i32);
    _ = bits;

    // fp_to_fp f32 → f16 with default rounding (rtne).
    const narrowed = kernel.fpToFp(fval, .f16);
    kernel.store(half_ptr, narrowed);

    const ir = try kernel.finish(&.{});
    defer std.testing.allocator.free(ir);

    try std.testing.expect(std.mem.indexOf(u8, ir, "tt.int_to_ptr") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "tt.ptr_to_int") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "tt.bitcast") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "tt.fp_to_fp") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "rounding = rtne") != null);

    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "math dialect round-trip (softmax-like bits)" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    var kernel = try Kernel.init(std.testing.allocator, ctx, "softmax_bits", &.{
        .{ .name = "x_ptr", .kind = .{ .ptr = .{ .dtype = .f32 } } },
        .{ .name = "out_ptr", .kind = .{ .ptr = .{ .dtype = .f32 } } },
    }, &.{});
    defer kernel.deinit();

    const x_ptr = kernel.arg(0);
    const out_ptr = kernel.arg(1);

    const x = kernel.load(x_ptr);

    const e = kernel.exp2(x);
    const sq = kernel.sqrt(e);
    const rq = kernel.rsqrt(sq);
    const t = kernel.tanh(rq);
    const e2 = kernel.erf(t);

    kernel.store(out_ptr, e2);

    const ir = try kernel.finish(&.{});
    defer std.testing.allocator.free(ir);

    try std.testing.expect(std.mem.indexOf(u8, ir, "math.exp2") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "math.sqrt") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "math.rsqrt") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "math.tanh") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "math.erf") != null);

    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "tt.atomic_rmw round-trip" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    var kernel = try Kernel.init(std.testing.allocator, ctx, "atomic_incr", &.{
        .{ .name = "counter", .kind = .{ .ptr = .{ .dtype = .i32 } } },
    }, &.{});
    defer kernel.deinit();

    const counter = kernel.arg(0);
    const one = kernel.lift(@as(i32, 1));
    _ = kernel.atomicRmw(.add, counter, one);

    const ir = try kernel.finish(&.{});
    defer std.testing.allocator.free(ir);

    try std.testing.expect(std.mem.indexOf(u8, ir, "tt.atomic_rmw") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "add") != null);

    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

// =============================================================================
// Parity with Triton's own canonical tests at `triton/test/Triton/ops.mlir`.
// Each test reproduces the body of a `tt.func` in that file, prints the IR,
// and round-trips it through `mlir.Module.parse` + `verify()`. Purpose: prove
// our native Zig builder produces IR that MLIR accepts for every op shape
// exercised by upstream.
// =============================================================================

fn parityPrintAndVerify(ctx: *mlir.Context, module: *mlir.Module) !void {
    var al: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer al.deinit();
    try al.writer.print("{f}", .{module.operation()});
    const parsed = try mlir.Module.parse(ctx, al.written());
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "ops_mlir parity: cast_ops" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    var k = try Kernel.init(std.testing.allocator, ctx, "cast_ops", &.{
        .{ .name = "scalar_ptr", .kind = .{ .ptr = .{ .dtype = .f32, .divisibility = null } } },
        .{ .name = "scalar_f32", .kind = .{ .scalar = .f32 } },
        .{ .name = "scalar_i64", .kind = .{ .scalar = .i64 } },
    }, &.{});
    defer k.deinit();

    const sp = k.arg(0);
    const sf = k.arg(1);
    const si = k.arg(2);

    // scalar ↔ scalar
    _ = k.intToPtr(si, .f32, 1);
    _ = k.ptrToInt(sp, .i64);
    _ = k.truncf(sf, .f16);

    // 0D-tensor ↔ 0D-tensor
    const t_ptr_0d = k.splat(sp, &.{});
    const t_f32_0d = k.splat(sf, &.{});
    const t_i64_0d = k.splat(si, &.{});
    _ = k.intToPtr(t_i64_0d, .f32, 1);
    _ = k.ptrToInt(t_ptr_0d, .i64);
    _ = k.truncf(t_f32_0d, .f16);

    // 1D-tensor ↔ 1D-tensor
    const t_ptr_1d = k.splat(sp, &.{16});
    const t_f32_1d = k.splat(sf, &.{16});
    const t_i64_1d = k.splat(si, &.{16});
    _ = k.intToPtr(t_i64_1d, .f32, 1);
    _ = k.ptrToInt(t_ptr_1d, .i64);
    _ = k.truncf(t_f32_1d, .f16);

    const __ir = try k.finish(&.{}); std.testing.allocator.free(__ir);
    // finish() also verifies; still round-trip the textual form.
    try parityPrintAndVerify(ctx, k.module);
}

test "ops_mlir parity: addptr_ops" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    var k = try Kernel.init(std.testing.allocator, ctx, "addptr_ops", &.{
        .{ .name = "scalar_ptr", .kind = .{ .ptr = .{ .dtype = .f32, .divisibility = null } } },
        .{ .name = "scalar_i32", .kind = .{ .scalar = .i32 } },
    }, &.{});
    defer k.deinit();

    const sp = k.arg(0);
    const si = k.arg(1);

    // scalar
    _ = k.addptr(sp, si);
    // 0D tensor
    const tp0 = k.splat(sp, &.{});
    const ti0 = k.splat(si, &.{});
    _ = k.addptr(tp0, ti0);
    // 1D tensor
    const tp1 = k.splat(sp, &.{16});
    const ti1 = k.splat(si, &.{16});
    _ = k.addptr(tp1, ti1);

    const __ir = try k.finish(&.{}); std.testing.allocator.free(__ir);
    try parityPrintAndVerify(ctx, k.module);
}

test "ops_mlir parity: load_store_ops_scalar" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    var k = try Kernel.init(std.testing.allocator, ctx, "load_store_ops_scalar", &.{
        .{ .name = "ptr", .kind = .{ .ptr = .{ .dtype = .f32, .divisibility = 16 } } },
        .{ .name = "mask", .kind = .{ .scalar = .i1 } },
    }, &.{});
    defer k.deinit();

    const ptr = k.arg(0);
    const mask = k.arg(1);

    const other = k.lift(@as(f32, 0.0));

    const a = k.load(ptr);
    const b = k.loadOpts(ptr, .{ .mask = mask });
    const c = k.loadOpts(ptr, .{ .mask = mask, .other = other });

    k.store(ptr, a);
    k.storeOpts(ptr, b, .{ .mask = mask });
    k.storeOpts(ptr, c, .{ .mask = mask });

    const __ir = try k.finish(&.{}); std.testing.allocator.free(__ir);
    try parityPrintAndVerify(ctx, k.module);
}

test "ops_mlir parity: reduce_ops_infer" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    var k = try Kernel.init(std.testing.allocator, ctx, "reduce_ops_infer", &.{
        .{ .name = "ptr", .kind = .{ .ptr = .{ .dtype = .f32, .divisibility = null } } },
        .{ .name = "v", .kind = .{ .tensor = .{ .shape = &.{ 1, 2, 4 }, .dtype = .f32 } } },
    }, &.{});
    defer k.deinit();

    const v = k.arg(1);
    const f32_ty = mlir.floatType(ctx, .f32);
    const loc: *const mlir.Location = .unknown(ctx);

    // Reduce axis=0: tensor<1x2x4xf32> -> tensor<2x4xf32>
    {
        const combine = mlir.Block.init(&.{ f32_ty, f32_ty }, &.{ loc, loc });
        const sum = arith.addf(ctx, combine.argument(0), combine.argument(1), loc);
        _ = sum.appendTo(combine);
        _ = ttir.reduce_return(ctx, &.{sum.result(0)}, loc).appendTo(combine);
        const res_ty = mlir.rankedTensorType(&.{ 2, 4 }, f32_ty);
        _ = ttir.reduce(ctx, &.{v.inner}, 0, combine, &.{res_ty}, loc).appendTo(k.currentBlock());
    }
    // Reduce axis=2: tensor<1x2x4xf32> -> tensor<1x2xf32>
    {
        const combine = mlir.Block.init(&.{ f32_ty, f32_ty }, &.{ loc, loc });
        const sum = arith.addf(ctx, combine.argument(0), combine.argument(1), loc);
        _ = sum.appendTo(combine);
        _ = ttir.reduce_return(ctx, &.{sum.result(0)}, loc).appendTo(combine);
        const res_ty = mlir.rankedTensorType(&.{ 1, 2 }, f32_ty);
        _ = ttir.reduce(ctx, &.{v.inner}, 2, combine, &.{res_ty}, loc).appendTo(k.currentBlock());
    }

    const __ir = try k.finish(&.{}); std.testing.allocator.free(__ir);
    try parityPrintAndVerify(ctx, k.module);
}

test "ops_mlir parity: dot_ops_infer" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    var k = try Kernel.init(std.testing.allocator, ctx, "dot_ops_infer", &.{
        .{ .name = "ptr", .kind = .{ .ptr = .{ .dtype = .f32, .divisibility = null } } },
        .{ .name = "v", .kind = .{ .scalar = .f32 } },
    }, &.{});
    defer k.deinit();

    const v = k.arg(1);

    const v128x32 = k.splat(v, &.{ 128, 32 });
    const v32x128 = k.splat(v, &.{ 32, 128 });
    const zero128 = k.lift(@as(f32, 0.0));
    const z128x128 = k.splat(zero128, &.{ 128, 128 });
    const z32x32 = k.splat(zero128, &.{ 32, 32 });

    _ = k.dot(v128x32, v32x128, z128x128);
    _ = k.dot(v32x128, v128x32, z32x32);

    const __ir = try k.finish(&.{}); std.testing.allocator.free(__ir);
    try parityPrintAndVerify(ctx, k.module);
}

test "ops_mlir parity: scan_op" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    var k = try Kernel.init(std.testing.allocator, ctx, "scan_op", &.{
        .{ .name = "v", .kind = .{ .tensor = .{ .shape = &.{ 1, 2, 4 }, .dtype = .f32 } } },
    }, &.{});
    defer k.deinit();

    const v = k.arg(0);
    const f32_ty = mlir.floatType(ctx, .f32);
    const loc: *const mlir.Location = .unknown(ctx);

    const combine = mlir.Block.init(&.{ f32_ty, f32_ty }, &.{ loc, loc });
    const sum = arith.addf(ctx, combine.argument(0), combine.argument(1), loc);
    _ = sum.appendTo(combine);
    _ = ttir.scan_return(ctx, &.{sum.result(0)}, loc).appendTo(combine);

    const res_ty = mlir.rankedTensorType(&.{ 1, 2, 4 }, f32_ty);
    _ = ttir.scan(ctx, &.{v.inner}, 1, false, combine, &.{res_ty}, loc).appendTo(k.currentBlock());

    const __ir = try k.finish(&.{}); std.testing.allocator.free(__ir);
    try parityPrintAndVerify(ctx, k.module);
}

test "ops_mlir parity: inline_asm tensor + scalar" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    // tensor form: tensor<512xi8> -> tensor<512xi8>, packed_element=4
    var kt = try Kernel.init(std.testing.allocator, ctx, "inline_asm", &.{
        .{ .name = "x", .kind = .{ .tensor = .{ .shape = &.{512}, .dtype = .i8 } } },
    }, &.{});
    defer kt.deinit();
    const x = kt.arg(0);
    const i8_1d = mlir.rankedTensorType(&.{512}, mlir.integerType(ctx, .i8));
    _ = kt.inlineAsmElementwiseOpts("shl.b32 $0, $0, 3;", "=r,r", &.{x}, &.{i8_1d}, .{ .is_pure = true, .pack = 4 });
    const __ir_kt = try kt.finish(&.{}); std.testing.allocator.free(__ir_kt);
    try parityPrintAndVerify(ctx, kt.module);

    // scalar form: i32 -> i32, packed_element=1
    var ks = try Kernel.init(std.testing.allocator, ctx, "inline_asm_scalar", &.{
        .{ .name = "x", .kind = .{ .scalar = .i32 } },
    }, &.{});
    defer ks.deinit();
    const xs = ks.arg(0);
    const i32_ty = mlir.integerType(ctx, .i32);
    _ = ks.inlineAsmElementwiseOpts("shl.b32 $0, $0, 3;", "=r,r", &.{xs}, &.{i32_ty}, .{ .is_pure = true, .pack = 1 });
    const __ir_ks = try ks.finish(&.{}); std.testing.allocator.free(__ir_ks);
    try parityPrintAndVerify(ctx, ks.module);
}

test "ops_mlir parity: reshape variants" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    var k = try Kernel.init(std.testing.allocator, ctx, "reshape_fn", &.{
        .{ .name = "x", .kind = .{ .tensor = .{ .shape = &.{512}, .dtype = .i32 } } },
    }, &.{});
    defer k.deinit();

    const x = k.arg(0);

    _ = k.reshape(x, &.{ 16, 32 });
    _ = k.reshapeOpts(x, &.{ 16, 32 }, .{ .can_reorder = true });
    _ = k.reshapeOpts(x, &.{ 16, 32 }, .{ .can_reorder = true, .efficient_layout = true });
    _ = k.reshapeOpts(x, &.{ 16, 32 }, .{ .efficient_layout = true });

    const ir = try k.finish(&.{});
    defer std.testing.allocator.free(ir);

    try std.testing.expect(std.mem.indexOf(u8, ir, "allow_reorder") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "efficient_layout") != null);

    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "ops_mlir parity: histogram + masked_histogram" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    {
        var k = try Kernel.init(std.testing.allocator, ctx, "histogram", &.{
            .{ .name = "x", .kind = .{ .tensor = .{ .shape = &.{512}, .dtype = .i32 } } },
        }, &.{});
        defer k.deinit();
        _ = k.histogram(k.arg(0), 16);
        const __ir = try k.finish(&.{}); std.testing.allocator.free(__ir);
        try parityPrintAndVerify(ctx, k.module);
    }
    {
        var k = try Kernel.init(std.testing.allocator, ctx, "masked_histogram", &.{
            .{ .name = "x", .kind = .{ .tensor = .{ .shape = &.{512}, .dtype = .i32 } } },
            .{ .name = "m", .kind = .{ .tensor = .{ .shape = &.{512}, .dtype = .i1 } } },
        }, &.{});
        defer k.deinit();
        _ = k.histogramOpts(k.arg(0), 16, .{ .mask = k.arg(1) });
        const __ir = try k.finish(&.{}); std.testing.allocator.free(__ir);
        try parityPrintAndVerify(ctx, k.module);
    }
}

test "ops_mlir parity: gather" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    var k = try Kernel.init(std.testing.allocator, ctx, "gather_op", &.{
        .{ .name = "src", .kind = .{ .tensor = .{ .shape = &.{ 128, 16 }, .dtype = .f32 } } },
        .{ .name = "idx", .kind = .{ .tensor = .{ .shape = &.{ 512, 16 }, .dtype = .i32 } } },
    }, &.{mlir.rankedTensorType(&.{ 512, 16 }, mlir.floatType(ctx, .f32))});
    defer k.deinit();

    const src = k.arg(0);
    const idx = k.arg(1);
    const g = k.gather(src, idx, 0);

    const __ir = try k.finish(&.{g}); std.testing.allocator.free(__ir);
    try parityPrintAndVerify(ctx, k.module);
}

test "ops_mlir parity: item (unsplat)" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    var k = try Kernel.init(std.testing.allocator, ctx, "item_fn", &.{
        .{ .name = "x", .kind = .{ .tensor = .{ .shape = &.{ 1, 1 }, .dtype = .f32 } } },
    }, &.{mlir.floatType(ctx, .f32)});
    defer k.deinit();

    const x = k.arg(0);
    const scalar = k.item(x);

    const __ir = try k.finish(&.{scalar}); std.testing.allocator.free(__ir);
    try parityPrintAndVerify(ctx, k.module);
}

// The tensordesc-using functions need an arg of type `!tt.tensordesc<...>`
// which Kernel's ArgSpec doesn't model directly. Build them with Layer A.
fn parityBuildModuleWithArgs(
    ctx: *mlir.Context,
    fname: []const u8,
    arg_types: []const *const mlir.Type,
    results: []const *const mlir.Type,
    build_body: *const fn (ctx: *mlir.Context, entry: *mlir.Block) void,
) *mlir.Module {
    const loc: *const mlir.Location = .unknown(ctx);
    const module: *mlir.Module = .init(loc);

    var arg_locs_buf: stdx.BoundedArray(*const mlir.Location, 8) = .{};
    for (0..arg_types.len) |_| arg_locs_buf.appendAssumeCapacity(loc);
    const entry = mlir.Block.init(arg_types, arg_locs_buf.constSlice());

    build_body(ctx, entry);
    _ = ttir.return_(ctx, &.{}, loc).appendTo(entry);

    const f = ttir.func(ctx, .{
        .name = fname,
        .args = arg_types,
        .results = results,
        .block = entry,
        .location = loc,
    });
    _ = f.appendTo(module.body());
    return module;
}

test "ops_mlir parity: descriptor_load" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    const f32_ty = mlir.floatType(ctx, .f32);
    const desc_ty = ttir.tensorDescType(&.{128}, f32_ty, null);

    const Body = struct {
        fn build(c: *mlir.Context, entry: *mlir.Block) void {
            const l: *const mlir.Location = .unknown(c);
            const i32_ = mlir.integerType(c, .i32);
            const f32_ = mlir.floatType(c, .f32);
            const r = mlir.rankedTensorType(&.{128}, f32_);
            const c0 = arith.constant_int(c, 0, i32_, l);
            _ = c0.appendTo(entry);
            _ = ttir.descriptor_load(c, entry.argument(0), &.{c0.result(0)}, r, .none, .normal, l).appendTo(entry);
        }
    };

    const module = parityBuildModuleWithArgs(ctx, "descriptor_load", &.{desc_ty}, &.{}, &Body.build);
    defer module.deinit();
    try std.testing.expect(module.operation().verify());
    try parityPrintAndVerify(ctx, module);
}

test "ops_mlir parity: tma_gather + tma_scatter" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    const bf16_ty = mlir.floatType(ctx, .bf16);
    const i32_ty = mlir.integerType(ctx, .i32);
    const desc_ty = ttir.tensorDescType(&.{ 1, 128 }, bf16_ty, null);
    const off_ty = mlir.rankedTensorType(&.{32}, i32_ty);
    const data_ty = mlir.rankedTensorType(&.{ 32, 128 }, bf16_ty);

    // tma_gather: (!tt.tensordesc<1x128xbf16>, tensor<32xi32>, i32) -> tensor<32x128xbf16>
    {
        const GatherBody = struct {
            fn build(c: *mlir.Context, entry: *mlir.Block) void {
                const l: *const mlir.Location = .unknown(c);
                const bf16_ = mlir.floatType(c, .bf16);
                const out_ty = mlir.rankedTensorType(&.{ 32, 128 }, bf16_);
                _ = ttir.descriptor_gather(c, entry.argument(0), entry.argument(1), entry.argument(2), out_ty, l).appendTo(entry);
            }
        };
        const module = parityBuildModuleWithArgs(ctx, "tma_gather", &.{ desc_ty, off_ty, i32_ty }, &.{}, &GatherBody.build);
        defer module.deinit();
        try parityPrintAndVerify(ctx, module);
    }

    // tma_scatter: (!tt.tensordesc<1x128xbf16>, tensor<32xi32>, i32, tensor<32x128xbf16>) -> ()
    {
        const ScatterBody = struct {
            fn build(c: *mlir.Context, entry: *mlir.Block) void {
                const l: *const mlir.Location = .unknown(c);
                _ = ttir.descriptor_scatter(c, entry.argument(0), entry.argument(1), entry.argument(2), entry.argument(3), l).appendTo(entry);
            }
        };
        const module = parityBuildModuleWithArgs(ctx, "tma_scatter", &.{ desc_ty, off_ty, i32_ty, data_ty }, &.{}, &ScatterBody.build);
        defer module.deinit();
        try parityPrintAndVerify(ctx, module);
    }
}

test "lift handles large comptime_int and unsigned runtime bit patterns" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    var k = try Kernel.init(std.testing.allocator, ctx, "lift_big_ints", &.{}, &.{});
    defer k.deinit();

    // comptime_int > i64 max, within u64 range → i64 with all-ones bit pattern.
    const c1 = k.lift(0xFFFF_FFFF_FFFF_FFFF);
    try std.testing.expectEqual(DType.i64, k.mlirElemToDType(c1.elemType()));

    // runtime u64 at u64 max → i64 with all-ones bit pattern (no panic).
    const v_u64: u64 = 0xFFFF_FFFF_FFFF_FFFF;
    const c2 = k.lift(v_u64);
    try std.testing.expectEqual(DType.i64, k.mlirElemToDType(c2.elemType()));

    // runtime u32 at u32 max → i32 with all-ones bit pattern (no panic).
    const v_u32: u32 = 0xFFFF_FFFF;
    const c3 = k.lift(v_u32);
    try std.testing.expectEqual(DType.i32, k.mlirElemToDType(c3.elemType()));

    const ir = try k.finish(&.{});
    defer std.testing.allocator.free(ir);
}

test "lift preserves f16 width through fluent ops" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    var k = try Kernel.init(std.testing.allocator, ctx, "lift_f16", &.{
        .{ .name = "a", .kind = .{ .scalar = .f16 } },
    }, &.{});
    defer k.deinit();

    // Direct lift of f16 stays f16.
    const f = k.lift(@as(f16, 1.5));
    try std.testing.expectEqual(DType.f16, k.mlirElemToDType(f.elemType()));

    // Fluent op on an f16 Value with an f16 rhs must verify cleanly
    // (pre-fix this emitted an f32 constant and failed arith.addf verification).
    const a = k.arg(0);
    _ = a.add(@as(f16, 2.0));

    const ir = try k.finish(&.{});
    defer std.testing.allocator.free(ir);
}

test "forLoop with Value bounds and no ctx" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    var k = try Kernel.init(std.testing.allocator, ctx, "loop_mixed", &.{
        .{ .name = "lb", .kind = .{ .scalar = .i32 } },
    }, &.{});
    defer k.deinit();

    const lb = k.arg(0);
    const ub = k.liftAs(@as(usize, 10), .i32);

    const Body = struct {
        fn b(_: *Kernel, _: Value, _: []const Value, _: void) []const Value {
            return &.{};
        }
    }.b;
    _ = k.forLoop({}, lb, ub, 1, .{ .body = Body });

    const ir = try k.finish(&.{});
    defer std.testing.allocator.free(ir);
}
