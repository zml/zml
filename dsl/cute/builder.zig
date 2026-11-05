const std = @import("std");

const arith = @import("mlir/dialects/arith");
const cf = @import("mlir/dialects/cf");
const common = @import("dsl/common/builder");
const cuda = @import("mlir/dialects/cuda");
const cute_ir = @import("mlir/dialects/cute_ir");
const cute_nvgpu = @import("mlir/dialects/cute_nvgpu");
const func = @import("mlir/dialects/func");
const gpu = @import("mlir/dialects/gpu");
const llvm = @import("mlir/dialects/llvm");
const math = @import("mlir/dialects/math");
const nvvm = @import("mlir/dialects/nvvm");
const scf = @import("mlir/dialects/scf");
const vector = @import("mlir/dialects/vector");
const mlir = @import("mlir");
const stdx = @import("stdx");

const dtypes = @import("dtype.zig");
pub const DType = dtypes.DType;
const isFloatDtype = dtypes.isFloatDtype;
const dtypeBitwidth = dtypes.dtypeBitwidth;

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Builder);
    std.testing.refAllDecls(Value);
    std.testing.refAllDecls(Tensor);
    std.testing.refAllDecls(View);
    std.testing.refAllDecls(Atom);
}

/// `nvvm` and `cuda` are NVIDIA's (`nvvm.ops`, `cuda.ops`, handles `cute_nvvm`
/// and `cuda`); `llvm` is upstream's, whose types and operations NVIDIA's
/// compiler reads as its own.
pub fn registerDialects(registry: *mlir.DialectRegistry) void {
    inline for (.{ func, gpu, cute_ir, cute_nvgpu, nvvm, cuda, llvm, arith, scf, math, cf, vector }) |dialect| dialect.registerDialect(registry);
}

pub const FinishError = error{InvalidMlir} || std.mem.Allocator.Error || std.Io.Writer.Error;

pub const MAX_RANK = 8;
pub const Dims = stdx.BoundedArray(i64, MAX_RANK);

pub const MemorySpace = cute_ir.MemorySpace;

/// A CuTe algebra expression assembled from Zig values. Kernel authors use
/// `Builder.expr`, `Builder.basis`, and `Builder.layoutSpec`; the DSL builds
/// the algebra attribute of the kind each use needs.
pub const AlgebraExpr = union(enum) {
    static: i64,
    /// A dynamic leaf known to be a multiple of this, 1 when nothing is known.
    dynamic: i64,
    /// A dynamic integer with an explicit bit width and divisibility.
    dynamic_int: struct { width: u8, divisible_by: i64 = 1 },
    all,
    basis: ScaledBasis,
    tuple: []const AlgebraExpr,
};

pub const AlgebraToken = enum {
    /// CuTe's slice placeholder (`_`).
    all,
    /// A dynamic algebra leaf (`?`).
    dynamic,
};

pub const ScaledBasis = struct {
    numerator: i64,
    denominator: i64 = 1,
    modes: [MAX_RANK]u8 = @splat(0),
    mode_count: u8,
};

/// A dynamic CuTe integer whose runtime value is known to be divisible by a
/// static amount. CuTe carries this fact in dependent coordinate-tensor types
/// after tiling and slicing, for example `?{div=128}`.
pub const ConstrainedDynamic = struct {
    divisible_by: u64,
};

pub const LayoutSpec = struct {
    shape: AlgebraExpr,
    stride: AlgebraExpr,
};

/// A scalar SSA value (`Int32`, `Float32`, ... in the Python DSL).
pub const Value = struct {
    inner: *const mlir.Value,
    kernel: ?*Builder = null,

    pub fn type_(self: Value) *const mlir.Type {
        return self.inner.type_();
    }

    fn kern(self: Value) *Builder {
        return self.kernel orelse @panic("Value has no owning kernel; use Builder.* helpers instead");
    }

    pub fn dtype(self: Value) DType {
        return dtypes.mlirToDType(self.kern().ctx, self.type_());
    }

    pub fn isFloat(self: Value) bool {
        const t = self.type_();
        inline for (@typeInfo(mlir.FloatTypes).@"enum".field_names) |f| {
            if (t.isA(mlir.FloatType(@field(mlir.FloatTypes, f))) != null) return true;
        }
        return false;
    }

    pub fn isInt(self: Value) bool {
        return self.type_().isA(mlir.IntegerType) != null;
    }

    pub fn isPtr(self: Value) bool {
        return self.type_().isA(cute_ir.PtrType) != null;
    }

    pub fn add(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(if (l.isFloat()) arith.addf(k.ctx, l.inner, r.inner, k.loc()) else arith.addi(k.ctx, l.inner, r.inner, k.loc()));
    }

    pub fn sub(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(if (l.isFloat()) arith.subf(k.ctx, l.inner, r.inner, k.loc()) else arith.subi(k.ctx, l.inner, r.inner, k.loc()));
    }

    pub fn mul(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(if (l.isFloat()) arith.mulf(k.ctx, l.inner, r.inner, k.loc()) else arith.muli(k.ctx, l.inner, r.inner, k.loc()));
    }

    /// Signed division on integers (Python `//` on Int32).
    pub fn div(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(if (l.isFloat()) arith.divf(k.ctx, l.inner, r.inner, k.loc()) else arith.divsi(k.ctx, l.inner, r.inner, k.loc()));
    }

    pub fn rem(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(if (l.isFloat()) arith.remf(k.ctx, l.inner, r.inner, k.loc()) else arith.remsi(k.ctx, l.inner, r.inner, k.loc()));
    }

    pub fn cdiv(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        if (l.isFloat()) @panic("Value.cdiv is integer-only");
        return k.emit(arith.ceildivsi(k.ctx, l.inner, r.inner, k.loc()));
    }

    pub fn neg(self: Value) Value {
        const k = self.kern();
        if (self.isFloat()) return k.emit(arith.negf(k.ctx, self.inner, k.loc()));
        return k.constant(self.dtype(), 0).sub(self);
    }

    pub fn abs(self: Value) Value {
        const k = self.kern();
        return k.emit(if (self.isFloat()) math.absf(k.ctx, self.inner, k.loc()) else math.absi(k.ctx, self.inner, k.loc()));
    }

    pub fn minimum(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(if (l.isFloat()) arith.minimumf(k.ctx, l.inner, r.inner, k.loc()) else arith.minsi(k.ctx, l.inner, r.inner, k.loc()));
    }

    pub fn maximum(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(if (l.isFloat()) arith.maximumf(k.ctx, l.inner, r.inner, k.loc()) else arith.maxsi(k.ctx, l.inner, r.inner, k.loc()));
    }

    /// Float max that ignores NaN operands (`arith.maxnumf`, a single `max.f32`), unlike
    /// the NaN-propagating `maximum`.
    pub fn maxnum(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(arith.maxnumf(k.ctx, l.inner, r.inner, k.loc()));
    }

    pub fn bitAnd(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(arith.andi(k.ctx, l.inner, r.inner, k.loc()));
    }

    pub fn bitOr(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(arith.ori(k.ctx, l.inner, r.inner, k.loc()));
    }

    pub fn bitXor(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(arith.xori(k.ctx, l.inner, r.inner, k.loc()));
    }

    pub fn shl(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(arith.shli(k.ctx, l.inner, r.inner, k.loc()));
    }

    pub fn shr(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(arith.shrsi(k.ctx, l.inner, r.inner, k.loc()));
    }

    pub fn shrLogical(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(arith.shrui(k.ctx, l.inner, r.inner, k.loc()));
    }

    pub fn bitCast(self: Value, dt: DType) Value {
        const k = self.kern();
        if (dtypeBitwidth(self.dtype()) != dtypeBitwidth(dt)) @panic("Value.bitCast requires equal bit widths");
        return k.emit(arith.bitcast(k.ctx, self.inner, dt.toMlir(k.ctx), k.loc()));
    }

    pub fn lt(self: Value, rhs: anytype) Value {
        return self.kern().cmp(self, rhs, .slt, .olt);
    }

    pub fn le(self: Value, rhs: anytype) Value {
        return self.kern().cmp(self, rhs, .sle, .ole);
    }

    pub fn gt(self: Value, rhs: anytype) Value {
        return self.kern().cmp(self, rhs, .sgt, .ogt);
    }

    pub fn ge(self: Value, rhs: anytype) Value {
        return self.kern().cmp(self, rhs, .sge, .oge);
    }

    pub fn eq(self: Value, rhs: anytype) Value {
        return self.kern().cmp(self, rhs, .eq, .oeq);
    }

    pub fn ne(self: Value, rhs: anytype) Value {
        return self.kern().cmp(self, rhs, .ne, .une);
    }

    pub fn exp(self: Value) Value {
        const k = self.kern();
        return k.emit(math.exp(k.ctx, self.inner, k.loc()));
    }

    pub fn log(self: Value) Value {
        const k = self.kern();
        return k.emit(math.log(k.ctx, self.inner, k.loc()));
    }

    pub fn log2(self: Value) Value {
        const k = self.kern();
        return k.emit(math.log2(k.ctx, self.inner, k.loc()));
    }

    pub fn exp2(self: Value) Value {
        const k = self.kern();
        return k.emit(math.exp2(k.ctx, self.inner, k.loc()));
    }

    pub fn ceil(self: Value) Value {
        const k = self.kern();
        if (!self.isFloat()) @panic("Value.ceil is float-only");
        return k.emit(math.ceil(k.ctx, self.inner, k.loc()));
    }

    pub fn sqrt(self: Value) Value {
        const k = self.kern();
        return k.emit(math.sqrt(k.ctx, self.inner, k.loc()));
    }

    pub fn rsqrt(self: Value) Value {
        const k = self.kern();
        return k.emit(math.rsqrt(k.ctx, self.inner, k.loc()));
    }

    pub fn tanh(self: Value) Value {
        const k = self.kern();
        return k.emit(math.tanh(k.ctx, self.inner, k.loc()));
    }

    /// Python `x.to(T)`: int <-> int is signed, int <-> float goes through
    /// `sitofp`/`fptosi`, same-width floats through `convertf`.
    pub fn to(self: Value, dt: DType) Value {
        const k = self.kern();
        const src = self.dtype();
        if (src == dt) return self;
        const ty = dt.toMlir(k.ctx);
        const src_f = isFloatDtype(src);
        const dst_f = isFloatDtype(dt);
        const sw = dtypeBitwidth(src);
        const dw = dtypeBitwidth(dt);
        if (src_f and dst_f) {
            if (sw == dw) return k.emit(arith.convertf(k.ctx, self.inner, ty, .{}, k.loc()));
            return k.emit(if (dw > sw) arith.extf(k.ctx, self.inner, ty, k.loc()) else arith.truncf(k.ctx, self.inner, ty, k.loc()));
        }
        if (src_f) return k.emit(arith.fptosi(k.ctx, self.inner, ty, k.loc()));
        if (dst_f) return k.emit(arith.sitofp(k.ctx, self.inner, ty, k.loc()));
        if (sw == dw) return self;
        if (src == .i1) return k.emit(arith.extui(k.ctx, self.inner, ty, k.loc()));
        return k.emit(if (dw > sw) arith.extsi(k.ctx, self.inner, ty, k.loc()) else arith.trunci(k.ctx, self.inner, ty, k.loc()));
    }
};

/// `cute.Layout`: a static shape and stride, carried in Zig so coordinates
/// and element counts need no MLIR queries.
pub const Layout = struct {
    inner: *const mlir.Value,
    shape: Dims,
    stride: Dims,

    pub fn rank(self: Layout) usize {
        return self.shape.len;
    }

    pub fn size(self: Layout) i64 {
        var n: i64 = 1;
        for (self.shape.constSlice()) |d| n *= d;
        return n;
    }

    pub fn dim(self: Layout, i: usize) i64 {
        return self.shape.get(i);
    }
};

/// `cute.Tensor`: an engine (pointer) composed with a layout. `get`/`set`
/// are the Python `t[coord]` / `t[coord] = v`.
pub const Tensor = struct {
    inner: *const mlir.Value,
    kernel: *Builder,
    layout: Layout,
    dtype: DType,
    space: MemorySpace,

    pub fn shape(self: Tensor) []const i64 {
        return self.layout.shape.constSlice();
    }

    pub fn rank(self: Tensor) usize {
        return self.layout.rank();
    }

    pub fn size(self: Tensor) i64 {
        return self.layout.size();
    }

    pub fn dim(self: Tensor, i: usize) i64 {
        return self.layout.dim(i);
    }

    /// `t[coord]`: `coord` is a tuple of `Value`s and ints, one per mode.
    pub fn get(self: Tensor, coord: anytype) Value {
        const k = self.kernel;
        const c = k.makeCoord(coord);
        return k.emit(cute_ir.memref_load(k.ctx, self.inner, c, self.dtype.toMlir(k.ctx), k.loc()));
    }

    /// `t[coord] = value`; the value is converted to the element type.
    pub fn set(self: Tensor, coord: anytype, value_: anytype) void {
        const k = self.kernel;
        const c = k.makeCoord(coord);
        const v = k.constant(self.dtype, value_).to(self.dtype);
        _ = cute_ir.memref_store(k.ctx, self.inner, c, v.inner, k.loc()).appendTo(k.currentBlock());
    }

    /// Erase Zig's tracked layout while retaining the MLIR memref. CuTe
    /// layout algebra derives the layouts returned by tiling/partitioning;
    /// those values are represented by `View` because Zig cannot recompute
    /// their dependent types.
    pub fn view(self: Tensor) View {
        return .{ .inner = self.inner, .kernel = self.kernel };
    }

    pub fn value(self: Tensor) Value {
        return .{ .inner = self.inner, .kernel = self.kernel };
    }
};

/// A CuTe tensor or pointer with a layout derived by the CuTe compiler.
pub const View = struct {
    inner: *const mlir.Value,
    kernel: *Builder,

    pub fn type_(self: View) *const mlir.Type {
        return self.inner.type_();
    }

    pub fn value(self: View) Value {
        return .{ .inner = self.inner, .kernel = self.kernel };
    }

    pub fn get(self: View, coord: anytype, result_dtype: DType) Value {
        const c = self.kernel.makeCoord(coord);
        return self.kernel.emit(cute_ir.memref_load(self.kernel.ctx, self.inner, c, result_dtype.toMlir(self.kernel.ctx), self.kernel.loc()));
    }

    pub fn set(self: View, coord: anytype, value_: Value) void {
        self.kernel.emitVoid(cute_ir.memref_store(self.kernel.ctx, self.inner, self.kernel.makeCoord(coord), value_.inner, self.kernel.loc()));
    }
};

/// A copy/MMA atom or a tiled atom. Atom fields are immutable SSA values;
/// `setField` returns the atom carrying the updated runtime field.
pub const Atom = struct {
    inner: *const mlir.Value,
    kernel: *Builder,

    pub fn type_(self: Atom) *const mlir.Type {
        return self.inner.type_();
    }

    pub fn value(self: Atom) Value {
        return .{ .inner = self.inner, .kernel = self.kernel };
    }

    pub fn setField(self: Atom, field: []const u8, value_: anytype) Atom {
        const T = @TypeOf(value_);
        const new_value = if (T == Value or T == Tensor or T == View or T == Atom or T == Layout)
            self.kernel.asValue(value_)
        else
            self.kernel.lift(value_);
        const result = self.kernel.emit(cute_nvgpu.atom_set_value(
            self.kernel.ctx,
            self.inner,
            new_value.inner,
            self.type_(),
            .string(self.kernel.ctx, field),
            self.kernel.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self.kernel };
    }
};

pub const TmaDescriptor = struct {
    atom: Atom,
    tensor: View,
};

pub const TmaPartition = struct {
    smem: View,
    targets: []const View,
};

/// Element encoding of a TMA transfer when it differs from the tensor dtype.
pub const TmaFormat = enum {
    default,
    /// Packed 4-bit elements, two per byte in shared memory (NVFP4 operands
    /// of the `mxf4nvf4` MMA).
    u4,
    /// 4-bit elements expanded to one per byte in shared memory (FP4 operands
    /// of the `mxf8f6f4` MMA).
    u4_unpack_u8,
    u16,
};

/// CuTe's 64-bit TMA `.L2::cache_hint` policies for SM90.
pub const CacheHintSm90 = enum(u64) {
    evict_normal = 0x1000000000000000,
    evict_first = 0x12F0000000000000,
    evict_last = 0x14F0000000000000,
};

/// CuTe's 64-bit TMA `.L2::cache_hint` policies for SM100.
pub const CacheHintSm100 = enum(u64) {
    evict_normal = 0x1000000000000000,
    evict_first = 0x12F0000000000000,
    evict_last = 0x14F0000000000000,
};

pub const CtaGroup = enum(u8) {
    one = 1,
    two = 2,

    fn toNvvm(self: CtaGroup) nvvm.CtaGroup {
        return switch (self) {
            .one => .cta_1,
            .two => .cta_2,
        };
    }
};
pub const MBarrierScope = nvvm.MBarrierScope;
pub const OperandMajorMode = enum(u1) { k = 0, mn = 1 };

/// Number of CTAs cooperating on one instruction, as in CuTe's CtaGroup.value.
pub fn ctaCount(group: CtaGroup) u8 {
    return @backingInt(group);
}

pub const Tcgen05SmemSwizzle = enum(u3) {
    none = 0,
    swizzle_128b_atom_32b = 1,
    swizzle_128b = 2,
    swizzle_64b = 4,
    swizzle_32b = 6,
};

/// CuTe's packed SM100 shared-memory descriptor. All offsets are in bytes.
pub const Tcgen05SmemDesc = struct {
    pub const Options = struct {
        leading_byte_offset: u32 = 0,
        stride_byte_offset: u32 = 0,
        version: u2 = 1,
        base_offset: u3 = 0,
        layout: Tcgen05SmemSwizzle = .none,
    };

    pub fn build(b: *Builder, start_address: Value, opts: Options) Value {
        std.debug.assert(opts.leading_byte_offset % 16 == 0 and opts.leading_byte_offset < 0x40000);
        std.debug.assert(opts.stride_byte_offset % 16 == 0 and opts.stride_byte_offset < 0x40000);
        const fields = (@as(u64, opts.leading_byte_offset >> 4) << 16) |
            (@as(u64, opts.stride_byte_offset >> 4) << 32) |
            (@as(u64, opts.version) << 46) |
            (@as(u64, opts.base_offset) << 49) |
            (@as(u64, @backingInt(opts.layout)) << 61);
        const address = if (start_address.isInt()) start_address else b.ptrToInt(start_address, .i32);
        return address.to(.i64).shrLogical(4).bitAnd(0x3fff).bitOr(b.constant(.i64, @as(i64, @bitCast(fields))));
    }
};

/// Dense SM100 instruction descriptor; the MMA kind disambiguates dtype encodings.
pub const Tcgen05InstrDesc = struct {
    pub const Options = struct {
        a_dtype: DType,
        b_dtype: DType,
        c_dtype: DType = .f32,
        m_dim: u16,
        n_dim: u16,
        a_major: OperandMajorMode = .k,
        b_major: OperandMajorMode = .k,
        a_negate: bool = false,
        b_negate: bool = false,
    };

    pub fn build(comptime opts: Options) u32 {
        comptime std.debug.assert(opts.m_dim > 0 and opts.m_dim <= 256 and opts.m_dim % 16 == 0);
        comptime std.debug.assert(opts.n_dim > 0 and opts.n_dim <= 256 and opts.n_dim % 8 == 0);
        const half_inputs = (opts.a_dtype == .f16 or opts.a_dtype == .bf16) and (opts.b_dtype == .f16 or opts.b_dtype == .bf16);
        const fp8_inputs = (opts.a_dtype == .f8e4m3fn or opts.a_dtype == .f8e5m2) and (opts.b_dtype == .f8e4m3fn or opts.b_dtype == .f8e5m2);
        const valid = (half_inputs and opts.c_dtype == .f32) or
            (opts.a_dtype == .f16 and opts.b_dtype == .f16 and opts.c_dtype == .f16) or
            (opts.a_dtype == .f32 and opts.b_dtype == .f32 and opts.c_dtype == .f32) or
            (fp8_inputs and (opts.c_dtype == .f16 or opts.c_dtype == .f32)) or
            (opts.a_dtype == .i8 and opts.b_dtype == .i8 and opts.c_dtype == .i32);
        if (!valid) @compileError("unsupported dense tcgen05 dtype combination");
        const c: u32 = switch (opts.c_dtype) {
            .f16 => 0,
            .f32 => 1,
            .i32 => 2,
            else => @compileError("unsupported tcgen05 accumulator dtype"),
        };
        return (c << 4) | (inputFormat(opts.a_dtype) << 7) | (inputFormat(opts.b_dtype) << 10) |
            (@as(u32, @intFromBool(opts.a_negate)) << 13) |
            (@as(u32, @intFromBool(opts.b_negate)) << 14) |
            (@as(u32, @backingInt(opts.a_major)) << 15) |
            (@as(u32, @backingInt(opts.b_major)) << 16) |
            (@as(u32, opts.n_dim >> 3) << 17) | (@as(u32, opts.m_dim >> 4) << 24);
    }

    fn inputFormat(comptime dtype: DType) u32 {
        return switch (dtype) {
            .f16, .f8e4m3fn => 0,
            .bf16, .f8e5m2, .i8 => 1,
            .f32 => 2,
            .f4e2m1fn => 5,
            else => @compileError("unsupported tcgen05 input dtype"),
        };
    }
};

/// Typed description of a non-multicast tiled TMA atom: the element type,
/// bits moved per copy, basis of one box, and coordinate layout of the tensor.
pub const TmaConfig = struct {
    dtype: DType,
    copy_bits: u32,
    global_basis: LayoutSpec,
    coordinate_layout: LayoutSpec,
    coordinate_rank: u8,
    format: TmaFormat = .default,
    /// Load completion group: with `.two`, each CTA's load completes on
    /// the mbarrier of the pair's even CTA. Ignored by TMA stores.
    cta_group: CtaGroup = .one,
    mode: enum { tiled, gather4 } = .tiled,
};

pub const CopyBulkTensorTileG2SOp = struct { cta_group: CtaGroup = .one };
pub const CopyBulkTensor2DGather4G2SOp = struct { cta_group: CtaGroup = .one };
pub const CopyBulkTensorTileS2GOp = struct {};

pub const BlockScaledMma = struct {
    atom: Atom,
    tiled: Atom,
};

/// SS, K-major SM100 block-scaled MMA instruction with FP4 weights as A.
/// The instruction `kind` fixes B, the scales and K.
pub const BlockScaledMmaConfig = struct {
    kind: Kind = .mxfp4_mxfp8,
    /// M is 128 for one CTA, or 128 or 256 for a CTA pair.
    m: u16 = 128,
    n: u16,
    cta_group: CtaGroup = .one,

    pub const Kind = enum {
        /// MXFP4 weights x MXFP8 activations: E8M0 scales every 32
        /// elements, K = 32 (PTX `kind::mxf8f6f4`).
        mxfp4_mxfp8,
        /// NVFP4 x NVFP4: E4M3 scales every 16 elements, K = 64 (PTX
        /// `kind::mxf4nvf4.block16`).
        nvfp4_nvfp4,
    };

    pub fn k(self: BlockScaledMmaConfig) u16 {
        return switch (self.kind) {
            .mxfp4_mxfp8 => 32,
            .nvfp4_nvfp4 => 64,
        };
    }

    fn bType(self: BlockScaledMmaConfig) DType {
        return switch (self.kind) {
            .mxfp4_mxfp8 => .f8e4m3fn,
            .nvfp4_nvfp4 => .f4e2m1fn,
        };
    }

    fn scaleType(self: BlockScaledMmaConfig) DType {
        return switch (self.kind) {
            .mxfp4_mxfp8 => .f8e8m0fnu,
            .nvfp4_nvfp4 => .f8e4m3fn,
        };
    }

    fn vecSize(self: BlockScaledMmaConfig) u8 {
        return switch (self.kind) {
            .mxfp4_mxfp8 => 32,
            .nvfp4_nvfp4 => 16,
        };
    }

    fn validate(self: BlockScaledMmaConfig) void {
        const m_ok = self.m == 128 or (self.cta_group == .two and self.m == 256);
        const n_multiple: u16 = if (self.cta_group == .two) 16 else 8;
        if (!m_ok or self.n < n_multiple or self.n > 256 or self.n % n_multiple != 0) {
            std.debug.panic("unsupported SM100 block-scaled MMA shape {d}x{d}x{d} on {d} CTAs", .{ self.m, self.n, self.k(), ctaCount(self.cta_group) });
        }
    }
};

fn tmaLoadKind(config: TmaConfig) cute_nvgpu.TiledTmaLoad {
    return switch (config.cta_group) {
        .one => .sm_90,
        .two => .sm_100_2sm,
    };
}

fn tmaDataFormat(format: TmaFormat, dtype: DType) cute_nvgpu.TmaDataFormat {
    return switch (format) {
        .u4 => .U4,
        .u4_unpack_u8 => .U4_UNPACK_U8,
        .u16 => .U16,
        .default => switch (dtype) {
            .f8e4m3fn => .U8,
            .f32 => .F32_RN,
            else => @panic("TMA format must be specified for this dtype"),
        },
    };
}

fn tmaFormatAttribute(b: *Builder, config: TmaConfig) ?*const mlir.Attribute {
    if (config.format == .default) return null;
    const attr = cute_nvgpu.TmaDataFormatAttr.get(b.ctx, .{ .value = tmaDataFormat(config.format, config.dtype) }) catch @panic("invalid TMA format");
    return attr.attribute();
}

pub const Index3 = struct { x: Value, y: Value, z: Value };

pub const LaunchConfig = struct {
    inner: *const mlir.Value,
    kernel: *Builder,

    pub fn value(self: LaunchConfig) Value {
        return .{ .inner = self.inner, .kernel = self.kernel };
    }
};

pub const ArgSpec = struct {
    name: []const u8,
    kind: Kind,

    /// XLA passes one raw device pointer per custom-call argument. A `tensor`
    /// argument is that pointer composed with a static layout at the top of
    /// the kernel, the way the Python DSL passes `cute.Tensor` arguments.
    pub const Kind = union(enum) {
        ptr: DType,
        tensor: TensorSpec,
        /// A type constructed through a dialect/type API.
        mlir_type: MlirTypeSpec,
    };

    pub const MlirTypeSpec = struct {
        type_: *const mlir.Type,
        grid_constant: bool = false,
        cuda_grid_constant: bool = false,
    };

    pub const TensorSpec = struct {
        dtype: DType,
        shape: []const i64,
        /// Defaults to XLA's row-major layout.
        stride: ?[]const i64 = null,
        alignment: u64 = 16,
    };
};

pub const LayoutOpts = struct {
    /// Defaults to CuTe's compact left-most stride, like `cute.make_layout`.
    stride: ?[]const i64 = null,
};

pub const IfOnlyScope = common.IfOnlyScope(Builder, Value);

pub fn IfScope(comptime N: usize) type {
    return common.IfScope(Builder, Value, N);
}

pub fn ForScope(comptime N: usize) type {
    return common.ForScope(Builder, Value, N);
}

pub fn WhileScope(comptime N: usize, comptime M: usize) type {
    return common.WhileScope(Builder, Value, N, M);
}

pub const Builder = struct {
    pub const FunctionKind = enum { func_only_kernel, cuda_kernel, host };

    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    ctx: *mlir.Context,
    module: *mlir.Module,
    gpu_body: ?*mlir.Block = null,
    name: []const u8,
    function_kind: FunctionKind = .func_only_kernel,
    func_op: ?*mlir.Operation = null,
    entry_block: ?*mlir.Block = null,
    block_stack: common.BlockStack = .{},
    args: []const ArgSpec = &.{},
    tensors: []?Tensor = &.{},

    pub fn open(allocator: std.mem.Allocator, ctx: *mlir.Context, name: []const u8) !Builder {
        const module: *mlir.Module = .init(.unknown(ctx));
        errdefer module.deinit();
        return .{
            .allocator = allocator,
            .arena = .init(allocator),
            .ctx = ctx,
            .module = module,
            .name = name,
        };
    }

    /// Open a complete CuTe program. Unlike `open`, which emits the compact
    /// func.func-only ABI consumed directly by XLA, a program owns a
    /// `gpu.module` for one or more `cuda.kernel` operations and can append a
    /// host `func.func` that constructs CuTe launch objects and calls
    /// `cuda.launch_ex`.
    pub fn openProgram(allocator: std.mem.Allocator, ctx: *mlir.Context, name: []const u8) !Builder {
        var self = try open(allocator, ctx, name);
        self.module.operation().setAttribute(gpu.containerModule(ctx));
        const body = mlir.Block.init(&.{}, &.{});
        const gpu_module = gpu.module(ctx, "kernels", body, self.loc());
        _ = gpu_module.appendTo(self.module.body());
        self.gpu_body = body;
        self.function_kind = .cuda_kernel;
        return self;
    }

    /// Start another function in a complete program after `endFunction`.
    pub fn beginFunction(self: *Builder, name: []const u8, kind: FunctionKind) void {
        std.debug.assert(self.entry_block == null and self.func_op == null);
        self.name = name;
        self.function_kind = kind;
    }

    pub fn deinit(self: *Builder) void {
        self.module.deinit();
        self.arena.deinit();
    }

    /// One field per kernel argument: `.{ .a = .{ .tensor = .{ .dtype = .f32, .shape = &.{n} } } }`
    /// yields a `Tensor`, `.{ .p = .{ .ptr = .f32 } }` a pointer `Value`.
    pub fn declareArgs(self: *Builder, spec: anytype) FinishError!ArgsOf(@TypeOf(spec)) {
        const Spec = @TypeOf(spec);
        const fields = @typeInfo(Spec).@"struct".field_names;

        const arg_specs = try self.arena.allocator().alloc(ArgSpec, fields.len);
        inline for (fields, 0..) |f, i| {
            const raw = @field(spec, f);
            const variant = @typeInfo(@TypeOf(raw)).@"struct".field_names[0];
            const inner = @field(raw, variant);
            arg_specs[i] = .{ .name = f, .kind = switch (@field(std.meta.Tag(ArgSpec.Kind), variant)) {
                .ptr => .{ .ptr = inner },
                .tensor => .{ .tensor = .{
                    .dtype = inner.dtype,
                    .shape = inner.shape,
                    .stride = if (@hasField(@TypeOf(inner), "stride")) inner.stride else null,
                    .alignment = if (@hasField(@TypeOf(inner), "alignment")) inner.alignment else 16,
                } },
                .mlir_type => .{ .mlir_type = .{
                    .type_ = inner.type_,
                    .grid_constant = if (@hasField(@TypeOf(inner), "grid_constant")) inner.grid_constant else false,
                    .cuda_grid_constant = if (@hasField(@TypeOf(inner), "cuda_grid_constant")) inner.cuda_grid_constant else false,
                } },
            } };
        }
        try self.declareArgsLow(arg_specs);

        var named: ArgsOf(Spec) = undefined;
        inline for (fields, 0..) |f, i| {
            @field(named, f) = switch (@FieldType(ArgsOf(Spec), f)) {
                Tensor => self.tensors[i].?,
                else => self.arg(i),
            };
        }
        return named;
    }

    fn ArgsOf(comptime Spec: type) type {
        const in = @typeInfo(Spec).@"struct".field_names;
        comptime var names: [in.len][]const u8 = undefined;
        comptime var types: [in.len]type = undefined;
        inline for (in, 0..) |f, i| {
            names[i] = f;
            types[i] = if (@hasField(@FieldType(Spec, f), "tensor")) Tensor else Value;
        }
        return @Struct(.auto, null, &names, &types, &@splat(.{}));
    }

    pub fn declareArgsLow(self: *Builder, args: []const ArgSpec) FinishError!void {
        std.debug.assert(self.entry_block == null);
        const ctx = self.ctx;
        const scratch = self.arena.allocator();

        const arg_types = try scratch.alloc(*const mlir.Type, args.len);
        const arg_locs = try scratch.alloc(*const mlir.Location, args.len);
        for (arg_types, arg_locs, args) |*ty, *l, a| {
            ty.* = switch (a.kind) {
                .ptr => |dt| self.ptrTy(dt, .gmem, 16),
                .tensor => |t| self.ptrTy(t.dtype, .gmem, t.alignment),
                .mlir_type => |typed| typed.type_,
            } catch return error.InvalidMlir;
            l.* = self.loc();
        }

        const entry = mlir.Block.init(arg_types, arg_locs);
        const empty_dict: *const mlir.Attribute = .dict(ctx, &.{});
        const arg_attrs = try scratch.alloc(*const mlir.Attribute, args.len);
        var any_arg_attr = false;
        for (arg_attrs, args) |*attr, a| {
            attr.* = switch (a.kind) {
                .mlir_type => |typed| if (typed.cuda_grid_constant) .dict(ctx, &.{.named(ctx, "cuda.grid_constant", .unit(ctx))}) else if (typed.grid_constant) .dict(ctx, &.{cute_nvgpu.gridConstant(ctx)}) else empty_dict,
                else => empty_dict,
            };
            if (attr.* != empty_dict) any_arg_attr = true;
        }

        const func_op = switch (self.function_kind) {
            .func_only_kernel => func.func(ctx, .{
                .name = self.name,
                .block = entry,
                .args_attributes = if (any_arg_attr) arg_attrs else null,
                .results = &.{},
                .location = self.loc(),
                .verify = false,
                .extra_attributes = &.{
                    cute_ir.kernel(ctx),
                    gpu.kernel(ctx),
                },
            }),
            .cuda_kernel => cuda.kernel(ctx, .{
                .name = self.name,
                .function_type = .function(ctx, arg_types, &.{}),
                .body = entry,
                .arg_attrs = .array(ctx, arg_attrs),
                .extra_attributes = &.{
                    cute_ir.kernel(ctx),
                    gpu.kernel(ctx),
                },
                .location = self.loc(),
            }),
            .host => func.func(ctx, .{
                .name = self.name,
                .block = entry,
                .args_attributes = if (any_arg_attr) arg_attrs else null,
                .results = &.{.int(ctx, .i32)},
                .location = self.loc(),
                .verify = false,
                .extra_attributes = &.{llvm.emitCInterface(ctx)},
            }),
        };
        _ = func_op.appendTo(if (self.function_kind == .cuda_kernel)
            self.gpu_body orelse @panic("cuda_kernel requires Builder.openProgram")
        else
            self.module.body());

        self.func_op = func_op;
        self.entry_block = entry;
        self.block_stack.push(self.arena.allocator(), entry);
        self.args = args;

        self.tensors = try scratch.alloc(?Tensor, args.len);
        for (self.tensors, args, 0..) |*t, a, i| {
            t.* = switch (a.kind) {
                .ptr => null,
                .tensor => |spec| self.makeTensor(self.arg(i), self.makeLayout(spec.shape, .{
                    .stride = spec.stride orelse self.rowMajor(spec.shape),
                })),
                .mlir_type => null,
            };
        }
    }

    pub fn arg(self: *Builder, i: usize) Value {
        const eb = self.entry_block orelse @panic("Builder.arg called before declareArgs");
        return .{ .inner = eb.argument(i), .kernel = self };
    }

    pub fn pushBlock(self: *Builder, b: *mlir.Block) void {
        self.block_stack.push(self.arena.allocator(), b);
    }

    pub fn popBlock(self: *Builder) void {
        self.block_stack.pop();
    }

    pub fn currentBlock(self: *Builder) *mlir.Block {
        return self.block_stack.current();
    }

    pub fn emit(self: *Builder, op: *mlir.Operation) Value {
        _ = op.appendTo(self.currentBlock());
        return .{ .inner = op.result(0), .kernel = self };
    }

    /// Append an operation which does not return an SSA value.
    pub fn emitVoid(self: *Builder, op: *mlir.Operation) void {
        std.debug.assert(op.numResults() == 0);
        _ = op.appendTo(self.currentBlock());
    }

    /// Append a multi-result operation and return all of its SSA values.
    ///
    /// Most of the small CuTe kernels only need one-result operations, but
    /// TMA descriptor construction and TMA partitioning return an atom and a
    /// coordinate tensor together.  Keeping both results from one operation
    /// is required: rebuilding either value changes the descriptor identity.
    pub fn emitResults(self: *Builder, comptime N: usize, op: *mlir.Operation) [N]Value {
        std.debug.assert(op.numResults() == N);
        _ = op.appendTo(self.currentBlock());
        var result: [N]Value = undefined;
        for (&result, 0..) |*value, i| value.* = .{ .inner = op.result(i), .kernel = self };
        return result;
    }

    /// Convert any DSL handle into a scalar wrapper without changing its
    /// MLIR type. This keeps the low-level generated bindings usable while
    /// callers work with `Tensor`, `View`, and `Atom` handles.
    pub fn asValue(self: *Builder, value_: anytype) Value {
        return switch (@TypeOf(value_)) {
            Value => value_,
            Tensor => value_.value(),
            View => value_.value(),
            Atom => value_.value(),
            LaunchConfig => value_.value(),
            Layout => .{ .inner = value_.inner, .kernel = self },
            else => @compileError("expected a CuTe DSL handle, got " ++ @typeName(@TypeOf(value_))),
        };
    }

    fn innerSlice(self: *Builder, values: anytype) []const *const mlir.Value {
        const out = self.arena.allocator().alloc(*const mlir.Value, values.len) catch @panic("OOM");
        switch (@typeInfo(@TypeOf(values))) {
            .@"struct" => {
                inline for (values, 0..) |value, i| out[i] = self.asValue(value).inner;
            },
            else => {
                for (values, out) |value, *dst| dst.* = self.asValue(value).inner;
            },
        }
        return out;
    }

    pub fn loc(self: *const Builder) *const mlir.Location {
        return .unknown(self.ctx);
    }

    /// `#cuda.dev_max_shared_memory_optin`, the device's opt-in maximum of
    /// shared memory per block, for a kernel's `cu_attrs`.
    pub fn devMaxSharedMemoryOptin(self: *Builder) *const mlir.Attribute {
        return (cuda.ops.DevMaxSharedMemoryOptinAttr.get(self.ctx, .{}) catch @panic("invalid CUDA attribute")).attribute();
    }

    pub fn setFunctionAttribute(self: *Builder, attribute: mlir.NamedAttribute) void {
        const function = self.func_op orelse @panic("setFunctionAttribute called before declareArgs");
        function.setAttribute(attribute);
    }

    // ==================== cuda host launch ====================

    /// `@kernels::@<kernel_name>`, a kernel of the nested GPU module.
    fn kernelSymbol(self: *Builder, kernel_name: []const u8) *const mlir.Attribute {
        return .symbolRef(self.ctx, "kernels", &.{.init(self.ctx, kernel_name)});
    }

    pub fn kernelSmemSize(self: *Builder, kernel_name: []const u8) Value {
        return self.emit(cute_ir.kernel_smem_size(
            self.ctx,
            .int(self.ctx, .i64),
            self.kernelSymbol(kernel_name),
            self.loc(),
        ));
    }

    /// Convert the stream placeholder used by a generated host entry into a
    /// CUDA stream. The native CuTe host bridge substitutes XLA's current
    /// stream when it executes the wrapper.
    pub fn cudaStream(self: *Builder) Value {
        return self.emit(cuda.cast(self.ctx, cuda.streamType(self.ctx), self.constant(.i64, 0).inner, self.loc()));
    }

    pub const LaunchOptions = struct {
        grid: [3]Value,
        block: [3]Value,
        dynamic_smem: Value,
        stream: Value,
        cluster: ?[3]Value = null,
        cooperative: bool = false,
        use_pdl: bool = false,
    };

    /// Build the mutable CUDA launch configuration used by `cuda.launch_ex`.
    pub fn makeLaunchConfig(self: *Builder, opts: LaunchOptions) LaunchConfig {
        const config = self.emit(cuda.launch_cfg_create(self.ctx, .{
            .max_attrs = 17,
            .block = .{ opts.block[0].inner, opts.block[1].inner, opts.block[2].inner },
            .dynamic_smem = opts.dynamic_smem.inner,
            .grid = .{ opts.grid[0].inner, opts.grid[1].inner, opts.grid[2].inner },
            .stream = opts.stream.inner,
            .location = self.loc(),
        }));
        if (opts.cluster) |cluster| {
            self.emitVoid(cuda.launch_cfg_cluster_dim(self.ctx, config.inner, .{ cluster[0].inner, cluster[1].inner, cluster[2].inner }, self.loc()));
        }
        self.emitVoid(cuda.launch_cfg_cooperative(self.ctx, config.inner, self.constant(.i32, @intFromBool(opts.cooperative)).inner, self.loc()));
        self.emitVoid(cuda.launch_cfg_programmatic_stream_serialization_allowed(self.ctx, config.inner, self.constant(.i32, @intFromBool(opts.use_pdl)).inner, self.loc()));
        return .{ .inner = config.inner, .kernel = self };
    }

    /// Request a CUDA kernel launch from a generated host function. The host
    /// bridge records this request; XLA performs or command-buffer-records the
    /// actual launch on its stream.
    pub fn launchEx(self: *Builder, kernel_name: []const u8, config: LaunchConfig, arguments: anytype) Value {
        const values = self.innerSlice(arguments);
        return self.emit(cuda.launch_ex(self.ctx, .{
            .config = config.inner,
            .inputs = values,
            .callee = self.kernelSymbol(kernel_name),
            .assume_kernel_attr = cuda.assumeKernelAttr(self.ctx, true),
            .location = self.loc(),
        }));
    }

    pub fn cudaResultStatus(self: *Builder, result: Value) Value {
        return self.emit(cuda.cast(self.ctx, .int(self.ctx, .i32), result.inner, self.loc()));
    }

    pub fn returnHostStatus(self: *Builder, status: Value) void {
        self.emitVoid(func.returns(self.ctx, &.{status.inner}, self.loc()));
    }

    /// Finish the current function but retain the surrounding module so a
    /// host launch function can be emitted next.
    pub fn endFunction(self: *Builder, block: ?[3]i32) void {
        const current = self.currentBlock();
        if (current.terminator() == null) {
            const terminator = if (self.function_kind == .cuda_kernel)
                cuda.return_(self.ctx, &.{}, self.loc())
            else
                func.returns(self.ctx, &.{}, self.loc());
            _ = terminator.appendTo(current);
        }
        if (block) |dims| {
            const function = self.func_op orelse @panic("endFunction before declareArgs");
            function.setAttribute(nvvm.reqntid(self.ctx, dims));
        }
        self.func_op = null;
        self.entry_block = null;
        self.block_stack.clear();
        self.args = &.{};
        self.tensors = &.{};
    }

    pub fn finishProgram(self: *Builder) FinishError![:0]const u8 {
        std.debug.assert(self.entry_block == null and self.func_op == null);
        try common.verify(self.module, "dsl/cute");
        return self.renderModule();
    }

    fn renderModule(self: *Builder) FinishError![:0]const u8 {
        var printed: std.Io.Writer.Allocating = .init(self.allocator);
        defer printed.deinit();
        // Always the generic form: NVIDIA's compiler parses the module with its
        // own MLIR revision, whose custom assembly differs from ours (e.g.
        // `cute.make_composed_layout`, or `llvm.load`/`llvm.store` printing
        // their alignment as a `<alignment = N>` property).
        try printed.writer.print("{f}", .{self.module.operation().fmt(.{ .print_generic_op_form = true })});
        return try self.allocator.dupeSentinel(u8, printed.written(), 0);
    }

    // ==================== types ====================

    pub fn ptrTy(self: *Builder, dt: DType, space: MemorySpace, alignment: u64) !*const mlir.Type {
        return cute_ir.pointerType(self.ctx, space, alignment, .{ .element_type = dt.toMlir(self.ctx) });
    }

    pub fn swizzledPtrTy(self: *Builder, dt: DType, space: MemorySpace, alignment: u64, swizzle: Swizzle) !*const mlir.Type {
        return (try cute_ir.PtrType.get(self.ctx, .{
            .valueType = dt.toMlir(self.ctx),
            .addressSpace = space,
            .alignment = alignment,
            .swizzle = (try self.swizzleAttr(swizzle)).attribute(),
        })).type_();
    }

    fn swizzleAttr(self: *Builder, swizzle: Swizzle) !*const cute_ir.SwizzleAttr {
        return cute_ir.SwizzleAttr.get(self.ctx, .{
            .bits = std.math.cast(u32, swizzle.bits) orelse return error.InvalidMlir,
            .base = std.math.cast(u32, swizzle.base) orelse return error.InvalidMlir,
            .shift = std.math.cast(i32, swizzle.shift) orelse return error.InvalidMlir,
        });
    }

    pub fn intToPtr(self: *Builder, address: anytype, dt: DType, space: MemorySpace, alignment: u64) Value {
        const ptr_type = self.ptrTy(dt, space, alignment) catch @panic("invalid CuTe pointer type");
        return self.emit(cute_ir.inttoptr(self.ctx, self.lift(address).inner, ptr_type, self.loc()));
    }

    /// `cute.ptrtoint`: the integer address of a pointer (`i32` for shared
    /// memory, `i64` for global memory), for hand-written PTX operands.
    pub fn ptrToInt(self: *Builder, pointer: anytype, dt: DType) Value {
        const ptr = self.asValue(pointer);
        return self.emit(if (ptr.isPtr()) cute_ir.ptrtoint(self.ctx, ptr.inner, dt.toMlir(self.ctx), self.loc()) else llvm.ptrtoint(self.ctx, ptr.inner, dt.toMlir(self.ctx), self.loc()));
    }

    /// Reinterpret a pointer without changing its address. This is the typed
    /// counterpart of Python CuTe's `recast_ptr`; packed MXFP4 buffers use it
    /// to expose their byte storage to a scalar correctness implementation.
    pub fn recastPointer(self: *Builder, pointer: anytype, dt: DType, space: MemorySpace, alignment: u64) Value {
        const result_type = self.ptrTy(dt, space, alignment) catch @panic("invalid recast pointer type");
        return self.emit(cute_ir.recast_iter(self.ctx, self.asValue(pointer).inner, result_type, self.loc()));
    }

    pub fn recastSwizzledPointer(self: *Builder, pointer: anytype, dt: DType, space: MemorySpace, alignment: u64, swizzle: Swizzle) Value {
        const result_type = self.swizzledPtrTy(dt, space, alignment, swizzle) catch @panic("invalid swizzled pointer type");
        return self.emit(cute_ir.recast_iter(self.ctx, self.asValue(pointer).inner, result_type, self.loc()));
    }

    /// The scalar of a single extent, otherwise the tuple of `dims`.
    fn dimsExpr(self: *Builder, dims: []const i64) AlgebraExpr {
        if (dims.len == 1) return .{ .static = dims[0] };
        const elements = self.arena.allocator().alloc(AlgebraExpr, dims.len) catch @panic("OOM");
        for (dims, elements) |d, *e| e.* = .{ .static = d };
        return .{ .tuple = elements };
    }

    /// Build a CuTe algebra expression from integers, nested Zig tuples,
    /// `AlgebraToken`s, `ConstrainedDynamic`s and `ScaledBasis` values.
    pub fn expr(self: *Builder, value: anytype) AlgebraExpr {
        const T = @TypeOf(value);
        if (T == AlgebraExpr) return value;
        if (T == AlgebraToken) return switch (value) {
            .all => .all,
            .dynamic => .{ .dynamic = 1 },
        };
        if (T == ScaledBasis) return .{ .basis = value };
        if (T == ConstrainedDynamic) {
            if (value.divisible_by == 0) @panic("CuTe constrained dynamic divisor must be positive");
            return .{ .dynamic = @intCast(value.divisible_by) };
        }
        switch (@typeInfo(T)) {
            .comptime_int, .int => return .{ .static = @intCast(value) },
            .@"struct" => |info| {
                const elements = self.arena.allocator().alloc(AlgebraExpr, info.field_names.len) catch @panic("OOM");
                inline for (info.field_names, 0..) |field, i| elements[i] = self.expr(@field(value, field));
                return .{ .tuple = elements };
            },
            else => @compileError("unsupported CuTe algebra value " ++ @typeName(T)),
        }
    }

    /// The algebra attribute `A` (`cute_ir.IntTupleAttr`, `ShapeAttr`, ...)
    /// of `e`.
    fn algebraAttr(self: *Builder, comptime A: type, e: AlgebraExpr) *const A {
        const result = switch (e) {
            .static => |v| if (@hasDecl(A, "getStatic")) A.getStatic(self.ctx, v) else std.debug.panic("{s} has no integer leaf", .{@typeName(A)}),
            .dynamic => |divisibility| if (@hasDecl(A, "getDynamic")) A.getDynamic(self.ctx, 32, divisibility) else std.debug.panic("{s} has no dynamic leaf", .{@typeName(A)}),
            .dynamic_int => |v| if (@hasDecl(A, "getDynamic")) A.getDynamic(self.ctx, v.width, v.divisible_by) else std.debug.panic("{s} has no dynamic leaf", .{@typeName(A)}),
            .all => if (@hasDecl(A, "getUnderscore")) A.getUnderscore(self.ctx) else std.debug.panic("{s} has no `_` leaf", .{@typeName(A)}),
            .basis => |b| if (@hasDecl(A, "getBasis")) blk: {
                const modes = self.arena.allocator().alloc(i32, b.mode_count) catch @panic("OOM");
                for (modes, b.modes[0..b.mode_count]) |*mode, m| mode.* = m;
                break :blk A.getBasis(self.ctx, b.numerator, b.denominator, modes);
            } else std.debug.panic("{s} has no scaled-basis leaf", .{@typeName(A)}),
            .tuple => |elements| blk: {
                const attrs = self.arena.allocator().alloc(*const A, elements.len) catch @panic("OOM");
                for (elements, attrs) |element, *attr| attr.* = self.algebraAttr(A, element);
                break :blk A.getTuple(self.ctx, attrs);
            },
        };
        return result catch std.debug.panic("invalid CuTe algebra for {s}", .{@typeName(A)});
    }

    fn layoutAttr(self: *Builder, spec: LayoutSpec) *const cute_ir.LayoutAttr {
        return cute_ir.LayoutAttr.get(self.ctx, .{
            .shape = self.algebraAttr(cute_ir.ShapeAttr, spec.shape),
            .stride = self.algebraAttr(cute_ir.StrideAttr, spec.stride),
        }) catch @panic("invalid CuTe layout");
    }

    fn intTupleType(self: *Builder, e: AlgebraExpr) *const mlir.Type {
        return (cute_ir.IntTupleType.get(self.ctx, .{ .attr = self.algebraAttr(cute_ir.IntTupleAttr, e).attribute() }) catch @panic("invalid CuTe integer tuple")).type_();
    }

    /// `numerator/denominator` along `modes`, the mode path outermost first like
    /// Python's `ScaledBasis(value, modes)`. CuTe prints it innermost first:
    /// `basis(1, 2, .{ 0, 1 })` is `1/2@1@0`.
    pub fn basis(self: *Builder, numerator: i64, denominator: i64, modes: anytype) ScaledBasis {
        _ = self;
        if (denominator <= 0) @panic("CuTe scaled-basis denominator must be positive");
        const fields = @typeInfo(@TypeOf(modes)).@"struct".field_names;
        if (fields.len == 0 or fields.len > MAX_RANK) @panic("CuTe scaled basis needs 1..MAX_RANK modes");
        var result: ScaledBasis = .{ .numerator = numerator, .denominator = denominator, .mode_count = fields.len };
        inline for (fields, 0..) |field, i| result.modes[i] = @intCast(@field(modes, field));
        return result;
    }

    pub fn layoutSpec(self: *Builder, shape: anytype, stride: anytype) LayoutSpec {
        return .{ .shape = self.expr(shape), .stride = self.expr(stride) };
    }

    pub fn layoutType(self: *Builder, spec: LayoutSpec) *const mlir.Type {
        return (cute_ir.LayoutType.get(self.ctx, .{ .attr = self.layoutAttr(spec).attribute() }) catch @panic("invalid CuTe layout")).type_();
    }

    pub fn composedLayoutType(self: *Builder, swizzle: Swizzle, offset: anytype, outer: LayoutSpec) *const mlir.Type {
        const attr = cute_ir.ComposedLayoutAttr.get(self.ctx, .{
            .inner = .{ .swizzle = self.swizzleAttr(swizzle) catch @panic("invalid CuTe swizzle") },
            .offset = self.algebraAttr(cute_ir.IntTupleAttr, self.expr(offset)),
            .outer = self.layoutAttr(outer),
        }) catch @panic("invalid CuTe composed layout");
        return (cute_ir.ComposedLayoutType.get(self.ctx, .{ .attr = attr.attribute() }) catch @panic("invalid CuTe composed layout")).type_();
    }

    pub fn memrefType(self: *Builder, dtype: DType, space: MemorySpace, alignment: u64, layout_type: *const mlir.Type) *const mlir.Type {
        const pointer = self.ptrTy(dtype, space, alignment) catch @panic("invalid CuTe pointer type");
        return (cute_ir.MemRefType.get(self.ctx, .{ .ptr = pointer, .layout = layout_type }) catch @panic("invalid CuTe memref type")).type_();
    }

    pub fn smemDescType(self: *Builder) *const mlir.Type {
        return (cute_nvgpu.SmemDescType.get(self.ctx, .{}) catch @panic("invalid shared-memory descriptor")).type_();
    }

    /// `!cute_nvgpu.smem_desc_view` of a shared-memory descriptor over `layout`.
    pub fn smemDescViewType(self: *Builder, layout: LayoutSpec) *const mlir.Type {
        return (cute_nvgpu.SmemDescViewType.get(self.ctx, .{
            .desc = self.smemDescType(),
            .layout = self.layoutAttr(layout).attribute(),
        }) catch @panic("invalid shared-memory descriptor view")).type_();
    }

    /// `!cute.tiled_copy` of `copy_atom` over `layout_copy_tv`, tiled by the
    /// modes of `tiler_mn`.
    pub fn tiledCopyType(self: *Builder, copy_atom: *const mlir.Type, layout_copy_tv: LayoutSpec, tiler_mn: []const LayoutSpec) *const mlir.Type {
        return (cute_ir.TiledCopyType.get(self.ctx, .{
            .copyAtom = copy_atom,
            .layoutCopyTv = self.layoutAttr(layout_copy_tv).attribute(),
            .tilerMn = self.tileAttr(tiler_mn).attribute(),
        }) catch @panic("invalid tiled copy")).type_();
    }

    pub fn memrefTypeFromPointer(self: *Builder, pointer_type: *const mlir.Type, layout_type: *const mlir.Type) *const mlir.Type {
        return (cute_ir.MemRefType.get(self.ctx, .{ .ptr = pointer_type, .layout = layout_type }) catch @panic("invalid CuTe memref type")).type_();
    }

    /// `cute.static` of an integer tuple such as `128` or `.{ 0, 4 }`.
    pub fn staticIntTuple(self: *Builder, value: anytype) View {
        const result = self.emit(cute_ir.static(self.ctx, self.intTupleType(self.expr(value)), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn staticLayout(self: *Builder, spec: LayoutSpec) View {
        const result = self.emit(cute_ir.static(self.ctx, self.layoutType(spec), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub const Swizzle = struct { bits: i64, base: i64, shift: i64 };

    pub fn staticSwizzle(self: *Builder, spec: Swizzle) View {
        const attr = self.swizzleAttr(spec) catch @panic("invalid CuTe swizzle");
        const ty = (cute_ir.SwizzleType.get(self.ctx, .{ .attr = attr.attribute() }) catch @panic("invalid CuTe swizzle")).type_();
        const result = self.emit(cute_ir.static(self.ctx, ty, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Construct `swizzle o offset o outer_layout` through the CuTe dialect.
    pub fn makeComposedLayout(self: *Builder, swizzle: Swizzle, offset: anytype, outer: LayoutSpec) View {
        const inner = self.staticSwizzle(swizzle);
        const offset_value = self.emit(cute_ir.make_int_tuple(self.ctx, &.{}, self.intTupleType(self.expr(offset)), self.loc()));
        const outer_value = self.staticLayout(outer);
        const result_ty = self.composedLayoutType(swizzle, offset, outer);
        const result = self.emit(cute_ir.make_composed_layout(self.ctx, inner.inner, offset_value.inner, outer_value.inner, result_ty, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Compose a raw pointer and any CuTe layout value. The memref result type
    /// is obtained through the dialect C API rather than parsed from assembly.
    pub fn makeTensorView(self: *Builder, pointer: anytype, layout: anytype) View {
        const p = self.asValue(pointer);
        const l = self.asValue(layout);
        const result_ty = (cute_ir.MemRefType.get(self.ctx, .{ .ptr = p.type_(), .layout = l.type_() }) catch @panic("invalid CuTe tensor view")).type_();
        const result = self.emit(cute_ir.make_view(self.ctx, p.inner, result_ty, .{ .layout = l.inner }, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn coordTensorType(self: *Builder, rank: usize, layout: LayoutSpec) *const mlir.Type {
        if (rank == 0 or rank > MAX_RANK) @panic("coordTensorType rank must be in 1..MAX_RANK");
        const zeros: [MAX_RANK]i64 = @splat(0);
        return self.coordTensorTypeWithTuple(self.dimsExpr(zeros[0..rank]), layout);
    }

    /// Coordinate-tensor type with an explicit arithmetic-tuple expression,
    /// such as `.{ 0, ConstrainedDynamic{ .divisible_by = 128 }, AlgebraToken.dynamic }`.
    /// This represents the divisibility information produced by CuTe tiling
    /// without requiring kernel code to spell textual MLIR types.
    pub fn coordTensorTypeWithTuple(self: *Builder, tuple: anytype, layout: LayoutSpec) *const mlir.Type {
        return (cute_ir.CoordTensorType.get(self.ctx, .{ .arithTuple = self.intTupleType(self.expr(tuple)), .layout = self.layoutType(layout) }) catch @panic("invalid coordinate tensor")).type_();
    }

    fn rowMajor(self: *Builder, shape: []const i64) []const i64 {
        const s = self.arena.allocator().alloc(i64, shape.len) catch @panic("OOM");
        var acc: i64 = 1;
        var i = shape.len;
        while (i > 0) : (i -= 1) {
            s[i - 1] = acc;
            acc *= shape[i - 1];
        }
        return s;
    }

    fn leftMost(self: *Builder, shape: []const i64) []const i64 {
        const s = self.arena.allocator().alloc(i64, shape.len) catch @panic("OOM");
        var acc: i64 = 1;
        for (shape, 0..) |d, i| {
            s[i] = acc;
            acc *= d;
        }
        return s;
    }

    // ==================== cute.core ====================

    fn layoutResult(comptime Shape: type) type {
        return switch (@typeInfo(Shape)) {
            .pointer => Layout,
            else => View,
        };
    }

    /// `cute.make_layout(shape, stride=...)`. Slices keep the tracked static
    /// Layout API; nested tuples accept mixed integer and SSA shape/stride leaves.
    pub fn makeLayout(self: *Builder, shape: anytype, opts: anytype) layoutResult(@TypeOf(shape)) {
        if (comptime layoutResult(@TypeOf(shape)) == Layout) return self.makeStaticLayout(shape, .{
            .stride = if (@hasField(@TypeOf(opts), "stride")) opts.stride else null,
        });
        var shape_values: stdx.BoundedArray(*const mlir.Value, MAX_RANK) = .empty;
        var stride_values: stdx.BoundedArray(*const mlir.Value, MAX_RANK) = .empty;
        const shape_expr = self.coordExpr(&shape_values, shape);
        const stride_expr = if (@hasField(@TypeOf(opts), "stride") and @TypeOf(opts.stride) != @TypeOf(null)) self.coordExpr(&stride_values, opts.stride) else blk: {
            var stride: AlgebraExpr = .{ .static = 1 };
            var stride_value: ?Value = null;
            var index: usize = 0;
            break :blk self.compactLayoutStride(shape_expr, shape_values.constSlice(), &index, &stride, &stride_value, &stride_values);
        };
        const shape_value = self.emit(cute_ir.make_shape(self.ctx, shape_values.constSlice(), self.shapeType(shape_expr), self.loc()));
        const stride_type = (cute_ir.StrideType.get(self.ctx, .{ .attr = self.algebraAttr(cute_ir.StrideAttr, stride_expr).attribute() }) catch @panic("invalid stride")).type_();
        const stride_value = self.emit(cute_ir.make_stride(self.ctx, stride_values.constSlice(), stride_type, self.loc()));
        const result = self.emit(cute_ir.make_layout(self.ctx, self.layoutType(.{ .shape = shape_expr, .stride = stride_expr }), .{ .shape = shape_value.inner, .stride = stride_value.inner }, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    fn compactLayoutStride(self: *Builder, shape: AlgebraExpr, values: []const *const mlir.Value, index: *usize, stride: *AlgebraExpr, stride_value: *?Value, dynamic: *stdx.BoundedArray(*const mlir.Value, MAX_RANK)) AlgebraExpr {
        if (shape == .tuple) {
            const elements = self.arena.allocator().alloc(AlgebraExpr, shape.tuple.len) catch @panic("OOM");
            for (shape.tuple, elements) |extent, *element| element.* = self.compactLayoutStride(extent, values, index, stride, stride_value, dynamic);
            return .{ .tuple = elements };
        }
        if (shape == .static and shape.static == 1) return .{ .static = 0 };
        const result = stride.*;
        if (stride_value.*) |v| dynamic.append(v.inner) catch @panic("too many stride leaves");
        if (shape == .static and stride.* == .static) {
            stride.static *= shape.static;
        } else {
            const extent = if (shape == .dynamic or shape == .dynamic_int) blk: {
                const v: Value = .{ .inner = values[index.*], .kernel = self };
                index.* += 1;
                break :blk v;
            } else self.lift(shape.static);
            stride_value.* = (stride_value.* orelse self.lift(stride.static)).mul(extent);
            stride.* = dynamicExpr(stride_value.*.?);
        }
        return result;
    }

    fn makeStaticLayout(self: *Builder, shape: []const i64, opts: LayoutOpts) Layout {
        if (shape.len == 0 or shape.len > MAX_RANK) std.debug.panic("makeLayout: rank {d} not in 1..{d}", .{ shape.len, MAX_RANK });
        const stride = opts.stride orelse self.leftMost(shape);
        if (stride.len != shape.len) std.debug.panic("makeLayout: stride rank {d} != shape rank {d}", .{ stride.len, shape.len });
        const layout_ty = self.layoutType(.{ .shape = self.dimsExpr(shape), .stride = self.dimsExpr(stride) });
        const l = self.emit(cute_ir.static(self.ctx, layout_ty, self.loc()));
        return .{ .inner = l.inner, .shape = Dims.fromSlice(shape) catch unreachable, .stride = Dims.fromSlice(stride) catch unreachable };
    }

    /// `cute.make_tensor(ptr, layout)`.
    pub fn makeTensor(self: *Builder, ptr: Value, layout: Layout) Tensor {
        const pt = ptr.type_().isA(cute_ir.PtrType) orelse std.debug.panic("makeTensor: {f} is not a !cute.ptr", .{ptr.type_()});
        const elem = pt.valueType() orelse @panic("makeTensor: untyped pointer");
        const memref_ty = (cute_ir.MemRefType.get(self.ctx, .{ .ptr = ptr.type_(), .layout = layout.inner.type_() }) catch @panic("bad memref")).type_();
        const v = self.emit(cute_ir.make_view(self.ctx, ptr.inner, memref_ty, .{ .layout = layout.inner }, self.loc()));
        return .{ .inner = v.inner, .kernel = self, .layout = layout, .dtype = dtypes.mlirToDType(self.ctx, elem), .space = pt.addressSpace() };
    }

    /// The algebra of a coordinate whose `Value` leaves are dynamic and
    /// appended to `dyn`. A one-element tuple is its element.
    fn coordExpr(self: *Builder, dyn: *stdx.BoundedArray(*const mlir.Value, MAX_RANK), coord: anytype) AlgebraExpr {
        const T = @TypeOf(coord);
        if (T == Value) {
            dyn.append(coord.inner) catch @panic("makeCoord: too many dynamic leaves");
            return dynamicExpr(coord);
        }
        if (T == AlgebraToken) return switch (coord) {
            .all => .all,
            .dynamic => @panic("makeCoord: AlgebraToken.dynamic needs an SSA value"),
        };
        switch (@typeInfo(T)) {
            .comptime_int, .int => return .{ .static = @intCast(coord) },
            .@"struct" => |info| {
                if (info.field_names.len == 0) @compileError("makeCoord: empty tuple");
                if (info.field_names.len == 1) return self.coordExpr(dyn, @field(coord, info.field_names[0]));
                const elements = self.arena.allocator().alloc(AlgebraExpr, info.field_names.len) catch @panic("OOM");
                inline for (info.field_names, 0..) |field, i| elements[i] = self.coordExpr(dyn, @field(coord, field));
                return .{ .tuple = elements };
            },
            else => @compileError("makeCoord: unsupported coordinate leaf " ++ @typeName(T)),
        }
    }

    fn dynamicExpr(value: Value) AlgebraExpr {
        std.debug.assert(value.isInt());
        const width = dtypeBitwidth(value.dtype());
        return if (width == 32) .{ .dynamic = 1 } else .{ .dynamic_int = .{ .width = @intCast(width) } };
    }

    /// A `!cute.coord` from nested tuples of `Value`s, integers, and `.all`
    /// placeholders. Integer leaves stay static in the dependent CuTe type.
    pub fn makeCoord(self: *Builder, coord: anytype) *const mlir.Value {
        if (@TypeOf(coord) == Value) return self.makeCoord(.{coord});
        var dyn: stdx.BoundedArray(*const mlir.Value, MAX_RANK) = .empty;
        const attr = self.algebraAttr(cute_ir.CoordAttr, self.coordExpr(&dyn, coord));
        const ty = (cute_ir.CoordType.get(self.ctx, .{ .attr = attr.attribute() }) catch @panic("bad coord")).type_();
        return self.emit(cute_ir.make_coord(self.ctx, dyn.constSlice(), ty, self.loc())).inner;
    }

    /// A dynamic/static CuTe integer tuple, used for pointer arithmetic in
    /// tensor and shared memory. It accepts the same nested Zig values as a
    /// coordinate, but preserves integer-tuple semantics.
    pub fn makeIntTuple(self: *Builder, values: anytype) View {
        var dyn: stdx.BoundedArray(*const mlir.Value, MAX_RANK) = .empty;
        const ty = self.intTupleType(self.coordExpr(&dyn, values));
        const value = self.emit(cute_ir.make_int_tuple(self.ctx, dyn.constSlice(), ty, self.loc()));
        return .{ .inner = value.inner, .kernel = self };
    }

    /// A one-element integer tuple holding a value known to be a multiple of
    /// `divisor`. Ops that derive alignment from an offset, such as
    /// `cute.add_offset`, need that to keep the operand's alignment.
    pub fn makeIntTupleDivBy(self: *Builder, value: anytype, comptime divisor: u32) View {
        const constrained = self.assumeDivBy(value, divisor);
        const tuple = self.emit(cute_ir.make_int_tuple(self.ctx, &.{constrained.inner}, self.intTupleType(.{ .dynamic = divisor }), self.loc()));
        return .{ .inner = tuple.inner, .kernel = self };
    }

    /// Refine an integer SSA value with the divisibility information CuTe
    /// derives while slicing a tiled coordinate tensor.
    pub fn assumeDivBy(self: *Builder, value: anytype, comptime divisor: u32) Value {
        if (divisor == 0) @compileError("assumeDivBy divisor must be non-zero");
        const constrained = cute_ir.constrainedIntType(self.ctx, .i32, divisor) catch @panic("invalid CuTe constrained integer");
        return self.emit(cute_ir.assume(self.ctx, self.lift(value).to(.i32).inner, constrained, self.loc()));
    }

    /// Build a coordinate tensor from a runtime origin and a static layout.
    /// This is the direct form of the `get_iter`/`make_arith_tuple_iter`/
    /// `make_view` sequence emitted by the Python CuTe front end after a tile
    /// is selected from a larger coordinate tensor.
    pub fn makeCoordTensor(self: *Builder, origin: anytype, layout: anytype) View {
        const tuple = self.makeIntTuple(origin);
        const layout_value = self.asValue(layout);
        const iterator_type = (cute_ir.ArithTupleIteratorType.get(self.ctx, .{ .arithTuple = tuple.type_() }) catch @panic("invalid arithmetic tuple iterator")).type_();
        const iterator = self.emit(cute_ir.make_arith_tuple_iter(self.ctx, iterator_type, .{ .value_ = tuple.inner }, self.loc()));
        const result_type = (cute_ir.CoordTensorType.get(self.ctx, .{ .arithTuple = tuple.type_(), .layout = layout_value.type_() }) catch @panic("invalid coordinate tensor")).type_();
        const result = self.emit(cute_ir.make_view(self.ctx, iterator.inner, result_type, .{ .layout = layout_value.inner }, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Rebuild a coordinate tensor from the arithmetic-tuple iterator of an
    /// existing tensor.  CuTe's Python front end emits this sequence after it
    /// slices a TMA-partitioned coordinate tensor:
    ///
    ///   get_iter -> deref_arith_tuple_iter -> get_leaves -> make_int_tuple
    ///            -> make_arith_tuple_iter -> make_view
    ///
    /// Keeping that sequence explicit matters for TMA.  The iterator carries
    /// the descriptor coordinates selected by `slice`; constructing a fresh
    /// tuple from the scheduler indices would lose CuTe's coordinate algebra.
    pub fn rebuildCoordTensorIterator(
        self: *Builder,
        comptime leaf_count: usize,
        input: anytype,
        tuple: anytype,
        comptime leaf_divisibility: [leaf_count]u32,
        layout: anytype,
    ) View {
        if (leaf_count == 0 or leaf_count > MAX_RANK) @panic("invalid coordinate tuple leaf count");
        const source = self.asValue(input);
        const source_type = source.type_().isA(cute_ir.CoordTensorType) orelse
            std.debug.panic("rebuildCoordTensorIterator: expected coordinate tensor, got {f}", .{source.type_()});
        const source_tuple_type = source_type.arithTuple();
        const source_iterator_type = (cute_ir.ArithTupleIteratorType.get(self.ctx, .{ .arithTuple = source_tuple_type }) catch
            @panic("invalid source arithmetic tuple iterator")).type_();
        const source_iterator = self.emit(cute_ir.get_iter(self.ctx, source.inner, source_iterator_type, self.loc()));
        const source_tuple = self.emit(cute_ir.deref_arith_tuple_iter(
            self.ctx,
            source_iterator.inner,
            source_tuple_type,
            self.loc(),
        ));

        const leaf_types = self.arena.allocator().alloc(*const mlir.Type, leaf_count) catch @panic("OOM");
        // 0 marks a leaf the producer knows to be statically zero, 1 a leaf
        // with no known divisibility.
        inline for (leaf_divisibility, 0..) |divisor, i| {
            leaf_types[i] = self.intTupleType(if (divisor == 0) .{ .static = 0 } else .{ .dynamic = divisor });
        }
        const leaves = self.emitResults(leaf_count, cute_ir.get_leaves(
            self.ctx,
            source_tuple.inner,
            leaf_types,
            self.loc(),
        ));

        const rebuilt_tuple_type = self.intTupleType(self.expr(tuple));
        const rebuilt_tuple = self.emit(cute_ir.make_int_tuple(
            self.ctx,
            self.innerSlice(leaves[0..]),
            rebuilt_tuple_type,
            self.loc(),
        ));
        const rebuilt_iterator_type = (cute_ir.ArithTupleIteratorType.get(self.ctx, .{ .arithTuple = rebuilt_tuple_type }) catch
            @panic("invalid rebuilt arithmetic tuple iterator")).type_();
        const rebuilt_iterator = self.emit(cute_ir.make_arith_tuple_iter(self.ctx, rebuilt_iterator_type, .{ .value_ = rebuilt_tuple.inner }, self.loc()));
        const layout_value = self.asValue(layout);
        const result_type = (cute_ir.CoordTensorType.get(self.ctx, .{
            .arithTuple = rebuilt_tuple_type,
            .layout = layout_value.type_(),
        }) catch @panic("invalid rebuilt coordinate tensor")).type_();
        const result = self.emit(cute_ir.make_view(self.ctx, rebuilt_iterator.inner, result_type, .{ .layout = layout_value.inner }, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// `!cute.shape` of a CuTe algebra value such as `.{ .{ 128, 16 }, 1, 1, 2 }`
    /// or `.{ AlgebraToken.dynamic, AlgebraToken.dynamic }`.
    pub fn shapeType(self: *Builder, shape: anytype) *const mlir.Type {
        return (cute_ir.ShapeType.get(self.ctx, .{ .attr = self.algebraAttr(cute_ir.ShapeAttr, self.expr(shape)).attribute() }) catch @panic("invalid CuTe shape")).type_();
    }

    /// Static `cute.make_shape` of a CuTe algebra value.
    pub fn makeShape(self: *Builder, shape: anytype) View {
        const value = self.emit(cute_ir.make_shape(self.ctx, &.{}, self.shapeType(shape), self.loc()));
        return .{ .inner = value.inner, .kernel = self };
    }

    /// Dynamic `cute.make_shape`. The result type describes which leaves are
    /// runtime values (for example `!cute.shape<"(?,?,?)">`). Keeping these
    /// leaves dynamic is significant to CuTe's TMA layout algebra when a
    /// runtime extent happens to be one.
    pub fn makeShapeValues(self: *Builder, values: anytype, result_type: *const mlir.Type) View {
        const value = self.emit(cute_ir.make_shape(self.ctx, self.innerSlice(values), result_type, self.loc()));
        return .{ .inner = value.inner, .kernel = self };
    }

    pub fn tileToShape(self: *Builder, input: anytype, shape: anytype, order: View, result_type: *const mlir.Type) View {
        const result = self.emit(cute_ir.tile_to_shape(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(shape).inner,
            order.inner,
            result_type,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Static `cute.make_tile` from mode extents.
    pub fn makeTile(self: *Builder, shape: []const i64) View {
        const modes = self.arena.allocator().alloc(LayoutSpec, shape.len) catch @panic("OOM");
        for (shape, modes) |extent, *mode| mode.* = .{ .shape = .{ .static = extent }, .stride = .{ .static = 1 } };
        const value = self.emit(cute_ir.make_tile(self.ctx, &.{}, self.tileType(modes), self.loc()));
        return .{ .inner = value.inner, .kernel = self };
    }

    /// `!cute.tile` whose modes are `layouts`, such as `[512:1;1:0;4:1]`.
    pub fn tileType(self: *Builder, layouts: []const LayoutSpec) *const mlir.Type {
        return (cute_ir.TileType.get(self.ctx, .{ .attr = self.tileAttr(layouts).attribute() }) catch @panic("invalid CuTe tile")).type_();
    }

    fn tileAttr(self: *Builder, layouts: []const LayoutSpec) *const cute_ir.TileAttr {
        const modes = self.arena.allocator().alloc(*const cute_ir.TileAttr, layouts.len) catch @panic("OOM");
        for (layouts, modes) |layout, *mode| mode.* = cute_ir.TileAttr.getLayout(self.ctx, self.layoutAttr(layout)) catch @panic("invalid CuTe tile mode");
        return cute_ir.TileAttr.getTuple(self.ctx, modes) catch @panic("invalid CuTe tile");
    }

    /// Construct a target atom whose complete architecture payload is
    /// encoded in `atom_type` (for example `!cute_nvgpu.sm100.mma_bs<...>`).
    pub fn makeAtom(self: *Builder, atom_type: *const mlir.Type, values: anytype) Atom {
        const result = self.emit(cute_ir.make_atom(self.ctx, self.innerSlice(values), atom_type, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn makeTiledMma(self: *Builder, atom: Atom, result_type: *const mlir.Type) Atom {
        const result = self.emit(cute_ir.make_tiled_mma(self.ctx, atom.inner, result_type, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Allocate shared memory with a type assembled by the Zig DSL.
    pub fn allocSmemView(self: *Builder, dtype: DType, alignment: u64, layout_type: *const mlir.Type) View {
        const memref_type = self.memrefType(dtype, .smem, alignment, layout_type);
        const result = self.emit(cute_ir.memref_alloc_smem(self.ctx, memref_type, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Allocate one compiler-visible shared-memory storage object. The field
    /// descriptions use the CuTe host compiler's `name:size:offset` format;
    /// they let its shared-memory sizing/layout pass reproduce a Python
    /// `SharedStorage` struct while Zig keeps typed pointers to its fields.
    pub fn allocSmemStorage(
        self: *Builder,
        dtype: DType,
        alignment: u64,
        layout_type: *const mlir.Type,
        partition_id: i32,
        fields: []const []const u8,
    ) View {
        const field_attrs = self.arena.allocator().alloc(*const mlir.Attribute, fields.len) catch @panic("OOM");
        for (fields, field_attrs) |field, *attr| attr.* = .string(self.ctx, field);
        const memref_type = self.memrefType(dtype, .smem, alignment, layout_type);
        const result = self.emit(mlir.Operation.make(self.ctx, "cute.memref.alloca", .{
            .results = .{ .flat = &.{memref_type} },
            .attributes = &.{
                .named(self.ctx, "smem.partition_id", .int(self.ctx, .i32, partition_id)),
                .named(self.ctx, "smem.struct_fields", .array(self.ctx, field_attrs)),
            },
            .location = self.loc(),
        }));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Allocate a register-memory tensor whose layout is represented by a
    /// normal CuTe layout value.
    pub fn allocRmem(self: *Builder, dtype: DType, alignment: u64, layout: anytype) View {
        const layout_value = self.asValue(layout);
        const memref_type = self.memrefType(dtype, .rmem, alignment, layout_value.type_());
        const result = self.emit(cute_ir.memref_alloca(self.ctx, memref_type, .{ .layout = layout_value.inner }, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn makeView(self: *Builder, pointer: anytype, layout: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute_ir.make_view(self.ctx, self.asValue(pointer).inner, result_type, .{ .layout = self.asValue(layout).inner }, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn makeViewNoLayout(self: *Builder, pointer: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute_ir.make_view(self.ctx, self.asValue(pointer).inner, result_type, .{}, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// `projection`, like Python's `proj`, selects the modes of `tile` that
    /// apply, such as `b.expr(.{ 1, AlgebraToken.all, 1 })`.
    pub fn localTile(self: *Builder, input: anytype, tile: anytype, coord: anytype, result_type: *const mlir.Type, projection: ?AlgebraExpr) View {
        const coord_value = if (@TypeOf(coord) == View) coord.inner else self.makeCoord(coord);
        const result = self.emit(cute_ir.local_tile(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(tile).inner,
            coord_value,
            result_type,
            .{ .proj = if (projection) |p| self.algebraAttr(cute_ir.CoordAttr, p).attribute() else null },
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn localPartition(self: *Builder, input: anytype, tiler: anytype, index: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute_ir.local_partition(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(tiler).inner,
            self.makeCoord(index),
            result_type,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn slice(self: *Builder, input: anytype, coord: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute_ir.slice(
            self.ctx,
            self.asValue(input).inner,
            self.makeCoord(coord),
            result_type,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Rebuild a tensor with the same iterator and a new layout. This is the
    /// tensor form of CuTe's `group_modes`: grouping changes only the layout,
    /// while the backing pointer or arithmetic-tuple iterator stays intact.
    /// The local CUTLASS schema exposes `cute.group_modes` only for layouts,
    /// so expressing the equivalent algebra this way also remains portable.
    pub fn withLayout(self: *Builder, input: anytype, layout: anytype) View {
        const value = self.asValue(input);
        const layout_value = self.asValue(layout);
        const input_type = value.type_();
        const result_type: *const mlir.Type = if (input_type.isA(cute_ir.MemRefType)) |memref|
            (cute_ir.MemRefType.get(self.ctx, .{ .ptr = memref.ptr_(), .layout = layout_value.type_() }) catch @panic("invalid regrouped memref")).type_()
        else if (input_type.isA(cute_ir.CoordTensorType)) |coords|
            (cute_ir.CoordTensorType.get(self.ctx, .{ .arithTuple = coords.arithTuple(), .layout = layout_value.type_() }) catch @panic("invalid regrouped coordinate tensor")).type_()
        else
            std.debug.panic("withLayout: expected memref or coordinate tensor, got {f}", .{input_type});

        const iterator_type: *const mlir.Type = if (input_type.isA(cute_ir.MemRefType)) |memref|
            memref.ptr_()
        else if (input_type.isA(cute_ir.CoordTensorType)) |coords|
            (cute_ir.ArithTupleIteratorType.get(self.ctx, .{ .arithTuple = coords.arithTuple() }) catch @panic("invalid arithmetic tuple iterator")).type_()
        else
            unreachable;
        const iterator = self.emit(cute_ir.get_iter(self.ctx, value.inner, iterator_type, self.loc()));
        const result = self.emit(cute_ir.make_view(self.ctx, iterator.inner, result_type, .{ .layout = layout_value.inner }, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn groupModes(self: *Builder, input: anytype, begin: u32, end: u32, result_type: *const mlir.Type) View {
        const result = self.emit(cute_ir.group_modes(
            self.ctx,
            self.asValue(input).inner,
            result_type,
            .int(self.ctx, .i32, begin),
            .int(self.ctx, .i32, end),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn tiledDivide(self: *Builder, input: anytype, tiler: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute_ir.tiled_divide(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(tiler).inner,
            result_type,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn flatDivide(self: *Builder, input: anytype, tiler: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute_ir.flat_divide(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(tiler).inner,
            result_type,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn coalesce(self: *Builder, input: anytype, target_profile: ?View, result_type: *const mlir.Type) View {
        const result = self.emit(cute_ir.coalesce(
            self.ctx,
            self.asValue(input).inner,
            result_type,
            .{ .target_profile = if (target_profile) |target| target.inner else null },
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn filterZeros(self: *Builder, input: anytype, target_profile: ?View, result_type: *const mlir.Type) View {
        const result = self.emit(cute_ir.filter_zeros(
            self.ctx,
            self.asValue(input).inner,
            result_type,
            .{ .target_profile = if (target_profile) |target| target.inner else null },
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn getLayout(self: *Builder, input: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute_ir.get_layout(self.ctx, self.asValue(input).inner, result_type, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// `cute.get_iter`: the iterator of `input`, of type `result_type`.
    pub fn getIter(self: *Builder, input: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute_ir.get_iter(self.ctx, self.asValue(input).inner, result_type, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn getShape(self: *Builder, input: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute_ir.get_shape(self.ctx, self.asValue(input).inner, result_type, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn addOffset(self: *Builder, input: anytype, offset: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute_ir.add_offset(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(offset).inner,
            result_type,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn recastIter(self: *Builder, input: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute_ir.recast_iter(self.ctx, self.asValue(input).inner, result_type, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn appendToRank(self: *Builder, input: anytype, element: anytype, rank: u32, result_type: *const mlir.Type) View {
        const result = self.emit(cute_ir.append_to_rank(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(element).inner,
            result_type,
            .int(self.ctx, .i32, rank),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn makeTiledCopy(self: *Builder, atom: Atom, result_type: *const mlir.Type) Atom {
        const result = self.emit(cute_ir.make_tiled_copy(self.ctx, atom.inner, result_type, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn tiledCopyPartition(self: *Builder, tiled: Atom, input: anytype, coord: anytype, destination: bool, result_type: *const mlir.Type) View {
        const op = if (destination)
            cute_ir.tiled_copy_partition_D(self.ctx, tiled.inner, self.asValue(input).inner, self.makeCoord(coord), result_type, self.loc())
        else
            cute_ir.tiled_copy_partition_S(self.ctx, tiled.inner, self.asValue(input).inner, self.makeCoord(coord), result_type, self.loc());
        const result = self.emit(op);
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn tiledCopyRetile(self: *Builder, tiled: Atom, input: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute_ir.tiled_copy_retile(
            self.ctx,
            tiled.inner,
            self.asValue(input).inner,
            result_type,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn tiledMmaPartition(self: *Builder, tiled: Atom, input: anytype, coord: anytype, operand_id: u2, result_type: *const mlir.Type) View {
        const result = self.emit(cute_ir.tiled_mma_partition(
            self.ctx,
            tiled.inner,
            self.asValue(input).inner,
            self.makeCoord(coord),
            result_type,
            .int(self.ctx, .i32, operand_id),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn mmaMakeFragment(self: *Builder, tiled: Atom, input: anytype, operand_id: u2, result_type: *const mlir.Type) View {
        const result = self.emit(cute_ir.mma_make_fragment(
            self.ctx,
            tiled.inner,
            self.asValue(input).inner,
            result_type,
            .int(self.ctx, .i32, operand_id),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub const CopyOptions = struct { pred: ?Value = null, tma_bar_ptr: ?Value = null, cache_policy: ?Value = null };

    pub fn copy(self: *Builder, atom: Atom, src: anytype, dst: anytype, opts: CopyOptions) void {
        var exec = atom;
        if (atom.type_().isA(cute_nvgpu.CopyAtomNonExecTiledTmaStoreType)) |ty| {
            const exec_type = (cute_nvgpu.CopyAtomTmaStoreType.get(self.ctx, .{
                .valType = ty.valType(),
                .copyBits = ty.copyBits(),
                .mode = .tiled,
                .gStride = self.emptyStrideType(),
                .tmaGbasis = ty.tmaGbasis(),
            }) catch @panic("invalid executable TMA store atom")).type_();
            exec = self.makeExecTma(atom, exec_type);
        }
        if (opts.tma_bar_ptr) |mbar| {
            if (atom.type_().isA(cute_nvgpu.CopyAtomNonExec2DGather4TmaLoadType)) |ty| {
                const exec_type = (cute_nvgpu.CopyAtomTmaLoadType.get(self.ctx, .{
                    .valType = ty.valType(),
                    .copyBits = ty.copyBits(),
                    .mode = .gather4,
                    .numCta = if (ty.kind() == .sm_100_2sm) 2 else 1,
                    .gStride = self.emptyStrideType(),
                    .tmaGbasis = ty.tmaGbasis(),
                }) catch @panic("invalid executable gather atom")).type_();
                exec = self.makeExecTma(atom, exec_type);
            } else if (atom.type_().isA(cute_nvgpu.CopyAtomNonExecTiledTmaLoadType)) |ty| {
                const exec_type = (cute_nvgpu.CopyAtomTmaLoadType.get(self.ctx, .{
                    .valType = ty.valType(),
                    .copyBits = ty.copyBits(),
                    .mode = .tiled,
                    .numCta = if (ty.kind() == .sm_100_2sm) 2 else 1,
                    .gStride = self.emptyStrideType(),
                    .tmaGbasis = ty.tmaGbasis(),
                }) catch @panic("invalid executable TMA atom")).type_();
                exec = self.makeExecTma(atom, exec_type);
            }
            exec = self.setTmaBarrier(exec, mbar);
        }
        if (opts.cache_policy) |policy| {
            const field = if (exec.type_().isA(cute_nvgpu.CopyAtomTmaLoadType) != null)
                (cute_nvgpu.AtomCopyFieldTmaLoadAttr.get(self.ctx, .{ .value = .cache_policy }) catch unreachable).attribute()
            else if (exec.type_().isA(cute_nvgpu.CopyAtomTmaStoreType) != null)
                (cute_nvgpu.AtomCopyFieldTmaStoreAttr.get(self.ctx, .{ .value = .cache_policy }) catch unreachable).attribute()
            else
                @panic("cache_policy requires an executable TMA copy atom");
            const updated = self.emit(cute_nvgpu.atom_set_value(self.ctx, exec.inner, policy.to(.i64).inner, exec.type_(), field, self.loc()));
            exec = .{ .inner = updated.inner, .kernel = self };
        }
        self.emitVoid(cute_ir.copy(self.ctx, exec.inner, self.innerSlice(src), self.innerSlice(dst), .{ .pred = if (opts.pred) |p| p.inner else null }, self.loc()));
    }

    pub fn gemm(self: *Builder, tiled: Atom, d: anytype, a: anytype, rhs: anytype, c: anytype) void {
        self.emitVoid(cute_ir.gemm(
            self.ctx,
            tiled.inner,
            self.asValue(d).inner,
            self.innerSlice(a),
            self.innerSlice(rhs),
            self.asValue(c).inner,
            self.loc(),
        ));
    }

    pub fn vectorExtract(self: *Builder, source: Value, lane: i64, dtype: DType) Value {
        return self.emit(vector.extract(
            self.ctx,
            source.inner,
            &.{lane},
            &.{},
            dtype.toMlir(self.ctx),
            self.loc(),
        ));
    }

    // ==================== cute_nvgpu.cpasync: TMA (SM90+) ====================
    // The host function builds tiled TMA atoms (Python `make_tiled_tma_atom`);
    // the kernel receives them as grid-constant arguments and makes them
    // executable before issuing copies.

    pub const TmaAtomOptions = struct { gmem_coord_tensor: ?View = null };

    fn inferValue(self: *Builder, name: []const u8, operands: []const *const mlir.Value, attributes: []const mlir.NamedAttribute) Value {
        return self.emit(mlir.Operation.make(self.ctx, name, .{
            .operands = .{ .flat = operands },
            .attributes = attributes,
            .result_type_inference = true,
            .location = self.loc(),
        }));
    }

    fn valueShape(self: *Builder, value: Value) Value {
        if (value.type_().isA(cute_ir.ShapeType) != null) return value;
        return self.inferValue("cute.get_shape", &.{value.inner}, &.{});
    }

    fn shapeRank(shape: Value) usize {
        return shape.type_().isA(cute_ir.ShapeType).?.attr().isA(cute_ir.ShapeAttr).?.rank();
    }

    /// Common CuTe TMA atom constructor. Gather uses the index tensor's layout
    /// at construction and passes data/index tensors together to partition/copy.
    pub fn makeTiledTmaAtom(self: *Builder, op: anytype, gmem: anytype, smem_layout: anytype, cta_tiler: anytype, config: TmaConfig, opts: TmaAtomOptions) TmaDescriptor {
        const Tiler = @TypeOf(cta_tiler);
        const tiler = if (Tiler == Value or Tiler == View or Tiler == Layout)
            self.asValue(cta_tiler)
        else if (@typeInfo(Tiler) == .pointer)
            self.makeShape(self.dimsExpr(cta_tiler)).value()
        else
            self.makeShape(cta_tiler).value();
        var smem = self.asValue(smem_layout);
        const tiler_rank = shapeRank(self.valueShape(tiler));
        const smem_rank = shapeRank(self.valueShape(smem));
        std.debug.assert(smem_rank == tiler_rank or smem_rank == tiler_rank + 1);
        if (smem_rank == tiler_rank + 1) {
            const modes = self.arena.allocator().alloc(i32, tiler_rank) catch @panic("OOM");
            for (modes, 0..) |*mode, i| mode.* = @intCast(i);
            smem = self.inferValue("cute.select", &.{smem.inner}, &.{.named(self.ctx, "mode", .denseArray(self.ctx, .i32, modes))});
        }
        const shape = self.valueShape(self.asValue(gmem));
        const identity = self.inferValue("cute.make_identity_layout", &.{shape.inner}, &.{});
        const cta_map = self.inferValue("cute.composition", &.{ identity.inner, tiler.inner }, &.{});
        var cfg = config;
        cfg.mode = .tiled;
        if (@TypeOf(op) == CopyBulkTensorTileS2GOp) return self.makeTiledTmaStoreAtom(gmem, smem, cta_map, cfg);
        cfg.cta_group = op.cta_group;
        if (@TypeOf(op) == CopyBulkTensorTileG2SOp) return self.makeTiledTmaLoadAtom(gmem, smem, cta_map, cfg);
        if (@TypeOf(op) != CopyBulkTensor2DGather4G2SOp) @compileError("unsupported TMA copy operation");
        cfg.mode = .gather4;
        const indices = opts.gmem_coord_tensor orelse @panic("gather4 requires gmem_coord_tensor");
        const index_type = indices.type_().isA(cute_ir.MemRefType) orelse @panic("gather4 indices must be a tensor");
        const layout = self.emit(cute_ir.get_layout(self.ctx, indices.inner, index_type.layout(), self.loc()));
        const kind: cute_nvgpu.GatherScatterTmaLoad = if (cfg.cta_group == .two) .sm_100_2sm else .sm_100;
        const results = self.emitResults(2, cute_nvgpu.atom_make_non_exec_2d_gather4_tma_load(self.ctx, self.asValue(gmem).inner, layout.inner, smem.inner, cta_map.inner, self.tmaLoadAtomType(cfg), self.coordTensorType(cfg.coordinate_rank, cfg.coordinate_layout), (cute_nvgpu.GatherScatterTmaLoadAttr.get(self.ctx, .{ .value = kind }) catch unreachable).attribute(), .{ .num_multicast = .int(self.ctx, .i32, 1), .tma_format = tmaFormatAttribute(self, cfg) }, self.loc()));
        return .{ .atom = .{ .inner = results[0].inner, .kernel = self }, .tensor = .{ .inner = results[1].inner, .kernel = self } };
    }

    /// Build the host-visible TMA descriptor and its coordinate tensor, whose
    /// types are assembled from `config`.
    pub fn makeTiledTmaLoadAtom(self: *Builder, gmem: anytype, smem_layout: anytype, cta_map: anytype, config: TmaConfig) TmaDescriptor {
        std.debug.assert(config.mode == .tiled);
        const results = self.emitResults(2, cute_nvgpu.atom_make_non_exec_tiled_tma_load(
            self.ctx,
            self.asValue(gmem).inner,
            self.asValue(smem_layout).inner,
            self.asValue(cta_map).inner,
            self.tmaLoadAtomType(config),
            self.coordTensorType(config.coordinate_rank, config.coordinate_layout),
            (cute_nvgpu.TiledTmaLoadAttr.get(self.ctx, .{ .value = tmaLoadKind(config) }) catch @panic("invalid tiled TMA load")).attribute(),
            .{ .num_multicast = .int(self.ctx, .i32, 1), .tma_format = tmaFormatAttribute(self, config) },
            self.loc(),
        ));
        return .{
            .atom = .{ .inner = results[0].inner, .kernel = self },
            .tensor = .{ .inner = results[1].inner, .kernel = self },
        };
    }

    /// Host-visible tiled TMA store atom and its coordinate tensor.
    pub fn makeTiledTmaStoreAtom(self: *Builder, gmem: anytype, smem_layout: anytype, cta_map: anytype, config: TmaConfig) TmaDescriptor {
        const results = self.emitResults(2, cute_nvgpu.atom_make_non_exec_tiled_tma_store(
            self.ctx,
            self.asValue(gmem).inner,
            self.asValue(smem_layout).inner,
            self.asValue(cta_map).inner,
            self.tmaStoreAtomType(config),
            self.coordTensorType(config.coordinate_rank, config.coordinate_layout),
            .{ .tma_format = tmaFormatAttribute(self, config) },
            self.loc(),
        ));
        return .{
            .atom = .{ .inner = results[0].inner, .kernel = self },
            .tensor = .{ .inner = results[1].inner, .kernel = self },
        };
    }

    /// Host-side tiled TMA load atom type described by `config`.
    pub fn tmaLoadAtomType(self: *Builder, config: TmaConfig) *const mlir.Type {
        if (config.mode == .gather4) return (cute_nvgpu.CopyAtomNonExec2DGather4TmaLoadType.get(self.ctx, .{
            .kind = if (config.cta_group == .two) .sm_100_2sm else .sm_100,
            .valType = config.dtype.toMlir(self.ctx),
            .copyBits = @intCast(config.copy_bits),
            .tmaGbasis = self.layoutType(config.global_basis),
            .tmaFormat = if (config.format == .default) null else tmaDataFormat(config.format, config.dtype),
        }) catch @panic("invalid gather TMA atom")).type_();
        return (cute_nvgpu.CopyAtomNonExecTiledTmaLoadType.get(self.ctx, .{
            .kind = tmaLoadKind(config),
            .valType = config.dtype.toMlir(self.ctx),
            .copyBits = @intCast(config.copy_bits),
            .tmaGbasis = self.layoutType(config.global_basis),
            .tmaFormat = tmaDataFormat(config.format, config.dtype),
        }) catch @panic("invalid TMA load atom")).type_();
    }

    /// Executable device-side form of a tiled TMA load atom. `make_exec_tma`
    /// preserves the tensor-map payload and adds the launch mode used by
    /// `copy`.
    pub fn tmaLoadExecAtomType(self: *Builder, config: TmaConfig) *const mlir.Type {
        return (cute_nvgpu.CopyAtomTmaLoadType.get(self.ctx, .{
            .valType = config.dtype.toMlir(self.ctx),
            .copyBits = @intCast(config.copy_bits),
            .mode = if (config.mode == .gather4) .gather4 else .tiled,
            .numCta = ctaCount(config.cta_group),
            .gStride = self.emptyStrideType(),
            .tmaGbasis = self.layoutType(config.global_basis),
        }) catch @panic("invalid TMA load atom")).type_();
    }

    /// Host-side tiled TMA store atom type described by `config`.
    pub fn tmaStoreAtomType(self: *Builder, config: TmaConfig) *const mlir.Type {
        return (cute_nvgpu.CopyAtomNonExecTiledTmaStoreType.get(self.ctx, .{
            .valType = config.dtype.toMlir(self.ctx),
            .copyBits = @intCast(config.copy_bits),
            .tmaGbasis = self.layoutType(config.global_basis),
            .tmaFormat = tmaDataFormat(config.format, config.dtype),
        }) catch @panic("invalid TMA store atom")).type_();
    }

    /// Executable device-side form of a tiled TMA store atom.
    pub fn tmaStoreExecAtomType(self: *Builder, config: TmaConfig) *const mlir.Type {
        return (cute_nvgpu.CopyAtomTmaStoreType.get(self.ctx, .{
            .valType = config.dtype.toMlir(self.ctx),
            .copyBits = @intCast(config.copy_bits),
            .mode = .tiled,
            .gStride = self.emptyStrideType(),
            .tmaGbasis = self.layoutType(config.global_basis),
        }) catch @panic("invalid TMA store atom")).type_();
    }

    /// The `()` global stride of tiled TMA atoms.
    fn emptyStrideType(self: *Builder) *const mlir.Type {
        return (cute_ir.StrideType.get(self.ctx, .{ .attr = self.algebraAttr(cute_ir.StrideAttr, .{ .tuple = &.{} }).attribute() }) catch @panic("invalid CuTe stride")).type_();
    }

    /// Generic address (i64) of the tensor map in an executable tiled TMA atom.
    pub fn tmaDescriptorAddress(self: *Builder, descriptor: Atom) Value {
        const ptr = self.emit(cute_nvgpu.get_tma_desc_addr(self.ctx, descriptor.inner, (cute_ir.PtrType.get(self.ctx, .{ .valueType = (cute_nvgpu.TmaDescriptorTiledType.get(self.ctx, .{}) catch unreachable).type_(), .addressSpace = .generic, .alignment = 64 }) catch unreachable).type_(), self.loc()));
        return self.ptrToInt(ptr, .i64);
    }

    pub fn prefetchTmaDesc(self: *Builder, descriptor: Atom) void {
        self.emitVoid(cute_nvgpu.prefetch_tma_desc(self.ctx, descriptor.inner, self.loc()));
    }

    pub fn makeExecTma(self: *Builder, descriptor: Atom, result_type: *const mlir.Type) Atom {
        const result = self.emit(cute_nvgpu.atom_make_exec_tma(self.ctx, descriptor.inner, result_type, .{}, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Attach the mbarrier that an executable TMA load atom completes.
    pub fn setTmaBarrier(self: *Builder, atom: Atom, mbar: Value) Atom {
        const result = self.emit(cute_nvgpu.atom_set_value(
            self.ctx,
            atom.inner,
            mbar.inner,
            atom.type_(),
            (cute_nvgpu.AtomCopyFieldTmaLoadAttr.get(self.ctx, .{ .value = .tma_bar }) catch @panic("invalid TMA load field")).attribute(),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Partition `smem` and `targets` for the TMA atom `descriptor`; the
    /// dependent result types are built by the caller.
    pub fn tmaPartition(
        self: *Builder,
        descriptor: Atom,
        cta_coord: anytype,
        cta_layout: anytype,
        smem: anytype,
        targets: anytype,
        smem_type: *const mlir.Type,
        target_types: []const *const mlir.Type,
    ) TmaPartition {
        if (targets.len != target_types.len) @panic("tmaPartition: target count does not match result type count");
        const views = self.arena.allocator().alloc(View, target_types.len) catch @panic("OOM");
        const op = cute_nvgpu.atom_tma_partition(
            self.ctx,
            descriptor.inner,
            self.makeCoord(cta_coord),
            self.asValue(cta_layout).inner,
            self.asValue(smem).inner,
            self.innerSlice(targets),
            smem_type,
            target_types,
            self.loc(),
        );
        _ = op.appendTo(self.currentBlock());
        for (views, 0..) |*view, i| view.* = .{ .inner = op.result(i + 1), .kernel = self };
        return .{ .smem = .{ .inner = op.result(0), .kernel = self }, .targets = views };
    }

    // ==================== cute_nvgpu.tcgen05 (SM100) ====================

    /// Tiled MMA type of the block-scaled atom described by `cfg`.
    pub fn blockScaledMmaType(self: *Builder, cfg: BlockScaledMmaConfig) *const mlir.Type {
        return (cute_ir.TiledMmaType.get(self.ctx, .{
            .mmaAtom = self.blockScaledAtomType(cfg),
            .atomLayoutMNK = self.layoutAttr(self.layoutSpec(.{ 1, 1, 1 }, .{ 0, 0, 0 })).attribute(),
            .permutationMNK = self.algebraAttr(cute_ir.TileAttr, self.expr(.{ AlgebraToken.all, AlgebraToken.all, AlgebraToken.all })).attribute(),
        }) catch @panic("invalid block-scaled tiled MMA")).type_();
    }

    fn blockScaledAtomType(self: *Builder, cfg: BlockScaledMmaConfig) *const mlir.Type {
        cfg.validate();
        return (cute_nvgpu.MmaAtomSM100UMMABlockScaledType.get(self.ctx, .{
            .shapeMnk = self.algebraAttr(cute_ir.ShapeAttr, self.expr(.{ cfg.m, cfg.n, cfg.k() })).attribute(),
            .numCta = ctaCount(cfg.cta_group),
            .aMajor = .k,
            .bMajor = .k,
            .aType = DType.f4e2m1fn.toMlir(self.ctx),
            .bType = cfg.bType().toMlir(self.ctx),
            .cType = DType.f32.toMlir(self.ctx),
            .sfType = cfg.scaleType().toMlir(self.ctx),
            .aFragKind = .smem_desc,
            .vecSize = cfg.vecSize(),
        }) catch @panic("invalid block-scaled MMA atom")).type_();
    }

    /// Block-scaled atom and its tiled MMA. Both scale-factor pointers start
    /// at `scale_ptr` and are patched after tensor-memory allocation.
    pub fn makeBlockScaledMma(self: *Builder, cfg: BlockScaledMmaConfig, scale_ptr: Value) BlockScaledMma {
        const disabled = self.constant(.i1, false);
        const atom = self.makeAtom(self.blockScaledAtomType(cfg), .{ disabled, disabled, disabled, scale_ptr, scale_ptr });
        const tiled = self.emit(cute_ir.make_tiled_mma(self.ctx, atom.inner, self.blockScaledMmaType(cfg), self.loc()));
        return .{ .atom = atom, .tiled = .{ .inner = tiled.inner, .kernel = self } };
    }

    /// Enable (or disable) accumulation into C for the next block-scaled MMA.
    pub fn setMmaAccumulate(self: *Builder, mma: Atom, enabled: Value) Atom {
        const result = self.emit(cute_nvgpu.atom_set_value(
            self.ctx,
            mma.inner,
            enabled.inner,
            mma.type_(),
            (cute_nvgpu.AtomMmaFieldSM100BlockScaledAttr.get(self.ctx, .{ .value = .accum_c }) catch @panic("invalid block-scaled MMA field")).attribute(),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub const Tcgen05CommitOptions = struct {
        /// Must match the group of the operations being tracked.
        cta_group: CtaGroup = .one,
        /// Arrive at the same barrier offset in each CTA whose rank bit is set.
        /// Without a mask, arrive only on the issuing CTA's barrier.
        multicast_mask: ?Value = null,
    };

    /// Track this thread's preceding tcgen05 MMA/copy operations of the
    /// selected group, signaling one arrival per destination on completion.
    /// The same elected thread that issues those operations must commit them.
    /// For `.two`, either CTA in the pair can be the issuer. All tcgen05
    /// operations in a kernel must use the same group.
    pub fn tcgen05Commit(self: *Builder, mbar: Value, opts: Tcgen05CommitOptions) void {
        self.emitVoid(nvvm.tcgen05_commit(
            self.ctx,
            self.sharedBarrierPtr(mbar).inner,
            opts.cta_group.toNvvm(),
            if (opts.multicast_mask) |mask| mask.to(.i16).inner else null,
            self.loc(),
        ));
    }

    /// Wait until asynchronous tensor-memory loads have reached registers.
    pub fn fenceTmemLoad(self: *Builder) void {
        self.emitVoid(nvvm.tcgen05_wait(self.ctx, .load, self.loc()));
    }

    pub const TmemOptions = struct { cta_group: CtaGroup = .one, exclusive: bool = false };

    pub fn allocTmem(self: *Builder, columns: u16, holding_ptr: anytype, opts: TmemOptions) void {
        self.emitVoid(cute_nvgpu.arch_sm100_alloc_tmem(
            self.ctx,
            self.constant(.i32, columns).inner,
            self.asValue(holding_ptr).inner,
            .{ .is_two_cta = if (opts.cta_group == .two) .unit(self.ctx) else null, .exclusive = if (opts.exclusive) .unit(self.ctx) else null },
            self.loc(),
        ));
    }

    pub fn retrieveTmemPtr(self: *Builder, holding_ptr: anytype, dtype: DType, alignment: u64) Value {
        const ptr_type = self.ptrTy(dtype, .tmem, alignment) catch @panic("invalid tensor-memory pointer type");
        return self.emit(cute_nvgpu.arch_sm100_retrieve_tmem_ptr(self.ctx, self.asValue(holding_ptr).inner, ptr_type, self.loc()));
    }

    pub fn relinquishTmemAllocPermit(self: *Builder, opts: TmemOptions) void {
        self.emitVoid(cute_nvgpu.arch_sm100_relinquish_tmem_alloc_permit(self.ctx, .{ .is_two_cta = if (opts.cta_group == .two) .unit(self.ctx) else null }, self.loc()));
    }

    pub fn deallocTmem(self: *Builder, pointer: anytype, columns: u16, opts: TmemOptions) void {
        self.emitVoid(cute_nvgpu.arch_sm100_dealloc_tmem(
            self.ctx,
            self.asValue(pointer).inner,
            self.constant(.i32, columns).inner,
            .{ .is_two_cta = if (opts.cta_group == .two) .unit(self.ctx) else null, .exclusive = if (opts.exclusive) .unit(self.ctx) else null },
            self.loc(),
        ));
    }

    pub const TmemLoadOptions = struct {
        num_dp: u16,
        num_b: u16,
        num_rep: u16,
        pack_16: bool = false,
    };

    pub fn tmemLoad(self: *Builder, src: anytype, result_type: *const mlir.Type, opts: TmemLoadOptions) View {
        const result = self.emit(cute_nvgpu.arch_copy_SM100_tmem_load(
            self.ctx,
            self.asValue(src).inner,
            result_type,
            .int(self.ctx, .i32, opts.num_dp),
            .int(self.ctx, .i32, opts.num_b),
            .int(self.ctx, .i32, opts.num_rep),
            .{ .pack_16 = if (opts.pack_16) .boolean(self.ctx, true) else null },
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Direct tensor-memory load into an SSA vector.  Higher-level tiled-copy
    /// helpers remain preferable when their dependent layouts are available.
    pub fn tmemLoadVector(self: *Builder, src: anytype, lanes: i64, opts: TmemLoadOptions) Value {
        // The NVVM-facing operation transports raw 32-bit register words.
        // Callers reinterpret each lane as the accumulator element type.
        const result_type = mlir.Type.vector(&.{lanes}, DType.i32.toMlir(self.ctx));
        return self.emit(cute_nvgpu.arch_copy_SM100_tmem_load(
            self.ctx,
            self.asValue(src).inner,
            result_type,
            .int(self.ctx, .i32, opts.num_dp),
            .int(self.ctx, .i32, opts.num_b),
            .int(self.ctx, .i32, opts.num_rep),
            .{ .pack_16 = if (opts.pack_16) .boolean(self.ctx, true) else null },
            self.loc(),
        ));
    }

    pub const S2tCopyConfig = struct {
        num_dp: i32,
        num_bit: i32,
        num_cta: i32 = 1,
        broadcast: cute_nvgpu.CopyS2TBroadcast = .none,
    };

    /// `!cute_nvgpu.atom.s2t_copy`, a `tcgen05.cp` atom moving `dtype`.
    pub fn s2tCopyAtomType(self: *Builder, dtype: DType, cfg: S2tCopyConfig) *const mlir.Type {
        return (cute_nvgpu.CopyAtomSM100CopyS2TType.get(self.ctx, .{
            .valType = dtype.toMlir(self.ctx),
            .numDp = cfg.num_dp,
            .numBit = cfg.num_bit,
            .numCta = cfg.num_cta,
            .broadcast = cfg.broadcast,
        }) catch @panic("invalid S2T copy atom")).type_();
    }

    pub fn makeS2tCopy(self: *Builder, atom: Atom, tmem: anytype, result_type: *const mlir.Type) Atom {
        const result = self.emit(cute_nvgpu.atom_make_s2t_copy(
            self.ctx,
            atom.inner,
            self.asValue(tmem).inner,
            result_type,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn makeTmemCopy(self: *Builder, atom: Atom, tmem: anytype, result_type: *const mlir.Type) Atom {
        const result = self.emit(cute_nvgpu.atom_make_tmem_copy(
            self.ctx,
            atom.inner,
            self.asValue(tmem).inner,
            result_type,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn s2tSmemDescriptor(self: *Builder, atom: Atom, input: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute_nvgpu.atom_get_copy_s2t_smem_desc_view(
            self.ctx,
            atom.inner,
            self.asValue(input).inner,
            result_type,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Raw SM100 TMA load. The descriptor address, destinations and
    /// coordinates come from the executable atom and partition operations.
    pub fn tmaLoad(
        self: *Builder,
        descriptor_addr: Value,
        data_addr: Value,
        barrier_addr: Value,
        coords: []const Value,
        multicast_mask: ?Value,
        offsets: []const Value,
        cache_policy: ?Value,
        mode: *const mlir.Attribute,
        num_cta: u8,
    ) void {
        self.emitVoid(cute_nvgpu.arch_copy_SM100_tma_load(
            self.ctx,
            descriptor_addr.inner,
            data_addr.inner,
            barrier_addr.inner,
            self.innerSlice(coords),
            self.innerSlice(offsets),
            mode,
            .int(self.ctx, .i32, num_cta),
            .{ .multicast_mask = if (multicast_mask) |v| v.inner else null, .cache_policy = if (cache_policy) |v| v.inner else null },
            self.loc(),
        ));
    }

    /// Raw SM100 TMA store.
    pub fn tmaStore(
        self: *Builder,
        descriptor_addr: Value,
        data_addr: Value,
        coords: []const Value,
        cache_policy: ?Value,
        mode: *const mlir.Attribute,
    ) void {
        self.emitVoid(cute_nvgpu.arch_copy_SM100_tma_store(
            self.ctx,
            descriptor_addr.inner,
            data_addr.inner,
            self.innerSlice(coords),
            mode,
            .{ .cache_policy = if (cache_policy) |v| v.inner else null },
            self.loc(),
        ));
    }

    // ==================== cute.arch ====================

    fn sreg(self: *Builder, comptime reg: []const u8) Index3 {
        var out: [3]Value = undefined;
        inline for (.{ "x", "y", "z" }, 0..) |axis, i| {
            out[i] = self.emit(nvvm.read_sreg(self.ctx, reg ++ "." ++ axis, self.loc()));
        }
        return .{ .x = out[0], .y = out[1], .z = out[2] };
    }

    pub fn threadIdx(self: *Builder) Index3 {
        return self.sreg("tid");
    }

    pub fn blockIdx(self: *Builder) Index3 {
        return self.sreg("ctaid");
    }

    pub fn blockDim(self: *Builder) Index3 {
        return self.sreg("ntid");
    }

    pub fn gridDim(self: *Builder) Index3 {
        return self.sreg("nctaid");
    }

    /// Mark a scalar as identical in every lane of its warp. CuTe uses this
    /// fact when lowering role branches that contain warp-collective TMA and
    /// tensor-memory operations.
    pub fn makeWarpUniform(self: *Builder, value: anytype) Value {
        const scalar = self.asValue(value);
        // The CuTe Python frontend legalizes predicates through an i32:
        //   extui i1 -> i32; make_warp_uniform i32; cmpi ne 0
        // `arch.make_warp_uniform` on i1 parses, but the native SM100
        // pipeline does not lower it all the way to NVVM.
        const uniform_input = if (scalar.dtype() == .i1) scalar.to(.i32) else scalar;
        const uniform = self.emit(cute_nvgpu.arch_make_warp_uniform(
            self.ctx,
            uniform_input.inner,
            uniform_input.type_(),
            self.loc(),
        ));
        return if (scalar.dtype() == .i1) uniform.ne(0) else uniform;
    }

    /// `cute.arch.sync_threads()`.
    pub fn syncThreads(self: *Builder) void {
        _ = nvvm.barrier(self.ctx, self.loc()).appendTo(self.currentBlock());
    }

    /// `bar.sync id, participants` (`barrier.cta.sync.aligned`).
    /// `id` is an integer or an i32 Value in [0, 15]. `participants` must be
    /// a positive warp multiple, and every participating thread must execute
    /// the same barrier id. The aligned form requires CTA-uniform control flow.
    pub fn barrierCtaSyncAligned(self: *Builder, id: anytype, participants: anytype) void {
        if (@TypeOf(participants) == ?Value) {
            if (participants) |count| return self.barrierCtaSyncAligned(id, count);
            return self.barrierCtaSyncAligned(id, null);
        }
        if (@TypeOf(participants) != Value and @TypeOf(participants) != @TypeOf(null)) {
            std.debug.assert(participants > 0 and participants <= 1024 and @mod(participants, 32) == 0);
        }
        if (@TypeOf(id) == Value) {
            std.debug.assert(id.dtype() == .i32);
        } else {
            const immediate: i32 = id;
            std.debug.assert(immediate >= 0 and immediate < 16);
        }
        // The LLVM intrinsic preserves convergence and both operands across
        // the embedded compiler's NVVM dialect version boundary.
        if (@TypeOf(participants) == @TypeOf(null)) {
            _ = self.callIntrinsic("llvm.nvvm.barrier.cta.sync.aligned", &.{self.lift(id).to(.i32)}, null);
        } else {
            _ = self.callIntrinsic("llvm.nvvm.barrier.cta.sync.aligned.count", &.{ self.lift(id).to(.i32), self.lift(participants).to(.i32) }, null);
        }
    }

    pub const BarrierOptions = struct { barrier_id: ?Value = null, number_of_threads: ?Value = null };

    pub fn barrier(self: *Builder, opts: BarrierOptions) void {
        self.barrierCtaSyncAligned(opts.barrier_id orelse self.constant(.i32, 0), opts.number_of_threads);
    }

    /// Build the CuTe first-class divisor used by CUTLASS's persistent tile
    /// scheduler.  Its arithmetic is lowered to the same multiply/shift
    /// sequence as Python CuTe DSL rather than an integer div/rem pair.
    pub fn fastDivmodCreate(self: *Builder, divisor_: anytype) Value {
        const divisor = self.lift(divisor_).to(.i32);
        return self.emit(cute_ir.fast_divmod_create_divisor(
            self.ctx,
            divisor.inner,
            (cute_ir.FastDivmodDivisorType.get(self.ctx, .{ .width = 32, .isPow2 = false }) catch @panic("invalid fast divmod divisor")).type_(),
            self.loc(),
        ));
    }

    /// Return `{ quotient, remainder }` for a CUTLASS FastDivmod divisor.
    pub fn fastDivmod(self: *Builder, dividend_: anytype, divisor: Value) [2]Value {
        const dividend = self.lift(dividend_).to(.i32);
        return self.emitResults(2, cute_ir.fast_divmod_compute(
            self.ctx,
            dividend.inner,
            divisor.inner,
            DType.i32.toMlir(self.ctx),
            DType.i32.toMlir(self.ctx),
            self.loc(),
        ));
    }

    /// Elect one lane from the active warp. Used by TMA issue, tensor-memory
    /// allocation, and tcgen05 commit operations.
    pub fn electSync(self: *Builder) Value {
        return self.emit(nvvm.elect_sync(self.ctx, self.loc()));
    }

    /// `cute.arch.shuffle_sync_bfly`: exchange a scalar with the lane at
    /// `lane_id ^ offset` in the current warp.
    /// Warp-wide maximum of a non-negative float, as one `redux.sync` instead
    /// of a shuffle reduction: IEEE bit patterns of non-negative floats order
    /// like unsigned integers.
    pub fn warpMaxNonNegative(self: *Builder, value: Value) Value {
        const dtype = value.dtype();
        if (dtypeBitwidth(dtype) != 32) @panic("warpMaxNonNegative requires a 32-bit scalar");
        const bits = if (dtype == .i32) value else value.bitCast(.i32);
        const reduced = self.emit(llvm.inline_asm(
            self.ctx,
            "redux.sync.max.u32 $0, $1, 0xffffffff;",
            &.{bits.inner},
            .{ .constraints = "=r,r", .result_type = .int(self.ctx, .i32) },
            self.loc(),
        ));
        return if (dtype == .i32) reduced else reduced.bitCast(dtype);
    }

    pub fn shuffleXor(self: *Builder, value: Value, offset: i32) Value {
        const dtype = value.dtype();
        if (dtypeBitwidth(dtype) != 32) @panic("shuffleXor currently requires a 32-bit scalar");
        const bits = if (dtype == .i32) value else value.bitCast(.i32);
        const shuffled = self.emit(llvm.inline_asm(
            self.ctx,
            "shfl.sync.bfly.b32 $0, $1, $2, 31, 0xffffffff;",
            &.{ bits.inner, self.constant(.i32, offset).inner },
            .{ .constraints = "=r,r,r", .result_type = .int(self.ctx, .i32) },
            self.loc(),
        ));
        return if (dtype == .i32) shuffled else shuffled.bitCast(dtype);
    }

    /// `cute.arch.shuffle_sync`: read a scalar from lane `lane` of the
    /// current warp. Every lane must participate.
    pub fn shuffleIdx(self: *Builder, value: Value, lane: anytype) Value {
        const dtype = value.dtype();
        if (dtypeBitwidth(dtype) != 32) @panic("shuffleIdx currently requires a 32-bit scalar");
        const bits = if (dtype == .i32) value else value.bitCast(.i32);
        const shuffled = self.emit(llvm.inline_asm(
            self.ctx,
            "shfl.sync.idx.b32 $0, $1, $2, 31, 0xffffffff;",
            &.{ bits.inner, self.constant(.i32, lane).inner },
            .{ .constraints = "=r,r,r", .result_type = .int(self.ctx, .i32) },
            self.loc(),
        ));
        return if (dtype == .i32) shuffled else shuffled.bitCast(dtype);
    }

    /// Four FP32 values rounded to FP8 E4M3 (saturating) and packed into one
    /// 32-bit word, `a` in the lowest byte.
    pub fn packFp8x4(self: *Builder, a: Value, b: Value, c: Value, d: Value) Value {
        return self.emit(llvm.inline_asm(
            self.ctx,
            "{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1; cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3; mov.b32 $0, {lo, hi}; }",
            &.{ a.to(.f32).inner, b.to(.f32).inner, c.to(.f32).inner, d.to(.f32).inner },
            .{ .constraints = "=r,f,f,f,f", .result_type = .int(self.ctx, .i32) },
            self.loc(),
        ));
    }

    /// Eight FP32 values rounded to FP4 E2M1 (saturating, nearest-even) and
    /// packed into one 32-bit word, `values[0]` in the lowest nibble: the
    /// packed NVFP4 storage order.
    pub fn packFp4x8(self: *Builder, values: [8]Value) Value {
        var operands: [8]*const mlir.Value = undefined;
        for (&operands, values) |*operand, value| operand.* = value.to(.f32).inner;
        return self.emit(llvm.inline_asm(
            self.ctx,
            "{ .reg .b8 b0, b1, b2, b3; cvt.rn.satfinite.e2m1x2.f32 b0, $2, $1; cvt.rn.satfinite.e2m1x2.f32 b1, $4, $3; " ++
                "cvt.rn.satfinite.e2m1x2.f32 b2, $6, $5; cvt.rn.satfinite.e2m1x2.f32 b3, $8, $7; mov.b32 $0, {b0, b1, b2, b3}; }",
            &operands,
            .{ .constraints = "=r,f,f,f,f,f,f,f,f", .result_type = .int(self.ctx, .i32) },
            self.loc(),
        ));
    }

    /// Two FP32 values rounded to FP4 E2M1 (saturating, nearest-even) in the
    /// low byte of a 32-bit word, `a` in the low nibble.
    pub fn packFp4x2(self: *Builder, a: Value, b: Value) Value {
        return self.emit(llvm.inline_asm(
            self.ctx,
            "{ .reg .b8 lo; cvt.rn.satfinite.e2m1x2.f32 lo, $2, $1; cvt.u32.u8 $0, lo; }",
            &.{ a.to(.f32).inner, b.to(.f32).inner },
            .{ .constraints = "=r,f,f", .result_type = .int(self.ctx, .i32) },
            self.loc(),
        ));
    }

    /// The FP8 E4M3 byte at the low end of `bits` (an i32), decoded to FP32.
    pub fn e4m3ToF32(self: *Builder, bits: Value) Value {
        return self.emit(llvm.inline_asm(
            self.ctx,
            "{ .reg .b16 lo, hl, hh; .reg .b32 pair; cvt.u16.u32 lo, $1; cvt.rn.f16x2.e4m3x2 pair, lo; mov.b32 {hl, hh}, pair; cvt.f32.f16 $0, hl; }",
            &.{bits.inner},
            .{ .constraints = "=f,r", .result_type = DType.f32.toMlir(self.ctx) },
            self.loc(),
        ));
    }

    /// One-operand FP32 PTX instruction, e.g. `ex2.approx.ftz.f32`.
    pub fn unaryF32(self: *Builder, comptime instruction: []const u8, value: Value) Value {
        return self.emit(llvm.inline_asm(
            self.ctx,
            instruction ++ " $0, $1;",
            &.{value.to(.f32).inner},
            .{ .constraints = "=f,f", .result_type = DType.f32.toMlir(self.ctx) },
            self.loc(),
        ));
    }

    /// Signal programmatic dependent launches after this grid has consumed
    /// its inputs. The caller must also launch the producer/consumer chain
    /// with PDL enabled.
    pub fn launchDependents(self: *Builder) void {
        _ = llvm.inline_asm(self.ctx, "griddepcontrol.launch_dependents;", &.{}, .{ .has_side_effects = true }, self.loc()).appendTo(self.currentBlock());
    }

    /// Wait until the preceding grid has made its programmatic dependency
    /// available. This is the consumer-side half of a PDL launch chain.
    pub fn waitForDependency(self: *Builder) void {
        _ = llvm.inline_asm(self.ctx, "griddepcontrol.wait;", &.{}, .{ .has_side_effects = true }, self.loc()).appendTo(self.currentBlock());
    }

    /// `cute.arch.alloc_smem(dtype, size_in_elems, alignment)`: a static
    /// shared-memory pointer, sized at compile time.
    pub fn allocSmem(self: *Builder, dt: DType, size_in_elems: i64, alignment: ?u64) Value {
        const al = alignment orelse @max(1, dtypeBitwidth(dt) / 8);
        const ty = self.ptrTy(dt, .smem, al) catch @panic("bad smem pointer");
        return self.emit(cute_nvgpu.arch_alloc_smem(self.ctx, ty, .int(self.ctx, .i32, size_in_elems), self.loc()));
    }

    // mbarriers, async-proxy fences and bulk-copy groups.

    /// mbarrier operands are `!llvm.ptr<3>` for NVVM. A barrier is either a CuTe
    /// shared-memory pointer or a raw `i32` shared-memory address.
    fn sharedBarrierPtr(self: *Builder, mbar: Value) Value {
        if (mbar.isInt()) return self.addressPtr(mbar, .shared);
        return self.emit(mlir.unrealized_conversion_cast(self.ctx, &.{mbar.inner}, &.{llvm.pointerType(self.ctx, NVPTXAddressSpace.shared)}, self.loc()));
    }

    /// Initialize a shared-memory mbarrier expecting `arrivals` per phase.
    pub fn mbarrierInit(self: *Builder, mbar: Value, arrivals: u32) void {
        self.emitVoid(nvvm.mbarrier_init(self.ctx, self.sharedBarrierPtr(mbar).inner, self.constant(.i32, arrivals).inner, self.loc()));
    }

    /// Arrive and expect `bytes` of asynchronous transactions (SM90+), as a
    /// TMA producer does before issuing its copies.
    pub fn mbarrierArriveExpectTx(self: *Builder, mbar: Value, bytes: u32) void {
        self.emitVoid(nvvm.mbarrier_arrive_expect_tx(self.ctx, self.sharedBarrierPtr(mbar).inner, self.constant(.i32, bytes).inner, .cta, self.loc()));
    }

    pub const MBarrierArriveOptions = struct {
        peer_cta_rank_in_cluster: ?Value = null,
        arrive_count: ?Value = null,
        scope: MBarrierScope = .cta,
        relaxed: bool = false,
    };

    /// Arrive locally or at the same offset in a peer CTA. Local arrivals
    /// return the opaque state token; remote arrivals have no state result.
    pub fn mbarrierArrive(self: *Builder, mbar: Value, opts: MBarrierArriveOptions) ?Value {
        var ptr = self.sharedBarrierPtr(mbar);
        if (opts.peer_cta_rank_in_cluster) |rank| {
            ptr = self.emit(nvvm.mapa(self.ctx, ptr.inner, rank.inner, llvm.pointerType(self.ctx, NVPTXAddressSpace.shared_cluster), self.loc()));
        }
        const op = nvvm.ops.mbarrier_arrive(self.ctx, ptr.inner, .{
            .count = if (opts.arrive_count) |count| count.to(.i32).inner else null,
            .scope = (nvvm.ops.MemScopeKindAttr.get(self.ctx, .{ .value = switch (opts.scope) {
                .cta => .cta,
                .cluster => .cluster,
            } }) catch unreachable).attribute(),
            .relaxed = .boolean(self.ctx, opts.relaxed),
            .res_type = if (opts.peer_cta_rank_in_cluster == null) mlir.Type.int(self.ctx, .i64) else null,
        }, self.loc());
        if (opts.peer_cta_rank_in_cluster == null) return self.emit(op);
        self.emitVoid(op);
        return null;
    }

    pub const MBarrierTryWaitOptions = struct {
        scope: MBarrierScope = .cta,
        time_limit: ?Value = null,
    };

    /// Try an acquire wait for phase `phase` (i32 parity, 0 or 1). May suspend
    /// briefly and return false before completion. `scope` selects which
    /// threads this wait can synchronize with; the barrier remains local.
    pub fn mbarrierTryWait(self: *Builder, mbar: Value, phase: Value, opts: MBarrierTryWaitOptions) Value {
        const ptr = self.sharedBarrierPtr(mbar);
        if (opts.time_limit) |ticks| {
            return self.emit(nvvm.mbarrier_try_wait_parity_timelimit(self.ctx, ptr.inner, phase.inner, ticks.to(.i32).inner, opts.scope, self.loc()));
        }
        return self.emit(nvvm.mbarrier_wait_parity(self.ctx, ptr.inner, phase.inner, .@"try", opts.scope, self.loc()));
    }

    pub const MBarrierWaitOptions = struct {
        scope: MBarrierScope = .cta,
        /// Hardware suspend hint in nanoseconds per attempt, not a total timeout.
        time_limit: u32 = 10_000_000,
        /// Probe before suspending, as in CuTe's pipeline consumer wait.
        test_first: bool = true,
    };

    fn mbarrierWaitLoop(self: *Builder, mbar: Value, phase: Value, opts: MBarrierWaitOptions) void {
        const ptr = self.sharedBarrierPtr(mbar);
        const ticks = self.constant(.i32, opts.time_limit);
        switch (opts.scope) {
            // Preserve the operation recognized by CuTe's pipeline compiler.
            .cta => self.emitVoid(nvvm.mbarrier_try_wait_parity(self.ctx, ptr.inner, phase.inner, ticks.inner, self.loc())),
            .cluster => {
                var pending = self.openWhile(.{}, .{});
                const ready = self.mbarrierTryWait(mbar, phase, .{ .scope = opts.scope, .time_limit = ticks });
                pending.yieldBefore(ready.eq(false), .{});
                pending.yieldAfter(.{});
            },
        }
    }

    /// Acquire a pipeline phase at `scope`: test without suspending, then
    /// block only when the phase is still outstanding. The initial test matters when
    /// a producer reaches a full ring and must yield to the consumer that
    /// recycles its next stage.
    pub fn mbarrierWait(self: *Builder, mbar: Value, phase: Value, opts: MBarrierWaitOptions) void {
        if (opts.test_first) {
            const ready = self.mbarrierTestWait(mbar, phase, .{ .scope = opts.scope });
            var pending = self.openIf(ready.eq(false));
            self.mbarrierWaitLoop(mbar, phase, opts);
            pending.close();
        } else {
            self.mbarrierWaitLoop(mbar, phase, opts);
        }
    }

    /// Make mbarrier initialization visible to the async proxy (SM90+).
    pub fn fenceMbarrierInit(self: *Builder) void {
        self.emitVoid(nvvm.fence_mbarrier_init(self.ctx, self.loc()));
    }

    /// Publish ordinary shared-memory writes to the asynchronous (TMA) proxy (SM90+).
    pub fn fenceProxyAsyncShared(self: *Builder) void {
        self.emitVoid(nvvm.fence_proxy(self.ctx, .@"async.shared", .cta, self.loc()));
    }

    /// Close the current group of bulk asynchronous copies, e.g. TMA stores (SM90+).
    pub fn cpAsyncBulkCommitGroup(self: *Builder) void {
        self.emitVoid(nvvm.cp_async_bulk_commit_group(self.ctx, self.loc()));
    }

    /// Wait until at most `pending` bulk copy groups are in flight; with
    /// `read`, only until their sources have been read (SM90+).
    pub fn cpAsyncBulkWaitGroup(self: *Builder, pending: u32, read: bool) void {
        self.emitVoid(nvvm.cp_async_bulk_wait_group(self.ctx, pending, read, self.loc()));
    }

    // ==================== raw-address SM90/SM100 primitives ====================
    // Hand-scheduled kernels address shared memory with `i32` offsets in the
    // shared window, global memory with `i64` addresses and tensor memory with
    // `i32` TMEM addresses (lane << 16 | column). These helpers emit the NVVM /
    // LLVM operations for them directly, using inline PTX where needed.

    pub const NVPTXAddressSpace = llvm.NVPTXAddressSpace;

    /// Typed LLVM pointer (`llvm.inttoptr`) for an integer address in `space`.
    pub fn addressPtr(self: *Builder, address: Value, comptime space: NVPTXAddressSpace) Value {
        return self.emit(llvm.inttoptr(self.ctx, address.inner, llvm.pointerType(self.ctx, space), self.loc()));
    }

    fn wordsType(self: *Builder, comptime n: usize) *const mlir.Type {
        const word = mlir.Type.int(self.ctx, .i32);
        return if (n == 1) word else mlir.Type.vector(&.{n}, word);
    }

    fn splitWords(self: *Builder, comptime n: usize, v: Value) [n]Value {
        var out: [n]Value = undefined;
        if (n == 1) {
            out[0] = v;
        } else {
            for (0..n) |i| out[i] = self.vectorExtract(v, @intCast(i), .i32);
        }
        return out;
    }

    fn joinWords(self: *Builder, comptime n: usize, words: [n]Value) Value {
        if (n == 1) return words[0];
        var inner: [n]*const mlir.Value = undefined;
        for (words, &inner) |w, *x| x.* = w.inner;
        return self.emit(vector.from_elements(self.ctx, &inner, self.wordsType(n), self.loc()));
    }

    pub const LoadOptions = struct {
        /// Read-only for the whole kernel (global: non-coherent `ld.global.nc`).
        invariant: bool = false,
    };

    /// Load `n` (1, 2 or 4) consecutive 32-bit words from a typed pointer.
    pub fn loadWordsPtr(self: *Builder, comptime n: usize, ptr: Value, opts: LoadOptions) [n]Value {
        const v = self.emit(llvm.load(self.ctx, ptr.inner, self.wordsType(n), .{ .alignment = 4 * n, .invariant = opts.invariant }, self.loc()));
        return self.splitWords(n, v);
    }

    /// Load `n` (1, 2 or 4) consecutive 32-bit words at an integer address.
    pub fn loadWords(self: *Builder, comptime n: usize, address: Value, comptime space: NVPTXAddressSpace, opts: LoadOptions) [n]Value {
        return self.loadWordsPtr(n, self.addressPtr(address, space), opts);
    }

    /// Store `n` (1, 2 or 4) consecutive 32-bit words through a typed pointer.
    pub fn storeWordsPtr(self: *Builder, comptime n: usize, ptr: Value, words: [n]Value) void {
        self.emitVoid(llvm.store(self.ctx, self.joinWords(n, words).inner, ptr.inner, .{ .alignment = 4 * n }, self.loc()));
    }

    /// Store `n` (1, 2 or 4) consecutive 32-bit words at an integer address.
    pub fn storeWords(self: *Builder, comptime n: usize, address: Value, comptime space: NVPTXAddressSpace, words: [n]Value) void {
        self.storeWordsPtr(n, self.addressPtr(address, space), words);
    }

    /// Store the low 16 bits of an `i32`.
    pub fn storeHalf(self: *Builder, address: Value, comptime space: NVPTXAddressSpace, word: Value) void {
        self.emitVoid(llvm.store(self.ctx, word.to(.i16).inner, self.addressPtr(address, space).inner, .{ .alignment = 2 }, self.loc()));
    }

    /// `bar.red.or.pred`: named barrier over `threads` threads that also returns whether
    /// `predicate` (i1) is true in any of them.
    pub fn barrierReduceOr(self: *Builder, id: i32, threads: i32, predicate: Value) Value {
        return self.callIntrinsic("llvm.nvvm.barrier.cta.red.or.aligned.count", &.{ self.constant(.i32, id), self.constant(.i32, threads), predicate }, mlir.Type.int(self.ctx, .i1)).?;
    }

    /// `bar.warp.sync 0xffffffff`.
    pub fn syncWarp(self: *Builder) void {
        self.emitVoid(nvvm.bar_warp_sync(self.ctx, self.constant(.i32, -1).inner, self.loc()));
    }

    /// Non-blocking acquire test of mbarrier phase `parity` (i32, 0 or 1).
    pub fn mbarrierTestWait(self: *Builder, mbar: Value, parity: Value, opts: MBarrierTryWaitOptions) Value {
        std.debug.assert(opts.time_limit == null);
        return self.emit(nvvm.mbarrier_wait_parity(self.ctx, self.sharedBarrierPtr(mbar).inner, parity.inner, .@"test", opts.scope, self.loc()));
    }

    /// Call an LLVM intrinsic by name (e.g. an `llvm.nvvm.*` intrinsic the NVVM dialect of
    /// the CuTe compiler does not model).
    pub fn callIntrinsic(self: *Builder, comptime name: []const u8, operands: []const Value, result: ?*const mlir.Type) ?Value {
        const alloc = self.arena.allocator();
        const inner = alloc.alloc(*const mlir.Value, operands.len) catch @panic("OOM");
        for (operands, inner) |o, *x| x.* = o.inner;
        const op = llvm.call_intrinsic(self.ctx, name, inner, .{ .result_type = result }, self.loc());
        if (result == null) {
            self.emitVoid(op);
            return null;
        }
        return self.emit(op);
    }

    /// 16-byte `cp.async.cg` global -> shared. With `valid` (i1) false, the destination is
    /// zero-filled and `src` is not read (src-size 0).
    pub fn cpAsync16(self: *Builder, dst: Value, src: Value, valid: ?Value) void {
        const d = self.addressPtr(dst, .shared);
        const g = self.addressPtr(src, .global);
        if (valid) |v| {
            _ = self.callIntrinsic("llvm.nvvm.cp.async.cg.shared.global.16.s", &.{ d, g, self.select(v, self.constant(.i32, 16), self.constant(.i32, 0)) }, null);
        } else {
            _ = self.callIntrinsic("llvm.nvvm.cp.async.cg.shared.global.16", &.{ d, g }, null);
        }
    }

    /// Arrive on `barrier` once this thread's prior cp.async copies have landed; with
    /// `noinc`, the arrival is counted in the barrier's expected count.
    pub fn cpAsyncMbarrierArrive(self: *Builder, mbar: Value, comptime noinc: bool) void {
        _ = self.callIntrinsic(if (noinc) "llvm.nvvm.cp.async.mbarrier.arrive.noinc.shared" else "llvm.nvvm.cp.async.mbarrier.arrive.shared", &.{self.sharedBarrierPtr(mbar)}, null);
    }

    /// Wait until all of this thread's cp.async copies have landed.
    pub fn cpAsyncWaitAll(self: *Builder) void {
        self.emitVoid(nvvm.cp_async_commit_group(self.ctx, self.loc()));
        self.emitVoid(nvvm.cp_async_wait_group(self.ctx, 0, self.loc()));
    }

    /// Prefetch the global line of `address` (i64) into L2 with evict-last priority.
    pub fn prefetchL2EvictLast(self: *Builder, address: Value) void {
        _ = self.callIntrinsic("llvm.nvvm.prefetch.global.L2.evict.last", &.{self.addressPtr(address, .global)}, null);
    }

    /// Warp-wide TMEM allocation of `columns`; the base address is written to `holder` (i32 smem).
    pub fn tmemAllocAt(self: *Builder, holder: Value, columns: u32) void {
        self.emitVoid(nvvm.tcgen05_alloc(self.ctx, self.addressPtr(holder, .shared).inner, self.constant(.i32, columns).inner, self.loc()));
    }

    pub fn tmemDeallocAt(self: *Builder, taddr: Value, columns: u32) void {
        self.emitVoid(nvvm.tcgen05_dealloc(self.ctx, self.addressPtr(taddr, .tensor).inner, self.constant(.i32, columns).inner, self.loc()));
    }

    pub const Tcgen05Fence = nvvm.Tcgen05FenceKind;

    /// `tcgen05.fence::{before,after}_thread_sync`.
    pub fn tcgen05Fence(self: *Builder, comptime kind: Tcgen05Fence) void {
        self.emitVoid(nvvm.tcgen05_fence(self.ctx, kind, self.loc()));
    }

    /// `tcgen05.wait::st`.
    pub fn fenceTmemStore(self: *Builder) void {
        self.emitVoid(nvvm.tcgen05_wait(self.ctx, .store, self.loc()));
    }

    /// `tcgen05.mma.cta_group::1.kind::f16`: D (TMEM at `d`) = A . B (+ D when
    /// `accumulate`, i1), with A and B given by shared-memory descriptors (i64) and the
    /// instruction descriptor `idesc` (i32).
    pub fn tcgen05MmaF16(self: *Builder, d: Value, adesc: Value, bdesc: Value, idesc: Value, accumulate: Value) void {
        self.emitVoid(nvvm.tcgen05_mma(self.ctx, self.addressPtr(d, .tensor).inner, adesc.inner, bdesc.inner, idesc.inner, accumulate.inner, .f16, .cta_1, self.loc()));
    }

    /// `tcgen05.ld.32x32b.x{n}`: thread i of the warp gets TMEM lane (warp base + i),
    /// `n` consecutive 32-bit columns from `taddr`.
    pub fn tmemLoad32x32b(self: *Builder, comptime n: usize, taddr: Value) [n]Value {
        const v = self.emit(nvvm.tcgen05_ld(self.ctx, self.addressPtr(taddr, .tensor).inner, self.wordsType(n), .shape_32x32b, self.loc()));
        return self.splitWords(n, v);
    }

    /// `tcgen05.st.32x32b.x{n}` (see `tmemLoad32x32b`).
    pub fn tmemStore32x32b(self: *Builder, comptime n: usize, taddr: Value, words: [n]Value) void {
        self.emitVoid(nvvm.tcgen05_st(self.ctx, self.addressPtr(taddr, .tensor).inner, self.joinWords(n, words).inner, .shape_32x32b, self.loc()));
    }

    /// Rank of this CTA in its cluster.
    pub fn clusterCtaRank(self: *Builder) Value {
        return self.emit(nvvm.read_sreg(self.ctx, "cluster.ctarank", self.loc()));
    }

    /// The same shared-memory location (i32 address) in CTA `rank` of the cluster, as a
    /// `shared::cluster` pointer.
    pub fn mapaShared(self: *Builder, address: Value, rank: Value) Value {
        return self.emit(nvvm.mapa(self.ctx, self.addressPtr(address, .shared).inner, rank.inner, llvm.pointerType(self.ctx, NVPTXAddressSpace.shared_cluster), self.loc()));
    }

    /// Full cluster barrier (arrive.release + wait.acquire, aligned).
    pub fn clusterSync(self: *Builder) void {
        self.emitVoid(nvvm.cluster_arrive(self.ctx, true, self.loc()));
        self.emitVoid(nvvm.cluster_wait(self.ctx, true, self.loc()));
    }

    /// Warp-wide maximum of an f32 (`redux.sync.max.f32`, SM100).
    pub fn warpMaxF32(self: *Builder, value: Value) Value {
        return self.callIntrinsic("llvm.nvvm.redux.sync.fmax", &.{ value, self.constant(.i32, -1) }, mlir.Type.float(self.ctx, .f32)).?;
    }

    /// Fast approximate 2^x (`ex2.approx.ftz.f32`).
    pub fn exp2Approx(self: *Builder, value: Value) Value {
        return self.emit(nvvm.ex2(self.ctx, value.inner, true, self.loc()));
    }

    /// Fast approximate 1/x (`rcp.approx.ftz.f32`).
    pub fn rcpApprox(self: *Builder, value: Value) Value {
        return self.emit(nvvm.rcp_approx_ftz(self.ctx, value.inner, self.loc()));
    }

    /// `vector<2 x dt>`, e.g. a bf16x2 pair packed in 32 bits.
    fn pairType(self: *Builder, comptime dt: DType) *const mlir.Type {
        return mlir.Type.vector(&.{2}, dt.toMlir(self.ctx));
    }

    fn bitcastTo(self: *Builder, value: Value, ty: *const mlir.Type) Value {
        return self.emit(llvm.bitcast(self.ctx, value.inner, ty, self.loc()));
    }

    /// Two FP8 e4m3 (low 16 bits of `pair`, i32) -> bf16x2 packed in an i32.
    pub fn e4m3x2ToBf16x2(self: *Builder, pair: Value) Value {
        const v = self.callIntrinsic("llvm.nvvm.e4m3x2.to.bf16x2.rn", &.{pair.to(.i16)}, self.pairType(.bf16)).?;
        return self.bitcastTo(v, mlir.Type.int(self.ctx, .i32));
    }

    /// Two FP4 e2m1 (the low byte of `pair`, the even element in the low nibble)
    /// -> bf16x2 packed in an i32.
    pub fn e2m1x2ToBf16x2(self: *Builder, pair: Value) Value {
        const v = self.callIntrinsic("llvm.nvvm.e2m1x2.to.bf16x2.rn", &.{pair.to(.i16)}, self.pairType(.bf16)).?;
        return self.bitcastTo(v, mlir.Type.int(self.ctx, .i32));
    }

    /// Eight FP4 e2m1 (one word, even elements in the low nibbles) -> four bf16x2 packed
    /// in i32s. The pairs are read as the low bytes of the 16-bit halves of `word` and
    /// `word >> 8`, which ptxas maps to operand selectors (one shift per word).
    pub fn e2m1x8ToBf16x2x4(self: *Builder, word: Value) [4]Value {
        const halves = self.pairType(.i16);
        const even = self.bitcastTo(word, halves);
        const odd = self.bitcastTo(word.shrLogical(8), halves);
        const pairs = [4]Value{
            self.vectorExtract(even, 0, .i16), self.vectorExtract(odd, 0, .i16),
            self.vectorExtract(even, 1, .i16), self.vectorExtract(odd, 1, .i16),
        };
        var out: [4]Value = undefined;
        for (pairs, &out) |pair, *o| o.* = self.e2m1x2ToBf16x2(pair);
        return out;
    }

    /// bf16x2 * bf16x2 (round to nearest), both packed in i32.
    pub fn mulBf16x2(self: *Builder, a: Value, b: Value) Value {
        const ty = self.pairType(.bf16);
        const p = self.emit(llvm.fmul(self.ctx, self.bitcastTo(a, ty).inner, self.bitcastTo(b, ty).inner, ty, self.loc()));
        return self.bitcastTo(p, mlir.Type.int(self.ctx, .i32));
    }

    /// Two f32 -> a packed pair of `dt` (bf16 or f16) in an i32, `lo` in the low half.
    pub fn packHalf2(self: *Builder, lo: Value, hi: Value, comptime dt: DType) Value {
        const pair = self.emit(vector.from_elements(self.ctx, &.{ lo.to(dt).inner, hi.to(dt).inner }, self.pairType(dt), self.loc()));
        return self.bitcastTo(pair, mlir.Type.int(self.ctx, .i32));
    }

    /// A packed pair of `dt` (bf16 or f16) in an i32 -> two f32 (low half first).
    pub fn unpackHalf2(self: *Builder, word: Value, comptime dt: DType) [2]Value {
        const pair = self.bitcastTo(word, self.pairType(dt));
        return .{ self.vectorExtract(pair, 0, dt).to(.f32), self.vectorExtract(pair, 1, dt).to(.f32) };
    }

    /// f32 -> bf16 bits (round to nearest even) in the low 16 bits of an i32.
    pub fn f32ToBf16Bits(self: *Builder, value: Value) Value {
        return self.emit(arith.extui(self.ctx, value.to(.bf16).bitCast(.i16).inner, mlir.Type.int(self.ctx, .i32), self.loc()));
    }

    /// An i32 equal to `value` at run time that the compiler cannot constant-fold
    /// (scaled by `%nctaid.z`, which is 1 for 2-D grids): e.g. a loop bound that must
    /// keep a loop rolled (small code, fewer instruction-cache misses).
    pub fn opaqueI32(self: *Builder, value: i32) Value {
        return self.gridDim().z.mul(value);
    }

    /// SM100 `tile::gather4` TMA load from a tiled 2-D descriptor address `desc` (i64).
    /// Gather four rows `r` (i32) at column `col` into 128-byte-aligned shared `dst` (i32),
    /// completing their bytes on an 8-byte-aligned local mbarrier `bar` (i32).
    /// The descriptor must have row box extent 1 and no interleaving. `col` and `r`
    /// are element coordinates; row length and out-of-bounds fill come from the
    /// descriptor (zero or OOB-NaN). Arm the barrier for all four rows' bytes before waiting.
    pub fn tmaGather4(self: *Builder, desc: Value, dst: Value, bar: Value, col: Value, r: [4]Value, policy: CacheHintSm100) void {
        std.debug.assert(desc.dtype() == .i64 and dst.dtype() == .i32 and bar.dtype() == .i32 and col.dtype() == .i32);
        for (r) |row| std.debug.assert(row.dtype() == .i32);
        _ = llvm.inline_asm(self.ctx, "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint [$0], [$1, {$2, $3, $4, $5, $6}], [$7], $8;", &.{ dst.inner, desc.inner, col.inner, r[0].inner, r[1].inner, r[2].inner, r[3].inner, bar.inner, self.constant(.i64, policy).inner }, .{ .constraints = "r,l,r,r,r,r,r,r,l,~{memory}", .has_side_effects = true }, self.loc()).appendTo(self.currentBlock());
    }

    pub const BulkPrefetchOptions = struct { l2_cache_hint: ?Value = null };

    /// Best-effort byte-range prefetch into L2; size must be a multiple of 16.
    pub fn cpAsyncBulkPrefetch(self: *Builder, addr: Value, bytes: anytype, opts: BulkPrefetchOptions) void {
        if (@TypeOf(bytes) != Value) std.debug.assert(bytes > 0 and bytes % 16 == 0);
        const address = if (addr.isInt()) self.addressPtr(addr, .global) else self.llvmPointer(addr, .global);
        self.emitVoid(nvvm.ops.cp_async_bulk_prefetch(self.ctx, address.inner, self.lift(bytes).to(.i32).inner, .{
            .l2CacheHint = if (opts.l2_cache_hint) |hint| hint.to(.i64).inner else null,
        }, self.loc()));
    }

    pub fn prefetchL1(self: *Builder, addr: Value) void {
        const address = if (addr.isInt()) addr else self.ptrToInt(addr, .i64);
        _ = llvm.inline_asm(self.ctx, "prefetch.global.L1 [$0];", &.{address.inner}, .{ .constraints = "l", .has_side_effects = true }, self.loc()).appendTo(self.currentBlock());
    }

    pub fn prefetchTensorMap(self: *Builder, desc: Value) void {
        const address = if (desc.isInt()) desc else self.ptrToInt(desc, .i64);
        _ = llvm.inline_asm(self.ctx, "prefetch.tensormap [$0];", &.{address.inner}, .{ .constraints = "l", .has_side_effects = true }, self.loc()).appendTo(self.currentBlock());
    }

    pub const OperandSource = union(enum) { smem: Value, tmem: Value };
    pub const Tcgen05MmaWsOptions = struct {
        collector_b_buffer: ?nvvm.ops.Tcgen05MMACollectorBBuffer = null,
        collector_op: ?nvvm.ops.Tcgen05MMACollectorOp = null,
        col_b_zero_mask: ?Value = null,
    };

    /// Single-thread, asynchronous weight-stationary MMA, as in Python primitives.
    /// The caller elects the issuer and commits/waits before consuming D.
    pub fn tcgen05MmaWs(self: *Builder, kind: nvvm.Tcgen05MmaKind, d: Value, a: OperandSource, bdesc: Value, idesc: anytype, enable_input_d: Value, opts: Tcgen05MmaWsOptions) void {
        std.debug.assert(kind == .f16 or kind == .tf32 or kind == .f8f6f4 or kind == .i8);
        const adesc = switch (a) {
            .smem => |desc| self.smemDescriptorToInt(desc),
            .tmem => |ptr| if (ptr.isInt()) self.addressPtr(ptr, .tensor) else self.llvmPointer(ptr, .tensor),
        };
        const dest = if (d.isInt()) self.addressPtr(d, .tensor) else self.llvmPointer(d, .tensor);
        self.emitVoid(nvvm.ops.tcgen05_mma_ws(self.ctx, dest.inner, adesc.inner, self.smemDescriptorToInt(bdesc).inner, self.lift(idesc).to(.i32).inner, enable_input_d.to(.i1).inner, (nvvm.ops.Tcgen05MMAKindAttr.get(self.ctx, .{ .value = kind }) catch unreachable).attribute(), .{
            .collectorBBuffer = if (opts.collector_b_buffer) |v| (nvvm.ops.Tcgen05MMACollectorBBufferAttr.get(self.ctx, .{ .value = v }) catch unreachable).attribute() else null,
            .collectorOp = if (opts.collector_op) |v| (nvvm.ops.Tcgen05MMACollectorOpAttr.get(self.ctx, .{ .value = v }) catch unreachable).attribute() else null,
            .zeroColMask = if (opts.col_b_zero_mask) |v| v.to(.i64).inner else null,
        }, self.loc()));
    }

    pub fn makeUmmaSmemDesc(self: *Builder, src: Value, layout: LayoutSpec, major: OperandMajorMode, next_src: ?Value) Value {
        return self.emit(cute_nvgpu.make_umma_smem_desc(self.ctx, src.inner, self.smemDescType(), self.layoutAttr(layout).attribute(), (cute_nvgpu.MajorModeAttr.get(self.ctx, .{ .value = switch (major) {
            .k => .k,
            .mn => .mn,
        } }) catch unreachable).attribute(), .{ .nextSrc = if (next_src) |v| v.inner else null }, self.loc()));
    }

    pub fn intToSmemDescriptor(self: *Builder, value: anytype) Value {
        return self.emit(mlir.unrealized_conversion_cast(self.ctx, &.{self.lift(value).to(.i64).inner}, &.{self.smemDescType()}, self.loc()));
    }

    pub fn smemDescriptorToInt(self: *Builder, desc: Value) Value {
        if (desc.isInt()) return desc.to(.i64);
        return self.emit(mlir.unrealized_conversion_cast(self.ctx, &.{desc.inner}, &.{mlir.Type.int(self.ctx, .i64)}, self.loc()));
    }

    pub fn setMaxRegisterIncrease(self: *Builder, comptime count: u32) void {
        self.setMaxRegister(count, .increase);
    }

    pub fn setMaxRegisterDecrease(self: *Builder, comptime count: u32) void {
        self.setMaxRegister(count, .decrease);
    }

    fn setMaxRegister(self: *Builder, comptime count: u32, comptime action: enum { increase, decrease }) void {
        comptime std.debug.assert(count >= 24 and count <= 256 and count % 8 == 0);
        const text = std.fmt.comptimePrint("setmaxnreg.{s}.sync.aligned.u32 {d};", .{ if (action == .increase) "inc" else "dec", count });
        _ = llvm.inline_asm(self.ctx, text, &.{}, .{ .has_side_effects = true }, self.loc()).appendTo(self.currentBlock());
    }

    /// Specialized busy wait for hand-scheduled kernels; never hardware-suspends.
    pub fn mbarrierSpin(self: *Builder, bar: Value, parity: Value, scope: MBarrierScope) void {
        std.debug.assert(bar.dtype() == .i32 and parity.dtype() == .i32);
        _ = llvm.inline_asm(self.ctx, switch (scope) {
            .cta => "{ .reg .pred p; spin${:uid}: mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 p, [$0], $1; @!p bra spin${:uid}; }",
            .cluster => "{ .reg .pred p; spin${:uid}: mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [$0], $1; @!p bra spin${:uid}; }",
        }, &.{ bar.inner, parity.inner }, .{ .constraints = "r,r,~{memory}", .has_side_effects = true }, self.loc()).appendTo(self.currentBlock());
    }

    /// Convert a CuTe or LLVM pointer to the LLVM pointer required by NVVM.
    fn llvmPointer(self: *Builder, ptr: Value, comptime space: NVPTXAddressSpace) Value {
        if (!ptr.isPtr()) return ptr;
        return self.emit(mlir.unrealized_conversion_cast(self.ctx, &.{ptr.inner}, &.{llvm.pointerType(self.ctx, space)}, self.loc()));
    }

    /// Map a local shared pointer/address to a typed pointer in peer CTA memory.
    pub fn mapDsmemPtr(self: *Builder, smem: Value, peer_cta_rank: Value) Value {
        const ptr = if (smem.isInt()) self.addressPtr(smem, .shared) else self.llvmPointer(smem, .shared);
        const remote = self.emit(nvvm.mapa(self.ctx, ptr.inner, peer_cta_rank.to(.i32).inner, llvm.pointerType(self.ctx, NVPTXAddressSpace.shared_cluster), self.loc()));
        if (smem.type_().isA(cute_ir.PtrType)) |ty| {
            std.debug.assert(ty.addressSpace() == .smem);
            const result_type = (cute_ir.PtrType.get(self.ctx, .{
                .valueType = ty.valueType(),
                .addressSpace = .dsmem,
                .alignment = ty.alignment(),
                .swizzle = ty.swizzle(),
                .bitlayout = ty.bitlayout(),
            }) catch @panic("invalid DSMEM pointer")).type_();
            return self.emit(cute_ir.inttoptr(self.ctx, self.ptrToInt(remote, .i32).inner, result_type, self.loc()));
        }
        return remote;
    }

    /// Store one, two, or four i32 words in a peer CTA, completing bytes on its
    /// mbarrier. Both inputs name local shared memory; both are mapped to the peer.
    pub fn storeAsyncDsmem(self: *Builder, smem: Value, value: anytype, mbar: Value, peer_cta_rank: Value) void {
        const n = if (@TypeOf(value) == Value) 1 else common.arity(@TypeOf(value), "storeAsyncDsmem: value");
        comptime std.debug.assert(n == 1 or n == 2 or n == 4);
        if (smem.type_().isA(cute_ir.PtrType)) |ty| {
            std.debug.assert(ty.addressSpace() == .smem and ty.alignment() % (n * 4) == 0);
        }
        if (mbar.type_().isA(cute_ir.PtrType)) |ty| {
            std.debug.assert(ty.addressSpace() == .smem and ty.alignment() % 8 == 0);
        }
        const packed_value = if (@TypeOf(value) == Value) value else blk: {
            var words: [n]Value = undefined;
            inline for (0..n) |i| words[i] = self.lift(value[i]).to(.i32);
            break :blk self.joinWords(n, words);
        };
        self.emitVoid(nvvm.ops.store_async(self.ctx, self.llvmPointer(self.mapDsmemPtr(smem, peer_cta_rank), .shared_cluster).inner, packed_value.inner, self.llvmPointer(self.mapDsmemPtr(mbar, peer_cta_rank), .shared_cluster).inner, self.loc()));
    }

    /// Explicit aligned variants retain the convergence requirement of raw PTX.
    pub fn barrierClusterArriveRelaxedAligned(self: *Builder) void {
        _ = llvm.inline_asm(self.ctx, "barrier.cluster.arrive.relaxed.aligned;", &.{}, .{ .constraints = "~{memory}", .has_side_effects = true }, self.loc()).appendTo(self.currentBlock());
    }

    pub fn barrierClusterWaitAligned(self: *Builder) void {
        self.emitVoid(nvvm.cluster_wait(self.ctx, true, self.loc()));
    }

    pub const ClusterArriveOptions = struct { aligned: bool = false };

    pub fn clusterArriveRelaxed(self: *Builder, opts: ClusterArriveOptions) void {
        if (opts.aligned) return self.barrierClusterArriveRelaxedAligned();
        _ = llvm.inline_asm(self.ctx, "barrier.cluster.arrive.relaxed;", &.{}, .{ .constraints = "~{memory}", .has_side_effects = true }, self.loc()).appendTo(self.currentBlock());
    }

    pub fn clusterWait(self: *Builder) void {
        self.emitVoid(nvvm.cluster_wait(self.ctx, false, self.loc()));
    }

    /// A null mask selects CuTe's FULL_MASK; integers and SSA masks also work.
    pub fn voteAnySync(self: *Builder, pred: Value, mask: anytype) Value {
        const active = if (@TypeOf(mask) == @TypeOf(null)) self.constant(.i32, -1) else self.lift(mask).to(.i32);
        return self.callIntrinsic("llvm.nvvm.vote.any.sync", &.{ active, pred }, mlir.Type.int(self.ctx, .i1)).?;
    }

    /// A null mask selects CuTe's FULL_MASK.
    pub fn voteBallotSync(self: *Builder, pred: Value, mask: anytype) Value {
        const active = if (@TypeOf(mask) == @TypeOf(null)) self.constant(.i32, -1) else self.lift(mask).to(.i32);
        return self.callIntrinsic("llvm.nvvm.vote.ballot.sync", &.{ active, pred }, mlir.Type.int(self.ctx, .i32)).?;
    }

    pub const FmaxOptions = struct { abs: bool = false, nan: bool = false, ftz: bool = false };

    pub fn fmax(self: *Builder, x: Value, y: Value, opts: FmaxOptions) Value {
        if (!opts.abs and !opts.nan and !opts.ftz) return self.callIntrinsic("llvm.nvvm.fmax.f", &.{ x, y }, mlir.Type.float(self.ctx, .f32)).?;
        return self.emit(nvvm.ops.fmax(self.ctx, x.inner, y.inner, mlir.Type.float(self.ctx, .f32), .{
            .abs = if (opts.abs) .unit(self.ctx) else null,
            .nan = if (opts.nan) .unit(self.ctx) else null,
            .ftz = if (opts.ftz) .unit(self.ctx) else null,
        }, self.loc()));
    }

    fn vec2(self: *Builder, x: [2]Value) Value {
        const ty = mlir.Type.vector(&.{2}, mlir.Type.float(self.ctx, .f32));
        return self.emit(vector.from_elements(self.ctx, &.{ x[0].inner, x[1].inner }, ty, self.loc()));
    }

    fn unvec2(self: *Builder, v: Value) [2]Value {
        return .{ self.vectorExtract(v, 0, .f32), self.vectorExtract(v, 1, .f32) };
    }

    pub const PackedF32x2Options = struct { rnd: nvvm.ops.FPRoundingMode = .rn, ftz: bool = false };

    fn packedF32x2Attrs(self: *Builder, opts: PackedF32x2Options) struct { rnd: ?*const mlir.Attribute, ftz: ?*const mlir.Attribute } {
        return .{
            .rnd = (nvvm.ops.FPRoundingModeAttr.get(self.ctx, .{ .value = opts.rnd }) catch unreachable).attribute(),
            .ftz = if (opts.ftz) .unit(self.ctx) else null,
        };
    }

    pub fn fmaPackedF32x2(self: *Builder, a: [2]Value, b: [2]Value, c: [2]Value, opts: PackedF32x2Options) [2]Value {
        return self.unvec2(self.emit(nvvm.ops.fma_packed_f32x2(self.ctx, self.vec2(a).inner, self.vec2(b).inner, self.vec2(c).inner, mlir.Type.vector(&.{2}, mlir.Type.float(self.ctx, .f32)), .{ .rnd = self.packedF32x2Attrs(opts).rnd, .ftz = self.packedF32x2Attrs(opts).ftz }, self.loc())));
    }

    pub fn addPackedF32x2(self: *Builder, a: [2]Value, b: [2]Value, opts: PackedF32x2Options) [2]Value {
        return self.unvec2(self.emit(nvvm.ops.add_packed_f32x2(self.ctx, self.vec2(a).inner, self.vec2(b).inner, mlir.Type.vector(&.{2}, mlir.Type.float(self.ctx, .f32)), .{ .rnd = self.packedF32x2Attrs(opts).rnd, .ftz = self.packedF32x2Attrs(opts).ftz }, self.loc())));
    }

    pub fn mulPackedF32x2(self: *Builder, a: [2]Value, b: [2]Value, opts: PackedF32x2Options) [2]Value {
        return self.unvec2(self.emit(nvvm.ops.mul_packed_f32x2(self.ctx, self.vec2(a).inner, self.vec2(b).inner, mlir.Type.vector(&.{2}, mlir.Type.float(self.ctx, .f32)), .{ .rnd = self.packedF32x2Attrs(opts).rnd, .ftz = self.packedF32x2Attrs(opts).ftz }, self.loc())));
    }

    pub const StoreOptions = struct {
        /// Address space for raw integer addresses; typed pointers retain theirs.
        space: NVPTXAddressSpace = .global,
        level1_eviction_priority: nvvm.ops.EvictKind = .evict_normal,
    };

    /// Scalar/vector store. Eight i32 words with no_allocate keep STG.E.256.
    pub fn store(self: *Builder, addr: Value, value: anytype, comptime opts: StoreOptions) void {
        if (@TypeOf(value) != Value) {
            const n = common.arity(@TypeOf(value), "store: value");
            comptime std.debug.assert(n == 1 or n == 2 or n == 4 or n == 8);
            var words: [n]Value = undefined;
            inline for (0..n) |i| words[i] = self.lift(value[i]).to(.i32);
            const is_global = if (addr.type_().isA(cute_ir.PtrType)) |ty| ty.addressSpace() == .gmem else addr.isInt() and opts.space == .global;
            if (n == 8 and is_global and opts.level1_eviction_priority == .evict_no_allocate) {
                const address = if (addr.isInt()) addr else self.ptrToInt(addr, .i64);
                var args: [9]*const mlir.Value = undefined;
                args[0] = address.inner;
                for (words, 0..) |word, i| args[i + 1] = word.inner;
                _ = llvm.inline_asm(self.ctx, "st.global.L1::no_allocate.v8.b32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};", &args, .{ .constraints = "l,r,r,r,r,r,r,r,r,~{memory}", .has_side_effects = true }, self.loc()).appendTo(self.currentBlock());
                return;
            }
            self.store(addr, self.joinWords(n, words), opts);
            return;
        }
        const ptr = if (addr.isInt()) self.addressPtr(addr, opts.space) else if (addr.type_().isA(cute_ir.PtrType)) |ty| blk: {
            const space: NVPTXAddressSpace = @fromBackingInt(@intCast(@backingInt(ty.addressSpace())));
            break :blk self.emit(mlir.unrealized_conversion_cast(self.ctx, &.{addr.inner}, &.{llvm.pointerType(self.ctx, space)}, self.loc()));
        } else addr;
        self.emitVoid(nvvm.ops.store_ext(self.ctx, value.inner, ptr.inner, .{
            .evict = if (opts.level1_eviction_priority == .evict_normal) null else (nvvm.ops.EvictKindAttr.get(self.ctx, .{ .value = opts.level1_eviction_priority }) catch unreachable).attribute(),
        }, self.loc()));
    }

    pub fn fma(self: *Builder, x: Value, y: Value, z: Value) Value {
        return self.callIntrinsic("llvm.fma.f32", &.{ x, y, z }, mlir.Type.float(self.ctx, .f32)).?;
    }

    pub fn clock(self: *Builder) Value {
        return self.emit(llvm.inline_asm(self.ctx, "mov.u32 $0, %clock;", &.{}, .{ .result_type = mlir.Type.int(self.ctx, .i32), .constraints = "=r", .has_side_effects = true }, self.loc()));
    }

    pub fn globalTimer(self: *Builder) Value {
        return self.emit(llvm.inline_asm(self.ctx, "mov.u64 $0, %globaltimer;", &.{}, .{ .result_type = mlir.Type.int(self.ctx, .i64), .constraints = "=l", .has_side_effects = true }, self.loc()));
    }

    // ==================== scalars ====================

    /// `value` as a constant of `dt`; a Value passes through and enums use
    /// their backing integers. An integer is signed or an unsigned bit
    /// pattern of `dt`'s width; anything else panics.
    pub fn constant(self: *Builder, dt: DType, value: anytype) Value {
        if (@TypeOf(value) == Value) return value;
        if (@typeInfo(@TypeOf(value)) == .@"enum") return self.constant(dt, @backingInt(value));
        const ctx = self.ctx;
        const T = @TypeOf(value);
        if (isFloatDtype(dt)) {
            const f: f64 = switch (@typeInfo(T)) {
                .comptime_int, .comptime_float => value,
                .int => @floatFromInt(value),
                .bool => @floatFromInt(@intFromBool(value)),
                .float => @floatCast(value),
                else => @compileError("Builder.constant: unsupported value type " ++ @typeName(T)),
            };
            return self.emit(switch (dt) {
                inline .f16, .bf16, .f32, .f64, .f4e2m1fn, .f8e4m3fn, .f8e5m2, .f8e8m0fnu => |ft| arith.constant_float(ctx, f, @field(mlir.FloatTypes, @tagName(ft)), self.loc()),
                else => unreachable,
            });
        }
        const i: i128 = switch (@typeInfo(T)) {
            .comptime_int, .int => value,
            .bool => @intFromBool(value),
            .comptime_float, .float => @intFromFloat(value),
            else => @compileError("Builder.constant: unsupported value type " ++ @typeName(T)),
        };
        return self.emit(arith.constant_int(ctx, intBits(i, @intCast(dtypeBitwidth(dt)), @tagName(dt)), dt.toMlir(ctx), self.loc()));
    }

    /// `value` as the `i64` an integer constant of `bits` bits is built from.
    /// It may be signed or an unsigned bit pattern, so it must lie in
    /// [-2^(bits-1), 2^bits).
    fn intBits(value: i128, bits: u7, dtype: []const u8) i64 {
        if (value < -(@as(i128, 1) << (bits - 1)) or value >= @as(i128, 1) << bits)
            std.debug.panic("Builder.constant: {d} does not fit {s}", .{ value, dtype });
        return @truncate(value);
    }

    fn fitsInt(i: i64, dt: DType) bool {
        if (dt == .i1) return i == 0 or i == 1;
        const bits = dtypeBitwidth(dt);
        if (bits >= 64) return true;
        const shift: u6 = @intCast(bits - 1);
        return i >= -(@as(i64, 1) << shift) and i < (@as(i64, 1) << shift);
    }

    fn literalFits(value: anytype, dt: DType) bool {
        return switch (@typeInfo(@TypeOf(value))) {
            .comptime_int, .int => fitsInt(@intCast(value), dt),
            .bool => !isFloatDtype(dt),
            else => isFloatDtype(dt),
        };
    }

    /// A Zig scalar as a constant: ints become `i32` when they fit, `i64`
    /// otherwise; floats `f32`, like the Python DSL's `Int32`/`Float32`.
    pub fn lift(self: *Builder, value: anytype) Value {
        const T = @TypeOf(value);
        if (T == Value) return value;
        return switch (@typeInfo(T)) {
            .comptime_int, .int => if (literalFits(value, .i32)) self.constant(.i32, value) else self.constant(.i64, value),
            .bool => self.constant(.i1, value),
            .comptime_float => self.constant(.f32, value),
            .float => |info| switch (info.bits) {
                16 => self.constant(.f16, value),
                32 => self.constant(.f32, value),
                64 => self.constant(.f64, value),
                else => @compileError("Builder.lift: unsupported float bitwidth"),
            },
            else => @compileError("Builder.lift: unsupported type " ++ @typeName(T)),
        };
    }

    /// `value` as a constant of `like`'s element type; a Value passes through.
    pub fn constantLike(self: *Builder, value: anytype, like: Value) Value {
        return self.constant(like.dtype(), value);
    }

    fn liftLike(self: *Builder, value: anytype, ref: Value) Value {
        const dt = ref.dtype();
        return if (literalFits(value, dt)) self.constant(dt, value) else self.lift(value);
    }

    /// Lift a literal to the other operand's dtype when it fits, else to its
    /// own width; then widen the narrower integer.
    pub fn coerce(self: *Builder, a: anytype, b: anytype) struct { Value, Value } {
        var av: Value = if (@TypeOf(a) == Value) a else if (@TypeOf(b) == Value) self.liftLike(a, b) else self.lift(a);
        var bv: Value = if (@TypeOf(b) == Value) b else self.liftLike(b, av);
        if (av.isInt() and bv.isInt()) {
            const aw = dtypeBitwidth(av.dtype());
            const bw = dtypeBitwidth(bv.dtype());
            if (aw < bw) av = av.to(bv.dtype());
            if (bw < aw) bv = bv.to(av.dtype());
        }
        return .{ av, bv };
    }

    fn cmp(self: *Builder, a: Value, b: anytype, ip: arith.CmpIPredicate, fp: arith.CmpFPredicate) Value {
        const l, const r = self.coerce(a, b);
        return self.emit(if (l.isFloat()) arith.cmpf(self.ctx, fp, l.inner, r.inner, self.loc()) else arith.cmpi(self.ctx, ip, l.inner, r.inner, self.loc()));
    }

    pub fn select(self: *Builder, cond: Value, t: anytype, f: anytype) Value {
        const tv, const fv = self.coerce(t, f);
        return self.emit(arith.select(self.ctx, cond.inner, tv.inner, fv.inner, self.loc()));
    }

    // ==================== scf ====================

    pub fn scalarTy(self: *const Builder, dtype: DType) *const mlir.Type {
        return dtype.toMlir(self.ctx);
    }

    pub fn openIf(self: *Builder, cond: Value) IfOnlyScope {
        return common.openIf(self, Value, cond);
    }

    pub fn openIfElse(self: *Builder, cond: Value, result_types: anytype) IfScope(common.arity(@TypeOf(result_types), "openIfElse: result_types")) {
        return common.openIfElse(self, Value, cond, result_types);
    }

    /// `for iv in range(lower, upper, step)` with loop-carried `inits`. Literal
    /// bounds take the type of a Value bound, else `lower` is lifted (`i32`
    /// when it fits).
    pub fn openFor(
        self: *Builder,
        lower: anytype,
        upper: anytype,
        step: anytype,
        inits: anytype,
    ) ForScope(common.arity(@TypeOf(inits), "openFor: inits")) {
        return self.openForOpts(lower, upper, step, inits, .{});
    }

    /// `openFor` with `scf.for` options.
    pub fn openForOpts(
        self: *Builder,
        lower: anytype,
        upper: anytype,
        step: anytype,
        inits: anytype,
        opts: common.ForOpts,
    ) ForScope(common.arity(@TypeOf(inits), "openFor: inits")) {
        return common.openFor(self, Value, common.forBounds(self, Value, lower, upper, step, Builder.lift), inits, opts);
    }

    /// `scf.while` with explicit loop-carried values. The before region sees
    /// `inits`, calls `yieldBefore(condition, forwarded)`, then the after
    /// region sees `forwarded` and calls `yieldAfter(next_inits)`. This is the
    /// control-flow shape emitted by the Python CuTe persistent scheduler.
    pub fn openWhile(
        self: *Builder,
        inits: anytype,
        after_types: anytype,
    ) WhileScope(common.arity(@TypeOf(inits), "openWhile: inits"), common.arity(@TypeOf(after_types), "openWhile: after_types")) {
        return common.openWhile(self, Value, inits, after_types);
    }

    // ==================== module ====================

    /// The module the CuTe compiler takes: the verified kernel as a public
    /// `func.func`, which the compiler turns into the kernel entry. Grid and
    /// block travel in the custom call; `block` is also pinned as
    /// `nvvm.reqntid`.
    pub fn finish(self: *Builder, block: [3]i32) FinishError![:0]const u8 {
        const current = self.currentBlock();
        if (current.terminator() == null) {
            _ = func.returns(self.ctx, &.{}, self.loc()).appendTo(current);
        }
        const func_op = self.func_op orelse return error.InvalidMlir;
        func_op.setAttribute(nvvm.reqntid(self.ctx, block));
        try common.verify(self.module, "dsl/cute");

        return self.renderModule();
    }
};

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    registerDialects(registry);
    func.registerExtensions(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

fn expectContains(haystack: []const u8, needle: []const u8) !void {
    if (std.mem.indexOf(u8, haystack, needle) == null) {
        std.debug.print("missing {s} in:\n{s}\n", .{ needle, haystack });
        return error.TestExpectedContains;
    }
}

test "naive elementwise add matches the Python notebook kernel" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "naive_elementwise_add_kernel");
    defer b.deinit();
    const a = try b.declareArgs(.{
        .gA = .{ .tensor = .{ .dtype = .f16, .shape = &.{ 16, 8 } } },
        .gB = .{ .tensor = .{ .dtype = .f16, .shape = &.{ 16, 8 } } },
        .gC = .{ .tensor = .{ .dtype = .f16, .shape = &.{ 16, 8 } } },
    });

    const tidx = b.threadIdx().x;
    const bidx = b.blockIdx().x;
    const bdim = b.blockDim().x;
    const thread_idx = bidx.mul(bdim).add(tidx);
    const n = a.gA.dim(1);
    const ni = thread_idx.rem(n);
    const mi = thread_idx.div(n);
    a.gC.set(.{ mi, ni }, a.gA.get(.{ mi, ni }).add(a.gB.get(.{ mi, ni })));

    // The module round-trips through the local dialects.
    const ir = try b.finish(.{ 128, 1, 1 });
    defer std.testing.allocator.free(ir);
    try expectContains(ir, "\"func.func\"() <{function_type = (!cute.ptr<f16, gmem, align<16>>, !cute.ptr<f16, gmem, align<16>>, !cute.ptr<f16, gmem, align<16>>) -> (), sym_name = \"naive_elementwise_add_kernel\"}>");
    try expectContains(ir, "}) {cute.kernel, gpu.kernel, nvvm.reqntid = array<i32: 128, 1, 1>} : () -> ()");
    try expectContains(ir, "\"nvvm.read.ptx.sreg.tid.x\"() : () -> i32");
    try expectContains(ir, "!cute.layout<\"(16,8):(8,1)\">");
    try expectContains(ir, "!cute.coord<\"(?,?)\">");
    try expectContains(ir, "cute.memref.load");
    try expectContains(ir, "cute.memref.store");
    try expectContains(ir, "arith.addf");
    try expectContains(ir, "\"func.return\"() : () -> ()\n  })");
    try std.testing.expect(std.mem.indexOf(u8, ir, "gpu.module") == null);
    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "guard, shared memory, sync, layouts, for loop" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "reverse_blocks");
    defer b.deinit();
    const a = try b.declareArgs(.{
        .x = .{ .ptr = .f32 },
        .out = .{ .tensor = .{ .dtype = .f32, .shape = &.{4096} } },
    });
    const gx = b.makeTensor(a.x, b.makeLayout(&.{4096}, .{}));
    const smem = b.makeTensor(b.allocSmem(.f32, 128, 16), b.makeLayout(&.{128}, .{}));

    const tid = b.threadIdx().x;
    const i = b.blockIdx().x.mul(128).add(tid);
    smem.set(.{tid}, gx.get(.{i}));
    b.syncThreads();
    b.barrierCtaSyncAligned(2, 128);
    b.launchDependents();
    b.waitForDependency();
    const j = b.constant(.i32, 127).sub(tid);
    var guard = b.openIf(i.lt(4096));
    {
        var acc = b.openFor(0, 4, 1, .{b.constant(.f32, 0.0)});
        {
            acc.yield(.{acc.carried[0].add(smem.get(.{j}).mul(acc.iv.to(.f32)))});
        }
        a.out.set(.{i}, acc.results[0]);
        guard.close();
    }

    const kernel = try b.finish(.{ 128, 1, 1 });
    defer std.testing.allocator.free(kernel);
    try expectContains(kernel, "cute_nvgpu.arch.alloc_smem");
    try expectContains(kernel, "!cute.ptr<f32, smem, align<16>>");
    try expectContains(kernel, "\"nvvm.barrier\"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> ()");
    try expectContains(kernel, "llvm.nvvm.barrier.cta.sync.aligned.count");
    try expectContains(kernel, "value = 2 : i32");
    try expectContains(kernel, "value = 128 : i32");
    try expectContains(kernel, "griddepcontrol.launch_dependents;");
    try expectContains(kernel, "griddepcontrol.wait;");
    try expectContains(kernel, "scf.if");
    try expectContains(kernel, "scf.for");
    try expectContains(kernel, "!cute.layout<\"4096:1\">");
    const parsed = try mlir.Module.parse(ctx, kernel);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "left-most default stride and static coordinate leaves" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "k");
    defer b.deinit();
    const a = try b.declareArgs(.{ .p = .{ .ptr = .f32 } });
    const t = b.makeTensor(a.p, b.makeLayout(&.{ 4, 4 }, .{}));
    try std.testing.expectEqualSlices(i64, &.{ 1, 4 }, t.layout.stride.constSlice());
    try std.testing.expectEqual(@as(i64, 16), t.size());
    const v = t.get(.{ b.threadIdx().x, 3 });
    try std.testing.expect(v.isFloat());
    const kernel = try b.finish(.{ 1, 1, 1 });
    defer std.testing.allocator.free(kernel);
    try expectContains(kernel, "!cute.layout<\"(4,4):(1,4)\">");
    try expectContains(kernel, "!cute.coord<\"(?,3)\">");
    const parsed = try mlir.Module.parse(ctx, kernel);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "casts, select, if-else, integer widening" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "k");
    defer b.deinit();
    const a = try b.declareArgs(.{ .p = .{ .tensor = .{ .dtype = .f16, .shape = &.{8} } } });
    const tid = b.threadIdx().x;

    // A literal that does not fit i32 widens the i32 Value instead of truncating.
    const big: u32 = 3_000_000_000;
    const wide = tid.add(big);
    try std.testing.expectEqual(DType.i64, wide.dtype());
    const bound = tid.lt(@as(i64, 5_000_000_000));
    try std.testing.expectEqual(DType.i1, bound.dtype());

    const f = tid.to(.f32);
    const shuffled = b.shuffleXor(f, 16);
    try std.testing.expectEqual(DType.f16, f.to(.f16).dtype());
    try std.testing.expectEqual(DType.f64, f.to(.f64).dtype());
    try std.testing.expectEqual(DType.bf16, f.to(.f16).to(.bf16).dtype());
    try std.testing.expectEqual(DType.i32, f.to(.i32).dtype());
    try std.testing.expectEqual(DType.i8, tid.to(.i8).dtype());
    try std.testing.expectEqual(DType.i32, bound.to(.i32).dtype());

    const picked = b.select(bound, shuffled, 2.0);
    var scope = b.openIfElse(tid.lt(4), .{.f16});
    {
        scope.yieldThen(.{picked.to(.f16)});
    }
    {
        scope.yieldElse(.{b.constant(.f16, 1)});
    }
    a.p.set(.{tid}, scope.results[0]);

    const kernel = try b.finish(.{ 1, 1, 1 });
    defer std.testing.allocator.free(kernel);
    try expectContains(kernel, "arith.extsi");
    try expectContains(kernel, "\"arith.constant\"() <{value = 3000000000 : i64}> : () -> i64");
    try expectContains(kernel, "arith.sitofp");
    try expectContains(kernel, "arith.truncf");
    try expectContains(kernel, "arith.extf");
    try expectContains(kernel, "arith.convertf");
    try expectContains(kernel, "arith.fptosi");
    try expectContains(kernel, "arith.trunci");
    try expectContains(kernel, "arith.extui");
    try expectContains(kernel, "arith.select");
    try expectContains(kernel, "shfl.sync.bfly.b32");
    try expectContains(kernel, "\"scf.if\"");
    try expectContains(kernel, "}, {");
    const parsed = try mlir.Module.parse(ctx, kernel);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "complete program emits a Zig-built CUDA host launch" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.openProgram(std.testing.allocator, ctx, "program_add");
    defer b.deinit();
    const device = try b.declareArgs(.{
        .input = .{ .tensor = .{ .dtype = .f32, .shape = &.{128} } },
        .output = .{ .tensor = .{ .dtype = .f32, .shape = &.{128} } },
    });
    const tid = b.threadIdx().x;
    device.output.set(.{tid}, device.input.get(.{tid}).add(1.0));
    b.endFunction(.{ 128, 1, 1 });

    b.beginFunction("program_add", .host);
    _ = try b.declareArgs(.{
        .input = .{ .ptr = DType.f32 },
        .output = .{ .ptr = DType.f32 },
    });
    const one = b.constant(.i32, 1);
    const config = b.makeLaunchConfig(.{
        .grid = .{ one, one, one },
        .block = .{ b.constant(.i32, 128), one, one },
        .dynamic_smem = b.kernelSmemSize("program_add"),
        .stream = b.cudaStream(),
        .use_pdl = true,
    });
    const launched = b.launchEx("program_add", config, .{ b.arg(0), b.arg(1) });
    b.returnHostStatus(b.cudaResultStatus(launched));
    b.endFunction(null);

    const ir = try b.finishProgram();
    defer std.testing.allocator.free(ir);
    try expectContains(ir, "\"gpu.module\"");
    try expectContains(ir, "cuda.kernel");
    try expectContains(ir, "sym_name = \"program_add\"");
    try expectContains(ir, "cuda.launch_ex");
}

test "SM100 block-scaled MMA atoms use the CuTe DSL instruction shapes" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    registerDialects(registry);
    func.registerExtensions(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();

    var b = try Builder.open(std.testing.allocator, ctx, "sm100_blockscaled_atoms");
    defer b.deinit();
    const tma: TmaConfig = .{
        .dtype = .f8e4m3fn,
        .copy_bits = 16384,
        .global_basis = b.layoutSpec(.{ 128, 16, 1 }, .{ b.basis(1, 1, .{1}), b.basis(1, 1, .{0}), b.basis(1, 1, .{2}) }),
        .coordinate_layout = b.layoutSpec(.{ 16, 128, 8 }, .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}), b.basis(1, 1, .{2}) }),
        .coordinate_rank = 3,
    };
    var pair_tma = tma;
    pair_tma.cta_group = .two;
    const args = try b.declareArgs(.{
        .single = .{ .mlir_type = .{ .type_ = b.tmaLoadAtomType(tma), .grid_constant = true } },
        .pair = .{ .mlir_type = .{ .type_ = b.tmaLoadAtomType(pair_tma), .grid_constant = true } },
    });
    _ = b.makeExecTma(.{ .inner = args.single.inner, .kernel = &b }, b.tmaLoadExecAtomType(tma));
    _ = b.makeExecTma(.{ .inner = args.pair.inner, .kernel = &b }, b.tmaLoadExecAtomType(pair_tma));

    const sf_ptr = b.intToPtr(0, .f8e8m0fnu, .tmem, 1);
    for ([_]CtaGroup{ .one, .two }) |group| {
        _ = b.makeBlockScaledMma(.{ .n = 16, .cta_group = group }, sf_ptr);
        _ = b.makeBlockScaledMma(.{ .n = 128, .cta_group = group }, sf_ptr);
    }
    _ = b.makeBlockScaledMma(.{ .m = 256, .n = 16, .cta_group = .two }, sf_ptr);
    _ = b.makeBlockScaledMma(.{ .m = 256, .n = 128, .cta_group = .two }, sf_ptr);
    const nvfp4_sf_ptr = b.intToPtr(0, .f8e4m3fn, .tmem, 1);
    _ = b.makeBlockScaledMma(.{ .kind = .nvfp4_nvfp4, .n = 16 }, nvfp4_sf_ptr);

    const ir = try b.finish(.{ 192, 1, 1 });
    defer std.testing.allocator.free(ir);
    try expectContains(ir, "sm100.mma_bs<128x16x32, num_cta = 1");
    try expectContains(ir, "sm100.mma_bs<128x128x32, num_cta = 1");
    try expectContains(ir, "sm100.mma_bs<128x16x32, num_cta = 2");
    try expectContains(ir, "sm100.mma_bs<128x128x32, num_cta = 2");
    try expectContains(ir, "sm100.mma_bs<256x16x32, num_cta = 2");
    try expectContains(ir, "sm100.mma_bs<256x128x32, num_cta = 2");
    try expectContains(ir, "sm100.mma_bs<128x16x64, num_cta = 1, ab_major = (k, k), elem_type = (f4E2M1FN, f4E2M1FN, f32), sf_type = f8E4M3FN, frag_kind = ss, vec_size = 16>");
    try expectContains(ir, "non_exec_tiled_tma_load<sm_90,");
    try expectContains(ir, "non_exec_tiled_tma_load<sm_100_2sm,");
    try expectContains(ir, "tma_load<f8E4M3FN, copy_bits = 16384, mode = tiled, num_cta = 1,");
    try expectContains(ir, "tma_load<f8E4M3FN, copy_bits = 16384, mode = tiled, num_cta = 2,");
    try expectContains(ir, "cute.make_tiled_mma");
    try expectContains(ir, "cute_nvgpu.grid_constant");

    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "barrier helpers preserve dynamic ids, CTA groups, masks and remote scopes" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = try Builder.open(std.testing.allocator, ctx, "barrier_operands");
    defer b.deinit();
    _ = try b.declareArgs(.{});

    const barrier = b.allocSmem(.i64, 1, 8);
    const peer = b.clusterCtaRank().bitXor(1);
    b.barrierCtaSyncAligned(2, 128);
    b.barrierCtaSyncAligned(b.threadIdx().x.div(128).mul(2).add(1), 128);
    b.tcgen05Commit(barrier, .{ .cta_group = .two });
    b.tcgen05Commit(barrier, .{ .cta_group = .two, .multicast_mask = b.constant(.i32, 3) });
    _ = b.mbarrierArrive(barrier, .{ .peer_cta_rank_in_cluster = peer, .arrive_count = b.constant(.i32, 1), .scope = .cta });
    _ = b.mbarrierArrive(barrier, .{ .peer_cta_rank_in_cluster = peer, .arrive_count = b.constant(.i32, 32), .scope = .cluster });

    const ir = try b.finish(.{ 256, 1, 1 });
    defer std.testing.allocator.free(ir);
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, ir, "llvm.nvvm.barrier.cta.sync.aligned.count"));
    try expectContains(ir, "value = 2 : i32");
    try expectContains(ir, "value = 128 : i32");
    try expectContains(ir, "#nvvm.cta_group<cta_2>");
    try expectContains(ir, "value = 3 : i32");
    try expectContains(ir, "value = 32 : i32");
    try expectContains(ir, "!llvm.ptr<7>");
    try expectContains(ir, "#nvvm.mem_scope<cta>");
    try expectContains(ir, "#nvvm.mem_scope<cluster>");

    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "mbarrier waits preserve acquire scope through polling and blocking retries" {
    const ctx = try testContext();
    defer ctx.deinit();
    for ([_]MBarrierScope{ .cta, .cluster }) |scope| {
        var b = try Builder.open(std.testing.allocator, ctx, "barrier_waits");
        defer b.deinit();
        _ = try b.declareArgs(.{});

        const barrier = b.allocSmem(.i64, 1, 8);
        const phase = b.constant(.i32, 0);
        _ = b.mbarrierTestWait(barrier, phase, .{ .scope = scope });
        _ = b.mbarrierTryWait(barrier, phase, .{ .scope = scope });
        b.mbarrierWait(barrier, phase, .{ .scope = scope, .time_limit = 1234, .test_first = false });
        b.mbarrierWait(barrier, phase, .{ .scope = scope });

        const ir = try b.finish(.{ 32, 1, 1 });
        defer std.testing.allocator.free(ir);
        try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, ir, "#nvvm.mbar_wait<test>"));
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, ir, "#nvvm.mbar_wait<try>"));
        try expectContains(ir, "value = 1234 : i32");
        try expectContains(ir, "\"scf.if\"");
        switch (scope) {
            .cta => {
                try std.testing.expectEqual(@as(usize, 3), std.mem.count(u8, ir, "#nvvm.mbar_scope<cta>"));
                try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, ir, "\"nvvm.mbarrier.try_wait.parity\""));
                try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, ir, "\"scf.while\""));
            },
            .cluster => {
                try std.testing.expectEqual(@as(usize, 5), std.mem.count(u8, ir, "#nvvm.mbar_scope<cluster>"));
                try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, ir, "#nvvm.mem_order<acquire>"));
                try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, ir, "\"nvvm.mbarrier.try_wait.parity.timelimit\""));
                try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, ir, "\"scf.while\""));
                try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, ir, "\"scf.condition\""));
                try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, ir, "#nvvm.mbar_scope<cta>"));
            },
        }

        const parsed = try mlir.Module.parse(ctx, ir);
        defer parsed.deinit();
        try std.testing.expect(parsed.operation().verify());
    }
}

test "SM100 raw primitives preserve operand types and round-trip through MLIR" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = try Builder.open(std.testing.allocator, ctx, "sm100_raw_primitives");
    defer b.deinit();
    const cfg: TmaConfig = .{
        .dtype = .f8e4m3fn,
        .copy_bits = 1024,
        .global_basis = b.layoutSpec(.{ 128, 1 }, .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }),
        .coordinate_layout = b.layoutSpec(.{ 1024, 16 }, .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }),
        .coordinate_rank = 2,
    };
    const a = try b.declareArgs(.{
        .gmem = .{ .ptr = DType.i32 },
        .tma = .{ .mlir_type = .{ .type_ = b.tmaLoadAtomType(cfg), .grid_constant = true } },
    });
    const atom = b.makeExecTma(.{ .inner = a.tma.inner, .kernel = &b }, b.tmaLoadExecAtomType(cfg));
    const desc = b.tmaDescriptorAddress(atom);
    const dst = b.constant(.i32, 1024);
    const bar = b.constant(.i32, 0);
    const zero = b.constant(.i32, 0);
    const gmem = b.ptrToInt(a.gmem, .i64);
    const pred = b.threadIdx().x.eq(0);
    b.tmaGather4(desc, dst, bar, zero, .{ zero, zero.add(1), zero.add(2), zero.add(3) }, .evict_last);
    b.cpAsyncBulkPrefetch(gmem, b.constant(.i32, 128), .{});
    b.prefetchL1(gmem);
    b.prefetchTensorMap(desc);
    const swizzled = Tcgen05SmemDesc.build(&b, dst, .{ .leading_byte_offset = 16, .stride_byte_offset = 1024, .layout = .swizzle_128b });
    const interleaved = Tcgen05SmemDesc.build(&b, dst, .{ .leading_byte_offset = 1024, .stride_byte_offset = 128 });
    b.tcgen05MmaWs(.f16, zero, .{ .tmem = dst }, swizzled, Tcgen05InstrDesc.build(.{ .a_dtype = .bf16, .b_dtype = .bf16, .m_dim = 64, .n_dim = 128, .a_major = .k, .b_major = .k }), pred, .{});
    b.tcgen05MmaWs(.f16, zero, .{ .smem = interleaved }, swizzled, Tcgen05InstrDesc.build(.{ .a_dtype = .bf16, .b_dtype = .bf16, .m_dim = 64, .n_dim = 256, .a_major = .k, .b_major = .mn }), pred, .{});
    b.setMaxRegisterIncrease(128);
    b.barrierCtaSyncAligned(0, 512);
    b.setMaxRegisterDecrease(128);
    b.barrierCtaSyncAligned(b.threadIdx().x.div(128), 128);
    b.mbarrierSpin(bar, zero, .cta);
    b.mbarrierSpin(bar, zero, .cluster);
    const state = b.mbarrierArrive(bar, .{}).?;
    const remote = b.mapDsmemPtr(dst, zero);
    _ = remote;
    _ = b.mbarrierArrive(bar, .{ .peer_cta_rank_in_cluster = zero, .arrive_count = b.constant(.i32, 1), .scope = .cluster });
    b.storeAsyncDsmem(dst, .{ zero, zero }, bar, zero);
    b.storeAsyncDsmem(dst, .{ zero, zero, zero, zero }, bar, zero);
    b.barrierClusterArriveRelaxedAligned();
    b.barrierClusterWaitAligned();
    const any = b.voteAnySync(pred, null);
    const mask = b.voteBallotSync(pred, null);
    const x = b.constant(.f32, 1.0);
    const y = b.constant(.f32, 2.0);
    const maximum = b.fmax(x, y, .{});
    const sum = b.addPackedF32x2(.{ x, y }, .{ y, x }, .{});
    const product = b.mulPackedF32x2(sum, .{ x, y }, .{});
    const fused = b.fmaPackedF32x2(product, sum, .{ x, y }, .{});
    const scalar = b.fma(maximum, fused[0], fused[1]);
    b.store(gmem, .{ scalar.bitCast(.i32), maximum.bitCast(.i32), mask, any.to(.i32), state.to(.i32), b.clock(), b.globalTimer().to(.i32), zero }, .{ .level1_eviction_priority = .evict_no_allocate });

    const ir = try b.finish(.{ 512, 1, 1 });
    defer std.testing.allocator.free(ir);
    for ([_][]const u8{
        "cute_nvgpu.get_tma_desc_addr",             "tile::gather4",                          "nvvm.cp.async.bulk.prefetch",
        "prefetch.global.L1",                       "prefetch.tensormap",                     "nvvm.tcgen05.mma.ws",
        "nvvm.tcgen05.mma.ws",                      "setmaxnreg.inc",                         "setmaxnreg.dec",
        "llvm.nvvm.barrier.cta.sync.aligned.count", "mbarrier.test_wait.parity.acquire.cta",  "mbarrier.test_wait.parity.acquire.cluster",
        "nvvm.mapa",                                "#nvvm.mem_scope<cluster>",               "nvvm.store.async",
        "vector<4xi32>",                            "barrier.cluster.arrive.relaxed.aligned", "nvvm.cluster.wait",
        "llvm.nvvm.vote.any.sync",                  "llvm.nvvm.vote.ballot.sync",             "llvm.nvvm.fmax.f",
        "nvvm.add.packed.f32x2",                    "nvvm.mul.packed.f32x2",                  "nvvm.fma.packed.f32x2",
        "llvm.fma.f32",                             "st.global.L1::no_allocate.v8.b32",       "%clock",
        "%globaltimer",
    }) |op| try expectContains(ir, op);
    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "SM100 BF16 weight-stationary descriptor encodes supported shapes and major modes" {
    try std.testing.expectEqual(@as(u32, 0x04200490), Tcgen05InstrDesc.build(.{ .a_dtype = .bf16, .b_dtype = .bf16, .m_dim = 64, .n_dim = 128, .a_major = .k, .b_major = .k }));
    try std.testing.expectEqual(@as(u32, 0x04410490), Tcgen05InstrDesc.build(.{ .a_dtype = .bf16, .b_dtype = .bf16, .m_dim = 64, .n_dim = 256, .a_major = .k, .b_major = .mn }));
    try std.testing.expectEqual(@as(u32, 0x08118490), Tcgen05InstrDesc.build(.{ .a_dtype = .bf16, .b_dtype = .bf16, .m_dim = 128, .n_dim = 64, .a_major = .mn, .b_major = .mn }));
    try std.testing.expectEqual(@as(u32, 0x02100490), Tcgen05InstrDesc.build(.{ .a_dtype = .bf16, .b_dtype = .bf16, .m_dim = 32, .n_dim = 64, .a_major = .k, .b_major = .k }));
}

test "SM100 shared descriptors preserve nonzero swizzle bases and boundary offsets" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = try Builder.open(std.testing.allocator, ctx, "smem_descriptors");
    defer b.deinit();
    const args = try b.declareArgs(.{ .out = .{ .ptr = DType.i64 } });
    const addr = b.constant(.i32, 128);
    const fields = [_]u64{
        0x4000404000010000, // sw128, base 0, leading 16 bytes, stride 1024 bytes
        0x4002404000010000, // sw128, pattern at byte 128: base 1
        0x400e404000010000, // sw128, pattern at byte 896: base 7
        0x00007fff3fff0000, // unswizzled, largest representable offsets
    };
    const descs = [_]Value{
        Tcgen05SmemDesc.build(&b, addr, .{ .leading_byte_offset = 16, .stride_byte_offset = 1024, .base_offset = 0, .layout = .swizzle_128b }),
        Tcgen05SmemDesc.build(&b, addr, .{ .leading_byte_offset = 16, .stride_byte_offset = 1024, .base_offset = 1, .layout = .swizzle_128b }),
        Tcgen05SmemDesc.build(&b, addr, .{ .leading_byte_offset = 16, .stride_byte_offset = 1024, .base_offset = 7, .layout = .swizzle_128b }),
        Tcgen05SmemDesc.build(&b, addr, .{ .leading_byte_offset = 0x3fff0, .stride_byte_offset = 0x3fff0 }),
    };
    const out = b.ptrToInt(args.out, .i64);
    for (descs, 0..) |desc, i| b.storeWords(2, out.add(@as(i64, @intCast(i * 8))), .global, .{ desc.to(.i32), desc.shrLogical(32).to(.i32) });
    const ir = try b.finish(.{ 32, 1, 1 });
    defer std.testing.allocator.free(ir);
    for (fields) |bits| {
        const expected = try std.fmt.allocPrint(std.testing.allocator, "value = {d} : i64", .{bits});
        defer std.testing.allocator.free(expected);
        try expectContains(ir, expected);
    }
    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "makeLayout preserves dynamic singleton and nested shape and stride leaves" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = try Builder.open(std.testing.allocator, ctx, "mixed_layout");
    defer b.deinit();
    _ = try b.declareArgs(.{});
    const n = b.blockIdx().x.add(1);
    const singleton = b.constant(.i32, 1);
    const compact = b.makeLayout(.{ .{ 4, n }, singleton }, .{});
    const explicit = b.makeLayout(.{ n, 16 }, .{ .stride = .{ 1, n } });
    const static_nested = b.makeLayout(.{ .{ 4, 8 }, 16 }, .{});
    _ = compact;
    _ = explicit;
    _ = static_nested;
    const ir = try b.finish(.{ 32, 1, 1 });
    defer std.testing.allocator.free(ir);
    try expectContains(ir, "((4,?),?):((1,4),?)");
    try expectContains(ir, "(?,16):(1,?)");
    try expectContains(ir, "((4,8),16):((1,4),32)");
    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "makeLayout preserves 64-bit shape and stride values" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = try Builder.open(std.testing.allocator, ctx, "wide_layout");
    defer b.deinit();
    const args = try b.declareArgs(.{ .extent = .{ .mlir_type = .{ .type_ = DType.i64.toMlir(ctx) } } });
    const wide = args.extent.add(@as(i64, 1) << 32);
    const explicit = b.makeLayout(.{ 8, 2 }, .{ .stride = .{ 1, wide } });
    const wide_leaf: AlgebraExpr = .{ .dynamic_int = .{ .width = 64 } };
    try std.testing.expect(explicit.type_().eql(b.layoutType(.{
        .shape = b.expr(.{ 8, 2 }),
        .stride = .{ .tuple = &.{ .{ .static = 1 }, wide_leaf } },
    })));
    _ = b.makeLayout(.{ .{ 4, wide }, 1, 8 }, .{ .stride = null });
    const ir = try b.finish(.{ 32, 1, 1 });
    defer std.testing.allocator.free(ir);
    try std.testing.expect(std.mem.indexOf(u8, ir, "arith.trunci") == null);
    try expectContains(ir, "((4,?{i64}),1,8):((1,4),0,?{i64})");
    try expectContains(ir, "value = 4294967296 : i64");
    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "TMA constructors normalize tilers and staged layouts and copy executes store atoms" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = try Builder.open(std.testing.allocator, ctx, "tma_stores");
    defer b.deinit();
    const cfg: TmaConfig = .{
        .dtype = .f32,
        .copy_bits = 2048,
        .global_basis = b.layoutSpec(.{ 16, 4 }, .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }),
        .coordinate_layout = b.layoutSpec(.{ 128, 64 }, .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }),
        .coordinate_rank = 2,
    };
    const args = try b.declareArgs(.{
        .data = .{ .tensor = .{ .dtype = .f32, .shape = &.{ 128, 64 }, .stride = &.{ 1, 128 } } },
    });
    const staged = b.makeLayout(.{ 16, 4, 2 }, .{ .stride = .{ 1, 16, 64 } });
    const store = b.makeTiledTmaAtom(CopyBulkTensorTileS2GOp{}, args.data, staged, .{ 16, 4 }, cfg, .{});
    const smem = b.makeTensorView(b.intToPtr(0, .f32, .smem, 128), b.makeLayout(.{ 16, 4 }, .{}));
    const tile = b.makeCoordTensor(.{ 0, 0 }, b.staticLayout(cfg.global_basis));
    b.copy(store.atom, .{smem}, .{tile}, .{});
    b.copy(store.atom, .{smem}, .{tile}, .{ .cache_policy = b.constant(.i64, CacheHintSm90.evict_last) });
    const swizzled = b.makeComposedLayout(.{ .bits = 1, .base = 4, .shift = 3 }, 0, b.layoutSpec(.{ 16, 4, 2 }, .{ 1, 16, 64 }));
    _ = b.makeTiledTmaAtom(CopyBulkTensorTileG2SOp{}, args.data, swizzled, b.makeTile(&.{ 16, 4 }), cfg, .{});
    _ = b.makeTiledTmaAtom(CopyBulkTensorTileG2SOp{}, args.data, staged, b.makeLayout(.{ 16, 4 }, .{ .stride = .{ 1, 128 } }), cfg, .{});
    const ir = try b.finish(.{ 32, 1, 1 });
    defer std.testing.allocator.free(ir);
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, ir, "\"cute_nvgpu.atom.make_exec_tma\""));
    try std.testing.expectEqual(@as(usize, 3), std.mem.count(u8, ir, "\"cute.select\""));
    try expectContains(ir, "(16,4):(1@0,1@1)");
    try expectContains(ir, "tma_store<f32, copy_bits = 2048");
    try expectContains(ir, "atom_copy_field_tmastore<cache_policy>");
    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "CuTe options preserve runtime masks, timed try waits, descriptors and store widths" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = try Builder.open(std.testing.allocator, ctx, "cute_options");
    defer b.deinit();
    const a = try b.declareArgs(.{ .out = .{ .ptr = DType.i32 } });
    const tid = b.threadIdx().x;
    const mbar = b.allocSmem(.i64, 1, 8);
    b.barrier(.{});
    b.barrier(.{ .barrier_id = tid.div(128), .number_of_threads = b.blockDim().x });
    b.tcgen05Commit(mbar, .{ .cta_group = .two, .multicast_mask = b.clusterCtaRank().shl(1).bitOr(1) });
    _ = b.mbarrierTryWait(mbar, b.constant(.i32, 0), .{ .time_limit = tid.add(100), .scope = .cluster });
    const state = b.mbarrierArrive(mbar, .{ .relaxed = true }).?;
    b.storeAsyncDsmem(b.constant(.i32, 128), tid, mbar, b.clusterCtaRank().bitXor(1));
    b.clusterArriveRelaxed(.{});
    b.clusterWait();
    b.clusterArriveRelaxed(.{ .aligned = true });
    b.barrierClusterWaitAligned();
    _ = b.voteAnySync(tid.eq(0), tid.bitOr(1));
    _ = b.voteBallotSync(tid.eq(0), 0xff);
    const x = tid.to(.f32);
    const y = b.fmax(x, b.constant(.f32, 0), .{ .nan = true, .ftz = true, .abs = true });
    const pair = b.addPackedF32x2(.{ x, y }, .{ y, x }, .{ .rnd = .rz, .ftz = true });
    b.store(a.out, state.to(.i32), .{});
    b.store(a.out, .{ pair[0].bitCast(.i32), pair[1].bitCast(.i32) }, .{});
    const desc = b.makeUmmaSmemDesc(b.intToPtr(0, .bf16, .smem, 16), b.layoutSpec(.{ 64, 16 }, .{ 16, 1 }), .k, null);
    const raw_desc = b.smemDescriptorToInt(desc);
    _ = b.intToSmemDescriptor(raw_desc);
    const mapped = b.mapDsmemPtr(b.intToPtr(128, .i32, .smem, 16), b.clusterCtaRank());
    const mapped_ty = mapped.type_().isA(cute_ir.PtrType).?;
    try std.testing.expectEqual(MemorySpace.dsmem, mapped_ty.addressSpace());
    try std.testing.expectEqual(@as(u64, 16), mapped_ty.alignment());
    b.store(mapped, tid, .{});
    const swizzles = [_]Tcgen05SmemSwizzle{ .none, .swizzle_32b, .swizzle_64b, .swizzle_128b, .swizzle_128b_atom_32b };
    for (swizzles) |layout| _ = Tcgen05SmemDesc.build(&b, tid.mul(16), .{ .layout = layout });
    const idesc = b.constant(.i32, Tcgen05InstrDesc.build(.{ .a_dtype = .bf16, .b_dtype = .bf16, .m_dim = 64, .n_dim = 128 }));
    b.tcgen05MmaWs(.f16, tid, .{ .tmem = tid }, desc, idesc, tid.ne(0), .{ .collector_b_buffer = .b2, .collector_op = .use, .col_b_zero_mask = state });
    b.cpAsyncBulkPrefetch(a.out, 128, .{ .l2_cache_hint = state });
    const ir = try b.finish(.{ 128, 1, 1 });
    defer std.testing.allocator.free(ir);
    try expectContains(ir, "nvvm.mbarrier.try_wait.parity.timelimit");
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, ir, "scf.while"));
    try expectContains(ir, "nvvm.store.ext");
    try expectContains(ir, "vector<2xi32>");
    try expectContains(ir, "#nvvm.fp_rnd_mode<rz>");
    try expectContains(ir, "cute_nvgpu.make_umma_smem_desc");
    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "gather4 uses CuTe TMA atom construction, partitioning and copy options" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = try Builder.openProgram(std.testing.allocator, ctx, "gather4_copy");
    defer b.deinit();
    const cfg: TmaConfig = .{
        .dtype = .f32,
        .copy_bits = 2048,
        .global_basis = b.layoutSpec(.{ 16, 1 }, .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }),
        .coordinate_layout = b.layoutSpec(.{ 128, 64 }, .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }),
        .coordinate_rank = 2,
        .mode = .gather4,
    };
    var pair = cfg;
    pair.cta_group = .two;
    const args = try b.declareArgs(.{
        .indices = .{ .tensor = .{ .dtype = .i32, .shape = &.{ 128, 64 }, .stride = &.{ 0, 1 } } },
        .atom = .{ .mlir_type = .{ .type_ = b.tmaLoadAtomType(cfg), .grid_constant = true } },
        .pair = .{ .mlir_type = .{ .type_ = b.tmaLoadAtomType(pair), .grid_constant = true } },
    });
    const smem_layout = b.makeLayout(.{ 16, 4 }, .{ .stride = .{ 1, 16 } });
    const smem = b.makeTensorView(b.intToPtr(0, .f32, .smem, 128), smem_layout);
    const coords = b.makeCoordTensor(.{ 0, 0 }, b.staticLayout(cfg.coordinate_layout));
    const atom: Atom = .{ .inner = args.atom.inner, .kernel = &b };
    const partitioned = b.tmaPartition(atom, 0, b.makeLayout(1, .{}), smem, .{ coords, args.indices.view() }, smem.type_(), &.{ coords.type_(), args.indices.inner.type_() });
    const mbar = b.allocSmem(.i64, 1, 8);
    b.copy(atom, .{ partitioned.targets[0], partitioned.targets[1] }, .{partitioned.smem}, .{
        .tma_bar_ptr = mbar,
        .cache_policy = b.constant(.i64, CacheHintSm100.evict_last),
    });
    b.copy(.{ .inner = args.pair.inner, .kernel = &b }, .{ partitioned.targets[0], partitioned.targets[1] }, .{partitioned.smem}, .{ .tma_bar_ptr = mbar });
    b.endFunction(.{ 32, 1, 1 });

    b.beginFunction("gather4_copy", .host);
    const host = try b.declareArgs(.{
        .data = .{ .tensor = .{ .dtype = .f32, .shape = &.{ 128, 64 }, .stride = &.{ 1, 128 } } },
        .indices = .{ .tensor = .{ .dtype = .i32, .shape = &.{ 128, 64 }, .stride = &.{ 0, 1 } } },
    });
    const host_layout = b.makeLayout(.{ 16, 4 }, .{ .stride = .{ 1, 16 } });
    const tiler = b.makeTile(&.{ 16, 4 });
    _ = b.makeTiledTmaAtom(CopyBulkTensor2DGather4G2SOp{}, host.data, host_layout, tiler, cfg, .{ .gmem_coord_tensor = host.indices.view() });
    _ = b.makeTiledTmaAtom(CopyBulkTensor2DGather4G2SOp{ .cta_group = .two }, host.data, host_layout, tiler, pair, .{ .gmem_coord_tensor = host.indices.view() });
    b.returnHostStatus(b.constant(.i32, 0));
    b.endFunction(null);
    const ir = try b.finishProgram();
    defer std.testing.allocator.free(ir);
    try expectContains(ir, "atom.make_non_exec_2d_gather4_tma_load");
    try expectContains(ir, "atom.tma_partition");
    try expectContains(ir, "mode = gather4, num_cta = 1");
    try expectContains(ir, "mode = gather4, num_cta = 2");
    try expectContains(ir, "non_exec_2d_gather4_tma_load<sm_100_2sm");
    try expectContains(ir, "cute.copy");
    try expectContains(ir, "cache_policy");
    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}
