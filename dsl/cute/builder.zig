const std = @import("std");

const arith = @import("mlir/dialects/arith");
const cf = @import("mlir/dialects/cf");
const cuda = @import("mlir/dialects/cuda");
const cute_ir = @import("mlir/dialects/cute_ir");
const cute_nvgpu = @import("mlir/dialects/cute_nvgpu");
const dsl = @import("dsl/common/builder");
const func = @import("mlir/dialects/func");
const gpu = @import("mlir/dialects/gpu");
const llvm = @import("mlir/dialects/llvm");
const math = @import("mlir/dialects/math");
const nvvm = @import("mlir/dialects/nvvm");
const scf = @import("mlir/dialects/scf");
const vector = @import("mlir/dialects/vector");
const tupleArity = dsl.tupleArity;
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
        return k.cst(self.dtype(), 0).sub(self);
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
        const v = k.liftAs(value_, self.dtype).to(self.dtype);
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
    u4_unpack_u8,
    u16,
};

/// Typed description of a tiled TMA atom: the element type, the bits moved
/// per copy, the basis of one box, and the coordinate layout of the tensor.
pub const TmaConfig = struct {
    dtype: DType,
    copy_bits: u32,
    global_basis: LayoutSpec,
    coordinate_layout: LayoutSpec,
    coordinate_rank: u8,
    format: TmaFormat = .default,
};

pub const BlockScaledMma = struct {
    atom: Atom,
    tiled: Atom,
};

/// Shape of the SM100 block-scaled MXFP4 x MXFP8 MMA instruction.
pub const BlockScaledMmaConfig = struct {
    m: u16 = 128,
    n: u16,
    k: u16 = 32,
    num_cta: u8 = 1,
    vec_size: u8 = 32,
};

fn tmaDataFormat(format: TmaFormat, dtype: DType) cute_nvgpu.TmaDataFormat {
    return switch (format) {
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

pub const IfOnlyScope = dsl.IfOnlyScope(Builder, Value);

pub fn IfScope(comptime N: usize) type {
    return dsl.IfScope(Builder, Value, N);
}

pub fn ForScope(comptime N: usize) type {
    return dsl.ForScope(Builder, Value, N);
}

pub fn WhileScope(comptime N: usize, comptime M: usize) type {
    return dsl.WhileScope(Builder, Value, N, M);
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
    block_stack: std.ArrayList(*mlir.Block) = .empty,
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
        self.module.operation().setAttributeByName("gpu.container_module", .unit(ctx));
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

    fn declareArgsLow(self: *Builder, args: []const ArgSpec) FinishError!void {
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
                .mlir_type => |typed| if (typed.grid_constant) .dict(ctx, &.{
                    .named(ctx, "cute_nvgpu.grid_constant", .unit(ctx)),
                }) else empty_dict,
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
                    .named(ctx, "cute.kernel", .unit(ctx)),
                    .named(ctx, "gpu.kernel", .unit(ctx)),
                },
            }),
            .cuda_kernel => cuda.kernel(ctx, .{
                .name = self.name,
                .function_type = .function(ctx, arg_types, &.{}),
                .body = entry,
                .arg_attrs = .array(ctx, arg_attrs),
                .extra_attributes = &.{
                    .named(ctx, "cute.kernel", .unit(ctx)),
                    .named(ctx, "gpu.kernel", .unit(ctx)),
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
                .extra_attributes = &.{.named(ctx, "llvm.emit_c_interface", .unit(ctx))},
            }),
        };
        _ = func_op.appendTo(if (self.function_kind == .cuda_kernel)
            self.gpu_body orelse @panic("cuda_kernel requires Builder.openProgram")
        else
            self.module.body());

        self.func_op = func_op;
        self.entry_block = entry;
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
        self.block_stack.append(self.arena.allocator(), b) catch @panic("Builder.pushBlock OOM");
    }

    pub fn popBlock(self: *Builder) void {
        _ = self.block_stack.pop();
    }

    pub fn currentBlock(self: *Builder) *mlir.Block {
        if (self.block_stack.items.len > 0) {
            return self.block_stack.items[self.block_stack.items.len - 1];
        }
        return self.entry_block orelse @panic("Builder has no current block; call declareArgs first");
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

    pub fn setFunctionAttribute(self: *Builder, name: []const u8, attribute: *const mlir.Attribute) void {
        const function = self.func_op orelse @panic("setFunctionAttribute called before declareArgs");
        function.setAttributeByName(name, attribute);
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
        return self.emit(cuda.cast(self.ctx, cuda.streamType(self.ctx), self.cst(.i64, 0).inner, self.loc()));
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
        self.emitVoid(cuda.launch_cfg_cooperative(self.ctx, config.inner, self.cst(.i32, @intFromBool(opts.cooperative)).inner, self.loc()));
        self.emitVoid(cuda.launch_cfg_programmatic_stream_serialization_allowed(self.ctx, config.inner, self.cst(.i32, @intFromBool(opts.use_pdl)).inner, self.loc()));
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
            function.setAttributeByName("nvvm.reqntid", .denseArray(self.ctx, .i32, &dims));
        }
        self.func_op = null;
        self.entry_block = null;
        self.block_stack.clearRetainingCapacity();
        self.args = &.{};
        self.tensors = &.{};
    }

    pub fn finishProgram(self: *Builder) FinishError![:0]const u8 {
        std.debug.assert(self.entry_block == null and self.func_op == null);
        if (!self.module.operation().verify()) return error.InvalidMlir;
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
        return self.emit(cute_ir.ptrtoint(self.ctx, self.asValue(pointer).inner, dt.toMlir(self.ctx), self.loc()));
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

    /// `cute.make_layout(shape, stride=...)`, static only.
    pub fn makeLayout(self: *Builder, shape: []const i64, opts: LayoutOpts) Layout {
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
            dyn.append(self.lift(coord).to(.i32).inner) catch @panic("makeCoord: too many dynamic leaves");
            return .{ .dynamic = 1 };
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

    pub fn copy(self: *Builder, atom: Atom, src: anytype, dst: anytype, pred: ?Value) void {
        self.emitVoid(cute_ir.copy(self.ctx, atom.inner, self.innerSlice(src), self.innerSlice(dst), .{ .pred = if (pred) |p| p.inner else null }, self.loc()));
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

    /// Build the host-visible TMA descriptor and its coordinate tensor, whose
    /// types are assembled from `config`.
    pub fn makeTiledTmaLoadAtom(self: *Builder, gmem: anytype, smem_layout: anytype, cta_map: anytype, config: TmaConfig) TmaDescriptor {
        const results = self.emitResults(2, cute_nvgpu.atom_make_non_exec_tiled_tma_load(
            self.ctx,
            self.asValue(gmem).inner,
            self.asValue(smem_layout).inner,
            self.asValue(cta_map).inner,
            self.tmaLoadAtomType(config),
            self.coordTensorType(config.coordinate_rank, config.coordinate_layout),
            (cute_nvgpu.TiledTmaLoadAttr.get(self.ctx, .{ .value = .sm_90 }) catch @panic("invalid tiled TMA load")).attribute(),
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
        return (cute_nvgpu.CopyAtomNonExecTiledTmaLoadType.get(self.ctx, .{
            .kind = .sm_90,
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
            .mode = .tiled,
            .numCta = 1,
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

    pub fn prefetchTmaDesc(self: *Builder, descriptor: Atom) void {
        self.emitVoid(cute_nvgpu.prefetch_tma_desc(self.ctx, descriptor.inner, self.loc()));
    }

    pub fn makeExecTma(self: *Builder, descriptor: Atom, result_type: *const mlir.Type) Atom {
        const result = self.emit(cute_nvgpu.atom_make_exec_tma(self.ctx, descriptor.inner, result_type, .{}, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Attach the mbarrier that an executable TMA load atom completes.
    pub fn setTmaBarrier(self: *Builder, atom: Atom, barrier: Value) Atom {
        const result = self.emit(cute_nvgpu.atom_set_value(
            self.ctx,
            atom.inner,
            barrier.inner,
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

    /// Tiled MMA type of the SM100 block-scaled MXFP4 x MXFP8 atom.
    pub fn blockScaledMmaType(self: *Builder, cfg: BlockScaledMmaConfig) *const mlir.Type {
        return (cute_ir.TiledMmaType.get(self.ctx, .{
            .mmaAtom = self.blockScaledAtomType(cfg),
            .atomLayoutMNK = self.layoutAttr(self.layoutSpec(.{ 1, 1, 1 }, .{ 0, 0, 0 })).attribute(),
            .permutationMNK = self.algebraAttr(cute_ir.TileAttr, self.expr(.{ AlgebraToken.all, AlgebraToken.all, AlgebraToken.all })).attribute(),
        }) catch @panic("invalid block-scaled tiled MMA")).type_();
    }

    fn blockScaledAtomType(self: *Builder, cfg: BlockScaledMmaConfig) *const mlir.Type {
        return (cute_nvgpu.MmaAtomSM100UMMABlockScaledType.get(self.ctx, .{
            .shapeMnk = self.algebraAttr(cute_ir.ShapeAttr, self.expr(.{ cfg.m, cfg.n, cfg.k })).attribute(),
            .numCta = cfg.num_cta,
            .aMajor = .k,
            .bMajor = .k,
            .aType = DType.f4e2m1fn.toMlir(self.ctx),
            .bType = DType.f8e4m3fn.toMlir(self.ctx),
            .cType = DType.f32.toMlir(self.ctx),
            .sfType = DType.f8e8m0fnu.toMlir(self.ctx),
            .aFragKind = .smem_desc,
            .vecSize = cfg.vec_size,
        }) catch @panic("invalid block-scaled MMA atom")).type_();
    }

    /// SS, K-major MXFP4 x MXFP8 block-scaled atom and its tiled MMA, as the
    /// CuTe DSL builds them. Both scale-factor pointers start at `scale_ptr`
    /// and are patched after tensor-memory allocation.
    pub fn makeBlockScaledMma(self: *Builder, cfg: BlockScaledMmaConfig, scale_ptr: Value) BlockScaledMma {
        if (cfg.m != 128 or cfg.k != 32 or cfg.n < 8 or cfg.n > 256 or cfg.n % 8 != 0) {
            std.debug.panic("unsupported SM100 block-scaled MMA shape {d}x{d}x{d}", .{ cfg.m, cfg.n, cfg.k });
        }
        const disabled = self.cst(.i1, false);
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

    /// Commit the preceding tcgen05 MMA operations to an mbarrier. One elected
    /// lane of the MMA warp issues this after each K-stage.
    pub fn tcgen05Commit(self: *Builder, barrier: Value) void {
        self.emitVoid(nvvm.tcgen05_commit(self.ctx, self.sharedBarrierPtr(barrier).inner, .cta_1, self.loc()));
    }

    /// Wait until asynchronous tensor-memory loads have reached registers.
    pub fn fenceTmemLoad(self: *Builder) void {
        self.emitVoid(nvvm.tcgen05_wait(self.ctx, .load, self.loc()));
    }

    pub const TmemOptions = struct { two_cta: bool = false, exclusive: bool = false };

    pub fn allocTmem(self: *Builder, columns: u16, holding_ptr: anytype, opts: TmemOptions) void {
        self.emitVoid(cute_nvgpu.arch_sm100_alloc_tmem(
            self.ctx,
            self.cst(.i32, columns).inner,
            self.asValue(holding_ptr).inner,
            .{ .is_two_cta = if (opts.two_cta) .unit(self.ctx) else null, .exclusive = if (opts.exclusive) .unit(self.ctx) else null },
            self.loc(),
        ));
    }

    pub fn retrieveTmemPtr(self: *Builder, holding_ptr: anytype, dtype: DType, alignment: u64) Value {
        const ptr_type = self.ptrTy(dtype, .tmem, alignment) catch @panic("invalid tensor-memory pointer type");
        return self.emit(cute_nvgpu.arch_sm100_retrieve_tmem_ptr(self.ctx, self.asValue(holding_ptr).inner, ptr_type, self.loc()));
    }

    pub fn relinquishTmemAllocPermit(self: *Builder, opts: TmemOptions) void {
        self.emitVoid(cute_nvgpu.arch_sm100_relinquish_tmem_alloc_permit(self.ctx, .{ .is_two_cta = if (opts.two_cta) .unit(self.ctx) else null }, self.loc()));
    }

    pub fn deallocTmem(self: *Builder, pointer: anytype, columns: u16, opts: TmemOptions) void {
        self.emitVoid(cute_nvgpu.arch_sm100_dealloc_tmem(
            self.ctx,
            self.asValue(pointer).inner,
            self.cst(.i32, columns).inner,
            .{ .is_two_cta = if (opts.two_cta) .unit(self.ctx) else null, .exclusive = if (opts.exclusive) .unit(self.ctx) else null },
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

    /// Synchronize a fixed subset of the CTA on a named hardware barrier.
    /// `participants` must be a warp multiple and every participating thread
    /// must execute the same barrier id.
    pub fn namedBarrier(self: *Builder, id: i32, participants: i32) void {
        // The embedded CuTe compiler currently reads NVVM through a different
        // MLIR revision.  A generic `nvvm.barrier` with optional operands is
        // accepted, but those operands are discarded and it lowers to
        // `bar.sync 0`.  Constant inline PTX preserves the named barrier ABI
        // used by CUTLASS (`2,160` for TMEM allocation and `1,128` for the
        // epilogue pipeline).
        const assembly = std.fmt.allocPrint(
            self.arena.allocator(),
            "bar.sync {d}, {d};",
            .{ id, participants },
        ) catch @panic("namedBarrier OOM");
        _ = llvm.inline_asm(self.ctx, assembly, &.{}, .{ .has_side_effects = true }, self.loc()).appendTo(self.currentBlock());
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
            &.{ bits.inner, self.cst(.i32, offset).inner },
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
            &.{ bits.inner, self.liftAs(lane, .i32).inner },
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
    fn sharedBarrierPtr(self: *Builder, barrier: Value) Value {
        if (barrier.isInt()) return self.addressPtr(barrier, .shared);
        return self.emit(mlir.Operation.make(self.ctx, "builtin.unrealized_conversion_cast", .{
            .operands = .{ .flat = &.{barrier.inner} },
            .results = .{ .flat = &.{llvm.pointerType(self.ctx, NVPTXAddressSpace.shared)} },
            .location = self.loc(),
        }));
    }

    /// Initialize a shared-memory mbarrier expecting `arrivals` per phase.
    pub fn mbarrierInit(self: *Builder, barrier: Value, arrivals: u32) void {
        self.emitVoid(nvvm.mbarrier_init(self.ctx, self.sharedBarrierPtr(barrier).inner, self.cst(.i32, arrivals).inner, self.loc()));
    }

    /// Arrive and expect `bytes` of asynchronous transactions (SM90+), as a
    /// TMA producer does before issuing its copies.
    pub fn mbarrierArriveExpectTx(self: *Builder, barrier: Value, bytes: u32) void {
        self.emitVoid(nvvm.mbarrier_arrive_expect_tx(self.ctx, self.sharedBarrierPtr(barrier).inner, self.cst(.i32, bytes).inner, .cta, self.loc()));
    }

    pub fn mbarrierArrive(self: *Builder, barrier: Value, count: u32) void {
        self.emitVoid(nvvm.mbarrier_arrive(self.ctx, self.sharedBarrierPtr(barrier).inner, self.cst(.i32, count).inner, .cta, self.loc()));
    }

    /// Test whether phase `phase` has completed, without suspending (SM90+).
    pub fn mbarrierWaitParity(self: *Builder, barrier: Value, phase: Value) Value {
        return self.emit(nvvm.mbarrier_wait_parity(self.ctx, self.sharedBarrierPtr(barrier).inner, phase.inner, .@"try", .cta, self.loc()));
    }

    /// Wait for phase `phase`, letting the warp sleep for up to
    /// `suspend_time` cycles per attempt (SM90+).
    pub fn mbarrierTryWaitParity(self: *Builder, barrier: Value, phase: Value, suspend_time: u32) void {
        self.emitVoid(nvvm.mbarrier_try_wait_parity(self.ctx, self.sharedBarrierPtr(barrier).inner, phase.inner, self.cst(.i32, suspend_time).inner, self.loc()));
    }

    /// Wait for a pipeline phase with the two-step sequence of CuTe's
    /// `PipelineTmaUmma`: test the barrier without suspending, then suspend
    /// only when the phase is still outstanding. The initial test matters when
    /// a producer reaches a full ring and must yield to the consumer that
    /// recycles its next stage.
    pub fn mbarrierWait(self: *Builder, barrier: Value, phase: Value) void {
        const ready = self.mbarrierWaitParity(barrier, phase);
        var pending = self.openIf(ready.eq(false));
        self.mbarrierTryWaitParity(barrier, phase, 10_000_000);
        pending.yieldThen(.{});
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
    // LLVM operations for them directly (no inline PTX).

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

    /// `bar.sync id, threads`: named barrier over `threads` threads.
    pub fn barrierSync(self: *Builder, id: i32, threads: i32) void {
        _ = self.callIntrinsic("llvm.nvvm.barrier.cta.sync.aligned.count", &.{ self.cst(.i32, id), self.cst(.i32, threads) }, null);
    }

    /// `bar.red.or.pred`: named barrier over `threads` threads that also returns whether
    /// `predicate` (i1) is true in any of them.
    pub fn barrierReduceOr(self: *Builder, id: i32, threads: i32, predicate: Value) Value {
        return self.callIntrinsic("llvm.nvvm.barrier.cta.red.or.aligned.count", &.{ self.cst(.i32, id), self.cst(.i32, threads), predicate }, mlir.Type.int(self.ctx, .i1)).?;
    }

    /// `bar.warp.sync 0xffffffff`.
    pub fn syncWarp(self: *Builder) void {
        self.emitVoid(nvvm.bar_warp_sync(self.ctx, self.cst(.i32, -1).inner, self.loc()));
    }

    /// Non-blocking test of mbarrier phase `parity` (i1).
    pub fn mbarrierTestParity(self: *Builder, barrier: Value, parity: Value) Value {
        return self.emit(nvvm.mbarrier_wait_parity(self.ctx, self.sharedBarrierPtr(barrier).inner, parity.inner, .@"test", .cta, self.loc()));
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
            _ = self.callIntrinsic("llvm.nvvm.cp.async.cg.shared.global.16.s", &.{ d, g, self.select(v, self.cst(.i32, 16), self.cst(.i32, 0)) }, null);
        } else {
            _ = self.callIntrinsic("llvm.nvvm.cp.async.cg.shared.global.16", &.{ d, g }, null);
        }
    }

    /// Arrive on `barrier` once this thread's prior cp.async copies have landed; with
    /// `noinc`, the arrival is counted in the barrier's expected count.
    pub fn cpAsyncMbarrierArrive(self: *Builder, barrier: Value, comptime noinc: bool) void {
        _ = self.callIntrinsic(if (noinc) "llvm.nvvm.cp.async.mbarrier.arrive.noinc.shared" else "llvm.nvvm.cp.async.mbarrier.arrive.shared", &.{self.sharedBarrierPtr(barrier)}, null);
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
        self.emitVoid(nvvm.tcgen05_alloc(self.ctx, self.addressPtr(holder, .shared).inner, self.cst(.i32, columns).inner, self.loc()));
    }

    pub fn tmemDeallocAt(self: *Builder, taddr: Value, columns: u32) void {
        self.emitVoid(nvvm.tcgen05_dealloc(self.ctx, self.addressPtr(taddr, .tensor).inner, self.cst(.i32, columns).inner, self.loc()));
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
        return self.callIntrinsic("llvm.nvvm.redux.sync.fmax", &.{ value, self.cst(.i32, -1) }, mlir.Type.float(self.ctx, .f32)).?;
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

    // ==================== scalars ====================

    pub fn cst(self: *Builder, dt: DType, value: anytype) Value {
        const ctx = self.ctx;
        const T = @TypeOf(value);
        if (isFloatDtype(dt)) {
            const f: f64 = switch (@typeInfo(T)) {
                .comptime_int, .comptime_float => value,
                .int => @floatFromInt(value),
                .bool => @floatFromInt(@intFromBool(value)),
                .float => @floatCast(value),
                else => @compileError("Builder.cst: unsupported value type " ++ @typeName(T)),
            };
            return self.emit(switch (dt) {
                inline .f16, .bf16, .f32, .f64, .f4e2m1fn, .f8e4m3fn, .f8e5m2, .f8e8m0fnu => |ft| arith.constant_float(ctx, f, @field(mlir.FloatTypes, @tagName(ft)), self.loc()),
                else => unreachable,
            });
        }
        const i: i64 = switch (@typeInfo(T)) {
            .comptime_int, .int => @intCast(value),
            .bool => @intFromBool(value),
            .comptime_float, .float => @intFromFloat(value),
            else => @compileError("Builder.cst: unsupported value type " ++ @typeName(T)),
        };
        if (!fitsInt(i, dt)) std.debug.panic("Builder.cst: {d} does not fit {s}", .{ i, @tagName(dt) });
        return self.emit(arith.constant_int(ctx, i, dt.toMlir(ctx), self.loc()));
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
            .comptime_int, .int => if (literalFits(value, .i32)) self.cst(.i32, value) else self.cst(.i64, value),
            .bool => self.cst(.i1, value),
            .comptime_float => self.cst(.f32, value),
            .float => |info| switch (info.bits) {
                16 => self.cst(.f16, value),
                32 => self.cst(.f32, value),
                64 => self.cst(.f64, value),
                else => @compileError("Builder.lift: unsupported float bitwidth"),
            },
            else => @compileError("Builder.lift: unsupported type " ++ @typeName(T)),
        };
    }

    pub fn liftAs(self: *Builder, value: anytype, dt: DType) Value {
        if (@TypeOf(value) == Value) return value;
        return self.cst(dt, value);
    }

    fn liftLike(self: *Builder, value: anytype, ref: Value) Value {
        const dt = ref.dtype();
        return if (literalFits(value, dt)) self.cst(dt, value) else self.lift(value);
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

    pub fn openIf(self: *Builder, cond: Value) IfOnlyScope {
        const then_block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(then_block);
        return .{ .kernel = self, .cond_inner = cond.inner, .then_block = then_block };
    }

    pub fn openIfElse(self: *Builder, cond: Value, result_types: anytype) IfScope(tupleArity(@TypeOf(result_types), "openIfElse: result_types")) {
        const N = comptime tupleArity(@TypeOf(result_types), "openIfElse: result_types");
        var types: [N]*const mlir.Type = undefined;
        inline for (@typeInfo(@TypeOf(result_types)).@"struct".field_names, 0..) |f, i| {
            types[i] = @field(result_types, f);
        }
        const then_block = mlir.Block.init(&.{}, &.{});
        const else_block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(then_block);
        return .{ .kernel = self, .cond_inner = cond.inner, .then_block = then_block, .else_block = else_block, .result_types = types };
    }

    /// `for iv in range(lower, upper, step)` with loop-carried `inits`.
    pub fn openFor(self: *Builder, lower: anytype, upper: anytype, step: anytype, inits: anytype) ForScope(tupleArity(@TypeOf(inits), "openFor: inits")) {
        const N = comptime tupleArity(@TypeOf(inits), "openFor: inits");
        const lb = self.lift(lower);
        const ub = self.liftAs(upper, lb.dtype());
        const st = self.liftAs(step, lb.dtype());

        var block_types: [N + 1]*const mlir.Type = undefined;
        var block_locs: [N + 1]*const mlir.Location = undefined;
        block_types[0] = lb.type_();
        block_locs[0] = self.loc();
        var inits_inner: [N]*const mlir.Value = undefined;
        inline for (@typeInfo(@TypeOf(inits)).@"struct".field_names, 0..) |f, i| {
            const v = self.lift(@field(inits, f));
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
            .lb_inner = lb.inner,
            .ub_inner = ub.inner,
            .step_inner = st.inner,
            .inits_inner = inits_inner,
            .iv = .{ .inner = body.argument(0), .kernel = self },
            .carried = carried,
        };
    }

    /// `scf.while` with explicit loop-carried values. The before region sees
    /// `inits`, calls `yieldBefore(condition, forwarded)`, then the after
    /// region sees `forwarded` and calls `yieldAfter(next_inits)`. This is the
    /// control-flow shape emitted by the Python CuTe persistent scheduler.
    pub fn openWhile(
        self: *Builder,
        inits: anytype,
        after_types: anytype,
    ) WhileScope(
        tupleArity(@TypeOf(inits), "openWhile: inits"),
        tupleArity(@TypeOf(after_types), "openWhile: after_types"),
    ) {
        const N = comptime tupleArity(@TypeOf(inits), "openWhile: inits");
        const M = comptime tupleArity(@TypeOf(after_types), "openWhile: after_types");

        var init_types: [N]*const mlir.Type = undefined;
        var init_locs: [N]*const mlir.Location = undefined;
        var inits_inner: [N]*const mlir.Value = undefined;
        inline for (@typeInfo(@TypeOf(inits)).@"struct".field_names, 0..) |f, i| {
            const v = self.lift(@field(inits, f));
            init_types[i] = v.type_();
            init_locs[i] = self.loc();
            inits_inner[i] = v.inner;
        }

        var result_types: [M]*const mlir.Type = undefined;
        var result_locs: [M]*const mlir.Location = undefined;
        inline for (@typeInfo(@TypeOf(after_types)).@"struct".field_names, 0..) |f, i| {
            result_types[i] = @field(after_types, f);
            result_locs[i] = self.loc();
        }

        const before = mlir.Block.init(&init_types, &init_locs);
        const after = mlir.Block.init(&result_types, &result_locs);
        self.pushBlock(before);
        var before_carried: [N]Value = undefined;
        for (0..N) |i| before_carried[i] = .{ .inner = before.argument(i), .kernel = self };
        return .{
            .kernel = self,
            .before_block = before,
            .after_block = after,
            .inits_inner = inits_inner,
            .after_types = result_types,
            .before_carried = before_carried,
        };
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
        func_op.setAttributeByName("nvvm.reqntid", .denseArray(self.ctx, .i32, &block));
        if (!self.module.operation().verify()) return error.InvalidMlir;

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
    b.namedBarrier(2, 128);
    b.launchDependents();
    b.waitForDependency();
    const j = b.cst(.i32, 127).sub(tid);
    var guard = b.openIf(i.lt(4096));
    var acc = b.openFor(0, 4, 1, .{b.cst(.f32, 0.0)});
    acc.yield(.{acc.carried[0].add(smem.get(.{j}).mul(acc.iv.to(.f32)))});
    a.out.set(.{i}, acc.results[0]);
    guard.yieldThen(.{});

    const kernel = try b.finish(.{ 128, 1, 1 });
    defer std.testing.allocator.free(kernel);
    try expectContains(kernel, "cute_nvgpu.arch.alloc_smem");
    try expectContains(kernel, "!cute.ptr<f32, smem, align<16>>");
    try expectContains(kernel, "\"nvvm.barrier\"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> ()");
    try expectContains(kernel, "bar.sync 2, 128;");
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
    var scope = b.openIfElse(tid.lt(4), .{DType.f16.toMlir(ctx)});
    scope.yieldThen(.{picked.to(.f16)});
    scope.yieldElse(.{b.cst(.f16, 1)});
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
    const one = b.cst(.i32, 1);
    const config = b.makeLaunchConfig(.{
        .grid = .{ one, one, one },
        .block = .{ b.cst(.i32, 128), one, one },
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
    const descriptor = try cute_nvgpu.CopyAtomNonExecTiledTmaLoadType.get(ctx, .{
        .kind = .sm_90,
        .valType = DType.f8e4m3fn.toMlir(ctx),
        .copyBits = 16384,
        .tmaGbasis = b.layoutType(b.layoutSpec(.{ 128, 16, 1 }, .{ b.basis(1, 1, .{1}), b.basis(1, 1, .{0}), b.basis(1, 1, .{2}) })),
        .tmaFormat = .U8,
    });
    _ = try b.declareArgs(.{ .descriptor = .{ .mlir_type = .{ .type_ = descriptor.type_(), .grid_constant = true } } });

    const sf_ptr = b.intToPtr(0, .f8e8m0fnu, .tmem, 1);
    _ = b.makeBlockScaledMma(.{ .n = 16 }, sf_ptr);
    _ = b.makeBlockScaledMma(.{ .n = 128 }, sf_ptr);

    const ir = try b.finish(.{ 192, 1, 1 });
    defer std.testing.allocator.free(ir);
    try std.testing.expect(std.mem.indexOf(u8, ir, "sm100.mma_bs<128x16x32") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "sm100.mma_bs<128x128x32") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "cute.make_tiled_mma") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "cute_nvgpu.grid_constant") != null);

    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}
