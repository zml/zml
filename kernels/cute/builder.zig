const std = @import("std");

const cute = @import("mlir/dialects/cute_ir");
const dialects = @import("mlir/dialects");
const dsl = @import("kernels/common");
const tupleArity = dsl.tupleArity;
const mlir = @import("mlir");
const stdx = @import("stdx");

const arith = dialects.arith;
const cuda = dialects.cuda;
const func = dialects.func;
const gpu = dialects.gpu;
const math = dialects.math;
const vector = dialects.vector;

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

/// `nvvm` is not linked into ZML: its ops are emitted unregistered, in
/// generic form; the CuTe compiler has the dialect.
pub const dialects_needed = [_][]const u8{ "func", "gpu", "cute", "cute_nvgpu", "arith", "scf", "math", "cf", "vector" };

pub const FinishError = error{InvalidMlir} || std.mem.Allocator.Error || std.Io.Writer.Error;

pub const MAX_RANK = 8;
pub const Dims = stdx.BoundedArray(i64, MAX_RANK);

pub const MemorySpace = cute.MemorySpace;

/// A CuTe algebra expression assembled from Zig values. Kernel authors use
/// `Builder.expr`, `Builder.basis`, and `Builder.layoutSpec`; only this DSL
/// layer serializes the expression required by CuTe's C API.
pub const AlgebraExpr = struct {
    text: []const u8,
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

    pub fn payload(self: LayoutSpec, b: *Builder) []const u8 {
        return std.fmt.allocPrint(b.arena.allocator(), "{s}:{s}", .{ self.shape.text, self.stride.text }) catch @panic("OOM");
    }
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
        inline for (std.meta.fields(mlir.FloatTypes)) |f| {
            if (t.isA(mlir.FloatType(@field(mlir.FloatTypes, f.name))) != null) return true;
        }
        return false;
    }

    pub fn isInt(self: Value) bool {
        return self.type_().isA(mlir.IntegerType) != null;
    }

    pub fn isPtr(self: Value) bool {
        return self.type_().isA(cute.PtrType) != null;
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
        return k.emit(cute.memref_load(k.ctx, self.inner, c, self.dtype.toMlir(k.ctx), k.loc()));
    }

    /// `t[coord] = value`; the value is converted to the element type.
    pub fn set(self: Tensor, coord: anytype, value_: anytype) void {
        const k = self.kernel;
        const c = k.makeCoord(coord);
        const v = k.liftAs(value_, self.dtype).to(self.dtype);
        _ = cute.memref_store(k.ctx, self.inner, c, v.inner, k.loc()).appendTo(k.currentBlock());
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
        return self.kernel.emit(cute.memref_load(self.kernel.ctx, self.inner, c, result_dtype.toMlir(self.kernel.ctx), self.kernel.loc()));
    }

    pub fn set(self: View, coord: anytype, value_: Value) void {
        self.kernel.emitVoid(cute.memref_store(self.kernel.ctx, self.inner, self.kernel.makeCoord(coord), value_.inner, self.kernel.loc()));
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
        const result = self.kernel.emit(cute.nvgpu.atom_set_value(
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

/// Attribute payload accepted by target-specific CuTe operations.
pub const Attr = union(enum) {
    str: []const u8,
    text: []const u8,
    raw: *const mlir.Attribute,

    pub fn get(self: Attr, ctx: *mlir.Context) *const mlir.Attribute {
        return switch (self) {
            .str => |s| .string(ctx, s),
            .text => |s| mlir.Attribute.parse(ctx, s) catch std.debug.panic("invalid MLIR attribute: {s}", .{s}),
            .raw => |a| a,
        };
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
    num_multicast: u32 = 1,
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

fn dtypeName(dtype: DType) []const u8 {
    return switch (dtype) {
        .i1 => "i1",
        .i8 => "i8",
        .i16 => "i16",
        .i32 => "i32",
        .i64 => "i64",
        .f16 => "f16",
        .bf16 => "bf16",
        .f32 => "f32",
        .f64 => "f64",
        .f4e2m1fn => "f4E2M1FN",
        .f8e4m3fn => "f8E4M3FN",
        .f8e5m2 => "f8E5M2",
        .f8e8m0fnu => "f8E8M0FNU",
    };
}

fn formatName(format: TmaFormat, dtype: DType) []const u8 {
    return switch (format) {
        .u4_unpack_u8 => "U4_UNPACK_U8",
        .u16 => "U16",
        .default => switch (dtype) {
            .f8e4m3fn => "U8",
            .f32 => "F32_RN",
            else => @panic("TMA format must be specified for this dtype"),
        },
    };
}

fn tmaFormatAttribute(b: *Builder, format: TmaFormat) ?*const mlir.Attribute {
    return switch (format) {
        .default => null,
        .u4_unpack_u8 => b.targetAttribute("#cute_nvgpu.tma_data_format<U4_UNPACK_U8>"),
        .u16 => b.targetAttribute("#cute_nvgpu.tma_data_format<U16>"),
    };
}

fn blockScaledPayload(b: *Builder, cfg: BlockScaledMmaConfig) []const u8 {
    return std.fmt.allocPrint(
        b.arena.allocator(),
        "<{d}x{d}x{d}, num_cta = {d}, ab_major = (k, k), elem_type = (f4E2M1FN, f8E4M3FN, f32), sf_type = f8E8M0FNU, frag_kind = ss, vec_size = {d}>",
        .{ cfg.m, cfg.n, cfg.k, cfg.num_cta, cfg.vec_size },
    ) catch @panic("OOM");
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
        /// A by-value kernel ABI argument such as a CUDA tensor-map atom.
        raw: RawSpec,
    };

    pub const MlirTypeSpec = struct {
        type_: *const mlir.Type,
        grid_constant: bool = false,
    };

    pub const RawSpec = struct {
        type_text: []const u8,
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
        // `nvvm.*` is built unregistered.
        ctx.setAllowUnregisteredDialects(true);
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
        const fields = @typeInfo(Spec).@"struct".fields;

        const arg_specs = try self.arena.allocator().alloc(ArgSpec, fields.len);
        inline for (fields, 0..) |f, i| {
            const raw = @field(spec, f.name);
            const variant = @typeInfo(@TypeOf(raw)).@"struct".fields[0].name;
            const inner = @field(raw, variant);
            arg_specs[i] = .{ .name = f.name, .kind = switch (@field(std.meta.Tag(ArgSpec.Kind), variant)) {
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
                .raw => .{ .raw = .{
                    .type_text = inner.type_text,
                    .grid_constant = if (@hasField(@TypeOf(inner), "grid_constant")) inner.grid_constant else false,
                } },
            } };
        }
        try self.declareArgsLow(arg_specs);

        var named: ArgsOf(Spec) = undefined;
        inline for (fields, 0..) |f, i| {
            @field(named, f.name) = switch (@FieldType(ArgsOf(Spec), f.name)) {
                Tensor => self.tensors[i].?,
                else => self.arg(i),
            };
        }
        return named;
    }

    fn ArgsOf(comptime Spec: type) type {
        const in = @typeInfo(Spec).@"struct".fields;
        comptime var names: [in.len][]const u8 = undefined;
        comptime var types: [in.len]type = undefined;
        inline for (in, 0..) |f, i| {
            names[i] = f.name;
            types[i] = if (@hasField(f.type, "tensor")) Tensor else Value;
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
                .raw => |raw| self.parseType(raw.type_text),
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
                .raw => |raw| if (raw.grid_constant) .dict(ctx, &.{
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
                .visibility = null,
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
                .visibility = null,
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
                .raw => null,
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

    pub fn setFunctionAttribute(self: *Builder, name: []const u8, attribute: *const mlir.Attribute) void {
        const function = self.func_op orelse @panic("setFunctionAttribute called before declareArgs");
        function.setAttributeByName(name, attribute);
    }

    // ==================== cuda host launch ====================

    pub fn kernelSmemSize(self: *Builder, kernel_name: []const u8) Value {
        const symbol = std.fmt.allocPrint(self.arena.allocator(), "@kernels::@{s}", .{kernel_name}) catch @panic("OOM");
        return self.emit(cute.kernel_smem_size(
            self.ctx,
            .int(self.ctx, .i64),
            self.parseAttribute(symbol),
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
    /// The segment attribute is written explicitly because the CUDA dialect is
    /// supplied by the runtime CuTe compiler and is intentionally unregistered
    /// in ZML's lightweight construction context.
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
        const symbol = std.fmt.allocPrint(self.arena.allocator(), "@kernels::@{s}", .{kernel_name}) catch @panic("OOM");
        return self.emit(cuda.launch_ex(self.ctx, .{
            .config = config.inner,
            .inputs = values,
            .callee = self.parseAttribute(symbol),
            .assume_kernel_attr = self.targetAttribute("#cuda.assume_kernel_attr<true>"),
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
        return self.renderModule(true);
    }

    const target_attr_marker = "__zml_cute_target_attr__";

    fn renderModule(self: *Builder, generic: bool) FinishError![:0]const u8 {
        var printed: std.Io.Writer.Allocating = .init(self.allocator);
        defer printed.deinit();
        // Generic form is stable across the lightweight construction dialect
        // and NVIDIA's runtime compiler dialect. In particular, the two
        // versions currently use different custom assembly printers for
        // `cute.make_composed_layout`.
        if (generic) {
            try printed.writer.print("{f}", .{self.module.operation().fmt(.{ .print_generic_op_form = true })});
        } else {
            try printed.writer.print("{f}", .{self.module.operation()});
        }

        // A target attribute is initially a quoted StringAttr so the local
        // verifier can carry it through an AnyAttr field. Replace the whole
        // quoted string with its CuTe assembly before handing the module to
        // the real compiler, which owns the attribute parser.
        var output: std.Io.Writer.Allocating = .init(self.allocator);
        defer output.deinit();
        var rest = printed.written();
        while (std.mem.indexOf(u8, rest, target_attr_marker)) |marker| {
            if (marker == 0 or rest[marker - 1] != '"') return error.InvalidMlir;
            try output.writer.writeAll(rest[0 .. marker - 1]);
            const payload = rest[marker + target_attr_marker.len ..];
            const end = std.mem.indexOfScalar(u8, payload, '"') orelse return error.InvalidMlir;
            try output.writer.writeAll(payload[0..end]);
            rest = payload[end + 1 ..];
        }
        try output.writer.writeAll(rest);
        return try self.allocator.dupeZ(u8, output.written());
    }

    // ==================== types ====================

    pub fn ptrTy(self: *Builder, dt: DType, space: MemorySpace, alignment: u64) !*const mlir.Type {
        return cute.pointerType(self.ctx, dt.toMlir(self.ctx), space, alignment);
    }

    pub fn swizzledPtrTy(self: *Builder, dt: DType, space: MemorySpace, alignment: u64, swizzle: Swizzle) !*const mlir.Type {
        const text = std.fmt.allocPrint(self.arena.allocator(), "S<{d},{d},{d}>", .{ swizzle.bits, swizzle.base, swizzle.shift }) catch @panic("OOM");
        const attr = try cute.SwizzleAttr.get(self.ctx, text);
        return (try cute.PtrType.get(self.ctx, .{
            .valueType = dt.toMlir(self.ctx),
            .memorySpace = .string(self.ctx, @tagName(space)),
            .alignment = alignment,
            .swizzle = attr.attribute(),
        })).type_();
    }

    pub fn intToPtr(self: *Builder, address: anytype, dt: DType, space: MemorySpace, alignment: u64) Value {
        const ptr_type = self.ptrTy(dt, space, alignment) catch @panic("invalid CuTe pointer type");
        return self.emit(cute.inttoptr(self.ctx, self.lift(address).inner, ptr_type, self.loc()));
    }

    /// Reinterpret a pointer without changing its address. This is the typed
    /// counterpart of Python CuTe's `recast_ptr`; packed MXFP4 buffers use it
    /// to expose their byte storage to a scalar correctness implementation.
    pub fn recastPointer(self: *Builder, pointer: anytype, dt: DType, space: MemorySpace, alignment: u64) Value {
        const result_type = self.ptrTy(dt, space, alignment) catch @panic("invalid recast pointer type");
        return self.emit(cute.recast_iter(self.ctx, self.asValue(pointer).inner, result_type, self.loc()));
    }

    pub fn recastSwizzledPointer(self: *Builder, pointer: anytype, dt: DType, space: MemorySpace, alignment: u64, swizzle: Swizzle) Value {
        const result_type = self.swizzledPtrTy(dt, space, alignment, swizzle) catch @panic("invalid swizzled pointer type");
        return self.emit(cute.recast_iter(self.ctx, self.asValue(pointer).inner, result_type, self.loc()));
    }

    /// Parse a CuTe/CuTe-NVGPU type that has no compact C-API constructor.
    /// This is primarily used for architecture-specific atom and descriptor
    /// types whose complete layout is part of the type itself.
    pub fn parseType(self: *Builder, text: []const u8) *const mlir.Type {
        return mlir.Type.parse(self.ctx, text) catch std.debug.panic("invalid MLIR type: {s}", .{text});
    }

    /// Parse a target-specific MLIR attribute.
    pub fn parseAttribute(self: *Builder, text: []const u8) *const mlir.Attribute {
        return mlir.Attribute.parse(self.ctx, text) catch std.debug.panic("invalid MLIR attribute: {s}", .{text});
    }

    /// Preserve an attribute owned by the runtime CuTe compiler when the
    /// lightweight ZML MLIR context has no parser hook for that attribute.
    /// It is represented as a verifier-safe string while constructing the
    /// module and restored to target assembly by `finish`/`finishProgram`.
    pub fn targetAttribute(self: *Builder, text: []const u8) *const mlir.Attribute {
        if (std.mem.indexOfScalar(u8, text, '"') != null) @panic("targetAttribute cannot contain a quote");
        const encoded = std.fmt.allocPrint(self.arena.allocator(), "__zml_cute_target_attr__{s}", .{text}) catch @panic("OOM");
        return .string(self.ctx, encoded);
    }

    fn algebra(self: *Builder, dims: []const i64) []const u8 {
        var out: std.Io.Writer.Allocating = .init(self.arena.allocator());
        if (dims.len == 1) {
            out.writer.print("{d}", .{dims[0]}) catch @panic("OOM");
        } else {
            out.writer.writeByte('(') catch @panic("OOM");
            for (dims, 0..) |d, i| {
                if (i > 0) out.writer.writeByte(',') catch @panic("OOM");
                out.writer.print("{d}", .{d}) catch @panic("OOM");
            }
            out.writer.writeByte(')') catch @panic("OOM");
        }
        return out.written();
    }

    fn writeAlgebra(self: *Builder, writer: *std.Io.Writer, value: anytype) void {
        const T = @TypeOf(value);
        if (T == AlgebraExpr) {
            writer.writeAll(value.text) catch @panic("OOM");
            return;
        }
        if (T == AlgebraToken) {
            writer.writeByte(switch (value) {
                .all => '_',
                .dynamic => '?',
            }) catch @panic("OOM");
            return;
        }
        if (T == ScaledBasis) {
            writer.print("{d}", .{value.numerator}) catch @panic("OOM");
            if (value.denominator != 1) writer.print("/{d}", .{value.denominator}) catch @panic("OOM");
            for (value.modes[0..value.mode_count]) |mode| writer.print("@{d}", .{mode}) catch @panic("OOM");
            return;
        }
        if (T == ConstrainedDynamic) {
            if (value.divisible_by == 0) @panic("CuTe constrained dynamic divisor must be positive");
            writer.print("?{{div={d}}}", .{value.divisible_by}) catch @panic("OOM");
            return;
        }
        switch (@typeInfo(T)) {
            .comptime_int, .int => writer.print("{d}", .{value}) catch @panic("OOM"),
            .@"struct" => |info| {
                writer.writeByte('(') catch @panic("OOM");
                inline for (info.fields, 0..) |field, i| {
                    if (i != 0) writer.writeByte(',') catch @panic("OOM");
                    self.writeAlgebra(writer, @field(value, field.name));
                }
                writer.writeByte(')') catch @panic("OOM");
            },
            else => @compileError("unsupported CuTe algebra value " ++ @typeName(T)),
        }
    }

    /// Build a CuTe algebra expression from integers, nested Zig tuples,
    /// `AlgebraToken`s and `ScaledBasis` values.
    pub fn expr(self: *Builder, value: anytype) AlgebraExpr {
        var out: std.Io.Writer.Allocating = .init(self.arena.allocator());
        self.writeAlgebra(&out.writer, value);
        return .{ .text = out.written() };
    }

    /// `coefficient @ mode...`, including rational coefficients such as
    /// `1/2@0`. The mode path maps a coordinate into a nested CuTe mode.
    pub fn basis(self: *Builder, numerator: i64, denominator: i64, modes: anytype) ScaledBasis {
        _ = self;
        if (denominator <= 0) @panic("CuTe scaled-basis denominator must be positive");
        const fields = @typeInfo(@TypeOf(modes)).@"struct".fields;
        if (fields.len == 0 or fields.len > MAX_RANK) @panic("CuTe scaled basis needs 1..MAX_RANK modes");
        var result: ScaledBasis = .{ .numerator = numerator, .denominator = denominator, .mode_count = fields.len };
        inline for (fields, 0..) |field, i| result.modes[i] = @intCast(@field(modes, field.name));
        return result;
    }

    pub fn layoutSpec(self: *Builder, shape: anytype, stride: anytype) LayoutSpec {
        return .{ .shape = self.expr(shape), .stride = self.expr(stride) };
    }

    pub fn layoutType(self: *Builder, spec: LayoutSpec) *const mlir.Type {
        return cute.algebraType(self.ctx, .layout, spec.payload(self)) catch @panic("invalid CuTe layout");
    }

    pub fn composedLayoutType(self: *Builder, swizzle: Swizzle, offset: anytype, outer: LayoutSpec) *const mlir.Type {
        const payload = std.fmt.allocPrint(
            self.arena.allocator(),
            "S<{d},{d},{d}> o {s} o {s}",
            .{ swizzle.bits, swizzle.base, swizzle.shift, self.expr(offset).text, outer.payload(self) },
        ) catch @panic("OOM");
        return cute.algebraType(self.ctx, .composed_layout, payload) catch @panic("invalid CuTe composed layout");
    }

    pub fn memrefType(self: *Builder, dtype: DType, space: MemorySpace, alignment: u64, layout_type: *const mlir.Type) *const mlir.Type {
        const pointer = self.ptrTy(dtype, space, alignment) catch @panic("invalid CuTe pointer type");
        return (cute.MemRefType.get(self.ctx, .{ .ptr = pointer, .layout = layout_type }) catch @panic("invalid CuTe memref type")).type_();
    }

    pub fn memrefTypeFromPointer(self: *Builder, pointer_type: *const mlir.Type, layout_type: *const mlir.Type) *const mlir.Type {
        return (cute.MemRefType.get(self.ctx, .{ .ptr = pointer_type, .layout = layout_type }) catch @panic("invalid CuTe memref type")).type_();
    }

    pub fn staticLayout(self: *Builder, spec: LayoutSpec) View {
        const result = self.emit(cute.static(self.ctx, self.layoutType(spec), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Materialize a rational scaled-basis layout used by block-scale TMA.
    /// The local standalone LayoutAttr constructor only accepts integer
    /// strides, while CoordTensorType owns the more general coordinate
    /// algebra. Recovering its layout through the typed C API avoids exposing
    /// that parser distinction to kernel code.
    pub fn staticCoordinateLayout(self: *Builder, rank: usize, spec: LayoutSpec) View {
        const tensor_type = self.coordTensorType(rank, spec);
        const coords = tensor_type.isA(cute.CoordTensorType) orelse @panic("invalid coordinate tensor type");
        const result = self.emit(cute.static(self.ctx, coords.getLayout(), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub const Swizzle = struct { bits: i64, base: i64, shift: i64 };

    pub fn staticSwizzle(self: *Builder, spec: Swizzle) View {
        const payload = std.fmt.allocPrint(self.arena.allocator(), "S<{d},{d},{d}>", .{ spec.bits, spec.base, spec.shift }) catch @panic("OOM");
        const ty = cute.algebraType(self.ctx, .swizzle, payload) catch @panic("invalid CuTe swizzle");
        const result = self.emit(cute.static(self.ctx, ty, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Construct `swizzle o offset o outer_layout` through the CuTe dialect.
    pub fn makeComposedLayout(self: *Builder, swizzle: Swizzle, offset: anytype, outer: LayoutSpec) View {
        const inner = self.staticSwizzle(swizzle);
        const offset_expr = self.expr(offset);
        const offset_ty = cute.algebraType(self.ctx, .int_tuple, offset_expr.text) catch @panic("invalid CuTe composed-layout offset");
        const offset_value = self.emit(cute.make_int_tuple(self.ctx, &.{}, offset_ty, self.loc()));
        const outer_value = self.staticLayout(outer);
        const result_ty = self.composedLayoutType(swizzle, offset, outer);
        const result = self.emit(cute.make_composed_layout(self.ctx, inner.inner, offset_value.inner, outer_value.inner, result_ty, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Compose a raw pointer and any CuTe layout value. The memref result type
    /// is obtained through the dialect C API rather than parsed from assembly.
    pub fn makeTensorView(self: *Builder, pointer: anytype, layout: anytype) View {
        const p = self.asValue(pointer);
        const l = self.asValue(layout);
        const result_ty = (cute.MemRefType.get(self.ctx, .{ .ptr = p.type_(), .layout = l.type_() }) catch @panic("invalid CuTe tensor view")).type_();
        const result = self.emit(cute.make_view(self.ctx, p.inner, l.inner, result_ty, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn coordTensorType(self: *Builder, rank: usize, layout: LayoutSpec) *const mlir.Type {
        if (rank == 0 or rank > MAX_RANK) @panic("coordTensorType rank must be in 1..MAX_RANK");
        var zeros: std.Io.Writer.Allocating = .init(self.arena.allocator());
        if (rank > 1) zeros.writer.writeByte('(') catch @panic("OOM");
        for (0..rank) |i| {
            if (i != 0) zeros.writer.writeByte(',') catch @panic("OOM");
            zeros.writer.writeByte('0') catch @panic("OOM");
        }
        if (rank > 1) zeros.writer.writeByte(')') catch @panic("OOM");
        const tuple = cute.algebraType(self.ctx, .int_tuple, zeros.written()) catch @panic("invalid coordinate tuple");
        const payload = layout.payload(self);
        // The standalone LayoutAttr C API intentionally restricts scaled
        // basis coefficients to integers. CoordTensorType additionally
        // accepts the rational basis used by block-scaled scale-factor maps.
        // Keep that parser workaround inside the DSL type constructor.
        if (std.mem.indexOfScalar(u8, payload, '/') != null) {
            const type_text = std.fmt.allocPrint(self.arena.allocator(), "!cute.coord_tensor<\"{s}\", \"{s}\">", .{ zeros.written(), payload }) catch @panic("OOM");
            return self.parseType(type_text);
        }
        return (cute.CoordTensorType.get(self.ctx, .{ .arithTuple = tuple, .layout = self.layoutType(layout) }) catch @panic("invalid coordinate tensor")).type_();
    }

    /// Coordinate-tensor type whose coordinate tuple contains dynamic and
    /// static leaves. `tuple_payload` is CuTe algebra such as `(0,?,?)`;
    /// layouts remain structured `LayoutSpec` values in Zig.
    pub fn coordTensorTypePayload(self: *Builder, tuple_payload: []const u8, layout: LayoutSpec) *const mlir.Type {
        const attr = cute.IntTupleAttr.get(self.ctx, tuple_payload) catch std.debug.panic("invalid coordinate tuple {s}", .{tuple_payload});
        const tuple = (cute.IntTupleType.get(self.ctx, .{ .attr = attr.attribute() }) catch @panic("invalid coordinate tuple type")).type_();
        return (cute.CoordTensorType.get(self.ctx, .{ .arithTuple = tuple, .layout = self.layoutType(layout) }) catch @panic("invalid coordinate tensor")).type_();
    }

    /// Coordinate-tensor type with an explicit arithmetic-tuple expression.
    /// This represents the divisibility information produced by CuTe tiling
    /// without requiring kernel code to spell textual MLIR types.
    pub fn coordTensorTypeWithTuple(self: *Builder, tuple: anytype, layout: LayoutSpec) *const mlir.Type {
        const tuple_text = self.expr(tuple).text;
        const tuple_type = cute.algebraType(self.ctx, .int_tuple, tuple_text) catch std.debug.panic("invalid coordinate tuple {s}", .{tuple_text});
        return (cute.CoordTensorType.get(self.ctx, .{ .arithTuple = tuple_type, .layout = self.layoutType(layout) }) catch @panic("invalid coordinate tensor")).type_();
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

    /// Materialize a value whose complete static CuTe type is already known.
    ///
    /// Nested layouts and integer tuples produced by CuTe's layout algebra
    /// cannot be represented by the flat `Layout` helper.  Keeping this
    /// escape hatch typed still lets MLIR verify every consumer.
    pub fn staticValue(self: *Builder, type_text: []const u8) View {
        const result = self.emit(cute.static(self.ctx, self.parseType(type_text), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// `cute.make_layout(shape, stride=...)`, static only.
    pub fn makeLayout(self: *Builder, shape: []const i64, opts: LayoutOpts) Layout {
        if (shape.len == 0 or shape.len > MAX_RANK) std.debug.panic("makeLayout: rank {d} not in 1..{d}", .{ shape.len, MAX_RANK });
        const stride = opts.stride orelse self.leftMost(shape);
        if (stride.len != shape.len) std.debug.panic("makeLayout: stride rank {d} != shape rank {d}", .{ stride.len, shape.len });
        const ctx = self.ctx;
        const text = std.fmt.allocPrint(self.arena.allocator(), "{s}:{s}", .{ self.algebra(shape), self.algebra(stride) }) catch @panic("OOM");
        const layout_ty = cute.algebraType(ctx, .layout, text) catch @panic("bad layout");
        const l = self.emit(cute.static(ctx, layout_ty, self.loc()));
        return .{ .inner = l.inner, .shape = Dims.fromSlice(shape) catch unreachable, .stride = Dims.fromSlice(stride) catch unreachable };
    }

    /// `cute.make_tensor(ptr, layout)`.
    pub fn makeTensor(self: *Builder, ptr: Value, layout: Layout) Tensor {
        const pt = ptr.type_().isA(cute.PtrType) orelse std.debug.panic("makeTensor: {f} is not a !cute.ptr", .{ptr.type_()});
        const elem = pt.getValueType() orelse @panic("makeTensor: untyped pointer");
        const memref_ty = (cute.MemRefType.get(self.ctx, .{ .ptr = ptr.type_(), .layout = layout.inner.type_() }) catch @panic("bad memref")).type_();
        const v = self.emit(cute.make_view(self.ctx, ptr.inner, layout.inner, memref_ty, self.loc()));
        return .{ .inner = v.inner, .kernel = self, .layout = layout, .dtype = dtypes.mlirToDType(self.ctx, elem), .space = pt.getAddressSpace() };
    }

    fn writeCoord(self: *Builder, writer: *std.Io.Writer, dyn: *stdx.BoundedArray(*const mlir.Value, MAX_RANK), coord: anytype) void {
        const T = @TypeOf(coord);
        if (T == Value) {
            writer.writeByte('?') catch @panic("OOM");
            dyn.append(self.lift(coord).to(.i32).inner) catch @panic("makeCoord: too many dynamic leaves");
            return;
        }
        if (T == AlgebraToken) {
            writer.writeByte(switch (coord) {
                .all => '_',
                .dynamic => @panic("makeCoord: AlgebraToken.dynamic needs an SSA value"),
            }) catch @panic("OOM");
            return;
        }
        switch (@typeInfo(T)) {
            .comptime_int, .int => writer.print("{d}", .{coord}) catch @panic("OOM"),
            .@"struct" => |info| {
                if (info.fields.len == 0) @compileError("makeCoord: empty tuple");
                if (info.fields.len > 1) writer.writeByte('(') catch @panic("OOM");
                inline for (info.fields, 0..) |field, i| {
                    if (i != 0) writer.writeByte(',') catch @panic("OOM");
                    self.writeCoord(writer, dyn, @field(coord, field.name));
                }
                if (info.fields.len > 1) writer.writeByte(')') catch @panic("OOM");
            },
            else => @compileError("makeCoord: unsupported coordinate leaf " ++ @typeName(T)),
        }
    }

    /// A `!cute.coord` from nested tuples of `Value`s, integers, and `.all`
    /// placeholders. Integer leaves stay static in the dependent CuTe type.
    pub fn makeCoord(self: *Builder, coord: anytype) *const mlir.Value {
        if (@TypeOf(coord) == Value) return self.makeCoord(.{coord});
        var text: std.Io.Writer.Allocating = .init(self.arena.allocator());
        var dyn: stdx.BoundedArray(*const mlir.Value, MAX_RANK) = .empty;
        self.writeCoord(&text.writer, &dyn, coord);
        const ty = cute.algebraType(self.ctx, .coord, text.written()) catch @panic("bad coord");
        return self.emit(cute.make_coord(self.ctx, dyn.constSlice(), ty, self.loc())).inner;
    }

    /// A dynamic/static CuTe integer tuple, used for pointer arithmetic in
    /// tensor and shared memory. It accepts the same nested Zig values as a
    /// coordinate, but preserves integer-tuple semantics.
    pub fn makeIntTuple(self: *Builder, values: anytype) View {
        var text: std.Io.Writer.Allocating = .init(self.arena.allocator());
        var dyn: stdx.BoundedArray(*const mlir.Value, MAX_RANK) = .empty;
        self.writeCoord(&text.writer, &dyn, values);
        const ty = cute.algebraType(self.ctx, .int_tuple, text.written()) catch @panic("bad integer tuple");
        const value = self.emit(cute.make_int_tuple(self.ctx, dyn.constSlice(), ty, self.loc()));
        return .{ .inner = value.inner, .kernel = self };
    }

    /// Refine an integer SSA value with the divisibility information CuTe
    /// derives while slicing a tiled coordinate tensor.
    pub fn assumeDivBy(self: *Builder, value: anytype, comptime divisor: u32) Value {
        if (divisor == 0) @compileError("assumeDivBy divisor must be non-zero");
        const type_text = std.fmt.allocPrint(self.arena.allocator(), "!cute.i32<divby {d}>", .{divisor}) catch @panic("OOM");
        return self.emit(cute.assume(self.ctx, self.lift(value).to(.i32).inner, self.parseType(type_text), self.loc()));
    }

    /// Build a coordinate tensor with an explicit dependent arithmetic-tuple
    /// type. This is the source-level equivalent of the tuple produced by
    /// `slice` after TMA partitioning (for example `?{div=128}`).
    pub fn makeCoordTensorWithTuple(self: *Builder, values: anytype, tuple_payload: []const u8, layout: anytype) View {
        const tuple_text = std.fmt.allocPrint(self.arena.allocator(), "!cute.int_tuple<\"{s}\">", .{tuple_payload}) catch @panic("OOM");
        const tuple_type = self.parseType(tuple_text);
        const tuple = self.emit(cute.make_int_tuple(self.ctx, self.innerSlice(values), tuple_type, self.loc()));
        const iterator_type = (cute.ArithTupleIteratorType.get(self.ctx, .{ .arithTuple = tuple_type }) catch @panic("invalid arithmetic tuple iterator")).type_();
        const iterator = self.emit(cute.make_arith_tuple_iter(self.ctx, tuple.inner, iterator_type, self.loc()));
        const layout_value = self.asValue(layout);
        const result_type = (cute.CoordTensorType.get(self.ctx, .{ .arithTuple = tuple_type, .layout = layout_value.type_() }) catch @panic("invalid coordinate tensor")).type_();
        const result = self.emit(cute.make_view(self.ctx, iterator.inner, layout_value.inner, result_type, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Build a coordinate tensor from a runtime origin and a static layout.
    /// This is the direct form of the `get_iter`/`make_arith_tuple_iter`/
    /// `make_view` sequence emitted by the Python CuTe front end after a tile
    /// is selected from a larger coordinate tensor.
    pub fn makeCoordTensor(self: *Builder, origin: anytype, layout: anytype) View {
        const tuple = self.makeIntTuple(origin);
        const layout_value = self.asValue(layout);
        const iterator_type = (cute.ArithTupleIteratorType.get(self.ctx, .{ .arithTuple = tuple.type_() }) catch @panic("invalid arithmetic tuple iterator")).type_();
        const iterator = self.emit(cute.make_arith_tuple_iter(self.ctx, tuple.inner, iterator_type, self.loc()));
        const result_type = (cute.CoordTensorType.get(self.ctx, .{ .arithTuple = tuple.type_(), .layout = layout_value.type_() }) catch @panic("invalid coordinate tensor")).type_();
        const result = self.emit(cute.make_view(self.ctx, iterator.inner, layout_value.inner, result_type, self.loc()));
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
        tuple_payload: []const u8,
        comptime leaf_divisibility: [leaf_count]u32,
        layout: anytype,
    ) View {
        if (leaf_count == 0 or leaf_count > MAX_RANK) @panic("invalid coordinate tuple leaf count");
        const source = self.asValue(input);
        const source_type = source.type_().isA(cute.CoordTensorType) orelse
            std.debug.panic("rebuildCoordTensorIterator: expected coordinate tensor, got {f}", .{source.type_()});
        const source_tuple_type = source_type.getArithTuple();
        const source_iterator_type = (cute.ArithTupleIteratorType.get(self.ctx, .{ .arithTuple = source_tuple_type }) catch
            @panic("invalid source arithmetic tuple iterator")).type_();
        const source_iterator = self.emit(cute.get_iter(self.ctx, source.inner, source_iterator_type, self.loc()));
        const source_tuple = self.emit(cute.deref_arith_tuple_iter(
            self.ctx,
            source_iterator.inner,
            source_tuple_type,
            self.loc(),
        ));

        const leaf_types = self.arena.allocator().alloc(*const mlir.Type, leaf_count) catch @panic("OOM");
        inline for (leaf_divisibility, 0..) |divisor, i| {
            const leaf_payload = if (divisor == 1)
                "?"
            else
                std.fmt.allocPrint(self.arena.allocator(), "?{{div={d}}}", .{divisor}) catch @panic("OOM");
            leaf_types[i] = cute.algebraType(self.ctx, .int_tuple, leaf_payload) catch
                std.debug.panic("invalid dynamic integer tuple leaf {s}", .{leaf_payload});
        }
        const leaves = self.emitResults(leaf_count, cute.get_leaves(
            self.ctx,
            source_tuple.inner,
            leaf_types,
            self.loc(),
        ));

        const rebuilt_tuple_type = cute.algebraType(self.ctx, .int_tuple, tuple_payload) catch
            std.debug.panic("invalid rebuilt coordinate tuple {s}", .{tuple_payload});
        const rebuilt_tuple = self.emit(cute.make_int_tuple(
            self.ctx,
            self.innerSlice(leaves[0..]),
            rebuilt_tuple_type,
            self.loc(),
        ));
        const rebuilt_iterator_type = (cute.ArithTupleIteratorType.get(self.ctx, .{ .arithTuple = rebuilt_tuple_type }) catch
            @panic("invalid rebuilt arithmetic tuple iterator")).type_();
        const rebuilt_iterator = self.emit(cute.make_arith_tuple_iter(
            self.ctx,
            rebuilt_tuple.inner,
            rebuilt_iterator_type,
            self.loc(),
        ));
        const layout_value = self.asValue(layout);
        const result_type = (cute.CoordTensorType.get(self.ctx, .{
            .arithTuple = rebuilt_tuple_type,
            .layout = layout_value.type_(),
        }) catch @panic("invalid rebuilt coordinate tensor")).type_();
        const result = self.emit(cute.make_view(
            self.ctx,
            rebuilt_iterator.inner,
            layout_value.inner,
            result_type,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// A coordinate made only of CuTe placeholders, such as `(_,_,_)`.
    /// Placeholders select every element of a mode and therefore cannot be
    /// represented by Zig integer values passed to `makeCoord`.
    pub fn staticCoord(self: *Builder, payload: []const u8) View {
        const type_text = std.fmt.allocPrint(self.arena.allocator(), "!cute.coord<\"{s}\">", .{payload}) catch @panic("OOM");
        const result = self.emit(cute.make_coord(self.ctx, &.{}, self.parseType(type_text), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Static `cute.make_shape`; `payload` is CuTe algebra such as
    /// `((128,16),1,1,2)`. Dynamic leaves can be added when a kernel needs
    /// them, but all current SM100 instruction tiles are static.
    pub fn makeShape(self: *Builder, payload: []const u8) View {
        const ty_text = std.fmt.allocPrint(self.arena.allocator(), "!cute.shape<\"{s}\">", .{payload}) catch @panic("OOM");
        const value = self.emit(cute.make_shape(self.ctx, &.{}, self.parseType(ty_text), self.loc()));
        return .{ .inner = value.inner, .kernel = self };
    }

    /// Dynamic `cute.make_shape`. The result type describes which leaves are
    /// runtime values (for example `!cute.shape<"(?,?,?)">`). Keeping these
    /// leaves dynamic is significant to CuTe's TMA layout algebra when a
    /// runtime extent happens to be one.
    pub fn makeShapeValues(self: *Builder, values: anytype, result_type: *const mlir.Type) View {
        const value = self.emit(cute.make_shape(self.ctx, self.innerSlice(values), result_type, self.loc()));
        return .{ .inner = value.inner, .kernel = self };
    }

    pub fn tileToShapeTyped(self: *Builder, input: anytype, shape: anytype, order: ?View, result_type: *const mlir.Type) View {
        const result = self.emit(cute.tile_to_shape(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(shape).inner,
            if (order) |value| value.inner else null,
            result_type,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Static `cute.make_tile` from mode extents.
    pub fn makeTile(self: *Builder, shape: []const i64) View {
        var payload: std.Io.Writer.Allocating = .init(self.arena.allocator());
        payload.writer.writeByte('[') catch @panic("OOM");
        for (shape, 0..) |extent, i| {
            if (i != 0) payload.writer.writeByte(';') catch @panic("OOM");
            payload.writer.print("{d}:1", .{extent}) catch @panic("OOM");
        }
        payload.writer.writeByte(']') catch @panic("OOM");
        const ty_text = std.fmt.allocPrint(self.arena.allocator(), "!cute.tile<\"{s}\">", .{payload.written()}) catch @panic("OOM");
        const value = self.emit(cute.make_tile(self.ctx, &.{}, self.parseType(ty_text), self.loc()));
        return .{ .inner = value.inner, .kernel = self };
    }

    /// Construct a target atom whose complete architecture payload is
    /// encoded in `atom_type` (for example `!cute_nvgpu.sm100.mma_bs<...>`).
    pub fn makeAtom(self: *Builder, atom_type: []const u8, values: anytype) Atom {
        const result = self.emit(cute.make_atom(self.ctx, self.innerSlice(values), self.parseType(atom_type), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn makeTiledMma(self: *Builder, atom: Atom, result_type: []const u8) Atom {
        const result = self.emit(cute.make_tiled_mma(self.ctx, atom.inner, self.parseType(result_type), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Allocate a statically typed shared-memory tensor. This is used for
    /// swizzled layouts which cannot be represented by `Layout` alone.
    pub fn allocSmemView(self: *Builder, memref_type: []const u8) View {
        const result = self.emit(cute.memref_alloc_smem(self.ctx, self.parseType(memref_type), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Allocate shared memory with a type assembled by the Zig DSL.  This is
    /// the non-string counterpart of `allocSmemView` used by source kernels.
    pub fn allocSmemTyped(self: *Builder, dtype: DType, alignment: u64, layout_type: *const mlir.Type) View {
        const memref_type = self.memrefType(dtype, .smem, alignment, layout_type);
        const result = self.emit(cute.memref_alloc_smem(self.ctx, memref_type, self.loc()));
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
    pub fn allocRmemTyped(self: *Builder, dtype: DType, alignment: u64, layout: anytype) View {
        const layout_value = self.asValue(layout);
        const memref_type = self.memrefType(dtype, .rmem, alignment, layout_value.type_());
        const result = self.emit(cute.memref_alloca(self.ctx, layout_value.inner, memref_type, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn makeView(self: *Builder, pointer: anytype, layout: anytype, result_type: []const u8) View {
        const result = self.emit(cute.make_view(
            self.ctx,
            self.asValue(pointer).inner,
            self.asValue(layout).inner,
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn makeViewTyped(self: *Builder, pointer: anytype, layout: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute.make_view(
            self.ctx,
            self.asValue(pointer).inner,
            self.asValue(layout).inner,
            result_type,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn makeViewNoLayout(self: *Builder, pointer: anytype, result_type: []const u8) View {
        const result = self.emit(cute.make_view(self.ctx, self.asValue(pointer).inner, null, self.parseType(result_type), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn localTile(self: *Builder, input: anytype, tile: anytype, coord: anytype, result_type: []const u8, projection: ?Attr) View {
        const coord_value = if (@TypeOf(coord) == View) coord.inner else self.makeCoord(coord);
        const result = self.emit(cute.local_tile(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(tile).inner,
            coord_value,
            self.parseType(result_type),
            if (projection) |p| p.get(self.ctx) else null,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn localTileTyped(self: *Builder, input: anytype, tile: anytype, coord: anytype, result_type: *const mlir.Type, projection: ?Attr) View {
        const coord_value = if (@TypeOf(coord) == View) coord.inner else self.makeCoord(coord);
        const result = self.emit(cute.local_tile(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(tile).inner,
            coord_value,
            result_type,
            if (projection) |p| p.get(self.ctx) else null,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn localPartition(self: *Builder, input: anytype, tiler: anytype, index: anytype, result_type: []const u8) View {
        const result = self.emit(cute.local_partition(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(tiler).inner,
            self.makeCoord(index),
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Apply CuTe layout algebra while keeping the compiler-derived result
    /// type explicit.  These operations are the building blocks used between
    /// TMA partitioning and the block-scaled MMA fragments.
    pub fn slice(self: *Builder, input: anytype, coord: anytype, result_type: []const u8) View {
        const result = self.emit(cute.slice(
            self.ctx,
            self.asValue(input).inner,
            self.makeCoord(coord),
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn sliceTyped(self: *Builder, input: anytype, coord: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute.slice(
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
        const result_type: *const mlir.Type = if (input_type.isA(cute.MemRefType)) |memref|
            (cute.MemRefType.get(self.ctx, .{ .ptr = memref.getPtr(), .layout = layout_value.type_() }) catch @panic("invalid regrouped memref")).type_()
        else if (input_type.isA(cute.CoordTensorType)) |coords|
            (cute.CoordTensorType.get(self.ctx, .{ .arithTuple = coords.getArithTuple(), .layout = layout_value.type_() }) catch @panic("invalid regrouped coordinate tensor")).type_()
        else
            std.debug.panic("withLayout: expected memref or coordinate tensor, got {f}", .{input_type});

        const iterator_type: *const mlir.Type = if (input_type.isA(cute.MemRefType)) |memref|
            memref.getPtr()
        else if (input_type.isA(cute.CoordTensorType)) |coords|
            (cute.ArithTupleIteratorType.get(self.ctx, .{ .arithTuple = coords.getArithTuple() }) catch @panic("invalid arithmetic tuple iterator")).type_()
        else
            unreachable;
        const iterator = self.emit(cute.get_iter(self.ctx, value.inner, iterator_type, self.loc()));
        const result = self.emit(cute.make_view(self.ctx, iterator.inner, layout_value.inner, result_type, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn groupModes(self: *Builder, input: anytype, begin: u32, end: u32, result_type: []const u8) View {
        const result = self.emit(cute.group_modes(
            self.ctx,
            self.asValue(input).inner,
            self.parseType(result_type),
            .int(self.ctx, .i32, begin),
            .int(self.ctx, .i32, end),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn tiledDivide(self: *Builder, input: anytype, tiler: anytype, result_type: []const u8) View {
        const result = self.emit(cute.tiled_divide(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(tiler).inner,
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn flatDivide(self: *Builder, input: anytype, tiler: anytype, result_type: []const u8) View {
        const result = self.emit(cute.flat_divide(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(tiler).inner,
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn coalesce(self: *Builder, input: anytype, target_profile: ?View, result_type: []const u8) View {
        const result = self.emit(cute.coalesce(
            self.ctx,
            self.asValue(input).inner,
            if (target_profile) |target| target.inner else null,
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn filterZeros(self: *Builder, input: anytype, target_profile: ?View, result_type: []const u8) View {
        const result = self.emit(cute.filter_zeros(
            self.ctx,
            self.asValue(input).inner,
            if (target_profile) |target| target.inner else null,
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn getLayout(self: *Builder, input: anytype, result_type: []const u8) View {
        const result = self.emit(cute.get_layout(self.ctx, self.asValue(input).inner, self.parseType(result_type), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn getIter(self: *Builder, input: anytype, result_type: []const u8) View {
        const result = self.emit(cute.get_iter(self.ctx, self.asValue(input).inner, self.parseType(result_type), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Typed form of `cute.get_iter`, used when the pointer type is already
    /// available through the CuTe C API and no textual type is necessary.
    pub fn getIterTyped(self: *Builder, input: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute.get_iter(self.ctx, self.asValue(input).inner, result_type, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn getShape(self: *Builder, input: anytype, result_type: []const u8) View {
        const result = self.emit(cute.get_shape(self.ctx, self.asValue(input).inner, self.parseType(result_type), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn addOffset(self: *Builder, input: anytype, offset: anytype, result_type: []const u8) View {
        const result = self.emit(cute.add_offset(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(offset).inner,
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn addOffsetTyped(self: *Builder, input: anytype, offset: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute.add_offset(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(offset).inner,
            result_type,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn recastIter(self: *Builder, input: anytype, result_type: []const u8) View {
        const result = self.emit(cute.recast_iter(self.ctx, self.asValue(input).inner, self.parseType(result_type), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn appendToRank(self: *Builder, input: anytype, element: anytype, rank: u32, result_type: []const u8) View {
        const result = self.emit(cute.append_to_rank(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(element).inner,
            self.parseType(result_type),
            .int(self.ctx, .i32, rank),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn makeTiledCopy(self: *Builder, atom: Atom, result_type: []const u8) Atom {
        const result = self.emit(cute.make_tiled_copy(self.ctx, atom.inner, self.parseType(result_type), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn tiledCopyPartition(self: *Builder, tiled: Atom, input: anytype, coord: anytype, destination: bool, result_type: []const u8) View {
        const op = if (destination)
            cute.tiled_copy_partition_D(self.ctx, tiled.inner, self.asValue(input).inner, self.makeCoord(coord), self.parseType(result_type), self.loc())
        else
            cute.tiled_copy_partition_S(self.ctx, tiled.inner, self.asValue(input).inner, self.makeCoord(coord), self.parseType(result_type), self.loc());
        const result = self.emit(op);
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn tiledCopyRetile(self: *Builder, tiled: Atom, input: anytype, result_type: []const u8) View {
        const result = self.emit(cute.tiled_copy_retile(
            self.ctx,
            tiled.inner,
            self.asValue(input).inner,
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn tiledMmaPartition(self: *Builder, tiled: Atom, input: anytype, coord: anytype, operand_id: u2, result_type: []const u8) View {
        const result = self.emit(cute.tiled_mma_partition(
            self.ctx,
            tiled.inner,
            self.asValue(input).inner,
            self.makeCoord(coord),
            self.parseType(result_type),
            .int(self.ctx, .i32, operand_id),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn tiledMmaPartitionTyped(self: *Builder, tiled: Atom, input: anytype, coord: anytype, operand_id: u2, result_type: *const mlir.Type) View {
        const result = self.emit(cute.tiled_mma_partition(
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

    pub fn mmaMakeFragment(self: *Builder, tiled: Atom, input: anytype, operand_id: u2, result_type: []const u8) View {
        const result = self.emit(cute.mma_make_fragment(
            self.ctx,
            tiled.inner,
            self.asValue(input).inner,
            self.parseType(result_type),
            .int(self.ctx, .i32, operand_id),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn copy(self: *Builder, atom: Atom, src: anytype, dst: anytype, pred: ?Value) void {
        self.emitVoid(cute.copy(
            self.ctx,
            atom.inner,
            self.innerSlice(src),
            self.innerSlice(dst),
            if (pred) |p| p.inner else null,
            self.loc(),
        ));
    }

    pub fn gemm(self: *Builder, tiled: Atom, d: anytype, a: anytype, rhs: anytype, c: anytype) void {
        self.emitVoid(cute.gemm(
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

    // ==================== cute.nvgpu.cpasync: TMA (SM90+) ====================
    // The host function builds tiled TMA atoms (Python `make_tiled_tma_atom`);
    // the kernel receives them as grid-constant arguments and makes them
    // executable before issuing copies.

    pub const TmaLoadOptions = struct {
        atom_type: []const u8,
        tensor_type: []const u8,
        kind: Attr = .{ .str = "tma_load" },
        num_multicast: u32 = 1,
        tma_format: ?Attr = null,
    };

    /// Build the host-visible TMA descriptor and its coordinate tensor. The
    /// result types are supplied explicitly because they depend on CuTe's
    /// layout algebra and may contain divisibility constraints.
    pub fn makeTiledTmaLoadAtom(self: *Builder, gmem: anytype, smem_layout: anytype, cta_map: anytype, opts: TmaLoadOptions) TmaDescriptor {
        const results = self.emitResults(2, cute.nvgpu.atom_make_non_exec_tiled_tma_load(
            self.ctx,
            self.asValue(gmem).inner,
            self.asValue(smem_layout).inner,
            self.asValue(cta_map).inner,
            self.parseType(opts.atom_type),
            self.parseType(opts.tensor_type),
            opts.kind.get(self.ctx),
            .int(self.ctx, .i32, opts.num_multicast),
            if (opts.tma_format) |format| format.get(self.ctx) else null,
            self.loc(),
        ));
        return .{
            .atom = .{ .inner = results[0].inner, .kernel = self },
            .tensor = .{ .inner = results[1].inner, .kernel = self },
        };
    }

    /// Typed counterpart of `makeTiledTmaLoadAtom`: the atom and coordinate
    /// tensor types are assembled from `config`.
    pub fn makeTiledTmaLoadAtomTyped(self: *Builder, gmem: anytype, smem_layout: anytype, cta_map: anytype, config: TmaConfig) TmaDescriptor {
        const results = self.emitResults(2, cute.nvgpu.atom_make_non_exec_tiled_tma_load(
            self.ctx,
            self.asValue(gmem).inner,
            self.asValue(smem_layout).inner,
            self.asValue(cta_map).inner,
            self.tmaLoadAtomType(config),
            self.coordTensorType(config.coordinate_rank, config.coordinate_layout),
            self.targetAttribute("#cute_nvgpu.tiled_tma_load<sm_90>"),
            .int(self.ctx, .i32, config.num_multicast),
            tmaFormatAttribute(self, config.format),
            self.loc(),
        ));
        return .{
            .atom = .{ .inner = results[0].inner, .kernel = self },
            .tensor = .{ .inner = results[1].inner, .kernel = self },
        };
    }

    pub const TmaStoreOptions = struct {
        atom_type: []const u8,
        tensor_type: []const u8,
        tma_format: ?Attr = null,
    };

    /// Host-visible tiled TMA store atom and its coordinate tensor.
    pub fn makeTiledTmaStoreAtom(self: *Builder, gmem: anytype, smem_layout: anytype, cta_map: anytype, opts: TmaStoreOptions) TmaDescriptor {
        const results = self.emitResults(2, cute.nvgpu.atom_make_non_exec_tiled_tma_store(
            self.ctx,
            self.asValue(gmem).inner,
            self.asValue(smem_layout).inner,
            self.asValue(cta_map).inner,
            self.parseType(opts.atom_type),
            self.parseType(opts.tensor_type),
            if (opts.tma_format) |format| format.get(self.ctx) else null,
            self.loc(),
        ));
        return .{
            .atom = .{ .inner = results[0].inner, .kernel = self },
            .tensor = .{ .inner = results[1].inner, .kernel = self },
        };
    }

    /// Typed counterpart of `makeTiledTmaStoreAtom`.
    pub fn makeTiledTmaStoreAtomTyped(self: *Builder, gmem: anytype, smem_layout: anytype, cta_map: anytype, config: TmaConfig) TmaDescriptor {
        const results = self.emitResults(2, cute.nvgpu.atom_make_non_exec_tiled_tma_store(
            self.ctx,
            self.asValue(gmem).inner,
            self.asValue(smem_layout).inner,
            self.asValue(cta_map).inner,
            self.tmaStoreAtomType(config),
            self.coordTensorType(config.coordinate_rank, config.coordinate_layout),
            tmaFormatAttribute(self, config.format),
            self.loc(),
        ));
        return .{
            .atom = .{ .inner = results[0].inner, .kernel = self },
            .tensor = .{ .inner = results[1].inner, .kernel = self },
        };
    }

    /// Host-side tiled TMA load atom type described by `config`.
    pub fn tmaLoadAtomType(self: *Builder, config: TmaConfig) *const mlir.Type {
        return self.parseType(std.fmt.allocPrint(
            self.arena.allocator(),
            "!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, {s}, copy_bits = {d}, tma_gbasis = <\"{s}\">, tma_format = {s}>",
            .{ dtypeName(config.dtype), config.copy_bits, config.global_basis.payload(self), formatName(config.format, config.dtype) },
        ) catch @panic("OOM"));
    }

    /// Executable device-side form of a tiled TMA load atom. `make_exec_tma`
    /// preserves the tensor-map payload and adds the launch mode used by
    /// `copy`.
    pub fn tmaLoadExecAtomType(self: *Builder, config: TmaConfig) *const mlir.Type {
        return self.parseType(std.fmt.allocPrint(
            self.arena.allocator(),
            "!cute_nvgpu.atom.tma_load<{s}, copy_bits = {d}, mode = tiled, num_cta = {d}, g_stride = <\"()\"> tma_gbasis = <\"{s}\">>",
            .{ dtypeName(config.dtype), config.copy_bits, config.num_multicast, config.global_basis.payload(self) },
        ) catch @panic("OOM"));
    }

    /// Host-side tiled TMA store atom type described by `config`.
    pub fn tmaStoreAtomType(self: *Builder, config: TmaConfig) *const mlir.Type {
        return self.parseType(std.fmt.allocPrint(
            self.arena.allocator(),
            "!cute_nvgpu.atom.non_exec_tiled_tma_store<{s}, copy_bits = {d}, tma_gbasis = <\"{s}\">, tma_format = {s}>",
            .{ dtypeName(config.dtype), config.copy_bits, config.global_basis.payload(self), formatName(config.format, config.dtype) },
        ) catch @panic("OOM"));
    }

    /// Executable device-side form of a tiled TMA store atom.
    pub fn tmaStoreExecAtomType(self: *Builder, config: TmaConfig) *const mlir.Type {
        return self.parseType(std.fmt.allocPrint(
            self.arena.allocator(),
            "!cute_nvgpu.atom.tma_store<{s}, copy_bits = {d}, mode = tiled, g_stride = <\"()\"> tma_gbasis = <\"{s}\">>",
            .{ dtypeName(config.dtype), config.copy_bits, config.global_basis.payload(self) },
        ) catch @panic("OOM"));
    }

    pub fn prefetchTmaDesc(self: *Builder, descriptor: Atom) void {
        self.emitVoid(cute.nvgpu.prefetch_tma_desc(self.ctx, descriptor.inner, self.loc()));
    }

    pub fn makeExecTma(self: *Builder, descriptor: Atom, result_type: []const u8) Atom {
        const result = self.emit(cute.nvgpu.atom_make_exec_tma(self.ctx, descriptor.inner, self.parseType(result_type), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn makeExecTmaTyped(self: *Builder, descriptor: Atom, result_type: *const mlir.Type) Atom {
        const result = self.emit(cute.nvgpu.atom_make_exec_tma(self.ctx, descriptor.inner, result_type, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Attach the mbarrier that an executable TMA load atom completes.
    pub fn setTmaBarrier(self: *Builder, atom: Atom, barrier: Value) Atom {
        const result = self.emit(cute.nvgpu.atom_set_value(
            self.ctx,
            atom.inner,
            barrier.inner,
            atom.type_(),
            self.targetAttribute("#cute_nvgpu.atom_copy_field_tmaload<tma_bar>"),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn tmaPartition(
        self: *Builder,
        descriptor: Atom,
        cta_coord: anytype,
        cta_layout: anytype,
        smem: anytype,
        targets: anytype,
        smem_type: []const u8,
        target_types: []const []const u8,
    ) TmaPartition {
        if (targets.len != target_types.len) @panic("tmaPartition: target count does not match result type count");
        const result_types = self.arena.allocator().alloc(*const mlir.Type, target_types.len) catch @panic("OOM");
        const views = self.arena.allocator().alloc(View, target_types.len) catch @panic("OOM");
        for (target_types, result_types) |type_text, *ty| ty.* = self.parseType(type_text);
        const op = cute.nvgpu.atom_tma_partition(
            self.ctx,
            descriptor.inner,
            self.makeCoord(cta_coord),
            self.asValue(cta_layout).inner,
            self.asValue(smem).inner,
            self.innerSlice(targets),
            self.parseType(smem_type),
            result_types,
            self.loc(),
        );
        _ = op.appendTo(self.currentBlock());
        for (views, 0..) |*view, i| view.* = .{ .inner = op.result(i + 1), .kernel = self };
        return .{ .smem = .{ .inner = op.result(0), .kernel = self }, .targets = views };
    }

    /// Typed form of `tmaPartition`.  The dependent result types are built by
    /// the Zig CuTe DSL instead of being round-tripped through textual MLIR.
    pub fn tmaPartitionTyped(
        self: *Builder,
        descriptor: Atom,
        cta_coord: anytype,
        cta_layout: anytype,
        smem: anytype,
        targets: anytype,
        smem_type: *const mlir.Type,
        target_types: []const *const mlir.Type,
    ) TmaPartition {
        if (targets.len != target_types.len) @panic("tmaPartitionTyped: target count does not match result type count");
        const views = self.arena.allocator().alloc(View, target_types.len) catch @panic("OOM");
        const op = cute.nvgpu.atom_tma_partition(
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

    // ==================== cute.nvgpu.tcgen05 (SM100) ====================

    /// Tiled MMA type of the SM100 block-scaled MXFP4 x MXFP8 atom.
    pub fn blockScaledMmaType(self: *Builder, cfg: BlockScaledMmaConfig) *const mlir.Type {
        return self.parseType(std.fmt.allocPrint(
            self.arena.allocator(),
            "!cute.tiled_mma<!cute_nvgpu.sm100.mma_bs{s}, atom_layout_MNK = <\"(1,1,1):(0,0,0)\">, permutation_MNK = <\"[_;_;_]\">>",
            .{blockScaledPayload(self, cfg)},
        ) catch @panic("OOM"));
    }

    /// SS, K-major MXFP4 x MXFP8 block-scaled atom and its tiled MMA, as the
    /// CuTe DSL builds them. Both scale-factor pointers start at `scale_ptr`
    /// and are patched after tensor-memory allocation.
    pub fn makeBlockScaledMma(self: *Builder, cfg: BlockScaledMmaConfig, scale_ptr: Value) BlockScaledMma {
        if (cfg.m != 128 or cfg.k != 32 or cfg.n < 8 or cfg.n > 256 or cfg.n % 8 != 0) {
            std.debug.panic("unsupported SM100 block-scaled MMA shape {d}x{d}x{d}", .{ cfg.m, cfg.n, cfg.k });
        }
        const disabled = self.cst(.i1, false);
        const atom_type = std.fmt.allocPrint(self.arena.allocator(), "!cute_nvgpu.sm100.mma_bs{s}", .{blockScaledPayload(self, cfg)}) catch @panic("OOM");
        const atom = self.makeAtom(atom_type, .{ disabled, disabled, disabled, scale_ptr, scale_ptr });
        const tiled = self.emit(cute.make_tiled_mma(self.ctx, atom.inner, self.blockScaledMmaType(cfg), self.loc()));
        return .{ .atom = atom, .tiled = .{ .inner = tiled.inner, .kernel = self } };
    }

    /// Enable (or disable) accumulation into C for the next block-scaled MMA.
    pub fn setMmaAccumulate(self: *Builder, mma: Atom, enabled: Value) Atom {
        const result = self.emit(cute.nvgpu.atom_set_value(
            self.ctx,
            mma.inner,
            enabled.inner,
            mma.type_(),
            self.targetAttribute("#cute_nvgpu.atom_mma_field_sm100_block_scaled<accum_c>"),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    /// Commit the preceding tcgen05 MMA operations to an mbarrier. One elected
    /// lane of the MMA warp issues this after each K-stage.
    pub fn tcgen05Commit(self: *Builder, barrier: Value) void {
        self.emitVoid(mlir.Operation.make(self.ctx, "nvvm.tcgen05.commit", .{
            .operands = .{ .flat = &.{self.sharedBarrierPtr(barrier).inner} },
            .location = self.loc(),
        }));
    }

    /// Wait until asynchronous tensor-memory loads have reached registers.
    pub fn fenceTmemLoad(self: *Builder) void {
        self.emitVoid(mlir.Operation.make(self.ctx, "nvvm.tcgen05.wait", .{
            .attributes = &.{.named(self.ctx, "kind", self.targetAttribute("#nvvm.tcgen05_wait<load>"))},
            .location = self.loc(),
        }));
    }

    pub const TmemOptions = struct { two_cta: bool = false, exclusive: bool = false };

    pub fn allocTmem(self: *Builder, columns: u16, holding_ptr: anytype, opts: TmemOptions) void {
        self.emitVoid(cute.nvgpu.arch_sm100_alloc_tmem(
            self.ctx,
            self.cst(.i32, columns).inner,
            self.asValue(holding_ptr).inner,
            if (opts.two_cta) .unit(self.ctx) else null,
            if (opts.exclusive) .unit(self.ctx) else null,
            self.loc(),
        ));
    }

    pub fn retrieveTmemPtr(self: *Builder, holding_ptr: anytype, dtype: DType, alignment: u64) Value {
        const ptr_type = self.ptrTy(dtype, .tmem, alignment) catch @panic("invalid tensor-memory pointer type");
        return self.emit(cute.nvgpu.arch_sm100_retrieve_tmem_ptr(self.ctx, self.asValue(holding_ptr).inner, ptr_type, self.loc()));
    }

    pub fn relinquishTmemAllocPermit(self: *Builder, opts: TmemOptions) void {
        self.emitVoid(cute.nvgpu.arch_sm100_relinquish_tmem_alloc_permit(self.ctx, if (opts.two_cta) .unit(self.ctx) else null, self.loc()));
    }

    pub fn deallocTmem(self: *Builder, pointer: anytype, columns: u16, opts: TmemOptions) void {
        self.emitVoid(cute.nvgpu.arch_sm100_dealloc_tmem(
            self.ctx,
            self.asValue(pointer).inner,
            self.cst(.i32, columns).inner,
            if (opts.two_cta) .unit(self.ctx) else null,
            if (opts.exclusive) .unit(self.ctx) else null,
            self.loc(),
        ));
    }

    pub const TmemLoadOptions = struct {
        num_dp: u16,
        num_b: u16,
        num_rep: u16,
        pack_16: bool = false,
    };

    pub fn tmemLoad(self: *Builder, src: anytype, result_type: []const u8, opts: TmemLoadOptions) View {
        const result = self.emit(cute.nvgpu.arch_copy_SM100_tmem_load(
            self.ctx,
            self.asValue(src).inner,
            self.parseType(result_type),
            .int(self.ctx, .i32, opts.num_dp),
            .int(self.ctx, .i32, opts.num_b),
            .int(self.ctx, .i32, opts.num_rep),
            if (opts.pack_16) .boolean(self.ctx, true) else null,
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
        return self.emit(cute.nvgpu.arch_copy_SM100_tmem_load(
            self.ctx,
            self.asValue(src).inner,
            result_type,
            .int(self.ctx, .i32, opts.num_dp),
            .int(self.ctx, .i32, opts.num_b),
            .int(self.ctx, .i32, opts.num_rep),
            if (opts.pack_16) .boolean(self.ctx, true) else null,
            self.loc(),
        ));
    }

    pub fn makeS2tCopy(self: *Builder, atom: Atom, tmem: anytype, result_type: []const u8) Atom {
        const result = self.emit(cute.nvgpu.atom_make_s2t_copy(
            self.ctx,
            atom.inner,
            self.asValue(tmem).inner,
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn makeTmemCopy(self: *Builder, atom: Atom, tmem: anytype, result_type: []const u8) Atom {
        const result = self.emit(cute.nvgpu.atom_make_tmem_copy(
            self.ctx,
            atom.inner,
            self.asValue(tmem).inner,
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    pub fn s2tSmemDescriptor(self: *Builder, atom: Atom, input: anytype, result_type: []const u8) View {
        const result = self.emit(cute.nvgpu.atom_get_copy_s2t_smem_desc_view(
            self.ctx,
            atom.inner,
            self.asValue(input).inner,
            self.parseType(result_type),
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
        self.emitVoid(cute.nvgpu.arch_copy_SM100_tma_load(
            self.ctx,
            descriptor_addr.inner,
            data_addr.inner,
            barrier_addr.inner,
            self.innerSlice(coords),
            if (multicast_mask) |v| v.inner else null,
            self.innerSlice(offsets),
            if (cache_policy) |v| v.inner else null,
            mode,
            .int(self.ctx, .i32, num_cta),
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
        self.emitVoid(cute.nvgpu.arch_copy_SM100_tma_store(
            self.ctx,
            descriptor_addr.inner,
            data_addr.inner,
            self.innerSlice(coords),
            if (cache_policy) |v| v.inner else null,
            mode,
            self.loc(),
        ));
    }

    // ==================== cute.arch ====================

    fn sreg(self: *Builder, comptime reg: []const u8) Index3 {
        var out: [3]Value = undefined;
        inline for (.{ "x", "y", "z" }, 0..) |axis, i| {
            out[i] = self.emit(mlir.Operation.make(self.ctx, "nvvm.read.ptx.sreg." ++ reg ++ "." ++ axis, .{
                .results = .{ .flat = &.{.int(self.ctx, .i32)} },
                .location = self.loc(),
            }));
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
        const uniform = self.emit(cute.nvgpu.arch_make_warp_uniform(
            self.ctx,
            uniform_input.inner,
            uniform_input.type_(),
            self.loc(),
        ));
        return if (scalar.dtype() == .i1) uniform.ne(0) else uniform;
    }

    /// `cute.arch.sync_threads()`.
    pub fn syncThreads(self: *Builder) void {
        _ = mlir.Operation.make(self.ctx, "nvvm.barrier", .{
            .attributes = &.{.named(self.ctx, "operandSegmentSizes", .denseArray(self.ctx, .i32, &.{ 0, 0 }))},
            .location = self.loc(),
        }).appendTo(self.currentBlock());
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
        _ = mlir.Operation.make(self.ctx, "llvm.inline_asm", .{
            .attributes = &.{
                .named(self.ctx, "asm_string", .string(self.ctx, assembly)),
                .named(self.ctx, "constraints", .string(self.ctx, "")),
                .named(self.ctx, "has_side_effects", .unit(self.ctx)),
            },
            .location = self.loc(),
        }).appendTo(self.currentBlock());
    }

    /// Build the CuTe first-class divisor used by CUTLASS's persistent tile
    /// scheduler.  Its arithmetic is lowered to the same multiply/shift
    /// sequence as Python CuTe DSL rather than an integer div/rem pair.
    pub fn fastDivmodCreate(self: *Builder, divisor_: anytype) Value {
        const divisor = self.lift(divisor_).to(.i32);
        return self.emit(cute.fast_divmod_create_divisor(
            self.ctx,
            divisor.inner,
            self.parseType("!cute.fast_divmod_divisor<32>"),
            self.loc(),
        ));
    }

    /// Return `{ quotient, remainder }` for a CUTLASS FastDivmod divisor.
    pub fn fastDivmod(self: *Builder, dividend_: anytype, divisor: Value) [2]Value {
        const dividend = self.lift(dividend_).to(.i32);
        return self.emitResults(2, cute.fast_divmod_compute(
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
        return self.emit(mlir.Operation.make(self.ctx, "nvvm.elect.sync", .{
            .results = .{ .flat = &.{.int(self.ctx, .i1)} },
            .location = self.loc(),
        }));
    }

    /// `cute.arch.shuffle_sync_bfly`: exchange a scalar with the lane at
    /// `lane_id ^ offset` in the current warp.
    pub fn shuffleXor(self: *Builder, value: Value, offset: i32) Value {
        const dtype = value.dtype();
        if (dtypeBitwidth(dtype) != 32) @panic("shuffleXor currently requires a 32-bit scalar");
        const bits = if (dtype == .i32) value else value.bitCast(.i32);
        const shuffled = self.emit(mlir.Operation.make(self.ctx, "llvm.inline_asm", .{
            .operands = .{ .flat = &.{
                bits.inner,
                self.cst(.i32, offset).inner,
            } },
            .results = .{ .flat = &.{.int(self.ctx, .i32)} },
            .attributes = &.{
                .named(self.ctx, "asm_string", .string(self.ctx, "shfl.sync.bfly.b32 $0, $1, $2, 31, 0xffffffff;")),
                .named(self.ctx, "constraints", .string(self.ctx, "=r,r,r")),
            },
            .location = self.loc(),
        }));
        return if (dtype == .i32) shuffled else shuffled.bitCast(dtype);
    }

    /// `cute.arch.shuffle_sync`: read a scalar from lane `lane` of the
    /// current warp. Every lane must participate.
    pub fn shuffleIdx(self: *Builder, value: Value, lane: anytype) Value {
        const dtype = value.dtype();
        if (dtypeBitwidth(dtype) != 32) @panic("shuffleIdx currently requires a 32-bit scalar");
        const bits = if (dtype == .i32) value else value.bitCast(.i32);
        const shuffled = self.emit(mlir.Operation.make(self.ctx, "llvm.inline_asm", .{
            .operands = .{ .flat = &.{
                bits.inner,
                self.liftAs(lane, .i32).inner,
            } },
            .results = .{ .flat = &.{.int(self.ctx, .i32)} },
            .attributes = &.{
                .named(self.ctx, "asm_string", .string(self.ctx, "shfl.sync.idx.b32 $0, $1, $2, 31, 0xffffffff;")),
                .named(self.ctx, "constraints", .string(self.ctx, "=r,r,r")),
            },
            .location = self.loc(),
        }));
        return if (dtype == .i32) shuffled else shuffled.bitCast(dtype);
    }

    /// Four FP32 values rounded to FP8 E4M3 (saturating) and packed into one
    /// 32-bit word, `a` in the lowest byte.
    pub fn packFp8x4(self: *Builder, a: Value, b: Value, c: Value, d: Value) Value {
        return self.emit(mlir.Operation.make(self.ctx, "llvm.inline_asm", .{
            .operands = .{ .flat = &.{ a.to(.f32).inner, b.to(.f32).inner, c.to(.f32).inner, d.to(.f32).inner } },
            .results = .{ .flat = &.{.int(self.ctx, .i32)} },
            .attributes = &.{
                .named(self.ctx, "asm_string", .string(self.ctx, "{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1; cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3; mov.b32 $0, {lo, hi}; }")),
                .named(self.ctx, "constraints", .string(self.ctx, "=r,f,f,f,f")),
            },
            .location = self.loc(),
        }));
    }

    /// One-operand FP32 PTX instruction, e.g. `ex2.approx.ftz.f32`.
    pub fn unaryF32(self: *Builder, comptime instruction: []const u8, value: Value) Value {
        return self.emit(mlir.Operation.make(self.ctx, "llvm.inline_asm", .{
            .operands = .{ .flat = &.{value.to(.f32).inner} },
            .results = .{ .flat = &.{DType.f32.toMlir(self.ctx)} },
            .attributes = &.{
                .named(self.ctx, "asm_string", .string(self.ctx, instruction ++ " $0, $1;")),
                .named(self.ctx, "constraints", .string(self.ctx, "=f,f")),
            },
            .location = self.loc(),
        }));
    }

    /// Signal programmatic dependent launches after this grid has consumed
    /// its inputs. The caller must also launch the producer/consumer chain
    /// with PDL enabled.
    pub fn launchDependents(self: *Builder) void {
        _ = mlir.Operation.make(self.ctx, "llvm.inline_asm", .{
            .attributes = &.{
                .named(self.ctx, "asm_string", .string(self.ctx, "griddepcontrol.launch_dependents;")),
                .named(self.ctx, "constraints", .string(self.ctx, "")),
                .named(self.ctx, "has_side_effects", .unit(self.ctx)),
            },
            .location = self.loc(),
        }).appendTo(self.currentBlock());
    }

    /// Wait until the preceding grid has made its programmatic dependency
    /// available. This is the consumer-side half of a PDL launch chain.
    pub fn waitForDependency(self: *Builder) void {
        _ = mlir.Operation.make(self.ctx, "llvm.inline_asm", .{
            .attributes = &.{
                .named(self.ctx, "asm_string", .string(self.ctx, "griddepcontrol.wait;")),
                .named(self.ctx, "constraints", .string(self.ctx, "")),
                .named(self.ctx, "has_side_effects", .unit(self.ctx)),
            },
            .location = self.loc(),
        }).appendTo(self.currentBlock());
    }

    /// `cute.arch.alloc_smem(dtype, size_in_elems, alignment)`: a static
    /// shared-memory pointer, sized at compile time.
    pub fn allocSmem(self: *Builder, dt: DType, size_in_elems: i64, alignment: ?u64) Value {
        const al = alignment orelse @max(1, dtypeBitwidth(dt) / 8);
        const ty = self.ptrTy(dt, .smem, al) catch @panic("bad smem pointer");
        return self.emit(cute.nvgpu.arch_alloc_smem(self.ctx, ty, .int(self.ctx, .i32, size_in_elems), self.loc()));
    }

    // mbarriers, async-proxy fences and bulk-copy groups.

    /// mbarrier operands are `!llvm.ptr<3>` for NVVM.
    fn sharedBarrierPtr(self: *Builder, barrier: Value) Value {
        return self.emit(mlir.Operation.make(self.ctx, "builtin.unrealized_conversion_cast", .{
            .operands = .{ .flat = &.{barrier.inner} },
            .results = .{ .flat = &.{self.parseType("!llvm.ptr<3>")} },
            .location = self.loc(),
        }));
    }

    /// Initialize a shared-memory mbarrier expecting `arrivals` per phase.
    pub fn mbarrierInit(self: *Builder, barrier: Value, arrivals: u32) void {
        self.emitVoid(mlir.Operation.make(self.ctx, "nvvm.mbarrier.init", .{
            .operands = .{ .flat = &.{ self.sharedBarrierPtr(barrier).inner, self.cst(.i32, arrivals).inner } },
            .location = self.loc(),
        }));
    }

    /// Arrive and expect `bytes` of asynchronous transactions (SM90+), as a
    /// TMA producer does before issuing its copies.
    pub fn mbarrierArriveExpectTx(self: *Builder, barrier: Value, bytes: u32) void {
        self.emitVoid(mlir.Operation.make(self.ctx, "nvvm.mbarrier.arrive.expect_tx", .{
            .operands = .{ .flat = &.{ self.sharedBarrierPtr(barrier).inner, self.cst(.i32, bytes).inner } },
            .location = self.loc(),
        }));
    }

    pub fn mbarrierArrive(self: *Builder, barrier: Value, count: u32) void {
        self.emitVoid(mlir.Operation.make(self.ctx, "nvvm.mbarrier.arrive", .{
            .operands = .{ .flat = &.{ self.sharedBarrierPtr(barrier).inner, self.cst(.i32, count).inner } },
            .location = self.loc(),
        }));
    }

    /// Test whether phase `phase` has completed, without suspending (SM90+).
    pub fn mbarrierWaitParity(self: *Builder, barrier: Value, phase: Value) Value {
        return self.emit(mlir.Operation.make(self.ctx, "nvvm.mbarrier.wait.parity", .{
            .operands = .{ .flat = &.{ self.sharedBarrierPtr(barrier).inner, phase.inner } },
            .results = .{ .flat = &.{.int(self.ctx, .i1)} },
            .attributes = &.{.named(self.ctx, "kind", self.parseAttribute("#nvvm.mbar_wait<\"try\">"))},
            .location = self.loc(),
        }));
    }

    /// Wait for phase `phase`, letting the warp sleep for up to
    /// `suspend_time` cycles per attempt (SM90+).
    pub fn mbarrierTryWaitParity(self: *Builder, barrier: Value, phase: Value, suspend_time: u32) void {
        self.emitVoid(mlir.Operation.make(self.ctx, "nvvm.mbarrier.try_wait.parity", .{
            .operands = .{ .flat = &.{ self.sharedBarrierPtr(barrier).inner, phase.inner, self.cst(.i32, suspend_time).inner } },
            .location = self.loc(),
        }));
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
        self.emitVoid(mlir.Operation.make(self.ctx, "nvvm.fence.mbarrier.init", .{ .location = self.loc() }));
    }

    /// Publish ordinary shared-memory writes to the asynchronous (TMA) proxy (SM90+).
    pub fn fenceProxyAsyncShared(self: *Builder) void {
        self.emitVoid(mlir.Operation.make(self.ctx, "nvvm.fence.proxy", .{
            .attributes = &.{
                .named(self.ctx, "kind", self.targetAttribute("#nvvm.proxy_kind<async.shared>")),
                .named(self.ctx, "space", self.targetAttribute("#nvvm.shared_space<cta>")),
            },
            .location = self.loc(),
        }));
    }

    /// Close the current group of bulk asynchronous copies, e.g. TMA stores (SM90+).
    pub fn cpAsyncBulkCommitGroup(self: *Builder) void {
        self.emitVoid(mlir.Operation.make(self.ctx, "nvvm.cp.async.bulk.commit.group", .{ .location = self.loc() }));
    }

    /// Wait until at most `pending` bulk copy groups are in flight; with
    /// `read`, only until their sources have been read (SM90+).
    pub fn cpAsyncBulkWaitGroup(self: *Builder, pending: u32, read: bool) void {
        var attrs: [2]mlir.NamedAttribute = undefined;
        var len: usize = 1;
        attrs[0] = .named(self.ctx, "group", .int(self.ctx, .i32, pending));
        if (read) {
            attrs[1] = .named(self.ctx, "read", .unit(self.ctx));
            len += 1;
        }
        self.emitVoid(mlir.Operation.make(self.ctx, "nvvm.cp.async.bulk.wait_group", .{
            .attributes = attrs[0..len],
            .location = self.loc(),
        }));
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
        inline for (@typeInfo(@TypeOf(result_types)).@"struct".fields, 0..) |f, i| {
            types[i] = @field(result_types, f.name);
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
        inline for (@typeInfo(@TypeOf(inits)).@"struct".fields, 0..) |f, i| {
            const v = self.lift(@field(inits, f.name));
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
        inline for (@typeInfo(@TypeOf(inits)).@"struct".fields, 0..) |f, i| {
            const v = self.lift(@field(inits, f.name));
            init_types[i] = v.type_();
            init_locs[i] = self.loc();
            inits_inner[i] = v.inner;
        }

        var result_types: [M]*const mlir.Type = undefined;
        var result_locs: [M]*const mlir.Location = undefined;
        inline for (@typeInfo(@TypeOf(after_types)).@"struct".fields, 0..) |f, i| {
            result_types[i] = @field(after_types, f.name);
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

        return self.renderModule(false);
    }
};

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (dialects_needed) |d| mlir.DialectHandle.fromString(d).insertDialect(registry);
    mlir.registerFuncExtensions(registry);
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
    try expectContains(ir, "module {\n  func.func @naive_elementwise_add_kernel(%arg0: !cute.ptr<f16, gmem, align<16>>, %arg1: !cute.ptr<f16, gmem, align<16>>, %arg2: !cute.ptr<f16, gmem, align<16>>)");
    try expectContains(ir, "attributes {cute.kernel, gpu.kernel, nvvm.reqntid = array<i32: 128, 1, 1>}");
    try expectContains(ir, "\"nvvm.read.ptx.sreg.tid.x\"() : () -> i32");
    try expectContains(ir, "!cute.layout<\"(16,8):(8,1)\">");
    try expectContains(ir, "!cute.coord<\"(?,?)\">");
    try expectContains(ir, "cute.memref.load");
    try expectContains(ir, "cute.memref.store");
    try expectContains(ir, "arith.addf");
    try expectContains(ir, "    return\n  }\n}\n");
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
    try expectContains(kernel, "\"nvvm.barrier\"() {operandSegmentSizes = array<i32: 0, 0>} : () -> ()");
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
    try expectContains(kernel, "arith.constant 3000000000 : i64");
    try expectContains(kernel, "arith.sitofp");
    try expectContains(kernel, "arith.truncf");
    try expectContains(kernel, "arith.extf");
    try expectContains(kernel, "arith.convertf");
    try expectContains(kernel, "arith.fptosi");
    try expectContains(kernel, "arith.trunci");
    try expectContains(kernel, "arith.extui");
    try expectContains(kernel, "arith.select");
    try expectContains(kernel, "shfl.sync.bfly.b32");
    try expectContains(kernel, "} else {");
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
    inline for (dialects_needed) |dialect| mlir.DialectHandle.fromString(dialect).insertDialect(registry);
    mlir.registerFuncExtensions(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();

    var b = try Builder.open(std.testing.allocator, ctx, "sm100_blockscaled_atoms");
    defer b.deinit();
    _ = try b.declareArgs(.{ .descriptor = .{ .raw = .{
        .type_text = "!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f8E4M3FN, copy_bits = 16384, tma_gbasis = <\"(128,16,1):(1@1,1@0,1@2)\">, tma_format = U8>",
        .grid_constant = true,
    } } });

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
