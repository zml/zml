//! A Zig front end for the CuTe DSL, shaped after `cutlass.cute`: a kernel
//! takes tensors, indexes them with coordinates, reads `thread_idx()` and
//! friends from `cute.arch`. `finish` wraps the kernel in the module the
//! Python DSL emits (a `gpu.module` plus a host launch function) as text.
const std = @import("std");

const cute = @import("mlir/dialects/cute_ir");
const dialects = @import("mlir/dialects");
const dsl = @import("kernels/common");
const tupleArity = dsl.tupleArity;
const mlir = @import("mlir");
const stdx = @import("stdx");

const arith = dialects.arith;
const func = dialects.func;
const math = dialects.math;

const dtypes = @import("dtype.zig");
pub const DType = dtypes.DType;
const isFloatDtype = dtypes.isFloatDtype;
const dtypeBitwidth = dtypes.dtypeBitwidth;

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Builder);
    std.testing.refAllDecls(Value);
    std.testing.refAllDecls(Tensor);
}

/// `nvvm`, `gpu` and `cuda` are not linked into ZML: their ops are emitted
/// unregistered (generic form) or as text by `finish`; cute-ir-compile has them.
pub const dialects_needed = [_][]const u8{ "func", "cute", "cute_nvgpu", "arith", "scf", "math", "cf" };

pub const FinishError = error{InvalidMlir} || std.mem.Allocator.Error || std.Io.Writer.Error;

pub const MAX_RANK = 8;
pub const Dims = stdx.BoundedArray(i64, MAX_RANK);

pub const MemorySpace = cute.MemorySpace;

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
    pub fn set(self: Tensor, coord: anytype, value: anytype) void {
        const k = self.kernel;
        const c = k.makeCoord(coord);
        const v = k.liftAs(value, self.dtype).to(self.dtype);
        _ = cute.memref_store(k.ctx, self.inner, c, v.inner, k.loc()).appendTo(k.currentBlock());
    }
};

pub const Index3 = struct { x: Value, y: Value, z: Value };

pub const ArgSpec = struct {
    name: []const u8,
    kind: Kind,

    /// XLA passes one raw device pointer per custom-call argument. A `tensor`
    /// argument is that pointer composed with a static layout at the top of
    /// the kernel, the way the Python DSL passes `cute.Tensor` arguments.
    pub const Kind = union(enum) {
        ptr: DType,
        tensor: TensorSpec,
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

pub const Launch = struct {
    grid: [3]i32 = .{ 1, 1, 1 },
    block: [3]i32 = .{ 1, 1, 1 },
};

pub const IfOnlyScope = dsl.IfOnlyScope(Builder, Value);

pub fn IfScope(comptime N: usize) type {
    return dsl.IfScope(Builder, Value, N);
}

pub fn ForScope(comptime N: usize) type {
    return dsl.ForScope(Builder, Value, N);
}

pub const Builder = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    ctx: *mlir.Context,
    module: *mlir.Module,
    name: []const u8,
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
            } catch return error.InvalidMlir;
            l.* = self.loc();
        }

        const entry = mlir.Block.init(arg_types, arg_locs);
        const func_op = func.func(ctx, .{
            .name = self.name,
            .block = entry,
            .results = &.{},
            .location = self.loc(),
            .visibility = null,
            .verify = false,
            .extra_attributes = &.{
                .named(ctx, "cute.kernel", .unit(ctx)),
                .named(ctx, "gpu.kernel", .unit(ctx)),
            },
        });
        _ = func_op.appendTo(self.module.body());

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

    pub fn loc(self: *const Builder) *const mlir.Location {
        return .unknown(self.ctx);
    }

    // ==================== types ====================

    pub fn ptrTy(self: *Builder, dt: DType, space: MemorySpace, alignment: u64) !*const mlir.Type {
        return cute.pointerType(self.ctx, dt.toMlir(self.ctx), space, alignment);
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
        std.debug.assert(shape.len > 0 and shape.len <= MAX_RANK);
        const stride = opts.stride orelse self.leftMost(shape);
        std.debug.assert(stride.len == shape.len);
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
        const space = std.meta.stringToEnum(MemorySpace, pt.getMemorySpace().isA(mlir.StringAttribute).?.value()) orelse .generic;
        return .{ .inner = v.inner, .kernel = self, .layout = layout, .dtype = dtypes.mlirToDType(self.ctx, elem), .space = space };
    }

    /// A `!cute.coord` from a tuple of `Value`s and ints; ints stay static in the type.
    pub fn makeCoord(self: *Builder, coord: anytype) *const mlir.Value {
        const T = @TypeOf(coord);
        if (T == Value) return self.makeCoord(.{coord});
        const fields = @typeInfo(T).@"struct".fields;
        var text: std.Io.Writer.Allocating = .init(self.arena.allocator());
        var dyn: stdx.BoundedArray(*const mlir.Value, MAX_RANK) = .empty;
        if (fields.len > 1) text.writer.writeByte('(') catch @panic("OOM");
        inline for (fields, 0..) |f, i| {
            if (i > 0) text.writer.writeByte(',') catch @panic("OOM");
            const leaf = @field(coord, f.name);
            switch (@typeInfo(f.type)) {
                .comptime_int, .int => text.writer.print("{d}", .{leaf}) catch @panic("OOM"),
                else => {
                    text.writer.writeByte('?') catch @panic("OOM");
                    dyn.appendAssumeCapacity(self.lift(leaf).to(.i32).inner);
                },
            }
        }
        if (fields.len > 1) text.writer.writeByte(')') catch @panic("OOM");
        const ty = cute.algebraType(self.ctx, .coord, text.written()) catch @panic("bad coord");
        return self.emit(cute.make_coord(self.ctx, dyn.constSlice(), ty, self.loc())).inner;
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

    /// `cute.arch.sync_threads()`.
    pub fn syncThreads(self: *Builder) void {
        _ = mlir.Operation.make(self.ctx, "nvvm.barrier", .{
            .attributes = &.{.named(self.ctx, "operandSegmentSizes", .denseArray(self.ctx, .i32, &.{ 0, 0 }))},
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

    // ==================== scalars ====================

    pub fn cst(self: *Builder, dt: DType, value: anytype) Value {
        const ctx = self.ctx;
        const T = @TypeOf(value);
        if (isFloatDtype(dt)) {
            const f: f64 = switch (@typeInfo(T)) {
                .comptime_int, .comptime_float => value,
                .int => @floatFromInt(value),
                .float => @floatCast(value),
                else => @compileError("Builder.cst: unsupported value type " ++ @typeName(T)),
            };
            return self.emit(switch (dt) {
                inline .f16, .bf16, .f32, .f64, .f8e4m3fn, .f8e5m2 => |ft| arith.constant_float(ctx, f, @field(mlir.FloatTypes, @tagName(ft)), self.loc()),
                else => unreachable,
            });
        }
        const i: i64 = switch (@typeInfo(T)) {
            .comptime_int, .int => @intCast(value),
            .bool => @intFromBool(value),
            .comptime_float, .float => @intFromFloat(value),
            else => @compileError("Builder.cst: unsupported value type " ++ @typeName(T)),
        };
        return self.emit(arith.constant_int(ctx, i, dt.toMlir(ctx), self.loc()));
    }

    /// A Zig scalar as a constant: ints become `i32`, floats `f32`, like the
    /// Python DSL's `Int32`/`Float32` defaults.
    pub fn lift(self: *Builder, value: anytype) Value {
        const T = @TypeOf(value);
        if (T == Value) return value;
        return switch (@typeInfo(T)) {
            .comptime_int => if (value >= std.math.minInt(i32) and value <= std.math.maxInt(i32)) self.cst(.i32, value) else self.cst(.i64, value),
            .int => |info| if (info.bits > 32) self.cst(.i64, value) else self.cst(.i32, value),
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

    /// Lift literals to the other operand's dtype; widen the narrower integer.
    pub fn coerce(self: *Builder, a: anytype, b: anytype) struct { Value, Value } {
        var av: Value = if (@TypeOf(a) == Value) a else if (@TypeOf(b) == Value) self.liftAs(a, b.dtype()) else self.lift(a);
        var bv: Value = if (@TypeOf(b) == Value) b else self.liftAs(b, av.dtype());
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

    // ==================== module ====================

    /// The kernel as text, `func.func` form; the module verifies here.
    fn kernelText(self: *Builder, launch: Launch) FinishError![]const u8 {
        const current = self.currentBlock();
        if (current.terminator() == null) {
            _ = func.returns(self.ctx, &.{}, self.loc()).appendTo(current);
        }
        const func_op = self.func_op orelse return error.InvalidMlir;
        func_op.setAttributeByName("nvvm.reqntid", .denseArray(self.ctx, .i32, &launch.block));
        if (!self.module.operation().verify()) return error.InvalidMlir;

        var al: std.Io.Writer.Allocating = .init(self.arena.allocator());
        try al.writer.print("{f}", .{func_op});
        return al.written();
    }

    /// The module cute-ir-compile takes: `gpu.module` with the kernel (a
    /// `cuda.kernel`, spelled by renaming the verified `func.func`) and the host
    /// `@launch` XLA reads the grid, block and dynamic smem from.
    pub fn finish(self: *Builder, launch: Launch) FinishError![:0]const u8 {
        const kernel = try self.kernelText(launch);
        const prefix = "func.func";
        if (!std.mem.startsWith(u8, kernel, prefix)) return error.InvalidMlir;

        var al: std.Io.Writer.Allocating = .init(self.allocator);
        defer al.deinit();
        const w = &al.writer;

        try w.writeAll("module attributes {gpu.container_module} {\n  gpu.module @kernels {\n    cuda.kernel");
        try w.writeAll(std.mem.trimEnd(u8, kernel[prefix.len..], "\n"));
        try w.writeAll("\n  }\n  func.func @launch(");
        for (self.args, 0..) |_, i| {
            if (i > 0) try w.writeAll(", ");
            try w.print("%arg{d}: {f}", .{ i, self.arg(i).type_() });
        }
        try w.print(
            \\) -> i32 attributes {{llvm.emit_c_interface}} {{
            \\    %smem = cute.kernel_smem_size @kernels::@{[name]s} : i64
            \\    %c0_i64 = arith.constant 0 : i64
            \\    %stream = cuda.cast %c0_i64 : i64 -> !cuda.stream
            \\    %bx = arith.constant {[bx]d} : i32
            \\    %by = arith.constant {[by]d} : i32
            \\    %bz = arith.constant {[bz]d} : i32
            \\    %gx = arith.constant {[gx]d} : i32
            \\    %gy = arith.constant {[gy]d} : i32
            \\    %gz = arith.constant {[gz]d} : i32
            \\    %cfg = cuda.launch_cfg.create<max_attrs = 17 : i32> (blockDim = (%bx, %by, %bz), dynamicSmemBytes = %smem, gridDim = (%gx, %gy, %gz), stream = %stream) : i32, i32, i32, i64, i32, i32, i32, !cuda.stream -> !cuda.launch_cfg<max_attrs = 17>
            \\    %r = cuda.launch_ex @kernels::@{[name]s}<%cfg> (
        , .{
            .name = self.name,
            .bx = launch.block[0],
            .by = launch.block[1],
            .bz = launch.block[2],
            .gx = launch.grid[0],
            .gy = launch.grid[1],
            .gz = launch.grid[2],
        });
        for (self.args, 0..) |_, i| {
            if (i > 0) try w.writeAll(", ");
            try w.print("%arg{d}", .{i});
        }
        try w.writeAll(") : !cuda.launch_cfg<max_attrs = 17>, (");
        for (self.args, 0..) |_, i| {
            if (i > 0) try w.writeAll(", ");
            try w.print("{f}", .{self.arg(i).type_()});
        }
        try w.writeAll(
            \\) -> !cuda.result
            \\    %status = cuda.cast %r : !cuda.result -> i32
            \\    return %status : i32
            \\  }
            \\}
            \\
        );
        return try self.allocator.dupeZ(u8, al.written());
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

    // The func.func form round-trips through the local dialects.
    const kernel = try b.kernelText(.{ .block = .{ 128, 1, 1 } });
    try expectContains(kernel, "func.func @naive_elementwise_add_kernel(%arg0: !cute.ptr<f16, gmem, align<16>>");
    try expectContains(kernel, "attributes {cute.kernel, gpu.kernel, nvvm.reqntid = array<i32: 128, 1, 1>}");
    try expectContains(kernel, "\"nvvm.read.ptx.sreg.tid.x\"() : () -> i32");
    try expectContains(kernel, "!cute.layout<\"(16,8):(8,1)\">");
    try expectContains(kernel, "!cute.coord<\"(?,?)\">");
    try expectContains(kernel, "cute.memref.load");
    try expectContains(kernel, "cute.memref.store");
    try expectContains(kernel, "arith.addf");
    const parsed = try mlir.Module.parse(ctx, kernel);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());

    const ir = try b.finish(.{ .grid = .{ 1, 1, 1 }, .block = .{ 128, 1, 1 } });
    defer std.testing.allocator.free(ir);
    try expectContains(ir, "module attributes {gpu.container_module}");
    try expectContains(ir, "gpu.module @kernels {\n    cuda.kernel @naive_elementwise_add_kernel(");
    try expectContains(ir, "func.func @launch(%arg0: !cute.ptr<f16, gmem, align<16>>, %arg1: !cute.ptr<f16, gmem, align<16>>, %arg2: !cute.ptr<f16, gmem, align<16>>) -> i32");
    try expectContains(ir, "cute.kernel_smem_size @kernels::@naive_elementwise_add_kernel : i64");
    try expectContains(ir, "blockDim = (%bx, %by, %bz)");
    try expectContains(ir, "cuda.launch_ex @kernels::@naive_elementwise_add_kernel<%cfg> (%arg0, %arg1, %arg2)");
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
    const j = b.cst(.i32, 127).sub(tid);
    var guard = b.openIf(i.lt(4096));
    var acc = b.openFor(0, 4, 1, .{b.cst(.f32, 0.0)});
    acc.yield(.{acc.carried[0].add(smem.get(.{j}).mul(acc.iv.to(.f32)))});
    a.out.set(.{i}, acc.results[0]);
    guard.yieldThen(.{});

    const kernel = try b.kernelText(.{ .block = .{ 128, 1, 1 } });
    try expectContains(kernel, "cute_nvgpu.arch.alloc_smem");
    try expectContains(kernel, "!cute.ptr<f32, smem, align<16>>");
    try expectContains(kernel, "\"nvvm.barrier\"() {operandSegmentSizes = array<i32: 0, 0>} : () -> ()");
    try expectContains(kernel, "scf.if");
    try expectContains(kernel, "scf.for");
    try expectContains(kernel, "!cute.layout<\"4096:1\">");
    const parsed = try mlir.Module.parse(ctx, kernel);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());

    const ir = try b.finish(.{ .grid = .{ 32, 1, 1 }, .block = .{ 128, 1, 1 } });
    defer std.testing.allocator.free(ir);
    try expectContains(ir, "%gx = arith.constant 32 : i32");
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
    const kernel = try b.kernelText(.{});
    try expectContains(kernel, "!cute.layout<\"(4,4):(1,4)\">");
    try expectContains(kernel, "!cute.coord<\"(?,3)\">");
}
