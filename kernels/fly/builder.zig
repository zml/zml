const std = @import("std");

const dialects = @import("mlir/dialects");
const arith = dialects.arith;
const gpu = dialects.gpu;
const scf = dialects.scf;
const fly = @import("mlir/dialects/fly");
const dsl = @import("kernels/common");
const tupleArity = dsl.tupleArity;
const mlir = @import("mlir");
const stdx = @import("stdx");

pub const layout = @import("layout.zig");
pub const dtypes = @import("dtype.zig");

pub const DType = dtypes.DType;
pub const IntTuple = layout.IntTuple;
pub const Layout = layout.Layout;
pub const Tile = layout.Tile;
pub const it = layout.it;
pub const L = layout.L;
pub const rowMajor = layout.rowMajor;
pub const colMajor = layout.colMajor;
pub const ordered = layout.ordered;
pub const tile = layout.tile;

pub const dialects_needed = fly.dialects_needed;

pub const FinishError = error{ InvalidMlir, OutOfMemory } || std.Io.Writer.Error;

pub const Arch = enum { gfx942, gfx950 };

pub const Dim = gpu.Dim;

pub const gpu_module_name = "zml_fly_kernels";

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Builder);
    std.testing.refAllDecls(Value);
}

// =============================================================================
// Value
// =============================================================================

pub const Value = struct {
    inner: *const mlir.Value,
    kernel: ?*Builder = null,

    pub const Kind = fly.TypeKind;

    pub fn type_(self: Value) *const mlir.Type {
        return self.inner.type_();
    }

    fn kern(self: Value) *Builder {
        return self.kernel orelse @panic("fly.Value has no owning Builder; use Builder.* helpers instead");
    }

    /// Derived, never stored: control_flow.zig builds Values with bare
    /// `.{ .inner, .kernel }` literals.
    pub fn kind(self: Value) Kind {
        return fly.typeKind(self.type_());
    }

    pub fn shapeStatic(self: Value) IntTuple {
        return self.kern().readIntTuple(fly.layoutLikeShape(self.type_()));
    }

    /// Host value, not an op; see `emitSize`.
    pub fn sizeStatic(self: Value) i64 {
        return self.shapeStatic().product() orelse std.debug.panic("fly: `{f}` has a dynamic shape", .{self.type_()});
    }

    pub fn intTupleStatic(self: Value) IntTuple {
        return self.kern().readIntTuple(fly.expect(self.type_(), .int_tuple));
    }

    pub fn elemDType(self: Value) DType {
        const t = fly.elemType(self.type_());
        return dtypes.fromMlir(self.kern().ctx, t) orelse std.debug.panic("fly: unsupported element type `{f}`", .{t});
    }

    fn isI64(self: Value) bool {
        const int_ty = self.type_().isA(mlir.IntegerType) orelse return false;
        return int_ty.width() == 64;
    }

    fn isFloatElem(self: Value) bool {
        return self.scalarElemType().isFloat();
    }

    /// The element type of a vector, or the type itself.
    fn scalarElemType(self: Value) *const mlir.Type {
        if (self.type_().isA(mlir.VectorType)) |vec| return vec.shaped().elementType();
        return self.type_();
    }

    fn scalarDType(self: Value) DType {
        const t = self.scalarElemType();
        return dtypes.fromMlir(self.kern().ctx, t) orelse std.debug.panic("fly: unsupported scalar type `{f}`", .{t});
    }

    // ------------------------------------------------------- layout algebra
    // Every op below takes a Value or a comptime Layout/Tile/tuple literal,
    // and FlyDSL infers the result type.

    fn unary(self: Value, comptime mnemonic: []const u8) Value {
        const k = self.kern();
        return k.emit(fly.inferred(k.ctx, mnemonic, &.{self.inner}, .empty, k.loc()));
    }

    fn binary(self: Value, comptime mnemonic: []const u8, rhs: anytype) Value {
        const k = self.kern();
        const r = k.lift(rhs);
        return k.emit(fly.inferred(k.ctx, mnemonic, &.{ self.inner, r.inner }, .empty, k.loc()));
    }

    pub fn flatDivide(self: Value, tiler: anytype) Value {
        return self.binary("flat_divide", tiler);
    }
    pub fn logicalDivide(self: Value, tiler: anytype) Value {
        return self.binary("logical_divide", tiler);
    }
    pub fn zippedDivide(self: Value, tiler: anytype) Value {
        return self.binary("zipped_divide", tiler);
    }
    pub fn tiledDivide(self: Value, tiler: anytype) Value {
        return self.binary("tiled_divide", tiler);
    }
    pub fn logicalProduct(self: Value, tiler: anytype) Value {
        return self.binary("logical_product", tiler);
    }
    pub fn zippedProduct(self: Value, tiler: anytype) Value {
        return self.binary("zipped_product", tiler);
    }
    pub fn tiledProduct(self: Value, tiler: anytype) Value {
        return self.binary("tiled_product", tiler);
    }
    pub fn flatProduct(self: Value, tiler: anytype) Value {
        return self.binary("flat_product", tiler);
    }
    pub fn blockedProduct(self: Value, tiler: anytype) Value {
        return self.binary("blocked_product", tiler);
    }
    pub fn rakedProduct(self: Value, tiler: anytype) Value {
        return self.binary("raked_product", tiler);
    }
    pub fn composition(self: Value, inner: anytype) Value {
        return self.binary("composition", inner);
    }
    pub fn rightInverse(self: Value) Value {
        return self.unary("right_inverse");
    }
    pub fn leftInverse(self: Value) Value {
        return self.unary("left_inverse");
    }
    pub fn coalesce(self: Value) Value {
        return self.unary("coalesce");
    }
    pub fn complement(self: Value) Value {
        return self.unary("complement");
    }

    /// `t[coord]` with a `null` somewhere; `coord` is a tuple literal
    /// (`.{ null, null, bx, by }`, nesting allowed) or an int-tuple Value.
    pub fn slice(self: Value, coord: anytype) Value {
        const k = self.kern();
        const c = k.intTuple(coord);
        return k.emit(fly.inferred(k.ctx, "slice", &.{ self.inner, c.inner }, .empty, k.loc()));
    }
    pub fn dice(self: Value, coord: anytype) Value {
        const k = self.kern();
        const c = k.intTuple(coord);
        return k.emit(fly.inferred(k.ctx, "dice", &.{ self.inner, c.inner }, .empty, k.loc()));
    }

    /// The sub-tuple / sub-layout at a mode path.
    pub fn get(self: Value, mode: []const i32) Value {
        const k = self.kern();
        return k.emit(fly.get(k.ctx, self.inner, mode, k.loc()));
    }
    pub fn selectModes(self: Value, indices: []const i32) Value {
        const k = self.kern();
        return k.emit(fly.select(k.ctx, self.inner, indices, k.loc()));
    }
    pub fn take(self: Value, begin: i32, end: i32) Value {
        const k = self.kern();
        return k.emit(fly.takeOrGroup(k.ctx, "take", self.inner, begin, end, k.loc()));
    }
    pub fn group(self: Value, begin: i32, end: i32) Value {
        const k = self.kern();
        return k.emit(fly.takeOrGroup(k.ctx, "group", self.inner, begin, end, k.loc()));
    }

    /// Emits an op; `shapeStatic` reads the type instead.
    pub fn emitShape(self: Value) Value {
        return self.unary("get_shape");
    }
    pub fn emitStride(self: Value) Value {
        return self.unary("get_stride");
    }
    pub fn emitLayout(self: Value) Value {
        return self.unary("get_layout");
    }
    pub fn emitIter(self: Value) Value {
        return self.unary("get_iter");
    }
    pub fn emitSize(self: Value) Value {
        return self.unary("size");
    }
    pub fn cosize(self: Value) Value {
        return self.unary("cosize");
    }
    pub fn productEach(self: Value) Value {
        return self.unary("int_tuple_product_each");
    }

    /// A register tensor shaped like `self`.
    pub fn makeFragmentLike(self: Value, dtype: ?DType) Value {
        const k = self.kern();
        const t: ?*const mlir.Type = if (dtype) |d| d.toMlir(k.ctx) else null;
        return k.emit(fly.makeFragmentLike(k.ctx, self.inner, t, k.loc()));
    }

    pub fn elemLess(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const r = k.intTuple(rhs);
        return k.emit(fly.typed(k.ctx, "elem_less", &.{ self.inner, r.inner }, &.{.int(k.ctx, .i1)}, .empty, k.loc()));
    }

    // ---------------------------------------------------------------- tensor access

    /// `t[coord]` with no `null` (a coord tensor yields an int tuple).
    pub fn at(self: Value, coord: anytype) Value {
        const k = self.kern();
        const c = k.intTuple(coord);
        return k.emit(fly.inferred(k.ctx, "memref.load", &.{ self.inner, c.inner }, .empty, k.loc()));
    }

    pub fn set(self: Value, coord: anytype, value: Value) void {
        const k = self.kern();
        const c = k.intTuple(coord);
        k.emitNone(fly.effect(k.ctx, "memref.store", &.{ value.inner, self.inner, c.inner }, .empty, k.loc()));
    }

    /// The whole register tensor as a `vector<NxT>`.
    pub fn load(self: Value) Value {
        return self.unary("memref.load_vec");
    }

    pub fn store(self: Value, vector: Value) void {
        const k = self.kern();
        k.emitNone(fly.effect(k.ctx, "memref.store_vec", &.{ vector.inner, self.inner }, .empty, k.loc()));
    }

    /// Store a splat vector of the tensor's static size.
    pub fn fill(self: Value, value: anytype) void {
        const k = self.kern();
        const n = self.sizeStatic();
        const dt = self.elemDType();
        const cst = k.constant(dt, value);
        const vec_ty = mlir.Type.vector(&.{n}, dt.toMlir(k.ctx));
        const vec = k.emit(dialects.vector.broadcast(k.ctx, cst.inner, vec_ty, k.loc()));
        self.store(vec);
    }

    // ---------------------------------------------------------------- pointers

    pub fn addOffset(self: Value, offset: anytype) Value {
        const k = self.kern();
        const o = k.intTuple(offset);
        return k.emit(fly.inferred(k.ctx, "add_offset", &.{ self.inner, o.inner }, .empty, k.loc()));
    }

    /// Scalar load.
    pub fn ptrLoad(self: Value) Value {
        const k = self.kern();
        const elem = self.elemDType().toMlir(k.ctx);
        return k.emit(fly.typed(k.ctx, "ptr.load", &.{self.inner}, &.{elem}, .empty, k.loc()));
    }

    pub fn ptrStore(self: Value, value: Value) void {
        const k = self.kern();
        k.emitNone(fly.effect(k.ctx, "ptr.store", &.{ value.inner, self.inner }, .empty, k.loc()));
    }

    /// A tensor over this pointer.
    pub fn view(self: Value, lay: anytype) Value {
        return self.binary("make_view", lay);
    }

    // ---------------------------------------------------------------- arithmetic

    /// Broadcast a scalar against a vector operand.
    fn coerceArith(self: Value, rhs: anytype) struct { Value, Value } {
        const k = self.kern();
        const r: Value = if (@TypeOf(rhs) == Value) rhs else k.constLike(rhs, self);
        const ln = self.vectorLen();
        const rn = r.vectorLen();
        if (ln == rn) return .{ self, r };
        if (rn == 1) return .{ self, k.emit(dialects.vector.broadcast(k.ctx, r.inner, self.type_(), k.loc())) };
        if (ln == 1) return .{ k.emit(dialects.vector.broadcast(k.ctx, self.inner, r.type_(), k.loc())), r };
        std.debug.panic("fly: cannot combine {f} with {f}", .{ self.type_(), r.type_() });
    }

    pub fn add(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = self.coerceArith(rhs);
        return k.emitFast(if (l.isFloatElem()) arith.addf(k.ctx, l.inner, r.inner, k.loc()) else arith.addi(k.ctx, l.inner, r.inner, k.loc()));
    }
    pub fn sub(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = self.coerceArith(rhs);
        return k.emitFast(if (l.isFloatElem()) arith.subf(k.ctx, l.inner, r.inner, k.loc()) else arith.subi(k.ctx, l.inner, r.inner, k.loc()));
    }
    pub fn mul(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = self.coerceArith(rhs);
        return k.emitFast(if (l.isFloatElem()) arith.mulf(k.ctx, l.inner, r.inner, k.loc()) else arith.muli(k.ctx, l.inner, r.inner, k.loc()));
    }
    pub fn div(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = self.coerceArith(rhs);
        return k.emitFast(if (l.isFloatElem()) arith.divf(k.ctx, l.inner, r.inner, k.loc()) else arith.divsi(k.ctx, l.inner, r.inner, k.loc()));
    }
    /// Signed.
    pub fn rem(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = self.coerceArith(rhs);
        return k.emit(arith.remsi(k.ctx, l.inner, r.inner, k.loc()));
    }

    pub fn toI32(self: Value) Value {
        const k = self.kern();
        return k.emit(arith.index_cast(k.ctx, self.inner, .int(k.ctx, .i32), k.loc()));
    }

    /// 1 for a scalar.
    pub fn vectorLen(self: Value) i64 {
        const vec = self.type_().isA(mlir.VectorType) orelse return 1;
        return vec.dimension(0);
    }

    /// Numeric conversion, scalar or vector. `i1` is treated as unsigned, so a
    /// predicate converts to 1, not -1; equal-width float pairs (f16 <-> bf16,
    /// the fp8 spellings) go through `arith.convertf`, which is the only op
    /// that accepts them.
    pub fn to(self: Value, dtype: DType) Value {
        const k = self.kern();
        const src = self.scalarDType();
        if (src == dtype) return self;
        const n = self.vectorLen();
        const result = if (n == 1) dtype.toMlir(k.ctx) else mlir.Type.vector(&.{n}, dtype.toMlir(k.ctx));
        const unsigned_src = src == .i1;
        const op = if (src.isFloat() and dtype.isFloat())
            (if (dtype.bitWidth() > src.bitWidth())
                arith.extf(k.ctx, self.inner, result, k.loc())
            else if (dtype.bitWidth() < src.bitWidth())
                arith.truncf(k.ctx, self.inner, result, k.loc())
            else
                arith.convertf(k.ctx, self.inner, result, .{}, k.loc()))
        else if (src.isFloat())
            arith.fptosi(k.ctx, self.inner, result, k.loc())
        else if (dtype.isFloat())
            (if (unsigned_src) arith.uitofp(k.ctx, self.inner, result, k.loc()) else arith.sitofp(k.ctx, self.inner, result, k.loc()))
        else if (dtype.bitWidth() > src.bitWidth())
            (if (unsigned_src) arith.extui(k.ctx, self.inner, result, k.loc()) else arith.extsi(k.ctx, self.inner, result, k.loc()))
        else
            arith.trunci(k.ctx, self.inner, result, k.loc());
        return k.emit(op);
    }

    pub const Cmp = enum { eq, ne, lt, le, gt, ge };

    /// Signed integer comparison against a Value or an integer literal.
    pub fn cmp(self: Value, pred: Cmp, rhs: anytype) Value {
        const k = self.kern();
        const r: Value = if (@TypeOf(rhs) == Value) rhs else k.constLike(rhs, self);
        const p: arith.CmpIPredicate = switch (pred) {
            .eq => .eq,
            .ne => .ne,
            .lt => .slt,
            .le => .sle,
            .gt => .sgt,
            .ge => .sge,
        };
        return k.emit(arith.cmpi(k.ctx, p, self.inner, r.inner, k.loc()));
    }

    /// Ordered.
    pub fn cmpf(self: Value, pred: Cmp, rhs: Value) Value {
        const k = self.kern();
        const p: arith.CmpFPredicate = switch (pred) {
            .eq => .oeq,
            .ne => .one,
            .lt => .olt,
            .le => .ole,
            .gt => .ogt,
            .ge => .oge,
        };
        return k.emit(arith.cmpf(k.ctx, p, self.inner, rhs.inner, k.loc()));
    }

    pub fn select(self: Value, a: Value, b: Value) Value {
        const k = self.kern();
        return k.emit(arith.select(k.ctx, self.inner, a.inner, b.inner, k.loc()));
    }

    /// Lane `k` reads lane `k ^ offset`, within `width` lanes.
    pub fn shuffleXor(self: Value, offset: i32, width: i32) Value {
        return self.shuffle(.xor, offset, width);
    }

    pub const ShuffleMode = enum { xor, up, down, idx };

    pub fn shuffle(self: Value, mode: ShuffleMode, offset: i32, width: i32) Value {
        const k = self.kern();
        const off = k.constant(.i32, offset);
        const wid = k.constant(.i32, width);
        const mode_attr = switch (mode) {
            inline else => |m| fly.parseAttr(k.ctx, "#gpu<shuffle_mode " ++ @tagName(m) ++ ">"),
        };
        const op = fly.make(k.ctx, "gpu.shuffle", .{
            .operands = .{ .flat = &.{ self.inner, off.inner, wid.inner } },
            .results = .{ .flat = &.{ self.type_(), .int(k.ctx, .i1) } },
            .attributes = &.{.named(k.ctx, "mode", mode_attr)},
            .location = k.loc(),
        });
        return k.emit(op);
    }

    pub const ReduceKind = enum { add, mul, minf, maxf, minsi, maxsi, and_, or_, xor };

    /// Reduce a vector to a scalar.
    pub fn reduce(self: Value, reduce_kind: ReduceKind) Value {
        const k = self.kern();
        const result = self.scalarElemType();
        const ck: dialects.vector.CombiningKind = switch (reduce_kind) {
            .add => .add,
            .mul => .mul,
            .minf => .minnumf,
            .maxf => .maxnumf,
            .minsi => .minsi,
            .maxsi => .maxsi,
            .and_ => .@"and",
            .or_ => .@"or",
            .xor => .xor,
        };
        const op = dialects.vector.reduction(k.ctx, ck, self.inner, result, .{}, k.loc());
        return k.emitFast(op);
    }

    /// Static position.
    pub fn extract(self: Value, i: i64) Value {
        const k = self.kern();
        return k.emit(dialects.vector.extract(k.ctx, self.inner, &.{i}, &.{}, self.scalarElemType(), k.loc()));
    }
};

// =============================================================================
// Tiled copy / MMA handles
// =============================================================================

/// A `!fly.tiled_copy` plus the tile it was built with.
pub const TiledCopy = struct {
    value: Value,
    tile: Value,

    pub fn getSlice(self: TiledCopy, thr_idx: Value) ThrCopy {
        return .{ .tiled_copy = self.value, .thr = self.value.kern().intTuple(thr_idx) };
    }

    pub fn tileMN(self: TiledCopy) Value {
        return self.tile;
    }
};

/// Per-thread slice of a TiledCopy.
pub const ThrCopy = struct {
    tiled_copy: Value,
    thr: Value,

    pub fn partitionS(self: ThrCopy, src: Value) Value {
        const k = self.tiled_copy.kern();
        return k.emit(fly.inferred(k.ctx, "tiled_copy.partition_src", &.{ self.tiled_copy.inner, src.inner, self.thr.inner }, .empty, k.loc()));
    }

    pub fn partitionD(self: ThrCopy, dst: Value) Value {
        const k = self.tiled_copy.kern();
        return k.emit(fly.inferred(k.ctx, "tiled_copy.partition_dst", &.{ self.tiled_copy.inner, dst.inner, self.thr.inner }, .empty, k.loc()));
    }

    pub fn retile(self: ThrCopy, t: Value) Value {
        const k = self.tiled_copy.kern();
        return k.emit(fly.inferred(k.ctx, "tiled_copy.retile", &.{ self.tiled_copy.inner, t.inner }, .empty, k.loc()));
    }
};

pub const TiledMma = struct {
    value: Value,

    pub fn getSlice(self: TiledMma, thr_idx: Value) ThrMma {
        return .{ .tiled_mma = self.value, .thr = self.value.kern().intTuple(thr_idx) };
    }

    pub const FragmentOpts = struct {
        stages: ?i32 = null,
    };

    fn makeFragment(self: TiledMma, operand: fly.MmaOperand, input: Value, opts: FragmentOpts) Value {
        const k = self.value.kern();
        var a: fly.Attrs = .empty;
        a.appendAssumeCapacity(.named(k.ctx, "operand_id", Builder.must(fly.attributes.mmaOperandAttr(k.ctx, operand))));
        if (opts.stages) |st| a.appendAssumeCapacity(.named(k.ctx, "stages", .int(k.ctx, .i32, st)));
        return k.emit(fly.inferred(k.ctx, "mma.make_fragment", &.{ self.value.inner, input.inner }, a, k.loc()));
    }

    /// Per-thread register fragment for the A operand.
    pub fn makeFragmentA(self: TiledMma, a: Value) Value {
        return self.makeFragment(.a, a, .{});
    }
    pub fn makeFragmentB(self: TiledMma, b: Value) Value {
        return self.makeFragment(.b, b, .{});
    }
    pub fn makeFragmentC(self: TiledMma, c: Value) Value {
        return self.makeFragment(.c, c, .{});
    }
    pub fn makeFragmentAOpts(self: TiledMma, a: Value, opts: FragmentOpts) Value {
        return self.makeFragment(.a, a, opts);
    }
    pub fn makeFragmentBOpts(self: TiledMma, b: Value, opts: FragmentOpts) Value {
        return self.makeFragment(.b, b, opts);
    }
};

pub const ThrMma = struct {
    tiled_mma: Value,
    thr: Value,

    fn partition(self: ThrMma, operand: fly.MmaOperand, t: Value) Value {
        const k = self.tiled_mma.kern();
        return k.emit(fly.inferred(k.ctx, "tiled_mma.partition", &.{ self.tiled_mma.inner, t.inner, self.thr.inner }, fly.attrs(&.{
            .named(k.ctx, "operand_id", fly.mmaOperandAttr(k.ctx, operand)),
        }), k.loc()));
    }

    pub fn partitionA(self: ThrMma, a: Value) Value {
        return self.partition(.a, a);
    }
    pub fn partitionB(self: ThrMma, b: Value) Value {
        return self.partition(.b, b);
    }
    pub fn partitionC(self: ThrMma, c: Value) Value {
        return self.partition(.c, c);
    }
};

// =============================================================================
// Builder
// =============================================================================

pub const Builder = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    ctx: *mlir.Context,
    /// `builtin.module attributes {gpu.container_module}`.
    module: *mlir.Module,
    gpu_module: *mlir.Operation,
    gpu_body: *mlir.Block,
    name: []const u8,
    func_op: ?*mlir.Operation = null,
    entry_block: ?*mlir.Block = null,
    block_stack: std.ArrayList(*mlir.Block) = .empty,
    arg_specs: []const ArgSpec = &.{},
    /// The `!fly.memref` view for tensor arguments, the raw `!fly.ptr` otherwise.
    arg_values: []Value = &.{},
    target: Arch = .gfx942,
    fast_math: bool = false,

    pub const ArgSpec = struct {
        name: []const u8,
        dtype: DType,
        space: fly.AddressSpace = .global,
        /// Row-major tensor argument; null for a bare pointer.
        dims: ?[]const i64 = null,
    };

    pub fn open(allocator: std.mem.Allocator, ctx: *mlir.Context, name: []const u8) !Builder {
        const loc_: *const mlir.Location = .unknown(ctx);
        const module: *mlir.Module = .init(loc_);
        errdefer module.deinit();
        module.operation().setAttributeByName("gpu.container_module", .unit(ctx));

        const gpu_body = mlir.Block.init(&.{}, &.{});
        const gpu_module = gpu.module(ctx, gpu_module_name, gpu_body, loc_);
        _ = gpu_module.appendTo(module.body());

        var arena: std.heap.ArenaAllocator = .init(allocator);
        errdefer arena.deinit();

        return .{
            .allocator = allocator,
            .arena = arena,
            .ctx = ctx,
            .module = module,
            .gpu_module = gpu_module,
            .gpu_body = gpu_body,
            .name = name,
        };
    }

    pub fn deinit(self: *Builder) void {
        self.module.deinit();
        self.arena.deinit();
    }

    /// Arguments by name. Each field is `.{ .ptr = dtype }` or
    /// `.{ .tensor = .{ .dtype = ..., .dims = &.{ ... } } }`.
    pub fn declareArgs(self: *Builder, spec: anytype) !dsl.NamedArgs(@TypeOf(spec), Value) {
        const Spec = @TypeOf(spec);
        const fields = @typeInfo(Spec).@"struct".fields;
        var specs: [fields.len]ArgSpec = undefined;
        inline for (fields, 0..) |f, i| {
            const raw = @field(spec, f.name);
            const variant = @typeInfo(@TypeOf(raw)).@"struct".fields[0].name;
            const inner = @field(raw, variant);
            specs[i] = if (comptime std.mem.eql(u8, variant, "ptr"))
                .{ .name = f.name, .dtype = inner }
            else if (comptime std.mem.eql(u8, variant, "tensor"))
                .{ .name = f.name, .dtype = inner.dtype, .dims = inner.dims, .space = if (@hasField(@TypeOf(inner), "space")) inner.space else .global }
            else
                @compileError("fly.declareArgs: argument `" ++ f.name ++ "` must be `.{ .ptr = dtype }` or `.{ .tensor = .{ .dtype, .dims } }`");
        }
        try self.declareArgsLow(&specs);
        var named: dsl.NamedArgs(Spec, Value) = undefined;
        inline for (fields, 0..) |f, i| @field(named, f.name) = self.arg(i);
        return named;
    }

    /// One `!fly.ptr<storage(dtype), space>` per spec, in order; tensor
    /// arguments are bridged to `!fly.memref` views.
    pub fn declareArgsLow(self: *Builder, specs: []const ArgSpec) !void {
        std.debug.assert(self.entry_block == null);
        const ctx = self.ctx;
        const scratch = self.arena.allocator();
        self.arg_specs = try scratch.dupe(ArgSpec, specs);

        const arg_types = try scratch.alloc(*const mlir.Type, specs.len);
        const arg_locs = try scratch.alloc(*const mlir.Location, specs.len);
        for (arg_types, arg_locs, specs) |*ty_, *a_loc, a| {
            ty_.* = self.ptrType(a.dtype.storageElem(), self.addressSpaceAttr(a.space));
            a_loc.* = self.loc().named(ctx, a.name);
        }

        const entry = mlir.Block.init(arg_types, arg_locs);
        const func_op = gpu.func(ctx, .{
            .name = self.name,
            .args = arg_types,
            .block = entry,
            .location = self.loc(),
        });
        _ = func_op.appendTo(self.gpu_body);
        self.func_op = func_op;
        self.entry_block = entry;

        // The view keeps the storage element type: recasting a predicate
        // argument back to `!fly.ptr<i1>` would make a copy atom's element
        // count `bits / 1` while it still moves `bits / 8` bytes.
        self.arg_values = try scratch.alloc(Value, specs.len);
        for (specs, 0..) |a, i| {
            const raw = self.rawArg(i);
            self.arg_values[i] = if (a.dims) |dims| raw.view(self.staticRowMajor(dims)) else raw;
        }
    }

    pub fn rawArg(self: *Builder, i: usize) Value {
        const eb = self.entry_block orelse @panic("Builder.rawArg called before declareArgs");
        return .{ .inner = eb.argument(i), .kernel = self };
    }

    /// A tensor view when the spec has dims, else the raw pointer.
    pub fn arg(self: *Builder, i: usize) Value {
        return self.arg_values[i];
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

    pub fn loc(self: *const Builder) *const mlir.Location {
        return .unknown(self.ctx);
    }

    pub fn emit(self: *Builder, op: *mlir.Operation) Value {
        _ = op.appendTo(self.currentBlock());
        return .{ .inner = op.result(0), .kernel = self };
    }

    pub fn emitNone(self: *Builder, op: *mlir.Operation) void {
        _ = op.appendTo(self.currentBlock());
    }

    // ---------------------------------------------------------------- types

    /// Arena-allocated.
    pub fn readIntTuple(self: *Builder, tuple_ty: *const fly.types.IntTupleType) IntTuple {
        if (tuple_ty.isLeaf()) {
            return switch (tuple_ty.getLeaf()) {
                .static => |v| IntTuple.static(v),
                .dynamic, .basis => IntTuple.dyn,
                .none => IntTuple.star,
            };
        }
        const n = tuple_ty.getNumElements();
        const elems = self.alloc(IntTuple, n);
        for (0..n) |i| elems[i] = self.readIntTuple(tuple_ty.getElement(i));
        return .{ .tup = elems };
    }

    fn alloc(self: *Builder, comptime T: type, n: usize) []T {
        return self.arena.allocator().alloc(T, n) catch @panic("fly: OOM");
    }

    /// The shape and stride of a static `!fly.layout`, as host literals.
    pub fn readLayout(self: *Builder, layout_ty: *const mlir.Type) Layout {
        const lay = fly.expect(layout_ty, .layout);
        const shape = lay.getShape();
        const stride = lay.getStride();
        if (!shape.isStatic() or !stride.isStatic()) {
            std.debug.panic("fly: `{f}` is not static", .{layout_ty});
        }
        return .{ .shape = self.readIntTuple(shape), .stride = self.readIntTuple(stride) };
    }

    /// A fly type or attribute is rejected only on malformed input, which is
    /// a bug in the kernel, not a runtime condition.
    pub fn must(result: anytype) @typeInfo(@TypeOf(result)).error_union.payload {
        return result catch |err| std.debug.panic("fly: cannot build type ({})", .{err});
    }

    pub fn intTupleType(self: *Builder, t: IntTuple) *const mlir.Type {
        return must(fly.types.IntTupleType.get(self.ctx, .{ .attr = must(t.toAttr(self.ctx)) })).type_();
    }

    pub fn layoutType(self: *Builder, lay: Layout) *const mlir.Type {
        return must(fly.types.LayoutType.get(self.ctx, .{ .attr = must(lay.toAttr(self.ctx)) })).type_();
    }

    pub fn tileType(self: *Builder, t: Tile) *const mlir.Type {
        return must(fly.types.TileType.get(self.ctx, .{ .attr = must(t.toAttr(self.ctx)) })).type_();
    }

    pub fn ptrType(self: *Builder, dtype: DType, space: *const mlir.Attribute) *const mlir.Type {
        return must(fly.types.PointerType.get(self.ctx, .{ .elemTy = dtype.toMlir(self.ctx), .addressSpace = space })).type_();
    }

    fn addressSpaceAttr(self: *Builder, space: fly.AddressSpace) *const mlir.Attribute {
        return must(fly.attributes.AddressSpaceAttr.get(self.ctx, .{ .addressSpace = space })).attribute();
    }

    /// `!fly.memref<dtype, space, lay>`.
    pub fn memRefType(self: *Builder, dtype: DType, space: fly.AddressSpace, lay: Layout) *const mlir.Type {
        return must(fly.types.MemRefType.get(self.ctx, .{
            .elemTy = dtype.toMlir(self.ctx),
            .addressSpace = self.addressSpaceAttr(space),
            .layout = must(lay.toAttr(self.ctx)).attribute(),
        })).type_();
    }

    // ---------------------------------------------------------------- statics

    /// `fly.static` of an already-built type.
    pub fn staticTy(self: *Builder, t: *const mlir.Type) Value {
        return self.emit(fly.static(self.ctx, t, self.loc()));
    }

    pub fn static(self: *Builder, v: anytype) Value {
        const T = @TypeOf(v);
        return self.staticTy(if (T == Layout)
            self.layoutType(v)
        else if (T == Tile)
            self.tileType(v)
        else if (T == IntTuple)
            self.intTupleType(v)
        else
            @compileError("fly.static: expected a Layout, Tile or IntTuple, got " ++ @typeName(T)));
    }

    pub fn staticRowMajor(self: *Builder, dims: []const i64) Value {
        if (dims.len == 1) return self.static(Layout{ .shape = .static(dims[0]), .stride = .static(1) });
        const shape = self.alloc(IntTuple, dims.len);
        const stride = self.alloc(IntTuple, dims.len);
        var acc: i64 = 1;
        var i: usize = dims.len;
        while (i > 0) {
            i -= 1;
            shape[i] = .static(dims[i]);
            stride[i] = .static(acc);
            acc *= dims[i];
        }
        return self.static(Layout{ .shape = .{ .tup = shape }, .stride = .{ .tup = stride } });
    }

    /// A Value passes through; a bare tuple literal (`.{ 64, 8 }`) is a tile.
    pub fn lift(self: *Builder, v: anytype) Value {
        const T = @TypeOf(v);
        if (T == Value) return v;
        if (T == Layout or T == Tile or T == IntTuple) return self.static(v);
        if (@typeInfo(T) == .@"struct" and @typeInfo(T).@"struct".is_tuple) return self.static(tile(v));
        @compileError("fly: expected a Value, Layout, Tile, IntTuple or tuple literal, got " ++ @typeName(T));
    }

    /// A (possibly nested) int tuple from integers (static leaves), `null`
    /// (`*`), `IntTuple` literals and `Value`s (dynamic leaves, passed as
    /// operands). An `!fly.int_tuple` Value passes through.
    pub fn intTuple(self: *Builder, elems: anytype) Value {
        if (@TypeOf(elems) == Value and elems.kind() == .int_tuple) return elems;
        var operands: stdx.BoundedArray(*const mlir.Value, 32) = .empty;
        const spec = self.intTupleSpec(&operands, elems);
        const t = self.intTupleType(spec);
        return self.emit(fly.makeIntTuple(self.ctx, operands.constSlice(), t, self.loc()));
    }

    /// `intTuple`, spelled for coordinates.
    pub const coord = intTuple;

    /// The literal for `intTuple`, collecting each dynamic leaf as an operand.
    fn intTupleSpec(self: *Builder, operands: *stdx.BoundedArray(*const mlir.Value, 32), v: anytype) IntTuple {
        const T = @TypeOf(v);
        if (T == Value) {
            if (v.kind() == .int_tuple) {
                std.debug.panic("fly.intTuple: nested !fly.int_tuple values are not supported; pass scalars", .{});
            }
            operands.appendAssumeCapacity(v.inner);
            return if (v.isI64()) .dynamic(64, 1) else .dyn;
        }
        if (T == IntTuple) return v;
        if (T == @TypeOf(null)) return .star;
        switch (@typeInfo(T)) {
            .comptime_int, .int => return .static(@intCast(v)),
            .@"struct" => |s| {
                if (!s.is_tuple) @compileError("fly.intTuple: expected an integer, null, Value, IntTuple or tuple, got " ++ @typeName(T));
                const elems = self.alloc(IntTuple, s.fields.len);
                inline for (s.fields, 0..) |f, i| elems[i] = self.intTupleSpec(operands, @field(v, f.name));
                return .{ .tup = elems };
            },
            else => @compileError("fly.intTuple: expected an integer, null, Value, IntTuple or tuple, got " ++ @typeName(T)),
        }
    }

    // ---------------------------------------------------------------- scalars & ids

    /// Only the representation `dtype` needs is computed: a float constant
    /// outside i64 range (inf, floatMax) must not be forced through
    /// `@intFromFloat`, which traps.
    pub fn constant(self: *Builder, dtype: DType, value: anytype) Value {
        const is_float_lit = switch (@typeInfo(@TypeOf(value))) {
            .float, .comptime_float => true,
            else => false,
        };
        return switch (dtype) {
            .f16, .bf16, .f32, .f64 => blk: {
                const f: f64 = if (is_float_lit) @floatCast(value) else @floatFromInt(value);
                break :blk self.emit(switch (dtype) {
                    .f16 => arith.constant_float(self.ctx, f, .f16, self.loc()),
                    .bf16 => arith.constant_float(self.ctx, f, .bf16, self.loc()),
                    .f32 => arith.constant_float(self.ctx, f, .f32, self.loc()),
                    else => arith.constant_float(self.ctx, f, .f64, self.loc()),
                });
            },
            .i1, .i8, .i16, .i32, .i64 => blk: {
                const i: i64 = if (is_float_lit) @intFromFloat(value) else @intCast(value);
                break :blk self.emit(arith.constant_int(self.ctx, i, dtype.toMlir(self.ctx), self.loc()));
            },
            else => std.debug.panic("fly.constant: no constant for {s}", .{@tagName(dtype)}),
        };
    }

    pub fn constIndex(self: *Builder, value: i64) Value {
        return self.emit(arith.constant_index(self.ctx, value, self.loc()));
    }

    fn idI32(self: *Builder, op: *mlir.Operation) Value {
        return self.emit(op).toI32();
    }

    /// Already cast to `i32`: `fly.make_int_tuple` takes only i32/i64.
    pub fn threadId(self: *Builder, dim: Dim) Value {
        return self.idI32(gpu.thread_id(self.ctx, dim, self.loc()));
    }
    pub fn blockId(self: *Builder, dim: Dim) Value {
        return self.idI32(gpu.block_id(self.ctx, dim, self.loc()));
    }
    pub fn blockDim(self: *Builder, dim: Dim) Value {
        return self.idI32(gpu.block_dim(self.ctx, dim, self.loc()));
    }
    pub fn gridDim(self: *Builder, dim: Dim) Value {
        return self.idI32(gpu.grid_dim(self.ctx, dim, self.loc()));
    }

    pub fn barrier(self: *Builder) void {
        self.emitNone(gpu.barrier(self.ctx, self.loc()));
    }

    /// Tag float arithmetic emitted from now on with `fastmath<fast>`.
    pub fn setFastMath(self: *Builder, on: bool) void {
        self.fast_math = on;
    }

    pub fn emitFast(self: *Builder, op: *mlir.Operation) Value {
        if (self.fast_math and op.numResults() > 0 and (Value{ .inner = op.result(0), .kernel = self }).isFloatElem()) {
            op.setAttributeByName("fastmath", fly.parseAttr(self.ctx, "#arith.fastmath<fast>"));
        }
        return self.emit(op);
    }

    pub fn rsqrt(self: *Builder, x: Value) Value {
        return self.emitFast(dialects.math.rsqrt(self.ctx, x.inner, self.loc()));
    }
    pub fn sqrt(self: *Builder, x: Value) Value {
        return self.emitFast(dialects.math.sqrt(self.ctx, x.inner, self.loc()));
    }
    pub fn exp(self: *Builder, x: Value) Value {
        return self.emitFast(dialects.math.exp(self.ctx, x.inner, self.loc()));
    }

    /// Static LDS, as an `n:1` tensor in shared memory. Each call is its own
    /// `fly.make_ptr {allocBytes, allocAlign}` leaf, which the plugin lowers
    /// to one LLVM global, so `shared_mem_bytes` may stay 0.
    pub fn sharedArray(self: *Builder, dtype: DType, n: i64, alignment: i32) Value {
        const bytes: i64 = @divExact(@as(i64, dtype.bitWidth()) * n, 8);
        const raw_ty = must(fly.types.PointerType.get(self.ctx, .{
            .elemTy = .int(self.ctx, .i8),
            .addressSpace = self.addressSpaceAttr(.shared),
            .alignment = must(fly.attributes.AlignAttr.get(self.ctx, .{ .alignment = alignment })),
        })).type_();
        const dict: *const mlir.Attribute = .dict(self.ctx, &.{
            .named(self.ctx, "allocBytes", .int(self.ctx, .i64, bytes)),
            .named(self.ctx, "allocAlign", .int(self.ctx, .i64, alignment)),
        });
        const raw = self.emit(fly.makePtr(self.ctx, &.{}, raw_ty, dict, self.loc()));
        const typed = self.recast(raw, dtype);
        return typed.view(self.static(Layout{ .shape = .static(n), .stride = .static(1) }));
    }

    /// A single atom-sized copy.
    pub fn copyAtomCall(self: *Builder, atom: Value, src: Value, dst: Value) void {
        self.emitNone(fly.effect(self.ctx, "copy_atom_call", &.{ atom.inner, src.inner, dst.inner }, .empty, self.loc()));
    }

    /// One atom-sized load straight into a `vector<NxT>`.
    ///
    /// Required over an rmem tensor + `copyAtomCall` + `load` in any kernel
    /// with control flow: the plugin lowers `scf` to `cf` before the Fly
    /// passes, and Fly's register-alloca promotion only handles one block.
    pub fn copyAtomLoad(self: *Builder, atom: Value, src: Value) Value {
        const n = src.sizeStatic();
        const dt = src.elemDType();
        const vec_ty = mlir.Type.vector(&.{n}, dt.toMlir(self.ctx));
        const op = fly.make(self.ctx, "fly.copy_atom_call_ssa", .{
            .operands = .{ .flat = &.{ atom.inner, src.inner } },
            .results = .{ .flat = &.{vec_ty} },
            .attributes = &.{.named(self.ctx, "operandSegmentSizes", .denseArray(self.ctx, .i32, &.{ 1, 1, 0, 0 }))},
            .location = self.loc(),
        });
        return self.emit(op);
    }

    /// One atom-sized store of a `vector<NxT>`; see `copyAtomLoad`.
    pub fn copyAtomStore(self: *Builder, atom: Value, vec: Value, dst: Value) void {
        const op = fly.make(self.ctx, "fly.copy_atom_call_ssa", .{
            .operands = .{ .flat = &.{ atom.inner, vec.inner, dst.inner } },
            .attributes = &.{.named(self.ctx, "operandSegmentSizes", .denseArray(self.ctx, .i32, &.{ 1, 1, 1, 0 }))},
            .location = self.loc(),
        });
        self.emitNone(op);
    }

    // ---------------------------------------------------------------- tensors

    pub fn makeView(self: *Builder, iter: Value, lay: anytype) Value {
        _ = self;
        return iter.view(lay);
    }

    /// A coordinate tensor: value == logical coord.
    pub fn identity(self: *Builder, shape: anytype) Value {
        const shp = self.intTuple(shape);
        const lay = self.emit(fly.inferred(self.ctx, "make_identity_layout", &.{shp.inner}, .empty, self.loc()));
        const rank_ = shp.intTupleStatic().rank();
        const zero: IntTuple = if (rank_ == 1) .static(0) else blk: {
            const zeros = self.alloc(IntTuple, rank_);
            for (zeros) |*z| z.* = .static(0);
            break :blk .{ .tup = zeros };
        };
        const base_ty = self.intTupleType(zero);
        const base = self.emit(fly.makeIntTuple(self.ctx, &.{}, base_ty, self.loc()));
        return base.view(lay);
    }

    /// A register tensor; see `copyAtomLoad` before using one in a kernel
    /// with control flow.
    pub fn rmemTensor(self: *Builder, comptime lay: Layout, dtype: DType) Value {
        const lay_v = self.static(lay);
        const t = self.memRefType(dtype, .register, lay);
        return self.emit(fly.typed(self.ctx, "memref.alloca", &.{lay_v.inner}, &.{t}, .empty, self.loc()));
    }

    /// `rmemTensor` with a runtime `Layout`.
    pub fn rmemTensorRuntime(self: *Builder, lay: Layout, dtype: DType) Value {
        const lay_v = self.static(lay);
        const t = self.memRefType(dtype, .register, lay);
        return self.emit(fly.typed(self.ctx, "memref.alloca", &.{lay_v.inner}, &.{t}, .empty, self.loc()));
    }

    /// Reinterpret a pointer as another element type, keeping the address
    /// space, alignment and swizzle.
    pub fn recast(self: *Builder, ptr: Value, dtype: DType) Value {
        const t = must(fly.ptrWithElem(self.ctx, ptr.type_(), dtype.toMlir(self.ctx))).type_();
        return self.emit(fly.typed(self.ctx, "recast_iter", &.{ptr.inner}, &.{t}, .empty, self.loc()));
    }

    // ---------------------------------------------------------------- atoms

    /// The copy-op type an atom wraps, by bit size.
    pub const CopyOp = union(enum) {
        universal: i32,
        buffer_copy: i32,
        buffer_copy_lds: i32,

        fn type_(self: CopyOp, ctx: *mlir.Context) *const mlir.Type {
            return switch (self) {
                .universal => |b| must(fly.types.CopyOpUniversalCopyType.get(ctx, .{ .bitSize = b })).type_(),
                .buffer_copy => |b| must(fly.rocdl.CopyOpCDNA3BufferCopyType.get(ctx, .{ .bitSize = b })).type_(),
                .buffer_copy_lds => |b| must(fly.rocdl.CopyOpCDNA3BufferCopyLDSType.get(ctx, .{ .bitSize = b })).type_(),
            };
        }
    };

    pub fn copyAtom(self: *Builder, op: CopyOp, dtype: DType) Value {
        const bits: i32 = @intCast(dtype.bitWidth());
        const t = must(fly.types.CopyAtomType.get(self.ctx, .{ .copyOp = op.type_(self.ctx), .valBits = bits })).type_();
        return self.emit(fly.makeCopyAtom(self.ctx, t, bits, self.loc()));
    }

    /// `op` is an MMA op type, such as a `fly.rocdl.MmaOpCDNA3MFMAType`.
    pub fn mmaAtom(self: *Builder, op: *const mlir.Type) Value {
        const t = must(fly.types.MmaAtomType.get(self.ctx, .{ .mmaOp = op })).type_();
        return self.emit(fly.makeMmaAtom(self.ctx, t, self.loc()));
    }

    pub fn tiledCopy(self: *Builder, atom: Value, layout_tv: anytype, tile_mn: anytype) TiledCopy {
        const tv = self.lift(layout_tv);
        const tl = self.lift(tile_mn);
        const v = self.emit(fly.inferred(self.ctx, "make_tiled_copy", &.{ atom.inner, tv.inner, tl.inner }, .empty, self.loc()));
        return .{ .value = v, .tile = tl };
    }

    /// raked product -> right inverse -> composition -> make_tiled_copy,
    /// with the tiler read off the inferred product shape.
    pub fn tiledCopyTV(self: *Builder, atom: Value, thr: Layout, val: Layout) TiledCopy {
        const thr_size = thr.size() orelse @panic("fly.tiledCopyTV: thr_layout must be static");
        const val_size = val.size() orelse @panic("fly.tiledCopyTV: val_layout must be static");
        const layout_mn = self.static(thr).rakedProduct(self.static(val));
        const tmp = self.static(Layout{
            .shape = .{ .tup = &.{ .static(thr_size), .static(val_size) } },
            .stride = .{ .tup = &.{ .static(1), .static(thr_size) } },
        });
        const layout_tv = layout_mn.rightInverse().composition(tmp);
        const tiler = layout_mn.emitShape().productEach();
        // The tiler as a tile literal: one leaf per mode, `[a|b]`.
        const shape = tiler.intTupleStatic();
        const tile_modes = self.alloc(Tile, shape.rank());
        for (tile_modes, 0..) |*m, i| m.* = .{ .leaf = switch (shape.at(i)) {
            .leaf => |l| l,
            else => std.debug.panic("fly.tiledCopyTV: tiler mode {d} is not a leaf", .{i}),
        } };
        return self.tiledCopy(atom, layout_tv, Tile{ .modes = tile_modes });
    }

    /// A TiledCopy matched to an MMA operand: the TV layout and tile size
    /// come off the `!fly.tiled_mma` type.
    pub fn tiledCopyFor(self: *Builder, operand: fly.MmaOperand, copy_atom: Value, tm: TiledMma) TiledCopy {
        const tm_ty = fly.expect(tm.value.type_(), .tiled_mma);
        const layout_tv = self.staticTy((switch (operand) {
            .a => tm_ty.getTiledThrValLayoutA(),
            .b => tm_ty.getTiledThrValLayoutB(),
            .c => tm_ty.getTiledThrValLayoutC(),
            .d => @panic("fly.tiledCopyFor: no D operand copy"),
        }).type_());
        const tile_size = self.staticTy(tm_ty.getTileSizeMNK().type_());
        const modes: [2]i32 = switch (operand) {
            .a => .{ 0, 2 },
            .b => .{ 1, 2 },
            .c => .{ 0, 1 },
            .d => unreachable,
        };
        const one = self.intTuple(1);
        const tile_modes = self.alloc(Tile, modes.len);
        for (modes, 0..) |m, i| {
            const lay = self.emit(fly.inferred(self.ctx, "make_layout", &.{ tile_size.selectModes(&.{m}).inner, one.inner }, .empty, self.loc()));
            tile_modes[i] = .{ .layout = self.readLayout(lay.type_()) };
        }
        return self.tiledCopy(copy_atom, layout_tv, Tile{ .modes = tile_modes });
    }

    pub fn tiledCopyA(self: *Builder, copy_atom: Value, tm: TiledMma) TiledCopy {
        return self.tiledCopyFor(.a, copy_atom, tm);
    }
    pub fn tiledCopyB(self: *Builder, copy_atom: Value, tm: TiledMma) TiledCopy {
        return self.tiledCopyFor(.b, copy_atom, tm);
    }
    pub fn tiledCopyC(self: *Builder, copy_atom: Value, tm: TiledMma) TiledCopy {
        return self.tiledCopyFor(.c, copy_atom, tm);
    }

    /// The same tensor over a CDNA buffer resource
    /// (`!fly.ptr<T, #fly_rocdl.buffer_desc>`), for hardware bounds-checked
    /// accesses and `buffer_copy` atoms. Descriptor size is the maximum.
    pub fn bufferTensor(self: *Builder, t: Value) Value {
        const ptr = t.emitIter();
        const lay = t.emitLayout();
        const c0 = self.constant(.i16, 0);
        const nrec = self.constant(.i64, 0xFFFFFFFF);
        // (7 << 12) | (4 << 15): the CDNA descriptor flags FlyDSL and XLA use.
        const flags = self.constant(.i32, 0x27000);
        const elem = t.elemDType();
        const buf_ty = self.ptrType(elem, (must(fly.rocdl.BufferDescAddressAttr.get(self.ctx))).attribute());
        const buf_ptr = self.emit(fly.makePtr(self.ctx, &.{ ptr.inner, c0.inner, nrec.inner, flags.inner }, buf_ty, null, self.loc()));
        return buf_ptr.view(lay);
    }

    pub fn tiledMma(self: *Builder, atom: Value, atom_layout: anytype, permutation: ?Value) TiledMma {
        const al = self.lift(atom_layout);
        const v = if (permutation) |p|
            self.emit(fly.inferred(self.ctx, "make_tiled_mma", &.{ atom.inner, al.inner, p.inner }, .empty, self.loc()))
        else
            self.emit(fly.inferred(self.ctx, "make_tiled_mma", &.{ atom.inner, al.inner }, .empty, self.loc()));
        return .{ .value = v };
    }

    pub const CopyOpts = struct {
        pred: ?Value = null,
    };

    pub fn copy(self: *Builder, atom: Value, src: Value, dst: Value, opts: CopyOpts) void {
        self.emitNone(fly.copy(self.ctx, atom.inner, src.inner, dst.inner, .{
            .pred = if (opts.pred) |p| p.inner else null,
        }, self.loc()));
    }

    pub fn gemm(self: *Builder, atom: Value, d: Value, a: Value, b: Value, c: Value) void {
        self.emitNone(fly.gemm(self.ctx, atom.inner, d.inner, a.inner, b.inner, c.inner, .{}, self.loc()));
    }

    // ---------------------------------------------------------------- control flow

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

    /// Loop-carried `inits`; the induction variable is `index` unless
    /// `lower` is a Value.
    pub fn openFor(
        self: *Builder,
        lower: anytype,
        upper: anytype,
        step: anytype,
        inits: anytype,
    ) ForScope(tupleArity(@TypeOf(inits), "openFor: inits")) {
        const N = comptime tupleArity(@TypeOf(inits), "openFor: inits");
        const fields = @typeInfo(@TypeOf(inits)).@"struct".fields;

        const lb_v: Value = if (@TypeOf(lower) == Value) lower else self.constIndex(lower);
        const ub_v: Value = if (@TypeOf(upper) == Value) upper else self.constLike(upper, lb_v);
        const step_v: Value = if (@TypeOf(step) == Value) step else self.constLike(step, lb_v);

        var block_types: [N + 1]*const mlir.Type = undefined;
        var block_locs: [N + 1]*const mlir.Location = undefined;
        block_types[0] = lb_v.type_();
        block_locs[0] = self.loc();
        var inits_inner: [N]*const mlir.Value = undefined;
        inline for (fields, 0..) |f, i| {
            if (f.type != Value) @compileError("openFor: every init must be a Value");
            const v: Value = @field(inits, f.name);
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

    pub fn constLike(self: *Builder, value: anytype, like: Value) Value {
        if (like.type_().isA(mlir.IndexType) != null) {
            const i: i64 = switch (@typeInfo(@TypeOf(value))) {
                .float, .comptime_float => @intFromFloat(value),
                else => @intCast(value),
            };
            return self.constIndex(i);
        }
        return self.constant(like.scalarDType(), value);
    }

    pub fn openIf(self: *Builder, cond: Value) IfOnlyScope {
        const then_block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(then_block);
        return .{ .kernel = self, .cond_inner = cond.inner, .then_block = then_block };
    }

    pub fn openIfElse(self: *Builder, cond: Value, result_types: anytype) IfScope(tupleArity(@TypeOf(result_types), "openIfElse: result_types")) {
        const N = comptime tupleArity(@TypeOf(result_types), "openIfElse: result_types");
        const fields = @typeInfo(@TypeOf(result_types)).@"struct".fields;
        var types: [N]*const mlir.Type = undefined;
        inline for (fields, 0..) |f, i| {
            if (f.type != *const mlir.Type) @compileError("openIfElse: every result_type must be *const mlir.Type");
            types[i] = @field(result_types, f.name);
        }
        const then_block = mlir.Block.init(&.{}, &.{});
        const else_block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(then_block);
        return .{ .kernel = self, .cond_inner = cond.inner, .then_block = then_block, .else_block = else_block, .result_types = types };
    }

    // ---------------------------------------------------------------- finalization

    /// Append `gpu.return`, verify, and print the module (no debug info: XLA
    /// scrubs locations). Owned by the builder's allocator.
    pub fn finish(self: *Builder) FinishError![:0]const u8 {
        _ = gpu.return_(self.ctx, &.{}, self.loc()).appendTo(self.currentBlock());

        if (!self.module.operation().verify()) {
            std.log.err("fly: module failed to verify:\n{f}", .{self.module.operation().fmt(.{ .debug_info = false })});
            return error.InvalidMlir;
        }

        var al: std.Io.Writer.Allocating = .init(self.allocator);
        defer al.deinit();
        try al.writer.print("{f}", .{self.module.operation().fmt(.{ .debug_info = false })});
        return try self.allocator.dupeZ(u8, al.written());
    }
};

// =============================================================================
// Tests
// =============================================================================

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    fly.insertDialects(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

/// C = A + B over (M, N) f32, 128-bit vectorized and predicated on the
/// ragged edge.
pub fn emitVectorAdd(b: *Builder, thr_layout: Layout, val_layout: Layout, a_in: Value, b_in: Value, c_out: Value) void {
    const shape = a_in.shapeStatic();
    const m = shape.at(0).leaf.s;
    const n = shape.at(1).leaf.s;

    const atom = b.copyAtom(.{ .universal = 128 }, .f32);
    const tc = b.tiledCopyTV(atom, thr_layout, val_layout);

    const tid = b.threadId(.x);
    const bx = b.blockId(.x);
    const by = b.blockId(.y);

    const idC = b.identity(.{ m, n });
    const tile_mn = tc.tileMN();

    const gA = a_in.flatDivide(tile_mn).slice(.{ null, null, bx, by });
    const gB = b_in.flatDivide(tile_mn).slice(.{ null, null, bx, by });
    const gC = c_out.flatDivide(tile_mn).slice(.{ null, null, bx, by });
    const cC = idC.flatDivide(tile_mn).slice(.{ null, null, bx, by });

    const thr = tc.getSlice(tid);
    const tgA = thr.partitionS(gA);
    const tgB = thr.partitionS(gB);
    const tgC = thr.partitionD(gC);
    const tcC = thr.partitionS(cC).slice(.{ .{ 0, null }, null, null });

    const rA = tgA.makeFragmentLike(null);
    const rB = tgB.makeFragmentLike(null);
    const rC = tgC.makeFragmentLike(null);
    const pC = tcC.makeFragmentLike(.i1);

    for (0..@intCast(pC.sizeStatic())) |i| {
        pC.set(i, tcC.at(i).elemLess(.{ m, n }));
    }

    b.copy(atom, tgA, rA, .{ .pred = pC });
    b.copy(atom, tgB, rB, .{ .pred = pC });
    rC.store(rA.load().add(rB.load()));
    b.copy(atom, rC, tgC, .{ .pred = pC });
}

test "vectorAdd builds, verifies and re-parses" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "vector_add");
    defer b.deinit();
    const a = try b.declareArgs(.{
        .a = .{ .tensor = .{ .dtype = .f32, .dims = &.{ 100, 1000 } } },
        .b = .{ .tensor = .{ .dtype = .f32, .dims = &.{ 100, 1000 } } },
        .c = .{ .tensor = .{ .dtype = .f32, .dims = &.{ 100, 1000 } } },
    });
    emitVectorAdd(&b, ordered(.{ 8, 16 }, .{ 1, 0 }), ordered(.{ 1, 4 }, .{ 0, 1 }), a.a, a.b, a.c);
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);

    const expected = [_][]const u8{
        "module attributes {gpu.container_module}",
        "gpu.module @zml_fly_kernels",
        "gpu.func @vector_add(%arg0: !fly.ptr<f32, global>, %arg1: !fly.ptr<f32, global>, %arg2: !fly.ptr<f32, global>) kernel",
        "fly.static : !fly.layout<(100,1000):(1000,1)>",
        "fly.make_copy_atom {valBits = 32 : i32} : !fly.copy_atom<!fly.universal_copy<128>, 32>",
        "fly.raked_product",
        "fly.static : !fly.tile<[8|64]>",
        "fly.make_tiled_copy",
        "gpu.thread_id x",
        "fly.tiled_copy.partition_src",
        "fly.tiled_copy.partition_dst",
        "fly.make_fragment_like",
        "fly.elem_less",
        "fly.copy(",
        "fly.memref.load_vec",
        "arith.addf",
        "fly.memref.store_vec",
        "gpu.return",
    };
    for (expected) |needle| {
        if (std.mem.indexOf(u8, ir, needle) == null) {
            std.debug.print("missing `{s}` in:\n{s}\n", .{ needle, ir });
            return error.TestUnexpectedResult;
        }
    }

    const reparsed = try mlir.Module.parse(ctx, ir);
    defer reparsed.deinit();
    try std.testing.expect(reparsed.operation().verify());
}

/// The matrix atom a device family provides. CDNA has MFMA over f32; RDNA has
/// WMMA, whose verifier requires M=N=K=16 and rejects f32 operands, so the
/// operand type and the K tile change with the family, not just the mnemonic.
pub const MmaFlavor = enum {
    cdna3_mfma,
    gfx11_wmma,
    gfx120x_wmma,

    /// MFMA takes f32 operands at 16x16x4; both WMMAs take f16 at 16x16x16.
    pub fn atomType(self: MmaFlavor, ctx: *mlir.Context) *const mlir.Type {
        const operand = self.operandDType().toMlir(ctx);
        const acc: *const mlir.Type = .float(ctx, .f32);
        return switch (self) {
            .cdna3_mfma => (fly.rocdl.MmaOpCDNA3MFMAType.get(ctx, .{
                .m = 16,
                .n = 16,
                .k = 4,
                .elemTyA = operand,
                .elemTyB = operand,
                .elemTyAcc = acc,
            }) catch unreachable).type_(),
            inline .gfx11_wmma, .gfx120x_wmma => |f| blk: {
                const T = if (f == .gfx11_wmma) fly.rocdl.MmaOpGFX11WMMAType else fly.rocdl.MmaOpGFX120XWMMAType;
                break :blk (T.get(ctx, .{
                    .m = 16,
                    .n = 16,
                    .k = 16,
                    .elemTyA = operand,
                    .elemTyB = operand,
                    .elemTyAcc = acc,
                }) catch unreachable).type_();
            },
        };
    }

    /// A and B operand type. The accumulator is f32 either way.
    pub fn operandDType(self: MmaFlavor) DType {
        return switch (self) {
            .cdna3_mfma => .f32,
            .gfx11_wmma, .gfx120x_wmma => .f16,
        };
    }

    /// Threads one atom occupies: MFMA runs on a wave64, WMMA on a wave32.
    /// Checked against the dialect's own `mma_atom.thr_layout` below.
    pub fn atomThreads(self: MmaFlavor) i32 {
        return switch (self) {
            .cdna3_mfma => 64,
            .gfx11_wmma, .gfx120x_wmma => 32,
        };
    }

    /// `emitTiledMma` lays the atom out (2,2,1), so a block is four of them.
    pub fn blockThreads(self: MmaFlavor) i32 {
        return 4 * self.atomThreads();
    }

    /// The block tile's K, a multiple of the atom's K.
    pub fn blockK(self: MmaFlavor) i64 {
        return switch (self) {
            .cdna3_mfma => 8,
            .gfx11_wmma, .gfx120x_wmma => 16,
        };
    }

    /// Buffer copies are CDNA-only: FlyROCDL declares no RDNA copy atom, so
    /// RDNA moves through the core universal copy over plain tensors.
    fn hasBufferCopy(self: MmaFlavor) bool {
        return self == .cdna3_mfma;
    }
};

/// A two-mode tile from extents known only at emit time.
fn tile2(b: *Builder, x: i64, y: i64) Tile {
    const modes = b.alloc(Tile, 2);
    modes[0] = .{ .leaf = .{ .s = x } };
    modes[1] = .{ .leaf = .{ .s = y } };
    return .{ .modes = modes };
}

/// C = A @ B^T in one block of 256 threads, tiled (2,2,1) over the family's
/// matrix atom.
pub fn emitTiledMma(b: *Builder, flavor: MmaFlavor, a_in: Value, b_in: Value, c_out: Value) void {
    const block_m = 64;
    const block_n = 64;
    const block_k = flavor.blockK();

    const tid = b.threadId(.x);
    const bid = b.blockId(.x);

    const A = if (flavor.hasBufferCopy()) b.bufferTensor(a_in) else a_in;
    const B = if (flavor.hasBufferCopy()) b.bufferTensor(b_in) else b_in;
    const C = if (flavor.hasBufferCopy()) b.bufferTensor(c_out) else c_out;

    const bA = A.zippedDivide(tile2(b, block_m, block_k)).slice(.{ null, bid });
    const bB = B.zippedDivide(tile2(b, block_n, block_k)).slice(.{ null, bid });
    const bC = C.zippedDivide(tile2(b, block_m, block_n)).slice(.{ null, bid });

    const mma_atom = b.mmaAtom(flavor.atomType(b.ctx));
    const tiled_mma = b.tiledMma(mma_atom, L(.{ 2, 2, 1 }, .{ 1, 2, 0 }), null);

    // A and B move at the operand type, C at the f32 accumulator type. One
    // element per instruction: a WMMA fragment's values are strided in memory,
    // so a wider copy would move the wrong neighbours.
    const copy_ab: Builder.CopyOp = if (flavor.hasBufferCopy()) .{ .buffer_copy = 32 } else .{ .universal = @intCast(flavor.operandDType().bitWidth()) };
    const copy_c: Builder.CopyOp = if (flavor.hasBufferCopy()) .{ .buffer_copy = 32 } else .{ .universal = @intCast(c_out.elemDType().bitWidth()) };
    const atom_ab = b.copyAtom(copy_ab, flavor.operandDType());
    const atom_c = b.copyAtom(copy_c, c_out.elemDType());

    const tiled_copy_a = b.tiledCopyA(atom_ab, tiled_mma);
    const tiled_copy_b = b.tiledCopyB(atom_ab, tiled_mma);
    const tiled_copy_c = b.tiledCopyC(atom_c, tiled_mma);

    const thr_copy_a = tiled_copy_a.getSlice(tid);
    const thr_copy_b = tiled_copy_b.getSlice(tid);
    const thr_copy_c = tiled_copy_c.getSlice(tid);

    const copy_src_a = thr_copy_a.partitionS(bA);
    const copy_src_b = thr_copy_b.partitionS(bB);
    const copy_dst_c = thr_copy_c.partitionS(bC);

    const frag_a = tiled_mma.makeFragmentA(bA);
    const frag_b = tiled_mma.makeFragmentB(bB);
    const frag_c = tiled_mma.makeFragmentC(bC);

    const copy_frag_a = thr_copy_a.retile(frag_a);
    const copy_frag_b = thr_copy_b.retile(frag_b);
    const copy_frag_c = thr_copy_c.retile(frag_c);

    b.copy(atom_ab, copy_src_a, copy_frag_a, .{});
    b.copy(atom_ab, copy_src_b, copy_frag_b, .{});

    frag_c.fill(0);
    b.gemm(mma_atom, frag_c, frag_a, frag_b, frag_c);

    b.copy(atom_c, copy_frag_c, copy_dst_c, .{});
}

test "tiledMma builds, verifies and re-parses" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "tiled_mma");
    defer b.deinit();
    const a = try b.declareArgs(.{
        .a = .{ .tensor = .{ .dtype = .f32, .dims = &.{ 64, 8 } } },
        .b = .{ .tensor = .{ .dtype = .f32, .dims = &.{ 64, 8 } } },
        .c = .{ .tensor = .{ .dtype = .f32, .dims = &.{ 64, 64 } } },
    });
    emitTiledMma(&b, .cdna3_mfma, a.a, a.b, a.c);
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);

    const expected = [_][]const u8{
        "!fly.ptr<f32, #fly_rocdl.buffer_desc>",
        "fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x4, (f32, f32) -> f32>>",
        "fly.make_tiled_mma",
        "fly.static : !fly.int_tuple<(32,32,4)>",
        "fly.mma.make_fragment",
        "fly.tiled_copy.retile",
        "vector.broadcast",
        "fly.gemm(",
    };
    for (expected) |needle| {
        if (std.mem.indexOf(u8, ir, needle) == null) {
            std.debug.print("missing `{s}` in:\n{s}\n", .{ needle, ir });
            return error.TestUnexpectedResult;
        }
    }
    const reparsed = try mlir.Module.parse(ctx, ir);
    defer reparsed.deinit();
    try std.testing.expect(reparsed.operation().verify());
}

test "MmaFlavor.atomThreads matches the dialect" {
    const ctx = try testContext();
    defer ctx.deinit();
    inline for (std.meta.fields(MmaFlavor)) |field| {
        const flavor: MmaFlavor = @enumFromInt(field.value);
        const atom = try fly.types.MmaAtomType.get(ctx, .{ .mmaOp = flavor.atomType(ctx) });
        // `mma_atom.thr_layout` is the wavefront an atom occupies; the launch
        // geometry is derived from it, so a wrong table silently corrupts C.
        const threads = switch (atom.getThrLayout().getShape().getLeaf()) {
            .static => |v| v,
            else => return error.TestUnexpectedResult,
        };
        try std.testing.expectEqual(@as(i64, flavor.atomThreads()), threads);
    }
}

test "tiledMma builds for the RDNA WMMA atom" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "tiled_mma_wmma");
    defer b.deinit();
    const k = MmaFlavor.gfx11_wmma.blockK();
    const a = try b.declareArgs(.{
        .a = .{ .tensor = .{ .dtype = .f16, .dims = &.{ 64, k } } },
        .b = .{ .tensor = .{ .dtype = .f16, .dims = &.{ 64, k } } },
        .c = .{ .tensor = .{ .dtype = .f32, .dims = &.{ 64, 64 } } },
    });
    emitTiledMma(&b, .gfx11_wmma, a.a, a.b, a.c);
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);

    // No buffer descriptor: FlyROCDL declares no RDNA copy atom, so this path
    // moves through the core universal copy over plain tensors.
    if (std.mem.indexOf(u8, ir, "buffer_desc") != null) {
        std.debug.print("unexpected buffer descriptor on the WMMA path:\n{s}\n", .{ir});
        return error.TestUnexpectedResult;
    }
    for ([_][]const u8{
        "!fly_rocdl.gfx11.wmma<16x16x16, (f16, f16) -> f32",
        "!fly.copy_atom<!fly.universal_copy<16>, 16>",
        "fly.gemm(",
    }) |needle| {
        if (std.mem.indexOf(u8, ir, needle) == null) {
            std.debug.print("missing `{s}` in:\n{s}\n", .{ needle, ir });
            return error.TestUnexpectedResult;
        }
    }

    const reparsed = try mlir.Module.parse(ctx, ir);
    defer reparsed.deinit();
    try std.testing.expect(reparsed.operation().verify());
}

test "scalar conversions and constants" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "conv");
    defer b.deinit();
    const a = try b.declareArgs(.{
        .x = .{ .tensor = .{ .dtype = .f16, .dims = &.{16} } },
        .p = .{ .tensor = .{ .dtype = .i1, .dims = &.{16} } },
    });

    // A predicate is unsigned: `true` must convert to 1, never -1.
    const pred = b.constant(.i32, 3).cmp(.lt, 7);
    _ = pred.to(.f32);
    _ = pred.to(.i32);
    // Equal-width float pairs are neither ext nor trunc.
    _ = b.constant(.f16, 1.0).to(.bf16);
    // A float constant outside i64 range must not go through @intFromFloat.
    _ = b.constant(.f32, -std.math.inf(f32));
    _ = b.constant(.f32, std.math.floatMax(f32));
    // A `.bool` argument keeps its storage element type end to end.
    try std.testing.expectEqual(DType.i8, a.p.elemDType());
    try std.testing.expectEqual(DType.f16, a.x.elemDType());

    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    for ([_][]const u8{
        "arith.uitofp",
        "arith.extui",
        "arith.convertf",
        "arith.constant 0xFF800000 : f32",
        "!fly.memref<i8, global, 16:1>",
    }) |needle| {
        if (std.mem.indexOf(u8, ir, needle) == null) {
            std.debug.print("missing `{s}` in:\n{s}\n", .{ needle, ir });
            return error.TestUnexpectedResult;
        }
    }
    if (std.mem.indexOf(u8, ir, "arith.sitofp") != null or std.mem.indexOf(u8, ir, "arith.extsi") != null) {
        std.debug.print("i1 was sign-extended:\n{s}\n", .{ir});
        return error.TestUnexpectedResult;
    }
}

test "static queries read inferred types" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "q");
    defer b.deinit();
    const a = try b.declareArgs(.{ .x = .{ .tensor = .{ .dtype = .bf16, .dims = &.{ 64, 32 } } } });
    try std.testing.expectEqual(@as(i64, 64 * 32), a.x.sizeStatic());
    try std.testing.expectEqual(DType.bf16, a.x.elemDType());
    try std.testing.expectEqual(Value.Kind.memref, a.x.kind());
    const tiled = a.x.flatDivide(tile(.{ 16, 8 }));
    try std.testing.expectEqual(@as(i64, 64 * 32), tiled.sizeStatic());
    try std.testing.expectEqual(@as(usize, 4), tiled.shapeStatic().rank());
}
