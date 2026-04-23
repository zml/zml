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
        const r = k.broadcastLike(rhs, self);
        return if (self.isFloatElem()) k.addf(self, r) else k.addi(self, r);
    }

    pub fn sub(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const r = k.broadcastLike(rhs, self);
        return if (self.isFloatElem()) k.subf(self, r) else k.subi(self, r);
    }

    pub fn mul(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const r = k.broadcastLike(rhs, self);
        return if (self.isFloatElem()) k.mulf(self, r) else k.muli(self, r);
    }

    pub fn div(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const r = k.broadcastLike(rhs, self);
        return if (self.isFloatElem()) k.divf(self, r) else k.divsi(self, r);
    }

    pub fn rem(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const r = k.broadcastLike(rhs, self);
        return if (self.isFloatElem()) k.remf(self, r) else k.remsi(self, r);
    }

    pub fn cdiv(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const r = k.broadcastLike(rhs, self);
        return k.ceildivsi(self, r);
    }

    pub fn bitAnd(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const r = k.broadcastLike(rhs, self);
        return k.andi(self, r);
    }

    pub fn bitOr(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const r = k.broadcastLike(rhs, self);
        return k.ori(self, r);
    }

    pub fn min(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const r = k.broadcastLike(rhs, self);
        return if (self.isFloatElem()) k.minimumf(self, r) else k.minsi(self, r);
    }

    pub fn max(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const r = k.broadcastLike(rhs, self);
        return if (self.isFloatElem()) k.maximumf(self, r) else k.maxsi(self, r);
    }

    // -------- fluent comparisons (int predicates default; float uses cmpf) --------

    pub fn lt(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const r = k.broadcastLike(rhs, self);
        return if (self.isFloatElem()) k.cmpf(.olt, self, r) else k.cmpi(.slt, self, r);
    }
    pub fn le(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const r = k.broadcastLike(rhs, self);
        return if (self.isFloatElem()) k.cmpf(.ole, self, r) else k.cmpi(.sle, self, r);
    }
    pub fn gt(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const r = k.broadcastLike(rhs, self);
        return if (self.isFloatElem()) k.cmpf(.ogt, self, r) else k.cmpi(.sgt, self, r);
    }
    pub fn ge(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const r = k.broadcastLike(rhs, self);
        return if (self.isFloatElem()) k.cmpf(.oge, self, r) else k.cmpi(.sge, self, r);
    }
    pub fn eq(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const r = k.broadcastLike(rhs, self);
        return if (self.isFloatElem()) k.cmpf(.oeq, self, r) else k.cmpi(.eq, self, r);
    }
    pub fn ne(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const r = k.broadcastLike(rhs, self);
        return if (self.isFloatElem()) k.cmpf(.one, self, r) else k.cmpi(.ne, self, r);
    }

    // -------- fluent casts (`to`) --------

    /// Cast to `dtype`, preserving shape. Auto-picks extsi/trunci/extf/truncf/
    /// sitofp/fptosi/fpToFp based on the current and target kinds.
    pub fn to(self: Value, dtype: DType) Value {
        const k = self.kern();
        const target_elem = dtype.toMlir(k.ctx);
        const cur_elem = self.elemType();
        if (cur_elem.eql(target_elem)) return self;

        const out_ty: *const mlir.Type = if (self.isTensor())
            mlir.rankedTensorType(self.shape().constSlice(), target_elem)
        else
            target_elem;

        const cur_is_float = self.isFloatElem();
        const tgt_is_float = isFloatDtype(dtype);

        if (cur_is_float and tgt_is_float) return k.fpToFp(self, out_ty, .{ .rounding = .rtne });
        if (cur_is_float and !tgt_is_float) return k.fptosi(self, out_ty);
        if (!cur_is_float and tgt_is_float) return k.sitofp(self, out_ty);
        // int → int: compare bit widths.
        const cur_bw = intElemBitwidth(k.ctx, cur_elem);
        const tgt_bw = dtypeBitwidth(dtype);
        if (tgt_bw > cur_bw) return k.extsi(self, out_ty);
        if (tgt_bw < cur_bw) return k.trunci(self, out_ty);
        return k.arithBitcast(self, out_ty);
    }

    // -------- shape manipulation --------

    /// Expand this 1-D vector to a 2-D `[m, n]` tensor by broadcasting along
    /// the axis opposite to `axis`. `axis=0` keeps the value as a column
    /// (replicates across rows); `axis=1` keeps it as a row. Matches Python's
    /// `vec[:, None]` (axis=1) / `vec[None, :]` (axis=0) idioms.
    pub fn broadcast2d(self: Value, axis: i32, m: i64, n: i64) Value {
        const k = self.kern();
        const expanded: [2]i64 = if (axis == 0) .{ 1, n } else .{ m, 1 };
        const exp = k.expandDims(self, axis, &expanded);
        const target: [2]i64 = .{ m, n };
        return k.broadcast(exp, &target);
    }

    /// Splat this scalar value to a tensor of the given shape.
    pub fn splatTo(self: Value, shape_: []const i64) Value {
        return self.kern().splat(self, shape_);
    }

    /// `addptr(self, offset)` where offset can be a Value or comptime int.
    pub fn addPtr(self: Value, offset: anytype) Value {
        const k = self.kern();
        const T = @TypeOf(offset);
        const off: Value = if (T == Value) offset else k.lift(offset);
        return k.addptr(self, off);
    }
};

fn isFloatDtype(dt: DType) bool {
    return switch (dt) {
        .f16, .bf16, .f32, .f64, .f8e4m3fn, .f8e5m2 => true,
        else => false,
    };
}

fn intElemBitwidth(ctx: *mlir.Context, ty: *const mlir.Type) usize {
    inline for (std.meta.fields(mlir.IntegerTypes)) |f| {
        const it = @field(mlir.IntegerTypes, f.name);
        if (ty.eql(mlir.integerType(ctx, it))) return it.bitwidth();
    }
    return 0;
}

fn dtypeBitwidth(dt: DType) usize {
    return switch (dt) {
        .i1 => 1,
        .i8 => 8,
        .i16 => 16,
        .i32 => 32,
        .i64 => 64,
        .f16, .bf16 => 16,
        .f32 => 32,
        .f64 => 64,
        .f8e4m3fn, .f8e5m2 => 8,
    };
}

/// Compute the result type of `reduce(src, axis)`: drop dim `axis` from `src_shape`.
/// When the source is 1-D, the result is a scalar (element type only).
fn computeReducedType(ctx: *mlir.Context, src_shape: []const i64, axis: i32, elem: *const mlir.Type) *const mlir.Type {
    if (src_shape.len <= 1) return elem;
    var out: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .{};
    for (src_shape, 0..) |d, i| {
        if (@as(i32, @intCast(i)) != axis) out.appendAssumeCapacity(d);
    }
    _ = ctx;
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

/// Key for the entry-block constant cache: DSL dtype + raw 64-bit payload
/// (int value as i64, float value as `@bitCast(f64)`). Every supported DType
/// fits in 64 bits of data, so this stays trivially `AutoHashMap`-friendly.
const ConstKey = struct {
    dtype: DType,
    bits: u64,

    fn fromInt(dtype: DType, value: i64) ConstKey {
        return .{ .dtype = dtype, .bits = @bitCast(value) };
    }

    fn fromFloat(dtype: DType, value: f64) ConstKey {
        return .{ .dtype = dtype, .bits = @bitCast(value) };
    }
};

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
    /// Deduplication cache for constants emitted at the entry block. Constants
    /// built inside nested regions (`forLoop` / `ifThenElse` bodies, reduce
    /// combiners, …) bypass the cache so their defining op still dominates
    /// their uses. Cleared on `deinit`.
    const_cache: std.AutoHashMapUnmanaged(ConstKey, Value) = .{},

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
        self.const_cache.deinit(self.allocator);
        self.module.deinit();
        self.arena.deinit();
    }

    /// True when the next op will be appended to the entry block (no active
    /// nested region). Constants built here dominate the whole function and
    /// can safely be shared via `const_cache`.
    fn atEntryBlock(self: *const Kernel) bool {
        return self.block_stack.len == 0;
    }

    fn cacheLookup(self: *Kernel, key: ConstKey) ?Value {
        if (!self.atEntryBlock()) return null;
        return self.const_cache.get(key);
    }

    fn cacheInsert(self: *Kernel, key: ConstKey, v: Value) void {
        if (!self.atEntryBlock()) return;
        self.const_cache.put(self.allocator, key, v) catch {};
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

    fn loc(self: *const Kernel) *const mlir.Location {
        return .unknown(self.ctx);
    }

    // ==================== type helpers ====================

    /// Scalar MLIR type for the given DSL element type.
    pub fn scalarTy(self: *const Kernel, dtype: DType) *const mlir.Type {
        return dtype.toMlir(self.ctx);
    }

    /// Ranked tensor MLIR type `tensor<shape x dtype>`.
    pub fn tensorTy(self: *const Kernel, shape: []const i64, dtype: DType) *const mlir.Type {
        return mlir.rankedTensorType(shape, dtype.toMlir(self.ctx));
    }

    // ==================== SPMD ====================

    pub fn programId(self: *Kernel, dim: ttir.ProgramIDDim) Value {
        return self.emit(ttir.get_program_id(self.ctx, dim, self.loc()));
    }

    pub fn numPrograms(self: *Kernel, dim: ttir.ProgramIDDim) Value {
        return self.emit(ttir.get_num_programs(self.ctx, dim, self.loc()));
    }

    // ==================== constants ====================

    pub fn constI32(self: *Kernel, value: i32) Value {
        return self.cachedInt(.i32, value);
    }

    pub fn constI64(self: *Kernel, value: i64) Value {
        return self.cachedInt(.i64, value);
    }

    pub fn constF32(self: *Kernel, value: f32) Value {
        return self.cachedFloat(.f32, @floatCast(value));
    }

    pub fn constF64(self: *Kernel, value: f64) Value {
        return self.cachedFloat(.f64, value);
    }

    fn cachedInt(self: *Kernel, dtype: DType, value: i64) Value {
        const key: ConstKey = .fromInt(dtype, value);
        if (self.cacheLookup(key)) |v| return v;
        const ty = dtype.toMlir(self.ctx);
        const v = self.emit(arith.constant_int(self.ctx, value, ty, self.loc()));
        self.cacheInsert(key, v);
        return v;
    }

    fn cachedFloat(self: *Kernel, dtype: DType, value: f64) Value {
        const key: ConstKey = .fromFloat(dtype, value);
        if (self.cacheLookup(key)) |v| return v;
        const v = switch (dtype) {
            .f32 => self.emit(arith.constant_float(self.ctx, value, .f32, self.loc())),
            .f64 => self.emit(arith.constant_float(self.ctx, value, .f64, self.loc())),
            else => blk: {
                const f32_scalar = self.cachedFloat(.f32, value);
                break :blk self.fpToFp(f32_scalar, dtype.toMlir(self.ctx), .{ .rounding = .rtne });
            },
        };
        self.cacheInsert(key, v);
        return v;
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
        const len: i64 = @intCast(end - start);
        const out_ty = self.tensorTy(&.{len}, dtype);
        return switch (dtype) {
            .i64 => self.extsi(r, out_ty),
            .i16, .i8 => self.trunci(r, out_ty),
            .f16, .bf16, .f32, .f64, .f8e4m3fn, .f8e5m2 => self.sitofp(r, out_ty),
            else => @panic("Kernel.arange: unsupported dtype"),
        };
    }

    /// `tt.splat` with polymorphic input: accepts a Value, a `comptime_int`, or
    /// a `comptime_float`. Comptime scalars are auto-lifted to a constant of
    /// the matching dtype (i32 for int, f32 for float) before splatting.
    pub fn splat(self: *Kernel, value: anytype, shape: []const i64) Value {
        const T = @TypeOf(value);
        const v: Value = if (T == Value) value else self.lift(value);
        const elem = v.type_();
        const ty = mlir.rankedTensorType(shape, elem);
        return self.emit(ttir.splat(self.ctx, v.inner, ty, self.loc()));
    }

    /// Lift a comptime int/float (or pass-through Value) to a scalar DSL Value.
    /// For runtime ints, the target DSL dtype follows the source's bitwidth
    /// (i8/i16/i32 → `i32`, i64 → `i64`). Comptime ints default to `i32`.
    /// Floats always lift to `f32`. To target a specific dtype, call
    /// `constMatching(value, dtype.toMlir(k.ctx))` instead.
    pub fn lift(self: *Kernel, value: anytype) Value {
        const T = @TypeOf(value);
        if (T == Value) return value;
        return switch (@typeInfo(T)) {
            .comptime_int => self.constI32(@intCast(value)),
            .int => |info| if (info.bits > 32) self.constI64(@intCast(value)) else self.constI32(@intCast(value)),
            .comptime_float, .float => self.constF32(@floatCast(value)),
            else => @compileError("Kernel.lift: unsupported type " ++ @typeName(T)),
        };
    }

    /// Lift `value` to a Value and — if `ref` is a tensor — splat it to match
    /// `ref`'s shape. For *comptime* scalars, the lifted constant's element
    /// type follows `ref`'s element type (so `v.mul(16)` matches whatever int/
    /// float kind `v` is). For runtime Zig scalars, the source width is
    /// preserved — pass `@as(i32, x)` vs `@as(i64, x)` to control the lift.
    pub fn broadcastLike(self: *Kernel, value: anytype, ref: Value) Value {
        const T = @TypeOf(value);
        const v: Value = if (T == Value) value else switch (@typeInfo(T)) {
            .comptime_int, .comptime_float => self.constMatching(value, ref.elemType()),
            else => self.lift(value),
        };
        if (!ref.isTensor()) return v;
        if (v.isTensor()) return v;
        return self.splat(v, ref.shape().constSlice());
    }

    /// Build an arith.constant of the given MLIR element type from a Zig numeric.
    /// Picks between integer/float codegen by inspecting the target type, and
    /// re-uses a cached constant at the entry block when possible.
    pub fn constMatching(self: *Kernel, value: anytype, elem: *const mlir.Type) Value {
        const T = @TypeOf(value);
        // Target is integer? find matching DSL DType and delegate to cachedInt.
        inline for (std.meta.fields(DType)) |f| {
            const dt = @field(DType, f.name);
            if (!isFloatDtype(dt)) {
                if (elem.eql(dt.toMlir(self.ctx))) {
                    const v64: i64 = switch (@typeInfo(T)) {
                        .comptime_int, .int => @intCast(value),
                        .comptime_float, .float => @intFromFloat(value),
                        else => @compileError("constMatching: unsupported scalar " ++ @typeName(T)),
                    };
                    return self.cachedInt(dt, v64);
                }
            }
        }
        // Float target.
        const v_f64: f64 = switch (@typeInfo(T)) {
            .comptime_int, .int => @floatFromInt(value),
            .comptime_float, .float => @floatCast(value),
            else => @compileError("constMatching: unsupported scalar " ++ @typeName(T)),
        };
        inline for (std.meta.fields(DType)) |f| {
            const dt = @field(DType, f.name);
            if (isFloatDtype(dt)) {
                if (elem.eql(dt.toMlir(self.ctx))) return self.cachedFloat(dt, v_f64);
            }
        }
        @panic("constMatching: element type not a recognized DSL DType");
    }

    /// Zero-valued tensor of the given shape and dtype.
    pub fn zeros(self: *Kernel, shape: []const i64, dtype: DType) Value {
        const scalar = if (isFloatDtype(dtype)) self.cachedFloat(dtype, 0.0) else self.cachedInt(dtype, 0);
        return self.splat(scalar, shape);
    }

    /// One-valued tensor of the given shape and dtype.
    pub fn ones(self: *Kernel, shape: []const i64, dtype: DType) Value {
        const scalar = if (isFloatDtype(dtype)) self.cachedFloat(dtype, 1.0) else self.cachedInt(dtype, 1);
        return self.splat(scalar, shape);
    }

    /// `tt.load` of a single scalar from a `!tt.ptr<T>`. Convenience wrapper
    /// around `load(ptr, scalarTy(dt), .{})`.
    pub fn loadScalar(self: *Kernel, ptr: Value, dtype: DType) Value {
        return self.load(ptr, self.scalarTy(dtype), .{});
    }

    /// `tt.load` of a tensor, with a mask and an implicit zero "other". The
    /// "other" is auto-created in the right dtype and splat to `shape`.
    /// Matches Python `tl.load(ptr, mask=mask, other=0)`.
    pub fn loadMasked(self: *Kernel, ptr: Value, shape: []const i64, dtype: DType, mask: Value) Value {
        const ty = self.tensorTy(shape, dtype);
        const zero = self.zeros(shape, dtype);
        return self.load(ptr, ty, .{ .mask = mask.inner, .other = zero.inner });
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

    /// `reduce` with an internal adder — `tl.sum(src, axis)`.
    pub fn reduceSum(self: *Kernel, src: Value, axis: i32) Value {
        const elem = src.elemType();
        const shape = src.shape();
        const result_ty: *const mlir.Type = computeReducedType(self.ctx, shape.constSlice(), axis, elem);
        const is_float = src.isFloatElem();
        const Combiner = struct {
            is_float: bool,
            fn combine(kk: *Kernel, lhs: Value, rhs: Value, ctx: anytype) Value {
                return if (ctx.is_float) kk.addf(lhs, rhs) else kk.addi(lhs, rhs);
            }
        };
        return self.reduce(src, axis, elem, result_ty, Combiner.combine, Combiner{ .is_float = is_float });
    }

    /// `reduce` with an internal maximum — `tl.max(src, axis)`.
    pub fn reduceMax(self: *Kernel, src: Value, axis: i32) Value {
        const elem = src.elemType();
        const shape = src.shape();
        const result_ty: *const mlir.Type = computeReducedType(self.ctx, shape.constSlice(), axis, elem);
        const is_float = src.isFloatElem();
        const Combiner = struct {
            is_float: bool,
            fn combine(kk: *Kernel, lhs: Value, rhs: Value, ctx: anytype) Value {
                return if (ctx.is_float) kk.maximumf(lhs, rhs) else kk.maxsi(lhs, rhs);
            }
        };
        return self.reduce(src, axis, elem, result_ty, Combiner.combine, Combiner{ .is_float = is_float });
    }

    /// `scan` with addition — `tl.cumsum(src, axis)`.
    pub fn scanSum(self: *Kernel, src: Value, axis: i32, reverse: bool) Value {
        const elem = src.elemType();
        const is_float = src.isFloatElem();
        const Combiner = struct {
            is_float: bool,
            fn combine(kk: *Kernel, lhs: Value, rhs: Value, ctx: anytype) Value {
                return if (ctx.is_float) kk.addf(lhs, rhs) else kk.addi(lhs, rhs);
            }
        };
        return self.scan(src, axis, reverse, elem, src.type_(), Combiner.combine, Combiner{ .is_float = is_float });
    }

    /// `tt.expand_dims` — insert a size-1 axis at `axis`.
    pub fn expandDims(self: *Kernel, value: Value, axis: i32, result_shape: []const i64) Value {
        const elem = mlir.RankedTensorType.fromShaped(value.inner.type_().isA(mlir.ShapedType).?).elementType();
        const ty = mlir.rankedTensorType(result_shape, elem);
        return self.emit(ttir.expand_dims(self.ctx, value.inner, axis, ty, self.loc()));
    }

    /// `tt.broadcast` — broadcast size-1 dims up to the target shape. Element
    /// type is taken from the input value.
    pub fn broadcast(self: *Kernel, value: Value, result_shape: []const i64) Value {
        const elem = mlir.RankedTensorType.fromShaped(value.inner.type_().isA(mlir.ShapedType).?).elementType();
        const ty = mlir.rankedTensorType(result_shape, elem);
        return self.emit(ttir.broadcast(self.ctx, value.inner, ty, self.loc()));
    }

    // ==================== pointer arithmetic ====================

    pub fn addptr(self: *Kernel, ptr: Value, offset: Value) Value {
        return self.emit(ttir.addptr(self.ctx, ptr.inner, offset.inner, self.loc()));
    }

    /// `tt.load` from a scalar (or tensor of) pointer. `result_type` specifies
    /// what's being loaded (for a scalar ptr this is the pointee type; for a
    /// tensor of ptrs this is the element tensor type).
    pub fn load(self: *Kernel, ptr: Value, result_type: *const mlir.Type, opts: ttir.LoadOpts) Value {
        return self.emit(ttir.load(self.ctx, ptr.inner, result_type, opts, self.loc()));
    }

    /// `tt.store` — no result.
    pub fn store(self: *Kernel, ptr: Value, value: Value, opts: ttir.StoreOpts) void {
        _ = ttir.store(self.ctx, ptr.inner, value.inner, opts, self.loc()).appendTo(self.currentBlock());
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

    pub fn select(self: *Kernel, cond: Value, t: Value, f: Value) Value {
        return self.emit(arith.select(self.ctx, cond.inner, t.inner, f.inner, self.loc()));
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

    // Casts
    pub fn extsi(self: *Kernel, src: Value, result_type: *const mlir.Type) Value {
        return self.emit(arith.extsi(self.ctx, src.inner, result_type, self.loc()));
    }
    pub fn extui(self: *Kernel, src: Value, result_type: *const mlir.Type) Value {
        return self.emit(arith.extui(self.ctx, src.inner, result_type, self.loc()));
    }
    pub fn extf(self: *Kernel, src: Value, result_type: *const mlir.Type) Value {
        return self.emit(arith.extf(self.ctx, src.inner, result_type, self.loc()));
    }
    pub fn trunci(self: *Kernel, src: Value, result_type: *const mlir.Type) Value {
        return self.emit(arith.trunci(self.ctx, src.inner, result_type, self.loc()));
    }
    pub fn truncf(self: *Kernel, src: Value, result_type: *const mlir.Type) Value {
        return self.emit(arith.truncf(self.ctx, src.inner, result_type, self.loc()));
    }
    pub fn sitofp(self: *Kernel, src: Value, result_type: *const mlir.Type) Value {
        return self.emit(arith.sitofp(self.ctx, src.inner, result_type, self.loc()));
    }
    pub fn uitofp(self: *Kernel, src: Value, result_type: *const mlir.Type) Value {
        return self.emit(arith.uitofp(self.ctx, src.inner, result_type, self.loc()));
    }
    pub fn fptosi(self: *Kernel, src: Value, result_type: *const mlir.Type) Value {
        return self.emit(arith.fptosi(self.ctx, src.inner, result_type, self.loc()));
    }
    pub fn fptoui(self: *Kernel, src: Value, result_type: *const mlir.Type) Value {
        return self.emit(arith.fptoui(self.ctx, src.inner, result_type, self.loc()));
    }
    pub fn arithBitcast(self: *Kernel, src: Value, result_type: *const mlir.Type) Value {
        return self.emit(arith.bitcast(self.ctx, src.inner, result_type, self.loc()));
    }

    // ==================== Triton-specific ops ====================

    /// `tt.bitcast` — like arith.bitcast but works on ptrs too.
    pub fn bitcast(self: *Kernel, src: Value, result_type: *const mlir.Type) Value {
        return self.emit(ttir.bitcast(self.ctx, src.inner, result_type, self.loc()));
    }
    pub fn intToPtr(self: *Kernel, src: Value, result_type: *const mlir.Type) Value {
        return self.emit(ttir.int_to_ptr(self.ctx, src.inner, result_type, self.loc()));
    }
    pub fn ptrToInt(self: *Kernel, src: Value, result_type: *const mlir.Type) Value {
        return self.emit(ttir.ptr_to_int(self.ctx, src.inner, result_type, self.loc()));
    }
    pub fn fpToFp(self: *Kernel, src: Value, result_type: *const mlir.Type, opts: ttir.FpToFpOpts) Value {
        return self.emit(ttir.fp_to_fp(self.ctx, src.inner, result_type, opts, self.loc()));
    }
    pub fn clampf(self: *Kernel, x: Value, min: Value, max: Value, propagate_nan: ttir.PropagateNan) Value {
        return self.emit(ttir.clampf(self.ctx, x.inner, min.inner, max.inner, propagate_nan, self.loc()));
    }
    pub fn preciseSqrt(self: *Kernel, x: Value) Value {
        return self.emit(ttir.precise_sqrt(self.ctx, x.inner, self.loc()));
    }
    pub fn preciseDivf(self: *Kernel, x: Value, y: Value) Value {
        return self.emit(ttir.precise_divf(self.ctx, x.inner, y.inner, self.loc()));
    }
    pub fn mulhiui(self: *Kernel, x: Value, y: Value) Value {
        return self.emit(ttir.mulhiui(self.ctx, x.inner, y.inner, self.loc()));
    }

    /// `tt.dot` — matmul with accumulator. `a`, `b`, `c_acc` are tensors.
    pub fn dot(self: *Kernel, a: Value, b: Value, c_acc: Value, result_type: *const mlir.Type, opts: ttir.DotOpts) Value {
        return self.emit(ttir.dot(self.ctx, a.inner, b.inner, c_acc.inner, result_type, opts, self.loc()));
    }

    /// `tt.reshape` with explicit result type and options.
    pub fn reshape(self: *Kernel, src: Value, result_type: *const mlir.Type, opts: ttir.ReshapeOpts) Value {
        return self.emit(ttir.reshape(self.ctx, src.inner, result_type, opts, self.loc()));
    }

    /// `tt.trans` — permute dimensions according to `order`. Result type must
    /// match the permuted shape.
    pub fn trans(self: *Kernel, src: Value, order: []const i32, result_type: *const mlir.Type) Value {
        return self.emit(ttir.trans(self.ctx, src.inner, order, result_type, self.loc()));
    }

    /// `tt.cat` — concatenate two tensors; may reorder elements.
    pub fn cat(self: *Kernel, lhs: Value, rhs: Value, result_type: *const mlir.Type) Value {
        return self.emit(ttir.cat(self.ctx, lhs.inner, rhs.inner, result_type, self.loc()));
    }

    /// `tt.join` — join two equal-shape tensors along a new minor dim.
    pub fn join(self: *Kernel, lhs: Value, rhs: Value, result_type: *const mlir.Type) Value {
        return self.emit(ttir.join(self.ctx, lhs.inner, rhs.inner, result_type, self.loc()));
    }

    /// `tt.split` — split a tensor whose last dim is 2 into two equal halves.
    /// Returns (lhs, rhs).
    pub fn split(self: *Kernel, src: Value, result_type: *const mlir.Type) [2]Value {
        const op = ttir.split(self.ctx, src.inner, result_type, self.loc());
        _ = op.appendTo(self.currentBlock());
        return .{
            .{ .inner = op.result(0), .kernel = self },
            .{ .inner = op.result(1), .kernel = self },
        };
    }

    /// `tt.unsplat` — 1-element tensor → scalar.
    pub fn unsplat(self: *Kernel, src: Value, result_type: *const mlir.Type) Value {
        return self.emit(ttir.unsplat(self.ctx, src.inner, result_type, self.loc()));
    }

    /// `tt.gather` — `src[indices]` along `axis`; output shape matches `indices`.
    pub fn gather(self: *Kernel, src: Value, indices: Value, axis: i32, result_type: *const mlir.Type, opts: ttir.GatherOpts) Value {
        return self.emit(ttir.gather(self.ctx, src.inner, indices.inner, axis, result_type, opts, self.loc()));
    }

    /// `tt.histogram` — returns a tensor whose shape defines the number of bins.
    pub fn histogram(self: *Kernel, src: Value, mask: ?Value, result_type: *const mlir.Type) Value {
        const mask_inner: ?*const mlir.Value = if (mask) |m| m.inner else null;
        return self.emit(ttir.histogram(self.ctx, src.inner, mask_inner, result_type, self.loc()));
    }

    /// `tt.assert` — device-side assert on a condition (i1 or tensor<...xi1>).
    pub fn assert_(self: *Kernel, condition: Value, message: []const u8) void {
        _ = ttir.assert_(self.ctx, condition.inner, message, self.loc()).appendTo(self.currentBlock());
    }

    /// `tt.atomic_rmw` — returns the old value at ptr before the RMW.
    pub fn atomicRmw(self: *Kernel, rmw: ttir.RMWOp, ptr: Value, val: Value, opts_arg: ttir.AtomicRMWOpts) Value {
        return self.emit(ttir.atomic_rmw(self.ctx, rmw, ptr.inner, val.inner, opts_arg, self.loc()));
    }

    /// `tt.atomic_cas` — compare-and-swap; returns the old value at ptr.
    pub fn atomicCas(self: *Kernel, ptr: Value, cmp: Value, val: Value, opts_arg: ttir.AtomicCasOpts) Value {
        return self.emit(ttir.atomic_cas(self.ctx, ptr.inner, cmp.inner, val.inner, opts_arg, self.loc()));
    }

    /// `tt.call` — call another tt.func in this module by symbol name.
    pub fn call(self: *Kernel, callee: []const u8, operands: []const Value, result_types: []const *const mlir.Type) []Value {
        var buf: stdx.BoundedArray(*const mlir.Value, 32) = .{};
        for (operands) |v| buf.appendAssumeCapacity(v.inner);
        const op = ttir.call(self.ctx, callee, buf.constSlice(), result_types, .{}, self.loc());
        return self.emitMulti(op, result_types.len);
    }

    /// `tt.dot_scaled` — dot with microscaling factors.
    pub fn dotScaled(
        self: *Kernel,
        a: Value,
        b: Value,
        c_acc: Value,
        a_scale: ?Value,
        b_scale: ?Value,
        a_elem_type: ttir.ScaleDotElemType,
        b_elem_type: ttir.ScaleDotElemType,
        result_type: *const mlir.Type,
        opts: ttir.DotScaledOpts,
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
            result_type,
            opts,
            self.loc(),
        ));
    }

    /// `tt.extern_elementwise` — call a library symbol pointwise.
    pub fn externElementwise(
        self: *Kernel,
        srcs: []const Value,
        result_type: *const mlir.Type,
        libname: []const u8,
        libpath: []const u8,
        symbol: []const u8,
        opts: ttir.ExternElementwiseOpts,
    ) Value {
        var buf: stdx.BoundedArray(*const mlir.Value, 8) = .{};
        for (srcs) |s| buf.appendAssumeCapacity(s.inner);
        return self.emit(ttir.extern_elementwise(
            self.ctx,
            buf.constSlice(),
            result_type,
            libname,
            libpath,
            symbol,
            opts,
            self.loc(),
        ));
    }

    /// `tt.print` — device-side print.
    pub fn print(
        self: *Kernel,
        prefix: []const u8,
        hex: bool,
        args: []const Value,
        is_signed: []const i32,
    ) void {
        var buf: stdx.BoundedArray(*const mlir.Value, 16) = .{};
        for (args) |a| buf.appendAssumeCapacity(a.inner);
        _ = ttir.print(self.ctx, prefix, hex, buf.constSlice(), is_signed, self.loc()).appendTo(self.currentBlock());
    }

    /// `tt.make_tensor_descriptor` — build a TMA descriptor.
    pub fn makeTensorDescriptor(
        self: *Kernel,
        base: Value,
        shape: []const Value,
        strides: []const Value,
        padding: ttir.PaddingOption,
        result_type: *const mlir.Type,
    ) Value {
        var shape_buf: stdx.BoundedArray(*const mlir.Value, 16) = .{};
        for (shape) |s| shape_buf.appendAssumeCapacity(s.inner);
        var strides_buf: stdx.BoundedArray(*const mlir.Value, 16) = .{};
        for (strides) |s| strides_buf.appendAssumeCapacity(s.inner);
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

    /// `tt.descriptor_load` — TMA load.
    pub fn descriptorLoad(
        self: *Kernel,
        desc: Value,
        indices: []const Value,
        result_type: *const mlir.Type,
        opts: ttir.DescriptorLoadOpts,
    ) Value {
        var idx_buf: stdx.BoundedArray(*const mlir.Value, 8) = .{};
        for (indices) |i| idx_buf.appendAssumeCapacity(i.inner);
        return self.emit(ttir.descriptor_load(self.ctx, desc.inner, idx_buf.constSlice(), result_type, opts, self.loc()));
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
    pub fn descriptorGather(
        self: *Kernel,
        desc: Value,
        x_offsets: Value,
        y_offset: Value,
        result_type: *const mlir.Type,
    ) Value {
        return self.emit(ttir.descriptor_gather(self.ctx, desc.inner, x_offsets.inner, y_offset.inner, result_type, self.loc()));
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
    pub fn mathClampf(self: *Kernel, value: Value, min: Value, max: Value) Value {
        return self.emit(math.clampf(self.ctx, value.inner, min.inner, max.inner, self.loc()));
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

    /// arith.convertf — same-bitwidth float cast (e.g. f16 ↔ bf16).
    pub fn convertf(self: *Kernel, src: Value, result_type: *const mlir.Type, opts: arith.ConvertFOpts) Value {
        return self.emit(arith.convertf(self.ctx, src.inner, result_type, opts, self.loc()));
    }
    /// arith.scaling_truncf — MXFP downcast with per-block scales.
    pub fn scalingTruncf(self: *Kernel, src: Value, scale: Value, result_type: *const mlir.Type, opts: arith.ConvertFOpts) Value {
        return self.emit(arith.scaling_truncf(self.ctx, src.inner, scale.inner, result_type, opts, self.loc()));
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

    /// `tt.elementwise_inline_asm` — inline asm pointwise op.
    pub fn elementwiseInlineAsm(
        self: *Kernel,
        asm_string: []const u8,
        constraints: []const u8,
        args: []const Value,
        result_types: []const *const mlir.Type,
        opts: ttir.InlineAsmOpts,
    ) []Value {
        var buf: stdx.BoundedArray(*const mlir.Value, 16) = .{};
        for (args) |a| buf.appendAssumeCapacity(a.inner);
        const op = ttir.elementwise_inline_asm(self.ctx, asm_string, constraints, buf.constSlice(), result_types, opts, self.loc());
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

    /// `tt.reduce` along `axis`, single-input. `combine_fn(k, lhs, rhs, body_ctx)`
    /// returns the value fed to `tt.reduce.return`. `elem_ty` is the element
    /// type of `src` (used for the combine region args); `result_ty` is the
    /// reduce op's result type (scalar or tensor-minus-axis).
    pub fn reduce(
        self: *Kernel,
        src: Value,
        axis: i32,
        elem_ty: *const mlir.Type,
        result_ty: *const mlir.Type,
        comptime combine_fn: anytype,
        body_ctx: anytype,
    ) Value {
        const region = mlir.Block.init(&.{ elem_ty, elem_ty }, &.{ self.loc(), self.loc() });

        self.pushBlock(region);
        const lhs: Value = .{ .inner = region.argument(0), .kernel = self };
        const rhs: Value = .{ .inner = region.argument(1), .kernel = self };
        const combined: Value = combine_fn(self, lhs, rhs, body_ctx);
        _ = ttir.reduce_return(self.ctx, &.{combined.inner}, self.loc()).appendTo(region);
        self.popBlock();

        const op = ttir.reduce(self.ctx, &.{src.inner}, axis, region, &.{result_ty}, self.loc());
        _ = op.appendTo(self.currentBlock());
        return .{ .inner = op.result(0), .kernel = self };
    }

    /// `tt.scan` along `axis`, single-input. `combine_fn(k, lhs, rhs, body_ctx)`
    /// returns the value fed to `tt.scan.return`.
    pub fn scan(
        self: *Kernel,
        src: Value,
        axis: i32,
        reverse: bool,
        elem_ty: *const mlir.Type,
        result_ty: *const mlir.Type,
        comptime combine_fn: anytype,
        body_ctx: anytype,
    ) Value {
        const region = mlir.Block.init(&.{ elem_ty, elem_ty }, &.{ self.loc(), self.loc() });

        self.pushBlock(region);
        const lhs: Value = .{ .inner = region.argument(0), .kernel = self };
        const rhs: Value = .{ .inner = region.argument(1), .kernel = self };
        const combined: Value = combine_fn(self, lhs, rhs, body_ctx);
        _ = ttir.scan_return(self.ctx, &.{combined.inner}, self.loc()).appendTo(region);
        self.popBlock();

        const op = ttir.scan(self.ctx, &.{src.inner}, axis, reverse, region, &.{result_ty}, self.loc());
        _ = op.appendTo(self.currentBlock());
        return .{ .inner = op.result(0), .kernel = self };
    }

    // ==================== SCF regions ====================

    /// `scf.for` with iter_args. `body_fn` receives:
    ///   fn(kernel: *Kernel, iv: Value, iter_vals: []const Value, ctx: @TypeOf(body_ctx)) []const Value
    /// and must return the values to yield. The returned slice's lifetime only
    /// needs to cover the call (they're copied into scf.yield's operand list).
    ///
    /// `lower`, `upper`, and `step` can be existing `Value`s or any Zig
    /// numeric. Comptime ints adopt `lower`'s element type (so
    /// `forLoop(0, N, 1, …)` with `N: Value` of type i32 lifts the literals to
    /// i32); runtime ints lift per their Zig width (see `Kernel.lift`).
    pub fn forLoop(
        self: *Kernel,
        lower: anytype,
        upper: anytype,
        step: anytype,
        iter_args: []const Value,
        comptime body_fn: anytype,
        body_ctx: anytype,
    ) []Value {
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

        const yielded: []const Value = body_fn(self, iv, barg_buf.constSlice(), body_ctx);

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

    /// `scf.if` — build a conditional with optional results.
    /// Both `then_fn` and `else_fn` are called to populate the respective
    /// blocks; each must return the values to yield (matching `result_types`).
    /// Pass `&.{}` for `result_types` to build a no-result if/else.
    pub fn ifThenElse(
        self: *Kernel,
        cond: Value,
        result_types: []const *const mlir.Type,
        comptime then_fn: anytype,
        comptime else_fn: anytype,
        body_ctx: anytype,
    ) []Value {
        const then_block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(then_block);
        const then_vals: []const Value = then_fn(self, body_ctx);
        var then_yield: stdx.BoundedArray(*const mlir.Value, 32) = .{};
        for (then_vals) |v| then_yield.appendAssumeCapacity(v.inner);
        _ = scf.yield(self.ctx, then_yield.constSlice(), self.loc()).appendTo(then_block);
        self.popBlock();

        const else_block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(else_block);
        const else_vals: []const Value = else_fn(self, body_ctx);
        var else_yield: stdx.BoundedArray(*const mlir.Value, 32) = .{};
        for (else_vals) |v| else_yield.appendAssumeCapacity(v.inner);
        _ = scf.yield(self.ctx, else_yield.constSlice(), self.loc()).appendTo(else_block);
        self.popBlock();

        const if_op = scf.if_(
            self.ctx,
            cond.inner,
            result_types,
            then_block,
            else_block,
            self.loc(),
        );
        _ = if_op.appendTo(self.currentBlock());

        const out = self.arena.allocator().alloc(Value, result_types.len) catch @panic("Kernel.ifThenElse OOM");
        for (0..result_types.len) |i| out[i] = .{ .inner = if_op.result(i), .kernel = self };
        return out;
    }

    /// `scf.while` — generic while loop with separate "before" (cond) and
    /// "after" (body) regions.
    ///   - `before_fn(k, args: []const Value, body_ctx) struct { cond: Value, forwarded: []const Value }`
    ///   - `after_fn(k, args: []const Value, body_ctx) []const Value` — values to feed back to "before"
    /// `arg_types_before` gives the types of the "before" region arguments
    /// (must match `inits`); `arg_types_after` gives the "after" region's
    /// arguments (which are the trailing operands of scf.condition).
    /// The op's results match `arg_types_after`.
    pub fn whileLoop(
        self: *Kernel,
        inits: []const Value,
        arg_types_after: []const *const mlir.Type,
        comptime before_fn: anytype,
        comptime after_fn: anytype,
        body_ctx: anytype,
    ) []Value {
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
        const before_out = before_fn(self, before_args.constSlice(), body_ctx);
        // `before_out` is expected to be a struct { cond: Value, forwarded: []const Value }
        var cond_args: stdx.BoundedArray(*const mlir.Value, 32) = .{};
        for (before_out.forwarded) |v| cond_args.appendAssumeCapacity(v.inner);
        _ = scf.condition(self.ctx, before_out.cond.inner, cond_args.constSlice(), self.loc()).appendTo(before_block);
        self.popBlock();

        // Populate "after".
        self.pushBlock(after_block);
        var after_args: stdx.BoundedArray(Value, 32) = .{};
        for (0..arg_types_after.len) |i| after_args.appendAssumeCapacity(.{ .inner = after_block.argument(i), .kernel = self });
        const after_out = after_fn(self, after_args.constSlice(), body_ctx);
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
    /// serialize to a NUL-terminated TTIR string owned by `allocator`.
    pub fn finish(
        self: *Kernel,
        results: []const Value,
        allocator: std.mem.Allocator,
    ) FinishError![:0]const u8 {
        var return_values: stdx.BoundedArray(*const mlir.Value, 16) = .{};
        for (results) |r| return_values.appendAssumeCapacity(r.inner);
        _ = ttir.return_(self.ctx, return_values.constSlice(), self.loc()).appendTo(self.entry_block);

        if (!self.module.operation().verify()) {
            return error.InvalidMlir;
        }

        var al: std.Io.Writer.Allocating = .init(allocator);
        defer al.deinit();
        try al.writer.print("{f}", .{self.module.operation()});

        return try allocator.dupeZ(u8, al.written());
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

    const loaded = kernel.load(a_ptr, mlir.floatType(ctx, .f32), .{});
    const one = kernel.constF32(1.0);
    const summed = kernel.addf(loaded, one);
    kernel.store(b_ptr, summed, .{});

    const ir = try kernel.finish(&.{}, std.testing.allocator);
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
    const lb = kernel.constI32(0);
    const ub = kernel.constI32(8);
    const step = kernel.constI32(1);
    const init_acc = kernel.constF32(0.0);

    const Ctx = struct { base: Value };
    const body = struct {
        fn call(k: *Kernel, iv: Value, iter: []const Value, c: Ctx) []const Value {
            _ = iv;
            const v = k.load(c.base, mlir.floatType(k.ctx, .f32), .{});
            const next = k.addf(iter[0], v);
            const out = k.arena.allocator().alloc(Value, 1) catch unreachable;
            out[0] = next;
            return out;
        }
    }.call;

    const loop_results = kernel.forLoop(lb, ub, step, &.{init_acc}, body, Ctx{ .base = base });
    kernel.store(base, loop_results[0], .{});

    const ir = try kernel.finish(&.{}, std.testing.allocator);
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
    const c1 = kernel.constI32(1);
    const c2 = kernel.constI32(2);
    const eq = kernel.cmpi(.eq, c1, c2);

    const then_fn = struct {
        fn call(k: *Kernel, _: void) []const Value {
            const one = k.constF32(1.0);
            const out = k.arena.allocator().alloc(Value, 1) catch unreachable;
            out[0] = one;
            return out;
        }
    }.call;
    const else_fn = struct {
        fn call(k: *Kernel, _: void) []const Value {
            const two = k.constF32(2.0);
            const out = k.arena.allocator().alloc(Value, 1) catch unreachable;
            out[0] = two;
            return out;
        }
    }.call;

    const f32_ty = mlir.floatType(ctx, .f32);
    const results = kernel.ifThenElse(eq, &.{f32_ty}, then_fn, else_fn, {});
    kernel.store(a_ptr, results[0], .{});

    const ir = try kernel.finish(&.{}, std.testing.allocator);
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

    const i64_ty = mlir.integerType(ctx, .i64);
    const f32_ty = mlir.floatType(ctx, .f32);
    const f16_ty = mlir.floatType(ctx, .f16);

    // Round-trip: i64 → ptr<f32> → int → ptr<f32>.
    const raw = kernel.load(iptr, i64_ty, .{});
    const f_ptr_ty = ttir.pointerType(f32_ty, 1);
    const cast_ptr = kernel.intToPtr(raw, f_ptr_ty);
    const back_int = kernel.ptrToInt(cast_ptr, i64_ty);
    _ = back_int;

    // tt.bitcast f32 → i32 scalar.
    const fval = kernel.load(fptr, f32_ty, .{});
    const bits = kernel.bitcast(fval, mlir.integerType(ctx, .i32));
    _ = bits;

    // fp_to_fp f32 → f16 with rounding.
    const narrowed = kernel.fpToFp(fval, f16_ty, .{ .rounding = .rtne });
    kernel.store(half_ptr, narrowed, .{});

    const ir = try kernel.finish(&.{}, std.testing.allocator);
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

    const f32_ty = mlir.floatType(ctx, .f32);
    const x = kernel.load(x_ptr, f32_ty, .{});

    const e = kernel.exp2(x);
    const sq = kernel.sqrt(e);
    const rq = kernel.rsqrt(sq);
    const t = kernel.tanh(rq);
    const e2 = kernel.erf(t);

    kernel.store(out_ptr, e2, .{});

    const ir = try kernel.finish(&.{}, std.testing.allocator);
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
    const one = kernel.constI32(1);
    _ = kernel.atomicRmw(.add, counter, one, .{});

    const ir = try kernel.finish(&.{}, std.testing.allocator);
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

    const f32_ty = mlir.floatType(ctx, .f32);
    const f16_ty = mlir.floatType(ctx, .f16);
    const i64_ty = mlir.integerType(ctx, .i64);
    const ptr_f32 = ttir.pointerType(f32_ty, 1);

    // scalar ↔ scalar
    _ = k.intToPtr(si, ptr_f32);
    _ = k.ptrToInt(sp, i64_ty);
    _ = k.truncf(sf, f16_ty);

    // 0D-tensor ↔ 0D-tensor
    const t_ptr_0d = k.splat(sp, &.{});
    const t_f32_0d = k.splat(sf, &.{});
    const t_i64_0d = k.splat(si, &.{});
    const ptr_0d_ty = mlir.rankedTensorType(&.{}, ptr_f32);
    const f32_0d_ty = mlir.rankedTensorType(&.{}, f32_ty);
    const f16_0d_ty = mlir.rankedTensorType(&.{}, f16_ty);
    const i64_0d_ty = mlir.rankedTensorType(&.{}, i64_ty);
    _ = k.intToPtr(t_i64_0d, ptr_0d_ty);
    _ = k.ptrToInt(t_ptr_0d, i64_0d_ty);
    _ = k.truncf(t_f32_0d, f16_0d_ty);
    _ = f32_0d_ty;

    // 1D-tensor ↔ 1D-tensor
    const t_ptr_1d = k.splat(sp, &.{16});
    const t_f32_1d = k.splat(sf, &.{16});
    const t_i64_1d = k.splat(si, &.{16});
    const ptr_1d_ty = mlir.rankedTensorType(&.{16}, ptr_f32);
    const f32_1d_ty = mlir.rankedTensorType(&.{16}, f32_ty);
    const f16_1d_ty = mlir.rankedTensorType(&.{16}, f16_ty);
    const i64_1d_ty = mlir.rankedTensorType(&.{16}, i64_ty);
    _ = k.intToPtr(t_i64_1d, ptr_1d_ty);
    _ = k.ptrToInt(t_ptr_1d, i64_1d_ty);
    _ = k.truncf(t_f32_1d, f16_1d_ty);
    _ = f32_1d_ty;

    const __ir = try k.finish(&.{}, std.testing.allocator); std.testing.allocator.free(__ir);
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

    const __ir = try k.finish(&.{}, std.testing.allocator); std.testing.allocator.free(__ir);
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
    const f32_ty = mlir.floatType(ctx, .f32);

    const other = k.constF32(0.0);

    const a = k.load(ptr, f32_ty, .{});
    const b = k.load(ptr, f32_ty, .{ .mask = mask.inner });
    const c = k.load(ptr, f32_ty, .{ .mask = mask.inner, .other = other.inner });

    k.store(ptr, a, .{});
    k.store(ptr, b, .{ .mask = mask.inner });
    k.store(ptr, c, .{ .mask = mask.inner });

    const __ir = try k.finish(&.{}, std.testing.allocator); std.testing.allocator.free(__ir);
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

    const __ir = try k.finish(&.{}, std.testing.allocator); std.testing.allocator.free(__ir);
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
    const f32_ty = mlir.floatType(ctx, .f32);

    const v128x32 = k.splat(v, &.{ 128, 32 });
    const v32x128 = k.splat(v, &.{ 32, 128 });
    const zero128 = k.constF32(0.0);
    const z128x128 = k.splat(zero128, &.{ 128, 128 });
    const z32x32 = k.splat(zero128, &.{ 32, 32 });

    _ = k.dot(v128x32, v32x128, z128x128, mlir.rankedTensorType(&.{ 128, 128 }, f32_ty), .{});
    _ = k.dot(v32x128, v128x32, z32x32, mlir.rankedTensorType(&.{ 32, 32 }, f32_ty), .{});

    const __ir = try k.finish(&.{}, std.testing.allocator); std.testing.allocator.free(__ir);
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

    const __ir = try k.finish(&.{}, std.testing.allocator); std.testing.allocator.free(__ir);
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
    _ = kt.elementwiseInlineAsm("shl.b32 $0, $0, 3;", "=r,r", &.{x}, &.{i8_1d}, .{ .pure = true, .packed_element = 4 });
    const __ir_kt = try kt.finish(&.{}, std.testing.allocator); std.testing.allocator.free(__ir_kt);
    try parityPrintAndVerify(ctx, kt.module);

    // scalar form: i32 -> i32, packed_element=1
    var ks = try Kernel.init(std.testing.allocator, ctx, "inline_asm_scalar", &.{
        .{ .name = "x", .kind = .{ .scalar = .i32 } },
    }, &.{});
    defer ks.deinit();
    const xs = ks.arg(0);
    const i32_ty = mlir.integerType(ctx, .i32);
    _ = ks.elementwiseInlineAsm("shl.b32 $0, $0, 3;", "=r,r", &.{xs}, &.{i32_ty}, .{ .pure = true, .packed_element = 1 });
    const __ir_ks = try ks.finish(&.{}, std.testing.allocator); std.testing.allocator.free(__ir_ks);
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
    const res_ty = mlir.rankedTensorType(&.{ 16, 32 }, mlir.integerType(ctx, .i32));

    _ = k.reshape(x, res_ty, .{});
    _ = k.reshape(x, res_ty, .{ .allow_reorder = true });
    _ = k.reshape(x, res_ty, .{ .allow_reorder = true, .efficient_layout = true });
    _ = k.reshape(x, res_ty, .{ .efficient_layout = true });

    const ir = try k.finish(&.{}, std.testing.allocator);
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

    const i32_ty = mlir.integerType(ctx, .i32);
    const in_ty = mlir.rankedTensorType(&.{512}, i32_ty);
    const mask_ty = mlir.rankedTensorType(&.{512}, mlir.integerType(ctx, .i1));
    const out_ty = mlir.rankedTensorType(&.{16}, i32_ty);

    {
        var k = try Kernel.init(std.testing.allocator, ctx, "histogram", &.{
            .{ .name = "x", .kind = .{ .tensor = .{ .shape = &.{512}, .dtype = .i32 } } },
        }, &.{});
        defer k.deinit();
        _ = k.histogram(k.arg(0), null, out_ty);
        const __ir = try k.finish(&.{}, std.testing.allocator); std.testing.allocator.free(__ir);
        try parityPrintAndVerify(ctx, k.module);
        _ = in_ty;
    }
    {
        var k = try Kernel.init(std.testing.allocator, ctx, "masked_histogram", &.{
            .{ .name = "x", .kind = .{ .tensor = .{ .shape = &.{512}, .dtype = .i32 } } },
            .{ .name = "m", .kind = .{ .tensor = .{ .shape = &.{512}, .dtype = .i1 } } },
        }, &.{});
        defer k.deinit();
        _ = k.histogram(k.arg(0), k.arg(1), out_ty);
        const __ir = try k.finish(&.{}, std.testing.allocator); std.testing.allocator.free(__ir);
        try parityPrintAndVerify(ctx, k.module);
        _ = mask_ty;
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
    const res_ty = mlir.rankedTensorType(&.{ 512, 16 }, mlir.floatType(ctx, .f32));
    const g = k.gather(src, idx, 0, res_ty, .{});

    const __ir = try k.finish(&.{g}, std.testing.allocator); std.testing.allocator.free(__ir);
    try parityPrintAndVerify(ctx, k.module);
}

test "ops_mlir parity: unsplat" {
    const ctx = try setupTestContext();
    defer ctx.deinit();

    var k = try Kernel.init(std.testing.allocator, ctx, "unsplat", &.{
        .{ .name = "x", .kind = .{ .tensor = .{ .shape = &.{ 1, 1 }, .dtype = .f32 } } },
    }, &.{mlir.floatType(ctx, .f32)});
    defer k.deinit();

    const x = k.arg(0);
    const scalar = k.unsplat(x, mlir.floatType(ctx, .f32));

    const __ir = try k.finish(&.{scalar}, std.testing.allocator); std.testing.allocator.free(__ir);
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
            _ = ttir.descriptor_load(c, entry.argument(0), &.{c0.result(0)}, r, .{}, l).appendTo(entry);
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
