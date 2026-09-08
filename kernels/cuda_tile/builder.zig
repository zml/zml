const std = @import("std");

const ct = @import("mlir/dialects/cuda_tile");
const dsl = @import("kernels/common");
const tupleArity = dsl.tupleArity;
const mlir = @import("mlir");
const stdx = @import("stdx");

const cf = @import("control_flow.zig");
const dtypes = @import("dtype.zig");
pub const DType = dtypes.DType;
const isFloatDtype = dtypes.isFloatDtype;
const dtypeBitwidth = dtypes.dtypeBitwidth;
const intBitwidth = dtypes.intBitwidth;

pub const RoundingMode = ct.RoundingMode;
pub const Signedness = ct.Signedness;
pub const IntegerOverflow = ct.IntegerOverflow;
pub const ComparisonPredicate = ct.ComparisonPredicate;
pub const ComparisonOrdering = ct.ComparisonOrdering;
pub const AtomicRMWMode = ct.AtomicRMWMode;
pub const MemoryScope = ct.MemoryScope;
pub const MemoryOrdering = ct.MemoryOrdering;
pub const PaddingValue = ct.PaddingValue;
pub const Arch = ct.Arch;
pub const EntryHint = ct.EntryHint;
pub const LoadStoreHint = ct.LoadStoreHint;
pub const PtrLoadStoreHint = ct.PtrLoadStoreHint;

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Builder);
    std.testing.refAllDecls(Value);
}

pub const dialects_needed = [_][]const u8{"cuda_tile"};

pub const FinishError = error{InvalidMlir} || std.mem.Allocator.Error || std.Io.Writer.Error;

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

    pub fn tile(self: Value) *const ct.TileType {
        return self.type_().isA(ct.TileType) orelse std.debug.panic("Value {f} is not a tile", .{self.type_()});
    }

    pub fn isTile(self: Value) bool {
        return self.type_().isA(ct.TileType) != null;
    }

    pub fn elemType(self: Value) *const mlir.Type {
        return self.tile().elementType();
    }

    pub fn rank(self: Value) usize {
        return self.tile().rank();
    }

    pub fn isScalar(self: Value) bool {
        return self.rank() == 0;
    }

    pub fn dim(self: Value, i: usize) i64 {
        return self.tile().dimension(i);
    }

    pub fn shape(self: Value) Value.Shape {
        var out: Value.Shape = .empty;
        const t = self.tile();
        for (0..t.rank()) |i| out.appendAssumeCapacity(t.dimension(i));
        return out;
    }

    pub fn isFloatElem(self: Value) bool {
        const et = self.elemType();
        inline for (comptime std.meta.fieldNames(mlir.FloatTypes)) |f_name| {
            if (et.isA(mlir.FloatType(@field(mlir.FloatTypes, f_name))) != null) return true;
        }
        return false;
    }

    pub fn isIntElem(self: Value) bool {
        return self.elemType().isA(mlir.IntegerType) != null;
    }

    pub fn isPtrElem(self: Value) bool {
        return self.elemType().isA(ct.PointerType) != null;
    }

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
        return if (l.isFloatElem()) k.divf(l, r) else k.divi(l, r, .signed);
    }

    pub fn rem(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.remf(l, r) else k.remi(l, r, .signed);
    }

    pub fn cdiv(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        if (l.isFloatElem()) @panic("Value.cdiv is integer-only; use div (and ceil) on floats");
        return k.diviOpts(l, r, .{ .signedness = .signed, .rounding = .positive_inf });
    }

    pub fn neg(self: Value) Value {
        const k = self.kern();
        return if (self.isFloatElem()) k.negf(self) else k.negi(self);
    }

    pub fn abs(self: Value) Value {
        const k = self.kern();
        return if (self.isFloatElem()) k.absf(self) else k.absi(self);
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

    pub fn bitXor(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.xori(l, r);
    }

    pub fn shl(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.shli(l, r);
    }

    pub fn shr(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.shri(l, r, .signed);
    }

    pub fn minimum(self: Value, rhs: anytype) Value {
        return self.kern().minimum(self, rhs);
    }

    pub fn maximum(self: Value, rhs: anytype) Value {
        return self.kern().maximum(self, rhs);
    }

    fn compare(self: Value, rhs: anytype, pred: ComparisonPredicate) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.cmpf(pred, l, r) else k.cmpi(pred, l, r, .signed);
    }

    pub fn lt(self: Value, rhs: anytype) Value {
        return self.compare(rhs, .less_than);
    }
    pub fn le(self: Value, rhs: anytype) Value {
        return self.compare(rhs, .less_than_or_equal);
    }
    pub fn gt(self: Value, rhs: anytype) Value {
        return self.compare(rhs, .greater_than);
    }
    pub fn ge(self: Value, rhs: anytype) Value {
        return self.compare(rhs, .greater_than_or_equal);
    }
    pub fn eq(self: Value, rhs: anytype) Value {
        return self.compare(rhs, .equal);
    }
    pub fn ne(self: Value, rhs: anytype) Value {
        return self.compare(rhs, .not_equal);
    }

    pub fn to(self: Value, dtype: DType) Value {
        return self.kern().cast(self, dtype);
    }

    pub fn reshape(self: Value, shape_: []const i64) Value {
        return self.kern().reshape(self, shape_);
    }

    pub fn broadcastTo(self: Value, shape_: []const i64) Value {
        return self.kern().broadcastTo(self, shape_);
    }

    pub fn expandDims(self: Value, axis: usize) Value {
        return self.kern().expandDims(self, axis);
    }

    pub fn permute(self: Value, order: []const i32) Value {
        return self.kern().permute(self, order);
    }

    pub fn sum(self: Value, axis: usize) Value {
        return self.kern().sum(self, axis);
    }

    pub fn max(self: Value, axis: usize) Value {
        return self.kern().max(self, axis);
    }

    pub fn min(self: Value, axis: usize) Value {
        return self.kern().min(self, axis);
    }

    pub fn cumsum(self: Value, axis: usize) Value {
        return self.kern().cumsum(self, axis);
    }

    pub fn offset(self: Value, off: anytype) Value {
        const k = self.kern();
        const o: Value = if (@TypeOf(off) == Value) off else k.lift(off);
        const p, const o2 = k.coerceShapes(self, o);
        return k.offset(p, o2);
    }
};

pub const Loaded = struct {
    tile: Value,
    token: Value,
};

pub const BlockId = struct { x: Value, y: Value, z: Value };

pub const ArgSpec = struct {
    name: []const u8,
    kind: Kind,

    /// XLA passes one raw device pointer per custom-call argument, so every
    /// parameter is a rank-0 `tile<ptr<T>>` and views are built inside the
    /// kernel. A scalar reaches the kernel through a pointer (`loadPtr` on a
    /// rank-0 pointer tile) or as a comptime constant in `cfg`.
    pub const Kind = union(enum) {
        ptr: DType,
        ptr_opts: PtrOpts,
    };

    /// `div_by` becomes a `cuda_tile.assume #cuda_tile.div_by<N>` at the top
    /// of the entry, and `arg(i)` returns the refined value. XLA guarantees
    /// only 16 bytes on the pointer it passes.
    pub const PtrOpts = struct {
        dtype: DType,
        div_by: ?u32 = 16,
    };
};

pub const Opts = struct {
    hints: []const EntryHint = &.{},
};

pub const FloatOpts = struct {
    rounding: RoundingMode = .nearest_even,
    flush_to_zero: bool = false,
};

pub const MinMaxOpts = struct {
    propagate_nan: bool = false,
    flush_to_zero: bool = false,
};

pub const IntOpts = struct {
    overflow: IntegerOverflow = .none,
};

pub const DivIOpts = struct {
    signedness: Signedness = .signed,
    rounding: ?RoundingMode = null,
};

/// `rounding` null picks the target's legal default: `nearest_even`, except
/// `f8e8m0fnu`, which the dialect only lets you reach with `zero` or
/// `positive_inf`.
pub const CastOpts = struct {
    rounding: ?RoundingMode = null,
    signedness: Signedness = .signed,
};

/// `ftoi` admits exactly one rounding mode (`nearest_int_to_zero`), so only
/// the signedness is a choice.
pub const FToIOpts = struct {
    signedness: Signedness = .signed,
};

pub const MmaOpts = struct {
    /// 13.3 only.
    fast_acc: bool = false,
};

pub const PartitionOpts = struct {
    padding: ?PaddingValue = .zero,
    dim_map: []const i32 = &.{},
};

pub const LoadOpts = struct {
    ordering: MemoryOrdering = .weak,
    scope: ?MemoryScope = null,
    token: ?Value = null,
    hints: []const LoadStoreHint = &.{},
};

pub const StoreOpts = LoadOpts;

/// The pointer arm takes `latency` hints only; `allow_tma` is a view-op hint.
pub const LoadPtrOpts = struct {
    mask: ?Value = null,
    padding: ?Value = null,
    ordering: MemoryOrdering = .weak,
    scope: ?MemoryScope = null,
    token: ?Value = null,
    hints: []const PtrLoadStoreHint = &.{},
};

pub const StorePtrOpts = struct {
    mask: ?Value = null,
    ordering: MemoryOrdering = .weak,
    scope: ?MemoryScope = null,
    token: ?Value = null,
    hints: []const PtrLoadStoreHint = &.{},
};

pub const AtomicOpts = struct {
    ordering: MemoryOrdering = .acq_rel,
    scope: MemoryScope = .device,
    mask: ?Value = null,
    token: ?Value = null,
};

/// `atomic_red_view_tko` is relaxed-only, scoped to the block or device, and
/// has no mask operand — the view's padding decides what is touched.
pub const RedViewOpts = struct {
    scope: MemoryScope = .device,
    token: ?Value = null,
};

pub const AllocaOpts = struct {
    alignment: i64 = 16,
    global: bool = false,
};

pub const GlobalOpts = struct {
    alignment: ?i64 = null,
    constant: bool = false,
    visibility: ?ct.SymbolVisibility = null,
};

pub const ScanOpts = struct {
    reverse: bool = false,
};

pub fn ForScope(comptime N: usize) type {
    return cf.ForScope(Builder, Value, N);
}

pub const IfOnlyScope = cf.IfOnlyScope(Builder, Value);

pub fn IfScope(comptime N: usize) type {
    return cf.IfScope(Builder, Value, N);
}

pub fn LoopScope(comptime N: usize) type {
    return cf.LoopScope(Builder, Value, N);
}

/// A custom reduction: `combine(builder, element, accumulator, ctx)` runs on
/// rank-0 tiles of `src`'s element type; `identity` is the element attribute
/// the dialect requires (see `Builder.identityAttr`).
pub fn ReduceArgs(comptime CtxT: type) type {
    return struct {
        src: Value,
        axis: usize,
        identity: *const mlir.Attribute,
        combine: *const fn (*Builder, Value, Value, CtxT) Value,
    };
}

pub fn ScanArgs(comptime CtxT: type) type {
    return struct {
        src: Value,
        axis: usize,
        reverse: bool = false,
        identity: *const mlir.Attribute,
        combine: *const fn (*Builder, Value, Value, CtxT) Value,
    };
}

/// A reduction over several same-shaped tiles at once (argmax, an online
/// softmax's (max, sum)): `combine(builder, elements, accumulators, ctx)`
/// gets one rank-0 tile per source in each slice and returns the new
/// accumulators, one per source; `identities` is one element attribute per
/// source.
pub fn ReduceMultiArgs(comptime CtxT: type) type {
    return struct {
        srcs: []const Value,
        axis: usize,
        identities: []const *const mlir.Attribute,
        combine: *const fn (*Builder, []const Value, []const Value, CtxT) []const Value,
    };
}

pub fn ScanMultiArgs(comptime CtxT: type) type {
    return struct {
        srcs: []const Value,
        axis: usize,
        reverse: bool = false,
        identities: []const *const mlir.Attribute,
        combine: *const fn (*Builder, []const Value, []const Value, CtxT) []const Value,
    };
}

pub const Builder = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    ctx: *mlir.Context,
    module: *mlir.Module,
    name: []const u8,
    func_op: ?*mlir.Operation,
    entry_block: ?*mlir.Block,
    block_stack: std.ArrayList(*mlir.Block),
    ct_module: ?*mlir.Operation = null,
    ct_body: ?*mlir.Block = null,
    refined_args: []?Value = &.{},

    pub fn init(
        allocator: std.mem.Allocator,
        ctx: *mlir.Context,
        name: []const u8,
        args: []const ArgSpec,
    ) !Builder {
        return initOpts(allocator, ctx, name, args, .{});
    }

    pub fn initOpts(
        allocator: std.mem.Allocator,
        ctx: *mlir.Context,
        name: []const u8,
        args: []const ArgSpec,
        opts: Opts,
    ) !Builder {
        var b = try open(allocator, ctx, name);
        errdefer b.deinit();
        try b.declareArgsLowOpts(args, opts);
        return b;
    }

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

    pub fn declareArgs(self: *Builder, spec: anytype) !dsl.NamedArgs(@TypeOf(spec), Value) {
        return self.declareArgsOpts(spec, .{});
    }

    pub fn declareArgsOpts(self: *Builder, spec: anytype, opts: Opts) !dsl.NamedArgs(@TypeOf(spec), Value) {
        const Spec = @TypeOf(spec);
        const field_names = @typeInfo(Spec).@"struct".field_names;

        var arg_specs: [field_names.len]ArgSpec = undefined;
        inline for (0.., field_names) |i, field_name| {
            const raw = @field(spec, field_name);
            const kind: ArgSpec.Kind = if (@TypeOf(raw) == ArgSpec.Kind) raw else blk: {
                const variant = @typeInfo(@TypeOf(raw)).@"struct".field_names[0];
                const tag = @field(std.meta.Tag(ArgSpec.Kind), variant);
                const inner = @field(raw, variant);
                break :blk switch (tag) {
                    .ptr => .{ .ptr = inner },
                    .ptr_opts => .{ .ptr_opts = .{
                        .dtype = inner.dtype,
                        .div_by = if (@hasField(@TypeOf(inner), "div_by")) inner.div_by else 16,
                    } },
                };
            };
            arg_specs[i] = .{ .name = field_name, .kind = kind };
        }

        try self.declareArgsLowOpts(&arg_specs, opts);

        var named: dsl.NamedArgs(Spec, Value) = undefined;
        inline for (0.., field_names) |i, field_name| {
            @field(named, field_name) = self.arg(i);
        }
        return named;
    }

    fn declareArgsLowOpts(self: *Builder, args: []const ArgSpec, opts: Opts) !void {
        std.debug.assert(self.entry_block == null);
        const ctx = self.ctx;
        const unknown_loc: *const mlir.Location = .unknown(ctx);
        const scratch = self.arena.allocator();

        const arg_types = try scratch.alloc(*const mlir.Type, args.len);
        const arg_locs = try scratch.alloc(*const mlir.Location, args.len);
        for (arg_types, arg_locs, args) |*ty, *a_loc, a| {
            ty.* = switch (a.kind) {
                .ptr => |dt| self.ptrTy(dt),
                .ptr_opts => |p| self.ptrTy(p.dtype),
            };
            a_loc.* = unknown_loc;
        }

        const entry = mlir.Block.init(arg_types, arg_locs);
        const ct_body = mlir.Block.init(&.{}, &.{});
        const ct_module = ct.module(ctx, self.name, ct_body, unknown_loc);
        _ = ct_module.appendTo(self.module.body());

        const func_op = ct.entry(ctx, .{
            .name = self.name,
            .block = entry,
            .optimization_hints = if (opts.hints.len > 0) ct.optimizationHints(ctx, opts.hints) else null,
            .location = unknown_loc,
        });
        _ = func_op.appendTo(ct_body);

        self.func_op = func_op;
        self.entry_block = entry;
        self.ct_module = ct_module;
        self.ct_body = ct_body;

        self.refined_args = try scratch.alloc(?Value, args.len);
        for (self.refined_args, args, 0..) |*refined, a, i| {
            refined.* = switch (a.kind) {
                .ptr => self.assumeDivBy(self.rawArg(i), 16),
                .ptr_opts => |p| if (p.div_by) |n| self.assumeDivBy(self.rawArg(i), n) else null,
            };
        }
    }

    pub fn deinit(self: *Builder) void {
        self.module.deinit();
        self.arena.deinit();
    }

    fn rawArg(self: *Builder, i: usize) Value {
        const eb = self.entry_block orelse @panic("Builder.arg called before declareArgs");
        return .{ .inner = eb.argument(i), .kernel = self };
    }

    /// Parameter `i`, refined by its `assume` when one was declared. The
    /// owner is re-stamped: `initOpts` fills `refined_args` through a Builder
    /// that is then moved out of its frame.
    pub fn arg(self: *Builder, i: usize) Value {
        if (i < self.refined_args.len) {
            if (self.refined_args[i]) |v| return .{ .inner = v.inner, .kernel = self };
        }
        return self.rawArg(i);
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

    fn innerSlice(self: *Builder, values: []const Value) []const *const mlir.Value {
        const out = self.arena.allocator().alloc(*const mlir.Value, values.len) catch @panic("Builder.innerSlice OOM");
        for (values, 0..) |v, i| out[i] = v.inner;
        return out;
    }

    fn innerOpt(v: ?Value) ?*const mlir.Value {
        return if (v) |x| x.inner else null;
    }

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

    /// `tile<SHAPExT>`; `&.{}` for a scalar.
    pub fn tileTy(self: *const Builder, shape: []const i64, dtype: DType) *const mlir.Type {
        if (!dtypes.tileLegal(shape)) {
            std.debug.panic("Builder.tileTy: {any} is not a legal tile shape (power-of-two dims, at most 2^24 elements)", .{shape});
        }
        return ct.tileType(self.ctx, shape, dtype.toMlir(self.ctx));
    }

    pub fn scalarTy(self: *const Builder, dtype: DType) *const mlir.Type {
        return self.tileTy(&.{}, dtype);
    }

    /// `tile<ptr<T>>` — the type of a buffer parameter.
    pub fn ptrTy(self: *const Builder, dtype: DType) *const mlir.Type {
        return self.ptrTileTy(&.{}, dtype);
    }

    /// `tile<SHAPExptr<T>>`. `i4` is a tile element type but not a pointee
    /// (Types.td: CudaTile_NumberType excludes Int4), so a 4-bit buffer is
    /// declared as `.i8` and `unpack`ed in the kernel.
    pub fn ptrTileTy(self: *const Builder, shape: []const i64, dtype: DType) *const mlir.Type {
        if (dtype == .i4) @panic("Builder: i4 is a tile element type only, never a pointee; declare the buffer as .i8 and unpack in-kernel");
        return ct.tileType(self.ctx, shape, ct.pointerType(self.ctx, dtype.toMlir(self.ctx)));
    }

    pub fn tokenTy(self: *const Builder) *const mlir.Type {
        return ct.tokenType(self.ctx);
    }

    fn tileLike(self: *const Builder, v: Value, elem: *const mlir.Type) *const mlir.Type {
        return ct.tileType(self.ctx, v.shape().constSlice(), elem);
    }

    fn mlirElemToDType(self: *const Builder, elem: *const mlir.Type) DType {
        return dtypes.mlirElemToDType(self.ctx, elem);
    }

    fn pointee(v: Value) *const mlir.Type {
        const p = v.elemType().isA(ct.PointerType) orelse std.debug.panic("expected a pointer tile, got {f}", .{v.type_()});
        return p.pointee();
    }

    fn elemAttr(self: *Builder, dtype: DType, value: anytype) *const mlir.Attribute {
        const elem = dtype.toMlir(self.ctx);
        const T = @TypeOf(value);
        if (isFloatDtype(dtype)) {
            const v: f64 = switch (@typeInfo(T)) {
                .comptime_int, .int => @floatFromInt(value),
                .comptime_float, .float => @floatCast(value),
                else => @compileError("Builder: unsupported scalar " ++ @typeName(T)),
            };
            return ct.floatElem(self.ctx, elem, v);
        }
        const v: i64 = switch (@typeInfo(T)) {
            .comptime_int => blk: {
                if (value >= std.math.minInt(i64) and value <= std.math.maxInt(i64))
                    break :blk @intCast(value);
                if (value >= 0 and value <= std.math.maxInt(u64))
                    break :blk @bitCast(@as(u64, value));
                @compileError("Builder: integer literal out of 64-bit range");
            },
            .int => |info| if (info.signedness == .signed) @intCast(value) else @bitCast(@as(u64, @intCast(value))),
            .bool => @intFromBool(value),
            .comptime_float, .float => @intFromFloat(value),
            else => @compileError("Builder: unsupported scalar " ++ @typeName(T)),
        };
        return ct.intElem(elem, v);
    }

    fn splat(self: *Builder, tile_ty: *const mlir.Type, dtype: DType, value: anytype) Value {
        return self.emit(ct.constant(self.ctx, ct.splatAttr(tile_ty, self.elemAttr(dtype, value)), tile_ty, self.loc()));
    }

    pub fn cst(self: *Builder, dtype: DType, value: anytype) Value {
        return self.splat(self.scalarTy(dtype), dtype, value);
    }

    pub fn full(self: *Builder, shape: []const i64, value: anytype, dtype: DType) Value {
        return self.splat(self.tileTy(shape, dtype), dtype, value);
    }

    pub fn zeros(self: *Builder, shape: []const i64, dtype: DType) Value {
        return self.full(shape, 0, dtype);
    }

    pub fn ones(self: *Builder, shape: []const i64, dtype: DType) Value {
        return self.full(shape, 1, dtype);
    }

    /// A dense constant from a slice of Zig scalars in row-major order. MLIR
    /// stores every element in whole bytes except `i1`, which it bit-packs;
    /// the slice must hold exactly that many bytes (use `full` for a splat).
    pub fn dense(self: *Builder, shape: []const i64, dtype: DType, values: anytype) Value {
        checkDenseBytes("Builder.dense", shape, dtype, values);
        const ty = self.tileTy(shape, dtype);
        return self.emit(ct.constant(self.ctx, ct.denseAttr(ty, values), ty, self.loc()));
    }

    fn checkDenseBytes(comptime who: []const u8, shape: []const i64, dtype: DType, values: anytype) void {
        var elems: usize = 1;
        for (shape) |d| elems *= @intCast(d);
        const bits = dtypeBitwidth(dtype);
        const expected: usize = if (bits == 1) (elems + 7) / 8 else ((bits + 7) / 8) * elems;
        const got = std.mem.sliceAsBytes(values).len;
        if (got != expected) {
            std.debug.panic("{s}: {d} bytes given for {any} x {s}, {d} expected", .{ who, got, shape, @tagName(dtype), expected });
        }
    }

    pub fn iota(self: *Builder, n: i64, dtype: DType) Value {
        return self.emit(ct.iota(self.ctx, self.tileTy(&.{n}, dtype), self.loc()));
    }

    /// The element attribute `reduce`/`scan` need as an identity.
    pub fn identityAttr(self: *Builder, dtype: DType, value: anytype) *const mlir.Attribute {
        return self.elemAttr(dtype, value);
    }

    /// Lift a Zig scalar to a rank-0 constant. Dtype follows the source type;
    /// unsigned ints are bit-cast. Use `liftAs` for an explicit dtype.
    pub fn lift(self: *Builder, value: anytype) Value {
        const T = @TypeOf(value);
        if (T == Value) return value;
        return switch (@typeInfo(T)) {
            .comptime_int => blk: {
                if (value >= std.math.minInt(i32) and value <= std.math.maxInt(i32))
                    break :blk self.cst(.i32, value);
                break :blk self.cst(.i64, value);
            },
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

    pub fn liftAs(self: *Builder, value: anytype, dtype: DType) Value {
        if (@TypeOf(value) == Value) return value;
        return self.cst(dtype, value);
    }

    fn liftMatching(self: *Builder, value: anytype, ref_elem: *const mlir.Type) Value {
        const T = @TypeOf(value);
        if (T == Value) return value;
        return switch (@typeInfo(T)) {
            .comptime_int, .comptime_float => self.liftAs(value, self.mlirElemToDType(ref_elem)),
            else => self.lift(value),
        };
    }

    fn constMatching(self: *Builder, value: anytype, ref: *const mlir.Type) Value {
        const elem = if (ref.isA(ct.TileType)) |t| t.elementType() else ref;
        return self.liftAs(value, self.mlirElemToDType(elem));
    }

    pub fn broadcastLike(self: *Builder, value: anytype, ref: Value) Value {
        const v = self.liftMatching(value, ref.elemType());
        return self.expandTo(v, ref.shape().constSlice());
    }

    fn expandTo(self: *Builder, v: Value, target: []const i64) Value {
        const cur = v.shape();
        if (std.mem.eql(i64, cur.constSlice(), target)) return v;
        var aligned = v;
        if (cur.len != target.len) {
            std.debug.assert(cur.len <= target.len);
            var padded: Value.Shape = .empty;
            for (0..target.len - cur.len) |_| padded.appendAssumeCapacity(1);
            padded.appendSliceAssumeCapacity(cur.constSlice());
            aligned = self.reshape(v, padded.constSlice());
        }
        if (std.mem.eql(i64, aligned.shape().constSlice(), target)) return aligned;
        return self.broadcastTo(aligned, target);
    }

    fn coerceShapes(self: *Builder, a: Value, b: Value) struct { Value, Value } {
        const a_sh = a.shape();
        const b_sh = b.shape();
        if (std.mem.eql(i64, a_sh.constSlice(), b_sh.constSlice())) return .{ a, b };
        const n = @max(a_sh.len, b_sh.len);
        var target: Value.Shape = .empty;
        for (0..n) |i| {
            const ai: i64 = if (i + a_sh.len >= n) a_sh.get(i + a_sh.len - n) else 1;
            const bi: i64 = if (i + b_sh.len >= n) b_sh.get(i + b_sh.len - n) else 1;
            if (ai == bi or bi == 1) {
                target.appendAssumeCapacity(ai);
            } else if (ai == 1) {
                target.appendAssumeCapacity(bi);
            } else {
                std.debug.panic("Builder.coerce: incompatible shapes {any} vs {any} at axis {d}", .{ a_sh.constSlice(), b_sh.constSlice(), i });
            }
        }
        return .{ self.expandTo(a, target.constSlice()), self.expandTo(b, target.constSlice()) };
    }

    pub fn coerce(self: *Builder, a: anytype, b: anytype) struct { Value, Value } {
        const a_ref: ?*const mlir.Type = if (@TypeOf(b) == Value) b.elemType() else null;
        const b_ref: ?*const mlir.Type = if (@TypeOf(a) == Value) a.elemType() else null;
        var av = if (a_ref) |t| self.liftMatching(a, t) else self.lift(a);
        var bv = if (b_ref) |t| self.liftMatching(b, t) else self.lift(b);

        if (av.isIntElem() and bv.isIntElem()) {
            const a_dt = self.mlirElemToDType(av.elemType());
            const b_dt = self.mlirElemToDType(bv.elemType());
            const aw = intBitwidth(a_dt);
            const bw = intBitwidth(b_dt);
            if (aw < bw) av = av.to(b_dt);
            if (bw < aw) bv = bv.to(a_dt);
        }

        return self.coerceShapes(av, bv);
    }

    pub fn tileBlockId(self: *Builder) BlockId {
        const r = self.emitMulti(ct.get_tile_block_id(self.ctx, self.loc()), 3);
        return .{ .x = r[0], .y = r[1], .z = r[2] };
    }

    pub fn numTileBlocks(self: *Builder) BlockId {
        const r = self.emitMulti(ct.get_num_tile_blocks(self.ctx, self.loc()), 3);
        return .{ .x = r[0], .y = r[1], .z = r[2] };
    }

    pub fn assumeDivBy(self: *Builder, v: Value, n: u64) Value {
        return self.emit(ct.assume(self.ctx, v.inner, ct.divBy(self.ctx, n, null, null), self.loc()));
    }

    pub fn assumeDivByEvery(self: *Builder, v: Value, n: u64, every: i64, along: i64) Value {
        return self.emit(ct.assume(self.ctx, v.inner, ct.divBy(self.ctx, n, every, along), self.loc()));
    }

    pub fn assumeSameElements(self: *Builder, v: Value, values: []const i64) Value {
        return self.emit(ct.assume(self.ctx, v.inner, ct.sameElements(self.ctx, values), self.loc()));
    }

    pub fn assumeBounded(self: *Builder, v: Value, lb: ?i64, ub: ?i64) Value {
        return self.emit(ct.assume(self.ctx, v.inner, ct.bounded(self.ctx, lb, ub), self.loc()));
    }

    pub fn tensorView(self: *Builder, base: Value, shape: []const i64, strides: []const i64) Value {
        const ty = ct.tensorViewType(self.ctx, pointee(base), shape, strides);
        return self.emit(ct.make_tensor_view(self.ctx, base.inner, &.{}, &.{}, ty, self.loc()));
    }

    pub const Dim = union(enum) {
        static: i64,
        dynamic: Value,
    };

    pub fn tensorViewDyn(self: *Builder, base: Value, shape: []const Dim, strides: []const Dim) Value {
        const scratch = self.arena.allocator();
        var static_shape: Value.Shape = .empty;
        var static_strides: Value.Shape = .empty;
        var dyn_shape = std.ArrayList(*const mlir.Value).empty;
        var dyn_strides = std.ArrayList(*const mlir.Value).empty;
        for (shape) |d| switch (d) {
            .static => |s| static_shape.appendAssumeCapacity(s),
            .dynamic => |v| {
                static_shape.appendAssumeCapacity(ct.TensorViewType.dynamic());
                dyn_shape.append(scratch, v.inner) catch @panic("OOM");
            },
        };
        for (strides) |d| switch (d) {
            .static => |s| static_strides.appendAssumeCapacity(s),
            .dynamic => |v| {
                static_strides.appendAssumeCapacity(ct.TensorViewType.dynamic());
                dyn_strides.append(scratch, v.inner) catch @panic("OOM");
            },
        };
        const ty = ct.tensorViewType(self.ctx, pointee(base), static_shape.constSlice(), static_strides.constSlice());
        return self.emit(ct.make_tensor_view(self.ctx, base.inner, dyn_shape.items, dyn_strides.items, ty, self.loc()));
    }

    fn i32Slice(self: *Builder, values: []const i64) []const i32 {
        const out = self.arena.allocator().alloc(i32, values.len) catch @panic("OOM");
        for (values, 0..) |v, i| out[i] = @intCast(v);
        return out;
    }

    /// Tiles `tv` into `tile`-shaped pieces addressed by tile-space indices.
    /// `padding` fills the ragged edge on load; stores past the edge are
    /// masked. That is why this DSL has no mask tile.
    pub fn partitionView(self: *Builder, tv: Value, tile: []const i64, opts: PartitionOpts) Value {
        if (!dtypes.tileLegal(tile)) std.debug.panic("Builder.partitionView: {any} is not a legal tile shape", .{tile});
        const ty: *const mlir.Type = @ptrCast(ct.PartitionViewType.get(self.ctx, self.i32Slice(tile), tv.type_(), opts.dim_map, opts.padding));
        return self.emit(ct.make_partition_view(self.ctx, tv.inner, ty, self.loc()));
    }

    /// 13.3: like `partitionView`, stepping by `traversal_strides` elements
    /// per index instead of a whole tile.
    pub fn stridedView(self: *Builder, tv: Value, tile: []const i64, traversal_strides: []const i64, opts: PartitionOpts) Value {
        if (!dtypes.tileLegal(tile)) std.debug.panic("Builder.stridedView: {any} is not a legal tile shape", .{tile});
        const ty: *const mlir.Type = @ptrCast(ct.StridedViewType.get(self.ctx, self.i32Slice(tile), self.i32Slice(traversal_strides), tv.type_(), opts.dim_map, opts.padding));
        return self.emit(ct.make_strided_view(self.ctx, tv.inner, ty, self.loc()));
    }

    /// 13.3: a view whose `sparse_dim` is indexed by a rank-1 tile of
    /// `tile[sparse_dim]` indices (every other index stays rank 0).
    pub fn gatherScatterView(self: *Builder, tv: Value, tile: []const i64, sparse_dim: u32, padding: ?PaddingValue) Value {
        if (!dtypes.tileLegal(tile)) std.debug.panic("Builder.gatherScatterView: {any} is not a legal tile shape", .{tile});
        const ty = ct.gatherScatterViewType(self.ctx, tile, tv.type_(), sparse_dim, padding);
        return self.emit(ct.make_gather_scatter_view(self.ctx, tv.inner, ty, self.loc()));
    }

    fn i32Scalars(self: *Builder, n: usize) []const *const mlir.Type {
        const out = self.arena.allocator().alloc(*const mlir.Type, n) catch @panic("OOM");
        for (out) |*t| t.* = self.scalarTy(.i32);
        return out;
    }

    pub fn indexSpaceShape(self: *Builder, view: Value) []Value {
        const n = ct.indexRankOfView(view.type_());
        return self.emitMulti(ct.get_index_space_shape(self.ctx, view.inner, self.i32Scalars(n), self.loc()), n);
    }

    pub fn tensorShape(self: *Builder, tv: Value) []Value {
        const t = tv.type_().isA(ct.TensorViewType) orelse std.debug.panic("expected a tensor_view, got {f}", .{tv.type_()});
        const n = t.rank();
        return self.emitMulti(ct.get_tensor_shape(self.ctx, tv.inner, self.i32Scalars(n), self.loc()), n);
    }

    fn memOpts(self: *Builder, ordering: MemoryOrdering, scope: ?MemoryScope, token: ?Value, hints: anytype) ct.MemOpts {
        return .{
            .ordering = ordering,
            .scope = scope,
            .token = innerOpt(token),
            .optimization_hints = if (hints.len > 0) ct.optimizationHints(self.ctx, hints) else null,
        };
    }

    pub fn load(self: *Builder, view: Value, index: []const Value) Value {
        return self.loadOpts(view, index, .{}).tile;
    }

    pub fn loadOpts(self: *Builder, view: Value, index: []const Value, opts: LoadOpts) Loaded {
        const expected = ct.indexRankOfView(view.type_());
        if (index.len != expected) std.debug.panic("Builder.load: view takes {d} indices, got {d}", .{ expected, index.len });
        const r = self.emitMulti(ct.load_view_tko(self.ctx, view.inner, self.innerSlice(index), self.memOpts(opts.ordering, opts.scope, opts.token, opts.hints), self.loc()), 2);
        return .{ .tile = r[0], .token = r[1] };
    }

    pub fn store(self: *Builder, tile: Value, view: Value, index: []const Value) Value {
        return self.storeOpts(tile, view, index, .{});
    }

    pub fn storeOpts(self: *Builder, tile: Value, view: Value, index: []const Value, opts: StoreOpts) Value {
        const expected = ct.indexRankOfView(view.type_());
        if (index.len != expected) std.debug.panic("Builder.store: view takes {d} indices, got {d}", .{ expected, index.len });
        return self.emit(ct.store_view_tko(self.ctx, tile.inner, view.inner, self.innerSlice(index), self.memOpts(opts.ordering, opts.scope, opts.token, opts.hints), self.loc()));
    }

    /// Gather through a tile of pointers, dropping the token. This arm cannot
    /// use TMA; prefer views for dense access.
    pub fn loadPtr(self: *Builder, ptrs: Value) Value {
        return self.loadPtrOpts(ptrs, .{}).tile;
    }

    pub fn loadPtrOpts(self: *Builder, ptrs: Value, opts: LoadPtrOpts) Loaded {
        const result_ty = self.tileLike(ptrs, pointee(ptrs));
        const mask: ?Value = if (opts.mask) |m| self.expandTo(m, ptrs.shape().constSlice()) else null;
        const r = self.emitMulti(ct.load_ptr_tko(self.ctx, ptrs.inner, result_ty, .{
            .mask = innerOpt(mask),
            .padding = innerOpt(opts.padding),
            .mem = self.memOpts(opts.ordering, opts.scope, opts.token, opts.hints),
        }, self.loc()), 2);
        return .{ .tile = r[0], .token = r[1] };
    }

    pub fn storePtr(self: *Builder, ptrs: Value, value: Value) Value {
        return self.storePtrOpts(ptrs, value, .{});
    }

    pub fn storePtrOpts(self: *Builder, ptrs: Value, value: Value, opts: StorePtrOpts) Value {
        const mask: ?Value = if (opts.mask) |m| self.expandTo(m, ptrs.shape().constSlice()) else null;
        return self.emit(ct.store_ptr_tko(self.ctx, ptrs.inner, value.inner, .{
            .mask = innerOpt(mask),
            .mem = self.memOpts(opts.ordering, opts.scope, opts.token, opts.hints),
        }, self.loc()));
    }

    pub fn makeToken(self: *Builder) Value {
        return self.emit(ct.make_token(self.ctx, self.loc()));
    }

    pub fn joinTokens(self: *Builder, tokens: []const Value) Value {
        return self.emit(ct.join_tokens(self.ctx, self.innerSlice(tokens), self.loc()));
    }

    pub fn print(self: *Builder, str: []const u8, args: []const Value) Value {
        return self.printOpts(str, args, null);
    }

    pub fn printOpts(self: *Builder, str: []const u8, args: []const Value, token: ?Value) Value {
        return self.emit(ct.print_tko(self.ctx, str, self.innerSlice(args), innerOpt(token), self.loc()));
    }

    fn atomicOpts(opts: AtomicOpts) ct.AtomicOpts {
        return .{ .ordering = opts.ordering, .scope = opts.scope, .mask = innerOpt(opts.mask), .token = innerOpt(opts.token) };
    }

    pub fn atomicCas(self: *Builder, ptrs: Value, cmp: Value, val: Value, opts: AtomicOpts) Loaded {
        const r = self.emitMulti(ct.atomic_cas_tko(self.ctx, ptrs.inner, cmp.inner, val.inner, atomicOpts(opts), self.loc()), 2);
        return .{ .tile = r[0], .token = r[1] };
    }

    pub fn atomicRmw(self: *Builder, mode: AtomicRMWMode, ptrs: Value, operand: Value, opts: AtomicOpts) Loaded {
        const r = self.emitMulti(ct.atomic_rmw_tko(self.ctx, ptrs.inner, mode, operand.inner, atomicOpts(opts), self.loc()), 2);
        return .{ .tile = r[0], .token = r[1] };
    }

    /// 13.3: an atomic reduction into an unpadded tiled view; returns the
    /// token. `xchg` is not a reduction and the dialect rejects it.
    pub fn atomicRedView(self: *Builder, mode: AtomicRMWMode, view: Value, index: []const Value, value: Value, opts: RedViewOpts) Value {
        return self.emit(ct.atomic_red_view_tko(self.ctx, view.inner, self.innerSlice(index), mode, value.inner, .{
            .scope = opts.scope,
            .token = innerOpt(opts.token),
        }, self.loc()));
    }

    /// 13.3: `num_elem` elements of `dtype`, as a rank-0 pointer tile.
    pub fn alloca(self: *Builder, dtype: DType, num_elem: i64, opts: AllocaOpts) Value {
        return self.emit(ct.alloca(self.ctx, num_elem, opts.alignment, opts.global, self.ptrTy(dtype), self.loc()));
    }

    pub fn global(self: *Builder, name: []const u8, shape: []const i64, dtype: DType, value: anytype, opts: GlobalOpts) void {
        const ty = self.tileTy(shape, dtype);
        const attr = ct.splatAttr(ty, self.elemAttr(dtype, value));
        self.appendGlobal(ct.global(self.ctx, name, attr, .{ .alignment = opts.alignment, .constant = opts.constant, .visibility = opts.visibility }, self.loc()));
    }

    /// A module-level dense buffer from a slice of Zig scalars (see `dense`
    /// for the byte-count rule).
    pub fn globalDense(self: *Builder, name: []const u8, shape: []const i64, dtype: DType, values: anytype, opts: GlobalOpts) void {
        checkDenseBytes("Builder.globalDense", shape, dtype, values);
        const ty = self.tileTy(shape, dtype);
        self.appendGlobal(ct.global(self.ctx, name, ct.denseAttr(ty, values), .{ .alignment = opts.alignment, .constant = opts.constant, .visibility = opts.visibility }, self.loc()));
    }

    fn appendGlobal(self: *Builder, op: *mlir.Operation) void {
        const body = self.ct_body orelse @panic("Builder.global called before declareArgs");
        const first = body.firstOperation() orelse {
            body.appendOwnedOperation(op);
            return;
        };
        body.insertOwnedOperationBefore(first, op);
    }

    pub fn getGlobal(self: *Builder, name: []const u8, dtype: DType) Value {
        return self.emit(ct.get_global(self.ctx, name, self.ptrTy(dtype), self.loc()));
    }

    pub fn addf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.addfOpts(lhs, rhs, .{});
    }
    pub fn addfOpts(self: *Builder, lhs: Value, rhs: Value, opts: FloatOpts) Value {
        return self.emit(ct.addf(self.ctx, lhs.inner, rhs.inner, opts.rounding, opts.flush_to_zero, self.loc()));
    }
    pub fn subf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.subfOpts(lhs, rhs, .{});
    }
    pub fn subfOpts(self: *Builder, lhs: Value, rhs: Value, opts: FloatOpts) Value {
        return self.emit(ct.subf(self.ctx, lhs.inner, rhs.inner, opts.rounding, opts.flush_to_zero, self.loc()));
    }
    pub fn mulf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.mulfOpts(lhs, rhs, .{});
    }
    pub fn mulfOpts(self: *Builder, lhs: Value, rhs: Value, opts: FloatOpts) Value {
        return self.emit(ct.mulf(self.ctx, lhs.inner, rhs.inner, opts.rounding, opts.flush_to_zero, self.loc()));
    }
    pub fn divf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.divfOpts(lhs, rhs, .{});
    }
    pub fn divfOpts(self: *Builder, lhs: Value, rhs: Value, opts: FloatOpts) Value {
        return self.emit(ct.divf(self.ctx, lhs.inner, rhs.inner, opts.rounding, opts.flush_to_zero, self.loc()));
    }
    pub fn remf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(ct.remf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn negf(self: *Builder, x: Value) Value {
        return self.emit(ct.negf(self.ctx, x.inner, self.loc()));
    }
    pub fn absf(self: *Builder, x: Value) Value {
        return self.emit(ct.absf(self.ctx, x.inner, self.loc()));
    }
    pub fn fma(self: *Builder, a: Value, b: Value, acc: Value) Value {
        return self.fmaOpts(a, b, acc, .{});
    }
    pub fn fmaOpts(self: *Builder, a: Value, b: Value, acc: Value, opts: FloatOpts) Value {
        return self.emit(ct.fma(self.ctx, a.inner, b.inner, acc.inner, opts.rounding, opts.flush_to_zero, self.loc()));
    }
    pub fn maxf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.maxfOpts(lhs, rhs, .{});
    }
    pub fn maxfOpts(self: *Builder, lhs: Value, rhs: Value, opts: MinMaxOpts) Value {
        return self.emit(ct.maxf(self.ctx, lhs.inner, rhs.inner, opts.propagate_nan, opts.flush_to_zero, self.loc()));
    }
    pub fn minf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.minfOpts(lhs, rhs, .{});
    }
    pub fn minfOpts(self: *Builder, lhs: Value, rhs: Value, opts: MinMaxOpts) Value {
        return self.emit(ct.minf(self.ctx, lhs.inner, rhs.inner, opts.propagate_nan, opts.flush_to_zero, self.loc()));
    }
    pub fn pow(self: *Builder, x: Value, exponent: Value) Value {
        return self.emit(ct.pow(self.ctx, x.inner, exponent.inner, self.loc()));
    }
    pub fn atan2(self: *Builder, x: Value, y: Value) Value {
        return self.emit(ct.atan2(self.ctx, x.inner, y.inner, self.loc()));
    }
    pub fn exp(self: *Builder, x: Value) Value {
        return self.emit(ct.exp(self.ctx, x.inner, null, self.loc()));
    }
    /// 13.3: `exp` with an explicit rounding mode.
    pub fn expRound(self: *Builder, x: Value, rounding: RoundingMode) Value {
        return self.emit(ct.exp(self.ctx, x.inner, rounding, self.loc()));
    }
    pub fn exp2(self: *Builder, x: Value) Value {
        return self.exp2Opts(x, false);
    }
    pub fn exp2Opts(self: *Builder, x: Value, flush_to_zero: bool) Value {
        return self.emit(ct.exp2(self.ctx, x.inner, flush_to_zero, self.loc()));
    }
    pub fn log(self: *Builder, x: Value) Value {
        return self.emit(ct.log(self.ctx, x.inner, self.loc()));
    }
    pub fn log2(self: *Builder, x: Value) Value {
        return self.emit(ct.log2(self.ctx, x.inner, self.loc()));
    }
    pub fn sqrt(self: *Builder, x: Value) Value {
        return self.sqrtOpts(x, .{});
    }
    pub fn sqrtOpts(self: *Builder, x: Value, opts: FloatOpts) Value {
        return self.emit(ct.sqrt(self.ctx, x.inner, opts.rounding, opts.flush_to_zero, self.loc()));
    }
    pub fn rsqrt(self: *Builder, x: Value) Value {
        return self.rsqrtOpts(x, false);
    }
    pub fn rsqrtOpts(self: *Builder, x: Value, flush_to_zero: bool) Value {
        return self.emit(ct.rsqrt(self.ctx, x.inner, flush_to_zero, self.loc()));
    }
    pub fn sin(self: *Builder, x: Value) Value {
        return self.emit(ct.sin(self.ctx, x.inner, self.loc()));
    }
    pub fn cos(self: *Builder, x: Value) Value {
        return self.emit(ct.cos(self.ctx, x.inner, self.loc()));
    }
    pub fn tan(self: *Builder, x: Value) Value {
        return self.emit(ct.tan(self.ctx, x.inner, self.loc()));
    }
    pub fn sinh(self: *Builder, x: Value) Value {
        return self.emit(ct.sinh(self.ctx, x.inner, self.loc()));
    }
    pub fn cosh(self: *Builder, x: Value) Value {
        return self.emit(ct.cosh(self.ctx, x.inner, self.loc()));
    }
    pub fn tanh(self: *Builder, x: Value) Value {
        return self.emit(ct.tanh(self.ctx, x.inner, null, self.loc()));
    }
    /// 13.2: `tanh` with an explicit rounding mode.
    pub fn tanhRound(self: *Builder, x: Value, rounding: RoundingMode) Value {
        return self.emit(ct.tanh(self.ctx, x.inner, rounding, self.loc()));
    }
    pub fn ceil(self: *Builder, x: Value) Value {
        return self.emit(ct.ceil(self.ctx, x.inner, self.loc()));
    }
    pub fn floor(self: *Builder, x: Value) Value {
        return self.emit(ct.floor(self.ctx, x.inner, self.loc()));
    }

    pub fn cmpf(self: *Builder, predicate: ComparisonPredicate, lhs: Value, rhs: Value) Value {
        return self.cmpfOpts(predicate, lhs, rhs, .ordered);
    }
    pub fn cmpfOpts(self: *Builder, predicate: ComparisonPredicate, lhs: Value, rhs: Value, ordering: ComparisonOrdering) Value {
        return self.emit(ct.cmpf(self.ctx, predicate, ordering, lhs.inner, rhs.inner, self.loc()));
    }

    pub fn mmaf(self: *Builder, a: Value, b: Value, acc: Value) Value {
        return self.mmafOpts(a, b, acc, .{});
    }
    pub fn mmafOpts(self: *Builder, a: Value, b: Value, acc: Value, opts: MmaOpts) Value {
        return self.emit(ct.mmaf(self.ctx, a.inner, b.inner, acc.inner, opts.fast_acc, self.loc()));
    }
    /// 13.3: block-scaled MMA. Scales are plain tiles in logical layout —
    /// `a_scale` is `[M, K/V]`, `b_scale` is `[K/V, N]` — never pre-swizzled.
    pub fn mmafScaled(self: *Builder, a: Value, b: Value, acc: Value, a_scale: Value, b_scale: Value) Value {
        return self.emit(ct.mmaf_scaled(self.ctx, a.inner, b.inner, acc.inner, a_scale.inner, b_scale.inner, self.loc()));
    }
    pub fn mmai(self: *Builder, a: Value, b: Value, acc: Value, a_signedness: Signedness, b_signedness: Signedness) Value {
        return self.emit(ct.mmai(self.ctx, a.inner, b.inner, acc.inner, a_signedness, b_signedness, self.loc()));
    }

    pub fn addi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.addiOpts(lhs, rhs, .{});
    }
    pub fn addiOpts(self: *Builder, lhs: Value, rhs: Value, opts: IntOpts) Value {
        return self.emit(ct.addi(self.ctx, lhs.inner, rhs.inner, opts.overflow, self.loc()));
    }
    pub fn subi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.subiOpts(lhs, rhs, .{});
    }
    pub fn subiOpts(self: *Builder, lhs: Value, rhs: Value, opts: IntOpts) Value {
        return self.emit(ct.subi(self.ctx, lhs.inner, rhs.inner, opts.overflow, self.loc()));
    }
    pub fn muli(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.muliOpts(lhs, rhs, .{});
    }
    pub fn muliOpts(self: *Builder, lhs: Value, rhs: Value, opts: IntOpts) Value {
        return self.emit(ct.muli(self.ctx, lhs.inner, rhs.inner, opts.overflow, self.loc()));
    }
    pub fn mulhii(self: *Builder, x: Value, y: Value) Value {
        return self.emit(ct.mulhii(self.ctx, x.inner, y.inner, self.loc()));
    }
    pub fn divi(self: *Builder, lhs: Value, rhs: Value, signedness: Signedness) Value {
        return self.diviOpts(lhs, rhs, .{ .signedness = signedness });
    }
    pub fn diviOpts(self: *Builder, lhs: Value, rhs: Value, opts: DivIOpts) Value {
        return self.emit(ct.divi(self.ctx, lhs.inner, rhs.inner, opts.signedness, opts.rounding, self.loc()));
    }
    pub fn remi(self: *Builder, lhs: Value, rhs: Value, signedness: Signedness) Value {
        return self.emit(ct.remi(self.ctx, lhs.inner, rhs.inner, signedness, self.loc()));
    }
    pub fn maxi(self: *Builder, lhs: Value, rhs: Value, signedness: Signedness) Value {
        return self.emit(ct.maxi(self.ctx, lhs.inner, rhs.inner, signedness, self.loc()));
    }
    pub fn mini(self: *Builder, lhs: Value, rhs: Value, signedness: Signedness) Value {
        return self.emit(ct.mini(self.ctx, lhs.inner, rhs.inner, signedness, self.loc()));
    }
    pub fn negi(self: *Builder, x: Value) Value {
        return self.emit(ct.negi(self.ctx, x.inner, .none, self.loc()));
    }
    pub fn absi(self: *Builder, x: Value) Value {
        return self.emit(ct.absi(self.ctx, x.inner, self.loc()));
    }
    pub fn shli(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(ct.shli(self.ctx, lhs.inner, rhs.inner, .none, self.loc()));
    }
    pub fn shri(self: *Builder, lhs: Value, rhs: Value, signedness: Signedness) Value {
        return self.emit(ct.shri(self.ctx, lhs.inner, rhs.inner, signedness, self.loc()));
    }
    pub fn andi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(ct.andi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn ori(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(ct.ori(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn xori(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(ct.xori(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    pub fn cmpi(self: *Builder, predicate: ComparisonPredicate, lhs: Value, rhs: Value, signedness: Signedness) Value {
        return self.emit(ct.cmpi(self.ctx, predicate, signedness, lhs.inner, rhs.inner, self.loc()));
    }

    pub fn maximum(self: *Builder, a: anytype, b: anytype) Value {
        const l, const r = self.coerce(a, b);
        return if (l.isFloatElem()) self.maxf(l, r) else self.maxi(l, r, .signed);
    }

    pub fn minimum(self: *Builder, a: anytype, b: anytype) Value {
        const l, const r = self.coerce(a, b);
        return if (l.isFloatElem()) self.minf(l, r) else self.mini(l, r, .signed);
    }

    pub fn select(self: *Builder, cond: Value, t: Value, f: Value) Value {
        return self.emit(ct.select(self.ctx, cond.inner, t.inner, f.inner, self.loc()));
    }

    pub fn where(self: *Builder, cond: Value, x: anytype, y: anytype) Value {
        var xv, var yv = self.coerce(x, y);
        var cv = cond;
        cv, xv = self.coerceShapes(cv, xv);
        xv, yv = self.coerceShapes(xv, yv);
        cv, _ = self.coerceShapes(cv, xv);
        return self.select(cv, xv, yv);
    }

    pub fn bitcast(self: *Builder, src: Value, dtype: DType) Value {
        return self.emit(ct.bitcast(self.ctx, src.inner, self.tileLike(src, dtype.toMlir(self.ctx)), self.loc()));
    }
    pub fn exti(self: *Builder, src: Value, dtype: DType, signedness: Signedness) Value {
        return self.emit(ct.exti(self.ctx, src.inner, signedness, self.tileLike(src, dtype.toMlir(self.ctx)), self.loc()));
    }
    pub fn trunci(self: *Builder, src: Value, dtype: DType) Value {
        return self.emit(ct.trunci(self.ctx, src.inner, self.tileLike(src, dtype.toMlir(self.ctx)), .none, self.loc()));
    }
    pub fn ftof(self: *Builder, src: Value, dtype: DType) Value {
        return self.ftofOpts(src, dtype, .{});
    }
    pub fn ftofOpts(self: *Builder, src: Value, dtype: DType, opts: CastOpts) Value {
        const rounding: RoundingMode = opts.rounding orelse (if (dtype == .f8e8m0fnu) .zero else .nearest_even);
        return self.emit(ct.ftof(self.ctx, src.inner, self.tileLike(src, dtype.toMlir(self.ctx)), rounding, self.loc()));
    }
    /// Float -> integer, truncating toward zero (the only rounding `ftoi`
    /// admits).
    pub fn ftoi(self: *Builder, src: Value, dtype: DType) Value {
        return self.ftoiOpts(src, dtype, .{});
    }
    pub fn ftoiOpts(self: *Builder, src: Value, dtype: DType, opts: FToIOpts) Value {
        return self.emit(ct.ftoi(self.ctx, src.inner, opts.signedness, .nearest_int_to_zero, self.tileLike(src, dtype.toMlir(self.ctx)), self.loc()));
    }
    /// Integer -> float. `f8e8m0fnu` is not a legal target; go through `f32`.
    pub fn itof(self: *Builder, src: Value, dtype: DType) Value {
        return self.itofOpts(src, dtype, .{});
    }
    pub fn itofOpts(self: *Builder, src: Value, dtype: DType, opts: CastOpts) Value {
        if (dtype == .f8e8m0fnu) @panic("Builder.itof: the dialect refuses integer -> f8e8m0fnu; convert to .f32 first (Builder.cast does)");
        return self.emit(ct.itof(self.ctx, src.inner, opts.signedness, opts.rounding orelse .nearest_even, self.tileLike(src, dtype.toMlir(self.ctx)), self.loc()));
    }
    pub fn intToPtr(self: *Builder, src: Value, pointee_dtype: DType) Value {
        return self.emit(ct.int_to_ptr(self.ctx, src.inner, self.ptrTileTy(src.shape().constSlice(), pointee_dtype), self.loc()));
    }
    pub fn ptrToInt(self: *Builder, src: Value) Value {
        return self.emit(ct.ptr_to_int(self.ctx, src.inner, self.tileLike(src, DType.i64.toMlir(self.ctx)), self.loc()));
    }
    pub fn ptrToPtr(self: *Builder, src: Value, pointee_dtype: DType) Value {
        return self.emit(ct.ptr_to_ptr(self.ctx, src.inner, self.ptrTileTy(src.shape().constSlice(), pointee_dtype), self.loc()));
    }
    /// 13.3: a sub-byte tile packed into `tile<SHAPExi8>`.
    pub fn pack(self: *Builder, src: Value, shape: []const i64) Value {
        return self.emit(ct.pack(self.ctx, src.inner, self.tileTy(shape, .i8), self.loc()));
    }
    /// 13.3: an i8 tile unpacked into `tile<SHAPExT>` of a sub-byte `dtype`.
    pub fn unpack(self: *Builder, src: Value, shape: []const i64, dtype: DType) Value {
        return self.emit(ct.unpack(self.ctx, src.inner, self.tileTy(shape, dtype), self.loc()));
    }

    /// Numeric cast with auto-dispatch: ftof / ftoi / itof / exti / trunci /
    /// bitcast by element kind and width. Integers are signed; i1 is not.
    pub fn cast(self: *Builder, src: Value, dtype: DType) Value {
        const cur_elem = src.elemType();
        const tgt_elem = dtype.toMlir(self.ctx);
        if (cur_elem.eql(tgt_elem)) return src;
        const cur_dtype = self.mlirElemToDType(cur_elem);
        const cur_is_float = src.isFloatElem();
        const tgt_is_float = isFloatDtype(dtype);
        if (cur_is_float and tgt_is_float) return self.ftof(src, dtype);
        if (cur_is_float) return self.ftoi(src, dtype);
        if (tgt_is_float) {
            const signedness: Signedness = if (cur_dtype == .i1) .unsigned else .signed;
            if (dtype == .f8e8m0fnu) return self.ftof(self.itofOpts(src, .f32, .{ .signedness = signedness }), dtype);
            return self.itofOpts(src, dtype, .{ .signedness = signedness });
        }
        const cur_bw = dtypeBitwidth(cur_dtype);
        const tgt_bw = dtypeBitwidth(dtype);
        if (tgt_bw > cur_bw) return self.exti(src, dtype, if (cur_dtype == .i1) .unsigned else .signed);
        if (tgt_bw < cur_bw) return self.trunci(src, dtype);
        return self.bitcast(src, dtype);
    }

    pub fn reshape(self: *Builder, src: Value, shape: []const i64) Value {
        return self.emit(ct.reshape(self.ctx, src.inner, self.tileTyOf(shape, src.elemType()), self.loc()));
    }

    fn tileTyOf(self: *const Builder, shape: []const i64, elem: *const mlir.Type) *const mlir.Type {
        if (!dtypes.tileLegal(shape)) std.debug.panic("Builder: {any} is not a legal tile shape", .{shape});
        return ct.tileType(self.ctx, shape, elem);
    }

    pub fn broadcastTo(self: *Builder, src: Value, shape: []const i64) Value {
        return self.emit(ct.broadcast(self.ctx, src.inner, self.tileTyOf(shape, src.elemType()), self.loc()));
    }

    pub fn expandDims(self: *Builder, src: Value, axis: usize) Value {
        const in_shape = src.shape();
        var out: Value.Shape = .empty;
        for (0..in_shape.len + 1) |i| {
            if (i == axis) out.appendAssumeCapacity(1) else out.appendAssumeCapacity(in_shape.get(if (i < axis) i else i - 1));
        }
        return self.reshape(src, out.constSlice());
    }

    pub fn cat(self: *Builder, lhs: Value, rhs: Value, dim: usize) Value {
        var out = lhs.shape();
        out.set(dim, out.get(dim) + rhs.dim(dim));
        return self.emit(ct.cat(self.ctx, lhs.inner, rhs.inner, @intCast(dim), self.tileTyOf(out.constSlice(), lhs.elemType()), self.loc()));
    }

    pub fn extract(self: *Builder, src: Value, indices: []const Value, shape: []const i64) Value {
        return self.emit(ct.extract(self.ctx, src.inner, self.innerSlice(indices), self.tileTyOf(shape, src.elemType()), self.loc()));
    }

    pub fn permute(self: *Builder, src: Value, order: []const i32) Value {
        const src_shape = src.shape();
        std.debug.assert(order.len == src_shape.len);
        var out: Value.Shape = .empty;
        for (order) |i| out.appendAssumeCapacity(src_shape.get(@intCast(i)));
        return self.emit(ct.permute(self.ctx, src.inner, order, self.tileTyOf(out.constSlice(), src.elemType()), self.loc()));
    }

    pub fn transpose(self: *Builder, src: Value) Value {
        std.debug.assert(src.rank() == 2);
        return self.permute(src, &.{ 1, 0 });
    }

    pub fn offset(self: *Builder, ptrs: Value, off: Value) Value {
        return self.emit(ct.offset(self.ctx, ptrs.inner, off.inner, self.loc()));
    }

    fn reducedShape(src: Value, axis: usize) Value.Shape {
        if (axis >= src.rank()) std.debug.panic("Builder.reduce: axis {d} out of range for rank {d}", .{ axis, src.rank() });
        var out: Value.Shape = .empty;
        const s = src.shape();
        for (0..s.len) |i| if (i != axis) out.appendAssumeCapacity(s.get(i));
        return out;
    }

    fn checkAxis(src: Value, axis: usize) void {
        if (axis >= src.rank()) std.debug.panic("Builder.scan: axis {d} out of range for rank {d}", .{ axis, src.rank() });
    }

    fn combineBody(self: *Builder, ctx: anytype, elem: *const mlir.Type, combine: anytype) *mlir.Block {
        const scalar = ct.tileType(self.ctx, &.{}, elem);
        const body = mlir.Block.init(&.{ scalar, scalar }, &.{ self.loc(), self.loc() });
        self.pushBlock(body);
        const element: Value = .{ .inner = body.argument(0), .kernel = self };
        const acc: Value = .{ .inner = body.argument(1), .kernel = self };
        const combined: Value = combine(self, element, acc, ctx);
        _ = ct.yield(self.ctx, &.{combined.inner}, self.loc()).appendTo(body);
        self.popBlock();
        return body;
    }

    pub fn reduce(self: *Builder, ctx: anytype, args: ReduceArgs(@TypeOf(ctx))) Value {
        const result_ty = self.tileTyOf(reducedShape(args.src, args.axis).constSlice(), args.src.elemType());
        const body = self.combineBody(ctx, args.src.elemType(), args.combine);
        return self.emit(ct.reduce(self.ctx, &.{args.src.inner}, @intCast(args.axis), &.{args.identity}, body, &.{result_ty}, self.loc()));
    }

    pub fn scan(self: *Builder, ctx: anytype, args: ScanArgs(@TypeOf(ctx))) Value {
        checkAxis(args.src, args.axis);
        const body = self.combineBody(ctx, args.src.elemType(), args.combine);
        return self.emit(ct.scan(self.ctx, &.{args.src.inner}, @intCast(args.axis), args.reverse, &.{args.identity}, body, &.{args.src.type_()}, self.loc()));
    }

    /// The `2N`-argument body of a variadic `reduce`/`scan`: arguments come in
    /// (element, accumulator) pairs, one pair per source.
    fn combineBodyMulti(self: *Builder, ctx: anytype, srcs: []const Value, combine: anytype) *mlir.Block {
        const scratch = self.arena.allocator();
        const n = srcs.len;
        const types = scratch.alloc(*const mlir.Type, 2 * n) catch @panic("OOM");
        const locs = scratch.alloc(*const mlir.Location, 2 * n) catch @panic("OOM");
        for (srcs, 0..) |s, i| {
            const scalar = ct.tileType(self.ctx, &.{}, s.elemType());
            types[2 * i] = scalar;
            types[2 * i + 1] = scalar;
            locs[2 * i] = self.loc();
            locs[2 * i + 1] = self.loc();
        }
        const body = mlir.Block.init(types, locs);
        self.pushBlock(body);
        const elements = scratch.alloc(Value, n) catch @panic("OOM");
        const accs = scratch.alloc(Value, n) catch @panic("OOM");
        for (0..n) |i| {
            elements[i] = .{ .inner = body.argument(2 * i), .kernel = self };
            accs[i] = .{ .inner = body.argument(2 * i + 1), .kernel = self };
        }
        const out: []const Value = combine(self, elements, accs, ctx);
        if (out.len != n) std.debug.panic("Builder.reduceMulti: combine returned {d} values for {d} sources", .{ out.len, n });
        _ = ct.yield(self.ctx, self.innerSlice(out), self.loc()).appendTo(body);
        self.popBlock();
        return body;
    }

    pub fn reduceMulti(self: *Builder, ctx: anytype, args: ReduceMultiArgs(@TypeOf(ctx))) []Value {
        std.debug.assert(args.srcs.len == args.identities.len);
        const result_types = self.arena.allocator().alloc(*const mlir.Type, args.srcs.len) catch @panic("OOM");
        for (args.srcs, 0..) |s, i| result_types[i] = self.tileTyOf(reducedShape(s, args.axis).constSlice(), s.elemType());
        const body = self.combineBodyMulti(ctx, args.srcs, args.combine);
        return self.emitMulti(ct.reduce(self.ctx, self.innerSlice(args.srcs), @intCast(args.axis), args.identities, body, result_types, self.loc()), args.srcs.len);
    }

    pub fn scanMulti(self: *Builder, ctx: anytype, args: ScanMultiArgs(@TypeOf(ctx))) []Value {
        std.debug.assert(args.srcs.len == args.identities.len);
        const result_types = self.arena.allocator().alloc(*const mlir.Type, args.srcs.len) catch @panic("OOM");
        for (args.srcs, 0..) |s, i| {
            checkAxis(s, args.axis);
            result_types[i] = s.type_();
        }
        const body = self.combineBodyMulti(ctx, args.srcs, args.combine);
        return self.emitMulti(ct.scan(self.ctx, self.innerSlice(args.srcs), @intCast(args.axis), args.reverse, args.identities, body, result_types, self.loc()), args.srcs.len);
    }

    const ReduceKind = enum { sum, max, min, prod };

    fn combineKind(k: *Builder, element: Value, acc: Value, kind: ReduceKind) Value {
        const is_float = element.isFloatElem();
        return switch (kind) {
            .sum => if (is_float) k.addf(element, acc) else k.addi(element, acc),
            .prod => if (is_float) k.mulf(element, acc) else k.muli(element, acc),
            .max => if (is_float) k.maxf(element, acc) else k.maxi(element, acc, .signed),
            .min => if (is_float) k.minf(element, acc) else k.mini(element, acc, .signed),
        };
    }

    fn kindIdentity(self: *Builder, src: Value, kind: ReduceKind) *const mlir.Attribute {
        const dt = self.mlirElemToDType(src.elemType());
        if (isFloatDtype(dt)) {
            return self.identityAttr(dt, switch (kind) {
                .sum => @as(f64, 0),
                .prod => @as(f64, 1),
                .max => -std.math.inf(f64),
                .min => std.math.inf(f64),
            });
        }
        const bits: u32 = intBitwidth(dt);
        const min_int: i64 = if (bits == 64) std.math.minInt(i64) else -(@as(i64, 1) << @intCast(bits - 1));
        const max_int: i64 = if (bits == 64) std.math.maxInt(i64) else (@as(i64, 1) << @intCast(bits - 1)) - 1;
        return self.identityAttr(dt, switch (kind) {
            .sum => @as(i64, 0),
            .prod => @as(i64, 1),
            .max => min_int,
            .min => max_int,
        });
    }

    fn reduceKind(self: *Builder, src: Value, axis: usize, comptime kind: ReduceKind) Value {
        const combine = struct {
            fn c(k: *Builder, element: Value, acc: Value, _: void) Value {
                return combineKind(k, element, acc, kind);
            }
        }.c;
        return self.reduce({}, .{ .src = src, .axis = axis, .identity = self.kindIdentity(src, kind), .combine = combine });
    }

    pub fn sum(self: *Builder, src: Value, axis: usize) Value {
        return self.reduceKind(src, axis, .sum);
    }
    pub fn max(self: *Builder, src: Value, axis: usize) Value {
        return self.reduceKind(src, axis, .max);
    }
    pub fn min(self: *Builder, src: Value, axis: usize) Value {
        return self.reduceKind(src, axis, .min);
    }
    pub fn prod(self: *Builder, src: Value, axis: usize) Value {
        return self.reduceKind(src, axis, .prod);
    }

    fn scanKind(self: *Builder, src: Value, axis: usize, opts: ScanOpts, comptime kind: ReduceKind) Value {
        const combine = struct {
            fn c(k: *Builder, element: Value, acc: Value, _: void) Value {
                return combineKind(k, element, acc, kind);
            }
        }.c;
        return self.scan({}, .{ .src = src, .axis = axis, .reverse = opts.reverse, .identity = self.kindIdentity(src, kind), .combine = combine });
    }

    pub fn cumsum(self: *Builder, src: Value, axis: usize) Value {
        return self.scanKind(src, axis, .{}, .sum);
    }
    pub fn cumsumOpts(self: *Builder, src: Value, axis: usize, opts: ScanOpts) Value {
        return self.scanKind(src, axis, opts, .sum);
    }
    pub fn cumprod(self: *Builder, src: Value, axis: usize) Value {
        return self.scanKind(src, axis, .{}, .prod);
    }
    pub fn cummax(self: *Builder, src: Value, axis: usize) Value {
        return self.scanKind(src, axis, .{}, .max);
    }
    pub fn cummin(self: *Builder, src: Value, axis: usize) Value {
        return self.scanKind(src, axis, .{}, .min);
    }

    /// `for iv in [lower, upper) step` carrying `inits`; close with
    /// `scope.yield(.{...})`, then read `scope.results`.
    pub fn openFor(
        self: *Builder,
        lower: anytype,
        upper: anytype,
        step: anytype,
        inits: anytype,
    ) ForScope(tupleArity(@TypeOf(inits), "openFor: inits")) {
        const N = comptime tupleArity(@TypeOf(inits), "openFor: inits");
        const fields = @typeInfo(@TypeOf(inits)).@"struct".fields;

        // Literals take the type of the first bound that is a Value; all
        // three must agree (`AllTypesMatch` on the op).
        const ref_ty: *const mlir.Type = if (@TypeOf(lower) == Value) lower.type_() else if (@TypeOf(upper) == Value) upper.type_() else if (@TypeOf(step) == Value) step.type_() else self.scalarTy(.i32);
        const lb_v: Value = if (@TypeOf(lower) == Value) lower else self.constMatching(lower, ref_ty);
        const ub_v: Value = if (@TypeOf(upper) == Value) upper else self.constMatching(upper, ref_ty);
        const step_v: Value = if (@TypeOf(step) == Value) step else self.constMatching(step, ref_ty);
        if (!lb_v.type_().eql(ub_v.type_()) or !lb_v.type_().eql(step_v.type_())) {
            std.debug.panic("Builder.openFor: bounds must share one type, got {f}, {f}, {f}", .{ lb_v.type_(), ub_v.type_(), step_v.type_() });
        }

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

    pub fn openIf(self: *Builder, cond: Value) IfOnlyScope {
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
    ) IfScope(tupleArity(@TypeOf(result_types), "openIfElse: result_types")) {
        const N = comptime tupleArity(@TypeOf(result_types), "openIfElse: result_types");
        const fields = @typeInfo(@TypeOf(result_types)).@"struct".fields;

        var types: [N]*const mlir.Type = undefined;
        inline for (fields, 0..) |f, i| {
            if (f.type != *const mlir.Type)
                @compileError("openIfElse: every result_type must be *const mlir.Type (use b.scalarTy/tileTy)");
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

    /// An unstructured loop carrying `inits`; leave with `scope.breakWith`
    /// inside an `if`, and close the body with `scope.yield`.
    pub fn openLoop(self: *Builder, inits: anytype) LoopScope(tupleArity(@TypeOf(inits), "openLoop: inits")) {
        const N = comptime tupleArity(@TypeOf(inits), "openLoop: inits");
        const fields = @typeInfo(@TypeOf(inits)).@"struct".fields;

        var block_types: [N]*const mlir.Type = undefined;
        var block_locs: [N]*const mlir.Location = undefined;
        var inits_inner: [N]*const mlir.Value = undefined;
        inline for (fields, 0..) |f, i| {
            const raw = @field(inits, f.name);
            const v: Value = if (f.type == Value) raw else self.lift(raw);
            block_types[i] = v.type_();
            block_locs[i] = self.loc();
            inits_inner[i] = v.inner;
        }

        const body = mlir.Block.init(&block_types, &block_locs);
        self.pushBlock(body);

        var carried: [N]Value = undefined;
        for (0..N) |i| carried[i] = .{ .inner = body.argument(i), .kernel = self };

        return .{
            .kernel = self,
            .body = body,
            .inits_inner = inits_inner,
            .result_types = block_types,
            .carried = carried,
        };
    }

    pub fn deviceAssert(self: *Builder, condition: Value, message: []const u8) void {
        _ = ct.assert_(self.ctx, condition.inner, message, self.loc()).appendTo(self.currentBlock());
    }

    /// Append `return`, verify the module, and serialize it to a NUL-terminated
    /// string owned by the kernel's allocator. `results` must be empty: an
    /// `entry` returns nothing; outputs reach the host through pointers.
    pub fn finish(self: *Builder, results: []const Value) FinishError![:0]const u8 {
        std.debug.assert(results.len == 0);
        const current = self.currentBlock();
        if (current.terminator() == null) {
            _ = ct.return_(self.ctx, &.{}, self.loc()).appendTo(current);
        }

        if (!self.module.operation().verify()) {
            return error.InvalidMlir;
        }

        var al: std.Io.Writer.Allocating = .init(self.allocator);
        defer al.deinit();

        try al.writer.print("{f}", .{self.module.operation()});

        return try self.allocator.dupeSentinel(u8, al.written(), 0);
    }
};

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (dialects_needed) |d| mlir.DialectHandle.fromString(d).insertDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

fn expectRoundTrip(ctx: *mlir.Context, ir: [:0]const u8) !void {
    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "empty entry round-trips" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "empty");
    defer b.deinit();
    _ = try b.declareArgs(.{ .p = .{ .ptr = .f32 } });

    const ir = try b.finish(&.{});
    defer std.testing.allocator.free(ir);

    try std.testing.expect(std.mem.find(u8, ir, "cuda_tile.module @empty") != null);
    try std.testing.expect(std.mem.find(u8, ir, "entry @empty(") != null);
    try std.testing.expect(std.mem.find(u8, ir, "assume") != null);
    try std.testing.expect(std.mem.find(u8, ir, "div_by<16>") != null);
    try std.testing.expect(std.mem.find(u8, ir, "cuda_tile.entry") == null);
    try std.testing.expect(std.mem.find(u8, ir, "return") != null);
    try expectRoundTrip(ctx, ir);
}

test "add_one over pointer tiles matches the XLA milestone kernel" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "add_one");
    defer b.deinit();
    const a = try b.declareArgsOpts(.{
        .in = .{ .ptr_opts = .{ .dtype = .f32, .div_by = null } },
        .out = .{ .ptr_opts = .{ .dtype = .f32, .div_by = null } },
    }, .{});

    const offsets = b.iota(128, .i32);
    const loaded = b.loadPtrOpts(a.in.offset(offsets), .{});
    const sum = loaded.tile.add(1.0);
    _ = b.storePtrOpts(a.out.offset(offsets), sum, .{ .token = loaded.token });

    const ir = try b.finish(&.{});
    defer std.testing.allocator.free(ir);

    try std.testing.expect(std.mem.find(u8, ir, "iota : tile<128xi32>") != null);
    try std.testing.expect(std.mem.find(u8, ir, "reshape") != null);
    try std.testing.expect(std.mem.find(u8, ir, "broadcast") != null);
    try std.testing.expect(std.mem.find(u8, ir, "offset") != null);
    try std.testing.expect(std.mem.find(u8, ir, "load_ptr_tko weak") != null);
    try std.testing.expect(std.mem.find(u8, ir, "addf") != null);
    try std.testing.expect(std.mem.find(u8, ir, "store_ptr_tko weak") != null);
    try std.testing.expect(std.mem.find(u8, ir, "token = ") != null);
    try expectRoundTrip(ctx, ir);
}

test "vector_add over partition views" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "vector_add");
    defer b.deinit();
    const a = try b.declareArgsOpts(.{
        .x = .{ .ptr = .f32 },
        .y = .{ .ptr = .f32 },
        .out = .{ .ptr = .f32 },
    }, .{ .hints = &.{.{ .arch = .sm_120, .num_worker_warps_per_cta = 4 }} });

    const shape = [_]i64{ 8192, 128 };
    const strides = [_]i64{ 128, 1 };
    const tile = [_]i64{ 64, 64 };
    const vx = b.partitionView(b.tensorView(a.x, &shape, &strides), &tile, .{});
    const vy = b.partitionView(b.tensorView(a.y, &shape, &strides), &tile, .{});
    const vo = b.partitionView(b.tensorView(a.out, &shape, &strides), &tile, .{});

    const bid = b.tileBlockId();
    const idx = [_]Value{ bid.x, bid.y };
    const xt = b.load(vx, &idx);
    const yt = b.load(vy, &idx);
    _ = b.store(xt.add(yt), vo, &idx);

    const ir = try b.finish(&.{});
    defer std.testing.allocator.free(ir);

    try std.testing.expect(std.mem.find(u8, ir, "optimization_hints=<sm_120 = {num_worker_warps_per_cta = 4}>") != null);
    try std.testing.expect(std.mem.find(u8, ir, "make_tensor_view") != null);
    try std.testing.expect(std.mem.find(u8, ir, "partition_view<tile=(64x64), padding_value = zero") != null);
    try std.testing.expect(std.mem.find(u8, ir, "get_tile_block_id") != null);
    try std.testing.expect(std.mem.find(u8, ir, "load_view_tko weak") != null);
    try std.testing.expect(std.mem.find(u8, ir, "rounding<nearest_even>") != null or std.mem.find(u8, ir, "addf") != null);
    try std.testing.expect(std.mem.find(u8, ir, "store_view_tko weak") != null);
    try expectRoundTrip(ctx, ir);
}

test "gemm with a for loop and mmaf" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "gemm");
    defer b.deinit();
    const a = try b.declareArgs(.{ .a = .{ .ptr = .bf16 }, .bm = .{ .ptr = .bf16 }, .c = .{ .ptr = .f32 } });

    const m: i64 = 1024;
    const n: i64 = 1024;
    const k: i64 = 512;
    const pa = b.partitionView(b.tensorView(a.a, &.{ m, k }, &.{ k, 1 }), &.{ 128, 64 }, .{});
    const pb = b.partitionView(b.tensorView(a.bm, &.{ k, n }, &.{ n, 1 }), &.{ 64, 128 }, .{});
    const pc = b.partitionView(b.tensorView(a.c, &.{ m, n }, &.{ n, 1 }), &.{ 128, 128 }, .{});

    const bid = b.tileBlockId();
    const acc0 = b.zeros(&.{ 128, 128 }, .f32);

    var loop = b.openFor(0, @divExact(k, 64), 1, .{acc0});
    {
        const kt = loop.iv;
        const acc = loop.carried[0];
        const at = b.load(pa, &.{ bid.x, kt });
        const bt = b.load(pb, &.{ kt, bid.y });
        loop.yield(.{b.mmaf(at, bt, acc)});
    }
    _ = b.store(loop.results[0], pc, &.{ bid.x, bid.y });

    const ir = try b.finish(&.{});
    defer std.testing.allocator.free(ir);

    try std.testing.expect(std.mem.find(u8, ir, "for ") != null);
    try std.testing.expect(std.mem.find(u8, ir, "iter_values") != null);
    try std.testing.expect(std.mem.find(u8, ir, "mmaf") != null);
    try std.testing.expect(std.mem.find(u8, ir, "continue ") != null);
    try expectRoundTrip(ctx, ir);
}

test "if, if-else, loop with break, reduce, scan, cast, select" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "control");
    defer b.deinit();
    // A scalar reaches the kernel through a rank-0 pointer.
    const a = try b.declareArgs(.{ .x = .{ .ptr = .f32 }, .n = .{ .ptr = .i32 } });
    const n = b.loadPtr(a.n);
    try std.testing.expect(n.isScalar());

    const v = b.partitionView(b.tensorView(a.x, &.{ 256, 64 }, &.{ 64, 1 }), &.{ 8, 64 }, .{});
    const bid = b.tileBlockId();
    const t = b.load(v, &.{ bid.x, bid.y });

    // Reductions and scans over the tile.
    const row_sum = b.sum(t, 1);
    const col_max = b.max(t, 0);
    const running = b.cumsum(t, 1);
    try std.testing.expectEqual(@as(usize, 1), row_sum.rank());
    try std.testing.expectEqual(@as(i64, 8), row_sum.dim(0));
    try std.testing.expectEqual(@as(i64, 64), col_max.dim(0));
    try std.testing.expectEqual(@as(usize, 2), running.rank());

    // A variadic reduce: argmax over axis 1 as (max, index).
    const cols = b.broadcastTo(b.reshape(b.iota(64, .i32), &.{ 1, 64 }), &.{ 8, 64 });
    const argmax = b.reduceMulti({}, .{
        .srcs = &.{ t, cols },
        .axis = 1,
        .identities = &.{ b.identityAttr(.f32, -std.math.inf(f64)), b.identityAttr(.i32, 0) },
        .combine = struct {
            fn c(k: *Builder, elements: []const Value, accs: []const Value, _: void) []const Value {
                const better = elements[0].gt(accs[0]);
                return k.yield(.{ k.select(better, elements[0], accs[0]), k.select(better, elements[1], accs[1]) });
            }
        }.c,
    });
    try std.testing.expectEqual(@as(usize, 2), argmax.len);
    try std.testing.expectEqual(@as(i64, 8), argmax[1].dim(0));

    // Casts and select.
    const as_i32 = t.to(.i32);
    const back = as_i32.to(.f32);
    const picked = b.where(t.gt(0.0), back, t.neg());

    // if / if-else carrying a value.
    const is_first = bid.x.eq(0);
    var only = b.openIf(is_first);
    _ = b.store(picked, v, &.{ bid.x, bid.y });
    only.yieldThen(.{});

    var branch = b.openIfElse(is_first, .{b.tileTy(&.{ 8, 64 }, .f32)});
    branch.yieldThen(.{picked});
    branch.yieldElse(.{t});
    const chosen = branch.results[0];

    // loop: count up to n, break when reached.
    var lp = b.openLoop(.{b.cst(.i32, 0)});
    {
        const i = lp.carried[0];
        var done = b.openIf(i.ge(n));
        lp.breakWith(.{i});
        done.yieldThen(.{});
        lp.yield(.{i.add(1)});
    }
    _ = lp.results[0];

    // for with an early `continue` inside an `if`, and i64 bounds from a Value.
    var skip = b.openFor(b.cst(.i64, 0), 16, 1, .{b.cst(.i64, 0)});
    {
        const acc = skip.carried[0];
        var odd = b.openIf(skip.iv.bitAnd(1).eq(1));
        skip.continueWith(.{acc});
        odd.yieldThen(.{});
        skip.yield(.{acc.add(skip.iv)});
    }
    try std.testing.expect(skip.results[0].isScalar());

    _ = b.store(chosen, v, &.{ bid.x, bid.y });

    const ir = try b.finish(&.{});
    defer std.testing.allocator.free(ir);

    try std.testing.expect(std.mem.find(u8, ir, "reduce ") != null);
    try std.testing.expect(std.mem.find(u8, ir, "scan ") != null);
    try std.testing.expect(std.mem.find(u8, ir, "ftoi") != null);
    try std.testing.expect(std.mem.find(u8, ir, "itof") != null);
    try std.testing.expect(std.mem.find(u8, ir, "select") != null);
    try std.testing.expect(std.mem.find(u8, ir, "loop ") != null);
    try std.testing.expect(std.mem.find(u8, ir, "break ") != null);
    try std.testing.expect(std.mem.find(u8, ir, "yield") != null);
    try expectRoundTrip(ctx, ir);
}

test "atomics, print, globals, alloca, dynamic views, strided view, tokens" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "misc");
    defer b.deinit();
    const a = try b.declareArgs(.{ .x = .{ .ptr = .f32 }, .cnt = .{ .ptr = .i32 }, .rows = .{ .ptr = .i32 } });
    b.global("lut", &.{16}, .f32, 1.5, .{ .constant = true });

    const rows = b.loadPtr(a.rows);
    const tv = b.tensorViewDyn(a.x, &.{ .{ .dynamic = rows }, .{ .static = 64 } }, &.{ .{ .static = 64 }, .{ .static = 1 } });
    const extents = b.tensorShape(tv);
    try std.testing.expectEqual(@as(usize, 2), extents.len);
    const v = b.partitionView(tv, &.{ 8, 64 }, .{ .padding = .nan });
    const tiles = b.indexSpaceShape(v);
    try std.testing.expectEqual(@as(usize, 2), tiles.len);
    const sv = b.stridedView(tv, &.{ 8, 64 }, &.{ 2, 1 }, .{});
    _ = b.load(sv, &.{ b.cst(.i32, 0), b.cst(.i32, 0) });

    const bid = b.tileBlockId();
    const loaded = b.loadOpts(v, &.{ bid.x, bid.y }, .{ .hints = &.{.{ .arch = .sm_120, .allow_tma = true }} });
    const tok = b.print("tile %f\n", &.{loaded.tile});

    const counter = a.cnt.offset(b.iota(8, .i32));
    const old = b.atomicRmw(.add, counter, b.full(&.{8}, 1, .i32), .{ .token = tok });
    const cas = b.atomicCas(counter, old.tile, b.full(&.{8}, 0, .i32), .{ .token = old.token });
    const joined = b.joinTokens(&.{ loaded.token, cas.token, b.makeToken() });
    // Atomic reductions refuse a padded view; the default options must verify.
    const unpadded = b.partitionView(tv, &.{ 8, 64 }, .{ .padding = null });
    _ = b.atomicRedView(.addf, unpadded, &.{ bid.x, bid.y }, loaded.tile, .{ .token = joined });
    _ = b.atomicRedView(.addf, unpadded, &.{ bid.x, bid.y }, loaded.tile, .{});

    // A gather/scatter view: the sparse index is a rank-1 tile of row ids.
    const gsv = b.gatherScatterView(tv, &.{ 8, 64 }, 0, .zero);
    const gathered = b.load(gsv, &.{ b.iota(8, .i32), b.cst(.i32, 0) });
    try std.testing.expectEqual(@as(i64, 8), gathered.dim(0));
    try std.testing.expectEqual(@as(i64, 64), gathered.dim(1));
    _ = b.store(gathered, gsv, &.{ b.iota(8, .i32), b.cst(.i32, 0) });

    const lut = b.getGlobal("lut", .f32);
    const scratch = b.alloca(.f32, 64, .{});
    _ = b.loadPtr(lut.offset(b.iota(16, .i32)));
    _ = b.storePtr(scratch.offset(b.iota(64, .i32)), b.zeros(&.{64}, .f32));

    const ir = try b.finish(&.{});
    defer std.testing.allocator.free(ir);

    try std.testing.expect(std.mem.find(u8, ir, "global @lut") != null);
    try std.testing.expect(std.mem.find(u8, ir, "get_global @lut") != null);
    try std.testing.expect(std.mem.find(u8, ir, "tensor_view<?x64xf32") != null);
    try std.testing.expect(std.mem.find(u8, ir, "get_tensor_shape") != null);
    try std.testing.expect(std.mem.find(u8, ir, "get_index_space_shape") != null);
    try std.testing.expect(std.mem.find(u8, ir, "strided_view") != null);
    try std.testing.expect(std.mem.find(u8, ir, "allow_tma = true") != null);
    try std.testing.expect(std.mem.find(u8, ir, "print_tko") != null);
    try std.testing.expect(std.mem.find(u8, ir, "atomic_rmw_tko") != null);
    try std.testing.expect(std.mem.find(u8, ir, "atomic_cas_tko") != null);
    try std.testing.expect(std.mem.find(u8, ir, "atomic_red_view_tko") != null);
    try std.testing.expect(std.mem.find(u8, ir, "join_tokens") != null);
    try std.testing.expect(std.mem.find(u8, ir, "alloca") != null);
    try std.testing.expect(std.mem.find(u8, ir, "gather_scatter_view") != null);
    try expectRoundTrip(ctx, ir);
}

test "Builder.init hands out values that outlive its frame" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.init(std.testing.allocator, ctx, "via_init", &.{
        .{ .name = "x", .kind = .{ .ptr = .f32 } },
    });
    defer b.deinit();
    // `arg(0)` is the assume-refined value cached by a Builder that has since
    // moved; it must still resolve to this one.
    const p = b.arg(0).offset(b.iota(16, .i32));
    _ = b.loadPtr(p);

    const ir = try b.finish(&.{});
    defer std.testing.allocator.free(ir);
    try expectRoundTrip(ctx, ir);
}

test "every remaining elementwise op verifies" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "elementwise");
    defer b.deinit();
    _ = try b.declareArgs(.{ .p = .{ .ptr = .f32 } });

    const f = b.full(&.{ 4, 8 }, 0.5, .f32);
    const g = b.full(&.{ 4, 8 }, 2.0, .f32);
    const i = b.full(&.{ 4, 8 }, 3, .i32);
    const j = b.full(&.{ 4, 8 }, 5, .i32);

    _ = b.subf(f, g);
    _ = b.mulfOpts(f, g, .{ .rounding = .zero, .flush_to_zero = true });
    _ = b.divf(f, g);
    _ = b.remf(f, g);
    _ = b.fma(f, g, f);
    _ = b.maxfOpts(f, g, .{ .propagate_nan = true });
    _ = b.minf(f, g);
    _ = b.pow(f, g);
    _ = b.atan2(f, g);
    _ = b.exp(f);
    _ = b.exp2(f);
    _ = b.log(f);
    _ = b.log2(f);
    _ = b.sqrt(f);
    _ = b.rsqrt(f);
    _ = b.sin(f);
    _ = b.cos(f);
    _ = b.tan(f);
    _ = b.sinh(f);
    _ = b.cosh(f);
    _ = b.tanh(f);
    _ = b.ceil(f);
    _ = b.floor(f);
    _ = b.absf(f);
    _ = b.negf(f);
    _ = b.cmpfOpts(.less_than, f, g, .unordered);

    _ = b.addiOpts(i, j, .{ .overflow = .no_signed_wrap });
    _ = b.subi(i, j);
    _ = b.muli(i, j);
    _ = b.mulhii(i, j);
    _ = b.diviOpts(i, j, .{ .signedness = .signed, .rounding = .negative_inf });
    _ = b.divi(i, j, .unsigned);
    _ = b.remi(i, j, .signed);
    _ = b.maxi(i, j, .unsigned);
    _ = b.mini(i, j, .signed);
    _ = b.negi(i);
    _ = b.absi(i);
    _ = b.shli(i, j);
    _ = b.shri(i, j, .unsigned);
    _ = b.andi(i, j);
    _ = b.ori(i, j);
    _ = b.xori(i, j);
    _ = b.cmpi(.not_equal, i, j, .unsigned);

    _ = b.bitcast(f, .i32);
    _ = b.exti(i, .i64, .unsigned);
    _ = b.trunci(i, .i16);
    _ = b.ftof(f, .bf16);
    _ = b.ftoi(f, .i32);
    _ = b.ftoiOpts(f, .i32, .{ .signedness = .unsigned });
    _ = b.itofOpts(i, .f16, .{ .signedness = .unsigned });
    // The MX scale type: only `zero`/`positive_inf` rounding reach it, and an
    // integer goes through f32.
    _ = b.ftof(f, .f8e8m0fnu);
    _ = i.to(.f8e8m0fnu);
    // i64 reductions (the identity is the full 64-bit range).
    _ = b.sum(b.full(&.{ 4, 8 }, 0, .i64), 1);
    _ = b.max(b.iota(128, .i64), 0);
    _ = b.min(b.full(&.{ 4, 8 }, 0, .i64), 0);
    _ = i.cdiv(3);
    const ints = b.full(&.{4}, 0, .i64);
    const ptrs = b.intToPtr(ints, .f32);
    _ = b.ptrToInt(ptrs);
    _ = b.ptrToPtr(ptrs, .i8);

    _ = b.cat(f, g, 0);
    _ = b.permute(f, &.{ 1, 0 });
    _ = b.transpose(f);
    _ = b.extract(f, &.{ b.cst(.i32, 0), b.cst(.i32, 1) }, &.{ 4, 4 });
    _ = b.expandDims(f, 0);
    _ = b.reshape(f, &.{32});
    _ = b.cumprod(f, 0);
    _ = b.cummax(f, 1);
    _ = b.cummin(f, 1);
    _ = b.prod(f, 0);
    _ = b.min(i, 1);
    _ = b.dense(&.{4}, .i32, &[_]i32{ 1, 2, 3, 4 });
    b.deviceAssert(i.lt(j), "i < j");
    _ = b.assumeBounded(i, 0, 100);
    _ = b.assumeSameElements(i, &.{ 1, 1 });

    const acc = b.zeros(&.{ 64, 64 }, .i32);
    const a8 = b.full(&.{ 64, 32 }, 1, .i8);
    const b8 = b.full(&.{ 32, 64 }, 1, .i8);
    _ = b.mmai(a8, b8, acc, .signed, .unsigned);

    const ir = try b.finish(&.{});
    defer std.testing.allocator.free(ir);
    try std.testing.expect(std.mem.find(u8, ir, "rounding<zero>") != null);
    try expectRoundTrip(ctx, ir);
}
