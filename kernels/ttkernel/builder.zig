//! TTKernel DSL: Tenstorrent thread kernels written against the `ttkernel`
//! dialect and the host program that launches them, after tt-mlir's
//! pykernel (`tools/pykernel`). Kernels are translated to tt-metal C++ here
//! and shipped in a `program.zig` artifact.
//!
//! | pykernel                                   | here                                                |
//! |--------------------------------------------|-----------------------------------------------------|
//! | `@reader_thread() def f(cb: CircularBuffer, addr, n)` | `const f = readerThread("f", fBody);` with `fn fBody(k: *Builder, cb: CircularBuffer, addr: Value, n: Value) void` |
//! | `x: CompileTimeValue`                      | a Zig integer parameter                             |
//! | `cb_reserve_back(cb, 1)`                   | `k.op(.cb_reserve_back, .{ cb, 1 })`                |
//! | `n = get_tile_size(cb)`                    | `const n = k.op(.get_tile_size, .{cb})`             |
//! | `for i in range(a, b, s):`                 | `var loop = k.openFor(a, b, s, .{}); ... loop.yield(.{});` |
//! | `if c:`                                    | `var s = k.openIf(c); ... s.yieldThen(.{});`        |
//! | `ii: int = start` / `ii += 1`              | `const ii = k.variable(start);` / `ii.set(ii.get().add(1))` |
//! | `TensorAccessorArgs(cta_base=1, crta_base=0)` | `k.tensorAccessorArgs(1, 0)`                     |
//! | `PyKernelOp.create_cb(t, 0)`               | `p.createCb(t, 0, .{})`                             |
//! | `create_kernel(f, cb, addr, n, tensor_accessor_configs=[t])` | `p.createKernel(f, .{ cb, t.address(), n }, .{ .tensor_accessors = &.{t} })` |
//!
//! As in pykernel, a kernel's circular-buffer parameters become its first
//! compile-time arguments (in parameter order), followed by the tensor
//! accessor arguments of `tensor_accessors`; integer and address host
//! arguments of `Value` parameters become common runtime arguments and
//! `PerCore` ones per-core runtime arguments, each counted in parameter
//! order. Integer literals are `i32`, loop induction variables `index`.

const std = @import("std");

const dialects = @import("mlir/dialects");
const arith = dialects.arith;
const func = dialects.func;
const memref = dialects.memref;
const dsl = @import("kernels/common");
const mlir = @import("mlir");
const ttkernel = @import("mlir/dialects/ttkernel");

pub const program = @import("kernels/ttkernel/program");

pub const ops = ttkernel.ops;
pub const ttcore = ttkernel.ttcore;
pub const ReduceType = ttkernel.ReduceType;
pub const ReduceDim = ttkernel.ReduceDim;
pub const BcastType = ttkernel.BcastType;

pub const FinishError = error{ InvalidMlir, OutOfMemory };

/// A context with the dialects a TTKernel kernel uses. Caller owns it.
pub fn newContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    ttkernel.insertDialects(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Builder);
    std.testing.refAllDecls(Value);
    std.testing.refAllDecls(Program);
}

// =============================================================================
// Kernel side
// =============================================================================

/// An SSA value of a kernel: integers (`i32` unless stated), `index` loop
/// variables, addresses, accessors.
pub const Value = struct {
    inner: *const mlir.Value,
    kernel: ?*Builder = null,

    pub fn type_(self: Value) *const mlir.Type {
        return self.inner.type_();
    }

    fn kern(self: Value) *Builder {
        return self.kernel orelse @panic("ttkernel.Value has no owning Builder");
    }

    fn binary(self: Value, comptime build: anytype, rhs: anytype) Value {
        const k = self.kern();
        const r = k.liftAs(rhs, self.type_());
        return k.emit(build(k.ctx, self.inner, r.inner, k.loc()));
    }

    pub fn add(self: Value, rhs: anytype) Value {
        return self.binary(arith.addi, rhs);
    }

    pub fn sub(self: Value, rhs: anytype) Value {
        return self.binary(arith.subi, rhs);
    }

    pub fn mul(self: Value, rhs: anytype) Value {
        return self.binary(arith.muli, rhs);
    }

    /// Signed division, Python's `//` on non-negative operands.
    pub fn div(self: Value, rhs: anytype) Value {
        return self.binary(arith.divsi, rhs);
    }

    pub fn rem(self: Value, rhs: anytype) Value {
        return self.binary(arith.remsi, rhs);
    }

    pub fn shl(self: Value, rhs: anytype) Value {
        return self.binary(arith.shli, rhs);
    }

    pub fn shr(self: Value, rhs: anytype) Value {
        return self.binary(arith.shrsi, rhs);
    }

    pub fn bitAnd(self: Value, rhs: anytype) Value {
        return self.binary(arith.andi, rhs);
    }

    pub fn bitOr(self: Value, rhs: anytype) Value {
        return self.binary(arith.ori, rhs);
    }

    pub fn bitXor(self: Value, rhs: anytype) Value {
        return self.binary(arith.xori, rhs);
    }

    pub fn min(self: Value, rhs: anytype) Value {
        return self.binary(arith.minsi, rhs);
    }

    pub fn max(self: Value, rhs: anytype) Value {
        return self.binary(arith.maxsi, rhs);
    }

    fn cmp(self: Value, predicate: arith.CmpIPredicate, rhs: anytype) Value {
        const k = self.kern();
        const r = k.liftAs(rhs, self.type_());
        return k.emit(arith.cmpi(k.ctx, predicate, self.inner, r.inner, k.loc()));
    }

    pub fn lt(self: Value, rhs: anytype) Value {
        return self.cmp(.slt, rhs);
    }

    pub fn le(self: Value, rhs: anytype) Value {
        return self.cmp(.sle, rhs);
    }

    pub fn gt(self: Value, rhs: anytype) Value {
        return self.cmp(.sgt, rhs);
    }

    pub fn ge(self: Value, rhs: anytype) Value {
        return self.cmp(.sge, rhs);
    }

    pub fn eq(self: Value, rhs: anytype) Value {
        return self.cmp(.eq, rhs);
    }

    pub fn ne(self: Value, rhs: anytype) Value {
        return self.cmp(.ne, rhs);
    }

    /// This integer as an `index` (`scf.for` bounds, memref indices).
    pub fn toIndex(self: Value) Value {
        const k = self.kern();
        if (self.type_().isA(mlir.IndexType) != null) return self;
        return k.emit(arith.index_cast(k.ctx, self.inner, .index(k.ctx), k.loc()));
    }

    /// This `index` or integer as an `i32`.
    pub fn toI32(self: Value) Value {
        const k = self.kern();
        const i32_type: *const mlir.Type = .int(k.ctx, .i32);
        if (self.type_().eql(i32_type)) return self;
        if (self.type_().isA(mlir.IndexType) != null) return k.emit(arith.index_cast(k.ctx, self.inner, i32_type, k.loc()));
        const width = self.type_().isA(mlir.IntegerType).?.width();
        return k.emit(if (width < 32) arith.extsi(k.ctx, self.inner, i32_type, k.loc()) else arith.trunci(k.ctx, self.inner, i32_type, k.loc()));
    }
};

/// A kernel parameter bound to a circular buffer: the next compile-time
/// argument, `!ttkernel.cb<pages, tile>` (pykernel's `CircularBuffer`).
pub const CircularBuffer = struct {
    inner: *const mlir.Value,
    kernel: ?*Builder = null,
    /// Picked at runtime (`Builder.selectCb`): tt-lang lowers circular
    /// buffer methods (`wait_front`, ...) on compile-time arguments only, so
    /// these take tt-metal's free functions (`cb_wait_front(id, n)`, ...).
    selected: bool = false,

    pub fn value(self: CircularBuffer) Value {
        return .{ .inner = self.inner, .kernel = self.kernel };
    }
};

/// A mutable `i32` of a kernel (pykernel's annotated `x: int`), kept in a
/// one-element memref.
pub const Variable = struct {
    mem: *const mlir.Value,
    kernel: *Builder,

    pub fn get(self: Variable) Value {
        const k = self.kernel;
        return k.emit(memref.load(k.ctx, self.mem, &.{k.index(0).inner}, .{}, k.loc()));
    }

    pub fn set(self: Variable, v: anytype) void {
        const k = self.kernel;
        const val = k.lift(v);
        k.emitNone(memref.store(k.ctx, val.inner, self.mem, &.{k.index(0).inner}, .{}, k.loc()));
    }
};

pub fn ForScope(comptime N: usize) type {
    return dsl.ForScope(Builder, Value, N);
}

pub const IfOnlyScope = dsl.IfOnlyScope(Builder, Value);

pub fn IfScope(comptime N: usize) type {
    return dsl.IfScope(Builder, Value, N);
}

/// A C++ expression written as is among a call's arguments (`Builder.call`),
/// for a parameter no value can stand for, such as an enumerator.
pub const Verbatim = struct {
    text: []const u8,
};

pub fn verbatim(text: []const u8) Verbatim {
    return .{ .text = text };
}

/// What `Builder.op(name, ...)` returns: nothing, a value, or several.
pub fn OpResult(comptime name: @EnumLiteral()) type {
    return switch (@field(ops.num_results, @tagName(name))) {
        0 => void,
        1 => Value,
        else => []const Value,
    };
}

/// The NoC operations whose omitted `noc` operand pykernel sets to NoC 0.
const noc_defaulted = [_][]const u8{
    "noc_async_read_tile",
    "noc_async_write_tile",
    "noc_async_read",
    "noc_async_write",
    "noc_async_read_barrier",
    "noc_async_write_barrier",
};

fn isNocDefaulted(comptime name: []const u8) bool {
    for (noc_defaulted) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

/// One thread kernel being built: a `func.func` with a `ttkernel.thread`
/// attribute and no arguments, in its own module.
pub const Builder = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    ctx: *mlir.Context,
    module: *mlir.Module,
    name: []const u8,
    func_op: *mlir.Operation,
    entry_block: *mlir.Block,
    block_stack: std.ArrayList(*mlir.Block) = .empty,
    /// `ttkernel.arg_spec` compile-time entries, one per circular buffer.
    ct_arg_spec: std.ArrayList(*const mlir.Attribute) = .empty,
    finished: bool = false,
    /// The NoC of the kernel's transfers: the data movement processor's own
    /// (tt-metal dedicates NoC 0 to the reader, RISC-V 1, and NoC 1 to the
    /// writer, RISC-V 0). pykernel's NoC 0 by default.
    default_noc: u1 = 0,

    pub fn open(allocator: std.mem.Allocator, ctx: *mlir.Context, name: []const u8, thread: ttkernel.ThreadType) !Builder {
        const unknown_loc: *const mlir.Location = .unknown(ctx);
        const module: *mlir.Module = .init(unknown_loc);
        errdefer module.deinit();
        const entry = mlir.Block.init(&.{}, &.{});
        const func_op = func.func(ctx, .{
            .name = name,
            .args = &.{},
            .results = &.{},
            .block = entry,
            .location = unknown_loc,
            .extra_attributes = &.{.named(ctx, "ttkernel.thread", ttkernel.threadAttr(ctx, thread))},
            // Verified once terminated, by `finish`.
            .verify = false,
        });
        _ = func_op.appendTo(module.body());
        return .{
            .allocator = allocator,
            .arena = .init(allocator),
            .ctx = ctx,
            .module = module,
            .name = name,
            .func_op = func_op,
            .entry_block = entry,
        };
    }

    pub fn deinit(self: *Builder) void {
        self.module.deinit();
        self.arena.deinit();
    }

    pub fn loc(self: *const Builder) *const mlir.Location {
        return .unknown(self.ctx);
    }

    pub fn pushBlock(self: *Builder, b: *mlir.Block) void {
        self.block_stack.append(self.arena.allocator(), b) catch @panic("ttkernel.Builder.pushBlock OOM");
    }

    pub fn popBlock(self: *Builder) void {
        _ = self.block_stack.pop();
    }

    pub fn currentBlock(self: *Builder) *mlir.Block {
        if (self.block_stack.items.len > 0) return self.block_stack.items[self.block_stack.items.len - 1];
        return self.entry_block;
    }

    /// Appends `op` and returns its first result.
    pub fn emit(self: *Builder, op_: *mlir.Operation) Value {
        _ = op_.appendTo(self.currentBlock());
        return .{ .inner = op_.result(0), .kernel = self };
    }

    pub fn emitNone(self: *Builder, op_: *mlir.Operation) void {
        _ = op_.appendTo(self.currentBlock());
    }

    pub fn wrap(self: *Builder, v: *const mlir.Value) Value {
        return .{ .inner = v, .kernel = self };
    }

    // ------------------------------------------------------------ constants

    /// An `i32` constant, what pykernel makes of an integer literal.
    pub fn int(self: *Builder, v: i64) Value {
        return self.constant(v, .int(self.ctx, .i32));
    }

    pub fn constant(self: *Builder, v: i64, type_: *const mlir.Type) Value {
        if (type_.isA(mlir.IndexType) != null) return self.index(v);
        return self.emit(arith.constant_int(self.ctx, v, type_, self.loc()));
    }

    pub fn index(self: *Builder, v: i64) Value {
        return self.emit(arith.constant_index(self.ctx, v, self.loc()));
    }

    /// `x` as a value: values pass through, integers become `i32` constants
    /// and booleans `i1` ones.
    pub fn lift(self: *Builder, x: anytype) Value {
        return self.liftAs(x, .int(self.ctx, .i32));
    }

    fn liftAs(self: *Builder, x: anytype, type_: *const mlir.Type) Value {
        const T = @TypeOf(x);
        if (T == Value) return x;
        if (T == CircularBuffer) return x.value();
        if (T == *const mlir.Value) return self.wrap(x);
        if (T == bool) return self.constant(@intFromBool(x), .int(self.ctx, .i1));
        return switch (@typeInfo(T)) {
            .comptime_int, .int => self.constant(@intCast(x), type_),
            .@"enum" => self.constant(@intFromEnum(x), type_),
            else => @compileError("ttkernel: cannot use a " ++ @typeName(T) ++ " as a value"),
        };
    }

    // ------------------------------------------------------------ arguments

    /// Compile-time argument `ct_index`, a circular buffer of `cb_type`
    /// (`!ttkernel.cb<...>`).
    pub fn circularBuffer(self: *Builder, ct_index: u32, cb_type: *const mlir.Type) CircularBuffer {
        const v = self.emit(ops.get_compile_time_arg_val(self.ctx, cb_type, .int(self.ctx, .i32, ct_index), self.loc()));
        self.ct_arg_spec.append(self.arena.allocator(), ttkernel.argAttr(self.ctx, .cb_port, ct_index, false)) catch @panic("OOM");
        return .{ .inner = v.inner, .kernel = self };
    }

    /// Compile-time argument `ct_index`, a circular buffer of `pages` tiles
    /// of `dtype`.
    pub fn circularBufferOf(self: *Builder, ct_index: u32, dtype: program.DataType, pages: u32) CircularBuffer {
        const tile = ttcore.TileType.get(self.ctx, 32, 32, tileDataType(dtype));
        return self.circularBuffer(ct_index, ttkernel.CBType.get(self.ctx, pages, tile.type_()).type_());
    }

    /// `a` if `cond`, else `b` (`arith.select`), integers of the type of `a`.
    pub fn select(self: *Builder, cond: anytype, a: anytype, b: anytype) Value {
        const c = self.condition(cond);
        const av = self.lift(a);
        const bv = self.liftAs(b, av.type_());
        return self.emit(arith.select(self.ctx, c.inner, av.inner, bv.inner, self.loc()));
    }

    /// `a` if `cond`, else `b`: circular buffers picked at runtime.
    pub fn selectCb(self: *Builder, cond: anytype, a: CircularBuffer, b: CircularBuffer) CircularBuffer {
        const c = self.condition(cond);
        const v = self.emit(arith.select(self.ctx, c.inner, a.inner, b.inner, self.loc()));
        return .{ .inner = v.inner, .kernel = self, .selected = true };
    }

    /// Compile-time argument `i` as an `i32` (`get_compile_time_arg_val(i)`).
    pub fn compileTimeArg(self: *Builder, i: u32) Value {
        return self.emit(ops.get_compile_time_arg_val(self.ctx, .int(self.ctx, .i32), .int(self.ctx, .i32, i), self.loc()));
    }

    /// NoC `n`, for the transfers whose NoC is explicit.
    pub fn noc(self: *Builder, n: u1) Value {
        return self.constant(n, .int(self.ctx, .i8));
    }

    /// The kernel's NoC (`default_noc`).
    pub fn kernelNoc(self: *Builder) Value {
        return self.noc(self.default_noc);
    }

    /// The L1 address `addr` as a pointer to `width`-bit words
    /// (`reinterpret_cast<volatile tt_l1_ptr uintN_t*>(addr)`).
    pub fn l1Ptr(self: *Builder, addr: Value, comptime width: u8) Value {
        const t = ttkernel.parseType(self.ctx, std.fmt.comptimePrint("!ttkernel.l1_addr_ptr<{d}>", .{width}));
        return self.emit(ops.reinterpret_cast(self.ctx, addr.inner, t, self.loc()));
    }

    /// Word `offset` of the L1 pointer `ptr` of `width`-bit words, read
    /// from memory (the L1 cache invalidated), widened to an `i32`.
    pub fn loadL1(self: *Builder, ptr: Value, offset: anytype, comptime width: u8) Value {
        const word: *const mlir.Type = .int(self.ctx, switch (width) {
            8 => .i8,
            16 => .i16,
            32 => .i32,
            else => @compileError("ttkernel.loadL1: 8, 16 or 32-bit words"),
        });
        const v = self.emit(ops.load_from_l1(self.ctx, ptr.inner, self.lift(offset).inner, word, self.loc()));
        return if (width == 32) v else self.emit(arith.extui(self.ctx, v.inner, .int(self.ctx, .i32), self.loc()));
    }

    /// Stores `value` at word `offset` of the L1 pointer `ptr` of
    /// `width`-bit words, the value narrowed to the word.
    pub fn storeL1(self: *Builder, ptr: Value, offset: anytype, value: anytype, comptime width: u8) void {
        const v = self.lift(value);
        const word = if (width == 32) v else self.emit(arith.trunci(self.ctx, v.inner, .int(self.ctx, switch (width) {
            8 => .i8,
            16 => .i16,
            else => @compileError("ttkernel.storeL1: 8, 16 or 32-bit words"),
        }), self.loc()));
        self.emitNone(ops.store_to_l1(self.ctx, word.inner, ptr.inner, self.lift(offset).inner, self.loc()));
    }

    /// Per-core runtime argument `i` (`get_arg_val<uint32_t>(i)`).
    pub fn runtimeArg(self: *Builder, i: u32) Value {
        return self.emit(ops.get_arg_val(self.ctx, self.index(i).inner, .int(self.ctx, .i32), self.loc()));
    }

    /// Per-core runtime argument at index `i`, a value.
    pub fn runtimeArgAt(self: *Builder, i: Value) Value {
        return self.emit(ops.get_arg_val(self.ctx, i.toIndex().inner, .int(self.ctx, .i32), self.loc()));
    }

    /// Common runtime argument `i` (`get_common_arg_val<uint32_t>(i)`).
    pub fn commonRuntimeArg(self: *Builder, i: u32) Value {
        return self.emit(ops.get_common_arg_val(self.ctx, self.index(i).inner, .int(self.ctx, .i32), self.loc()));
    }

    // ------------------------------------------------------------ control flow

    /// `scf.for` over `[lower, upper)` by `step`, all cast to `index` like
    /// pykernel's `range`. `inits` are loop-carried values.
    pub fn openFor(self: *Builder, lower: anytype, upper: anytype, step: anytype, inits: anytype) ForScope(dsl.tupleArity(@TypeOf(inits), "openFor: inits")) {
        const N = comptime dsl.tupleArity(@TypeOf(inits), "openFor: inits");
        const index_type: *const mlir.Type = .index(self.ctx);
        const lb = self.liftAs(lower, index_type).toIndex();
        const ub = self.liftAs(upper, index_type).toIndex();
        const st = self.liftAs(step, index_type).toIndex();

        var block_types: [N + 1]*const mlir.Type = undefined;
        var block_locs: [N + 1]*const mlir.Location = undefined;
        block_types[0] = index_type;
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
        for (0..N) |i| carried[i] = self.wrap(body.argument(i + 1));
        return .{
            .kernel = self,
            .body = body,
            .lb_inner = lb.inner,
            .ub_inner = ub.inner,
            .step_inner = st.inner,
            .inits_inner = inits_inner,
            .iv = self.wrap(body.argument(0)),
            .carried = carried,
        };
    }

    /// `scf.while` carrying `inits`: the condition block first
    /// (`yieldBefore(cond, forwarded)`), then the body over `after_types`
    /// (`yieldAfter(next)`).
    pub fn openWhile(self: *Builder, inits: anytype, after_types: anytype) dsl.WhileScope(
        Builder,
        Value,
        dsl.tupleArity(@TypeOf(inits), "openWhile: inits"),
        dsl.tupleArity(@TypeOf(after_types), "openWhile: after_types"),
    ) {
        const N = comptime dsl.tupleArity(@TypeOf(inits), "openWhile: inits");
        const M = comptime dsl.tupleArity(@TypeOf(after_types), "openWhile: after_types");
        var before_types: [N]*const mlir.Type = undefined;
        var before_locs: [N]*const mlir.Location = undefined;
        var inits_inner: [N]*const mlir.Value = undefined;
        inline for (@typeInfo(@TypeOf(inits)).@"struct".fields, 0..) |f, i| {
            const v = self.lift(@field(inits, f.name));
            before_types[i] = v.type_();
            before_locs[i] = self.loc();
            inits_inner[i] = v.inner;
        }
        var after_tys: [M]*const mlir.Type = undefined;
        var after_locs: [M]*const mlir.Location = undefined;
        inline for (@typeInfo(@TypeOf(after_types)).@"struct".fields, 0..) |f, i| {
            after_tys[i] = @field(after_types, f.name);
            after_locs[i] = self.loc();
        }
        const before_block = mlir.Block.init(&before_types, &before_locs);
        const after_block = mlir.Block.init(&after_tys, &after_locs);
        self.pushBlock(before_block);
        var before_carried: [N]Value = undefined;
        for (0..N) |i| before_carried[i] = self.wrap(before_block.argument(i));
        return .{
            .kernel = self,
            .before_block = before_block,
            .after_block = after_block,
            .inits_inner = inits_inner,
            .after_types = after_tys,
            .before_carried = before_carried,
        };
    }

    /// `cond` as an `i1`: other integers compare unequal to zero, as in
    /// pykernel's C-style conditions.
    fn condition(self: *Builder, cond: anytype) Value {
        const v = self.lift(cond);
        if (v.type_().eql(.int(self.ctx, .i1))) return v;
        return v.ne(0);
    }

    pub fn openIf(self: *Builder, cond: anytype) IfOnlyScope {
        const c = self.condition(cond);
        const then_block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(then_block);
        return .{ .kernel = self, .cond_inner = c.inner, .then_block = then_block };
    }

    pub fn openIfElse(self: *Builder, cond: anytype, result_types: anytype) IfScope(dsl.tupleArity(@TypeOf(result_types), "openIfElse: result_types")) {
        const N = comptime dsl.tupleArity(@TypeOf(result_types), "openIfElse: result_types");
        const c = self.condition(cond);
        var types: [N]*const mlir.Type = undefined;
        inline for (@typeInfo(@TypeOf(result_types)).@"struct".fields, 0..) |f, i| types[i] = @field(result_types, f.name);
        const then_block = mlir.Block.init(&.{}, &.{});
        const else_block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(then_block);
        return .{ .kernel = self, .cond_inner = c.inner, .then_block = then_block, .else_block = else_block, .result_types = types };
    }

    /// A mutable `i32` initialized to `init`.
    pub fn variable(self: *Builder, init: anytype) Variable {
        const v = self.lift(init);
        const mem_type: *const mlir.Type = .memRef(v.type_(), &.{1}, null, null);
        // `memref.alloca` checks its allocation scope: verified by `finish`.
        const mem = self.emit(mlir.Operation.make(self.ctx, "memref.alloca", .{
            .results = .{ .flat = &.{mem_type} },
            .attributes = &.{.named(self.ctx, "operandSegmentSizes", .denseArray(self.ctx, .i32, &[_]i32{ 0, 0 }))},
            .verify = false,
            .location = self.loc(),
        }));
        const variable_: Variable = .{ .mem = mem.inner, .kernel = self };
        variable_.set(v);
        return variable_;
    }

    // ------------------------------------------------------------ operations

    /// Appends the `ttkernel` operation `name` (its mnemonic, dots as
    /// underscores) built from `args`, in its builder's order (operands,
    /// result types, attributes; see `ops`), and returns its results.
    /// Values and integers (`i32` constants) stand for operands; booleans,
    /// integers, floats and the `ttkernel` enums for attributes. Missing
    /// trailing optional arguments are absent, except the `noc` of the NoC
    /// transfers and barriers, which is NoC 0 as in pykernel.
    pub fn op(self: *Builder, comptime name: @EnumLiteral(), args: anytype) OpResult(name) {
        const tag = @tagName(name);
        const build = @field(ops, tag);
        const params = @typeInfo(@TypeOf(build)).@"fn".params;
        const fields = @typeInfo(@TypeOf(args)).@"struct".fields;
        if (fields.len > params.len - 2) @compileError("ttkernel.op(." ++ tag ++ "): too many arguments");

        var call_args: std.meta.ArgsTuple(@TypeOf(build)) = undefined;
        call_args[0] = self.ctx;
        inline for (params[1 .. params.len - 1], 1..) |p, i| {
            const P = p.type.?;
            call_args[i] = if (i - 1 < fields.len)
                self.coerce(P, @field(args, fields[i - 1].name))
            else
                self.omitted(P, tag, i - 1);
        }
        call_args[params.len - 1] = self.loc();

        const operation: *mlir.Operation = @call(.auto, build, call_args);
        _ = operation.appendTo(self.currentBlock());
        return switch (OpResult(name)) {
            void => {},
            Value => self.wrap(operation.result(0)),
            else => blk: {
                const out = self.arena.allocator().alloc(Value, operation.numResults()) catch @panic("OOM");
                for (out, 0..) |*v, i| v.* = self.wrap(operation.result(i));
                break :blk out;
            },
        };
    }

    fn omitted(self: *Builder, comptime P: type, comptime tag: []const u8, comptime i: usize) P {
        return switch (P) {
            ?*const mlir.Value => if (comptime isNocDefaulted(tag)) self.kernelNoc().inner else null,
            ?*const mlir.Attribute => null,
            []const *const mlir.Value, []const *const mlir.Type => &.{},
            else => @compileError(std.fmt.comptimePrint("ttkernel.op(.{s}): argument {d} is required", .{ tag, i })),
        };
    }

    fn coerce(self: *Builder, comptime P: type, arg: anytype) P {
        const A = @TypeOf(arg);
        return switch (P) {
            *const mlir.Value => self.lift(arg).inner,
            ?*const mlir.Value => if (A == @TypeOf(null)) null else self.lift(arg).inner,
            []const *const mlir.Value => blk: {
                const info = @typeInfo(A);
                if (info == .@"struct" and info.@"struct".is_tuple) {
                    const out = self.arena.allocator().alloc(*const mlir.Value, info.@"struct".fields.len) catch @panic("OOM");
                    inline for (info.@"struct".fields, 0..) |f, j| out[j] = self.lift(@field(arg, f.name)).inner;
                    break :blk out;
                }
                const out = self.arena.allocator().alloc(*const mlir.Value, arg.len) catch @panic("OOM");
                for (arg, 0..) |v, j| out[j] = self.lift(v).inner;
                break :blk out;
            },
            *const mlir.Attribute => self.attribute(arg),
            ?*const mlir.Attribute => if (A == @TypeOf(null)) null else self.attribute(arg),
            else => arg,
        };
    }

    fn attribute(self: *Builder, arg: anytype) *const mlir.Attribute {
        const A = @TypeOf(arg);
        if (A == *const mlir.Attribute) return arg;
        if (A == bool) return .boolean(self.ctx, arg);
        if (A == ReduceType) return ttkernel.reduceTypeAttr(self.ctx, arg);
        if (A == ReduceDim) return ttkernel.reduceDimAttr(self.ctx, arg);
        if (A == BcastType) return ttkernel.bcastTypeAttr(self.ctx, arg);
        return switch (@typeInfo(A)) {
            .comptime_int, .int => .int(self.ctx, .i32, arg),
            .comptime_float, .float => .float(self.ctx, .f32, arg),
            else => @compileError("ttkernel: cannot use a " ++ @typeName(A) ++ " as an attribute"),
        };
    }

    // ------------------------------------------------------------ opaque calls

    /// A call to `callee`, declared in `header`, for the tt-metal functions
    /// without an operation (tt-lang's `opaque_call`). `template_args` are
    /// C++ template arguments (integers, booleans, or strings written as is,
    /// such as enumerators), `args` the arguments (values, circular buffers,
    /// `i32` integers).
    pub fn call(self: *Builder, comptime header: []const u8, comptime callee: []const u8, template_args: anytype, args: anytype) void {
        _ = self.opaqueCall(header, callee, template_args, args, &.{});
    }

    /// The compute threads.
    pub const ComputeThread = enum { unpack, math, pack };

    /// `call` made by compute thread `thread` only (tt-metal's `MATH((...))`,
    /// `PACK((...))`, `UNPACK((...))`), for the functions only that thread
    /// declares.
    pub fn callOn(self: *Builder, thread: ComputeThread, comptime header: []const u8, comptime callee: []const u8, template_args: anytype, args: anytype) void {
        const operation = self.opaqueCall(header, callee, template_args, args, &.{});
        operation.setAttributeByName("ttlang.thread", .string(self.ctx, @tagName(thread)));
    }

    /// `call` of a function returning a `result_type`.
    pub fn callValue(self: *Builder, comptime header: []const u8, comptime callee: []const u8, template_args: anytype, args: anytype, result_type: *const mlir.Type) Value {
        return self.wrap(self.opaqueCall(header, callee, template_args, args, &.{result_type}).result(0));
    }

    fn isText(comptime T: type) bool {
        return switch (@typeInfo(T)) {
            .pointer => |p| p.child == u8 or (@typeInfo(p.child) == .array and @typeInfo(p.child).array.child == u8),
            else => false,
        };
    }

    fn opaqueCall(self: *Builder, comptime header: []const u8, comptime callee: []const u8, template_args: anytype, args: anytype, result_types: []const *const mlir.Type) *mlir.Operation {
        const a = self.arena.allocator();
        const arg_fields = @typeInfo(@TypeOf(args)).@"struct".fields;
        var operands: std.ArrayList(*const mlir.Value) = .empty;
        // The arguments in order when some are verbatim: operand indices
        // and the expressions.
        const order = a.alloc(*const mlir.Attribute, arg_fields.len) catch @panic("OOM");
        var any_verbatim = false;
        inline for (arg_fields, 0..) |f, i| {
            const arg = @field(args, f.name);
            if (@TypeOf(arg) == Verbatim) {
                order[i] = .string(self.ctx, arg.text);
                any_verbatim = true;
            } else {
                order[i] = .int(self.ctx, .i64, @as(i64, @intCast(operands.items.len)));
                operands.append(a, self.lift(arg).inner) catch @panic("OOM");
            }
        }

        const tmpl_fields = @typeInfo(@TypeOf(template_args)).@"struct".fields;
        const tmpl: ?*const mlir.Attribute = if (tmpl_fields.len == 0) null else blk: {
            const attrs = a.alloc(*const mlir.Attribute, tmpl_fields.len) catch @panic("OOM");
            inline for (tmpl_fields, 0..) |f, i| {
                const t = @field(template_args, f.name);
                const T = @TypeOf(t);
                attrs[i] = if (T == bool)
                    .boolean(self.ctx, t)
                else if (comptime isText(T))
                    .string(self.ctx, t)
                else
                    .int(self.ctx, .si32, @as(i64, @intCast(t)));
            }
            break :blk .array(self.ctx, attrs);
        };
        const operation = ops.opaque_call(self.ctx, operands.items, result_types, .string(self.ctx, callee), .string(self.ctx, header), tmpl, null, null, self.loc());
        if (any_verbatim) operation.setAttributeByName("ttlang.args", .array(self.ctx, order));
        _ = operation.appendTo(self.currentBlock());
        return operation;
    }

    // ------------------------------------------------------------ pykernel builtins

    /// `TensorAccessorArgs(cta_base, crta_base)`: the accessor arguments
    /// starting at compile-time argument `cta_base` and common runtime
    /// argument `crta_base`.
    pub fn tensorAccessorArgs(self: *Builder, cta_base: u32, crta_base: u32) Value {
        return self.emit(ops.TensorAccessorArgs(self.ctx, self.int(cta_base).inner, self.int(crta_base).inner, null, null, null, self.loc()));
    }

    /// The accessor arguments following `prev`'s.
    pub fn nextTensorAccessorArgs(self: *Builder, prev: Value) Value {
        return self.emit(ops.TensorAccessorArgs(self.ctx, null, null, prev.inner, null, null, self.loc()));
    }

    /// `TensorAccessor(args, address, page_size)`.
    pub fn tensorAccessor(self: *Builder, args: Value, address: Value, page_size: anytype) Value {
        return self.emit(ops.TensorAccessor(self.ctx, args.inner, address.inner, self.lift(page_size).inner, self.loc()));
    }

    // ------------------------------------------------------------ result

    /// Terminates the kernel and records its argument spec.
    pub fn finish(self: *Builder) FinishError!void {
        std.debug.assert(!self.finished);
        std.debug.assert(self.block_stack.items.len == 0);
        _ = func.returns(self.ctx, &.{}, self.loc()).appendTo(self.entry_block);
        self.func_op.setAttributeByName("ttkernel.arg_spec", ttkernel.argSpecAttr(self.ctx, &.{}, self.ct_arg_spec.items));
        self.finished = true;
        if (!self.module.operation().verify()) return error.InvalidMlir;
    }

    /// The tt-metal C++ source of the finished kernel. Lowers the module to
    /// EmitC: the builder cannot be extended afterwards. Caller owns the
    /// result.
    pub fn cpp(self: *Builder, allocator: std.mem.Allocator) FinishError![]u8 {
        std.debug.assert(self.finished);
        try ttkernel.convertToEmitC(self.module);
        return ttkernel.translateToCpp(allocator, self.module, self.name);
    }
};

// =============================================================================
// Host side
// =============================================================================

/// An io-tensor of the op: an input, or an output after the inputs.
pub const Tensor = struct {
    /// Index in the op's io-tensors.
    index: u32,
    /// Element shape.
    shape: []const i64,
    dtype: program.DataType,

    pub fn volume(self: Tensor) u64 {
        var n: u64 = 1;
        for (self.shape) |d| n *= @intCast(d);
        return n;
    }

    /// Number of 32x32 tiles, `ceil(volume / 1024)` as in pykernel.
    pub fn numTiles(self: Tensor) u32 {
        return @intCast(std.math.divCeil(u64, self.volume(), 1024) catch unreachable);
    }

    /// The tensor's base address, as a runtime argument (pykernel's
    /// `tensor.buffer_address()`).
    pub fn address(self: Tensor) Address {
        return .{ .tensor = self.index };
    }
};

/// Host argument: the base address of an io-tensor.
pub const Address = struct { tensor: u32 };

/// Host argument: a per-core runtime argument, one value per core of the
/// kernel's range, x fastest.
pub const PerCore = struct { values: []const u32 };

/// Host argument: the id of a program semaphore.
pub const Semaphore = struct { index: u32 };

/// A circular buffer of the program (pykernel's `OpCircularBuffer`).
pub const Cb = struct {
    index: u32,
    dtype: program.DataType,
    num_pages: u32,
};

pub fn tileDataType(dtype: program.DataType) ttcore.DataType {
    return switch (dtype) {
        inline else => |d| @field(ttcore.DataType, @tagName(d)),
    };
}

pub const ThreadKind = enum { reader, writer, compute };

/// A thread kernel: `body` is `fn (*Builder, params...) void`.
pub fn Thread(comptime Body: type) type {
    return struct {
        kind: ThreadKind,
        name: []const u8,
        body: Body,
    };
}

/// pykernel's `@reader_thread()`.
pub fn readerThread(comptime name: []const u8, comptime body: anytype) Thread(@TypeOf(body)) {
    return .{ .kind = .reader, .name = name, .body = body };
}

/// pykernel's `@writer_thread()`.
pub fn writerThread(comptime name: []const u8, comptime body: anytype) Thread(@TypeOf(body)) {
    return .{ .kind = .writer, .name = name, .body = body };
}

/// pykernel's `@compute_thread()`.
pub fn computeThread(comptime name: []const u8, comptime body: anytype) Thread(@TypeOf(body)) {
    return .{ .kind = .compute, .name = name, .body = body };
}

/// The host program of an op (pykernel's `PyKernelOp`): its circular
/// buffers, semaphores and kernels.
pub const Program = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    ctx: *mlir.Context,
    tensors: []const Tensor,
    /// Default core range of circular buffers, semaphores and kernels.
    cores: program.CoreRange = .single(.{ .x = 0, .y = 0 }),
    kernels: std.ArrayList(program.Kernel) = .empty,
    circular_buffers: std.ArrayList(program.CircularBuffer) = .empty,
    semaphores: std.ArrayList(program.Semaphore) = .empty,

    pub fn init(allocator: std.mem.Allocator, ctx: *mlir.Context, tensors: []const Tensor) Program {
        return .{ .allocator = allocator, .arena = .init(allocator), .ctx = ctx, .tensors = tensors };
    }

    pub fn deinit(self: *Program) void {
        self.arena.deinit();
    }

    pub fn tensor(self: *const Program, i: usize) Tensor {
        return self.tensors[i];
    }

    pub const CbOpts = struct {
        /// Pages (tiles); pykernel sizes the buffer for the whole tensor.
        num_pages: ?u32 = null,
        dtype: ?program.DataType = null,
        cores: ?program.CoreRange = null,
    };

    /// A circular buffer of `t`'s tiles at `buffer_index` (pykernel's
    /// `create_cb`).
    pub fn createCb(self: *Program, t: Tensor, buffer_index: u32, opts: CbOpts) !Cb {
        const dtype = opts.dtype orelse t.dtype;
        const num_pages = opts.num_pages orelse t.numTiles();
        try self.circular_buffers.append(self.arena.allocator(), .{
            .index = buffer_index,
            .dtype = dtype,
            .page_size = dtype.tileSize(),
            .num_pages = num_pages,
            .cores = opts.cores orelse self.cores,
        });
        return .{ .index = buffer_index, .dtype = dtype, .num_pages = num_pages };
    }

    pub fn createSemaphore(self: *Program, initial_value: u32, cores: ?program.CoreRange) !Semaphore {
        try self.semaphores.append(self.arena.allocator(), .{ .cores = cores orelse self.cores, .initial_value = initial_value });
        return .{ .index = @intCast(self.semaphores.items.len - 1) };
    }

    pub const KernelOpts = struct {
        cores: ?program.CoreRange = null,
        /// Tensors whose accessor arguments follow the circular buffers in
        /// the compile-time arguments (pykernel's `tensor_accessor_configs`).
        tensor_accessors: []const Tensor = &.{},
        compute: program.ComputeConfig = .{},
    };

    /// Builds `thread`'s kernel for `args`, one host argument per parameter
    /// after the builder (pykernel's `create_kernel`).
    pub fn createKernel(self: *Program, comptime thread: anytype, args: anytype, opts: KernelOpts) !void {
        const Body = @TypeOf(thread.body);
        const params = @typeInfo(Body).@"fn".params;
        const fields = @typeInfo(@TypeOf(args)).@"struct".fields;
        if (params.len == 0 or params[0].type.? != *Builder) @compileError("ttkernel: a thread kernel takes a *Builder first");
        if (fields.len != params.len - 1) @compileError("ttkernel.createKernel: one argument per kernel parameter");

        const allocator = self.arena.allocator();
        const cores = opts.cores orelse self.cores;
        var k = try Builder.open(self.allocator, self.ctx, thread.name, if (thread.kind == .compute) .compute else .noc);
        defer k.deinit();

        var ct_args: std.ArrayList(program.Arg) = .empty;
        var common_args: std.ArrayList(program.Arg) = .empty;
        var per_core: std.ArrayList([]const u32) = .empty;

        var call_args: std.meta.ArgsTuple(Body) = undefined;
        call_args[0] = &k;
        // pykernel's prologue: circular buffers, then per-core, then common
        // runtime arguments.
        inline for (fields, 1..) |f, i| {
            if (params[i].type.? == CircularBuffer) {
                const cb: Cb = @field(args, f.name);
                const tile = ttcore.TileType.get(self.ctx, 32, 32, tileDataType(cb.dtype));
                const cb_type = ttkernel.CBType.get(self.ctx, cb.num_pages, tile.type_());
                call_args[i] = k.circularBuffer(@intCast(ct_args.items.len), cb_type.type_());
                try ct_args.append(allocator, .{ .cb = cb.index });
            }
        }
        inline for (fields, 1..) |f, i| {
            if (params[i].type.? == Value and @TypeOf(@field(args, f.name)) == PerCore) {
                const values = @field(args, f.name).values;
                if (values.len != cores.numCores()) return error.InvalidPerCoreArgs;
                call_args[i] = k.runtimeArg(@intCast(per_core.items.len));
                try per_core.append(allocator, values);
            }
        }
        inline for (fields, 1..) |f, i| {
            const arg = @field(args, f.name);
            const A = @TypeOf(arg);
            if (params[i].type.? == Value and A != PerCore) {
                call_args[i] = k.commonRuntimeArg(@intCast(common_args.items.len));
                try common_args.append(allocator, if (A == Address)
                    .{ .tensor_address = arg.tensor }
                else if (A == Semaphore)
                    .{ .semaphore = arg.index }
                else
                    .{ .value = @intCast(arg) });
            } else if (params[i].type.? != Value and params[i].type.? != CircularBuffer) {
                call_args[i] = arg;
            }
        }
        for (opts.tensor_accessors) |t| try ct_args.append(allocator, .{ .tensor_accessor_args = t.index });

        @call(.auto, thread.body, call_args);
        try k.finish();
        const source = try k.cpp(allocator);

        var runtime_args: std.ArrayList(program.CoreArgs) = .empty;
        if (per_core.items.len > 0) {
            const width = cores.end.x - cores.start.x + 1;
            var y = cores.start.y;
            while (y <= cores.end.y) : (y += 1) {
                var x = cores.start.x;
                while (x <= cores.end.x) : (x += 1) {
                    const linear = (y - cores.start.y) * width + (x - cores.start.x);
                    const core_args = try allocator.alloc(program.Arg, per_core.items.len);
                    for (core_args, per_core.items) |*a, values| a.* = .{ .value = values[linear] };
                    try runtime_args.append(allocator, .{ .core = .{ .x = x, .y = y }, .args = core_args });
                }
            }
        }

        try self.kernels.append(allocator, .{
            .thread = switch (thread.kind) {
                .reader => .reader,
                .writer => .writer,
                .compute => .{ .compute = opts.compute },
            },
            .source = source,
            .cores = cores,
            .compile_time_args = ct_args.items,
            .common_runtime_args = common_args.items,
            .runtime_args = runtime_args.items,
        });
    }

    pub fn descriptor(self: *const Program) program.Program {
        return .{
            .kernels = self.kernels.items,
            .circular_buffers = self.circular_buffers.items,
            .semaphores = self.semaphores.items,
        };
    }

    /// The `tt.tt_lang_op` artifact of the program. Caller owns it.
    pub fn artifact(self: *const Program, allocator: std.mem.Allocator) ![]u8 {
        return self.descriptor().artifact(allocator);
    }
};

// =============================================================================
// Tests
// =============================================================================

// pykernel's `test/pykernel/demo/eltwise_sfpu_demo.py`.

fn eltwiseSfpu(k: *Builder, cb_in: CircularBuffer, cb_out: CircularBuffer, per_core_block_cnt: u32, per_core_block_dim: u32) void {
    k.op(.unary_op_init_common, .{ cb_in, cb_out });
    var blocks = k.openFor(0, per_core_block_cnt, 1, .{});
    {
        k.op(.cb_reserve_back, .{ cb_out, per_core_block_dim });
        var tiles = k.openFor(0, per_core_block_dim, 1, .{});
        {
            k.op(.tile_regs_acquire, .{});
            k.op(.cb_wait_front, .{ cb_in, 1 });
            k.op(.copy_tile_init, .{cb_in});
            k.op(.copy_tile, .{ cb_in, 0, 0 });
            k.op(.exp_tile_init, .{});
            k.op(.exp_tile, .{0});
            k.op(.tile_regs_commit, .{});
            k.op(.tile_regs_wait, .{});
            k.op(.pack_tile, .{ 0, cb_out, 0 });
            k.op(.cb_pop_front, .{ cb_in, 1 });
            k.op(.tile_regs_release, .{});
        }
        tiles.yield(.{});
        k.op(.cb_push_back, .{ cb_out, per_core_block_dim });
    }
    blocks.yield(.{});
}

fn writerUnaryInterleaved(k: *Builder, cb_out: CircularBuffer, dst_addr: Value, num_tiles: Value, start_id: Value) void {
    const onetile = 1;
    const tile_bytes = k.op(.get_tile_size, .{cb_out});
    const s0 = k.tensorAccessor(k.tensorAccessorArgs(1, 0), dst_addr, tile_bytes);
    const end_id = start_id.add(num_tiles);
    const ii = k.variable(start_id);
    var loop = k.openFor(start_id, end_id, onetile, .{});
    {
        k.op(.cb_wait_front, .{ cb_out, onetile });
        const l1_read_addr = k.op(.get_read_ptr, .{cb_out});
        k.op(.noc_async_write_tile, .{ ii.get(), s0, l1_read_addr });
        k.op(.noc_async_write_barrier, .{});
        k.op(.cb_pop_front, .{ cb_out, onetile });
        ii.set(ii.get().add(onetile));
    }
    loop.yield(.{});
}

fn readerUnaryInterleaved(k: *Builder, cb_in: CircularBuffer, src_addr: Value, num_tiles: Value, start_id: Value) void {
    const onetile = 1;
    const tile_bytes = k.op(.get_tile_size, .{cb_in});
    const s0 = k.tensorAccessor(k.tensorAccessorArgs(1, 0), src_addr, tile_bytes);
    const end_id = start_id.add(num_tiles);
    const ii = k.variable(start_id);
    var loop = k.openFor(start_id, end_id, onetile, .{});
    {
        k.op(.cb_reserve_back, .{ cb_in, onetile });
        const l1_write_addr = k.op(.get_write_ptr, .{cb_in});
        k.op(.noc_async_read_tile, .{ ii.get(), s0, l1_write_addr });
        k.op(.noc_async_read_barrier, .{});
        k.op(.cb_push_back, .{ cb_in, onetile });
        ii.set(ii.get().add(onetile));
    }
    loop.yield(.{});
}

pub const eltwise_sfpu = computeThread("eltwise_sfpu", eltwiseSfpu);
pub const writer_unary_interleaved = writerThread("writer_unary_interleaved", writerUnaryInterleaved);
pub const reader_unary_interleaved = readerThread("reader_unary_interleaved", readerUnaryInterleaved);

/// pykernel's `EltwiseSFPUPyKernelOp.invoke`: `out = exp(in)`.
pub fn eltwiseExp(p: *Program) !void {
    const in_tensor = p.tensor(0);
    const out_tensor = p.tensor(1);
    const cb_in = try p.createCb(in_tensor, 0, .{ .num_pages = 2 });
    const cb_out = try p.createCb(out_tensor, 1, .{ .num_pages = 2 });
    const num_tiles = in_tensor.numTiles();
    try p.createKernel(eltwise_sfpu, .{ cb_in, cb_out, num_tiles, 1 }, .{});
    try p.createKernel(writer_unary_interleaved, .{ cb_out, out_tensor.address(), num_tiles, 0 }, .{ .tensor_accessors = &.{out_tensor} });
    try p.createKernel(reader_unary_interleaved, .{ cb_in, in_tensor.address(), num_tiles, 0 }, .{ .tensor_accessors = &.{in_tensor} });
}

test "eltwise sfpu demo" {
    const ctx = try newContext();
    defer ctx.deinit();

    const shape: []const i64 = &.{ 1, 4, 32, 32 };
    var p: Program = .init(std.testing.allocator, ctx, &.{
        .{ .index = 0, .shape = shape, .dtype = .bf16 },
        .{ .index = 1, .shape = shape, .dtype = .bf16 },
    });
    defer p.deinit();
    try eltwiseExp(&p);

    const d = p.descriptor();
    try std.testing.expectEqual(3, d.kernels.len);
    try std.testing.expect(std.mem.indexOf(u8, d.kernels[0].source, "exp_tile(") != null);
    try std.testing.expect(std.mem.indexOf(u8, d.kernels[2].source, "void kernel_main()") != null);
    try std.testing.expectEqualDeep(&[_]program.Arg{ .{ .cb = 1 }, .{ .tensor_accessor_args = 1 } }, d.kernels[1].compile_time_args);
    try std.testing.expectEqualDeep(&[_]program.Arg{ .{ .tensor_address = 0 }, .{ .value = 4 }, .{ .value = 0 } }, d.kernels[2].common_runtime_args);

    const art = try p.artifact(std.testing.allocator);
    defer std.testing.allocator.free(art);
    try std.testing.expect(std.mem.startsWith(u8, art, "{\"format_version\": 3"));
}

test "opaque calls and runtime-picked circular buffers" {
    const ctx = try newContext();
    defer ctx.deinit();
    var k = try Builder.open(std.testing.allocator, ctx, "bcast", .compute);
    defer k.deinit();
    const a = k.circularBufferOf(0, .f32, 1);
    const b = k.circularBufferOf(1, .f32, 1);
    const picked = k.selectCb(k.runtimeArg(0), a, b);
    k.call("api/compute/bcast.h", "mul_tiles_bcast_cols", .{}, .{ picked, b, 0, 0, 0 });
    k.call("api/compute/cb_api.h", "cb_pop_front", .{}, .{ picked, 1 });
    k.call("api/compute/eltwise_unary/exp.h", "exp_tile", .{true}, .{0});
    k.call("api/compute/compute_kernel_hw_startup.h", "compute_kernel_hw_startup", .{"SrcOrder::Reverse"}, .{ a, b, a });
    k.callOn(.math, "api/compute/experimental/sdpa.h", "fused_max_sub_exp_add_tile", .{true}, .{ 0, 16256 });
    k.call("api/compute/binary_max_min.h", "binary_max_tile", .{}, .{ 0, 1, 0, verbatim("VectorMode::C") });
    var spin = k.openWhile(.{}, .{});
    spin.yieldBefore(k.runtimeArgAt(k.int(2)).eq(0), .{});
    spin.yieldAfter(.{});
    try k.finish();
    const src = try k.cpp(std.testing.allocator);
    defer std.testing.allocator.free(src);
    try std.testing.expect(std.mem.indexOf(u8, src, "#include \"api/compute/bcast.h\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "mul_tiles_bcast_cols(") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "exp_tile<true>(") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "compute_kernel_hw_startup<SrcOrder::Reverse>(") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "#ifdef TRISC_MATH") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "VectorMode::C)") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "while") != null);
}
