//! TTL DSL: tensor-level Tenstorrent kernels against tt-lang's `ttl`
//! dialect, after tt-lang's Python API (`python/ttl`, the multi-kernel form
//! of `@ttl.operation`). tt-lang's pipeline lowers them to TTKernel and
//! tt-metal C++ here; the result is a `program.zig` artifact.
//!
//! | tt-lang                                          | here                                                  |
//! |--------------------------------------------------|-------------------------------------------------------|
//! | `@ttl.operation(grid=(cols, rows))`              | `op.grid = .{ cols, rows };`                          |
//! | `d = ttl.make_dataflow_buffer_like(t, shape=(1, 1), block_count=2)` | `const d = op.makeDataflowBufferLike(t, .{ .shape = &.{ 1, 1 } });` |
//! | `@ttl.compute() def f(): ...` (captures `d`)     | `try op.compute("f", fBody, .{d});` with `fn fBody(k: *Builder, d: DataflowBuffer) void` |
//! | `@ttl.datamovement() def g(): ...` (captures `t`)| `try op.datamovement("g", gBody, .{ t, d });` with a `TensorArg` parameter |
//! | `with a.wait() as x, b.reserve() as y:`          | `const x = a.wait(); defer x.pop(); const y = b.reserve(); defer y.push();` |
//! | `y.store(x + z)` / `ttl.math.exp(x)`             | `y.store(x.add(z))` / `x.exp(.{})`                     |
//! | `ttl.math.reduce_sum(x, dims=[1])`               | `x.reduceSum(&.{1})`                                  |
//! | `ttl.block.broadcast(x, dims=[1], shape=(1, 4))` | `x.broadcast(&.{1}, &.{ 1, 4 })`                      |
//! | `ttl.copy(t[r, c], blk).wait()`                  | `k.copy(t.at(.{ r, c }), blk).wait()`                 |
//! | `for i in range(n):`                             | `var loop = k.openFor(0, n, 1, .{}); ... loop.yield(.{});` |
//! | `x, y = ttl.node(dims=2)`                        | `const node = k.node();` (`node.x`, `node.y`)         |
//!
//! Dataflow buffers are numbered in creation order, like tt-lang's; the
//! first data-movement thread is the reader (NoC 0), the second the writer
//! (NoC 1), and an operation has one compute and two data-movement threads.
//! Scalars (loop variables, node coordinates, tile indices) are `index`.

const std = @import("std");

const dialects = @import("mlir/dialects");
const arith = dialects.arith;
const func = dialects.func;
const dsl = @import("kernels/common");
const mlir = @import("mlir");
const ttkernel = @import("mlir/dialects/ttkernel");
const ttl = @import("mlir/dialects/ttl");

pub const program = @import("kernels/ttkernel/program");

const ttcore = ttkernel.ttcore;

pub const ops = ttl.ops;

/// A context with the dialects a TTL kernel uses. Caller owns it.
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
    std.testing.refAllDecls(Block);
    std.testing.refAllDecls(Operation);
}

fn tileDataType(dtype: program.DataType) ttcore.DataType {
    return switch (dtype) {
        inline else => |d| @field(ttcore.DataType, @tagName(d)),
    };
}

fn programDataType(dtype: ttcore.DataType) !program.DataType {
    return switch (dtype) {
        inline else => |d| if (@hasField(program.DataType, @tagName(d))) @field(program.DataType, @tagName(d)) else error.UnsupportedDataType,
    };
}

fn tileType(ctx: *mlir.Context, dtype: program.DataType) *const mlir.Type {
    return ttcore.TileType.get(ctx, 32, 32, tileDataType(dtype)).type_();
}

fn indexAttr(ctx: *mlir.Context, v: i64) *const mlir.Attribute {
    return .intFromType(@ptrCast(mlir.Type.index(ctx)), v);
}

// =============================================================================
// Host side
// =============================================================================

/// An io-tensor of the operation: an input, or an output after the inputs.
/// Tiled, interleaved in DRAM.
pub const Tensor = struct {
    /// Index in the operation's io-tensors.
    index: u32,
    /// Element shape, rank 2 or more.
    shape: []const i64,
    dtype: program.DataType,

    /// The shape in tiles: the last two dimensions divided by 32.
    pub fn tileShape(self: Tensor, buf: []i64) []const i64 {
        const rank = self.shape.len;
        @memcpy(buf[0..rank], self.shape);
        buf[rank - 2] = std.math.divCeil(i64, self.shape[rank - 2], 32) catch unreachable;
        buf[rank - 1] = std.math.divCeil(i64, self.shape[rank - 1], 32) catch unreachable;
        return buf[0..rank];
    }
};

/// A dataflow buffer of the operation (tt-lang's `DataflowBuffer`): blocks
/// of `shape` tiles, `block_count` of them.
pub const Dfb = struct {
    index: u32,
    shape: []const i64,
    block_count: u32,
    dtype: program.DataType,
};

pub const ThreadKind = enum { compute, datamovement };

/// A TTL operation being built (tt-lang's `@ttl.operation`): its dataflow
/// buffers and threads, compiled to a `program.zig` program.
pub const Operation = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    ctx: *mlir.Context,
    module: *mlir.Module,
    tensors: []const Tensor,
    /// `(cols, rows)` of the core grid the threads run on.
    grid: [2]u32 = .{ 1, 1 },
    fp32_dest_acc_en: ?bool = null,
    dst_full_sync_en: ?bool = null,
    math_fidelity: program.MathFidelity = .hifi4,
    arch: ttcore.Arch = .blackhole,
    dfbs: std.ArrayList(Dfb) = .empty,
    threads: std.ArrayList(*mlir.Operation) = .empty,
    num_datamovement: u32 = 0,

    pub fn init(allocator: std.mem.Allocator, ctx: *mlir.Context, tensors: []const Tensor) Operation {
        return .{
            .allocator = allocator,
            .arena = .init(allocator),
            .ctx = ctx,
            .module = .init(.unknown(ctx)),
            .tensors = tensors,
        };
    }

    pub fn deinit(self: *Operation) void {
        self.module.deinit();
        self.arena.deinit();
    }

    pub fn tensor(self: *const Operation, i: usize) Tensor {
        return self.tensors[i];
    }

    pub const DfbOpts = struct {
        /// Block shape in tiles.
        shape: []const i64,
        block_count: u32 = 2,
    };

    /// tt-lang's `make_dataflow_buffer_like(t, shape, block_count)`.
    pub fn makeDataflowBufferLike(self: *Operation, t: Tensor, opts: DfbOpts) Dfb {
        return self.makeDfb(t.dtype, opts);
    }

    /// tt-lang's `make_dfb(dtype, shape, block_count)`.
    pub fn makeDfb(self: *Operation, dtype: program.DataType, opts: DfbOpts) Dfb {
        const dfb: Dfb = .{
            .index = @intCast(self.dfbs.items.len),
            .shape = self.arena.allocator().dupe(i64, opts.shape) catch @panic("OOM"),
            .block_count = opts.block_count,
            .dtype = dtype,
        };
        self.dfbs.append(self.arena.allocator(), dfb) catch @panic("OOM");
        return dfb;
    }

    /// The `#ttl.layout` of an io-tensor: tiled, interleaved in DRAM.
    fn layout(self: *Operation, t: Tensor) *const mlir.Attribute {
        return ttl.layoutAttr(self.ctx, .{
            .shape = t.shape,
            .element_type = tileType(self.ctx, t.dtype),
            .buffer = .dram,
            .grid = &.{ self.grid[1], self.grid[0] },
            .memory = .interleaved,
        });
    }

    /// The `tensor<...>` a thread sees an io-tensor as: tiles, with its
    /// layout.
    fn tensorType(self: *Operation, t: Tensor) *const mlir.Type {
        var buf: [8]i64 = undefined;
        return .rankedTensorWithEncoding(t.tileShape(&buf), tileType(self.ctx, t.dtype), @constCast(self.layout(t)));
    }

    /// tt-lang's `@ttl.compute()`: `body` is `fn (*Builder, params...) void`.
    pub fn compute(self: *Operation, name: []const u8, comptime body: anytype, args: anytype) !void {
        try self.thread(.compute, name, body, args);
    }

    /// tt-lang's `@ttl.datamovement()`: the first is the reader, the second
    /// the writer.
    pub fn datamovement(self: *Operation, name: []const u8, comptime body: anytype, args: anytype) !void {
        try self.thread(.datamovement, name, body, args);
    }

    fn thread(self: *Operation, kind: ThreadKind, name: []const u8, comptime body: anytype, args: anytype) !void {
        const Body = @TypeOf(body);
        const params = @typeInfo(Body).@"fn".params;
        const fields = @typeInfo(@TypeOf(args)).@"struct".fields;
        if (params.len == 0 or params[0].type.? != *Builder) @compileError("ttl: a thread takes a *Builder first");
        if (fields.len != params.len - 1) @compileError("ttl: one argument per thread parameter");

        const ctx = self.ctx;
        const allocator = self.arena.allocator();
        // Captured io-tensors are the function's arguments.
        var arg_types: std.ArrayList(*const mlir.Type) = .empty;
        var arg_locs: std.ArrayList(*const mlir.Location) = .empty;
        var crta: std.ArrayList(*const mlir.Attribute) = .empty;
        inline for (fields, 1..) |f, i| {
            if (params[i].type.? == TensorArg) {
                const t: Tensor = @field(args, f.name);
                try arg_types.append(allocator, self.tensorType(t));
                try arg_locs.append(allocator, .unknown(ctx));
                try crta.append(allocator, .int(ctx, .i32, t.index));
            }
        }
        const entry = mlir.Block.init(arg_types.items, arg_locs.items);

        var attrs: std.ArrayList(mlir.NamedAttribute) = .empty;
        try attrs.append(allocator, .named(ctx, "ttl.kernel_thread", ttkernel.threadAttr(ctx, if (kind == .compute) .compute else .noc)));
        try attrs.append(allocator, .named(ctx, "ttl.crta_indices", .array(ctx, crta.items)));
        try attrs.append(allocator, .named(ctx, "ttl.logical_kernel", try mlir.Attribute.parse(ctx, if (kind == .compute)
            "#ttl.logical_kernel<kind = compute>"
        else
            "#ttl.logical_kernel<kind = data_movement>")));
        switch (kind) {
            .compute => {
                if (self.fp32_dest_acc_en) |v| try attrs.append(allocator, .named(ctx, "fp32_dest_acc_en", .boolean(ctx, v)));
                if (self.dst_full_sync_en) |v| try attrs.append(allocator, .named(ctx, "dst_full_sync_en", .boolean(ctx, v)));
            },
            .datamovement => {
                try attrs.append(allocator, .named(ctx, "ttl.noc_index", .int(ctx, .i32, self.num_datamovement)));
                self.num_datamovement += 1;
            },
        }
        const func_op = func.func(ctx, .{
            .name = name,
            .args = arg_types.items,
            .results = &.{},
            .block = entry,
            .location = .unknown(ctx),
            .extra_attributes = attrs.items,
            .verify = false,
        });
        _ = func_op.appendTo(self.module.body());
        try self.threads.append(allocator, func_op);

        var k: Builder = .{ .op = self, .ctx = ctx, .entry_block = entry };
        var call_args: std.meta.ArgsTuple(Body) = undefined;
        call_args[0] = &k;
        var next_tensor: usize = 0;
        inline for (fields, 1..) |f, i| {
            const arg = @field(args, f.name);
            const P = params[i].type.?;
            if (P == TensorArg) {
                call_args[i] = .{ .inner = entry.argument(next_tensor), .kernel = &k, .tensor = arg, .layout = self.layout(arg) };
                next_tensor += 1;
            } else if (P == DataflowBuffer) {
                call_args[i] = k.bindCb(arg);
            } else {
                call_args[i] = arg;
            }
        }
        @call(.auto, body, call_args);
        std.debug.assert(k.block_stack.items.len == 0);
        _ = func.returns(ctx, &.{}, k.loc()).appendTo(entry);
    }

    /// tt-lang's pass pipeline (`ttl_api.py`), with its default compiler
    /// options.
    fn pipeline(self: *const Operation, buf: []u8) ![]const u8 {
        var w: std.Io.Writer = .fixed(buf);
        try w.writeAll("builtin.module(" ++
            "func.func(ttl-form-accumulation-scopes{strategy=auto},ttl-lower-accumulation-scopes{strategy=auto},ttl-materialize-loop-state)," ++
            "func.func(ttl-insert-copy-wait)," ++
            "func.func(ttl-insert-cb-sync{sync-user-dfbs=1},ttl-coalesce-dfb-acquires{sync-user-dfbs=1})," ++
            "func.func(ttl-insert-accumulation-scopes{kind=dfb})," ++
            "func.func(ttl-lower-accumulation-scopes{kind=dfb})," ++
            "func.func(ttl-create-producer-compute)," ++
            "func.func(ttl-insert-intermediate-dfbs{enable=1})," ++
            "func.func(convert-ttl-to-compute)," ++
            "func.func(ttl-insert-cb-sync{sync-user-dfbs=1})," ++
            "ttl-verify-pipenet," ++
            "ttl-form-pipe-transports{group-size=0}," ++
            "func.func(ttl-coalesce-dfb-acquires{sync-user-dfbs=1})," ++
            "ttl-finalize-dfb-indices{reuse-user-dfbs=1 unsafe-assume-allocation-groups=0 exact-coloring-search-limit=1000000 l1-budget-override=0}," ++
            "ttl-set-compute-kernel-config{");
        if (self.fp32_dest_acc_en) |v| try w.print("fp32-dest-acc-en={s} ", .{if (v) "enabled" else "disabled"});
        if (self.dst_full_sync_en) |v| try w.print("dst-full-sync-en={s} ", .{if (v) "enabled" else "disabled"});
        try w.writeAll("reduce-full-fp32=1 matmul-full-fp32=1 enable-fpu-binary-ops=1}," ++
            "func.func(ttl-assign-dst)," ++
            "func.func(ttl-subblock-compute-for-dst{subblock-sync=false strict-f32-acc=false})," ++
            "func.func(ttl-lower-to-loops{dst-accumulation=true use-block-matmul=true})," ++
            "func.func(ttl-schedule-operations)," ++
            "func.func(ttl-annotate-cb-associations)," ++
            "ttl-verify-dfb-spsc," ++
            "ttl-verify-dfb-lifecycle," ++
            "ttl-erase-pipenet-scopes," ++
            "ttl-validate-cb-budget," ++
            "ttl-lower-dprint-to-emitc," ++
            "convert-ttl-to-ttkernel{reduce-full-fp32=1 pipe-computed-addresses=1 pipe-capacity-sync=1 pipe-global-semaphores-only=0 fabric-mux=1 l1-budget-override=0}," ++
            "func.func(ttkernel-lower-scalar-fp-types)," ++
            "ttkernel-insert-inits," ++
            "ttkernel-insert-l1-accumulation," ++
            "func.func(ttkernel-combine-pack-tiles)," ++
            "canonicalize," ++
            "cse," ++
            "ttkernel-cleanup-and-finalize-runtime-args," ++
            "ttl-lower-signpost-to-emitc," ++
            "convert-ttkernel-to-emitc," ++
            "symbol-dce)");
        return w.buffered();
    }

    /// Lowers the operation and returns its program (tt-lang's
    /// `CompiledTTNNKernel`). The operation cannot be extended afterwards.
    pub fn compile(self: *Operation) !program.Program {
        const ctx = self.ctx;
        const allocator = self.arena.allocator();
        for (self.threads.items) |t| t.setAttributeByName("ttl.base_cta_index", .int(ctx, .i32, self.dfbs.items.len));
        const module_op = self.module.operation();
        module_op.setAttributeByName("ttl.launch_grid", .array(ctx, &.{ .int(ctx, .i64, self.grid[0]), .int(ctx, .i64, self.grid[1]) }));
        module_op.setAttributeByName("ttl.target_arch", ttcore.archAttr(ctx, self.arch));
        if (!module_op.verify()) return error.InvalidMlir;

        var pipeline_buf: [4096]u8 = undefined;
        try ttl.runPassPipeline(self.module, try self.pipeline(&pipeline_buf));
        const desc = try ttl.describeKernels(allocator, self.module);

        if ((desc.pipe_sync_semaphores orelse 0) > 0 or (desc.pipe_global_semaphores orelse 0) > 0 or
            (desc.pipe_sram_scratch_bytes orelse 0) > 0 or (desc.dfb_resets orelse 0) > 0 or desc.dfb_reconfiguration)
            return error.UnsupportedPipes;

        const cores: program.CoreRange = .grid(self.grid[0], self.grid[1]);

        // Circular buffers: one per storage, dataflow buffers sharing a
        // storage are aliases of the largest.
        var cbs: std.ArrayList(program.CircularBuffer) = .empty;
        for (desc.dfbs) |d| {
            if (d.storage_segments or d.remote or d.tile_height != 32 or d.tile_width != 32) return error.UnsupportedDfb;
            const dtype = try programDataType(@enumFromInt(d.data_type));
            const num_pages = d.num_tiles * d.block_count;
            const existing = for (cbs.items, 0..) |cb, i| {
                if (desc.dfbs[cb.index].storage_index == d.storage_index) break i;
            } else null;
            if (existing) |i| {
                var cb = &cbs.items[i];
                var aliases: std.ArrayList(program.CircularBuffer.Alias) = .fromOwnedSlice(@constCast(cb.aliases));
                if (num_pages * d.page_size > cb.num_pages * cb.page_size) {
                    try aliases.append(allocator, .{ .index = cb.index, .dtype = cb.dtype, .page_size = cb.page_size });
                    cb.index = d.dfb_index;
                    cb.dtype = dtype;
                    cb.page_size = d.page_size;
                    cb.num_pages = num_pages;
                } else {
                    try aliases.append(allocator, .{ .index = d.dfb_index, .dtype = dtype, .page_size = d.page_size });
                }
                cb.aliases = aliases.items;
            } else {
                try cbs.append(allocator, .{ .index = d.dfb_index, .dtype = dtype, .page_size = d.page_size, .num_pages = num_pages, .cores = cores });
            }
        }

        // Compile-time arguments: the dataflow buffer indices, then, for
        // data movement, the accessor arguments of every io-tensor.
        // Common runtime arguments: the addresses of the kernel's tensors.
        const num_dfbs = desc.dfbs.len;
        var kernels: std.ArrayList(program.Kernel) = .empty;
        for (desc.kernels) |kd| {
            if (kd.specialized or kd.local_tensors or kd.fabric or kd.pipe_computed_address_dfb_indices.len > 0) return error.UnsupportedKernel;
            if (kd.base_cta_index) |b| if (b != num_dfbs) return error.InvalidMlir;
            const is_compute = std.mem.eql(u8, kd.thread, "compute");

            var ct_args: std.ArrayList(program.Arg) = .empty;
            for (0..num_dfbs) |i| try ct_args.append(allocator, .{ .cb = @intCast(i) });
            if (!is_compute) for (0..self.tensors.len) |i| try ct_args.append(allocator, .{ .tensor_accessor_args = @intCast(i) });
            var common: std.ArrayList(program.Arg) = .empty;
            for (kd.crta_indices) |i| try common.append(allocator, .{ .tensor_address = i });

            const source = try ttkernel.translateToCpp(allocator, self.module, kd.name);
            try kernels.append(allocator, .{
                .thread = if (is_compute) .{ .compute = .{
                    .math_fidelity = self.math_fidelity,
                    .fp32_dest_acc_en = self.fp32_dest_acc_en orelse kd.fp32_dest_acc_en orelse false,
                    .dst_full_sync_en = self.dst_full_sync_en orelse kd.dst_full_sync_en orelse false,
                    .unpack_to_dest_fp32 = kd.unpack_to_dest_fp32,
                } } else if ((kd.noc_index orelse 0) == 0) .reader else .writer,
                .source = source,
                .cores = cores,
                .compile_time_args = ct_args.items,
                .common_runtime_args = common.items,
            });
        }
        return .{ .kernels = kernels.items, .circular_buffers = cbs.items };
    }

    /// The `tt.tt_lang_op` artifact of the compiled operation. Caller owns it.
    pub fn artifact(self: *Operation, allocator: std.mem.Allocator) ![]u8 {
        const p = try self.compile();
        return p.artifact(allocator);
    }
};

// =============================================================================
// Kernel side
// =============================================================================

/// An `index` scalar: loop variable, node coordinate, tile index.
pub const Scalar = struct {
    inner: *const mlir.Value,
    kernel: ?*Builder = null,

    pub fn type_(self: Scalar) *const mlir.Type {
        return self.inner.type_();
    }

    fn kern(self: Scalar) *Builder {
        return self.kernel orelse @panic("ttl.Scalar has no owning Builder");
    }

    fn binary(self: Scalar, comptime build: anytype, rhs: anytype) Scalar {
        const k = self.kern();
        return k.emit(build(k.ctx, self.inner, k.lift(rhs).inner, k.loc()));
    }

    pub fn add(self: Scalar, rhs: anytype) Scalar {
        return self.binary(arith.addi, rhs);
    }

    pub fn sub(self: Scalar, rhs: anytype) Scalar {
        return self.binary(arith.subi, rhs);
    }

    pub fn mul(self: Scalar, rhs: anytype) Scalar {
        return self.binary(arith.muli, rhs);
    }

    pub fn div(self: Scalar, rhs: anytype) Scalar {
        return self.binary(arith.divsi, rhs);
    }

    pub fn rem(self: Scalar, rhs: anytype) Scalar {
        return self.binary(arith.remsi, rhs);
    }

    pub fn min(self: Scalar, rhs: anytype) Scalar {
        return self.binary(arith.minsi, rhs);
    }

    pub fn max(self: Scalar, rhs: anytype) Scalar {
        return self.binary(arith.maxsi, rhs);
    }

    fn cmp(self: Scalar, predicate: arith.CmpIPredicate, rhs: anytype) Scalar {
        const k = self.kern();
        return k.emit(arith.cmpi(k.ctx, predicate, self.inner, k.lift(rhs).inner, k.loc()));
    }

    pub fn lt(self: Scalar, rhs: anytype) Scalar {
        return self.cmp(.slt, rhs);
    }

    pub fn le(self: Scalar, rhs: anytype) Scalar {
        return self.cmp(.sle, rhs);
    }

    pub fn gt(self: Scalar, rhs: anytype) Scalar {
        return self.cmp(.sgt, rhs);
    }

    pub fn ge(self: Scalar, rhs: anytype) Scalar {
        return self.cmp(.sge, rhs);
    }

    pub fn eq(self: Scalar, rhs: anytype) Scalar {
        return self.cmp(.eq, rhs);
    }

    pub fn ne(self: Scalar, rhs: anytype) Scalar {
        return self.cmp(.ne, rhs);
    }
};

/// A thread's view of a dataflow buffer: `!ttl.cb<[shape], tile, blocks>`.
pub const DataflowBuffer = struct {
    cb: *const mlir.Value,
    kernel: *Builder,
    dfb: Dfb,

    fn blockType(self: DataflowBuffer) *const mlir.Type {
        return .rankedTensor(self.dfb.shape, tileType(self.kernel.ctx, self.dfb.dtype));
    }

    /// Consumer acquire (`dfb.wait()`); release with `Block.pop`.
    pub fn wait(self: DataflowBuffer) Block {
        const k = self.kernel;
        const view = k.emit(ops.cb_wait(k.ctx, self.cb, self.blockType(), null, k.loc()));
        return self.attach(view.inner, .wait);
    }

    /// Producer acquire (`dfb.reserve()`); release with `Block.push`.
    pub fn reserve(self: DataflowBuffer) Block {
        const k = self.kernel;
        const view = k.emit(ops.cb_reserve(k.ctx, self.cb, self.blockType(), null, k.loc()));
        return self.attach(view.inner, .reserve);
    }

    fn attach(self: DataflowBuffer, view: *const mlir.Value, acquisition: Block.Acquisition) Block {
        const k = self.kernel;
        const attached = k.emit(ops.attach_cb(k.ctx, view, self.cb, view.type_(), k.loc()));
        return .{ .inner = attached.inner, .kernel = k, .acquired = .{ .cb = self.cb, .view = view, .acquisition = acquisition } };
    }
};

/// A block of tiles: acquired from a dataflow buffer, or computed.
pub const Block = struct {
    inner: *const mlir.Value,
    kernel: *Builder,
    acquired: ?Acquired = null,

    pub const Acquisition = enum { wait, reserve };

    pub const Acquired = struct {
        cb: *const mlir.Value,
        /// The `cb_wait`/`cb_reserve` result, what `store` writes.
        view: *const mlir.Value,
        acquisition: Acquisition,
    };

    pub fn type_(self: Block) *const mlir.Type {
        return self.inner.type_();
    }

    fn tensorType(self: Block) *const mlir.RankedTensorType {
        return self.type_().isA(mlir.RankedTensorType).?;
    }

    pub fn rank(self: Block) usize {
        return self.tensorType().rank();
    }

    pub fn dim(self: Block, i: usize) i64 {
        return self.tensorType().dimension(i);
    }

    fn withShape(self: Block, shape: []const i64) *const mlir.Type {
        return .rankedTensor(shape, self.tensorType().elementType());
    }

    fn result(self: Block, op_: *mlir.Operation) Block {
        _ = op_.appendTo(self.kernel.currentBlock());
        return .{ .inner = op_.result(0), .kernel = self.kernel };
    }

    /// Releases a `wait` acquisition (`blk.pop()`).
    pub fn pop(self: Block) void {
        const k = self.kernel;
        std.debug.assert(self.acquired.?.acquisition == .wait);
        k.emitNone(ops.cb_pop(k.ctx, self.acquired.?.cb, null, k.loc()));
    }

    /// Releases a `reserve` acquisition (`blk.push()`).
    pub fn push(self: Block) void {
        const k = self.kernel;
        std.debug.assert(self.acquired.?.acquisition == .reserve);
        k.emitNone(ops.cb_push(k.ctx, self.acquired.?.cb, null, k.loc()));
    }

    /// `blk.store(x)`: writes `x` into this reserved block.
    pub fn store(self: Block, x: Block) void {
        const k = self.kernel;
        k.emitNone(ops.store(k.ctx, x.inner, self.acquired.?.view, null, null, k.loc()));
    }

    /// `blk += x`: accumulates `x` into this block.
    pub fn storeAccumulate(self: Block, x: Block) void {
        const k = self.kernel;
        switch (self.acquired.?.acquisition) {
            .reserve => k.emitNone(ops.store(k.ctx, x.inner, self.acquired.?.view, .unit(k.ctx), null, k.loc())),
            .wait => k.emitNone(ops.store(k.ctx, self.add(x).inner, self.acquired.?.view, null, null, k.loc())),
        }
    }

    fn binaryOp(self: Block, comptime build: anytype, rhs: Block) Block {
        return self.result(build(self.kernel.ctx, self.inner, rhs.inner, self.type_(), self.kernel.loc()));
    }

    pub fn add(self: Block, rhs: Block) Block {
        return self.binaryOp(ops.add, rhs);
    }

    pub fn sub(self: Block, rhs: Block) Block {
        return self.binaryOp(ops.sub, rhs);
    }

    /// Tile-wise product with a block, or scaling by a float.
    pub fn mul(self: Block, rhs: anytype) Block {
        if (@TypeOf(rhs) == Block) return self.binaryOp(ops.mul, rhs);
        const k = self.kernel;
        return self.result(ops.mul_unary_const(k.ctx, self.inner, self.type_(), .float(k.ctx, .f32, rhs), k.loc()));
    }

    pub fn div(self: Block, rhs: Block) Block {
        return self.binaryOp(ops.div, rhs);
    }

    pub fn max(self: Block, rhs: Block) Block {
        return self.binaryOp(ops.max, rhs);
    }

    pub fn min(self: Block, rhs: Block) Block {
        return self.binaryOp(ops.min, rhs);
    }

    pub fn eq(self: Block, rhs: Block) Block {
        return self.binaryOp(ops.eq, rhs);
    }

    pub fn ne(self: Block, rhs: Block) Block {
        return self.binaryOp(ops.ne, rhs);
    }

    pub fn gt(self: Block, rhs: Block) Block {
        return self.binaryOp(ops.gt, rhs);
    }

    pub fn lt(self: Block, rhs: Block) Block {
        return self.binaryOp(ops.lt, rhs);
    }

    /// The elementwise unary functions of `ttl.math` (tt-lang's
    /// `TTLElementwiseOps.def`); `exp` has its own method.
    pub const Unary = enum {
        exp2,
        log,
        sqrt,
        rsqrt,
        tanh,
        abs,
        neg,
        relu,
        sigmoid,
        floor,
        recip,
        sin,
        cos,
        tan,
        asin,
        acos,
        atan,
        ceil,
        sign,
        gelu,
        silu,
        hardsigmoid,
        expm1,
        square,
        softsign,
        signbit,
        frac,
        trunc,
    };

    pub fn unary(self: Block, comptime f: Unary) Block {
        return self.result(@field(ops, @tagName(f))(self.kernel.ctx, self.inner, self.type_(), self.kernel.loc()));
    }

    pub fn log(self: Block) Block {
        return self.unary(.log);
    }

    pub fn sqrt(self: Block) Block {
        return self.unary(.sqrt);
    }

    pub fn rsqrt(self: Block) Block {
        return self.unary(.rsqrt);
    }

    pub fn recip(self: Block) Block {
        return self.unary(.recip);
    }

    pub fn neg(self: Block) Block {
        return self.unary(.neg);
    }

    pub fn relu(self: Block) Block {
        return self.unary(.relu);
    }

    pub fn sigmoid(self: Block) Block {
        return self.unary(.sigmoid);
    }

    pub fn silu(self: Block) Block {
        return self.unary(.silu);
    }

    pub fn tanh(self: Block) Block {
        return self.unary(.tanh);
    }

    pub const ExpOpts = struct {
        approx: bool = false,
        /// Computes `exp(scale * x)`.
        scale: ?f32 = null,
    };

    /// `ttl.math.exp(x, approx, scale)`.
    pub fn exp(self: Block, opts: ExpOpts) Block {
        const k = self.kernel;
        return self.result(ops.exp(
            k.ctx,
            self.inner,
            self.type_(),
            if (opts.approx) .boolean(k.ctx, true) else null,
            if (opts.scale) |s| .float(k.ctx, .f32, s) else null,
            null,
            null,
            k.loc(),
        ));
    }

    /// `ttl.math.matmul(self, rhs, transpose_rhs)`: `[M, K] x [K, N]`, or
    /// `[M, K] x [N, K]^T`.
    pub fn matmul(self: Block, rhs: Block, opts: struct { transpose_rhs: bool = false }) Block {
        const k = self.kernel;
        const n = if (opts.transpose_rhs) rhs.dim(0) else rhs.dim(1);
        return self.result(ops.matmul(k.ctx, self.inner, rhs.inner, self.withShape(&.{ self.dim(0), n }), if (opts.transpose_rhs) .unit(k.ctx) else null, k.loc()));
    }

    fn reduce(self: Block, reduce_type: ttl.ReduceType, dims: []const i64) Block {
        const k = self.kernel;
        var shape: [8]i64 = undefined;
        for (0..self.rank()) |i| shape[i] = self.dim(i);
        for (dims) |d| shape[@intCast(@mod(d, @as(i64, @intCast(self.rank()))))] = 1;
        // The reduction scaler: one tile of ones.
        const scaler = k.emit(ops.fill(k.ctx, self.withShape(&.{ 1, 1 }), .float(k.ctx, .f32, 1.0), k.loc()));
        return self.result(ops.reduce(
            k.ctx,
            self.inner,
            scaler.inner,
            self.withShape(shape[0..self.rank()]),
            .int(k.ctx, .i32, @intFromEnum(reduce_type)),
            .denseArray(k.ctx, .i64, dims),
            k.loc(),
        ));
    }

    /// `ttl.math.reduce_sum(x, dims)`: reduced dimensions become 1.
    pub fn reduceSum(self: Block, dims: []const i64) Block {
        return self.reduce(.sum, dims);
    }

    /// `ttl.math.reduce_max(x, dims)`.
    pub fn reduceMax(self: Block, dims: []const i64) Block {
        return self.reduce(.max, dims);
    }

    /// `ttl.block.broadcast(x, dims, shape)`.
    pub fn broadcast(self: Block, dims: []const i64, shape: []const i64) Block {
        const k = self.kernel;
        return self.result(ops.block_broadcast(k.ctx, self.inner, self.withShape(shape), .denseArray(k.ctx, .i64, dims), .denseArray(k.ctx, .i64, shape), k.loc()));
    }

    /// `ttl.math.transpose(x)`: `[R, C]` tiles become `[C, R]`, each tile
    /// transposed.
    pub fn transpose(self: Block) Block {
        const k = self.kernel;
        return self.result(ops.transpose(k.ctx, self.inner, self.withShape(&.{ self.dim(1), self.dim(0) }), k.loc()));
    }

    /// `ttl.math.typecast(x, dtype)`.
    pub fn typecast(self: Block, dtype: program.DataType) Block {
        const k = self.kernel;
        var shape: [8]i64 = undefined;
        for (0..self.rank()) |i| shape[i] = self.dim(i);
        return self.result(ops.typecast(k.ctx, self.inner, .rankedTensor(shape[0..self.rank()], tileType(k.ctx, dtype)), k.loc()));
    }
};

/// A thread's view of an io-tensor: tiles, interleaved in DRAM.
pub const TensorArg = struct {
    inner: *const mlir.Value,
    kernel: *Builder,
    tensor: Tensor,
    layout: *const mlir.Attribute,

    /// `t[i, j, ...]`: the region starting at tile coordinates `indices`
    /// (one per dimension) that `Builder.copy` moves; its extent is the
    /// block's shape. Leading dimensions beyond the block's rank are
    /// squeezed.
    pub fn at(self: TensorArg, indices: anytype) Subscript {
        const k = self.kernel;
        const fields = @typeInfo(@TypeOf(indices)).@"struct".fields;
        const out = k.op.arena.allocator().alloc(*const mlir.Value, fields.len) catch @panic("OOM");
        inline for (fields, 0..) |f, i| out[i] = k.lift(@field(indices, f.name)).inner;
        return .{ .tensor = self, .indices = out };
    }
};

pub const Subscript = struct {
    tensor: TensorArg,
    indices: []const *const mlir.Value,

    fn slice(self: Subscript, shape: []const i64) *const mlir.Value {
        const k = self.tensor.kernel;
        const tensor_type = self.tensor.inner.type_().isA(mlir.RankedTensorType).?;
        const slice_type: *const mlir.Type = .rankedTensorWithEncoding(shape, tensor_type.elementType(), @constCast(self.tensor.layout));
        return k.emit(ops.tensor_slice(k.ctx, self.tensor.inner, self.indices, slice_type, k.loc())).inner;
    }
};

/// An asynchronous copy (`ttl.copy`'s transfer handle).
pub const Transfer = struct {
    inner: *const mlir.Value,
    kernel: *Builder,

    /// `tx.wait()`.
    pub fn wait(self: Transfer) void {
        const k = self.kernel;
        k.emitNone(ops.wait(k.ctx, self.inner, k.loc()));
    }
};

/// One thread of the operation being built: a `func.func` taking the
/// thread's io-tensors.
pub const Builder = struct {
    op: *Operation,
    ctx: *mlir.Context,
    entry_block: *mlir.Block,
    block_stack: std.ArrayList(*mlir.Block) = .empty,

    pub fn loc(self: *const Builder) *const mlir.Location {
        return .unknown(self.ctx);
    }

    pub fn pushBlock(self: *Builder, b: *mlir.Block) void {
        self.block_stack.append(self.op.arena.allocator(), b) catch @panic("OOM");
    }

    pub fn popBlock(self: *Builder) void {
        _ = self.block_stack.pop();
    }

    pub fn currentBlock(self: *Builder) *mlir.Block {
        if (self.block_stack.items.len > 0) return self.block_stack.items[self.block_stack.items.len - 1];
        return self.entry_block;
    }

    pub fn emit(self: *Builder, op_: *mlir.Operation) Scalar {
        _ = op_.appendTo(self.currentBlock());
        return .{ .inner = op_.result(0), .kernel = self };
    }

    pub fn emitNone(self: *Builder, op_: *mlir.Operation) void {
        _ = op_.appendTo(self.currentBlock());
    }

    pub fn index(self: *Builder, v: i64) Scalar {
        return self.emit(arith.constant_index(self.ctx, v, self.loc()));
    }

    /// `x` as an `index` scalar: scalars pass through, integers become
    /// constants.
    pub fn lift(self: *Builder, x: anytype) Scalar {
        const T = @TypeOf(x);
        if (T == Scalar) return x;
        return switch (@typeInfo(T)) {
            .comptime_int, .int => self.index(@intCast(x)),
            else => @compileError("ttl: cannot use a " ++ @typeName(T) ++ " as a scalar"),
        };
    }

    fn bindCb(self: *Builder, dfb: Dfb) DataflowBuffer {
        const ctx = self.ctx;
        const cb_type = ttl.CircularBufferType.get(ctx, dfb.shape, tileType(ctx, dfb.dtype), dfb.block_count);
        const cb = self.emit(ops.bind_cb(
            ctx,
            cb_type.type_(),
            indexAttr(ctx, dfb.index),
            .int(ctx, .i64, dfb.block_count),
            indexAttr(ctx, dfb.index),
            null,
            null,
            null,
            self.loc(),
        ));
        return .{ .cb = cb.inner, .kernel = self, .dfb = dfb };
    }

    // ------------------------------------------------------------ control flow

    pub fn ForScope(comptime N: usize) type {
        return dsl.ForScope(Builder, Scalar, N);
    }

    /// `for i in range(lower, upper, step)`; `inits` are loop-carried
    /// scalars.
    pub fn openFor(self: *Builder, lower: anytype, upper: anytype, step: anytype, inits: anytype) ForScope(dsl.tupleArity(@TypeOf(inits), "openFor: inits")) {
        const N = comptime dsl.tupleArity(@TypeOf(inits), "openFor: inits");
        const lb = self.lift(lower);
        const ub = self.lift(upper);
        const st = self.lift(step);
        var block_types: [N + 1]*const mlir.Type = undefined;
        var block_locs: [N + 1]*const mlir.Location = undefined;
        block_types[0] = .index(self.ctx);
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
        var carried: [N]Scalar = undefined;
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

    pub fn openIf(self: *Builder, cond: Scalar) dsl.IfOnlyScope(Builder, Scalar) {
        const then_block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(then_block);
        return .{ .kernel = self, .cond_inner = cond.inner, .then_block = then_block };
    }

    // ------------------------------------------------------------ queries

    pub const Node = struct { x: Scalar, y: Scalar };

    /// `ttl.node(dims=2)`: this core's (column, row) in the grid.
    pub fn node(self: *Builder) Node {
        return .{
            .x = self.emit(ops.core_x(self.ctx, self.loc())),
            .y = self.emit(ops.core_y(self.ctx, self.loc())),
        };
    }

    /// `ttl.node(dims=1)`: `x * rows + y`.
    pub fn nodeLinear(self: *Builder) Scalar {
        const n = self.node();
        return n.x.mul(self.op.grid[1]).add(n.y);
    }

    // ------------------------------------------------------------ data movement

    /// `ttl.copy(src, dst)`: a tensor region into a reserved block, or a
    /// waited block into a tensor region.
    pub fn copy(self: *Builder, src: anytype, dst: anytype) Transfer {
        const ctx = self.ctx;
        if (@TypeOf(src) == Subscript and @TypeOf(dst) == Block) {
            std.debug.assert(dst.acquired.?.acquisition == .reserve);
            const slice = src.slice(shapeOf(dst));
            const xf = self.emit(ops.copy(ctx, slice, dst.acquired.?.cb, ttl.transferHandleType(ctx, .read), null, self.loc()));
            return .{ .inner = xf.inner, .kernel = self };
        } else if (@TypeOf(src) == Block and @TypeOf(dst) == Subscript) {
            std.debug.assert(src.acquired.?.acquisition == .wait);
            const slice = dst.slice(shapeOf(src));
            const xf = self.emit(ops.copy(ctx, src.acquired.?.cb, slice, ttl.transferHandleType(ctx, .write), null, self.loc()));
            return .{ .inner = xf.inner, .kernel = self };
        } else @compileError("ttl.copy: a tensor subscript and a block");
    }

    fn shapeOf(b: Block) []const i64 {
        const t = b.tensorType();
        const out = b.kernel.op.arena.allocator().alloc(i64, t.rank()) catch @panic("OOM");
        for (out, 0..) |*d, i| d.* = t.dimension(i);
        return out;
    }

    /// `ttl.block.fill(value, shape, dtype)`.
    pub fn fill(self: *Builder, value: f32, shape: []const i64, dtype: program.DataType) Block {
        const op_ = ops.fill(self.ctx, .rankedTensor(shape, tileType(self.ctx, dtype)), .float(self.ctx, .f32, value), self.loc());
        _ = op_.appendTo(self.currentBlock());
        return .{ .inner = op_.result(0), .kernel = self };
    }
};

// =============================================================================
// Tests
// =============================================================================

// tt-lang's `test/python/simple_add.py`, over `n` tiles.

fn addCompute(k: *Builder, lhs_dfb: DataflowBuffer, rhs_dfb: DataflowBuffer, out_dfb: DataflowBuffer, n: u32) void {
    var loop = k.openFor(0, n, 1, .{});
    {
        const l = lhs_dfb.wait();
        defer l.pop();
        const r = rhs_dfb.wait();
        defer r.pop();
        const o = out_dfb.reserve();
        defer o.push();
        o.store(l.add(r));
    }
    loop.yield(.{});
}

fn addRead(k: *Builder, lhs: TensorArg, rhs: TensorArg, lhs_dfb: DataflowBuffer, rhs_dfb: DataflowBuffer, n: u32) void {
    var loop = k.openFor(0, n, 1, .{});
    {
        const l = lhs_dfb.reserve();
        defer l.push();
        k.copy(lhs.at(.{ 0, loop.iv }), l).wait();
        const r = rhs_dfb.reserve();
        defer r.push();
        k.copy(rhs.at(.{ 0, loop.iv }), r).wait();
    }
    loop.yield(.{});
}

fn addWrite(k: *Builder, out: TensorArg, out_dfb: DataflowBuffer, n: u32) void {
    var loop = k.openFor(0, n, 1, .{});
    {
        const o = out_dfb.wait();
        defer o.pop();
        k.copy(o, out.at(.{ 0, loop.iv })).wait();
    }
    loop.yield(.{});
}

/// `out = lhs + rhs` over `[32, 32 * n]` bf16 tensors, one tile at a time.
pub fn eltwiseAdd(op: *Operation) !void {
    const lhs = op.tensor(0);
    const rhs = op.tensor(1);
    const out = op.tensor(2);
    const n: u32 = @intCast(@divExact(lhs.shape[1], 32));
    const lhs_dfb = op.makeDataflowBufferLike(lhs, .{ .shape = &.{ 1, 1 } });
    const rhs_dfb = op.makeDataflowBufferLike(rhs, .{ .shape = &.{ 1, 1 } });
    const out_dfb = op.makeDataflowBufferLike(out, .{ .shape = &.{ 1, 1 } });
    try op.compute("add_compute", addCompute, .{ lhs_dfb, rhs_dfb, out_dfb, n });
    try op.datamovement("dm_read", addRead, .{ lhs, rhs, lhs_dfb, rhs_dfb, n });
    try op.datamovement("dm_write", addWrite, .{ out, out_dfb, n });
}

test "every elementwise function is a ttl operation" {
    inline for (comptime std.meta.fieldNames(Block.Unary)) |name| _ = @field(ops, name);
}

test "simple add" {
    const ctx = try newContext();
    defer ctx.deinit();
    const shape: []const i64 = &.{ 32, 128 };
    var op: Operation = .init(std.testing.allocator, ctx, &.{
        .{ .index = 0, .shape = shape, .dtype = .bf16 },
        .{ .index = 1, .shape = shape, .dtype = .bf16 },
        .{ .index = 2, .shape = shape, .dtype = .bf16 },
    });
    defer op.deinit();
    try eltwiseAdd(&op);
    const p = try op.compile();

    try std.testing.expectEqual(3, p.kernels.len);
    try std.testing.expectEqual(3, p.circular_buffers.len);
    for (p.kernels) |kernel| try std.testing.expect(std.mem.indexOf(u8, kernel.source, "void kernel_main()") != null);
    try std.testing.expectEqualDeep(&[_]program.Arg{ .{ .tensor_address = 0 }, .{ .tensor_address = 1 } }, p.kernels[1].common_runtime_args);
}
