const asynk = @import("async");
const builtin = @import("builtin");
const dialect = @import("mlir/dialects");
const runfiles = @import("runfiles");
const std = @import("std");
const stdx = @import("stdx");
const xla_pb = @import("//xla:xla_proto");

const meta = @import("meta.zig");
const mlir = @import("mlir.zig");
const ops = @import("ops.zig");
const pjrt = @import("pjrtx.zig");
const aio = @import("aio.zig");

const Buffer = @import("buffer.zig").Buffer;
const Bufferized = @import("tensor.zig").Bufferized;
const Context = @import("context.zig").Context;
const Location = mlir.Location;
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;
const ShapeOf = @import("tensor.zig").ShapeOf;
const Target = @import("platform.zig").Target;
const Tensor = @import("tensor.zig").Tensor;
const Tracer = @import("tools/tracer.zig").Tracer;

const assert = std.debug.assert;
const log = std.log.scoped(.@"zml/module");

test {
    std.testing.refAllDecls(@This());
}

pub const BlockKind = enum { open, hermetic };

const Block = union(BlockKind) {
    open: mlir.Block,
    hermetic: mlir.Block,

    pub fn block(self: Block) mlir.Block {
        return switch (self) {
            inline .open, .hermetic => |t| t,
        };
    }

    fn appendTensorRecursive(self: Block, x: *const Tensor) void {
        self.appendValueRecursive(x.value());
    }

    fn appendValueRecursive(self: Block, value: mlir.Value) void {
        switch (value.kind()) {
            .op_result => |parent_op| self.appendOperationRecursive(parent_op),
            .block_argument => |arg| {
                // Hermetic blocks are not allowed to use arguments from other blocks.
                std.debug.assert(self == .open or self.block().eql(arg.block()));
            },
            .null => @panic("InvalidMlir"),
        }
    }

    fn appendOperationRecursive(self: Block, op: mlir.Operation) void {
        if (op.block()) |prev_block| {
            // Hermetic blocks are not allowed to reference values from other blocks.
            std.debug.assert(self == .open or prev_block.equals(self.block()));
            return;
        }
        for (0..op.numOperands()) |i| {
            self.appendValueRecursive(op.operand(i));
        }
        self.block().appendOperation(op);
    }
};

pub const CompilationContext = struct {
    _platform: Platform,
    _name: []const u8,

    _mlir_ctx: mlir.Context,
    _mlir_registry: mlir.Registry,
    _mlir_canonicalizer: mlir.PassManager,

    _module: mlir.Module,

    _blocks: std.BoundedArray(Block, 64),
    _fn_cache: FnCache,
    _allocator: std.mem.Allocator,

    _buffer_to_arg: TensorToBlockArg = .{},
    _unique_id: u64 = 10000,
    _tracer: Tracer,

    _previous: ?*CompilationContext = null,
    threadlocal var _current: ?*CompilationContext = null;

    const TensorToBlockArg = std.AutoHashMapUnmanaged(Tensor._Id, struct { mlir.Value, Tensor._Donation });
    const AttributeList = std.BoundedArray(mlir.NamedAttribute, 3);

    pub fn init(allocator: std.mem.Allocator, name: []const u8, platform: Platform) !CompilationContext {
        const mlir_registry = mlir.Registry.init() catch unreachable;
        inline for (.{ "func", "stablehlo" }) |d| {
            mlir.DialectHandle.fromString(d).insertDialect(mlir_registry);
        }
        var mlir_ctx = mlir.Context.initWithRegistry(mlir_registry, false) catch unreachable;
        mlir_ctx.loadAllAvailableDialects();

        const loc = mlir_ctx.location(@src()).named(mlir_ctx, "main");
        const module = mlir.Module.init(loc);
        module.op().setAttributeByName("sym_name", mlir.StringAttribute.init(mlir_ctx, name).as(mlir.Attribute).?);

        var canonicalizer = try mlir.PassManager.init(mlir_ctx);
        {
            var opm = canonicalizer.asOpPassManager();
            try opm.addPipeline("canonicalize");
            try opm.addPipeline("cse");
            try opm.addPipeline("canonicalize");
        }

        return .{
            ._platform = platform,
            ._name = name,
            ._mlir_ctx = mlir_ctx,
            ._mlir_registry = mlir_registry,
            ._mlir_canonicalizer = canonicalizer,
            ._module = module,
            ._blocks = .{},
            ._fn_cache = FnCache.init(allocator),
            ._allocator = allocator,
            ._tracer = Tracer.init("ai.zml.compilation"),
        };
    }

    pub fn deinit(self: *CompilationContext) void {
        self._fn_cache.deinit();
        self._mlir_ctx.deinit();
        self._mlir_registry.deinit();
        self._buffer_to_arg.deinit(self._allocator);
    }

    pub fn activate(self: *CompilationContext) void {
        self._previous = _current;
        _current = self;
    }

    pub fn deactivate(self: *CompilationContext) void {
        assert(_current != null and _current.? == self);
        _current = self._previous;
        self._previous = null;
    }

    pub fn current() *CompilationContext {
        return _current.?;
    }

    pub fn target(self: *const CompilationContext) Target {
        return self._platform.target;
    }

    pub fn mlirCtx(self: *const CompilationContext) mlir.Context {
        return self._mlir_ctx;
    }

    fn currentBlock(self: *const CompilationContext) ?Block {
        return if (self._blocks.len > 0) self._blocks.get(self._blocks.len - 1) else null;
    }

    pub fn openBlock(self: *CompilationContext, kind: BlockKind, args: []const mlir.Type, locs: []const mlir.Location) !Block {
        const mlir_block = try mlir.Block.init(args, locs);
        const block: Block = switch (kind) {
            .open => .{ .open = mlir_block },
            .hermetic => .{ .hermetic = mlir_block },
        };
        self.pushBlock(block);
        return block;
    }

    pub fn closeBlock(self: *CompilationContext, block: Block) void {
        const popped = self._blocks.pop();
        std.debug.assert(block.block().eql(popped.block()));
    }

    fn pushBlock(self: *CompilationContext, block: Block) void {
        self._blocks.appendAssumeCapacity(block);
    }

    /// Transform a Tensor -> Tensor function into an Mlir block.
    /// `blkctx` represents values from outside the block that can be accessed inside the block.
    /// Returns both the mlir.Block created and also the Tensors returned by `func`.
    /// The returned tensors should not be returned to the user,
    /// because their `mlir.Value` must not escape the block that created them.
    /// But their shapes/tags can be safely propagated further.
    pub fn makeBlock(
        self: *CompilationContext,
        kind: BlockKind,
        comptime S: ops.BlockSignature,
        func: *const S.Fn,
        blkctx: S.BlkCtx,
        args: S.Args,
    ) struct { mlir.Block, S.Return } {
        const N = S.nIn;
        const loc = self.mlirCtx().location(@src());
        const locations = .{loc} ** N;
        var input_types: [N]mlir.Type = undefined;
        fillMlirTypes(&args, self.mlirCtx(), &input_types);

        // Before creating a new block, assign all received values to previous block,
        // otherwise they will be assign to this block
        if (self.currentBlock()) |prev_block| {
            meta.visit(Block.appendTensorRecursive, prev_block, &blkctx);
        }

        const block = self.openBlock(kind, &input_types, &locations) catch unreachable;
        defer self.closeBlock(block);

        // Here we want to create the block with the correct mlir types.
        // but we don't want to use the values themselves.
        // So we create a copy of the arguments, and replace values
        // by the block arguments.
        var blk_args = args;
        std.debug.assert(assignBlockArguments(&blk_args, block.block(), 0) == N);

        const block_res = @call(.auto, func, S.blkArgs(blkctx, blk_args));
        var block_res_values: [S.nOut]mlir.Value = undefined;
        self.extractValues(&block_res, &block_res_values);
        const block_ret = dialect.stablehlo.returns_(self.mlirCtx(), &block_res_values, loc);
        block.appendOperationRecursive(block_ret);

        return .{ block.block(), block_res };
    }

    /// Generate an MLIR function from a ZML function.
    /// The caller is responsible to have properly created the input
    /// tensors with unique tensor ids.
    pub fn generateBytecode(
        self: *CompilationContext,
        allocator: std.mem.Allocator,
        fn_name: []const u8,
        comptime func: anytype,
        model: *const ModuleSignature(func).ModelT,
        args: *const ModuleSignature(func).ArgsT,
    ) error{OutOfMemory}!MlirFn {
        const frame = self._tracer.frameStart("generateBytecode.emit");
        errdefer self._tracer.frameEnd(frame, "generateBytecode.emit");

        // Note: only temp allocations are done in the arena,
        // the other allocations are managed by the caller.
        var arena_state = std.heap.ArenaAllocator.init(allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        const model_tensor_count = countTensors(model);
        const args_tensor_count = countTensors(args);

        const tensor_count = model_tensor_count + args_tensor_count;

        const mlir_ctx = self.mlirCtx();
        const loc = mlir_ctx.location(@src());

        const locations = try arena.alloc(mlir.Location, tensor_count);
        @memset(locations, mlir.Location.unknown(mlir_ctx));

        var input_shapes = try std.ArrayList(Shape).initCapacity(arena, tensor_count);
        meta.collect(Tensor.shape, {}, &input_shapes, model) catch unreachable;
        stdx.debug.internalAssert(input_shapes.items.len == model_tensor_count, "model has changed ?", .{});
        meta.collect(Tensor.shape, {}, &input_shapes, args) catch unreachable;
        stdx.debug.internalAssert(input_shapes.items.len == tensor_count, "args have changed ?", .{});

        const input_types = try arena.alloc(mlir.Type, tensor_count);
        for (input_types, input_shapes.items) |*t, sh| t.* = mlir.ext.mlirType(mlir_ctx, sh);

        // Note: this isn't stricly necessary. We call `countTensor` on `fn_res`.
        // But it forces user to have simpler function.
        const out_tensor_count = comptime ops.staticCountTensors(ModuleSignature(func).ReturnT) orelse @compileError("Can't use " ++ @typeName(ModuleSignature(func).ReturnT) ++ " in an MLIR function, because it has a variable number of tensors");
        // Those are returned to caller so we don't put them in the arena.
        const fn_res_types = try allocator.alloc(mlir.Type, out_tensor_count);
        const fn_res_shapes = try allocator.alloc(Shape, out_tensor_count);
        const fn_res_donations = try allocator.alloc(Tensor._Donation, out_tensor_count);
        var fn_body = self.openBlock(.hermetic, input_types, locations) catch unreachable;
        {
            defer self.closeBlock(fn_body);
            // Note: we could shrink self._buffer_to_arg once we called `func`.
            // But for now we are only compiling one function per CompilationContext.
            // So we don't need to do this since we won't reuse self._buffer_to_arg anyway.
            // const n = self._buffer_to_arg.count();
            // defer self._buffer_to_arg.shrinkRetainingCapacity(n);

            try self._buffer_to_arg.ensureUnusedCapacity(self._allocator, @intCast(tensor_count));
            const assigned_model_count = self.mapBlockArguments(model, fn_body.block(), 0);
            const assigned_args_count = self.mapBlockArguments(args, fn_body.block(), assigned_model_count);
            assert(assigned_model_count == model_tensor_count);
            assert(assigned_args_count == tensor_count);

            const fn_res = forward: {
                self.activate();
                defer self.deactivate();
                break :forward @call(.auto, func, .{model.*} ++ args.*);
            };

            var fn_res_values: [out_tensor_count]mlir.Value = undefined;
            self.extractValuesAndTypes(&fn_res, &fn_res_values, fn_res_types, fn_res_shapes, fn_res_donations);

            const fn_ret = dialect.func.return_(mlir_ctx, &fn_res_values, loc);
            fn_body.appendOperationRecursive(fn_ret);
        }

        const arg_attrs = try arena.alloc(AttributeList, tensor_count);
        @memset(arg_attrs, .{});

        const res_attrs = try arena.alloc(AttributeList, out_tensor_count);
        @memset(res_attrs, .{});

        // Donations attributes only make sense on the main function.
        self.addDonationsAttributes(arg_attrs, fn_res_donations);

        if (self._platform.sharding().num_partitions > 1) {
            self.addShardingAttributes(arg_attrs, res_attrs, input_shapes.items, fn_res_shapes);
        }
        const mlir_fn = dialect.func.func(self.mlirCtx(), .{
            .sym_name = fn_name,
            .args = input_types,
            .arg_attrs = try finalizeAttributeList(arena, mlir_ctx, arg_attrs),
            .results = fn_res_types,
            .res_attrs = try finalizeAttributeList(arena, mlir_ctx, res_attrs),
            .block = fn_body.block(),
            .location = loc,
        });

        self._tracer.frameEnd(frame, "generateBytecode.emit");
        const canonicalize_frame = self._tracer.frameStart("generateBytecode.canonicalize");
        defer self._tracer.frameEnd(canonicalize_frame, "generateBytecode.canonicalize");
        self._mlir_canonicalizer.runOnOp(mlir_fn) catch |err| switch (err) {
            error.InvalidMlir => {
                log.err("Failed to canonicalize invalid mlir: {}", .{mlir_fn.mlirFormatter(.{})});
                // user errors should have triggered a panic before we reach this.
                @panic("ZML generated invalid mlir. Please open a bug report");
            },
        };

        return .{
            .mlir_fn = mlir_fn,
            .name = fn_name,
            .n_model = @intCast(model_tensor_count),
            .n_args = @intCast(args_tensor_count),
            .res_types = fn_res_types,
            .res_shapes = fn_res_shapes,
            .res_donations = fn_res_donations,
        };
    }

    /// Given a list of donations mapping output buffers to input buffers,
    /// generate donation attribute for each `n_args` input argument.
    fn addDonationsAttributes(self: CompilationContext, attributes: []AttributeList, donations: []const Tensor._Donation) void {
        if (self.target() == .neuron) {
            return;
        }
        var n_donations: usize = 0;
        for (donations, 0..) |donation, index| {
            switch (donation) {
                .no_buffer => {},
                // This is an input buffer that has been returned,
                // but without explicitly calling `reuseBuffer`.
                // So we assume the intent was to return a new buffer.
                .input_buffer => {},
                .arg => |a| {
                    n_donations += 1;
                    // This will break the day we writer another attribute before donation.
                    // When the time come, do a more fancy lookup here to check if an argument
                    // is donated twice.
                    stdx.debug.assert(attributes[a].len == 0, "Donation error ! Argument {} has been donated twice ! To {} and to {}", .{ a, index, attributes[a].buffer[0] });
                    attributes[a].appendAssumeCapacity(
                        mlir.NamedAttribute.init(
                            mlir.Identifier.get(self.mlirCtx(), "tf.aliasing_output"),
                            mlir.IntegerAttribute(.i32).init(self.mlirCtx(), @intCast(index)).as(mlir.Attribute).?,
                        ),
                    );
                    // log.debug("attribute: {}", .{attributes[a].constSlice()});
                },
            }
        }
    }

    test addDonationsAttributes {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const allocator = arena.allocator();

        const s = Shape.init(.{8}, .f16);

        const Local = struct {
            bias: Tensor,

            pub fn forward(self: @This(), x: Tensor) Tensor {
                const y = x.add(self.bias);
                return y.reuseBuffer(x);
            }
        };

        const model: Local = .{
            .bias = zml.Tensor{ ._shape = s, ._id = .{ .buffer_id = 0 } },
        };

        var comp = try zml.module.CompilationContext.init(allocator, "test", platform);
        defer comp.deinit();
        var tensor_args = .{Tensor{ ._shape = s, ._id = .{ .arg_id = 1234 } }};
        const f = try comp.generateBytecode(allocator, "test.generateBytecode.Local.forward", Local.forward, &model, &tensor_args);

        var mlir_bytecode: std.ArrayListUnmanaged(u8) = .{};
        try mlir_bytecode.writer(allocator).print("{}", .{f.mlir_fn.mlirFormatter(.{})});

        // Check that the `x` input argument gives its buffer to the result tensor.
        // `%arg0` is the bias of the model, `%arg1` is `x`.
        try std.testing.expectEqual(1, f.n_model);
        try std.testing.expectEqual(1, f.n_args);
        std.testing.expect(std.mem.indexOf(u8, mlir_bytecode.items, "tf.aliasing_output = 0 : i32") != null) catch |err| {
            log.warn("Didn't produced the expected IR:\n{s}", .{mlir_bytecode.items});
            return err;
        };
    }

    pub fn getShardingAttr(self: CompilationContext, shape: Shape) mlir.StringAttribute {
        const mlir_ctx = self.mlirCtx();

        const num_partitions = self._platform.sharding().num_partitions;
        var sharding_str: std.BoundedArray(u8, 128) = .{};

        writeShardingRepresentation(shape, num_partitions, sharding_str.writer()) catch unreachable;
        return mlir.StringAttribute.init(mlir_ctx, sharding_str.constSlice());
    }

    fn addShardingAttributes(self: CompilationContext, arg_attrs: []AttributeList, res_attrs: []AttributeList, input_shapes: []const Shape, output_shapes: []const Shape) void {
        const mlir_ctx = self.mlirCtx();
        if (!self._platform.compilation_options.sharding_enabled) return;

        const mhlo_default_layout = mlir.NamedAttribute.init(
            mlir.Identifier.get(mlir_ctx, "mhlo.layout_mode"),
            mlir.StringAttribute.init(mlir_ctx, "default").asAttr(),
        );
        for (arg_attrs, input_shapes) |*attr, shape| {
            attr.appendAssumeCapacity(mhlo_default_layout);

            const sharding_attr = self.getShardingAttr(shape);
            attr.appendAssumeCapacity(mlir.NamedAttribute.init(
                mlir.Identifier.get(mlir_ctx, "mhlo.sharding"),
                sharding_attr.asAttr(),
            ));
        }

        for (res_attrs, output_shapes) |*attr, shape| {
            attr.appendAssumeCapacity(mhlo_default_layout);

            const sharding_attr = self.getShardingAttr(shape);

            attr.appendAssumeCapacity(mlir.NamedAttribute.init(
                mlir.Identifier.get(mlir_ctx, "mhlo.sharding"),
                sharding_attr.asAttr(),
            ));
        }
    }

    fn writeShardingRepresentation(shape: Shape, num_partitions: u8, writer: anytype) @TypeOf(writer).Error!void {
        const n_sharded: u8 = @popCount(@as(u8, @bitCast(shape._sharding_info)));
        if (n_sharded == 0 or num_partitions == 1) {
            try writer.writeAll("{replicated}");
            return;
        }
        try writer.writeAll("{devices=[");
        for (0..shape.rank()) |i| {
            try writer.print("{d}", .{if (shape._sharding_info[i]) num_partitions else 1});
            if (i < shape.rank() - 1) try writer.writeByte(',');
        }
        try writer.print("]<=[{d}]}}", .{num_partitions});
    }

    test writeShardingRepresentation {
        var rule: [64]u8 = undefined;
        const x = Shape.init(.{ 16, 8 }, .f32);

        // By default tensors are replicated.
        {
            var fbs = std.io.fixedBufferStream(&rule);
            try writeShardingRepresentation(x, 4, fbs.writer());
            try std.testing.expectEqualStrings("{replicated}", fbs.getWritten());
        }
        // Shard along first axis.
        {
            var fbs = std.io.fixedBufferStream(&rule);
            try writeShardingRepresentation(x.withSharding(.{0}), 4, fbs.writer());
            try std.testing.expectEqualStrings("{devices=[4,1]<=[4]}", fbs.getWritten());
        }
        // Also shard along second axis.
        {
            var fbs = std.io.fixedBufferStream(&rule);
            try writeShardingRepresentation(x.withSharding(.{ 0, 1 }), 2, fbs.writer());
            try std.testing.expectEqualStrings("{devices=[2,2]<=[2]}", fbs.getWritten());
        }
    }

    fn finalizeAttributeList(allocator: std.mem.Allocator, mlir_ctx: mlir.Context, attributes: []AttributeList) ![]mlir.Attribute {
        const res = try allocator.alloc(mlir.Attribute, attributes.len);
        for (res, attributes) |*r, attr| {
            r.* = mlir.DictionaryAttribute.init(mlir_ctx, attr.constSlice()).asAttr();
        }
        return res;
    }

    /// Generates an MLIR `func.call` of the given function.
    /// If the function has not been seen yet, we generate MLIR for it,
    /// in a independent function.
    /// The main benefit of this is to generate MLIR that maps more closely
    /// to the Zig code, but compilation speed stays similar.
    pub fn callFunc(
        self: *CompilationContext,
        func_name: [:0]const u8,
        comptime func: anytype,
        model: *const ModuleSignature(func).ModelT,
        args: *ModuleSignature(func).ArgsT,
    ) ModuleSignature(func).ReturnT {
        var arena_state = std.heap.ArenaAllocator.init(self._allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        // first, do the "compile" and check the bytecode
        // the result of this will also have the correct tags of the result shapes
        const dummy_result = self.generateMlirBytecodeForFunction(
            arena,
            func_name,
            func,
            model,
            args,
        ) catch unreachable; // TODO: do we like unreachable?
        const bytecode_hash = hashArgs(dummy_result.bytecode_tmp);

        const key: FnCache.Key = .{ .fn_ptr = &func, .input_hash = bytecode_hash };
        const function = self._fn_cache.getEntry(key) orelse b: {
            const full_name: [:0]const u8 = if (std.mem.eql(u8, "main", func_name))
                arena.dupeZ(u8, func_name) catch unreachable
            else
                std.fmt.allocPrintZ(arena, "{s}.{s}_{x}", .{ @typeName(ModuleSignature(func).ModelT), func_name, key.input_hash }) catch unreachable;

            log.info("addFuncToModule {any} {s}", .{ key, full_name });

            const value = self.addFuncToModule(
                arena,
                full_name,
                func,
                model,
                args,
            ) catch unreachable;

            break :b self._fn_cache.addEntry(key, value) catch unreachable;
        };

        // Note: we won't increase the size of the cache until next `call` so
        // we can use the memory there without worrying about fragmentation.

        const loc = self.mlirCtx().location(@src());

        const values = arena.alloc(mlir.Value, function.n_model + function.n_args) catch unreachable;
        self.extractValues(&model, values[0..function.n_model]);
        self.extractValues(&args, values[function.n_model..]);

        const op = dialect.func.call(self.mlirCtx(), function.name, values, function.res_types, loc);
        // TODO: tags seem to be lost by `callFunc`.
        var res: stdx.meta.FnResult(func) = undefined;
        assignResults(op, &res, function.res_shapes);
        return res;
    }

    /// Visit the given struct and recursively associate the `block` arguments with the `value` field of each encountered Tensor.
    ///
    /// This is done so that we have a mapping between the arguments of the kernel associated with a module and the actual Tensors
    /// stored in the Module.
    /// Caller need to allocate required memory in self._buffer_to_arg.
    pub fn mapBlockArguments(self: *CompilationContext, v: anytype, block: mlir.Block, start: usize) usize {
        const LocalContext = struct {
            index: usize,
            block: mlir.Block,
            self: *CompilationContext,
        };
        var context = LocalContext{ .self = self, .block = block, .index = start };
        meta.visit((struct {
            fn cb(ctx: *LocalContext, tensor: *const Tensor) void {
                const arg_value = ctx.block.argument(ctx.index);
                // log.debug("mapping {} to arg {}", .{ tensor._id, ctx.index });

                const res = ctx.self._buffer_to_arg.getOrPutAssumeCapacity(tensor._id);
                if (res.found_existing) {
                    stdx.debug.panic("Failed compilation because received two tensors arguments with the same ID: {} and {}({}).", .{ res.key_ptr.*, tensor, tensor._id });
                } else {
                    res.value_ptr.* = .{ arg_value, .{ .arg = @intCast(ctx.index) } };
                }
                ctx.index += 1;
            }
        }).cb, &context, v);
        return context.index;
    }

    /// Create tensor from the given shapes.
    /// Each created tensor will receive a unique id, local to this CompilationContext.
    pub fn tensorFromShapes(self: *CompilationContext, ArgsT: type, allocator: std.mem.Allocator, args_shapes: anytype) ArgsT {
        const Local = struct {
            fn tensorFromShape(arg_id: *u64, shape: Shape) Tensor {
                defer arg_id.* += 1;
                return Tensor{
                    ._shape = shape,
                    ._id = .{ .arg_id = arg_id.* },
                    ._donation = .input_buffer,
                };
            }
        };
        var tensor_args: ArgsT = undefined;
        try meta.mapAlloc(Local.tensorFromShape, allocator, &self._unique_id, args_shapes, &tensor_args);
        return tensor_args;
    }

    /// Visit the given struct and extract the mlir.Value and mlir.Type associated with each tensor found.
    pub fn extractValuesAndTypes(self: *const CompilationContext, v: anytype, values: []mlir.Value, types: []mlir.Type, shapes: []Shape, donations: []Tensor._Donation) void {
        assert(values.len == types.len);
        const LocalContext = struct {
            self: *const CompilationContext,
            index: usize = 0,
            values: []mlir.Value,
            types: []mlir.Type,
            shapes: []Shape,
            donations: []Tensor._Donation,
        };
        var context = LocalContext{ .self = self, .values = values, .types = types, .shapes = shapes, .donations = donations };
        meta.visit((struct {
            fn cb(ctx: *LocalContext, tensor: *const Tensor) void {
                const value, const donation = ctx.self.getValueAndDonation(tensor.*);
                ctx.values[ctx.index] = value;
                ctx.types[ctx.index] = value.getType();
                ctx.shapes[ctx.index] = tensor._shape;
                ctx.donations[ctx.index] = donation;
                ctx.index += 1;
            }
        }).cb, &context, v);
        assert(context.index == values.len);
    }

    pub fn getValueAndDonation(self: *const CompilationContext, tensor: Tensor) struct { mlir.Value, Tensor._Donation } {
        return switch (tensor._id) {
            .buffer_id, .arg_id => if (self._buffer_to_arg.get(tensor._id)) |res|
                .{ res[0], res[1] }
            else {
                log.err("Found unknown tensor id {}({})", .{ tensor, tensor._id });
                @panic("Found unknown tensor id");
            },
            .mlir => |v| .{ v, tensor._donation },
        };
    }

    fn getValue(self: *const CompilationContext, tensor: Tensor) mlir.Value {
        return self.getValueAndDonation(tensor)[0];
    }

    pub fn extractValues(self: *const CompilationContext, v: anytype, values: []mlir.Value) void {
        meta.collectBuf(getValue, self, v, values);
    }
};

/// Visit the given struct and recursively counts the number of tensors found.
pub fn countTensors(v: anytype) usize {
    const LocalContext = struct {
        count: usize = 0,
    };
    var context = LocalContext{};
    meta.visit((struct {
        fn cb(inner_context: *LocalContext, _: *const Tensor) void {
            inner_context.count += 1;
        }
    }).cb, &context, v);
    return context.count;
}

/// Visit the given struct and recursively fill the `types` slice with the mlir.Type associated with encountered Tensor.
pub fn fillMlirTypes(v: anytype, mlir_ctx: mlir.Context, types: []mlir.Type) void {
    const LocalContext = struct {
        index: usize = 0,
        mlir_ctx: mlir.Context,
        types: []mlir.Type,
    };
    var context = LocalContext{ .mlir_ctx = mlir_ctx, .types = types };
    meta.visit((struct {
        fn cb(inner_context: *LocalContext, tensor: *const Tensor) void {
            inner_context.types[inner_context.index] = mlir.ext.mlirType(inner_context.mlir_ctx, tensor.shape());
            inner_context.index += 1;
        }
    }).cb, &context, v);
    assert(context.index == types.len);
}

/// Visit the given struct and recursively associate the `block` arguments with the `value` field of each encountered Tensor.
///
/// This is done so that we have a mapping between the arguments of the kernel associated with a module and the actual Tensors
/// stored in the Module.
fn assignBlockArguments(v: anytype, block: mlir.Block, start: usize) usize {
    const LocalContext = struct { index: usize, block: mlir.Block };
    var context = LocalContext{ .block = block, .index = start };
    meta.visit((struct {
        fn cb(ctx: *LocalContext, tensor: *Tensor) void {
            tensor._id = .{ .mlir = ctx.block.argument(ctx.index) };
            tensor._donation = .{ .arg = @intCast(ctx.index) };
            ctx.index += 1;
        }
    }).cb, &context, v);
    return context.index;
}

/// Visit the given struct and fill the `buffers` slice with the buffer associated with encountered Tensor.
fn fillBuffers(v: anytype, buffers: []const [*]*pjrt.Buffer, start: u32) u32 {
    const LocalContext = struct {
        index: u32,
        buffers: []const [*]*pjrt.Buffer,
    };
    var context: LocalContext = .{
        .index = start,
        .buffers = buffers,
    };
    meta.visit((struct {
        fn cb(ctx: *LocalContext, buffer: *const Buffer) void {
            // stdx.debug.assert(!buffer._data.isDeleted(), "Can't use {} (argument buffer {}) because its pjrt buffer has been donated", .{ buffer, ctx.index });
            const model_sharding = ctx.buffers.len;
            stdx.debug.assert(buffer._shards.len == model_sharding, "Can't feed a {}-sharded tensor into a {}-sharded model", .{ buffer._shards.len, ctx.buffers.len });
            for (buffer._shards.constSlice(), 0..) |shard, d| {
                ctx.buffers[d][ctx.index] = shard;
            }
            ctx.index += 1;
        }
    }).cb, &context, v);
    return context.index;
}

/// Visit the given struct and override tensors by creating a new one using the provided PJRT buffers.
pub fn assignRawBuffers(v: anytype, platform: Platform, buffers: []const [*]*pjrt.Buffer, buffer_shapes: []Shape) void {
    const LocalContext = struct {
        index: u32,
        platform: Platform,
        buffers: []const [*]*pjrt.Buffer,
        buffer_shapes: []Shape,
    };
    var local_ctx: LocalContext = .{
        .index = 0,
        .platform = platform,
        .buffers = buffers,
        .buffer_shapes = buffer_shapes,
    };
    meta.visit((struct {
        fn cb(ctx: *LocalContext, buffer: *Buffer) void {
            const i = ctx.index;
            ctx.index += 1;
            if (i >= ctx.buffer_shapes.len) return;

            var shards: Buffer.Shards = .{};
            for (ctx.buffers) |buff| {
                shards.appendAssumeCapacity(buff[i]);
            }
            buffer.* = Buffer.fromPjrtBuffers(ctx.platform, ctx.buffer_shapes[i], shards.constSlice());
        }
    }).cb, &local_ctx, v);
    stdx.debug.internalAssert(local_ctx.index == buffer_shapes.len, "Pjrt call returned {} tensors, but the return type {s}, contains {} Buffers. Note that modules need to have a comptime know number of returned tensors.", .{ buffers.len, @typeName(@TypeOf(v)), local_ctx.index });
}

/// Visit the given struct and assign op results to each tensor found.
fn assignResults(op: mlir.Operation, v: anytype, shapes: []Shape) void {
    const LocalContext = struct {
        index: usize,
        op: mlir.Operation,
        shapes: ?[]Shape,
    };
    var context = LocalContext{ .index = 0, .op = op, .shapes = shapes };
    meta.visit((struct {
        fn cb(inner_ctx: *LocalContext, tensor: *Tensor) void {
            var new = Tensor.fromMlirValue(inner_ctx.op.result(inner_ctx.index));
            if (inner_ctx.shapes) |sh| {
                new._shape = sh[inner_ctx.index];
            } else {
                new._shape._tags = tensor._shape._tags;
            }
            tensor.* = new;
            inner_ctx.index += 1;
        }
    }).cb, &context, v);
    assert(context.index == op.numResults());
}

/// Represents an MLIR module compiled into a PJRT executable.
/// The BaseExe is a plain old struct and doesn't have information
/// about Zig types.
const BaseExe = struct {
    /// The platform for which this module was compiled.
    platform: Platform,

    /// The PJRT executable representing the compiled module.
    exe: *pjrt.LoadedExecutable,

    /// Pre-allocated slice of buffers to use as inputs when the module is called.
    input_per_device: []const [*]*pjrt.Buffer,

    /// Pre-allocated slice of buffers to use as outputs when the module is called.
    output_per_device: []const [*]*pjrt.Buffer,

    /// Number of buffers already fed to the executable.
    ready_buffer_count: u32,

    /// Total number of buffers needed by this executable.
    input_buffer_count: u32,

    result_shapes: []Shape,

    /// Num devices used (>1 for sharded executable)
    num_devices: u8,

    /// Allocator backing memory
    _arena: std.heap.ArenaAllocator,

    pub fn init(parent_allocator: std.mem.Allocator, platform: Platform, exe: *pjrt.LoadedExecutable, args: struct { n_in: u32, result_shapes: []const Shape, n_devices: u8 }) !BaseExe {
        var arena = std.heap.ArenaAllocator.init(parent_allocator);
        errdefer arena.deinit();
        const allocator = arena.allocator();
        const n_out = args.result_shapes.len;
        const n_devices = args.n_devices;
        // Allocate once for all the *pjrt.Buffer we need to store ...
        const all_buffers = try allocator.alloc(*pjrt.Buffer, (args.n_in + n_out) * n_devices);
        const all_input_buffers, const all_output_buffers = splitBuffer(*pjrt.Buffer, all_buffers, .{ args.n_in * n_devices, n_out * n_devices });

        // ... and once for all the [*]*pjrt.Buffer.
        const all_per_device = try allocator.alloc([*]*pjrt.Buffer, 2 * n_devices);
        const input_per_device, const output_per_device = splitBuffer([*]*pjrt.Buffer, all_per_device, .{ n_devices, n_devices });

        for (0..n_devices) |i| {
            input_per_device[i] = all_input_buffers[i * args.n_in ..].ptr;
            output_per_device[i] = all_output_buffers[i * n_out ..].ptr;
        }

        return .{
            .platform = platform,
            .exe = exe,
            .ready_buffer_count = 0,
            .input_buffer_count = args.n_in,
            .num_devices = args.n_devices,
            .input_per_device = input_per_device,
            .output_per_device = output_per_device,
            .result_shapes = try allocator.dupe(Shape, args.result_shapes),
            ._arena = arena,
        };
    }

    pub fn deinit(self: BaseExe) void {
        self._arena.deinit();
    }

    pub fn call(self: BaseExe) void {
        stdx.debug.assert(self.input_buffer_count == self.ready_buffer_count, "BaseExe isn't ready to be called, expected {} buffer inputs got {}", .{ self.input_buffer_count, self.ready_buffer_count });
        return self._unsafeCall();
    }

    pub fn _unsafeCall(self: BaseExe) void {
        var events = [_]?*pjrt.Event{null} ** Platform.MAX_NUM_DEVICES;
        const sharding = self.platform.sharding();

        self.exe.execute(self.platform.pjrt_api, .{
            .arguments = self.input_per_device,
            .num_args = self.input_buffer_count,
            .results = self.output_per_device,
            .events = events[0..sharding.num_partitions],
            // this allows to tell a specific buffer shouldn't be donated,
            // even if it has been marked as "can be donated" during compilation.
            // TODO: expose it ?
            .non_donatable_input_indices = &.{},
        }) catch unreachable;

        for (events[0..sharding.num_partitions]) |e| {
            if (e) |ev| {
                ev.await_(self.platform.pjrt_api) catch unreachable;
            }
        }
    }

    pub fn serialize(self: BaseExe, writer: anytype) !void {
        var executable = try self.exe.getExecutable(self.platform.pjrt_api);
        var serialize_result = try executable.serialize(self.platform.pjrt_api);
        defer serialize_result.deinit();
        try writer.writeAll(serialize_result.bytes);
    }

    // pub fn deserialize(allocator: std.mem.Allocator, platform: Platform, reader: anytype) !Self {
    //     const bytes = try reader.readToEndAlloc(allocator, max_pjrt_executable_size);
    //     defer allocator.free(bytes);
    //     return platform.pjrt_client.deserializeAndLoad(platform.pjrt_api, bytes);
    // }

    pub fn feed(self: *BaseExe, x: anytype) void {
        const n = fillBuffers(&x, self.ready_buffer_count);
        self.ready_buffer_count += n;
    }

    pub fn getOutputBuffer(self: BaseExe, i: usize) Buffer {
        var shards: Buffer.Shards = .{};
        for (self.output_per_device) |dev_out| {
            shards.appendAssumeCapacity(dev_out[i]);
        }

        const out_shape = self.inner.result_buffer_shapes[i];
        return Buffer.fromPjrtBuffers(self.platform(), out_shape, shards.constSlice());
    }
};

/// Represents a ZML model, compiled into a PJRT executable.
///
/// It's not directly callable, as it doesn't have associated model weights.
/// use `prepare` to assign weights and pre allocate memory needed to call.
pub fn Exe(ArgsT: type, ReturnT: type) type {
    return struct {
        const Self = @This();

        /// The raw untyped compiled module.
        inner: BaseExe,

        pub fn deinit(self: Self) void {
            self.inner.deinit();
        }

        pub fn bakeArg(self: *Self, first_arg: ArgsT[0]) !Exe(ArgsT[1..], ReturnT) {
            var new: Exe(ArgsT[1..], ReturnT) = .{ .inner = self.inner };
            new.inner.bake(first_arg);
            return new;
        }

        pub fn serialize(self: Self, writer: anytype) !void {
            return try self.inner.serialize(writer);
        }

        pub fn platform(self: Self) Platform {
            return self.inner.platform;
        }

        pub fn call(self: Self, args: Bufferized(ArgsT)) Bufferized(ReturnT) {
            const n_args = fillBuffers(&args, self.inner.input_per_device, self.inner.ready_buffer_count);
            std.debug.assert(self.inner.ready_buffer_count + n_args == self.inner.input_buffer_count);
            self.inner._unsafeCall();
            var result: Bufferized(ReturnT) = undefined;
            assignRawBuffers(&result, self.inner.platform, self.inner.output_per_device, self.inner.result_shapes);
            return result;
        }
    };
}

/// Represents a ZML model, compiled into a PJRT executable, and ready to call.
/// The buffers for the model weights are saved inside the struct and will be used in `call`.
/// You only need to pass the remaining arguments.
pub fn ModuleExe(comptime func: anytype) type {
    const sign = ModuleSignature(func);
    return Exe(sign.ArgsT, sign.ReturnT);
}

/// Compiles the given module with the given arguments.
/// The `model` (first fn argument), is treated differently from the other args.
/// This helps to have two separate lifetimes for the model buffers,
/// and for the arguments buffer.
fn compileInternal(
    allocator: std.mem.Allocator,
    context: *CompilationContext,
    comptime func: anytype,
    model: ModuleSignature(func).ModelT,
    args: ShapeOf(ModuleSignature(func).ArgsT),
) !BaseExe {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var timer = std.time.Timer.start() catch null;
    const tensor_args = context.tensorFromShapes(ModuleSignature(func).ArgsT, arena, args);
    // Run in a dedicated thread because compilation relies on `threadlocal`.
    const f = try asynk.callBlocking(CompilationContext.generateBytecode, .{ context, arena, "main", func, &model, &tensor_args });
    context._module.getBody().appendOperation(f.mlir_fn);

    const sharding = context._platform.sharding();
    const mlir_ctx = context._mlir_ctx;
    context._module.op().setAttributeByName("mhlo.num_replicas", mlir.IntegerAttribute(.i32).init(mlir_ctx, sharding.num_replicas).asAttr());
    context._module.op().setAttributeByName("mhlo.num_partitions", mlir.IntegerAttribute(.i32).init(mlir_ctx, sharding.num_partitions).asAttr());

    if (context._platform.compilation_options.xla_dump_to) |xla_dump_to| {
        // Write the mlir to a file. All errors are discarded, since this is for debugging only.
        if (std.fs.openDirAbsolute(xla_dump_to, .{})) |dir| {
            const name = context._name;
            const file_name = std.fmt.allocPrint(arena, "{s}.mlir", .{name}) catch name;
            if (dir.createFile(file_name, .{ .truncate = true })) |file| {
                context._module.op().print(file.writer(), .{ .debug_info = true, .debug_info_pretty_form = false });
                log.info("Wrote MLIR to {s}/{s}", .{ xla_dump_to, file_name });
            } else |_| {
                log.warn("Failed to open {s}", .{file_name});
            }
        } else |_| {
            log.warn("Folder not found {s}", .{xla_dump_to});
        }
    }

    const loaded_executable = loadOrCompilePjrtExecutable(arena, context._platform, context._module) catch |err| {
        log.err(
            "pjrt-{s} failed to compile following valid MLIR:\n{}\n{}",
            .{ @tagName(context._platform.target), context._module.op().mlirFormatter(.{}), err },
        );
        return err;
    };

    log.debug("******** ZML generated MLIR ********", .{});
    log.debug("{}", .{context._module.op().mlirFormatter(.{})});

    if (timer) |*t| {
        const time_ms = @divFloor(t.lap(), std.time.ns_per_ms);
        if (time_ms > 1000) log.info("Compilation took {d:.3}s", .{stdx.math.divFloat(f32, time_ms, 1000)});
    }

    return BaseExe.init(
        allocator,
        context._platform,
        loaded_executable,
        .{
            .n_in = f.n_model + f.n_args,
            .result_shapes = f.res_shapes,
            .n_devices = sharding.num_replicas * sharding.num_partitions,
        },
    );
}

pub fn load(
    allocator: std.mem.Allocator,
    comptime Model: type,
    init_args: anytype,
    comptime func: @TypeOf(.literal),
    args_shapes: ShapeOf(ModuleSignature(@field(Model, @tagName(func))).ArgsT),
    buffer_store: aio.BufferStore,
    platform: Platform,
) !ModuleExe(@field(Model, @tagName(func))) {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var model = try aio.populateModel(Model, arena, buffer_store);

    // If the Model has a "init" function, call it with the given parameters.
    if (@hasDecl(Model, "init")) {
        // TODO(Corentin,@Improvement): Add a warning/error if there is no init function but init_args is non-void.
        @call(.auto, Model.init, .{@as(*Model, &model)} ++ init_args);
    }

    return compileModel(allocator, model, func, args_shapes, platform);
}

/// Compiles a Model struct with the given configuration and shapes, for the given platform.
/// The steps are:
/// * lookup at tensors available in the store and create a `model: Model` struct with them
/// * call `model.init(init_args)` to fields of the model that aren't Tensor, ie hyperparemeters/config
/// * generate MLIR by calling `model.forward` with tensor of the given shapes and other arguments
pub fn compile(
    allocator: std.mem.Allocator,
    comptime func: anytype,
    init_args: anytype,
    args_shapes: ShapeOf(ModuleSignature(func).ArgsT),
    buffer_store: aio.BufferStore,
    platform: Platform,
) !ModuleExe(func) {
    const ModelT = ModuleSignature(func).ModelT;

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var model = try aio.populateModel(ModelT, arena, buffer_store);

    // If the Model has a "init" function, call it with the given parameters.
    if (@hasDecl(ModelT, "init")) {
        // TODO(Corentin,@Improvement): Add a warning/error if there is no init function but init_args is non-void.
        @call(.auto, ModelT.init, .{@as(*ModelT, &model)} ++ init_args);
    }

    return compileModel(allocator, func, model, args_shapes, platform);
}

/// Compiles a Model struct with the given configuration and shapes, for the given platform.
/// Generate MLIR by calling `model.forward` with tensor of the given shapes and other arguments
pub fn compileModel(
    allocator: std.mem.Allocator,
    comptime func: anytype,
    model: ModuleSignature(func).ModelT,
    args_shapes: ShapeOf(ModuleSignature(func).ArgsT),
    platform: Platform,
) !ModuleExe(func) {
    const ModelT = ModuleSignature(func).ModelT;
    const name = @typeName(ModelT) ++ ".forward";
    log.info("Compiling {s} with {}", .{ name, args_shapes });

    var context = try CompilationContext.init(allocator, name, platform);
    defer context.deinit();

    const raw_module = try compileInternal(allocator, &context, func, model, args_shapes);

    return .{ .inner = raw_module };
}

/// Compiles a function with the given configuration and shapes, for the given platform.
/// Generate MLIR by calling the given function with tensor of the given shapes.
pub fn compileFn(
    allocator: std.mem.Allocator,
    comptime func: anytype,
    args: ShapeOf(stdx.meta.FnArgs(func)),
    platform: Platform,
) !FnExe(func) {
    const name = @typeName(@TypeOf(func));
    var context = try CompilationContext.init(allocator, name, platform);
    defer context.deinit();

    const Local = struct {
        // This is the function we will actually compile.
        pub fn forward(_: void, inner_args: stdx.meta.FnArgs(func)) stdx.meta.FnResult(func) {
            return @call(.auto, func, inner_args);
        }
    };

    const void_model: void = {};
    const base_exe = try compileInternal(allocator, &context, Local.forward, void_model, .{args});
    // But we set the signature so that you can call the module as you would call the function.
    return .{ .inner = base_exe };
}

pub fn FnExe(comptime func: anytype) type {
    return Exe(stdx.meta.FnArgs(func), stdx.meta.FnResult(func));
}

fn computeModuleHash(platform: Platform, module: mlir.Module) u64 {
    var hasher = std.hash.XxHash64.init(0);
    var hasher_writer = xxHash64Writer(&hasher);
    const writer = hasher_writer.writer();

    // Hash the canonicalized IR, without debug information that can change across builds.
    module.op().writeBytecode(writer);
    //module.op().print(writer, .{ .debug_info = false });
    // Writes can't fail because we are writing to a hasher.
    writer.writeAll(platform.pjrt_client.getPlatformName(platform.pjrt_api)) catch unreachable;
    const api_version = platform.pjrt_api.version();
    writer.writeInt(i64, api_version.major, .little) catch unreachable;
    writer.writeInt(i64, api_version.minor, .little) catch unreachable;

    return hasher.final();
}

const max_pjrt_executable_size = 400 * 1024 * 1024;

fn loadPjrtExecutable(arena: std.mem.Allocator, platform: Platform, module_hash: u64, compilation_cache_location: []const u8) !*pjrt.LoadedExecutable {
    const resolved_path = try std.fs.cwd().realpathAlloc(arena, compilation_cache_location);
    const compilation_cache_dir = try std.fs.openDirAbsolute(resolved_path, .{});
    var buf: [16]u8 = undefined;
    const filename = try std.fmt.bufPrint(&buf, "{x}", .{module_hash});
    const loaded_executable_file = try compilation_cache_dir.openFile(filename, .{});
    defer loaded_executable_file.close();

    const bytes = try loaded_executable_file.readToEndAlloc(arena, max_pjrt_executable_size);

    return platform.pjrt_client.deserializeAndLoad(platform.pjrt_api, bytes);
}

fn storePjrtExecutable(arena: std.mem.Allocator, platform: Platform, loaded_executable: *pjrt.LoadedExecutable, module_hash: u64, compilation_cache_location: []const u8) !void {
    const resolved_path = try std.fs.cwd().realpathAlloc(arena, compilation_cache_location);
    const compilation_cache_dir = std.fs.openDirAbsolute(resolved_path, .{}) catch blk: {
        try std.fs.makeDirAbsolute(resolved_path);
        break :blk try std.fs.openDirAbsolute(resolved_path, .{});
    };

    const loaded_executable_file = try compilation_cache_dir.createFile(try std.fmt.allocPrint(arena, "{x}", .{module_hash}), .{});
    defer loaded_executable_file.close();

    var executable = try loaded_executable.getExecutable(platform.pjrt_api);
    defer executable.deinit(platform.pjrt_api);

    var serialize_result = try executable.serialize(platform.pjrt_api);
    defer serialize_result.deinit();

    try loaded_executable_file.writeAll(serialize_result.bytes);
}

fn loadOrCompilePjrtExecutable(
    arena: std.mem.Allocator,
    platform: Platform,
    module: mlir.Module,
) !*pjrt.LoadedExecutable {
    const tracer = Tracer.init("ai.zml.compilation");
    const compile_frame = tracer.frameStart("pjrt cached compilation");
    defer tracer.frameEnd(compile_frame, "pjrt cached compilation");
    const module_hash = computeModuleHash(platform, module);

    if (platform.compilation_options.cache_location) |compilation_cache_location| {
        log.debug("Loading module from {s}", .{compilation_cache_location});
        return loadPjrtExecutable(arena, platform, module_hash, compilation_cache_location) catch |err| {
            log.debug("Failed to load module: {}", .{err});
            return compileModuleToPjrtExecutable(arena, platform, module, module_hash);
        };
    } else {
        return compileModuleToPjrtExecutable(arena, platform, module, module_hash);
    }
}

fn compileModuleToPjrtExecutable(arena: std.mem.Allocator, platform: Platform, module: mlir.Module, module_hash: u64) !*pjrt.LoadedExecutable {
    const sharding = platform.sharding();

    // NOTE(Corendos): Hack needed because Protobuf struct are not public.
    const DeviceAssignmentProto = @TypeOf(xla_pb.CompileOptionsProto.init().executable_build_options.?.device_assignment.?);
    var options: xla_pb.CompileOptionsProto = .{
        .executable_build_options = .{
            .device_ordinal = -1,
            .num_replicas = sharding.num_replicas,
            .num_partitions = sharding.num_partitions,
            .use_spmd_partitioning = sharding.num_partitions > 1 or sharding.num_replicas > 1,
            .device_assignment = .{
                .replica_count = sharding.num_replicas,
                .computation_count = sharding.num_partitions,
                .computation_devices = blk: {
                    var computation_devices = try std.ArrayListUnmanaged(DeviceAssignmentProto.ComputationDevice).initCapacity(arena, sharding.num_partitions);
                    for (0..sharding.num_partitions) |i| {
                        var replica_device_ids = std.ArrayListUnmanaged(i64).initCapacity(arena, 1) catch unreachable;
                        replica_device_ids.appendAssumeCapacity(@intCast(i));
                        computation_devices.appendAssumeCapacity(.{ .replica_device_ids = replica_device_ids });
                    }
                    break :blk computation_devices;
                },
            },
        },
    };

    // Let the arena deinit, zig-protobuf deinit is very slow.
    if (platform.compilation_options.xla_dump_to) |xla_dump_to| {
        try options.env_option_overrides.append(arena, .{
            .key = .{ .Const = "xla_dump_to" },
            .value = .{ .value = .{ .string_field = .{ .Const = xla_dump_to } } },
        });
        if (platform.compilation_options.xla_dump_fusion_visualization) {
            try options.env_option_overrides.append(arena, .{
                .key = .{ .Const = "xla_dump_hlo_as_html" },
                .value = .{ .value = .{ .bool_field = true } },
            });
            try options.env_option_overrides.append(arena, .{
                .key = .{ .Const = "xla_dump_hlo_as_dot" },
                .value = .{ .value = .{ .bool_field = true } },
            });
            try options.env_option_overrides.append(arena, .{
                .key = .{ .Const = "xla_dump_fusion_visualization" },
                .value = .{ .value = .{ .bool_field = true } },
            });
        }
    }
    switch (platform.target) {
        .cuda => cuda_dir: {
            // NVIDIA recommends to disable Triton GEMM on JAX:
            // https://github.com/NVIDIA/JAX-Toolbox?tab=readme-ov-file#environment-variables
            try options.env_option_overrides.append(arena, .{
                .key = .{ .Const = "xla_gpu_enable_triton_gemm" },
                .value = .{ .value = .{ .bool_field = false } },
            });
            // try options.env_option_overrides.append(arena, .{
            //     .key = .{ .Const = "xla_gpu_enable_latency_hiding_scheduler" },
            //     .value = .{ .value = .{ .bool_field = true } },
            // });
            var r_ = try runfiles.Runfiles.create(.{ .allocator = arena }) orelse {
                log.warn("Bazel runfile not found !", .{});
                break :cuda_dir;
            };
            defer r_.deinit(arena);
            const source_repo = @import("bazel_builtin").current_repository;
            const r = r_.withSourceRepo(source_repo);
            const cuda_data_dir = (try r.rlocationAlloc(arena, "libpjrt_cuda/sandbox")).?;
            log.info("xla_gpu_cuda_data_dir: {s}", .{cuda_data_dir});
            try options.env_option_overrides.append(arena, .{
                .key = .{ .Const = "xla_gpu_cuda_data_dir" },
                .value = .{
                    .value = .{
                        .string_field = .{ .Const = cuda_data_dir },
                    },
                },
            });
        },
        .rocm => {
            // Disable Triton GEMM on ROCM. For some reason it's much, much slower when
            // enabled on CDNA and it's used on RDNA. Disable it altogether.
            try options.env_option_overrides.append(arena, .{
                .key = .{ .Const = "xla_gpu_enable_triton_gemm" },
                .value = .{ .value = .{ .bool_field = false } },
            });
        },
        else => {},
    }

    const options_bytes = try options.encode(arena);

    const loaded_executable = try platform.pjrt_client.compile(platform.pjrt_api, arena, module, options_bytes);
    errdefer loaded_executable.deinit();

    if (platform.compilation_options.cache_location) |compilation_cache_location| {
        log.debug("Storing module to {s}", .{compilation_cache_location});
        storePjrtExecutable(arena, platform, loaded_executable, module_hash, compilation_cache_location) catch |err| {
            log.debug("Failed to store module: {}", .{err});
        };
    }

    return loaded_executable;
}

pub const XxHash64Writer = struct {
    hasher: *std.hash.XxHash64,

    pub const Error = error{};
    pub const Writer = std.io.Writer(*XxHash64Writer, Error, write);

    pub fn writer(self: *XxHash64Writer) Writer {
        return .{ .context = self };
    }

    pub fn write(self: *XxHash64Writer, bytes: []const u8) Error!usize {
        self.hasher.update(bytes);
        return bytes.len;
    }
};

pub fn xxHash64Writer(hasher: *std.hash.XxHash64) XxHash64Writer {
    return .{ .hasher = hasher };
}

pub fn hasTensors(comptime T: type) bool {
    if (T == Tensor) return true;

    return switch (@typeInfo(T)) {
        inline .Array, .Pointer, .Optional => |info| hasTensors(info.child),
        inline .Struct, .Union => |info| {
            inline for (info.fields) |field| {
                if (hasTensors(field.type)) return true;
            }
            return false;
        },
        else => false,
    };
}

test "hasTensors" {
    comptime {
        try std.testing.expect(hasTensors(?Tensor));
        try std.testing.expect(hasTensors(struct { u8, ?Tensor }));
        try std.testing.expect(!hasTensors(struct { u8, usize }));
    }
}

pub fn hasConstTensors(comptime T: type, comptime self_const: bool) bool {
    if (T == Tensor) return self_const;

    return switch (@typeInfo(T)) {
        inline .Array, .Optional => |info| hasTensors(info.child) and self_const,
        .Pointer => |ptr_info| hasConstTensors(ptr_info.child, ptr_info.is_const),
        inline .Struct, .Union => |info| {
            inline for (info.fields) |field| {
                if (hasConstTensors(field.type, self_const)) return true;
            }
            return false;
        },
        else => false,
    };
}

test "hasConstTensors" {
    try std.testing.expect(!hasConstTensors(?Tensor, false));
    try std.testing.expect(hasConstTensors(struct { u8, ?Tensor }, true));
    try std.testing.expect(!hasConstTensors(struct { u8, *Tensor }, true));
    try std.testing.expect(hasConstTensors(struct { u8, *const Tensor }, false));
    try std.testing.expect(!hasConstTensors(struct { *Tensor }, false));
    try std.testing.expect(!hasConstTensors(std.meta.Tuple(&[_]type{*Tensor}), false));
    try std.testing.expect(!hasConstTensors(struct { u8, usize }, false));
    try std.testing.expect(hasConstTensors(struct { [5]Tensor, usize }, true));
    try std.testing.expect(!hasConstTensors(struct { [5]Tensor, usize }, false));
}

// making this a struct force all fields to be evaluted on creation,
// which gives a better error stacktrace
// than delaying the error to when the object fields are read.
const Sign = struct {
    FuncT: type,
    ModelT: type,
    FlatArgsT: []const type,
    ArgsT: type,
    ReturnT: type,
};

pub fn ModuleSignature(comptime func: anytype) Sign {
    const FuncT = if (@TypeOf(func) == type) func else @TypeOf(func);
    const function_info = @typeInfo(FuncT);

    const n_args = @max(function_info.Fn.params.len - 1, 0);
    comptime var argument_field_list: [n_args]type = undefined;
    for (0..n_args) |i| {
        const arg = function_info.Fn.params[i + 1];
        const T = arg.type orelse @compileError("cannot create ModuleSignature for function with an 'anytype' parameter");
        argument_field_list[i] = T;
    }

    return .{
        .FuncT = FuncT,
        .ModelT = @typeInfo(FuncT).Fn.params[0].type orelse @compileError("cannot create ModuleSignature for function with an 'anytype' parameter"),
        .ArgsT = if (n_args > 0) std.meta.Tuple(&argument_field_list) else @TypeOf(.{}),
        .FlatArgsT = &argument_field_list,
        .ReturnT = @typeInfo(FuncT).Fn.return_type.?,
    };
}

pub const MlirFn = struct {
    name: []const u8,
    n_model: u32,
    n_args: u32,
    res_types: []mlir.Type,
    res_shapes: []Shape,
    res_donations: []Tensor._Donation,
    mlir_fn: mlir.Operation,
};

pub const FnCache = struct {
    pub const Key = struct { fn_ptr: *const anyopaque, input_hash: u64 };

    // TODO: merge arenas
    cache: std.AutoHashMapUnmanaged(Key, MlirFn),
    // Arena for the cache entries
    cache_arena: std.heap.ArenaAllocator,
    // Arena for the cache data (name, res_type)
    cache_data_arena: std.heap.ArenaAllocator,

    pub fn init(allocator: std.mem.Allocator) FnCache {
        return .{
            .cache = .{},
            .cache_arena = std.heap.ArenaAllocator.init(allocator),
            .cache_data_arena = std.heap.ArenaAllocator.init(allocator),
        };
    }

    pub fn deinit(self: FnCache) void {
        self.cache_arena.deinit();
        self.cache_data_arena.deinit();
    }

    pub fn getEntry(self: *const FnCache, key: Key) ?MlirFn {
        return self.cache.get(key);
    }

    pub fn addEntry(self: *FnCache, key: Key, value: MlirFn) !MlirFn {
        var cache_data_allocator = self.cache_data_arena.allocator();

        const res_types_copy = try cache_data_allocator.dupe(mlir.Type, value.res_types);
        errdefer cache_data_allocator.free(res_types_copy);

        const res_shapes_copy = try cache_data_allocator.dupe(Shape, value.res_shapes);
        errdefer cache_data_allocator.free(res_shapes_copy);

        const res_donations_copy = try cache_data_allocator.dupe(Tensor._Donation, value.res_donations);
        errdefer cache_data_allocator.free(res_donations_copy);

        const name_copy = try cache_data_allocator.dupeZ(u8, value.name);
        errdefer cache_data_allocator.free(name_copy);

        const owned_value: MlirFn = .{
            .name = name_copy,
            .mlir_fn = value.mlir_fn,
            .n_model = value.n_model,
            .n_args = value.n_args,
            .res_types = res_types_copy,
            .res_shapes = res_shapes_copy,
            .res_donations = res_donations_copy,
        };

        try self.cache.putNoClobber(self.cache_arena.allocator(), key, owned_value);
        return owned_value;
    }
};

test FnCache {
    const zml = @import("zml.zig");
    const platform = zml.testing.env();

    const NN = struct {
        const NN_ = @This();
        layer_weights: [3]Tensor,
        layer_biases: [3]Tensor,

        pub fn forward(self: NN_, x0: Tensor) Tensor {
            var x = x0;
            for (self.layer_weights, self.layer_biases) |w, b| {
                // TODO use the `call` magic helper
                // x = ops.callFunc(ctx, NN_, "reluLayer", .{ w, b, x });
                x = NN_.reluLayer(w, b, x);
            }
            return x;
        }

        pub fn forwardRefImpl(self: NN_, x0: Tensor) Tensor {
            var x = x0;
            for (self.layer_weights, self.layer_biases) |w, b| {
                x = NN_.reluLayer(w, b, x);
            }
            return x;
        }

        pub fn reluLayer(w: Tensor, b: Tensor, x: Tensor) Tensor {
            const wx = w.dotGeneral(x, &.{.{ -1, 0 }}, &.{});
            return wx.add(b.broadcastLeft(wx.shape())).relu();
        }
    };

    const x = try zml.Buffer.fromSlice(platform, .{2}, &[_]f16{ -1, 1 });
    const nn: zml.Bufferized(NN) = .{
        .layer_weights = .{
            try zml.Buffer.fromSlice(platform, .{ 2, 2 }, &[_]f16{ 1, -1, 0, 1 }),
            try zml.Buffer.fromSlice(platform, .{ 2, 2 }, &[_]f16{ 1, 2, 1, -1 }),
            // third layer is different
            try zml.Buffer.fromSlice(platform, .{ 3, 2 }, &[_]f16{ 1, 2, 0, 1, -1, 0 }),
        },
        .layer_biases = .{
            try zml.Buffer.fromSlice(platform, .{2}, &[_]f16{ 0, 0 }),
            try zml.Buffer.fromSlice(platform, .{2}, &[_]f16{ 10, 10 }),
            try zml.Buffer.fromSlice(platform, .{3}, &[_]f16{ -10, -10, -10 }),
        },
    };
    const res = try zml.testing.compileAndCall(platform, NN.forward, .{ nn, x });
    const expected = try zml.testing.compileAndCall(platform, NN.forwardRefImpl, .{ nn, x });
    try zml.testing.expectClose(expected, res, 1e-4);
}

pub fn hashArgs(mod: anytype) u64 {
    var hasher = std.hash.Wyhash.init(0);
    hash(&hasher, mod, .DeepRecursive);
    return hasher.final();
}

pub fn hashShape(hasher: *std.hash.Wyhash, shape: Shape) void {
    // Note: if we enforced 0-init dims then we could hash dims instead.
    hashArray(hasher, shape.dims(), .Shallow);
    hash(hasher, shape.dtype(), .Shallow);
    for (shape.tags()) |tag| {
        hash(hasher, @intFromPtr(tag), .Shallow);
    }
}

const HashStrategy = std.hash.Strategy;
const tensorAwareHash = hash; // alias for when "hash" is ambiguous

/// Provides generic hashing for any eligible type.
/// Strategy is provided to determine if pointers should be followed or not.
pub fn hash(hasher: *std.hash.Wyhash, key: anytype, comptime strat: HashStrategy) void {
    const Key = @TypeOf(key);
    if (Key == Tensor) return hashShape(hasher, key.shape());
    if (Key == Shape) return hashShape(hasher, key);

    if (strat == .Shallow and std.meta.hasUniqueRepresentation(Key)) {
        hasher.update(std.mem.asBytes(&key));
        return;
    }

    switch (@typeInfo(Key)) {
        .NoReturn, .Opaque, .Undefined, .Null, .ComptimeFloat, .ComptimeInt, .Type, .EnumLiteral, .Frame, .Void => return,

        // Help the optimizer see that hashing an int is easy by inlining!
        // TODO Check if the situation is better after #561 is resolved.
        .Int => |int| switch (int.signedness) {
            .signed => hash(hasher, @as(@Type(.{ .Int = .{
                .bits = int.bits,
                .signedness = .unsigned,
            } }), @bitCast(key)), strat),
            .unsigned => {
                if (std.meta.hasUniqueRepresentation(Key)) {
                    hasher.update(std.mem.asBytes(&key));
                } else {
                    // Take only the part containing the key value, the remaining
                    // bytes are undefined and must not be hashed!
                    const byte_size = comptime std.math.divCeil(comptime_int, @bitSizeOf(Key), 8) catch unreachable;
                    hasher.update(std.mem.asBytes(&key)[0..byte_size]);
                }
            },
        },
        // Note: contrary to Zig we accept hashing floats.
        // Typically the float we are going to hash here are hyperparameters,
        // and not the result of an operation, so bytes should be the same everytime.
        .Float => hasher.update(std.mem.asBytes(&key)),
        .Bool => hash(hasher, @intFromBool(key), strat),
        .Enum => hash(hasher, @intFromEnum(key), strat),
        .ErrorSet => hash(hasher, @intFromError(key), strat),
        .AnyFrame, .Fn => hash(hasher, @intFromPtr(key), strat),
        .Pointer => |info| switch (info.size) {
            .One => switch (strat) {
                .Shallow => hash(hasher, @intFromPtr(key), .Shallow),
                .Deep => hash(hasher, key.*, .Shallow),
                .DeepRecursive => switch (@typeInfo(info.child)) {
                    .Opaque, .Fn => hash(hasher, @intFromPtr(key), .Shallow),
                    else => hash(hasher, key.*, .DeepRecursive),
                },
            },
            .Slice => {
                switch (strat) {
                    .Shallow => hash(hasher, @intFromPtr(key.ptr), .Shallow),
                    .Deep => hashArray(hasher, key, .Shallow),
                    .DeepRecursive => hashArray(hasher, key, .DeepRecursive),
                }
                hash(hasher, key.len, .Shallow);
            },
            .Many,
            .C,
            => switch (strat) {
                .Shallow => hash(hasher, @intFromPtr(key), .Shallow),
                else => @compileError(
                    \\ unknown-length pointers and C pointers cannot be hashed deeply.
                    \\ Consider providing your own hash function.
                ),
            },
        },
        .Optional => if (key) |k| hash(hasher, k, strat),

        .Array => hashArray(hasher, key, strat),

        .Vector => |info| {
            if (std.meta.hasUniqueRepresentation(Key)) {
                hasher.update(std.mem.asBytes(&key));
            } else {
                comptime var i = 0;
                inline while (i < info.len) : (i += 1) {
                    hash(hasher, key[i], strat);
                }
            }
        },

        .Struct => |info| {
            inline for (info.fields) |field| {
                // We reuse the hash of the previous field as the seed for the
                // next one so that they're dependant.
                hash(hasher, @field(key, field.name), strat);
            }
        },

        .Union => |info| {
            if (info.tag_type) |tag_type| {
                const tag = std.meta.activeTag(key);
                hash(hasher, tag, strat);
                inline for (info.fields) |field| {
                    if (@field(tag_type, field.name) == tag) {
                        if (field.type != void) {
                            hash(hasher, @field(key, field.name), strat);
                        }
                        // TODO use a labelled break when it does not crash the compiler. cf #2908
                        // break :blk;
                        return;
                    }
                }
                unreachable;
            } else @compileError("cannot hash untagged union type: " ++ @typeName(Key) ++ ", provide your own hash function");
        },

        .ErrorUnion => blk: {
            const payload = key catch |err| {
                hash(hasher, err, strat);
                break :blk;
            };
            hash(hasher, payload, strat);
        },
    }
}

fn hashArray(hasher: anytype, key: anytype, comptime strat: HashStrategy) void {
    for (key) |element| {
        hash(hasher, element, strat);
    }
}

fn splitBuffer(T: type, buffer: []T, lengths: anytype) [lengths.len][]T {
    var res: [lengths.len][]T = undefined;
    var i: usize = 0;
    inline for (&res, lengths) |*r, len| {
        r.* = buffer[i .. i + len];
        i += len;
    }
    std.debug.assert(i == buffer.len);
    return res;
}
