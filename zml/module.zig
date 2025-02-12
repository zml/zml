const std = @import("std");

const asynk = @import("async");
const dialect = @import("mlir/dialects");
const runfiles = @import("runfiles");
const stdx = @import("stdx");
const xla_pb = @import("//xla:xla_proto");

const meta = @import("meta.zig");
const mlir = @import("mlir.zig");
const ops = @import("ops.zig");
const pjrt = @import("pjrtx.zig");

const BaseExe = @import("exe.zig").BaseExe;
const Buffer = @import("buffer.zig").Buffer;
const Bufferized = @import("tensor.zig").Bufferized;
const Location = mlir.Location;
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;
const ShapeOf = @import("tensor.zig").ShapeOf;
const Target = @import("platform.zig").Target;
const Tensor = @import("tensor.zig").Tensor;
const Tracer = @import("tools/tracer.zig").Tracer;

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
                stdx.debug.assert(self == .open or self.block().eql(arg.block()), "Can't add {} from {?x} block to {?x} block", .{ arg, arg.block()._inner.ptr, self.block()._inner.ptr });
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

pub const MlirFn = struct {
    name: []const u8,
    num_args: u32,
    res_tensors: *const anyopaque,
    res_types: []mlir.Type,
    res_shapes: []Shape,
    res_donations: []Tensor._Donation,
    mlir_fn: mlir.Operation,

    pub const Kind = enum {
        main,
        private,
    };
};

pub const CompilationContext = struct {
    _platform: Platform,
    _name: []const u8,

    _arena: std.heap.ArenaAllocator,
    _mlir_ctx: mlir.Context,
    _mlir_registry: mlir.Registry,
    _mlir_canonicalizer: mlir.PassManager,

    _module: mlir.Module,

    _blocks: std.BoundedArray(Block, 64) = .{},
    _fn_cache: FnCache = .{},

    _block_args: TensorToBlockArg = .{},
    _unique_id: u64 = 10000,
    _tracer: Tracer,

    _previous: ?*CompilationContext = null,
    threadlocal var _current: ?*CompilationContext = null;

    const TensorToBlockArg = std.AutoHashMapUnmanaged(Tensor._Id, struct { mlir.Value, Tensor._Donation });
    const AttributeList = std.BoundedArray(mlir.NamedAttribute, 3);

    pub fn init(allocator_: std.mem.Allocator, full_name: []const u8, platform: Platform) !CompilationContext {
        const mlir_registry = mlir.Registry.init() catch unreachable;
        inline for (.{ "func", "stablehlo" }) |d| {
            mlir.DialectHandle.fromString(d).insertDialect(mlir_registry);
        }
        var mlir_ctx = mlir.Context.initWithRegistry(mlir_registry, false) catch unreachable;
        mlir_ctx.loadAllAvailableDialects();

        // Too long module names create too long file paths and files failed to create.
        // * leave half of the space for parent folder and XLA generated filename,
        // * leave 17 bytes for the module hash (16 + 1 for underscore).
        const max_name_len = @divFloor(std.fs.max_path_bytes, 2) - 17;
        const name = full_name[0..@min(max_name_len, full_name.len)];

        const loc = mlir_ctx.location(@src()).named(mlir_ctx, "main");
        const module = mlir.Module.init(loc);
        module.op().setAttributeByName("sym_name", mlir.StringAttribute.init(mlir_ctx, "zml").as(mlir.Attribute).?);

        var canonicalizer = try mlir.PassManager.init(mlir_ctx);
        {
            var opm = canonicalizer.asOpPassManager();
            try opm.addPipeline("canonicalize");
            try opm.addPipeline("cse");
            try opm.addPipeline("canonicalize");
        }

        var arena = std.heap.ArenaAllocator.init(allocator_);
        _ = try arena.allocator().alloc(u8, 4096);
        _ = arena.reset(.retain_capacity);

        return .{
            ._platform = platform,
            ._name = try arena.allocator().dupe(u8, name),
            ._mlir_ctx = mlir_ctx,
            ._mlir_registry = mlir_registry,
            ._mlir_canonicalizer = canonicalizer,
            ._module = module,
            ._blocks = .{},
            ._fn_cache = .{},
            ._arena = arena,
            ._tracer = Tracer.init("ai.zml.compilation"),
        };
    }

    pub fn deinit(self: *CompilationContext) void {
        // No need to deinit self._fn_cache cause it uses our arena
        self._mlir_ctx.deinit();
        self._mlir_registry.deinit();
        self._arena.deinit();
    }

    pub fn allocator(self: *CompilationContext) std.mem.Allocator {
        return self._arena.allocator();
    }

    pub fn activate(self: *CompilationContext) void {
        self._previous = _current;
        _current = self;
    }

    pub fn deactivate(self: *CompilationContext) void {
        std.debug.assert(_current != null and _current.? == self);
        _current = self._previous;
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

    pub fn location(self: *const CompilationContext, src: std.builtin.SourceLocation, comptime name: [:0]const u8, args: anytype) mlir.Location {
        return self._mlir_ctx.location(src).namedFmt(self._mlir_ctx, name, args);
    }

    /// Compiles the given function with the given arguments.
    /// This is the untyped API and is not meant to be use directly.
    ///
    /// * allocator is used to allocate the result Exe
    /// * args can contain a mix of tensors and shapes, allowing to pass a "model struct" containig tensors.
    pub fn compileInternal(
        self: *CompilationContext,
        allocator_: std.mem.Allocator,
        comptime func: anytype,
        args: anytype,
    ) !BaseExe {
        const arena = self.allocator();

        var timer = std.time.Timer.start() catch null;
        const tensor_args = try self.tensorFromShapes(stdx.meta.FnArgs(func), arena, args);
        // Run in a dedicated thread because compilation relies on `threadlocal`.
        const f = try asynk.callBlocking(CompilationContext.emitMlir, .{ self, func, &tensor_args, CompilationContext.EmitMlirOpts{ .name = "main", .kind = .main } });
        const module = self._module;
        module.getBody().appendOperation(f.mlir_fn);

        const sharding = self._platform.sharding();
        const mlir_ctx = self._mlir_ctx;
        module.op().setAttributeByName("mhlo.num_replicas", mlir.IntegerAttribute(.i32).init(mlir_ctx, sharding.num_replicas).asAttr());
        module.op().setAttributeByName("mhlo.num_partitions", mlir.IntegerAttribute(.i32).init(mlir_ctx, sharding.num_partitions).asAttr());

        const module_hash = computeModuleHash(self._platform, module);
        var module_dir: ?[]const u8 = null;
        var pjrt_location: ?[:0]const u8 = null;

        if (self._platform.compilation_options.xla_dump_to) |xla_dump_to| {
            const sep = std.fs.path.sep_str;
            const module_dir_name = try std.fmt.allocPrint(arena, "{s}{s}{s}{s}{s}_{x}", .{ xla_dump_to, sep, @tagName(self._platform.target), sep, self._name, module_hash });
            try std.fs.cwd().makePath(module_dir_name);
            module_dir = try std.fs.cwd().realpathAlloc(arena, module_dir_name);
            const cache_dir = try std.fs.cwd().openDir(module_dir.?, .{});

            // Write the mlir to a file. All errors are discarded, since this is for debugging only.
            const mlir_name = "module.mlir";
            if (cache_dir.createFile(mlir_name, .{ .truncate = true })) |file| {
                module.op().print(file.writer(), .{ .debug_info = true, .debug_info_pretty_form = false });
                log.info("Wrote MLIR to {s}/{s}", .{ module_dir.?, mlir_name });
            } else |_| {
                log.warn("Failed to open {s}", .{mlir_name});
            }

            pjrt_location = try std.fs.path.joinZ(arena, &.{ module_dir.?, "module.pjrt" });
        }

        const loaded_executable: *pjrt.LoadedExecutable = blk: {
            if (pjrt_location) |pjrt_loc| {
                if (loadPjrtExecutable(arena, self._platform, pjrt_loc)) |exe| {
                    log.info("Loaded pre-compiled module from {s}", .{pjrt_loc});
                    break :blk exe;
                } else |err| {
                    if (err != error.FileNotFound) log.warn("Failed to load pre-compiled module: {} at {s}", .{ err, pjrt_loc });
                }
            }

            const loaded_executable = compileModuleToPjrtExecutable(arena, self._platform, module, module_dir) catch |err| {
                log.err("pjrt-{s} failed to compile: {}", .{ @tagName(self._platform.target), err });
                if (module_dir) |dir| log.err("mlir can be found at {s}/module.mlir", .{dir});
                return err;
            };

            if (pjrt_location) |pjrt_loc| {
                storePjrtExecutable(self._platform, loaded_executable, pjrt_loc) catch |err| {
                    log.warn("Failed to store compiled module: {} at {s}", .{ err, pjrt_loc });
                };
            }
            break :blk loaded_executable;
        };

        log.debug("******** ZML generated MLIR ********", .{});
        log.debug("{}", .{module.op().mlirFormatter(.{})});

        if (timer) |*t| {
            const time_ms = @divFloor(t.lap(), std.time.ns_per_ms);
            if (time_ms > 1000) log.info("Compilation took {d:.3}s", .{stdx.math.divFloat(f32, time_ms, 1000)});
        }

        return BaseExe.init(
            allocator_,
            self._platform,
            loaded_executable,
            .{
                .n_in = f.num_args,
                .result_shapes = f.res_shapes,
                .n_devices = sharding.num_replicas * sharding.num_partitions,
            },
        );
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
        std.debug.assert(block.block().eql(popped.?.block()));
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

    pub const EmitMlirOpts = struct {
        name: []const u8,
        kind: MlirFn.Kind = .private,
    };

    /// Generate an MLIR function from a ZML function.
    /// The caller is responsible to have properly created the input
    /// tensors with unique tensor ids.
    pub fn emitMlir(
        self: *CompilationContext,
        comptime func: anytype,
        args: *const stdx.meta.FnArgs(func),
        opts: EmitMlirOpts,
    ) error{OutOfMemory}!MlirFn {
        const frame = self._tracer.frameStart("emitMlir.emit");
        errdefer self._tracer.frameEnd(frame, "emitMlir.emit");

        const res_allocator = self.allocator();
        // Note: only temp allocations are done in the arena,
        // the other allocations are in the context allocator.
        var arena_state = std.heap.ArenaAllocator.init(self._arena.child_allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        const tensor_count = meta.count(Tensor, args);

        const mlir_ctx = self.mlirCtx();
        const loc = mlir_ctx.location(@src());

        const locations = try arena.alloc(mlir.Location, tensor_count);
        @memset(locations, mlir.Location.unknown(mlir_ctx));

        var input_shapes = try std.ArrayList(Shape).initCapacity(arena, tensor_count);
        meta.collect(Tensor.shape, {}, &input_shapes, args) catch unreachable;
        stdx.debug.internalAssert(input_shapes.items.len == tensor_count, "args have changed ?", .{});

        const input_types = try arena.alloc(mlir.Type, tensor_count);
        for (input_types, input_shapes.items) |*t, sh| t.* = mlir.ext.mlirType(mlir_ctx, sh);

        const og_block_args = self._block_args;
        defer {
            self._block_args.deinit(self.allocator());
            self._block_args = og_block_args;
        }

        // Reset the buffer -> assignement
        self._block_args = .{};

        // Note: this isn't stricly necessary. We call `countTensor` on `fn_res`.
        // But it forces user to have simpler function.
        const ReturnT = stdx.meta.FnResult(func);
        const out_tensor_count = comptime ops.staticCountTensors(ReturnT) orelse @compileError("Can't use " ++ @typeName(ReturnT) ++ " in an MLIR function, because it has a variable number of tensors");

        // Those are returned to caller so we don't put them in the arena, but in the module allocator.
        const fn_res = try res_allocator.create(ReturnT);
        const fn_res_types = try res_allocator.alloc(mlir.Type, out_tensor_count);
        const fn_res_shapes = try res_allocator.alloc(Shape, out_tensor_count);
        const fn_res_donations = try res_allocator.alloc(Tensor._Donation, out_tensor_count);
        var fn_body = self.openBlock(.hermetic, input_types, locations) catch unreachable;
        {
            defer self.closeBlock(fn_body);

            try self._block_args.ensureUnusedCapacity(self.allocator(), @intCast(tensor_count));
            const assigned_args_count = self.mapBlockArguments(args, fn_body.block(), 0);
            std.debug.assert(assigned_args_count == tensor_count);

            fn_res.* = forward: {
                self.activate();
                defer self.deactivate();
                break :forward @call(.auto, func, args.*);
            };

            var fn_res_values: [out_tensor_count]mlir.Value = undefined;
            self.extractValuesAndTypes(fn_res, &fn_res_values, fn_res_types, fn_res_shapes, fn_res_donations);

            const fn_ret = dialect.func.return_(mlir_ctx, &fn_res_values, loc);
            fn_body.appendOperationRecursive(fn_ret);
        }

        const arg_attrs = try arena.alloc(AttributeList, tensor_count);
        @memset(arg_attrs, .{});

        const res_attrs = try arena.alloc(AttributeList, out_tensor_count);
        @memset(res_attrs, .{});

        if (opts.kind == .main) {
            self.addDonationsAttributes(arg_attrs, fn_res_donations);
            if (self._platform.sharding().num_partitions > 1) {
                self.addShardingAttributes(arg_attrs, res_attrs, input_shapes.items, fn_res_shapes);
            }
        }

        const mlir_fn = dialect.func.func(self.mlirCtx(), .{
            .sym_name = opts.name,
            .args = input_types,
            .arg_attrs = try finalizeAttributeList(arena, mlir_ctx, arg_attrs),
            .results = fn_res_types,
            .res_attrs = try finalizeAttributeList(arena, mlir_ctx, res_attrs),
            .block = fn_body.block(),
            .location = loc,
        });

        self._tracer.frameEnd(frame, "emitMlir.emit");
        const canonicalize_frame = self._tracer.frameStart("emitMlir.canonicalize");
        defer self._tracer.frameEnd(canonicalize_frame, "emitMlir.canonicalize");
        self._mlir_canonicalizer.runOnOp(mlir_fn) catch |err| switch (err) {
            error.InvalidMlir => {
                log.err("Failed to canonicalize invalid mlir: {}", .{mlir_fn.mlirFormatter(.{})});
                // user errors should have triggered a panic before we reach this.
                @panic("ZML generated invalid mlir. Please open a bug report");
            },
        };

        return .{
            .mlir_fn = mlir_fn,
            .name = opts.name,
            .num_args = @intCast(tensor_count),
            .res_tensors = fn_res,
            .res_types = fn_res_types,
            .res_shapes = fn_res_shapes,
            .res_donations = fn_res_donations,
        };
    }

    /// Given a list of donations mapping output buffers to input buffers,
    /// generate donation attribute for each `n_args` input argument.
    fn addDonationsAttributes(self: CompilationContext, attributes: []AttributeList, donations: []const Tensor._Donation) void {
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

        const s = Shape.init(.{8}, .f16);

        const Local = struct {
            bias: Tensor,

            pub fn _fwd(self: @This(), x: Tensor, y: Tensor) [2]Tensor {
                const x1 = zml.ops.call(self, ._inner, .{x});
                const x2 = zml.ops.call(self, ._inner, .{x1});
                return .{ x1.reuseBuffer(y), x2 };
            }

            pub fn _inner(self: @This(), x: Tensor) Tensor {
                const y = x.add(self.bias);
                return y.reuseBuffer(x);
            }
        };

        const model: Local = .{
            .bias = zml.Tensor{ ._shape = s, ._id = .{ .buffer_id = 0 } },
        };

        var comp = try zml.module.CompilationContext.init(std.testing.allocator, "test", platform);
        defer comp.deinit();
        var tensor_args = .{ model, Tensor{ ._shape = s, ._id = .{ .buffer_id = 1234 } }, Tensor{ ._shape = s, ._id = .{ .buffer_id = 1235 } } };
        const f = try comp.emitMlir(Local._fwd, &tensor_args, .{ .name = "test.emitMlir.Local.forward", .kind = .main });

        var mlir_bytecode = std.ArrayList(u8).init(std.testing.allocator);
        defer mlir_bytecode.deinit();
        try mlir_bytecode.writer().print("{}", .{f.mlir_fn.mlirFormatter(.{})});

        // Check that the `x` input argument gives its buffer to the result tensor.
        // `%arg0` is the bias of the model, `%arg1` is `x`, `%arg2` is `y`.
        try std.testing.expectEqual(3, f.num_args);
        // We should have two buffers being donated.
        const template = "tf.aliasing_output = {d} : i32";
        var buf = template.*;
        for (0..2) |i| {
            const alias_attr = std.fmt.bufPrint(&buf, template, .{i}) catch unreachable;
            std.testing.expect(std.mem.indexOf(u8, mlir_bytecode.items, alias_attr) != null) catch |err| {
                log.warn("Didn't produced the expected IR:\n{s}", .{mlir_bytecode.items});
                return err;
            };
        }
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

    fn finalizeAttributeList(allocator_: std.mem.Allocator, mlir_ctx: mlir.Context, attributes: []AttributeList) ![]mlir.Attribute {
        const res = try allocator_.alloc(mlir.Attribute, attributes.len);
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
        args: stdx.meta.FnArgs(func),
    ) error{OutOfMemory}!stdx.meta.FnResult(func) {
        var arena_state = std.heap.ArenaAllocator.init(self._arena.child_allocator);
        defer arena_state.deinit();
        // This arena is used for allocations which won't outlive the function call,
        // but the function creation uses `self.allocator()` which we'll live for the duration of the compilation.
        const arena = arena_state.allocator();

        // first, do the "compile" and check the bytecode
        // the result of this will also have the correct tags of the result shapes
        const args_hash = hashArgs(args);
        const key: FnKey = .{ .fn_ptr = &func, .input_hash = args_hash };

        const function = self._fn_cache.get(key) orelse b: {
            const full_name: [:0]const u8 = if (std.mem.eql(u8, "main", func_name))
                try self.allocator().dupeZ(u8, func_name)
            else
                try std.fmt.allocPrintZ(self.allocator(), "{s}_{x}", .{ func_name, key.input_hash });

            var arg_id: u16 = 0;
            var tensor_args: @TypeOf(args) = args;
            try meta.mapAlloc(struct {
                fn cb(arg_id_: *u16, x: Tensor) Tensor {
                    const a = arg_id_.*;
                    arg_id_.* += 1;
                    return Tensor{ ._shape = x._shape, ._id = .{ .arg_id = a }, ._donation = .{ .arg = a } };
                }
            }.cb, arena, &arg_id, args, &tensor_args);

            const f = try self.emitMlir(
                func,
                &tensor_args,
                .{ .name = full_name },
            );
            self._module.getBody().appendOperation(f.mlir_fn);

            try self._fn_cache.putNoClobber(self.allocator(), key, f);
            break :b f;
        };

        const loc = self.mlirCtx().location(@src());

        const values = try arena.alloc(mlir.Value, function.num_args);
        self.extractValues(&args, values);

        const donations = try arena.alloc(Tensor._Donation, function.num_args);
        meta.collectBuf(struct {
            pub fn cb(ctx: *const CompilationContext, x: Tensor) Tensor._Donation {
                return ctx.getValueAndDonation(x)[1];
            }
        }.cb, self, &args, donations);

        const op = dialect.func.call(self.mlirCtx(), @ptrCast(function.name), values, function.res_types, loc);
        // Create the result tensor object by combining the operand results,
        // as well as the registered shapes and donations.
        // Note: this assume res can be stack-allocated.
        var res = @as(*const stdx.meta.FnResult(func), @alignCast(@ptrCast(function.res_tensors))).*;
        const LocalContext = struct { index: usize = 0, op: mlir.Operation, function: MlirFn, donations: []Tensor._Donation };
        var context: LocalContext = .{ .op = op, .function = function, .donations = donations };
        meta.visit((struct {
            fn cb(ctx: *LocalContext, tensor: *Tensor) void {
                const i = ctx.index;
                ctx.index += 1;
                var new = Tensor.fromMlirValue(ctx.op.result(i));
                new._shape = ctx.function.res_shapes[i];
                new._donation = switch (ctx.function.res_donations[i]) {
                    .no_buffer => .no_buffer,
                    .arg => |input_arg| ctx.donations[input_arg],
                    .input_buffer => .no_buffer, // user escaped the sandbox
                };
                tensor.* = new;
            }
        }).cb, &context, &res);
        std.debug.assert(context.index == op.numResults());
        return res;
    }

    /// Visit the given struct and recursively associate the `block` arguments with the `value` field of each encountered Tensor.
    ///
    /// This is done so that we have a mapping between the arguments of the kernel associated with a module and the actual Tensors
    /// stored in the Module.
    /// Caller need to allocate required memory in self._block_args.
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

                const res = ctx.self._block_args.getOrPutAssumeCapacity(tensor._id);
                if (res.found_existing) {
                    stdx.debug.panic("Failed compilation because received two tensors arguments with the same ID: {} and {} at index {} ({}).", .{ res.value_ptr.*[0], tensor, ctx.index, tensor._id });
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
    pub fn tensorFromShapes(self: *CompilationContext, ArgsT: type, allocator_: std.mem.Allocator, args_shapes: anytype) !ArgsT {
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
        try meta.mapAlloc(Local.tensorFromShape, allocator_, &self._unique_id, args_shapes, &tensor_args);
        return tensor_args;
    }

    /// Visit the given struct and extract the mlir.Value and mlir.Type associated with each tensor found.
    pub fn extractValuesAndTypes(self: *const CompilationContext, v: anytype, values: []mlir.Value, types: []mlir.Type, shapes: []Shape, donations: []Tensor._Donation) void {
        std.debug.assert(values.len == types.len);
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
        std.debug.assert(context.index == values.len);
    }

    pub fn getValueAndDonation(self: *const CompilationContext, tensor: Tensor) struct { mlir.Value, Tensor._Donation } {
        return switch (tensor._id) {
            .buffer_id, .arg_id => if (self._block_args.get(tensor._id)) |res|
                .{ res[0], res[1] }
            else {
                log.err("Found unknown tensor id {}({})", .{ tensor, tensor._id });
                @panic("Found unknown tensor id");
            },
            .mlir => |v| .{ v, tensor._donation },
        };
    }

    pub fn getValue(self: *const CompilationContext, tensor: Tensor) mlir.Value {
        return self.getValueAndDonation(tensor)[0];
    }

    pub fn extractValues(self: *const CompilationContext, v: anytype, values: []mlir.Value) void {
        meta.collectBuf(getValue, self, v, values);
    }
};

fn computeModuleHash(platform: Platform, module: mlir.Module) u64 {
    var hasher = std.hash.XxHash64.init(0);
    var hasher_writer = xxHash64Writer(&hasher);
    const writer = hasher_writer.writer();

    // Hash the canonicalized IR, without debug information that can change across builds.
    module.op().print(writer, .{ .debug_info = false });
    // Note: before we where using module.op().writeBytecode(writer),
    // but it crashes on some inputs, notably for unused variables.
    // So we use the text representation of the mlir.
    // See https://github.com/zml/zml/issues/97.
    // Writes can't fail because we are writing to a hasher.
    writer.writeAll(platform.pjrt_client.getPlatformName(platform.pjrt_api)) catch unreachable;
    const api_version = platform.pjrt_api.version();
    writer.writeInt(i64, api_version.major, .little) catch unreachable;
    writer.writeInt(i64, api_version.minor, .little) catch unreachable;

    return hasher.final();
}

const max_pjrt_executable_size = 400 * 1024 * 1024;

fn loadPjrtExecutable(arena: std.mem.Allocator, platform: Platform, absolute_file: [:0]const u8) !*pjrt.LoadedExecutable {
    const tracer = Tracer.init("ai.zml.load_exe");
    const compile_frame = tracer.frameStart("pjrt load executable");
    defer tracer.frameEnd(compile_frame, "pjrt load executable");

    const loaded_executable_file = try std.fs.openFileAbsoluteZ(absolute_file, .{});
    defer loaded_executable_file.close();

    const exe_size = if (loaded_executable_file.stat()) |stat| stat.size else |_| max_pjrt_executable_size;
    const bytes = try arena.alloc(u8, exe_size);
    defer arena.free(bytes);

    const size = try loaded_executable_file.readAll(bytes);
    return try platform.pjrt_client.deserializeAndLoad(platform.pjrt_api, bytes[0..size]);
}

fn storePjrtExecutable(platform: Platform, loaded_executable: *pjrt.LoadedExecutable, absolute_file: [:0]const u8) !void {
    const loaded_executable_file = try std.fs.createFileAbsoluteZ(absolute_file, .{});
    defer loaded_executable_file.close();

    var executable = try loaded_executable.getExecutable(platform.pjrt_api);
    defer executable.deinit(platform.pjrt_api);

    var serialize_result = try executable.serialize(platform.pjrt_api);
    defer serialize_result.deinit();

    try loaded_executable_file.writeAll(serialize_result.bytes);
}

fn compileModuleToPjrtExecutable(arena: std.mem.Allocator, platform: Platform, module: mlir.Module, xla_dump_to_: ?[]const u8) !*pjrt.LoadedExecutable {
    const tracer = Tracer.init("ai.zml.compilation");
    const compile_frame = tracer.frameStart("pjrt compilation");
    defer tracer.frameEnd(compile_frame, "pjrt compilation");

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
    try options.env_option_overrides.ensureUnusedCapacity(arena, 16);
    if (xla_dump_to_ orelse platform.compilation_options.xla_dump_to) |xla_dump_to| {
        setFlag(&options, "xla_dump_to", xla_dump_to);
        setFlag(&options, "xla_dump_hlo_as_proto", true);
        if (platform.compilation_options.xla_dump_fusion_visualization) {
            setFlag(&options, "xla_dump_fusion_visualization", true);
        }
        if (platform.compilation_options.xla_dump_hlo_pass_re) |re| {
            setFlag(&options, "xla_dump_hlo_pass_re", re);
        }
    }
    switch (platform.target) {
        .cuda => {
            // NVIDIA recommends these settings
            // https://github.com/NVIDIA/JAX-Toolbox?tab=readme-ov-file#environment-variables
            setFlag(&options, "xla_gpu_enable_triton_gemm", false);
            setFlag(&options, "xla_gpu_enable_latency_hiding_scheduler", true);
            setFlag(&options, "xla_gpu_enable_llvm_module_compilation_parallelism", true);
            setFlag(&options, "xla_gpu_enable_libnvptxcompiler", true);
            //  setFlag(&options, "xla_gpu_enable_cudnn_fmha", true);
            //  setFlag(&options, "xla_gpu_fused_attention_use_cudnn_rng", true);
            //  setFlag(&options, "xla_gpu_enable_cudnn_layer_norm", true);
            //  setFlag(&options, "xla_gpu_enable_custom_fusions", true);
            //  setFlags(&options, "xla_gpu_enable_address_computation_fusion", true);
            //  setFlag(&options, "xla_gpu_enable_dynamic_slice_fusion", true);
            //  setFlag(&options, "xla_gpu_enable_while_loop_double_buffering", true);
            //  setFlag(&options, "xla_gpu_use_runtime_fusion", true);
        },
        .rocm => {
            // Disable Triton GEMM on ROCM. For some reason it's much, much slower when
            // enabled on CDNA and it's used on RDNA. Disable it altogether.
            setFlag(&options, "xla_gpu_enable_triton_gemm", false);
        },
        else => {},
    }

    const options_bytes = try options.encode(arena);

    const loaded_executable = try platform.pjrt_client.compile(platform.pjrt_api, arena, module, options_bytes);
    errdefer loaded_executable.deinit();

    return loaded_executable;
}

fn setFlag(options: *xla_pb.CompileOptionsProto, comptime flag: [:0]const u8, value: anytype) void {
    const option: xla_pb.OptionOverrideProto = switch (@typeInfo(@TypeOf(value))) {
        .bool => .{ .value = .{ .bool_field = value } },
        .comptime_int, .int => .{ .value = .{ .int_field = value } },
        .comptime_float, .float => .{ .value = .{ .double_field = value } },
        else => .{ .value = .{ .string_field = .{ .Const = value } } },
    };
    options.env_option_overrides.appendAssumeCapacity(.{ .key = .{ .Const = flag }, .value = option });
}

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
    std.debug.assert(context.index == types.len);
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

pub const FnCache = std.AutoHashMapUnmanaged(FnKey, MlirFn);
pub const FnKey = struct { fn_ptr: *const anyopaque, input_hash: u64 };

test FnCache {
    const zml = @import("zml.zig");
    const platform = zml.testing.env();

    const Layer = struct {
        const Layer_ = @This();

        w: Tensor,
        b: Tensor,

        pub fn _fwd(self: Layer_, x: Tensor) Tensor {
            const wx = self.w.dotGeneral(x, &.{.{ -1, 0 }}, &.{});
            return wx.add(self.b.broad(wx.shape())).relu();
        }
    };

    const NN = struct {
        const NN_ = @This();
        layers: [3]Layer,

        pub fn _fwd(self: NN_, x0: Tensor) Tensor {
            var x = x0;
            for (self.layers) |layer| {
                x = ops.call(layer, ._fwd, .{x});
            }
            return x;
        }

        pub fn _forwardRefImpl(self: NN_, x0: Tensor) Tensor {
            var x = x0;
            for (self.layers) |layer| {
                x = layer._fwd(x);
            }
            return x;
        }
    };

    const x = try zml.Buffer.fromSlice(platform, .{2}, &[_]f16{ -1, 1 });
    const nn: zml.Bufferized(NN) = .{
        .layers = .{
            .{
                .w = try zml.Buffer.fromSlice(platform, .{ 2, 2 }, &[_]f16{ 1, -1, 0, 1 }),
                .b = try zml.Buffer.fromSlice(platform, .{2}, &[_]f16{ 0, 0 }),
            },
            .{
                .w = try zml.Buffer.fromSlice(platform, .{ 2, 2 }, &[_]f16{ 1, 2, 1, -1 }),
                .b = try zml.Buffer.fromSlice(platform, .{2}, &[_]f16{ 10, 10 }),
            },
            // third layer is different
            .{
                .w = try zml.Buffer.fromSlice(platform, .{ 3, 2 }, &[_]f16{ 1, 2, 0, 1, -1, 0 }),
                .b = try zml.Buffer.fromSlice(platform, .{3}, &[_]f16{ -10, -10, -10 }),
            },
        },
    };
    const res = try zml.testing.compileAndCall(platform, NN._fwd, .{ nn, x });
    const expected = try zml.testing.compileAndCall(platform, NN._forwardRefImpl, .{ nn, x });
    try zml.testing.expectClose(expected, res, 1e-4);
}

test "FnCache with mixed integer/tensor" {
    const zml = @import("zml.zig");
    const platform = zml.testing.env();

    const Layer = struct {
        const Layer_ = @This();
        var num_call: u32 = 0;

        w: Tensor,

        pub fn _fwd(self: Layer_, x: Tensor) struct { Tensor, usize } {
            const wx = self.w.dotGeneral(x, &.{.{ -1, 0 }}, &.{});
            // Note: this is for testing only, it's a bad idea to mutate global state
            // from a forward function because it can mess with caching.
            num_call += 1;
            return .{ wx.addConstant(num_call), num_call };
        }
    };

    const NN = struct {
        const NN_ = @This();
        layers: [3]Layer,

        pub fn _fwd(self: NN_, x0: Tensor) Tensor {
            var x = x0;
            var y: usize = 0;
            x, y = ops.call(self.layers[0], ._fwd, .{x});
            std.debug.assert(Layer.num_call == 1);
            std.debug.assert(y == 1);
            // Here we call a second time but since first two layers have the same shape,
            // We hit the function cache, and "num_call" is not incremented.
            x, y = ops.call(self.layers[1], ._fwd, .{x});
            std.debug.assert(Layer.num_call == 1);
            std.debug.assert(y == 1);
            x, y = ops.call(self.layers[2], ._fwd, .{x});
            std.debug.assert(Layer.num_call == 2);
            std.debug.assert(y == 2);
            return x;
        }

        pub fn _forwardRefImpl(self: NN_, x0: Tensor) Tensor {
            var x = x0;
            for (self.layers, &[_]u32{ 1, 1, 2 }) |layer, bias| {
                const wx = layer.w.dotGeneral(x, &.{.{ -1, 0 }}, &.{});
                x = wx.addConstant(bias);
            }
            return x;
        }
    };

    const x = try zml.Buffer.fromSlice(platform, .{2}, &[_]f16{ -1, 1 });
    const nn: zml.Bufferized(NN) = .{
        .layers = .{
            .{ .w = try zml.Buffer.fromSlice(platform, .{ 2, 2 }, &[_]f16{ 1, -1, 0, 1 }) },
            .{ .w = try zml.Buffer.fromSlice(platform, .{ 2, 2 }, &[_]f16{ 1, 2, 1, -1 }) },
            // third layer has different shape
            .{ .w = try zml.Buffer.fromSlice(platform, .{ 3, 2 }, &[_]f16{ 1, 2, 0, 1, -1, 0 }) },
        },
    };
    const res = try zml.testing.compileAndCall(platform, NN._fwd, .{ nn, x });
    const expected = try zml.testing.compileAndCall(platform, NN._forwardRefImpl, .{ nn, x });
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
    hash(hasher, shape._dtype, .Shallow);
    hash(hasher, shape._sharding_info, .Shallow);
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
        .noreturn, .@"opaque", .undefined, .null, .comptime_float, .comptime_int, .type, .enum_literal, .frame, .void => return,

        // Help the optimizer see that hashing an int is easy by inlining!
        // TODO Check if the situation is better after #561 is resolved.
        .int => |int| switch (int.signedness) {
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
        .float => hasher.update(std.mem.asBytes(&key)),
        .bool => hash(hasher, @intFromBool(key), strat),
        .@"enum" => hash(hasher, @intFromEnum(key), strat),
        .error_set => hash(hasher, @intFromError(key), strat),
        .@"anyframe", .@"fn" => hash(hasher, @intFromPtr(key), strat),
        .pointer => |info| switch (info.size) {
            .one => switch (strat) {
                .shallow => hash(hasher, @intFromPtr(key), .Shallow),
                .deep => hash(hasher, key.*, .Shallow),
                .deeprecursive => switch (@typeInfo(info.child)) {
                    .@"opaque", .@"fn" => hash(hasher, @intFromPtr(key), .Shallow),
                    else => hash(hasher, key.*, .DeepRecursive),
                },
            },
            .slice => {
                switch (strat) {
                    .shallow => hash(hasher, @intFromPtr(key.ptr), .Shallow),
                    .deep => hashArray(hasher, key, .Shallow),
                    .deeprecursive => hashArray(hasher, key, .DeepRecursive),
                }
                hash(hasher, key.len, .Shallow);
            },
            .many,
            .c,
            => switch (strat) {
                .shallow => hash(hasher, @intFromPtr(key), .Shallow),
                else => @compileError(
                    \\ unknown-length pointers and C pointers cannot be hashed deeply.
                    \\ Consider providing your own hash function.
                ),
            },
        },
        .optional => if (key) |k| hash(hasher, k, strat),

        .array => hashArray(hasher, key, strat),

        .vector => |info| {
            if (std.meta.hasUniqueRepresentation(Key)) {
                hasher.update(std.mem.asBytes(&key));
            } else {
                comptime var i = 0;
                inline while (i < info.len) : (i += 1) {
                    hash(hasher, key[i], strat);
                }
            }
        },

        .@"struct" => |info| {
            inline for (info.fields) |field| {
                // We reuse the hash of the previous field as the seed for the
                // next one so that they're dependant.
                hash(hasher, @field(key, field.name), strat);
            }
        },

        .@"union" => |info| {
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

        .error_union => blk: {
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
