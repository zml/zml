const std = @import("std");

const c = @import("c");
const dialects = @import("mlir/dialects");
const mlir = @import("mlir");
const pjrt = @import("pjrt");
const stdx = @import("stdx");
const upb = @import("upb");

const Buffer = @import("buffer.zig").Buffer;
const DataType = @import("dtype.zig").DataType;
const Exe = @import("exe.zig").Exe;
const Memory = @import("platform.zig").Memory;
const meta = @import("meta.zig");
const mlirx = @import("mlirx.zig");
const ops = @import("ops.zig");
const pjrtx = @import("pjrtx.zig");
const Platform = @import("platform.zig").Platform;
const tracer = @import("profiling/tracer.zig");
const Shape = @import("shape.zig").Shape;
const Sharding = @import("Sharding.zig");
const Partitioning = Sharding.Partitioning;
const Tensor = @import("tensor.zig").Tensor;

const Compiler = @This();
const log = std.log.scoped(.@"zml/compiler");

allocator: std.mem.Allocator,
io: std.Io,
arena: std.heap.ArenaAllocator,

mlir_registry: *mlir.DialectRegistry,
mlir_ctx: *mlir.Context,
mlir_pass_manager: *mlir.PassManager,
module: *mlir.Module,
platform: *const Platform,
partitioning: Sharding.Partitioning,

mlir_known_types: std.enums.EnumArray(DataType, *const mlir.Type),

scopes: stdx.BoundedArray(Scope, 16) = .empty,
manual_computation_depth: usize = 0,

channel_id: i64 = 0,
composite_id: i64 = 0,

threadlocal var _current: ?*Compiler = null;
var mlir_global_init_mutex: std.Io.Mutex = .init;
var mlir_global_registry: ?*mlir.DialectRegistry = null;

fn mlirRegistry(io: std.Io) *mlir.DialectRegistry {
    mlir_global_init_mutex.lockUncancelable(io);
    defer mlir_global_init_mutex.unlock(io);

    if (mlir_global_registry == null) {
        mlir.registerPasses("Transforms");

        const mlir_registry = mlir.DialectRegistry.init() catch unreachable;
        inline for (.{ "func", "stablehlo", "sdy" }) |d| {
            mlir.DialectHandle.fromString(d).insertDialect(mlir_registry);
        }
        mlir.registerFuncExtensions(mlir_registry);

        mlir_global_registry = mlir_registry;
    }

    return mlir_global_registry.?;
}

pub const Options = struct {
    shardings: []const Sharding = &.{},
    // If null, will be initialized from the target
    partitioner: ?Sharding.Partitioner = null,
    // Debugging options
    program_name: []const u8 = "zml",
    xla_dump_to: ?[]const u8 = null,
    xla_dump_fusion_visualization: bool = false,
    xla_dump_hlo_pass_re: ?[]const u8 = null,
    xla_dump_emitter_re: ?[]const u8 = null,
};

/// Errors surfaced while lowering and compiling a ZML program.
pub const Error = std.mem.Allocator.Error ||
    std.Io.Writer.Error ||
    mlir.Error ||
    upb.SerializeError ||
    pjrtx.Client.CompileError ||
    error{MissingDeviceInTile};

pub const Donation = union(enum) {
    implicit: u32,
    explicit: u32,
};

pub const Scope = struct {
    compiler: *Compiler,
    block: *mlir.Block,
    id_to_argument: std.AutoArrayHashMapUnmanaged(Tensor.Id, *const mlir.Value),
    id_to_donation: std.AutoArrayHashMapUnmanaged(Tensor.Id, Donation),
    id_to_memory: std.AutoArrayHashMapUnmanaged(Tensor.Id, Memory.Kind),
    arena: std.heap.ArenaAllocator,

    pub fn initFromBlock(compiler: *Compiler, block: *mlir.Block) Scope {
        const arena: std.heap.ArenaAllocator = .init(compiler.allocator);
        return .{
            .compiler = compiler,
            .block = block,
            .id_to_argument = .empty,
            .id_to_donation = .empty,
            .id_to_memory = .empty,
            .arena = arena,
        };
    }

    pub fn pop(self: *Scope) void {
        const compiler = self.compiler;
        std.debug.assert(self == compiler.currentScope());
        var arena = self.arena;
        _ = compiler.scopes.pop();
        arena.deinit();
    }

    pub fn registerTensorAsBlockArgument(scope: *Scope, id: Tensor.Id, arg_id: usize) void {
        scope.id_to_argument.put(scope.arena.allocator(), id, scope.block.argument(arg_id)) catch @panic("OOM");
    }
};

pub fn init(allocator: std.mem.Allocator, io: std.Io, platform: *const Platform, opts: Options) Compiler {
    var arena = std.heap.ArenaAllocator.init(allocator);
    const mlir_registry = mlirRegistry(io);
    var mlir_ctx = mlir.Context.init(.{ .registry = mlir_registry, .threading = false }) catch unreachable;
    mlir_ctx.loadAllAvailableDialects();

    const module = mlir.Module.init(.unknown(mlir_ctx));
    module.operation().setAttributeByName("sym_name", .string(mlir_ctx, opts.program_name));

    const pass_manager = mlir.PassManager.init(mlir_ctx);
    {
        var opm = pass_manager.asOpPassManager();
        const passes: []const []const u8 = &.{
            "canonicalize",
            "cse",
            "canonicalize",
        };
        for (passes) |pass| {
            opm.addPipeline(pass) catch unreachable;
        }
    }

    var mlir_known_types: std.enums.EnumArray(DataType, *const mlir.Type) = .initUndefined();
    {
        for (0.., &mlir_known_types.values) |i, *mlir_type| {
            mlir_type.* = mlirx.Type.fromDType(mlir_ctx, @enumFromInt(i));
        }
    }

    // Ensure replicated sharding is always included as a fallback option.
    var shardings = std.ArrayList(Sharding).initCapacity(arena.allocator(), opts.shardings.len + 1) catch @panic("OOM");
    var needs_replicated: bool = true;
    for (opts.shardings) |sharding| {
        if (sharding.data == platform.replicated_sharding.data) needs_replicated = false;
        shardings.appendAssumeCapacity(sharding.resolve(platform));
    }
    if (needs_replicated) shardings.appendAssumeCapacity(platform.replicated_sharding);

    const partitioning = Sharding.Partitioning.init(opts.partitioner orelse .fromTarget(platform.target), shardings.items) catch @panic("OOM");

    return .{
        .allocator = allocator,
        .io = io,
        .arena = arena,
        .mlir_registry = mlir_registry,
        .mlir_ctx = mlir_ctx,
        .mlir_pass_manager = pass_manager,
        .mlir_known_types = mlir_known_types,
        .module = module,
        .platform = platform,
        .partitioning = partitioning,
    };
}

pub fn deinit(self: *Compiler) void {
    if (_current == self) _current = null;
    std.debug.assert(self.scopes.len == 0);
    self.mlir_pass_manager.deinit();
    self.module.deinit();
    self.mlir_ctx.deinit();
    self.arena.deinit();
}

pub fn activate(self: *Compiler) void {
    std.debug.assert(_current == null);
    _current = self;
}

pub fn deactivate(self: *Compiler) void {
    _ = self;
    _current = null;
}

pub fn current() *Compiler {
    return _current.?;
}

pub fn currentOrNull() ?*Compiler {
    return _current;
}

pub fn currentScope(self: *Compiler) *Scope {
    return &self.scopes.slice()[self.scopes.len - 1];
}

pub fn pushBlock(self: *Compiler, block: *mlir.Block) *Scope {
    const scope = Scope.initFromBlock(self, block);
    self.scopes.appendAssumeCapacity(scope);
    return self.currentScope();
}

pub fn nextChannelId(self: *Compiler) i64 {
    self.channel_id += 1;
    return self.channel_id;
}

pub fn nextCompositeId(self: *Compiler) i64 {
    self.composite_id += 1;
    return self.composite_id;
}

pub fn mlirType(self: *const Compiler, dt: DataType) *const mlir.Type {
    return self.mlir_known_types.get(dt);
}

pub fn dtype(self: *const Compiler, mlir_type: *const mlir.Type) DataType {
    @setRuntimeSafety(false);
    for (0.., &self.mlir_known_types.values) |i, known_type| {
        if (known_type == mlir_type) return @enumFromInt(i);
    }
    std.debug.panic("Can't convert unknown mlir type to dtype: {f}", .{mlir_type});
}

pub fn abortOOM(self: *Compiler) noreturn {
    _ = self;
    @panic("OOM");
}

pub fn alloc(self: *Compiler, T: type, n: usize) []T {
    return self.arena.allocator().alloc(T, n) catch self.abortOOM();
}

pub fn allocPrint(self: *Compiler, comptime fmt: []const u8, args: anytype) []u8 {
    return std.fmt.allocPrint(self.arena.allocator(), fmt, args) catch self.abortOOM();
}

pub fn Typed(comptime func: anytype) type {
    return struct {
        pub fn compile(
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const Platform,
            opts: Options,
            args: std.meta.ArgsTuple(@TypeOf(func)),
        ) Error!Exe {
            return Compiler.compile(allocator, io, platform, func, args, opts);
        }
    };
}

pub fn compile(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    comptime func: anytype,
    args: std.meta.ArgsTuple(@TypeOf(func)),
    opts: Options,
) Error!Exe {
    // TODO: Here we have somewhat of a requirement
    // Emitting MLIR requires to have the compiler context available at all times using `Compiler.current()`.
    // If in the future, we inject an Io that is not thread-based, we might have some surprises.
    //
    // I think the correct implementation would be to dispatch `emitMlir` to a thread pool, then wait for the result
    // asynchronously using the provided Io. For now, we'll simply make that blocking as it's not a big deal but keep
    // in mind we might want to revisit that later.
    _ = io;
    var st_io: std.Io.Threaded = .init_single_threaded;
    defer st_io.deinit();

    const span_name = try tracer.formatSpanName(allocator, "zml.module.compile", .{
        .program_name = opts.program_name,
        .arg_count = args.len,
    });
    defer allocator.free(span_name);
    var span = tracer.Span.start(span_name);
    defer span.end();

    var compiler: Compiler = .init(allocator, st_io.io(), platform, opts);
    defer compiler.deinit();

    var result = emitMlir(&compiler, func, args) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => unreachable,
    };
    defer result.output_info.deinit(compiler.allocator);
    defer result.input_info.deinit(compiler.allocator);

    try addPartitionerOperations(&compiler);

    _ = result.func.appendTo(compiler.module.body());

    const num_partitions = compiler.partitioning.numPartitions();
    const num_replicas = compiler.partitioning.numReplicas();
    const num_devices = compiler.partitioning.numDevices();

    compiler.module.operation().setAttributeByName(
        "mhlo.num_partitions",
        .int(compiler.mlir_ctx, .i32, num_partitions),
    );
    compiler.module.operation().setAttributeByName(
        "mhlo.num_replicas",
        .int(compiler.mlir_ctx, .i32, num_replicas),
    );

    compiler.mlir_pass_manager.runOnOp(compiler.module.operation()) catch |err| switch (err) {
        error.MlirUnexpected => {
            std.log.err("Failed to canonicalize invalid mlir: \n {f} \n ", .{compiler.module.operation()});
            @panic("ZML generated invalid mlir. Please open a bug report");
        },
    };

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const loaded_executable = try compileModuleToPjrtExecutable(arena.allocator(), st_io.io(), platform, compiler.module, compiler.partitioning, opts);
    log.debug("\n******** ZML generated MLIR ********\n{f}", .{compiler.module.operation()});

    const exe = try Exe.init(
        allocator,
        platform,
        loaded_executable,
        @intCast(num_devices),
        num_partitions,
        // This will get copied into exe
        result.input_info.items(.shape),
        result.output_info.items(.shape),
        result.input_info.items(.sharding),
        result.output_info.items(.sharding),
    );
    errdefer exe.deinit();

    return exe;
}

fn addPartitionerOperations(ctx: *Compiler) !void {
    const allocator = ctx.arena.allocator();
    const mlir_ctx = ctx.mlir_ctx;
    const module = ctx.module;
    const partitioning = ctx.partitioning;

    switch (partitioning.partitioner) {
        .gspmd => {},
        .shardy => {
            for (partitioning.shardings) |sharding| {
                const attr_str = try sharding.data.sdyMeshAttr(allocator);
                defer allocator.free(attr_str);

                const name = sharding.data.name;
                const mesh_attr = try mlir.Attribute.parse(mlir_ctx, attr_str);

                const mesh_op = mlir.Operation.make(mlir_ctx, "sdy.mesh", .{
                    .attributes = &.{
                        .named(mlir_ctx, "sym_name", .string(mlir_ctx, name)),
                        .named(mlir_ctx, "mesh", mesh_attr),
                    },
                    .location = .unknown(mlir_ctx),
                    .verify = false,
                });

                _ = mesh_op.appendTo(module.body());
            }
        },
    }
}

const EmitMlirResult = struct {
    func: *mlir.Operation,
    input_info: std.MultiArrayList(TensorInfo),
    output_info: std.MultiArrayList(TensorInfo),
};

pub const TensorInfo = struct {
    id: Tensor.Id,
    shape: Shape,
    sharding: Sharding,
    value: *const mlir.Value,

    // Only used for input tensors, stores which output tensor ends up with their buffer
    aliasing_output: ?u32 = null,

    pub fn attributes(info: TensorInfo, arena: std.mem.Allocator, compiler: *const Compiler, scope: *const Scope) error{OutOfMemory}!AttributeList {
        var attrs: AttributeList = .empty;

        const mlir_ctx = compiler.mlir_ctx;
        const sharding_attr = try compiler.partitioning.tensorShardingAttr(arena, mlir_ctx, info.shape, info.sharding);
        const name = switch (compiler.partitioning.partitioner) {
            .gspmd => "mhlo.sharding",
            .shardy => "sdy.sharding",
        };
        attrs.appendAssumeCapacity(.named(mlir_ctx, name, sharding_attr));

        const memory = scope.id_to_memory.get(info.id) orelse .device;
        if (memory != .device) {
            attrs.appendAssumeCapacity(.named(
                mlir_ctx,
                "mhlo.memory_kind",
                .string(mlir_ctx, compiler.platform.memoryKind(memory)),
            ));
        }

        if (info.aliasing_output) |output_index| {
            attrs.appendAssumeCapacity(
                .named(mlir_ctx, "tf.aliasing_output", .int(mlir_ctx, .i32, output_index)),
            );
        }

        return attrs;
    }
};

const AttributeList = stdx.BoundedArray(mlir.NamedAttribute, 3);

fn emitMlir(compiler: *Compiler, comptime func: anytype, args: std.meta.ArgsTuple(@TypeOf(func))) !EmitMlirResult {
    const module = mlir.Module.init(.unknown(compiler.mlir_ctx));
    errdefer module.deinit();

    const block = mlir.Block.init(&.{}, &.{});
    errdefer block.deinit();

    const fn_scope: *Scope = compiler.pushBlock(block);
    defer fn_scope.pop();

    var input_info = try createBlockArguments(compiler, fn_scope, &args);
    errdefer input_info.deinit(compiler.allocator);

    var result = result: {
        compiler.activate();
        defer compiler.deactivate();

        break :result @call(.auto, func, args);
    };

    var output_info = try collectOutputInfo(compiler, fn_scope, &result);
    errdefer output_info.deinit(compiler.allocator);

    // Finalize in a separate function that doesn't depend on `func` type.
    return try finalizeMlirFunc(compiler, fn_scope, input_info, output_info);
}

fn createBlockArguments(compiler: *Compiler, scope: *Scope, v: anytype) error{OutOfMemory}!std.MultiArrayList(TensorInfo) {
    const CreateBlockArgumentsCb = struct {
        compiler: *Compiler,
        scope: *Scope,

        current_argument_id: u32 = 0,
        infos: std.MultiArrayList(TensorInfo) = .empty,

        fn cb(ctx: *@This(), tensor: *const Tensor) error{OutOfMemory}!void {
            // Declare the program argument as a packed u8
            const og_shape = tensor._shape;
            const packed_shape = og_shape.packedShape();
            const mlir_type = mlirx.Type.rankedTensor(ctx.compiler.mlir_ctx, packed_shape);
            var value = ctx.scope.block.addArgument(mlir_type, .unknown(ctx.compiler.mlir_ctx));

            // But immediately create the MLIR value corresponding to the fp4 data.
            // That way the user code only sees a proper fp4 tensor.
            if (tensor.dtype().bitSizeOf() < 8) {
                value = unpack(ctx.compiler, ctx.scope, og_shape, value);
            }
            const gop = ctx.scope.id_to_argument.getOrPutValue(ctx.scope.arena.allocator(), tensor.id, value) catch @panic("OOM");
            if (gop.found_existing) std.debug.panic("Tensor with id {} has already been used once as an argument", .{tensor.id});

            // Associate each input argument with their own buffer
            ctx.scope.id_to_donation.putNoClobber(ctx.scope.arena.allocator(), tensor.id, .{ .implicit = ctx.current_argument_id }) catch @panic("OOM");

            defer ctx.current_argument_id += 1;

            const input_sharding = ctx.compiler.partitioning.selectSharding(packed_shape) catch |err| switch (err) {
                error.NoSuitableSharding => std.debug.panic(
                    "Failed to resolve sharding for input {f}({d}) because it's using unknown sharding. Pass more shardings to `platform.compile`. Known shardings: {f}",
                    .{ packed_shape, ctx.current_argument_id, stdx.fmt.slice(ctx.compiler.partitioning.shardings) },
                ),
            };
            try ctx.infos.append(ctx.compiler.allocator, .{
                .id = tensor.id,
                .shape = og_shape,
                .sharding = input_sharding,
                .value = value,
            });
        }
    };

    var context: CreateBlockArgumentsCb = .{ .compiler = compiler, .scope = scope };
    errdefer context.infos.deinit(compiler.allocator);

    try meta.visit(CreateBlockArgumentsCb.cb, &context, v);

    return context.infos;
}

fn collectOutputInfo(compiler: *Compiler, scope: *Scope, v: anytype) error{OutOfMemory}!std.MultiArrayList(TensorInfo) {
    const CollectOutputInfoCb = struct {
        compiler: *Compiler,
        scope: *Scope,

        infos: std.MultiArrayList(TensorInfo) = .empty,

        fn cb(ctx: *@This(), tensor: *const Tensor) !void {
            const og_shape = tensor.shape();
            const packed_shape = og_shape.packedShape();
            var value = ctx.scope.id_to_argument.get(tensor.id) orelse
                tensor._value orelse
                @panic("no value found for output tensor");

            if (tensor.dtype().bitSizeOf() < 8) {
                value = repack(ctx.compiler, ctx.scope, og_shape, value);
            }

            try ctx.infos.append(ctx.compiler.allocator, .{
                .id = tensor.id,
                .shape = og_shape,
                // Note: the panic should have been triggered during createBlockArguments or emitMlir
                .sharding = ctx.compiler.partitioning.selectSharding(packed_shape) catch @panic("failed to resolve output sharding"),
                .value = value,
            });
        }
    };

    var context: CollectOutputInfoCb = .{ .compiler = compiler, .scope = scope };
    errdefer context.infos.deinit(compiler.allocator);

    try meta.visit(CollectOutputInfoCb.cb, &context, v);
    return context.infos;
}

fn finalizeMlirFunc(compiler: *Compiler, fn_scope: *Scope, input_info: std.MultiArrayList(TensorInfo), output_info: std.MultiArrayList(TensorInfo)) error{OutOfMemory}!EmitMlirResult {
    var arena_state = std.heap.ArenaAllocator.init(compiler.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const mlir_ctx = compiler.mlir_ctx;

    const fn_return = dialects.func.returns(mlir_ctx, output_info.items(.value), .unknown(mlir_ctx));
    _ = fn_return.appendTo(fn_scope.block);

    // Resolve donations
    for (0.., output_info.items(.id)) |output_index, output_id| {
        if (fn_scope.id_to_donation.get(output_id)) |donation| {
            const donated_input = switch (donation) {
                // don't emit implicit donation since they modify the way the function is called
                .implicit => continue,
                .explicit => |input| input,
            };
            const aliasing_output: *?u32 = &input_info.items(.aliasing_output)[donated_input];
            if (aliasing_output.*) |previous_aliased_output| {
                std.debug.panic("Input {d} buffer {} was reused twice with `reuseBuffer` for output {} and output {}. Expected `reuseBuffer` to be called at most once", .{ donated_input, input_info.items(.shape)[donated_input], previous_aliased_output, output_index });
            }
            aliasing_output.* = @intCast(output_index);
        }
    }

    // Input sharding/memory/aliasing attributes
    const input_attributes = try arena.alloc(*const mlir.Attribute, input_info.len);
    for (0.., input_attributes) |i, *input_attrs| {
        const attrs_list = try input_info.get(i).attributes(arena, compiler, fn_scope);
        input_attrs.* = .dict(mlir_ctx, attrs_list.constSlice());
    }

    // Output sharding/memory attributes
    const output_attributes = try arena.alloc(*const mlir.Attribute, output_info.len);
    for (0.., output_attributes) |i, *output_attrs| {
        const attrs_list = try output_info.get(i).attributes(arena, compiler, fn_scope);
        output_attrs.* = .dict(mlir_ctx, attrs_list.constSlice());
    }

    const mlir_func = dialects.func.func(mlir_ctx, .{
        .name = "main",
        .block = fn_scope.block,
        .location = .unknown(mlir_ctx),
        .args_attributes = input_attributes,
        .results_attributes = output_attributes,
        .verify = false,
    });

    return .{
        .func = mlir_func,
        .input_info = input_info,
        .output_info = output_info,
    };
}

fn unpack(compiler: *Compiler, scope: *Scope, og_shape: Shape, tensor_value: *const mlir.Value) *const mlir.Value {
    const true_type = mlirx.Type.rankedTensor(compiler.mlir_ctx, og_shape);
    const elem_per_bytes = @divExact(8, og_shape.dtype().bitSizeOf());
    const intermediary_type = mlirx.Type.rankedTensor(compiler.mlir_ctx, og_shape.splitAxis(-1, .{ -1, elem_per_bytes }));

    const bit_cast_op = dialects.stablehlo.bitcast_convert(
        compiler.mlir_ctx,
        tensor_value,
        intermediary_type,
        .unknown(compiler.mlir_ctx),
    ).appendTo(scope.block);

    const reshape_op = dialects.stablehlo.reshape(
        compiler.mlir_ctx,
        bit_cast_op.result(0),
        true_type,
        .unknown(compiler.mlir_ctx),
    ).appendTo(scope.block);

    return reshape_op.result(0);
}

fn repack(compiler: *Compiler, scope: *Scope, og_shape: Shape, value: *const mlir.Value) *const mlir.Value {
    const packed_type = mlirx.Type.rankedTensor(compiler.mlir_ctx, og_shape.packedShape());
    const elem_per_bytes = @divExact(8, og_shape.dtype().bitSizeOf());
    const intermediary_type = mlirx.Type.rankedTensor(compiler.mlir_ctx, og_shape.splitAxis(-1, .{ -1, elem_per_bytes }));

    const reshape_op = dialects.stablehlo.reshape(
        compiler.mlir_ctx,
        value,
        intermediary_type,
        .unknown(compiler.mlir_ctx),
    ).appendTo(scope.block);

    const bit_cast_op = dialects.stablehlo.bitcast_convert(
        compiler.mlir_ctx,
        reshape_op.result(0),
        packed_type,
        .unknown(compiler.mlir_ctx),
    ).appendTo(scope.block);

    return bit_cast_op.result(0);
}

fn setXlaOverrideFlag(map: *c.upb_Map, flag: []const u8, value: anytype, upb_arena: *c.upb_Arena) !void {
    const result = c.upb_Map_Set(
        map,
        .{ .str_val = upb.stringView(flag) },
        .{ .msg_val = blk: {
            const field = try upb.new(c.xla_OptionOverrideProto, upb_arena);
            switch (@typeInfo(@TypeOf(value))) {
                .bool => c.xla_OptionOverrideProto_set_bool_field(field, value),
                .comptime_int, .int => c.xla_OptionOverrideProto_set_int_field(field, @intCast(value)),
                .comptime_float, .float => c.xla_OptionOverrideProto_set_double_field(field, @floatCast(value)),
                else => c.xla_OptionOverrideProto_set_string_field(field, upb.stringView(value)),
            }
            break :blk @ptrCast(field);
        } },
        upb_arena,
    );

    if (result == false) {
        return std.mem.Allocator.Error.OutOfMemory;
    }
}

fn compileModuleToPjrtExecutable(arena: std.mem.Allocator, io: std.Io, platform: *const Platform, module: *const mlir.Module, partitioning: Partitioning, opts: Options) !*pjrt.LoadedExecutable {
    var upb_alloc: upb.Allocator = .init(arena);
    const upb_arena = c.upb_Arena_Init(null, 0, upb_alloc.inner());
    defer c.upb_Arena_Free(upb_arena);

    const use_shardy_partitioner = switch (partitioning.partitioner) {
        .shardy => true,
        .gspmd => false,
    };

    const num_partitions = partitioning.numPartitions();
    const num_replicas = partitioning.numReplicas();

    const device_assignment = try partitioning.deviceAssignment(arena);

    const options = blk: {
        const options = try upb.new(c.xla_CompileOptionsProto, upb_arena);
        c.xla_CompileOptionsProto_set_executable_build_options(options, executable_build_options_blk: {
            const exec_build_options = try upb.new(c.xla_ExecutableBuildOptionsProto, upb_arena);
            c.xla_ExecutableBuildOptionsProto_set_device_ordinal(exec_build_options, -1);
            c.xla_ExecutableBuildOptionsProto_set_num_replicas(exec_build_options, num_replicas);
            c.xla_ExecutableBuildOptionsProto_set_num_partitions(exec_build_options, num_partitions);
            c.xla_ExecutableBuildOptionsProto_set_use_spmd_partitioning(exec_build_options, true);
            c.xla_ExecutableBuildOptionsProto_set_use_shardy_partitioner(exec_build_options, use_shardy_partitioner);

            // Tell XLA how much device memory PJRT actually made available.
            // Without this, the GPU scheduler falls back to 80% of physical
            // memory. Programs whose donated inputs and outputs exceed that
            // artificial limit get a zero-byte temporary-memory budget and
            // can trigger excessive rematerialization.
            var device_memory_size: ?u64 = null;
            for (platform.devices) |device| {
                const bytes_limit = device.memoryStats().bytes_limit orelse {
                    device_memory_size = null;
                    break;
                };
                device_memory_size = @min(device_memory_size orelse bytes_limit, bytes_limit);
            }
            if (device_memory_size) |bytes_limit| {
                c.xla_ExecutableBuildOptionsProto_set_device_memory_size(exec_build_options, @intCast(bytes_limit));
            }

            c.xla_ExecutableBuildOptionsProto_set_device_assignment(exec_build_options, device_assignment_blk: {
                const device_assignment_proto = try upb.new(c.xla_DeviceAssignmentProto, upb_arena);

                c.xla_DeviceAssignmentProto_set_replica_count(device_assignment_proto, num_replicas);
                c.xla_DeviceAssignmentProto_set_computation_count(device_assignment_proto, num_partitions);

                const computation_devices = c.xla_DeviceAssignmentProto_resize_computation_devices(
                    device_assignment_proto,
                    @intCast(num_partitions),
                    upb_arena,
                );

                for (computation_devices[0..@intCast(num_partitions)], 0..) |*computation_device, i| {
                    computation_device.* = try upb.new(c.xla_DeviceAssignmentProto_ComputationDevice, upb_arena);
                    _ = c.xla_DeviceAssignmentProto_ComputationDevice_add_replica_device_ids(
                        computation_device.*,
                        @intCast(device_assignment[@intCast(i)]),
                        upb_arena,
                    );
                }

                break :device_assignment_blk device_assignment_proto;
            });

            break :executable_build_options_blk exec_build_options;
        });

        const overrides_map = c._xla_CompileOptionsProto_env_option_overrides_mutable_upb_map(options, upb_arena);
        switch (platform.target) {
            .cuda => {
                // NVIDIA recommends these settings
                // https://github.com/NVIDIA/JAX-Toolbox?tab=readme-ov-file#environment-variables
                try setXlaOverrideFlag(overrides_map, "xla_gpu_enable_latency_hiding_scheduler", true, upb_arena);
            },
            .rocm => {
                // Use lld from libllvm instead of invoking the ld.lld binary.
                // This saves us from having to sandbox it.
                try setXlaOverrideFlag(overrides_map, "xla_gpu_use_inprocess_lld", true, upb_arena);

                // Do not enable the FUSION command buffer to avoid some weird crashes.
                // This is what AMD recommendeds in the meantime.
                try setXlaOverrideFlag(overrides_map, "xla_gpu_enable_command_buffer", "CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN,DYNAMIC_SLICE_FUSION", upb_arena);
            },
            .oneapi => {
                // More efficient for the allgather/broadcast implementation of the collective permute.
                try setXlaOverrideFlag(overrides_map, "xla_gpu_collective_permute_connected_components", true, upb_arena);
                try setXlaOverrideFlag(overrides_map, "xla_gpu_autotune_level", 0, upb_arena);
                try setXlaOverrideFlag(overrides_map, "xla_gpu_enable_command_buffer", "", upb_arena);

                // Not supported by OneAPI
                try setXlaOverrideFlag(overrides_map, "xla_disable_hlo_passes", "scan-rewriter", upb_arena);
                try setXlaOverrideFlag(overrides_map, "xla_gpu_experimental_use_ragged_dot_grouped_gemm", false, upb_arena);
                try setXlaOverrideFlag(overrides_map, "xla_gpu_enable_cub_radix_sort", false, upb_arena);
            },
            else => {},
        }

        if (opts.xla_dump_to) |xla_dump_to| {
            try setXlaOverrideFlag(overrides_map, "xla_dump_to", xla_dump_to, upb_arena);
            try setXlaOverrideFlag(overrides_map, "xla_dump_hlo_as_proto", true, upb_arena);
            if (opts.xla_dump_fusion_visualization) {
                try setXlaOverrideFlag(overrides_map, "xla_dump_fusion_visualization", true, upb_arena);
            }
            if (opts.xla_dump_hlo_pass_re) |re| {
                try setXlaOverrideFlag(overrides_map, "xla_dump_hlo_pass_re", re, upb_arena);
            }
            if (opts.xla_dump_emitter_re) |re| {
                try setXlaOverrideFlag(overrides_map, "xla_dump_emitter_re", re, upb_arena);
            }
        }

        switch (platform.target) {
            .rocm, .cuda, .oneapi => if (std.c.getenv("ZML_AUTOTUNE_CACHE_DIR")) |path| {
                try setXlaOverrideFlag(overrides_map, "xla_gpu_per_fusion_autotune_cache_dir", std.mem.span(path), upb_arena);
            },
            else => {},
        }

        break :blk options;
    };

    const loaded_executable = try pjrtx.Client.compile(
        platform.pjrt_client,
        platform.pjrt_api,
        arena,
        io,
        module,
        try upb.serialize(options, upb_arena),
    );
    errdefer loaded_executable.deinit();

    return loaded_executable;
}
