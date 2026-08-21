const std = @import("std");

const c = @import("c");
const dialects = @import("mlir/dialects");
const mlir = @import("mlir");
const pjrt = @import("pjrt");
const stdx = @import("stdx");
const upb = @import("upb");

const Buffer = @import("buffer.zig").Buffer;
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

const zml_module = @This();
const log = std.log.scoped(.@"zml/module");

pub const autotuneStartCallTarget = "zml$autotune_start";
pub const autotuneStopCallTarget = "zml$autotune_stop";

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
pub const CompilationOptions = struct {
    pub const ExecutionTiming = enum {
        none,
        device,
    };

    shardings: []const Sharding = &.{},
    // If null, will be initialized from the target
    partitioner: ?Sharding.Partitioner = null,
    // Debugging options
    program_name: []const u8 = "zml",
    xla_dump_to: ?[]const u8 = null,
    xla_dump_fusion_visualization: bool = false,
    xla_dump_hlo_pass_re: ?[]const u8 = null,
    xla_dump_emitter_re: ?[]const u8 = null,
    /// Adds GPU-native start/stop markers around the entry function. The
    /// resulting executable must be used with an attached ExecutionTimer and
    /// is intended only for benchmarking, never for production execution.
    execution_timing: ExecutionTiming = .none,
};

/// Errors surfaced while lowering and compiling a ZML program.
pub const CompileError = std.mem.Allocator.Error ||
    std.Io.Writer.Error ||
    mlir.Error ||
    upb.SerializeError ||
    pjrtx.Client.CompileError ||
    error{ MissingDeviceInTile, UnsupportedExecutionTiming };

const AttributeList = stdx.BoundedArray(mlir.NamedAttribute, 3);

pub const CompilationContext = struct {
    pub const Scope = struct {
        block: *mlir.Block,
        id_to_argument: std.AutoArrayHashMapUnmanaged(usize, usize),
        id_to_donation: std.AutoArrayHashMapUnmanaged(usize, usize),
        id_to_output_memory_kind: std.AutoArrayHashMapUnmanaged(usize, Memory.Kind),
        id_to_input_memory_kind: std.AutoArrayHashMapUnmanaged(usize, Memory.Kind),
        arena: std.heap.ArenaAllocator,

        pub fn initFromBlock(allocator: std.mem.Allocator, block: *mlir.Block) Scope {
            const arena: std.heap.ArenaAllocator = .init(allocator);
            return .{
                .block = block,
                .id_to_argument = .empty,
                .id_to_donation = .empty,
                .id_to_output_memory_kind = .empty,
                .id_to_input_memory_kind = .empty,
                .arena = arena,
            };
        }

        pub fn deinit(self: *Scope) void {
            self.arena.deinit();
        }
    };

    allocator: std.mem.Allocator,
    io: std.Io,
    arena: std.heap.ArenaAllocator,

    mlir_registry: *mlir.DialectRegistry,
    mlir_ctx: *mlir.Context,
    mlir_pass_manager: *mlir.PassManager,
    module: *mlir.Module,
    platform: *const Platform,
    partitioning: Sharding.Partitioning,

    scopes: stdx.BoundedArray(Scope, 16) = .empty,
    manual_computation_depth: usize = 0,

    channel_id: i64 = 0,

    composite_id: i64 = 0,

    threadlocal var _current: ?*CompilationContext = null;

    pub fn init(allocator: std.mem.Allocator, io: std.Io, platform: *const Platform, opts: CompilationOptions) CompilationContext {
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
            .module = module,
            .platform = platform,
            .partitioning = partitioning,
        };
    }

    pub fn deinit(self: *CompilationContext) void {
        if (_current == self) _current = null;
        for (self.scopes.slice()) |*scope| {
            scope.deinit();
        }
        self.mlir_pass_manager.deinit();
        self.module.deinit();
        self.mlir_ctx.deinit();
        self.arena.deinit();
    }

    pub fn activate(self: *CompilationContext) void {
        std.debug.assert(_current == null);
        _current = self;
    }

    pub fn deactivate(self: *CompilationContext) void {
        _ = self;
        _current = null;
    }

    pub fn current() *CompilationContext {
        return _current.?;
    }

    pub fn currentOrNull() ?*CompilationContext {
        return _current;
    }

    pub fn currentScope(self: *CompilationContext) *Scope {
        return &self.scopes.slice()[self.scopes.len - 1];
    }

    pub fn pushBlock(self: *CompilationContext, block: *mlir.Block) void {
        const scope = Scope.initFromBlock(self.allocator, block);
        self.scopes.appendAssumeCapacity(scope);
    }

    pub fn popBlock(self: *CompilationContext) void {
        var maybe_popped_scope = self.scopes.pop();
        if (maybe_popped_scope) |*popped| {
            popped.deinit();
        }
    }

    pub fn nextChannelId(self: *CompilationContext) i64 {
        self.channel_id += 1;
        return self.channel_id;
    }

    pub fn nextCompositeId(self: *CompilationContext) i64 {
        self.composite_id += 1;
        return self.composite_id;
    }

    pub fn abortOOM(self: *CompilationContext) noreturn {
        _ = self;
        @panic("OOM");
    }

    pub fn alloc(self: *CompilationContext, T: type, n: usize) []T {
        return self.arena.allocator().alloc(T, n) catch self.abortOOM();
    }

    pub fn allocPrint(self: *CompilationContext, comptime fmt: []const u8, args: anytype) []u8 {
        return std.fmt.allocPrint(self.arena.allocator(), fmt, args) catch self.abortOOM();
    }
};

pub fn Compiler(comptime func: anytype) type {
    return struct {
        pub fn compile(
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const Platform,
            opts: CompilationOptions,
            args: std.meta.ArgsTuple(@TypeOf(func)),
        ) CompileError!Exe {
            return zml_module.compile(allocator, io, func, args, platform, opts);
        }
    };
}

pub fn compile(
    allocator: std.mem.Allocator,
    io: std.Io,
    comptime func: anytype,
    args: std.meta.ArgsTuple(@TypeOf(func)),
    platform: *const Platform,
    opts: CompilationOptions,
) CompileError!Exe {
    // TODO: Here we have somewhat of a requirement
    // Emitting MLIR requires to have the compilation context available at all times using `CompilationContext.current()`.
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

    var compilation_context: CompilationContext = .init(allocator, st_io.io(), platform, opts);
    defer compilation_context.deinit();

    const result = emitMlir(&compilation_context, func, args) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => unreachable,
    };
    defer result.output_info.deinit(compilation_context.allocator);
    defer result.input_info.deinit(compilation_context.allocator);

    if (opts.execution_timing == .device) {
        try instrumentExecutionTiming(
            &compilation_context,
            result.func.region(0).firstBlock().?,
            result.output_info.values,
            result.output_info.shapes,
        );
    }

    try addPartitionerOperations(&compilation_context);

    _ = result.func.appendTo(compilation_context.module.body());

    const num_partitions = compilation_context.partitioning.numPartitions();
    const num_replicas = compilation_context.partitioning.numReplicas();
    const num_devices = compilation_context.partitioning.numDevices();

    compilation_context.module.operation().setAttributeByName(
        "mhlo.num_partitions",
        .int(compilation_context.mlir_ctx, .i32, num_partitions),
    );
    compilation_context.module.operation().setAttributeByName(
        "mhlo.num_replicas",
        .int(compilation_context.mlir_ctx, .i32, num_replicas),
    );

    compilation_context.mlir_pass_manager.runOnOp(compilation_context.module.operation()) catch |err| switch (err) {
        error.MlirUnexpected => {
            std.log.err("Failed to canonicalize invalid mlir: \n {f} \n ", .{compilation_context.module.operation()});
            @panic("ZML generated invalid mlir. Please open a bug report");
        },
    };

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const loaded_executable = try compileModuleToPjrtExecutable(arena.allocator(), st_io.io(), platform, compilation_context.module, compilation_context.partitioning, opts);
    log.debug("\n******** ZML generated MLIR ********\n{f}", .{compilation_context.module.operation()});

    var exe = try Exe.init(
        allocator,
        platform,
        loaded_executable,
        @intCast(num_devices),
        num_partitions,
        result.input_info.shapes,
        result.output_info.shapes,
        result.input_info.shardings,
        result.output_info.shardings,
    );
    errdefer exe.deinit();
    exe.requires_execution_timer = opts.execution_timing == .device;

    return exe;
}

fn addPartitionerOperations(ctx: *CompilationContext) !void {
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

pub const OutputInfo = struct {
    shapes: []Shape,
    shardings: []Sharding,
    values: []*const mlir.Value,
    donations: []?usize,
    output_memory_kinds: []Memory.Kind,

    pub fn deinit(self: OutputInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.shapes);
        allocator.free(self.shardings);
        allocator.free(self.values);
        allocator.free(self.donations);
        allocator.free(self.output_memory_kinds);
    }
};

fn collectOutputInfo(allocator: std.mem.Allocator, partitioning: Sharding.Partitioning, v: anytype) !OutputInfo {
    const LocalContext = struct {
        shape_list: *std.array_list.Managed(Shape),
        sharding_list: *std.array_list.Managed(Sharding),
        value_list: *std.array_list.Managed(*const mlir.Value),
        donation_list: *std.array_list.Managed(?usize),
        output_memory_kind_list: *std.array_list.Managed(Memory.Kind),
        partitioning: Sharding.Partitioning,
    };

    var shape_list = std.array_list.Managed(Shape).init(allocator);
    errdefer shape_list.deinit();
    var sharding_list = std.array_list.Managed(Sharding).init(allocator);
    errdefer sharding_list.deinit();
    var value_list = std.array_list.Managed(*const mlir.Value).init(allocator);
    errdefer value_list.deinit();
    var donation_list = std.array_list.Managed(?usize).init(allocator);
    errdefer donation_list.deinit();
    var output_memory_kind_list = std.array_list.Managed(Memory.Kind).init(allocator);
    errdefer output_memory_kind_list.deinit();

    var context: LocalContext = .{
        .shape_list = &shape_list,
        .sharding_list = &sharding_list,
        .value_list = &value_list,
        .donation_list = &donation_list,
        .output_memory_kind_list = &output_memory_kind_list,
        .partitioning = partitioning,
    };

    try meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *const Tensor) !void {
            try ctx_.shape_list.append(tensor.shape());
            try ctx_.sharding_list.append(try ctx_.partitioning.selectSharding(tensor.shape()));
            try ctx_.value_list.append(tensor.value());
            try ctx_.donation_list.append(tensor.donation());
            try ctx_.output_memory_kind_list.append(tensor.outputMemoryKind());
        }
    }.cb, &context, v);

    return .{
        .shapes = try shape_list.toOwnedSlice(),
        .shardings = try sharding_list.toOwnedSlice(),
        .values = try value_list.toOwnedSlice(),
        .donations = try donation_list.toOwnedSlice(),
        .output_memory_kinds = try output_memory_kind_list.toOwnedSlice(),
    };
}

pub const InputInfo = struct {
    shapes: []Shape,
    shardings: []Sharding,
    memory_kinds: []?Memory.Kind,

    pub fn deinit(self: InputInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.shapes);
        allocator.free(self.shardings);
        allocator.free(self.memory_kinds);
    }
};

fn collectInputInfo(allocator: std.mem.Allocator, partitioning: Sharding.Partitioning, v: anytype) !InputInfo {
    const LocalContext = struct {
        shape_list: *std.array_list.Managed(Shape),
        sharding_list: *std.array_list.Managed(Sharding),
        memory_kind_list: *std.array_list.Managed(?Memory.Kind),
        partitioning: Sharding.Partitioning,
    };

    var shape_list = std.array_list.Managed(Shape).init(allocator);
    errdefer shape_list.deinit();

    var sharding_list = std.array_list.Managed(Sharding).init(allocator);
    errdefer sharding_list.deinit();

    var memory_kind_list = std.array_list.Managed(?Memory.Kind).init(allocator);
    errdefer memory_kind_list.deinit();

    var context: LocalContext = .{
        .shape_list = &shape_list,
        .sharding_list = &sharding_list,
        .memory_kind_list = &memory_kind_list,
        .partitioning = partitioning,
    };

    try meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *const Tensor) !void {
            try ctx_.shape_list.append(tensor.shape());
            try ctx_.sharding_list.append(try ctx_.partitioning.selectSharding(tensor.shape()));
            try ctx_.memory_kind_list.append(tensor.inputMemoryKind());
        }
    }.cb, &context, v);

    return .{
        .shapes = try shape_list.toOwnedSlice(),
        .shardings = try sharding_list.toOwnedSlice(),
        .memory_kinds = try memory_kind_list.toOwnedSlice(),
    };
}

const EmitMlirResult = struct {
    func: *mlir.Operation,
    input_info: InputInfo,
    output_info: OutputInfo,
};

fn finalizeAttributeList(allocator_: std.mem.Allocator, mlir_ctx: *mlir.Context, attributes: []AttributeList) ![]*const mlir.Attribute {
    const res = try allocator_.alloc(*const mlir.Attribute, attributes.len);
    for (res, attributes) |*r, attr| {
        r.* = .dict(mlir_ctx, attr.constSlice());
    }
    return res;
}

fn emitMlir(compilation_context: *CompilationContext, comptime func: anytype, args: std.meta.ArgsTuple(@TypeOf(func))) !EmitMlirResult {
    var arena = std.heap.ArenaAllocator.init(compilation_context.allocator);
    defer arena.deinit();

    const module = mlir.Module.init(.unknown(compilation_context.mlir_ctx));
    errdefer module.deinit();

    const block = mlir.Block.init(&.{}, &.{});
    errdefer block.deinit();

    compilation_context.pushBlock(block);
    defer compilation_context.popBlock();

    const LocalContext = struct {
        compilation_context: *CompilationContext,
        current_argument_id: usize = 0,
    };
    var context: LocalContext = .{
        .compilation_context = compilation_context,
    };
    meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *const Tensor) void {
            const mlir_type = mlirx.Type.rankedTensor(ctx_.compilation_context.mlir_ctx, tensor.shape());
            _ = ctx_.compilation_context.currentScope().block.addArgument(mlir_type, .unknown(ctx_.compilation_context.mlir_ctx));
            const gop = ctx_.compilation_context.currentScope().id_to_argument.getOrPut(ctx_.compilation_context.currentScope().arena.allocator(), tensor.id) catch unreachable;
            if (gop.found_existing) std.debug.panic("Tensor with id {} has already been used once as an argument", .{tensor.id});
            gop.value_ptr.* = ctx_.current_argument_id;
            ctx_.current_argument_id += 1;
        }
    }.cb, &context, &args);

    const output_info, const input_info = b: {
        compilation_context.activate();
        defer compilation_context.deactivate();

        const result = @call(.auto, func, args);

        const input_info = try collectInputInfo(compilation_context.allocator, compilation_context.partitioning, &args);
        errdefer input_info.deinit(compilation_context.allocator);

        const output_info = try collectOutputInfo(compilation_context.allocator, compilation_context.partitioning, &result);
        errdefer output_info.deinit(compilation_context.allocator);

        break :b .{ output_info, input_info };
    };
    errdefer input_info.deinit(compilation_context.allocator);
    errdefer output_info.deinit(compilation_context.allocator);

    const input_attributes = try arena.allocator().alloc(AttributeList, input_info.shapes.len);
    @memset(input_attributes, .empty);

    const output_attributes = try arena.allocator().alloc(AttributeList, output_info.shapes.len);
    @memset(output_attributes, .empty);

    for (output_info.donations, 0..) |donation, index| if (donation) |argument_index| {
        input_attributes[argument_index].appendAssumeCapacity(.named(compilation_context.mlir_ctx, "tf.aliasing_output", .int(compilation_context.mlir_ctx, .i32, index)));
    };
    for (output_info.output_memory_kinds, 0..) |output_memory_kind, index| {
        if (output_memory_kind == .device) continue;
        output_attributes[index].appendAssumeCapacity(.named(
            compilation_context.mlir_ctx,
            "mhlo.memory_kind",
            .string(
                compilation_context.mlir_ctx,
                compilation_context.platform.memoryKind(output_memory_kind),
            ),
        ));
    }
    _ = dialects.func.returns(compilation_context.mlir_ctx, output_info.values, .unknown(compilation_context.mlir_ctx)).appendTo(compilation_context.currentScope().block);

    for (input_info.shapes, input_info.shardings, input_info.memory_kinds, 0..) |shape, sharding, maybe_memory_kind, i| {
        const attr = try compilation_context.partitioning.tensorShardingAttr(compilation_context.arena.allocator(), compilation_context.mlir_ctx, shape, sharding);
        const name = switch (compilation_context.partitioning.partitioner) {
            .gspmd => "mhlo.sharding",
            .shardy => "sdy.sharding",
        };

        input_attributes[i].appendAssumeCapacity(.named(compilation_context.mlir_ctx, name, attr));

        if (maybe_memory_kind) |memory_kind| {
            if (memory_kind == .device) continue;
            input_attributes[i].appendAssumeCapacity(.named(
                compilation_context.mlir_ctx,
                "mhlo.memory_kind",
                .string(
                    compilation_context.mlir_ctx,
                    compilation_context.platform.memoryKind(memory_kind),
                ),
            ));
        }
    }

    for (output_info.shapes, output_info.shardings, 0..) |shape, sharding, i| {
        const attr = try compilation_context.partitioning.tensorShardingAttr(compilation_context.arena.allocator(), compilation_context.mlir_ctx, shape, sharding);
        const name = switch (compilation_context.partitioning.partitioner) {
            .gspmd => "mhlo.sharding",
            .shardy => "sdy.sharding",
        };

        output_attributes[i].appendAssumeCapacity(.named(compilation_context.mlir_ctx, name, attr));
    }

    const mlir_func = dialects.func.func(compilation_context.mlir_ctx, .{
        .name = "main",
        .block = compilation_context.currentScope().block,
        .location = .unknown(compilation_context.mlir_ctx),
        .args_attributes = try finalizeAttributeList(arena.allocator(), compilation_context.mlir_ctx, input_attributes),
        .results_attributes = try finalizeAttributeList(arena.allocator(), compilation_context.mlir_ctx, output_attributes),
        .verify = false,
    });

    return .{
        .func = mlir_func,
        .input_info = input_info,
        .output_info = output_info,
    };
}

/// Inserts a dependency chain which brackets all input-dependent entry
/// function work with side-effecting typed FFI calls. The start call produces
/// a marker buffer initialized on the FFI stream; routing every argument
/// through the optimization barrier prevents the program body from moving
/// before that marker. The stop call consumes every flattened output.
fn instrumentExecutionTiming(
    ctx: *CompilationContext,
    block: *mlir.Block,
    outputs: []const *const mlir.Value,
    output_shapes: []const Shape,
) error{UnsupportedExecutionTiming}!void {
    if (ctx.platform.target != .cuda and ctx.platform.target != .rocm) {
        return error.UnsupportedExecutionTiming;
    }
    if (block.numArguments() == 0 or outputs.len == 0 or outputs.len != output_shapes.len or outputs.len > dialects.stablehlo.CustomCallOpts.MAX_OPERANDS) {
        return error.UnsupportedExecutionTiming;
    }

    const allocator = ctx.arena.allocator();
    const mlir_ctx = ctx.mlir_ctx;
    const marker_shape = Shape.scalar(.u8);
    const marker_type: *const mlir.Type = .rankedTensor(&.{}, .int(mlir_ctx, .u8));
    const first_op = block.firstOperation().?;
    const terminator = block.terminator().?;

    const start_marker = switch (ctx.partitioning.partitioner) {
        .shardy => try insertShardyTimingStart(ctx, marker_shape, marker_type, first_op),
        .gspmd => try insertGspmdTimingStart(ctx, marker_shape, marker_type, first_op),
    };
    _ = insertTimingBarrier(allocator, mlir_ctx, block, first_op, start_marker);

    // Read return operands after rewriting argument uses. This matters for an
    // identity-like program whose output is itself a function argument.
    const stop_inputs = allocator.alloc(*const mlir.Value, outputs.len) catch @panic("OOM");
    defer allocator.free(stop_inputs);
    for (stop_inputs, 0..) |*input, i| input.* = terminator.operand(i);

    switch (ctx.partitioning.partitioner) {
        .shardy => _ = try insertShardyTimingStop(ctx, stop_inputs, output_shapes, marker_type, terminator),
        .gspmd => _ = try insertGspmdTimingStop(ctx, stop_inputs, output_shapes, marker_type, terminator),
    }
}

/// XLA does not allow a generic side-effecting custom call to have replicated
/// or tiled sharding. A Shardy manual computation is lowered as local code in
/// every partition, which gives the marker access to each partition's local
/// stream without registering an XLA custom-call partitioner.
fn insertShardyTimingStart(
    ctx: *CompilationContext,
    marker_shape: Shape,
    marker_type: *const mlir.Type,
    before: *mlir.Operation,
) error{UnsupportedExecutionTiming}!*const mlir.Value {
    const allocator = ctx.arena.allocator();
    const mlir_ctx = ctx.mlir_ctx;
    const location = mlir.Location.unknown(mlir_ctx);
    const empty_per_value = dialects.shardy.TensorShardingPerValueAttribute.init(mlir_ctx, &.{}).asAttr();
    const out_shardings = ctx.partitioning.sdyPerValueShardingAttr(allocator, mlir_ctx, &.{marker_shape}) catch return error.UnsupportedExecutionTiming;
    const manual_axes = ctx.partitioning.sdyManualAxesAttr(allocator, mlir_ctx, &.{}, &.{marker_shape}) catch return error.UnsupportedExecutionTiming;

    const manual_block = mlir.Block.init(&.{}, &.{});
    const start = timingStartCall(mlir_ctx, marker_type, &.{}).appendTo(manual_block);
    _ = mlir.Operation.make(mlir_ctx, "sdy.return", .{
        .operands = .{ .flat = &.{start.result(0)} },
        .verify = false,
        .location = location,
    }).appendTo(manual_block);

    const manual = mlir.Operation.make(mlir_ctx, "sdy.manual_computation", .{
        .results = .{ .flat = &.{marker_type} },
        .blocks = &.{manual_block},
        .attributes = &.{
            .named(mlir_ctx, "in_shardings", empty_per_value),
            .named(mlir_ctx, "out_shardings", out_shardings),
            .named(mlir_ctx, "manual_axes", manual_axes),
        },
        .verify = true,
        .location = location,
    });
    before.block().?.insertOwnedOperationBefore(before, manual);
    return manual.result(0);
}

fn insertShardyTimingStop(
    ctx: *CompilationContext,
    inputs: []const *const mlir.Value,
    input_shapes: []const Shape,
    marker_type: *const mlir.Type,
    before: *mlir.Operation,
) error{UnsupportedExecutionTiming}!*mlir.Operation {
    const allocator = ctx.arena.allocator();
    const mlir_ctx = ctx.mlir_ctx;
    const location = mlir.Location.unknown(mlir_ctx);
    const in_shardings = ctx.partitioning.sdyPerValueShardingAttr(allocator, mlir_ctx, input_shapes) catch return error.UnsupportedExecutionTiming;
    const empty_per_value = dialects.shardy.TensorShardingPerValueAttribute.init(mlir_ctx, &.{}).asAttr();
    const manual_axes = ctx.partitioning.sdyManualAxesAttr(allocator, mlir_ctx, input_shapes, &.{}) catch return error.UnsupportedExecutionTiming;

    const local_types = allocator.alloc(*const mlir.Type, input_shapes.len) catch @panic("OOM");
    defer allocator.free(local_types);
    const local_locations = allocator.alloc(*const mlir.Location, input_shapes.len) catch @panic("OOM");
    defer allocator.free(local_locations);
    for (input_shapes, local_types, local_locations) |shape, *local_type, *local_location| {
        const local_shape = ctx.partitioning.localShapeForShape(shape) catch return error.UnsupportedExecutionTiming;
        local_type.* = mlirx.Type.rankedTensor(mlir_ctx, local_shape);
        local_location.* = location;
    }

    const manual_block = mlir.Block.init(local_types, local_locations);
    const local_inputs = allocator.alloc(*const mlir.Value, inputs.len) catch @panic("OOM");
    defer allocator.free(local_inputs);
    for (local_inputs, 0..) |*input, i| input.* = manual_block.argument(i);
    const stop = timingStopCall(mlir_ctx, local_inputs, marker_type, &.{}).appendTo(manual_block);
    _ = mlir.Operation.make(mlir_ctx, "sdy.return", .{
        .verify = false,
        .location = location,
    }).appendTo(manual_block);

    const manual = mlir.Operation.make(mlir_ctx, "sdy.manual_computation", .{
        .operands = .{ .flat = inputs },
        .blocks = &.{manual_block},
        .attributes = &.{
            .named(mlir_ctx, "in_shardings", in_shardings),
            .named(mlir_ctx, "out_shardings", empty_per_value),
            .named(mlir_ctx, "manual_axes", manual_axes),
        },
        .verify = true,
        .location = location,
    });
    before.block().?.insertOwnedOperationBefore(before, manual);
    return stop;
}

fn insertGspmdTimingStart(
    ctx: *CompilationContext,
    marker_shape: Shape,
    marker_type: *const mlir.Type,
    before: *mlir.Operation,
) error{UnsupportedExecutionTiming}!*const mlir.Value {
    const mlir_ctx = ctx.mlir_ctx;
    const location = mlir.Location.unknown(mlir_ctx);
    const manual_sharding = mlir.NamedAttribute.named(mlir_ctx, "mhlo.sharding", .string(mlir_ctx, "{manual}"));
    const start = timingStartCall(mlir_ctx, marker_type, &.{manual_sharding});
    before.block().?.insertOwnedOperationBefore(before, start);

    const global_sharding = ctx.partitioning.tensorShardingAttr(ctx.arena.allocator(), mlir_ctx, marker_shape, null) catch return error.UnsupportedExecutionTiming;
    const shard_to_full = dialects.stablehlo.custom_call(
        mlir_ctx,
        &.{start.result(0)},
        &.{marker_type},
        .{
            .call_target_name = "SPMDShardToFullShape",
            .has_side_effect = false,
            .backend_config = .{ .original = "" },
            .additional_attributes = &.{.named(mlir_ctx, "mhlo.sharding", global_sharding)},
        },
        location,
    );
    before.block().?.insertOwnedOperationBefore(before, shard_to_full);
    return shard_to_full.result(0);
}

fn insertGspmdTimingStop(
    ctx: *CompilationContext,
    inputs: []const *const mlir.Value,
    input_shapes: []const Shape,
    marker_type: *const mlir.Type,
    before: *mlir.Operation,
) error{UnsupportedExecutionTiming}!*mlir.Operation {
    const allocator = ctx.arena.allocator();
    const mlir_ctx = ctx.mlir_ctx;
    const location = mlir.Location.unknown(mlir_ctx);
    const local_inputs = allocator.alloc(*const mlir.Value, inputs.len) catch @panic("OOM");
    defer allocator.free(local_inputs);

    for (inputs, input_shapes, local_inputs) |input, shape, *local_input| {
        const local_shape = ctx.partitioning.localShapeForShape(shape) catch return error.UnsupportedExecutionTiming;
        const local_type = mlirx.Type.rankedTensor(mlir_ctx, local_shape);
        const full_to_shard = dialects.stablehlo.custom_call(
            mlir_ctx,
            &.{input},
            &.{local_type},
            .{
                .call_target_name = "SPMDFullToShardShape",
                .has_side_effect = false,
                .backend_config = .{ .original = "" },
                .additional_attributes = &.{.named(mlir_ctx, "mhlo.sharding", .string(mlir_ctx, "{manual}"))},
            },
            location,
        );
        before.block().?.insertOwnedOperationBefore(before, full_to_shard);
        local_input.* = full_to_shard.result(0);
    }

    const stop = timingStopCall(
        mlir_ctx,
        local_inputs,
        marker_type,
        &.{.named(mlir_ctx, "mhlo.sharding", .string(mlir_ctx, "{manual}"))},
    );
    before.block().?.insertOwnedOperationBefore(before, stop);
    return stop;
}

fn timingStartCall(
    mlir_ctx: *mlir.Context,
    marker_type: *const mlir.Type,
    additional_attributes: []const mlir.NamedAttribute,
) *mlir.Operation {
    return dialects.stablehlo.custom_call(
        mlir_ctx,
        &.{},
        &.{marker_type},
        .{
            .call_target_name = autotuneStartCallTarget,
            .has_side_effect = true,
            .backend_config = .{ .typed_ffi = .dict(mlir_ctx, &.{}) },
            .additional_attributes = additional_attributes,
        },
        .unknown(mlir_ctx),
    );
}

fn timingStopCall(
    mlir_ctx: *mlir.Context,
    inputs: []const *const mlir.Value,
    marker_type: *const mlir.Type,
    additional_attributes: []const mlir.NamedAttribute,
) *mlir.Operation {
    return dialects.stablehlo.custom_call(
        mlir_ctx,
        inputs,
        &.{marker_type},
        .{
            .call_target_name = autotuneStopCallTarget,
            .has_side_effect = true,
            .backend_config = .{ .typed_ffi = .dict(mlir_ctx, &.{}) },
            .additional_attributes = additional_attributes,
        },
        .unknown(mlir_ctx),
    );
}

fn insertTimingBarrier(
    allocator: std.mem.Allocator,
    mlir_ctx: *mlir.Context,
    block: *mlir.Block,
    before: *mlir.Operation,
    marker: *const mlir.Value,
) *mlir.Operation {
    const barrier_len = block.numArguments() + 1;
    const barrier_inputs = allocator.alloc(*const mlir.Value, barrier_len) catch @panic("OOM");
    defer allocator.free(barrier_inputs);
    const barrier_types = allocator.alloc(*const mlir.Type, barrier_len) catch @panic("OOM");
    defer allocator.free(barrier_types);
    barrier_inputs[0] = marker;
    barrier_types[0] = marker.type_();
    for (1..barrier_len) |i| {
        const argument = block.argument(i - 1);
        barrier_inputs[i] = argument;
        barrier_types[i] = argument.type_();
    }

    const barrier = dialects.stablehlo.optimizationBarrier(
        mlir_ctx,
        barrier_inputs,
        barrier_types,
        .unknown(mlir_ctx),
    );
    block.insertOwnedOperationBefore(before, barrier);

    // Replacing all uses also rewrites the barrier's own operands. Restore
    // those operands afterward so the barrier does not become self-referential.
    for (1..barrier_len) |i| {
        const argument = block.argument(i - 1);
        argument.replaceAllUsesWith(barrier.result(i));
        barrier.setOperand(i, argument);
    }
    return barrier;
}

fn instrumentExecutionTimingBlock(
    allocator: std.mem.Allocator,
    mlir_ctx: *mlir.Context,
    block: *mlir.Block,
    outputs: []const *const mlir.Value,
    marker_type: *const mlir.Type,
    marker_attributes: []const mlir.NamedAttribute,
) struct { start: *mlir.Operation, barrier: *mlir.Operation, stop: *mlir.Operation } {
    std.debug.assert(block.numArguments() > 0);
    std.debug.assert(outputs.len > 0 and outputs.len <= dialects.stablehlo.CustomCallOpts.MAX_OPERANDS);

    const first_op = block.firstOperation().?;
    const terminator = block.terminator().?;
    std.debug.assert(terminator.numOperands() == outputs.len);
    const start = timingStartCall(mlir_ctx, marker_type, marker_attributes);
    block.insertOwnedOperationBefore(first_op, start);
    const barrier = insertTimingBarrier(allocator, mlir_ctx, block, first_op, start.result(0));

    // Read the return operands after rewriting argument uses. This matters for
    // identity-like programs whose output is itself a function argument.
    const stop_inputs = allocator.alloc(*const mlir.Value, outputs.len) catch @panic("OOM");
    defer allocator.free(stop_inputs);
    for (stop_inputs, 0..) |*input, i| input.* = terminator.operand(i);

    const stop = timingStopCall(mlir_ctx, stop_inputs, marker_type, marker_attributes);
    block.insertOwnedOperationBefore(terminator, stop);
    return .{ .start = start, .barrier = barrier, .stop = stop };
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

fn compileModuleToPjrtExecutable(arena: std.mem.Allocator, io: std.Io, platform: *const Platform, module: *const mlir.Module, partitioning: Partitioning, opts: CompilationOptions) !*pjrt.LoadedExecutable {
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
            .rocm, .cuda => if (std.c.getenv("ZML_AUTOTUNE_CACHE_DIR")) |path| {
                try setXlaOverrideFlag(overrides_map, "xla_gpu_experimental_autotuner_cache_dir", std.mem.span(path), upb_arena);
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

test "device timing dependency chain survives canonicalization" {
    var threaded_io: std.Io.Threaded = .init_single_threaded;
    defer threaded_io.deinit();

    const registry = mlirRegistry(threaded_io.io());
    const mlir_ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer mlir_ctx.deinit();
    mlir_ctx.loadAllAvailableDialects();

    const module_ = mlir.Module.init(.unknown(mlir_ctx));
    defer module_.deinit();
    const location = mlir.Location.unknown(mlir_ctx);
    const tensor_type: *const mlir.Type = .rankedTensor(&.{4}, .float(mlir_ctx, .f32));
    const block = mlir.Block.init(&.{tensor_type}, &.{location});
    const body = dialects.stablehlo.add(mlir_ctx, block.argument(0), block.argument(0), location).appendTo(block);
    _ = dialects.func.returns(mlir_ctx, &.{body.result(0)}, location).appendTo(block);
    const func = dialects.func.func(mlir_ctx, .{
        .name = "main",
        .block = block,
        .location = location,
        .verify = false,
    });
    _ = func.appendTo(module_.body());

    const marker_type: *const mlir.Type = .rankedTensor(&.{}, .int(mlir_ctx, .u8));
    const markers = instrumentExecutionTimingBlock(std.testing.allocator, mlir_ctx, block, &.{body.result(0)}, marker_type, &.{});

    const pass_manager = mlir.PassManager.init(mlir_ctx);
    defer pass_manager.deinit();
    try pass_manager.asOpPassManager().addPipeline("canonicalize");
    try pass_manager.runOnOp(module_.operation());

    try std.testing.expectEqualStrings("stablehlo.custom_call", markers.start.name());
    try std.testing.expectEqualStrings("stablehlo.optimization_barrier", markers.barrier.name());
    try std.testing.expectEqualStrings("stablehlo.custom_call", markers.stop.name());
    try std.testing.expect(markers.barrier.operand(0).owner() == markers.start);
    try std.testing.expect(body.operand(0).owner() == markers.barrier);
    try std.testing.expect(body.operand(1).owner() == markers.barrier);
    try std.testing.expect(markers.stop.operand(0).owner() == body);
}

test "device timing stop consumes rewritten identity output" {
    var threaded_io: std.Io.Threaded = .init_single_threaded;
    defer threaded_io.deinit();

    const registry = mlirRegistry(threaded_io.io());
    const mlir_ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer mlir_ctx.deinit();
    mlir_ctx.loadAllAvailableDialects();

    const module_ = mlir.Module.init(.unknown(mlir_ctx));
    defer module_.deinit();
    const location = mlir.Location.unknown(mlir_ctx);
    const tensor_type: *const mlir.Type = .rankedTensor(&.{4}, .float(mlir_ctx, .f32));
    const block = mlir.Block.init(&.{tensor_type}, &.{location});
    _ = dialects.func.returns(mlir_ctx, &.{block.argument(0)}, location).appendTo(block);
    const func = dialects.func.func(mlir_ctx, .{
        .name = "identity",
        .block = block,
        .location = location,
        .verify = false,
    });
    _ = func.appendTo(module_.body());

    const marker_type: *const mlir.Type = .rankedTensor(&.{}, .int(mlir_ctx, .u8));
    const markers = instrumentExecutionTimingBlock(std.testing.allocator, mlir_ctx, block, &.{block.argument(0)}, marker_type, &.{});
    const pass_manager = mlir.PassManager.init(mlir_ctx);
    defer pass_manager.deinit();
    try pass_manager.asOpPassManager().addPipeline("canonicalize");
    try pass_manager.runOnOp(module_.operation());

    try std.testing.expect(markers.stop.operand(0).owner() == markers.barrier);
    try std.testing.expect(block.terminator().?.operand(0).owner() == markers.barrier);
}
