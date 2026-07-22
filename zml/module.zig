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
const pjrtx = @import("pjrtx.zig");
const Platform = @import("platform.zig").Platform;
const tracer = @import("profiling/tracer.zig");
const Shape = @import("shape.zig").Shape;
const Sharding = @import("Sharding.zig");
const Partitioning = Sharding.Partitioning;
const Tensor = @import("tensor.zig").Tensor;

const zml_module = @This();
const log = std.log.scoped(.@"zml/module");

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
};

pub fn Compiler(comptime func: anytype) type {
    return struct {
        pub fn compile(
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const Platform,
            opts: CompilationOptions,
            args: std.meta.ArgsTuple(@TypeOf(func)),
        ) !Exe {
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
) !Exe {
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

    const result = emitMlir(&compilation_context, func, args) catch unreachable;
    defer result.output_info.deinit(compilation_context.allocator);
    defer result.input_info.deinit(compilation_context.allocator);

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

    const exe = try Exe.init(
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
            .rocm, .rocm_hrx => {
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
            .rocm, .rocm_hrx, .cuda => if (std.c.getenv("ZML_AUTOTUNE_CACHE_DIR")) |path| {
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
