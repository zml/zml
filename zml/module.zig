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
const Shape = @import("shape.zig").Shape;
const Sharding = @import("sharding.zig").Sharding;
const Tensor = @import("tensor.zig").Tensor;

const log = std.log.scoped(.@"zml/module");

pub const CompilationOpts = struct {
    shardings: []const Sharding,
};

const AttributeList = stdx.BoundedArray(mlir.NamedAttribute, 3);

pub const CompilationContext = struct {
    pub const Scope = struct {
        block: *mlir.Block,
        id_to_argument: std.AutoArrayHashMapUnmanaged(usize, usize),
        id_to_donation: std.AutoArrayHashMapUnmanaged(usize, usize),
        id_to_output_memory_kind: std.AutoArrayHashMapUnmanaged(usize, Memory.Kind),
        arena: std.heap.ArenaAllocator,

        pub fn initFromBlock(allocator: std.mem.Allocator, block: *mlir.Block) Scope {
            const arena: std.heap.ArenaAllocator = .init(allocator);
            return .{
                .block = block,
                .id_to_argument = .empty,
                .id_to_donation = .empty,
                .id_to_output_memory_kind = .empty,
                .arena = arena,
            };
        }

        pub fn deinit(self: *Scope) void {
            self.arena.deinit();
        }
    };

    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,

    mlir_registry: *mlir.DialectRegistry,
    mlir_ctx: *mlir.Context,
    mlir_pass_manager: *mlir.PassManager,
    module: *mlir.Module,
    platform: *const Platform,
    opts: CompilationOpts,

    scopes: stdx.BoundedArray(Scope, 16) = .{},

    threadlocal var _current: ?*CompilationContext = null;

    var mlir_once = std.once(struct {
        fn call() void {
            mlir.registerPasses("Transforms");
        }
    }.call);

    pub fn init(allocator: std.mem.Allocator, platform: *const Platform, opts: CompilationOpts) CompilationContext {
        mlir_once.call();
        const mlir_registry = mlir.DialectRegistry.init() catch unreachable;
        inline for (.{ "func", "stablehlo", "sdy" }) |d| {
            mlir.DialectHandle.fromString(d).insertDialect(mlir_registry);
        }
        var mlir_ctx = mlir.Context.init(.{ .registry = mlir_registry, .threading = false }) catch unreachable;
        mlir_ctx.loadAllAvailableDialects();
        mlir_ctx.allowUnregisteredDialects(true);

        //const loc = mlir.Location.fromSrc(mlir_ctx, @src()).named(mlir_ctx, "main");
        const module = mlir.Module.init(.unknown(mlir_ctx));
        module.operation().setAttributeByName("sym_name", mlir.stringAttribute(mlir_ctx, "zml"));

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

        return .{
            .allocator = allocator,
            .arena = std.heap.ArenaAllocator.init(allocator),
            .mlir_registry = mlir_registry,
            .mlir_ctx = mlir_ctx,
            .mlir_pass_manager = pass_manager,
            .module = module,
            .platform = platform,
            .opts = opts,
        };
    }

    pub fn deinit(self: *CompilationContext) void {
        self.mlir_pass_manager.deinit();
        self.module.deinit();
        self.mlir_ctx.deinit();
        self.mlir_registry.deinit();
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
};

// todo: move
fn addShardyMeshes(compilation_context: *CompilationContext, shardings: []const Sharding) !void {
    var seen: std.StringHashMap(void) = .init(compilation_context.allocator);
    defer seen.deinit();

    for (shardings) |sharding| {
        const name = sharding.meshName();
        if (seen.contains(name)) continue;
        try seen.put(name, {});

        const mesh_attr_str = try sharding.meshAttrString(compilation_context.arena.allocator());
        const mesh_attr = try mlir.Attribute.parse(compilation_context.mlir_ctx, mesh_attr_str);

        const mesh_op = mlir.Operation.make(compilation_context.mlir_ctx, "sdy.mesh", .{
            .attributes = &.{
                .named(compilation_context.mlir_ctx, "sym_name", mlir.stringAttribute(compilation_context.mlir_ctx, name)),
                .named(compilation_context.mlir_ctx, "mesh", @ptrCast(mesh_attr)),
            },
            .location = .unknown(compilation_context.mlir_ctx),
            .verify = false,
        });

        _ = mesh_op.appendTo(compilation_context.module.body());
    }
}

pub fn compile(allocator: std.mem.Allocator, io: std.Io, comptime func: anytype, args: stdx.meta.FnArgs(func), platform: *const Platform, opts: CompilationOpts) !Exe {
    var compilation_context: CompilationContext = .init(allocator, platform, opts);
    defer compilation_context.deinit();

    const result = emitMlir(&compilation_context, func, args, opts) catch unreachable;
    defer result.output_info.deinit(compilation_context.allocator);
    defer result.input_info.deinit(compilation_context.allocator);

    try addShardyMeshes(&compilation_context, opts.shardings);

    _ = result.func.appendTo(compilation_context.module.body());

    compilation_context.module.operation().setAttributeByName("mhlo.num_partitions", mlir.integerAttribute(compilation_context.mlir_ctx, .i32, opts.shardings[0].numPartitions()));
    compilation_context.module.operation().setAttributeByName("mhlo.num_replicas", mlir.integerAttribute(compilation_context.mlir_ctx, .i32, opts.shardings[0].numReplicas()));

    compilation_context.mlir_pass_manager.runOnOp(compilation_context.module.operation()) catch |err| switch (err) {
        error.MlirUnexpected => {
            std.log.err("Failed to canonicalize invalid mlir: \n {f} \n ", .{compilation_context.module.operation()});
            @panic("ZML generated invalid mlir. Please open a bug report");
        },
    };

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const loaded_executable = compileModuleToPjrtExecutable(arena.allocator(), io, platform, compilation_context.module, opts) catch unreachable;

    log.info("\n******** ZML generated MLIR ********\n{f}", .{compilation_context.module.operation()});

    const exe = try Exe.init(allocator, platform, loaded_executable, result.input_info.shapes, result.output_info.shapes, result.input_info.shardings, result.output_info.shardings);
    errdefer exe.deinit();

    return exe;
}

fn shardingCoversShape(sharding: Sharding, shape: Shape) bool {
    for (0..shape.rank()) |ax| {
        switch (shape.partition(ax)) {
            .axis => |tag| if (sharding.binding(tag) == null) return false,
            else => {},
        }
    }
    return true;
}

fn selectSharding(shape: Shape, shardings: []const Sharding) !Sharding {
    for (shardings) |s| {
        if (shardingCoversShape(s, shape)) return s;
    }

    return error.NoSuitableSharding;
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

fn collectOutputInfo(allocator: std.mem.Allocator, shardings: []const Sharding, v: anytype) !OutputInfo {
    const LocalContext = struct {
        shape_list: *std.array_list.Managed(Shape),
        sharding_list: *std.array_list.Managed(Sharding),
        value_list: *std.array_list.Managed(*const mlir.Value),
        donation_list: *std.array_list.Managed(?usize),
        output_memory_kind_list: *std.array_list.Managed(Memory.Kind),
        shardings: []const Sharding,
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
        .shardings = shardings,
    };

    try meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *const Tensor) !void {
            try ctx_.shape_list.append(tensor.shape());
            try ctx_.sharding_list.append(try selectSharding(tensor.shape(), ctx_.shardings));
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

    pub fn deinit(self: InputInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.shapes);
        allocator.free(self.shardings);
    }
};

fn collectInputInfo(allocator: std.mem.Allocator, shardings: []const Sharding, v: anytype) !InputInfo {
    const LocalContext = struct {
        shape_list: *std.array_list.Managed(Shape),
        sharding_list: *std.array_list.Managed(Sharding),
        shardings: []const Sharding,
    };

    var shape_list = std.array_list.Managed(Shape).init(allocator);
    errdefer shape_list.deinit();

    var sharding_list = std.array_list.Managed(Sharding).init(allocator);
    errdefer sharding_list.deinit();

    var context: LocalContext = .{
        .shape_list = &shape_list,
        .sharding_list = &sharding_list,
        .shardings = shardings,
    };

    try meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *const Tensor) !void {
            try ctx_.shape_list.append(tensor.shape());
            try ctx_.sharding_list.append(try selectSharding(tensor.shape(), ctx_.shardings));
        }
    }.cb, &context, v);

    return .{
        .shapes = try shape_list.toOwnedSlice(),
        .shardings = try sharding_list.toOwnedSlice(),
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
        r.* = mlir.dictionaryAttribute(mlir_ctx, attr.constSlice());
    }
    return res;
}

fn emitMlir(compilation_context: *CompilationContext, comptime func: anytype, args: stdx.meta.FnArgs(func), opts: CompilationOpts) !EmitMlirResult {
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
            const mlir_type = mlir.rankedTensorType(
                tensor.dims(),
                mlirx.Type.fromDType(ctx_.compilation_context.mlir_ctx, tensor.dtype()),
            );
            _ = ctx_.compilation_context.currentScope().block.addArgument(mlir_type, .unknown(ctx_.compilation_context.mlir_ctx));
            ctx_.compilation_context.currentScope().id_to_argument.put(ctx_.compilation_context.currentScope().arena.allocator(), tensor.id, ctx_.current_argument_id) catch unreachable;
            ctx_.current_argument_id += 1;
        }
    }.cb, &context, &args);

    const input_info = try collectInputInfo(compilation_context.allocator, opts.shardings, &args);
    errdefer input_info.deinit(compilation_context.allocator);

    const input_attributes = try arena.allocator().alloc(AttributeList, input_info.shapes.len);
    @memset(input_attributes, .{});

    const output_info = b: {
        compilation_context.activate();
        defer compilation_context.deactivate();

        const result = @call(.auto, func, args);

        const output_info = try collectOutputInfo(compilation_context.allocator, opts.shardings, &result);
        errdefer output_info.deinit(compilation_context.allocator);

        break :b output_info;
    };
    errdefer output_info.deinit(compilation_context.allocator);

    const output_attributes = try arena.allocator().alloc(AttributeList, output_info.shapes.len);
    @memset(output_attributes, .{});

    for (output_info.donations, 0..) |donation, index| if (donation) |argument_index| {
        input_attributes[argument_index].appendAssumeCapacity(.named(compilation_context.mlir_ctx, "tf.aliasing_output", mlir.integerAttribute(compilation_context.mlir_ctx, .i32, index)));
    };
    for (output_info.output_memory_kinds, 0..) |output_memory_kind, index| {
        if (output_memory_kind == .device) continue;
        output_attributes[index].appendAssumeCapacity(.named(
            compilation_context.mlir_ctx,
            "mhlo.memory_kind",
            mlir.stringAttribute(
                compilation_context.mlir_ctx,
                compilation_context.platform.memoryKind(output_memory_kind),
            ),
        ));
    }
    _ = dialects.func.returns(compilation_context.mlir_ctx, output_info.values, .unknown(compilation_context.mlir_ctx)).appendTo(compilation_context.currentScope().block);

    for (input_info.shapes, input_info.shardings, 0..) |shape, sharding, i| {
        if (try sharding.shardingAttrForShape(compilation_context.arena.allocator(), shape)) |attr_str| {
            const parsed = try mlir.Attribute.parse(compilation_context.mlir_ctx, attr_str);
            input_attributes[i].appendAssumeCapacity(.named(
                compilation_context.mlir_ctx,
                "sdy.sharding",
                @ptrCast(parsed),
            ));
        }
    }

    for (output_info.shapes, output_info.shardings, 0..) |shape, sharding, i| {
        if (try sharding.shardingAttrForShape(compilation_context.arena.allocator(), shape)) |attr_str| {
            const parsed = try mlir.Attribute.parse(compilation_context.mlir_ctx, attr_str);
            output_attributes[i].appendAssumeCapacity(.named(
                compilation_context.mlir_ctx,
                "sdy.sharding",
                @ptrCast(parsed),
            ));
        }
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

fn compileModuleToPjrtExecutable(arena: std.mem.Allocator, io: std.Io, platform: *const Platform, module: *const mlir.Module, opts: CompilationOpts) !*pjrt.LoadedExecutable {
    var upb_alloc: upb.Allocator = .init(arena);
    const upb_arena = c.upb_Arena_Init(null, 0, upb_alloc.inner());
    defer c.upb_Arena_Free(upb_arena);

    const sharding = opts.shardings[0]; // todo: refactor
    const device_assignment = try sharding.deviceAssignment(arena);
    log.info("Device assignment: {any}", .{device_assignment});

    const options = blk: {
        const options = try upb.new(c.xla_CompileOptionsProto, upb_arena);
        c.xla_CompileOptionsProto_set_executable_build_options(options, executable_build_options_blk: {
            const exec_build_options = try upb.new(c.xla_ExecutableBuildOptionsProto, upb_arena);
            c.xla_ExecutableBuildOptionsProto_set_use_shardy_partitioner(exec_build_options, true); // todo: check partitioner
            c.xla_ExecutableBuildOptionsProto_set_device_ordinal(exec_build_options, -1);
            c.xla_ExecutableBuildOptionsProto_set_num_replicas(exec_build_options, sharding.numReplicas());
            c.xla_ExecutableBuildOptionsProto_set_num_partitions(exec_build_options, sharding.numPartitions());
            c.xla_ExecutableBuildOptionsProto_set_use_spmd_partitioning(exec_build_options, sharding.numDevices() > 1);

            c.xla_ExecutableBuildOptionsProto_set_device_assignment(exec_build_options, device_assignment_blk: {
                const device_assignment_proto = try upb.new(c.xla_DeviceAssignmentProto, upb_arena);

                c.xla_DeviceAssignmentProto_set_replica_count(device_assignment_proto, sharding.numReplicas());
                c.xla_DeviceAssignmentProto_set_computation_count(device_assignment_proto, sharding.numPartitions());

                const computation_devices = c.xla_DeviceAssignmentProto_resize_computation_devices(
                    device_assignment_proto,
                    @intCast(sharding.numPartitions()),
                    upb_arena,
                );

                for (computation_devices[0..@intCast(sharding.numPartitions())], 0..) |*computation_device, i| {
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
                try setXlaOverrideFlag(overrides_map, "xla_gpu_enable_llvm_module_compilation_parallelism", true, upb_arena);
            },
            .rocm => {
                // enable only on CDNA
                try setXlaOverrideFlag(overrides_map, "xla_gpu_enable_triton_gemm", gemm_blk: {
                    const supported: std.StaticStringMap(void) = .initComptime(.{
                        .{ "gfx950", {} },
                        .{ "gfx950", {} },
                        .{ "gfx942", {} },
                        .{ "gfx942", {} },
                        .{ "gfx942", {} },
                        .{ "gfx90a", {} },
                        .{ "gfx90a", {} },
                        .{ "gfx90a", {} },
                        .{ "gfx908", {} },
                        .{ "gfx906", {} },
                        .{ "gfx900", {} },
                    });
                    const first_device = platform.pjrt_client.devices(platform.pjrt_api)[0];
                    const description = first_device.getDescription(platform.pjrt_api);
                    break :gemm_blk supported.has(description.attribute(platform.pjrt_api, "compute_capability").?.string);
                }, upb_arena);
                // Use lld from libllvm instead of invoking the ld.lld binary.
                // This saves us from having to sandbox it.
                try setXlaOverrideFlag(overrides_map, "xla_gpu_use_inprocess_lld", true, upb_arena);
                // Disable command buffer to avoid some weird crashes.
                // This is what AMD recommended in the meantime.
                try setXlaOverrideFlag(overrides_map, "xla_gpu_enable_command_buffer", "", upb_arena);
            },
            else => {},
        }

        if (platform.compilation_options.xla_dump_to) |xla_dump_to| {
            try setXlaOverrideFlag(overrides_map, "xla_dump_to", xla_dump_to, upb_arena);
            try setXlaOverrideFlag(overrides_map, "xla_dump_hlo_as_proto", true, upb_arena);
            if (platform.compilation_options.xla_dump_fusion_visualization) {
                try setXlaOverrideFlag(overrides_map, "xla_dump_fusion_visualization", true, upb_arena);
            }
            if (platform.compilation_options.xla_dump_hlo_pass_re) |re| {
                try setXlaOverrideFlag(overrides_map, "xla_dump_hlo_pass_re", re, upb_arena);
            }
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
