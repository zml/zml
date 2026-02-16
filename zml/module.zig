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
const Tensor = @import("tensor.zig").Tensor;

const log = std.log.scoped(.@"zml/module");

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
    //mlir_op_pass_manager: *mlir.OpPassManager,
    module: *mlir.Module,
    platform: *const Platform,

    scopes: stdx.BoundedArray(Scope, 16) = .{},

    threadlocal var _current: ?*CompilationContext = null;

    pub fn init(allocator: std.mem.Allocator, io: std.Io, platform: *const Platform) CompilationContext {
        _ = io; // autofix
        mlir.registerPasses("Transforms");
        const mlir_registry = mlir.DialectRegistry.init() catch unreachable;
        inline for (.{ "func", "stablehlo" }) |d| {
            mlir.DialectHandle.fromString(d).insertDialect(mlir_registry);
        }
        var mlir_ctx = mlir.Context.init(.{ .registry = mlir_registry, .threading = false }) catch unreachable;
        mlir_ctx.loadAllAvailableDialects();

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

    //pub fn addShardingAttributes(
    //    self: *const CompilationContext,
    //    mesh: Mesh,
    //    arg_attrs: []AttributeList,
    //    res_attrs: []AttributeList,
    //    input_shapes: []const Shape,
    //    input_meshes: []const ?Mesh,
    //    output_info: OutputInfo,
    //) void {
    //    if (mesh.isSinglePartition()) return; // No sharding attributes for single partition.

    //    const default_layout = mlir.NamedAttribute.named(self.mlir_ctx, "mhlo.layout_mode", mlir.stringAttribute(self.mlir_ctx, "default"));
    //    for (arg_attrs, input_shapes, input_meshes) |*attr, shape, input_mesh| {
    //        attr.appendAssumeCapacity(default_layout);
    //        if (input_mesh) |m| {
    //            // log.warn("addShardingAttributes for {} with input mesh: {}", .{ shape, m });
    //            attr.appendAssumeCapacity(.named(self.mlir_ctx, "mhlo.sharding", self.getShardingAttr(Sharding.init(m, shape))));
    //        } else {
    //            // log.warn("addShardingAttributes for {} with progran mesh: {}", .{ shape, mesh });
    //            attr.appendAssumeCapacity(.named(self.mlir_ctx, "mhlo.sharding", self.getShardingAttr(Sharding.init(mesh, shape))));
    //        }
    //    }

    //    for (res_attrs, output_info.shapes) |*attr, shape| {
    //        attr.appendAssumeCapacity(default_layout);
    //        attr.appendAssumeCapacity(.named(self.mlir_ctx, "mhlo.sharding", self.getShardingAttr(Sharding.init(mesh, shape))));
    //    }
    //}

    //pub fn getShardingAttr(self: *const CompilationContext, sharding: Sharding) *const mlir.Attribute {
    //    var sharding_str: stdx.BoundedArray(u8, 128) = .{};
    //    sharding.writeShardingRepresentation(sharding_str.writer()) catch unreachable;
    //    return mlir.stringAttribute(self.mlir_ctx, sharding_str.constSlice());
    //}
};

pub fn compile(
    allocator: std.mem.Allocator,
    io: std.Io,
    comptime func: anytype,
    args: std.meta.ArgsTuple(@TypeOf(func)),
    platform: *const Platform,
) !Exe {
    var compilation_context: CompilationContext = .init(allocator, io, platform);
    defer compilation_context.deinit();

    const result = emitMlir(&compilation_context, func, args) catch unreachable;
    defer result.output_info.deinit(compilation_context.allocator);
    defer result.input_info.deinit(compilation_context.allocator);

    _ = result.func.appendTo(compilation_context.module.body());
    const sharding = platform.sharding();
    compilation_context.module.operation().setAttributeByName("mhlo.num_replicas", mlir.integerAttribute(compilation_context.mlir_ctx, .i32, sharding.num_replicas));
    compilation_context.module.operation().setAttributeByName("mhlo.num_partitions", mlir.integerAttribute(compilation_context.mlir_ctx, .i32, sharding.num_partitions));

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const loaded_executable = compileModuleToPjrtExecutable(arena.allocator(), io, platform, compilation_context.module, null) catch unreachable;

    log.debug("\n******** ZML generated MLIR ********\n{f}", .{compilation_context.module.operation()});

    const num_devices = sharding.num_partitions * sharding.num_replicas;
    const exe = try Exe.init(allocator, platform, loaded_executable, result.input_info.shapes, result.output_info.shapes, num_devices);
    errdefer exe.deinit();

    return exe;
}

fn collectShapes(allocator: std.mem.Allocator, v: anytype) ![]Shape {
    const LocalContext = struct {
        list: *std.array_list.Managed(Shape),
    };
    var list = std.array_list.Managed(Shape).init(allocator);
    errdefer list.deinit();

    var context: LocalContext = .{ .list = &list };
    try meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *const Tensor) !void {
            try ctx_.list.append(tensor.shape());
        }
    }.cb, &context, v);

    return try list.toOwnedSlice();
}

pub const OutputInfo = struct {
    shapes: []Shape,
    values: []*const mlir.Value,
    donations: []?usize,
    output_memory_kinds: []Memory.Kind,

    pub fn deinit(self: OutputInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.shapes);
        allocator.free(self.values);
        allocator.free(self.donations);
        allocator.free(self.output_memory_kinds);
    }
};

fn collectOutputInfo(allocator: std.mem.Allocator, v: anytype) !OutputInfo {
    const LocalContext = struct {
        shape_list: *std.array_list.Managed(Shape),
        value_list: *std.array_list.Managed(*const mlir.Value),
        donation_list: *std.array_list.Managed(?usize),
        output_memory_kind_list: *std.array_list.Managed(Memory.Kind),
    };

    var shape_list = std.array_list.Managed(Shape).init(allocator);
    errdefer shape_list.deinit();
    var value_list = std.array_list.Managed(*const mlir.Value).init(allocator);
    errdefer value_list.deinit();
    var donation_list = std.array_list.Managed(?usize).init(allocator);
    errdefer donation_list.deinit();
    var output_memory_kind_list = std.array_list.Managed(Memory.Kind).init(allocator);
    errdefer output_memory_kind_list.deinit();

    var context: LocalContext = .{
        .shape_list = &shape_list,
        .value_list = &value_list,
        .donation_list = &donation_list,
        .output_memory_kind_list = &output_memory_kind_list,
    };

    try meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *const Tensor) !void {
            try ctx_.shape_list.append(tensor.shape());
            try ctx_.value_list.append(tensor.value());
            try ctx_.donation_list.append(tensor.donation());
            try ctx_.output_memory_kind_list.append(tensor.outputMemoryKind());
        }
    }.cb, &context, v);

    return .{
        .shapes = try shape_list.toOwnedSlice(),
        .values = try value_list.toOwnedSlice(),
        .donations = try donation_list.toOwnedSlice(),
        .output_memory_kinds = try output_memory_kind_list.toOwnedSlice(),
    };
}

pub const InputInfo = struct {
    shapes: []Shape,

    pub fn deinit(self: InputInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.shapes);
    }
};

fn collectInputInfo(allocator: std.mem.Allocator, v: anytype) !InputInfo {
    const LocalContext = struct {
        shape_list: *std.array_list.Managed(Shape),
    };

    var shape_list = std.array_list.Managed(Shape).init(allocator);
    errdefer shape_list.deinit();

    var context: LocalContext = .{ .shape_list = &shape_list };

    try meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *const Tensor) !void {
            try ctx_.shape_list.append(tensor.shape());
        }
    }.cb, &context, v);

    return .{
        .shapes = try shape_list.toOwnedSlice(),
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

fn emitMlir(compilation_context: *CompilationContext, comptime func: anytype, args: stdx.meta.FnArgs(func)) !EmitMlirResult {
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

    const input_info = try collectInputInfo(compilation_context.allocator, &args);
    errdefer input_info.deinit(compilation_context.allocator);

    const input_attributes = try arena.allocator().alloc(AttributeList, input_info.shapes.len);
    @memset(input_attributes, .{});

    const output_info = b: {
        compilation_context.activate();
        defer compilation_context.deactivate();

        const result = @call(.auto, func, args);

        const output_info = try collectOutputInfo(compilation_context.allocator, &result);
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

    const mlir_func = dialects.func.func(compilation_context.mlir_ctx, .{
        .name = "main",
        .block = compilation_context.currentScope().block,
        .location = .unknown(compilation_context.mlir_ctx),
        .args_attributes = try finalizeAttributeList(arena.allocator(), compilation_context.mlir_ctx, input_attributes),
        .results_attributes = try finalizeAttributeList(arena.allocator(), compilation_context.mlir_ctx, output_attributes),
    });

    compilation_context.mlir_pass_manager.runOnOp(mlir_func) catch |err| switch (err) {
        error.MlirUnexpected => {
            std.log.err("Failed to canonicalize invalid mlir: {f}", .{mlir_func});
            // user errors should have triggered a panic before we reach this.
            @panic("ZML generated invalid mlir. Please open a bug report");
        },
    };

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

fn compileModuleToPjrtExecutable(arena: std.mem.Allocator, io: std.Io, platform: *const Platform, module: *const mlir.Module, xla_dump_to_: ?[]const u8) !*pjrt.LoadedExecutable {
    //const tracer = Tracer.init("ai.zml.compilation");
    //const compile_frame = tracer.frameStart("pjrt compilation");
    //defer tracer.frameEnd(compile_frame, "pjrt compilation");

    const sharding = platform.sharding();

    var upb_alloc: upb.Allocator = .init(arena);
    const upb_arena = c.upb_Arena_Init(null, 0, upb_alloc.inner());
    defer c.upb_Arena_Free(upb_arena);

    const options = blk: {
        const options = try upb.new(c.xla_CompileOptionsProto, upb_arena);
        c.xla_CompileOptionsProto_set_executable_build_options(options, executable_build_options_blk: {
            const exec_build_options = try upb.new(c.xla_ExecutableBuildOptionsProto, upb_arena);

            c.xla_ExecutableBuildOptionsProto_set_device_ordinal(exec_build_options, -1);
            c.xla_ExecutableBuildOptionsProto_set_num_replicas(exec_build_options, sharding.num_replicas);
            c.xla_ExecutableBuildOptionsProto_set_num_partitions(exec_build_options, sharding.num_partitions);
            c.xla_ExecutableBuildOptionsProto_set_use_spmd_partitioning(exec_build_options, sharding.num_partitions > 1 or sharding.num_replicas > 1);

            c.xla_ExecutableBuildOptionsProto_set_device_assignment(exec_build_options, device_assignment_blk: {
                const device_assignment = try upb.new(c.xla_DeviceAssignmentProto, upb_arena);

                c.xla_DeviceAssignmentProto_set_replica_count(device_assignment, sharding.num_replicas);
                c.xla_DeviceAssignmentProto_set_computation_count(device_assignment, sharding.num_partitions);

                const computation_devices = c.xla_DeviceAssignmentProto_resize_computation_devices(device_assignment, sharding.num_partitions, upb_arena);
                for (computation_devices[0..sharding.num_partitions], 0..) |*computation_device, i| {
                    computation_device.* = try upb.new(c.xla_DeviceAssignmentProto_ComputationDevice, upb_arena);
                    _ = c.xla_DeviceAssignmentProto_ComputationDevice_add_replica_device_ids(computation_device.*, @intCast(i), upb_arena);
                }
                break :device_assignment_blk device_assignment;
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
                    break :gemm_blk supported.has(platform.devices[0].pjrt_desc.attribute(platform.pjrt_api, "compute_capability").?.string);
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

        if (xla_dump_to_ orelse platform.compilation_options.xla_dump_to) |xla_dump_to| {
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
