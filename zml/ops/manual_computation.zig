const std = @import("std");

const mlir = @import("mlir");
const shardy = @import("mlir/dialects/shardy");
const stablehlo = @import("mlir/dialects/stablehlo");
const stdx = @import("stdx");

const Compiler = @import("../Compiler.zig");
const mlirx = @import("../mlirx.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const Tensor = @import("../tensor.zig").Tensor;

const zml = struct {
    // Nested to disambiguate from stdx.meta
    const meta = @import("../meta.zig");
    const testing = @import("../testing.zig");
};

const log = std.log.scoped(.@"zml/Compiler");

/// Run `body_fn` on local shards of the supplied global tensors.
/// `layout.mesh`, `layout.in_specs` and `layout.out_specs` are required.
/// Specs mirror the tensor/shape fields of their corresponding arguments;
/// each leaf is a logical partition mapping or `.replicated`. `.replicated`
/// can also explicitly replicate a whole subtree. Non-tensor metadata needs no spec.
/// Omitted dimensions in a mapping are replicated; open dimensions are rejected.
/// The body receives local input tensors and local output shapes. Returned tensors
/// retain the global sharding declared by `layout.out_specs`.
pub fn manualComputation(
    comptime body_fn: anytype,
    inputs: stdx.meta.FnParam(body_fn, 0),
    output_shapes: stdx.meta.FnParam(body_fn, 1),
    layout: anytype,
) stdx.meta.FnReturn(body_fn) {
    const Inputs = stdx.meta.FnParam(body_fn, 0);
    const OutputShapes = stdx.meta.FnParam(body_fn, 1);
    const Outputs = stdx.meta.FnReturn(body_fn);

    const compiler = Compiler.current();
    var arena_state: std.heap.ArenaAllocator = .init(compiler.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // The body may return containers borrowed from its inputs or output shapes.
    const body_arena = compiler.arena.allocator();
    var owned_inputs = applySpecs(Tensor, body_arena, layout.mesh, inputs, layout.in_specs) catch |err|
        std.debug.panic("manualComputation input layout: {s}", .{@errorName(err)});
    const global_output_shapes = applySpecs(Shape, body_arena, layout.mesh, output_shapes, layout.out_specs) catch |err|
        std.debug.panic("manualComputation output layout: {s}", .{@errorName(err)});
    var local_output_shapes: OutputShapes = undefined;
    zml.meta.mapAlloc((struct {
        fn local(_: void, shape: Shape) Shape {
            return Sharding.shardedShape(shape) catch unreachable;
        }
    }).local, body_arena, {}, global_output_shapes, &local_output_shapes) catch @panic("OOM");
    const flat_input_tensors: []*Tensor = zml.meta.collectMutPtrs(Tensor, arena, &owned_inputs) catch @panic("OOM");
    defer arena.free(flat_input_tensors);

    const flat_output_shapes: []const Shape = zml.meta.collectValues(Shape, arena, &global_output_shapes) catch @panic("OOM");

    const WrappedInputs = struct {
        inputs: *const Inputs,
        output_shapes: *const OutputShapes,

        fn wrap(allocator: std.mem.Allocator, opaque_wrapper: *const anyopaque, opaque_outputs: *anyopaque) error{OutOfMemory}![]*Tensor {
            const wrapped: *const @This() = @ptrCast(@alignCast(opaque_wrapper));
            const outputs_ptr: *Outputs = @ptrCast(@alignCast(opaque_outputs));

            outputs_ptr.* = body_fn(wrapped.inputs.*, wrapped.output_shapes.*);

            return try zml.meta.collectMutPtrs(Tensor, allocator, outputs_ptr);
        }
    };

    var outputs_struct: Outputs = undefined;
    var wrapped_inputs: WrappedInputs = .{ .inputs = &owned_inputs, .output_shapes = &local_output_shapes };
    _ = manualComputationInternal(
        compiler,
        arena,
        layout.mesh,
        @typeName(@TypeOf(body_fn)),
        &wrapped_inputs,
        flat_input_tensors,
        @ptrCast(&outputs_struct),
        flat_output_shapes,
        WrappedInputs.wrap,
    ) catch |err| switch (err) {
        error.OutOfMemory => @panic("OOM"),
    };
    return outputs_struct;
}

const LocalComputationFn = *const fn (std.mem.Allocator, *const anyopaque, *anyopaque) error{OutOfMemory}![]*Tensor;

/// Reified version of manualComputation.
/// This one is compiled exactly once, user code just need to compile the `collectPtrs`
noinline fn manualComputationInternal(
    compiler: *Compiler,
    arena: std.mem.Allocator,
    mesh: *const Sharding.Mesh,
    functor_name: []const u8,
    input_struct: *anyopaque,
    input_tensors: []*Tensor,
    output_struct: *anyopaque,
    global_output_shapes: []const Shape,
    local_fn: LocalComputationFn,
) error{OutOfMemory}![]*Tensor {
    const global_scope = compiler.currentScope();
    const mlir_ctx = compiler.mlir_ctx;

    log.debug("manualComputation({s}) will use mesh {s}", .{ functor_name, mesh.name });

    const global_input_shapes = try arena.alloc(Shape, input_tensors.len);
    for (global_input_shapes, input_tensors) |*shape, tensor| shape.* = tensor._shape;

    const input_types = try arena.alloc(*const mlir.Type, input_tensors.len);
    for (input_types, input_tensors) |*input_ty, tensor_ptr| {
        const global = tensor_ptr._shape;
        const local = Sharding.shardedShape(global) catch std.debug.panic("can't shard {f} for {f}", .{ global, mesh });
        log.debug("\t- {f} -> {f}", .{ global, local });
        input_ty.* = mlirx.Type.rankedTensor(mlir_ctx, local);
    }
    const input_locs = try arena.alloc(*const mlir.Location, input_tensors.len);
    @memset(input_locs, compiler.unknown_location);

    const input_values = try arena.alloc(*const mlir.Value, input_tensors.len);
    for (input_values, input_tensors) |*input_val, tensor_ptr| {
        if (compiler.partitioner == .gspmd) {
            tensor_ptr.* = tensor_ptr.withPartitioning(mesh, tensor_ptr.shape()._sharding.partition);
        }
        input_val.* = global_scope.value(tensor_ptr.*);
    }

    const manual_block = switch (compiler.partitioner) {
        .shardy => mlir.Block.init(input_types, input_locs),
        .gspmd => global_scope.block,
    };
    errdefer if (compiler.partitioner == .shardy) manual_block.deinit();

    const local_scope = compiler.pushBlock(manual_block);
    defer local_scope.pop();
    for (0.., input_tensors, global_input_shapes) |i, tensor_ptr, global| {
        const local = Sharding.shardedShape(global) catch unreachable; // we already computed this above
        tensor_ptr.* = switch (compiler.partitioner) {
            .shardy => Tensor._result(local, manual_block.argument(i)),
            .gspmd => gspmd.globalToLocal(mlir_ctx, global_scope, tensor_ptr.*, local, compiler.location),
        };
    }

    compiler.manual_computation_depth += 1;
    defer compiler.manual_computation_depth -= 1;

    const local_outputs = try local_fn(arena, input_struct, output_struct);
    stdx.debug.assert(local_outputs.len == global_output_shapes.len, "manualComputation body returned {} values, expected {}", .{ local_outputs.len, global_output_shapes.len });
    for (0.., local_outputs, global_output_shapes) |i, local_tensor, global_shape| {
        const expected_local = Sharding.shardedShape(global_shape) catch std.debug.panic("can't shard output {f} for {f}", .{ global_shape, mesh });
        stdx.debug.assert(expected_local.eql(local_tensor._shape), "manualComputation body output {d} expected {f}, got {f}", .{ i, expected_local, local_tensor._shape });
    }

    const global_output_types = try arena.alloc(*const mlir.Type, global_output_shapes.len);
    for (global_output_shapes, 0..) |output_shape, i| {
        global_output_types[i] = mlirx.Type.rankedTensor(mlir_ctx, output_shape);
    }

    const local_output_values = try arena.alloc(*const mlir.Value, local_outputs.len);
    for (local_output_values, local_outputs) |*output_val, tensor_ptr| {
        output_val.* = local_scope.value(tensor_ptr.*);
    }

    const op: *const mlir.Operation = switch (compiler.partitioner) {
        .shardy => op: {
            // Shardy creates a sdy.manual_computation block.
            _ = shardy.return_(mlir_ctx, local_output_values, compiler.location).appendTo(manual_block);

            const out_mesh_attrs = try arena.alloc(*const shardy.TensorShardingAttr, global_output_shapes.len);
            for (out_mesh_attrs, global_output_shapes) |*attr, shape| {
                attr.* = try mesh.sdyShardingAttrForShape(arena, mlir_ctx, shape);
            }

            break :op shardy.manual_computation(
                mlir_ctx,
                input_values,
                global_output_types,
                manual_block,
                try Sharding.sdyPerValueShardingAttr(arena, mlir_ctx, global_input_shapes, mesh),
                .get(mlir_ctx, out_mesh_attrs),
                try Sharding.sdyManualAxesAttr(arena, mlir_ctx, global_input_shapes, global_output_shapes, mesh),
                compiler.location,
            ).appendTo(global_scope.block);
        },
        .gspmd => op: {
            // Gspmd calls SPMDShardToFullShape and wrap into an optimizationBarrier
            // Shortcut to avoid empty optimization barrier
            if (global_output_shapes.len == 0) return &.{};

            const global_values = try arena.alloc(*const mlir.Value, global_output_shapes.len);
            const global_types = try arena.alloc(*const mlir.Type, global_output_shapes.len);
            for (global_output_shapes, 0..) |global_shape, i| {
                global_values[i], global_types[i] = try gspmd.localToGlobal(compiler, arena, global_scope, local_output_values[i], global_shape);
            }

            break :op stablehlo.optimizationBarrier(
                mlir_ctx,
                global_values,
                global_types,
                compiler.location,
            ).appendTo(global_scope.block);
        },
    };

    // Returned tensors describe the operation's global output shapes and layouts.
    for (0.., local_outputs, global_output_shapes) |o, output_tensor_ptr, output_shape| {
        output_tensor_ptr.* = Tensor._result(output_shape, op.result(o));
        output_tensor_ptr._shape._sharding = output_shape._sharding;
    }
    return local_outputs;
}

fn partitionShape(shape: Shape, mesh: *const Sharding.Mesh, spec: anytype) !Shape {
    const partitioned = if (@TypeOf(spec) == @EnumLiteral()) blk: {
        if (spec != .replicated) @compileError("manualComputation expects .replicated or a partition mapping");
        break :blk shape.withPartitioning(mesh, Sharding.Partitioning.replicated(shape.rank()));
    } else shape.withPartitioning(mesh, spec);
    if (partitioned.containsPartitionSpec(.open)) return error.OpenManualSharding;
    for (0..partitioned.rank()) |dim| {
        if (partitioned.partition(dim).meshAxis()) |axis| {
            if (axis >= mesh.logical.axes.len) return error.InvalidMeshAxis;
        }
    }
    _ = try Sharding.shardedShape(partitioned);
    return partitioned;
}

fn applySpecs(comptime Leaf: type, arena: std.mem.Allocator, mesh: *const Sharding.Mesh, value: anytype, specs: anytype) !@TypeOf(value) {
    const T = @TypeOf(value);
    if (T == Leaf) {
        if (Leaf == Shape) return partitionShape(value, mesh, specs);
        var tensor = value;
        tensor._shape = try partitionShape(tensor.shape(), mesh, specs);
        return tensor;
    }
    if (!comptime zml.meta.Contains(T, Leaf)) return value;
    const replicated = @TypeOf(specs) == @EnumLiteral();
    if (replicated and specs != .replicated) @compileError("Expected .replicated or a tree of partition specs");
    var result = value;
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            if (!replicated) inline for (comptime std.meta.fieldNames(@TypeOf(specs))) |name| {
                if (!@hasField(T, name)) @compileError("Unknown manualComputation spec field: " ++ name);
            };
            inline for (info.field_names, info.field_types) |name, Field| {
                if (comptime zml.meta.Contains(Field, Leaf)) {
                    if (!replicated and !@hasField(@TypeOf(specs), name)) @compileError("Missing manualComputation spec for " ++ name);
                    @field(result, name) = try applySpecs(Leaf, arena, mesh, @field(value, name), if (replicated) .replicated else @field(specs, name));
                }
            }
        },
        .@"union" => switch (value) {
            inline else => |field, tag| {
                if (comptime zml.meta.Contains(@TypeOf(field), Leaf)) {
                    result = @unionInit(T, @tagName(tag), try applySpecs(Leaf, arena, mesh, field, if (replicated) .replicated else @field(specs, @tagName(tag))));
                }
            },
        },
        .optional => if (value) |v| {
            result = try applySpecs(Leaf, arena, mesh, v, specs);
        },
        .array => {
            if (!replicated) std.debug.assert(value.len == specs.len);
            inline for (0..@typeInfo(T).array.len) |i| result[i] = try applySpecs(Leaf, arena, mesh, value[i], if (replicated) .replicated else specs[i]);
        },
        .pointer => |ptr| switch (ptr.size) {
            .one => {
                const copy = try arena.create(ptr.child);
                copy.* = try applySpecs(Leaf, arena, mesh, value.*, specs);
                result = copy;
            },
            .slice => {
                if (!replicated) std.debug.assert(value.len == specs.len);
                const copy = try arena.alloc(ptr.child, value.len);
                for (value, copy, 0..) |v, *dest, i| dest.* = try applySpecs(Leaf, arena, mesh, v, if (replicated) .replicated else specs[i]);
                result = copy;
            },
            else => @compileError("Unsupported manualComputation input: " ++ @typeName(T)),
        },
        else => @compileError("Unsupported manualComputation input: " ++ @typeName(T)),
    }
    return result;
}

const gspmd = struct {
    fn globalToLocal(mlir_ctx: *mlir.Context, scope: *Compiler.Scope, local_tensor: Tensor, local_shape: Shape, location: *const mlir.Location) Tensor {
        const local_type = mlirx.Type.rankedTensor(mlir_ctx, local_shape);
        const full_to_shard = stablehlo.custom_call(
            mlir_ctx,
            &.{scope.value(local_tensor)},
            &.{local_type},
            .{
                .call_target_name = "SPMDFullToShardShape",
                .has_side_effect = false,
                .backend_config = .{ .original = "" },
                .additional_attributes = &.{
                    .named(mlir_ctx, "mhlo.sharding", .string(mlir_ctx, "{manual}")),
                },
            },
            location,
        ).appendTo(scope.block);
        return Tensor._result(local_shape, full_to_shard.result(0));
    }

    fn localToGlobal(compiler: *Compiler, arena: std.mem.Allocator, scope: *Compiler.Scope, local_value: *const mlir.Value, global_shape: Shape) !struct { *const mlir.Value, *const mlir.Type } {
        const gspmd_attr = try compiler.tensorShardingAttr(arena, global_shape);
        const global_type = mlirx.Type.rankedTensor(compiler.mlir_ctx, global_shape);

        const shard_to_full = stablehlo.custom_call(
            compiler.mlir_ctx,
            &.{local_value},
            &.{global_type},
            .{
                .call_target_name = "SPMDShardToFullShape",
                .has_side_effect = false,
                .backend_config = .{ .original = "" },
                .additional_attributes = &.{
                    .named(compiler.mlir_ctx, "mhlo.sharding", gspmd_attr),
                },
            },
            compiler.location,
        ).appendTo(scope.block);
        return .{ shard_to_full.result(0), global_type };
    }
};

test "manualComputation passes local output shapes to the callback" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    const mesh = zml.testing.tpMesh();
    const partitions = mesh.numPartitionsForLogicalAxis(.model);
    if (partitions == 1) return error.SkipZigTest;

    var compiler: Compiler = .init(allocator, io, platform, .{ .meshes = &.{mesh} });
    defer compiler.deinit();
    compiler.activate();
    defer compiler.deactivate();

    const block = mlir.Block.init(&.{}, &.{});
    defer block.deinit();
    const scope = compiler.pushBlock(block);
    defer scope.pop();

    const global_shape = Shape.init(.{ .b = 4, .h = 3 * partitions, .hd = 128 }, .bf16)
        .withPartitioning(mesh, .{ .h = .model });
    const local_shape = Shape.init(.{ .b = 4, .h = 3, .hd = 128 }, .bf16);
    const input = Tensor.zeroes(global_shape).withPartitioning(mesh, .{ .h = .model });
    var received_shape: ?Shape = null;
    const Context = struct {
        input: Tensor,
        received_shape: *?Shape,

        fn body(self: @This(), output_shape: Shape) Tensor {
            self.received_shape.* = output_shape;
            return self.input;
        }
    };
    const output = manualComputation(
        Context.body,
        .{ .input = input, .received_shape = &received_shape },
        global_shape,
        .{ .mesh = mesh, .in_specs = .{ .input = .{ .h = .model } }, .out_specs = .{ .h = .model } },
    );

    try std.testing.expect(received_shape != null);
    try std.testing.expect(received_shape.?.eql(local_shape));
    try std.testing.expect(output.shape().eql(global_shape));
    try std.testing.expectEqualDeep(global_shape._sharding, output.shape()._sharding);
}

test "manual layouts reject open and incompatible dimensions" {
    const mesh = zml.testing.env().meshes.get("model").?;
    const shape = Shape.init(.{ .a = 2 * mesh.numPartitionsForLogicalAxis(.model) }, .f32);
    try std.testing.expectError(error.OpenManualSharding, partitionShape(shape, mesh, .{ .a = .open }));
    try std.testing.expectError(error.InvalidMeshAxis, partitionShape(shape, mesh, Sharding.Partitioning.init(&.{.mesh_axis_7})));
    if (mesh.numPartitionsForLogicalAxis(.model) > 1) {
        try std.testing.expectError(error.IncompatibleSharding, partitionShape(shape.setDim(.a, shape.dim(.a) + 1), mesh, .{ .a = .model }));
    }
    try std.testing.expect((try partitionShape(shape, mesh, .replicated)).isFullyReplicated());
}

test "manual specs copy borrowed tensor slices" {
    const mesh = zml.testing.env().meshes.get("model").?;
    const source = [_]Tensor{Tensor.init(.{ .a = 2 * mesh.numPartitionsForLogicalAxis(.model) }, .f32)};
    const specs = [_]Sharding.Partitioning{.init(&.{.mesh_axis_0})};
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const copy = try applySpecs(Tensor, arena.allocator(), mesh, @as([]const Tensor, &source), &specs);
    try std.testing.expect(copy[0].shape()._sharding.mesh == mesh);
    try std.testing.expect(source[0].shape()._sharding.mesh == null);
    try std.testing.expect(copy.ptr != &source);
}

fn checkManualPipeline(platform: *const @import("../platform.zig").Platform, mesh: *const Sharding.Mesh) !void {
    const api = @import("../zml.zig");
    const Functions = struct {
        fn local(x: Tensor, output: Shape) Tensor {
            std.debug.assert(output.dim(.a) == 2);
            std.debug.assert(x.shape().eql(output));
            return x.addConstant(1).reshape(output);
        }

        fn forward(x: Tensor, m: *const Sharding.Mesh) Tensor {
            const first = manualComputation(local, x.convert(.f32), x.shape().withDtype(.f32), .{
                .mesh = m,
                .in_specs = .{ .a = .model },
                .out_specs = .{ .a = .model },
            });
            std.debug.assert(first.shape()._sharding.mesh == m);
            std.debug.assert(first.shape().partition(.a) == x.shape().withPartitioning(m, .{ .a = .model }).partition(.a));
            std.debug.assert(!first.shape().containsPartitionSpec(.open));
            const converted = first.convert(.i32);
            std.debug.assert(converted.shape()._sharding.mesh == null);
            std.debug.assert(converted.shape().containsPartitionSpec(.open));
            return manualComputation(local, converted, x.shape(), .{
                .mesh = m,
                .in_specs = .{ .a = .model },
                .out_specs = .{ .a = .model },
            });
        }
    };
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const shape = Shape.init(.{ .a = 2 * mesh.numPartitionsForLogicalAxis(.model) }, .i32);
    const values = try allocator.alloc(i32, shape.count());
    defer allocator.free(values);
    for (values, 0..) |*v, i| v.* = @intCast(i);
    var input = try api.Buffer.fromBytes(io, platform, shape, std.mem.sliceAsBytes(values));
    defer input.deinit();
    for ([_]Sharding.Partitioner{ .shardy, .gspmd }) |partitioner| {
        const exe = try platform.compileFn(allocator, io, Functions.forward, .{ Tensor.fromShape(shape), mesh }, .{
            .meshes = &.{ platform.replicated_mesh, mesh },
            .partitioner = partitioner,
        });
        defer exe.deinit();
        var output = try exe.eval(allocator, io, .{input});
        defer output.deinit();
        const actual = try output.toSliceAlloc(allocator, io);
        defer actual.free(allocator);
        for (values, actual.constItems(i32)) |v, result| try std.testing.expectEqual(v + 2, result);
        try std.testing.expect(output.shape()._sharding.mesh == mesh);
        try std.testing.expectEqual(shape.withPartitioning(mesh, .{ .a = .model }).partition(.a), output.shape().partition(.a));
        try std.testing.expect(!output.shape().containsPartitionSpec(.open));
    }
}

test "manual computations preserve output sharding and compose after casts" {
    const platform = zml.testing.env();
    try checkManualPipeline(platform, platform.meshes.get("model").?);
}

test "manual logical meshes are independent of physical rank" {
    const api = @import("../zml.zig");
    if (comptime !api.platform.Target.cpu.isEnabled()) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    for ([_]bool{ false, true }) |torus| {
        const platform = try api.Platform.init(allocator, io, .cpu, .{
            .cpu = .{ .device_count = 8 },
            .physical_mesh = if (torus) .{ .custom = Sharding.PhysicalMesh.torus3d } else .auto,
        });
        defer platform.deinit(allocator, io);
        const logical = if (torus)
            try Sharding.LogicalMesh.init(.{.model}, &.{.balanced}, &.{8})
        else
            try Sharding.LogicalMesh.init(.{ .data, .model, .pipeline }, &.{ .balanced, .balanced, .balanced }, &.{ 2, 2, 2 });
        const mesh = try platform.registerMesh("manual", logical);
        try checkManualPipeline(platform, mesh);
    }
}
