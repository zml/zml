const std = @import("std");

const mlir = @import("mlir");
const shardy = @import("mlir/dialects/shardy");
const stablehlo = @import("mlir/dialects/stablehlo");
const stdx = @import("stdx");

const Compiler = @import("../Compiler.zig");
const meta = @import("../meta.zig");
const mlirx = @import("../mlirx.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const Tensor = @import("../tensor.zig").Tensor;

const log = std.log.scoped(.@"zml/Compiler");

/// Describes how the inputs and outputs of `body_fn` are partitioned across a mesh.
/// Returns the type of the `partitioning` argument to `manualComputation`.
///
/// * `in_specs` specifies the sharding of each tensor in the body's first argument.
///   It follows the argument's structure, omitting non-tensor metadata.
/// * `out_specs` specifies the sharding of each shape in the body's second argument.
///   It follows the output shapes' structure and can be omitted for a `void` argument.
///
/// Each spec is a `Sharding.Partitioning.Logical`. A mapping such as
/// `.parse(.{ .hidden = .model })` splits the tensor's `.hidden` dimension across
/// the mesh's `.model` axis. Omitted dimensions are replicated; `.replicated`
/// replicates the entire tensor. The specs must not contain open dimensions.
/// Slice lengths, optional presence and active union tags must match the arguments.
///
/// For a body taking `struct { x: Tensor, bias: Tensor, scale: f32 }` and a single
/// output `Shape`, the partitioning can be declared as:
///
/// ```zig
/// const partitioning: ManualPartitioning(body) = .{
///     .in_specs = .{ .x = .parse(.{ .hidden = .model }), .bias = .replicated },
///     .out_specs = .parse(.{ .hidden = .model }),
/// };
/// ```
pub fn ManualPartitioning(comptime body_fn: anytype) type {
    const Logical = Sharding.Partitioning.Logical;

    const Inputs = stdx.meta.FnParam(body_fn, 0);
    const OutputShapes = stdx.meta.FnParam(body_fn, 1);

    const InputSpecs = meta.MapRestrict(Tensor, Logical).map(Inputs);
    const OutputSpecs = meta.MapRestrict(Shape, Logical).map(OutputShapes);

    if (OutputSpecs == void) return struct {
        in_specs: InputSpecs,
        out_specs: void = {},
    };

    return struct {
        in_specs: InputSpecs,
        out_specs: OutputSpecs,
    };
}

/// Describes a computation on local shards of tensors distributed across `mesh`.
/// ZML operations normally describe work on global tensors. The partitioner
/// distributes the inputs according to `in_specs`, and `body_fn` describes the work
/// performed on each shard. The results are exposed to the surrounding graph as
/// global tensors with the sharding declared by `out_specs`.
///
/// * `body_fn` must be `fn (inputs: I, output_shapes: O) R`, where:
///   - `I` contains the local input tensors and any metadata needed by the body.
///   - `O` is `void`, `Shape`, or a container of shapes describing the local outputs.
///   - `R` is `void`, `Tensor`, or a container of tensors matching those local shapes
///     in traversal order.
/// * `inputs` contains the global tensors and metadata. The partitioner reshards
///   the tensors as needed to satisfy `in_specs`; metadata is passed to the body unchanged.
/// * `output_shapes` contains the global output shapes. The body receives their
///   local shard shapes, computed using `out_specs` and the sizes of the mesh axes.
/// * `mesh` is the logical mesh whose axis names are referenced by the specs.
/// * `partitioning` declares the input and output shardings; see `ManualPartitioning`.
///   Each partitioned dimension must be divisible by the size of its mesh axis.
///
/// For example, splitting a dimension of size 64 across a mesh axis of size 8 gives
/// the body a dimension of size 8. A replicated dimension keeps its global size.
/// The caller supplies the global output shapes because the body only describes
/// local results: their shapes alone do not determine the global tensor shapes.
pub fn manualComputation(
    comptime body_fn: anytype,
    inputs: stdx.meta.FnParam(body_fn, 0),
    output_shapes: stdx.meta.FnParam(body_fn, 1),
    mesh: *const Sharding.Mesh,
    partitioning: ManualPartitioning(body_fn),
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

    var owned_inputs: Inputs = undefined;
    meta.mapAlloc((struct {
        fn copy(_: void, tensor: Tensor) Tensor {
            return tensor;
        }
    }).copy, body_arena, {}, inputs, &owned_inputs) catch @panic("OOM");

    const flat_input_tensors: []*Tensor = meta.collectMutPtrs(Tensor, arena, &owned_inputs) catch @panic("OOM");
    const in_specs = meta.collectValues(Sharding.Partitioning.Logical, arena, &partitioning.in_specs) catch @panic("OOM");
    stdx.debug.assert(flat_input_tensors.len == in_specs.len, "manualComputation has {} input tensors but {} input specs", .{ flat_input_tensors.len, in_specs.len });

    for (flat_input_tensors, in_specs) |tensor, spec| {
        tensor._shape = tensor.shape().withPartitioning(mesh, spec);
        stdx.debug.assert(!tensor.shape().containsPartitionSpec(.open), "manualComputation input sharding must be closed: {f}", .{tensor.shape()});
    }

    var local_output_shapes: OutputShapes = undefined;
    meta.mapAlloc((struct {
        fn copy(_: void, shape: Shape) Shape {
            return shape;
        }
    }).copy, body_arena, {}, output_shapes, &local_output_shapes) catch @panic("OOM");

    const flat_local_output_shapes = meta.collectMutPtrs(Shape, arena, &local_output_shapes) catch @panic("OOM");
    const out_specs = meta.collectValues(Sharding.Partitioning.Logical, arena, &partitioning.out_specs) catch @panic("OOM");
    stdx.debug.assert(flat_local_output_shapes.len == out_specs.len, "manualComputation has {} output shapes but {} output specs", .{ flat_local_output_shapes.len, out_specs.len });

    const flat_output_shapes = arena.alloc(Shape, flat_local_output_shapes.len) catch @panic("OOM");
    for (flat_output_shapes, flat_local_output_shapes, out_specs) |*global, local, spec| {
        global.* = local.withPartitioning(mesh, spec);
        stdx.debug.assert(!global.containsPartitionSpec(.open), "manualComputation output sharding must be closed: {f}", .{global});
        local.* = Sharding.shardedShape(global.*) catch std.debug.panic("can't shard output {f} for {f}", .{ global, mesh });
    }

    const WrappedInputs = struct {
        inputs: *const Inputs,
        output_shapes: *const OutputShapes,

        fn wrap(allocator: std.mem.Allocator, opaque_wrapper: *const anyopaque, opaque_outputs: *anyopaque) error{OutOfMemory}![]*Tensor {
            const wrapped: *const @This() = @ptrCast(@alignCast(opaque_wrapper));
            const outputs_ptr: *Outputs = @ptrCast(@alignCast(opaque_outputs));

            outputs_ptr.* = body_fn(wrapped.inputs.*, wrapped.output_shapes.*);

            return try meta.collectMutPtrs(Tensor, allocator, outputs_ptr);
        }
    };

    var outputs_struct: Outputs = undefined;
    var wrapped_inputs: WrappedInputs = .{ .inputs = &owned_inputs, .output_shapes = &local_output_shapes };

    _ = manualComputationInternal(
        compiler,
        arena,
        mesh,
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

    // Returned tensors describe the operation's global output shapes and sharding.
    for (0.., local_outputs, global_output_shapes) |o, output_tensor_ptr, output_shape| {
        output_tensor_ptr.* = Tensor._result(output_shape, op.result(o));
        output_tensor_ptr._shape._sharding = output_shape._sharding;
    }
    return local_outputs;
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

test manualComputation {
    const zml = @import("../zml.zig");
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    const mesh = zml.testing.tpMesh();

    const LocalSum = struct {
        x: Tensor,
        bias: Tensor,
        scale: i32,

        fn body(self: @This(), output_shape: Shape) Tensor {
            return self.x.sum(.x).scale(self.scale).add(self.bias).broad(output_shape);
        }

        fn forward(x: Tensor, mesh_: *const Sharding.Mesh) Tensor {
            return manualComputation(
                body,
                .{ .x = x, .bias = .scalar(10, .i32), .scale = 2 },
                x.shape(),
                mesh_,
                .{ .in_specs = .{ .x = .parse(.{ .x = .model }), .bias = .replicated }, .out_specs = .parse(.{ .x = .model }) },
            );
        }
    };

    // Each shard sums two elements and broadcasts the result to its local output shape.
    const shape: Shape = .init(.{ .x = 2 * mesh.numPartitionsForLogicalAxis(.model) }, .i32);
    const values = try allocator.alloc(i32, shape.count());
    defer allocator.free(values);

    for (values, 0..) |*value, i| value.* = @intCast(i + 1);

    var input = try zml.Buffer.fromBytes(io, platform, shape, std.mem.sliceAsBytes(values));
    defer input.deinit();

    for ([_]Sharding.Partitioner{ .shardy, .gspmd }) |partitioner| {
        const exe = try platform.compileFn(allocator, io, LocalSum.forward, .{ .fromShape(shape), mesh }, .{
            .meshes = &.{ platform.replicated_mesh, mesh },
            .partitioner = partitioner,
        });
        defer exe.deinit();

        var output = try exe.eval(allocator, io, .{input});
        defer output.deinit();

        const actual = try output.toSliceAlloc(allocator, io);
        defer actual.free(allocator);

        for (actual.constItems(i32), 0..) |value, i| {
            const pair = values[i / 2 * 2 ..][0..2];
            try std.testing.expectEqual((pair[0] + pair[1]) * 2 + 10, value);
        }

        try std.testing.expect(output.shape().eqlWithTags(shape));
        try std.testing.expectEqualDeep(shape.withPartitioning(mesh, .{ .x = .model })._sharding, output.shape()._sharding);
    }
}

test "manualComputation composes across logical and physical meshes" {
    const zml = @import("../zml.zig");
    if (comptime !zml.Target.cpu.isEnabled()) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Topologies = enum { flat, torus };

    const Model = struct {
        fn increment(x: Tensor, _: Shape) Tensor {
            return x.addConstant(1);
        }

        fn forward(x: Tensor, mesh: *const Sharding.Mesh) Tensor {
            const partitioning: ManualPartitioning(increment) = .{
                .in_specs = .parse(.{ .x = .model }),
                .out_specs = .parse(.{ .x = .model }),
            };

            const first = manualComputation(increment, x, x.shape(), mesh, partitioning);
            const converted = first.convert(.f32);
            return manualComputation(increment, converted, converted.shape(), mesh, partitioning);
        }
    };

    const shape: Shape = .init(.{ .x = 16 }, .i32);

    var values: [16]i32 = undefined;
    var expected: [16]f32 = undefined;
    for (&values, &expected, 0..) |*value, *result, i| {
        value.* = @intCast(i);
        result.* = @floatFromInt(i + 2);
    }

    for ([_]Topologies{ .flat, .torus }) |topology| {
        const platform = try zml.Platform.init(allocator, io, .cpu, .{
            .cpu = .{ .device_count = 8 },
            .physical_mesh = if (topology == .torus) .{ .custom = Sharding.PhysicalMesh.torus3d } else .auto,
        });
        defer platform.deinit(allocator, io);

        // Fold three physical axes into one logical axis, or split one into three.
        const logical: Sharding.LogicalMesh = if (topology == .torus)
            try .init(.{.model}, &.{.balanced}, &.{8})
        else
            try Sharding.LogicalMesh.init(.{ .data, .model, .pipeline }, &.{ .low_bandwidth, .high_bandwidth, .balanced }, &.{ 2, 2, 2 });

        const mesh = try platform.registerMesh("manual", logical);

        var input: zml.Buffer = try .fromBytes(io, platform, shape, std.mem.asBytes(&values));
        defer input.deinit();

        for ([_]Sharding.Partitioner{ .shardy, .gspmd }) |partitioner| {
            const exe = try platform.compileFn(allocator, io, Model.forward, .{ Tensor.fromShape(shape), mesh }, .{
                .meshes = &.{ platform.replicated_mesh, mesh },
                .partitioner = partitioner,
            });
            defer exe.deinit();

            var output = try exe.eval(allocator, io, .{input});
            defer output.deinit();

            try std.testing.expectEqual(expected, try output.getValue([16]f32, io));
            try std.testing.expect(output.shape().eqlWithTags(shape.withDtype(.f32)));
            try std.testing.expectEqualDeep(shape.withPartitioning(mesh, .{ .x = .model })._sharding, output.shape()._sharding);
        }
    }
}
