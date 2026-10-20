const std = @import("std");
const builtin = @import("builtin");

const shardy = @import("mlir/dialects/shardy");
const stablehlo = @import("mlir/dialects/stablehlo");
const mlir = @import("mlir");
const stdx = @import("stdx");

const zml = struct {
    // Nested to disambiguate from stdx.meta
    const meta = @import("../meta.zig");
    const testing = @import("../testing.zig");
};

const Compiler = @import("../Compiler.zig");
const DataType = @import("../dtype.zig").DataType;
const mlirx = @import("../mlirx.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const Tensor = @import("../tensor.zig").Tensor;
const log = std.log.scoped(.@"zml/Compiler");

/// ZML operator describes computation across several devices,
/// but `manualComputation` allows to partition the devices in small group,
/// split the `Tensor` on those partitions,
/// and describes sub-computation that work inside a partition.
///
/// See:
/// https://github.com/openxla/shardy/blob/main/docs/sdy_dialect.md#sdymanual_computation-sdymanualcomputationop
///
/// * `body_fn` must be `fn (inputs: I, output_shapes: O) R` where:
///     - `I` is user defined and contains inputs tensors and any metadata required by the computation
///     - `O` contains the partionned output shapes, types must be `void`, `Shape` or `[]const Shape` for 0, 1 or many outputs
///     - `R` must be `void`, `Tensor, or `[]const Tensor` for 0, 1 or many outputs
/// * `inputs` must be using the same mesh (using .withPartitioning), or be fully replicated
/// * `outputs` contains the global output shapes,
/// and `manualComputation` will pass to `body_fn` the shapes of each local output shard.
///
/// This depart from the rest of ZML where output shapes are generally inferred,
/// but in this case since `body_fn` explains how to compute parts of the result,
/// it's not clear how to put them back together,
/// hence why the user is asked to explicitly provided the expected output shape.
pub fn manualComputation(
    comptime body_fn: anytype,
    inputs: stdx.meta.FnParam(body_fn, 0),
    output_shapes: stdx.meta.FnParam(body_fn, 1),
) stdx.meta.FnReturn(body_fn) {
    const Inputs = stdx.meta.FnParam(body_fn, 0);
    const OutputShapes = stdx.meta.FnParam(body_fn, 1);
    const Outputs = stdx.meta.FnReturn(body_fn);

    const compiler = Compiler.current();
    var arena_state: std.heap.ArenaAllocator = .init(compiler.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var owned_inputs: Inputs = inputs;
    const flat_input_tensors: []*Tensor = zml.meta.collectMutPtrs(Tensor, arena, &owned_inputs) catch @panic("OOM");
    defer arena.free(flat_input_tensors);

    var owned_output_shapes: OutputShapes = output_shapes;
    const flat_output_shapes: []*Shape = zml.meta.collectMutPtrs(Shape, arena, &owned_output_shapes) catch @panic("OOM");
    defer arena.free(flat_output_shapes);

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
    var wrapped_inputs: WrappedInputs = .{ .inputs = &owned_inputs, .output_shapes = &owned_output_shapes };
    _ = manualComputationInternal(
        compiler,
        arena,
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
    functor_name: []const u8,
    input_struct: *anyopaque,
    input_tensors: []*Tensor,
    output_struct: *anyopaque,
    output_shapes: []*Shape,
    local_fn: LocalComputationFn,
) error{OutOfMemory}![]*Tensor {
    const global_scope = compiler.currentScope();
    const mlir_ctx = compiler.mlir_ctx;

    const mesh = switch (chooseCommonMesh(input_tensors, compiler.platform.replicated_mesh)) {
        .mesh => |m| m,
        .incompatible => |incomp| std.debug.panic("manualComputation expects all inputs and outputs to use the same mesh, got input {}: {} and input {}: {}", .{ incomp[0], input_tensors[incomp[0]], incomp[1], input_tensors[incomp[1]] }),
    };
    const manual_axes = collectManualAxes(mesh, input_tensors);
    log.debug("manualComputation({s}) will use mesh {s}, manual axes: {f}", .{ functor_name, mesh.name, stdx.fmt.stringsZ(manual_axes.constSlice()) });

    const global_input_shapes = try arena.alloc(Shape, input_tensors.len);
    for (global_input_shapes, input_tensors) |*shape, tensor| shape.* = tensor._shape;
    const global_output_shapes = try arena.alloc(Shape, output_shapes.len);
    for (global_output_shapes, output_shapes) |*shape, shape_ptr| shape.* = shape_ptr.*;

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
        input_val.* = global_scope.value(tensor_ptr.*);
    }

    const manual_block = mlir.Block.init(input_types, input_locs);
    errdefer manual_block.deinit();

    const local_scope = compiler.pushBlock(manual_block);
    defer local_scope.pop();
    for (0.., input_tensors, global_input_shapes) |i, tensor_ptr, global| {
        const local = Sharding.shardedShape(global) catch unreachable; // we already computed this above
        tensor_ptr.* = switch (compiler.partitioner) {
            .shardy => Tensor._result(local, manual_block.argument(i)),
            .gspmd => gspmd.globalToLocal(mlir_ctx, global_scope, tensor_ptr.*, local, compiler.location),
        };
    }

    for (output_shapes, global_output_shapes) |shape_ptr, global| {
        shape_ptr.* = Sharding.shardedShape(global) catch std.debug.panic("can't shard output {f} for {f}", .{ global, mesh });
    }

    compiler.manual_computation_depth += 1;
    defer compiler.manual_computation_depth -= 1;

    const local_outputs = try local_fn(arena, input_struct, output_struct);
    stdx.debug.assert(local_outputs.len == global_output_shapes.len, "manualComputation body returned {} values, expected {}", .{ local_outputs.len, global_output_shapes.len });
    for (0.., local_outputs, output_shapes) |i, local_tensor, local_shape| {
        stdx.debug.assert(local_shape.eql(local_tensor._shape), "manualComputation body output {d} expected {f}, got {f}", .{ i, local_shape.*, local_tensor._shape });
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
            _ = mlir.Operation.make(mlir_ctx, "sdy.return", .{
                .operands = .{ .flat = local_output_values },
                .verify = false,
                .location = compiler.location,
            }).appendTo(manual_block);

            const out_mesh_attrs = try arena.alloc(*const shardy.TensorShardingAttribute, global_output_shapes.len);
            for (out_mesh_attrs, global_output_shapes) |*attr, shape| {
                attr.* = try mesh.sdyShardingAttrForShape(arena, mlir_ctx, shape);
            }

            break :op mlir.Operation.make(mlir_ctx, "sdy.manual_computation", .{
                .operands = .{ .flat = input_values },
                .results = .{ .flat = global_output_types },
                .blocks = &.{manual_block},
                .attributes = &.{
                    .named(mlir_ctx, "in_shardings", try Sharding.sdyPerValueShardingAttr(arena, mlir_ctx, global_input_shapes, mesh)),
                    .named(mlir_ctx, "out_shardings", shardy.attr.shardingsPerValue(mlir_ctx, out_mesh_attrs)),
                    .named(mlir_ctx, "manual_axes", try Sharding.sdyManualAxesAttr(arena, mlir_ctx, global_input_shapes, global_output_shapes, mesh)),
                },
                .verify = true,
                .location = compiler.location,
            }).appendTo(global_scope.block);
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

    // reuse local_outputs buffer to store the final result.
    for (0.., local_outputs, global_output_shapes) |o, output_tensor_ptr, output_shape| {
        output_tensor_ptr.* = Tensor._resultPropagateSharding(output_shape, op.result(o));
    }
    return local_outputs;
}

const ChosenMesh = union(enum) {
    mesh: *const Sharding.Mesh,
    incompatible: [2]u32,
};

fn chooseCommonMesh(inputs: []const *const Tensor, fallback: *const Sharding.Mesh) ChosenMesh {
    var mesh: ?*const Sharding.Mesh = null;
    var mesh_source_id: u32 = @intCast(inputs.len);
    for (0.., inputs) |i, input| {
        if (input.shape()._sharding.mesh) |shape_mesh| {
            if (mesh) |m| {
                if (m != shape_mesh)
                    return .{ .incompatible = .{ mesh_source_id, @intCast(i) } };
            } else {
                mesh = shape_mesh;
                mesh_source_id = @intCast(i);
            }
        }
    }
    return .{ .mesh = mesh orelse fallback };
}

test chooseCommonMesh {
    const fallback: Sharding.Mesh = .initTest(.fallback, .{.all});
    const first: Sharding.Mesh = .initTest(.first, .{ .data, .model });
    const second: Sharding.Mesh = .initTest(.second, .{ .experts, .model });
    const replicated = Tensor.init(.{ .a = 8 }, .f32);
    const sharded = Tensor.init(.{ .a = 8 }, .f32).withPartitioning(&first, .{ .a = .data });
    const other = Tensor.init(.{ .a = 8 }, .f32).withPartitioning(&second, .{ .a = .experts });

    try std.testing.expectEqual(ChosenMesh{ .mesh = &fallback }, chooseCommonMesh(&.{}, &fallback));
    try std.testing.expectEqual(ChosenMesh{ .mesh = &fallback }, chooseCommonMesh(&.{&replicated}, &fallback));
    try std.testing.expectEqual(ChosenMesh{ .mesh = &first }, chooseCommonMesh(&.{&sharded}, &fallback));
    try std.testing.expectEqual(ChosenMesh{ .mesh = &first }, chooseCommonMesh(&.{ &replicated, &sharded, &sharded }, &fallback));
    try std.testing.expectEqual(ChosenMesh{ .incompatible = .{ 0, 1 } }, chooseCommonMesh(&.{ &sharded, &other }, &fallback));
}

fn collectManualAxes(mesh: *const Sharding.Mesh, inputs: []const *const Tensor) stdx.BoundedArray(Shape.Tag, Sharding.MAX_MESH_RANK) {
    var used: [Sharding.MAX_MESH_RANK]bool = @splat(false);
    for (inputs) |input| {
        const sharding = input.shape()._sharding;
        if (sharding.mesh == null) continue;
        std.debug.assert(sharding.mesh.? == mesh);
        for (0..input.rank()) |dim| {
            if (sharding.partition.get(dim).meshAxis()) |axis| used[axis] = true;
        }
    }

    var axes: stdx.BoundedArray(Shape.Tag, Sharding.MAX_MESH_RANK) = .empty;
    for (mesh.logical.axes.constSlice(), 0..) |axis, i| {
        if (used[i]) axes.appendAssumeCapacity(axis);
    }
    return axes;
}

test collectManualAxes {
    const mesh: Sharding.Mesh = .initTest(.manual, .{ .data, .model, .unused });
    var data = Tensor.init(.{ .a = 8 }, .f32).withPartitioning(&mesh, .{ .a = .data });
    var model = Tensor.init(.{ .a = 8 }, .f32).withPartitioning(&mesh, .{ .a = .model });
    const replicated = Tensor.init(.{ .a = 8 }, .f32);
    const open = Tensor.init(.{ .a = 8 }, .f32).withPartitioning(&mesh, .{ .a = .open });

    const axes = collectManualAxes(&mesh, &.{ &model, &replicated, &data, &model, &open });
    data = replicated;
    model = replicated;
    try std.testing.expectEqual(@as(usize, 2), axes.len);
    try std.testing.expectEqualStrings("data", std.mem.span(axes.get(0)));
    try std.testing.expectEqualStrings("model", std.mem.span(axes.get(1)));
    try std.testing.expectEqual(@as(usize, 0), collectManualAxes(&mesh, &.{ &replicated, &open }).len);
    try std.testing.expectEqual(@as(usize, 0), collectManualAxes(&mesh, &.{}).len);

    const registry: *mlir.DialectRegistry = try .init();
    defer registry.deinit();
    registry.registerDialect("sdy");
    const ctx: *mlir.Context = try .init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    const expected = shardy.ManualAxesAttribute.init(ctx, &.{
        .init(ctx, "data"),
        .init(ctx, "model"),
    }).asAttr();
    try std.testing.expect(shardy_.manualAxesAttr(ctx, axes.constSlice()).eql(expected));
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

const shardy_ = struct {
    pub fn manualAxesAttr(mlir_ctx: *mlir.Context, axes: []const Shape.Tag) *const mlir.Attribute {
        var attrs: stdx.BoundedArray(*const mlir.StringAttribute, Sharding.MAX_MESH_RANK) = .empty;
        for (axes) |axis| attrs.appendAssumeCapacity(.init(mlir_ctx, std.mem.span(axis)));
        return shardy.ManualAxesAttribute.init(mlir_ctx, attrs.constSlice()).asAttr();
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
    );

    try std.testing.expect(received_shape != null);
    try std.testing.expect(received_shape.?.eql(local_shape));
    try std.testing.expect(output.shape().eql(global_shape));
    try std.testing.expectEqualDeep(global_shape._sharding, output.shape()._sharding);
}
