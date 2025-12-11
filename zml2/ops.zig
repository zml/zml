const std = @import("std");

const dialects = @import("mlir/dialects");
const mlir = @import("mlir");
const stdx = @import("stdx");

const CompilationContext = @import("module.zig").CompilationContext;
const constants = @import("constants.zig");
const meta = @import("meta.zig");
const mlirx = @import("mlirx.zig");
const Shape = @import("shape.zig").Shape;
const Tensor = @import("tensor.zig").Tensor;

pub const ReduceArgs = struct {
    left: Tensor,
    right: Tensor,
};

pub fn reduce(inputs: anytype, inits: anytype, axes_: []const i64, comptime func: anytype, context: anytype) stdx.meta.FnResult(func) {
    var arena = std.heap.ArenaAllocator.init(CompilationContext.current().allocator);
    defer arena.deinit();

    const mlir_ctx = CompilationContext.current().mlir_ctx;

    const reduce_block, var result = b: {
        const ArgsType = std.meta.Tuple(&[1]type{ReduceArgs} ** inits.len);
        var args: ArgsType = undefined;
        var block_types: [2 * inits.len]*const mlir.Type = undefined;

        inline for (0..inits.len) |i| {
            args[i].left = Tensor.init(inits[i].shape());
            args[i].right = Tensor.init(inits[i].shape());

            block_types[i] = mlir.rankedTensorType(args[i].left.dims(), mlirx.Type.fromDType(mlir_ctx, args[i].left.dtype()));
            block_types[i + inits.len] = mlir.rankedTensorType(args[i].right.dims(), mlirx.Type.fromDType(mlir_ctx, args[i].right.dtype()));
        }

        const block_locs: [2 * inits.len]*const mlir.Location = @splat(mlir.Location.unknown(mlir_ctx));
        const reduce_block = mlir.Block.init(&block_types, &block_locs);
        errdefer reduce_block.deinit();

        CompilationContext.current().pushBlock(reduce_block);
        defer CompilationContext.current().popBlock();

        const scope = CompilationContext.current().currentScope();
        inline for (0..inits.len) |i| {
            scope.id_to_argument.put(scope.arena.allocator(), args[i].left.id, i) catch unreachable;
            scope.id_to_argument.put(scope.arena.allocator(), args[i].right.id, i + inits.len) catch unreachable;
        }

        var result = @call(.auto, func, args ++ context);

        var result_values: [inits.len]*const mlir.Value = undefined;
        inline for (0..inits.len) |i| {
            result_values[i] = result[i].value();
        }

        _ = dialects.stablehlo.returns(mlir_ctx, &result_values, .unknown(mlir_ctx)).appendTo(reduce_block);
        break :b .{ reduce_block, result };
    };
    var input_values: [inputs.len]*const mlir.Value = undefined;
    inline for (0..inputs.len) |i| {
        input_values[i] = inputs[i].value();
    }

    var init_values: [inputs.len]*const mlir.Value = undefined;
    inline for (0..inits.len) |i| init_values[i] = inits[i].value();

    const reduce_op = mlir.Operation.make(mlir_ctx, "stablehlo.reduce", .{
        .operands = .{ .variadic = &.{ &input_values, &init_values } },
        .result_type_inference = true,
        .blocks = &.{reduce_block},
        .attributes = &.{
            .named(mlir_ctx, "dimensions", mlir.denseArrayAttribute(mlir_ctx, .i64, axes_)),
        },
        .verify = true,
        .location = .unknown(mlir_ctx),
    }).appendTo(CompilationContext.current().currentScope().block);

    // `stablehlo.reduce` drops axes. We want to avoid that to propagate tags.
    // So we need to broadcast the output of `stablehlo.reduce` to the input shapes.
    // To that order, we initialize `result` to `inputs`, then we use stdx.meta.visit,
    // to find the correct mlir.Value, but we first broadcast before creating the final
    // Tensor struct.
    var broadcasting_axes: stdx.BoundedArray(i64, constants.MAX_RANK) = .{};
    for (0..constants.MAX_RANK) |i| {
        if (std.mem.indexOfScalar(i64, axes_, @intCast(i)) == null) {
            broadcasting_axes.append(@intCast(i)) catch unreachable;
        }
    }

    inline for (0..result.len) |i| {
        var reduced_shape: Shape = inputs[i].shape();
        for (axes_) |a| {
            reduced_shape = reduced_shape.setDim(a, 1);
        }

        const tensor_type = mlir.rankedTensorType(reduced_shape.dims(), mlirx.Type.fromDType(mlir_ctx, reduced_shape.dtype()));
        const broad_op = dialects.stablehlo.broadcast_in_dim(
            mlir_ctx,
            reduce_op.result(i),
            broadcasting_axes.slice()[0 .. reduced_shape.rank() - axes_.len],
            tensor_type,
            .unknown(mlir_ctx),
        ).appendTo(CompilationContext.current().currentScope().block);

        result[i] = Tensor._result(reduced_shape, broad_op.result(0));
    }

    return result;
}

pub const ReduceWindowOpts = struct {
    // TODO replace with Shape
    window_dimensions: []const i64,
    window_strides: []const i64,
    base_dilations: []const i64,
    window_dilations: []const i64,
    padding: []const [2]i64,
};

pub fn reduceWindow(inputs: anytype, inits: anytype, opts: ReduceWindowOpts, comptime func: anytype, context: anytype) stdx.meta.FnResult(func) {
    var arena = std.heap.ArenaAllocator.init(CompilationContext.current().allocator);
    defer arena.deinit();

    const mlir_ctx = CompilationContext.current().mlir_ctx;

    const reduce_block, var result = b: {
        const ArgsType = std.meta.Tuple(&[1]type{ReduceArgs} ** inits.len);
        var args: ArgsType = undefined;
        var block_types: [2 * inits.len]*const mlir.Type = undefined;

        inline for (0..inits.len) |i| {
            args[i].left = Tensor.init(inits[i].shape());
            args[i].right = Tensor.init(inits[i].shape());

            block_types[i] = mlir.rankedTensorType(args[i].left.dims(), mlirx.Type.fromDType(mlir_ctx, args[i].left.dtype()));
            block_types[i + inits.len] = mlir.rankedTensorType(args[i].right.dims(), mlirx.Type.fromDType(mlir_ctx, args[i].right.dtype()));
        }

        const block_locs: [2 * inits.len]*const mlir.Location = @splat(mlir.Location.unknown(mlir_ctx));
        const reduce_block = mlir.Block.init(&block_types, &block_locs);
        errdefer reduce_block.deinit();

        CompilationContext.current().pushBlock(reduce_block);
        defer CompilationContext.current().popBlock();

        const scope = CompilationContext.current().currentScope();
        inline for (0..inits.len) |i| {
            scope.id_to_argument.put(scope.arena.allocator(), args[i].left.id, i) catch unreachable;
            scope.id_to_argument.put(scope.arena.allocator(), args[i].right.id, i + inits.len) catch unreachable;
        }

        var result = @call(.auto, func, args ++ context);

        var result_values: [inits.len]*const mlir.Value = undefined;
        inline for (0..inits.len) |i| {
            result_values[i] = result[i].value();
        }

        _ = dialects.stablehlo.returns(mlir_ctx, &result_values, .unknown(mlir_ctx)).appendTo(reduce_block);
        break :b .{ reduce_block, result };
    };
    var input_values: [inputs.len]*const mlir.Value = undefined;
    inline for (0..inputs.len) |i| {
        input_values[i] = inputs[i].value();
    }

    var init_values: [inputs.len]*const mlir.Value = undefined;
    inline for (0..inits.len) |i| init_values[i] = inits[i].value();

    const reduce_op = mlir.Operation.make(mlir_ctx, "stablehlo.reduce_window", .{
        .operands = .{ .variadic = &.{ &input_values, &init_values } },
        .result_type_inference = true,
        .blocks = &.{reduce_block},
        .attributes = &.{
            .named(mlir_ctx, "window_dimensions", mlir.denseArrayAttribute(mlir_ctx, .i64, opts.window_dimensions)),
            .named(mlir_ctx, "window_strides", mlir.denseArrayAttribute(mlir_ctx, .i64, opts.window_strides)),
            .named(mlir_ctx, "base_dilations", mlir.denseArrayAttribute(mlir_ctx, .i64, opts.base_dilations)),
            .named(mlir_ctx, "window_dilations", mlir.denseArrayAttribute(mlir_ctx, .i64, opts.window_dilations)),
            // Cast the [][2]i64 to []i64 (safe)
            .named(mlir_ctx, "padding", mlir.denseElementsAttribute(mlir.RankedTensorType.get(&.{ @intCast(opts.padding.len), 2 }, mlir.integerType(mlir_ctx, .i64), null).shaped(), @as([]const i64, @ptrCast(opts.padding)))),
        },
        .verify = true,
        .location = .unknown(mlir_ctx),
    }).appendTo(CompilationContext.current().currentScope().block);

    inline for (0..result.len) |i| {
        result[i] = Tensor.fromMlirValue(reduce_op.result(i)).withTags(inputs[i].shape());
    }

    return result;
}

pub const SortArgs = struct {
    left: Tensor,
    right: Tensor,
};

pub fn sort(inputs: anytype, axis_: i64, comptime func: anytype, context: anytype, is_stable: bool) [inputs.len]Tensor {
    var arena = std.heap.ArenaAllocator.init(CompilationContext.current().allocator);
    defer arena.deinit();

    const mlir_ctx = CompilationContext.current().mlir_ctx;

    const sort_block = b: {
        const ArgsType = std.meta.Tuple(&[1]type{SortArgs} ** inputs.len);
        var args: ArgsType = undefined;
        var block_types: [2 * inputs.len]*const mlir.Type = undefined;

        inline for (0..inputs.len) |i| {
            args[i].left = Tensor.init(Shape.init(.{}, inputs[i].shape().dtype()));
            args[i].right = Tensor.init(Shape.init(.{}, inputs[i].shape().dtype()));

            block_types[2 * i] = mlir.rankedTensorType(args[i].left.dims(), mlirx.Type.fromDType(mlir_ctx, args[i].left.dtype()));
            block_types[2 * i + 1] = mlir.rankedTensorType(args[i].right.dims(), mlirx.Type.fromDType(mlir_ctx, args[i].right.dtype()));
        }

        const block_locs: [2 * inputs.len]*const mlir.Location = @splat(mlir.Location.unknown(mlir_ctx));
        const sort_block = mlir.Block.init(&block_types, &block_locs);
        errdefer sort_block.deinit();

        CompilationContext.current().pushBlock(sort_block);
        defer CompilationContext.current().popBlock();

        const scope = CompilationContext.current().currentScope();
        inline for (0..inputs.len) |i| {
            scope.id_to_argument.put(scope.arena.allocator(), args[i].left.id, 2 * i) catch unreachable;
            scope.id_to_argument.put(scope.arena.allocator(), args[i].right.id, 2 * i + 1) catch unreachable;
        }

        var result = @call(.auto, func, args ++ context);

        _ = dialects.stablehlo.return_(mlir_ctx, result.value(), .unknown(mlir_ctx)).appendTo(sort_block);
        break :b sort_block;
    };

    var input_values: [inputs.len]*const mlir.Value = undefined;
    inline for (0..inputs.len) |i| {
        input_values[i] = inputs[i].value();
    }

    const sort_op = mlir.Operation.make(mlir_ctx, "stablehlo.sort", .{
        .operands = .{ .variadic = &.{&input_values} },
        .result_type_inference = true,
        .blocks = &.{sort_block},
        .attributes = &.{
            .named(mlir_ctx, "dimension", mlir.integerAttribute(mlir_ctx, .i64, axis_)),
            .named(mlir_ctx, "is_stable", mlir.boolAttribute(mlir_ctx, is_stable)),
        },
        .verify = true,
        .location = .unknown(mlir_ctx),
    }).appendTo(CompilationContext.current().currentScope().block);

    var result: [inputs.len]Tensor = undefined;
    inline for (0..inputs.len) |i| {
        result[i] = Tensor._result(inputs[i].shape(), sort_op.result(i));
    }

    return result;
}

pub const TritonOps = struct {
    debug: bool = false,
    name: [:0]const u8,
    ir: [:0]const u8,
    grid: [3]i32,
    num_stages: i32,
    num_warps: i32,
    output_operand_aliases: []const i64 = &.{},
};

/// Generate an MLIR call to the given member function with the given tensors.
pub fn triton(inputs: anytype, outputs: anytype, opts: TritonOps) [outputs.len]Tensor {
    const mlir_ctx = CompilationContext.current().mlir_ctx;
    var arena = std.heap.ArenaAllocator.init(CompilationContext.current().allocator);
    defer arena.deinit();

    var values: [inputs.len]*const mlir.Value = undefined;
    inline for (0..inputs.len) |i| {
        values[i] = inputs[i].value();
    }

    var res_types: [outputs.len]*const mlir.Type = undefined;
    inline for (outputs, 0..) |output, i| {
        res_types[i] = mlir.rankedTensorType(output.dims(), mlirx.Type.fromDType(mlir_ctx, output.dtype()));
    }

    const backend_config = mlir.dictionaryAttribute(mlir_ctx, &.{
        mlir.NamedAttribute.named(mlir_ctx, "name", mlir.stringAttribute(mlir_ctx, opts.name)),
        mlir.NamedAttribute.named(mlir_ctx, "ir", mlir.stringAttribute(mlir_ctx, opts.ir)),
        mlir.NamedAttribute.named(mlir_ctx, "grid_x", mlir.integerAttribute(mlir_ctx, .i32, opts.grid[0])),
        mlir.NamedAttribute.named(mlir_ctx, "grid_y", mlir.integerAttribute(mlir_ctx, .i32, opts.grid[1])),
        mlir.NamedAttribute.named(mlir_ctx, "grid_z", mlir.integerAttribute(mlir_ctx, .i32, opts.grid[2])),
        mlir.NamedAttribute.named(mlir_ctx, "num_stages", mlir.integerAttribute(mlir_ctx, .i32, opts.num_stages)),
        mlir.NamedAttribute.named(mlir_ctx, "num_warps", mlir.integerAttribute(mlir_ctx, .i32, opts.num_warps)),
    });

    var operands_layouts: [inputs.len][]const usize = undefined;
    inline for (inputs, 0..) |input, i| {
        operands_layouts[i] = arena.allocator().dupe(usize, toUsize(constants.minorToMajor(input.rank())).constSlice()) catch unreachable;
    }

    var results_layouts: [outputs.len][]const usize = undefined;
    inline for (outputs, 0..) |output, i| {
        results_layouts[i] = arena.allocator().dupe(usize, toUsize(constants.minorToMajor(output.rank())).constSlice()) catch unreachable;
    }

    const op = dialects.stablehlo.custom_call(
        mlir_ctx,
        &values,
        &res_types,
        .{
            .call_target_name = "__gpu$xla.gpu.triton",
            .backend_config = .{ .typed_ffi = backend_config },
            .has_side_effect = false,
            .operand_layouts = &operands_layouts,
            .result_layouts = &results_layouts,
            .output_operand_aliases = opts.output_operand_aliases,
        },
        .unknown(mlir_ctx),
    ).appendTo(CompilationContext.current().currentScope().block);

    var outputs_: [outputs.len]Tensor = undefined;
    inline for (outputs, 0..) |output, i| {
        outputs_[i] = Tensor._result(output, op.result(i));
    }

    return outputs_;
}

test "triton" {
    const zml = @import("zml.zig");
    const platform = zml.testing.env();

    if (platform.target != .cuda and platform.target != .rocm) return error.SkipZigTest;

    const ir =
        \\ module {
        \\   tt.func public @add_one(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg3: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}) {
        \\     %0 = tt.get_program_id x : i32
        \\     %1 = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
        \\     %2 = tt.load %arg1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
        \\     %cst = arith.constant 1.000000e+00 : f32
        \\     %3 = arith.addf %1, %cst : f32
        \\     tt.store %arg2, %3 {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<f32>
        \\     tt.store %arg3, %2 {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<f32>
        \\     tt.return
        \\   }
        \\ }
    ;

    const TritonMod = struct {
        pub fn forward(a: Tensor, b: Tensor) [2]Tensor {
            return triton(.{ a, b }, .{ a.shape(), b.shape() }, .{
                .debug = false,
                .name = "add_one",
                .ir = ir,
                .grid = .{ 1, 1, 1 },
                .num_stages = 1,
                .num_warps = 1,
            });
        }
    };

    const a: zml.Tensor = .init(Shape.init(.{}, .f32));
    const b: zml.Tensor = .init(Shape.init(.{}, .f32));

    var exe = try zml.module.compile(std.testing.allocator, std.testing.io, TritonMod.forward, .{ a, b }, platform);
    defer exe.deinit();

    var a_buffer: zml.Buffer = try .fromBytes(platform, a.shape(), std.mem.sliceAsBytes(&[1]f32{1}), std.testing.io);
    defer a_buffer.deinit();
    var b_buffer: zml.Buffer = try .fromBytes(platform, b.shape(), std.mem.sliceAsBytes(&[1]f32{3}), std.testing.io);
    defer b_buffer.deinit();

    const results = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, TritonMod.forward, .{ a_buffer, b_buffer });
    defer results[0].deinit();
    defer results[1].deinit();

    var cpu_result_0 = try results[0].toSliceAlloc(std.testing.allocator, std.testing.io);
    defer cpu_result_0.free(std.testing.allocator);
    var cpu_result_1 = try results[1].toSliceAlloc(std.testing.allocator, std.testing.io);
    defer cpu_result_1.free(std.testing.allocator);

    const expected_result_a: f32 = 2.0;
    const expected_result_b: f32 = 3.0;

    try std.testing.expectEqual(expected_result_a, cpu_result_0.items(f32)[0]);
    try std.testing.expectEqual(expected_result_b, cpu_result_1.items(f32)[0]);
}

pub const ScatterArgs = struct {
    input: Tensor,
    update: Tensor,
};

/// Generalized version of scatter to many inputs.
/// See `zml.Tensor.scatterSlices` for documentation on scatter.
///
/// This allows to use the same indices to update several tensors at once,
/// and where the update function is allow to look at elements from the different tensors
/// to compute the final value.
///
/// This sounds nice but in practice XLA doesn't support this well on GPU,
/// and will generate slow code. In practice stick with `zml.Tensor.scatterSlices`.
pub fn scatter(
    inputs: anytype,
    index_tensors: anytype,
    updates: anytype,
    comptime func: anytype,
    context: anytype,
    opts: Tensor.ScatterOpts,
) stdx.meta.FnResult(func) {
    var arena = std.heap.ArenaAllocator.init(CompilationContext.current().allocator);
    defer arena.deinit();

    const mlir_ctx = CompilationContext.current().mlir_ctx;

    const update_block, var result = b: {
        const ArgsType = std.meta.Tuple(&[1]type{ScatterArgs} ** inputs.len);
        var args: ArgsType = undefined;
        var block_types: [2 * inputs.len]*const mlir.Type = undefined;

        inline for (0..inputs.len) |i| {
            args[i].input = Tensor.init(Shape.init(.{}, inputs[i].dtype()));
            args[i].update = Tensor.init(Shape.init(.{}, inputs[i].dtype()));

            block_types[i] = mlir.rankedTensorType(args[i].input.dims(), mlirx.Type.fromDType(mlir_ctx, args[i].input.dtype()));
            block_types[i + inputs.len] = mlir.rankedTensorType(args[i].update.dims(), mlirx.Type.fromDType(mlir_ctx, args[i].update.dtype()));
        }

        const block_locs: [2 * inputs.len]*const mlir.Location = @splat(mlir.Location.unknown(mlir_ctx));
        const update_block = mlir.Block.init(&block_types, &block_locs);
        errdefer update_block.deinit();

        CompilationContext.current().pushBlock(update_block);
        defer CompilationContext.current().popBlock();

        const scope = CompilationContext.current().currentScope();
        inline for (0..inputs.len) |i| {
            scope.id_to_argument.put(scope.arena.allocator(), args[i].input.id, i) catch unreachable;
            scope.id_to_argument.put(scope.arena.allocator(), args[i].update.id, i + inputs.len) catch unreachable;
        }

        var result = @call(.auto, func, args ++ context);

        var result_values: [inputs.len]*const mlir.Value = undefined;
        inline for (0..inputs.len) |i| {
            result_values[i] = result[i].value();
        }

        _ = dialects.stablehlo.returns(mlir_ctx, &result_values, .unknown(mlir_ctx)).appendTo(update_block);
        break :b .{ update_block, result };
    };

    // Note: I was a bit lazy here, and I only look at tags on the first tensor.
    // we probably should check all of them.
    const self = meta.first(Tensor, inputs);
    const update = meta.first(Tensor, updates);
    var indices_per_axis, var indices_axes = Shape.parseStruct(Tensor, index_tensors);

    if (indices_per_axis.len == 0) return inputs;

    // validate coord axes: all coord_axes should exist inside self
    for (indices_axes.constSlice()) |t| {
        stdx.debug.assert(self._shape.hasTag(t) != null, "zml.ops.scatter expects axes of indices to be axes of inputs, got input={f} and indices={any}", .{ self, indices_axes.constSlice() });
    }

    // Handle scalar indices by broadcasting them to the indices with the highest rank.
    const indices_shape = blk: {
        var higher_rank = indices_per_axis.get(0).shape();
        for (indices_per_axis.constSlice()[1..]) |indices| {
            if (indices.rank() > higher_rank.rank()) {
                higher_rank = indices.shape();
            }
        }
        break :blk higher_rank;
    };
    for (indices_per_axis.slice()) |*idx| {
        stdx.debug.assert(idx.shape().canBroadcastTo(indices_shape), "zml.ops.scatter expects all indices tensor to have the same shape, got {any}", .{indices_per_axis.slice()});
        stdx.debug.assert(idx.dtype() == indices_shape.dtype(), "zml.ops.scatter expects all indices tensor to have the same dtype, got {any}", .{indices_per_axis.slice()});
        idx.* = idx.broad(indices_shape);
    }

    // rewrite simple scatters to dynamicUpdateSlice.
    if (@TypeOf(inputs) == struct { Tensor } and indices_shape.rank() == 0) {
        return .{self.dynamicUpdateSlice(index_tensors, update)};
    }

    // TODO: ideally we should catch all possible scatter errors and provide nice error messages.
    var config = scatterConfig(self.shape(), update.shape(), indices_per_axis, indices_axes);
    const indices = scatterPrepareIndices(&config, self.shape(), update.shape(), &indices_per_axis, &indices_axes);
    // const n_indices_axes = update.rank() - _collectAxes(AxisKind, up_kind, .update_window).len;
    // stdx.debug.assert(n_indices_axe == indices_axes.len, "scatter({f}, {any}) expects 'updates' to contain all axes from 'indices', got indices={s}, updates={f}", .{ self, index_tensors, indices_axes.constSlice(), update });

    var input_values: [inputs.len]*const mlir.Value = undefined;
    inline for (0..inputs.len) |i| {
        input_values[i] = inputs[i].value();
    }

    var updates_values: [inputs.len]*const mlir.Value = undefined;
    inline for (0..updates.len) |i| updates_values[i] = updates[i].value();

    const op = dialects.stablehlo.scatter(
        mlir_ctx,
        &input_values,
        &.{indices.value()},
        &updates_values,
        update_block,
        .{
            .update_window_dims = _collectAxes(ScatterAxisKind, config.up_kind, .update_window).constSlice(),
            .inserted_window_dims = _collectAxes(ScatterAxisKind, config.op_kind, .inserted_window).constSlice(),
            .input_batching_dims = _collectAxes(ScatterAxisKind, config.op_kind, .batching).constSlice(),
            .scatter_indices_batching_dims = config.indices_batch_axes.constSlice(),
            .scatter_dims_to_operand_dims = config.scatter_to_operand_axes.constSlice(),
            .index_vector_dim = indices.rank() - 1,
            .indices_are_sorted = opts.indices_are_sorted,
            .unique_indices = opts.indices_are_unique,
        },
        .unknown(mlir_ctx),
    ).appendTo(CompilationContext.current().currentScope().block);

    inline for (0..result.len) |i| {
        result[i] = Tensor._result(inputs[i].shape(), op.result(i));
    }

    return result;
}

const ScatterConfig = struct {
    op_kind: stdx.BoundedArray(ScatterAxisKind, constants.MAX_RANK) = .{},
    up_kind: stdx.BoundedArray(ScatterAxisKind, constants.MAX_RANK) = .{},
    indices_batch_axes: Shape.DimsArray = .{},
    scatter_to_operand_axes: Shape.DimsArray = .{},
    updates_transpose: Shape.AxesArray = .{},
};

const ScatterAxisKind = enum { batching, update_window, inserted_window, window_id };

fn scatterConfig(
    op: Shape,
    update: Shape,
    indices_per_axis: stdx.BoundedArray(Tensor, constants.MAX_RANK),
    indices_axes: Shape.TagsArray,
) ScatterConfig {
    var op_kind: stdx.BoundedArray(ScatterAxisKind, constants.MAX_RANK) = .{};
    var up_kind: stdx.BoundedArray(ScatterAxisKind, constants.MAX_RANK) = .{};
    var indices_batch_axes: Shape.DimsArray = .{};
    var scatter_to_operand_axes: Shape.DimsArray = .{};
    var updates_transpose: Shape.AxesArray = .{};

    const tagged_api = indices_axes.len > 0;
    const indices = indices_per_axis.get(0).shape();

    if (tagged_api) {
        for (indices_axes.constSlice()) |t| {
            scatter_to_operand_axes.appendAssumeCapacity(op.axis(t));
        }
        for (indices.tags()) |t| {
            stdx.debug.assert(update.hasTag(t) != null, "scatter expects 'updates' to have all axes of 'indices', got self={f}, updates={f} and indices={f}", .{ op, update, indices });
            updates_transpose.appendAssumeCapacity(update.axis(t));
        }

        for (op.tags()) |t| {
            if (update.hasTag(t)) |up_ax| {
                updates_transpose.appendAssumeCapacity(up_ax);

                if (indices.hasTag(t)) |id_ax| {
                    if (std.mem.indexOfScalar(Shape.Tag, indices_axes.constSlice(), t) != null) {
                        // tag is in indices AND in coords -> it's a batching dim that has been rewritten to a regular insertion dim
                        op_kind.appendAssumeCapacity(.inserted_window);
                    } else {
                        // tag is in op, indices and updates -> it's a batching dim
                        op_kind.appendAssumeCapacity(.batching);
                        indices_batch_axes.appendAssumeCapacity(@intCast(id_ax));
                    }
                } else {
                    op_kind.appendAssumeCapacity(.update_window);
                }
            } else {
                op_kind.appendAssumeCapacity(.inserted_window);
            }
        }

        for (update.tags(), 0..) |t, up_ax| {
            // Handle batch axes right away.
            if (op.hasTag(t)) |self_ax| {
                if (op_kind.get(self_ax) == .batching) {
                    up_kind.appendAssumeCapacity(.batching);
                    continue;
                }
            }
            if (indices.hasTag(t) != null) {
                up_kind.appendAssumeCapacity(.window_id);
            } else if (op.hasTag(t)) |self_ax| {
                stdx.debug.assert(update.dim(up_ax) <= op.dim(self_ax), "scatter expects the slices described in 'updates' to fit inside 'op', but along axis .{s} it doesn't. Got op={f}, updates={f}.", .{ t, op, update });
                up_kind.appendAssumeCapacity(.update_window);
            } else {
                // TODO: consider accepting untagged update here.
                std.debug.panic("scatter expects 'updates' to be made of axes from op={f} and from indices={any}, got unknown tag {s} in {f}", .{ op, indices_axes.constSlice(), std.mem.sliceTo(t, 0), update });
            }
        }
    } else {
        for (0..indices_per_axis.len) |i| {
            op_kind.appendAssumeCapacity(.inserted_window);
            scatter_to_operand_axes.appendAssumeCapacity(@intCast(i));
            up_kind.appendAssumeCapacity(.window_id);
        }
        for (indices_per_axis.len..op.rank()) |_| {
            op_kind.appendAssumeCapacity(.update_window);
        }
        for (indices_per_axis.len..update.rank()) |_| {
            up_kind.appendAssumeCapacity(.update_window);
        }
        for (0..update.rank()) |i| {
            updates_transpose.appendAssumeCapacity(@intCast(i));
        }
    }

    return .{
        .op_kind = op_kind,
        .up_kind = up_kind,
        .indices_batch_axes = indices_batch_axes,
        .scatter_to_operand_axes = scatter_to_operand_axes,
        .updates_transpose = updates_transpose,
    };
}

test scatterConfig {
    const zml = @import("zml.zig");
    const platform = zml.testing.env();

    var comp = zml.module.CompilationContext.init(std.testing.allocator, platform);
    defer comp.deinit();
    comp.activate();
    defer comp.deactivate();

    const block = mlir.Block.init(&.{}, &.{});
    comp.pushBlock(block);
    defer comp.popBlock();

    const Local = struct {
        pub fn _idx(idx_shape: anytype) Tensor {
            return Tensor.constant(.{ .i32 = 0 }).broad(Shape.init(idx_shape, .i32));
        }
    };

    const idx = Local._idx;
    const op = Shape.init(.{ .a = 10, .b = 20 }, .f32);

    // Use .a as a batching axis with .a=10 x .n=8 updates of 2 elements of .b
    {
        const indices, const coords_tags = Shape.parseStruct(Tensor, .{ .b = idx(.{ .a = 10, .n = 8 }) });
        const update = Shape.init(.{ .a = 10, .n = 8, .b = 2 }, .f32);

        const cfg = scatterConfig(op, update, indices, coords_tags);
        try std.testing.expectEqualSlices(ScatterAxisKind, &.{ .batching, .update_window }, cfg.op_kind.constSlice());
        try std.testing.expectEqualSlices(ScatterAxisKind, &.{ .batching, .window_id, .update_window }, cfg.up_kind.constSlice());
    }

    // similar, but use the normalized form where .a is no longer an explicit batching axis.
    {
        const indices, const coords_tags = Shape.parseStruct(Tensor, .{ .a = idx(.{ .a = 10, .n = 8 }), .b = idx(.{ .a = 10, .n = 8 }) });
        const update = Shape.init(.{ .a = 10, .n = 8, .b = 2 }, .f32);

        const cfg = scatterConfig(op, update, indices, coords_tags);
        try std.testing.expectEqualSlices(ScatterAxisKind, &.{ .inserted_window, .update_window }, cfg.op_kind.constSlice());
        try std.testing.expectEqualSlices(ScatterAxisKind, &.{ .window_id, .window_id, .update_window }, cfg.up_kind.constSlice());
    }
}

/// Concatenate all indices tensor in one tensor.
///
/// Is allowed to reorder stuff to simplify the job of the backend,
/// and to expand the batching dims.
fn scatterPrepareIndices(
    cfg: *ScatterConfig,
    op: Shape,
    update: Shape,
    indices_per_axis: *stdx.BoundedArray(Tensor, constants.MAX_RANK),
    indices_axes: *Shape.TagsArray,
) Tensor {
    var old_scatter_to_op_axes = cfg.scatter_to_operand_axes;
    const batching = _collectAxes(ScatterAxisKind, cfg.op_kind, .batching);
    for (batching.constSlice()) |batch_ax| {
        const id_shape = indices_per_axis.get(0).shape();
        // batching requires tagging, so we're sure to have a tag here.
        const batch_tag = op.tag(batch_ax);
        indices_axes.appendAssumeCapacity(batch_tag);
        const batch_id = Tensor.iota(id_shape, batch_tag).convert(id_shape.dtype());
        indices_per_axis.appendAssumeCapacity(batch_id);
        cfg.op_kind.buffer[@intCast(batch_ax)] = .inserted_window;
        cfg.up_kind.buffer[update.axis(batch_tag)] = .window_id;
        old_scatter_to_op_axes.appendAssumeCapacity(batch_ax);
    }
    cfg.indices_batch_axes = .{};

    // Reorder the axes so that in indices_per_axis is ordered like in op if possible.
    // TODO: transpose updates if needed
    var indices: stdx.BoundedArray(Tensor, constants.MAX_RANK) = .{};
    var scatter_to_op_axes: Shape.DimsArray = .{};

    while (old_scatter_to_op_axes.len > 0) {
        const scatter_ax = std.sort.argMin(i64, old_scatter_to_op_axes.constSlice(), {}, std.sort.asc(i64)).?;
        const op_ax = old_scatter_to_op_axes.orderedRemove(scatter_ax);
        const scatter_idx = indices_per_axis.orderedRemove(scatter_ax);

        scatter_to_op_axes.appendAssumeCapacity(op_ax);
        indices.appendAssumeCapacity(scatter_idx);
    }
    cfg.scatter_to_operand_axes = scatter_to_op_axes;

    for (scatter_to_op_axes.constSlice(), 0..) |sc_ax, i| {
        if (i != sc_ax) {
            //log.warn("Found a slow scatter pattern, which is going to generate a while loop: scatter({f}, {any}, {f}). Because the index axes aren't the major ones in the input tensor.", .{ op, scatter_to_op_axes.constSlice(), update });
            break;
        }
    }
    return Tensor.stack(indices.constSlice(), .last, .coord);
}

pub const GatherAxisKind = enum { batching, offset, collapsed, indices };

pub const GatherOpts = struct { indices_are_sorted: bool = false };

pub fn gather(self: Tensor, idx_axes: []const u3, idx_per_axis: []const Tensor, opts: GatherOpts) Tensor {
    const mlir_ctx = CompilationContext.current().mlir_ctx;

    stdx.debug.assert(idx_axes.len > 0, "gather expects 1 or more axes to operate one, received none. Example: `x.gather(.a, indices, .{{}})`", .{});
    for (idx_axes, 0..) |a, i| {
        if (i > 0) {
            stdx.debug.assert(a == idx_axes[i - 1] + 1, "gather expects 'idx_axes' to be sequential. But {any} aren't sequential in {f}", .{ idx_axes, self });
        }
    }
    var indices_shape = idx_per_axis[0].shape();
    for (idx_per_axis[1..]) |idx| {
        if (idx.rank() > indices_shape.rank()) {
            indices_shape = idx.shape();
        }
    }
    for (idx_per_axis) |idx| {
        stdx.debug.assert(idx.shape().canBroadcastTo(indices_shape), "gather indices can't be broadcasted together {any}", .{idx_per_axis});
    }

    var idx_batch_axes: Shape.DimsArray = .{};

    var self_kind: stdx.BoundedArray(GatherAxisKind, constants.MAX_RANK) = .{ .buffer = @splat(.offset), .len = self.rank() };

    for (self._shape.tags(), 0..self.rank()) |t, self_ax| {
        const is_gather_axis = std.mem.containsAtLeastScalar(u3, idx_axes, 1, @intCast(self_ax));
        if (indices_shape.hasTag(t)) |id_ax| {
            // tag is both in self and indices -> it's a batching dim
            // Note: tags are required for batching.
            self_kind.buffer[self_ax] = .batching;
            idx_batch_axes.appendAssumeCapacity(id_ax);
            stdx.debug.assert(!is_gather_axis, "gather expects axes to appear at most twice. Axis {s} has been found both in 'self={f}', in 'idx_axes={any}' and in 'indices={f}'", .{ t, self, idx_axes, indices_shape });
        } else if (is_gather_axis) {
            // we collapsed all gathered axes
            self_kind.buffer[self_ax] = .collapsed;
            // idx_kind.buffer[id_ax] = .indices;
        } else {
            self_kind.buffer[self_ax] = .offset;
        }
    }

    // compute res shape
    var res_shape = Shape.init(.{}, self.dtype());
    var res_kind: stdx.BoundedArray(GatherAxisKind, constants.MAX_RANK) = .{};
    for (self_kind.slice(), 0..) |kind, ax_usize| {
        const ax: u3 = @intCast(ax_usize);
        if (ax == idx_axes[0]) {
            // The first val_ax is special cause this is the place where we insert indices axes.
            for (0.., indices_shape.tags(), indices_shape.dims()) |id_axis_order, id_axis, id_inserted_dim| {
                const is_batching_axis = std.mem.containsAtLeastScalar(i64, idx_batch_axes.constSlice(), 1, @intCast(id_axis_order));
                // Batching axis is already in self.
                if (is_batching_axis) continue;

                res_shape = res_shape.appendDim(id_inserted_dim, id_axis);
                res_kind.appendAssumeCapacity(.indices);
            }
        }
        switch (kind) {
            .collapsed => continue,
            else => {
                res_shape = res_shape.appendDim(self.dim(ax), self._shape.tag(ax));
                res_kind.appendAssumeCapacity(kind);
            },
        }
    }

    // This is not a gather, but a dynamicSlice.
    // Sometimes the backend recognize this pattern, but not always.
    // So let us handle that.
    if (indices_shape.count() == 1 and idx_axes.len == 1) {
        return self.dynamicSlice1d(idx_axes[0], .{ .start = idx_per_axis[0].asScalar(), .len = 1 }).reshape(res_shape);
    }

    var slice_dims: Shape.DimsArray = .{};
    for (self_kind.slice(), self.dims()) |k, d| {
        slice_dims.appendAssumeCapacity(switch (k) {
            .batching, .collapsed => 1,
            .offset => d,
            .indices => unreachable,
        });
    }

    // TODO: try changing .last by other axis and see the perf impact.
    const indices = Tensor.stack(idx_per_axis, .last, .coord);
    // scoped_log.debug("gather --> {} {any}", .{ res_shape, res_kind.constSlice() });
    const gather_op = dialects.stablehlo.gather(
        mlir_ctx,
        self.value(),
        indices.value(),
        slice_dims.constSlice(),
        .{
            .offset_dims = _collectAxes(GatherAxisKind, res_kind, .offset).constSlice(),
            .collapsed_slice_dims = _collectAxes(GatherAxisKind, self_kind, .collapsed).constSlice(),
            .operand_batching_dims = _collectAxes(GatherAxisKind, self_kind, .batching).constSlice(),
            .start_indices_batching_dims = idx_batch_axes.constSlice(),
            .start_index_map = _collectAxes(GatherAxisKind, self_kind, .collapsed).constSlice(),
            .index_vector_dim = indices.axis(.coord),
            .indices_are_sorted = opts.indices_are_sorted,
        },
        .unknown(mlir_ctx),
    ).appendTo(CompilationContext.current().currentScope().block);

    const mlir_shape = Tensor.fromMlirValue(gather_op.result(0)).shape();
    stdx.debug.assert(mlir_shape.eql(res_shape), "gather expects that batching indices appear in the same order in 'self' and 'indices', got: self={f}, indices={f}. You should transpose one or the other.", .{ self, indices });
    return Tensor._result(res_shape, gather_op.result(0));
}

pub fn _collectAxes(T: type, bounded_array: stdx.BoundedArray(T, constants.MAX_RANK), value: T) stdx.BoundedArray(i64, constants.MAX_RANK) {
    var res: stdx.BoundedArray(i64, constants.MAX_RANK) = .{};
    for (bounded_array.constSlice(), 0..) |v, ax| {
        if (v == value) {
            res.appendAssumeCapacity(@intCast(ax));
        }
    }
    return res;
}

fn TensorOrTensorArray(comptime T: type) type {
    const type_info = @typeInfo(T);
    return switch (type_info) {
        .@"struct" => |struct_info| b: {
            if (T == Tensor) break :b Tensor;
            if (!struct_info.is_tuple) @compileError("Expected tuple");
            break :b if (struct_info.fields.len == 1)
                Tensor
            else
                [struct_info.fields.len]Tensor;
        },
        .array => |array_info| b: {
            break :b if (array_info.len == 1)
                Tensor
            else
                [array_info.len]Tensor;
        },
        .pointer => |pointer_info| b: {
            if (pointer_info.size != .slice) @compileError("Expected slice");
            break :b []Tensor;
        },
        else => @compileError("Unsupported type: " ++ @typeName(T)),
    };
}

pub const CustomCallOptions = struct {
    has_side_effect: bool,
    output_operand_aliases: ?[]const i64 = null,
};

pub fn customCall(target_name: [:0]const u8, inputs: anytype, outputs: anytype, metadata: anytype, opts: CustomCallOptions) TensorOrTensorArray(@TypeOf(outputs)) {
    // Transform generic inputs to flat slice.
    const inputs_: []const Tensor = switch (@typeInfo(@TypeOf(inputs))) {
        .@"struct" => |struct_info| b: {
            if (@TypeOf(inputs) == Tensor) {
                break :b &[1]Tensor{inputs};
            }
            if (!struct_info.is_tuple) @compileError("Expected tuple");
            var inputs_: [struct_info.fields.len]Tensor = undefined;
            meta.collectBuf((struct {
                pub fn func(t: Tensor) Tensor {
                    return t;
                }
            }).func, {}, &inputs, &inputs_);
            break :b &inputs_;
        },
        .array => &inputs,
        .pointer => |pointer_info| b: {
            if (pointer_info.size != .slice) @compileError("Expected slice");
            break :b inputs;
        },
        else => @compileError("Unsupported type: " ++ @typeName(@TypeOf(inputs))),
    };

    // Transform generic outputs to flat slice.
    const output_shapes: []const Shape = switch (@typeInfo(@TypeOf(outputs))) {
        .@"struct" => |struct_info| b: {
            if (@TypeOf(outputs) == Shape) {
                break :b &[1]Shape{outputs};
            }
            if (!struct_info.is_tuple) @compileError("Expected tuple");
            var output_shapes: [struct_info.fields.len]Shape = undefined;
            meta.collectBuf((struct {
                pub fn func(t: Shape) Shape {
                    return t;
                }
            }).func, {}, &outputs, &output_shapes);
            break :b &output_shapes;
        },
        .array => &outputs,
        .pointer => |pointer_info| b: {
            if (pointer_info.size != .slice) @compileError("Expected slice");
            break :b outputs;
        },
        else => @compileError("Unsupported type: " ++ @typeName(@TypeOf(outputs))),
    };

    const outputs_flat = customCallInternal(target_name, inputs_, output_shapes, metadata, opts);

    // Transform flat slice to generic outputs.
    return switch (@typeInfo(@TypeOf(outputs))) {
        .@"struct" => |struct_info| b: {
            if (@TypeOf(outputs) == Shape) break :b outputs_flat[0];
            if (!struct_info.is_tuple) @compileError("Expected tuple");
            if (struct_info.fields.len == 1) break :b outputs_flat[0];
            var outputs_: [struct_info.fields.len]Tensor = undefined;
            @memcpy(&outputs_, outputs_flat);
            break :b outputs_;
        },
        .array => |array_info| b: {
            if (array_info.len == 1) break :b outputs_flat[0];
            var outputs_: [array_info.fields.len]Tensor = undefined;
            @memcpy(&outputs_, outputs_flat);
            break :b outputs_;
        },
        .pointer => |pointer_info| b: {
            if (pointer_info.size != .slice) @compileError("Expected slice");
            break :b outputs_flat;
        },
        else => @compileError("Unsupported type: " ++ @typeName(@TypeOf(outputs))),
    };
}

fn customCallInternal(target_name: [:0]const u8, inputs: []const Tensor, outputs: []const Shape, metadata: anytype, opts: CustomCallOptions) []Tensor {
    const ctx = CompilationContext.current();
    var arena = std.heap.ArenaAllocator.init(ctx.allocator);
    defer arena.deinit();

    const values = arena.allocator().alloc(*const mlir.Value, inputs.len) catch unreachable;
    for (0..inputs.len) |i| {
        values[i] = inputs[i].value();
    }

    const res_types = arena.allocator().alloc(*const mlir.Type, outputs.len) catch unreachable;
    for (outputs, 0..) |output, i| {
        res_types[i] = mlir.rankedTensorType(output.dims(), mlirx.Type.fromDType(ctx.mlir_ctx, output.dtype()));
    }

    const metadata_type_info = @typeInfo(@TypeOf(metadata));
    var metadata_attributes: [metadata_type_info.@"struct".fields.len]mlir.NamedAttribute = undefined;
    inline for (metadata_type_info.@"struct".fields, 0..) |field, i| {
        const attribute: *const mlir.Attribute = switch (@typeInfo(field.type)) {
            .int, .comptime_int => mlir.integerAttribute(ctx.mlir_ctx, .u64, @as(u64, @bitCast(@field(metadata, field.name)))),
            else => @compileError("Unsupported metadata type: " ++ @typeName(field.type)),
        };
        metadata_attributes[i] = mlir.NamedAttribute.named(ctx.mlir_ctx, field.name, attribute);
    }

    const backend_config = mlir.dictionaryAttribute(ctx.mlir_ctx, &(metadata_attributes ++ [_]mlir.NamedAttribute{
        .named(ctx.mlir_ctx, "pjrt_api", mlir.integerAttribute(ctx.mlir_ctx, .u64, @as(u64, @bitCast(@intFromPtr(ctx.platform.pjrt_api))))),
        .named(ctx.mlir_ctx, "pjrt_client", mlir.integerAttribute(ctx.mlir_ctx, .u64, @as(u64, @bitCast(@intFromPtr(ctx.platform.pjrt_client))))),
    }));

    const operands_layouts = arena.allocator().alloc([]const usize, inputs.len) catch unreachable;
    for (inputs, 0..) |input, i| {
        operands_layouts[i] = arena.allocator().dupe(usize, toUsize(constants.minorToMajor(input.rank())).constSlice()) catch unreachable;
    }

    const results_layouts = arena.allocator().alloc([]const usize, outputs.len) catch unreachable;
    for (outputs, 0..) |output, i| {
        results_layouts[i] = arena.allocator().dupe(usize, toUsize(constants.minorToMajor(output.rank())).constSlice()) catch unreachable;
    }

    const op = dialects.stablehlo.custom_call(
        ctx.mlir_ctx,
        values,
        res_types,
        .{
            .call_target_name = target_name,
            .backend_config = .{ .typed_ffi = backend_config },
            .has_side_effect = opts.has_side_effect,
            .operand_layouts = operands_layouts,
            .result_layouts = results_layouts,
            .output_operand_aliases = opts.output_operand_aliases orelse &.{},
        },
        .unknown(ctx.mlir_ctx),
    );

    const outputs_ = ctx.arena.allocator().alloc(Tensor, outputs.len) catch unreachable;
    for (outputs, 0..) |output, i| {
        outputs_[i] = Tensor._result(output, op.result(i));
    }

    return outputs_;
}

test customCall {
    const zml = @import("zml.zig");
    const platform = zml.testing.env();

    var comp = zml.module.CompilationContext.init(std.testing.allocator, platform);
    defer comp.deinit();
    comp.activate();
    defer comp.deactivate();

    const block = mlir.Block.init(&.{}, &.{});
    comp.pushBlock(block);
    defer comp.popBlock();

    const input = Tensor.constant(zml.DataType.bf16.constant(0)).broad(Shape.init(.{128}, .bf16));
    const output = customCall("my_custom_call", .{input}, .{zml.Shape.init(.{128}, .bf16)}, .{}, .{
        .has_side_effect = false,
        .output_operand_aliases = &.{0},
    });

    try zml.testing.expectEqualShapes(input.shape(), output.shape());
}

fn toUsize(values: anytype) stdx.BoundedArray(usize, constants.MAX_RANK) {
    var res: stdx.BoundedArray(usize, constants.MAX_RANK) = .{};
    for (values) |val| res.appendAssumeCapacity(@intCast(val));
    return res;
}
