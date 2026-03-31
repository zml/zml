const std = @import("std");

const dialects = @import("mlir/dialects");
const mlir = @import("mlir");
const pjrt = @import("pjrt");
const stdx = @import("stdx");

const Buffer = @import("buffer.zig").Buffer;
const Bufferized = @import("mem.zig").Bufferized;
const CompilationContext = @import("module.zig").CompilationContext;
const constants = @import("constants.zig");
const CustomCallBuffer = @import("pjrtx.zig").CustomCallBuffer;
const DataType = @import("dtype.zig").DataType;
const meta = @import("meta.zig");
const mlirx = @import("mlirx.zig");
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;
const ShapeToCustomCallBuffer = @import("pjrtx.zig").ShapeToCustomCallBuffer;
const Tensor = @import("tensor.zig").Tensor;
const TensorToCustomCallBuffer = @import("pjrtx.zig").TensorToCustomCallBuffer;

pub const ReduceArgs = struct {
    left: Tensor,
    right: Tensor,
};

pub fn reduce(inputs: anytype, inits: anytype, axes_: []const i64, comptime func: anytype, context: anytype) stdx.meta.FnReturn(func) {
    var arena = std.heap.ArenaAllocator.init(CompilationContext.current().allocator);
    defer arena.deinit();

    const mlir_ctx = CompilationContext.current().mlir_ctx;

    const reduce_block, var result = b: {
        const ArgsType = std.meta.Tuple(&[1]type{ReduceArgs} ** inits.len);
        var args: ArgsType = undefined;
        var block_types: [2 * inits.len]*const mlir.Type = undefined;

        inline for (0..inits.len) |i| {
            args[i].left = .fromShape(inits[i].shape());
            args[i].right = .fromShape(inits[i].shape());

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

pub fn reduceWindow(inputs: anytype, inits: anytype, opts: ReduceWindowOpts, comptime func: anytype, context: anytype) stdx.meta.FnReturn(func) {
    var arena = std.heap.ArenaAllocator.init(CompilationContext.current().allocator);
    defer arena.deinit();

    const mlir_ctx = CompilationContext.current().mlir_ctx;

    const reduce_block, var result = b: {
        const ArgsType = std.meta.Tuple(&[1]type{ReduceArgs} ** inits.len);
        var args: ArgsType = undefined;
        var block_types: [2 * inits.len]*const mlir.Type = undefined;

        inline for (0..inits.len) |i| {
            args[i].left = .fromShape(inits[i].shape());
            args[i].right = .fromShape(inits[i].shape());

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
            args[i].left = Tensor.init(.{}, inputs[i].shape().dtype());
            args[i].right = Tensor.init(.{}, inputs[i].shape().dtype());

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

pub fn @"while"(operands: anytype, comptime cond: anytype, comptime body: anytype, context: anytype) stdx.meta.FnReturn(body) {
    stdx.debug.assertComptime(stdx.meta.isTupleOf(@TypeOf(operands), Tensor), "zml.ops.while expects a tuple of Tensor operands, got {}", .{@TypeOf(operands)});
    stdx.debug.assertComptime(stdx.meta.isTuple(@TypeOf(context)), "zml.ops.while expects tuple context like .{{ ... }}, got {}", .{@TypeOf(context)});

    var arena = std.heap.ArenaAllocator.init(CompilationContext.current().allocator);
    defer arena.deinit();

    const mlir_ctx = CompilationContext.current().mlir_ctx;
    const location = mlir.Location.unknown(mlir_ctx);

    var captured_context: @TypeOf(context) = undefined;
    meta.mapAlloc(struct {
        fn capture(_: void, tensor: Tensor) Tensor {
            return Tensor._result(tensor.shape(), tensor.value());
        }
    }.capture, arena.allocator(), {}, context, &captured_context) catch unreachable;

    var block_types: [operands.len]*const mlir.Type = undefined;
    var block_locs: [operands.len]*const mlir.Location = @splat(location);
    var operand_values: [operands.len]*const mlir.Value = undefined;
    var operand_shapes: [operands.len]Shape = undefined;

    inline for (0..operands.len) |i| {
        operand_shapes[i] = operands[i].shape();
        block_types[i] = mlir.rankedTensorType(operand_shapes[i].dims(), mlirx.Type.fromDType(mlir_ctx, operand_shapes[i].dtype()));
        operand_values[i] = operands[i].value();
    }

    const cond_block = b: {
        var cond_args: @TypeOf(operands) = undefined;
        inline for (0..operands.len) |i| {
            cond_args[i] = .fromShape(operand_shapes[i]);
        }

        const block = mlir.Block.init(&block_types, &block_locs);
        errdefer block.deinit();

        CompilationContext.current().pushBlock(block);
        defer CompilationContext.current().popBlock();

        const scope = CompilationContext.current().currentScope();
        inline for (0..operands.len) |i| {
            scope.id_to_argument.put(scope.arena.allocator(), cond_args[i].id, i) catch unreachable;
        }

        const cond_result = @call(.auto, cond, cond_args ++ captured_context);
        stdx.debug.assert(meta.count(Tensor, &cond_result) == 1, "zml.ops.while expects cond to return exactly one Tensor, got {}", .{@TypeOf(cond_result)});

        const cond_tensor = meta.first(Tensor, cond_result);
        stdx.debug.assert(cond_tensor.rank() == 0 and cond_tensor.dtype() == .bool, "zml.ops.while expects cond to return a scalar bool Tensor, got {f}", .{cond_tensor});

        _ = dialects.stablehlo.return_(mlir_ctx, cond_tensor.value(), location).appendTo(block);
        break :b block;
    };

    const body_block, var result = b: {
        var body_args: @TypeOf(operands) = undefined;
        inline for (0..operands.len) |i| {
            body_args[i] = .fromShape(operand_shapes[i]);
        }

        const block = mlir.Block.init(&block_types, &block_locs);
        errdefer block.deinit();

        CompilationContext.current().pushBlock(block);
        defer CompilationContext.current().popBlock();

        const scope = CompilationContext.current().currentScope();
        inline for (0..operands.len) |i| {
            scope.id_to_argument.put(scope.arena.allocator(), body_args[i].id, i) catch unreachable;
        }

        const result = @call(.auto, body, body_args ++ captured_context);
        stdx.debug.assert(meta.count(Tensor, &result) == operands.len, "zml.ops.while expects body to return {d} Tensor values, got {}", .{ operands.len, @TypeOf(result) });

        var result_tensors: [operands.len]Tensor = undefined;
        meta.collectBuf(struct {
            fn cb(tensor: Tensor) Tensor {
                return tensor;
            }
        }.cb, {}, &result, &result_tensors);

        var result_values: [operands.len]*const mlir.Value = undefined;
        inline for (0..operands.len) |i| {
            stdx.debug.assert(result_tensors[i].shape().eql(operand_shapes[i]), "zml.ops.while expects body to preserve loop state shape, got {f} for operand {d} but expected {f}", .{ result_tensors[i], i, operands[i] });
            result_values[i] = result_tensors[i].value();
        }

        _ = dialects.stablehlo.returns(mlir_ctx, &result_values, location).appendTo(block);
        break :b .{ block, result };
    };

    const op = dialects.stablehlo.while_(
        mlir_ctx,
        &operand_values,
        &block_types,
        cond_block,
        body_block,
        location,
    ).appendTo(CompilationContext.current().currentScope().block);

    const n_operands = operands.len;
    const AssignResultCtx = struct {
        op_: *mlir.Operation,
        shapes: [n_operands]Shape,
        idx: usize = 0,
    };
    var assign_result_ctx: AssignResultCtx = .{
        .op_ = op,
        .shapes = operand_shapes,
    };
    meta.visit(struct {
        fn cb(ctx: *AssignResultCtx, tensor: *Tensor) void {
            tensor.* = Tensor._result(ctx.shapes[ctx.idx], ctx.op_.result(ctx.idx));
            ctx.idx += 1;
        }
    }.cb, &assign_result_ctx, &result);

    return result;
}

pub const while_ = @"while";

test @"while" {
    const zml = @import("zml.zig");
    const platform = zml.testing.env();
    const replicated_sharding = zml.testing.replicatedSharding();

    const initial_i: zml.Tensor = .init(.{}, .i64);
    const initial_sum: zml.Tensor = .init(.{}, .i64);

    const Local = struct {
        fn cond(i: Tensor, sum: Tensor) Tensor {
            _ = sum;
            return i.cmp(.LT, Tensor.scalar(10, .i64));
        }

        fn body(i: Tensor, sum: Tensor) [2]Tensor {
            const one = Tensor.scalar(1, .i64);
            return .{
                i.add(one),
                sum.add(one),
            };
        }

        fn forward(i: Tensor, sum: Tensor) [2]Tensor {
            return zml.ops.@"while"(.{ i, sum }, cond, body, .{});
        }
    };

    var exe = try zml.module.compile(
        std.testing.allocator,
        std.testing.io,
        Local.forward,
        .{ initial_i, initial_sum },
        platform,
        .{ .shardings = &.{replicated_sharding} },
    );
    defer exe.deinit();

    var i_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, initial_i.shape(), replicated_sharding, std.mem.sliceAsBytes(&[1]i64{1}));
    defer i_buffer.deinit();
    var sum_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, initial_sum.shape(), replicated_sharding, std.mem.sliceAsBytes(&[1]i64{0}));
    defer sum_buffer.deinit();

    var results = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Local.forward, .{ i_buffer, sum_buffer });
    defer results[0].deinit();
    defer results[1].deinit();

    var cpu_i = try results[0].toSliceAlloc(std.testing.allocator, std.testing.io);
    defer cpu_i.free(std.testing.allocator);
    var cpu_sum = try results[1].toSliceAlloc(std.testing.allocator, std.testing.io);
    defer cpu_sum.free(std.testing.allocator);

    try std.testing.expectEqual(@as(i64, 10), cpu_i.items(i64)[0]);
    try std.testing.expectEqual(@as(i64, 9), cpu_sum.items(i64)[0]);
}

pub const TritonOps = struct {
    debug: bool = false,
    name: [:0]const u8,
    ir: [:0]const u8,
    grid: [3]i32,
    num_stages: i32,
    num_warps: i32,
    output_operand_aliases: []const dialects.stablehlo.CustomCallOpts.OutputOperandAlias = &.{},
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

    const a: zml.Tensor = .init(.{}, .f32);
    const b: zml.Tensor = .init(.{}, .f32);

    var exe = try zml.module.compile(std.testing.allocator, std.testing.io, TritonMod.forward, .{ a, b }, platform, .{ .shardings = &.{zml.testing.replicatedSharding()} });
    defer exe.deinit();

    var a_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, a.shape(), zml.testing.replicatedSharding(), std.mem.sliceAsBytes(&[1]f32{1}));
    defer a_buffer.deinit();
    var b_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, b.shape(), zml.testing.replicatedSharding(), std.mem.sliceAsBytes(&[1]f32{3}));
    defer b_buffer.deinit();

    var results = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, TritonMod.forward, .{ a_buffer, b_buffer });
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
) stdx.meta.FnReturn(func) {
    var arena = std.heap.ArenaAllocator.init(CompilationContext.current().allocator);
    defer arena.deinit();

    const mlir_ctx = CompilationContext.current().mlir_ctx;

    const update_block, var result = b: {
        const ArgsType = std.meta.Tuple(&[1]type{ScatterArgs} ** inputs.len);
        var args: ArgsType = undefined;
        var block_types: [2 * inputs.len]*const mlir.Type = undefined;

        inline for (0..inputs.len) |i| {
            args[i].input = Tensor.init(.{}, inputs[i].dtype());
            args[i].update = Tensor.init(.{}, inputs[i].dtype());

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

    var comp = zml.module.CompilationContext.init(std.testing.allocator, std.testing.io, platform, .{ .shardings = &.{zml.testing.replicatedSharding()} });
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

fn CustomCallResultTypeFromOutputSpec(comptime OutputSpecT: type) type {
    const type_info = @typeInfo(OutputSpecT);
    return switch (type_info) {
        .void => void,
        .@"struct" => |struct_info| b: {
            if (OutputSpecT == Shape) break :b Tensor;
            if (!struct_info.is_tuple) @compileError("Expected tuple output shape spec");
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
            if (pointer_info.size != .slice) @compileError("Expected output shape slice");
            break :b []Tensor;
        },
        else => @compileError("Unsupported customCall output shape spec type: " ++ @typeName(OutputSpecT)),
    };
}

pub const CustomCallOptions = struct {
    has_side_effect: bool,
    output_operand_aliases: ?[]const dialects.stablehlo.CustomCallOpts.OutputOperandAlias = null,
};

fn customCallOutputOperandAliases(
    comptime I: type,
    comptime O: type,
    comptime aliases: ?CustomCallOutputOperandAliases(I, O),
) ?[]const dialects.stablehlo.CustomCallOpts.OutputOperandAlias {
    const aliases_ = aliases orelse return null;

    const output_fields = @typeInfo(O).@"struct".fields;
    comptime var num_aliases = 0;
    inline for (output_fields) |field| {
        if (@field(aliases_, field.name) != null) num_aliases += 1;
    }

    const output_operand_aliases = comptime blk: {
        var aliases_buffer: [num_aliases]dialects.stablehlo.CustomCallOpts.OutputOperandAlias = undefined;
        var i = 0;
        for (output_fields, 0..) |field, output_index| {
            if (@field(aliases_, field.name)) |operand| {
                aliases_buffer[i] = .{
                    .output_index = @intCast(output_index),
                    .operand_index = @intCast(@intFromEnum(operand)),
                };
                i += 1;
            }
        }
        break :blk aliases_buffer;
    };

    return &output_operand_aliases;
}

pub fn customCall(target_name: [:0]const u8, inputs: anytype, outputs: anytype, metadata: anytype, opts: CustomCallOptions) CustomCallResultTypeFromOutputSpec(@TypeOf(outputs)) {
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
        .void => &.{},
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
    if (comptime CustomCallResultTypeFromOutputSpec(@TypeOf(outputs)) == void) {
        stdx.debug.assert(outputs_flat.len == 0, "customCall expected zero outputs for void return, got {}", .{outputs_flat.len});
        return;
    }

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

pub fn manualComputation(
    inputs: anytype,
    outputs: anytype,
    body_context: anytype,
    comptime body_fn: anytype,
) manualComputationReturnType(body_fn) {
    const inputs_: []const Tensor = switch (@typeInfo(@TypeOf(inputs))) {
        .@"struct" => |struct_info| b: {
            if (@TypeOf(inputs) == Tensor) {
                break :b &[1]Tensor{inputs};
            }
            if (!struct_info.is_tuple) @compileError("Expected tuple inputs");
            var inputs_flat: [struct_info.fields.len]Tensor = undefined;
            meta.collectBuf((struct {
                pub fn func(t: Tensor) Tensor {
                    return t;
                }
            }).func, {}, &inputs, &inputs_flat);
            break :b &inputs_flat;
        },
        .array => &inputs,
        .pointer => |pointer_info| b: {
            if (pointer_info.size != .slice) @compileError("Expected input slice");
            break :b inputs;
        },
        else => @compileError("Unsupported manualComputation input type: " ++ @typeName(@TypeOf(inputs))),
    };

    const output_shapes: []const Shape = switch (@typeInfo(@TypeOf(outputs))) {
        .void => &.{},
        .@"struct" => |struct_info| b: {
            if (@TypeOf(outputs) == Shape) {
                break :b &[1]Shape{outputs};
            }
            if (!struct_info.is_tuple) @compileError("Expected tuple output shapes");
            var output_shapes_flat: [struct_info.fields.len]Shape = undefined;
            meta.collectBuf((struct {
                pub fn func(t: Shape) Shape {
                    return t;
                }
            }).func, {}, &outputs, &output_shapes_flat);
            break :b &output_shapes_flat;
        },
        .array => &outputs,
        .pointer => |pointer_info| b: {
            if (pointer_info.size != .slice) @compileError("Expected output shape slice");
            break :b outputs;
        },
        else => @compileError("Unsupported manualComputation output type: " ++ @typeName(@TypeOf(outputs))),
    };

    const ReturnT = manualComputationReturnType(body_fn);
    return manualComputationSliceToReturn(ReturnT, manualComputationInternal(inputs_, output_shapes, body_context, body_fn));
}

fn manualComputationInternal(
    inputs: []const Tensor,
    outputs: []const Shape,
    body_context: anytype,
    comptime body_fn: anytype,
) []Tensor {
    const BodyReturnT = manualComputationReturnType(body_fn);
    const BodyInputsT = stdx.meta.FnParam(body_fn, 2);
    const BodyOutputShapesT = stdx.meta.FnParam(body_fn, 3);

    const ctx = CompilationContext.current();
    const allocator = ctx.arena.allocator();

    const input_shapes = allocator.alloc(Shape, inputs.len) catch unreachable;
    const input_values = allocator.alloc(*const mlir.Value, inputs.len) catch unreachable;

    for (inputs, 0..) |input, i| {
        input_shapes[i] = input.shape();
        input_values[i] = input.value();
    }

    return switch (ctx.partitioning.partitioner) {
        .shardy => blk: {
            const local_input_shapes = allocator.alloc(Shape, inputs.len) catch unreachable;
            const local_output_shapes = allocator.alloc(Shape, outputs.len) catch unreachable;

            for (input_shapes, 0..) |input_shape, i| {
                local_input_shapes[i] = ctx.partitioning.localShapeForShape(input_shape) catch unreachable;
            }
            for (outputs, 0..) |output_shape, i| {
                local_output_shapes[i] = ctx.partitioning.localShapeForShape(output_shape) catch unreachable;
            }

            const in_shardings_str = ctx.partitioning.sdyPerValueShardingAttr(allocator, input_shapes) catch unreachable;
            const out_shardings_str = ctx.partitioning.sdyPerValueShardingAttr(allocator, outputs) catch unreachable;
            const manual_axes_str = ctx.partitioning.sdyManualAxesAttr(allocator, input_shapes, outputs) catch unreachable;
            const in_shardings_attr = mlir.Attribute.parse(ctx.mlir_ctx, in_shardings_str) catch unreachable;
            const out_shardings_attr = mlir.Attribute.parse(ctx.mlir_ctx, out_shardings_str) catch unreachable;
            const manual_axes_attr = mlir.Attribute.parse(ctx.mlir_ctx, manual_axes_str) catch unreachable;

            const block_types = allocator.alloc(*const mlir.Type, inputs.len) catch unreachable;
            const block_locs = allocator.alloc(*const mlir.Location, inputs.len) catch unreachable;
            for (local_input_shapes, 0..) |input_shape, i| {
                block_types[i] = mlir.rankedTensorType(input_shape.dims(), mlirx.Type.fromDType(ctx.mlir_ctx, input_shape.dtype()));
                block_locs[i] = mlir.Location.unknown(ctx.mlir_ctx);
            }

            const parent_block = ctx.currentScope().block;
            const manual_block = mlir.Block.init(block_types, block_locs);
            errdefer manual_block.deinit();

            ctx.pushBlock(manual_block);
            defer ctx.popBlock();

            const local_inputs = allocator.alloc(Tensor, inputs.len) catch unreachable;
            for (0..inputs.len) |i| {
                local_inputs[i] = Tensor._result(local_input_shapes[i], manual_block.argument(i));
            }

            ctx.manual_computation_depth += 1;
            defer ctx.manual_computation_depth -= 1;

            const body_inputs = manualComputationInputsArg(BodyInputsT, local_inputs);
            const body_output_shapes = manualComputationOutputShapesArg(BodyOutputShapesT, local_output_shapes);
            const body_result = @call(.auto, body_fn, .{ body_context, allocator, body_inputs, body_output_shapes });
            const local_outputs = manualComputationBodyToSlice(BodyReturnT, allocator, body_result);
            stdx.debug.assert(local_outputs.len == outputs.len, "manualComputation body returned {} values, expected {}", .{ local_outputs.len, outputs.len });

            const local_output_values = allocator.alloc(*const mlir.Value, outputs.len) catch unreachable;
            for (0..outputs.len) |i| {
                stdx.debug.assert(local_outputs[i].shape().eql(local_output_shapes[i]), "manualComputation body returned shape {f}, expected {f}", .{ local_outputs[i].shape(), local_output_shapes[i] });
                local_output_values[i] = local_outputs[i].value();
            }

            _ = mlir.Operation.make(ctx.mlir_ctx, "sdy.return", .{
                .operands = .{ .flat = local_output_values },
                .verify = false,
                .location = .unknown(ctx.mlir_ctx),
            }).appendTo(manual_block);

            const global_result_types = allocator.alloc(*const mlir.Type, outputs.len) catch unreachable;
            for (outputs, 0..) |output_shape, i| {
                global_result_types[i] = mlir.rankedTensorType(output_shape.dims(), mlirx.Type.fromDType(ctx.mlir_ctx, output_shape.dtype()));
            }

            const op = mlir.Operation.make(ctx.mlir_ctx, "sdy.manual_computation", .{
                .operands = .{ .flat = input_values },
                .results = .{ .flat = global_result_types },
                .blocks = &.{manual_block},
                .attributes = &.{
                    .named(ctx.mlir_ctx, "in_shardings", in_shardings_attr),
                    .named(ctx.mlir_ctx, "out_shardings", out_shardings_attr),
                    .named(ctx.mlir_ctx, "manual_axes", manual_axes_attr),
                },
                .verify = true,
                .location = .unknown(ctx.mlir_ctx),
            }).appendTo(parent_block);

            const outputs_ = allocator.alloc(Tensor, outputs.len) catch unreachable;
            for (outputs, 0..) |output, i| {
                outputs_[i] = Tensor._result(output, op.result(i));
            }
            break :blk outputs_;
        },
        .gspmd => blk: {
            const manual_sharding = "{manual}";
            const output_shardings = allocator.alloc([]const u8, outputs.len) catch unreachable;
            const local_input_shapes = allocator.alloc(Shape, inputs.len) catch unreachable;
            const local_output_shapes = allocator.alloc(Shape, outputs.len) catch unreachable;

            for (input_shapes, 0..) |input_shape, i| {
                local_input_shapes[i] = ctx.partitioning.localShapeForShape(input_shape) catch unreachable;
            }
            for (outputs, 0..) |output_shape, i| {
                local_output_shapes[i] = ctx.partitioning.localShapeForShape(output_shape) catch unreachable;
                output_shardings[i] = ctx.partitioning.tensorShardingAttr(allocator, output_shape, null) catch unreachable;
            }

            const local_input_values = allocator.alloc(*const mlir.Value, inputs.len) catch unreachable;
            for (0..inputs.len) |i| {
                const local_type = mlir.rankedTensorType(local_input_shapes[i].dims(), mlirx.Type.fromDType(ctx.mlir_ctx, local_input_shapes[i].dtype()));
                const full_to_shard = dialects.stablehlo.custom_call(
                    ctx.mlir_ctx,
                    &.{input_values[i]},
                    &.{local_type},
                    .{
                        .call_target_name = "SPMDFullToShardShape",
                        .has_side_effect = false,
                        .backend_config = .{ .original = "" },
                        .additional_attributes = &.{
                            .named(ctx.mlir_ctx, "mhlo.sharding", mlir.stringAttribute(ctx.mlir_ctx, manual_sharding)),
                        },
                    },
                    .unknown(ctx.mlir_ctx),
                ).appendTo(ctx.currentScope().block);
                local_input_values[i] = full_to_shard.result(0);
            }

            const local_inputs = allocator.alloc(Tensor, inputs.len) catch unreachable;
            for (0..inputs.len) |i| {
                local_inputs[i] = Tensor._result(local_input_shapes[i], local_input_values[i]);
            }

            ctx.manual_computation_depth += 1;
            defer ctx.manual_computation_depth -= 1;
            const body_inputs = manualComputationInputsArg(BodyInputsT, local_inputs);
            const body_output_shapes = manualComputationOutputShapesArg(BodyOutputShapesT, local_output_shapes);
            const body_result = @call(.auto, body_fn, .{ body_context, allocator, body_inputs, body_output_shapes });
            const local_outputs = manualComputationBodyToSlice(BodyReturnT, allocator, body_result);
            stdx.debug.assert(local_outputs.len == outputs.len, "manualComputation body returned {} values, expected {}", .{ local_outputs.len, outputs.len });
            for (0..outputs.len) |i| {
                stdx.debug.assert(local_outputs[i].shape().eql(local_output_shapes[i]), "manualComputation body returned shape {f}, expected {f}", .{ local_outputs[i].shape(), local_output_shapes[i] });
            }

            if (outputs.len == 0) {
                break :blk allocator.alloc(Tensor, 0) catch unreachable;
            }

            const global_values = allocator.alloc(*const mlir.Value, outputs.len) catch unreachable;
            const global_types = allocator.alloc(*const mlir.Type, outputs.len) catch unreachable;
            for (outputs, 0..) |output_shape, i| {
                global_types[i] = mlir.rankedTensorType(output_shape.dims(), mlirx.Type.fromDType(ctx.mlir_ctx, output_shape.dtype()));
                const shard_to_full = dialects.stablehlo.custom_call(
                    ctx.mlir_ctx,
                    &.{local_outputs[i].value()},
                    &.{global_types[i]},
                    .{
                        .call_target_name = "SPMDShardToFullShape",
                        .has_side_effect = false,
                        .backend_config = .{ .original = "" },
                        .additional_attributes = &.{
                            .named(ctx.mlir_ctx, "mhlo.sharding", mlir.stringAttribute(ctx.mlir_ctx, output_shardings[i])),
                        },
                    },
                    .unknown(ctx.mlir_ctx),
                ).appendTo(ctx.currentScope().block);
                global_values[i] = shard_to_full.result(0);
            }

            const barrier = mlir.Operation.make(ctx.mlir_ctx, "stablehlo.optimization_barrier", .{
                .operands = .{ .flat = global_values },
                .results = .{ .flat = global_types },
                .location = .unknown(ctx.mlir_ctx),
            }).appendTo(ctx.currentScope().block);

            const outputs_ = allocator.alloc(Tensor, outputs.len) catch unreachable;
            for (outputs, 0..) |output_shape, i| {
                outputs_[i] = Tensor._result(output_shape, barrier.result(i));
            }
            break :blk outputs_;
        },
    };
}

fn manualComputationReturnType(comptime body_fn: anytype) type {
    const ReturnT = stdx.meta.FnReturn(body_fn);
    if (ReturnT == void or ReturnT == Tensor) return ReturnT;

    return switch (@typeInfo(ReturnT)) {
        .pointer => |pointer_info| b: {
            if (pointer_info.size != .slice or pointer_info.child != Tensor) {
                @compileError("manualComputation body must return void, Tensor, []Tensor, or []const Tensor; got " ++ @typeName(ReturnT));
            }
            break :b ReturnT;
        },
        else => @compileError("manualComputation body must return void, Tensor, []Tensor, or []const Tensor; got " ++ @typeName(ReturnT)),
    };
}

fn manualComputationBodyToSlice(comptime ReturnT: type, allocator: std.mem.Allocator, result: ReturnT) []const Tensor {
    if (ReturnT == void) return &.{};
    if (ReturnT == Tensor) {
        const out = allocator.alloc(Tensor, 1) catch unreachable;
        out[0] = result;
        return out;
    }

    return result;
}

fn manualComputationInputsArg(comptime InputsT: type, local_inputs: []const Tensor) InputsT {
    if (InputsT == void) {
        stdx.debug.assert(local_inputs.len == 0, "manualComputation body expects no inputs, got {}", .{local_inputs.len});
        return;
    }
    if (InputsT == Tensor) {
        stdx.debug.assert(local_inputs.len == 1, "manualComputation body expects one input, got {}", .{local_inputs.len});
        return local_inputs[0];
    }

    return switch (@typeInfo(InputsT)) {
        .pointer => |pointer_info| b: {
            if (pointer_info.size != .slice or pointer_info.child != Tensor) {
                @compileError("manualComputation body inputs must be void, Tensor, []Tensor, or []const Tensor; got " ++ @typeName(InputsT));
            }
            break :b if (pointer_info.is_const)
                local_inputs
            else
                @constCast(local_inputs);
        },
        else => @compileError("manualComputation body inputs must be void, Tensor, []Tensor, or []const Tensor; got " ++ @typeName(InputsT)),
    };
}

fn manualComputationOutputShapesArg(comptime OutputShapesT: type, local_output_shapes: []const Shape) OutputShapesT {
    if (OutputShapesT == void) {
        stdx.debug.assert(local_output_shapes.len == 0, "manualComputation body expects no output shapes, got {}", .{local_output_shapes.len});
        return;
    }
    if (OutputShapesT == Shape) {
        stdx.debug.assert(local_output_shapes.len == 1, "manualComputation body expects one output shape, got {}", .{local_output_shapes.len});
        return local_output_shapes[0];
    }

    return switch (@typeInfo(OutputShapesT)) {
        .pointer => |pointer_info| b: {
            if (pointer_info.size != .slice or pointer_info.child != Shape) {
                @compileError("manualComputation body output shapes must be void, Shape, []Shape, or []const Shape; got " ++ @typeName(OutputShapesT));
            }
            break :b if (pointer_info.is_const)
                local_output_shapes
            else
                @constCast(local_output_shapes);
        },
        else => @compileError("manualComputation body output shapes must be void, Shape, []Shape, or []const Shape; got " ++ @typeName(OutputShapesT)),
    };
}

fn manualComputationSliceToReturn(comptime ReturnT: type, outputs: []Tensor) ReturnT {
    if (ReturnT == void) {
        stdx.debug.assert(outputs.len == 0, "manualComputation expected no outputs, got {}", .{outputs.len});
        return;
    }
    if (ReturnT == Tensor) {
        stdx.debug.assert(outputs.len == 1, "manualComputation expected one output, got {}", .{outputs.len});
        return outputs[0];
    }
    return outputs;
}

fn metadataIntegerFieldToMlirAttribute(mlir_ctx: *mlir.Context, int_field: std.builtin.Type.Int, value: anytype) *const mlir.Attribute {
    return switch (int_field.signedness) {
        .signed => switch (int_field.bits) {
            8 => mlir.integerAttribute(mlir_ctx, .i8, value),
            16 => mlir.integerAttribute(mlir_ctx, .i16, value),
            32 => mlir.integerAttribute(mlir_ctx, .i32, value),
            64 => mlir.integerAttribute(mlir_ctx, .i64, value),
            else => @panic("Unsupported DataType"),
        },
        .unsigned => switch (int_field.bits) {
            8 => mlir.integerAttribute(mlir_ctx, .u8, value),
            16 => mlir.integerAttribute(mlir_ctx, .u16, value),
            32 => mlir.integerAttribute(mlir_ctx, .u32, value),
            64 => mlir.integerAttribute(mlir_ctx, .u64, value),
            else => @panic("Unsupported DataType"),
        },
    };
}

fn metadataFloatFieldToMlirAttribute(mlir_ctx: *mlir.Context, float_field: std.builtin.Type.Float, value: anytype) *const mlir.Attribute {
    return switch (float_field.bits) {
        16 => mlir.floatAttribute(mlir_ctx, .f16, @as(f64, value)),
        32 => mlir.floatAttribute(mlir_ctx, .f32, @as(f64, value)),
        64 => mlir.floatAttribute(mlir_ctx, .f64, @as(f64, value)),
        else => @panic("Unsupported DataType"),
    };
}

fn metadataFieldToMlirAttribute(mlir_ctx: *mlir.Context, comptime T: type, value: anytype) ?*const mlir.Attribute {
    const type_info = @typeInfo(T);
    return switch (type_info) {
        .comptime_int => mlir.integerAttribute(mlir_ctx, .u64, @as(u64, value)),
        .int => |int_field| metadataIntegerFieldToMlirAttribute(mlir_ctx, int_field, value),
        .float => |float_field| metadataFloatFieldToMlirAttribute(mlir_ctx, float_field, value),
        .bool => mlir.boolAttribute(mlir_ctx, value),
        .pointer => |pointer_info| switch (pointer_info.size) {
            .slice => if (pointer_info.child == u8)
                mlir.stringAttribute(mlir_ctx, value)
            else
                @panic("Unsupported pointer type in metadata"),
            else => @panic("Unsupported pointer type in metadata"),
        },
        .optional => |optional_info| if (value) |wrapped_value| switch (@typeInfo(optional_info.child)) {
            .int => |int_field| metadataIntegerFieldToMlirAttribute(mlir_ctx, int_field, wrapped_value),
            .float => |float_field| metadataFloatFieldToMlirAttribute(mlir_ctx, float_field, wrapped_value),
            .bool => mlir.boolAttribute(mlir_ctx, wrapped_value),
            else => @panic("Unsupported optional child type in metadata"),
        } else null,
        else => @compileError("Unsupported metadata type: " ++ @typeName(T)),
    };
}

fn customCallInternal(target_name: [:0]const u8, inputs: []const Tensor, outputs: []const Shape, metadata: anytype, opts: CustomCallOptions) []Tensor {
    const ctx = CompilationContext.current();
    const allocator = ctx.arena.allocator();

    stdx.debug.assert(!opts.has_side_effect or ctx.manual_computation_depth > 0, "side-effect customCall '{s}' must be emitted inside manualComputation", .{target_name});

    const values = allocator.alloc(*const mlir.Value, inputs.len) catch unreachable;
    const input_shapes = allocator.alloc(Shape, inputs.len) catch unreachable;
    for (inputs, 0..) |input, i| {
        values[i] = input.value();
        input_shapes[i] = input.shape();
    }

    const result_types = allocator.alloc(*const mlir.Type, outputs.len) catch unreachable;
    for (outputs, 0..) |shape, i| {
        result_types[i] = mlir.rankedTensorType(shape.dims(), mlirx.Type.fromDType(ctx.mlir_ctx, shape.dtype()));
    }

    const metadata_type_info = @typeInfo(@TypeOf(metadata));
    var metadata_attribute_list = std.ArrayList(mlir.NamedAttribute).initCapacity(allocator, metadata_type_info.@"struct".fields.len + 2) catch unreachable;
    inline for (metadata_type_info.@"struct".fields) |field| {
        if (metadataFieldToMlirAttribute(ctx.mlir_ctx, field.type, @field(metadata, field.name))) |attribute| {
            metadata_attribute_list.appendAssumeCapacity(mlir.NamedAttribute.named(ctx.mlir_ctx, field.name, attribute));
        }
    }

    metadata_attribute_list.appendSliceAssumeCapacity(&[_]mlir.NamedAttribute{
        .named(ctx.mlir_ctx, "pjrt_api", mlir.integerAttribute(ctx.mlir_ctx, .u64, @as(u64, @bitCast(@intFromPtr(ctx.platform.pjrt_api))))),
        .named(ctx.mlir_ctx, "pjrt_client", mlir.integerAttribute(ctx.mlir_ctx, .u64, @as(u64, @bitCast(@intFromPtr(ctx.platform.pjrt_client))))),
    });

    const backend_config = mlir.dictionaryAttribute(ctx.mlir_ctx, metadata_attribute_list.items);

    const operand_layouts = allocator.alloc([]const usize, input_shapes.len) catch unreachable;
    for (input_shapes, 0..) |shape, i| {
        operand_layouts[i] = allocator.dupe(usize, toUsize(constants.minorToMajor(shape.rank())).constSlice()) catch unreachable;
    }

    const result_layouts = allocator.alloc([]const usize, outputs.len) catch unreachable;
    for (outputs, 0..) |shape, i| {
        result_layouts[i] = allocator.dupe(usize, toUsize(constants.minorToMajor(shape.rank())).constSlice()) catch unreachable;
    }

    const op = dialects.stablehlo.custom_call(
        ctx.mlir_ctx,
        values,
        result_types,
        .{
            .call_target_name = target_name,
            .backend_config = .{ .typed_ffi = backend_config },
            .has_side_effect = opts.has_side_effect,
            .operand_layouts = operand_layouts,
            .result_layouts = result_layouts,
            .output_operand_aliases = opts.output_operand_aliases orelse &.{},
            .additional_attributes = &.{},
        },
        .unknown(ctx.mlir_ctx),
    ).appendTo(ctx.currentScope().block);

    if (ctx.manual_computation_depth > 0 and ctx.partitioning.partitioner == .gspmd) {
        op.setAttributeByName("mhlo.sharding", mlir.stringAttribute(ctx.mlir_ctx, "{manual}"));
    }

    const outputs_ = allocator.alloc(Tensor, outputs.len) catch unreachable;
    for (outputs, 0..) |output_shape, i| {
        outputs_[i] = Tensor._result(output_shape, op.result(i));
    }

    return outputs_;
}

pub fn CustomCall(
    // A, I and O are assumed to be simple struct without any nesting like: { a: zml.Tensor }
    I: type,
    O: type,
    A: type,
    // Receives the custom call buffers for the input and outputs a struct with the same fields as I and O
    // and the provided attributes. So if I is `{a: zml.Tensor}`, the first argument is `{a: CustomCallBuffer}`
    // (TensorToCustomCallBuffer(i), ShapeToCustomCallBuffer(O), A) -> !?*pjrt.ffi.Error
    comptime func: anytype,
    comptime params: struct {
        /// Name of the custom call, must be unique.
        name: [:0]const u8,
        /// Custom call will see the data of a single device. So for a sharded buffer, it will only see a single partition. This applies
        /// for both the input and output buffers. By default, the partitioning algorithm will assume nothing on the custom call input/output,
        /// so everything will be forced as replicated. If do you know / ensure that the custom call is consistent with the provided input/output
        /// sharding, this setting will avoid the shuffling of the buffer's data.
        sharding_aware: bool,
        /// Whether the function has any side-effect, meaning that XLA cannot re-order it/optimize it out. A typical example is print.
        has_side_effect: bool,
        /// Similar to reuseBuffer on tensors, tells XLA that the custom call re-uses an input buffer for its output.
        /// Use the input/output field names to express the mapping: `{ .<out field> = .<in field> }`
        output_operand_aliases: ?CustomCallOutputOperandAliases(I, O) = null,
    },
) type {
    return struct {
        pub const InputTensors = I;
        pub const OutputShapes = O;
        pub const Attributes = A;

        /// Register the custom call so you can use it.
        pub fn register(platform: *const Platform) !void {
            try platform.registerFfi(.{
                .name = params.name,
                .platform_name = null,
                .handler = @This().handler,
                .traits = .{
                    .command_buffer_compatible = true,
                },
            });
        }

        pub fn call(input_tensors: I, output_shapes: O, attributes: A) ShapeToTensor(O) {
            const args: CustomCallArgs(I, O, A) = .{
                .input_tensors = input_tensors,
                .output_shapes = output_shapes,
                .attributes = attributes,
            };
            const opts: CustomCallOptions = .{
                .has_side_effect = params.has_side_effect,
                .output_operand_aliases = comptime customCallOutputOperandAliases(I, O, params.output_operand_aliases),
            };
            if (params.sharding_aware) {
                return shardingAwareTypedCustomCall(
                    I,
                    O,
                    A,
                    params.name,
                    opts,
                    args,
                );
            } else {
                return typedCustomCall(
                    I,
                    O,
                    A,
                    params.name,
                    opts,
                    args,
                );
            }
        }

        fn handler(call_frame: *pjrt.ffi.CallFrame) callconv(.c) ?*pjrt.ffi.Error {
            if (call_frame.registeringHook()) return null;

            const input_buffers: TensorToCustomCallBuffer(I), const output_buffers: ShapeToCustomCallBuffer(O), const attributes: A = customCallArgsFromPjrtCallFrame(I, O, A, call_frame);
            return func(call_frame, input_buffers, output_buffers, attributes) catch |err|
                pjrt.ffi.Error.create(call_frame.api, .unknown, @errorName(err));
        }
    };
}

pub fn CustomCallArgs(I: type, O: type, A: type) type {
    return struct {
        input_tensors: I,
        output_shapes: O,
        attributes: A,
    };
}

pub fn ShapeToTensor(T: type) type {
    return meta.MapRestrict(Shape, Tensor).map(T);
}

pub fn CustomCallOutputOperandAliases(I: type, O: type) type {
    return meta.MapRestrict(Shape, ?std.meta.FieldEnum(I)).map(O);
}

pub fn typedShardedCustomCall(
    I: type,
    O: type,
    A: type,
    comptime target_name: [:0]const u8,
    args: CustomCallArgs(I, O, A),
    comptime opts: CustomCallOptions,
) ShapeToTensor(O) {
    return shardingAwareTypedCustomCall(I, O, A, target_name, opts, args);
}
pub fn shardingAwareTypedCustomCall(
    I: type,
    O: type,
    A: type,
    comptime target_name: [:0]const u8,
    comptime opts: CustomCallOptions,
    args: CustomCallArgs(I, O, A),
) ShapeToTensor(O) {
    // Convert to slices so that manualComputationInternal can inspect the input/output.
    var input_tensors: [@typeInfo(@TypeOf(args.input_tensors)).@"struct".fields.len]Tensor = undefined;
    inline for (@typeInfo(@TypeOf(args.input_tensors)).@"struct".fields, 0..) |field, i| {
        input_tensors[i] = @field(args.input_tensors, field.name);
    }

    var output_shapes: [@typeInfo(@TypeOf(args.output_shapes)).@"struct".fields.len]Shape = undefined;
    inline for (@typeInfo(@TypeOf(args.output_shapes)).@"struct".fields, 0..) |field, i| {
        output_shapes[i] = @field(args.output_shapes, field.name);
    }

    const output_tensors = manualComputationInternal(&input_tensors, &output_shapes, args.attributes, (struct {
        fn body(attributes: A, _: std.mem.Allocator, sharded_input_tensors: []const Tensor, sharded_output_shapes: []const Shape) []const Tensor {
            return typedCustomCall(
                []const Tensor,
                []const Shape,
                A,
                target_name,
                opts,
                .{
                    .input_tensors = sharded_input_tensors,
                    .output_shapes = sharded_output_shapes,
                    .attributes = attributes,
                },
            );
        }
    }).body);

    // Convert the slice back to a struct
    var out: ShapeToTensor(O) = undefined;
    inline for (@typeInfo(@TypeOf(args.output_shapes)).@"struct".fields, 0..) |field, i| {
        @field(out, field.name) = output_tensors[i];
    }
    return out;
}

pub fn typedCustomCall(
    I: type,
    O: type,
    A: type,
    comptime target_name: [:0]const u8,
    comptime opts: CustomCallOptions,
    args: CustomCallArgs(I, O, A),
) ShapeToTensor(O) {
    const ctx = CompilationContext.current();
    const allocator = ctx.arena.allocator();

    stdx.debug.assert(!opts.has_side_effect or ctx.manual_computation_depth > 0, "side-effect customCall '{s}' must be emitted inside manualComputation", .{target_name});

    const input_tensors: []const Tensor = switch (@typeInfo(@TypeOf(args.input_tensors))) {
        .@"struct" => |struct_info| b: {
            var input_tensors: [struct_info.fields.len]Tensor = undefined;
            inline for (struct_info.fields, 0..) |field, i| {
                input_tensors[i] = @field(args.input_tensors, field.name);
            }
            break :b &input_tensors;
        },
        // Extra case to support []const Tensor from shardingAwareTypedCustomCall
        .pointer => |pointer_info| b: {
            if (pointer_info.size != .slice) @compileError("Expected input slice");
            break :b args.input_tensors;
        },
        else => @compileError("Unsupported input type: " ++ @typeName(@TypeOf(args.input_tensors))),
    };

    const output_shapes: []const Shape = switch (@typeInfo(@TypeOf(args.output_shapes))) {
        .@"struct" => |struct_info| b: {
            var output_shapes: [struct_info.fields.len]Shape = undefined;
            inline for (struct_info.fields, 0..) |field, i| {
                output_shapes[i] = @field(args.output_shapes, field.name);
            }
            break :b &output_shapes;
        },
        // Extra case to support []const Shape from shardingAwareTypedCustomCall
        .pointer => |pointer_info| b: {
            if (pointer_info.size != .slice) @compileError("Expected input slice");
            break :b args.output_shapes;
        },
        else => @compileError("Unsupported output type: " ++ @typeName(@TypeOf(args.output_shapes))),
    };

    const metadata_attributes =
        switch (@typeInfo(@TypeOf(args.attributes))) {
            .@"struct" => |struct_info| b: {
                var mlir_attributes_array: [struct_info.fields.len]mlir.NamedAttribute = undefined;
                inline for (struct_info.fields, 0..) |field, i| {
                    const attribute: *const mlir.Attribute = switch (@typeInfo(field.type)) {
                        .comptime_int => mlir.integerAttribute(ctx.mlir_ctx, .u64, @as(u64, @field(args.attributes, field.name))),
                        .@"enum" => |enum_field| switch (@typeInfo(enum_field.tag_type)) {
                            .int => |int_tag| switch (int_tag.signedness) {
                                .signed => switch (int_tag.bits) {
                                    8 => mlir.integerAttribute(ctx.mlir_ctx, .i8, @intFromEnum(@field(args.attributes, field.name))),
                                    16 => mlir.integerAttribute(ctx.mlir_ctx, .i16, @intFromEnum(@field(args.attributes, field.name))),
                                    32 => mlir.integerAttribute(ctx.mlir_ctx, .i32, @intFromEnum(@field(args.attributes, field.name))),
                                    64 => mlir.integerAttribute(ctx.mlir_ctx, .i64, @intFromEnum(@field(args.attributes, field.name))),
                                    else => @panic("Unsupported DataType"),
                                },
                                .unsigned => switch (int_tag.bits) {
                                    8 => mlir.integerAttribute(ctx.mlir_ctx, .u8, @intFromEnum(@field(args.attributes, field.name))),
                                    16 => mlir.integerAttribute(ctx.mlir_ctx, .u16, @intFromEnum(@field(args.attributes, field.name))),
                                    32 => mlir.integerAttribute(ctx.mlir_ctx, .u32, @intFromEnum(@field(args.attributes, field.name))),
                                    64 => mlir.integerAttribute(ctx.mlir_ctx, .u64, @intFromEnum(@field(args.attributes, field.name))),
                                    else => @panic("Unsupported DataType"),
                                },
                            },
                            else => @compileError("Unsupported tag type for enum metadata: " ++ @typeName(enum_field.tag_type)),
                        },
                        .int => |int_field| switch (int_field.signedness) {
                            .signed => switch (int_field.bits) {
                                8 => mlir.integerAttribute(ctx.mlir_ctx, .i8, @field(args.attributes, field.name)),
                                16 => mlir.integerAttribute(ctx.mlir_ctx, .i16, @field(args.attributes, field.name)),
                                32 => mlir.integerAttribute(ctx.mlir_ctx, .i32, @field(args.attributes, field.name)),
                                64 => mlir.integerAttribute(ctx.mlir_ctx, .i64, @field(args.attributes, field.name)),
                                else => @panic("Unsupported DataType"),
                            },
                            .unsigned => switch (int_field.bits) {
                                8 => mlir.integerAttribute(ctx.mlir_ctx, .u8, @field(args.attributes, field.name)),
                                16 => mlir.integerAttribute(ctx.mlir_ctx, .u16, @field(args.attributes, field.name)),
                                32 => mlir.integerAttribute(ctx.mlir_ctx, .u32, @field(args.attributes, field.name)),
                                64 => mlir.integerAttribute(ctx.mlir_ctx, .u64, @field(args.attributes, field.name)),
                                else => @panic("Unsupported DataType"),
                            },
                        },
                        .float => |float_field| switch (float_field.bits) {
                            16 => mlir.floatAttribute(ctx.mlir_ctx, .f16, @as(f64, @field(args.attributes, field.name))),
                            32 => mlir.floatAttribute(ctx.mlir_ctx, .f32, @as(f64, @field(args.attributes, field.name))),
                            64 => mlir.floatAttribute(ctx.mlir_ctx, .f64, @as(f64, @field(args.attributes, field.name))),
                            else => @panic("Unsupported DataType"),
                        },
                        .bool => mlir.boolAttribute(ctx.mlir_ctx, @field(args.attributes, field.name)),
                        .pointer => |pointer_info| switch (pointer_info.size) {
                            .slice => if (pointer_info.child == u8)
                                mlir.stringAttribute(ctx.mlir_ctx, @field(args.attributes, field.name))
                            else
                                @panic("Unsupported pointer type in metadata"),
                            else => @panic("Unsupported pointer type in metadata"),
                        },
                        else => @compileError("Unsupported metadata type: " ++ @typeName(field.type)),
                    };
                    mlir_attributes_array[i] = mlir.NamedAttribute.named(ctx.mlir_ctx, field.name, attribute);
                }

                break :b mlir_attributes_array;
            },
            .void => [_]mlir.NamedAttribute{},
            else => @compileError("Unsupported type: " ++ @typeName(@TypeOf(args.attributes))),
        };

    const values = allocator.alloc(*const mlir.Value, input_tensors.len) catch unreachable;
    const input_shapes = allocator.alloc(Shape, input_tensors.len) catch unreachable;
    for (input_tensors, 0..) |input, i| {
        values[i] = input.value();
        input_shapes[i] = input.shape();
    }

    const result_types = allocator.alloc(*const mlir.Type, output_shapes.len) catch unreachable;
    for (output_shapes, 0..) |shape, i| {
        result_types[i] = mlir.rankedTensorType(shape.dims(), mlirx.Type.fromDType(ctx.mlir_ctx, shape.dtype()));
    }

    const backend_config = mlir.dictionaryAttribute(ctx.mlir_ctx, &(metadata_attributes ++ [_]mlir.NamedAttribute{
        .named(ctx.mlir_ctx, "pjrt_api", mlir.integerAttribute(ctx.mlir_ctx, .u64, @as(u64, @bitCast(@intFromPtr(ctx.platform.pjrt_api))))),
        .named(ctx.mlir_ctx, "pjrt_client", mlir.integerAttribute(ctx.mlir_ctx, .u64, @as(u64, @bitCast(@intFromPtr(ctx.platform.pjrt_client))))),
    }));

    const operand_layouts = allocator.alloc([]const usize, input_shapes.len) catch unreachable;
    for (input_shapes, 0..) |shape, i| {
        operand_layouts[i] = allocator.dupe(usize, toUsize(constants.minorToMajor(shape.rank())).constSlice()) catch unreachable;
    }

    const result_layouts = allocator.alloc([]const usize, output_shapes.len) catch unreachable;
    for (output_shapes, 0..) |shape, i| {
        result_layouts[i] = allocator.dupe(usize, toUsize(constants.minorToMajor(shape.rank())).constSlice()) catch unreachable;
    }

    const op = dialects.stablehlo.custom_call(
        ctx.mlir_ctx,
        values,
        result_types,
        .{
            .call_target_name = target_name,
            .backend_config = .{ .typed_ffi = backend_config },
            .has_side_effect = opts.has_side_effect,
            .operand_layouts = operand_layouts,
            .result_layouts = result_layouts,
            .output_operand_aliases = opts.output_operand_aliases orelse &.{},
            .additional_attributes = &.{},
        },
        .unknown(ctx.mlir_ctx),
    ).appendTo(ctx.currentScope().block);

    if (ctx.manual_computation_depth > 0 and ctx.partitioning.partitioner == .gspmd) {
        op.setAttributeByName("mhlo.sharding", mlir.stringAttribute(ctx.mlir_ctx, "{manual}"));
    }

    return typedCustomCallOutputs(O, allocator, output_shapes, op);
}

fn typedCustomCallOutputs(
    comptime O: type,
    allocator: std.mem.Allocator,
    output_shapes: []const Shape,
    op: *mlir.Operation,
) ShapeToTensor(O) {
    return switch (@typeInfo(O)) {
        .@"struct" => |struct_info| b: {
            var out: ShapeToTensor(O) = undefined;
            inline for (struct_info.fields, 0..) |field, i| {
                @field(out, field.name) = Tensor._result(output_shapes[i], op.result(i));
            }
            break :b out;
        },
        .pointer => |pointer_info| b: {
            if (pointer_info.size != .slice or pointer_info.child != Shape) {
                @compileError("Unsupported output type: " ++ @typeName(O));
            }

            const outputs_ = allocator.alloc(Tensor, output_shapes.len) catch unreachable;
            for (output_shapes, 0..) |output_shape, i| {
                outputs_[i] = Tensor._result(output_shape, op.result(i));
            }
            break :b if (pointer_info.is_const)
                outputs_
            else
                @constCast(outputs_);
        },
        else => @compileError("Unsupported output type: " ++ @typeName(O)),
    };
}

fn customCallArgsFromPjrtCallFrame(I: type, O: type, A: type, call_frame: *pjrt.ffi.CallFrame) struct {
    TensorToCustomCallBuffer(I),
    ShapeToCustomCallBuffer(O),
    A,
} {
    var input: TensorToCustomCallBuffer(I) = undefined;
    inline for (@typeInfo(I).@"struct".fields, 0..) |field, i| {
        const buf = call_frame.args.buffers()[i];
        @field(input, field.name) = .fromPjrt(buf);
    }

    var output: ShapeToCustomCallBuffer(O) = undefined;
    inline for (@typeInfo(O).@"struct".fields, 0..) |field, i| {
        const shape_buf = call_frame.results.buffers()[i];
        @field(output, field.name) = .fromPjrt(shape_buf);
    }

    const attributes: A = switch (@typeInfo(A)) {
        .void => {},
        .@"struct" => blk: {
            var attrs: A = undefined;
            inline for (@typeInfo(A).@"struct".fields) |field| {
                // TODO: Use getByIndex instead?
                const attribute = call_frame.attrs.getByName(.scalar, field.name) orelse
                    @panic("Attribute not found: " ++ field.name);
                @field(attrs, field.name) = attribute.get(field.type);
            }
            break :blk attrs;
        },
        else => @compileError("Unsupported attributes type: " ++ @typeName(A)),
    };

    return .{ input, output, attributes };
}

test customCall {
    const zml = @import("zml.zig");
    const platform = zml.testing.env();

    var comp = zml.module.CompilationContext.init(std.testing.allocator, std.testing.io, platform, .{ .shardings = &.{zml.testing.replicatedSharding()} });
    defer comp.deinit();
    comp.activate();
    defer comp.deactivate();

    const block = mlir.Block.init(&.{}, &.{});
    comp.pushBlock(block);
    defer comp.popBlock();

    const shape = zml.Shape.init(.{128}, .bf16).withPartitioning(.{ ._0 = .x });
    const input = Tensor.constant(zml.DataType.bf16.constant(0)).broad(shape);
    const output = customCall("my_custom_call", .{input}, .{zml.Shape.init(.{128}, .bf16)}, .{}, .{
        .has_side_effect = false,
        .output_operand_aliases = &.{.{ .output_index = 0, .operand_index = 0 }},
    });

    try zml.testing.expectEqualShapes(input.shape(), output.shape());
}

test "CustomCallOutputOperandAliases maps named fields to indices" {
    const Input = struct {
        q: Tensor,
        k: Tensor,
        v: Tensor,
    };
    const Output = struct {
        attn: Shape,
        scratch: Shape,
    };

    const aliases = comptime customCallOutputOperandAliases(Input, Output, .{
        .attn = .q,
        .scratch = .v,
    }).?;

    try std.testing.expectEqual(@as(usize, 2), aliases.len);
    try std.testing.expectEqualDeep(
        dialects.stablehlo.CustomCallOpts.OutputOperandAlias{ .output_index = 0, .operand_index = 0 },
        aliases[0],
    );
    try std.testing.expectEqualDeep(
        dialects.stablehlo.CustomCallOpts.OutputOperandAlias{ .output_index = 1, .operand_index = 2 },
        aliases[1],
    );
}

fn toUsize(values: anytype) stdx.BoundedArray(usize, constants.MAX_RANK) {
    var res: stdx.BoundedArray(usize, constants.MAX_RANK) = .{};
    for (values) |val| res.appendAssumeCapacity(@intCast(val));
    return res;
}

pub fn scaled_dot(lhs: Tensor, rhs: Tensor, lhs_scale: Tensor, rhs_scale: Tensor, args: anytype) Tensor {
    const real_lhs = lhs.convert(.bf16).mul(lhs_scale.convert(.bf16));
    const real_rhs = rhs.convert(.bf16).mul(rhs_scale.convert(.bf16));

    return real_lhs.dot(real_rhs, args).convert(.bf16);
}
