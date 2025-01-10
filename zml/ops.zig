const std = @import("std");
const stdx = @import("stdx");

const buffer = @import("buffer.zig");
const helpers = @import("helpers.zig");
const meta = @import("meta.zig");
const mlir = @import("mlir.zig");
const module = @import("module.zig");

const Buffer = buffer.Buffer;
const CompilationContext = module.CompilationContext;
const Context = @import("context.zig").Context;
const Data = @import("dtype.zig").Data;
const DataType = @import("dtype.zig").DataType;
const EnumLiteral = @TypeOf(.enum_literal);
const HostBuffer = @import("hostbuffer.zig").HostBuffer;
const Shape = @import("shape.zig").Shape;
const Tensor = @import("tensor.zig").Tensor;
const _collectAxes = @import("tensor.zig")._collectAxes;

const dialect = struct {
    const stablehlo = @import("mlir/dialects").stablehlo;
};

const assert = std.debug.assert;
const log = std.log.scoped(.@"zml/tensor");

test {
    std.testing.refAllDecls(@This());
}

/// Generate an MLIR call to the given member function with the given tensors.
pub fn call(self: anytype, comptime func: stdx.meta.DeclEnum(@TypeOf(self)), args: anytype) @TypeOf(@call(.auto, @field(stdx.meta.UnwrapPtr(@TypeOf(self)), @tagName(func)), .{self} ++ args)) {
    const ctx = CompilationContext.current();
    const name = @typeName(@TypeOf(self)) ++ "." ++ @tagName(func);
    const actual_fn = @field(@TypeOf(self), @tagName(func));
    return ctx.callFunc(name, actual_fn, .{self} ++ args);
}

pub fn while_(
    comptime cond_fn: anytype,
    comptime body_fn: anytype,
    blkctx: BlockSign(body_fn).BlkCtx,
    inputs: BlockSign(body_fn).Args,
) BlockSign(body_fn).Return {
    const CondS = comptime BlockSign(cond_fn);
    const BodyS = comptime BlockSign(body_fn);
    if (CondS.Args != BodyS.Args) {
        @compileError("cond_fn and body_fn signatures don't match ! " ++ @typeName(@TypeOf(cond_fn)) ++ " and " ++ @typeName(@TypeOf(body_fn)));
    }
    const ctx = CompilationContext.current();
    const cond_block, _ = ctx.makeBlock(.open, CondS, &cond_fn, blkctx, inputs);
    const body_block, const body_res = ctx.makeBlock(.open, BodyS, &body_fn, blkctx, inputs);
    var input_values: [BodyS.nIn]mlir.Value = undefined;
    ctx.extractValues(&inputs, &input_values);

    const loc = ctx.mlirCtx().location(@src());

    const op = mlir.Operation.make(ctx.mlirCtx(), "stablehlo.while", .{
        .variadic_operands = &.{&input_values},
        .result_type_inference = true,
        .blocks = &.{ cond_block, body_block },
        // We can't verify right away, cause the weights captured by the while haven't been added yet.
        .verify = false,
        .location = loc,
    });

    return fromMlirOperationWithTags(op, body_res);
}

test "simple while" {
    const CountInts = struct {
        step: Tensor,
        end: Tensor,
        const CountInts = @This();

        pub fn hasNext(self: CountInts, i: Tensor, sum: Tensor) Tensor {
            _ = sum;
            return i.cmp(.LT, self.end);
        }

        pub fn next(self: CountInts, i: Tensor, sum: Tensor) [2]Tensor {
            const r1 = i.add(self.step);
            const r2 = sum.add(i);
            return .{ r1, r2 };
        }

        pub fn forward(self: CountInts, init_i: Tensor, init_sum: Tensor) [2]Tensor {
            const x = init_i.scale(2);
            return while_(CountInts.hasNext, CountInts.next, self, .{ x, init_sum });
        }

        pub fn zigForward(step: i64, end: i64, init_i: i64, init_sum: i64) [2]i64 {
            const x = init_i * 2;
            var i = x;
            var sum = init_sum;
            while (i < end) {
                const r1 = i + step;
                const r2 = sum + i;
                i, sum = .{ r1, r2 };
            }
            return .{ i, sum };
        }
    };
    const zml = @import("zml.zig");
    const platform = zml.testing.env();

    const init_i = try zml.Buffer.fromSlice(platform, .{}, &[_]i64{0});
    const init_sum = try zml.Buffer.fromSlice(platform, .{}, &[_]i64{0});
    const counter = .{
        .step = try zml.Buffer.fromSlice(platform, .{}, &[_]i64{1}),
        .end = try zml.Buffer.fromSlice(platform, .{}, &[_]i64{10}),
    };
    const res0, const res1 = try zml.testing.compileAndCall(platform, CountInts.forward, .{ counter, init_i, init_sum });
    const last_i = try res0.getValue(i64);
    const sum = try res1.getValue(i64);

    try std.testing.expectEqual(10, last_i);
    try std.testing.expectEqual(45, sum);

    try std.testing.expectEqual(.{ 10, 45 }, CountInts.zigForward(1, 10, 0, 0));
}

pub fn reduce(
    comptime body_fn: anytype,
    inputs: stdx.meta.FnParam(body_fn, 0),
    inits: stdx.meta.FnParam(body_fn, 0),
    axes: []const i64,
) BlockSignNoCtx(body_fn).Return {
    // TODO: actualAxes
    const BodyS = comptime BlockSignNoCtx(body_fn);
    comptime {
        if (BodyS.Return != @TypeOf(inputs)) @compileError("reduce body function need to have the following signature `fn (left: T, right: T) T`, got: " ++ @typeName(body_fn));
    }
    const ctx = CompilationContext.current();
    const N = comptime @divExact(BodyS.nIn, 2);
    var input_values: [N]mlir.Value = undefined;
    ctx.extractValues(&inputs, &input_values);
    var init_values: [N]mlir.Value = undefined;
    ctx.extractValues(&inits, &init_values);

    const body_block, _ = ctx.makeBlock(.hermetic, BodyS, &body_fn, {}, .{ inits, inits });

    const loc = ctx.mlirCtx().location(@src());

    const op = mlir.Operation.make(ctx.mlirCtx(), "stablehlo.reduce", .{
        .variadic_operands = &.{ &input_values, &init_values },
        .result_type_inference = true,
        .blocks = &.{body_block},
        .attributes = &.{
            .{ "dimensions", mlir.DenseArrayAttribute(.i64).init(ctx.mlirCtx(), axes).as(mlir.Attribute).? },
        },
        // We can't verify right away, cause the weights captured by the reduce haven't been added yet.
        .verify = false,
        .location = loc,
    });

    // `stablehlo.reduce` drops axes. We want to avoid that to propagate tags.
    // So we need to broadcast the output of `stablehlo.reduce` to the input shapes.
    // To that order, we initialize `result` to `inputs`, then we use stdx.meta.visit,
    // to find the correct mlir.Value, but we first broadcast before creating the final
    // Tensor struct.
    var broadcasting_axes: std.BoundedArray(i64, Tensor.MAX_RANK) = .{};
    for (0..Tensor.MAX_RANK) |i| {
        if (std.mem.indexOfScalar(i64, axes, @intCast(i)) == null) {
            broadcasting_axes.append(@intCast(i)) catch unreachable;
        }
    }
    var res: BodyS.Return = inputs;
    const LocalContext = struct {
        axes: []const i64,
        broadcasting_axes: []const i64,
        n_reduced: u8,
        op: mlir.Operation,
        loc: mlir.Location,
        index: usize = 0,
    };
    var local_context = LocalContext{
        .axes = axes,
        .broadcasting_axes = broadcasting_axes.constSlice(),
        .n_reduced = @intCast(axes.len),
        .op = op,
        .loc = loc,
    };
    meta.visit((struct {
        fn cb(inner_ctx: *LocalContext, tensor: *Tensor) void {
            const val = inner_ctx.op.result(inner_ctx.index);
            // compute the target reduced shape
            var reduced_shape = tensor.shape();
            for (inner_ctx.axes) |a| {
                reduced_shape = reduced_shape.setDim(a, 1);
            }

            const mlir_ctx = CompilationContext.current().mlirCtx();

            const broad_val = dialect.stablehlo.broadcast_in_dim(
                mlir_ctx,
                val,
                inner_ctx.broadcasting_axes[0 .. tensor.rank() - inner_ctx.n_reduced],
                mlir.ext.RankedTensorType.fromShape(mlir_ctx, reduced_shape).as(mlir.Type).?,
                inner_ctx.loc,
            );
            tensor.* = Tensor._result(reduced_shape, broad_val.result(0));
            inner_ctx.index += 1;
        }
    }).cb, &local_context, &res);
    assert(local_context.index == op.numResults());
    return res;
}

pub const ReduceWindowOpts = struct {
    // TODO replace with Shape
    window_dimensions: []const i64,
    window_strides: []const i64,
    base_dilations: []const i64,
    window_dilations: []const i64,
    padding: []const [2]i64,
};

pub fn reduceWindow(
    comptime body_fn: anytype,
    inputs: stdx.meta.FnParam(body_fn, 0),
    inits: stdx.meta.FnParam(body_fn, 0),
    opts: ReduceWindowOpts,
) stdx.meta.FnResult(body_fn) {
    const BodyS = comptime BlockSignNoCtx(body_fn);
    comptime {
        if (BodyS.Return != @TypeOf(inputs)) @compileError("reduce body function need to have the following signature `fn (left: T, right: T) T`, got: " ++ @typeName(body_fn));
    }
    const ctx = CompilationContext.current();
    const body_block, _ = ctx.makeBlock(.hermetic, BodyS, &body_fn, {}, .{ inits, inits });
    const N = comptime @divExact(BodyS.nIn, 2);
    var input_values: [N]mlir.Value = undefined;
    ctx.extractValues(&inputs, &input_values);
    var init_values: [N]mlir.Value = undefined;
    ctx.extractValues(&inits, &init_values);

    const loc = ctx.mlirCtx().location(@src());

    const pad_shape = mlir.RankedTensorType.init(
        &.{ @intCast(opts.padding.len), 2 },
        mlir.ext.Type.fromDType(ctx.mlirCtx(), .i64),
    ).as(mlir.Type).?;
    const op = mlir.Operation.make(ctx.mlirCtx(), "stablehlo.reduce_window", .{
        .variadic_operands = &.{ input_values[0..], init_values[0..] },
        .result_type_inference = true,
        .blocks = &.{body_block},
        .attributes = &.{
            .{ "window_dimensions", mlir.DenseArrayAttribute(.i64).init(ctx.mlirCtx(), opts.window_dimensions).as(mlir.Attribute).? },
            .{ "window_strides", mlir.DenseArrayAttribute(.i64).init(ctx.mlirCtx(), opts.window_strides).as(mlir.Attribute).? },
            .{ "base_dilations", mlir.DenseArrayAttribute(.i64).init(ctx.mlirCtx(), opts.base_dilations).as(mlir.Attribute).? },
            .{ "window_dilations", mlir.DenseArrayAttribute(.i64).init(ctx.mlirCtx(), opts.window_dilations).as(mlir.Attribute).? },
            .{ "padding", mlir.DenseIntOrFPElementsAttribute(.i64).init(pad_shape, std.mem.sliceAsBytes(opts.padding)).as(mlir.Attribute).? },
        },
        .location = loc,
    });

    return fromMlirOperationWithTags(op, inputs);
}

/// Runs a given function for several steps, and returns a stack of each step output.
/// The step outputs will be stacked along the first axis.
pub fn for_(comptime func: anytype, blk_ctx: BlockSign(func).BlkCtx, num_steps_: anytype) BlockSign(func).Return {
    const num_steps: u32, const step_tag = blk: {
        const dims, const tags = Shape.parseDimensions(num_steps_);
        stdx.debug.assert(dims.len == 1, "zml.for_ only supports one num_step, Received: {any}", .{num_steps_});
        break :blk .{ @intCast(dims.get(0)), tags.get(0) };
    };
    const S = comptime BlockSign(func);

    const ForBlk = struct {
        blk_ctx: S.BlkCtx,
        step_tag: Shape.Tag,
        num_steps: u32,
        const Self = @This();

        fn next(self: Self, res: S.Return, idx: Tensor) struct { S.Return, Tensor } {
            const step_res = @call(.auto, func, .{ self.blk_ctx, idx });
            var buf: [@sizeOf(S.Return) * 2]u8 = undefined;
            var fba = std.heap.FixedBufferAllocator.init(&buf);
            return .{
                meta.zip(updateResBuffer, fba.allocator(), &[_]S.Return{ res, step_res }, .{idx}) catch unreachable,
                idx.addConstant(1),
            };
        }

        fn done(self: Self, res: S.Return, idx: Tensor) Tensor {
            _ = res;
            return idx.cmp(.LT, Tensor.scalar(self.num_steps, idx.dtype()));
        }

        fn updateResBuffer(inputs: []const Tensor, idx: Tensor) Tensor {
            stdx.debug.internalAssert(inputs.len == 2, "too many tensors", .{});
            const res, const step_res = inputs[0..2].*;
            return res.dynamicUpdateSlice1d(step_res.insertAxes(0, .{._}), 0, idx);
        }

        /// Prepare buffer to store all results steps.
        fn prep(self: Self, first_step: Tensor) Tensor {
            const shape = first_step.shape().insertTag(0, 1, self.step_tag);
            // Reuse the first step Tensor.
            // TODO: this is needed because of https://github.com/zml/zml/issues/97
            // Normally I'd rather NOT reuse first_step to streamline the stablehlo IR.
            return first_step.reshape(shape).pad(0, .{ ._0 = .{ .high = self.num_steps - 1 } });
        }

        fn wrapFirstStep(tag_: @TypeOf(step_tag), x: Tensor) Tensor {
            var shape = x.shape();
            shape._dims.insert(0, 1) catch unreachable;
            shape._tags.insert(0, tag_) catch unreachable;
            return x.reshape(shape);
        }
    };

    // Compute first step to infer the output shapes.
    // Normally this shouldn't be reused apart from the unrolled cases,
    // but because of https://github.com/zml/zml/issues/97 we also reuse it to start the while_ loop.
    const first_step = @call(.auto, func, .{ blk_ctx, Tensor.scalar(0, .i32) });
    log.debug("for_ first_step: {}", .{first_step});
    const allocator = CompilationContext.current().allocator();
    // Optimize for small num reps
    if (num_steps == 1) {
        var res = first_step;
        meta.mapAlloc(ForBlk.wrapFirstStep, allocator, step_tag, first_step, &res) catch unreachable;
        return res;
    }

    if (num_steps <= 4) {
        var steps: [4]S.Return = undefined;
        steps[0] = first_step;
        for (1..num_steps) |i| {
            steps[i] = @call(.auto, func, .{ blk_ctx, Tensor.scalar(i, .i32) });
        }

        const res = meta.zip(Tensor.stack, allocator, steps[0..num_steps], .{ 0, step_tag }) catch unreachable;
        return res;
    }

    const for_blk: ForBlk = .{ .blk_ctx = blk_ctx, .step_tag = step_tag, .num_steps = num_steps };
    var result_buffers: @TypeOf(first_step) = undefined;
    try meta.mapAlloc(ForBlk.prep, allocator, for_blk, first_step, &result_buffers);

    return while_(
        ForBlk.done,
        ForBlk.next,
        for_blk,
        .{
            result_buffers,
            // First step is already done
            Tensor.scalar(1, .i32),
        },
    )[0];
}

test for_ {
    const Squares = struct {
        const Squares = @This();

        pub fn sq(self: Squares, i: Tensor) Tensor {
            _ = self;
            const f = i.convert(.f32);
            return f.mul(f);
        }

        pub fn forward(num_steps: u63) Tensor {
            return for_(Squares.sq, .{}, .{num_steps});
        }
    };
    const zml = @import("zml.zig");
    const platform = zml.testing.env();

    // Just one baby step
    {
        const squares = try zml.testing.compileAndCall(platform, Squares.forward, .{1});
        try zml.testing.expectEqualShapes(Shape.init(.{1}, .f32), squares.shape());
        try std.testing.expectEqual(0, squares.getValue(f32));
    }
    // Wow 4 in rows !
    {
        const squares = try zml.testing.compileAndCall(platform, Squares.forward, .{4});
        try zml.testing.expectEqualShapes(Shape.init(.{4}, .f32), squares.shape());
        try std.testing.expectEqual([_]f32{ 0, 1, 4, 9 }, try squares.getValue([4]f32));
    }
    // AGI is coming, computing 10 squares as it's nothing.
    {
        const squares = try zml.testing.compileAndCall(platform, Squares.forward, .{10});
        try zml.testing.expectEqualShapes(Shape.init(.{10}, .f32), squares.shape());
        try std.testing.expectEqual(
            [_]f32{ 0, 1, 4, 9, 16, 25, 36, 49, 64, 81 },
            try squares.getValue([10]f32),
        );
    }
}

test "nested for" {
    const OuterProd = struct {
        const OuterProd = @This();

        x: Tensor,
        x_row: Tensor,

        pub fn forward(x: Tensor) Tensor {
            return for_(OuterProd.scanRow, x, .{x.dim(0)});
        }

        pub fn scanRow(x: Tensor, i: Tensor) Tensor {
            const row = x.dynamicSlice(.{.{ .start = i, .len = 1 }});
            return for_(OuterProd.scanCol, .{ .x = x, .x_row = row }, .{x.dim(0)});
        }

        pub fn scanCol(self: OuterProd, j: Tensor) Tensor {
            const col = self.x.dynamicSlice(.{.{ .start = j, .len = 1 }});
            return self.x_row.mul(col);
        }
    };

    const zml = @import("zml.zig");
    const platform = zml.testing.env();

    // 5 to prevent inlining
    const x = try zml.Buffer.fromArray(platform, [5]f32{ 0, 1.0, -1.0, 2.0, -2.0 });
    const outer_prod = try zml.testing.compileAndCall(platform, OuterProd.forward, .{x});
    const expected: [5][5]f32 = .{
        .{ 0, 0, 0, 0, 0 },
        .{ 0, 1.0, -1.0, 2.0, -2.0 },
        .{ 0, -1.0, 1.0, -2.0, 2.0 },
        .{ 0, 2.0, -2.0, 4.0, -4.0 },
        .{ 0, -2.0, 2.0, -4.0, 4.0 },
    };
    try std.testing.expectEqual(expected, outer_prod.getValue(@TypeOf(expected)));
}

pub fn if_2(pred: Tensor, comptime Closure: type, blkctx: BlockSignNoArgs(@field(Closure, "then")).BlkCtx) BlockSignNoArgs(@field(Closure, "then")).Return {
    return if_(pred, @field(Closure, "then"), @field(Closure, "else_"), blkctx);
}

pub fn if_(
    pred: Tensor,
    comptime true_branch_fn: anytype,
    comptime false_branch_fn: anytype,
    blkctx: BlockSignNoArgs(true_branch_fn).BlkCtx,
) BlockSignNoArgs(true_branch_fn).Return {
    const TrueBlockSignature = comptime BlockSignNoArgs(true_branch_fn);
    const FalseBlockSignature = comptime BlockSignNoArgs(false_branch_fn);
    if (TrueBlockSignature.Return != FalseBlockSignature.Return) {
        @compileError("true_branch_fn and false_branch_fn return types don't match ! " ++ @typeName(TrueBlockSignature.Return) ++ " and " ++ @typeName(FalseBlockSignature.Return));
    }
    const ctx = CompilationContext.current();
    const true_branch_block, const true_branch_res = ctx.makeBlock(.open, TrueBlockSignature, &true_branch_fn, blkctx, {});
    const false_branch_block, const false_branch_res = ctx.makeBlock(.open, TrueBlockSignature, &false_branch_fn, blkctx, {});
    stdx.debug.assert(false_branch_res.shape().eqlWithTags(true_branch_res.shape()), "zml.ops.if_ expects true and false branch to produce outputs of the same shape, but it produced true={} and false={}", .{ true_branch_res, false_branch_res });

    const loc = ctx.mlirCtx().location(@src());
    const op = mlir.Operation.make(ctx.mlirCtx(), "stablehlo.if", .{
        .operands = &.{pred.value()},
        .result_type_inference = true,
        .blocks = &.{ true_branch_block, false_branch_block },
        // We can't verify right away, cause the weights captured by the if haven't been added yet.
        .verify = false,
        .location = loc,
    });

    return fromMlirOperationWithTags(op, true_branch_res);
}

test "if" {
    const zml = @import("zml.zig");
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;

    const IfMod = struct {
        pub fn forward(pred: Tensor, a: Tensor, b: Tensor) Tensor {
            const result = if_(pred.convert(.bool), condTrue, condFalse, .{ a, b });
            return result;
        }

        pub fn condTrue(a: Tensor, b: Tensor) Tensor {
            return a.matmul(b);
        }

        pub fn condFalse(a: Tensor, b: Tensor) Tensor {
            return b.matmul(a);
        }
    };

    {
        const pred = Shape.init(.{}, .i32);
        const a = Shape.init(.{ 4, 4 }, .f32);
        const b = Shape.init(.{ 4, 4 }, .f32);
        const mod = try zml.compileFn(allocator, IfMod.forward, .{ pred, a, b }, platform);
        defer mod.deinit();
    }
}

pub fn sort(
    comptime comp_fn: anytype,
    blkctx: BlockSign(comp_fn).BlkCtx,
    inputs: [@divExact(BlockSign(comp_fn).nIn, 2)]Tensor,
    dimension: i64,
    is_stable: bool,
) [@divExact(BlockSign(comp_fn).nIn, 2)]Tensor {
    const BodyS = comptime BlockSign(comp_fn);
    var inits: BlockSign(comp_fn).Args = undefined;
    inline for (0..@divExact(BlockSign(comp_fn).nIn, 2)) |i| {
        const arg_shape = Shape.init(.{}, inputs[i].dtype());
        // Note: the id doesn't matter cause makeBlock will correctly fill it.
        inits[i * 2] = Tensor{ ._shape = arg_shape, ._id = undefined, ._donation = .no_buffer };
        inits[i * 2 + 1] = Tensor{ ._shape = arg_shape, ._id = undefined, ._donation = .no_buffer };
    }
    const ctx = CompilationContext.current();
    const block, _ = ctx.makeBlock(.hermetic, BodyS, &comp_fn, blkctx, inits);
    var input_values: [@divExact(BodyS.nIn, 2)]mlir.Value = undefined;
    ctx.extractValues(&inputs, &input_values);

    const loc = ctx.mlirCtx().location(@src()).namedFmt(ctx.mlirCtx(), "sort(dimension={d}, is_stable={})", .{ dimension, is_stable });

    const op = mlir.Operation.make(ctx.mlirCtx(), "stablehlo.sort", .{
        .variadic_operands = &.{&input_values},
        .result_type_inference = true,
        .blocks = &.{block},
        .attributes = &.{
            .{ "dimension", mlir.IntegerAttribute(.i64).init(ctx.mlirCtx(), dimension).as(mlir.Attribute).? },
            .{ "is_stable", mlir.BoolAttribute.init(ctx.mlirCtx(), is_stable).as(mlir.Attribute).? },
        },
        .location = loc,
    });

    var res: [@divExact(BlockSign(comp_fn).nIn, 2)]Tensor = undefined;
    inline for (0..@divExact(BlockSign(comp_fn).nIn, 2)) |i| {
        res[i] = Tensor._result(inputs[i].shape(), op.result(i));
    }
    return res;
}

pub const BlockSignature = struct {
    Fn: type,
    BlkCtx: type,
    Args: type,
    FullArgs: type,
    Return: type,
    nIn: usize,
    nOut: usize,

    pub inline fn blkArgs(self: BlockSignature, blk_ctx: self.BlkCtx, args: self.Args) self.FullArgs {
        if (self.BlkCtx == void) return args;
        if (self.Args == void) return blk_ctx;
        return .{blk_ctx} ++ args;
    }
};

const BlockType = enum { default, no_ctx, no_args };

pub inline fn BlockSign(comptime func: anytype) BlockSignature {
    return _BlockSign(func, .default);
}

pub inline fn BlockSignNoCtx(comptime func: anytype) BlockSignature {
    return _BlockSign(func, .no_ctx);
}

pub inline fn BlockSignNoArgs(comptime func: anytype) BlockSignature {
    return _BlockSign(func, .no_args);
}

pub fn fnInfo(comptime func: anytype) std.builtin.Type.Fn {
    if (@TypeOf(func) == type) {
        if (@typeInfo(func) == .Struct and @hasDecl(func, "forward")) {
            return fnInfo(func.forward);
        }
        @compileError("Given type doesn't have a forward function: " ++ @typeName(func));
    }
    const type_info = @typeInfo(@TypeOf(func));
    const err_msg = "`func` must be a function and return one or more `Tensor`. Got: ";
    if (type_info != .Fn or type_info.Fn.return_type == null) {
        @compileError(err_msg ++ @typeName(@TypeOf(func)));
    }
    return type_info.Fn;
}

fn _BlockSign(comptime func: anytype, blk_type: BlockType) BlockSignature {
    const fn_info = fnInfo(func);
    const err_msg = "`func` must be a function and return one or more `Tensor`. Got: ";

    var full_args: [fn_info.params.len]type = undefined;
    const arg_start = switch (blk_type) {
        .default => 1,
        .no_ctx => 0,
        .no_args => fn_info.params.len,
    };
    var n_tensors: usize = 0;
    // var n_inner_tensors: usize = 0;
    inline for (fn_info.params, 0..) |arg, i| {
        const ArgType = if (arg.type) |T| T else @compileError(err_msg ++ @typeName(@TypeOf(func)));
        full_args[i] = ArgType;
        if (i >= arg_start) {
            n_tensors += staticCountTensors(ArgType) orelse @compileError("Can't use " ++ @typeName(ArgType) ++ " in an MLIR function, because it has a variable number of tensors");
        }
    }
    const FullArgs = std.meta.Tuple(&full_args);
    const BlkCtx = switch (blk_type) {
        .default => full_args[0],
        .no_ctx => void,
        .no_args => FullArgs,
    };
    const Args = switch (blk_type) {
        .default => std.meta.Tuple(full_args[1..]),
        .no_ctx => FullArgs,
        .no_args => void,
    };

    return .{
        .Fn = @TypeOf(func),
        .BlkCtx = BlkCtx,
        .Args = Args,
        .FullArgs = FullArgs,
        .Return = fn_info.return_type.?,
        .nIn = n_tensors,
        .nOut = staticCountTensors(fn_info.return_type.?) orelse @compileError("Can't use " ++ @typeName(fn_info.return_type.?) ++ " in an MLIR function, because it has a variable number of tensors"),
    };
}

pub fn staticIsOnlyTensors(comptime T: type) bool {
    if (T == Tensor) return true;

    return switch (@typeInfo(T)) {
        .Array => |array_info| staticIsOnlyTensors(array_info.child),
        .Pointer => |ptr_info| ptr_info.size == .One and staticIsOnlyTensors(ptr_info.child),
        .Struct => |struct_info| {
            inline for (struct_info.fields) |field| {
                if (!staticIsOnlyTensors(field.type)) return false;
            }
            return true;
        },
        else => false,
    };
}

pub fn staticCountTensors(comptime T: type) ?usize {
    if (T == Tensor) return 1;

    return switch (@typeInfo(T)) {
        .Array => |array_info| array_info.len * (staticCountTensors(array_info.child) orelse return null),
        .Pointer => |ptr_info| {
            const n = staticCountTensors(ptr_info.child) orelse return null;
            if (ptr_info.size != .One and n > 0) return null;
            return n;
        },
        .Struct => |struct_info| {
            var count: usize = 0;
            inline for (struct_info.fields) |field| {
                count += staticCountTensors(field.type) orelse return null;
            }
            return count;
        },
        else => 0,
    };
}

/// Create a Tensor struct similar to base, keeping base tags,
/// but using mlir value and dims from the mlir operation.
pub fn fromMlirOperationWithTags(op: mlir.Operation, base: anytype) @TypeOf(base) {
    const LocalContext = struct {
        index: usize,
        op: mlir.Operation,
    };
    var context = LocalContext{ .index = 0, .op = op };
    var res = base;
    meta.visit((struct {
        fn cb(inner_ctx: *LocalContext, tensor: *Tensor) void {
            var new = Tensor.fromMlirValue(inner_ctx.op.result(inner_ctx.index));
            stdx.debug.internalAssert(new.rank() == tensor.rank(), "expected operand result to have rank {} but got {}", .{ tensor.rank(), new });
            // copy tags and sharding info over
            // some ops can change dims eg reduceWindow, so we trust mlir here.
            new._shape._tags = tensor._shape._tags;
            new._shape._sharding_info = tensor._shape._sharding_info;
            tensor.* = new;
            inner_ctx.index += 1;
        }
    }).cb, &context, &res);
    assert(context.index == op.numResults());
    return res;
}

/// Produces a custom call to `name` that takes a tensor and returns it.
///
/// For example, this can be used to extract tokens quickly if they run on a loop on the
/// GPU.
pub fn identityCustomCall(name: [:0]const u8, input: Tensor, context: *anyopaque) Tensor {
    const address: [8]u8 = @bitCast(@intFromPtr(context));
    var backend_config: [8:0]u8 = undefined;
    @memcpy(backend_config[0..8], address[0..8]);
    const ctx = CompilationContext.current();

    const loc = ctx.mlirCtx().location(@src()).namedFmt(ctx.mlirCtx(), "custom_call({s})", .{name});

    const op = dialect.stablehlo.custom_call(
        ctx.mlirCtx(),
        &.{input.value()},
        .{
            .api_version = 1,
            .has_side_effect = false,
            .call_target_name = name,
            .backend_config = backend_config[0..],
            .output_operand_aliases = &.{0},
        },
        &.{input.value().getType()},
        loc,
    );
    return Tensor._result(input.shape(), op.result(0));
}

/// At runtime the given tensor will be materialized and copied to host,
/// and the callback will be called on it.
pub fn addHostCallback(
    callback: *const fn (HostBuffer) void,
    input: Tensor,
) Tensor {
    // TODO: implement addCallback that exposes a pjrt.Buffer, so that the user can decide if they need to copy.
    if (input.getContext().target() != .cuda) return input;

    const len = input.byteSize();
    // Reserve memory to be able to log the runtime Buffer later during the computation.
    // This memory is leaked, we currently have no way to tie this lifetime to the lifetime of the module being compiled.
    const HostCallbackCtx = Context.HostCallbackCtx;
    const full_data = std.heap.page_allocator.alignedAlloc(u8, 32, len + 2 * @sizeOf(HostCallbackCtx)) catch {
        log.err("Failed to pre-allocate buffer to print {}.", .{input});
        return input;
    };

    // Save the HostBuffer inside the same memory slice, so that it's still present at runtime.
    // Use an fba to have the stable buffer at an aligned offset.
    var fba = std.heap.FixedBufferAllocator.init(full_data[len..]);
    const stable_ctx_ptr = fba.allocator().create(HostCallbackCtx) catch unreachable;
    stable_ctx_ptr.* = .{
        .host = HostBuffer.fromBytes(input.shape(), full_data[0..len]),
    };

    const backend_config: [2:null]?*const anyopaque = .{ callback, stable_ctx_ptr };
    const ctx = CompilationContext.current();

    const loc = ctx.mlirCtx().location(@src());
    const op = dialect.stablehlo.custom_call(
        ctx.mlirCtx(),
        &.{input.value()},
        .{
            .api_version = 1,
            .has_side_effect = false,
            .call_target_name = "zmlHostBufferCallback",
            .backend_config = @ptrCast(std.mem.sliceAsBytes(&backend_config)),
            .output_operand_aliases = &.{0},
        },
        &.{input.value().getType()},
        loc,
    );
    return Tensor._result(input.shape(), op.result(0));
}

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
    comptime T: type,
    comptime BlkCtx: type,
    comptime update_fn: fn (BlkCtx, T, T) T,
    inputs: T,
    blkctx: BlkCtx,
    index_tensors: anytype,
    updates: T,
    opts: Tensor.ScatterOpts,
) T {
    const loc = @src();
    const ctx = CompilationContext.current();

    const n_inputs = meta.count(Tensor, &inputs);
    const n_updates = meta.count(Tensor, &updates);
    stdx.debug.assert(n_inputs == n_updates, "zml.ops.scatter expects the same number of tensors in inputs and updates, got {} and {}", .{ n_inputs, n_updates });
    // TODO: check all dtypes are the same, this is currently an XLA limitation on scatter

    // Note: I was a bit lazy here, and I only look at tags on the first tensor.
    // we probably should check all of them.
    const self = meta.first(Tensor, inputs);
    const update = meta.first(Tensor, updates);
    var indices_per_axis, var coord_axes_ = Shape.parseStruct(Tensor, index_tensors);

    if (indices_per_axis.len == 0) return inputs;

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
        idx.* = idx.broad(indices_shape);
    }

    // The rewrite to dynamicUpdateSlice1d doesn't work for {layer=32,b=8,k=1024,h=8!,hd=128,bf16}.scatterSlices(.{ .layer = Tensor({i32}) }, Tensor({b=8,k=1024,h=8!,hd=128,bf16}))
    // const tagged_api = coord_axes_.len > 0;
    // if (T == Tensor and indices_per_axis.len == 1 and indices_per_axis.get(0).count() == 1 and ) {
    //     return self.dynamicUpdateSlice1d(
    //         updates,
    //         if (tagged_api) self.axis(coord_axes_.get(0)) else 0,
    //         indices_per_axis.get(0).reshape(.{}),
    //     );
    // }

    // TODO: validate coord axes: all coord_axes should exist inside self
    // TODO: ideally we should catch all possible scatter errors and provide nice error messages.
    // TODO: simplify writing scatter by transposing updates

    var config = scatterConfig(self.shape(), update.shape(), indices_per_axis, coord_axes_);
    const indices = scatterPrepareIndices(&config, self.shape(), update.shape(), &indices_per_axis, &coord_axes_);
    // const n_indices_axes = update.rank() - _collectAxes(AxisKind, up_kind, .update_window).len;
    // stdx.debug.assert(n_indices_axe == coord_axes_.len, "scatter({_}, {any}) expects 'updates' to contain all axes from 'indices', got indices={s}, updates={_}", .{ self, index_tensors, coord_axes_.constSlice(), update });

    const mlir_ctx = ctx.mlirCtx();
    var _scalar: T = inputs;
    meta.visit(struct {
        pub fn cb(_: void, x: *Tensor) void {
            x.* = .{ ._shape = Shape.init(.{}, x.dtype()), ._id = undefined };
        }
    }.cb, {}, &_scalar);

    const UpdateS = BlockSign(update_fn);
    const update_block, _ = ctx.makeBlock(.hermetic, UpdateS, update_fn, blkctx, .{ _scalar, _scalar });

    var input_values = std.ArrayList(mlir.Value).initCapacity(ctx.allocator(), n_inputs) catch @panic("OOM");
    defer input_values.deinit();
    meta.collect(CompilationContext.getValue, ctx, &input_values, &inputs) catch unreachable;
    var updates_values = std.ArrayList(mlir.Value).initCapacity(ctx.allocator(), n_updates) catch @panic("OOM");
    defer updates_values.deinit();
    meta.collect(CompilationContext.getValue, ctx, &updates_values, &updates) catch unreachable;

    const op = dialect.stablehlo.scatter(
        mlir_ctx,
        input_values.items,
        &.{indices.value()},
        updates_values.items,
        update_block,
        .{
            .update_window_dims = _collectAxes(AxisKind, config.up_kind, .update_window).constSlice(),
            .inserted_window_dims = _collectAxes(AxisKind, config.op_kind, .inserted_window).constSlice(),
            // TODO: the batching_dims is a lie
            // although they are in stablehlo, they aren't handled well by "scatter_simplifier" pass that insert extra transposes when it sees them,
            // while they act as extra coordinates. we should rewrite them
            .input_batching_dims = _collectAxes(AxisKind, config.op_kind, .batching).constSlice(),
            .scatter_indices_batching_dims = config.indices_batch_axes.constSlice(),
            .scatter_dims_to_operand_dims = config.scatter_to_operand_axes.constSlice(),
            .index_vector_dim = indices.rank() - 1,
            .indices_are_sorted = opts.indices_are_sorted,
            .unique_indices = opts.indices_are_unique,
        },
        mlir_ctx.location(loc),
    );

    var res: T = inputs;
    const LocalContext = struct {
        op: mlir.Operation,
        index: usize = 0,
    };
    var local_context = LocalContext{ .op = op };
    meta.visit((struct {
        fn cb(inner_ctx: *LocalContext, tensor: *Tensor) void {
            const val = inner_ctx.op.result(inner_ctx.index);
            tensor.* = Tensor._result(tensor.shape(), val);
            inner_ctx.index += 1;
        }
    }).cb, &local_context, &res);
    assert(local_context.index == op.numResults());
    return res;
}

const ScatterConfig = struct {
    op_kind: std.BoundedArray(AxisKind, Tensor.MAX_RANK) = .{},
    up_kind: std.BoundedArray(AxisKind, Tensor.MAX_RANK) = .{},
    indices_batch_axes: Shape.DimsArray = .{},
    scatter_to_operand_axes: Shape.DimsArray = .{},
};

const AxisKind = enum { batching, update_window, inserted_window, window_id };

fn scatterConfig(
    op: Shape,
    update: Shape,
    indices_per_axis: std.BoundedArray(Tensor, Tensor.MAX_RANK),
    coord_axes_: Shape.TagsArray,
) ScatterConfig {
    var op_kind: std.BoundedArray(AxisKind, Tensor.MAX_RANK) = .{};
    var up_kind: std.BoundedArray(AxisKind, Tensor.MAX_RANK) = .{};
    var indices_batch_axes: Shape.DimsArray = .{};
    var scatter_to_operand_axes: Shape.DimsArray = .{};

    const tagged_api = coord_axes_.len > 0;
    const indices = indices_per_axis.get(0).shape();

    if (tagged_api) {
        for (coord_axes_.constSlice()) |t| {
            scatter_to_operand_axes.appendAssumeCapacity(op.axis(t));
        }

        for (op.tags()) |t| {
            if (update.hasTag(t)) |_| {
                if (indices.hasTag(t)) |id_ax| {
                    if (std.mem.indexOfScalar(Shape.Tag, coord_axes_.constSlice(), t) != null) {
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

        // Note: we assume the scatter_dims appear in the same order inside indices and inside op.
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
                stdx.debug.assert(update.dim(up_ax) <= op.dim(self_ax), "scatter expects the slices described in 'updates' to fit inside 'op', but along axis .{s} it doesn't. Got op={_}, updates={_}.", .{ t, op, update });
                up_kind.appendAssumeCapacity(.update_window);
            } else {
                // TODO: consider accepting untagged update here.
                std.debug.panic("scatter expects 'updates' to be made of axes from op={_} and from indices={s}, got unknown tag {s} in {_}", .{ op, coord_axes_.constSlice(), t, update });
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
    }

    return .{
        .op_kind = op_kind,
        .up_kind = up_kind,
        .indices_batch_axes = indices_batch_axes,
        .scatter_to_operand_axes = scatter_to_operand_axes,
    };
}

test scatterConfig {
    const zml = @import("zml.zig");
    const platform = zml.testing.env();

    var comp = try zml.module.CompilationContext.init(std.testing.allocator, "test", platform);
    defer comp.deinit();
    comp.activate();
    defer comp.deactivate();

    const Local = struct {
        pub fn idx(idx_shape: anytype) Tensor {
            return Tensor.constant(idx_shape, .{ .i32 = 0 });
        }
    };

    const idx = Local.idx;
    const op = Shape.init(.{ .a = 10, .b = 20 }, .f32);

    // Use .a as a batching axis with .a=10 x .n=8 updates of 2 elements of .b
    {
        const indices, const coords_tags = Shape.parseStruct(Tensor, .{ .b = idx(.{ .a = 10, .n = 8 }) });
        const update = Shape.init(.{ .a = 10, .n = 8, .b = 2 }, .f32);

        const cfg = scatterConfig(op, update, indices, coords_tags);
        try std.testing.expectEqualSlices(AxisKind, &.{ .batching, .update_window }, cfg.op_kind.constSlice());
        try std.testing.expectEqualSlices(AxisKind, &.{ .batching, .window_id, .update_window }, cfg.up_kind.constSlice());
    }

    // similar, but use the normalized form where .a is no longer an explicit batching axis.
    {
        const indices, const coords_tags = Shape.parseStruct(Tensor, .{ .a = idx(.{ .a = 10, .n = 8 }), .b = idx(.{ .a = 10, .n = 8 }) });
        const update = Shape.init(.{ .a = 10, .n = 8, .b = 2 }, .f32);

        const cfg = scatterConfig(op, update, indices, coords_tags);
        try std.testing.expectEqualSlices(AxisKind, &.{ .inserted_window, .update_window }, cfg.op_kind.constSlice());
        try std.testing.expectEqualSlices(AxisKind, &.{ .window_id, .window_id, .update_window }, cfg.up_kind.constSlice());
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
    indices_per_axis: *std.BoundedArray(Tensor, Tensor.MAX_RANK),
    indices_axes: *Shape.TagsArray,
) Tensor {
    var old_scatter_to_op_axes = cfg.scatter_to_operand_axes;
    const batching = _collectAxes(AxisKind, cfg.op_kind, .batching);
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
    var indices: std.BoundedArray(Tensor, Tensor.MAX_RANK) = .{};
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
            log.warn("Found a slow scatter pattern, which is going to generate a while loop: scatter({_}, {any}, {_}). Because the index axes aren't the major ones in the input tensor.", .{ op, scatter_to_op_axes.constSlice(), update });
            break;
        }
    }
    return Tensor.stack(indices.constSlice(), .last, .coord);
}

inline fn toI64(values: anytype) []i64 {
    var res: [Tensor.MAX_RANK]i64 = undefined;
    for (values, 0..) |val, i| res[i] = @intCast(val);
    return res[0..values.len];
}
