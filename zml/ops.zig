const std = @import("std");
const mlir = @import("mlir.zig");
const buffer = @import("buffer.zig");

const helpers = @import("helpers.zig");
const module = @import("module.zig");
const meta = @import("meta.zig");

const CompilationContext = module.CompilationContext;
const Tensor = @import("tensor.zig").Tensor;
const Shape = @import("shape.zig").Shape;
const Data = @import("dtype.zig").Data;
const DataType = @import("dtype.zig").DataType;
const Buffer = buffer.Buffer;
const EnumLiteral = @TypeOf(.enum_literal);

const dialect = struct {
    const stablehlo = @import("mlir/dialects").stablehlo;
};

const assert = std.debug.assert;
const log = std.log.scoped(.zml_tensor);

/// Generate an MLIR call to the given member function with the given tensors.
pub fn call(self: anytype, comptime func: meta.DeclEnum(@TypeOf(self)), args: anytype) @TypeOf(@call(.auto, @field(meta.UnwrapPtr(@TypeOf(self)), @tagName(func)), .{self} ++ args)) {
    // TODO: this should use `self.getContext().callFunc(self, args)`

    return @call(.auto, @field(@TypeOf(self), @tagName(func)), .{self} ++ args);
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
    const cond_block = ctx.makeBlock(cond_fn, CondS, blkctx, inputs);
    const body_block = ctx.makeBlock(body_fn, BodyS, blkctx, inputs);
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

    var res: BodyS.Args = inputs;
    module.assignResults(&res, null, op);
    return res;
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
    inputs: meta.FnParam(body_fn, 0),
    inits: meta.FnParam(body_fn, 0),
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

    const body_block = ctx.makeBlock(body_fn, BodyS, {}, .{ inits, inits });

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
    // To that order, we initialize `result` to `inputs`, then we use meta.visit,
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
    padding_values: []const i64,
    padding_shape: []const i64,
};

pub fn reduceWindow(
    comptime body_fn: anytype,
    inputs: meta.FnParam(body_fn, 0),
    inits: meta.FnParam(body_fn, 0),
    opts: ReduceWindowOpts,
) meta.FnResult(body_fn) {
    const BodyS = comptime BlockSignNoCtx(body_fn);
    comptime {
        if (BodyS.Return != @TypeOf(inputs)) @compileError("reduce body function need to have the following signature `fn (left: T, right: T) T`, got: " ++ @typeName(body_fn));
    }
    const ctx = CompilationContext.current();
    const body_block = ctx.makeBlock(body_fn, BodyS, {}, .{ inits, inits });
    const N = comptime @divExact(BodyS.nIn, 2);
    var input_values: [N]mlir.Value = undefined;
    ctx.extractValues(&inputs, &input_values);
    var init_values: [N]mlir.Value = undefined;
    ctx.extractValues(&inits, &init_values);

    const loc = ctx.mlirCtx().location(@src());

    const pad_shape = mlir.RankedTensorType.init(opts.padding_shape, mlir.ext.Type.fromDType(ctx.mlirCtx(), .i64)).as(mlir.Type).?;
    const op = mlir.Operation.make(ctx.mlirCtx(), "stablehlo.reduce_window", .{
        .variadic_operands = &.{ input_values[0..], init_values[0..] },
        .result_type_inference = true,
        .blocks = &.{body_block},
        .attributes = &.{
            .{ "window_dimensions", mlir.DenseArrayAttribute(.i64).init(ctx.mlirCtx(), opts.window_dimensions).as(mlir.Attribute).? },
            .{ "window_strides", mlir.DenseArrayAttribute(.i64).init(ctx.mlirCtx(), opts.window_strides).as(mlir.Attribute).? },
            .{ "base_dilations", mlir.DenseArrayAttribute(.i64).init(ctx.mlirCtx(), opts.base_dilations).as(mlir.Attribute).? },
            .{ "window_dilations", mlir.DenseArrayAttribute(.i64).init(ctx.mlirCtx(), opts.window_dilations).as(mlir.Attribute).? },
            .{ "padding", mlir.DenseIntOrFPElementsAttribute(.i64).init(pad_shape, std.mem.sliceAsBytes(opts.padding_values)).as(mlir.Attribute).? },
        },
        .location = loc,
    });

    var res: BodyS.Return = inputs;
    module.assignResults(&res, null, op);
    return res;
}

/// Runs a given function for several steps, and returns a stack of each step output.
/// The step outputs will be stacked along the first axis.
pub fn for_(comptime func: anytype, blk_ctx: BlockSign(func).BlkCtx, num_steps_: anytype) BlockSign(func).Return {
    const num_steps: u32, const step_tag = blk: {
        const dims, const tags = Shape.parseDimensions(num_steps_);
        meta.assert(dims.len == 1, "zml.for_ only supports one num_step, Received: {any}", .{num_steps_});
        break :blk .{ @intCast(dims.get(0)), tags.get(0) };
    };
    const S = comptime BlockSign(func);

    const ForBlk = struct {
        blk_ctx: S.BlkCtx,
        step_tag: @TypeOf(step_tag), // This is a Shape.Tag, but we rather keep it private
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
            meta.internalAssert(inputs.len == 2, "too many tensors", .{});
            const res, const step_res = inputs[0..2].*;
            return res.dynamicUpdateSlice1d(step_res.insertAxes(0, .{._}), 0, idx);
        }

        /// Prepare buffer to store all results steps.
        fn prep(self: Self, x: Tensor) Tensor {
            var shape = x.shape();
            shape._dims.insert(0, self.num_steps) catch unreachable;
            shape._tags.insert(0, self.step_tag) catch unreachable;
            return Tensor.constant(shape, x.dtype().zero());
        }

        fn wrapFirstStep(x: Tensor, tag_: @TypeOf(step_tag)) Tensor {
            var shape = x.shape();
            shape._dims.insert(0, 1) catch unreachable;
            shape._tags.insert(0, tag_) catch unreachable;
            return x.reshape(shape);
        }
    };

    // This first step won't appear in the generated MLIR,
    // it's only used to infer the output shapes.
    const first_step = @call(.auto, func, .{ blk_ctx, Tensor.scalar(0, .i32) });
    log.debug("for_ first_step: {}", .{first_step});
    // Optimize for small num reps
    if (num_steps == 1) {
        // return helpers.mapTensors(ForBlk.wrapFirstStep, first_step, .{ step_tag });
        return first_step;
    }

    const allocator = CompilationContext.current()._allocator;
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
            Tensor.scalar(0, .i32),
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
        try std.testing.expectEqual(0, squares.getValue(f32));
    }
    // Wow 4 in rows !
    {
        const squares = try zml.testing.compileAndCall(platform, Squares.forward, .{4});
        try std.testing.expectEqual([_]f32{ 0, 1, 4, 9 }, try squares.getValue([4]f32));
    }
    // AGI is coming, computing 10 squares as it's nothing.
    {
        const squares = try zml.testing.compileAndCall(platform, Squares.forward, .{10});
        try std.testing.expectEqual(
            [_]f32{ 0, 1, 4, 9, 16, 25, 36, 49, 64, 81 },
            try squares.getValue([10]f32),
        );
    }
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
    const true_branch_block = ctx.makeBlock(true_branch_fn, TrueBlockSignature, blkctx, {});
    const false_branch_block = ctx.makeBlock(false_branch_fn, TrueBlockSignature, blkctx, {});
    const loc = ctx.mlirCtx().location(@src());

    const op = mlir.Operation.make(ctx.mlirCtx(), "stablehlo.if", .{
        .operands = &.{pred.value()},
        .result_type_inference = true,
        .blocks = &.{ true_branch_block, false_branch_block },
        // We can't verify right away, cause the weights captured by the if haven't been added yet.
        .verify = false,
        .location = loc,
    });

    var res: TrueBlockSignature.Return = undefined;
    module.assignResults(&res, null, op);
    return res;
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
    const block = ctx.makeBlock(comp_fn, BodyS, blkctx, inits);
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

        // if (arg.type) |ArgType| {
        //     full_args[i] = ArgType;
        //     if (i >= arg_start) {
        //         n_tensors += staticCountTensors(ArgType) orelse @compileError("Can't use " ++ @typeName(ArgType) ++ " in an MLIR function, because it has a variable number of tensors");
        //     }
        // } else {
        //     // anytype are considered to not have tensors.
        //     // violation of this will be detected when calling `compile()` but not at Zig compile time.
        //     full_args[i] = void;
        // }
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

    const xx = .{
        .BlkCtx = BlkCtx,
        .Args = Args,
        .FullArgs = FullArgs,
        .Return = fn_info.return_type.?,
        .nIn = n_tensors,
        .nOut = staticCountTensors(fn_info.return_type.?) orelse @compileError("Can't use " ++ @typeName(fn_info.return_type.?) ++ " in an MLIR function, because it has a variable number of tensors"),
    };
    return xx;
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

/// Produces a custom call to `name` that takes a tensor and returns it.
///
/// For example, this can be used to extract tokens quickly if they run on a loop on the
/// GPU.
pub fn identityCustomCall(name: [:0]const u8, input: Tensor, context: *anyopaque) Tensor {
    const address: [8]u8 = @bitCast(@intFromPtr(context));
    var backend_config: [8:0]u8 = undefined;
    @memcpy(backend_config[0..8], address[0..8]);
    const ctx = CompilationContext.current();

    const loc = ctx.mlirCtx().location(@src()).namedFmt(ctx.mlirCtx(), "name={s}", .{name});

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
