const std = @import("std");

const asynk = @import("async");
const dialects = @import("mlir/dialects");
const mlir = @import("mlir");
const stdx = @import("stdx");
const zml = @import("zml");
const Shape = zml.Shape;

var global_compilation_context: ?*CompilationContext = null;
pub const MAX_RANK: u8 = 8;

pub const TensorOriginType = enum {
    argument,
    value,
};

pub const TensorOrigin = union(TensorOriginType) {
    argument: void,
    value: *const mlir.Value,
};

pub const Tensor = struct {
    var current_id: std.atomic.Value(usize) = .{ .raw = 1 };

    id: usize,
    auto_broadcast: bool = false,
    _shape: zml.Shape,
    tensor_origin: TensorOrigin = .{ .argument = {} },

    pub const tensor = init;

    fn mlirType(self: Tensor, mlir_ctx: *mlir.Context) *const mlir.Type {
        return mlir.rankedTensorType(
            self.dims(),
            zml.mlir.Type.fromDType(mlir_ctx, self.dtype()),
        );
    }

    fn mlirCtx() *mlir.Context {
        return CompilationContext.current().mlir_ctx;
    }

    fn currentBlock() *mlir.Block {
        return CompilationContext.current().currentBlock();
    }

    pub fn init(shape_: zml.Shape) Tensor {
        return .{ .id = Tensor.current_id.fetchAdd(1, .seq_cst), ._shape = shape_ };
    }

    pub fn shape(self: Tensor) Shape {
        return self._shape;
    }

    pub fn dtype(self: Tensor) zml.DataType {
        return self._shape.dtype();
    }

    pub fn rank(self: Tensor) u4 {
        return self._shape.rank();
    }

    pub fn dim(self: Tensor, axis_: anytype) i64 {
        return self._shape.dim(axis_);
    }

    pub fn dims(self: *const Tensor) []const i64 {
        return self._shape.dims();
    }

    pub fn axis(self: Tensor, axis_: anytype) u3 {
        return self._shape.axis(axis_);
    }

    pub fn count(self: Tensor) usize {
        return self._shape.count();
    }

    pub fn autoBroadcast(self: Tensor) Tensor {
        var ret = self;
        ret.auto_broadcast = true;
        return ret;
    }

    pub fn matmul(lhs: Tensor, rhs: Tensor) Tensor {
        //stdx.debug.assert(lhs.rank() >= 1 and rhs.rank() >= 1, "Can't matmul({f}, {f}) ! The two tensors need to have at least rank 1.", .{ lhs, rhs });

        const contracting = [_][2]i8{.{ -1, if (rhs.rank() >= 2) rhs.rank() - 2 else 0 }};
        if (lhs.rank() == 1 or rhs.rank() <= 2) {
            // When lhs is a vector or rhs is small the torch semantics match the dot_general semantics and life is easy.
            return lhs.dotGeneral(rhs, &contracting, &.{});
        }

        //stdx.debug.assert(lhs.rank() == 2, "Can't matmul({f}, {f}) ! One of the two tensors need to have a rank less than 2.", .{ lhs, rhs });

        // Pytorch treats the extra dimensions of rhs has batching dimensions,
        // and implicitly broadcast lhs along those.
        // We make this broadcasting explicit.
        var left_shape = rhs.shape();
        left_shape._dims.set(left_shape.axis(-2), lhs.dim(-2));
        left_shape._tags.set(left_shape.axis(-2), lhs.shape().tag(-2));
        left_shape._dims.set(left_shape.axis(-1), lhs.dim(-1));
        left_shape._tags.set(left_shape.axis(-1), lhs.shape().tag(-1));
        const lhs_broad = lhs.broadcastLeft(left_shape);

        const n_batching_axes = rhs.rank() - lhs.rank();
        var batching: [MAX_RANK][2]i8 = undefined;
        for (0..n_batching_axes) |i| {
            batching[i] = .{ @intCast(i), @intCast(i) };
        }
        return lhs_broad.dotGeneral(rhs, &contracting, batching[0..n_batching_axes]);
    }

    pub const DotArgs = struct {
        contract_on: []const [2]i8,
        batch_on: ?[]const [2]i8 = null,
    };

    pub fn dot(lhs: Tensor, rhs: Tensor, args: DotArgs) Tensor {
        return lhs.dotGeneral(rhs, args.contract_on, args.batch_on orelse &.{});
    }

    const ArgsKind = enum {
        simple,
        contracting_only,
        full,
    };

    fn isFullArgsKind(comptime T: type) bool {
        return std.meta.fieldIndex(T, "contracting") != null and std.meta.fieldIndex(T, "batching") != null;
    }

    fn getArgsKind(comptime T: type) ArgsKind {
        const type_info = @typeInfo(T);
        return switch (type_info) {
            .enum_literal => .simple,
            .@"struct" => if (isFullArgsKind(T)) .full else .contracting_only,
            else => unreachable,
        };
    }

    pub fn dot2(lhs: Tensor, rhs: Tensor, args: anytype) Tensor {
        _ = lhs; // autofix
        _ = rhs; // autofix
        const args_kind = getArgsKind(@TypeOf(args));
        std.debug.print("{}\n", .{args_kind});
        unreachable;
    }

    pub fn dotGeneral(
        lhs: Tensor,
        rhs: Tensor,
        contracting_axes: []const [2]i8,
        batching_axes: []const [2]i8,
    ) Tensor {
        stdx.debug.assert(lhs.dtype() == rhs.dtype(), "dotGeneral expects tensors to be of the same type, got {} and {}", .{ lhs.dtype(), rhs.dtype() });

        const Axes = stdx.BoundedArray(i64, MAX_RANK);

        var res_shape: Shape = .{ ._dtype = lhs.dtype() };
        // Validate batching axes
        var lhs_batching_axes: Axes = .{};
        var rhs_batching_axes: Axes = .{};
        for (batching_axes) |b_axes| {
            const l, const r = b_axes;
            stdx.debug.assert(lhs._shape.dim(l) == rhs._shape.dim(r), "dotGeneral expects batching dimensions to be equal, got {} and {} in {f} and {f}", .{ l, r, lhs, rhs });
            var t = lhs._shape.tag(l);
            if (t == Shape.TagUnknown) t = rhs._shape.tag(r);
            res_shape = res_shape.appendDim(lhs._shape.dim(l), t);
            lhs_batching_axes.appendAssumeCapacity(lhs._shape.axis(l));
            rhs_batching_axes.appendAssumeCapacity(rhs._shape.axis(r));
        }

        // Validate contracting axes
        var lhs_contracting_axes: Axes = .{};
        var rhs_contracting_axes: Axes = .{};
        for (contracting_axes) |c_axes| {
            const l, const r = c_axes;
            stdx.debug.assert(lhs._shape.dim(l) == rhs._shape.dim(r), "dotGeneral expects contracting dimensions to be equal, got {} and {} in {f} and {f}", .{ l, r, lhs, rhs });
            lhs_contracting_axes.appendAssumeCapacity(lhs._shape.axis(l));
            rhs_contracting_axes.appendAssumeCapacity(rhs._shape.axis(r));
        }

        // Result shape is obtained by concatenating batching dimensions, (already done)
        // then dimensions from lhs axes that aren't contracting nor batching,
        // then dimensions from rhs axes that aren't contracting nor batching.
        for (0..lhs.rank()) |l| {
            if (std.mem.indexOfScalar(i64, lhs_contracting_axes.constSlice(), @intCast(l))) |_| {
                continue;
            }
            if (std.mem.indexOfScalar(i64, lhs_batching_axes.constSlice(), @intCast(l))) |_| {
                continue;
            }
            res_shape = res_shape.appendDim(lhs._shape.dim(l), lhs._shape.tag(l));
        }
        for (0..rhs.rank()) |r| {
            if (std.mem.indexOfScalar(i64, rhs_contracting_axes.constSlice(), @intCast(r))) |_| {
                continue;
            }
            if (std.mem.indexOfScalar(i64, rhs_batching_axes.constSlice(), @intCast(r))) |_| {
                continue;
            }
            res_shape = res_shape.appendDim(rhs._shape.dim(r), rhs._shape.tag(r));
        }

        const mlir_ctx = lhs.getContext().mlir_ctx;
        //const loc = lhs.getContext().location(@src(), "dot({f},{f},contracting={any},batching={any}", .{ lhs, rhs, contracting_axes, batching_axes });
        const loc = mlir.Location.unknown(mlirCtx());
        const op = dialects.stablehlo.dot_general(
            mlir_ctx,
            lhs.value(),
            rhs.value(),
            mlir.rankedTensorType(res_shape.dims(), zml.mlir.Type.fromDType(mlir_ctx, res_shape.dtype())),
            .{
                .lhs_batching_dimensions = lhs_batching_axes.constSlice(),
                .rhs_batching_dimensions = rhs_batching_axes.constSlice(),
                .lhs_contracting_dimensions = lhs_contracting_axes.constSlice(),
                .rhs_contracting_dimensions = rhs_contracting_axes.constSlice(),
                .dot_precision = .fast,
            },
            loc,
        );
        CompilationContext.current().currentBlock().appendOwnedOperation(op);
        return _result(res_shape, op.result(0));
    }

    pub fn _result(sh: Shape, val: *const mlir.Value) Tensor {
        const res: Tensor = .{
            ._shape = sh,
            .tensor_origin = .{ .value = val },
            .id = Tensor.current_id.fetchAdd(1, .seq_cst),
        };

        //if (builtin.mode == .Debug) {
        //    // Check that the MLIR value actually have the same shape.
        //    const other = fromMlirValue(val);
        //    stdx.debug.internalAssert(sh.eql(other._shape), "Created a {f} from Mlir value but expected {f}", .{ other._shape, res._shape });
        //}

        return res;
    }

    pub fn constant(val: zml.DataType.Value) Tensor {
        const op = dialects.stablehlo.constant(
            mlirCtx(),
            &.{},
            zml.mlir.Type.fromDType(mlirCtx(), val.dtype()),
            val.constSlice(),
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());
        return _result(.init(&.{}, val.dtype()), op.result(0)).autoBroadcast();
    }

    pub fn add(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "add", dialects.stablehlo.add)(self, other);
    }

    pub fn maximum(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "maximum", dialects.stablehlo.maximum)(self, other);
    }

    pub fn relu(self: Tensor) Tensor {
        return self.maximum(.constant(self.dtype().zero()));
    }

    pub fn flattenAll(self: Tensor) Tensor {
        // TODO: rename to just flatten, once flatten is moved to torch
        return self.reshape(.{self.count()});
    }

    pub fn broadcast(self: Tensor, output_shape: Shape, axes_: []const i64) Tensor {
        stdx.debug.assert(axes_.len == self.rank(), "broadcast expects axes_ to map all axes from self to axes of the output shape, got broadcast({f}, {f}, {any})", .{ self, output_shape, axes_ });
        for (0.., axes_) |self_ax, other_ax| {
            const d = self.dim(self_ax);
            stdx.debug.assert(d == 1 or d == output_shape.dim(other_ax), "broadcast expects shape axes to either be 1-sized or to match the target size. got broadcast({f}, {f}, {any}), error on self axis {d} mapping to other axis {d}", .{ self, output_shape, axes_, self_ax, other_ax });
        }

        const res_shape = output_shape.withDtype(self.dtype());
        if (std.mem.eql(i64, self.dims(), output_shape.dims())) {
            // No broadcast needed. We don't emit a new stablehlo value
            // but we propagate output_shape tags.
            return _result(res_shape, self.value());
        }
        const ctx = self.getContext();
        const result_type = mlir.rankedTensorType(
            res_shape.dims(),
            zml.mlir.Type.fromDType(ctx.mlir_ctx, res_shape.dtype()),
        );
        const broadcast_op = dialects.stablehlo.broadcast_in_dim(ctx.mlir_ctx, self.value(), axes_, result_type, .unknown(ctx.mlir_ctx)).appendTo(ctx.currentBlock());
        return _result(res_shape, broadcast_op.result(0));
    }

    pub fn broad(self: Tensor, other: Shape) Tensor {
        // TODO: broad is too restrictive because sometime you only want to specify one specific axis
        // Note: if you code below, make sure to update Shape.canBroadcastTo.
        stdx.debug.assert(self._shape.canBroadcastTo(other), "Can't broadcast {f} to {f}", .{ self, other });

        // Already the right shape
        if (std.mem.eql(i64, self.dims(), other.dims())) return self;

        // Non ambiguous broadcasting
        // TODO: broad is error prone because of this:
        // it will happily broadcast .{ .a = 10, .b = 1 } to .{ .b = 10, .a = 5 }
        if (self._shape.rank() == 0 or self._shape.rank() == other.rank()) {
            const all_axes = [MAX_RANK]i64{ 0, 1, 2, 3, 4, 5, 6, 7 };
            return self.broadcast(other, all_axes[0..self.rank()]);
        }

        // check that each axis of self maps to an axis of other
        var axes_: stdx.BoundedArray(i64, MAX_RANK) = .{};
        for (self._shape.tags()) |t| {
            axes_.appendAssumeCapacity(@intCast(other.axis(t)));
        }
        return self.broadcast(other, axes_.constSlice());
    }

    pub fn broadcastLeft(self: Tensor, output_shape: Shape) Tensor {
        stdx.debug.assert(self.rank() <= output_shape.rank(), "broadcastLeft expects tensor rank to be less than output tensor rank, got {d} and {d}", .{ self.rank(), output_shape.rank() });

        const a = output_shape.rank() - self.rank();
        if (self.rank() == output_shape.rank() and std.mem.eql(i64, self.dims(), output_shape.dims())) {
            return self;
        }

        return self.broadcast(output_shape, Shape.range(output_shape.rank(), output_shape.dtype()).dims()[a..]);
    }

    pub const ArangeArgs = struct {
        start: i64 = 0,
        end: i64,
        step: i64 = 1,
    };

    pub fn arange(args: ArangeArgs, dt: zml.DataType) Tensor {
        stdx.debug.assert(args.start <= args.end, "arange expects 'args.start' to be less than 'args.end', got {} and {}", .{ args.start, args.end });
        stdx.debug.assert(args.step > 0, "arange expects 'args.step' to be positive, got {}", .{args.step});

        const ctx = CompilationContext.current();
        //const loc = ctx.location(@src(), "arange({}, dtype={})", .{ args, dt });
        const loc = mlir.Location.unknown(ctx.mlir_ctx);

        const n_steps = std.math.divCeil(i64, args.end - args.start, args.step) catch unreachable;
        const sh = Shape.init(.{n_steps}, dt);
        var op = dialects.stablehlo.iota(ctx.mlir_ctx, 0, mlir.rankedTensorType(sh.dims(), zml.mlir.Type.fromDType(ctx.mlir_ctx, sh.dtype())), loc).appendTo(CompilationContext.current().currentBlock());
        var res = _result(sh, op.result(0));
        _ = &res;

        //if (args.step != 1) {
        //    res = res.scale(args.step);
        //}

        //if (args.start != 0) {
        //    res = res.addConstant(args.start);
        //}

        return res;
    }

    pub fn reshape(self: Tensor, output_shape_: anytype) Tensor {
        const output_shape = self._shape.reshape(output_shape_);
        const ctx = self.getContext().mlir_ctx;
        const tensor_type = mlir.rankedTensorType(output_shape.dims(), zml.mlir.Type.fromDType(ctx, output_shape.dtype()));
        //const loc = self.getContext().location(@src(), "reshape({f})", .{output_shape});
        const reshape_op = dialects.stablehlo.reshape(
            self.getContext().mlir_ctx,
            self.value(),
            tensor_type,
            .unknown(ctx),
        ).appendTo(CompilationContext.current().currentBlock());
        return _result(output_shape, reshape_op.result(0));
    }

    pub fn convert(self: Tensor, to: zml.DataType) Tensor {
        if (to == self.dtype()) {
            return self;
        }
        //const loc = self.getContext().location(@src(), "convert({f},to={s})", .{ self, @tagName(to) });

        const mlir_ctx = self.getContext().mlir_ctx;
        const res_type = mlir.rankedTensorType(self.dims(), zml.mlir.Type.fromDType(mlir_ctx, to));
        const op = dialects.stablehlo.convert(
            mlir_ctx,
            self.value(),
            res_type,
            .unknown(mlir_ctx),
        ).appendTo(CompilationContext.current().currentBlock());
        return _result(self._shape.withDtype(to), op.result(0));
    }

    pub fn select(bool_tensor: Tensor, on_true: Tensor, on_false: Tensor) Tensor {
        stdx.debug.assert(bool_tensor.dtype() == .bool, "select expects input tensor type to be a boolean, got {}", .{bool_tensor.dtype()});
        stdx.debug.assert(on_true.dtype() == on_false.dtype(), "select expects 'on_true' and 'on_false' tensor types to be equal, got {} and {}", .{ on_true.dtype(), on_false.dtype() });

        if (bool_tensor.rank() != 0 and on_true.rank() == 0) {
            return bool_tensor.select(on_true.broad(bool_tensor.shape()), on_false);
        }
        if (bool_tensor.rank() != 0 and on_false.rank() == 0) {
            return bool_tensor.select(on_true, on_false.broad(bool_tensor.shape()));
        }

        stdx.debug.assert(bool_tensor._shape.eqlDims(on_true._shape), "select expects input tensor and 'on_true' tensor dimensions to match, got {f} and {f}", .{ bool_tensor._shape, on_true._shape });
        stdx.debug.assert(bool_tensor._shape.eqlDims(on_false._shape), "select expects input tensor and 'on_false' tensor dimensions to match, got {f} and {f}", .{ bool_tensor._shape, on_false._shape });

        //const loc = bool_tensor.mlirCtx().location(@src());
        const loc = mlir.Location.unknown(mlirCtx());
        const op = dialects.stablehlo.select(
            mlirCtx(),
            bool_tensor.value(),
            on_true.value(),
            on_false.value(),
            loc,
        ).appendTo(CompilationContext.current().currentBlock());

        return _result(on_true._shape, op.result(0));
    }

    pub const ArgMaxRes = struct {
        values: Tensor,
        indices: Tensor,

        fn cmpOld2(left: ArgMaxRes, right: ArgMaxRes) ArgMaxRes {
            const left_val = left.values;
            const right_val = right.values;
            const left_idx = left.indices;
            const right_idx = right.indices;

            const left_gt_right = left_val.cmp(.GT, right_val);
            const is_nan = left_val.cmp(.NE, left_val);
            const left_gt_or_nan = left_gt_right.logical(.OR, is_nan);
            // we are bubbling up Nan.
            const max_val = left_gt_or_nan.select(left_val, right_val);

            // If left_val == right_val: keep the smallest idx.
            const is_same = left_val.cmp(.EQ, right_val);
            const is_first = left_idx.cmp(.LT, right_idx);
            const is_same_but_first = is_same.logical(.AND, is_first);
            const keep_left_idx = left_gt_or_nan.logical(.OR, is_same_but_first);
            const max_idx = keep_left_idx.select(left_idx, right_idx);

            return .{ .values = max_val, .indices = max_idx };
        }

        fn cmpOld(left: [2]Tensor, right: [2]Tensor) [2]Tensor {
            const left_val = left[0];
            const right_val = right[0];
            const left_idx = left[1];
            const right_idx = right[1];

            const left_gt_right = left_val.cmp(.GT, right_val);
            const is_nan = left_val.cmp(.NE, left_val);
            const left_gt_or_nan = left_gt_right.logical(.OR, is_nan);
            // we are bubbling up Nan.
            const max_val = left_gt_or_nan.select(left_val, right_val);

            // If left_val == right_val: keep the smallest idx.
            const is_same = left_val.cmp(.EQ, right_val);
            const is_first = left_idx.cmp(.LT, right_idx);
            const is_same_but_first = is_same.logical(.AND, is_first);
            const keep_left_idx = left_gt_or_nan.logical(.OR, is_same_but_first);
            const max_idx = keep_left_idx.select(left_idx, right_idx);

            return .{ max_val, max_idx };
        }

        fn cmp(values: ReduceArgs, indices: ReduceArgs) struct { Tensor, Tensor } {
            const left_gt_right = values.left.cmp(.GT, values.right);
            const is_nan = values.left.cmp(.NE, values.left);
            const left_gt_or_nan = left_gt_right.logical(.OR, is_nan);
            // we are bubbling up Nan.
            const max_val = left_gt_or_nan.select(values.left, values.right);

            // If values.left == values.right: keep the smallest idx.
            const is_same = values.left.cmp(.EQ, values.right);
            const is_first = indices.left.cmp(.LT, indices.right);
            const is_same_but_first = is_same.logical(.AND, is_first);
            const keep_left_idx = left_gt_or_nan.logical(.OR, is_same_but_first);
            const max_idx = keep_left_idx.select(indices.left, indices.right);

            return .{ max_val, max_idx };
        }

        fn inputs(x: Tensor, dt: zml.DataType, axis_: i8) ArgMaxRes {
            return .{ .values = x, .indices = Tensor.arange(.{ .end = x.dim(axis_) }, dt).broadcast(x.shape(), &.{axis_}) };
        }

        fn inits(x: Tensor, dt: zml.DataType) ArgMaxRes {
            return .{ .values = Tensor.constant(x.dtype().minValue()), .indices = Tensor.constant(dt.zero()) };
        }

        fn operand(x: Tensor, dt: zml.DataType) ArgMaxRes {
            return .{ .values = .init(.init(&.{}, x.dtype())), .indices = .init(.init(&.{}, dt)) };
        }
    };

    pub const ReduceArgs = struct {
        left: Tensor,
        right: Tensor,
    };

    //pub fn argMax(x: Tensor, axis_: anytype) ArgMaxRes {
    //    const a = x.axis(axis_);
    //    const dt: zml.DataType = if (x.dim(a) <= std.math.maxInt(i32)) .i32 else .i64;

    //    return ops.reduce(
    //        ArgMaxRes.cmp,
    //        .{ .values = x, .indices = Tensor.arange(.{ .end = x.dim(a) }, dt).broadcast(x.shape(), &.{a}) },
    //        .{ .values = Tensor.constant(&.{}, x.dtype().minValue()), .indices = Tensor.scalar(0, dt) },
    //        &.{a},
    //    );
    //}

    pub const LogicalOp = enum { OR, XOR, AND };
    pub fn logical(self: Tensor, comptime logical_op: LogicalOp, other: Tensor) Tensor {
        return switch (logical_op) {
            .OR => binaryOp(@src(), "or", dialects.stablehlo.or_)(self, other),
            .XOR => binaryOp(@src(), "xor", dialects.stablehlo.xor)(self, other),
            .AND => binaryOp(@src(), "and", dialects.stablehlo.and_)(self, other),
        };
    }

    fn getComparisonType(dt: zml.DataType) dialects.stablehlo.CompareType.Type {
        return switch (dt) {
            .i4, .i8, .i16, .i32, .i64 => .SIGNED,
            .bool, .u4, .u8, .u16, .u32, .u64 => .UNSIGNED,
            .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2, .f8e5m2fnuz, .bf16, .f16, .f32, .f64 => .FLOAT,
            .c64, .c128 => @panic("Can't compare complex numbers"),
        };
    }

    pub fn cmp(self: Tensor, direction: dialects.stablehlo.ComparisonDirection.Direction, other: Tensor) Tensor {
        stdx.debug.assert(self.dtype() == other.dtype(), "cmp expects input tensors to be of the same type, got {t} and {t}", .{ self.dtype(), other.dtype() });

        if (self.rank() == 0 and other.rank() != 0) return self.broadcast(other._shape, &.{}).cmp(direction, other);
        if (self.rank() != 0 and other.rank() == 0) return self.cmp(direction, other.broadcast(self._shape, &.{}));

        stdx.debug.assert(self._shape.eql(other._shape), "cmp expects input tensor shapes to match, got {f} and {f}", .{ self._shape, other._shape });

        //const loc = self.getContext().location(@src(), "cmp(.{s})", .{@tagName(direction)});
        const loc = mlir.Location.unknown(mlirCtx());
        const op = dialects.stablehlo.compare(
            mlirCtx(),
            self.value(),
            other.value(),
            direction,
            getComparisonType(self.dtype()),
            loc,
        ).appendTo(CompilationContext.current().currentBlock());

        return _result(self._shape.withDtype(.bool), op.result(0));
    }

    pub fn argMax(x: Tensor, axis_: anytype) ArgMaxRes {
        const a = x.axis(axis_);
        const dt: zml.DataType = if (x.dim(a) <= std.math.maxInt(i32)) .i32 else .i64;

        //return reduce(ArgMaxRes.cmp, .inputs(x, dt, a), .inits(x, dt), .operand(x, dt), .operand(x, dt), &.{a});
        const values, const indices = reduce(
            .{ x, Tensor.arange(.{ .end = x.dim(axis_) }, dt).broadcast(x.shape(), &.{a}) },
            .{ Tensor.constant(x.dtype().minValue()), Tensor.constant(dt.zero()) },
            &.{a},
            struct {
                fn cmp(values: ReduceArgs, indices: ReduceArgs) struct { Tensor, Tensor } {
                    const left_gt_right = values.left.cmp(.GT, values.right);
                    const is_nan = values.left.cmp(.NE, values.left);
                    const left_gt_or_nan = left_gt_right.logical(.OR, is_nan);
                    // we are bubbling up Nan.
                    const max_val = left_gt_or_nan.select(values.left, values.right);

                    // If values.left == values.right: keep the smallest idx.
                    const is_same = values.left.cmp(.EQ, values.right);
                    const is_first = indices.left.cmp(.LT, indices.right);
                    const is_same_but_first = is_same.logical(.AND, is_first);
                    const keep_left_idx = left_gt_or_nan.logical(.OR, is_same_but_first);
                    const max_idx = keep_left_idx.select(indices.left, indices.right);

                    return .{ max_val, max_idx };
                }
            }.cmp,
            .{},
        );
        return .{ .values = values, .indices = indices };
    }

    fn collectValues(allocator: std.mem.Allocator, v: anytype) []const *const mlir.Value {
        const LocalContext = struct {
            list: *std.ArrayList(*const mlir.Value),
            allocator: std.mem.Allocator,
        };

        var list: std.ArrayList(*const mlir.Value) = .{};
        var context: LocalContext = .{ .list = &list, .allocator = allocator };
        zml.meta.visit(struct {
            fn cb(context_: *LocalContext, t: *const Tensor) void {
                context_.list.append(context_.allocator, t.value()) catch unreachable;
            }
        }.cb, &context, v);

        return list.toOwnedSlice(allocator) catch unreachable;
    }

    fn collectShapes(allocator: std.mem.Allocator, v: anytype) []const zml.Shape {
        const LocalContext = struct {
            list: *std.ArrayList(zml.Shape),
            allocator: std.mem.Allocator,
        };

        var list: std.ArrayList(zml.Shape) = .{};
        var context: LocalContext = .{ .list = &list, .allocator = allocator };
        zml.meta.visit(struct {
            fn cb(context_: *LocalContext, t: *const Tensor) void {
                context_.list.append(context_.allocator, t.shape()) catch unreachable;
            }
        }.cb, &context, v);

        return list.toOwnedSlice(allocator) catch unreachable;
    }

    fn mapToArguments(compilation_context: *CompilationContext, v: anytype, starting_index: usize) usize {
        const LocalContext = struct {
            current_argument_index: usize,
            compilation_context: *CompilationContext,
        };

        var context: LocalContext = .{
            .current_argument_index = starting_index,
            .compilation_context = compilation_context,
        };

        zml.meta.visit(struct {
            fn cb(context_: *LocalContext, t: *const Tensor) void {
                _ = context_.compilation_context.currentBlock().addArgument(
                    mlir.rankedTensorType(t.dims(), zml.mlir.Type.fromDType(mlirCtx(), t.dtype())),
                    .unknown(mlirCtx()),
                );
                context_.compilation_context.currentMapping().put(context_.compilation_context.allocator, t.id, context_.current_argument_index) catch unreachable;
                context_.current_argument_index += 1;
            }
        }.cb, &context, v);

        return context.current_argument_index;
    }

    fn ReduceResult(comptime T: anytype) type {
        const type_info = @typeInfo(T);
        return switch (type_info.@"struct".fields.len) {
            0 => unreachable,
            1 => type_info.@"struct".fields[0].type,
            else => |len| @Type(.{ .array = .{
                .len = len,
                .sentinel_ptr = null,
                .child = type_info.@"struct".fields[0].type,
            } }),
        };
    }

    fn reduce(inputs: anytype, inits: anytype, axes: []const i64, comptime func: anytype, context: anytype) stdx.meta.FnResult(func) {
        var arena = std.heap.ArenaAllocator.init(CompilationContext.current().allocator);
        defer arena.deinit();

        const reduce_block, var result = b: {
            const ArgsType = std.meta.Tuple(&[1]type{ReduceArgs} ** inits.len);
            var args: ArgsType = undefined;
            var block_types: [2 * inits.len]*const mlir.Type = undefined;

            inline for (0..inits.len) |i| {
                args[i].left = Tensor.init(inits[i].shape());
                args[i].right = Tensor.init(inits[i].shape());

                block_types[i] = mlir.rankedTensorType(args[i].left.dims(), zml.mlir.Type.fromDType(mlirCtx(), args[i].left.dtype()));
                block_types[i + inits.len] = mlir.rankedTensorType(args[i].right.dims(), zml.mlir.Type.fromDType(mlirCtx(), args[i].right.dtype()));
            }

            const block_locs: [2 * inits.len]*const mlir.Location = @splat(mlir.Location.unknown(mlirCtx()));
            const reduce_block = mlir.Block.init(&block_types, &block_locs);
            errdefer reduce_block.deinit();

            CompilationContext.current().pushBlock(reduce_block);
            defer CompilationContext.current().popBlock();

            inline for (0..inits.len) |i| {
                CompilationContext.current().currentMapping().put(CompilationContext.current().allocator, args[i].left.id, i) catch unreachable;
                CompilationContext.current().currentMapping().put(CompilationContext.current().allocator, args[i].right.id, i + inits.len) catch unreachable;
            }

            var result = @call(.auto, func, args ++ context);

            const result_values = collectValues(arena.allocator(), &result);

            _ = dialects.stablehlo.returns(mlirCtx(), result_values, .unknown(mlirCtx())).appendTo(reduce_block);
            break :b .{ reduce_block, result };
        };
        var input_values: [inputs.len]*const mlir.Value = undefined;
        inline for (0..inputs.len) |i| {
            input_values[i] = inputs[i].value();
        }

        var init_values: [inputs.len]*const mlir.Value = undefined;
        inline for (0..inits.len) |i| init_values[i] = inits[i].value();

        const reduce_op = mlir.Operation.make(mlirCtx(), "stablehlo.reduce", .{
            .operands = .{ .variadic = &.{ &input_values, &init_values } },
            .result_type_inference = true,
            .blocks = &.{reduce_block},
            .attributes = &.{
                .named(mlirCtx(), "dimensions", mlir.denseArrayAttribute(mlirCtx(), .i64, axes)),
            },
            .verify = false,
            .location = .unknown(mlirCtx()),
        }).appendTo(CompilationContext.current().currentBlock());

        inline for (0..result.len) |i| {
            const val = reduce_op.result(i);
            const reduced_shape = inputs[i].shape().removeMany(axes);

            result[i] = ._result(reduced_shape, val).broad(inputs[i].shape());
        }

        return result;
    }

    fn reduceOld(comptime func: anytype, inputs: anytype, inits: anytype, axes: []const i64) ReduceResult(@TypeOf(inputs)) {
        var arena = std.heap.ArenaAllocator.init(CompilationContext.current().allocator);
        defer arena.deinit();

        const reduce_block = b: {
            const reduce_block = mlir.Block.init(&.{}, &.{});
            errdefer reduce_block.deinit();

            CompilationContext.current().pushBlock(reduce_block);
            defer CompilationContext.current().popBlock();

            var lhs: [inputs.len]Tensor = undefined;
            var rhs: [inputs.len]Tensor = undefined;
            inline for (0..inits.len) |i| {
                lhs[i] = Tensor.init(inits[i].shape());
                rhs[i] = Tensor.init(inits[i].shape());
            }
            const arg_count = mapToArguments(CompilationContext.current(), &lhs, 0);
            _ = mapToArguments(CompilationContext.current(), &rhs, arg_count);

            var result = if (inputs.len == 1) func(lhs[0], rhs[0]) else func(lhs, rhs);

            const result_values = collectValues(arena.allocator(), &result);

            _ = dialects.stablehlo.returns(mlirCtx(), result_values, .unknown(mlirCtx())).appendTo(reduce_block);
            break :b reduce_block;
        };

        const input_values = collectValues(arena.allocator(), &inputs);
        const init_values = collectValues(arena.allocator(), &inits);

        const reduce_op = mlir.Operation.make(mlirCtx(), "stablehlo.reduce", .{
            .operands = .{ .variadic = &.{ input_values, init_values } },
            .result_type_inference = true,
            .blocks = &.{reduce_block},
            .attributes = &.{
                .named(mlirCtx(), "dimensions", mlir.denseArrayAttribute(mlirCtx(), .i64, axes)),
            },
            .verify = false,
            .location = .unknown(mlirCtx()),
        }).appendTo(CompilationContext.current().currentBlock());

        var broadcasting_axes: stdx.BoundedArray(i64, MAX_RANK) = .{};
        for (0..MAX_RANK) |i| {
            if (std.mem.indexOfScalar(i64, axes, @intCast(i)) == null) {
                broadcasting_axes.append(@intCast(i)) catch unreachable;
            }
        }

        const LocalContext = struct {
            axes: []const i64,
            broadcasting_axes: []const i64,
            n_reduced: u8,
            current_id: usize = 0,
            reduce_op: *mlir.Operation,
        };
        var context2: LocalContext = .{
            .axes = axes,
            .broadcasting_axes = broadcasting_axes.constSlice(),
            .n_reduced = @intCast(axes.len),
            .reduce_op = reduce_op,
        };
        var result: ReduceResult(@TypeOf(inputs)) = inputs;
        zml.meta.visit(struct {
            fn cb(context_: *LocalContext, t: *Tensor) void {
                const val = context_.reduce_op.result(context_.current_id);
                var reduced_shape = t.shape();
                for (context_.axes) |a| {
                    reduced_shape = reduced_shape.setDim(a, 1);
                }

                const broadcast_op = dialects.stablehlo.broadcast_in_dim(
                    mlirCtx(),
                    val,
                    context_.broadcasting_axes[0 .. t.rank() - context_.n_reduced],
                    mlir.rankedTensorType(reduced_shape.dims(), zml.mlir.Type.fromDType(mlirCtx(), reduced_shape.dtype())),
                    .unknown(mlirCtx()),
                ).appendTo(CompilationContext.current().currentBlock());
                t.* = Tensor._result(reduced_shape, broadcast_op.result(0));
                context_.current_id += 1;
            }
        }.cb, &context2, &result);

        return result;
    }

    fn reduceOld2(
        comptime func: anytype,
        inputs: stdx.meta.FnParam(func, 0),
        inits: stdx.meta.FnParam(func, 0),
        lhs: stdx.meta.FnParam(func, 0),
        rhs: stdx.meta.FnParam(func, 0),
        axes: []const i64,
    ) stdx.meta.FnResult(func) {
        var arena = std.heap.ArenaAllocator.init(CompilationContext.current().allocator);
        defer arena.deinit();

        const reduce_block = b: {
            const reduce_block = mlir.Block.init(&.{}, &.{});
            errdefer reduce_block.deinit();

            CompilationContext.current().pushBlock(reduce_block);
            defer CompilationContext.current().popBlock();

            const arg_count = mapToArguments(CompilationContext.current(), &lhs, 0);
            _ = mapToArguments(CompilationContext.current(), &rhs, arg_count);

            var result = func(lhs, rhs);

            const result_values = collectValues(arena.allocator(), &result);

            _ = dialects.stablehlo.returns(mlirCtx(), result_values, .unknown(mlirCtx())).appendTo(reduce_block);
            break :b reduce_block;
        };

        const input_values = collectValues(arena.allocator(), &inputs);
        const init_values = collectValues(arena.allocator(), &inits);

        const reduce_op = mlir.Operation.make(mlirCtx(), "stablehlo.reduce", .{
            .operands = .{ .variadic = &.{ input_values, init_values } },
            .result_type_inference = true,
            .blocks = &.{reduce_block},
            .attributes = &.{
                .named(mlirCtx(), "dimensions", mlir.denseArrayAttribute(mlirCtx(), .i64, axes)),
            },
            .verify = false,
            .location = .unknown(mlirCtx()),
        }).appendTo(CompilationContext.current().currentBlock());

        var broadcasting_axes: stdx.BoundedArray(i64, MAX_RANK) = .{};
        for (0..MAX_RANK) |i| {
            if (std.mem.indexOfScalar(i64, axes, @intCast(i)) == null) {
                broadcasting_axes.append(@intCast(i)) catch unreachable;
            }
        }

        const LocalContext = struct {
            axes: []const i64,
            broadcasting_axes: []const i64,
            n_reduced: u8,
            current_id: usize = 0,
            reduce_op: *mlir.Operation,
        };
        var context2: LocalContext = .{
            .axes = axes,
            .broadcasting_axes = broadcasting_axes.constSlice(),
            .n_reduced = @intCast(axes.len),
            .reduce_op = reduce_op,
        };
        var result = inputs;
        zml.meta.visit(struct {
            fn cb(context_: *LocalContext, t: *Tensor) void {
                const val = context_.reduce_op.result(context_.current_id);
                var reduced_shape = t.shape();
                for (context_.axes) |a| {
                    reduced_shape = reduced_shape.setDim(a, 1);
                }

                const broadcast_op = dialects.stablehlo.broadcast_in_dim(
                    mlirCtx(),
                    val,
                    context_.broadcasting_axes[0 .. t.rank() - context_.n_reduced],
                    mlir.rankedTensorType(reduced_shape.dims(), zml.mlir.Type.fromDType(mlirCtx(), reduced_shape.dtype())),
                    .unknown(mlirCtx()),
                ).appendTo(CompilationContext.current().currentBlock());
                t.* = Tensor._result(reduced_shape, broadcast_op.result(0));
                context_.current_id += 1;
            }
        }.cb, &context2, &result);

        return result;
    }

    fn reduceOld3(comptime func: anytype, inputs: stdx.meta.FnParam(func, 0), inits: stdx.meta.FnParam(func, 0), axes: []const i64) stdx.meta.FnResult(func) {
        var arena = std.heap.ArenaAllocator.init(CompilationContext.current().allocator);
        defer arena.deinit();

        const mlir_ctx = CompilationContext.current().mlir_ctx;

        const input_values, const input_shapes = b: {
            var values_list: std.ArrayList(*const mlir.Value) = .{};
            var shapes_list: std.ArrayList(zml.Shape) = .{};
            const LocalContext = struct {
                values_list: *std.ArrayList(*const mlir.Value),
                shapes_list: *std.ArrayList(zml.Shape),
                allocator: std.mem.Allocator,
            };
            var context: LocalContext = .{
                .values_list = &values_list,
                .shapes_list = &shapes_list,
                .allocator = arena.allocator(),
            };
            zml.meta.visit(struct {
                fn cb(context_: *LocalContext, t: *const Tensor) void {
                    context_.values_list.append(context_.allocator, t.value()) catch unreachable;
                    context_.shapes_list.append(context_.allocator, t.shape()) catch unreachable;
                }
            }.cb, &context, &inputs);
            break :b .{
                values_list.toOwnedSlice(arena.allocator()) catch unreachable,
                shapes_list.toOwnedSlice(arena.allocator()) catch unreachable,
            };
        };
        const init_values = b: {
            var list: std.ArrayList(*const mlir.Value) = .{};
            const LocalContext = struct {
                list: *std.ArrayList(*const mlir.Value),
                allocator: std.mem.Allocator,
            };
            var context: LocalContext = .{ .list = &list, .allocator = arena.allocator() };
            zml.meta.visit(struct {
                fn cb(context_: *LocalContext, t: *const Tensor) void {
                    context_.list.append(context_.allocator, t.value()) catch unreachable;
                }
            }.cb, &context, &inits);
            break :b list.toOwnedSlice(arena.allocator()) catch unreachable;
        };

        const reduce_block, var result = b2: {
            const reduce_block = mlir.Block.init(&.{}, &.{});
            errdefer reduce_block.deinit();

            CompilationContext.current().pushBlock(reduce_block);
            defer CompilationContext.current().popBlock();

            const LocalContext = struct {
                current_argument_id: usize = 0,
            };
            var context: LocalContext = .{};

            var lhs = inputs;
            zml.meta.visit(struct {
                fn cb(context_: *LocalContext, t: *Tensor) void {
                    const dt = t.dtype();
                    t.* = .init(.init(&.{}, dt));
                    _ = CompilationContext.current().currentBlock().addArgument(
                        mlir.rankedTensorType(t.dims(), zml.mlir.Type.fromDType(CompilationContext.current().mlir_ctx, t.dtype())),
                        .unknown(CompilationContext.current().mlir_ctx),
                    );
                    CompilationContext.current().currentMapping().put(CompilationContext.current().allocator, t.id, context_.current_argument_id) catch unreachable;
                    context_.current_argument_id += 1;
                }
            }.cb, &context, &lhs);

            var rhs = inits;
            zml.meta.visit(struct {
                fn cb(context_: *LocalContext, t: *Tensor) void {
                    const dt = t.dtype();
                    t.* = .init(.init(&.{}, dt));
                    _ = CompilationContext.current().currentBlock().addArgument(
                        mlir.rankedTensorType(t.dims(), zml.mlir.Type.fromDType(CompilationContext.current().mlir_ctx, t.dtype())),
                        .unknown(CompilationContext.current().mlir_ctx),
                    );
                    CompilationContext.current().currentMapping().put(CompilationContext.current().allocator, t.id, context_.current_argument_id) catch unreachable;
                    context_.current_argument_id += 1;
                }
            }.cb, &context, &rhs);

            var result = func(lhs, rhs);
            const result_values = b: {
                var list: std.ArrayList(*const mlir.Value) = .{};
                const LocalContext2 = struct {
                    list: *std.ArrayList(*const mlir.Value),
                    allocator: std.mem.Allocator,
                };
                var context2: LocalContext2 = .{ .list = &list, .allocator = arena.allocator() };
                zml.meta.visit(struct {
                    fn cb(context_: *LocalContext2, t: *const Tensor) void {
                        context_.list.append(context_.allocator, t.value()) catch unreachable;
                    }
                }.cb, &context2, &result);
                break :b list.toOwnedSlice(arena.allocator()) catch unreachable;
            };
            _ = dialects.stablehlo.returns(mlir_ctx, result_values, .unknown(mlir_ctx)).appendTo(reduce_block);

            break :b2 .{ reduce_block, result };
        };
        const reduce_op = mlir.Operation.make(CompilationContext.current().mlir_ctx, "stablehlo.reduce", .{
            .operands = .{ .variadic = &.{ input_values, init_values } },
            .result_type_inference = true,
            .blocks = &.{reduce_block},
            .attributes = &.{
                .named(CompilationContext.current().mlir_ctx, "dimensions", mlir.denseArrayAttribute(CompilationContext.current().mlir_ctx, .i64, axes)),
            },
            .verify = false,
            .location = .unknown(CompilationContext.current().mlir_ctx),
        }).appendTo(CompilationContext.current().currentBlock());

        var broadcasting_axes: stdx.BoundedArray(i64, MAX_RANK) = .{};
        for (0..MAX_RANK) |i| {
            if (std.mem.indexOfScalar(i64, axes, @intCast(i)) == null) {
                broadcasting_axes.append(@intCast(i)) catch unreachable;
            }
        }

        const LocalContext2 = struct {
            axes: []const i64,
            broadcasting_axes: []const i64,
            n_reduced: u8,
            current_id: usize = 0,
            reduce_op: *mlir.Operation,
            input_shapes: []const zml.Shape,
        };
        var context2: LocalContext2 = .{
            .axes = axes,
            .broadcasting_axes = broadcasting_axes.constSlice(),
            .n_reduced = @intCast(axes.len),
            .reduce_op = reduce_op,
            .input_shapes = input_shapes,
        };
        zml.meta.visit(struct {
            fn cb(context_: *LocalContext2, t: *Tensor) void {
                const mlir_ctx_ = CompilationContext.current().mlir_ctx;

                const val = context_.reduce_op.result(context_.current_id);
                var reduced_shape = context_.input_shapes[context_.current_id];
                for (context_.axes) |a| {
                    reduced_shape = reduced_shape.setDim(a, 1);
                }
                const broadcast_op = dialects.stablehlo.broadcast_in_dim(
                    mlir_ctx_,
                    val,
                    context_.broadcasting_axes[0 .. context_.input_shapes[context_.current_id].rank() - context_.n_reduced],
                    mlir.rankedTensorType(reduced_shape.dims(), zml.mlir.Type.fromDType(mlir_ctx_, reduced_shape.dtype())),
                    .unknown(mlir_ctx_),
                ).appendTo(CompilationContext.current().currentBlock());
                t.* = Tensor._result(reduced_shape, broadcast_op.result(0));
                context_.current_id += 1;
            }
        }.cb, &context2, &result);

        return result;
    }

    fn binaryOp(
        src: std.builtin.SourceLocation,
        op_name: []const u8,
        comptime op_fn: fn (*mlir.Context, *const mlir.Value, *const mlir.Value, *const mlir.Location) *mlir.Operation,
    ) fn (Tensor, Tensor) Tensor {
        _ = op_name; // autofix
        _ = src; // autofix
        return struct {
            pub fn binaryOpHelper(self: Tensor, other: Tensor) Tensor {
                const other_ = if (other.auto_broadcast) other.broad(self.shape()) else other;
                //stdx.debug.assert(self.dtype() == other.dtype(), "{s} expects tensor to be of same type, got {f} and {f}", .{ op_name, self, other });

                //if (self.rank() == 0 and other.rank() != 0) {
                //    return binaryOpHelper(self.broad(other._shape), other);
                //}

                //if (self.rank() != 0 and other.rank() == 0) {
                //    return binaryOpHelper(self, other.broad(self._shape));
                //}

                //stdx.debug.assert(self._shape.eql(other._shape), "{s} expects tensor shapes to match, got {f} and {f}", .{ op_name, self._shape, other._shape });

                const ctx = self.getContext();
                //const location = ctx.location(src, "{s}({f}, {f})", .{ op_name, self, other });
                const ret = op_fn(mlirCtx(), self.value(), other_.value(), .unknown(ctx.mlir_ctx)).appendTo(currentBlock());
                return _result(self._shape, ret.result(0));
            }
        }.binaryOpHelper;
    }

    pub fn value(self: Tensor) *const mlir.Value {
        return switch (self.tensor_origin) {
            .argument => b: {
                const argument_index = self.getContext().getArgumentIndex(self.id);
                break :b self.getContext().currentBlock().argument(argument_index);
            },
            .value => |v| v,
        };
    }

    pub fn getContext(self: Tensor) *CompilationContext {
        _ = self; // autofix
        return global_compilation_context.?;
    }

    pub fn format(
        self: Tensor,
        writer: *std.Io.Writer,
    ) !void {
        // TODO(0.15.0) handle format
        // const bare_fmt = fmt.len == 1 and fmt[0] == '_';
        try writer.print("{s}({f})", .{ @typeName(Tensor), self._shape });
    }
};

/// Model definition
const Mnist = struct {
    layers: [2]Layer,

    const Layer = struct {
        weight: Tensor,
        bias: Tensor,

        pub fn forward(self: Layer, input: Tensor) Tensor {
            return self.weight.matmul(input).add(self.bias).relu();
        }
    };

    pub fn init(buffer_store: zml.aio.BufferStore) Mnist {
        return .{
            .layers = .{
                .{
                    .weight = .init(buffer_store.get("fc1.weight").?.shape()),
                    .bias = .init(buffer_store.get("fc1.bias").?.shape()),
                },
                .{
                    .weight = .init(buffer_store.get("fc2.weight").?.shape()),
                    .bias = .init(buffer_store.get("fc2.bias").?.shape()),
                },
            },
        };
    }

    /// just two linear layers + relu activation
    pub fn forward(self: Mnist, input: Tensor) Tensor {
        // std.log.info("Compiling for target: {s}", .{@tagName(input.getContext().target())});
        var x = input.flattenAll().convert(.f32);
        for (self.layers) |layer| {
            x = layer.forward(x);
        }
        return x.argMax(0).indices.convert(.u8);
        //return x;
    }
};

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    const allocator = std.heap.c_allocator;

    const process_args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, process_args);
    const pt_model = process_args[1];
    const t10kfilename = process_args[2];
    _ = t10kfilename; // autofix

    // Read model shapes.
    // Note this works because Mnist struct uses the same layer names as the pytorch model
    var buffer_store = try zml.aio.detectFormatAndOpen(allocator, pt_model);
    defer buffer_store.deinit();

    const mnist: Mnist = .init(buffer_store);
    _ = mnist; // autofix

    const input: Tensor = .init(zml.Shape.init(.{ 28, 28 }, .u8));
    _ = input; // autofix

    //compileModel(allocator, Mnist.forward, mnist, .{input});

    compile(allocator, testDot, .{ Tensor.init(.init(.{ .m = 256, .n = 128 }, .f32)), Tensor.init(.init(.{ .b = 16, .n = 128, .k = 256 }, .f32)) });
}

fn testDot(a: Tensor, b: Tensor) Tensor {
    _ = a.dot2(b, .n);
    _ = a.dot2(b, .{ .n = .k });
    _ = a.dot2(b, .{ .contracting = .{ .n = .k }, .batching = .{.b} });
    _ = a.dot2(b, .{ .contracting = .{.n}, .batching = .{.b} });
    unreachable;
    //return a.dotGeneral(b, &.{.{ 1, 1 }}, &.{});
}

const CompilationContext = struct {
    allocator: std.mem.Allocator,

    mlir_registry: *mlir.DialectRegistry,
    mlir_ctx: *mlir.Context,
    module: *mlir.Module,

    blocks: stdx.BoundedArray(*mlir.Block, 16) = .{},
    mappings: stdx.BoundedArray(std.AutoArrayHashMapUnmanaged(usize, usize), 16) = .{},

    pub fn init(allocator: std.mem.Allocator) CompilationContext {
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

        return .{
            .allocator = allocator,
            .mlir_registry = mlir_registry,
            .mlir_ctx = mlir_ctx,
            .module = module,
        };
    }

    pub fn deinit(self: *CompilationContext) void {
        self.module.deinit();
        self.mlir_ctx.deinit();
        self.mlir_registry.deinit();
    }

    //pub fn registerTensor(self: *CompilationContext, tensor_id: usize, argument_id: usize) !void {
    //    const gop = try self.mapping.getOrPut(self.allocator, tensor_id);
    //    std.debug.assert(!gop.found_existing);

    //    gop.value_ptr.* = argument_id;
    //}

    pub fn currentBlock(self: *const CompilationContext) *mlir.Block {
        return self.blocks.get(self.blocks.len - 1);
    }

    pub fn currentMapping(self: *CompilationContext) *std.AutoArrayHashMapUnmanaged(usize, usize) {
        return &self.mappings.slice()[self.blocks.len - 1];
    }

    pub fn getArgumentIndex(self: *CompilationContext, tensor_id: usize) usize {
        return self.currentMapping().get(tensor_id).?;
    }

    pub fn pushBlock(self: *CompilationContext, block: *mlir.Block) void {
        self.blocks.appendAssumeCapacity(block);
        self.mappings.appendAssumeCapacity(.{});
    }

    pub fn popBlock(self: *CompilationContext) void {
        _ = self.blocks.pop();
        var maybe_popped = self.mappings.pop();
        if (maybe_popped) |*popped| {
            popped.deinit(self.allocator);
        }
    }

    pub fn current() *CompilationContext {
        return global_compilation_context.?;
    }
};

const Context = struct {
    compilation_context: *CompilationContext,
    current_id: usize = 0,
};

fn compileModel(allocator: std.mem.Allocator, comptime func: anytype, model: stdx.meta.Head(stdx.meta.FnArgs(func)), args: stdx.meta.Tail(stdx.meta.FnArgs(func))) void {
    compile(allocator, func, .{model} ++ args);
}

fn compile(allocator: std.mem.Allocator, comptime func: anytype, args: stdx.meta.FnArgs(func)) void {
    var compilation_context: CompilationContext = .init(allocator);
    defer compilation_context.deinit();

    emitMlir(&compilation_context, func, args) catch unreachable;
}

fn emitMlir(compilation_context: *CompilationContext, comptime func: anytype, args: stdx.meta.FnArgs(func)) !void {
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
    zml.meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *const Tensor) void {
            _ = ctx_.compilation_context.currentBlock().addArgument(
                tensor.mlirType(ctx_.compilation_context.mlir_ctx),
                .unknown(ctx_.compilation_context.mlir_ctx),
            );
            ctx_.compilation_context.currentMapping().put(ctx_.compilation_context.allocator, tensor.id, ctx_.current_argument_id) catch unreachable;
            ctx_.current_argument_id += 1;
        }
    }.cb, &context, &args);

    {
        global_compilation_context = compilation_context;
        defer global_compilation_context = null;
        const result = @call(.auto, func, args);
        _ = dialects.func.return_(compilation_context.mlir_ctx, result.value(), .unknown(compilation_context.mlir_ctx)).appendTo(compilation_context.currentBlock());
    }

    const mlir_fn = dialects.func.func(compilation_context.mlir_ctx, .{
        .name = "main",
        .block = compilation_context.currentBlock(),
        .location = .unknown(compilation_context.mlir_ctx),
    });

    var pass_manager = mlir.PassManager.init(compilation_context.mlir_ctx);
    pass_manager.enableIRPrinting(.{
        .printBeforeAll = true,
    });
    {
        var opm = pass_manager.asOpPassManager();
        const passes: []const []const u8 = &.{
            "canonicalize",
            "cse",
            "canonicalize",
        };
        for (passes) |pass| {
            try opm.addPipeline(pass);
        }
    }

    pass_manager.runOnOp(mlir_fn) catch |err| {
        std.debug.print("Failed to optimize module: {any}\n", .{err});
        return err;
    };

    _ = mlir_fn.appendTo(module.body());

    std.debug.print("{f}\n", .{module});
}
