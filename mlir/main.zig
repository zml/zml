const std = @import("std");

const llama = @import("llama.zig");

const c = zml.c;
const asynk = @import("async");
const dialects = @import("mlir/dialects");
const mlir = @import("mlir");
const stdx = @import("stdx");
const upb = zml.upb;
const zml = @import("zml");
const pjrt = zml.pjrt;
const Shape = zml.Shape;
const EnumLiteral = @TypeOf(.enum_literal);
const Platform = zml.Platform;

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

    /// Returns the shape of a Tensor.
    pub fn shape(self: Tensor) Shape {
        return self._shape;
    }

    /// Returns the datatype of a Tensor.
    pub fn dtype(self: Tensor) zml.DataType {
        return self._shape.dtype();
    }

    /// Returns the rank of a Tensor.
    pub fn rank(self: Tensor) u4 {
        return self._shape.rank();
    }

    /// Returns the number of element of a Tensor.
    pub fn count(self: Tensor) usize {
        return self._shape.count();
    }

    /// Returns the size in bytes of a Tensor.
    pub fn byteSize(self: Tensor) usize {
        return self._shape.byteSize();
    }

    /// Returns the dimension of axis 'axis_'.
    ///
    /// 'axis_' can be an integer or a tag.
    pub fn dim(self: Tensor, axis_: anytype) i64 {
        return self._shape.dim(axis_);
    }

    /// Returns the dimensions of a Tensor as a slice.
    pub fn dims(self: *const Tensor) []const i64 {
        return self._shape.dims();
    }

    /// Returns the index of axis 'axis_'.
    ///
    /// 'axis_' can be an integer or a tag.
    pub fn axis(self: Tensor, axis_: anytype) u3 {
        return self._shape.axis(axis_);
    }

    /// Returns the indices of each of the given axes.
    ///
    /// 'axis_' can be an integer or a tag.
    pub fn axes(self: Tensor, axes_: anytype) stdx.BoundedArray(u3, Tensor.MAX_RANK) {
        return self._shape.axes(axes_);
    }

    /// Returns a Tensor tagged with the tags in 'tagz'.
    pub fn withTags(self: Tensor, tagz: anytype) Tensor {
        var res = self;
        res._shape = self._shape.withTags(tagz);
        return res;
    }

    /// Returns a Tensor tagged partially with the tags in 'tagz'.
    ///
    /// If 'tagz' is of length n, the n last dimensions of the Tensor will be tagged.
    pub fn withPartialTags(self: Tensor, tagz: anytype) Tensor {
        var res = self;
        res._shape = self._shape.withPartialTags(tagz);
        return res;
    }

    /// Returns a Tensor with new tag names.
    pub fn rename(self: Tensor, renames: anytype) Tensor {
        var res = self;
        res._shape = self._shape.rename(renames);
        return res;
    }

    pub fn renameAxis(self: Tensor, ax: i8, name: EnumLiteral) Tensor {
        var res = self;
        res._shape._tags.set(self.axis(ax), @tagName(name).ptr);
        return res;
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

    pub fn dot(lhs: Tensor, rhs: Tensor, args: anytype) Tensor {
        //const args_kind = getArgsKind(@TypeOf(args));
        //std.debug.print("{}\n", .{args_kind});
        const lhs_contracting_dim: i8 = @intCast(lhs.shape().hasTag(args).?);
        const rhs_contracting_dim: i8 = @intCast(rhs.shape().hasTag(args).?);
        return lhs.dotGeneral(rhs, &.{.{ lhs_contracting_dim, rhs_contracting_dim }}, &.{});
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

    pub fn mul(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "mul", dialects.stablehlo.multiply)(self, other);
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
    };

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

    pub const ReduceArgs = struct {
        left: Tensor,
        right: Tensor,
    };

    fn reduce(inputs: anytype, inits: anytype, axes_: []const i64, comptime func: anytype, context: anytype) stdx.meta.FnResult(func) {
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

            var result_values: [inits.len]*const mlir.Value = undefined;
            inline for (0..inits.len) |i| {
                result_values[i] = result[i].value();
            }

            _ = dialects.stablehlo.returns(mlirCtx(), &result_values, .unknown(mlirCtx())).appendTo(reduce_block);
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
                .named(mlirCtx(), "dimensions", mlir.denseArrayAttribute(mlirCtx(), .i64, axes_)),
            },
            .verify = false,
            .location = .unknown(mlirCtx()),
        }).appendTo(CompilationContext.current().currentBlock());

        inline for (0..result.len) |i| {
            const val = reduce_op.result(i);
            var reduced_shape: Shape = inputs[i].shape();
            for (axes_) |a| {
                reduced_shape = reduced_shape.setDim(a, 1);
            }

            result[i] = Tensor._result(result[i].shape(), val).broad(reduced_shape);
        }

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

    pub fn concatenate(tensors: []const Tensor, axis_: anytype) Tensor {
        if (tensors.len == 1) return tensors[0];
        stdx.debug.assert(tensors.len <= 32, "concatenate only supports up to 32 tensors, got {}", .{tensors.len});
        var buffer: [32]*const mlir.Value = undefined;
        std.debug.assert(tensors.len <= buffer.len);
        std.debug.assert(tensors.len > 0);
        const a = tensors[0].axis(axis_);
        // TODO(Corendos): Check that tensor axes match.

        var concatenated_dim: i64 = 0;
        for (tensors, 0..) |t, i| {
            buffer[i] = t.value();
            concatenated_dim += t.dim(a);
        }

        const res_shape = tensors[0]._shape.set(a, concatenated_dim);
        const ctx = tensors[0].getContext();
        const op = dialects.stablehlo.concatenate(ctx.mlir_ctx, buffer[0..tensors.len], a, .unknown(ctx.mlir_ctx)).appendTo(currentBlock());
        // log.debug("concatenate({}, {}, {d}) -> {d}", .{ tensors[0], tensors[1], a, res_shape });
        return _result(res_shape, op.result(0));
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

const MoE = struct {
    experts: Mlp,
    router: zml.nn.Linear,

    pub fn init(tensor_descriptor: TensorDescriptor.View) MoE {
        return .{
            .experts = .{
                .gate_up_proj = .{
                    // We need to bitcast the scale cause safetensors doesn't encode f8 types correctly
                    .scale = tensor_descriptor.getWithTags("gate_up_proj_scales", .{ .expert, .out, .d }).?,
                    // We don't bitcast here because PJRT doesn't handle packed host buffers
                    .blocks = tensor_descriptor.getWithTags("gate_up_proj_blocks", .{ .expert, .out, .d, .d_block }),
                    .blocks_dtype = .f4e2m1,
                    .bias = tensor_descriptor.getWithTags("gate_up_proj_bias", .{ .expert, .d }),
                },
                .down_proj = .{
                    .blocks = tensor_descriptor.getWithTags("down_proj_blocks", .{ .expert, .out, .d, .d_block }),
                    .blocks_dtype = .f4e2m1,
                    .scale = tensor_descriptor.getWithTags("down_proj_scales", .{ .expert, .out, .d }),
                    .bias = tensor_descriptor.getWithTags("down_proj_bias", .{ .expert, .d }),
                },
            },
            .router = .{
                .weight = tensor_descriptor.getWithTags("router.weight", .{ .dout, .d }).?,
                .bias = tensor_descriptor.getWithTags("router.bias", .{.d}).?,
            },
        };
    }

    pub fn loadBuffers(allocator: std.mem.Allocator, moe: MoE, buffer_store: BufferStore4.View, platform: Platform) !Bufferized(MoE) {
        var transfer, const entries = try multiTransfer(allocator, .{
            moe.experts.gate_up_proj.scale.shape(),
            moe.experts.gate_up_proj.blocks.shape(),
            moe.experts.gate_up_proj.bias.shape(),
            moe.experts.down_proj.scale.shape(),
            moe.experts.down_proj.blocks.shape(),
            moe.experts.down_proj.bias.shape(),
            moe.router.weight.shape(),
            moe.router.bias.shape(),
        }, platform);
        defer transfer.deinit(allocator, platform);

        const gate_up_proj_scales, const gate_up_proj_blocks, const gate_up_proj_bias, const down_proj_scales, const down_proj_blocks, const down_proj_bias, const router_weight, const router_bias = entries;
        buffer_store.get("gate_up_proj_scales").?.reader().stream(gate_up_proj_scales.writer);
        buffer_store.get("gate_up_proj_blocks").?.reader().stream(gate_up_proj_blocks.writer);
        buffer_store.get("gate_up_proj_bias").?.reader().stream(gate_up_proj_bias.writer);
        buffer_store.get("down_proj_scales").?.reader().stream(down_proj_scales.writer);
        buffer_store.get("down_proj_blocks").?.reader().stream(down_proj_blocks.writer);
        buffer_store.get("down_proj_bias").?.reader().stream(down_proj_bias.writer);
        buffer_store.get("router.weight").?.reader().stream(router_weight.writer);
        buffer_store.get("router.bias").?.reader().stream(router_bias.writer);

        return .{
            .experts = .{
                .gate_up_proj = .{
                    .scale = gate_up_proj_scales,
                    .blocks = gate_up_proj_blocks,
                    .bias = gate_up_proj_bias,
                },
                .down_proj = .{
                    .scale = down_proj_scales,
                    .blocks = down_proj_blocks,
                    .bias = down_proj_bias,
                },
            },
            .router = .{
                .weight = router_weight,
                .bias = router_bias,
            },
        };
    }
};

pub const BlockScaledLinear = struct {
    blocks: Tensor,
    scale: Tensor,
    bias: ?Tensor = null,
    blocks_dtype: zml.DataType,
};

pub const Mlp = struct {
    gate_up_proj: BlockScaledLinear, // {.out = intermediate_size * 2, .d = hidden_size / block_size, .d_block = block_size }
    down_proj: BlockScaledLinear, // {.out = hidden_size * 2, .d = intermediate_size / block_size, .d_block = block_size }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const dt = x.dtype();
        var gate, var up = zml.nn.splitRealImg(self.gate_up_proj.forward(x), .interleaved);
        gate = .minimum(gate, .scalar(7, dt));
        up = .clamp(up, .scalar(-7, dt), .scalar(7, dt));

        const out = gate.quickGelu().mul(up.addConstant(1));
        return zml.call(self.down_proj, .forward, .{out});
    }

    pub fn format(self: Mlp, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("Mlp(gate_up_proj=.{f}, down_proj=.{f})", .{ self.gate_up_proj, self.down_proj });
    }
};

/// Model definition
const Mnist = struct {
    layers: [2]Layer,

    const Layer = struct {
        weight: Tensor,
        bias: Tensor,

        pub fn init(tensor_descriptor: TensorDescriptor.View) Layer {
            return .{
                .weight = tensor_descriptor.getWithTags("weight", .{ .dout, .d }).?,
                .bias = tensor_descriptor.getWithTags("bias", .{.d}).?,
            };
        }

        pub fn forward(self: Layer, input: Tensor) Tensor {
            return self.weight.dot(input, .d).add(self.bias).relu();
        }

        pub fn loadBuffers(allocator: std.mem.Allocator, layer: Layer, buffer_store: BufferStore4.View, platform: Platform) !Bufferized(Layer) {
            var transfer, const entries = try multiTransfer(allocator, .{ layer.weight.shape(), layer.bias.shape() }, platform);
            defer transfer.deinit(allocator, platform);

            const weight, const bias = entries;
            buffer_store.get("weight").?.reader().stream(weight.writer);
            buffer_store.get("bias").?.reader().stream(bias.writer);

            return .{
                .weight = weight.buffer,
                .bias = bias.buffer,
            };
        }
    };

    pub fn init(tensor_descriptor: TensorDescriptor.View) Mnist {
        return .{ .layers = .{
            Layer.init(tensor_descriptor.withPrefix("fc1")),
            Layer.init(tensor_descriptor.withPrefix("fc2")),
        } };
    }

    /// just two linear layers + relu activation
    pub fn forward(self: Mnist, input: Tensor) Tensor {
        var x = input.flattenAll().convert(.f32).withTags(.{.d});
        std.debug.print("x.shape: {f}\n", .{x.shape()});
        for (self.layers) |layer| {
            x = layer.forward(x).withTags(.{.d});
        }
        return x.argMax(0).indices.convert(.u8);
    }

    pub fn loadBuffers(allocator: std.mem.Allocator, model: Mnist, buffer_store: BufferStore4.View, platform: Platform) !Bufferized(Mnist) {
        return .{
            .layers = .{
                try Layer.loadBuffers(allocator, model.layers[0], buffer_store.withPrefix("fc1"), platform),
                //try Layer.loadBuffers(allocator, model.layers[1], buffer_store.withPrefix("fc2"), platform),
                //try autoLoad(allocator, model.layers[1], buffer_store.withPrefix("fc2"), platform),
                try loadBuffersFromId(allocator, model.layers[1], buffer_store.withPrefix("fc2"), platform),
            },
        };
    }
};

pub const TransferEntry = struct {
    buffer: zml.Buffer,
    writer: Transfer.Writer,
};

fn TransferReturnType(comptime ShapeType: type) type {
    if (ShapeType == zml.Shape) {
        return std.meta.Tuple(&.{ Transfer, TransferEntry });
    }
    const type_info = @typeInfo(ShapeType);
    return switch (type_info) {
        .@"struct" => |struct_type_info| b: {
            if (!struct_type_info.is_tuple) unreachable;
            inline for (struct_type_info.fields) |field| {
                if (field.type != zml.Shape) unreachable;
            }
            const types = [1]type{Transfer} ++ [_]type{TransferEntry} ** struct_type_info.fields.len;
            break :b std.meta.Tuple(&types);
        },
        else => unreachable,
    };
}

pub fn singleTransfer(allocator: std.mem.Allocator, shape: Shape, platform: Platform) !struct { Transfer, TransferEntry } {
    var transfer = try Transfer.init(allocator, &.{shape}, platform);
    errdefer transfer.deinit(allocator, platform);

    const entry = .{
        .buffer = zml.Buffer.fromPjrtBuffers(platform, shape, &.{transfer.get(0).buffer}),
        .writer = transfer.get(0).writer(),
    };

    return .{ transfer, entry };
}

pub fn multiTransfer(allocator: std.mem.Allocator, shapes: anytype, platform: Platform) !struct { Transfer, [shapes.len]TransferEntry } {
    var shapes_array: [shapes.len]zml.Shape = undefined;
    inline for (shapes, 0..) |shape, index| shapes_array[index] = shape;

    var transfer = try Transfer.init(allocator, &shapes_array, platform);
    errdefer transfer.deinit(allocator, platform);

    var entries: [shapes.len]TransferEntry = undefined;
    inline for (shapes, 0..) |shape, index| {
        entries[index].buffer = zml.Buffer.fromPjrtBuffers(platform, shape, &.{transfer.get(index).buffer});
        entries[index].writer = transfer.get(index).writer();
    }

    return .{ transfer, entries };
}

fn autoLoad(allocator: std.mem.Allocator, model: anytype, buffer_store: BufferStore4.View, platform: Platform) !Bufferized(@TypeOf(model)) {
    const Model = @TypeOf(model);
    var result: Bufferized(Model) = undefined;

    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();

    const shapes = try collectShapes(arena.allocator(), &model);

    var transfer: Transfer = try .init(arena.allocator(), shapes, platform);
    defer transfer.deinit(arena.allocator(), platform);

    const readers = try arena.allocator().alloc(BufferStore4.Reader, shapes.len);

    {
        var index: usize = 0;
        const type_info = @typeInfo(Model);
        switch (type_info) {
            .@"struct" => |struct_type_info| {
                inline for (struct_type_info.fields) |field| {
                    const reader = buffer_store.get(field.name).?.reader();
                    readers[index] = reader;
                    index += 1;
                }
            },
            else => unreachable,
        }
    }

    const LocalContext = struct {
        readers: []BufferStore4.Reader,
        shapes: []const Shape,
        platform: Platform,
        transfer: *Transfer,
        index: usize = 0,
    };
    var context: LocalContext = .{ .readers = readers, .shapes = shapes, .platform = platform, .transfer = &transfer };
    zml.meta.visit(struct {
        fn cb(context_: *LocalContext, buffer: *zml.Buffer) void {
            const writer = context_.transfer.get(context_.index).writer();
            context_.readers[context_.index].stream(writer);

            buffer.* = zml.Buffer.fromPjrtBuffers(context_.platform, context_.shapes[context_.index], &.{context_.transfer.get(context_.index).buffer});
            context_.index += 1;
        }
    }.cb, &context, &result);

    return result;
}

pub fn loadBuffersFromId(allocator: std.mem.Allocator, model: anytype, buffer_store: BufferStore5.View, platform: Platform) !Bufferized(@TypeOf(model)) {
    const Model = @TypeOf(model);
    var result: Bufferized(Model) = undefined;
    initBufferizedFrom(model, &result);

    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();

    const shapes = try collectShapes(arena.allocator(), &model);
    std.debug.print("Found {d} shapes\n", .{shapes.len});

    var transfer: Transfer = try .init(arena.allocator(), shapes, platform);
    defer transfer.deinit(arena.allocator(), platform);

    const readers = try collectReaders(arena.allocator(), buffer_store, &model);
    std.debug.print("Found {d} readers\n", .{readers.len});

    const LocalContext = struct {
        readers: []BufferStore5.Reader,
        shapes: []const Shape,
        platform: Platform,
        transfer: *Transfer,
        index: usize = 0,
    };
    var context: LocalContext = .{ .readers = readers, .shapes = shapes, .platform = platform, .transfer = &transfer };
    zml.meta.visit(struct {
        fn cb(context_: *LocalContext, buffer: *zml.Buffer) void {
            const writer = context_.transfer.get(context_.index).writer();
            context_.readers[context_.index].stream(writer);

            buffer.* = zml.Buffer.fromPjrtBuffers(context_.platform, context_.shapes[context_.index], &.{context_.transfer.get(context_.index).buffer});
            context_.index += 1;
        }
    }.cb, &context, &result);

    return result;
}

pub fn initBufferizedFrom(model: anytype, bufferized_: *Bufferized(@TypeOf(model))) void {
    const Model = @TypeOf(model);
    const type_info = @typeInfo(Bufferized(Model));
    switch (type_info) {
        .@"struct" => |struct_type_info| {
            if (Bufferized(Model) == zml.Buffer) return;
            inline for (struct_type_info.fields) |field| {
                initBufferizedFrom(@field(model, field.name), &@field(bufferized_, field.name));
            }
        },
        .@"union" => {
            switch (model) {
                inline else => |v, tag| {
                    bufferized_.* = @unionInit(Bufferized(Model), @tagName(tag), undefined);
                    initBufferizedFrom(v, @field(bufferized_, @tagName(tag)));
                },
            }
        },
        .optional => {
            if (model == null) {
                bufferized_.* = null;
            } else {
                bufferized_.* = undefined;
                initBufferizedFrom(model.?, &bufferized_.*.?);
            }
        },
        .void, .int, .@"enum", .bool, .enum_literal, .float, .pointer, .vector => {},
        else => unreachable,
    }
}

pub fn structTransfer(allocator: std.mem.Allocator, model: anytype, platform: Platform) !struct { Transfer, Transferized(@TypeOf(model)) } {
    const Model = @TypeOf(model);
    var result: Transferized(Model) = undefined;
    initTransferizedFrom(model, &result);

    const shapes = try collectShapes(allocator, &model);
    defer allocator.free(shapes);
    std.debug.print("Found {d} shapes\n", .{shapes.len});

    var transfer: Transfer = try .init(allocator, shapes, platform);
    errdefer transfer.deinit(allocator, platform);

    const LocalContext = struct {
        shapes: []const Shape,
        platform: Platform,
        transfer: *Transfer,
        index: usize = 0,
    };
    var context: LocalContext = .{ .shapes = shapes, .platform = platform, .transfer = &transfer };
    zml.meta.visit(struct {
        fn cb(context_: *LocalContext, transfer_entry: *TransferEntry) void {
            transfer_entry.buffer = zml.Buffer.fromPjrtBuffers(context_.platform, context_.shapes[context_.index], &.{context_.transfer.get(context_.index).buffer});
            transfer_entry.writer = context_.transfer.get(context_.index).writer();
            context_.index += 1;
        }
    }.cb, &context, &result);

    return .{ transfer, result };
}

pub fn Transferized(comptime T: type) type {
    return zml.meta.MapRestrict(Tensor, TransferEntry).map(T);
}

fn initTransferizedFrom(model: anytype, transferized: *Transferized(@TypeOf(model))) void {
    const Model = @TypeOf(model);
    const type_info = @typeInfo(Transferized(Model));
    switch (type_info) {
        .@"struct" => |struct_type_info| {
            if (Transferized(Model) == TransferEntry) return;
            inline for (struct_type_info.fields) |field| {
                initTransferizedFrom(@field(model, field.name), &@field(transferized, field.name));
            }
        },
        .@"union" => {
            switch (model) {
                inline else => |v, tag| {
                    transferized.* = @unionInit(Bufferized(Model), @tagName(tag), undefined);
                    initTransferizedFrom(v, @field(transferized, @tagName(tag)));
                },
            }
        },
        .optional => {
            if (model == null) {
                transferized.* = null;
            } else {
                transferized.* = undefined;
                initTransferizedFrom(model.?, &transferized.*.?);
            }
        },
        .void, .int, .@"enum", .bool, .enum_literal, .float, .pointer, .vector => {},
        else => unreachable,
    }
}

pub fn bufferTypeFromDtype(dt: zml.DataType) pjrt.BufferType {
    return switch (dt) {
        inline else => |tag| @field(pjrt.BufferType, @tagName(tag)),
    };
}

const Transfer = struct {
    entries: []Entry,
    transfer_manager: *pjrt.AsyncHostToDeviceTransferManager,

    pub const Writer = struct {
        entry: *Entry,
    };

    pub const Entry = struct {
        shape: Shape,
        transfer_manager: *pjrt.AsyncHostToDeviceTransferManager,
        buffer_index: usize,
        buffer: *pjrt.Buffer,
        platform: Platform,

        pub fn writer(entry: *Entry) Writer {
            return .{ .entry = entry };
        }
    };

    pub fn init(allocator: std.mem.Allocator, shapes: []const Shape, platform: Platform) !Transfer {
        const shape_specs = try allocator.alloc(pjrt.ShapeSpec, shapes.len);
        defer allocator.free(shape_specs);

        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();

        for (shape_specs, shapes) |*spec, shape| {
            const dims = try arena.allocator().dupe(i64, shape.dims());
            spec.* = pjrt.ShapeSpec.init(dims, bufferTypeFromDtype(shape.dtype()));
        }

        const memory = platform.pjrt_client.memoryByKind(platform.pjrt_api, .unpinned_host).?;

        const transfer_manager = try platform.pjrt_client.createBuffersForAsyncHostToDevice(platform.pjrt_api, .{ .shape_specs = shape_specs, .memory = memory });
        errdefer transfer_manager.deinit(platform.pjrt_api);

        const count = transfer_manager.bufferCount(platform.pjrt_api) catch unreachable;
        for (0..count) |index| {
            const buffer_size = transfer_manager.bufferSize(platform.pjrt_api, index) catch unreachable;
            std.debug.print("index: {d} - size: {d}\n", .{ index, buffer_size });
        }

        const entries = try allocator.alloc(Entry, shapes.len);
        errdefer allocator.free(entries);

        for (entries, shapes, 0..) |*e, shape, index| {
            const buffer = try transfer_manager.retrieveBuffer(platform.pjrt_api, index);
            e.* = .{
                .shape = shape,
                .transfer_manager = transfer_manager,
                .buffer_index = index,
                .buffer = buffer,
                .platform = platform,
            };
        }

        return .{ .entries = entries, .transfer_manager = transfer_manager };
    }

    pub fn deinit(self: Transfer, allocator: std.mem.Allocator, platform: Platform) void {
        allocator.free(self.entries);
        self.transfer_manager.deinit(platform.pjrt_api);
    }

    pub fn get(self: *const Transfer, index: usize) *Entry {
        return &self.entries[index];
    }
};

pub const Linear = struct {
    weight: Tensor,
    bias: ?Tensor = null,
    tag: Shape.Tag,

    pub fn init(weight: Tensor, bias: ?Tensor, dot_tag: anytype) Linear {
        return .{
            .weight = weight,
            .bias = bias,
            .tag = zml.Shape.toTag(dot_tag),
        };
    }

    pub fn forward(self: Linear, x: Tensor) Tensor {
        var y = x.dot(self.weight.convert(x.dtype()), self.tag);

        // log.debug("Linear({*}): {d} -> {d} -> {d}", .{ self, x.dims(), y.dims(), if (self.bias) |bias| y.add(bias).dims() else y.dims() });
        return if (self.bias) |bias| y.add(bias.autoBroadcast()) else y;
    }
};

const KvCacheKind = enum {
    split,
    unified,
};

const KvCache = union(KvCacheKind) {
    split: struct {
        k: Tensor,
        v: Tensor,
    },
    unified: Tensor,
};

fn testKvCache(kv_cache: KvCache, input: Tensor) KvCache {
    return switch (kv_cache) {
        .split => |split_kv_cache| .{
            .split = .{
                .k = split_kv_cache.k.add(input.autoBroadcast()),
                .v = split_kv_cache.v.add(input.autoBroadcast()),
            },
        },
        .unified => |unified_kv_cache| .{ .unified = unified_kv_cache.add(input.autoBroadcast()) },
    };
}

const RmsNorm = struct {
    weight: Tensor,
    eps: f32 = 1e-5,

    /// L2 normalization of input tensor along `.d` axis.
    pub fn forward(self: RmsNorm, input: Tensor) Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        const normalized = zml.nn.rmsNorm(x, .d, self.eps);
        return normalized.mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()));
    }
};

pub fn testLinear2(linear1: Linear, linear2: Linear, x: Tensor) Tensor {
    const out1 = linear1.forward(x).rename(.{ .dout = .d });
    std.debug.print("out1: {f}\n", .{out1});
    const out2 = linear2.forward(out1).rename(.{ .dout = .d });
    std.debug.print("out2: {f}\n", .{out1});
    return out2;
}

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain2() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    //const allocator = std.heap.c_allocator;
    const allocator = gpa.allocator();

    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);

    const exe = b: {
        const linear1: Linear = .init(
            Tensor.init(zml.Shape.init(.{ .dout = 128, .d = 128 }, .f32)),
            Tensor.init(zml.Shape.init(.{ .dout = 128 }, .f32)),
            .d,
        );
        const linear2: Linear = .init(
            Tensor.init(zml.Shape.init(.{ .dout = 64, .d = 128 }, .f32)),
            Tensor.init(zml.Shape.init(.{ .dout = 64 }, .f32)),
            .d,
        );

        const x = Tensor.init(zml.Shape.init(.{ .b = 16, .s = 16, .d = 128 }, .f32));

        const exe = try compile(allocator, testLinear2, .{ linear1, linear2, x }, platform);
        errdefer exe.deinit();

        break :b exe;
    };
    defer exe.deinit();
}

pub fn asyncMain() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    //const allocator = std.heap.c_allocator;
    const allocator = gpa.allocator();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const process_args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, process_args);

    const model_config_path = try std.fs.path.join(allocator, &.{ process_args[1], "config.json" });
    defer allocator.free(model_config_path);

    const model_weights_path = b: {
        const simple_path = try std.fs.path.join(allocator, &.{ process_args[1], "model.safetensors" });
        if (asynk.File.access(simple_path, .{})) {
            break :b simple_path;
        } else |_| {
            allocator.free(simple_path);
        }

        const sharded_path = try std.fs.path.join(allocator, &.{ process_args[1], "model.safetensors.index.json" });
        break :b sharded_path;
    };
    defer allocator.free(model_weights_path);

    const config = blk: {
        var config_json_file = try asynk.File.open(model_config_path, .{ .mode = .read_only });
        defer config_json_file.close() catch unreachable;
        var config_json_buffer: [256]u8 = undefined;
        var config_reader = config_json_file.reader(&config_json_buffer);
        var reader = std.json.Reader.init(allocator, &config_reader.interface);
        defer reader.deinit();
        const config_obj = try std.json.parseFromTokenSourceLeaky(llama.LlamaLM.Config, arena.allocator(), &reader, .{ .ignore_unknown_fields = true });
        break :blk config_obj;
    };

    var old_buffer_store = try zml.aio.detectFormatAndOpen(allocator, model_weights_path);
    defer old_buffer_store.deinit();

    var buffer_store: BufferStore5 = try .fromBufferStore(allocator, old_buffer_store);
    defer buffer_store.deinit();

    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);

    var llama_model = try llama.LlamaLM.init(allocator, buffer_store.view(), config);
    defer llama_model.deinit(allocator);

    const llama_buffers = try llama.LlamaLM.loadBuffers(allocator, llama_model, buffer_store.view(), platform);
    defer llama_buffers.deinit(allocator);

    //const process_args = try std.process.argsAlloc(allocator);
    //defer std.process.argsFree(allocator, process_args);
    //const pt_model = process_args[1];
    //const t10kfilename = process_args[2];
    //_ = t10kfilename; // autofix

    //// Read model shapes.
    //// Note this works because Mnist struct uses the same layer names as the pytorch model
    //var buffer_store = try zml.aio.detectFormatAndOpen(allocator, pt_model);
    //defer buffer_store.deinit();

    //var buffer_store2 = try BufferStore2.fromBufferStore(allocator, buffer_store, platform);
    //defer buffer_store2.deinit();

    //var tensor_descriptor = try TensorDescriptor.fromBufferStore(allocator, buffer_store);
    //defer tensor_descriptor.deinit();

    //// Compile MNIST Model
    //const mnist: Mnist = .init(tensor_descriptor.view());
    //const input: Tensor = .init(zml.Shape.init(.{ 28, 28 }, .u8));

    //var buffer_store3 = try BufferStore3.fromTensorDescriptor(allocator, tensor_descriptor, buffer_store, platform);
    //defer buffer_store3.deinit();

    //std.debug.print("{f}\n", .{mnist.layers[0].weight.shape()});

    //const exe = try compileModel(allocator, Mnist.forward, mnist, .{input}, platform);
    //defer exe.deinit();

    //// Create model, input and output buffers.
    //const input_buffer = try zml.Buffer.fromBytes(platform, input.shape(), std.mem.sliceAsBytes(&[_]u8{0} ** (28 * 28)));
    //_ = input_buffer; // autofix
    ////const mnist_buffer = try Mnist.initBuffer(buffer_store2);

    //const mnist_buffer = try bufferized(allocator, mnist, buffer_store3);
    //_ = mnist_buffer; // autofix

    //var buffer_store4: BufferStore4 = try .fromBufferStore(allocator, buffer_store, tensor_descriptor);
    //defer buffer_store4.deinit();

    //const mnist_buffer2 = try Mnist.loadBuffers(allocator, mnist, buffer_store4.view(), platform);
    //_ = mnist_buffer2; // autofix

    //var it = tensor_descriptor.map.iterator();
    //while (it.next()) |entry| {
    //    std.debug.print("key: \"{s}\" - value: {any}\n", .{ entry.key_ptr.*, entry.value_ptr.* });
    //}

    //const mnist_buffer, const arena = try bufferized(mnist, buffer_store2);

    // Create arguments (also prepare them) and results helper.
    //var args = try exe.args(allocator);
    //defer args.deinit(allocator);
    //args.set(.{ mnist_buffer, input_buffer });

    //var results = try exe.results(allocator);
    //defer results.deinit(allocator);

    //exe.call(args, &results);
    //const output = results.get(zml.Buffer);
    //defer output.deinit();

    //const linear: Linear = .{
    //    .weight = Tensor.init(Shape.init(.{ .d = 128, .d_out = 128 }, .f32)),
    //    .bias = Tensor.init(Shape.init(.{ .d_out = 128 }, .f32)),
    //    .tag = Shape.toTag(.d),
    //};

    //const mlp: Mlp = .{
    //    .up_proj = .{
    //        .weight = Tensor.init(Shape.init(.{ .d_hidden = 128, .d = 128 }, .f32)),
    //        .bias = Tensor.init(Shape.init(.{ .d_hidden = 128 }, .f32)),
    //    },
    //    .gate_proj = .{
    //        .weight = Tensor.init(Shape.init(.{ .d_hidden = 128, .d = 128 }, .f32)),
    //        .bias = Tensor.init(Shape.init(.{ .d_hidden = 128 }, .f32)),
    //    },
    //    .down_proj = .{
    //        .weight = Tensor.init(Shape.init(.{ .d = 128, .d_hidden = 128 }, .f32)),
    //        .bias = Tensor.init(Shape.init(.{ .d = 128 }, .f32)),
    //    },
    //};
    //_ = mlp; // autofix

    //const x = Tensor.init(Shape.init(.{ .b = 16, .d = 128 }, .f32));

    //compileModel(allocator, Mnist.forward, mnist, .{input});
    //const exe = try compile(allocator, testLinear, .{ linear, x }, platform);
    //defer exe.deinit();

    //const linear_buffers = zml.meta.MapRestrict(Tensor, zml.Buffer).map(Linear){
    //    .weight = try zml.Buffer.fromBytes(platform, linear.weight.shape(), std.mem.sliceAsBytes(&[_]f32{2} ** (128 * 128))),
    //    .bias = try zml.Buffer.fromBytes(platform, linear.bias.?.shape(), std.mem.sliceAsBytes(&[_]f32{3} ** (128))),
    //};
    //const x_buffer = try zml.Buffer.fromBytes(platform, x.shape(), std.mem.sliceAsBytes(&[_]f32{1} ** (16 * 128)));

    //var inputs: [3]zml.Buffer = undefined;
    //var outputs: [1]zml.Buffer = undefined;

    //serialize(&.{ linear_buffers, x_buffer }, &inputs);
    //exe.call(&inputs, &outputs);
    //var result: zml.Buffer = undefined;
    //deserialize(&result, outputs[0..1]);

    //var host = try result.toHostAlloc(allocator);
    //defer host.deinit(allocator);

    //std.debug.print("{any}\n", .{host.items(f32)});

    //compile(allocator, testMlp, .{ mlp, x });

    //compile(allocator, testDot, .{ Tensor.init(.init(.{ .m = 256, .n = 128 }, .f32)), Tensor.init(.init(.{ .b = 16, .n = 128, .k = 256 }, .f32)) });

    //try testSplitKvCache(allocator, platform);
    //try testUnifiedKvCache(allocator, platform);
    //try testMnist(allocator, buffer_store2, mnist, platform);
}

//fn testUnifiedKvCache(allocator: std.mem.Allocator, platform: Platform) !void {
//    const kv_cache: KvCache = .{
//        .unified = Tensor.init(Shape.init(.{ .layer = 10, .h = 16, .hd = 128 }, .f32)),
//    };
//
//    const input = Tensor.init(Shape.init(.{ .hd = 128 }, .f32));
//    const exe = try compile(allocator, testKvCache, .{ kv_cache, input }, platform);
//    defer exe.deinit();
//
//    var kv_cache_buffers: zml.meta.MapRestrict(Tensor, zml.Buffer).map(KvCache) = .{
//        .unified = try zml.Buffer.fromBytes(platform, kv_cache.unified.shape(), std.mem.sliceAsBytes(&[_]f32{1} ** (10 * 16 * 128))),
//    };
//    const input_buffer = try zml.Buffer.fromBytes(platform, input.shape(), std.mem.sliceAsBytes(&[_]f32{1} ** 128));
//
//    var inputs: [2]zml.Buffer = undefined;
//    var outputs: [1]zml.Buffer = undefined;
//
//    serialize(&.{ kv_cache_buffers, input_buffer }, &inputs);
//    exe.call(&inputs, &outputs);
//    deserialize(&kv_cache_buffers, &outputs);
//}

//fn testSplitKvCache(allocator: std.mem.Allocator, platform: Platform) !void {
//    const kv_cache: KvCache = .{
//        .split = .{
//            .k = Tensor.init(Shape.init(.{ .layer = 10, .h = 8, .hd = 128 }, .f32)),
//            .v = Tensor.init(Shape.init(.{ .layer = 10, .h = 8, .hd = 128 }, .f32)),
//        },
//    };
//
//    const input = Tensor.init(Shape.init(.{ .hd = 128 }, .f32));
//    const exe = try compile(allocator, testKvCache, .{ kv_cache, input }, platform);
//    defer exe.deinit();
//
//    var kv_cache_buffers: zml.meta.MapRestrict(Tensor, zml.Buffer).map(KvCache) = .{
//        .split = .{
//            .k = try zml.Buffer.fromBytes(platform, kv_cache.split.k.shape(), std.mem.sliceAsBytes(&[_]f32{1} ** (10 * 8 * 128))),
//            .v = try zml.Buffer.fromBytes(platform, kv_cache.split.k.shape(), std.mem.sliceAsBytes(&[_]f32{1} ** (10 * 8 * 128))),
//        },
//    };
//    const input_buffer = try zml.Buffer.fromBytes(platform, input.shape(), std.mem.sliceAsBytes(&[_]f32{1} ** 128));
//
//    var inputs: [3]zml.Buffer = undefined;
//    var outputs: [2]zml.Buffer = undefined;
//
//    serialize(&.{ kv_cache_buffers, input_buffer }, &inputs);
//    exe.call(&inputs, &outputs);
//    deserialize(&kv_cache_buffers, &outputs);
//}

fn testDot(a: Tensor, b: Tensor) Tensor {
    _ = a.dot(b, .n);
    _ = a.dot(b, .{ .n = .k });
    _ = a.dot(b, .{ .contracting = .{ .n = .k }, .batching = .{.b} });
    _ = a.dot(b, .{ .contracting = .{.n}, .batching = .{.b} });
    unreachable;
    //return a.dotGeneral(b, &.{.{ 1, 1 }}, &.{});
}

fn testLinear(linear: Linear, x: Tensor) Tensor {
    return linear.forward(x);
}

const CompilationContext = struct {
    allocator: std.mem.Allocator,

    mlir_registry: *mlir.DialectRegistry,
    mlir_ctx: *mlir.Context,
    mlir_pass_manager: *mlir.PassManager,
    //mlir_op_pass_manager: *mlir.OpPassManager,
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

        const pass_manager = mlir.PassManager.init(mlir_ctx);
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
                opm.addPipeline(pass) catch unreachable;
            }
        }

        return .{
            .allocator = allocator,
            .mlir_registry = mlir_registry,
            .mlir_ctx = mlir_ctx,
            .mlir_pass_manager = pass_manager,
            .module = module,
        };
    }

    pub fn deinit(self: *CompilationContext) void {
        self.mlir_pass_manager.deinit();
        self.module.deinit();
        self.mlir_ctx.deinit();
        self.mlir_registry.deinit();
    }

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

fn compileModel(allocator: std.mem.Allocator, comptime func: anytype, model: stdx.meta.Head(stdx.meta.FnArgs(func)), args: stdx.meta.Tail(stdx.meta.FnArgs(func)), platform: Platform) !Exe {
    return compile(allocator, func, .{model} ++ args, platform);
}

pub fn compile(allocator: std.mem.Allocator, comptime func: anytype, args: stdx.meta.FnArgs(func), platform: Platform) !Exe {
    var compilation_context: CompilationContext = .init(allocator);
    defer compilation_context.deinit();

    const result = emitMlir(&compilation_context, func, args) catch unreachable;

    _ = result.func.appendTo(compilation_context.module.body());
    defer compilation_context.allocator.free(result.output_shapes);
    defer compilation_context.allocator.free(result.input_shapes);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const loaded_executable = compileModuleToPjrtExecutable(arena.allocator(), platform, compilation_context.module, null) catch unreachable;

    const num_devices = platform.sharding().num_replicas * platform.sharding().num_partitions;
    const exe = try Exe.init(allocator, platform, loaded_executable, result.input_shapes, result.output_shapes, num_devices);
    errdefer exe.deinit();

    return exe;
}

fn collectShapes(allocator: std.mem.Allocator, v: anytype) ![]Shape {
    const LocalContext = struct {
        list: *std.array_list.Managed(Shape),
    };
    var list = std.array_list.Managed(Shape).init(allocator);
    var context: LocalContext = .{ .list = &list };
    zml.meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *const Tensor) void {
            ctx_.list.append(tensor.shape()) catch unreachable;
        }
    }.cb, &context, v);

    return try list.toOwnedSlice();
}

fn collectValues(allocator: std.mem.Allocator, v: anytype) ![]*const mlir.Value {
    const LocalContext = struct {
        list: *std.array_list.Managed(*const mlir.Value),
    };
    var list = std.array_list.Managed(*const mlir.Value).init(allocator);
    var context: LocalContext = .{ .list = &list };
    zml.meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *const Tensor) void {
            ctx_.list.append(tensor.value()) catch unreachable;
        }
    }.cb, &context, v);

    return try list.toOwnedSlice();
}

fn collectReaders(allocator: std.mem.Allocator, buffer_store: BufferStore5.View, v: anytype) ![]BufferStore5.Reader {
    const LocalContext = struct {
        list: *std.array_list.Managed(BufferStore5.Reader),
        buffer_store: BufferStore5.View,
    };
    var list = std.array_list.Managed(BufferStore5.Reader).init(allocator);
    var context: LocalContext = .{ .list = &list, .buffer_store = buffer_store };
    zml.meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *const Tensor) void {
            const reader = ctx_.buffer_store.getReaderFromId(tensor.id).?;
            ctx_.list.append(reader) catch unreachable;
        }
    }.cb, &context, v);

    return try list.toOwnedSlice();
}

const EmitMlirResult = struct {
    func: *mlir.Operation,
    input_shapes: []const Shape,
    output_shapes: []const Shape,
};

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

    const input_shapes = try collectShapes(compilation_context.allocator, &args);
    errdefer compilation_context.allocator.free(input_shapes);

    const output_shapes = b: {
        global_compilation_context = compilation_context;
        defer global_compilation_context = null;
        const result = @call(.auto, func, args);
        const output_shapes = try collectShapes(compilation_context.allocator, &result);
        const output_values = try collectValues(arena.allocator(), &result);
        _ = dialects.func.returns(compilation_context.mlir_ctx, output_values, .unknown(compilation_context.mlir_ctx)).appendTo(compilation_context.currentBlock());
        break :b output_shapes;
    };
    errdefer compilation_context.allocator.free(output_shapes);

    const mlir_func = dialects.func.func(compilation_context.mlir_ctx, .{
        .name = "main",
        .block = compilation_context.currentBlock(),
        .location = .unknown(compilation_context.mlir_ctx),
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
        .input_shapes = input_shapes,
        .output_shapes = output_shapes,
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

fn compileModuleToPjrtExecutable(arena: std.mem.Allocator, platform: Platform, module: *const mlir.Module, xla_dump_to_: ?[]const u8) !*pjrt.LoadedExecutable {
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
                try setXlaOverrideFlag(overrides_map, "xla_gpu_enable_triton_gemm", false, upb_arena);
                try setXlaOverrideFlag(overrides_map, "xla_gpu_enable_latency_hiding_scheduler", true, upb_arena);
                try setXlaOverrideFlag(overrides_map, "xla_gpu_enable_llvm_module_compilation_parallelism", true, upb_arena);
                try setXlaOverrideFlag(overrides_map, "xla_gpu_enable_libnvptxcompiler", true, upb_arena);
            },
            .rocm => {
                // Disable Triton GEMM on ROCM. For some reason it's much, much slower when
                // enabled on CDNA and it's used on RDNA. Disable it altogether.
                try setXlaOverrideFlag(overrides_map, "xla_gpu_enable_triton_gemm", false, upb_arena);
                // Use lld from libllvm instead of invoking the ld.lld binary.
                // This saves us from having to sandbox it.
                try setXlaOverrideFlag(overrides_map, "xla_gpu_use_inprocess_lld", true, upb_arena);
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

    const loaded_executable = try platform.pjrt_client.compile(
        platform.pjrt_api,
        arena,
        module,
        try upb.serialize(options, upb_arena),
    );
    errdefer loaded_executable.deinit();

    return loaded_executable;
}

const Exe = struct {
    platform: Platform,
    exe: *pjrt.LoadedExecutable,

    context: ?*pjrt.ExecuteContext = null,

    input_shapes: []const Shape,
    output_shapes: []const Shape,

    num_devices: u8,

    arena: std.heap.ArenaAllocator,

    pub fn init(
        allocator: std.mem.Allocator,
        platform: Platform,
        exe: *pjrt.LoadedExecutable,
        input_shapes: []const Shape,
        output_shapes: []const Shape,
        num_devices: u8,
    ) !Exe {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        const input_shapes_copy = try arena.allocator().dupe(Shape, input_shapes);
        const output_shapes_copy = try arena.allocator().dupe(Shape, output_shapes);

        return .{
            .platform = platform,
            .exe = exe,
            .input_shapes = input_shapes_copy,
            .output_shapes = output_shapes_copy,
            .num_devices = num_devices,
            .arena = arena,
        };
    }

    pub fn deinit(self: *const Exe) void {
        self.arena.deinit();
    }

    pub fn args(self: *const Exe, allocator: std.mem.Allocator) !Arguments {
        return Arguments.init(allocator, self.input_shapes, self.num_devices);
    }

    pub fn results(self: *const Exe, allocator: std.mem.Allocator) !Results {
        return Results.init(allocator, self.output_shapes, self.num_devices, self.platform);
    }

    pub const FlatBuffers = struct {
        buffers: []const [*]*pjrt.Buffer,
        raw_buffers: []const *pjrt.Buffer,

        num_devices: usize,

        pub fn init(allocator: std.mem.Allocator, count: usize, num_devices: usize) !FlatBuffers {
            const raw_buffers = try allocator.alloc(*pjrt.Buffer, num_devices * count);
            errdefer allocator.free(raw_buffers);

            const buffers = try allocator.alloc([*]*pjrt.Buffer, num_devices);
            errdefer allocator.free(buffers);

            for (0..num_devices) |i| {
                buffers[i] = raw_buffers[i * count ..].ptr;
            }

            return .{
                .buffers = buffers,
                .raw_buffers = raw_buffers,
                .num_devices = num_devices,
            };
        }

        pub fn deinit(self: *const FlatBuffers, allocator: std.mem.Allocator) void {
            allocator.free(self.buffers);
            allocator.free(self.raw_buffers);
        }
    };

    pub const Arguments = struct {
        flat_buffers: FlatBuffers,
        expected_shapes: []const Shape,

        pub fn init(allocator: std.mem.Allocator, shapes: []const Shape, num_devices: usize) !Arguments {
            const flat_buffers = try FlatBuffers.init(allocator, shapes.len, num_devices);
            errdefer flat_buffers.deinit(allocator);

            const expected_shapes = try allocator.dupe(Shape, shapes);
            errdefer allocator.free(expected_shapes);

            return .{
                .flat_buffers = flat_buffers,
                .expected_shapes = expected_shapes,
            };
        }

        pub fn deinit(self: *const Arguments, allocator: std.mem.Allocator) void {
            allocator.free(self.expected_shapes);
            self.flat_buffers.deinit(allocator);
        }

        pub fn set(self: *Arguments, v: anytype) void {
            return self.setPartial(v, 0);
        }

        pub fn setPartial(self: *Arguments, v: anytype, offset: usize) void {
            const LocalContext = struct {
                self: *Arguments,
                current_index: usize = 0,
            };
            var context: LocalContext = .{ .self = self, .current_index = offset };
            zml.meta.visit(struct {
                fn cb(context_: *LocalContext, buffer: *const zml.Buffer) void {
                    for (0..context_.self.flat_buffers.num_devices) |device_index| {
                        context_.self.flat_buffers.buffers[device_index][context_.current_index] = buffer._shards.get(device_index);
                    }

                    context_.current_index += 1;
                }
            }.cb, &context, &v);
        }
    };

    pub const Results = struct {
        platform: Platform,
        flat_buffers: FlatBuffers,

        expected_shapes: []const Shape,

        pub fn init(allocator: std.mem.Allocator, shapes: []const Shape, num_devices: usize, platform: Platform) !Results {
            const flat_buffers = try FlatBuffers.init(allocator, shapes.len, num_devices);
            errdefer flat_buffers.deinit(allocator);

            const expected_shapes = try allocator.dupe(Shape, shapes);
            errdefer allocator.free(expected_shapes);

            return .{
                .platform = platform,
                .flat_buffers = flat_buffers,
                .expected_shapes = expected_shapes,
            };
        }

        pub fn deinit(self: *const Results, allocator: std.mem.Allocator) void {
            allocator.free(self.expected_shapes);
            self.flat_buffers.deinit(allocator);
        }

        pub fn get(self: *Results, comptime T: type) T {
            var result: T = undefined;
            const LocalContext = struct {
                self: *Results,
                current_index: usize = 0,
            };
            var context: LocalContext = .{ .self = self, .current_index = 0 };
            zml.meta.visit(struct {
                fn cb(context_: *LocalContext, buffer: *zml.Buffer) void {
                    var shards: zml.Buffer.Shards = .{};
                    for (0..context_.self.flat_buffers.num_devices) |device_index| {
                        shards.appendAssumeCapacity(context_.self.flat_buffers.buffers[device_index][context_.current_index]);
                    }
                    buffer.* = zml.Buffer.fromPjrtBuffers(context_.self.platform, context_.self.expected_shapes[context_.current_index], shards.constSlice());
                    context_.current_index += 1;
                }
            }.cb, &context, &result);
            return result;
        }

        pub fn fill(self: *Results, v: anytype) void {
            const LocalContext = struct {
                self: *Results,
                current_index: usize = 0,
            };
            var context: LocalContext = .{ .self = self, .current_index = 0 };
            zml.meta.visit(struct {
                fn cb(context_: *LocalContext, buffer: *zml.Buffer) void {
                    var shards: zml.Buffer.Shards = .{};
                    for (0..context_.self.flat_buffers.num_devices) |device_index| {
                        shards.appendAssumeCapacity(context_.self.flat_buffers.buffers[device_index][context_.current_index]);
                    }
                    buffer.* = zml.Buffer.fromPjrtBuffers(context_.self.platform, context_.self.expected_shapes[context_.current_index], shards.constSlice());
                    context_.current_index += 1;
                }
            }.cb, &context, v);
        }
    };

    pub fn call(self: *const Exe, arguments: Arguments, results_: *Results) void {
        var events = [_]?*pjrt.Event{null} ** Platform.MAX_NUM_DEVICES;
        const sharding = self.platform.sharding();

        self.exe.execute(self.platform.pjrt_api, .{
            .arguments = arguments.flat_buffers.buffers,
            .num_args = arguments.expected_shapes.len,
            .results = results_.flat_buffers.buffers,
            .events = events[0..sharding.num_partitions],
            // this allows to tell a specific buffer shouldn't be donated,
            // even if it has been marked as "can be donated" during compilation.
            // TODO: expose it ?
            .non_donatable_input_indices = &.{},
            .context = self.context,
        }) catch |err| {
            std.debug.panic("PJRT_LoadedExecutable_Execute failed with: {}", .{err});
        };

        for (events[0..sharding.num_partitions]) |e| {
            if (e) |ev| {
                ev.await_(self.platform.pjrt_api) catch unreachable;
            }
        }
    }
};

pub const TensorDescriptor = struct {
    map: std.StringHashMapUnmanaged(Entry),
    allocator: std.mem.Allocator,

    const Entry = struct {
        shape: zml.Shape,
        associated_tensor_id: ?usize = null,
    };

    pub const View = struct {
        tensor_descriptor: *const TensorDescriptor,

        prefix_buffer: [256]u8 = undefined,
        prefix_length: usize = 0,

        pub fn withPrefix(self: *const View, prefix_: []const u8) View {
            var buffer: [256]u8 = undefined;
            const new_prefix = std.fmt.bufPrint(&buffer, "{s}{s}.", .{ self.prefix() orelse "", prefix_ }) catch unreachable;

            return .{
                .tensor_descriptor = self.tensor_descriptor,
                .prefix_buffer = buffer,
                .prefix_length = new_prefix.len,
            };
        }

        fn prefix(self: *const View) ?[]const u8 {
            return if (self.prefix_length == 0) null else self.prefix_buffer[0..self.prefix_length];
        }

        pub fn get(self: *const View, subkey: []const u8) ?zml.Shape {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            std.debug.print("Trying to get: {s}\n", .{key});
            const entry = self.tensor_descriptor.getPtr(key) orelse return null;
            return entry.shape;
        }

        pub fn getWithTags(self: View, subkey: []const u8, tagz: anytype) ?Tensor {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            std.debug.print("Trying to get: {s}\n", .{key});
            const ptr = self.tensor_descriptor.getPtr(key) orelse return null;
            ptr.shape = ptr.shape.withTags(tagz).withSharding(.{0});
            const tensor = Tensor.init(ptr.shape);
            ptr.associated_tensor_id = tensor.id;
            return tensor;
        }
    };

    pub fn init(allocator: std.mem.Allocator) !TensorDescriptor {
        return .{
            .map = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TensorDescriptor) void {
        var it = self.map.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.map.deinit(self.allocator);
    }

    pub fn add(self: *TensorDescriptor, key: []const u8, shape: zml.Shape) !void {
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);

        const gop = try self.map.getOrPut(self.allocator, key_copy);
        if (gop.found_existing) {
            return error.AlreadyExists;
        }

        errdefer self.map.removeByPtr(gop.key_ptr);

        gop.value_ptr.* = .{ .shape = shape };
    }

    pub fn fromBufferStore(allocator: std.mem.Allocator, buffer_store: zml.aio.BufferStore) !TensorDescriptor {
        var new_tensor_descriptor: TensorDescriptor = try .init(allocator);
        errdefer new_tensor_descriptor.deinit();

        var it = buffer_store.buffers.iterator();
        while (it.next()) |entry| {
            try new_tensor_descriptor.add(entry.key_ptr.*, entry.value_ptr.shape());
        }

        return new_tensor_descriptor;
    }

    pub fn withPrefix(self: *const TensorDescriptor, prefix_: []const u8) TensorDescriptor {
        var buffer: [256]u8 = undefined;
        const new_prefix = std.fmt.bufPrint(&buffer, "{s}{s}.", .{ self.prefix() orelse "", prefix_ }) catch unreachable;

        return .{
            .map = self.map,
            .allocator = self.allocator,
            .prefix_buffer = buffer,
            .prefix_length = new_prefix.len,
        };
    }

    fn prefix(self: *const TensorDescriptor) ?[]const u8 {
        return if (self.prefix_length == 0) null else self.prefix_buffer[0..self.prefix_length];
    }

    fn getPtr(self: *const TensorDescriptor, key: []const u8) ?*Entry {
        return self.map.getPtr(key);
    }

    pub fn view(self: *const TensorDescriptor) View {
        return .{ .tensor_descriptor = self };
    }
};

const BufferStore2 = struct {
    buffers: std.StringHashMapUnmanaged(zml.Buffer),
    allocator: std.mem.Allocator,

    prefix_buffer: [256]u8 = undefined,
    prefix_length: usize = 0,

    pub fn init(allocator: std.mem.Allocator) !BufferStore2 {
        return .{
            .buffers = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BufferStore2) void {
        var it = self.buffers.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit();
            self.allocator.free(entry.key_ptr.*);
        }
        self.buffers.deinit(self.allocator);
    }

    pub fn add(self: *BufferStore2, key: []const u8, buffer: zml.Buffer) !void {
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);

        const gop = try self.buffers.getOrPut(self.allocator, key_copy);
        if (gop.found_existing) {
            return error.AlreadyExists;
        }

        errdefer self.buffers.removeByPtr(gop.key_ptr);

        gop.value_ptr.* = buffer;
    }

    pub fn fromBufferStore(allocator: std.mem.Allocator, buffer_store: zml.aio.BufferStore, platform: Platform) !BufferStore2 {
        var new_buffer_store: BufferStore2 = try .init(allocator);
        errdefer new_buffer_store.deinit();

        var it = buffer_store.buffers.iterator();
        while (it.next()) |entry| {
            try new_buffer_store.add(entry.key_ptr.*, try entry.value_ptr.toDevice(platform));
        }

        return new_buffer_store;
    }

    pub fn withPrefix(self: *const BufferStore2, prefix_: []const u8) BufferStore2 {
        var buffer: [256]u8 = undefined;
        const new_prefix = std.fmt.bufPrint(&buffer, "{s}{s}.", .{ self.prefix() orelse "", prefix_ }) catch unreachable;

        return .{
            .buffers = self.buffers,
            .allocator = self.allocator,
            .prefix_buffer = buffer,
            .prefix_length = new_prefix.len,
        };
    }

    fn prefix(self: *const BufferStore2) ?[]const u8 {
        return if (self.prefix_length == 0) null else self.prefix_buffer[0..self.prefix_length];
    }

    fn get(self: *const BufferStore2, subkey: []const u8) ?zml.Buffer {
        var buffer: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
        std.debug.print("Trying to get: {s}\n", .{key});
        return self.buffers.get(key);
    }
};

const BufferStore3 = struct {
    buffers: std.ArrayList(zml.Buffer) = .{},
    id_to_buffer: std.AutoHashMapUnmanaged(usize, usize) = .{},
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !BufferStore3 {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BufferStore3) void {
        self.buffers.deinit(self.allocator);
        self.id_to_buffer.deinit(self.allocator);
    }

    pub fn fromTensorDescriptor(allocator: std.mem.Allocator, tensor_descriptor: TensorDescriptor, old_buffer_store: zml.aio.BufferStore, platform: Platform) !BufferStore3 {
        var buffer_store: BufferStore3 = try .init(allocator);
        errdefer for (buffer_store.buffers.items) |*item| item.deinit();
        errdefer buffer_store.deinit();

        var it = tensor_descriptor.map.iterator();
        while (it.next()) |entry| {
            var host_buffer = old_buffer_store.get(entry.key_ptr.*).?;
            host_buffer._shape = entry.value_ptr.shape;
            const buffer = try host_buffer.toDevice(platform);
            errdefer buffer.deinit();

            try buffer_store.add(buffer, entry.value_ptr.associated_tensor_id.?);
        }
        return buffer_store;
    }

    pub fn add(self: *BufferStore3, buffer: zml.Buffer, id: usize) !void {
        const gop = try self.id_to_buffer.getOrPut(self.allocator, id);
        if (gop.found_existing) return error.AlreadyExists;

        errdefer self.id_to_buffer.removeByPtr(gop.key_ptr);

        const new_index = self.buffers.items.len;

        try self.buffers.append(self.allocator, buffer);
        errdefer _ = self.buffers.pop();

        gop.value_ptr.* = new_index;
    }

    pub fn getFromId(self: *const BufferStore3, id: usize) ?zml.Buffer {
        const index = self.id_to_buffer.get(id) orelse return null;
        return self.buffers.items[index];
    }
};

pub fn bufferized(allocator: std.mem.Allocator, model: anytype, buffer_store: BufferStore3) !Bufferized(@TypeOf(model)) {
    const T = @TypeOf(model);
    const BufferizedT = Bufferized(T);
    var buf: BufferizedT = undefined;

    const LocalContext = struct {
        buffer_store: *const BufferStore3,
    };

    var context: LocalContext = .{ .buffer_store = &buffer_store };

    try zml.meta.mapAlloc(struct {
        fn cb(context_: *LocalContext, tensor: Tensor) zml.Buffer {
            const buffer = context_.buffer_store.getFromId(tensor.id).?;
            return buffer;
        }
    }.cb, allocator, &context, model, &buf);

    return buf;
}

pub fn Bufferized(comptime T: type) type {
    // TODO: we should strip out the non-buffer fields.
    // Currently it's confusing cause the Bufferized struct contains field that are never read.
    // Also it will simplify the layout of the Bufferized struct.
    // accelerating the calls to execute.
    return zml.meta.MapRestrict(Tensor, zml.Buffer).map(T);
}

pub const ShardWriter = struct {
    tensor_shape: Shape,
    shard_shape: Shape,
    interface: std.Io.Writer,
    out: *std.Io.Writer,

    tensor_index: usize = 0,
    shard_index: usize = 0,
    shard_coordinates: [2]usize,

    fn init(out: *std.Io.Writer, buffer: []u8, tensor_shape: Shape, shard_shape: Shape) ShardWriter {
        return .{
            .tensor_shape = tensor_shape,
            .shard_shape = shard_shape,
            .interface = .{
                .buffer = buffer,
                .vtable = &.{ .drain = drain },
            },
            .out = out,
        };
    }

    pub fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *ShardWriter = @alignCast(@fieldParentPtr("interface", w));
        _ = self; // autofix
        _ = data; // autofix
        _ = splat; // autofix
    }

    fn computeCoordinates(self: ShardWriter, offset: usize) [2]usize {
        const y = offset / self.tensor_shape.dim(0);
        const x = offset % self.tensor_shape.dim(0);

        const x_shard = @divFloor(x, self.shard_shape.dim(1));
        const y_shard = @divFloor(y, self.shard_shape.dim(0));

        return .{ x_shard, y_shard };
    }
};

const Slice = struct {
    shape: Shape,
    data: []const u8,
};

pub const BufferStore4 = struct {
    slices: std.StringHashMapUnmanaged(Entry),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !BufferStore4 {
        return .{ .slices = .{}, .allocator = allocator };
    }

    pub fn deinit(self: *BufferStore4) void {
        var it = self.slices.iterator();
        while (it.next()) |entry| self.allocator.free(entry.key_ptr.*);
        self.slices.deinit(self.allocator);
    }

    pub fn add(self: *BufferStore4, key: []const u8, slice: Slice, associated_tensor_id: ?usize) !void {
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);

        const gop = try self.slices.getOrPut(self.allocator, key_copy);
        if (gop.found_existing) {
            return error.AlreadyExists;
        }

        errdefer self.slices.removeByPtr(gop.key_ptr);

        gop.value_ptr.* = .{ .slice = slice, .associated_tensor_id = associated_tensor_id };
    }

    pub fn fromBufferStore(allocator: std.mem.Allocator, buffer_store: zml.aio.BufferStore, tensor_descriptor: TensorDescriptor) !BufferStore4 {
        var new_buffer_store: BufferStore4 = try .init(allocator);
        errdefer new_buffer_store.deinit();

        var it = buffer_store.buffers.iterator();
        while (it.next()) |entry| {
            const descriptor_entry = tensor_descriptor.getPtr(entry.key_ptr.*).?;
            const slice: Slice = .{ .shape = entry.value_ptr.shape(), .data = entry.value_ptr.bytes() };
            try new_buffer_store.add(entry.key_ptr.*, slice, descriptor_entry.associated_tensor_id);
        }

        return new_buffer_store;
    }

    pub const Reader = struct {
        entry: *const Entry,
        pub fn stream(self: Reader, writer: Transfer.Writer) void {
            const chunk_size = 1024 * 1024;
            const chunk_count = (self.entry.slice.data.len + chunk_size - 1) / chunk_size;

            for (0..chunk_count) |chunk_index| {
                const start = chunk_index * chunk_size;
                const end = @min((chunk_index + 1) * chunk_size, self.entry.slice.data.len);
                const is_last_transfer = chunk_index == chunk_count - 1;
                _ = writer.entry.transfer_manager.transferData(writer.entry.platform.pjrt_api, writer.entry.buffer_index, self.entry.slice.data[start..end], @intCast(start), is_last_transfer) catch unreachable;
            }
        }
    };

    pub const Entry = struct {
        slice: Slice,
        associated_tensor_id: ?usize = null,
        pub fn reader(entry: *const Entry) Reader {
            return .{ .entry = entry };
        }
    };

    pub fn get(self: BufferStore4, key: []const u8) ?*Entry {
        std.debug.print("Trying to get {s}\n", .{key});
        return self.slices.getPtr(key);
    }

    pub fn view(self: *const BufferStore4) View {
        return .{ .buffer_store = self };
    }

    pub const View = struct {
        buffer_store: *const BufferStore4,

        prefix_buffer: [256]u8 = undefined,
        prefix_length: usize = 0,

        pub fn withPrefix(self: *const View, prefix_: []const u8) View {
            var buffer: [256]u8 = undefined;
            const new_prefix = std.fmt.bufPrint(&buffer, "{s}{s}.", .{ self.prefix() orelse "", prefix_ }) catch unreachable;

            return .{
                .buffer_store = self.buffer_store,
                .prefix_buffer = buffer,
                .prefix_length = new_prefix.len,
            };
        }

        fn prefix(self: *const View) ?[]const u8 {
            return if (self.prefix_length == 0) null else self.prefix_buffer[0..self.prefix_length];
        }

        pub fn get(self: *const View, subkey: []const u8) ?*Entry {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            std.debug.print("Trying to get: {s}\n", .{key});
            return self.buffer_store.get(key);
        }
    };
};

pub const BufferStore5 = struct {
    key_map: std.StringHashMapUnmanaged(usize),
    id_map: std.AutoHashMapUnmanaged(usize, usize),
    entries: std.ArrayList(Entry),
    allocator: std.mem.Allocator,

    pub const Entry = struct {
        shape: zml.Shape,
        // TODO(Corentin): Replace with reader ?
        data: []const u8,
        associated_tensor_id: ?usize = null,
    };

    pub fn init(allocator: std.mem.Allocator) !BufferStore5 {
        return .{
            .key_map = .empty,
            .id_map = .empty,
            .entries = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BufferStore5) void {
        var it = self.key_map.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.key_map.deinit(self.allocator);
        self.id_map.deinit(self.allocator);
        self.entries.deinit(self.allocator);
    }

    pub fn add(self: *BufferStore5, key: []const u8, shape: zml.Shape, data: []const u8) !void {
        const new_entry_index = self.entries.items.len;
        const new_entry = try self.entries.addOne(self.allocator);
        errdefer _ = self.entries.pop();

        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);

        const gop = try self.key_map.getOrPut(self.allocator, key_copy);
        if (gop.found_existing) {
            return error.AlreadyExists;
        }
        errdefer self.key_map.removeByPtr(gop.key_ptr);

        gop.value_ptr.* = new_entry_index;
        new_entry.* = .{ .shape = shape, .data = data };
    }

    fn bindIdToKey(self: *BufferStore5, key: []const u8, id: usize) !void {
        const index = self.key_map.get(key).?;

        const gop = try self.id_map.getOrPut(self.allocator, id);
        if (gop.found_existing) {
            stdx.debug.panic("Key {s} already has an associated tensor (id: {})", .{ key, gop.value_ptr.* });
        }
        errdefer self.id_map.removeByPtr(gop.key_ptr);

        gop.value_ptr.* = index;
    }

    pub fn fromBufferStore(allocator: std.mem.Allocator, buffer_store: zml.aio.BufferStore) !BufferStore5 {
        var new_buffer_store: BufferStore5 = try .init(allocator);
        errdefer new_buffer_store.deinit();

        var it = buffer_store.buffers.iterator();
        while (it.next()) |entry| {
            try new_buffer_store.add(entry.key_ptr.*, entry.value_ptr.shape(), entry.value_ptr.bytes());
        }

        return new_buffer_store;
    }

    fn getPtrFromKey(self: *const BufferStore5, key: []const u8) ?*Entry {
        const index = self.key_map.get(key) orelse return null;
        return &self.entries.items[index];
    }

    fn getPtrFromId(self: *const BufferStore5, id: usize) ?*Entry {
        const index = self.id_map.get(id) orelse return null;
        return &self.entries.items[index];
    }

    pub fn view(self: *BufferStore5) View {
        return .{ .store = self };
    }

    pub const Reader = struct {
        entry: *const Entry,
        pub fn stream(self: Reader, writer: Transfer.Writer) void {
            const chunk_size = 1024 * 1024;
            const chunk_count = (self.entry.data.len + chunk_size - 1) / chunk_size;

            for (0..chunk_count) |chunk_index| {
                const start = chunk_index * chunk_size;
                const end = @min((chunk_index + 1) * chunk_size, self.entry.data.len);
                const is_last_transfer = chunk_index == chunk_count - 1;
                _ = writer.entry.transfer_manager.transferData(writer.entry.platform.pjrt_api, writer.entry.buffer_index, self.entry.data[start..end], @intCast(start), is_last_transfer) catch unreachable;
            }
        }
    };

    pub const View = struct {
        store: *BufferStore5,

        prefix_buffer: [256]u8 = undefined,
        prefix_length: usize = 0,

        pub fn withPrefix(self: *const View, prefix_: []const u8) View {
            var buffer: [256]u8 = undefined;
            const new_prefix = std.fmt.bufPrint(&buffer, "{s}{s}.", .{ self.prefix() orelse "", prefix_ }) catch unreachable;

            return .{
                .store = self.store,
                .prefix_buffer = buffer,
                .prefix_length = new_prefix.len,
            };
        }

        fn prefix(self: *const View) ?[]const u8 {
            return if (self.prefix_length == 0) null else self.prefix_buffer[0..self.prefix_length];
        }

        pub fn maybeCreateTensor(self: View, subkey: []const u8) ?Tensor {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;

            const ptr = self.store.getPtrFromKey(key) orelse return null;

            const tensor = Tensor.init(ptr.shape);
            self.store.bindIdToKey(key, tensor.id) catch unreachable;

            return tensor;
        }

        pub fn createTensor(self: View, subkey: []const u8) Tensor {
            return self.maybeCreateTensor(subkey).?;
        }

        pub fn maybeCreateTensorWithTags(self: View, subkey: []const u8, tagz: anytype) ?Tensor {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;

            const ptr = self.store.getPtrFromKey(key) orelse return null;
            ptr.shape = ptr.shape.withTags(tagz).withSharding(.{0});

            const tensor = Tensor.init(ptr.shape);
            self.store.bindIdToKey(key, tensor.id) catch unreachable;

            return tensor;
        }

        pub fn createTensorWithTags(self: View, subkey: []const u8, tagz: anytype) Tensor {
            return self.maybeCreateTensorWithTags(subkey, tagz).?;
        }

        pub fn getReader(self: View, subkey: []const u8) ?Reader {
            return self.getMaybeReader(subkey).?;
        }

        pub fn getMaybeReader(self: View, subkey: []const u8) ?Reader {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            const entry_ptr = self.store.getPtrFromKey(key) orelse return null;
            return .{ .entry = entry_ptr };
        }

        pub fn getReaderFromId(self: View, id: usize) ?Reader {
            const entry_ptr = self.store.getPtrFromId(id) orelse return null;
            return .{ .entry = entry_ptr };
        }

        pub fn getShape(self: View, subkey: []const u8) ?Shape {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            const entry_ptr = self.store.getPtrFromKey(key) orelse return null;
            return entry_ptr.shape;
        }
    };
};
