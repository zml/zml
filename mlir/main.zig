const asynk = @import("async");
const std = @import("std");
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
    _shape: zml.Shape,
    tensor_origin: TensorOrigin = .{ .argument = {} },

    pub const tensor = init;

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

    pub fn count(self: Tensor) usize {
        return self._shape.count();
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
        var batching: [Tensor.MAX_RANK][2]i8 = undefined;
        for (0..n_batching_axes) |i| {
            batching[i] = .{ @intCast(i), @intCast(i) };
        }
        return lhs_broad.dotGeneral(rhs, &contracting, batching[0..n_batching_axes]);
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
        const loc = lhs.getContext().location(@src(), "dot({f},{f},contracting={any},batching={any}", .{ lhs, rhs, contracting_axes, batching_axes });
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
                .precision = .fast,
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

    pub fn constant(dimz: anytype, val: zml.Data) Tensor {
        const sh = Shape.init(dimz, val.dtype());
        const ctx = CompilationContext.current().mlir_ctx;
        //const loc = CompilationContext.current().location(@src(), "dims={f}, value={}", .{ sh, val });
        const loc = mlir.Location.unknown(ctx);

        var constant_op = switch (val.dtype()) {
            inline .bool, .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64, .bf16, .f16, .f32, .f64 => |t| dialects.stablehlo.constant(ctx, &.{}, zml.mlir.Type.fromDType(ctx, t), val.constSlice(), loc),
            inline else => b: {
                const val_f32 = val.as(f32);
                break :b dialects.stablehlo.constant(ctx, &.{}, mlir.floatType(ctx, .f32), std.mem.asBytes(&val_f32), loc);
            },
        };
        CompilationContext.current().currentBlock().appendOwnedOperation(constant_op);

        if (sh.rank() > 0) {
            constant_op = dialects.stablehlo.broadcast_in_dim(ctx, constant_op.result(0), &.{}, mlir.rankedTensorType(sh.dims(), zml.mlir.Type.fromDType(ctx, sh.dtype())), loc);
            CompilationContext.current().currentBlock().appendOwnedOperation(constant_op);
        }
        return _result(sh, constant_op.result(0)).convert(val.dtype());
    }

    pub fn add(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "add", dialects.stablehlo.add)(self, other);
    }

    pub fn maximum(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "maximum", dialects.stablehlo.maximum)(self, other);
    }

    pub fn relu(self: Tensor) Tensor {
        return self.maximum(Tensor.constant(self.dims(), self.dtype().zero()));
    }

    pub fn flattenAll(self: Tensor) Tensor {
        // TODO: rename to just flatten, once flatten is moved to torch
        return self.reshape(.{self.count()});
    }

    pub fn reshape(self: Tensor, output_shape_: anytype) Tensor {
        const output_shape = self._shape.reshape(output_shape_);
        const ctx = self.getContext().mlir_ctx;
        const tensor_type = mlir.rankedTensorType(output_shape.dims(), zml.mlir.Type.fromDType(ctx, output_shape.dtype()));
        //const loc = self.getContext().location(@src(), "reshape({f})", .{output_shape});
        const reshape_op = dialects.stablehlo.reshape(self.getContext().mlir_ctx, self.value(), tensor_type, .unknown(ctx));
        CompilationContext.current().currentBlock().appendOwnedOperation(reshape_op);
        return _result(output_shape, reshape_op.result(0));
    }

    pub fn convert(self: Tensor, to: zml.DataType) Tensor {
        if (to == self.dtype()) {
            return self;
        }
        //const loc = self.getContext().location(@src(), "convert({f},to={s})", .{ self, @tagName(to) });

        const mlir_ctx = self.getContext().mlir_ctx;
        const res_type = mlir.rankedTensorType(self.dims(), zml.mlir.Type.fromDType(mlir_ctx, to));
        const op = dialects.stablehlo.convert(mlir_ctx, self.value(), res_type, .unknown(mlir_ctx));
        CompilationContext.current().currentBlock().appendOwnedOperation(op);
        return _result(self._shape.withDtype(to), op.result(0));
    }

    pub const ArgMaxRes = struct {
        values: Tensor,
        indices: Tensor,

        fn cmp(left: ArgMaxRes, right: ArgMaxRes) ArgMaxRes {
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

    fn binaryOp(
        src: std.builtin.SourceLocation,
        op_name: []const u8,
        comptime op_fn: fn (*mlir.Context, *const mlir.Value, *const mlir.Value, *const mlir.Location) *mlir.Operation,
    ) fn (Tensor, Tensor) Tensor {
        _ = op_name; // autofix
        _ = src; // autofix
        return struct {
            pub fn binaryOpHelper(self: Tensor, other: Tensor) Tensor {
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
                const ret = op_fn(ctx.mlir_ctx, self.value(), other.value(), .unknown(ctx.mlir_ctx));
                CompilationContext.current().currentBlock().appendOwnedOperation(ret);
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
    fc1: Layer,
    fc2: Layer,

    const Layer = struct {
        weight: Tensor,
        bias: Tensor,

        pub fn forward(self: Layer, input: Tensor) Tensor {
            _ = self; // autofix
            //return self.weight.matmul(input).add(self.bias).relu();
            return input.relu();
        }
    };

    pub fn init(buffer_store: zml.aio.BufferStore) Mnist {
        return .{
            .fc1 = .{
                .weight = .init(buffer_store.get("fc1.weight").?.shape()),
                .bias = .init(buffer_store.get("fc1.bias").?.shape()),
            },
            .fc2 = .{
                .weight = .init(buffer_store.get("fc2.weight").?.shape()),
                .bias = .init(buffer_store.get("fc2.bias").?.shape()),
            },
        };
    }

    /// just two linear layers + relu activation
    pub fn forward(self: Mnist, input: Tensor) Tensor {
        // std.log.info("Compiling for target: {s}", .{@tagName(input.getContext().target())});
        var x = input.flattenAll().convert(.f32);
        const layers: []const Layer = &.{ self.fc1, self.fc2 };
        for (layers) |layer| {
            x = layer.forward(x);
        }
        //return x.argMax(0).indices.convert(.u8);
        return x;
    }
};

const dialects = @import("mlir/dialects");
const mlir = @import("mlir");

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

    const input: Tensor = .init(zml.Shape.init(.{ 28, 28 }, .u8));

    compileModel(allocator, Mnist.forward, mnist, .{input});
}

const CompilationContext = struct {
    allocator: std.mem.Allocator,

    mlir_registry: *mlir.DialectRegistry,
    mlir_ctx: *mlir.Context,
    module: *mlir.Module,

    blocks: stdx.BoundedArray(*mlir.Block, 16) = .{},
    mappings: stdx.BoundedArray(std.AutoArrayHashMapUnmanaged(usize, usize), 16) = .{},

    pub fn init(allocator: std.mem.Allocator) CompilationContext {
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
    var compilation_context: CompilationContext = .init(allocator);
    defer compilation_context.deinit();

    //var context: Context = .{ .compilation_context = &compilation_context };
    //zml.meta.visit(struct {
    //    fn cb(inner_context: *Context, tensor: *const Tensor) void {
    //        std.debug.print("Registering {}\n", .{tensor.id});
    //        inner_context.compilation_context.registerTensor(tensor.id, inner_context.current_id) catch unreachable;
    //        inner_context.current_id += 1;
    //    }
    //}.cb, &context, &model);
    //zml.meta.visit(struct {
    //    fn cb(inner_context: *Context, tensor: *const Tensor) void {
    //        std.debug.print("Registering {}\n", .{tensor.id});
    //        inner_context.compilation_context.registerTensor(tensor.id, inner_context.current_id) catch unreachable;
    //        inner_context.current_id += 1;
    //    }
    //}.cb, &context, &args);

    emitMlir(&compilation_context, func, .{model} ++ args);
}

fn emitMlir(compilation_context: *CompilationContext, comptime func: anytype, args: stdx.meta.FnArgs(func)) void {
    var arena = std.heap.ArenaAllocator.init(compilation_context.allocator);
    defer arena.deinit();

    const block = mlir.Block.init(&.{}, &.{});
    errdefer block.deinit();

    compilation_context.pushBlock(block);
    defer compilation_context.popBlock();

    var input_types_list = std.array_list.Managed(*const mlir.Type).init(arena.allocator());
    defer input_types_list.deinit();

    const LocalContext = struct {
        compilation_context: *CompilationContext,
        input_types_list: std.array_list.Managed(*const mlir.Type),
        current_argument_id: usize = 0,
    };
    var context: LocalContext = .{
        .compilation_context = compilation_context,
        .input_types_list = input_types_list,
    };
    zml.meta.visit(struct {
        fn cb(inner_context: *LocalContext, tensor: *const Tensor) void {
            _ = inner_context.compilation_context.currentBlock().addArgument(
                mlir.rankedTensorType(tensor.dims(), zml.mlir.Type.fromDType(inner_context.compilation_context.mlir_ctx, tensor.dtype())),
                .unknown(inner_context.compilation_context.mlir_ctx),
            );
            inner_context.compilation_context.currentMapping().put(inner_context.compilation_context.allocator, tensor.id, inner_context.current_argument_id) catch unreachable;
            inner_context.input_types_list.append(mlir.rankedTensorType(tensor.dims(), zml.mlir.Type.fromDType(inner_context.compilation_context.mlir_ctx, tensor.dtype()))) catch unreachable;
            inner_context.current_argument_id += 1;
        }
    }.cb, &context, &args);

    {
        global_compilation_context = compilation_context;
        defer global_compilation_context = null;
        const result = @call(.auto, func, args);
        const fn_op = dialects.func.returns(compilation_context.mlir_ctx, &.{result.value()}, .unknown(compilation_context.mlir_ctx));
        compilation_context.currentBlock().appendOwnedOperation(fn_op);
    }

    const mlir_fn = dialects.func.func(compilation_context.mlir_ctx, .{
        .name = "mnist",
        .args = null,
        .args_attributes = null,
        .results = null,
        .results_attributes = null,
        .block = compilation_context.currentBlock(),
        .location = .unknown(compilation_context.mlir_ctx),
    });
    _ = mlir_fn; // autofix

    std.debug.print("{f}\n", .{compilation_context.currentBlock()});
}
