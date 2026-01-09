const std = @import("std");
const builtin = @import("builtin");

const dialects = @import("mlir/dialects");
const mlir = @import("mlir");
const stdx = @import("stdx");

const Buffer = @import("buffer.zig").Buffer;
const Memory = Buffer.Memory;
const Bufferized = @import("zml.zig").Bufferized;
const CompilationContext = @import("module.zig").CompilationContext;
const constants = @import("constants.zig");
const DataType = @import("dtype.zig").DataType;
const Memory = @import("platform.zig").Memory;
const meta = @import("meta.zig");
const mlirx = @import("mlirx.zig");
const ops = @import("ops.zig");
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;

pub const Tensor = struct {
    var current_id: std.atomic.Value(usize) = .{ .raw = 1 };

    id: usize,
    auto_broadcast: bool = false,
    _shape: Shape,
    _value: ?*const mlir.Value = null,

    pub fn init(shape_like: anytype, dt: DataType) Tensor {
        return .fromShape(.init(shape_like, dt));
    }

    pub fn fromShape(shape_: Shape) Tensor {
        return .{ .id = Tensor.current_id.fetchAdd(1, .seq_cst), ._shape = shape_ };
    }

    pub fn format(self: Tensor, writer: *std.Io.Writer) !void {
        try writer.print("Tensor({f})", .{self._shape});
    }

    /// Returns the shape of a Tensor.
    pub fn shape(self: Tensor) Shape {
        return self._shape;
    }

    /// Returns the datatype of a Tensor.
    pub fn dtype(self: Tensor) DataType {
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

    /// Internal use
    ///
    /// Creates a tensor from a Shape and an mlir.Value.
    pub fn _result(sh: Shape, val: *const mlir.Value) Tensor {
        const res: Tensor = .{
            ._shape = sh,
            ._value = val,
            .id = Tensor.current_id.fetchAdd(1, .seq_cst),
        };

        if (builtin.mode == .Debug) {
            // Check that the MLIR value actually have the same shape.
            const other = fromMlirValue(val);
            stdx.debug.internalAssert(sh.eql(other._shape), "Created a {f} from Mlir value but expected {f}", .{ other._shape, res._shape });
        }

        return res;
    }

    /// Creates a Tensor from a mlir.Value
    ///
    /// The shape is derived from the type of the mlir.Value.
    pub fn fromMlirValue(val: *const mlir.Value) Tensor {
        const ranked_tensor = val.type_().isA(mlir.RankedTensorType).?;
        const n = ranked_tensor.rank();

        stdx.debug.assert(n <= constants.MAX_RANK, "Can't represent MLIR tensor of rank {}, max supported rank is {}.", .{ n, constants.MAX_RANK });

        var sh: Shape = .{ ._dtype = mlirx.Type.toDType(mlirCtx(), ranked_tensor.elementType()) };
        for (0..n) |i| {
            sh._dims.appendAssumeCapacity(ranked_tensor.dimension(i));
        }
        sh._tags.resize(n) catch unreachable;
        sh._partitioning.resize(n) catch unreachable;

        return .{ ._shape = sh, ._value = val, .id = Tensor.current_id.fetchAdd(1, .seq_cst) };
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
    pub fn axes(self: Tensor, axes_: anytype) stdx.BoundedArray(u3, constants.MAX_RANK) {
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

    // TODO(Corentin)
    pub fn withSharding(self: Tensor, axes_: anytype) Tensor {
        _ = self;
        _ = axes_;
        @panic("Unimplemented");
    }

    pub fn toMemory(self: Tensor, kind: Memory) Tensor {
        const ctx = CompilationContext.current();
        if (ctx.platform.target == .cpu) return self;

        const frontend_attributes = mlir.dictionaryAttribute(ctx.mlir_ctx, &.{
            .named(ctx.mlir_ctx, "_xla_buffer_placement", mlir.stringAttribute(
                ctx.mlir_ctx,
                ctx.platform.memoryKind(kind),
            )),
        });

        const op = dialects.stablehlo.custom_call(
            ctx.mlir_ctx,
            &.{self.value()},
            &.{self.value().type_()},
            .{
                .call_target_name = "annotate_device_placement",
                .has_side_effect = true,
                .backend_config = .{ .original = "" },
                .additional_attributes = &.{
                    .named(ctx.mlir_ctx, "mhlo.frontend_attributes", frontend_attributes),
                },
            },
            .unknown(ctx.mlir_ctx),
        ).appendTo(currentBlock());

        var res = _result(self._shape, op.result(0));
        ctx.currentScope().id_to_output_memory_kind.put(ctx.currentScope().arena.allocator(), res.id, kind) catch unreachable;
        return res;
    }

    /// Returns a Tensor with new tag names.
    pub fn rename(self: Tensor, renames: anytype) Tensor {
        var res = self;
        res._shape = self._shape.rename(renames);
        return res;
    }

    pub fn renameAxis(self: Tensor, ax: i8, name: @EnumLiteral()) Tensor {
        var res = self;
        res._shape._tags.set(self.axis(ax), @tagName(name).ptr);
        return res;
    }

    /// Returns the mlir.Value associated with the Tensor.
    ///
    /// This will fail if used outside of a compilation context.
    pub fn value(self: Tensor) *const mlir.Value {
        if (CompilationContext.current().currentScope().id_to_argument.get(self.id)) |argument_index| {
            return CompilationContext.current().currentScope().block.argument(argument_index);
        } else if (self._value) |v| {
            return v;
        } else @panic("Something went really wrong, tensor is not an argument nor has an mlir.Value");
    }

    /// Tell PJRT compiler that memory should be reuse between the two tensors.
    /// The compiler is already aggressively reusing tensors for intermediate results,
    /// but this API allows to reuse buffer between input and output arguments
    /// of a given function.
    /// Note this is visible from the outside. The caller of a function with donations
    /// is not allowed to reuse the donated input buffer after the call.
    /// For `reuseBuffer` to be effective, it needs to propagate all the way through the output.
    pub fn reuseBuffer(self: Tensor, origin: Tensor) Tensor {
        const compilation_context = CompilationContext.current();
        const scope = compilation_context.currentScope();
        if (scope.id_to_argument.get(origin.id)) |argument_index| {
            const gop = scope.id_to_donation.getOrPut(scope.arena.allocator(), self.id) catch unreachable;
            gop.value_ptr.* = argument_index;
        } else if (scope.id_to_donation.get(origin.id)) |origin_donation| {
            const gop = scope.id_to_donation.getOrPut(scope.arena.allocator(), self.id) catch unreachable;
            gop.value_ptr.* = origin_donation;
        }
        return self;
    }

    /// Returns a Tensor containing the absolute value of each element of the input Tensor.
    pub fn abs(self: Tensor) Tensor {
        const op = dialects.stablehlo.abs(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        const dt = switch (self.dtype()) {
            .c64 => .f32,
            .c128 => .f64,
            else => self.dtype(),
        };

        return _result(self._shape.withDtype(dt), op.result(0));
    }

    /// Returns a Tensor whose elements have been bitcast to a target datatype.
    ///
    /// The Tensor shape needs to be compatible with the target datatype.
    pub fn bitCast(self: Tensor, dt: DataType) Tensor {
        const src_bit_size = self.dtype().bitSizeOf();
        const tgt_bit_size = dt.bitSizeOf();
        var res_shape = if (src_bit_size == tgt_bit_size)
            self._shape
        else if (src_bit_size > tgt_bit_size) gt: {
            const new_dim = std.math.divExact(u16, src_bit_size, tgt_bit_size) catch std.debug.panic("bitcast expects target datatype to be a multiple of source datatype when upcasting, got {} (bitsize of {}) and {} (bitsize of {})", .{ self.dtype(), src_bit_size, dt, tgt_bit_size });
            var res = self._shape;
            res = res.append(.{ .bitcast = new_dim });
            break :gt res;
        } else lt: {
            // several contiguous elements of self maps to one element of the result
            stdx.debug.assert(self.dim(-1) * src_bit_size == tgt_bit_size, "bitcast expects elements of the input tensor last dimension to map to one element of the target datatype, got {0} elements (bitsize of {0}x{1}={2}) and {3} (bitsize of {4})", .{ self.dim(-1), src_bit_size, self.dim(-1) * src_bit_size, dt, tgt_bit_size });
            break :lt self._shape.remove(-1);
        };

        res_shape = res_shape.withDtype(dt);

        const op = dialects.stablehlo.bitcast_convert(
            mlirCtx(),
            self.value(),
            mlir.rankedTensorType(res_shape.dims(), mlirx.Type.fromDType(mlirCtx(), res_shape.dtype())),
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());

        return _result(res_shape, op.result(0));
    }

    /// Returns the given tensor as one contiguous buffer of bytes.
    pub fn bytes(self: Tensor) Tensor {
        return self.bitCast(.u8).flatten().withTags(.{.bytes});
    }

    /// Returns a Tensor containing the element-wise number of leading 0 bits in the input Tensor.
    pub fn countLeadingZeros(self: Tensor) Tensor {
        const op = dialects.stablehlo.count_leading_zeros(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, op.result(0));
    }

    /// Returns a Tensor containing booleans indicating if each element of the input Tensor is finite.
    pub fn isFinite(self: Tensor) Tensor {
        const op = dialects.stablehlo.is_finite(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape.withDtype(.bool), op.result(0));
    }

    /// Returns a Tensor containing the element-wise number of bits set in the input Tensor.
    pub fn popcnt(self: Tensor) Tensor {
        stdx.debug.assert(self.dtype().isInteger(), "popcnt expects tensor type to be an integer, got {}", .{self.dtype()});
        const op = dialects.stablehlo.popcnt(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, op.result(0));
    }

    /// Returns a Tensor containing the sign of the input Tensor element-wise.
    pub fn sign(self: Tensor) Tensor {
        const op = dialects.stablehlo.sign(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, op.result(0));
    }

    /// Returns a Tensor containing the element-wise remainder of dividend 'self' and divisor 'other'.
    ///
    /// See https://pytorch.org/docs/stable/generated/torch.fmod.html for more details.
    pub fn fmod(self: Tensor, divisor: f32) Tensor {
        return self.remainder(scalar(divisor, .f32).broadcast(self._shape, &.{}));
    }

    test fmod {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        const inputs: [2][6]f32 = .{ .{ -3.0, -2, -1, 1, 2, 3 }, .{ 1, 2, 3, 4, 5, -5 } };
        const expectations: [2][6]f32 = .{ .{ -1.0, -0.0, -1.0, 1.0, 0.0, 1.0 }, .{ 1.0000, 0.5000, 0.0000, 1.0000, 0.5000, -0.5000 } };
        const divisors: [2]f32 = .{ 2, -1.5 };

        inline for (inputs, expectations, divisors) |i, e, d| {
            const input: zml.Tensor = .init(.{6}, .f32);

            const exe = try zml.module.compile(std.testing.allocator, std.testing.io, Tensor.fmod, .{ input, d }, platform);
            defer exe.deinit();

            var input_buffer = try zml.Buffer.fromBytes(std.testing.io, platform, input.shape(), std.mem.sliceAsBytes(&i));
            defer input_buffer.deinit();

            const output = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Tensor.fmod, .{input_buffer});

            try zml.testing.expectClose(std.testing.io, zml.Slice.init(zml.Shape.init(.{6}, .f32), std.mem.sliceAsBytes(&e)), output, 1e-4);
        }
    }

    /// Returns a Tensor containing the element-wise left-shift operation of 'self' by 'other'.
    pub fn shiftLeft(self: Tensor, other: Tensor) Tensor {
        return binaryOp("shiftLeft", dialects.stablehlo.shift_left)(self, other);
    }

    /// Returns a Tensor containing the element-wise arithmetic right-shift operation of 'self' by 'other'.
    pub fn shiftRightArithmetic(self: Tensor, other: Tensor) Tensor {
        return binaryOp("shiftRightArithmetic", dialects.stablehlo.shift_right_arithmetic)(self, other);
    }

    /// Returns a Tensor containing the element-wise logical right-shift operation of 'self' by 'other'.
    pub fn shiftRightLogical(self: Tensor, other: Tensor) Tensor {
        return binaryOp("shiftRightLogical", dialects.stablehlo.shift_right_logical)(self, other);
    }

    /// Returns the Cholesky decomposition of the input Tensor.
    ///
    /// 'lower' controls the form of the output Tensor. The output will be lower-triangular if 'lower' is true
    /// and upper-triangular otherwise.
    pub fn cholesky(self: Tensor, lower: bool) Tensor {
        stdx.debug.assert(self.rank() <= 2, "cholesky expects tensor rank to be <= 2, got {}", .{self.rank()});

        const op = dialects.stablehlo.cholesky(mlirCtx(), self.value(), lower, .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, op.result(0));
    }

    /// Solves the system of linear equations formed by the input tensors.
    pub fn triangularSolve(self: Tensor, other: Tensor, opts: dialects.stablehlo.TriangularSolveOpts) Tensor {
        stdx.debug.assert(self.dtype() == other.dtype(), "triangularSolve expects tensors to be of the same type, got {} and {}", .{ self.dtype(), other.dtype() });
        stdx.debug.assert(self.rank() <= 2 and self.rank() == other.rank(), "triangularSolve expects tensors to have the same rank and be <= 2, got {} and {}", .{ self.rank(), other.rank() });

        const op = dialects.stablehlo.triangular_solve(mlirCtx(), self.value(), other.value(), opts, .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, op.result(0));
    }

    /// Returns a Tensor containing the element-wise rounding towards the nearest integer, breaking ties away from zero, of the input Tensor.
    pub fn roundNearestAfz(self: Tensor) Tensor {
        stdx.debug.assert(self.dtype().isFloat(), "roundNearestAfz expects tensor type to be a float, got {}", .{self.dtype()});

        const op = dialects.stablehlo.round_nearest_afz(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, op.result(0));
    }

    /// Returns a Tensor containing the element-wise rounding towards the nearest integer, breaking ties towards the even integer, of the input Tensor.
    pub fn roundNearestEven(self: Tensor) Tensor {
        stdx.debug.assert(self.dtype().isFloat(), "roundNearestEven expects tensor type to be a float, got {}", .{self.dtype()});

        const op = dialects.stablehlo.round_nearest_even(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, op.result(0));
    }

    /// Returns a Tensor of complex number converted from a pair of real and imaginary Tensors.
    pub fn complex(re: Tensor, im: Tensor) Tensor {
        stdx.debug.assert(re._shape.eql(im._shape), "complex expects tensor shapes to match, got {f} and {f}", .{ re._shape, im._shape });
        stdx.debug.assert(re.dtype() == .f32 or re.dtype() == .f64, "complex expects tensors type to be f32 or f64, got {}", .{re.dtype()});

        const op = dialects.stablehlo.complex(mlirCtx(), re.value(), im.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        const dt: DataType = if (re.dtype() == .f32) .c64 else .c128;
        return _result(re._shape.withDtype(dt), op.result(0));
    }

    /// Returns a Tensor containing the element-wise real part of the input Tensor.
    ///
    /// Tensor type can float or complex.
    pub fn real(self: Tensor) Tensor {
        stdx.debug.assert(self.dtype().isComplex() or self.dtype().isFloat(), "real expects tensor type to be a float or a complex, got {}", .{self.dtype()});

        if (self.dtype().isFloat()) {
            return self;
        }

        const dt: DataType = switch (self.dtype()) {
            .c64 => .f32,
            .c128 => .f64,
            else => unreachable,
        };
        const op = dialects.stablehlo.real(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape.withDtype(dt), op.result(0));
    }

    /// Returns a Tensor containing the element-wise imaginary part of the input Tensor.
    ///
    /// Tensor type can float or complex.
    pub fn imag(self: Tensor) Tensor {
        stdx.debug.assert(self.dtype().isFloat() or self.dtype().isComplex(), "imag expects tensor type to be a float or a complex, got {}", .{self.dtype()});

        // Real tensors don't have imaginary part.
        if (self.dtype().isFloat()) {
            return Tensor.constant(self.dtype().zero()).broad(self._shape);
        }

        const dt: DataType = switch (self.dtype()) {
            .bf16, .f16, .f32, .f64 => self.dtype(),
            .c64 => .f32,
            .c128 => .f64,
            else => unreachable,
        };
        const op = dialects.stablehlo.imag(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape.withDtype(dt), op.result(0));
    }

    /// Returns the Fast Fourier Transform (FFT) of the input Tensor.
    pub fn fft(self: Tensor, opts: dialects.stablehlo.FftOpts) Tensor {
        // TODO: support tagged API.

        stdx.debug.assert(1 <= opts.length.len and opts.length.len <= 3, "fft expects 'opts.length' length to be between 1 and 3 (inclusive), got {}", .{opts.length.len});
        stdx.debug.assert(opts.length.len <= self.rank(), "fft expects 'opts.length' length to be less than tensor rank, got {} and {}", .{ opts.length.len, self.rank() });

        const sh = switch (opts.kind) {
            .FFT, .IFFT => blk: {
                stdx.debug.assert(self.dtype().isComplex(), "fft({any}) expects tensor type to be complex, got {}", .{ opts, self.dtype() });

                break :blk self._shape;
            },
            .RFFT => blk: {
                stdx.debug.assert(self.dtype() == .f32 or self.dtype() == .f64, "fft({}) expects tensor type to be f32 or f64, got {}", .{ opts, self.dtype() });
                stdx.debug.assert(std.mem.eql(i64, self.dims()[self.rank() - opts.length.len ..], opts.length), "fft({}) expects tensor last dimensions to match given lengths, got {} and {}", .{ opts, self.dims()[self.rank() - opts.length.len ..].len, opts.length.len });

                const dt: DataType = switch (self.dtype()) {
                    .f32 => .c64,
                    else => .c128,
                };
                const shape_ = self._shape.setDim(-1, @divExact(self.dim(-1), 2) + 1);
                break :blk shape_.withDtype(dt);
            },
            .IRFFT => blk: {
                stdx.debug.assert(self.dtype().isComplex(), "fft({any}) expects tensor type to be complex, got {}", .{ opts, self.dtype() });
                stdx.debug.assert(std.mem.eql(i64, self.dims()[self.rank() - opts.length.len ..], opts.length), "fft({any}) expects tensor last dimensions to match given lengths, got {} and {}", .{ opts, self.dims()[self.rank() - opts.length.len ..].len, opts.length.len });

                const dt: DataType = switch (self.dtype()) {
                    .c64 => .f32,
                    else => .f64,
                };
                const shape_ = self._shape.setDim(-1, @divExact(self.dim(-1) - 1, 2));
                break :blk shape_.withDtype(dt);
            },
        };

        const op = dialects.stablehlo.fft(mlirCtx(), self.value(), opts, .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(sh, op.result(0));
    }

    pub const Rng = struct {
        _state: Tensor,
        algorithm: dialects.stablehlo.RngAlgorithm.Type = .DEFAULT,

        pub fn init() Rng {
            return .{ ._state = .init(.{2}, .u64) };
        }

        pub fn initBuffer(platform: *const Platform, seed: u128, io: std.Io) !Bufferized(Rng) {
            return .{
                ._state = try .fromBytes(io, platform, Shape.init(.{2}, .u64), std.mem.asBytes(&seed)),
            };
        }

        pub fn deinitBuffer(self: *Bufferized(Rng)) void {
            self._state.deinit();
        }

        /// Returns a Tensor of the given shape, filled with uniform random bits, and a new Rng state.
        ///
        /// The given Rng state should not be used anymore (or you'll get the same numbers again).
        /// The output is guaranteed to be deterministic function of `self` Rng state,
        /// but it is not guaranteed to be deterministic between implementations.
        pub fn bitGenerator(self: Rng, sh: Shape) struct { Rng, Tensor } {
            const op = dialects.stablehlo.rng_bit_generator(
                mlirCtx(),
                self.algorithm,
                self._state.value(),
                mlir.rankedTensorType(self._state.dims(), mlirx.Type.fromDType(mlirCtx(), self._state.dtype())),
                mlir.rankedTensorType(sh.dims(), mlirx.Type.fromDType(mlirCtx(), sh.dtype())),
                .unknown(mlirCtx()),
            ).appendTo(currentBlock());
            return .{ self.update(op.result(0)), _result(sh, op.result(1)) };
        }

        fn update(self: Rng, new_state: *const mlir.Value) Rng {
            return .{
                ._state = _result(self._state._shape, new_state).reuseBuffer(self._state),
                .algorithm = self.algorithm,
            };
        }

        /// Returns a Tensor of the given shape, filled with uniformly sampled floating point numbers from an interval,
        /// and a new Rng state.
        ///
        /// https://en.wikipedia.org/wiki/Continuous_uniform_distribution
        pub fn uniform(
            self: Rng,
            shape_: Shape,
            opts: struct { min: f64 = 0, max: f64 = 1 },
        ) struct { Rng, Tensor } {
            const dt = if (shape_.dtype().isFloat()) shape_.dtype() else .f32;

            const mantissa_bit_count = @import("dtype.zig").mantissaSize(dt);
            const bit_count: usize = dt.bitSizeOf();
            const rng_bit_count = if (mantissa_bit_count < 8) 8 else bit_count;
            const uint_dtype: DataType = switch (bit_count) {
                8 => .u8,
                16 => .u16,
                32 => .u32,
                64 => .u64,
                else => stdx.debug.panic("uniform don't support non-byte aligned dtype. Got: {f}", .{shape_}),
            };

            const rng, const bits = self.bitGenerator(shape_.withDtype(uint_dtype));

            // Erase bits outside of mantissa.
            var float_bits = bits.shiftRightLogical(scalar(rng_bit_count - mantissa_bit_count, uint_dtype));

            // Set exponent bits to represent e^0 (eg 127 for f32).
            float_bits = float_bits.logical(.OR, scalar(1, dt).bitCast(uint_dtype));

            // float_bits now uniformly represents number in [1, 2[ range.
            // Let's convert to floats, and subtract one to go to [0, 1[ range.
            var floats = float_bits.bitCast(dt).sub(scalar(1, dt));
            floats = floats.mul(scalar(opts.max - opts.min, dt)).addConstant(opts.min);

            // Convert back to integer if needed.
            return .{ rng, floats.convert(shape_.dtype()) };
        }

        test uniform {
            const zml = @import("zml.zig");
            const Stats = struct {
                const Stats = @This();

                mean: Tensor,
                variance: Tensor,
                min: Tensor,
                max: Tensor,

                pub fn uniformStats(
                    rand: Rng,
                    shape_: Shape,
                    opts: struct { min: f64, max: f64 },
                ) struct { Rng, Stats } {
                    const rng, const data = rand.uniform(shape_, .{ .min = opts.min, .max = opts.max });
                    const mean_ = data.mean(0);
                    const variance = data.sub(mean_.broad(data.shape())).pow(Tensor.scalar(2, .f32)).mean(0);
                    return .{ rng, .{
                        .mean = mean_,
                        .variance = variance,
                        .min = data.min(0),
                        .max = data.max(0),
                    } };
                }
            };

            const platform = zml.testing.env();
            // Compute stats over a uniform distribution on [-2, 10].
            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Stats.uniformStats, .{ Rng.init(), Shape.init(.{1024}, .f32), .{ .min = -2, .max = 10 } }, platform);
            defer exe.deinit();

            var rng_buffer = try Rng.initBuffer(platform, 1234, std.testing.io);
            defer rng_buffer._state.deinit();

            var rand, var stats = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Stats.uniformStats, .{rng_buffer});
            defer rand._state.deinit();
            defer stats.mean.deinit();
            defer stats.variance.deinit();
            defer stats.min.deinit();
            defer stats.max.deinit();

            // Check the Rng state has been modified.
            try std.testing.expect(try rand._state.getValue(u128, std.testing.io) != 1234);

            // Check the mean and variance are close to theoritical values.
            const mean_ = try stats.mean.getValue(f32, std.testing.io);
            try std.testing.expectApproxEqAbs(4, mean_, 0.03);

            const variance = try stats.variance.getValue(f32, std.testing.io);
            try std.testing.expectApproxEqAbs(12.0 * 12.0 / 12.0, variance, 0.01);

            // Check that no value is outside of the interval
            // and we have samples close to the edges.
            const min_ = try stats.min.getValue(f32, std.testing.io);
            try std.testing.expect(min_ >= -2);
            try std.testing.expectApproxEqAbs(-2, min_, 0.05);

            const max_ = try stats.max.getValue(f32, std.testing.io);
            try std.testing.expect(max_ < 10);
            try std.testing.expectApproxEqAbs(10, max_, 0.05);
        }

        /// Returns a Tensor of the given shape, filled with floating point numbers sampled from a normal distribution.
        ///
        /// Note: this uses stablehlo.rng which is deprecated.
        /// https://github.com/openxla/stablehlo/blob/main/rfcs/20240503-opset-deprecations.md
        pub fn normal(sh: Shape, opts: struct { mean: f64 = 0, stddev: f64 = 1 }) Tensor {
            stdx.debug.assert(sh.dtype().isFloat(), "normal expects tensor type to be a float, got {}", .{sh.dtype()});

            const a = Tensor.constant(DataType.Value.init(sh.dtype(), opts.mean));
            const b = Tensor.constant(DataType.Value.init(sh.dtype(), opts.stddev));
            const res_tensor_shape = Tensor.constantTensor(Shape.init(.{sh.rank()}, .i64), std.mem.sliceAsBytes(sh.dims()));
            const op = dialects.stablehlo.rng(mlirCtx(), a.value(), b.value(), res_tensor_shape.value(), .NORMAL, .unknown(mlirCtx())).appendTo(currentBlock());
            return _result(sh, op.result(0));
        }

        /// Returns a Tensor of the given shape, filled with floating point numbers sampled from a Gumbel distribution, and a new Rng state.
        ///
        /// Often used in ML because of the reparametrization tricks.
        /// Sampling from a gumbel distribution is equivalent to sample
        /// from a softmax distribution, but doesn't require to compute the sum of exponentials.
        /// https://en.wikipedia.org/wiki/Gumbel_distribution#Gumbel_reparametrization_tricks
        /// See `sampleTokens` for a practical use case.
        /// Note: we only implement the μ=0, β=1 version.
        pub fn gumbel(self: Rng, shape_: Shape) struct { Rng, Tensor } {
            const rand, const u = self.uniform(
                // Always use .f32 to have a big enough mantissa.
                shape_.withDtype(.f32),
                // We don't want 0 to be sampled otherwise `log` will return -inf.
                .{ .min = std.math.floatEps(f32), .max = 1 },
            );
            return .{ rand, u.log().scale(-1).log().scale(-1).convert(shape_.dtype()) };
        }

        test gumbel {
            const zml = @import("zml.zig");
            const Stats = struct {
                const Stats = @This();

                mean: Tensor,
                variance: Tensor,
                actual_dist: Tensor,

                pub fn gumbelStats(rand: Rng, target_dist: Tensor) struct { Rng, Stats } {
                    const s = Shape.init(.{ .n = 1024, .d = 4 }, .f32);
                    const rng, const data = rand.gumbel(s);
                    const flat = data.flatten();
                    const mean_ = flat.mean(0);
                    const variance = flat.sub(mean_.broad(flat.shape())).pow(Tensor.scalar(2, .f32)).mean(0);

                    // Test out the gumbel reparametrization trick
                    var x = target_dist.log().withTags(.{.d}).broad(s);
                    x = x.add(data);
                    const samples = x.argMax(.d).indices.squeeze(.d);

                    // count 0, 1, 2 and 3 in samples:
                    // - map 0 to 1, 1 to 2**16, 2 to 2**32, 3 to N**58
                    // - sum in u64
                    // - split to [4]u16
                    const powers = blk: {
                        var powers: [4]u64 = undefined;
                        for (&powers, 0..) |*p, i| p.* = std.math.pow(u64, 2, i * 16);
                        break :blk powers;
                    };
                    const values = Tensor.constantTensor(Shape.init(.{4}, .u64), std.mem.sliceAsBytes(&powers)).withTags(.{.d});
                    const counts = values.gather(.{ .d = samples }, .{}).sum(.n).bitCast(.u16);
                    const actual_dist = counts.reshape(target_dist.shape()).convert(target_dist.dtype()).divByConst(s.dim(.n));
                    return .{ rng, .{ .mean = mean_, .variance = variance, .actual_dist = actual_dist } };
                }
            };

            const platform = zml.testing.env();
            const tgt_dist_data = [_]f32{ 2.0, 1.0, 4.0, 3.0 };
            const tgt_dist: Tensor = .init(.{tgt_dist_data.len}, .f32);

            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Stats.gumbelStats, .{ Rng.init(), tgt_dist }, platform);
            defer exe.deinit();

            var rng_buffer = try Rng.initBuffer(platform, 1234, std.testing.io);
            defer rng_buffer._state.deinit();
            var tgt_dist_buffer: Buffer = try .fromBytes(std.testing.io, platform, tgt_dist.shape(), std.mem.sliceAsBytes(&tgt_dist_data));
            defer tgt_dist_buffer.deinit();

            var rand, var stats = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Stats.gumbelStats, .{ rng_buffer, tgt_dist_buffer });
            defer rand._state.deinit();
            defer stats.mean.deinit();
            defer stats.variance.deinit();
            defer stats.actual_dist.deinit();

            // Check the Rng state has been modified.
            try std.testing.expect(try rand._state.getValue(i128, std.testing.io) != 1234);

            // Check the mean and variance are close to theoritical values.
            const mean_ = try stats.mean.getValue(f32, std.testing.io);
            try std.testing.expectApproxEqAbs(0.5772, mean_, 0.02);

            const variance = try stats.variance.getValue(f32, std.testing.io);
            const pi = std.math.pi;
            try std.testing.expectApproxEqAbs(pi * pi / 6.0, variance, 0.03);

            // Check the distribution obtained with the gumbel trick matches the target distribution.
            const actual_dist = try stats.actual_dist.getValue([4]f32, std.testing.io);
            for (tgt_dist_data, actual_dist) |tgt, actual| {
                // We normalize tgt_dist to make it a well formed distribution.
                // We didn't do it before calling gumbel, because the gumbel trick
                // doesn't require normalized distributions as input.
                try std.testing.expectApproxEqAbs(tgt / 10.0, actual, 0.05);
            }
        }
    };

    /// Returns a Tensor containing the element-wise conversion to another floating point type.
    pub fn reducePrecision(self: Tensor, exponent_bits: i32, mantissa_bits: i32) Tensor {
        stdx.debug.assert(self.dtype().isFloat(), "reducePrecision expects tensor type to be a float, got {}", .{self.dtype()});
        stdx.debug.assert(1 <= exponent_bits, "reducePrecision expects 'exponent_bits' to be >= 1, got {}", .{exponent_bits});
        stdx.debug.assert(0 <= mantissa_bits, "reducePrecision expects 'mantissa_bits' to be positive, got {}", .{mantissa_bits});

        const op = dialects.stablehlo.reduce_precision(mlirCtx(), self.value(), exponent_bits, mantissa_bits, .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, op.result(0));
    }

    fn convolution(self: Tensor, other: Tensor, opts: dialects.stablehlo.ConvolutionOpts) Tensor {
        stdx.debug.assert(self.rank() == other.rank(), "convolution expects tensor ranks to match, got {} and {}", .{ self.rank(), other.rank() });
        const N = self.rank();
        stdx.debug.guard(opts.window_strides.len == N - 2, @src());
        for (opts.window_strides) |s| stdx.debug.guard(0 < s, @src());
        stdx.debug.guard(opts.lhs_dilation.len == N - 2, @src());
        for (opts.lhs_dilation) |d| stdx.debug.guard(0 < d, @src());
        stdx.debug.guard(opts.rhs_dilation.len == N - 2, @src());
        for (opts.rhs_dilation) |d| stdx.debug.guard(0 < d, @src());
        stdx.debug.guard(opts.window_reversal.len == N - 2, @src());
        stdx.debug.guard(@rem(self.dim(opts.input_batch_dimension), opts.batch_group_count) == 0, @src());
        stdx.debug.guard(@rem(self.dim(opts.input_feature_dimension), opts.feature_group_count) == 0, @src());
        stdx.debug.guard(opts.input_spatial_dimensions.len == N - 2, @src());
        stdx.debug.guard(opts.input_batch_dimension != opts.input_feature_dimension, @src());
        stdx.debug.guard(0 <= opts.input_batch_dimension and opts.input_batch_dimension < N, @src());
        stdx.debug.guard(0 <= opts.input_feature_dimension and opts.input_feature_dimension < N, @src());
        for (opts.input_spatial_dimensions, 0..) |d, i| {
            stdx.debug.guard(d != opts.input_batch_dimension, @src());
            stdx.debug.guard(d != opts.input_feature_dimension, @src());
            stdx.debug.guard(0 <= d and d < N, @src());
            if (i < opts.input_spatial_dimensions.len - 1) continue;
            stdx.debug.guard(std.mem.indexOfScalar(i64, opts.input_spatial_dimensions[i + 1 ..], d) == null, @src());
        }
        stdx.debug.guard(other.dim(opts.kernel_input_feature_dimension) == @divTrunc(self.dim(opts.input_feature_dimension), opts.feature_group_count), @src());
        stdx.debug.guard(@rem(other.dim(opts.kernel_output_feature_dimension), opts.batch_group_count) == 0, @src());
        stdx.debug.guard(@rem(other.dim(opts.kernel_output_feature_dimension), opts.feature_group_count) == 0, @src());
        stdx.debug.guard(opts.kernel_spatial_dimensions.len == N - 2, @src());
        stdx.debug.guard(opts.kernel_input_feature_dimension != opts.kernel_output_feature_dimension, @src());
        stdx.debug.guard(0 <= opts.kernel_input_feature_dimension and opts.kernel_input_feature_dimension < N, @src());
        stdx.debug.guard(0 <= opts.kernel_output_feature_dimension and opts.kernel_output_feature_dimension < N, @src());
        for (opts.kernel_spatial_dimensions, 0..) |d, i| {
            stdx.debug.guard(d != opts.kernel_input_feature_dimension, @src());
            stdx.debug.guard(d != opts.kernel_output_feature_dimension, @src());
            stdx.debug.guard(0 <= d and d < N, @src());
            if (i < opts.kernel_spatial_dimensions.len - 1) continue;
            stdx.debug.guard(std.mem.indexOfScalar(i64, opts.kernel_spatial_dimensions[i + 1 ..], d) == null, @src());
        }
        stdx.debug.guard(opts.output_spatial_dimensions.len == N - 2, @src());
        stdx.debug.guard(opts.output_batch_dimension != opts.output_feature_dimension, @src());
        stdx.debug.guard(0 <= opts.output_batch_dimension and opts.output_batch_dimension < N, @src());
        stdx.debug.guard(0 <= opts.output_feature_dimension and opts.output_feature_dimension < N, @src());
        for (opts.output_spatial_dimensions, 0..) |d, i| {
            stdx.debug.guard(d != opts.output_batch_dimension, @src());
            stdx.debug.guard(d != opts.output_feature_dimension, @src());
            stdx.debug.guard(0 <= d and d < N, @src());
            if (i < opts.output_spatial_dimensions.len - 1) continue;
            stdx.debug.guard(std.mem.indexOfScalar(i64, opts.output_spatial_dimensions[i + 1 ..], d) == null, @src());
        }
        stdx.debug.guard(0 < opts.feature_group_count, @src());
        stdx.debug.guard(0 < opts.batch_group_count, @src());
        stdx.debug.guard(opts.feature_group_count == 1 or opts.batch_group_count == 1, @src());
        var used_opts = opts;
        used_opts.pad_shape = &.{ @intCast(N - 2), 2 };
        used_opts.precision_config = &.{ .DEFAULT, .DEFAULT };

        var new_shape = self._shape;
        var res_dim: i64 = undefined;

        for (0..N) |i| {
            if (i == @as(usize, @intCast(opts.output_batch_dimension))) {
                res_dim = @divTrunc(self.dim(opts.input_batch_dimension), opts.batch_group_count);
            } else if (i == @as(usize, @intCast(opts.output_feature_dimension))) {
                res_dim = other.dim(opts.kernel_output_feature_dimension);
            } else {
                // calculate spatial dimension value
                const spatial_dim: usize = std.mem.indexOfScalar(i64, opts.output_spatial_dimensions, @as(i64, @intCast(i))).?;
                const lhs_dim = opts.input_spatial_dimensions[spatial_dim];
                const rhs_dim = opts.kernel_spatial_dimensions[spatial_dim];
                const dilated_input_shape_lhs_dim: i64 = if (self.dim(lhs_dim) == 0) 0 else (self.dim(lhs_dim) - 1) * opts.lhs_dilation[spatial_dim] + 1;
                const left_pad_value, const right_pad_value = if (opts.pad_value.len == 1)
                    .{ opts.pad_value[0], opts.pad_value[0] }
                else
                    .{ opts.pad_value[2 * spatial_dim], opts.pad_value[2 * spatial_dim + 1] };
                const padded_input_shape_lhs_dim = left_pad_value + dilated_input_shape_lhs_dim + right_pad_value;
                const dilated_window_shape_lhs_dim: i64 = if (other.dim(rhs_dim) == 0) 0 else (other.dim(rhs_dim) - 1) * opts.rhs_dilation[spatial_dim] + 1;
                const is_empty_window_lhs_dim = padded_input_shape_lhs_dim == 0 or dilated_window_shape_lhs_dim > padded_input_shape_lhs_dim;
                res_dim = if (is_empty_window_lhs_dim) 0 else @divTrunc(padded_input_shape_lhs_dim - dilated_window_shape_lhs_dim, opts.window_strides[spatial_dim]) + 1;
            }

            new_shape = new_shape.set(i, res_dim);
        }

        // inferred shape '[1, 256, 1, 12008]' is incompatible with return type of operation '[1, 256, 1, 11978]'
        const op = dialects.stablehlo.convolution(
            mlirCtx(),
            self.value(),
            other.value(),
            mlir.rankedTensorType(new_shape.dims(), mlirx.Type.fromDType(mlirCtx(), new_shape.dtype())),
            used_opts,
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());

        return _result(new_shape, op.result(0));
    }

    /// Returns a Tensor containing the result of the 1D convolution of 'input' by 'kernel'.
    pub fn conv1d(
        input: Tensor,
        kernel: Tensor,
        opts: struct {
            window_strides: i64 = 1,
            padding: []const i64 = &.{ 0, 0 },
            lhs_dilation: i64 = 1,
            rhs_dilation: i64 = 1,
            window_reversal: bool = false,
            input_batch_dimension: i64 = 0,
            input_feature_dimension: i64 = 1,
            input_spatial_dimensions: i64 = 2,
            kernel_output_feature_dimension: i64 = 0,
            kernel_input_feature_dimension: i64 = 1,
            kernel_spatial_dimensions: i64 = 2,
            output_batch_dimension: i64 = 0,
            output_feature_dimension: i64 = 1,
            output_spatial_dimensions: i64 = 2,
            feature_group_count: i64 = 1,
            batch_group_count: i64 = 1,
        },
    ) Tensor {
        return input.convolution(kernel, .{
            .window_strides = &.{opts.window_strides},
            .pad_value = opts.padding,
            .lhs_dilation = &.{opts.lhs_dilation},
            .rhs_dilation = &.{opts.rhs_dilation},
            .window_reversal = &.{opts.window_reversal},
            .input_batch_dimension = opts.input_batch_dimension,
            .input_feature_dimension = opts.input_feature_dimension,
            .input_spatial_dimensions = &.{opts.input_spatial_dimensions},
            .kernel_input_feature_dimension = opts.kernel_input_feature_dimension,
            .kernel_output_feature_dimension = opts.kernel_output_feature_dimension,
            .kernel_spatial_dimensions = &.{opts.kernel_spatial_dimensions},
            .output_batch_dimension = opts.output_batch_dimension,
            .output_feature_dimension = opts.output_feature_dimension,
            .output_spatial_dimensions = &.{opts.output_spatial_dimensions},
            .feature_group_count = opts.feature_group_count,
            .batch_group_count = opts.batch_group_count,
        });
    }

    /// Returns a Tensor containing the result of the 2D convolution of 'input' by 'kernel'.
    /// Defaults values correspond to a (B, C_in, W, H) image, (C_out, C_in, W, H) kernel weights and (B, C_out, W, H) output.
    pub fn conv2d(
        input: Tensor,
        kernel: Tensor,
        opts: struct {
            window_strides: []const i64 = &.{ 1, 1 },
            padding: []const i64 = &.{ 0, 0, 0, 0 },
            lhs_dilation: []const i64 = &.{ 1, 1 },
            rhs_dilation: []const i64 = &.{ 1, 1 },
            window_reversal: []const bool = &.{ false, false },
            input_batch_dimension: i64 = 0,
            input_feature_dimension: i64 = 1,
            input_spatial_dimensions: []const i64 = &.{ 2, 3 },
            kernel_input_feature_dimension: i64 = 1,
            kernel_output_feature_dimension: i64 = 0,
            kernel_spatial_dimensions: []const i64 = &.{ 2, 3 },
            output_batch_dimension: i64 = 0,
            output_feature_dimension: i64 = 1,
            output_spatial_dimensions: []const i64 = &.{ 2, 3 },
            feature_group_count: i64 = 1,
            batch_group_count: i64 = 1,
        },
    ) Tensor {
        return input.convolution(kernel, .{
            .window_strides = opts.window_strides,
            .pad_value = opts.padding,
            .lhs_dilation = opts.lhs_dilation,
            .rhs_dilation = opts.rhs_dilation,
            .window_reversal = opts.window_reversal,
            .input_batch_dimension = opts.input_batch_dimension,
            .input_feature_dimension = opts.input_feature_dimension,
            .input_spatial_dimensions = opts.input_spatial_dimensions,
            .kernel_input_feature_dimension = opts.kernel_input_feature_dimension,
            .kernel_output_feature_dimension = opts.kernel_output_feature_dimension,
            .kernel_spatial_dimensions = opts.kernel_spatial_dimensions,
            .output_batch_dimension = opts.output_batch_dimension,
            .output_feature_dimension = opts.output_feature_dimension,
            .output_spatial_dimensions = opts.output_spatial_dimensions,
            .feature_group_count = opts.feature_group_count,
            .batch_group_count = opts.batch_group_count,
        });
    }

    /// Returns a Tensor containing the element-wise addition of the input Tensors.
    pub fn add(self: Tensor, other: Tensor) Tensor {
        return binaryOp("add", dialects.stablehlo.add)(self, other);
    }

    /// Returns a Tensor containing the element-wise subtraction of the input Tensors.
    pub fn sub(self: Tensor, other: Tensor) Tensor {
        return binaryOp("subtract", dialects.stablehlo.subtract)(self, other);
    }

    /// Returns a Tensor containing the element-wise multiplication of the input Tensors.
    pub fn mul(self: Tensor, other: Tensor) Tensor {
        return binaryOp("mul", dialects.stablehlo.multiply)(self, other);
    }

    /// Returns a Tensor containing the element-wise division of the input Tensors.
    pub fn div(self: Tensor, other: Tensor) Tensor {
        return binaryOp("div", dialects.stablehlo.divide)(self, other);
    }

    /// Returns a Tensor containing the element-wise exponentiation of the input Tensors.
    pub fn pow(self: Tensor, other: Tensor) Tensor {
        return binaryOp("pow", dialects.stablehlo.power)(self, other);
    }

    /// Returns a Tensor containing the element-wise maximum operation of the input Tensors.
    pub fn maximum(self: Tensor, other: Tensor) Tensor {
        return binaryOp("maximum", dialects.stablehlo.maximum)(self, other);
    }

    /// Returns a Tensor containing the element-wise minimum operation of the input Tensors.
    pub fn minimum(self: Tensor, other: Tensor) Tensor {
        return binaryOp("minimum", dialects.stablehlo.minimum)(self, other);
    }

    /// Returns a Tensor containing the element-wise remainder of dividend 'self' and divisor 'other'.
    pub fn remainder(self: Tensor, other: Tensor) Tensor {
        return binaryOp("remainder", dialects.stablehlo.remainder)(self, other);
    }

    /// Returns a Tensor containing the element-wise addition of the input Tensor with a constant.
    pub fn addConstant(self: Tensor, b: anytype) Tensor {
        return self.add(.scalar(b, self.dtype()));
    }

    /// Returns a Tensor containing the element-wise division of the input Tensor by a constant.
    pub fn divByConst(self: Tensor, b: anytype) Tensor {
        return self.div(.scalar(b, self.dtype()));
    }

    /// Returns a Tensor containing the element-wise power of the input Tensor by a constant.
    pub fn powByConst(self: Tensor, b: anytype) Tensor {
        return self.pow(.scalar(b, self.dtype()));
    }

    /// Returns a Tensor containing the element-wise multiplication of the input Tensor by a constant.
    pub fn scale(self: Tensor, val: anytype) Tensor {
        return self.mul(.scalar(val, self.dtype()));
    }

    pub const LogicalOp = enum { OR, XOR, AND };

    /// Returns a Tensor containing the element-wise logical operation of the input Tensors.
    pub fn logical(self: Tensor, comptime logical_op: LogicalOp, other: Tensor) Tensor {
        return switch (logical_op) {
            .OR => binaryOp("or", dialects.stablehlo.or_)(self, other),
            .XOR => binaryOp("xor", dialects.stablehlo.xor)(self, other),
            .AND => binaryOp("and", dialects.stablehlo.and_)(self, other),
        };
    }

    /// Returns a Tensor containing the element-wise floor operation of the input Tensor.
    pub fn floor(self: Tensor) Tensor {
        return _result(self._shape, dialects.stablehlo.floor(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock()).result(0));
    }

    /// Returns a Tensor containing the element-wise ceil operation of the input Tensor.
    pub fn ceil(self: Tensor) Tensor {
        return _result(self._shape, dialects.stablehlo.ceil(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock()).result(0));
    }

    /// Returns a Tensor containing the element-wise conversion to another type.
    pub fn convert(self: Tensor, to: DataType) Tensor {
        if (to == self.dtype()) {
            return self;
        }

        const res_type = mlir.rankedTensorType(self.shape().dims(), mlirx.Type.fromDType(mlirCtx(), to));
        const op = dialects.stablehlo.convert(mlirCtx(), self.value(), res_type, .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape.withDtype(to), op.result(0));
    }

    test convert {
        const floats = @import("floats.zig");
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        // f4e2m1
        {
            const x = [_]f32{ 0.0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6 };
            var x_f4: [x.len]floats.Float4E2M1 = undefined;
            for (&x_f4, &x) |*xi_f4, xi| xi_f4.* = .fromF32(xi);

            const x_d: Tensor = .init(.{x.len}, .f32);
            const exe = try zml.module.compile(std.testing.allocator, std.testing.io, Tensor.convert, .{ x_d, .f4e2m1 }, platform);
            defer exe.deinit();

            var x_d_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x_d.shape(), std.mem.sliceAsBytes(&x));
            defer x_d_buffer.deinit();

            var x_f4_xla_d = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Tensor.convert, .{x_d_buffer});
            const x_f4_xla = try x_f4_xla_d.toSliceAlloc(std.testing.allocator, std.testing.io);
            defer x_f4_xla.free(std.testing.allocator);

            errdefer std.log.warn("convert(.f4e2m1) failed !\ninput f32:\n{e}\nzml.floats computed:\n{any}\nxla computed:\n{any}", .{ stdx.fmt.slice(&x), x_f4, x_f4_xla });
            try std.testing.expectEqualDeep(&x_f4, x_f4_xla.items(floats.Float4E2M1));
        }

        // f8e3m4
        {
            const x = [_]f32{ 1.1 / 4.0, 1.1 / 8.0, 1.1 / 16.0, 1.1 / 32.0, 1.1 / 64.0, 1.1 / 128.0 };
            var x_f8e3: [x.len]floats.Float8E3M4 = undefined;
            for (&x_f8e3, &x) |*xi_f8e3, xi| xi_f8e3.* = .fromF32(xi);

            const x_d: Tensor = .init(.{x.len}, .f32);
            const exe = try zml.module.compile(std.testing.allocator, std.testing.io, Tensor.convert, .{ x_d, .f8e3m4 }, platform);
            defer exe.deinit();

            var x_d_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x_d.shape(), std.mem.sliceAsBytes(&x));
            defer x_d_buffer.deinit();

            var x_f8e3_xla_d = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Tensor.convert, .{x_d_buffer});
            const x_f8e3_xla = try x_f8e3_xla_d.toSliceAlloc(std.testing.allocator, std.testing.io);
            defer x_f8e3_xla.free(std.testing.allocator);

            errdefer std.log.warn("convert(.f8e3m4) failed !\ninput f32:\n{e}\nzml.floats computed:\n{any}\nxla computed:\n{any}", .{ stdx.fmt.slice(&x), x_f8e3, x_f8e3_xla });
            try std.testing.expectEqualDeep(&x_f8e3, x_f8e3_xla.items(floats.Float8E3M4));
        }
    }

    /// Returns a Tensor containing the element-wise rounding operation of the input Tensor.
    pub fn round(self: Tensor) Tensor {
        const round_op = dialects.stablehlo.round_nearest_even(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, round_op.result(0));
    }

    /// Returns a Tensor containing the element-wise clamping operation of the input Tensor.
    pub fn clamp(self: Tensor, min_: Tensor, max_: Tensor) Tensor {
        const op = dialects.stablehlo.clamp(mlirCtx(), min_.value(), self.value(), max_.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, op.result(0));
    }

    /// Matrix multiplication, where contracting axes are specified using their tags.
    /// eg dot(.{ .a, .b, .c }, .{ .a, .c, .d }, .c) -> .{ .a, .b, .d }
    /// Axes with the same tag on both sides, and which aren't contracting,
    /// are considered "batching axes".
    pub fn dot(lhs: Tensor, rhs: Tensor, args: anytype) Tensor {
        stdx.debug.assert(lhs.shape().hasTag(args) != null, "Expected lhs to have {any} tag, got {f}", .{ args, lhs.shape() });
        stdx.debug.assert(rhs.shape().hasTag(args) != null, "Expected rhs to have {any} tag, got {f}", .{ args, rhs.shape() });

        const lhs_contracting_dim: i8 = @intCast(lhs.shape().hasTag(args).?);
        const rhs_contracting_dim: i8 = @intCast(rhs.shape().hasTag(args).?);

        var batching_axes: stdx.BoundedArray([2]i8, constants.MAX_RANK) = .{};
        for (0..lhs.rank()) |lhs_tag_index| {
            const lhs_tag = lhs.shape().tag(lhs_tag_index);
            if (lhs_tag == Shape.toTag(args)) continue;
            if (rhs.shape().hasTag(lhs_tag)) |rhs_tag_index| {
                batching_axes.appendAssumeCapacity(.{ @intCast(lhs_tag_index), @intCast(rhs_tag_index) });
            }
        }
        return lhs.dotGeneral(rhs, &.{.{ lhs_contracting_dim, rhs_contracting_dim }}, batching_axes.slice());
    }

    test dot {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        inline for (.{
            .{ .{ .c = 20 }, .{ .c = 20 }, .c, .{} },
            .{
                .{ .a = 20, .b = 21, .c = 22 },
                .{ .a = 20, .d = 23, .c = 22 },
                .c,
                .{ .a = 20, .b = 21, .d = 23 },
            },
            .{
                .{ .a = 20, .b = 21, .c = 22 },
                .{ .c = 22, .d = 23, .e = 24 },
                .c,
                .{ .a = 20, .b = 21, .d = 23, .e = 24 },
            },
            // TODO(Corentin): Re-enable that
            //.{
            //    .{ .a = 20, .b = 21, .c = 22 },
            //    .{ .c = 22, .d = 23, .a = 20 },
            //    .{ .c, .a },
            //    .{ .b = 21, .d = 23 },
            //},
        }) |testcase| {
            const x: Tensor = .init(testcase[0], .f32);
            const y: Tensor = .init(testcase[1], .f32);
            const ctr = Shape.toTag(testcase[2]);
            const z_shape: Shape = .init(testcase[3], .f32);
            const forward = struct {
                fn forward(x_: Tensor, y_: Tensor, tag: Shape.Tag) Tensor {
                    return x_.dot(y_, tag);
                }
            }.forward;

            const exe = try zml.module.compile(std.testing.allocator, std.testing.io, forward, .{ x, y, ctr }, platform);
            defer exe.deinit();

            try zml.testing.expectEqualShapes(Shape.init(z_shape, .f32), exe.output_shapes[0]);
        }
    }

    /// Generalized matrix multiplication of two tensors along the specified axes.
    /// In this version batching dimensions need to be explicitly specified.
    /// The result shape is made of (batching_axes ++ lhs_result_axes ++ rhs_result_axes.
    /// Where "result axes" are non-contracting, non-batching axes of each input tensor.
    pub fn dotGeneral(
        lhs: Tensor,
        rhs: Tensor,
        contracting_axes: []const [2]i8,
        batching_axes: []const [2]i8,
    ) Tensor {
        stdx.debug.assert(lhs.dtype() == rhs.dtype(), "dotGeneral expects tensors to be of the same type, got {} and {}", .{ lhs.dtype(), rhs.dtype() });

        const Axes = stdx.BoundedArray(i64, constants.MAX_RANK);

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

        const op = dialects.stablehlo.dot_general(
            mlirCtx(),
            lhs.value(),
            rhs.value(),
            mlir.rankedTensorType(res_shape.dims(), mlirx.Type.fromDType(mlirCtx(), res_shape.dtype())),
            .{
                .lhs_batching_dimensions = lhs_batching_axes.constSlice(),
                .rhs_batching_dimensions = rhs_batching_axes.constSlice(),
                .lhs_contracting_dimensions = lhs_contracting_axes.constSlice(),
                .rhs_contracting_dimensions = rhs_contracting_axes.constSlice(),
                .dot_precision = .fast,
            },
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());
        return _result(res_shape, op.result(0));
    }

    /// Returns a Tensor containing the sigmoid function applied to each element of the input Tensor.
    pub fn sigmoid(self: Tensor) Tensor {
        const op = dialects.stablehlo.logistic(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, op.result(0));
    }

    pub const logistic = sigmoid;

    /// Returns a Tensor containing the ReLU activation function applied to each element of the input Tensor.
    pub fn relu(self: Tensor) Tensor {
        return self.maximum(Tensor.constant(self.dtype().zero()).broad(.init(self.dims(), self.dtype())));
    }

    /// Returns a Tensor containing the leaky-ReLU activation function applied to each element of the input Tensor.
    ///
    /// LeakyReLU(x) = max(0,x) + negative_slope * min(0,x)
    /// ref: https://paperswithcode.com/method/leaky-relu
    pub fn leakyReLU(self: Tensor, negative_slope: f32) Tensor {
        const below_zero = self.scale(negative_slope).minimum(.scalar(0, self.dtype()));
        return self.relu().add(below_zero);
    }

    test leakyReLU {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        const input: Tensor = .init(.{2}, .f32);
        var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Tensor.leakyReLU, .{ input, 0.1 }, platform);
        defer exe.deinit();

        var input_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, input.shape(), std.mem.sliceAsBytes(&[_]f32{ -0.6884, 1.6795 }));
        defer input_buffer.deinit();

        var res = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Tensor.leakyReLU, .{input_buffer});
        defer res.deinit();

        const expectation: zml.Slice = .init(input.shape(), std.mem.sliceAsBytes(&[2]f32{ -0.0688, 1.6795 }));
        try zml.testing.expectClose(std.testing.io, expectation, res, 1e-4);
    }

    /// Returns a Tensor containing the SwiGLU activation function applied to the input Tensor.
    pub fn swiglu(self: Tensor, beta: f32, w: Tensor, b: Tensor, tag: Shape.Tag) Tensor {
        const sigmoid_tensor = self.mul(Tensor.constant(DataType.Value.init(self.dtype(), beta)).broad(.init(self._shape, self.dtype()))).sigmoid();
        const one_minus_sigmoid_tensor = Tensor.constant(.init(self.dtype(), 1)).broad(.init(self._shape, self.dtype())).sub(sigmoid_tensor);

        return self.mul(sigmoid_tensor).add(one_minus_sigmoid_tensor.mul(w.dot(self, tag).add(b)));
    }

    /// Returns a Tensor containing the Gaussian Error Linear Units (GeLU) activation function applied to each element of the input Tensor.
    ///
    /// We use an approximation of the erf function using tanh:
    ///   gelu(x) ≃ 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    /// see: https://paperswithcode.com/method/gelu
    pub fn gelu(x: Tensor) Tensor {
        const scaled_x_cube = x.mul(x).mul(x).scale(0.044715);
        const beta = std.math.sqrt(2.0 / std.math.pi);
        const tanh_ = x.add(scaled_x_cube).scale(beta).tanh();
        return tanh_.addConstant(1).mul(x).scale(0.5);
    }

    /// Returns a Tensor containing an approximation of the Gaussian Error Linear Units (GeLU) activation function applied to each element of the input Tensor.
    ///
    /// It's an even more crude approximation than gelu.
    pub fn quickGelu(x: Tensor) Tensor {
        return x.scale(1.702).sigmoid().mul(x);
    }

    /// Returns a Tensor containing the Sigmoid Linear Unit (SiLU) activation function applied to each element of the input Tensor.
    ///
    /// silu(x) = x σ(x)
    pub fn silu(x: Tensor) Tensor {
        return x.mul(.sigmoid(x));
    }

    /// Computes softmax along the given axis.
    /// y[i] = exp(x[i]) / ( Σ_k exp(x[k]) + bias )
    pub fn softmax(self: Tensor, axis_: anytype) Tensor {
        const a = self.axis(axis_);
        const max_val = self.max(a);
        const row_mask = max_val.cmp(.GT, .scalar(-std.math.inf(f64), self.dtype()));

        const exp_diff_max = self.sub(max_val).convert(.f32).exp();
        const res = exp_diff_max.div(exp_diff_max.sum(a)).convert(self.dtype());

        // If a row is full -inf return full 0 instead of full nan,
        // this fix attention when mask hides a full row.
        return row_mask.broad(self.shape()).select(res, .scalar(0, self.dtype()));
    }

    /// Computes softmax, but adds a bias to the sum of exponentiel.
    /// y[i] = exp(x[i]) / ( Σ_k exp(x[k]) + bias )
    pub fn softmaxBiased(self: Tensor, axis_: anytype, bias: ?Tensor) Tensor {
        const a = self.axis(axis_);

        if (bias == null) return self.softmax(axis_);
        const b = bias.?.convert(self.dtype()).broad(self.shape().setDim(a, 1));
        const max_val: Tensor = maximum(self.max(a), b);
        const exp_diff_max = self.sub(max_val).exp();
        const bias_diff_max = b.sub(max_val).exp();
        const res = exp_diff_max.div(exp_diff_max.sum(a).add(bias_diff_max));

        // The bias means that denominator won't be 0, we don't need to handle that case.
        return res;
    }

    /// Returns a Tensor containing the log of the sum of exponential over the given axis.
    pub fn logSumExp(self: Tensor, axis_: anytype) Tensor {
        const a = self.axis(axis_);
        // stabilization: shift `self` by it's max value before passing to exponent.
        const max_ = self.max(a);
        const log_sum_exp = log(sum(exp(self.sub(max_.broad(self._shape))), a));
        // restore the shift again
        return max_.add(log_sum_exp);
    }

    /// Returns a Tensor containing the sum of elements over the given axis.
    /// Output shape is the input shape with the axis_ dim set to 1.
    pub fn sum(self: Tensor, axis_: anytype) Tensor {
        const a = self.axis(axis_);
        return ops.reduce(.{self}, .{Tensor.constant(self.dtype().zero())}, &.{a}, struct {
            pub fn acc(args: ops.ReduceArgs) struct { Tensor } {
                return .{args.right.add(args.left.convert(args.right.dtype()))};
            }
        }.acc, .{})[0];
    }

    /// Returns a Tensor containing the mean of elements over the given axis.
    /// Output shape is the input shape with the axis_ dim set to 1.
    pub fn mean(self: Tensor, axis_: anytype) Tensor {
        return self.sum(axis_).divByConst(self.dim(axis_));
    }

    /// Returns a Tensor containing the cumulative sum of elements over the given axis.
    /// Output shape is the same as input shape.
    /// [0, 1, 0, 1, 0, 0, 1, 1].cumulativeSum(0) -> [0, 1, 1, 2, 2, 2, 3, 4]
    /// The last value contains the sum of all element in the array.
    pub fn cumulativeSum(self: Tensor, axis_: anytype) Tensor {
        const rk = self.rank();
        const a = self.axis(axis_);

        const ones = [_]i64{1} ** constants.MAX_RANK;
        var window_dimensions = ones;
        window_dimensions[a] = self.dim(a);
        var padding = [_][2]i64{.{ 0, 0 }} ** constants.MAX_RANK;
        padding[a] = .{ self.dim(a) - 1, 0 };

        const result = ops.reduceWindow(
            .{self},
            .{Tensor.scalar(0, self.dtype())},
            .{
                .base_dilations = ones[0..rk],
                .window_dilations = ones[0..rk],
                .window_strides = ones[0..rk],
                .window_dimensions = window_dimensions[0..rk],
                .padding = padding[0..rk],
            },
            struct {
                fn add(values: ops.ReduceArgs) struct { Tensor } {
                    return .{values.left.add(values.right)};
                }
            }.add,
            .{},
        );

        return result[0];
    }

    test cumulativeSum {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        const Local = struct {
            pub fn _cumsum(input: Tensor) Tensor {
                const x = input.withPartialTags(.{.n});
                const y = x.cumulativeSum(.n);
                // Check that tags are propagated
                std.debug.assert(y.shape().eqlWithTags(x.shape()));
                return y;
            }
        };

        const x: Tensor = .init(.{ 2, 5 }, .f32);

        var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local._cumsum, .{x}, platform);
        defer exe.deinit();

        var x_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&[2][5]f32{ .{ 0, 1, 1, 0, 1 }, .{ 3, 1, 0, 2, 1 } }));
        defer x_buffer.deinit();

        var res = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Local._cumsum, .{x_buffer});
        defer res.deinit();

        try std.testing.expectEqual([2][5]f32{ .{ 0, 1, 2, 2, 3 }, .{ 3, 4, 4, 6, 7 } }, try res.getValue([2][5]f32, std.testing.io));
    }

    /// Returns a transposed Tensor computed using the given axes.
    pub fn transpose(self: Tensor, axes_: anytype) Tensor {
        const axes__ = self.axes(axes_).constSlice();
        const no_op = constants.AXES_IOTA;
        const default_perm = b: {
            var buf = constants.AXES_IOTA;
            std.mem.reverse(i64, &buf);
            break :b buf;
        };

        const permutation: []const i64 = if (axes__.len == 0)
            default_perm[constants.MAX_RANK - self.rank() ..]
        else
            toI64(axes__).constSlice();

        stdx.debug.assert(permutation.len == self.rank(), "transpose expects input tensor rank and 'axes_' length to be equal, got {f} and {any}", .{ self, permutation[0..@min(permutation.len, constants.MAX_RANK + 2)] });

        if (std.mem.eql(i64, permutation, no_op[0..self.rank()])) {
            return self;
        }

        const res_shape = self._shape.transpose(permutation);
        if (transposeIsJustAReshape(self.shape(), permutation)) {
            return self.reshape(res_shape);
        }

        const op = dialects.stablehlo.transpose(
            mlirCtx(),
            self.value(),
            mlir.rankedTensorType(res_shape.dims(), mlirx.Type.fromDType(mlirCtx(), res_shape.dtype())),
            .{ .permutation = toI64(permutation).constSlice() },
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());
        return _result(res_shape, op.result(0));
    }

    pub fn swapAxes(self: Tensor, a: anytype, b: anytype) Tensor {
        if (self.axis(a) == self.axis(b)) return self;
        var perm: Shape.AxesArray = .{};
        for (0..self.rank()) |i| {
            perm.appendAssumeCapacity(@intCast(i));
        }
        perm.set(self.axis(a), self.axis(b));
        perm.set(self.axis(b), self.axis(a));
        return self.transpose(perm.constSlice());
    }

    /// Returns a Tensor with the given axis unflattened.
    ///
    /// unflatten((d0, d1, axis_m, d3), 2, n) -> (d0, d1, n, d2_m, d3)
    pub fn unflatten(self: Tensor, axis_: i8, n: i64) Tensor {
        // TODO: move to torch.zig, this equivalent to `spitAxis`
        stdx.debug.assert(self.rank() < constants.MAX_RANK, "unflatten expects input tensor rank to be less than {}, got {}", .{ constants.MAX_RANK, self.rank() });

        const a = if (axis_ >= 0) self.axis(axis_) else self.axis(axis_) + 1;
        const new_dim = std.math.divExact(i64, self.dim(a), n) catch std.debug.panic("unflatten expects chosen dimension to be divisible by 'n' but {} is not divisible by {}", .{ self.dim(a), n });
        const new_shape = self._shape.set(a, n).insert(a + 1, .{ ._ = new_dim });

        const reshaped_val = dialects.stablehlo.reshape(
            mlirCtx(),
            self.value(),
            mlir.rankedTensorType(new_shape.dims(), mlirx.Type.fromDType(mlirCtx(), new_shape.dtype())),
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());
        return _result(new_shape, reshaped_val.result(0));
    }

    /// Splits the given axis in several axes.
    /// eg: `Tensor.init(.{ .a = 10, .b = 3 }).split(.a, .{.a1 = 5, .a2 = 2});`
    /// The number of elements in the split shape must match the number of element
    /// in the target axis.
    pub fn splitAxis(self: Tensor, ax: anytype, split_shape: anytype) Tensor {
        const new_shape = self._shape.splitAxis(ax, split_shape);

        const reshaped_val = dialects.stablehlo.reshape(
            mlirCtx(),
            self.value(),
            mlir.rankedTensorType(new_shape.dims(), mlirx.Type.fromDType(mlirCtx(), new_shape.dtype())),
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());
        return _result(new_shape, reshaped_val.result(0));
    }

    /// Merges two or more contiguous axes into one axis.
    pub fn merge(self: Tensor, merges_: anytype) Tensor {
        return self.reshape(self._shape.mergeAxes(merges_));
    }

    /// Merges two or more non-contiguous axes into one axis.
    /// Will make a transpose if needed.
    /// .{ .a, .b, .c }.mergeTranspose(.{ .a, .c }, .ac) -> .{ .b, .ac }
    pub fn mergeTranspose(self: Tensor, axes_: anytype, merged: @EnumLiteral()) Tensor {
        const cont = self.contiguous(axes_);
        return cont.reshape(cont._shape.mergeAxis(merged, axes_));
    }

    /// Transposes the input Tensor, such has the given axes end up in contiguous position.
    /// .{ .a, .b, .c, .d }.contiguous(.{ .c, .a }) -> .{ .b, .d, .c, .a }
    pub fn contiguous(self: Tensor, axes_: anytype) Tensor {
        const perm = self._shape.contiguousPerm(axes_);
        return self.transpose(perm.constSlice());
    }

    pub fn flatten(self: Tensor) Tensor {
        return self.reshape(.{self.count()});
    }

    pub const Slice = struct {
        start: i64 = 0,
        end: i64 = to_the_end,
        step: i32 = 1,
        singleton: bool = false,

        const full = .{ .start = 0, .end = to_the_end, .step = 1 };

        pub fn single(offset: i64) Slice {
            return .{ .start = offset, .end = offset + 1, .singleton = true };
        }

        pub fn absolute(s: Slice, d: i64) Slice {
            const start = if (s.start < 0) d + s.start else s.start;
            const end = if (s.end == to_the_end) d else if (s.end < 0) d + s.end else s.end;
            const res: Slice = .{ .start = start, .end = end, .step = s.step, .singleton = s.singleton };
            stdx.debug.assert(start < end, "Slice {f} is invalid for axis of dimension {d} (resolved to {f}", .{ s, d, res });
            return res;
        }

        const to_the_end = std.math.maxInt(i64);

        pub fn format(self: Slice, writer: *std.Io.Writer) !void {
            if (self.singleton) {
                try writer.print("[{d}]", .{self.start});
            } else if (self.end == to_the_end and self.step == 1) {
                try writer.print("[{d}..]", .{self.start});
            } else if (self.step == 1) {
                try writer.print("[{d}..{d}]", .{ self.start, self.end });
            } else {
                try writer.print("[{d}..{d}:{d}]", .{ self.start, self.end, self.step });
            }
        }
    };

    /// Slices the input Tensor over the given axis using the given parameters.
    pub fn slice1d(self: Tensor, axis_: anytype, s: Slice) Tensor {
        var slices = [_]Slice{.{}} ** constants.MAX_RANK;
        slices[self.axis(axis_)] = s;
        return self.slice(slices[0..self.rank()]);
    }

    /// Slices the input Tensor using the given parameters.
    pub fn slice(self: Tensor, slices: []const Slice) Tensor {
        var start_indices: [constants.MAX_RANK]i64 = undefined;
        var strides: [constants.MAX_RANK]i64 = undefined;
        var limit_indices: [constants.MAX_RANK]i64 = undefined;
        var res_shape: Shape = self._shape;

        for (slices, 0..) |s, a| {
            stdx.debug.assert(s.step > 0, "slice expects 'step' to be positive, got {} at index {}", .{ s.step, a });

            const args: Slice = s.absolute(self.dim(a));
            start_indices[a] = args.start;
            limit_indices[a] = args.end;
            strides[a] = args.step;
            res_shape = res_shape.setDim(a, std.math.divCeil(i64, args.end - args.start, args.step) catch unreachable);
        }

        const result_type = mlir.rankedTensorType(res_shape.dims(), mlirx.Type.fromDType(mlirCtx(), res_shape.dtype()));
        const slice_op = dialects.stablehlo.slice(
            mlirCtx(),
            self.value(),
            start_indices[0..self.rank()],
            limit_indices[0..self.rank()],
            strides[0..self.rank()],
            result_type,
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());

        var res = _result(res_shape, slice_op.result(0));
        var to_remove: Shape.AxesArray = .{};
        for (slices, 0..) |s, a| {
            if (s.singleton) to_remove.appendAssumeCapacity(@intCast(a));
        }
        return res.reshape(res_shape.removeMany(to_remove.constSlice()));
    }

    test slice {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        const x: Tensor = .init(.{ 2, 5 }, .f32);

        var x_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&[_]f32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }));
        defer x_buffer.deinit();

        // Wrap slice1d to hide the anytype in the signature.
        const Local = struct {
            pub fn _slice1dAxis(input: Tensor, ax: i8, slice_: Tensor.Slice) Tensor {
                return input.slice1d(ax, slice_);
            }
        };

        inline for (.{
            .{ 0, Tensor.Slice{ .end = 1 }, [5]f32{ 0, 1, 2, 3, 4 } },
            .{ 1, Tensor.Slice{ .start = 1, .step = 2 }, [4]f32{ 1, 3, 6, 8 } },
            .{ -1, Tensor.Slice{ .start = -2 }, [4]f32{ 3, 4, 8, 9 } },
        }) |testcase| {
            const ax, const slice_, const expectation = testcase;
            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local._slice1dAxis, .{ x, ax, slice_ }, platform);
            defer exe.deinit();

            var res = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Local._slice1dAxis, .{x_buffer});
            defer res.deinit();

            try std.testing.expectEqual(expectation, try res.getValue(@TypeOf(expectation), std.testing.io));
        }
    }

    inline fn wrapIndex(self: Tensor, axis_: usize, idx: i64) i64 {
        return if (idx < 0) self.dim(axis_) + idx else idx;
    }

    pub fn choose1d(self: Tensor, axis_: anytype, i: i64) Tensor {
        return self.slice1d(axis_, .single(i));
    }

    pub fn choose(self: Tensor, offsets: anytype) Tensor {
        const off, const tags = Shape.parseDimensions(offsets);
        var slices = [_]Slice{.{}} ** constants.MAX_RANK;
        for (off.constSlice(), tags.constSlice()) |o, t| {
            const ax = self.axis(t);
            slices[ax] = .single(o);
        }
        return self.slice(slices[0..self.rank()]);
    }

    /// Concatenates the input Tensors along the given axis.
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
        const op = dialects.stablehlo.concatenate(mlirCtx(), buffer[0..tensors.len], a, .unknown(mlirCtx())).appendTo(currentBlock());
        // log.debug("concatenate({}, {}, {d}) -> {d}", .{ tensors[0], tensors[1], a, res_shape });
        return _result(res_shape, op.result(0));
    }

    /// Concatenates the input Tensors along a new axis. The Tensors must have the same shape.
    /// For x, y, z of shape .{ .a = 10, .b = 11, .c = 12 }:
    /// - Tensor.stack(&.{x, y, z}, .b, .layers) -> .{ .a, .layers, .b, .c }
    /// - Tensor.stack(&.{x, y, z}, 1, .layers) -> .{ .a, .layers, .b, .c }
    /// - Tensor.stack(&.{x, y, z}, .last, .layers) -> .{ .a, .b, .c, .layers }
    pub fn stack(tensors: []const Tensor, axis_: anytype, tag: anytype) Tensor {
        // Note: we could ask the compilation context for some memory instead of stack allocating
        stdx.debug.assert(tensors.len <= 32, "stack only supports up to 32 tensors, got {}", .{tensors.len});

        const shape0 = tensors[0]._shape;
        const res_shape = shape0.insertTag(axis_, 1, tag);

        for (tensors[1..]) |tensor| {
            stdx.debug.assert(shape0.eqlWithTags(tensor._shape), "stack expects tensor shapes to match, got {f} and {f}", .{ shape0, tensor._shape });
        }

        var reshaped: [32]Tensor = undefined;
        for (tensors, 0..) |tensor, i| {
            reshaped[i] = tensor.reshape(res_shape);
        }

        // Be careful here: we need to resolve ax before calling concatenate,
        // because we added an axis, so all
        const ax = if (@TypeOf(axis_) == @EnumLiteral() and axis_ == .last)
            shape0.rank()
        else
            shape0.axis(axis_);

        return Tensor.concatenate(reshaped[0..tensors.len], ax);
    }

    /// Repeats a Tensor several times along the given axis.
    ///
    /// * repeat1d(x, axis, 4) = concat(&.{x, x, x, x}, axis);
    /// * repeat1d([0, 1, 2, 3], 0, 2) = [0, 1, 2, 3, 0, 1, 2, 3]
    pub fn repeat1d(self: Tensor, axis_: anytype, n_rep: u63) Tensor {
        if (n_rep == 1) {
            return self;
        }

        const a = self.axis(axis_);
        const res_shape = self._shape.setDim(a, self.dim(a) * n_rep);

        const broadshape = self._shape.insert(a, .{n_rep});
        const repeat_dims = Shape.range(self.rank() + 1, self.dtype()).remove(a);

        return self.broadcast(broadshape, repeat_dims.dims()).reshape(res_shape);
    }

    /// Repeats a Tensor several times along the given axes.
    pub fn repeat(self: Tensor, n_reps: []const u63) Tensor {
        // TODO: this should support the tagged syntax: x.repeat(.{ .a = 3, .b = 2});
        stdx.debug.assert(n_reps.len == self.rank(), "repeat expects tensor rank and 'n_reps' length to be equal, got {} and {}", .{ self.rank(), n_reps.len });

        var res = self;
        for (n_reps, 0..) |n_rep, a| {
            if (n_rep == 1) continue;

            res = res.repeat1d(a, n_rep);
        }
        return res;
    }

    test repeat1d {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        const Local = struct {
            fn repeat1d(x: Tensor, axis_: u3, n_reps: u32) Tensor {
                return x.repeat1d(axis_, n_reps);
            }
        };

        inline for (.{
            .{ [3]u8{ 1, 2, 3 }, Shape.init(.{3}, .u8), [6]u8{ 1, 2, 3, 1, 2, 3 }, 0, 2 },
            .{ [2][3]u8{ .{ 1, 2, 3 }, .{ 4, 5, 6 } }, Shape.init(.{ 2, 3 }, .u8), [2][6]u8{ .{ 1, 2, 3, 1, 2, 3 }, .{ 4, 5, 6, 4, 5, 6 } }, 1, 2 },
        }) |testcase| {
            const input_data, const shape_, const expectation, const ax, const reps = testcase;
            const input: Tensor = .fromShape(shape_);

            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local.repeat1d, .{ input, ax, reps }, platform);
            defer exe.deinit();

            var input_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, input.shape(), std.mem.sliceAsBytes(&input_data));
            defer input_buffer.deinit();

            const output = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Local.repeat1d, .{input_buffer});
            try std.testing.expectEqual(expectation, try output.getValue(@TypeOf(expectation), std.testing.io));
        }
    }

    /// Repeats in line each value along the given axis.
    ///
    /// * stutter1d([0, 1, 2, 3], -1, 2) = [0, 0, 1, 1, 2, 2, 3, 3]
    /// This is equivalent to repeat(ax+1) unless ax is the last axis.
    pub fn stutter1d(self: Tensor, axis_: i8, n_rep: u63) Tensor {
        const a = self.axis(axis_);
        const broadshape = self._shape.insert(a + 1, .{n_rep});
        const res_shape = self._shape.setDim(a, self.dim(a) * n_rep);

        const stutter_dims = Shape.range(self.rank() + 1, self.dtype()).remove(a + 1);
        return self.broadcast(broadshape, stutter_dims.dims()).reshape(res_shape);
    }

    /// Repeats in line each value along the given axes.
    pub fn stutter(self: Tensor, n_reps: []const u63) Tensor {
        // TODO: this should support the tagged syntax: x.repeat(.{ .a = 3, .b = 2});
        stdx.debug.assert(n_reps.len == self.rank(), "stutter expects tensor rank and 'n_reps' length to be equal, got {} and {}", .{ self.rank(), n_reps.len });

        var res = self;
        for (n_reps, 0..) |n_rep, a| {
            if (n_rep == 1) continue;
            res = res.stutter1d(@intCast(a), n_rep);
        }
        return res;
    }

    test stutter1d {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        const Local = struct {
            fn stutter1d(x: Tensor, axis_: u3, n_reps: u32) Tensor {
                return x.stutter1d(axis_, n_reps);
            }
        };

        inline for (.{
            .{ [3]u8{ 1, 2, 3 }, Shape.init(.{3}, .u8), [6]u8{ 1, 1, 2, 2, 3, 3 }, 0, 2 },
            .{ [2][3]u8{ .{ 1, 2, 3 }, .{ 4, 5, 6 } }, Shape.init(.{ 2, 3 }, .u8), [2][6]u8{ .{ 1, 1, 2, 2, 3, 3 }, .{ 4, 4, 5, 5, 6, 6 } }, 1, 2 },
            .{ [2][3]u8{ .{ 1, 2, 3 }, .{ 4, 5, 6 } }, Shape.init(.{ 2, 3 }, .u8), [2][6]u8{ .{ 1, 2, 3, 1, 2, 3 }, .{ 4, 5, 6, 4, 5, 6 } }, 0, 2 },
        }) |testcase| {
            const input_data, const shape_, const expectation, const ax, const reps = testcase;
            const input: Tensor = .fromShape(shape_);

            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local.stutter1d, .{ input, ax, reps }, platform);
            defer exe.deinit();

            var input_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, input.shape(), std.mem.sliceAsBytes(&input_data));
            defer input_buffer.deinit();

            const output = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Local.stutter1d, .{input_buffer});
            try std.testing.expectEqual(expectation, try output.getValue(@TypeOf(expectation), std.testing.io));
        }
    }

    /// Returns a Tensor containing the element-wise negation of the input Tensor.
    pub fn negate(self: Tensor) Tensor {
        const negate_op = dialects.stablehlo.negate(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, negate_op.result(0));
    }

    /// Returns a Tensor containing the element-wise cosine of the input Tensor.
    pub fn cos(self: Tensor) Tensor {
        const cosine_op = dialects.stablehlo.cosine(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, cosine_op.result(0));
    }

    /// Returns a Tensor containing the element-wise sine of the input Tensor.
    pub fn sin(self: Tensor) Tensor {
        const sine_op = dialects.stablehlo.sine(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, sine_op.result(0));
    }

    /// Returns a Tensor containing the element-wise exponential operation of the input Tensor.
    pub fn exp(self: Tensor) Tensor {
        const op = dialects.stablehlo.exponential(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, op.result(0));
    }

    /// Returns a Tensor containing the element-wise logarithm operation of the input Tensor.
    pub fn log(self: Tensor) Tensor {
        const op = dialects.stablehlo.log(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, op.result(0));
    }

    /// Returns a Tensor containing the element-wise square-root of the input Tensor.
    pub fn sqrt(self: Tensor) Tensor {
        const sqrt_op = dialects.stablehlo.sqrt(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, sqrt_op.result(0));
    }

    /// Returns a Tensor containing the element-wise reverse square-root of the input Tensor.
    pub fn rsqrt(self: Tensor) Tensor {
        const rsqrt_op = dialects.stablehlo.rsqrt(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, rsqrt_op.result(0));
    }

    /// Returns a Tensor containing the element-wise hyperbolic tangent of the input Tensor.
    pub fn tanh(self: Tensor) Tensor {
        const tanh_op = dialects.stablehlo.tanh(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, tanh_op.result(0));
    }

    /// Returns a Tensor containing the element-wise exponential minus one operation of the input Tensor.
    pub fn exponentialMinusOne(self: Tensor) Tensor {
        const expm1_op = dialects.stablehlo.exponential_minus_one(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, expm1_op.result(0));
    }

    pub const ArangeArgs = struct {
        start: i64 = 0,
        end: i64,
        step: i64 = 1,
    };
    ///
    /// Returns a Tensor containing evenly spaced values within a given interval.
    pub fn arange(args: ArangeArgs, dt: DataType) Tensor {
        stdx.debug.assert(args.start <= args.end, "arange expects 'args.start' to be less than 'args.end', got {} and {}", .{ args.start, args.end });
        stdx.debug.assert(args.step > 0, "arange expects 'args.step' to be positive, got {}", .{args.step});

        const n_steps = std.math.divCeil(i64, args.end - args.start, args.step) catch unreachable;
        const sh = Shape.init(.{n_steps}, dt);
        var op = dialects.stablehlo.iota(
            mlirCtx(),
            0,
            mlir.rankedTensorType(sh.dims(), mlirx.Type.fromDType(mlirCtx(), sh.dtype())),
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());
        var res = _result(sh, op.result(0));

        if (args.step != 1) {
            res = res.scale(args.step);
        }

        if (args.start != 0) {
            res = res.addConstant(args.start);
        }

        return res;
    }

    /// Returns a Tensor containing values in increasing order starting from 0 along the given axis.
    ///
    /// The output dtype will be `.i32`, unless the given axis has a too big dimension, in that case we use `.i64`.
    /// In most program this shouldn't matter, because typically this will be used in a comparison,
    /// or explicitly converted by the user to do floating point arithmetic.
    pub fn iota(sh: Shape, axis_: anytype) Tensor {
        const a = sh.axis(axis_);
        const dt: DataType = if (sh.dim(a) <= std.math.maxInt(i32)) .i32 else .i64;
        const res_shape = sh.withDtype(dt);

        var op = dialects.stablehlo.iota(
            mlirCtx(),
            a,
            mlir.rankedTensorType(res_shape.dims(), mlirx.Type.fromDType(mlirCtx(), res_shape.dtype())),
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());
        return _result(res_shape, op.result(0));
    }

    pub const LinspaceArgs = struct {
        start: f64,
        end: f64,
        steps: i64,
    };

    /// Returns a Tensor containing 'args.steps' values evenly spaced from 'args.start' to 'args.end', inclusive.
    pub fn linspace(args: LinspaceArgs, dt: DataType) Tensor {
        stdx.debug.assert(args.start < args.end, "linspace expects 'args.start' to be less than 'args.end', got {} and {}", .{ args.start, args.end });
        stdx.debug.assert(args.steps > 0, "linspace expects 'args.steps' to be positive, got {}", .{args.steps});
        stdx.debug.assert(dt.isFloat(), "linspace expects type to be a float, got {} (hint: use arange instead)", .{dt});

        const sh = Shape.init(.{args.steps}, dt);
        var iota_op = dialects.stablehlo.iota(
            mlirCtx(),
            0,
            mlir.rankedTensorType(sh.dims(), mlirx.Type.fromDType(mlirCtx(), sh.dtype())),
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());
        var res = _result(sh, iota_op.result(0));

        if (args.steps != 1) {
            res = res.scale(args.steps);
        }

        if (args.start != 0) {
            res = res.addConstant(args.start);
        }

        return res;
    }

    /// Returns a 0-rank Tensor with the given value.
    pub fn scalar(val: anytype, dt: DataType) Tensor {
        const data = DataType.Value.init(dt, val);
        switch (dt.class()) {
            .float => stdx.debug.assert(!std.math.isNan(val), "scalar(NaN) is probably due to compiling a model with an uninitialized field", .{}),
            else => {},
        }
        return Tensor.constant(data);
    }

    test scalar {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        const Local = struct {
            pub fn _fwd() [6]Tensor {
                var res: [6]Tensor = undefined;
                const dtypes = .{ .bool, .u8, .i32, .f32, .bf16, .u64 };
                inline for (0..6) |i| res[i] = scalar(0, dtypes[i]);
                return res;
            }
        };

        var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local._fwd, .{}, platform);
        defer exe.deinit();
    }

    /// Returns a constant Tensor with the given value.
    pub fn constant(val: DataType.Value) Tensor {
        const op = dialects.stablehlo.constant(
            mlirCtx(),
            &.{},
            mlirx.Type.fromDType(mlirCtx(), val.dtype()),
            val.asBytes(),
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());
        return _result(.init(&.{}, val.dtype()), op.result(0));
    }

    pub fn zeroes(sh: Shape) Tensor {
        return Tensor.constant(sh.dtype().zero()).broad(sh);
    }

    /// Embeds a buffer with concrete values into an Mlir program.
    pub fn constantTensor(sh: Shape, bytes_: []const u8) Tensor {
        const elem_type = mlirx.Type.fromDType(mlirCtx(), sh.dtype());
        //const elem_type = mlirx.denseElementAttrType(val.dtype()) orelse std.debug.panic("constantTensor expects a dtype that can be serialized to MLIR, like f32 or i32, got {f}", .{val.shape()});
        const constant_op = dialects.stablehlo.constant(mlirCtx(), sh.dims(), elem_type, bytes_, .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(sh, constant_op.result(0));
    }

    /// Returns a Tensor containing the result of the outer product between the input Tensors.
    pub fn outer(self: Tensor, other: Tensor) Tensor {
        if (self.rank() + other.rank() == 1) {
            return self.mul(other);
        }

        const res_shape = self.shape().outer(other.shape());
        return self.broad(res_shape).mul(other.broad(res_shape));
    }

    /// Given a tensor and a shape of the same rank,
    /// will "broadcast" the given axes, so that `self` has the given shape.
    /// This happens by virtually repeating the data several time along each give axes.
    /// Note: most of the time the optimizer will make it so that the broadcast doesn't trigger a copy.
    /// Note: the tags of the return tensor will be from the `output_shape`.
    /// This means if you use and un-tagged broadcast on a tagged tensor,
    /// you will lose the tags.
    /// To avoid use favorise `.broad(shape)` when working with tagged tensors.
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
        const result_type = mlir.rankedTensorType(
            res_shape.dims(),
            mlirx.Type.fromDType(mlirCtx(), res_shape.dtype()),
        );
        const broadcast_op = dialects.stablehlo.broadcast_in_dim(mlirCtx(), self.value(), axes_, result_type, .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(res_shape, broadcast_op.result(0));
    }

    /// Broadcasts a Tensor to the given shape, adding axes at the beginning.
    pub fn broadcastLeft(self: Tensor, output_shape: Shape) Tensor {
        stdx.debug.assert(self.rank() <= output_shape.rank(), "broadcastLeft expects tensor rank to be less than output tensor rank, got {d} and {d}", .{ self.rank(), output_shape.rank() });

        const a = output_shape.rank() - self.rank();
        if (self.rank() == output_shape.rank() and std.mem.eql(i64, self.dims(), output_shape.dims())) {
            return self;
        }

        return self.broadcast(output_shape, Shape.range(output_shape.rank(), output_shape.dtype()).dims()[a..]);
    }

    /// Broadcasts a Tensor to the given shape, adding axes at the end.
    pub fn broadcastRight(self: Tensor, output_shape: Shape) Tensor {
        stdx.debug.assert(self.rank() <= output_shape.rank(), "broadcastRight expects tensor rank to be less than output tensor rank, got {d} and {d}", .{ self.rank(), output_shape.rank() });

        if (self.rank() == output_shape.rank() and self._shape.eql(output_shape)) {
            return self;
        }

        return self.broadcast(output_shape, Shape.range(self.rank(), output_shape.dtype()).dims());
    }

    /// Broadcasts a Tensor to the given shape, extending dimensions if needed.
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
            return self.broadcast(other, constants.AXES_IOTA[0..self.rank()]);
        }

        // check that each axis of self maps to an axis of other
        var axes_: stdx.BoundedArray(i64, constants.MAX_RANK) = .{};
        for (self._shape.tags()) |t| {
            axes_.appendAssumeCapacity(@intCast(other.axis(t)));
        }
        return self.broadcast(other, axes_.constSlice());
    }

    /// Reshapes the input Tensor with the given shape.
    pub fn reshape(self: Tensor, output_shape_: anytype) Tensor {
        const output_shape = self._shape.reshape(output_shape_);
        const tensor_type = mlir.rankedTensorType(output_shape.dims(), mlirx.Type.fromDType(mlirCtx(), output_shape.dtype()));
        const reshape_value = dialects.stablehlo.reshape(mlirCtx(), self.value(), tensor_type, .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(output_shape, reshape_value.result(0));
    }

    /// Converts the given 1 element Tensor into a 0-rank Tensor.
    pub fn asScalar(self: Tensor) Tensor {
        stdx.debug.assert(self.count() == 1, "Tensor.asScalar expects an input with exactly 1-element got {f}", .{self});
        return if (self.rank() == 0) self else self.reshape(.{});
    }

    pub const Pad = struct {
        low: i64 = 0,
        high: i64 = 0,
        interior: i64 = 0,
    };

    /// Pads the input Tensor with the given values.
    /// Usage: x.pad(0, .{ .a = .{ .low = 1, .high = 1 }});
    pub fn pad(self: Tensor, padding_value: anytype, paddings: anytype) Tensor {
        const _paddings = self.shape().parseAxesOptions(Pad, paddings, .{});

        const ZEROS = [_]i64{0} ** constants.MAX_RANK;
        var low = ZEROS;
        var high = ZEROS;
        var interior = ZEROS;

        var res_shape = self._shape;
        for (_paddings.constSlice(), 0..) |padding, i| {
            low[i] = padding.low;
            high[i] = padding.high;
            interior[i] = padding.interior;

            var d: i64 = self.dim(i);
            d += low[i] + (@max(d - 1, 0) * interior[i]) + high[i];
            res_shape._dims.set(i, d);
        }

        const rk = self.rank();
        const pad_op = dialects.stablehlo.pad(
            mlirCtx(),
            self.value(),
            Tensor.scalar(padding_value, self.dtype()).value(),
            .{ .low = low[0..rk], .high = high[0..rk], .interior = interior[0..rk] },
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());

        return _result(res_shape, pad_op.result(0));
    }

    /// Inserts 1-dim axes at the given position, with the given tags.
    /// `.{.a = 5, .b = 4}.insert(.b, .{ .c, .d }) -> .{ .a = 5, .c = 1, .d = 1, .b = 4 }`
    pub fn insertAxes(self: Tensor, axis_: anytype, tags: anytype) Tensor {
        const tags_ = Shape.parseTags(tags);
        const ax = if (@TypeOf(axis_) == @EnumLiteral() and axis_ == .last)
            self.rank()
        else
            self.axis(axis_);

        var res_shape = self._shape;
        const ones = [_]i64{1} ** constants.MAX_RANK;
        res_shape._dims.insertSlice(ax, ones[0..tags_.len]) catch unreachable;
        res_shape._tags.insertSlice(ax, tags_.constSlice()) catch unreachable;

        return self.reshape(res_shape);
    }

    /// Appends a 1-dim axis, with the given tag.
    pub fn appendAxes(self: Tensor, t: anytype) Tensor {
        // stdx.debug.assert(self.rank() < Tensor.MAX_RANK - t.len, "appendAxis expects tensor rank to be small enough in order to extend it, got {} and {} (max is {})", .{ self.rank(), t.len, Tensor.MAX_RANK });

        return self.insertAxes(.last, t);
    }

    /// Drops a 1-dim axis at the given index
    pub fn squeeze(self: Tensor, axis_: anytype) Tensor {
        const a = self.axis(axis_);
        stdx.debug.assert(self.dim(a) == 1, "squeeze expects axis to be squeezed to have a dimension of 1, got {}", .{self.dim(a)});

        const new_shape = self._shape.remove(a);
        // log.debug("squeeze({}, {d}={d}) -> ({})", .{ self, axis, a, new_shape });

        return _result(new_shape, self.reshape(new_shape).value());
    }

    /// Returns a Tensor with the given axes reversed.
    pub fn reverse(self: Tensor, axes_: anytype) Tensor {
        const actual_axes = self._shape.axes(axes_);

        const reverse_op = dialects.stablehlo.reverse(mlirCtx(), self.value(), toI64(actual_axes.constSlice()).constSlice(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, reverse_op.result(0));
    }

    pub const GatherOpts = ops.GatherOpts;

    /// `gather` extracts slices from the given tensor at the specified offsets.
    /// example: `values.gather(.{ .a = idx }, .{})`
    ///
    /// * indices is a named list of integer tensors: eg `.{ .a = idx }`.
    /// Each names specify a gathering axis, it must refer to an axis of self,
    /// and the corresponding idx Tensor must contains valid indices into axis .a.
    /// All indices must have the same shape or broadcast to the same shape.
    ///
    /// * result is a tensor whose shape is similar to the input shape
    /// where the gathered axes have been replaced by axes from 'indices'.
    ///
    /// Some example input for the base case where we work on one axis:
    /// - gather(f:[a], .{ .a = idx:[n]})[n] == f[idx[n]]
    /// - gather(f:[a, b], .a, idx:[n])[n, b] == f[idx[n], b]
    /// - gather(f:[a,b,c], .{.b = idx:[n,m]})[a, n, m, c] == f[a, idx[n, m], c]
    ///
    /// If an axis in common between `self` and `indices`,
    /// it is treated as a "batching" axis, meaning that semantically
    /// the operator is doing a gather one time per dimension of this axis:
    /// - gather(f: [a,b,c], .{.b=idx: [a,n]})[a, n] == f[a, idx[a, n]]
    ///
    /// It's possible to pass several indices:
    /// - gather(f: [a,b,c], .{.b=idx_b[n], .c=idx_c[n]})[a, n] == f[a, idx_b[n], idx_c[n]]
    /// - gather(f: [a,b,c,d], .{.b=idx_b[a,n], .c=idx_c[a, n]})[a, n, d] == f[a, idx_b[a, n], idx_c[a, n], d]
    ///
    /// If `self` isn't tagged, you can use `gather_` to specify gathered axis by their position but batching won't be available.
    ///
    /// For performance it's better to have batching and gathering axes of `self` be the first one,
    /// so that gather can
    pub fn gather(self: Tensor, _indices: anytype, opts: GatherOpts) Tensor {
        const idx_per_axis, const idx_tags = Shape.parseStruct(Tensor, _indices);
        var idx_axes: Shape.AxesArray = .{};
        for (idx_tags.slice()) |t| {
            idx_axes.appendAssumeCapacity(self.axis(t));
        }

        // TODO: sort indices following self.shape instead of asking the user to do it.
        return ops.gather(self, idx_axes.slice(), idx_per_axis.slice(), opts);
    }

    test gather {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        const Local = struct {
            pub fn _idx(idx_shape: anytype) Tensor {
                return Tensor.constant(.{ .i32 = 0 }).broad(Shape.init(idx_shape, .i64));
            }
        };

        const idx = Local._idx;

        {
            // Only test shapes
            var comp = zml.module.CompilationContext.init(std.testing.allocator, platform);
            defer comp.deinit();
            comp.activate();
            defer comp.deactivate();

            const block = mlir.Block.init(&.{}, &.{});
            comp.pushBlock(block);
            defer comp.popBlock();

            inline for (.{
                .{ .{ .a = 10 }, .{ .a = idx(.{}) }, .{} },
                .{ .{ .a = 10 }, .{ .a = idx(.{ .n = 8 }) }, .{ .n = 8 } },
                .{ .{ .a = 10, .b = 20 }, .{ .a = idx(.{}) }, .{ .b = 20 } },
                .{ .{ .a = 10, .b = 20 }, .{ .a = idx(.{ .n = 8 }) }, .{ .n = 8, .b = 20 } },
                // .{ .{ .a = 10, .b = 20 }, 0, idx(.{ .n = 8 }), .{ .n = 8, .b = 20 } },
                // Favor val shape, instead of indices shape.
                .{ .{ .a = 10, .b = 20 }, .{ .b = idx(.{ .n = 8 }) }, .{ .a = 10, .n = 8 } },
                .{ .{ .a = 10, .b = 20, .c = 30 }, .{ .b = idx(.{ .n = 8 }) }, .{ .a = 10, .n = 8, .c = 30 } },
                // batching axes are implicits.
                .{ .{ .a = 10, .b = 20 }, .{ .b = idx(.{ .a = 10 }) }, .{ .a = 10 } },
                .{ .{ .a = 10, .b = 20 }, .{ .a = idx(.{ .b = 20 }) }, .{ .b = 20 } },
                .{ .{ .a = 10, .b = 20 }, .{ .b = idx(.{ .a = 10, .n = 8 }) }, .{ .a = 10, .n = 8 } },
                // stablehlo.gather is biased toward indices shape (like gatherSlice).
                // This make it awkward to use when you have both batching dimension and new indices dimensions.
                // For now we reject those, and let user explicitly transpose self or indices if needed.
                // .{ .{ .a = 10, .b = 20 }, .{.b = idx(.{ .n = 8, .a = 10 })}, .{ .a = 10, .n = 8 } },
                // Also handle tuples
                .{ .{ .a = 10, .b = 20 }, .{ .a = idx(.{ .n = 8 }), .b = idx(.{ .n = 8 }) }, .{ .n = 8 } },
            }) |testcase| {
                const x_shape, const indices, const res_shape = testcase;
                const x = Tensor.constant(.{ .f16 = 0 }).broad(Shape.init(x_shape, .i64));
                const y = gather(x, indices, .{});
                try zml.testing.expectEqualShapes(Shape.init(res_shape, .f16), y.shape());
                try std.testing.expect(y.value().owner().verify());
            }

            inline for (.{
                .{ .{ 10, 20 }, &[_]u3{ 0, 1 }, &[_]Tensor{ idx(.{8}), idx(.{8}) }, .{8} },
            }) |testcase| {
                const x_shape, const idx_axes, const idx_per_axis, const res_shape = testcase;
                const x = Tensor.constant(.{ .f16 = 0 }).broad(Shape.init(x_shape, .i64));
                const y = ops.gather(x, idx_axes, idx_per_axis, .{});
                try zml.testing.expectEqualShapes(Shape.init(res_shape, .f16), y.shape());
                try std.testing.expect(y.value().owner().verify());
            }
        }
    }

    /// Gathers slices along the given axes with runtime indices.
    /// * slice_shape represents the shape of the slices to extract,
    ///   it must be smaller than original shape.
    ///   It must use a subset of self axes.
    ///   If slice_shape is **not** tagged, then it must have the same rank than self.
    /// * `indices` represents a set of coordinates.
    ///   The coordinates are read from the `.coord` axis, or last axis if `.coord` is not found.
    ///   The coordinate axis must have `slice_shape.rank()` dims.
    ///   The coordinates represent the "top-left" corner of the slice to extract.
    /// * the output tensor starts with axes from `indices`.
    /// * if the input tensor has tagged axes, matching `indices` axes,
    ///    they will be considered "batching" axes.
    ///
    /// Sample input/output shapes:
    /// * gatherSlices([A, B, C, D], .{.b=B', .c=C'}, [N, 2]) -> [N, A, B', C', D]
    /// * gatherSlices(x(a,b,c,d), .{.b=B', .c=C'}, g(n,m)) = z(n, a, b', c', d) = x(a, g(n, 0) + b', g(n, 1) + c', d)
    ///
    /// Note: the axis order of the result is different from gatherValues.
    /// This is because gatherSlices, favorizes contiguous copy of the extracted slices,
    /// while gatherValues, always copy values one by one, and as such don't have the same issues.
    /// In our example the contiguous dimension .d is not sliced
    /// and gatherSlices can copy data by group of C'*D elements.
    pub fn gatherSlices(self: Tensor, slice_shape_: anytype, indices: Tensor, opts: GatherOpts) Tensor {
        const slice_shape = if (@TypeOf(slice_shape_) == Shape) slice_shape_ else Shape.init(slice_shape_, .i32);
        // scoped_log.debug("gatherSlice({}, {f}, {})", .{ self, slice_shape, indices });

        const tagged_api = slice_shape.isFullyTagged();
        if (tagged_api) {
            for (slice_shape.tags()) |t| {
                stdx.debug.assert(self._shape.hasTag(t) != null, "gatherSlices expects `slices_shape` to only use tags from `self`. But {s} wasn't found in {f}", .{ t, self });
            }
        } else {
            // For untagged api, we require all slices to be specified.
            // Note: we could relax this and right align the slice.
            stdx.debug.assert(slice_shape.rank() == self.rank(), "gatherSlices expects `slice_shape.rank()` to match `self.rank()`. Got: gatherSlices({f}, slice={f}). To avoid specifying all axes in `slice_shape`, you can use tags.", .{ self, slice_shape });
        }

        const index_coord_axis = indices._shape.hasTag(.coord) orelse indices._shape.axis(-1);
        stdx.debug.assert(indices.dim(index_coord_axis) == slice_shape.rank(), "gatherSlices({f}, slice={f}, indices) expects 'indices' to be a tensor [..., {}], got {f}", .{ self, slice_shape, slice_shape.rank(), indices });

        // Compute result shape
        var res_shape = indices._shape.remove(index_coord_axis).withDtype(self.dtype());
        var slice_dims = self._shape._dims;
        var self_batch_axes: stdx.BoundedArray(i64, constants.MAX_RANK) = .{};
        var indices_batch_axes: stdx.BoundedArray(i64, constants.MAX_RANK) = .{};
        var start_index_map: stdx.BoundedArray(i64, constants.MAX_RANK) = .{};
        var self_offset_axes: stdx.BoundedArray(i64, constants.MAX_RANK) = .{};
        for (self._shape.tags(), 0..self.rank()) |t, self_ax| {
            const maybe_slice_ax: ?u3 = if (tagged_api) slice_shape.hasTag(t) else @intCast(self_ax);

            if (tagged_api and indices._shape.hasTag(t) != null) {
                // tag is both in self and indices -> it's a batching dim
                // Note: tags are required for batching.
                self_batch_axes.appendAssumeCapacity(@intCast(self_ax));
                indices_batch_axes.appendAssumeCapacity(indices._shape.axis(t));
                slice_dims.set(self_ax, 1);
                stdx.debug.assert(slice_shape.hasTag(t) == null, "gatherSlices expect axes to be either batches or slices axes. Axis {s} has been found both in `slices={f}` and `indices={f}`", .{ t, slice_shape, indices });
            } else if (maybe_slice_ax) |slice_ax| {
                // Specified axes contains the start offset of the slices,
                // and are collected in `start_index_map`.
                const slice_dim = slice_shape.dim(slice_ax);
                stdx.debug.assert(slice_dim <= self._shape.dim(self_ax), "gatherSlices expects `slice_shape` to be smaller than `self.shape()`. On axis {s}, got {f} > {f}.", .{ t, slice_shape, self._shape });
                slice_dims.set(self_ax, slice_dim);
                res_shape = res_shape.appendDim(slice_dim, t);
                start_index_map.appendAssumeCapacity(@intCast(self_ax));
                self_offset_axes.appendAssumeCapacity(res_shape.rank() - 1);
            } else {
                // non-batching, non-indexed axes
                res_shape = res_shape.appendDim(self.dim(self_ax), t);
                self_offset_axes.appendAssumeCapacity(res_shape.rank() - 1);
            }
        }

        const gather_op = dialects.stablehlo.gather(
            mlirCtx(),
            self.value(),
            indices.value(),
            slice_dims.constSlice(),
            .{
                .offset_dims = self_offset_axes.constSlice(),
                .collapsed_slice_dims = &.{},
                .operand_batching_dims = self_batch_axes.constSlice(),
                .start_indices_batching_dims = indices_batch_axes.constSlice(),
                .start_index_map = start_index_map.constSlice(),
                .index_vector_dim = index_coord_axis,
                .indices_are_sorted = opts.indices_are_sorted,
            },
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());
        return _result(res_shape, gather_op.result(0));
    }

    test gatherSlices {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        const Local = struct {
            pub fn _gatherSlices(self: Tensor, slice_shape: Shape, indices: Tensor, opts: GatherOpts) Tensor {
                return self.gatherSlices(slice_shape, indices, opts);
            }
        };

        {
            // Only test shapes
            var comp = zml.module.CompilationContext.init(std.testing.allocator, platform);
            defer comp.deinit();
            comp.activate();
            defer comp.deactivate();

            const block = mlir.Block.init(&.{}, &.{});
            comp.pushBlock(block);
            defer comp.popBlock();

            inline for (.{
                .{ .{ .a = 10 }, .{}, .{ ._ = 0 }, .{ .a = 10 } },
                .{ .{ .a = 10 }, .{ .a = 7 }, .{ ._ = 1 }, .{ .a = 7 } },
                .{ .{ .a = 10 }, .{ .a = 7 }, .{ .n = 8, ._ = 1 }, .{ .n = 8, .a = 7 } },
                .{ .{ .a = 10 }, .{ .a = 7 }, .{ .coord = 1, .n = 8 }, .{ .n = 8, .a = 7 } },
                // tags aren't required.
                .{ .{10}, .{7}, .{ .n = 8, ._ = 1 }, .{ .n = 8, ._ = 7 } },
                .{ .{ .a = 10, .b = 20 }, .{ .a = 7 }, .{ ._ = 1 }, .{ .a = 7, .b = 20 } },
                .{ .{ .a = 10, .b = 20 }, .{ .a = 7 }, .{ .n = 8, ._ = 1 }, .{ .n = 8, .a = 7, .b = 20 } },
                .{ .{ .a = 10, .b = 20 }, .{ .a = 7 }, .{ .n = 8, .coord = 1, .m = 9 }, .{ .n = 8, .m = 9, .a = 7, .b = 20 } },
                .{ .{ .a = 10, .b = 20 }, .{ .b = 17 }, .{ .n = 8, ._ = 1 }, .{ .n = 8, .a = 10, .b = 17 } },
                .{ .{ .a = 10, .b = 20 }, .{ .a = 7, .b = 17 }, .{ .n = 8, ._ = 2 }, .{ .n = 8, .a = 7, .b = 17 } },
                // Note: currently the order of the axes in the slice is not used.
                .{ .{ .a = 10, .b = 20 }, .{ .b = 17, .a = 7 }, .{ .n = 8, ._ = 2 }, .{ .n = 8, .a = 7, .b = 17 } },
                .{ .{ .a = 10, .b = 20, .c = 20 }, .{ .b = 17 }, .{ .n = 8, ._ = 1 }, .{ .n = 8, .a = 10, .b = 17, .c = 20 } },
                // batching dims
                .{ .{ .a = 10, .b = 20 }, .{ .b = 17 }, .{ .a = 10, ._ = 1 }, .{ .a = 10, .b = 17 } },
                .{ .{ .b = 200, .a = 100, .c = 300 }, .{ .c = 300 }, .{ .a = 100, .b = 200, ._ = 1 }, .{ .a = 100, .b = 200, .c = 300 } },
            }) |testcase| {
                const x_shape, const slice_dims, const idx_shape, const res_shape = testcase;
                const x = Tensor.constant(.{ .f16 = 0 }).broad(Shape.init(x_shape, .f16));
                const slice_shape = Shape.init(slice_dims, .u16);
                const idx = Tensor.constant(.{ .i32 = 0 }).broad(Shape.init(idx_shape, .i32));
                const y = gatherSlices(x, slice_shape, idx, .{});
                try zml.testing.expectEqualShapes(Shape.init(res_shape, .f16), y.shape());
                try std.testing.expect(y.value().owner().verify());

                _ = Local._gatherSlices(x, slice_shape, idx, .{ .indices_are_sorted = true });
            }
        }

        // Test with actual values.
        const operand: Tensor = .init(.{ .a = 2, .b = 4, .c = 6 }, .u16);
        const indices: Tensor = .init(.{ .n = 2, ._ = 2 }, .i32);
        var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local._gatherSlices, .{ operand, Shape.init(.{ .b = 2, .c = 3 }, .u16), indices, .{} }, platform);
        defer exe.deinit();

        var indices_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, indices.shape(), std.mem.sliceAsBytes(&[2][2]i32{ .{ 2, 1 }, .{ 0, 3 } }));
        defer indices_buffer.deinit();
        var operand_buffer: zml.Buffer = b: {
            const temp_func = struct {
                fn forward() Tensor {
                    return Tensor.arange(.{ .end = 2 * 4 * 6 }, .u16).reshape(.{ .a = 2, .b = 4, .c = 6 });
                }
            }.forward;
            var temp_exe = try zml.module.compile(std.testing.allocator, std.testing.io, temp_func, .{}, platform);
            defer temp_exe.deinit();

            const buffer = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &temp_exe, temp_func, {});
            break :b buffer;
        };
        defer operand_buffer.deinit();
        var result = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Local._gatherSlices, .{ operand_buffer, indices_buffer });
        defer result.deinit();

        const expected: zml.Slice = .init(Shape.init(.{ 2, 2, 2, 3 }, .u16), std.mem.sliceAsBytes(&[2][2][2][3]u16{
            .{
                .{ .{ 13, 14, 15 }, .{ 19, 20, 21 } },
                .{ .{ 37, 38, 39 }, .{ 43, 44, 45 } },
            },
            .{
                .{ .{ 3, 4, 5 }, .{ 9, 10, 11 } },
                .{ .{ 27, 28, 29 }, .{ 33, 34, 35 } },
            },
        }));
        try zml.testing.expectClose(std.testing.io, expected, result, 0);
    }

    pub const ScatterOpts = struct {
        /// Promise scatter that all coordinates in `indices` are sorted, wrt to the final offset in `self`
        /// Result is undefined if the promise is violated.
        indices_are_sorted: bool = false,

        /// Promise scatter that slices don't overlap.
        /// Result is undefined if the promise is violated.
        /// This allows for better code generation, because it means that updates can be applied in parallel.
        indices_are_unique: bool = false,

        /// Function used to update previous value in `self` with values from `updates`.
        /// If `update_fn` is not associative (ie the order of execution matters),
        /// then you should make sure the slices don't overlap,
        /// otherwise the result will depend on the runtime scheduling
        /// of the operator which is backend specific.
        update_fn: *const fn (ops.ScatterArgs) struct { Tensor } = increment,

        pub fn increment(values: ops.ScatterArgs) struct { Tensor } {
            return .{values.input.add(values.update)};
        }

        pub fn override(values: ops.ScatterArgs) struct { Tensor } {
            return .{values.update};
        }
    };

    /// Update the given tensor, by copying `values` into slice by slice into `self`.
    /// The slices are chosen at runtime by interpreting indices as coordinates into `self`.
    /// This is a generalized version of `dynamicUpdateSlice` where more than one offset can be specified at a time.
    ///
    /// ### Arguments
    ///
    /// - Return a tensor with same shape than `self`, with updated content.
    /// - `indices` is a set of Tensor (typically rank 1), representing coordinates into `self`.
    ///   all indices must have the same shape, but scalars are accepted.
    /// - each `indices` entry contains offset along an axes into `self`.
    /// Typically axes are identified by their tags, but in the absence of tags on `indices`,
    /// The entry in indices will be assigned to axes of `self` from major to minor axis.
    /// It is recommended to have indices referencing only major axes of `self` for better performance.
    /// - `values` shape is obtained by concatenating the shape of `indices` with the shape of the slices to be extracted.
    /// - `opts`: `zml.Tensor.ScatterOpts` des
    ///
    /// ### Sample input/output shapes with corresponding pseudo-code.
    ///
    /// Basic `scatterSlices` with the first two axes (.a, .b) being indexed, and full (.c, .d) slice copies:
    ///
    /// ```
    /// fn scatterSlices(x[A, B, C, D], .{.a=off_a[N], .b=off_b[N]}, y[N, C, D]) [A, B, C, D] {
    ///     var z = x;
    ///     for (0..N) |n| {
    ///         for (0..C) |c| for (0..D) |d| {{
    ///             z[off_a[n],off_b[n],c,d] += y[n, c, d];
    ///         }}
    ///     }
    ///     return z;
    /// }
    /// ```
    ///
    /// `scatterSlices` with the first three axes (.a, .b, .c) being indexed, and a partial copy of (.c, .d).
    /// Note that .c axis is present both in the indices and updates, and `updates.dim(.c) < self.dim(.c)`.
    ///
    /// ```
    /// fn scatterSlices(x[A, B, C, D], .{.a=off_a[N], .b=off_b[N], .c=off_c[N]}, y[N, C', D]) [A, B, C, D] {
    ///     var z = x;
    ///     for (0..N) |n| {
    ///        for (0..C') |c| for (0..D) |d| {{
    ///           z[off_a[n],off_b[n],off_c[n]+c,d] += y[n, c, d];
    ///        }}
    ///     }
    ///     return z;
    /// }
    /// ```
    ///
    /// `scatterSlices` with the first axis .a being indexed, and where .b is used as a batching axis.
    /// Note that here .b axis is present in `self`, `off_a`, and `updates`,
    /// and is not mentionned in the axes of indices.
    ///
    /// ```
    /// fn scatterSlices(x[A, B, C, D], .{.a=off_a[B,N]}, y[N, B, C, D]) [A, B, C, D] {
    ///     var z = x;
    ///     for (0..B) |b| {
    ///         for (0..N) |n| {
    ///             for (0..C) |c| for (0..D) |d| {{
    ///                 z[off_a[b,n],b,c,d] += y[n, b, c, d];
    ///             }}
    ///         }
    ///     }
    ///     return z;
    /// }
    /// ```
    ///
    /// ### Warnings
    ///
    /// - if `opts.update_fn` is not associative not all calls to `scatterSlices` are sound.
    /// In particular if you scatter overlapping slices, with `zml.Tensor.ScatterOpts.override`,
    /// then the result will depend on the execution order that you don't control.
    /// - `scatterSlices` is a very expressive operator, and can lead to complicated code generation
    /// that requires host<->device synchronization.
    /// ZML tries to generate the easiest to optimize IR, and will warn you if it generates known problematic IR.
    pub fn scatterSlices(self: Tensor, indices: anytype, updates: Tensor, opts: ScatterOpts) Tensor {
        //scoped_log.debug("scatterSlices({}, {any}, {})", .{ self, indices, updates });

        const UpdateType = @TypeOf(ScatterOpts.increment);

        const Custom = struct {
            pub fn inc(values: ops.ScatterArgs, custom: *const UpdateType) struct { Tensor } {
                return @call(.auto, custom, .{values});
            }
        };

        return ops.scatter(.{self}, indices, .{updates}, Custom.inc, .{opts.update_fn}, opts)[0];
    }

    test scatterSlices {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        const Local = struct {
            pub fn _scatter(self: Tensor, indices: []const Tensor, updates: Tensor) Tensor {
                return self.scatterSlices(
                    indices,
                    updates,
                    .{ .update_fn = ScatterOpts.increment },
                );
            }

            pub fn _scatterCB(self: Tensor, coords: Tensor, updates: Tensor) Tensor {
                return self.scatterSlices(
                    .{ .c = coords.choose1d(.coord, 0), .b = coords.choose1d(.coord, 1) },
                    updates,
                    .{ .update_fn = ScatterOpts.increment },
                );
            }

            pub fn _idx(idx_shape: anytype) Tensor {
                return Tensor.constant(.{ .i32 = 0 }).broad(Shape.init(idx_shape, .i32));
            }
        };

        {
            // Only test shapes
            var comp = zml.module.CompilationContext.init(std.testing.allocator, platform);
            defer comp.deinit();
            comp.activate();
            defer comp.deactivate();

            const block = mlir.Block.init(&.{}, &.{});
            defer block.deinit();
            comp.pushBlock(block);
            defer comp.popBlock();

            const idx = Local._idx;

            inline for (.{
                // This is equivalent to a dynamic update slice, update 3 values at given offset of axis .a:
                .{ .{ .a = 10 }, .{ .a = idx(.{}) }, .{ .a = 3 } },
                // Use .a as a batching axis with .a=10 x .n=8 updates of 2 elements of .b
                .{ .{ .a = 10, .b = 20 }, .{ .b = idx(.{ .a = 10, .n = 8 }) }, .{ .a = 10, .n = 8, .b = 2 } },
                // Same but with update transposed
                .{ .{ .a = 10, .b = 20 }, .{ .b = idx(.{ .a = 10, .n = 8 }) }, .{ .a = 10, .b = 2, .n = 8 } },
                // similar, but use the normalized form where a is no longer an explicit batching axis.
                .{ .{ .a = 10, .b = 20 }, .{ .a = idx(.{ .a2 = 10, .n = 8 }), .b = idx(.{ .a2 = 10, .n = 8 }) }, .{ .a2 = 10, .n = 8, .b = 2 } },
                .{ .{ .a = 10, .b = 20 }, .{ .a = idx(.{ .a = 10, .n = 8 }), .b = idx(.{ .a = 10, .n = 8 }) }, .{ .a = 10, .n = 8, .b = 2 } },
                .{ .{ .a = 10, .b = 20 }, .{ .a = idx(.{ .n = 8 }) }, .{ .n = 8, .a = 2 } },
                .{ .{ .a = 10, .b = 20 }, .{ .b = idx(.{ .n = 8 }), .a = idx(.{ .n = 8 }) }, .{ .n = 8, .a = 3, .b = 2 } },
                .{ .{ .a = 10, .b = 20 }, .{ .a = idx(.{ .n = 8 }), .b = idx(.{ .n = 8 }) }, .{ .a = 3, .n = 8, .b = 2 } },
            }) |testcase| {
                const x_shape, const indices, const updates_shapes = testcase;
                const x = Tensor.constant(.{ .f16 = 0 }).broad(Shape.init(x_shape, .f16));
                const updates = Tensor.constant(.{ .f16 = 0 }).broad(Shape.init(updates_shapes, .f16));

                const y = scatterSlices(x, indices, updates, .{});
                // Shape doesn't change with scatterSlices
                try zml.testing.expectEqualShapes(x.shape(), y.shape());
                try std.testing.expect(y.value().owner().verify());
            }
        }
        // Test with actual values, no batching.
        {
            const a: Tensor = .init(.{ 3, 3 }, .i32);

            const scatter_indices = Tensor.init(.{2}, .i32).withTags(.{.n});
            const updates = Tensor.init(.{ 2, 3 }, .i32).withTags(.{ .n, .b });

            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local._scatter, .{ a, &.{scatter_indices}, updates }, platform);
            defer exe.deinit();

            var a_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, a.shape(), std.mem.sliceAsBytes(&[9]i32{ 0, 1, 2, 3, 4, 5, 6, 7, 8 }));
            defer a_buffer.deinit();
            var scatter_indices_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, scatter_indices.shape(), std.mem.sliceAsBytes(&[2]i32{ 0, 2 }));
            defer scatter_indices_buffer.deinit();
            var updates_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, updates.shape(), std.mem.sliceAsBytes(&[2][3]i32{ .{ 10, 20, 30 }, .{ 70, 80, 90 } }));
            defer updates_buffer.deinit();

            var result = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Local._scatter, .{ a_buffer, &.{scatter_indices_buffer}, updates_buffer });
            defer result.deinit();

            const expected = [3][3]i32{ .{ 10, 21, 32 }, .{ 3, 4, 5 }, .{ 76, 87, 98 } };
            try std.testing.expect(a.shape().eql(result.shape()));
            try std.testing.expectEqual(expected, result.getValue(@TypeOf(expected), std.testing.io));
        }
        // Test with setting individual values (no batching)
        {
            const a: Tensor = .init(.{9}, .i32);

            const scatter_indices = Tensor.init(.{2}, .i32).withTags(.{.n});
            const updates = Tensor.init(.{2}, .i32).withTags(.{.n});

            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local._scatter, .{ a, &.{scatter_indices}, updates }, platform);
            defer exe.deinit();

            var a_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, a.shape(), std.mem.sliceAsBytes(&[9]i32{ 0, 1, 2, 3, 4, 5, 6, 7, 8 }));
            defer a_buffer.deinit();
            var scatter_indices_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, scatter_indices.shape(), std.mem.sliceAsBytes(&[2]i32{ 2, 7 }));
            defer scatter_indices_buffer.deinit();
            var updates_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, updates.shape(), std.mem.sliceAsBytes(&[2]i32{ 20, 70 }));
            defer updates_buffer.deinit();

            var result = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Local._scatter, .{ a_buffer, &.{scatter_indices_buffer}, updates_buffer });
            defer result.deinit();

            const expected = [9]i32{ 0, 1, 22, 3, 4, 5, 6, 77, 8 };
            try std.testing.expect(a.shape().eql(result.shape()));
            try std.testing.expectEqual(expected, result.getValue(@TypeOf(expected), std.testing.io));
        }
        {
            // Test with actual values and batching along axis .a
            const operand: Tensor = .init(.{ .a = 2, .b = 3, .c = 4, .d = 2 }, .u16);
            const start_indices: Tensor = .init(.{ .n = 2, .a = 2, .m = 3, .coord = 2 }, .i32);
            const values: Tensor = .init(.{ .n = 2, .a = 2, .m = 3, .c = 2, .d = 2 }, .u16);

            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local._scatterCB, .{ operand, start_indices, values }, platform);
            defer exe.deinit();

            var operand_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, operand.shape(), std.mem.sliceAsBytes(&@as([2 * 3 * 4 * 2]u16, @splat(0))));
            defer operand_buffer.deinit();
            var start_indices_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, start_indices.shape(), std.mem.sliceAsBytes(&[2][2][3][2]i32{
                .{
                    .{ .{ 0, 0 }, .{ 1, 0 }, .{ 2, 1 } },
                    .{ .{ 0, 1 }, .{ 1, 1 }, .{ 0, 9 } },
                },
                .{
                    .{ .{ 0, 0 }, .{ 2, 1 }, .{ 2, 2 } },
                    .{ .{ 1, 2 }, .{ 0, 1 }, .{ 1, 0 } },
                },
            }));
            defer start_indices_buffer.deinit();
            var values_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, values.shape(), std.mem.sliceAsBytes(&@as([2 * 2 * 3 * 2 * 2]u16, @splat(1))));
            defer values_buffer.deinit();

            var result = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Local._scatterCB, .{ operand_buffer, start_indices_buffer, values_buffer });
            defer result.deinit();

            const expected = [2][3][4][2]u16{
                .{
                    .{ .{ 2, 2 }, .{ 3, 3 }, .{ 1, 1 }, .{ 0, 0 } },
                    .{ .{ 0, 0 }, .{ 0, 0 }, .{ 2, 2 }, .{ 2, 2 } },
                    .{ .{ 0, 0 }, .{ 0, 0 }, .{ 1, 1 }, .{ 1, 1 } },
                },
                .{
                    .{ .{ 0, 0 }, .{ 1, 1 }, .{ 1, 1 }, .{ 0, 0 } },
                    .{ .{ 2, 2 }, .{ 3, 3 }, .{ 1, 1 }, .{ 0, 0 } },
                    .{ .{ 0, 0 }, .{ 1, 1 }, .{ 1, 1 }, .{ 0, 0 } },
                },
            };
            try std.testing.expect(operand.shape().eql(result.shape()));
            try std.testing.expectEqual(expected, result.getValue(@TypeOf(expected), std.testing.io));
        }
    }

    /// Returns a Tensor containing the maximum over a given axis.
    pub fn max(self: Tensor, axis_: anytype) Tensor {
        const a = self.axis(axis_);
        return ops.reduce(
            .{self},
            .{Tensor.constant(self.dtype().minValue())},
            &.{a},
            struct {
                pub fn cmp(args: ops.ReduceArgs) struct { Tensor } {
                    return .{args.left.maximum(args.right)};
                }
            }.cmp,
            .{},
        )[0];
    }

    /// Returns a Tensor containing the minimum over a given axis.
    pub fn min(self: Tensor, axis_: anytype) Tensor {
        const a = self.axis(axis_);
        return ops.reduce(
            .{self},
            .{Tensor.constant(self.dtype().maxValue())},
            &.{a},
            struct {
                pub fn cmp(args: ops.ReduceArgs) struct { Tensor } {
                    return .{args.left.minimum(args.right)};
                }
            }.cmp,
            .{},
        )[0];
    }

    pub const ArgMaxRes = struct {
        values: Tensor,
        indices: Tensor,

        fn cmp(values: ops.ReduceArgs, indices: ops.ReduceArgs) struct { Tensor, Tensor } {
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
    };

    pub fn argMax(x: Tensor, axis_: anytype) ArgMaxRes {
        const a = x.axis(axis_);
        const dt: DataType = if (x.dim(a) <= std.math.maxInt(i32)) .i32 else .i64;

        const values, const indices = ops.reduce(
            .{ x, Tensor.arange(.{ .end = x.dim(axis_) }, dt).broadcast(x.shape(), &.{a}) },
            .{ Tensor.constant(x.dtype().minValue()), Tensor.constant(dt.zero()) },
            &.{a},
            ArgMaxRes.cmp,
            .{},
        );
        return .{ .values = values, .indices = indices };
    }

    test argMax {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();
        const allocator = std.testing.allocator;
        const ArgMaxTest = struct {
            pub fn _fwd(x: Tensor) Tensor.ArgMaxRes {
                return x.argMax(1);
            }
        };

        const x: Tensor = .init(.{ 1, 5 }, .f32);
        var exe = try zml.module.compile(allocator, std.testing.io, ArgMaxTest._fwd, .{x}, platform);
        defer exe.deinit();

        {
            var x_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&[1][5]f32{.{ 5.0, 4.1, 7.9, 0, 7.9 }}));
            defer x_buffer.deinit();

            var res = try zml.testing.autoCall(allocator, std.testing.io, &exe, ArgMaxTest._fwd, .{x_buffer});
            defer res.values.deinit();
            defer res.indices.deinit();
            const max_ = res.values.getValue(f32, std.testing.io);
            const max_idx = res.indices.getValue(i32, std.testing.io);
            try std.testing.expectEqual(max_, 7.9);
            // We should always return the first max found.
            try std.testing.expectEqual(max_idx, 2);
        }

        {
            var x_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&[1][5]f32{.{ 5.0, std.math.nan(f32), 7.9, 0, 7.9 }}));
            defer x_buffer.deinit();

            var res = try zml.testing.autoCall(allocator, std.testing.io, &exe, ArgMaxTest._fwd, .{x_buffer});
            defer res.values.deinit();
            defer res.indices.deinit();
            const max_ = try res.values.getValue(f32, std.testing.io);
            const max_idx = try res.indices.getValue(i32, std.testing.io);
            try std.testing.expect(std.math.isNan(max_));
            try std.testing.expectEqual(max_idx, 1);
        }
    }

    pub const SortRes = ArgMaxRes;

    pub const SortOpts = struct { descending: bool = false };

    /// Returns two Tensors. The first contains the sorted values and the second one contains the sorted indices.
    pub fn sort(self: Tensor, axis_: anytype, opts: SortOpts) SortRes {
        const a = self.axis(axis_);
        const indices = Tensor.arange(.{ .end = self.dim(a) }, .i32).broadcast(self._shape, &.{a});
        const direction: dialects.stablehlo.ComparisonDirection.Direction = if (opts.descending) .GT else .LT;
        const res = ops.sort(.{ self, indices }, a, struct {
            fn call(values: ops.SortArgs, _: ops.SortArgs, direction_: dialects.stablehlo.ComparisonDirection.Direction) Tensor {
                return values.left.cmp(direction_, values.right);
            }
        }.call, .{direction}, true);
        return .{ .values = res[0], .indices = res[1] };
    }

    pub const ArgSortOpts = SortOpts;

    /// Returns a Tensor containing the indices corresponding to the sorted values over the given axis.
    pub fn argsort(self: Tensor, axis_: anytype, opts: ArgSortOpts) Tensor {
        return self.sort(axis_, .{ .descending = opts.descending }).indices;
    }

    test argsort {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        const Local = struct {
            pub fn _argsort(x: Tensor, axis_: u3, opts: ArgSortOpts) Tensor {
                return x.argsort(axis_, opts);
            }
        };

        // 2D Tensor - dim = 1, ascending
        {
            const x: Tensor = .init(.{ 2, 5 }, .f32);

            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local._argsort, .{ x, 1, .{} }, platform);
            defer exe.deinit();

            var x_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&[_]f32{ -0.9264, 0.7156, 1.0202, 0.3992, 1.2349, 1.0003, -0.1932, 1.3935, 0.7316, 0.0851 }));
            defer x_buffer.deinit();

            var res = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Local._argsort, .{x_buffer});
            defer res.deinit();

            const res_cpu = try res.toSliceAlloc(std.testing.allocator, std.testing.io);
            defer res_cpu.free(std.testing.allocator);

            try std.testing.expectEqualSlices(i32, &.{ 0, 3, 1, 2, 4, 1, 4, 3, 0, 2 }, res_cpu.items(i32));
        }

        // 3D Tensor, dim = 1, descending
        {
            const x: Tensor = .init(.{ 1, 5, 10 }, .f16);

            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local._argsort, .{ x, 1, .{ .descending = true } }, platform);
            defer exe.deinit();

            var x_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&[_]f16{
                -0.2505, 1.2520,  -0.7041, 0.1066,  1.2773,  -1.7246, 0.8389,  1.1094,  0.0601,  1.0684,
                0.9619,  1.3916,  1.2246,  -0.1406, 0.3674,  -1.2480, -1.7051, -0.0934, 0.3435,  0.4373,
                1.3809,  0.5444,  -0.6079, 1.2031,  -0.6880, 1.2979,  -0.1869, 0.2991,  0.0156,  0.1847,
                0.6626,  -0.3040, -0.8726, -1.4805, -1.6943, 1.1055,  -2.0078, -0.5288, 0.8813,  0.8008,
                2.0527,  1.1230,  0.5430,  0.2494,  -0.9434, 0.7876,  0.1818,  0.9258,  -2.4902, 1.5918,
            }));
            defer x_buffer.deinit();

            var res = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Local._argsort, .{x_buffer});
            defer res.deinit();

            const res_cpu = try res.toSliceAlloc(std.testing.allocator, std.testing.io);
            defer res_cpu.free(std.testing.allocator);

            try std.testing.expectEqualSlices(i32, &.{
                4, 1, 1, 2, 0, 2, 0, 0, 3, 4,
                2, 0, 4, 4, 1, 3, 4, 4, 1, 0,
                1, 4, 2, 0, 2, 4, 2, 2, 0, 3,
                3, 2, 0, 1, 4, 1, 1, 1, 2, 1,
                0, 3, 3, 3, 3, 0, 3, 3, 4, 2,
            }, res_cpu.items(i32));
        }

        // 4D Tensor, dim = 3, ascending
        {
            const x: Tensor = .init(.{ 4, 2, 1, 4 }, .i32);

            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local._argsort, .{ x, 3, .{} }, platform);
            defer exe.deinit();

            var x_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&[_]i32{
                89, 31, 22, 42,
                64, 39, 0,  30,
                64, 71, 46, 31,
                89, 82, 78, 86,
                55, 32, 43, 19,
                93, 24, 45, 72,
                64, 86, 62, 88,
                57, 21, 19, 12,
            }));
            defer x_buffer.deinit();

            var res = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Local._argsort, .{x_buffer});
            defer res.deinit();

            const res_cpu = try res.toSliceAlloc(std.testing.allocator, std.testing.io);
            defer res_cpu.free(std.testing.allocator);

            try std.testing.expectEqualSlices(i32, &.{
                2, 1, 3, 0,
                2, 3, 1, 0,
                3, 2, 0, 1,
                2, 1, 3, 0,
                3, 1, 2, 0,
                1, 2, 3, 0,
                2, 0, 1, 3,
                3, 2, 1, 0,
            }, res_cpu.items(i32));
        }
    }

    pub const TopKOpts = struct { descending: bool = true };

    /// Returns a Tensor representing the result of Top-K over the given axis.
    pub fn topK(self: Tensor, named_axis_: anytype, k: u32, opts: TopKOpts) SortRes {
        stdx.debug.assert(k > 0, "topK expects a k > 0, got 0", .{});
        const err_msg = "topK named axis should be an integer or a named axis, eg `x.topK(.{{ .best_token = .token }}, 16)` or `x.topK(-1, 16)`";
        const has_name: ?[:0]const u8, const a = switch (@typeInfo(@TypeOf(named_axis_))) {
            .int, .comptime_int => .{ null, self.axis(@as(i64, @intCast(named_axis_))) },
            .@"struct" => |info| blk: {
                stdx.debug.assertComptime(info.fields.len == 1, err_msg, .{});
                break :blk .{ info.fields[0].name, self.axis(@field(named_axis_, info.fields[0].name)) };
            },
            else => stdx.debug.compileError(err_msg, .{}),
        };
        var result = self.sort(a, .{ .descending = opts.descending });
        result.values = result.values.slice1d(a, .{ .end = k });
        result.indices = result.indices.slice1d(a, .{ .end = k });
        if (has_name) |new_name| {
            result.values._shape._tags.set(a, new_name.ptr);
            result.indices._shape._tags.set(a, new_name.ptr);
        }
        return result;
    }

    pub const MaxPoolRes = ArgMaxRes;

    /// Computes the 1d maxPool operation on the input Tensor.
    pub fn maxPool1d(self: Tensor, opts: struct {
        window_dimensions: i64,
        window_strides: ?i64,
        base_dilations: i64 = 1,
        window_dilations: i64 = 1,
        padding: [2]i64 = .{ 0, 0 },
    }) MaxPoolRes {
        // TODO migrate to the following syntax.
        // maxPool(.{.a = .{ .stride = 5, .dilation = 2, .padding = .{0, 1} },
        //              .b = .{ .stride = 8, .dilation = 2, .padding = .{0, 1} }),
        // maxPool(.{
        //     .stride = .{ .a = 5, .b = 8 },
        //     .dilation = .{ .a = 2, .b = 2 },
        //     .padding = .{ .a = .{ 0, 2 }, .b = .{0, 2}
        // })

        // TODO: support maxPool on non last axis
        const a = self.axis(-1);
        const ones = [_]i64{1} ** constants.MAX_RANK;
        var window_dimensions = ones;
        window_dimensions[a] = opts.window_dimensions;
        var window_strides = window_dimensions;
        if (opts.window_strides) |stride| window_strides[a] = stride;

        var base_dilations = ones;
        base_dilations[a] = opts.base_dilations;
        var window_dilations = ones;
        window_dilations[a] = opts.window_dilations;

        var padding = [_][2]i64{.{ 0, 0 }} ** constants.MAX_RANK;
        padding[a] = opts.padding;

        const result = ops.reduceWindow(
            .{ self, iota(self._shape, a) },
            .{ Tensor.constant(self.dtype().minValue()), Tensor.scalar(0, .i32) },
            .{
                .window_dimensions = window_dimensions[0..self.rank()],
                .window_strides = window_strides[0..self.rank()],
                .base_dilations = base_dilations[0..self.rank()],
                .window_dilations = window_dilations[0..self.rank()],
                .padding = padding[0..self.rank()],
            },
            MaxPoolRes.cmp,
            .{},
        );
        return .{ .values = result[0], .indices = result[1] };
    }

    /// Computes the 2d maxPool operation on the input Tensor.
    pub fn maxPool2d(self: Tensor, opts: struct {
        window_dimensions: [2]i64,
        window_strides: ?[2]i64 = null,
        base_dilations: [2]i64 = .{ 1, 1 },
        window_dilations: [2]i64 = .{ 1, 1 },
        padding: [2][2]i64 = .{ .{ 0, 0 }, .{ 0, 0 } },
    }) MaxPoolRes {
        // TODO: rewrite using modern ZML
        stdx.debug.guard(self.rank() == 3 or self.rank() == 4, @src());

        // TODO: support maxPool on non last axis
        // Note: the problem is initPoolArg assuming last axis
        const a = self.axis(-1);

        const window_dimensions = initPoolArg(self.rank(), &opts.window_dimensions);
        const window_strides = if (opts.window_strides) |ws| initPoolArg(self.rank(), &ws) else window_dimensions;
        const base_dilation = initPoolArg(self.rank(), &opts.base_dilations);
        const window_dilations = initPoolArg(self.rank(), &opts.window_dilations);

        var padding = [_][2]i64{.{ 0, 0 }} ** constants.MAX_RANK;
        padding[a - 1] = opts.padding[0];
        padding[a] = opts.padding[1];

        const result = ops.reduceWindow(
            .{ self, iota(self._shape, a) },
            .{ Tensor.constant(self.dtype().minValue()), Tensor.scalar(0, .i32) },
            .{
                .window_dimensions = window_dimensions[0..self.rank()],
                .window_strides = window_strides[0..self.rank()],
                .base_dilations = base_dilation[0..self.rank()],
                .window_dilations = window_dilations[0..self.rank()],
                .padding = padding[0..self.rank()],
            },
            MaxPoolRes.cmp,
            .{},
        );
        return .{ .values = result[0], .indices = result[1] };
    }

    /// Chunk a given tensor into exactly n parts of equal shape.
    /// `self.dim(axis_)` must be divisible by n_chunks.
    pub fn chunkExact(self: Tensor, axis_: anytype, n_chunks: comptime_int) [n_chunks]Tensor {
        const a = self.axis(axis_);
        const d = self.dim(a);
        const chunk_size = @divExact(d, n_chunks);
        var chunks: [n_chunks]Tensor = undefined;
        for (0..n_chunks) |i| {
            const start: i64 = @as(i64, @intCast(i)) * chunk_size;
            chunks[i] = self.slice1d(a, .{ .start = start, .end = start + chunk_size });
        }
        return chunks;
    }

    test chunkExact {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        // Only test shapes
        var comp = zml.module.CompilationContext.init(std.testing.allocator, platform);
        defer comp.deinit();
        comp.activate();
        defer comp.deactivate();

        const block = mlir.Block.init(&.{}, &.{});
        defer block.deinit();
        comp.pushBlock(block);
        defer comp.popBlock();

        inline for (.{
            .{ .{ .a = 12 }, .a, 3, .{ .a = 4 } },
            .{ .{ .a = 12, .b = 2 }, .a, 3, .{ .a = 4, .b = 2 } },
            .{ .{ 12, 2 }, 0, 3, .{ 4, 2 } },
        }) |testcase| {
            const x_shape, const ax, const n_chunks, const res = testcase;
            const x = Tensor.constant(.{ .f16 = 0 }).broad(Shape.init(x_shape, .f16));
            const chunks = x.chunkExact(ax, n_chunks);

            const res_shape = Shape.init(res, .f16);
            for (chunks) |chk| {
                try zml.testing.expectEqualShapes(res_shape, chk.shape());
            }
        }
    }

    /// Chunk a given tensor into n parts of equal shape, and one part with the remaining items.
    /// When `self.dim(axis_)` is divisible by `n_chunks` it will return exactly `n_chunks`.
    pub fn chunkAllowTrailing(
        self: Tensor,
        axis_: i64,
        n_chunks: comptime_int,
    ) ![]Tensor {
        const a = self.axis(axis_);
        const d = self.dim(a);
        const chunk_size: i64 = @divFloor(d, n_chunks);
        const tail_chunk_size: i64 = @rem(d, chunk_size);

        const allocator = CompilationContext.current().arena.allocator();

        var chunks = std.ArrayList(Tensor).initCapacity(allocator, n_chunks + 1) catch @panic("OOM");
        defer chunks.deinit(allocator);

        for (0..n_chunks) |i| {
            const start: i64 = @as(i64, @intCast(i)) * chunk_size;
            chunks.appendAssumeCapacity(
                self.slice1d(a, .{ .start = start, .end = start + chunk_size }),
            );
        }
        if (tail_chunk_size != 0) {
            const start: i64 = n_chunks * chunk_size;
            chunks.appendAssumeCapacity(self.slice1d(a, .{ .start = start }));
        }
        return chunks.toOwnedSlice(allocator);
    }

    test chunkAllowTrailing {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        // Only test shapes
        var comp = zml.module.CompilationContext.init(std.testing.allocator, platform);
        defer comp.deinit();
        comp.activate();
        defer comp.deactivate();

        const block = mlir.Block.init(&.{}, &.{});
        defer block.deinit();
        comp.pushBlock(block);
        defer comp.popBlock();

        inline for (.{
            .{ .{ .a = 10 }, .a, 3, .{ .a = 3 }, .{ .a = 1 } },
            .{ .{ .a = 10, .b = 2 }, .a, 3, .{ .a = 3, .b = 2 }, .{ .a = 1, .b = 2 } },
            .{ .{ 10, 2 }, 0, 3, .{ 3, 2 }, .{ 1, 2 } },
            .{ .{ 12, 2 }, 0, 3, .{ 4, 2 }, .{} },
        }) |testcase| {
            const x_shape, const ax, const n_chunks, const res, const trailing = testcase;
            const x = Tensor.constant(.{ .f16 = 0 }).broad(Shape.init(x_shape, .f16));
            const chunks = try x.chunkAllowTrailing(x.axis(ax), n_chunks);

            const res_shape = Shape.init(res, .f16);
            for (chunks[0..n_chunks]) |chk| {
                try zml.testing.expectEqualShapes(res_shape, chk.shape());
            }
            const trailing_shape = Shape.init(trailing, .f16);
            if (trailing_shape.rank() > 0) {
                try std.testing.expectEqual(n_chunks + 1, chunks.len);
                try zml.testing.expectEqualShapes(trailing_shape, chunks[n_chunks].shape());
            } else {
                try std.testing.expectEqual(n_chunks, chunks.len);
            }
        }
    }

    pub fn split(self: Tensor, axis_: anytype, split_sizes: []const i64) []Tensor {
        stdx.debug.assert(split_sizes.len > 0, "split expects at least one 'split_sizes', got 0", .{});

        const a = self.axis(axis_);
        const d = self.dim(a);
        var split_sum: i64 = 0;
        for (split_sizes) |n| split_sum += n;
        stdx.debug.assert(split_sum == d, "split expects sum of 'split_sizes' values and axis dimension to be equal, got {} and {}", .{ split_sum, d });

        var arena = std.heap.ArenaAllocator.init(CompilationContext.current().allocator);
        defer arena.deinit();

        const res = arena.allocator().alloc(Tensor, split_sizes.len) catch @panic("OOM");

        var start: i64 = 0;
        for (split_sizes, 0..) |n, i| {
            res[i] = self.slice1d(a, .{ .start = start, .end = start + n });
            start += n;
        }
        return res;
    }

    /// Slices the input Tensor along a specific axis, with a start offset known at runtime.
    /// Note: this doesn't support tagging, if you have tags,
    /// you should use `dynamicSlice` directly.
    pub fn dynamicSlice1d(self: Tensor, axis_: i8, slice_: DynSlice) Tensor {
        stdx.debug.assert(slice_.start.rank() == 0, "dynamicSlice1d expects 'slice_.start' tensor rank to be a scalar, got {f}", .{slice_.start});

        const a = self.axis(axis_);
        const new_shape = self._shape.set(a, slice_.len);

        var start_indices = [_]*const mlir.Value{constant(slice_.start.dtype().zero()).value()} ** constants.MAX_RANK;
        start_indices[a] = slice_.start.value();

        const op = dialects.stablehlo.dynamic_slice(
            mlirCtx(),
            self.value(),
            new_shape.dims(),
            start_indices[0..self.rank()],
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());

        return _result(new_shape, op.result(0));
    }

    pub const DynSlice = struct { start: Tensor, len: i64 };

    /// Slices a Tensor across many axes, with runtime known offsets.
    ///
    /// Due to the nature of stablehlo, the length of the slices need to be known when compiling the IR.
    /// When using the tagged API it is allowed to not specify some axes.
    /// But with the non-tagged API all slices need to be specified.
    /// Examples:
    /// ```
    /// Tensor(.{.a=20,.b=30,.c=40 }).dynamicSlice(.{ .a = .{ .start = a_off, .len = 11});
    /// Tensor(.{.a=20,.b=30,.c=40 }).dynamicSlice(.{
    ///     .a = .{ .start = a_off, .len = 11 },
    ///     .b = .{ .start = b_off, .len = 12 },
    ///   });
    /// Tensor(.{ 20,30,40}).dynamicSlice(.{.{ .start = scalar(0, .i32), .len = 20 }, .{ .start = b_off, .len = 12 }, .{ .start = scalar(0, .i32), .len = 40 }});
    /// ```
    pub fn dynamicSlice(self: Tensor, slices_: anytype) Tensor {
        // TODO: the untagged api is a bit verbose. Should I allow: `Tensor(.{ 20,30,40}).dynamicSlice(.{.{}, .{ .start = b_off, .len = 12 }, .{}});` ??
        //
        const slices, const slices_tags = Shape.parseStruct(DynSlice, slices_);

        // TODO use slices and slices_tags for the format.
        // Currently this prints: "dynSlice(struct{q: struct{start: tensor.Tensor, comptime len: comptime_int = 1}}{ .q = struct{start: tensor.Tensor, comptime len: comptime_int = 1}{ .start = Tensor({1,10}, dtype=.i64), .len = 1 } })"
        // which is kinda ugly.

        const idx_dtype = if (slices.len > 0) slices.get(0).start.dtype() else .i32;
        const zero = Tensor.scalar(0, idx_dtype).value();
        var offset_values = [_]*const mlir.Value{zero} ** constants.MAX_RANK;
        var res_shape = self._shape;
        for (slices.constSlice(), 0..) |slice_, i| {
            const offset = slice_.start;
            const len = slice_.len;
            if (slices_tags.len == 0) {
                stdx.debug.assert(self.rank() == slices.len, "dynamicSlice expects tensor rank and 'slices_' length to be equal, got {d} and {d}", .{ self.rank(), slices.len });

                offset_values[i] = offset.value();
                res_shape._dims.set(i, len);

                stdx.debug.assert(len <= self.dim(i), "dynamicSlice expects slices 'len' to be less than or equal to their corresponding dimension in input tensor, got {d} and {d} for index {d}", .{ len, self.dim(i), i });
            } else {
                const t = slices_tags.get(i);
                const a = res_shape.hasTag(t) orelse stdx.debug.panic("dynamicSlice expects input tensor to have tags used in 'slices_' but {s} is missing (input shape is {f})", .{ t, self._shape });

                stdx.debug.assert(len <= self.dim(a), "dynamicSlice expects slices 'len' to be less than their corresponding dimension in input tensor, got {d} and {d} for axis {s}", .{ len, self.dim(a), t });

                offset_values[a] = offset.value();
                res_shape._dims.set(a, len);
            }
        }
        const op = dialects.stablehlo.dynamic_slice(mlirCtx(), self.value(), res_shape.dims(), offset_values[0..self.rank()], .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(res_shape, op.result(0));
    }

    test dynamicSlice {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        inline for (.{
            .{ [10]f32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, Shape.init(.{10}, .f32), [2]f32{ 4, 5 }, 4, 0 },
            .{ [2][5]f32{ .{ 0, 1, 2, 3, 4 }, .{ 5, 6, 7, 8, 9 } }, Shape.init(.{ 2, 5 }, .f32), [4]f32{ 3, 4, 8, 9 }, 3, 1 },
        }) |testcase| {
            const x_data, const x_shape, const expectation, const z_value: i32, const ax = testcase;
            const x: Tensor = .fromShape(x_shape);
            const z: Tensor = .init(.{}, .i32);

            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Tensor.dynamicSlice1d, .{ x, ax, .{ .len = 2, .start = z } }, platform);
            defer exe.deinit();

            var x_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&x_data));
            defer x_buffer.deinit();
            var z_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, z.shape(), std.mem.asBytes(&z_value));
            defer z_buffer.deinit();

            var res = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Tensor.dynamicSlice1d, .{ x_buffer, .{ .start = z_buffer } });
            defer res.deinit();

            try std.testing.expectEqual(expectation, try res.getValue(@TypeOf(expectation), std.testing.io));
        }
    }

    /// Updates a slice of the input Tensor along a specific axis using the given 'update' Tensor, with a start offset known at runtime.
    /// Note this is the untagged api, if you have tags, you should use dynamicUpdateSlice directly.
    pub fn dynamicUpdateSlice1d(self: Tensor, update: Tensor, axis_: i64, offset: Tensor) Tensor {
        const placeholder = Tensor.scalar(0, .i32);
        var start_indices = [_]Tensor{placeholder} ** constants.MAX_RANK;
        start_indices[self.axis(axis_)] = offset;
        return self.dynamicUpdateSlice(start_indices[0..self.rank()], update);
    }

    /// Updates a part of the input Tensor using the given 'update' Tensor, with runtime known offsets.
    ///
    /// The offsets are specified similarly to the dynamicSlice api.
    /// It's semantically equivalent to:
    /// self.dynamicSlice(offsets_) := update
    /// Examples:
    /// ```
    /// Tensor(.{ .a = 2, .b = 5 }).dynamicUpdateSlice(.{ .a = scalar(1, .i32) }, Tensor(.{ .b = 5 }));
    /// ```
    pub fn dynamicUpdateSlice(self: Tensor, offset_: anytype, update_: Tensor) Tensor {
        // TODO: add updateSlice for when the offset isn't dynamic
        stdx.debug.assert(self.dtype() == update_.dtype(), "dynamicUpdateSlice expects input and 'update_' tensors to be of the same type, got {} and {}", .{ self.dtype(), update_.dtype() });

        const offset, const offset_tags = Shape.parseStruct(Tensor, offset_);
        // log.debug("offset: {any}, offset_tags: {any}", .{ offset, offset_tags });
        for (offset.constSlice(), 0..) |start_idx, i| {
            stdx.debug.assert(start_idx.rank() == 0, "dynamicUpdateSlice expects 'offset_' tensor ranks to be equal to 0, got {} at index {}", .{ start_idx.rank(), i });
        }

        const tagged_api = update_._shape.isFullyTagged() and self._shape.isFullyTagged() and offset_tags.len > 0;
        // When using tags, we can safely insert axis with a 1-dim.
        // the offset into the inserted axis will need to be specified through indices.
        var update = update_;
        if (tagged_api) {
            // Check that all update tags are known.
            for (update._shape._tags.constSlice()) |t| {
                stdx.debug.assert(self._shape.hasTag(t) != null, "dynamicUpdateSlice expects 'update_' tensor tags to be a subset of input tensor tags but {s} is missing (input shape is {f})", .{ t, self._shape });
            }

            var update_shape = self._shape;
            var prev_ax: i8 = -1;
            for (self._shape.tags(), 0..) |t, self_ax| {
                if (update._shape.hasTag(t)) |up_ax| {
                    stdx.debug.assert(up_ax == prev_ax + 1, "dynamicUpdateSlice expects 'update_' and input tensor axis to have the same order, got {f} and {f}. (hint: you need to explicitly transpose 'update_')", .{ update_, self });

                    update_shape._dims.set(self_ax, update.dim(up_ax));
                    prev_ax = up_ax;
                } else {
                    update_shape._dims.set(self_ax, 1);
                }
            }
            update = update.reshape(update_shape);
        }

        stdx.debug.assert(self.rank() == update.rank(), "dynamicUpdateSlice expects input and computed update tensors to have the same rank, got {f} and {f}", .{ self, update });

        for (self.dims(), update.dims(), 0..) |self_d, up_d, ax| {
            const t = self._shape.debugTag(ax);
            stdx.debug.assert(up_d <= self_d, "dynamicUpdateSlice expects 'update_' dimensions to be less than or equal to their corresponding dimension in input tensor, got {} and {} for axis .{s}", .{ up_d, self_d, t });

            if (tagged_api and up_d < self_d) {
                const axis_has_offset = std.mem.indexOfScalar(Shape.Tag, offset_tags.constSlice(), self._shape._tags.get(ax)) != null;

                stdx.debug.assert(axis_has_offset, "dynamicUpdateSlice expects 'update_' dimensions to be equal to their corresponding dimension in input tensor, got {} and {} for axis .{s} (hint: you need to provide an offset)", .{ up_d, self_d, t });
            }
        }

        const idx_dtype = if (offset.len > 0) offset.get(0).dtype() else .i32;
        const zero = Tensor.scalar(0, idx_dtype).value();
        var offset_values: [constants.MAX_RANK]*const mlir.Value = undefined;
        if (offset_tags.len == 0) {
            // Without offset tags we need the same number of offset than rank.
            stdx.debug.assert(self.rank() == offset.len, "dynamicUpdateSlice expects input tensor rank and 'offset_' length to be equal, got {} and {}", .{ self.rank(), offset.len });

            for (offset.constSlice(), 0..) |idx, i| {
                offset_values[i] = idx.value();
            }
        } else {
            // If an axis isn't specified, update the full slice.
            // This is only allowed when using tagged sliced.
            offset_values = .{zero} ** constants.MAX_RANK;
            for (offset.constSlice(), offset_tags.constSlice()) |start, t| {
                const a = self._shape.hasTag(t) orelse stdx.debug.panic("dynamicUpdateSlice expects input tensor to have tags used in 'offset_' but {s} is missing (input shape is {f})", .{ t, self._shape });
                offset_values[a] = start.value();
            }
        }

        const op = dialects.stablehlo.dynamic_update_slice(
            mlirCtx(),
            self.value(),
            update.value(),
            offset_values[0..self.rank()],
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());
        return _result(self._shape, op.result(0));
    }

    test dynamicUpdateSlice {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        {
            const x = Tensor.init(.{10}, .f32).withTags(.{.a});
            const y = Tensor.init(.{2}, .f32).withTags(.{.a});
            const ids: Tensor = .init(.{}, .i32);
            const forward = struct {
                pub fn _fwd(x_: Tensor, idx_: struct { a: Tensor }, y_: Tensor) Tensor {
                    return x_.dynamicUpdateSlice(idx_, y_);
                }
            }._fwd;
            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, forward, .{ x, .{ .a = ids }, y }, platform);
            defer exe.deinit();

            var x_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&[10]f32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }));
            defer x_buffer.deinit();
            var y_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, y.shape(), std.mem.sliceAsBytes(&[2]f32{ -1, -1 }));
            defer y_buffer.deinit();
            var ids_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, ids.shape(), std.mem.sliceAsBytes(&[1]i32{4}));
            defer ids_buffer.deinit();

            var res = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, forward, .{ x_buffer, .{ .a = ids_buffer }, y_buffer });
            defer res.deinit();

            try std.testing.expectEqual([10]f32{ 0, 1, 2, 3, -1, -1, 6, 7, 8, 9 }, try res.getValue([10]f32, std.testing.io));
        }

        {
            // Updates 2D, tagged api
            const x = Tensor.init(.{ 2, 5 }, .f32).withTags(.{ .a, .b });
            const y = Tensor.init(.{2}, .f32).withTags(.{.a});
            const ids: Tensor = .init(.{}, .i32);
            const forward = struct {
                pub fn _fwd(x_: Tensor, idx_: Tensor, y_: Tensor) Tensor {
                    return x_.dynamicUpdateSlice(.{ .b = idx_ }, y_);
                }
            }._fwd;
            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, forward, .{ x, ids, y }, platform);
            defer exe.deinit();

            var x_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&[2][5]f32{ .{ 0, 1, 2, 3, 4 }, .{ 5, 6, 7, 8, 9 } }));
            defer x_buffer.deinit();
            var y_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, y.shape(), std.mem.sliceAsBytes(&[2]f32{ -1, -1 }));
            defer y_buffer.deinit();
            var ids_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, ids.shape(), std.mem.sliceAsBytes(&[1]i32{3}));
            defer ids_buffer.deinit();

            var res = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, forward, .{ x_buffer, ids_buffer, y_buffer });
            defer res.deinit();

            try std.testing.expectEqualDeep(
                [2][5]f32{ .{ 0, 1, 2, -1, 4 }, .{ 5, 6, 7, -1, 9 } },
                try res.getValue([2][5]f32, std.testing.io),
            );
        }

        {
            // Updates 2D slice, un-tagged api. Note that `y` needs to have a 1 dimension axis.
            const x: Tensor = .init(.{ 2, 5 }, .f32);
            const y: Tensor = .init(.{ 2, 1 }, .f32);
            const ids: Tensor = .init(.{}, .i32);
            const forward = struct {
                pub fn _fwd(x_: Tensor, idx_: Tensor, y_: Tensor) Tensor {
                    return x_.dynamicUpdateSlice(.{ zml.Tensor.scalar(0, .i32), idx_ }, y_);
                }
            }._fwd;
            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, forward, .{ x, ids, y }, platform);
            defer exe.deinit();

            var x_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&[2][5]f32{ .{ 0, 1, 2, 3, 4 }, .{ 5, 6, 7, 8, 9 } }));
            defer x_buffer.deinit();
            var y_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, y.shape(), std.mem.sliceAsBytes(&[2][1]f32{ .{-1}, .{-1} }));
            defer y_buffer.deinit();
            var ids_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, ids.shape(), std.mem.sliceAsBytes(&[1]i32{3}));
            defer ids_buffer.deinit();

            var res = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, forward, .{ x_buffer, ids_buffer, y_buffer });
            defer res.deinit();

            try std.testing.expectEqualDeep(
                [2][5]f32{ .{ 0, 1, 2, -1, 4 }, .{ 5, 6, 7, -1, 9 } },
                res.getValue([2][5]f32, std.testing.io),
            );
        }

        {
            // Updates 2D, partial update
            const x = Tensor.init(.{ 2, 5 }, .f32).withTags(.{ .a, .b });
            const y = Tensor.init(.{1}, .f32).withTags(.{.a});
            const idx_a: Tensor = .init(.{}, .i32);
            const idx_b: Tensor = .init(.{}, .i32);
            const forward = struct {
                pub fn _fwd(x_: Tensor, idx_: struct { a: Tensor, b: Tensor }, y_: Tensor) Tensor {
                    return x_.dynamicUpdateSlice(idx_, y_);
                }
            }._fwd;
            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, forward, .{ x, .{ .a = idx_a, .b = idx_b }, y }, platform);
            defer exe.deinit();

            var x_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&[2][5]f32{ .{ 0, 1, 2, 3, 4 }, .{ 5, 6, 7, 8, 9 } }));
            defer x_buffer.deinit();
            var y_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, y.shape(), std.mem.sliceAsBytes(&[1]f32{-1}));
            defer y_buffer.deinit();
            var idx_a_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, idx_a.shape(), std.mem.sliceAsBytes(&[1]i32{1}));
            defer idx_a_buffer.deinit();
            var idx_b_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, idx_b.shape(), std.mem.sliceAsBytes(&[1]i32{3}));
            defer idx_b_buffer.deinit();

            var res = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, forward, .{ x_buffer, .{ .a = idx_a_buffer, .b = idx_b_buffer }, y_buffer });
            defer res.deinit();

            try std.testing.expectEqualDeep(
                [2][5]f32{ .{ 0, 1, 2, 3, 4 }, .{ 5, 6, 7, -1, 9 } },
                res.getValue([2][5]f32, std.testing.io),
            );
        }

        {
            // Updates 2D, partial update, un-tagged api.
            const x: Tensor = .init(.{ 2, 5 }, .f32);
            const y: Tensor = .init(.{ 1, 1 }, .f32);
            const idx_a: Tensor = .init(.{}, .i32);
            const idx_b: Tensor = .init(.{}, .i32);
            const forward = struct {
                pub fn _fwd(x_: Tensor, idx_: [2]Tensor, y_: Tensor) Tensor {
                    return x_.dynamicUpdateSlice(&idx_, y_);
                }
            }._fwd;
            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, forward, .{ x, .{ idx_a, idx_b }, y }, platform);
            defer exe.deinit();

            var x_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&[2][5]f32{ .{ 0, 1, 2, 3, 4 }, .{ 5, 6, 7, 8, 9 } }));
            defer x_buffer.deinit();
            var y_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, y.shape(), std.mem.sliceAsBytes(&[1][1]f32{.{-1}}));
            defer y_buffer.deinit();
            var idx_a_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, idx_a.shape(), std.mem.sliceAsBytes(&[1]i32{1}));
            defer idx_a_buffer.deinit();
            var idx_b_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, idx_b.shape(), std.mem.sliceAsBytes(&[1]i32{3}));
            defer idx_b_buffer.deinit();

            var res = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, forward, .{ x_buffer, .{ idx_a_buffer, idx_b_buffer }, y_buffer });
            defer res.deinit();

            try std.testing.expectEqualDeep(
                [2][5]f32{ .{ 0, 1, 2, 3, 4 }, .{ 5, 6, 7, -1, 9 } },
                res.getValue([2][5]f32, std.testing.io),
            );
        }
    }

    /// Returns a Tensor containing the element-wise result of the given 'cmp' comparison between the two input Tensors.
    pub fn cmp(self: Tensor, direction: dialects.stablehlo.ComparisonDirection.Direction, other: Tensor) Tensor {
        stdx.debug.assert(self.dtype() == other.dtype(), "cmp expects input tensors to be of the same type, got {t} and {t}", .{ self.dtype(), other.dtype() });

        if (self.rank() == 0 and other.rank() != 0) return self.broadcast(other._shape, &.{}).cmp(direction, other);
        if (self.rank() != 0 and other.rank() == 0) return self.cmp(direction, other.broadcast(self._shape, &.{}));

        stdx.debug.assert(self._shape.eql(other._shape), "cmp expects input tensor shapes to match, got {f} and {f}", .{ self._shape, other._shape });

        const op = dialects.stablehlo.compare(
            mlirCtx(),
            self.value(),
            other.value(),
            dialects.stablehlo.ComparisonDirection.init(mlirCtx(), direction).getValue(),
            getComparisonType(mlirCtx(), self.dtype()).getValue(),
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());

        return _result(self._shape.withDtype(.bool), op.result(0));
    }

    /// For each vector in the input tensor,
    /// creates a diagonal-matrix where diagonal values are set to the vector values.
    pub fn toDiagonal(self: Tensor, axis_: anytype, new_tags: [2]@EnumLiteral()) Tensor {
        stdx.debug.assert(self.rank() < constants.MAX_RANK - 1, "toDiagonal expects input up to {d} rank, got {f}", .{ constants.MAX_RANK - 1, self });
        const a = self.axis(axis_);
        const d = self.dim(a);
        const p = self.shape()._partitioning.get(a);
        var res_shape = self._shape;
        res_shape._dims.replaceRange(a, 1, &.{ d, d }) catch unreachable;
        res_shape._tags.replaceRange(a, 1, &.{ @tagName(new_tags[0]), @tagName(new_tags[1]) }) catch unreachable;
        // TODO(Corentin): Not sure about that
        res_shape._partitioning.replaceRange(a, 1, &.{ p, p }) catch unreachable;

        const values = self.insertAxes(a + 1, .{new_tags[1]}).broad(res_shape);
        const zeros = Tensor.constant(self.dtype().zero()).broad(res_shape);

        const x = Tensor.iota(res_shape, a);
        const y = Tensor.iota(res_shape, a + 1);
        var res = x.cmp(.EQ, y).select(values, zeros);
        res._shape = res_shape;
        return res;
    }

    test toDiagonal {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        const Local = struct {
            pub fn _toDiag(input: Tensor) Tensor {
                const res = input.toDiagonal(-1, .{ .x, .y });
                std.debug.assert(res.dim(.x) == input.dim(-1));
                std.debug.assert(res.dim(.y) == input.dim(-1));
                return res;
            }
        };

        const x: Tensor = .init(.{ 2, 2 }, .u8);

        var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local._toDiag, .{x}, platform);
        defer exe.deinit();

        var x_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&[2][2]u8{ .{ 1, 2 }, .{ 3, 4 } }));
        defer x_buffer.deinit();

        var res = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Local._toDiag, .{x_buffer});
        defer res.deinit();
        try std.testing.expectEqual(
            [2][2][2]u8{ .{
                .{ 1, 0 },
                .{ 0, 2 },
            }, .{
                .{ 3, 0 },
                .{ 0, 4 },
            } },
            try res.getValue([2][2][2]u8, std.testing.io),
        );
    }

    /// For each matrix specified by the two axes, returns the lower triangular part of it.
    /// The other elements are set to 0.
    /// Usage: `.{ .b = 32, .w = 20, .h = 20 }.triangular(.{ .w, .h}, 0);`
    ///
    /// * if `num_diagonals` is set to 0, the diagonal is not modified.
    /// * if set to -1, the diagonal is set to 0
    /// * if set to n, the n "quasi diagonal" above the diagonal are conserved.
    ///
    /// To get the upper triangular part, swap the order of axes:
    /// `.{ .b = 32, .w = 20, .h = 20 }.triangular(.{ .h, .w }, 0);`
    pub fn triangular(self: Tensor, axes_: anytype, num_diagonals: i32) Tensor {
        stdx.debug.assertComptime(stdx.meta.isTuple(@TypeOf(axes_)) and axes_.len == 2, "triangular expects exactly two axes to work on.", .{});
        const _axes = self.axes(axes_);

        const x = Tensor.iota(self.shape(), _axes.get(0));
        const y = Tensor.iota(self.shape(), _axes.get(1));

        const zeros = Tensor.constant(self.dtype().zero()).broad(self.shape());
        return x.addConstant(num_diagonals).cmp(.GE, y).select(self, zeros);
    }

    test triangular {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        const Local = struct {
            pub fn _tri(input: Tensor, num_diagonals: i32) Tensor {
                return input.triangular(.{ -2, -1 }, num_diagonals);
            }
        };

        const x: Tensor = .init(.{ 3, 3 }, .u8);

        var x_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&[3][3]u8{
            .{ 1, 1, 1 },
            .{ 1, 1, 1 },
            .{ 1, 1, 1 },
        }));
        defer x_buffer.deinit();

        {
            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local._tri, .{ x, 0 }, platform);
            defer exe.deinit();

            var res = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Local._tri, .{x_buffer});
            defer res.deinit();

            try std.testing.expectEqual(
                [3][3]u8{
                    .{ 1, 0, 0 },
                    .{ 1, 1, 0 },
                    .{ 1, 1, 1 },
                },
                try res.getValue([3][3]u8, std.testing.io),
            );
        }
        {
            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local._tri, .{ x, 1 }, platform);
            defer exe.deinit();

            var res = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Local._tri, .{x_buffer});
            defer res.deinit();

            try std.testing.expectEqual(
                [3][3]u8{
                    .{ 1, 1, 0 },
                    .{ 1, 1, 1 },
                    .{ 1, 1, 1 },
                },
                try res.getValue([3][3]u8, std.testing.io),
            );
        }
        {
            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local._tri, .{ x, -1 }, platform);
            defer exe.deinit();

            var res = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Local._tri, .{x_buffer});
            defer res.deinit();

            try std.testing.expectEqual(
                [3][3]u8{
                    .{ 0, 0, 0 },
                    .{ 1, 0, 0 },
                    .{ 1, 1, 0 },
                },
                try res.getValue([3][3]u8, std.testing.io),
            );
        }
    }

    /// For each element at index `i`, if `bool_tensor[i] == true`, `output[i] = on_true[i]`
    /// otherwise, if `bool_tensor[i] == false`, `output[i] = on_false[i]`
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

        const op = dialects.stablehlo.select(
            mlirCtx(),
            bool_tensor.value(),
            on_true.value(),
            on_false.value(),
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());

        return _result(on_true._shape, op.result(0));
    }

    /// Returns a Tensor containing the element-wise not logical operation of the input Tensor.
    pub fn not(self: Tensor) Tensor {
        const op = dialects.stablehlo.not(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, op.result(0));
    }

    /// Returns a Tensor containing boolean indicating if there is a non-zero value over the given axis.
    pub fn any(self: Tensor, axis_: anytype) Tensor {
        const pred = self.cmp(.NE, Tensor.constant(self.dims(), self.dtype().zero()));
        return ops.reduce(
            .{pred},
            .{Tensor.scalar(false, .bool)},
            &.{self.axis(axis_)},
            struct {
                pub fn acc(args: ops.ReduceArgs) Tensor {
                    return args.left.logical(.OR, args.right);
                }
            }.acc,
            .{},
        )[0];
    }

    /// Returns a Tensor containing boolean indicating if there is a non-zero value over the given axis.
    pub fn all(self: Tensor, axis_: anytype) Tensor {
        const pred = if (self.dtype() == .bool) self else self.cmp(.NE, Tensor.scalar(0, self.dtype()));
        return ops.reduce(
            .{pred},
            .{Tensor.scalar(true, .bool)},
            &.{self.axis(axis_)},
            struct {
                pub fn acc(args: ops.ReduceArgs) Tensor {
                    return args.left.logical(.AND, args.right);
                }
            }.acc,
            .{},
        )[0];
    }

    /// Given a set of N vectors of lengths A, B, C, D,
    /// returns N tensors of rank N, and shape (A, B, C, D).
    /// For any coordinate (a, b, c, d), we have:
    ///
    /// - res[0][a, b, c, d] == A[a]
    /// - res[1][a, b, c, d] == B[b]
    /// - res[2][a, b, c, d] == C[c]
    /// - res[3][a, b, c, d] == D[d]
    ///
    /// This is implemented with broadcasting, so typically it won't copy.
    /// In Pytorch/Numpy this is know as `meshgrid` with "ij" mode.
    /// See `zml.torch.meshgrid` for the "xy" mode.
    pub fn cartesianProduct(comptime N: u3, vectors: [N]Tensor) [N]Tensor {
        var out: @TypeOf(vectors) = undefined;
        _cartesianProduct(&vectors, &out);
        return out;
    }

    fn _cartesianProduct(vectors: []const Tensor, out: []Tensor) void {
        stdx.debug.assert(vectors.len >= 1, "cartesianProduct expects at least one input.", .{});
        stdx.debug.assert(vectors.len < constants.MAX_RANK, "cartesianProduct expects at most {} input vectors, received {} !", .{ constants.MAX_RANK - 1, vectors.len });
        for (vectors) |x| {
            stdx.debug.assert(x.rank() <= 1, "cartesianProduct expects 0 or 1 rank input vectors. Got: {any}", .{vectors});
            stdx.debug.assert(vectors[0].dtype() == x.dtype(), "cartesianProduct expects input vectors to have all the same dtype. Got: {any}", .{vectors});
        }

        var res_shape = Shape.init(.{}, vectors[0].dtype());
        for (vectors) |x| {
            if (x.rank() == 0) {
                res_shape = res_shape.appendDim(1, null);
            } else {
                res_shape = res_shape.appendDim(x.dim(0), x.shape().tag(0));
            }
        }

        for (out, vectors, 0..) |*o, x, i| {
            o.* = x.broadcast(res_shape, &[1]i64{@intCast(i)});
        }
    }

    test cartesianProduct {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        const x: Tensor = .init(.{6}, .i32);
        const y: Tensor = .init(.{4}, .i32);

        const Local = struct {
            pub fn _cartesianProduct2(a: Tensor, b: Tensor) [2]Tensor {
                return cartesianProduct(2, .{ a, b });
            }
        };

        var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local._cartesianProduct2, .{ x, y }, platform);
        defer exe.deinit();

        var x_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&[_]i32{ 0, 1, 2, 3, 4, 5 }));
        defer x_buffer.deinit();
        var y_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, y.shape(), std.mem.sliceAsBytes(&[_]i32{ 0, 1, 2, 3 }));
        defer y_buffer.deinit();

        var xs, var ys = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Local._cartesianProduct2, .{ x_buffer, y_buffer });
        defer xs.deinit();
        defer ys.deinit();

        try std.testing.expectEqualSlices(i64, &.{ 6, 4 }, xs.shape().dims());
        try std.testing.expectEqualSlices(i64, &.{ 6, 4 }, ys.shape().dims());
        try std.testing.expectEqualDeep(
            [6][4]i32{
                .{ 0, 0, 0, 0 },
                .{ 1, 1, 1, 1 },
                .{ 2, 2, 2, 2 },
                .{ 3, 3, 3, 3 },
                .{ 4, 4, 4, 4 },
                .{ 5, 5, 5, 5 },
            },
            try xs.getValue([6][4]i32, std.testing.io),
        );
        try std.testing.expectEqualDeep(
            [6][4]i32{
                .{ 0, 1, 2, 3 },
                .{ 0, 1, 2, 3 },
                .{ 0, 1, 2, 3 },
                .{ 0, 1, 2, 3 },
                .{ 0, 1, 2, 3 },
                .{ 0, 1, 2, 3 },
            },
            try ys.getValue([6][4]i32, std.testing.io),
        );
    }

    /// Given a set of N vectors of lengths A, B, C, D,
    /// returns 1 tensors of rank N+1, and shape (A, B, C, D, N).
    /// For any coordinate (a, b, c, d), we have:
    ///
    /// - res[a, b, c, d] == (A[a], B[b], C[c], D[d])
    pub fn cartesianProductStacked(vectors: []const Tensor) Tensor {
        var out = stdx.BoundedArray(Tensor, constants.MAX_RANK).init(vectors.len) catch unreachable;
        _cartesianProduct(vectors, out.slice());

        return Tensor.stack(out.constSlice(), .last, .coord);
    }

    test cartesianProductStacked {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();
        const x: Tensor = .init(.{6}, .i32);
        const y: Tensor = .init(.{4}, .i32);

        const Local = struct {
            pub fn _fwd(a: Tensor, b: Tensor) Tensor {
                return cartesianProductStacked(&.{ a, b });
            }
        };

        var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local._fwd, .{ x, y }, platform);
        defer exe.deinit();

        var x_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&[_]i32{ 0, 1, 2, 3, 4, 5 }));
        defer x_buffer.deinit();
        var y_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, y.shape(), std.mem.sliceAsBytes(&[_]i32{ 0, 1, 2, 3 }));
        defer y_buffer.deinit();

        var z = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Local._fwd, .{ x_buffer, y_buffer });
        defer z.deinit();

        try std.testing.expectEqualDeep(
            [6][4][2]i32{
                .{ .{ 0, 0 }, .{ 0, 1 }, .{ 0, 2 }, .{ 0, 3 } },
                .{ .{ 1, 0 }, .{ 1, 1 }, .{ 1, 2 }, .{ 1, 3 } },
                .{ .{ 2, 0 }, .{ 2, 1 }, .{ 2, 2 }, .{ 2, 3 } },
                .{ .{ 3, 0 }, .{ 3, 1 }, .{ 3, 2 }, .{ 3, 3 } },
                .{ .{ 4, 0 }, .{ 4, 1 }, .{ 4, 2 }, .{ 4, 3 } },
                .{ .{ 5, 0 }, .{ 5, 1 }, .{ 5, 2 }, .{ 5, 3 } },
            },
            try z.getValue([6][4][2]i32, std.testing.io),
        );
    }

    fn binaryOp(
        op_name: []const u8,
        op_fn: fn (*mlir.Context, *const mlir.Value, *const mlir.Value, *const mlir.Location) *mlir.Operation,
    ) fn (Tensor, Tensor) Tensor {
        return struct {
            pub fn binaryOpHelper(self: Tensor, other: Tensor) Tensor {
                stdx.debug.assert(self.dtype() == other.dtype(), "{s} expects tensor to be of same type, got {f} and {f}", .{ op_name, self, other });

                if (self.rank() == 0 and other.rank() != 0) {
                    return binaryOpHelper(self.broad(other._shape), other);
                }

                if (self.rank() != 0 and other.rank() == 0) {
                    return binaryOpHelper(self, other.broad(self._shape));
                }

                var other_ = other;
                var same_shape = self._shape.eql(other._shape);
                if (!same_shape and std.mem.eql(Shape.Tag, self._shape.tags(), other._shape.tags()) and other._shape.canBroadcastTo(self._shape)) {
                    // Only a restrictive version of broadcasting is allowed here, where all the tags matches already.
                    // Typical use case: `x.div(x.sum(.a))`
                    same_shape = true;
                    other_ = other.broad(self._shape);
                }

                stdx.debug.assert(same_shape, "{s} expects tensor shapes to match, got {f} and {f}", .{ op_name, self._shape, other._shape });

                const ret = @call(.auto, op_fn, .{ mlirCtx(), self.value(), other_.value(), mlir.Location.unknown(mlirCtx()) }).appendTo(currentBlock());
                return _result(self._shape, ret.result(0));
            }
        }.binaryOpHelper;
    }

    /// Insert code that will print the content of the given buffer at runtime.
    /// Use the name parameter to differentiate different print calls in the output.
    /// Only for debug purpose, it inserts device to host synchronization
    /// so it will slow down the program execution.
    pub fn print(input: Tensor, name: []const u8) void {
        _ = ops.customCall("zml$print", .{input}, .{}, .{ .name = name }, .{ .has_side_effect = true });
    }

    fn mlirCtx() *mlir.Context {
        return CompilationContext.current().mlir_ctx;
    }

    fn currentBlock() *mlir.Block {
        return CompilationContext.current().currentScope().block;
    }

    /// Returns the donation data of the tensor.
    pub fn donation(self: Tensor) ?usize {
        return CompilationContext.current().currentScope().id_to_donation.get(self.id);
    }

    /// Returns the output memory kind of the tensor.
    pub fn outputMemoryKind(self: Tensor) Memory.Kind {
        return CompilationContext.current().currentScope().id_to_output_memory_kind.get(self.id) orelse .device;
    }
};

fn initPoolArg(rank: usize, data: []const i64) [constants.MAX_RANK]i64 {
    // TODO use shape
    var result = [_]i64{1} ** constants.MAX_RANK;
    const start = rank - data.len;
    @memcpy(result[start .. start + data.len], data);
    return result;
}

fn getPoolResDims(dt: DataType, in_dims: []const i64, base_dilations: @Vector(constants.MAX_RANK, i64), padding: []const i64, window_dimensions: @Vector(constants.MAX_RANK, i64), window_dilations: @Vector(constants.MAX_RANK, i64), window_strides: @Vector(constants.MAX_RANK, i64)) Shape {
    // TODO use shape
    var input_dims = [_]i64{1} ** constants.MAX_RANK;
    @memcpy(input_dims[0..in_dims.len], in_dims);

    const input_dims_: @Vector(constants.MAX_RANK, i64) = input_dims;
    const splat_one: @Vector(constants.MAX_RANK, i64) = @splat(1);
    const dilated_input_shape: @Vector(constants.MAX_RANK, i64) = (input_dims_ - splat_one) * base_dilations + splat_one;
    var pad_slice0: @Vector(constants.MAX_RANK, i64) = @splat(padding[0]);
    var pad_slice1: @Vector(constants.MAX_RANK, i64) = @splat(padding[0]);
    if (padding.len > 1) {
        var idx: usize = 0;
        while (idx < in_dims.len * 2) : (idx += 2) {
            pad_slice0[idx / 2] = padding[idx];
            pad_slice1[idx / 2] = padding[idx + 1];
        }
    }
    const padded_input_shape: @Vector(constants.MAX_RANK, i64) = pad_slice0 + dilated_input_shape + pad_slice1;
    const dilated_window_shape = (window_dimensions - splat_one) * window_dilations + splat_one;
    const dims = @divFloor(padded_input_shape - dilated_window_shape, window_strides) + splat_one;
    const dims_arr: [constants.MAX_RANK]i64 = @bitCast(dims);
    return Shape.init(dims_arr[0..in_dims.len], dt);
}

fn getComparisonType(ctx: *mlir.Context, dtype: DataType) *const dialects.stablehlo.CompareType {
    return dialects.stablehlo.CompareType.init(ctx, switch (dtype.class()) {
        .bool => .UNSIGNED,
        .integer => if (dtype.isSignedInt()) .SIGNED else .UNSIGNED,
        .float => .FLOAT,
        .complex => @panic("Can't compare complex numbers"),
    });
}

test "Tensor.maxPool1d" {
    const zml = @import("zml.zig");
    const platform = zml.testing.env();

    const MaxPool = struct {
        pub fn _fwd(x: zml.Tensor) Tensor.ArgMaxRes {
            return x.maxPool1d(.{
                .window_dimensions = 3,
                .window_strides = 2,
            });
        }
    };

    var data: [20]f32 = undefined;
    for (&data, 0..) |*v, i| v.* = @floatFromInt(i);

    const x: Tensor = .init(.{ 2, 2, 5 }, .f32);

    var exe = try zml.module.compile(std.testing.allocator, std.testing.io, MaxPool._fwd, .{x}, platform);
    defer exe.deinit();

    var x_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&data));
    defer x_buffer.deinit();

    var result = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, MaxPool._fwd, .{x_buffer});
    defer result.values.deinit();
    defer result.indices.deinit();

    try zml.testing.expectEqualShapes(.init(.{ 2, 2, 2 }, .f32), result.values.shape());
    try zml.testing.expectEqualShapes(.init(.{ 2, 2, 2 }, .i32), result.indices.shape());
    try std.testing.expectEqualDeep(
        [2][2][2]f32{
            .{ .{ 2, 4 }, .{ 7, 9 } },
            .{ .{ 12, 14 }, .{ 17, 19 } },
        },
        result.values.getValue([2][2][2]f32, std.testing.io),
    );
}

test "Tensor.maxPool2d" {
    const zml = @import("zml.zig");
    const platform = zml.testing.env();

    const MaxPool = struct {
        pub fn _fwd(x: Tensor) Tensor.ArgMaxRes {
            return x.maxPool2d(.{
                .window_dimensions = .{ 3, 2 },
                .window_strides = .{ 2, 1 },
            });
        }
    };

    var data: [100]f32 = undefined;
    for (&data, 0..) |*v, i| v.* = @floatFromInt(i);

    const x: Tensor = .init(.{ 2, 2, 5, 5 }, .f32);

    var exe = try zml.module.compile(std.testing.allocator, std.testing.io, MaxPool._fwd, .{x}, platform);
    defer exe.deinit();

    var x_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&data));
    defer x_buffer.deinit();

    var result = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, MaxPool._fwd, .{x_buffer});
    defer result.values.deinit();
    defer result.indices.deinit();

    try zml.testing.expectEqualShapes(Shape.init(.{ 2, 2, 2, 4 }, .f32), result.values.shape());
    try zml.testing.expectEqualShapes(Shape.init(.{ 2, 2, 2, 4 }, .i32), result.indices.shape());
    var buffer: [2][2][2][4]f32 = undefined;
    _ = try result.values.toSlice(std.testing.io, .init(result.values.shape(), std.mem.asBytes(&buffer)));
    try std.testing.expectEqualDeep(
        [2][2][2][4]f32{
            .{
                .{ .{ 11, 12, 13, 14 }, .{ 21, 22, 23, 24 } },
                .{ .{ 36, 37, 38, 39 }, .{ 46, 47, 48, 49 } },
            },
            .{
                .{ .{ 61, 62, 63, 64 }, .{ 71, 72, 73, 74 } },
                .{ .{ 86, 87, 88, 89 }, .{ 96, 97, 98, 99 } },
            },
        },
        buffer,
    );
}

fn _parseGatherCoord(self: Tensor, axes_: anytype) struct { bool, stdx.BoundedArray(u3, constants.MAX_RANK) } {
    const AxesT = @TypeOf(axes_);
    const axes_is_scalar = AxesT == @EnumLiteral() or AxesT == comptime_int or @typeInfo(AxesT) == .int;

    const coord_axes = if (axes_is_scalar)
        stdx.BoundedArray(u3, constants.MAX_RANK).fromSlice(&.{self.axis(axes_)}) catch unreachable
    else
        self.axes(axes_);

    return .{ axes_is_scalar, coord_axes };
}

fn toI64(values: anytype) stdx.BoundedArray(i64, constants.MAX_RANK) {
    var res: stdx.BoundedArray(i64, constants.MAX_RANK) = .{};
    for (values) |val| res.appendAssumeCapacity(@intCast(val));
    return res;
}

fn transposeIsJustAReshape(x: Shape, permutation: []const i64) bool {
    var perm: stdx.BoundedArray(struct { u8, bool }, constants.MAX_RANK) = .{};
    // Don't rewrite on invalid inputs.
    if (permutation.len > x.rank()) return false;
    for (permutation) |ax| {
        const squeezable = x.dim(ax) == 1;
        perm.appendAssumeCapacity(.{ @intCast(ax), squeezable });
    }

    var effective_ax: u8 = 0;
    for (0..perm.len) |i| {
        const ax, const squeezable = perm.get(i);
        if (squeezable) {
            // Effectively squeeze this axis by decrementing axes coming after by 1.
            for (i..perm.len) |j| {
                if (perm.buffer[j][0] > ax) {
                    perm.buffer[j][0] -= 1;
                }
            }
            continue;
        }

        if (ax != effective_ax) return false;
        effective_ax += 1;
    }

    return true;
}

test transposeIsJustAReshape {
    try std.testing.expect(transposeIsJustAReshape(Shape.init(.{ 5, 1, 3 }, .i32), &.{ 0, 1, 2 }));
    try std.testing.expect(transposeIsJustAReshape(Shape.init(.{ 5, 1, 3 }, .i32), &.{ 1, 0, 2 }));
    try std.testing.expect(!transposeIsJustAReshape(Shape.init(.{ 5, 1, 3 }, .i32), &.{ 2, 1, 0 }));
    try std.testing.expect(transposeIsJustAReshape(Shape.init(.{ 64, 8, 1, 128 }, .bf16), &.{ 0, 2, 1, 3 }));
    try std.testing.expect(!transposeIsJustAReshape(Shape.init(.{ 64, 8, 155, 128 }, .bf16), &.{ 0, 2, 1, 3 }));
    try std.testing.expect(transposeIsJustAReshape(Shape.init(.{ 64, 1, 1, 128 }, .bf16), &.{ 1, 2, 0, 3 }));
    try std.testing.expect(!transposeIsJustAReshape(Shape.init(.{ .b = 1, .h = 10, .q = 155, .hd = 1 }, .f32), &.{ 0, 2, 1, 3 }));
    try std.testing.expect(!transposeIsJustAReshape(Shape.init(.{ 1, 10, 155, 1 }, .f32), &.{ 0, 2, 3, 1 }));
    try std.testing.expect(transposeIsJustAReshape(Shape.init(.{ 1, 10, 155, 1 }, .f32), &.{ 0, 1, 3, 2 }));
}

test "unused tensor" {
    const zml = @import("zml.zig");
    const platform = zml.testing.env();

    const Local = struct {
        pub fn _fwd(x: Tensor) Tensor {
            const y = x.addConstant(1);
            _ = y;
            return x;
        }
    };

    var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local._fwd, .{Tensor.init(.{10}, .f32)}, platform);
    defer exe.deinit();
}
