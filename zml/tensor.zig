const std = @import("std");
const builtin = @import("builtin");

const mlir = @import("mlir");
const stdx = @import("stdx");

const Buffer = @import("buffer.zig").Buffer;
const CompilationContext = @import("module.zig").CompilationContext;
const Data = @import("dtype.zig").Data;
const DataType = @import("dtype.zig").DataType;
const HostBuffer = @import("hostbuffer.zig").HostBuffer;
const Memory = @import("buffer.zig").Buffer.Memory;
const meta = @import("meta.zig");
const mlirx = @import("mlirx.zig");
const ops = @import("ops.zig");
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;

const EnumLiteral = @TypeOf(.enum_literal);

const dialect = struct {
    const stablehlo = @import("mlir/dialects").stablehlo;
};

const scoped_log = std.log.scoped(.@"zml/tensor");

test {
    std.testing.refAllDecls(Tensor);
}

/// Represents an abstract Tensor object, which can be the input,
/// output, weight or activations of a neural network.
/// Tensor are abstract in the sense they only represent a computation,
/// but not a specific memory buffer.
/// Tensor namespace contains most of linear algebra needed to
/// represent mathematical operations.
/// More operations are available in `zml.nn` and `zml.torch` namespaces.
pub const Tensor = struct {
    _shape: Shape,
    _id: _Id,
    _donation: _Donation = .no_buffer,
    _output_memory_kind: Memory = .device,

    pub const _Donation = union(enum) { no_buffer, input_buffer, arg: u16 };
    pub const _Id = union(enum) { mlir: mlir.Value, buffer_id: u64, arg_id: u64 };
    pub const MAX_RANK = Shape.MAX_RANK;

    /// Returns the current compilation context.
    pub fn getContext(self: Tensor) *CompilationContext {
        _ = self;
        return CompilationContext.current();
    }

    pub fn format(
        self: Tensor,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = options;
        const bare_fmt = fmt.len == 1 and fmt[0] == '_';
        try writer.print(if (bare_fmt) "{_}" else "Tensor({_})", .{self._shape});
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
    pub inline fn rank(self: Tensor) u4 {
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
    pub fn _result(sh: Shape, val: mlir.Value) Tensor {
        const res: Tensor = .{
            ._shape = sh,
            ._id = .{ .mlir = val },
        };

        if (builtin.mode == .Debug) {
            // Check that the MLIR value actually have the same shape.
            const other = fromMlirValue(val);
            stdx.debug.internalAssert(sh.eql(other._shape), "Created a {} from Mlir value but expected {}", .{ other._shape, res._shape });
        }

        return res;
    }

    /// Creates a Tensor from a mlir.Value
    ///
    /// The shape is derived from the type of the mlir.Value.
    pub fn fromMlirValue(val: mlir.Value) Tensor {
        const ranked_tensor = val.getType().as(mlir.RankedTensorType).?;
        const n = ranked_tensor.getRank();

        stdx.debug.assert(n <= MAX_RANK, "Can't represent MLIR tensor of rank {}, max supported rank is {}.", .{ n, MAX_RANK });

        var sh: Shape = .{ ._dtype = mlirx.Type.toDType(ranked_tensor.getElementType()) };
        for (0..n) |i| {
            sh._dims.appendAssumeCapacity(ranked_tensor.getDimension(i));
        }
        sh._tags.resize(n) catch unreachable;

        return .{ ._shape = sh, ._id = .{ .mlir = val } };
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
    pub fn axes(self: Tensor, axes_: anytype) std.BoundedArray(u3, Tensor.MAX_RANK) {
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

    pub fn withSharding(self: Tensor, axes_: anytype) Tensor {
        return switch (self._id) {
            .arg_id, .mlir => {
                const ctx = self.getContext();
                const mlir_ctx = ctx.mlirCtx();
                var res = self;
                res._shape = self._shape.withSharding(axes_);

                if (ctx.numPartitions() <= 1) return self;
                const op = dialect.stablehlo.custom_call(
                    mlir_ctx,
                    &.{self.value()},
                    .{
                        .call_target_name = "Sharding",
                        .has_side_effect = false,
                        .backend_config = null,
                        .additional_attributes = &.{.{ "mhlo.sharding", ctx.getShardingAttr(res._shape) }},
                        .api_version = .original,
                    },
                    &.{self.value().getType()},
                    mlir_ctx.location(@src()),
                );

                return _result(res._shape, op.result(0));
            },
            .buffer_id => {
                var res = self;
                res._shape = self._shape.withSharding(axes_);
                return res;
            },
        };
    }

    pub fn toMemory(self: Tensor, kind: Memory) Tensor {
        return switch (self._id) {
            .arg_id, .mlir => {
                const ctx = self.getContext();
                const mlir_ctx = ctx.mlirCtx();
                if (ctx.target() == .cpu) return self;

                const memory_kind = @tagName(kind.toPjrtMemory());

                const frontend_attributes = mlir.Attribute.dict(mlir_ctx, &.{
                    .{ "_xla_buffer_placement", .string(mlir_ctx, memory_kind) },
                });

                const op = dialect.stablehlo.custom_call(mlir_ctx, &.{self.value()}, .{
                    .call_target_name = "annotate_device_placement",
                    .has_side_effect = true,
                    .backend_config = null,
                    .additional_attributes = &.{.{ "mhlo.frontend_attributes", frontend_attributes }},
                    .api_version = .original,
                }, &.{self.value().getType()}, mlir_ctx.location(@src()));

                var res = _result(self._shape, op.result(0));
                res._output_memory_kind = kind;
                return res;
            },
            .buffer_id => {
                var res = self;
                res._output_memory_kind = kind;
                return res;
            },
        };
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

    /// Returns the mlir.Value associated with the Tensor.
    ///
    /// This will fail if used outside of a compilation context.
    pub fn value(self: Tensor) mlir.Value {
        return self.getContext().getValueAndDonation(self)[0];
    }

    /// Tell PJRT compiler that memory should be reuse between the two tensors.
    /// The compiler is already aggressively reusing tensors for intermediate results,
    /// but this API allows to reuse buffer between input and output arguments
    /// of a given function.
    /// Note this is visible from the outside. The caller of a function with donations
    /// is not allowed to reuse the donated input buffer after the call.
    /// For `reuseBuffer` to be effective, it needs to propagate all the way through the output.
    pub fn reuseBuffer(self: Tensor, origin: Tensor) Tensor {
        // Note: check donation docs, this may be too permissive.
        stdx.debug.assert(self.byteSize() == origin.byteSize(), "Can't reuse buffers between tensors of different size: {} and {}", .{ self, origin });

        // TODO: should we store all donations inside the context ?
        var res = self;
        res._donation = self.getContext().getValueAndDonation(origin)[1];

        return res;
    }

    var _global_tensor_counter: u64 = 0;

    /// Internal use
    pub fn _reserveIdRange(len: u32) u64 {
        return @atomicRmw(u64, &_global_tensor_counter, .Add, len, .seq_cst);
    }

    /// Internal use
    pub fn setUniqueId(self: *Tensor) void {
        self._id = .{ .buffer_id = _reserveIdRange(1) };
    }

    /// Returns a Tensor containing the absolute value of each element of the input Tensor.
    pub fn abs(self: Tensor) Tensor {
        const loc = self.getContext().mlirCtx().location(@src());
        const op = dialect.stablehlo.abs(self.getContext().mlirCtx(), self.value(), loc);
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

        const loc = self.getContext().location(@src(), "bitCast({s})", .{@tagName(dt)});
        const op = dialect.stablehlo.bitcast_convert(
            self.getContext().mlirCtx(),
            self.value(),
            mlirx.tensorType(self.getContext().mlirCtx(), res_shape),
            loc,
        );

        return _result(res_shape, op.result(0));
    }

    /// Returns a Tensor containing the element-wise number of leading 0 bits in the input Tensor.
    pub fn countLeadingZeros(self: Tensor) Tensor {
        const loc = self.getContext().mlirCtx().location(@src());
        const op = dialect.stablehlo.count_leading_zeros(self.getContext().mlirCtx(), self.value(), loc);
        return _result(self._shape, op.result(0));
    }

    /// Returns a Tensor containing booleans indicating if each element of the input Tensor is finite.
    pub fn isFinite(self: Tensor) Tensor {
        const loc = self.getContext().mlirCtx().location(@src());
        const op = dialect.stablehlo.is_finite(self.getContext().mlirCtx(), self.value(), loc);
        return _result(self._shape.withDtype(.bool), op.result(0));
    }

    /// Returns a Tensor containing the element-wise number of bits set in the input Tensor.
    pub fn popcnt(self: Tensor) Tensor {
        stdx.debug.assert(self.dtype().isInteger(), "popcnt expects tensor type to be an integer, got {}", .{self.dtype()});
        const loc = self.getContext().mlirCtx().location(@src());
        const op = dialect.stablehlo.popcnt(self.getContext().mlirCtx(), self.value(), loc);
        return _result(self._shape, op.result(0));
    }

    /// Returns a Tensor containing the sign of the input Tensor element-wise.
    pub fn sign(self: Tensor) Tensor {
        const loc = self.getContext().mlirCtx().location(@src());
        const op = dialect.stablehlo.sign(self.getContext().mlirCtx(), self.value(), loc);
        return _result(self._shape, op.result(0));
    }

    /// Returns a Tensor containing the element-wise remainder of dividend 'self' and divisor 'other'.
    ///
    /// See https://pytorch.org/docs/stable/generated/torch.fmod.html for more details.
    pub fn fmod(self: Tensor, divisor: f32) Tensor {
        return self.remainder(Tensor.scalar(divisor, .f32).broadcast(self._shape, &.{}));
    }

    test fmod {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        const inputs: [2][6]f32 = .{ .{ -3.0, -2, -1, 1, 2, 3 }, .{ 1, 2, 3, 4, 5, -5 } };
        const expectations: [2][6]f32 = .{ .{ -1.0, -0.0, -1.0, 1.0, 0.0, 1.0 }, .{ 1.0000, 0.5000, 0.0000, 1.0000, 0.5000, -0.5000 } };
        const divisors: [2]f32 = .{ 2, -1.5 };

        inline for (inputs, expectations, divisors) |i, e, d| {
            const input = try zml.Buffer.fromSlice(platform, .{6}, &i);
            const output = try zml.testing.compileAndCall(platform, Tensor.fmod, .{ input, d });

            try zml.testing.expectClose(zml.HostBuffer.fromSlice(.{6}, &e), output, 1e-4);
        }
    }

    /// Returns a Tensor containing the element-wise left-shift operation of 'self' by 'other'.
    pub fn shiftLeft(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "shiftLeft", dialect.stablehlo.shift_left)(self, other);
    }

    /// Returns a Tensor containing the element-wise arithmetic right-shift operation of 'self' by 'other'.
    pub fn shiftRightArithmetic(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "shiftRightArithmetic", dialect.stablehlo.shift_right_arithmetic)(self, other);
    }

    /// Returns a Tensor containing the element-wise logical right-shift operation of 'self' by 'other'.
    pub fn shiftRightLogical(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "shiftRightLogical", dialect.stablehlo.shift_right_logical)(self, other);
    }

    /// Returns the Cholesky decomposition of the input Tensor.
    ///
    /// 'lower' controls the form of the output Tensor. The output will be lower-triangular if 'lower' is true
    /// and upper-triangular otherwise.
    pub fn cholesky(self: Tensor, lower: bool) Tensor {
        stdx.debug.assert(self.rank() <= 2, "cholesky expects tensor rank to be <= 2, got {}", .{self.rank()});

        const loc = self.getContext().location(@src(), "lower={}", .{lower});
        const op = dialect.stablehlo.cholesky(self.getContext().mlirCtx(), self.value(), lower, loc);
        return _result(self._shape, op.result(0));
    }

    /// Solves the system of linear equations formed by the input tensors.
    pub fn triangularSolve(self: Tensor, other: Tensor, opts: dialect.stablehlo.TriangularSolveOpts) Tensor {
        stdx.debug.assert(self.dtype() == other.dtype(), "triangularSolve expects tensors to be of the same type, got {} and {}", .{ self.dtype(), other.dtype() });
        stdx.debug.assert(self.rank() <= 2 and self.rank() == other.rank(), "triangularSolve expects tensors to have the same rank and be <= 2, got {} and {}", .{ self.rank(), other.rank() });

        const loc = self.getContext().location(@src(), "triangularSolve({_}, {})", .{ self, opts });
        const op = dialect.stablehlo.triangular_solve(self.getContext().mlirCtx(), self.value(), other.value(), loc, opts);
        return _result(self._shape, op.result(0));
    }

    /// Returns a Tensor containing the element-wise rounding towards the nearest integer, breaking ties away from zero, of the input Tensor.
    pub fn roundNearestAfz(self: Tensor) Tensor {
        stdx.debug.assert(self.dtype().isFloat(), "roundNearestAfz expects tensor type to be a float, got {}", .{self.dtype()});

        const loc = self.getContext().mlirCtx().location(@src());
        const op = dialect.stablehlo.round_nearest_afz(self.getContext().mlirCtx(), self.value(), loc);
        return _result(self._shape, op.result(0));
    }

    /// Returns a Tensor containing the element-wise rounding towards the nearest integer, breaking ties towards the even integer, of the input Tensor.
    pub fn roundNearestEven(self: Tensor) Tensor {
        stdx.debug.assert(self.dtype().isFloat(), "roundNearestEven expects tensor type to be a float, got {}", .{self.dtype()});

        const loc = self.getContext().mlirCtx().location(@src());
        const op = dialect.stablehlo.round_nearest_even(self.getContext().mlirCtx(), self.value(), loc);
        return _result(self._shape, op.result(0));
    }

    /// Returns a Tensor of complex number converted from a pair of real and imaginary Tensors.
    pub fn complex(re: Tensor, im: Tensor) Tensor {
        stdx.debug.assert(re._shape.eql(im._shape), "complex expects tensor shapes to match, got {} and {}", .{ re._shape, im._shape });
        stdx.debug.assert(re.dtype() == .f32 or re.dtype() == .f64, "complex expects tensors type to be f32 or f64, got {}", .{re.dtype()});

        const loc = re.getContext().mlirCtx().location(@src());
        const op = dialect.stablehlo.complex(re.getContext().mlirCtx(), re.value(), im.value(), loc);
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
        const loc = self.getContext().mlirCtx().location(@src());
        const op = dialect.stablehlo.real(self.getContext().mlirCtx(), self.value(), loc);
        return _result(self._shape.withDtype(dt), op.result(0));
    }

    /// Returns a Tensor containing the element-wise imaginary part of the input Tensor.
    ///
    /// Tensor type can float or complex.
    pub fn imag(self: Tensor) Tensor {
        stdx.debug.assert(self.dtype().isFloat() or self.dtype().isComplex(), "imag expects tensor type to be a float or a complex, got {}", .{self.dtype()});

        // Real tensors don't have imaginary part.
        if (self.dtype().isFloat()) {
            return Tensor.constant(self._shape, self.dtype().zero());
        }

        const dt: DataType = switch (self.dtype()) {
            .bf16, .f16, .f32, .f64 => self.dtype(),
            .c64 => .f32,
            .c128 => .f64,
            else => unreachable,
        };
        const loc = self.getContext().mlirCtx().location(@src());
        const op = dialect.stablehlo.imag(self.getContext().mlirCtx(), self.value(), loc);
        return _result(self._shape.withDtype(dt), op.result(0));
    }

    /// Returns the Fast Fourier Transform (FFT) of the input Tensor.
    pub fn fft(self: Tensor, opts: dialect.stablehlo.FftOpts) Tensor {
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

        const loc = self.getContext().location(@src(), "fft({_},{})", .{ self, opts });
        const op = dialect.stablehlo.fft(self.getContext().mlirCtx(), self.value(), loc, opts);
        return _result(sh, op.result(0));
    }

    pub const Rng = struct {
        _state: Tensor = .{ ._shape = .init(.{2}, .u64), ._id = .{ .buffer_id = 0 } },
        algorithm: dialect.stablehlo.RngAlgorithm.Type = .DEFAULT,

        pub fn shape() ShapeOf(Rng) {
            return .{
                ._state = Shape.init(.{2}, .u64),
            };
        }

        pub fn init(platform: Platform, seed: u128) !Bufferized(Rng) {
            return .{
                ._state = try Buffer.fromBytes(platform, Rng.shape()._state, std.mem.asBytes(&seed)),
            };
        }

        /// Returns a Tensor of the given shape, filled with uniform random bits, and a new Rng state.
        ///
        /// The given Rng state should not be used anymore (or you'll get the same numbers again).
        /// The output is guaranteed to be deterministic function of `self` Rng state,
        /// but it is not guaranteed to be deterministic between implementations.
        pub fn bitGenerator(self: Rng, sh: Shape) struct { Rng, Tensor } {
            const ctx = CompilationContext.current();
            const loc = ctx.location(@src(), "rand.bitGen({_})", .{sh});
            const op = dialect.stablehlo.rng_bit_generator(
                ctx.mlirCtx(),
                self.algorithm,
                self._state.value(),
                mlirx.tensorType(ctx.mlirCtx(), self._state._shape),
                mlirx.tensorType(ctx.mlirCtx(), sh),
                loc,
            );
            return .{ self.update(op.result(0)), _result(sh, op.result(1)) };
        }

        fn update(self: Rng, new_state: mlir.Value) Rng {
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
                else => stdx.debug.panic("uniform don't support non-byte aligned dtype. Got: {}", .{shape_}),
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
            const rand, const stats = try zml.testing.compileAndCallWithTensors(
                platform,
                Stats.uniformStats,
                .{ Rng.shape(), zml.Shape.init(.{1024}, .f32), .{ .min = -2, .max = 10 } },
                .{try Rng.init(platform, 1234)},
            );

            // Check the Rng state has been modified.
            try std.testing.expect(try rand._state.getValue(u128) != 1234);

            // Check the mean and variance are close to theoritical values.
            const mean_ = try stats.mean.getValue(f32);
            try std.testing.expectApproxEqAbs(4, mean_, 0.03);

            const variance = try stats.variance.getValue(f32);
            try std.testing.expectApproxEqAbs(12.0 * 12.0 / 12.0, variance, 0.01);

            // Check that no value is outside of the interval
            // and we have samples close to the edges.
            const min_ = try stats.min.getValue(f32);
            try std.testing.expect(min_ >= -2);
            try std.testing.expectApproxEqAbs(-2, min_, 0.05);

            const max_ = try stats.max.getValue(f32);
            try std.testing.expect(max_ < 10);
            try std.testing.expectApproxEqAbs(10, max_, 0.05);
        }

        /// Returns a Tensor of the given shape, filled with floating point numbers sampled from a normal distribution.
        ///
        /// Note: this uses stablehlo.rng which is deprecated.
        /// https://github.com/openxla/stablehlo/blob/main/rfcs/20240503-opset-deprecations.md
        pub fn normal(sh: Shape, opts: struct { mean: f64 = 0, stddev: f64 = 1 }) Tensor {
            stdx.debug.assert(sh.dtype().isFloat(), "normal expects tensor type to be a float, got {}", .{sh.dtype()});

            const ctx = CompilationContext.current();
            const loc = ctx.location(@src(), "rand.normal({_}, mean={},stddev={})", .{ sh, opts.mean, opts.stddev });
            const a = Tensor.constant(.{}, Data.init(sh.dtype(), opts.mean));
            const b = Tensor.constant(.{}, Data.init(sh.dtype(), opts.stddev));
            const res_shape = Tensor.constantTensor(HostBuffer.fromSlice(.{sh.rank()}, sh.dims()));
            const op = dialect.stablehlo.rng(ctx.mlirCtx(), a.value(), b.value(), res_shape.value(), .NORMAL, loc);
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
                    const flat = data.flattenAll();
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
                    const values = Tensor.constantTensor(HostBuffer.fromArray(&powers)).withTags(.{.d});
                    const counts = values.gatherValues(.d, samples, .{}).sum(.n).bitCast(.u16);
                    const actual_dist = counts.reshape(target_dist.shape()).convert(target_dist.dtype()).divByConst(s.dim(.n));
                    return .{ rng, .{ .mean = mean_, .variance = variance, .actual_dist = actual_dist } };
                }
            };

            const platform = zml.testing.env();
            const tgt_dist = [_]f32{ 2.0, 1.0, 4.0, 3.0 };
            const rand, const stats = try zml.testing.compileAndCallWithTensors(
                platform,
                Stats.gumbelStats,
                .{ Rng.shape(), zml.Shape.init(.{tgt_dist.len}, .f32) },
                .{ try Rng.init(platform, 1234), try .fromArray(platform, tgt_dist) },
            );
            // Check the Rng state has been modified.
            try std.testing.expect(try rand._state.getValue(i128) != 1234);

            // Check the mean and variance are close to theoritical values.
            const mean_ = try stats.mean.getValue(f32);
            try std.testing.expectApproxEqAbs(0.5772, mean_, 0.02);

            const variance = try stats.variance.getValue(f32);
            const pi = std.math.pi;
            try std.testing.expectApproxEqAbs(pi * pi / 6.0, variance, 0.03);

            // Check the distribution obtained with the gumbel trick matches the target distribution.
            const actual_dist = try stats.actual_dist.getValue([4]f32);
            scoped_log.debug("tgt_dist: {d}, actual_dist: {d}", .{ tgt_dist, actual_dist });
            for (tgt_dist, actual_dist) |tgt, actual| {
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

        const loc = self.getContext().location(@src(), "reducePrecision(exponent_bits={}, mantissa_bits={})", .{ exponent_bits, mantissa_bits });
        const op = dialect.stablehlo.reduce_precision(self.getContext().mlirCtx(), self.value(), exponent_bits, mantissa_bits, loc);
        return _result(self._shape, op.result(0));
    }

    inline fn convolution(self: Tensor, other: Tensor, opts: dialect.stablehlo.ConvolutionOpts, loc: mlir.Location) Tensor {
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
        const op = dialect.stablehlo.convolution(
            self.getContext().mlirCtx(),
            self.value(),
            other.value(),
            used_opts,
            mlirx.tensorType(self.getContext().mlirCtx(), new_shape),
            loc,
        );

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
        const loc = input.getContext().location(@src(), "opts={}", .{opts});
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
        }, loc);
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
        const loc = input.getContext().location(@src(), "opts={}", .{opts});
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
        }, loc);
    }

    /// Returns a Tensor containing the element-wise addition of the input Tensors.
    pub fn add(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "add", dialect.stablehlo.add)(self, other);
    }

    /// Returns a Tensor containing the element-wise subtraction of the input Tensors.
    pub fn sub(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "subtract", dialect.stablehlo.subtract)(self, other);
    }

    /// Returns a Tensor containing the element-wise multiplication of the input Tensors.
    pub fn mul(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "mul", dialect.stablehlo.multiply)(self, other);
    }

    /// Returns a Tensor containing the element-wise division of the input Tensors.
    pub fn div(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "div", dialect.stablehlo.divide)(self, other);
    }

    /// Returns a Tensor containing the element-wise exponentiation of the input Tensors.
    pub fn pow(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "pow", dialect.stablehlo.power)(self, other);
    }

    /// Returns a Tensor containing the element-wise maximum operation of the input Tensors.
    pub fn maximum(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "maximum", dialect.stablehlo.maximum)(self, other);
    }

    /// Returns a Tensor containing the element-wise minimum operation of the input Tensors.
    pub fn minimum(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "minimum", dialect.stablehlo.minimum)(self, other);
    }

    /// Returns a Tensor containing the element-wise remainder of dividend 'self' and divisor 'other'.
    pub fn remainder(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "remainder", dialect.stablehlo.remainder)(self, other);
    }

    /// Returns a Tensor containing the element-wise addition of the input Tensor with a constant.
    pub fn addConstant(self: Tensor, b: anytype) Tensor {
        return self.add(Tensor.scalar(b, self.dtype()));
    }

    /// Returns a Tensor containing the element-wise division of the input Tensor by a constant.
    pub fn divByConst(self: Tensor, b: anytype) Tensor {
        return self.div(Tensor.scalar(b, self.dtype()));
    }

    /// Returns a Tensor containing the element-wise multiplication of the input Tensor by a constant.
    pub inline fn scale(self: Tensor, val: anytype) Tensor {
        return self.mul(Tensor.scalar(val, self.dtype()));
    }

    pub const LogicalOp = enum { OR, XOR, AND };

    /// Returns a Tensor containing the element-wise logical operation of the input Tensors.
    pub fn logical(self: Tensor, comptime logical_op: LogicalOp, other: Tensor) Tensor {
        return switch (logical_op) {
            .OR => binaryOp(@src(), "or", dialect.stablehlo.or_)(self, other),
            .XOR => binaryOp(@src(), "xor", dialect.stablehlo.xor)(self, other),
            .AND => binaryOp(@src(), "and", dialect.stablehlo.and_)(self, other),
        };
    }

    /// Returns a Tensor containing the element-wise floor operation of the input Tensor.
    pub fn floor(self: Tensor) Tensor {
        const loc = self.getContext().mlirCtx().location(@src());
        return _result(self._shape, dialect.stablehlo.floor(self.getContext().mlirCtx(), self.value(), loc).result(0));
    }

    /// Returns a Tensor containing the element-wise ceil operation of the input Tensor.
    pub fn ceil(self: Tensor) Tensor {
        const loc = self.getContext().mlirCtx().location(@src());
        return _result(self._shape, dialect.stablehlo.ceil(self.getContext().mlirCtx(), self.value(), loc).result(0));
    }

    /// Returns a Tensor containing the element-wise conversion to another type.
    pub fn convert(self: Tensor, to: DataType) Tensor {
        if (to == self.dtype()) {
            return self;
        }
        const loc = self.getContext().location(@src(), "convert({_},to={s})", .{ self, @tagName(to) });

        const mlir_ctx = self.getContext().mlirCtx();
        const res_type = mlirx.tensorType(mlir_ctx, self.shape().withDtype(to));
        const op = dialect.stablehlo.convert(mlir_ctx, self.value(), res_type, loc);
        return _result(self._shape.withDtype(to), op.result(0));
    }

    /// Returns a Tensor containing the element-wise rounding operation of the input Tensor.
    pub fn round(self: Tensor) Tensor {
        const loc = self.getContext().mlirCtx().location(@src());
        const sine_op = dialect.stablehlo.round_nearest_even(self.getContext().mlirCtx(), self.value(), loc);
        return _result(self._shape, sine_op.result(0));
    }

    /// Returns a Tensor containing the element-wise clamping operation of the input Tensor.
    pub fn clamp(self: Tensor, min_: Tensor, max_: Tensor) Tensor {
        const loc = self.getContext().mlirCtx().location(@src());
        const op = dialect.stablehlo.clamp(self.getContext().mlirCtx(), min_.value(), self.value(), max_.value(), loc);
        return _result(self._shape, op.result(0));
    }

    /// See torch.matmul
    pub fn matmul(lhs: Tensor, rhs: Tensor) Tensor {
        return @import("torch.zig").matmul(lhs, rhs);
    }

    /// Matrix multiplication, where contracting axes are specified using their tags.
    /// eg dot(.{ .a, .b, .c }, .{ .a, .c, .d }, .{ .c }) -> .{ .a, .c, .d }
    /// Axes with the same tag on both sides, and which aren't contracting,
    /// are considered "batching axes".
    pub fn dot(lhs: Tensor, rhs: Tensor, comptime contracting: anytype) Tensor {
        var contracting_axes: [contracting.len][2]i8 = undefined;
        inline for (contracting, 0..) |c, i| {
            contracting_axes[i] = .{ lhs.axis(c), rhs.axis(c) };
        }

        var batching_axes: [MAX_RANK][2]i8 = undefined;
        var n_batching: u8 = 0;
        for (lhs._shape.tags(), 0..) |l, li| {
            stdx.debug.assert(l != Shape.TagUnknown, "Can't use `dot(..., {any})` on {any}, it need to be explictily tagged.", .{ contracting, lhs });

            for (rhs._shape.tags(), 0..) |r, ri| {
                stdx.debug.assert(r != Shape.TagUnknown, "Can't use `dot(..., {any})` on {any}, it need to be explictily tagged.", .{ contracting, rhs });

                if (l == r) {
                    for (contracting_axes) |ct| {
                        if (l == lhs._shape.tag(ct[0])) {
                            break;
                        }
                    } else {
                        // tag is both in lhs and rhs but not in contracting -> it's a batching dim.
                        batching_axes[n_batching] = .{ @intCast(li), @intCast(ri) };
                        n_batching += 1;
                    }
                }
            }
        }

        return dotGeneral(lhs, rhs, contracting_axes[0..], batching_axes[0..n_batching]);
    }

    test dot {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        var comp = try zml.module.CompilationContext.init(std.testing.allocator, "test", platform);
        defer comp.deinit();

        comp.activate();
        defer comp.deactivate();

        inline for (.{
            .{ .{ .c = 20 }, .{ .c = 20 }, .{.c}, .{} },
            .{
                .{ .a = 20, .b = 21, .c = 22 },
                .{ .a = 20, .d = 23, .c = 22 },
                .{.c},
                .{ .a = 20, .b = 21, .d = 23 },
            },
            .{
                .{ .a = 20, .b = 21, .c = 22 },
                .{ .c = 22, .d = 23, .e = 24 },
                .{.c},
                .{ .a = 20, .b = 21, .d = 23, .e = 24 },
            },
            .{
                .{ .a = 20, .b = 21, .c = 22 },
                .{ .c = 22, .d = 23, .a = 20 },
                .{ .c, .a },
                .{ .b = 21, .d = 23 },
            },
        }) |testcase| {
            const x_shape, const y_shape, const ctr, const z_shape = testcase;
            const x = Tensor.constant(x_shape, .{ .f32 = 0.0 });
            const y = Tensor.constant(y_shape, .{ .f32 = 0.0 });
            const z = x.dot(y, ctr);

            try zml.testing.expectEqualShapes(Shape.init(z_shape, .f32), z.shape());
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

        const Axes = std.BoundedArray(i64, MAX_RANK);

        var res_shape: Shape = .{ ._dtype = lhs.dtype() };
        // Validate batching axes
        var lhs_batching_axes: Axes = .{};
        var rhs_batching_axes: Axes = .{};
        for (batching_axes) |b_axes| {
            const l, const r = b_axes;
            stdx.debug.assert(lhs._shape.dim(l) == rhs._shape.dim(r), "dotGeneral expects batching dimensions to be equal, got {} and {} in {} and {}", .{ l, r, lhs, rhs });
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
            stdx.debug.assert(lhs._shape.dim(l) == rhs._shape.dim(r), "dotGeneral expects contracting dimensions to be equal, got {} and {} in {} and {}", .{ l, r, lhs, rhs });
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

        const mlir_ctx = lhs.getContext().mlirCtx();
        const loc = lhs.getContext().location(@src(), "dot({_},{_},contracting={any},batching={any}", .{ lhs, rhs, contracting_axes, batching_axes });
        const op = dialect.stablehlo.dot_general(
            mlir_ctx,
            lhs.value(),
            rhs.value(),
            mlirx.tensorType(mlir_ctx, res_shape),
            loc,
            .{
                .lhs_batching_dimensions = lhs_batching_axes.constSlice(),
                .rhs_batching_dimensions = rhs_batching_axes.constSlice(),
                .lhs_contracting_dimensions = lhs_contracting_axes.constSlice(),
                .rhs_contracting_dimensions = rhs_contracting_axes.constSlice(),
                .precision = .fast,
            },
        );
        return _result(res_shape, op.result(0));
    }

    /// Returns a Tensor containing the sigmoid function applied to each element of the input Tensor.
    pub fn sigmoid(self: Tensor) Tensor {
        const loc = self.getContext().mlirCtx().location(@src());
        const op = dialect.stablehlo.logistic(self.getContext().mlirCtx(), self.value(), loc);
        return _result(self._shape, op.result(0));
    }

    pub const logistic = sigmoid;

    /// Returns a Tensor containing the ReLU activation function applied to each element of the input Tensor.
    pub fn relu(self: Tensor) Tensor {
        return self.maximum(Tensor.constant(self.dims(), self.dtype().zero()));
    }

    /// Returns a Tensor containing the leaky-ReLU activation function applied to each element of the input Tensor.
    ///
    /// LeakyReLU(x) = max(0,x) + negative_slope * min(0,x)
    /// ref: https://paperswithcode.com/method/leaky-relu
    pub fn leakyReLU(self: Tensor, negative_slope: f32) Tensor {
        const below_zero = self.scale(negative_slope).minimum(Tensor.scalar(0, self.dtype()));
        return self.relu().add(below_zero);
    }

    test leakyReLU {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        const input = try zml.Buffer.fromSlice(platform, .{2}, &[_]f32{ -0.6884, 1.6795 });
        const res = try zml.testing.compileAndCall(platform, leakyReLU, .{ input, 0.1 });

        const expectation = zml.HostBuffer.fromArray(&[2]f32{ -0.0688, 1.6795 });
        try zml.testing.expectClose(expectation, res, 1e-4);
    }

    /// Returns a Tensor containing the SwiGLU activation function applied to the input Tensor.
    pub fn swiglu(self: Tensor, beta: f32, w: Tensor, b: Tensor) Tensor {
        const sigmoid_tensor = self.mul(Tensor.constant(self._shape, Data.init(self.dtype(), beta))).sigmoid();
        const one_minus_sigmoid_tensor = Tensor.constant(self._shape, Data.init(self.dtype(), 1)).sub(sigmoid_tensor);

        return self.mul(sigmoid_tensor).add(one_minus_sigmoid_tensor.mul(w.matmul(self).add(b)));
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
    /// https://paperswithcode.com/method/silu
    pub fn silu(x: Tensor) Tensor {
        return x.mul(x.sigmoid());
    }

    /// Returns a Tensor containing the softmax function applied to each element of the input Tensor.
    pub fn softmax(self: Tensor, axis_: anytype) Tensor {
        const a = self.axis(axis_);
        const max_val = self.max(a);
        const row_mask = max_val.cmp(.GT, Tensor.scalar(-std.math.inf(f64), self.dtype()));

        const exp_diff_max = self.sub(self.max(a).broad(self._shape)).exp();
        const res = exp_diff_max.div(exp_diff_max.sum(a).broad(self._shape));

        // If a row is full -inf return full 0 instead of full nan,
        // this fix attention when mask hides a full row.
        return row_mask.broad(self.shape()).select(res, Tensor.scalar(0, self.dtype()));
    }

    /// Returns a Tensor containing the log of the sum of exponential over the given axis.
    pub fn logSumExp(self: Tensor, axis_: anytype) Tensor {
        const a = self.axis(axis_);
        // stabilization: shift `self` by it's max value before passing to exponent.
        const M = self.max(a);
        const log_sum_exp = log(sum(exp(self.sub(M.broad(self._shape))), a));
        // restore the shift again
        return M.add(log_sum_exp);
    }

    /// Returns a Tensor containing the sum of elements over the given axis.
    /// Output shape is the input shape with the axis_ dim set to 1.
    pub fn sum(self: Tensor, axis_: anytype) Tensor {
        const a = self.axis(axis_);
        return ops.reduce(
            struct {
                pub fn acc(x: Tensor, res: Tensor) Tensor {
                    return res.add(x.convert(res.dtype()));
                }
            }.acc,
            self,
            Tensor.scalar(0, self.dtype()),
            &.{a},
        );
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

        const ones = [_]i64{1} ** MAX_RANK;
        var window_dimensions = ones;
        window_dimensions[a] = self.dim(a);
        var padding = [_][2]i64{.{ 0, 0 }} ** MAX_RANK;
        padding[a] = .{ self.dim(a) - 1, 0 };

        return ops.reduceWindow(
            Tensor.add,
            self,
            Tensor.scalar(0, self.dtype()),
            .{
                .base_dilations = ones[0..rk],
                .window_dilations = ones[0..rk],
                .window_strides = ones[0..rk],
                .window_dimensions = window_dimensions[0..rk],
                .padding = padding[0..rk],
            },
        );
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

        const x = try zml.Buffer.fromArray(
            platform,
            [2][5]f32{ .{ 0, 1, 1, 0, 1 }, .{ 3, 1, 0, 2, 1 } },
        );
        const res = try zml.testing.compileAndCall(platform, Local._cumsum, .{x});
        try std.testing.expectEqual(
            [2][5]f32{ .{ 0, 1, 2, 2, 3 }, .{ 3, 4, 4, 6, 7 } },
            try res.getValue([2][5]f32),
        );
    }

    /// Returns a transposed Tensor computed using the given axes.
    pub fn transpose(self: Tensor, axes_: anytype) Tensor {
        const axes__ = self.axes(axes_).constSlice();
        const default_perm = [MAX_RANK]i64{ 7, 6, 5, 4, 3, 2, 1, 0 };
        const no_op = [MAX_RANK]i64{ 0, 1, 2, 3, 4, 5, 6, 7 };

        const permutation: []const i64 = if (axes__.len == 0)
            default_perm[MAX_RANK - self.rank() ..]
        else
            toI64(axes__);

        stdx.debug.assert(permutation.len == self.rank(), "transpose expects input tensor rank and 'axes_' length to be equal, got {_} and {d}", .{ self, permutation[0..@min(permutation.len, MAX_RANK + 2)] });

        if (std.mem.eql(i64, permutation, no_op[0..self.rank()])) {
            return self;
        }

        const res_shape = self._shape.transpose(permutation);
        if (transposeIsJustAReshape(self.shape(), permutation)) {
            return self.reshape(res_shape);
        }

        const loc = self.getContext().location(@src(), "transpose({_}, {d})", .{ self, permutation });
        const op = dialect.stablehlo.transpose(
            self.getContext().mlirCtx(),
            self.value(),
            mlirx.tensorType(self.getContext().mlirCtx(), res_shape),
            loc,
            .{ .permutation = toI64(permutation) },
        );
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
        stdx.debug.assert(self.rank() < Tensor.MAX_RANK, "unflatten expects input tensor rank to be less than {}, got {}", .{ Tensor.MAX_RANK, self.rank() });

        const a = if (axis_ >= 0) self.axis(axis_) else self.axis(axis_) + 1;
        const new_dim = std.math.divExact(i64, self.dim(a), n) catch std.debug.panic("unflatten expects chosen dimension to be divisible by 'n' but {} is not divisible by {}", .{ self.dim(a), n });
        const new_shape = self._shape.set(a, n).insert(a + 1, .{ ._ = new_dim });

        const loc = self.getContext().location(@src(), "axis={}, n={}", .{ axis_, n });
        const reshaped_val = dialect.stablehlo.reshape(
            self.getContext().mlirCtx(),
            self.value(),
            mlirx.tensorType(self.getContext().mlirCtx(), new_shape),
            loc,
        );
        return _result(new_shape, reshaped_val.result(0));
    }

    /// Splits the given axis in several axes.
    /// eg: `Tensor.init(.{ .a = 10, .b = 3 }).split(.a, .{.a1 = 5, .a2 = 2});`
    /// The number of elements in the split shape must match the number of element
    /// in the target axis.
    pub fn splitAxis(self: Tensor, ax: anytype, split_shape: anytype) Tensor {
        const new_shape = self._shape.splitAxis(ax, split_shape);

        const loc = self.getContext().location(@src(), "splitAxis({}, {any})", .{ ax, split_shape });
        const reshaped_val = dialect.stablehlo.reshape(
            self.getContext().mlirCtx(),
            self.value(),
            mlirx.tensorType(self.getContext().mlirCtx(), new_shape),
            loc,
        );
        return _result(new_shape, reshaped_val.result(0));
    }

    /// Merges two or more contiguous axes into one axis.
    pub fn merge(self: Tensor, merges_: anytype) Tensor {
        return self.reshape(self._shape.mergeAxes(merges_));
    }

    /// Merges two or more non-contiguous axes into one axis.
    /// Will make a transpose if needed.
    /// .{ .a, .b, .c }.mergeTranspose(.{ .a, .c }, .ac) -> .{ .b, .ac }
    pub fn mergeTranspose(self: Tensor, axes_: anytype, merged: EnumLiteral) Tensor {
        const cont = self.contiguous(axes_);
        return cont.reshape(cont._shape.mergeAxis(merged, axes_));
    }

    /// Transposes the input Tensor, such has the given axes end up in contiguous position.
    /// .{ .a, .b, .c, .d }.contiguous(.{ .c, .a }) -> .{ .b, .d, .c, .a }
    pub fn contiguous(self: Tensor, axes_: anytype) Tensor {
        const perm = self._shape.contiguousPerm(axes_);
        return self.transpose(perm.constSlice());
    }

    /// Flattens the given axis and the next one, into one new axis.
    pub fn flatten(self: Tensor, axis_: anytype) Tensor {
        // TODO: move to torch.zig, this is equivalent to merge
        const old_shape = self._shape;
        const a = self.axis(axis_);
        // stdx.debug.assert(a + 1 < self.rank(), "Can't flatten {} on the last axis {}.", .{ self, axis });
        const new_shape = old_shape.remove(a + 1).set(a, old_shape.dim(a) * old_shape.dim(a + 1));

        const loc = self.getContext().location(@src(), "flatten({_},{})", .{ self, axis_ });
        const reshaped_val = dialect.stablehlo.reshape(
            self.getContext().mlirCtx(),
            self.value(),
            mlirx.tensorType(self.getContext().mlirCtx(), new_shape),
            loc,
        );
        // log.debug("flatten({d}, {d}) -> {d}", .{ self.dims(), axis_, new_shape[0 .. self.rank() - 1] });
        return _result(new_shape, reshaped_val.result(0));
    }

    pub inline fn flattenAll(self: Tensor) Tensor {
        // TODO: rename to just flatten, once flatten is moved to torch
        return self.reshape(.{self.count()});
    }

    pub const Slice = struct {
        start: i64 = 0,
        end: i64 = to_the_end,
        step: i32 = 1,
        singleton: bool = false,

        pub fn single(offset: i64) Slice {
            return .{ .start = offset, .end = offset + 1, .singleton = true };
        }

        const to_the_end = std.math.maxInt(i64);

        pub fn format(
            self: Slice,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            if (self.singleton) {
                try writer.print("[{}]", .{self.start});
            } else if (self.end == to_the_end and self.step == 1) {
                try writer.print("[{}..]", .{self.start});
            } else if (self.step == 1) {
                try writer.print("[{}..{}]", .{ self.start, self.end });
            } else {
                try writer.print("[{}..{}:{}]", .{ self.start, self.end, self.step });
            }
        }
    };

    /// Slices the input Tensor over the given axis using the given parameters.
    pub fn slice1d(self: Tensor, axis_: anytype, s: Slice) Tensor {
        var slices = [_]Slice{.{}} ** MAX_RANK;
        slices[self.axis(axis_)] = s;
        return self.slice(slices[0..self.rank()]);
    }

    /// Slices the input Tensor using the given parameters.
    pub fn slice(self: Tensor, slices: []const Slice) Tensor {
        var start_indices: [MAX_RANK]i64 = undefined;
        var strides: [MAX_RANK]i64 = undefined;
        var limit_indices: [MAX_RANK]i64 = undefined;
        var res_shape: Shape = self._shape;

        for (slices, 0..) |s, a| {
            stdx.debug.assert(s.step > 0, "slice expects 'step' to be positive, got {} at index {}", .{ s.step, a });

            const args: Slice = .{
                .start = self.wrapIndex(a, s.start),
                .end = if (s.end == Slice.to_the_end) self.dim(a) else self.wrapIndex(a, s.end),
                .step = s.step,
            };
            start_indices[a] = args.start;
            limit_indices[a] = args.end;
            strides[a] = args.step;
            res_shape = res_shape.setDim(a, std.math.divCeil(i64, args.end - args.start, args.step) catch unreachable);
        }

        const mlir_ctx = self.getContext().mlirCtx();
        const loc = mlir_ctx.location(@src()).namedFmt(mlir_ctx, "slices={any}", .{slices});
        const result_type = mlirx.tensorType(mlir_ctx, res_shape);
        const slice_op = dialect.stablehlo.slice(
            mlir_ctx,
            self.value(),
            start_indices[0..self.rank()],
            limit_indices[0..self.rank()],
            strides[0..self.rank()],
            result_type,
            loc,
        );

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

        const x = try zml.Buffer.fromSlice(platform, .{ 2, 5 }, &[_]f32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });

        // Wrap slice1d to hide the anytype in the signature.
        const Local = struct {
            pub fn _slice1dAxis(input: Tensor, ax: i8, slice_: Tensor.Slice) Tensor {
                return input.slice1d(ax, slice_);
            }
        };

        {
            const res = try zml.testing.compileAndCall(platform, Local._slice1dAxis, .{ x, 0, .{ .end = 1 } });
            try std.testing.expectEqual([5]f32{ 0, 1, 2, 3, 4 }, try res.getValue([5]f32));
        }
        {
            const res = try zml.testing.compileAndCall(platform, Local._slice1dAxis, .{ x, 1, .{ .start = 1, .step = 2 } });
            try std.testing.expectEqual([4]f32{ 1, 3, 6, 8 }, try res.getValue([4]f32));
        }
        {
            const res = try zml.testing.compileAndCall(platform, Local._slice1dAxis, .{ x, -1, .{ .start = -2 } });
            try std.testing.expectEqual([4]f32{ 3, 4, 8, 9 }, try res.getValue([4]f32));
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
        var slices = [_]Slice{.{}} ** MAX_RANK;
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
        var buffer: [32]mlir.Value = undefined;
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
        const loc = ctx.location(@src(), "axis={}", .{axis_});
        const op = dialect.stablehlo.concatenate(ctx.mlirCtx(), buffer[0..tensors.len], a, loc);
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
            stdx.debug.assert(shape0.eqlWithTags(tensor._shape), "stack expects tensor shapes to match, got {} and {}", .{ shape0, tensor._shape });
        }

        var reshaped: [32]Tensor = undefined;
        for (tensors, 0..) |tensor, i| {
            reshaped[i] = tensor.reshape(res_shape);
        }

        // Be careful here: we need to resolve ax before calling concatenate,
        // because we added an axis, so all
        const ax = if (@TypeOf(axis_) == EnumLiteral and axis_ == .last)
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
        const broadshape = self._shape.insert(a + 1, .{n_rep});
        const repeat_dims = Shape.range(self.rank() + 1, self.dtype()).remove(a + 1);

        var res = self.broadcast(broadshape, repeat_dims.dims()).flatten(a);
        // Restor the tag that has been lost by flatten.
        res._shape._tags.set(a, self._shape.tag(a));

        return res;
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

    /// Repeats in line each value along the given axis.
    ///
    /// * stutter1d([0, 1, 2, 3], 0, 2) = [0, 0, 1, 1, 2, 2, 3, 3]
    pub fn stutter1d(self: Tensor, axis_: i64, n_rep: u63) Tensor {
        const a = self.axis(axis_);
        const broadshape = self._shape.insert(a + 1, .{n_rep});
        const stutter_dims = Shape.range(self.rank() + 1, self.dtype()).remove(a + 1);

        return self.broadcast(broadshape, stutter_dims.dims()).flatten(a);
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

    /// Returns a Tensor containing the element-wise negation of the input Tensor.
    pub fn negate(self: Tensor) Tensor {
        const loc = self.getContext().mlirCtx().location(@src());
        const negate_op = dialect.stablehlo.negate(self.getContext().mlirCtx(), self.value(), loc);
        return _result(self._shape, negate_op.result(0));
    }

    /// Returns a Tensor containing the element-wise cosine of the input Tensor.
    pub fn cos(self: Tensor) Tensor {
        const loc = self.getContext().mlirCtx().location(@src());
        const cosine_op = dialect.stablehlo.cosine(self.getContext().mlirCtx(), self.value(), loc);
        return _result(self._shape, cosine_op.result(0));
    }

    /// Returns a Tensor containing the element-wise sine of the input Tensor.
    pub fn sin(self: Tensor) Tensor {
        const loc = self.getContext().mlirCtx().location(@src());
        const sine_op = dialect.stablehlo.sine(self.getContext().mlirCtx(), self.value(), loc);
        return _result(self._shape, sine_op.result(0));
    }

    /// Returns a Tensor containing the element-wise exponential operation of the input Tensor.
    pub fn exp(self: Tensor) Tensor {
        const loc = self.getContext().mlirCtx().location(@src());
        const op = dialect.stablehlo.exponential(self.getContext().mlirCtx(), self.value(), loc);
        return _result(self._shape, op.result(0));
    }

    /// Returns a Tensor containing the element-wise logarithm operation of the input Tensor.
    pub fn log(self: Tensor) Tensor {
        const loc = self.getContext().mlirCtx().location(@src());
        const op = dialect.stablehlo.log(self.getContext().mlirCtx(), self.value(), loc);
        return _result(self._shape, op.result(0));
    }

    /// Returns a Tensor containing the element-wise square-root of the input Tensor.
    pub fn sqrt(self: Tensor) Tensor {
        const loc = self.getContext().mlirCtx().location(@src());
        const sqrt_op = dialect.stablehlo.sqrt(self.getContext().mlirCtx(), self.value(), loc);
        return _result(self._shape, sqrt_op.result(0));
    }

    /// Returns a Tensor containing the element-wise reverse square-root of the input Tensor.
    pub fn rsqrt(self: Tensor) Tensor {
        const loc = self.getContext().mlirCtx().location(@src());
        const rsqrt_op = dialect.stablehlo.rsqrt(self.getContext().mlirCtx(), self.value(), loc);
        return _result(self._shape, rsqrt_op.result(0));
    }

    /// Returns a Tensor containing the element-wise hyperbolic tangent of the input Tensor.
    pub fn tanh(self: Tensor) Tensor {
        const loc = self.getContext().mlirCtx().location(@src());
        const tanh_op = dialect.stablehlo.tanh(self.getContext().mlirCtx(), self.value(), loc);
        return _result(self._shape, tanh_op.result(0));
    }

    /// Returns a Tensor containing the element-wise exponential minus one operation of the input Tensor.
    pub fn exponentialMinusOne(self: Tensor) Tensor {
        const loc = self.getContext().mlirCtx().location(@src());
        const expm1_op = dialect.stablehlo.exponential_minus_one(self.getContext().mlirCtx(), self.value(), loc);
        return _result(self._shape, expm1_op.result(0));
    }

    pub const ArangeArgs = HostBuffer.ArangeArgs;

    /// Returns a Tensor containing evenly spaced values within a given interval.
    pub fn arange(args: ArangeArgs, dt: DataType) Tensor {
        stdx.debug.assert(args.start <= args.end, "arange expects 'args.start' to be less than 'args.end', got {} and {}", .{ args.start, args.end });
        stdx.debug.assert(args.step > 0, "arange expects 'args.step' to be positive, got {}", .{args.step});

        const ctx = CompilationContext.current();
        const loc = ctx.location(@src(), "arange({}, dtype={})", .{ args, dt });

        const n_steps = std.math.divCeil(i64, args.end - args.start, args.step) catch unreachable;
        const sh = Shape.init(.{n_steps}, dt);
        var op = dialect.stablehlo.iota(ctx.mlirCtx(), 0, mlirx.tensorType(ctx.mlirCtx(), sh), loc);
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
        const ctx = CompilationContext.current();
        const loc = ctx.location(@src(), "iota({_}, {})", .{ res_shape, a });

        const mlir_ctx = ctx.mlirCtx();
        var op = dialect.stablehlo.iota(
            mlir_ctx,
            a,
            mlirx.tensorType(mlir_ctx, res_shape),
            loc,
        );
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

        const ctx = CompilationContext.current();
        const loc = ctx.location(@src(), "linspace({}, dtype={})", .{ args, dt });

        const sh = Shape.init(.{args.steps}, dt);
        var iota_op = dialect.stablehlo.iota(ctx.mlirCtx(), 0, mlirx.tensorType(ctx.mlirCtx(), sh), loc);
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
        const data = Data.init(dt, val);
        switch (dt.class()) {
            .float => stdx.debug.assert(!std.math.isNan(val), "scalar(NaN) is probably due to compiling a model with an uninitialized field", .{}),
            else => {},
        }
        return Tensor.constant(.{}, data);
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

        _ = try zml.testing.compileAndCall(platform, Local._fwd, .{});
    }

    /// Returns a constant Tensor with the given value.
    pub fn constant(dimz: anytype, val: Data) Tensor {
        const sh = Shape.init(dimz, val.dtype());
        const ctx = CompilationContext.current().mlirCtx();
        const loc = CompilationContext.current().location(@src(), "dims={d}, value={}", .{ sh, val });

        var constant_op = if (mlirx.denseElementAttrType(val.dtype())) |elem_type|
            dialect.stablehlo.constant(ctx, &.{}, elem_type, val.constSlice(), loc)
        else blk: {
            // Not all dtype can be serialized in the IR. If that's not possible, use f32.
            const val_f32 = val.as(f32);
            break :blk dialect.stablehlo.constant(ctx, &.{}, .f32, std.mem.asBytes(&val_f32), loc);
        };

        if (sh.rank() > 0) {
            constant_op = dialect.stablehlo.broadcast_in_dim(ctx, constant_op.result(0), &.{}, mlirx.tensorType(ctx, sh), loc);
        }
        return _result(sh, constant_op.result(0)).convert(val.dtype());
    }

    /// Embeds a buffer with concrete values into an Mlir program.
    pub fn constantTensor(val: HostBuffer) Tensor {
        const ctx = CompilationContext.current().mlirCtx();
        const loc = ctx.location(@src());
        const elem_type = mlirx.denseElementAttrType(val.dtype()) orelse std.debug.panic("constantTensor expects a dtype that can be serialized to MLIR, like f32 or i32, got {}", .{val.shape()});
        const constant_op = dialect.stablehlo.constant(ctx, val.shape().dims(), elem_type, val.bytes(), loc);
        return _result(val.shape(), constant_op.result(0));
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
        stdx.debug.assert(axes_.len == self.rank(), "broadcast expects axes_ to map all axes from self to axes of the output shape, got broadcast({}, {}, {d})", .{ self, output_shape, axes_ });
        for (0.., axes_) |self_ax, other_ax| {
            const d = self.dim(self_ax);
            stdx.debug.assert(d == 1 or d == output_shape.dim(other_ax), "broadcast expects shape axes to either be 1-sized or to match the target size. got broadcast({}, {}, {d}), error on self axis {} mapping to other axis {}", .{ self, output_shape, axes_, self_ax, other_ax });
        }

        const res_shape = output_shape.withDtype(self.dtype());
        if (std.mem.eql(i64, self.dims(), output_shape.dims())) {
            // No broadcast needed. We don't emit a new stablehlo value
            // but we propagate output_shape tags.
            return _result(res_shape, self.value());
        }
        const ctx = self.getContext();
        const result_type = mlirx.tensorType(ctx.mlirCtx(), res_shape);
        const loc = ctx.location(@src(), "broadcast({_}, {_}, axes={d})", .{ self, res_shape, axes_ });
        const broadcast_op = dialect.stablehlo.broadcast_in_dim(ctx.mlirCtx(), self.value(), axes_, result_type, loc);

        return _result(res_shape, broadcast_op.result(0));
    }

    /// Broadcasts a Tensor to the given shape, adding axes at the beginning.
    pub fn broadcastLeft(self: Tensor, output_shape: Shape) Tensor {
        stdx.debug.assert(self.rank() <= output_shape.rank(), "broadcastLeft expects tensor rank to be less than output tensor rank, got {} and {}", .{ self.rank(), output_shape.rank() });

        const a = output_shape.rank() - self.rank();
        if (self.rank() == output_shape.rank() and std.mem.eql(i64, self.dims(), output_shape.dims())) {
            return self;
        }

        return self.broadcast(output_shape, Shape.range(output_shape.rank(), output_shape.dtype()).dims()[a..]);
    }

    /// Broadcasts a Tensor to the given shape, adding axes at the end.
    pub fn broadcastRight(self: Tensor, output_shape: Shape) Tensor {
        stdx.debug.assert(self.rank() <= output_shape.rank(), "broadcastRight expects tensor rank to be less than output tensor rank, got {} and {}", .{ self.rank(), output_shape.rank() });

        if (self.rank() == output_shape.rank() and self._shape.eql(output_shape)) {
            return self;
        }

        return self.broadcast(output_shape, Shape.range(self.rank(), output_shape.dtype()).dims());
    }

    /// Broadcasts a Tensor to the given shape, extending dimensions if needed.
    pub fn broad(self: Tensor, other: Shape) Tensor {
        // TODO: broad is too restrictive because sometime you only want to specify one specific axis
        // Note: if you code below, make sure to update Shape.canBroadcastTo.
        stdx.debug.assert(self._shape.canBroadcastTo(other), "Can't broadcast {} to {}", .{ self, other });

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
        var axes_: std.BoundedArray(i64, MAX_RANK) = .{};
        for (self._shape.tags()) |t| {
            axes_.appendAssumeCapacity(@intCast(other.axis(t)));
        }
        return self.broadcast(other, axes_.constSlice());
    }

    /// Reshapes the input Tensor with the given shape.
    pub fn reshape(self: Tensor, output_shape_: anytype) Tensor {
        const output_shape = self._shape.reshape(output_shape_);
        const tensor_type = mlirx.tensorType(self.getContext().mlirCtx(), output_shape);
        const loc = self.getContext().location(@src(), "reshape({any})", .{output_shape});
        const reshape_value = dialect.stablehlo.reshape(self.getContext().mlirCtx(), self.value(), tensor_type, loc);
        return _result(output_shape, reshape_value.result(0));
    }

    /// Converts the given 1 element Tensor into a 0-rank Tensor.
    pub fn asScalar(self: Tensor) Tensor {
        stdx.debug.assert(self.count() == 1, "Tensor.asScalar expects an input with exactly 1-element got {}", .{self});
        return self.reshape(.{});
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

        const ZEROS = [_]i64{0} ** MAX_RANK;
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
        const mlir_ctx = self.getContext().mlirCtx();
        const loc = mlir_ctx.location(@src()).namedFmt(mlir_ctx, "pad({},{})", .{ padding_value, _paddings });
        const pad_op = dialect.stablehlo.pad(
            mlir_ctx,
            self.value(),
            Tensor.scalar(padding_value, self.dtype()).value(),
            .{ .low = low[0..rk], .high = high[0..rk], .interior = interior[0..rk] },
            loc,
        );

        return _result(res_shape, pad_op.result(0));
    }

    /// Inserts 1-dim axes at the given position, with the given tags.
    /// `.{.a = 5, .b = 4}.insert(.b, .{ .c, .d }) -> .{ .a = 5, .c = 1, .d = 1, .b = 4 }`
    pub fn insertAxes(self: Tensor, axis_: anytype, tags: anytype) Tensor {
        const tags_ = Shape.parseTags(tags);
        const ax = if (@TypeOf(axis_) == EnumLiteral and axis_ == .last)
            self.rank()
        else
            self.axis(axis_);

        var res_shape = self._shape;
        const ones = [_]i64{1} ** MAX_RANK;
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

        const loc = self.getContext().location(@src(), "reverse({any})", .{axes_});
        const reverse_op = dialect.stablehlo.reverse(self.getContext().mlirCtx(), self.value(), toI64(actual_axes.constSlice()), loc);
        return _result(self._shape, reverse_op.result(0));
    }

    pub const GatherOpts = struct { indices_are_sorted: bool = false };

    /// For each coordinate in `indices`,
    /// `gatherValues` extracts a single value of the given tensor.
    ///
    /// * axes_ is a single axis, or a tuple of axis: .b, or .{ .b, .c }
    /// * indices is an integer tensor
    /// * result is a tensor whose shape is similar to the input shape
    /// where the gathered axes have been replaced by axes from 'indices'.
    ///
    /// Some example input for the base case where we work on one axis:
    /// - gatherValues(f:[a]->float, .a, ind:[n]->int)[n] == f[ind[n]]
    /// - gatherValues(f:[a, b], .a, ind:[n])[n, b] == f[ind[n], b]
    /// - gatherValues(f: [a,b,c], .{.b}, ind: [n,m])[a, n, m, c] == f[a, ind[n, m], c]
    ///
    /// If an axis in common between `self` and `indices`,
    /// it is treated as a "batching" axis, meaning that semantically
    /// the operator is doing a gatherValues one time per dimension of this axis:
    /// - gatherValues(f: [a,b,c], .{.b}, ind: [a,n])[a, n] == f[a, ind[a, n]]
    ///
    /// It is an error to have an axis present in `self`, `axes_` and `indices`.
    ///
    /// If several axes are passed, then the last axis of indices is treated as coordinates:
    /// - gatherValues(f: [a,b,c], .{.b, .c}, ind: [n,2])[a, n] == f[a, ind[n][0], ind[n][1]]
    /// - gatherValues(f: [a,b,c,d], .{.b, .c}, ind: [a, n,2])[a, n, d] == f[a, ind[a, n][0], ind[a, n][1], d]
    ///
    /// It is possible to use gatherValues without tags, but batching won't be available.
    pub fn gatherValues(self: Tensor, coord_axes: anytype, indices: Tensor, opts: GatherOpts) Tensor {
        // scoped_log.debug("gatherValues({}, {any}, {})", .{ self, coord_axes, indices });
        const single_coord, const coord_axes_ = _parseGatherCoord(self, coord_axes);

        stdx.debug.assert(coord_axes_.len > 0, "gatherValues expects 1 or more axes to operate one, received none. Example: `x.gatherValues(.a, indices, .{{}})`", .{});
        for (coord_axes_.constSlice(), 0..) |a, i| {
            if (i > 0) {
                stdx.debug.assert(a == coord_axes_.get(i - 1) + 1, "gatherValues expects 'coord_axes' to be sequential. But {any} aren't sequential in {}", .{ coord_axes, self });
            }
        }

        const AxisKind = enum { batching, offset, collapsed, indices };
        var self_kind: std.BoundedArray(AxisKind, MAX_RANK) = .{};
        var indices_batch_axes: Shape.DimsArray = .{};
        for (self._shape.tags(), 0..self.rank()) |t, self_ax| {
            const maybe_coord_ax = std.mem.indexOfScalar(u3, coord_axes_.constSlice(), @intCast(self_ax));
            if (indices._shape.hasTag(t)) |id_ax| {
                // tag is both in self and indices -> it's a batching dim
                // Note: tags are required for batching.
                self_kind.appendAssumeCapacity(.batching);
                indices_batch_axes.appendAssumeCapacity(id_ax);
                stdx.debug.assert(maybe_coord_ax == null, "gatherValues expects axes to appear at most twice. Axis {s} has been found both in 'self={any}', in 'coord_axes_={any}' and in 'indices={}'", .{ self._shape._tags.get(self_ax), self, coord_axes, indices });
            } else if (maybe_coord_ax) |_| {
                // for gatherValues we collapsed all gathered axes
                // (contrary to gatherSlices where we collapse none)
                self_kind.appendAssumeCapacity(.collapsed);
            } else {
                self_kind.appendAssumeCapacity(.offset);
            }
        }

        // When we receive several coord_axes we need an extra dimension to store
        // one index per axis, which makes the coordinates of one value.
        // Otherwi se stablehlo uses the "indices.rank()" default value.
        const index_coord_axis = if (single_coord)
            indices.rank()
        else blk: {
            const ax = indices._shape.hasTag(.coord) orelse indices._shape.axis(-1);
            stdx.debug.assert(indices.dim(ax) == coord_axes_.len, "gatherValues with axes={any}, expects indices to be of shape [..., {}], got: {}", .{ coord_axes, coord_axes_.len, indices });
            break :blk ax;
        };

        // compute res shape
        var res_shape = Shape.init(.{}, self.dtype());
        var res_kind: std.BoundedArray(AxisKind, MAX_RANK) = .{};
        for (self_kind.constSlice(), 0..) |kind, ax_usize| {
            const ax: u3 = @intCast(ax_usize);
            if (ax == coord_axes_.get(0)) {
                // The first val_ax is special cause this is the place where we insert indices axes.
                for (indices._shape.tags(), 0..indices.rank()) |t, id_ax| {
                    if (id_ax == index_coord_axis) continue;
                    if (std.mem.indexOfScalar(i64, indices_batch_axes.constSlice(), @intCast(id_ax))) |_| {
                        // batching dim are already in res
                        continue;
                    }

                    res_shape = res_shape.appendDim(indices.dim(id_ax), t);
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
        if (indices.count() == 1) {
            return self.dynamicSlice1d(coord_axes_.get(0), .{ .start = indices.flattenAll().squeeze(0), .len = 1 }).reshape(res_shape);
        }

        var slice_dims: Shape.DimsArray = .{};
        for (self_kind.constSlice(), self.dims()) |k, d| {
            slice_dims.appendAssumeCapacity(switch (k) {
                .batching, .collapsed => 1,
                .offset => d,
                .indices => unreachable,
            });
        }

        // scoped_log.debug("gatherValues --> {} {any}", .{ res_shape, res_kind.constSlice() });
        const loc = self.getContext().mlirCtx().location(@src());
        const gather_op = dialect.stablehlo.gather(
            self.getContext().mlirCtx(),
            self.value(),
            indices.value(),
            slice_dims.constSlice(),
            loc,
            .{
                .offset_dims = _collectAxes(AxisKind, res_kind, .offset).constSlice(),
                .collapsed_slice_dims = _collectAxes(AxisKind, self_kind, .collapsed).constSlice(),
                .operand_batching_dims = _collectAxes(AxisKind, self_kind, .batching).constSlice(),
                .start_indices_batching_dims = indices_batch_axes.constSlice(),
                .start_index_map = _collectAxes(AxisKind, self_kind, .collapsed).constSlice(),
                .index_vector_dim = index_coord_axis,
                .indices_are_sorted = opts.indices_are_sorted,
            },
        );

        const mlir_shape = fromMlirValue(gather_op.result(0)).shape();
        stdx.debug.assert(mlir_shape.eql(res_shape), "gatherValues expects that batching indices appear in the same order in 'self' and 'indices', got: self={}, indices={}. You should transpose one or the other.", .{ self, indices });
        return _result(res_shape, gather_op.result(0));
    }

    test gatherValues {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        {
            // Only test shapes
            var comp = try zml.module.CompilationContext.init(std.testing.allocator, "test", platform);
            defer comp.deinit();
            comp.activate();
            defer comp.deactivate();

            inline for (.{
                .{ .{ .a = 10 }, .a, .{}, .{} },
                .{ .{ .a = 10 }, .a, .{ .n = 8 }, .{ .n = 8 } },
                .{ .{ .a = 10, .b = 20 }, .a, .{}, .{ .b = 20 } },
                .{ .{ .a = 10, .b = 20 }, .a, .{ .n = 8 }, .{ .n = 8, .b = 20 } },
                .{ .{ .a = 10, .b = 20 }, 0, .{ .n = 8 }, .{ .n = 8, .b = 20 } },
                // Favor val shape, instead of indices shape.
                .{ .{ .a = 10, .b = 20 }, .b, .{ .n = 8 }, .{ .a = 10, .n = 8 } },
                .{ .{ .a = 10, .b = 20, .c = 30 }, .b, .{ .n = 8 }, .{ .a = 10, .n = 8, .c = 30 } },
                // batching axes are implicits.
                .{ .{ .a = 10, .b = 20 }, .b, .{ .a = 10 }, .{ .a = 10 } },
                .{ .{ .a = 10, .b = 20 }, .a, .{ .b = 20 }, .{ .b = 20 } },
                .{ .{ .a = 10, .b = 20 }, .b, .{ .a = 10, .n = 8 }, .{ .a = 10, .n = 8 } },
                // stablehlo.gather is biased toward indices shape (like gatherSlice).
                // This make it awkward to use when you have both batching dimension and new indices dimensions.
                // For now we reject those, and let user explicitly transpose self or indices if needed.
                // .{ .{ .a = 10, .b = 20 }, .b, .{ .n = 8, .a = 10 }, .{ .a = 10, .n = 8 } },
                // Also handle tuples
                .{ .{ .a = 10, .b = 20 }, .{ .a, .b }, .{ .n = 8, ._ = 2 }, .{ .n = 8 } },
                .{ .{ 10, 20 }, .{ -2, -1 }, .{ 8, 2 }, .{8} },
                // and 1-tuple
                .{ .{ .a = 10, .b = 20 }, .{.b}, .{ .n = 8, ._ = 1 }, .{ .a = 10, .n = 8 } },
            }) |testcase| {
                const x_shape, const tag, const idx_shape, const res_shape = testcase;
                const x = Tensor.constant(x_shape, .{ .f16 = 0 });
                const idx = Tensor.constant(idx_shape, .{ .i32 = 0 });
                const y = gatherValues(x, tag, idx, .{});
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
        // scoped_log.debug("gatherSlice({}, {_}, {})", .{ self, slice_shape, indices });

        const tagged_api = slice_shape.isFullyTagged();
        if (tagged_api) {
            for (slice_shape.tags()) |t| {
                stdx.debug.assert(self._shape.hasTag(t) != null, "gatherSlices expects `slices_shape` to only use tags from `self`. But {s} wasn't found in {}", .{ t, self });
            }
        } else {
            // For untagged api, we require all slices to be specified.
            // Note: we could relax this and right align the slice.
            stdx.debug.assert(slice_shape.rank() == self.rank(), "gatherSlices expects `slice_shape.rank()` to match `self.rank()`. Got: gatherSlices({}, slice={_}). To avoid specifying all axes in `slice_shape`, you can use tags.", .{ self, slice_shape });
        }

        const index_coord_axis = indices._shape.hasTag(.coord) orelse indices._shape.axis(-1);
        stdx.debug.assert(indices.dim(index_coord_axis) == slice_shape.rank(), "gatherSlices({}, slice={_}, indices) expects 'indices' to be a tensor [..., {}], got {}", .{ self, slice_shape, slice_shape.rank(), indices });

        // Compute result shape
        var res_shape = indices._shape.remove(index_coord_axis).withDtype(self.dtype());
        var slice_dims = self._shape._dims;
        var self_batch_axes: std.BoundedArray(i64, MAX_RANK) = .{};
        var indices_batch_axes: std.BoundedArray(i64, MAX_RANK) = .{};
        var start_index_map: std.BoundedArray(i64, MAX_RANK) = .{};
        var self_offset_axes: std.BoundedArray(i64, MAX_RANK) = .{};
        for (self._shape.tags(), 0..self.rank()) |t, self_ax| {
            const maybe_slice_ax: ?u3 = if (tagged_api) slice_shape.hasTag(t) else @intCast(self_ax);

            if (tagged_api and indices._shape.hasTag(t) != null) {
                // tag is both in self and indices -> it's a batching dim
                // Note: tags are required for batching.
                self_batch_axes.appendAssumeCapacity(@intCast(self_ax));
                indices_batch_axes.appendAssumeCapacity(indices._shape.axis(t));
                slice_dims.set(self_ax, 1);
                stdx.debug.assert(slice_shape.hasTag(t) == null, "gatherSlices expect axes to be either batches or slices axes. Axis {s} has been found both in `slices={_}` and `indices={}`", .{ t, slice_shape, indices });
            } else if (maybe_slice_ax) |slice_ax| {
                // Specified axes contains the start offset of the slices,
                // and are collected in `start_index_map`.
                const slice_dim = slice_shape.dim(slice_ax);
                stdx.debug.assert(slice_dim <= self._shape.dim(self_ax), "gatherSlices expects `slice_shape` to be smaller than `self.shape()`. On axis {s}, got {} > {}.", .{ t, slice_shape, self._shape });
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

        const loc = self.getContext().location(@src(), "gatherSlices({_}, slice_shape={_}, idx={_})", .{ self, slice_shape, indices });
        const gather_op = dialect.stablehlo.gather(
            self.getContext().mlirCtx(),
            self.value(),
            indices.value(),
            slice_dims.constSlice(),
            loc,
            .{
                .offset_dims = self_offset_axes.constSlice(),
                .collapsed_slice_dims = &.{},
                .operand_batching_dims = self_batch_axes.constSlice(),
                .start_indices_batching_dims = indices_batch_axes.constSlice(),
                .start_index_map = start_index_map.constSlice(),
                .index_vector_dim = index_coord_axis,
                .indices_are_sorted = opts.indices_are_sorted,
            },
        );
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
            var comp = try zml.module.CompilationContext.init(std.testing.allocator, "test", platform);
            defer comp.deinit();
            comp.activate();
            defer comp.deactivate();

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
                const x = Tensor.constant(x_shape, .{ .f16 = 0 });
                const slice_shape = Shape.init(slice_dims, .u16);
                const idx = Tensor.constant(idx_shape, .{ .i32 = 0 });
                const y = gatherSlices(x, slice_shape, idx, .{});
                try zml.testing.expectEqualShapes(Shape.init(res_shape, .f16), y.shape());
                try std.testing.expect(y.value().owner().verify());

                const mod = try zml.compileFn(
                    std.testing.allocator,
                    Local._gatherSlices,
                    .{ x.shape(), slice_shape, idx.shape(), .{ .indices_are_sorted = true } },
                    platform,
                );
                defer mod.deinit();
            }
        }

        // Test with actual values.
        const range = try zml.HostBuffer.arange(std.testing.allocator, .{ .end = 2 * 4 * 6 }, .u16);
        defer range.deinit(std.testing.allocator);
        const operand = try range.reshape(.{ .a = 2, .b = 4, .c = 6 }).toDevice(platform);
        defer operand.deinit();
        const start_indices = (try zml.Buffer.fromArray(platform, [2][2]i32{ .{ 2, 1 }, .{ 0, 3 } })).withTags(.{ .n, ._ });
        defer start_indices.deinit();

        const result = try zml.testing.compileAndCall(platform, Local._gatherSlices, .{ operand, Shape.init(.{ .b = 2, .c = 3 }, .u16), start_indices, .{} });

        const expected = zml.HostBuffer.fromArray(&[2][2][2][3]u16{
            .{
                .{ .{ 13, 14, 15 }, .{ 19, 20, 21 } },
                .{ .{ 37, 38, 39 }, .{ 43, 44, 45 } },
            },
            .{
                .{ .{ 3, 4, 5 }, .{ 9, 10, 11 } },
                .{ .{ 27, 28, 29 }, .{ 33, 34, 35 } },
            },
        });
        try zml.testing.expectClose(expected, result, 0);
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
        update_fn: *const fn (Tensor, Tensor) Tensor = increment,

        pub fn increment(old_value: Tensor, new_value: Tensor) Tensor {
            return old_value.add(new_value);
        }

        pub fn override(old_value: Tensor, new_value: Tensor) Tensor {
            _ = old_value;
            return new_value;
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
        scoped_log.debug("scatterSlices({}, {any}, {})", .{ self, indices, updates });

        const UpdateType = @TypeOf(ScatterOpts.increment);

        const Custom = struct {
            pub fn inc(custom: *const UpdateType, old_value: Tensor, new_value: Tensor) Tensor {
                return @call(.auto, custom, .{ old_value, new_value });
            }
        };

        return ops.scatter(Tensor, *const UpdateType, Custom.inc, self, opts.update_fn, indices, updates, opts);
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
                return Tensor.constant(idx_shape, .{ .i32 = 0 });
            }
        };

        {
            // Only test shapes
            var comp = try zml.module.CompilationContext.init(std.testing.allocator, "test", platform);
            defer comp.deinit();
            comp.activate();
            defer comp.deactivate();
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
                const x = Tensor.constant(x_shape, .{ .f16 = 0 });
                const updates = Tensor.constant(updates_shapes, .{ .f16 = 0 });

                const y = scatterSlices(x, indices, updates, .{});
                // Shape doesn't change with scatterSlices
                try zml.testing.expectEqualShapes(x.shape(), y.shape());
                try std.testing.expect(y.value().owner().verify());
            }
        }
        // Test with actual values, no batching.
        {
            const a_host = try zml.HostBuffer.arange(std.testing.allocator, .{ .end = 9 }, .i32);
            const a = (try zml.Buffer.from(platform, a_host.reshape(.{ 3, 3 }), .{})).withTags(.{ .a, .b });
            defer a.deinit();
            a_host.deinit(std.testing.allocator);

            const scatter_indices = try zml.Buffer.fromArray(platform, [2]i32{ 0, 2 });
            const updates = try zml.Buffer.fromArray(platform, [2][3]i32{ .{ 10, 20, 30 }, .{ 70, 80, 90 } });

            const expected = [3][3]i32{ .{ 10, 21, 32 }, .{ 3, 4, 5 }, .{ 76, 87, 98 } };
            const result = try zml.testing.compileAndCall(platform, Local._scatter, .{
                a,
                &.{scatter_indices.withTags(.{.n})},
                updates.withTags(.{ .n, .b }),
            });
            try std.testing.expect(a.shape().eql(result.shape()));
            try std.testing.expectEqual(expected, result.getValue(@TypeOf(expected)));
        }
        // Test with setting individual values (no batching)
        {
            const a_host = try zml.HostBuffer.arange(std.testing.allocator, .{ .end = 9 }, .i32);
            const a = try zml.Buffer.from(platform, a_host, .{});
            defer a.deinit();
            a_host.deinit(std.testing.allocator);

            const scatter_indices = try zml.Buffer.fromArray(platform, [2]i32{ 2, 7 });
            const updates = try zml.Buffer.fromArray(platform, [2]i32{ 20, 70 });

            const expected = [9]i32{ 0, 1, 22, 3, 4, 5, 6, 77, 8 };
            const result = try zml.testing.compileAndCall(platform, Local._scatter, .{
                a,
                &.{scatter_indices.withTags(.{.n})},
                updates.withTags(.{.n}),
            });
            try std.testing.expect(a.shape().eql(result.shape()));
            try std.testing.expectEqual(expected, result.getValue(@TypeOf(expected)));
        }
        {
            // Test with actual values and batching along axis .a
            const operand = try zml.Buffer.constant(platform, Shape.init(.{ .a = 2, .b = 3, .c = 4, .d = 2 }, .u16), 0);
            defer operand.deinit();
            const start_indices = (try zml.Buffer.fromArray(
                platform,
                [2][2][3][2]i32{
                    .{
                        .{ .{ 0, 0 }, .{ 1, 0 }, .{ 2, 1 } },
                        .{ .{ 0, 1 }, .{ 1, 1 }, .{ 0, 9 } },
                    },
                    .{
                        .{ .{ 0, 0 }, .{ 2, 1 }, .{ 2, 2 } },
                        .{ .{ 1, 2 }, .{ 0, 1 }, .{ 1, 0 } },
                    },
                },
            )).withTags(.{ .n, .a, .m, .coord });
            defer start_indices.deinit();

            const values = try zml.Buffer.constant(
                platform,
                Shape.init(.{ .n = 2, .a = 2, .m = 3, .c = 2, .d = 2 }, .u16),
                1,
            );
            defer values.deinit();

            const result = try zml.testing.compileAndCall(platform, Local._scatterCB, .{ operand, start_indices, values });

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
            try std.testing.expectEqual(expected, result.getValue(@TypeOf(expected)));
        }
    }

    /// Returns a Tensor containing the maximum over a given axis.
    pub fn max(self: Tensor, axis_: anytype) Tensor {
        const a = self.axis(axis_);
        return ops.reduce(
            struct {
                pub fn cmp(x: Tensor, res: Tensor) Tensor {
                    return res.maximum(x.convert(res.dtype()));
                }
            }.cmp,
            self,
            Tensor.constant(&.{}, self.dtype().minValue()),
            &.{a},
        );
    }

    /// Returns a Tensor containing the minimum over a given axis.
    pub fn min(self: Tensor, axis_: anytype) Tensor {
        const a = self.axis(axis_);
        return ops.reduce(
            struct {
                pub fn cmp(x: Tensor, res: Tensor) Tensor {
                    return res.minimum(x.convert(res.dtype()));
                }
            }.cmp,
            self,
            Tensor.constant(&.{}, self.dtype().maxValue()),
            &.{a},
        );
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

    /// Returns two Tensors containing the maximum and the index of this maximum over a given axis.
    ///
    /// Stable argmax:
    /// * bubbles up Nan
    /// * in case of equality the smallest index matching the maximum
    pub fn argMax(x: Tensor, axis_: anytype) ArgMaxRes {
        const a = x.axis(axis_);
        const dt: DataType = if (x.dim(a) <= std.math.maxInt(i32)) .i32 else .i64;

        return ops.reduce(
            ArgMaxRes.cmp,
            .{ .values = x, .indices = Tensor.arange(.{ .end = x.dim(a) }, dt).broadcast(x.shape(), &.{a}) },
            .{ .values = Tensor.constant(&.{}, x.dtype().minValue()), .indices = Tensor.scalar(0, dt) },
            &.{a},
        );
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

        const argmax = try zml.compileFn(allocator, ArgMaxTest._fwd, .{Shape.init(.{ 1, 5 }, .f32)}, platform);
        defer argmax.deinit();
        // Test with tie
        {
            const x = try zml.Buffer.fromArray(platform, [1][5]f32{.{ 5.0, 4.1, 7.9, 0, 7.9 }});
            const res = argmax.call(.{x});
            const max_ = res.values.getValue(f32);
            const max_idx = res.indices.getValue(i32);
            try std.testing.expectEqual(max_, 7.9);
            // We should always return the first max found.
            try std.testing.expectEqual(max_idx, 2);
        }

        // Test with Nan
        {
            const x = try zml.Buffer.fromArray(platform, [1][5]f32{.{ 5.0, std.math.nan(f32), 7.9, 0, 7.9 }});
            const res = argmax.call(.{x});
            const max_ = try res.values.getValue(f32);
            const max_idx = try res.indices.getValue(i32);
            try std.testing.expect(std.math.isNan(max_));
            try std.testing.expectEqual(max_idx, 1);
        }
    }

    pub const SortRes = ArgMaxRes;

    /// Returns two Tensors. The first contains the sorted values and the second one contains the sorted indices.
    pub fn sort(self: Tensor, axis_: anytype, opts: struct { descending: bool = false }) SortRes {
        const a = self.axis(axis_);
        const indices = Tensor.arange(.{ .end = self.dim(a) }, .i32).broadcast(self._shape, &.{a});
        const res = ops.sort(
            struct {
                fn call(direction: dialect.stablehlo.ComparisonDirection.Direction, lhs: Tensor, rhs: Tensor, _: Tensor, _: Tensor) Tensor {
                    return lhs.cmp(direction, rhs);
                }
            }.call,
            if (opts.descending) .GT else .LT,
            .{ self, indices },
            self.axis(axis_),
            true,
        );
        return .{ .values = res[0], .indices = res[1] };
    }

    pub const ArgSortOpts = struct { descending: bool = false };

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

        var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena_state.deinit();
        const allocator = arena_state.allocator();
        // 2D Tensor - dim = 1, ascending
        {
            const x = try zml.Buffer.fromSlice(platform, .{ 2, 5 }, &[_]f32{ -0.9264, 0.7156, 1.0202, 0.3992, 1.2349, 1.0003, -0.1932, 1.3935, 0.7316, 0.0851 });
            const res = try zml.testing.compileAndCall(platform, Local._argsort, .{ x, 1, .{} });
            const res_cpu = try res.toHostAlloc(allocator);
            try std.testing.expectEqualSlices(i32, &.{ 0, 3, 1, 2, 4, 1, 4, 3, 0, 2 }, res_cpu.items(i32));
        }
        // 3D Tensor, dim = 1, descending
        {
            const x = try zml.Buffer.fromSlice(platform, .{ 1, 5, 10 }, &[_]f16{
                -0.2505, 1.2520,  -0.7041, 0.1066,  1.2773,  -1.7246, 0.8389,  1.1094,  0.0601,  1.0684,
                0.9619,  1.3916,  1.2246,  -0.1406, 0.3674,  -1.2480, -1.7051, -0.0934, 0.3435,  0.4373,
                1.3809,  0.5444,  -0.6079, 1.2031,  -0.6880, 1.2979,  -0.1869, 0.2991,  0.0156,  0.1847,
                0.6626,  -0.3040, -0.8726, -1.4805, -1.6943, 1.1055,  -2.0078, -0.5288, 0.8813,  0.8008,
                2.0527,  1.1230,  0.5430,  0.2494,  -0.9434, 0.7876,  0.1818,  0.9258,  -2.4902, 1.5918,
            });
            const res_dev = try zml.testing.compileAndCall(platform, Local._argsort, .{ x, 1, .{ .descending = true } });
            const res = try res_dev.toHostAlloc(allocator);
            try std.testing.expectEqualSlices(i32, &.{
                4, 1, 1, 2, 0, 2, 0, 0, 3, 4,
                2, 0, 4, 4, 1, 3, 4, 4, 1, 0,
                1, 4, 2, 0, 2, 4, 2, 2, 0, 3,
                3, 2, 0, 1, 4, 1, 1, 1, 2, 1,
                0, 3, 3, 3, 3, 0, 3, 3, 4, 2,
            }, res.items(i32));
        }
        // 4D Tensor, dim = 3, ascending
        {
            const x = try zml.Buffer.fromSlice(platform, .{ 4, 2, 1, 4 }, &[_]i32{
                89, 31, 22, 42,
                64, 39, 0,  30,
                64, 71, 46, 31,
                89, 82, 78, 86,
                55, 32, 43, 19,
                93, 24, 45, 72,
                64, 86, 62, 88,
                57, 21, 19, 12,
            });
            const res_dev = try zml.testing.compileAndCall(platform, Local._argsort, .{ x, 3, .{} });
            const res = try res_dev.toHostAlloc(allocator);
            try std.testing.expectEqualSlices(i32, &.{
                2, 1, 3, 0,
                2, 3, 1, 0,
                3, 2, 0, 1,
                2, 1, 3, 0,
                3, 1, 2, 0,
                1, 2, 3, 0,
                2, 0, 1, 3,
                3, 2, 1, 0,
            }, res.items(i32));
        }
    }

    /// Returns a Tensor representing the result of Top-K over the given axis.
    pub fn topK(self: Tensor, k: u32, axis_: anytype, opts: struct { descending: bool = true }) SortRes {
        const a = self.axis(axis_);
        const result = self.sort(a, .{ .descending = opts.descending });
        return .{
            .values = result.values.slice1d(a, .{ .end = k }),
            .indices = result.indices.slice1d(a, .{ .end = k }),
        };
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
        const ones = [_]i64{1} ** Tensor.MAX_RANK;
        var window_dimensions = ones;
        window_dimensions[a] = opts.window_dimensions;
        var window_strides = window_dimensions;
        if (opts.window_strides) |stride| window_strides[a] = stride;

        var base_dilations = ones;
        base_dilations[a] = opts.base_dilations;
        var window_dilations = ones;
        window_dilations[a] = opts.window_dilations;

        var padding = [_][2]i64{.{ 0, 0 }} ** Tensor.MAX_RANK;
        padding[a] = opts.padding;

        return ops.reduceWindow(
            MaxPoolRes.cmp,
            .{ .values = self, .indices = iota(self._shape, a) },
            .{ .values = Tensor.constant(.{}, self.dtype().minValue()), .indices = Tensor.scalar(0, .i32) },
            .{
                .window_dimensions = window_dimensions[0..self.rank()],
                .window_strides = window_strides[0..self.rank()],
                .base_dilations = base_dilations[0..self.rank()],
                .window_dilations = window_dilations[0..self.rank()],
                .padding = padding[0..self.rank()],
            },
        );
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

        var padding = [_][2]i64{.{ 0, 0 }} ** Tensor.MAX_RANK;
        padding[a - 1] = opts.padding[0];
        padding[a] = opts.padding[1];

        return ops.reduceWindow(
            MaxPoolRes.cmp,
            .{ .values = self, .indices = iota(self._shape, a) },
            .{ .values = Tensor.constant(.{}, self.dtype().minValue()), .indices = Tensor.scalar(0, .i32) },
            .{
                .window_dimensions = window_dimensions[0..self.rank()],
                .window_strides = window_strides[0..self.rank()],
                .base_dilations = base_dilation[0..self.rank()],
                .window_dilations = window_dilations[0..self.rank()],
                .padding = padding[0..self.rank()],
            },
        );
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
        var comp = try zml.module.CompilationContext.init(std.testing.allocator, "test", platform);
        defer comp.deinit();
        comp.activate();
        defer comp.deactivate();

        inline for (.{
            .{ .{ .a = 12 }, .a, 3, .{ .a = 4 } },
            .{ .{ .a = 12, .b = 2 }, .a, 3, .{ .a = 4, .b = 2 } },
            .{ .{ 12, 2 }, 0, 3, .{ 4, 2 } },
        }) |testcase| {
            const x_shape, const ax, const n_chunks, const res = testcase;
            const x = Tensor.constant(x_shape, .{ .f16 = 0 });
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
    ) []Tensor {
        const a = self.axis(axis_);
        const d = self.dim(a);
        const chunk_size: i64 = @divFloor(d, n_chunks);
        const tail_chunk_size: i64 = @rem(d, chunk_size);

        const allocator = self.getContext().allocator();
        var chunks = std.ArrayListUnmanaged(Tensor).initCapacity(allocator, n_chunks + 1) catch @panic("OOM");
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
        return chunks.items;
    }

    test chunkAllowTrailing {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        // Only test shapes
        var comp = try zml.module.CompilationContext.init(std.testing.allocator, "test", platform);
        defer comp.deinit();
        comp.activate();
        defer comp.deactivate();

        inline for (.{
            .{ .{ .a = 10 }, .a, 3, .{ .a = 3 }, .{ .a = 1 } },
            .{ .{ .a = 10, .b = 2 }, .a, 3, .{ .a = 3, .b = 2 }, .{ .a = 1, .b = 2 } },
            .{ .{ 10, 2 }, 0, 3, .{ 3, 2 }, .{ 1, 2 } },
            .{ .{ 12, 2 }, 0, 3, .{ 4, 2 }, .{} },
        }) |testcase| {
            const x_shape, const ax, const n_chunks, const res, const trailing = testcase;
            const x = Tensor.constant(x_shape, .{ .f16 = 0 });
            const chunks = x.chunkAllowTrailing(x.axis(ax), n_chunks);

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

        const allocator = self.getContext().allocator();
        const res = allocator.alloc(Tensor, split_sizes.len) catch @panic("OOM");
        errdefer allocator.dealloc(res);

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
        stdx.debug.assert(slice_.start.rank() == 0, "dynamicSlice1d expects 'slice_.start' tensor rank to be a scalar, got {}", .{slice_.start});

        const a = self.axis(axis_);
        const new_shape = self._shape.set(a, slice_.len);
        const loc = self.getContext().location(@src(), "dynSlice({}, len={})", .{ axis_, slice_.len });

        var start_indices = [_]mlir.Value{constant(.{}, slice_.start.dtype().zero()).value()} ** MAX_RANK;
        start_indices[a] = slice_.start.value();

        const op = dialect.stablehlo.dynamic_slice(
            self.getContext().mlirCtx(),
            self.value(),
            new_shape.dims(),
            start_indices[0..self.rank()],
            loc,
        );

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
        const loc = self.getContext().location(@src(), "dynSlice({any})", .{slices_});

        const idx_dtype = if (slices.len > 0) slices.get(0).start.dtype() else .i32;
        const zero = Tensor.scalar(0, idx_dtype).value();
        var offset_values = [_]mlir.Value{zero} ** MAX_RANK;
        var res_shape = self._shape;
        for (slices.constSlice(), 0..) |slice_, i| {
            const offset = slice_.start;
            const len = slice_.len;
            if (slices_tags.len == 0) {
                stdx.debug.assert(self.rank() == slices.len, "dynamicSlice expects tensor rank and 'slices_' length to be equal, got {} and {}", .{ self.rank(), slices.len });

                offset_values[i] = offset.value();
                res_shape._dims.set(i, len);

                stdx.debug.assert(len <= self.dim(i), "dynamicSlice expects slices 'len' to be less than or equal to their corresponding dimension in input tensor, got {} and {} for index {}", .{ len, self.dim(i), i });
            } else {
                const t = slices_tags.get(i);
                const a = res_shape.hasTag(t) orelse stdx.debug.panic("dynamicSlice expects input tensor to have tags used in 'slices_' but {s} is missing (input shape is {})", .{ t, self._shape });

                stdx.debug.assert(len <= self.dim(a), "dynamicSlice expects slices 'len' to be less than their corresponding dimension in input tensor, got {} and {} for axis {s}", .{ len, self.dim(a), t });

                offset_values[a] = offset.value();
                res_shape._dims.set(a, len);
            }
        }
        const op = dialect.stablehlo.dynamic_slice(self.getContext().mlirCtx(), self.value(), res_shape.dims(), offset_values[0..self.rank()], loc);
        return _result(res_shape, op.result(0));
    }

    test dynamicSlice {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();
        const T = f32;

        {
            const x = try zml.Buffer.fromArray(platform, [10]T{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            const z = try zml.Buffer.scalar(platform, 4, .i32);
            const res = try zml.testing.compileAndCall(platform, Tensor.dynamicSlice1d, .{ x, 0, .{ .len = 2, .start = z } });

            try std.testing.expectEqual([2]T{ 4, 5 }, try res.getValue([2]T));
        }

        {
            // Strided
            const x = try zml.Buffer.fromArray(platform, [2][5]T{ .{ 0, 1, 2, 3, 4 }, .{ 5, 6, 7, 8, 9 } });
            const z = try zml.Buffer.scalar(platform, 3, .i32);

            const res = try zml.testing.compileAndCall(platform, Tensor.dynamicSlice1d, .{ x, 1, .{ .len = 2, .start = z } });
            try std.testing.expectEqual([4]T{ 3, 4, 8, 9 }, res.getValue([4]T));
        }
    }

    /// Updates a slice of the input Tensor along a specific axis using the given 'update' Tensor, with a start offset known at runtime.
    /// Note this is the untagged api, if you have tags, you should use dynamicUpdateSlice directly.
    pub fn dynamicUpdateSlice1d(self: Tensor, update: Tensor, axis_: i64, offset: Tensor) Tensor {
        const placeholder = Tensor.scalar(0, .i32);
        var start_indices = [_]Tensor{placeholder} ** MAX_RANK;
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
                stdx.debug.assert(self._shape.hasTag(t) != null, "dynamicUpdateSlice expects 'update_' tensor tags to be a subset of input tensor tags but {s} is missing (input shape is {})", .{ t, self._shape });
            }

            var update_shape = self._shape;
            var prev_ax: i8 = -1;
            for (self._shape.tags(), 0..) |t, self_ax| {
                if (update._shape.hasTag(t)) |up_ax| {
                    stdx.debug.assert(up_ax == prev_ax + 1, "dynamicUpdateSlice expects 'update_' and input tensor axis to have the same order, got {} and {}. (hint: you need to explicitly transpose 'update_')", .{ update_, self });

                    update_shape._dims.set(self_ax, update.dim(up_ax));
                    prev_ax = up_ax;
                } else {
                    update_shape._dims.set(self_ax, 1);
                }
            }
            update = update.reshape(update_shape);
        }

        stdx.debug.assert(self.rank() == update.rank(), "dynamicUpdateSlice expects input and computed update tensors to have the same rank, got {} and {}", .{ self, update });

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
        var offset_values: [MAX_RANK]mlir.Value = undefined;
        if (offset_tags.len == 0) {
            // Without offset tags we need the same number of offset than rank.
            stdx.debug.assert(self.rank() == offset.len, "dynamicUpdateSlice expects input tensor rank and 'offset_' length to be equal, got {} and {}", .{ self.rank(), offset.len });

            for (offset.constSlice(), 0..) |idx, i| {
                offset_values[i] = idx.value();
            }
        } else {
            // If an axis isn't specified, update the full slice.
            // This is only allowed when using tagged sliced.
            offset_values = .{zero} ** MAX_RANK;
            for (offset.constSlice(), offset_tags.constSlice()) |start, t| {
                const a = self._shape.hasTag(t) orelse stdx.debug.panic("dynamicUpdateSlice expects input tensor to have tags used in 'offset_' but {s} is missing (input shape is {})", .{ t, self._shape });
                offset_values[a] = start.value();
            }
        }

        const loc = self.getContext().mlirCtx().location(@src());
        const op = dialect.stablehlo.dynamic_update_slice(
            self.getContext().mlirCtx(),
            self.value(),
            update.value(),
            offset_values[0..self.rank()],
            loc,
        );
        return _result(self._shape, op.result(0));
    }

    test dynamicUpdateSlice {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        {
            const x = try zml.Buffer.fromArray(platform, [10]f32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            const y = try zml.Buffer.fromArray(platform, [2]f32{ -1, -1 });
            const idx = try zml.Buffer.scalar(platform, 4, .i32);
            const res = try zml.testing.compileAndCall(
                platform,
                struct {
                    pub fn _fwd(x_: Tensor, idx_: struct { a: Tensor }, y_: Tensor) Tensor {
                        return x_.dynamicUpdateSlice(idx_, y_);
                    }
                }._fwd,
                .{ x.withTags(.{.a}), .{ .a = idx }, y.withTags(.{.a}) },
            );
            try std.testing.expectEqual([10]f32{ 0, 1, 2, 3, -1, -1, 6, 7, 8, 9 }, try res.getValue([10]f32));
        }

        {
            // Updates 2D, tagged api
            const x = try zml.Buffer.fromArray(platform, [2][5]f32{ .{ 0, 1, 2, 3, 4 }, .{ 5, 6, 7, 8, 9 } });
            const y = try zml.Buffer.fromArray(platform, [2]f32{ -1, -1 });
            const idx = try zml.Buffer.scalar(platform, 3, .i32);

            const res = try zml.testing.compileAndCall(
                platform,
                struct {
                    pub fn _fwd(x_: Tensor, idx_: Tensor, y_: Tensor) Tensor {
                        return x_.dynamicUpdateSlice(.{ .b = idx_ }, y_);
                    }
                }._fwd,
                .{ x.withTags(.{ .a, .b }), idx, y.withTags(.{.a}) },
            );
            try std.testing.expectEqualDeep(
                [2][5]f32{ .{ 0, 1, 2, -1, 4 }, .{ 5, 6, 7, -1, 9 } },
                try res.getValue([2][5]f32),
            );
        }

        {
            // Updates 2D slice, un-tagged api. Note that `y` needs to have a 1 dimension axis.
            const x = try zml.Buffer.fromArray(platform, [2][5]f32{ .{ 0, 1, 2, 3, 4 }, .{ 5, 6, 7, 8, 9 } });
            const y = try zml.Buffer.fromArray(platform, [2][1]f32{ .{-1}, .{-1} });
            const idx = try zml.Buffer.scalar(platform, 3, .i32);
            const res = try zml.testing.compileAndCall(
                platform,
                struct {
                    pub fn _fwd(x_: Tensor, idx_: Tensor, y_: Tensor) Tensor {
                        return x_.dynamicUpdateSlice(.{ zml.Tensor.scalar(0, .i32), idx_ }, y_);
                    }
                }._fwd,
                .{ x, idx, y },
            );
            try std.testing.expectEqualDeep(
                [2][5]f32{ .{ 0, 1, 2, -1, 4 }, .{ 5, 6, 7, -1, 9 } },
                res.getValue([2][5]f32),
            );
        }

        {
            // Updates 2D, partial update
            const x = try zml.Buffer.fromArray(platform, [2][5]f32{ .{ 0, 1, 2, 3, 4 }, .{ 5, 6, 7, 8, 9 } });
            const y = try zml.Buffer.fromArray(platform, [1]f32{-1});
            const idx_a = try zml.Buffer.scalar(platform, 1, .i32);
            const idx_b = try zml.Buffer.scalar(platform, 3, .i32);
            const res = try zml.testing.compileAndCall(
                platform,
                struct {
                    pub fn _fwd(x_: Tensor, idx_: struct { a: Tensor, b: Tensor }, y_: Tensor) Tensor {
                        return x_.dynamicUpdateSlice(idx_, y_);
                    }
                }._fwd,
                .{ x.withTags(.{ .a, .b }), .{ .a = idx_a, .b = idx_b }, y.withTags(.{.a}) },
            );
            try std.testing.expectEqualDeep(
                [2][5]f32{ .{ 0, 1, 2, 3, 4 }, .{ 5, 6, 7, -1, 9 } },
                res.getValue([2][5]f32),
            );
        }

        {
            // Updates 2D, partial update, un-tagged api.
            const x = try zml.Buffer.fromArray(platform, [2][5]f32{ .{ 0, 1, 2, 3, 4 }, .{ 5, 6, 7, 8, 9 } });
            const y = try zml.Buffer.fromArray(platform, [1][1]f32{.{-1}});
            const idx_a = try zml.Buffer.scalar(platform, 1, .i32);
            const idx_b = try zml.Buffer.scalar(platform, 3, .i32);
            const A = struct {
                pub fn _fwd(x_: Tensor, idx_: [2]Tensor, y_: Tensor) Tensor {
                    return x_.dynamicUpdateSlice(&idx_, y_);
                }
            };
            const res = try zml.testing.compileAndCall(platform, A._fwd, .{ x, .{ idx_a, idx_b }, y });
            try std.testing.expectEqualDeep(
                [2][5]f32{ .{ 0, 1, 2, 3, 4 }, .{ 5, 6, 7, -1, 9 } },
                res.getValue([2][5]f32),
            );
        }
    }

    /// Returns a Tensor containing the element-wise result of the given 'cmp' comparison between the two input Tensors.
    pub fn cmp(self: Tensor, direction: dialect.stablehlo.ComparisonDirection.Direction, other: Tensor) Tensor {
        stdx.debug.assert(self.dtype() == other.dtype(), "cmp expects input tensors to be of the same type, got {} and {}", .{ self.dtype(), other.dtype() });

        if (self.rank() == 0 and other.rank() != 0) return self.broadcast(other._shape, &.{}).cmp(direction, other);
        if (self.rank() != 0 and other.rank() == 0) return self.cmp(direction, other.broadcast(self._shape, &.{}));

        stdx.debug.assert(self._shape.eql(other._shape), "cmp expects input tensor shapes to match, got {} and {}", .{ self._shape, other._shape });

        const loc = self.getContext().location(@src(), "cmp(.{s})", .{@tagName(direction)});
        const op = dialect.stablehlo.compare(
            self.getContext().mlirCtx(),
            self.value(),
            other.value(),
            dialect.stablehlo.ComparisonDirection.init(self.getContext().mlirCtx(), direction),
            getComparisonType(self.getContext().mlirCtx(), self.dtype()),
            loc,
        );

        return _result(self._shape.withDtype(.bool), op.result(0));
    }

    /// For each vector in the input tensor,
    /// creates a diagonal-matrix where diagonal values are set to the vector values.
    pub fn toDiagonal(self: Tensor, axis_: anytype, new_tags: [2]EnumLiteral) Tensor {
        stdx.debug.assert(self.rank() < MAX_RANK - 1, "toDiagonal expects input up to {} rank, got {}", .{ MAX_RANK - 1, self });
        const a = self.axis(axis_);
        const d = self.dim(a);
        var res_shape = self._shape;
        res_shape._dims.replaceRange(a, 1, &.{ d, d }) catch unreachable;
        res_shape._tags.replaceRange(a, 1, &.{ @tagName(new_tags[0]), @tagName(new_tags[1]) }) catch unreachable;

        const values = self.insertAxes(a + 1, .{new_tags[1]}).broad(res_shape);
        const zeros = Tensor.constant(res_shape, self.dtype().zero());

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

        const x = try zml.Buffer.fromArray(platform, [2][2]u8{ .{ 1, 2 }, .{ 3, 4 } });
        {
            const res = try zml.testing.compileAndCall(platform, Local._toDiag, .{x});
            try std.testing.expectEqual(
                [2][2][2]u8{ .{
                    .{ 1, 0 },
                    .{ 0, 2 },
                }, .{
                    .{ 3, 0 },
                    .{ 0, 4 },
                } },
                try res.getValue([2][2][2]u8),
            );
        }
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

        const zeros = Tensor.constant(self.shape(), self.dtype().zero());
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

        const x = try zml.Buffer.fromArray(platform, [3][3]u8{
            .{ 1, 1, 1 },
            .{ 1, 1, 1 },
            .{ 1, 1, 1 },
        });
        {
            const res = try zml.testing.compileAndCall(platform, Local._tri, .{ x, 0 });
            try std.testing.expectEqual(
                [3][3]u8{
                    .{ 1, 0, 0 },
                    .{ 1, 1, 0 },
                    .{ 1, 1, 1 },
                },
                try res.getValue([3][3]u8),
            );
        }
        {
            const res = try zml.testing.compileAndCall(platform, Local._tri, .{ x, 1 });
            try std.testing.expectEqual(
                [3][3]u8{
                    .{ 1, 1, 0 },
                    .{ 1, 1, 1 },
                    .{ 1, 1, 1 },
                },
                try res.getValue([3][3]u8),
            );
        }
        {
            const res = try zml.testing.compileAndCall(platform, Local._tri, .{ x, -1 });
            try std.testing.expectEqual(
                [3][3]u8{
                    .{ 0, 0, 0 },
                    .{ 1, 0, 0 },
                    .{ 1, 1, 0 },
                },
                try res.getValue([3][3]u8),
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

        stdx.debug.assert(bool_tensor._shape.eqlDims(on_true._shape), "select expects input tensor and 'on_true' tensor dimensions to match, got {} and {}", .{ bool_tensor._shape, on_true._shape });
        stdx.debug.assert(bool_tensor._shape.eqlDims(on_false._shape), "select expects input tensor and 'on_false' tensor dimensions to match, got {} and {}", .{ bool_tensor._shape, on_false._shape });

        const loc = bool_tensor.getContext().mlirCtx().location(@src());
        const op = dialect.stablehlo.select(
            bool_tensor.getContext().mlirCtx(),
            bool_tensor.value(),
            on_true.value(),
            on_false.value(),
            loc,
        );

        return _result(on_true._shape, op.result(0));
    }

    /// Returns a Tensor containing the element-wise not logical operation of the input Tensor.
    pub fn not(self: Tensor) Tensor {
        const loc = self.getContext().mlirCtx().location(@src());
        const op = dialect.stablehlo.not(self.getContext().mlirCtx(), self.value(), loc);
        return _result(self._shape, op.result(0));
    }

    /// Returns a Tensor containing boolean indicating if there is a non-zero value over the given axis.
    pub fn any(self: Tensor, axis_: anytype) Tensor {
        const pred = self.cmp(.NE, Tensor.constant(self.dims(), self.dtype().zero()));
        return ops.reduce(
            struct {
                pub fn acc(x: Tensor, res: Tensor) Tensor {
                    return res.logical(.OR, x);
                }
            }.acc,
            pred,
            Tensor.scalar(false, .bool),
            &.{self.axis(axis_)},
        );
    }

    /// Returns a Tensor containing boolean indicating if there is a non-zero value over the given axis.
    pub fn all(self: Tensor, axis_: anytype) Tensor {
        const pred = if (self.dtype() == .bool) self else self.cmp(.NE, Tensor.scalar(0, self.dtype()));
        return ops.reduce(
            struct {
                pub fn acc(x: Tensor, res: Tensor) Tensor {
                    return res.logical(.AND, x);
                }
            }.acc,
            pred,
            Tensor.scalar(true, .bool),
            &.{self.axis(axis_)},
        );
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
        stdx.debug.assert(vectors.len < Tensor.MAX_RANK, "cartesianProduct expects at most {} input vectors, received {} !", .{ Tensor.MAX_RANK - 1, vectors.len });
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
        const client = zml.testing.env();

        const x = try zml.Buffer.fromSlice(client, .{6}, &[_]i32{ 0, 1, 2, 3, 4, 5 });
        const y = try zml.Buffer.fromSlice(client, .{4}, &[_]i32{ 0, 1, 2, 3 });

        const Local = struct {
            pub fn _cartesianProduct2(a: Tensor, b: Tensor) [2]Tensor {
                return cartesianProduct(2, .{ a, b });
            }
        };

        {
            const xs, const ys = try zml.testing.compileAndCall(client, Local._cartesianProduct2, .{ x, y });
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
                try xs.getValue([6][4]i32),
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
                try ys.getValue([6][4]i32),
            );
        }
    }

    /// Given a set of N vectors of lengths A, B, C, D,
    /// returns 1 tensors of rank N+1, and shape (A, B, C, D, N).
    /// For any coordinate (a, b, c, d), we have:
    ///
    /// - res[a, b, c, d] == (A[a], B[b], C[c], D[d])
    pub fn cartesianProductStacked(vectors: []const Tensor) Tensor {
        var out = std.BoundedArray(Tensor, Tensor.MAX_RANK).init(vectors.len) catch unreachable;
        _cartesianProduct(vectors, out.slice());

        return Tensor.stack(out.constSlice(), .last, .coord);
    }

    test cartesianProductStacked {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();
        const x = try zml.Buffer.fromSlice(platform, .{6}, &[_]i32{ 0, 1, 2, 3, 4, 5 });
        const y = try zml.Buffer.fromSlice(platform, .{4}, &[_]i32{ 0, 1, 2, 3 });

        const Local = struct {
            pub fn _fwd(a: Tensor, b: Tensor) Tensor {
                return cartesianProductStacked(&.{ a, b });
            }
        };

        const z = try zml.testing.compileAndCall(platform, Local._fwd, .{ x, y });
        try std.testing.expectEqualDeep(
            [6][4][2]i32{
                .{ .{ 0, 0 }, .{ 0, 1 }, .{ 0, 2 }, .{ 0, 3 } },
                .{ .{ 1, 0 }, .{ 1, 1 }, .{ 1, 2 }, .{ 1, 3 } },
                .{ .{ 2, 0 }, .{ 2, 1 }, .{ 2, 2 }, .{ 2, 3 } },
                .{ .{ 3, 0 }, .{ 3, 1 }, .{ 3, 2 }, .{ 3, 3 } },
                .{ .{ 4, 0 }, .{ 4, 1 }, .{ 4, 2 }, .{ 4, 3 } },
                .{ .{ 5, 0 }, .{ 5, 1 }, .{ 5, 2 }, .{ 5, 3 } },
            },
            try z.getValue([6][4][2]i32),
        );
    }

    fn binaryOp(
        src: std.builtin.SourceLocation,
        op_name: []const u8,
        op_fn: fn (mlir.Context, mlir.Value, mlir.Value, mlir.Location) mlir.Operation,
    ) fn (Tensor, Tensor) Tensor {
        return struct {
            pub fn binaryOpHelper(self: Tensor, other: Tensor) Tensor {
                stdx.debug.assert(self.dtype() == other.dtype(), "{s} expects tensor to be of same type, got {} and {}", .{ op_name, self, other });

                if (self.rank() == 0 and other.rank() != 0) {
                    return binaryOpHelper(self.broad(other._shape), other);
                }

                if (self.rank() != 0 and other.rank() == 0) {
                    return binaryOpHelper(self, other.broad(self._shape));
                }

                stdx.debug.assert(self._shape.eql(other._shape), "{s} expects tensor shapes to match, got {} and {}", .{ op_name, self._shape, other._shape });

                const ctx = self.getContext();
                const location = ctx.location(src, "{s}({_}, {_})", .{ op_name, self, other });
                const ret = @call(.auto, op_fn, .{ ctx.mlirCtx(), self.value(), other.value(), location });
                return _result(self._shape, ret.result(0));
            }
        }.binaryOpHelper;
    }

    /// Insert code that will print the content of the given buffer at runtime.
    /// Only for debug purpose, it inserts device to host synchronization
    /// so it will slow down the program execution.
    pub fn print(input: Tensor) Tensor {
        // TODO: find a way of doing print that doesn't involve a H2D copy.
        return ops.addHostCallback(
            &printCallback,
            null,
            &.{input},
            &.{input.shape()},
            .{ .output_operand_aliases = &.{0} },
        )[0];
    }

    fn printCallback(_: ?*anyopaque, inputs: []const HostBuffer, outputs: []const HostBuffer) void {
        const host_buffer = inputs[0];
        std.log.defaultLog(.info, .zml, "Device buffer: {}: {}", .{ host_buffer.shape(), host_buffer.pretty() });
        // This is true because of the operand aliases.
        // Since the result is already pointing to the input we don't need to modify the buffer.
        std.debug.assert(host_buffer._data == outputs[0]._data);
    }
};

fn initPoolArg(rank: usize, data: []const i64) [Tensor.MAX_RANK]i64 {
    // TODO use shape
    var result = [_]i64{1} ** Tensor.MAX_RANK;
    const start = rank - data.len;
    @memcpy(result[start .. start + data.len], data);
    return result;
}

fn getPoolResDims(dt: DataType, in_dims: []const i64, base_dilations: @Vector(Tensor.MAX_RANK, i64), padding: []const i64, window_dimensions: @Vector(Tensor.MAX_RANK, i64), window_dilations: @Vector(Tensor.MAX_RANK, i64), window_strides: @Vector(Tensor.MAX_RANK, i64)) Shape {
    // TODO use shape
    var input_dims = [_]i64{1} ** Tensor.MAX_RANK;
    @memcpy(input_dims[0..in_dims.len], in_dims);

    const input_dims_: @Vector(Tensor.MAX_RANK, i64) = input_dims;
    const splat_one: @Vector(Tensor.MAX_RANK, i64) = @splat(1);
    const dilated_input_shape: @Vector(Tensor.MAX_RANK, i64) = (input_dims_ - splat_one) * base_dilations + splat_one;
    var pad_slice0: @Vector(Tensor.MAX_RANK, i64) = @splat(padding[0]);
    var pad_slice1: @Vector(Tensor.MAX_RANK, i64) = @splat(padding[0]);
    if (padding.len > 1) {
        var idx: usize = 0;
        while (idx < in_dims.len * 2) : (idx += 2) {
            pad_slice0[idx / 2] = padding[idx];
            pad_slice1[idx / 2] = padding[idx + 1];
        }
    }
    const padded_input_shape: @Vector(Tensor.MAX_RANK, i64) = pad_slice0 + dilated_input_shape + pad_slice1;
    const dilated_window_shape = (window_dimensions - splat_one) * window_dilations + splat_one;
    const dims = @divFloor(padded_input_shape - dilated_window_shape, window_strides) + splat_one;
    const dims_arr: [Tensor.MAX_RANK]i64 = @bitCast(dims);
    return Shape.init(dims_arr[0..in_dims.len], dt);
}

fn getComparisonType(ctx: mlir.Context, dtype: DataType) dialect.stablehlo.CompareType {
    return dialect.stablehlo.CompareType.init(ctx, switch (dtype) {
        .i4, .i8, .i16, .i32, .i64 => .SIGNED,
        .bool, .u4, .u8, .u16, .u32, .u64 => .UNSIGNED,
        .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2, .f8e5m2fnuz, .bf16, .f16, .f32, .f64 => .FLOAT,
        .c64, .c128 => @panic("Can't compare complex numbers"),
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

    const x = try zml.Buffer.fromSlice(platform, .{ 2, 2, 5 }, &data);
    const result = try zml.testing.compileAndCall(platform, MaxPool._fwd, .{x});
    try zml.testing.expectEqualShapes(Shape.init(.{ 2, 2, 2 }, .f32), result.values.shape());
    try zml.testing.expectEqualShapes(Shape.init(.{ 2, 2, 2 }, .i32), result.indices.shape());
    const buffer = result.values.getValue([2][2][2]f32);
    try std.testing.expectEqualDeep(
        [2][2][2]f32{
            [2][2]f32{
                [2]f32{ 2, 4 },
                [2]f32{ 7, 9 },
            },
            [2][2]f32{
                [2]f32{ 12, 14 },
                [2]f32{ 17, 19 },
            },
        },
        buffer,
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
    const x = try zml.Buffer.fromSlice(platform, .{ 2, 2, 5, 5 }, &data);

    const result = try zml.testing.compileAndCall(platform, MaxPool._fwd, .{x});
    try zml.testing.expectEqualShapes(Shape.init(.{ 2, 2, 2, 4 }, .f32), result.values.shape());
    try zml.testing.expectEqualShapes(Shape.init(.{ 2, 2, 2, 4 }, .i32), result.indices.shape());
    var buffer: [2][2][2][4]f32 = undefined;
    _ = try result.values.toHost(std.mem.asBytes(&buffer));
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

pub fn Bufferized(comptime T: type) type {
    // TODO: we should strip out the non-buffer fields.
    // Currently it's confusing cause the Bufferized struct contains field that are never read.
    // Also it will simplify the layout of the Bufferized struct.
    // accelerating the calls to execute.
    return meta.MapRestrict(Tensor, Buffer).map(T);
}

/// Return a clone of a type with Tensors replaced by Shapes.
/// Recursively descends into the type.
/// See also: shapesOf() and its tests, and meta.MapType().
pub fn ShapeOf(comptime T: type) type {
    const M = meta.MapType(Tensor, Shape);
    return M.map(T);
}

/// Return a clone of the argument where each instance of a Tensor is replaced
/// by its Shape. This is similar to ShapeOf(), but with runtime values.
/// See also: meta.mapAlloc().
pub fn shapesOf(model: anytype, allocator: std.mem.Allocator) !ShapeOf(@TypeOf(model)) {
    var shapes: ShapeOf(@TypeOf(model)) = undefined;
    try meta.mapAlloc(struct {
        fn shapeFromTensorCallback(_: void, tensor: Tensor) Shape {
            return tensor.shape();
        }
    }.shapeFromTensorCallback, allocator, {}, model, &shapes);
    return shapes;
}

test shapesOf {
    const alloc = std.testing.allocator;

    // Tensor in struct
    {
        const S = struct {
            a: Tensor,
        };
        const shape = Shape.init(.{ 28, 28 }, .f32);
        const s: S = .{
            .a = Tensor{ ._shape = shape, ._id = undefined },
        };
        const shapes = try shapesOf(s, alloc);
        try std.testing.expectEqual(shape, shapes.a);
    }

    // single Tensor
    {
        const shape = Shape.init(.{ 28, 28 }, .f32);
        const tensor = Tensor{ ._shape = shape, ._id = undefined };
        const shapes = try shapesOf(tensor, alloc);
        try std.testing.expectEqual(shape, shapes);
    }

    // nn linear layer, no bias
    {
        const nn = @import("nn.zig");
        const shape = Shape.init(.{ 28, 28 }, .f32);
        const layer: nn.Linear = .{
            .weight = Tensor{ ._shape = shape, ._id = undefined },
            .bias = null,
        };

        const shapes = try shapesOf(layer, alloc);
        try std.testing.expectEqual(shape, shapes.weight);
        try std.testing.expectEqual(null, shapes.bias);
    }

    // model
    {
        const Mnist = struct {
            fc1: Layer,
            fc2: Layer,

            const Layer = struct {
                weight: Tensor,
                bias: Tensor,
            };
        };

        const fc1_weight_shape = Shape.init(.{ 500, 784 }, .f32);
        const fc1_bias_shape = Shape.init(.{500}, .f32);
        const fc2_weight_shape = Shape.init(.{ 10, 500 }, .f32);
        const fc2_bias_shape = Shape.init(.{10}, .f32);
        const mnist: Mnist = .{
            .fc1 = .{
                .weight = Tensor{ ._shape = fc1_weight_shape, ._id = undefined },
                .bias = Tensor{ ._shape = fc1_bias_shape, ._id = undefined },
            },
            .fc2 = .{
                .weight = Tensor{ ._shape = fc2_weight_shape, ._id = undefined },
                .bias = Tensor{ ._shape = fc2_bias_shape, ._id = undefined },
            },
        };

        const shapes = try shapesOf(mnist, alloc);
        try std.testing.expectEqual(fc1_weight_shape, shapes.fc1.weight);
        try std.testing.expectEqual(fc1_bias_shape, shapes.fc1.bias);
        try std.testing.expectEqual(fc2_weight_shape, shapes.fc2.weight);
        try std.testing.expectEqual(fc2_bias_shape, shapes.fc2.bias);
    }
}

pub fn _collectAxes(T: type, bounded_array: std.BoundedArray(T, Tensor.MAX_RANK), value: T) std.BoundedArray(i64, Tensor.MAX_RANK) {
    var res: std.BoundedArray(i64, Tensor.MAX_RANK) = .{};
    for (bounded_array.constSlice(), 0..) |v, ax| {
        if (v == value) {
            res.appendAssumeCapacity(@intCast(ax));
        }
    }
    return res;
}

fn _parseGatherCoord(self: Tensor, axes_: anytype) struct { bool, std.BoundedArray(u3, Tensor.MAX_RANK) } {
    const AxesT = @TypeOf(axes_);
    const axes_is_scalar = AxesT == EnumLiteral or AxesT == comptime_int or @typeInfo(AxesT) == .int;

    const coord_axes = if (axes_is_scalar)
        std.BoundedArray(u3, Tensor.MAX_RANK).fromSlice(&.{self.axis(axes_)}) catch unreachable
    else
        self.axes(axes_);

    return .{ axes_is_scalar, coord_axes };
}

fn parseArrayInfo(T: type) Shape {
    return switch (@typeInfo(T)) {
        .Array => |arr| {
            const s = parseArrayInfo(arr.child);
            return s.insert(0, .{arr.len});
        },
        else => .{ ._dtype = DataType.fromZigType(T) },
    };
}

inline fn toI64(values: anytype) []i64 {
    var res: [Tensor.MAX_RANK]i64 = undefined;
    for (values, 0..) |val, i| res[i] = @intCast(val);
    return res[0..values.len];
}

fn transposeIsJustAReshape(x: Shape, permutation: []const i64) bool {
    var perm: std.BoundedArray(struct { u8, bool }, Tensor.MAX_RANK) = .{};
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

    const mod = try zml.compileFn(std.testing.allocator, Local._fwd, .{Shape.init(.{10}, .f32)}, platform);
    defer mod.deinit();
}
