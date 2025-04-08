const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

pub const abs = functors.unary_fn("stablehlo.abs").call;
pub const cosine = functors.unary_fn("stablehlo.cosine").call;
pub const sine = functors.unary_fn("stablehlo.sine").call;
pub const exponential = functors.unary_fn("stablehlo.exponential").call;
pub const exponential_minus_one = functors.unary_fn("stablehlo.exponential_minus_one").call;
pub const floor = functors.unary_fn("stablehlo.floor").call;
pub const log = functors.unary_fn("stablehlo.log").call;
pub const log_plus_one = functors.unary_fn("stablehlo.log_plus_one").call;
pub const not = functors.unary_fn("stablehlo.not").call;
pub const negate = functors.unary_fn("stablehlo.negate").call;
pub const sqrt = functors.unary_fn("stablehlo.sqrt").call;
pub const tanh = functors.unary_fn("stablehlo.tanh").call;
pub const cbrt = functors.unary_fn("stablehlo.cbrt").call;
pub const ceil = functors.unary_fn("stablehlo.ceil").call;
pub const rsqrt = functors.unary_fn("stablehlo.rsqrt").call;
pub const count_leading_zeros = functors.unary_fn("stablehlo.count_leading_zeros").call;
pub const is_finite = functors.unary_fn("stablehlo.is_finite").call;
pub const logistic = functors.unary_fn("stablehlo.logistic").call;
pub const popcnt = functors.unary_fn("stablehlo.popcnt").call;
pub const sign = functors.unary_fn("stablehlo.sign").call;
pub const real = functors.unary_fn("stablehlo.real").call;
pub const imag = functors.unary_fn("stablehlo.imag").call;

pub const add = functors.binary_fn("stablehlo.add").call;
pub const multiply = functors.binary_fn("stablehlo.multiply").call;
pub const divide = functors.binary_fn("stablehlo.divide").call;
pub const subtract = functors.binary_fn("stablehlo.subtract").call;
pub const or_ = functors.binary_fn("stablehlo.or").call;
pub const xor = functors.binary_fn("stablehlo.xor").call;
pub const and_ = functors.binary_fn("stablehlo.and").call;
pub const atan2 = functors.binary_fn("stablehlo.atan2").call;
pub const maximum = functors.binary_fn("stablehlo.maximum").call;
pub const minimum = functors.binary_fn("stablehlo.minimum").call;
pub const power = functors.binary_fn("stablehlo.power").call;
pub const remainder = functors.binary_fn("stablehlo.remainder").call;
pub const shift_left = functors.binary_fn("stablehlo.shift_left").call;
pub const shift_right_arithmetic = functors.binary_fn("stablehlo.shift_right_arithmetic").call;
pub const shift_right_logical = functors.binary_fn("stablehlo.shift_right_logical").call;
pub const complex = functors.binary_fn("stablehlo.complex").call;

const functors = struct {
    fn unary_fn(comptime op_name: [:0]const u8) type {
        return struct {
            pub fn call(ctx: mlir.Context, value: mlir.Value, location: mlir.Location) mlir.Operation {
                return mlir.Operation.make(ctx, op_name, .{
                    .operands = &.{value},
                    .result_type_inference = true,
                    .location = location,
                });
            }
        };
    }

    pub fn binary_fn(comptime op_name: [:0]const u8) type {
        return struct {
            pub fn call(ctx: mlir.Context, lhs: mlir.Value, rhs: mlir.Value, location: mlir.Location) mlir.Operation {
                return mlir.Operation.make(ctx, op_name, .{
                    .operands = &.{ lhs, rhs },
                    .result_type_inference = true,
                    .location = location,
                });
            }
        };
    }
};

pub fn return_(ctx: mlir.Context, value: mlir.Value, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.return", .{
        .variadic_operands = &.{&.{value}},
        .verify = false,
        .location = location,
    });
}

pub fn returns_(ctx: mlir.Context, values: []const mlir.Value, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.return", .{
        .variadic_operands = &.{values},
        .verify = false,
        .location = location,
    });
}

pub fn bitcast_convert(ctx: mlir.Context, value: mlir.Value, result_type: mlir.Type, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.bitcast_convert", .{
        .operands = &.{value},
        .results = &.{result_type},
        .location = location,
    });
}

pub fn cholesky(ctx: mlir.Context, value: mlir.Value, lower: bool, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.cholesky", .{
        .operands = &.{value},
        .result_type_inference = true,
        .attributes = &.{
            .{ "lower", .i1FromBool(ctx, lower) },
        },
        .location = location,
    });
}

pub fn clamp(ctx: mlir.Context, min: mlir.Value, value: mlir.Value, max: mlir.Value, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.clamp", .{
        .operands = &.{ min, value, max },
        .result_type_inference = true,
        .location = location,
    });
}

pub const DotPrecision = union(enum) {
    fast,
    high,
    highest,
    algorithm: DotAlgorithm,

    pub fn precisionAttr(self: DotPrecision, ctx: mlir.Context) mlir.Attribute {
        const precision = PrecisionAttribute.init(ctx, switch (self) {
            .fast => .DEFAULT,
            .high => .HIGH,
            .highest => .HIGHEST,
            // When we specify the dot algorithm, we should not specify the precision.
            .algorithm => .DEFAULT,
        });
        return precision.as(mlir.Attribute);
    }

    pub fn algorithmAttr(self: DotPrecision, ctx: mlir.Context, operand_type: mlir.Type) ?mlir.Attribute {
        return switch (self) {
            .algorithm => |algo| algo.asAttr(ctx, operand_type),
            else => null,
        };
    }
};

pub const DotAlgorithm = struct {
    accumulation: mlir.FloatTypes,
    // Note stablehlo distinguish between left/right component_count
    // but all the supported algorithm have the same component_count on both side.
    component_count: u8 = 1,
    num_primitive_operations: u8 = 1,
    allow_imprecise_accumulation: bool = false,

    // bf16_6x: each input is decomposed to 3 bf16 components, then 6 dot operations are done on those components, and the result is accumulated in f32.
    // not sure where this is available.
    pub const bf16_6x: DotAlgorithm = .{
        .operand = .bf16,
        .accumulation = .f32,
        .component_count = 1,
        .num_primitive_operations = 6,
        .allow_imprecise_accumulation = false,
    };

    pub fn asAttr(self: DotAlgorithm, ctx: mlir.Context, operand_type: mlir.Type) mlir.Attribute {
        const tensor_type = operand_type.as(mlir.RankedTensorType);
        const elem_type = tensor_type.getElementType();

        return mlir.Attribute.wrap(c.stablehloDotAlgorithmGet(
            ctx.inner(),
            elem_type.inner(),
            elem_type.inner(),
            self.accumulation.asType(ctx).inner(),
            self.component_count,
            self.component_count,
            self.num_primitive_operations,
            self.allow_imprecise_accumulation,
        ));
    }
};

/// General matrix multiplication "a la Einstein sum"
/// Note: stablehlo doesn't do type inference for dot_general
pub fn dot_general(
    ctx: mlir.Context,
    lhs: mlir.Value,
    rhs: mlir.Value,
    result_type: mlir.Type,
    location: mlir.Location,
    opts: struct {
        lhs_batching_dimensions: []const i64,
        rhs_batching_dimensions: []const i64,
        lhs_contracting_dimensions: []const i64,
        rhs_contracting_dimensions: []const i64,
        precision: DotPrecision,
    },
) mlir.Operation {
    const precisions: [2]mlir.Attribute = @splat(opts.precision.precisionAttr(ctx));
    const attributes = [3]mlir.AttrTuple{
        .{
            "dot_dimension_numbers", DotDimensionNumbersAttribute.init(ctx, .{
                .lhs_batching_dimensions = opts.lhs_batching_dimensions,
                .rhs_batching_dimensions = opts.rhs_batching_dimensions,
                .lhs_contracting_dimensions = opts.lhs_contracting_dimensions,
                .rhs_contracting_dimensions = opts.rhs_contracting_dimensions,
            }).as(mlir.Attribute),
        },
        .{ "precision_config", .array(ctx, &precisions) },
        // keep algorithm as the last attribute so we can omit it when it's not set.
        .{ "algorithm", opts.precision.algorithmAttr(ctx, lhs.getType()) orelse undefined },
    };
    const n_attributes = if (opts.precision == .algorithm) attributes.len else attributes.len - 1;
    return mlir.Operation.make(ctx, "stablehlo.dot_general", .{
        .operands = &.{ lhs, rhs },
        .results = &.{result_type},
        .attributes = attributes[0..n_attributes],
        .location = location,
    });
}

pub fn constant(
    ctx: mlir.Context,
    result_type: mlir.RankedTensorType,
    elem_type: mlir.DenseElementsAttributeTypes,
    raw_bytes: []const u8,
    location: mlir.Location,
) mlir.Operation {
    const attribute = switch (elem_type) {
        inline else => |dt| mlir.DenseElementsAttribute(dt).init(result_type.as(mlir.Type), raw_bytes).as(mlir.Attribute),
    };

    return mlir.Operation.make(ctx, "stablehlo.constant", .{
        .operands = &.{},
        .results = &.{result_type.as(mlir.Type)},
        .attributes = &.{.{ "value", attribute }},
        .location = location,
    });
}

pub fn convert(ctx: mlir.Context, value: mlir.Value, result_type: mlir.Type, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.convert", .{
        .operands = &.{value},
        .results = &.{result_type},
        .location = location,
    });
}

pub fn broadcast_in_dim(ctx: mlir.Context, operand: mlir.Value, dims: []const i64, result_type: mlir.Type, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.broadcast_in_dim", .{
        .operands = &.{operand},
        .results = &.{result_type},
        .attributes = &.{
            .{ "broadcast_dimensions", .dense(ctx, .i64, dims) },
        },
        .location = location,
    });
}

pub fn transpose(ctx: mlir.Context, value: mlir.Value, result_type: mlir.Type, location: mlir.Location, opts: struct { permutation: []const i64 }) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.transpose", .{
        .operands = &.{value},
        .results = &.{result_type},
        .attributes = &.{
            .{ "permutation", .dense(ctx, .i64, opts.permutation) },
        },
        .location = location,
    });
}

pub fn slice(ctx: mlir.Context, operand: mlir.Value, start_indices: []const i64, limit_indices: []const i64, strides: []const i64, result_type: mlir.Type, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.slice", .{
        .operands = &.{operand},
        .results = &.{result_type},
        .attributes = &.{
            .{ "start_indices", .dense(ctx, .i64, start_indices) },
            .{ "limit_indices", .dense(ctx, .i64, limit_indices) },
            .{ "strides", .dense(ctx, .i64, strides) },
        },
        .location = location,
    });
}

pub fn concatenate(ctx: mlir.Context, inputs: []const mlir.Value, dimension: i64, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.concatenate", .{
        .operands = inputs,
        .result_type_inference = true,
        .attributes = &.{
            .{ "dimension", .int(ctx, .i64, dimension) },
        },
        .location = location,
    });
}

pub fn reshape(ctx: mlir.Context, value: mlir.Value, result_type: mlir.RankedTensorType, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.reshape", .{
        .operands = &.{value},
        .results = &.{result_type.as(mlir.Type)},
        .location = location,
    });
}

pub fn select(ctx: mlir.Context, condition: mlir.Value, then: mlir.Value, else_: mlir.Value, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.select", .{
        .operands = &.{ condition, then, else_ },
        .results = &.{then.getType()},
        .location = location,
    });
}

pub fn gather(
    ctx: mlir.Context,
    value: mlir.Value,
    indices: mlir.Value,
    slice_sizes: []const i64,
    location: mlir.Location,
    args: struct {
        offset_dims: []const i64,
        collapsed_slice_dims: []const i64,
        operand_batching_dims: []const i64,
        start_indices_batching_dims: []const i64,
        start_index_map: []const i64,
        index_vector_dim: i64,
        indices_are_sorted: bool = false,
    },
) mlir.Operation {
    return mlir.Operation.make(
        ctx,
        "stablehlo.gather",
        .{
            .operands = &.{ value, indices },
            .result_type_inference = true,
            .attributes = &.{
                .{ "dimension_numbers", GatherDimensionNumbersAttribute.init(
                    ctx,
                    args.offset_dims,
                    args.collapsed_slice_dims,
                    args.operand_batching_dims,
                    args.start_indices_batching_dims,
                    args.start_index_map,
                    args.index_vector_dim,
                ).as(mlir.Attribute) },
                .{ "slice_sizes", .dense(ctx, .i64, slice_sizes) },
                .{ "indices_are_sorted", .boolean(ctx, args.indices_are_sorted) },
            },
            .location = location,
        },
    );
}

fn elementTypeOrSelf(typ: mlir.Type) mlir.Type {
    return if (typ.as(mlir.ShapedType)) |shaped| {
        return shaped.elementType();
    } else typ;
}

pub const ScatterArgs = struct {
    update_window_dims: []const i64,
    inserted_window_dims: []const i64,
    input_batching_dims: []const i64,
    scatter_indices_batching_dims: []const i64,
    scatter_dims_to_operand_dims: []const i64,
    index_vector_dim: i64,
    indices_are_sorted: bool = false,
    unique_indices: bool = false,

    pub fn getScatterDimensionNumbers(self: ScatterArgs, ctx: mlir.Context) mlir.Attribute {
        return mlir.Attribute.wrap(
            c.stablehloScatterDimensionNumbersGet(
                ctx.inner(),
                @intCast(self.update_window_dims.len),
                self.update_window_dims.ptr,
                @intCast(self.inserted_window_dims.len),
                self.inserted_window_dims.ptr,
                @intCast(self.input_batching_dims.len),
                self.input_batching_dims.ptr,
                @intCast(self.scatter_indices_batching_dims.len),
                self.scatter_indices_batching_dims.ptr,
                @intCast(self.scatter_dims_to_operand_dims.len),
                self.scatter_dims_to_operand_dims.ptr,
                self.index_vector_dim,
            ),
        );
    }
};

pub fn scatter(
    ctx: mlir.Context,
    inputs: []const mlir.Value,
    scatter_indices: []const mlir.Value,
    updates: []const mlir.Value,
    update_block: mlir.Block,
    args: ScatterArgs,
    location: mlir.Location,
) mlir.Operation {
    return mlir.Operation.make(
        ctx,
        "stablehlo.scatter",
        .{
            .variadic_operands = &.{ inputs, scatter_indices, updates },
            .blocks = &.{update_block},
            .attributes = &.{
                .{ "scatter_dimension_numbers", args.getScatterDimensionNumbers(ctx) },
                .{ "indices_are_sorted", .boolean(ctx, args.indices_are_sorted) },
                .{ "unique_indices", .boolean(ctx, args.unique_indices) },
            },
            .result_type_inference = true,
            .location = location,
        },
    );
}

pub fn iota(ctx: mlir.Context, dimension: i64, result_type: mlir.Type, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.iota", .{
        .operands = &.{},
        .results = &.{result_type},
        .attributes = &.{
            .{ "iota_dimension", .int(ctx, .i64, dimension) },
        },
        .location = location,
    });
}

pub fn reverse(ctx: mlir.Context, operand: mlir.Value, dimensions: []const i64, location: mlir.Location) mlir.Operation {
    const result_type = operand.getType();
    return mlir.Operation.make(ctx, "stablehlo.reverse", .{
        .operands = &.{operand},
        .results = &.{result_type},
        .attributes = &.{
            .{ "dimensions", .dense(ctx, .i64, dimensions) },
        },
        .location = location,
    });
}

pub fn compare(ctx: mlir.Context, lhs: mlir.Value, rhs: mlir.Value, comparison_direction: ComparisonDirection, compare_type: CompareType, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.compare", .{
        .operands = &.{ lhs, rhs },
        .result_type_inference = true,
        .attributes = &.{
            .{ "comparison_direction", comparison_direction.as(mlir.Attribute) },
            .{ "compare_type", compare_type.as(mlir.Attribute) },
        },
        .location = location,
    });
}

pub fn reduce(
    ctx: mlir.Context,
    inputs: []const mlir.Value,
    init_values: []const mlir.Value,
    dimensions: []const i64,
    blkctx: anytype,
    blkfn: fn (anytype, mlir.Context, []const mlir.Value, []const mlir.Value) mlir.Operation,
    location: mlir.Location,
) mlir.Operation {
    const MaxBlockArguments = 32;

    const block_n_args = inputs.len + init_values.len;
    const locations = ([_]mlir.Location{mlir.Location.unknown(ctx)} ** MaxBlockArguments)[0..block_n_args];
    var reduce_elem_types: [MaxBlockArguments]mlir.Type = undefined;
    for (inputs, 0..) |input, i| {
        const arg_type: mlir.Type = .tensor(&.{}, elementTypeOrSelf(input.getType()));
        reduce_elem_types[i] = arg_type;
        reduce_elem_types[inputs.len + i] = arg_type;
    }
    var block = mlir.Block.open(reduce_elem_types[0..block_n_args], locations) catch unreachable;
    {
        defer block.close();

        var block_inputs: [MaxBlockArguments / 2]mlir.Value = undefined;
        var block_accs: [MaxBlockArguments / 2]mlir.Value = undefined;
        for (0..inputs.len) |i| {
            block_inputs[i] = block.argument(i);
            block_accs[i] = block.argument(inputs.len + i);
        }
        _ = blkfn(blkctx, ctx, block_inputs[0..inputs.len], block_accs[0..init_values.len]);
    }

    return mlir.Operation.make(ctx, "stablehlo.reduce", .{
        .variadic_operands = &.{ inputs, init_values },
        .result_type_inference = true,
        .block = block,
        .attributes = &.{
            .{ "dimensions", .dense(ctx, .i64, dimensions) },
        },
        .location = location,
    });
}

pub fn sort(
    ctx: mlir.Context,
    inputs: []const mlir.Value,
    dimension: i64,
    is_stable: bool,
    blkctx: anytype,
    compfn: fn (anytype, mlir.Context, []const mlir.Value) mlir.Operation,
    location: mlir.Location,
) mlir.Operation {
    const MaxBlockArguments = 32;

    const locations = ([_]mlir.Location{mlir.Location.unknown(ctx)} ** MaxBlockArguments)[0 .. inputs.len * 2];
    var sort_elem_types: [MaxBlockArguments]mlir.Type = undefined;
    for (inputs, 0..) |input, i| {
        const arg_type: mlir.Type = .tensor(&.{}, elementTypeOrSelf(input.getType()));
        sort_elem_types[i * 2] = arg_type;
        sort_elem_types[i * 2 + 1] = arg_type;
    }
    var block = mlir.Block.init(sort_elem_types[0 .. inputs.len * 2], locations) catch unreachable;

    var block_inputs: [MaxBlockArguments]mlir.Value = undefined;
    for (0..inputs.len * 2) |i| {
        block_inputs[i] = block.argument(i);
    }
    _ = compfn(blkctx, ctx, block_inputs[0 .. inputs.len * 2]);

    return mlir.Operation.make(ctx, "stablehlo.sort", .{
        .variadic_operands = &.{inputs},
        .result_type_inference = true,
        .block = block,
        .attributes = &.{
            .{ "dimension", .int(ctx, .i64, dimension) },
            .{ "is_stable", .boolean(ctx, is_stable) },
        },
        .location = location,
    });
}

pub fn dynamic_slice(ctx: mlir.Context, operand: mlir.Value, new_dims: []const i64, start_indices: []const mlir.Value, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.dynamic_slice", .{
        .variadic_operands = &.{ &.{operand}, start_indices },
        .result_type_inference = true,
        .attributes = &.{
            .{ "slice_sizes", .dense(ctx, .i64, new_dims) },
        },
        .location = location,
    });
}

pub fn round_nearest_afz(ctx: mlir.Context, value: mlir.Value, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.round_nearest_afz", .{
        .operands = &.{value},
        .result_type_inference = true,
        .location = location,
    });
}

pub fn round_nearest_even(ctx: mlir.Context, value: mlir.Value, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.round_nearest_even", .{
        .operands = &.{value},
        .result_type_inference = true,
        .location = location,
    });
}

pub const PadOpts = struct {
    low: []const i64,
    high: []const i64,
    interior: []const i64,
};

pub fn pad(ctx: mlir.Context, value: mlir.Value, padding_value: mlir.Value, opts: PadOpts, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.pad", .{
        .operands = &.{ value, padding_value },
        .result_type_inference = true,
        .attributes = &.{
            .{ "edge_padding_low", .dense(ctx, .i64, opts.low) },
            .{ "edge_padding_high", .dense(ctx, .i64, opts.high) },
            .{ "interior_padding", .dense(ctx, .i64, opts.interior) },
        },
        .location = location,
    });
}

pub const TriangularSolveOpts = struct {
    left_side: bool,
    lower: bool,
    unit_diagonal: bool,
    transpose_a: Transpose.Type,
};

pub fn triangular_solve(ctx: mlir.Context, value: mlir.Value, other: mlir.Value, location: mlir.Location, opts: TriangularSolveOpts) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.triangular_solve", .{
        .operands = &.{ value, other },
        .result_type_inference = true,
        .attributes = &.{
            .{ "left_side", .i1FromBool(ctx, opts.left_side) },
            .{ "lower", .i1FromBool(ctx, opts.lower) },
            .{ "unit_diagonal", .i1FromBool(ctx, opts.unit_diagonal) },
            .{ "transpose_a", Transpose.init(ctx, opts.transpose_a).as(mlir.Attribute) },
        },
        .location = location,
    });
}

pub const FftOpts = struct {
    kind: FftType.Type,
    length: []const i64,
};

pub fn fft(ctx: mlir.Context, value: mlir.Value, location: mlir.Location, opts: FftOpts) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.fft", .{
        .operands = &.{value},
        .result_type_inference = true,
        .attributes = &.{
            .{ "fft_type", FftType.init(ctx, opts.kind).as(mlir.Attribute) },
            .{ "fft_length", .dense(ctx, .i64, opts.length) },
        },
        .location = location,
    });
}

pub fn rng(ctx: mlir.Context, a: mlir.Value, b: mlir.Value, shape: mlir.Value, rng_distribution: RngDistribution.Type, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.rng", .{
        .operands = &.{ a, b, shape },
        .result_type_inference = true,
        .attributes = &.{
            .{ "rng_distribution", RngDistribution.init(ctx, rng_distribution).as(mlir.Attribute) },
        },
        .location = location,
    });
}

pub fn rng_bit_generator(ctx: mlir.Context, rng_algorithm: RngAlgorithm.Type, initial_state: mlir.Value, res_state_type: mlir.Type, res_type: mlir.Type, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.rng_bit_generator", .{
        .operands = &.{initial_state},
        .results = &.{ res_state_type, res_type },
        .attributes = &.{
            .{ "rng_algorithm", RngAlgorithm.init(ctx, rng_algorithm).as(mlir.Attribute) },
        },
        .location = location,
    });
}

pub fn reduce_precision(ctx: mlir.Context, value: mlir.Value, exponent_bits: i32, mantissa_bits: i32, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.reduce_precision", .{
        .operands = &.{value},
        .result_type_inference = true,
        .attributes = &.{
            .{ "exponent_bits", .int(ctx, .i32, exponent_bits) },
            .{ "mantissa_bits", .int(ctx, .i32, mantissa_bits) },
        },
        .location = location,
    });
}

pub fn dynamic_update_slice(ctx: mlir.Context, operand: mlir.Value, update: mlir.Value, start_indices: []const mlir.Value, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.dynamic_update_slice", .{
        .variadic_operands = &.{ &.{operand}, &.{update}, start_indices },
        .result_type_inference = true,
        .location = location,
    });
}

pub fn tuple(ctx: mlir.Context, values: []const mlir.Value, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.tuple", .{
        .operands = values,
        .result_type_inference = true,
        .location = location,
    });
}

pub fn get_tuple_element(ctx: mlir.Context, tuple_value: mlir.Value, index: i64, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.get_tuple_element", .{
        .operands = &.{tuple_value},
        .result_type_inference = true,
        .attributes = &.{
            .{ "index", .int(ctx, .i32, index) },
        },
        .location = location,
    });
}

pub const ConvolutionOpts = struct {
    window_strides: []const i64,
    pad_value: []const i64,
    pad_shape: []const i64 = &.{},
    lhs_dilation: []const i64,
    rhs_dilation: []const i64,
    window_reversal: []const bool,
    input_batch_dimension: i64,
    input_feature_dimension: i64,
    input_spatial_dimensions: []const i64,
    kernel_input_feature_dimension: i64,
    kernel_output_feature_dimension: i64,
    kernel_spatial_dimensions: []const i64,
    output_batch_dimension: i64,
    output_feature_dimension: i64,
    output_spatial_dimensions: []const i64,
    feature_group_count: i64,
    batch_group_count: i64,
    precision_config: []const PrecisionAttribute.Precision = &.{},
};

pub fn convolution(
    ctx: mlir.Context,
    lhs: mlir.Value,
    rhs: mlir.Value,
    opts: ConvolutionOpts,
    res_type: mlir.Type,
    location: mlir.Location,
) mlir.Operation {
    var max_precisions: [2]mlir.Attribute = undefined;
    for (opts.precision_config, 0..) |p, i| {
        max_precisions[i] = PrecisionAttribute.init(ctx, p).as(mlir.Attribute);
    }
    var window_reversal: [3]i32 = undefined;
    for (opts.window_reversal, 0..) |w, i| {
        window_reversal[i] = @intCast(@intFromBool(w));
    }
    return mlir.Operation.make(ctx, "stablehlo.convolution", .{
        .operands = &.{ lhs, rhs },
        .results = &.{res_type},
        .attributes = &.{
            .{ "window_strides", .dense(ctx, .i64, opts.window_strides) },
            .{ "padding", .denseElements(ctx, opts.pad_shape, .i64, opts.pad_value) },
            .{ "lhs_dilation", .dense(ctx, .i64, opts.lhs_dilation) },
            .{ "rhs_dilation", .dense(ctx, .i64, opts.rhs_dilation) },
            .{ "window_reversal", .dense(ctx, .bool, window_reversal[0..opts.window_reversal.len]) },
            .{
                "dimension_numbers", ConvDimensionNumbersAttribute.init(ctx, .{
                    .input_batch_dimension = opts.input_batch_dimension,
                    .input_feature_dimension = opts.input_feature_dimension,
                    .input_spatial_dimensions = opts.input_spatial_dimensions,
                    .kernel_input_feature_dimension = opts.kernel_input_feature_dimension,
                    .kernel_output_feature_dimension = opts.kernel_output_feature_dimension,
                    .kernel_spatial_dimensions = opts.kernel_spatial_dimensions,
                    .output_batch_dimension = opts.output_batch_dimension,
                    .output_feature_dimension = opts.output_feature_dimension,
                    .output_spatial_dimensions = opts.output_spatial_dimensions,
                }).as(mlir.Attribute),
            },
            .{ "feature_group_count", .int(ctx, .i64, opts.feature_group_count) },
            .{ "batch_group_count", .int(ctx, .i64, opts.batch_group_count) },
            .{ "precision_config", .array(ctx, &max_precisions) },
        },
        .location = location,
    });
}

pub const CustomCallOpts = struct {
    pub const ApiVersion = enum(i32) {
        original = 1,
        status_returning = 2,
        status_returning_unified = 3,
        typed_ffi = 4,
    };

    call_target_name: [:0]const u8,
    has_side_effect: bool,
    backend_config: ?mlir.Attribute,
    operand_layouts: ?[]const []const usize = null,
    result_layouts: ?[]const []const usize = null,
    output_operand_aliases: []const i64 = &.{},
    additional_attributes: []const mlir.AttrTuple = &.{},
    api_version: ApiVersion,
};

pub fn custom_call(ctx: mlir.Context, inputs: []const mlir.Value, opts: CustomCallOpts, res_types: []const mlir.Type, location: mlir.Location) mlir.Operation {
    const MAX_OPERANDS = 64;
    const MAX_RESULTS = 16;

    const backend_config = opts.backend_config orelse mlir.Attribute.string(ctx, "");
    if (@intFromEnum(opts.api_version) < @intFromEnum(CustomCallOpts.ApiVersion.typed_ffi)) {
        stdx.debug.assert(
            backend_config.is_a(mlir.StringAttribute),
            "API version < 4 requires a string as backend_config, got {}",
            .{backend_config},
        );
    } else {
        stdx.debug.assert(
            backend_config.is_a(mlir.DictionaryAttribute),
            "API version >= 4 requires a dictionary as backend_config, got {}",
            .{backend_config},
        );
    }

    var attrs: std.BoundedArray(mlir.AttrTuple, 32) = .{};
    attrs.appendSliceAssumeCapacity(&[_]mlir.AttrTuple{
        .{ "api_version", .int(ctx, .i32, @intFromEnum(opts.api_version)) },
        .{ "call_target_name", .string(ctx, opts.call_target_name) },
        .{ "has_side_effect", .boolean(ctx, opts.has_side_effect) },
        .{ "backend_config", backend_config },
    });

    {
        var output_operand_aliases: std.BoundedArray(mlir.Attribute, MAX_RESULTS) = .{};
        for (opts.output_operand_aliases) |alias| {
            output_operand_aliases.appendAssumeCapacity(
                OutputOperandAliasAttribute.init(ctx, &.{}, alias, &.{}).as(mlir.Attribute),
            );
        }
        attrs.appendAssumeCapacity(.{ "output_operand_aliases", .array(ctx, output_operand_aliases.constSlice()) });
    }

    if (opts.operand_layouts) |layouts| {
        var operand_layouts: std.BoundedArray(mlir.Attribute, MAX_OPERANDS) = .{};
        for (layouts) |ol| {
            operand_layouts.appendAssumeCapacity(.denseElements(ctx, &.{@intCast(ol.len)}, .index, ol));
        }
        attrs.appendAssumeCapacity(.{ "operand_layouts", .array(ctx, operand_layouts.constSlice()) });
    }

    if (opts.result_layouts) |layouts| {
        var result_layouts: std.BoundedArray(mlir.Attribute, MAX_RESULTS) = .{};
        for (layouts) |rl| {
            result_layouts.appendAssumeCapacity(.denseElements(ctx, &.{@intCast(rl.len)}, .index, rl));
        }
        attrs.appendAssumeCapacity(.{ "result_layouts", .array(ctx, result_layouts.constSlice()) });
    }

    attrs.appendSlice(opts.additional_attributes) catch @panic("Too many additional_attributes");

    return mlir.Operation.make(ctx, "stablehlo.custom_call", .{
        .operands = inputs,
        .results = res_types,
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

pub const DotDimensionNumbersAttribute = struct {
    _inner: c.MlirAttribute,

    pub usingnamespace mlir.MlirHelpers(DotDimensionNumbersAttribute, .{
        .is_a_fn = c.stablehloAttributeIsADotDimensionNumbers,
        .is_null_fn = c.mlirAttributeIsNull,
        .dump_fn = c.mlirAttributeDump,
        .equal_fn = c.mlirAttributeEqual,
    });
    const Self = DotDimensionNumbersAttribute;

    pub fn init(ctx: mlir.Context, args: struct {
        lhs_batching_dimensions: []const i64,
        rhs_batching_dimensions: []const i64,
        lhs_contracting_dimensions: []const i64,
        rhs_contracting_dimensions: []const i64,
    }) Self {
        return Self.wrap(
            c.stablehloDotDimensionNumbersGet(
                ctx.inner(),
                @intCast(args.lhs_batching_dimensions.len),
                args.lhs_batching_dimensions.ptr,
                @intCast(args.rhs_batching_dimensions.len),
                args.rhs_batching_dimensions.ptr,
                @intCast(args.lhs_contracting_dimensions.len),
                args.lhs_contracting_dimensions.ptr,
                @intCast(args.rhs_contracting_dimensions.len),
                args.rhs_contracting_dimensions.ptr,
            ),
        );
    }

    pub fn getLhsBatchingDimensionsSize(self: Self) usize {
        return @intCast(c.stablehloDotDimensionNumbersGetLhsBatchingDimensionsSize(self.inner()));
    }

    pub fn getLhsBatchingDimensionsElem(self: Self, pos: usize) i64 {
        return c.stablehloDotDimensionNumbersGetLhsBatchingDimensionsElem(self.inner(), @intCast(pos));
    }

    pub fn getRhsBatchingDimensionsSize(self: Self) usize {
        return @intCast(c.stablehloDotDimensionNumbersGetRhsBatchingDimensionsSize(self.inner()));
    }

    pub fn getRhsBatchingDimensionsElem(self: Self, pos: usize) i64 {
        return c.stablehloDotDimensionNumbersGetRhsBatchingDimensionsElem(self.inner(), @intCast(pos));
    }

    pub fn getLhsContractingDimensionsSize(self: Self) usize {
        return @intCast(c.stablehloDotDimensionNumbersGetLhsContractingDimensionsSize(self.inner()));
    }

    pub fn getLhsContractingDimensionsElem(self: Self, pos: usize) i64 {
        return c.stablehloDotDimensionNumbersGetLhsContractingDimensionsElem(self.inner(), @intCast(pos));
    }

    pub fn getRhsContractingDimensionsSize(self: Self) usize {
        return @intCast(c.stablehloDotDimensionNumbersGetRhsContractingDimensionsSize(self.inner()));
    }

    pub fn getRhsContractingDimensionsElem(self: Self, pos: usize) i64 {
        return c.stablehloDotDimensionNumbersGetRhsContractingDimensionsElem(self.inner(), @intCast(pos));
    }
};

pub const GatherDimensionNumbersAttribute = struct {
    _inner: c.MlirAttribute,

    pub usingnamespace mlir.MlirHelpers(GatherDimensionNumbersAttribute, .{
        .is_a_fn = c.stablehloAttributeIsAGatherDimensionNumbers,
        .is_null_fn = c.mlirAttributeIsNull,
        .dump_fn = c.mlirAttributeDump,
        .equal_fn = c.mlirAttributeEqual,
    });
    const Self = GatherDimensionNumbersAttribute;

    pub fn init(
        ctx: mlir.Context,
        offset_dims: []const i64,
        collapsed_slice_dims: []const i64,
        operand_batching_dims: []const i64,
        start_indices_batching_dims: []const i64,
        start_index_map: []const i64,
        index_vector_dim: i64,
    ) Self {
        return Self.wrap(
            c.stablehloGatherDimensionNumbersGet(
                ctx.inner(),
                @intCast(offset_dims.len),
                offset_dims.ptr,
                @intCast(collapsed_slice_dims.len),
                collapsed_slice_dims.ptr,
                @intCast(operand_batching_dims.len),
                operand_batching_dims.ptr,
                @intCast(start_indices_batching_dims.len),
                start_indices_batching_dims.ptr,
                @intCast(start_index_map.len),
                start_index_map.ptr,
                index_vector_dim,
            ),
        );
    }

    pub fn getOffsetDimsSize(self: Self) usize {
        return @intCast(c.stablehloGatherDimensionNumbersGetOffsetDimsSize(self.inner()));
    }

    pub fn getOffsetDimsElem(self: Self, pos: usize) i64 {
        return c.stablehloGatherDimensionNumbersGetOffsetDimsElem(self.inner(), @intCast(pos));
    }

    pub fn getCollapsedSliceDimsSize(self: Self) usize {
        return @intCast(c.stablehloGatherDimensionNumbersGetCollapsedSliceDimsSize(self.inner()));
    }

    pub fn getCollapsedSliceDimsElem(self: Self, pos: usize) i64 {
        return c.stablehloGatherDimensionNumbersGetCollapsedSliceDimsElem(self.inner(), @intCast(pos));
    }

    pub fn getStartIndexMapSize(self: Self) usize {
        return @intCast(c.stablehloGatherDimensionNumbersGetStartIndexMapSize(self.inner()));
    }

    pub fn getOperandBatchingDimsSize(self: Self) usize {
        return @intCast(c.stablehloGatherDimensionNumbersGetOperandBatchingDimsSize(self.inner()));
    }

    pub fn getOperandBatchingDimsElem(self: Self, pos: usize) i64 {
        return c.stablehloGatherDimensionNumbersGetOperandBatchingDimsElem(self.inner(), @intCast(pos));
    }

    pub fn getStartIndicesBatchingDimsSize(self: Self) usize {
        return @intCast(c.stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsSize(self.inner()));
    }

    pub fn getStartIndicesBatchingDimsElem(self: Self, pos: usize) i64 {
        return c.stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsElem(self.inner(), @intCast(pos));
    }

    pub fn getStartIndexMapElem(self: Self, pos: usize) i64 {
        return c.stablehloGatherDimensionNumbersGetStartIndexMapElem(self.inner(), @intCast(pos));
    }

    pub fn getIndexVectorDim(self: Self) usize {
        return @intCast(c.stablehloGatherDimensionNumbersGetIndexVectorDim(self.inner()));
    }
};

pub const ConvDimensionNumbersAttribute = struct {
    _inner: c.MlirAttribute,

    pub usingnamespace mlir.MlirHelpers(ConvDimensionNumbersAttribute, .{
        .is_a_fn = c.stablehloAttributeIsAConvDimensionNumbers,
        .is_null_fn = c.mlirAttributeIsNull,
        .dump_fn = c.mlirAttributeDump,
        .equal_fn = c.mlirAttributeEqual,
    });
    const Self = ConvDimensionNumbersAttribute;

    pub fn init(ctx: mlir.Context, args: struct {
        input_batch_dimension: i64,
        input_feature_dimension: i64,
        input_spatial_dimensions: []const i64,
        kernel_input_feature_dimension: i64,
        kernel_output_feature_dimension: i64,
        kernel_spatial_dimensions: []const i64,
        output_batch_dimension: i64,
        output_feature_dimension: i64,
        output_spatial_dimensions: []const i64,
    }) Self {
        return Self.wrap(
            c.stablehloConvDimensionNumbersGet(
                ctx.inner(),
                args.input_batch_dimension,
                args.input_feature_dimension,
                @intCast(args.input_spatial_dimensions.len),
                args.input_spatial_dimensions.ptr,
                args.kernel_input_feature_dimension,
                args.kernel_output_feature_dimension,
                @intCast(args.kernel_spatial_dimensions.len),
                args.kernel_spatial_dimensions.ptr,
                args.output_batch_dimension,
                args.output_feature_dimension,
                @intCast(args.output_spatial_dimensions.len),
                args.output_spatial_dimensions.ptr,
            ),
        );
    }

    pub fn getInputBatchDimension(self: Self) i64 {
        return c.stablehloConvDimensionNumbersGetInputBatchDimension(self.inner());
    }

    pub fn getInputFeatureDimension(self: Self) i64 {
        return c.stablehloConvDimensionNumbersGetInputFeatureDimension(self.inner());
    }

    pub fn getInputSpatialDimensionsSize(self: Self) usize {
        return @intCast(c.stablehloConvDimensionNumbersGetInputSpatialDimensionsSize(self.inner()));
    }

    pub fn getInputSpatialDimensionsElem(self: Self, pos: usize) i64 {
        return c.stablehloConvDimensionNumbersGetInputSpatialDimensionsElem(self.inner(), @intCast(pos));
    }

    pub fn getKernelInputFeatureDimension(self: Self) i64 {
        return c.stablehloConvDimensionNumbersGetKernelInputFeatureDimension(self.inner());
    }

    pub fn getKernelOutputFeatureDimension(self: Self) i64 {
        return c.stablehloConvDimensionNumbersGetKernelOutputFeatureDimension(self.inner());
    }

    pub fn getKernelSpatialDimensionsSize(self: Self) usize {
        return @intCast(c.stablehloConvDimensionNumbersGetKernelSpatialDimensionsSize(self.inner()));
    }

    pub fn getKernelSpatialDimensionsElem(self: Self, pos: usize) i64 {
        return c.stablehloConvDimensionNumbersGetKernelSpatialDimensionsElem(self.inner(), @intCast(pos));
    }

    pub fn getOutputBatchDimension(self: Self) i64 {
        return c.stablehloConvDimensionNumbersGetOutputBatchDimension(self.inner());
    }

    pub fn getOutputFeatureDimension(self: Self) i64 {
        return c.stablehloConvDimensionNumbersGetOutputFeatureDimension(self.inner());
    }

    pub fn getOutputSpatialDimensionsSize(self: Self) usize {
        return @intCast(c.stablehloConvDimensionNumbersGetOutputSpatialDimensionsSize(self.inner()));
    }

    pub fn getOutputSpatialDimensionsElem(self: Self, pos: usize) i64 {
        return c.stablehloConvDimensionNumbersGetOutputSpatialDimensionsElem(self.inner(), @intCast(pos));
    }
};

pub const OutputOperandAliasAttribute = struct {
    _inner: c.MlirAttribute,

    pub usingnamespace mlir.MlirHelpers(OutputOperandAliasAttribute, .{
        .is_a_fn = c.stablehloAttributeIsAOutputOperandAlias,
        .is_null_fn = c.mlirAttributeIsNull,
        .dump_fn = c.mlirAttributeDump,
        .equal_fn = c.mlirAttributeEqual,
    });

    pub fn init(
        ctx: mlir.Context,
        output_tuple_indices: []const i64,
        operand_index: i64,
        operand_tuple_indices: []const i64,
    ) OutputOperandAliasAttribute {
        return OutputOperandAliasAttribute.wrap(c.stablehloOutputOperandAliasGet(
            ctx.inner(),
            @intCast(output_tuple_indices.len),
            output_tuple_indices.ptr,
            @intCast(operand_index),
            @intCast(operand_tuple_indices.len),
            operand_tuple_indices.ptr,
        ));
    }
};

pub const PrecisionAttribute = struct {
    _inner: c.MlirAttribute,

    pub usingnamespace mlir.MlirHelpers(PrecisionAttribute, .{
        .is_a_fn = c.stablehloAttributeIsAPrecisionAttr,
        .is_null_fn = c.mlirAttributeIsNull,
        .dump_fn = c.mlirAttributeDump,
        .equal_fn = c.mlirAttributeEqual,
    });
    const Self = PrecisionAttribute;

    pub const Precision = enum {
        DEFAULT,
        HIGH,
        HIGHEST,
    };

    pub fn init(ctx: mlir.Context, value: Precision) Self {
        return Self.wrap(c.stablehloPrecisionAttrGet(ctx.inner(), mlir.stringRef(@tagName(value))));
    }

    pub fn getValue(self: Self) Precision {
        const value = mlir.fromStringRef(c.stablehloPrecisionAttrGetValue(self.inner()));
        return std.meta.stringToEnum(Precision, value) orelse unreachable;
    }
};

pub const ComparisonDirection = struct {
    _inner: c.MlirAttribute,

    pub usingnamespace mlir.MlirHelpers(ComparisonDirection, .{
        .is_a_fn = c.stablehloAttributeIsAComparisonDirectionAttr,
        .is_null_fn = c.mlirAttributeIsNull,
        .dump_fn = c.mlirAttributeDump,
        .equal_fn = c.mlirAttributeEqual,
    });
    const Self = ComparisonDirection;

    pub const Direction = enum {
        EQ,
        NE,
        GE,
        GT,
        LE,
        LT,
    };

    pub fn init(ctx: mlir.Context, value: Direction) Self {
        return Self.wrap(c.stablehloComparisonDirectionAttrGet(ctx.inner(), mlir.stringRef(@tagName(value))));
    }

    pub fn getValue(self: Self) Direction {
        const value = mlir.fromStringRef(c.stablehloComparisonDirectionAttrGetValue(self.inner()));
        return std.meta.stringToEnum(Direction, value) orelse unreachable;
    }
};

pub const CompareType = struct {
    _inner: c.MlirAttribute,

    pub usingnamespace mlir.MlirHelpers(CompareType, .{
        .is_a_fn = c.stablehloAttributeIsAComparisonTypeAttr,
        .is_null_fn = c.mlirAttributeIsNull,
        .dump_fn = c.mlirAttributeDump,
        .equal_fn = c.mlirAttributeEqual,
    });
    const Self = CompareType;

    pub const Type = enum {
        SIGNED,
        UNSIGNED,
        FLOAT,
        TOTALORDER,
    };

    pub fn init(ctx: mlir.Context, value: Type) Self {
        return Self.wrap(c.stablehloComparisonTypeAttrGet(ctx.inner(), mlir.stringRef(@tagName(value))));
    }

    pub fn getValue(self: Self) Type {
        const value = mlir.fromStringRef(c.stablehloComparisonTypeAttrGetValue(self.inner()));
        return std.meta.stringToEnum(Type, value) orelse unreachable;
    }
};

pub const Transpose = struct {
    _inner: c.MlirAttribute,

    pub usingnamespace mlir.MlirHelpers(Transpose, .{
        .is_a_fn = c.stablehloAttributeIsATransposeAttr,
        .is_null_fn = c.mlirAttributeIsNull,
        .dump_fn = c.mlirAttributeDump,
        .equal_fn = c.mlirAttributeEqual,
    });
    const Self = Transpose;

    pub const Type = enum {
        NO_TRANSPOSE,
        TRANSPOSE,
        ADJOINT,
    };

    pub fn init(ctx: mlir.Context, value: Type) Self {
        return Self.wrap(c.stablehloTransposeAttrGet(ctx.inner(), mlir.stringRef(@tagName(value))));
    }

    pub fn getValue(self: Self) Type {
        const value = mlir.fromStringRef(c.stablehloTransposeAttrGetValue(self.inner()));
        return std.meta.stringToEnum(Type, value) orelse unreachable;
    }
};

pub const FftType = struct {
    _inner: c.MlirAttribute,

    pub usingnamespace mlir.MlirHelpers(FftType, .{
        .is_a_fn = c.stablehloAttributeIsAFftTypeAttr,
        .is_null_fn = c.mlirAttributeIsNull,
        .dump_fn = c.mlirAttributeDump,
        .equal_fn = c.mlirAttributeEqual,
    });
    const Self = FftType;

    pub const Type = enum {
        FFT,
        IFFT,
        RFFT,
        IRFFT,
    };

    pub fn init(ctx: mlir.Context, value: Type) Self {
        return Self.wrap(c.stablehloFftTypeAttrGet(ctx.inner(), mlir.stringRef(@tagName(value))));
    }

    pub fn getValue(self: Self) Type {
        const value = mlir.fromStringRef(c.stablehloFftTypeAttrGetValue(self.inner()));
        return std.meta.stringToEnum(Type, value) orelse unreachable;
    }
};

pub const RngDistribution = struct {
    _inner: c.MlirAttribute,

    pub usingnamespace mlir.MlirHelpers(RngDistribution, .{
        .is_a_fn = c.stablehloAttributeIsARngDistributionAttr,
        .is_null_fn = c.mlirAttributeIsNull,
        .dump_fn = c.mlirAttributeDump,
        .equal_fn = c.mlirAttributeEqual,
    });
    const Self = RngDistribution;

    pub const Type = enum {
        UNIFORM,
        NORMAL,
    };

    pub fn init(ctx: mlir.Context, value: Type) Self {
        return Self.wrap(c.stablehloRngDistributionAttrGet(ctx.inner(), mlir.stringRef(@tagName(value))));
    }

    pub fn getValue(self: Self) Type {
        const value = mlir.fromStringRef(c.stablehloRngDistributionAttrGetValue(self.inner()));
        return std.meta.stringToEnum(Type, value) orelse unreachable;
    }
};

pub const RngAlgorithm = struct {
    _inner: c.MlirAttribute,

    pub usingnamespace mlir.MlirHelpers(RngAlgorithm, .{
        .is_a_fn = c.stablehloAttributeIsARngAlgorithmAttr,
        .is_null_fn = c.mlirAttributeIsNull,
        .dump_fn = c.mlirAttributeDump,
        .equal_fn = c.mlirAttributeEqual,
    });
    const Self = RngAlgorithm;

    pub const Type = enum {
        DEFAULT,
        THREE_FRY,
        PHILOX,
    };

    pub fn init(ctx: mlir.Context, value: Type) Self {
        return Self.wrap(c.stablehloRngAlgorithmAttrGet(ctx.inner(), mlir.stringRef(@tagName(value))));
    }

    pub fn getValue(self: Self) Type {
        const value = mlir.fromStringRef(c.stablehloRngAlgorithmAttrGetValue(self.inner()));
        return std.meta.stringToEnum(Type, value) orelse unreachable;
    }
};

pub fn stablehloVersionFromCompatibilityRequirement(requirement: c.MlirStablehloCompatibilityRequirement) []const u8 {
    const state = struct {
        var buf: [32]u8 = undefined;

        fn call(req: c.MlirStablehloCompatibilityRequirement) []u8 {
            var stream = std.io.fixedBufferStream(&buf);
            var context = .{ .writer = stream.writer() };
            const WriterContext = @TypeOf(context);

            c.stablehloVersionFromCompatibilityRequirement(req, (struct {
                pub fn callback(mlir_str: c.MlirStringRef, userdata: ?*anyopaque) callconv(.C) void {
                    const inner_ctx: *WriterContext = @ptrCast(@alignCast(userdata));
                    _ = inner_ctx.writer.write(mlir.fromStringRef(mlir_str)) catch unreachable;
                }
            }).callback, &context);

            return buf[0..stream.pos];
        }
    };

    return state.call(requirement);
}

pub fn stablehloGetSmallerVersion(version1: []const u8, version2: []const u8) []const u8 {
    var buf: [32]u8 = undefined;

    var stream = std.io.fixedBufferStream(&buf);
    var context = .{ .writer = stream.writer() };
    const WriterContext = @TypeOf(context);

    _ = c.stablehloGetSmallerVersion(mlir.stringRef(version1), mlir.stringRef(version2), (struct {
        pub fn callback(mlir_str: c.MlirStringRef, userdata: ?*anyopaque) callconv(.C) void {
            const inner_ctx: *WriterContext = @ptrCast(@alignCast(userdata));
            _ = inner_ctx.writer.write(mlir.fromStringRef(mlir_str)) catch unreachable;
        }
    }).callback, &context);

    return if (std.mem.eql(u8, buf[0..stream.pos], version1)) version1 else version2;
}

pub fn getCurrentVersion() []const u8 {
    const state = struct {
        var buf: [32]u8 = undefined;
        var str: []const u8 = undefined;
        var once = std.once(call);

        fn call() void {
            var stream = std.io.fixedBufferStream(&buf);
            var writer_ = stream.writer();
            const ContextWriter = @TypeOf(writer_);

            c.stablehloGetCurrentVersion((struct {
                pub fn callback(mlir_str: c.MlirStringRef, userdata: ?*anyopaque) callconv(.C) void {
                    const writer: *ContextWriter = @ptrCast(@alignCast(userdata));
                    _ = writer.write(mlir.fromStringRef(mlir_str)) catch unreachable;
                }
            }).callback, &writer_);

            str = buf[0..stream.pos];
        }
    };

    state.once.call();
    return state.str;
}

pub fn getMinimumVersion() []const u8 {
    const state = struct {
        var buf: [32]u8 = undefined;
        var str: []const u8 = undefined;
        var once = std.once(call);

        fn call() void {
            var stream = std.io.fixedBufferStream(&buf);
            var context = .{ .writer = stream.writer() };
            const WriterContext = @TypeOf(context);

            c.stablehloGetMinimumVersion((struct {
                pub fn callback(mlir_str: c.MlirStringRef, userdata: ?*anyopaque) callconv(.C) void {
                    const inner_ctx: *WriterContext = @ptrCast(@alignCast(userdata));
                    _ = inner_ctx.writer.write(mlir.fromStringRef(mlir_str)) catch unreachable;
                }
            }).callback, &context);

            str = buf[0..stream.pos];
        }
    };

    state.once.call();
    return state.str;
}

pub fn serializePortableArtifact(bytecode: []const u8, target_version: []const u8, writer: anytype) !void {
    var context = .{ .writer = writer };
    const WriterContext = @TypeOf(context);

    try mlir.successOr(c.stablehloSerializePortableArtifactFromStringRef(mlir.stringRef(bytecode), mlir.stringRef(target_version), (struct {
        pub fn callback(mlir_str: c.MlirStringRef, userdata: ?*anyopaque) callconv(.C) void {
            const inner_ctx: *WriterContext = @ptrCast(@alignCast(userdata));
            _ = inner_ctx.writer.write(mlir.fromStringRef(mlir_str)) catch unreachable;
        }
    }).callback, &context), error.InvalidMlirBytecodeVersion);
}
