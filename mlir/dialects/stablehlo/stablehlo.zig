const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stringRef = mlir.stringRef;
const stdx = @import("stdx");

pub const dialect_namespace = "stablehlo";

pub fn registerDialect(registry: *mlir.DialectRegistry) void {
    registry.registerDialect("stablehlo");
}

pub const operation_names: []const []const u8 = &.{
    "stablehlo.abs",
    "stablehlo.add",
    "stablehlo.all_reduce",
    "stablehlo.and",
    "stablehlo.atan2",
    "stablehlo.bitcast_convert",
    "stablehlo.broadcast_in_dim",
    "stablehlo.cbrt",
    "stablehlo.ceil",
    "stablehlo.cholesky",
    "stablehlo.clamp",
    "stablehlo.compare",
    "stablehlo.complex",
    "stablehlo.composite",
    "stablehlo.concatenate",
    "stablehlo.constant",
    "stablehlo.convert",
    "stablehlo.convolution",
    "stablehlo.cosine",
    "stablehlo.count_leading_zeros",
    "stablehlo.custom_call",
    "stablehlo.divide",
    "stablehlo.dot_general",
    "stablehlo.dynamic_slice",
    "stablehlo.dynamic_update_slice",
    "stablehlo.exponential",
    "stablehlo.exponential_minus_one",
    "stablehlo.fft",
    "stablehlo.floor",
    "stablehlo.gather",
    "stablehlo.get_tuple_element",
    "stablehlo.imag",
    "stablehlo.iota",
    "stablehlo.is_finite",
    "stablehlo.log",
    "stablehlo.log_plus_one",
    "stablehlo.logistic",
    "stablehlo.maximum",
    "stablehlo.minimum",
    "stablehlo.multiply",
    "stablehlo.negate",
    "stablehlo.not",
    "stablehlo.optimization_barrier",
    "stablehlo.or",
    "stablehlo.pad",
    "stablehlo.partition_id",
    "stablehlo.popcnt",
    "stablehlo.power",
    "stablehlo.real",
    "stablehlo.reduce_precision",
    "stablehlo.remainder",
    "stablehlo.reshape",
    "stablehlo.return",
    "stablehlo.reverse",
    "stablehlo.rng",
    "stablehlo.rng_bit_generator",
    "stablehlo.round_nearest_afz",
    "stablehlo.round_nearest_even",
    "stablehlo.rsqrt",
    "stablehlo.scatter",
    "stablehlo.select",
    "stablehlo.shift_left",
    "stablehlo.shift_right_arithmetic",
    "stablehlo.shift_right_logical",
    "stablehlo.sign",
    "stablehlo.sine",
    "stablehlo.slice",
    "stablehlo.sqrt",
    "stablehlo.subtract",
    "stablehlo.tanh",
    "stablehlo.transpose",
    "stablehlo.triangular_solve",
    "stablehlo.tuple",
    "stablehlo.while",
    "stablehlo.xor",
};

pub const abs = unaryOp("stablehlo.abs").call;
pub const cosine = unaryOp("stablehlo.cosine").call;
pub const sine = unaryOp("stablehlo.sine").call;
pub const exponential = unaryOp("stablehlo.exponential").call;
pub const exponential_minus_one = unaryOp("stablehlo.exponential_minus_one").call;
pub const floor = unaryOp("stablehlo.floor").call;
pub const log = unaryOp("stablehlo.log").call;
pub const log_plus_one = unaryOp("stablehlo.log_plus_one").call;
pub const not = unaryOp("stablehlo.not").call;
pub const negate = unaryOp("stablehlo.negate").call;
pub const sqrt = unaryOp("stablehlo.sqrt").call;
pub const tanh = unaryOp("stablehlo.tanh").call;
pub const cbrt = unaryOp("stablehlo.cbrt").call;
pub const ceil = unaryOp("stablehlo.ceil").call;
pub const rsqrt = unaryOp("stablehlo.rsqrt").call;
pub const count_leading_zeros = unaryOp("stablehlo.count_leading_zeros").call;
pub const is_finite = unaryOp("stablehlo.is_finite").call;
pub const logistic = unaryOp("stablehlo.logistic").call;
pub const popcnt = unaryOp("stablehlo.popcnt").call;
pub const sign = unaryOp("stablehlo.sign").call;
pub const real = unaryOp("stablehlo.real").call;
pub const imag = unaryOp("stablehlo.imag").call;
pub const round_nearest_afz = unaryOp("stablehlo.round_nearest_afz").call;
pub const round_nearest_even = unaryOp("stablehlo.round_nearest_even").call;
pub const tuple = unaryOp("stablehlo.tuple").call;

pub const add = binaryOp("stablehlo.add").call;
pub const multiply = binaryOp("stablehlo.multiply").call;
pub const divide = binaryOp("stablehlo.divide").call;
pub const subtract = binaryOp("stablehlo.subtract").call;
pub const or_ = binaryOp("stablehlo.or").call;
pub const xor = binaryOp("stablehlo.xor").call;
pub const and_ = binaryOp("stablehlo.and").call;
pub const atan2 = binaryOp("stablehlo.atan2").call;
pub const maximum = binaryOp("stablehlo.maximum").call;
pub const minimum = binaryOp("stablehlo.minimum").call;
pub const power = binaryOp("stablehlo.power").call;
pub const remainder = binaryOp("stablehlo.remainder").call;
pub const shift_left = binaryOp("stablehlo.shift_left").call;
pub const shift_right_arithmetic = binaryOp("stablehlo.shift_right_arithmetic").call;
pub const shift_right_logical = binaryOp("stablehlo.shift_right_logical").call;
pub const complex = binaryOp("stablehlo.complex").call;

pub const bitcast_convert = castOp("stablehlo.bitcast_convert").call;
pub const convert = castOp("stablehlo.convert").call;
pub const reshape = castOp("stablehlo.reshape").call;

fn castOp(comptime op_name: []const u8) type {
    return struct {
        pub fn call(ctx: *mlir.Context, value: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
            return mlir.Operation.make(ctx, op_name, .{
                .operands = .{ .flat = &.{value} },
                .results = .{ .flat = &.{result_type} },
                .location = location,
            });
        }
    };
}

fn unaryOp(comptime op_name: []const u8) type {
    return struct {
        pub fn call(ctx: *mlir.Context, value: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
            return mlir.Operation.make(ctx, op_name, .{
                .operands = .{ .flat = &.{value} },
                .result_type_inference = true,
                .location = location,
            });
        }
    };
}

pub fn binaryOp(comptime op_name: []const u8) type {
    return struct {
        pub fn call(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
            return mlir.Operation.make(ctx, op_name, .{
                .operands = .{ .flat = &.{ lhs, rhs } },
                .result_type_inference = true,
                .location = location,
            });
        }
    };
}

pub fn cholesky(ctx: *mlir.Context, value: *const mlir.Value, lower: bool, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.cholesky", .{
        .operands = .{ .flat = &.{value} },
        .result_type_inference = true,
        .attributes = &.{
            .named(ctx, "lower", .int(ctx, .i1, @intFromBool(lower))),
        },
        .location = location,
    });
}

pub fn clamp(ctx: *mlir.Context, min: *const mlir.Value, value: *const mlir.Value, max: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.clamp", .{
        .operands = .{ .flat = &.{ min, value, max } },
        .result_type_inference = true,
        .location = location,
    });
}

pub const CompositeOpts = struct {
    name: []const u8,
    decomposition: []const u8,
    composite_attributes: []const mlir.NamedAttribute = &.{},
    version: i32 = 0,
};

pub fn composite(
    ctx: *mlir.Context,
    inputs: []const *const mlir.Value,
    result_types: []const *const mlir.Type,
    opts: CompositeOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.composite", .{
        .operands = .{ .flat = inputs },
        .results = .{ .flat = result_types },
        .attributes = &.{
            .named(ctx, "name", .string(ctx, opts.name)),
            .named(ctx, "decomposition", .flatSymbolRef(ctx, opts.decomposition)),
            .named(ctx, "composite_attributes", .dict(ctx, opts.composite_attributes)),
            .named(ctx, "version", .int(ctx, .i32, opts.version)),
        },
        .verify = false,
        .location = location,
    });
}

pub const DotPrecision = union(enum) {
    fast,
    high,
    highest,
    algorithm: DotAlgorithm,

    pub fn precision(self: DotPrecision) PrecisionAttr.Precision {
        return switch (self) {
            .fast => .DEFAULT,
            .high => .HIGH,
            .highest => .HIGHEST,
            // When we specify the dot algorithm, we should not specify the precision.
            .algorithm => .DEFAULT,
        };
    }

    pub fn algorithmAttr(self: DotPrecision, ctx: *mlir.Context, operand_type: mlir.RankedTensorType) ?mlir.Attribute {
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
        .accumulation = .f32,
        .component_count = 1,
        .num_primitive_operations = 6,
        .allow_imprecise_accumulation = false,
    };
};

pub fn dotAlgorithmAttr(ctx: *mlir.Context, dot_algorithm: DotAlgorithm, element_type: *const mlir.Type) *const mlir.Attribute {
    return @ptrCast(c.stablehloDotAlgorithmGet(
        ctx.ptr(),
        element_type.ptr(),
        element_type.ptr(),
        mlir.Type.float(ctx, dot_algorithm.accumulation).ptr(),
        dot_algorithm.component_count,
        dot_algorithm.component_count,
        dot_algorithm.num_primitive_operations,
        dot_algorithm.allow_imprecise_accumulation,
    ).ptr);
}

/// General matrix multiplication "a la Einstein sum"
/// Note: stablehlo doesn't do type inference for dot_general
pub fn dot_general(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    result_type: *const mlir.Type,
    opts: struct {
        lhs_batching_dimensions: []const i64,
        rhs_batching_dimensions: []const i64,
        lhs_contracting_dimensions: []const i64,
        rhs_contracting_dimensions: []const i64,
        dot_precision: DotPrecision,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    const precisions: [2]*const mlir.Attribute = @splat(precisionAttr(ctx, opts.dot_precision.precision()));
    const attributes = [3]mlir.NamedAttribute{
        .named(
            ctx,
            "dot_dimension_numbers",
            dotDimensionNumbersAttr(ctx, .{
                .lhs_batching_dimensions = opts.lhs_batching_dimensions,
                .rhs_batching_dimensions = opts.rhs_batching_dimensions,
                .lhs_contracting_dimensions = opts.lhs_contracting_dimensions,
                .rhs_contracting_dimensions = opts.rhs_contracting_dimensions,
            }),
        ),
        .named(ctx, "precision_config", .array(ctx, &precisions)),
        // keep algorithm as the last attribute so we can omit it when it's not set.
        .named(ctx, "algorithm", switch (opts.dot_precision) {
            .algorithm => |v| dotAlgorithmAttr(ctx, v, lhs.type_()),
            else => undefined,
        }),
    };
    const n_attributes = if (opts.dot_precision == .algorithm) attributes.len else attributes.len - 1;
    return mlir.Operation.make(ctx, "stablehlo.dot_general", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .attributes = attributes[0..n_attributes],
        .location = location,
    });
}

pub fn constant(
    ctx: *mlir.Context,
    dims: []const i64,
    elem_type: *const mlir.Type,
    raw_bytes: []const u8,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.constant", .{
        .operands = .{ .flat = &.{} },
        .results = .{ .flat = &.{.rankedTensor(dims, elem_type)} },
        .attributes = &.{
            .named(ctx, "value", .denseElements(.rankedTensor(dims, elem_type), raw_bytes)),
        },
        .location = location,
    });
}

pub fn broadcast_in_dim(
    ctx: *mlir.Context,
    operand: *const mlir.Value,
    dims: []const i64,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.broadcast_in_dim", .{
        .operands = .{ .flat = &.{operand} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "broadcast_dimensions", .denseArray(ctx, .i64, dims)),
        },
        .location = location,
    });
}

pub fn transpose(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    result_type: *const mlir.Type,
    opts: struct { permutation: []const i64 },
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.transpose", .{
        .operands = .{ .flat = &.{value} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "permutation", .denseArray(ctx, .i64, opts.permutation)),
        },
        .location = location,
    });
}

pub fn slice(
    ctx: *mlir.Context,
    operand: *const mlir.Value,
    start_indices: []const i64,
    limit_indices: []const i64,
    strides: []const i64,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.slice", .{
        .operands = .{ .flat = &.{operand} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "start_indices", .denseArray(ctx, .i64, start_indices)),
            .named(ctx, "limit_indices", .denseArray(ctx, .i64, limit_indices)),
            .named(ctx, "strides", .denseArray(ctx, .i64, strides)),
        },
        .location = location,
    });
}

pub fn concatenate(ctx: *mlir.Context, inputs: []const *const mlir.Value, dimension: i64, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.concatenate", .{
        .operands = .{ .flat = inputs },
        .result_type_inference = true,
        .attributes = &.{
            .named(ctx, "dimension", .int(ctx, .i64, dimension)),
        },
        .location = location,
    });
}

pub fn select(
    ctx: *mlir.Context,
    condition: *const mlir.Value,
    then: *const mlir.Value,
    else_: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.select", .{
        .operands = .{ .flat = &.{ condition, then, else_ } },
        .results = .{ .flat = &.{then.type_()} },
        .location = location,
    });
}

pub fn gather(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    indices: *const mlir.Value,
    slice_sizes: []const i64,
    args: struct {
        offset_dims: []const i64,
        collapsed_slice_dims: []const i64,
        operand_batching_dims: []const i64,
        start_indices_batching_dims: []const i64,
        start_index_map: []const i64,
        index_vector_dim: i64,
        indices_are_sorted: bool = false,
    },
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(
        ctx,
        "stablehlo.gather",
        .{
            .operands = .{ .flat = &.{ value, indices } },
            .result_type_inference = true,
            .attributes = &.{
                .named(ctx, "dimension_numbers", gatherDimensionNumbersAttr(
                    ctx,
                    args.offset_dims,
                    args.collapsed_slice_dims,
                    args.operand_batching_dims,
                    args.start_indices_batching_dims,
                    args.start_index_map,
                    args.index_vector_dim,
                )),
                .named(ctx, "slice_sizes", .denseArray(ctx, .i64, slice_sizes)),
                .named(ctx, "indices_are_sorted", .boolean(ctx, args.indices_are_sorted)),
            },
            .location = location,
        },
    );
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

    pub fn scatterDimensionNumbers(self: ScatterArgs, ctx: *mlir.Context) *const mlir.Attribute {
        return @ptrCast(c.stablehloScatterDimensionNumbersGet(
            ctx.ptr(),
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
        ).ptr);
    }
};

pub fn scatter(
    ctx: *mlir.Context,
    inputs: []const *const mlir.Value,
    scatter_indices: []const *const mlir.Value,
    updates: []const *const mlir.Value,
    update_block: *mlir.Block,
    args: ScatterArgs,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(
        ctx,
        "stablehlo.scatter",
        .{
            .operands = .{ .variadic = &.{ inputs, scatter_indices, updates } },
            .blocks = &.{update_block},
            .attributes = &.{
                .named(ctx, "scatter_dimension_numbers", args.scatterDimensionNumbers(ctx)),
                .named(ctx, "indices_are_sorted", .boolean(ctx, args.indices_are_sorted)),
                .named(ctx, "unique_indices", .boolean(ctx, args.unique_indices)),
            },
            .result_type_inference = true,
            .location = location,
        },
    );
}

pub fn iota(ctx: *mlir.Context, dimension: i64, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.iota", .{
        .operands = .{ .flat = &.{} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "iota_dimension", .int(ctx, .i64, dimension)),
        },
        .location = location,
    });
}

pub fn reverse(ctx: *mlir.Context, operand: *const mlir.Value, dimensions: []const i64, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.reverse", .{
        .operands = .{ .flat = &.{operand} },
        .results = .{ .flat = &.{operand.type_()} },
        .attributes = &.{
            .named(ctx, "dimensions", .denseArray(ctx, .i64, dimensions)),
        },
        .location = location,
    });
}

pub fn compare(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    comparison_direction: ComparisonDirectionAttr.Direction,
    compare_type: CompareTypeAttr.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.compare", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .result_type_inference = true,
        .attributes = &.{
            .named(ctx, "comparison_direction", comparisonDirectionAttr(ctx, comparison_direction)),
            .named(ctx, "compare_type", compareTypeAttr(ctx, compare_type)),
        },
        .location = location,
    });
}

pub fn dynamic_slice(
    ctx: *mlir.Context,
    operand: *const mlir.Value,
    new_dims: []const i64,
    start_indices: []const *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.dynamic_slice", .{
        .operands = .{ .variadic = &.{ &.{operand}, start_indices } },
        .result_type_inference = true,
        .attributes = &.{
            .named(ctx, "slice_sizes", .denseArray(ctx, .i64, new_dims)),
        },
        .location = location,
    });
}

pub const PadOpts = struct {
    low: []const i64,
    high: []const i64,
    interior: []const i64,
};

pub fn pad(ctx: *mlir.Context, value: *const mlir.Value, padding_value: *const mlir.Value, opts: PadOpts, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.pad", .{
        .operands = .{ .flat = &.{ value, padding_value } },
        .result_type_inference = true,
        .attributes = &.{
            .named(ctx, "edge_padding_low", .denseArray(ctx, .i64, opts.low)),
            .named(ctx, "edge_padding_high", .denseArray(ctx, .i64, opts.high)),
            .named(ctx, "interior_padding", .denseArray(ctx, .i64, opts.interior)),
        },
        .location = location,
    });
}

pub const TriangularSolveOpts = struct {
    left_side: bool,
    lower: bool,
    unit_diagonal: bool,
    transpose_a: TransposeAttr.Type,
};

pub fn triangular_solve(ctx: *mlir.Context, value: *const mlir.Value, other: *const mlir.Value, opts: TriangularSolveOpts, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.triangular_solve", .{
        .operands = .{ .flat = &.{ value, other } },
        .result_type_inference = true,
        .attributes = &.{
            .named(ctx, "left_side", .int(ctx, .i1, @intFromBool(opts.left_side))),
            .named(ctx, "lower", .int(ctx, .i1, @intFromBool(opts.lower))),
            .named(ctx, "unit_diagonal", .int(ctx, .i1, @intFromBool(opts.unit_diagonal))),
            .named(ctx, "transpose_a", transposeAttr(ctx, opts.transpose_a)),
        },
        .location = location,
    });
}

pub const FftOpts = struct {
    kind: FftTypeAttr.Type,
    length: []const i64,
};

pub fn fft(ctx: *mlir.Context, value: *const mlir.Value, opts: FftOpts, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.fft", .{
        .operands = .{ .flat = &.{value} },
        .result_type_inference = true,
        .attributes = &.{
            .named(ctx, "fft_type", fftTypeAttr(ctx, opts.kind)),
            .named(ctx, "fft_length", .denseArray(ctx, .i64, opts.length)),
        },
        .location = location,
    });
}

pub const RngDistributionAttr = opaque {
    const M = mlir.Methods(RngDistributionAttr, c.MlirAttribute);

    pub const isAFn = c.stablehloAttributeIsARngDistributionAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub const Type = enum {
        UNIFORM,
        NORMAL,
    };

    pub const InitArgs = struct {
        value: Type,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) *const RngDistributionAttr {
        return @ptrCast(c.stablehloRngDistributionAttrGet(ctx.ptr(), stringRef(@tagName(args.value))).ptr);
    }

    pub fn value(self: *const RngDistributionAttr) Type {
        return std.meta.stringToEnum(
            Type,
            mlir.string(c.stablehloRngDistributionAttrGetValue(self.ptr())),
        ) orelse unreachable;
    }

    pub fn attribute(self: *const RngDistributionAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

pub fn rngDistributionAttr(ctx: *mlir.Context, value: RngDistributionAttr.Type) *const mlir.Attribute {
    return RngDistributionAttr.get(ctx, .{ .value = value }).attribute();
}

pub const RngAlgorithmAttr = opaque {
    const M = mlir.Methods(RngAlgorithmAttr, c.MlirAttribute);

    pub const isAFn = c.stablehloAttributeIsARngAlgorithmAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub const Type = enum {
        DEFAULT,
        THREE_FRY,
        PHILOX,
    };

    pub const InitArgs = struct {
        value: Type,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) *const RngAlgorithmAttr {
        return @ptrCast(c.stablehloRngAlgorithmAttrGet(ctx.ptr(), stringRef(@tagName(args.value))).ptr);
    }

    pub fn value(self: *const RngAlgorithmAttr) Type {
        return std.meta.stringToEnum(
            Type,
            mlir.string(c.stablehloRngAlgorithmAttrGetValue(self.ptr())),
        ) orelse unreachable;
    }

    pub fn attribute(self: *const RngAlgorithmAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

pub fn rngAlgorithmAttr(ctx: *mlir.Context, value: RngAlgorithmAttr.Type) *const mlir.Attribute {
    return RngAlgorithmAttr.get(ctx, .{ .value = value }).attribute();
}

pub fn rng(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, shape: *const mlir.Value, rng_distribution: RngDistributionAttr.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.rng", .{
        .operands = .{ .flat = &.{ a, b, shape } },
        .result_type_inference = true,
        .attributes = &.{
            .named(ctx, "rng_distribution", rngDistributionAttr(ctx, rng_distribution)),
        },
        .location = location,
    });
}

pub fn rng_bit_generator(
    ctx: *mlir.Context,
    rng_algorithm: RngAlgorithmAttr.Type,
    initial_state: *const mlir.Value,
    res_state_type: *const mlir.Type,
    res_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.rng_bit_generator", .{
        .operands = .{ .flat = &.{initial_state} },
        .results = .{ .flat = &.{ res_state_type, res_type } },
        .attributes = &.{
            .named(ctx, "rng_algorithm", rngAlgorithmAttr(ctx, rng_algorithm)),
        },
        .location = location,
    });
}

pub fn reduce_precision(ctx: *mlir.Context, value: *const mlir.Value, exponent_bits: i32, mantissa_bits: i32, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.reduce_precision", .{
        .operands = .{ .flat = &.{value} },
        .result_type_inference = true,
        .attributes = &.{
            .named(ctx, "exponent_bits", .int(ctx, .i32, exponent_bits)),
            .named(ctx, "mantissa_bits", .int(ctx, .i32, mantissa_bits)),
        },
        .location = location,
    });
}

pub fn dynamic_update_slice(ctx: *mlir.Context, operand: *const mlir.Value, update: *const mlir.Value, start_indices: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.dynamic_update_slice", .{
        .operands = .{ .variadic = &.{ &.{operand}, &.{update}, start_indices } },
        .result_type_inference = true,
        .location = location,
    });
}

pub fn while_(
    ctx: *mlir.Context,
    operands: []const *const mlir.Value,
    result_types: []const *const mlir.Type,
    cond_block: *mlir.Block,
    body_block: *mlir.Block,
    location: *const mlir.Location,
) *mlir.Operation {
    _ = ctx;
    var state = mlir.OperationState.init("stablehlo.while", location);
    state.addOperands(operands);
    state.addResults(result_types);

    const cond_region = mlir.Region.init();
    cond_region.appendOwnedBlock(cond_block);
    const body_region = mlir.Region.init();
    body_region.appendOwnedBlock(body_block);
    state.addOwnedRegions(&.{ cond_region, body_region });

    return mlir.Operation.init(&state) catch @panic("Failed to create stablehlo.while operation");
}

pub fn get_tuple_element(ctx: *mlir.Context, tuple_value: *const mlir.Value, index: i64, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.get_tuple_element", .{
        .operands = .{ .flat = &.{tuple_value} },
        .result_type_inference = true,
        .attributes = &.{
            .named(ctx, "index", .int(ctx, .i32, index)),
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
    precision_config: []const PrecisionAttr.Precision = &.{},
};

pub fn convolution(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    res_type: *const mlir.Type,
    opts: ConvolutionOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var max_precisions: [2]*const mlir.Attribute = undefined;
    for (opts.precision_config, 0..) |p, i| {
        max_precisions[i] = precisionAttr(ctx, p);
    }
    var window_reversal: [3]i32 = undefined;
    for (opts.window_reversal, 0..) |w, i| {
        window_reversal[i] = @intCast(@intFromBool(w));
    }
    return mlir.Operation.make(ctx, "stablehlo.convolution", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "window_strides", .denseArray(ctx, .i64, opts.window_strides)),
            .named(ctx, "padding", .denseElements(.rankedTensor(opts.pad_shape, .int(ctx, .i64)), opts.pad_value)),
            .named(ctx, "lhs_dilation", .denseArray(ctx, .i64, opts.lhs_dilation)),
            .named(ctx, "rhs_dilation", .denseArray(ctx, .i64, opts.rhs_dilation)),
            .named(ctx, "window_reversal", .denseArray(ctx, .bool, window_reversal[0..opts.window_reversal.len])),
            .named(ctx, "dimension_numbers", convDimensionNumbersAttr(ctx, .{
                .input_batch_dimension = opts.input_batch_dimension,
                .input_feature_dimension = opts.input_feature_dimension,
                .input_spatial_dimensions = opts.input_spatial_dimensions,
                .kernel_input_feature_dimension = opts.kernel_input_feature_dimension,
                .kernel_output_feature_dimension = opts.kernel_output_feature_dimension,
                .kernel_spatial_dimensions = opts.kernel_spatial_dimensions,
                .output_batch_dimension = opts.output_batch_dimension,
                .output_feature_dimension = opts.output_feature_dimension,
                .output_spatial_dimensions = opts.output_spatial_dimensions,
            })),
            .named(ctx, "feature_group_count", .int(ctx, .i64, opts.feature_group_count)),
            .named(ctx, "batch_group_count", .int(ctx, .i64, opts.batch_group_count)),
            .named(ctx, "precision_config", .array(ctx, &max_precisions)),
        },
        .location = location,
    });
}

pub const CustomCallOpts = struct {
    pub const MAX_OPERANDS = 64;
    pub const MAX_RESULTS = 16;

    pub const ApiVersion = enum(i32) {
        original = 1,
        status_returning = 2,
        status_returning_unified = 3,
        typed_ffi = 4,
    };

    pub const BackendConfig = union(ApiVersion) {
        original: []const u8,
        status_returning: []const u8,
        status_returning_unified: []const u8,
        typed_ffi: *const mlir.Attribute,
    };

    pub const OutputOperandAlias = struct {
        output_index: i64,
        operand_index: i64,
    };

    call_target_name: []const u8,
    has_side_effect: ?bool = null,
    api_version: ?ApiVersion = null,
    backend_config: ?BackendConfig = null,
    operand_layouts: ?[]const []const usize = null,
    result_layouts: ?[]const []const usize = null,
    output_operand_aliases: ?[]const OutputOperandAlias = null,
    additional_attributes: []const mlir.NamedAttribute = &.{},
};

pub fn custom_call(ctx: *mlir.Context, inputs: []const *const mlir.Value, result_types: []const *const mlir.Type, opts: CustomCallOpts, location: *const mlir.Location) *mlir.Operation {
    const MAX_OPERANDS = CustomCallOpts.MAX_OPERANDS;
    const MAX_RESULTS = CustomCallOpts.MAX_RESULTS;
    const MINOR_TO_MAJOR = comptime blk: {
        var ret: [mlir.ShapedType.MAX_RANK]usize = undefined;
        for (0..ret.len) |i| {
            ret[i] = @intCast(mlir.ShapedType.MAX_RANK - i - 1);
        }
        break :blk ret;
    };

    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 32) = .empty;
    attrs.appendSliceAssumeCapacity(&.{
        .named(ctx, "call_target_name", .string(ctx, opts.call_target_name)),
    });
    attrs.appendSliceAssumeCapacity(opts.additional_attributes);

    if (opts.has_side_effect) |has_side_effects| {
        attrs.appendAssumeCapacity(
            .named(ctx, "has_side_effect", .boolean(ctx, has_side_effects)),
        );
    }

    const backend_config_api_version: ?CustomCallOpts.ApiVersion = if (opts.backend_config) |bc| bc else null;
    if (opts.api_version orelse backend_config_api_version) |v| {
        attrs.appendAssumeCapacity(
            .named(ctx, "api_version", .int(ctx, .i32, @backingInt(v))),
        );
    }
    if (opts.backend_config) |backend_config| {
        attrs.appendAssumeCapacity(
            .named(ctx, "backend_config", switch (backend_config) {
                .typed_ffi => |v| @ptrCast(v),
                inline else => |v| .string(ctx, v),
            }),
        );
    }

    if (opts.output_operand_aliases) |output_operand_aliases| {
        var buffer: stdx.BoundedArray(*const mlir.Attribute, MAX_RESULTS) = .empty;
        for (output_operand_aliases) |alias| {
            const output_tuple_indices = if (result_types.len > 1) &[1]i64{alias.output_index} else &.{};
            buffer.appendAssumeCapacity(
                outputOperandAliasAttr(ctx, .{ .operand_index = alias.operand_index, .output_tuple_indices = output_tuple_indices }),
            );
        }
        attrs.appendAssumeCapacity(
            .named(ctx, "output_operand_aliases", .array(ctx, buffer.constSlice())),
        );
    }

    var layouts_buffer: stdx.BoundedArray(*const mlir.Attribute, MAX_OPERANDS) = .empty;
    if (opts.operand_layouts) |layouts| {
        for (layouts) |ol| {
            layouts_buffer.appendAssumeCapacity(
                .denseElements(
                    .rankedTensor(&.{@intCast(ol.len)}, .index(ctx)),
                    ol,
                ),
            );
        }
    } else {
        for (inputs) |input| {
            const shaped_type = input.type_().isA(mlir.ShapedType).?;
            const ol = MINOR_TO_MAJOR[MINOR_TO_MAJOR.len - shaped_type.rank() ..];
            layouts_buffer.appendAssumeCapacity(
                .denseElements(
                    .rankedTensor(&.{@intCast(ol.len)}, .index(ctx)),
                    ol,
                ),
            );
        }
    }
    attrs.appendAssumeCapacity(
        .named(ctx, "operand_layouts", .array(ctx, layouts_buffer.constSlice())),
    );

    layouts_buffer.clear();
    if (opts.result_layouts) |layouts| {
        for (layouts) |rl| {
            layouts_buffer.appendAssumeCapacity(
                .denseElements(
                    .rankedTensor(&.{@intCast(rl.len)}, .index(ctx)),
                    rl,
                ),
            );
        }
    } else {
        for (result_types) |result_type| {
            const shaped_type = result_type.isA(mlir.ShapedType).?;
            const rl = MINOR_TO_MAJOR[MINOR_TO_MAJOR.len - shaped_type.rank() ..];
            layouts_buffer.appendAssumeCapacity(
                .denseElements(
                    .rankedTensor(&.{@intCast(rl.len)}, .index(ctx)),
                    rl,
                ),
            );
        }
    }
    attrs.appendAssumeCapacity(
        .named(ctx, "result_layouts", .array(ctx, layouts_buffer.constSlice())),
    );

    return mlir.Operation.make(ctx, "stablehlo.custom_call", .{
        .operands = .{ .flat = inputs },
        .results = .{ .flat = result_types },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

pub fn optimizationBarrier(ctx: *mlir.Context, inputs: []const *const mlir.Value, result_types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.optimization_barrier", .{
        .operands = .{ .flat = inputs },
        .results = .{ .flat = result_types },
        .location = location,
    });
}

pub fn createBuffer(ctx: *mlir.Context, value: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return custom_call(ctx, &.{}, &.{value}, .{
        .call_target_name = "CreateBuffer",
        .api_version = .typed_ffi,
    }, location);
}

pub fn pin(ctx: *mlir.Context, value: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return custom_call(
        ctx,
        &.{value},
        &.{@ptrCast(mlir.MemRefType.fromShaped(value.type_().isA(mlir.ShapedType).?))},
        .{
            .call_target_name = "Pin",
            .api_version = .typed_ffi,
        },
        location,
    );
}

pub fn unpin(ctx: *mlir.Context, value: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return custom_call(
        ctx,
        &.{value},
        &.{@ptrCast(mlir.RankedTensorType.fromShaped(value.type_().isA(mlir.ShapedType).?))},
        .{
            .call_target_name = "Unpin",
            .api_version = .typed_ffi,
        },
        location,
    );
}

pub fn partition_id(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.partition_id", .{
        .results = .{ .flat = &.{.rankedTensor(&.{}, .int(ctx, .u32))} },
        .location = location,
    });
}

pub const DotDimensionNumbersAttr = opaque {
    const M = mlir.Methods(DotDimensionNumbersAttr, c.MlirAttribute);

    pub const isAFn = c.stablehloAttributeIsADotDimensionNumbers;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub const InitArgs = struct {
        lhs_batching_dimensions: []const i64,
        rhs_batching_dimensions: []const i64,
        lhs_contracting_dimensions: []const i64,
        rhs_contracting_dimensions: []const i64,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) *const DotDimensionNumbersAttr {
        return @ptrCast(c.stablehloDotDimensionNumbersGet(
            ctx.ptr(),
            @intCast(args.lhs_batching_dimensions.len),
            args.lhs_batching_dimensions.ptr,
            @intCast(args.rhs_batching_dimensions.len),
            args.rhs_batching_dimensions.ptr,
            @intCast(args.lhs_contracting_dimensions.len),
            args.lhs_contracting_dimensions.ptr,
            @intCast(args.rhs_contracting_dimensions.len),
            args.rhs_contracting_dimensions.ptr,
        ).ptr);
    }

    pub fn lhsBatchingDimensionsSize(self: *const DotDimensionNumbersAttr) usize {
        return @intCast(c.stablehloDotDimensionNumbersGetLhsBatchingDimensionsSize(self.ptr()));
    }

    pub fn lhsBatchingDimensionsElem(self: *const DotDimensionNumbersAttr, pos: usize) i64 {
        return c.stablehloDotDimensionNumbersGetLhsBatchingDimensionsElem(self.ptr(), @intCast(pos));
    }

    pub fn rhsBatchingDimensionsSize(self: *const DotDimensionNumbersAttr) usize {
        return @intCast(c.stablehloDotDimensionNumbersGetRhsBatchingDimensionsSize(self.ptr()));
    }

    pub fn rhsBatchingDimensionsElem(self: *const DotDimensionNumbersAttr, pos: usize) i64 {
        return c.stablehloDotDimensionNumbersGetRhsBatchingDimensionsElem(self.ptr(), @intCast(pos));
    }

    pub fn lhsContractingDimensionsSize(self: *const DotDimensionNumbersAttr) usize {
        return @intCast(c.stablehloDotDimensionNumbersGetLhsContractingDimensionsSize(self.ptr()));
    }

    pub fn lhsContractingDimensionsElem(self: *const DotDimensionNumbersAttr, pos: usize) i64 {
        return c.stablehloDotDimensionNumbersGetLhsContractingDimensionsElem(self.ptr(), @intCast(pos));
    }

    pub fn rhsContractingDimensionsSize(self: *const DotDimensionNumbersAttr) usize {
        return @intCast(c.stablehloDotDimensionNumbersGetRhsContractingDimensionsSize(self.ptr()));
    }

    pub fn rhsContractingDimensionsElem(self: *const DotDimensionNumbersAttr, pos: usize) i64 {
        return c.stablehloDotDimensionNumbersGetRhsContractingDimensionsElem(self.ptr(), @intCast(pos));
    }

    pub fn attribute(self: *const DotDimensionNumbersAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

pub fn dotDimensionNumbersAttr(ctx: *mlir.Context, args: DotDimensionNumbersAttr.InitArgs) *const mlir.Attribute {
    return @ptrCast(DotDimensionNumbersAttr.get(ctx, args));
}

pub const GatherDimensionNumbersAttr = opaque {
    const M = mlir.Methods(GatherDimensionNumbersAttr, c.MlirAttribute);

    pub const isAFn = c.stablehloAttributeIsAGatherDimensionNumbers;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.eql(c.mlirAttributePrint);

    pub fn get(
        ctx: *mlir.Context,
        offset_dims: []const i64,
        collapsed_slice_dims: []const i64,
        operand_batching_dims: []const i64,
        start_indices_batching_dims: []const i64,
        start_index_map: []const i64,
        index_vector_dim: i64,
    ) *const GatherDimensionNumbersAttr {
        return @ptrCast(c.stablehloGatherDimensionNumbersGet(
            ctx.ptr(),
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
        ).ptr);
    }

    pub fn offsetDimsSize(self: *const GatherDimensionNumbersAttr) usize {
        return @intCast(c.stablehloGatherDimensionNumbersGetOffsetDimsSize(self.ptr()));
    }

    pub fn offsetDimsElem(self: *const GatherDimensionNumbersAttr, pos: usize) i64 {
        return c.stablehloGatherDimensionNumbersGetOffsetDimsElem(self.ptr(), @intCast(pos));
    }

    pub fn collapsedSliceDimsSize(self: *const GatherDimensionNumbersAttr) usize {
        return @intCast(c.stablehloGatherDimensionNumbersGetCollapsedSliceDimsSize(self.ptr()));
    }

    pub fn collapsedSliceDimsElem(self: *const GatherDimensionNumbersAttr, pos: usize) i64 {
        return c.stablehloGatherDimensionNumbersGetCollapsedSliceDimsElem(self.ptr(), @intCast(pos));
    }

    pub fn startIndexMapSize(self: *const GatherDimensionNumbersAttr) usize {
        return @intCast(c.stablehloGatherDimensionNumbersGetStartIndexMapSize(self.ptr()));
    }

    pub fn operandBatchingDimsSize(self: *const GatherDimensionNumbersAttr) usize {
        return @intCast(c.stablehloGatherDimensionNumbersGetOperandBatchingDimsSize(self.ptr()));
    }

    pub fn operandBatchingDimsElem(self: *const GatherDimensionNumbersAttr, pos: usize) i64 {
        return c.stablehloGatherDimensionNumbersGetOperandBatchingDimsElem(self.ptr(), @intCast(pos));
    }

    pub fn startIndicesBatchingDimsSize(self: *const GatherDimensionNumbersAttr) usize {
        return @intCast(c.stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsSize(self.ptr()));
    }

    pub fn startIndicesBatchingDimsElem(self: *const GatherDimensionNumbersAttr, pos: usize) i64 {
        return c.stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsElem(self.ptr(), @intCast(pos));
    }

    pub fn startIndexMapElem(self: *const GatherDimensionNumbersAttr, pos: usize) i64 {
        return c.stablehloGatherDimensionNumbersGetStartIndexMapElem(self.ptr(), @intCast(pos));
    }

    pub fn indexVectorDim(self: *const GatherDimensionNumbersAttr) usize {
        return @intCast(c.stablehloGatherDimensionNumbersGetIndexVectorDim(self.ptr()));
    }
};

pub fn gatherDimensionNumbersAttr(
    ctx: *mlir.Context,
    offset_dims: []const i64,
    collapsed_slice_dims: []const i64,
    operand_batching_dims: []const i64,
    start_indices_batching_dims: []const i64,
    start_index_map: []const i64,
    index_vector_dim: i64,
) *const mlir.Attribute {
    return @ptrCast(GatherDimensionNumbersAttr.get(
        ctx,
        offset_dims,
        collapsed_slice_dims,
        operand_batching_dims,
        start_indices_batching_dims,
        start_index_map,
        index_vector_dim,
    ));
}

pub const ConvDimensionNumbersAttr = opaque {
    const M = mlir.Methods(ConvDimensionNumbersAttr, c.MlirAttribute);

    pub const isAFn = c.stablehloAttributeIsAConvDimensionNumbers;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub const InitArgs = struct {
        input_batch_dimension: i64,
        input_feature_dimension: i64,
        input_spatial_dimensions: []const i64,
        kernel_input_feature_dimension: i64,
        kernel_output_feature_dimension: i64,
        kernel_spatial_dimensions: []const i64,
        output_batch_dimension: i64,
        output_feature_dimension: i64,
        output_spatial_dimensions: []const i64,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) *const ConvDimensionNumbersAttr {
        return @ptrCast(c.stablehloConvDimensionNumbersGet(
            ctx.ptr(),
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
        ).ptr);
    }

    pub fn inputBatchDimension(self: *const ConvDimensionNumbersAttr) i64 {
        return c.stablehloConvDimensionNumbersGetInputBatchDimension(self.ptr());
    }

    pub fn inputFeatureDimension(self: *const ConvDimensionNumbersAttr) i64 {
        return c.stablehloConvDimensionNumbersGetInputFeatureDimension(self.ptr());
    }

    pub fn inputSpatialDimensionsSize(self: *const ConvDimensionNumbersAttr) usize {
        return @intCast(c.stablehloConvDimensionNumbersGetInputSpatialDimensionsSize(self.ptr()));
    }

    pub fn inputSpatialDimensionsElem(self: *const ConvDimensionNumbersAttr, pos: usize) i64 {
        return c.stablehloConvDimensionNumbersGetInputSpatialDimensionsElem(self.ptr(), @intCast(pos));
    }

    pub fn kernelInputFeatureDimension(self: *const ConvDimensionNumbersAttr) i64 {
        return c.stablehloConvDimensionNumbersGetKernelInputFeatureDimension(self.ptr());
    }

    pub fn kernelOutputFeatureDimension(self: *const ConvDimensionNumbersAttr) i64 {
        return c.stablehloConvDimensionNumbersGetKernelOutputFeatureDimension(self.ptr());
    }

    pub fn kernelSpatialDimensionsSize(self: *const ConvDimensionNumbersAttr) usize {
        return @intCast(c.stablehloConvDimensionNumbersGetKernelSpatialDimensionsSize(self.ptr()));
    }

    pub fn kernelSpatialDimensionsElem(self: *const ConvDimensionNumbersAttr, pos: usize) i64 {
        return c.stablehloConvDimensionNumbersGetKernelSpatialDimensionsElem(self.ptr(), @intCast(pos));
    }

    pub fn outputBatchDimension(self: *const ConvDimensionNumbersAttr) i64 {
        return c.stablehloConvDimensionNumbersGetOutputBatchDimension(self.ptr());
    }

    pub fn outputFeatureDimension(self: *const ConvDimensionNumbersAttr) i64 {
        return c.stablehloConvDimensionNumbersGetOutputFeatureDimension(self.ptr());
    }

    pub fn outputSpatialDimensionsSize(self: *const ConvDimensionNumbersAttr) usize {
        return @intCast(c.stablehloConvDimensionNumbersGetOutputSpatialDimensionsSize(self.ptr()));
    }

    pub fn outputSpatialDimensionsElem(self: *const ConvDimensionNumbersAttr, pos: usize) i64 {
        return c.stablehloConvDimensionNumbersGetOutputSpatialDimensionsElem(self.ptr(), @intCast(pos));
    }

    pub fn attribute(self: *const ConvDimensionNumbersAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

pub fn convDimensionNumbersAttr(ctx: *mlir.Context, args: ConvDimensionNumbersAttr.InitArgs) *const mlir.Attribute {
    return @ptrCast(ConvDimensionNumbersAttr.get(ctx, args));
}

pub const OutputOperandAliasAttr = opaque {
    const M = mlir.Methods(OutputOperandAliasAttr, c.MlirAttribute);

    pub const isAFn = c.stablehloAttributeIsAOutputOperandAlias;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub const InitArgs = struct {
        output_tuple_indices: []const i64 = &.{},
        operand_index: i64,
        operand_tuple_indices: []const i64 = &.{},
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) *const OutputOperandAliasAttr {
        return @ptrCast(c.stablehloOutputOperandAliasGet(
            ctx.ptr(),
            @intCast(args.output_tuple_indices.len),
            @ptrCast(args.output_tuple_indices),
            @intCast(args.operand_index),
            @intCast(args.operand_tuple_indices.len),
            @ptrCast(args.operand_tuple_indices),
        ).ptr);
    }

    pub fn attribute(self: *const OutputOperandAliasAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

pub fn outputOperandAliasAttr(ctx: *mlir.Context, args: OutputOperandAliasAttr.InitArgs) *const mlir.Attribute {
    return @ptrCast(OutputOperandAliasAttr.get(ctx, args));
}

pub const PrecisionAttr = opaque {
    const M = mlir.Methods(PrecisionAttr, c.MlirAttribute);

    pub const isAFn = c.stablehloAttributeIsAPrecisionAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub const Precision = enum {
        DEFAULT,
        HIGH,
        HIGHEST,
    };

    pub const InitArgs = struct {
        value: Precision,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) *const PrecisionAttr {
        return @ptrCast(c.stablehloPrecisionAttrGet(ctx.ptr(), stringRef(@tagName(args.value))).ptr);
    }

    pub fn value(self: *const PrecisionAttr) Precision {
        const name = mlir.string(c.stablehloPrecisionAttrGetValue(self.ptr()));
        return std.meta.stringToEnum(Precision, name) orelse unreachable;
    }

    pub fn attribute(self: *const PrecisionAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

pub fn precisionAttr(ctx: *mlir.Context, precision: PrecisionAttr.Precision) *const mlir.Attribute {
    return PrecisionAttr.get(ctx, .{ .value = precision }).attribute();
}

pub const ComparisonDirectionAttr = opaque {
    const M = mlir.Methods(ComparisonDirectionAttr, c.MlirAttribute);

    pub const isAFn = c.stablehloAttributeIsAComparisonDirectionAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub const Direction = enum {
        EQ,
        NE,
        GE,
        GT,
        LE,
        LT,
    };

    pub const InitArgs = struct {
        value: Direction,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) *const ComparisonDirectionAttr {
        return @ptrCast(c.stablehloComparisonDirectionAttrGet(ctx.ptr(), mlir.stringRef(@tagName(args.value))).ptr);
    }

    pub fn value(self: *const ComparisonDirectionAttr) Direction {
        const name = mlir.string(c.stablehloComparisonDirectionAttrGetValue(self.ptr()));
        return std.meta.stringToEnum(Direction, name) orelse unreachable;
    }

    pub fn attribute(self: *const ComparisonDirectionAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

pub fn comparisonDirectionAttr(ctx: *mlir.Context, value: ComparisonDirectionAttr.Direction) *const mlir.Attribute {
    return ComparisonDirectionAttr.get(ctx, .{ .value = value }).attribute();
}

pub const CompareTypeAttr = opaque {
    const M = mlir.Methods(CompareTypeAttr, c.MlirAttribute);

    pub const isAFn = c.stablehloAttributeIsAComparisonTypeAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub const Type = enum {
        SIGNED,
        UNSIGNED,
        FLOAT,
        TOTALORDER,
    };

    pub const InitArgs = struct {
        value: Type,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) *const CompareTypeAttr {
        return @ptrCast(c.stablehloComparisonTypeAttrGet(ctx.ptr(), stringRef(@tagName(args.value))).ptr);
    }

    pub fn value(self: *const CompareTypeAttr) Type {
        const name = mlir.string(c.stablehloComparisonTypeAttrGetValue(self.ptr()));
        return std.meta.stringToEnum(Type, name) orelse unreachable;
    }

    pub fn attribute(self: *const CompareTypeAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

pub fn compareTypeAttr(ctx: *mlir.Context, value: CompareTypeAttr.Type) *const mlir.Attribute {
    return CompareTypeAttr.get(ctx, .{ .value = value }).attribute();
}

pub const TransposeAttr = opaque {
    const M = mlir.Methods(TransposeAttr, c.MlirAttribute);

    pub const isAFn = c.stablehloAttributeIsATransposeAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub const Type = enum {
        NO_TRANSPOSE,
        TRANSPOSE,
        ADJOINT,
    };

    pub const InitArgs = struct {
        value: Type,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) *const TransposeAttr {
        return @ptrCast(c.stablehloTransposeAttrGet(ctx.ptr(), mlir.stringRef(@tagName(args.value))).ptr);
    }

    pub fn value(self: *const TransposeAttr) Type {
        const name = mlir.string(c.stablehloTransposeAttrGetValue(self.ptr()));
        return std.meta.stringToEnum(Type, name) orelse unreachable;
    }

    pub fn attribute(self: *const TransposeAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

pub fn transposeAttr(ctx: *mlir.Context, value: TransposeAttr.Type) *const mlir.Attribute {
    return TransposeAttr.get(ctx, .{ .value = value }).attribute();
}

pub const FftTypeAttr = opaque {
    const M = mlir.Methods(FftTypeAttr, c.MlirAttribute);

    pub const isAFn = c.stablehloAttributeIsAFftTypeAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    pub const Type = enum {
        FFT,
        IFFT,
        RFFT,
        IRFFT,
    };

    pub const InitArgs = struct {
        value: Type,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) *const FftTypeAttr {
        return @ptrCast(c.stablehloFftTypeAttrGet(ctx.ptr(), mlir.stringRef(@tagName(args.value))).ptr);
    }

    pub fn value(self: *const FftTypeAttr) Type {
        const name = mlir.string(c.stablehloFftTypeAttrGetValue(self.ptr()));
        return std.meta.stringToEnum(Type, name) orelse unreachable;
    }

    pub fn attribute(self: *const FftTypeAttr) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

pub fn fftTypeAttr(ctx: *mlir.Context, value: FftTypeAttr.Type) *const mlir.Attribute {
    return FftTypeAttr.get(ctx, .{ .value = value }).attribute();
}

fn stringFromStream(buf: []u8, streamFn: anytype, args: anytype) []const u8 {
    var writer = std.Io.Writer.fixed(buf);
    var sctx: mlir.StringCallbackCtx = .{ .writer = &writer };
    _ = @call(.auto, streamFn, args ++ .{ mlir.stringCallback, &sctx });
    return writer.buffered();
}

pub fn versionFromCompatibilityRequirement(buffer: []u8, requirement: c.MlirStablehloCompatibilityRequirement) []const u8 {
    return stringFromStream(buffer, c.stablehloVersionFromCompatibilityRequirement, .{requirement});
}

pub fn smallerVersion(version1: []const u8, version2: []const u8) []const u8 {
    var buf: [32]u8 = undefined;
    const result = stringFromStream(&buf, c.stablehloGetSmallerVersion, .{ stringRef(version1), stringRef(version2) });
    return if (std.mem.eql(u8, result, version1)) version1 else version2;
}

pub fn currentVersion() []const u8 {
    const state = struct {
        var buf: [32]u8 = undefined;
    };
    return stringFromStream(&state.buf, c.stablehloGetCurrentVersion, .{});
}

pub fn minimumVersion() []const u8 {
    const state = struct {
        var buf: [32]u8 = undefined;
    };
    return stringFromStream(&state.buf, c.stablehloGetMinimumVersion, .{});
}

pub fn serializePortableArtifact(bytecode: []const u8, target_version: []const u8, writer: *std.Io.Writer) !void {
    var sctx: mlir.StringCallbackCtx = .{ .writer = writer };
    const result = c.stablehloSerializePortableArtifactFromStringRef(
        stringRef(bytecode),
        stringRef(target_version),
        mlir.stringCallback,
        &sctx,
    );
    if (sctx.err) |err| {
        return err;
    }
    if (result.value == 0) {
        return error.InvalidMlirBytecodeVersion;
    }
}

pub fn serializePortableArtifact2(module: *mlir.Module, target_version: []const u8, writer: *std.Io.Writer) !void {
    var sctx: mlir.StringCallbackCtx = .{ .writer = writer };
    const result = c.stablehloSerializePortableArtifactFromModule(
        module.ptr(),
        stringRef(target_version),
        mlir.stringCallback,
        &sctx,
        true,
    );
    if (sctx.err) |err| {
        return err;
    }
    // TODO(Corentin): Find out why this crashes on macOS
    //if (c.mlirLogicalResultIsFailure(result)) {
    if (@as(c_int, result.value) == 0) {
        return error.InvalidMlirBytecodeVersion;
    }
}

pub fn return_(ctx: *mlir.Context, value: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.return", .{
        .operands = .{ .flat = &.{value} },
        .location = location,
        .verify = false,
    });
}

pub fn returns(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.return", .{
        .operands = .{ .variadic = &.{values} },
        .verify = false,
        .location = location,
    });
}

pub const ChannelType = enum(i64) {
    collective = 0,
    send_recv = 1,
};

pub fn channelHandleAttr(
    ctx: *mlir.Context,
    handle: i64,
    channel_type: ChannelType,
) *const mlir.Attribute {
    return @ptrCast(c.stablehloChannelHandleGet(
        ctx.ptr(),
        handle,
        @backingInt(channel_type),
    ).ptr);
}

pub fn allReduce(
    ctx: *mlir.Context,
    inputs: []const *const mlir.Value,
    result_types: []const *const mlir.Type,
    reducer_block: *mlir.Block,
    replica_groups: *const mlir.Attribute,
    channel_handle: *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.all_reduce", .{
        .operands = .{ .variadic = &.{inputs} },
        .results = .{ .flat = result_types },
        .blocks = &.{reducer_block},
        .attributes = &.{
            .named(ctx, "replica_groups", replica_groups),
            .named(ctx, "channel_handle", channel_handle),
            .named(ctx, "use_global_device_ids", .unit(ctx)),
        },
        .verify = true,
        .location = location,
    });
}
