const std = @import("std");

const Context = @import("../context.zig").Context;
const module = @import("../module.zig");
const mlir = @import("../mlir.zig");
const dialect = @import("mlir/dialects");

const Tensor = @import("../tensor.zig").Tensor;
const Shape = @import("../shape.zig").Shape;
const SdpaOpts = @import("../nn.zig").SdpaOpts;
const DataType = @import("../dtype.zig").DataType;
const Data = @import("../dtype.zig").Data;
const CompilationContext = module.CompilationContext;

pub fn canUseCudnnSdpa(q_shape: Shape) bool {
    const ctx = CompilationContext.current();
    // TODO(Corendos): Check cuda version, cudnn version, device compatibility.
    if (ctx.target() != .cuda) return false;

    if (q_shape.rank() != 4) return false;

    // NOTE(Corentin): In Cudnn fused MHA head_dim is limited to 128.
    if (q_shape.dim(.hd) > 128) return false;

    // NOTE(Corentin): In Cudnn fused MHA data type is limited to F16 and BF16.
    if (q_shape.dtype() != .f16 and q_shape.dtype() != .bf16) return false;

    return true;
}

fn elementTypeFromDataType(dtype: DataType) [:0]const u8 {
    return switch (dtype) {
        .f16 => "F16",
        .bf16 => "BF16",
        else => @panic("Unsupported DataType"),
    };
}

pub fn sdpa(q_: Tensor, k_: Tensor, v_: Tensor, opts: SdpaOpts) Tensor {
    const ctx = CompilationContext.current();
    const q = q_.transpose(.{ .b, .h, .q, .hd });
    var k = k_.transpose(.{ .b, .h, .k, .hd });
    const v = v_.transpose(.{ .b, .h, .k, .hd });

    const sqrtHeadDim: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(q.dim(.hd))));
    const kernel_scale = if (opts.scale) |_| 1.0 else sqrtHeadDim;
    if (opts.scale) |s| {
        k = k.mul(s);
    }

    var buffer: [4096]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buffer);
    const allocator = fba.allocator();

    const backend_config = std.fmt.allocPrintZ(
        allocator,
        \\{{
        \\  "operation_queue_id":"0",
        \\  "wait_on_operation_queues":[],
        \\  "cudnn_fmha_backend_config": {{
        \\    "algorithm": {{
        \\      "algo_id": "0",
        \\      "math_type": "TENSOR_OP_MATH",
        \\      "tuning_knobs": {{"17": "1", "24": "0"}},
        \\      "is_cudnn_frontend": true,
        \\      "workspace_size": "0",
        \\    }},
        \\    "fmha_scale": {},
        \\    "dropout_rate": 0,
        \\    "intermediate_tensor_shape": {{
        \\      "element_type": "{s}",
        \\      "dimensions": ["{}", "{}", "{}", "{}"],
        \\      "tuple_shapes": [],
        \\      "layout": {{
        \\        "dim_level_types": [],
        \\        "dim_unique": [],
        \\        "dim_ordered": [],
        \\        "minor_to_major": ["3", "2", "1", "0"],
        \\        "tiles": [],
        \\        "element_size_in_bits": "0",
        \\        "memory_space": "0",
        \\        "index_primitive_type": "PRIMITIVE_TYPE_INVALID",
        \\        "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID",
        \\        "dynamic_shape_metadata_prefix_bytes": "0",
        \\      }},
        \\      "is_dynamic_dimension": [false, false, false, false],
        \\    }},
        \\    "seed": 0,
        \\    "is_flash_attention": true,
        \\    "mask_type": "NO_MASK",
        \\    "bmm1_dot_dimension_numbers": {{
        \\       "lhs_contracting_dimensions": ["3"],
        \\       "rhs_contracting_dimensions": ["3"],
        \\       "lhs_batch_dimensions": ["0", "1"],
        \\       "rhs_batch_dimensions": ["0", "1"],
        \\    }},
        \\    "bmm2_dot_dimension_numbers": {{
        \\       "lhs_contracting_dimensions": ["3"],
        \\       "rhs_contracting_dimensions": ["2"],
        \\       "lhs_batch_dimensions": ["0", "1"],
        \\       "rhs_batch_dimensions": ["0", "1"],
        \\    }}
        \\  }}
        \\}}
    ,
        .{
            kernel_scale,
            elementTypeFromDataType(q.dtype()),
            q.dim(.b),
            q.dim(.h),
            q.dim(.q),
            k.dim(.k),
        },
    ) catch unreachable;

    var bias = Tensor.constant(Shape.init(.{ .b = q.dim(.b), .h = q.dim(.h), .q = q.dim(.q), .k = k.dim(.k) }, q.dtype()), Data.init(q.dtype(), 0));

    if (opts.attn_mask) |attn_mask| {
        bias = bias.add(attn_mask.broad(bias.shape()));
    }

    const mlir_ctx = ctx.mlirCtx();
    const loc = mlir_ctx.location(@src());
    const op = dialect.stablehlo.custom_call(
        mlir_ctx,
        &.{ q.value(), k.value(), v.value(), bias.value() },
        .{
            .call_target_name = "__cudnn$fmhaScaleBiasSoftmax",
            .backend_config = .{ .string = backend_config },
            .has_side_effect = false,
            .api_version = .original,
        },
        &.{
            mlir.ext.mlirType(mlir_ctx, q.shape()),
            mlir.RankedTensorType.init(&.{0}, mlir.IntegerType(.u8).init(mlir_ctx).as(mlir.Type)).as(mlir.Type),
        },
        loc,
    );
    const result = Tensor._result(q.shape(), op.result(0)).transpose(q_.shape());
    return result;
}

pub const SdpaPagedOpts = struct {
    attn_mask: ?Tensor = null,
    scale: ?Tensor = null,
    max_seq_len_kv: usize,
};

pub fn sdpaPaged(q_: Tensor, k_: Tensor, v_: Tensor, sequence_length_q: Tensor, sequence_length_kv: Tensor, page_table: Tensor, opts: SdpaPagedOpts) Tensor {
    const ctx = CompilationContext.current();
    const q = q_.transpose(.{ .b, .h, .q, .hd });
    var k = k_.transpose(.{ .page, .h, .k_chunk, .hd });
    const v = v_.transpose(.{ .page, .h, .k_chunk, .hd });

    const sqrtHeadDim: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(q.dim(.hd))));
    const kernel_scale = if (opts.scale) |_| 1.0 else sqrtHeadDim;
    if (opts.scale) |s| {
        k = k.mul(s);
    }

    var buffer: [4096]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buffer);
    const allocator = fba.allocator();

    const backend_config =
        std.fmt.allocPrintZ(
        allocator,
        \\{{
        \\  "operation_queue_id":"0",
        \\  "wait_on_operation_queues":[],
        \\  "cudnn_fmha_backend_config": {{
        \\    "algorithm": {{
        \\      "algo_id": "0",
        \\      "math_type": "TENSOR_OP_MATH",
        \\      "tuning_knobs": {{"17": "1", "24": "0"}},
        \\      "is_cudnn_frontend": true,
        \\      "workspace_size": "0",
        \\    }},
        \\    "fmha_scale": {},
        \\    "dropout_rate": 0,
        \\    "intermediate_tensor_shape": {{
        \\      "element_type": "{s}",
        \\      "dimensions": ["{}", "{}", "{}", "{}"],
        \\      "tuple_shapes": [],
        \\      "layout": {{
        \\        "dim_level_types": [],
        \\        "dim_unique": [],
        \\        "dim_ordered": [],
        \\        "minor_to_major": ["3", "2", "1", "0"],
        \\        "tiles": [],
        \\        "element_size_in_bits": "0",
        \\        "memory_space": "0",
        \\        "index_primitive_type": "PRIMITIVE_TYPE_INVALID",
        \\        "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID",
        \\        "dynamic_shape_metadata_prefix_bytes": "0",
        \\      }},
        \\      "is_dynamic_dimension": [false, false, false, false],
        \\    }},
        \\    "seed": 0,
        \\    "is_flash_attention": true,
        \\    "mask_type": "NO_MASK",
        \\    "bmm1_dot_dimension_numbers": {{
        \\       "lhs_contracting_dimensions": ["3"],
        \\       "rhs_contracting_dimensions": ["3"],
        \\       "lhs_batch_dimensions": ["0", "1"],
        \\       "rhs_batch_dimensions": ["0", "1"],
        \\    }},
        \\    "bmm2_dot_dimension_numbers": {{
        \\       "lhs_contracting_dimensions": ["3"],
        \\       "rhs_contracting_dimensions": ["2"],
        \\       "lhs_batch_dimensions": ["0", "1"],
        \\       "rhs_batch_dimensions": ["0", "1"],
        \\    }},
        \\    "max_sequence_length_kv": {},
        \\  }}
        \\}}
    ,
        .{
            kernel_scale,
            elementTypeFromDataType(q.dtype()),
            q.dim(.b),
            q.dim(.h),
            q.dim(.q),
            k.dim(.k_chunk),
            opts.max_seq_len_kv,
        },
    ) catch unreachable;

    std.debug.print("backend_config: {s}\n", .{backend_config});

    var bias = Tensor.constant(Shape.init(.{ .b = q.dim(.b), .h = q.dim(.h), .q = q.dim(.q), .k = opts.max_seq_len_kv }, q.dtype()), Data.init(q.dtype(), 0));

    if (opts.attn_mask) |attn_mask| {
        bias = bias.add(attn_mask.broad(bias.shape()));
    }

    const output_shape = Shape.init(.{ .b = q.dim(.b), .h = q.dim(.h), .q = q.dim(.q), .hd = q.dim(.hd) }, q.dtype());
    const mlir_ctx = ctx.mlirCtx();
    const loc = mlir_ctx.location(@src());
    const op = dialect.stablehlo.custom_call(
        mlir_ctx,
        &.{ q.value(), k.value(), v.value(), sequence_length_q.value(), sequence_length_kv.value(), page_table.value(), page_table.value() },
        .{
            .call_target_name = "__cudnn$fmhaScaleBiasSoftmax",
            .backend_config = backend_config,
            .api_version = 2,
            .has_side_effect = false,
            .output_operand_aliases = &.{},
        },
        &.{
            mlir.ext.mlirType(mlir_ctx, output_shape),
            mlir.RankedTensorType.init(&.{0}, mlir.IntegerType(.u8).init(mlir_ctx).as(mlir.Type).?).asType(),
        },
        loc,
    );
    const result = Tensor._result(output_shape, op.result(0)).transpose(q_.shape());
    return result;
}
