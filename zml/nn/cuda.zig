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

pub fn canUseCudnnSdpa(head_dim: i64, dtype: DataType) bool {
    const ctx = CompilationContext.current();
    // TODO(Corendos): Check cuda version, cudnn version, device compatibility.
    if (!ctx.targetIs(.cuda)) return false;

    // NOTE(Corentin): In Cudnn fused MHA head_dim is limited to 128.
    if (head_dim > 128) return false;

    // NOTE(Corentin): In Cudnn fused MHA data type is limited to F16 and BF16.
    if (dtype != .f16 and dtype != .bf16) return false;

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
    const scale: f32 = if (opts.scale) |_| 1.0 else sqrtHeadDim;

    if (opts.scale) |s| k = k.mul(s);

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
        \\    "fmha_scale": {d},
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
            scale,
            elementTypeFromDataType(q.dtype()),
            q.dim(.b),
            q.dim(.h),
            q.dim(.q),
            k.dim(.k),
        },
    ) catch unreachable;

    var bias = Tensor.constant(Shape.init(.{ .b = q.dim(.b), .h = q.dim(.h), .q = q.dim(.q), .k = k.dim(.k) }, q.dtype()), Data.init(q.dtype(), 0));

    if (opts.attn_mask) |attn_mask| {
        const mask = attn_mask.withTags(.{ .q, .k }).broad(bias.shape());
        bias = bias.add(mask);
    }
    if (opts.bias) |b| {
        bias = bias.add(b);
    }

    const loc = ctx.mlirCtx().location(@src());
    const op = dialect.stablehlo.custom_call(
        ctx.mlirCtx(),
        &.{ q.value(), k.value(), v.value(), bias.value() },
        .{
            .call_target_name = "__cudnn$fmhaScaleBiasSoftmax",
            .backend_config = backend_config,
            .api_version = 2,
            .has_side_effect = false,
            .output_operand_aliases = &.{},
        },
        &.{
            mlir.ext.mlirType(ctx.mlirCtx(), q.shape()),
            mlir.RankedTensorType.init(&.{0}, mlir.IntegerType(.u8).init(ctx.mlirCtx()).as(mlir.Type).?).as(mlir.Type).?,
        },
        loc,
    );
    const result = Tensor._result(q.shape(), op.result(0)).transpose(q_.shape());
    return result;
}
