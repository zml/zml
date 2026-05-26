//! TT-specific `stablehlo.custom_call` emission. Pure mlir helpers — no zml
//! dep, so zml/ can call in without forming a cycle. All callers stay in
//! `platforms/tt/`; the zml/ wrappers just Tensor↔value plumb around these.

const std = @import("std");

const mlir = @import("mlir");
const dialects = @import("mlir/dialects");

/// `tt.sharding_constraint` — tt-xla's frontend doesn't consume the upstream
/// `sdy.sharding_constraint` op; the custom_call carries the same sdy attr as
/// a frontend-attribute string.
pub fn shardingConstraint(
    ctx: *mlir.Context,
    block: *mlir.Block,
    value: *const mlir.Value,
    sdy_sharding_str: []const u8,
) *const mlir.Value {
    const frontend_attributes: *const mlir.Attribute = .dict(ctx, &.{
        .named(ctx, "xla.sdy.sharding", .string(ctx, sdy_sharding_str)),
    });
    const op = dialects.stablehlo.custom_call(
        ctx,
        &.{value},
        &.{value.type_()},
        .{
            .call_target_name = "tt.sharding_constraint",
            .has_side_effect = false,
            .backend_config = .{ .original = "" },
            .additional_attributes = &.{
                .named(ctx, "mhlo.frontend_attributes", frontend_attributes),
            },
        },
        .unknown(ctx),
    ).appendTo(block);
    return op.result(0);
}

/// `tt.mark_argument` — tags a block arg as `<parameter>` so tt-xla keeps it
/// device-resident across trace captures. `has_side_effect=true` prevents the
/// canonicalizer from dropping the op.
pub fn markArgument(
    ctx: *mlir.Context,
    block: *mlir.Block,
    arg: *const mlir.Value,
    name: []const u8,
) void {
    const frontend_attributes: *const mlir.Attribute = .dict(ctx, &.{
        .named(ctx, "ttcore.argument_type", .string(ctx, "parameter")),
        .named(ctx, "ttir.name", .string(ctx, name)),
    });
    _ = dialects.stablehlo.custom_call(
        ctx,
        &.{arg},
        &.{arg.type_()},
        .{
            .call_target_name = "tt.mark_argument",
            .has_side_effect = true,
            .backend_config = .{ .original = "" },
            .additional_attributes = &.{
                .named(ctx, "mhlo.frontend_attributes", frontend_attributes),
            },
        },
        .unknown(ctx),
    ).appendTo(block);
}

/// `tenstorrent.rms_norm` — fused RMSNorm + affine scale. Math-decomposed
/// RMSNorm trips `sdy.reshard` "duplicate axis ref" at sharding propagation;
/// the single op sidesteps it.
pub fn rmsNormFused(
    ctx: *mlir.Context,
    block: *mlir.Block,
    x: *const mlir.Value,
    weight: *const mlir.Value,
    out_type: *const mlir.Type,
    normalized_dim: i64,
    eps: f32,
) *const mlir.Value {
    const composite_attrs = mlir.Attribute.dict(ctx, &.{
        .named(ctx, "normalized_shape", mlir.Attribute.array(ctx, &.{
            mlir.Attribute.int(ctx, .i64, normalized_dim),
        })),
        .named(ctx, "epsilon", mlir.Attribute.float(ctx, .f32, eps)),
    });
    const op = dialects.stablehlo.custom_call(
        ctx,
        &.{ x, weight },
        &.{out_type},
        .{
            .call_target_name = "tenstorrent.rms_norm",
            .has_side_effect = false,
            .additional_attributes = &.{
                .named(ctx, "tt.has_custom_sharding", .unit(ctx)),
                .named(ctx, "tt.composite_attributes", composite_attrs),
            },
        },
        .unknown(ctx),
    ).appendTo(block);
    return op.result(0);
}

/// `tt.sampling` — temperature scale + multinomial sample on pre-top-k'd
/// candidates in one op. Inputs: `{values, indices, k, p, t}`. Broken at
/// `trace + opt>=1`: tt-xla #4570 (workaround `ZML_TT_OPTIMIZATION_LEVEL=0`).
pub fn sampling(
    ctx: *mlir.Context,
    block: *mlir.Block,
    vals: *const mlir.Value,
    idxs: *const mlir.Value,
    k: *const mlir.Value,
    p: *const mlir.Value,
    t: *const mlir.Value,
    out_type: *const mlir.Type,
    seed: []const u8,
) *const mlir.Value {
    const frontend_attrs = mlir.Attribute.dict(ctx, &.{
        .named(ctx, "seed", .string(ctx, seed)),
    });
    const op = dialects.stablehlo.custom_call(
        ctx,
        &.{ vals, idxs, k, p, t },
        &.{out_type},
        .{
            .call_target_name = "tt.sampling",
            .has_side_effect = false,
            .additional_attributes = &.{
                .named(ctx, "mhlo.frontend_attributes", frontend_attrs),
            },
        },
        .unknown(ctx),
    ).appendTo(block);
    return op.result(0);
}

/// `tt.scaled_dot_product_attention[_decode]` — TTNN-native SDPA. Operands are
/// `[B,H,S,D]` (decode `q` is `[1,B,H,D]`); decode passes `cur_pos[B]` as a
/// 4th operand. Attrs are parsed as strings (`parseBoolFromStringAttr`).
pub fn scaledDotProductAttention(
    ctx: *mlir.Context,
    block: *mlir.Block,
    q: *const mlir.Value,
    k: *const mlir.Value,
    v: *const mlir.Value,
    cur_pos: ?*const mlir.Value,
    out_type: *const mlir.Type,
    is_decode: bool,
) *const mlir.Value {
    var attrs_buf: [3]mlir.NamedAttribute = undefined;
    var n_attrs: usize = 0;
    if (!is_decode) {
        attrs_buf[n_attrs] = .named(ctx, "is_causal", .string(ctx, "true"));
        n_attrs += 1;
    }
    attrs_buf[n_attrs] = .named(ctx, "has_attention_mask", .string(ctx, "false"));
    n_attrs += 1;
    attrs_buf[n_attrs] = .named(ctx, "has_attention_sink", .string(ctx, "false"));
    n_attrs += 1;
    const frontend_attrs = mlir.Attribute.dict(ctx, attrs_buf[0..n_attrs]);

    var inputs_buf: [4]*const mlir.Value = .{ q, k, v, undefined };
    var n_inputs: usize = 3;
    if (cur_pos) |cp| {
        inputs_buf[3] = cp;
        n_inputs = 4;
    }

    const op_name = if (is_decode)
        "tt.scaled_dot_product_attention_decode"
    else
        "tt.scaled_dot_product_attention";

    const op = dialects.stablehlo.custom_call(
        ctx,
        inputs_buf[0..n_inputs],
        &.{out_type},
        .{
            .call_target_name = op_name,
            .has_side_effect = false,
            .additional_attributes = &.{
                .named(ctx, "mhlo.frontend_attributes", frontend_attrs),
            },
        },
        .unknown(ctx),
    ).appendTo(block);
    return op.result(0);
}
