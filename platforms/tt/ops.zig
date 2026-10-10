//! TT-specific `stablehlo.custom_call` emission. Pure mlir helpers — no zml
//! dep, so zml/ can call in without forming a cycle. All callers stay in
//! `platforms/tt/`; the zml/ wrappers just Tensor↔value plumb around these.

const std = @import("std");

const mlir = @import("mlir");
const dialects = @import("mlir/dialects");

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
