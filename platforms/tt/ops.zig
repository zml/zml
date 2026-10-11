//! TT-specific `stablehlo.custom_call` emission. Pure mlir helpers — no zml
//! dep, so zml/ can call in without forming a cycle. All callers stay in
//! `platforms/tt/`; the zml/ wrappers just Tensor↔value plumb around these.

const std = @import("std");

const mlir = @import("mlir");
const dialects = @import("mlir/dialects");

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

/// `tt.tt_lang_op` carrying a pre-built kernel artifact (see tt-mlir's
/// `--ttnn-lower-tt-lang-to-generic`): runs custom kernels on Tenstorrent.
/// `inputs` are the "in" operands; one result per `out_types` entry. tt-mlir
/// allocates the outputs and passes `inputs ++ outputs` to the kernels as
/// io-tensors, in that order.
pub fn kernel(
    ctx: *mlir.Context,
    block: *mlir.Block,
    inputs: []const *const mlir.Value,
    out_types: []const *const mlir.Type,
    kernel_id: []const u8,
    arg_roles: []const u8,
    artifact: []const u8,
) *mlir.Operation {
    const frontend_attrs = mlir.Attribute.dict(ctx, &.{
        .named(ctx, "kernel_id", .string(ctx, kernel_id)),
        .named(ctx, "version_tag", .string(ctx, "zml")),
        .named(ctx, "arg_roles", .string(ctx, arg_roles)),
        .named(ctx, "kernel_artifact", .string(ctx, artifact)),
    });
    return dialects.stablehlo.custom_call(
        ctx,
        inputs,
        out_types,
        .{
            .call_target_name = "tt.tt_lang_op",
            .has_side_effect = false,
            .additional_attributes = &.{
                .named(ctx, "mhlo.frontend_attributes", frontend_attrs),
            },
        },
        .unknown(ctx),
    ).appendTo(block);
}
