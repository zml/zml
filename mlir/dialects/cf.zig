//! Layer A builders for the `cf` dialect (unstructured control flow:
//! non-region branches on a CFG of SSA blocks).
//!
//! Mirrors ControlFlowOps.td one-to-one:
//!   * `cf.assert`   — runtime assertion with string message.
//!   * `cf.br`       — unconditional branch with forwarded operands.
//!   * `cf.cond_br`  — conditional branch (i1 → true/false dest + operands).
//!   * `cf.switch`   — integer flag → default dest or matching case dest.

const std = @import("std");

const mlir = @import("mlir");
const stdx = @import("stdx");

// =============================================================================
// cf.assert
// =============================================================================

/// cf.assert %cond, "message" — aborts execution with `msg` when `cond` is
/// false. `cond` is i1.
pub fn assert(
    ctx: *mlir.Context,
    cond: *const mlir.Value,
    msg: []const u8,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "cf.assert", .{
        .operands = .{ .flat = &.{cond} },
        .attributes = &.{
            .named(ctx, "msg", mlir.stringAttribute(ctx, msg)),
        },
        .location = location,
    });
}

// =============================================================================
// cf.br
// =============================================================================

/// cf.br ^dest(operands...) — unconditional branch. Terminator of its block.
/// The types of `dest_operands` must match the block arguments of `dest`.
pub fn br(
    ctx: *mlir.Context,
    dest: *mlir.Block,
    dest_operands: []const *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    _ = ctx;
    var state: mlir.OperationState = .init("cf.br", location);
    state.addOperands(dest_operands);
    var successors: [1]*const mlir.Block = .{dest};
    state.addSuccessors(&successors);
    return mlir.Operation.init(&state) catch @panic("Failed to create cf.br");
}

// =============================================================================
// cf.cond_br
// =============================================================================

/// cf.cond_br %cond, ^trueDest(trueOps...), ^falseDest(falseOps...) — branch
/// on an i1. Terminator of its block.
///
/// Operand layout (flat) is `[cond, *trueOperands, *falseOperands]` with an
/// `operandSegmentSizes = [1, |trueOperands|, |falseOperands|]` attribute per
/// the `AttrSizedOperandSegments` trait on the op.
///
/// `branch_weights` (optional) lets profile-guided lowering bias the branch;
/// when provided, its length must be 2 (true weight, false weight).
pub fn cond_br(
    ctx: *mlir.Context,
    cond: *const mlir.Value,
    true_dest: *mlir.Block,
    true_operands: []const *const mlir.Value,
    false_dest: *mlir.Block,
    false_operands: []const *const mlir.Value,
    branch_weights: ?[]const i32,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 128) = .empty;
    operands_buf.appendAssumeCapacity(cond);
    operands_buf.appendSliceAssumeCapacity(true_operands);
    operands_buf.appendSliceAssumeCapacity(false_operands);

    const seg_sizes = [3]i32{ 1, @intCast(true_operands.len), @intCast(false_operands.len) };

    var state: mlir.OperationState = .init("cf.cond_br", location);
    state.addOperands(operands_buf.constSlice());
    var successors: [2]*const mlir.Block = .{ true_dest, false_dest };
    state.addSuccessors(&successors);

    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "operandSegmentSizes", mlir.denseArrayAttribute(ctx, .i32, &seg_sizes)));
    if (branch_weights) |w| {
        std.debug.assert(w.len == 2);
        attrs.appendAssumeCapacity(.named(ctx, "branch_weights", mlir.denseArrayAttribute(ctx, .i32, w)));
    }
    state.addAttributes(attrs.constSlice());

    return mlir.Operation.init(&state) catch @panic("Failed to create cf.cond_br");
}

// =============================================================================
// cf.switch
// =============================================================================

/// cf.switch %flag : iN, [ default: ^def(defOps...), caseN: ^bbN(opsN...) ]
///
/// `flag` is any integer type. `case_values` are matched signed; the matching
/// case's block is jumped to (with the corresponding operand segment
/// forwarded). If no case matches, `default_dest` is taken with
/// `default_operands`.
///
/// Operand layout: `[flag, *defaultOperands, *caseOperands[0], ...,
/// *caseOperands[N-1]]`. `case_operand_segments[i]` carries the length of
/// each case's operand segment. The `AttrSizedOperandSegments` trait tracks
/// the top-level split between `defaultOperands` and the lumped
/// `caseOperands`.
pub fn switch_(
    ctx: *mlir.Context,
    flag: *const mlir.Value,
    default_dest: *mlir.Block,
    default_operands: []const *const mlir.Value,
    case_values: []const i32,
    case_dests: []const *mlir.Block,
    case_operands: []const []const *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    std.debug.assert(case_values.len == case_dests.len);
    std.debug.assert(case_values.len == case_operands.len);

    var operands_buf: stdx.BoundedArray(*const mlir.Value, 256) = .empty;
    operands_buf.appendAssumeCapacity(flag);
    operands_buf.appendSliceAssumeCapacity(default_operands);

    var case_seg_sizes: stdx.BoundedArray(i32, 64) = .empty;
    var case_total: i32 = 0;
    for (case_operands) |segment| {
        operands_buf.appendSliceAssumeCapacity(segment);
        case_seg_sizes.appendAssumeCapacity(@intCast(segment.len));
        case_total += @intCast(segment.len);
    }

    // Top-level AttrSizedOperandSegments: [flag=1, defaultOperands, caseOperands lumped].
    const top_seg_sizes = [3]i32{ 1, @intCast(default_operands.len), case_total };

    var state: mlir.OperationState = .init("cf.switch", location);
    state.addOperands(operands_buf.constSlice());

    var successors_buf: stdx.BoundedArray(*const mlir.Block, 64) = .empty;
    successors_buf.appendAssumeCapacity(default_dest);
    for (case_dests) |d| successors_buf.appendAssumeCapacity(d);
    state.addSuccessors(@constCast(successors_buf.constSlice()));

    // case_values is stored as a DenseIntElementsAttr (tensor<N x iFlagBits>);
    // the td uses `AnyIntElementsAttr`, and the parser prints decimal literals
    // matching the flag's bit width. We use i32 for the common path (matches
    // Python Triton's flag type for pid compares).
    const shaped = mlir.RankedTensorType.get(
        &.{@intCast(case_values.len)},
        mlir.integerType(ctx, .i32),
        null,
    ).shaped();

    state.addAttributes(&.{
        .named(ctx, "operandSegmentSizes", mlir.denseArrayAttribute(ctx, .i32, &top_seg_sizes)),
        .named(ctx, "case_values", mlir.denseElementsAttribute(shaped, case_values)),
        .named(ctx, "case_operand_segments", mlir.denseArrayAttribute(ctx, .i32, case_seg_sizes.constSlice())),
    });

    return mlir.Operation.init(&state) catch @panic("Failed to create cf.switch");
}

test {
    std.testing.refAllDecls(@This());
}
