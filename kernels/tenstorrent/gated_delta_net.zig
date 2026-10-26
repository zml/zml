//! Recurrent Gated DeltaNet in TTIR. Sequence positions are statically unrolled;
//! active_length masks the tail so prefill padding cannot change recurrent state.
const std = @import("std");
const mlir = @import("mlir");
const builder = @import("builder.zig");

pub fn run(b: *builder.Builder, _: void) builder.FinishError!void {
    const q = b.arg(0);
    const k = b.arg(1);
    const v = b.arg(2);
    const alpha = b.arg(3);
    const beta = b.arg(4);
    var state = b.arg(5);
    const active_length = b.arg(6);
    const qs = q.shape();
    const vs = v.shape();
    const ss = state.shape();
    std.debug.assert(qs.len == 3 and vs.len == 3 and ss.len == 3);
    const seq: usize = @intCast(qs.get(0));
    const heads = qs.get(1);
    const key_dim = qs.get(2);
    const value_dim = vs.get(2);
    std.debug.assert(seq > 0);
    const outputs = try b.allocator.alloc(builder.Value, seq);
    defer b.allocator.free(outputs);
    const element_type = mlir.Type.float(b.ctx, .f32);
    std.debug.assert(state.tensorType().elementType().eql(element_type));
    const y_type = mlir.Type.rankedTensor(&.{ 1, heads, value_dim }, element_type);
    const zero = b.splat(y_type, @as(f32, 0));
    for (0..seq) |i| {
        const ki = b.sequenceStep(k, i);
        const qi = b.sequenceStep(q, i);
        const vi = b.sequenceStep(v, i);
        const ai = b.reshape(b.sequenceStep(alpha, i), &.{ heads, 1 });
        const bi = b.reshape(b.sequenceStep(beta, i), &.{ heads, 1 });
        const prediction = b.reshape(b.matmul(state, b.reshape(ki, &.{ heads, key_dim, 1 }), false, false), &.{ heads, value_dim });
        const decayed_prediction = b.binary(.multiply, prediction, ai);
        const delta = b.binary(.multiply, b.binary(.subtract, vi, decayed_prediction), bi);
        const update = b.binary(.multiply, b.reshape(delta, &.{ heads, value_dim, 1 }), b.reshape(ki, &.{ heads, 1, key_dim }));
        const next_state = b.binary(.add, b.binary(.multiply, state, b.reshape(ai, &.{ heads, 1, 1 })), update);
        const yi = b.reshape(b.matmul(next_state, b.reshape(qi, &.{ heads, key_dim, 1 }), false, false), &.{ 1, heads, value_dim });
        const step = b.splat(active_length.type_(), @as(i32, @intCast(i)));
        const active = b.binary(.lt, step, active_length);
        state = b.where(active, next_state, state);
        outputs[i] = b.where(active, yi, zero);
    }
    const ys = if (seq == 1) outputs[0] else try b.concat(outputs, 0);
    b.returns(&.{ state, ys });
}
