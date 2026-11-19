//! FP32 online-softmax state update shared by dense TCL attention kernels.

const tcl = @import("dsl/tcl/builder");

pub const Update = struct {
    weights: tcl.Tensor,
    sum: tcl.Tensor,
    max: tcl.Tensor,
    rescale: tcl.Tensor,

    pub fn accumulate(self: Update, b: *tcl.Builder, previous: tcl.Tensor, current: tcl.Tensor, opts: tcl.TensorOperation.Options) tcl.FinishError!tcl.Tensor {
        const op = b.tensorOperation(opts);
        return op.commit(op.fetch(previous, .{}).mulf(op.fetch(self.rescale, .{})).addf(op.fetch(current, .{})), .{});
    }
};

/// Scores are already scaled and masked. Keep the caller's elementwise tactics.
pub fn update(
    b: *tcl.Builder,
    masked: tcl.Tensor,
    mask: tcl.Tensor,
    keyAxis: tcl.Axis,
    sumAcc: tcl.Tensor,
    maxAcc: tcl.Tensor,
    opts: tcl.TensorOperation.Options,
) tcl.FinishError!Update {
    var op = b.tensorOperation(.{});
    const maxCur = try op.commit(op.fetch(masked, .{}).reduce(&.{keyAxis}, .maxf), .{});
    op = b.tensorOperation(.{ .tactic = .elementwise });
    const maxNew = try op.commit(op.fetch(maxCur, .{}).maxf(op.fetch(maxAcc, .{})), .{});
    op = b.tensorOperation(.{ .tactic = .elementwise });
    const rescale = try op.commit(op.fetch(maxAcc, .{}).subf(op.fetch(maxNew, .{})).exp(), .{});

    op = b.tensorOperation(opts);
    // Mask exponentials too: exp(sentinel - sentinel) is 1 for masked entries.
    const exp = op.fetch(masked, .{}).subf(op.fetch(maxNew, .{})).exp();
    const weights = try op.commit(op.where(op.fetch(mask, .{}), .ne, 0, exp, 0.0), .{});
    op = b.tensorOperation(.{});
    const sumCur = try op.commit(op.fetch(weights, .{}).reduce(&.{keyAxis}, .addf), .{});
    op = b.tensorOperation(opts);
    const sumNew = try op.commit(op.fetch(sumAcc, .{}).mulf(op.fetch(rescale, .{})).addf(op.fetch(sumCur, .{})), .{});

    return .{ .weights = weights, .sum = sumNew, .max = maxNew, .rescale = rescale };
}
