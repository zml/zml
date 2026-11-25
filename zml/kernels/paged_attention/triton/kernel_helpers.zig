const zml = @import("zml/core");
const tri = zml.kernel.triton;
const Builder = tri.Builder;
const Value = tri.Value;
const DType = tri.DType;

/// `log_2(e)` — pre-multiply with this to turn `exp(x)` into `exp2(x * RCP_LN2)`.
pub const RCP_LN2: f32 = 1.4426950408889634;

pub fn isFp8(dt: DType) bool {
    return dt == .f8e4m3fn or dt == .f8e5m2;
}

pub fn cdivFn(x: Value, y: anytype) Value {
    return switch (@typeInfo(@TypeOf(y))) {
        .comptime_int, .comptime_float => x.add(y - 1).div(y),
        else => x.add(y).sub(1).div(y),
    };
}

pub fn applySoftcap(k: *Builder, s_val: Value, x: Value) Value {
    const sdiv = s_val.div(x);
    const p1 = k.exp2(sdiv);
    const p2 = k.exp2(k.negf(sdiv));
    return x.mul(p1.sub(p2)).div(p1.add(p2));
}

pub fn findSeqIdx(
    k: *Builder,
    query_start_len_ptr: Value,
    target_idx: Value,
    num_seqs: Value,
    block_q: i64,
    use_q_block_mode: bool,
) Value {
    const left_init = k.constant(.i32, 0);

    var w = k.openWhile(.{ left_init, num_seqs }, .{ .i32, .i32 });
    {
        const left = w.before_carried[0];
        const right = w.before_carried[1];
        w.yieldBefore(left.lt(right), .{ left, right });
    }
    {
        const left = w.after_carried[0];
        const right = w.after_carried[1];
        const mid = left.add(right).div(2);
        const val = k.load(query_start_len_ptr.addPtr(mid));
        const mid_val = if (use_q_block_mode)
            val.div(@as(i32, @intCast(block_q))).add(mid)
        else
            val;

        var i = k.openIfElse(mid_val.le(target_idx), .{ .i32, .i32 });
        {
            i.yieldThen(.{ mid.add(1), right });
        }
        {
            i.yieldElse(.{ left, mid });
        }
        w.yieldAfter(.{ i.results[0], i.results[1] });
    }

    return w.results[0].sub(1);
}

pub fn dequantKv(loaded: Value, scale: Value, kv_is_fp8: bool, q_dtype: DType) Value {
    if (!kv_is_fp8 or isFp8(q_dtype)) return loaded;
    return loaded.to(.f32).mul(scale).to(q_dtype);
}
