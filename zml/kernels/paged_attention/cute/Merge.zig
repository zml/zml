//! Combines the fp32 partials of a split-KV attention kernel: every output row weighs each split by its softmax maximum and adds the sink.
const std = @import("std");

const zml = @import("zml/core");
const cute = zml.kernel.cute;
const B = cute.Builder;

const Self = @This();

batch: i64,
tokens: i64,
kv_heads: i64,
group: i64,
head_dim: i64,
dtype: cute.DType,
sink: bool,
splits: i64,

pub const Kernel = cute.Kernel(Self, .{
    .name = "paged_attention_cute_sm120_merge",
    .inputs = &.{ "partials", "starts", "sink" },
    .outputs = &.{"out"},
    .run = run,
});

// `attention_config` is the Config of the attention kernel that produced `partials`; an unsplit kernel already wrote the final output.
pub fn mergeSplits(attention_config: anytype, partials: zml.Tensor, starts: zml.Tensor, sink: zml.Tensor, final_shape: zml.Shape) zml.Tensor {
    if (attention_config.splits == 1) {
        return partials;
    }

    const c: Self = .{
        .batch = attention_config.batch,
        .tokens = attention_config.tokens,
        .kv_heads = attention_config.kv_heads,
        .group = attention_config.group,
        .head_dim = attention_config.head_dim,
        .dtype = attention_config.dtype,
        .sink = attention_config.sink,
        .splits = attention_config.splits,
    };
    return Kernel.call(.{ .partials = partials, .starts = starts, .sink = sink }, .{ .out = final_shape }, .{
        .cfg = c,
        .grid = .{ @intCast(c.tokens), @intCast(c.kv_heads * c.group), 1 },
        .block = .{ 128, 1, 1 },
    }).out;
}

fn run(b: *B, c: Self) cute.FinishError!void {
    const a = try b.declareArgs(.{
        .partials = .{ .tensor = .{ .dtype = .f32, .shape = &.{ c.splits, c.tokens, c.kv_heads * c.group, c.head_dim + 2 } } },
        .starts = .{ .tensor = .{ .dtype = .i32, .shape = &.{c.batch + 1} } },
        .sink = .{ .tensor = .{ .dtype = .f32, .shape = &.{c.kv_heads * c.group} } },
        .out = .{ .tensor = .{ .dtype = c.dtype, .shape = &.{ c.tokens, c.kv_heads * c.group, c.head_dim } } },
    });
    const token = b.blockIdx().x;
    const head = b.blockIdx().y;
    var valid = b.openIf(token.lt(a.starts.get(.{c.batch})));
    {
        const zero = b.constant(.f32, 0);
        const sink = if (c.sink) a.sink.get(.{head}).mul(std.math.log2e) else b.constant(.f32, -std.math.inf(f32));
        var maximum = b.openFor(0, c.splits, 1, .{sink});
        {
            maximum.yield(.{maximum.carried[0].maximum(a.partials.get(.{ maximum.iv, token, head, c.head_dim }))});
        }
        const m = maximum.results[0].maximum(-std.math.floatMax(f32));
        var dims = b.openFor(b.threadIdx().x, c.head_dim, 128, .{});
        {
            var sum = b.openFor(0, c.splits, 1, .{ zero, if (c.sink) b.exp2Approx(sink.sub(m)) else zero });
            {
                const weight = b.exp2Approx(a.partials.get(.{ sum.iv, token, head, c.head_dim }).sub(m));
                sum.yield(.{
                    sum.carried[0].add(weight.mul(a.partials.get(.{ sum.iv, token, head, dims.iv }))),
                    sum.carried[1].add(weight.mul(a.partials.get(.{ sum.iv, token, head, c.head_dim + 1 }))),
                });
            }
            a.out.set(.{ token, head, dims.iv }, sum.results[0].mul(b.select(sum.results[1].gt(0), b.rcpApprox(sum.results[1]), zero)).to(c.dtype));
            dims.close();
        }
        valid.close();
    }
}
