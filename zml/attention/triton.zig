const std = @import("std");
const stdx = @import("stdx");

const zml = @import("../zml.zig");
const ttir_compile_sandbox = @import("triton_ttir_compile_sandbox.zig");

const RCP_LN2: f32 = 1.4426950216;

pub const Parameters = struct {
    kernel_ttir: ?[:0]const u8 = null,
    softmax_scale: ?f32 = null,

    pub const InitOptions = Parameters;

    pub fn init(opts: InitOptions) Parameters {
        return opts;
    }
};

pub const Metadata = struct {
    pub const InitOptions = struct {};

    pub fn init(_: InitOptions) Metadata {
        return .{};
    }
};

pub const CompileOptions = struct {
    batch: u32 = 1,
    seq_len: u32 = 64,
    num_query_heads: u32 = 8,
    num_kv_heads: u32 = 8,
    head_size: u32 = 128,
    causal: bool = true,
    sliding_window_q: i32 = 0,
    sliding_window_k: i32 = 0,
    softmax_scale: ?f32 = null,
};

pub fn compilePrefillAttentionTtir(allocator: std.mem.Allocator, io: std.Io, opts: CompileOptions) ![:0]u8 {
    const softmax_scale = opts.softmax_scale orelse 1.0 / @sqrt(@as(f32, @floatFromInt(opts.head_size)));
    const args_json = try std.fmt.allocPrint(allocator,
        "{{\"batch\":{},\"seq_len\":{},\"num_query_heads\":{},\"num_kv_heads\":{},\"head_size\":{},\"causal\":{s},\"softmax_scale\":{d},\"sliding_window_q\":{},\"sliding_window_k\":{}}}",
        .{ opts.batch, opts.seq_len, opts.num_query_heads, opts.num_kv_heads, opts.head_size, if (opts.causal) "true" else "false", softmax_scale, opts.sliding_window_q, opts.sliding_window_k },
    );
    defer allocator.free(args_json);

    return ttir_compile_sandbox.getPrefillAttentionTtir(allocator, io, args_json);
}

fn vanillaAttention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    const seq_len = k.dim(.k);
    var attn_mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, q.dtype(), null);
    attn_mask = attn_mask.gatherSlices(zml.Shape.init(.{ .q = q.dim(.q) }, attn_mask.dtype()), token_index.reshape(.{ .coord = 1 }), .{});
    return zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask, .allow_cudnn = true });
}

pub fn attention(q_: zml.Tensor, k_: zml.Tensor, v_: zml.Tensor, token_index_: zml.Tensor, _: Metadata, parameters: Parameters) zml.Tensor {
    const kernel_ttir = parameters.kernel_ttir orelse return vanillaAttention(q_, k_, v_, token_index_);
    if (q_.dim(.q) <= 1) return vanillaAttention(q_, k_, v_, token_index_);

    stdx.debug.assert(q_.shape().hasTag(.b) == null or q_.dim(.b) == 1, "triton prefill backend currently supports batch size 1", .{});

    const has_batch = q_.shape().hasTag(.b) != null;
    const q: zml.Tensor = if (has_batch) q_.squeeze(.b).rename(.{ .q = .tok }) else q_.rename(.{ .q = .tok });
    const k: zml.Tensor = if (has_batch) k_.squeeze(.b).rename(.{ .k = .tok }) else k_.rename(.{ .k = .tok });
    const v: zml.Tensor = if (has_batch) v_.squeeze(.b).rename(.{ .k = .tok }) else v_.rename(.{ .k = .tok });
    const token_index: zml.Tensor = if (has_batch and token_index_.shape().hasTag(.b) != null) token_index_.squeeze(.b) else token_index_;

    const b_start_loc = zml.Tensor.constantTensor(zml.Shape.init(.{1}, .i32), std.mem.sliceAsBytes(&[1]i32{0}));
    const b_seq_len = token_index.addConstant(q_.dim(.q)).reshape(.{1}).convert(.i32);

    const q_strides = q.shape().computeElementStrides();
    const k_strides = k.shape().computeElementStrides();
    const v_strides = v.shape().computeElementStrides();

    const softmax_scale = (parameters.softmax_scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(q_.dim(.hd)))))) * RCP_LN2;

    const block_m: i32 = 64;
    const max_input_len_i32: i32 = @intCast(q_.dim(.q));
    const head_i32: i32 = @intCast(q_.dim(.h));
    const grid: [3]i32 = .{
        1,
        head_i32,
        @intCast(@divFloor(max_input_len_i32 + block_m - 1, block_m)),
    };

    const out = zml.ops.triton(.{
        q,
        k,
        v,
        zml.Tensor.scalar(softmax_scale, .f32),
        b_start_loc,
        b_seq_len,
        zml.Tensor.scalar(q_strides.get(0), .i64),
        zml.Tensor.scalar(q_strides.get(1), .i64),
        zml.Tensor.scalar(k_strides.get(0), .i64),
        zml.Tensor.scalar(k_strides.get(1), .i64),
        zml.Tensor.scalar(v_strides.get(0), .i64),
        zml.Tensor.scalar(v_strides.get(1), .i64),
        zml.Tensor.scalar(q_strides.get(0), .i64),
        zml.Tensor.scalar(q_strides.get(1), .i64),
    }, .{q.shape()}, .{
        .name = "wrapped_fwd_kernel",
        .ir = kernel_ttir,
        .grid = grid,
        .num_stages = 1,
        .num_warps = if (q_.dim(.hd) <= 64) 4 else 8,
        .debug = false,
        .output_operand_aliases = &.{},
    })[0].rename(.{ .tok = .q });

    return if (has_batch)
        out.insertAxes(.q, .{.b})
    else
        out;
}
