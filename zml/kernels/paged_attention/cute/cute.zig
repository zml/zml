const stdx = @import("stdx");

const zml = @import("zml/core");
pub const AttentionOptions = @import("zml/kernels/paged_attention/options").AttentionOptions;
const kernel = @import("attention.zig");
const sm10x = @import("sm10x/sm10x.zig");
const sm120 = @import("sm120/sm120.zig");

pub const Options = struct {
    batch_size: usize,
    max_num_pages: usize,
    max_seqlen_q: usize,
    is_prefill: bool,

    pub fn isPrefill(self: Options) bool {
        return self.is_prefill;
    }

    pub fn maxNumPages(self: Options) usize {
        return self.max_num_pages;
    }
};

pub const Parameters = struct {
    block_table: zml.Tensor,
    seq_lens: zml.Tensor,
    query_start_len: zml.Tensor,
    options_: Options,

    pub fn init(options_: Options) Parameters {
        return .{
            .block_table = .init(.{ .b = options_.batch_size, .p = options_.max_num_pages }, .i32),
            .seq_lens = .init(.{ .b = options_.batch_size }, .i32),
            .query_start_len = .init(.{ .b = options_.batch_size + 1 }, .i32),
            .options_ = options_,
        };
    }

    pub fn allocationSize(self: Parameters) usize {
        return self.block_table.byteSize() + self.seq_lens.byteSize() + self.query_start_len.byteSize();
    }

    pub fn options(self: Parameters) Options {
        return self.options_;
    }
};

// TODO(Corentin): This should be moved in the platform probably
pub fn multiprocessorCount(platform: *const zml.Platform) usize {
    const attribute = platform.devices[0].pjrt_desc.attribute(platform.pjrt_api, "core_count") orelse return 1;
    if (attribute.int64 <= 0) return 1;
    return @intCast(attribute.int64);
}

test {
    _ = kernel;
    _ = sm10x;
    _ = sm120;
}

pub fn isAvailable(platform: *const zml.Platform) bool {
    if (platform.target != .cuda) return false;
    const cc = zml.platform.cuda.computeCapability(platform) orelse return false;
    return cc.major > 8 or (cc.major == 8 and cc.minor >= 9);
}

pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, opts: AttentionOptions) zml.Tensor {
    stdx.debug.assert(isAvailable(zml.Compiler.current().platform), "CuTe paged attention requires SM89 or newer", .{});
    stdx.debug.assert(q.dtype() == k.dtype() and q.dtype() == v.dtype() and (q.dtype() == .bf16 or q.dtype() == .f16), "CuTe paged attention requires matching BF16 or FP16 Q/K/V", .{});
    stdx.debug.assert(q.dim(.hd) == 32 or q.dim(.hd) == 64 or q.dim(.hd) == 128 or q.dim(.hd) == 256 or q.dim(.hd) == 512, "CuTe paged attention supports head dimensions 32, 64, 128, 256 and 512", .{});
    if (sm10x.isAvailable(zml.Compiler.current().platform) and opts.mask != .indices)
        return sm10x.pagedAttention(parameters, q, k, v, opts);
    if (sm120.isAvailable(zml.Compiler.current().platform) and opts.mask != .indices)
        return sm120.pagedAttention(parameters, q, k, v, opts);
    const q_ = q.transpose(.{ .b, .hkv, .hg, .hd });
    const k_ = k.transpose(.{ .page, .k_chunk, .hkv, .hd });
    const v_ = v.transpose(.{ .page, .k_chunk, .hkv, .hd });
    stdx.debug.assert(q_.dim(.hkv) == k_.dim(.hkv) and k_.shape().eql(v_.shape()) and k_.dim(.hd) == q_.dim(.hd) and q_.dim(.hg) > 0, "CuTe paged attention requires aligned local Q/KV heads and matching caches", .{});
    const indices = switch (opts.mask) {
        .indices => |indices| indices.transpose(.{ .b, .topk }),
        else => zml.Tensor.zeroes(.init(.{ q_.dim(.b), 1 }, .i32)),
    };
    const cfg: kernel.Config = .{
        .head_dim = q_.dim(.hd),
        .tokens = q_.dim(.b),
        .batch = parameters.block_table.dim(.b),
        .kv_heads = k_.dim(.hkv),
        .group = q_.dim(.hg),
        .pages = k_.dim(.page),
        .pages_per_seq = parameters.block_table.dim(.p),
        .page_size = k_.dim(.k_chunk),
        .dtype = zml.kernel.cute.from(q_.dtype()),
        .causal = opts.mask.isCausal(),
        .window = opts.mask.slidingWindow() orelse 0,
        .scale = opts.scale orelse (1 / @sqrt(@as(f32, @floatFromInt(q_.dim(.hd))))),
        .sink = opts.sink != null,
        .indexed = opts.mask == .indices,
        .topk = indices.dim(-1),
    };
    const sink = if (opts.sink) |s| s.convert(.f32).reshape(.{cfg.kv_heads * cfg.group}) else zml.Tensor.zeroes(.init(.{cfg.kv_heads * cfg.group}, .f32));
    return kernel.Kernel.call(.{
        .q = q_,
        .k = k_,
        .v = v_,
        .table = parameters.block_table,
        .lengths = parameters.seq_lens,
        .starts = parameters.query_start_len,
        .sink = sink,
        .indices = indices,
    }, .{ .out = q_.shape() }, .{
        .cfg = cfg,
        .grid = .{ @intCast(cfg.tokens), @intCast(cfg.kv_heads * cfg.group), 1 },
        .block = .{ 32, 1, 1 },
    }).out.transpose(q.shape());
}
