const stdx = @import("stdx");

const zml = @import("../../../zml.zig");
const paged = @import("../paged_attention.zig");
const sm10x = @import("sm10x/sm10x.zig");
const sm120 = @import("sm120/sm120.zig");
const kernel = @import("attention.zig");

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

pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, opts: paged.AttentionOptions) zml.Tensor {
    stdx.debug.assert(isAvailable(zml.Compiler.current().platform), "CuTe paged attention requires SM89 or newer", .{});
    stdx.debug.assert(q.dtype() == k.dtype() and q.dtype() == v.dtype() and (q.dtype() == .bf16 or q.dtype() == .f16), "CuTe paged attention requires matching BF16 or FP16 Q/K/V", .{});
    stdx.debug.assert(q.dim(.hd) == 32 or q.dim(.hd) == 64 or q.dim(.hd) == 128 or q.dim(.hd) == 256 or q.dim(.hd) == 512, "CuTe paged attention supports head dimensions 32, 64, 128, 256 and 512", .{});
    if (sm10x.isAvailable(zml.Compiler.current().platform) and opts.mask != .indices)
        return sm10x.pagedAttention(parameters, q, k, v, opts);
    if (sm120.isAvailable(zml.Compiler.current().platform) and opts.mask != .indices)
        return sm120.pagedAttention(parameters, q, k, v, opts);
    return zml.ops.manualComputation(Context.body, Context{ .q = q, .k = k, .v = v, .parameters = parameters, .opts = opts }, q.shape());
}

const Context = struct {
    q: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
    parameters: Parameters,
    opts: paged.AttentionOptions,

    fn body(self: Context, output_shape: zml.Shape) zml.Tensor {
        const q = self.q.transpose(.{ .b, .hkv, .hg, .hd });
        const k = self.k.transpose(.{ .page, .k_chunk, .hkv, .hd });
        const v = self.v.transpose(.{ .page, .k_chunk, .hkv, .hd });
        stdx.debug.assert(q.dim(.hkv) == k.dim(.hkv) and k.shape().eql(v.shape()) and k.dim(.hd) == q.dim(.hd) and q.dim(.hg) > 0, "CuTe paged attention requires aligned local Q/KV heads and matching caches", .{});
        const p = self.parameters;
        const indices = switch (self.opts.mask) {
            .indices => |indices| indices.transpose(.{ .b, .topk }),
            else => zml.Tensor.zeroes(.init(.{ q.dim(.b), 1 }, .i32)),
        };
        const cfg: kernel.Config = .{
            .head_dim = q.dim(.hd),
            .tokens = q.dim(.b),
            .batch = p.block_table.dim(.b),
            .kv_heads = k.dim(.hkv),
            .group = q.dim(.hg),
            .pages = k.dim(.page),
            .pages_per_seq = p.block_table.dim(.p),
            .page_size = k.dim(.k_chunk),
            .dtype = zml.kernel.cute.from(q.dtype()),
            .causal = self.opts.mask.isCausal(),
            .window = self.opts.mask.slidingWindow() orelse 0,
            .scale = self.opts.scale orelse (1 / @sqrt(@as(f32, @floatFromInt(q.dim(.hd))))),
            .sink = self.opts.sink != null,
            .indexed = self.opts.mask == .indices,
            .topk = indices.dim(-1),
        };
        const sink = if (self.opts.sink) |s| s.convert(.f32).reshape(.{cfg.kv_heads * cfg.group}) else zml.Tensor.zeroes(.init(.{cfg.kv_heads * cfg.group}, .f32));
        return kernel.Kernel.call(.{
            .q = q,
            .k = k,
            .v = v,
            .table = p.block_table,
            .lengths = p.seq_lens,
            .starts = p.query_start_len,
            .sink = sink,
            .indices = indices,
        }, .{ .out = q.shape() }, .{
            .cfg = cfg,
            .grid = .{ @intCast(cfg.tokens), @intCast(cfg.kv_heads * cfg.group), 1 },
            .block = .{ 32, 1, 1 },
        }).out.transpose(output_shape);
    }
};
