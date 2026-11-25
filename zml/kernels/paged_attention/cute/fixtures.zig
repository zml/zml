const std = @import("std");

const zml = @import("zml");

const cases = @import("cases.zig");

pub const Backend = enum { cute, cute_generic };

pub fn backends(c: cases.Case) []const Backend {
    return if (c.mask == .indices) &.{.cute} else std.enums.values(Backend);
}

pub const Fixture = struct {
    q: []zml.floats.BFloat16,
    k: []zml.floats.BFloat16,
    v: []zml.floats.BFloat16,
    table: []i32,
    lengths: []i32,
    starts: []i32,
    sinks: []f32,
    indices: []i32,

    // The caller owns an arena; all arrays live until the case is complete.
    pub fn init(a: std.mem.Allocator, c: cases.Case) !Fixture {
        const f: Fixture = .{
            .q = try a.alloc(zml.floats.BFloat16, c.tokens() * c.heads() * c.head_dim),
            .k = try a.alloc(zml.floats.BFloat16, c.batch * c.pages() * cases.page_size * c.kv_heads * c.head_dim),
            .v = try a.alloc(zml.floats.BFloat16, c.batch * c.pages() * cases.page_size * c.kv_heads * c.head_dim),
            .table = try a.alloc(i32, c.batch * c.pages()),
            .lengths = try a.alloc(i32, c.batch),
            .starts = try a.alloc(i32, c.batch + 1),
            .sinks = try a.alloc(f32, c.heads()),
            .indices = try a.alloc(i32, c.tokens() * c.topk),
        };
        var rng = std.Random.DefaultPrng.init(42);
        const random = rng.random();
        for (f.q) |*v| v.* = .fromF32(random.float(f32) * 2 - 1);
        for (f.k) |*v| v.* = .fromF32(random.float(f32) * 2 - 1);
        for (f.v) |*v| v.* = .fromF32(random.float(f32) * 2 - 1);
        for (f.sinks, 0..) |*v, h| v.* = c.sink_value orelse (@as(f32, @floatFromInt(h % 5)) - 2);
        @memset(f.indices, -1);
        f.starts[0] = 0;
        for (0..c.batch) |s| {
            const len = c.kvLen(s);
            f.lengths[s] = @intCast(len);
            f.starts[s + 1] = f.starts[s] + @as(i32, @intCast(c.queryLen(s)));
            for (0..c.pages()) |p| {
                f.table[s * c.pages() + p] = if (p * cases.page_size < len) @intCast(s * c.pages() + p) else -1;
            }
            const live_pages = std.math.divCeil(usize, len, cases.page_size) catch unreachable;
            if (c.shuffle_pages) random.shuffle(i32, f.table[s * c.pages() ..][0..live_pages]);
            // Unused physical slots must not contribute to PV, even via zero * NaN.
            for (len..c.pages() * cases.page_size) |j| {
                const page = if (j / cases.page_size < live_pages)
                    @as(usize, @intCast(f.table[s * c.pages() + j / cases.page_size]))
                else
                    s * c.pages() + j / cases.page_size;
                const offset = (page * cases.page_size + j % cases.page_size) * c.kv_heads * c.head_dim;
                @memset(f.k[offset..][0 .. c.kv_heads * c.head_dim], .fromF32(std.math.nan(f32)));
                @memset(f.v[offset..][0 .. c.kv_heads * c.head_dim], .fromF32(std.math.nan(f32)));
            }
            for (@intCast(f.starts[s])..@intCast(f.starts[s + 1])) |t| {
                const l: i32 = @intCast(len);
                const boundary = [_]i32{ 0, l - 1, @divTrunc(l, 2), -1, l, 0, 1, l - 2 };
                for (f.indices[t * c.topk ..][0..c.topk], 0..) |*index, j| {
                    index.* = if (j < boundary.len) boundary[j] else switch (j % 17) {
                        0 => -1,
                        1 => l,
                        2 => 0,
                        else => @intCast((j * 37 + t * 13) % len),
                    };
                }
            }
        }
        if (c.empty_indices) {
            for (f.indices, 0..) |*index, i| index.* = if (i % 2 == 0) -1 else @intCast(c.kv_len);
            @memset(f.k, .fromF32(std.math.nan(f32)));
            @memset(f.v, .fromF32(std.math.nan(f32)));
        }
        const active_tokens: usize = @intCast(f.starts[c.batch]);
        @memset(f.q[active_tokens * c.heads() * c.head_dim ..], .fromF32(std.math.nan(f32)));
        return f;
    }

    pub fn quantizeF16(f: Fixture) void {
        // Both host reference and native FP16 uploads receive identical values.
        for ([_][]zml.floats.BFloat16{ f.q, f.k, f.v }) |values| {
            for (values) |*v| if (std.math.isFinite(v.toF32())) {
                v.* = .fromF32(@round(v.toF32() * 64) / 64);
            };
        }
    }

    pub fn kvOffset(f: Fixture, c: cases.Case, seq: usize, key: usize, head: usize) usize {
        const page: usize = @intCast(f.table[seq * c.pages() + key / cases.page_size]);
        return ((page * cases.page_size + key % cases.page_size) * c.kv_heads + head) * c.head_dim;
    }
};

pub const Inputs = struct {
    q: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
    table: zml.Tensor,
    lengths: zml.Tensor,
    starts: zml.Tensor,
    sinks: zml.Tensor,
    indices: zml.Tensor,

    pub fn init(c: cases.Case) Inputs {
        return initDType(c, .bf16);
    }

    pub fn initDType(c: cases.Case, dtype: zml.DataType) Inputs {
        return .{
            .q = .init(.{ .b = c.tokens(), .hkv = c.kv_heads, .hg = c.group, .hd = c.head_dim }, dtype),
            .k = .init(.{ .page = c.batch * c.pages(), .k_chunk = cases.page_size, .hkv = c.kv_heads, .hd = c.head_dim }, dtype),
            .v = .init(.{ .page = c.batch * c.pages(), .k_chunk = cases.page_size, .hkv = c.kv_heads, .hd = c.head_dim }, dtype),
            .table = .init(.{ .b = c.batch, .p = c.pages() }, .i32),
            .lengths = .init(.{ .b = c.batch }, .i32),
            .starts = .init(.{ .b = c.batch + 1 }, .i32),
            .sinks = .init(.{c.heads()}, .f32),
            .indices = .init(.{ .b = c.tokens(), .topk = c.topk }, .i32),
        };
    }

    fn options(i: Inputs, c: cases.Case) zml.paged_attention.AttentionOptions {
        return .{
            .mask = switch (c.mask) {
                .causal => .{ .standard = .{} },
                .non_causal => .none,
                .window => .{ .standard = .{ .sliding_window = c.window } },
                .non_causal_window => .{ .standard = .{ .causal = false, .sliding_window = c.window } },
                .indices => .{ .indices = i.indices },
            },
            .sink = if (c.sink) i.sinks else null,
            .scale = c.scale,
        };
    }
};

pub fn forward(i: Inputs, c: cases.Case) zml.Tensor {
    return forwardBackend(i, c, .cute);
}

pub fn forwardBackend(i: Inputs, c: cases.Case, backend: Backend) zml.Tensor {
    const parameters: zml.paged_attention.cute.Parameters = .{
        .block_table = i.table,
        .seq_lens = i.lengths,
        .query_start_len = i.starts,
        .options_ = .{
            .batch_size = c.batch,
            .max_num_pages = c.pages(),
            .max_seqlen_q = if (c.prefill > 0) c.chunk else 1,
            .is_prefill = c.prefill > 0,
        },
    };
    return switch (backend) {
        .cute => zml.paged_attention.cute.pagedAttention(parameters, i.q, i.k, i.v, i.options(c)),
        .cute_generic => zml.paged_attention.cute.pagedAttentionGeneric(parameters, i.q, i.k, i.v, i.options(c)),
    };
}

pub fn upload(a: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, inputs: Inputs, f: Fixture) !zml.Bufferized(Inputs) {
    var result: zml.Bufferized(Inputs) = undefined;
    var initialized: usize = 0;
    errdefer inline for (@typeInfo(Inputs).@"struct".field_names, 0..) |name, index| {
        if (index < initialized) @field(result, name).deinit();
    };
    inline for (@typeInfo(Inputs).@"struct".field_names) |name| {
        const shape = @field(inputs, name).shape();
        const values = @field(f, name);
        if (comptime @TypeOf(values) == []zml.floats.BFloat16) {
            if (shape.dtype() == .f16) {
                const converted = try a.alloc(f16, values.len);
                defer a.free(converted);
                for (converted, values) |*dst, src| dst.* = @floatCast(src.toF32());
                @field(result, name) = try .fromBytes(io, platform, shape, std.mem.sliceAsBytes(converted));
            } else {
                @field(result, name) = try .fromBytes(io, platform, shape, std.mem.sliceAsBytes(values));
            }
        } else {
            @field(result, name) = try .fromBytes(io, platform, shape, std.mem.sliceAsBytes(values));
        }
        initialized += 1;
    }
    return result;
}
