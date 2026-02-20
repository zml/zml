const std = @import("std");

const zml = @import("zml");
const marlin_moe = zml.marlin_moe;
const stdx = zml.stdx;

const log = std.log.scoped(.marlin_test);

const BATCH_SIZE: usize = 16;
const SIZE_K: usize = 256;
const TOP_K: usize = 4;
const SIZE_N: usize = 256;
const NUM_EXPERTS: usize = 32;
const MOE_BLOCK_SIZE: usize = 8;
const NUM_GROUPS: usize = 8;
const GROUP_SIZE: i32 = 32;
const PACK_FACTOR_U4B8: usize = 8;
const MIN_THREAD_N: usize = 64;

const MARLIN_MOE_SCALAR_FE2M1F: i64 = 562949953487106;
const MARLIN_MOE_SCALAR_FE8M0FNU: i64 = 2814749767106568;
const MARLIN_MOE_SCALAR_BFLOAT16: i64 = 1125899906909960;

const Routing = struct {
    sorted_token_ids: []i32,
    expert_ids: []i32,
    topk_weights: []f32,
    num_tokens_past_padded: i32,
    num_moe_blocks: usize,
    sorted_len: usize,

    fn deinit(self: *Routing, allocator: std.mem.Allocator) void {
        allocator.free(self.sorted_token_ids);
        allocator.free(self.expert_ids);
        allocator.free(self.topk_weights);
    }
};

fn ceilDiv(x: usize, y: usize) usize {
    return (x + y - 1) / y;
}

fn buildRouting(allocator: std.mem.Allocator) !Routing {
    const total_assignments: usize = BATCH_SIZE * TOP_K;
    const sentinel: i32 = @intCast(total_assignments);

    var counts = try allocator.alloc(usize, NUM_EXPERTS);
    defer allocator.free(counts);
    @memset(counts, 0);

    for (0..BATCH_SIZE) |token| {
        for (0..TOP_K) |slot| {
            const expert = (token + slot) % NUM_EXPERTS;
            counts[expert] += 1;
        }
    }

    var offsets = try allocator.alloc(usize, NUM_EXPERTS + 1);
    defer allocator.free(offsets);
    offsets[0] = 0;
    for (0..NUM_EXPERTS) |e| {
        offsets[e + 1] = offsets[e] + counts[e];
    }

    var cursor = try allocator.alloc(usize, NUM_EXPERTS);
    defer allocator.free(cursor);
    for (0..NUM_EXPERTS) |e| {
        cursor[e] = offsets[e];
    }

    var assignments = try allocator.alloc(i32, total_assignments);
    defer allocator.free(assignments);

    for (0..BATCH_SIZE) |token| {
        for (0..TOP_K) |slot| {
            const expert = (token + slot) % NUM_EXPERTS;
            const flat_index: i32 = @intCast(token * TOP_K + slot);
            assignments[cursor[expert]] = flat_index;
            cursor[expert] += 1;
        }
    }

    var num_moe_blocks: usize = 0;
    for (0..NUM_EXPERTS) |e| {
        num_moe_blocks += ceilDiv(counts[e], MOE_BLOCK_SIZE);
    }
    if (num_moe_blocks == 0) num_moe_blocks = 1;

    const sorted_len: usize = num_moe_blocks * MOE_BLOCK_SIZE;

    var sorted_token_ids = try allocator.alloc(i32, sorted_len);
    var expert_ids = try allocator.alloc(i32, num_moe_blocks);
    var topk_weights = try allocator.alloc(f32, total_assignments);

    for (0..total_assignments) |i| {
        topk_weights[i] = 1.0;
    }

    var block: usize = 0;
    for (0..NUM_EXPERTS) |e| {
        var pos = offsets[e];
        const end = offsets[e + 1];
        while (pos < end) {
            expert_ids[block] = @intCast(e);
            for (0..MOE_BLOCK_SIZE) |i| {
                const out_idx = block * MOE_BLOCK_SIZE + i;
                sorted_token_ids[out_idx] = if (pos < end)
                    assignments[pos]
                else
                    sentinel;
                if (pos < end) pos += 1;
            }
            block += 1;
        }
    }

    while (block < num_moe_blocks) {
        expert_ids[block] = 0;
        for (0..MOE_BLOCK_SIZE) |i| {
            sorted_token_ids[block * MOE_BLOCK_SIZE + i] = sentinel;
        }
        block += 1;
    }

    return .{
        .sorted_token_ids = sorted_token_ids,
        .expert_ids = expert_ids,
        .topk_weights = topk_weights,
        .num_tokens_past_padded = @intCast(sorted_len),
        .num_moe_blocks = num_moe_blocks,
        .sorted_len = sorted_len,
    };
}

const Demo = struct {
    pub fn forward(
        a: zml.Tensor,
        b_q_weight: zml.Tensor,
        b_scales: zml.Tensor,
        sorted_token_ids: zml.Tensor,
        expert_ids: zml.Tensor,
        num_tokens_past_padded: zml.Tensor,
        topk_weights: zml.Tensor,
        workspace: zml.Tensor,
    ) zml.Tensor {
        return marlin_moe.marlinMoEForward(
            a,
            b_q_weight,
            b_scales,
            null,
            null,
            null,
            null,
            null,
            null,
            sorted_token_ids,
            expert_ids,
            num_tokens_past_padded,
            topk_weights,
            null,
            null,
            workspace,
            .{
                .moe_block_size = @intCast(MOE_BLOCK_SIZE),
                .num_experts = @intCast(NUM_EXPERTS),
                .top_k = @intCast(TOP_K),
                .mul_topk_weights = 1,
                .size_m = @intCast(BATCH_SIZE),
                .size_n = @intCast(SIZE_N),
                .size_k = @intCast(SIZE_K),
                .a_type_id = MARLIN_MOE_SCALAR_BFLOAT16,
                .b_type_id = MARLIN_MOE_SCALAR_FE2M1F,
                .c_type_id = MARLIN_MOE_SCALAR_BFLOAT16,
                .s_type_id = MARLIN_MOE_SCALAR_FE8M0FNU,
                .has_act_order = 0,
                .is_k_full = 0,
                .has_zp = 0,
                .num_groups = @intCast(NUM_GROUPS),
                .group_size = GROUP_SIZE,
                .thread_k = 128,
                .thread_n = 128,
                .blocks_per_sm = 1,
                .use_atomic_add = 0,
                .use_fp32_reduce = 0,
                .is_zp_float = 0,
                .output_shape = zml.Shape.init(.{ BATCH_SIZE * TOP_K, SIZE_N }, .bf16),
            },
        );
    }
};

pub fn main() !void {
    const allocator = std.heap.c_allocator;

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    const io = threaded.io();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    if (platform.target != .cuda) {
        log.warn("Platform is not CUDA, skipping execution. This example requires CUDA.", .{});
        return;
    }

    try marlin_moe.load(allocator, io);
    try marlin_moe.register(platform);

    var routing = try buildRouting(allocator);
    defer routing.deinit(allocator);

    const max_n_tiles = SIZE_N / MIN_THREAD_N;
    var workspace_elems = max_n_tiles * (routing.sorted_len / MOE_BLOCK_SIZE);
    if (workspace_elems < 1) workspace_elems = 1;

    const a_t: zml.Tensor = .init(.{ BATCH_SIZE, SIZE_K }, .bf16);
    const b_q_weight_t: zml.Tensor = .init(.{ NUM_EXPERTS, SIZE_K / 16, (SIZE_N / PACK_FACTOR_U4B8) * 16 }, .i32);
    const b_scales_t: zml.Tensor = .init(.{ NUM_EXPERTS, NUM_GROUPS, SIZE_N }, .f8e8m0);
    const sorted_token_ids_t: zml.Tensor = .init(.{routing.sorted_len}, .i32);
    const expert_ids_t: zml.Tensor = .init(.{routing.num_moe_blocks}, .i32);
    const num_tokens_past_padded_t: zml.Tensor = .init(.{}, .i32);
    const topk_weights_t: zml.Tensor = .init(.{ BATCH_SIZE, TOP_K }, .f32);
    const workspace_t: zml.Tensor = .init(.{workspace_elems}, .i32);

    log.info("Compiling marlin MoE custom call...", .{});
    var exe = try platform.compileFn(
        allocator,
        io,
        Demo.forward,
        .{ a_t, b_q_weight_t, b_scales_t, sorted_token_ids_t, expert_ids_t, num_tokens_past_padded_t, topk_weights_t, workspace_t },
    );
    defer exe.deinit();
    log.info("Compiled", .{});

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    var a_host = try zml.Slice.alloc(allocator, zml.Shape.init(.{ BATCH_SIZE, SIZE_K }, .bf16));
    defer a_host.free(allocator);
    var b_q_weight_host = try zml.Slice.alloc(allocator, zml.Shape.init(.{ NUM_EXPERTS, SIZE_K / 16, (SIZE_N / PACK_FACTOR_U4B8) * 16 }, .i32));
    defer b_q_weight_host.free(allocator);
    var b_scales_host = try zml.Slice.alloc(allocator, zml.Shape.init(.{ NUM_EXPERTS, NUM_GROUPS, SIZE_N }, .f8e8m0));
    defer b_scales_host.free(allocator);
    var sorted_token_ids_host = try zml.Slice.alloc(allocator, zml.Shape.init(.{routing.sorted_len}, .i32));
    defer sorted_token_ids_host.free(allocator);
    var expert_ids_host = try zml.Slice.alloc(allocator, zml.Shape.init(.{routing.num_moe_blocks}, .i32));
    defer expert_ids_host.free(allocator);
    var topk_weights_host = try zml.Slice.alloc(allocator, zml.Shape.init(.{ BATCH_SIZE, TOP_K }, .f32));
    defer topk_weights_host.free(allocator);
    var workspace_host = try zml.Slice.alloc(allocator, zml.Shape.init(.{workspace_elems}, .i32));
    defer workspace_host.free(allocator);

    {
        const a = a_host.items(zml.floats.BFloat16);
        for (0..a.len) |i| {
            a[i] = if ((i % 3) == 0)
                zml.floats.BFloat16.fromF32(1.0)
            else
                zml.floats.BFloat16.fromF32(0.5);
        }
    }
    @memset(b_q_weight_host.items(i32), 1);
    {
        const s = b_scales_host.items(zml.floats.Float8E8M0);
        for (0..s.len) |i| {
            s[i] = zml.floats.Float8E8M0.fromF32(1.0);
        }
    }
    @memcpy(sorted_token_ids_host.items(i32), routing.sorted_token_ids);
    @memcpy(expert_ids_host.items(i32), routing.expert_ids);
    @memcpy(topk_weights_host.items(f32), routing.topk_weights);
    @memset(workspace_host.items(i32), 0);

    const num_tokens_val: [1]i32 = .{routing.num_tokens_past_padded};
    const num_tokens_slice: zml.Slice = .init(zml.Shape.init(.{}, .i32), std.mem.sliceAsBytes(&num_tokens_val));

    var a_buf = try zml.Buffer.fromSlice(io, platform, a_host);
    defer a_buf.deinit();
    var b_q_weight_buf = try zml.Buffer.fromSlice(io, platform, b_q_weight_host);
    defer b_q_weight_buf.deinit();
    var b_scales_buf = try zml.Buffer.fromSlice(io, platform, b_scales_host);
    defer b_scales_buf.deinit();
    var sorted_token_ids_buf = try zml.Buffer.fromSlice(io, platform, sorted_token_ids_host);
    defer sorted_token_ids_buf.deinit();
    var expert_ids_buf = try zml.Buffer.fromSlice(io, platform, expert_ids_host);
    defer expert_ids_buf.deinit();
    var num_tokens_buf = try zml.Buffer.fromSlice(io, platform, num_tokens_slice);
    defer num_tokens_buf.deinit();
    var topk_weights_buf = try zml.Buffer.fromSlice(io, platform, topk_weights_host);
    defer topk_weights_buf.deinit();
    var workspace_buf = try zml.Buffer.fromSlice(io, platform, workspace_host);
    defer workspace_buf.deinit();

    args.set(.{ a_buf, b_q_weight_buf, b_scales_buf, sorted_token_ids_buf, expert_ids_buf, num_tokens_buf, topk_weights_buf, workspace_buf });
    exe.callOpts(io, args, &results, .{ .wait = true });

    var out_buf: zml.Buffer = results.get(zml.Buffer);
    defer out_buf.deinit();

    var out_host = try zml.Slice.alloc(allocator, zml.Shape.init(.{ BATCH_SIZE * TOP_K, SIZE_N }, .bf16));
    defer out_host.free(allocator);

    _ = try out_buf.await(io);
    try out_buf.toSlice(io, out_host);

    var stdout = std.Io.File.stdout().writer(io, &.{});
    const w = &stdout.interface;

    try w.print("Launch OK. Output C shape = [{d}, {d}]\n", .{ BATCH_SIZE * TOP_K, SIZE_N });
    try w.print("First 8 fp16 values (d): ", .{});
    const out_vals_dec = out_host.items(zml.floats.BFloat16);
    for (0..8) |i| {
        try w.print("{d} ", .{out_vals_dec[i].toF32()});
    }
    try w.print("\n", .{});
    const out_vals = out_host.items(zml.floats.BFloat16);
    for (0..8) |i| {
        try w.print("{x:0>4} ", .{@as(u16, @bitCast(out_vals[i]))});
    }
    try w.print("\n", .{});

    stdx.debug.assert(out_vals.len == BATCH_SIZE * TOP_K * SIZE_N, "unexpected output size", .{});
    log.info("marlin_test done", .{});
}
