const std = @import("std");

const zml = @import("zml");
const marlin_moe = zml.marlin_moe;
const stdx = zml.stdx;

const log = std.log.scoped(.marlin_playground);

const MIN_THREAD_N: u32 = 64;

// ScalarType ids from fused-moe/marlin_moe_c/include/marlin_moe_c_api.h
const MARLIN_MOE_SCALAR_FE2M1F: i64 = 562949953487106;
const MARLIN_MOE_SCALAR_FE8M0FNU: i64 = 2814749767106568;
const MARLIN_MOE_SCALAR_BFLOAT16: i64 = 1125899906909960;

const LayerName = "marlin_gate_up";

const MoEPlayground = struct {
    output_shape: zml.Shape,
    workspace: zml.Tensor,
    opts: Opts,

    const Opts = struct {
        moe_block_size: u32,
        num_experts: u32,
        top_k: u32,
        mul_topk_weights: u32,
        size_m: u32,
        size_n: u32,
        size_k: u32,
        num_groups: u32,
        group_size: u32,
        thread_k: i32,
        thread_n: i32,
    };

    pub fn forward(
        self: MoEPlayground,
        a: zml.Tensor,
        b_q_weight: zml.Tensor,
        b_scales: zml.Tensor,
        b_bias: zml.Tensor,
        sorted_token_ids: zml.Tensor,
        expert_ids: zml.Tensor,
        num_tokens_past_padded: zml.Tensor,
        topk_weights: zml.Tensor,
    ) zml.Tensor {
        return marlin_moe.marlinMoEForward(
            a,
            b_q_weight,
            b_scales,
            null,
            b_bias,
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
            self.workspace,
            .{
                .moe_block_size = self.opts.moe_block_size,
                .num_experts = self.opts.num_experts,
                .top_k = self.opts.top_k,
                .mul_topk_weights = self.opts.mul_topk_weights,
                .size_m = self.opts.size_m,
                .size_n = self.opts.size_n,
                .size_k = self.opts.size_k,
                .a_type_id = MARLIN_MOE_SCALAR_BFLOAT16,
                .b_type_id = MARLIN_MOE_SCALAR_FE2M1F,
                .c_type_id = MARLIN_MOE_SCALAR_BFLOAT16,
                .s_type_id = MARLIN_MOE_SCALAR_FE8M0FNU,
                .has_act_order = 0,
                .is_k_full = 0,
                .has_zp = 0,
                .num_groups = self.opts.num_groups,
                .group_size = self.opts.group_size,
                .thread_k = self.opts.thread_k,
                .thread_n = self.opts.thread_n,
                .blocks_per_sm = 1,
                .use_atomic_add = 0,
                .use_fp32_reduce = 0,
                .is_zp_float = 0,
                .output_shape = self.output_shape,
            },
        );
    }
};

fn registerAlias(registry: *zml.safetensors.TensorRegistry, alias: []const u8, source: []const u8) !void {
    const tensor_ptr = registry.tensors.getPtr(source) orelse {
        log.err("Missing tensor in safetensors: {s}", .{source});
        return error.MissingTensor;
    };
    var tensor = tensor_ptr.*;
    tensor.name = alias;
    try registry.registerTensor(tensor);
}

pub fn main() !void {
    const allocator = std.heap.c_allocator;

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    const io = threaded.io();

    var safetensors_path: []const u8 = "/home/louislechevalier/mxfp4_moe_marlin_gemm_first_call.safetensors";
    var it = std.process.args();
    defer it.deinit();
    while (it.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "--safetensors=")) {
            safetensors_path = arg["--safetensors=".len..];
        }
    }

    var registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, safetensors_path);
    defer registry.deinit();

    try registerAlias(&registry, LayerName ++ ".in.0", "gate_up_input");
    try registerAlias(&registry, LayerName ++ ".in.1", "w1");
    try registerAlias(&registry, LayerName ++ ".in.2", "w1_scale");
    try registerAlias(&registry, LayerName ++ ".in.3", "bias1");
    try registerAlias(&registry, LayerName ++ ".in.4", "sorted_token_ids");
    try registerAlias(&registry, LayerName ++ ".in.5", "expert_ids");
    try registerAlias(&registry, LayerName ++ ".in.6", "num_tokens_post_padded");
    try registerAlias(&registry, LayerName ++ ".in.7", "topk_weights");
    try registerAlias(&registry, LayerName ++ ".out.0", "intermediate_cache1_output");

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const view = store.view();

    const input_shape = view.getShape("gate_up_input").?;
    const output_shape = view.getShape("intermediate_cache1_output").?;
    const w1_shape = view.getShape("w1").?;
    const w1_scale_shape = view.getShape("w1_scale").?;
    const topk_shape = view.getShape("topk_weights").?;
    const expert_ids_shape = view.getShape("expert_ids").?;

    const size_m: u32 = @intCast(input_shape.dim(0));
    const size_k: u32 = @intCast(input_shape.dim(1));
    const size_n: u32 = @intCast(output_shape.dim(1));
    const num_experts: u32 = @intCast(w1_shape.dim(0));
    const top_k: u32 = @intCast(topk_shape.dim(1));
    const num_groups: u32 = @intCast(w1_scale_shape.dim(1));

    stdx.debug.assert(size_k % num_groups == 0, "size_k must be divisible by num_groups", .{});
    const group_size: u32 = size_k / num_groups;

    const num_moe_blocks: usize = @intCast(expert_ids_shape.dim(0));
    const max_n_tiles: usize = @intCast(size_n / MIN_THREAD_N);
    var workspace_elems: usize = max_n_tiles * num_moe_blocks;
    if (workspace_elems < 1) workspace_elems = 1;

    log.info("Loaded safetensors: {s}", .{safetensors_path});
    log.info("size_m={d} size_n={d} size_k={d} num_experts={d} top_k={d}", .{ size_m, size_n, size_k, num_experts, top_k });
    log.info("num_groups={d} group_size={d} num_moe_blocks={d} workspace_elems={d}", .{ num_groups, group_size, num_moe_blocks, workspace_elems });

    var platform = try zml.Platform.auto(allocator, io, .{});
    defer platform.deinit(allocator);

    if (platform.target != .cuda) {
        log.warn("Platform is not CUDA, skipping execution. This test requires CUDA.", .{});
        return;
    }

    try marlin_moe.load(allocator, io);
    try marlin_moe.register(platform);

    const bias_shape = view.getShape("bias1").?;
    const sorted_token_ids_shape = view.getShape("sorted_token_ids").?;
    const num_tokens_post_padded_shape = view.getShape("num_tokens_post_padded").?;

    log.info("kernel input shapes:", .{});
    log.info("  gate_up_input: {f}", .{input_shape});
    log.info("  w1: {f}", .{w1_shape});
    log.info("  w1_scale: {f}", .{w1_scale_shape});
    log.info("  bias1: {f}", .{bias_shape});
    log.info("  sorted_token_ids: {f}", .{sorted_token_ids_shape});
    log.info("  expert_ids: {f}", .{expert_ids_shape});
    log.info("  num_tokens_post_padded: {f}", .{num_tokens_post_padded_shape});
    log.info("  topk_weights: {f}", .{topk_shape});

    const layer = MoEPlayground{
        .output_shape = output_shape,
        .workspace = zml.Tensor.init(.{workspace_elems}, .i32),
        .opts = .{
            .moe_block_size = 64,
            .num_experts = num_experts,
            .top_k = top_k,
            .mul_topk_weights = 0,
            .size_m = size_m,
            .size_n = size_n,
            .size_k = size_k,
            .num_groups = num_groups,
            .group_size = group_size,
            .thread_k = 64,
            .thread_n = 128,
        },
    };

    var workspace_host = try zml.Slice.alloc(allocator, zml.Shape.init(.{workspace_elems}, .i32));
    defer workspace_host.free(allocator);
    @memset(workspace_host.items(i32), 0);

    var workspace_buf = try zml.Buffer.fromSlice(io, platform, workspace_host);
    defer workspace_buf.deinit();

    const layer_weights: zml.Bufferized(MoEPlayground) = .{ .workspace = workspace_buf };

    try zml.testing.testLayer(
        allocator,
        io,
        platform,
        layer,
        .forward,
        view,
        LayerName,
        layer_weights,
        .{ .absolute_tolerance = 2e-4, .relative_tolerance = 2e-4, .minimum_close_fraction = 0.999 },
    );

    log.info("Layer test passed", .{});
}
