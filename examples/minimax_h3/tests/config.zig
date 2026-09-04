const std = @import("std");

const zml = @import("zml");

const memory_mod = @import("../recipe/memory.zig");
const policy_mod = @import("../recipe/policy.zig");
const config = @import("../recipe/config.zig");
const packing = @import("../draft/packing.zig");
const pipeline = @import("../draft/pipeline.zig");
const repo = @import("../serve/repo.zig");
const sku = @import("../recipe/sku.zig");
const sharding_mod = @import("../recipe/shard.zig");

// =============================================================================
// tests/config.zig — config, TP degree, memory plan
// =============================================================================

pub fn run(allocator: std.mem.Allocator) !void {
    try testConfig();
    try testDuration();
    try testSharding(allocator);
    try testFrameGeometry();
    try testPrompt();
    try testCheckpoint();
    try testMemoryPlan(allocator);
    try testOfficialPin();
    try testTokenizerRelpaths();
    try testWeightEntrypoints();
    try testAttentionPolicy();
    try testMemoryPlanExact(allocator);
}

fn testConfig() !void {
    const cfg = config.Config.official();
    try std.testing.expectEqual(@as(i64, 5376), cfg.hidden_size);
    try std.testing.expectEqual(@as(i64, 50), cfg.num_layers);
    try std.testing.expectEqual(@as(i64, 56), cfg.num_attention_heads);
    try std.testing.expectEqual(@as(i64, 128), cfg.attention_head_dim);
    try std.testing.expectEqual(@as(i64, 7168), cfg.innerDim());
    try std.testing.expectEqual(@as(i64, 96), cfg.rotaryDim());
    try std.testing.expectEqual(@as(i64, 24 * 1 * 2 * 2), cfg.videoPatchDim());
    try std.testing.expectEqual(@as(i64, 96768), cfg.adalnOutFeatures());
    try std.testing.expectEqual(@as(i64, 10752), cfg.finalAdalnOutFeatures());
    try std.testing.expectEqualStrings("FL2VA", config.task_dir);

    const enc = config.EncoderConfig.official();
    try std.testing.expectEqual(@as(i64, 5120), enc.hidden_size);
    try std.testing.expectEqual(@as(i64, 50), enc.used_hidden_layers);
    try std.testing.expectEqual(@as(i64, 151936), enc.vocab_size);
    const from_file = (config.EncoderFileConfig{}).resolve();
    try std.testing.expectEqual(enc.hidden_size, from_file.hidden_size);
    try std.testing.expectEqual(enc.num_hidden_layers, from_file.num_hidden_layers);
    try std.testing.expectEqual(enc.vocab_size, from_file.vocab_size);
}

fn testDuration() !void {
    try config.checkDuration(5);
    try config.checkDuration(15);
    try std.testing.expectError(error.InvalidDuration, config.checkDuration(4.99));
    try std.testing.expectError(error.InvalidDuration, config.checkDuration(4));
    try std.testing.expectError(error.InvalidDuration, config.checkDuration(3));
    try std.testing.expectError(error.InvalidDuration, config.checkDuration(15.01));
    try std.testing.expectError(error.InvalidDuration, config.checkDuration(16));
    try std.testing.expectError(error.InvalidDuration, config.checkDuration(std.math.nan(f32)));
    try std.testing.expectError(error.InvalidDuration, config.checkDuration(std.math.inf(f32)));
    try std.testing.expectError(error.InvalidDuration, config.checkDuration(-std.math.inf(f32)));
    try std.testing.expectError(error.InvalidDuration, config.resolveFrames(std.math.nan(f32), 0));
    try std.testing.expectError(error.InvalidDuration, config.resolveFrames(std.math.inf(f32), 0));
    try std.testing.expectError(error.InvalidDuration, config.resolveFrames(-std.math.inf(f32), 0));
    try std.testing.expectEqual(@as(u32, 0), config.frameCount(std.math.nan(f32)));
    try std.testing.expectEqual(@as(u32, 0), config.frameCount(std.math.inf(f32)));
    try std.testing.expectEqual(@as(u32, 0), config.frameCount(-std.math.inf(f32)));
}

fn testSharding(allocator: std.mem.Allocator) !void {
    try std.testing.expectEqual(@as(usize, 0), sharding_mod.tensorParallelDegree(0));
    try std.testing.expectEqual(@as(usize, 1), sharding_mod.tensorParallelDegree(1));
    try std.testing.expectEqual(@as(usize, 2), sharding_mod.tensorParallelDegree(2));
    try std.testing.expectEqual(@as(usize, 4), sharding_mod.tensorParallelDegree(4));
    try std.testing.expectEqual(@as(usize, 8), sharding_mod.tensorParallelDegree(8));
    try std.testing.expectEqual(@as(usize, 8), sharding_mod.tensorParallelDegree(16));
    try std.testing.expectEqual(@as(usize, 8), sharding_mod.tensorParallelDegree(32));
    try std.testing.expectEqual(@as(usize, 8), sharding_mod.tensorParallelDegree(64));
    try std.testing.expectEqual(@as(usize, 4), sharding_mod.tensorParallelDegree(5));
    try std.testing.expectEqual(@as(usize, 2), sharding_mod.tensorParallelDegree(3));
    try std.testing.expectEqual(@as(usize, 4), sharding_mod.tensorParallelDegree(6));
    try std.testing.expectEqual(@as(usize, 4), sharding_mod.tensorParallelDegree(7));
    try std.testing.expectEqual(@as(usize, 8), sharding_mod.tensorParallelMax(56, 64, 8));
    {
        const gcd = sharding_mod.tensorParallelMaxAll(sharding_mod.officialHeadCounts());
        try std.testing.expectEqual(@as(usize, 8), gcd);
        var n: usize = 1;
        while (n <= 32) : (n += 1) {
            const d = sharding_mod.tensorParallelDegree(n);
            try std.testing.expect(d >= 1);
            try std.testing.expect(d <= n);
            try std.testing.expect(d <= gcd);
            try std.testing.expect(gcd % d == 0);
            try std.testing.expect(sharding_mod.officialHeadsOk(@intCast(d)));
        }
        try std.testing.expectEqual(@as(usize, 1), sharding_mod.tensorParallelDegree(1));
        try std.testing.expectEqual(@as(usize, 2), sharding_mod.tensorParallelDegree(2));
        try std.testing.expectEqual(@as(usize, 4), sharding_mod.tensorParallelDegree(4));
        try std.testing.expectEqual(@as(usize, 8), sharding_mod.tensorParallelDegree(8));
        try std.testing.expectEqual(@as(usize, 8), sharding_mod.tensorParallelDegree(9));
        try std.testing.expectEqual(@as(usize, 8), sharding_mod.tensorParallelDegree(12));
        try std.testing.expectEqual(@as(usize, 8), sharding_mod.tensorParallelDegree(24));
    }
    try std.testing.expectEqual(@as(usize, 2), sharding_mod.tensorParallelDegreeFor(6, 48, 48, 6));
    try std.testing.expectEqual(@as(usize, 4), sharding_mod.tensorParallelDegreeFor(6, 48, 48, 8));
    try std.testing.expectEqual(@as(usize, 8), sharding_mod.tensorParallelMax(64, 64, 16));
    try std.testing.expectEqual(@as(usize, 8), sharding_mod.tensorParallelDegreeFor(16, 64, 64, 16));
    try std.testing.expectEqual(@as(usize, 8), sharding_mod.tensorParallelDegreeFor(32, 64, 64, 16));
    try std.testing.expectEqual(@as(usize, 8), sharding_mod.tensorParallelDegreeFor(20, 64, 64, 16));
    try std.testing.expectEqual(@as(usize, 1), sharding_mod.tensorParallelMax(0, 64, 8));
    sharding_mod.preparePhysicalMesh(.{ .dit = 64, .enc = 64, .kv = 16 });
    try std.testing.expectEqual(@as(usize, 8), sharding_mod.tensorParallelDegree(16));
    sharding_mod.preparePhysicalMesh(sharding_mod.officialHeadCounts());
    try std.testing.expect(sharding_mod.officialHeadsOk(1));
    try std.testing.expect(sharding_mod.officialHeadsOk(2));
    try std.testing.expect(sharding_mod.officialHeadsOk(4));
    try std.testing.expect(sharding_mod.officialHeadsOk(8));
    try std.testing.expect(!sharding_mod.officialHeadsOk(16));
    try std.testing.expect(!sharding_mod.officialHeadsOk(7));
    try std.testing.expect(!sharding_mod.officialHeadsOk(0));
    try std.testing.expect(!sharding_mod.tensorParallelHeadsOk(8, 56, 64, 7));

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    inline for (std.meta.tags(zml.Target)) |target| {
        const primary = sharding_mod.tensorParallelPrimaryAxis(target);
        var mesh = try testMesh(arena, target, &.{primary}, &.{2});
        const axes = sharding_mod.presentShardableAxes(&mesh);
        try std.testing.expectEqual(primary, axes.get(0));
        const strategy = try sharding_mod.tensorParallelStrategy(&mesh);
        try std.testing.expectEqual(@as(usize, 1), strategy.bindings.len);
        try std.testing.expectEqual(primary, strategy.bindings.get(0).physical.get(0));
        try std.testing.expectEqual(@as(usize, 0), strategy.folding.len);
        const data = try zml.Sharding.Data.init("model", &mesh, .mesh(.{ .model = .high_bandwidth }), strategy);
        try std.testing.expectEqual(@as(i64, 2), data.numPartitionsForLogicalAxis(.model));
        try std.testing.expect(sharding_mod.officialHeadsOk(data.numPartitionsForLogicalAxis(.model)));
    }

    {
        var tpu = try testMesh(arena, .tpu, &.{ .link_x, .link_y }, &.{ 2, 2 });
        const strategy = try sharding_mod.tensorParallelStrategy(&tpu);
        try std.testing.expectEqual(@as(usize, 1), strategy.folding.len);
        try std.testing.expectEqualSlices(
            zml.Sharding.PhysicalAxisTag,
            &.{ .link_x, .link_y },
            strategy.folding.get(0).sources.constSlice(),
        );
        const data = try zml.Sharding.Data.init("model", &tpu, .mesh(.{ .model = .high_bandwidth }), strategy);
        try std.testing.expectEqual(@as(i64, 4), data.numPartitionsForLogicalAxis(.model));
    }

    {
        var tpu = try testMesh(arena, .tpu, &.{ .link_x, .link_y, .link_z }, &.{ 2, 2, 2 });
        const strategy = try sharding_mod.tensorParallelStrategy(&tpu);
        try std.testing.expectEqualSlices(
            zml.Sharding.PhysicalAxisTag,
            &.{ .link_x, .link_y, .link_z },
            strategy.folding.get(0).sources.constSlice(),
        );
        const data = try zml.Sharding.Data.init("model", &tpu, .mesh(.{ .model = .high_bandwidth }), strategy);
        const degree = data.numPartitionsForLogicalAxis(.model);
        try std.testing.expectEqual(@as(i64, 8), degree);
        const mesh_shard: zml.Sharding = .{ .data = &data };
        const q = zml.Shape.init(.{ .dout = 7168, .d = 5376 }, .bf16).withPartitioning(.{ .dout = .model, .d = .replicated });
        const q_pl = try mesh_shard.placement(q);
        try std.testing.expectEqual(@as(i64, 896), q_pl.shape.dim(.dout));
        const heads = zml.Shape.init(.{ .h = 56, .hd = 128 }, .bf16).withPartitioning(.{ .h = .model });
        const h_pl = try mesh_shard.placement(heads);
        try std.testing.expectEqual(@as(i64, 7), h_pl.shape.dim(.h));
        const kv = zml.Shape.init(.{ .h = 8, .hd = 128 }, .bf16).withPartitioning(.{ .h = .model });
        const kv_pl = try mesh_shard.placement(kv);
        try std.testing.expectEqual(@as(i64, 1), kv_pl.shape.dim(.h));
    }

    {
        var neuron = try testMesh(arena, .neuron, &.{ .link, .link_x, .link_y, .link_z }, &.{ 2, 2, 2, 2 });
        const strategy = try sharding_mod.tensorParallelStrategy(&neuron);
        try std.testing.expectEqual(sharding_mod.tensorParallelPrimaryAxis(.neuron), strategy.bindings.get(0).physical.get(0));
        try std.testing.expectEqualSlices(
            zml.Sharding.PhysicalAxisTag,
            &.{ .link, .link_x, .link_y },
            strategy.folding.get(0).sources.constSlice(),
        );
        const data = try zml.Sharding.Data.init("model", &neuron, .mesh(.{ .model = .high_bandwidth }), strategy);
        try std.testing.expectEqual(@as(i64, 8), data.numPartitionsForLogicalAxis(.model));
        try std.testing.expect(sharding_mod.officialHeadsOk(8));
    }

    {
        var oneapi = try testMesh(arena, .oneapi, &.{ .link, .bus }, &.{ 2, 2 });
        const strategy = try sharding_mod.tensorParallelStrategy(&oneapi);
        try std.testing.expectEqualSlices(
            zml.Sharding.PhysicalAxisTag,
            &.{ .link, .bus },
            strategy.folding.get(0).sources.constSlice(),
        );
        const data = try zml.Sharding.Data.init("model", &oneapi, .mesh(.{ .model = .high_bandwidth }), strategy);
        try std.testing.expectEqual(@as(i64, 4), data.numPartitionsForLogicalAxis(.model));
    }

    {
        var cuda16 = try testMesh(arena, .cuda, &.{.link}, &.{16});
        try std.testing.expectError(error.IncompatibleSharding, sharding_mod.tensorParallelStrategy(&cuda16));
        try std.testing.expectError(
            error.IncompatibleSharding,
            sharding_mod.tensorParallelStrategyFor(&cuda16, 56, 64, 8),
        );
    }

    {
        var cuda16 = try testMesh(arena, .cuda, &.{.link}, &.{16});
        const wide_heads = sharding_mod.HeadCounts{
            .dit = 64,
            .enc = 64,
            .kv = 16,
            .ltx = 32,
            .gemma = 16,
            .gemma_kv = 16,
        };
        const strategy = try sharding_mod.tensorParallelStrategyForAll(&cuda16, wide_heads);
        const data = try zml.Sharding.Data.init("wide", &cuda16, .mesh(.{ .model = .high_bandwidth }), strategy);
        try std.testing.expectEqual(@as(i64, 16), data.numPartitionsForLogicalAxis(.model));
        const mesh_shard: zml.Sharding = .{ .data = &data };
        const heads = zml.Shape.init(.{ .h = 64 }, .bf16).withPartitioning(.{ .h = .model });
        const h_pl = try mesh_shard.placement(heads);
        try std.testing.expectEqual(@as(i64, 4), h_pl.shape.dim(.h));
    }
}

fn testMesh(
    allocator: std.mem.Allocator,
    target: zml.Target,
    tags: []const zml.Sharding.PhysicalAxisTag,
    sizes: []const usize,
) !zml.Sharding.PhysicalMesh {
    var next_id: u32 = 0;
    const root = try testMeshNode(allocator, tags, sizes, 0, &next_id);
    return zml.Sharding.PhysicalMesh.fromTree(allocator, target, root);
}

fn testMeshNode(
    allocator: std.mem.Allocator,
    tags: []const zml.Sharding.PhysicalAxisTag,
    sizes: []const usize,
    depth: usize,
    next_id: *u32,
) !zml.Sharding.PhysicalNode {
    if (depth == tags.len) {
        const id = next_id.*;
        next_id.* += 1;
        return .{ .leaf = .{ .id = id, .coords = @splat(0xff) } };
    }
    const children = try allocator.alloc(zml.Sharding.PhysicalNode, sizes[depth]);
    for (children) |*child| {
        child.* = try testMeshNode(allocator, tags, sizes, depth + 1, next_id);
    }
    return .{
        .branch = .{
            .tag = tags[depth],
            .geometry = switch (targetGeometry(tags[depth])) {
                .torus => .{ .mesh = .torus },
                .p2p => .point_to_point,
                .tree => .tree,
            },
            .children = children,
        },
    };
}

fn targetGeometry(tag: zml.Sharding.PhysicalAxisTag) enum { torus, p2p, tree } {
    return switch (tag) {
        .link_x, .link_y, .link_z => .torus,
        .link => .p2p,
        .bus => .tree,
    };
}

fn testFrameGeometry() !void {
    try std.testing.expectEqual(@as(u32, 120), config.frameCount(5.0));
    try std.testing.expectEqual(@as(u32, 5), config.alignFrameCount(1));
    try std.testing.expectEqual(@as(u32, 5), config.alignFrameCount(5));
    try std.testing.expectEqual(@as(u32, 22), config.alignFrameCount(17));
    try std.testing.expectEqual(@as(u32, 124), config.alignFrameCount(120));
    try std.testing.expectEqual(@as(u32, 37), config.videoLatentFrames(124));
    try std.testing.expectEqual(@as(u32, 207), config.audioLatentFromFrames(124));
    const five = try config.resolveFrames(5.0, 0);
    try std.testing.expectEqual(@as(u32, 120), five.raw);
    try std.testing.expectEqual(@as(u32, 124), five.aligned);
    try std.testing.expectApproxEqAbs(@as(f32, 124.0 / 24.0), five.seconds(), 1e-6);
    try std.testing.expectError(error.InvalidDuration, config.resolveFrames(4.99, 0));
    try std.testing.expectError(error.InvalidDuration, config.resolveFrames(15.01, 0));
    const five_frames = try config.resolveFrames(0, 120);
    try std.testing.expectEqual(@as(u32, 120), five_frames.raw);
    try std.testing.expectEqual(@as(u32, 124), five_frames.aligned);
    try std.testing.expectError(error.InvalidDuration, config.resolveFrames(0, 119));
    const exact = try config.resolveFrames(5.0, 124);
    try std.testing.expectEqual(@as(u32, 124), exact.raw);
    try std.testing.expectEqual(@as(u32, 124), exact.aligned);
}

fn testPrompt() !void {
    try sku.validatePrompt("hi");
    try std.testing.expectError(error.IntentEmpty, sku.validatePrompt("   "));
}

fn testCheckpoint() !void {
    const official = [_][]const u8{
        "proj_in.weight",
        "time_embedder.linear_1.weight",
        "transformer_blocks.0.adaln_proj.linear.weight",
    };
    try std.testing.expect(repo.inspect(&official).has_adaln_proj);
    try std.testing.expect(repo.inspect(&official).has_time);
    try std.testing.expect(repo.refuseReason(repo.inspect(&official)) == null);

    const table_only = [_][]const u8{ "adaln_t_table", "proj_in.weight" };
    try std.testing.expect(!repo.inspect(&table_only).has_adaln_proj);
    try std.testing.expect(!repo.inspect(&table_only).has_time);
    try std.testing.expect(repo.refuseReason(repo.inspect(&table_only)) != null);

    const no_time = [_][]const u8{"transformer_blocks.0.adaln_proj.linear.weight"};
    try std.testing.expect(repo.inspect(&no_time).has_adaln_proj);
    try std.testing.expect(!repo.inspect(&no_time).has_time);
    try std.testing.expect(repo.refuseReason(repo.inspect(&no_time)) != null);
    try std.testing.expect(repo.refuseReason(repo.inspect(&.{})) != null);
}

fn testMemoryPlan(allocator: std.mem.Allocator) !void {
    const geo: pipeline.Geometry = .{
        .pixel_w = 896,
        .pixel_h = 512,
        .frames = 124,
        .latent_t = 37,
        .latent_h = 32,
        .latent_w = 56,
        .audio_t = 207,
        .video_tokens = 16,
        .audio_tokens = 16,
        .video_patch_dim = 96,
        .audio_dim = 32,
    };
    var layout = try packing.build(allocator, .{
        .text_len = 4,
        .latent_t = 2,
        .latent_h = 8,
        .latent_w = 14,
        .audio_t = 8,
        .video_t = 0,
        .audio_t_noise = 0,
    });
    defer layout.deinit(allocator);
    const tiny = memory_mod.plan(.{
        .geo = .init(geo),
        .layout = layout,
        .hidden = 256,
        .steps = 4,
        .device_bytes = 24 * 1024 * 1024 * 1024,
        .tp = 2,
    });
    try std.testing.expect(tiny.safe);
}

fn peaksOverBudget(p: memory_mod.Plan, device_bytes: u64) bool {
    if (device_bytes == 0) return false;
    const budget = device_bytes * policy_mod.safety_numer / policy_mod.safety_denom;
    return p.denoise_peak_bytes > budget or
        p.encoder_peak_bytes > budget or
        p.audio_vae_peak_bytes > budget or
        p.refine_peak_bytes > budget;
}

fn expectPlannerDecides(p: memory_mod.Plan, device_bytes: u64) !void {
    try std.testing.expectEqual(!peaksOverBudget(p, device_bytes), p.safe);
    if (p.safe) {
        try std.testing.expectEqualStrings("ok", p.reason);
    } else {
        try std.testing.expect(std.mem.indexOf(u8, p.reason, "exceeds 85% of device memory") != null);
    }
}

fn testOfficialPin() !void {
    try std.testing.expectEqualStrings("MiniMaxAI/MiniMax-H3", config.official_repo);
    try std.testing.expectEqualStrings("42ed227ee7df40d41602854ae760620d6eb651fe", config.official_revision);
    var buf: [256]u8 = undefined;
    const uri = try config.officialTokenizerUri(&buf);
    try std.testing.expectEqualStrings(
        "hf://MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA/tokenizer/tokenizer.json",
        uri,
    );
}

fn testTokenizerRelpaths() !void {
    try std.testing.expectEqual(@as(usize, 2), repo.tokenizer_relpaths.len);
    try std.testing.expectEqualStrings("tokenizer/tokenizer.json", repo.tokenizer_relpaths[0]);
    try std.testing.expectEqualStrings("tokenizer.json", repo.tokenizer_relpaths[1]);
}

fn testWeightEntrypoints() !void {
    try std.testing.expectEqual(@as(usize, 4), repo.weight_entrypoints.len);
    try std.testing.expectEqualStrings("model.safetensors.index.json", repo.weight_entrypoints[0]);
    try std.testing.expectEqualStrings("diffusion_pytorch_model.safetensors.index.json", repo.weight_entrypoints[2]);
    try std.testing.expectEqualStrings("diffusion_pytorch_model.safetensors", repo.weight_entrypoints[3]);
    try std.testing.expectEqualStrings("fused.safetensors", repo.fused_overlay_name);
}

fn testAttentionPolicy() !void {
    const short = policy_mod.selectAttention(.{
        .target = .cuda,
        .dtype = .bf16,
        .head_dim = 128,
        .heads = 56,
        .seq = 64,
        .causal = false,
        .tp = 2,
    });
    try std.testing.expectEqual(zml.attention.Backend.vanilla, short);

    const long = policy_mod.selectAttention(.{
        .target = .cuda,
        .dtype = .bf16,
        .head_dim = 128,
        .heads = 56,
        .seq = 7440,
        .causal = false,
        .tp = 2,
    });
    try std.testing.expectEqual(zml.attention.Backend.cuda_fa2, long);

    try std.testing.expectEqual(zml.attention.Backend.vanilla, policy_mod.selectAttention(.{
        .target = .cpu,
        .dtype = .bf16,
        .head_dim = 128,
        .heads = 56,
        .seq = 7440,
        .causal = false,
        .tp = 1,
    }));
    try std.testing.expect(policy_mod.isFlash(.cuda_fa2));
    try std.testing.expect(policy_mod.isFlash(.cuda_fa3));
    try std.testing.expect(!policy_mod.isFlash(.vanilla));
    try std.testing.expect(try policy_mod.parseAttnOverride("") == null);
    try std.testing.expect(try policy_mod.parseAttnOverride("auto") == null);
    try std.testing.expectEqual(zml.attention.Backend.vanilla, (try policy_mod.parseAttnOverride("sdpa")).?);
    try std.testing.expectEqual(zml.attention.Backend.cuda_fa2, (try policy_mod.parseAttnOverride("fa2")).?);
    try std.testing.expectError(error.InvalidAttn, policy_mod.parseAttnOverride("flash"));
}

fn testMemoryPlanExact(allocator: std.mem.Allocator) !void {
    const geo: pipeline.Geometry = .{
        .pixel_w = 896,
        .pixel_h = 512,
        .frames = 124,
        .latent_t = 37,
        .latent_h = 32,
        .latent_w = 56,
        .audio_t = 207,
        .video_tokens = 7040,
        .audio_tokens = 414,
        .video_patch_dim = 96,
        .audio_dim = 32,
    };
    var layout = try packing.build(allocator, .{
        .text_len = 8,
        .latent_t = 8,
        .latent_h = 22,
        .latent_w = 40,
        .audio_t = 200,
        .video_t = 0,
        .audio_t_noise = 0,
    });
    defer layout.deinit(allocator);
    const seq = layout.seqLen();
    try std.testing.expect(policy_mod.sdpaScoreBytes(7440, 56, 1) > 10 * 1024 * 1024 * 1024);
    try std.testing.expect(policy_mod.fa2ScratchBytes(seq, 56, 128, 2) > 0);
    try std.testing.expect(policy_mod.fa2ScratchBytes(seq, 56, 128, 2) < policy_mod.sdpaScoreBytes(seq, 56, 2));
    try std.testing.expect(policy_mod.adalnTableBytes(9, 5376, 50, 2) > 0);

    const fa2 = memory_mod.plan(.{
        .geo = .init(geo),
        .layout = layout,
        .hidden = 5376,
        .steps = 4,
        .device_bytes = 48 * 1024 * 1024 * 1024,
        .tp = 2,
        .target = .cuda,
        .heads = 56,
        .head_dim = 128,
        .layers = 50,
    });
    try std.testing.expectEqual(zml.attention.Backend.cuda_fa2, fa2.attention);
    try expectPlannerDecides(fa2, 48 * 1024 * 1024 * 1024);
    try std.testing.expect(policy_mod.groupSize(0) == 1);
    try std.testing.expect(policy_mod.groupSize(8) == 8);
    try std.testing.expectEqual(@as(u32, 4), policy_mod.enc_prefetch);
    try std.testing.expectEqual(@as(u32, 4), policy_mod.encPrefetch(50));
    try std.testing.expectEqual(@as(u32, 46), policy_mod.ditKeepBlocks(46, 50));
    try std.testing.expectEqual(@as(u32, 50), policy_mod.ditKeepBlocks(50, 50));

    const over = memory_mod.plan(.{
        .geo = .init(geo),
        .layout = layout,
        .hidden = 5376,
        .steps = 4,
        .device_bytes = 64 * 1024 * 1024 * 1024,
        .tp = 1,
        .target = .cuda,
        .heads = 56,
        .head_dim = 128,
        .layers = 50,
        .encoder_weight_bytes = 64 * 1024 * 1024 * 1024,
    });
    try std.testing.expect(!over.safe);
    try std.testing.expectEqualStrings("estimated text-encoder peak exceeds 85% of device memory", over.reason);
}
