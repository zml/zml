const std = @import("std");

const zml = @import("zml");

const audio_vae = @import("audio_vae.zig");
const conditions = @import("conditions.zig");
const config = @import("config.zig");
const ir = @import("ir.zig");
const media = @import("media.zig");
const noise = @import("noise.zig");
const sharding_mod = @import("sharding.zig");
const packing = @import("packing.zig");
const scheduler = @import("scheduler.zig");
const vae = @import("vae.zig");
const vision = @import("vision.zig");
const visual_vae = @import("visual_vae.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try testConfig();
    try testSharding(allocator);
    try testSplitComma(allocator);
    try testScheduler(allocator);
    try testTimestepEmbedding();
    try testPackingT2va(allocator);
    try testPackingTimestepSlots(allocator);
    try testPackingFl2va(allocator);
    try testPackingRef2va(allocator);
    try testEncodeVideoLatentT();
    try testVisionSpatial();
    try testPatchify(allocator);
    try testNchwToThwc();
    try testCanvas();
    try testVaeGeometry();
    try testMmRopeHost();
    try testOfficialSpatialGrid();
    try testOfficialRotateHalf();
    try testPromptingGuidance(allocator);
    try testOpenH3irAssets(allocator);
    try testIrLlm(allocator);
    try testIrPipeline(allocator);
    try testCanvasPresets();
    try testAudioRefGuard();
    try testVaeTiling(allocator);
    try testVitCoords();
    try testImagenet();
    try testSnake();
    try testOfficialAudioLatents();
    try testOfficialVisualLatents();
    try testTokenDrop();
    try testAudioRowBct();
    try testTorchNoise(allocator);

    std.debug.print("minimax_h3 tests: all passed\n", .{});
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

    var aliased: config.Config = .{
        .token_refiner_num_layers = 2,
        .ffn_hidden_size = 14336,
        .latents_dim = 24,
        .audio_latents_dim = 32,
        .timestep_input_dim = 256,
        .time_embed_hidden_size = 5376,
        .rope_inv_freq_len = 16,
    };
    aliased = aliased.resolve();
    try std.testing.expectEqual(@as(i64, 2), aliased.num_refiner_layers);
    try std.testing.expectEqual(@as(i64, 24), aliased.in_channels);
    try std.testing.expectEqual(@as(i64, 16), aliased.rope_freq_dim);

    try std.testing.expectEqual(config.TaskFamily.fl2va, config.Variant.t2va.taskFamily());
    try std.testing.expectEqual(config.TaskFamily.fl2va, config.Variant.fl2va.taskFamily());
    try std.testing.expectEqual(config.TaskFamily.ref2va, config.Variant.ref2va.taskFamily());
    try std.testing.expectEqualStrings("FL2VA", config.Variant.t2va.dirName());
    try std.testing.expectEqualStrings("FL2VA", config.Variant.fl2va.dirName());
    try std.testing.expectEqualStrings("Ref2VA", config.Variant.ref2va.dirName());
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
            &.{ .link, .link_x, .link_y, .link_z },
            strategy.folding.get(0).sources.constSlice(),
        );
        const data = try zml.Sharding.Data.init("model", &neuron, .mesh(.{ .model = .high_bandwidth }), strategy);
        try std.testing.expectEqual(@as(i64, 16), data.numPartitionsForLogicalAxis(.model));
        try std.testing.expect(!sharding_mod.officialHeadsOk(16));
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
        const strategy = try sharding_mod.tensorParallelStrategy(&cuda16);
        const data = try zml.Sharding.Data.init("model", &cuda16, .mesh(.{ .model = .high_bandwidth }), strategy);
        try std.testing.expectEqual(@as(i64, 16), data.numPartitionsForLogicalAxis(.model));
        const mesh_shard: zml.Sharding = .{ .data = &data };
        const heads = zml.Shape.init(.{ .h = 56 }, .bf16).withPartitioning(.{ .h = .model });
        try std.testing.expectError(error.IncompatibleSharding, mesh_shard.placement(heads));
        const q = zml.Shape.init(.{ .dout = 7168 }, .bf16).withPartitioning(.{ .dout = .model });
        const q_pl = try mesh_shard.placement(q);
        try std.testing.expectEqual(@as(i64, 448), q_pl.shape.dim(.dout));
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

fn testSplitComma(allocator: std.mem.Allocator) !void {
    const empty = try conditions.splitComma(allocator, "");
    try std.testing.expectEqual(@as(usize, 0), empty.len);

    const parts = try conditions.splitComma(allocator, "a.png, b.mp4,,bed.wav");
    defer allocator.free(parts);
    try std.testing.expectEqual(@as(usize, 3), parts.len);
    try std.testing.expectEqualStrings("a.png", parts[0]);
    try std.testing.expectEqualStrings("b.mp4", parts[1]);
    try std.testing.expectEqualStrings("bed.wav", parts[2]);

    const blanks = try conditions.splitComma(allocator, ",, ,");
    defer allocator.free(blanks);
    try std.testing.expectEqual(@as(usize, 0), blanks.len);
}

fn testScheduler(allocator: std.mem.Allocator) !void {
    const sched = try scheduler.Schedule.init(allocator, 12.0, 8);
    defer sched.deinit(allocator);
    try std.testing.expect(sched.sigmas[0] > sched.sigmas[sched.sigmas.len - 1]);
    try std.testing.expectEqual(@as(f32, 0.0), sched.sigmas[sched.sigmas.len - 1]);
    try std.testing.expectEqual(sched.timesteps.len + 1, sched.sigmas.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sched.timesteps[0] + sched.sigmas[0], 1e-6);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), scheduler.shiftSigma(1.0, 12.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0 / 13.0), scheduler.shiftSigma(0.5, 12.0), 1e-6);
    try std.testing.expectEqual(@as(f32, 0.0), scheduler.shiftSigma(0.0, 12.0));

    const audio = scheduler.timeShiftSigma(0.5, 12.0, 3.0);
    try std.testing.expect(audio > 0.0 and audio < 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), scheduler.Schedule.scaleNoise(0.5, 1.0, 0.0), 1e-6);

    var sample = [_]f32{ 1.0, -1.0 };
    const velocity = [_]f32{ 0.0, 0.0 };
    sched.step(0, &sample, &velocity);
    try std.testing.expect(std.math.isFinite(sample[0]));

    const dual = try scheduler.DualSchedule.initOfficial(allocator, 10);
    defer dual.deinit(allocator);
    try std.testing.expectEqual(@as(f32, 12.0), dual.video.shift);
    try std.testing.expectEqual(@as(f32, 3.0), dual.audio.shift);
}

fn testTimestepEmbedding() !void {
    const t = [_]f32{ 0.0, 1.0 };
    var out: [512]f32 = undefined;
    scheduler.timestepEmbedding(&t, 256, true, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[128], 1e-5);
}

fn testPackingT2va(allocator: std.mem.Allocator) !void {
    const layout = try packing.buildT2va(allocator, 4, 2, 4, 4, 3, 0.25, 0.6);
    defer layout.deinit(allocator);

    const video_tokens = config.videoTokenCount(2, 4, 4, .{ 1, 2, 2 });
    const audio_tokens: u32 = 3 * 2;
    try std.testing.expectEqual(4 + audio_tokens + video_tokens, layout.seqLen());
    try std.testing.expectEqual(@as(usize, 4), layout.text_indices.len);
    try std.testing.expectEqual(@as(usize, video_tokens), layout.video_indices.len);
    try std.testing.expectEqual(@as(usize, audio_tokens), layout.audio_indices.len);
    try std.testing.expectEqual(@as(u8, 1), layout.token_tags[0]);
    try std.testing.expectEqual(@as(u8, 2), layout.token_tags[layout.target_audio_start]);
    try std.testing.expectEqual(@as(u8, 0), layout.token_tags[layout.target_video_start]);

    const first_video = layout.adalnIndex(layout.target_video_start);
    try std.testing.expectEqual(first_video % 3, 0);
    const first_text = layout.adalnIndex(0);
    try std.testing.expectEqual(first_text % 3, 1);
    const first_audio = layout.adalnIndex(layout.target_audio_start);
    try std.testing.expectEqual(first_audio % 3, 2);
    try std.testing.expectEqual(packing.timestep_slot_count, @as(u32, @intCast(layout.timesteps.len)));
    try std.testing.expectEqual(@as(u32, 0), layout.timestep_indices[layout.target_video_start]);
    try std.testing.expectEqual(@as(u32, 1), layout.timestep_indices[layout.target_audio_start]);
}

fn testPackingTimestepSlots(allocator: std.mem.Allocator) !void {
    const early = packing.timestepValues(0.99, 0.8);
    const late = packing.timestepValues(0.1, 0.2);
    try std.testing.expectEqual(@as(usize, 4), early.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.99), early[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.999), early[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), late[3], 1e-6);

    const a = try packing.buildT2va(allocator, 3, 2, 4, 4, 2, 0.99, 0.8);
    defer a.deinit(allocator);
    const b = try packing.buildT2va(allocator, 3, 2, 4, 4, 2, 0.1, 0.2);
    defer b.deinit(allocator);
    try std.testing.expectEqualSlices(u32, a.timestep_indices, b.timestep_indices);
    try std.testing.expectEqualSlices(u32, a.video_indices, b.video_indices);
    var buf: [4]f32 = undefined;
    packing.writeTimesteps(&buf, 0.1, 0.2);
    try std.testing.expectEqualSlices(f32, &late, &buf);
}

fn testPackingFl2va(allocator: std.mem.Allocator) !void {
    const first = [_]packing.ConditionVideo{.{
        .latent_t = 1,
        .latent_h = 4,
        .latent_w = 4,
        .keyframe_index = 0,
    }};
    const first_layout = try packing.build(allocator, .{
        .text_len = 2,
        .latent_t = 2,
        .latent_h = 4,
        .latent_w = 4,
        .audio_t = 2,
        .video_t = 0.2,
        .audio_t_noise = 0.4,
        .condition_videos = &first,
    });
    defer first_layout.deinit(allocator);
    try std.testing.expect(first_layout.seqLen() > 2 + 4 + 8);
    try std.testing.expect(first_layout.video_indices.len > config.videoTokenCount(2, 4, 4, .{ 1, 2, 2 }));
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), first_layout.positions[first_layout.video_indices[0]].t, 1e-5);

    const last = [_]packing.ConditionVideo{.{
        .latent_t = 1,
        .latent_h = 4,
        .latent_w = 4,
        .keyframe_index = 1,
    }};
    const last_layout = try packing.build(allocator, .{
        .text_len = 2,
        .latent_t = 2,
        .latent_h = 4,
        .latent_w = 4,
        .audio_t = 2,
        .video_t = 0.2,
        .audio_t_noise = 0.4,
        .condition_videos = &last,
    });
    defer last_layout.deinit(allocator);
    const last_t = 2.0 + packing.videoDuration(2) - config.frame_rescale;
    try std.testing.expectApproxEqAbs(last_t, last_layout.positions[last_layout.video_indices[0]].t, 1e-5);
    try std.testing.expect(last_layout.positions[last_layout.video_indices[0]].t > first_layout.positions[first_layout.video_indices[0]].t);
}

fn testPackingRef2va(allocator: std.mem.Allocator) !void {
    const videos = [_]packing.ConditionVideo{.{
        .latent_t = 1,
        .latent_h = 4,
        .latent_w = 4,
    }};
    const refs = [_]packing.ReferenceBlock{.{
        .kind = .image,
        .video_index = 0,
    }};
    const layout = try packing.build(allocator, .{
        .text_len = 3,
        .latent_t = 2,
        .latent_h = 4,
        .latent_w = 4,
        .audio_t = 2,
        .video_t = 0.3,
        .audio_t_noise = 0.5,
        .condition_videos = &videos,
        .references = &refs,
    });
    defer layout.deinit(allocator);
    try std.testing.expect(layout.video_indices.len > config.videoTokenCount(2, 4, 4, .{ 1, 2, 2 }));
    try std.testing.expect(layout.seqLen() > 3);

    const audios = [_]packing.ConditionAudio{.{ .latent_t = 3 }};
    const av_refs = [_]packing.ReferenceBlock{.{
        .kind = .video_audio,
        .video_index = 0,
        .audio_index = 0,
    }};
    const av = try packing.build(allocator, .{
        .text_len = 3,
        .latent_t = 2,
        .latent_h = 4,
        .latent_w = 4,
        .audio_t = 2,
        .video_t = 0.3,
        .audio_t_noise = 0.5,
        .condition_videos = &videos,
        .condition_audios = &audios,
        .references = &av_refs,
    });
    defer av.deinit(allocator);
    try std.testing.expect(av.audio_indices.len > 4);
    try std.testing.expect(av.video_indices.len > config.videoTokenCount(2, 4, 4, .{ 1, 2, 2 }));
}

fn testEncodeVideoLatentT() !void {
    try std.testing.expectEqual(@as(u32, 2), vae.encodeVideoLatentT(vae.official_visual, 5));
    try std.testing.expectEqual(@as(u32, 2), vae.encodeVideoLatentT(vae.official_visual, 17));
    try std.testing.expectEqual(@as(u32, 7), vae.encodeVideoLatentT(vae.official_visual, 34));
    try std.testing.expectEqual(@as(u32, 37), vae.encodeVideoLatentT(vae.official_visual, 120));
    try std.testing.expectEqual(
        config.videoLatentFrames(config.alignFrameCount(120)),
        vae.encodeVideoLatentT(vae.official_visual, 120),
    );
}

fn testVisionSpatial() !void {
    var cfg = vision.Config{};
    cfg.out_hidden_size = 5120;
    const spec = vision.spatialTokens(cfg, 256, 256, false);
    try std.testing.expectEqual(@as(u32, 0), spec.seq % 4);
    try std.testing.expectEqual(spec.seq / 4, spec.merged);
    var cursor: f32 = 0;
    var pos: [12]f32 = undefined;
    vision.applyVisionPositions(&pos, 0, 4, 4, 4, 1, &cursor);
    try std.testing.expect(cursor > 0);
}

fn testPatchify(allocator: std.mem.Allocator) !void {
    const t: u32 = 2;
    const h: u32 = 4;
    const w: u32 = 4;
    const c: u32 = 2;
    const src = try allocator.alloc(f32, t * h * w * c);
    defer allocator.free(src);
    for (src, 0..) |*v, i| v.* = @floatFromInt(i);

    const rows = try packing.patchify(allocator, src, t, h, w, c, .{ 1, 2, 2 });
    defer allocator.free(rows);
    try std.testing.expectEqual(@as(usize, 2 * 2 * 2 * (2 * 1 * 2 * 2)), rows.len);
    // First 2×2 patch, channel-major: ch0 of four voxels, then ch1.
    try std.testing.expectEqual(@as(f32, 0), rows[0]);
    try std.testing.expectEqual(@as(f32, 2), rows[1]);
    try std.testing.expectEqual(@as(f32, 8), rows[2]);
    try std.testing.expectEqual(@as(f32, 10), rows[3]);

    const back = try packing.unpatchify(allocator, rows, t, h, w, c, .{ 1, 2, 2 });
    defer allocator.free(back);
    try std.testing.expectEqualSlices(f32, src, back);
}

fn testNchwToThwc() !void {
    const src = [_]f32{ 0, 1, 2, 3, 10, 11, 12, 13 };
    var dst: [8]f32 = undefined;
    packing.nchwToThwc(&dst, &src, 2, 2, 1, 2);
    try std.testing.expectEqualSlices(f32, &.{ 0, 10, 1, 11, 2, 12, 3, 13 }, &dst);
}

fn testCanvas() !void {
    const p = config.pixelSize(.@"16:9", 768);
    try std.testing.expectEqual(@as(u32, 768), p.h);
    try std.testing.expectEqual(@as(u32, 0), p.w % 32);
    try std.testing.expectEqual(@as(u32, 0), p.h % 32);
    try std.testing.expect(p.w > p.h);

    const square = config.pixelSize(.@"1:1", 768);
    try std.testing.expectEqual(square.w, square.h);

    const portrait = config.pixelSize(.@"9:16", 768);
    try std.testing.expect(portrait.h > portrait.w);
    try std.testing.expectEqual(@as(u32, 120), config.frameCount(5.0));
    try std.testing.expectEqual(@as(u32, 5), config.alignFrameCount(1));
    try std.testing.expectEqual(@as(u32, 5), config.alignFrameCount(5));
    try std.testing.expectEqual(@as(u32, 22), config.alignFrameCount(17));
    try std.testing.expectEqual(@as(u32, 124), config.alignFrameCount(120));
    try std.testing.expectEqual(@as(u32, 37), config.videoLatentFrames(124));
}

fn testVaeGeometry() !void {
    const lat = vae.official_visual.latentFromPixels(768, 1376, 120);
    try std.testing.expectEqual(@as(u32, 37), lat.t);
    try std.testing.expectEqual(@as(u32, 48), lat.h);
    try std.testing.expectEqual(@as(u32, 86), lat.w);
    try std.testing.expectEqual(@as(u32, 96), vae.official_visual.patchDim());
    try std.testing.expectEqual(@as(u32, 200), vae.official_audio.tokenCount(100));
}

fn testMmRopeHost() !void {
    const cfg = config.Config.official();
    const theta: f32 = cfg.rope_theta;
    const freq: f32 = @floatFromInt(cfg.rope_freq_dim);
    var inv: [16]f32 = undefined;
    for (&inv, 0..) |*f, i| {
        f.* = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(i)) / freq);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), inv[0], 1e-6);
    try std.testing.expect(inv[15] < inv[0]);
    try std.testing.expectEqual(@as(i64, 96), cfg.rotaryDim());
    try std.testing.expect(cfg.rotaryDim() < cfg.attention_head_dim);
}

fn testOfficialSpatialGrid() !void {
    var buf: [8]f32 = undefined;
    const axis = packing.spatialAxis(8, 8, &buf);
    try std.testing.expectEqual(@as(usize, 4), axis.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), axis[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), axis[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), axis[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 24.0), axis[3], 1e-6);
}

fn testOfficialRotateHalf() !void {
    const x = [_]f32{ 1, 2, 3, 4 };
    const rotated = [_]f32{ -3, -4, 1, 2 };
    var out: [4]f32 = undefined;
    const half = x.len / 2;
    for (0..half) |i| {
        out[i] = -x[half + i];
        out[half + i] = x[i];
    }
    try std.testing.expectEqualSlices(f32, &rotated, &out);
}

fn testPromptingGuidance(allocator: std.mem.Allocator) !void {
    const brief = try ir.promptingGuidance(allocator, .{
        .prompt = "a lighthouse keeper lights the lamp",
        .variant = .t2va,
        .duration_s = 5,
    });
    defer allocator.free(brief);
    try std.testing.expect(std.mem.indexOf(u8, brief, "integrated_multimodal_description:") != null);
    try std.testing.expect(std.mem.indexOf(u8, brief, "overall_soundscape:") != null);
    try std.testing.expect(std.mem.indexOf(u8, brief, "non_diegetic_music:") != null);
    try std.testing.expect(ir.alreadyCompiled(brief));

    const passthrough = try ir.promptingGuidance(allocator, .{
        .prompt = brief,
        .variant = .t2va,
    });
    defer allocator.free(passthrough);
    try std.testing.expectEqualStrings(brief, passthrough);

    const fl = try ir.promptingGuidance(allocator, .{
        .prompt = "a lighthouse keeper lights the lamp",
        .variant = .fl2va,
        .duration_s = 5,
    });
    defer allocator.free(fl);
    try std.testing.expect(std.mem.indexOf(u8, fl, "Picture 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, fl, "Picture 2") != null);

    const one = try ir.promptingGuidance(allocator, .{
        .prompt = "a lighthouse keeper lights the lamp",
        .variant = .fl2va,
        .duration_s = 5,
        .image = "first.png",
    });
    defer allocator.free(one);
    try std.testing.expect(std.mem.indexOf(u8, one, "Picture 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, one, "Picture 2") == null);

    const ref = try ir.promptingGuidance(allocator, .{
        .prompt = "a lighthouse keeper lights the lamp",
        .variant = .ref2va,
        .duration_s = 5,
        .refs = "face.png,clip.mp4",
    });
    defer allocator.free(ref);
    try std.testing.expect(std.mem.indexOf(u8, ref, "subject_definitions:") != null);
    try std.testing.expect(std.mem.indexOf(u8, ref, "retention_analysis:") != null);
    try std.testing.expect(std.mem.indexOf(u8, ref, "<Picture 1>") != null);
    try std.testing.expect(std.mem.indexOf(u8, ref, "<Video 1>") != null);
}

fn testOpenH3irAssets(allocator: std.mem.Allocator) !void {
    const fl = try ir.collectAssets(allocator, .{
        .prompt = "x",
        .variant = .fl2va,
        .image = "first.png",
        .last_image = "last.png",
    });
    defer allocator.free(fl);
    try std.testing.expectEqual(@as(usize, 2), fl.len);
    try std.testing.expectEqualStrings("frame_anchor_first", fl[0].role.?);
    try std.testing.expectEqualStrings("frame_anchor_last", fl[1].role.?);

    const refs = try ir.collectAssets(allocator, .{
        .prompt = "x",
        .variant = .ref2va,
        .refs = "face.png,clip.mp4,bed.wav,alone.wav",
    });
    defer allocator.free(refs);
    try std.testing.expectEqual(@as(usize, 4), refs.len);
    try std.testing.expectEqualStrings("image", refs[0].kind);
    try std.testing.expectEqualStrings("video", refs[1].kind);
    try std.testing.expectEqualStrings("audio", refs[2].kind);
    try std.testing.expectEqualStrings("clip.mp4", refs[2].paired_video_path.?);
    try std.testing.expectEqualStrings("audio", refs[3].kind);
    try std.testing.expect(refs[3].paired_video_path == null);
}

fn testIrLlm(allocator: std.mem.Allocator) !void {
    const v1 = try ir.resolveChatUrl(allocator, "http://127.0.0.1:8000/v1/");
    defer allocator.free(v1);
    try std.testing.expectEqualStrings("http://127.0.0.1:8000/v1/chat/completions", v1);
    const full = try ir.resolveChatUrl(allocator, "http://host/v1/chat/completions");
    defer allocator.free(full);
    try std.testing.expectEqualStrings("http://host/v1/chat/completions", full);
    const bare = try ir.resolveChatUrl(allocator, "http://host:8000");
    defer allocator.free(bare);
    try std.testing.expectEqualStrings("http://host:8000/v1/chat/completions", bare);

    const assets = try ir.collectAssets(allocator, .{
        .prompt = "waves",
        .variant = .fl2va,
        .image = "first.png",
        .last_image = "last.png",
    });
    defer allocator.free(assets);
    const user = try ir.userMessage(allocator, .{
        .prompt = "waves at dusk",
        .variant = .fl2va,
        .duration_s = 5.17,
        .aspect = "16:9",
        .creativity = .bold,
        .image = "first.png",
        .last_image = "last.png",
    }, assets);
    defer allocator.free(user);
    try std.testing.expect(std.mem.indexOf(u8, user, "variant: fl2va") != null);
    try std.testing.expect(std.mem.indexOf(u8, user, "Picture 1: image role=frame_anchor_first") != null);
    try std.testing.expect(std.mem.indexOf(u8, user, "Picture 2: image role=frame_anchor_last") != null);

    const text = try ir.parseChatContent(allocator,
        \\{"choices":[{"message":{"content":"```\ndetailed_description:\n[Shot 1] ok\n```"}}]}
    );
    defer allocator.free(text);
    try std.testing.expectEqualStrings("detailed_description:\n[Shot 1] ok", text);
    try std.testing.expect(ir.alreadyCompiled(text));
    try std.testing.expectError(error.H3irEmpty, ir.parseChatContent(allocator, "{\"choices\":[]}"));
    try std.testing.expect(!ir.hasLlm(.{ .prompt = "x" }));
    try std.testing.expect(ir.hasLlm(.{ .prompt = "x", .llm_url = "http://127.0.0.1:8000/v1" }));
    try std.testing.expectEqual(@as(f32, 0.5), ir.Creativity.balanced.temperature());
}

fn hasIrCode(findings: []const ir.Finding, code: []const u8) bool {
    for (findings) |finding| {
        if (std.mem.eql(u8, finding.code, code)) return true;
    }
    return false;
}

fn testIrPipeline(allocator: std.mem.Allocator) !void {
    try std.testing.expectEqual(@as(u32, 1), ir.shotCount(5.17, .balanced));
    try std.testing.expectEqual(@as(u32, 2), ir.shotCount(10, .balanced));
    try std.testing.expectEqual(@as(u32, 3), ir.shotCount(10, .extreme));
    try std.testing.expectEqual(@as(u32, 3), ir.shotCount(13, .balanced));
    try std.testing.expectApproxEqAbs(@as(f32, 5.17), ir.effectiveSeconds(5.0), 0.005);

    var over: [10]ir.Asset = undefined;
    for (&over) |*asset| asset.* = .{ .kind = "image", .path = "x.png" };
    try std.testing.expectError(error.TooManyRefImages, ir.checkCapacity(&over));
    const files = [_]ir.Asset{.{ .kind = "image", .path = "x.png" }} ** 12 ++ [_]ir.Asset{.{ .kind = "video", .path = "y.mp4" }};
    try std.testing.expectError(error.TooManyRefs, ir.checkCapacity(&files));

    const req: ir.Request = .{
        .prompt = "waves at dusk",
        .variant = .fl2va,
        .duration_s = 5,
        .image = "first.png",
        .last_image = "last.png",
    };
    const assets = try ir.collectAssets(allocator, req);
    defer allocator.free(assets);
    const cards = try ir.labelAssets(allocator, assets);
    defer ir.freeCards(allocator, cards);
    const wrap = try ir.promptingGuidance(allocator, req);
    defer allocator.free(wrap);
    const clean = try ir.validate(allocator, wrap, .fl2va, cards, 5);
    defer ir.freeFindings(allocator, clean);
    try std.testing.expectEqual(@as(u32, 0), ir.countErrors(clean));

    const missing = try ir.validate(allocator, "overall_soundscape: wind\n", .t2va, &.{}, 5);
    defer ir.freeFindings(allocator, missing);
    try std.testing.expect(hasIrCode(missing, "S1-missing-section"));

    const fence = try ir.validate(allocator,
        \\```
        \\integrated_multimodal_description: [Shot 1] x
        \\
        \\overall_soundscape: wind
        \\
        \\non_diegetic_music: N/A
        \\```
    , .t2va, &.{}, 5);
    defer ir.freeFindings(allocator, fence);
    try std.testing.expect(hasIrCode(fence, "S4-code-fence"));

    const phantom = try ir.validate(allocator,
        \\integrated_multimodal_description: [Shot 1] follows <Picture 1>
        \\
        \\overall_soundscape: wind
        \\
        \\non_diegetic_music: N/A
        \\
    , .t2va, &.{}, 5);
    defer ir.freeFindings(allocator, phantom);
    try std.testing.expect(hasIrCode(phantom, "L3-phantom-media"));

    const unknown = try ir.validate(allocator,
        \\integrated_multimodal_description: [Shot 1] follows <Image 1>
        \\
        \\overall_soundscape: wind
        \\
        \\non_diegetic_music: N/A
        \\
    , .t2va, &.{}, 5);
    defer ir.freeFindings(allocator, unknown);
    try std.testing.expect(hasIrCode(unknown, "L1-unknown-label"));

    const no_align = try ir.validate(allocator, wrap[std.mem.indexOf(u8, wrap, "integrated_multimodal_description:").?..], .fl2va, cards, 5);
    defer ir.freeFindings(allocator, no_align);
    try std.testing.expect(hasIrCode(no_align, "I1-instruction-line-missing"));

    const unused = try ir.validate(allocator,
        \\subject_definitions:
        \\A supplied face is the reference.
        \\
        \\summary:
        \\[reference generation] a wave
        \\
        \\retention_analysis:
        \\appearance stays consistent
        \\
        \\detailed_description:
        \\[Shot 1] a wave
        \\
        \\overall_soundscape: wind
        \\
        \\non_diegetic_music: N/A
        \\
    , .ref2va, cards[0..1], 5);
    defer ir.freeFindings(allocator, unused);
    try std.testing.expect(hasIrCode(unused, "L4-unused-media"));

    var threaded: std.Io.Threaded = .init_single_threaded;
    const compiled = try ir.compile(allocator, threaded.io(), .{
        .prompt = "waves at dusk",
        .mode = .prompt,
        .variant = .t2va,
        .duration_s = 5,
    });
    defer compiled.deinit(allocator);
    try std.testing.expect(ir.alreadyCompiled(compiled.text));
    try std.testing.expectError(error.TooManyRefImages, ir.compile(allocator, threaded.io(), .{
        .prompt = "x",
        .mode = .prompt,
        .variant = .ref2va,
        .refs = "a.png,b.png,c.png,d.png,e.png,f.png,g.png,h.png,i.png,j.png",
    }));
}

fn testCanvasPresets() !void {
    try std.testing.expectError(error.ConflictingCanvas, config.parseCanvas(true, true, false));
    try std.testing.expectEqual(config.Canvas.auto, try config.parseCanvas(false, false, false));
    try std.testing.expectEqual(config.Canvas.tiny, try config.parseCanvas(true, false, false));
    try std.testing.expectEqual(config.Canvas.preview, try config.parseCanvas(false, true, false));
    try std.testing.expectEqual(config.Canvas.full, try config.parseCanvas(false, false, true));

    const cpu = config.canvasForTarget(.cpu, .auto, 0);
    try std.testing.expectEqual(config.preview_short_side, cpu.short_side);
    const full = config.canvasForTarget(.cpu, .full, 0);
    try std.testing.expectEqual(config.default_short_side, full.short_side);
    const tiny = config.canvasForTarget(.cuda, .tiny, 0);
    try std.testing.expectEqual(config.tiny_short_side, tiny.short_side);
    const cuda = config.canvasForTarget(.cuda, .auto, 0);
    try std.testing.expectEqual(config.default_short_side, cuda.short_side);
    const consumer = config.canvasForTarget(.cuda, .auto, 24 * 1024 * 1024 * 1024);
    try std.testing.expectEqual(config.preview_short_side, consumer.short_side);
    const large = config.canvasForTarget(.cuda, .auto, 80 * 1024 * 1024 * 1024);
    try std.testing.expectEqual(config.default_short_side, large.short_side);
    const metal = config.canvasForTarget(.metal, .auto, 0);
    try std.testing.expectEqual(config.preview_short_side, metal.short_side);
    const rocm_unknown = config.canvasForTarget(.rocm, .auto, 0);
    try std.testing.expectEqual(config.default_short_side, rocm_unknown.short_side);
    const rocm_consumer = config.canvasForTarget(.rocm, .auto, 24 * 1024 * 1024 * 1024);
    try std.testing.expectEqual(config.preview_short_side, rocm_consumer.short_side);
}

fn testAudioRefGuard() !void {
    const consumer = 24 * 1024 * 1024 * 1024;
    const large = 80 * 1024 * 1024 * 1024;
    try std.testing.expect(config.audioRefsNeedTiny(config.preview_short_side, consumer));
    try std.testing.expect(!config.audioRefsNeedTiny(config.tiny_short_side, consumer));
    try std.testing.expect(!config.audioRefsNeedTiny(config.preview_short_side, large));
    try std.testing.expect(!config.audioRefsNeedTiny(config.preview_short_side, 0));
    try std.testing.expect(media.refsContainAudio("a.png,b.wav"));
    try std.testing.expect(!media.refsContainAudio("a.png,b.mp4"));
}

fn testVaeTiling(allocator: std.mem.Allocator) !void {
    const one = try vae.splitTiles(allocator, 128, 256, 64, 16);
    defer one.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), one.count());
    try std.testing.expectEqual(@as(u32, 128), one.lengths[0]);

    const many = try vae.splitTiles(allocator, 640, 256, 64, 16);
    defer many.deinit(allocator);
    try std.testing.expect(many.count() >= 3);
    try std.testing.expectEqual(@as(u32, 0), many.starts[0]);
}

fn testVitCoords() !void {
    var buf: [4]f32 = undefined;
    const axis = vae.vitCoords(4, &buf);
    try std.testing.expectApproxEqAbs(@as(f32, -0.75), axis[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.25), axis[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), axis[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), axis[3], 1e-6);
}

fn testImagenet() !void {
    var px = [_]f32{ 0.0, 0.0, 0.0 };
    vae.denormImagenetRgb(&px);
    try std.testing.expectApproxEqAbs(vae.imagenet_mean[0], px[0], 1e-5);
}

fn testSnake() !void {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), audio_vae.snake(0, 1), 1e-6);
    const y = audio_vae.snake(1.0, 1.0);
    try std.testing.expect(y > 1.0);
}

fn testOfficialAudioLatents() !void {
    const cfg = audio_vae.Config.official();
    try std.testing.expectEqualSlices(f32, &audio_vae.official_latents_mean, &cfg.latents_mean);
    try std.testing.expectEqualSlices(f32, &audio_vae.official_latents_std, &cfg.latents_std);
    try std.testing.expect(cfg.latents_std[0] != 1.0);
}

fn testOfficialVisualLatents() !void {
    const cfg = visual_vae.Config.official();
    try std.testing.expectEqualSlices(f32, &visual_vae.official_latents_mean, &cfg.latents_mean);
    try std.testing.expectEqualSlices(f32, &visual_vae.official_latents_std, &cfg.latents_std);
    try std.testing.expectEqual(@as(i64, 48), cfg.rotaryDim());
    try std.testing.expect(cfg.latents_std[0] != 1.0);
}

fn testTokenDrop() !void {
    const spec = vae.official_visual;
    try std.testing.expectEqual(@as(u32, 5), spec.tokensChunkSize());
    try std.testing.expectEqual(@as(u32, 2), spec.tokenOverlap());
    try std.testing.expectEqual(@as(u32, 3), spec.framePrePadding());
    try std.testing.expectEqual(@as(u32, 5), spec.frameOverlap());
}

fn testTorchNoise(allocator: std.mem.Allocator) !void {
    var gen = noise.Generator.init(1);
    const want_u = [_]f32{ 0.7576315999031067, 0.2793108820915222, 0.40306925773620605, 0.7346844673156738, 0.029281556606292725, 0.7998586297035217, 0.3971373438835144, 0.7543719410896301 };
    for (want_u) |w| try std.testing.expectApproxEqAbs(w, gen.uniform01(), 1e-7);

    var gen_n = noise.Generator.init(1);
    var n16: [16]f32 = undefined;
    noise.randn(&gen_n, &n16);
    const want_n = [_]f32{ -1.5255959033966064, -0.7502318024635315, -0.6539809107780457, -1.6094847917556763, -0.1001671776175499, -0.6091889142990112, -0.9797722697257996, -1.6090962886810303 };
    for (want_n, n16[0..8]) |w, g| try std.testing.expectApproxEqAbs(w, g, 2e-5);

    var gen_v = noise.Generator.init(1);
    const video = try noise.drawVideo(allocator, &gen_v, &.{}, &.{}, 2, 4, 4, .{ 1, 2, 2 });
    defer allocator.free(video);
    try std.testing.expectEqual(@as(usize, 2 * 2 * 2 * 96), video.len);

    const audio = try noise.drawAudio(allocator, &gen_v, &.{}, 32, 3);
    defer allocator.free(audio);
    try std.testing.expectEqual(@as(usize, 2 * 3 * 32), audio.len);

    var gen_c = noise.Generator.init(7);
    const conds = [_]packing.ConditionVideo{.{ .latent_t = 1, .latent_h = 2, .latent_w = 2 }};
    const clean = [_]f32{0} ** 96;
    const mixed = try noise.drawVideo(allocator, &gen_c, &conds, &clean, 2, 4, 4, .{ 1, 2, 2 });
    defer allocator.free(mixed);
    try std.testing.expectEqual(@as(usize, 96 + 2 * 2 * 2 * 96), mixed.len);
}

fn testAudioRowBct() !void {
    const channels: u32 = 2;
    const t: u32 = 3;
    const rows = [_]f32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    var bct: [12]f32 = undefined;
    vae.audioRowsToBct(&bct, &rows, channels, t);
    try std.testing.expectEqualSlices(f32, &.{ 0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11 }, &bct);
    var back: [12]f32 = undefined;
    vae.audioBctToRows(&back, &bct, channels, t);
    try std.testing.expectEqualSlices(f32, &rows, &back);
}
