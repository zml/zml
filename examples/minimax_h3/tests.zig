const std = @import("std");

const zml = @import("zml");

const generate = @import("generate.zig");
const model = @import("model.zig");
const vae_mod = @import("vae.zig");

const audio_vae = vae_mod.audio;
const checkpoint = generate.checkpoint;
const memory_mod = generate.memory;
const policy_mod = model.policy;
const multistep_mod = model.multistep;
const config = model.config;
const dit = model.dit;
const geom = generate.cond_geom;
const ir = generate.ir;
const media = generate.media;
const noise = model.noise;
const packing = model.packing;
const pipeline = generate.pipeline;
const repo = generate.ckpt;
const presentation = generate.presentation;
const request_mod = generate.request;
const scheduler = model.scheduler;
const sharding_mod = generate.sharding;
const vae = vae_mod.geom;
const vision = model.vision;
const visual_vae = vae_mod.visual;

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try testConfig();
    try testCli();
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
    try testOfficialCanvas();
    try testGeomHost(allocator);
    try testPresentation(allocator);
    try testRequest(allocator);
    try testCheckpoint();
    try testPosterior(allocator);
    try testLastOnlyFl2va(allocator);
    try testManifestRoundTrip(allocator);
    try testMultistepSampler();
    try testMemoryPlan(allocator);
    try testResample(allocator);
    try testMediaErrors();
    try testOutputTarget();
    try testExportVideo(allocator);
    try testRowMask();
    try testOfficialPin();
    try testTokenizerRelpaths();
    try testConvrotMarker();
    try testRefSize();
    try testGroupRefs(allocator);
    try testPixelCrc(allocator);
    try testRngReset(allocator);
    try testStandaloneAudio(allocator);
    try testFirstLastFl2va(allocator);
    try testSchemaFixtures();
    try testTableCoord();
    try testHostFwht();
    try testMultistepAb2();
    try testAttentionPolicy();
    try testMemoryPlanExact(allocator);
    try testAdalnIndexLayout(allocator);
    try testSchedulerFormula();

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
    try std.testing.expectEqualStrings(config.taskDirName(.fl2va), config.official_task_dirs[0]);
    try std.testing.expectEqualStrings(config.taskDirName(.ref2va), config.official_task_dirs[1]);

    const enc = config.EncoderConfig.official();
    try std.testing.expectEqual(@as(i64, 5120), enc.hidden_size);
    try std.testing.expectEqual(@as(i64, 50), enc.used_hidden_layers);
    try std.testing.expectEqual(@as(i64, 151936), enc.vocab_size);
    const from_file = (config.EncoderFileConfig{}).resolve();
    try std.testing.expectEqual(enc.hidden_size, from_file.hidden_size);
    try std.testing.expectEqual(enc.num_hidden_layers, from_file.num_hidden_layers);
    try std.testing.expectEqual(enc.vocab_size, from_file.vocab_size);
}

fn testCli() !void {
    try config.checkDuration(5);
    try std.testing.expectError(error.InvalidDuration, config.checkDuration(3));
    try std.testing.expectError(error.InvalidDuration, config.checkDuration(16));
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
    const empty = try request_mod.splitComma(allocator, "");
    try std.testing.expectEqual(@as(usize, 0), empty.len);

    const parts = try request_mod.splitComma(allocator, "a.png, b.mp4,,bed.wav");
    defer allocator.free(parts);
    try std.testing.expectEqual(@as(usize, 3), parts.len);
    try std.testing.expectEqualStrings("a.png", parts[0]);
    try std.testing.expectEqualStrings("b.mp4", parts[1]);
    try std.testing.expectEqualStrings("bed.wav", parts[2]);

    const blanks = try request_mod.splitComma(allocator, ",, ,");
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

    const dual = try scheduler.DualSchedule.init(allocator, 10, config.video_shift, config.audio_shift);
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
    const layout = try packing.build(allocator, .{
        .text_len = 4,
        .latent_t = 2,
        .latent_h = 4,
        .latent_w = 4,
        .audio_t = 3,
        .video_t = 0.25,
        .audio_t_noise = 0.6,
    });
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

    const a = try packing.build(allocator, .{
        .text_len = 3,
        .latent_t = 2,
        .latent_h = 4,
        .latent_w = 4,
        .audio_t = 2,
        .video_t = 0.99,
        .audio_t_noise = 0.8,
    });
    defer a.deinit(allocator);
    const b = try packing.build(allocator, .{
        .text_len = 3,
        .latent_t = 2,
        .latent_h = 4,
        .latent_w = 4,
        .audio_t = 2,
        .video_t = 0.1,
        .audio_t_noise = 0.2,
    });
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
    try std.testing.expectEqual(@as(u32, 1344), p.w);
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
    const official = vae.official_visual.latentFromPixels(768, 1344, 120);
    try std.testing.expectEqual(@as(u32, 84), official.w);
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

fn testIrPipeline(allocator: std.mem.Allocator) !void {
    try std.testing.expectEqual(@as(u32, 2), ir.shotCount(5.17, .t2va));
    try std.testing.expectEqual(@as(u32, 1), ir.shotCount(5.17, .fl2va));
    try std.testing.expectEqual(@as(u32, 3), ir.shotCount(10, .t2va));
    try std.testing.expectEqual(@as(u32, 4), ir.shotCount(13, .t2va));
    try std.testing.expectEqual(@as(u32, 3), ir.shotCount(10, .ref2va));
    try std.testing.expectApproxEqAbs(@as(f32, 5.17), ir.effectiveSeconds(5.0), 0.005);

    var over: [10]ir.Asset = undefined;
    for (&over) |*asset| asset.* = .{ .kind = "image", .path = "x.png" };
    try std.testing.expectError(error.TooManyRefImages, ir.checkCapacity(&over));
    const files = [_]ir.Asset{.{ .kind = "image", .path = "x.png" }} ** 12 ++ [_]ir.Asset{.{ .kind = "video", .path = "y.mp4" }};
    try std.testing.expectError(error.TooManyRefs, ir.checkCapacity(&files));

    const compiled = try ir.compile(allocator, .{
        .prompt = "waves at dusk",
        .variant = .t2va,
        .duration_s = 5,
    });
    defer compiled.deinit(allocator);
    try std.testing.expect(ir.alreadyCompiled(compiled.text));
    const ten = try ir.compile(allocator, .{
        .prompt = "waves at dusk",
        .variant = .t2va,
        .duration_s = 10,
    });
    defer ten.deinit(allocator);
    const fifteen = try ir.compile(allocator, .{
        .prompt = "waves at dusk",
        .variant = .t2va,
        .duration_s = 15,
    });
    defer fifteen.deinit(allocator);
    const last_only = try ir.compile(allocator, .{
        .prompt = "waves at dusk",
        .variant = .fl2va,
        .duration_s = 4,
        .last_image = "last.png",
    });
    defer last_only.deinit(allocator);
    try std.testing.expect(std.mem.indexOf(u8, last_only.text, "4.46-second") != null);
    try std.testing.expectError(error.TooManyRefImages, ir.compile(allocator, .{
        .prompt = "x",
        .variant = .ref2va,
        .refs = "a.png,b.png,c.png,d.png,e.png,f.png,g.png,h.png,i.png,j.png",
    }));
    try std.testing.expectError(error.IntentEmpty, ir.compile(allocator, .{
        .prompt = "   ",
        .variant = .t2va,
    }));
}

fn testCanvasPresets() !void {
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
    const oneapi = config.canvasForTarget(.oneapi, .auto, 0);
    try std.testing.expectEqual(config.preview_short_side, oneapi.short_side);
    const rocm_unknown = config.canvasForTarget(.rocm, .auto, 0);
    try std.testing.expectEqual(config.default_short_side, rocm_unknown.short_side);
    const rocm_consumer = config.canvasForTarget(.rocm, .auto, 24 * 1024 * 1024 * 1024);
    try std.testing.expectEqual(config.preview_short_side, rocm_consumer.short_side);

    const consumer_bytes = 24 * 1024 * 1024 * 1024;
    try std.testing.expectError(error.FullCanvasTooLarge, config.checkCanvas(.full, .t2va, config.default_short_side, consumer_bytes));
    try std.testing.expectError(error.ConditionedPreviewTooLarge, config.checkCanvas(.preview, .fl2va, config.preview_short_side, consumer_bytes));
    try config.checkCanvas(.tiny, .fl2va, config.tiny_short_side, consumer_bytes);
    try config.checkCanvas(.preview, .t2va, config.preview_short_side, consumer_bytes);
}

fn testAudioRefGuard() !void {
    const consumer = 24 * 1024 * 1024 * 1024;
    try std.testing.expect(config.conditionedPreviewNeedsTiny(.fl2va, config.preview_short_side, consumer));
    try std.testing.expect(!config.conditionedPreviewNeedsTiny(.t2va, config.preview_short_side, consumer));
    try std.testing.expect(!config.conditionedPreviewNeedsTiny(.ref2va, config.tiny_short_side, consumer));
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
    const video = try noise.drawVideo(allocator, &gen_v, &.{}, &.{}, 2, 4, 4, .{ 1, 2, 2 }, false);
    defer allocator.free(video);
    try std.testing.expectEqual(@as(usize, 2 * 2 * 2 * 96), video.len);

    const audio = try noise.drawAudio(allocator, &gen_v, &.{}, 32, 3);
    defer allocator.free(audio);
    try std.testing.expectEqual(@as(usize, 2 * 3 * 32), audio.len);

    var gen_c = noise.Generator.init(7);
    const conds = [_]packing.ConditionVideo{.{ .latent_t = 1, .latent_h = 2, .latent_w = 2 }};
    const clean = [_]f32{0} ** 96;
    const mixed = try noise.drawVideo(allocator, &gen_c, &conds, &clean, 2, 4, 4, .{ 1, 2, 2 }, false);
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

fn testOfficialCanvas() !void {
    const p = config.pixelSize(.@"16:9", 768);
    try std.testing.expectEqual(@as(u32, 1344), p.w);
    try std.testing.expectEqual(@as(u32, 768), p.h);
    const portrait = config.pixelSize(.@"9:16", 768);
    try std.testing.expectEqual(@as(u32, 768), portrait.w);
    try std.testing.expectEqual(@as(u32, 1344), portrait.h);
    const preview = config.pixelSize(.@"16:9", 352);
    try std.testing.expectEqual(@as(u32, 640), preview.w);
    try std.testing.expectEqual(@as(u32, 352), preview.h);
    const tiny = config.pixelSize(.@"16:9", 128);
    try std.testing.expectEqual(@as(u32, 224), tiny.w);
    try std.testing.expectEqual(@as(u32, 128), tiny.h);
}

fn testGeomHost(allocator: std.mem.Allocator) !void {
    var buf: [16]u8 = undefined;
    try std.testing.expectEqualStrings("0.2", geom.formatSeconds1(0.25, &buf));
    try std.testing.expectEqualStrings("0.8", geom.formatSeconds1(0.75, &buf));
    try std.testing.expectEqualStrings("1.2", geom.formatSeconds1(1.25, &buf));

    const ref = try geom.refImageSize(2048, 2048, 640, 352);
    try std.testing.expectEqual(@as(u32, 480), ref.w);
    try std.testing.expectEqual(@as(u32, 480), ref.h);
    try std.testing.expectError(error.InvalidAspect, geom.refImageSize(100, 10, 640, 352));

    const box = geom.coverCropBox(100, 50, 32, 32);
    try std.testing.expectEqual(@as(u32, 64), box.w);
    try std.testing.expectEqual(@as(u32, 32), box.h);
    try std.testing.expectEqual(@as(u32, 16), box.x);
    try std.testing.expectEqual(@as(u32, 0), box.y);

    const idx = try geom.resampleFrameIndices(2, 12, 24, allocator);
    defer allocator.free(idx);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 1, 1 }, idx);

    const sampled = try geom.sampleVideoConditionFrames(24, 24, 2, 2);
    try std.testing.expectEqual(@as(u32, 2), sampled.indices_len);
    try std.testing.expectEqual(@as(u32, 1), sampled.block_count);
    var qidx: [4]u32 = undefined;
    try std.testing.expectEqual(@as(u32, 2), geom.fillVideoConditionIndices(24, 24, 2, &qidx));
    try std.testing.expectEqual(@as(u32, 0), qidx[0]);
    try std.testing.expectEqual(@as(u32, 12), qidx[1]);
    var ts: [1]f32 = undefined;
    try std.testing.expectEqual(@as(u32, 1), geom.fillBlockTimestamps(2, 2, 2, &ts));
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), ts[0], 1e-6);
}

const StubEnc = struct {
    pub fn encodeAlloc(_: @This(), allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        const out = try allocator.alloc(u32, text.len);
        for (text, out) |c, *d| d.* = c;
        return out;
    }
};

fn testPresentation(allocator: std.mem.Allocator) !void {
    const enc = StubEnc{};
    var t2 = try presentation.assembleT2va(allocator, enc, "hello");
    defer t2.deinit(allocator);
    try std.testing.expectEqualSlices(u32, &.{ 'h', 'e', 'l', 'l', 'o' }, t2.tokens);
    try std.testing.expectEqual(@as(usize, 0), t2.spans.len);

    const fl_specs = [_]presentation.VisualSpec{.{
        .kind = .image,
        .merged = 4,
        .grid_h = 2,
        .grid_w = 2,
    }};
    var fl = try presentation.assembleFl2va(allocator, enc, &fl_specs, "ZZ");
    defer fl.deinit(allocator);
    try std.testing.expectEqual(@as(u32, '<'), fl.tokens[0]);
    try std.testing.expect(std.mem.indexOfScalar(u32, fl.tokens, vision.VISION_START) != null);
    try std.testing.expect(std.mem.indexOfScalar(u32, fl.tokens, vision.IMAGE_PAD) != null);
    try std.testing.expectEqual(@as(u32, 'Z'), fl.tokens[fl.tokens.len - 2]);
    try std.testing.expectEqual(@as(u32, 'Z'), fl.tokens[fl.tokens.len - 1]);
    try std.testing.expectEqual(@as(usize, 1), fl.spans.len);

    const ts = [_]f32{0.25};
    const ref_specs = [_]presentation.VisualSpec{.{
        .kind = .video_audio,
        .merged = 2,
        .grid_h = 1,
        .grid_w = 2,
        .temporal = 1,
        .timestamps = &ts,
        .has_audio = true,
    }};
    var ref = try presentation.assembleRef2va(allocator, enc, &ref_specs, "p");
    defer ref.deinit(allocator);
    try std.testing.expect(containsAscii(ref.tokens, "<Audio 1>: "));
    try std.testing.expect(containsAscii(ref.tokens, "<Video 1>: "));
    try std.testing.expect(containsAscii(ref.tokens, "<0.2 seconds>"));
    const audio_at = indexOfAscii(ref.tokens, "<Audio 1>: ").?;
    const video_at = indexOfAscii(ref.tokens, "<Video 1>: ").?;
    try std.testing.expect(audio_at < video_at);
    try std.testing.expect(std.mem.indexOfScalar(u32, ref.tokens, vision.VIDEO_PAD) != null);
    try std.testing.expectEqual(@as(u32, 'p'), ref.tokens[ref.tokens.len - 1]);
}

fn containsAscii(tokens: []const u32, text: []const u8) bool {
    return indexOfAscii(tokens, text) != null;
}

fn indexOfAscii(tokens: []const u32, text: []const u8) ?usize {
    if (text.len == 0 or text.len > tokens.len) return null;
    var i: usize = 0;
    while (i + text.len <= tokens.len) : (i += 1) {
        var ok = true;
        for (text, 0..) |c, j| {
            if (tokens[i + j] != c) {
                ok = false;
                break;
            }
        }
        if (ok) return i;
    }
    return null;
}

fn testRequest(allocator: std.mem.Allocator) !void {
    const refs = try request_mod.refsFromComma(allocator, "a.png, clip.mp4, bed.wav");
    defer request_mod.freeRefs(allocator, refs, false);
    try std.testing.expectEqual(@as(usize, 2), refs.len);
    try std.testing.expectEqual(packing.ReferenceKind.image, refs[0].kind);
    try std.testing.expectEqual(packing.ReferenceKind.video_audio, refs[1].kind);
    try std.testing.expect(request_mod.hasAudio(refs));

    const manifest =
        \\[{"kind":"image","path":"x.png"},{"kind":"video","path":"y.mp4","soundtrack":"z.wav"}]
    ;
    const parsed = try request_mod.refsFromManifest(allocator, manifest);
    defer request_mod.freeRefs(allocator, parsed, true);
    try std.testing.expectEqual(@as(usize, 2), parsed.len);
    try std.testing.expectEqual(packing.ReferenceKind.video_audio, parsed[1].kind);

    try request_mod.validate(.{ .prompt = "hi", .variant = .t2va });
    try std.testing.expectError(error.T2vaRejectsMedia, request_mod.validate(.{
        .prompt = "hi",
        .variant = .t2va,
        .refs = refs,
    }));
    try std.testing.expectError(error.Fl2vaNeedsImage, request_mod.validate(.{
        .prompt = "hi",
        .variant = .fl2va,
    }));
    const audio_only = try request_mod.refsFromComma(allocator, "a.wav");
    defer request_mod.freeRefs(allocator, audio_only, false);
    try request_mod.validateRefs(audio_only);
    try std.testing.expectEqual(config.Variant.t2va, try request_mod.inferVariant("", "", &.{}));
    try std.testing.expectEqual(config.Variant.fl2va, try request_mod.inferVariant("a.png", "", &.{}));
    try std.testing.expectEqual(config.Variant.ref2va, try request_mod.inferVariant("", "", refs));
    try std.testing.expectError(error.Ref2vaRejectsKeyframes, request_mod.inferVariant("a.png", "", refs));
    const csv = try request_mod.refsToCsv(allocator, parsed);
    defer allocator.free(csv);
    try std.testing.expectEqualStrings("x.png,y.mp4,z.wav", csv);
}

fn testCheckpoint() !void {
    const full = [_][]const u8{
        "video_patch_proj.weight",
        "time_embedder.proj_in.weight",
        "blocks.0.adaln_proj.linear.weight",
        "final_layer.adaln_proj.linear.weight",
    };
    try std.testing.expect(checkpoint.inspect(&full).has_adaln_proj);
    try std.testing.expect(checkpoint.inspect(&full).has_time);
    try std.testing.expect(checkpoint.refuseReason(checkpoint.inspect(&full)) == null);

    const table_only = [_][]const u8{ "adaln_t_table", "video_patch_proj.weight" };
    try std.testing.expect(!checkpoint.inspect(&table_only).has_adaln_proj);
    try std.testing.expect(checkpoint.inspect(&table_only).has_time);
    try std.testing.expect(checkpoint.refuseReason(checkpoint.inspect(&table_only)) != null);

    const no_time = [_][]const u8{"blocks.0.adaln_proj.linear.weight"};
    try std.testing.expect(checkpoint.inspect(&no_time).has_adaln_proj);
    try std.testing.expect(!checkpoint.inspect(&no_time).has_time);
    try std.testing.expect(checkpoint.refuseReason(checkpoint.inspect(&no_time)) != null);

    const rank8 = [_][]const u8{ "adaln_t_table", "blocks.0.adaln_proj.linear.weight" };
    try std.testing.expect(checkpoint.inspect(&rank8).has_adaln_proj);
    try std.testing.expect(checkpoint.inspect(&rank8).has_time);
    try std.testing.expect(checkpoint.refuseReason(checkpoint.inspect(&rank8)) == null);

    try std.testing.expect(checkpoint.safetensorsContains("minimax_h3_fl2va_pruned_int8.safetensors", &.{"fl2va"}));
    try std.testing.expect(checkpoint.safetensorsContains("minimax_h3_ref2va_fp8_scaled.safetensors", &.{"ref2va"}));
    try std.testing.expect(!checkpoint.safetensorsContains("minimax_h3_fl2va_pruned_int8.safetensors", &.{"ref2va"}));
    try std.testing.expect(!checkpoint.safetensorsContains("notes.txt", &.{"fl2va"}));
    try std.testing.expect(checkpoint.safetensorsContains("qwen3vl_32b_minimax_h3_int8_convrot.safetensors", &.{}));
    try std.testing.expect(checkpoint.safetensorsContains("minimax_h3_video_vae_fp16.safetensors", &.{ "video", "vae" }));
    try std.testing.expect(checkpoint.safetensorsContains("minimax_h3_audio_vae_fp32.safetensors", &.{ "audio", "vae" }));
    try std.testing.expect(!checkpoint.safetensorsContains("minimax_h3_audio_vae_fp32.safetensors", &.{ "video", "vae" }));
    try std.testing.expect(checkpoint.isBundleLeaf("text_encoders"));
    try std.testing.expect(!checkpoint.isBundleLeaf("transformer"));
}

fn testPosterior(allocator: std.mem.Allocator) !void {
    var moments: [48]f32 = undefined;
    @memset(moments[0..24], 1.0);
    @memset(moments[24..], 0.0);
    const mean = try vae.sampleVisualPosteriorNchw(allocator, &moments, 1, 1, 1, .mean);
    defer allocator.free(mean);
    try std.testing.expectEqualSlices(f32, moments[0..24], mean);

    const a = try vae.sampleVisualPosteriorNchw(allocator, &moments, 1, 1, 1, .sample_seed42);
    defer allocator.free(a);
    const b = try vae.sampleVisualPosteriorNchw(allocator, &moments, 1, 1, 1, .sample_seed42);
    defer allocator.free(b);
    try std.testing.expectEqualSlices(f32, a, b);
    try std.testing.expect(!std.mem.eql(f32, a, mean));
}

const OfficialEnc = struct {
    pub fn encodeAlloc(_: @This(), allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        if (std.mem.eql(u8, text, "<Picture 1>: ")) return allocator.dupe(u32, &.{ 21604, 3826, 220, 16, 26818, 220 });
        if (std.mem.eql(u8, text, "<Audio 1>: ")) return allocator.dupe(u32, &.{ 65406, 220, 16, 26818, 220 });
        if (std.mem.eql(u8, text, "<Video 1>: ")) return allocator.dupe(u32, &.{ 27, 10724, 220, 16, 26818, 220 });
        if (std.mem.eql(u8, text, "<0.2 seconds>")) return allocator.dupe(u32, &.{ 27, 15, 13, 17, 6486, 29 });
        if (std.mem.eql(u8, text, "hello")) return allocator.dupe(u32, &.{14990});
        const out = try allocator.alloc(u32, text.len);
        for (text, out) |c, *d| d.* = c;
        return out;
    }
};

fn testLastOnlyFl2va(allocator: std.mem.Allocator) !void {
    const specs = [_]presentation.VisualSpec{.{
        .kind = .image,
        .merged = 2,
        .grid_h = 1,
        .grid_w = 2,
    }};
    var assembled = try presentation.assembleFl2va(allocator, OfficialEnc{}, &specs, "hello");
    defer assembled.deinit(allocator);
    try std.testing.expectEqualSlices(u32, &.{ 21604, 3826, 220, 16, 26818, 220 }, assembled.tokens[0..6]);
    try std.testing.expectEqual(@as(u32, 14990), assembled.tokens[assembled.tokens.len - 1]);
    try std.testing.expectEqual(@as(usize, 1), assembled.spans.len);
}

fn testManifestRoundTrip(allocator: std.mem.Allocator) !void {
    const src = try request_mod.refsFromComma(allocator, "a.png, clip.mp4, bed.wav");
    defer request_mod.freeRefs(allocator, src, false);
    const json = try request_mod.refsToManifest(allocator, src);
    defer allocator.free(json);
    const back = try request_mod.refsFromManifest(allocator, json);
    defer request_mod.freeRefs(allocator, back, true);
    try std.testing.expectEqual(src.len, back.len);
    try std.testing.expectEqual(src[0].kind, back[0].kind);
    try std.testing.expectEqual(src[1].kind, back[1].kind);
}

fn testMultistepSampler() !void {
    var x = [_]f32{1.0};
    const v = [_]f32{1.0};
    const sig = [_]f32{ 1.0, 0.5, 0.0 };
    multistep_mod.resMultistep(&sig, 0, &x, &v, null);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), x[0], 1e-6);
    const prev = [_]f32{1.0};
    x[0] = 1.0;
    multistep_mod.resMultistep(&sig, 1, &x, &v, &prev);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), x[0], 1e-6);
}

fn testMemoryPlan(allocator: std.mem.Allocator) !void {
    const geo: pipeline.Geometry = .{
        .pixel_w = 224,
        .pixel_h = 128,
        .frames = 5,
        .latent_t = 2,
        .latent_h = 8,
        .latent_w = 14,
        .audio_t = 8,
        .video_tokens = 16,
        .audio_tokens = 16,
        .target_video_tokens = 16,
        .target_audio_tokens = 16,
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
    const tiny = memory_mod.plan(geo, layout, 256, 4, 24 * 1024 * 1024 * 1024, 2);
    try std.testing.expect(tiny.safe);
    const huge_geo = blk: {
        var g = geo;
        g.pixel_w = 1344;
        g.pixel_h = 768;
        break :blk g;
    };
    const full = memory_mod.plan(huge_geo, layout, 5376, 30, 24 * 1024 * 1024 * 1024, 2);
    try std.testing.expect(!full.safe);
}

fn testResample(allocator: std.mem.Allocator) !void {
    const stereo = [_]f32{ 0, 0, 1, 1 };
    const out = try geom.resampleLinear(allocator, &stereo, 2, 4);
    defer allocator.free(out);
    try std.testing.expectEqual(@as(usize, 8), out.len);
    try std.testing.expectEqual(@as(f32, 0), out[0]);
    try std.testing.expectEqual(@as(f32, 1), out[out.len - 1]);
}

fn testMediaErrors() !void {
    try std.testing.expectError(error.BadWav, media.parseWavHeader("not a wav"));
}

fn testOutputTarget() !void {
    const def = media.Output.parse("");
    try std.testing.expectEqualStrings("output", def.dir);
    try std.testing.expectEqualStrings("output.mp4", def.mp4_name);
    try std.testing.expect(!def.isCwd());

    const dir = media.Output.parse("out_t2va");
    try std.testing.expectEqualStrings("out_t2va", dir.dir);
    try std.testing.expectEqualStrings("output.mp4", dir.mp4_name);

    const file = media.Output.parse("clips/waves.mp4");
    try std.testing.expectEqualStrings("clips", file.dir);
    try std.testing.expectEqualStrings("waves.mp4", file.mp4_name);

    const cwd = media.Output.parse("output.mp4");
    try std.testing.expectEqualStrings(".", cwd.dir);
    try std.testing.expectEqualStrings("output.mp4", cwd.mp4_name);
    try std.testing.expect(cwd.isCwd());
}

fn testExportVideo(allocator: std.mem.Allocator) !void {
    var threaded: std.Io.Threaded = .init_single_threaded;
    const io = threaded.io();
    var scratch = try media.Scratch.init(allocator);
    defer scratch.deinit(allocator);
    const dest_path = scratch.path;
    var dest = try media.openPath(io, dest_path);
    defer dest.close(io);

    const nchw = [_]f32{1} ** (16 * 16 * 3);
    const pcm = [_]i16{0} ** 16000;
    const muxed = try media.writeGeneratedVideo(
        allocator,
        io,
        dest,
        dest_path,
        "clip.mp4",
        &nchw,
        1,
        16,
        16,
        &pcm,
        32000,
    );
    if (dest.openFile(io, "frame_0000.ppm", .{ .mode = .read_only })) |f| {
        f.close(io);
        return error.TestUnexpectedResult;
    } else |_| {}
    if (muxed) {
        var mp4 = dest.openFile(io, "clip.mp4", .{ .mode = .read_only }) catch return error.TestUnexpectedResult;
        mp4.close(io);
    } else {
        var frame = dest.openFile(io, "frames/frame_0000.ppm", .{ .mode = .read_only }) catch return error.TestUnexpectedResult;
        frame.close(io);
        var wav = dest.openFile(io, "audio.wav", .{ .mode = .read_only }) catch return error.TestUnexpectedResult;
        wav.close(io);
    }
}

fn testRowMask() !void {
    var idx = [_]u32{ 0, 1, 2 };
    const mask = [_]u8{ 1, 0, 1 };
    packing.applyRowMask(&idx, &mask, 3);
    try std.testing.expectEqualSlices(u32, &.{ 3, 1, 3 }, &idx);
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
    try std.testing.expectEqual(@as(usize, 4), repo.tokenizer_relpaths.len);
    try std.testing.expectEqualStrings("tokenizer/tokenizer.json", repo.tokenizer_relpaths[0]);
    try std.testing.expectEqualStrings("processor/tokenizer.json", repo.tokenizer_relpaths[1]);
    try std.testing.expectEqualStrings("text_encoder/tokenizer.json", repo.tokenizer_relpaths[2]);
    try std.testing.expectEqualStrings("tokenizer.json", repo.tokenizer_relpaths[3]);
}

fn testConvrotMarker() !void {
    try std.testing.expectEqual(@as(?u32, 256), try zml.safetensors.convrotGroupFromMarker(
        "{\"format\":\"int8_tensorwise\",\"convrot\":true,\"convrot_groupsize\":256}",
    ));
    try std.testing.expectEqual(@as(?u32, 0), try zml.safetensors.convrotGroupFromMarker("{\"format\":\"int8_tensorwise\"}"));
    try std.testing.expectEqual(@as(?u32, 0), try zml.safetensors.convrotGroupFromMarker(
        "{\"format\":\"nvfp4\",\"full_precision_matrix_mult\":true}",
    ));
    try std.testing.expectError(error.UnsupportedConvrotGroup, zml.safetensors.convrotGroupFromMarker(
        "{\"convrot\":true,\"convrot_groupsize\":64}",
    ));
}

fn testRefSize() !void {
    const match = try geom.refImageSize(2048, 2048, 640, 352);
    try std.testing.expectEqual(@as(u32, 480), match.w);
    try std.testing.expectEqual(@as(u32, 480), match.h);
    const no_up = try geom.refImageSize(256, 256, 640, 352);
    try std.testing.expectEqual(@as(u32, 256), no_up.w);
    try std.testing.expectEqual(@as(u32, 256), no_up.h);
    const small_vid = try geom.videoCanvas(320, 180);
    try std.testing.expect(small_vid.w <= 320 + 32);
    var ts: [3]f32 = undefined;
    try std.testing.expectEqual(@as(u32, 3), geom.fillVideoTimestamps(3, &ts));
    try std.testing.expectEqual(@as(f32, 0), ts[0]);
    try std.testing.expectEqual(@as(f32, 1.0), ts[2]);
}

fn testGroupRefs(allocator: std.mem.Allocator) !void {
    const src = try request_mod.refsFromComma(allocator, "a.wav, b.png, c.mp4");
    defer request_mod.freeRefs(allocator, src, false);
    const grouped = try request_mod.groupRefs(allocator, src);
    defer allocator.free(grouped);
    try std.testing.expectEqual(packing.ReferenceKind.image, grouped[0].kind);
    try std.testing.expectEqual(packing.ReferenceKind.video, grouped[1].kind);
    try std.testing.expectEqual(packing.ReferenceKind.audio, grouped[2].kind);
}

fn testPixelCrc(allocator: std.mem.Allocator) !void {
    const src = [_]u8{ 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255 };
    const a = try geom.stretchLanczos(allocator, &src, 2, 2, 4, 4);
    defer allocator.free(a);
    const b = try geom.stretchLanczos(allocator, &src, 2, 2, 4, 4);
    defer allocator.free(b);
    try std.testing.expectEqual(std.hash.Crc32.hash(a), std.hash.Crc32.hash(b));
    try std.testing.expectEqual(@as(usize, 48), a.len);
    const crop = try geom.coverCropLanczos(allocator, &src, 2, 2, 2, 2);
    defer allocator.free(crop);
    try std.testing.expectEqual(@as(usize, 12), crop.len);
}

fn testRngReset(allocator: std.mem.Allocator) !void {
    const conds = [_]packing.ConditionVideo{.{ .latent_t = 1, .latent_h = 2, .latent_w = 2 }};
    const clean = [_]f32{0} ** 96;
    var sequential = noise.Generator.init(3);
    const seq = try noise.drawVideo(allocator, &sequential, &conds, &clean, 2, 4, 4, .{ 1, 2, 2 }, false);
    defer allocator.free(seq);
    var reset = noise.Generator.init(3);
    const rst = try noise.drawVideo(allocator, &reset, &conds, &clean, 2, 4, 4, .{ 1, 2, 2 }, true);
    defer allocator.free(rst);
    try std.testing.expectEqual(seq.len, rst.len);
    try std.testing.expect(!std.mem.eql(f32, seq[96..], rst[96..]));
}

fn testStandaloneAudio(allocator: std.mem.Allocator) !void {
    const specs = [_]presentation.VisualSpec{.{
        .kind = .audio,
        .merged = 0,
        .grid_h = 1,
        .grid_w = 1,
        .has_audio = true,
    }};
    var assembled = try presentation.assembleRef2va(allocator, OfficialEnc{}, &specs, "hello");
    defer assembled.deinit(allocator);
    try std.testing.expectEqualSlices(u32, &.{ 65406, 220, 16, 26818, 220 }, assembled.tokens[0..5]);
    try std.testing.expectEqual(@as(u32, 14990), assembled.tokens[assembled.tokens.len - 1]);
    try std.testing.expectEqual(@as(usize, 0), assembled.spans.len);
}

fn testFirstLastFl2va(allocator: std.mem.Allocator) !void {
    const specs = [_]presentation.VisualSpec{
        .{ .kind = .image, .merged = 2, .grid_h = 1, .grid_w = 2 },
        .{ .kind = .image, .merged = 2, .grid_h = 1, .grid_w = 2 },
    };
    var assembled = try presentation.assembleFl2va(allocator, OfficialEnc{}, &specs, "hello");
    defer assembled.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 2), assembled.spans.len);
    try std.testing.expectEqualSlices(u32, &.{ 21604, 3826, 220, 16, 26818, 220 }, assembled.tokens[0..6]);
}

fn testSchemaFixtures() !void {
    try std.testing.expect(checkpoint.refuseReason(checkpoint.inspect(&.{})) != null);

    const int8 = [_][]const u8{ "blocks.0.adaln_proj.linear.weight", "weight_scale", "adaln_t_table" };
    try std.testing.expect(checkpoint.inspect(&int8).has_adaln_proj);
    try std.testing.expect(checkpoint.inspect(&int8).has_time);
    try std.testing.expect(checkpoint.refuseReason(checkpoint.inspect(&int8)) == null);
}

fn testTableCoord() !void {
    const a = dit.tableCoord(0, 1025);
    try std.testing.expectEqual(@as(u32, 0), a.i0);
    try std.testing.expectEqual(@as(u32, 1), a.i1);
    try std.testing.expectApproxEqAbs(@as(f32, 0), a.frac, 1e-6);

    const b = dit.tableCoord(1, 1025);
    try std.testing.expectEqual(@as(u32, 1024), b.i0);
    try std.testing.expectEqual(@as(u32, 1024), b.i1);
    try std.testing.expectApproxEqAbs(@as(f32, 0), b.frac, 1e-6);

    const c = dit.tableCoord(0.5, 5);
    try std.testing.expectEqual(@as(u32, 2), c.i0);
    try std.testing.expectEqual(@as(u32, 3), c.i1);
    try std.testing.expectApproxEqAbs(@as(f32, 0), c.frac, 1e-6);
}

fn fwht4(v: [4]f32) [4]f32 {
    const s1 = [4]f32{ v[0] + v[1], v[0] - v[1], v[2] + v[3], v[2] - v[3] };
    const s2 = [4]f32{ s1[0] + s1[2], s1[1] + s1[3], s1[0] - s1[2], s1[1] - s1[3] };
    return .{ s2[0] * 0.5, s2[1] * 0.5, s2[2] * 0.5, s2[3] * 0.5 };
}

fn testHostFwht() !void {
    const x = [4]f32{ 1.0, 2.0, 3.0, 4.0 };
    const y = fwht4(x);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), y[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), y[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), y[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[3], 1e-6);
    const z = fwht4(y);
    try std.testing.expectApproxEqAbs(x[0], z[0], 1e-6);
    try std.testing.expectApproxEqAbs(x[1], z[1], 1e-6);
    try std.testing.expectApproxEqAbs(x[2], z[2], 1e-6);
    try std.testing.expectApproxEqAbs(x[3], z[3], 1e-6);

    const w = [4]f32{ 0.5, -0.25, 0.75, 0.0 };
    const wh = fwht4(w);
    var acc: f32 = 0;
    var ref: f32 = 0;
    for (x, y, w, wh) |xi, yi, wi, whi| {
        acc += yi * whi;
        ref += xi * wi;
    }
    try std.testing.expectApproxEqAbs(ref, acc, 1e-6);
}

fn testMultistepAb2() !void {
    var x = [_]f32{1.0};
    const v = [_]f32{2.0};
    const prev = [_]f32{0.0};
    const sig = [_]f32{ 1.0, 0.5, 0.25 };
    multistep_mod.resMultistep(&sig, 1, &x, &v, &prev);
    try std.testing.expectApproxEqAbs(@as(f32, 1.75), x[0], 1e-6);
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
    try std.testing.expectEqual(policy_mod.AttnKind.vanilla, short);

    const long = policy_mod.selectAttention(.{
        .target = .cuda,
        .dtype = .bf16,
        .head_dim = 128,
        .heads = 56,
        .seq = 7440,
        .causal = false,
        .tp = 2,
    });
    try std.testing.expectEqual(policy_mod.AttnKind.cuda_fa2, long);

    try std.testing.expectEqual(policy_mod.AttnKind.vanilla, policy_mod.selectAttention(.{
        .target = .cpu,
        .dtype = .bf16,
        .head_dim = 128,
        .heads = 56,
        .seq = 7440,
        .causal = false,
        .tp = 1,
    }));
    try std.testing.expectEqual(policy_mod.AttnKind.vanilla, policy_mod.selectAttention(.{
        .target = .cuda,
        .dtype = .f32,
        .head_dim = 128,
        .heads = 56,
        .seq = 7440,
        .causal = false,
        .tp = 2,
    }));

    for ([_]u32{ 1, 2, 4, 8 }) |tp| {
        try std.testing.expectEqual(policy_mod.AttnKind.cuda_fa2, policy_mod.selectAttention(.{
            .target = .cuda,
            .dtype = .bf16,
            .head_dim = 128,
            .heads = 56,
            .seq = 7440,
            .causal = false,
            .tp = tp,
        }));
        try std.testing.expectEqual(policy_mod.AttnKind.vanilla, policy_mod.selectAttention(.{
            .target = .cuda,
            .dtype = .bf16,
            .head_dim = 128,
            .heads = 56,
            .seq = 64,
            .causal = false,
            .tp = tp,
        }));
        try std.testing.expectEqual(policy_mod.AttnKind.vanilla, policy_mod.selectAttention(.{
            .target = .cuda,
            .dtype = .bf16,
            .head_dim = 128,
            .heads = 56,
            .seq = 256,
            .causal = false,
            .tp = tp,
        }));
    }
}

fn testMemoryPlanExact(allocator: std.mem.Allocator) !void {
    const geo: pipeline.Geometry = .{
        .pixel_w = 640,
        .pixel_h = 352,
        .frames = 107,
        .latent_t = 8,
        .latent_h = 22,
        .latent_w = 40,
        .audio_t = 200,
        .video_tokens = 7040,
        .audio_tokens = 400,
        .target_video_tokens = 7040,
        .target_audio_tokens = 400,
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
    const sdpa = memory_mod.planWith(.{
        .geo = geo,
        .layout = layout,
        .hidden = 5376,
        .steps = 9,
        .device_bytes = 48 * 1024 * 1024 * 1024,
        .tp = 1,
        .target = .cpu,
        .heads = 56,
        .head_dim = 128,
        .layers = 50,
    });
    try std.testing.expectEqual(policy_mod.sdpaScoreBytes(seq, 56, 1), sdpa.score_bytes);
    try std.testing.expect(policy_mod.sdpaScoreBytes(7440, 56, 1) > 10 * 1024 * 1024 * 1024);

    const fa2 = memory_mod.planWith(.{
        .geo = geo,
        .layout = layout,
        .hidden = 5376,
        .steps = 9,
        .device_bytes = 48 * 1024 * 1024 * 1024,
        .tp = 2,
        .target = .cuda,
        .heads = 56,
        .head_dim = 128,
        .layers = 50,
    });
    try std.testing.expectEqual(policy_mod.AttnKind.cuda_fa2, fa2.attention);
    try std.testing.expect(fa2.fa2_scratch_bytes > 0);
    try std.testing.expect(fa2.fa2_scratch_bytes < fa2.score_bytes);
    try std.testing.expect(fa2.adaln_table_bytes > 0);
    try std.testing.expect(policy_mod.groupSize(0) == 1);
    try std.testing.expect(policy_mod.groupSize(2) == 2);
    try std.testing.expect(policy_mod.groupSize(8) == 4);
    try std.testing.expectEqual(@as(u32, 1), policy_mod.tileBatch(0, 1, 100, 2));
    try std.testing.expectEqual(@as(u32, 4), policy_mod.tileBatch(4, 1, 100, 1));
    try std.testing.expectEqual(@as(u32, 1), policy_mod.tileBatch(3, 100, 50, 2));
    try std.testing.expectEqual(@as(u32, 4), policy_mod.tileBatch(5, 1, 100, 2));
    try std.testing.expect(fa2.tile_batch >= 1);
    try std.testing.expect(pipeline.partitionsVaeBatch(6, 2));
    try std.testing.expect(!pipeline.partitionsVaeBatch(6, 1));
    try std.testing.expect(!pipeline.partitionsVaeBatch(5, 2));
    try std.testing.expect(!pipeline.partitionsVaeBatch(1, 2));
}

fn testAdalnIndexLayout(allocator: std.mem.Allocator) !void {
    const layout = try packing.build(allocator, .{
        .text_len = 4,
        .latent_t = 2,
        .latent_h = 4,
        .latent_w = 4,
        .audio_t = 3,
        .video_t = 0.25,
        .audio_t_noise = 0.6,
    });
    defer layout.deinit(allocator);
    const idx = try pipeline.adalnIndices(allocator, layout);
    defer allocator.free(idx);
    try std.testing.expectEqual(layout.seqLen(), idx.len);
    try std.testing.expectEqual(layout.adalnIndex(0), idx[0]);
    try std.testing.expectEqual(@as(u32, 1), idx[0] % 3);
    try std.testing.expectEqual(@as(u32, 2), idx[layout.target_audio_start] % 3);
    try std.testing.expectEqual(@as(u32, 0), idx[layout.target_video_start] % 3);

    const slots = packing.timestep_slot_count;
    const steps: u32 = 4;
    const hidden: i64 = 16;
    const bytes = policy_mod.adalnTableBytes(steps, hidden, 2, 2);
    const per_block = @as(u64, steps) * slots * 3 * 6 * 16 * 2;
    const final = @as(u64, steps) * slots * 2 * 16 * 2;
    try std.testing.expectEqual(per_block * 2 + final, bytes);
}

fn testSchedulerFormula() !void {
    const x = [_]f32{1.0};
    const v = [_]f32{1.0};
    const prev = [_]f32{0.5};
    const sig = [_]f32{ 1.0, 0.5, 0.0 };
    var y = x;
    multistep_mod.resMultistep(&sig, 0, &y, &v, null);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), y[0], 1e-6);
    y = x;
    multistep_mod.resMultistep(&sig, 1, &y, &v, &prev);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), y[0], 1e-6);
}
