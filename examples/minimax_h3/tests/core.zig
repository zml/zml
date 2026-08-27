const std = @import("std");

const zml = @import("zml");

const memory_mod = @import("../core/memory.zig");
const policy_mod = @import("../core/policy.zig");
const config = @import("../core/config.zig");
const packing = @import("../model/packing.zig");
const pipeline = @import("../runtime/pipeline.zig");
const repo = @import("../runtime/repository.zig");
const request_mod = @import("../core/request.zig");
const sharding_mod = @import("../core/sharding.zig");
const vae = @import("../vae/geometry.zig");

pub fn run(allocator: std.mem.Allocator) !void {
    try testConfig();
    try testCli();
    try testSharding(allocator);
    try testSplitComma(allocator);
    try testFrameGeometry();
    try testCanvasPresets();
    try testRequest(allocator);
    try testCheckpoint();
    try testMemoryPlan(allocator);
    try testOfficialPin();
    try testTokenizerRelpaths();
    try testWeightEntrypoints();
    try testConvrotMarker();
    try testGroupRefs(allocator);
    try testSchemaFixtures();
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
    try std.testing.expectError(error.InvalidDuration, config.checkDuration(4));
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
fn testFrameGeometry() !void {
    try std.testing.expectEqual(@as(u32, 120), config.frameCount(5.0));
    try std.testing.expectEqual(@as(u32, 5), config.alignFrameCount(1));
    try std.testing.expectEqual(@as(u32, 5), config.alignFrameCount(5));
    try std.testing.expectEqual(@as(u32, 22), config.alignFrameCount(17));
    try std.testing.expectEqual(@as(u32, 124), config.alignFrameCount(120));
    try std.testing.expectEqual(@as(u32, 37), config.videoLatentFrames(124));
    try std.testing.expectEqual(@as(u32, 207), config.audioLatentLength(5.0));
}
fn testCanvasPresets() !void {
    const official = try config.parseSize("1344x768");
    try std.testing.expectEqual(@as(u32, 1344), official.w);
    try std.testing.expectEqual(@as(u32, 768), official.h);
    const snapped = try config.parseSize("1340x770");
    try std.testing.expectEqual(@as(u32, 1344), snapped.w);
    try std.testing.expectEqual(@as(u32, 768), snapped.h);
    try std.testing.expectError(error.InvalidSize, config.parseSize("1344"));
    try std.testing.expectError(error.InvalidAspect, config.parseSize("100x10"));
    try std.testing.expectError(error.SizeTooLarge, config.parseSize("1920x1080"));
    try config.checkSteps(30);
    try std.testing.expectError(error.TooFewSteps, config.checkSteps(1));

    const consumer_bytes = 24 * 1024 * 1024 * 1024;
    try std.testing.expectError(error.SizeTooLarge, config.checkDeviceForSize(1344, 768, consumer_bytes));
    try config.checkDeviceForSize(640, 352, consumer_bytes);
    try config.checkDeviceForSize(1344, 768, 80 * 1024 * 1024 * 1024);
    try config.checkDeviceForSize(1344, 768, 0);
}
fn testRequest(allocator: std.mem.Allocator) !void {
    const refs = try request_mod.refsFromComma(allocator, "a.png, clip.mp4, bed.wav");
    defer request_mod.freeRefs(allocator, refs, false);
    try std.testing.expectEqual(@as(usize, 2), refs.len);
    try std.testing.expectEqual(packing.ReferenceKind.image, refs[0].kind);
    try std.testing.expectEqual(packing.ReferenceKind.video_audio, refs[1].kind);
    try std.testing.expectEqualStrings("clip.mp4", refs[1].path);
    try std.testing.expectEqualStrings("bed.wav", refs[1].soundtrack);

    try request_mod.validate(.{ .prompt = "hi", .variant = .t2va });
    try std.testing.expectError(error.IntentEmpty, request_mod.validate(.{ .prompt = "   ", .variant = .t2va }));
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
    try std.testing.expectError(error.AudioRefNeedsVisual, request_mod.validateRefs(audio_only));
    const too_many = try request_mod.refsFromComma(allocator, "a.png,b.png,c.png,d.png,e.png,f.png,g.png,h.png,i.png,j.png");
    defer request_mod.freeRefs(allocator, too_many, false);
    try std.testing.expectError(error.TooManyRefImages, request_mod.validateRefs(too_many));
    try std.testing.expectEqual(config.Variant.t2va, try request_mod.inferVariant("", "", &.{}));
    try std.testing.expectEqual(config.Variant.fl2va, try request_mod.inferVariant("a.png", "", &.{}));
    try std.testing.expectEqual(config.Variant.fl2va, try request_mod.inferVariant("", "a.png", &.{}));
    try std.testing.expectEqual(config.Variant.fl2va, try request_mod.inferVariant("a.png", "b.png", &.{}));
    try std.testing.expectEqual(config.Variant.ref2va, try request_mod.inferVariant("", "", refs));
    try std.testing.expectError(error.Ref2vaRejectsKeyframes, request_mod.inferVariant("a.png", "", refs));
}
fn testCheckpoint() !void {
    const full = [_][]const u8{
        "video_patch_proj.weight",
        "time_embedder.proj_in.weight",
        "blocks.0.adaln_proj.linear.weight",
        "final_layer.adaln_proj.linear.weight",
    };
    try std.testing.expect(repo.inspect(&full).has_adaln_proj);
    try std.testing.expect(repo.inspect(&full).has_time);
    try std.testing.expect(repo.refuseReason(repo.inspect(&full)) == null);

    const table_only = [_][]const u8{ "adaln_t_table", "video_patch_proj.weight" };
    try std.testing.expect(!repo.inspect(&table_only).has_adaln_proj);
    try std.testing.expect(repo.inspect(&table_only).has_time);
    try std.testing.expect(repo.refuseReason(repo.inspect(&table_only)) != null);

    const no_time = [_][]const u8{"blocks.0.adaln_proj.linear.weight"};
    try std.testing.expect(repo.inspect(&no_time).has_adaln_proj);
    try std.testing.expect(!repo.inspect(&no_time).has_time);
    try std.testing.expect(repo.refuseReason(repo.inspect(&no_time)) != null);

    const rank8 = [_][]const u8{ "adaln_t_table", "blocks.0.adaln_proj.linear.weight" };
    try std.testing.expect(repo.inspect(&rank8).has_adaln_proj);
    try std.testing.expect(repo.inspect(&rank8).has_time);
    try std.testing.expect(repo.refuseReason(repo.inspect(&rank8)) == null);

    const official = [_][]const u8{
        "proj_in.weight",
        "time_embedder.linear_1.weight",
        "transformer_blocks.0.adaln_proj.linear.weight",
    };
    try std.testing.expect(repo.inspect(&official).has_adaln_proj);
    try std.testing.expect(repo.inspect(&official).has_time);
    try std.testing.expect(repo.refuseReason(repo.inspect(&official)) == null);

    try std.testing.expect(repo.safetensorsContains("minimax_h3_fl2va_pruned_int8.safetensors", &.{"fl2va"}));
    try std.testing.expect(repo.safetensorsContains("minimax_h3_ref2va_fp8_scaled.safetensors", &.{"ref2va"}));
    try std.testing.expect(!repo.safetensorsContains("minimax_h3_fl2va_pruned_int8.safetensors", &.{"ref2va"}));
    try std.testing.expect(!repo.safetensorsContains("notes.txt", &.{"fl2va"}));
    try std.testing.expect(repo.safetensorsContains("qwen3vl_32b_minimax_h3_int8_convrot.safetensors", &.{}));
    try std.testing.expect(repo.safetensorsContains("minimax_h3_video_vae_fp16.safetensors", &.{ "video", "vae" }));
    try std.testing.expect(repo.safetensorsContains("minimax_h3_audio_vae_fp32.safetensors", &.{ "audio", "vae" }));
    try std.testing.expect(!repo.safetensorsContains("minimax_h3_audio_vae_fp32.safetensors", &.{ "video", "vae" }));
    try std.testing.expect(repo.isBundleLeaf("text_encoders"));
    try std.testing.expect(!repo.isBundleLeaf("transformer"));
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
    const tiny = memory_mod.plan(.{
        .geo = geo,
        .layout = layout,
        .hidden = 256,
        .steps = 4,
        .device_bytes = 24 * 1024 * 1024 * 1024,
        .tp = 2,
    });
    try std.testing.expect(tiny.safe);
    const huge_geo = blk: {
        var g = geo;
        g.pixel_w = 1344;
        g.pixel_h = 768;
        break :blk g;
    };
    const full = memory_mod.plan(.{
        .geo = huge_geo,
        .layout = layout,
        .hidden = 5376,
        .steps = 30,
        .device_bytes = 24 * 1024 * 1024 * 1024,
        .tp = 2,
    });
    try std.testing.expect(!full.safe);
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
fn testWeightEntrypoints() !void {
    try std.testing.expectEqual(@as(usize, 4), repo.weight_entrypoints.len);
    try std.testing.expectEqualStrings("model.safetensors.index.json", repo.weight_entrypoints[0]);
    try std.testing.expectEqualStrings("diffusion_pytorch_model.safetensors.index.json", repo.weight_entrypoints[2]);
    try std.testing.expectEqualStrings("diffusion_pytorch_model.safetensors", repo.weight_entrypoints[3]);
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
fn testGroupRefs(allocator: std.mem.Allocator) !void {
    const src = try request_mod.refsFromComma(allocator, "a.wav, b.png, c.mp4");
    defer request_mod.freeRefs(allocator, src, false);
    try std.testing.expectEqual(packing.ReferenceKind.audio, src[0].kind);
    try std.testing.expectEqual(packing.ReferenceKind.image, src[1].kind);
    try std.testing.expectEqual(packing.ReferenceKind.video, src[2].kind);
}
fn testSchemaFixtures() !void {
    try std.testing.expect(repo.refuseReason(repo.inspect(&.{})) != null);

    const int8 = [_][]const u8{ "blocks.0.adaln_proj.linear.weight", "weight_scale", "adaln_t_table" };
    try std.testing.expect(repo.inspect(&int8).has_adaln_proj);
    try std.testing.expect(repo.inspect(&int8).has_time);
    try std.testing.expect(repo.refuseReason(repo.inspect(&int8)) == null);
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
    try std.testing.expectEqual(zml.attention.Backend.vanilla, policy_mod.selectAttention(.{
        .target = .cuda,
        .dtype = .f32,
        .head_dim = 128,
        .heads = 56,
        .seq = 7440,
        .causal = false,
        .tp = 2,
    }));

    for ([_]u32{ 1, 2, 4, 8 }) |tp| {
        try std.testing.expectEqual(zml.attention.Backend.cuda_fa2, policy_mod.selectAttention(.{
            .target = .cuda,
            .dtype = .bf16,
            .head_dim = 128,
            .heads = 56,
            .seq = 7440,
            .causal = false,
            .tp = tp,
        }));
        try std.testing.expectEqual(zml.attention.Backend.vanilla, policy_mod.selectAttention(.{
            .target = .cuda,
            .dtype = .bf16,
            .head_dim = 128,
            .heads = 56,
            .seq = 64,
            .causal = false,
            .tp = tp,
        }));
        try std.testing.expectEqual(zml.attention.Backend.vanilla, policy_mod.selectAttention(.{
            .target = .cuda,
            .dtype = .bf16,
            .head_dim = 128,
            .heads = 56,
            .seq = 256,
            .causal = false,
            .tp = tp,
        }));
        try std.testing.expectEqual(zml.attention.Backend.cuda_fa3, policy_mod.selectAttention(.{
            .target = .cuda,
            .dtype = .bf16,
            .head_dim = 128,
            .heads = 56,
            .seq = 7440,
            .causal = false,
            .tp = tp,
            .flash = .cuda_fa3,
        }));
    }
    try std.testing.expect(policy_mod.isFlash(.cuda_fa2));
    try std.testing.expect(policy_mod.isFlash(.cuda_fa3));
    try std.testing.expect(!policy_mod.isFlash(.vanilla));
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
    const sdpa = memory_mod.plan(.{
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

    const fa2 = memory_mod.plan(.{
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
    try std.testing.expectEqual(zml.attention.Backend.cuda_fa2, fa2.attention);
    try std.testing.expect(fa2.fa2_scratch_bytes > 0);
    try std.testing.expect(fa2.fa2_scratch_bytes < fa2.score_bytes);
    const fa3 = memory_mod.plan(.{
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
        .flash = .cuda_fa3,
    });
    try std.testing.expectEqual(zml.attention.Backend.cuda_fa3, fa3.attention);
    try std.testing.expectEqual(fa2.fa2_scratch_bytes, fa3.fa2_scratch_bytes);
    try std.testing.expect(fa2.adaln_table_bytes > 0);
    try std.testing.expect(policy_mod.groupSize(0) == 1);
    try std.testing.expect(policy_mod.groupSize(2) == 2);
    try std.testing.expect(policy_mod.groupSize(8) == 4);
    try std.testing.expectEqual(@as(u32, 4), policy_mod.enc_prefetch);
    try std.testing.expectEqual(@as(u32, 8), policy_mod.vae_load_window);
    try std.testing.expectEqual(@as(u32, 50), policy_mod.ditKeepBlocks(46, 50));
    try std.testing.expectEqual(@as(u32, 50), policy_mod.ditKeepBlocks(50, 50));
    try std.testing.expectEqual(@as(u32, 40), policy_mod.ditKeepBlocks(40, 50));
    try std.testing.expectEqual(@as(u32, 0), policy_mod.ditKeepBlocks(0, 50));
    try std.testing.expectEqual(@as(u32, 3), policy_mod.ditKeepBlocks(2, 3));
    const gib: u64 = 1024 * 1024 * 1024;
    const enc_layer: u64 = gib;
    try std.testing.expectEqual(@as(u32, 0), policy_mod.encKeepLayers(24 * gib, enc_layer, 50));
    try std.testing.expectEqual(@as(u32, 50), policy_mod.encKeepLayers(284 * gib, enc_layer, 50));
    try std.testing.expectEqual(@as(u32, 50), policy_mod.encKeepLayers(0, enc_layer, 50));
    try std.testing.expectEqual(@as(u32, 50), policy_mod.encKeepLayers(48 * gib, 0, 50));
    try std.testing.expectEqual(@as(u32, 0), policy_mod.encKeepLayers(48 * gib, enc_layer, 0));
    const gb300 = policy_mod.decide(.{
        .target = .cuda,
        .seq = 8632,
        .hidden = 5376,
        .heads = 56,
        .head_dim = 128,
        .layers = 50,
        .steps = 9,
        .dtype = .bf16,
        .device_bytes = 284 * gib,
        .tp = 4,
        .devices = 4,
        .block_core_bytes = 1300 * 1024 * 1024,
        .dtype_bytes = 2,
    });
    try std.testing.expectEqual(@as(u32, 50), gb300.resident_blocks);
    try std.testing.expectEqual(@as(u32, 50), policy_mod.ditKeepBlocks(gb300.resident_blocks, 50));
    const unreported = policy_mod.decide(.{
        .target = .cuda,
        .seq = 8632,
        .hidden = 5376,
        .heads = 56,
        .head_dim = 128,
        .layers = 50,
        .steps = 9,
        .dtype = .bf16,
        .device_bytes = 0,
        .tp = 4,
        .devices = 4,
        .block_core_bytes = 1300 * 1024 * 1024,
        .dtype_bytes = 2,
    });
    try std.testing.expectEqual(@as(u32, 50), unreported.resident_blocks);
    try std.testing.expectEqual(@as(u32, 1), policy_mod.tileBatch(0, 1, 100, 2));
    try std.testing.expectEqual(@as(u32, 4), policy_mod.tileBatch(4, 1, 100, 1));
    try std.testing.expectEqual(@as(u32, 1), policy_mod.tileBatch(3, 100, 50, 2));
    try std.testing.expectEqual(@as(u32, 4), policy_mod.tileBatch(5, 1, 100, 2));
    try std.testing.expect(fa2.tile_batch >= 1);
    try std.testing.expect(pipeline.partitionsVaeBatch(6, 2));
    try std.testing.expect(!pipeline.partitionsVaeBatch(6, 1));
    try std.testing.expect(!pipeline.partitionsVaeBatch(5, 2));
    try std.testing.expect(!pipeline.partitionsVaeBatch(1, 2));
    try std.testing.expect(pipeline.partitionsVaeBatch(28, 2));
    try std.testing.expect(pipeline.partitionsVaeBatch(28, 4));
    try std.testing.expect(!pipeline.partitionsVaeBatch(28, 8));
}
