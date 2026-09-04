const std = @import("std");

const euler = @import("../refine/euler.zig");
const gemma4 = @import("../refine/gemma.zig");
const prepare = @import("../refine/handoff.zig");
const load = @import("../refine/load.zig");
const lora = @import("../recipe/lora.zig");
const ltx_rope = @import("../refine/ltx_rope.zig");
const ltx_upsampler = @import("../refine/ltx_up.zig");
const sol_attn = @import("../refine/sol_attn.zig");
const taeh3 = @import("../draft/taeh3.zig");
const taehv = @import("../refine/taehv.zig");
const noise = @import("../draft/noise.zig");
const sharding = @import("../recipe/shard.zig");
const sku = @import("../recipe/sku.zig");

// =============================================================================
// tests/recipe.zig — SKU catalog, LoRA map, TAEHV stitch
// =============================================================================

pub fn run(allocator: std.mem.Allocator) !void {
    _ = allocator;
    try testRecipe();
    try testMapOfficial();
    try testMergeInto();
    try testCrop();
    try testVisibleDevices();
    try testSingleDeviceApi();
    try testHandoffTrim();
    try testTaehvChunk();
    try testVideoRopeMid();
    try testConstNoiseMatchesTorch();
    try testGemmaPad();
    sharding.prepareDeviceCap(0);
}

fn testGemmaPad() !void {
    var dst: [8]u32 = undefined;
    gemma4.padPromptTokens(&dst, &.{ 10, 11, 12 });
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 0, 0, gemma4.bos_id, 10, 11, 12 }, &dst);
    gemma4.padPromptTokens(&dst, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    try std.testing.expectEqualSlices(u32, &.{ 2, 3, 4, 5, 6, 7, 8, 9 }, &dst);
}

fn testVideoRopeMid() !void {
    const first = ltx_rope.videoPixelMid(0, 0, 0, 24);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5 / 24.0), first[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 16.0), first[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 16.0), first[2], 1e-12);
    const next = ltx_rope.videoPixelMid(1, 1, 1, 24);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 24.0), next[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 48.0), next[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 48.0), next[2], 1e-12);
}

fn testConstNoiseMatchesTorch() !void {
    var gen = noise.Generator.init(42);
    var drawn: [32]f32 = undefined;
    noise.randn(&gen, &drawn);
    const torch = [_]f32{ 1.926915, 1.4872842, 0.9007172, -2.1055214, 0.67841846, -1.234545, -0.04306748, -1.604667 };
    for (drawn[0..8], torch) |g, w| try std.testing.expectApproxEqAbs(w, g, 2e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), euler.mixConst(0.5, 0.0, 1.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.090625), euler.mixConst(0.909375, 1.0, 0.0), 1e-6);
}

fn testHandoffTrim() !void {
    try std.testing.expectEqual(@as(u32, 1), prepare.trimFrames(1));
    try std.testing.expectEqual(@as(u32, 9), prepare.trimFrames(10));
    try std.testing.expectEqual(@as(u32, 121), prepare.trimFrames(124));
}

fn testTaehvChunk() !void {
    try std.testing.expectEqual(@as(u32, 121), taehv.outFrames(16));
    try std.testing.expectEqual(@as(u32, 241), taehv.outFrames(31));
    try std.testing.expectEqual(@as(u32, 361), taehv.outFrames(46));
    try std.testing.expectEqual(@as(u32, 0), taehv.chunkDrop(0, 0));
    // After a t=16 window, the next start=8 window must drop overlap*8 - trim, not overlap*8.
    try std.testing.expectEqual(@as(u32, 57), taehv.chunkDrop(121, 8));
    try std.testing.expectEqual(@as(u32, 9), taehv.chunkDrop(121, 14));
    try std.testing.expectEqual(taehv.outFrames(16), taehv.chunkedCoverage(16, 8, 16));
    try std.testing.expectEqual(taehv.outFrames(31), taehv.chunkedCoverage(16, 8, 31));
    try std.testing.expectEqual(taehv.outFrames(46), taehv.chunkedCoverage(16, 8, 46));

    const keep = taehv.outFrames(31);
    var dst: [3 * 241]f32 = undefined;
    try std.testing.expectEqual(keep, taehv.stitchPattern(&dst, 16, 8, 31, 8));
    var i: u32 = 0;
    while (i < keep) : (i += 1) {
        const want = @as(f32, @floatFromInt(i));
        try std.testing.expectApproxEqAbs(want, dst[i], 1e-4);
        try std.testing.expectApproxEqAbs(want, dst[keep + i], 1e-4);
        try std.testing.expectApproxEqAbs(want, dst[2 * keep + i], 1e-4);
    }

    const keep_long = taehv.outFrames(46);
    var dst_long: [3 * 361]f32 = undefined;
    try std.testing.expectEqual(keep_long, taehv.stitchPattern(&dst_long, 16, 8, 46, 8));
    i = 0;
    while (i < keep_long) : (i += 1) {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(i)), dst_long[i], 1e-4);
    }
}

fn testRecipe() !void {
    try std.testing.expectEqual(@as(u32, 896), sku.draft_width);
    try std.testing.expectEqual(@as(u32, 512), sku.draft_height);
    try std.testing.expectEqual(@as(u32, 4), sku.denoise_evals);
    try std.testing.expectEqual(@as(u32, 5), sku.schedule_points);
    try std.testing.expectEqual(@as(f32, 12.0), sku.turbo_video_shift);
    try std.testing.expectEqual(@as(f32, 3.0), sku.turbo_audio_shift);
    try std.testing.expectEqual(@as(f32, 1.0), sku.lora_strength);
    try std.testing.expectEqual(@as(u32, 3), sku.ltx_refine_evals);
    try std.testing.expectEqual(@as(f32, 0.8), sku.ltx_lora_strength);
    try std.testing.expectEqual(@as(f32, 0.909375), sku.ltx_stage2_sigmas[0]);
    try std.testing.expectEqual(@as(f32, 0.725), sku.ltx_stage2_sigmas[1]);
    try std.testing.expectEqual(@as(f32, 0.421875), sku.ltx_stage2_sigmas[2]);
    try std.testing.expectEqual(@as(u32, 256), sku.prompt_tokens);
    try std.testing.expectEqualStrings("output/xla-cache", sku.default_cache_dir);
    try std.testing.expectEqualStrings("hf://MiniMaxAI/MiniMax-H3", sku.default_model);
    try std.testing.expect(std.mem.startsWith(u8, sku.default_lora_path, "hf://"));
    try std.testing.expect(std.mem.startsWith(u8, sku.hf_ltx_dit, "hf://Lightricks/LTX-2.5/"));
    try std.testing.expect(load.isUri(sku.default_taeh3_path));
    try std.testing.expect(load.isUri(sku.http_taehv));
    try std.testing.expect(!load.isUri("/var/models/super-accel/taeh3.safetensors"));
    try std.testing.expectEqual(sku.prompt_tokens, gemma4.pad_len);
    try std.testing.expectEqual(@as(u32, 5), sku.h3OutFrames(2));
    try std.testing.expectEqual(@as(u32, 22), sku.h3OutFrames(7));
    try std.testing.expectEqual(@as(u32, 124), sku.h3OutFrames(37));
    const draft = sku.draftCanvas();
    try std.testing.expectEqual(@as(u32, 896), draft.w);
    try std.testing.expectEqual(@as(u32, 512), draft.h);
    const half = try sku.refineEncodeSize(1344, 768);
    try std.testing.expectEqual(@as(u32, 672), half.w);
    try std.testing.expectEqual(@as(u32, 384), half.h);
    const hd_enc = try sku.refineEncodeSize(1920, 1088);
    try std.testing.expectEqual(@as(u32, 960), hd_enc.w);
    try std.testing.expectEqual(@as(u32, 544), hd_enc.h);
    try std.testing.expectEqual(@as(usize, 6), sku.skus.len);
    try std.testing.expectEqualStrings("5s", sku.default_sku_id);
    try std.testing.expect(sku.byId("5s") != null);
    try std.testing.expect(sku.byId("10s") != null);
    try std.testing.expect(sku.byId("15s") != null);
    try std.testing.expect(sku.byId("5s-hd") != null);
    try std.testing.expect(sku.byId("10s-hd") != null);
    try std.testing.expect(sku.byId("15s-hd") != null);
    try std.testing.expectEqual(@as(f32, 15), sku.byId("15s-hd").?.duration_s);
    try std.testing.expect(sku.isHd(sku.byId("15s-hd").?));
    try std.testing.expect(sku.byId("nope") == null);
    const hd = sku.byId("5s-hd").?;
    try std.testing.expectEqual(@as(u32, 1280), hd.draft_w);
    try std.testing.expectEqual(@as(u32, 704), hd.draft_h);
    try std.testing.expectEqual(@as(u32, 1920), hd.target_w);
    try std.testing.expectEqual(@as(u32, 1088), hd.target_h);
    try std.testing.expect(sku.isHd(hd));
    try std.testing.expect(!sku.isHd(sku.byId("5s").?));
    try std.testing.expect(sku.isRequired(sku.byId("5s").?));
    try std.testing.expect(!sku.isRequired(hd));
    try std.testing.expectEqualStrings("Super", sku.familyLabel(sku.byId("5s").?));
    try std.testing.expectEqualStrings("Full HD", sku.familyLabel(hd));
    try std.testing.expectEqual(@as(u32, 34), sku.hdUpsampledH());
    try std.testing.expectEqual(@as(u32, 10), gemma4.keep_tokens);
    try std.testing.expectEqual(@as(u32, 49), gemma4.stack_layers);
    try std.testing.expectEqual(@as(i64, 6144), gemma4.proj_dim);
    try std.testing.expectEqual(@as(i64, 16), taehv.window_t);
    try std.testing.expectEqual(@as(i64, 16), taehv.decodeTime(31, 34, sku.hdUpsampledH()));
    try std.testing.expectEqual(@as(i64, 8), taehv.decodeTime(8, 34, sku.hdUpsampledH()));
    try std.testing.expectEqual(@as(i64, 31), taehv.decodeTime(31, 24, sku.hdUpsampledH()));
    try std.testing.expectEqual(@as(u32, 16), taeh3.spatial_scale);
    try std.testing.expectEqual(@as(u32, 2), ltx_upsampler.spatial_factor);
    try std.testing.expect(sol_attn.tokensOk(16128));
    try std.testing.expect(sol_attn.tokensOk(32640));
    try std.testing.expect(!sol_attn.tokensOk(31248));
    try std.testing.expect(!sol_attn.tokensOk(64));
    try std.testing.expect(sku.useRuntimeLora("", "", false));
    try std.testing.expect(!sku.useRuntimeLora("", "", true));
    try std.testing.expect(!sku.useRuntimeLora("/fused", "", true));
    try std.testing.expect(sku.useRuntimeLora("/fused", "/lora.safetensors", true));
    try std.testing.expectEqualStrings("", sku.resolvedDit("", false));
    try std.testing.expectEqualStrings(sku.default_fused_dit, sku.resolvedDit("", true));
    try std.testing.expectEqualStrings("/custom", sku.resolvedDit("/custom", true));
}

fn testMapOfficial() !void {
    var buf: [256]u8 = undefined;
    const q = lora.mapOfficial("transformer_blocks.3.attn.to_q.weight", &buf).?;
    try std.testing.expectEqual(lora.Part.q, q.part);
    try std.testing.expectEqualStrings("blocks.3.attn.qkv_proj", q.base);

    const k = lora.mapOfficial("transformer_blocks.3.attn.to_k", &buf).?;
    try std.testing.expectEqual(lora.Part.k, k.part);
    try std.testing.expectEqualStrings("blocks.3.attn.qkv_proj", k.base);

    const v = lora.mapOfficial("transformer_blocks.0.attn.to_v.weight", &buf).?;
    try std.testing.expectEqual(lora.Part.v, v.part);

    const out = lora.mapOfficial("transformer_blocks.1.attn.to_out.0.weight", &buf).?;
    try std.testing.expectEqual(lora.Part.full, out.part);
    try std.testing.expectEqualStrings("blocks.1.attn.out_proj", out.base);

    const fc1 = lora.mapOfficial("transformer_blocks.2.ff.net.0.proj.weight", &buf).?;
    try std.testing.expectEqualStrings("blocks.2.mlp.fc1", fc1.base);

    const fc2 = lora.mapOfficial("transformer_blocks.2.ff.net.2.weight", &buf).?;
    try std.testing.expectEqualStrings("blocks.2.mlp.fc2", fc2.base);

    const adaln = lora.mapOfficial("transformer_blocks.4.adaln_proj.linear.weight", &buf).?;
    try std.testing.expectEqualStrings("blocks.4.adaln_proj.linear", adaln.base);

    const fin = lora.mapOfficial("norm_out.linear.weight", &buf).?;
    try std.testing.expectEqualStrings("final_layer.adaln_proj.linear", fin.base);

    const ref = lora.mapOfficial("token_refiner.refiner_blocks.0.attn.to_q.weight", &buf).?;
    try std.testing.expectEqual(lora.Part.q, ref.part);
    try std.testing.expectEqualStrings("token_refiner.blocks.0.attn.qkv_proj", ref.base);

    try std.testing.expect(lora.mapOfficial("proj_in.weight", &buf) == null);
}

fn testMergeInto() !void {
    var w = [_]f32{ 1, 0, 0, 1 };
    const a = [_]f32{ 1, 2 };
    const b = [_]f32{ 3, 4 };
    lora.mergeInto(&w, &a, &b, 2, 2, 1, 1.0);
    try std.testing.expectEqual(@as(f32, 4), w[0]);
    try std.testing.expectEqual(@as(f32, 6), w[1]);
    try std.testing.expectEqual(@as(f32, 4), w[2]);
    try std.testing.expectEqual(@as(f32, 9), w[3]);
}

fn testCrop() !void {
    const c = sku.centerCrop(896, 512, 672, 384);
    try std.testing.expectEqual(@as(u32, 112), c.x);
    try std.testing.expectEqual(@as(u32, 64), c.y);
    try std.testing.expectEqual(@as(u32, 672), c.w);
    try std.testing.expectEqual(@as(u32, 384), c.h);
}

fn testVisibleDevices() !void {
    const one = sku.visibleDevices(1);
    try std.testing.expectEqualStrings("0", std.mem.sliceTo(&one, 0));
    const two = sku.visibleDevices(2);
    try std.testing.expectEqualStrings("0,1", std.mem.sliceTo(&two, 0));
    const four = sku.visibleDevices(4);
    try std.testing.expectEqualStrings("0,1,2,3", std.mem.sliceTo(&four, 0));
    const ten = sku.visibleDevices(10);
    try std.testing.expectEqualStrings("0,1,2,3,4,5,6,7,8,9", std.mem.sliceTo(&ten, 0));
}

fn testSingleDeviceApi() !void {
    try std.testing.expectEqual(@as(usize, 1), sharding.tensorParallelDegreeFor(1, 56, 64, 8));
}
