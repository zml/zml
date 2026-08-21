const std = @import("std");

const zml = @import("zml");
const common = @import("common.zig");
const model = @import("kimi_k3/model.zig");
const runtime_weights = @import("kimi_k3/runtime_weights.zig");

const Args = struct {
    model: []const u8,
    devices: usize = 1,
    tensor_parallel: usize = 1,

    pub const help =
        \\Use kimi_k3_readiness_tests --model=<checkpoint-directory> [options]
        \\
        \\Validate the complete 93-layer logical, cache, streaming-load, and
        \\distributed ownership plans without opening missing weight shards.
        \\
        \\Options:
        \\  --devices=<count>          Planned accelerator count (default: 1)
        \\  --tensor-parallel=<count>  Tensor-parallel degree (default: 1)
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    const parsed = try common.parseConfig(model.Config, allocator, io, repo);
    defer parsed.deinit();
    try parsed.value.validate();

    var logical = try model.ModelPlan.init(allocator, parsed.value, .{});
    defer logical.deinit(allocator);
    if (logical.layers.len != 93) return error.KimiK3LogicalLayerCountMismatch;

    var kda_dense: usize = 0;
    var kda_moe: usize = 0;
    var mla_moe: usize = 0;
    for (logical.layers) |kind| switch (kind) {
        .kda_dense => kda_dense += 1,
        .kda_moe => kda_moe += 1,
        .mla_moe => mla_moe += 1,
    };
    if (kda_dense != 1 or kda_moe != 68 or mla_moe != 24) {
        return error.KimiK3LogicalFamilyCountMismatch;
    }

    var caches = try model.PackedCachePlan.init(allocator, parsed.value);
    defer caches.deinit(allocator);
    if (caches.kda_count != 69 or caches.mla_count != 24 or caches.attn_res_persisted) {
        return error.KimiK3PackedCacheCountMismatch;
    }
    const one_million = try caches.memoryBytes(1, 1_000_000);
    if (one_million.total_bytes != 28_102_459_392) return error.KimiK3CacheMemoryMismatch;

    const streaming = try runtime_weights.StreamingLoadPlan.init(logical.layers.len);
    if (streaming.resident_layers != 1 or streaming.staged_layers != 92) {
        return error.KimiK3StreamingPlanMismatch;
    }

    const distributed = try runtime_weights.DistributedPlan.init(args.devices, args.tensor_parallel);
    var next_expert: usize = 0;
    var smallest_partition: usize = runtime_weights.expert_count;
    var largest_partition: usize = 0;
    for (0..distributed.expert_parallel) |expert_rank| {
        const partition = try distributed.expertPartition(expert_rank);
        if (partition.first != next_expert) return error.KimiK3ExpertPartitionGap;
        next_expert = partition.end;
        smallest_partition = @min(smallest_partition, partition.count());
        largest_partition = @max(largest_partition, partition.count());
    }
    if (next_expert != runtime_weights.expert_count or largest_partition - smallest_partition > 1) {
        return error.KimiK3ExpertPartitionCoverageMismatch;
    }

    const source_slots = std.math.divCeil(usize, logical.layers.len, 12) catch unreachable;
    if (source_slots != 8) return error.KimiK3AttentionResidualSourceMismatch;

    var stdout = std.Io.File.stdout().writerStreaming(io, &.{});
    try stdout.interface.print(
        "KIMI_K3_FULL_READINESS_PASS layers={} kda_dense={} kda_moe={} mla_moe={} " ++
            "kda_caches={} mla_caches={} source_slots={} cache_1m_bytes={} " ++
            "resident_layers={} staged_layers={} host_staging_bytes={} expert_bank_bytes={} " ++
            "devices={} tensor_parallel={} expert_parallel={} experts_per_rank={}..{}\n",
        .{
            logical.layers.len,
            kda_dense,
            kda_moe,
            mla_moe,
            caches.kda_count,
            caches.mla_count,
            source_slots,
            one_million.total_bytes,
            streaming.resident_layers,
            streaming.staged_layers,
            streaming.peak_host_staging_bytes,
            streaming.expert_bank_device_bytes,
            distributed.device_count,
            distributed.tensor_parallel,
            distributed.expert_parallel,
            smallest_partition,
            largest_partition,
        },
    );
    try stdout.interface.flush();
}
