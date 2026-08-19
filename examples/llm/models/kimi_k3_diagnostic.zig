const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const common = @import("common.zig");
const kimi_k3 = @import("kimi_k3.zig");
const model = kimi_k3.model;
const models = @import("../models.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,
    layer_limit: ?usize = null,
    first_layer: usize = 0,
    metadata_only: bool = false,
    explain: bool = false,
    expect_missing: bool = false,

    pub const help =
        \\Use kimi_k3_diagnostic --model=<path> [--layer-limit=<count>] [options]
        \\
        \\Metadata-only Kimi K3 construction and tensor-contract diagnostic.
        \\
        \\Options:
        \\  --first-layer=<index>  First zero-based logical layer (default: 0)
        \\  --metadata-only        Validate config/schedule without binding tensors
        \\  --explain              Print the selected layer/operator/cache flow
        \\  --expect-missing       Pass only when a required tensor is missing
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = stdx.flags.parse(init.minimal.args, Args);
    const selection: model.LayerSelection = .{
        .first_layer = args.first_layer,
        .layer_limit = args.layer_limit,
    };

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    const parsed = try common.parseConfig(model.Config, allocator, io, repo);
    defer parsed.deinit();
    try parsed.value.validate();
    const last = try selection.end(parsed.value);
    const selected_count = last - selection.first_layer;
    if (try models.detectModelType(allocator, io, repo) != .kimi_k3) return error.ModelRegistrationMismatch;
    var model_plan = try model.ModelPlan.init(allocator, parsed.value, selection);
    defer model_plan.deinit(allocator);
    if (model_plan.layers.len != selected_count) return error.ModelPlanCountMismatch;
    var cache_plan = try model.CachePlan.init(allocator, parsed.value, selection);
    defer cache_plan.deinit(allocator);
    if (cache_plan.layers.len != selected_count) return error.CachePlanCountMismatch;
    var packed_cache = try model.PackedCachePlan.init(allocator, parsed.value);
    defer packed_cache.deinit(allocator);
    if (packed_cache.kda_count != 69 or packed_cache.mla_count != 24 or packed_cache.attn_res_persisted) return error.PackedCachePlanMismatch;
    if (try packed_cache.ordinal(0) != .kda or (try packed_cache.ordinal(0)).kda != 0) return error.KdaCacheOrdinalMismatch;
    if (try packed_cache.ordinal(3) != .mla or (try packed_cache.ordinal(3)).mla != 0) return error.MlaCacheOrdinalMismatch;
    if (try packed_cache.ordinal(92) != .mla or (try packed_cache.ordinal(92)).mla != 23) return error.FinalMlaCacheOrdinalMismatch;
    const million_token_memory = try packed_cache.memoryBytes(1, 1_000_000);
    if (million_token_memory.kda_bytes != 454_459_392 or
        million_token_memory.mla_bytes != 27_648_000_000 or
        million_token_memory.total_bytes != 28_102_459_392)
    {
        return error.PackedCacheMemoryMismatch;
    }
    if (try model.PackedCachePlan.validateAppend(1_048_575, 1, 1_048_576) != 1_048_576) return error.CacheAppendBoundaryMismatch;
    if (model.PackedCachePlan.validateAppend(1_048_576, 1, 1_048_576)) |_| {
        return error.ExpectedCacheCapacityError;
    } else |err| if (err != error.CacheCapacityExceeded) return err;
    if (model.PackedCachePlan.validateAppend(std.math.maxInt(u64), 1, std.math.maxInt(u64))) |_| {
        return error.ExpectedCachePositionOverflow;
    } else |err| if (err != error.CachePositionOverflow) return err;
    if (!model.Model.isIgnoredTensorName("vision_tower.blocks.0.weight") or
        !model.Model.isIgnoredTensorName("mm_projector.proj.weight") or
        model.Model.isIgnoredTensorName("language_model.model.layers.0.input_layernorm.weight"))
    {
        return error.VisionIgnorePolicyMismatch;
    }

    var stdout = std.Io.File.stdout().writerStreaming(io, &.{});
    if (args.explain) try explain(&stdout.interface, parsed.value, selection);

    if (args.metadata_only) {
        if (args.expect_missing) return error.ExpectedTensorConstruction;
        try stdout.interface.print(
            "KIMI_K3_METADATA_PASS layers={} requested_tensors={} kda_caches={} mla_caches={} mla_1m_bytes={} attnres_persisted={}\n",
            .{
                selected_count,
                plannedTensorCount(parsed.value, selection),
                packed_cache.kda_count,
                packed_cache.mla_count,
                million_token_memory.mla_bytes,
                packed_cache.attn_res_persisted,
            },
        );
        try stdout.interface.flush();
        return;
    }

    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    // KIMI_K3_TEMP_REMOVE_M20: this construction-only diagnostic exists for
    // bring-up and must be removed when the production CLI has an equivalent
    // explain/validation path at the cleanup milestone.
    var instance = model.Model.initSelected(
        allocator,
        store.view(),
        parsed.value,
        .{ .max_seq_len = parsed.value.text_config.max_position_embeddings },
        selection,
    ) catch |err| {
        if (args.expect_missing and err == error.MissingRequiredTensor) {
            try stdout.interface.writeAll("KIMI_K3_EXPECTED_MISSING_PASS\n");
            try stdout.interface.flush();
            return;
        }
        return err;
    };
    defer instance.deinit(allocator);

    if (args.expect_missing) return error.ExpectedMissingTensor;
    const planned = plannedTensorCount(parsed.value, selection);
    if (instance.requestedTensorCount() != planned) return error.TensorPlanCountMismatch;
    var unique_names: std.StringHashMapUnmanaged(void) = .empty;
    defer unique_names.deinit(allocator);
    for (instance.layers) |layer| {
        for (layer.weights().tensors) |requirement| {
            if (!std.mem.startsWith(u8, requirement.name, "language_model.model.layers.")) return error.InvalidTensorPrefix;
            if (std.mem.indexOf(u8, requirement.name, "vision") != null) return error.VisionTensorWasRequested;
            const gop = try unique_names.getOrPut(allocator, requirement.name);
            if (gop.found_existing) return error.DuplicateTensorRequirement;
        }
    }
    if (selectedRegistryTensorCount(&registry, selection, last) != planned) return error.SelectedRegistryMismatch;
    try stdout.interface.print(
        "KIMI_K3_CONSTRUCTION_PASS layers={} requested_tensors={} registry_tensors={}\n",
        .{ instance.layers.len, instance.requestedTensorCount(), registry.tensors.count() },
    );
    try stdout.interface.flush();
}

fn selectedRegistryTensorCount(
    registry: *zml.safetensors.TensorRegistry,
    selection: model.LayerSelection,
    last: usize,
) usize {
    var count: usize = 0;
    var iterator = registry.tensors.iterator();
    while (iterator.next()) |entry| {
        for (selection.first_layer..last) |logical_index| {
            var prefix_buffer: [128]u8 = undefined;
            const prefix = std.fmt.bufPrint(
                &prefix_buffer,
                "language_model.model.layers.{d}.",
                .{logical_index},
            ) catch unreachable;
            if (std.mem.startsWith(u8, entry.key_ptr.*, prefix)) {
                count += 1;
                break;
            }
        }
    }
    return count;
}

fn plannedTensorCount(config: model.Config, selection: model.LayerSelection) usize {
    const last = selection.end(config) catch unreachable;
    var total: usize = 0;
    for (selection.first_layer..last) |logical_index| {
        total += switch (config.text_config.layerKind(logical_index)) {
            .kda_dense => 23,
            .kda_moe => 5404,
            .mla_moe => 5398,
        };
    }
    return total;
}

fn explain(writer: *std.Io.Writer, config: model.Config, selection: model.LayerSelection) !void {
    const last = try selection.end(config);
    try writer.print(
        "KIMI_K3_EXPLAIN model_type={s} logical_layers={} selected=[{}, {}) inference=disabled\n",
        .{ config.model_type, config.text_config.num_hidden_layers, selection.first_layer, last },
    );
    for (selection.first_layer..last) |logical_index| {
        const kind = config.text_config.layerKind(logical_index);
        const cache = switch (kind) {
            .kda_dense, .kda_moe => "kda(conv_qkv[12288,4],recurrent[96,128,128]f32)",
            .mla_moe => "mla(latent=512,extra_key=64,paged-later)",
        };
        const feed_forward = switch (kind) {
            .kda_dense => "dense_situ",
            .kda_moe, .mla_moe => "latent_moe_896_top16_mxfp4",
        };
        try writer.print(
            "layer={} kind={s} flow=attnres->{s}->{s}->attnres cache={s}\n",
            .{ logical_index, @tagName(kind), if (kind == .mla_moe) "mla" else "kda", feed_forward, cache },
        );
    }
}
