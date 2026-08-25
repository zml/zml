const std = @import("std");

const zml = @import("zml");
const common = @import("common.zig");
const inference = @import("kimi_k3/inference.zig");
const model = @import("kimi_k3/model.zig");
const runtime_weights = @import("kimi_k3/runtime_weights.zig");
const session_impl = @import("kimi_k3/session.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    weights: []const u8,
    tokenizer: []const u8,
    token_count: usize = 4,
    repeats: usize = 2,
    decode_one: bool = false,
    layer_limit: usize = 4,
    compile_only: bool = false,
    resident: bool = false,
    distributed: bool = false,
    partition_experts: bool = false,
    force_eos_after_prefill: bool = false,
    cache_dump_prefill: []const u8 = "",
    cache_dump_decode: []const u8 = "",

    pub const help =
        \\Use kimi_k3_session_tests --weights=<S4-directory> --tokenizer=<tokenizer.json> [options]
        \\
        \\Run the fixed Moonshot prefix through the reusable NVIDIA CUDA session.
        \\
        \\Options:
        \\  --token-count=<1..4>  Prefix tokens to execute (default: 4)
        \\  --repeats=<count>     Reset-and-repeat count (default: 2)
        \\  --decode-one          Stream exactly one generated continuation
        \\  --layer-limit=<count> Selected prefix depth (default: 4)
        \\  --compile-only        Compile selected families without loading weights
        \\  --resident            Keep every selected prefix layer resident across tokens
        \\  --distributed         Require physical TP4 or TP8 on four/eight GPUs
        \\  --partition-experts   Test-only shared-axis TP+EP expert placement
        \\  --force-eos-after-prefill  Test-only exercise of the EOS stop branch
        \\  --cache-dump-prefill=<path>  Test-only raw cache output after prefill
        \\  --cache-dump-decode=<path>   Test-only raw cache output after continuation
        \\
    ;
};

const official_prefix = [_]u32{ 1, 42, 32000, 160000 };
const official_prefix4_greedy: u32 = 95385;
const example_resident_layer_loads: usize = 46;
// 35 KDA+MoE layers x 5,404 reads plus 11 MLA+MoE layers x 5,398 reads.
const example_resident_payload_reads: u64 = 248_518;
const example_resident_payload_bytes: u64 = 776_886_773_760;
const full_resident_layer_loads: usize = 92;
// 68 KDA+MoE layers x 5,404 reads plus 24 MLA+MoE layers x 5,398 reads.
const full_resident_payload_reads: u64 = 497_024;
const full_resident_payload_bytes: u64 = 1_552_926_730_240;
const example_expert_hbm_per_rank: u64 = 180_807_008_256;

fn elapsedUs(io: std.Io, started: i96) i96 {
    return @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - started, 1000);
}

fn initSelectedModel(
    allocator: std.mem.Allocator,
    io: std.Io,
    repo: std.Io.Dir,
    store: zml.io.TensorStore.View,
    layer_limit: usize,
    compile_only: bool,
    expert_placement: model.ExpertPlacement,
) !model.LoadedModel {
    const parsed = try common.parseConfig(model.Config, allocator, io, repo);
    errdefer parsed.deinit();
    const selection: model.LayerSelection = .{ .layer_limit = layer_limit };
    const inner = if (compile_only)
        try model.Model.initCompileOnly(allocator, store, parsed.value, selection)
    else
        try model.Model.initSelected(allocator, store, parsed.value, .{
            .max_seq_len = parsed.value.text_config.max_position_embeddings,
        }, selection);
    return .{ .inner = inner, .parsed_config = parsed, .expert_placement = expert_placement, .repo = repo };
}

fn hashBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    hasher: *std.crypto.hash.sha2.Sha256,
    buffer: zml.Buffer,
) !void {
    var host = try buffer.toSliceAlloc(allocator, io);
    defer host.free(allocator);
    hasher.update(host.constData());
}

fn sessionCacheDigest(session: *const session_impl.Session) ![64]u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(std.mem.asBytes(&session.position));
    for (session.kda_caches) |cache| {
        try hashBuffer(session.allocator, session.io, &hasher, cache.q_conv);
        try hashBuffer(session.allocator, session.io, &hasher, cache.k_conv);
        try hashBuffer(session.allocator, session.io, &hasher, cache.v_conv);
        try hashBuffer(session.allocator, session.io, &hasher, cache.recurrent_state);
    }
    for (session.mla_caches) |cache| {
        try hashBuffer(session.allocator, session.io, &hasher, cache.compressed);
        try hashBuffer(session.allocator, session.io, &hasher, cache.extra_key);
    }
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

fn dumpSessionCache(session: *const session_impl.Session, path: []const u8) !void {
    const file = try std.Io.Dir.createFile(.cwd(), session.io, path, .{});
    defer file.close(session.io);
    var write_buffer: [64 * 1024]u8 = undefined;
    var writer = file.writer(session.io, &write_buffer);
    for (session.kda_caches) |cache| {
        inline for (.{ cache.q_conv, cache.k_conv, cache.v_conv, cache.recurrent_state }) |buffer| {
            var host = try buffer.toSliceAlloc(session.allocator, session.io);
            defer host.free(session.allocator);
            try writer.interface.writeAll(host.bytes);
        }
    }
    for (session.mla_caches) |cache| {
        inline for (.{ cache.compressed, cache.extra_key }) |buffer| {
            var host = try buffer.toSliceAlloc(session.allocator, session.io);
            defer host.free(session.allocator);
            try writer.interface.writeAll(host.bytes);
        }
    }
    try writer.interface.flush();
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    if (args.token_count == 0 or args.token_count > official_prefix.len) return error.InvalidTokenCount;
    if (args.repeats == 0) return error.InvalidRepeatCount;
    if (args.decode_one and args.repeats != 1) return error.DecodeGateRequiresOneRepeat;
    if (args.force_eos_after_prefill and !args.decode_one) return error.ForceEosRequiresDecodeGate;
    if (args.layer_limit == 0 or args.layer_limit > 93) return error.InvalidLayerLimit;
    if (args.resident and args.compile_only) return error.InvalidResidentSessionMode;
    if (args.resident and args.layer_limit > 4 and !args.distributed) return error.ExtendedResidentSessionRequiresDistributedDevices;
    if (args.partition_experts and !args.distributed) return error.PartitionedExpertsRequireDistributedDevices;

    const platform: *zml.Platform = try .init(allocator, io, .cuda, .{
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false, .memory_fraction = 0.90 } } },
    });
    defer platform.deinit(allocator, io);
    if (platform.target != .cuda) return error.NvidiaCudaRequired;
    if (args.distributed and !runtime_weights.ExpertPlacement.isSupportedSharedAxisDeviceCount(platform.devices.len))
        return error.KimiK3DistributedSessionRequiresFourOrEightDevices;
    if (!args.distributed and platform.devices.len != 1) return error.KimiK3Gpu0SessionRequiresOneDevice;
    const layout: []const u8 = if (args.partition_experts)
        if (platform.devices.len == 4) "tp4_ep4_shared_axis" else "tp8_ep8_shared_axis"
    else if (args.distributed)
        if (platform.devices.len == 4) "tp4_ep1" else "tp8_ep1"
    else
        "gpu0";

    const repo = try zml.safetensors.resolveModelRepo(io, args.weights);
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var loaded_model = try initSelectedModel(
        allocator,
        io,
        repo,
        store.view(),
        args.layer_limit,
        args.compile_only,
        if (args.partition_experts) .shared_axis else .replicated,
    );
    defer loaded_model.deinit(allocator);
    const shardings: common.Shardings = try .init(platform);
    var progress = std.Progress.start(io, .{ .root_name = "Kimi K3 session gate" });
    defer progress.end();

    const seqlen = if (args.compile_only) 1 else args.token_count + @intFromBool(args.decode_one);
    const compile_started = std.Io.Clock.now(.real, io).toNanoseconds();
    var compiled: inference.CompiledModel = try loaded_model.compile(
        allocator,
        io,
        platform,
        .vanilla,
        shardings,
        seqlen,
        &progress,
    );
    defer compiled.deinit();
    const compile_us = elapsedUs(io, compile_started);

    if (args.compile_only) {
        const expected_sources = std.math.divCeil(usize, args.layer_limit, 12) catch unreachable;
        if (compiled.params.source_slots != expected_sources) return error.KimiK3SourceSlotMismatch;
        if (args.layer_limit > 12 and compiled.kda_moe_boundary == null) {
            return error.MissingKdaMoeBoundaryExecutable;
        }
        var stdout_file = std.Io.File.stdout().writerStreaming(io, &.{});
        try stdout_file.interface.print(
            "KIMI_K3_SESSION_FULL_COMPILE_PASS layers={} source_slots={} " ++ "kda_boundary={} mla_boundary={} compile_us={} backend=cuda devices={} layout={s}\n",
            .{
                args.layer_limit,
                compiled.params.source_slots,
                compiled.kda_moe_boundary != null,
                compiled.mla_moe_boundary != null,
                compile_us,
                platform.devices.len,
                layout,
            },
        );
        try stdout_file.interface.flush();
        return;
    }

    var buffers = if (args.resident)
        try loaded_model.loadResidentBuffers(allocator, io, platform, &store, &progress, shardings)
    else
        try loaded_model.loadStreamingBuffers(allocator, io, platform, &store, &progress, shardings);
    defer loaded_model.unloadBuffers(&buffers, allocator);
    var tokenizer = try zml.tokenizer.Tokenizer.fromFile(allocator, io, args.tokenizer);
    defer tokenizer.deinit();
    var session = try session_impl.Session.init(allocator, io, platform, tokenizer, &compiled, &buffers);
    const resident_load_stats = buffers.load_stats.*;
    defer session.deinit();
    const canonical_resident = args.resident and args.partition_experts and
        ((args.layer_limit == model.example_resident_layer_count and platform.devices.len == 4) or
            (args.layer_limit == model.full_model_layer_count and platform.devices.len == 8));
    if (canonical_resident) {
        const expected_loads = if (platform.devices.len == 4)
            example_resident_layer_loads
        else
            full_resident_layer_loads;
        const expected_reads = if (platform.devices.len == 4)
            example_resident_payload_reads
        else
            full_resident_payload_reads;
        const expected_bytes = if (platform.devices.len == 4)
            example_resident_payload_bytes
        else
            full_resident_payload_bytes;
        if (@divExact(runtime_weights.expert_component_bytes.device_bank, platform.devices.len) *
            expected_loads != example_expert_hbm_per_rank)
        {
            return error.KimiK3ResidentExpertHbmAccountingMismatch;
        }
        if (resident_load_stats.layer_loads != expected_loads or
            resident_load_stats.payload_reads != expected_reads or
            resident_load_stats.payload_bytes != expected_bytes)
        {
            return error.KimiK3ResidentLoadAccountingMismatch;
        }
        if (resident_load_stats.packed_expert_layers +
            resident_load_stats.fallback_expert_layers != expected_loads)
        {
            return error.KimiK3ResidentExpertLoadModeAccountingMismatch;
        }
        if (resident_load_stats.packed_expert_layers == expected_loads and
            resident_load_stats.expert_physical_read_extents != expected_loads * 24)
        {
            return error.KimiK3PackedExpertExtentAccountingMismatch;
        }

        for (platform.devices) |device| {
            const memory = device.memoryStats();
            if (memory.bytes_limit) |limit| {
                if (memory.bytes_in_use >= limit) return error.KimiK3ResidentAllocatorLimitExceeded;
            }
        }
    }

    var first_greedy: ?u32 = null;
    var first_cache_digest: ?[64]u8 = null;
    var stdout_file = std.Io.File.stdout().writerStreaming(io, &.{});
    for (0..args.repeats) |repeat| {
        const started = std.Io.Clock.now(.real, io).toNanoseconds();
        try session.runPrefill(official_prefix[0..args.token_count]);
        if (args.resident and !std.meta.eql(resident_load_stats, buffers.load_stats.*)) return error.KimiK3ResidentWeightsReloaded;
        const greedy = session.last_generated_token;
        const cache_digest = try sessionCacheDigest(&session);
        if (repeat == 0 and args.cache_dump_prefill.len != 0) {
            try dumpSessionCache(&session, args.cache_dump_prefill);
        }
        if (first_cache_digest) |expected| {
            if (!std.mem.eql(u8, &expected, &cache_digest)) return error.KimiK3SessionCacheResetMismatch;
        } else {
            first_cache_digest = cache_digest;
        }
        if (first_greedy) |expected| {
            if (greedy != expected) return error.KimiK3SessionResetMismatch;
        } else {
            first_greedy = greedy;
        }
        if (args.layer_limit == 4 and args.token_count == official_prefix.len and greedy != official_prefix4_greedy) {
            return error.KimiK3OfficialGreedyMismatch;
        }
        try stdout_file.interface.print(
            "KIMI_K3_SESSION_PASS repeat={} tokens={} greedy={} compile_us={} session_us={} " ++
                "backend=cuda weights={s} cache_sha256={s} devices={} layout={s}\n",
            .{
                repeat,
                args.token_count,
                greedy,
                compile_us,
                elapsedUs(io, started),
                if (args.resident) "resident" else "streaming",
                &cache_digest,
                platform.devices.len,
                layout,
            },
        );
        try stdout_file.interface.flush();
        if (args.decode_one) {
            var history = try std.ArrayList(u32).initCapacity(allocator, seqlen);
            defer history.deinit(allocator);
            try history.appendSlice(allocator, official_prefix[0..args.token_count]);
            if (args.force_eos_after_prefill) {
                session.last_generated_token = session.tokenizer.tokenId("<|end_of_msg|>") orelse return error.KimiK3MissingEosToken;
            }
            const streamed = session.last_generated_token;
            try session.runDecode(&history, &stdout_file.interface);
            if (args.resident and !std.meta.eql(resident_load_stats, buffers.load_stats.*)) return error.KimiK3ResidentWeightsReloaded;
            if (args.force_eos_after_prefill) {
                if (history.items.len != args.token_count or session.last_generated_token != streamed) {
                    return error.KimiK3ForcedEosStopMismatch;
                }
            } else if (history.items.len != seqlen or history.items[seqlen - 1] != greedy) {
                return error.KimiK3DecodeHistoryMismatch;
            }
            if (args.cache_dump_decode.len != 0) {
                try dumpSessionCache(&session, args.cache_dump_decode);
            }
            try stdout_file.interface.print(
                "\nKIMI_K3_SESSION_DECODE_PASS streamed={} next={} history_tokens={} capacity={} " ++
                    "cache_sha256={s} forced_eos={}\n",
                .{
                    streamed,
                    session.last_generated_token,
                    history.items.len,
                    seqlen,
                    &(try sessionCacheDigest(&session)),
                    args.force_eos_after_prefill,
                },
            );
            try stdout_file.interface.flush();
        }
    }
    for (platform.devices, 0..) |device, rank| {
        const memory = device.memoryStats();
        try stdout_file.interface.print(
            "KIMI_K3_SESSION_MEMORY rank={} bytes_in_use={} peak_bytes_in_use={} bytes_limit={}\n",
            .{
                rank,
                memory.bytes_in_use,
                memory.peak_bytes_in_use orelse 0,
                memory.bytes_limit orelse 0,
            },
        );
    }
    const device_memory = platform.devices[0].memoryStats();
    try stdout_file.interface.print(
        "KIMI_K3_SESSION_ALL_PASS reset_deterministic=true official_prefix_checked={} " ++
            "weights={s} layer_loads={} payload_reads={} payload_bytes={} devices={} layout={s} " ++
            "device_bytes_in_use={} device_peak_bytes_in_use={} device_bytes_limit={}\n",
        .{
            args.layer_limit == 4 and args.token_count == official_prefix.len,
            if (args.resident) "resident" else "streaming",
            buffers.load_stats.layer_loads,
            buffers.load_stats.payload_reads,
            buffers.load_stats.payload_bytes,
            platform.devices.len,
            layout,
            device_memory.bytes_in_use,
            device_memory.peak_bytes_in_use orelse 0,
            device_memory.bytes_limit orelse 0,
        },
    );
    try stdout_file.interface.flush();
}
