const std = @import("std");

const zml = @import("zml");

pub const SessionOptions = struct {
    seqlen: u32,
    backend: zml.attention.Backend,
};

pub const GenerationOptions = struct {
    sampling_strategy: zml.nn.SamplingStrategy = .{},
};

pub const Phase = enum {
    prefill,
    decode,

    pub fn isPrefill(self: Phase) bool {
        return self == .prefill;
    }

    pub fn label(self: Phase) []const u8 {
        return @tagName(self);
    }

    pub fn startMessage(self: Phase, comptime component: []const u8) []const u8 {
        return switch (self) {
            .prefill => "Compiling prefill " ++ component ++ "...",
            .decode => "Compiling decode " ++ component ++ "...",
        };
    }

    pub fn programName(self: Phase, comptime model_name: []const u8, comptime component: []const u8) []const u8 {
        return switch (self) {
            .prefill => "llm_" ++ model_name ++ "_prefill_" ++ component,
            .decode => "llm_" ++ model_name ++ "_decode_" ++ component,
        };
    }

    pub fn logCompileDone(self: Phase, logger: anytype, comptime component: []const u8, io: std.Io, from: std.Io.Timestamp) void {
        logger.info("Compiled {s} " ++ component ++ " [{f}]", .{ self.label(), from.untilNow(io, .awake) });
    }
};

pub const Meshes = struct {
    model: *const zml.Mesh,
    experts: *const zml.Mesh,

    pub fn init(platform: *zml.Platform) !Meshes {
        switch (platform.target) {
            .tpu => {
                const has_link_y = platform.physical_mesh.hasAxis(.link_y);

                var strategy_experts: zml.Sharding.Strategy = .parseBindings(.{ .experts = .link_x });
                var strategy_model: zml.Sharding.Strategy = .parseBindings(.{ .model = .link_x });
                if (has_link_y) {
                    strategy_experts.addFold(.link_x, &.{ .link_x, .link_y });
                    strategy_model.addFold(.link_x, &.{ .link_x, .link_y });
                }

                return .{
                    .model = try platform.registerMeshWithStrategy("model", .mesh(.{ .model = .high_bandwidth }), strategy_model),
                    .experts = try platform.registerMeshWithStrategy("experts", .mesh(.{ .experts = .high_bandwidth }), strategy_experts),
                };
            },
            .tenstorrent => {
                // Tensor parallelism spans every axis of the view, as on TPU,
                // instead of the first axis with replicas along the others.
                var strategy_model: zml.Sharding.Strategy = .parseBindings(.{ .model = .link_x });
                var strategy_experts: zml.Sharding.Strategy = .parseBindings(.{ .experts = .link_x });
                var axes: zml.stdx.BoundedArray(zml.Sharding.PhysicalAxisTag, 3) = .empty;
                for ([_]zml.Sharding.PhysicalAxisTag{ .link_x, .link_y, .link_z }) |tag| {
                    if (platform.physical_mesh.hasAxis(tag)) axes.appendAssumeCapacity(tag);
                }
                if (axes.len > 1) {
                    strategy_model.addFold(.link_x, axes.constSlice());
                    strategy_experts.addFold(.link_x, axes.constSlice());
                }

                return .{
                    .model = try platform.registerMeshWithStrategy("model", .mesh(.{ .model = .high_bandwidth }), strategy_model),
                    .experts = try platform.registerMeshWithStrategy("experts", .mesh(.{ .experts = .high_bandwidth }), strategy_experts),
                };
            },
            .cuda, .rocm, .oneapi, .neuron, .metal, .cpu, .furiosa => return .{
                .model = try platform.registerMesh("model", .mesh(.{ .model = .high_bandwidth })),
                .experts = try platform.registerMesh("experts", .mesh(.{ .experts = .high_bandwidth })),
            },
        }
    }

    pub fn all(self: Meshes) [2]*const zml.Mesh {
        return .{ self.model, self.experts };
    }
};

pub fn parseConfig(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(T) {
    const file = try dir.openFile(io, "config.json", .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    return try std.json.parseFromTokenSource(T, allocator, &reader, .{ .ignore_unknown_fields = true });
}

/// Uploads every tensor of `loaded.inner` (a family's `LoadedModel`) into a
/// fresh `zml.Bufferized` of it, each tensor in the device layout of
/// `layouts` (one per tensor, `meta.visit` order: those the programs the
/// buffers feed report, `zml.Exe.commonModelLayouts`), or in the platform's
/// default layouts when null. Shared by every family, so none needs its own
/// wiring for the layouts its programs take the weights in.
pub fn loadBuffers(
    loaded: anytype,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    layouts: ?[]const zml.pjrt.MemoryLayout,
) !zml.Bufferized(@TypeOf(loaded.inner)) {
    const Model = @TypeOf(loaded.inner);
    progress.increaseEstimatedTotalItems(store.view().count());
    const now: std.Io.Timestamp = .now(io, .awake);

    var buffers = try zml.mem.bufferize(allocator, Model, &loaded.inner);
    errdefer loaded.unloadBuffers(&buffers, allocator);

    var loader: zml.io.Loader = try .init(allocator, platform, .{
        .dma_chunks = 32,
        .dma_chunk_size = 256 * zml.MiB,
        .parallelism = 16,
    });
    defer loader.deinit();

    try loader.load(io, Model, &loaded.inner, &buffers, store, .{ .progress = progress, .layouts = layouts });
    try loader.await(io);

    const took = now.untilNow(io, .awake);
    const total_bytes: u64 = loader.bytes_loaded.raw;
    const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
    std.log.scoped(.llm).info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });

    return buffers;
}
