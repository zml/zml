const std = @import("std");

const zml = @import("zml");

const config = @import("config.zig");

const log = std.log.scoped(.minimax_h3);

/// Largest tensor-parallel degree that divides DiT heads (56), encoder heads (64), and GQA KV heads (8).
pub const tensor_parallel_max: usize = 8;

pub fn tensorParallelDegree(device_count: usize) usize {
    if (device_count == 0) return 0;
    if (device_count >= tensor_parallel_max) return tensor_parallel_max;
    if (device_count >= 4) return 4;
    if (device_count >= 2) return 2;
    return 1;
}

pub fn tensorParallelHeadsOk(degree: i64, dit_heads: i64, enc_heads: i64, kv_heads: i64) bool {
    if (degree <= 0) return false;
    return @rem(dit_heads, degree) == 0 and
        @rem(enc_heads, degree) == 0 and
        @rem(kv_heads, degree) == 0;
}

pub fn officialHeadsOk(degree: i64) bool {
    const dit = config.Config.official();
    const enc = config.EncoderConfig{};
    return tensorParallelHeadsOk(degree, dit.num_attention_heads, enc.num_attention_heads, enc.num_key_value_heads);
}

pub fn tensorParallelPrimaryAxis(target: zml.Target) zml.Sharding.PhysicalAxisTag {
    return switch (target) {
        .tpu => .link_x,
        .neuron => .link,
        .cuda, .rocm, .oneapi => .link,
        .cpu, .metal => .bus,
    };
}

pub fn presentShardableAxes(mesh: *const zml.Sharding.PhysicalMesh) zml.stdx.BoundedArray(zml.Sharding.PhysicalAxisTag, zml.Sharding.MAX_MESH_RANK) {
    var out: zml.stdx.BoundedArray(zml.Sharding.PhysicalAxisTag, zml.Sharding.MAX_MESH_RANK) = .empty;
    for (mesh.shardableAxes()) |tag| {
        if (mesh.hasAxis(tag)) out.appendAssumeCapacity(tag);
    }
    if (out.len == 0) {
        for (mesh.axisOrder().slice()) |tag| out.appendAssumeCapacity(tag);
    }
    return out;
}

/// Bind `.model` to the fastest present shardable axis and fold every other present axis into it.
pub fn tensorParallelStrategy(mesh: *const zml.Sharding.PhysicalMesh) error{InvalidPhysicalMesh}!zml.Sharding.Strategy {
    const axes = presentShardableAxes(mesh);
    if (axes.len == 0) return error.InvalidPhysicalMesh;
    var strategy: zml.Sharding.Strategy = .init;
    strategy.addBinding(.model, axes.get(0));
    if (axes.len > 1) strategy.addFold(axes.get(0), axes.constSlice());
    return strategy;
}

/// Use all devices when the count is a legal H3 TP degree. Larger power-of-two
/// meshes (16/32/64) keep the first 8 so head-parallel TP stays exact.
pub fn physicalMesh(
    allocator: std.mem.Allocator,
    target: zml.Target,
    devices: []const zml.platform.Device,
) anyerror!zml.Sharding.PhysicalMesh {
    if (devices.len == 0) return error.MissingDevices;
    const degree = tensorParallelDegree(devices.len);
    if (degree == 0 or degree > devices.len) return error.MissingDevices;
    if (degree == devices.len) {
        return zml.Sharding.PhysicalMesh.auto(allocator, target, devices);
    }
    log.warn(
        "H3 tensor parallel uses {d} of {d} devices (DiT 56 heads and encoder GQA 8 require degree 1, 2, 4, or 8)",
        .{ degree, devices.len },
    );
    return tensorParallelLine(allocator, target, devices[0..degree]);
}

fn tensorParallelLine(
    allocator: std.mem.Allocator,
    target: zml.Target,
    devices: []const zml.platform.Device,
) !zml.Sharding.PhysicalMesh {
    const nodes = try allocator.alloc(zml.Sharding.PhysicalNode, devices.len);
    errdefer allocator.free(nodes);
    for (nodes, devices) |*node, device| node.* = .device(device);
    const root: zml.Sharding.PhysicalNode = .{
        .branch = .{
            .tag = tensorParallelPrimaryAxis(target),
            .geometry = switch (target) {
                .tpu => .{ .mesh = .torus },
                .neuron, .cuda, .rocm, .oneapi => .point_to_point,
                .cpu, .metal => .tree,
            },
            .children = nodes,
        },
    };
    const mesh = try zml.Sharding.PhysicalMesh.fromTree(allocator, target, root);
    allocator.free(nodes);
    return mesh;
}

pub const Shardings = struct {
    model: zml.Sharding,

    pub fn init(platform: *zml.Platform) !Shardings {
        const strategy = try tensorParallelStrategy(&platform.physical_mesh);
        const model = try platform.registerShardingWithStrategy(
            "model",
            .mesh(.{ .model = .high_bandwidth }),
            strategy,
        );
        const degree = model.numPartitionsForLogicalAxis(.model);
        if (!officialHeadsOk(degree)) {
            log.err(
                "H3 tensor parallel degree {d} does not divide DiT heads 56, encoder heads 64, and GQA KV heads 8. Use 1, 2, 4, or 8 devices.",
                .{degree},
            );
            return error.IncompatibleSharding;
        }
        return .{ .model = model };
    }

    pub fn all(self: Shardings) [1]zml.Sharding {
        return .{self.model};
    }

    pub fn checkLoaded(self: Shardings, dit_cfg: config.Config, enc_cfg: config.EncoderConfig) !void {
        const degree = self.model.numPartitionsForLogicalAxis(.model);
        if (!tensorParallelHeadsOk(degree, dit_cfg.num_attention_heads, enc_cfg.num_attention_heads, enc_cfg.num_key_value_heads)) {
            log.err(
                "Loaded heads dit={d} encoder={d} kv={d} do not divide by tensor-parallel degree {d}",
                .{ dit_cfg.num_attention_heads, enc_cfg.num_attention_heads, enc_cfg.num_key_value_heads, degree },
            );
            return error.IncompatibleSharding;
        }
    }
};
