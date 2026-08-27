const std = @import("std");

const zml = @import("zml");

const config = @import("config.zig");

const log = std.log.scoped(.minimax_h3);

fn gcdU64(a: u64, b: u64) u64 {
    var x = a;
    var y = b;
    while (y != 0) {
        const t = x % y;
        x = y;
        y = t;
    }
    return x;
}

fn gcdPos(a: i64, b: i64) u64 {
    return gcdU64(@intCast(@max(a, 0)), @intCast(@max(b, 0)));
}

/// Largest even head-split: every attention head count must divide by the degree.
pub fn tensorParallelMax(dit_heads: i64, enc_heads: i64, kv_heads: i64) usize {
    const g = gcdU64(gcdPos(dit_heads, enc_heads), @intCast(@max(kv_heads, 0)));
    return if (g == 0) 1 else @intCast(g);
}

pub fn tensorParallelHeadsOk(degree: i64, dit_heads: i64, enc_heads: i64, kv_heads: i64) bool {
    if (degree <= 0) return false;
    return @rem(dit_heads, degree) == 0 and
        @rem(enc_heads, degree) == 0 and
        @rem(kv_heads, degree) == 0;
}

/// Largest `d ≤ device_count` that splits DiT heads, encoder heads, and KV heads evenly.
/// Leftover-fit / replication is slower than dropping idle ranks (3 cards → TP=2).
pub fn tensorParallelDegreeFor(device_count: usize, dit_heads: i64, enc_heads: i64, kv_heads: i64) usize {
    if (device_count == 0) return 0;
    var d = @min(device_count, tensorParallelMax(dit_heads, enc_heads, kv_heads));
    while (d > 1) : (d -= 1) {
        if (tensorParallelHeadsOk(@intCast(d), dit_heads, enc_heads, kv_heads)) return d;
    }
    return 1;
}

pub fn officialHeadCounts() struct { dit: i64, enc: i64, kv: i64 } {
    const dit = config.Config.official();
    const enc = config.EncoderConfig{};
    return .{
        .dit = dit.num_attention_heads,
        .enc = enc.num_attention_heads,
        .kv = enc.num_key_value_heads,
    };
}

pub fn tensorParallelDegree(device_count: usize) usize {
    const h = officialHeadCounts();
    return tensorParallelDegreeFor(device_count, h.dit, h.enc, h.kv);
}

pub fn officialHeadsOk(degree: i64) bool {
    const h = officialHeadCounts();
    return tensorParallelHeadsOk(degree, h.dit, h.enc, h.kv);
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

pub fn physicalMesh(
    allocator: std.mem.Allocator,
    target: zml.Target,
    devices: []const zml.platform.Device,
) anyerror!zml.Sharding.PhysicalMesh {
    if (devices.len == 0) return error.MissingDevices;
    const h = officialHeadCounts();
    const degree = tensorParallelDegreeFor(devices.len, h.dit, h.enc, h.kv);
    if (degree == 0 or degree > devices.len) return error.MissingDevices;
    if (degree == devices.len) {
        return zml.Sharding.PhysicalMesh.auto(allocator, target, devices);
    }
    log.warn(
        "tensor parallel uses {d} of {d} devices (largest even split of dit={d} enc={d} kv={d} heads)",
        .{ degree, devices.len, h.dit, h.enc, h.kv },
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
        const h = officialHeadCounts();
        if (!tensorParallelHeadsOk(degree, h.dit, h.enc, h.kv)) {
            log.err(
                "tensor parallel degree {d} does not divide dit={d} enc={d} kv={d} heads",
                .{ degree, h.dit, h.enc, h.kv },
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
