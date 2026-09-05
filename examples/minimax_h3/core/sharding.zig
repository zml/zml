const std = @import("std");

const zml = @import("zml");

const config = @import("config.zig");

const log = std.log.scoped(.minimax_h3);

pub const HeadCounts = struct {
    dit: i64,
    enc: i64,
    kv: i64,
};

/// `Platform.CreatePhysicalMeshFn` has no context. Set immediately before `Platform.auto`.
var pending_physical_heads: ?HeadCounts = null;

pub fn preparePhysicalMesh(heads: HeadCounts) void {
    pending_physical_heads = heads;
}

fn gcd3(a: i64, b: i64, c: i64) u64 {
    if (a <= 0 or b <= 0 or c <= 0) return 1;
    return std.math.gcd(
        std.math.gcd(@as(u64, @intCast(a)), @as(u64, @intCast(b))),
        @as(u64, @intCast(c)),
    );
}

/// Largest even head-split: every attention head count must divide by the degree.
pub fn tensorParallelMax(dit_heads: i64, enc_heads: i64, kv_heads: i64) usize {
    const g = gcd3(dit_heads, enc_heads, kv_heads);
    return if (g == 0) 1 else @intCast(g);
}

pub fn tensorParallelHeadsOk(degree: i64, dit_heads: i64, enc_heads: i64, kv_heads: i64) bool {
    if (degree <= 0) return false;
    return @rem(dit_heads, degree) == 0 and
        @rem(enc_heads, degree) == 0 and
        @rem(kv_heads, degree) == 0;
}

/// Largest `d ≤ device_count` that splits DiT heads, encoder heads, and KV heads evenly.
/// Replica on leftover ranks is slower than an even split on fewer devices.
pub fn tensorParallelDegreeFor(device_count: usize, dit_heads: i64, enc_heads: i64, kv_heads: i64) usize {
    if (device_count == 0) return 0;
    const g = tensorParallelMax(dit_heads, enc_heads, kv_heads);
    var d = @min(device_count, g);
    while (d > 1) : (d -= 1) {
        if (g % d == 0) return d;
    }
    return 1;
}

pub fn officialHeadCounts() HeadCounts {
    const dit = config.Config.official();
    const enc = config.EncoderConfig.official();
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

/// Bind `.model` to the fastest axis. Fold extra axes only while the product still even-splits heads.
pub fn tensorParallelStrategyFor(
    mesh: *const zml.Sharding.PhysicalMesh,
    dit_heads: i64,
    enc_heads: i64,
    kv_heads: i64,
) error{ InvalidPhysicalMesh, IncompatibleSharding }!zml.Sharding.Strategy {
    const axes = presentShardableAxes(mesh);
    if (axes.len == 0) return error.InvalidPhysicalMesh;
    const primary = axes.get(0);
    var degree: i64 = mesh.axis(primary);
    if (degree <= 0) degree = 1;
    if (!tensorParallelHeadsOk(degree, dit_heads, enc_heads, kv_heads)) return error.IncompatibleSharding;
    var strategy: zml.Sharding.Strategy = .init;
    strategy.addBinding(.model, primary);
    var folded: zml.stdx.BoundedArray(zml.Sharding.PhysicalAxisTag, zml.Sharding.MAX_MESH_RANK) = .empty;
    folded.appendAssumeCapacity(primary);
    var i: usize = 1;
    while (i < axes.len) : (i += 1) {
        const tag = axes.get(i);
        const next = degree * mesh.axis(tag);
        if (!tensorParallelHeadsOk(next, dit_heads, enc_heads, kv_heads)) break;
        folded.appendAssumeCapacity(tag);
        degree = next;
    }
    if (folded.len > 1) strategy.addFold(primary, folded.constSlice());
    return strategy;
}

pub fn tensorParallelStrategy(mesh: *const zml.Sharding.PhysicalMesh) error{ InvalidPhysicalMesh, IncompatibleSharding }!zml.Sharding.Strategy {
    const h = officialHeadCounts();
    return tensorParallelStrategyFor(mesh, h.dit, h.enc, h.kv);
}

fn physicalMeshFor(
    allocator: std.mem.Allocator,
    target: zml.Target,
    devices: []const zml.platform.Device,
    heads: HeadCounts,
) anyerror!zml.Sharding.PhysicalMesh {
    if (devices.len == 0) return error.MissingDevices;
    const degree = tensorParallelDegreeFor(devices.len, heads.dit, heads.enc, heads.kv);
    if (degree == 0 or degree > devices.len) return error.MissingDevices;
    if (degree < devices.len) {
        log.warn(
            "tensor parallel uses {d} of {d} devices (largest even split of dit={d} enc={d} kv={d} heads)",
            .{ degree, devices.len, heads.dit, heads.enc, heads.kv },
        );
    }
    return zml.Sharding.PhysicalMesh.auto(allocator, target, devices[0..degree]);
}

pub fn physicalMesh(
    allocator: std.mem.Allocator,
    target: zml.Target,
    devices: []const zml.platform.Device,
) anyerror!zml.Sharding.PhysicalMesh {
    return physicalMeshFor(allocator, target, devices, pending_physical_heads orelse officialHeadCounts());
}

pub const Shardings = struct {
    model: zml.Sharding,

    pub fn init(platform: *zml.Platform, heads: HeadCounts) !Shardings {
        const strategy = try tensorParallelStrategyFor(&platform.physical_mesh, heads.dit, heads.enc, heads.kv);
        const model = try platform.registerShardingWithStrategy(
            "model",
            .mesh(.{ .model = .high_bandwidth }),
            strategy,
        );
        const degree = model.numPartitionsForLogicalAxis(.model);
        if (!tensorParallelHeadsOk(degree, heads.dit, heads.enc, heads.kv)) {
            log.err(
                "tensor parallel degree {d} does not divide dit={d} enc={d} kv={d} heads",
                .{ degree, heads.dit, heads.enc, heads.kv },
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
