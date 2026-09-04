const std = @import("std");

const zml = @import("zml");

const config = @import("config.zig");

const log = std.log.scoped(.minimax_h3);

// =============================================================================
// recipe/shard.zig — one-axis Megatron TP for N GPUs
//
// Degree = max { d : d | head_gcd, d ≤ GPU count }. Official gcd is 8
// (H3 56, encoder 64, KV 8, LTX 32, Gemma 16, Gemma-KV 8) → 1/2/4/8.
// Sol-Attn gathers to 32 heads (kernel is not local-head). Embed, VAE,
// TAE, handoff, and Euler stay replicated.
// =============================================================================

pub const ltx_heads: i64 = 32;
pub const gemma_heads: i64 = 16;
pub const gemma_kv_heads: i64 = 8;

pub const HeadCounts = struct {
    dit: i64,
    enc: i64,
    kv: i64,
    ltx: i64 = ltx_heads,
    gemma: i64 = gemma_heads,
    gemma_kv: i64 = gemma_kv_heads,

    pub fn values(self: HeadCounts) [6]i64 {
        return .{ self.dit, self.enc, self.kv, self.ltx, self.gemma, self.gemma_kv };
    }
};

/// `Platform.CreatePhysicalMeshFn` has no context. Set immediately before `Platform.auto`.
var pending_physical_heads: ?HeadCounts = null;
var pending_device_cap: ?usize = null;

pub fn preparePhysicalMesh(heads: HeadCounts) void {
    pending_physical_heads = heads;
}

pub fn prepareDeviceCap(n: usize) void {
    pending_device_cap = if (n == 0) null else n;
}

fn gcdN(values: []const i64) u64 {
    var g: u64 = 0;
    for (values) |v| {
        if (v <= 0) continue;
        const u = @as(u64, @intCast(v));
        g = if (g == 0) u else std.math.gcd(g, u);
    }
    return if (g == 0) 1 else g;
}

/// Largest even head-split: every attention head count must divide by the degree.
pub fn tensorParallelMax(dit_heads: i64, enc_heads: i64, kv_heads: i64) usize {
    if (dit_heads <= 0 or enc_heads <= 0 or kv_heads <= 0) return 1;
    return @intCast(gcdN(&.{ dit_heads, enc_heads, kv_heads, ltx_heads, gemma_heads, gemma_kv_heads }));
}

pub fn tensorParallelMaxAll(h: HeadCounts) usize {
    return @intCast(gcdN(&h.values()));
}

pub fn tensorParallelHeadsOk(degree: i64, dit_heads: i64, enc_heads: i64, kv_heads: i64) bool {
    return tensorParallelHeadsOkAll(degree, .{
        .dit = dit_heads,
        .enc = enc_heads,
        .kv = kv_heads,
    });
}

pub fn tensorParallelHeadsOkAll(degree: i64, h: HeadCounts) bool {
    if (degree <= 0) return false;
    for (h.values()) |n| {
        if (@rem(n, degree) != 0) return false;
    }
    return true;
}

/// Largest `d ≤ device_count` that splits all head counts evenly.
pub fn tensorParallelDegreeFor(device_count: usize, dit_heads: i64, enc_heads: i64, kv_heads: i64) usize {
    return tensorParallelDegreeForAll(device_count, .{
        .dit = dit_heads,
        .enc = enc_heads,
        .kv = kv_heads,
    });
}

pub fn tensorParallelDegreeForAll(device_count: usize, h: HeadCounts) usize {
    if (device_count == 0) return 0;
    const g = tensorParallelMaxAll(h);
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
    return tensorParallelDegreeForAll(device_count, officialHeadCounts());
}

pub fn officialHeadsOk(degree: i64) bool {
    return tensorParallelHeadsOkAll(degree, officialHeadCounts());
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
    return tensorParallelStrategyForAll(mesh, .{
        .dit = dit_heads,
        .enc = enc_heads,
        .kv = kv_heads,
    });
}

pub fn tensorParallelStrategyForAll(
    mesh: *const zml.Sharding.PhysicalMesh,
    h: HeadCounts,
) error{ InvalidPhysicalMesh, IncompatibleSharding }!zml.Sharding.Strategy {
    const axes = presentShardableAxes(mesh);
    if (axes.len == 0) return error.InvalidPhysicalMesh;
    const primary = axes.get(0);
    var degree: i64 = mesh.axis(primary);
    if (degree <= 0) degree = 1;
    if (!tensorParallelHeadsOkAll(degree, h)) return error.IncompatibleSharding;
    var strategy: zml.Sharding.Strategy = .init;
    strategy.addBinding(.model, primary);
    var folded: zml.stdx.BoundedArray(zml.Sharding.PhysicalAxisTag, zml.Sharding.MAX_MESH_RANK) = .empty;
    folded.appendAssumeCapacity(primary);
    var i: usize = 1;
    while (i < axes.len) : (i += 1) {
        const tag = axes.get(i);
        const next = degree * mesh.axis(tag);
        if (!tensorParallelHeadsOkAll(next, h)) break;
        folded.appendAssumeCapacity(tag);
        degree = next;
    }
    if (folded.len > 1) strategy.addFold(primary, folded.constSlice());
    return strategy;
}

pub fn tensorParallelStrategy(mesh: *const zml.Sharding.PhysicalMesh) error{ InvalidPhysicalMesh, IncompatibleSharding }!zml.Sharding.Strategy {
    return tensorParallelStrategyForAll(mesh, officialHeadCounts());
}

fn physicalMeshFor(
    allocator: std.mem.Allocator,
    target: zml.Target,
    devices: []const zml.platform.Device,
    heads: HeadCounts,
) anyerror!zml.Sharding.PhysicalMesh {
    if (devices.len == 0) return error.MissingDevices;
    const cap = pending_device_cap orelse devices.len;
    const used = devices[0..@min(cap, devices.len)];
    const degree = tensorParallelDegreeForAll(used.len, heads);
    if (degree == 0 or degree > used.len) return error.MissingDevices;
    if (degree < used.len) {
        const max = tensorParallelMaxAll(heads);
        log.warn(
            "tensor parallel uses {d} of {d} devices (head gcd={d}; leftover GPUs idle)",
            .{ degree, used.len, max },
        );
    } else {
        log.info("tensor parallel degree={d} on {d} GPUs (head gcd={d})", .{
            degree,
            used.len,
            tensorParallelMaxAll(heads),
        });
    }
    return zml.Sharding.PhysicalMesh.auto(allocator, target, used[0..degree]);
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
    replicated: zml.Sharding,

    pub fn init(platform: *zml.Platform, heads: HeadCounts) !Shardings {
        const strategy = try tensorParallelStrategyForAll(&platform.physical_mesh, heads);
        const model = try platform.registerShardingWithStrategy(
            "model",
            .mesh(.{ .model = .high_bandwidth }),
            strategy,
        );
        const degree = model.numPartitionsForLogicalAxis(.model);
        if (!tensorParallelHeadsOkAll(degree, heads)) {
            log.err(
                "tensor parallel degree {d} does not divide dit={d} enc={d} kv={d} ltx={d} gemma={d}",
                .{ degree, heads.dit, heads.enc, heads.kv, heads.ltx, heads.gemma },
            );
            return error.IncompatibleSharding;
        }
        return .{ .model = model, .replicated = platform.replicated_sharding };
    }

    pub fn all(self: Shardings) [1]zml.Sharding {
        return .{self.model};
    }

    pub fn rep(self: Shardings) [1]zml.Sharding {
        return .{self.replicated};
    }

    pub fn checkLoaded(self: Shardings, dit_cfg: config.Config, enc_cfg: config.EncoderConfig) !void {
        const degree = self.model.numPartitionsForLogicalAxis(.model);
        if (!tensorParallelHeadsOkAll(degree, .{
            .dit = dit_cfg.num_attention_heads,
            .enc = enc_cfg.num_attention_heads,
            .kv = enc_cfg.num_key_value_heads,
        })) {
            log.err(
                "Loaded heads dit={d} encoder={d} kv={d} do not divide by tensor-parallel degree {d}",
                .{ dit_cfg.num_attention_heads, enc_cfg.num_attention_heads, enc_cfg.num_key_value_heads, degree },
            );
            return error.IncompatibleSharding;
        }
    }
};
