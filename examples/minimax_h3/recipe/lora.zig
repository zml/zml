const std = @import("std");

const zml = @import("zml");

const weights = @import("weights.zig");

const log = std.log.scoped(.minimax_h3);

// =============================================================================
// recipe/lora.zig — Turbo / LTX LoRA merge
//
// Maps official adapter names onto H3 / LTX weights and folds A·B in.
// =============================================================================

pub const Part = enum { full, q, k, v };

pub const Pair = struct {
    a: []f32,
    b: []f32,
    rank: u32,
    in: u32,
    out: u32,
};

pub const Bundle = struct {
    pairs: std.StringHashMapUnmanaged(Pair),
    store: std.heap.ArenaAllocator,
    strength: f32,

    pub fn deinit(self: *Bundle) void {
        self.pairs.deinit(self.store.allocator());
        self.store.deinit();
    }

    pub fn get(self: *const Bundle, base: []const u8) ?Pair {
        return self.pairs.get(base);
    }

    pub fn mergeLinear(
        self: *const Bundle,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        weight: *zml.Buffer,
        official: []const u8,
    ) !void {
        var map_buf: [256]u8 = undefined;
        const mapped = mapOfficial(official, &map_buf) orelse return;
        const pair = self.get(mapped.base) orelse return;
        const shape = weight.shape();
        const dt = shape.dtype();
        const local_out: u32 = @intCast(shape.dim(.dout));
        const in: u32 = @intCast(shape.dim(.d));
        if (in != pair.in) return error.LoraInMismatch;

        const host_n = shape.count();
        const raw = try allocator.alloc(u8, host_n * dt.sizeOf());
        defer allocator.free(raw);
        try weight.toSlice(io, .init(shape, raw));

        const w = try allocator.alloc(f32, host_n);
        defer allocator.free(w);
        try bytesToF32(raw, dt, w);

        const b_off, const b_out = switch (mapped.part) {
            .full => .{ @as(u32, 0), pair.out },
            .q => .{ @as(u32, 0), pair.out / 3 },
            .k => .{ pair.out / 3, pair.out / 3 },
            .v => .{ 2 * (pair.out / 3), pair.out / 3 },
        };
        if (local_out != b_out) return error.LoraOutMismatch;
        const b = pair.b[@as(usize, b_off) * pair.rank ..][0 .. @as(usize, b_out) * pair.rank];
        mergeInto(w, pair.a, b, b_out, in, pair.rank, self.strength);

        const merged = try f32ToBytes(allocator, w, dt);
        defer allocator.free(merged);
        const sharding: zml.Sharding = if (shape.hasAtLeastOnePartitionedAxis())
            platform.shardings.get("model") orelse .replicated
        else
            .replicated;
        weight.deinit();
        weight.* = try weights.fromItemsSharded(io, platform, shape, sharding, merged);
    }

    pub fn mergeCore(self: *const Bundle, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, bufs: anytype, index: usize) !void {
        const start: std.Io.Timestamp = .now(io, .awake);
        var name: [96]u8 = undefined;
        try self.mergeLinear(allocator, io, platform, &bufs.attn.q.weight, try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.to_q.weight", .{index}));
        try self.mergeLinear(allocator, io, platform, &bufs.attn.k.weight, try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.to_k.weight", .{index}));
        try self.mergeLinear(allocator, io, platform, &bufs.attn.v.weight, try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.to_v.weight", .{index}));
        try self.mergeLinear(allocator, io, platform, &bufs.attn.out.weight, try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.to_out.0.weight", .{index}));
        try self.mergeLinear(allocator, io, platform, &bufs.mlp.fc1.weight, try std.fmt.bufPrint(&name, "transformer_blocks.{d}.ff.net.0.proj.weight", .{index}));
        try self.mergeLinear(allocator, io, platform, &bufs.mlp.fc2.weight, try std.fmt.bufPrint(&name, "transformer_blocks.{d}.ff.net.2.weight", .{index}));
        if (index % 10 == 0) log.info("lora merge core {d} [{f}]", .{ index, start.untilNow(io, .awake) });
    }

    pub fn mergeAdaln(self: *const Bundle, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, bufs: anytype, index: usize) !void {
        var name: [80]u8 = undefined;
        try self.mergeLinear(allocator, io, platform, &bufs.linear.weight, try std.fmt.bufPrint(&name, "transformer_blocks.{d}.adaln_proj.linear.weight", .{index}));
    }

    pub fn mergeRefiner(self: *const Bundle, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, block: anytype, index: usize) !void {
        var name: [96]u8 = undefined;
        try self.mergeLinear(allocator, io, platform, &block.attn.q.weight, try std.fmt.bufPrint(&name, "token_refiner.refiner_blocks.{d}.attn.to_q.weight", .{index}));
        try self.mergeLinear(allocator, io, platform, &block.attn.k.weight, try std.fmt.bufPrint(&name, "token_refiner.refiner_blocks.{d}.attn.to_k.weight", .{index}));
        try self.mergeLinear(allocator, io, platform, &block.attn.v.weight, try std.fmt.bufPrint(&name, "token_refiner.refiner_blocks.{d}.attn.to_v.weight", .{index}));
        try self.mergeLinear(allocator, io, platform, &block.attn.out.weight, try std.fmt.bufPrint(&name, "token_refiner.refiner_blocks.{d}.attn.to_out.0.weight", .{index}));
        try self.mergeLinear(allocator, io, platform, &block.mlp.fc1.weight, try std.fmt.bufPrint(&name, "token_refiner.refiner_blocks.{d}.ff.net.0.proj.weight", .{index}));
        try self.mergeLinear(allocator, io, platform, &block.mlp.fc2.weight, try std.fmt.bufPrint(&name, "token_refiner.refiner_blocks.{d}.ff.net.2.weight", .{index}));
    }

    pub fn mergeFinalAdaln(self: *const Bundle, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, bufs: anytype) !void {
        try self.mergeLinear(allocator, io, platform, &bufs.linear.weight, "norm_out.linear.weight");
    }

    /// LTX distilled LoRA keys are `diffusion_model.<module>` with no H3 remapping.
    pub fn mergeLtx(
        self: *const Bundle,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        weight: *zml.Buffer,
        module: []const u8,
    ) !void {
        var key_buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "diffusion_model.{s}", .{module}) catch return;
        const pair = self.get(key) orelse return;
        const shape = weight.shape();
        const dt = shape.dtype();
        const local_out: u32 = @intCast(shape.dim(.dout));
        const in: u32 = @intCast(shape.dim(.d));
        if (in != pair.in) return error.LoraInMismatch;
        if (local_out != pair.out) return error.LoraOutMismatch;

        const host_n = shape.count();
        const raw = try allocator.alloc(u8, host_n * dt.sizeOf());
        defer allocator.free(raw);
        try weight.toSlice(io, .init(shape, raw));

        const w = try allocator.alloc(f32, host_n);
        defer allocator.free(w);
        try bytesToF32(raw, dt, w);
        mergeInto(w, pair.a, pair.b, pair.out, in, pair.rank, self.strength);

        const merged = try f32ToBytes(allocator, w, dt);
        defer allocator.free(merged);
        const sharding: zml.Sharding = if (shape.hasAtLeastOnePartitionedAxis())
            platform.shardings.get("model") orelse .replicated
        else
            .replicated;
        weight.deinit();
        weight.* = try weights.fromItemsSharded(io, platform, shape, sharding, merged);
    }
};

pub fn mergeInto(w: []f32, a: []const f32, b: []const f32, out: u32, in: u32, rank: u32, scale: f32) void {
    std.debug.assert(w.len == @as(usize, out) * in);
    std.debug.assert(a.len == @as(usize, rank) * in);
    std.debug.assert(b.len == @as(usize, out) * rank);
    @setRuntimeSafety(false);
    const Vec = @Vector(8, f32);
    var o: usize = 0;
    while (o < out) : (o += 1) {
        const wp = w.ptr + o * in;
        const bp = b.ptr + o * rank;
        var r: usize = 0;
        while (r < rank) : (r += 1) {
            const br = scale * bp[r];
            if (br == 0) continue;
            const splat: Vec = @splat(br);
            const ap = a.ptr + r * in;
            var i: usize = 0;
            const n8 = in & ~@as(usize, 7);
            while (i < n8) : (i += 8) {
                const wv: Vec = @as(*const [8]f32, @ptrCast(wp + i)).*;
                const av: Vec = @as(*const [8]f32, @ptrCast(ap + i)).*;
                @as(*[8]f32, @ptrCast(wp + i)).* = wv + splat * av;
            }
            while (i < in) : (i += 1) {
                wp[i] += br * ap[i];
            }
        }
    }
}

pub const Map = struct {
    base: []const u8,
    part: Part,
};

/// ComfyUI Turbo LoRA names → official H3 keys.
pub fn mapOfficial(official: []const u8, buf: []u8) ?Map {
    var tmp: [256]u8 = undefined;
    const key = if (std.mem.endsWith(u8, official, ".weight")) official[0 .. official.len - 7] else official;
    if (std.mem.eql(u8, key, "norm_out.linear")) {
        return .{ .base = "final_layer.adaln_proj.linear", .part = .full };
    }
    if (rewrite(key, "transformer_blocks.", "blocks.", &tmp)) |b| {
        return mapBlock(b, buf);
    }
    if (rewrite(key, "token_refiner.refiner_blocks.", "token_refiner.blocks.", &tmp)) |b| {
        return mapBlock(b, buf);
    }
    return null;
}

fn rewrite(key: []const u8, from: []const u8, to: []const u8, buf: []u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, key, from)) return null;
    const rest = key[from.len..];
    if (to.len + rest.len > buf.len) return null;
    @memcpy(buf[0..to.len], to);
    @memcpy(buf[to.len..][0..rest.len], rest);
    return buf[0 .. to.len + rest.len];
}

fn mapBlock(name: []const u8, buf: []u8) ?Map {
    if (std.mem.endsWith(u8, name, ".attn.to_q")) return .{ .base = replaceTail(name, ".attn.to_q", ".attn.qkv_proj", buf) orelse return null, .part = .q };
    if (std.mem.endsWith(u8, name, ".attn.to_k")) return .{ .base = replaceTail(name, ".attn.to_k", ".attn.qkv_proj", buf) orelse return null, .part = .k };
    if (std.mem.endsWith(u8, name, ".attn.to_v")) return .{ .base = replaceTail(name, ".attn.to_v", ".attn.qkv_proj", buf) orelse return null, .part = .v };
    if (std.mem.endsWith(u8, name, ".attn.to_out.0")) return .{ .base = replaceTail(name, ".attn.to_out.0", ".attn.out_proj", buf) orelse return null, .part = .full };
    if (std.mem.endsWith(u8, name, ".ff.net.0.proj")) return .{ .base = replaceTail(name, ".ff.net.0.proj", ".mlp.fc1", buf) orelse return null, .part = .full };
    if (std.mem.endsWith(u8, name, ".ff.net.2")) return .{ .base = replaceTail(name, ".ff.net.2", ".mlp.fc2", buf) orelse return null, .part = .full };
    if (std.mem.endsWith(u8, name, ".adaln_proj.linear")) {
        if (name.len > buf.len) return null;
        @memcpy(buf[0..name.len], name);
        return .{ .base = buf[0..name.len], .part = .full };
    }
    return null;
}

fn replaceTail(name: []const u8, old: []const u8, new: []const u8, buf: []u8) ?[]const u8 {
    if (!std.mem.endsWith(u8, name, old)) return null;
    const head = name[0 .. name.len - old.len];
    if (head.len + new.len > buf.len) return null;
    @memcpy(buf[0..head.len], head);
    @memcpy(buf[head.len..][0..new.len], new);
    return buf[0 .. head.len + new.len];
}

pub fn load(allocator: std.mem.Allocator, io: std.Io, path: []const u8, strength: f32) !Bundle {
    var registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, path);
    defer registry.deinit();

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const a = arena.allocator();
    var pairs: std.StringHashMapUnmanaged(Pair) = .empty;
    errdefer pairs.deinit(a);

    var it = registry.iterator();
    while (it.next()) |e| {
        const name = e.key_ptr.*;
        if (!std.mem.endsWith(u8, name, ".lora_A.weight")) continue;
        const base = name[0 .. name.len - ".lora_A.weight".len];
        var bname_buf: [256]u8 = undefined;
        const bname = std.fmt.bufPrint(&bname_buf, "{s}.lora_B.weight", .{base}) catch return error.LoraName;
        const a_f = try readF32(a, io, &registry, name);
        const b_f = try readF32(a, io, &registry, bname);
        const rank: u32 = @intCast(e.value_ptr.shape.dim(0));
        const in: u32 = @intCast(e.value_ptr.shape.dim(1));
        const out: u32 = @intCast(b_f.len / rank);
        const key = try a.dupe(u8, base);
        try pairs.put(a, key, .{
            .a = a_f,
            .b = b_f,
            .rank = rank,
            .in = in,
            .out = out,
        });
    }
    log.info("lora: {d} pairs strength={d:.2} {s}", .{ pairs.count(), strength, path });
    return .{ .pairs = pairs, .store = arena, .strength = strength };
}

fn readF32(allocator: std.mem.Allocator, io: std.Io, registry: *zml.safetensors.TensorRegistry, name: []const u8) ![]f32 {
    const tensor = registry.tensors.get(name) orelse return error.LoraTensorMissing;
    const n = tensor.shape.count();
    const raw = try allocator.alloc(u8, tensor.byteSize());
    defer allocator.free(raw);
    try readExact(io, tensor, raw);
    const out = try allocator.alloc(f32, n);
    switch (tensor.shape.dtype()) {
        .f32 => @memcpy(out, std.mem.bytesAsSlice(f32, raw)),
        .bf16 => {
            const src = std.mem.bytesAsSlice(zml.floats.BFloat16, raw);
            for (out, src) |*d, s| d.* = s.toF32();
        },
        .f16 => {
            const src = std.mem.bytesAsSlice(f16, raw);
            for (out, src) |*d, s| d.* = s;
        },
        else => return error.UnsupportedLoraDtype,
    }
    return out;
}

fn readExact(io: std.Io, tensor: zml.safetensors.Tensor, out: []u8) !void {
    const file = try std.Io.Dir.cwd().openFile(io, tensor.file_uri, .{ .mode = .read_only });
    defer file.close(io);
    var tmp: [64 * 1024]u8 = undefined;
    var reader = file.reader(io, &tmp);
    try reader.seekTo(tensor.offset);
    try reader.interface.readSliceAll(out);
}

fn bytesToF32(raw: []const u8, dt: zml.DataType, out: []f32) !void {
    switch (dt) {
        .f32 => @memcpy(out, std.mem.bytesAsSlice(f32, raw)),
        .bf16 => {
            const src = std.mem.bytesAsSlice(zml.floats.BFloat16, raw);
            for (out, src) |*d, s| d.* = s.toF32();
        },
        .f16 => {
            const src = std.mem.bytesAsSlice(f16, raw);
            for (out, src) |*d, s| d.* = s;
        },
        else => return error.UnsupportedLoraDtype,
    }
}

fn f32ToBytes(allocator: std.mem.Allocator, w: []const f32, dt: zml.DataType) ![]u8 {
    switch (dt) {
        .f32 => return allocator.dupe(u8, std.mem.sliceAsBytes(w)),
        .bf16 => {
            const out = try allocator.alloc(u8, w.len * 2);
            const dst = std.mem.bytesAsSlice(zml.floats.BFloat16, out);
            for (dst, w) |*d, s| d.* = .fromF32(s);
            return out;
        },
        else => return error.UnsupportedLoraDtype,
    }
}
