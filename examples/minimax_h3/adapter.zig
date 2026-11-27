//! Native Pi-H3 checkpoint composition. Source descriptors always describe
//! the bytes on disk; FP16 -> BF16 conversion belongs to the consuming graph.
const std = @import("std");
const zml = @import("zml");

const Registry = zml.safetensors.TensorRegistry;
const log = std.log.scoped(.pi_h3);

const AdapterConfig = struct {
    _class_name: enum { MiniMaxH3Transformer3DModel },
    adaln_rank: u32,
    time_table_size: u32,
    num_layers: u32,
    hidden_size: u32,
    dwd_head_config: struct {
        linear_head: bool,
        first_frame_period: u32,
        pixel_patch_size: [3]u32,
        first_frame_pixel_patch_size: [3]u32,
        pixel_refiner: struct { type: enum { SolH3DWDHead } },
    },
};

fn validateConfig(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !void {
    const bytes = try dir.readFileAlloc(io, "config.json", allocator, .limited(64 * 1024));
    defer allocator.free(bytes);
    const parsed = try std.json.parseFromSlice(AdapterConfig, allocator, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const c = parsed.value;
    const head = c.dwd_head_config;
    if (c.adaln_rank != 8 or c.time_table_size != 1025 or c.num_layers != 50 or c.hidden_size != 5376 or
        head.linear_head or head.first_frame_period != 5 or
        !std.mem.eql(u32, &head.pixel_patch_size, &.{ 4, 32, 32 }) or
        !std.mem.eql(u32, &head.first_frame_pixel_patch_size, &.{ 1, 32, 32 })) return error.UnsupportedAdapterConfig;
}

const BaseManifest = struct {
    reg: *const Registry,
    count: usize = 0,

    fn require(self: *BaseManifest, prefix: []const u8, suffix: []const u8, dtype: zml.DataType, dims: []const i64) !void {
        var buf: [160]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix });
        const entry = self.reg.tensors.get(name) orelse {
            log.err("missing required pruned tensor {s}", .{name});
            return error.MissingPrunedTensor;
        };
        if (entry.shape.dtype() != dtype or !std.mem.eql(i64, entry.shape.dims(), dims)) {
            log.err("unexpected pruned tensor: {f}", .{entry});
            return error.InvalidPrunedShape;
        }
        self.count += 1;
    }

    fn linear(self: *BaseManifest, prefix: []const u8, dtype: zml.DataType, out: i64, in: i64) !void {
        try self.require(prefix, ".weight", dtype, &.{ out, in });
        try self.require(prefix, ".bias", dtype, &.{out});
    }

    fn block(self: *BaseManifest, prefix: []const u8) !void {
        inline for (.{ "attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight" }) |name|
            try self.require(prefix, name, .bf16, &.{ 7168, 5376 });
        inline for (.{ "attn.norm_q.weight", "attn.norm_k.weight" }) |name|
            try self.require(prefix, name, .bf16, &.{128});
        inline for (.{ "norm1.weight", "norm2.weight" }) |name|
            try self.require(prefix, name, .bf16, &.{5376});
        try self.require(prefix, "attn.to_out.0.weight", .bf16, &.{ 5376, 7168 });
        try self.require(prefix, "ff.net.0.proj.weight", .bf16, &.{ 28672, 5376 });
        try self.require(prefix, "ff.net.2.weight", .bf16, &.{ 5376, 14336 });
    }

    fn validate(reg: *const Registry) !void {
        var m: BaseManifest = .{ .reg = reg };
        try m.require("", "time_embedder.table", .f32, &.{ 1025, 8 });
        try m.require("", "adaln_basis", .f32, &.{ 8, 2688 });
        try m.require("", "adaln_mean", .f32, &.{2688});
        try m.linear("context_embedder", .bf16, 5376, 5120);
        try m.linear("proj_in", .f32, 5376, 96);
        try m.linear("proj_out", .f32, 96, 5376);
        try m.linear("audio_proj_in", .f32, 5376, 32);
        try m.linear("audio_proj_out", .f32, 32, 5376);
        try m.require("", "norm_out.linear.weight", .bf16, &.{ 10752, 8 });
        try m.require("", "norm_out.folded_bias", .f32, &.{10752});
        try m.require("", "norm_out.norm.weight", .bf16, &.{5376});
        try m.require("", "token_refiner.final_norm.weight", .bf16, &.{5376});
        var buf: [80]u8 = undefined;
        for (0..2) |i| try m.block(try std.fmt.bufPrint(&buf, "token_refiner.refiner_blocks.{d}.", .{i}));
        for (0..50) |i| {
            const prefix = try std.fmt.bufPrint(&buf, "transformer_blocks.{d}.", .{i});
            try m.block(prefix);
            try m.require(prefix, "adaln_proj.linear.weight", .bf16, &.{ 96768, 8 });
            try m.require(prefix, "adaln_proj.folded_bias", .f32, &.{96768});
        }
        if (reg.tensors.count() != m.count) return error.UnexpectedPrunedTensor;
    }
};

fn copyBase(out: *Registry, input: zml.safetensors.Tensor) !void {
    if (std.mem.eql(u8, input.name, "adaln_mean") or std.mem.eql(u8, input.name, "adaln_basis")) return;
    var entry = input;
    const suffix = ".folded_bias";
    var buf: [160]u8 = undefined;
    if (std.mem.endsWith(u8, input.name, suffix))
        entry.name = try std.fmt.bufPrint(&buf, "{s}.linear.bias", .{input.name[0 .. input.name.len - suffix.len]});
    if (out.tensors.contains(entry.name)) return error.DuplicateCanonicalName;
    try out.registerTensor(entry);
}

fn overlay(out: *Registry, entry: zml.safetensors.Tensor) !void {
    if (entry.shape.dtype() != .f16) return error.InvalidAdapterDtype;
    if (out.tensors.get(entry.name)) |replaced| {
        if (!std.mem.eql(i64, replaced.shape.dims(), entry.shape.dims())) return error.InvalidReplacementShape;
    } else if (!std.mem.startsWith(u8, entry.name, "dwd_") and std.mem.find(u8, entry.name, ".lora_") == null) {
        return error.UnknownAdapterTensor;
    }
    try out.registerTensor(entry);
}

/// Return an owning registry; all names and URIs have been copied before
/// either source registry is freed. No stores may bind before composition.
pub fn open(allocator: std.mem.Allocator, io: std.Io, base_path: []const u8, adapter_path: []const u8) !Registry {
    var base_dir = try std.Io.Dir.cwd().openDir(io, base_path, .{});
    defer base_dir.close(io);
    const index_name = "diffusion_pytorch_model.safetensors.index.json";
    const index = try base_dir.openFile(io, index_name, .{});
    defer index.close(io);
    var base = try zml.safetensors.fetchRegistry(allocator, io, base_dir, index, index_name);
    defer base.deinit();
    try BaseManifest.validate(&base);

    var dir = try std.Io.Dir.cwd().openDir(io, adapter_path, .{});
    defer dir.close(io);
    try validateConfig(allocator, io, dir);
    const file = try dir.openFile(io, "diffusion_pytorch_model.safetensors", .{});
    defer file.close(io);
    var adapter = Registry.init(allocator);
    defer adapter.deinit();
    try zml.safetensors.parseSafetensors(allocator, io, &adapter, file);
    if (adapter.tensors.count() != 456) return error.InvalidAdapterInventory;
    var out = Registry.init(allocator);
    errdefer out.deinit();
    for (base.tensors.values()) |entry| try copyBase(&out, entry);
    for (adapter.tensors.values()) |entry| try overlay(&out, entry);
    return out;
}
