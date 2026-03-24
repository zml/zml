const std = @import("std");

const zml = @import("zml");

pub const SessionOptions = struct {
    seqlen: u32,
    backend: zml.attention.attention.Backend,
    single: bool,
};

pub const Shardings = struct {
    replicated: zml.sharding.Sharding,
    model: zml.sharding.Sharding,

    pub fn init(platform: *zml.Platform) !Shardings {
        const model_mesh: zml.sharding.LogicalMesh = try .init("model", .{ .model = .high_bandwidth });
        const model_sharding_strategy: zml.sharding.Strategy = try .suggest(model_mesh, platform.physical_mesh);
        return .{
            .replicated = try zml.sharding.replicatedSharding(platform),
            .model = try .initFromStrategy(platform, model_mesh, model_sharding_strategy),
        };
    }

    pub fn all(self: *const Shardings) [2]zml.sharding.Sharding {
        return .{ self.replicated, self.model };
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
