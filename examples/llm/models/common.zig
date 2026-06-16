const std = @import("std");

const zml = @import("zml");

pub const SessionOptions = struct {
    seqlen: u32,
    backend: zml.attention.attention.Backend,
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

pub const Shardings = struct {
    model: zml.Sharding,
    experts: zml.Sharding,

    pub fn init(platform: *zml.Platform) !Shardings {
        switch (platform.target) {
            .tpu => {
                var strategy_experts: zml.Sharding.Strategy = .parseBindings(.{ .experts = .link_x });
                strategy_experts.addFold(.link_x, &.{ .link_x, .link_y });
                var strategy_model: zml.Sharding.Strategy = .parseBindings(.{ .model = .link_x });
                strategy_model.addFold(.link_x, &.{ .link_x, .link_y });

                return .{
                    .model = try platform.registerShardingWithStrategy("model", .mesh(.{ .model = .high_bandwidth }), strategy_model),
                    .experts = try platform.registerShardingWithStrategy("experts", .mesh(.{ .experts = .high_bandwidth }), strategy_experts),
                };
            },
            .cuda, .rocm, .oneapi, .neuron, .cpu => return .{
                .model = try platform.registerSharding("model", .mesh(.{ .model = .high_bandwidth })),
                .experts = try platform.registerSharding("experts", .mesh(.{ .experts = .high_bandwidth })),
            },
        }
    }

    pub fn all(self: Shardings) [2]zml.Sharding {
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
