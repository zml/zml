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
    model: zml.Meshe,
    experts: zml.Meshe,

    pub fn init(platform: *zml.Platform) !Meshes {
        switch (platform.target) {
            .tpu => {
                const has_link_y = platform.physical_mesh.hasAxis(.link_y);

                var strategy_experts: zml.Mesh.Strategy = .parseBindings(.{ .experts = .link_x });
                var strategy_model: zml.Mesh.Strategy = .parseBindings(.{ .model = .link_x });
                if (has_link_y) {
                    strategy_experts.addFold(.link_x, &.{ .link_x, .link_y });
                    strategy_model.addFold(.link_x, &.{ .link_x, .link_y });
                }

                return .{
                    .model = try platform.registerMesheWithStrategy("model", .mesh(.{ .model = .high_bandwidth }), strategy_model),
                    .experts = try platform.registerMesheWithStrategy("experts", .mesh(.{ .experts = .high_bandwidth }), strategy_experts),
                };
            },
            .cuda, .rocm, .oneapi, .neuron, .metal, .cpu => return .{
                .model = try platform.registerMeshe("model", .mesh(.{ .model = .high_bandwidth })),
                .experts = try platform.registerMeshe("experts", .mesh(.{ .experts = .high_bandwidth })),
            },
        }
    }

    pub fn all(self: Meshes) [2]zml.Meshe {
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
