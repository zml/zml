const std = @import("std");
const pi = @import("zml-smi/info").process_info;

pub const ProcessEnricher = struct {
    pub fn init(_: std.mem.Allocator, _: std.Io) !ProcessEnricher {
        return .{};
    }

    pub fn deinit(_: *ProcessEnricher) void {}

    pub fn enrich(_: *ProcessEnricher, _: std.Io, _: []pi.ProcessInfo) void {}
};
