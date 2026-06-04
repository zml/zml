const std = @import("std");

pub const Stats = struct {
    logical_bytes: u64 = 0,
    xorb_bytes: u64 = 0,
    decode_ns: u64 = 0,
};

pub const AtomicStats = struct {
    logical_bytes: std.atomic.Value(u64) = .init(0),
    xorb_bytes: std.atomic.Value(u64) = .init(0),
    decode_ns: std.atomic.Value(u64) = .init(0),

    pub fn reset(self: *AtomicStats) void {
        self.logical_bytes.store(0, .monotonic);
        self.xorb_bytes.store(0, .monotonic);
        self.decode_ns.store(0, .monotonic);
    }

    pub fn snapshot(self: *const AtomicStats) Stats {
        return .{
            .logical_bytes = self.logical_bytes.load(.monotonic),
            .xorb_bytes = self.xorb_bytes.load(.monotonic),
            .decode_ns = self.decode_ns.load(.monotonic),
        };
    }

    pub fn addLogicalBytes(self: *AtomicStats, value: u64) void {
        if (value != 0) _ = self.logical_bytes.fetchAdd(value, .monotonic);
    }

    pub fn addXorbBytes(self: *AtomicStats, value: u64) void {
        if (value != 0) _ = self.xorb_bytes.fetchAdd(value, .monotonic);
    }

    pub fn addDecodeNs(self: *AtomicStats, value: u64) void {
        if (value != 0) _ = self.decode_ns.fetchAdd(value, .monotonic);
    }
};
