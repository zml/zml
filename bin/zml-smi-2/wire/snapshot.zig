const std = @import("std");
const contract = @import("zml-smi-2/contract");

/// Device and sample indices match. Process rows refer to those same indices.
pub const Snapshot = struct {
    devices: []const contract.Device,
    samples: []const contract.Metrics,
    processes: []const contract.ProcessInfo = &.{},
    host: contract.HostData = .{},

    /// The allocator and input bytes must outlive the snapshot.
    pub fn parse(arena: std.mem.Allocator, body: []const u8) !Snapshot {
        const snapshot = try std.json.parseFromSliceLeaky(Snapshot, arena, body, .{
            .ignore_unknown_fields = true,
        });
        if (snapshot.devices.len != snapshot.samples.len) return error.InvalidSnapshot;
        if (snapshot.devices.len > std.math.maxInt(u16)) return error.TooManyDevices;
        return snapshot;
    }
};
