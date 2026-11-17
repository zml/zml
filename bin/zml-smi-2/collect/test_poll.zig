const std = @import("std");
const contract = @import("zml-smi-2/contract");
const polling = @import("zml-smi-2/poll");
const source = @import("zml-smi-2/source");

const Fixture = struct {
    devices: []const contract.Device = &.{.{ .target = .cuda }},
    reads: usize = 0,

    pub const poll = polling.PerDevice(@This()).poll;

    pub const Probe = struct {
        reads: *usize,

        pub fn temperature(_: *const Probe) ?u64 {
            return 0;
        }

        pub fn memory(self: *const Probe) contract.Reply(&.{ .mem_used_bytes, .mem_total_bytes }) {
            self.reads.* += 1;
            return .{ .mem_used_bytes = 24, .mem_total_bytes = 48 };
        }
    };

    pub fn probe(self: *Fixture, _: usize) Probe {
        return .{ .reads = &self.reads };
    }
};

test "source projection invokes a grouped probe once and preserves zero" {
    const allocator = std.testing.allocator;
    var fixture: Fixture = .{};
    const reader: source.Source = .from(&fixture);
    var samples = [_]contract.Metrics{.{}};
    var reading: source.Reading = .{ .samples = &samples, .allocator = allocator };
    try reader.poll(reader.context, allocator, &reading);
    try std.testing.expectEqual(1, fixture.reads);
    try std.testing.expectEqual(24, samples[0].get(.mem_used_bytes));
    try std.testing.expectEqual(48, samples[0].get(.mem_total_bytes));
    try std.testing.expectEqual(0, samples[0].get(.temperature));
    try std.testing.expectEqual(null, samples[0].get(.power_mw));
    samples[0].put(.temperature, null);
    try std.testing.expectEqual(null, samples[0].get(.temperature));
}
