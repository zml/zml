const std = @import("std");

pub fn MacWriter(comptime T: type) type {
    return struct {
        const Self = @This();

        mac: *T,
        interface: std.Io.Writer,

        pub fn init(mac_: *T) MacWriter(T) {
            return .{
                .mac = mac_,
                .interface = .{
                    .buffer = &.{},
                    .vtable = &.{
                        .drain = drain,
                    },
                },
            };
        }

        pub fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
            const self: *Self = @alignCast(@fieldParentPtr("interface", w));
            var total: usize = 0;
            for (data[0 .. data.len - 1]) |chunk| {
                self.mac.update(chunk);
                total += chunk.len;
            }
            // The last chunk is written `splat` times, which may be zero.
            const pattern = data[data.len - 1];
            for (0..splat) |_| {
                self.mac.update(pattern);
                total += pattern.len;
            }
            return total;
        }
    };
}

pub fn hmacWriter(hmac: anytype) MacWriter(std.meta.Child(@TypeOf(hmac))) {
    return .init(hmac);
}

test MacWriter {
    const Sha256 = std.crypto.hash.sha2.Sha256;
    const cases: []const struct { splat: usize, expected: []const u8 } = &.{
        .{ .splat = 0, .expected = "abc" },
        .{ .splat = 1, .expected = "abcx" },
        .{ .splat = 3, .expected = "abcxxx" },
    };

    for (cases) |case| {
        var sha: Sha256 = .init(.{});
        var sha_writer = hmacWriter(&sha);
        const n = try sha_writer.interface.writeSplat(&.{ "abc", "x" }, case.splat);
        try std.testing.expectEqual(case.expected.len, n);

        var expected: [Sha256.digest_length]u8 = undefined;
        Sha256.hash(case.expected, &expected, .{});
        try std.testing.expectEqualSlices(u8, &expected, &sha.finalResult());
    }
}
