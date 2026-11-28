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
    const cases: []const struct { splat: usize, written: usize, expected: [Sha256.digest_length]u8 } = &.{
        // SHA-256 of "abc", "abcx" and "abcxxx".
        .{ .splat = 0, .written = 3, .expected = digest("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") },
        .{ .splat = 1, .written = 4, .expected = digest("7571ce1f8e21c6b13dd7ec2c5ec7c9e4dd9852e209869511853f2f1f74b17927") },
        .{ .splat = 3, .written = 6, .expected = digest("93a23a22387411b9341a9856d404d0278ca032cb4be90d8485db596c770766e5") },
    };

    for (cases) |case| {
        var sha: Sha256 = .init(.{});
        var sha_writer = hmacWriter(&sha);
        const n = try sha_writer.interface.writeSplat(&.{ "abc", "x" }, case.splat);
        try std.testing.expectEqual(case.written, n);
        try std.testing.expectEqualSlices(u8, &case.expected, &sha.finalResult());
    }
}

fn digest(comptime hex: *const [64]u8) [32]u8 {
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, hex) catch unreachable;
    return out;
}
