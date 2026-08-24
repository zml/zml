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
            for (data) |chunk| {
                self.mac.update(chunk);
                total += chunk.len;
            }
            const last = data[data.len - 1];
            for (0..splat - 1) |_| {
                self.mac.update(last);
                total += last.len;
            }
            return total;
        }
    };
}

pub fn hmacWriter(hmac: anytype) MacWriter(std.meta.Child(@TypeOf(hmac))) {
    return .init(hmac);
}
