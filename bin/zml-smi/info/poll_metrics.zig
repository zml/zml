const std = @import("std");

pub fn poll(comptime DB: type, comptime Dev: type, comptime table: anytype) fn (DB, Dev) void {
    return struct {
        fn f(db: DB, dev: Dev) void {
            if (@hasField(Dev, "arena")) {
                _ = dev.arena.reset(.retain_capacity);
            }

            const back = db.back();
            back.* = db.front().*;

            inline for (table) |m| {
                @field(back, m.field) = m.query(dev) catch null;
            }

            db.swap();
        }
    }.f;
}
