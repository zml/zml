const std = @import("std");

pub fn poll(comptime DB: type, comptime Dev: type, comptime table: anytype) fn (?*std.heap.ArenaAllocator, DB, Dev) void {
    return struct {
        fn f(arena: ?*std.heap.ArenaAllocator, db: DB, dev: Dev) void {
            if (arena) |a| _ = a.reset(.retain_capacity);

            const back = db.back();
            back.* = db.front().*;

            inline for (table) |m| {
                @field(back, m.field) = m.query(dev) catch null;
            }

            db.swap();
        }
    }.f;
}
