const std = @import("std");

pub fn apply(comptime table: anytype, dest: anytype, src: anytype) void {
    inline for (table) |m| {
        @field(dest, m.field) = m.query(src) catch null;
    }
}

pub fn poll(comptime DB: type, comptime Dev: type, comptime table: anytype) fn (?*std.heap.ArenaAllocator, DB, Dev) void {
    return struct {
        fn f(arena: ?*std.heap.ArenaAllocator, db: DB, dev: Dev) void {
            if (arena) |a| _ = a.reset(.retain_capacity);

            const back = db.back();
            back.* = db.front().*;
            apply(table, back, dev);
            db.swap();
        }
    }.f;
}
