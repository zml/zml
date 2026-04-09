//! Zig compiler segfault: `@Struct()` + recursive comptime-generic `visit`
//!
//!   OK  https://codeberg.org/ziglang/zig/commit/f16eb18ce8c24
//!   BAD https://codeberg.org/ziglang/zig/commit/fd2718f82ab70
//!
//! `zig build-exe repro_simple.zig`  →  Segmentation fault
const std = @import("std");
const Tensor = struct { };


const Device = struct { platform: *const Platform };
const Platform = struct { devices: []const Device };

const Buffer = struct { platform: *const Platform };

fn Contains(comptime H: type, comptime T: type) bool {
    @setEvalBranchQuota(10_000);
    if (H == T or H == ?T) return true;
    return switch (@typeInfo(H)) {
        .@"struct" => |i| {
            inline for (i.fields) |f| {
                if (!f.is_comptime and Contains(f.type, T)) return true;
            }
            return false;
        },
        .array => |i| Contains(i.child, T),
        .pointer => |i| Contains(i.child, T),
        .optional => |i| Contains(i.child, T),
        else => false,
    };
}

fn MapRestrict(comptime From: type, comptime To: type) type {
    return struct {
        pub fn map(comptime T: type) type {
            if (T == From) return To;
            if (T == ?From) return ?To;
            if (!Contains(T, From)) return void;
            return switch (@typeInfo(T)) {
                .@"struct" => |si| {
                    const fields = si.fields;
                    var n: usize = 0;
                    var names: [fields.len][]const u8 = undefined;
                    var types: [fields.len]type = undefined;
                    var attrs: [fields.len]std.builtin.Type.StructField.Attributes = undefined;
                    for (fields) |f| {
                        if (!f.is_comptime and Contains(f.type, From)) {
                            names[n] = f.name;
                            types[n] = map(f.type);
                            attrs[n] = .{ .@"align" = @alignOf(types[n]) };
                            if (types[n] == ?To) attrs[n].default_value_ptr = &@as(?To, null);
                            n += 1;
                        }
                    }
                    if (n == 0) return void;
                    return @Struct(.auto, null, names[0..n], types[0..n], attrs[0..n]);
                },
                else => T,
            };
        }
    };
}

fn visit(comptime cb: anytype, ctx: @typeInfo(@TypeOf(cb)).@"fn".params[0].type.?, v: anytype) void {
    const Ptr = @TypeOf(v);
    const pi = @typeInfo(Ptr).pointer;
    const Child = pi.child;
    const K = @typeInfo(@typeInfo(@TypeOf(cb)).@"fn".params[1].type.?).pointer.child;

    switch (Ptr) {
        *const K, *K => return cb(ctx, v),
        *const ?K, *?K => return if (v.*) |*val| cb(ctx, val) else {},
        else => {},
    }

    switch (pi.size) {
        .one => switch (@typeInfo(Child)) {
            .@"struct" => |s| inline for (s.fields) |f| {
                switch (@typeInfo(f.type)) {
                    .pointer => visit(cb, ctx, @field(v, f.name)),
                    .@"struct" => visit(cb, ctx, &@field(v, f.name)),
                    .array, .optional => visit(cb, ctx, &@field(v, f.name)),
                    else => {},
                }
            },
            .array => for (v) |*e| visit(cb, ctx, e),
            .optional => if (v.* != null) visit(cb, ctx, &v.*.?),
            else => @compileError("unsupported"),
        },
        .slice => for (v) |*ve| {
            switch (@typeInfo(Child)) {
                .@"struct" => |s| inline for (s.fields) |f| {
                    if (f.is_comptime or comptime !Contains(f.type, K)) continue;
                    if (@typeInfo(f.type) == .pointer)
                        visit(cb, ctx, @field(ve, f.name))
                    else
                        visit(cb, ctx, &@field(ve, f.name));
                },
                else => @compileError("unsupported"),
            }
        },
        .many, .c => @compileError("unsupported"),
    }
}

const Linear = struct { weight: Tensor };

pub fn main() void {
    var ctx: struct { n: usize = 0 } = .{};
    const m: MapRestrict(Tensor, Buffer).map(Linear) = .{
        .weight = undefined,
    };
    visit(struct {
        fn call(_ctx: *@TypeOf(ctx), _: *const Buffer) void {
            _ctx.n += 1;
        }
    }.call, &ctx, &m);
    std.debug.print("{}\n", .{ctx.n});
}
