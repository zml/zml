const std = @import("std");

pub fn NamedArgs(comptime Spec: type, comptime ValueT: type) type {
    const in = @typeInfo(Spec).@"struct".fields;
    comptime var names: [in.len][]const u8 = undefined;
    for (in, 0..) |f, i| names[i] = f.name;
    return @Struct(.auto, null, &names, &@splat(ValueT), &@splat(.{}));
}

pub fn Built(comptime Spec: type, comptime BuilderT: type, comptime ValueT: type) type {
    return struct {
        kernel: BuilderT,
        args: NamedArgs(Spec, ValueT),
        allocator: std.mem.Allocator,

        pub fn deinit(self: *@This()) void {
            self.kernel.deinit();
            self.allocator.destroy(self);
        }
    };
}
