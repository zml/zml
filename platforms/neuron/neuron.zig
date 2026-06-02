const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/neuron");

pub const CompilerTarget = enum {
    trn1,
    inf2,
    trn1n,
    trn2,
    trn2n,
    trn3,
};

pub const Family = enum(u32) {
    unknown = 0,
    inf1 = 1,
    trn1 = 2,
    trn1n = 3,
    inf2 = 4,
    trn2 = 5,
    trn2n = 6,
    inf2e = 7,
    trn2p = 8,
    trn2u = 9,
    trn2e = 10,
    trn2eu = 11,
    trn2ac = 12,
    trn2uac = 13,
    trn3 = 14,
    trn3pds98 = 15,

    pub fn compilerTarget(self: Family) CompilerTarget {
        return switch (self) {
            .unknown, .inf1 => unreachable,
            .trn1 => .trn1,
            .trn1n => .trn1n,
            .inf2, .inf2e => .inf2,
            .trn2, .trn2n, .trn2p, .trn2u, .trn2e, .trn2eu, .trn2ac, .trn2uac => .trn2,
            .trn3, .trn3pds98 => .trn3,
        };
    }
};

pub const Size = enum(u32) {
    xl1 = 0,
    xl2 = 1,
    xl4 = 2,
    xl6 = 3,
    xl8 = 4,
    xl24 = 5,
    xl32 = 6,
    xl48 = 7,
    xl3 = 8,
    unknown = 9,
};

pub const Instance = struct {
    family: Family,
    size: Size,

    pub fn compilerTarget(self: Instance) CompilerTarget {
        return self.family.compilerTarget();
    }

    pub fn coresPerChip(self: Instance) !usize {
        return switch (self.family) {
            .inf1 => 4,
            .trn1, .trn1n, .inf2, .inf2e => 2,
            .trn2, .trn2n, .trn2p, .trn2u, .trn2e, .trn2eu, .trn2ac, .trn2uac, .trn3, .trn3pds98 => 8,
            .unknown => unreachable,
        };
    }
};

pub fn instance() !Instance {
    var info = std.mem.zeroes(c.struct_nrt_instance_info);
    info.family = @intFromEnum(Family.unknown);
    info.size = @intFromEnum(Size.unknown);

    if (c.nrt_get_instance_info(&info, @sizeOf(c.struct_nrt_instance_info)) != c.NRT_SUCCESS) {
        return error.Unavailable;
    }

    return .{
        .family = @enumFromInt(info.family),
        .size = @enumFromInt(info.size),
    };
}

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_NEURON");
}

fn hasNeuronDevice(io: std.Io) bool {
    std.Io.Dir.accessAbsolute(io, "/dev/neuron0", .{ .read = true }) catch return false;
    return true;
}

fn isRunningOnEC2(io: std.Io) !bool {
    const AmazonEC2 = "Amazon EC2";
    var buffer: [AmazonEC2.len]u8 = undefined;
    return std.mem.eql(
        u8,
        AmazonEC2,
        try std.Io.Dir.readFile(
            .cwd(),
            io,
            "/sys/devices/virtual/dmi/id/sys_vendor",
            &buffer,
        ),
    );
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    _ = allocator;
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }
    if (!(isRunningOnEC2(io) catch false)) {
        return error.Unavailable;
    }
    if (!hasNeuronDevice(io)) {
        return error.Unavailable;
    }

    const r = try bazel.runfiles(bazel_builtin.current_repository);

    var sandbox_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libpjrt_neuron/sandbox", &sandbox_path_buf) orelse {
        log.err("Failed to find sandbox path for NEURON runtime", .{});
        return error.FileNotFound;
    };

    return blk: {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libpjrt_neuron.so" });
        break :blk .loadFrom(path);
    };
}
