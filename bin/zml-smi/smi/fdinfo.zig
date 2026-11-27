const std = @import("std");
const procfs = @import("procfs.zig");
const sysfs = @import("sysfs.zig");

pub const Fdinfo = struct {
    pdev: ?[]const u8 = null,
    client_id: ?u64 = null,
    region_buffer: [8]Region = undefined,
    region_count: u8 = 0,
    engine_buffer: [16]Engine = undefined,
    engine_count: u8 = 0,

    pub const Region = struct {
        name: []const u8 = "",
        bytes: u64 = 0,
    };

    pub const Engine = struct {
        class: []const u8 = "",
        busy: ?std.Io.Duration = null,
        cycles: ?u64 = null,
        total_cycles: ?u64 = null,
    };

    pub fn regions(self: *const Fdinfo) []const Region {
        return self.region_buffer[0..self.region_count];
    }

    pub fn engines(self: *const Fdinfo) []const Engine {
        return self.engine_buffer[0..self.engine_count];
    }

    fn region(self: *Fdinfo, name: []const u8) ?*Region {
        return slot(Region, &self.region_buffer, &self.region_count, "name", name);
    }

    fn engine(self: *Fdinfo, class: []const u8) ?*Engine {
        return slot(Engine, &self.engine_buffer, &self.engine_count, "class", class);
    }
};

pub fn read(dir: sysfs.Dir, sub_path: []const u8, buffer: []u8) !Fdinfo {
    return parse(try dir.read(sub_path, buffer));
}

/// The DRM clients of every process. A client shared through dup() or fork() is returned once.
pub const Clients = struct {
    holders: procfs.FdHolders,
    scratch: std.mem.Allocator,
    seen: std.AutoHashMapUnmanaged(struct { u64, u64 }, void) = .empty,
    buffer: [8192]u8 = undefined,

    pub const Client = struct {
        pid: u32,
        info: Fdinfo,
    };

    pub fn init(io: std.Io, scratch: std.mem.Allocator) Clients {
        return .{ .holders = .init(io, &.{"/dev/dri/"}), .scratch = scratch };
    }

    pub fn deinit(self: *Clients) void {
        self.seen.deinit(self.scratch);
        self.holders.deinit();
    }

    pub fn next(self: *Clients) ?Client {
        const dir: sysfs.Dir = .{ .handle = self.holders.proc orelse return null, .io = self.holders.io };

        while (self.holders.next()) |holder| {
            var path: [48]u8 = undefined;
            const sub_path = std.fmt.bufPrint(&path, "{d}/fdinfo/{d}", .{ holder.pid, holder.fd }) catch continue;
            const info = read(dir, sub_path, &self.buffer) catch continue;

            if (info.client_id) |id| {
                const device = std.hash.Wyhash.hash(0, info.pdev orelse "");
                if ((self.seen.getOrPut(self.scratch, .{ device, id }) catch continue).found_existing) continue;
            }

            return .{ .pid = holder.pid, .info = info };
        }

        return null;
    }
};

pub fn parse(text: []const u8) Fdinfo {
    var info: Fdinfo = .{};
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const key, const raw_value = std.mem.cutScalar(u8, line, ':') orelse continue;
        const value = std.mem.trim(u8, raw_value, " \t");

        if (std.mem.eql(u8, key, "drm-pdev") or std.mem.eql(u8, key, "pdev")) {
            info.pdev = value;
        } else if (std.mem.eql(u8, key, "drm-client-id")) {
            info.client_id = sysfs.firstNumber(value);
        } else if (std.mem.startsWith(u8, key, "drm-engine-capacity-")) {
            continue;
        } else if (std.mem.cutPrefix(u8, key, "drm-engine-")) |class| {
            if (sysfs.firstNumber(value)) |ns| {
                if (info.engine(class)) |engine| engine.busy = .fromNanoseconds(ns);
            }
        } else if (std.mem.cutPrefix(u8, key, "drm-total-cycles-")) |class| {
            if (info.engine(class)) |engine| engine.total_cycles = sysfs.firstNumber(value);
        } else if (std.mem.cutPrefix(u8, key, "drm-cycles-")) |class| {
            if (info.engine(class)) |engine| engine.cycles = sysfs.firstNumber(value);
        } else if (std.mem.cutPrefix(u8, key, "drm-resident-") orelse
            std.mem.cutPrefix(u8, key, "drm-memory-") orelse
            std.mem.cutSuffix(u8, key, " mem")) |name|
        {
            if (bytes(value)) |size| {
                if (info.region(name)) |region| region.bytes = size;
            }
        }
    }

    return info;
}

fn slot(comptime T: type, buffer: []T, count: *u8, comptime key: []const u8, name: []const u8) ?*T {
    for (buffer[0..count.*]) |*item| {
        if (std.mem.eql(u8, @field(item, key), name)) return item;
    }

    if (count.* == buffer.len) return null;

    const item = &buffer[count.*];
    count.* += 1;
    item.* = .{};
    @field(item, key) = name;

    return item;
}

fn bytes(value: []const u8) ?u64 {
    var words = std.mem.tokenizeAny(u8, value, " \t");
    const size = std.fmt.parseInt(u64, words.next() orelse return null, 10) catch return null;
    const unit = words.next() orelse return size;

    const shift: u6 = blk: {
        if (std.ascii.eqlIgnoreCase(unit, "KiB") or std.ascii.eqlIgnoreCase(unit, "kB")) break :blk 10;
        if (std.ascii.eqlIgnoreCase(unit, "MiB")) break :blk 20;
        if (std.ascii.eqlIgnoreCase(unit, "GiB")) break :blk 30;

        break :blk 0;
    };

    return std.math.mul(u64, size, @as(u64, 1) << shift) catch null;
}

test "i915 engines and regions" {
    const info = parse(
        \\pos: 0
        \\drm-driver: i915
        \\drm-pdev: 0000:03:00.0
        \\drm-client-id: 42
        \\drm-total-system0: 2048 KiB
        \\drm-resident-local0: 3 MiB
        \\drm-engine-render: 100 ns
        \\drm-engine-capacity-video: 2
        \\drm-engine-video-decode: 50 ns
    );
    try std.testing.expectEqualStrings("0000:03:00.0", info.pdev.?);
    try std.testing.expectEqual(42, info.client_id);
    try std.testing.expectEqual(1, info.regions().len);
    try std.testing.expectEqualStrings("local0", info.regions()[0].name);
    try std.testing.expectEqual(3 * 1024 * 1024, info.regions()[0].bytes);
    try std.testing.expectEqual(2, info.engines().len);
    try std.testing.expectEqualStrings("render", info.engines()[0].class);
    try std.testing.expectEqual(std.Io.Duration.fromNanoseconds(100), info.engines()[0].busy);
    try std.testing.expectEqualStrings("video-decode", info.engines()[1].class);
}

test "xe cycles" {
    const info = parse(
        \\drm-pdev: 0000:03:00.0
        \\drm-cycles-rcs: 10
        \\drm-total-cycles-rcs: 100
        \\drm-cycles-ccs: 30
        \\drm-total-cycles-ccs: 100
        \\drm-total-vram0: 4096 KiB
        \\drm-resident-vram0: 3072 KiB
    );
    try std.testing.expectEqual(2, info.engines().len);
    try std.testing.expectEqual(10, info.engines()[0].cycles);
    try std.testing.expectEqual(100, info.engines()[0].total_cycles);
    try std.testing.expectEqual(null, info.engines()[0].busy);
    try std.testing.expectEqual(1, info.regions().len);
    try std.testing.expectEqual(3072 * 1024, info.regions()[0].bytes);
}

test "amdgpu keys" {
    const old = parse("pdev:\t0000:0c:00.0\nvram mem:\t1024 kB\n");
    try std.testing.expectEqualStrings("0000:0c:00.0", old.pdev.?);
    try std.testing.expectEqualStrings("vram", old.regions()[0].name);
    try std.testing.expectEqual(1024 * 1024, old.regions()[0].bytes);

    const current = parse("drm-pdev:\t0000:0c:00.0\ndrm-memory-vram:\t2048 KiB\ndrm-engine-gfx:\t5 ns\n");
    try std.testing.expectEqual(2048 * 1024, current.regions()[0].bytes);
    try std.testing.expectEqual(std.Io.Duration.fromNanoseconds(5), current.engines()[0].busy);
}
