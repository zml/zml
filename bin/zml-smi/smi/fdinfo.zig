const std = @import("std");
const sysfs = @import("sysfs.zig");

/// One /proc/<pid>/fdinfo/<fd> file of a DRM client. Its strings point into the text it was parsed from.
pub const Fdinfo = struct {
    /// The PCI address of the device: "0000:03:00.0".
    pdev: ?[]const u8 = null,
    client_id: ?u64 = null,
    region_buffer: [8]Region = undefined,
    region_count: u8 = 0,
    engine_buffer: [16]Engine = undefined,
    engine_count: u8 = 0,

    /// Memory of one region, in bytes.
    pub const Region = struct {
        name: []const u8 = "",
        total: ?u64 = null,
        resident: ?u64 = null,
    };

    /// The busy time of one engine class: nanoseconds, or for xe, busy cycles out of total cycles.
    pub const Engine = struct {
        class: []const u8 = "",
        busy_ns: ?u64 = null,
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

/// Reads one fdinfo file into `buffer`, which the strings of the result point into.
pub fn read(dir: sysfs.Dir, sub_path: []const u8, buffer: []u8) !Fdinfo {
    return parse(try dir.read(sub_path, buffer));
}

pub fn parse(text: []const u8) Fdinfo {
    var info: Fdinfo = .{};
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const key = line[0..colon];
        const value = std.mem.trim(u8, line[colon + 1 ..], " \t");

        if (std.mem.eql(u8, key, "drm-pdev") or std.mem.eql(u8, key, "pdev")) {
            info.pdev = value;
        } else if (std.mem.eql(u8, key, "drm-client-id")) {
            info.client_id = sysfs.firstNumber(value);
        } else if (std.mem.startsWith(u8, key, "drm-engine-capacity-")) {
            continue;
        } else if (cut(key, "drm-engine-")) |class| {
            if (info.engine(class)) |engine| engine.busy_ns = sysfs.firstNumber(value);
        } else if (cut(key, "drm-total-cycles-")) |class| {
            if (info.engine(class)) |engine| engine.total_cycles = sysfs.firstNumber(value);
        } else if (cut(key, "drm-cycles-")) |class| {
            if (info.engine(class)) |engine| engine.cycles = sysfs.firstNumber(value);
        } else if (cut(key, "drm-total-")) |name| {
            if (info.region(name)) |region| region.total = bytes(value);
        } else if (cut(key, "drm-resident-")) |name| {
            if (info.region(name)) |region| region.resident = bytes(value);
        } else if (cut(key, "drm-memory-")) |name| {
            // The older amdgpu name of resident memory.
            if (info.region(name)) |region| region.resident = bytes(value);
        } else if (std.mem.endsWith(u8, key, " mem")) {
            // The oldest amdgpu one: "vram mem".
            if (info.region(key[0 .. key.len - " mem".len])) |region| region.resident = bytes(value);
        }
    }
    return info;
}

fn cut(key: []const u8, prefix: []const u8) ?[]const u8 {
    return if (std.mem.startsWith(u8, key, prefix)) key[prefix.len..] else null;
}

/// The entry called `name`, added when missing. Null past the capacity.
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

/// A size such as "4096 KiB", in bytes. A size without a unit is in bytes.
fn bytes(value: []const u8) ?u64 {
    var words = std.mem.tokenizeAny(u8, value, " \t");
    const size = std.fmt.parseInt(u64, words.next() orelse return null, 10) catch return null;
    const unit = words.next() orelse return size;
    const shift: u6 = if (std.ascii.eqlIgnoreCase(unit, "KiB") or std.ascii.eqlIgnoreCase(unit, "kB"))
        10
    else if (std.ascii.eqlIgnoreCase(unit, "MiB"))
        20
    else if (std.ascii.eqlIgnoreCase(unit, "GiB"))
        30
    else
        0;
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
    try std.testing.expectEqual(2, info.regions().len);
    try std.testing.expectEqualStrings("system0", info.regions()[0].name);
    try std.testing.expectEqual(2048 * 1024, info.regions()[0].total);
    try std.testing.expectEqual(3 * 1024 * 1024, info.regions()[1].resident);
    try std.testing.expectEqual(2, info.engines().len);
    try std.testing.expectEqualStrings("render", info.engines()[0].class);
    try std.testing.expectEqual(100, info.engines()[0].busy_ns);
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
    try std.testing.expectEqual(null, info.engines()[0].busy_ns);
    try std.testing.expectEqual(4096 * 1024, info.regions()[0].total);
    try std.testing.expectEqual(3072 * 1024, info.regions()[0].resident);
}

test "amdgpu keys" {
    const old = parse("pdev:\t0000:0c:00.0\nvram mem:\t1024 kB\n");
    try std.testing.expectEqualStrings("0000:0c:00.0", old.pdev.?);
    try std.testing.expectEqualStrings("vram", old.regions()[0].name);
    try std.testing.expectEqual(1024 * 1024, old.regions()[0].resident);

    const current = parse("drm-pdev:\t0000:0c:00.0\ndrm-memory-vram:\t2048 KiB\ndrm-engine-gfx:\t5 ns\n");
    try std.testing.expectEqual(2048 * 1024, current.regions()[0].resident);
    try std.testing.expectEqual(5, current.engines()[0].busy_ns);
}
