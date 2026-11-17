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
};

/// Reads one fdinfo file into `buffer`, which the strings of the result point into.
pub fn read(dir: sysfs.Dir, sub_path: []const u8, buffer: []u8) !Fdinfo {
    return parse(try dir.read(sub_path, buffer));
}

pub fn parse(text: []const u8) Fdinfo {
    var info: Fdinfo = .{};
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    lines_loop: while (lines.next()) |line| {
        const key, const raw_value = std.mem.cutScalar(u8, line, ':') orelse continue;
        const value = std.mem.trim(u8, raw_value, " \t");

        if (std.mem.eql(u8, key, "drm-pdev") or std.mem.eql(u8, key, "pdev")) {
            info.pdev = value;
            continue;
        }
        if (std.mem.eql(u8, key, "drm-client-id")) {
            info.client_id = sysfs.firstNumber(value);
            continue;
        }
        if (std.mem.startsWith(u8, key, "drm-engine-capacity-")) continue;
        inline for (.{
            .{ "drm-engine-", "busy_ns" },
            .{ "drm-total-cycles-", "total_cycles" },
            .{ "drm-cycles-", "cycles" },
        }) |binding| {
            if (std.mem.cutPrefix(u8, key, binding[0])) |class| {
                const engine = slot(&info.engine_buffer, &info.engine_count, "class", class) orelse
                    continue :lines_loop;
                @field(engine, binding[1]) = sysfs.firstNumber(value);
                continue :lines_loop;
            }
        }
        // "drm-memory-" and " mem" are older amdgpu resident-memory keys.
        const total = std.mem.cutPrefix(u8, key, "drm-total-");
        const name = total orelse std.mem.cutPrefix(u8, key, "drm-resident-") orelse
            std.mem.cutPrefix(u8, key, "drm-memory-") orelse
            std.mem.cutSuffix(u8, key, " mem") orelse continue;
        const region = slot(&info.region_buffer, &info.region_count, "name", name) orelse continue;
        const property = if (total != null) &region.total else &region.resident;
        property.* = bytes(value);
    }
    return info;
}

/// The entry called `name`, added when missing. Null past the capacity.
fn slot(buffer: anytype, count: *u8, comptime key: []const u8, name: []const u8) ?@TypeOf(&buffer[0]) {
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
    const spellings = [_][]const u8{ "kB", "KiB", "MiB", "GiB" };
    for (spellings, [_]u6{ 10, 10, 20, 30 }) |spelling, bits| {
        if (std.ascii.eqlIgnoreCase(unit, spelling)) {
            return std.math.mul(u64, size, @as(u64, 1) << bits) catch null;
        }
    }
    return size;
}
