const std = @import("std");

const xet = @import("xet.zig");

pub const RangeRegistration = struct { file_uri: []const u8, offset: u64, len: u64 };
pub const Lookup = struct { range: xet.ReconstructionRange, prefetch_ranges: []const xet.ReconstructionRange = &.{} };
pub const Registry = std.ArrayList(RangeRegistration);

pub fn registerRanges(registry: *Registry, allocator: std.mem.Allocator, ranges: []const RangeRegistration) !void {
    for (ranges) |range| {
        if (range.len != 0) try registry.append(allocator, range);
    }
    std.mem.sort(RangeRegistration, registry.items, {}, lessThan);
}

pub fn registerStore(registry: *Registry, allocator: std.mem.Allocator, store: anytype) !void {
    var it = store.registry.iterator();
    while (it.next()) |entry| {
        const tensor = entry.value_ptr.*;
        const len = tensor.byteSize();
        if (len != 0) try registry.append(allocator, .{ .file_uri = tensor.file_uri, .offset = tensor.offset, .len = len });
    }
    std.mem.sort(RangeRegistration, registry.items, {}, lessThan);
}

pub fn lookup(registry: Registry, file_uri: []const u8, offset: u64, take: u64, prefetch_out: []xet.ReconstructionRange) Lookup {
    const fallback_range: xet.ReconstructionRange = .{ .offset = offset, .len = take };
    for (registry.items, 0..) |range, i| {
        if (!sameFile(file_uri, range.file_uri)) continue;
        if (offset < range.offset or offset + take > range.offset + range.len) continue;

        var count: usize = 0;
        if (offset == range.offset) {
            for (registry.items[i + 1 ..]) |next| {
                if (count == prefetch_out.len or count == xet.METADATA_PREFETCH_DISTANCE) break;
                if (!sameFile(file_uri, next.file_uri)) continue;
                prefetch_out[count] = .{ .offset = next.offset, .len = next.len };
                count += 1;
            }
        }
        return .{ .range = .{ .offset = range.offset, .len = range.len }, .prefetch_ranges = prefetch_out[0..count] };
    }
    return .{ .range = fallback_range };
}

fn lessThan(_: void, lhs: RangeRegistration, rhs: RangeRegistration) bool {
    const file_order = std.mem.order(u8, trimScheme(lhs.file_uri), trimScheme(rhs.file_uri));
    if (file_order != .eq) return file_order == .lt;
    return lhs.offset < rhs.offset;
}

fn sameFile(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, trimScheme(a), trimScheme(b));
}

fn trimScheme(file_uri: []const u8) []const u8 {
    return if (std.mem.startsWith(u8, file_uri, "hf://")) file_uri["hf://".len..] else file_uri;
}
