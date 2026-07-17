const std = @import("std");

const fast_in_place_limit = 128 * 1024;

/// Degroup bytes in place from BG4 grouped layout to output order.
pub fn degroupBytesInPlace(bytes: []u8) void {
    degroupInPlace(bytes);
}

/// Degroup BG4 grouped bytes from `grouped` into `out`.
/// This mirrors xet-core's out-of-place `bg4_regroup_together` style.
pub fn degroupBytesInto(grouped: []const u8, out: []u8) !void {
    if (grouped.len != out.len) return error.SizeMismatch;

    const n = grouped.len;
    const split = n / 4;
    const rem = n % 4;

    const g0 = grouped;
    const g1_start = split + @as(usize, if (rem >= 1) 1 else 0);
    const g2_start = g1_start + split + @as(usize, if (rem >= 2) 1 else 0);
    const g3_start = g2_start + split + @as(usize, if (rem >= 3) 1 else 0);
    const g1 = grouped[g1_start..];
    const g2 = grouped[g2_start..];
    const g3 = grouped[g3_start..];

    var i: usize = 0;
    while (i < split) : (i += 1) {
        const o = 4 * i;
        out[o] = g0[i];
        out[o + 1] = g1[i];
        out[o + 2] = g2[i];
        out[o + 3] = g3[i];
    }

    switch (rem) {
        1 => {
            out[4 * split] = g0[split];
        },
        2 => {
            out[4 * split] = g0[split];
            out[4 * split + 1] = g1[split];
        },
        3 => {
            out[4 * split] = g0[split];
            out[4 * split + 1] = g1[split];
            out[4 * split + 2] = g2[split];
        },
        else => {},
    }
}

pub const DegroupWriter = struct {
    interface: std.Io.Writer,

    pub fn init(output: []u8) DegroupWriter {
        return .{
            .interface = .{
                .buffer = output,
                .vtable = &.{
                    .drain = drain,
                    .flush = flush,
                    .rebase = rebase,
                },
            },
        };
    }

    fn drain(_: *std.Io.Writer, _: []const []const u8, _: usize) std.Io.Writer.Error!usize {
        return error.WriteFailed;
    }

    fn flush(writer: *std.Io.Writer) std.Io.Writer.Error!void {
        if (writer.end != writer.buffer.len) return error.WriteFailed;
        degroupInPlace(writer.buffer[0..writer.end]);
    }

    fn rebase(_: *std.Io.Writer, preserve: usize, capacity: usize) std.Io.Writer.Error!void {
        if (preserve != 0 or capacity != 0) return error.WriteFailed;
    }
};

fn degroupInPlace(bytes: []u8) void {
    if (bytes.len <= fast_in_place_limit) {
        var visited_storage: [visitedByteLen(fast_in_place_limit)]u8 = undefined;
        const visited = visited_storage[0..visitedByteLen(bytes.len)];
        @memset(visited, 0);
        degroupInPlaceTracked(bytes, visited);
        return;
    }

    degroupInPlaceCycleLeaders(bytes);
}

fn degroupInPlaceTracked(bytes: []u8, visited: []u8) void {
    const lanes = LaneLayout.init(bytes.len);
    var start: usize = 0;
    while (start < bytes.len) : (start += 1) {
        if (isVisited(visited, start)) continue;

        var carry = bytes[start];
        var src = start;
        while (!isVisited(visited, src)) {
            markVisited(visited, src);
            const dst = lanes.outputIndex(src);
            const next = bytes[dst];
            bytes[dst] = carry;
            carry = next;
            src = dst;
        }
    }
}

fn degroupInPlaceCycleLeaders(bytes: []u8) void {
    const lanes = LaneLayout.init(bytes.len);
    var start: usize = 0;
    while (start < bytes.len) : (start += 1) {
        const first_dst = lanes.outputIndex(start);
        if (first_dst == start or !isCycleLeader(lanes, start)) continue;

        var carry = bytes[start];
        var src = start;
        while (true) {
            const dst = lanes.outputIndex(src);
            const next = bytes[dst];
            bytes[dst] = carry;
            carry = next;
            src = dst;
            if (src == start) break;
        }
    }
}

fn visitedByteLen(bit_len: usize) usize {
    return (bit_len + 7) / 8;
}

fn isVisited(visited: []const u8, index: usize) bool {
    return (visited[index / 8] & (@as(u8, 1) << @intCast(index % 8))) != 0;
}

fn markVisited(visited: []u8, index: usize) void {
    visited[index / 8] |= @as(u8, 1) << @intCast(index % 8);
}

fn isCycleLeader(lanes: LaneLayout, start: usize) bool {
    var cursor = lanes.outputIndex(start);
    while (cursor != start) {
        if (cursor < start) return false;
        cursor = lanes.outputIndex(cursor);
    }
    return true;
}

const LaneLayout = struct {
    g1_start: usize,
    g2_start: usize,
    g3_start: usize,
    g3_len: usize,

    fn init(len: usize) LaneLayout {
        const g0_len = (len + 3) / 4;
        const g1_len = (len + 2) / 4;
        const g2_len = (len + 1) / 4;
        return .{
            .g1_start = g0_len,
            .g2_start = g0_len + g1_len,
            .g3_start = g0_len + g1_len + g2_len,
            .g3_len = len / 4,
        };
    }

    fn outputIndex(self: LaneLayout, grouped_index: usize) usize {
        if (grouped_index < self.g1_start) return grouped_index * 4;
        if (grouped_index < self.g2_start) return (grouped_index - self.g1_start) * 4 + 1;
        if (grouped_index < self.g3_start) return (grouped_index - self.g2_start) * 4 + 2;
        return (grouped_index - self.g3_start) * 4 + 3;
    }
};
