const std = @import("std");

const LaneLayout = struct {
    g0_len: usize,
    g1_start: usize,
    g2_start: usize,
    g3_start: usize,

    fn init(len: usize) LaneLayout {
        const g0_len = (len + 3) / 4;
        const g1_len = (len + 2) / 4;
        const g2_len = (len + 1) / 4;
        return .{
            .g0_len = g0_len,
            .g1_start = g0_len,
            .g2_start = g0_len + g1_len,
            .g3_start = g0_len + g1_len + g2_len,
        };
    }
};

pub const DegroupWriter = struct {
    output: []u8,
    interface: std.Io.Writer,

    pub fn init(grouped: []u8, output: []u8) DegroupWriter {
        std.debug.assert(grouped.len >= output.len);
        return .{
            .output = output,
            .interface = .{
                .buffer = grouped[0..output.len],
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
        const self: *DegroupWriter = @alignCast(@fieldParentPtr("interface", writer));
        if (writer.end != self.output.len) return error.WriteFailed;
        const output = self.output;
        const grouped = writer.buffer[0..writer.end];

        const lanes = LaneLayout.init(output.len);
        const g0 = grouped[0..lanes.g0_len];
        const g1 = grouped[lanes.g1_start..lanes.g2_start];
        const g2 = grouped[lanes.g2_start..lanes.g3_start];
        const g3 = grouped[lanes.g3_start..];

        var i: usize = 0;

        const vec_len: comptime_int = std.simd.suggestVectorLength(u8) orelse 16;

        const Vec = @Vector(vec_len, u8);
        const OutVec = @Vector(vec_len * 4, u8);

        while (i + vec_len <= g3.len) : (i += vec_len) {
            const v0: Vec = @as(*align(1) const Vec, @ptrCast(g0[i..].ptr)).*;
            const v1: Vec = @as(*align(1) const Vec, @ptrCast(g1[i..].ptr)).*;
            const v2: Vec = @as(*align(1) const Vec, @ptrCast(g2[i..].ptr)).*;
            const v3: Vec = @as(*align(1) const Vec, @ptrCast(g3[i..].ptr)).*;
            @as(*align(1) OutVec, @ptrCast(output[i * 4 ..].ptr)).* = std.simd.interlace(.{ v0, v1, v2, v3 });
        }

        while (i < g3.len) : (i += 1) {
            output[i * 4] = g0[i];
            output[i * 4 + 1] = g1[i];
            output[i * 4 + 2] = g2[i];
            output[i * 4 + 3] = g3[i];
        }

        const tail_start = i * 4;

        if (tail_start < output.len) output[tail_start] = g0[i];
        if (tail_start + 1 < output.len) output[tail_start + 1] = g1[i];
        if (tail_start + 2 < output.len) output[tail_start + 2] = g2[i];
    }

    fn rebase(_: *std.Io.Writer, preserve: usize, capacity: usize) std.Io.Writer.Error!void {
        if (preserve != 0 or capacity != 0) return error.WriteFailed;
    }
};
