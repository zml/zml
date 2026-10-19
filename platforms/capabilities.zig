//! Hardware identities and tuning capabilities, independent of kernel backends.
const std = @import("std");

const cuda = @import("platforms/cuda");
const oneapi = @import("platforms/oneapi");
const rocm = @import("platforms/rocm");

const Platform = @import("platforms.zig").Platform;

/// A conservative view shared by every device in a platform.
/// Architecture disagreements and missing information stay explicit.
pub const Capabilities = union(Platform) {
    cpu: void,
    cuda: ?cuda.Capabilities,
    rocm: ?rocm.Capabilities,
    tpu: void,
    neuron: void,
    oneapi: ?oneapi.Capabilities,
    metal: void,
    furiosa: void,

    pub const Cuda = cuda.Capabilities;
    pub const Rocm = rocm.Capabilities;
    pub const OneApi = oneapi.Capabilities;

    pub fn unknown(target: Platform) Capabilities {
        return switch (target) {
            .cuda => .{ .cuda = null },
            .rocm => .{ .rocm = null },
            .oneapi => .{ .oneapi = null },
            .cpu => .cpu,
            .tpu => .tpu,
            .neuron => .neuron,
            .metal => .metal,
            .furiosa => .furiosa,
        };
    }

    pub fn merge(self: Capabilities, other: Capabilities) Capabilities {
        std.debug.assert(std.meta.activeTag(self) == std.meta.activeTag(other));
        return switch (self) {
            .cuda => |caps| .{ .cuda = Cuda.merge(caps, other.cuda) },
            .rocm => |caps| .{ .rocm = Rocm.merge(caps, other.rocm) },
            .oneapi => |caps| .{ .oneapi = OneApi.merge(caps, other.oneapi) },
            .cpu, .tpu, .neuron, .metal, .furiosa => self,
        };
    }
};

test "capability merges keep backend tags when compute information is missing or differs" {
    const Case = struct { known: Capabilities, other: Capabilities };
    for ([_]Case{
        .{ .known = .{ .cuda = .init(.sm90, .{}) }, .other = .{ .cuda = .init(.sm100, .{}) } },
        .{ .known = .{ .rocm = .init(.gfx942, .{}) }, .other = .{ .rocm = .init(.gfx950, .{}) } },
        .{ .known = .{ .oneapi = .init(.pvc, .{}) }, .other = .{ .oneapi = .init(.bmg, .{}) } },
    }) |case| {
        const missing: Capabilities = .unknown(std.meta.activeTag(case.known));
        try std.testing.expectEqualDeep(case.known, case.known.merge(case.known));
        for ([_]Capabilities{ missing, case.other }) |caps| {
            try std.testing.expectEqualDeep(missing, case.known.merge(caps));
            try std.testing.expectEqualDeep(missing, caps.merge(case.known));
            try std.testing.expectEqualDeep(missing, case.known.merge(caps).merge(case.known));
            try std.testing.expectEqualDeep(missing, case.known.merge(caps.merge(case.known)));
        }
    }
}
