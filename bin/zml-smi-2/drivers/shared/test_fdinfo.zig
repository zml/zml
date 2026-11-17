const std = @import("std");
const fdinfo = @import("zml-smi-2/native").fdinfo;

test "i915 engines and regions" {
    const info = fdinfo.parse(
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
    const info = fdinfo.parse(
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
    const old = fdinfo.parse("pdev:\t0000:0c:00.0\nvram mem:\t1024 kB\n");
    try std.testing.expectEqualStrings("0000:0c:00.0", old.pdev.?);
    try std.testing.expectEqualStrings("vram", old.regions()[0].name);
    try std.testing.expectEqual(1024 * 1024, old.regions()[0].resident);

    const current = fdinfo.parse("drm-pdev:\t0000:0c:00.0\ndrm-memory-vram:\t2048 KiB\ndrm-engine-gfx:\t5 ns\n");
    try std.testing.expectEqual(2048 * 1024, current.regions()[0].resident);
    try std.testing.expectEqual(5, current.engines()[0].busy_ns);
}
