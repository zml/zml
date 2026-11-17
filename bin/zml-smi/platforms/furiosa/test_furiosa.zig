const std = @import("std");
const model = @import("zml-smi/model");
const native = @import("zml-smi/native");
const Furiosa = @import("zml-smi/platforms/furiosa");
const source = @import("zml-smi/source");

const support = Furiosa.testing;
const Node = support.Node;
const Chip = support.Chip;
const models = support.models;
const cores = support.cores;
const mgmtNode = support.mgmtNode;

test Node {
    try std.testing.expectEqual(Node{ .chip = 3, .cores = .{ .mask = 0x0f } }, support.parseNode("npu3pe0-3").?);
    try std.testing.expectEqual(Node{ .chip = 0, .cores = .{ .mask = 0x20 } }, support.parseNode("npu0pe5").?);
    try std.testing.expectEqual(Node{ .chip = 12, .cores = .{ .mask = 0x01 } }, support.parseNode("npu12").?);
    for ([_][]const u8{ "npu0ch0", "npu0ch0r", "npu0mgmt", "npu0bar4", "npu0dmar", "npu0pe3-1", "npu0pe8", "npu0pe", "npu" }) |name| {
        try std.testing.expectEqual(null, support.parseNode(name));
    }
    try std.testing.expectEqual(7, mgmtNode("rngd!npu7mgmt"));
    try std.testing.expectEqual(null, mgmtNode("rngd!npu7pe0"));
}

test "utilization and memory of fused cores" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const gib: u64 = 1 << 30;
    const share = models[0].memory_bytes / cores;

    // Cores 0 to 3 are one group, core 5 is alone, and core 6 was released after the occupancy was read:
    // the files of a core not in use are empty.
    try tmp.dir.createDirPath(io, "rngd!npu0mgmt");
    try tmp.dir.createDirPath(io, "rngd!npu0pe6");
    try tmp.dir.writeFile(io, .{ .sub_path = "rngd!npu0pe6/metrics", .data = "" });
    try tmp.dir.writeFile(io, .{ .sub_path = "rngd!npu0pe6/alloc_status", .data = "" });
    try tmp.dir.writeFile(io, .{ .sub_path = "rngd!npu0mgmt/pe_occupancy", .data = "1\n1\n1\n1\n0\n1\n1\n0\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "rngd!npu0mgmt/npu_governor", .data = "powersave\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "rngd!npu0mgmt/device_state", .data = "0\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "rngd!npu0mgmt/throttle_reason", .data = "Throttle reason = 0x00000003\n" });
    var text: [128]u8 = undefined;
    var path: [32]u8 = undefined;
    for ([_]u8{ 0, 1, 2, 3, 5 }) |core| {
        const fused = core < 4;
        try tmp.dir.createDirPath(io, try std.fmt.bufPrint(&path, "rngd!npu0pe{d}", .{core}));
        try tmp.dir.writeFile(io, .{
            .sub_path = try std.fmt.bufPrint(&path, "rngd!npu0pe{d}/alloc_status", .{core}),
            .data = try std.fmt.bufPrint(&text, "dram_capacity: 0x{x}\ndram_usage: 0x{x}\nshared_capacity: 0x0\nshared_usage: 0x0\n", .{
                if (fused) 4 * share else share, if (fused) 3 * gib else gib,
            }),
        });
    }

    const class: native.sysfs.Dir = .{ .handle = tmp.dir, .io = io };
    var chips = [_]Chip{.{ .node = 0, .model = &models[0], .mgmt = try class.openDir("rngd!npu0mgmt"), .hwmon = null, .usable = .full }};
    defer chips[0].mgmt.close();
    var furiosa: Furiosa = .{
        .devices = &.{.{ .target = .furiosa }},
        .class = class,
        .chips = &chips,
    };

    // Between the polls, the group spends half its cycles on tasks and core 5 all of them.
    for ([_]u64{ 0, 37 }, [_]u64{ 1000, 2000 }) |util, cycles| {
        for ([_]u8{ 0, 1, 2, 3, 5 }) |core| {
            const task = if (core == 5) cycles else cycles / 2 + 250;
            try tmp.dir.writeFile(io, .{
                .sub_path = try std.fmt.bufPrint(&path, "rngd!npu0pe{d}/metrics", .{core}),
                .data = try std.fmt.bufPrint(&text, "CycleCount: 0x{x}\nTaskCycleCount: 0x{x}\n", .{ cycles, task }),
            });
        }
        var samples = [_]model.Sample{.init(.furiosa)};
        var reading: source.Reading = .{
            .samples = &samples,
            .allocator = std.testing.allocator,
        };
        defer reading.processes.deinit(std.testing.allocator);
        try furiosa.poll(std.testing.allocator, &reading);
        const chip = samples[0].furiosa;
        try std.testing.expectEqual(util, chip.util_percent);
        try std.testing.expectEqual(4 * gib, chip.mem_used_bytes);
        try std.testing.expectEqual(@backingInt(model.Furiosa.Governor.powersave), chip.governor);
        try std.testing.expectEqual(@backingInt(model.Furiosa.Liveness.alive), chip.liveness);
        try std.testing.expectEqual(0b11, chip.throttle_reasons);
    }

    // A process that holds the group sees its share.
    const row = support.processRow(&chips[0], 7, 0, .{ .mask = 0x0f }, .{ .util_percent = 37, .mem_used_bytes = 4 * gib });
    try std.testing.expectEqual(50, row.dev_util_percent);
    try std.testing.expectEqual(3 * gib / 1024, row.dev_mem_kib);

    // A malformed allocation must not claim more cores than the chip makes usable.
    try tmp.dir.writeFile(io, .{
        .sub_path = "rngd!npu0pe0/alloc_status",
        .data = "dram_capacity: 0xffffffffffffffff\n" ++
            "dram_usage: 0x0\nshared_capacity: 0x0\nshared_usage: 0x0\n",
    });
    try std.testing.expectEqual(null, support.memoryInUse(&furiosa, &chips[0], .full));
}
