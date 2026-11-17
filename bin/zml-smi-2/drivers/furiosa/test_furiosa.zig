const std = @import("std");
const contract = @import("zml-smi-2/contract");
const native = @import("zml-smi-2/native");
const Furiosa = @import("zml-smi-2/drivers/furiosa");
const source = @import("zml-smi-2/source");

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
    try tmp.dir.createDirPath(io, "hwmon");
    try tmp.dir.writeFile(io, .{ .sub_path = "hwmon/temp1_input", .data = "47000\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "hwmon/temp2_input", .data = "31000\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "hwmon/power1_average", .data = "66001000\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "hwmon/power1_max", .data = "150000000\n" });
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
    const mgmt = try class.openDir("rngd!npu0mgmt");
    defer mgmt.close();
    const hwmon = try class.openDir("hwmon");
    defer hwmon.close();
    var chips = [_]Chip{.{
        .node = 0,
        .model = &models[0],
        .mgmt = mgmt,
        .hwmon = .{
            .dir = hwmon,
            .temperature = "temp1_input",
            .ambient = "temp2_input",
            .power = "power1_average",
        },
        .usable = .full,
    }};
    var furiosa: Furiosa = .{
        .devices = &.{.{ .target = .furiosa }},
        .class = class,
        .chips = &chips,
    };

    // Between the polls, the group spends half its cycles on tasks and core 5 all of them.
    var samples = [_]contract.Metrics{.{}};
    for ([_]u64{ 0, 37 }, [_]u64{ 1000, 2000 }) |util, cycles| {
        for ([_]u8{ 0, 1, 2, 3, 5 }) |core| {
            const task = if (core == 5) cycles else cycles / 2 + 250;
            try tmp.dir.writeFile(io, .{
                .sub_path = try std.fmt.bufPrint(&path, "rngd!npu0pe{d}/metrics", .{core}),
                .data = try std.fmt.bufPrint(&text, "CycleCount: 0x{x}\nTaskCycleCount: 0x{x}\n", .{ cycles, task }),
            });
        }
        samples[0] = .{};
        var reading: source.Reading = .{
            .samples = &samples,
            .allocator = std.testing.allocator,
        };
        defer reading.processes.deinit(std.testing.allocator);
        try furiosa.poll(std.testing.allocator, &reading);
        const chip = &samples[0];
        try std.testing.expectEqual(util, chip.get(.util_percent));
        try std.testing.expectEqual(4 * gib, chip.get(.mem_used_bytes));
        try std.testing.expectEqual(models[0].memory_bytes, chip.get(.mem_total_bytes));
        try std.testing.expectEqual(@backingInt(contract.Governor.powersave), chip.get(.governor));
        try std.testing.expectEqual(@backingInt(contract.Liveness.alive), chip.get(.liveness));
        try std.testing.expectEqual(0b11, chip.get(.throttle_reasons));
        try std.testing.expectEqual(47, chip.get(.temperature));
        try std.testing.expectEqual(31, chip.get(.ambient_temperature));
        try std.testing.expectEqual(66_001, chip.get(.power_mw));
        try std.testing.expectEqual(150_000, chip.get(.power_limit_mw));
    }

    // A process that holds the group sees its share.
    const row = support.processRow(&chips[0], 7, 0, .{ .mask = 0x0f }, &samples[0]);
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
