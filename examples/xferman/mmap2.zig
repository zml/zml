const std = @import("std");
const stdx = @import("stdx");
const zml = @import("zml");
const asynk = @import("async");

const asyncc = asynk.asyncc;

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

fn checkSlicesForOverlaps(alloc: std.mem.Allocator, entire_buffer: []const u8, subslices: [][]const u8) !void {
    std.log.info("Checking for overlaps...", .{});
    var bytefield = try alloc.alloc(bool, entire_buffer.len);
    defer alloc.free(bytefield);

    for (0..bytefield.len) |i| {
        bytefield[i] = false;
    }

    for (subslices, 0..) |sub, idx| {
        const start: usize = @intFromPtr(sub.ptr) - @intFromPtr(entire_buffer.ptr);
        if (start + sub.len > entire_buffer.len) {
            std.log.err("Error: subslice {d} reaches outside of mmapped file: file(0..{d}), subslice({d}..{d})", .{
                idx,
                entire_buffer.len,
                start,
                start + sub.len,
            });
            return error.Overflow;
        }

        for (start..start + sub.len) |index| {
            if (bytefield[index] == true) {
                return error.Overlap;
            }
            bytefield[index] = true;
        }
    }
    std.log.info("Checking for overlaps...done", .{});
}

pub fn asyncMain() !void {
    // Short lived allocations
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = std.process.args();
    // Skip executable path
    _ = args.next().?;

    const file = if (args.next()) |path| blk: {
        std.debug.print("File path: {s}\n", .{path});
        break :blk path;
    } else {
        std.debug.print("Missing file path argument\n", .{});
        std.debug.print("Try: bazel run -c opt //xferman:mmap -- /path/to/mymodel.safetensors or /path/to/model.safetensors.index.json \n", .{});
        std.process.exit(0);
    };

    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);

    var mapped_file = try zml.aio.MemoryMappedFile.init(try asynk.File.open(file, .{}));
    errdefer mapped_file.deinit();
    mapped_file.data_offset = layer_data_offset;

    var buffer_store = try zml.aio.BufferStore.init(allocator, &.{mapped_file});
    defer buffer_store.deinit();
    var slice_list = std.ArrayList([]const u8).init(allocator);
    defer slice_list.deinit();

    for (layers) |layer| {
        const filedata = mapped_file.mappedSlice(layer.start, layer.len);
        try buffer_store.registerBuffer(
            buffer_store.arena.allocator(),
            layer.key,
            layer.shape,
            filedata,
        );
        try slice_list.append(filedata);
    }

    // try checkSlicesForOverlaps(allocator, mapped_file.mappedSlice(0, mapped_file.data.len), slice_list.items);

    var total_bytes: usize = 0;
    var timer = try std.time.Timer.start();

    var bit = buffer_store.buffers.iterator();
    while (bit.next()) |item| {
        const buffer = item.value_ptr;
        const key = item.key_ptr.*;
        std.log.info("Buffer {d} : {s} {} = {d} bytes @ {*}", .{ bit.index, key, buffer.shape, buffer.data.len, buffer.data.ptr });
    }
    const memory_kind: zml.pjrt.Memory.Kind = switch (platform.target) {
        .cpu => .unpinned_host,
        else => .device,
    };
    const events = try buffer_store.starTransferToDevice(platform, memory_kind);
    for (events, 0..) |event, idx| {
        std.log.info("awaiting event {d}", .{idx});
        try event.awaitt(platform.pjrt_api);
    }
    std.debug.print("Received {d} events\n", .{events.len});

    var it = buffer_store.buffers.iterator();
    var i: usize = 0;
    std.debug.print("\nStart to read {d} buffers from store..\n", .{buffer_store.buffers.count()});

    while (it.next()) |entry| : (i += 1) {
        total_bytes += entry.value_ptr.*.data.len;
        std.debug.print("Buffer: {s} ({any} / {any})\n", .{ entry.key_ptr.*, i + 1, buffer_store.buffers.count() });
    }

    const stop = timer.read();
    const time_in_s = stdx.math.divFloat(f64, stop, std.time.ns_per_s);
    const mbs = stdx.math.divFloat(f64, total_bytes, 1024 * 1024);

    std.debug.print("\nLoading speed: {d:.2} MB/s\n\n", .{mbs / time_in_s});
}

const Shape = zml.Shape;
const Layer = struct {
    key: []const u8,
    start: usize,
    len: usize,
    shape: Shape,
};

const layer_data_offset: usize = 16808;
const layers: []const Layer = &.{
    .{ .key = "model.embed_tokens.weight", .start = 0, .len = 525336576, .shape = Shape.init(.{ 128256, 2048 }, .bf16) },
    .{ .key = "model.layers.0.input_layernorm.weight", .start = 525336576, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.0.mlp.down_proj.weight", .start = 525340672, .len = 33554432, .shape = Shape.init(.{ 2048, 8192 }, .bf16) },
    .{ .key = "model.layers.0.mlp.gate_proj.weight", .start = 558895104, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.0.mlp.up_proj.weight", .start = 592449536, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.0.post_attention_layernorm.weight", .start = 626003968, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.0.self_attn.k_proj.weight", .start = 626008064, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.0.self_attn.o_proj.weight", .start = 628105216, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.0.self_attn.q_proj.weight", .start = 636493824, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.0.self_attn.v_proj.weight", .start = 644882432, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.1.input_layernorm.weight", .start = 646979584, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.1.mlp.down_proj.weight", .start = 646983680, .len = 33554432, .shape = Shape.init(.{ 2048, 8192 }, .bf16) },
    .{ .key = "model.layers.1.mlp.gate_proj.weight", .start = 680538112, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.1.mlp.up_proj.weight", .start = 714092544, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.1.post_attention_layernorm.weight", .start = 747646976, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.1.self_attn.k_proj.weight", .start = 747651072, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.1.self_attn.o_proj.weight", .start = 749748224, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.1.self_attn.q_proj.weight", .start = 758136832, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.1.self_attn.v_proj.weight", .start = 766525440, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.10.input_layernorm.weight", .start = 768622592, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.10.mlp.down_proj.weight", .start = 768626688, .len = 33554432, .shape = Shape.init(.{ 2048, 8192 }, .bf16) },
    .{ .key = "model.layers.10.mlp.gate_proj.weight", .start = 802181120, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.10.mlp.up_proj.weight", .start = 835735552, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.10.post_attention_layernorm.weight", .start = 869289984, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.10.self_attn.k_proj.weight", .start = 869294080, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.10.self_attn.o_proj.weight", .start = 871391232, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.10.self_attn.q_proj.weight", .start = 879779840, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.10.self_attn.v_proj.weight", .start = 888168448, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.11.input_layernorm.weight", .start = 890265600, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.11.mlp.down_proj.weight", .start = 890269696, .len = 33554432, .shape = Shape.init(.{ 2048, 8192 }, .bf16) },
    .{ .key = "model.layers.11.mlp.gate_proj.weight", .start = 923824128, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.11.mlp.up_proj.weight", .start = 957378560, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.11.post_attention_layernorm.weight", .start = 990932992, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.11.self_attn.k_proj.weight", .start = 990937088, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.11.self_attn.o_proj.weight", .start = 993034240, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.11.self_attn.q_proj.weight", .start = 1001422848, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.11.self_attn.v_proj.weight", .start = 1009811456, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.12.input_layernorm.weight", .start = 1011908608, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.12.mlp.down_proj.weight", .start = 1011912704, .len = 33554432, .shape = Shape.init(.{ 2048, 8192 }, .bf16) },
    .{ .key = "model.layers.12.mlp.gate_proj.weight", .start = 1045467136, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.12.mlp.up_proj.weight", .start = 1079021568, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.12.post_attention_layernorm.weight", .start = 1112576000, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.12.self_attn.k_proj.weight", .start = 1112580096, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.12.self_attn.o_proj.weight", .start = 1114677248, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.12.self_attn.q_proj.weight", .start = 1123065856, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.12.self_attn.v_proj.weight", .start = 1131454464, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.13.input_layernorm.weight", .start = 1133551616, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.13.mlp.down_proj.weight", .start = 1133555712, .len = 33554432, .shape = Shape.init(.{ 2048, 8192 }, .bf16) },
    .{ .key = "model.layers.13.mlp.gate_proj.weight", .start = 1167110144, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.13.mlp.up_proj.weight", .start = 1200664576, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.13.post_attention_layernorm.weight", .start = 1234219008, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.13.self_attn.k_proj.weight", .start = 1234223104, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.13.self_attn.o_proj.weight", .start = 1236320256, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.13.self_attn.q_proj.weight", .start = 1244708864, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.13.self_attn.v_proj.weight", .start = 1253097472, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.14.input_layernorm.weight", .start = 1255194624, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.14.mlp.down_proj.weight", .start = 1255198720, .len = 33554432, .shape = Shape.init(.{ 2048, 8192 }, .bf16) },
    .{ .key = "model.layers.14.mlp.gate_proj.weight", .start = 1288753152, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.14.mlp.up_proj.weight", .start = 1322307584, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.14.post_attention_layernorm.weight", .start = 1355862016, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.14.self_attn.k_proj.weight", .start = 1355866112, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.14.self_attn.o_proj.weight", .start = 1357963264, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.14.self_attn.q_proj.weight", .start = 1366351872, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.14.self_attn.v_proj.weight", .start = 1374740480, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.15.input_layernorm.weight", .start = 1376837632, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.15.mlp.down_proj.weight", .start = 1376841728, .len = 33554432, .shape = Shape.init(.{ 2048, 8192 }, .bf16) },
    .{ .key = "model.layers.15.mlp.gate_proj.weight", .start = 1410396160, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.15.mlp.up_proj.weight", .start = 1443950592, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.15.post_attention_layernorm.weight", .start = 1477505024, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.15.self_attn.k_proj.weight", .start = 1477509120, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.15.self_attn.o_proj.weight", .start = 1479606272, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.15.self_attn.q_proj.weight", .start = 1487994880, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.15.self_attn.v_proj.weight", .start = 1496383488, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.2.input_layernorm.weight", .start = 1498480640, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.2.mlp.down_proj.weight", .start = 1498484736, .len = 33554432, .shape = Shape.init(.{ 2048, 8192 }, .bf16) },
    .{ .key = "model.layers.2.mlp.gate_proj.weight", .start = 1532039168, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.2.mlp.up_proj.weight", .start = 1565593600, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.2.post_attention_layernorm.weight", .start = 1599148032, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.2.self_attn.k_proj.weight", .start = 1599152128, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.2.self_attn.o_proj.weight", .start = 1601249280, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.2.self_attn.q_proj.weight", .start = 1609637888, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.2.self_attn.v_proj.weight", .start = 1618026496, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.3.input_layernorm.weight", .start = 1620123648, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.3.mlp.down_proj.weight", .start = 1620127744, .len = 33554432, .shape = Shape.init(.{ 2048, 8192 }, .bf16) },
    .{ .key = "model.layers.3.mlp.gate_proj.weight", .start = 1653682176, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.3.mlp.up_proj.weight", .start = 1687236608, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.3.post_attention_layernorm.weight", .start = 1720791040, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.3.self_attn.k_proj.weight", .start = 1720795136, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.3.self_attn.o_proj.weight", .start = 1722892288, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.3.self_attn.q_proj.weight", .start = 1731280896, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.3.self_attn.v_proj.weight", .start = 1739669504, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.4.input_layernorm.weight", .start = 1741766656, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.4.mlp.down_proj.weight", .start = 1741770752, .len = 33554432, .shape = Shape.init(.{ 2048, 8192 }, .bf16) },
    .{ .key = "model.layers.4.mlp.gate_proj.weight", .start = 1775325184, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.4.mlp.up_proj.weight", .start = 1808879616, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.4.post_attention_layernorm.weight", .start = 1842434048, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.4.self_attn.k_proj.weight", .start = 1842438144, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.4.self_attn.o_proj.weight", .start = 1844535296, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.4.self_attn.q_proj.weight", .start = 1852923904, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.4.self_attn.v_proj.weight", .start = 1861312512, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.5.input_layernorm.weight", .start = 1863409664, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.5.mlp.down_proj.weight", .start = 1863413760, .len = 33554432, .shape = Shape.init(.{ 2048, 8192 }, .bf16) },
    .{ .key = "model.layers.5.mlp.gate_proj.weight", .start = 1896968192, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.5.mlp.up_proj.weight", .start = 1930522624, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.5.post_attention_layernorm.weight", .start = 1964077056, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.5.self_attn.k_proj.weight", .start = 1964081152, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.5.self_attn.o_proj.weight", .start = 1966178304, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.5.self_attn.q_proj.weight", .start = 1974566912, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.5.self_attn.v_proj.weight", .start = 1982955520, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.6.input_layernorm.weight", .start = 1985052672, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.6.mlp.down_proj.weight", .start = 1985056768, .len = 33554432, .shape = Shape.init(.{ 2048, 8192 }, .bf16) },
    .{ .key = "model.layers.6.mlp.gate_proj.weight", .start = 2018611200, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.6.mlp.up_proj.weight", .start = 2052165632, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.6.post_attention_layernorm.weight", .start = 2085720064, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.6.self_attn.k_proj.weight", .start = 2085724160, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.6.self_attn.o_proj.weight", .start = 2087821312, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.6.self_attn.q_proj.weight", .start = 2096209920, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.6.self_attn.v_proj.weight", .start = 2104598528, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.7.input_layernorm.weight", .start = 2106695680, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.7.mlp.down_proj.weight", .start = 2106699776, .len = 33554432, .shape = Shape.init(.{ 2048, 8192 }, .bf16) },
    .{ .key = "model.layers.7.mlp.gate_proj.weight", .start = 2140254208, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.7.mlp.up_proj.weight", .start = 2173808640, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.7.post_attention_layernorm.weight", .start = 2207363072, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.7.self_attn.k_proj.weight", .start = 2207367168, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.7.self_attn.o_proj.weight", .start = 2209464320, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.7.self_attn.q_proj.weight", .start = 2217852928, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.7.self_attn.v_proj.weight", .start = 2226241536, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.8.input_layernorm.weight", .start = 2228338688, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.8.mlp.down_proj.weight", .start = 2228342784, .len = 33554432, .shape = Shape.init(.{ 2048, 8192 }, .bf16) },
    .{ .key = "model.layers.8.mlp.gate_proj.weight", .start = 2261897216, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.8.mlp.up_proj.weight", .start = 2295451648, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.8.post_attention_layernorm.weight", .start = 2329006080, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.8.self_attn.k_proj.weight", .start = 2329010176, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.8.self_attn.o_proj.weight", .start = 2331107328, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.8.self_attn.q_proj.weight", .start = 2339495936, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.8.self_attn.v_proj.weight", .start = 2347884544, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.9.input_layernorm.weight", .start = 2349981696, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.9.mlp.down_proj.weight", .start = 2349985792, .len = 33554432, .shape = Shape.init(.{ 2048, 8192 }, .bf16) },
    .{ .key = "model.layers.9.mlp.gate_proj.weight", .start = 2383540224, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.9.mlp.up_proj.weight", .start = 2417094656, .len = 33554432, .shape = Shape.init(.{ 8192, 2048 }, .bf16) },
    .{ .key = "model.layers.9.post_attention_layernorm.weight", .start = 2450649088, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
    .{ .key = "model.layers.9.self_attn.k_proj.weight", .start = 2450653184, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.layers.9.self_attn.o_proj.weight", .start = 2452750336, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.9.self_attn.q_proj.weight", .start = 2461138944, .len = 8388608, .shape = Shape.init(.{ 2048, 2048 }, .bf16) },
    .{ .key = "model.layers.9.self_attn.v_proj.weight", .start = 2469527552, .len = 2097152, .shape = Shape.init(.{ 512, 2048 }, .bf16) },
    .{ .key = "model.norm.weight", .start = 2471624704, .len = 4096, .shape = Shape.init(.{2048}, .bf16) },
};
