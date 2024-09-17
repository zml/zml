const asynk = @import("async");
const eval = @import("torch/eval.zig");
const std = @import("std");
const yaml = @import("zig-yaml");
const zml = @import("../zml.zig");

const Decoder = @import("torch/parser.zig").Decoder;

const StringBuilder = std.ArrayListUnmanaged(u8);

pub fn open(allocator: std.mem.Allocator, path: []const u8) !zml.aio.BufferStore {
    var res: zml.aio.BufferStore = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
    };
    errdefer res.arena.deinit();

    const arena = res.arena.allocator();

    var mapped_file = try zml.aio.MemoryMappedFile.init(try asynk.File.open(path, .{}));
    errdefer mapped_file.deinit();

    var file_name_buffer: [std.fs.max_path_bytes]u8 = undefined;
    var link_name_buffer: [std.fs.max_path_bytes]u8 = undefined;
    var tar_iter = std.tar.iterator(
        mapped_file.file.reader(),
        .{
            .file_name_buffer = &file_name_buffer,
            .link_name_buffer = &link_name_buffer,
        },
    );
    while (try tar_iter.next()) |file| {
        if (std.mem.endsWith(u8, file.name, ".yaml")) {
            const yaml_data = try file.reader().readAllAlloc(arena, file.size);
            const parsed = try yaml.Yaml.load(arena, yaml_data);

            var prefix_buf: [1024]u8 = undefined;
            try zml.aio.yaml.parseMetadata(arena, &res, StringBuilder.initBuffer(&prefix_buf), parsed.docs.items[0].map);
        } else if (std.mem.endsWith(u8, file.name, ".ckpt") or std.mem.endsWith(u8, file.name, ".pt")) {
            const start = try mapped_file.file.getPos();
            var tmp: zml.aio.torch.PickleData = .{
                .data = try Decoder.fromTarFile(arena, mapped_file, file),
                .memo = undefined,
                .stack = undefined,
            };
            tmp.stack, tmp.memo = try eval.evaluate(arena, tmp.data.ops, true);
            try tmp.parseModel(arena, &res);
            // Since we directly manipulate the file handle pointer,
            // reset to the end of file so iterator does not error
            // and avoid `skipBytes`.
            try mapped_file.file.seekTo(start + file.size);
            file.unread_bytes.* = 0;
        } else if (std.mem.eql(u8, file.name, "./model_weights/")) @panic(".NeMo sharded weights are not yet supported") else continue;
    }

    return res;
}
