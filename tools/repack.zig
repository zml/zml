const std = @import("std");

const Tensor = struct {
    name: []const u8,
    dtype: []const u8,
    shape: []const i64,
    offset: u64,
    size: u64,
};

const Output = struct {
    name: []const u8,
    dtype: []const u8,
    shape: []const i64,
    sources: []const Tensor,
    interleave: bool = false,

    fn size(self: Output) u64 {
        var n: u64 = 0;
        for (self.sources) |source| n += source.size;
        return n;
    }
};

const ExpertKey = struct { prefix: []const u8, expert: usize, leaf: []const u8 };
const Group = struct { prefix: []const u8, leaf: []const u8, names: std.ArrayList([]const u8) = .empty };

fn parseExpertKey(name: []const u8) ?ExpertKey {
    if (!std.mem.startsWith(u8, name, "layers.")) return null;
    var it = std.mem.splitScalar(u8, name, '.');
    if (!std.mem.eql(u8, it.next() orelse return null, "layers")) return null;
    _ = std.fmt.parseInt(usize, it.next() orelse return null, 10) catch return null;
    if (!std.mem.eql(u8, it.next() orelse return null, "ffn")) return null;
    if (!std.mem.eql(u8, it.next() orelse return null, "experts")) return null;
    const expert_text = it.next() orelse return null;
    const expert = std.fmt.parseInt(usize, expert_text, 10) catch return null;
    const weight = it.next() orelse return null;
    const kind = it.next() orelse return null;
    if (it.next() != null) return null;
    if (!(std.mem.eql(u8, weight, "w1") or std.mem.eql(u8, weight, "w2") or std.mem.eql(u8, weight, "w3"))) return null;
    if (!(std.mem.eql(u8, kind, "weight") or std.mem.eql(u8, kind, "scale"))) return null;
    const marker = std.mem.indexOf(u8, name, ".experts.") orelse return null;
    return .{ .prefix = name[0 .. marker + ".experts".len], .expert = expert, .leaf = name[name.len - weight.len - kind.len - 1 ..] };
}

fn readFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only });
    defer file.close(io);
    var reader = file.reader(io, &.{});
    return try reader.interface.readAlloc(allocator, try file.length(io));
}

fn tensorFromValue(allocator: std.mem.Allocator, name: []const u8, value: std.json.Value, data_start: u64) !Tensor {
    const obj = value.object;
    const shape_json = (obj.get("shape") orelse return error.InvalidSafetensors).array;
    const shape = try allocator.alloc(i64, shape_json.items.len);
    for (shape_json.items, shape) |dim, *out| {
        if (dim.integer < 0) return error.InvalidSafetensors;
        out.* = dim.integer;
    }
    const offsets = (obj.get("data_offsets") orelse return error.InvalidSafetensors).array.items;
    if (offsets.len != 2 or offsets[0].integer < 0 or offsets[1].integer < offsets[0].integer) return error.InvalidSafetensors;
    const begin: u64 = @intCast(offsets[0].integer);
    const end: u64 = @intCast(offsets[1].integer);
    return .{ .name = name, .dtype = (obj.get("dtype") orelse return error.InvalidSafetensors).string, .shape = shape, .offset = data_start + begin, .size = end - begin };
}

fn appendJsonString(w: *std.Io.Writer, s: []const u8) !void {
    try w.print("{f}", .{std.json.fmt(s, .{})});
}

fn encodeHeader(allocator: std.mem.Allocator, outputs: []const Output, metadata: ?std.json.Value) ![]u8 {
    var w: std.Io.Writer.Allocating = .init(allocator);
    errdefer w.deinit();
    try w.writer.writeByte('{');
    var first = true;
    var offset: u64 = 0;
    for (outputs) |output| {
        if (!first) try w.writer.writeByte(',');
        first = false;
        try appendJsonString(&w.writer, output.name);
        try w.writer.writeAll(":{\"dtype\":");
        try appendJsonString(&w.writer, output.dtype);
        try w.writer.writeAll(",\"shape\":[");
        for (output.shape, 0..) |dim, i| {
            if (i != 0) try w.writer.writeByte(',');
            try w.writer.print("{d}", .{dim});
        }
        const end = offset + output.size();
        try w.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ offset, end });
        offset = end;
    }
    if (metadata) |meta| {
        if (!first) try w.writer.writeByte(',');
        try w.writer.writeAll("\"__metadata__\":");
        try w.writer.print("{f}", .{std.json.fmt(meta, .{})});
    }
    try w.writer.writeByte('}');
    while (w.writer.end % 8 != 0) try w.writer.writeByte(' ');
    return try w.toOwnedSlice();
}

fn copyRange(io: std.Io, input: std.Io.File, output: std.Io.File, output_offset: *u64, source_offset: u64, size: u64, buffer: []u8) !void {
    var offset = source_offset;
    var remaining = size;
    while (remaining != 0) {
        const count: usize = @intCast(@min(remaining, buffer.len));
        if (try input.readPositionalAll(io, buffer[0..count], offset) != count) return error.UnexpectedEof;
        try output.writePositionalAll(io, buffer[0..count], output_offset.*);
        output_offset.* += count;
        offset += count;
        remaining -= count;
    }
}

fn interleaveRows(io: std.Io, input: std.Io.File, output: std.Io.File, output_offset: *u64, left: Tensor, right: Tensor, buffer: []u8) !void {
    const rows: usize = @intCast(left.shape[0]);
    if (rows == 0 or left.size % rows != 0) return error.InvalidShape;
    const row_size: usize = @intCast(left.size / rows);
    if (row_size == 0) return;
    if (row_size > buffer.len / 2) {
        for (0..rows) |row| {
            try copyRange(io, input, output, output_offset, left.offset + row * row_size, row_size, buffer);
            try copyRange(io, input, output, output_offset, right.offset + row * row_size, row_size, buffer);
        }
        return;
    }
    const rows_per_chunk = @max(1, buffer.len / (2 * row_size));
    var row_start: usize = 0;
    while (row_start < rows) {
        const row_count = @min(rows_per_chunk, rows - row_start);
        for (0..row_count) |row| {
            const destination = buffer[(2 * row * row_size)..][0..row_size];
            if (try input.readPositionalAll(io, destination, left.offset + (row_start + row) * row_size) != row_size) return error.UnexpectedEof;
            const right_destination = buffer[((2 * row + 1) * row_size)..][0..row_size];
            if (try input.readPositionalAll(io, right_destination, right.offset + (row_start + row) * row_size) != row_size) return error.UnexpectedEof;
        }
        const bytes = row_count * 2 * row_size;
        try output.writePositionalAll(io, buffer[0..bytes], output_offset.*);
        output_offset.* += bytes;
        row_start += row_count;
    }
}

fn sameShape(a: Tensor, b: Tensor) bool {
    return std.mem.eql(i64, a.shape, b.shape) and std.mem.eql(u8, a.dtype, b.dtype) and a.size == b.size;
}

fn rewriteShard(allocator: std.mem.Allocator, io: std.Io, path: []const u8, keys: []const []const u8, expected_experts: ?usize, buffer_size: usize, verbose: bool) !struct { std.StringArrayHashMapUnmanaged([]const u8), u64 } {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var input = try std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only });
    defer input.close(io);
    var len_buf: [8]u8 = undefined;
    if (try input.readPositionalAll(io, &len_buf, 0) != 8) return error.InvalidSafetensors;
    const header_len = std.mem.readInt(u64, &len_buf, .little);
    const header_buf = try arena.alloc(u8, @intCast(header_len));
    if (try input.readPositionalAll(io, header_buf, 8) != header_buf.len) return error.InvalidSafetensors;
    const header = try std.json.parseFromSliceLeaky(std.json.Value, arena, header_buf, .{});
    var tensors: std.StringHashMapUnmanaged(Tensor) = .empty;
    var hit = header.object.iterator();
    while (hit.next()) |entry| {
        if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;
        try tensors.put(arena, entry.key_ptr.*, try tensorFromValue(arena, entry.key_ptr.*, entry.value_ptr.*, 8 + header_len));
    }

    var groups: std.StringArrayHashMapUnmanaged(Group) = .empty;
    var expert_inputs: std.StringHashMapUnmanaged(void) = .empty;
    for (keys) |key| if (parseExpertKey(key)) |parsed| {
        const group_name = try std.fmt.allocPrint(arena, "{s}.{s}", .{ parsed.prefix, parsed.leaf });
        const gop = try groups.getOrPut(arena, group_name);
        if (!gop.found_existing) gop.value_ptr.* = .{ .prefix = parsed.prefix, .leaf = parsed.leaf };
        while (gop.value_ptr.names.items.len <= parsed.expert) try gop.value_ptr.names.append(arena, "");
        if (gop.value_ptr.names.items[parsed.expert].len != 0) return error.DuplicateExpert;
        gop.value_ptr.names.items[parsed.expert] = key;
        try expert_inputs.put(arena, key, {});
    };
    if (groups.count() == 0) {
        var unchanged: std.StringArrayHashMapUnmanaged([]const u8) = .empty;
        for (keys) |key| try unchanged.put(allocator, try allocator.dupe(u8, key), try allocator.dupe(u8, path));
        var total: u64 = 0;
        for (keys) |key| total += (tensors.get(key) orelse return error.IndexMismatch).size;
        return .{ unchanged, total };
    }
    const count = expected_experts orelse groups.values()[0].names.items.len;
    for (groups.values()) |group| {
        if (group.names.items.len != count) return error.ExpertCountMismatch;
        for (group.names.items) |name| if (name.len == 0) return error.NonContiguousExperts;
    }

    var outputs: std.ArrayList(Output) = .empty;
    for (keys) |key| if (!expert_inputs.contains(key)) {
        const source = tensors.get(key) orelse return error.IndexMismatch;
        const sources = try arena.alloc(Tensor, 1);
        sources[0] = source;
        try outputs.append(arena, .{ .name = key, .dtype = source.dtype, .shape = source.shape, .sources = sources });
    };
    var consumed: std.StringHashMapUnmanaged(void) = .empty;
    for (groups.values()) |group| {
        if (consumed.contains(try std.fmt.allocPrint(arena, "{s}.{s}", .{ group.prefix, group.leaf }))) continue;
        const is_w1 = std.mem.startsWith(u8, group.leaf, "w1.");
        if (is_w1) {
            const suffix = group.leaf[3..];
            const right_name = try std.fmt.allocPrint(arena, "{s}.w3.{s}", .{ group.prefix, suffix });
            const right = groups.get(right_name) orelse return error.MissingW3;
            var sources = try arena.alloc(Tensor, count * 2);
            for (0..count) |i| {
                sources[2 * i] = tensors.get(group.names.items[i]) orelse return error.IndexMismatch;
                sources[2 * i + 1] = tensors.get(right.names.items[i]) orelse return error.IndexMismatch;
                if (!sameShape(sources[2 * i], sources[2 * i + 1]) or !sameShape(sources[0], sources[2 * i])) return error.TensorMismatch;
            }
            if (sources[0].shape.len == 0) return error.InvalidShape;
            const shape = try arena.alloc(i64, sources[0].shape.len + 1);
            shape[0] = @intCast(count);
            shape[1] = sources[0].shape[0] * 2;
            @memcpy(shape[2..], sources[0].shape[1..]);
            try outputs.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}.w13.{s}", .{ group.prefix, suffix }), .dtype = sources[0].dtype, .shape = shape, .sources = sources, .interleave = true });
            try consumed.put(arena, right_name, {});
        } else if (!std.mem.startsWith(u8, group.leaf, "w3.")) {
            const sources = try arena.alloc(Tensor, count);
            for (0..count) |i| {
                sources[i] = tensors.get(group.names.items[i]) orelse return error.IndexMismatch;
                if (!sameShape(sources[0], sources[i])) return error.TensorMismatch;
            }
            const shape = try arena.alloc(i64, sources[0].shape.len + 1);
            shape[0] = @intCast(count);
            @memcpy(shape[1..], sources[0].shape);
            try outputs.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}.{s}", .{ group.prefix, group.leaf }), .dtype = sources[0].dtype, .shape = shape, .sources = sources });
        }
    }
    std.mem.sort(Output, outputs.items, {}, struct {
        fn less(_: void, a: Output, b: Output) bool {
            return std.mem.lessThan(u8, a.name, b.name);
        }
    }.less);
    const encoded = try encodeHeader(arena, outputs.items, header.object.get("__metadata__"));
    const tmp_path = try std.fmt.allocPrint(arena, "{s}.repack.tmp", .{path});
    var output = try std.Io.Dir.createFile(.cwd(), io, tmp_path, .{ .truncate = true });
    errdefer std.Io.Dir.deleteFile(.cwd(), io, tmp_path) catch {};
    defer output.close(io);
    std.mem.writeInt(u64, &len_buf, encoded.len, .little);
    try output.writePositionalAll(io, &len_buf, 0);
    try output.writePositionalAll(io, encoded, 8);
    const buffer = try allocator.alloc(u8, buffer_size);
    defer allocator.free(buffer);
    var output_offset: u64 = 8 + encoded.len;
    for (outputs.items) |item| {
        if (verbose) std.debug.print("  {s}: {any} {s}\n", .{ item.name, item.shape, item.dtype });
        if (!item.interleave) {
            for (item.sources) |source| try copyRange(io, input, output, &output_offset, source.offset, source.size, buffer);
        } else {
            for (0..count) |i| {
                const left = item.sources[2 * i];
                const right = item.sources[2 * i + 1];
                try interleaveRows(io, input, output, &output_offset, left, right, buffer);
            }
        }
    }
    try std.Io.Dir.rename(.cwd(), tmp_path, .cwd(), path, io);
    var map: std.StringArrayHashMapUnmanaged([]const u8) = .empty;
    var total: u64 = 0;
    for (outputs.items) |item| {
        try map.put(allocator, try allocator.dupe(u8, item.name), try allocator.dupe(u8, path));
        total += item.size();
    }
    return .{ map, total };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var input_dir: []const u8 = ".";
    var expected_experts: ?usize = null;
    var buffer_mb: usize = 8;
    var verbose = false;
    var args = init.minimal.args.iterate();
    _ = args.next();
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--input-dir")) input_dir = args.next() orelse return error.MissingArgument else if (std.mem.eql(u8, arg, "--num-experts")) expected_experts = try std.fmt.parseInt(usize, args.next() orelse return error.MissingArgument, 10) else if (std.mem.eql(u8, arg, "--buffer-size-mb")) buffer_mb = try std.fmt.parseInt(usize, args.next() orelse return error.MissingArgument, 10) else if (std.mem.eql(u8, arg, "--verbose")) verbose = true else if (std.mem.eql(u8, arg, "--help")) {
            std.debug.print("usage: repack [--input-dir DIR] [--num-experts N] [--buffer-size-mb N] [--verbose]\nRewrites a DeepSeek V4 sharded safetensors checkpoint in place.\n", .{});
            return;
        } else return error.UnknownArgument;
    }
    if (buffer_mb == 0) return error.InvalidBufferSize;
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const index_path = try std.fs.path.join(arena, &.{ input_dir, "model.safetensors.index.json" });
    const index_data = try readFile(arena, io, index_path);
    var index = try std.json.parseFromSliceLeaky(std.json.Value, arena, index_data, .{});
    const weight_map = (index.object.get("weight_map") orelse return error.MissingWeightMap).object;
    var shards: std.StringArrayHashMapUnmanaged(std.ArrayList([]const u8)) = .empty;
    var wit = weight_map.iterator();
    while (wit.next()) |entry| {
        const gop = try shards.getOrPut(arena, entry.value_ptr.string);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        try gop.value_ptr.append(arena, entry.key_ptr.*);
    }
    if (expected_experts == null) {
        const config_path = try std.fs.path.join(arena, &.{ input_dir, "config.json" });
        if (readFile(arena, io, config_path)) |data| {
            const config = try std.json.parseFromSliceLeaky(std.json.Value, arena, data, .{});
            if (config.object.get("n_routed_experts")) |n| expected_experts = @intCast(n.integer);
        } else |_| {}
    }
    var new_map: std.json.ObjectMap = .empty;
    var total: u64 = 0;
    for (shards.keys(), shards.values()) |shard, keys| {
        const path = try std.fs.path.join(arena, &.{ input_dir, shard });
        var result = try rewriteShard(allocator, io, path, keys.items, expected_experts, buffer_mb * 1024 * 1024, verbose);
        defer result[0].deinit(allocator);
        var mit = result[0].iterator();
        while (mit.next()) |entry| {
            try new_map.put(arena, try arena.dupe(u8, entry.key_ptr.*), .{ .string = shard });
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        total += result[1];
        std.debug.print("Repacked {s}\n", .{shard});
    }
    try index.object.put(arena, "weight_map", .{ .object = new_map });
    var metadata = index.object.get("metadata") orelse std.json.Value{ .object = .empty };
    try metadata.object.put(arena, "total_size", .{ .integer = @intCast(total) });
    try index.object.put(arena, "metadata", metadata);
    var json: std.Io.Writer.Allocating = .init(arena);
    try json.writer.print("{f}\n", .{std.json.fmt(index, .{ .whitespace = .indent_2 })});
    const tmp_index = try std.fmt.allocPrint(arena, "{s}.repack.tmp", .{index_path});
    var file = try std.Io.Dir.createFile(.cwd(), io, tmp_index, .{ .truncate = true });
    defer file.close(io);
    try file.writePositionalAll(io, json.written(), 0);
    try std.Io.Dir.rename(.cwd(), tmp_index, .cwd(), index_path, io);
    std.debug.print("Done. Rewrote checkpoint in place at {s}\n", .{input_dir});
}
