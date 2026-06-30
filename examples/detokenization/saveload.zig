const std = @import("std");
const zml = @import("zml");

const main = @import("main.zig");
const graph = @import("graph.zig");
const model_ = @import("model.zig");
const llm_ = @import("llm.zig");
const svd_ = @import("svd.zig");
const inference = @import("inference.zig");

const Zml_handler = main.Zml_handler;
const Tokenizer = zml.tokenizer.Tokenizer;

pub fn loadSafetensorSliceFromRegistry(zml_handler: *Zml_handler, registry: *zml.safetensors.TensorRegistry, tensor_name: []const u8) !zml.Slice {
    const tensor = registry.tensors.get(tensor_name) orelse return error.TensorNotFound;
    const slice = try zml.Slice.alloc(zml_handler.allocator, tensor.shape);
    errdefer slice.free(zml_handler.allocator);

    const io_buffer = try zml_handler.allocator.alloc(u8, 128 * 1024 * 1024);
    defer zml_handler.allocator.free(io_buffer);
    var reader = try registry.reader(zml_handler.io, tensor_name, io_buffer);
    defer reader.deinit();
    try readSliceDataChunked(&reader.interface, slice.data());
    return slice;
}

pub fn readSliceDataChunked(reader: *std.Io.Reader, data: []u8) !void {
    const max_read_chunk = 512 * 1024 * 1024;
    var offset: usize = 0;
    while (offset < data.len) {
        const chunk_len = @min(max_read_chunk, data.len - offset);
        _ = try reader.readSliceAll(data[offset..][0..chunk_len]);
        offset += chunk_len;
    }
}

pub fn loadSafetensorSlice(zml_handler: *Zml_handler, repo_uri: []const u8, entrypoint_name: []const u8, tensor_name: []const u8) !zml.Slice {
    std.log.info("Load slice {s} from repo {s} entrypoint {s}", .{ tensor_name, repo_uri, entrypoint_name });
    const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, repo_uri);
    var registry: zml.safetensors.TensorRegistry = try .fromRepoFile(zml_handler.allocator, zml_handler.io, repo, entrypoint_name);
    defer registry.deinit();

    const tensor = registry.tensors.get(tensor_name) orelse return error.TensorNotFound;
    std.log.info("Tensor shape: {any}", .{tensor.shape.dtype()});
    const slice = try zml.Slice.alloc(zml_handler.allocator, tensor.shape);
    errdefer slice.free(zml_handler.allocator);

    const io_buffer = try zml_handler.allocator.alloc(u8, 128 * 1024 * 1024);
    defer zml_handler.allocator.free(io_buffer);
    var reader = try registry.reader(zml_handler.io, tensor_name, io_buffer);
    defer reader.deinit();

    try readSliceDataChunked(&reader.interface, slice.data());
    return slice;
}


pub fn getSlice(zml_handler: *Zml_handler, file_name: []const u8, tensor_name: []const u8, dtype: zml.dtype.DataType, transpose: bool) !zml.Slice {
    //std.log.info("Getting slice {s}", .{tensor_name});

    //std.log.info("Init store", .{});
    const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.checkpoint);
    var registry: zml.safetensors.TensorRegistry = try .fromRepoFile(zml_handler.allocator, zml_handler.io, repo, file_name);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(zml_handler.allocator, &registry);
    defer store.deinit();

    //std.log.info("Init extractor", .{});
    const model: TensorExtractor = .init(store.view(), tensor_name, transpose);

    //std.log.info("Compile extract", .{});
    const extract_exe = try zml_handler.platform.compile(zml_handler.allocator, zml_handler.io, model, .forward, .{ dtype }, .{});
    defer extract_exe.deinit();

    //std.log.info("Load weights", .{});
    var model_buffers = try model.load(zml_handler, &store);
    defer TensorExtractor.unloadBuffers(&model_buffers);

    //std.log.info("Init slice and buffer", .{});
    const slice: zml.Slice = try .alloc(zml_handler.allocator, .init(model.shape(dtype), dtype));
    var buffer: zml.Buffer = try .fromSlice(zml_handler.io, zml_handler.platform, slice, .replicated);
    defer buffer.deinit();

    var extract_args = try extract_exe.args(zml_handler.allocator);
    defer extract_args.deinit(zml_handler.allocator);
    var extract_results = try extract_exe.results(zml_handler.allocator);
    defer extract_results.deinit(zml_handler.allocator);

    //std.log.info("Call extract", .{});
    extract_args.set(.{model_buffers});
    extract_exe.call(extract_args, &extract_results);
    extract_results.fill(.{&buffer});

    try buffer.toSlice(zml_handler.io, slice);
    //std.log.info("Return slice", .{});
    return slice;
}

const TensorExtractor = struct {
    tensor: zml.Tensor,
    transpose: bool,
    pub fn init(store: zml.io.TensorStore.View, tensor_name: []const u8, transpose: bool) TensorExtractor {
        return .{
            .tensor = store.createTensor(tensor_name, .{ .d, .n }, .{ .d = .replicated, .n = .replicated }),
            .transpose = transpose,
        };
    }
    pub fn load(self: *const TensorExtractor, zml_handler: *Zml_handler, store: *zml.io.TensorStore) !zml.Bufferized(TensorExtractor) {
        return zml.io.load(TensorExtractor, self, zml_handler.arena.allocator(), zml_handler.io, zml_handler.platform, store, .{
            .parallelism = 16,
            .dma_chunks = 1,
            .dma_chunk_size = 128 * 1024 * 1024,
        });
    }
    pub fn unloadBuffers(self: *zml.Bufferized(TensorExtractor)) void {
        self.tensor.deinit();
    }
    pub fn shape(self: TensorExtractor, dtype: zml.dtype.DataType) zml.Shape {
        return if (self.transpose) zml.Shape.init(.{ .n = self.tensor.shape().dim(1), .d = self.tensor.shape().dim(0) }, dtype) else self.tensor.shape();
    }
    pub fn forward(self: TensorExtractor, dtype: zml.dtype.DataType) zml.Tensor {
        return if (self.transpose) self.tensor.transpose(.{ .n, .d }).convert(dtype) else self.tensor.convert(dtype);
    }
};


pub fn printSafetensors(registry: zml.safetensors.TensorRegistry) !void {
    const tensors: zml.safetensors.Tensors = registry.tensors;
    const data = tensors.entries;
    for (0..data.len) |i| {
        const entry = data.get(i);
        const tensor: zml.safetensors.Tensor = tensors.get(entry.key).?;
        std.log.info("Tensor(name={s} shape={f} size={d})", .{
            tensor.name,
            tensor.shape,
            tensor.byteSize(),
        });
    }
}

pub fn parseConfig(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(T) {
    const file = try dir.openFile(io, "config.json", .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    return try std.json.parseFromTokenSource(T, allocator, &reader, .{ .ignore_unknown_fields = true });
}
