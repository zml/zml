const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;
const attention = zml.attention.attention;

const model = @import("model.zig");

pub fn run(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, config: model.Config, mdl: model.Model, model_buffers: *zml.Bufferized(model.Model), store: *zml.io.TensorStore, attention_metadata: attention.Metadata, attention_parameters: attention.Parameters) !void {
    _ = store;
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, "/Users/brabier/github/huggingface/transformers/lfm_activations.safetensors");
    defer registry.deinit();

    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer activation_store.deinit();

    var ctx = TestContext{ .allocator = allocator, .io = io, .platform = platform, .activations_store = &activation_store, .attention_metadata = attention_metadata, .attention_parameters = attention_parameters };

    try ctx.testLayer("embed_tokens", .{ .batch, .seq }, mdl.embed_tokens, model_buffers.embed_tokens, .{});

    var num_attn_layers: usize = 0;
    var num_conv_layers: usize = 0;
    for (0..config.num_hidden_layers) |i| {
        const kind = std.meta.stringToEnum(model.OperatorKind, config.layer_types[i]) orelse {
            stdx.debug.assert(false, "Unsupported layer type {s}", .{config.layer_types[i]});
            unreachable;
        };
        const layer = mdl.layers[i];
        const layer_buffers = model_buffers.layers[i];
        switch (kind) {
            .conv => {
                try ctx.testLayerPrint("layers.{d}.conv.in_proj", .{i}, .{ .batch, .seq, .d }, layer.operator.conv.in_proj, layer_buffers.operator.conv.in_proj, .{});
                try ctx.testLayerPrint("layers.{d}.conv.out_proj", .{i}, .{ .batch, .seq, .d }, layer.operator.conv.out_proj, layer_buffers.operator.conv.out_proj, .{});
                try ctx.testLayerPrint("layers.{d}.feed_forward.w1", .{i}, .{ .batch, .seq, .d }, layer.feed_forward.w1, layer_buffers.feed_forward.w1, .{});
                try ctx.testLayerPrint("layers.{d}.feed_forward.w2", .{i}, .{ .batch, .seq, .d }, layer.feed_forward.w2, layer_buffers.feed_forward.w2, .{});
                try ctx.testLayerPrint("layers.{d}.feed_forward.w3", .{i}, .{ .batch, .seq, .d }, layer.feed_forward.w3, layer_buffers.feed_forward.w3, .{});
                try ctx.testLayerPrint("layers.{d}.feed_forward", .{i}, .{ .batch, .seq, .d }, layer.feed_forward, layer_buffers.feed_forward, .{});
                try ctx.testConvLayer(i, num_conv_layers, mdl.layers[i].operator.conv, model_buffers.layers[i].operator.conv, .{});
                num_conv_layers += 1;
            },
            .full_attention => {
                try ctx.testLayerPrint("layers.{d}.self_attn.k_proj", .{i}, .{ .batch, .seq, .d }, layer.operator.self_attn.k_proj, layer_buffers.operator.self_attn.k_proj, .{});
                try ctx.testLayerPrint("layers.{d}.self_attn.q_proj", .{i}, .{ .batch, .seq, .d }, layer.operator.self_attn.q_proj, layer_buffers.operator.self_attn.q_proj, .{});
                try ctx.testLayerPrint("layers.{d}.self_attn.out_proj", .{i}, .{ .batch, .seq, .d }, layer.operator.self_attn.out_proj, layer_buffers.operator.self_attn.out_proj, .{});
                try ctx.testLayerPrint("layers.{d}.self_attn.v_proj", .{i}, .{ .batch, .seq, .d }, layer.operator.self_attn.v_proj, layer_buffers.operator.self_attn.v_proj, .{});
                try ctx.testLayerPrint("layers.{d}.feed_forward.w1", .{i}, .{ .batch, .seq, .d }, layer.feed_forward.w1, layer_buffers.feed_forward.w1, .{});
                try ctx.testLayerPrint("layers.{d}.feed_forward.w2", .{i}, .{ .batch, .seq, .d }, layer.feed_forward.w2, layer_buffers.feed_forward.w2, .{});
                try ctx.testLayerPrint("layers.{d}.feed_forward.w3", .{i}, .{ .batch, .seq, .d }, layer.feed_forward.w3, layer_buffers.feed_forward.w3, .{});
                try ctx.testLayerPrint("layers.{d}.feed_forward", .{i}, .{ .batch, .seq, .d }, layer.feed_forward, layer_buffers.feed_forward, .{});
                try ctx.testAttnLayer(i, num_attn_layers, mdl.layers[i].operator.self_attn, model_buffers.layers[i].operator.self_attn, .{});
                num_attn_layers += 1;
            },
        }
    }
}

const TestContext = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activations_store: *zml.io.TensorStore,
    attention_metadata: attention.Metadata,
    attention_parameters: attention.Parameters,

    fn testLayerPrint(
        self: *TestContext,
        comptime name_fmt: []const u8,
        name_args: anytype,
        tagz: anytype,
        layer: anytype,
        layer_buffers: anytype,
        opts: zml.testing.CompareOpts,
    ) !void {
        const name = try std.fmt.allocPrint(self.allocator, name_fmt, name_args);
        defer self.allocator.free(name);
        try self.testLayer(name, tagz, layer, layer_buffers, opts);
    }

    fn testLayer(self: *TestContext, name: []const u8, tagz: anytype, layer: anytype, layer_buffers: anytype, opts: zml.testing.CompareOpts) !void {
        std.log.info("Testing layer: {s}", .{name});

        const in_key = try std.fmt.allocPrint(self.allocator, "{s}.in", .{name});
        defer self.allocator.free(in_key);
        const in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, in_key);
        const in_tensor = zml.Tensor.fromShape(in_buffer.shape()).withTags(tagz);

        const out_key = try std.fmt.allocPrint(self.allocator, "{s}.out", .{name});
        defer self.allocator.free(out_key);
        const out_buffer_expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, out_key);

        const exe = try self.platform.compileFn(self.allocator, self.io, @TypeOf(layer).forward, .{ layer, in_tensor });
        defer exe.deinit();

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{ layer_buffers, in_buffer });

        var res = try exe.results(self.allocator);
        defer res.deinit(self.allocator);

        exe.call(args, &res);

        const out_result = res.get(zml.Buffer);

        try zml.testing.expectClose(self.io, out_result, out_buffer_expected, opts);
        std.log.info("Layer {s} passed!", .{name});
    }

    fn testConvLayer(self: *TestContext, ix: usize, cache_ix: usize, layer: model.ShortConv, layer_buffers: zml.Bufferized(model.ShortConv), opts: zml.testing.CompareOpts) !void {
        const name = try std.fmt.allocPrint(self.allocator, "layers.{d}.conv", .{ix});
        defer self.allocator.free(name);
        std.log.info("Testing layer: {s}", .{name});

        const in_key = try std.fmt.allocPrint(self.allocator, "{s}.in", .{name});
        defer self.allocator.free(in_key);
        const in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, in_key);
        const in_tensor = zml.Tensor.fromShape(in_buffer.shape()).withTags(.{ .batch, .seq, .d });

        const cache_key = try std.fmt.allocPrint(self.allocator, "{s}.cache", .{name});
        defer self.allocator.free(cache_key);
        const cache_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, cache_key);
        const cache_tensor = zml.Tensor.fromShape(cache_buffer.shape()).withTags(.{ .layer, .batch, .seq, .d });

        const cache_pos_key = try std.fmt.allocPrint(self.allocator, "{s}.cache_position", .{name});
        defer self.allocator.free(cache_pos_key);
        const cache_pos_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, cache_pos_key);
        const cache_pos_tensor = zml.Tensor.fromShape(cache_pos_buffer.shape()).withTags(.{.batch});

        const out_key = try std.fmt.allocPrint(self.allocator, "{s}.out", .{name});
        defer self.allocator.free(out_key);
        const out_buffer_expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, out_key);

        const actual_seq_len_tensor: zml.Tensor = .init(.{}, .u32);
        const actual_seq_len: u32 = @intCast(in_tensor.dim(.seq));

        const cache_index_tensor: zml.Tensor = .init(.{}, .u32);

        const exe = try self.platform.compileFn(self.allocator, self.io, @TypeOf(layer).forward, .{ layer, in_tensor, cache_pos_tensor, actual_seq_len_tensor, model.ConvCache{ .state = cache_tensor }, cache_index_tensor, model.ConvParameters{ .is_prefill = false } });
        defer exe.deinit();

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        const conv_cache: zml.Bufferized(model.ConvCache) = .{ .state = cache_buffer };
        const actual_seq_len_slice: zml.Slice = .init(zml.Shape.init(.{}, .u32), std.mem.sliceAsBytes(&[_]u32{actual_seq_len}));
        var actual_seq_len_buf: zml.Buffer = try .fromSlice(self.io, self.platform, actual_seq_len_slice);
        defer actual_seq_len_buf.deinit();
        const cache_index_slice: zml.Slice = .init(zml.Shape.init(.{}, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(cache_ix)}));
        var cache_index_buf: zml.Buffer = try .fromSlice(self.io, self.platform, cache_index_slice);
        defer cache_index_buf.deinit();
        args.set(.{ layer_buffers, in_buffer, cache_pos_buffer, actual_seq_len_buf, conv_cache, cache_index_buf });

        var res = try exe.results(self.allocator);
        defer res.deinit(self.allocator);

        exe.call(args, &res);

        const out_result = res.get(zml.Buffer);

        try zml.testing.expectClose(self.io, out_result, out_buffer_expected, opts);
        std.log.info("Layer {s} passed!", .{name});
    }

    fn testAttnLayer(self: *TestContext, ix: usize, cache_ix: usize, layer: model.Attention, layer_buffers: zml.Bufferized(model.Attention), opts: zml.testing.CompareOpts) !void {
        const name = try std.fmt.allocPrint(self.allocator, "layers.{d}.self_attn", .{ix});
        defer self.allocator.free(name);
        std.log.info("Testing layer: {s}", .{name});

        const in_key = try std.fmt.allocPrint(self.allocator, "{s}.in", .{name});
        defer self.allocator.free(in_key);
        const in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, in_key);
        const in_tensor = zml.Tensor.fromShape(in_buffer.shape()).withTags(.{ .batch, .seq, .d });

        const key_cache_key = try std.fmt.allocPrint(self.allocator, "{s}.cache.key", .{name});
        defer self.allocator.free(key_cache_key);
        const key_cache_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, key_cache_key);
        const key_cache_tensor = zml.Tensor.fromShape(key_cache_buffer.shape()).withTags(.{ .layer, .batch, .h, .k, .hd });

        const value_cache_key = try std.fmt.allocPrint(self.allocator, "{s}.cache.value", .{name});
        defer self.allocator.free(value_cache_key);
        const value_cache_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, value_cache_key);
        const value_cache_tensor = zml.Tensor.fromShape(value_cache_buffer.shape()).withTags(.{ .layer, .batch, .h, .k, .hd });

        const cache_pos_key = try std.fmt.allocPrint(self.allocator, "{s}.cache_position", .{name});
        defer self.allocator.free(cache_pos_key);
        const cache_pos_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, cache_pos_key);
        const cache_pos_tensor = zml.Tensor.fromShape(cache_pos_buffer.shape()).withTags(.{.batch});

        const out_key = try std.fmt.allocPrint(self.allocator, "{s}.out", .{name});
        defer self.allocator.free(out_key);
        const out_buffer_expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, out_key);

        const cache_index_tensor: zml.Tensor = .init(.{}, .u32);

        const exe = try self.platform.compileFn(self.allocator, self.io, @TypeOf(layer).forward, .{ layer, in_tensor, cache_pos_tensor, model.KvCache{ .k = key_cache_tensor, .v = value_cache_tensor }, cache_index_tensor, self.attention_metadata, self.attention_parameters });
        defer exe.deinit();

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        const kv_cache: zml.Bufferized(model.KvCache) = .{ .k = key_cache_buffer, .v = value_cache_buffer };
        var attention_metadata_buffers = try self.attention_metadata.initBuffer(self.io, self.platform);
        defer attention.Metadata.deinitBuffer(&attention_metadata_buffers);
        const cache_index_slice: zml.Slice = .init(zml.Shape.init(.{}, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(cache_ix)}));
        var cache_index_buf: zml.Buffer = try .fromSlice(self.io, self.platform, cache_index_slice);
        defer cache_index_buf.deinit();
        args.set(.{ layer_buffers, in_buffer, cache_pos_buffer, kv_cache, cache_index_buf, attention_metadata_buffers });

        var res = try exe.results(self.allocator);
        defer res.deinit(self.allocator);

        exe.call(args, &res);

        var out_result, var updated_kv = res.get(struct { zml.Buffer, zml.Bufferized(model.KvCache) });
        defer out_result.deinit();
        defer model.KvCache.unloadBuffers(&updated_kv);

        try zml.testing.expectClose(self.io, out_result, out_buffer_expected, opts);
        std.log.info("Layer {s} passed!", .{name});
    }
};

fn loadBufferFromStore(allocator: std.mem.Allocator, io: anytype, platform: *zml.Platform, store: *zml.io.TensorStore, key: []const u8) !zml.Buffer {
    const shape = store.view().getShape(key) orelse return error.NotFound;

    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.view().getReader(key, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(host_bytes);

    return zml.Buffer.fromBytes(io, platform, shape, host_bytes);
}
