const std = @import("std");
const zml = @import("zml");
const common = @import("common.zig");
const model = @import("llama/model.zig");
const inference = @import("llama/inference.zig");

const Args = struct {
    model: []const u8,
    tokens: []const u8,
    output: []const u8,
    seqlen: u32 = 128,
};

const HeadInput = struct { head: model.LmHead, hidden: zml.Tensor };
const HeadOutput = struct { logits: zml.Tensor };
fn headLogits(input: HeadInput) HeadOutput {
    const hidden = input.head.norm.forward(input.hidden);
    const logits = if (input.head.lm_head) |head|
        head.forward(hidden).rename(.{ .dout = .voc })
    else
        input.head.embed_tokens.weight.withTags(.{ .voc, .d }).dot(hidden, .d);
    return .{ .logits = logits.transpose(.{ .s, .voc }).convert(.bf16) };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    if (args.seqlen == 0 or args.seqlen > 4096) return error.InvalidSequenceLength;
    const token_file = try std.Io.Dir.cwd().openFile(io, args.tokens, .{});
    defer token_file.close(io);
    var reader = token_file.reader(io, &.{});
    const token_bytes = try reader.interface.readAlloc(allocator, try token_file.length(io));
    defer allocator.free(token_bytes);
    // Each window has S input tokens plus the next-token target for position S-1.
    const window_bytes = (@as(usize, args.seqlen) + 1) * 4;
    if (token_bytes.len == 0 or token_bytes.len % window_bytes != 0) return error.InvalidTokenFile;
    const windows = token_bytes.len / window_bytes;
    const platform = try zml.Platform.auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    if (platform.target != .furiosa and platform.target != .furiosa2) return error.FuriosaRequired;
    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    defer repo.close(io);
    var registry = try zml.safetensors.TensorRegistry.fromRepo(allocator, io, repo);
    defer registry.deinit();
    var store = zml.io.TensorStore.fromRegistry(allocator, &registry);
    defer store.deinit();
    var loaded = try model.LoadedModel.init(allocator, io, repo, store.view(), .{});
    defer loaded.deinit(allocator);
    const mdl = loaded.inner;
    const vocab = mdl.model.embed_tokens.weight.dim(.voc);
    const shardings = try common.Shardings.init(platform);
    const params = inference.CompilationParameters.init(mdl, mdl.config, args.seqlen, .vanilla, shardings);
    const hidden_shape = zml.Shape.init(.{ .s = args.seqlen, .d = mdl.config.hidden_size }, .bf16).withPartitioning(.{ .d = .replicated });
    var progress = std.Progress.start(io, .{ .root_name = args.model });
    defer progress.end();
    const embed_exe = try zml.FnExe(model.EmbedTokens.forward).compile(allocator, io, platform, .{ .shardings = &shardings.all() }, .{.{
        .embedding = .{ .embed_tokens = mdl.model.embed_tokens },
        .tokens = params.prefill_tokens,
    }});
    defer embed_exe.deinit();
    const layer_exe = try zml.FnExe(model.TransformerLayer.forward).compile(allocator, io, platform, .{ .shardings = &shardings.all() }, .{.{
        .layer = mdl.model.layers[0],
        .hidden = zml.Tensor.fromShape(hidden_shape),
        .token_index = params.token_index,
        .kv_cache = params.kv_cache,
        .kv_cache_index = zml.Tensor.init(.{}, .u32),
        .attention_metadata = params.attention_metadata,
        .attention_parameters = params.prefill_attention_parameters,
    }});
    defer layer_exe.deinit();
    const head_exe = try zml.FnExe(headLogits).compile(allocator, io, platform, .{ .shardings = &shardings.all() }, .{.{
        .head = model.LmHead.init(mdl),
        .hidden = zml.Tensor.fromShape(hidden_shape),
    }});
    defer head_exe.deinit();
    var buffers = try loaded.loadBuffers(allocator, io, platform, &store, &progress, shardings);
    defer loaded.unloadBuffers(&buffers, allocator);
    var embed = try zml.FnExe(model.EmbedTokens.forward).Runner(.{.embedding}).init(&embed_exe, allocator, .{
        .embedding = .{ .embed_tokens = buffers.model.embed_tokens },
    });
    defer embed.deinit(allocator);
    const LayerRunner = zml.FnExe(model.TransformerLayer.forward).Runner(.{.layer});
    const layers = try allocator.alloc(LayerRunner, buffers.model.layers.len);
    defer allocator.free(layers);
    var initialized: usize = 0;
    defer for (layers[0..initialized]) |*layer| layer.deinit(allocator);
    for (layers, buffers.model.layers) |*layer, weights| {
        layer.* = try LayerRunner.init(&layer_exe, allocator, .{ .layer = weights });
        initialized += 1;
    }
    var head = try zml.FnExe(headLogits).Runner(.{.head}).init(&head_exe, allocator, .{
        .head = .{ .lm_head = buffers.lm_head, .embed_tokens = buffers.model.embed_tokens, .norm = buffers.model.norm },
    });
    defer head.deinit(allocator);
    const zeros = try allocator.alloc(u8, params.kv_cache.k.shape().byteSize());
    defer allocator.free(zeros);
    @memset(zeros, 0);
    var cache: model.KvCache.Buffer = blk: {
        var k = try zml.Buffer.fromBytes(io, platform, params.kv_cache.k.shape(), shardings.model, zeros);
        errdefer k.deinit();
        const v = try zml.Buffer.fromBytes(io, platform, params.kv_cache.v.shape(), shardings.model, zeros);
        break :blk .{ .k = k, .v = v };
    };
    defer model.KvCache.deinitBuffer(&cache);
    var position = try zml.Buffer.scalar(io, platform, 0, .u32);
    defer position.deinit();
    const layer_indices = try allocator.alloc(zml.Buffer, layers.len);
    defer allocator.free(layer_indices);
    var index_count: usize = 0;
    defer for (layer_indices[0..index_count]) |*index| index.deinit();
    for (layer_indices, 0..) |*index, i| {
        index.* = try zml.Buffer.scalar(io, platform, i, .u32);
        index_count += 1;
    }
    const output_file = try std.Io.Dir.cwd().createFile(io, args.output, .{ .exclusive = true });
    defer output_file.close(io);
    var write_buffer: [65536]u8 = undefined;
    var writer = output_file.writer(io, &write_buffer);
    var header: [52]u8 = undefined;
    @memcpy(header[0..8], "ZMLLGTS1");
    std.mem.writeInt(u32, header[8..12], args.seqlen, .little);
    std.mem.writeInt(u32, header[12..16], @intCast(vocab), .little);
    std.mem.writeInt(u32, header[16..20], @intCast(windows), .little);
    std.crypto.hash.sha2.Sha256.hash(token_bytes, header[20..52], .{});
    try writer.interface.writeAll(&header);
    for (0..windows) |window| {
        const bytes = token_bytes[window * window_bytes ..][0 .. args.seqlen * 4];
        for (0..args.seqlen) |i| {
            if (std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little) >= vocab) return error.InvalidToken;
        }
        var tokens = try zml.Buffer.fromBytes(io, platform, params.prefill_tokens.shape(), .replicated, bytes);
        defer tokens.deinit();
        var hidden: zml.Buffer = undefined;
        embed.run(io, .{ .inputs = .{ .tokens = tokens }, .outputs = .{ .hidden = &hidden } });
        defer hidden.deinit();
        for (layers, layer_indices) |*layer, index| {
            var previous = hidden;
            layer.run(io, .{
                .inputs = .{ .hidden = hidden, .token_index = position, .kv_cache = cache, .kv_cache_index = index, .attention_metadata = .vanilla },
                .outputs = .{ .hidden = &hidden, .kv_cache = &cache },
            });
            previous.deinit();
        }
        var logits: zml.Buffer = undefined;
        head.run(io, .{ .inputs = .{ .hidden = hidden }, .outputs = .{ .logits = &logits } });
        defer logits.deinit();
        const host = try logits.toSliceAlloc(allocator, io);
        defer host.free(allocator);
        try writer.interface.writeAll(host.bytes);
        std.log.info("Scored window {}/{} on Furiosa", .{ window + 1, windows });
    }
    try writer.interface.flush();
}
