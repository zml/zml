const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const dflash = @import("dflash_model.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const log = std.log.scoped(.dflash);

const block_size = 16;
const first_token: u32 = 151_644;
const placeholder_token: u32 = 151_643;

const Args = struct {
    model: []const u8,
    target_model: []const u8,

    pub const help =
        \\Use dflash --model=<path> --target-model=<path>
        \\
        \\Compile the DFlash draft model, apply the target lm_head, and print greedy tokens.
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
        var working_dir = try std.Io.Dir.openDirAbsolute(init.io, build_working_directory, .{});
        defer working_dir.close(init.io);
        try std.process.setCurrentDir(init.io, working_dir);
    }

    const args = stdx.flags.parse(init.minimal.args, Args);

    var vfs_file: zml.io.VFS.File = .init(allocator, init.io, .{});
    defer vfs_file.deinit();

    var http_client: std.http.Client = .{ .allocator = allocator, .io = init.io };
    defer http_client.deinit();

    var hf_vfs: zml.io.VFS.HF = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer hf_vfs.deinit();

    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());
    try vfs.register("hf", hf_vfs.io());

    const io = vfs.io();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    log.info("\n{f}", .{platform.fmtVerbose()});

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    const target_repo = try zml.safetensors.resolveModelRepo(io, args.target_model);

    const parsed_config = try parseConfig(dflash.Config, allocator, io, repo);
    defer parsed_config.deinit();

    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    var target_registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, target_repo);
    defer target_registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var target_store: zml.io.TensorStore = .fromRegistry(allocator, &target_registry);
    defer target_store.deinit();

    var tokenizer = try loadTokenizer(allocator, io, target_repo);
    defer tokenizer.deinit();

    const draft_model = try dflash.Model.init(allocator, store.view(), parsed_config.value);
    defer draft_model.deinit(allocator);

    const target_model: TargetModel = .init(target_store.view());

    const shardings: Shardings = try .init(platform);
    const all_shardings = shardings.all();

    const target_hidden_tensor: zml.Tensor = .init(.{ .s = block_size, .d = draft_model.fc.weight.dim(.d) }, .f32);
    const noise_embedding_tensor: zml.Tensor = .init(.{ .s = block_size, .d = draft_model.fc.weight.dim(.dout) }, .f32);
    const position_ids_tensor: zml.Tensor = .init(.{ .s = block_size * 2 }, .u32);

    log.info("Compiling model...", .{});
    var exe = try platform.compileFn(
        allocator,
        io,
        forwardTokens,
        .{ draft_model, target_model, target_hidden_tensor, noise_embedding_tensor, position_ids_tensor },
        .{ .shardings = &all_shardings },
    );
    defer exe.deinit();

    log.info("Loading weights...", .{});
    var model_buffers = try zml.io.load(dflash.Model, &draft_model, allocator, io, platform, &store, .{
        .parallelism = 16,
        .shardings = &all_shardings,
        .dma_chunks = 32,
        .dma_chunk_size = 128 * zml.MiB,
    });
    defer dflash.Model.unloadBuffers(&model_buffers, allocator);

    var target_buffers = try zml.io.load(TargetModel, &target_model, allocator, io, platform, &target_store, .{
        .parallelism = 1,
        .shardings = &all_shardings,
        .dma_chunks = 4,
        .dma_chunk_size = 128 * zml.MiB,
    });
    defer TargetModel.unloadBuffers(&target_buffers);

    var token_ids: [block_size]u32 = @splat(placeholder_token);
    token_ids[0] = first_token;

    var target_hidden = try zml.Slice.alloc(allocator, target_hidden_tensor.shape());
    defer target_hidden.free(allocator);
    @memset(target_hidden.items(f32), 0);

    var noise_embedding = try zml.Slice.alloc(allocator, noise_embedding_tensor.shape());
    defer noise_embedding.free(allocator);
    fillNoiseEmbedding(noise_embedding.items(f32), &token_ids, noise_embedding_tensor.dim(.d));

    var position_ids: [block_size * 2]u32 = undefined;
    for (&position_ids, 0..) |*pos, i| pos.* = @intCast(i);

    var target_hidden_buffer: zml.Buffer = try .fromSlice(io, platform, target_hidden, .replicated);
    defer target_hidden_buffer.deinit();

    var noise_embedding_buffer: zml.Buffer = try .fromSlice(io, platform, noise_embedding, .replicated);
    defer noise_embedding_buffer.deinit();

    var position_ids_buffer: zml.Buffer = try .fromSlice(
        io,
        platform,
        zml.Slice.init(position_ids_tensor.shape(), std.mem.asBytes(&position_ids)),
        .replicated,
    );
    defer position_ids_buffer.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    exe_args.set(.{ model_buffers, target_buffers, target_hidden_buffer, noise_embedding_buffer, position_ids_buffer });
    exe.call(exe_args, &results);

    var tokens: zml.Buffer = results.get(zml.Buffer);
    defer tokens.deinit();

    var tokens_slice = try tokens.toSliceAlloc(allocator, io);
    defer tokens_slice.free(allocator);

    var stdout = std.Io.File.stdout().writerStreaming(io, &.{});
    try printTokens(allocator, &tokenizer, &stdout.interface, "input", &token_ids);
    try stdout.interface.print("sampled_tokens shape: {f}\n", .{tokens_slice.shape});

    const sampled_tokens = tokens_slice.constItems(u32);
    try printTokens(allocator, &tokenizer, &stdout.interface, "sampled", sampled_tokens);
    try stdout.interface.flush();
}

fn forwardTokens(
    draft_model: dflash.Model,
    target_model: TargetModel,
    target_hidden: zml.Tensor,
    noise_embedding: zml.Tensor,
    position_ids: zml.Tensor,
) zml.Tensor {
    const hidden = draft_model.forward(target_hidden, noise_embedding, position_ids);
    const logits = target_model.lm_head.forward(hidden)
        .rename(.{ .dout = .voc });
    return logits.argMax(.voc).indices.squeeze(.voc).convert(.u32);
}

const TargetModel = struct {
    lm_head: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) TargetModel {
        return .{
            .lm_head = .init(
                store.withPrefix("lm_head").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                null,
                .d,
            ),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TargetModel)) void {
        self.lm_head.weight.deinit();
        if (self.lm_head.bias) |*bias| bias.deinit();
    }
};

fn fillNoiseEmbedding(out: []f32, token_ids: *const [block_size]u32, hidden_size: i64) void {
    const hidden: usize = @intCast(hidden_size);
    for (token_ids, 0..) |token_id, token_index| {
        const base = token_index * hidden;
        const token_value = @as(f32, @floatFromInt(token_id % 1024)) / 1024.0;
        for (out[base .. base + hidden], 0..) |*value, dim| {
            value.* = token_value + (@as(f32, @floatFromInt(dim % 17)) * 0.0001);
        }
    }
}

fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !zml.tokenizer.Tokenizer {
    const file = try dir.openFile(io, "tokenizer.json", .{});
    defer file.close(io);

    var reader = file.reader(io, &.{});
    const bytes = try reader.interface.readAlloc(allocator, try file.length(io));
    defer allocator.free(bytes);

    return try .fromBytes(allocator, bytes);
}

fn printTokens(
    allocator: std.mem.Allocator,
    tokenizer: *const zml.tokenizer.Tokenizer,
    stdout: *std.Io.Writer,
    comptime name: []const u8,
    token_ids: []const u32,
) !void {
    try stdout.print("{s}_ids: {any}\n", .{ name, token_ids });

    var decoder = try tokenizer.decoder();
    defer decoder.deinit();
    var text = try decoder.decodeAlloc(allocator, token_ids);
    defer text.deinit(allocator);
    try stdout.print("{s}_text: {s}\n", .{ name, text.items });

    try stdout.print("{s}_tokens: [", .{name});
    for (token_ids, 0..) |token_id, i| {
        if (i != 0) try stdout.writeAll(", ");
        const one = [1]u32{token_id};
        var piece_decoder = try tokenizer.decoder();
        defer piece_decoder.deinit();
        var piece = try piece_decoder.decodeAlloc(allocator, &one);
        defer piece.deinit(allocator);
        try stdout.print("{}=\"{s}\"", .{ token_id, piece.items });
    }
    try stdout.writeAll("]\n");
}

const Shardings = struct {
    model: zml.Sharding,

    pub fn init(platform: *zml.Platform) !Shardings {
        return .{
            .model = try platform.registerSharding("model", .mesh(.{ .model = .high_bandwidth })),
        };
    }

    pub fn all(self: Shardings) [1]zml.Sharding {
        return .{self.model};
    }
};

fn parseConfig(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(T) {
    const file = try dir.openFile(io, "config.json", .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    return try std.json.parseFromTokenSource(T, allocator, &reader, .{ .ignore_unknown_fields = true });
}
