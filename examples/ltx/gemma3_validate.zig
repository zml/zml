// Standalone validation binary for the Gemma-3 text encoder.
//
// Loads Gemma-3-12b-it weights, tokenizes a prompt, runs the encoder
// forward pass, and saves the output hidden states to a safetensors file
// for comparison against the Python reference.
//
// Usage:
//   bazel run //examples/ltx:gemma3_validate -- \
//     --gemma-ckpt /home/ubuntu/models/gemma-3-12b-it \
//     --prompt "A beautiful sunset over the ocean" \
//     --output /tmp/zig_hidden_states.safetensors

const std = @import("std");
const zml = @import("zml");
const gemma3 = @import("gemma3_encoder.zig");

const log = std.log.scoped(.@"ltx/gemma3_validate");

pub const std_options = std.Options{
    .log_level = .info,
};

const CliArgs = struct {
    @"gemma-ckpt": []const u8 = "/home/ubuntu/models/gemma-3-12b-it",
    prompt: []const u8 = "A beautiful sunset over the ocean",
    output: []const u8 = "/tmp/zig_pos_hidden_states.safetensors",
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const args = zml.stdx.flags.parse(init.minimal.args, CliArgs);

    const ckpt_path = args.@"gemma-ckpt";
    const prompt = args.prompt;
    const output_path = args.output;

    log.info("Gemma checkpoint: {s}", .{ckpt_path});
    log.info("Prompt: {s}", .{prompt});
    log.info("Output: {s}", .{output_path});

    // ---- 1. Load tokenizer ----
    log.info("Loading tokenizer...", .{});
    const tokenizer_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{ckpt_path});
    defer allocator.free(tokenizer_path);

    var tokenizer = try zml.tokenizer.Tokenizer.fromFile(allocator, io, tokenizer_path);
    defer tokenizer.deinit();

    // ---- 2. Tokenize and pad ----
    log.info("Tokenizing...", .{});
    const tok_result = try gemma3.tokenizeAndPad(allocator, &tokenizer, prompt);
    log.info("Real token count: {d}", .{tok_result.real_token_count});

    // Log first few real tokens for verification.
    const pad_len = gemma3.MAX_SEQ_LEN - tok_result.real_token_count;
    log.info("First real tokens: {any}", .{tok_result.input_ids[pad_len..@min(pad_len + 10, gemma3.MAX_SEQ_LEN)]});

    // ---- 3. Open weight store ----
    log.info("Opening weight store...", .{});
    const index_path = try std.fmt.allocPrint(allocator, "{s}/model.safetensors.index.json", .{ckpt_path});
    defer allocator.free(index_path);

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, index_path);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    // ---- 4. Init model (bind weight shapes from store) ----
    log.info("Initializing encoder...", .{});
    const config: gemma3.Config = .{};
    var encoder = try gemma3.Encoder.init(allocator, store.view(), config);
    defer encoder.deinit(allocator);

    // ---- 5. Platform + sharding ----
    log.info("Initializing platform...", .{});
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    const sharding = try zml.sharding.replicatedSharding(platform);

    // ---- 6. Compile ----
    log.info("Compiling encoder forward pass...", .{});
    const input_ids: zml.Tensor = .init(.{ .s = gemma3.MAX_SEQ_LEN }, .u32);
    const attn_mask: zml.Tensor = .init(.{ .s = gemma3.MAX_SEQ_LEN }, .u32);

    var exe = try platform.compile(
        allocator,
        io,
        encoder,
        .forward,
        .{ input_ids, attn_mask },
        .{ .shardings = &.{sharding} },
    );
    defer exe.deinit();

    // ---- 7. Load weights ----
    log.info("Loading weights...", .{});
    const weight_bufs = try zml.io.load(
        gemma3.Encoder,
        &encoder,
        allocator,
        io,
        platform,
        &store,
        .{
            .shardings = &.{sharding},
            .parallelism = 16,
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
        },
    );
    defer allocator.free(weight_bufs.layers);

    // ---- 8. Prepare input buffers ----
    log.info("Preparing input buffers...", .{});
    const ids_shape: zml.Shape = .init(.{ .s = gemma3.MAX_SEQ_LEN }, .u32);
    var ids_buf: zml.Buffer = try .fromSlice(
        io,
        platform,
        zml.Slice.init(ids_shape, std.mem.sliceAsBytes(&tok_result.input_ids)),
        sharding,
    );
    defer ids_buf.deinit();

    const mask_shape: zml.Shape = .init(.{ .s = gemma3.MAX_SEQ_LEN }, .u32);
    var mask_buf: zml.Buffer = try .fromSlice(
        io,
        platform,
        zml.Slice.init(mask_shape, std.mem.sliceAsBytes(&tok_result.attention_mask)),
        sharding,
    );
    defer mask_buf.deinit();

    // ---- 9. Run ----
    log.info("Running encoder forward pass...", .{});
    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    exe_args.set(.{ weight_bufs, ids_buf, mask_buf });
    exe.callOpts(io, exe_args, &results, .{ .wait = true });

    var output_buf: zml.Buffer = results.get(zml.Buffer);
    defer output_buf.deinit();

    log.info("Output shape: {any}, dtype: {s}", .{
        output_buf.shape().dims(),
        @tagName(output_buf.shape().dtype()),
    });

    // ---- 10. Copy to host and save ----
    log.info("Saving output to {s}...", .{output_path});
    const output_slice = try output_buf.toSliceAlloc(allocator, io);
    defer output_slice.free(allocator);

    // Write as raw binary for now — can be compared with Python.
    const out_file = try std.Io.Dir.createFile(.cwd(), io, output_path, .{});
    defer out_file.close(io);
    var write_buf: [64 * 1024]u8 = undefined;
    var writer = out_file.writer(io, &write_buf);
    try writer.interface.writeAll(output_slice.constData());
    try writer.interface.flush();

    log.info("Done. Output written ({d} bytes).", .{output_slice.constData().len});
}
