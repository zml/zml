const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const acellm_ = @import("acellm.zig");
const aceemb_ = @import("aceemb.zig");
//const acedit_ = @import("acedit.zig");
//const acevae_ = @import("acevae.zig");
const inference = @import("inference.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};


pub const Shardings = struct {
    replicated: zml.sharding.Sharding,
    model: zml.sharding.Sharding,

    pub fn init(platform: *zml.Platform) !Shardings {
        const model_mesh: zml.sharding.LogicalMesh = try .init("model", .{ .model = .high_bandwidth });
        const model_sharding_strategy: zml.sharding.Strategy = try .suggest(model_mesh, platform.physical_mesh);
        return .{
            .replicated = try zml.sharding.replicatedSharding(platform),
            .model = try .initFromStrategy(platform, model_mesh, model_sharding_strategy),
        };
    }

    pub fn all(self: *const Shardings) [2]zml.sharding.Sharding {
        return .{ self.replicated, self.model };
    }
};

pub const Zml_handler = struct {
    allocator: std.mem.Allocator,
    arena: *std.heap.ArenaAllocator,
    io: std.Io,
    platform: *zml.Platform
};


pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const zml_handler: Zml_handler = .{
        .allocator = allocator,
        .arena = arena,
        .io = io,
        .platform = platform,
    };
    var acellm = try acellm_.AceLlm_handler.initFromFile(zml_handler);
    defer acellm.deinit(allocator);

    // Test model activations
    //try acellm.testModel(zml_handler);

    //const raw_prompt = "a short electric guitar solo\n\ninstrumental: true";
    //const audio_metadata = try inference.runPhase1(raw_prompt, zml_handler, &acellm);
    const audio_metadata: inference.AudioMetadata = .initExample();
    //defer audio_metadata.deinit(allocator);
    //const audio_codes = try inference.runPhase2(audio_metadata, zml_handler, &acellm);
    const audio_codes: inference.AudioCodes = try .initExample(allocator);
    defer audio_codes.deinit(allocator);

    acellm.unloadBuffers();

    const max_emb_len = try aceemb_.maxEmbeddingLength(zml_handler, audio_metadata);
    var aceemb = try aceemb_.AceEmb_handler.initFromFile(zml_handler, max_emb_len);
    defer aceemb.deinit(allocator);

    // Test model activations
    //try acedit.testModel(zml_handler, aceemb);

    const text_emb: inference.TextEmbedding = try inference.embedTextInputs(zml_handler, audio_metadata, &aceemb);
    defer text_emb.deinit(allocator);
    
    var stdout = std.Io.File.stdout().writer(io, &.{});
    const writer: *std.Io.Writer = &stdout.interface;
    const options: std.fmt.Number = .{};
    
    std.log.info("Embedding shape", .{});
    try text_emb.caption_embedding.shape.format(writer);
    try text_emb.caption_embedding.prettyPrint(writer, options);

    aceemb.unloadBuffers();
}


pub fn printSafetensors(allocator: std.mem.Allocator, io: std.Io, fpath: []const u8) !void {
    // Read model shapes.
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fpath);
    defer registry.deinit();
    std.log.info("Found {d} activations in {s}", .{ registry.tensors.count(), fpath });

    // Print model shapes
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
