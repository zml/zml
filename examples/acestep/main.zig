const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const acellm_ = @import("acellm.zig");
const aceemb_ = @import("aceemb.zig");
const acedit_ = @import("acedit.zig");
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
    platform: *zml.Platform,
    
    pub fn fromInit(init: std.process.Init) !Zml_handler {
        return .{
            .arena = init.arena,
            .allocator = init.gpa,
            .io = init.io,
            .platform = try .auto(init.gpa, init.io, .{}),
        };
    }
    
    pub fn deinit(self: *Zml_handler) void {
        self.platform.deinit(self.allocator);
    }
};


pub fn main(init: std.process.Init) !void {
    var zml_handler: Zml_handler = try .fromInit(init);
    defer zml_handler.deinit();
    
    //const raw_prompt = "a short electric guitar solo\n\ninstrumental: true";
    
    // ------------------------------------------------
    // Thinking/Inspiration phase : 5Hz LLM model
    // ------------------------------------------------
    
    //var acellm = try acellm_.AceLlm_handler.initFromFile(zml_handler);
    //defer acellm.deinit(zml_handler.allocator);

    // Test model activations
    //try acellm.testModel(zml_handler);

    //const audio_metadata = try inference.runPhase1(raw_prompt, zml_handler, &acellm);
    const audio_metadata: inference.AudioMetadata = .initExample();
    //defer audio_metadata.deinit(zml_handler.allocator);
    //const audio_codes = try inference.runPhase2(audio_metadata, zml_handler, &acellm);
    const audio_codes: inference.AudioCodes = try .initExample(zml_handler.allocator);
    defer audio_codes.deinit(zml_handler.allocator);

    //acellm.unloadBuffers();
    
    // ------------------------------------------------
    // The text inputs of the DiT need to be embedded
    // using the AceEmb model embedding, not 5Hz
    // ------------------------------------------------

    const full_emb, const partial_emb = try aceemb_.embeddingLengths(zml_handler, audio_metadata);
    var aceemb = try aceemb_.AceEmb_handler.initFromFile(zml_handler, full_emb, partial_emb);
    defer aceemb.deinit(zml_handler.allocator);

    // Test model activations
    //try acedit.testModel(zml_handler, aceemb);

    const text_emb: inference.TextEmbedding = try inference.embedTextInputs(zml_handler, audio_metadata, &aceemb);
    defer text_emb.deinit(zml_handler.allocator);
    
    //try text_emb.print(zml_handler.io);
    aceemb.unloadBuffers();
    
    // ------------------------------------------------
    // Generation phase : diffusion with DiT model
    // ------------------------------------------------

    var acedit = try acedit_.AceDit_handler.initFromFile(zml_handler, full_emb, partial_emb, audio_codes.len());
    defer acedit.deinit(zml_handler.allocator);

    // Test model activations
    //try acedit.testModel(zml_handler);

    const diffused_latents = try inference.runDiffusion(zml_handler, &acedit, text_emb, audio_codes);
    defer diffused_latents.deinit(zml_handler.allocator);

    try diffused_latents.print(zml_handler.io);
    acedit.unloadBuffers();

    // ------------------------------------------------
    // Output latents of the DiT model are decoded
    // with the VAE model
    // ------------------------------------------------

    // TODO

}


pub fn printSafetensors(allocator: std.mem.Allocator, io: std.Io, fpath: []const u8) !void {
    // Read model shapes.
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fpath);
    defer registry.deinit();
    std.log.debug("Found {d} activations in {s}", .{ registry.tensors.count(), fpath });

    // Print model shapes
    const tensors: zml.safetensors.Tensors = registry.tensors;
    const data = tensors.entries;
    for (0..data.len) |i| {
        const entry = data.get(i);
        const tensor: zml.safetensors.Tensor = tensors.get(entry.key).?;
        std.log.debug("Tensor(name={s} shape={f} size={d})", .{
            tensor.name,
            tensor.shape,
            tensor.byteSize(),
        });
    }
}
