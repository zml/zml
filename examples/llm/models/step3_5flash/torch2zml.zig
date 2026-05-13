const std = @import("std");
const zml = @import("zml");

// --- 1. ACTIVATION NODE ---
// This replaces zml.nn.Linear/LayerNorm. 
// Using `?zml.Tensor` (optional) prevents crashes if an MoE expert wasn't used!
pub const NodeActs = struct {
    input_0: ?zml.Tensor = null,
    output: ?zml.Tensor = null,
};

// --- 2. ACTIVATION BLUEPRINT ---
pub const Step3p5LayerActs = struct {
    input_layernorm: NodeActs,
    self_attn: struct {
        q_proj: NodeActs,
        k_proj: NodeActs,
        v_proj: NodeActs,
        o_proj: NodeActs,
        q_norm: NodeActs,
        k_norm: NodeActs,
    },
    post_attention_layernorm: NodeActs,
    moe: struct {
        gate: NodeActs,
        experts: []NodeActs, 
    },
};

pub const Step3p5FlashActs = struct {
    embed_tokens: NodeActs, 
    layers: []Step3p5LayerActs,
    norm: NodeActs,
    lm_head: NodeActs,
};

pub const Step3p5ModelActs = struct {
    model: Step3p5FlashActs, 
};

pub fn main(init: std.process.Init) !void {
    // Zig 0.16.0 Juicy Main Init
    const allocator = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;
    
    var vfs = try zml.io.VFS.init(allocator, io);
    defer vfs.deinit();
    const vfs_io = vfs.io();

    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) return;

    // Resolve Absolute Path
    const weights_path_raw = args[1];
    const cwd = std.Io.Dir.cwd();
    const absolute_path = try cwd.realPathFileAlloc(io, weights_path_raw, arena);

    const platform = try zml.Platform.auto(allocator, vfs_io, .{});
    defer platform.deinit(allocator, vfs_io); 

    var registry = try zml.safetensors.TensorRegistry.fromPath(allocator, vfs_io, absolute_path);
    defer registry.deinit();

    // Dynamic Layer/Expert Counting
    var max_layer: usize = 0;
    var max_expert: usize = 0;
    var it = registry.tensors.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (std.mem.indexOf(u8, name, "layers.")) |idx| {
            const rest = name[idx + 7 ..];
            var parts = std.mem.splitScalar(u8, rest, '.');
            if (parts.next()) |num_str| {
                const val = std.fmt.parseInt(usize, num_str, 10) catch continue;
                if (val >= max_layer) max_layer = val + 1;
            }
        }
        if (std.mem.indexOf(u8, name, "experts.")) |idx| {
            const rest = name[idx + 8 ..];
            var parts = std.mem.splitScalar(u8, rest, '.');
            if (parts.next()) |num_str| {
                const val = std.fmt.parseInt(usize, num_str, 10) catch continue;
                if (val >= max_expert) max_expert = val + 1;
            }
        }
    }

    std.log.info("ZML/v2: Verification Mode. Detected {d} layers and {d} experts max.", .{ max_layer, max_expert });

    // Safely Allocate and Initialize the Slices
    const layers = try allocator.alloc(Step3p5LayerActs, max_layer);
    defer allocator.free(layers);
    
    for (layers) |*layer| {
        layer.* = .{
            .input_layernorm = .{},
            .self_attn = .{
                .q_proj = .{}, .k_proj = .{}, .v_proj = .{}, .o_proj = .{}, .q_norm = .{}, .k_norm = .{}
            },
            .post_attention_layernorm = .{},
            .moe = .{
                .gate = .{},
                .experts = try allocator.alloc(NodeActs, max_expert),
            },
        };
        // Initialize experts to nulls so the loader doesn't crash on empty memory
        @memset(layer.moe.experts, .{});
    }
    defer for (layers) |layer| allocator.free(layer.moe.experts);

    const model_blueprint = Step3p5FlashActs{
        .embed_tokens = .{},
        .layers = layers,
        .norm = .{},
        .lm_head = .{},
    };
    
    const root_blueprint = Step3p5ModelActs{ .model = model_blueprint };

    // Load Activations to Metal
    var store = zml.io.TensorStore.fromRegistry(allocator, &registry);    
    const model = try zml.io.load(Step3p5ModelActs, &root_blueprint, arena, vfs_io, platform, &store, .{
        .parallelism = 1,
        .dma_chunks = 2,
        .dma_chunk_size = 1024 * 1024 * 128, 
    });
    _ = model; 

    std.log.info("all captured activations loaded", .{});
}
