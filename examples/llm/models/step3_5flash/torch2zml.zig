const std = @import("std");
const zml = @import("zml");

// --- THE BLUEPRINT ---
pub const Step3p5Model = struct {
    model: Step3p5Flash, 
};

pub const Step3p5Layer = struct {
    input_layernorm: zml.nn.LayerNorm,
    self_attn: struct {
        q_proj: zml.nn.Linear,
        k_proj: zml.nn.Linear,
        v_proj: zml.nn.Linear,
        o_proj: zml.nn.Linear,
        q_norm: zml.nn.LayerNorm,
        k_norm: zml.nn.LayerNorm,
    },
    post_attention_layernorm: zml.nn.LayerNorm,
    moe: struct {
        gate: zml.nn.Linear,
        experts: []zml.nn.Linear, 
    },
};

pub const Step3p5Flash = struct {
    embed_tokens: zml.nn.TokenEmbedding, 
    layers: []Step3p5Layer,
    norm: zml.nn.LayerNorm,
    lm_head: zml.nn.Linear,

    pub fn forward(self: *Step3p5Flash, input: zml.Tensor) !zml.Tensor {
        _ = self; return input; 
    }
};

pub fn main(init: std.process.Init) !void {
    // 1. ZIG 0.16.0 "Juicy Main" Initialization
    const allocator = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;
    
    // Initialize ZML VFS
    var vfs = try zml.io.VFS.init(allocator, io);
    defer vfs.deinit();
    const vfs_io = vfs.io();

    // Use the new arena allocator for args so we don't need to manually free
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) {
        std.debug.print("Usage: torch2zml <path_to_index_json>\n", .{});
        return;
    }

    // 2. FIX: The official Zig 0.16.0 std.Io.Dir absolute path resolution
    const weights_path_raw = args[1];
    const cwd = std.Io.Dir.cwd();
    const absolute_path = try cwd.realPathFileAlloc(io, weights_path_raw, arena);    
    // 3. Platform init (Blackwell cluster)
    const platform = try zml.Platform.auto(allocator, vfs_io, .{});
    defer platform.deinit(allocator, vfs_io); 

    // 4. Load Registry using the absolute path
    var registry = try zml.safetensors.TensorRegistry.fromPath(allocator, vfs_io, absolute_path);
    defer registry.deinit();

    // 5. Dynamic Counting Logic
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

    std.log.info("ZML/v2: Detected {d} layers and {d} experts per layer.", .{ max_layer, max_expert });

    // 6. Allocate Slices
    const layers = try allocator.alloc(Step3p5Layer, max_layer);
    defer allocator.free(layers);
    for (layers) |*layer| {
        layer.moe.experts = try allocator.alloc(zml.nn.Linear, max_expert);
    }
    defer for (layers) |layer| allocator.free(layer.moe.experts);

    const model_blueprint = Step3p5Flash{
        .embed_tokens = undefined,
        .layers = layers,
        .norm = undefined,
        .lm_head = undefined,
    };
    
    // Wrap in Step3p5Model to match the 'model.' prefix
    const root_blueprint = Step3p5Model{ .model = model_blueprint };

    // 7. Load Weights to Metal
    var store = zml.io.TensorStore.fromRegistry(allocator, &registry);
    const model = try zml.io.load(Step3p5Model, &root_blueprint, allocator, vfs_io, platform, &store, .{
        .parallelism = 1,
        .dma_chunks = 2,
        .dma_chunk_size = 1024 * 1024 * 128, 
    });
    _ = model; 

    std.log.info("🚀 SUCCESS: All {d} tensors mapped to Blackwells!", .{registry.tensors.count()});
}
