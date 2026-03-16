const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;

/// Which parameter set to use at runtime.
pub const Stage = enum {
    stage1,
    stage2,
};

pub const Config = struct {
    num_transformer_blocks: usize,
};

/// FeedForward module shared by stage 1 and stage 2.
///
/// Forward contract for LTX 2.3 video FF:
/// [B, T, 4096] -> Linear(4096->16384) -> GELU(tanh approx) -> Linear(16384->4096)
pub const FeedForward = struct {
    pub const Params = struct {
        proj: zml.nn.Linear,
        out: zml.nn.Linear,
    };

    pub fn initParams(store: zml.io.TensorStore.View) Params {
        const proj_store = store.withPrefix("net").withLayer(0).withPrefix("proj");
        const out_store = store.withPrefix("net").withLayer(2);

        return .{
            .proj = .init(
                proj_store.createTensor("weight", .{ .d_ff, .d }, null),
                proj_store.createTensor("bias", .{.d_ff}, null),
                .d,
            ),
            .out = .init(
                out_store.createTensor("weight", .{ .d, .d_ff }, null),
                out_store.createTensor("bias", .{.d}, null),
                .d_ff,
            ),
        };
    }

    pub fn forward(self: FeedForward, x: Tensor, params: Params) Tensor {
        _ = self;
        const x_ = x.withPartialTags(.{ .b, .t, .d });
        const h1 = params.proj.forward(x_);
        const h2 = h1.gelu();
        return params.out.forward(h2);
    }

    pub fn unloadBuffers(params: *zml.Bufferized(Params)) void {
        params.proj.weight.deinit();
        if (params.proj.bias) |*b| b.deinit();
        params.out.weight.deinit();
        if (params.out.bias) |*b| b.deinit();
    }
};

/// Placeholder attention module.
/// This uses the same code for stage1/stage2 and only params differ.
pub const Attention = struct {
    pub const Params = struct {};

    pub fn initParams(_: zml.io.TensorStore.View) Params {
        return .{};
    }

    pub fn forward(self: Attention, x: Tensor, params: Params) Tensor {
        _ = self;
        _ = params;
        // TODO: implement LTX Attention forward.
        return x;
    }

    pub fn unloadBuffers(_: *zml.Bufferized(Params)) void {}
};

/// One AV transformer block. Shared implementation across stages.
pub const BasicAVTransformerBlock = struct {
    ff: FeedForward,
    attn: Attention,

    pub const Params = struct {
        ff: FeedForward.Params,
        attn: Attention.Params,
    };

    pub fn init() BasicAVTransformerBlock {
        return .{
            .ff = .{},
            .attn = .{},
        };
    }

    pub fn initParams(store: zml.io.TensorStore.View) Params {
        return .{
            .ff = FeedForward.initParams(store.withPrefix("ff")),
            .attn = Attention.initParams(store),
        };
    }

    pub fn forward(self: BasicAVTransformerBlock, x: Tensor, params: Params) Tensor {
        _ = self.attn.forward(x, params.attn);
        // NOTE: residual/gating/adaln wiring belongs at block level and will be added here.
        return self.ff.forward(x, params.ff);
    }

    pub fn unloadBuffers(params: *zml.Bufferized(Params)) void {
        FeedForward.unloadBuffers(&params.ff);
    }
};

/// Velocity model (transformer stack) shared across stages.
pub const LTXModel = struct {
    blocks: []BasicAVTransformerBlock,

    pub const Params = struct {
        blocks: []BasicAVTransformerBlock.Params,

        pub fn deinit(self: *Params, allocator: std.mem.Allocator) void {
            allocator.free(self.blocks);
        }

        pub fn unloadBuffers(self: *zml.Bufferized(Params), allocator: std.mem.Allocator) void {
            for (self.blocks) |*bp| {
                BasicAVTransformerBlock.unloadBuffers(bp);
            }
            allocator.free(self.blocks);
        }
    };

    pub fn init(allocator: std.mem.Allocator, config: Config) !LTXModel {
        const blocks = try allocator.alloc(BasicAVTransformerBlock, config.num_transformer_blocks);
        for (blocks) |*b| b.* = BasicAVTransformerBlock.init();

        return .{ .blocks = blocks };
    }

    pub fn deinit(self: *LTXModel, allocator: std.mem.Allocator) void {
        allocator.free(self.blocks);
    }

    pub fn initParams(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Params {
        const blocks = try allocator.alloc(BasicAVTransformerBlock.Params, config.num_transformer_blocks);

        const blocks_store = store.withPrefix("transformer_blocks");
        for (blocks, 0..) |*bp, i| {
            bp.* = BasicAVTransformerBlock.initParams(blocks_store.withLayer(i));
        }

        return .{ .blocks = blocks };
    }

    pub fn forward(self: LTXModel, x: Tensor, params: Params) Tensor {
        std.debug.assert(self.blocks.len == params.blocks.len);

        var h = x;
        for (self.blocks, params.blocks) |block, block_params| {
            h = block.forward(h, block_params);
        }
        return h;
    }
};

/// Top-level wrapper used by the LTX pipeline.
pub const X0Model = struct {
    velocity_model: LTXModel,

    pub const Params = struct {
        velocity_model: LTXModel.Params,

        pub fn deinit(self: *Params, allocator: std.mem.Allocator) void {
            self.velocity_model.deinit(allocator);
        }
    };

    pub fn init(allocator: std.mem.Allocator, config: Config) !X0Model {
        return .{
            .velocity_model = try LTXModel.init(allocator, config),
        };
    }

    pub fn deinit(self: *X0Model, allocator: std.mem.Allocator) void {
        self.velocity_model.deinit(allocator);
    }

    pub fn initParams(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Params {
        const root = selectTransformerRoot(store);
        return .{
            .velocity_model = try LTXModel.initParams(allocator, root, config),
        };
    }

    pub fn forward(self: X0Model, x: Tensor, params: Params) Tensor {
        return self.velocity_model.forward(x, params.velocity_model);
    }

    /// Focused entrypoint used for FF parity bring-up.
    pub fn forwardBlock0FF(_: X0Model, x: Tensor, params: Params) Tensor {
        const ff = FeedForward{};
        return ff.forward(x, params.velocity_model.blocks[0].ff);
    }

    pub fn unloadBuffers(params: *zml.Bufferized(Params), allocator: std.mem.Allocator) void {
        LTXModel.Params.unloadBuffers(&params.velocity_model, allocator);
    }
};

/// Active runtime parameter set (single stage at a time).
/// This keeps code shared while allowing atomic stage-by-stage validation.
pub const ActiveParams = struct {
    stage: Stage,
    params: X0Model.Params,

    pub fn deinit(self: *ActiveParams, allocator: std.mem.Allocator) void {
        self.params.deinit(allocator);
    }
};

/// Build one active stage-specific parameter set from one checkpoint/store.
pub fn initActiveParams(
    allocator: std.mem.Allocator,
    stage: Stage,
    store: zml.io.TensorStore.View,
    config: Config,
) !ActiveParams {
    return .{
        .stage = stage,
        .params = try X0Model.initParams(allocator, store, config),
    };
}

fn selectTransformerRoot(store: zml.io.TensorStore.View) zml.io.TensorStore.View {
    if (store.hasKey("model.velocity_model.transformer_blocks.0.ff.net.0.proj.weight")) {
        return store.withPrefix("model").withPrefix("velocity_model");
    }
    if (store.hasKey("model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight")) {
        return store.withPrefix("model").withPrefix("diffusion_model");
    }
    if (store.hasKey("velocity_model.transformer_blocks.0.ff.net.0.proj.weight")) {
        return store.withPrefix("velocity_model");
    }
    if (store.hasKey("diffusion_model.transformer_blocks.0.ff.net.0.proj.weight")) {
        return store.withPrefix("diffusion_model");
    }
    // Keep previous behavior as a fallback.
    return store.withPrefix("velocity_model");
}

/// Free-function entrypoint for parity tooling: block0 FF only.
pub fn forwardBlock0FF(x: Tensor, params: X0Model.Params) Tensor {
    const ff = FeedForward{};
    return ff.forward(x, params.velocity_model.blocks[0].ff);
}

/// Focused FF-only entrypoint for parity bring-up.
pub fn forwardFF(x: Tensor, params: FeedForward.Params) Tensor {
    const ff = FeedForward{};
    return ff.forward(x, params);
}

/// Build only block0.ff params from the selected transformer root.
pub fn initBlock0FFParams(store: zml.io.TensorStore.View) FeedForward.Params {
    const root = selectTransformerRoot(store);
    return FeedForward.initParams(root.withPrefix("transformer_blocks").withLayer(0).withPrefix("ff"));
}

pub fn unloadBlock0FFBuffers(params: *zml.Bufferized(FeedForward.Params)) void {
    FeedForward.unloadBuffers(params);
}
