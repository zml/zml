const std = @import("std");
const zml = @import("zml");
const tools = @import("tools.zig");
const stdx = zml.stdx;
const Tensor = zml.Tensor;

const log = std.log.scoped(.autoencoder_kl_flux2);

pub const Config = struct {
    in_channels: i64 = 3,
    out_channels: i64 = 3,
    block_out_channels: []const i64 = &.{ 128, 256, 512, 512 },
    layers_per_block: i64 = 2,
    latent_channels: i64 = 32,
    norm_num_groups: i64 = 32,
    patch_size: []const i64 = &.{ 2, 2 },
    sample_size: i64 = 1024,
    batch_norm_eps: f32 = 1e-4,
    use_quant_conv: bool = true,
    use_post_quant_conv: bool = true,
};

// --- Helpers ---

pub fn unloadWeights(allocator: std.mem.Allocator, weights: anytype) void {
    const T = @TypeOf(weights.*);
    const type_info = @typeInfo(T);
    switch (type_info) {
        .@"struct" => |info| {
            if (T == zml.Buffer) {
                weights.deinit();
                return;
            }
            inline for (info.fields) |field| {
                unloadWeights(allocator, &@field(weights, field.name));
            }
        },
        .optional => {
            if (weights.*) |*w| {
                unloadWeights(allocator, w);
            }
        },
        .pointer => |info| {
            if (info.size == .slice) {
                for (weights.*) |*item| {
                    unloadWeights(allocator, item);
                }
                allocator.free(weights.*);
            }
        },
        else => {},
    }
}

const Conv2d = struct {
    weight: Tensor,
    bias: ?Tensor,
    stride: i64 = 1,
    padding: i64 = 0,
    dilation: i64 = 1,
    groups: i64 = 1,

    pub fn init(store: zml.io.TensorStore.View, stride: i64, padding: i64) Conv2d {
        // weights: [Out, In, H, W] -> zml [out, in, k_h, k_w]
        const weight = store.createTensor("weight");
        const bias = store.maybeCreateTensor("bias");
        return .{
            .weight = weight,
            .bias = bias,
            .stride = stride,
            .padding = padding,
        };
    }

    pub fn deinit(self: @This()) void {
        _ = self; // autofix
    }

    pub fn forward(self: Conv2d, x: Tensor) Tensor {
        // NCHW layout
        // Weight: OIHW (Output, Input, Height, Width)
        const input_spatial = [_]i64{ 2, 3 };
        const kernel_spatial = [_]i64{ 2, 3 };
        const output_spatial = [_]i64{ 2, 3 };

        const pad = self.padding;
        const padding_flat = [_]i64{ pad, pad, pad, pad };
        const stride = [_]i64{ self.stride, self.stride };
        const dilation = [_]i64{ self.dilation, self.dilation };

        var w = self.weight;
        if (w.rank() == 2) {
            w = w.reshape(.{ w.dim(0), w.dim(1), 1, 1 });
        }

        var y = x.convolution(w, .{
            .input_batch_dimension = 0,
            .input_feature_dimension = 1,
            .input_spatial_dimensions = &input_spatial,
            .kernel_input_feature_dimension = 1,
            .kernel_output_feature_dimension = 0,
            .kernel_spatial_dimensions = &kernel_spatial,
            .output_batch_dimension = 0,
            .output_feature_dimension = 1,
            .output_spatial_dimensions = &output_spatial,
            .window_strides = &stride,
            .pad_value = &padding_flat,
            .lhs_dilation = &.{ 1, 1 },
            .rhs_dilation = &dilation,
            .window_reversal = &.{ false, false },
            .feature_group_count = self.groups,
            .batch_group_count = 1,
            .precision_config = &.{ .DEFAULT, .DEFAULT }, // Check precision enum
        });

        if (self.bias) |b| {
            // b: [C] -> [1, C, 1, 1]
            y = y.add(b.reshape(.{ 1, b.dim(0), 1, 1 }).broad(y.shape()));
        }
        return y;
    }
};

const GroupNorm = struct {
    weight: ?Tensor, // gamma
    bias: ?Tensor, // beta
    num_groups: i64,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, num_groups: i64, eps: f32, affine: bool) GroupNorm {
        return .{
            .weight = if (affine) store.createTensor("weight") else null,
            .bias = if (affine) store.createTensor("bias") else null,
            .num_groups = num_groups,
            .eps = eps,
        };
    }

    pub fn forward(self: GroupNorm, x: Tensor) Tensor {
        // specific group norm implementation or fallback
        // zml.nn.groupNorm might be missing.
        // x: [B, C, H, W]
        // Reshape to [B, G, C/G, H*W]
        // Normalize over last 2 dims?
        // Let's rely on manual implementation for now to be safe.
        const B = x.shape().dim(0);
        const C = x.shape().dim(1);
        const H = x.shape().dim(2);
        const W = x.shape().dim(3);
        const G = self.num_groups;
        const C_per_G = @divExact(C, G);

        // [B, G, C/G, H, W]
        const x_g = x.reshape(.{ B, G, C_per_G, H, W });

        // Mean/Var over C/G, H, W
        // Axes: 2, 3, 4
        // zml.nn.normalize (x, axes, eps) ??
        // zml.nn.mean(x, axes)

        // Flatten G, C/G, H, W into one dim for stats
        // x_g: [B, G, C/G, H, W]
        const flat = x_g.reshape(.{ B, G, -1 }); // [B, G, N]
        const mean = flat.mean(2); // [B, G]
        const mean_broad = mean.reshape(.{ B, G, 1, 1, 1 }).broad(x_g.shape());

        const diff = x_g.sub(mean_broad);
        const sq_diff = diff.mul(diff);

        // flatten sq_diff to mean
        const sq_diff_flat = sq_diff.reshape(.{ B, G, -1 });
        const var_val = sq_diff_flat.mean(2); // [B, G]
        const std_val = var_val.add(Tensor.scalar(self.eps, var_val.dtype())).sqrt();
        const std_broad = std_val.reshape(.{ B, G, 1, 1, 1 }).broad(x_g.shape());

        var out = diff.div(std_broad);
        out = out.reshape(x.shape()); // [B, C, H, W]

        if (self.weight) |gamma| {
            out = out.mul(gamma.reshape(.{ 1, C, 1, 1 }).broad(x.shape()));
        }
        if (self.bias) |beta| {
            out = out.add(beta.reshape(.{ 1, C, 1, 1 }).broad(x.shape()));
        }
        return out;
    }
};

const BN = struct {
    running_mean: Tensor,
    running_var: Tensor,
    config: Config,

    pub fn init(store: zml.io.TensorStore.View, config: Config) BN {
        return .{
            .running_mean = store.createTensor("running_mean"),
            .running_var = store.createTensor("running_var"),
            .config = config,
        };
    }

    pub fn deinit(self: @This()) void {
        _ = self; // autofix
    }
};

// --- ResNet Block ---

const ResnetBlock2D = struct {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    conv_shortcut: ?Conv2d,

    pub fn init(store: zml.io.TensorStore.View, in_channels: i64, out_channels: i64, temb_channels: ?i64, eps: f32, groups: i64) ResnetBlock2D {
        _ = temb_channels; // Unused in VAE ResNet generally
        return .{
            .norm1 = GroupNorm.init(store.withPrefix("norm1"), groups, eps, true),
            .conv1 = Conv2d.init(store.withPrefix("conv1"), 1, 1), // stride 1, padding 1 (3x3 kernel assumed typically, check weights)
            .norm2 = GroupNorm.init(store.withPrefix("norm2"), groups, eps, true),
            .conv2 = Conv2d.init(store.withPrefix("conv2"), 1, 1),
            .conv_shortcut = if (in_channels != out_channels)
                Conv2d.init(store.withPrefix("conv_shortcut"), 1, 0) // 1x1 conv usually
            else
                null,
        };
    }

    pub fn forward(self: ResnetBlock2D, input: Tensor, temb: ?Tensor) Tensor {
        _ = temb;
        var h = self.norm1.forward(input);
        h = h.silu();
        h = self.conv1.forward(h);

        h = self.norm2.forward(h);
        h = h.silu();
        h = self.conv2.forward(h);

        var shortcut = input;
        if (self.conv_shortcut) |conv| {
            shortcut = conv.forward(input);
        }

        return h.add(shortcut);
    }
};

// --- Attention Block (VAE) ---

const Attention = struct {
    group_norm: GroupNorm,
    to_q: Conv2d,
    to_k: Conv2d,
    to_v: Conv2d,
    to_out: Conv2d,
    num_heads: i64,

    pub fn init(store: zml.io.TensorStore.View, in_channels: i64, num_groups: i64, eps: f32) Attention {
        _ = in_channels;
        const num_heads = 1;
        return .{
            .group_norm = GroupNorm.init(store.withPrefix("group_norm"), num_groups, eps, true),
            .to_q = Conv2d.init(store.withPrefix("to_q"), 1, 0), // 1x1 conv
            .to_k = Conv2d.init(store.withPrefix("to_k"), 1, 0),
            .to_v = Conv2d.init(store.withPrefix("to_v"), 1, 0),
            .to_out = Conv2d.init(store.withPrefix("to_out.0"), 1, 0),
            .num_heads = num_heads,
        };
    }

    pub fn forward(self: Attention, x: Tensor) Tensor {
        const B = x.shape().dim(0);
        const C = x.shape().dim(1);
        const H = x.shape().dim(2);
        const W = x.shape().dim(3);

        const residual = x;
        const norm = self.group_norm.forward(x);

        const q = self.to_q.forward(norm).reshape(.{ B, -1, H * W }).transpose(.{ 0, 2, 1 }); // [B, HW, C]
        const k = self.to_k.forward(norm).reshape(.{ B, -1, H * W }).transpose(.{ 0, 2, 1 });
        const v = self.to_v.forward(norm).reshape(.{ B, -1, H * W }).transpose(.{ 0, 2, 1 });

        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(C)));

        // q: [B, HW, C]
        // k: [B, HW, C] -> k_t: [B, C, HW]
        // Use tags for dot product
        const q_tagged = q.withPartialTags(.{ .b, .y, .c });
        const k_tagged = k.withPartialTags(.{ .b, .x, .c });
        const k_t_tagged = k_tagged.transpose(.{ 0, 2, 1 }); // [b, c, x]

        var attn = q_tagged.dot(k_t_tagged, .c).mul(Tensor.scalar(scale, q_tagged.dtype())); // [b, y, x]
        attn = attn.softmax(-1);

        const v_tagged = v.withPartialTags(.{ .b, .x, .c });
        const out_attn = attn.dot(v_tagged, .x); // [b, y, c]

        const out_transposed = out_attn.transpose(.{ 0, 2, 1 }).reshape(.{ B, C, H, W });
        const result = self.to_out.forward(out_transposed);

        return residual.add(result);
    }
};

const UpDecoderBlock2D = struct {
    resnets: []ResnetBlock2D,
    upsamplers: []Conv2d, // Usually just one upsampler or none/empty
    allocator: std.mem.Allocator,

    pub fn init(store: zml.io.TensorStore.View, allocator: std.mem.Allocator, in_channels: i64, out_channels: i64, num_layers: i64, eps: f32, num_groups: i64, add_upsample: bool) !UpDecoderBlock2D {
        const ResnetList = std.ArrayList(ResnetBlock2D);
        var resnets = try ResnetList.initCapacity(allocator, @intCast(num_layers));

        // First resnet: in -> out
        try resnets.append(allocator, ResnetBlock2D.init(store.withPrefix("resnets.0"), in_channels, out_channels, null, eps, num_groups));

        // Subsequent resnets: out -> out
        for (1..@intCast(num_layers)) |i| {
            var prefix_buf: [32]u8 = undefined;
            const prefix = try std.fmt.bufPrint(&prefix_buf, "resnets.{d}", .{i});
            try resnets.append(allocator, ResnetBlock2D.init(store.withPrefix(prefix), out_channels, out_channels, null, eps, num_groups));
        }

        var upsamplers: std.ArrayList(Conv2d) = undefined;
        if (add_upsample) {
            const ConvList = std.ArrayList(Conv2d);
            upsamplers = try ConvList.initCapacity(allocator, 1);
            try upsamplers.append(allocator, Conv2d.init(store.withPrefix("upsamplers.0.conv"), 1, 1));
        } else {
            upsamplers = try std.ArrayList(Conv2d).initCapacity(allocator, 0);
        }

        return .{
            .resnets = try resnets.toOwnedSlice(allocator),
            .upsamplers = try upsamplers.toOwnedSlice(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: UpDecoderBlock2D) void {
        self.allocator.free(self.resnets);
        self.allocator.free(self.upsamplers);
    }

    pub fn forward(self: UpDecoderBlock2D, x: Tensor) Tensor {
        var h = x;
        for (self.resnets) |resnet| {
            h = resnet.forward(h, null);
        }

        for (self.upsamplers) |conv| {
            // Resize (Nearest Neighbor 2x)
            h = zml.nn.upsample(h, .{ .mode = .nearest, .scale_factor = &.{ 2.0, 2.0 } });
            h = conv.forward(h);
        }
        return h;
    }
};

const UNetMidBlock2D = struct {
    resnets: []ResnetBlock2D,
    attn: Attention,
    allocator: std.mem.Allocator,

    pub fn init(store: zml.io.TensorStore.View, allocator: std.mem.Allocator, in_channels: i64, eps: f32, num_groups: i64, attention_head_dim: i64) !UNetMidBlock2D {
        _ = attention_head_dim;
        const resnets = try allocator.alloc(ResnetBlock2D, 2);
        resnets[0] = ResnetBlock2D.init(store.withPrefix("resnets.0"), in_channels, in_channels, null, eps, num_groups);
        resnets[1] = ResnetBlock2D.init(store.withPrefix("resnets.1"), in_channels, in_channels, null, eps, num_groups);

        return .{
            .resnets = resnets,
            .attn = Attention.init(store.withPrefix("attentions.0"), in_channels, num_groups, eps),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: UNetMidBlock2D) void {
        self.allocator.free(self.resnets);
    }

    pub fn forward(self: UNetMidBlock2D, x: Tensor) Tensor {
        var h = x;
        h = self.resnets[0].forward(h, null);
        h = self.attn.forward(h);
        h = self.resnets[1].forward(h, null);
        return h;
    }
};

pub const Decoder = struct {
    conv_in: Conv2d,
    up_blocks: []UpDecoderBlock2D,
    mid_block: UNetMidBlock2D,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,

    allocator: std.mem.Allocator,

    pub fn init(store: zml.io.TensorStore.View, allocator: std.mem.Allocator, config: Config) !Decoder {
        const block_out_channels = config.block_out_channels;
        const layers_per_block = config.layers_per_block;
        const norm_num_groups = config.norm_num_groups;

        // Initial Conv
        const conv_in = Conv2d.init(store.withPrefix("conv_in"), 1, 1);

        // Mid Block
        const mid_channels = block_out_channels[block_out_channels.len - 1];
        const mid_block = try UNetMidBlock2D.init(store.withPrefix("mid_block"), allocator, mid_channels, 1e-6, norm_num_groups, 1);

        // Up Blocks
        const UpBlockList = std.ArrayList(UpDecoderBlock2D);
        // Capacity equal to number of blocks
        var up_blocks = try UpBlockList.initCapacity(allocator, block_out_channels.len);

        var idx: usize = block_out_channels.len;
        while (idx > 0) {
            idx -= 1;
            const out_ch = block_out_channels[idx];
            const in_ch = if (idx == block_out_channels.len - 1) mid_channels else block_out_channels[idx + 1];

            const add_upsample = idx > 0;

            var prefix_buf: [32]u8 = undefined;
            // Weight index: (N-1) - i
            const weight_idx = (block_out_channels.len - 1) - idx;
            const prefix_z = try std.fmt.bufPrint(&prefix_buf, "up_blocks.{d}", .{weight_idx});

            try up_blocks.append(allocator, try UpDecoderBlock2D.init(store.withPrefix(prefix_z), allocator, in_ch, out_ch, layers_per_block + 1, 1e-6, norm_num_groups, add_upsample));
        }

        const conv_norm_out = GroupNorm.init(store.withPrefix("conv_norm_out"), norm_num_groups, 1e-6, true);
        const conv_out = Conv2d.init(store.withPrefix("conv_out"), 1, 1);

        return .{
            .conv_in = conv_in,
            .up_blocks = try up_blocks.toOwnedSlice(allocator),
            .mid_block = mid_block,
            .conv_norm_out = conv_norm_out,
            .conv_out = conv_out,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: Decoder) void {
        self.mid_block.deinit();
        for (self.up_blocks) |b| b.deinit();
        self.allocator.free(self.up_blocks);
    }

    pub fn forward(self: Decoder, z: Tensor) Tensor {
        var h = self.conv_in.forward(z);
        h = self.mid_block.forward(h);

        for (self.up_blocks) |blk| {
            h = blk.forward(h);
        }

        h = self.conv_norm_out.forward(h);
        h = h.silu();
        h = self.conv_out.forward(h);
        return h;
    }
};

pub const AutoencoderKLFlux2Model = struct {
    decoder: Decoder,
    post_quant_conv: ?Conv2d,
    bn: BN, // BatchNorm stats container
    config: Config,

    pub fn init(store: zml.io.TensorStore.View, allocator: std.mem.Allocator, config: Config) !AutoencoderKLFlux2Model {
        return .{
            .decoder = try Decoder.init(store.withPrefix("decoder"), allocator, config),
            .post_quant_conv = if (config.use_post_quant_conv)
                Conv2d.init(store.withPrefix("post_quant_conv"), 1, 0)
            else
                null,
            .bn = BN.init(store.withPrefix("bn"), config),
            .config = config,
        };
    }

    pub fn deinit(
        self: *AutoencoderKLFlux2Model,
    ) void {
        self.decoder.deinit();
        if (self.post_quant_conv) |conv| {
            conv.deinit();
        }
        self.bn.deinit();
    }

    pub fn decode(self: AutoencoderKLFlux2Model, z: Tensor) Tensor {
        var h = z;
        if (self.post_quant_conv) |conv| {
            h = conv.forward(h.convert(conv.weight.dtype()));
        }
        return self.decoder.forward(h);
    }
};

pub const VariationalAutoEncoder = struct {
    pub fn forward(self: VariationalAutoEncoder, model: AutoencoderKLFlux2Model, latents: Tensor) Tensor {
        _ = self;
        var z = latents;

        // Denormalize
        const mean = model.bn.running_mean;
        const var_val = model.bn.running_var;
        const eps = model.config.batch_norm_eps;

        const std_dev = var_val.add(Tensor.scalar(eps, var_val.dtype())).sqrt();

        const C = z.shape().dim(1);
        const mean_b = mean.reshape(.{ 1, C, 1, 1 }).broad(z.shape());
        const std_b = std_dev.reshape(.{ 1, C, 1, 1 }).broad(z.shape());

        z = z.mul(std_b.convert(z.dtype())).add(mean_b.convert(z.dtype()));

        // Unpatchify
        // [B, C, H, W] -> [B, C/4, H*2, W*2]
        const B = z.shape().dim(0);
        const H = z.shape().dim(2);
        const W = z.shape().dim(3);
        const C_out = @divExact(C, 4);

        z = z.reshape(.{ B, C_out, 2, 2, H, W });
        z = z.transpose(.{ 0, 1, 4, 2, 5, 3 });
        z = z.reshape(.{ B, C_out, H * 2, W * 2 });

        return model.decode(z);
    }
};

pub const AutoencoderKLFlux2 = struct {
    model: AutoencoderKLFlux2Model,
    store: zml.io.TensorStore,
    registry: zml.safetensors.TensorRegistry,
    config: Config,
    config_json: std.json.Parsed(Config),
    weights: zml.Bufferized(AutoencoderKLFlux2Model),
    exe: zml.Exe,

    pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
        unloadWeights(allocator, &self.weights);
        self.model.deinit();
        self.store.deinit();
        self.registry.deinit();
        self.config_json.deinit();
        self.exe.deinit();
    }

    pub fn loadFromFile(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, repo_dir: std.Io.Dir, image_height: usize, image_width: usize, progress: ?*std.Progress.Node, options: struct { subfolder: []const u8 = "vae", json_name: []const u8 = "config.json", safetensors_name: []const u8 = "diffusion_pytorch_model.safetensors" }) !@This() {
        const timer_start = std.Io.Clock.awake.now(io);

        var config_json: std.json.Parsed(Config) = try tools.parseConfig(Config, allocator, io, repo_dir, .{ .subfolder = options.subfolder, .json_name = options.json_name });
        errdefer config_json.deinit();

        const vae_dir = try repo_dir.openDir(io, options.subfolder, .{});
        defer vae_dir.close(io);

        var tensor_registry = try zml.safetensors.TensorRegistry.fromFile(allocator, io, vae_dir, options.safetensors_name);
        errdefer tensor_registry.deinit();

        var tensor_store = zml.io.TensorStore.fromRegistry(allocator, &tensor_registry);
        errdefer tensor_store.deinit();

        var model = try AutoencoderKLFlux2Model.init(tensor_store.view(), allocator, config_json.value);
        errdefer model.deinit();

        var weights = try zml.io.load(
            AutoencoderKLFlux2Model,
            &model,
            allocator,
            io,
            platform,
            .{ .parallelism = 1, .store = &tensor_store, .dma_chunks = 4, .dma_chunk_size = 128 * 1024 * 1024, .progress = progress },
        );
        errdefer unloadWeights(allocator, &weights);

        // Compile VAE Decode Step
        const VAEDecodeStep = struct {
            pub fn forward(self: @This(), model_inner: AutoencoderKLFlux2Model, latents_tensor: zml.Tensor) zml.Tensor {
                _ = self;
                return VariationalAutoEncoder.forward(VariationalAutoEncoder{}, model_inner, latents_tensor);
            }
        };

        const batch_size = 1;
        // In Flux signal flow:
        // Image [1, 3, H, W] -> VAE Encode -> [1, 16, H/8, W/8] -> Patchify -> [1, 64, H/16, W/16]
        // Transformer operates on [1, 64, H/16, W/16]
        // Input to VAE Decode is the output of Transformer (unpacked): [1, 64, H/16, W/16]
        // VariationalAutoEncoder.forward handles the unpatchify (depth_to_space) logic.
        const c_dim = @as(usize, @intCast(config_json.value.latent_channels * 4));
        const h_dim = image_height / 16;
        const w_dim = image_width / 16;
        const latents_shape = zml.Shape.init(.{ batch_size, @as(i64, @intCast(c_dim)), @as(i64, @intCast(h_dim)), @as(i64, @intCast(w_dim)) }, .bf16);
        const sym_latents = zml.Tensor.fromShape(latents_shape);

        var vae_exe = try platform.compile(allocator, io, VAEDecodeStep{}, .forward, .{ model, sym_latents });
        errdefer vae_exe.deinit();

        log.info("Loaded AutoencoderKLFlux2 Model in {} ms", .{timer_start.untilNow(io, .awake).toMilliseconds()});

        return .{
            .model = model,
            .store = tensor_store,
            .registry = tensor_registry,
            .config = config_json.value,
            .config_json = config_json,
            .weights = weights,
            .exe = vae_exe,
        };
    }

    pub fn decode(self: *const @This(), allocator: std.mem.Allocator, latents: zml.Buffer) !zml.Buffer {
        var vae_args = try self.exe.args(allocator);
        defer vae_args.deinit(allocator);
        vae_args.set(.{ self.weights, latents });

        var vae_res = try self.exe.results(allocator);
        defer vae_res.deinit(allocator);

        self.exe.call(vae_args, &vae_res);

        return vae_res.get(zml.Buffer);
    }
};
