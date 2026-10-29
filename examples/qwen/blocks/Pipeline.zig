const std = @import("std");
const stdx = @import("stdx");
const zml = @import("zml");
const c = @import("c");

const tokenizer = @import("./tokenizer.zig");
const Scheduler = @import("./Scheduler.zig");
const CpuGenerator = @import("./noise/CpuGenerator.zig");
const GpuGenerator = @import("./noise/GpuGenerator.zig");
const CompileArgs = @import("../compiler.zig").CompileArgs;
const blocks = @import("../blocks.zig");
const store = @import("../store.zig");

const Pipeline = @This();

allocator: std.mem.Allocator,
io: std.Io,
platform: *zml.Platform,

model_dir: []const u8,

language_model_store: store.LanguageModelStore,
transformer_model_store: store.TransformerStore,
vae_model_store: store.VaeModelStore,

pub fn init(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    model_dir: [:0]const u8,
) !Pipeline {
    var language_model_store: store.LanguageModelStore = undefined;
    try language_model_store.load_tensors(allocator, io, platform.meshes.values(), model_dir);

    var transformer_model_store: store.TransformerStore = undefined;
    try transformer_model_store.load_tensors(allocator, io, platform.meshes.values(), model_dir);

    var vae_model_store: store.VaeModelStore = undefined;
    try vae_model_store.load_tensors(allocator, io, platform.meshes.values(), model_dir);

    return .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .model_dir = model_dir,
        .language_model_store = language_model_store,
        .transformer_model_store = transformer_model_store,
        .vae_model_store = vae_model_store,
    };
}

pub fn deinit(self: *Pipeline, allocator: std.mem.Allocator) void {
    self.language_model_store.deinit(allocator);
    self.transformer_model_store.deinit(allocator);
    self.vae_model_store.deinit(allocator);
}

/// The `torch.Generator` device diffusers draws the initial noise with.
pub const Noise = enum { cpu, cuda };

const Params = struct {
    seq_len: usize,
    compile_args: CompileArgs,
    compile_args_buffers: zml.Bufferized(CompileArgs),
    /// Same noise as diffusers with `generator=torch.Generator(noise).manual_seed(seed)`
    seed: u64 = 42,
    noise: Noise = .cpu,
    /// The GPU the `.cuda` noise is replayed for: torch's CUDA randn depends on its SM count (see `noise/gpu.zig`).
    cuda_device: GpuGenerator.Device = .rtx_5090,
};

pub fn run_once(
    self: *Pipeline,
    prompt: []const u8,
    params: Params,
) !void {
    // Tokenizer
    var tokenized = try tokenizer.tokenize(self.allocator, self.io, self.model_dir, prompt, .{ .templated = true });
    defer tokenized.deinit(self.allocator);

    var scheduler = Scheduler{
        .num_train_steps = @intCast(params.compile_args.num_inference_steps),
        .base_seq_len = @floatFromInt(params.compile_args.base_image_seq_len orelse 256),
        .max_seq_len = @floatFromInt(params.compile_args.max_image_seq_len orelse 4096),
        .base_shift = params.compile_args.base_shift orelse 0.5,
        .max_shift = params.compile_args.max_shift orelse 1.15,
    };
    const timesteps = try scheduler.init_timesteps(
        self.allocator,
        @floatFromInt(params.compile_args.head_dim * params.compile_args.head_dim),
    );
    defer scheduler.deinit(self.allocator);

    // Compile the model
    const wrapper = PipelineWrapper{
        .lang_model = self.language_model_store.model,
        .transformer_model = self.transformer_model_store.model,
        .vae_model = self.vae_model_store.model,
        .args = params.compile_args,
        .timesteps = timesteps,
        .scheduler = &scheduler,
    };

    const latents_mean_shape = zml.shape.Shape.init(.{ .bs = 1, .c = latents_mean_data.len, .t = 1, .kx = 1, .ky = 1 }, .f32);
    const latents_mean_slice = try zml.Slice.alloc(self.allocator, latents_mean_shape);
    defer latents_mean_slice.free(self.allocator);
    @memcpy(latents_mean_slice.items(f32)[0..latents_mean_data.len], &latents_mean_data.*);
    var latents_mean_buffer = try zml.Buffer.fromSlice(self.io, self.platform, latents_mean_slice);
    defer latents_mean_buffer.deinit();
    const latents_mean_tensor = zml.Tensor.init(latents_mean_shape, .f32);

    const latents_std_shape = zml.shape.Shape.init(.{ .bs = 1, .c = latents_std_data.len, .t = 1, .kx = 1, .ky = 1 }, .f32);
    const latents_std_slice = try zml.Slice.alloc(self.allocator, latents_std_shape);
    defer latents_std_slice.free(self.allocator);
    @memcpy(latents_std_slice.items(f32)[0..latents_std_data.len], &latents_std_data.*);
    var latents_std_buffer = try zml.Buffer.fromSlice(self.io, self.platform, latents_std_slice);
    defer latents_std_buffer.deinit();
    const latents_std_tensor = zml.Tensor.init(latents_std_shape, .f32);

    // diffusers' prepare_latents: randn((bs, 1, C, H, W)), flattened to (bs, C, H * W)
    const latent_h = 2 * (params.compile_args.height / (params.compile_args.vae_scale_factor * 2));
    const latent_w = 2 * (params.compile_args.width / (params.compile_args.vae_scale_factor * 2));
    const noise_shape = zml.Shape.init(.{ .bs = 1, .d = params.compile_args.in_channels, .seq_len = latent_h * latent_w }, .f32);
    const noise_data = switch (params.noise) {
        .cpu => cpu: {
            var generator: CpuGenerator = .init(params.seed);
            break :cpu try generator.randn(self.allocator, @intCast(noise_shape.count()));
        },
        .cuda => cuda: {
            var generator: GpuGenerator = .init(params.seed, params.cuda_device);
            break :cuda try generator.randn(self.allocator, @intCast(noise_shape.count()));
        },
    };
    defer self.allocator.free(noise_data);
    const noise_slice = zml.Slice.init(noise_shape, std.mem.sliceAsBytes(noise_data));
    var noise_buffer = try zml.Buffer.fromSlice(self.io, self.platform, noise_slice);
    defer noise_buffer.deinit();
    const noise_tensor: zml.Tensor = .fromShape(noise_shape);

    std.debug.print("\n\nStart compile\n\n", .{});

    const input_tensor = zml.Tensor.init(.{ .bs = 1, .seq_len = tokenized.tokens.len }, .u32);
    const exe = try self.platform.compile(
        self.allocator,
        self.io,
        wrapper,
        .forward,
        .{ input_tensor, noise_tensor, latents_mean_tensor, latents_std_tensor },
        .{ .meshes = self.platform.meshes.values() },
    );
    defer exe.deinit();

    std.debug.print("\n\nEnd compile\n\n", .{});

    try self.language_model_store.load_buffers(self.allocator, self.io, self.platform);
    defer self.language_model_store.deinit(self.allocator);
    try self.transformer_model_store.load_buffers(self.allocator, self.io, self.platform);
    defer self.transformer_model_store.deinit(self.allocator);
    try self.vae_model_store.load_buffers(self.allocator, self.io, self.platform);
    defer self.vae_model_store.deinit(self.allocator);

    // Create the buffers
    var bufferized = try tokenizer.bufferize(self.allocator, self.io, self.platform, tokenized);
    defer bufferized.deinit();

    // Run the model
    var exe_args = try exe.args(self.allocator);
    defer exe_args.deinit(self.allocator);

    var exe_results = try exe.results(self.allocator);
    defer exe_results.deinit(self.allocator);

    exe_args.set(.{
        .{
            .lang_model = self.language_model_store.model_buffer,
            .transformer_model = self.transformer_model_store.model_buffer,
            .vae_model = self.vae_model_store.model_buffer,
            .args = params.compile_args_buffers,
            .timesteps = timesteps,
        },
        bufferized.input_ids,
        noise_buffer,
        latents_mean_buffer,
        latents_std_buffer,
    });

    std.debug.print("\n\nStart exec\n\n", .{});
    exe.call(self.io, exe_args, &exe_results, .{ .wait = true });
    std.debug.print("\n\nEnd exec\n\n", .{});

    const results = try self.allocator.alloc(zml.Buffer, 1);
    defer self.allocator.free(results);

    exe_results.fill(.{results});
    const image = results[0];
    const slice = try image.toSliceAlloc(self.allocator, self.io);
    defer slice.free(self.allocator);
    std.debug.print("results[0]._shape: {f}\n", .{image._shape});

    const path: [:0]const u8 = "/home/erwan/zml/examples/qwen/pytorch/skusku.png";
    const shape = image.shape();
    const res = c.stbi_write_png(path.ptr, @intCast(shape.dim(.ky)), @intCast(shape.dim(.kx)), @intCast(shape.dim(.c)), @constCast(slice.bytes.ptr), 0);
    if (res != 0) {
        std.debug.print("Image {s} saved\n", .{path});
    } else {
        std.debug.print("Image could not be saved\n", .{});
    }
}

// This value is the start of the system prompt
const DROP_IDX = 14;

/// One denoising step per iteration: transformer forward then the scheduler's Euler step.
const Denoise = struct {
    transformer_model: blocks.QwenImage21Transformer2DModel,
    args: CompileArgs,
    prompt_embeds: zml.Tensor,
    timesteps: zml.Tensor, // {.step}, bf16
    sigmas: zml.Tensor, // {.step = timesteps + 1}, f32

    pub const State = struct {
        step: zml.Tensor,
        latents: zml.Tensor,
    };

    pub fn cond(self: Denoise, state: State) zml.Tensor {
        return state.step.cmp(.LT, zml.Tensor.scalar(self.timesteps.dim(.step), .i32));
    }

    pub fn body(self: Denoise, state: State) State {
        const latents = state.latents;
        const timestep = self.timesteps.slice(.step, .dynSingle(state.step))
            .divByConst(1000)
            .broad(.init(.{ .bs = latents.dim(0) }, .bf16));
        const noise_pred = self.transformer_model.forward(
            latents,
            self.prompt_embeds,
            timestep,
            zml.Tensor.zeroes(.init(.{ .bs = 1 }, .f32)), // unused
            self.args,
        )
            // The size after the forward is [1, TextSeqLen + ImgSeqLen, 64] we only care about [1, ImgSeqLen, 64] so we skip TextSeqLen
            .slice(1, .{ .start = self.prompt_embeds.dim(.seq_len) });

        return .{
            .step = state.step.addConstant(1),
            .latents = Scheduler.step(self.sigmas, state.step, noise_pred, latents),
        };
    }
};

pub const PipelineWrapper = struct {
    lang_model: blocks.Qwen3VLModel,
    transformer_model: blocks.QwenImage21Transformer2DModel,
    vae_model: blocks.AutoencoderKLQwenImage21,
    args: CompileArgs,
    timesteps: []f32,
    scheduler: *Scheduler,

    pub fn forward(
        self: PipelineWrapper,
        input_ids: zml.Tensor,
        noise: zml.Tensor,
        latents_mean: zml.Tensor,
        latents_std: zml.Tensor,
    ) zml.Tensor {
        const prompt_embeds = self.encode_prompt(input_ids);

        // `noise` is {bs, d, seq_len} f32 from CpuGenerator. diffusers' _pack_latents transposes it to
        // {bs, seq_len, d}, and torch's bf16 randn is the f32 sample rounded to nearest-even, like convert.
        const latents = noise.transpose(.{ .bs, .seq_len, .d }).convert(.bf16);

        const img = generate(self.transformer_model, self.vae_model, self.args, self.scheduler, prompt_embeds, latents, latents_mean, latents_std);
        return to_pixels(img);
    }

    // The python impl does nothing more since we don't have images in the prompt
    fn encode_prompt(self: PipelineWrapper, input_ids: zml.Tensor) zml.Tensor {
        return self.get_qwen_prompt_embeds(input_ids);
    }

    fn get_qwen_prompt_embeds(self: PipelineWrapper, input_ids: zml.Tensor) zml.Tensor {
        const lang = self.lang_model.forward(input_ids, self.args);
        return lang.slice(.seq_len, .{ .start = DROP_IDX });
    }

    fn calculate_shift(self: PipelineWrapper, image_seq_len: f32) f32 {
        const base_seq_len: f32 = @floatFromInt(self.args.base_image_seq_len orelse 256);
        const max_seq_len: f32 = @floatFromInt(self.args.max_image_seq_len orelse 4096);
        const base_shift = self.args.base_shift orelse 0.5;
        const max_shift = self.args.max_shift orelse 1.15;

        const m = (max_shift - base_shift) / (max_seq_len - base_seq_len);
        const b = base_shift - m * base_seq_len;
        return image_seq_len * m + b;
    }
};

/// Denoises `latents` ({bs, seq_len, d} bf16, packed like diffusers' `prepare_latents`) conditioned on
/// `prompt_embeds`, then decodes them. Returns the VAE output `vae.decode(latents)[0][:, :, 0]`:
/// {bs, c, kx, ky} f32 in [-1, 1].
pub fn generate(
    transformer_model: blocks.QwenImage21Transformer2DModel,
    vae_model: blocks.AutoencoderKLQwenImage21,
    args: CompileArgs,
    scheduler: *const Scheduler,
    prompt_embeds: zml.Tensor,
    latents: zml.Tensor,
    latents_mean: zml.Tensor,
    latents_std: zml.Tensor,
) zml.Tensor {
    const vae_input = denoise(transformer_model, args, scheduler, prompt_embeds, latents, latents_mean, latents_std);
    const img = vae_model.forward(vae_input);
    stdx.debug.assert(img.dim(.t) == 1, "Expected only one temporal dimension, got {}", .{img.dim(.t)});
    return img.squeeze(.t);
}

/// Runs the denoising loop and un-normalizes the result. Returns the VAE input
/// ({bs, c, t = 1, kx, ky} f32), `latents * latents_std + latents_mean` in diffusers.
pub fn denoise(
    transformer_model: blocks.QwenImage21Transformer2DModel,
    args: CompileArgs,
    scheduler: *const Scheduler,
    prompt_embeds: zml.Tensor,
    latents_: zml.Tensor,
    latents_mean: zml.Tensor,
    latents_std: zml.Tensor,
) zml.Tensor {
    // A `stablehlo.while` traces the transformer once instead of unrolling one copy per timestep.
    const timesteps: zml.Tensor = .constantTensor(.init(.{ .step = scheduler.timesteps.len }, .f32), std.mem.sliceAsBytes(scheduler.timesteps));
    const latents = zml.ops.@"while"(Denoise, .{
        .transformer_model = transformer_model,
        .args = args,
        .prompt_embeds = prompt_embeds,
        .timesteps = timesteps.convert(.bf16),
        .sigmas = scheduler.sigmasTensor(),
    }, .{
        .step = .scalar(0, .i32),
        .latents = latents_,
    }).latents;

    const unpacked = unpackLatents(args, latents).convert(.f32);
    return unpacked.mul(latents_std.broad(unpacked.shape())).add(latents_mean.broad(unpacked.shape()));
}

/// {bs, seq_len, d} -> {bs, c, t = 1, kx, ky}, same as diffusers' `_unpack_latents` (2.1 latents are not patched).
fn unpackLatents(args: CompileArgs, latents: zml.Tensor) zml.Tensor {
    const h: i64 = @intCast(2 * (args.height / (args.vae_scale_factor * 2)));
    return latents.transpose(.{ .bs, .d, .seq_len }).unflatten(.seq_len, h).withTags(.{ .bs, .c, .kx, .ky }).insertAxes(.kx, .{.t});
}

/// Same as diffusers' `VaeImageProcessor.postprocess` to PIL: [-1, 1] -> [0, 255] u8, channels last.
/// {bs = 1, c, kx, ky} f32 -> {kx, ky, c} u8, ready for stb.
fn to_pixels(img: zml.Tensor) zml.Tensor {
    const normalized = img.scale(0.5).addConstant(0.5).clamp(
        zml.Tensor.scalar(0, .f32).broad(img.shape()),
        zml.Tensor.scalar(1, .f32).broad(img.shape()),
    );
    return normalized.squeeze(.bs).transpose(.{ .kx, .ky, .c }).scale(255).round().convert(.u8);
}

pub const latents_mean_data: *const [64]f32 = &.{ 0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.651, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304 };
pub const latents_std_data: *const [64]f32 = &.{ 3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308, 3.0904, 2.8778, 3.7675, 3.732, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489, 3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467, 3.9037, 3.2947, 5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875, 3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.408, 3.4959, 3.8747, 3.7608, 3.5735, 3.149, 3.7662, 3.6746, 3.4563, 3.8161 };
