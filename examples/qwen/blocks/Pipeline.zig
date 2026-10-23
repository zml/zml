const std = @import("std");
const stdx = @import("stdx");
const zml = @import("zml");

const tokenizer = @import("./tokenizer.zig");
const Scheduler = @import("./Scheduler.zig");
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
    try language_model_store.load_tensors(allocator, io, model_dir);

    var transformer_model_store: store.TransformerStore = undefined;
    try transformer_model_store.load_tensors(allocator, io, model_dir);

    var vae_model_store: store.VaeModelStore = undefined;
    try vae_model_store.load_tensors(allocator, io, model_dir);

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

const Params = struct {
    seq_len: usize,
    compile_args: CompileArgs,
    compile_args_buffers: zml.Bufferized(CompileArgs),
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
        .num_train_steps = @floatFromInt(params.compile_args.num_inference_steps),
        .base_seq_len = @floatFromInt(params.compile_args.base_image_seq_len orelse 256),
        .max_seq_len = @floatFromInt(params.compile_args.max_image_seq_len orelse 4096),
        .base_shift = params.compile_args.base_shift orelse 0.5,
        .max_shift = params.compile_args.max_shift orelse 1.15,
    };
    const timesteps = try scheduler.init_timesteps(
        self.allocator,
        @intCast(params.compile_args.num_inference_steps),
        @floatFromInt(params.compile_args.head_dim * params.compile_args.head_dim),
    );
    defer scheduler.deinit(self.allocator);

    // Compile the model
    const rng = zml.Tensor.Rng.init();
    const wrapper = PipelineWrapper{
        .lang_model = self.language_model_store.model,
        .transformer_model = self.transformer_model_store.model,
        .vae_model = self.vae_model_store.model,
        .args = params.compile_args,
        .rng = rng,
        .timesteps = timesteps,
        .scheduler = &scheduler,
    };

    const latents_mean_shape = zml.shape.Shape.init(.{ .bs = 1, .c = latents_mean_data.len, .t = 1, .kx = 1, .ky = 1 }, .f32);
    const latents_mean_slice = try zml.Slice.alloc(self.allocator, latents_mean_shape);
    defer latents_mean_slice.free(self.allocator);
    @memcpy(latents_mean_slice.items(f32)[0..latents_mean_data.len], &latents_mean_data.*);
    const latents_mean_buffer = try zml.Buffer.fromSlice(self.io, self.platform, latents_mean_slice, .replicated);
    const latents_mean_tensor = zml.Tensor.init(latents_mean_shape, .f32);

    const latents_std_shape = zml.shape.Shape.init(.{ .bs = 1, .c = latents_std_data.len, .t = 1, .kx = 1, .ky = 1 }, .f32);
    const latents_std_slice = try zml.Slice.alloc(self.allocator, latents_std_shape);
    defer latents_std_slice.free(self.allocator);
    @memcpy(latents_std_slice.items(f32)[0..latents_std_data.len], &latents_std_data.*);
    const latents_std_buffer = try zml.Buffer.fromSlice(self.io, self.platform, latents_std_slice, .replicated);
    const latents_std_tensor = zml.Tensor.init(latents_std_shape, .f32);

    std.debug.print("\n\nStart compile\n\n", .{});

    const input_tensor = zml.Tensor.init(.{ .bs = 1, .seq_len = tokenized.tokens.len }, .u32);
    const exe = try self.platform.compile(
        self.allocator,
        self.io,
        wrapper,
        .forward,
        .{ input_tensor, latents_mean_tensor, latents_std_tensor },
        .{ .shardings = self.platform.shardings.values() },
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

    var rng_buffer = try zml.Tensor.Rng.initBuffer(self.io, self.platform, .replicated, 43);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffer);

    exe_args.set(.{
        .{
            .lang_model = self.language_model_store.model_buffer,
            .transformer_model = self.transformer_model_store.model_buffer,
            .vae_model = self.vae_model_store.model_buffer,
            .args = params.compile_args_buffers,
            .rng = rng_buffer,
            .timesteps = timesteps,
        },
        bufferized.input_ids,
        latents_mean_buffer,
        latents_std_buffer,
    });

    std.debug.print("\n\nStart exec\n\n", .{});
    exe.callOpts(self.io, exe_args, &exe_results, .{ .wait = true });
    std.debug.print("\n\nEnd exec\n\n", .{});

    const results = try self.allocator.alloc(zml.Buffer, 1);
    defer self.allocator.free(results);

    exe_results.fill(.{results});
    std.debug.print("results[0]._shape: {f}\n", .{results[0]._shape});
}

// This value is the start of the system prompt
const DROP_IDX = 14;

const PipelineWrapper = struct {
    lang_model: blocks.Qwen3VLModel,
    transformer_model: blocks.QwenImage21Transformer2DModel,
    vae_model: blocks.AutoencoderKLQwenImage21,
    args: CompileArgs,
    rng: zml.Tensor.Rng,
    timesteps: []f32,
    scheduler: *Scheduler,

    pub fn forward(
        self: PipelineWrapper,
        input_ids: zml.Tensor,
        latents_mean: zml.Tensor,
        latents_std: zml.Tensor,
    ) zml.Tensor {
        const prompt_embeds = self.encode_prompt(input_ids);

        const latent_w = 2 * self.args.width / (self.args.vae_scale_factor * 2);
        const latent_h = 2 * self.args.height / (self.args.vae_scale_factor * 2);
        var latents = self.rng.bitGenerator(.init(.{ .bs = 1, .seq_len = latent_w * latent_h, .d = self.args.in_channels }, .bf16))[1];

        for (self.timesteps) |t| {
            const timestep = zml.Tensor.scalar(t, .bf16).broad(.init(.{ .bs = latents.dim(0) }, .bf16));
            const noise_pred = self.transformer_model.forward(
                latents,
                prompt_embeds,
                timestep,
                zml.Tensor.zeroes(.init(.{ .bs = 1 }, .f32)), // unused
                self.args,
            )
                // The size after the forward is [1, TextSeqLen + ImgSeqLen, 64] we only care about [1, ImgSeqLen, 64] so we skip TextSeqLen
                .slice(1, .{ .start = prompt_embeds.dim(.seq_len) });

            latents = self.scheduler.step(noise_pred, latents);
        }

        const unpacked = self._unpack_latents(latents).convert(.f32);
        const std_mean = unpacked.mul(latents_std.broad(unpacked.shape())).add(latents_mean.broad(unpacked.shape()));
        const img = self.vae_model.forward(std_mean);
        stdx.debug.assert(img.dim(.t) == 1, "Expected only one temporal dimension, got {}", .{img.dim(.t)});

        return img;
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

    fn _unpack_latents(self: PipelineWrapper, latents: zml.Tensor) zml.Tensor {
        // const w = self.args.width / (self.args.vae_scale_factor * 2);
        const h: i64 = @intCast(self.args.height / (self.args.vae_scale_factor * 2));
        return latents.transpose(.{ .bs, .d, .seq_len }).unflatten(.seq_len, h * 2).withTags(.{ .bs, .c, .kx, .ky }).insertAxes(.kx, .{.t});
    }
};

const latents_mean_data: *const [64]f32 = &.{ 0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.651, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304 };
const latents_std_data: *const [64]f32 = &.{ 3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308, 3.0904, 2.8778, 3.7675, 3.732, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489, 3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467, 3.9037, 3.2947, 5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875, 3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.408, 3.4959, 3.8747, 3.7608, 3.5735, 3.149, 3.7662, 3.6746, 3.4563, 3.8161 };
