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

    return .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .model_dir = model_dir,
        .language_model_store = language_model_store,
        .transformer_model_store = transformer_model_store,
    };
}

pub fn deinit(self: *Pipeline, allocator: std.mem.Allocator) void {
    self.language_model_store.deinit(allocator);
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

    // Compile the model
    const rng = zml.Tensor.Rng.init();
    const wrapper = PipelineWrapper{
        .lang_model = self.language_model_store.model,
        .transformer_model = self.transformer_model_store.model,
        .args = params.compile_args,
        .rng = rng,
        .timesteps = timesteps,
        .scheduler = &scheduler,
    };

    const input_tensor = zml.Tensor.init(.{ .bs = 1, .seq_len = tokenized.tokens.len }, .u32);
    const exe = try self.platform.compile(
        self.allocator,
        self.io,
        wrapper,
        .forward,
        .{input_tensor},
        .{ .shardings = self.platform.shardings.values() },
    );
    defer exe.deinit();

    try self.language_model_store.load_buffers(self.allocator, self.io, self.platform);
    try self.transformer_model_store.load_buffers(self.allocator, self.io, self.platform);

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
            .args = params.compile_args_buffers,
            .rng = rng_buffer,
            .timesteps = timesteps,
        },
        bufferized.input_ids,
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
    args: CompileArgs,
    rng: zml.Tensor.Rng,
    timesteps: []f32,
    scheduler: *Scheduler,

    pub fn forward(self: PipelineWrapper, input_ids: zml.Tensor) zml.Tensor {
        const prompt_embeds = self.encode_prompt(input_ids);

        const latent_w = 2 * self.args.width / (self.args.vae_scale_factor * 2);
        const latent_h = 2 * self.args.height / (self.args.vae_scale_factor * 2);
        var latents = self.rng.bitGenerator(.init(.{ .bs = 1, .seq_len = latent_w * latent_h, .d = self.args.in_channels }, .bf16))[1];

        for (self.timesteps, 0..) |t, i| {
            _ = i; // autofix
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

        return latents;
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
