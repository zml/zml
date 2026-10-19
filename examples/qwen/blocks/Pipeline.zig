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

pub fn init(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    model_dir: [:0]const u8,
) !Pipeline {
    var language_model_store: store.LanguageModelStore = undefined;
    try language_model_store.load(allocator, io, platform, model_dir);

    return .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .model_dir = model_dir,
        .language_model_store = language_model_store,
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
    self: Pipeline,
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
        .args = params.compile_args,
        .rng = rng,
        .timesteps = timesteps,
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
            .model = self.language_model_store.model_buffer,
            .args = params.compile_args_buffers,
            .rng = rng,
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
    args: CompileArgs,
    rng: zml.Tensor.Rng,
    timesteps: []f32,

    pub fn forward(self: PipelineWrapper, input_ids: zml.Tensor) zml.Tensor {
        const prompt_embeds = self.encode_prompt(input_ids);
        // std.debug.print("prompt_embeds: {f}\n", .{prompt_embeds.shape()}); // prompt_embeds: {bs=1,seq_len=31,d=4096,bf16}

        const latent_w = 2 * self.args.width / (self.args.vae_scale_factor * 2);
        const latent_h = 2 * self.args.height / (self.args.vae_scale_factor * 2);
        const latents = self.rng.bitGenerator(.init(.{ .bs = 1, .d = latent_w * latent_h, .latents = self.args.in_channels }, .f32))[1];
        // std.debug.print("latents: {f}\n", .{latents.shape()}); // latents: {bs=1,d=16384,latents=64,f32}

        const image_pad_mask = zml.Tensor.concatenate(&.{
            zml.Tensor.constant(.init(.bool, false)).broad(.init(.{ .seq_len = prompt_embeds.dim(.seq_len) }, .bool)),
            zml.Tensor.constant(.init(.bool, true)).broad(.init(.{ .seq_len = @divExact(latents.dim(.d), 4) }, .bool)),
        }, .seq_len);
        _ = image_pad_mask; // autofix
        // std.debug.print("image_pad_mask: {f}\n", .{image_pad_mask.shape()}); // image_pad_mask: {seq_len=4127,bool}

        for (self.timesteps, 0..) |t, i| {
            _ = t; // autofix
            _ = i; // autofix
            // self.lang_model
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
