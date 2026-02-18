const std = @import("std");
const zml = @import("zml");

const Flux2Transformer2D = @import("flux2_transformer2d_model.zig").Flux2Transformer2D;
const FlowMatchEulerDiscreteScheduler = @import("scheduling_flow_match_euler_discrete.zig").FlowMatchEulerDiscreteScheduler;
const AutoencoderKLFlux2 = @import("autoencoder_kl_flux2.zig").AutoencoderKLFlux2;
const Qwen2TokenizerFast = @import("tokenization_qwen2_fast.zig").Qwen2TokenizerFast;
const Qwen3ForCausalLM = @import("modeling_qwen3.zig").Qwen3ForCausalLM;
const utils = @import("utils.zig");

const log = std.log.scoped(.flux_pipeline);

pub const FluxPipeline = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,

    tokenizer: Qwen2TokenizerFast,
    qwen3: Qwen3ForCausalLM.ModelContext,
    transformer: Flux2Transformer2D,
    scheduler: *FlowMatchEulerDiscreteScheduler,
    vae: AutoencoderKLFlux2,
    config: GenerationOptions,

    pub const GenerationOptions = struct {
        prompt: []const u8,
        output_image_path: ?[]const u8 = null,
        kitty_output: bool = false,
        num_inference_steps: usize = 4,
        random_seed: u64 = 0,
        generator_type: utils.GeneratorType = .torch,
        max_sequence_length: usize = 256,
        output_image_dim: utils.ResolutionInfo = .{ .width = 1024, .height = 1024 },
    };

    pub fn deinit(self: *FluxPipeline) void {
        self.tokenizer.deinit();
        self.qwen3.deinit(self.allocator);
        self.transformer.deinit(self.allocator);
        self.scheduler.deinit();
        self.vae.deinit(self.allocator);
    }

    pub fn generate(self: *FluxPipeline, options: ?GenerationOptions) !void {
        const timer_compute_start = std.Io.Clock.awake.now(self.io);
        defer log.info(" ================= Total compute time (tokenization to RGB conversion) is {d:.2} ms. ================= ", .{timer_compute_start.untilNow(self.io, .awake).toMilliseconds()});

        const _options = options orelse self.config;
        // ==================== Tokenizing Prompt ====================
        const messages = [_]Qwen2TokenizerFast.ChatMessage{
            .{ .role = "user", .content = _options.prompt },
        };

        const text_templated = try self.tokenizer.applyChatTemplate(&messages, .{
            .add_generation_prompt = true,
        });
        defer self.allocator.free(text_templated);

        var tokens = try self.tokenizer.tokenize(self.io, self.platform, text_templated, .{
            .max_length = _options.max_sequence_length,
            .truncation = true,
        });
        defer tokens.deinit();

        // ==================== Encoding Prompt ====================

        const timer_qwen3_start = std.Io.Clock.awake.now(self.io);
        var embeding_output: Qwen3ForCausalLM.EmbedingOutput = try Qwen3ForCausalLM.pipelineRun(self.allocator, self.io, self.platform, self.qwen3, tokens);
        log.info("Qwen3 completed in {d:.2} ms.", .{timer_qwen3_start.untilNow(self.io, .awake).toMilliseconds()});
        defer embeding_output.deinit();

        // ==================== Prepare Latents ====================
        const timer_prepare_latents_start = std.Io.Clock.awake.now(self.io);
        var latent_buf, var latent_ids_buf = try utils.get_latents(self.allocator, self.io, self.platform, self.transformer.config, _options.output_image_dim, _options.generator_type, _options.random_seed);
        log.info("Latents prepared in {d:.2} ms.", .{timer_prepare_latents_start.untilNow(self.io, .awake).toMilliseconds()});
        defer latent_buf.deinit();
        defer latent_ids_buf.deinit();

        // ==================== Scheduler ====================
        const timer_scheduler_start = std.Io.Clock.awake.now(self.io);
        var latents_out = try utils.schedule(
            &self.transformer,
            self.scheduler,
            latent_buf,
            latent_ids_buf,
            embeding_output.text_embedding,
            embeding_output.text_ids,
            _options.num_inference_steps,
            self.allocator,
            self.io,
            self.platform,
        );
        log.info("Scheduler completed in {d:.2} ms.", .{timer_scheduler_start.untilNow(self.io, .awake).toMilliseconds()});
        defer latents_out.deinit();

        // ==================== VAE Decode ====================
        const timer_vae_start = std.Io.Clock.awake.now(self.io);
        var image_decoded_buf: zml.Buffer = try self.vae.decode(self.allocator, latents_out);
        log.info("VAE completed in {d:.2} ms.", .{timer_vae_start.untilNow(self.io, .awake).toMilliseconds()});
        defer image_decoded_buf.deinit();

        // ==================== RGB Conversion & Output ====================
        const timer_rgb_start = std.Io.Clock.awake.now(self.io);
        const rgb_image_buffer = try utils.decodeImageToRgb(self.allocator, self.io, image_decoded_buf);
        log.info("RGB conversion completed in {d:.2} ms.", .{timer_rgb_start.untilNow(self.io, .awake).toMilliseconds()});
        defer rgb_image_buffer.free(self.allocator);

        if (_options.kitty_output) {
            log.info(">>> Printing Image to Terminal...", .{});
            try utils.printFluxImageToTerminalKittyFromBuffer(self.allocator, &rgb_image_buffer);
        }

        if (_options.output_image_path) |output_image_path| {
            log.info(">>> Saving Image to Disk at {s}...", .{output_image_path});
            try utils.saveFluxImageToPng(self.allocator, &rgb_image_buffer, output_image_path);
        }
    }
};
