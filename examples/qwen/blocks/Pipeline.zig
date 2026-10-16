const std = @import("std");
const stdx = @import("stdx");
const zml = @import("zml");

const tokenizer = @import("./tokenizer.zig");
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
    var tokenized = try tokenizer.tokenize(self.allocator, self.io, self.model_dir, prompt, .{});
    defer tokenized.deinit(self.allocator);

    // Compile the model
    const wrapper = PipelineWrapper{
        .model = self.language_model_store.model,
        .args = params.compile_args,
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

    exe_args.set(.{ .{ .model = self.language_model_store.model_buffer, .args = params.compile_args_buffers }, bufferized.input_ids });

    std.debug.print("\n\nStart exec\n\n", .{});
    exe.callOpts(self.io, exe_args, &exe_results, .{ .wait = true });
    std.debug.print("\n\nEnd exec\n\n", .{});

    const results = try self.allocator.alloc(zml.Buffer, 1);
    defer self.allocator.free(results);

    exe_results.fill(.{results});
    std.debug.print("{f}\n", .{results[0]._shape});
}

const PipelineWrapper = struct {
    model: blocks.Qwen3VLModel,
    args: CompileArgs,

    pub fn forward(self: PipelineWrapper, input_ids: zml.Tensor) zml.Tensor {
        return self.model.forward(input_ids, self.args);
    }
};
