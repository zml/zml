const std = @import("std");

const zml = @import("zml");
const meta = zml.meta;
const asynk = @import("async");
const flags = @import("tigerbeetle/flags");

const log = std.log.scoped(.sdxl);

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain, .{});
}

pub fn asyncMain() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const tmp = try std.fs.openDirAbsolute("/tmp", .{});
    try tmp.makePath("zml/sdxl/cache");
    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform().withCompilationOptions(.{
        .cache_location = "/tmp/zml/sdxl/cache",
        .xla_dump_to = "/tmp/zml/sdxl",
    });
    _ = platform; // autofix

    const args = try std.process.argsAlloc(allocator);
    const vae_model_path = args[1];
    const unet_model_path = args[2];
    _ = unet_model_path; // autofix
    const prompt_encoders_model_path = args[3];
    _ = prompt_encoders_model_path; // autofix
    const vocab_path = args[4];
    _ = vocab_path; // autofix
    const prompt = args[5];

    log.info("Prompt: {s}", .{prompt});

    var vae_weights = try zml.aio.detectFormatAndOpen(allocator, vae_model_path);
    defer vae_weights.deinit();

    log.info("Loaded VAE from {s}, found {} buffers.", .{ vae_model_path, vae_weights.buffers.count() });
}
