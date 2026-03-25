const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const log = std.log.scoped(.conv_compile_repro);

const Args = struct {
    pub const help =
        \\conv_compile_repro [options]
        \\
        \\Compile the grouped causal conv1d used by Qwen 3.5's linear attention path.
        \\
        \\Options:
        \\  --channels=<n>          Number of channels / feature groups (default: 4096)
        \\  --kernel-size=<n>       Convolution kernel size (default: 4)
        \\  --decode-seqlen=<n>     Sequence length for decode compile (default: 4)
        \\  --prefill-seqlen=<n>    Sequence length for prefill compile (default: 2048)
        \\  --dtype=<name>          Tensor dtype (default: bf16)
        \\
    ;

    channels: i64 = 4096,
    kernel_size: i64 = 4,
    decode_seqlen: i64 = 4,
    prefill_seqlen: i64 = 2048,
    dtype: zml.DataType = .bf16,
};

fn groupedCausalConv1d(input: zml.Tensor, kernel: zml.Tensor) zml.Tensor {
    return zml.Tensor.conv1d(
        input,
        kernel,
        .{
            .padding = &.{ kernel.dim(.kernel_size) - 1, 0 },
            .input_batch_dimension = 0,
            .input_feature_dimension = 2,
            .input_spatial_dimensions = 1,
            .kernel_output_feature_dimension = 0,
            .kernel_input_feature_dimension = 1,
            .kernel_spatial_dimensions = 2,
            .output_batch_dimension = 0,
            .output_feature_dimension = 2,
            .output_spatial_dimensions = 1,
            .feature_group_count = input.dim(.mix),
        },
    );
}

fn compileCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    sharding: zml.sharding.Sharding,
    seq_len: i64,
    channels: i64,
    kernel_size: i64,
    dtype: zml.DataType,
    label: []const u8,
) !void {
    const input: zml.Tensor = .init(.{ .b = 1, .s = seq_len, .mix = channels }, dtype);
    const kernel: zml.Tensor = .init(.{ .out = channels, .in = 1, .kernel_size = kernel_size }, dtype);

    log.info("Compiling {s}: input={f}, kernel={f}", .{ label, input.shape(), kernel.shape() });
    const now: std.Io.Timestamp = .now(io, .awake);

    var exe = try platform.compileFn(
        allocator,
        io,
        groupedCausalConv1d,
        .{ input, kernel },
        .{ .shardings = &.{sharding} },
    );
    defer exe.deinit();

    log.info("Compiled {s} [{f}]", .{ label, now.untilNow(io, .awake) });
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = stdx.flags.parse(init.minimal.args, Args);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    log.info("\n{f}", .{platform.fmtVerbose()});

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    try compileCase(
        allocator,
        io,
        platform,
        replicated_sharding,
        args.decode_seqlen,
        args.channels,
        args.kernel_size,
        args.dtype,
        "decode",
    );
    try compileCase(
        allocator,
        io,
        platform,
        replicated_sharding,
        args.prefill_seqlen,
        args.channels,
        args.kernel_size,
        args.dtype,
        "prefill",
    );
}
