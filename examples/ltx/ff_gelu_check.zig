const std = @import("std");
const zml = @import("zml");
const model = @import("model.zig");
const check_utils = @import("check_utils.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Mode = enum {
    bf16,
    f32,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var it = init.minimal.args.iterate();
    _ = it.next(); // exe

    const net0_fixture_path = it.next() orelse {
        std.log.err("Usage: bazel run //examples/ltx:ff_gelu_check -- <ff_net0_fixture.safetensors> <ff_net2_fixture.safetensors> [token_limit] [bf16|f32]", .{});
        return error.InvalidArgs;
    };

    const net2_fixture_path = it.next() orelse {
        std.log.err("Usage: bazel run //examples/ltx:ff_gelu_check -- <ff_net0_fixture.safetensors> <ff_net2_fixture.safetensors> [token_limit] [bf16|f32]", .{});
        return error.InvalidArgs;
    };

    var token_limit: ?usize = null;
    var mode: Mode = .bf16;

    if (it.next()) |arg3| {
        token_limit = std.fmt.parseInt(usize, arg3, 10) catch null;
        if (token_limit == null) {
            mode = try parseMode(arg3);
        }
    }

    if (it.next()) |arg4| {
        mode = try parseMode(arg4);
    }

    if (it.next() != null) {
        std.log.err("Too many arguments", .{});
        return error.InvalidArgs;
    }

    var net0_registry: zml.safetensors.TensorRegistry = zml.safetensors.TensorRegistry.fromPath(allocator, io, net0_fixture_path) catch |err| {
        std.log.err("Failed to open net0 fixture: {s}", .{net0_fixture_path});
        return err;
    };
    defer net0_registry.deinit();

    var net2_registry: zml.safetensors.TensorRegistry = zml.safetensors.TensorRegistry.fromPath(allocator, io, net2_fixture_path) catch |err| {
        std.log.err("Failed to open net2 fixture: {s}", .{net2_fixture_path});
        return err;
    };
    defer net2_registry.deinit();

    var net0_store: zml.io.TensorStore = .fromRegistry(allocator, &net0_registry);
    defer net0_store.deinit();

    var net2_store: zml.io.TensorStore = .fromRegistry(allocator, &net2_registry);
    defer net2_store.deinit();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    // net0 fixture output is pre-GELU h1 from ff.net.0.proj.
    var gelu_input = try check_utils.loadBufferFromStore(allocator, io, platform, &net0_store, "ff.output0", replicated_sharding);
    defer gelu_input.deinit();

    // net2 fixture input is post-GELU h2 fed into ff.net.2.
    var gelu_expected = try check_utils.loadBufferFromStore(allocator, io, platform, &net2_store, "ff.input0", replicated_sharding);
    defer gelu_expected.deinit();

    if (token_limit) |limit| {
        gelu_input = try check_utils.sliceTokenPrefix(io, platform, gelu_input, replicated_sharding, limit);
        gelu_expected = try check_utils.sliceTokenPrefix(io, platform, gelu_expected, replicated_sharding, limit);
        std.log.info("Using token_limit={d}; sliced fixture tensors", .{limit});
    }

    const input_tensor = zml.Tensor.fromShape(gelu_input.shape());
    std.log.info("Compiling GELU graph (mode={s})...", .{@tagName(mode)});

    var exe = switch (mode) {
        .bf16 => try platform.compileFn(
            allocator,
            io,
            model.forwardFFGeluBf16,
            .{input_tensor},
            .{ .shardings = &.{replicated_sharding} },
        ),
        .f32 => try platform.compileFn(
            allocator,
            io,
            model.forwardFFGeluF32,
            .{input_tensor},
            .{ .shardings = &.{replicated_sharding} },
        ),
    };
    defer exe.deinit();
    std.log.info("Compile completed", .{});

    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    args.set(.{gelu_input});
    std.log.info("Executing GELU forward...", .{});
    exe.call(args, &results);
    std.log.info("Execution completed", .{});

    var gelu_output = results.get(zml.Buffer);
    defer gelu_output.deinit();

    std.log.info("Starting strict comparison...", .{});
    try zml.testing.expectClose(io, gelu_output, gelu_expected, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });

    std.log.info("FF GELU parity PASSED (mode={s})", .{@tagName(mode)});
}

fn parseMode(v: []const u8) !Mode {
    if (std.mem.eql(u8, v, "bf16")) return .bf16;
    if (std.mem.eql(u8, v, "f32")) return .f32;

    std.log.err("Invalid mode: {s}. Expected one of: bf16, f32", .{v});
    return error.InvalidArgs;
}

