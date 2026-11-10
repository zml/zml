const std = @import("std");
const zml = @import("zml");

test {
    std.testing.refAllDecls(@This());
}

// --- Loading ---

// Logical runfiles names. Repositories, revisions and checksums live in MODULE.bazel.
pub const gradium_phonon_31m_en = "phonon_31m_en/model.safetensors";
pub const gradium_phonon_31m_en_tokenizer = "phonon_31m_en/tokenizer.5.0.0.json";
pub const gradium_phonon_voice = "phonon_voices/voices/Tek4tJXiX6_yvXq7.safetensors";

pub fn assetPath(allocator: std.mem.Allocator, name: []const u8) ![]const u8 {
    const bazel_builtin = @import("bazel_builtin");
    const runfiles = try zml.bazel.runfiles(bazel_builtin.current_repository);
    return try runfiles.rlocationAlloc(allocator, name) orelse error.AssetNotFound;
}

pub fn readTensorSlice(allocator: std.mem.Allocator, io: std.Io, tensor: zml.safetensors.Tensor) !zml.Slice {
    const slice = try zml.Slice.alloc(allocator, tensor.shape);
    errdefer slice.free(allocator);

    var read_buffer: [8192]u8 = undefined;
    var reader = try tensor.reader(io, &read_buffer, .{});
    defer reader.deinit();

    try reader.interface.readSliceAll(slice.data());

    return slice;
}

pub const generated_latents_path = "phonon.normalized.safetensors";

// --- Testing ---

/// Similar in spirit to `zml.testing.testLayer`.
pub fn test_module(
    comptime Module: type,
    model_path: []const u8,
    activations_path: []const u8,
    module_prefix: []const u8,
) !void {
    return test_module_inputs(
        Module,
        model_path,
        activations_path,
        module_prefix,
        .{"in"},
        if (@hasDecl(Module, "forward")) .forward else .forward_uncached,
    );
}

pub fn test_module_inputs(
    comptime Module: type,
    model_path: []const u8,
    activations_path: []const u8,
    module_prefix: []const u8,
    comptime input_names: anytype,
    comptime forward_method: std.meta.DeclEnum(Module),
) !void {
    return test_module_outputs(Module, model_path, activations_path, module_prefix, input_names, forward_method, .{"out"});
}

/// Compare either one tensor output or named fields of a tensor-output struct.
pub fn test_module_outputs(
    comptime Module: type,
    model_path: []const u8,
    activations_path: []const u8,
    module_prefix: []const u8,
    comptime input_names: anytype,
    comptime forward_method: std.meta.DeclEnum(Module),
    comptime output_names: anytype,
) !void {
    const allocator = std.testing.allocator;
    const io_base = std.testing.io;

    var arena_impl: std.heap.ArenaAllocator = .init(allocator);
    defer arena_impl.deinit();
    const arena = arena_impl.allocator();

    const io = io_base;
    const local_model_path = try assetPath(allocator, model_path);
    defer allocator.free(local_model_path);

    var model_registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, local_model_path);
    defer model_registry.deinit();

    const platform = zml.testing.env();

    var activations_registry = try zml.safetensors.TensorRegistry.fromPath(
        allocator,
        io,
        activations_path,
    );
    defer activations_registry.deinit();

    if (false)
        for (model_registry.tensors.values()) |v|
            std.log.info("{f}\n", .{v});

    if (false)
        for (activations_registry.tensors.values()) |v|
            std.log.info("{f}\n", .{v});

    var model_store: zml.io.TensorStore = .fromRegistry(allocator, &model_registry, &.{});
    defer model_store.deinit();

    var activations_store: zml.io.TensorStore = .fromRegistry(allocator, &activations_registry, &.{});
    defer activations_store.deinit();

    const module_view = if (module_prefix.len == 0) model_store.view() else model_store.view().withPrefix(module_prefix);
    const module: Module = .init(module_view);

    var input_tensors: @Tuple(&@as([input_names.len]type, @splat(zml.Tensor))) = undefined;
    var input_buffers: @Tuple(&@as([input_names.len]type, @splat(zml.Buffer))) = undefined;
    var loaded: usize = 0;
    defer {
        inline for (0..input_names.len) |i| {
            if (i < loaded) input_buffers[i].deinit();
        }
    }
    inline for (input_names, 0..) |name, i| {
        const tensor = activations_registry.tensors.get(name).?;
        input_tensors[i] = .fromShape(tensor.shape);
        const slice = try readTensorSlice(allocator, io, tensor);
        defer slice.free(allocator);
        input_buffers[i] = try .fromSlice(io, platform, slice);
        loaded += 1;
    }

    var exe = blk: {
        std.log.info("Compiling model...\n", .{});
        const start: std.Io.Timestamp = .now(io, .awake);
        defer std.log.info("✅ Compiled model [{f}]\n", .{start.untilNow(io, .awake)});
        break :blk try platform.compile(allocator, io, module, forward_method, input_tensors, .{});
    };
    defer exe.deinit();

    var module_weights = blk: {
        std.log.info("Transfering weights...\n", .{});
        const start: std.Io.Timestamp = .now(io, .awake);
        defer std.log.info("✅ Transferred weights [{f}]\n", .{start.untilNow(io, .awake)});
        break :blk try module.load(arena, io, platform, &model_store);
    };
    defer Module.unloadBuffers(&module_weights);

    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    args.set(.{module_weights} ++ input_buffers);
    exe.call(io, args, &results, .{});
    const Output = @typeInfo(@TypeOf(@field(Module, @tagName(forward_method)))).@"fn".return_type.?;
    var result = results.get(zml.Bufferized(Output));
    defer zml.Buffer.deinitAll(Output, &result);

    const tolerance: zml.testing.CompareOpts = .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1.0,
    };
    inline for (output_names) |name| {
        const output_safetensor = activations_registry.tensors.get(name) orelse return error.OutputNotFound;
        const output_slice = try readTensorSlice(allocator, io, output_safetensor);
        defer output_slice.free(allocator);
        const buffer = if (Output == zml.Tensor) result else @field(result, name);
        zml.testing.expectClose(io, output_slice, buffer, tolerance) catch |err| {
            std.log.err("{s}.{s} does not match the Python reference", .{ @typeName(Module), name });
            return err;
        };
        std.log.info("{s}.{s} matches the Python reference ({d} values)", .{
            @typeName(Module), name, output_safetensor.shape.count(),
        });
    }
}
