const std = @import("std");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const stdx = @import("stdx");

const neuron = @import("platforms/neuron");

const log = std.log.scoped(.@"zml/platforms/neuron/nki_kernel");

pub const TensorSignature = struct {
    dtype: []const u8,
    dims: []const i64,
};

pub const Kernel = struct {
    name: []const u8,
    entrypoint: []const u8,
    source_path: []const u8,
    compiler_target: []const u8,
    inputs: []const TensorSignature,
    outputs: []const TensorSignature,
};

const CompilerRequest = struct {
    entrypoint: []const u8,
    source: []const u8,
    target: []const u8,
    tool_bin_dir: []const u8,
    neuronx_cc_args: []const []const u8,
    inputs: []const TensorSignature,
    outputs: []const TensorSignature,
};

const CompilerResponse = struct {
    backend_config_b64: []const u8,
};

pub fn compilerTargetFromInstance() neuron.CompilerTarget {
    const instance = neuron.instance() catch |err| {
        std.debug.panic("failed to query Neuron instance for NKI compiler target: {}", .{err});
    };
    return instance.compilerTarget();
}

pub fn compileNkiKernel(allocator: std.mem.Allocator, io: std.Io, kernel: Kernel) ![]const u8 {
    const compiler_flags = std.mem.span(std.c.getenv("NEURON_CC_FLAGS") orelse return error.MissingNeuronCompilerFlags);
    var compiler_args: std.ArrayList([]const u8) = .empty;
    defer compiler_args.deinit(allocator);
    var compiler_flags_iter = std.mem.tokenizeScalar(u8, compiler_flags, ' ');
    while (compiler_flags_iter.next()) |arg| {
        try compiler_args.append(allocator, arg);
    }

    const tmp_root_path = if (std.c.getenv("TMPDIR")) |tmpdir| std.mem.span(tmpdir) else "/tmp";
    var tmp_root = try std.Io.Dir.openDir(.cwd(), io, tmp_root_path, .{});
    defer tmp_root.close(io);

    var dir_name_buf: [128]u8 = undefined;
    const dir_name = try std.fmt.bufPrint(&dir_name_buf, "zml-nki-{d}", .{std.Io.Timestamp.now(io, .real).nanoseconds});

    var tmp_dir = try tmp_root.createDirPathOpen(io, dir_name, .{ .permissions = .fromMode(0o700) });
    defer tmp_dir.close(io);

    var cwd_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const cwd = try tmp_dir.realPath(io, &cwd_buf);

    const request_file = try tmp_dir.createFile(io, "kernel.request.json", .{ .read = true });
    defer request_file.close(io);

    const result_file = try tmp_dir.createFile(io, "kernel.result.json", .{ .read = true });
    defer result_file.close(io);

    var request_file_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const request_file_path = try request_file.realPath(io, &request_file_buf);

    var result_file_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const result_file_path = try result_file.realPath(io, &result_file_buf);

    const r = try bazel.runfiles(bazel_builtin.current_repository);

    var sandbox_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libpjrt_neuron/sandbox", &sandbox_path_buf) orelse return error.FileNotFound;

    var source_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const source_path = try r.rlocation(kernel.source_path, &source_path_buf) orelse return error.FileNotFound;
    const source = try std.Io.Dir.cwd().readFileAlloc(io, source_path, allocator, .unlimited);
    defer allocator.free(source);

    var nki_cc_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const nki_cc_path = try stdx.Io.Dir.path.bufJoin(&nki_cc_buf, &.{ sandbox_path, "bin", "nki-cc" });

    const bin_dirpath = std.fs.path.dirname(nki_cc_path) orelse return error.InvalidSandboxPath;

    const request: CompilerRequest = .{
        .entrypoint = kernel.entrypoint,
        .source = source,
        .target = kernel.compiler_target,
        .tool_bin_dir = bin_dirpath,
        .neuronx_cc_args = compiler_args.items,
        .inputs = kernel.inputs,
        .outputs = kernel.outputs,
    };

    const request_json = try std.fmt.allocPrint(allocator, "{f}", .{std.json.fmt(request, .{ .emit_null_optional_fields = false })});
    defer allocator.free(request_json);

    try request_file.writePositionalAll(io, request_json, 0);

    var spawn_threaded: std.Io.Threaded = .init(std.heap.c_allocator, .{});
    defer spawn_threaded.deinit();

    const spawn_io = spawn_threaded.io();

    var child = try std.process.spawn(spawn_io, .{
        .argv = &.{
            nki_cc_path,
            "--request",
            request_file_buf[0..request_file_path],
            "--result",
            result_file_buf[0..result_file_path],
        },
        .stdin = .ignore,
        .stdout = .inherit,
        .stderr = .inherit,
        .cwd = .{ .path = cwd_buf[0..cwd] },
    });

    const term = try child.wait(spawn_io);
    switch (term) {
        .exited => |code| {
            if (code != 0) {
                log.err("nki-cc exited with code {}", .{code});
                return error.NkiCcFailed;
            }
        },
        .signal => |sig| {
            log.err("nki-cc terminated by signal {}", .{sig});
            return error.NkiCcFailed;
        },
        else => |status| {
            log.err("nki-cc terminated unexpectedly: {}", .{status});
            return error.NkiCcFailed;
        },
    }

    var result_reader = result_file.reader(io, &.{});
    const data = try result_reader.interface.readAlloc(allocator, try result_file.length(io));
    defer allocator.free(data);

    const parsed_response = try std.json.parseFromSlice(CompilerResponse, allocator, data, .{
        .ignore_unknown_fields = true,
    });
    defer parsed_response.deinit();

    const backend_config = try allocator.dupe(u8, parsed_response.value.backend_config_b64);
    errdefer allocator.free(backend_config);

    return backend_config;
}
