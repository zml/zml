const std = @import("std");

const c = @import("c");
const stdx = @import("stdx");

const config = @import("custom_call_config.zig");

const log = std.log.scoped(.@"zml/platforms/neuron/nki/kernel_compiler");

// Request/result contract for the Zig -> `nki-cc` boundary. Python owns the
// internal scratch layout behind this interface; Zig owns the outer Neuron
// workspace lifecycle that those artifacts live under.
const CompilerRequest = struct {
    entrypoint: []const u8,
    source: []const u8,
    target: []const u8,
    tool_bin_dir: []const u8,
    inputs: []const config.TensorSignature,
    outputs: []const config.TensorSignature,
};

const CompilerResponse = struct {
    backend_config_b64: []const u8,
    input_names: [][]const u8,
    output_names: [][]const u8,
};

pub const CompiledKernel = struct {
    backend_config_bytes: []const u8,
    input_names: [][]const u8,
    output_names: [][]const u8,
};

fn readFileAlloc(allocator: std.mem.Allocator, io: std.Io, file: std.Io.File) ![]const u8 {
    const stat = try file.stat(io);
    const buf = try allocator.alloc(u8, stat.size);
    errdefer allocator.free(buf);
    _ = try file.readPositionalAll(io, buf, 0);
    return buf;
}

// Serialize the single structured request consumed by `nki-cc`.
fn writeCompilerRequest(
    allocator: std.mem.Allocator,
    io: std.Io,
    file: std.Io.File,
    request: CompilerRequest,
) !void {
    const request_json = try std.fmt.allocPrint(
        allocator,
        "{f}",
        .{std.json.fmt(request, .{ .emit_null_optional_fields = false })},
    );
    defer allocator.free(request_json);
    try file.writePositionalAll(io, request_json, 0);
}

// Parse the single structured response produced by `nki-cc`.
fn readCompilerResponse(
    allocator: std.mem.Allocator,
    io: std.Io,
    file: std.Io.File,
) !CompiledKernel {
    const data = try readFileAlloc(allocator, io, file);
    const response = try std.json.parseFromSliceLeaky(CompilerResponse, allocator, data, .{
        .ignore_unknown_fields = true,
    });
    return .{
        // `AwsNeuronCustomNativeKernel` expects backend_config as base64 text.
        .backend_config_bytes = try allocator.dupe(u8, response.backend_config_b64),
        .input_names = response.input_names,
        .output_names = response.output_names,
    };
}

// Compile one embedded NKI kernel through `nki-cc` and return the payload
// needed to rewrite the corresponding HLO custom-call.
pub fn compileKernel(
    allocator: std.mem.Allocator,
    io: std.Io,
    tmp_dir: std.Io.Dir,
    kernel: config.KernelConfig,
    target: []const u8,
    kernel_index: usize,
) !CompiledKernel {
    if (std.mem.eql(u8, target, "inf1")) return error.UnsupportedNkiTarget;

    const stem = try std.fmt.allocPrint(allocator, "kernel-{d}", .{kernel_index});
    const request_file = try tmp_dir.createFile(io, try std.fmt.allocPrint(allocator, "{s}.request.json", .{stem}), .{
        .read = true,
        .truncate = true,
    });
    const result_file = try tmp_dir.createFile(io, try std.fmt.allocPrint(allocator, "{s}.result.json", .{stem}), .{
        .read = true,
        .truncate = true,
    });

    var cwd_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const cwd = try tmp_dir.realPath(io, &cwd_buf);

    var request_file_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const request_file_path = try request_file.realPath(io, &request_file_buf);

    var result_file_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const result_file_path = try result_file.realPath(io, &result_file_buf);

    var nki_cc_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const nki_cc_path = try stdx.Io.Dir.path.bufJoin(&nki_cc_buf, &.{
        stdx.process.selfSharedObjectDirPath(),
        "..",
        "bin",
        "nki-cc",
    });
    const bin_dirpath = std.fs.path.dirname(nki_cc_path).?;

    try writeCompilerRequest(allocator, io, request_file, .{
        .entrypoint = kernel.entrypoint,
        .source = kernel.source,
        .target = target,
        .tool_bin_dir = bin_dirpath,
        .inputs = kernel.inputs,
        .outputs = kernel.outputs,
    });

    {
        const gil_state = c.PyEval_SaveThread();
        defer c.PyEval_RestoreThread(gil_state);

        var child = try std.process.spawn(io, .{
            .argv = &.{
                nki_cc_path,
                "--request",
                request_file_buf[0..request_file_path],
                "--result",
                result_file_buf[0..result_file_path],
            },
            .stdin = .ignore,
            .stdout = .ignore,
            .stderr = .ignore,
            .cwd = .{ .path = cwd_buf[0..cwd] },
        });
        const term = try child.wait(io);
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
    }

    return try readCompilerResponse(allocator, io, result_file);
}
