const std = @import("std");

const c = @import("c");
const stdx = @import("stdx");

const config = @import("nki_custom_call_config.zig");

const log = std.log.scoped(.@"zml/platforms/neuron/nki_compiler_client");

// Client for the embedded-kernel compiler subprocess. The outer Neuron hook
// owns the scratch directory; this file owns the request/result JSON contract
// used to call the `nki-cc` helper binary.

pub const CompiledKernel = struct {
    backend_config_bytes: []const u8,
    input_names: [][]const u8,
    output_names: [][]const u8,
};

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

// Compile one embedded NKI kernel and return the payload needed to rewrite the
// corresponding HLO custom-call. The request file carries ZML's inline kernel
// source and tensor signatures; the response carries the Neuron backend_config
// plus compiler-visible IO names.
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

    const request: CompilerRequest = .{
        .entrypoint = kernel.entrypoint,
        .source = kernel.source,
        .target = target,
        .tool_bin_dir = bin_dirpath,
        .inputs = kernel.inputs,
        .outputs = kernel.outputs,
    };

    const request_json = try std.fmt.allocPrint(allocator, "{f}", .{std.json.fmt(request, .{ .emit_null_optional_fields = false })});
    defer allocator.free(request_json);
    try request_file.writePositionalAll(io, request_json, 0);

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
            .stdout = if (std.c.getenv("NEURON_RT_LOG_LEVEL")) |_| .inherit else .ignore,
            .stderr = if (std.c.getenv("NEURON_RT_LOG_LEVEL")) |_| .inherit else .ignore,
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

    const data = try readFileAlloc(allocator, io, result_file);

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

fn readFileAlloc(allocator: std.mem.Allocator, io: std.Io, file: std.Io.File) ![]const u8 {
    const stat = try file.stat(io);
    const buf = try allocator.alloc(u8, stat.size);
    errdefer allocator.free(buf);
    _ = try file.readPositionalAll(io, buf, 0);
    return buf;
}
