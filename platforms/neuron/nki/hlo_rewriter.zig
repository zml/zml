const std = @import("std");

const c = @import("c");
const stdx = @import("stdx");
const upb = @import("upb");

const log = std.log.scoped(.@"zml/platforms/neuron/nki/hlo_rewriter");

pub const zml_neuron_nki_target = "zml$neuron$nki";
const aws_neuron_custom_native_kernel_target = "AwsNeuronCustomNativeKernel";
const neff_input_name_attr = "neff_input_name";
const neff_input_names_attr = "neff_input_names";
const neff_output_names_attr = "neff_output_names";

// This file owns the embedded-kernel lowering boundary, not the whole-program
// Neuron compile flow.
//
// Input contract:
// - ZML emits synthetic `zml$neuron$nki` custom-calls in StableHLO.
// - Their backend_config carries the inline Python source, entrypoint, and
//   tensor signatures encoded in the ZML-specific text format parsed below.
//
// Output contract:
// - those synthetic calls are rewritten into `AwsNeuronCustomNativeKernel`
//   custom-calls that `neuronx-cc` understands
// - the rewritten backend_config payload is produced by `nki-cc` through a
//   single request/result JSON contract
// - frontend attributes are updated so outer parameter/root names match the
//   compiled NKI kernel IO names

const TensorSignature = struct {
    name: ?[]const u8 = null,
    dtype: []const u8,
    dims: []const usize,
};

const NkiKernelConfig = struct {
    name: []const u8,
    entrypoint: []const u8,
    source: []const u8,
    inputs: []TensorSignature,
    outputs: []TensorSignature,
};

const NkiCompiledKernel = struct {
    backend_config_bytes: []const u8,
    input_names: [][]const u8,
    output_names: [][]const u8,
};

const NkiCompilerRequest = struct {
    entrypoint: []const u8,
    source: []const u8,
    target: []const u8,
    tool_bin_dir: []const u8,
    inputs: []const TensorSignature,
    outputs: []const TensorSignature,
};

const NkiCompilerResponse = struct {
    backend_config_b64: []const u8,
    input_names: [][]const u8,
    output_names: [][]const u8,
};

fn base64DecodeAlloc(allocator: std.mem.Allocator, bytes: []const u8) ![]const u8 {
    const decoder = std.base64.standard.Decoder;
    const decoded = try allocator.alloc(u8, try decoder.calcSizeForSlice(bytes));
    try decoder.decode(decoded, bytes);
    return decoded;
}

fn parseTensorSignature(allocator: std.mem.Allocator, value: []const u8) !TensorSignature {
    const sep = std.mem.indexOfScalar(u8, value, '|') orelse return error.InvalidNkiBackendConfig;
    const dims_text = value[sep + 1 ..];

    var dims: std.ArrayList(usize) = .empty;
    if (dims_text.len != 0) {
        var dims_it = std.mem.tokenizeScalar(u8, dims_text, ',');
        while (dims_it.next()) |dim_text| {
            try dims.append(allocator, try std.fmt.parseInt(usize, dim_text, 10));
        }
    }

    return .{
        .dtype = try allocator.dupe(u8, value[0..sep]),
        .dims = try dims.toOwnedSlice(allocator),
    };
}

fn parseNkiKernelConfig(allocator: std.mem.Allocator, backend_config: []const u8) !NkiKernelConfig {
    // Parse the ZML-side textual backend_config attached to the synthetic
    // `zml$neuron$nki` custom-call. This is intentionally separate from the
    // request/result JSON contract used by `nki-cc`.
    var inputs: std.ArrayList(TensorSignature) = .empty;
    var outputs: std.ArrayList(TensorSignature) = .empty;

    var name: ?[]const u8 = null;
    var entrypoint: ?[]const u8 = null;
    var source: ?[]const u8 = null;

    var lines = std.mem.tokenizeScalar(u8, backend_config, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;

        const eq = std.mem.indexOfScalar(u8, line, '=') orelse return error.InvalidNkiBackendConfig;
        const key = line[0..eq];
        const value = line[eq + 1 ..];

        if (std.mem.eql(u8, key, "name")) {
            name = try base64DecodeAlloc(allocator, value);
        } else if (std.mem.eql(u8, key, "entrypoint")) {
            entrypoint = try base64DecodeAlloc(allocator, value);
        } else if (std.mem.eql(u8, key, "source")) {
            source = try base64DecodeAlloc(allocator, value);
        } else if (std.mem.startsWith(u8, key, "input")) {
            try inputs.append(allocator, try parseTensorSignature(allocator, value));
        } else if (std.mem.startsWith(u8, key, "output")) {
            try outputs.append(allocator, try parseTensorSignature(allocator, value));
        }
    }

    return .{
        .name = name.?,
        .entrypoint = entrypoint.?,
        .source = source.?,
        .inputs = try inputs.toOwnedSlice(allocator),
        .outputs = try outputs.toOwnedSlice(allocator),
    };
}

fn readFileAlloc(allocator: std.mem.Allocator, io: std.Io, file: std.Io.File) ![]const u8 {
    const stat = try file.stat(io);
    const buf = try allocator.alloc(u8, stat.size);
    errdefer allocator.free(buf);
    _ = try file.readPositionalAll(io, buf, 0);
    return buf;
}

fn writeNkiCompilerRequest(
    allocator: std.mem.Allocator,
    io: std.Io,
    file: std.Io.File,
    request: NkiCompilerRequest,
) !void {
    // The Zig/Python boundary is intentionally a single structured request
    // file. Python owns all compiler scratch layout behind that boundary.
    const request_json = try std.fmt.allocPrint(
        allocator,
        "{f}",
        .{std.json.fmt(request, .{ .emit_null_optional_fields = false })},
    );
    defer allocator.free(request_json);
    try file.writePositionalAll(io, request_json, 0);
}

fn rewriteInstructionAsCustomNativeKernel(
    instruction: *c.xla_HloInstructionProto,
    backend_config_bytes: []const u8,
) void {
    c.xla_HloInstructionProto_set_opcode(instruction, upb.stringView("custom-call"));
    c.xla_HloInstructionProto_set_custom_call_target(instruction, upb.stringView(aws_neuron_custom_native_kernel_target));
    c.xla_HloInstructionProto_set_custom_call_api_version(instruction, c.xla_API_VERSION_UNSPECIFIED);
    c.xla_HloInstructionProto_set_backend_config(instruction, upb.stringView(backend_config_bytes));
}

fn findInstructionById(
    instructions: []const [*c]c.xla_HloInstructionProto,
    id: i64,
) ?*c.xla_HloInstructionProto {
    for (instructions) |instruction| {
        if (c.xla_HloInstructionProto_id(instruction) == id) return instruction;
    }
    return null;
}

fn deleteInstructionFrontendAttribute(
    upb_arena: *c.upb_Arena,
    instruction: *c.xla_HloInstructionProto,
    key: []const u8,
) void {
    const attrs = c.xla_HloInstructionProto_mutable_frontend_attributes(instruction, upb_arena);
    _ = c.xla_FrontendAttributes_map_delete(attrs, upb.stringView(key));
}

fn setFrontendAttribute(
    upb_arena: *c.upb_Arena,
    attrs: *c.xla_FrontendAttributes,
    key: []const u8,
    value: []const u8,
) void {
    const map = c._xla_FrontendAttributes_map_mutable_upb_map(attrs, upb_arena);
    _ = c.upb_Map_Set(
        map,
        .{ .str_val = upb.stringView(key) },
        .{ .str_val = upb.stringView(value) },
        upb_arena,
    );
}

fn setInstructionFrontendAttribute(
    upb_arena: *c.upb_Arena,
    instruction: *c.xla_HloInstructionProto,
    key: []const u8,
    value: []const u8,
) void {
    const attrs = c.xla_HloInstructionProto_mutable_frontend_attributes(instruction, upb_arena);
    setFrontendAttribute(upb_arena, attrs, key, value);
}

fn setOuterCustomNativeKernelIoAttributes(
    allocator: std.mem.Allocator,
    upb_arena: *c.upb_Arena,
    comp: *c.xla_HloComputationProto,
    instructions: []const [*c]c.xla_HloInstructionProto,
    instruction: *c.xla_HloInstructionProto,
    input_names: []const []const u8,
    output_names: []const []const u8,
) void {
    const root_instruction = findInstructionById(instructions, c.xla_HloComputationProto_root_id(comp)) orelse return;
    const root_opcode = upb.slice(c.xla_HloInstructionProto_opcode(root_instruction)) orelse return;
    if (!std.mem.eql(u8, root_opcode, "tuple")) return;

    var root_operand_count: usize = 0;
    const root_operand_ids = c.xla_HloInstructionProto_operand_ids(root_instruction, &root_operand_count)[0..root_operand_count];
    if (root_operand_ids.len != 1 or root_operand_ids[0] != c.xla_HloInstructionProto_id(instruction)) return;

    var operand_count: usize = 0;
    const operand_ids = c.xla_HloInstructionProto_operand_ids(instruction, &operand_count)[0..operand_count];
    for (operand_ids, 0..) |operand_id, i| {
        if (i >= input_names.len) break;
        const operand = findInstructionById(instructions, operand_id) orelse continue;
        const opcode = upb.slice(c.xla_HloInstructionProto_opcode(operand)) orelse continue;
        if (!std.mem.eql(u8, opcode, "parameter")) continue;
        deleteInstructionFrontendAttribute(upb_arena, operand, neff_input_names_attr);
        setInstructionFrontendAttribute(upb_arena, operand, neff_input_name_attr, input_names[i]);
    }

    if (output_names.len == 0) return;

    const output_names_value = if (output_names.len == 1)
        output_names[0]
    else
        std.mem.join(allocator, ",", output_names) catch return;
    setInstructionFrontendAttribute(upb_arena, root_instruction, neff_output_names_attr, output_names_value);
}

fn readNkiCompilerResponse(
    allocator: std.mem.Allocator,
    io: std.Io,
    file: std.Io.File,
) !NkiCompiledKernel {
    // `backend_config_b64` stays base64 text because `AwsNeuronCustomNativeKernel`
    // expects that exact payload format in the rewritten HLO backend_config.
    const data = try readFileAlloc(allocator, io, file);
    const response = try std.json.parseFromSliceLeaky(NkiCompilerResponse, allocator, data, .{
        .ignore_unknown_fields = true,
    });
    return .{
        .backend_config_bytes = try allocator.dupe(u8, response.backend_config_b64),
        .input_names = response.input_names,
        .output_names = response.output_names,
    };
}

fn compileNkiKernel(
    allocator: std.mem.Allocator,
    io: std.Io,
    tmp_dir: std.Io.Dir,
    config: NkiKernelConfig,
    target: []const u8,
    kernel_index: usize,
) !NkiCompiledKernel {
    if (std.mem.eql(u8, target, "inf1")) return error.UnsupportedNkiTarget;

    // Keep the outer tmp_dir ownership in Zig so the whole Neuron compile
    // workspace, including NKI scratch artifacts referenced by the compiled
    // backend_config, shares one lifecycle.
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

    try writeNkiCompilerRequest(allocator, io, request_file, .{
        .entrypoint = config.entrypoint,
        .source = config.source,
        .target = target,
        .tool_bin_dir = bin_dirpath,
        .inputs = config.inputs,
        .outputs = config.outputs,
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
            .stdout = .inherit,
            .stderr = .inherit,
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

    return try readNkiCompilerResponse(allocator, io, result_file);
}

pub fn rewriteCustomCalls(
    allocator: std.mem.Allocator,
    io: std.Io,
    tmp_dir: std.Io.Dir,
    hlo_code: []const u8,
    target: []const u8,
) !?[]const u8 {
    // This pass is intentionally narrow: only rewrite synthetic ZML NKI
    // custom-calls. If none are present, the caller should compile the original
    // StableHLO module unchanged.
    var upb_alloc: upb.Allocator = .init(allocator);
    const upb_arena = c.upb_Arena_Init(null, 0, upb_alloc.inner());

    const hlo_module = try upb.parse(c.xla_HloModuleProto, upb_arena, hlo_code);

    var rewritten: usize = 0;
    var kernel_index: usize = 0;

    var computations_len: usize = undefined;
    const computations = c.xla_HloModuleProto_mutable_computations(hlo_module, &computations_len)[0..computations_len];
    for (computations) |comp| {
        var instructions_len: usize = undefined;
        const instructions = c.xla_HloComputationProto_mutable_instructions(comp, &instructions_len)[0..instructions_len];
        for (instructions) |instruction| {
            const opcode = upb.slice(c.xla_HloInstructionProto_opcode(instruction)) orelse continue;
            if (!std.mem.eql(u8, opcode, "custom-call")) continue;

            const call_target = upb.slice(c.xla_HloInstructionProto_custom_call_target(instruction)) orelse continue;
            if (!std.mem.eql(u8, call_target, zml_neuron_nki_target)) continue;

            const backend_config = upb.slice(c.xla_HloInstructionProto_backend_config(instruction)) orelse return error.InvalidNkiBackendConfig;
            const kernel_config = try parseNkiKernelConfig(allocator, backend_config);
            log.info("Rewriting custom-call {s} as AwsNeuronCustomNativeKernel", .{kernel_config.name});
            const compiled_kernel = try compileNkiKernel(allocator, io, tmp_dir, kernel_config, target, kernel_index);
            rewriteInstructionAsCustomNativeKernel(instruction, compiled_kernel.backend_config_bytes);
            setOuterCustomNativeKernelIoAttributes(
                allocator,
                upb_arena,
                comp,
                instructions,
                instruction,
                compiled_kernel.input_names,
                compiled_kernel.output_names,
            );

            rewritten += 1;
            kernel_index += 1;
        }
    }

    if (rewritten == 0) return null;
    log.info("Rewrote {d} ZML NKI custom-call(s) for target {s}", .{ rewritten, target });
    return try upb.serialize(hlo_module, upb_arena);
}
