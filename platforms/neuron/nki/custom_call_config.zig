const std = @import("std");

// ZML-side textual backend_config format carried by the synthetic
// `zml$neuron$nki` custom-call before the NKI lowering pass rewrites it.
pub const TensorSignature = struct {
    name: ?[]const u8 = null,
    dtype: []const u8,
    dims: []const usize,
};

pub const KernelConfig = struct {
    name: []const u8,
    entrypoint: []const u8,
    source: []const u8,
    inputs: []TensorSignature,
    outputs: []TensorSignature,
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

// Parse the ZML-side textual backend_config attached to a synthetic
// `zml$neuron$nki` custom-call.
pub fn parseKernelConfig(allocator: std.mem.Allocator, backend_config: []const u8) !KernelConfig {
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
