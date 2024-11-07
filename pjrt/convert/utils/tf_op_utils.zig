const mvzr = @import("mvzr");
const std = @import("std");

pub const Category = enum {
    unknown,
    tensorflow,
    jax,
    tf_data,
    memcpy_h_to_d,
    memcpy_d_to_h,
    memcpy_d_to_d,
    memcpy_h_to_h,
};

const unknown_op = "";
const dataset_op = "Dataset";
const memcpy_h_to_d_op = "MemcpyHToD";
const memcpy_d_to_h_op = "MemcpyDToH";
const memcpy_d_to_d_op = "MemcpyDToD";
const memcpy_h_to_h_op = "MemcpyHToH";

const iterator = "Iterator";
const name_scope_separator = '/';
const op_name_suffix_separator = '_';

pub fn getMemcpyCategory(tf_op_fullname: []const u8) ?Category {
    if (std.ascii.startsWithIgnoreCase(tf_op_fullname, "MEMCPYHToD")) {
        return .memcpy_h_to_d;
    }
    if (std.ascii.startsWithIgnoreCase(tf_op_fullname, "MEMCPYDToH")) {
        return .memcpy_d_to_h;
    }
    if (std.ascii.startsWithIgnoreCase(tf_op_fullname, "MEMCPYDToD")) {
        return .memcpy_d_to_d;
    } else if (std.ascii.startsWithIgnoreCase(tf_op_fullname, "MEMCPYHToH")) {
        return .memcpy_h_to_h;
    }
    return null;
}

pub fn fullMatch(haystack: []const u8, needle: []const u8) bool {
    if (mvzr.compile(needle)) |regex| {
        if (regex.match(haystack)) |m| {
            return m.start == 0 and m.end == haystack.len;
        }
    }
    return false;
}

// Example inputs: "MyOpName", "MyNamespace>MyOpName"
pub fn isTfOpName(op_name: []const u8) bool {
    const tf_op_name_regex = "[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*";
    return fullMatch(op_name, tf_op_name_regex);
}

pub fn isJaxOpType(op_name: []const u8) bool {
    const jax_op_type_regex = "[a-z_][a-z0-9_]*(\\[.*\\])?";
    return fullMatch(op_name, jax_op_type_regex);
}

/// Returns an op type derived from an op name.
fn deriveOpType(full_op_name: []const u8) []const u8 {
    // Use the op name without name scopes and suffix as an op type. A full op
    // name consists of name scopes, an op type, and optionally a numeric suffix
    // (e.g., model/layer/MatMul_1).
    var name_scopes_and_op_name = std.mem.splitScalar(u8, full_op_name, name_scope_separator);
    var op_name: []const u8 = undefined;
    while (name_scopes_and_op_name.next()) |part| op_name = part;
    var op_type_and_maybe_suffix = std.mem.splitScalar(u8, op_name, op_name_suffix_separator);
    var maybe_suffix: []const u8 = undefined;
    while (op_type_and_maybe_suffix.next()) |part| maybe_suffix = part;
    var op_type = op_name;
    if (std.fmt.parseInt(i64, maybe_suffix, 10)) |_| {
        // NOTE: assuming a numeric suffix is not part of an op type while
        // technically it is allowed.
        op_type = op_type[0 .. op_name.len - maybe_suffix.len - 1];
    } else |_| {}
    return op_type;
}

pub fn parseTfOpCategory(tf_op_fullname: []const u8) Category {
    // For op types below, they all have the format "<op_name>:<op_type>", though
    // op_type could be empty.
    var split = std.mem.splitScalar(u8, tf_op_fullname, ':');
    var size: usize = 0;
    while (split.next()) |_| : (size += 1) {}
    split.reset();

    if (size != 2) {
        // Two possibilities here: GPU memcpy op or invalid op.
        if (getMemcpyCategory(split.first())) |cat| {
            return cat;
        } else return .unknown;
    }
    const parts: [2][]const u8 = [_][]const u8{ split.first(), split.rest() };

    // Check for a Dataset op.
    if (std.mem.eql(u8, parts[0], iterator)) {
        // Dataset Op names (e.g., Iterator::Batch::Map::TFRecord) do not follow the
        // format of TF Op names. But we still want to capture them for
        // input-pipeline analysis.
        return .tf_data;
    }

    // Check for Tensorflow Op.
    if (isTfOpName(parts[0]) and isTfOpName(parts[1])) {
        return .tensorflow;
    }

    // Check for JAX op.
    const op_type: []const u8 = if (parts[1].len == 0) deriveOpType(parts[0]) else parts[1];
    if (isJaxOpType(op_type)) {
        // JAX category introduces op_type with '[]' including unnecessary details
        // to represent a group of ops.
        // We need to striping the brackets and contents inside. Based on our
        // analysis, all the op_type ends with a closing ']' if it contains
        // brakets. It's safe to remove all the characters starting with the
        // position of '['.
        // Example:
        //    "transpose[permutation=(0, 3, 1, 2)]"  =>  "transpose"
        // See: go/xprof-jax-op-type
        return .jax;
    }

    if (parts[1].len == 0) {
        return .tensorflow;
    }

    return .unknown;
}
