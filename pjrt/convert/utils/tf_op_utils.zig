const std = @import("std");
const c = @import("c");

pub const Category = enum {
    kUnknown,
    kTensorFlow,
    kJax,
    kTfData,
    kMemcpyHToD,
    kMemcpyDToH,
    kMemcpyDToD,
    kMemcpyHToH,
};

pub const TfOp = struct {
    category: Category = .kUnknown,
    name: []const u8,
    type_: []const u8,
};

const kUnknownOp = "";
const kDatasetOp = "Dataset";
const kMemcpyHToDOp = "MemcpyHToD";
const kMemcpyDToHOp = "MemcpyDToH";
const kMemcpyDToDOp = "MemcpyDToD";
const kMemcpyHToHOp = "MemcpyHToH";

const kIterator = "Iterator";
const kSeparator = "::";
const kNameScopeSeparator = '/';
const kOpNameSuffixSeparator = '_';
pub fn getMemcpyOp(tf_op_fullname: []const u8) ?TfOp {
    var tf_op: TfOp = undefined;
    tf_op.name = tf_op_fullname;
    if (std.ascii.startsWithIgnoreCase(tf_op_fullname, "MEMCPYHToD")) {
        tf_op.category = .kMemcpyHToD;
        tf_op.type_ = kMemcpyHToDOp;
        return tf_op;
    }
    if (std.ascii.startsWithIgnoreCase(tf_op_fullname, "MEMCPYDToH")) {
        tf_op.category = .kMemcpyDToH;
        tf_op.type_ = kMemcpyDToHOp;
        return tf_op;
    }
    if (std.ascii.startsWithIgnoreCase(tf_op_fullname, "MEMCPYDToD")) {
        tf_op.category = .kMemcpyDToD;
        tf_op.type_ = kMemcpyDToDOp;
        return tf_op;
    } else if (std.ascii.startsWithIgnoreCase(tf_op_fullname, "MEMCPYHToH")) {
        tf_op.category = .kMemcpyHToH;
        tf_op.type_ = kMemcpyHToHOp;
        return tf_op;
    }
    return null;
}

// Example inputs: "MyOpName", "MyNamespace>MyOpName"
pub fn isTfOpName(op_name: []const u8) bool {
    return c.isTfOpName(op_name.ptr, op_name.len);
}

/// Returns an op type derived from an op name.
fn deriveOpType(full_op_name: []const u8) []const u8 {
    // Use the op name without name scopes and suffix as an op type. A full op
    // name consists of name scopes, an op type, and optionally a numeric suffix
    // (e.g., model/layer/MatMul_1).
    var name_scopes_and_op_name = std.mem.splitScalar(u8, full_op_name, kNameScopeSeparator);
    var op_name: []const u8 = undefined;
    while (name_scopes_and_op_name.next()) |part| op_name = part;
    var op_type_and_maybe_suffix = std.mem.splitScalar(u8, op_name, kOpNameSuffixSeparator);
    var maybe_suffix: []const u8 = undefined;
    while (op_type_and_maybe_suffix.next()) |part| maybe_suffix = part;
    var op_type = op_name;
    if (c.isNumber(maybe_suffix.ptr, maybe_suffix.len)) {
        // NOTE: assuming a numeric suffix is not part of an op type while
        // technically it is allowed.
        op_type = op_type[0 .. op_name.len - maybe_suffix.len - 1];
    }
    return op_type;
}

pub fn parseTfOpFullName(tf_op_fullname: []const u8) TfOp {
    // For op types below, they all have the format "<op_name>:<op_type>", though
    // op_type could be empty.
    var tf_op: TfOp = .{ .category = .kUnknown, .name = tf_op_fullname, .type_ = kUnknownOp };
    var split = std.mem.splitScalar(u8, tf_op_fullname, ':');

    var size: usize = 0;
    while (split.next()) |_| : (size += 1) {}
    split.reset();

    if (size != 2) {
        // Two possibilities here: GPU memcpy op or invalid op.
        if (getMemcpyOp(split.first())) |tfop| return tfop;
        return tf_op;
    }
    const parts: [2][]const u8 = [_][]const u8{ split.first(), split.rest() };

    // Check for a Dataset op.
    if (std.mem.eql(u8, parts[0], kIterator)) {
        // Dataset Op names (e.g., Iterator::Batch::Map::TFRecord) do not follow the
        // format of TF Op names. But we still want to capture them for
        // input-pipeline analysis.
        tf_op.category = .kTfData;
        tf_op.type_ = kDatasetOp;
        return tf_op;
    }

    // Check for Tensorflow Op.
    if (isTfOpName(parts[0]) and isTfOpName(parts[1])) {
        tf_op.category = .kTensorFlow;
        tf_op.name = parts[0];
        tf_op.type_ = parts[1];
        return tf_op;
    }

    // Check for JAX op.
    const op_type: []const u8 = if (parts[1].len == 0) deriveOpType(parts[0]) else parts[1];
    if (c.isJaxOpType(op_type.ptr, op_type.len)) {
        // JAX category introduces op_type with '[]' including unnecessary details
        // to represent a group of ops.
        // We need to striping the brackets and contents inside. Based on our
        // analysis, all the op_type ends with a closing ']' if it contains
        // brakets. It's safe to remove all the characters starting with the
        // position of '['.
        // Example:
        //    "transpose[permutation=(0, 3, 1, 2)]"  =>  "transpose"
        // See: go/xprof-jax-op-type
        tf_op.category = .kJax;
        tf_op.name = parts[0];
        tf_op.type_ = op_type[0 .. std.mem.indexOfScalar(u8, op_type, '[') orelse op_type.len];
        return tf_op;
    }

    if (parts[1].len == 0) {
        tf_op.category = .kTensorFlow;
        tf_op.name = parts[0];
        tf_op.type_ = op_type;
        return tf_op;
    }

    return tf_op;
}
