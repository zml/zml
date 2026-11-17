const std = @import("std");

pub const GpuInfo = struct {
    memory_bandwidth: u64,
    pcie_bandwidth: u64,

    pub fn overhead(self: GpuInfo, communication: ?u64, memory: ?u64) ?f64 {
        if (self.memory_bandwidth == 0 or self.pcie_bandwidth == 0) return null;
        const comm = communication orelse return null;
        const reads = memory orelse return null;
        if (reads == 0) return if (comm == 0) null else std.math.inf(f64);
        return (@as(f64, @floatFromInt(comm)) / @as(f64, @floatFromInt(self.pcie_bandwidth))) /
            (@as(f64, @floatFromInt(reads)) / @as(f64, @floatFromInt(self.memory_bandwidth)));
    }
};

pub fn bucket(overhead: f64) []const u8 {
    return if (overhead > 0.5) "severe" else if (overhead >= 0.2) "moderate" else "low";
}

// Read only the two scalar fields in gpu_device_info. A tokenizer keeps quoted
// descriptions, comments, and identically named nested fields out of the result.
pub fn parse(text: []const u8) ?GpuInfo {
    var tokens: Tokens = .{ .text = text };
    var depth: usize = 0;
    var deviceDepth: ?usize = null;
    var memory: ?u64 = null;
    var pcie: ?u64 = null;
    while (tokens.next()) |token| {
        if (std.mem.eql(u8, token, "gpu_device_info") and depth == 0) {
            if (deviceDepth != null) return null;
            var open = tokens.next() orelse return null;
            if (std.mem.eql(u8, open, ":")) open = tokens.next() orelse return null;
            if (!std.mem.eql(u8, open, "{") and !std.mem.eql(u8, open, "<")) return null;
            depth += 1;
            deviceDepth = depth;
        } else if (std.mem.eql(u8, token, "{") or std.mem.eql(u8, token, "<")) {
            depth += 1;
        } else if (std.mem.eql(u8, token, "}") or std.mem.eql(u8, token, ">")) {
            if (depth == 0) return null;
            if (deviceDepth == depth) deviceDepth = null;
            depth -= 1;
        } else if (deviceDepth == depth and (std.mem.eql(u8, token, "memory_bandwidth") or std.mem.eql(u8, token, "pcie_bandwidth"))) {
            const colon = tokens.next() orelse return null;
            if (!std.mem.eql(u8, colon, ":")) return null;
            const number = tokens.next() orelse return null;
            const value = std.fmt.parseInt(u64, number, 10) catch return null;
            if (value == 0) return null;
            const target = if (std.mem.eql(u8, token, "memory_bandwidth")) &memory else &pcie;
            if (target.* != null) return null;
            target.* = value;
        }
    }
    if (depth != 0 or tokens.invalid) return null;
    return .{ .memory_bandwidth = memory orelse return null, .pcie_bandwidth = pcie orelse return null };
}

const Tokens = struct {
    text: []const u8,
    cursor: usize = 0,
    invalid: bool = false,

    fn next(self: *Tokens) ?[]const u8 {
        while (self.cursor < self.text.len) {
            const c = self.text[self.cursor];
            if (std.ascii.isWhitespace(c) or c == ',' or c == ';') {
                self.cursor += 1;
            } else if (c == '#') {
                while (self.cursor < self.text.len and self.text[self.cursor] != '\n') self.cursor += 1;
            } else break;
        }
        if (self.cursor == self.text.len) return null;
        const start = self.cursor;
        const c = self.text[self.cursor];
        self.cursor += 1;
        if (c == '"' or c == '\'') {
            while (self.cursor < self.text.len) {
                const value = self.text[self.cursor];
                self.cursor += 1;
                if (value == c) return self.text[start..self.cursor];
                if (value == '\\') self.cursor += 1;
            }
            self.invalid = true;
            self.cursor = self.text.len;
            return null;
        }
        if (std.mem.indexOfScalar(u8, "{}<>:", c) != null) return self.text[start..self.cursor];
        while (self.cursor < self.text.len) : (self.cursor += 1) {
            const value = self.text[self.cursor];
            if (std.ascii.isWhitespace(value) or std.mem.indexOfScalar(u8, "{}<>:,;#", value) != null) break;
        }
        return self.text[start..self.cursor];
    }
};

test "GPU bandwidth parser scopes fields and tolerates other target metadata" {
    const info = (parse(
        \\gpu_device_info {
        \\ name: "GPU memory_bandwidth: 42 { }"
        \\ nested { memory_bandwidth: 1 }
        \\ memory_bandwidth: 1792128000000 # bytes per second
        \\ pcie_bandwidth: 68719476736
        \\}
        \\platform_name: "CUDA"
    )).?;
    try std.testing.expectEqual(@as(u64, 1792128000000), info.memory_bandwidth);
    try std.testing.expectEqual(@as(u64, 68719476736), info.pcie_bandwidth);
    try std.testing.expectEqual(@as(?GpuInfo, null), parse("gpu_device_info { memory_bandwidth: 1 }"));
    try std.testing.expectEqual(@as(?GpuInfo, null), parse("memory_bandwidth: 1 pcie_bandwidth: 2"));
    try std.testing.expectEqual(@as(?GpuInfo, null), parse("gpu_device_info { memory_bandwidth: 0 pcie_bandwidth: 2 }"));
    try std.testing.expectEqual(@as(?GpuInfo, null), parse("gpu_device_info { memory_bandwidth: -1 pcie_bandwidth: 2 }"));
    try std.testing.expectEqual(@as(?GpuInfo, null), parse("gpu_device_info { memory_bandwidth: 1 memory_bandwidth: 2 }"));
    try std.testing.expectEqual(@as(?GpuInfo, null), parse("gpu_device_info { memory_bandwidth: 1"));
    try std.testing.expectEqual(@as(?GpuInfo, null), parse("gpu_device_info { memory_bandwidth: 100 pcie_bandwidth: 10 } name: \"unterminated"));
}

test "communication overhead thresholds and unavailable estimates" {
    const info: GpuInfo = .{ .memory_bandwidth = 1000, .pcie_bandwidth = 100 };
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), info.overhead(5, 100).?, 1e-12);
    try std.testing.expectEqualStrings("low", bucket(info.overhead(1, 100).?));
    try std.testing.expectEqualStrings("moderate", bucket(0.2));
    try std.testing.expectEqualStrings("moderate", bucket(0.5));
    try std.testing.expectEqualStrings("severe", bucket(0.500001));
    try std.testing.expectEqualStrings("severe", bucket(info.overhead(1, 0).?));
    try std.testing.expectEqual(@as(?f64, null), info.overhead(0, 0));
    try std.testing.expectEqual(@as(?f64, null), info.overhead(null, 10));
    try std.testing.expectEqual(@as(?f64, null), info.overhead(10, null));
}
