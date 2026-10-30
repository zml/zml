const std = @import("std");
const hlo = @import("@xla//xla/service:hlo_proto").proto;

pub const PartitionId = usize;

pub const ComputationPartition = struct {
    instructions: std.AutoHashMapUnmanaged(i64, PartitionId) = .{},
    node_counts: []usize = &.{},

    pub fn deinit(self: *ComputationPartition, allocator: std.mem.Allocator) void {
        self.instructions.deinit(allocator);
        allocator.free(self.node_counts);
        self.* = .{};
    }
};

/// Computation IDs scope instruction-to-partition mappings. Partition IDs are
/// communication depths, local to a computation, and ordered from inputs to root.
pub const GraphPartition = struct {
    computations: std.AutoHashMapUnmanaged(i64, ComputationPartition) = .{},

    pub fn deinit(self: *GraphPartition, allocator: std.mem.Allocator) void {
        var values = self.computations.valueIterator();
        while (values.next()) |value| value.deinit(allocator);
        self.computations.deinit(allocator);
        self.* = .{};
    }
};

pub fn partitionModule(allocator: std.mem.Allocator, module: hlo.HloModuleProto) !GraphPartition {
    var result: GraphPartition = .{};
    errdefer result.deinit(allocator);
    for (module.computations.items) |computation| {
        const entry = try result.computations.getOrPut(allocator, computation.id);
        if (entry.found_existing) return error.DuplicateComputationId;
        entry.value_ptr.* = .{};
        entry.value_ptr.* = try partitionComputation(allocator, computation);
    }
    return result;
}

fn isBoundary(opcode: []const u8) bool {
    // Async collectives cross the boundary at start; done stays in that stage.
    inline for (.{ "all-gather", "all-to-all", "all-gather-start", "all-to-all-start" }) |name| {
        if (std.mem.eql(u8, opcode, name)) return true;
    }
    return false;
}

/// Cut at all-gather/all-to-all: a collective starts a new stage after all
/// dependencies. Parallel collectives share a stage; joins take the latest
/// incoming stage. A reverse pass moves preparation work toward its first user,
/// avoiding a huge initial partition of parameters/constants and their chains.
/// Both passes include control dependencies and run in O(nodes + edges).
pub fn partitionComputation(allocator: std.mem.Allocator, computation: hlo.HloComputationProto) !ComputationPartition {
    var scratch = std.heap.ArenaAllocator.init(allocator);
    defer scratch.deinit();
    const temp = scratch.allocator();
    const instructions = computation.instructions.items;
    var result: ComputationPartition = .{};
    errdefer result.deinit(allocator);
    if (instructions.len == 0) return result;

    var indices: std.AutoHashMapUnmanaged(i64, usize) = .{};
    const users = try temp.alloc(std.ArrayList(usize), instructions.len);
    @memset(users, .empty);
    const remaining = try temp.alloc(usize, instructions.len);
    @memset(remaining, 0);
    const stages = try temp.alloc(PartitionId, instructions.len);
    @memset(stages, 0);
    for (instructions, 0..) |instr, i| {
        const entry = try indices.getOrPut(temp, instr.id);
        if (entry.found_existing) return error.DuplicateInstructionId;
        entry.value_ptr.* = i;
    }
    for (instructions, 0..) |instr, i| {
        for ([_][]const i64{ instr.operand_ids.items, instr.control_predecessor_ids.items }) |dependencies| {
            for (dependencies) |id| {
                const source = indices.get(id) orelse return error.MissingDependency;
                try users[source].append(temp, i);
                remaining[i] += 1;
            }
        }
    }

    // Kahn traversal supports protobufs whose instruction list is not sorted.
    var order: std.ArrayList(usize) = .empty;
    for (remaining, 0..) |count, i| {
        if (count == 0) try order.append(temp, i);
    }
    var cursor: usize = 0;
    var max_stage: usize = 0;
    while (cursor < order.items.len) : (cursor += 1) {
        const i = order.items[cursor];
        stages[i] += @intFromBool(isBoundary(instructions[i].opcode));
        max_stage = @max(max_stage, stages[i]);
        for (users[i].items) |user| {
            stages[user] = @max(stages[user], stages[i]);
            remaining[user] -= 1;
            if (remaining[user] == 0) try order.append(temp, user);
        }
    }
    if (order.items.len != instructions.len) return error.CyclicComputation;

    var reverse = order.items.len;
    while (reverse > 0) {
        reverse -= 1;
        const i = order.items[reverse];
        if (isBoundary(instructions[i].opcode) or users[i].items.len == 0) continue;
        var latest: usize = max_stage;
        for (users[i].items) |user| {
            latest = @min(latest, stages[user] - @intFromBool(isBoundary(instructions[user].opcode)));
        }
        stages[i] = latest;
    }

    result.node_counts = try allocator.alloc(usize, max_stage + 1);
    @memset(result.node_counts, 0);
    for (instructions, stages) |instr, stage| {
        try result.instructions.put(allocator, instr.id, stage);
        // The converter bypasses tuple-element nodes when emitting edges.
        if (!std.mem.eql(u8, instr.opcode, "get-tuple-element")) result.node_counts[stage] += 1;
    }
    return result;
}

fn instruction(id: i64, opcode: []const u8, operands: []const i64) hlo.HloInstructionProto {
    return .{ .id = id, .opcode = opcode, .operand_ids = .{ .items = @constCast(operands), .capacity = operands.len } };
}

fn testComputation(instructions: []hlo.HloInstructionProto) hlo.HloComputationProto {
    return .{ .instructions = .{ .items = instructions, .capacity = instructions.len } };
}

test "parallel collectives, residual joins and late preparation" {
    var nodes = [_]hlo.HloInstructionProto{
        instruction(9, "add", &.{ 5, 8, 1 }),
        instruction(1, "parameter", &.{}),
        instruction(2, "all-gather", &.{1}),
        instruction(3, "all-to-all", &.{1}),
        instruction(4, "add", &.{ 2, 3 }),
        instruction(5, "all-to-all", &.{4}),
        instruction(6, "parameter", &.{}),
        instruction(7, "constant", &.{}),
        instruction(8, "multiply", &.{ 6, 7 }),
    };
    var result = try partitionComputation(std.testing.allocator, testComputation(&nodes));
    defer result.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(usize, &.{ 1, 3, 5 }, result.node_counts);
    for ([_]i64{ 2, 3, 4 }) |id| try std.testing.expectEqual(@as(usize, 1), result.instructions.get(id).?);
    for ([_]i64{ 5, 6, 7, 8, 9 }) |id| try std.testing.expectEqual(@as(usize, 2), result.instructions.get(id).?);
}

test "async collective, control dependencies and hidden tuple elements" {
    var nodes = [_]hlo.HloInstructionProto{
        instruction(1, "parameter", &.{}),
        instruction(2, "all-gather-start", &.{1}),
        instruction(3, "all-gather-done", &.{2}),
        instruction(4, "get-tuple-element", &.{3}),
        instruction(5, "all-to-all", &.{4}),
        instruction(6, "copy", &.{1}),
    };
    nodes[5].control_predecessor_ids = .{ .items = @constCast(&[_]i64{5}), .capacity = 1 };
    var result = try partitionComputation(std.testing.allocator, testComputation(&nodes));
    defer result.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 2 }, result.node_counts);
    try std.testing.expectEqual(@as(usize, 1), result.instructions.get(4).?);
    try std.testing.expectEqual(@as(usize, 2), result.instructions.get(6).?);
}

test "no collective, empty graph and invalid dependencies" {
    var nodes = [_]hlo.HloInstructionProto{ instruction(1, "parameter", &.{}), instruction(2, "all-reduce", &.{1}) };
    var result = try partitionComputation(std.testing.allocator, testComputation(&nodes));
    defer result.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(usize, &.{2}, result.node_counts);
    var empty = try partitionComputation(std.testing.allocator, testComputation(&.{}));
    defer empty.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), empty.node_counts.len);
    nodes[0] = instruction(1, "copy", &.{2});
    try std.testing.expectError(error.CyclicComputation, partitionComputation(std.testing.allocator, testComputation(&nodes)));
    nodes[0] = instruction(1, "copy", &.{99});
    try std.testing.expectError(error.MissingDependency, partitionComputation(std.testing.allocator, testComputation(&nodes)));
    nodes[0] = instruction(2, "parameter", &.{});
    try std.testing.expectError(error.DuplicateInstructionId, partitionComputation(std.testing.allocator, testComputation(&nodes)));
}
