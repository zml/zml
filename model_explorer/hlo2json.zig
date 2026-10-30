const builtin = @import("builtin");
const std = @import("std");
const hlo = @import("@xla//xla/service:hlo_proto").proto;
const log = std.log.scoped(.hlo2json);

const dump_context = @import("dump_context.zig");
const json_writer = @import("json_writer.zig");
const gpu_info = @import("gpu_info.zig");
pub const GpuInfo = gpu_info.GpuInfo;
const graph_partitioner = @import("graph_partitioner.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    if (args.len < 2 or args.len > 3) {
        std.debug.print("Usage: hlo2json <input.hlo.pb> [output.json]\n", .{});
        return error.InvalidArguments;
    }
    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |path| {
        var dir = try std.Io.Dir.openDirAbsolute(init.io, path, .{});
        defer dir.close(init.io);
        try std.process.setCurrentDir(init.io, dir);
    }
    const output_path = if (args.len == 3) args[2] else try std.fmt.allocPrint(allocator, "{s}.json", .{args[1]});
    // Decode before opening the output, so invalid input cannot truncate it.
    var output: std.Io.Writer.Allocating = .init(allocator);
    try convertFile(allocator, init.io, args[1], &output.writer);
    const file = try std.Io.Dir.cwd().createFile(init.io, output_path, .{});
    defer file.close(init.io);
    try file.writeStreamingAll(init.io, output.written());
}

pub fn convertFile(allocator: std.mem.Allocator, io: std.Io, input_hlo_path: []const u8, writer: *std.Io.Writer) !void {
    const input_file = try std.Io.Dir.cwd().openFile(io, input_hlo_path, .{});
    defer input_file.close(io);
    var buffer: [8192]u8 = undefined;
    var reader = input_file.reader(io, &buffer);
    var proto_arena = std.heap.ArenaAllocator.init(allocator);
    defer proto_arena.deinit();
    const proto = try hlo.HloProto.decode(&reader.interface, proto_arena.allocator());
    var converter: Converter = .{ .allocator = proto_arena.allocator() };
    const suffix = ".after_spmd_partitioner.hlo.pb";
    if (std.mem.endsWith(u8, input_hlo_path, suffix)) {
        const sibling = try std.fmt.allocPrint(proto_arena.allocator(), "{s}.before_optimizations.hlo.pb", .{input_hlo_path[0 .. input_hlo_path.len - suffix.len]});
        if (std.Io.Dir.cwd().openFile(io, sibling, .{})) |file| {
            defer file.close(io);
            var ref_buffer: [8192]u8 = undefined;
            var ref_reader = file.reader(io, &ref_buffer);
            const reference = try hlo.HloProto.decode(&ref_reader.interface, proto_arena.allocator());
            if (reference.hlo_module) |module| {
                if (proto.hlo_module != null and std.mem.eql(u8, module.name, proto.hlo_module.?.name)) {
                    for (module.computations.items) |*comp| {
                        if (comp.id == module.entry_computation_id) converter.reference = comp;
                    }
                }
            }
        } else |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        }
    }
    if (proto.hlo_module) |module| {
        converter.context.load(proto_arena.allocator(), io, input_hlo_path, module.name) catch |err| {
            log.warn("Could not load optional dump context: {s}", .{@errorName(err)});
        };
    }
    const gpu = loadGpuInfo(proto_arena.allocator(), io, input_hlo_path) catch |err| blk: {
        log.warn("Could not load optional GPU target config: {s}", .{@errorName(err)});
        break :blk null;
    };
    try converter.emitModule(writer, proto, gpu);
}

threadlocal var py_writer: std.Io.Writer.Allocating = .init(std.heap.page_allocator);

pub const Slice = extern struct { ptr: u64 = 0, len: u64 = 0 };

pub export fn convertHloToJson(input_hlo_path_ptr: [*]const u8, len: u32) Slice {
    const input_hlo_path = input_hlo_path_ptr[0..len];
    py_writer.clearRetainingCapacity();
    var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{ .environ = .empty });
    defer threaded.deinit();
    convertFile(std.heap.page_allocator, threaded.io(), input_hlo_path, &py_writer.writer) catch |err| {
        log.err("Failed to read hlo file: {s} ({})", .{ input_hlo_path, err });
        return .{};
    };
    return .{ .ptr = @intFromPtr(py_writer.written().ptr), .len = py_writer.written().len };
}

const State = enum {
    header,
    stylesheet,
    nodes,
    edges,
};

pub const Converter = struct {
    line_count: u32 = 0,
    state: State = .header,

    allocator: std.mem.Allocator,
    // Map instruction id to its instruction.
    instructions: std.AutoArrayHashMapUnmanaged(i64, *const hlo.HloInstructionProto) = .{},
    // Map instruction id to its buffer when it has one.
    physical_buffer: std.AutoArrayHashMapUnmanaged(i64, Assigned) = .{},
    // Map computation id to its first caller instruction. This allow to have extra context about the computation.
    computation_caller: std.AutoArrayHashMapUnmanaged(i64, *const hlo.HloInstructionProto) = .{},
    computations: std.AutoArrayHashMapUnmanaged(i64, *const hlo.HloComputationProto) = .{},
    reduce_name: std.AutoArrayHashMapUnmanaged(i64, *const hlo.HloInstructionProto) = .{},

    context: dump_context.Context = .{},
    reference: ?*const hlo.HloComputationProto = null,
    entry_id: i64 = -1,
    device_count: ?u64 = null,
    stack_frame_index: ?hlo.StackFrameIndexProto = null,

    const Assigned = struct { physical: i64, start: i64, len: i64 };

    pub fn emitModule(self: *Converter, writer: anytype, hlo_proto: hlo.HloProto, gpu: ?GpuInfo) !void {
        const module = hlo_proto.hlo_module orelse return error.NoModule;
        self.entry_id = module.entry_computation_id;
        if (module.device_assignment) |assignment| {
            if (assignment.replica_count > 0 and assignment.computation_count > 0) {
                self.device_count = std.math.mul(u64, @intCast(assignment.replica_count), @intCast(assignment.computation_count)) catch null;
            }
        }
        self.stack_frame_index = module.stack_frame_index;

        if (hlo_proto.buffer_assignment) |buffer_assignement| {
            // Logical buffer id to instruction id
            var logical_to_instr: std.AutoArrayHashMapUnmanaged(i64, i64) = .{};
            try logical_to_instr.ensureUnusedCapacity(self.allocator, buffer_assignement.logical_buffers.items.len);
            for (buffer_assignement.logical_buffers.items) |logical| {
                logical_to_instr.putAssumeCapacityNoClobber(logical.id, logical.defined_at.?.instruction_id);
            }

            try self.physical_buffer.ensureTotalCapacity(self.allocator, 1024);
            for (buffer_assignement.buffer_allocations.items) |alloc| {
                // If a buffer is not reused it means that its a readonly buffer eg an input buffer.
                // No point in displaying those.
                if (alloc.assigned.items.len <= 1) continue;
                for (alloc.assigned.items) |assigned| {
                    const instr = logical_to_instr.get(assigned.logical_buffer_id) orelse continue;
                    try self.physical_buffer.put(self.allocator, instr, .{ .physical = alloc.index, .start = assigned.offset, .len = assigned.size });
                }
            }
        }

        // TODO: Figure out how to render this
        // log.warn("module.input_output_alias: {?}", .{module.input_output_alias});
        // log.warn("module.spmd_output_sharding: {?}", .{module.spmd_output_sharding});

        // Improve names of each computation by using the name of the calling node to name the callee.
        // Because otherwise names are just "fusion.xx"
        try self.computations.ensureUnusedCapacity(self.allocator, hlo_proto.hlo_module.?.computations.items.len);
        for (hlo_proto.hlo_module.?.computations.items) |*computation| {
            self.computations.putAssumeCapacityNoClobber(computation.id, computation);
            try self.instructions.ensureUnusedCapacity(self.allocator, computation.instructions.items.len);
            for (computation.instructions.items) |*instr| {
                self.instructions.putAssumeCapacityNoClobber(instr.id, instr);
                if (std.mem.eql(u8, instr.opcode, "fusion") and instr.called_computation_ids.items.len > 0) {
                    const callee = instr.called_computation_ids.items[0];
                    _ = try self.computation_caller.getOrPutValue(self.allocator, callee, instr);
                }
            }
        }

        var partitions = try graph_partitioner.partitionModule(self.allocator, module);
        defer partitions.deinit(self.allocator);

        const W = @TypeOf(writer);
        var top_level = try json_writer.dict(writer);

        try top_level.write("label", module.name);
        var graphs = try top_level.openList("graphs");

        for (module.computations.items) |computation| {
            var graph = try graphs.openDict();
            try self.emitComputation(W, &graph, computation, partitions.computations.getPtr(computation.id).?, gpu);
            try graph.close();
        }
        try graphs.close();
        try top_level.close();
    }

    pub fn emitComputation(self: *const Converter, Writer: type, graph: *json_writer.Dict(Writer), computation: hlo.HloComputationProto, partition: *const graph_partitioner.ComputationPartition, gpu: ?GpuInfo) !void {
        const comp_caller = self.computation_caller.get(computation.id);
        // The name of the caller node is better cause computation name is typically just "fusion".
        try graph.write("id", if (comp_caller) |caller| caller.name else computation.name);

        const metrics = try self.partitionMetrics(computation, partition);
        defer self.allocator.free(metrics);
        const namespaces = try self.allocator.alloc([]const u8, metrics.len);
        defer self.allocator.free(namespaces);
        for (metrics, 0..) |metric, stage| {
            namespaces[stage] = label: {
                if (gpu) |g| {
                    if (g.overhead(metric.communication, metric.memory)) |ov| {
                        break :label try std.fmt.allocPrint(self.allocator, "stage_{d:0>3} ({s} CMR)", .{ stage, gpu_info.bucket(ov) });
                    }
                }
                break :label try std.fmt.allocPrint(self.allocator, "stage_{d:0>3}", .{stage});
            };
        }
        defer for (namespaces) |namespace| {
            if (namespace.len > 0) self.allocator.free(namespace);
        };
        var groups = try graph.openDict("groupNodeAttributes");
        for (metrics, 0..) |metric, stage| {
            var attrs = try groups.openDict(namespaces[stage]);
            try attrs.write("CMR", try std.fmt.allocPrint(self.allocator, "{f}", .{Cmr{ .metric = metric }}));
            if (gpu) |info| {
                const overhead = info.overhead(metric.communication, metric.memory);
                try attrs.write("communication overhead", if (overhead) |value| gpu_info.bucket(value) else "unavailable");
            }
            try attrs.writeTemplate("communication bytes", "\"{?B:.1}\"", .{metric.communication});
            try attrs.writeTemplate("memory bytes", "\"{?B:.1}\"", .{metric.memory});
            try attrs.writeTemplate("opaque ops", "\"{d}\"", .{metric.opaque_operations});
            try attrs.close();
        }
        try groups.close();

        var nodes = try graph.openList("nodes");
        for (computation.instructions.items) |instr| {
            if (std.mem.eql(u8, instr.opcode, "get-tuple-element")) continue;
            var node = try nodes.openDict();
            // node name + attrs
            try node.writeTemplate("id", "\"{d}\"", .{instr.id});
            try node.write("namespace", namespaces[partition.instructions.get(instr.id).?]);
            try node.write("label", try std.fmt.allocPrint(self.allocator, "{f}\nreads: {?B:.1}", .{ self.instrLabel(instr), self.instructionReads(instr) }));
            {
                var attrs = try node.openList("attrs");
                try self.writeAttr(Writer, &attrs, instr, comp_caller);
                try attrs.writeTemplate("{{\"key\":\"read bytes\",\"value\":\"{?d}\"}}", .{self.instructionReads(instr)});
                try attrs.close();
            }

            {
                var outputs = try node.openList("outputsMetadata");
                const out_shape = instr.shape.?;
                var out_shapes: []const @TypeOf(out_shape) = &.{out_shape};
                if (out_shapes[0].element_type == .TUPLE) {
                    out_shapes = out_shapes[0].tuple_shapes.items;
                }

                for (0.., out_shapes) |o, o_shape| {
                    var output = try outputs.openDict();
                    try output.write("id", try std.fmt.allocPrint(self.allocator, "{d}", .{o}));
                    var attrs = try output.openList("attrs");
                    try attrs.write(.{ .key = "tensor_shape", .value = try std.fmt.allocPrint(self.allocator, "{f}", .{InstrShape{ .shape = &o_shape }}) });
                    if (logicalBytes(o_shape)) |bytes| try attrs.write(.{ .key = "local logical bytes", .value = try std.fmt.allocPrint(self.allocator, "{d}", .{bytes}) });
                    if (computation.id == self.entry_id) {
                        if (self.referenceInstruction(instr)) |ref| {
                            if (ref.shape) |global| {
                                if (global.element_type != .TUPLE and o_shape.element_type == global.element_type and global.dimensions.items.len == o_shape.dimensions.items.len) {
                                    try attrs.write(.{ .key = "pre-SPMD shape", .value = try std.fmt.allocPrint(self.allocator, "{f}", .{InstrShape{ .shape = &global }}) });
                                    try attrs.write(.{ .key = "shape source", .value = try std.fmt.allocPrint(self.allocator, "before_optimizations: {s} ({s})", .{ ref.name, if (std.mem.eql(u8, instr.opcode, "parameter")) "parameter number" else "operation provenance; may describe an earlier value" }) });
                                    if (ref.sharding) |sharding| try attrs.write(.{ .key = "pre-SPMD sharding", .value = try shardingText(self.allocator, sharding) });
                                }
                            }
                        }
                    }
                    try attrs.write(.{ .key = "recorded sharding", .value = if (instr.sharding) |sharding| try shardingText(self.allocator, sharding) else "unknown (not recorded after partitioning)" });
                    try attrs.close();
                    try output.close();
                }
                try outputs.close();
            }

            // node edges
            {
                var edges = try node.openList("incomingEdges");
                for (instr.operand_ids.items, 0..) |input_id, i| {
                    const input_instr = self.instructions.get(input_id) orelse continue;

                    // The source node, and output_id to handle when input node produces several values.
                    var source_id: i64 = input_id;
                    var source_output_id: i64 = 0;
                    if (std.mem.eql(u8, input_instr.opcode, "get-tuple-element")) {
                        source_id = input_instr.operand_ids.items[0];
                        source_output_id = input_instr.tuple_index;
                    }
                    // edge_metadata could be added here if needed.
                    try edges.writeTemplate(
                        \\{{"sourceNodeId":"{d}","sourceNodeOutputId":"{d}", "targetNodeInputId":"{d}"}}
                    , .{ source_id, source_output_id, i });
                }
                try edges.close();
            }

            if (instr.called_computation_ids.items.len > 0) {
                var subpgraphs = try node.openList("subgraphIds");
                for (instr.called_computation_ids.items) |called| {
                    if (self.computation_caller.get(called)) |meta| {
                        try subpgraphs.write(meta.name);
                    } else {
                        const called_comp = self.computations.get(called).?;
                        try subpgraphs.write(called_comp.name);
                    }
                }
                try subpgraphs.close();
            }
            try node.close();
        }
        try nodes.close();
    }

    fn partitionMetrics(self: *const Converter, computation: hlo.HloComputationProto, partition: *const graph_partitioner.ComputationPartition) ![]PartitionMetrics {
        const metrics = try self.allocator.alloc(PartitionMetrics, partition.node_counts.len);
        @memset(metrics, .{});
        for (computation.instructions.items) |instr| {
            const metric = &metrics[partition.instructions.get(instr.id).?];
            if (isMetadataOnly(instr.opcode) or std.mem.endsWith(u8, instr.opcode, "-done") or std.mem.eql(u8, instr.opcode, "async-update")) continue;
            if (self.collectivePayload(instr)) |payload| {
                metric.communication = addEstimate(metric.communication, payload.bytes);
                continue;
            }
            // Opaque kernels retain a full-input estimate. Fusion reads are traced
            // to parameters without charging intermediate values as HBM traffic.
            if (std.mem.eql(u8, instr.opcode, "custom-call") or std.mem.eql(u8, instr.opcode, "async-start") or std.mem.eql(u8, instr.opcode, "call") or std.mem.eql(u8, instr.opcode, "while") or std.mem.eql(u8, instr.opcode, "conditional")) metric.opaque_operations += 1;
            metric.memory = addEstimate(metric.memory, self.instructionReads(instr));
        }
        return metrics;
    }

    fn instructionReads(self: *const Converter, instr: hlo.HloInstructionProto) ?u64 {
        if (isMetadataOnly(instr.opcode) or std.mem.endsWith(u8, instr.opcode, "-done") or std.mem.eql(u8, instr.opcode, "async-update") or self.collectivePayload(instr) != null) return 0;
        if (std.mem.eql(u8, instr.opcode, "fusion") and instr.called_computation_ids.items.len == 1) {
            if (self.computations.get(instr.called_computation_ids.items[0])) |comp| {
                return self.fusionReads(comp.*);
            }
        }
        var total: ?u64 = 0;
        for (instr.operand_ids.items, 0..) |_, i| total = addEstimate(total, self.operandReads(instr, i, instructionBytes(instr)));
        return total;
    }

    // Logical input footprint of a fused kernel. Intermediate tensors are not
    // charged as HBM reads. Fan-out demands are summed, capped at each value's
    // size: overlap/cache-line effects and repeated dynamic indices are unknown.
    fn fusionReads(self: *const Converter, comp: hlo.HloComputationProto) ?u64 {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const allocator = arena.allocator();
        const nodes = comp.instructions.items;
        var indices: std.AutoHashMapUnmanaged(i64, usize) = .{};
        const users = allocator.alloc(usize, nodes.len) catch return null;
        @memset(users, 0);
        const demands = allocator.alloc(?u64, nodes.len) catch return null;
        @memset(demands, 0);
        for (nodes, 0..) |node, i| indices.put(allocator, node.id, i) catch return null;
        for (nodes) |node| for (node.operand_ids.items) |id| {
            const i = indices.get(id) orelse return null;
            users[i] += 1;
        };
        const root = indices.get(comp.root_id) orelse return null;
        demands[root] = instructionBytes(nodes[root]);
        var queue: std.ArrayList(usize) = .empty;
        for (users, 0..) |count, i| if (count == 0) queue.append(allocator, i) catch return null;
        var cursor: usize = 0;
        var total: ?u64 = 0;
        while (cursor < queue.items.len) : (cursor += 1) {
            const i = queue.items[cursor];
            const node = nodes[i];
            if (std.mem.eql(u8, node.opcode, "parameter")) total = addEstimate(total, demands[i]);
            for (node.operand_ids.items, 0..) |id, operand_index| {
                const source = indices.get(id).?;
                const requested = if (demands[i] == 0) @as(?u64, 0) else self.operandReads(node, operand_index, demands[i]);
                demands[source] = capEstimate(addEstimate(demands[source], requested), instructionBytes(nodes[source]));
                users[source] -= 1;
                if (users[source] == 0) queue.append(allocator, source) catch return null;
            }
        }
        return if (cursor == nodes.len) total else null;
    }

    fn operandReads(self: *const Converter, instr: hlo.HloInstructionProto, index: usize, demand: ?u64) ?u64 {
        const operand = self.instructions.get(instr.operand_ids.items[index]) orelse return null;
        const full = instructionBytes(operand.*);
        const op = instr.opcode;
        if (std.mem.eql(u8, op, "slice") or std.mem.eql(u8, op, "dynamic-slice") or std.mem.eql(u8, op, "gather")) {
            // One data element per result element; indices are separate reads.
            return if (index == 0) demand else full;
        }
        if (std.mem.eql(u8, op, "scatter")) {
            const count = instr.operand_ids.items.len;
            if (count < 3 or count % 2 != 1) return null;
            const outputs = (count - 1) / 2;
            if (index == outputs) return full; // scatter indices
            const parameter = if (index < outputs) index else index - 1;
            if (!(self.scatterUsesParameter(instr, parameter) orelse return null)) return 0;
            if (index < outputs) {
                // Optimized scatters update their destination in place; any
                // materializing copy is a separate operation. Only touched old
                // values are read, and overwrite updates do not read them at all.
                const updates = self.instructions.get(instr.operand_ids.items[outputs + 1 + index]) orelse return null;
                return instructionBytes(updates.*);
            }
            return full;
        }
        if (std.mem.eql(u8, op, "bitcast") or std.mem.eql(u8, op, "reshape") or std.mem.eql(u8, op, "transpose")) return capEstimate(demand, full);
        if (isElementwiseRead(op)) {
            const output_bytes = instructionBytes(instr) orelse return null;
            const input_bytes = full orelse return null;
            const requested = demand orelse return null;
            if (output_bytes == 0) return 0;
            const product = std.math.mul(u128, requested, input_bytes) catch return null;
            const scaled = std.math.divCeil(u128, product, output_bytes) catch return null;
            return @intCast(@min(input_bytes, scaled));
        }
        // Broadcasts reuse their input; reduction/contraction and opaque
        // operations conservatively consume their full inputs.
        return full;
    }

    fn scatterUsesParameter(self: *const Converter, instr: hlo.HloInstructionProto, parameter: usize) ?bool {
        if (instr.called_computation_ids.items.len != 1) return null;
        const comp = self.computations.get(instr.called_computation_ids.items[0]) orelse return null;
        var seen: std.AutoHashMapUnmanaged(i64, void) = .{};
        defer seen.deinit(self.allocator);
        var pending: std.ArrayList(i64) = .empty;
        defer pending.deinit(self.allocator);
        pending.append(self.allocator, comp.root_id) catch return null;
        while (pending.pop()) |id| {
            const entry = seen.getOrPut(self.allocator, id) catch return null;
            if (entry.found_existing) continue;
            const node = self.instructions.get(id) orelse return null;
            if (std.mem.eql(u8, node.opcode, "parameter") and node.parameter_number == parameter) return true;
            pending.appendSlice(self.allocator, node.operand_ids.items) catch return null;
        }
        return false;
    }

    fn collectivePayload(self: *const Converter, instr: hlo.HloInstructionProto) ?struct { bytes: ?u64 } {
        if (isCollective(instr.opcode)) return .{ .bytes = if (instr.shape) |shape| payloadBytes(shape) else null };
        if (std.mem.eql(u8, instr.opcode, "async-start")) {
            for (instr.called_computation_ids.items) |id| {
                const comp = self.computations.get(id) orelse continue;
                for (comp.instructions.items) |root| {
                    if (root.id == comp.root_id and isCollective(root.opcode)) return .{ .bytes = if (root.shape) |shape| payloadBytes(shape) else null };
                }
            }
        }
        if (std.mem.endsWith(u8, instr.opcode, "-start") and isCollective(instr.opcode[0 .. instr.opcode.len - "-start".len])) {
            // Dedicated collective starts return (input buffers, output buffers).
            const shape = instr.shape orelse return .{ .bytes = null };
            return .{ .bytes = if (shape.element_type == .TUPLE and shape.tuple_shapes.items.len >= 2) payloadBytes(shape.tuple_shapes.items[1]) else null };
        }
        return null;
    }

    pub fn writeAttr(self: *const Converter, Writer: type, list: *json_writer.List(Writer), instr: hlo.HloInstructionProto, comp_caller: ?*const hlo.HloInstructionProto) !void {
        try list.write(.{ .key = "instruction", .value = instr.name });
        try list.write(.{ .key = "opcode", .value = instr.opcode });
        if (instr.custom_call_target.len > 0) try list.write(.{ .key = "custom call target", .value = instr.custom_call_target });
        if (isCollective(instr.opcode)) {
            var input_bytes: ?u64 = if (instr.operand_ids.items.len > 0) 0 else null;
            for (instr.operand_ids.items) |id| {
                const operand = self.instructions.get(id) orelse {
                    input_bytes = null;
                    break;
                };
                const shape = operand.shape orelse {
                    input_bytes = null;
                    break;
                };
                const bytes = logicalBytes(shape) orelse {
                    input_bytes = null;
                    break;
                };
                input_bytes = std.math.add(u64, input_bytes.?, bytes) catch null;
                if (input_bytes == null) break;
            }
            const output_bytes = if (instr.shape) |shape| logicalBytes(shape) else null;
            try list.write(.{ .key = "bytes/device", .value = try std.fmt.allocPrint(self.allocator, "out: {?B:.1}, in: {?B:.1}", .{ output_bytes, input_bytes }) });
            const connected = collectiveGroupSize(instr);
            const total = self.context.device_count orelse self.device_count orelse collectiveDeviceCount(instr);
            try list.write(.{ .key = "device connected", .value = try std.fmt.allocPrint(self.allocator, "{?d} / {?d}", .{ connected, total }) });
        }

        if (instr.metadata) |meta| {
            if (try self.sourceLocation(meta)) |location| {
                try list.write(.{ .key = "loc", .value = location });
            }
            if (meta.op_name.len != 0) {
                try list.write(.{ .key = "op_name", .value = try std.fmt.allocPrint(self.allocator, "{s}:{s}", .{ meta.op_type, meta.op_name }) });
            }
            if (meta.deduplicated_name.len != 0) {
                try list.write(.{ .key = "deduplicated_name", .value = meta.deduplicated_name });
            }
        }

        if (self.physical_buffer.get(instr.id)) |assigned| {
            // TODO show ratio of buffer usage
            try list.writeTemplate(
                \\{{"key":"buffer","value":"buffer_{d}[{}..{}]"}}
            , .{ assigned.physical, assigned.start, assigned.start + assigned.len });
        }

        if (comp_caller) |caller| {
            // Note: this is cute, but it only adds a static label to the graph, not an hyperlink.
            // I'm using subgraph to represent fusion blocks, but with the current UI it means we can't
            // link between a subgraph input and the caller argument in the main graph.
            if (instr.parameter_number >= caller.operand_ids.items.len) return;
            if (std.mem.eql(u8, instr.opcode, "parameter")) {
                const arg = caller.operand_ids.items[@intCast(instr.parameter_number)];
                if (self.instructions.get(arg)) |arg_instr| {
                    try list.write(.{ .key = "aka", .value = try std.fmt.allocPrint(self.allocator, "{f}", .{self.instrLabel(arg_instr.*)}) });
                }
            }
        }
    }

    fn sourceLocation(self: *const Converter, meta: hlo.OpMetadata) !?[]const u8 {
        if (self.stack_frame_index) |index| {
            // All references in StackFrameIndexProto are 1-based; zero is absent.
            // Resolve only the instruction's frame, not its parent call stack.
            if (indexed(index.stack_frames.items, meta.stack_frame_id)) |frame| {
                if (indexed(index.file_locations.items, frame.file_location_id)) |location| {
                    if (indexed(index.file_names.items, location.file_name_id)) |file| {
                        if (file.len > 0 and location.line > 0) {
                            if (location.column > 0) return try std.fmt.allocPrint(self.allocator, "{s}:{d}:{d}", .{ file, location.line, location.column });
                            return try std.fmt.allocPrint(self.allocator, "{s}:{d}", .{ file, location.line });
                        }
                    }
                }
            }
        }
        if (meta.source_file.len > 0) return try std.fmt.allocPrint(self.allocator, "{s}:{d}", .{ meta.source_file, meta.source_line });
        return null;
    }

    fn referenceInstruction(self: *const Converter, instr: hlo.HloInstructionProto) ?*const hlo.HloInstructionProto {
        const reference = self.reference orelse return null;
        var result: ?*const hlo.HloInstructionProto = null;
        for (reference.instructions.items) |*candidate| {
            const matches = if (std.mem.eql(u8, instr.opcode, "parameter"))
                std.mem.eql(u8, candidate.opcode, "parameter") and candidate.parameter_number == instr.parameter_number
            else if (instr.metadata) |meta|
                meta.op_name.len > 0 and std.mem.eql(u8, candidate.name, meta.op_name)
            else
                false;
            if (matches) {
                if (result != null) return null;
                result = candidate;
            }
        }
        return result;
    }

    fn instrLabel(self: *const Converter, instr: hlo.HloInstructionProto) InstrLabel {
        return .{ .instr = instr, .converter = self };
    }
};

const InstrLabel = struct {
    instr: hlo.HloInstructionProto,
    converter: *const Converter,

    pub fn format(
        self: InstrLabel,
        writer: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        const instr = self.instr;
        const converter = self.converter;
        const opcode = instr.opcode;
        if (std.mem.eql(u8, opcode, "parameter")) {
            return try writer.print("Parameter {d}", .{instr.parameter_number});
        }

        if (std.mem.eql(u8, opcode, "constant") and instr.literal != null) {
            return try renderLiteral(writer, instr.literal.?);
        }

        if (std.mem.eql(u8, opcode, "fusion")) {
            return try writer.print("{s}", .{instr.name});
        }

        if (std.mem.eql(u8, opcode, "custom-call")) {
            return try writer.print("custom-call: {s}", .{if (instr.custom_call_target.len > 0) instr.custom_call_target else instr.name});
        }

        if (std.mem.eql(u8, opcode, "copy") or isCollective(opcode)) return writer.writeAll(opcode);

        if (std.mem.eql(u8, opcode, "reduce")) r: {
            const reducer = converter.computations.get(instr.called_computation_ids.items[0]).?;
            // Complex reduction don't give a name cause it's probably misleading.
            if (reducer.instructions.items.len > 3) break :r;
            const root = self.converter.instructions.get(reducer.root_id).?;
            const reduce_label = if (std.mem.eql(u8, root.opcode, "compare"))
                root.comparison_direction
            else
                root.opcode;
            return try writer.print("{s}.{s}", .{ instr.opcode, reduce_label });
        }

        if (std.mem.eql(u8, opcode, "get-tuple-element")) {
            const tuple_instr = self.converter.instructions.get(instr.operand_ids.items[0]).?.*;
            return try writer.print("{f}[{d}]", .{ self.converter.instrLabel(tuple_instr), instr.tuple_index });
        }

        // The HLO instruction name contains usually the opcode, e.g. "%add.42" is
        // an add instruction.  In this case we render just the name.
        if (std.mem.startsWith(u8, instr.name, opcode)) {
            var name = instr.name;
            if (name.len > opcode.len) {
                if (std.fmt.parseInt(u64, name[opcode.len + 1 ..], 10)) |_| {
                    name = opcode;
                } else |_| {}
            }
            return try writer.writeAll(name);
        }

        return try writer.print("{s}.{s}", .{ instr.name, instr.opcode });
    }
};

const InstrShape = struct {
    shape: *const hlo.ShapeProto,

    pub fn format(
        self: InstrShape,
        writer: *std.Io.Writer,
    ) !void {

        // Use syntax supported by the frontend. The "tensor<>" won't be displayed, only what's inside.
        try writer.writeAll("tensor<");
        const dims = self.shape.dimensions.items;
        for (dims) |d| {
            try writer.print("{d}x", .{d});
        }
        try writer.print("{s}>", .{@tagName(self.shape.element_type)});
    }
};

const LiteralProto = hlo.LiteralProto;

fn renderLiteral(writer: anytype, literal: LiteralProto) !void {
    const dtype = literal.shape.?.element_type;
    const dims = literal.shape.?.dimensions.items;
    var n: usize = 1;
    for (dims) |d| n *= @intCast(d);
    // Only show first 16 elements. The shape will be displayed anyway.
    n = @min(n, 16);

    // The code below assumes little endian to correctly load f16, u16 and bf16 which aren't natively supported by protobuf.
    std.debug.assert(builtin.target.cpu.arch.endian() == .little);
    // Only render scalar literals.
    _ = try writer.writeAll("constant(");
    switch (dtype) {
        // TODO check how sub-bytes member are encoded
        // .PRED => ,
        // .S1 => ,
        // .S2 => ,
        // .S4 => ,
        .S8 => try renderValues(writer, "d", std.mem.bytesAsSlice(i8, literal.s8s), n),
        .S16 => try renderValues(writer, "d", std.mem.bytesAsSlice(i16, literal.s16s), n),
        .S32 => try renderValues(writer, "d", literal.s32s.items, n),
        .S64 => try renderValues(writer, "d", literal.s64s.items, n),
        // .U1 => ,
        // .U2 => ,
        // .U4 => ,
        .U8 => try renderValues(writer, "d", literal.u8s, n),
        .U16 => try renderValues(writer, "d", std.mem.bytesAsSlice(u16, literal.u16s), n),
        .U32 => try renderValues(writer, "d", literal.u32s.items, n),
        .U64 => try renderValues(writer, "d", literal.u64s.items, n),
        .F32 => try renderValues(writer, "e", literal.f32s.items, n),
        .F64 => try renderValues(writer, "e", literal.f64s.items, n),
        .F16 => try renderValues(writer, "e", std.mem.bytesAsSlice(f16, literal.f16s), n),
        .BF16 => try renderValues(writer, "e", std.mem.bytesAsSlice(BFloat16, literal.bf16s), n),
        .F8E5M2 => try renderValues(writer, "e", std.mem.bytesAsSlice(F8E5M2, literal.f8e5m2s), n),
        .F8E5M2FNUZ => try renderValues(writer, "e", std.mem.bytesAsSlice(F8E5M2, literal.f8e5m2fnuzs), n),
        .F8E3M4 => try renderValues(writer, "e", std.mem.bytesAsSlice(F8E3M4, literal.f8e3m4s), n),
        .F8E4M3 => try renderValues(writer, "e", std.mem.bytesAsSlice(F8E4M3, literal.f8e4m3s), n),
        .F8E4M3FN => try renderValues(writer, "e", std.mem.bytesAsSlice(F8E4M3, literal.f8e4m3fns), n),
        .F8E4M3FNUZ => try renderValues(writer, "e", std.mem.bytesAsSlice(F8E4M3, literal.f8e4m3fnuzs), n),
        .F8E4M3B11FNUZ => try renderValues(writer, "e", std.mem.bytesAsSlice(F8E4M3, literal.f8e4m3b11fnuzs), n),
        // C64 = 15;
        // C128 = 18;
        else => _ = try writer.writeAll("???"),
    }
    try writer.print(", {s})", .{@tagName(dtype)});
}

fn renderValues(writer: *std.Io.Writer, comptime fmt: []const u8, values: anytype, n: usize) std.Io.Writer.Error!void {
    try writer.writeAll("{ ");
    for (values[0..@min(n, values.len)], 0..) |value, i| {
        if (i != 0) try writer.writeAll(", ");
        const number = if (@typeInfo(@TypeOf(value)) == .@"struct") value.toF32() else value;
        try writer.print("{" ++ fmt ++ "}", .{number});
    }
    try writer.writeAll(" }");
}

fn FloatType(exponent_bits: u8, mantissa_bits: u8) type {
    const sign_bits = 1;
    const bit_size = sign_bits + exponent_bits + mantissa_bits;
    if (bit_size % 8 != 0) @compileError("FloatType should have a number of bits divisible by 8");

    return packed struct(@Int(.unsigned, bit_size)) {
        const Self = @This();

        mantissa: @Int(.unsigned, mantissa_bits),
        exponent: @Int(.unsigned, exponent_bits),
        sign: u1,

        /// Lossless conversion to f32.
        pub fn toF32(self: Self) f32 {
            var vf32: Float32 = undefined;
            if (@hasDecl(Self, "isInf") and self.isInf()) {
                return if (self.sign == 0) std.math.inf(f32) else -std.math.inf(f32);
            }
            vf32 = .{
                .sign = self.sign,
                .exponent = if (self.exponent == 0) 0 else @intCast(@as(i16, self.exponent) + Float32.expBias() - Self.expBias()),
                .mantissa = self.f32Mantissa(),
            };
            return @bitCast(vf32);
        }

        fn f32Mantissa(self: Self) @FieldType(Float32, "mantissa") {
            @setRuntimeSafety(false);
            const f32_mantissa_bits = @bitSizeOf(@FieldType(Float32, "mantissa"));

            const Res = @FieldType(Float32, "mantissa");
            return @shlExact(@as(Res, self.mantissa), f32_mantissa_bits - mantissa_bits);
        }

        fn expBias() u8 {
            if (exponent_bits == 1) return 0;
            return (1 << (exponent_bits - 1)) - 1;
        }

        pub fn format(
            self: @This(),
            writer: *std.Io.Writer,
        ) !void {
            try writer.print("{e}", .{self.toF32()});
        }
    };
}
const BFloat16 = FloatType(8, 7);
const Float32 = FloatType(8, 23);
const F8E5M2 = FloatType(5, 2);
const F8E4M3 = FloatType(4, 3);
const F8E3M4 = FloatType(3, 4);

test "decode HLO and emit Model Explorer graphs" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    // Exercise the mutually recursive XLA messages through binary decoding.
    var physical_shape: hlo.ShapeProto = .{ .element_type = .F32 };
    var layout: hlo.LayoutProto = .{ .physical_shape = &physical_shape };
    const shape: hlo.ShapeProto = .{ .element_type = .F32, .layout = &layout };
    var instructions = [_]hlo.HloInstructionProto{
        .{ .id = 1, .name = "parameter.1", .opcode = "parameter", .shape = shape },
        .{
            .id = 2,
            .name = "constant.2",
            .opcode = "constant",
            .shape = shape,
            .literal = .{ .shape = shape, .f32s = .fromOwnedSlice(@constCast(&[_]f32{2.5})) },
        },
        .{
            .id = 3,
            .name = "call\"\\\n",
            .opcode = "custom-call",
            .shape = shape,
            .operand_ids = .fromOwnedSlice(@constCast(&[_]i64{ 1, 2 })),
            .metadata = .{ .source_file = "a\"\\b.zig", .source_line = 42, .op_name = "op\nname", .op_type = "type" },
        },
    };
    var computations = [_]hlo.HloComputationProto{.{
        .id = 1,
        .name = "entry",
        .root_id = 3,
        .instructions = .fromOwnedSlice(&instructions),
    }};
    const proto: hlo.HloProto = .{ .hlo_module = .{
        .name = "module",
        .computations = .fromOwnedSlice(&computations),
    } };
    var encoded: std.Io.Writer.Allocating = .init(allocator);
    try proto.encode(&encoded.writer, allocator);
    var reader: std.Io.Reader = .fixed(encoded.written());
    const decoded = try hlo.HloProto.decode(&reader, allocator);
    try std.testing.expectEqual(.F32, decoded.hlo_module.?.computations.items[0].instructions.items[0].shape.?.layout.?.physical_shape.?.element_type);

    var output: std.Io.Writer.Allocating = .init(allocator);
    var converter: Converter = .{ .allocator = allocator };
    try converter.emitModule(&output.writer, decoded, null);
    const json = try std.json.parseFromSlice(std.json.Value, allocator, output.written(), .{});
    const root = json.value.object;
    try std.testing.expectEqualStrings("module", root.get("label").?.string);
    const graph = root.get("graphs").?.array.items[0].object;
    try std.testing.expectEqualStrings("entry", graph.get("id").?.string);
    const nodes = graph.get("nodes").?.array.items;
    try std.testing.expectEqual(@as(usize, 3), nodes.len);
    try std.testing.expectEqualStrings("Parameter 0\nreads: 0B", nodes[0].object.get("label").?.string);
    try std.testing.expectEqualStrings("constant({ 2.5e0 }, F32)\nreads: 0B", nodes[1].object.get("label").?.string);
    try std.testing.expectEqualStrings("custom-call: call\"\\\n\nreads: 8B", nodes[2].object.get("label").?.string);
    const attrs = nodes[2].object.get("attrs").?.array.items;
    try std.testing.expectEqualStrings("a\"\\b.zig:42", attrs[2].object.get("value").?.string);
    try std.testing.expectEqualStrings("type:op\nname", attrs[3].object.get("value").?.string);
    const edges = nodes[2].object.get("incomingEdges").?.array.items;
    try std.testing.expectEqual(@as(usize, 2), edges.len);
    try std.testing.expectEqualStrings("1", edges[0].object.get("sourceNodeId").?.string);
    try std.testing.expectEqualStrings("1", edges[1].object.get("targetNodeInputId").?.string);
    const outputs = nodes[0].object.get("outputsMetadata").?.array.items;
    try std.testing.expectEqualStrings("tensor<F32>", outputs[0].object.get("attrs").?.array.items[0].object.get("value").?.string);
}

test "missing HLO module and truncated protobuf are rejected" {
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    var converter: Converter = .{ .allocator = std.testing.allocator };
    try std.testing.expectError(error.NoModule, converter.emitModule(&output.writer, .{}, null));
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var reader: std.Io.Reader = .fixed("\x0a\x05\x01");
    if (hlo.HloProto.decode(&reader, arena.allocator())) |_| {
        return error.ExpectedDecodeError;
    } else |_| {}
}

test "JSON closing delimiters propagate writer failures" {
    var buffer: [1]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);
    var object = try json_writer.dict(&writer);
    try std.testing.expectError(error.WriteFailed, object.close());
}

fn isCollective(opcode: []const u8) bool {
    inline for (.{ "all-gather", "all-reduce", "reduce-scatter", "all-to-all", "collective-permute", "collective-broadcast" }) |name| {
        if (std.mem.eql(u8, opcode, name)) return true;
    }
    return false;
}

fn logicalBytes(shape: hlo.ShapeProto) ?u64 {
    for (shape.is_dynamic_dimension.items) |dynamic| if (dynamic) return null;
    const bits: u64 = switch (shape.element_type) {
        .PRED, .S8, .U8 => 8,
        .S16, .U16, .F16, .BF16 => 16,
        .S32, .U32, .F32 => 32,
        .S64, .U64, .F64, .C64 => 64,
        .C128 => 128,
        else => return null,
    };
    var size = bits;
    for (shape.dimensions.items) |dim| {
        if (dim < 0) return null;
        size = std.math.mul(u64, size, @intCast(dim)) catch return null;
    }
    return std.math.divCeil(u64, size, 8) catch null;
}

fn shardingText(allocator: std.mem.Allocator, sharding: hlo.OpSharding) ![]const u8 {
    if (sharding.named_sharding != null) return std.fmt.allocPrint(allocator, "named: {any}", .{sharding.named_sharding.?});
    if (sharding.type == .REPLICATED) return "replicated across the partition group";
    if (sharding.type != .OTHER) return std.fmt.allocPrint(allocator, "{s}", .{@tagName(sharding.type)});
    var replication: i64 = 1;
    const dims = sharding.tile_assignment_dimensions.items;
    if (sharding.replicate_on_last_tile_dim and dims.len > 0) {
        replication = dims[dims.len - 1];
    } else if (sharding.last_tile_dims.items.len <= dims.len) {
        for (sharding.last_tile_dims.items, dims[dims.len - sharding.last_tile_dims.items.len ..]) |kind, dim| {
            if (kind == .REPLICATED) replication = std.math.mul(i64, replication, dim) catch return error.InvalidSharding;
        }
    }
    return std.fmt.allocPrint(allocator, "tiles={any}; replication factor={d}; subgroup types={any}; devices={any}; iota reshape={any}; iota transpose={any}", .{ dims, replication, sharding.last_tile_dims.items, sharding.tile_assignment_devices.items, sharding.iota_reshape_dims.items, sharding.iota_transpose_perm.items });
}

const ReplicaGroupList = @typeInfo(@FieldType(hlo.HloInstructionProto, "replica_group_list")).optional.child;

fn collectiveGroupsText(allocator: std.mem.Allocator, groups: ReplicaGroupList) ![]const u8 {
    switch (groups) {
        .collective_device_list => |list| return std.fmt.allocPrint(allocator, "explicit groups: {any}", .{list.replica_groups.items}),
        .iota_collective_device_list => |list| return std.fmt.allocPrint(allocator, "{d} groups of {d}; iota reshape={any}, transpose={any}", .{ list.num_replica_groups, list.num_devices_per_group, list.iota_reshape_dims.items, list.iota_transpose_perm.items }),
        .mesh_axes_replica_group_list => |list| {
            var text: std.Io.Writer.Allocating = .init(allocator);
            if (list.mesh) |mesh| {
                try text.writer.writeAll("mesh[");
                for (mesh.axes.items, 0..) |axis, i| try text.writer.print("{s}{s}={d}", .{ if (i > 0) ", " else "", axis.name, axis.size });
                try text.writer.writeAll("]; collective axes[");
                for (list.axes.items, 0..) |axis, i| {
                    try text.writer.print("{s}{d}", .{ if (i > 0) ", " else "", axis.mesh_axis_index });
                    if (axis.sub_axis_info) |sub| try text.writer.print("(pre={d}, size={d})", .{ sub.pre_size, sub.size });
                }
                try text.writer.print("]; devices={any}", .{mesh.device_ids.items});
                if (mesh.device_ids.items.len == 0) try text.writer.writeAll(" (implicit iota order)");
                if (mesh.iota_transform) |transform| try text.writer.print(" reshape={any}, transpose={any}", .{ transform.reshape_dims.items, transform.transpose_perm.items });
            } else try text.writer.writeAll("unknown mesh");
            return text.toOwnedSlice();
        },
    }
}

fn collectiveGroupSize(instr: hlo.HloInstructionProto) ?u64 {
    if (instr.replica_group_list) |groups| switch (groups) {
        .collective_device_list => |list| return uniformGroupSize(list.replica_groups.items),
        .iota_collective_device_list => |list| return if (list.num_devices_per_group > 0) @intCast(list.num_devices_per_group) else null,
        .mesh_axes_replica_group_list => |list| {
            const mesh = list.mesh orelse return null;
            var count: u64 = 1;
            for (list.axes.items) |axis| {
                if (axis.mesh_axis_index < 0 or axis.mesh_axis_index >= mesh.axes.items.len) return null;
                const size = if (axis.sub_axis_info) |sub| sub.size else mesh.axes.items[@intCast(axis.mesh_axis_index)].size;
                if (size <= 0) return null;
                count = std.math.mul(u64, count, @intCast(size)) catch return null;
            }
            return count;
        },
    };
    return uniformGroupSize(instr.replica_groups.items);
}

fn uniformGroupSize(groups: []const hlo.ReplicaGroup) ?u64 {
    if (groups.len == 0 or groups[0].replica_ids.items.len == 0) return null;
    const size = groups[0].replica_ids.items.len;
    for (groups) |group| if (group.replica_ids.items.len != size) return null;
    return size;
}

fn ringSendBytes(opcode: []const u8, bytes: u64, count: u64) ?u64 {
    if (count == 0) return null;
    const product = std.math.mul(u64, bytes, count - 1) catch return null;
    if (std.mem.eql(u8, opcode, "all-gather")) return product;
    if (std.mem.eql(u8, opcode, "all-reduce")) {
        const doubled = std.math.mul(u64, product, 2) catch return null;
        if (doubled % count != 0) return null;
        return doubled / count;
    }
    return null;
}

test "logical payloads and ideal ring traffic" {
    var dims = [_]i64{ 1, 16, 4, 128 };
    var shape: hlo.ShapeProto = .{ .element_type = .BF16, .dimensions = .fromOwnedSlice(&dims) };
    try std.testing.expectEqual(@as(?u64, 16384), logicalBytes(shape));
    try std.testing.expectEqual(@as(?u64, 16384), ringSendBytes("all-gather", 16384, 2));
    try std.testing.expectEqual(@as(?u64, 131072), ringSendBytes("all-reduce", 131072, 2));
    try std.testing.expectEqual(@as(?u64, 0), ringSendBytes("all-reduce", 131072, 1));
    try std.testing.expectEqual(@as(?u64, null), ringSendBytes("all-gather", std.math.maxInt(u64), 3));
    var dynamic = [_]bool{true};
    shape.is_dynamic_dimension = .fromOwnedSlice(&dynamic);
    try std.testing.expectEqual(@as(?u64, null), logicalBytes(shape));
}

test "explicit partial replication and missing collective groups" {
    var dims = [_]i64{ 2, 1, 4 };
    const description = try shardingText(std.testing.allocator, .{ .type = .OTHER, .tile_assignment_dimensions = .fromOwnedSlice(&dims), .replicate_on_last_tile_dim = true });
    defer std.testing.allocator.free(description);
    try std.testing.expect(std.mem.indexOf(u8, description, "replication factor=4") != null);
    try std.testing.expectEqual(@as(?u64, null), collectiveGroupSize(.{}));
}

test "global references match parameter numbers and reject ambiguous provenance" {
    var candidates = [_]hlo.HloInstructionProto{
        .{ .name = "arg", .opcode = "parameter", .parameter_number = 3 },
        .{ .name = "dot.1", .opcode = "dot" },
        .{ .name = "dot.1", .opcode = "dot" },
    };
    const reference: hlo.HloComputationProto = .{ .instructions = .fromOwnedSlice(&candidates) };
    const converter: Converter = .{ .allocator = std.testing.allocator, .reference = &reference };
    try std.testing.expect(converter.referenceInstruction(.{ .opcode = "parameter", .parameter_number = 3 }) == &candidates[0]);
    try std.testing.expect(converter.referenceInstruction(.{ .metadata = .{ .op_name = "dot.1" } }) == null);
    try std.testing.expect(converter.referenceInstruction(.{ .name = "arg", .opcode = "parameter", .parameter_number = 4 }) == null);
}

test {
    _ = gpu_info;
    _ = dump_context;
    _ = graph_partitioner;
}

fn indexed(items: anytype, id: i32) ?@typeInfo(@TypeOf(items)).pointer.child {
    if (id <= 0 or id > items.len) return null;
    return items[@intCast(id - 1)];
}

test "source locations resolve protobuf stack frames and safely fall back" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var index: hlo.StackFrameIndexProto = .{};
    try index.file_names.append(allocator, "mnist.zig");
    try index.file_locations.append(allocator, .{ .file_name_id = 1, .line = 71, .column = 38 });
    try index.stack_frames.append(allocator, .{ .file_location_id = 1, .parent_frame_id = 1 });
    var module: hlo.HloModuleProto = .{ .name = "locations", .stack_frame_index = index };
    var comp: hlo.HloComputationProto = .{ .name = "entry" };
    try comp.instructions.append(allocator, .{ .name = "layer", .opcode = "copy", .shape = .{ .element_type = .F32 }, .metadata = .{ .stack_frame_id = 1 } });
    try module.computations.append(allocator, comp);
    var encoded: std.Io.Writer.Allocating = .init(allocator);
    try (hlo.HloProto{ .hlo_module = module }).encode(&encoded.writer, allocator);
    var reader: std.Io.Reader = .fixed(encoded.written());
    const decoded = try hlo.HloProto.decode(&reader, allocator);
    var converter: Converter = .{ .allocator = allocator };
    var json: std.Io.Writer.Allocating = .init(allocator);
    try converter.emitModule(&json.writer, decoded, null);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json.written(), .{});
    const attrs = parsed.value.object.get("graphs").?.array.items[0].object.get("nodes").?.array.items[0].object.get("attrs").?.array.items;
    var found = false;
    for (attrs) |attr| {
        if (std.mem.eql(u8, attr.object.get("key").?.string, "loc")) {
            try std.testing.expectEqualStrings("mnist.zig:71:38", attr.object.get("value").?.string);
            found = true;
        }
    }
    try std.testing.expect(found);
    try std.testing.expectEqualStrings("legacy.zig:42", (try converter.sourceLocation(.{ .stack_frame_id = 99, .source_file = "legacy.zig", .source_line = 42 })).?);
    try std.testing.expect(try converter.sourceLocation(.{ .stack_frame_id = -1 }) == null);
    try std.testing.expect(try converter.sourceLocation(.{}) == null);
    converter.stack_frame_index.?.stack_frames.items[0].file_location_id = 99;
    try std.testing.expect(try converter.sourceLocation(.{ .stack_frame_id = 1 }) == null);
    converter.stack_frame_index.?.stack_frames.items[0].file_location_id = 1;
    converter.stack_frame_index.?.file_locations.items[0].file_name_id = 99;
    try std.testing.expect(try converter.sourceLocation(.{ .stack_frame_id = 1 }) == null);
}

fn collectiveDeviceCount(instr: hlo.HloInstructionProto) ?u64 {
    if (instr.replica_group_list) |groups| switch (groups) {
        .mesh_axes_replica_group_list => |list| {
            const mesh = list.mesh orelse return null;
            var count: u64 = 1;
            for (mesh.axes.items) |axis| {
                if (axis.size <= 0) return null;
                count = std.math.mul(u64, count, @intCast(axis.size)) catch return null;
            }
            return if (mesh.axes.items.len > 0) count else null;
        },
        .iota_collective_device_list => |list| {
            if (list.num_replica_groups <= 0 or list.num_devices_per_group <= 0) return null;
            return std.math.mul(u64, @intCast(list.num_replica_groups), @intCast(list.num_devices_per_group)) catch null;
        },
        // Explicit groups can cover only a subset of the available devices.
        .collective_device_list => return null,
    };
    return null;
}

test "compact collective bytes and device summaries" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var group: hlo.ReplicaGroup = .{};
    try group.replica_ids.appendSlice(allocator, &.{ 0, 1 });
    var instr: hlo.HloInstructionProto = .{ .opcode = "all-reduce" };
    try instr.replica_groups.append(allocator, group);
    try std.testing.expectEqual(@as(?u64, 2), collectiveGroupSize(instr));
    try group.replica_ids.appendSlice(allocator, &.{ 2, 3 });
    instr.replica_groups.items[0] = group;
    instr.opcode = "all-to-all";
    try std.testing.expectEqual(@as(?u64, 4), collectiveGroupSize(instr));
}

test "partition namespaces retain edges and called computations" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const shape: hlo.ShapeProto = .{ .element_type = .F32 };
    var instructions = [_]hlo.HloInstructionProto{
        .{ .id = 1, .opcode = "parameter", .shape = shape },
        .{ .id = 2, .opcode = "all-gather", .shape = shape, .operand_ids = .fromOwnedSlice(@constCast(&[_]i64{1})) },
        .{ .id = 3, .opcode = "call", .shape = shape, .operand_ids = .fromOwnedSlice(@constCast(&[_]i64{2})), .called_computation_ids = .fromOwnedSlice(@constCast(&[_]i64{20})) },
    };
    var callee = [_]hlo.HloInstructionProto{.{ .id = 4, .opcode = "parameter", .shape = shape }};
    var computations = [_]hlo.HloComputationProto{
        .{ .id = 10, .name = "entry", .instructions = .fromOwnedSlice(&instructions) },
        .{ .id = 20, .name = "callee", .instructions = .fromOwnedSlice(&callee) },
    };
    var converter: Converter = .{ .allocator = allocator };
    var output: std.Io.Writer.Allocating = .init(allocator);
    try converter.emitModule(&output.writer, .{ .hlo_module = .{ .computations = .fromOwnedSlice(&computations) } }, null);
    const json = try std.json.parseFromSlice(std.json.Value, allocator, output.written(), .{});
    const graphs = json.value.object.get("graphs").?.array.items;
    try std.testing.expectEqual(@as(usize, 2), graphs.len);
    const nodes = graphs[0].object.get("nodes").?.array.items;
    try std.testing.expectEqualStrings("stage_000", nodes[0].object.get("namespace").?.string);
    try std.testing.expectEqualStrings("stage_001", nodes[1].object.get("namespace").?.string);
    try std.testing.expectEqualStrings("stage_001", nodes[2].object.get("namespace").?.string);
    try std.testing.expectEqualStrings("1", nodes[1].object.get("incomingEdges").?.array.items[0].object.get("sourceNodeId").?.string);
    try std.testing.expectEqualStrings("callee", nodes[2].object.get("subgraphIds").?.array.items[0].string);
    try std.testing.expectEqualStrings("stage_000", graphs[1].object.get("nodes").?.array.items[0].object.get("namespace").?.string);
}

const PartitionMetrics = struct {
    communication: ?u64 = 0,
    memory: ?u64 = 0,
    opaque_operations: u64 = 0,
};

fn addEstimate(a: ?u64, b: ?u64) ?u64 {
    return std.math.add(u64, a orelse return null, b orelse return null) catch null;
}

fn isMetadataOnly(opcode: []const u8) bool {
    inline for (.{ "bitcast", "tuple", "get-tuple-element", "parameter", "constant", "after-all" }) |name| {
        if (std.mem.eql(u8, opcode, name)) return true;
    }
    return false;
}

fn payloadBytes(shape: hlo.ShapeProto) ?u64 {
    if (shape.element_type == .TOKEN) return 0;
    if (shape.element_type == .TUPLE) {
        var total: ?u64 = 0;
        for (shape.tuple_shapes.items) |child| total = addEstimate(total, payloadBytes(child));
        return total;
    }
    return logicalBytes(shape);
}

const Cmr = struct {
    metric: PartitionMetrics,
    separator: []const u8 = "/",

    pub fn format(self: Cmr, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        const memory = self.metric.memory orelse return writer.writeAll("unknown (logical reads unavailable)");
        // Both sides must share the denominator's unit; independent {B}
        // formatting would choose a different unit for each side.
        const units = [_][]const u8{ "B", "KB", "MB", "GB", "TB", "PB", "EB" };
        var unit: usize = 0;
        var divisor: u64 = 1;
        while (unit + 1 < units.len and memory / divisor >= 1000) : (unit += 1) divisor *= 1000;
        const denominator = @as(f64, @floatFromInt(memory)) / @as(f64, @floatFromInt(divisor));
        const communication = self.metric.communication orelse return writer.print("(? {s} {d:.1} {s})", .{ self.separator, denominator, units[unit] });
        const numerator = @as(f64, @floatFromInt(communication)) / @as(f64, @floatFromInt(divisor));
        if (communication > 0 and numerator < 0.05) return writer.print("({e:.1} {s} {d:.1} {s})", .{ numerator, self.separator, denominator, units[unit] });
        return writer.print("({d:.1} {s} {d:.1} {s})", .{ numerator, self.separator, denominator, units[unit] });
    }
};

test "CMR excludes views and async bookkeeping and counts kernel inputs once" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var dims = [_]i64{256};
    const shape: hlo.ShapeProto = .{ .element_type = .F32, .dimensions = .fromOwnedSlice(&dims) };
    var nodes = [_]hlo.HloInstructionProto{
        .{ .id = 1, .opcode = "parameter", .shape = shape },
        .{ .id = 2, .opcode = "async-start", .shape = shape, .operand_ids = .fromOwnedSlice(@constCast(&[_]i64{1})), .called_computation_ids = .fromOwnedSlice(@constCast(&[_]i64{20})) },
        .{ .id = 3, .opcode = "async-done", .shape = shape, .operand_ids = .fromOwnedSlice(@constCast(&[_]i64{2})) },
        .{ .id = 4, .opcode = "bitcast", .shape = shape, .operand_ids = .fromOwnedSlice(@constCast(&[_]i64{3})) },
        .{ .id = 5, .opcode = "fusion", .shape = shape, .operand_ids = .fromOwnedSlice(@constCast(&[_]i64{4})) },
        .{ .id = 6, .opcode = "custom-call", .shape = shape, .operand_ids = .fromOwnedSlice(@constCast(&[_]i64{ 4, 5 })) },
        .{ .id = 7, .opcode = "tuple", .shape = shape, .operand_ids = .fromOwnedSlice(@constCast(&[_]i64{ 5, 6 })) },
    };
    var collective = [_]hlo.HloInstructionProto{
        .{ .id = 8, .opcode = "parameter", .shape = shape },
        .{ .id = 9, .opcode = "all-to-all", .shape = shape, .operand_ids = .fromOwnedSlice(@constCast(&[_]i64{8})) },
    };
    var comps = [_]hlo.HloComputationProto{
        .{ .id = 10, .name = "entry", .root_id = 7, .instructions = .fromOwnedSlice(&nodes) },
        .{ .id = 20, .name = "async_collective", .root_id = 9, .instructions = .fromOwnedSlice(&collective) },
    };
    const module: hlo.HloModuleProto = .{ .computations = .fromOwnedSlice(&comps) };
    var partitions = try graph_partitioner.partitionModule(allocator, module);
    defer partitions.deinit(allocator);
    const partition = partitions.computations.getPtr(10).?;
    try std.testing.expectEqual(@as(usize, 2), partition.node_counts.len);
    for (2..8) |id| try std.testing.expectEqual(@as(usize, 1), partition.instructions.get(@intCast(id)).?);
    var converter: Converter = .{ .allocator = allocator };
    var output: std.Io.Writer.Allocating = .init(allocator);
    try converter.emitModule(&output.writer, .{ .hlo_module = module }, null);
    const metrics = try converter.partitionMetrics(comps[0], partition);
    try std.testing.expectEqual(@as(?u64, 0), metrics[0].memory);
    try std.testing.expectEqual(@as(?u64, 1024), metrics[1].communication);
    try std.testing.expectEqual(@as(?u64, 3072), metrics[1].memory);
    try std.testing.expectEqual(@as(u64, 1), metrics[1].opaque_operations);
    const json = try std.json.parseFromSlice(std.json.Value, allocator, output.written(), .{});
    const attrs = json.value.object.get("graphs").?.array.items[0].object.get("groupNodeAttributes").?.object.get("stage_001").?.object;
    try std.testing.expectEqualStrings("(1.0 / 3.1 KB)", attrs.get("CMR").?.string);
    // Unknown shapes must not silently reduce the denominator.
    nodes[4].shape = null;
    const unknown = try converter.partitionMetrics(comps[0], partition);
    try std.testing.expectEqual(@as(?u64, null), unknown[1].memory);
}

test "CMR preserves denominator magnitude, tuple payloads and unknown estimates" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    try std.testing.expectEqualStrings("(1.0 / 52.0 MB)", try std.fmt.allocPrint(allocator, "{f}", .{Cmr{ .metric = .{ .communication = 1_000_000, .memory = 52_000_000 } }}));
    try std.testing.expectEqualStrings("(0.0 / 0.0 B)", try std.fmt.allocPrint(allocator, "{f}", .{Cmr{ .metric = .{} }}));
    try std.testing.expectEqualStrings("(? / 52.0 MB)", try std.fmt.allocPrint(allocator, "{f}", .{Cmr{ .metric = .{ .communication = null, .memory = 52_000_000 } }}));
    try std.testing.expectEqualStrings("unknown (logical reads unavailable)", try std.fmt.allocPrint(allocator, "{f}", .{Cmr{ .metric = .{ .memory = null } }}));
    try std.testing.expect(std.mem.indexOf(u8, try std.fmt.allocPrint(allocator, "{f}", .{Cmr{ .metric = .{ .communication = 1, .memory = 1_000_000_000 } }}), "e") != null);
    var shapes = [_]hlo.ShapeProto{ .{ .element_type = .F32 }, .{ .element_type = .BF16 }, .{ .element_type = .TOKEN } };
    const tuple: hlo.ShapeProto = .{ .element_type = .TUPLE, .tuple_shapes = .fromOwnedSlice(&shapes) };
    try std.testing.expectEqual(@as(?u64, 6), payloadBytes(tuple));
    const converter: Converter = .{ .allocator = allocator };
    try std.testing.expectEqual(@as(?u64, 2), converter.collectivePayload(.{ .opcode = "all-gather-start", .shape = tuple }).?.bytes);
    try std.testing.expectEqual(@as(?u64, null), addEstimate(std.math.maxInt(u64), 1));
}

fn instructionBytes(instr: hlo.HloInstructionProto) ?u64 {
    return if (instr.shape) |shape| payloadBytes(shape) else null;
}

fn capEstimate(value: ?u64, limit: ?u64) ?u64 {
    if (value == 0) return 0;
    return @min(value orelse return null, limit orelse return null);
}

fn isElementwiseRead(opcode: []const u8) bool {
    inline for (.{ "convert", "bitcast-convert", "add", "subtract", "multiply", "divide", "maximum", "minimum", "negate", "abs", "exp", "exponential", "log", "sqrt", "rsqrt", "sine", "cosine", "tanh", "compare", "select", "clamp", "and", "or", "xor", "not", "shift-left", "shift-right-arithmetic", "shift-right-logical" }) |name| {
        if (std.mem.eql(u8, opcode, name)) return true;
    }
    return false;
}

test "fusion reads propagate slices through views and conversions without intermediate traffic" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var nodes = [_]hlo.HloInstructionProto{
        readTestInstruction(1, "parameter", .F32, 1024, &.{}),
        readTestInstruction(2, "bitcast", .F32, 1024, &.{1}),
        readTestInstruction(3, "convert", .BF16, 1024, &.{2}),
        readTestInstruction(4, "slice", .BF16, 16, &.{3}),
        readTestInstruction(5, "add", .BF16, 16, &.{ 4, 4 }),
    };
    var converter: Converter = .{ .allocator = allocator };
    for (&nodes) |*node| try converter.instructions.put(allocator, node.id, node);
    const comp: hlo.HloComputationProto = .{ .root_id = 5, .instructions = .fromOwnedSlice(&nodes) };
    try std.testing.expectEqual(@as(?u64, 64), converter.fusionReads(comp));
    try std.testing.expectEqual(@as(?u64, 0), converter.instructionReads(nodes[1]));
    try std.testing.expectEqual(@as(?u64, 32), converter.instructionReads(nodes[3]));
    try converter.computations.put(allocator, 20, &comp);
    var outer = [_]hlo.HloInstructionProto{readTestInstruction(10, "fusion", .BF16, 16, &.{1})};
    outer[0].called_computation_ids = .fromOwnedSlice(@constCast(&[_]i64{20}));
    var partition: graph_partitioner.ComputationPartition = .{};
    try partition.instructions.put(allocator, 10, 0);
    partition.node_counts = try allocator.dupe(usize, &.{1});
    const metrics = try converter.partitionMetrics(.{ .instructions = .fromOwnedSlice(&outer) }, &partition);
    try std.testing.expectEqual(@as(?u64, 64), metrics[0].memory);
    nodes[3].shape = null;
    try std.testing.expectEqual(@as(?u64, null), converter.fusionReads(comp));
}

test "gather and dynamic slice count selected data and index reads" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var nodes = [_]hlo.HloInstructionProto{
        readTestInstruction(1, "parameter", .F32, 1024, &.{}),
        readTestInstruction(2, "parameter", .S32, 8, &.{}),
        readTestInstruction(3, "gather", .F32, 8, &.{ 1, 2 }),
    };
    var converter: Converter = .{ .allocator = allocator };
    for (&nodes) |*node| try converter.instructions.put(allocator, node.id, node);
    const comp: hlo.HloComputationProto = .{ .root_id = 3, .instructions = .fromOwnedSlice(&nodes) };
    try std.testing.expectEqual(@as(?u64, 64), converter.fusionReads(comp));
    try std.testing.expectEqual(@as(?u64, 64), converter.instructionReads(nodes[2]));
    nodes[1] = readTestInstruction(2, "parameter", .S32, 1, &.{});
    nodes[2] = readTestInstruction(3, "dynamic-slice", .F32, 16, &.{ 1, 2 });
    try std.testing.expectEqual(@as(?u64, 68), converter.fusionReads(comp));
    try std.testing.expectEqual(@as(?u64, 68), converter.instructionReads(nodes[2]));
}

test "scatter reads distinguish overwrite from read modify write" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var nodes = [_]hlo.HloInstructionProto{
        readTestInstruction(1, "parameter", .F32, 1024, &.{}),
        readTestInstruction(2, "parameter", .S32, 8, &.{}),
        readTestInstruction(3, "parameter", .F32, 16, &.{}),
        readTestInstruction(4, "scatter", .F32, 1024, &.{ 1, 2, 3 }),
    };
    nodes[3].called_computation_ids = .fromOwnedSlice(@constCast(&[_]i64{20}));
    var reducer = [_]hlo.HloInstructionProto{
        readTestInstruction(5, "parameter", .F32, 1, &.{}),
        readTestInstruction(6, "parameter", .F32, 1, &.{}),
        readTestInstruction(7, "add", .F32, 1, &.{ 5, 6 }),
    };
    reducer[1].parameter_number = 1;
    var update: hlo.HloComputationProto = .{ .root_id = 6, .instructions = .fromOwnedSlice(&reducer) };
    var converter: Converter = .{ .allocator = allocator };
    for (&nodes) |*node| try converter.instructions.put(allocator, node.id, node);
    for (&reducer) |*node| try converter.instructions.put(allocator, node.id, node);
    try converter.computations.put(allocator, 20, &update);
    const comp: hlo.HloComputationProto = .{ .root_id = 4, .instructions = .fromOwnedSlice(&nodes) };
    try std.testing.expectEqual(@as(?u64, 96), converter.fusionReads(comp));
    try std.testing.expectEqual(@as(?u64, 96), converter.instructionReads(nodes[3]));
    update.root_id = 7;
    try std.testing.expectEqual(@as(?u64, 160), converter.fusionReads(comp));
    try std.testing.expectEqual(@as(?u64, 160), converter.instructionReads(nodes[3]));
    converter.allocator = std.testing.failing_allocator;
    try std.testing.expectEqual(@as(?u64, null), converter.fusionReads(comp));
    try std.testing.expectEqual(@as(?u64, null), converter.instructionReads(nodes[3]));
}

fn readTestInstruction(id: i64, opcode: []const u8, comptime dtype: hlo.PrimitiveType, comptime elements: i64, operands: []const i64) hlo.HloInstructionProto {
    return .{
        .id = id,
        .opcode = opcode,
        .shape = .{ .element_type = dtype, .dimensions = .fromOwnedSlice(@constCast(&[_]i64{elements})) },
        .operand_ids = .fromOwnedSlice(@constCast(operands)),
    };
}

fn loadGpuInfo(allocator: std.mem.Allocator, io: std.Io, input: []const u8) !?GpuInfo {
    const stem = dump_context.dumpStem(input) orelse if (std.mem.endsWith(u8, input, ".hlo.pb")) input[0 .. input.len - ".hlo.pb".len] else return null;
    const path = try std.fmt.allocPrint(allocator, "{s}.gpu_target_config.pbtxt", .{stem});
    defer allocator.free(path);
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer file.close(io);
    var buffer: [8192]u8 = undefined;
    var reader = file.reader(io, &buffer);
    const content = try reader.interface.allocRemaining(allocator, .limited(1024 * 1024));
    defer allocator.free(content);
    return gpu_info.parse(content);
}
