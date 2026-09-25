const builtin = @import("builtin");
const std = @import("std");
const hlo = @import("@xla//xla/service:hlo_proto").proto;
const log = std.log.scoped(.hlo2json);

const json_writer = @import("json_writer.zig");

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
    try converter.emitModule(writer, proto);
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

    reference: ?*const hlo.HloComputationProto = null,
    entry_id: i64 = -1,

    const Assigned = struct { physical: i64, start: i64, len: i64 };

    pub fn emitModule(self: *Converter, writer: anytype, hlo_proto: hlo.HloProto) !void {
        const module = hlo_proto.hlo_module orelse return error.NoModule;
        self.entry_id = module.entry_computation_id;

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

        const W = @TypeOf(writer);
        var top_level = try json_writer.dict(writer);

        try top_level.write("label", module.name);
        var graphs = try top_level.openList("graphs");

        for (module.computations.items) |computation| {
            var graph = try graphs.openDict();
            try self.emitComputation(W, &graph, computation);
            try graph.close();
        }
        try graphs.close();
        try top_level.close();
    }

    pub fn emitComputation(self: *const Converter, Writer: type, graph: *json_writer.Dict(Writer), computation: hlo.HloComputationProto) !void {
        const comp_caller = self.computation_caller.get(computation.id);
        // The name of the caller node is better cause computation name is typically just "fusion".
        try graph.write("id", if (comp_caller) |caller| caller.name else computation.name);

        var nodes = try graph.openList("nodes");
        for (computation.instructions.items) |instr| {
            if (std.mem.eql(u8, instr.opcode, "get-tuple-element")) continue;
            var node = try nodes.openDict();
            // node name + attrs
            try node.writeTemplate("id", "\"{d}\"", .{instr.id});
            try node.write("namespace", "");
            try node.write("label", try std.fmt.allocPrint(self.allocator, "{f}", .{self.instrLabel(instr)}));
            {
                var attrs = try node.openList("attrs");
                try self.writeAttr(Writer, &attrs, instr, comp_caller);
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

    pub fn writeAttr(self: *const Converter, Writer: type, list: *json_writer.List(Writer), instr: hlo.HloInstructionProto, comp_caller: ?*const hlo.HloInstructionProto) !void {
        try list.write(.{ .key = "instruction", .value = instr.name });
        try list.write(.{ .key = "opcode", .value = instr.opcode });
        if (instr.custom_call_target.len > 0) try list.write(.{ .key = "custom call target", .value = instr.custom_call_target });
        if (isCollective(instr.opcode)) {
            try list.write(.{ .key = "channel", .value = try std.fmt.allocPrint(self.allocator, "{d}", .{instr.channel_id}) });
            try list.write(.{ .key = "global device IDs", .value = if (instr.use_global_device_ids) "true" else "false" });
            if (instr.replica_group_list) |groups| try list.write(.{ .key = "collective groups (XLA proto)", .value = try std.fmt.allocPrint(self.allocator, "{any}", .{groups}) });
            if (instr.replica_groups.items.len > 0) try list.write(.{ .key = "replica groups", .value = try std.fmt.allocPrint(self.allocator, "{any}", .{instr.replica_groups.items}) });
            if (instr.dimensions.items.len > 0) try list.write(.{ .key = "collective dimensions", .value = try std.fmt.allocPrint(self.allocator, "{any}", .{instr.dimensions.items}) });
            for (instr.operand_ids.items, 0..) |id, index| {
                if (self.instructions.get(id)) |operand| {
                    if (operand.shape) |shape| {
                        if (logicalBytes(shape)) |bytes| try list.write(.{ .key = try std.fmt.allocPrint(self.allocator, "input {d} payload bytes per device", .{index}), .value = try std.fmt.allocPrint(self.allocator, "{d}", .{bytes}) });
                    }
                }
            }
            if (instr.shape) |shape| {
                if (logicalBytes(shape)) |bytes| try list.write(.{ .key = "output payload bytes per device", .value = try std.fmt.allocPrint(self.allocator, "{d}", .{bytes}) });
            }
            try list.write(.{ .key = "traffic accounting", .value = "Logical payload, not measured wire traffic. Algorithm, protocol and topology are not specified by this HLO." });
        }
        if (instr.metadata) |meta| {
            try list.write(.{ .key = "loc", .value = try std.fmt.allocPrint(self.allocator, "{s}:{d}", .{ meta.source_file, meta.source_line }) });
            if (meta.op_name.len != 0) {
                try list.write(.{ .key = "op_name", .value = try std.fmt.allocPrint(self.allocator, "{s}:{s}", .{ meta.op_type, meta.op_name }) });
            }
            if (meta.deduplicated_name.len != 0) {
                try list.write(.{ .key = "deduplicated_name", .value = meta.deduplicated_name });
            }
            // stack_frame_id doesn't seem interesting
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
            return try writer.print("custom-call: {s}", .{instr.custom_call_target});
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

    return packed struct(std.meta.Int(.unsigned, bit_size)) {
        const Self = @This();

        mantissa: std.meta.Int(.unsigned, mantissa_bits),
        exponent: std.meta.Int(.unsigned, exponent_bits),
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
            .literal = .{ .shape = shape, .f32s = .{ .items = @constCast(&[_]f32{2.5}), .capacity = 1 } },
        },
        .{
            .id = 3,
            .name = "call\"\\\n",
            .opcode = "custom-call",
            .shape = shape,
            .operand_ids = .{ .items = @constCast(&[_]i64{ 1, 2 }), .capacity = 2 },
            .metadata = .{ .source_file = "a\"\\b.zig", .source_line = 42, .op_name = "op\nname", .op_type = "type" },
        },
    };
    var computations = [_]hlo.HloComputationProto{.{
        .id = 1,
        .name = "entry",
        .root_id = 3,
        .instructions = .{ .items = &instructions, .capacity = instructions.len },
    }};
    const proto: hlo.HloProto = .{ .hlo_module = .{
        .name = "module",
        .computations = .{ .items = &computations, .capacity = computations.len },
    } };
    var encoded: std.Io.Writer.Allocating = .init(allocator);
    try proto.encode(&encoded.writer, allocator);
    var reader: std.Io.Reader = .fixed(encoded.written());
    const decoded = try hlo.HloProto.decode(&reader, allocator);
    try std.testing.expectEqual(.F32, decoded.hlo_module.?.computations.items[0].instructions.items[0].shape.?.layout.?.physical_shape.?.element_type);

    var output: std.Io.Writer.Allocating = .init(allocator);
    var converter: Converter = .{ .allocator = allocator };
    try converter.emitModule(&output.writer, decoded);
    const json = try std.json.parseFromSlice(std.json.Value, allocator, output.written(), .{});
    const root = json.value.object;
    try std.testing.expectEqualStrings("module", root.get("label").?.string);
    const graph = root.get("graphs").?.array.items[0].object;
    try std.testing.expectEqualStrings("entry", graph.get("id").?.string);
    const nodes = graph.get("nodes").?.array.items;
    try std.testing.expectEqual(@as(usize, 3), nodes.len);
    try std.testing.expectEqualStrings("Parameter 0", nodes[0].object.get("label").?.string);
    try std.testing.expectEqualStrings("constant({ 2.5e0 }, F32)", nodes[1].object.get("label").?.string);
    try std.testing.expectEqualStrings("call\"\\\n", nodes[2].object.get("label").?.string);
    const attrs = nodes[2].object.get("attrs").?.array.items;
    try std.testing.expectEqualStrings("a\"\\b.zig:42", attrs[0].object.get("value").?.string);
    try std.testing.expectEqualStrings("type:op\nname", attrs[1].object.get("value").?.string);
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
    try std.testing.expectError(error.NoModule, converter.emitModule(&output.writer, .{}));
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
