const builtin = @import("builtin");
const std = @import("std");
const hlo = @import("@xla//xla/service:hlo_proto");
const log = std.log.scoped(.hlo2json);

const json_writer = @import("json_writer.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    var args = std.process.args();
    _ = args.next();

    const input_json_path = args.next() orelse "/Users/guw/Documents/monorepo/pretty_xla/rope.hlo.pb";
    const output_json_path = args.next() orelse try std.mem.joinZ(arena.allocator(), "", &.{ input_json_path, ".json" });

    const output_file = try std.fs.cwd().createFileZ(output_json_path, .{});
    defer output_file.close();
    var writer = std.io.bufferedWriter(output_file.writer());
    try convertFile(arena.allocator(), input_json_path, writer.writer());
    try writer.flush();
}

pub fn convertFile(allocator: std.mem.Allocator, input_hlo_path: []const u8, writer: anytype) !void {
    const input_file = try std.fs.cwd().openFile(input_hlo_path, .{});
    var reader = std.io.bufferedReader(input_file.reader());
    const file_size = (try input_file.stat()).size;

    const file_content = try std.heap.page_allocator.alloc(u8, file_size);
    _ = try reader.reader().readAll(file_content);
    defer std.heap.page_allocator.free(file_content);

    var proto_arena = std.heap.ArenaAllocator.init(allocator);
    defer proto_arena.deinit();

    const proto = try hlo.HloProto.decode(file_content, proto_arena.allocator());

    var converter: Converter = .{ .allocator = allocator };
    try converter.emitModule(writer, proto);
}

threadlocal var py_writer = std.ArrayList(u8).init(std.heap.page_allocator);

pub const Slice = extern struct { ptr: u64 = 0, len: u64 = 0 };

pub export fn convertHloToJson(input_hlo_path_ptr: [*]const u8, len: u32) Slice {
    const input_hlo_path = input_hlo_path_ptr[0..len];
    if (py_writer.capacity == 0) py_writer.ensureTotalCapacity(std.heap.page_size_min * 4) catch return .{};
    py_writer.shrinkRetainingCapacity(0);

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    convertFile(arena.allocator(), input_hlo_path, py_writer.writer()) catch |err| {
        log.err("Failed to read hlo file: {s} ({})", .{ input_hlo_path, err });
        return .{};
    };
    return .{ .ptr = @intFromPtr(py_writer.items.ptr), .len = py_writer.items.len };
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
    out_buf: std.BoundedArray(u8, 1024) = .{},
    namespaces: std.BoundedArray([]const u8, 128) = .{},

    allocator: std.mem.Allocator,
    // Map instruction id to its instruction.
    instructions: std.AutoArrayHashMapUnmanaged(i64, *const hlo.HloInstructionProto) = .{},
    // Map instruction id to its buffer when it has one.
    physical_buffer: std.AutoArrayHashMapUnmanaged(i64, Assigned) = .{},
    // Map computation id to its first caller instruction. This allow to have extra context about the computation.
    computation_caller: std.AutoArrayHashMapUnmanaged(i64, *const hlo.HloInstructionProto) = .{},
    computations: std.AutoArrayHashMapUnmanaged(i64, *const hlo.HloComputationProto) = .{},
    reduce_name: std.AutoArrayHashMapUnmanaged(i64, *const hlo.HloInstructionProto) = .{},

    const Assigned = struct { physical: i64, start: i64, len: i64 };

    pub fn emitModule(self: *Converter, writer: anytype, hlo_proto: hlo.HloProto) !void {
        const module = hlo_proto.hlo_module orelse return error.NoModule;

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
                if (std.mem.eql(u8, instr.opcode.getSlice(), "fusion") and instr.called_computation_ids.items.len > 0) {
                    const callee = instr.called_computation_ids.items[0];
                    _ = try self.computation_caller.getOrPutValue(self.allocator, callee, instr);
                }
            }
        }

        const W = @TypeOf(writer);
        var top_level = try json_writer.dict(writer);
        defer top_level.close();

        try top_level.write("label", module.name.getSlice());
        var graphs = try top_level.openList("graphs");
        defer graphs.close();

        for (module.computations.items) |computation| {
            var graph = try graphs.openDict();
            defer graph.close();
            try self.emitComputation(W, &graph, computation);
        }
    }

    pub fn emitComputation(self: *const Converter, Writer: type, graph: *json_writer.Dict(Writer), computation: hlo.HloComputationProto) !void {
        const comp_caller = self.computation_caller.get(computation.id);
        // The name of the caller node is better cause computation name is typically just "fusion".
        try graph.write("id", if (comp_caller) |caller| caller.name.getSlice() else computation.name.getSlice());

        var nodes = try graph.openList("nodes");
        defer nodes.close();
        for (computation.instructions.items) |instr| {
            if (std.mem.eql(u8, instr.opcode.getSlice(), "get-tuple-element")) continue;
            var node = try nodes.openDict();
            defer node.close();
            // node name + attrs
            try node.writeTemplate("id", "\"{d}\"", .{instr.id});
            try node.write("namespace", "");
            try node.writeTemplate("label", "\"{}\"", .{self.instrLabel(instr)});
            {
                var attrs = try node.openList("attrs");
                defer attrs.close();
                try self.writeAttr(Writer, &attrs, instr, comp_caller);
            }

            {
                var outputs = try node.openList("outputsMetadata");
                defer outputs.close();
                const out_shape = instr.shape.?.*;
                var out_shapes: []const @TypeOf(out_shape) = &.{out_shape};
                if (out_shapes[0].element_type == .TUPLE) {
                    out_shapes = out_shapes[0].tuple_shapes.items;
                }

                for (0.., out_shapes) |o, o_shape| {
                    try outputs.writeTemplate(
                        \\{{"id":"{d}","attrs":[{{"key":"tensor_shape","value":"{}"}}]}}
                    , .{ o, InstrShape{ .shape = &o_shape } });
                }
            }

            // node edges
            {
                var edges = try node.openList("incomingEdges");
                defer edges.close();
                for (instr.operand_ids.items, 0..) |input_id, i| {
                    const input_instr = self.instructions.get(input_id) orelse continue;

                    // The source node, and output_id to handle when input node produces several values.
                    var source_id: i64 = input_id;
                    var source_output_id: i64 = 0;
                    if (std.mem.eql(u8, input_instr.opcode.getSlice(), "get-tuple-element")) {
                        source_id = input_instr.operand_ids.items[0];
                        source_output_id = input_instr.tuple_index;
                    }
                    // edge_metadata could be added here if needed.
                    try edges.writeTemplate(
                        \\{{"sourceNodeId":"{d}","sourceNodeOutputId":"{d}", "targetNodeInputId":"{d}"}}
                    , .{ source_id, source_output_id, i });
                }
            }

            if (instr.called_computation_ids.items.len > 0) {
                var subpgraphs = try node.openList("subgraphIds");
                defer subpgraphs.close();
                for (instr.called_computation_ids.items) |called| {
                    if (self.computation_caller.get(called)) |meta| {
                        try subpgraphs.write(meta.name.getSlice());
                    } else {
                        const called_comp = self.computations.get(called).?;
                        try subpgraphs.write(called_comp.name.getSlice());
                    }
                }
            }
        }
    }

    pub fn writeAttr(self: *const Converter, Writer: type, list: *json_writer.List(Writer), instr: hlo.HloInstructionProto, comp_caller: ?*const hlo.HloInstructionProto) !void {
        if (instr.metadata) |meta| {
            try list.writeTemplate(
                \\{{"key":"loc","value":"{s}:{d}"}}
            , .{ meta.source_file.getSlice(), meta.source_line });
            if (!meta.op_name.isEmpty()) {
                try list.writeTemplate(
                    \\{{"key":"op_name","value":"{s}:{s}"}}
                , .{ meta.op_type.getSlice(), meta.op_name.getSlice() });
            }
            if (!meta.deduplicated_name.isEmpty()) {
                try list.writeTemplate(
                    \\{{"key":"deduplicated_name","value":"{s}"}}
                , .{meta.deduplicated_name.getSlice()});
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
            if (std.mem.eql(u8, instr.opcode.getSlice(), "parameter")) {
                const arg = caller.operand_ids.items[@intCast(instr.parameter_number)];
                if (self.instructions.get(arg)) |arg_instr| {
                    try list.writeTemplate(
                        \\{{"key":"aka","value":"{s}"}}
                    , .{self.instrLabel(arg_instr.*)});
                }
            }
        }
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
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        const instr = self.instr;
        const converter = self.converter;
        const opcode = instr.opcode.getSlice();
        if (std.mem.eql(u8, opcode, "parameter")) {
            return try writer.print("Parameter {d}", .{instr.parameter_number});
        }

        if (std.mem.eql(u8, opcode, "constant") and instr.literal != null) {
            return try renderLiteral(writer, instr.literal.?.*);
        }

        if (std.mem.eql(u8, opcode, "fusion")) {
            return try writer.print("{s}", .{instr.name.getSlice()});
        }

        if (std.mem.eql(u8, opcode, "custom-call")) {
            return try writer.print("{s}", .{instr.name.getSlice()});
        }

        if (std.mem.eql(u8, opcode, "reduce")) r: {
            const reducer = converter.computations.get(instr.called_computation_ids.items[0]).?;
            // Complex reduction don't give a name cause it's probably misleading.
            if (reducer.instructions.items.len > 3) break :r;
            const root = self.converter.instructions.get(reducer.root_id).?;
            const reduce_label = if (std.mem.eql(u8, root.opcode.getSlice(), "compare"))
                root.comparison_direction.getSlice()
            else
                root.opcode.getSlice();
            return try writer.print("{s}.{s}", .{ instr.opcode.getSlice(), reduce_label });
        }

        if (std.mem.eql(u8, opcode, "get-tuple-element")) {
            const tuple_instr = self.converter.instructions.get(instr.operand_ids.items[0]).?.*;
            return try writer.print("{s}[{d}]", .{ self.converter.instrLabel(tuple_instr), instr.tuple_index });
        }

        // The HLO instruction name contains usually the opcode, e.g. "%add.42" is
        // an add instruction.  In this case we render just the name.
        if (std.mem.startsWith(u8, instr.name.getSlice(), opcode)) {
            var name = instr.name.getSlice();
            if (name.len > opcode.len) {
                if (std.fmt.parseInt(u64, name[opcode.len + 1 ..], 10)) |_| {
                    name = opcode;
                } else |_| {}
            }
            return try writer.writeAll(name);
        }

        return try writer.print("{s}.{s}", .{ instr.name.getSlice(), instr.opcode.getSlice() });
    }
};

const InstrShape = struct {
    shape: std.meta.FieldType(hlo.HloInstructionProto, .shape),

    pub fn format(
        self: InstrShape,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        // Use syntax supported by the frontend. The "tensor<>" won't be displayed, only what's inside.
        try writer.writeAll("tensor<");
        const dims = self.shape.?.dimensions.items;
        for (dims) |d| {
            try writer.print("{d}x", .{d});
        }
        try writer.print("{s}>", .{@tagName(self.shape.?.element_type)});
    }
};

const LiteralProto = @TypeOf(hlo.HloInstructionProto.init().literal.?.*);

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
        .S8 => try writer.print("{d}", .{std.mem.bytesAsSlice(i8, literal.s8s.getSlice())[0..n]}),
        .S16 => try writer.print("{d}", .{std.mem.bytesAsSlice(i16, literal.s16s.getSlice())[0..n]}),
        .S32 => try writer.print("{d}", .{literal.s32s.items[0..n]}),
        .S64 => try writer.print("{d}", .{literal.s64s.items[0..n]}),
        // .U1 => ,
        // .U2 => ,
        // .U4 => ,
        .U8 => try writer.print("{d}", .{literal.u8s.getSlice()[0..n]}),
        .U16 => try writer.print("{d}", .{std.mem.bytesAsSlice(u16, literal.u16s.getSlice())[0..n]}),
        .U32 => try writer.print("{d}", .{literal.u32s.items[0..n]}),
        .U64 => try writer.print("{d}", .{literal.u64s.items[0..n]}),
        .F32 => try writer.print("{e}", .{literal.f32s.items[0..n]}),
        .F64 => try writer.print("{e}", .{literal.f64s.items[0..n]}),
        .F16 => try writer.print("{e}", .{std.mem.bytesAsSlice(f16, literal.f16s.getSlice())[0..n]}),
        .BF16 => try writer.print("{e}", .{std.mem.bytesAsSlice(BFloat16, literal.bf16s.getSlice())[0..n]}),
        .F8E5M2 => try writer.print("{e}", .{std.mem.bytesAsSlice(F8E5M2, literal.f8e5m2s.getSlice())[0..n]}),
        .F8E5M2FNUZ => try writer.print("{e}", .{std.mem.bytesAsSlice(F8E5M2, literal.f8e5m2fnuzs.getSlice())[0..n]}),
        .F8E3M4 => try writer.print("{e}", .{std.mem.bytesAsSlice(F8E3M4, literal.f8e3m4s.getSlice())[0..n]}),
        .F8E4M3 => try writer.print("{e}", .{std.mem.bytesAsSlice(F8E4M3, literal.f8e4m3s.getSlice())[0..n]}),
        .F8E4M3FN => try writer.print("{e}", .{std.mem.bytesAsSlice(F8E4M3, literal.f8e4m3fns.getSlice())[0..n]}),
        .F8E4M3FNUZ => try writer.print("{e}", .{std.mem.bytesAsSlice(F8E4M3, literal.f8e4m3fnuzs.getSlice())[0..n]}),
        .F8E4M3B11FNUZ => try writer.print("{e}", .{std.mem.bytesAsSlice(F8E4M3, literal.f8e4m3b11fnuzs.getSlice())[0..n]}),
        // C64 = 15;
        // C128 = 18;
        else => _ = try writer.writeAll("???"),
    }
    try writer.print(", {s})", .{@tagName(dtype)});
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

        fn f32Mantissa(self: Self) std.meta.FieldType(Float32, .mantissa) {
            @setRuntimeSafety(false);
            const f32_mantissa_bits = @bitSizeOf(std.meta.FieldType(Float32, .mantissa));

            const Res = std.meta.FieldType(Float32, .mantissa);
            return @shlExact(@as(Res, self.mantissa), f32_mantissa_bits - mantissa_bits);
        }

        fn expBias() u8 {
            if (exponent_bits == 1) return 0;
            return (1 << (exponent_bits - 1)) - 1;
        }

        pub fn format(
            self: @This(),
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = options;
            try writer.print("{" ++ fmt ++ "}", .{self.toF32()});
        }
    };
}
const BFloat16 = FloatType(8, 7);
const Float32 = FloatType(8, 23);
const F8E5M2 = FloatType(5, 2);
const F8E4M3 = FloatType(4, 3);
const F8E3M4 = FloatType(3, 4);
