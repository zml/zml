const std = @import("std");
const zml = @import("zml");
const model = @import("model.zig");

const WeightRef = struct {
    group: usize,
    index: usize,
};

const Layout = zml.meta.MapType(zml.Tensor, WeightRef).map(model.Model);

pub const Weights = struct {
    tensors: []zml.Tensor,
    layout: Layout,

    pub fn unpack(self: Weights) model.Model {
        const ctx = zml.Compiler.current();
        var result: model.Model = undefined;
        zml.meta.mapAlloc(struct {
            fn call(weights: []const zml.Tensor, ref: WeightRef) zml.Tensor {
                return weights[ref.group].slice(0, .single(@intCast(ref.index)));
            }
        }.call, ctx.arena.allocator(), self.tensors, self.layout, &result) catch ctx.abortOOM();
        return result;
    }
};

pub const Buffers = zml.Bufferized(Weights);

// Group checkpoint tensors into contiguous, equally shaped rows. Each row is
// recovered by a constant slice inside the single compiled forward pass.
// Packing happens on upload, never in the token loop.
pub const Plan = struct {
    arena: std.heap.ArenaAllocator,
    weights: Weights,
    groups: []const Group,

    const Group = struct {
        shape: zml.Shape,
        sources: std.ArrayList(zml.Tensor.Id) = .empty,
    };

    pub fn init(allocator: std.mem.Allocator, mdl: model.Model) !Plan {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        errdefer arena.deinit();
        const alloc = arena.allocator();
        const count = zml.meta.count(zml.Tensor, &mdl);
        const tensors = try alloc.alloc(zml.Tensor, count);
        zml.meta.forEachVisit(&mdl, *const zml.Tensor, struct {
            fn call(i: usize, tensor: *const zml.Tensor, out: []zml.Tensor) void {
                out[i] = tensor.*;
            }
        }.call, .{tensors});

        var groups: std.ArrayList(Group) = .empty;
        const refs = try alloc.alloc(WeightRef, count);
        for (tensors, refs) |tensor, *ref| {
            const shape = tensor.shape();
            const group_index = for (groups.items, 0..) |group, i| {
                if (!group.shape.eqlWithTags(shape)) continue;
                var same_partitioning = true;
                for (0..shape.rank()) |axis| {
                    if (!group.shape.partition(axis).eql(shape.partition(axis))) same_partitioning = false;
                }
                if (same_partitioning and (group.sources.items.len + 1) * shape.byteSize() <= 512 * zml.MiB) break i;
            } else blk: {
                try groups.append(alloc, .{ .shape = shape });
                break :blk groups.items.len - 1;
            };
            ref.* = .{ .group = group_index, .index = groups.items[group_index].sources.items.len };
            try groups.items[group_index].sources.append(alloc, tensor.id);
        }
        const packed_tensors = try alloc.alloc(zml.Tensor, groups.items.len);
        for (groups.items, packed_tensors) |group, *tensor| {
            var shape = group.shape.insertTag(0, @intCast(group.sources.items.len), .weight);
            // withPartitioning resets unspecified axes; preserve every original
            // weight axis while making only the new grouping axis replicated.
            shape._partitioning.set(0, .replicated);
            tensor.* = .fromShape(shape);
        }
        const Mapper = struct {
            refs: []const WeightRef,
            index: usize = 0,
            fn call(self: *@This(), _: zml.Tensor) WeightRef {
                defer self.index += 1;
                return self.refs[self.index];
            }
        };
        var mapper: Mapper = .{ .refs = refs };
        var layout: Layout = undefined;
        try zml.meta.mapAlloc(Mapper.call, alloc, &mapper, mdl, &layout);
        std.debug.assert(mapper.index == refs.len);
        return .{ .arena = arena, .weights = .{ .tensors = packed_tensors, .layout = layout }, .groups = try groups.toOwnedSlice(alloc) };
    }

    pub fn deinit(self: *Plan) void {
        self.arena.deinit();
    }

    pub fn load(self: *const Plan, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, progress: *std.Progress.Node, shardings: []const zml.Sharding) !Buffers {
        const start: std.Io.Timestamp = .now(io, .awake);
        var total_bytes: usize = 0;
        const buffers = try allocator.alloc(zml.Buffer, self.groups.len);
        errdefer allocator.free(buffers);
        var loaded: usize = 0;
        errdefer for (buffers[0..loaded]) |*buffer| buffer.deinit();
        var node = progress.start("Uploading packed Llama weights", self.groups.len);
        defer node.end();
        for (self.groups, self.weights.tensors, buffers) |group, tensor, *buffer| {
            const sharding = zml.Sharding.pickSharding(shardings, tensor.shape(), .explicit_axis_binding) orelse platform.replicated_sharding;
            var writer = try zml.io.BufferedMemoryWriter.init(allocator, io, platform, tensor.shape(), sharding, buffer);
            defer writer.deinit(allocator);
            for (group.sources.items) |id| {
                const binding = store.getSourcesById(id) orelse return error.MissingWeight;
                if (binding.transformed or binding.tensors.len != 1) return error.UnsupportedWeightTransform;
                var reader = try binding.tensors[0].reader(io, &.{}, .{});
                defer reader.deinit();
                const bytes = try reader.interface.streamRemaining(&writer.interface);
                if (bytes != group.shape.byteSize()) return error.WeightSizeMismatch;
            }
            try writer.interface.flush();
            loaded += 1;
            total_bytes += tensor.shape().byteSize();
            node.completeOne();
        }
        std.log.info("Loaded packed Llama weights [{Bi:.2}, {} buffers, {f}]", .{ total_bytes, buffers.len, start.untilNow(io, .awake) });
        return .{ .tensors = buffers };
    }

    pub fn unload(buffers: *Buffers, allocator: std.mem.Allocator) void {
        for (buffers.tensors) |*buffer| buffer.deinit();
        allocator.free(buffers.tensors);
    }
};
