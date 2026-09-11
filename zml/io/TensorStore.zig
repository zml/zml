//! Associates model tensors with checkpoint sources and prefixed views.
const std = @import("std");
const stdx = @import("stdx");

const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const Tensor = @import("../tensor.zig").Tensor;

const TensorStore = @This();

pub const Binding = struct {
    tensors: []*safetensors.Tensor,
    transformed: bool,
};

registry: *safetensors.TensorRegistry,
id_to_sources: std.AutoHashMapUnmanaged(Tensor.Id, Binding),
allocator: std.mem.Allocator,
arena: std.heap.ArenaAllocator,

pub fn fromRegistry(allocator: std.mem.Allocator, registry: *safetensors.TensorRegistry) TensorStore {
    const arena: std.heap.ArenaAllocator = .init(allocator);
    return .{
        .registry = registry,
        .id_to_sources = .empty,
        .allocator = allocator,
        .arena = arena,
    };
}

pub fn deinit(self: *TensorStore) void {
    self.id_to_sources.deinit(self.allocator);
    self.arena.deinit();
}

pub fn getReader(self: *const TensorStore, key: []const u8, io: std.Io, buffer: []u8) !safetensors.TensorReader {
    return self.registry.reader(io, key, buffer);
}

pub fn getSourcesById(self: *const TensorStore, id: Tensor.Id) ?Binding {
    return self.id_to_sources.get(id);
}

pub fn getShape(self: *const TensorStore, key: []const u8) ?Shape {
    const entry_ptr = self.getPtrFromKey(key) orelse return null;
    return entry_ptr.shape;
}

pub fn view(self: *TensorStore) View {
    return .{ .store = self };
}

pub const View = struct {
    store: *TensorStore,

    prefix_buffer: [256]u8 = undefined,
    prefix_length: usize = 0,

    pub fn root(self: *const View) View {
        return .{
            .store = self.store,
        };
    }

    pub fn withPrefix(self: *const View, prefix_: []const u8) View {
        var buffer: [256]u8 = undefined;
        const new_prefix = makeKey(&buffer, "{s}{s}.", .{ self.prefix() orelse "", prefix_ });

        return .{
            .store = self.store,
            .prefix_buffer = buffer,
            .prefix_length = new_prefix.len,
        };
    }

    pub fn withLayer(self: *const View, index: usize) View {
        var buffer: [256]u8 = undefined;
        const new_prefix = makeKey(&buffer, "{s}{d}.", .{ self.prefix() orelse "", index });

        return .{
            .store = self.store,
            .prefix_buffer = buffer,
            .prefix_length = new_prefix.len,
        };
    }

    pub fn prefix(self: *const View) ?[]const u8 {
        return if (self.prefix_length == 0) null else self.prefix_buffer[0..self.prefix_length];
    }

    pub fn hasKey(self: *const View, subkey: []const u8) bool {
        var buffer: [256]u8 = undefined;
        const key = makeKey(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
        return for (self.store.registry.tensors.keys()) |k| {
            if (std.mem.startsWith(u8, k, key)) break true;
        } else false;
    }

    pub fn maybeCreateTensor(self: View, subkey: []const u8, tagz: anytype, partitioning: anytype) ?Tensor {
        var buffer: [256]u8 = undefined;
        const key = makeKey(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
        const source = self.store.dupeSource(key) orelse return null;

        const sources = self.store.arena.allocator().alloc(*safetensors.Tensor, 1) catch |e| std.debug.panic("Not handling {} errors", .{e});
        errdefer self.store.arena.allocator().free(sources);
        sources[0] = source;

        var shape = source.shape;
        shape = applyTags(shape, tagz);
        shape = applyPartitioning(shape, partitioning);

        const tensor: Tensor = .fromShape(shape);
        self.store.putSourcesNoClobber(tensor.id, .{ .tensors = sources, .transformed = false }) catch |e| std.debug.panic("Not handling {} errors", .{e});

        return tensor;
    }

    pub fn createTensor(self: View, subkey: []const u8, tagz: anytype, partitioning: anytype) Tensor {
        return self.maybeCreateTensor(subkey, tagz, partitioning) orelse
            stdx.debug.panic("Checkpoint has no tensor named {s}{s}", .{ self.prefix() orelse "", subkey });
    }

    pub fn maybeCreateBinding(self: View, sources: []const []const u8, shape: Shape) ?Tensor {
        const arena = self.store.arena.allocator();

        var tensor_list = std.ArrayList(*safetensors.Tensor).initCapacity(arena, sources.len) catch |e| std.debug.panic("Not handling {} errors", .{e});
        defer tensor_list.deinit(arena);

        var buffer: [256]u8 = undefined;
        for (sources) |subkey| {
            const key = makeKey(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
            const tensor = self.store.dupeSource(key) orelse return null;
            tensor_list.appendAssumeCapacity(tensor);
        }

        const tensors = tensor_list.toOwnedSlice(arena) catch unreachable;
        errdefer arena.free(tensors);

        const tensor: Tensor = .fromShape(shape);
        self.store.putSourcesNoClobber(tensor.id, .{ .tensors = tensors, .transformed = true }) catch |e| std.debug.panic("Not handling {} errors", .{e});

        return tensor;
    }

    pub fn getShape(self: View, subkey: []const u8) ?Shape {
        var buffer: [256]u8 = undefined;
        const key = makeKey(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
        return self.store.getShape(key);
    }

    pub fn getReader(self: View, subkey: []const u8, io: std.Io, buffer: []u8) !safetensors.TensorReader {
        var key_buffer: [256]u8 = undefined;
        const key = makeKey(&key_buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
        return self.store.getReader(key, io, buffer);
    }

    pub fn count(self: View) usize {
        var count_: usize = 0;
        const prefix_ = self.prefix() orelse "";
        var it = self.store.registry.tensors.iterator();
        while (it.next()) |item| {
            const key = item.key_ptr.*;
            if (std.mem.startsWith(u8, key, prefix_)) {
                count_ += 1;
            }
        }
        return count_;
    }

    fn applyTags(shape_: Shape, tagz: anytype) Shape {
        var shape = shape_;
        if (@TypeOf(tagz) != @TypeOf(null)) {
            switch (@typeInfo(@TypeOf(tagz))) {
                .optional => if (tagz) |t| {
                    shape = shape.withTags(t);
                },
                else => shape = shape.withTags(tagz),
            }
        }
        return shape;
    }

    fn applyPartitioning(shape_: Shape, partitioning: anytype) Shape {
        var shape = shape_;

        if (@TypeOf(partitioning) == @TypeOf(null)) {
            @compileError("TensorStore.View.createTensor partitioning cannot be null; pass .replicated or an explicit partitioning");
        }

        switch (@typeInfo(@TypeOf(partitioning))) {
            .optional => @compileError("TensorStore.View.createTensor partitioning cannot be optional; pass .replicated or an explicit partitioning"),
            .enum_literal => switch (partitioning) {
                .replicated => shape = shape.withReplicatedPartitioning(),
                else => @compileError("Only .replicated is supported as a standalone partitioning enum literal"),
            },
            else => shape = shape.withPartitioning(partitioning),
        }

        return shape;
    }

    fn makeKey(buffer: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
        const key = std.fmt.bufPrint(buffer, fmt, args) catch
            std.debug.panic("Expected key to be less than {} characters", .{buffer.len});
        return key;
    }
};

fn putSourcesNoClobber(self: *TensorStore, id: Tensor.Id, sources: Binding) std.mem.Allocator.Error!void {
    const gop = try self.id_to_sources.getOrPut(self.allocator, id);
    if (gop.found_existing) {
        stdx.debug.panic("Id {} already has associated sources", .{id});
    }
    errdefer self.id_to_sources.removeByPtr(gop.key_ptr);

    gop.value_ptr.* = sources;
}

fn getPtrFromKey(self: *const TensorStore, key: []const u8) ?*safetensors.Tensor {
    const tensor_desc_ptr = self.registry.tensors.getPtr(key) orelse return null;
    return tensor_desc_ptr;
}

fn dupeSource(self: *TensorStore, key: []const u8) ?*safetensors.Tensor {
    const entry = self.getPtrFromKey(key) orelse return null;

    const copy = self.arena.allocator().create(safetensors.Tensor) catch @panic("OOM");
    copy.* = entry.*;

    return copy;
}
