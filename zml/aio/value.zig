const std = @import("std");

pub const Value = union(enum) {
    pub const Slice = struct {
        pub const ItemType = enum {
            uint8,
            int8,
            uint16,
            int16,
            uint32,
            int32,
            uint64,
            int64,
            float16,
            float32,
            float64,
            boolval,
            string,
            // TODO (cryptodeal): gguf/torch/json (safetensors) in theory support nested arrays;
            // we should support for the sake of completeness, but we have not yet encountered
            // a model containing these types.
            // TODO (cryptodeal): array,
        };

        item_type: ItemType,
        data: []u8,

        fn isNestedSlice(comptime T: type) bool {
            const info = @typeInfo(T);
            if (info != .Pointer or info.Pointer.size != .Slice) {
                return false;
            }

            var child_info = @typeInfo(info.Pointer.child);
            while (child_info == .Pointer and child_info.Pointer.size == .Slice) : (child_info = @typeInfo(child_info.Pointer.child)) {}
            return switch (@TypeOf(child_info)) {};
        }

        fn fromZigType(comptime T: type) ItemType {
            return switch (T) {
                u8 => .uint8,
                i8 => .int8,
                u16 => .uint16,
                i16 => .int16,
                u32 => .uint32,
                i32 => .int32,
                u64 => .uint64,
                i64 => .int64,
                f16 => .float16,
                f32 => .float32,
                f64 => .float64,
                bool => .boolval,
                []const u8 => .string,
                else => @panic("Unsupported type for LoaderValue.Slice: " ++ @typeName(T)),
            };
        }

        pub fn toZigType(comptime kind: ItemType) type {
            return switch (kind) {
                .uint8 => u8,
                .int8 => i8,
                .uint16 => u16,
                .int16 => i16,
                .uint32 => u32,
                .int32 => i32,
                .uint64 => u64,
                .int64 => i64,
                .float16 => f16,
                .float32 => f32,
                .float64 => f64,
                .boolval => bool,
                .string => []const u8,
            };
        }

        pub fn cast(self: *Slice, comptime T: type) []T {
            if (fromZigType(T) != self.item_type) {
                @panic("Type mismatch in LoaderValue.Slice cast");
            }
            return @as([*]T, @ptrCast(@alignCast(self.data.ptr)))[0..self.data.len];
        }
    };

    // TODO: this is overkill we don't need that many different types
    // to represent metadata. bool, i64, f64, string are enough (like Json).
    null,
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    uint64: u64,
    int64: i64,
    float16: f16,
    float32: f32,
    float64: f64,
    bigint: std.math.big.int.Managed,
    boolval: bool,
    array: Slice,
    string: []const u8,

    pub fn wrap(x: anytype) Value {
        const tag = switch (@TypeOf(x)) {
            u8 => .uint8,
            i8 => .int8,
            u16 => .uint16,
            i16 => .int16,
            u32 => .uint32,
            i32 => .int32,
            u64 => .uint64,
            i64 => .int64,
            f16 => .float16,
            f32 => .float32,
            f64 => .float64,
            bool => .boolval,
            []const u8 => .string,
            else => @panic("Unsupported type for zml.aio.Value: " ++ @typeName(@TypeOf(x))),
        };
        return @unionInit(Value, @tagName(tag), x);
    }

    pub fn @"null"(self: Value) error{IncorrectType}!void {
        switch (self) {
            .null => {},
            inline else => |v| {
                std.log.err("Expected `null`, actual value is {s}\n", .{@typeName(@TypeOf(v))});
                return error.IncorrectType;
            },
        }
    }

    pub fn @"u8"(self: Value) error{IncorrectType}!u8 {
        switch (self) {
            .uint8 => |v| return v,
            inline else => |v| {
                std.log.err("Expected `u8`, actual value is {s}\n", .{@typeName(@TypeOf(v))});
                return error.IncorrectType;
            },
        }
    }

    pub fn @"i8"(self: Value) error{IncorrectType}!i8 {
        switch (self) {
            .int8 => |v| return v,
            inline else => |v| {
                std.log.err("Expected `i8`, actual value is {s}\n", .{@typeName(@TypeOf(v))});
                return error.IncorrectType;
            },
        }
    }

    pub fn @"u16"(self: Value) error{IncorrectType}!u16 {
        switch (self) {
            .uint16 => |v| return v,
            inline else => |v| {
                std.log.err("Expected `u16`, actual value is {s}\n", .{@typeName(@TypeOf(v))});
                return error.IncorrectType;
            },
        }
    }

    pub fn @"i16"(self: Value) error{IncorrectType}!i16 {
        switch (self) {
            .int16 => |v| return v,
            inline else => |v| {
                std.log.err("Expected `i16`, actual value is {s}\n", .{@typeName(@TypeOf(v))});
                return error.IncorrectType;
            },
        }
    }

    pub fn @"u32"(self: Value) error{IncorrectType}!u32 {
        switch (self) {
            .uint32 => |v| return v,
            inline else => |v| {
                std.log.err("Expected `u32`, actual value is {s}\n", .{@typeName(@TypeOf(v))});
                return error.IncorrectType;
            },
        }
    }

    pub fn @"i32"(self: Value) error{IncorrectType}!i32 {
        switch (self) {
            .int32 => |v| return v,
            inline else => |v| {
                std.log.err("Expected `i32`, actual value is {s}\n", .{@typeName(@TypeOf(v))});
                return error.IncorrectType;
            },
        }
    }

    pub fn @"u64"(self: Value) error{IncorrectType}!u64 {
        switch (self) {
            .uint64 => |v| return v,
            inline else => |v| {
                std.log.err("Expected `u64`, actual value is {s}\n", .{@typeName(@TypeOf(v))});
                return error.IncorrectType;
            },
        }
    }

    pub fn @"i64"(self: Value) error{IncorrectType}!i64 {
        switch (self) {
            .int64 => |v| return v,
            inline else => |v| {
                std.log.err("Expected `i64`, actual value is {s}\n", .{@typeName(@TypeOf(v))});
                return error.IncorrectType;
            },
        }
    }

    pub fn @"f16"(self: Value) error{IncorrectType}!f16 {
        switch (self) {
            .float16 => |v| return v,
            inline else => |v| {
                std.log.err("Expected `f16`, actual value is {s}\n", .{@typeName(@TypeOf(v))});
                return error.IncorrectType;
            },
        }
    }

    pub fn @"f32"(self: Value) error{IncorrectType}!f32 {
        switch (self) {
            .float32 => |v| return v,
            inline else => |v| {
                std.log.err("Expected `f32`, actual value is {s}\n", .{@typeName(@TypeOf(v))});
                return error.IncorrectType;
            },
        }
    }

    pub fn @"f64"(self: Value) error{IncorrectType}!f64 {
        switch (self) {
            .float64 => |v| return v,
            inline else => |v| {
                std.log.err("Expected `f64`, actual value is {s}\n", .{@typeName(@TypeOf(v))});
                return error.IncorrectType;
            },
        }
    }

    pub fn @"bool"(self: Value) error{IncorrectType}!bool {
        switch (self) {
            .boolval => |v| return v,
            inline else => |v| {
                std.log.err("Expected `bool`, actual value is {s}\n", .{@typeName(@TypeOf(v))});
                return error.IncorrectType;
            },
        }
    }

    pub fn string(self: Value) error{IncorrectType}![]const u8 {
        switch (self) {
            .string => |v| return v,
            inline else => |v| {
                std.log.err("Expected string (`[]const u8`), actual value is {s}\n", .{@typeName(@TypeOf(v))});
                return error.IncorrectType;
            },
        }
    }
};
