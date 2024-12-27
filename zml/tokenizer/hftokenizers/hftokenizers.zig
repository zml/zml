const std = @import("std");
const c = @import("c");
const ffi = @import("ffi");

pub const Encoder = struct {
    inner: *HFTokenizer,
    current_ids: ?[]const u32 = null,

    fn init(inner: *HFTokenizer) Encoder {
        return .{ .inner = inner };
    }

    pub fn reset(self: *Encoder) void {
        if (self.current_ids) |current_ids_| {
            c.hftokenizers_tokens_drop(ffi.ZigSlice.from(current_ids_));
            self.current_ids = null;
        }
    }

    pub fn deinit(self: *Encoder) void {
        self.reset();
    }

    pub fn encode(self: *Encoder, input: []const u8) ![]const u32 {
        self.reset();
        self.current_ids = ffi.ZigSlice.to(u32, c.hftokenizers_encode(@ptrCast(self.inner), ffi.ZigSlice.from(input)));
        return self.ids();
    }

    pub fn ids(self: *const Encoder) []const u32 {
        return self.current_ids orelse &.{};
    }
};

pub const Decoder = struct {
    inner: *HFTokenizer,
    current_string: ?[]const u8 = null,

    fn init(inner: *HFTokenizer) Decoder {
        return .{ .inner = inner };
    }

    pub fn deinit(self: *Decoder) void {
        self.reset();
    }

    pub fn reset(self: *Decoder) void {
        if (self.current_string) |current_string_| {
            c.hftokenizers_str_drop(ffi.ZigSlice.from(current_string_));
            self.current_string = null;
        }
    }

    pub fn decode(self: *Decoder, ids: []const u32) ![]const u8 {
        self.reset();
        self.current_string = ffi.ZigSlice.to(u8, c.hftokenizers_decode(@ptrCast(self.inner), ffi.ZigSlice.from(ids)));
        return self.string();
    }

    pub fn string(self: *const Decoder) []const u8 {
        return self.current_string orelse &.{};
    }
};

pub const HFTokenizer = opaque {
    pub fn from_file(model: []const u8) !*HFTokenizer {
        return @ptrCast(c.hftokenizers_new(ffi.ZigSlice.from(model)));
    }

    pub fn deinit(self: *HFTokenizer) void {
        return c.hftokenizers_drop(self);
    }

    pub fn encoder(self: *HFTokenizer) !Encoder {
        return Encoder.init(self);
    }

    pub fn decoder(self: *HFTokenizer) !Decoder {
        return Decoder.init(self);
    }
};
