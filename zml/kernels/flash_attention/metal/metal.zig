const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../../../zml.zig");

const log = std.log.scoped(.@"zml/attention/metal");

pub const Metadata = struct {
    num_tokens: zml.Tensor,

    pub fn init() Metadata {
        return .{ .num_tokens = .fromShape(.scalar(.u32)) };
    }

    pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(Metadata) {
        _ = self;
        return .{ .num_tokens = try zml.Buffer.scalar(io, platform, 0, .u32) };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
        self.num_tokens.deinit();
    }
};

pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor, metadata: Metadata) zml.Tensor {
    const qc = q.transpose(.{ .h, .q, .hd });
    const kc = k.transpose(.{ .h, .k, .hd });
    const vc = v.transpose(.{ .h, .k, .hd });
    const tok_i32 = token_index.convert(.i32);

    const attn = if (q.dim(.q) > 1)
        zml.ops.customCall("zml$flash_attn", .{ qc, kc, vc, tok_i32, metadata.num_tokens }, qc.shape(), .{}, .{ .has_side_effect = false })
    else
        zml.ops.customCall("zml$flash_attn", .{ qc, kc, vc, tok_i32 }, qc.shape(), .{}, .{ .has_side_effect = false });

    return attn.transpose(q.shape());
}
