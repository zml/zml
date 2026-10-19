const zml = @import("zml");

const QwenImage21ZeroCenterRMSNorm = @This();

// Without this field, the LSP doesn't understand that the struct is a struct
_phantom: struct {},

pub fn init() QwenImage21ZeroCenterRMSNorm {
    return QwenImage21ZeroCenterRMSNorm{ ._phantom = .{} };
}
