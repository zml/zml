const std = @import("std");

// KIMI_K3_TEMP_REMOVE_M20: compile-time placeholder keeps model registration
// honest while inference operators are introduced; replace it before cleanup.
pub const CompiledModel = struct {
    pub fn deinit(self: *CompiledModel) void {
        _ = self;
    }
};

pub const CompilationParameters = struct {};
pub const CompilationOptions = CompilationParameters;
