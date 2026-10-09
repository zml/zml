const zml = @import("zml");

pub const CompileArgs = struct {
    // sharding: *const zml.Sharding,
    attention_metadata: zml.attention.Metadata,
    attention_parameters: zml.attention.Parameters,
};
