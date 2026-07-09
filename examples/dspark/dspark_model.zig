const zml = @import("zml");

pub const Config = struct {
    vocab_size: u32,
    hidden_size: u32,
};

pub const Buffers = zml.Bufferized(Model);

pub const Model = struct {};
