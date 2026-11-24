//! Chat and model code of the LLM example, for binaries that reuse it from
//! other packages (e.g. //examples/sharding:llm).

pub const chat = @import("chat.zig");
pub const models = @import("models.zig");
