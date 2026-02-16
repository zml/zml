const c_interface = @import("c");

pub const enable_stage_debug = @hasDecl(c_interface, "ZML_ENABLE_STAGE_DEBUG");
