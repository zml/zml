const config = @import("config.zig");

pub const fields = config.fields;
pub const build_components = config.build_components;
pub const get_components = config.get_components;

pub const tables: []const config.Table = &.{
    .{
        .fields = &.{
            "east_asian_width",
            "grapheme_break",
            "general_category",
            "is_emoji_presentation",
        },
    },
};
