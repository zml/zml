const config = @import("config.zig");
const config_x = @import("config.x.zig");

pub const tables = [_]config.Table{
    .{
        .extensions = &.{},
        .fields = &config._resolveFields(
            config_x,
            &.{
                "east_asian_width",
                "grapheme_break",
                "general_category",
                "is_emoji_presentation",
            },
            &.{},
        ),
    },
};
