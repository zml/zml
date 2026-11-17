const vaxis = @import("vaxis");

const Color = vaxis.Cell.Color;
const Style = vaxis.Cell.Style;

/// Returns a color on a green → yellow → red gradient based on percentage (0-100).
pub fn colorForPercent(pct: u8) Color {
    if (pct <= 50) {
        // Green (#00cc66) → Yellow (#ffcc00)
        const t: u16 = @as(u16, pct) * 255 / 50;
        return .{ .rgb = .{ @intCast(t), 204, @intCast(102 -| (t * 102 / 255)) } };
    } else if (pct <= 80) {
        // Yellow (#ffcc00) → Orange (#ff6600)
        const t: u16 = (@as(u16, pct) - 50) * 255 / 30;
        return .{ .rgb = .{ 255, @intCast(204 -| (t * 102 / 255)), 0 } };
    } else {
        // Orange (#ff6600) → Red (#ff2200)
        const t: u16 = (@as(u16, pct) - 80) * 255 / 20;
        return .{ .rgb = .{ 255, @intCast(102 -| (t * 68 / 255)), 0 } };
    }
}

// ── Style Constants ──────────────────────────────────────────────

pub const accent: Color = .{ .rgb = .{ 120, 180, 255 } };
pub const dim: Color = .{ .rgb = .{ 100, 100, 120 } };
pub const text_primary: Color = .{ .rgb = .{ 230, 230, 240 } };
pub const text_secondary: Color = .{ .rgb = .{ 160, 160, 175 } };
pub const gauge_empty: Color = .{ .rgb = .{ 50, 50, 65 } };

pub const header_style: Style = .{ .bold = true, .fg = accent };
pub const title_style: Style = .{ .bold = true, .fg = .{ .rgb = .{ 200, 160, 255 } } };
pub const label_style: Style = .{ .fg = text_secondary };
pub const value_style: Style = .{ .bold = true, .fg = text_primary };
pub const border_style: Style = .{ .fg = .{ .rgb = .{ 70, 70, 90 } } };
pub const dim_style: Style = .{ .fg = dim };
