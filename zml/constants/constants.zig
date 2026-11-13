pub const MAX_RANK: u8 = 8;
pub const AXES_IOTA: [MAX_RANK]i64 = b: {
    var values: [MAX_RANK]i64 = undefined;
    for (0..MAX_RANK) |i| values[i] = i;
    break :b values;
};
pub const MINOR_TO_MAJOR: [MAX_RANK]i64 = b: {
    var values: [MAX_RANK]i64 = undefined;
    for (0..MAX_RANK) |i| values[i] = MAX_RANK - i - 1;
    break :b values;
};

pub fn minorToMajor(rank: u8) []const i64 {
    return MINOR_TO_MAJOR[MINOR_TO_MAJOR.len - rank ..];
}

pub const ones_i64: [MAX_RANK]i64 = @splat(1);
pub const zeroes_i64: [MAX_RANK]i64 = @splat(0);
