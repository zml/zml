pub const MAX_RANK: u8 = 8;
pub const AXES_IOTA: [MAX_RANK]i64 = b: {
    var values: [MAX_RANK]i64 = undefined;
    for (0..MAX_RANK) |i| values[i] = i;
    break :b values;
};
