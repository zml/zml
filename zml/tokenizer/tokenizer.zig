pub const Tokenizer = struct {
    pub const Decoder = struct {
        pub fn step(self: *Decoder, token_id: u32) []const u8 {
            return input;
        }

        pub fn reset(self: *Decoder) void {
            _ = self; // autofix
        }
    };

    pub fn init() Tokenizer {
        return Tokenizer{};
    }

    pub fn encode(self: *Tokenizer, input: []const u8) []const u32 {
        _ = input; // autofix
        _ = self; // autofix
        return &.{};
    }

    pub fn decode(self: *Tokenizer, input: []const u32) []const u8 {
        _ = input; // autofix
        _ = self; // autofix
        return &.{};
    }
};
