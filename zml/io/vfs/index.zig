const log = std.log.scoped(.@"zml/io/vfs/vfs");

const std = @import("std");

pub const File = @import("file.zig").File;
pub const GCS = @import("gcs.zig").GCS;
pub const HF = @import("hf.zig").HF;
pub const HTTP = @import("http.zig").HTTP;
pub const S3 = @import("s3.zig").S3;
pub const VFSBase = @import("base.zig").VFSBase;
pub const Backend = @import("base.zig").Backend;
pub const ReadHints = @import("base.zig").ReadHints;
pub const ReadTimingBucket = @import("base.zig").ReadTimingBucket;
pub const ReadStats = @import("base.zig").ReadStats;
pub const ReadStatsProvider = @import("base.zig").ReadStatsProvider;
pub const read_timing_bucket_sizes = @import("base.zig").read_timing_bucket_sizes;

test {
    _ = @import("http_acceptance_test.zig");
}
