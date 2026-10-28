//! Checkpoint loading and virtual filesystem integration. The user guide is
//! docs/learn/loader.md; each file's header holds its internals:
//!
//! - io/loader.zig: options and their checks, the set-up task, starts,
//!   program compiles and groups, admission, and the one check of every
//!   binding before the backends see it (`LoadSpec`).
//! - io/backend.zig: backend selection and the `LoadSpec` contract.
//! - io/TensorStore.zig, io/Source.zig: bindings, views, and the sources
//!   that describe a tensor's checkpoint bytes.
//! - io/direct_loader.zig: sizing, per-file planning, FIFO claims, source
//!   workers, per-device pumps and transfer completion.
//! - io/ShardGrid.zig: where a destination's logical bytes land.
//! - io/buffered_loader.zig: whole-tensor staging and positional reads.
//! - io/dma_calibration.zig: DMA block selection.
//! - io/host_memory.zig: pinned host arenas and their blocks.
//! - io/log.zig: log scopes. Loader test support is in `zml.testing`.
//! - vfs: local and remote files behind one `std.Io`, the remote error
//!   contract, retries and the rate-limit hold (vfs/request.zig).

const std = @import("std");

pub const VFS = @import("vfs");

const loader = @import("io/loader.zig");
pub const Loader = loader.Loader;
pub const TensorStore = @import("io/TensorStore.zig");

pub const dma_calibration = @import("io/dma_calibration.zig");

test {
    std.testing.refAllDecls(@This());
}
