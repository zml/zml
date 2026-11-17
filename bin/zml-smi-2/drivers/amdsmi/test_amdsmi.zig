const std = @import("std");
const AmdSmi = @import("zml-smi-2/drivers/amdsmi");

const pciAddress = AmdSmi.testing.pciAddress;

test pciAddress {
    try std.testing.expectEqualStrings("0000:03:00.0", &pciAddress(0x0300).?);
    // Bits 28 to 31 hold a partition id, which is not part of the address.
    try std.testing.expectEqualStrings("0001:c1:00.1", &pciAddress(0x1_1000_c101).?);
}
