const nn = @import("../../nn.zig");
const Tensor = @import("../../tensor.zig").Tensor;

pub fn unpackedWeight(linear: nn.Linear) Tensor {
    const quantization = linear.quantization orelse return linear.weight;
    return if (nn.isPackedFp4(quantization.scheme, linear.weight.dtype()))
        nn.unpackFp4(linear.weight, linear.tag, linear.tag)
    else
        linear.weight;
}
