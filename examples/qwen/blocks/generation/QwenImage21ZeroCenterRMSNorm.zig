const zml = @import("zml");

const QwenImage21ZeroCenterRMSNorm = @This();

// Stored zero-centered: the effective scale is `weight + 1`.
weight: zml.Tensor,
eps: f32 = 1e-5,

pub fn unloadBuffers(self: *zml.Bufferized(QwenImage21ZeroCenterRMSNorm)) void {
    zml.Buffer.deinitAll(QwenImage21ZeroCenterRMSNorm, self);
}

pub fn forward(self: QwenImage21ZeroCenterRMSNorm, x: zml.Tensor) zml.Tensor {
    const floats = x.convert(.f32);

    const twos = zml.Tensor.scalar(2, .f32).broad(floats.shape());
    const teps = zml.Tensor.scalar(self.eps, .f32).broad(floats.shape());

    const pow = floats.pow(twos);
    const rrms = zml.Tensor.rsqrt(zml.Tensor.mean(pow, -1).broad(teps.shape()).add(teps));

    const scale = self.weight.convert(.f32).addConstant(1).broad(floats.shape());
    return floats.mul(rrms).mul(scale).convert(x.dtype());
}
