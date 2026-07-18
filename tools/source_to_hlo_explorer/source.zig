const zml = @import("zml");

pub fn forward(x: zml.Tensor, y: zml.Tensor) zml.Tensor {
    const sum = x.addAt(y, @src());
    const doubled = sum.mulConstantAt(2, @src());
    return doubled;
}
