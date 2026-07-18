const zml = @import("zml");

pub fn forward(x: zml.Tensor, y: zml.Tensor) zml.Tensor {
    const sum = x.add(y);
    const doubled = sum.mulConstant(2);
    return doubled;
}
