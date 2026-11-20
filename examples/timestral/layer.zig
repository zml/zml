const zml = @import("zml");

pub const ResidualBlock = struct {
    hidden_layer: zml.nn.Linear,
    output_layer: zml.nn.Linear,
    residual_layer: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) ResidualBlock {
        const hl = store.withPrefix("hidden_layer");
        const hidden_layer: zml.nn.Linear = .init(
            hl.createTensor(
                "weight",
                .{ .h, .d },
                .replicated,
                .{ .h = .replicated, .d = .replicated },
            ),
            hl.maybeCreateTensor("bias", .{.h}, .replicated, .{ .h = .replicated }),
            .d,
        );
        const ol = store.withPrefix("output_layer");
        const output_layer: zml.nn.Linear = .init(
            ol.createTensor(
                "weight",
                .{ .out, .h },
                .replicated,
                .{ .out = .replicated, .h = .replicated },
            ),
            ol.maybeCreateTensor("bias", .{.out}, .replicated, .{ .out = .replicated }),
            .h,
        );
        const rl = store.withPrefix("residual_layer");
        const residual_layer: zml.nn.Linear = .init(
            rl.createTensor(
                "weight",
                .{ .out, .d },
                .replicated,
                .{ .out = .replicated, .d = .replicated },
            ),
            rl.maybeCreateTensor("bias", .{.out}, .replicated, .{ .out = .replicated }),
            .d,
        );

        return .{
            .hidden_layer = hidden_layer,
            .output_layer = output_layer,
            .residual_layer = residual_layer,
        };
    }

    pub fn forward(self: ResidualBlock, x: zml.Tensor) zml.Tensor {
        const h = self.hidden_layer.forward(x, x.dtype()).silu();
        const out = self.output_layer.forward(h, h.dtype());
        return out.add(self.residual_layer.forward(x, x.dtype()));
    }
};
