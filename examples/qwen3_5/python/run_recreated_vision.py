import torch
import torch.nn as nn
from safetensors.torch import save_file

MODEL = "/var/models/Qwen/Qwen3.5-0.8B"
SAFETENSORS_FILE = "/home/tristan/zml/examples/qwen3_5/safetensors/vision_tests.safetensors"


class VisionModel(nn.Module):
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        _ = grid_thw
        return pixel_values


def main() -> None:
    torch.manual_seed(0)

    model = VisionModel()

    pixel_values = torch.randn((1, 3, 16, 16), dtype=torch.float32)
    grid_thw = torch.tensor([1, 16, 16], dtype=torch.int64)

    output = model(pixel_values, grid_thw).clone()

    tensors = {
        "pixel_values": pixel_values.contiguous(),
        "grid_thw": grid_thw.contiguous(),
        "output": output.contiguous(),
    }
    save_file(tensors, SAFETENSORS_FILE)
    print(f"Saved {SAFETENSORS_FILE}")


if __name__ == "__main__":
    main()
