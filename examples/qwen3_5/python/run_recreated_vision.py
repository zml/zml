from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

MODEL = "/var/models/Qwen/Qwen3.5-0.8B"
SAFETENSORS_FILE = "/home/tristan/zml/examples/qwen3_5/safetensors/vision_tests.safetensors"

#========================Qwen3.5 classes========================

class Qwen3_5VisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states



#========================Test setup========================

class TestVisionModel(nn.Module):
    def __init__(self, model_path: str, weights: dict[str, torch.Tensor]) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.patch_embed = Qwen3_5VisionPatchEmbed(config.vision_config)
        self.patch_embed.proj.weight.data.copy_(
            weights["model.visual.patch_embed.proj.weight"].to(dtype=self.patch_embed.proj.weight.dtype)
        )
        self.patch_embed.proj.bias.data.copy_(
            weights["model.visual.patch_embed.proj.bias"].to(dtype=self.patch_embed.proj.bias.dtype)
        )

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        _ = grid_thw
        return self.patch_embed(pixel_values)


def resolve_single_checkpoint_file(model_path: str) -> Path:
    model_dir = Path(model_path)
    candidates = [
        model_dir / "model.safetensors-00001-of-00001.safetensors",
        model_dir / "model.safetensors",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find single safetensors checkpoint in {model_path}")


def main() -> None:
    torch.manual_seed(0)
    checkpoint_file = resolve_single_checkpoint_file(MODEL)
    weights = load_file(str(checkpoint_file))

    test_model = TestVisionModel(MODEL, weights)

    # Qwen3_5Model.get_image_features expects pixel_values as (batch, channels, image_size, image_size).
    # Use batch=2 so patch_embed's internal view with temporal_patch_size=2 is valid.
    pixel_values = torch.randn((2, 3, 16, 16), dtype=torch.float32)
    grid_thw = torch.tensor([2, 16, 16], dtype=torch.int64)

    output = test_model(pixel_values, grid_thw).clone()

    tensors = {
        "pixel_values": pixel_values.contiguous(),
        "grid_thw": grid_thw.contiguous(),
        "output": output.contiguous(),
    }
    save_file(tensors, SAFETENSORS_FILE)
    print(f"Saved {SAFETENSORS_FILE}")


if __name__ == "__main__":
    main()
