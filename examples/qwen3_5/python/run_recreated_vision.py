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
        vision_config = config.vision_config
        self.patch_embed = Qwen3_5VisionPatchEmbed(vision_config)
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.pos_embed = nn.Embedding(vision_config.num_position_embeddings, vision_config.hidden_size)
        self.num_grid_per_side = int(vision_config.num_position_embeddings**0.5)
        self.patch_embed.proj.weight.data.copy_(
            weights["model.visual.patch_embed.proj.weight"].to(dtype=self.patch_embed.proj.weight.dtype)
        )
        self.patch_embed.proj.bias.data.copy_(
            weights["model.visual.patch_embed.proj.bias"].to(dtype=self.patch_embed.proj.bias.dtype)
        )
        self.pos_embed.weight.data.copy_(weights["model.visual.pos_embed.weight"].to(dtype=self.pos_embed.weight.dtype))

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states =  self.patch_embed(pixel_values)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        return hidden_states

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds


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
    pixel_values = torch.randn((16, 3, 16, 16), dtype=torch.float32)
    grid_thw = torch.tensor([[2, 2, 2]], dtype=torch.int64)

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
