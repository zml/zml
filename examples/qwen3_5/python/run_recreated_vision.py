from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

MODEL = "/var/models/Qwen/Qwen3.5-0.8B"
SAFETENSORS_FILE = "/home/tristan/zml/examples/qwen3_5/safetensors/vision_tests.safetensors"
DEVICE = "cuda"

#========================Qwen3.5 classes========================

class Qwen3_5VisionPatchMerger(nn.Module):
    def __init__(self, config, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


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



class Qwen3_5VisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Qwen3_5VisionBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5VisionAttention(config=config)
        self.mlp = Qwen3_5VisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3_5VisionAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1)
        key_states = key_states.transpose(0, 1)
        value_states = value_states.transpose(0, 1)

        outputs: list[torch.Tensor] = []
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            s = int(start.item())
            e = int(end.item())
            q = query_states[:, s:e, :].unsqueeze(0)
            k = key_states[:, s:e, :].unsqueeze(0)
            v = value_states[:, s:e, :].unsqueeze(0)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False).squeeze(0)
            outputs.append(y)

        attn_output = torch.cat(outputs, dim=1).transpose(0, 1).reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3_5VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = F.silu

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))




#========================Test setup========================

class TestVisionModel(nn.Module):
    def __init__(self, model_path: str, weights: dict[str, torch.Tensor], device: str = DEVICE) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        vision_config = config.vision_config
        self.device = torch.device(device)
        self.patch_embed = Qwen3_5VisionPatchEmbed(vision_config)
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.pos_embed = nn.Embedding(vision_config.num_position_embeddings, vision_config.hidden_size)
        self.blocks = nn.ModuleList([Qwen3_5VisionBlock(vision_config) for _ in range(vision_config.depth)])
        self.merger = Qwen3_5VisionPatchMerger(vision_config, use_postshuffle_norm=False)
        self.num_grid_per_side = int(vision_config.num_position_embeddings**0.5)
        self.to(self.device)
        self._copy_param(self.patch_embed.proj.weight, weights["model.visual.patch_embed.proj.weight"])
        self._copy_param(self.patch_embed.proj.bias, weights["model.visual.patch_embed.proj.bias"])
        self._copy_param(self.pos_embed.weight, weights["model.visual.pos_embed.weight"])
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(vision_config.hidden_size // vision_config.num_heads // 2)
        self.rotary_pos_emb.to(self.device)
        self._load_block_weights(weights)
        self._load_merger_weights(weights)

    @staticmethod
    def _copy_param(param: torch.Tensor, src: torch.Tensor) -> None:
        # Keep checkpoint dtype (bf16/f32/...) instead of forcing module default dtype.
        param.data = src.to(device=param.device).clone().contiguous()

    def _load_block_weights(self, weights: dict[str, torch.Tensor]) -> None:
        for i, block in enumerate(self.blocks):
            prefix = f"model.visual.blocks.{i}"
            self._copy_param(block.norm1.weight, weights[f"{prefix}.norm1.weight"])
            self._copy_param(block.norm1.bias, weights[f"{prefix}.norm1.bias"])
            self._copy_param(block.norm2.weight, weights[f"{prefix}.norm2.weight"])
            self._copy_param(block.norm2.bias, weights[f"{prefix}.norm2.bias"])
            self._copy_param(block.attn.qkv.weight, weights[f"{prefix}.attn.qkv.weight"])
            self._copy_param(block.attn.qkv.bias, weights[f"{prefix}.attn.qkv.bias"])
            self._copy_param(block.attn.proj.weight, weights[f"{prefix}.attn.proj.weight"])
            self._copy_param(block.attn.proj.bias, weights[f"{prefix}.attn.proj.bias"])
            self._copy_param(block.mlp.linear_fc1.weight, weights[f"{prefix}.mlp.linear_fc1.weight"])
            self._copy_param(block.mlp.linear_fc1.bias, weights[f"{prefix}.mlp.linear_fc1.bias"])
            self._copy_param(block.mlp.linear_fc2.weight, weights[f"{prefix}.mlp.linear_fc2.weight"])
            self._copy_param(block.mlp.linear_fc2.bias, weights[f"{prefix}.mlp.linear_fc2.bias"])

    def _load_merger_weights(self, weights: dict[str, torch.Tensor]) -> None:
        self._copy_param(self.merger.norm.weight, weights["model.visual.merger.norm.weight"])
        self._copy_param(self.merger.norm.bias, weights["model.visual.merger.norm.bias"])
        self._copy_param(self.merger.linear_fc1.weight, weights["model.visual.merger.linear_fc1.weight"])
        self._copy_param(self.merger.linear_fc1.bias, weights["model.visual.merger.linear_fc1.bias"])
        self._copy_param(self.merger.linear_fc2.weight, weights["model.visual.merger.linear_fc2.weight"])
        self._copy_param(self.merger.linear_fc2.bias, weights["model.visual.merger.linear_fc2.bias"])


    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(pixel_values)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = rotary_pos_emb.cos()
        sin = rotary_pos_emb.sin()

        grid_thw_list = grid_thw.tolist()
        lengths = [t * h * w for t, h, w in grid_thw_list]
        cu_seqlens = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(lengths, dtype=torch.int64), dim=0).tolist()),
            dtype=torch.int64,
            device=hidden_states.device,
        )

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=(cos, sin))
        hidden_states = self.merger(hidden_states)
        return hidden_states


    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()

        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw_list:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings


    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, device=device)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, device=device)

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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    checkpoint_file = resolve_single_checkpoint_file(MODEL)
    weights = load_file(str(checkpoint_file), device=DEVICE)

    test_model = TestVisionModel(MODEL, weights, device=DEVICE)

    # Qwen3_5Model.get_image_features expects pixel_values as (batch, channels, image_size, image_size).
    # Use batch=2 so patch_embed's internal view with temporal_patch_size=2 is valid.
    pixel_values = torch.randn((16, 3, 16, 16), dtype=torch.float32, device=DEVICE)
    grid_thw = torch.tensor([[2, 2, 2]], dtype=torch.int64, device=DEVICE)

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
