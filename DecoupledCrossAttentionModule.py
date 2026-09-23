import torch
import torch.nn as nn
from typing import Tuple

class PositionalEncoding2D(nn.Module):
    """
    Generates fixed 2D sinusoidal positional encodings.
    """
    def __init__(self,
                 feature_dim: int,
                 temperature: float = 10000.0,
                 normalize: bool = False,
                 scale: float = None):
        super().__init__()
        assert feature_dim % 2 == 0, \
            f"feature_dim must be an even number, but got {feature_dim}"
        if normalize:
            assert scale is not None, "scale must be provided when normalize is True"

        self.feature_dim = feature_dim
        self.half_dim = feature_dim // 2
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        H, W = x.shape[-2], x.shape[-1]
        device = x.device
        mask = torch.ones(B, H, W, dtype=torch.bool, device=device)
        y_embed = (mask.cumsum(1, dtype=torch.float32) - 1)
        x_embed = (mask.cumsum(2, dtype=torch.float32) - 1)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.half_dim, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.half_dim)

        pos_y = y_embed.unsqueeze(3) / dim_t
        pos_x = x_embed.unsqueeze(3) / dim_t
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_encoding = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos_encoding


class DecoupledCrossAttentionWithPE(nn.Module):
    """
    Args:
        num_classes: The number of distinct classes (C).
        feature_dim: The channel dimension of the input feature map (D).
        num_heads: The number of parallel attention heads.
        temperature: Temperature for the positional encoding.
        normalize: Whether to normalize positional encoding coordinates.
        scale: Scaling factor for normalized positional encoding.
    """
    def __init__(self,
                 num_classes: int,
                 feature_dim: int,
                 num_heads: int,
                 temperature: float = 10000.0,
                 normalize: bool = False,
                 scale: float = None):
        super().__init__()

        self.pos_encoder = PositionalEncoding2D(
            feature_dim=feature_dim,
            temperature=temperature,
            normalize=normalize,
            scale=scale
        )

        self.class_queries = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature_map: The input 2D feature map from the backbone.
                                        Shape: (B, D, H, W).
        """
        pos_encoding = self.pos_encoder(feature_map)
        B, D, H, W = feature_map.shape
        kv_input = feature_map + pos_encoding
        kv_input = kv_input.flatten(2).permute(0, 2, 1)  # (B, H*W, D)
        query = self.class_queries.unsqueeze(0).expand(B, -1, -1) # (B, C, D)
        context, attention_weights = self.attention(
            query=query,
            key=kv_input,
            value=kv_input
        )
        enhanced_features = self.norm(query + context)
        return enhanced_features, attention_weights
