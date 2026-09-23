import torch
import torch.nn as nn

class TransformerClassifierHead(nn.Module):
    """
    Args:
        feature_dim: The input dimension of each class vector, D.
        num_heads: The number of heads in the self-attention mechanism.
        ffn_hidden_dim: The hidden dimension of the feed-forward network.
        dropout_rate: The dropout rate used in attention and FFN layers.
    """
    def __init__(self,
                 feature_dim: int,
                 num_heads: int,
                 ffn_hidden_dim: int,
                 dropout_rate: float):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(feature_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_hidden_dim, feature_dim)
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(feature_dim)

        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: The enhanced class features. Shape: (B, C, D).
        """
        attn_output, _ = self.self_attn(query=x, key=x, value=x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        logits = self.classifier(x)

        # Input: (B, C, 1) --> Output: (B, C)
        logits = logits.squeeze(-1)

        return logits
