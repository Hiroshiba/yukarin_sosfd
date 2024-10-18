# Original Code Copyright ESPnet
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)


from typing import Optional

from torch import Tensor, nn
from torch.nn import LayerNorm

from ...network.transformer.attention import MultiHeadedAttention


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        self_attn: MultiHeadedAttention,
        conv_module: Optional[nn.Module],
        feed_forward: nn.Module,
        dropout_rate: float,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.conv_module = conv_module
        self.feed_forward = feed_forward
        self.norm_ff = LayerNorm(hidden_size, eps=1e-12)
        self.norm_mha = LayerNorm(hidden_size, eps=1e-12)
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(hidden_size, eps=1e-12)
            self.norm_final = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: Tensor,  # (B, T, ?)
        pos_emb: Tensor,  # (1, T, ?)
        mask: Tensor,  # (B, 1, T)
    ):
        residual = x
        x = self.norm_mha(x)
        x_att = self.self_attn(x, x, x, pos_emb, mask)
        x = residual + self.dropout(x_att)

        if self.conv_module is not None:
            residual = x
            x = self.norm_conv(x)
            x = residual + self.dropout(self.conv_module(x))

        residual = x  # FIXME: 再代入なくして最初のxを足す形でも良いかも
        x = self.norm_ff(x)
        x = residual + self.dropout(self.feed_forward(x))

        return x, pos_emb, mask
