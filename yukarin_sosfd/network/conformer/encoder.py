# Original Code Copyright ESPnet
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch
from torch import Tensor, nn

from .convolution import ConvGLUModule
from .encoder_layer import EncoderLayer
from ..transformer.attention import RelPositionMultiHeadedAttention
from ..transformer.embedding import RelPositionalEncoding
from ..transformer.multi_layer_conv import FastSpeechTwoConv


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_blocks: int,
        dropout_rate: float,
        positional_dropout_rate: float,
        attention_head_size: int,
        attention_dropout_rate: float,
        use_conv_glu_module: bool,
        conv_glu_module_kernel_size: int,
        feed_forward_hidden_size: int,
        feed_forward_kernel_size: int,
    ):
        super().__init__()

        self.embed = RelPositionalEncoding(hidden_size, positional_dropout_rate)

        self.encoders = nn.Sequential(
            *[
                EncoderLayer(
                    hidden_size=hidden_size,
                    self_attn=RelPositionMultiHeadedAttention(
                        head_size=attention_head_size,
                        hidden_size=hidden_size,
                        dropout_rate=attention_dropout_rate,
                    ),
                    conv_module=(
                        ConvGLUModule(
                            hidden_size=hidden_size,
                            kernel_size=conv_glu_module_kernel_size,
                            activation=Swish,
                        )
                        if use_conv_glu_module
                        else None
                    ),
                    feed_forward=FastSpeechTwoConv(
                        inout_size=hidden_size,
                        hidden_size=feed_forward_hidden_size,
                        kernel_size=feed_forward_kernel_size,
                        dropout_rate=dropout_rate,
                    ),
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )
        self.after_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(
        self,
        x: Tensor,  # (B, T, ?)
        mask: Tensor,  # (B, 1, T)
    ):
        x, pos_emb = self.embed(x)
        for encoder in self.encoders:
            x, pos_emb, mask = encoder(x=x, pos_emb=pos_emb, mask=mask)
        x = self.after_norm(x)
        return x, mask
