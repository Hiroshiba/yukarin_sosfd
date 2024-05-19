from typing import List, Optional

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from yukarin_sosfd.config import NetworkConfig
from yukarin_sosfd.data.statistic import DataStatistics
from yukarin_sosfd.network.conformer.encoder import Encoder
from yukarin_sosfd.network.transformer.utility import make_non_pad_mask


class Predictor(nn.Module):
    def __init__(
        self,
        speaker_size: int,
        speaker_embedding_size: int,
        phoneme_size: int,
        phoneme_embedding_size: int,
        hidden_size: int,
        encoder: Encoder,
        statistics: Optional[DataStatistics] = None,  # 話者ごとの統計情報
    ):
        super().__init__()

        self.speaker_embedder = nn.Embedding(
            num_embeddings=speaker_size,
            embedding_dim=speaker_embedding_size,
        )

        self.phoneme_embedder = nn.Embedding(
            num_embeddings=phoneme_size,
            embedding_dim=phoneme_embedding_size,
        )

        input_size = (
            1 + 1 + 1 + 4 + phoneme_embedding_size + speaker_embedding_size + 1 + 1 + 1
        )  # lf0 + vuv + volume + accent + phoneme + speaker + lf0_t + vuv_t + vol_t
        self.pre = torch.nn.Linear(input_size, hidden_size)

        self.encoder = encoder

        output_size = 1 + 1 + 1  # lf0 + vuv + volume
        self.post = torch.nn.Linear(hidden_size, output_size)

        self.lf0_mean: Tensor
        self.lf0_std: Tensor
        self.vuv_mean: Tensor
        self.vuv_std: Tensor
        self.vol_mean: Tensor
        self.vol_std: Tensor
        if statistics is not None:
            self.register_buffer("lf0_mean", torch.from_numpy(statistics.lf0_mean))
            self.register_buffer("lf0_std", torch.from_numpy(statistics.lf0_std))
            self.register_buffer("vuv_mean", torch.from_numpy(statistics.vuv_mean))
            self.register_buffer("vuv_std", torch.from_numpy(statistics.vuv_std))
            self.register_buffer("vol_mean", torch.from_numpy(statistics.vol_mean))
            self.register_buffer("vol_std", torch.from_numpy(statistics.vol_std))
        else:
            self.register_buffer("lf0_mean", torch.full((speaker_size,), torch.nan))
            self.register_buffer("lf0_std", torch.full((speaker_size,), torch.nan))
            self.register_buffer("vuv_mean", torch.full((speaker_size,), torch.nan))
            self.register_buffer("vuv_std", torch.full((speaker_size,), torch.nan))
            self.register_buffer("vol_mean", torch.full((speaker_size,), torch.nan))
            self.register_buffer("vol_std", torch.full((speaker_size,), torch.nan))

    def forward(
        self,
        lf0_list: List[Tensor],  # [(L, 1)]
        vuv_list: List[Tensor],  # [(L, 1)]
        vol_list: List[Tensor],  # [(L, 1)]
        accent_list: List[Tensor],  # [(L, 4)]
        phoneme_list: List[Tensor],  # [(L, 1)]
        speaker_id: Tensor,  # (B, )
        lf0_t_list: List[Tensor],  # [(L, 1)]
        vuv_t_list: List[Tensor],  # [(L, 1)]
        vol_t_list: List[Tensor],  # [(L, 1)]
    ):
        """
        B: batch size
        L: length
        """
        length_list = [t.shape[0] for t in lf0_list]

        lf0 = pad_sequence(lf0_list, batch_first=True)  # (B, L, ?)
        vuv = pad_sequence(vuv_list, batch_first=True)  # (B, L, ?)
        vol = pad_sequence(vol_list, batch_first=True)  # (B, L, ?)
        accent = pad_sequence(accent_list, batch_first=True)  # (B, L, ?)
        lf0_t = pad_sequence(lf0_t_list, batch_first=True)  # (B, L, ?)
        vuv_t = pad_sequence(vuv_t_list, batch_first=True)  # (B, L, ?)
        vol_t = pad_sequence(vol_t_list, batch_first=True)  # (B, L, ?)

        phoneme = pad_sequence(phoneme_list, batch_first=True).squeeze(2)  # (B, L)
        phoneme = self.phoneme_embedder(phoneme)  # (B, L, ?)

        speaker_id = self.speaker_embedder(speaker_id)
        speaker_id = speaker_id.unsqueeze(dim=1)  # (B, 1, ?)
        speaker_id = speaker_id.expand(
            speaker_id.shape[0], lf0.shape[1], speaker_id.shape[2]
        )  # (B, L, ?)

        t = torch.cat((lf0_t, vuv_t, vol_t), dim=2)  # (B, L, ?)
        h = torch.cat(
            (lf0, vuv, vol, accent, phoneme, speaker_id, t), dim=2
        )  # (B, L, ?)
        h = self.pre(h)

        mask = make_non_pad_mask(length_list).unsqueeze(-2).to(h.device)
        h, _ = self.encoder(x=h, cond=t, mask=mask)  # (B, L, ?)

        output = self.post(h)  # (B, L, ?)
        return (
            [output[i, :l, 0] for i, l in enumerate(length_list)],  # lf0
            [output[i, :l, 1] for i, l in enumerate(length_list)],  # vuv
            [output[i, :l, 2] for i, l in enumerate(length_list)],  # volume
        )


def create_predictor(
    config: NetworkConfig, statistics: Optional[DataStatistics] = None
):
    encoder = Encoder(
        hidden_size=config.hidden_size,
        condition_size=3 if config.with_condition_t else 0,
        block_num=config.block_num,
        dropout_rate=config.dropout_rate,
        positional_dropout_rate=config.positional_dropout_rate,
        attention_head_size=2,
        attention_dropout_rate=config.attention_dropout_rate,
        use_conv_glu_module=config.use_conv_glu_module,
        conv_glu_module_kernel_size=31,
        feed_forward_hidden_size=config.hidden_size * 4,
        feed_forward_kernel_size=3,
    )
    return Predictor(
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        phoneme_size=config.phoneme_size,
        phoneme_embedding_size=config.phoneme_embedding_size,
        hidden_size=config.hidden_size,
        encoder=encoder,
        statistics=statistics,
    )
