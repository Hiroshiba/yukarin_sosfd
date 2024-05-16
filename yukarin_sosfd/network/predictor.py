from typing import List, Optional

import torch
from espnet_pytorch_library.nets_utils import make_non_pad_mask
from espnet_pytorch_library.tacotron2.decoder import Postnet
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from yukarin_sosfd.config import NetworkConfig
from yukarin_sosfd.dataset import DatasetStatistics
from yukarin_sosfd.network.conformer.encoder import Encoder


class Predictor(nn.Module):
    def __init__(
        self,
        speaker_size: int,
        speaker_embedding_size: int,
        phoneme_size: int,
        phoneme_embedding_size: int,
        hidden_size: int,
        block_num: int,
        post_layer_num: int,
        dropout_rate: bool,
        positional_dropout_rate: bool,
        attention_dropout_rate: bool,
        statistics: Optional[DatasetStatistics] = None,  # 話者ごとの統計情報
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
            1 + 1 + 4 + phoneme_embedding_size + speaker_embedding_size + 1
        )  # lf0 + vuv + accent + phoneme + speaker + t
        self.pre = torch.nn.Linear(input_size, hidden_size)

        self.encoder = Encoder(
            hidden_size=hidden_size,
            num_blocks=block_num,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_head_size=2,
            attention_dropout_rate=attention_dropout_rate,
            use_conv_glu_module=True,
            conv_glu_module_kernel_size=31,
            feed_forward_hidden_size=hidden_size * 4,
            feed_forward_kernel_size=3,
        )

        output_size = 1 + 1  # lf0 + vuv
        self.post = torch.nn.Linear(hidden_size, output_size)

        if post_layer_num > 0:
            self.postnet = Postnet(
                idim=output_size,
                odim=output_size,
                n_layers=post_layer_num,
                n_chans=hidden_size,
                n_filts=5,
                use_batch_norm=True,
                dropout_rate=0.5,
            )
        else:
            self.postnet = None

        self.lf0_mean: Tensor
        self.lf0_std: Tensor
        self.vuv_mean: Tensor
        self.vuv_std: Tensor
        if statistics is not None:
            self.register_buffer("lf0_mean", torch.from_numpy(statistics.lf0_mean))
            self.register_buffer("lf0_std", torch.from_numpy(statistics.lf0_std))
            self.register_buffer("vuv_mean", torch.from_numpy(statistics.vuv_mean))
            self.register_buffer("vuv_std", torch.from_numpy(statistics.vuv_std))
        else:
            self.register_buffer("lf0_mean", torch.full((speaker_size,), torch.nan))
            self.register_buffer("lf0_std", torch.full((speaker_size,), torch.nan))
            self.register_buffer("vuv_mean", torch.full((speaker_size,), torch.nan))
            self.register_buffer("vuv_std", torch.full((speaker_size,), torch.nan))

    def _mask(self, length: Tensor):
        x_masks = make_non_pad_mask(length).to(length.device)
        return x_masks.unsqueeze(-2)

    def forward(
        self,
        lf0_list: List[Tensor],  # [(L, 1)]
        vuv_list: List[Tensor],  # [(L, 1)]
        accent_list: List[Tensor],  # [(L, 4)]
        phoneme_list: List[Tensor],  # [(L, 1)]
        speaker_id: Tensor,  # (B, )
        t: Tensor,  # (B, )
    ):
        """
        B: batch size
        L: length
        """
        length_list = [t.shape[0] for t in lf0_list]

        length = torch.tensor(length_list, device=lf0_list[0].device)
        lf0 = pad_sequence(lf0_list, batch_first=True)  # (B, L, ?)
        vuv = pad_sequence(vuv_list, batch_first=True)  # (B, L, ?)
        accent = pad_sequence(accent_list, batch_first=True)  # (B, L, ?)

        phoneme = pad_sequence(phoneme_list, batch_first=True).squeeze(2)  # (B, L)
        phoneme = self.phoneme_embedder(phoneme)  # (B, L, ?)

        speaker_id = self.speaker_embedder(speaker_id)
        speaker_id = speaker_id.unsqueeze(dim=1)  # (B, 1, ?)
        speaker_id = speaker_id.expand(
            speaker_id.shape[0], lf0.shape[1], speaker_id.shape[2]
        )  # (B, L, ?)

        t = t.unsqueeze(dim=1).unsqueeze(dim=2)  # (B, 1, ?)
        t = t.expand(t.shape[0], lf0.shape[1], t.shape[2])  # (B, L, ?)

        h = torch.cat((lf0, vuv, accent, phoneme, speaker_id, t), dim=2)  # (B, L, ?)
        h = self.pre(h)

        mask = self._mask(length)
        h, _ = self.encoder(h, mask)

        output1 = self.post(h)
        if self.postnet is not None:
            output2 = output1 + self.postnet(output1.transpose(1, 2)).transpose(1, 2)
        else:
            output2 = output1

        return (
            [output2[i, :l, 0] for i, l in enumerate(length_list)],  # lf0
            [output2[i, :l, 1] for i, l in enumerate(length_list)],  # vuv
        )


def create_predictor(
    config: NetworkConfig, statistics: Optional[DatasetStatistics] = None
):
    return Predictor(
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        phoneme_size=config.phoneme_size,
        phoneme_embedding_size=config.phoneme_embedding_size,
        hidden_size=config.hidden_size,
        block_num=config.block_num,
        post_layer_num=config.post_layer_num,
        dropout_rate=config.dropout_rate,
        positional_dropout_rate=config.positional_dropout_rate,
        attention_dropout_rate=config.attention_dropout_rate,
        statistics=statistics,
    )
