from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import TypedDict

from .config import ModelConfig
from .dataset import DatasetOutput
from .network.predictor import Predictor


class ModelOutput(TypedDict):
    loss: Tensor
    loss_f0: Tensor
    loss_vuv: Tensor
    loss_vol: Tensor
    data_num: int


def reduce_result(results: List[ModelOutput]):
    result: Dict[str, Any] = {}
    sum_data_num = sum([r["data_num"] for r in results])
    for key in set(results[0].keys()) - {"data_num"}:
        values = [r[key] * r["data_num"] for r in results]
        if isinstance(values[0], Tensor):
            result[key] = torch.stack(values).sum() / sum_data_num
        else:
            result[key] = sum(values) / sum_data_num
    return result


class Model(nn.Module):
    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, data: DatasetOutput) -> ModelOutput:
        output_lf0_list, output_vuv_list, output_vol_list = self.predictor(
            lf0_list=data["input_lf0"],
            vuv_list=data["input_vuv"],
            vol_list=data["input_vol"],
            accent_list=data["accent"],
            phoneme_list=data["phoneme"],
            speaker_id=torch.stack(data["speaker_id"]),
            lf0_t_list=data["lf0_t"],
            vuv_t_list=data["vuv_t"],
            vol_t_list=data["vol_t"],
        )

        output_lf0 = torch.cat(output_lf0_list)
        output_vuv = torch.cat(output_vuv_list)
        output_vol = torch.cat(output_vol_list)

        voiced = torch.cat(data["voiced"]).squeeze(1)

        target_lf0 = torch.cat(data["target_lf0"]).squeeze(1)
        noise_lf0 = torch.cat(data["noise_lf0"]).squeeze(1)
        diff_lf0 = target_lf0[voiced] - noise_lf0[voiced]
        loss_lf0 = F.mse_loss(output_lf0[voiced], diff_lf0)

        target_vuv = torch.cat(data["target_vuv"]).squeeze(1)
        noise_vuv = torch.cat(data["noise_vuv"]).squeeze(1)
        diff_vuv = target_vuv - noise_vuv
        loss_vuv = F.mse_loss(output_vuv, diff_vuv)

        target_vol = torch.cat(data["target_vol"]).squeeze(1)
        noise_vol = torch.cat(data["noise_vol"]).squeeze(1)
        diff_vol = target_vol - noise_vol
        loss_vol = F.mse_loss(output_vol, diff_vol)

        loss = loss_lf0 + loss_vuv + loss_vol

        return ModelOutput(
            loss=loss,
            loss_f0=loss_lf0,
            loss_vuv=loss_vuv,
            loss_vol=loss_vol,
            data_num=len(data),
        )
