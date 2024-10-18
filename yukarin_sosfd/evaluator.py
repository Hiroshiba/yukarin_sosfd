from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import Literal, TypedDict

from .dataset import DatasetOutput
from .generator import Generator, GeneratorOutput


class EvaluatorOutput(TypedDict):
    value: Tensor
    diff_f0: Tensor
    precision_voiced: Tensor
    recall_voiced: Tensor
    precision_unvoiced: Tensor
    recall_unvoiced: Tensor
    diff_vol: Tensor
    data_num: int


def calc_pr(output: Tensor, target: Tensor):
    tp = ((output >= 0.5) & (target == 1)).float().sum()
    fp = ((output >= 0.5) & (target == 0)).float().sum()
    fn = ((output < 0.5) & (target == 1)).float().sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall


class Evaluator(nn.Module):
    judge: Literal["min", "max"] = "min"

    def __init__(self, generator: Generator, step_num: int):
        super().__init__()
        self.generator = generator
        self.step_num = step_num

    def forward(self, data: DatasetOutput) -> EvaluatorOutput:
        output_list: List[GeneratorOutput] = self.generator(
            noise_lf0_list=data["noise_lf0"],
            noise_vuv_list=data["noise_vuv"],
            noise_vol_list=data["noise_vol"],
            accent_list=data["accent"],
            phoneme_list=data["phoneme"],
            speaker_id=torch.stack(data["speaker_id"]),
            step_num=self.step_num,
        )

        output_lf0 = torch.cat([output["lf0"] for output in output_list])
        output_vuv = torch.cat([output["vuv"] for output in output_list])
        output_vol = torch.cat([output["vol"] for output in output_list])

        voiced = torch.cat(data["voiced"]).squeeze(1)

        target_lf0 = torch.cat(data["target_lf0"]).squeeze(1)
        diff_lf0 = F.mse_loss(output_lf0[voiced], target_lf0[voiced])

        precision_voiced, recall_voiced = calc_pr(output_vuv, voiced)
        precision_unvoiced, recall_unvoiced = calc_pr(1 - output_vuv, ~voiced)

        target_vol = torch.cat(data["target_vol"]).squeeze(1)
        diff_vol = F.mse_loss(output_vol, target_vol)

        value = diff_lf0 + diff_vol

        return EvaluatorOutput(
            value=value,
            diff_f0=diff_lf0,
            precision_voiced=precision_voiced,
            recall_voiced=recall_voiced,
            precision_unvoiced=precision_unvoiced,
            recall_unvoiced=recall_unvoiced,
            diff_vol=diff_vol,
            data_num=len(data),
        )
