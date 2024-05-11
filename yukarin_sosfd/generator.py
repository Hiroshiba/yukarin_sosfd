from pathlib import Path
from typing import Any, List, Union

import numpy
import torch
from torch import Tensor, nn
from typing_extensions import TypedDict

from yukarin_sosfd.config import Config
from yukarin_sosfd.network.predictor import Predictor, create_predictor


class GeneratorOutput(TypedDict):
    lf0: Tensor
    vuv: Tensor


def to_tensor(array: Union[Tensor, numpy.ndarray, Any]):
    if not isinstance(array, (Tensor, numpy.ndarray)):
        array = numpy.asarray(array)
    if isinstance(array, numpy.ndarray):
        return torch.from_numpy(array)
    else:
        return array


class Generator(nn.Module):
    def __init__(
        self,
        config: Config,
        predictor: Union[Predictor, Path],
        use_gpu: bool,
    ):
        super().__init__()

        self.config = config
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor, map_location=self.device)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

    def forward(
        self,
        noise_lf0_list: List[Union[numpy.ndarray, Tensor]],
        noise_vuv_list: List[Union[numpy.ndarray, Tensor]],
        accent_list: List[Union[numpy.ndarray, Tensor]],
        phoneme_list: List[Union[numpy.ndarray, Tensor]],
        speaker_id: Union[numpy.ndarray, Tensor],
        step_num: int,
        return_every_step: bool = False,
    ):
        noise_lf0_list = [
            to_tensor(noise_lf0).to(self.device) for noise_lf0 in noise_lf0_list
        ]
        noise_vuv_list = [
            to_tensor(noise_vuv).to(self.device) for noise_vuv in noise_vuv_list
        ]
        accent_list = [to_tensor(accent).to(self.device) for accent in accent_list]
        phoneme_list = [to_tensor(phoneme).to(self.device) for phoneme in phoneme_list]
        speaker_id = to_tensor(speaker_id).to(self.device)

        t = torch.linspace(0, 1, steps=step_num, device=self.device)

        lf0_list_step = []
        vuv_list_step = []

        with torch.inference_mode():
            lf0_list = [t.clone() for t in noise_lf0_list]
            vuv_list = [t.clone() for t in noise_vuv_list]
            vuv_hat_list = []

            if return_every_step:
                lf0_list_step.append([t.clone() for t in lf0_list])
                vuv_list_step.append([t.clone() for t in vuv_list])

            for i in range(step_num):
                if i == 0:
                    correct_lf0_list = lf0_list
                else:
                    correct_lf0_list = [
                        torch.lerp(lf0, torch.randn_like(lf0), (vuv_hat < 0.5).float())
                        for lf0, vuv_hat in zip(lf0_list, vuv_hat_list)
                    ]

                output_lf0_list, output_vuv_list = self.predictor(
                    lf0_list=correct_lf0_list,
                    vuv_list=vuv_list,
                    accent_list=accent_list,
                    phoneme_list=phoneme_list,
                    speaker_id=speaker_id,
                    t=t[i].expand(len(lf0_list)),
                )

                if return_every_step:
                    lf0_list_step.append(
                        [
                            (lf0 + output.unsqueeze(1) * (step_num - i) / step_num)
                            for lf0, output in zip(lf0_list, output_lf0_list)
                        ]
                    )
                    vuv_list_step.append(
                        [
                            (vuv + output.unsqueeze(1) * (step_num - i) / step_num)
                            for vuv, output in zip(vuv_list, output_vuv_list)
                        ]
                    )

                vuv_hat_list = self._denorm_vuv(
                    vuv_list=[
                        (vuv + output.unsqueeze(1) * (step_num - i) / step_num)
                        for vuv, output in zip(vuv_list, output_vuv_list)
                    ],
                    speaker_id_list=speaker_id.split(1),
                )

                for lf0, vuv, output_lf0, output_vuv in zip(
                    lf0_list, vuv_list, output_lf0_list, output_vuv_list
                ):
                    lf0 += output_lf0.unsqueeze(1) / step_num
                    vuv += output_vuv.unsqueeze(1) / step_num

        if not return_every_step:
            return [
                GeneratorOutput(lf0=lf0.squeeze(1), vuv=vuv.squeeze(1))
                for lf0, vuv in zip(lf0_list, vuv_list)
            ]
        else:
            return [
                [
                    GeneratorOutput(lf0=lf0.squeeze(1), vuv=vuv.squeeze(1))
                    for lf0, vuv in zip(lf0_list, vuv_list)
                ]
                for lf0_list, vuv_list in zip(lf0_list_step, vuv_list_step)
            ]

    def denorm(
        self,
        output_list: List[GeneratorOutput],
        speaker_id: Union[numpy.ndarray, Tensor],
    ):
        return [
            GeneratorOutput(
                lf0=(
                    output["lf0"] * self.predictor.lf0_std[speaker_id]
                    + self.predictor.lf0_mean[speaker_id]
                ).float(),
                vuv=(
                    output["vuv"] * self.predictor.vuv_std[speaker_id]
                    + self.predictor.vuv_mean[speaker_id]
                ),
            )
            for output, speaker_id in zip(
                output_list, to_tensor(speaker_id).to(self.device).split(1)
            )
        ]

    def _denorm_vuv(
        self,
        vuv_list: List[Tensor],
        speaker_id_list: List[Tensor],
    ):
        return [
            (
                vuv * self.predictor.vuv_std[speaker_id]
                + self.predictor.vuv_mean[speaker_id]
            )
            for vuv, speaker_id in zip(vuv_list, speaker_id_list)
        ]
