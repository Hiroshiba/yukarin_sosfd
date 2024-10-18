from pathlib import Path
from typing import Any, List, Optional, Union

import numpy
import torch
from torch import Tensor, nn
from typing_extensions import TypedDict

from .config import Config
from .network.predictor import Predictor, create_predictor


class GeneratorOutput(TypedDict):
    lf0: Tensor
    vuv: Tensor
    vol: Tensor


def to_tensor(array: Union[Tensor, numpy.ndarray, Any]):
    if not isinstance(array, (Tensor, numpy.ndarray)):
        array = numpy.asarray(array)
    if isinstance(array, numpy.ndarray):
        return torch.from_numpy(array)
    else:
        return array


def apply(func: callable, *gen: GeneratorOutput) -> GeneratorOutput:
    return GeneratorOutput(
        lf0=func(*(g["lf0"] for g in gen)),
        vuv=func(*(g["vuv"] for g in gen)),
        vol=func(*(g["vol"] for g in gen)),
    )


def apply_map(
    func: callable, *gen_list: List[GeneratorOutput]
) -> List[GeneratorOutput]:
    return [apply(func, *gen) for gen in zip(*gen_list)]


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
        input_lf0_list: List[Union[numpy.ndarray, Tensor]],
        input_vuv_list: List[Union[numpy.ndarray, Tensor]],
        input_vol_list: List[Union[numpy.ndarray, Tensor]],
        t_lf0_list: Optional[List[Union[numpy.ndarray, Tensor]]],
        t_vuv_list: Optional[List[Union[numpy.ndarray, Tensor]]],
        t_vol_list: Optional[List[Union[numpy.ndarray, Tensor]]],
        accent_list: List[Union[numpy.ndarray, Tensor]],
        phoneme_list: List[Union[numpy.ndarray, Tensor]],
        speaker_id: Union[numpy.ndarray, Tensor],
        step_num: int,
        return_every_step: bool = False,
    ):
        def prepare_tensors(
            array_list: List[Union[numpy.ndarray, Tensor]]
        ) -> List[Tensor]:
            return [to_tensor(array).to(self.device) for array in array_list]

        input_list = [
            GeneratorOutput(lf0=lf0, vuv=vuv, vol=vol)
            for lf0, vuv, vol in zip(
                prepare_tensors(input_lf0_list),
                prepare_tensors(input_vuv_list),
                prepare_tensors(input_vol_list),
            )
        ]
        accent_list = prepare_tensors(accent_list)
        phoneme_list = prepare_tensors(phoneme_list)
        speaker_id = to_tensor(speaker_id).to(self.device)

        num = step_num
        del step_num

        input_t_list = [
            GeneratorOutput(lf0=t_lf0, vuv=t_vuv, vol=t_vol)
            for t_lf0, t_vuv, t_vol in zip(
                (
                    prepare_tensors(t_lf0_list)
                    if t_lf0_list is not None
                    else [torch.zeros_like(input["lf0"]) for input in input_list]
                ),
                (
                    prepare_tensors(t_vuv_list)
                    if t_vuv_list is not None
                    else [torch.zeros_like(input["vuv"]) for input in input_list]
                ),
                (
                    prepare_tensors(t_vol_list)
                    if t_vol_list is not None
                    else [torch.zeros_like(input["vol"]) for input in input_list]
                ),
            )
        ]

        hat_list_step: List[List[GeneratorOutput]] = []

        with torch.inference_mode():
            gen_list = apply_map(lambda x: x.clone(), input_list)
            hat_list_step.append(
                self.denorm(
                    apply_map(lambda x: x.clone(), gen_list), speaker_id=speaker_id
                )
            )

            for i in range(num):
                if i == 0:
                    correct_lf0_list = [gen["lf0"] for gen in gen_list]
                else:
                    correct_lf0_list = [
                        torch.lerp(gen["lf0"], noise["lf0"], (hat["vuv"] < 0.5).float())
                        for gen, noise, hat in zip(
                            gen_list, input_list, hat_list_step[-1]
                        )
                    ]

                t_list = apply_map(lambda x: x + (1 - x) * i / num, input_t_list)

                out_lf0_list, out_vuv_list, out_vol_list = self.predictor(
                    lf0_list=correct_lf0_list,
                    vuv_list=[gen["vuv"] for gen in gen_list],
                    vol_list=[gen["vol"] for gen in gen_list],
                    accent_list=accent_list,
                    phoneme_list=phoneme_list,
                    speaker_id=speaker_id,
                    lf0_t_list=[t["lf0"] for t in t_list],
                    vuv_t_list=[t["vuv"] for t in t_list],
                    vol_t_list=[t["vol"] for t in t_list],
                )

                out_list = [
                    GeneratorOutput(lf0=out_lf0, vuv=out_vuv, vol=out_vol)
                    for out_lf0, out_vuv, out_vol in zip(
                        out_lf0_list, out_vuv_list, out_vol_list
                    )
                ]

                gen_list = apply_map(
                    lambda x, y, z: x + y.unsqueeze(1) / num * (1 - z),
                    gen_list,
                    out_list,
                    input_t_list,
                )

                hat_list_step.append(
                    self.denorm(
                        apply_map(
                            lambda x, y, z: x + y.unsqueeze(1) * (1 - z),
                            gen_list,
                            out_list,
                            t_list,
                        ),
                        speaker_id=speaker_id,
                    )
                )

        def _to_return(gen_list):
            return apply_map(lambda x: x.squeeze(1), gen_list)

        if not return_every_step:
            return _to_return(self.denorm(gen_list, speaker_id=speaker_id))
        else:
            return [_to_return(hat_list) for hat_list in hat_list_step]

    def denorm(
        self,
        out_list: List[GeneratorOutput],
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
                ).float(),
                vol=(
                    output["vol"] * self.predictor.vol_std[speaker_id]
                    + self.predictor.vol_mean[speaker_id]
                ).float(),
            )
            for output, speaker_id in zip(
                out_list, to_tensor(speaker_id).to(self.device).split(1)
            )
        ]
