import json
from dataclasses import dataclass
from functools import partial
from itertools import groupby
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from typing_extensions import TypedDict

from yukarin_sosfd.config import DatasetConfig, DatasetFileConfig
from yukarin_sosfd.data.phoneme import OjtPhoneme
from yukarin_sosfd.data.sampling_data import SamplingData
from yukarin_sosfd.utility.dataset_utility import get_stem_to_paths


@dataclass
class DatasetStatistics:
    lf0_mean: numpy.ndarray
    lf0_std: numpy.ndarray
    vuv_mean: numpy.ndarray
    vuv_std: numpy.ndarray


@dataclass
class DatasetInput:
    lf0: SamplingData
    start_accent_list: numpy.ndarray
    end_accent_list: numpy.ndarray
    start_accent_phrase_list: numpy.ndarray
    end_accent_phrase_list: numpy.ndarray
    phoneme_list: List[OjtPhoneme]
    silence: SamplingData
    speaker_id: int


@dataclass
class LazyDatasetInput:
    lf0_path: Path
    start_accent_list_path: Path
    end_accent_list_path: Path
    start_accent_phrase_list_path: Path
    end_accent_phrase_list_path: Path
    phoneme_list_path: Path
    silence_path: Path
    speaker_id: int

    def generate(self):
        return DatasetInput(
            lf0=SamplingData.load(self.lf0_path),
            start_accent_list=numpy.array(
                [bool(int(s)) for s in self.start_accent_list_path.read_text().split()]
            ),
            end_accent_list=numpy.array(
                [bool(int(s)) for s in self.end_accent_list_path.read_text().split()]
            ),
            start_accent_phrase_list=numpy.array(
                [
                    bool(int(s))
                    for s in self.start_accent_phrase_list_path.read_text().split()
                ]
            ),
            end_accent_phrase_list=numpy.array(
                [
                    bool(int(s))
                    for s in self.end_accent_phrase_list_path.read_text().split()
                ]
            ),
            phoneme_list=OjtPhoneme.load_julius_list(self.phoneme_list_path),
            silence=SamplingData.load(self.silence_path),
            speaker_id=self.speaker_id,
        )


class DatasetOutput(TypedDict):
    input_lf0: Tensor
    target_lf0: Tensor
    noise_lf0: Tensor
    input_vuv: Tensor
    target_vuv: Tensor
    noise_vuv: Tensor
    accent: Tensor
    phoneme: Tensor
    voiced: Tensor
    silence: Tensor
    speaker_id: Tensor
    t: Tensor


def get_notsilence_range(silence: numpy.ndarray, prepost_silence_length: int):
    """
    最初と最後の無音を除去したrangeを返す。
    一番最初や最後が無音でない場合はノイズとみなしてその区間も除去する。
    最小でもprepost_silence_lengthだけは確保する。
    """
    length = len(silence)

    ps = numpy.argwhere(numpy.logical_and(silence[:-1], ~silence[1:]))
    pre_length = ps[0][0] + 1 if len(ps) > 0 else 0
    pre_index = max(0, pre_length - prepost_silence_length)

    ps = numpy.argwhere(numpy.logical_and(~silence[:-1], silence[1:]))
    post_length = length - (ps[-1][0] + 1) if len(ps) > 0 else 0
    post_index = length - max(0, post_length - prepost_silence_length)
    return range(pre_index, post_index)


def calc_statistics(
    ds: list[DatasetInput], prepost_silence_length: int
) -> DatasetStatistics:
    """話者ごとの統計情報を取得"""
    max_speaker_id = max(d.speaker_id for d in ds)
    statistics = DatasetStatistics(
        lf0_mean=numpy.full(max_speaker_id + 1, numpy.nan, dtype=numpy.float64),
        lf0_std=numpy.full(max_speaker_id + 1, numpy.nan, dtype=numpy.float64),
        vuv_mean=numpy.full(max_speaker_id + 1, numpy.nan, dtype=numpy.float64),
        vuv_std=numpy.full(max_speaker_id + 1, numpy.nan, dtype=numpy.float64),
    )

    datas_dict = groupby(
        sorted(ds, key=lambda d: d.speaker_id), key=lambda d: d.speaker_id
    )
    for speaker_id, ds in datas_dict:
        lf0_list = []
        for d in ds:
            lf0 = d.lf0.array.astype(numpy.float64)
            silence = d.silence.resample(d.lf0.rate)
            length = min(len(lf0), len(silence))
            notsilence_range = get_notsilence_range(
                silence=silence[:length],
                prepost_silence_length=prepost_silence_length,
            )
            lf0_list.append(lf0[notsilence_range])

        lf0 = numpy.concatenate(lf0_list)
        vuv = (lf0 > 0).astype(numpy.float64)
        lf0 = lf0[lf0 > 0]
        statistics.lf0_mean[speaker_id] = lf0.mean()
        statistics.lf0_std[speaker_id] = lf0.std()
        statistics.vuv_mean[speaker_id] = vuv.mean()
        statistics.vuv_std[speaker_id] = vuv.std()
    return statistics


def make_frame_array(values: numpy.ndarray, indexes: numpy.ndarray, max_length: int):
    assert len(values) == len(
        indexes
    ), f"make_frame_array {len(values)} != {len(indexes)}"
    assert indexes[-1] <= max_length, f"make_frame_array {indexes[-1]} <= {max_length}"
    repeats = numpy.diff(indexes.tolist() + [max_length])
    frame_array = numpy.repeat(values, repeats)
    return frame_array


def sigmoid(a: Union[float, numpy.ndarray]):
    return 1 / (1 + numpy.exp(-a))


def preprocess(
    d: DatasetInput,
    statistics: Optional[DatasetStatistics],
    frame_rate: float,
    prepost_silence_length: int,
    max_sampling_length: Optional[int],
):
    # 長さを揃える
    lf0 = d.lf0.resample(frame_rate).astype(numpy.float64)
    silence = d.silence.resample(frame_rate)
    speaker_id = d.speaker_id

    length = min(len(lf0), len(silence))
    assert numpy.abs(len(lf0) - length) < 5, f"len(lf0) {len(lf0)} != {length}"
    assert (
        numpy.abs(len(silence) - length) < 5
    ), f"len(silence) {len(silence)} != {length}"

    lf0 = lf0[:length]
    silence = silence[:length]

    # voiced
    voiced = lf0 != 0
    lf0[~voiced] = numpy.nan

    # フレーム数を計算
    frame_indexes = numpy.round(
        [numpy.rint(p.start * frame_rate) for p in d.phoneme_list]
    ).astype(numpy.int32)

    # 音素レベルをフレームレベルに
    start_accent = make_frame_array(
        values=d.start_accent_list.astype(numpy.int32),
        indexes=frame_indexes,
        max_length=length,
    )
    end_accent = make_frame_array(
        values=d.end_accent_list.astype(numpy.int32),
        indexes=frame_indexes,
        max_length=length,
    )
    start_accent_phrase = make_frame_array(
        values=d.start_accent_phrase_list.astype(numpy.int32),
        indexes=frame_indexes,
        max_length=length,
    )
    end_accent_phrase = make_frame_array(
        values=d.end_accent_phrase_list.astype(numpy.int32),
        indexes=frame_indexes,
        max_length=length,
    )
    phoneme = make_frame_array(
        values=numpy.array([p.phoneme_id for p in d.phoneme_list]).astype(numpy.int32),
        indexes=frame_indexes,
        max_length=length,
    )

    # アクセントをまとめる
    accent = numpy.stack(
        [
            start_accent,
            end_accent,
            start_accent_phrase,
            end_accent_phrase,
        ],
        axis=1,
    ).astype(numpy.float64)

    # 最初と最後の無音を除去する
    notsilence_range = get_notsilence_range(
        silence=silence[:length],
        prepost_silence_length=prepost_silence_length,
    )
    lf0 = lf0[notsilence_range]
    silence = silence[notsilence_range]
    voiced = voiced[notsilence_range]
    accent = accent[notsilence_range]
    phoneme = phoneme[notsilence_range]
    length = len(lf0)

    # サンプリング長調整
    if max_sampling_length is not None and length > max_sampling_length:
        offset = numpy.random.randint(length - max_sampling_length + 1)
        offset_slice = slice(offset, offset + max_sampling_length)
        lf0 = lf0[offset_slice]
        silence = silence[offset_slice]
        voiced = voiced[offset_slice]
        accent = accent[offset_slice]
        phoneme = phoneme[offset_slice]
        length = max_sampling_length

    # 正規化・ノイズ付与
    t = sigmoid(numpy.random.randn())

    target_lf0 = (lf0 - statistics.lf0_mean[speaker_id]) / statistics.lf0_std[
        speaker_id
    ]
    noise_lf0 = numpy.random.randn(*lf0.shape)
    input_lf0 = noise_lf0 + t * (target_lf0 - noise_lf0)
    input_lf0[~voiced] = noise_lf0[~voiced]

    target_vuv = (
        voiced.astype(numpy.float64) - statistics.vuv_mean[speaker_id]
    ) / statistics.vuv_std[speaker_id]
    noise_vuv = numpy.random.randn(*voiced.shape)
    input_vuv = noise_vuv + t * (target_vuv - noise_vuv)

    output_data = DatasetOutput(
        input_lf0=torch.from_numpy(input_lf0.reshape(-1, 1)).float(),
        target_lf0=torch.from_numpy(target_lf0.reshape(-1, 1)).float(),
        noise_lf0=torch.from_numpy(noise_lf0.reshape(-1, 1)).float(),
        input_vuv=torch.from_numpy(input_vuv.reshape(-1, 1)).float(),
        target_vuv=torch.from_numpy(target_vuv.reshape(-1, 1)).float(),
        noise_vuv=torch.from_numpy(noise_vuv.reshape(-1, 1)).float(),
        accent=torch.from_numpy(accent).float(),
        phoneme=torch.from_numpy(phoneme.reshape(-1, 1)),
        voiced=torch.from_numpy(voiced),
        silence=torch.from_numpy(silence),
        speaker_id=torch.tensor(speaker_id),
        t=torch.tensor(t).float(),
    )
    return output_data


class FeatureTargetDataset(Dataset):
    def __init__(
        self,
        datas: list[Union[DatasetInput, LazyDatasetInput]],
        statistics: Optional[DatasetStatistics],
        frame_rate: float,
        prepost_silence_length: int,
        max_sampling_length: Optional[int],
    ):
        self.datas = datas
        self.preprocessor = partial(
            preprocess,
            statistics=statistics,
            frame_rate=frame_rate,
            prepost_silence_length=prepost_silence_length,
            max_sampling_length=max_sampling_length,
        )

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        data = self.datas[i]
        if isinstance(data, LazyDatasetInput):
            data = data.generate()
        return self.preprocessor(data)


def get_datas(config: DatasetFileConfig):
    (
        fn_list,
        lf0_paths,
        start_accent_list_paths,
        end_accent_list_paths,
        start_accent_phrase_list_paths,
        end_accent_phrase_list_paths,
        phoneme_list_paths,
        silence_paths,
    ) = get_stem_to_paths(
        config.lf0_glob,
        config.start_accent_list_glob,
        config.end_accent_list_glob,
        config.start_accent_phrase_list_glob,
        config.end_accent_phrase_list_glob,
        config.phoneme_list_glob,
        config.silence_glob,
    )

    fn_each_speaker: Dict[str, List[str]] = json.loads(
        config.speaker_dict_path.read_text()
    )
    speaker_ids = {
        fn: speaker_id
        for speaker_id, (_, fns) in enumerate(fn_each_speaker.items())
        for fn in fns
    }
    assert set(fn_list).issubset(set(speaker_ids.keys()))

    datas = [
        LazyDatasetInput(
            lf0_path=lf0_paths[fn],
            start_accent_list_path=start_accent_list_paths[fn],
            end_accent_list_path=end_accent_list_paths[fn],
            start_accent_phrase_list_path=start_accent_phrase_list_paths[fn],
            end_accent_phrase_list_path=end_accent_phrase_list_paths[fn],
            phoneme_list_path=phoneme_list_paths[fn],
            silence_path=silence_paths[fn],
            speaker_id=speaker_ids[fn],
        )
        for fn in fn_list
    ]
    return datas


def create_dataset(config: DatasetConfig):
    datas = get_datas(config.train_file)

    with Pool() as pool:
        statistics = calc_statistics(
            list(
                tqdm(
                    pool.imap_unordered(LazyDatasetInput.generate, datas),
                    total=len(datas),
                    desc="calc_statistics",
                )
            ),
            prepost_silence_length=config.prepost_silence_length,
        )
        print(f"statistics: {statistics}")

    if config.seed is not None:
        numpy.random.RandomState(config.seed).shuffle(datas)

    tests, trains = datas[: config.test_num], datas[config.test_num :]

    valids = get_datas(config.valid_file)

    def dataset_wrapper(datas, is_eval: bool):
        dataset = FeatureTargetDataset(
            datas=datas,
            statistics=statistics,
            frame_rate=config.frame_rate,
            prepost_silence_length=config.prepost_silence_length,
            max_sampling_length=(config.max_sampling_length if not is_eval else None),
        )
        return dataset

    return (
        {
            "train": dataset_wrapper(trains, is_eval=False),
            "test": dataset_wrapper(tests, is_eval=False),
            "eval": dataset_wrapper(tests, is_eval=True),
            "valid": dataset_wrapper(valids, is_eval=True),
        },
        statistics,
    )
