from dataclasses import dataclass
from functools import partial
from itertools import groupby
from multiprocessing import Pool
from os import PathLike
import numpy
from tqdm import tqdm

from ..data.data import calc_volume, get_notsilence_range
from ..data.sampling_data import SamplingData


@dataclass
class StatisticDataInput:
    lf0_path: PathLike
    wave_path: PathLike
    silence_path: PathLike
    speaker_id: int


@dataclass
class DataStatistics:
    lf0_mean: numpy.ndarray
    lf0_std: numpy.ndarray
    vuv_mean: numpy.ndarray
    vuv_std: numpy.ndarray
    vol_mean: numpy.ndarray
    vol_std: numpy.ndarray


def _trim_prepost_silence(
    array: numpy.ndarray,
    silence_data: SamplingData,
    target_rate: float,
    prepost_silence_length: int,
):
    silence = silence_data.resample(target_rate)
    assert abs(len(array) - len(silence)) < 5, f"{len(array)} != {len(silence)}"
    length = min(len(array), len(silence))
    notsilence_range = get_notsilence_range(
        silence=silence[:length],
        prepost_silence_length=prepost_silence_length,
    )
    return array[notsilence_range]


def _preprocess(d: StatisticDataInput, frame_rate: float, prepost_silence_length: int):
    silence_data = SamplingData.load(d.silence_path)

    lf0_data = SamplingData.load(d.lf0_path)
    lf0 = lf0_data.array.astype(numpy.float64)
    lf0 = _trim_prepost_silence(
        array=lf0,
        silence_data=silence_data,
        target_rate=lf0_data.rate,
        prepost_silence_length=prepost_silence_length,
    )

    wave_data = SamplingData.load(d.wave_path)
    vol = calc_volume(wave_data, frame_rate=frame_rate).astype(numpy.float64)
    vol = _trim_prepost_silence(
        array=vol,
        silence_data=silence_data,
        target_rate=frame_rate,
        prepost_silence_length=prepost_silence_length,
    )

    return {"lf0": lf0, "vol": vol}


def calc_statistics(
    ds: list[StatisticDataInput], frame_rate: float, prepost_silence_length: int
) -> DataStatistics:
    """話者ごとの統計情報を取得"""
    max_speaker_id = max(d.speaker_id for d in ds)
    statistics = DataStatistics(
        lf0_mean=numpy.full(max_speaker_id + 1, numpy.nan, dtype=numpy.float64),
        lf0_std=numpy.full(max_speaker_id + 1, numpy.nan, dtype=numpy.float64),
        vuv_mean=numpy.full(max_speaker_id + 1, numpy.nan, dtype=numpy.float64),
        vuv_std=numpy.full(max_speaker_id + 1, numpy.nan, dtype=numpy.float64),
        vol_mean=numpy.full(max_speaker_id + 1, numpy.nan, dtype=numpy.float64),
        vol_std=numpy.full(max_speaker_id + 1, numpy.nan, dtype=numpy.float64),
    )

    datas_dict = groupby(
        sorted(ds, key=lambda d: d.speaker_id), key=lambda d: d.speaker_id
    )

    pool = Pool()
    preprocess = partial(
        _preprocess,
        frame_rate=frame_rate,
        prepost_silence_length=prepost_silence_length,
    )
    for speaker_id, ds in datas_dict:
        lf0_list = []
        vol_list = []
        it = tqdm(
            pool.imap_unordered(preprocess, ds, chunksize=16),
            desc=f"calc_statistics preprocess: speaker_id: {speaker_id}",
        )
        for d in it:
            lf0_list.append(d["lf0"])
            vol_list.append(d["vol"])

        lf0 = numpy.concatenate(lf0_list)
        vuv = (lf0 > 0).astype(numpy.float64)
        lf0 = lf0[lf0 > 0]
        statistics.lf0_mean[speaker_id] = lf0.mean()
        statistics.lf0_std[speaker_id] = lf0.std()
        statistics.vuv_mean[speaker_id] = vuv.mean()
        statistics.vuv_std[speaker_id] = vuv.std()

        vol = numpy.concatenate(vol_list)
        statistics.vol_mean[speaker_id] = vol.mean()
        statistics.vol_std[speaker_id] = vol.std()

    pool.close()
    return statistics
