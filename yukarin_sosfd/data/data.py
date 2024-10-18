import librosa
import numpy

from ..data.sampling_data import SamplingData


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


def calc_volume(wave_data: SamplingData, frame_rate: float):
    assert wave_data.rate == 24000
    assert wave_data.array.ndim == 1 or wave_data.array.shape[1] == 1
    wave = wave_data.array[:, 0]

    hop_length = int(24000 / frame_rate)
    frame_length = hop_length * 4

    wave = numpy.pad(
        wave,
        (int((frame_length - hop_length) / 2), int((frame_length - hop_length) / 2)),
        mode="reflect",
    )
    volume = librosa.feature.rms(
        y=wave,
        frame_length=frame_length,
        hop_length=hop_length,
        center=False,
        pad_mode="reflect",
    ).squeeze(0)
    return volume
