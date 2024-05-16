import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from yukarin_sosfd.utility import dataclass_utility
from yukarin_sosfd.utility.git_utility import get_branch_name, get_commit_id


@dataclass
class DatasetFileConfig:
    lf0_glob: str
    start_accent_list_glob: str
    end_accent_list_glob: str
    start_accent_phrase_list_glob: str
    end_accent_phrase_list_glob: str
    phoneme_list_glob: str
    silence_glob: str
    speaker_dict_path: Path


@dataclass
class DatasetConfig:
    train_file: DatasetFileConfig
    valid_file: DatasetFileConfig
    frame_rate: float
    prepost_silence_length: int
    max_sampling_length: Optional[int]
    test_num: int
    seed: int = 0


@dataclass
class NetworkConfig:
    speaker_size: int
    speaker_embedding_size: int
    phoneme_size: int
    phoneme_embedding_size: int
    hidden_size: int
    block_num: int
    dropout_rate: float
    positional_dropout_rate: float
    attention_dropout_rate: float
    use_conv_glu_module: bool
    with_condition_t: bool


@dataclass
class ModelConfig:
    pass


@dataclass
class TrainConfig:
    diffusion_step_num: int
    batch_size: int
    eval_batch_size: int
    log_epoch: int
    eval_epoch: int
    snapshot_epoch: int
    stop_epoch: int
    model_save_num: int
    optimizer: Dict[str, Any]
    scheduler: Dict[str, Any]
    weight_initializer: Optional[str] = None
    pretrained_predictor_path: Optional[Path] = None
    num_processes: int = 4
    use_gpu: bool = True
    use_amp: bool = True


@dataclass
class ProjectConfig:
    name: str
    tags: Dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None


@dataclass
class Config:
    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        backward_compatible(d)
        return dataclass_utility.convert_from_dict(cls, copy.deepcopy(d))

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_utility.convert_to_dict(self)

    def add_git_info(self):
        self.project.tags["git-commit-id"] = get_commit_id()
        self.project.tags["git-branch-name"] = get_branch_name()


def backward_compatible(d: Dict[str, Any]):
    if "dropout_rate" not in d["network"]:
        d["network"]["dropout_rate"] = 0.2
    if "positional_dropout_rate" not in d["network"]:
        d["network"]["positional_dropout_rate"] = 0.2
    if "attention_dropout_rate" not in d["network"]:
        d["network"]["attention_dropout_rate"] = 0.2

    if "experimental_use_myconformer" in d["network"]:
        assert (
            d["network"]["experimental_use_myconformer"] is True
        ), "experimental_use_myconformer must be True."
        del d["network"]["experimental_use_myconformer"]
    if "concat_after" in d["network"]:
        assert d["network"]["concat_after"] is False, "concat_after must be False."
        del d["network"]["concat_after"]
    if "post_layer_num" in d["network"]:
        assert d["network"]["post_layer_num"] == 0, "post_layer_num must be 0."
        del d["network"]["post_layer_num"]

    if "use_conv_glu_module" not in d["network"]:
        d["network"]["use_conv_glu_module"] = True
    if "with_condition_t" not in d["network"]:
        d["network"]["with_condition_t"] = False
