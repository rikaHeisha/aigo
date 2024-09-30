from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataCfg:
    base_path: str
    train_split_percent: float = 0.8
    randomize_train_split: bool = True
    use_dynamic_dataset: bool = True


@dataclass
class ResultCfg:
    exp_name: str
    dir: str


@dataclass
class SimCfg:
    data_cfg: DataCfg
    result_cfg: ResultCfg
