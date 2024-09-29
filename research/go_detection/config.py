from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataCfg:
    name: str = "PandaScene"
    base_path: str = ""


@dataclass
class SimCfg:
    data_cfg: DataCfg
    exp_name: str
    exp_dir: str = "/home/rmenon/Desktop/dev/ml_results"
