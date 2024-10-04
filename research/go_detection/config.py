import os
from dataclasses import dataclass, field
from typing import List, Optional, cast

from go_detection.common.asset_io import AssetIO


# Fully qaualified class name can be obtained by: <class>.__module__ + "." + <class>.__qualname__
@dataclass
class DataCfg:
    base_path: str
    train_split_percent: float = 0.8
    randomize_train_split: bool = True
    use_dynamic_dataset: bool = True
    train_batch_size: int = 4
    test_batch_size: int = 1

    _target_: str = f"{__module__}.{__qualname__}"

    def get_asset_io(self) -> AssetIO:
        return AssetIO(self.base_path)


@dataclass
class EvalCfg:
    render_grid: bool = True

    render_index: List = field(default_factory=list)

    _target_: str = f"{__module__}.{__qualname__}"


@dataclass
class ResultCfg:
    dir: str
    eval_cfg: EvalCfg

    name: str = "${exp_name}"
    _target_: str = f"{__module__}.{__qualname__}"

    def get_asset_io(self) -> AssetIO:
        return AssetIO(os.path.join(self.dir, self.name))


@dataclass
class ModelCfg:
    _target_: str = f"{__module__}.{__qualname__}"


@dataclass
class SimCfg:
    data_cfg: DataCfg
    model_cfg: ModelCfg
    result_cfg: ResultCfg

    exp_name: str
    iters: int
    i_eval: int
    i_weight: int
    i_print: int
    i_tf_writer: int = cast(int, "${i_print}")  # TODO(rishi) move this to a tf_config
    _target_: str = f"{__module__}.{__qualname__}"
