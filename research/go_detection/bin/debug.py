import logging
import os
from os import path
from typing import cast

import debugpy
import hydra
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from go_detection.common.asset_io import AssetIO
from go_detection.common.matplotlib_utils import draw_histogram
from go_detection.config import SimCfg
from go_detection.dataloader import (
    DataPoint,
    DataPoints,
    DistSampler,
    NonReplacementSampler,
    UniformSampler,
    create_datasets,
)
from go_detection.dataloader_viz import visualize_accuracy_over_num_pieces
from go_detection.trainer import GoTrainer
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def do_main(cfg: SimCfg):
    train, test = create_datasets(cfg.data_cfg)
    train_dataset, test_dataset = train.dataset, test.dataset

    # base_path = (
    #     "/home/rmenon/Desktop/dev/ml_results/aigo_results/go2_basic_adam_1016/results/"
    # )
    # # full_eval/test/plot_accuracy.png
    # paths = [
    #     "full_eval",
    #     "iter_02000",
    #     "iter_04000",
    #     "iter_06000",
    #     "iter_08000",
    # ]
    # for path_iter in paths:
    #     for eval_type in ["train", "test"]:
    #         aio = AssetIO(os.path.join(base_path, path_iter, eval_type))
    #         report_data = aio.load_yaml("report.yaml")

    #         # points = [
    #         #     (v["num_pieces"], float(v["accuracy"].partition(" %")[0]) / 100.0)
    #         #     for k, v in report_data.items()
    #         #     if k != "overall"
    #         # ]
    #         points = []
    #         for k, v in report_data.items():
    #             if k != "overall":
    #                 points.append(
    #                     (
    #                         v["num_pieces"],
    #                         float(v["accuracy"].partition(" %")[0]) / 100.0,
    #                     )
    #                 )

    #         visualize_accuracy_over_num_pieces(points, aio.get_abs("plot_accuracy.png"))


@hydra.main(config_path="../config", config_name="basic", version_base="1.2")
def main(cfg):
    OmegaConf.set_readonly(cfg, True)
    cfg = instantiate(cfg)  # Converts the DictConfig to native python classes

    cfg_yaml = OmegaConf.to_yaml(cfg)
    logger.info("Config:\n%s", cfg_yaml)

    do_main(cfg)


# def common_main(task_function: Callable[[], None]) -> Callable[[], None]:
#     @functools.wraps(task_function)
#     def decorator():
#         cs = ConfigStore.instance()
#         cs.store(name="sim_cfg_default", node=SimCfg)

#         if os.environ.get("ENABLE_DEBUGPY"):
#             print("")
#             print("\033[31mWaiting for debugger to connect\033[0m")
#             debugpy.listen(5678)
#             debugpy.wait_for_client()

#         # @hydra.main(config_path="../config", config_name="basic", version_base="1.2")
#         with initialize(version_base="1.2", config_path="../config", job_name=None):
#             overrides = sys.argv[1:]
#             cfg = compose(config_name="basic", overrides=overrides)
#             cfg = instantiate(cfg)

#         task_function(cfg)

#     return decorator

if __name__ == "__main__":
    # Register configs
    cs = ConfigStore.instance()
    cs.store(name="sim_cfg_default", node=SimCfg)

    if os.environ.get("ENABLE_DEBUGPY"):
        print("")
        print("\033[31mWaiting for debugger to connect\033[0m")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    main()
