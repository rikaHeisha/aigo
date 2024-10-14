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

    def draw_hist_with_sampler(sampler, fig_path, num_samples):
        data = []
        for _ in range(num_samples):
            idx = next(sampler_iter)
            data.append(train_dataset.num_pieces[idx].item())
        data = np.array(data)
        draw_histogram(
            data,
            fig_path,
            bins=data.max(),
        )

    sampler_iter = iter(NonReplacementSampler(len(train_dataset), False))
    draw_hist_with_sampler(
        sampler_iter,
        "/home/rmenon/Desktop/dev/projects/aigo/research/rishi.png",
        len(train_dataset),
    )

    sampler_iter = iter(UniformSampler(len(train_dataset), 1000))
    draw_hist_with_sampler(
        sampler_iter,
        "/home/rmenon/Desktop/dev/projects/aigo/research/rishi_2.png",
        10**6,
    )

    # t = np.arange(0.0, 2.0, 0.01)

    # data = [
    #     (5, 0.7),
    #     (5, 0.6),
    #     (1, 0.9),
    #     (1, 0.8),
    #     (3, 0.1),
    #     (2, 0.2),
    #     (2, 0.3),
    #     (4, 0.5),
    # ]
    # visualize_accuracy_over_num_pieces(
    #     data, "/home/rmenon/Desktop/dev/projects/aigo/test.png"
    # )

    # # plt.show()
    # #######################
    # weights = np.array([1, 2, 2, 4])
    # pmf = weights / weights.sum()

    # sampler = DistSampler(pmf, 3)
    # sampler_iter = iter(sampler)

    # indices = []
    # for _ in range(1000):
    #     idx = next(sampler_iter)
    #     indices.append(idx)

    # print(f"First few indices: {indices[:10]}")
    # draw_histogram(
    #     indices,
    #     "/home/rmenon/Desktop/dev/projects/aigo/research/rishi.png",
    #     bins=len(pmf),
    # )


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
