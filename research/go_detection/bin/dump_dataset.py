import logging
import os
import time
from os import path
from typing import cast

import debugpy
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from go_detection.common.asset_io import AssetIO
from go_detection.common.matplotlib_utils import draw_histogram
from go_detection.config import DumpDatasetCfg, SimCfg
from go_detection.dataloader import (
    DataPoint,
    DataPoints,
    DistSampler,
    NonReplacementSampler,
    UniformSampler,
    _load_single,
    create_datasets,
    get_all_datapoints,
)
from go_detection.dataloader_viz import visualize_accuracy_over_num_pieces
from go_detection.trainer import GoTrainer
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _convert_label_to_text(label):
    lines = []
    for line_idx in range(label.shape[0]):
        line = []
        for ch_idx in range(label.shape[1]):
            if label[line_idx, ch_idx] == 0:
                line.append("B")
            elif label[line_idx, ch_idx] == 1:
                line.append(".")
            elif label[line_idx, ch_idx] == 2:
                line.append("W")
            else:
                assert False, f"Unknown label: {label[line_idx, ch_idx]}"

            line.append(" ")

        line.append("\n")
        lines.append("".join(line))

    return lines


def _convert_board_pts_to_np(board_pt, image):
    board_pt_scaled = board_pt * torch.tensor([image.shape[2], image.shape[1]])
    return board_pt_scaled.tolist()


def do_main(cfg: DumpDatasetCfg):
    all_datapoint_paths = get_all_datapoints(cfg.base_path)

    out_io = AssetIO(cfg.output_path)
    out_io.mkdir()
    out_io.save_yaml(
        "dataset_info.yaml",
        {
            "version": 1,
            "has_pt_files": True,
        },
    )
    for board_idx, list_images in tqdm(
        enumerate(all_datapoint_paths),
        desc="Dumping all boards",
        total=len(all_datapoint_paths),
        colour="red",
    ):
        for image_idx, data_point_path in tqdm(
            enumerate(list_images),
            desc=f"Dumping board {board_idx:02d}",
            total=len(list_images),
        ):
            image_dir = out_io.cd(
                path.join(f"board_{board_idx:03d}", f"image_{image_idx:03d}")
            )
            image_dir.mkdir()
            datapoint = _load_single(data_point_path, AssetIO(cfg.base_path))
            # Save the pt files
            image_dir.save_torch("image.pt", datapoint.image)
            image_dir.save_torch("label.pt", datapoint.label)
            image_dir.save_torch("board_info.pt", datapoint.board_pt)

            # Save human readable files
            image_dir.save_image("raw_image.png", datapoint.image)
            image_dir.save_text(
                "raw_label.txt", _convert_label_to_text(datapoint.label)
            )
            image_dir.save_yaml(
                "raw_board_info.yaml",
                {
                    "pts_clicks": _convert_board_pts_to_np(
                        datapoint.board_pt, datapoint.image
                    )
                },
            )

    print("a")


@hydra.main(config_path="../config", config_name="dump_dataset", version_base="1.2")
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
    cs.store(name="dump_dataset_default", node=DumpDatasetCfg)

    if os.environ.get("ENABLE_DEBUGPY"):
        print("")
        print("\033[31mWaiting for debugger to connect\033[0m")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    main()
