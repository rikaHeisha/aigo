import os

import numpy as np
import torch
from go_detection.common.asset_io import AssetIO
from go_detection.config import DataCfg, SimCfg
from go_detection.dataloader import create_datasets, visualize_datapoints
from go_detection.model import GoModel
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class GoTrainer:
    def __init__(self, cfg: SimCfg):
        super().__init__()
        self.cfg = cfg

        # TODO(rishi): If we are loading a checkpoint, then load the same datasets again. This is so that we don't mix training and test datasets
        self.train_dataloader, self.test_dataloader = create_datasets(cfg.data_cfg)
        self.results_io = AssetIO(os.path.join(cfg.result_cfg.dir, cfg.result_cfg.name))
        self.results_io.mkdir("tf")
        # Run: tensorboard --logdir=./tf
        self.tf_writer = SummaryWriter(self.results_io.get_abs("tf"))

        # for i in range(400):
        #     self.tf_writer.add_scalar(
        #         "loss/mse", 1.0 * np.sin(i / 100.0 * 8 * np.pi), i
        #     )

        #     if i % 10 == 0:
        #         self.tf_writer.add_scalar(
        #             "loss/mse_2", 0.7 * np.sin(i / 100.0 * 8 * np.pi), i
        #         )

        self.model = GoModel(self.cfg.model_cfg)

        data_point = next(iter(self.train_dataloader))
        visualize_datapoints(
            data_point, "/home/rmenon/Desktop/dev/projects/aigo/research/rishi.png"
        )

    def start(self):
        pass
