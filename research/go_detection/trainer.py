import logging
import os

import numpy as np
import torch
from go_detection.common.asset_io import AssetIO
from go_detection.config import DataCfg, SimCfg
from go_detection.dataloader import (
    DataPointPath,
    create_datasets,
    load_datasets,
    visualize_datapoints,
    visualize_single_datapoint,
)
from go_detection.model import GoModel
from torch import nn
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class GoTrainer:
    def __init__(self, cfg: SimCfg):
        super().__init__()
        self.cfg = cfg

        # Create the tf writer
        self.results_io = cfg.result_cfg.get_asset_io()
        self.results_io.mkdir("tf")
        self.tf_writer = SummaryWriter(self.results_io.get_abs("tf"))

        # Create the model
        self.model = GoModel(self.cfg.model_cfg)
        # self.optimizer = torch.optim.SGD(self.get_parameters(), lr=1e-6)

        self.train_dataloader, self.test_dataloader = self._load_or_create_dataloader()
        # for i in range(400):
        #     self.tf_writer.add_scalar(
        #         "loss/mse", 1.0 * np.sin(i / 100.0 * 8 * np.pi), i
        #     )

        #     if i % 10 == 0:
        #         self.tf_writer.add_scalar(
        #             "loss/mse_2", 0.7 * np.sin(i / 100.0 * 8 * np.pi), i
        #         )

        # data_point = next(iter(self.train_dataloader))
        # visualize_datapoints(
        #     data_point,
        #     "/home/rmenon/Desktop/dev/projects/aigo/research/rishi.png",
        #     max_viz_images=2,
        # )
        # visualize_single_datapoint(
        #     data_point,
        #     "/home/rmenon/Desktop/dev/projects/aigo/research/rishi.png",
        #     0,
        # )

    def _load_or_create_dataloader(self):
        logger.info("Loading dataset")

        if self.results_io.is_file("dataset_split.pt"):
            # Load dataset
            torch.serialization.add_safe_globals([DataPointPath])
            dataset_split = self.results_io.load_torch("dataset_split.pt")

            train_dataloader, test_dataloader = load_datasets(
                self.cfg.data_cfg, dataset_split["train"], dataset_split["test"]
            )
        else:
            train_dataloader, test_dataloader = create_datasets(self.cfg.data_cfg)

            self.results_io.save_torch(
                "dataset_split.pt",
                {
                    "train": train_dataloader.dataset.datapoint_paths,
                    "test": test_dataloader.dataset.datapoint_paths,
                },
            )

            # torch.serialization.add_safe_globals([DataPointPath])
            # dataset_split = self.results_io.load_torch("dataset_split.pt")
            # assert dataset_split["train"] == train_dataloader.dataset.datapoint_paths
            # assert dataset_split["test"] == test_dataloader.dataset.datapoint_paths

        # Print dataset info
        num_train = len(train_dataloader.dataset)
        num_test = len(test_dataloader.dataset)
        percent_train = 100.0 * num_train / (num_train + num_test)
        percent_test = 100.0 * num_test / (num_train + num_test)
        logger.info(
            f"Loaded datasets: Train: {num_train} ({percent_train:.1f} %), Test: {num_test} ({percent_test:.1f} %)"
        )
        return train_dataloader, test_dataloader

    def get_parameters(self):
        return self.model.parameters()

    def start(self):
        pass

    def step(self):
        pass
