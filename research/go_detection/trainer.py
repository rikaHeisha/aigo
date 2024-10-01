import logging
import os
from typing import cast

import numpy as np
import torch
from go_detection.common.asset_io import AssetIO
from go_detection.config import DataCfg, SimCfg
from go_detection.dataloader import (
    DataPointPath,
    DataPoints,
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

        # Create the model and optimizer
        self.iter = 1
        self.model = GoModel(self.cfg.model_cfg)
        self.optimizer = torch.optim.SGD(self.get_parameters(), lr=1e-6)

        # Create the dataloader and load checkpoint
        self.train_dataloader, self.test_dataloader = self._load_or_create_dataloader()
        self.load_checkpoint()

        # Create losses
        self.nll_loss = nn.NLLLoss()

        self.model = self.model.cuda()

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

        if self.results_io.has_file("dataset_split.pt"):
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

    def load_checkpoint(self):
        if not self.results_io.has_file("checkpoint.pt"):
            return

        checkpoint_data = self.results_io.load_torch("checkpoint.pt")
        self.iter = cast(int, checkpoint_data["cur_iter"])
        self.model.load_state_dict(checkpoint_data["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        logging.info(f"Loaded checkpoint at iter: {self.iter}")

    def save_checkpoint(self):
        checkpoint_data = {
            "cur_iter": self.iter,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        logger.info(f"Saving checkpoint at iter: {self.iter}")
        self.results_io.save_torch("checkpoint.pt", checkpoint_data)

    def get_parameters(self):
        parameters = self.model.parameters()
        return parameters

    def start(self):
        logging.info("Starting training")

        while self.iter < self.cfg.iters:
            self.step()

            if self.iter % self.cfg.i_weight:
                pass

            self.iter += 1

    def step(self):

        # Iterate through all the training datasets
        loss_map = {}
        for idx, datapoints in enumerate(self.train_dataloader):
            loss_map = {}

            datapoints = DataPoints(
                datapoints[0].cuda(),
                datapoints[1].cuda(),
                datapoints[2].cuda(),
            )
            output = self.model(datapoints.images)

            # assert datapoints.labels.max() < output.shape[1] and datapoints.labels.min() >= 0
            nll_loss = self.nll_loss(output, datapoints.labels)
            loss_map["loss_nll"] = (nll_loss, 100.0)

            # Use print so this does not end up in the logs
            total_loss = torch.tensor(0.0).cuda()
            for key, value in loss_map.items():
                if "loss" in key:
                    total_loss = total_loss + value[0] * value[1]

            loss_map["loss_total"] = (total_loss, 1.0)

            memory_gb = torch.cuda.max_memory_allocated() / 1024**3

            print(
                f"Exp: {self.cfg.result_cfg.name} Step: {self.iter}-{idx} / {self.cfg.iters}, Memory: {memory_gb}, total_loss: {total_loss:.4f}"
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        if self.iter % self.cfg.i_print:
            logger.info(
                f"Iter: [{self.iter} out of {self.cfg.iters}] - total_loss: {total_loss:.4f}"
            )

        if self.iter % self.cfg.i_tf_writer:
            pass

        if self.iter % self.cfg.i_eval:
            pass
