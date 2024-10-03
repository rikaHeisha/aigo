import logging
import os
import time
from os import path
from typing import Dict, Tuple, TypeVar, Union, cast

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

MetricPrimitive = Union[float, torch.Tensor, int]
MetricValue = Union[MetricPrimitive, Tuple[MetricPrimitive, float]]


# For debugging
# def _optimizer_get_devices(optim):
#     devices = []
#     for param in optim.state.values():
#         # Not sure there are any global tensors in the state dict
#         if isinstance(param, torch.Tensor):
#             devices.append(param.data.device)
#             if param._grad is not None:
#                 print("Has global grad")
#                 devices.append(param._grad.data.device)
#         elif isinstance(param, dict):
#             for subparam in param.values():
#                 if isinstance(subparam, torch.Tensor):
#                     devices.append(subparam.data.device)
#                     if subparam._grad is not None:
#                         print("Has grad")
#                         devices.append(subparam._grad.data.device)
#     return devices


# def optimizer_to_device(optim, device):
#     # See: https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/2
#     for param in optim.state.values():
#         # Not sure there are any global tensors in the state dict
#         if isinstance(param, torch.Tensor):
#             param.data = param.data.to(device)
#             if param._grad is not None:
#                 param._grad.data = param._grad.data.to(device)
#         elif isinstance(param, dict):
#             for subparam in param.values():
#                 if isinstance(subparam, torch.Tensor):
#                     subparam.data = subparam.data.to(device)
#                     if subparam._grad is not None:
#                         subparam._grad.data = subparam._grad.data.to(device)


class GoTrainer:
    def __init__(self, cfg: SimCfg):
        super().__init__()
        self.cfg = cfg

        # Create the tf writer
        self.results_io = cfg.result_cfg.get_asset_io()
        self.results_io.mkdir("tf")
        run_number = (
            len([_ for _ in self.results_io.ls("tf") if self.results_io.has_dir(_)]) + 1
        )

        tf_path = path.join("tf", f"run_{run_number}")
        assert self.results_io.has(tf_path) == False
        self.results_io.mkdir(tf_path)
        self.tf_writer = SummaryWriter(self.results_io.get_abs(tf_path))

        # Create the model and optimizer
        self.iter = 1
        self.model = GoModel(self.cfg.model_cfg)
        self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(self.get_parameters(), lr=1e-4)

        # Create the dataloader and load checkpoint
        self.train_dataloader, self.test_dataloader = self._load_or_create_dataloader()
        self.load_checkpoint()

        # Create losses
        self.nll_loss = nn.NLLLoss(reduction="mean")

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

        if self.results_io.has_file("dataset_split.pt"):
            # Load dataset
            logger.info("Loading dataset split from file")
            torch.serialization.add_safe_globals([DataPointPath])
            dataset_split = self.results_io.load_torch("dataset_split.pt")

            train_dataloader, test_dataloader = load_datasets(
                self.cfg.data_cfg, dataset_split["train"], dataset_split["test"]
            )
        else:
            logger.info("Creating dataset split")
            train_dataloader, test_dataloader = create_datasets(self.cfg.data_cfg)

            self.results_io.save_torch(
                "dataset_split.pt",
                {
                    "train": train_dataloader.dataset.datapoint_paths,
                    "test": test_dataloader.dataset.datapoint_paths,
                },
            )

        # Print dataset info
        num_train = len(train_dataloader.dataset)
        num_test = len(test_dataloader.dataset)
        percent_train = 100.0 * num_train / (num_train + num_test)
        percent_test = 100.0 * num_test / (num_train + num_test)
        logger.info(
            f"Loaded datasets: Train: {num_train} ({percent_train:.1f} %), Test: {num_test} ({percent_test:.1f} %)"
        )
        return train_dataloader, test_dataloader

    def _upload_metrics_to_tf(
        self,
        step,
        map_metrics: Dict[str, MetricValue],
        prefix: str = "",
    ):
        metric_time = time.time()
        for key, metric in map_metrics.items():
            metric_name = f"{prefix}{key}"

            if isinstance(metric, MetricPrimitive):
                self.tf_writer.add_scalar(metric_name, metric, step, metric_time)
            else:
                assert isinstance(metric, tuple)
                self.tf_writer.add_scalar(metric_name, metric[0], step, metric_time)
                self.tf_writer.add_scalar(
                    f"{metric_name}_weighted", metric[0] * metric[1], step, metric_time
                )

        # TODO(rishi): See if this is necessary or not
        self.tf_writer.flush()

    def load_checkpoint(self):
        # TODO(rishi) add a load config
        if not self.results_io.has_file("checkpoint.pt"):
            return

        checkpoint_data = self.results_io.load_torch("checkpoint.pt")

        cur_iter = cast(int, checkpoint_data["cur_iter"])
        self.iter = cur_iter + 1
        self.model.load_state_dict(checkpoint_data["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        logging.info(
            f"Loaded checkpoint at iter {cur_iter}. Resuming from iter {self.iter}"
        )

        # If you face device issues then try uncommentings these lines
        # self.model = self.model.cuda()
        # _optimizer_to_device(self.optimizer, "cuda")

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

    def evaluate(
        self, path: str, render_corner_points: bool = True, render_labels: bool = True
    ):
        pass  # TODO(rishi) do this tomorrow

    def start(self):
        logging.info("Starting training")

        while self.iter <= self.cfg.iters:
            output_map = self.train_step()

            if self.iter % self.cfg.i_print == 0:
                logger.info(
                    f'Exp: {self.cfg.result_cfg.name}, Iter: {self.iter} / {self.cfg.iters}, Memory: {output_map["memory"]: .2f} GB, total_loss: {output_map["total_loss"]:.4f}'
                )

            if self.iter % self.cfg.i_weight == 0:
                logger.info("Starting saving checkpoint")
                self.save_checkpoint()

            if self.iter % self.cfg.i_tf_writer == 0:
                # Upload the output_map of the last mini batch
                logger.info("Uploading epoch to tensorboard")
                self._upload_metrics_to_tf(self.iter, output_map, "epoch__")

            if self.iter % self.cfg.i_eval == 0:
                logger.info("Starting evaluation")

            self.iter += 1

        logging.info("Finished training")
        self.tf_writer.flush()

    def train_step(self):
        # Iterate through all the training datasets
        output_map: Dict[str, MetricValue] = {}
        self.model.train()

        for idx, datapoints in enumerate(self.train_dataloader, 1):
            # Reset output map every iteration
            output_map = {}

            num_images = datapoints.images.shape[0]

            datapoints = DataPoints(
                datapoints[0].cuda(),
                datapoints[1].cuda(),
                datapoints[2].cuda(),
            )
            output = self.model(datapoints.images)
            gt_labels = datapoints.labels.reshape(num_images, -1)

            # Check that the shape matches what we are giving to self.nll_loss
            assert (
                gt_labels.shape[1:] == output.shape[2:]
                and gt_labels.shape[0] == num_images
                and output.shape[0] == num_images
            )
            assert gt_labels.max() < output.shape[1] and gt_labels.min() >= 0
            nll_loss = self.nll_loss(output, gt_labels)

            # Manuall calculate the NLL loss
            # target_label_one_hot = torch.nn.functional.one_hot(gt_labels, output.shape[1])
            # nll_loss_2 = (-output * target_label_one_hot.transpose(1,2)).sum(dim=1).mean()

            output_map["nll_loss"] = (nll_loss, 100.0)

            #########################
            # Calculate total loss
            #########################
            total_loss = torch.tensor(0.0).cuda()
            for key, value in output_map.items():
                if "loss" in key:
                    assert isinstance(value, tuple)
                    total_loss = total_loss + value[0] * value[1]

            output_map["total_loss"] = total_loss

            memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            output_map["memory"] = memory_gb

            # Use print so this does not end up in the logs
            print(
                f"Exp: {self.cfg.result_cfg.name}, Step: {self.iter}-{idx} / {self.cfg.iters}, Memory: {memory_gb: .2f} GB, total_loss: {total_loss:.4f}"
            )

            mini_batch_idx = (self.iter - 1) * len(self.train_dataloader) + idx
            self._upload_metrics_to_tf(
                mini_batch_idx, output_map, "batch__"
            )  # TODO(rishi): add a config for this so we dont push every tick?

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            self.optimizer.step()

        # Completed one epoch
        return output_map
