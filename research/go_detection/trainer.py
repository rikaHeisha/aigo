import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from os import mkdir, path
from typing import Dict, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import torch
from go_detection.common.asset_io import AssetIO
from go_detection.config import DataCfg, SimCfg
from go_detection.dataloader import (
    DataPoint,
    DataPointPath,
    DataPoints,
    GoDataset,
    GoDynamicDataset,
    create_datasets,
    load_datasets,
)
from go_detection.dataloader_viz import visualize_grid
from go_detection.model import GoModel
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MetricValue:
    def __init__(
        self, base_value: Union[float, torch.Tensor, int], weight: float | None = None
    ):
        if isinstance(base_value, torch.Tensor):
            self.base_value = base_value
        else:
            self.base_value = torch.tensor(base_value)

        self.weight = weight

    def has_weight(self):
        return self.weight is not None

    @property
    def weighted_value(self):
        if self.weight is None:
            return self.base_value
        else:
            return self.base_value * self.weight

    def __str__(self):
        return f"MetricValue:\n  base_value: {self.base_value}\n  weight: {self.weight or 'None'}"

    def __repr__(self):
        return f"MetricValue:\n  base_value: {self.base_value}\n  weight: {self.weight or 'None'}"

    def __format__(self, spec):
        if self.weight is None:
            return f"{self.base_value:{spec}}"
        else:
            return f"({self.base_value:{spec}}, {self.weight:{spec}})"


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

            if not metric.has_weight():
                self.tf_writer.add_scalar(
                    metric_name, metric.base_value, step, metric_time
                )
            else:
                self.tf_writer.add_scalar(
                    metric_name, metric.base_value, step, metric_time
                )
                self.tf_writer.add_scalar(
                    f"{metric_name}_weighted",
                    metric.weighted_value,
                    step,
                    metric_time,
                )

        # TODO(rishi): See if this is necessary or not
        self.tf_writer.flush()

    def load_checkpoint(self):
        # TODO(rishi) add a config for loading checkpoints
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

    def _convert_render_dir_to_indices(
        self,
        datapoint_paths: List[DataPointPath],
        render_dirs: List[int],
        images_per_dir: Optional[int],
    ):
        map_dir_name_to_indexes = defaultdict(list)
        for idx, datapoint_path in enumerate(datapoint_paths):
            dir_name = datapoint_path.image_path.split("/", 1)[0]
            map_dir_name_to_indexes[dir_name].append(idx)

        list_dirs = [
            dir_name
            for idx, dir_name in enumerate(map_dir_name_to_indexes.keys())
            if render_dirs == [] or idx in render_dirs
        ]
        [map_dir_name_to_indexes[dir_name] for dir_name in list_dirs]
        indices = []
        for dir_name in list_dirs:
            indices.extend(map_dir_name_to_indexes[dir_name][:images_per_dir])

        return indices

    def evaluate(
        self,
        evaluate_path: str,
        render_dirs: List[int],
        images_per_dir: Optional[int],
        render_grid: bool = True,
    ):
        eval_io = self.results_io.cd(evaluate_path)
        eval_io.mkdir("render_grid")

        # indices = render_index or list(range(len(self.test_dataloader.dataset)))
        # indices = [_ for _ in indices if _ < len(self.test_dataloader.dataset)]

        # TODO(rishi): Use the dataloader instead of directly accessing the dataset. For this, we need to get rid of the render_index parameter and find a better config that the user can set
        # Actually a great way of doing this will be to iterate through the whole dataset and selectively rendering. This is a good idea since we anyway have to iterate through the entire thing

        report_data = {}
        datapoint_paths = cast(
            GoDynamicDataset | GoDataset, self.test_dataloader.dataset
        ).datapoint_paths

        # Convert render_dirs and images_per_dir to a list of indices
        indices = self._convert_render_dir_to_indices(
            datapoint_paths, render_dirs, images_per_dir
        )

        self.model.eval()
        with torch.no_grad():
            # TODO(rishi) support multiple images at once
            list_accuracy = []
            for idx in tqdm(indices, desc=f"Rendering for iter {self.iter}"):
                assert idx < len(self.test_dataloader.dataset)

                data_point = cast(DataPoint, self.test_dataloader.dataset[idx])
                data_points = data_point.to_data_points().cuda()

                output = self.model(data_points.images)

                (_, predicted_label) = output[0].max(dim=0)
                predicted_label = predicted_label.reshape(data_points.labels[0].shape)
                gt_label = data_points.labels[0]

                num_correct = (gt_label == predicted_label).sum().item()
                num_incorrect = (gt_label != predicted_label).sum().item()

                image_name = f"image_{idx:04}"
                if render_grid:
                    img_path = eval_io.get_abs(
                        path.join("render_grid", f"{image_name}.png"),
                    )
                    visualize_grid(data_points, img_path, 0, predicted_label)

                # Append this image to the report.yaml
                accuracy = num_correct / (num_correct + num_incorrect)
                list_accuracy.append(accuracy)

                report_data[image_name] = {}
                report_data[image_name]["path"] = datapoint_paths[idx].image_path
                report_data[image_name]["num_correct"] = num_correct
                report_data[image_name]["num_incorrect"] = num_incorrect
                report_data[image_name]["accuracy"] = f"{100 * accuracy:.4f} %"

            report_data["overall"] = {}
            report_data["overall"][
                "accuracy"
            ] = f"{100 * sum(list_accuracy) / len(list_accuracy):.4f} %"
            eval_io.save_yaml("report.yaml", report_data)

    def start(self):
        # logging.info("Starting training")

        # self.evaluate(
        #     path.join("results", f"iter_{self.iter:05}"),
        #     self.cfg.result_cfg.eval_cfg.render_index,
        #     self.cfg.result_cfg.eval_cfg.render_grid,
        # )
        # sys.exit(0)

        # dataset = cast(GoDynamicDataset | GoDataset, self.train_dataloader.dataset)
        # for idx in range(len(dataset)):
        #     data_point = cast(DataPoint, dataset[idx])
        #     logger.info(
        #         f"Loading image: {idx}, path: {dataset.datapoint_paths[idx].image_path}"
        #     )
        # sys.exit(0)

        while self.iter <= self.cfg.iters:
            output_map = self.train_step()

            if self.iter % self.cfg.i_print == 0:
                logger.info(
                    f'Exp: {self.cfg.result_cfg.name}, Iter: {self.iter} / {self.cfg.iters}, Memory: {output_map["memory"]:.2f} GB, total_loss: {output_map["total_loss"]:.4f}'
                )

            if self.iter % self.cfg.i_weight == 0:
                self.save_checkpoint()

            if self.iter % self.cfg.i_tf_writer == 0:
                # Upload the output_map of the last mini batch
                self._upload_metrics_to_tf(self.iter, output_map, "epoch__")

                # Upload output map of eval
                eval_output_map = self.eval_step()
                self._upload_metrics_to_tf(self.iter, eval_output_map, "test__")

            if self.iter % self.cfg.i_eval == 0:
                logger.info("Starting evaluation")

                evaluate_path = path.join("results", f"iter_{self.iter:05}")

                eval_cfg = self.cfg.result_cfg.eval_cfg
                self.evaluate(
                    evaluate_path,
                    eval_cfg.render_dirs,
                    eval_cfg.images_per_dir,
                    eval_cfg.render_grid,
                )

            self.iter += 1

        logging.info("Finished training")
        self.tf_writer.flush()

    def _calculate_metrics(self, datapoints, model_output):
        map_metrics: Dict[str, MetricValue] = {}

        num_images = datapoints.images.shape[0]
        gt_labels = datapoints.labels.reshape(num_images, -1)

        # Check that the shape matches what we are giving to self.nll_loss
        assert (
            gt_labels.shape[1:] == model_output.shape[2:]
            and gt_labels.shape[0] == num_images
            and model_output.shape[0] == num_images
        )
        assert gt_labels.max() < model_output.shape[1] and gt_labels.min() >= 0
        nll_loss = self.nll_loss(model_output, gt_labels)
        # Manuall calculate the NLL loss for verifying
        # target_label_one_hot = torch.nn.functional.one_hot(gt_labels, model_output.shape[1])
        # nll_loss_2 = (-model_output * target_label_one_hot.transpose(1,2)).sum(dim=1).mean()
        map_metrics["nll_loss"] = MetricValue(nll_loss, 1.0)

        # We have finished calculating all the losses. Calculate total loss now
        total_loss = torch.tensor(0.0).cuda()
        for metric_name, metric in map_metrics.items():
            if "loss" in metric_name:
                total_loss = total_loss + metric.weighted_value

        map_metrics["total_loss"] = MetricValue(total_loss)

        memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        map_metrics["memory"] = MetricValue(memory_gb)

        return map_metrics

    def train_step(self):
        # Iterate through all the training datasets
        map_metrics: Dict[str, MetricValue] = {}
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        for idx, datapoints in enumerate(self.train_dataloader, 1):
            datapoints = cast(DataPoints, datapoints)
            model_output = self.model(datapoints.images)
            map_metrics = self._calculate_metrics(datapoints, model_output)

            # Use print so this does not end up in the logs
            print(
                f'Exp: {self.cfg.result_cfg.name}, Step: {self.iter}-{idx} / {self.cfg.iters}, Memory: {map_metrics["memory"]:.2f} GB, total_loss: {map_metrics["total_loss"]:.4f}'
            )

            mini_batch_idx = (self.iter - 1) * len(self.train_dataloader) + idx
            self._upload_metrics_to_tf(
                mini_batch_idx, map_metrics, "step__"
            )  # TODO(rishi): add a config for this so we dont push every tick?

            self.optimizer.zero_grad(set_to_none=True)
            map_metrics["total_loss"].base_value.backward()
            self.optimizer.step()

            break

        # Completed one epoch
        return map_metrics

    def eval_step(self):
        """Calculate the test loss"""
        aggregated_map_metrics: Dict[str, List[MetricValue]] = defaultdict(list)
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            for idx, datapoints in enumerate(self.test_dataloader, 1):
                datapoints = cast(DataPoints, datapoints)
                model_output = self.model(datapoints.images)
                map_metrics = self._calculate_metrics(datapoints, model_output)

                for matric_name, metric in map_metrics.items():
                    aggregated_map_metrics[matric_name].append(metric)

                print(
                    f'Eval Step: {self.iter}-{idx} / {self.cfg.iters}, Memory: {map_metrics["memory"]:.2f} GB, total_loss: {map_metrics["total_loss"]:.4f}'
                )

                if idx >= 0:
                    break

            map_metrics: Dict[str, MetricValue] = {}
            for metric_name, list_metrics in aggregated_map_metrics.items():
                if metric_name in ["total_loss"]:
                    list_values = [_.base_value for _ in list_metrics]
                    map_metrics[metric_name] = MetricValue(
                        torch.stack(list_values, dim=0).mean()
                    )

            return map_metrics
