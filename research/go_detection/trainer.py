import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from math import isclose
from os import mkdir, path
from typing import Dict, List, Literal, Optional, Tuple, TypeVar, Union, cast

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
from go_detection.dataloader_viz import (
    visualize_accuracy_over_num_pieces,
    visualize_grid,
)
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
        return (
            f"MetricValue(base_value={self.base_value}, weight={self.weight or 'None'})"
        )

    def __repr__(self):
        return (
            f"MetricValue(base_value={self.base_value}, weight={self.weight or 'None'})"
        )

    def __format__(self, spec):
        if self.weight is None:
            return f"{self.base_value:{spec}}"
        else:
            return f"({self.base_value:{spec}}, {self.weight:{spec}})"

    def detach(self) -> "MetricValue":
        return MetricValue(self.base_value.detach(), self.weight)


def combine_metrics(
    list_metrics: List[MetricValue], reduction: Literal["mean", "sum"] = "mean"
) -> MetricValue:
    assert len(list_metrics) > 0, "Cannot combine an empty list of metrics"
    if len(list_metrics) == 1:
        return list_metrics[0]

    for metric in list_metrics:
        if list_metrics[0].has_weight():
            assert (
                metric.has_weight()
            ), "Cannot combine metrics, some have weights and some don't"
            assert isclose(
                list_metrics[0].weight, metric.weight
            ), f"Cannot combine metrics, expected the weight of all metrics to be the same"
        else:
            assert (
                not metric.has_weight()
            ), "Cannot combine metrics, some have weights and some don't"

    # They all have the same weight, combine the base_value
    base_values = torch.stack([_.base_value for _ in list_metrics], dim=0)
    if reduction == "mean":
        metric = MetricValue(base_values.mean(), list_metrics[0].weight)
    elif reduction == "sum":
        metric = MetricValue(base_values.sum(), list_metrics[0].weight)
    else:
        assert False, f"Unknown reduction mode: {reduction}"

    return metric


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

        # Calculate the run number
        existing_run_files = [
            int(_.removeprefix("tf/run_"))
            for _ in self.results_io.ls("tf")
            if self.results_io.has_dir(_)
        ]
        # run_number = len(existing_run_files) + 1 # Naive solution
        run_number = (max(existing_run_files) + 1) if existing_run_files != [] else 1

        tf_path = path.join("tf", f"run_{run_number}")
        assert self.results_io.has(tf_path) == False
        self.results_io.mkdir(tf_path)
        self.tf_writer = SummaryWriter(self.results_io.get_abs(tf_path))

        # Create the model and optimizer
        self.iter = 1
        self.model = GoModel(self.cfg.model_cfg)
        self.model = self.model.cuda()

        if self.cfg.model_cfg.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(
                self.get_parameters(),
                lr=1e-3,
                weight_decay=self.cfg.model_cfg.optimizer_weight_decay,
            )
        elif self.cfg.model_cfg.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(
                self.get_parameters(),
                lr=1e-3,
                weight_decay=self.cfg.model_cfg.optimizer_weight_decay,
            )
        else:
            assert False, f"Unknown optimizer type: {self.cfg.model_cfg.optimizer_type}"

        # Create the dataloader and load checkpoint
        self.train_dataloader, self.test_dataloader = self._load_or_create_dataloader()
        self.load_checkpoint()

        # Create losses
        self.nll_loss = nn.NLLLoss(reduction="mean")

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
        self.results_io.save_torch(
            "checkpoint.pt", checkpoint_data
        )  # TODO(rishi): Save all pt files to a different folder

        verify_checkpoint = True
        if verify_checkpoint:
            logger.info("Verifying checkpoint")
            checkpoint_data = self.results_io.load_torch("checkpoint.pt")
            for key, param_orig in self.model.state_dict().items():
                param_checkpoint = checkpoint_data["model_state_dict"][key].cuda()
                atol = 1e-7
                if not torch.isclose(param_orig, param_checkpoint, atol=atol).all():
                    a = (
                        ~torch.isclose(param_orig, param_checkpoint, atol=atol)
                    ).nonzero()
                    # self.results_io.save_torch("temp.pt", param_orig)
                    # b = self.results_io.load_torch("temp.pt")

                    logger.error(f"Checkpoint model params does not match: {key}")

            for key, value in self.optimizer.state_dict()["state"].items():
                ckpt_value = checkpoint_data["optimizer_state_dict"]["state"][key]
                for param_key in value.keys():
                    if not torch.isclose(
                        value[param_key].cpu(), ckpt_value[param_key]
                    ).all():
                        logger.error(
                            f"Checkpoint optimizer state does not match: {key} {param_key}"
                        )

            for param_orig, param_ckpt in zip(
                self.optimizer.state_dict()["param_groups"],
                checkpoint_data["optimizer_state_dict"]["param_groups"],
            ):
                if param_orig != param_ckpt:
                    logger.error(f"Checkpoint optimizer param_groups does not match")

    def get_parameters(self):
        parameters = self.model.parameters()
        return parameters

    def _convert_render_dir_to_indices(
        self,
        datapoint_paths: List[DataPointPath],
        render_dirs: List[int],
        images_per_dir: Optional[int],
    ) -> List[int]:
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

    def _evaluate_dataset(
        self,
        dataset: (
            GoDynamicDataset | GoDataset
        ),  # TODO(rishi): change this to dataloader?
        evaluate_path: str,
        render_dirs: List[int],
        images_per_dir: Optional[int],
        render_grid: bool = True,
    ):
        eval_io = self.results_io.cd(evaluate_path)
        eval_io.mkdir("render_grid")

        # TODO(rishi): Use the dataloader instead of directly accessing the dataset. For this, we need to get rid of the render_index parameter and find a better config that the user can set
        # Actually a great way of doing this will be to iterate through the whole dataset and selectively rendering. This is a good idea since we anyway have to iterate through the entire thing

        report_data = {}
        datapoint_paths = cast(List[DataPointPath], dataset.datapoint_paths)

        # Convert render_dirs and images_per_dir to a list of indices
        indices = self._convert_render_dir_to_indices(
            datapoint_paths, render_dirs, images_per_dir
        )

        self.model.eval()
        with torch.no_grad():
            # TODO(rishi) support multiple images at once
            for idx in tqdm(indices, desc=f"Rendering for iter {self.iter}"):
                assert idx < len(
                    dataset
                ), f"Index {idx} outside length of dataset {len(dataset)}"

                data_point = cast(DataPoint, dataset[idx])
                data_points = data_point.to_data_points().cuda()

                output = self.model(data_points.images)

                (_, predicted_label) = output[0].max(dim=0)
                predicted_label = predicted_label.reshape(data_points.labels[0].shape)
                gt_label = data_points.labels[0]

                num_pieces = (gt_label != 1).sum().item()
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

                report_data[image_name] = {}
                report_data[image_name]["path"] = datapoint_paths[idx].image_path
                report_data[image_name]["num_pieces"] = num_pieces
                report_data[image_name]["num_correct"] = num_correct
                report_data[image_name]["num_incorrect"] = num_incorrect
                report_data[image_name]["accuracy"] = accuracy

        list_accuracy = [v["accuracy"] for v in report_data.values()]

        # Plot accuracy over num_pieces
        points = [(v["num_pieces"], v["accuracy"]) for v in report_data.values()]
        visualize_accuracy_over_num_pieces(points, eval_io.get_abs("plot_accuracy.png"))

        report_data["overall"] = {}
        report_data["overall"]["accuracy"] = sum(list_accuracy) / len(list_accuracy)

        # Draw plot of accuracy over num_pieces
        map_num_pieces_accuracy = defaultdict(list)
        for image_name, value in report_data.items():
            if image_name == "overall":
                continue
            map_num_pieces_accuracy[value["num_pieces"]].append(value["accuracy"])

        map_num_pieces_accuracy = {
            k: sum(v) / len(v) for k, v in map_num_pieces_accuracy.items()
        }

        # this gets stored to yaml
        report_data_sanitized = {}
        for image_name, value in report_data.items():
            report_data_sanitized[image_name] = {}
            for prop_name, value_sub in value.items():
                if prop_name == "accuracy":
                    report_data_sanitized[image_name][
                        prop_name
                    ] = f"{100.0 * value_sub:.4f} %"
                else:
                    report_data_sanitized[image_name][prop_name] = value_sub

        eval_io.save_yaml("report.yaml", report_data_sanitized)

    def evaluate(
        self,
        evaluate_path: str,
        render_dirs: List[int],
        images_per_dir: Optional[int],
        render_grid: bool = True,
    ):
        evaluate_train = True

        if evaluate_train:
            self._evaluate_dataset(
                self.train_dataloader.dataset,
                path.join(evaluate_path, "train"),
                render_dirs=[0, 1, 2, 3],
                images_per_dir=3,
                render_grid=True,
            )

        self._evaluate_dataset(
            self.test_dataloader.dataset,
            path.join(evaluate_path, "test"),
            render_dirs=render_dirs,
            images_per_dir=images_per_dir,
            render_grid=render_grid,
        )

    def start(self):
        logging.info("Starting training")

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
            last_iter = self.iter == self.cfg.iters
            output_map = self.train_step()
            eval_output_map = None
            if self.iter % self.cfg.i_tf_writer == 0 or last_iter:
                eval_output_map = self.eval_step()

            if self.iter % self.cfg.i_print == 0 or last_iter:
                logger.info(
                    f'Exp: {self.cfg.result_cfg.name}, Iter: {self.iter} / {self.cfg.iters}, Memory: {output_map["memory"]:.2f} GB, total_loss: {output_map["total_loss"]:.4f}'
                )

            if self.iter % self.cfg.i_weight == 0 or last_iter:
                self.save_checkpoint()

            if self.iter % self.cfg.i_tf_writer == 0 or last_iter:
                # Upload the output_map of the last mini batch
                self._upload_metrics_to_tf(self.iter, output_map, "epoch__")

                # Upload output map of eval
                if eval_output_map is not None:
                    logger.info(
                        f'Eval Iter: {self.iter} / {self.cfg.iters}, Memory: {eval_output_map["memory"]:.2f} GB, total_loss: {eval_output_map["total_loss"]:.4f}'
                    )
                    self._upload_metrics_to_tf(self.iter, eval_output_map, "test__")

            # At last iter we evaluate the full dataset
            if self.iter % self.cfg.i_eval == 0 and not last_iter:
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
        logging.info("Evaluating full dataest")
        self.evaluate(
            path.join("results", f"full_eval"),
            render_dirs=[],
            images_per_dir=None,
            render_grid=True,
        )
        self.tf_writer.flush()

    def _calculate_metrics(self, datapoints: DataPoints, model_output: torch.Tensor):
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

        _, model_label = model_output.max(dim=1)
        num_correct = (gt_labels == model_label).sum()
        num_incorrect = (gt_labels != model_label).sum()
        map_metrics["accuracy"] = MetricValue(
            num_correct / (num_correct + num_incorrect)
        )

        return map_metrics

    def train_step(self):
        # Iterate through all the training datasets
        aggregated_map_metrics: Dict[str, List[MetricValue]] = defaultdict(list)

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

            # Add to aggregated map AFTER we do optimizer.step. This is so that we don't keep storing the gradients
            for matric_name, metric in map_metrics.items():
                aggregated_map_metrics[matric_name].append(metric.detach())

            # if idx >= 1:
            #     break

        # Completed one epoch
        map_metrics: Dict[str, MetricValue] = {}
        for metric_name, list_metrics in aggregated_map_metrics.items():
            map_metrics[metric_name] = combine_metrics(list_metrics)

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
                    f'Eval Iter: {self.iter}-{idx} / {self.cfg.iters}, Memory: {map_metrics["memory"]:.2f} GB, total_loss: {map_metrics["total_loss"]:.4f}'
                )

                # if idx >= 1:
                #     break

            map_metrics: Dict[str, MetricValue] = {}
            for metric_name, list_metrics in aggregated_map_metrics.items():
                map_metrics[metric_name] = combine_metrics(list_metrics)

            return map_metrics
