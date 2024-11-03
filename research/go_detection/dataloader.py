import itertools
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from os import path
from typing import List, Tuple, cast

import numpy as np
import torch
import torchvision.transforms as transforms
from go_detection.common.asset_io import AssetIO
from go_detection.config import DataCfg
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms.functional import crop
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DataPointPath:
    image_path: str
    label_path: str
    board_path: str


@dataclass
class DataPoint:
    image: torch.Tensor
    label: torch.Tensor
    board_pt: torch.Tensor

    def cpu(self):
        return DataPoint(
            self.image.cpu(),
            self.label.cpu(),
            self.board_pt.cpu(),
        )

    def cuda(self):
        return DataPoint(
            self.image.cuda(),
            self.label.cuda(),
            self.board_pt.cuda(),
        )

    def to_data_points(self) -> "DataPoints":
        return DataPoints(
            self.image.unsqueeze(0),
            self.label.unsqueeze(0),
            self.board_pt.unsqueeze(0),
        )


@dataclass
class DataPoints:
    images: torch.Tensor
    labels: torch.Tensor
    board_pts: torch.Tensor

    def cpu(self):
        return DataPoints(
            self.images.cpu(),
            self.labels.cpu(),
            self.board_pts.cpu(),
        )

    def cuda(self):
        return DataPoints(
            self.images.cuda(),
            self.labels.cuda(),
            self.board_pts.cuda(),
        )

    def get_point(self, index) -> DataPoint:
        assert index < self.images.shape[0]
        return DataPoint(
            self.images[index],
            self.labels[index],
            self.board_pts[index],
        )


class DistSampler(Sampler):
    def __init__(self, pmf: List[int]):
        super().__init__()
        self.pmf = pmf
        self.indices = list(range(len(self.pmf)))

    def sample(self, requested_sample_size: int):
        return np.random.choice(
            self.indices, requested_sample_size, replace=True, p=self.pmf
        )

    def __iter__(self):
        while True:
            order = self.sample(len(self.pmf))
            yield from order


class UniformSampler(DistSampler):
    def __init__(self, length: int, batch_size: int):
        pmf = np.array([1.0 / length for _ in range(length)])
        super().__init__(pmf, batch_size)


# This is a sampler without replacement
class NonReplacementSampler(Sampler):
    def __init__(self, length: int, shuffle: bool = True, repeat: bool = True):
        super().__init__()
        assert length > 0
        self.length = length
        self.shuffle = shuffle
        self.repeat = repeat

    def sample(self, requested_sample_size: int):
        order = list(range(self.length))
        if self.shuffle:
            random.shuffle(order)

        batch_size = min(requested_sample_size, self.length)
        return order[0:batch_size]

    def __iter__(self):
        while True:
            order = self.sample(self.length)
            yield from order

            if not self.repeat:
                return


def custom_collate_fn(batches) -> DataPoints:
    images = []
    labels = []
    board_pts = []

    for batch in batches:
        batch = cast(DataPoint, batch)
        images.append(batch.image)
        labels.append(batch.label)
        board_pts.append(batch.board_pt)

        assert images[0].shape == batch.image.shape
        assert labels[0].shape == batch.label.shape
        assert board_pts[0].shape == batch.board_pt.shape

    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    board_pts = torch.stack(board_pts, dim=0)
    return DataPoints(images, labels, board_pts).cuda()


def _load_single(data_point: DataPointPath, data_io: AssetIO) -> DataPoint:
    image = data_io.load_torch(data_point.image_path)
    label = data_io.load_torch(data_point.label_path)
    assert label.shape == (19, 19)
    board_pts = data_io.load_torch(data_point.board_path)
    return DataPoint(image, label, board_pts)


def _load_all(entire_data: List[DataPointPath], data_io: AssetIO, include_logs=True):
    images = []
    labels = []
    board_pts = []

    for data_point in tqdm(entire_data, desc="Loading dataset"):
        image, label, board_pt = _load_single(data_point)
        images.append(image)
        labels.append(label)
        board_pts.append(board_pt)

    return images, labels, board_pts


def _load_num_pieces(data_io: AssetIO, datapoint_paths: List[DataPointPath]):
    list_num_pieces = []
    for datapoint_path in datapoint_paths:
        label = data_io.load_torch(datapoint_path.label_path)
        assert label.shape == (19, 19)
        num_pieces = (label != 1).sum()
        list_num_pieces.append(num_pieces)
    return torch.stack(list_num_pieces, dim=0)


class GoBaseDataset(Dataset):
    def __init__(
        self,
        datapoint_paths: List[DataPointPath],
        base_path: str,
    ):
        self.datapoint_paths = datapoint_paths
        self.base_path = base_path
        self.asset_io = AssetIO(base_path)

        self.num_pieces = _load_num_pieces(AssetIO(base_path), datapoint_paths)

    def __len__(self):
        return len(self.datapoint_paths)

    def __getitem__(self, idx) -> DataPoint:
        return self.getitem(idx).cpu()

    def getitem(self) -> DataPoint:
        raise NotImplementedError("Base class does not implement this")


class GoDataset(GoBaseDataset):
    def __init__(
        self,
        datapoint_paths: List[DataPointPath],
        base_path: str,
    ):
        super().__init__(datapoint_paths, base_path)

        self._images, self._labels, self._board_pts = _load_all(
            datapoint_paths, self.asset_io
        )

    def getitem(self, idx) -> DataPoint:
        data_point = DataPoint(
            self._images[idx], self._labels[idx], self._board_pts[idx]
        )
        return data_point


class GoDynamicDataset(GoBaseDataset):
    """
    We cannot load the entire dataset into memory so load it dynamically
    """

    def __init__(
        self,
        datapoint_paths: List[DataPointPath],
        base_path: str,
    ):
        super().__init__(datapoint_paths, base_path)

    def getitem(self, idx) -> DataPoint:
        data_point = _load_single(self.datapoint_paths[idx], self.asset_io)
        return data_point


def _create_sampler(sampler_type: str, dataset: GoBaseDataset):
    assert sampler_type in [
        "non_replacement",
        "uniform",
        "dist_equal",
    ], f"Unknown train sampler type: {sampler_type}"

    if sampler_type == "non_replacement":
        return NonReplacementSampler(len(dataset), True, True)
    elif sampler_type == "uniform":
        return UniformSampler(len(dataset))
    elif sampler_type == "dist_equal":
        total_possibilities = (
            19 * 19 + 1
        )  # Note: This may only work for fixed board sizes. How to handle other go board sizes?

        # calculate the pmf of the dataset
        list_num_pieces = []
        for i in range(len(dataset)):
            num_pieces = dataset.num_pieces[i].item()
            list_num_pieces.append(num_pieces)
        list_num_pieces = np.array(list_num_pieces)

        hist, _ = np.histogram(
            list_num_pieces, bins=np.arange(0, total_possibilities + 1)
        )
        original_pmf = hist / hist.sum()
        # original_cdf = np.cumsum(original_pmf)

        original_pmf[original_pmf == 0.0] = np.inf
        weights = 1.0 / original_pmf

        weights_per_image = []
        for i in range(len(dataset)):
            num_pieces = dataset.num_pieces[i].item()
            weights_per_image.append(weights[num_pieces])
        weights_per_image = np.array(weights_per_image)

        sample_pmf = weights_per_image / weights_per_image.sum()

        # verify that the sample pmf results in uniform distribution sampling
        verify_values = [0.0 for _ in range(total_possibilities)]
        for i in range(len(dataset)):
            num_pieces = dataset.num_pieces[i].item()
            verify_values[num_pieces] += sample_pmf[i]
        verify_values = np.array(verify_values)
        non_null = verify_values[verify_values != 0.0]
        assert np.isclose(
            non_null, non_null[0]
        ).all()  # All the values should be equal to each other

        return DistSampler(sample_pmf)


def load_datasets(
    cfg: DataCfg,
    train_datapoint_paths: List[DataPointPath],
    test_datapoint_paths: List[DataPointPath],
):
    if cfg.use_dynamic_dataset:
        train_dataset = GoDynamicDataset(train_datapoint_paths, cfg.base_path)
        test_dataset = GoDynamicDataset(test_datapoint_paths, cfg.base_path)
    else:
        train_dataset = GoDataset(train_datapoint_paths, cfg.base_path)
        test_dataset = GoDataset(test_datapoint_paths, cfg.base_path)

    train_sampler = _create_sampler(cfg.train_sampler_type, train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        # shuffle=True,
        sampler=train_sampler,
        collate_fn=custom_collate_fn,
        # num_workers=4,
        # multiprocessing_context="spawn",
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.test_batch_size,
        sampler=NonReplacementSampler(len(test_dataset), False, False),
        collate_fn=custom_collate_fn,
        # num_workers=4,
        # multiprocessing_context="spawn",
    )
    return train_dataloader, test_dataloader


def _get_all_datapoints(base_path: str) -> List[List[DataPointPath]]:
    asset_io = AssetIO(base_path)
    version = 0
    assert asset_io.has_file("dataset_info.yaml")
    dataset_info = asset_io.load_yaml("dataset_info.yaml")
    version = dataset_info["version"]

    if version == 1:
        return _get_all_datapoints_v1(base_path)
    else:
        assert False, f"Unknown version: {version}"


def _get_all_datapoints_v1(base_path: str) -> List[List[DataPointPath]]:
    data_io = AssetIO(base_path)
    board_dirs = sorted(data_io.ls())
    entire_data: List[List[DataPointPath]] = []
    for board_dir in board_dirs:
        if not data_io.has_dir(board_dir):
            continue

        datapoint_paths = []
        image_dirs = data_io.ls(board_dir)
        for image_dir in image_dirs:
            files = data_io.ls(image_dir, True)
            assert (
                "image.pt" in files
            ), f"Expected image.pt file in {board_dir}/{image_dir}"
            assert (
                "label.pt" in files
            ), f"Expected label.pt file in {board_dir}/{image_dir}"
            assert (
                "board_info.pt" in files
            ), f"Expected board_info.pt file in {board_dir}/{image_dir}"

            datapoint_path = DataPointPath(
                path.join(image_dir, "image.pt"),
                path.join(image_dir, "label.pt"),
                path.join(image_dir, "board_info.pt"),
            )

            datapoint_paths.append(datapoint_path)

        entire_data.append(datapoint_paths)

    return entire_data


def create_datasets_split(
    cfg: DataCfg,
) -> Tuple[List[DataPointPath], List[DataPointPath]]:

    entire_data = _get_all_datapoints(cfg.base_path)
    if cfg.randomize_train_split:
        random.shuffle(entire_data)

    # Calculate the train test split
    directory_counts = [len(_) for _ in entire_data]
    directory_cumsum = np.cumsum(directory_counts)
    directory_cumsum = directory_cumsum / directory_cumsum[-1]

    # Split the train and test dataset based on the directorys. This is done because one directory has the same board and background. So we don't want images to spill from train dataset to test dataset
    split_index = np.searchsorted(directory_cumsum, cfg.train_split_percent, "right")
    train, test = entire_data[:split_index], entire_data[split_index:]
    assert len(train) + len(test) == len(entire_data)

    # If you want validation dataset as well
    # split_index = np.searchsorted(directory_cumsum, [0.7, 0.9], "right")
    # train, validate, test = (
    #     entire_data[: split_index[0]],
    #     entire_data[split_index[0] : split_index[1]],
    #     entire_data[split_index[1] :],
    # )
    # assert len(train) + len(validate) + len(test) == len(entire_data)

    train = list(itertools.chain(*train))
    test = list(itertools.chain(*test))

    # TODO(rishi): check if this is safe to uncomment
    # if cfg.randomize_train_split:
    #     random.shuffle(train)
    #     random.shuffle(test)
    return train, test


def create_datasets(cfg: DataCfg):
    train, test = create_datasets_split(cfg)
    return load_datasets(cfg, train, test)
