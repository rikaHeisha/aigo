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
from matplotlib import pyplot as plt
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

    @property
    def has_pt_files(self):
        has_pt = self.image_path.endswith(".pt")
        return has_pt


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


class LabelPieces(IntEnum):
    BLACK = 0
    EMPTY = 1
    WHITE = 2

    @staticmethod
    def from_str(ch) -> "LabelPieces":
        if ch in {"B", "b"}:
            return LabelPieces.BLACK
        elif ch in {" ", "."}:
            return LabelPieces.EMPTY
        elif ch in {"W", "w"}:
            return LabelPieces.WHITE
        else:
            assert False, f"Unknown character: {ch}"

    def to_str(self) -> str:
        if self == LabelPieces.BLACK:
            return "B"
        elif self == LabelPieces.EMPTY:
            return "."
        elif self == LabelPieces.WHITE:
            return "W"
        else:
            assert False, f"Unknown enum: {self}"


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


def _read_raw_image(data_io: AssetIO, image_path: str, board_pts: torch.Tensor):
    """
    Returns tuple:
        - resized image
        - original (width x height) before resizing
    """
    orig_image = data_io.load_image(image_path)
    _, height, width = orig_image.shape
    original_size = torch.tensor([width, height])
    new_size = (1024, 1024)  # Specify the new size (height, width)

    if width == height:
        resize_transform = transforms.Resize(new_size)
        resized_tensor = resize_transform(orig_image)
        return resized_tensor, original_size

    center_square = board_pts.mean(dim=0)
    # square_half_length will perfectly keep one dimension. If width is larger, then square_half_length will be half_height of image.
    square_half_length = torch.tensor(min(width, height) / 2).float().ceil()
    if width > height:
        center_square[1] = height / 2
    else:
        assert width < height
        center_square[0] = width / 2

    all_points_are_inside = all(
        [
            ((board_pts - (center_square - square_half_length)) >= 0.0).all(),
            ((board_pts - (center_square + square_half_length)) <= 0.0).all(),
        ]
    )

    if not all_points_are_inside:
        # increase square_half_length a small amount until the board perfectly fits in

        required_half_length = (board_pts - center_square).abs().max().ceil()
        assert (
            min(width, height) < 2 * required_half_length < max(width, height)
        ), f"The required length cannot be bigger than both width and height. We expect it to only be bigger than one of them"

        # Check that all the points are inside this bigger region
        assert all(
            [
                ((board_pts - (center_square - required_half_length)) >= 0.0).all(),
                ((board_pts - (center_square + required_half_length)) <= 0.0).all(),
            ]
        ), "Expected all the board points to be inside the square after increasing square_half_length"

        load_mode = 1
        if load_mode == 0:
            # In this mode, we do not want aspect ratio to change at all. This mode starts with the best fit square, and expands it in BOTH dimension till it fits all the points. This will cause some black padding to appear. Then we can further expand it using the extra_expand parameter. A higher value will cause more black padding, but will also increase the amount of background infomation in the image
            extra_expand = 60
            rectangle_half_length = torch.tensor(
                [required_half_length, required_half_length]
            )
            rectangle_half_length = rectangle_half_length + extra_expand

        elif load_mode == 1:
            # In this mode, we do not want any black padding at all. This mode starts with the best fit square, and expands it in ONE dimension till it fits all the points. Then we can furthur expand it using extra_expand parameter. A higher value will increase the amount of background info in the image. Once we crop the rectangular region, it gets resized to a square, so this mode does not preserve the aspect ratio
            extra_expand = 60
            rectangle_half_length = (
                [required_half_length + extra_expand, square_half_length]
                if width > height
                else [square_half_length, required_half_length + extra_expand]
            )
            rectangle_half_length = torch.tensor(rectangle_half_length)

        else:
            assert False, f"Unknown load mode: {load_mode}"
    else:
        # only need to crop
        rectangle_half_length = torch.tensor([square_half_length, square_half_length])

    # Check that all the points fit inside the crop region
    all(
        [
            ((board_pts - (center_square - rectangle_half_length)) >= 0.0).all(),
            ((board_pts - (center_square + rectangle_half_length)) <= 0.0).all(),
        ]
    ), "Expected all the board points to be inside the rectangular crop region"
    intermediate_image = crop(
        orig_image,
        int(center_square[1] - rectangle_half_length[1]),
        int(center_square[0] - rectangle_half_length[0]),
        int(2 * rectangle_half_length[1]),
        int(2 * rectangle_half_length[0]),
    )

    resize_transform = transforms.Resize(new_size)
    resized_image = resize_transform(intermediate_image)

    # asset_io = AssetIO("/home/rmenon/Desktop/dev/projects/aigo/research")
    # asset_io.save_image("rishi_orig.png", orig_image)
    # asset_io.save_image("rishi_intermediate.png", intermediate_image)
    # asset_io.save_image("rishi_final.png", resized_image)

    return resized_image, original_size


def _read_label(data_io: AssetIO, label_path: str):
    """
    Returns
        A tensor of Boardsize x Boardsize
    """

    if label_path.endswith(".pt"):
        return data_io.load_torch(label_path)

    with open(data_io.get_abs(label_path), "r") as file:
        lines = file.read()
        label = []
        for line in lines.split("\n"):
            if line == "":
                continue

            label_line = []
            for ch in line.split(" "):
                digit = LabelPieces.from_str(ch.upper())
                # assert (
                #     False
                # ), f"Unknown character: '{ch}' in line '{line}', file: {label_path}"

                label_line.append(digit)
            label.append(label_line)

        label = torch.tensor(label)
        return label


def _load_single(data_point: DataPointPath, data_io: AssetIO) -> DataPoint:
    if data_point.has_pt_files:
        image = data_io.load_torch(data_point.image_path)
        label = data_io.load_torch(data_point.label_path)
        board_pts = data_io.load_torch(data_point.board_path)

    else:
        board_pts = torch.tensor(
            data_io.load_yaml(data_point.board_path)["pts_clicks"]
        ).float()

        label = _read_label(data_io, data_point.label_path)
        image, original_size = _read_raw_image(
            data_io, data_point.image_path, board_pts
        )
        image = image[:3, :, :]  # Remove the alpha channel
        # board_pts is a list of 4 points. The first point is the top left corner, and then the points are in clockwise order
        board_pts = board_pts / original_size

    return DataPoint(image, label, board_pts)


def _load(entire_data: List[DataPointPath], data_io: AssetIO, include_logs=True):
    images = []
    labels = []
    board_pts = []

    if include_logs:
        iters = tqdm(entire_data, desc="Loading dataset")
    else:
        iters = iter(entire_data)

    for data_point in iters:
        image, label, board_pt = _load_single(data_point)
        images.append(image)
        labels.append(label)
        board_pts.append(board_pt)

    return images, labels, board_pts


def _load_num_pieces(data_io: AssetIO, datapoint_paths: List[DataPointPath]):
    list_num_pieces = []
    for datapoint_path in datapoint_paths:
        label = _read_label(data_io, datapoint_path.label_path)
        num_pieces = (label != 1).sum()
        list_num_pieces.append(num_pieces)
    return torch.stack(list_num_pieces, dim=0)


class GoDataset(Dataset):
    def __init__(
        self,
        datapoint_paths: List[DataPointPath],
        base_path: str,
    ):
        self.datapoint_paths = datapoint_paths
        self.num_pieces = _load_num_pieces(AssetIO(base_path), datapoint_paths)

        self._images, self._labels, self._board_pts = _load(
            datapoint_paths, AssetIO(base_path)
        )

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx) -> DataPoint:
        data_point = DataPoint(
            self._images[idx], self._labels[idx], self._board_pts[idx]
        )
        return data_point.cpu()


class GoDynamicDataset(Dataset):
    """
    We cannot load the entire dataset into memory so load it dynamically
    """

    def __init__(
        self,
        datapoint_paths: List[DataPointPath],
        base_path: str,
    ):
        self._data_io = AssetIO(base_path)
        self.datapoint_paths = datapoint_paths
        self.num_pieces = _load_num_pieces(AssetIO(base_path), datapoint_paths)

    def __len__(self):
        return len(self.datapoint_paths)

    def __getitem__(self, idx) -> DataPoint:
        data_point = _load_single(self.datapoint_paths[idx], self._data_io)
        return data_point.cpu()


def _create_sampler(sampler_type: str, dataset: GoDataset | GoDynamicDataset):
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
    if asset_io.has_file("dataset_info.yaml"):
        dataset_info = asset_io.load_yaml("dataset_info.yaml")
        version = dataset_info["version"]

    if version == 0:
        return _get_all_datapoints_v0(base_path)
    elif version == 1:
        return _get_all_datapoints_v1(base_path)
    else:
        assert False, f"Unknown version: {version}"


def _get_all_datapoints_v1(base_path: str) -> List[List[DataPointPath]]:
    use_pt_files = True

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
            if use_pt_files:
                assert "image.pt" in files
                assert "label.pt" in files
                assert "board_info.pt" in files
                datapoint_path = DataPointPath(
                    path.join(image_dir, "image.pt"),
                    path.join(image_dir, "label.pt"),
                    path.join(image_dir, "board_info.pt"),
                )
            else:
                assert "raw_image.png" in files
                assert "raw_label.txt" in files
                assert "raw_board_info.yaml" in files

                datapoint_path = DataPointPath(
                    path.join(image_dir, "raw_image.png"),
                    path.join(image_dir, "raw_label.txt"),
                    path.join(image_dir, "raw_board_info.yaml"),
                )

            datapoint_paths.append(datapoint_path)

        entire_data.append(datapoint_paths)

    return entire_data


def _get_all_datapoints_v0(base_path: str) -> List[List[DataPointPath]]:
    data_io = AssetIO(base_path)
    directories = sorted(data_io.ls())

    entire_data: List[List[DataPointPath]] = []
    for directory in directories:
        if not data_io.has_dir(directory):
            continue

        files = data_io.ls(directory)
        map_name_to_dict = defaultdict(dict)
        board_file = None

        for file_name in files:
            file_split = file_name.rsplit(".", 1)

            key = None
            if len(file_split) == 2 and file_split[-1].lower() in ["png", "jpg"]:
                key = "image"
            elif len(file_split) == 2 and file_split[-1].lower() in ["txt"]:
                key = "label"
            elif "board_extractor_state" in file_split[0]:
                board_file = file_name

            if key is None:
                continue

            map_name_to_dict[file_split[0]][key] = file_name

        directory_data: List[DataPointPath] = []
        for val in map_name_to_dict.values():
            if "label" in val and "image" in val:
                data = DataPointPath(
                    val["image"],
                    val["label"],
                    board_file,
                )
                directory_data.append(data)

        if directory_data:
            entire_data.append(directory_data)

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
    return train, test


def create_datasets(cfg: DataCfg):
    train, test = create_datasets_split(cfg)
    return load_datasets(cfg, train, test)


# Only used by dump_dataset script. TODO(rishi): create a dataloader_original script that loads the old format and this file should only contain loading for the new format
def get_all_datapoints(base_path: str):
    entire_data = _get_all_datapoints(base_path)
    return entire_data
