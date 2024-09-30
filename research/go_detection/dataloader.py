import itertools
import logging
import random
from collections import defaultdict, namedtuple
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from go_detection.common.asset_io import AssetIO
from go_detection.config import DataCfg
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

DataPoint = namedtuple("DataPoint", ["image_path", "label_path", "board_path"])

logger = logging.getLogger(__name__)


class InfiniteSampler(Sampler):
    def __init__(self, length: int, shuffle: bool, repeat: bool):
        assert length > 0
        self.length = length
        self.shuffle = shuffle
        self.repeat = repeat

    def __len__(self):
        return self.length

    def __iter__(self):
        order = list(range(self.length))
        while True:
            if self.shuffle:
                random.shuffle(order)

            for idx in order:
                yield idx

            if not self.repeat:
                return


def custom_collate_fn(batches):
    images = []
    labels = []
    board_pts = []

    for batch in batches:
        images.append(batch[0])
        labels.append(batch[1])
        board_pts.append(batch[2])

        assert images[0].shape == batch[0].shape
        assert labels[0].shape == batch[1].shape
        assert board_pts[0].shape == batch[2].shape

    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    board_pts = torch.stack(board_pts, dim=0)
    return images, labels, board_pts


def _read_image(data_io: AssetIO, image_path: str, board_metadata):
    """
    Returns tuple:
        - resized image
        - original (width x height) before resizing
    """
    image_tensor = data_io.load_image(image_path)
    new_size = (1024, 1024)  # Specify the new size as a tuple
    resize = transforms.Resize(new_size)
    resized_tensor = resize(image_tensor)

    original_size = torch.tensor([image_tensor.shape[2], image_tensor.shape[1]])
    return resized_tensor, original_size


def _read_label(data_io: AssetIO, label_path: str):
    """
    Returns a tensor where:
        Empty: 0
        White: +1
        Black: -1
    """
    with open(data_io.get_abs(label_path), "r") as file:
        lines = file.read()
        label = []
        for line in lines.split("\n"):
            if line == "":
                continue

            label_line = []
            for ch in line.split(" "):
                if ch == ".":
                    digit = 0
                elif ch.upper() == "W":
                    digit = 1
                elif ch.upper() == "B":
                    digit = -1
                else:
                    assert (
                        False
                    ), f"Unknown character: '{ch}' in line '{line}', file: {label_path}"
                label_line.append(digit)
            label.append(label_line)

        label = torch.tensor(label)
        return label


def _load_single(data_point: DataPoint, data_io: AssetIO):
    board_metadata = data_io.load_yaml(data_point.board_path)

    label = _read_label(data_io, data_point.label_path)
    image, original_size = _read_image(data_io, data_point.image_path, board_metadata)
    image = image[:3, :, :]  # Remove the alpha channel
    board_pt = torch.tensor(board_metadata.pts_clicks) / original_size

    return image, label, board_pt


def _load(entire_data: List[DataPoint], data_io: AssetIO, include_logs=True):
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


class GoDataset(Dataset):
    def __init__(
        self,
        entire_data: List[DataPoint],
        base_path: str,
    ):
        data_io = AssetIO(base_path)
        self.images, self.labels, self.board_pts = _load(entire_data, data_io)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.board_pts[idx]


class GoDynamicDataset(Dataset):
    """
    We cannot load the entire dataset into memory. So load it on the fly
    """

    def __init__(
        self,
        entire_data: List[DataPoint],
        base_path: str,
    ):
        self.data_io = AssetIO(base_path)
        self.entire_data = entire_data

    def __len__(self):
        return len(self.entire_data)

    def __getitem__(self, idx):
        images, labels, board_pts = _load_single(self.entire_data[idx], self.data_io)
        return images, labels, board_pts


def create_datasets(cfg: DataCfg):
    data_io = AssetIO(cfg.base_path)
    directories = sorted(data_io.ls())

    entire_data: List[List[DataPoint]] = []
    for directory in directories:
        if not data_io.is_dir(directory):
            continue

        files = data_io.ls(directory, only_file_name=False)
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

        directory_data: List[DataPoint] = []
        for val in map_name_to_dict.values():
            if "label" in val and "image" in val:
                data = DataPoint(
                    val["image"],
                    val["label"],
                    board_file,
                )
                directory_data.append(data)

        if directory_data:
            entire_data.append(directory_data)

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

    if cfg.use_dynamic_dataset:
        train_dataset, test_dataset = GoDynamicDataset(
            train, cfg.base_path
        ), GoDynamicDataset(test, cfg.base_path)
    else:
        train_dataset, test_dataset = GoDataset(train, cfg.base_path), GoDataset(
            test, cfg.base_path
        )

    logger.info(
        f"Loaded datasets: Train: {len(train_dataset)} ({100.0 * len(train_dataset) / (len(train_dataset) + len(test_dataset)) : .1f} %), Test: {len(test_dataset)} ({100.0 * len(test_dataset) / (len(train_dataset) + len(test_dataset)) : .1f} %)"
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        # shuffle=True,
        sampler=InfiniteSampler(len(train_dataset), True, False),
        collate_fn=custom_collate_fn,
        # num_workers=4,
        # multiprocessing_context="spawn",
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=InfiniteSampler(len(test_dataset), False, False),
        collate_fn=custom_collate_fn,
        # num_workers=4,
        # multiprocessing_context="spawn",
    )

    # for i, data in enumerate(test_dataloader_iter):
    #     print(f"Test data point: {i}, shape: {data[0].shape}")
    return train_dataloader, test_dataloader
