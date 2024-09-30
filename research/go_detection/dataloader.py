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
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

DataPointPath = namedtuple("DataPointPath", ["image_path", "label_path", "board_path"])
DataPoint = namedtuple("DataPoint", ["image", "label", "board_pt"])
DataPoints = namedtuple("DataPoints", ["images", "labels", "board_pts"])

logger = logging.getLogger(__name__)


def _convert_points(points: torch.Tensor, board_pt: torch.Tensor):
    """
    Points is a tensor of (Nx2) points in normalized image coordinates (range [0,1]).
    The points gets mapped to board_pt using bilinear interpolation. The board_pt contains the corners of the cube on the image
    Returns:
        A tensor of shape (Nx2) containing the mapped points (image points)
    """
    assert points.min() >= 0.0 and points.max() <= 1.0

    # Perform lerp on the top edge
    diff_x = board_pt[1] - board_pt[0]
    points_top = board_pt[0] + points[:, 0].unsqueeze(1) * diff_x

    # Perform lerp on the bottom edge
    diff_x = board_pt[2] - board_pt[3]
    points_bottom = board_pt[3] + points[:, 0].unsqueeze(1) * diff_x

    # Perform lerp
    diff_y = points_bottom - points_top
    image_points = points_top + points[:, 1].unsqueeze(1) * diff_y

    # # Perform lerp on the left edge
    # diff_y = board_pt[3] - board_pt[0]
    # points_left = board_pt[0] + points[:, 1].unsqueeze(1) * diff_y

    # # Perform lerp on the right edge
    # diff_y = board_pt[2] - board_pt[1]
    # points_right = board_pt[1] + points[:, 1].unsqueeze(1) * diff_y

    # # Perform lerp
    # diff_x = points_right - points_left
    # image_points_2 = points_left + points[:, 0].unsqueeze(1) * diff_x

    return image_points


def _visualize_single_helper(
    axis,
    image: torch.Tensor,
    label: torch.Tensor,
    board_pt: torch.Tensor,
    viz_corner_points: bool = True,
    viz_all_points: bool = False,
):
    def _get_color(lab):
        if lab == -1:
            return "black"
        elif lab == 0:
            return "green"
        else:
            assert lab == 1
            return "white"

    (_, height, width) = image.shape
    image = image.transpose(0, 1).transpose(1, 2)  # Convert CHW to HWC
    image = image.clamp(0.0, 1.0)
    axis.imshow(image)

    # Corner points
    if viz_corner_points:
        axis.scatter(
            (board_pt[:, 0] * width).int(),
            (board_pt[:, 1] * height).int(),
            # facecolors="none",
            # edgecolors=["red", "red", "green", "green"],
            c="red",
            marker="s",
            s=200,
        )

    if viz_all_points:
        xs = torch.linspace(0.0, 1.0, steps=label.shape[0])
        ys = torch.linspace(0.0, 1.0, steps=label.shape[1])
        meshgrid = torch.meshgrid(xs, ys, indexing="xy")
        grid_pt = torch.stack(meshgrid, dim=2).reshape(-1, 2)
        image_points = _convert_points(grid_pt, board_pt)

        colors = [_get_color(l) for l in label.reshape(-1)]

        axis.scatter(
            (image_points[:, 0] * width).int(),
            (image_points[:, 1] * height).int(),
            # facecolors="none",
            # edgecolors=["red", "red", "green", "green"],
            c=colors,
            marker="s",
            s=100,
        )

    axis.axis("off")
    axis.set_aspect("auto")


def visualize_single_datapoint(data_points: DataPoints, output_path: str, index: int):
    num_images = data_points.images.shape[0]
    assert index < num_images

    fig, axis = plt.subplots(figsize=(25, 25))
    _visualize_single_helper(
        axis,
        data_points.images[index],
        data_points.labels[index],
        data_points.board_pts[index],
    )

    # plt.show()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def visualize_datapoints(
    data_point: DataPoints,
    output_path: str,
    max_viz_images: int | None = None,
):
    def _get_layout(num_images: int):
        known_layouts = [
            (1, (1, 1)),
            (2, (2, 1)),
            (4, (2, 2)),
            (6, (3, 2)),
            (8, (2, 4)),
            (9, (3, 3)),
            (11, (3, 4)),
            (12, (3, 4)),
            (16, (4, 4)),
        ]

        for idx, layout in known_layouts:
            if num_images <= idx:
                return layout

        # Unknown layout. Return something
        return (num_images, 1)

    (num_images, _, height, width) = data_point.images.shape
    if max_viz_images:
        num_images = min(num_images, max_viz_images)

    # gridspec_kw={"wspace": 0, "hspace": 0}
    layout = _get_layout(num_images)
    fig, axes = plt.subplots(
        layout[0], layout[1], gridspec_kw={"wspace": 0, "hspace": 0}, figsize=(25, 25)
    )
    if isinstance(axes, np.ndarray):
        if axes.ndim == 1:
            axes = list(axes)
        else:
            axes = list(itertools.chain(*axes))
    else:
        axes = [axes]

    for i in range(num_images):
        _visualize_single_helper(
            axes[i], data_point.images[i], data_point.labels[i], data_point.board_pts[i]
        )

    for i in range(num_images, len(axes)):
        axes[i].axis("off")
        axes[i].set_aspect("auto")

    # plt.show()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


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


def custom_collate_fn(batches) -> DataPoints:
    images = []
    labels = []
    board_pts = []

    for batch in batches:
        images.append(batch.image)
        labels.append(batch.label)
        board_pts.append(batch.board_pt)

        assert images[0].shape == batch[0].shape
        assert labels[0].shape == batch[1].shape
        assert board_pts[0].shape == batch[2].shape

    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    board_pts = torch.stack(board_pts, dim=0)
    return DataPoints(images, labels, board_pts)


def _read_image(data_io: AssetIO, image_path: str, board_metadata):
    """
    Returns tuple:
        - resized image
        - original (width x height) before resizing
    """
    image_tensor = data_io.load_image(image_path)
    new_size = (1024, 1024)  # Specify the new size (height, width)
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


def _load_single(data_point: DataPointPath, data_io: AssetIO) -> DataPoint:
    board_metadata = data_io.load_yaml(data_point.board_path)

    label = _read_label(data_io, data_point.label_path)
    image, original_size = _read_image(data_io, data_point.image_path, board_metadata)
    image = image[:3, :, :]  # Remove the alpha channel
    # board_pt is a list of 4 points. The first point is the top left corner, and then the points are in clockwise order
    board_pt = torch.tensor(board_metadata.pts_clicks) / original_size

    return DataPoint(image, label, board_pt)


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


class GoDataset(Dataset):
    def __init__(
        self,
        datapoint_paths: List[DataPointPath],
        base_path: str,
    ):
        self.datapoint_paths = datapoint_paths
        self.images, self.labels, self.board_pts = _load(
            datapoint_paths, AssetIO(base_path)
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> DataPoint:
        return DataPoint(self.images[idx], self.labels[idx], self.board_pts[idx])


class GoDynamicDataset(Dataset):
    """
    We cannot load the entire dataset into memory so load it dynamically
    """

    def __init__(
        self,
        datapoint_paths: List[DataPointPath],
        base_path: str,
    ):
        self.datapoint_paths = datapoint_paths
        self.data_io = AssetIO(base_path)

    def __len__(self):
        return len(self.datapoint_paths)

    def __getitem__(self, idx) -> DataPoint:
        data_point = _load_single(self.datapoint_paths[idx], self.data_io)
        return data_point


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
    return train_dataloader, test_dataloader


def create_datasets(cfg: DataCfg):
    data_io = AssetIO(cfg.base_path)
    directories = sorted(data_io.ls())

    entire_data: List[List[DataPointPath]] = []
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

    return load_datasets(cfg, train, test)
