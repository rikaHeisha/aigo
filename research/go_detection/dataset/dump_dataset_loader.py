import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from os import path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvf
from go_detection.common.asset_io import AssetIO

logger = logging.getLogger(__name__)


@dataclass
class RawDatasetPaths:
    image_path: str
    label_path: str
    board_path: str


@dataclass
class RawDatasetPoint:
    image: torch.Tensor
    label: torch.Tensor
    board_pt: torch.Tensor


class RawDatasetLabel(IntEnum):
    BLACK = 0
    EMPTY = 1
    WHITE = 2

    @staticmethod
    def from_str(ch) -> "RawDatasetLabel":
        if ch in {"B", "b"}:
            return RawDatasetLabel.BLACK
        elif ch in {" ", "."}:
            return RawDatasetLabel.EMPTY
        elif ch in {"W", "w"}:
            return RawDatasetLabel.WHITE
        else:
            assert False, f"Unknown character: {ch}"

    def to_str(self) -> str:
        if self == RawDatasetLabel.BLACK:
            return "B"
        elif self == RawDatasetLabel.EMPTY:
            return "."
        elif self == RawDatasetLabel.WHITE:
            return "W"
        else:
            assert False, f"Unknown enum: {self}"


def _transform_points_resize(
    points: torch.Tensor,
    image_size: torch.Tensor,  # width x height
    new_image_size: torch.Tensor,  # width x height
):
    assert (
        (points[:, 0] >= 0.0)
        & (points[:, 0] <= image_size[0])
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= image_size[1])
    ).all(), "Points should be inside the image"

    factor = new_image_size / image_size

    return points * factor


def _transform_points_crop(
    points: torch.Tensor,
    image_size: torch.Tensor,  # width x height
    crop_center: torch.Tensor,
    half_length: torch.Tensor,
):
    (width, height) = image_size
    # Check that all corners of the crop region are within the image
    crop_corner_points = torch.stack(
        [
            crop_center + half_length * torch.tensor([-1, -1]),
            crop_center + half_length * torch.tensor([1, -1]),
            crop_center + half_length * torch.tensor([1, 1]),
            crop_center + half_length * torch.tensor([-1, 1]),
        ]
    )
    assert (
        (crop_corner_points[:, 0] >= 0.0)
        & (crop_corner_points[:, 0] <= width)
        & (crop_corner_points[:, 1] >= 0.0)
        & (crop_corner_points[:, 1] <= height)
    ).all(), "Crop corners exceed past the image which is not good"

    top_left = crop_corner_points[0]
    new_points = points - top_left
    return new_points


def _read_raw_image(data_io: AssetIO, image_path: str, board_pts: torch.Tensor):
    """
    Returns tuple:
        - resized image
        - board_pts after the transformation
    """
    orig_image = data_io.load_image(image_path)
    _, height, width = orig_image.shape
    new_size = (1024, 1024)  # Specify the new size (height, width)

    if width == height:
        resized_tensor = tvf.resize(orig_image, new_size)
        new_board_pts = _transform_points_resize(
            board_pts, torch.tensor([width, height]), torch.tensor(new_size)
        )
        return resized_tensor, new_board_pts

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
    assert all(
        [
            ((board_pts - (center_square - rectangle_half_length)) >= 0.0).all(),
            ((board_pts - (center_square + rectangle_half_length)) <= 0.0).all(),
        ]
    ), "Expected all the board points to be inside the rectangular crop region"

    intermediate_image = tvf.crop(
        orig_image,
        int(center_square[1] - rectangle_half_length[1]),
        int(center_square[0] - rectangle_half_length[0]),
        int(2 * rectangle_half_length[1]),
        int(2 * rectangle_half_length[0]),
    )
    new_board_pts = _transform_points_crop(
        board_pts, torch.tensor([width, height]), center_square, rectangle_half_length
    )

    resized_image = tvf.resize(intermediate_image, new_size)
    new_board_pts = _transform_points_resize(
        new_board_pts, rectangle_half_length * 2, torch.tensor(new_size)
    )

    # asset_io = AssetIO("/home/rmenon/Desktop/dev/projects/aigo/research")
    # asset_io.save_image("rishi_orig.png", orig_image)
    # asset_io.save_image("rishi_intermediate.png", intermediate_image)
    # asset_io.save_image("rishi_final.png", resized_image)

    return resized_image, new_board_pts


def _read_label(data_io: AssetIO, label_path: str):
    """
    Returns
        A tensor of Boardsize x Boardsize
    """

    with open(data_io.get_abs(label_path), "r") as file:
        lines = file.read()
        label = []
        for line in lines.split("\n"):
            if line == "":
                continue

            label_line = []
            for ch in line.split(" "):
                digit = RawDatasetLabel.from_str(ch.upper())
                # assert (
                #     False
                # ), f"Unknown character: '{ch}' in line '{line}', file: {label_path}"

                label_line.append(digit)
            label.append(label_line)

        label = torch.tensor(label)
        return label


def load_dataset_path(data_point: RawDatasetPaths, data_io: AssetIO) -> RawDatasetPoint:
    # board_pts is a list of 4 points. The first point is the top left corner, and then the points are in clockwise order
    board_pts = torch.tensor(
        data_io.load_yaml(data_point.board_path)["pts_clicks"]
    ).float()

    label = _read_label(data_io, data_point.label_path)
    image, board_pts = _read_raw_image(data_io, data_point.image_path, board_pts)
    image = image[:3, :, :]  # Remove the alpha channel
    return RawDatasetPoint(image, label, board_pts)


def get_all_datapoints(base_path: str) -> List[List[RawDatasetPaths]]:
    data_io = AssetIO(base_path)
    directories = sorted(data_io.ls())

    entire_data: List[List[RawDatasetPaths]] = []
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

        directory_data: List[RawDatasetPaths] = []
        for val in map_name_to_dict.values():
            if "label" in val and "image" in val:
                data = RawDatasetPaths(
                    val["image"],
                    val["label"],
                    board_file,
                )
                directory_data.append(data)

        if directory_data:
            entire_data.append(directory_data)

    return entire_data
