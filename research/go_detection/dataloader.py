import random
from collections import defaultdict, namedtuple
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from go_detection.common.asset_io import AssetIO
from go_detection.config import DataCfg
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DataPoint = namedtuple("DataPoint", ["image_path", "label_path", "board_path"])


class GoDataset(Dataset):
    def __init__(
        self,
        entire_data: List[List[DataPoint]],
        base_path: str,
    ):
        data_io = AssetIO(base_path)
        self.labels, self.images, self.board_pts = self._load(entire_data, data_io)
        print(f"Length dataset: {len(self.labels)}")

    def _load(self, entire_data: List[List[DataPoint]], data_io: AssetIO):
        labels = []
        images = []
        board_pts = []

        for list_data_points in tqdm(entire_data, desc="Loading dataset"):
            # All the data points in list_data have the same board metadata
            assert len(set([_.board_path for _ in list_data_points])) == 1
            board_metadata = data_io.load_yaml(list_data_points[0].board_path)

            for data_point in list_data_points:
                label = self._read_label(data_io, data_point.label_path)
                image, original_size = self._read_image(
                    data_io, data_point.image_path, board_metadata
                )
                image = image[:3, :, :]  # Remove the alpha channel
                board_pt = torch.tensor(board_metadata.pts_clicks) / original_size

                labels.append(label)
                images.append(image)
                board_pts.append(board_pt)

        return labels, images, board_pts

    def _read_image(self, data_io: AssetIO, image_path: str, board_metadata):
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

    def _read_label(self, data_io: AssetIO, label_path: str):
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
                label_line = []
                for ch in line.split(" "):
                    if ch == ".":
                        digit = 0
                    elif ch.upper() == "W":
                        digit = 1
                    elif ch.upper() == "B":
                        digit = -1
                    else:
                        assert False, f"Unknown character: '{ch}' in line '{line}'"
                    label_line.append(digit)
                label.append(label_line)

            label = torch.tensor(label)
            return label

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


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

    train_dataset, test_dataset = GoDataset(train, cfg.base_path), GoDataset(
        test, cfg.base_path
    )

    return train_dataset, test_dataset
