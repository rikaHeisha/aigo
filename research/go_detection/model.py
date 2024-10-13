import numpy as np
import torch
from go_detection.config import ModelCfg
from torch import nn


class DownNet(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int = 3,
        padding: int = 0,
        stride: int = 1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_c, out_c, kernel_size=kernel_size, padding=padding, stride=stride
            ),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_c),
            # nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        x1 = self.net(x)
        return x1


class GoModel(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg

        cs = [3, 4, 8, 32, 64, 128, 128, 128]
        self.stage1 = DownNet(cs[0], cs[1], 3, 0)
        self.stage2 = DownNet(cs[1], cs[2], 3, 0)
        self.stage3 = DownNet(cs[2], cs[3], 3, 0)
        self.stage4 = DownNet(cs[3], cs[4], 3, 0)
        self.stage5 = DownNet(cs[4], cs[5], 3, 0)
        self.stage6 = DownNet(cs[5], cs[6], 3, 0)
        self.stage7 = DownNet(cs[6], cs[7], 3, 0)

        self.head = nn.Sequential(
            # nn.Linear(cs[-1] * 6 * 6, 3),
            nn.Conv2d(cs[-1], 19 * 19 * 3, kernel_size=6, padding=0, stride=1),
        )

        # self.head = nn.Sequential(
        #     nn.Conv2d(cs[4], 16, kernel_size=1, padding=0, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 3, kernel_size=1, padding=0, stride=1),
        # )

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, images):
        assert images.ndim == 4
        num_images = images.shape[0]

        x0 = images
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        x6 = self.stage6(x5)
        x7 = self.stage7(x6)

        # x8 = self.head(x7.reshape(x7.shape[0], -1))
        x8 = self.head(x7)
        x9 = x8.reshape(num_images, 3, 19 * 19)

        # x_out = torch.softmax(x8, dim=1)
        x_out = self.log_softmax(x9)

        # x6 = x5[:, :, :19, :19]
        # x7 = self.head(x6)
        # x8 = self.log_softmax(x7)
        # x_out = x8

        return x_out
