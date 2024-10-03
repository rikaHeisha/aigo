import numpy as np
import torch
from go_detection.config import ModelCfg
from torch import nn


class GoModel(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.tensor([2.0]))

        cs = [3, 6, 16, 32, 32]
        self.stage1 = nn.Sequential(
            nn.Conv2d(cs[0], cs[0], kernel_size=3, padding=0, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(cs[0], cs[1], kernel_size=3, padding=0, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(cs[1], cs[2], kernel_size=3, padding=0, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(cs[2], cs[3], kernel_size=3, padding=0, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )
        self.stage5 = nn.Sequential(
            nn.Conv2d(cs[3], cs[4], kernel_size=3, padding=0, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )
        self.stage6 = nn.Sequential(
            nn.Conv2d(cs[4], cs[4], kernel_size=3, padding=0, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )
        self.stage7 = nn.Sequential(
            nn.Conv2d(cs[4], cs[4], kernel_size=3, padding=0, stride=1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.stage8 = nn.Sequential(
            nn.Linear(cs[-1] * 6 * 6, 3),
        )

        # self.head = nn.Sequential(
        #     nn.Conv2d(cs[4], 16, kernel_size=1, padding=0, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 3, kernel_size=1, padding=0, stride=1),
        # )

        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, images):
        x0 = images
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)

        x6 = self.stage6(x5)
        x7 = self.stage7(x6)
        x8 = self.stage8(x7.reshape(x7.shape[0], -1))
        x_out = torch.softmax(x8, dim=1)

        # x6 = x5[:, :, :19, :19]
        # x7 = self.head(x6)
        # x8 = self.log_softmax(x7)
        # x_out = x8

        return x_out
