import numpy as np
import torch
from go_detection.config import ModelCfg
from torch import nn


class GoModel(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.tensor([2.0]))

    def forward(self, images):
        a = 1
