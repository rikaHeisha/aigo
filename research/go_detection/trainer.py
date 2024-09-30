import torch
from torch import nn

from go_detection.config import DataCfg, SimCfg


class GoTrainer:
    def __init__(self, cfg: SimCfg):
        super().__init__()
        self.cfg = cfg
