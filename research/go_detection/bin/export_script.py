import logging
import os
from os import path
from typing import cast

import debugpy
import hydra
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from go_detection.common.asset_io import AssetIO
from go_detection.config import SimCfg
from go_detection.dataloader import DataPoint, DataPoints
from go_detection.export_util import export_model
from go_detection.trainer import GoTrainer
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def do_main(cfg: SimCfg):
    result_io = AssetIO(path.join(cfg.result_cfg.dir, cfg.result_cfg.name))
    result_io.mkdir("export")

    go_trainer = GoTrainer(cfg)
    assert go_trainer.iter > 2, "Model should be trained before exporting"

    export_path = path.join("export", "model.tflite")
    datapoints = cast(
        DataPoint, go_trainer.train_dataloader.dataset[0]
    ).to_data_points()
    export_model(
        go_trainer.model, result_io.get_abs(export_path), datapoints.images.shape
    )


@hydra.main(config_path="config", config_name="basic", version_base="1.2")
def main(cfg):
    if os.environ.get("ENABLE_DEBUGPY"):
        print("")
        print("\033[31mWaiting for debugger to connect\033[0m")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    OmegaConf.set_readonly(cfg, True)
    cfg = instantiate(cfg)  # Converts the DictConfig to native python classes

    cfg_yaml = OmegaConf.to_yaml(cfg)
    logger.info("Config:\n%s", cfg_yaml)

    logger.info("Hydra dir set to: %s", HydraConfig.get().run.dir)
    logger.info(f"Log Level: {HydraConfig.get().job_logging.root.level}")

    do_main(cfg)


if __name__ == "__main__":
    # Register configs
    cs = ConfigStore.instance()
    cs.store(name="sim_cfg_default", node=SimCfg)

    main()
