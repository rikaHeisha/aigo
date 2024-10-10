import logging
import os
import sys
from datetime import datetime
from os import path
from typing import List, Optional

import debugpy
import hydra
from common.asset_io import AssetIO
from config import SimCfg
from go_detection.common.git_utils import get_git_info
from go_detection.dataloader import create_datasets
from go_detection.trainer import GoTrainer
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def do_main(cfg: SimCfg):
    go_trainer = GoTrainer(cfg)
    go_trainer.start()


@hydra.main(config_path="config", config_name="basic", version_base="1.2")
def main(cfg: SimCfg):
    if os.environ.get("ENABLE_DEBUGPY"):
        print("")
        print("\033[31mWaiting for debugger to connect\033[0m")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    OmegaConf.set_readonly(cfg, True)

    cfg = instantiate(cfg)  # Converts the DictConfig to native python classes

    # Save config info after instantiating. instantiation causes interpolations to get resolved. So the config file generated will have
    # all interpolations resolved
    exp_io = AssetIO(path.join(cfg.result_cfg.dir, cfg.result_cfg.name))
    exp_io.mkdir(".")
    exp_io.mkdir("log")
    exp_io.save_yaml("config.yaml", cfg)

    # Save git infomation
    git_info = get_git_info()

    args = " ".join(sys.argv[1:])
    run_cmd = f"python go_detection/main.py {args}"
    export_cmd = f"python go_detection/export_script.py {args}"
    exp_io.save_yaml(
        "branch_info.yaml",
        {
            "branch_name": git_info.branch_name,
            "current_commit": git_info.current_commit,
            "repo_clean": "clean" if git_info.repo_clean else "modified",
            "run_command": run_cmd,
            "export_command": export_cmd,
        },
    )

    ########################
    # Logging
    ########################
    # Setup file logger manually (after the log folder is created)
    # Add cli arg hydra.job_logging.root.level=ERROR to set log level

    # str_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # rel_path = path.join("log", f"{str_now}.log")
    run_number = (
        len([_ for _ in exp_io.ls("log") if exp_io.has_file(_) and _.endswith(".log")])
        + 1
    )
    rel_path = path.join("log", f"log_{run_number}.log")
    assert exp_io.has(rel_path) == False
    log_file = exp_io.get_abs(rel_path)
    fh = logging.FileHandler(filename=log_file)
    fh.setFormatter(
        logging.Formatter(fmt="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    )
    logging.getLogger().addHandler(fh)

    ########################
    # Print config
    ########################
    cfg_yaml = OmegaConf.to_yaml(cfg)
    logger.info("Config:\n%s", cfg_yaml)

    logger.info("Hydra dir set to: %s", HydraConfig.get().run.dir)
    logger.info(
        f"Log Level: {HydraConfig.get().job_logging.root.level}, Log File: {log_file}"
    )

    do_main(cfg)


if __name__ == "__main__":
    # Register configs
    cs = ConfigStore.instance()
    cs.store(name="sim_cfg_default", node=SimCfg)

    main()
