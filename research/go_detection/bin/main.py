import cProfile
import logging
import os
import sys
import tempfile
import uuid
from datetime import datetime
from os import path
from typing import List, Optional

import debugpy
import hydra
from go_detection.common.asset_io import AssetIO
from go_detection.common.git_utils import get_git_info
from go_detection.config import SimCfg
from go_detection.dataloader import create_datasets
from go_detection.trainer import GoTrainer
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def _setup_logger(exp_io: AssetIO, log_base_path: str):
    # str_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # rel_path = path.join("log", f"{str_now}.log")
    exp_io.mkdir(log_base_path)

    existing_run_files = [
        int(_.removeprefix(f"{log_base_path}/log_").removesuffix(".log"))
        for _ in exp_io.ls(log_base_path)
        if exp_io.has_file(_) and _.endswith(".log")
    ]

    # run_number = len(existing_run_files) + 1 # Naive solution
    run_number = (max(existing_run_files) + 1) if existing_run_files != [] else 1

    log_rel_path = path.join(log_base_path, f"log_{run_number}.log")
    assert exp_io.has(log_rel_path) == False
    fh = logging.FileHandler(filename=exp_io.get_abs(log_rel_path))
    fh.setFormatter(
        logging.Formatter(fmt="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    )
    logging.getLogger().addHandler(fh)

    return exp_io.get_abs(log_rel_path)


def _save_git_info(exp_io: AssetIO, rel_path: str):
    git_info = get_git_info()

    args = " ".join(sys.argv[1:])
    run_cmd = f"python go_detection/main.py {args}"
    export_cmd = f"python go_detection/export_script.py {args}"
    exp_io.save_yaml(
        rel_path,
        {
            "branch_name": git_info.branch_name,
            "current_commit": git_info.current_commit,
            "repo_clean": "clean" if git_info.repo_clean else "modified",
            "run_command": run_cmd,
            "export_command": export_cmd,
        },
    )


@hydra.main(config_path="../config", config_name="basic", version_base="1.2")
def main(cfg: SimCfg):
    OmegaConf.set_readonly(cfg, True)

    cfg = instantiate(cfg)  # Converts the DictConfig to native python classes
    exp_io = AssetIO(path.join(cfg.result_cfg.dir, cfg.result_cfg.name))
    exp_io.mkdir(".")

    # Save config info after instantiating. instantiation causes interpolations to get resolved. So the config file generated will have all interpolations resolved
    exp_io.save_yaml("config.yaml", cfg)

    # Save git infomation
    _save_git_info(exp_io, "branch_info.yaml")

    # Logging
    log_file = _setup_logger(exp_io, "exp_info/log")

    # Print config
    cfg_yaml = OmegaConf.to_yaml(cfg)
    logger.info("Config:\n%s", cfg_yaml)

    logger.info("Hydra dir set to: %s", HydraConfig.get().run.dir)
    logger.info(
        f"Log Level: {HydraConfig.get().job_logging.root.level}, Log File: {log_file}"
    )

    # Profile / run
    if cfg.profile:
        # file_path = f"/tmp/go_{str(uuid.uuid4())}.hprof"
        file_path = datetime.now().strftime(
            "/tmp/aigo/profiling_go_%Y-%m-%d_%H-%M-%S.hprof"
        )
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # with tempfile.NamedTemporaryFile(
        #     prefix="go_", suffix=".hprof", delete=False
        # ) as file:

        logger.info(f"Dumping stats to {file_path}")
        with cProfile.Profile() as pr:
            do_main(cfg)
            pr.dump_stats(file=file_path)

        logger.info(f"Dumping stats to {file_path}")
    else:
        do_main(cfg)


def do_main(cfg: SimCfg):
    go_trainer = GoTrainer(cfg)
    go_trainer.start()


if __name__ == "__main__":
    # Register configs
    cs = ConfigStore.instance()
    cs.store(name="sim_cfg_default", node=SimCfg)

    if os.environ.get("ENABLE_DEBUGPY"):
        print("")
        print("\033[31mWaiting for debugger to connect\033[0m")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    main()
